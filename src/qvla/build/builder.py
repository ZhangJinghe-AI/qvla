"""Offline QVLA pack builder.

End-to-end pipeline:

1. Adapter builds the full-precision model (eager mode).
2. We list every target Linear with :func:`wrap.list_target_modules`.
3. Optional Fisher pass (``perm_score=fisher``).
4. When adaptive act_clip is combined with a non-empty rotation pipeline,
   fit per-channel ``act_clip`` in **original** space (no Hessian).
5. **Pipeline-wise rotation fit** — for each scope pipeline step that needs
   activation stats, run a calibration pass on ``clip(x)`` then
   ``prefix_transform.apply(x)`` (clip omitted when unused), then fit that
   step; weight-only steps fit immediately from ``W``.
6. **Final calibration pass** on ``clip(x)`` then full ``rotation.apply(x)``
   for GPTQ / act-scale tables.
7. Quantize rotated weights and serialize the :class:`Pack`.
"""

from __future__ import annotations

import gc
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from qvla.adapters import ModelAdapter
from qvla.build.collector import (
    AmaxCollectPlan,
    LayerStats,
    RotatedActivationCollector,
    assert_dit_clip_has_per_step_table,
    assert_smooth_pmean_step_coverage,
)
from qvla.build.utils import (
    choose_collector_device,
    install_crash_handler,
    log_memory,
)
from qvla.config import QVLAConfig, ScopeConfig
from qvla.core.pack import LayerPack, Pack
from qvla.core.quantize import (
    QuantizedWeight,
    FP4_E2M1_MAX,
    is_no_quant,
    percentile_amax,
    quantize_weight,
    symmetric_quant_range,
)
from qvla.core.pipeline import (
    PipelineBuild,
    Transform,
    apply_input_pipeline,
    identity_transform,
    step_needs_activation_calibration,
)
from qvla.runtime.wrap import list_target_modules


logger = logging.getLogger(__name__)


def _scope_cfg(config: QVLAConfig, scope: str) -> ScopeConfig:
    return config.llm if scope == "llm" else config.dit


def _fisher_targets_for_gptq(
    target_modules: list[tuple[str, str, nn.Module]],
    config: QVLAConfig,
) -> list[tuple[str, str, nn.Module]]:
    """Layers whose scope enables ``fisher_gptq``."""
    out: list[tuple[str, str, nn.Module]] = []
    for name, scope, mod in target_modules:
        if _scope_cfg(config, scope).fisher_gptq:
            out.append((name, scope, mod))
    return out


def _needs_offline_act_scale(scope_cfg: ScopeConfig) -> bool:
    """True when the runtime uses a baked ``act_scale_table`` (not dynamic)."""
    return scope_cfg.act_scale_mode in ("static", "per_step")



def _amax_collect_plan(scope_cfg: ScopeConfig) -> AmaxCollectPlan:
    """Select which amax streams the final calibration must keep for this scope.

    The final calibration pass collects Hessian and act-scale statistics in
    the fully-transformed space.  Clip and smooth stats are NOT needed here
    because they are always fitted during the pipeline-build phase via
    ``PipelineBuild.fit_step``.
    """
    if _needs_offline_act_scale(scope_cfg):
        p = float(scope_cfg.act_percentile)
        if scope_cfg.act_percentile_mode == "inner":
            if scope_cfg.act_scale_granularity == "per_token":
                return AmaxCollectPlan(collect_inner_token=True, inner_percentile=p)
            return AmaxCollectPlan(collect_inner_channel=True, inner_percentile=p)
        if scope_cfg.act_scale_granularity == "per_token":
            return AmaxCollectPlan(collect_cross_token=True)
        return AmaxCollectPlan(collect_cross_channel=True)
    return AmaxCollectPlan()




def _act_scale_amax(
    stats: LayerStats, scope_cfg: ScopeConfig, *, step: int | None = None
) -> torch.Tensor:
    """Resolve offline act amax; ``step`` is forwarded into LayerStats accessors."""
    p = scope_cfg.act_percentile
    if scope_cfg.act_scale_granularity == "per_token":
        if scope_cfg.act_percentile_mode == "cross":
            return percentile_amax(stats.act_token_cross_amax(step=step), p)
        return stats.act_token_inner_amax(step=step)
    if scope_cfg.act_percentile_mode == "cross":
        return percentile_amax(stats.act_channel_cross_amax(step=step), p)
    return stats.act_channel_inner_amax(step=step)


def _build_act_scale_table(
    stats: LayerStats, scope_cfg: ScopeConfig
) -> torch.Tensor | None:
    """Convert collector stats into the runtime layer's ``act_scale_table``."""
    if scope_cfg.act_scale_mode == "dynamic":
        return None
    if is_no_quant(scope_cfg.act_bits):
        return None
    _, act_qmax = symmetric_quant_range(scope_cfg.act_bits)
    grid_max = FP4_E2M1_MAX if scope_cfg.act_format == "fp" else float(act_qmax)
    if scope_cfg.act_scale_mode == "static":
        amax = _act_scale_amax(stats, scope_cfg)
        return (amax.clamp_min(1e-12) / grid_max).to(torch.float32)
    # per_step: step count lives on whichever stream the plan collected.
    if scope_cfg.act_scale_granularity == "per_token":
        step_buf = (
            stats.per_step_cross_token_amax
            if scope_cfg.act_percentile_mode == "cross"
            else stats.per_step_inner_token_amax
        )
    else:
        step_buf = (
            stats.per_step_cross_channel_amax
            if scope_cfg.act_percentile_mode == "cross"
            else stats.per_step_inner_channel_amax
        )
    rows = [
        _act_scale_amax(stats, scope_cfg, step=s) for s in range(step_buf.shape[0])
    ]
    table = torch.stack(rows, dim=0).clamp_min(1e-12) / grid_max
    return table.to(torch.float32)


def _rotate_weight(w: torch.Tensor, rotation: Transform) -> torch.Tensor:
    """Apply the weight-side pipeline transform.

    For orthogonal steps the transform is identical to the activation side.
    Smooth is inverted (``W * s`` instead of ``x / s``) because
    ``y = (x/s) @ (W*s)^T = x @ W^T``.  Clip is skipped (activation-only).
    """
    if rotation.is_identity:
        return w.contiguous()
    out = w.to(torch.float32)
    for step in rotation.pipeline:
        if step == "clip":
            continue
        elif step == "smooth":
            if rotation.smooth_scale is not None:
                s = rotation.smooth_scale.to(device=out.device, dtype=out.dtype)
                out = out * s.unsqueeze(0)
        else:
            out = apply_input_pipeline(
                out,
                pipeline=(step,),
                u_blocks=rotation.u_blocks,
                perm=rotation.perm,
                block_size=rotation.block_size,
                d=rotation.d,
                random_hadamard_blocks=rotation.random_hadamard_blocks,
            )
    return out.contiguous()


def _effective_noise_ensemble_k(
    targets: list[tuple[str, str, nn.Module]],
    config: QVLAConfig,
) -> int:
    """K>1 only when DiT layers are being calibrated.

    Diffusion noise mainly affects the DiT action head; LLM prefix activations
    are essentially noise-invariant, so ensemble loops there waste compute.
    """
    k = config.noise_ensemble_k
    if k < 1:
        raise ValueError(f"noise_ensemble_k must be >= 1, got {k}.")
    if any(scope == "dit" for _, scope, _ in targets):
        return k
    return 1


def _compute_dit_step_weights(
    method: str, num_steps: int
) -> dict[int, float] | None:
    """Per-step XᵀX weights for DiT calibration aggregation.

    Returns ``None`` for uniform (current behaviour, all weights = 1.0).
    """
    if method == "uniform" or num_steps <= 1:
        return None
    if method == "late_mean":
        start = num_steps // 2
        return {s: (1.0 if s >= start else 0.0) for s in range(num_steps)}
    if method == "very_late_mean":
        start = num_steps - max(1, num_steps // 5)
        return {s: (1.0 if s >= start else 0.0) for s in range(num_steps)}
    if method == "weighted_linear":
        return {s: float(s + 1) for s in range(num_steps)}
    if method == "max":
        raise ValueError(
            "calibration_step_aggregation='max' is not supported for XᵀX; "
            "use 'uniform', 'late_mean', 'very_late_mean', or 'weighted_linear'."
        )
    raise ValueError(f"Unknown calibration_step_aggregation: {method!r}")


def _run_transformed_calibration(
    adapter: ModelAdapter,
    model: nn.Module,
    targets: list[tuple[str, str, nn.Module]],
    transforms: dict[str, Transform],
    *,
    config: QVLAConfig,
    num_samples: int,
    log_label: str,
    progress: Callable[[str, float], None] | None,
    progress_base: float,
    progress_span: float,
    amax_plan_by_scope: dict[str, AmaxCollectPlan],
    adaptive_token_scope: str | None = None,
) -> dict[str, LayerStats]:
    """Forward ``num_samples × K``; collect on ``transform.apply(x)``.

    ``K = noise_ensemble_k`` when ``targets`` include DiT layers, else ``K = 1``.
    LLM layer hooks are skipped for ``noise_index > 0`` because LLM activations
    are diffusion-noise-invariant; collecting once is enough.

    When ``adaptive_token_scope`` is set, ``adapter.calibration_outlier_token_keep_mask``
    is called per batch and fed to the collector for adaptive tip/rest splitting.
    """
    if not amax_plan_by_scope:
        raise ValueError("amax_plan_by_scope must be a non-empty dict.")
    k = _effective_noise_ensemble_k(targets, config)
    num_steps = {"llm": 1, "dit": config.dit.num_steps}
    dit_step_weights = _compute_dit_step_weights(
        config.calibration_step_aggregation, config.dit.num_steps
    ) or {}
    total = num_samples * k
    plan = next(iter(amax_plan_by_scope.values()))
    model_device = getattr(targets[0][2], "weight").device
    chunk_size = int(adapter.calibration_noise_spec().chunk_size)
    collector_device = choose_collector_device(
        targets, model_device,
        amax_plan=plan,
        num_samples=total,
        model_kind=config.model_kind,
        chunk_size=chunk_size,
        num_steps_by_scope=num_steps,
    )
    with RotatedActivationCollector(
        targets,
        transforms,
        num_steps_by_scope=num_steps,
        device=collector_device,
        amax_plan_by_scope=amax_plan_by_scope,
        noise_ensemble_k=k,
        dit_step_weights=dit_step_weights,
    ) as collector:
        def step_cb(step: int) -> None:
            collector.set_current_step(step)

        log_memory(f"{log_label}: before calibration loop")
        with torch.inference_mode():
            done = 0
            for i, batch in enumerate(adapter.iter_calibration_batches(num_samples)):
                if (
                    adaptive_token_scope is not None
                    and adaptive_token_scope not in ("all", "skip_first")
                ):
                    mask = adapter.calibration_outlier_token_keep_mask(
                        batch, token_scope=adaptive_token_scope,
                    )
                    collector.set_adaptive_token_keep_mask(mask)
                for noise_index in range(k):
                    collector.set_noise_index(noise_index)
                    logger.info(
                        "%s sample %d/%d noise %d/%d ...",
                        log_label,
                        i + 1,
                        num_samples,
                        noise_index + 1,
                        k,
                    )
                    adapter.forward_for_calibration(
                        model,
                        batch,
                        step_callback=step_cb,
                        sample_index=i,
                        noise_index=noise_index,
                    )
                    done += 1
                    if progress:
                        frac = progress_base + progress_span * done / max(1, total)
                        progress("calibrate", frac)
                gc.collect()
                torch.cuda.empty_cache()
                if (i + 1) % 3 == 0 or i == num_samples - 1:
                    log_memory(f"{log_label}: after sample {i + 1}/{num_samples}")
    return {name: collector.stats[name] for name, _s, _m in targets}



def _build_rotations_pipeline_wise(
    adapter: ModelAdapter,
    model: nn.Module,
    target_modules: list[tuple[str, str, nn.Module]],
    config: QVLAConfig,
    num_samples: int,
    fisher_sensitivities: dict[str, torch.Tensor],
    progress: Callable[[str, float], None] | None,
) -> dict[str, Transform]:
    """Fit each layer transform by walking the scope pipeline step-by-step."""
    by_scope: dict[str, list[tuple[str, str, nn.Module]]] = defaultdict(list)
    for item in target_modules:
        by_scope[item[1]].append(item)

    rotations: dict[str, Transform] = {}
    step_idx = 0

    for scope, scope_targets in by_scope.items():
        scope_cfg = _scope_cfg(config, scope)
        pipeline = scope_cfg.pipeline
        if not pipeline:
            for name, _s, mod in scope_targets:
                in_f = int(getattr(mod, "in_features"))
                rotations[name] = identity_transform(
                    in_f, scope_cfg.rotation_block_size
                )
            continue

        builders: dict[str, PipelineBuild] = {}
        for name, _s, mod in scope_targets:
            w = getattr(mod, "weight").detach()
            in_f = int(w.shape[1])
            builders[name] = PipelineBuild(
                d=in_f,
                block_size=scope_cfg.rotation_block_size,
                weight=w,
                pipeline=pipeline,
                perm_score=scope_cfg.perm_score,
                svd_source=scope_cfg.svd_source,
                sensitivity=fisher_sensitivities.get(name),
                layer_name=name,
                build_seed=config.build_seed,
                clip_epsilon=float(scope_cfg.smooth_epsilon),
                clip_kappa=float(scope_cfg.act_outlier_kappa),
                clip_bulk_percentile=float(scope_cfg.act_outlier_bulk_percentile),
                clip_std_k=float(scope_cfg.act_outlier_std_k),
                clip_std_k_down=float(scope_cfg.act_outlier_std_k_down),
                clip_std_k_up=float(scope_cfg.act_outlier_std_k_up),
                clip_selective_channels=bool(
                    scope_cfg.act_outlier_selective_channels
                ),
                clip_global=bool(scope_cfg.act_outlier_global),
                clip_skip_first_token=(
                    scope_cfg.act_outlier_fit_tokens == "skip_first"
                ),
                smooth_alpha=float(scope_cfg.smooth_alpha),
                smooth_epsilon=float(scope_cfg.smooth_epsilon),
                smooth_act_percentile=float(scope_cfg.smooth_act_percentile),
                smooth_fisher=fisher_sensitivities.get(name),
                smooth_fisher_beta=float(scope_cfg.smooth_fisher_beta),
                smooth_step_pmean_p=scope_cfg.smooth_step_pmean_p,
            )

        for step_index, step in enumerate(pipeline):
            if step_needs_activation_calibration(
                step,
                perm_score=scope_cfg.perm_score,
                svd_source=scope_cfg.svd_source,
            ):
                first_builder = next(iter(builders.values()))
                amax_plan = first_builder.step_amax_plan(step_index)
                prefix_transforms: dict[str, Transform] = {}
                for name, _, _ in scope_targets:
                    prefix_transforms[name] = builders[name].prefix_transform(step_index)
                logger.info(
                    "Pipeline calibration (%s step %d/%d: %s on prefix %s) ...",
                    scope,
                    step_index + 1,
                    len(pipeline),
                    step,
                    "+".join(pipeline[:step_index]) or "none",
                )
                token_scope = None
                if step == "clip" and amax_plan.collect_adaptive_inner_channel:
                    token_scope = scope_cfg.act_outlier_fit_tokens
                stats = _run_transformed_calibration(
                    adapter,
                    model,
                    scope_targets,
                    prefix_transforms,
                    config=config,
                    num_samples=num_samples,
                    log_label=f"{scope} pipeline-{step}",
                    progress=progress,
                    progress_base=0.35 + 0.20 * step_idx / max(1, len(pipeline)),
                    progress_span=0.20 / max(1, len(pipeline)),
                    amax_plan_by_scope={scope: amax_plan},
                    adaptive_token_scope=token_scope,
                )
                if step == "smooth" and scope_cfg.smooth_step_pmean_p is not None:
                    logger.info(
                        "%s Smooth a_j: p-mean over denoise steps (p=%s).",
                        scope,
                        scope_cfg.smooth_step_pmean_p,
                    )
                    assert_smooth_pmean_step_coverage(stats)
                step_idx += 1
                for name, _, _ in scope_targets:
                    builders[name].fit_step(step_index, stats=stats[name])
                if step == "clip" and scope == "dit":
                    clips = {
                        name: builders[name].act_clip
                        for name, _, _ in scope_targets
                    }
                    none_names = [n for n, c in clips.items() if c is None]
                    if none_names:
                        raise RuntimeError(
                            "DiT clip fitted no act_clip for: "
                            + ", ".join(none_names)
                        )
                    assert_dit_clip_has_per_step_table(clips)
                del stats
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                for name, _, _ in scope_targets:
                    builders[name].fit_step(step_index, stats=None)

        for name, _, _ in scope_targets:
            rotations[name] = builders[name].finish()

    return rotations



def _make_layer_pack(
    name: str,
    scope: str,
    module: torch.nn.Module,
    quant_stats: LayerStats,
    scope_cfg: ScopeConfig,
    rotation: Transform,
    fisher_rotated: torch.Tensor | None = None,
) -> LayerPack:
    """Quantize one layer.

    ``act_clip`` and ``smooth_scale`` already live on ``rotation``
    (fitted during the pipeline-build phase).  The Hessian and act scales
    from the final calibration pass are already in the fully-transformed
    space, so no additional smooth adjustment is needed here.
    """
    w = getattr(module, "weight").detach()
    bias_attr = getattr(module, "bias", None)
    bias = bias_attr.detach().to(torch.float32) if bias_attr is not None else None
    bias_present = bias is not None

    N, K = w.shape
    in_features = K
    out_features = N
    device = w.device

    W_fp = w.to(torch.float32)
    W_rot = _rotate_weight(W_fp, rotation)
    H_rot = quant_stats.hessian().to(device)

    if scope_cfg.fisher_gptq:
        if scope_cfg.weight_quantizer != "gptq":
            raise ValueError(
                f"fisher_gptq=True requires weight_quantizer='gptq', "
                f"got {scope_cfg.weight_quantizer!r} for layer {name!r}."
            )
        if fisher_rotated is None:
            raise ValueError(
                f"fisher_gptq=True but no rotated-space Fisher sensitivity "
                f"available for layer {name!r}."
            )
        from qvla.build.fisher import normalize_fisher_sensitivity

        F_norm = normalize_fisher_sensitivity(fisher_rotated.to(device))
        d_sqrt = F_norm.sqrt()
        H_rot = H_rot * d_sqrt.unsqueeze(1) * d_sqrt.unsqueeze(0)
        logger.info(
            "Fisher-GPTQ %s: F_norm max/min ratio = %.1f "
            "(max=%.2e min=%.2e; mean=1.0; %d rotation blocks of %d).",
            name,
            F_norm.max().item() / max(F_norm.min().item(), 1e-12),
            F_norm.max().item(),
            F_norm.min().item(),
            K // rotation.block_size,
            rotation.block_size,
        )

    qw: QuantizedWeight = quantize_weight(
        W_rot,
        scope_cfg.weight_quantizer,
        group_size=scope_cfg.group_size,
        weight_bits=scope_cfg.weight_bits,
        weight_format=scope_cfg.weight_format,
        hessian=H_rot,
        gptq_block_size=scope_cfg.gptq_block_size,
        gptq_damp_percent=scope_cfg.gptq_damp_percent,
    )

    act_scale_table: torch.Tensor | None = None
    if _needs_offline_act_scale(scope_cfg):
        act_scale_table = _build_act_scale_table(quant_stats, scope_cfg)

    extras: dict = {}
    if is_no_quant(qw.weight_bits) and qw.fp_weight is not None:
        extras["fp_weight"] = qw.fp_weight.to(torch.bfloat16)

    return LayerPack(
        name=name,
        scope=scope,
        in_features=in_features,
        out_features=out_features,
        bias_present=bias_present,
        qweight=qw.qweight,
        weight_scale=qw.weight_scale,
        group_size=qw.group_size,
        weight_bits=qw.weight_bits,
        weight_format=qw.weight_format,
        weight_scale_2=qw.weight_scale_2,
        rotation=rotation,
        act_bits=scope_cfg.act_bits,
        act_format=scope_cfg.act_format,
        act_scale_mode=scope_cfg.act_scale_mode,
        act_scale_granularity=scope_cfg.act_scale_granularity,
        act_scale_table=act_scale_table,
        bias=bias,
        residual=qw.residual,
        extras=extras,
    )


def _validate_fisher_batch_config(config: QVLAConfig) -> None:
    if config.fisher_num_samples < 1:
        raise ValueError(
            f"fisher_num_samples must be >= 1 when Fisher is enabled, "
            f"got {config.fisher_num_samples}."
        )
    if config.fisher_batch_size < 1:
        raise ValueError(
            f"fisher_batch_size must be >= 1, got {config.fisher_batch_size}."
        )
    if config.fisher_batch_size > config.fisher_num_samples:
        raise ValueError(
            f"fisher_batch_size={config.fisher_batch_size} cannot exceed "
            f"fisher_num_samples={config.fisher_num_samples}."
        )


def _configure_adapter_for_fisher(adapter: ModelAdapter, config: QVLAConfig) -> None:
    """Raise engine batch capacity before ``build_model`` when Fisher runs."""
    needs = (
        config.needs_fisher or config.llm.fisher_gptq or config.dit.fisher_gptq
    )
    if not needs:
        return
    _validate_fisher_batch_config(config)
    required = int(config.fisher_batch_size)
    cfg = getattr(adapter, "cfg", None)
    if cfg is None or not hasattr(cfg, "max_batch_size"):
        raise RuntimeError(
            f"fisher_batch_size={config.fisher_batch_size} requires an adapter "
            "with cfg.max_batch_size; got "
            f"{type(adapter).__name__}."
        )
    if adapter.engine is not None:
        engine_bs = int(adapter.engine.entry.scheduler.max_batch_size)
        raise RuntimeError(
            "Adapter engine is already built; this build path expects an "
            "uninitialized adapter so fisher batch sizing can be configured "
            "before build_model(). "
            f"current max_batch_size={engine_bs}, required={required}."
        )
    prev = int(getattr(cfg, "max_batch_size", 1))
    cfg.max_batch_size = max(prev, required)
    if cfg.max_batch_size != prev:
        logger.info(
            "Raised adapter max_batch_size %d → %d for Fisher batching.",
            prev,
            cfg.max_batch_size,
        )


def build_pack(
    adapter: ModelAdapter,
    *,
    output_path: str | Path,
    config: QVLAConfig,
    num_samples: int = 10,
    progress: Callable[[str, float], None] | None = None,
) -> Pack:
    """Run the full offline calibration + quantization pipeline."""
    install_crash_handler()
    t0 = time.time()

    _configure_adapter_for_fisher(adapter, config)

    if progress:
        progress("build_model", 0.0)
    log_memory("build_pack: start")
    model, config = adapter.prepare_model(config)

    target_modules = list_target_modules(model, config)
    if not target_modules:
        raise RuntimeError(
            "Configured regex matched zero linear modules. Run "
            "`python scripts/build_pack.py debug-regex --model <model> --checkpoint <path>` "
            "to see what your model exposes."
        )
    logger.info(
        "Found %d target linears: %d LLM + %d DiT",
        len(target_modules),
        sum(1 for _, s, _ in target_modules if s == "llm"),
        sum(1 for _, s, _ in target_modules if s == "dit"),
    )

    adapter.warmup_for_calibration(model)

    from qvla.build.calibration_noise import install_per_sample_calibration_noise

    install_per_sample_calibration_noise(
        adapter,
        build_seed=config.build_seed,
        noise_ensemble_k=config.noise_ensemble_k,
        mode=config.calibration_noise_mode,
    )
    if config.noise_ensemble_k > 1:
        logger.info(
            "Noise-ensemble calibration: %d noises, mode=%s (DiT targets only; "
            "LLM-only passes use K=1).",
            config.noise_ensemble_k,
            config.calibration_noise_mode,
        )

    fisher_action_dim: int | None = None
    if config.needs_fisher or config.llm.fisher_gptq or config.dit.fisher_gptq:
        from qvla.build.fisher import (
            compute_fisher_sensitivity,
            resolve_fisher_action_dim,
        )

        fisher_action_dim = resolve_fisher_action_dim(adapter)

    # Fisher in original space — used for perm (perm_score=fisher) and
    # Fisher-weighted SVD when sensitivity is available.
    fisher_sensitivities: dict[str, torch.Tensor] = {}
    if config.needs_fisher:
        if progress:
            progress("fisher", 0.05)
        logger.info("Fisher pass (original space) ...")
        fisher_sensitivities = compute_fisher_sensitivity(
            adapter,
            target_modules,
            action_dim=fisher_action_dim,
            num_samples=config.fisher_num_samples,
            num_dit_steps=config.dit.num_steps,
            step_aggregation=config.fisher_step_aggregation,
            noise_ensemble_k=config.noise_ensemble_k,
            action_timestep=config.fisher_action_timestep,
            batch_size=config.fisher_batch_size,
            fisher_type=config.fisher_type,
            method=config.fisher_method,
            hutchinson_probes=config.fisher_hutchinson_probes,
            progress=progress,
        )
        logger.info(
            "Fisher pass done — %d layers with non-zero sensitivity.",
            sum(1 for v in fisher_sensitivities.values() if v.abs().sum() > 0),
        )

    if progress:
        progress("build_rotation", 0.30)
    rotations = _build_rotations_pipeline_wise(
        adapter,
        model,
        target_modules,
        config,
        num_samples,
        fisher_sensitivities,
        progress,
    )

    # Fisher in rotated space — used for Fisher-GPTQ.
    # Computed AFTER rotations are fitted so that rotation.apply(grad)
    # preserves cross-channel correlations through Hadamard (unlike the
    # diagonal approximation which uniformises within rotation blocks).
    fisher_gptq_sensitivities: dict[str, torch.Tensor] = {}
    if config.llm.fisher_gptq or config.dit.fisher_gptq:
        if progress:
            progress("fisher_gptq", 0.50)
        fisher_gptq_targets = _fisher_targets_for_gptq(target_modules, config)
        gptq_rots = {
            name: rotations[name]
            for name, _, _ in fisher_gptq_targets
            if name in rotations
        }
        logger.info(
            "Fisher pass (rotated space for GPTQ) on %d / %d layers ...",
            len(fisher_gptq_targets),
            len(target_modules),
        )
        fisher_gptq_sensitivities = compute_fisher_sensitivity(
            adapter,
            fisher_gptq_targets,
            action_dim=fisher_action_dim,
            num_samples=config.fisher_num_samples,
            num_dit_steps=config.dit.num_steps,
            step_aggregation=config.fisher_step_aggregation,
            noise_ensemble_k=config.noise_ensemble_k,
            action_timestep=config.fisher_action_timestep,
            batch_size=config.fisher_batch_size,
            fisher_type=config.fisher_type,
            method=config.fisher_method,
            hutchinson_probes=config.fisher_hutchinson_probes,
            rotations=gptq_rots,
            progress=progress,
        )
        logger.info(
            "Fisher-GPTQ pass done — %d layers with non-zero sensitivity.",
            sum(1 for v in fisher_gptq_sensitivities.values() if v.abs().sum() > 0),
        )

    if progress:
        progress("calibrate", 0.55)
    log_memory("build_pack: before final calibration")
    logger.info(
        "Final calibration pass on full pipeline (Hessian + act scales) ..."
    )
    quant_stats = _run_transformed_calibration(
        adapter,
        model,
        target_modules,
        rotations,
        config=config,
        num_samples=num_samples,
        log_label="quant",
        progress=progress,
        progress_base=0.55,
        progress_span=0.20,
        amax_plan_by_scope={
            "llm": _amax_collect_plan(config.llm),
            "dit": _amax_collect_plan(config.dit),
        },
    )

    layer_packs: dict[str, LayerPack] = {}
    for j, (name, scope, mod) in enumerate(target_modules):
        scope_cfg = _scope_cfg(config, scope)
        try:
            lp = _make_layer_pack(
                name,
                scope,
                mod,
                quant_stats[name],
                scope_cfg,
                rotations[name],
                fisher_rotated=fisher_gptq_sensitivities.get(name),
            )
        except Exception as e:
            if not config.skip_incompatible:
                raise RuntimeError(
                    f"Failed to quantize layer {name!r} (scope={scope!r}). "
                    "Set skip_incompatible=True to skip incompatible layers."
                ) from e
            logger.warning("Skipping %s due to error: %s", name, e)
            continue
        layer_packs[name] = lp
        if progress:
            frac = 0.75 + 0.25 * (j + 1) / len(target_modules)
            progress("quantize", frac)

    pack = Pack(
        config=config,
        layers=layer_packs,
        meta={
            "model_kind": adapter.model_kind,
            "num_samples": num_samples,
            "calibration_noise_mode": config.calibration_noise_mode,
            "calibration_noise_seed": config.build_seed,
            "noise_ensemble_k": config.noise_ensemble_k,
            "build_seconds": round(time.time() - t0, 2),
        },
    )
    pack.save(output_path)
    if progress:
        progress("done", 1.0)
    return pack


__all__ = ["build_pack"]
