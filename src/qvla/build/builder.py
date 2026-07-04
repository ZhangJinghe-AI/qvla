"""Offline QVLA pack builder.

End-to-end pipeline:

1. Adapter builds the full-precision model (eager mode).
2. We list every target Linear with :func:`wrap.list_target_modules`.
3. Optional Fisher pass (``perm_score=fisher``).
4. **Pipeline-wise rotation fit** — for each scope pipeline step that needs
   activation stats, run a calibration pass on ``prefix_rotation.apply(x)``,
   then fit that step; weight-only steps fit immediately from ``W``.
5. **Final calibration pass** on the full ``rotation.apply(x)`` for GPTQ /
   act-scale tables.
6. Quantize rotated weights and serialize the :class:`Pack`.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from qvla.adapters import ModelAdapter
from qvla.build.collector import (
    LayerStats,
    RotatedActivationCollector,
)
from qvla.config import QVLAConfig, ScopeConfig
from qvla.core.pack import LayerPack, Pack
from qvla.core.quantize import (
    QuantizedWeight,
    is_no_quant,
    percentile_amax,
    quantize_weight,
    symmetric_quant_range,
)
from qvla.core.rotation import (
    PipelineRotationBuild,
    Rotation,
    identity_rotation,
    step_needs_activation_calibration,
)
from qvla.runtime.wrap import list_target_modules


logger = logging.getLogger(__name__)


def _scope_cfg(config: QVLAConfig, scope: str) -> ScopeConfig:
    return config.llm if scope == "llm" else config.dit


def _needs_offline_act_scale(scope_cfg: ScopeConfig) -> bool:
    """True when the runtime uses a baked ``act_scale_table`` (not dynamic)."""
    return scope_cfg.act_scale_mode in ("static", "per_step")


def _act_scale_amax(
    stats: LayerStats, scope_cfg: ScopeConfig, *, step: int | None = None
) -> torch.Tensor:
    p = scope_cfg.act_percentile
    if scope_cfg.act_percentile_mode == "cross_channel":
        row = stats.static_amax if step is None else stats.per_step_amax[step]
        return percentile_amax(row, p)
    return stats.act_percentile_amax(p, step=step)


def _build_act_scale_table(
    stats: LayerStats, scope_cfg: ScopeConfig
) -> torch.Tensor | None:
    """Convert collector stats into the runtime layer's ``act_scale_table``."""
    if scope_cfg.act_scale_mode == "dynamic":
        return None
    if is_no_quant(scope_cfg.act_bits):
        return None
    _, act_qmax = symmetric_quant_range(scope_cfg.act_bits)
    if scope_cfg.act_scale_mode == "static":
        amax = _act_scale_amax(stats, scope_cfg)
        return (amax.clamp_min(1e-12) / act_qmax).to(torch.float32)
    if stats.per_step_amax is None:
        logger.warning(
            "Requested per_step scales but collector has no per-step amax; "
            "downgrading to static."
        )
        amax = _act_scale_amax(stats, scope_cfg)
        return (amax.clamp_min(1e-12) / act_qmax).unsqueeze(0).to(torch.float32)
    rows = [
        _act_scale_amax(stats, scope_cfg, step=s)
        for s in range(stats.per_step_amax.shape[0])
    ]
    table = torch.stack(rows, dim=0).clamp_min(1e-12) / act_qmax
    return table.to(torch.float32)


def _rotate_weight(w: torch.Tensor, rotation: Rotation) -> torch.Tensor:
    """Apply DuQuant input transform to ``W`` (matches runtime ``Rotation.apply``)."""
    if rotation.is_identity:
        return w.contiguous()
    return rotation.apply(w.to(torch.float32)).contiguous()


def _run_transformed_calibration(
    adapter: ModelAdapter,
    model: nn.Module,
    targets: list[tuple[str, str, nn.Module]],
    transforms: dict[str, Rotation],
    *,
    config: QVLAConfig,
    num_samples: int,
    quant_stats: bool,
    log_label: str,
    progress: Callable[[str, float], None] | None,
    progress_base: float,
    progress_span: float,
) -> dict[str, LayerStats]:
    """Forward ``num_samples`` batches; collect stats on ``transform.apply(x)``."""
    num_steps = {"llm": 1, "dit": config.dit.num_steps}
    with RotatedActivationCollector(
        targets,
        transforms,
        num_steps_by_scope=num_steps,
        device="cpu",
        quant_stats=quant_stats,
    ) as collector:
        def step_cb(step: int) -> None:
            collector.set_current_step(step)

        with torch.inference_mode():
            for i, batch in enumerate(adapter.iter_calibration_batches(num_samples)):
                logger.info("%s sample %d/%d ...", log_label, i + 1, num_samples)
                adapter.forward_for_calibration(model, batch, step_callback=step_cb)
                if progress:
                    frac = progress_base + progress_span * (i + 1) / max(1, num_samples)
                    progress("calibrate", frac)
    return {name: collector.stats[name] for name, _s, _m in targets}


def _build_rotations_pipeline_wise(
    adapter: ModelAdapter,
    model: nn.Module,
    target_modules: list[tuple[str, str, nn.Module]],
    config: QVLAConfig,
    num_samples: int,
    fisher_sensitivities: dict[str, torch.Tensor],
    progress: Callable[[str, float], None] | None,
) -> dict[str, Rotation]:
    """Fit each layer rotation by walking the scope pipeline step-by-step."""
    by_scope: dict[str, list[tuple[str, str, nn.Module]]] = defaultdict(list)
    for item in target_modules:
        by_scope[item[1]].append(item)

    rotations: dict[str, Rotation] = {}
    step_idx = 0

    for scope, scope_targets in by_scope.items():
        scope_cfg = _scope_cfg(config, scope)
        pipeline = scope_cfg.pipeline
        if not pipeline:
            for name, _s, mod in scope_targets:
                in_f = int(getattr(mod, "in_features"))
                rotations[name] = identity_rotation(
                    in_f, scope_cfg.rotation_block_size
                )
            continue

        builders: dict[str, PipelineRotationBuild] = {}
        for name, _s, mod in scope_targets:
            w = getattr(mod, "weight").detach()
            in_f = int(w.shape[1])
            builders[name] = PipelineRotationBuild(
                d=in_f,
                block_size=scope_cfg.rotation_block_size,
                weight=w,
                pipeline=pipeline,
                perm_score=scope_cfg.perm_score,
                svd_source=scope_cfg.svd_source,
                sensitivity=fisher_sensitivities.get(name),
                layer_name=name,
                build_seed=config.build_seed,
            )

        for step_index, step in enumerate(pipeline):
            if step_needs_activation_calibration(
                step,
                perm_score=scope_cfg.perm_score,
                svd_source=scope_cfg.svd_source,
            ):
                prefix = {name: builders[name].prefix_rotation(step_index) for name, _, _ in scope_targets}
                logger.info(
                    "Pipeline calibration (%s step %d/%d: %s on prefix %s) ...",
                    scope,
                    step_index + 1,
                    len(pipeline),
                    step,
                    "+".join(pipeline[:step_index]) or "none",
                )
                stats = _run_transformed_calibration(
                    adapter,
                    model,
                    scope_targets,
                    prefix,
                    config=config,
                    num_samples=num_samples,
                    quant_stats=False,
                    log_label=f"{scope} pipeline-{step}",
                    progress=progress,
                    progress_base=0.05 + 0.25 * step_idx / max(1, len(pipeline)),
                    progress_span=0.25 / max(1, len(pipeline)),
                )
                step_idx += 1
                for name, _, _ in scope_targets:
                    builders[name].fit_step(step_index, stats=stats[name])
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
    rotation: Rotation,
) -> LayerPack | None:
    """Quantize one layer using stats on ``rotation.apply(x)``."""
    w = getattr(module, "weight").detach()
    bias_attr = getattr(module, "bias", None)
    bias = bias_attr.detach().to(torch.float32) if bias_attr is not None else None
    bias_present = bias is not None

    N, K = w.shape
    in_features = K
    out_features = N
    device = w.device

    W_rot = _rotate_weight(w.to(torch.float32), rotation)
    H_rot = quant_stats.hessian().to(device)

    try:
        qw: QuantizedWeight = quantize_weight(
            W_rot,
            scope_cfg.weight_quantizer,
            group_size=scope_cfg.group_size,
            weight_bits=scope_cfg.weight_bits,
            hessian=H_rot,
            gptq_block_size=scope_cfg.gptq_block_size,
            gptq_damp_percent=scope_cfg.gptq_damp_percent,
        )
    except ValueError as e:
        logger.warning("Skipping %s (quantizer error): %s", name, e)
        return None

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
        rotation=rotation,
        act_bits=scope_cfg.act_bits,
        act_scale_mode=scope_cfg.act_scale_mode,
        act_scale_table=act_scale_table,
        bias=bias,
        residual=qw.residual,
        extras=extras,
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
    t0 = time.time()

    if progress:
        progress("build_model", 0.0)
    model, config = adapter.prepare_model(config)

    target_modules = list_target_modules(model, config)
    if not target_modules:
        raise RuntimeError(
            "Configured regex matched zero linear modules. Run "
            "`python scripts/build_pi05_pack.py debug-regex --checkpoint <path>` "
            "to see what your model exposes."
        )
    logger.info(
        "Found %d target linears: %d LLM + %d DiT",
        len(target_modules),
        sum(1 for _, s, _ in target_modules if s == "llm"),
        sum(1 for _, s, _ in target_modules if s == "dit"),
    )

    adapter.warmup_for_calibration(model)

    fisher_sensitivities: dict[str, torch.Tensor] = {}
    if config.needs_fisher:
        if progress:
            progress("fisher", 0.05)
        logger.info("Fisher perm requested — running Fisher pass ...")
        from qvla.build.fisher import compute_fisher_sensitivity

        fisher_sensitivities = compute_fisher_sensitivity(
            adapter,
            target_modules,
            num_samples=config.fisher_num_samples,
            num_dit_steps=config.dit.num_steps,
            step_aggregation=config.fisher_step_aggregation,
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

    if progress:
        progress("calibrate", 0.55)
    logger.info("Final calibration pass on full pipeline (Hessian + act scales) ...")
    quant_stats = _run_transformed_calibration(
        adapter,
        model,
        target_modules,
        rotations,
        config=config,
        num_samples=num_samples,
        quant_stats=True,
        log_label="quant",
        progress=progress,
        progress_base=0.55,
        progress_span=0.20,
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
            )
        except Exception as e:
            if not config.skip_incompatible:
                raise RuntimeError(
                    f"Failed to quantize layer {name!r} (scope={scope!r}). "
                    "Set skip_incompatible=True to skip incompatible layers."
                ) from e
            logger.warning("Skipping %s due to error: %s", name, e)
            continue
        if lp is None:
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
            "build_seconds": round(time.time() - t0, 2),
        },
    )
    pack.save(output_path)
    if progress:
        progress("done", 1.0)
    return pack


__all__ = ["build_pack"]
