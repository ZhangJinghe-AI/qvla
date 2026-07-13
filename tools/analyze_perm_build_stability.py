#!/usr/bin/env python
"""Perm index stability across calibration builds (no pack build required).

Fits zigzag ``perm`` the same way as ``build_pi05_pack`` / ``PipelineRotationBuild``
(activation ``static_channel_amax`` → ``zigzag_permutation``), then measures how often
repeated "builds" agree.

Three heatmaps (linear layer × denoise step):

0. Fixed sample + fixed noise: per (layer, step) perm, then cross-step overlap within each layer
1. Fixed sample, varying noise: pairwise overlap across trials (first 32 perm slots)
2. Fixed noise, varying samples: pairwise overlap across trials (first 32 perm slots)

Example::

    uv run python tools/analyze_perm_build_stability.py \\
        --checkpoint /data/share/pi05_libero_finetuned_v044 \\
        --calibration-source file \\
        --calibration-data ../calibration_data/libero_object_16_7.npz \\
        --sample-index 0 \\
        --num-noise-trials 16 \\
        --num-samples 8 \\
        --output-dir tools/img/perm_stability
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

_TOOLS = Path(__file__).resolve().parent
_ROOT = _TOOLS.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_TOOLS))

import analyze_dit_noise_sensitivity as dns  # noqa: E402
import analyze_dit_step_activations as dsa  # noqa: E402

from qvla.adapters import get_adapter  # noqa: E402
from qvla.config import QVLAConfig  # noqa: E402
from qvla.core.rotation import (  # noqa: E402
    PermScore,
    PipelineRotationBuild,
    SvdSource,
    parse_pipeline_string,
    step_needs_activation_calibration,
    validate_pipeline,
)
from qvla.runtime import list_target_modules  # noqa: E402
from weight_row_distribution import (  # noqa: E402
    _apply_input_step,
    _layer_stats_from_tokens,
)


@dataclass
class PermOverlapCell:
    layer_name: str
    layer_idx: int
    kind: dsa.LayerKind
    step: int
    mean_pairwise_agreement: float
    min_pairwise_agreement: float
    max_pairwise_agreement: float
    all_identical: bool


def _resolve_targets(
    model: torch.nn.Module,
    config: QVLAConfig,
    *,
    scope: str,
    layer_regex: str | None,
) -> list[tuple[str, str, torch.nn.Module]]:
    if scope == "dit":
        return dsa._resolve_dit_targets(model, config, layer_regex=layer_regex)
    targets = list_target_modules(model, config)
    if layer_regex:
        import re

        pattern = re.compile(layer_regex)
        targets = [(n, s, m) for n, s, m in targets if pattern.search(n)]
    if not targets:
        raise SystemExit(f"No layers matched scope={scope!r}; check --layer-regex.")
    return sorted(targets, key=lambda t: (t[1], t[0]))


def _collect_tokens(
    adapter,
    batch: dict,
    targets: list[tuple[str, str, torch.nn.Module]],
    *,
    noise: torch.Tensor,
) -> dict[str, dict[int, torch.Tensor]]:
    """Pre-linear tokens per (layer, Euler step) from one forward."""
    stores: dict[str, dict[int, list[torch.Tensor]]] = {name: {} for name, _, _ in targets}
    current_step: list[int | None] = [None]
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(layer_name: str):
        def hook(_mod, inputs):
            step = current_step[0]
            if step is None or not inputs or not torch.is_tensor(inputs[0]):
                return
            step_store = stores[layer_name]
            if step not in step_store:
                step_store[step] = []
            step_store[step].append(
                inputs[0].reshape(-1, inputs[0].shape[-1]).detach().cpu()
            )

        return hook

    for name, _, mod in targets:
        handles.append(mod.register_forward_pre_hook(make_hook(name)))

    def step_cb(step: int | None) -> None:
        current_step[0] = step

    try:
        dns._forward_with_noise(adapter, batch, noise, step_cb)
    finally:
        for handle in handles:
            handle.remove()

    out: dict[str, dict[int, torch.Tensor]] = {}
    for name, _, _ in targets:
        step_dict = stores[name]
        if not step_dict:
            raise RuntimeError(f"No activations captured for {name}.")
        out[name] = {
            step: torch.cat(chunks, dim=0).to(torch.float32)
            for step, chunks in sorted(step_dict.items())
        }
    return out


def fit_perm_indices(
    weight: torch.Tensor,
    activation_tokens: torch.Tensor,
    *,
    block_size: int,
    pipeline: tuple[str, ...],
    perm_score: PermScore,
    svd_source: SvdSource,
    layer_name: str | None = None,
    build_seed: int = 0,
) -> torch.Tensor:
    """Fit zigzag perm indices only (no SVD/Hadamard materialization)."""
    steps = validate_pipeline(pipeline)  # type: ignore[arg-type]
    if "perm" not in steps:
        raise ValueError(f"pipeline must include 'perm', got {pipeline!r}.")
    perm_idx = steps.index("perm")

    w = weight.detach().to(torch.float32).clone()
    x = activation_tokens.detach().to(torch.float32).clone()
    d = int(w.shape[1])
    if x.shape[-1] != d:
        raise ValueError(f"activation in_features={x.shape[-1]} != weight in_features={d}.")

    builder = PipelineRotationBuild(
        d=d,
        block_size=block_size,
        weight=weight,
        pipeline=steps,  # type: ignore[arg-type]
        perm_score=perm_score,
        svd_source=svd_source,
        layer_name=layer_name,
        build_seed=build_seed,
    )

    for i in range(perm_idx + 1):
        step = steps[i]
        stats = None
        if step_needs_activation_calibration(
            step, perm_score=perm_score, svd_source=svd_source
        ):
            stats = _layer_stats_from_tokens(x, step, in_features=d)
        builder.fit_step(i, stats=stats)
        if i == perm_idx:
            break
        step_kwargs = {
            "perm": builder.perm if step == "perm" else None,
            "u_blocks": builder.u_blocks if step == "svd" else None,
            "random_hadamard_blocks": (
                builder.random_hadamard_blocks if step == "random_hadamard" else None
            ),
            "block_size": block_size,
            "d": d,
        }
        w = _apply_input_step(w, step, **step_kwargs)
        x = _apply_input_step(x, step, **step_kwargs)

    if builder.perm is None:
        raise RuntimeError("perm fit failed.")
    return builder.perm.cpu()


def perm_slot_agreement(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    prefix_len: int | None = None,
) -> float:
    """Fraction of output slots mapping to the same source channel."""
    if prefix_len is not None:
        a = a[:prefix_len]
        b = b[:prefix_len]
    if a.shape != b.shape:
        raise ValueError(f"perm shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}.")
    return float((a == b).float().mean().item())


def _pairwise_perm_stats(
    perms: list[torch.Tensor],
    *,
    prefix_len: int | None = None,
) -> tuple[float, float, float, bool]:
    agreements: list[float] = []
    for i in range(len(perms)):
        for j in range(i + 1, len(perms)):
            agreements.append(perm_slot_agreement(perms[i], perms[j], prefix_len=prefix_len))
    if not agreements:
        return 1.0, 1.0, 1.0, True
    return (
        sum(agreements) / len(agreements),
        min(agreements),
        max(agreements),
        all(a == 1.0 for a in agreements),
    )


def _compute_overlap_grid(
    perms_by_run: list[dict[tuple[str, int], torch.Tensor]],
    *,
    label: str,
    prefix_len: int | None = None,
) -> list[PermOverlapCell]:
    keys = sorted(perms_by_run[0])
    rows: list[PermOverlapCell] = []
    suffix = f", first {prefix_len} slots" if prefix_len is not None else ""
    print(f"Computing perm overlap: {label} ({len(keys)} layer×step cells{suffix})...")
    for key in keys:
        name, step = key
        perms = [run[key] for run in perms_by_run]
        mean_ag, min_ag, max_ag, identical = _pairwise_perm_stats(perms, prefix_len=prefix_len)
        rows.append(
            PermOverlapCell(
                layer_name=name,
                layer_idx=dsa._layer_idx(name),
                kind=dsa._layer_kind(name),
                step=step,
                mean_pairwise_agreement=mean_ag,
                min_pairwise_agreement=min_ag,
                max_pairwise_agreement=max_ag,
                all_identical=identical,
            )
        )
    return rows


def _compute_step_cross_overlap(
    perms: dict[tuple[str, int], torch.Tensor],
    *,
    label: str,
    prefix_len: int | None = None,
) -> list[PermOverlapCell]:
    """Within each layer, compare perm at one step vs all other steps."""
    by_layer: dict[str, list[tuple[int, torch.Tensor]]] = {}
    for (name, step), perm in perms.items():
        by_layer.setdefault(name, []).append((step, perm))

    rows: list[PermOverlapCell] = []
    print(f"Computing step cross-overlap: {label} ({len(perms)} layer×step cells)...")
    for name in sorted(by_layer):
        step_list = sorted(by_layer[name], key=lambda x: x[0])
        for step, perm in step_list:
            others = [p for s, p in step_list if s != step]
            if not others:
                mean_ag, min_ag, max_ag, identical = 1.0, 1.0, 1.0, True
            else:
                agreements = [
                    perm_slot_agreement(perm, other, prefix_len=prefix_len)
                    for other in others
                ]
                mean_ag = sum(agreements) / len(agreements)
                min_ag = min(agreements)
                max_ag = max(agreements)
                identical = all(a == 1.0 for a in agreements)
            rows.append(
                PermOverlapCell(
                    layer_name=name,
                    layer_idx=dsa._layer_idx(name),
                    kind=dsa._layer_kind(name),
                    step=step,
                    mean_pairwise_agreement=mean_ag,
                    min_pairwise_agreement=min_ag,
                    max_pairwise_agreement=max_ag,
                    all_identical=identical,
                )
            )
    return rows


def _layer_labels(rows: list[PermOverlapCell]) -> list[tuple[str, str]]:
    kind_order = {"qkv_proj": 0, "o_proj": 1, "gate_up_proj": 2, "down_proj": 3, "other": 4}
    keys = sorted(
        {(r.layer_name, r.layer_idx, r.kind) for r in rows},
        key=lambda t: (t[1], kind_order.get(t[2], 9), t[0]),
    )
    return [
        (name, f"L{layer_idx:02d}.{kind.replace('_proj', '').replace('gate_up', 'gu')}")
        for name, layer_idx, kind in keys
    ]


def _plot_perm_overlap(
    rows: list[PermOverlapCell],
    path: Path,
    *,
    num_steps: int,
    suptitle: str,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    layer_axis = _layer_labels(rows)
    ylabels = [label for _, label in layer_axis]
    steps = list(range(num_steps))

    def _matrix(attr: str) -> np.ndarray:
        mat = np.full((len(layer_axis), num_steps), np.nan)
        lookup = {(r.layer_name, r.step): getattr(r, attr) for r in rows}
        for i, (name, _label) in enumerate(layer_axis):
            for step in steps:
                val = lookup.get((name, step))
                if val is not None:
                    mat[i, step] = val
        return mat

    panels = [
        (_matrix("mean_pairwise_agreement"), "Mean pairwise slot agreement", "viridis", 0.0, 1.0),
        (_matrix("min_pairwise_agreement"), "Min pairwise slot agreement", "viridis", 0.0, 1.0),
    ]

    fig, axes = plt.subplots(
        1, 2, figsize=(14, max(8, len(layer_axis) * 0.16)), sharey=True
    )
    for ax, (mat, title, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_xlabel("Denoise step")
        ax.set_title(title)
        ax.set_xticks(steps)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    axes[0].set_ylabel("DiT linear layer")
    y_stride = max(1, len(ylabels) // 18)
    axes[0].set_yticks(range(0, len(ylabels), y_stride))
    axes[0].set_yticklabels([ylabels[i] for i in range(0, len(ylabels), y_stride)], fontsize=7)

    fig.suptitle(suptitle, fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def _print_summary(rows: list[PermOverlapCell], *, title: str) -> None:
    import statistics

    means = [r.mean_pairwise_agreement for r in rows]
    mins = [r.min_pairwise_agreement for r in rows]
    maxs = [r.max_pairwise_agreement for r in rows]
    identical = sum(1 for r in rows if r.all_identical)
    print(f"\n=== {title} ===")
    print(f"  layer×step cells: {len(rows)}")
    print(f"  mean slot agreement (cell avg): {statistics.mean(means):.4f}")
    print(f"  median mean agreement:          {statistics.median(means):.4f}")
    print(f"  min of per-cell min agreement:  {min(mins):.4f}")
    print(f"  max of per-cell max agreement:  {max(maxs):.4f}")
    print(f"  cells with 100% identical perm: {identical}/{len(rows)}")

    ranked = sorted(rows, key=lambda r: r.mean_pairwise_agreement)
    print("  lowest agreement cells:")
    for r in ranked[:5]:
        print(
            f"    {r.layer_name} step {r.step}: mean={r.mean_pairwise_agreement:.3f} "
            f"min={r.min_pairwise_agreement:.3f} max={r.max_pairwise_agreement:.3f}"
        )


def _fit_perms_for_forward(
    targets: list[tuple[str, str, torch.nn.Module]],
    tokens: dict[str, dict[int, torch.Tensor]],
    *,
    block_size: int,
    pipeline: tuple[str, ...],
    perm_score: PermScore,
    svd_source: SvdSource,
) -> dict[tuple[str, int], torch.Tensor]:
    out: dict[tuple[str, int], torch.Tensor] = {}
    for name, _, module in targets:
        weight = module.weight.detach().to(torch.float32)
        for step, step_tokens in tokens[name].items():
            out[(name, step)] = fit_perm_indices(
                weight,
                step_tokens,
                block_size=block_size,
                pipeline=pipeline,
                perm_score=perm_score,
                svd_source=svd_source,
                layer_name=name,
            )
    return out


def _run_noise_sweep(
    adapter,
    targets,
    batch: dict,
    *,
    num_trials: int,
    noise_seed_base: int,
    sample_index: int,
    block_size: int,
    pipeline: tuple[str, ...],
    perm_score: PermScore,
    svd_source: SvdSource,
) -> list[dict[tuple[str, int], torch.Tensor]]:
    print(
        f"\n[2/3] Fixed input (sample {sample_index}), varying noise — "
        f"{num_trials} trials, seeds {noise_seed_base}..{noise_seed_base + num_trials - 1}"
    )
    runs: list[dict[tuple[str, int], torch.Tensor]] = []
    for i in range(num_trials):
        seed = noise_seed_base + i
        tokens = _collect_tokens(
            adapter,
            batch,
            targets,
            noise=dns._make_noise(adapter, seed),
        )
        runs.append(
            _fit_perms_for_forward(
                targets,
                tokens,
                block_size=block_size,
                pipeline=pipeline,
                perm_score=perm_score,
                svd_source=svd_source,
            )
        )
        print(f"  noise trial {i + 1}/{num_trials} (seed={seed})")
    return runs


def _run_sample_sweep(
    adapter,
    targets,
    *,
    num_samples: int,
    sample_start: int,
    fixed_noise_seed: int,
    block_size: int,
    pipeline: tuple[str, ...],
    perm_score: PermScore,
    svd_source: SvdSource,
) -> list[dict[tuple[str, int], torch.Tensor]]:
    sample_indices = list(range(sample_start, sample_start + num_samples))
    print(
        f"\n[3/3] Fixed noise (seed {fixed_noise_seed}), varying input — "
        f"samples {sample_indices[0]}..{sample_indices[-1]}"
    )
    noise = dns._make_noise(adapter, fixed_noise_seed)
    runs: list[dict[tuple[str, int], torch.Tensor]] = []
    for i, sample_idx in enumerate(sample_indices):
        batch = dns._get_calibration_batch(adapter, sample_idx)
        tokens = _collect_tokens(adapter, batch, targets, noise=noise)
        runs.append(
            _fit_perms_for_forward(
                targets,
                tokens,
                block_size=block_size,
                pipeline=pipeline,
                perm_score=perm_score,
                svd_source=svd_source,
            )
        )
        print(f"  input sample {i + 1}/{num_samples} (calib index={sample_idx})")
    return runs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--calibration-data", type=Path)
    p.add_argument("--calibration-source", choices=("file", "synthetic"), required=True)
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--num-noise-trials", type=int, default=16)
    p.add_argument("--noise-seed-base", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--sample-start", type=int, default=0)
    p.add_argument("--sample-fixed-noise-seed", type=int, default=0)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument(
        "--pipeline",
        type=parse_pipeline_string,
        default=parse_pipeline_string("perm,svd,hadamard"),
        help="Pack pipeline; only steps up to perm affect indices (default: perm,svd,hadamard).",
    )
    p.add_argument(
        "--perm-score",
        choices=("weight", "activation", "activation_weight"),
        default="activation",
    )
    p.add_argument(
        "--svd-source",
        choices=("weight", "activation"),
        default="activation",
        help="Used when pipeline has svd before perm.",
    )
    p.add_argument("--scope", choices=("dit", "all"), default="dit")
    p.add_argument("--layer-regex")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--overlap-prefix-len",
        type=int,
        default=32,
        help="Compare only the first N perm slots in noise/sample sweeps (default: 32).",
    )
    args = p.parse_args(argv)

    if args.calibration_source == "file" and args.calibration_data is None:
        p.error("--calibration-data is required when --calibration-source=file.")
    if args.num_noise_trials < 2:
        p.error("--num-noise-trials must be >= 2.")
    if args.num_samples < 2:
        p.error("--num-samples must be >= 2.")

    config = QVLAConfig.pi05_default()
    adapter_kwargs: dict = {
        "checkpoint_path": args.checkpoint,
        "calibration_source": args.calibration_source,
    }
    if args.calibration_source == "file":
        adapter_kwargs["calibration_data_path"] = args.calibration_data

    adapter = get_adapter("pi05", **adapter_kwargs)
    model = adapter.build_model()
    adapter.warmup_for_calibration(model)

    targets = _resolve_targets(model, config, scope=args.scope, layer_regex=args.layer_regex)
    batch = dns._get_calibration_batch(adapter, args.sample_index)
    num_steps = adapter.dit_step_count(config)
    pipeline = tuple(args.pipeline)
    perm_score: PermScore = args.perm_score  # type: ignore[assignment]
    svd_source: SvdSource = args.svd_source  # type: ignore[assignment]

    scope_tag = args.scope
    step_out = args.output_dir / f"{scope_tag}_perm_step_cross_overlap_linear_x_step.png"
    noise_out = args.output_dir / f"{scope_tag}_perm_noise_overlap_linear_x_step_top{args.overlap_prefix_len}.png"
    sample_out = args.output_dir / f"{scope_tag}_perm_sample_overlap_linear_x_step_top{args.overlap_prefix_len}.png"
    fit_kwargs = {
        "block_size": args.block_size,
        "pipeline": pipeline,
        "perm_score": perm_score,
        "svd_source": svd_source,
    }

    try:
        print(
            f"\n[1/3] Fixed sample {args.sample_index} + fixed noise seed "
            f"{args.sample_fixed_noise_seed} — cross-step overlap within each layer"
        )
        fixed_noise = dns._make_noise(adapter, args.sample_fixed_noise_seed)
        fixed_tokens = _collect_tokens(adapter, batch, targets, noise=fixed_noise)
        fixed_perms = _fit_perms_for_forward(targets, fixed_tokens, **fit_kwargs)
        step_rows = _compute_step_cross_overlap(
            fixed_perms,
            label="fixed sample+noise / cross-step",
        )
        _plot_perm_overlap(
            step_rows,
            step_out,
            num_steps=num_steps,
            suptitle=(
                f"Perm slot agreement across denoise steps within each layer "
                f"(sample {args.sample_index}, noise seed {args.sample_fixed_noise_seed}, "
                f"perm_score={perm_score})"
            ),
        )
        _print_summary(step_rows, title="Step cross-overlap (fixed sample+noise)")

        noise_runs = _run_noise_sweep(
            adapter,
            targets,
            batch,
            num_trials=args.num_noise_trials,
            noise_seed_base=args.noise_seed_base,
            sample_index=args.sample_index,
            **fit_kwargs,
        )
        noise_rows = _compute_overlap_grid(
            noise_runs,
            label="fixed input / varying noise",
            prefix_len=args.overlap_prefix_len,
        )
        _plot_perm_overlap(
            noise_rows,
            noise_out,
            num_steps=num_steps,
            suptitle=(
                f"Perm slot agreement across noise trials (first {args.overlap_prefix_len} slots) "
                f"(fixed sample {args.sample_index}, "
                f"seeds {args.noise_seed_base}.."
                f"{args.noise_seed_base + args.num_noise_trials - 1}, "
                f"perm_score={perm_score})"
            ),
        )
        _print_summary(noise_rows, title="Noise sweep")

        sample_runs = _run_sample_sweep(
            adapter,
            targets,
            num_samples=args.num_samples,
            sample_start=args.sample_start,
            fixed_noise_seed=args.sample_fixed_noise_seed,
            **fit_kwargs,
        )
        sample_rows = _compute_overlap_grid(
            sample_runs,
            label="fixed noise / varying input",
            prefix_len=args.overlap_prefix_len,
        )
        _plot_perm_overlap(
            sample_rows,
            sample_out,
            num_steps=num_steps,
            suptitle=(
                f"Perm slot agreement across input samples (first {args.overlap_prefix_len} slots) "
                f"(fixed noise seed {args.sample_fixed_noise_seed}, "
                f"samples {args.sample_start}.."
                f"{args.sample_start + args.num_samples - 1}, "
                f"perm_score={perm_score})"
            ),
        )
        _print_summary(sample_rows, title="Sample sweep")
    except ImportError as e:
        raise SystemExit("Install matplotlib (uv sync --group dev) for plots.") from e
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
