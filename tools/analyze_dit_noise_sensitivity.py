#!/usr/bin/env python
"""DiT per-channel L2 similarity: noise sweep and input sweep.

Two linear×step heatmaps (Pearson / Spearman / Cosine, mean pairwise):

1. Fixed calibration input, varying diffusion noise
2. Fixed diffusion noise, varying calibration inputs

Outputs land flat under ``--output-dir``::

    dit_noise_similarity_by_channel_l2_linear_x_step.png
    dit_sample_similarity_by_channel_l2_linear_x_step.png

With ``--compare-sample-splits``, also writes::

    dit_sample_split_pearson_compare.png

Example::

    uv run python tools/analyze_dit_noise_sensitivity.py \\
        --checkpoint /data/share/pi05_libero_finetuned_v044 \\
        --calibration-source file \\
        --calibration-data ../calibration_data/libero_object_16_7.npz \\
        --sample-index 0 \\
        --num-noise-trials 16 \\
        --num-samples 8 \\
        --output-dir tools/img/dit_sensitivity
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

import analyze_dit_step_activations as dsa  # noqa: E402

from qvla.adapters import get_adapter  # noqa: E402
from qvla.adapters.pi05.obs import build_pi05_request  # noqa: E402
from qvla.adapters.pi05.step_hook import find_expert_runner, patched_one_step  # noqa: E402
from qvla.config import QVLAConfig  # noqa: E402


@dataclass
class SimilarityCell:
    layer_name: str
    layer_idx: int
    kind: dsa.LayerKind
    step: int
    pearson: float
    spearman: float
    cosine: float


def _get_calibration_batch(adapter, sample_index: int) -> dict:
    batches = list(adapter.iter_calibration_batches(sample_index + 1))
    if sample_index >= len(batches):
        raise SystemExit(
            f"sample index {sample_index} out of range; "
            f"calibration provides {len(batches)} sample(s)."
        )
    return batches[sample_index]


def _make_noise(adapter, seed: int) -> torch.Tensor:
    sched = adapter._engine.entry.scheduler
    cfg = sched.cfg
    generator = torch.Generator(device=sched.device)
    generator.manual_seed(seed)
    return torch.randn(
        1,
        cfg.chunk_size,
        cfg.max_action_dim,
        generator=generator,
        device=sched.device,
        dtype=sched.params_dtype,
    )


def _forward_with_noise(adapter, batch: dict, noise: torch.Tensor, step_callback) -> None:
    from dataclasses import replace

    adapter._ensure_processor()
    request = build_pi05_request(
        adapter._processor, batch, state_dim=adapter.cfg.state_dim
    )
    request = replace(request, noise=noise)
    runner = find_expert_runner(adapter._engine)
    with patched_one_step(runner, step_callback):
        step_callback(None)
        _ = adapter._engine.step(request)


def _collect_activations(
    adapter,
    batch: dict,
    targets: list[tuple[str, str, torch.nn.Module]],
    *,
    noise: torch.Tensor,
    act_metric: dsa.ActMetric,
) -> dict[str, dict[int, torch.Tensor]]:
    in_features = {name: mod.weight.shape[1] for name, _, mod in targets}
    store: dict[str, dict[int, dsa._PerStepAccumulator]] = {
        name: {} for name, _, _ in targets
    }
    current_step: list[int | None] = [None]
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(layer_name: str):
        def hook(_mod, inputs):
            step = current_step[0]
            if step is None or not inputs or not torch.is_tensor(inputs[0]):
                return
            step_dict = store[layer_name]
            if step not in step_dict:
                step_dict[step] = dsa._PerStepAccumulator(act_metric, in_features[layer_name])
            step_dict[step].update(inputs[0])

        return hook

    for name, _, mod in targets:
        handles.append(mod.register_forward_pre_hook(make_hook(name)))

    def step_cb(step: int | None) -> None:
        current_step[0] = step

    try:
        _forward_with_noise(adapter, batch, noise, step_cb)
    finally:
        for handle in handles:
            handle.remove()

    return {
        name: {step: acc.channel_scores() for step, acc in step_dict.items()}
        for name, step_dict in store.items()
    }


def _pairwise_mean_similarity(vectors: list[torch.Tensor]) -> tuple[float, float, float]:
    pearsons: list[float] = []
    spearmans: list[float] = []
    cosines: list[float] = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            pearsons.append(dsa._pearson(vectors[i], vectors[j]))
            spearmans.append(dsa._spearman(vectors[i], vectors[j]))
            cosines.append(dsa._cosine(vectors[i], vectors[j]))
    if not pearsons:
        return 1.0, 1.0, 1.0
    n = len(pearsons)
    return sum(pearsons) / n, sum(spearmans) / n, sum(cosines) / n


def _compute_similarity_grid(
    runs: list[dict[str, dict[int, torch.Tensor]]],
    *,
    num_steps: int,
    label: str,
) -> list[SimilarityCell]:
    rows: list[SimilarityCell] = []
    layer_names = sorted(runs[0])
    total = len(layer_names) * num_steps
    done = 0
    print(f"Computing similarity grid: {label} ({total} cells)...")

    for name in layer_names:
        idx = dsa._layer_idx(name)
        kind = dsa._layer_kind(name)
        for step in range(num_steps):
            vectors = [r[name][step] for r in runs if step in r[name]]
            done += 1
            if len(vectors) < 2:
                continue
            p, sp, cos = _pairwise_mean_similarity(vectors)
            rows.append(
                SimilarityCell(
                    layer_name=name,
                    layer_idx=idx,
                    kind=kind,
                    step=step,
                    pearson=p,
                    spearman=sp,
                    cosine=cos,
                )
            )
            if done % 72 == 0:
                print(f"  {label}: {done}/{total} cells")
    print(f"  {label}: done ({len(rows)} cells with data)")
    return rows


def _layer_labels(rows: list[SimilarityCell]) -> list[tuple[str, str]]:
    kind_order = {"qkv_proj": 0, "o_proj": 1, "gate_up_proj": 2, "down_proj": 3, "other": 4}
    keys = sorted(
        {(r.layer_name, r.layer_idx, r.kind) for r in rows},
        key=lambda t: (t[1], kind_order.get(t[2], 9), t[0]),
    )
    return [
        (name, f"L{layer_idx:02d}.{kind.replace('_proj', '').replace('gate_up', 'gu')}")
        for name, layer_idx, kind in keys
    ]


def _plot_linear_step_similarity(
    rows: list[SimilarityCell],
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
        (_matrix("pearson"), "Pearson r", "RdBu_r", -1.0, 1.0),
        (_matrix("spearman"), "Spearman rho", "RdBu_r", -1.0, 1.0),
        (_matrix("cosine"), "Cosine", "viridis", 0.0, 1.0),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, max(10, len(layer_axis) * 0.16)), sharey=True)
    for ax, (mat, title, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_xlabel("Denoise step")
        ax.set_title(title)
        ax.set_xticks(steps)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if np.isnan(val):
                    continue
                dark = val < (vmin + 0.35 * (vmax - vmin))
                ax.text(
                    j, i, f"{val:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if dark else "black",
                )
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


def _calibration_size(
    calibration_source: str,
    calibration_data: Path | None,
    *,
    num_samples: int,
) -> int:
    if calibration_source == "file":
        from qvla.calibration.file import load_calibration_dataset

        batches, _meta = load_calibration_dataset(calibration_data)
        return len(batches)
    return num_samples * 2


def _pearson_matrix(
    rows: list[SimilarityCell],
    *,
    num_steps: int,
) -> tuple[list[str], object]:
    import numpy as np

    layer_axis = _layer_labels(rows)
    ylabels = [label for _, label in layer_axis]
    mat = np.full((len(layer_axis), num_steps), np.nan)
    lookup = {(r.layer_name, r.step): r.pearson for r in rows}
    for i, (name, _label) in enumerate(layer_axis):
        for step in range(num_steps):
            val = lookup.get((name, step))
            if val is not None:
                mat[i, step] = val
    return ylabels, mat


def _print_split_compare_stats(
    first_rows: list[SimilarityCell],
    last_rows: list[SimilarityCell],
    *,
    first_indices: list[int],
    last_indices: list[int],
    metric: str,
) -> None:
    import statistics

    lookup_a = {(r.layer_name, r.step): getattr(r, metric) for r in first_rows}
    lookup_b = {(r.layer_name, r.step): getattr(r, metric) for r in last_rows}
    keys = sorted(set(lookup_a) & set(lookup_b))
    diffs = [lookup_b[k] - lookup_a[k] for k in keys]
    abs_diffs = [abs(d) for d in diffs]

    print(
        f"\n=== {metric} : samples {first_indices[0]}..{first_indices[-1]}"
        f" vs {last_indices[0]}..{last_indices[-1]} ==="
    )
    print(f"  mean({metric}) first half : {statistics.mean(lookup_a[k] for k in keys):.4f}")
    print(f"  mean({metric}) last half  : {statistics.mean(lookup_b[k] for k in keys):.4f}")
    print(f"  mean diff (last-first)    : {statistics.mean(diffs):+.4f}")
    print(f"  mean |diff|              : {statistics.mean(abs_diffs):.4f}")
    print(f"  max |diff|                : {max(abs_diffs):.4f}")

    ranked = sorted(keys, key=lambda k: abs(lookup_b[k] - lookup_a[k]), reverse=True)
    print("  top-5 |diff| cells:")
    for name, step in ranked[:5]:
        a, b = lookup_a[(name, step)], lookup_b[(name, step)]
        print(f"    {name} step {step}: {a:.3f} -> {b:.3f} (Δ{b - a:+.3f})")


def _plot_sample_split_compare(
    first_rows: list[SimilarityCell],
    last_rows: list[SimilarityCell],
    path: Path,
    *,
    num_steps: int,
    num_samples: int,
    first_indices: list[int],
    last_indices: list[int],
    num_calib: int,
    fixed_noise_seed: int,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    ylabels, mat_a = _pearson_matrix(first_rows, num_steps=num_steps)
    _, mat_b = _pearson_matrix(last_rows, num_steps=num_steps)
    mat_diff = mat_b - mat_a
    diff_vmax = max(float(np.nanmax(np.abs(mat_diff))), 0.05)

    fig, axes = plt.subplots(1, 3, figsize=(24, max(10, len(ylabels) * 0.16)))
    y_stride = max(1, len(ylabels) // 18)
    yt = range(0, len(ylabels), y_stride)
    ytl = [ylabels[i] for i in yt]

    for ax, mat, title, cmap, vlo, vhi in (
        (axes[0], mat_a, f"First {num_samples} (0..{first_indices[-1]})", "RdYlBu_r", -1.0, 1.0),
        (
            axes[1],
            mat_b,
            f"Last {num_samples} ({last_indices[0]}..{last_indices[-1]})",
            "RdYlBu_r",
            -1.0,
            1.0,
        ),
        (axes[2], mat_diff, "Δ Pearson (last − first)", "coolwarm", -diff_vmax, diff_vmax),
    ):
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vlo, vmax=vhi, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("Denoise step")
        ax.set_xticks(range(num_steps))
        ax.set_yticks(list(yt))
        ax.set_yticklabels(ytl, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("DiT linear layer")
    fig.suptitle(
        f"Cross-sample Pearson, fixed noise seed {fixed_noise_seed}; "
        f"first vs last half of {num_calib} calibration samples",
        fontsize=11,
        y=1.01,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def _run_noise_sweep(
    adapter,
    targets,
    batch: dict,
    *,
    num_steps: int,
    num_trials: int,
    noise_seed_base: int,
    act_metric: dsa.ActMetric,
    sample_index: int,
) -> list[SimilarityCell]:
    print(
        f"\n[1/2] Fixed input (sample {sample_index}), varying noise — "
        f"{num_trials} trials, seeds {noise_seed_base}..{noise_seed_base + num_trials - 1}"
    )
    runs: list[dict[str, dict[int, torch.Tensor]]] = []
    for i in range(num_trials):
        seed = noise_seed_base + i
        runs.append(
            _collect_activations(
                adapter,
                batch,
                targets,
                noise=_make_noise(adapter, seed),
                act_metric=act_metric,
            )
        )
        print(f"  noise trial {i + 1}/{num_trials} (seed={seed})")
    return _compute_similarity_grid(
        runs,
        num_steps=num_steps,
        label="fixed input / varying noise",
    )


def _run_sample_sweep(
    adapter,
    targets,
    *,
    num_steps: int,
    num_samples: int,
    sample_start: int,
    fixed_noise_seed: int,
    act_metric: dsa.ActMetric,
) -> list[SimilarityCell]:
    sample_indices = list(range(sample_start, sample_start + num_samples))
    print(
        f"\n[2/2] Fixed noise (seed {fixed_noise_seed}), varying input — "
        f"samples {sample_indices[0]}..{sample_indices[-1]}"
    )
    noise = _make_noise(adapter, fixed_noise_seed)
    runs: list[dict[str, dict[int, torch.Tensor]]] = []
    for i, sample_idx in enumerate(sample_indices):
        batch = _get_calibration_batch(adapter, sample_idx)
        runs.append(
            _collect_activations(
                adapter,
                batch,
                targets,
                noise=noise,
                act_metric=act_metric,
            )
        )
        print(f"  input sample {i + 1}/{num_samples} (calib index={sample_idx})")
    return _compute_similarity_grid(
        runs,
        num_steps=num_steps,
        label="fixed noise / varying input",
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--calibration-data", type=Path)
    p.add_argument("--calibration-source", choices=("file", "synthetic"), required=True)
    p.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Calibration sample for the noise sweep (fixed input).",
    )
    p.add_argument(
        "--num-noise-trials",
        type=int,
        default=16,
        help="Noise trials for the fixed-input sweep.",
    )
    p.add_argument(
        "--noise-seed-base",
        type=int,
        default=0,
        help="Trial i in the noise sweep uses seed = base + i.",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of calibration inputs for the fixed-noise sweep.",
    )
    p.add_argument(
        "--sample-start",
        type=int,
        default=0,
        help="First calibration index for the fixed-noise sweep.",
    )
    p.add_argument(
        "--sample-fixed-noise-seed",
        type=int,
        default=0,
        help="Shared noise seed for the varying-input sweep.",
    )
    p.add_argument("--layer-regex")
    p.add_argument("--act-metric", choices=("max_abs", "l2"), default="l2")
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory; filenames are chosen by the script.",
    )
    p.add_argument(
        "--compare-sample-splits",
        action="store_true",
        help=(
            "Also compare cross-sample similarity for the first N vs last N "
            "calibration samples (N = --num-samples)."
        ),
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

    num_steps = adapter.dit_step_count(config)
    targets = dsa._resolve_dit_targets(model, config, layer_regex=args.layer_regex)
    batch = _get_calibration_batch(adapter, args.sample_index)

    metric_tag = args.act_metric.replace("_", "")
    noise_out = args.output_dir / f"dit_noise_similarity_by_channel_{metric_tag}_linear_x_step.png"
    sample_out = args.output_dir / f"dit_sample_similarity_by_channel_{metric_tag}_linear_x_step.png"

    try:
        noise_rows = _run_noise_sweep(
            adapter,
            targets,
            batch,
            num_steps=num_steps,
            num_trials=args.num_noise_trials,
            noise_seed_base=args.noise_seed_base,
            act_metric=args.act_metric,
            sample_index=args.sample_index,
        )
        _plot_linear_step_similarity(
            noise_rows,
            noise_out,
            num_steps=num_steps,
            suptitle=(
                f"Per-channel {args.act_metric} similarity across noise trials "
                f"(fixed input sample {args.sample_index}, "
                f"seeds {args.noise_seed_base}..{args.noise_seed_base + args.num_noise_trials - 1})"
            ),
        )

        sample_rows = _run_sample_sweep(
            adapter,
            targets,
            num_steps=num_steps,
            num_samples=args.num_samples,
            sample_start=args.sample_start,
            fixed_noise_seed=args.sample_fixed_noise_seed,
            act_metric=args.act_metric,
        )
        _plot_linear_step_similarity(
            sample_rows,
            sample_out,
            num_steps=num_steps,
            suptitle=(
                f"Per-channel {args.act_metric} similarity across input samples "
                f"(fixed noise seed {args.sample_fixed_noise_seed}, "
                f"samples {args.sample_start}..{args.sample_start + args.num_samples - 1})"
            ),
        )

        if args.compare_sample_splits:
            num_calib = _calibration_size(
                args.calibration_source,
                args.calibration_data,
                num_samples=args.num_samples,
            )
            last_start = num_calib - args.num_samples
            if last_start < args.num_samples:
                raise SystemExit(
                    f"--compare-sample-splits needs at least {2 * args.num_samples} "
                    f"calibration samples; got {num_calib}."
                )
            first_indices = list(range(args.num_samples))
            last_indices = list(range(last_start, num_calib))
            print(
                f"\n[optional] Sample split compare: first {args.num_samples} "
                f"(0..{first_indices[-1]}) vs last {args.num_samples} "
                f"({last_indices[0]}..{last_indices[-1]}) of {num_calib}"
            )
            first_rows = _run_sample_sweep(
                adapter,
                targets,
                num_steps=num_steps,
                num_samples=args.num_samples,
                sample_start=first_indices[0],
                fixed_noise_seed=args.sample_fixed_noise_seed,
                act_metric=args.act_metric,
            )
            last_rows = _run_sample_sweep(
                adapter,
                targets,
                num_steps=num_steps,
                num_samples=args.num_samples,
                sample_start=last_indices[0],
                fixed_noise_seed=args.sample_fixed_noise_seed,
                act_metric=args.act_metric,
            )
            for metric in ("pearson", "spearman", "cosine"):
                _print_split_compare_stats(
                    first_rows,
                    last_rows,
                    first_indices=first_indices,
                    last_indices=last_indices,
                    metric=metric,
                )
            split_out = args.output_dir / "dit_sample_split_pearson_compare.png"
            _plot_sample_split_compare(
                first_rows,
                last_rows,
                split_out,
                num_steps=num_steps,
                num_samples=args.num_samples,
                first_indices=first_indices,
                last_indices=last_indices,
                num_calib=num_calib,
                fixed_noise_seed=args.sample_fixed_noise_seed,
            )
    except ImportError as e:
        raise SystemExit("Install matplotlib (uv sync --group dev) for plots.") from e
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
