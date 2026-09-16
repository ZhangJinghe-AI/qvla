#!/usr/bin/env python
"""Per-Euler-step Fisher sensitivity for DiT linear layers (fixed calibration input).

Measures whether per-channel Fisher changes across denoise steps and how large
the change is. Use ``--fixed-noise-seed`` so only the schedule / x_t path differs.

Example::

    CUDA_VISIBLE_DEVICES=2 uv run python tools/analyze_fisher_dit_step.py \\
        --checkpoint /data/share/pi05_libero_finetuned_v044 \\
        --calibration-source file \\
        --calibration-data ../calibration_data/libero_object_16_7.npz \\
        --sample-index 0 \\
        --fixed-noise-seed 0 \\
        --output-dir tools/img/fisher/dit_step
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
from qvla.adapters.pi05.step_hook import patched_one_step  # noqa: E402
from qvla.build.fisher import InputGradFisherCollector, resolve_fisher_action_dim  # noqa: E402
from qvla.config import QVLAConfig  # noqa: E402
from qvla.runtime import list_target_modules  # noqa: E402


@dataclass
class LayerStepSummary:
    name: str
    layer_idx: int
    kind: str
    steps: list[int]
    mean_per_step: list[float]
    spearman_offdiag: float
    step0_last_spearman: float
    mean_fisher_cv: float
    channel_swing_mean: float


def _resolve_dit_targets(model, config: QVLAConfig):
    targets = [(n, s, m) for n, s, m in list_target_modules(model, config) if s == "dit"]
    if not targets:
        raise SystemExit("No DiT target layers matched.")
    return sorted(targets, key=lambda t: t[0])


def _get_batch(adapter, sample_index: int) -> dict:
    batches = list(adapter.iter_calibration_batches(sample_index + 1))
    if sample_index >= len(batches):
        raise SystemExit(f"sample-index {sample_index} out of range ({len(batches)} samples).")
    return batches[sample_index]


def _make_noise(adapter, seed: int) -> torch.Tensor:
    sched = adapter.engine.entry.scheduler
    cfg = sched.cfg
    gen = torch.Generator(device=sched.device)
    gen.manual_seed(seed)
    return torch.randn(
        1, cfg.chunk_size, cfg.max_action_dim,
        generator=gen, device=sched.device, dtype=sched.params_dtype,
    )


def _differentiable_forward(adapter, sched, batch: dict, *, noise: torch.Tensor | None):
    from dataclasses import replace

    from qvla.adapters.pi05.differentiable_forward import differentiable_step

    adapter._ensure_processor()
    request = build_pi05_request(adapter._processor, batch, state_dim=adapter.cfg.state_dim)
    request.pixel_values = request.pixel_values.detach()
    if noise is not None:
        request = replace(request, noise=noise)
    return differentiable_step(sched, request)


def _collect_fisher_per_step(
    adapter,
    targets,
    batch: dict,
    *,
    num_dit_steps: int,
    noise: torch.Tensor | None,
    fisher_method: str = "exact",
    hutchinson_probes: int = 8,
) -> dict[str, dict[int, torch.Tensor]]:
    from qvla.adapters.pi05.differentiable_forward import (
        differentiable_inference_context,
        force_eager_runners,
        reset_differentiable_state,
    )
    from qvla.runtime.step_context import reset_step_counters

    sched = adapter.engine.entry.scheduler
    model = adapter.engine.entry.model
    force_eager_runners(sched)
    runner = sched.expert_runner

    with differentiable_inference_context(sched):
        with InputGradFisherCollector(
            targets, num_dit_steps, action_dim=resolve_fisher_action_dim(adapter)
        ) as fc:
            fc.begin_sample()
            reset_differentiable_state(sched)
            reset_step_counters(model)
            model.zero_grad(set_to_none=True)

            def _step_cb(step, _fc=fc):
                _fc.set_current_step(None if step is None else int(step))

            with patched_one_step(runner, _step_cb):
                fc.set_current_step(None)
                actions = _differentiable_forward(adapter, sched, batch, noise=noise)
            fc.compute_jacobian_sensitivity(
                actions,
                model=model,
                method=fisher_method,
                hutchinson_probes=hutchinson_probes,
            )
            results = fc.get_results()

    out: dict[str, dict[int, torch.Tensor]] = {}
    for name, result in results.items():
        per_step = {
            int(step): vec.detach().cpu()
            for step, vec in result.sensitivity_per_step().items()
            if step is not None
        }
        if per_step:
            out[name] = per_step
    return out


def _layer_summary(name: str, per_step: dict[int, torch.Tensor]) -> LayerStepSummary:
    steps = sorted(per_step)
    vectors = [per_step[s].float() for s in steps]
    stacked = torch.stack(vectors, dim=0)
    means = stacked.mean(dim=1)
    mean_vals = [float(m.item()) for m in means]
    mu = float(means.mean().item())
    mean_fisher_cv = float(means.std(unbiased=False).item() / max(abs(mu), 1e-12))
    channel_swing = (stacked.max(dim=0).values - stacked.min(dim=0).values)
    channel_denom = stacked.mean(dim=0).clamp_min(1e-12)
    channel_swing_mean = float((channel_swing / channel_denom).mean().item())

    spearman = dsa._step_similarity_matrix(vectors, metric="spearman")
    off = spearman[~torch.eye(len(steps), dtype=bool)]
    return LayerStepSummary(
        name=name,
        layer_idx=dsa._layer_idx(name),
        kind=dsa._layer_kind(name),
        steps=steps,
        mean_per_step=mean_vals,
        spearman_offdiag=float(off.mean().item()) if off.numel() else 1.0,
        step0_last_spearman=float(spearman[0, -1].item()) if len(steps) > 1 else 1.0,
        mean_fisher_cv=mean_fisher_cv,
        channel_swing_mean=channel_swing_mean,
    )


def _topk_overlap_matrix(vectors: list[torch.Tensor], k: int) -> torch.Tensor:
    n = len(vectors)
    mat = torch.eye(n, dtype=torch.float64)
    for i in range(n):
        for j in range(i + 1, n):
            v = dsa._topk_overlap(vectors[i], vectors[j], k)
            mat[i, j] = v
            mat[j, i] = v
    return mat


def _topk_ref_coverage(vectors: list[torch.Tensor], k: int, *, ref: int = 0) -> list[float]:
    k_eff = min(k, vectors[0].numel())
    ref_idx = set(vectors[ref].topk(k_eff).indices.tolist())
    out: list[float] = []
    for v in vectors:
        cur = set(v.topk(k_eff).indices.tolist())
        out.append(len(ref_idx & cur) / k_eff)
    return out


def _plot_topk_summary(
    path: Path,
    fisher_map: dict[str, dict[int, torch.Tensor]],
    summaries: list[LayerStepSummary],
    *,
    num_steps: int,
    top_k: int,
) -> None:
    """Layer×step retention of step-0 top-k, plus mean step×step top-k overlap."""
    import matplotlib.pyplot as plt
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(summaries, key=lambda s: (s.layer_idx, s.kind))
    steps = list(range(num_steps))

    retention_rows: list[list[float]] = []
    ylabels: list[str] = []
    step_mats: list[np.ndarray] = []

    for s in ordered:
        per_step = fisher_map[s.name]
        if len(per_step) != num_steps:
            continue
        vectors = [per_step[t].float() for t in steps]
        retention_rows.append(_topk_ref_coverage(vectors, top_k))
        ylabels.append(f"L{s.layer_idx}.{s.kind.split('_')[0]}")
        step_mats.append(_topk_overlap_matrix(vectors, top_k).numpy())

    fig, (ax_ret, ax_pair) = plt.subplots(1, 2, figsize=(13, max(5, 0.18 * len(retention_rows) + 2)))

    if retention_rows:
        data = np.array(retention_rows, dtype=np.float64)
        im = ax_ret.imshow(data, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn", interpolation="nearest")
        ax_ret.set_xlabel("Denoise step")
        ax_ret.set_ylabel("Layer")
        ax_ret.set_title(f"Step-0 top-{top_k} retention\n|topk(step0) ∩ topk(s)| / k")
        ax_ret.set_xticks(steps)
        y_stride = max(1, len(ylabels) // 14)
        ax_ret.set_yticks(range(0, len(ylabels), y_stride))
        ax_ret.set_yticklabels([ylabels[i] for i in range(0, len(ylabels), y_stride)], fontsize=6)
        fig.colorbar(im, ax=ax_ret, fraction=0.03, pad=0.02)

    if step_mats:
        mean_pair = np.mean(np.stack(step_mats, axis=0), axis=0)
        im2 = ax_pair.imshow(mean_pair, vmin=0, vmax=1, cmap="RdYlGn", interpolation="nearest")
        ax_pair.set_xticks(steps)
        ax_pair.set_yticks(steps)
        ax_pair.set_xlabel("Step")
        ax_pair.set_ylabel("Step")
        ax_pair.set_title(f"Mean top-{top_k} overlap across layers\n|topk(i) ∩ topk(j)| / k")
        fig.colorbar(im2, ax=ax_pair, fraction=0.046, pad=0.04)

    fig.suptitle(f"DiT Fisher top-{top_k} channel coverage across steps", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def _plot_summary(path: Path, summaries: list[LayerStepSummary], *, num_steps: int) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(summaries, key=lambda s: (s.layer_idx, s.kind))
    steps = list(range(num_steps))

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.28)

    ax_a = fig.add_subplot(gs[0, 0])
    agg = np.zeros(num_steps, dtype=np.float64)
    counts = np.zeros(num_steps, dtype=np.int64)
    for s in ordered:
        for step, mean_f in zip(s.steps, s.mean_per_step, strict=True):
            agg[step] += mean_f
            counts[step] += 1
    agg /= np.maximum(counts, 1)
    ax_a.plot(steps, agg, marker="o", ms=4, lw=1.8, color="#4C72B0")
    ax_a.set_xlabel("Denoise step")
    ax_a.set_ylabel("Mean Fisher (avg over layers)")
    ax_a.set_title("Aggregate Fisher scale vs step")
    ax_a.grid(True, alpha=0.25)

    ax_b = fig.add_subplot(gs[0, 1])
    mat, labels = [], []
    for s in ordered:
        if len(s.mean_per_step) != num_steps:
            continue
        mat.append(s.mean_per_step)
        labels.append(f"L{s.layer_idx}.{s.kind.split('_')[0]}")
    if mat:
        data = np.array(mat, dtype=np.float64)
        normed = (data - data.min(1, keepdims=True)) / np.maximum(
            data.max(1, keepdims=True) - data.min(1, keepdims=True), 1e-12,
        )
        im = ax_b.imshow(normed, aspect="auto", cmap="viridis", interpolation="nearest")
        ax_b.set_xlabel("Denoise step")
        ax_b.set_ylabel("Layer (row-normalized mean Fisher)")
        ax_b.set_title("Per-layer step pattern")
        y_stride = max(1, len(labels) // 12)
        ax_b.set_yticks(range(0, len(labels), y_stride))
        ax_b.set_yticklabels([labels[i] for i in range(0, len(labels), y_stride)], fontsize=6)
        fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.04)

    ax_c = fig.add_subplot(gs[1, 0])
    metrics = ["spearman_offdiag", "step0_last_spearman"]
    titles = ["Step×step Spearman (off-diag mean)", "Step 0 vs last Spearman"]
    mat_m = np.array([[getattr(s, m) for m in metrics] for s in ordered], dtype=np.float64)
    im = ax_c.imshow(mat_m, aspect="auto", vmin=-1, vmax=1, cmap="RdYlGn", interpolation="nearest")
    ax_c.set_xticks(range(len(metrics)), titles, fontsize=9)
    y_stride = max(1, len(ordered) // 12)
    ax_c.set_yticks(range(0, len(ordered), y_stride))
    ax_c.set_yticklabels(
        [f"L{ordered[i].layer_idx}.{ordered[i].kind.split('_')[0]}" for i in range(0, len(ordered), y_stride)],
        fontsize=6,
    )
    ax_c.set_title("Step stability (1 = identical across steps)")
    fig.colorbar(im, ax=ax_c, fraction=0.046, pad=0.04)

    ax_d = fig.add_subplot(gs[1, 1])
    x = np.arange(len(ordered))
    cv = [s.mean_fisher_cv for s in ordered]
    swing = [s.channel_swing_mean for s in ordered]
    w = 0.4
    ax_d.bar(x - w / 2, cv, width=w, label="mean Fisher CV across steps", color="#DD8452")
    ax_d.bar(x + w / 2, swing, width=w, label="mean channel swing (max-min)/mean", color="#4C72B0")
    ax_d.set_xlabel("Layer index (sorted)")
    ax_d.set_ylabel("Relative change magnitude")
    ax_d.set_title("How much Fisher moves across steps")
    ax_d.legend(fontsize=8)
    ax_d.grid(True, axis="y", alpha=0.25)

    fig.suptitle("DiT Fisher sensitivity vs denoise step (fixed input)", y=1.01)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def _plot_layer_detail(
    name: str, per_step: dict[int, torch.Tensor], path: Path, *, num_steps: int, top_k: int,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    steps = sorted(per_step)
    vectors = [per_step[s].float() for s in steps]
    stacked = torch.stack(vectors, dim=0).numpy()
    pos = stacked[stacked > 0]
    use_log = pos.size and float(pos.max()) / max(float(pos.min()), 1e-12) > 100

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_sim, ax_topk, ax_dist) = plt.subplots(1, 3, figsize=(15, 4.5))

    spearman = dsa._step_similarity_matrix(vectors, metric="spearman").numpy()
    im = ax_sim.imshow(spearman, vmin=-1, vmax=1, cmap="RdBu_r", interpolation="nearest")
    ax_sim.set_xticks(steps)
    ax_sim.set_yticks(steps)
    ax_sim.set_xlabel("Step")
    ax_sim.set_ylabel("Step")
    ax_sim.set_title("Spearman (channel curves)")
    fig.colorbar(im, ax=ax_sim, fraction=0.046, pad=0.04)

    topk_mat = _topk_overlap_matrix(vectors, top_k).numpy()
    im2 = ax_topk.imshow(topk_mat, vmin=0, vmax=1, cmap="RdYlGn", interpolation="nearest")
    ax_topk.set_xticks(steps)
    ax_topk.set_yticks(steps)
    ax_topk.set_xlabel("Step")
    ax_topk.set_ylabel("Step")
    ax_topk.set_title(f"Top-{top_k} overlap\n|topk(i) ∩ topk(j)| / k")
    fig.colorbar(im2, ax=ax_topk, fraction=0.046, pad=0.04)

    coverage = _topk_ref_coverage(vectors, top_k)
    ax_dist.plot(steps, coverage, marker="o", ms=4, lw=1.8, color="#4C72B0", label=f"vs step-0 top-{top_k}")
    p50 = np.percentile(stacked, 50, axis=1)
    p90 = np.percentile(stacked, 90, axis=1)
    pmax = stacked.max(axis=1)
    ax_dist2 = ax_dist.twinx()
    ax_dist2.plot(steps, p50, color="#DD8452", lw=1.2, alpha=0.8, label="p50")
    ax_dist2.plot(steps, p90, color="#55A868", lw=1.0, ls="--", alpha=0.8, label="p90")
    ax_dist2.plot(steps, pmax, color="#C44E52", lw=1.0, alpha=0.7, label="max")
    ax_dist.set_xlabel("Denoise step")
    ax_dist.set_ylabel(f"Step-0 top-{top_k} retention")
    ax_dist.set_ylim(0, 1.05)
    ax_dist2.set_ylabel("Fisher sensitivity" + (" (log)" if use_log else ""))
    if use_log:
        ax_dist2.set_yscale("log")
    ax_dist.set_title("Top-k retention vs distribution")
    ax_dist.grid(True, alpha=0.25)
    lines1, labels1 = ax_dist.get_legend_handles_labels()
    lines2, labels2 = ax_dist2.get_legend_handles_labels()
    ax_dist.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="center left")

    fig.suptitle(dsa._short_name(name), fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--calibration-data", type=Path)
    p.add_argument("--calibration-source", choices=("file", "synthetic"), required=True)
    p.add_argument("--sample-index", type=int, default=0, help="Fixed calibration sample (default: 0).")
    p.add_argument("--fixed-noise-seed", type=int, default=0, help="Fixed diffusion noise (default: 0).")
    p.add_argument("--top-k", type=int, default=32, help="Top-k channels for overlap/coverage (default: 32).")
    p.add_argument(
        "--fisher-method",
        choices=("exact", "hutchinson"),
        default="exact",
        help=(
            "Fisher estimator: exact (one backward per action dim) or "
            "hutchinson (random-projection approximation)."
        ),
    )
    p.add_argument(
        "--fisher-hutchinson-probes",
        type=int,
        default=8,
        help="Number of random projections when --fisher-method=hutchinson.",
    )
    p.add_argument("--output-dir", type=Path, default=_TOOLS / "img" / "fisher" / "dit_step")
    args = p.parse_args(argv)

    if args.calibration_source == "file" and args.calibration_data is None:
        p.error("--calibration-data is required when --calibration-source=file.")
    if args.fisher_hutchinson_probes < 1:
        p.error("--fisher-hutchinson-probes must be >= 1.")

    config = QVLAConfig.pi05_default()
    adapter_kwargs = {"checkpoint_path": args.checkpoint, "calibration_source": args.calibration_source}
    if args.calibration_source == "file":
        adapter_kwargs["calibration_data_path"] = args.calibration_data

    adapter = get_adapter("pi05", **adapter_kwargs)
    model = adapter.build_model()
    adapter.warmup_for_calibration(model)
    targets = _resolve_dit_targets(model, config)
    num_steps = adapter.dit_step_count(config)
    batch = _get_batch(adapter, args.sample_index)
    noise = _make_noise(adapter, args.fixed_noise_seed)

    print(
        f"Fisher per-step: sample={args.sample_index}, noise_seed={args.fixed_noise_seed}, "
        f"{len(targets)} DiT layer(s), {num_steps} steps, method={args.fisher_method}"
        + (
            f", probes={args.fisher_hutchinson_probes}"
            if args.fisher_method == "hutchinson"
            else ""
        )
    )
    fisher_map = _collect_fisher_per_step(
        adapter,
        targets,
        batch,
        num_dit_steps=num_steps,
        noise=noise,
        fisher_method=args.fisher_method,
        hutchinson_probes=args.fisher_hutchinson_probes,
    )

    summaries = [_layer_summary(name, fisher_map[name]) for name in sorted(fisher_map)]

    try:
        out = args.output_dir
        _plot_summary(out / "fisher_dit_step_summary.png", summaries, num_steps=num_steps)
        _plot_topk_summary(
            out / "fisher_dit_step_topk_coverage_summary.png",
            fisher_map,
            summaries,
            num_steps=num_steps,
            top_k=args.top_k,
        )
        for name in sorted(fisher_map):
            suffix = dsa._layer_file_suffix(name)
            _plot_layer_detail(
                name,
                fisher_map[name],
                out / f"fisher_dit_step_{suffix}.png",
                num_steps=num_steps,
                top_k=args.top_k,
            )
    except ImportError as e:
        raise SystemExit("Install matplotlib (uv sync --group dev) for plots.") from e
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
