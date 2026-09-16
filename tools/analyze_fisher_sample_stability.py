#!/usr/bin/env python
"""Fisher channel sensitivity stability across calibration samples (all quant layers).

For each calibration index, Fisher is computed on that sample alone. Reports
pairwise value/rank similarity and optional top-k channel overlap per layer.

Fix diffusion noise with ``--fixed-noise-seed`` so differences come from input,
not random noise (recommended).

Example::

    CUDA_VISIBLE_DEVICES=2 uv run python tools/analyze_fisher_sample_stability.py \\
        --checkpoint /data/share/pi05_libero_finetuned_v044 \\
        --calibration-source file \\
        --calibration-data ../calibration_data/libero_object_16_7.npz \\
        --num-samples 8 \\
        --fixed-noise-seed 0 \\
        --compare-pooled \\
        --output-dir tools/img/fisher/sample_stability
"""

from __future__ import annotations

import argparse
import csv
import re
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


def _get_calibration_batch(adapter, sample_index: int) -> dict:
    batches = list(adapter.iter_calibration_batches(sample_index + 1))
    if sample_index >= len(batches):
        raise SystemExit(
            f"sample index {sample_index} out of range; "
            f"calibration provides {len(batches)} sample(s)."
        )
    return batches[sample_index]


def _make_noise(adapter, seed: int) -> torch.Tensor:
    sched = adapter.engine.entry.scheduler
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


def _layer_file_stem(name: str, scope: str) -> str:
    m = re.search(r"layers\.(\d+)\.", name)
    if m:
        idx = int(m.group(1))
        for kind in ("qkv_proj", "o_proj", "gate_up_proj", "down_proj"):
            if name.endswith(kind):
                return f"{scope}_layer{idx:02d}_{kind}"
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", name)
    return f"{scope}_{safe}"


def _layer_short_label(name: str, scope: str) -> str:
    m = re.search(r"layers\.(\d+)\.", name)
    if m:
        idx = int(m.group(1))
        for kind in ("qkv_proj", "o_proj", "gate_up_proj", "down_proj"):
            if name.endswith(kind):
                short = kind.replace("_proj", "").replace("gate_up", "gu")
                return f"{scope[:3]}.L{idx:02d}.{short}"
    return name.rsplit(".", 1)[-1][:24]


def _resolve_targets(model, config: QVLAConfig):
    targets = list_target_modules(model, config)
    if not targets:
        raise SystemExit("No quant target layers matched.")
    return sorted(targets, key=lambda t: (t[1], t[0]))


def _aggregate_steps(per_step: dict[int | None, torch.Tensor]) -> torch.Tensor:
    if not per_step:
        raise RuntimeError("Empty Fisher result.")
    return torch.stack(list(per_step.values()), dim=0).mean(dim=0).cpu()


def _differentiable_forward(adapter, sched, batch: dict, *, noise: torch.Tensor | None):
    from dataclasses import replace

    from qvla.adapters.pi05.differentiable_forward import differentiable_step

    adapter._ensure_processor()
    request = build_pi05_request(adapter._processor, batch, state_dim=adapter.cfg.state_dim)
    request.pixel_values = request.pixel_values.detach()
    if noise is not None:
        request = replace(request, noise=noise)
    return differentiable_step(sched, request)


def _collect_fisher(
    adapter,
    targets,
    batches: list[dict],
    *,
    num_dit_steps: int,
    noise: torch.Tensor | None,
    fisher_method: str = "exact",
    hutchinson_probes: int = 8,
) -> dict[str, torch.Tensor]:
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
            for batch in batches:
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

    return {name: _aggregate_steps(r.sensitivity_per_step()) for name, r in results.items()}


def _topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int) -> float:
    k = min(k, a.numel())
    ia = set(a.topk(k).indices.tolist())
    ib = set(b.topk(k).indices.tolist())
    return len(ia & ib) / k


@dataclass
class LayerStabilityRow:
    layer_name: str
    scope: str
    short_label: str
    pearson: float
    spearman: float
    cosine: float
    topk_overlap: float
    pooled_pearson: float | None = None
    pooled_spearman: float | None = None


def _pairwise_stats(
    vectors: list[torch.Tensor],
    *,
    top_k: int,
) -> tuple[float, float, float, float]:
    pearsons: list[float] = []
    spearmans: list[float] = []
    cosines: list[float] = []
    topks: list[float] = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            a, b = vectors[i], vectors[j]
            pearsons.append(dsa._pearson(a, b))
            spearmans.append(dsa._spearman(a, b))
            cosines.append(dsa._cosine(a, b))
            topks.append(_topk_overlap(a, b, top_k))
    if not pearsons:
        return 1.0, 1.0, 1.0, 1.0
    n = len(pearsons)
    return (
        sum(pearsons) / n,
        sum(spearmans) / n,
        sum(cosines) / n,
        sum(topks) / n,
    )


def _plot_pairwise_heatmap(
    vectors: list[torch.Tensor],
    sample_indices: list[int],
    *,
    layer_name: str,
    output: Path,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    n = len(vectors)
    mat = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            val = dsa._spearman(vectors[i], vectors[j])
            mat[i, j] = mat[j, i] = val

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, vmin=-1.0, vmax=1.0, cmap="RdBu_r")
    labels = [str(i) for i in sample_indices]
    ax.set_xticks(range(n), labels)
    ax.set_yticks(range(n), labels)
    ax.set_xlabel("calibration sample index")
    ax.set_ylabel("calibration sample index")
    ax.set_title(f"{layer_name}\npairwise Spearman rho")
    fig.colorbar(im, ax=ax, fraction=0.046)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


def _write_summary_csv(rows: list[LayerStabilityRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "layer_name",
        "scope",
        "short_label",
        "pearson",
        "spearman",
        "cosine",
        "topk_overlap",
        "pooled_pearson",
        "pooled_spearman",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "layer_name": row.layer_name,
                    "scope": row.scope,
                    "short_label": row.short_label,
                    "pearson": f"{row.pearson:.6f}",
                    "spearman": f"{row.spearman:.6f}",
                    "cosine": f"{row.cosine:.6f}",
                    "topk_overlap": f"{row.topk_overlap:.6f}",
                    "pooled_pearson": "" if row.pooled_pearson is None else f"{row.pooled_pearson:.6f}",
                    "pooled_spearman": "" if row.pooled_spearman is None else f"{row.pooled_spearman:.6f}",
                }
            )
    print(f"Wrote {path}")


def _plot_summary_heatmap(rows: list[LayerStabilityRow], path: Path, *, top_k: int) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = ["pearson", "spearman", "cosine", "topk_overlap"]
    titles = ["Pearson r", "Spearman rho", "Cosine", f"Top-{top_k} overlap"]
    ylabels = [r.short_label for r in rows]
    mat = np.array([[getattr(r, m) for m in metrics] for r in rows], dtype=np.float64)

    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), max(4, 0.22 * len(rows))))
    if len(metrics) == 1:
        axes = [axes]
    for ax, title, col in zip(axes, titles, range(len(metrics)), strict=True):
        data = mat[:, col : col + 1]
        vmin, vmax = (-1.0, 1.0) if col < 3 else (0.0, 1.0)
        im = ax.imshow(data, aspect="auto", vmin=vmin, vmax=vmax, cmap="RdYlGn")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks(range(len(ylabels)), ylabels, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Fisher sample stability (pairwise mean across calibration samples)", y=1.02)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--calibration-data", type=Path)
    p.add_argument("--calibration-source", choices=("file", "synthetic"), required=True)
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--sample-start", type=int, default=0)
    p.add_argument(
        "--fixed-noise-seed",
        type=int,
        default=None,
        help="Use the same diffusion noise for every sample (recommended).",
    )
    p.add_argument("--top-k", type=int, default=32, help="Top-k channel overlap (default: 32).")
    p.add_argument(
        "--compare-pooled",
        action="store_true",
        help="Also run Fisher averaged over all samples and report per-sample vs pooled.",
    )
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
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(argv)

    if args.calibration_source == "file" and args.calibration_data is None:
        p.error("--calibration-data is required when --calibration-source=file.")
    if args.num_samples < 2:
        p.error("--num-samples must be >= 2.")
    if args.fisher_hutchinson_probes < 1:
        p.error("--fisher-hutchinson-probes must be >= 1.")

    sample_indices = list(range(args.sample_start, args.sample_start + args.num_samples))
    config = QVLAConfig.pi05_default()
    adapter_kwargs = {
        "checkpoint_path": args.checkpoint,
        "calibration_source": args.calibration_source,
    }
    if args.calibration_source == "file":
        adapter_kwargs["calibration_data_path"] = args.calibration_data

    adapter = get_adapter("pi05", **adapter_kwargs)
    model = adapter.build_model()
    adapter.warmup_for_calibration(model)
    targets = _resolve_targets(model, config)
    num_dit_steps = adapter.dit_step_count(config)
    print(f"Analyzing {len(targets)} quant target layer(s)...")
    noise = _make_noise(adapter, args.fixed_noise_seed) if args.fixed_noise_seed is not None else None
    if noise is not None:
        print(f"Fixed diffusion noise seed={args.fixed_noise_seed}")

    per_sample: dict[int, dict[str, torch.Tensor]] = {}
    for i, idx in enumerate(sample_indices):
        print(f"Fisher sample {i + 1}/{args.num_samples} (calib index={idx})")
        per_sample[idx] = _collect_fisher(
            adapter,
            targets,
            [_get_calibration_batch(adapter, idx)],
            num_dit_steps=num_dit_steps,
            noise=noise,
            fisher_method=args.fisher_method,
            hutchinson_probes=args.fisher_hutchinson_probes,
        )

    pooled: dict[str, torch.Tensor] | None = None
    if args.compare_pooled:
        print(f"Pooled Fisher over samples {sample_indices[0]}..{sample_indices[-1]}")
        batches = [_get_calibration_batch(adapter, idx) for idx in sample_indices]
        pooled = _collect_fisher(
            adapter,
            targets,
            batches,
            num_dit_steps=num_dit_steps,
            noise=noise,
            fisher_method=args.fisher_method,
            hutchinson_probes=args.fisher_hutchinson_probes,
        )

    print(f"\n=== Fisher sample stability ({len(targets)} layer(s)) ===")
    summary_rows: list[LayerStabilityRow] = []

    for name, scope, _mod in targets:
        vectors = [per_sample[idx][name] for idx in sample_indices]
        mean_p, mean_sp, mean_cos, mean_topk = _pairwise_stats(vectors, top_k=args.top_k)
        row = LayerStabilityRow(
            layer_name=name,
            scope=scope,
            short_label=_layer_short_label(name, scope),
            pearson=mean_p,
            spearman=mean_sp,
            cosine=mean_cos,
            topk_overlap=mean_topk,
        )
        print(f"\n{name}")
        print(
            f"  pairwise mean  Pearson={mean_p:+.4f}  Spearman={mean_sp:+.4f}  "
            f"cosine={mean_cos:+.4f}  top-{args.top_k} overlap={mean_topk:.3f}"
        )
        if pooled is not None:
            pv = pooled[name]
            ps = [dsa._pearson(v, pv) for v in vectors]
            ss = [dsa._spearman(v, pv) for v in vectors]
            row.pooled_pearson = sum(ps) / len(ps)
            row.pooled_spearman = sum(ss) / len(ss)
            print(
                f"  vs pooled      Pearson mean={row.pooled_pearson:+.4f}  "
                f"Spearman mean={row.pooled_spearman:+.4f}"
            )
        summary_rows.append(row)

    try:
        _write_summary_csv(summary_rows, args.output_dir / "fisher_sample_stability.csv")
        _plot_summary_heatmap(
            summary_rows,
            args.output_dir / "fisher_sample_stability_summary.png",
            top_k=args.top_k,
        )
        for name, scope, _mod in targets:
            vectors = [per_sample[idx][name] for idx in sample_indices]
            _plot_pairwise_heatmap(
                vectors,
                sample_indices,
                layer_name=name,
                output=args.output_dir / f"{_layer_file_stem(name, scope)}_sample_stability.png",
            )
    except ImportError as e:
        raise SystemExit("Install matplotlib (uv sync --group dev) for plots.") from e
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
