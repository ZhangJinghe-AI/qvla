#!/usr/bin/env python
"""Per-Euler-step activation distributions for pi0.5 DiT (expert_stack) layers.

Collects pre-linear input activations separately for each denoising step and
reports how channel statistics evolve across the schedule.

Outputs land flat under ``tools/img/`` with descriptive names::

    dit_step_act_stats.csv              per-layer per-step statistics
    dit_step_act_summary.png            scale / heatmap / outlier / stability overview
    dit_step_act_channels_layer0_qkv_proj.png           per-channel curves (single-layer)
    dit_step_act_channels_step_similarity_layer0_qkv_proj.png  step×step Pearson/Spearman

Example (v044 finetuned + LIBERO calibration)::

    uv run python tools/analyze_dit_step_activations.py \\
        --checkpoint /data/share/pi05_libero_finetuned_v044 \\
        --calibration-data ../calibration_data/libero_object_16_7.npz \\
        --num-samples 8

Example (single layer, step-wise channel curves)::

    uv run python tools/analyze_dit_step_activations.py \\
        --checkpoint /data/share/pi05_libero_finetuned_v044 \\
        --calibration-data ../calibration_data/libero_object_16_7.npz \\
        --layer-regex 'expert_stack\\.layers\\.0\\.qkv_proj$' \\
        --channel-curves
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch

_ROOT = Path(__file__).resolve().parent.parent
_IMG_DIR = Path(__file__).resolve().parent / "img"
sys.path.insert(0, str(_ROOT / "src"))

from qvla.adapters import get_adapter  # noqa: E402
from qvla.config import QVLAConfig  # noqa: E402
from qvla.runtime import list_target_modules  # noqa: E402

ActMetric = Literal["max_abs", "l2"]
LayerKind = Literal["qkv_proj", "o_proj", "gate_up_proj", "down_proj", "other"]


@dataclass
class StepChannelStats:
    """Summary of per-channel activation scores at one Euler step."""

    step: int
    n_tokens: int
    mean: float
    median: float
    std: float
    p99: float
    max_val: float
    max_ch: int
    max_min_ratio: float


@dataclass
class LayerStepRecord:
    name: str
    layer_idx: int
    kind: LayerKind
    in_features: int
    per_step: list[StepChannelStats] = field(default_factory=list)

    @property
    def mean_curve(self) -> list[float]:
        return [s.mean for s in self.per_step]

    @property
    def ratio_curve(self) -> list[float]:
        return [s.max_min_ratio for s in self.per_step]


class _PerStepAccumulator:
    """Streaming per-channel stats for one (layer, step) pair."""

    __slots__ = ("metric", "in_features", "n_tokens", "sum_sq", "max_abs")

    def __init__(self, metric: ActMetric, in_features: int) -> None:
        self.metric = metric
        self.in_features = in_features
        self.n_tokens = 0
        self.sum_sq = torch.zeros(in_features, dtype=torch.float64)
        self.max_abs = torch.zeros(in_features, dtype=torch.float32)

    def update(self, x: torch.Tensor) -> None:
        flat = x.reshape(-1, self.in_features).detach().to(device="cpu", dtype=torch.float32)
        if flat.shape[0] == 0:
            return
        self.n_tokens += flat.shape[0]
        self.sum_sq += (flat.double() * flat.double()).sum(dim=0)
        self.max_abs = torch.maximum(self.max_abs, flat.abs().amax(dim=0))

    def channel_scores(self) -> torch.Tensor:
        if self.n_tokens == 0:
            raise RuntimeError("No tokens accumulated.")
        if self.metric == "max_abs":
            return self.max_abs
        return self.sum_sq.sqrt().to(torch.float32)

    def summarize(self, step: int, *, eps: float = 1e-12) -> StepChannelStats:
        vals = self.channel_scores()
        vmax = float(vals.max().item())
        vmin = max(float(vals.min().item()), eps)
        return StepChannelStats(
            step=step,
            n_tokens=self.n_tokens,
            mean=float(vals.mean().item()),
            median=float(vals.median().item()),
            std=float(vals.std(unbiased=False).item()),
            p99=float(torch.quantile(vals, 0.99).item()),
            max_val=vmax,
            max_ch=int(vals.argmax().item()),
            max_min_ratio=vmax / vmin,
        )


def _layer_kind(name: str) -> LayerKind:
    for kind in ("qkv_proj", "o_proj", "gate_up_proj", "down_proj"):
        if name.endswith(kind):
            return kind  # type: ignore[return-value]
    return "other"


def _layer_idx(name: str) -> int:
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else -1


def _short_name(full_name: str) -> str:
    parts = full_name.split(".")
    return ".".join(parts[-4:]) if len(parts) > 4 else full_name


def _layer_file_suffix(name: str) -> str:
    """``expert_stack.layers.0.qkv_proj`` -> ``layer0_qkv_proj``."""
    idx = _layer_idx(name)
    kind = _layer_kind(name)
    return f"layer{idx}_{kind}" if idx >= 0 else kind


def _resolve_dit_targets(
    model: torch.nn.Module,
    config: QVLAConfig,
    *,
    layer_regex: str | None,
) -> list[tuple[str, str, torch.nn.Module]]:
    targets = [(n, s, m) for n, s, m in list_target_modules(model, config) if s == "dit"]
    if layer_regex:
        pattern = re.compile(layer_regex)
        targets = [(n, s, m) for n, s, m in targets if pattern.search(n)]
    if not targets:
        raise SystemExit("No DiT layers matched; check --layer-regex.")
    return sorted(targets, key=lambda t: t[0])


def _collect_per_step_activations(
    adapter,
    model: torch.nn.Module,
    targets: list[tuple[str, str, torch.nn.Module]],
    *,
    num_samples: int,
    num_steps: int,
    act_metric: ActMetric,
) -> dict[str, dict[int, _PerStepAccumulator]]:
    """Return ``layer_name -> step -> accumulator``."""
    store: dict[str, dict[int, _PerStepAccumulator]] = {
        name: {} for name, _, _ in targets
    }
    in_features = {name: mod.weight.shape[1] for name, _, mod in targets}
    current_step: list[int | None] = [None]
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(layer_name: str):
        def hook(_mod, inputs):
            step = current_step[0]
            if step is None or not inputs or not torch.is_tensor(inputs[0]):
                return
            step_dict = store[layer_name]
            if step not in step_dict:
                step_dict[step] = _PerStepAccumulator(act_metric, in_features[layer_name])
            step_dict[step].update(inputs[0])

        return hook

    for name, _, mod in targets:
        handles.append(mod.register_forward_pre_hook(make_hook(name)))

    def step_cb(step: int | None) -> None:
        current_step[0] = step

    try:
        for i, batch in enumerate(adapter.iter_calibration_batches(num_samples)):
            adapter.forward_for_calibration(model, batch, step_callback=step_cb, sample_index=i)
    finally:
        for handle in handles:
            handle.remove()

    missing = [n for n, _, _ in targets if not store[n]]
    if missing:
        raise RuntimeError(f"No activations captured for: {missing[:5]}")

    # Ensure every layer has entries for observed steps (warn on gaps).
    for name, step_dict in store.items():
        observed = sorted(step_dict)
        if observed and (observed[0] != 0 or observed[-1] != num_steps - 1):
            print(
                f"  warning: {name} observed steps {observed[0]}..{observed[-1]} "
                f"(expected 0..{num_steps - 1})"
            )
    return store


def _build_records(
    store: dict[str, dict[int, _PerStepAccumulator]],
    *,
    num_steps: int,
) -> list[LayerStepRecord]:
    records: list[LayerStepRecord] = []
    for name in sorted(store):
        step_dict = store[name]
        in_f = next(iter(step_dict.values())).in_features
        rec = LayerStepRecord(
            name=name,
            layer_idx=_layer_idx(name),
            kind=_layer_kind(name),
            in_features=in_f,
        )
        for step in range(num_steps):
            if step not in step_dict:
                continue
            rec.per_step.append(step_dict[step].summarize(step))
        records.append(rec)
    return records


def _topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int) -> float:
    k = min(k, a.numel())
    sa = set(a.topk(k).indices.tolist())
    sb = set(b.topk(k).indices.tolist())
    return len(sa & sb) / k


def _channel_stability(
    store: dict[str, dict[int, _PerStepAccumulator]],
    *,
    num_steps: int,
    top_k: int,
) -> dict[str, float]:
    """Mean pairwise top-k channel overlap between first and last step."""
    out: dict[str, float] = {}
    for name, step_dict in store.items():
        if 0 not in step_dict or (num_steps - 1) not in step_dict:
            continue
        a0 = step_dict[0].channel_scores()
        a1 = step_dict[num_steps - 1].channel_scores()
        out[name] = _topk_overlap(a0, a1, top_k)
    return out


def _relative_change(first: float, last: float) -> float:
    denom = max(abs(first), 1e-12)
    return (last - first) / denom


def _write_csv(path: Path, records: list[LayerStepRecord], *, act_metric: ActMetric) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "short_name",
                "layer_idx",
                "kind",
                "in_features",
                "step",
                "n_tokens",
                "mean",
                "median",
                "std",
                "p99",
                "max_val",
                "max_ch",
                "max_min_ratio",
                "act_metric",
            ],
        )
        w.writeheader()
        for rec in records:
            for s in rec.per_step:
                w.writerow(
                    {
                        "name": rec.name,
                        "short_name": _short_name(rec.name),
                        "layer_idx": rec.layer_idx,
                        "kind": rec.kind,
                        "in_features": rec.in_features,
                        "step": s.step,
                        "n_tokens": s.n_tokens,
                        "mean": f"{s.mean:.6f}",
                        "median": f"{s.median:.6f}",
                        "std": f"{s.std:.6f}",
                        "p99": f"{s.p99:.6f}",
                        "max_val": f"{s.max_val:.6f}",
                        "max_ch": s.max_ch,
                        "max_min_ratio": f"{s.max_min_ratio:.6f}",
                        "act_metric": act_metric,
                    }
                )
    print(f"Wrote CSV to {path}")


def _aggregate_by_kind(records: list[LayerStepRecord], num_steps: int) -> dict[LayerKind, list[float]]:
    """Mean channel score averaged across layers of each kind, per step."""
    by_kind: dict[LayerKind, list[list[float]]] = {}
    for rec in records:
        curve = rec.mean_curve
        if len(curve) != num_steps:
            continue
        by_kind.setdefault(rec.kind, []).append(curve)
    return {k: torch.tensor(v).mean(dim=0).tolist() for k, v in by_kind.items()}


def _print_conclusions(
    records: list[LayerStepRecord],
    *,
    num_steps: int,
    stability: dict[str, float],
    top_k: int,
    act_metric: ActMetric,
) -> None:
    print(f"\n=== DiT step activation analysis ({act_metric}) ===")
    print(f"Layers: {len(records)}   Steps: {num_steps}")

    # Per-kind step curves (mean channel score).
    kind_curves = _aggregate_by_kind(records, num_steps)
    print("\nMean channel score by layer kind (avg over layers):")
    header = "step " + "".join(f"{s:>8d}" for s in range(num_steps))
    print(header)
    for kind in ("qkv_proj", "o_proj", "gate_up_proj", "down_proj"):
        if kind not in kind_curves:
            continue
        vals = kind_curves[kind]
        row = f"{kind:12s}" + "".join(f"{v:8.2f}" for v in vals)
        print(row)
        rel = _relative_change(vals[0], vals[-1])
        trend = "↑ grows" if rel > 0.05 else "↓ shrinks" if rel < -0.05 else "~ flat"
        print(f"             step0→step{num_steps - 1}: {rel:+.1%}  ({trend})")

    # Max/min ratio trend by kind.
    print("\nMax/min ratio by layer kind (avg over layers):")
    ratio_by_kind: dict[LayerKind, list[list[float]]] = {}
    for rec in records:
        if len(rec.ratio_curve) != num_steps:
            continue
        ratio_by_kind.setdefault(rec.kind, []).append(rec.ratio_curve)
    ratio_avg_by_kind: dict[LayerKind, list[float]] = {}
    for kind in ("qkv_proj", "o_proj", "gate_up_proj", "down_proj"):
        if kind not in ratio_by_kind:
            continue
        avg = torch.tensor(ratio_by_kind[kind]).mean(dim=0).tolist()
        ratio_avg_by_kind[kind] = avg
        rel = _relative_change(avg[0], avg[-1])
        print(
            f"  {kind:12s}  step0={avg[0]:.1f}x  step{num_steps - 1}={avg[-1]:.1f}x  "
            f"Δ={rel:+.1%}"
        )

    # Outlier channel stability.
    st_t: torch.Tensor | None = None
    if stability:
        st_vals = list(stability.values())
        st_t = torch.tensor(st_vals)
        print(f"\nTop-{top_k} outlier channel overlap (step0 vs step{num_steps - 1}):")
        print(f"  mean={st_t.mean():.2f}  median={st_t.median():.2f}  min={st_t.min():.2f}")
        by_kind_stab: dict[LayerKind, list[float]] = {}
        for rec in records:
            if rec.name in stability:
                by_kind_stab.setdefault(rec.kind, []).append(stability[rec.name])
        for kind in ("qkv_proj", "o_proj", "gate_up_proj", "down_proj"):
            if kind in by_kind_stab:
                m = torch.tensor(by_kind_stab[kind]).mean().item()
                print(f"  {kind:12s}  overlap={m:.2f}")

    # Layers with largest step-to-step swing.
    swings: list[tuple[float, str, LayerKind]] = []
    for rec in records:
        if len(rec.mean_curve) < 2:
            continue
        swing = max(rec.mean_curve) - min(rec.mean_curve)
        swings.append((swing, rec.name, rec.kind))
    swings.sort(reverse=True)
    print("\nLargest mean-score swing across steps (top 5):")
    for swing, name, kind in swings[:5]:
        print(f"  Δmean={swing:.2f}  [{kind}]  {_short_name(name)}")

    print("\n--- Quick takeaways ---")
    if kind_curves:
        down = kind_curves.get("down_proj")
        qkv = kind_curves.get("qkv_proj")
        if down and qkv:
            down_swing = abs(_relative_change(down[0], down[-1]))
            qkv_swing = abs(_relative_change(qkv[0], qkv[-1]))
            if down_swing > qkv_swing * 1.5:
                print(
                    f"  • down_proj mean activation shrinks most across steps "
                    f"({down_swing:.0%} vs qkv {qkv_swing:.0%})."
                )
            elif qkv_swing > down_swing * 1.5:
                print(
                    f"  • qkv_proj mean activation changes more than down_proj "
                    f"({qkv_swing:.0%} vs down {down_swing:.0%})."
                )
            else:
                print("  • Mean activation scale is fairly stable for qkv/o_proj across steps.")
        dr = ratio_avg_by_kind.get("down_proj")
        if dr:
            if dr[-1] > dr[0] * 1.5:
                print("  • down_proj outlier severity grows toward late denoise steps.")
            elif dr[-1] < dr[0] * 0.67:
                print("  • down_proj outlier max/min ratio drops toward late steps.")
        gr = ratio_avg_by_kind.get("gate_up_proj")
        if gr and gr[-1] > gr[0] * 2:
            print("  • gate_up_proj outlier severity increases sharply by the last step.")
        if st_t is not None and st_t.mean() > 0.5:
            print("  • Outlier channels are fairly stable across the schedule.")
        elif st_t is not None and st_t.mean() < 0.3:
            print("  • Outlier channels shift a lot between early and late steps.")


def _plot_step_analysis(
    records: list[LayerStepRecord],
    store: dict[str, dict[int, _PerStepAccumulator]],
    path: Path,
    *,
    num_steps: int,
    act_metric: ActMetric,
    stability: dict[str, float],
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    kind_colors = {
        "qkv_proj": "#1f77b4",
        "o_proj": "#ff7f0e",
        "gate_up_proj": "#2ca02c",
        "down_proj": "#d62728",
    }

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)

    # A: mean channel score vs step, by kind (avg over layers).
    ax_a = fig.add_subplot(gs[0, 0])
    kind_curves = _aggregate_by_kind(records, num_steps)
    steps = list(range(num_steps))
    for kind, color in kind_colors.items():
        if kind not in kind_curves:
            continue
        ax_a.plot(steps, kind_curves[kind], marker="o", ms=4, lw=1.8, label=kind, color=color)
    ax_a.set_xlabel("Denoise step")
    ax_a.set_ylabel(f"Mean per-channel {act_metric}")
    ax_a.set_title("Activation scale vs step (avg by layer kind)")
    ax_a.legend(fontsize=8)
    ax_a.grid(True, alpha=0.25)

    # B: heatmap — layers (y) x step (x), mean channel score.
    ax_b = fig.add_subplot(gs[0, 1])
    ordered = sorted(records, key=lambda r: (r.layer_idx, r.kind))
    mat = []
    ylabels = []
    for rec in ordered:
        if len(rec.per_step) != num_steps:
            continue
        mat.append([s.mean for s in rec.per_step])
        ylabels.append(f"L{rec.layer_idx}.{rec.kind.split('_')[0]}")
    if mat:
        data = np.array(mat, dtype=np.float64)
        # Normalize each row to [0,1] to highlight step pattern within layer.
        row_min = data.min(axis=1, keepdims=True)
        row_max = data.max(axis=1, keepdims=True)
        normed = (data - row_min) / np.maximum(row_max - row_min, 1e-12)
        im = ax_b.imshow(normed, aspect="auto", cmap="viridis", interpolation="nearest")
        ax_b.set_xlabel("Denoise step")
        ax_b.set_ylabel("Layer (row-normalized mean)")
        ax_b.set_title("Per-layer step pattern (row-normalized)")
        step_stride = max(1, num_steps // 10)
        ax_b.set_xticks(range(0, num_steps, step_stride))
        y_stride = max(1, len(ylabels) // 12)
        ax_b.set_yticks(range(0, len(ylabels), y_stride))
        ax_b.set_yticklabels([ylabels[i] for i in range(0, len(ylabels), y_stride)], fontsize=6)
        fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.04)

    # C: max/min ratio vs step by kind.
    ax_c = fig.add_subplot(gs[1, 0])
    ratio_by_kind: dict[LayerKind, list[list[float]]] = {}
    for rec in records:
        if len(rec.ratio_curve) != num_steps:
            continue
        ratio_by_kind.setdefault(rec.kind, []).append(rec.ratio_curve)
    for kind, color in kind_colors.items():
        if kind not in ratio_by_kind:
            continue
        avg = torch.tensor(ratio_by_kind[kind]).mean(dim=0).tolist()
        ax_c.plot(steps, avg, marker="o", ms=4, lw=1.8, label=kind, color=color)
    ax_c.set_yscale("log")
    ax_c.set_xlabel("Denoise step")
    ax_c.set_ylabel("Max/min ratio (log)")
    ax_c.set_title("Outlier severity vs step")
    ax_c.legend(fontsize=8)
    ax_c.grid(True, alpha=0.25, which="both")

    # D: outlier channel stability (step0 vs last) by kind.
    ax_d = fig.add_subplot(gs[1, 1])
    kind_stab: dict[LayerKind, list[float]] = {}
    for rec in records:
        if rec.name in stability:
            kind_stab.setdefault(rec.kind, []).append(stability[rec.name])
    kinds = [k for k in kind_colors if k in kind_stab]
    if kinds:
        positions = range(len(kinds))
        means = [torch.tensor(kind_stab[k]).mean().item() for k in kinds]
        ax_d.bar(positions, means, color=[kind_colors[k] for k in kinds], width=0.6)
        ax_d.set_xticks(list(positions))
        ax_d.set_xticklabels(kinds, rotation=20, ha="right", fontsize=8)
        ax_d.set_ylim(0, 1.05)
        ax_d.axhline(0.5, color="#999999", ls="--", lw=0.7)
        ax_d.set_ylabel("Top-k channel overlap")
        ax_d.set_title("Outlier channel stability (step 0 vs last)")

    fig.suptitle(f"DiT activation vs denoise step ({act_metric})", fontsize=12, y=0.98)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote overview plot to {path}")


def _plot_single_layer_channels(
    store: dict[str, dict[int, _PerStepAccumulator]],
    layer_name: str,
    path: Path,
    *,
    num_steps: int,
    act_metric: ActMetric,
    highlight_steps: tuple[int, ...],
) -> None:
    import matplotlib.pyplot as plt

    step_dict = store[layer_name]
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.cm.plasma
    for step in highlight_steps:
        if step not in step_dict:
            continue
        vals = step_dict[step].channel_scores().numpy()
        color = cmap(step / max(1, num_steps - 1))
        ax.plot(vals, color=color, alpha=0.9, lw=1.5, label=f"step {step}")

    ax.set_xlabel("Input channel")
    ax.set_ylabel(f"Per-channel {act_metric}")
    ax.set_title(_short_name(layer_name))
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote per-channel curves to {path}")


def _channel_vectors_by_step(
    store: dict[str, dict[int, _PerStepAccumulator]],
    layer_name: str,
    num_steps: int,
) -> list[torch.Tensor]:
    """Per-step channel score vectors, ordered by step index."""
    step_dict = store[layer_name]
    return [step_dict[s].channel_scores() for s in range(num_steps) if s in step_dict]


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1].item())


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra = a.argsort().argsort().to(torch.float64)
    rb = b.argsort().argsort().to(torch.float64)
    return _pearson(ra, rb)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def _step_similarity_matrix(
    vectors: list[torch.Tensor],
    *,
    metric: Literal["pearson", "spearman", "cosine"],
) -> torch.Tensor:
    n = len(vectors)
    mat = torch.eye(n, dtype=torch.float64)
    fn = {"pearson": _pearson, "spearman": _spearman, "cosine": _cosine}[metric]
    for i in range(n):
        for j in range(i + 1, n):
            v = fn(vectors[i], vectors[j])
            mat[i, j] = v
            mat[j, i] = v
    return mat


def _print_step_channel_similarity(
    layer_name: str,
    vectors: list[torch.Tensor],
    *,
    num_steps: int,
    act_metric: ActMetric,
) -> None:
    pearson = _step_similarity_matrix(vectors, metric="pearson")
    spearman = _step_similarity_matrix(vectors, metric="spearman")
    cosine = _step_similarity_matrix(vectors, metric="cosine")

    off_diag = pearson[~torch.eye(num_steps, dtype=bool)]
    print(f"\n=== Step curve similarity: {_short_name(layer_name)} ({act_metric}) ===")
    print(f"  Pearson  off-diag mean={off_diag.mean():+.4f}  min={off_diag.min():+.4f}")
    print(f"  Spearman off-diag mean={spearman[~torch.eye(num_steps, dtype=bool)].mean():+.4f}")
    print(f"  Cosine   off-diag mean={cosine[~torch.eye(num_steps, dtype=bool)].mean():+.4f}")
    print(f"  step0 vs step{num_steps - 1}:  r={pearson[0, -1]:+.4f}  "
          f"rho={spearman[0, -1]:+.4f}  cos={cosine[0, -1]:+.4f}")
    print("  vs step0 (Pearson r):")
    for s in range(num_steps):
        print(f"    step {s:2d}: {pearson[0, s]:+.4f}")

    # Adjacent-step drift: low adjacent correlation => shape shifts gradually.
    adj = [pearson[s, s + 1].item() for s in range(num_steps - 1)]
    worst = min(range(num_steps - 1), key=lambda s: adj[s])
    print(f"  lowest adjacent Pearson: step{worst}↔step{worst + 1} = {adj[worst]:+.4f}")


def _plot_step_channel_similarity(
    layer_name: str,
    vectors: list[torch.Tensor],
    path: Path,
    *,
    num_steps: int,
    act_metric: ActMetric,
) -> None:
    """Heatmaps of step×step similarity between per-channel score curves."""
    import matplotlib.pyplot as plt
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    pearson = _step_similarity_matrix(vectors, metric="pearson").numpy()
    spearman = _step_similarity_matrix(vectors, metric="spearman").numpy()
    steps = list(range(num_steps))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, mat, title in zip(
        axes,
        (pearson, spearman),
        ("Pearson r (shape + scale)", "Spearman rho (rank / peak location)"),
    ):
        im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto", interpolation="nearest")
        ax.set_xticks(steps)
        ax.set_yticks(steps)
        ax.set_xlabel("Step")
        ax.set_ylabel("Step")
        ax.set_title(title)
        for i in range(num_steps):
            for j in range(num_steps):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7, color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Step curve similarity — {_short_name(layer_name)} ({act_metric})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote step similarity heatmap to {path}")


def _plot_step_corr_vs_ref(
    layer_name: str,
    vectors: list[torch.Tensor],
    path: Path,
    *,
    num_steps: int,
    act_metric: ActMetric,
    ref_step: int = 0,
) -> None:
    """Line plot: correlation with a reference step."""
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    ref = vectors[ref_step]
    steps = list(range(num_steps))
    pearson = [_pearson(ref, v) for v in vectors]
    spearman = [_spearman(ref, v) for v in vectors]
    cosine = [_cosine(ref, v) for v in vectors]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, pearson, marker="o", ms=5, lw=1.8, label="Pearson r")
    ax.plot(steps, spearman, marker="s", ms=4, lw=1.5, label="Spearman rho")
    ax.plot(steps, cosine, marker="^", ms=4, lw=1.5, label="Cosine")
    ax.axhline(1.0, color="#cccccc", lw=0.6)
    ax.set_xlabel("Step")
    ax.set_ylabel(f"Similarity vs step {ref_step}")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Curve drift vs step {ref_step} — {_short_name(layer_name)}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote step-vs-ref plot to {path}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/data/share/pi05_libero_finetuned_v044"),
    )
    p.add_argument("--calibration-data", type=Path, help=".npz calibration file.")
    p.add_argument(
        "--calibration-source",
        choices=("file", "synthetic"),
        default="file",
        help="Use --calibration-data with file (default), or synthetic for quick tests.",
    )
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument(
        "--layer-regex",
        help="Optional regex filter on expert_stack layers.",
    )
    p.add_argument(
        "--act-metric",
        choices=("max_abs", "l2"),
        default="l2",
    )
    p.add_argument("--top-k", type=int, default=64, help="Top-k for outlier overlap.")
    p.add_argument(
        "--channel-curves",
        action="store_true",
        help="Also save per-channel curves (best with a single --layer-regex match).",
    )
    args = p.parse_args(argv)

    if args.calibration_source == "file" and args.calibration_data is None:
        default_cal = _ROOT.parent / "calibration_data" / "libero_object_16_7.npz"
        if default_cal.is_file():
            args.calibration_data = default_cal
        else:
            p.error("--calibration-data is required when --calibration-source=file.")

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
    targets = _resolve_dit_targets(model, config, layer_regex=args.layer_regex)
    print(
        f"Collecting DiT activations: {len(targets)} layer(s), "
        f"{num_steps} step(s), {args.num_samples} sample(s)..."
    )

    store = _collect_per_step_activations(
        adapter,
        model,
        targets,
        num_samples=args.num_samples,
        num_steps=num_steps,
        act_metric=args.act_metric,
    )
    records = _build_records(store, num_steps=num_steps)
    stability = _channel_stability(store, num_steps=num_steps, top_k=args.top_k)

    _print_conclusions(
        records,
        num_steps=num_steps,
        stability=stability,
        top_k=args.top_k,
        act_metric=args.act_metric,
    )

    _IMG_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(_IMG_DIR / "dit_step_act_stats.csv", records, act_metric=args.act_metric)

    try:
        _plot_step_analysis(
            records,
            store,
            _IMG_DIR / "dit_step_act_summary.png",
            num_steps=num_steps,
            act_metric=args.act_metric,
            stability=stability,
        )
        if args.channel_curves or (args.layer_regex and len(targets) == 1):
            layer_name = targets[0][0]
            suffix = _layer_file_suffix(layer_name)
            hi = (0, num_steps // 2, num_steps - 1)
            _plot_single_layer_channels(
                store,
                layer_name,
                _IMG_DIR / f"dit_step_act_channels_{suffix}.png",
                num_steps=num_steps,
                act_metric=args.act_metric,
                highlight_steps=hi,
            )
            vectors = _channel_vectors_by_step(store, layer_name, num_steps)
            _print_step_channel_similarity(
                layer_name, vectors, num_steps=num_steps, act_metric=args.act_metric
            )
            _plot_step_channel_similarity(
                layer_name,
                vectors,
                _IMG_DIR / f"dit_step_act_channels_step_similarity_{suffix}.png",
                num_steps=num_steps,
                act_metric=args.act_metric,
            )
            _plot_step_corr_vs_ref(
                layer_name,
                vectors,
                _IMG_DIR / f"dit_step_act_channels_step_corr_vs_ref_{suffix}.png",
                num_steps=num_steps,
                act_metric=args.act_metric,
            )
    except ImportError as e:
        raise SystemExit("Install matplotlib (uv sync --group dev) for plots.") from e

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
