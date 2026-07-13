#!/usr/bin/env python
"""Per-channel activation vs weight correlation for one or all quant target layers.

Example (single layer + scatter)::

    uv run python tools/analyze_act_weight_correlation.py \\
        --checkpoint /data/share/pi05-libero \\
        --layer-regex 'paligemma_lm\\.layers\\.0\\.qkv_proj$' \\
        --calibration-data ../calibration_data/libero_object_16_7.npz \\
        --scatter-output ./figures/act_weight_scatter.png

Example (all quant layers + summary)::

    uv run python tools/analyze_act_weight_correlation.py \\
        --checkpoint /data/share/pi05-libero \\
        --calibration-data ../calibration_data/libero_object_16_7.npz \\
        --num-samples 10 \\
        --output-csv ./figures/act_weight_corr.csv \\
        --summary-output ./figures/act_weight_corr_summary.png
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from qvla.adapters import get_adapter  # noqa: E402
from qvla.config import QVLAConfig  # noqa: E402
from qvla.runtime import list_target_modules  # noqa: E402


@dataclass
class LayerResult:
    name: str
    scope: str
    pearson: float
    spearman: float
    topk_overlap: float
    n_channels: int


class _ActAccumulator:
    """Online per-channel activation stats (no full tensor storage)."""

    __slots__ = ("metric", "n_tokens", "sum_sq", "max_abs")

    def __init__(self, metric: str, in_features: int) -> None:
        self.metric = metric
        self.n_tokens = 0
        self.sum_sq = torch.zeros(in_features, dtype=torch.float64)
        self.max_abs = torch.zeros(in_features, dtype=torch.float32)

    def update(self, x: torch.Tensor) -> None:
        flat = x.reshape(-1, x.shape[-1]).detach().to(device="cpu", dtype=torch.float32)
        self.n_tokens += flat.shape[0]
        self.sum_sq += (flat.double() * flat.double()).sum(dim=0)
        self.max_abs = torch.maximum(self.max_abs, flat.abs().amax(dim=0))

    def finalize(self) -> torch.Tensor:
        if self.n_tokens == 0:
            raise RuntimeError("No tokens collected for activation accumulator.")
        if self.metric == "max_abs":
            return self.max_abs
        if self.metric == "l2":
            return self.sum_sq.sqrt().to(torch.float32)
        if self.metric == "col_mean_sq":
            return (self.sum_sq / self.n_tokens).to(torch.float32)
        raise ValueError(f"Unknown act_metric {self.metric!r}")


def _resolve_targets(model, config, *, layer_regex: str | None, scope: str | None):
    targets = list_target_modules(model, config)
    if scope is not None:
        targets = [(n, s, m) for n, s, m in targets if s == scope]
    if layer_regex:
        pattern = re.compile(layer_regex)
        targets = [(n, s, m) for n, s, m in targets if pattern.search(n)]
    if not targets:
        raise SystemExit("No target layers matched; check --layer-regex / --scope.")
    return targets


def _weight_scores(weight: torch.Tensor, *, w_metric: str) -> torch.Tensor:
    w = weight.detach().to(device="cpu", dtype=torch.float32)
    if w_metric == "col_l2":
        return w.norm(dim=0, p=2)
    if w_metric == "col_mean_sq":
        return (w * w).mean(dim=0)
    raise ValueError(f"Unknown w_metric {w_metric!r}")


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1].item())


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra = a.argsort().argsort().to(torch.float64)
    rb = b.argsort().argsort().to(torch.float64)
    return _pearson(ra, rb)


def _topk_overlap(act: torch.Tensor, wt: torch.Tensor, k: int) -> float:
    k = min(k, act.numel())
    top_act = set(act.topk(k).indices.tolist())
    bot_wt = set(wt.topk(k, largest=False).indices.tolist())
    return len(top_act & bot_wt) / k


def _analyze_layer(
    name: str,
    scope: str,
    weight: torch.Tensor,
    act: torch.Tensor,
    *,
    w_metric: str,
    top_k: int,
) -> LayerResult:
    wt = _weight_scores(weight, w_metric=w_metric)
    return LayerResult(
        name=name,
        scope=scope,
        pearson=_pearson(act, wt),
        spearman=_spearman(act, wt),
        topk_overlap=_topk_overlap(act, wt, top_k),
        n_channels=int(act.numel()),
    )


def _collect_all_activations(
    adapter,
    model,
    targets: list[tuple[str, str, torch.nn.Module]],
    *,
    num_samples: int,
    act_metric: str,
) -> dict[str, torch.Tensor]:
    accumulators: dict[str, _ActAccumulator] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(layer_name: str, in_features: int):
        def hook(_mod, inputs):
            if not inputs or not torch.is_tensor(inputs[0]):
                return
            if layer_name not in accumulators:
                accumulators[layer_name] = _ActAccumulator(act_metric, in_features)
            accumulators[layer_name].update(inputs[0])

        return hook

    for name, _, mod in targets:
        in_features = mod.weight.shape[1]
        handles.append(mod.register_forward_pre_hook(make_hook(name, in_features)))

    try:
        for _, batch in enumerate(adapter.iter_calibration_batches(num_samples)):
            adapter.forward_for_calibration(model, batch, step_callback=lambda _: None)
    finally:
        for handle in handles:
            handle.remove()

    missing = [n for n, _, _ in targets if n not in accumulators]
    if missing:
        raise RuntimeError(f"No activations captured for: {missing[:5]}")

    return {name: acc.finalize() for name, acc in accumulators.items()}


def _short_name(full_name: str) -> str:
    """Keep the tail for compact axis labels."""
    parts = full_name.split(".")
    if len(parts) <= 4:
        return full_name
    return ".".join(parts[-4:])


def _print_layer_report(result: LayerResult, *, act_metric: str, w_metric: str, top_k: int) -> None:
    print(f"Layer: {result.name} ({result.scope})")
    print(f"  channels: {result.n_channels}")
    print(f"  activation metric: {act_metric}")
    print(f"  weight metric:     {w_metric}")
    print(f"  Pearson r:         {result.pearson:+.4f}")
    print(f"  Spearman rho:      {result.spearman:+.4f}")
    print(f"  top-{top_k} act ∩ bottom-{top_k} wt: {result.topk_overlap * 100:.1f}%")


def _print_summary(results: list[LayerResult]) -> None:
    pearsons = [r.pearson for r in results]
    t = torch.tensor(pearsons)
    neg = sum(1 for p in pearsons if p < -0.2)
    pos = sum(1 for p in pearsons if p > 0.2)
    weak = len(pearsons) - neg - pos
    print(f"\nSummary ({len(results)} layers)")
    print(f"  Pearson mean: {t.mean():+.4f}  median: {t.median():+.4f}  std: {t.std():.4f}")
    print(f"  negative (< -0.2): {neg}   weak: {weak}   positive (> +0.2): {pos}")
    for scope in ("llm", "dit"):
        vals = [r.pearson for r in results if r.scope == scope]
        if vals:
            s = torch.tensor(vals)
            print(f"  {scope:3s}: n={len(vals):3d}  mean={s.mean():+.4f}  median={s.median():+.4f}")


def _write_csv(path: Path, results: list[LayerResult], *, act_metric: str, w_metric: str, top_k: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "scope",
                "short_name",
                "pearson",
                "spearman",
                "topk_overlap",
                "n_channels",
                "act_metric",
                "w_metric",
                "top_k",
            ],
        )
        w.writeheader()
        for r in results:
            w.writerow(
                {
                    "name": r.name,
                    "scope": r.scope,
                    "short_name": _short_name(r.name),
                    "pearson": f"{r.pearson:.6f}",
                    "spearman": f"{r.spearman:.6f}",
                    "topk_overlap": f"{r.topk_overlap:.6f}",
                    "n_channels": r.n_channels,
                    "act_metric": act_metric,
                    "w_metric": w_metric,
                    "top_k": top_k,
                }
            )
    print(f"Wrote CSV to {path}")


def _plot_scatter(act: torch.Tensor, wt: torch.Tensor, path: Path, *, layer: str, pearson: float) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(act.numpy(), wt.numpy(), s=4, alpha=0.35, c="#1f77b4", edgecolors="none")
    ax.set_xlabel("Activation channel score")
    ax.set_ylabel("Weight column score")
    ax.set_title(f"{_short_name(layer)}\nPearson r = {pearson:+.3f}")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote scatter plot to {path}")


def _plot_summary(results: list[LayerResult], path: Path, *, act_metric: str, w_metric: str) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    path.parent.mkdir(parents=True, exist_ok=True)
    scope_colors = {"llm": "#1f77b4", "dit": "#ff7f0e"}

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1], hspace=0.35, wspace=0.25)

    # --- panel A: Pearson bar chart per layer (sorted), split by scope rows ---
    ax_bar = fig.add_subplot(gs[0, :])
    ordered = sorted(results, key=lambda r: (r.scope, r.pearson))
    labels = [_short_name(r.name) for r in ordered]
    vals = [r.pearson for r in ordered]
    colors = [scope_colors.get(r.scope, "#888888") for r in ordered]
    y = range(len(ordered))
    ax_bar.barh(list(y), vals, color=colors, height=0.72)
    ax_bar.axvline(0, color="#333333", lw=0.8)
    ax_bar.axvline(-0.2, color="#999999", ls="--", lw=0.6)
    ax_bar.axvline(0.2, color="#999999", ls="--", lw=0.6)
    ax_bar.set_xlabel("Pearson r")
    ax_bar.set_title(f"Per-layer activation–weight correlation ({act_metric} vs {w_metric})")
    step = max(1, len(ordered) // 25)
    ax_bar.set_yticks(list(y)[::step])
    ax_bar.set_yticklabels(labels[::step], fontsize=6)
    ax_bar.invert_yaxis()
    handles = [mpatches.Patch(color=c, label=s) for s, c in scope_colors.items()]
    ax_bar.legend(handles=handles, loc="lower right", fontsize=8)

    # --- panel B: histogram ---
    ax_hist = fig.add_subplot(gs[1, 0])
    ax_hist.hist(vals, bins=30, color="#6baed6", edgecolor="white")
    ax_hist.axvline(0, color="#333333", lw=0.8)
    ax_hist.set_xlabel("Pearson r")
    ax_hist.set_ylabel("Layer count")
    ax_hist.set_title("Distribution across layers")

    # --- panel C: Pearson vs Spearman ---
    ax_sc = fig.add_subplot(gs[1, 1])
    for scope in ("llm", "dit"):
        pts = [r for r in results if r.scope == scope]
        if not pts:
            continue
        ax_sc.scatter(
            [r.pearson for r in pts],
            [r.spearman for r in pts],
            s=18,
            alpha=0.7,
            c=scope_colors[scope],
            label=scope,
            edgecolors="none",
        )
    lim = max(0.5, max(abs(v) for v in vals) * 1.1)
    ax_sc.plot([-lim, lim], [-lim, lim], ls="--", c="#cccccc", lw=0.8)
    ax_sc.axhline(0, c="#dddddd", lw=0.6)
    ax_sc.axvline(0, c="#dddddd", lw=0.6)
    ax_sc.set_xlim(-lim, lim)
    ax_sc.set_ylim(-lim, lim)
    ax_sc.set_xlabel("Pearson r")
    ax_sc.set_ylabel("Spearman rho")
    ax_sc.set_title("Linear vs rank correlation")
    ax_sc.legend(fontsize=8)

    t = torch.tensor(vals)
    fig.suptitle(
        f"{len(results)} quant layers  |  mean r = {t.mean():+.3f}  median = {t.median():+.3f}",
        fontsize=11,
        y=0.98,
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote summary plot to {path}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--layer-regex",
        help="Optional regex filter. Omit to analyze all quant target layers.",
    )
    p.add_argument(
        "--scope",
        choices=("llm", "dit"),
        help="Optional scope filter (default: both llm and dit).",
    )
    p.add_argument("--calibration-data", type=Path, required=True)
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument(
        "--act-metric",
        choices=("max_abs", "l2", "col_mean_sq"),
        default="l2",
    )
    p.add_argument(
        "--w-metric",
        choices=("col_l2", "col_mean_sq"),
        default="col_l2",
    )
    p.add_argument("--top-k", type=int, default=32)
    p.add_argument("--output-csv", type=Path, help="CSV table (recommended for all-layer runs).")
    p.add_argument(
        "--summary-output",
        type=Path,
        help="Overview figure: per-layer bars + histogram + Pearson vs Spearman.",
    )
    p.add_argument(
        "--scatter-output",
        type=Path,
        help="Per-channel scatter (single layer only, or use with one --layer-regex match).",
    )
    args = p.parse_args(argv)

    if args.scatter_output and not args.layer_regex:
        p.error("--scatter-output requires --layer-regex selecting a single layer.")

    config = QVLAConfig.pi05_default()
    adapter = get_adapter(
        "pi05",
        checkpoint_path=args.checkpoint,
        calibration_source="file",
        calibration_data_path=args.calibration_data,
    )
    model = adapter.build_model()
    adapter.warmup_for_calibration(model)

    targets = _resolve_targets(model, config, layer_regex=args.layer_regex, scope=args.scope)
    print(f"Analyzing {len(targets)} layer(s) with {args.num_samples} calibration sample(s)...")

    act_by_name = _collect_all_activations(
        adapter, model, targets, num_samples=args.num_samples, act_metric=args.act_metric
    )

    results: list[LayerResult] = []
    mod_by_name = {name: mod for name, _, mod in targets}
    for name, scope, _ in targets:
        results.append(
            _analyze_layer(
                name,
                scope,
                mod_by_name[name].weight,
                act_by_name[name],
                w_metric=args.w_metric,
                top_k=args.top_k,
            )
        )

    results.sort(key=lambda r: r.name)

    if len(results) == 1:
        _print_layer_report(
            results[0], act_metric=args.act_metric, w_metric=args.w_metric, top_k=args.top_k
        )
    else:
        for r in results:
            print(
                f"  {r.pearson:+.3f}  {r.spearman:+.3f}  "
                f"overlap={r.topk_overlap * 100:5.1f}%  {_short_name(r.name)}"
            )
        _print_summary(results)

    if args.output_csv is not None:
        _write_csv(
            args.output_csv, results, act_metric=args.act_metric, w_metric=args.w_metric, top_k=args.top_k
        )

    if args.summary_output is not None:
        try:
            _plot_summary(
                results, args.summary_output, act_metric=args.act_metric, w_metric=args.w_metric
            )
        except ImportError as e:
            raise SystemExit("Install matplotlib (uv sync --group dev) for --summary-output.") from e

    if args.scatter_output is not None:
        if len(results) != 1:
            raise SystemExit("--scatter-output requires exactly one matched layer.")
        r = results[0]
        wt = _weight_scores(mod_by_name[r.name].weight, w_metric=args.w_metric)
        try:
            _plot_scatter(act_by_name[r.name], wt, args.scatter_output, layer=r.name, pearson=r.pearson)
        except ImportError as e:
            raise SystemExit("Install matplotlib (uv sync --group dev) for --scatter-output.") from e

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
