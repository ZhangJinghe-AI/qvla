#!/usr/bin/env python
r"""Match a fixed L1 fraction with selective mean+std clip, then rank action impact.

Channel selection is the repo's MAD rule. On selected channels the bound is
still per-channel ``μ + k·σ``; unselected channels are untouched. ``k`` is
solved so that the removed L1 fraction equals ``α``. If selected channels
cannot supply that much L1 under this bound form, the run raises.

Each DiT linear is intervened in isolation. ``--target-step`` clips only one
denoise step; omit it to clip every denoise step. ``--samples`` / ``--noise-seeds``
average action RMSE across several calibration inputs and noise draws.

Example:

    CUDA_VISIBLE_DEVICES=5 uv run python \
      tools/analyze_dit_selective_matched_l1_action_impact.py \
      --checkpoint /data/share/pi05_libero_finetuned_v044 \
      --calibration-data ../calibration_data/libero_goal_30_7_demo.npz \
      --samples 0,1,2,3 --noise-seeds 0,1 --l1-remove 0.005 \
      --output-dir tools/img/dit_selective_matched_l1_a005_allsteps_s4n2

"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
_TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_TOOLS))

import analyze_dit_matched_clip_action_impact as matched  # noqa: E402
import analyze_dit_outlier_action_detail as detail  # noqa: E402
from qvla.config import QVLAConfig  # noqa: E402
from qvla.core.clip import select_channels_by_robust_amax  # noqa: E402

L1_MATCH_ATOL = 1e-6
K_SEARCH_ITERS = 80


@dataclass(frozen=True)
class SelectiveClipStats:
    k: float
    n_selected_channels: int
    removed_l1_frac: float
    token_clip_frac: float


@dataclass(frozen=True)
class LayerResult:
    layer_name: str
    layer_idx: int
    kind: str
    mean_k: float
    n_selected_channels: int
    mean_removed_l1_frac: float
    mean_token_clip_frac: float
    metrics: detail.ActionMetrics


def _removed_l1_frac(abs_x: torch.Tensor, bounds: torch.Tensor) -> float:
    values = abs_x.detach().to(torch.float64)
    cap = bounds.detach().to(torch.float64).view(1, -1)
    total = float(values.sum().item())
    if total <= 0.0:
        raise RuntimeError("Activation L1 norm is non-positive.")
    removed = float((values - values.clamp(max=cap)).sum().item())
    return removed / total


def _bounds_for_k(
    mu: torch.Tensor,
    std: torch.Tensor,
    amax: torch.Tensor,
    selected: torch.Tensor,
    k: float,
) -> torch.Tensor:
    raw = (mu + float(k) * std).clamp(min=0.0)
    return torch.where(selected, torch.minimum(amax, raw), amax)


def _clip_selective_matched_k(
    x: torch.Tensor, alpha: float
) -> tuple[torch.Tensor, SelectiveClipStats]:
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    if x.ndim != 2 or min(x.shape) < 2:
        raise ValueError(f"Need 2-D activations with T,C >= 2, got {tuple(x.shape)}.")
    abs_x = x.detach().abs().to(torch.float32)
    if not bool(torch.isfinite(abs_x).all().item()):
        raise RuntimeError("Selective matched clip requires finite |x|.")
    amax = abs_x.amax(dim=0)
    selected, _, _, _ = select_channels_by_robust_amax(amax)
    n_selected = int(selected.sum().item())
    if n_selected < 1:
        raise RuntimeError("Selective rule selected no channels.")
    mu = abs_x.mean(dim=0)
    std = abs_x.std(dim=0, unbiased=False)
    vary = selected & (std > 0.0)
    if not bool(vary.any().item()):
        raise RuntimeError(
            "Selected channels have zero token-wise std; mean+std cannot clip them."
        )
    eps = torch.finfo(torch.float32).tiny
    k_hi = float(((amax - mu) / std.clamp(min=eps))[vary].max().item()) + 1.0
    k_lo = float(((-mu) / std.clamp(min=eps))[vary].min().item()) - 1.0
    max_frac = _removed_l1_frac(abs_x, _bounds_for_k(mu, std, amax, selected, k_lo))
    min_frac = _removed_l1_frac(abs_x, _bounds_for_k(mu, std, amax, selected, k_hi))
    if max_frac + L1_MATCH_ATOL < float(alpha):
        raise RuntimeError(
            f"Selected mean+std clip can remove at most {max_frac:.8g} L1, "
            f"cannot hit alpha={alpha:.8g}."
        )
    if min_frac > float(alpha) + L1_MATCH_ATOL:
        raise RuntimeError(
            f"Even the weakest selective mean+std clip removes {min_frac:.8g} L1, "
            f"above alpha={alpha:.8g}."
        )
    k = k_hi
    hit = False
    for _ in range(K_SEARCH_ITERS):
        k = 0.5 * (k_lo + k_hi)
        frac = _removed_l1_frac(abs_x, _bounds_for_k(mu, std, amax, selected, k))
        if abs(frac - float(alpha)) <= L1_MATCH_ATOL:
            hit = True
            break
        if frac > float(alpha):
            k_lo = k
        else:
            k_hi = k
    bounds = _bounds_for_k(mu, std, amax, selected, k)
    removed_frac = _removed_l1_frac(abs_x, bounds)
    if (not hit) and abs(removed_frac - float(alpha)) > L1_MATCH_ATOL:
        raise RuntimeError(
            f"Selective k search missed alpha: want {alpha:.8g}, "
            f"got {removed_frac:.8g} (k={k:.8g})."
        )
    token_frac = float(
        (abs_x.to(torch.float64) > bounds.to(torch.float64)).to(torch.float64).mean().item()
    )
    if token_frac <= 0.0:
        raise RuntimeError("Selective matched clip selected no elements.")
    live = x.to(torch.float32)
    clipped = live.sign() * live.abs().clamp(max=bounds.to(device=live.device))
    unselected = ~selected.to(device=live.device)
    if bool(unselected.any().item()) and not torch.equal(
        clipped[:, unselected], live[:, unselected]
    ):
        raise RuntimeError("Unselected channels were modified.")
    return clipped.to(dtype=x.dtype), SelectiveClipStats(
        k=float(k),
        n_selected_channels=n_selected,
        removed_l1_frac=removed_frac,
        token_clip_frac=token_frac,
    )


def _mean(xs: list[float]) -> float:
    if not xs:
        raise ValueError("Cannot average an empty list.")
    return sum(xs) / len(xs)


def _parse_nonneg_ints(text: str, *, name: str) -> list[int]:
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if not parts:
        raise ValueError(f"{name} must list at least one integer.")
    values: list[int] = []
    seen: set[int] = set()
    for part in parts:
        value = int(part)
        if value < 0:
            raise ValueError(f"{name} values must be >= 0, got {value}.")
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def _depth_corr(results: list[LayerResult]) -> tuple[float, float]:
    if len(results) < 3:
        raise ValueError("Need at least 3 layers for a depth correlation.")
    layers = np.array([row.layer_idx for row in results], dtype=np.float64)
    rmse = np.array([row.metrics.total_rmse for row in results], dtype=np.float64)
    pearson = float(np.corrcoef(layers, rmse)[0, 1])
    spearman = float(
        np.corrcoef(layers.argsort().argsort(), rmse.argsort().argsort())[0, 1]
    )
    return pearson, spearman


def _mean_metrics(items: list[detail.ActionMetrics]) -> detail.ActionMetrics:
    names = [field.name for field in fields(detail.ActionMetrics)]
    return detail.ActionMetrics(
        **{
            name: _mean([float(getattr(item, name)) for item in items])
            for name in names
        }
    )


def _mean_layer_results(groups: list[list[LayerResult]]) -> list[LayerResult]:
    if not groups:
        raise ValueError("Need at least one condition to average.")
    n_layers = len(groups[0])
    if n_layers < 1:
        raise ValueError("Each condition must contain at least one layer.")
    if any(len(group) != n_layers for group in groups):
        raise RuntimeError("Condition layer counts do not match.")
    averaged: list[LayerResult] = []
    for index in range(n_layers):
        rows = [group[index] for group in groups]
        names = {row.layer_name for row in rows}
        if len(names) != 1:
            raise RuntimeError(f"Layer order mismatch at index {index}: {sorted(names)}.")
        averaged.append(
            LayerResult(
                layer_name=rows[0].layer_name,
                layer_idx=rows[0].layer_idx,
                kind=rows[0].kind,
                mean_k=_mean([row.mean_k for row in rows]),
                n_selected_channels=int(
                    round(_mean([float(row.n_selected_channels) for row in rows]))
                ),
                mean_removed_l1_frac=_mean([row.mean_removed_l1_frac for row in rows]),
                mean_token_clip_frac=_mean([row.mean_token_clip_frac for row in rows]),
                metrics=_mean_metrics([row.metrics for row in rows]),
            )
        )
    return averaged


def _eval_layers(
    adapter,
    request,
    layers: list[tuple[str, torch.nn.Module]],
    *,
    runtime: matched.ClipRuntime,
    alpha: float,
    target_step: int | None,
) -> list[LayerResult]:
    num_steps = runtime.num_steps
    horizon = runtime.action_horizon
    action_dim = runtime.action_dim
    results: list[LayerResult] = []
    for index, (layer_name, layer) in enumerate(layers, start=1):
        baseline_actions, baseline_activations, _ = matched._run_layer(
            adapter,
            request,
            layer,
            num_steps=num_steps,
            horizon=horizon,
            n_tokens=runtime.n_tokens,
            alpha=None,
        )
        assert baseline_activations is not None
        if baseline_actions.shape[:2] != (1, horizon):
            raise RuntimeError(
                f"Expected actions (1,{horizon},*), got {baseline_actions.shape}."
            )
        if int(baseline_actions.shape[2]) < action_dim:
            raise RuntimeError(
                f"Action width {baseline_actions.shape[2]} < {action_dim}."
            )
        changed_actions, _, clip_stats = matched._run_layer(
            adapter,
            request,
            layer,
            num_steps=num_steps,
            horizon=horizon,
            n_tokens=runtime.n_tokens,
            alpha=alpha,
            baseline_activations=baseline_activations,
            target_step=target_step,
            clip_fn=_clip_selective_matched_k,
        )
        stats = list(clip_stats)
        if not stats:
            raise RuntimeError(f"No clip stats for {layer_name}.")
        metrics = detail._action_metrics(
            changed_actions[0, :, :action_dim],
            baseline_actions[0, :, :action_dim],
        )
        row = LayerResult(
            layer_name=layer_name,
            layer_idx=matched._layer_idx(layer_name),
            kind=matched._layer_kind(layer_name),
            mean_k=_mean([item.k for item in stats]),
            n_selected_channels=int(stats[0].n_selected_channels)
            if len(stats) == 1
            else int(round(_mean([float(item.n_selected_channels) for item in stats]))),
            mean_removed_l1_frac=_mean([item.removed_l1_frac for item in stats]),
            mean_token_clip_frac=_mean([item.token_clip_frac for item in stats]),
            metrics=metrics,
        )
        results.append(row)
        print(
            f"[{index}/{len(layers)}] {layer_name}  "
            f"L1={100 * row.mean_removed_l1_frac:.3f}%  "
            f"k={row.mean_k:.4g}  "
            f"sel_ch={row.n_selected_channels}  "
            f"tokens={100 * row.mean_token_clip_frac:.3f}%  "
            f"total={metrics.total_rmse:.3e}  "
            f"end={metrics.arm_endpoint_rmse:.3e}  "
            f"local={metrics.arm_local_rmse:.3e}  "
            f"grip={metrics.gripper_rmse:.3e}"
        )
    return results


def _plot(
    results: list[LayerResult],
    output: Path,
    *,
    alpha: float,
    target_step: int | None,
    condition_label: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    xs = np.arange(len(results))
    labels = [f"L{row.layer_idx}.{row.kind.replace('_proj', '')}" for row in results]
    bar_colors = [matched.KIND_COLORS.get(row.kind, "#9D755D") for row in results]

    ax = axes[0, 0]
    ax.bar(xs, [row.metrics.total_rmse for row in results], color=bar_colors)
    ax.set_title("Total action RMSE after selective matched-L1 clip")
    ax.set_ylabel("action RMSE")

    ax = axes[0, 1]
    ax.plot(xs, [row.metrics.arm_mean_shift_rmse for row in results], "o-", label="mean")
    ax.plot(xs, [row.metrics.arm_endpoint_rmse for row in results], "o-", label="endpoint")
    ax.plot(xs, [row.metrics.arm_net_disp_rmse for row in results], "o-", label="start-to-end")
    ax.set_title("Overall arm change")
    ax.set_ylabel("RMSE")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(xs, [row.metrics.arm_local_rmse for row in results], "o-", label="local")
    ax.plot(xs, [row.metrics.arm_step_rmse for row in results], "o-", label="step")
    ax.plot(xs, [row.metrics.gripper_rmse for row in results], "o-", label="gripper")
    ax.set_title("Local / gripper change")
    ax.set_ylabel("RMSE")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(xs, [100.0 * row.mean_token_clip_frac for row in results], "o-", label="token clip %")
    ax.axhline(100.0 * alpha, color="black", linestyle="--", label=f"L1 target {100 * alpha:.2f}%")
    ax.plot(
        xs,
        [100.0 * row.mean_removed_l1_frac for row in results],
        "s--",
        label="removed L1 %",
    )
    ax.set_title("Clip intensity (L1 matched; tokens may differ)")
    ax.set_ylabel("percent")
    ax.legend(fontsize=8)

    stride = max(1, len(labels) // 18)
    ticks = list(range(0, len(labels), stride))
    for ax in axes.flat:
        ax.set_xticks(ticks)
        ax.set_xticklabels([labels[i] for i in ticks], rotation=75, fontsize=7)
        ax.grid(alpha=0.25, axis="y")
        ax.set_xlabel("DiT linear")
    step_label = (
        "every denoise step"
        if target_step is None
        else f"denoise step {target_step} only"
    )
    extra = f", {condition_label}" if condition_label else ""
    fig.suptitle(
        f"Selective mean+std clip, matched L1 α={alpha:g}, {step_label}{extra}",
        fontsize=12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Failed to write {output}.")


def _write_condition_csv(
    records: list[tuple[int, int, LayerResult]], output: Path
) -> None:
    metric_names = [field.name for field in fields(detail.ActionMetrics)]
    fieldnames = [
        "sample",
        "noise_seed",
        "layer",
        "layer_idx",
        "kind",
        "mean_k",
        "n_selected_channels",
        "mean_removed_l1_frac",
        "mean_token_clip_frac",
        *metric_names,
    ]
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for sample, noise_seed, row in records:
            item = {
                "sample": sample,
                "noise_seed": noise_seed,
                "layer": row.layer_name,
                "layer_idx": row.layer_idx,
                "kind": row.kind,
                "mean_k": row.mean_k,
                "n_selected_channels": row.n_selected_channels,
                "mean_removed_l1_frac": row.mean_removed_l1_frac,
                "mean_token_clip_frac": row.mean_token_clip_frac,
            }
            for name in metric_names:
                item[name] = getattr(row.metrics, name)
            writer.writerow(item)


def _write_corr_csv(
    rows: list[tuple[int, int, float, float]], output: Path
) -> None:
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("sample", "noise_seed", "pearson", "spearman"),
        )
        writer.writeheader()
        for sample, noise_seed, pearson, spearman in rows:
            writer.writerow(
                {
                    "sample": sample,
                    "noise_seed": noise_seed,
                    "pearson": pearson,
                    "spearman": spearman,
                }
            )


def _write_csv(results: list[LayerResult], output: Path) -> None:
    metric_names = [field.name for field in fields(detail.ActionMetrics)]
    fieldnames = [
        "layer",
        "layer_idx",
        "kind",
        "mean_k",
        "n_selected_channels",
        "mean_removed_l1_frac",
        "mean_token_clip_frac",
        *metric_names,
    ]
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            item = {
                "layer": row.layer_name,
                "layer_idx": row.layer_idx,
                "kind": row.kind,
                "mean_k": row.mean_k,
                "n_selected_channels": row.n_selected_channels,
                "mean_removed_l1_frac": row.mean_removed_l1_frac,
                "mean_token_clip_frac": row.mean_token_clip_frac,
            }
            for name in metric_names:
                item[name] = getattr(row.metrics, name)
            writer.writerow(item)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-data", type=Path, required=True)
    matched.add_model_cli(parser)
    parser.add_argument("--layer-regex", default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--samples",
        default=None,
        help="Comma-separated sample indices. Default: --sample-index.",
    )
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument(
        "--noise-seeds",
        default=None,
        help="Comma-separated noise seeds. Default: --noise-seed.",
    )
    parser.add_argument("--l1-remove", type=float, default=0.005)
    parser.add_argument(
        "--target-step",
        type=int,
        default=None,
        help="Clip only this 0-indexed denoise step. Default: clip every step.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--params-dtype", default="bfloat16")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tools/img/dit_selective_matched_l1_a005_step4"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(args.checkpoint)
    if not args.calibration_data.is_file():
        raise FileNotFoundError(args.calibration_data)
    if args.sample_index < 0:
        raise ValueError("--sample-index must be >= 0.")
    if not (0.0 < args.l1_remove < 1.0):
        raise ValueError(f"--l1-remove must be in (0, 1), got {args.l1_remove}.")
    if args.target_step is not None and args.target_step < 0:
        raise ValueError(f"--target-step must be >= 0, got {args.target_step}.")
    sample_ids = (
        _parse_nonneg_ints(args.samples, name="--samples")
        if args.samples is not None
        else [int(args.sample_index)]
    )
    noise_ids = (
        _parse_nonneg_ints(args.noise_seeds, name="--noise-seeds")
        if args.noise_seeds is not None
        else [int(args.noise_seed)]
    )

    adapter = matched.adapter_from_args(args)
    model = adapter.build_model()
    model.eval()
    adapter.warmup_for_calibration(model)
    config = QVLAConfig.for_model_kind(args.model)
    layers = matched._dit_layers(model, args.layer_regex, config=config)
    runtime = matched.clip_runtime(adapter, config)
    num_steps = runtime.num_steps
    horizon = runtime.action_horizon
    action_dim = runtime.action_dim
    if args.target_step is not None and args.target_step >= num_steps:
        raise ValueError(
            f"--target-step must be in [0, {num_steps}), got {args.target_step}."
        )

    need = max(sample_ids) + 1
    batches = list(adapter.iter_calibration_batches(need))
    if len(batches) != need:
        raise RuntimeError(
            f"Requested samples {sample_ids}, calibration yielded "
            f"{len(batches)} samples."
        )
    conditions = [(sample, seed) for sample in sample_ids for seed in noise_ids]
    print(
        f"model={args.model}, layers={len(layers)}, samples={sample_ids}, "
        f"noise_seeds={noise_ids}, n_conditions={len(conditions)}, "
        f"steps={num_steps}, horizon={horizon}, n_tokens={runtime.n_tokens}, "
        f"action_dim={action_dim}, l1_remove={args.l1_remove}, "
        f"target_step={args.target_step}, clip=selective_mean_std_matched_k"
    )

    probe_name, probe_layer = layers[0]
    first_sample, first_seed = conditions[0]
    probe_request = matched._fixed_noise_request(
        adapter, batches[first_sample], runtime, noise_seed=first_seed
    )
    baseline_actions, baseline_activations, _ = matched._run_layer(
        adapter,
        probe_request,
        probe_layer,
        num_steps=num_steps,
        horizon=horizon,
        n_tokens=runtime.n_tokens,
        alpha=None,
    )
    assert baseline_activations is not None
    identity_actions, _, _ = matched._run_layer(
        adapter,
        probe_request,
        probe_layer,
        num_steps=num_steps,
        horizon=horizon,
        n_tokens=runtime.n_tokens,
        alpha=None,
        baseline_activations=baseline_activations,
        target_step=args.target_step,
        identity_writeback=True,
    )
    if not torch.equal(identity_actions, baseline_actions):
        delta = (identity_actions - baseline_actions).to(torch.float64)
        rmse = float(delta.square().mean().sqrt().item())
        raise RuntimeError(
            f"Identity write-back moved actions on {probe_name}: RMSE={rmse:.6e}."
        )
    print(
        f"identity write-back on {probe_name} "
        f"(sample={first_sample}, noise={first_seed}): actions unchanged"
    )

    grouped: list[list[LayerResult]] = []
    corr_rows: list[tuple[int, int, float, float]] = []
    long_rows: list[tuple[int, int, LayerResult]] = []
    for index, (sample, seed) in enumerate(conditions, start=1):
        request = matched._fixed_noise_request(
            adapter, batches[sample], runtime, noise_seed=seed
        )
        print(
            f"\n=== condition {index}/{len(conditions)} "
            f"sample={sample} noise={seed} ==="
        )
        results = _eval_layers(
            adapter,
            request,
            layers,
            runtime=runtime,
            alpha=args.l1_remove,
            target_step=args.target_step,
        )
        pearson, spearman = _depth_corr(results)
        print(f"  Pearson(layer, RMSE)={pearson:.3f}  Spearman={spearman:.3f}")
        grouped.append(results)
        corr_rows.append((sample, seed, pearson, spearman))
        long_rows.extend((sample, seed, row) for row in results)

    mean_results = _mean_layer_results(grouped)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "selective_matched_l1_clip_action_impact.csv"
    png = args.output_dir / "selective_matched_l1_clip_action_impact.png"
    long_path = args.output_dir / "selective_matched_l1_clip_action_impact_per_condition.csv"
    corr_path = args.output_dir / "selective_matched_l1_clip_depth_corr.csv"
    _write_csv(mean_results, csv_path)
    _write_condition_csv(long_rows, long_path)
    _write_corr_csv(corr_rows, corr_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {long_path}")
    print(f"Wrote {corr_path}")
    condition_label = (
        f"mean over {len(sample_ids)} sample(s) × {len(noise_ids)} noise seed(s)"
    )
    _plot(
        mean_results,
        png,
        alpha=args.l1_remove,
        target_step=args.target_step,
        condition_label=condition_label,
    )
    print(f"Wrote {png}")
    ranked = sorted(mean_results, key=lambda row: row.metrics.total_rmse, reverse=True)
    print("\nTop-5 layers by mean total action RMSE:")
    for row in ranked[:5]:
        print(
            f"  {row.layer_name}  total={row.metrics.total_rmse:.4e}  "
            f"end={row.metrics.arm_endpoint_rmse:.4e}  "
            f"local={row.metrics.arm_local_rmse:.4e}  k={row.mean_k:.4g}"
        )
    print(f"\nMean over {len(conditions)} conditions:")
    matched._print_depth_trend(mean_results)
    print("\nPer-condition depth correlation:")
    for sample, seed, pearson, spearman in corr_rows:
        print(
            f"  sample={sample} noise={seed}  "
            f"Pearson={pearson:.3f}  Spearman={spearman:.3f}"
        )
    mean_pearson = _mean([row[2] for row in corr_rows])
    mean_spearman = _mean([row[3] for row in corr_rows])
    print(
        f"  mean of per-condition Pearson={mean_pearson:.3f}  "
        f"Spearman={mean_spearman:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
