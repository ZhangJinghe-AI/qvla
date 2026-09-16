#!/usr/bin/env python
r"""Compare per-step MAD+μ+3σ clip error to same-observation noise spread.

Phase 1 generates unclipped actions for several noises on each observation.
The reference action for that observation is the **medoid**: the generated
chunk with the smallest mean RMSE to the other noises. That is an actual
model sample, so a later clip run can use the same noise. Using a medoid
avoids a lucky/unlucky seed sitting on the edge of the cloud and inflating
every distance.

The noise band is RMSE of the other noises to that medoid. Two unmatched
clips then run on the medoid noise, all per-step DiT linears, one denoise
step at a time (plus one all-steps run):

* massive: MAD channels, clamp every |x| > μ+3σ to μ+3σ
* normal: on those channels, clamp MAD-floor < |x| ≤ μ+3σ to the MAD-floor
  (all-token median+3×1.4826×MAD, same formula as MAD channel selection).
  Tokens above μ+3σ are left unchanged. No excess-L1 matching.

Clip RMSE is scored against the same medoid action.

Example:

    CUDA_VISIBLE_DEVICES=4 uv run python \
      tools/analyze_dit_clip_vs_noise_band.py \
      --checkpoint /data/share/pi05_libero_finetuned_v044 \
      --calibration-data ../calibration_data/libero_goal_30_7_demo.npz \
      --samples 0,1,2,3,4,5,6,7 --noise-seeds 0,1,2,3,4,5,6,7 \
      --output-dir tools/img/dit_clip_vs_noise_band_pi05
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
import analyze_dit_outlier_action_detail_all as all_exp  # noqa: E402
from qvla.config import QVLAConfig  # noqa: E402

ALL_STEPS_LABEL = "all"
METRIC_NAMES = [field.name for field in fields(detail.ActionMetrics)]
PLOT_METRICS = (
    ("total_rmse", "Total action RMSE"),
    ("arm_endpoint_rmse", "Arm endpoint RMSE"),
    ("arm_local_rmse", "Arm local RMSE"),
    ("gripper_rmse", "Gripper RMSE"),
)
CLIP_KINDS = ("massive", "normal")
CLIP_STYLE = {
    "massive": {
        "mean": "#1f4e79",
        "per": "#8fa4c4",
        "marker": "o",
        "all_marker": "D",
        "sample_marker": "x",
    },
    "normal": {
        "mean": "#C44E52",
        "per": "#E88B8E",
        "marker": "s",
        "all_marker": "P",
        "sample_marker": "+",
    },
}
CLIP_CSV_FIELDS = [
    "sample",
    "medoid_noise",
    "kind",
    "step",
    "n_sites",
    "n_clipped",
    "n_empty",
    "n_insufficient",
    "outlier_count",
    "selected_channels",
    "mean_removed_l1_pct",
] + METRIC_NAMES


@dataclass(frozen=True)
class QuantileBand:
    minimum: float
    p10: float
    median: float
    p90: float
    maximum: float
    n: int


@dataclass(frozen=True)
class MedoidChoice:
    sample: int
    noise: int
    index: int
    mean_rmse_to_others: float


@dataclass(frozen=True)
class NoiseRow:
    sample: int
    noise: int
    is_medoid: bool
    metrics: detail.ActionMetrics


@dataclass(frozen=True)
class ClipRow:
    sample: int
    medoid_noise: int
    kind: str
    step: int | None
    n_sites: int
    n_clipped: int
    n_empty: int
    n_insufficient: int
    outlier_count: int
    selected_channels: int
    mean_removed_l1_pct: float
    metrics: detail.ActionMetrics


def _total_rmse(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        raise ValueError(
            f"RMSE needs matching shapes, got {tuple(left.shape)} vs "
            f"{tuple(right.shape)}."
        )
    delta = left.to(torch.float64) - right.to(torch.float64)
    detail._finite(delta, "action delta")
    return float(delta.square().mean().sqrt().item())


def _pairwise_total_rmse(actions: torch.Tensor) -> torch.Tensor:
    """Return an (N, N) RMSE matrix for actions shaped (N, horizon, dims)."""
    if actions.ndim != 3 or int(actions.shape[0]) < 2:
        raise ValueError(
            "Need >=2 actions of shape (N, horizon, dims), got "
            f"{tuple(actions.shape)}."
        )
    n = int(actions.shape[0])
    out = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        for j in range(i + 1, n):
            value = _total_rmse(actions[i], actions[j])
            out[i, j] = value
            out[j, i] = value
    return out


def _medoid_index(
    actions: torch.Tensor,
    *,
    pairwise: torch.Tensor | None = None,
) -> int:
    """Index of the generated action with smallest mean RMSE to the others.

    Ties keep the smallest index.
    """
    if actions.ndim != 3 or int(actions.shape[0]) < 2:
        raise ValueError(
            "Need >=2 actions to choose a medoid, got "
            f"{tuple(actions.shape)}."
        )
    n = int(actions.shape[0])
    dist = pairwise if pairwise is not None else _pairwise_total_rmse(actions)
    if tuple(dist.shape) != (n, n):
        raise ValueError(
            f"pairwise RMSE shape {tuple(dist.shape)} != ({n}, {n})."
        )
    mean_to_others = dist.sum(dim=1) / (n - 1)
    order = np.lexsort(
        (
            np.arange(n, dtype=np.int64),
            mean_to_others.detach().cpu().numpy(),
        )
    )
    return int(order[0])


def _quantile_band(values: list[float] | np.ndarray) -> QuantileBand:
    if isinstance(values, np.ndarray):
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
    else:
        arr = np.asarray(list(values), dtype=np.float64)
    if arr.size < 1:
        raise ValueError("Need at least one value for a quantile band.")
    if not np.isfinite(arr).all():
        raise ValueError("Quantile band values must be finite.")
    return QuantileBand(
        minimum=float(arr.min()),
        p10=float(np.quantile(arr, 0.10)),
        median=float(np.quantile(arr, 0.50)),
        p90=float(np.quantile(arr, 0.90)),
        maximum=float(arr.max()),
        n=int(arr.size),
    )


def _metric_values(
    rows: list[NoiseRow] | list[ClipRow],
    name: str,
    *,
    skip_medoid: bool = False,
) -> list[float]:
    if name not in METRIC_NAMES:
        raise ValueError(f"Unknown action metric {name!r}.")
    out: list[float] = []
    for row in rows:
        if skip_medoid and isinstance(row, NoiseRow) and row.is_medoid:
            continue
        out.append(float(getattr(row.metrics, name)))
    return out


def _slice_action(
    actions: torch.Tensor,
    *,
    horizon: int,
    action_dim: int,
) -> torch.Tensor:
    if actions.ndim != 3 or actions.shape[:2] != (1, horizon):
        raise RuntimeError(
            f"Expected actions (1,{horizon},width), got {tuple(actions.shape)}."
        )
    if int(actions.shape[2]) < action_dim:
        raise RuntimeError(
            f"Action width {int(actions.shape[2])} < action_dim={action_dim}."
        )
    sliced = actions[0, :, :action_dim].detach().to(torch.float32).cpu()
    detail._finite(sliced, "scored action")
    return sliced


def _generate_unclipped(adapter, request) -> torch.Tensor:
    with torch.inference_mode():
        actions = adapter.engine.step(request)
    if not torch.is_tensor(actions):
        raise RuntimeError(f"Engine returned {type(actions)}, expected tensor.")
    out = actions.detach().to(torch.float32).cpu()
    detail._finite(out, "predicted actions")
    return out


def _zero_metrics() -> detail.ActionMetrics:
    return detail.ActionMetrics(
        total_rmse=0.0,
        arm_mean_shift_rmse=0.0,
        arm_endpoint_rmse=0.0,
        arm_net_disp_rmse=0.0,
        arm_local_rmse=0.0,
        arm_step_rmse=0.0,
        gripper_rmse=0.0,
        gripper_switch_shift=0.0,
    )


def _choose_medoid(
    sample: int,
    noise_ids: list[int],
    chunks: torch.Tensor,
) -> MedoidChoice:
    pairwise = _pairwise_total_rmse(chunks)
    index = _medoid_index(chunks, pairwise=pairwise)
    n = int(chunks.shape[0])
    return MedoidChoice(
        sample=int(sample),
        noise=int(noise_ids[index]),
        index=index,
        mean_rmse_to_others=float((pairwise[index].sum() / (n - 1)).item()),
    )


def _noise_rows(
    sample: int,
    noise_ids: list[int],
    chunks: torch.Tensor,
    medoid: MedoidChoice,
) -> list[NoiseRow]:
    center = chunks[medoid.index]
    rows: list[NoiseRow] = []
    for i, noise in enumerate(noise_ids):
        if i == medoid.index:
            metrics = _zero_metrics()
        else:
            metrics = detail._action_metrics(chunks[i], center)
        rows.append(
            NoiseRow(
                sample=int(sample),
                noise=int(noise),
                is_medoid=i == medoid.index,
                metrics=metrics,
            )
        )
    return rows


def _clip_row(
    sample: int,
    medoid_noise: int,
    kind: str,
    step: int | None,
    stats: all_exp.SiteStats,
    metrics: detail.ActionMetrics,
) -> ClipRow:
    if kind not in CLIP_KINDS:
        raise ValueError(f"Unknown clip kind {kind!r}.")
    removed = (
        stats.mean_bulk_removed_l1_pct
        if kind == "normal"
        else stats.mean_removed_l1_pct
    )
    return ClipRow(
        sample=int(sample),
        medoid_noise=int(medoid_noise),
        kind=kind,
        step=step,
        n_sites=int(stats.n_sites),
        n_clipped=int(stats.n_clipped),
        n_empty=int(stats.n_empty),
        n_insufficient=int(stats.n_ch_insufficient),
        outlier_count=int(stats.outlier_count),
        selected_channels=int(stats.selected_channels),
        mean_removed_l1_pct=float(removed),
        metrics=metrics,
    )


def _rows_of_kind(rows: list[ClipRow], kind: str) -> list[ClipRow]:
    return [row for row in rows if row.kind == kind]


def _present_kinds(rows: list[ClipRow]) -> tuple[str, ...]:
    seen = {row.kind for row in rows}
    return tuple(kind for kind in CLIP_KINDS if kind in seen)


def _matched_clip_kwargs(kind: str, step: int | None) -> dict:
    if kind not in CLIP_KINDS:
        raise ValueError(f"Unknown clip kind {kind!r}.")
    kwargs: dict = {
        "kind": "selective" if kind == "massive" else "random",
        "allow_empty": True,
        "target_step": step,
    }
    if kind == "normal":
        kwargs.update(
            normal_ref_l1=True,
            normal_ref_mode="channel_mad",
            normal_full_clip=True,
            bulk_seed=1000 if step is None else 1000 + int(step) * 17,
        )
    return kwargs


def _run_clip_on_medoid(
    adapter,
    request,
    layers: list[tuple[str, torch.nn.Module]],
    *,
    runtime: matched.ClipRuntime,
    std_k: float,
    sample: int,
    medoid_noise: int,
    phase1_action: torch.Tensor,
    skip_first_token: bool,
) -> tuple[torch.Tensor, list[ClipRow]]:
    num_steps = runtime.num_steps
    n_tokens = runtime.n_tokens
    horizon = runtime.action_horizon
    action_dim = runtime.action_dim
    layer_index = {name: index for index, (name, _layer) in enumerate(layers)}

    captured_actions, baseline_activations, _ = all_exp._run_all(
        adapter,
        request,
        layers,
        num_steps=num_steps,
        n_tokens=n_tokens,
        std_k=std_k,
        skip_first_token=skip_first_token,
    )
    assert baseline_activations is not None
    captured = _slice_action(
        captured_actions, horizon=horizon, action_dim=action_dim
    )
    recapture_rmse = _total_rmse(captured, phase1_action)
    if recapture_rmse > 0.0:
        print(
            f"  recapture vs phase-1 medoid RMSE={recapture_rmse:.6e}; "
            "clip and the noise band both use the recaptured action."
        )
    center = captured

    identity_actions, _, _ = all_exp._run_all(
        adapter,
        request,
        layers,
        num_steps=num_steps,
        n_tokens=n_tokens,
        std_k=std_k,
        baseline_activations=baseline_activations,
        identity_writeback=True,
        skip_first_token=skip_first_token,
    )
    detail._assert_actions_equal(
        identity_actions,
        captured_actions,
        name="identity write-back",
    )
    print("  identity write-back: actions unchanged")

    restore_actions, _, _ = all_exp._run_all(
        adapter,
        request,
        layers,
        num_steps=num_steps,
        n_tokens=n_tokens,
        std_k=std_k,
        baseline_activations=baseline_activations,
        restore_after_clip=True,
        skip_first_token=skip_first_token,
    )
    detail._assert_actions_equal(
        restore_actions,
        captured_actions,
        name="clip-then-restore write-back",
    )
    print("  clip-then-restore write-back: actions unchanged")

    def run_kind(kind: str, step: int | None) -> ClipRow:
        changed, _, stats = all_exp._run_all(
            adapter,
            request,
            layers,
            num_steps=num_steps,
            n_tokens=n_tokens,
            std_k=std_k,
            baseline_activations=baseline_activations,
            layer_index=layer_index,
            skip_first_token=skip_first_token,
            **_matched_clip_kwargs(kind, step),
        )
        metrics = detail._action_metrics(
            _slice_action(changed, horizon=horizon, action_dim=action_dim),
            center,
        )
        row = _clip_row(sample, medoid_noise, kind, step, stats, metrics)
        print(
            f"  [{kind} step={_step_key(step)}] "
            f"clip={row.n_clipped}/{row.n_sites} empty={row.n_empty} "
            f"insuff={row.n_insufficient} "
            f"L1={row.mean_removed_l1_pct:.4f}% "
            f"total={metrics.total_rmse:.3e} "
            f"end={metrics.arm_endpoint_rmse:.3e} "
            f"local={metrics.arm_local_rmse:.3e} "
            f"grip={metrics.gripper_rmse:.3e}"
        )
        return row

    rows: list[ClipRow] = []
    for step in range(num_steps):
        for kind in CLIP_KINDS:
            rows.append(run_kind(kind, step))
    for kind in CLIP_KINDS:
        rows.append(run_kind(kind, None))
    return center, rows


def _bands_from_noise(rows: list[NoiseRow]) -> dict[str, QuantileBand]:
    return {
        name: _quantile_band(_metric_values(rows, name, skip_medoid=True))
        for name in METRIC_NAMES
    }


def _step_key(step: int | None) -> str:
    return ALL_STEPS_LABEL if step is None else str(int(step))


def _clip_by_step(rows: list[ClipRow]) -> dict[int | None, list[ClipRow]]:
    grouped: dict[int | None, list[ClipRow]] = {}
    for row in rows:
        grouped.setdefault(row.step, []).append(row)
    return grouped


def _mean_metric(rows: list[ClipRow], name: str) -> float:
    return float(detail._mean(_metric_values(rows, name)))


def _placement(value: float, band: QuantileBand) -> str:
    if value <= band.p10:
        return "below_p10"
    if value <= band.p90:
        return "inside_p10_p90"
    if value <= band.maximum:
        return "between_p90_and_max"
    return "above_max"


def _summary(
    medoids: list[MedoidChoice],
    noise_rows: list[NoiseRow],
    clip_rows: list[ClipRow],
) -> str:
    noise_bands = _bands_from_noise(noise_rows)
    total_band = noise_bands["total_rmse"]
    end_band = noise_bands["arm_endpoint_rmse"]
    lines = [
        "Reference action: per-sample medoid (min mean RMSE to other noises).",
        "Noise band: other noises vs that medoid; clip error vs the same action.",
        (
            "Clip pair: channel_mad full clip, unmatched energy. "
            "massive = clamp |x| > μ+3σ to μ+3σ on MAD channels; "
            "normal = clamp MAD-floor < |x| ≤ μ+3σ to MAD-floor "
            "(all-token median+3×1.4826×MAD); massive tips stay on the "
            "normal side. No excess-L1 matching."
        ),
        (
            f"Noise vs medoid total RMSE: n={total_band.n}  "
            f"min={total_band.minimum:.4e}  p10={total_band.p10:.4e}  "
            f"median={total_band.median:.4e}  p90={total_band.p90:.4e}  "
            f"max={total_band.maximum:.4e}"
        ),
        (
            f"Noise vs medoid endpoint RMSE: "
            f"p10={end_band.p10:.4e}  median={end_band.median:.4e}  "
            f"p90={end_band.p90:.4e}  max={end_band.maximum:.4e}"
        ),
        "",
        "Medoids:",
    ]
    for item in medoids:
        lines.append(
            f"  sample={item.sample} noise={item.noise}  "
            f"mean_to_others={item.mean_rmse_to_others:.4e}"
        )
    lines.append("")
    lines.append("Clip vs noise band (total RMSE, pooled p10-p90 / max):")
    kinds = _present_kinds(clip_rows)
    if not kinds:
        raise RuntimeError("No clip rows to summarize.")
    for kind in kinds:
        kind_rows = _rows_of_kind(clip_rows, kind)
        grouped = _clip_by_step(kind_rows)
        steps = sorted((step for step in grouped if step is not None))
        ordered = [*steps, None]
        lines.append(f"  {kind}:")
        for step in ordered:
            rows = grouped[step]
            mean_total = _mean_metric(rows, "total_rmse")
            mean_end = _mean_metric(rows, "arm_endpoint_rmse")
            mean_sites = detail._mean([float(row.n_clipped) for row in rows])
            mean_insuff = detail._mean([float(row.n_insufficient) for row in rows])
            n_own_max = 0
            n_global_p90 = 0
            n_global_max = 0
            for row in rows:
                own = [
                    item.metrics.total_rmse
                    for item in noise_rows
                    if item.sample == row.sample and not item.is_medoid
                ]
                own_max = max(own) if own else float("nan")
                if row.metrics.total_rmse <= own_max:
                    n_own_max += 1
                if row.metrics.total_rmse <= total_band.p90:
                    n_global_p90 += 1
                if row.metrics.total_rmse <= total_band.maximum:
                    n_global_max += 1
            label = _step_key(step)
            lines.append(
                f"    step={label:>3}  mean_total={mean_total:.4e} "
                f"({_placement(mean_total, total_band)})  "
                f"mean_end={mean_end:.4e}  "
                f"mean_clipped={mean_sites:.1f}  "
                f"mean_insuff={mean_insuff:.1f}  "
                f"within_sample_max={n_own_max}/{len(rows)}  "
                f"within_global_p90={n_global_p90}/{len(rows)}  "
                f"within_global_max={n_global_max}/{len(rows)}"
            )
    return "\n".join(lines)


def _plot(
    clip_rows: list[ClipRow],
    noise_rows: list[NoiseRow],
    output: Path,
    *,
    num_steps: int,
) -> None:
    import matplotlib.pyplot as plt

    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}.")
    kinds = _present_kinds(clip_rows)
    if not kinds:
        raise RuntimeError("No clip rows to plot.")
    noise_bands = _bands_from_noise(noise_rows)
    samples = sorted({row.sample for row in clip_rows})
    xs = np.arange(num_steps, dtype=np.float64)
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.6), constrained_layout=True)
    for axis, (name, title) in zip(axes.reshape(-1), PLOT_METRICS):
        band = noise_bands[name]
        axis.axhspan(
            band.minimum,
            band.maximum,
            color="0.85",
            label=f"noise min-max (n={band.n})",
        )
        axis.axhspan(
            band.p10,
            band.p90,
            color="0.70",
            label="noise p10-p90 vs medoid",
        )
        for kind in kinds:
            style = CLIP_STYLE[kind]
            kind_rows = _rows_of_kind(clip_rows, kind)
            all_x: list[float] = []
            all_y: list[float] = []
            for sample in samples:
                step_rows = sorted(
                    (
                        row
                        for row in kind_rows
                        if row.sample == sample and row.step is not None
                    ),
                    key=lambda row: int(row.step),
                )
                if [int(row.step) for row in step_rows] != list(range(num_steps)):
                    raise RuntimeError(
                        f"sample {sample} kind={kind} is missing a per-step clip result."
                    )
                ys = [float(getattr(row.metrics, name)) for row in step_rows]
                axis.plot(
                    xs,
                    ys,
                    color=style["per"],
                    linewidth=0.9,
                    alpha=0.85,
                    label=f"{kind} per sample" if sample == samples[0] else None,
                )
                all_rows = [
                    row
                    for row in kind_rows
                    if row.sample == sample and row.step is None
                ]
                if len(all_rows) != 1:
                    raise RuntimeError(
                        f"sample {sample} kind={kind} needs exactly one "
                        "all-steps clip row."
                    )
                all_x.append(float(num_steps))
                all_y.append(float(getattr(all_rows[0].metrics, name)))
            mean_curve = [
                _mean_metric(
                    [row for row in kind_rows if row.step == step],
                    name,
                )
                for step in range(num_steps)
            ]
            axis.plot(
                xs,
                mean_curve,
                color=style["mean"],
                marker=style["marker"],
                linewidth=2.0,
                label=f"{kind} mean",
            )
            axis.scatter(
                all_x,
                all_y,
                color=style["per"],
                marker=style["sample_marker"],
                s=28,
                zorder=3,
                label=f"{kind} all (per sample)",
            )
            axis.scatter(
                [float(num_steps)],
                [float(np.mean(all_y))],
                color=style["mean"],
                marker=style["all_marker"],
                s=36,
                zorder=4,
                label=f"{kind} all (mean)",
            )
        axis.set_title(title)
        axis.set_xlabel("denoise step")
        axis.set_ylabel("RMSE vs medoid action")
        axis.set_xticks([*range(num_steps), num_steps])
        axis.set_xticklabels([str(step) for step in range(num_steps)] + [ALL_STEPS_LABEL])
        axis.grid(alpha=0.25)
        axis.legend(fontsize=6.5, loc="upper left", ncol=2)
    fig.suptitle(
        "Massive vs channel-MAD-band normal clip vs same-obs noise "
        "(full clip, unmatched energy; ref = per-sample medoid)",
        fontsize=11,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Failed to write {output}.")


def _write_csv(rows: list[object], output: Path, *, fieldnames: list[str]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if isinstance(row, MedoidChoice):
                writer.writerow(
                    {
                        "sample": row.sample,
                        "medoid_noise": row.noise,
                        "medoid_index": row.index,
                        "mean_rmse_to_others": row.mean_rmse_to_others,
                    }
                )
            elif isinstance(row, NoiseRow):
                item = {
                    "sample": row.sample,
                    "noise": row.noise,
                    "is_medoid": int(row.is_medoid),
                }
                for name in METRIC_NAMES:
                    item[name] = getattr(row.metrics, name)
                writer.writerow(item)
            elif isinstance(row, ClipRow):
                item = {
                    "sample": row.sample,
                    "medoid_noise": row.medoid_noise,
                    "kind": row.kind,
                    "step": _step_key(row.step),
                    "n_sites": row.n_sites,
                    "n_clipped": row.n_clipped,
                    "n_empty": row.n_empty,
                    "n_insufficient": row.n_insufficient,
                    "outlier_count": row.outlier_count,
                    "selected_channels": row.selected_channels,
                    "mean_removed_l1_pct": row.mean_removed_l1_pct,
                }
                for name in METRIC_NAMES:
                    item[name] = getattr(row.metrics, name)
                writer.writerow(item)
            else:
                raise TypeError(f"Unsupported CSV row type {type(row)!r}.")


def _metrics_from_mapping(row: dict[str, str]) -> detail.ActionMetrics:
    return detail.ActionMetrics(
        **{name: float(row[name]) for name in METRIC_NAMES}
    )


def _parse_step_cell(text: str) -> int | None:
    value = str(text).strip()
    if value == ALL_STEPS_LABEL:
        return None
    return int(value)


def _load_noise_rows(path: Path) -> list[NoiseRow]:
    with path.open(newline="", encoding="utf-8") as file:
        return [
            NoiseRow(
                sample=int(row["sample"]),
                noise=int(row["noise"]),
                is_medoid=bool(int(row["is_medoid"])),
                metrics=_metrics_from_mapping(row),
            )
            for row in csv.DictReader(file)
        ]


def _load_clip_rows(path: Path) -> list[ClipRow]:
    with path.open(newline="", encoding="utf-8") as file:
        return [
            ClipRow(
                sample=int(row["sample"]),
                medoid_noise=int(row["medoid_noise"]),
                kind=str(row["kind"] if row.get("kind") else "massive"),
                step=_parse_step_cell(row["step"]),
                n_sites=int(row["n_sites"]),
                n_clipped=int(row["n_clipped"]),
                n_empty=int(row["n_empty"]),
                n_insufficient=int(row["n_insufficient"] if row.get("n_insufficient") else 0),
                outlier_count=int(row["outlier_count"]),
                selected_channels=int(row["selected_channels"]),
                mean_removed_l1_pct=float(row["mean_removed_l1_pct"]),
                metrics=_metrics_from_mapping(row),
            )
            for row in csv.DictReader(file)
        ]


def _load_medoids(path: Path) -> list[MedoidChoice]:
    with path.open(newline="", encoding="utf-8") as file:
        return [
            MedoidChoice(
                sample=int(row["sample"]),
                noise=int(row["medoid_noise"]),
                index=int(row["medoid_index"]),
                mean_rmse_to_others=float(row["mean_rmse_to_others"]),
            )
            for row in csv.DictReader(file)
        ]


def _num_steps_from_clip(rows: list[ClipRow]) -> int:
    steps = [int(row.step) for row in rows if row.step is not None]
    if not steps:
        raise RuntimeError("No per-step clip rows.")
    return max(steps) + 1


def _replot_from_dir(output_dir: Path) -> int:
    if not output_dir.is_dir():
        raise FileNotFoundError(output_dir)
    noise_rows = _load_noise_rows(output_dir / "noise_vs_medoid.csv")
    clip_rows = _load_clip_rows(output_dir / "clip_by_step.csv")
    medoids = _load_medoids(output_dir / "medoids.csv")
    if not noise_rows or not clip_rows or not medoids:
        raise RuntimeError(f"Incomplete results in {output_dir}.")
    png = output_dir / "clip_vs_noise_band.png"
    summary_path = output_dir / "summary.txt"
    _plot(clip_rows, noise_rows, png, num_steps=_num_steps_from_clip(clip_rows))
    text = _summary(medoids, noise_rows, clip_rows)
    summary_path.write_text(text + "\n", encoding="utf-8")
    csv_medoid = output_dir / "medoids.csv"
    _write_csv(
        medoids,
        csv_medoid,
        fieldnames=[
            "sample",
            "medoid_noise",
            "medoid_index",
            "mean_rmse_to_others",
        ],
    )
    print(text)
    print(f"Wrote {png}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {csv_medoid}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--calibration-data", type=Path, default=None)
    matched.add_model_cli(parser)
    parser.add_argument("--layer-regex", default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--samples", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--noise-seeds", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--outlier-std-k", type=float, default=3.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--params-dtype", default="bfloat16")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tools/img/dit_clip_vs_noise_band_pi05"),
    )
    parser.add_argument(
        "--replot-from",
        type=Path,
        default=None,
        help="Rebuild the figure and summary from a previous output directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.replot_from is not None:
        return _replot_from_dir(args.replot_from)
    if args.checkpoint is None or args.calibration_data is None:
        raise ValueError("Need --checkpoint and --calibration-data, or --replot-from.")
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(args.checkpoint)
    if not args.calibration_data.is_file():
        raise FileNotFoundError(args.calibration_data)
    if args.sample_index < 0:
        raise ValueError("--sample-index must be >= 0.")
    if args.noise_seed < 0:
        raise ValueError("--noise-seed must be >= 0.")
    if args.outlier_std_k <= 0.0:
        raise ValueError("--outlier-std-k must be > 0.")
    sample_ids = detail._parse_nonneg_ints(args.samples, name="--samples")
    noise_ids = detail._parse_nonneg_ints(args.noise_seeds, name="--noise-seeds")
    if len(noise_ids) < 2:
        raise ValueError("Need at least two --noise-seeds to form a medoid and a band.")

    adapter = matched.adapter_from_args(args)
    model = adapter.build_model()
    model.eval()
    adapter.warmup_for_calibration(model)
    config = QVLAConfig.for_model_kind(args.model)
    layers = matched._dit_layers(model, args.layer_regex, config=config)
    runtime = matched.clip_runtime(adapter, config)
    need = max(sample_ids) + 1
    batches = list(adapter.iter_calibration_batches(need))
    if len(batches) != need:
        raise RuntimeError(
            f"Requested samples {sample_ids}, calibration yielded "
            f"{len(batches)} samples."
        )
    skip_first_token = args.model == "groot_n17"
    print(
        f"model={args.model}, layers={len(layers)}, samples={sample_ids}, "
        f"noise_seeds={noise_ids}, steps={runtime.num_steps}, "
        f"horizon={runtime.action_horizon}, n_tokens={runtime.n_tokens}, "
        f"action_dim={runtime.action_dim}, std_k={args.outlier_std_k}, "
        f"skip_first_token={skip_first_token}, reference=per-sample medoid, "
        f"clip=channel_mad full unmatched (massive+normal)"
    )

    chunks_by_sample: dict[int, torch.Tensor] = {}
    medoids: list[MedoidChoice] = []
    noise_rows: list[NoiseRow] = []
    for sample in sample_ids:
        scored: list[torch.Tensor] = []
        for seed in noise_ids:
            request = matched._fixed_noise_request(
                adapter, batches[sample], runtime, noise_seed=seed
            )
            actions = _generate_unclipped(adapter, request)
            scored.append(
                _slice_action(
                    actions,
                    horizon=runtime.action_horizon,
                    action_dim=runtime.action_dim,
                )
            )
            print(
                f"[phase1] sample={sample} noise={seed} "
                f"action={tuple(scored[-1].shape)}"
            )
        chunks = torch.stack(scored, dim=0)
        medoid = _choose_medoid(sample, noise_ids, chunks)
        medoids.append(medoid)
        noise_rows.extend(_noise_rows(sample, noise_ids, chunks, medoid))
        chunks_by_sample[sample] = chunks
        print(
            f"[medoid] sample={sample} noise={medoid.noise} "
            f"mean_to_others={medoid.mean_rmse_to_others:.4e}"
        )

    clip_rows: list[ClipRow] = []
    for medoid in medoids:
        sample = medoid.sample
        request = matched._fixed_noise_request(
            adapter, batches[sample], runtime, noise_seed=medoid.noise
        )
        print(
            f"\n=== clip sample={sample} medoid_noise={medoid.noise} ==="
        )
        center, rows = _run_clip_on_medoid(
            adapter,
            request,
            layers,
            runtime=runtime,
            std_k=args.outlier_std_k,
            sample=sample,
            medoid_noise=medoid.noise,
            phase1_action=chunks_by_sample[sample][medoid.index],
            skip_first_token=skip_first_token,
        )
        chunks_by_sample[sample] = chunks_by_sample[sample].clone()
        chunks_by_sample[sample][medoid.index] = center
        noise_rows = [
            row
            for row in noise_rows
            if row.sample != sample
        ]
        noise_rows.extend(
            _noise_rows(
                sample,
                noise_ids,
                chunks_by_sample[sample],
                medoid,
            )
        )
        clip_rows.extend(rows)

    text = _summary(medoids, noise_rows, clip_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_noise = args.output_dir / "noise_vs_medoid.csv"
    csv_clip = args.output_dir / "clip_by_step.csv"
    csv_medoid = args.output_dir / "medoids.csv"
    png = args.output_dir / "clip_vs_noise_band.png"
    summary_path = args.output_dir / "summary.txt"
    npz_path = args.output_dir / "actions.npz"
    _write_csv(
        medoids,
        csv_medoid,
        fieldnames=[
            "sample",
            "medoid_noise",
            "medoid_index",
            "mean_rmse_to_others",
        ],
    )
    _write_csv(
        noise_rows,
        csv_noise,
        fieldnames=["sample", "noise", "is_medoid", *METRIC_NAMES],
    )
    _write_csv(
        clip_rows,
        csv_clip,
        fieldnames=CLIP_CSV_FIELDS,
    )
    _plot(
        clip_rows,
        noise_rows,
        png,
        num_steps=runtime.num_steps,
    )
    summary_path.write_text(text + "\n", encoding="utf-8")
    np.savez(
        npz_path,
        samples=np.asarray(sample_ids, dtype=np.int64),
        noises=np.asarray(noise_ids, dtype=np.int64),
        medoid_noises=np.asarray([item.noise for item in medoids], dtype=np.int64),
        actions=np.stack(
            [chunks_by_sample[sample].numpy() for sample in sample_ids],
            axis=0,
        ),
    )
    print("\n" + text)
    for path in (csv_medoid, csv_noise, csv_clip, png, summary_path, npz_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Failed to write {path}.")
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
