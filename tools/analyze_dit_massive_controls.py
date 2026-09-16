#!/usr/bin/env python
r"""What channel-level massive outliers move in the generated action.

Massive = MAD-selected channels, |x| > μ+3σ, clipped all the way to μ+3σ
(no L1 matching). The script:

1. Counts massive tokens by DiT linear kind and by action-chunk token
   (pi05 tokens are the horizon steps).
2. Clips massive on all linears, then on each kind alone, at the last
   denoise step and across every step.
3. Scores the action delta vs the unclipped medoid, next to the existing
   same-obs noise band.

If down_proj (most massive mass) clip barely moves the action, those
spikes are not the control signal.

Example::

    CUDA_VISIBLE_DEVICES=5 uv run python \
      tools/analyze_dit_massive_controls.py \
      --checkpoint /data/share/pi05_libero_finetuned_v044 \
      --calibration-data ../calibration_data/libero_goal_30_7_demo.npz \
      --results-dir tools/img/dit_clip_vs_noise_band_pi05 \
      --output-dir tools/img/dit_massive_controls_pi05
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
_TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(_TOOLS))
sys.path.insert(0, str(_ROOT / "src"))

import analyze_dit_clip_vs_noise_band as band  # noqa: E402
import analyze_dit_matched_clip_action_impact as matched  # noqa: E402
import analyze_dit_outlier_action_detail as detail  # noqa: E402
import analyze_dit_outlier_action_detail_all as all_exp  # noqa: E402
from qvla.config import QVLAConfig  # noqa: E402

METRIC_NAMES = [field.name for field in fields(detail.ActionMetrics)]
KIND_GROUPS = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")
DIM_LABELS = ("x", "y", "z", "rot0", "rot1", "rot2", "gripper")
CLIP_TARGETS = (("last", "last denoise step"), ("all", "all denoise steps"))


@dataclass(frozen=True)
class OccupancyRow:
    sample: int
    step: int
    kind: str
    token: int
    n_massive: int


@dataclass(frozen=True)
class ClipGroupRow:
    sample: int
    group: str
    target: str
    n_sites: int
    n_clipped: int
    outlier_count: int
    mean_removed_l1_pct: float
    metrics: detail.ActionMetrics


def _kind_of(layer_name: str) -> str:
    return matched._layer_kind(layer_name)


def _token_massive_counts(
    live: torch.Tensor,
    *,
    std_k: float,
    skip_first_token: bool,
) -> torch.Tensor:
    _values, _selected, _bounds, over = detail._outlier_over_and_bounds(
        live, std_k, skip_first_token=skip_first_token
    )
    return over.sum(dim=1).to(torch.int64)


def _occupancy_from_activations(
    activations: dict[str, torch.Tensor],
    *,
    sample: int,
    std_k: float,
    skip_first_token: bool,
) -> list[OccupancyRow]:
    rows: list[OccupancyRow] = []
    by_kind: dict[tuple[int, str], torch.Tensor] = {}
    n_tokens = None
    for name, stacked in activations.items():
        if stacked.ndim != 3:
            raise ValueError(
                f"{name} captured shape {tuple(stacked.shape)}, expected (S,T,C)."
            )
        kind = _kind_of(name)
        if n_tokens is None:
            n_tokens = int(stacked.shape[1])
        elif int(stacked.shape[1]) != n_tokens:
            raise RuntimeError("Captured token counts differ across layers.")
        for step in range(int(stacked.shape[0])):
            counts = _token_massive_counts(
                stacked[step],
                std_k=std_k,
                skip_first_token=skip_first_token,
            ).cpu()
            key = (step, kind)
            if key not in by_kind:
                by_kind[key] = torch.zeros(n_tokens, dtype=torch.int64)
            by_kind[key] += counts
    for (step, kind), counts in sorted(by_kind.items()):
        for token, n_massive in enumerate(counts.tolist()):
            rows.append(
                OccupancyRow(
                    sample=sample,
                    step=int(step),
                    kind=kind,
                    token=int(token),
                    n_massive=int(n_massive),
                )
            )
    return rows


def _clip_kwargs(target: str, num_steps: int) -> dict:
    if target == "all":
        step = None
    elif target == "last":
        step = int(num_steps) - 1
    else:
        raise ValueError(f"Unknown clip target {target!r}.")
    return {
        "kind": "selective",
        "allow_empty": True,
        "target_step": step,
    }


def _summary(
    clip_rows: list[ClipGroupRow],
    noise_rows: list[band.NoiseRow],
    occ_rows: list[OccupancyRow],
) -> str:
    bands = band._bands_from_noise(noise_rows)
    total_n = sum(row.n_massive for row in occ_rows)
    lines = [
        "Massive = MAD-channel |x| > μ+3σ, clipped to μ+3σ. No L1 matching.",
        "Noise band is other noises vs the per-sample medoid.",
        "",
        "Massive occupancy by linear kind (summed over samples, steps, tokens):",
    ]
    for kind in KIND_GROUPS:
        n = sum(row.n_massive for row in occ_rows if row.kind == kind)
        share = 0.0 if total_n == 0 else 100.0 * n / total_n
        lines.append(f"  {kind:14} n={n}  share={share:.1f}%")
    lines.append("")
    lines.append("Clip RMSE vs noise band (total / endpoint / local / gripper):")
    groups = ["all", *KIND_GROUPS]
    for target, _label in CLIP_TARGETS:
        lines.append(f"  target={target}:")
        for group in groups:
            rows = [
                row
                for row in clip_rows
                if row.group == group and row.target == target
            ]
            if not rows:
                continue
            mean_total = float(np.mean([row.metrics.total_rmse for row in rows]))
            mean_end = float(np.mean([row.metrics.arm_endpoint_rmse for row in rows]))
            mean_local = float(np.mean([row.metrics.arm_local_rmse for row in rows]))
            mean_grip = float(np.mean([row.metrics.gripper_rmse for row in rows]))
            switches = float(np.mean([row.metrics.gripper_switch_shift for row in rows]))
            lines.append(
                f"    {group:14} total={mean_total:.4e} "
                f"({band._placement(mean_total, bands['total_rmse'])})  "
                f"end={mean_end:.4e} "
                f"({band._placement(mean_end, bands['arm_endpoint_rmse'])})  "
                f"local={mean_local:.4e} "
                f"({band._placement(mean_local, bands['arm_local_rmse'])})  "
                f"grip={mean_grip:.4e} "
                f"({band._placement(mean_grip, bands['gripper_rmse'])})  "
                f"switch={switches:.2f}"
            )
    return "\n".join(lines)


def _plot(
    clip_rows: list[ClipGroupRow],
    noise_rows: list[band.NoiseRow],
    occ_rows: list[OccupancyRow],
    output: Path,
    *,
    n_tokens: int,
) -> None:
    import matplotlib.pyplot as plt

    bands = band._bands_from_noise(noise_rows)
    groups = ["all", *KIND_GROUPS]
    colors = {
        "all": "#1f4e79",
        "qkv_proj": matched.KIND_COLORS["qkv_proj"],
        "o_proj": matched.KIND_COLORS["o_proj"],
        "gate_up_proj": matched.KIND_COLORS["gate_up_proj"],
        "down_proj": matched.KIND_COLORS["down_proj"],
    }
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.4), constrained_layout=True)

    metric_axes = (
        (axes[0, 0], "total_rmse", "Total action RMSE"),
        (axes[0, 1], "arm_endpoint_rmse", "Arm endpoint RMSE"),
    )
    xs = np.arange(len(groups))
    for axis, name, title in metric_axes:
        band_q = bands[name]
        axis.axhspan(band_q.minimum, band_q.maximum, color="0.85", label="noise min-max")
        axis.axhspan(band_q.p10, band_q.p90, color="0.70", label="noise p10-p90")
        width = 0.35
        for offset, (target, tlabel) in enumerate(CLIP_TARGETS):
            means = []
            for group in groups:
                rows = [
                    row
                    for row in clip_rows
                    if row.group == group and row.target == target
                ]
                means.append(
                    float(np.mean([getattr(row.metrics, name) for row in rows]))
                    if rows
                    else float("nan")
                )
            axis.bar(
                xs + (offset - 0.5) * width,
                means,
                width=width,
                color=["#1f4e79", "#C44E52"][offset],
                alpha=0.9 if offset == 0 else 0.7,
                label=tlabel,
            )
        axis.set_xticks(xs)
        axis.set_xticklabels(groups, rotation=20, ha="right")
        axis.set_ylabel("RMSE vs medoid")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=7, loc="upper left")

    kind_share = []
    total = sum(row.n_massive for row in occ_rows)
    for kind in KIND_GROUPS:
        n = sum(row.n_massive for row in occ_rows if row.kind == kind)
        kind_share.append(0.0 if total == 0 else 100.0 * n / total)
    axes[1, 0].bar(KIND_GROUPS, kind_share, color=[colors[k] for k in KIND_GROUPS])
    axes[1, 0].set_ylabel("share of massive tokens (%)")
    axes[1, 0].set_title("Where massive tokens sit")
    axes[1, 0].tick_params(axis="x", rotation=20)
    axes[1, 0].grid(axis="y", alpha=0.25)

    tokens = np.arange(n_tokens)
    for kind in KIND_GROUPS:
        ys = np.zeros(n_tokens, dtype=np.float64)
        grouped: dict[int, list[int]] = defaultdict(list)
        for row in occ_rows:
            if row.kind != kind:
                continue
            grouped[row.token].append(row.n_massive)
        for token in range(n_tokens):
            vals = grouped.get(token, [])
            ys[token] = float(np.mean(vals)) if vals else 0.0
        axes[1, 1].plot(tokens, ys, color=colors[kind], label=kind, linewidth=1.6)
    axes[1, 1].set_xlabel("action-chunk token (= horizon step on pi05)")
    axes[1, 1].set_ylabel("mean massive count")
    axes[1, 1].set_title("Massive count along the action chunk")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(fontsize=7)

    fig.suptitle(
        "What MAD-channel massive outliers move "
        "(clip to μ+3σ, unmatched; ref = per-sample medoid)",
        fontsize=11,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Failed to write {output}.")


def _write_occupancy(rows: list[OccupancyRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["sample", "step", "kind", "token", "n_massive"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sample": row.sample,
                    "step": row.step,
                    "kind": row.kind,
                    "token": row.token,
                    "n_massive": row.n_massive,
                }
            )


def _write_clip(rows: list[ClipGroupRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sample",
                "group",
                "target",
                "n_sites",
                "n_clipped",
                "outlier_count",
                "mean_removed_l1_pct",
                *METRIC_NAMES,
            ],
        )
        writer.writeheader()
        for row in rows:
            item = {
                "sample": row.sample,
                "group": row.group,
                "target": row.target,
                "n_sites": row.n_sites,
                "n_clipped": row.n_clipped,
                "outlier_count": row.outlier_count,
                "mean_removed_l1_pct": row.mean_removed_l1_pct,
            }
            for name in METRIC_NAMES:
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
    parser.add_argument("--samples", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--outlier-std-k", type=float, default=3.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--params-dtype", default="bfloat16")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("tools/img/dit_clip_vs_noise_band_pi05"),
        help="Existing noise-band run (medoids + noise CSV).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tools/img/dit_massive_controls_pi05"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(args.checkpoint)
    if not args.calibration_data.is_file():
        raise FileNotFoundError(args.calibration_data)
    sample_ids = detail._parse_nonneg_ints(args.samples, name="--samples")
    medoids = band._load_medoids(args.results_dir / "medoids.csv")
    noise_rows = band._load_noise_rows(args.results_dir / "noise_vs_medoid.csv")
    by_sample = {item.sample: item for item in medoids}
    missing = [sample for sample in sample_ids if sample not in by_sample]
    if missing:
        raise RuntimeError(f"No medoids for samples {missing} in {args.results_dir}.")
    if not noise_rows:
        raise RuntimeError(f"No noise-band rows in {args.results_dir}.")

    adapter = matched.adapter_from_args(args)
    model = adapter.build_model()
    model.eval()
    adapter.warmup_for_calibration(model)
    config = QVLAConfig.for_model_kind(args.model)
    all_layers = matched._dit_layers(model, None, config=config)
    runtime = matched.clip_runtime(adapter, config)
    skip_first_token = args.model == "groot_n17"
    groups = {"all": all_layers}
    for kind in KIND_GROUPS:
        groups[kind] = matched._dit_layers(model, kind, config=config)
        if not groups[kind]:
            raise RuntimeError(f"No DiT linears matched kind={kind}.")
    need = max(sample_ids) + 1
    batches = list(adapter.iter_calibration_batches(need))
    if len(batches) != need:
        raise RuntimeError(
            f"Requested samples {sample_ids}, calibration yielded {len(batches)}."
        )
    print(
        f"model={args.model}, layers={len(all_layers)}, samples={sample_ids}, "
        f"steps={runtime.num_steps}, clip=massive μ+3σ unmatched, "
        f"groups={list(groups)}"
    )

    occ_rows: list[OccupancyRow] = []
    clip_rows: list[ClipGroupRow] = []
    n_tokens = runtime.n_tokens
    for sample in sample_ids:
        medoid = by_sample[sample]
        request = matched._fixed_noise_request(
            adapter, batches[sample], runtime, noise_seed=medoid.noise
        )
        captured_actions, activations, _ = all_exp._run_all(
            adapter,
            request,
            all_layers,
            num_steps=runtime.num_steps,
            n_tokens=n_tokens,
            std_k=args.outlier_std_k,
            skip_first_token=skip_first_token,
        )
        assert activations is not None
        center = band._slice_action(
            captured_actions,
            horizon=runtime.action_horizon,
            action_dim=runtime.action_dim,
        )
        occ = _occupancy_from_activations(
            activations,
            sample=sample,
            std_k=args.outlier_std_k,
            skip_first_token=skip_first_token,
        )
        occ_rows.extend(occ)
        print(
            f"[occ] sample={sample} medoid_noise={medoid.noise} "
            f"massive={sum(row.n_massive for row in occ)}"
        )
        layer_index_all = {name: i for i, (name, _) in enumerate(all_layers)}
        for group, layers in groups.items():
            layer_index = {name: layer_index_all[name] for name, _ in layers}
            for target, _label in CLIP_TARGETS:
                changed, _, stats = all_exp._run_all(
                    adapter,
                    request,
                    layers,
                    num_steps=runtime.num_steps,
                    n_tokens=n_tokens,
                    std_k=args.outlier_std_k,
                    baseline_activations={
                        name: activations[name] for name, _ in layers
                    },
                    layer_index=layer_index,
                    skip_first_token=skip_first_token,
                    **_clip_kwargs(target, runtime.num_steps),
                )
                metrics = detail._action_metrics(
                    band._slice_action(
                        changed,
                        horizon=runtime.action_horizon,
                        action_dim=runtime.action_dim,
                    ),
                    center,
                )
                row = ClipGroupRow(
                    sample=sample,
                    group=group,
                    target=target,
                    n_sites=stats.n_sites,
                    n_clipped=stats.n_clipped,
                    outlier_count=stats.outlier_count,
                    mean_removed_l1_pct=stats.mean_removed_l1_pct,
                    metrics=metrics,
                )
                clip_rows.append(row)
                print(
                    f"  [{group} {target}] clip={row.n_clipped}/{row.n_sites} "
                    f"out={row.outlier_count} L1={row.mean_removed_l1_pct:.4f}% "
                    f"total={metrics.total_rmse:.3e} end={metrics.arm_endpoint_rmse:.3e} "
                    f"local={metrics.arm_local_rmse:.3e} grip={metrics.gripper_rmse:.3e}"
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_occ = args.output_dir / "massive_occupancy.csv"
    csv_clip = args.output_dir / "massive_by_group.csv"
    png = args.output_dir / "massive_controls.png"
    summary_path = args.output_dir / "summary.txt"
    _write_occupancy(occ_rows, csv_occ)
    _write_clip(clip_rows, csv_clip)
    _plot(clip_rows, noise_rows, occ_rows, png, n_tokens=n_tokens)
    text = _summary(clip_rows, noise_rows, occ_rows)
    summary_path.write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
    for path in (csv_occ, csv_clip, png, summary_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Failed to write {path}.")
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
