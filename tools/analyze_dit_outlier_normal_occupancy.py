#!/usr/bin/env python
r"""Occupancy diagnostic for unselected-channel μ+3σ normal-outliers.

Massive: MAD-selected channels, ``|x| > μ_j+3σ_j``.
Normal: same channels, not massive, but still above the unselected-channel
``μ+3σ``. Selected leftover below that ruler is selected bulk.

Capture only; no clip. Also reports unselected p95 occupancy as a looser
sensitivity check. If the mean+3σ class is empty, do not treat leftover L1
matching as a substitute definition.

Example::

    CUDA_VISIBLE_DEVICES=5 HF_ENDPOINT=https://hf-mirror.com uv run python \
      tools/analyze_dit_outlier_normal_occupancy.py \
      --checkpoint /data/share/pi05_libero_finetuned_v044 \
      --calibration-data ../calibration_data/libero_goal_30_7_demo.npz \
      --samples 0,1,2,3 --noise-seeds 0,1 \
      --output-dir tools/img/dit_outlier_normal_occupancy
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
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


@dataclass(frozen=True)
class SiteRow:
    sample: int
    noise: int
    step: int
    layer: str
    n_selected_ch: int
    n_massive: int
    n_normal: int
    n_selected_bulk: int
    n_normal_p95: int
    l1_massive: float
    l1_normal: float
    l1_selected_bulk: float
    excess_massive: float
    excess_normal: float
    t_ref: float
    t_p95: float


def _p95_count(values: torch.Tensor, split: detail.NormalOutlierSplit) -> tuple[int, float]:
    ref_vals = values[split.ref_mask]
    if int(ref_vals.numel()) < 2:
        return 0, float("nan")
    t_p95 = float(torch.quantile(ref_vals, 0.95).item())
    leftover = split.selected_bulk | split.normal
    n_p95 = int((leftover & (values > t_p95)).sum().item())
    return n_p95, t_p95


def _rows_for_activations(
    activations: dict[str, torch.Tensor],
    *,
    sample: int,
    noise: int,
    std_k: float,
    skip_first_token: bool,
    ref_mode: str,
) -> list[SiteRow]:
    rows: list[SiteRow] = []
    for name, stacked in activations.items():
        if stacked.ndim != 3:
            raise ValueError(f"{name} captured shape {tuple(stacked.shape)}, expected (S,T,C).")
        for step in range(int(stacked.shape[0])):
            live = stacked[step]
            split = detail._normal_outlier_split(
                live,
                std_k=std_k,
                skip_first_token=skip_first_token,
                ref_mode=ref_mode,
            )
            values = live.abs().to(torch.float32)
            if ref_mode == "unselected_std":
                n_p95, t_p95 = _p95_count(values, split)
            else:
                n_p95, t_p95 = 0, float("nan")
            t_ref_t = split.t_ref
            if t_ref_t.ndim == 0:
                t_ref_f = float(t_ref_t.item())
            else:
                finite = torch.isfinite(t_ref_t)
                t_ref_f = (
                    float(t_ref_t[finite].mean().item())
                    if bool(finite.any().item())
                    else float("nan")
                )
            rows.append(
                SiteRow(
                    sample=sample,
                    noise=noise,
                    step=step,
                    layer=name,
                    n_selected_ch=int(split.n_selected_ch.item()),
                    n_massive=int(split.n_massive.item()),
                    n_normal=int(split.n_normal.item()),
                    n_selected_bulk=int(split.n_selected_bulk.item()),
                    n_normal_p95=n_p95,
                    l1_massive=float(split.l1_massive.item()),
                    l1_normal=float(split.l1_normal.item()),
                    l1_selected_bulk=float(split.l1_selected_bulk.item()),
                    excess_massive=float(split.excess_massive.item()),
                    excess_normal=float(split.excess_normal.item()),
                    t_ref=t_ref_f,
                    t_p95=t_p95,
                )
            )
    return rows


def _plot_occupancy(rows: list[SiteRow], output: Path) -> None:
    import matplotlib.pyplot as plt

    grouped: dict[int, list[SiteRow]] = defaultdict(list)
    for row in rows:
        grouped[row.step].append(row)
    steps = sorted(grouped)

    def _mean(key: str) -> list[float]:
        return [float(np.mean([getattr(r, key) for r in grouped[s]])) for s in steps]

    xs = np.asarray(steps, dtype=np.float64)
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.2), constrained_layout=True)
    axes[0].plot(xs, _mean("n_massive"), "o-", label="massive")
    axes[0].plot(xs, _mean("n_normal"), "s-", label="normal")
    axes[0].plot(xs, _mean("n_selected_bulk"), "^--", label="selected bulk")
    if any(row.n_normal_p95 > 0 for row in rows):
        axes[0].plot(xs, _mean("n_normal_p95"), "d:", label="selected leftover > unsel p95")
    axes[0].set_title("Tokens / site")
    axes[0].set_ylabel("mean count")
    axes[1].plot(xs, _mean("excess_massive"), "o-", label="massive excess")
    axes[1].plot(xs, _mean("excess_normal"), "s-", label="normal excess to t_ref")
    axes[1].plot(xs, _mean("l1_normal"), "s--", label="normal L1")
    axes[1].plot(xs, _mean("l1_selected_bulk"), "^--", label="selected-bulk L1")
    axes[1].set_title("Energy / site")
    axes[1].set_ylabel("mean L1")
    leftover_n = []
    frac_n = []
    for step in steps:
        block = grouped[step]
        left = np.asarray([r.n_normal + r.n_selected_bulk for r in block], dtype=np.float64)
        frac = np.asarray(
            [
                r.n_normal / (r.n_normal + r.n_selected_bulk)
                if (r.n_normal + r.n_selected_bulk) > 0
                else np.nan
                for r in block
            ],
            dtype=np.float64,
        )
        leftover_n.append(float(np.mean(left)))
        frac_n.append(float(np.nanmean(frac)))
    axes[2].plot(xs, frac_n, "s-", label="normal / selected leftover")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_title("Share of selected leftover")
    axes[2].set_ylabel("fraction")
    for ax in axes:
        ax.set_xlabel("denoise step")
        ax.set_xticks(list(steps))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle("Normal-outlier occupancy", fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Failed to write {output}.")
    del leftover_n


def _write_csv(rows: list[SiteRow], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample",
        "noise",
        "step",
        "layer",
        "n_selected_ch",
        "n_massive",
        "n_normal",
        "n_selected_bulk",
        "n_normal_p95",
        "l1_massive",
        "l1_normal",
        "l1_selected_bulk",
        "excess_massive",
        "excess_normal",
        "t_ref",
        "t_p95",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: getattr(row, name) for name in fieldnames})


def _verdict(rows: list[SiteRow]) -> list[str]:
    if not rows:
        raise ValueError("No occupancy rows.")
    n_sites = len(rows)
    n_with_sel = sum(1 for r in rows if r.n_selected_ch > 0)
    n_with_mass = sum(1 for r in rows if r.n_massive > 0)
    n_with_normal = sum(1 for r in rows if r.n_normal > 0)
    n_with_p95 = sum(1 for r in rows if r.n_normal_p95 > 0)
    mean_mass = float(np.mean([r.n_massive for r in rows]))
    mean_normal = float(np.mean([r.n_normal for r in rows]))
    mean_bulk = float(np.mean([r.n_selected_bulk for r in rows]))
    mean_p95 = float(np.mean([r.n_normal_p95 for r in rows]))
    mean_ex_m = float(np.mean([r.excess_massive for r in rows]))
    mean_ex_n = float(np.mean([r.excess_normal for r in rows]))
    leftover = np.asarray([r.n_normal + r.n_selected_bulk for r in rows], dtype=np.float64)
    frac = np.asarray(
        [
            r.n_normal / (r.n_normal + r.n_selected_bulk)
            if (r.n_normal + r.n_selected_bulk) > 0
            else np.nan
            for r in rows
        ],
        dtype=np.float64,
    )
    mean_frac = float(np.nanmean(frac)) if np.isfinite(frac).any() else float("nan")
    if n_with_normal / max(n_sites, 1) < 0.05 or mean_normal < 1.0:
        kind = (
            "EMPTY: unselected μ+3σ does not mark a second class on MAD channels. "
            "Do not clip this as normal-outlier."
        )
    elif np.isfinite(mean_frac) and mean_frac >= 0.5:
        kind = (
            "CHANNEL_BODY: most leftover tokens on MAD channels sit above unselected "
            "μ+3σ. This is a persistent outlier-channel body, not a sparse shoulder."
        )
    else:
        kind = (
            "TOKEN_CLASS: a minority of MAD-channel leftover exceeds unselected μ+3σ. "
            "That set is a candidate normal-outlier class."
        )
    p95_note = (
        f"p95 sensitivity: sites with selected leftover > unsel p95 = "
        f"{n_with_p95}/{n_sites}  mean tokens={mean_p95:.3g}"
    )
    return [
        kind,
        f"sites={n_sites}  with_selected_ch={n_with_sel}  with_massive={n_with_mass}  "
        f"with_normal={n_with_normal}",
        f"mean tokens/site  massive={mean_mass:.3g}  normal={mean_normal:.3g}  "
        f"selected_bulk={mean_bulk:.3g}",
        f"mean excess L1/site  massive={mean_ex_m:.3g}  normal={mean_ex_n:.3g}  "
        f"normal/selected-leftover tokens={mean_frac:.3f}",
        p95_note,
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-data", type=Path, required=True)
    matched.add_model_cli(parser)
    parser.add_argument("--layer-regex", default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--samples", default=None)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--noise-seeds", default=None)
    parser.add_argument("--outlier-std-k", type=float, default=3.0)
    parser.add_argument(
        "--ref-mode",
        default="unselected_std",
        choices=("unselected_std", "next_largest", "leftover_mad", "channel_mad"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--params-dtype", default="bfloat16")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tools/img/dit_outlier_normal_occupancy"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(args.checkpoint)
    if not args.calibration_data.is_file():
        raise FileNotFoundError(args.calibration_data)
    sample_ids = (
        detail._parse_nonneg_ints(args.samples, name="--samples")
        if args.samples is not None
        else [int(args.sample_index)]
    )
    noise_ids = (
        detail._parse_nonneg_ints(args.noise_seeds, name="--noise-seeds")
        if args.noise_seeds is not None
        else [int(args.noise_seed)]
    )
    skip_first_token = args.model == "groot_n17"
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
        raise RuntimeError(f"Calibration yielded {len(batches)}, need {need}.")
    conditions = [(sample, seed) for sample in sample_ids for seed in noise_ids]
    print(
        f"model={args.model} layers={len(layers)} steps={runtime.num_steps} "
        f"n_tokens={runtime.n_tokens} skip_first={skip_first_token} "
        f"conditions={len(conditions)} capture=occupancy ref_mode={args.ref_mode}"
    )
    rows: list[SiteRow] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, (sample, seed) in enumerate(conditions, start=1):
        request = matched._fixed_noise_request(
            adapter, batches[sample], runtime, noise_seed=seed
        )
        print(f"\n=== condition {index}/{len(conditions)} sample={sample} noise={seed} ===")
        _, activations, _ = all_exp._run_all(
            adapter,
            request,
            layers,
            num_steps=runtime.num_steps,
            n_tokens=runtime.n_tokens,
            std_k=args.outlier_std_k,
            skip_first_token=skip_first_token,
        )
        assert activations is not None
        cond_rows = _rows_for_activations(
            activations,
            sample=sample,
            noise=seed,
            std_k=args.outlier_std_k,
            skip_first_token=skip_first_token,
            ref_mode=args.ref_mode,
        )
        n_norm = sum(1 for row in cond_rows if row.n_normal > 0)
        mean_n = float(np.mean([row.n_normal for row in cond_rows]))
        mean_m = float(np.mean([row.n_massive for row in cond_rows]))
        print(
            f"sites={len(cond_rows)}  with_normal={n_norm}  "
            f"mean n_massive={mean_m:.3g}  mean n_normal={mean_n:.3g}"
        )
        rows.extend(cond_rows)
        del activations
        torch.cuda.empty_cache()
    lines = _verdict(rows)
    print("\n=== occupancy verdict ===")
    for line in lines:
        print(line)
    csv_path = args.output_dir / "outlier_normal_occupancy.csv"
    plot_path = args.output_dir / "outlier_normal_occupancy.png"
    verdict_path = args.output_dir / "occupancy.txt"
    _write_csv(rows, csv_path)
    _plot_occupancy(rows, plot_path)
    verdict_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for path in (csv_path, plot_path, verdict_path):
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
