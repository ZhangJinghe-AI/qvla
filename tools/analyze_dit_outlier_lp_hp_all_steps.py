#!/usr/bin/env python
r"""Clip massive vs shoulder outliers on **every** denoise step, score overall/LP/HP.

Same intervention as ``analyze_dit_outlier_action_detail_all.py`` (all per-step
DiT linears, MAD-selected channels, matched L1). Massive = values above μ+kσ;
shoulder = leftover on those channels in [MAD floor, μ+kσ], massive sites
untouched. Unlike the by-step LP/HP experiment, one forward clips **all**
denoise steps on the live activation.

Scoring: orthonormal DCT along chunk time on arm dims. First K modes = coarse
path (主体); remainder = detail (细节). Also report overall arm RMSE and
gripper. GR00T never clips the state token; scoring uses the valid 16-step
action prefix.

Hypothesis: massive clip mainly raises detail RMSE; shoulder clip mainly
raises coarse RMSE.

Example::

    CUDA_VISIBLE_DEVICES=5 HF_ENDPOINT=https://hf-mirror.com uv run python \
      tools/analyze_dit_outlier_lp_hp_all_steps.py \
      --checkpoint /data/share/pi05_libero_finetuned_v044 \
      --calibration-data ../calibration_data/libero_goal_30_7_demo.npz \
      --samples 0,1,2,3 --noise-seeds 0,1 \
      --output-dir tools/img/dit_outlier_lp_hp_all_steps
"""

from __future__ import annotations

import argparse
import csv
import sys
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
import analyze_dit_outlier_lp_hp_impact as lp_hp  # noqa: E402
from qvla.config import QVLAConfig  # noqa: E402

LP_RATIO_MIN = lp_hp.LP_RATIO_MIN
DIM_LABELS = lp_hp.DIM_LABELS


@dataclass(frozen=True)
class CondRow:
    sample: int
    noise: int
    cutoff: int
    massive_overall: float
    shoulder_overall: float
    massive: lp_hp.FreqMetrics
    shoulder: lp_hp.FreqMetrics
    massive_n_sites: int
    massive_n_clipped: int
    shoulder_n_expanded: int


def _arm_overall(changed: np.ndarray, baseline: np.ndarray) -> float:
    if changed.shape != baseline.shape or changed.ndim != 2 or changed.shape[1] < 2:
        raise ValueError(
            f"Need matching (horizon, dims>=2), got {changed.shape} vs {baseline.shape}."
        )
    return lp_hp._rmse(changed[:, :-1] - baseline[:, :-1])


def _verdict_for_cutoff(rows: list[CondRow], cutoff: int) -> str:
    chosen = [row for row in rows if row.cutoff == cutoff]
    if not chosen:
        raise ValueError(f"No rows for cutoff={cutoff}.")
    overall_m = float(np.mean([row.massive_overall for row in chosen]))
    overall_s = float(np.mean([row.shoulder_overall for row in chosen]))
    lp_m = float(np.mean([row.massive.lp_rmse for row in chosen]))
    lp_s = float(np.mean([row.shoulder.lp_rmse for row in chosen]))
    hp_m = float(np.mean([row.massive.hp_rmse for row in chosen]))
    hp_s = float(np.mean([row.shoulder.hp_rmse for row in chosen]))
    lp_ratio = lp_s / lp_m if lp_m > 0.0 else float("inf")
    hp_ratio = hp_m / hp_s if hp_s > 0.0 else float("inf")
    mass_dom = hp_m / lp_m if lp_m > 0.0 else float("inf")
    shol_dom = hp_s / lp_s if lp_s > 0.0 else float("inf")
    overall_ratio = overall_s / overall_m if overall_m > 0.0 else float("inf")
    coarse_ok = lp_ratio >= LP_RATIO_MIN
    detail_ok = hp_ratio >= 1.0 and mass_dom > shol_dom
    header = (
        f"K={cutoff}  overall shoulder/massive={overall_ratio:.3f}  "
        f"LP shoulder/massive={lp_ratio:.3f}  "
        f"HP massive/shoulder={hp_ratio:.3f}  "
        f"HP/LP massive={mass_dom:.3f} shoulder={shol_dom:.3f}"
    )
    if coarse_ok and detail_ok:
        return (
            f"HOLDS: {header}. Shoulder moves the coarse path more; "
            "massive is more detail-heavy."
        )
    if (not coarse_ok) and (not detail_ok):
        return f"FAILS: {header}. Neither arm of the hypothesis holds."
    return f"MIXED: {header}. Only one arm of the hypothesis holds."


def _plot_bars(rows: list[CondRow], cutoff: int, output: Path) -> None:
    import matplotlib.pyplot as plt

    chosen = [row for row in rows if row.cutoff == cutoff]
    if not chosen:
        raise ValueError(f"No rows for cutoff={cutoff}.")

    def _mean_std(values: list[float]) -> tuple[float, float]:
        array = np.asarray(values, dtype=np.float64)
        return float(array.mean()), float(array.std(ddof=0))

    labels = ["overall", "LP coarse", "HP detail", "gripper"]
    massive = [
        _mean_std([row.massive_overall for row in chosen]),
        _mean_std([row.massive.lp_rmse for row in chosen]),
        _mean_std([row.massive.hp_rmse for row in chosen]),
        _mean_std([row.massive.gripper_rmse for row in chosen]),
    ]
    shoulder = [
        _mean_std([row.shoulder_overall for row in chosen]),
        _mean_std([row.shoulder.lp_rmse for row in chosen]),
        _mean_std([row.shoulder.hp_rmse for row in chosen]),
        _mean_std([row.shoulder.gripper_rmse for row in chosen]),
    ]
    xs = np.arange(len(labels), dtype=np.float64)
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.4, 4.4), constrained_layout=True)
    ax.bar(
        xs - width / 2,
        [item[0] for item in massive],
        width,
        yerr=[item[1] for item in massive],
        capsize=3,
        label="massive",
    )
    ax.bar(
        xs + width / 2,
        [item[0] for item in shoulder],
        width,
        yerr=[item[1] for item in shoulder],
        capsize=3,
        label="shoulder",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("RMSE vs baseline")
    ax.set_title(f"All denoise steps clipped  (DCT K={cutoff})")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Failed to write {output}.")


def _plot_traj(
    baseline: np.ndarray,
    massive: np.ndarray,
    shoulder: np.ndarray,
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    n_dims = int(baseline.shape[1])
    horizon = int(baseline.shape[0])
    time = np.arange(horizon)
    fig, axes = plt.subplots(n_dims, 1, figsize=(8.6, 1.45 * n_dims), sharex=True)
    fig.suptitle("sample0 noise0  clip every denoise step  black=baseline", fontsize=11)
    for dim, ax in enumerate(np.atleast_1d(axes)):
        ax.plot(time, baseline[:, dim], color="black", lw=1.8, label="baseline" if dim == 0 else None)
        ax.plot(time, massive[:, dim], color="C0", lw=1.1, label="massive" if dim == 0 else None)
        ax.plot(time, shoulder[:, dim], color="C1", lw=1.1, label="shoulder" if dim == 0 else None)
        ax.set_ylabel(DIM_LABELS[dim] if dim < len(DIM_LABELS) else f"dim {dim}", fontsize=8)
        ax.grid(alpha=0.2)
        if dim == 0:
            ax.legend(fontsize=8, loc="best")
    np.atleast_1d(axes)[-1].set_xlabel("chunk time")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Failed to write {output}.")


def _plot_xyz_freq(
    baseline: np.ndarray,
    massive: np.ndarray,
    shoulder: np.ndarray,
    dct: np.ndarray,
    cutoff: int,
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    time = np.arange(baseline.shape[0])
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 5.6), sharex=True, constrained_layout=True)
    fig.suptitle(f"xyz DCT split  K={cutoff}  clip every denoise step", fontsize=12)
    for dim in range(3):
        lp_b, hp_b = lp_hp._split_dct(baseline[:, dim : dim + 1], dct, cutoff)
        lp_m, hp_m = lp_hp._split_dct(massive[:, dim : dim + 1], dct, cutoff)
        lp_s, hp_s = lp_hp._split_dct(shoulder[:, dim : dim + 1], dct, cutoff)
        axes[0, dim].plot(time, lp_b[:, 0], color="black", lw=1.8, label="baseline")
        axes[0, dim].plot(time, lp_m[:, 0], color="C0", lw=1.1, label="massive")
        axes[0, dim].plot(time, lp_s[:, 0], color="C1", lw=1.1, label="shoulder")
        axes[0, dim].set_title(f"{DIM_LABELS[dim]}  LP")
        axes[1, dim].plot(time, hp_b[:, 0], color="black", lw=1.8)
        axes[1, dim].plot(time, hp_m[:, 0], color="C0", lw=1.1)
        axes[1, dim].plot(time, hp_s[:, 0], color="C1", lw=1.1)
        axes[1, dim].set_title(f"{DIM_LABELS[dim]}  HP")
        axes[1, dim].set_xlabel("chunk time")
        for row in range(2):
            axes[row, dim].grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8, loc="best")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Failed to write {output}.")


def _write_csv(rows: list[CondRow], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample",
        "noise",
        "cutoff",
        "massive_overall",
        "massive_lp",
        "massive_hp",
        "massive_gripper",
        "shoulder_overall",
        "shoulder_lp",
        "shoulder_hp",
        "shoulder_gripper",
        "massive_n_sites",
        "massive_n_clipped",
        "shoulder_n_expanded",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sample": row.sample,
                    "noise": row.noise,
                    "cutoff": row.cutoff,
                    "massive_overall": row.massive_overall,
                    "massive_lp": row.massive.lp_rmse,
                    "massive_hp": row.massive.hp_rmse,
                    "massive_gripper": row.massive.gripper_rmse,
                    "shoulder_overall": row.shoulder_overall,
                    "shoulder_lp": row.shoulder.lp_rmse,
                    "shoulder_hp": row.shoulder.hp_rmse,
                    "shoulder_gripper": row.shoulder.gripper_rmse,
                    "massive_n_sites": row.massive_n_sites,
                    "massive_n_clipped": row.massive_n_clipped,
                    "shoulder_n_expanded": row.shoulder_n_expanded,
                }
            )


def _experiment(
    adapter,
    request,
    layers: list[tuple[str, torch.nn.Module]],
    *,
    runtime: matched.ClipRuntime,
    std_k: float,
    cutoffs: list[int],
    sample: int,
    noise: int,
    skip_first_token: bool,
    save_examples: bool,
    example_dir: Path,
) -> list[CondRow]:
    num_steps = runtime.num_steps
    n_tokens = runtime.n_tokens
    action_dim = runtime.action_dim
    horizon = runtime.action_horizon
    layer_index = {name: index for index, (name, _layer) in enumerate(layers)}
    run_kw = dict(
        num_steps=num_steps,
        n_tokens=n_tokens,
        std_k=std_k,
        skip_first_token=skip_first_token,
    )
    baseline_actions, baseline_activations, _ = all_exp._run_all(
        adapter, request, layers, **run_kw
    )
    assert baseline_activations is not None
    baseline_full = baseline_actions[0, :, :action_dim]
    keep = (
        all_exp._valid_action_horizon(baseline_full)
        if skip_first_token
        else int(horizon)
    )
    if keep < horizon:
        print(f"Scoring first {keep} of {horizon} action steps (padding dropped).")
    if any(cutoff >= keep for cutoff in cutoffs):
        raise ValueError(f"cutoffs {cutoffs} must be < scored horizon {keep}.")
    baseline = lp_hp._to_numpy_action(baseline_actions, keep, action_dim)
    dct = lp_hp._dct_matrix(keep)

    identity_actions, _, _ = all_exp._run_all(
        adapter,
        request,
        layers,
        baseline_activations=baseline_activations,
        identity_writeback=True,
        **run_kw,
    )
    detail._assert_actions_equal(identity_actions, baseline_actions, name="identity write-back")
    restore_actions, _, _ = all_exp._run_all(
        adapter,
        request,
        layers,
        baseline_activations=baseline_activations,
        restore_after_clip=True,
        **run_kw,
    )
    detail._assert_actions_equal(
        restore_actions, baseline_actions, name="clip-then-restore write-back"
    )
    print("identity and clip-then-restore write-back: actions unchanged")

    massive_actions, _, massive_stats = all_exp._run_all(
        adapter,
        request,
        layers,
        baseline_activations=baseline_activations,
        kind="selective",
        layer_index=layer_index,
        **run_kw,
    )
    shoulder_actions, _, shoulder_stats = all_exp._run_all(
        adapter,
        request,
        layers,
        baseline_activations=baseline_activations,
        kind="random",
        bulk_seed=1000,
        layer_index=layer_index,
        **run_kw,
    )
    massive = lp_hp._to_numpy_action(massive_actions, keep, action_dim)
    shoulder = lp_hp._to_numpy_action(shoulder_actions, keep, action_dim)
    overall_m = _arm_overall(massive, baseline)
    overall_s = _arm_overall(shoulder, baseline)
    rows: list[CondRow] = []
    for cutoff in cutoffs:
        mass_m = lp_hp._freq_metrics(massive, baseline, dct, cutoff)
        shol_m = lp_hp._freq_metrics(shoulder, baseline, dct, cutoff)
        rows.append(
            CondRow(
                sample=sample,
                noise=noise,
                cutoff=cutoff,
                massive_overall=overall_m,
                shoulder_overall=overall_s,
                massive=mass_m,
                shoulder=shol_m,
                massive_n_sites=massive_stats.n_sites,
                massive_n_clipped=massive_stats.n_clipped,
                shoulder_n_expanded=shoulder_stats.n_expanded,
            )
        )
        if cutoff == cutoffs[0]:
            print(
                f"massive sites={massive_stats.n_clipped}/{massive_stats.n_sites} "
                f"overall={overall_m:.3e} LP={mass_m.lp_rmse:.3e} HP={mass_m.hp_rmse:.3e} | "
                f"shoulder expand={shoulder_stats.n_expanded} "
                f"overall={overall_s:.3e} LP={shol_m.lp_rmse:.3e} HP={shol_m.hp_rmse:.3e}"
            )
    if save_examples:
        _plot_traj(baseline, massive, shoulder, example_dir / "sample0_noise0_traj.png")
        primary = 5 if 5 in cutoffs else cutoffs[0]
        _plot_xyz_freq(
            baseline,
            massive,
            shoulder,
            dct,
            primary,
            example_dir / f"sample0_noise0_xyz_k{primary}.png",
        )
    return rows


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
    parser.add_argument("--cutoffs", default="3,5,8")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--params-dtype", default="bfloat16")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tools/img/dit_outlier_lp_hp_all_steps"),
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
    keep_guess = 16 if skip_first_token else runtime.action_horizon
    cutoffs = lp_hp._parse_cutoffs(args.cutoffs, keep_guess)
    need = max(sample_ids) + 1
    batches = list(adapter.iter_calibration_batches(need))
    if len(batches) != need:
        raise RuntimeError(f"Calibration yielded {len(batches)}, need {need}.")
    conditions = [(sample, seed) for sample in sample_ids for seed in noise_ids]
    print(
        f"model={args.model} layers={len(layers)} steps={runtime.num_steps} "
        f"n_tokens={runtime.n_tokens} skip_first={skip_first_token} "
        f"cutoffs={cutoffs} conditions={len(conditions)} clip=all_steps"
    )
    rows: list[CondRow] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, (sample, seed) in enumerate(conditions, start=1):
        request = matched._fixed_noise_request(
            adapter, batches[sample], runtime, noise_seed=seed
        )
        print(f"\n=== condition {index}/{len(conditions)} sample={sample} noise={seed} ===")
        rows.extend(
            _experiment(
                adapter,
                request,
                layers,
                runtime=runtime,
                std_k=args.outlier_std_k,
                cutoffs=cutoffs,
                sample=sample,
                noise=seed,
                skip_first_token=skip_first_token,
                save_examples=(sample == sample_ids[0] and seed == noise_ids[0]),
                example_dir=args.output_dir,
            )
        )
    primary = 5 if 5 in cutoffs else cutoffs[0]
    lines = [_verdict_for_cutoff(rows, cutoff) for cutoff in cutoffs]
    print("\n=== verdict ===")
    for line in lines:
        print(line)
    csv_path = args.output_dir / "outlier_lp_hp_all_steps.csv"
    verdict_path = args.output_dir / "hypothesis.txt"
    _write_csv(rows, csv_path)
    verdict_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for cutoff in cutoffs:
        _plot_bars(rows, cutoff, args.output_dir / f"outlier_lp_hp_all_steps_k{cutoff}.png")
    for path in (csv_path, verdict_path):
        print(f"Wrote {path}")
    print(f"primary cutoff K={primary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
