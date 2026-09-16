#!/usr/bin/env python
r"""All-token MAD band vs massive, by-step LP/HP, **no energy matching**.

On MAD-selected channels, fit μ+3σ and median+3×1.4826×MAD on every token
of that column (same MAD formula as channel selection). Then:

* massive clip: every |x| > μ+3σ is clamped to μ+3σ
* normal clip: every MAD < |x| ≤ μ+3σ is clamped to the MAD floor;
  values above μ+3σ are left unchanged
* |x| ≤ MAD floor: ordinary, untouched

The two arms are independent. Excess L1 is not matched.

Example::

    CUDA_VISIBLE_DEVICES=5 HF_ENDPOINT=https://hf-mirror.com uv run python \
      tools/analyze_dit_outlier_lp_hp_channel_mad_full_by_step.py \
      --checkpoint /data/share/pi05_libero_finetuned_v044 \
      --calibration-data ../calibration_data/libero_goal_30_7_demo.npz \
      --samples 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --noise-seeds 0,1,2,3 \
      --output-dir tools/img/dit_outlier_lp_hp_channel_mad_full_by_step_s16n4
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
import analyze_dit_outlier_lp_hp_all_steps as all_steps  # noqa: E402
import analyze_dit_outlier_lp_hp_impact as lp_hp  # noqa: E402
from qvla.config import QVLAConfig  # noqa: E402

LP_RATIO_MIN = lp_hp.LP_RATIO_MIN


@dataclass(frozen=True)
class StepRow:
    sample: int
    noise: int
    step: int
    cutoff: int
    massive_overall: float
    shoulder_overall: float
    massive: lp_hp.FreqMetrics
    shoulder: lp_hp.FreqMetrics
    massive_n_ch_target: int
    massive_n_ch_expand_enough: int
    massive_n_ch_insufficient: int


def _plot_metrics(rows: list[StepRow], cutoff: int, output: Path) -> None:
    import matplotlib.pyplot as plt

    chosen = [row for row in rows if row.cutoff == cutoff]
    grouped: dict[int, list[StepRow]] = defaultdict(list)
    for row in chosen:
        grouped[row.step].append(row)
    steps = sorted(grouped)

    def _mean_std(values: list[float]) -> tuple[float, float]:
        array = np.asarray(values, dtype=np.float64)
        return float(array.mean()), float(array.std(ddof=0))

    ov_m, ov_m_s = zip(*[_mean_std([r.massive_overall for r in grouped[s]]) for s in steps])
    ov_b, ov_b_s = zip(*[_mean_std([r.shoulder_overall for r in grouped[s]]) for s in steps])
    lp_m, lp_m_s = zip(*[_mean_std([r.massive.lp_rmse for r in grouped[s]]) for s in steps])
    lp_b, lp_b_s = zip(*[_mean_std([r.shoulder.lp_rmse for r in grouped[s]]) for s in steps])
    hp_m, hp_m_s = zip(*[_mean_std([r.massive.hp_rmse for r in grouped[s]]) for s in steps])
    hp_b, hp_b_s = zip(*[_mean_std([r.shoulder.hp_rmse for r in grouped[s]]) for s in steps])
    xs = np.asarray(steps, dtype=np.float64)
    fig, axes = plt.subplots(1, 4, figsize=(17.4, 4.2), constrained_layout=True)
    axes[0].errorbar(xs, ov_m, yerr=ov_m_s, marker="o", capsize=3, label="massive")
    axes[0].errorbar(xs, ov_b, yerr=ov_b_s, marker="s", capsize=3, label="shoulder")
    axes[0].set_title("Overall arm")
    axes[1].errorbar(xs, lp_m, yerr=lp_m_s, marker="o", capsize=3, label="massive")
    axes[1].errorbar(xs, lp_b, yerr=lp_b_s, marker="s", capsize=3, label="shoulder")
    axes[1].set_title(f"Coarse path LP (K={cutoff})")
    axes[2].errorbar(xs, hp_m, yerr=hp_m_s, marker="o", capsize=3, label="massive")
    axes[2].errorbar(xs, hp_b, yerr=hp_b_s, marker="s", capsize=3, label="shoulder")
    axes[2].set_title(f"Detail HP (K={cutoff})")
    lp_ratio = [s / m if m > 0 else np.nan for m, s in zip(lp_m, lp_b)]
    hp_ratio = [m / s if s > 0 else np.nan for m, s in zip(hp_m, hp_b)]
    axes[3].plot(xs, lp_ratio, "s--", label="LP shoulder/massive")
    axes[3].plot(xs, hp_ratio, "o-", label="HP massive/shoulder")
    axes[3].axhline(1.0, color="0.5", lw=0.8)
    axes[3].axhline(LP_RATIO_MIN, color="0.4", lw=0.8, ls=":")
    axes[3].set_title("Hypothesis ratios")
    for ax in axes:
        ax.set_xlabel("denoise step")
        ax.set_xticks(list(steps))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        if ax is not axes[3]:
            ax.set_ylabel("RMSE vs baseline")
    fig.suptitle(
        "Normal = MAD-to-μ+3σ, full clip, no L1 match  |  one denoise step",
        fontsize=12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Failed to write {output}.")


def _plot_channel_counts(rows: list[StepRow], output: Path) -> None:
    import matplotlib.pyplot as plt

    primary = rows[0].cutoff
    chosen = [row for row in rows if row.cutoff == primary]
    grouped: dict[int, list[StepRow]] = defaultdict(list)
    for row in chosen:
        grouped[row.step].append(row)
    steps = sorted(grouped)
    xs = np.asarray(steps, dtype=np.float64)
    target = [float(np.mean([r.massive_n_ch_target for r in grouped[s]])) for s in steps]
    expand = [float(np.mean([r.massive_n_ch_expand_enough for r in grouped[s]])) for s in steps]
    short = [float(np.mean([r.massive_n_ch_insufficient for r in grouped[s]])) for s in steps]
    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    ax.plot(xs, target, "k-o", label="sites with massive L1")
    ax.plot(xs, expand, "C1-s", label="expand enough")
    ax.plot(xs, short, "C3-^", label="insufficient skip")
    ax.set_xlabel("denoise step")
    ax.set_ylabel("sites / step (mean over conditions)")
    ax.set_xticks(list(steps))
    ax.set_title("Full clip counts (no L1 match)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Failed to write {output}.")


def _write_csv(rows: list[StepRow], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample",
        "noise",
        "step",
        "cutoff",
        "massive_overall",
        "massive_lp",
        "massive_hp",
        "massive_gripper",
        "shoulder_overall",
        "shoulder_lp",
        "shoulder_hp",
        "shoulder_gripper",
        "massive_n_ch_target",
        "massive_n_ch_expand_enough",
        "massive_n_ch_insufficient",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sample": row.sample,
                    "noise": row.noise,
                    "step": row.step,
                    "cutoff": row.cutoff,
                    "massive_overall": row.massive_overall,
                    "massive_lp": row.massive.lp_rmse,
                    "massive_hp": row.massive.hp_rmse,
                    "massive_gripper": row.massive.gripper_rmse,
                    "shoulder_overall": row.shoulder_overall,
                    "shoulder_lp": row.shoulder.lp_rmse,
                    "shoulder_hp": row.shoulder.hp_rmse,
                    "shoulder_gripper": row.shoulder.gripper_rmse,
                    "massive_n_ch_target": row.massive_n_ch_target,
                    "massive_n_ch_expand_enough": row.massive_n_ch_expand_enough,
                    "massive_n_ch_insufficient": row.massive_n_ch_insufficient,
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
    example_steps: list[int],
    example_dir: Path,
) -> list[StepRow]:
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

    clip_kw = dict(
        **run_kw,
        normal_ref_l1=True,
        normal_ref_mode="channel_mad",
        normal_full_clip=True,
        allow_empty=True,
    )
    rows: list[StepRow] = []
    for step in range(num_steps):
        print(f"--- denoise step {step}/{num_steps - 1} ---")
        massive_actions, _, massive_stats = all_exp._run_all(
            adapter,
            request,
            layers,
            baseline_activations=baseline_activations,
            kind="selective",
            layer_index=layer_index,
            target_step=step,
            **clip_kw,
        )
        shoulder_actions, _, shoulder_stats = all_exp._run_all(
            adapter,
            request,
            layers,
            baseline_activations=baseline_activations,
            kind="random",
            bulk_seed=1000 + step * 17,
            layer_index=layer_index,
            target_step=step,
            **clip_kw,
        )
        massive = lp_hp._to_numpy_action(massive_actions, keep, action_dim)
        shoulder = lp_hp._to_numpy_action(shoulder_actions, keep, action_dim)
        overall_m = all_steps._arm_overall(massive, baseline)
        overall_s = all_steps._arm_overall(shoulder, baseline)
        n_normal_tokens = int(shoulder_stats.bulk_count)
        n_massive_tokens = int(massive_stats.outlier_count)
        del shoulder_stats
        for cutoff in cutoffs:
            mass_m = lp_hp._freq_metrics(massive, baseline, dct, cutoff)
            shol_m = lp_hp._freq_metrics(shoulder, baseline, dct, cutoff)
            rows.append(
                StepRow(
                    sample=sample,
                    noise=noise,
                    step=step,
                    cutoff=cutoff,
                    massive_overall=overall_m,
                    shoulder_overall=overall_s,
                    massive=mass_m,
                    shoulder=shol_m,
                    massive_n_ch_target=massive_stats.n_ch_target,
                    massive_n_ch_expand_enough=massive_stats.n_ch_expand_enough,
                    massive_n_ch_insufficient=massive_stats.n_ch_insufficient,
                )
            )
            if cutoff == cutoffs[0]:
                print(
                    f"massive sites={massive_stats.n_clipped}/{massive_stats.n_sites} "
                    f"overall={overall_m:.3e} LP={mass_m.lp_rmse:.3e} HP={mass_m.hp_rmse:.3e} | "
                    f"ch target={massive_stats.n_ch_target} "
                    f"n_massive={n_massive_tokens} n_normal={n_normal_tokens} "
                    f"insufficient={massive_stats.n_ch_insufficient} | "
                    f"shoulder overall={overall_s:.3e} LP={shol_m.lp_rmse:.3e} "
                    f"HP={shol_m.hp_rmse:.3e}"
                )
        if save_examples and step in example_steps:
            lp_hp._plot_example(
                baseline,
                massive,
                shoulder,
                step,
                example_dir / f"sample{sample}_noise{noise}_step{step}_traj.png",
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
        default=Path("tools/img/dit_outlier_lp_hp_channel_mad_full_by_step"),
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
        f"cutoffs={cutoffs} conditions={len(conditions)} "
        f"clip=by_step channel_mad_full"
    )
    rows: list[StepRow] = []
    example_steps = sorted({0, runtime.num_steps // 2, runtime.num_steps - 1})
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
                example_steps=example_steps,
                example_dir=args.output_dir,
            )
        )
    primary = 5 if 5 in cutoffs else cutoffs[0]
    lines = [lp_hp._verdict_for_cutoff(rows, cutoff) for cutoff in cutoffs]
    uniq = [row for row in rows if row.cutoff == primary]
    n_target = int(np.sum([row.massive_n_ch_target for row in uniq]))
    n_expand = int(np.sum([row.massive_n_ch_expand_enough for row in uniq]))
    n_short = int(np.sum([row.massive_n_ch_insufficient for row in uniq]))
    lines.append(
        f"sites with massive L1={n_target}  expand_enough={n_expand}  "
        f"insufficient_skip={n_short}  (summed over conditions x steps, massive path)"
    )
    print("\n=== verdict ===")
    for line in lines:
        print(line)
    csv_path = args.output_dir / "outlier_lp_hp_channel_mad_full_by_step.csv"
    verdict_path = args.output_dir / "hypothesis.txt"
    _write_csv(rows, csv_path)
    verdict_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for cutoff in cutoffs:
        _plot_metrics(
            rows, cutoff, args.output_dir / f"outlier_lp_hp_channel_mad_full_by_step_k{cutoff}.png"
        )
    _plot_channel_counts(
        rows, args.output_dir / "outlier_lp_hp_channel_mad_full_by_step_counts.png"
    )
    for path in (csv_path, verdict_path):
        print(f"Wrote {path}")
    print(f"primary cutoff K={primary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
