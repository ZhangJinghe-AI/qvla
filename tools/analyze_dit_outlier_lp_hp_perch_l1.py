#!/usr/bin/env python
r"""All-step massive vs shoulder clip with **per-channel** matched L1.

Same all-denoise-step setup as ``analyze_dit_outlier_lp_hp_all_steps.py``, but
each MAD-selected channel matches its own massive-removed L1 on leftover of
that channel (shoulder band first, then expand below the MAD floor). Channels
whose leftover still cannot exceed the massive L1 are skipped on both sides.

Counts recorded per forward (summed over layers × steps):

* ``n_ch_expand_enough``: needed downward expand and then matched
* ``n_ch_insufficient``: even all leftover was not enough (skipped)

Example::

    CUDA_VISIBLE_DEVICES=5 HF_ENDPOINT=https://hf-mirror.com uv run python \
      tools/analyze_dit_outlier_lp_hp_perch_l1.py \
      --checkpoint /data/share/pi05_libero_finetuned_v044 \
      --calibration-data ../calibration_data/libero_goal_30_7_demo.npz \
      --samples 0,1,2,3 --noise-seeds 0,1 \
      --output-dir tools/img/dit_outlier_lp_hp_perch_l1
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
import analyze_dit_outlier_lp_hp_all_steps as all_steps  # noqa: E402
import analyze_dit_outlier_lp_hp_impact as lp_hp  # noqa: E402
from qvla.config import QVLAConfig  # noqa: E402


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
    massive_n_ch_target: int
    massive_n_ch_expand_enough: int
    massive_n_ch_insufficient: int
    shoulder_n_ch_target: int
    shoulder_n_ch_expand_enough: int
    shoulder_n_ch_insufficient: int


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
        "massive_n_ch_target",
        "massive_n_ch_expand_enough",
        "massive_n_ch_insufficient",
        "shoulder_n_ch_target",
        "shoulder_n_ch_expand_enough",
        "shoulder_n_ch_insufficient",
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
                    "massive_n_ch_target": row.massive_n_ch_target,
                    "massive_n_ch_expand_enough": row.massive_n_ch_expand_enough,
                    "massive_n_ch_insufficient": row.massive_n_ch_insufficient,
                    "shoulder_n_ch_target": row.shoulder_n_ch_target,
                    "shoulder_n_ch_expand_enough": row.shoulder_n_ch_expand_enough,
                    "shoulder_n_ch_insufficient": row.shoulder_n_ch_insufficient,
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

    clip_kw = dict(**run_kw, per_channel_l1=True)
    massive_actions, _, massive_stats = all_exp._run_all(
        adapter,
        request,
        layers,
        baseline_activations=baseline_activations,
        kind="selective",
        layer_index=layer_index,
        **clip_kw,
    )
    shoulder_actions, _, shoulder_stats = all_exp._run_all(
        adapter,
        request,
        layers,
        baseline_activations=baseline_activations,
        kind="random",
        bulk_seed=1000,
        layer_index=layer_index,
        **clip_kw,
    )
    massive = lp_hp._to_numpy_action(massive_actions, keep, action_dim)
    shoulder = lp_hp._to_numpy_action(shoulder_actions, keep, action_dim)
    overall_m = all_steps._arm_overall(massive, baseline)
    overall_s = all_steps._arm_overall(shoulder, baseline)
    print(
        "per-ch L1 massive "
        f"target={massive_stats.n_ch_target} "
        f"expand_enough={massive_stats.n_ch_expand_enough} "
        f"insufficient={massive_stats.n_ch_insufficient} | "
        "shoulder "
        f"target={shoulder_stats.n_ch_target} "
        f"expand_enough={shoulder_stats.n_ch_expand_enough} "
        f"insufficient={shoulder_stats.n_ch_insufficient}"
    )
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
                massive_n_ch_target=massive_stats.n_ch_target,
                massive_n_ch_expand_enough=massive_stats.n_ch_expand_enough,
                massive_n_ch_insufficient=massive_stats.n_ch_insufficient,
                shoulder_n_ch_target=shoulder_stats.n_ch_target,
                shoulder_n_ch_expand_enough=shoulder_stats.n_ch_expand_enough,
                shoulder_n_ch_insufficient=shoulder_stats.n_ch_insufficient,
            )
        )
        if cutoff == cutoffs[0]:
            print(
                f"massive sites={massive_stats.n_clipped}/{massive_stats.n_sites} "
                f"overall={overall_m:.3e} LP={mass_m.lp_rmse:.3e} HP={mass_m.hp_rmse:.3e} | "
                f"shoulder overall={overall_s:.3e} LP={shol_m.lp_rmse:.3e} "
                f"HP={shol_m.hp_rmse:.3e}"
            )
    if save_examples:
        all_steps._plot_traj(
            baseline, massive, shoulder, example_dir / "sample0_noise0_traj.png"
        )
        primary = 5 if 5 in cutoffs else cutoffs[0]
        all_steps._plot_xyz_freq(
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
        default=Path("tools/img/dit_outlier_lp_hp_perch_l1"),
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
        f"cutoffs={cutoffs} conditions={len(conditions)} clip=all_steps per_channel_l1"
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
    lines = [all_steps._verdict_for_cutoff(rows, cutoff) for cutoff in cutoffs]
    uniq = [row for row in rows if row.cutoff == primary]
    n_target = int(np.sum([row.massive_n_ch_target for row in uniq]))
    n_expand = int(np.sum([row.massive_n_ch_expand_enough for row in uniq]))
    n_short = int(np.sum([row.massive_n_ch_insufficient for row in uniq]))
    count_line = (
        f"channels with massive L1={n_target}  expand_enough={n_expand}  "
        f"insufficient_skip={n_short}  (summed over conditions, massive path)"
    )
    lines.append(count_line)
    print("\n=== verdict ===")
    for line in lines:
        print(line)
    csv_path = args.output_dir / "outlier_lp_hp_perch_l1.csv"
    verdict_path = args.output_dir / "hypothesis.txt"
    _write_csv(rows, csv_path)
    verdict_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for cutoff in cutoffs:
        all_steps._plot_bars(
            rows, cutoff, args.output_dir / f"outlier_lp_hp_perch_l1_k{cutoff}.png"
        )
    for path in (csv_path, verdict_path):
        print(f"Wrote {path}")
    print(f"primary cutoff K={primary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
