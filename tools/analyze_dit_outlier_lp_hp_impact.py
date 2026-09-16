#!/usr/bin/env python
r"""Clip massive vs shoulder outliers, score coarse vs detail with DCT.

Intervention is the same as ``analyze_dit_outlier_action_detail_all.py``:
all per-step DiT linears, one denoise step at a time, MAD-selected channels,
matched L1. Massive = values above μ+kσ; shoulder = leftover on the same
channels in [MAD floor, μ+kσ]. Shoulder clip does not touch massive sites.

Scoring replaces mean-shift / local with an orthonormal DCT along chunk time
on the arm dims. The first K modes are the coarse path (主体); the rest are
detail (细节). Gripper is reported separately.

Hypothesis: massive clip mainly raises detail RMSE; shoulder clip mainly
raises coarse RMSE.

Example::

    CUDA_VISIBLE_DEVICES=4 HF_ENDPOINT=https://hf-mirror.com uv run python \
      tools/analyze_dit_outlier_lp_hp_impact.py \
      --checkpoint /data/share/pi05_libero_finetuned_v044 \
      --calibration-data ../calibration_data/libero_goal_30_7_demo.npz \
      --samples 0,1,2,3 --noise-seeds 0,1 \
      --output-dir tools/img/dit_outlier_lp_hp_impact
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

DIM_LABELS = (
    "x (pos)",
    "y (pos)",
    "z (pos)",
    "rot0",
    "rot1",
    "rot2",
    "gripper",
)
LP_RATIO_MIN = 1.5


@dataclass(frozen=True)
class FreqMetrics:
    cutoff: int
    lp_rmse: float
    hp_rmse: float
    gripper_rmse: float


@dataclass(frozen=True)
class StepRow:
    sample: int
    noise: int
    step: int
    cutoff: int
    massive: FreqMetrics
    shoulder: FreqMetrics


def _dct_matrix(horizon: int) -> np.ndarray:
    if horizon < 2:
        raise ValueError(f"horizon must be >= 2, got {horizon}.")
    n = np.arange(horizon, dtype=np.float64)[None, :]
    k = np.arange(horizon, dtype=np.float64)[:, None]
    matrix = np.cos(np.pi / horizon * (n + 0.5) * k)
    matrix[0] *= np.sqrt(1.0 / horizon)
    matrix[1:] *= np.sqrt(2.0 / horizon)
    return matrix


def _split_dct(
    action: np.ndarray, dct: np.ndarray, cutoff: int
) -> tuple[np.ndarray, np.ndarray]:
    if action.ndim != 2 or dct.shape != (action.shape[0], action.shape[0]):
        raise ValueError(
            f"Expected action (H,D) and dct (H,H), got {action.shape}, {dct.shape}."
        )
    if not 1 <= int(cutoff) < action.shape[0]:
        raise ValueError(f"cutoff must be in [1, {action.shape[0]}), got {cutoff}.")
    coeff = dct @ action
    coarse = dct[:cutoff].T @ coeff[:cutoff]
    return coarse, action - coarse


def _rmse(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot compute RMSE on empty values.")
    return float(np.sqrt(np.mean(np.square(array))))


def _freq_metrics(
    changed: np.ndarray, baseline: np.ndarray, dct: np.ndarray, cutoff: int
) -> FreqMetrics:
    if changed.shape != baseline.shape or changed.ndim != 2 or changed.shape[1] < 2:
        raise ValueError(
            f"Need matching (horizon, dims>=2), got {changed.shape} vs {baseline.shape}."
        )
    lp_c, hp_c = _split_dct(changed[:, :-1], dct, cutoff)
    lp_b, hp_b = _split_dct(baseline[:, :-1], dct, cutoff)
    return FreqMetrics(
        cutoff=int(cutoff),
        lp_rmse=_rmse(lp_c - lp_b),
        hp_rmse=_rmse(hp_c - hp_b),
        gripper_rmse=_rmse(changed[:, -1] - baseline[:, -1]),
    )


def _to_numpy_action(actions: torch.Tensor, keep: int, action_dim: int) -> np.ndarray:
    return (
        actions[0, :keep, :action_dim]
        .detach()
        .to(dtype=torch.float32)
        .cpu()
        .numpy()
        .astype(np.float64)
    )


def _parse_cutoffs(text: str, horizon: int) -> list[int]:
    values = detail._parse_nonneg_ints(text, name="--cutoffs")
    if any(value < 1 or value >= horizon for value in values):
        raise ValueError(f"--cutoffs must be in [1, {horizon}), got {values}.")
    return values


def _verdict_for_cutoff(rows: list[StepRow], cutoff: int) -> str:
    chosen = [row for row in rows if row.cutoff == cutoff]
    if not chosen:
        raise ValueError(f"No rows for cutoff={cutoff}.")
    lp_m = float(np.mean([row.massive.lp_rmse for row in chosen]))
    lp_s = float(np.mean([row.shoulder.lp_rmse for row in chosen]))
    hp_m = float(np.mean([row.massive.hp_rmse for row in chosen]))
    hp_s = float(np.mean([row.shoulder.hp_rmse for row in chosen]))
    lp_ratio = lp_s / lp_m if lp_m > 0.0 else float("inf")
    hp_ratio = hp_m / hp_s if hp_s > 0.0 else float("inf")
    mass_dom = hp_m / lp_m if lp_m > 0.0 else float("inf")
    shol_dom = hp_s / lp_s if lp_s > 0.0 else float("inf")
    coarse_ok = lp_ratio >= LP_RATIO_MIN
    detail_ok = hp_ratio >= 1.0 and mass_dom > shol_dom
    header = (
        f"K={cutoff}  LP shoulder/massive={lp_ratio:.3f}  "
        f"HP massive/shoulder={hp_ratio:.3f}  "
        f"HP/LP massive={mass_dom:.3f} shoulder={shol_dom:.3f}"
    )
    if coarse_ok and detail_ok:
        return f"HOLDS: {header}. Shoulder moves the coarse path more; massive is more detail-heavy."
    if (not coarse_ok) and (not detail_ok):
        return f"FAILS: {header}. Neither arm of the hypothesis holds."
    return f"MIXED: {header}. Only one arm of the hypothesis holds."


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

    lp_m, lp_m_s = zip(*[_mean_std([r.massive.lp_rmse for r in grouped[s]]) for s in steps])
    lp_b, lp_b_s = zip(*[_mean_std([r.shoulder.lp_rmse for r in grouped[s]]) for s in steps])
    hp_m, hp_m_s = zip(*[_mean_std([r.massive.hp_rmse for r in grouped[s]]) for s in steps])
    hp_b, hp_b_s = zip(*[_mean_std([r.shoulder.hp_rmse for r in grouped[s]]) for s in steps])
    xs = np.asarray(steps, dtype=np.float64)
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.3), constrained_layout=True)
    axes[0].errorbar(xs, lp_m, yerr=lp_m_s, marker="o", capsize=3, label="massive")
    axes[0].errorbar(xs, lp_b, yerr=lp_b_s, marker="s", capsize=3, label="shoulder")
    axes[0].set_title(f"Coarse path LP (DCT K={cutoff})")
    axes[0].set_ylabel("RMSE vs baseline")
    axes[1].errorbar(xs, hp_m, yerr=hp_m_s, marker="o", capsize=3, label="massive")
    axes[1].errorbar(xs, hp_b, yerr=hp_b_s, marker="s", capsize=3, label="shoulder")
    axes[1].set_title(f"Detail HP (DCT K={cutoff})")
    axes[1].set_ylabel("RMSE vs baseline")
    lp_ratio = [s / m if m > 0 else np.nan for m, s in zip(lp_m, lp_b)]
    hp_ratio = [m / s if s > 0 else np.nan for m, s in zip(hp_m, hp_b)]
    axes[2].plot(xs, lp_ratio, "s--", label="LP shoulder/massive")
    axes[2].plot(xs, hp_ratio, "o-", label="HP massive/shoulder")
    axes[2].axhline(1.0, color="0.5", lw=0.8)
    axes[2].axhline(LP_RATIO_MIN, color="0.4", lw=0.8, ls=":")
    axes[2].set_title("Hypothesis ratios")
    axes[2].set_ylabel("ratio")
    for ax in axes:
        ax.set_xlabel("denoise step")
        ax.set_xticks(list(steps))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle(
        "Massive vs shoulder clip; coarse=DCT low modes, detail=remainder",
        fontsize=12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Failed to write {output}.")


def _plot_example(
    baseline: np.ndarray,
    massive: np.ndarray,
    shoulder: np.ndarray,
    step: int,
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    n_dims = int(baseline.shape[1])
    horizon = int(baseline.shape[0])
    time = np.arange(horizon)
    fig, axes = plt.subplots(n_dims, 1, figsize=(8.6, 1.45 * n_dims), sharex=True)
    fig.suptitle(
        f"sample0 noise0  clip at denoise step {step}  black=baseline",
        fontsize=11,
    )
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


def _write_csv(rows: list[StepRow], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample",
        "noise",
        "step",
        "cutoff",
        "massive_lp",
        "massive_hp",
        "massive_gripper",
        "shoulder_lp",
        "shoulder_hp",
        "shoulder_gripper",
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
                    "massive_lp": row.massive.lp_rmse,
                    "massive_hp": row.massive.hp_rmse,
                    "massive_gripper": row.massive.gripper_rmse,
                    "shoulder_lp": row.shoulder.lp_rmse,
                    "shoulder_hp": row.shoulder.hp_rmse,
                    "shoulder_gripper": row.shoulder.gripper_rmse,
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
    baseline = _to_numpy_action(baseline_actions, keep, action_dim)
    dct = _dct_matrix(keep)

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
            **run_kw,
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
            **run_kw,
        )
        massive = _to_numpy_action(massive_actions, keep, action_dim)
        shoulder = _to_numpy_action(shoulder_actions, keep, action_dim)
        for cutoff in cutoffs:
            mass_m = _freq_metrics(massive, baseline, dct, cutoff)
            shol_m = _freq_metrics(shoulder, baseline, dct, cutoff)
            rows.append(
                StepRow(
                    sample=sample,
                    noise=noise,
                    step=step,
                    cutoff=cutoff,
                    massive=mass_m,
                    shoulder=shol_m,
                )
            )
            if cutoff == cutoffs[0]:
                print(
                    f"massive sites={massive_stats.n_clipped}/{massive_stats.n_sites} "
                    f"LP={mass_m.lp_rmse:.3e} HP={mass_m.hp_rmse:.3e} | "
                    f"shoulder expand={shoulder_stats.n_expanded} "
                    f"LP={shol_m.lp_rmse:.3e} HP={shol_m.hp_rmse:.3e}"
                )
        if save_examples and step in example_steps:
            _plot_example(
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
        default=Path("tools/img/dit_outlier_lp_hp_impact"),
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
    cutoffs = _parse_cutoffs(args.cutoffs, keep_guess)
    need = max(sample_ids) + 1
    batches = list(adapter.iter_calibration_batches(need))
    if len(batches) != need:
        raise RuntimeError(f"Calibration yielded {len(batches)}, need {need}.")
    conditions = [(sample, seed) for sample in sample_ids for seed in noise_ids]
    print(
        f"model={args.model} layers={len(layers)} steps={runtime.num_steps} "
        f"n_tokens={runtime.n_tokens} skip_first={skip_first_token} "
        f"cutoffs={cutoffs} conditions={len(conditions)}"
    )
    rows: list[StepRow] = []
    example_steps = sorted(
        {0, runtime.num_steps // 2, runtime.num_steps - 1}
    )
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
    lines = [_verdict_for_cutoff(rows, cutoff) for cutoff in cutoffs]
    print("\n=== verdict ===")
    for line in lines:
        print(line)
    csv_path = args.output_dir / "outlier_lp_hp.csv"
    verdict_path = args.output_dir / "hypothesis.txt"
    _write_csv(rows, csv_path)
    verdict_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for cutoff in cutoffs:
        _plot_metrics(
            rows, cutoff, args.output_dir / f"outlier_lp_hp_k{cutoff}.png"
        )
    for path in (csv_path, verdict_path):
        print(f"Wrote {path}")
    print(f"primary cutoff K={primary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
