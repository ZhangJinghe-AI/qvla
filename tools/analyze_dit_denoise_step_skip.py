#!/usr/bin/env python
r"""Shrink one Euler update by a matched RMS and test coarse-vs-fine roles.

pi0.5 refines a single initial action noise with a 10-step Euler loop:

    x <- x + dt * v(x, t)

Whole-step skip is unfair: later ``||dt * v||`` is larger, and later skips
cannot be repaired. This script instead removes the **same** RMS from
exactly one step, using the minimum unskipped ``||dt * v||`` on that
trajectory as the matched budget. Other steps run as usual.

Early matched shrinks should move ``arm_mean_shift`` if those steps set
the overall path; late shrinks should move ``arm_local`` if they set
detail. The no-shrink patched loop must be bit-identical to baseline.

Example:
    CUDA_VISIBLE_DEVICES=5 uv run python \
      tools/analyze_dit_denoise_step_skip.py \
      --checkpoint /data/share/pi05_libero_finetuned_v044 \
      --calibration-data ../calibration_data/libero_goal_30_7_demo.npz \
      --samples 0,1,2,3 --noise-seeds 0,1 \
      --output-dir tools/img/dit_denoise_step_equal_shrink_all_by_step
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import sys
from collections import defaultdict
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
from qvla.adapters.pi05.step_hook import find_expert_runner  # noqa: E402
from qvla.config import QVLAConfig  # noqa: E402


@dataclass(frozen=True)
class SkipRow:
    sample: int
    noise: int
    step: int
    n_steps: int
    update_rms: float
    remove_rms: float
    metrics: detail.ActionMetrics


def _delta_rms(delta: torch.Tensor) -> float:
    value = delta.detach().to(torch.float32).square().mean().sqrt()
    return float(value.item())


def _keep_scale(update_rms: float, remove_rms: float) -> float:
    if remove_rms < 0.0:
        raise ValueError(f"remove_rms must be >= 0, got {remove_rms}.")
    if update_rms <= 0.0:
        raise RuntimeError(f"Cannot shrink a non-positive update RMS={update_rms}.")
    if remove_rms > update_rms * (1.0 + 1e-5):
        raise RuntimeError(
            f"remove_rms={remove_rms:.6g} exceeds update_rms={update_rms:.6g}."
        )
    keep = 1.0 - remove_rms / update_rms
    return min(max(keep, 0.0), 1.0)


def _euler_with_optional_skip(
    runner,
    noise: torch.Tensor,
    *,
    skip_step: int | None,
    remove_rms: float | None = None,
) -> tuple[torch.Tensor, list[float]]:
    """Run the Euler loop, intervening at one step.

    ``skip_step is None``: no intervention.
    ``remove_rms is None``: drop the whole ``dt * v`` at ``skip_step``.
    Otherwise scale that one update so the removed RMS equals ``remove_rms``.
    """
    num_steps = int(runner._num_steps)
    if num_steps < 1:
        raise RuntimeError(f"Euler loop has no steps (num_steps={num_steps}).")
    if skip_step is not None and not (0 <= int(skip_step) < num_steps):
        raise ValueError(f"skip_step must be in [0, {num_steps}), got {skip_step}.")
    if remove_rms is not None and skip_step is None:
        raise ValueError("remove_rms requires skip_step.")
    dt = float(runner._dt)
    x_t = noise
    update_rms: list[float] = []
    for step in range(num_steps):
        v_t = runner._one_step(x_t, step)
        delta = dt * v_t.to(dtype=x_t.dtype)
        rms = _delta_rms(delta)
        update_rms.append(rms)
        if skip_step is None or int(step) != int(skip_step):
            x_t = x_t + delta
            continue
        if remove_rms is None:
            continue
        keep = _keep_scale(rms, float(remove_rms))
        x_t = x_t + delta * keep
    return x_t, update_rms


@contextlib.contextmanager
def _patched_euler_skip(
    runner,
    *,
    skip_step: int | None,
    remove_rms: float | None = None,
):
    if getattr(runner, "graph", None) is not None:
        raise RuntimeError(
            "Euler skip needs the eager loop; the expert runner has a CUDA graph."
        )
    original = runner._fwd_loop
    recorded: list[float] = []

    def _fwd_loop(*, noise: torch.Tensor) -> torch.Tensor:
        x_t, update_rms = _euler_with_optional_skip(
            runner, noise, skip_step=skip_step, remove_rms=remove_rms
        )
        recorded.clear()
        recorded.extend(update_rms)
        return x_t

    runner._fwd_loop = _fwd_loop
    try:
        yield recorded
    finally:
        runner._fwd_loop = original


def _early_late_steps(n_steps: int) -> tuple[list[int], list[int]]:
    if n_steps < 2:
        raise ValueError(f"Need at least 2 denoise steps, got {n_steps}.")
    split = n_steps // 2
    early = list(range(split))
    late = list(range(split, n_steps))
    if not early or not late:
        raise RuntimeError(f"Cannot split {n_steps} steps into early/late halves.")
    return early, late


def _mean_for_steps(rows: list[SkipRow], steps: list[int], attr: str) -> float:
    values = [
        float(getattr(row.metrics, attr))
        for row in rows
        if row.step in steps
    ]
    return detail._mean(values)


def _verdict(rows: list[SkipRow]) -> str:
    if not rows:
        raise ValueError("Cannot judge an empty result list.")
    n_steps = rows[0].n_steps
    if any(row.n_steps != n_steps for row in rows):
        raise RuntimeError("Step counts do not match across rows.")
    early, late = _early_late_steps(n_steps)
    early_mean = _mean_for_steps(rows, early, "arm_mean_shift_rmse")
    late_mean = _mean_for_steps(rows, late, "arm_mean_shift_rmse")
    early_local = _mean_for_steps(rows, early, "arm_local_rmse")
    late_local = _mean_for_steps(rows, late, "arm_local_rmse")
    mean_ratio = early_mean / late_mean if late_mean > 0.0 else float("inf")
    local_ratio = late_local / early_local if early_local > 0.0 else float("inf")
    print("\n=== coarse-vs-fine hypothesis (matched-RMS shrink of one Euler step) ===")
    print(
        f"early steps {early[0]}-{early[-1]}  "
        f"mean_shift={early_mean:.4e}  local={early_local:.4e}"
    )
    print(
        f"late  steps {late[0]}-{late[-1]}  "
        f"mean_shift={late_mean:.4e}  local={late_local:.4e}"
    )
    print(
        f"ratio early/late mean_shift={mean_ratio:.3f}  "
        f"late/early local={local_ratio:.3f}"
    )
    mean_ok = early_mean > late_mean
    local_ok = late_local > early_local
    if mean_ok and local_ok:
        verdict = (
            "HOLDS: shrinking an early step by a matched RMS moves the coarse "
            "path more than the same shrink late, and a late shrink moves local "
            "detail more than an early shrink."
        )
    elif (not mean_ok) and (not local_ok):
        verdict = (
            "FAILS: early matched shrinks do not dominate coarse-path shift, "
            "and late shrinks do not dominate local detail. Denoise steps do "
            "not split into overall-then-detail the way image DiT does."
        )
    else:
        verdict = (
            "MIXED: only one of the two predictions holds "
            "(early=coarse path, late=local detail)."
        )
    print(verdict)
    return verdict


def _step_actions(adapter, request) -> torch.Tensor:
    with torch.inference_mode():
        actions = adapter.engine.step(request)
    if not torch.is_tensor(actions):
        raise RuntimeError(f"Engine returned {type(actions)}, expected tensor.")
    actions = actions.detach().to(torch.float32).cpu()
    detail._finite(actions, "predicted actions")
    return actions


def _experiment(
    adapter,
    request,
    *,
    runtime: matched.ClipRuntime,
    sample: int,
    noise: int,
) -> list[SkipRow]:
    num_steps = runtime.num_steps
    action_dim = runtime.action_dim
    runner = find_expert_runner(adapter.engine)
    if int(runner._num_steps) != int(num_steps):
        raise RuntimeError(
            f"Runner steps {runner._num_steps} != runtime.num_steps {num_steps}."
        )

    baseline = _step_actions(adapter, request)
    if baseline.ndim != 3 or baseline.shape[:2] != (1, runtime.action_horizon):
        raise RuntimeError(
            f"Expected actions (1,{runtime.action_horizon},width), got {baseline.shape}."
        )
    if int(baseline.shape[2]) < action_dim:
        raise RuntimeError(
            f"Action width {baseline.shape[2]} < action_dim={action_dim}."
        )
    baseline_arm = baseline[0, :, :action_dim]

    with _patched_euler_skip(runner, skip_step=None) as identity_rms:
        identity = _step_actions(adapter, request)
    detail._assert_actions_equal(identity, baseline, name="no-skip Euler patch")
    if len(identity_rms) != num_steps:
        raise RuntimeError(
            f"Recorded {len(identity_rms)} update RMS values, expected {num_steps}."
        )
    print(
        "no-skip Euler patch: actions unchanged; "
        "update_rms=["
        + ", ".join(f"{value:.3e}" for value in identity_rms)
        + "]"
    )
    remove_rms = min(identity_rms)
    if remove_rms <= 0.0:
        raise RuntimeError(f"Matched remove RMS is non-positive: {remove_rms}.")
    print(f"matched remove_rms={remove_rms:.3e} (min unskipped ||dt·v||)")

    rows: list[SkipRow] = []
    for step in range(num_steps):
        keep = _keep_scale(identity_rms[step], remove_rms)
        with _patched_euler_skip(
            runner, skip_step=step, remove_rms=remove_rms
        ):
            skipped = _step_actions(adapter, request)
        if torch.equal(skipped, baseline):
            raise RuntimeError(
                f"Shrinking step {step} left actions identical to baseline; "
                f"update_rms={identity_rms[step]:.3e}, remove_rms={remove_rms:.3e}."
            )
        metrics = detail._action_metrics(skipped[0, :, :action_dim], baseline_arm)
        row = SkipRow(
            sample=sample,
            noise=noise,
            step=step,
            n_steps=num_steps,
            update_rms=identity_rms[step],
            remove_rms=remove_rms,
            metrics=metrics,
        )
        rows.append(row)
        print(
            f"  shrink step={step} update_rms={row.update_rms:.3e} "
            f"remove={row.remove_rms:.3e} keep={keep:.3f} "
            f"mean={metrics.arm_mean_shift_rmse:.3e} "
            f"local={metrics.arm_local_rmse:.3e}"
        )
    return rows


def _rows_by_step(rows: list[SkipRow]) -> dict[int, list[SkipRow]]:
    grouped: dict[int, list[SkipRow]] = defaultdict(list)
    for row in rows:
        grouped[row.step].append(row)
    return dict(grouped)


def _mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot reduce empty metric list.")
    return float(array.mean()), float(array.std(ddof=0))


def _write_csv(rows: list[SkipRow], output: Path) -> None:
    metric_names = [field.name for field in fields(detail.ActionMetrics)]
    fieldnames = [
        "sample",
        "noise",
        "step",
        "n_steps",
        "update_rms",
        "remove_rms",
        *metric_names,
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            item = {
                "sample": row.sample,
                "noise": row.noise,
                "step": row.step,
                "n_steps": row.n_steps,
                "update_rms": row.update_rms,
                "remove_rms": row.remove_rms,
            }
            for name in metric_names:
                item[name] = getattr(row.metrics, name)
            writer.writerow(item)


def _plot(rows: list[SkipRow], output: Path) -> None:
    import matplotlib.pyplot as plt

    grouped = _rows_by_step(rows)
    steps = sorted(grouped)
    xs = np.asarray(steps, dtype=np.float64)
    mean_shift, mean_shift_std = zip(
        *[
            _mean_std([row.metrics.arm_mean_shift_rmse for row in grouped[step]])
            for step in steps
        ]
    )
    local, local_std = zip(
        *[
            _mean_std([row.metrics.arm_local_rmse for row in grouped[step]])
            for step in steps
        ]
    )
    update, update_std = zip(
        *[
            _mean_std([row.update_rms for row in grouped[step]])
            for step in steps
        ]
    )
    removed, removed_std = zip(
        *[
            _mean_std([row.remove_rms for row in grouped[step]])
            for step in steps
        ]
    )

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), constrained_layout=True)
    axes[0].errorbar(xs, mean_shift, yerr=mean_shift_std, marker="o", capsize=3)
    axes[0].set_title("Arm mean shift after matched shrink")
    axes[0].set_ylabel("RMSE vs full denoise")
    axes[1].errorbar(xs, local, yerr=local_std, marker="o", capsize=3, color="C1")
    axes[1].set_title("Arm local residual after matched shrink")
    axes[1].set_ylabel("RMSE vs full denoise")
    axes[2].errorbar(
        xs, update, yerr=update_std, marker="s", capsize=3, color="C2", label="full ||dt·v||"
    )
    axes[2].errorbar(
        xs,
        removed,
        yerr=removed_std,
        marker="o",
        capsize=3,
        color="C3",
        label="removed RMS",
    )
    axes[2].set_title("Update size vs matched remove")
    axes[2].set_ylabel("RMS")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("intervened denoise step")
        ax.set_xticks(list(steps))
        ax.grid(alpha=0.25)
    fig.suptitle(
        "Remove the same RMS from one Euler update (x ← x + (1-α) dt·v)",
        fontsize=12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Failed to write {output}.")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-data", type=Path, required=True)
    matched.add_model_cli(parser)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--samples", default=None)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--noise-seeds", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--params-dtype", default="bfloat16")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tools/img/dit_denoise_step_equal_shrink_all_by_step"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.model != "pi05":
        raise RuntimeError("Denoise-step skip is implemented for pi05 only.")
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(args.checkpoint)
    if not args.calibration_data.is_file():
        raise FileNotFoundError(args.calibration_data)
    if args.sample_index < 0:
        raise ValueError("--sample-index must be >= 0.")
    if args.noise_seed < 0:
        raise ValueError("--noise-seed must be >= 0.")
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

    adapter = matched.adapter_from_args(args)
    model = adapter.build_model()
    model.eval()
    adapter.warmup_for_calibration(model)
    config = QVLAConfig.for_model_kind(args.model)
    runtime = matched.clip_runtime(adapter, config)
    need = max(sample_ids) + 1
    batches = list(adapter.iter_calibration_batches(need))
    if len(batches) != need:
        raise RuntimeError(
            f"Requested samples {sample_ids}, calibration yielded "
            f"{len(batches)} samples."
        )
    conditions = [(sample, seed) for sample in sample_ids for seed in noise_ids]
    print(
        f"model={args.model}, samples={sample_ids}, noise_seeds={noise_ids}, "
        f"n_conditions={len(conditions)}, steps={runtime.num_steps}, "
        f"horizon={runtime.action_horizon}, action_dim={runtime.action_dim}, "
        f"mode=matched-rms-shrink-one-euler-step"
    )

    rows: list[SkipRow] = []
    for index, (sample, seed) in enumerate(conditions, start=1):
        request = matched._fixed_noise_request(
            adapter, batches[sample], runtime, noise_seed=seed
        )
        print(
            f"\n=== condition {index}/{len(conditions)} "
            f"sample={sample} noise={seed} ==="
        )
        rows.extend(
            _experiment(
                adapter,
                request,
                runtime=runtime,
                sample=sample,
                noise=seed,
            )
        )

    verdict = _verdict(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "denoise_step_equal_shrink.csv"
    png = args.output_dir / "denoise_step_equal_shrink.png"
    verdict_path = args.output_dir / "hypothesis.txt"
    _write_csv(rows, csv_path)
    _plot(rows, png)
    verdict_path.write_text(verdict + "\n", encoding="utf-8")
    for path in (csv_path, png, verdict_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Failed to write {path}.")
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
