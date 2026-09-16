#!/usr/bin/env python
r"""Corrupt a finished action to time t_s and resume Euler from step s.

pi0.5 uses linear flow matching with ``t = 1 - s/N`` (s=0 is pure noise,
t=0 is the clean action). After generating action ``a`` from noise
``eps0``, this script forms

    x(t) = t * eps + (1-t) * a

and runs only Euler steps ``s ... N-1``. That asks what is still undecided
at noise level t: overall path (``arm_mean_shift``) or local detail
(``arm_local``). It does not skip a step on an intact trajectory, so later
steps are not "repairing a hole".

Sanity: ``s=0`` with ``eps = eps0`` must reproduce ``a`` bit-identically.
The plotted experiment uses fresh ``eps``. ``s=0`` with fresh ``eps`` is a
full redraw, not the causal effect of step 0.

Besides RMSE, the script dumps overlay trajectories: action[0:2] (EEF xy
if the chunk is absolute), the prefix-sum of those two dims (if the chunk
is delta), gripper vs time, and the per-step predicted clean action
``a_hat = x_t - t * v_t`` along the original generation.

Example:
    CUDA_VISIBLE_DEVICES=5 uv run python \
      tools/analyze_dit_denoise_sdedit.py \
      --checkpoint /data/share/pi05_libero_finetuned_v044 \
      --calibration-data ../calibration_data/libero_goal_30_7_demo.npz \
      --samples 0,1 --noise-seeds 0 --random-trials 4 \
      --output-dir tools/img/dit_denoise_sdedit_traj
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass, fields, replace
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
class EditRow:
    sample: int
    noise: int
    trial: int
    step: int
    n_steps: int
    time_t: float
    metrics: detail.ActionMetrics


@dataclass(frozen=True)
class TrajRecord:
    sample: int
    noise: int
    n_steps: int
    time_t: np.ndarray
    baseline: np.ndarray
    x0_pred: np.ndarray
    edited: np.ndarray


def _time_at_step(step: int, n_steps: int) -> float:
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}.")
    if not (0 <= int(step) <= int(n_steps)):
        raise ValueError(f"step must be in [0, {n_steps}], got {step}.")
    return 1.0 - float(step) / float(n_steps)


def _mix_action(action: torch.Tensor, eps: torch.Tensor, time_t: float) -> torch.Tensor:
    if action.shape != eps.shape:
        raise ValueError(
            f"action shape {tuple(action.shape)} != eps {tuple(eps.shape)}."
        )
    if time_t < 0.0 or time_t > 1.0:
        raise ValueError(f"time_t must be in [0, 1], got {time_t}.")
    t = action.new_tensor(float(time_t))
    return t * eps + (1.0 - t) * action


def _local_share(metrics: detail.ActionMetrics) -> float:
    total = float(metrics.arm_mean_shift_rmse) + float(metrics.arm_local_rmse)
    if total <= 0.0:
        return 0.0
    return float(metrics.arm_local_rmse) / total


def _euler_from_step(
    runner,
    noise: torch.Tensor,
    *,
    start_step: int,
) -> torch.Tensor:
    num_steps = int(runner._num_steps)
    if num_steps < 1:
        raise RuntimeError(f"Euler loop has no steps (num_steps={num_steps}).")
    if not (0 <= int(start_step) <= num_steps):
        raise ValueError(f"start_step must be in [0, {num_steps}], got {start_step}.")
    if int(start_step) == num_steps:
        return noise
    dt = float(runner._dt)
    x_t = noise
    for step in range(int(start_step), num_steps):
        v_t = runner._one_step(x_t, step)
        x_t = x_t + dt * v_t.to(dtype=x_t.dtype)
    return x_t


def _predicted_clean(
    x_t: torch.Tensor, v_t: torch.Tensor, time_t: float
) -> torch.Tensor:
    """Invert linear flow matching: x = t ε + (1-t) a, v = ε - a ⇒ a = x - t v."""
    if time_t < 0.0 or time_t > 1.0:
        raise ValueError(f"time_t must be in [0, 1], got {time_t}.")
    return x_t - float(time_t) * v_t.to(dtype=x_t.dtype)


def _euler_record_clean(
    runner, noise: torch.Tensor
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    num_steps = int(runner._num_steps)
    if num_steps < 1:
        raise RuntimeError(f"Euler loop has no steps (num_steps={num_steps}).")
    dt = float(runner._dt)
    x_t = noise
    predicted: list[torch.Tensor] = []
    for step in range(num_steps):
        time_t = _time_at_step(step, num_steps)
        v_t = runner._one_step(x_t, step)
        predicted.append(_predicted_clean(x_t, v_t, time_t).detach())
        x_t = x_t + dt * v_t.to(dtype=x_t.dtype)
    return x_t, predicted


@contextlib.contextmanager
def _patched_euler_record_clean(runner):
    if getattr(runner, "graph", None) is not None:
        raise RuntimeError(
            "Recording x0 needs the eager loop; the expert runner has a CUDA graph."
        )
    original = runner._fwd_loop
    recorded: list[list[torch.Tensor]] = []

    def _fwd_loop(*, noise: torch.Tensor) -> torch.Tensor:
        final, predicted = _euler_record_clean(runner, noise)
        recorded.clear()
        recorded.append(predicted)
        return final

    runner._fwd_loop = _fwd_loop
    try:
        yield recorded
    finally:
        runner._fwd_loop = original


@contextlib.contextmanager
def _patched_euler_from_step(runner, *, start_step: int):
    if getattr(runner, "graph", None) is not None:
        raise RuntimeError(
            "Resuming Euler needs the eager loop; the expert runner has a CUDA graph."
        )
    original = runner._fwd_loop

    def _fwd_loop(*, noise: torch.Tensor) -> torch.Tensor:
        return _euler_from_step(runner, noise, start_step=start_step)

    runner._fwd_loop = _fwd_loop
    try:
        yield
    finally:
        runner._fwd_loop = original


def _early_late_steps(n_steps: int) -> tuple[list[int], list[int]]:
    """Halves of s=1..N-1. s=0 with fresh noise is a full redraw, not used here."""
    if n_steps < 3:
        raise ValueError(f"Need at least 3 denoise steps, got {n_steps}.")
    rest = list(range(1, n_steps))
    split = len(rest) // 2
    early = rest[:split]
    late = rest[split:]
    if not early or not late:
        raise RuntimeError(f"Cannot split steps 1..{n_steps - 1} into halves.")
    return early, late


def _mean_for_steps(rows: list[EditRow], steps: list[int], *, share: bool) -> float:
    values = [
        _local_share(row.metrics) if share else float(row.metrics.arm_mean_shift_rmse)
        for row in rows
        if row.step in steps
    ]
    return detail._mean(values)


def _verdict(rows: list[EditRow]) -> str:
    if not rows:
        raise ValueError("Cannot judge an empty result list.")
    n_steps = rows[0].n_steps
    if any(row.n_steps != n_steps for row in rows):
        raise RuntimeError("Step counts do not match across rows.")
    early, late = _early_late_steps(n_steps)
    early_mean = _mean_for_steps(rows, early, share=False)
    late_mean = _mean_for_steps(rows, late, share=False)
    early_share = _mean_for_steps(rows, early, share=True)
    late_share = _mean_for_steps(rows, late, share=True)
    mean_ratio = early_mean / late_mean if late_mean > 0.0 else float("inf")
    print("\n=== coarse-vs-fine hypothesis (SDEdit resume from step s) ===")
    print(
        f"early start s={early[0]}-{early[-1]}  "
        f"mean_shift={early_mean:.4e}  local_share={early_share:.3f}"
    )
    print(
        f"late  start s={late[0]}-{late[-1]}  "
        f"mean_shift={late_mean:.4e}  local_share={late_share:.3f}"
    )
    print(
        f"ratio early/late mean_shift={mean_ratio:.3f}  "
        f"local_share late-early={late_share - early_share:+.3f}"
    )
    print("s=0 with fresh eps is a full redraw and is excluded from the verdict.")
    mag_ok = early_mean > late_mean
    share_ok = late_share > early_share
    if mag_ok and share_ok:
        verdict = (
            "HOLDS: restarting from a dirtier (earlier) state moves the coarse "
            "path more, and the remaining error after a late restart is more "
            "local detail."
        )
    elif (not mag_ok) and (not share_ok):
        verdict = (
            "FAILS: earlier restarts do not move the coarse path more, and "
            "late restarts are not more local. Denoise time does not split "
            "into overall-then-detail the way image DiT does."
        )
    else:
        verdict = (
            "MIXED: only one of the two predictions holds "
            "(earlier=larger mean shift, later=higher local share)."
        )
    print(verdict)
    return verdict


def _step_actions(adapter, request) -> torch.Tensor:
    with torch.inference_mode():
        actions = adapter.engine.step(request)
    if not torch.is_tensor(actions):
        raise RuntimeError(f"Engine returned {type(actions)}, expected tensor.")
    detail._finite(actions.detach(), "predicted actions")
    return actions.detach()


def _cpu_arm(actions: torch.Tensor, action_dim: int) -> torch.Tensor:
    if actions.ndim != 3 or int(actions.shape[0]) != 1:
        raise RuntimeError(f"Expected actions (1,horizon,width), got {actions.shape}.")
    if int(actions.shape[2]) < action_dim:
        raise RuntimeError(
            f"Action width {actions.shape[2]} < action_dim={action_dim}."
        )
    arm = actions[0, :, :action_dim].detach().to(torch.float32).cpu()
    detail._finite(arm, "arm actions")
    return arm


def _draw_eps(request_noise: torch.Tensor, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=request_noise.device)
    generator.manual_seed(int(seed))
    return torch.randn(
        request_noise.shape,
        generator=generator,
        device=request_noise.device,
        dtype=request_noise.dtype,
    )


def _experiment(
    adapter,
    request,
    *,
    runtime: matched.ClipRuntime,
    sample: int,
    noise: int,
    random_trials: int,
    random_seed: int,
) -> tuple[list[EditRow], TrajRecord]:
    num_steps = runtime.num_steps
    action_dim = runtime.action_dim
    runner = find_expert_runner(adapter.engine)
    if int(runner._num_steps) != int(num_steps):
        raise RuntimeError(
            f"Runner steps {runner._num_steps} != runtime.num_steps {num_steps}."
        )
    if request.noise is None:
        raise RuntimeError("Request has no frozen generation noise.")

    baseline = _step_actions(adapter, request)
    baseline_arm = _cpu_arm(baseline, action_dim)
    if baseline.shape[1] != runtime.action_horizon:
        raise RuntimeError(
            f"Action horizon {baseline.shape[1]} != {runtime.action_horizon}."
        )

    with _patched_euler_from_step(runner, start_step=0):
        identity = _step_actions(adapter, request)
    detail._assert_actions_equal(
        identity.cpu(),
        baseline.cpu(),
        name="resume-from-0 Euler patch",
    )
    mixed0 = _mix_action(baseline, request.noise, _time_at_step(0, num_steps))
    if not torch.equal(mixed0, request.noise):
        raise RuntimeError("t=1 mix with eps0 did not recover the generation noise.")
    with _patched_euler_from_step(runner, start_step=0):
        identity_mix = _step_actions(adapter, replace(request, noise=mixed0))
    detail._assert_actions_equal(
        identity_mix.cpu(),
        baseline.cpu(),
        name="s=0 mix with eps0",
    )
    with _patched_euler_record_clean(runner) as recorded:
        identity_x0 = _step_actions(adapter, request)
    detail._assert_actions_equal(
        identity_x0.cpu(),
        baseline.cpu(),
        name="record-x0 Euler patch",
    )
    if not recorded:
        raise RuntimeError("Record-x0 patch did not capture predicted clean actions.")
    x0_pred = np.stack(
        [_cpu_arm(step_x0, action_dim).numpy() for step_x0 in recorded[0]],
        axis=0,
    )
    print("identity: resume-from-0, s=0 mix(eps0), and record-x0 match the baseline")

    rows: list[EditRow] = []
    edited = np.zeros(
        (num_steps, random_trials, int(baseline_arm.shape[0]), action_dim),
        dtype=np.float32,
    )
    times = np.zeros(num_steps, dtype=np.float64)
    for step in range(num_steps):
        time_t = _time_at_step(step, num_steps)
        times[step] = time_t
        trial_metrics: list[detail.ActionMetrics] = []
        for trial in range(random_trials):
            eps = _draw_eps(
                request.noise,
                seed=int(random_seed) + trial * 1_000_003 + sample * 97 + noise * 13,
            )
            mixed = _mix_action(baseline, eps, time_t)
            with _patched_euler_from_step(runner, start_step=step):
                changed = _step_actions(adapter, replace(request, noise=mixed))
            arm = _cpu_arm(changed, action_dim)
            edited[step, trial] = arm.numpy()
            metrics = detail._action_metrics(arm, baseline_arm)
            trial_metrics.append(metrics)
            rows.append(
                EditRow(
                    sample=sample,
                    noise=noise,
                    trial=trial,
                    step=step,
                    n_steps=num_steps,
                    time_t=time_t,
                    metrics=metrics,
                )
            )
        mean_shift = detail._mean([row.arm_mean_shift_rmse for row in trial_metrics])
        local = detail._mean([row.arm_local_rmse for row in trial_metrics])
        share = detail._mean([_local_share(row) for row in trial_metrics])
        print(
            f"  s={step} t={time_t:.2f} mean={mean_shift:.3e} "
            f"local={local:.3e} local_share={share:.3f}"
        )
    traj = TrajRecord(
        sample=sample,
        noise=noise,
        n_steps=num_steps,
        time_t=times,
        baseline=baseline_arm.numpy().astype(np.float32, copy=True),
        x0_pred=x0_pred.astype(np.float32, copy=False),
        edited=edited,
    )
    return rows, traj


def _rows_by_step(rows: list[EditRow]) -> dict[int, list[EditRow]]:
    grouped: dict[int, list[EditRow]] = defaultdict(list)
    for row in rows:
        grouped[row.step].append(row)
    return dict(grouped)


def _mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot reduce empty metric list.")
    return float(array.mean()), float(array.std(ddof=0))


def _write_csv(rows: list[EditRow], output: Path) -> None:
    metric_names = [field.name for field in fields(detail.ActionMetrics)]
    fieldnames = [
        "sample",
        "noise",
        "trial",
        "step",
        "n_steps",
        "time_t",
        "local_share",
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
                "trial": row.trial,
                "step": row.step,
                "n_steps": row.n_steps,
                "time_t": row.time_t,
                "local_share": _local_share(row.metrics),
            }
            for name in metric_names:
                item[name] = getattr(row.metrics, name)
            writer.writerow(item)


def _plot(rows: list[EditRow], output: Path) -> None:
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
    share, share_std = zip(
        *[
            _mean_std([_local_share(row.metrics) for row in grouped[step]])
            for step in steps
        ]
    )

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), constrained_layout=True)
    axes[0].errorbar(xs, mean_shift, yerr=mean_shift_std, marker="o", capsize=3)
    axes[0].set_title("Arm mean shift vs original action")
    axes[0].set_ylabel("RMSE")
    axes[1].errorbar(xs, local, yerr=local_std, marker="o", capsize=3, color="C1")
    axes[1].set_title("Arm local residual vs original action")
    axes[1].set_ylabel("RMSE")
    axes[2].errorbar(xs, share, yerr=share_std, marker="s", capsize=3, color="C2")
    axes[2].set_title("Local share of remaining error")
    axes[2].set_ylabel("local / (mean_shift + local)")
    axes[2].set_ylim(0.0, 1.0)
    for ax in axes:
        ax.set_xlabel("resume denoise step s")
        ax.set_xticks(list(steps))
        ax.grid(alpha=0.25)
    fig.suptitle(
        "Corrupt a to t=1-s/N with fresh noise; resume Euler from step s",
        fontsize=12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Failed to write {output}.")


def _span(values: np.ndarray, pad: float = 0.08) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise RuntimeError("Cannot set axis limits on empty or non-finite values.")
    lo = float(finite.min())
    hi = float(finite.max())
    if hi == lo:
        delta = 1.0 if hi == 0.0 else abs(hi) * 0.05
        return lo - delta, hi + delta
    extra = (hi - lo) * pad
    return lo - extra, hi + extra


def _xy_limits(arms: list[np.ndarray], *, cumsum: bool) -> tuple[tuple[float, float], tuple[float, float]]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for arm in arms:
        xy = np.asarray(arm[:, :2], dtype=np.float64)
        if cumsum:
            xy = np.cumsum(xy, axis=0)
        xs.append(xy[:, 0])
        ys.append(xy[:, 1])
    return _span(np.concatenate(xs)), _span(np.concatenate(ys))


def _path_xy(arm: np.ndarray, *, cumsum: bool) -> np.ndarray:
    xy = np.asarray(arm[:, :2], dtype=np.float64)
    if cumsum:
        return np.cumsum(xy, axis=0)
    return xy


def _draw_path(ax, xy: np.ndarray, *, color: str, lw: float, alpha: float, zorder: int, label: str | None = None) -> None:
    ax.plot(xy[:, 0], xy[:, 1], color=color, lw=lw, alpha=alpha, zorder=zorder, label=label)
    ax.scatter([xy[0, 0]], [xy[0, 1]], s=16, color=color, alpha=min(1.0, alpha + 0.15), zorder=zorder + 1)
    ax.scatter(
        [xy[-1, 0]],
        [xy[-1, 1]],
        s=28,
        marker="*",
        color=color,
        alpha=min(1.0, alpha + 0.15),
        zorder=zorder + 1,
    )


def _draw_gripper(ax, arm: np.ndarray, *, color: str, lw: float, alpha: float, zorder: int, label: str | None = None) -> None:
    gripper = np.asarray(arm[:, -1], dtype=np.float64)
    ax.plot(np.arange(gripper.shape[0]), gripper, color=color, lw=lw, alpha=alpha, zorder=zorder, label=label)


def _trial_color(trial: int) -> str:
    return f"C{int(trial) % 10}"


def _plot_one_grid(
    record: TrajRecord,
    output: Path,
    *,
    kind: str,
) -> None:
    import matplotlib.pyplot as plt

    n_steps = int(record.n_steps)
    n_trials = int(record.edited.shape[1])
    if kind == "sdedit":
        arms = [record.baseline, *list(record.edited.reshape(-1, record.edited.shape[-2], record.edited.shape[-1]))]
        title = (
            f"SDEdit fan  sample={record.sample} noise={record.noise}  "
            "black=original, colors=fresh-noise resumes"
        )
    elif kind == "x0":
        arms = [record.baseline, *list(record.x0_pred)]
        title = (
            f"Predicted clean a_hat=x-t v  sample={record.sample} "
            f"noise={record.noise}  black=final action"
        )
    else:
        raise ValueError(f"Unknown plot kind {kind!r}.")
    raw_xlim, raw_ylim = _xy_limits(arms, cumsum=False)
    sum_xlim, sum_ylim = _xy_limits(arms, cumsum=True)
    grip_lim = _span(np.concatenate([arm[:, -1] for arm in arms]))
    fig, axes = plt.subplots(3, n_steps, figsize=(2.05 * n_steps, 7.6), squeeze=False)
    fig.suptitle(title, fontsize=11)
    for step in range(n_steps):
        t = float(record.time_t[step])
        ax_xy, ax_sum, ax_g = axes[0, step], axes[1, step], axes[2, step]
        if kind == "sdedit":
            for trial in range(n_trials):
                color = _trial_color(trial)
                arm = record.edited[step, trial]
                label = f"trial {trial}" if step == 0 else None
                _draw_path(ax_xy, _path_xy(arm, cumsum=False), color=color, lw=1.1, alpha=0.75, zorder=2, label=label)
                _draw_path(ax_sum, _path_xy(arm, cumsum=True), color=color, lw=1.1, alpha=0.75, zorder=2)
                _draw_gripper(ax_g, arm, color=color, lw=1.0, alpha=0.75, zorder=2)
        else:
            arm = record.x0_pred[step]
            _draw_path(ax_xy, _path_xy(arm, cumsum=False), color="C0", lw=1.4, alpha=0.95, zorder=3, label="a_hat")
            _draw_path(ax_sum, _path_xy(arm, cumsum=True), color="C0", lw=1.4, alpha=0.95, zorder=3)
            _draw_gripper(ax_g, arm, color="C0", lw=1.2, alpha=0.95, zorder=3)
        _draw_path(
            ax_xy,
            _path_xy(record.baseline, cumsum=False),
            color="black",
            lw=2.0,
            alpha=0.95,
            zorder=5,
            label="original" if step == 0 else None,
        )
        _draw_path(
            ax_sum,
            _path_xy(record.baseline, cumsum=True),
            color="black",
            lw=2.0,
            alpha=0.95,
            zorder=5,
        )
        _draw_gripper(ax_g, record.baseline, color="black", lw=1.8, alpha=0.95, zorder=5)
        ax_xy.set_xlim(*raw_xlim)
        ax_xy.set_ylim(*raw_ylim)
        ax_xy.set_aspect("equal", adjustable="box")
        ax_sum.set_xlim(*sum_xlim)
        ax_sum.set_ylim(*sum_ylim)
        ax_sum.set_aspect("equal", adjustable="box")
        ax_g.set_ylim(*grip_lim)
        ax_g.set_xlim(0, record.baseline.shape[0] - 1)
        ax_xy.set_title(f"s={step}  t={t:.2f}", fontsize=9)
        ax_xy.grid(alpha=0.2)
        ax_sum.grid(alpha=0.2)
        ax_g.grid(alpha=0.2)
        if step == 0:
            ax_xy.set_ylabel("action[0] vs [1]")
            ax_sum.set_ylabel("cumsum [0],[1]")
            ax_g.set_ylabel("gripper")
            ax_xy.legend(fontsize=7, loc="best")
        else:
            ax_xy.set_yticklabels([])
            ax_sum.set_yticklabels([])
            ax_g.set_yticklabels([])
        if step != n_steps // 2:
            ax_g.set_xlabel("")
        else:
            ax_g.set_xlabel("chunk time")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Failed to write {output}.")


def _overview_steps(n_steps: int) -> list[int]:
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}.")
    if n_steps <= 5:
        return list(range(n_steps))
    picks = [0, n_steps // 4, n_steps // 2, (3 * n_steps) // 4, n_steps - 1]
    return sorted(dict.fromkeys(picks))


def _chunks(items: list[TrajRecord], size: int) -> list[list[TrajRecord]]:
    if size < 1:
        raise ValueError(f"chunk size must be >= 1, got {size}.")
    return [items[i : i + size] for i in range(0, len(items), size)]


def _plot_overview(records: list[TrajRecord], output: Path) -> None:
    import matplotlib.pyplot as plt

    if not records:
        raise ValueError("Cannot plot an empty trajectory list.")
    n_steps = records[0].n_steps
    steps = _overview_steps(n_steps)
    fig, axes = plt.subplots(
        len(records),
        len(steps),
        figsize=(2.3 * len(steps), 2.15 * len(records)),
        squeeze=False,
    )
    fig.suptitle(
        "SDEdit xy fan (action[0] vs [1]). Early columns should spray if layout is still free.",
        fontsize=11,
    )
    for row, record in enumerate(records):
        arms = [record.baseline, *list(record.edited.reshape(-1, record.edited.shape[-2], record.edited.shape[-1]))]
        xlim, ylim = _xy_limits(arms, cumsum=False)
        n_trials = int(record.edited.shape[1])
        for col, step in enumerate(steps):
            ax = axes[row, col]
            for trial in range(n_trials):
                _draw_path(
                    ax,
                    _path_xy(record.edited[step, trial], cumsum=False),
                    color=_trial_color(trial),
                    lw=1.1,
                    alpha=0.75,
                    zorder=2,
                )
            _draw_path(
                ax,
                _path_xy(record.baseline, cumsum=False),
                color="black",
                lw=2.0,
                alpha=0.95,
                zorder=5,
            )
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.2)
            if row == 0:
                ax.set_title(f"s={step} t={record.time_t[step]:.2f}", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"sample {record.sample}\nnoise {record.noise}")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Failed to write {output}.")


def _condition_metric(
    rows: list[EditRow],
    sample: int,
    noise: int,
    steps: list[int],
    attr: str,
) -> float:
    values = [
        float(getattr(row.metrics, attr))
        for row in rows
        if row.sample == sample and row.noise == noise and row.step in steps
    ]
    if not values:
        raise RuntimeError(
            f"No rows for sample={sample} noise={noise} steps={steps} attr={attr}."
        )
    return detail._mean(values)


def _layout_rank_table(rows: list[EditRow]) -> list[dict[str, float | int]]:
    keys = sorted({(row.sample, row.noise) for row in rows})
    n_steps = rows[0].n_steps
    early = [step for step in (0, 1, 2) if step < n_steps]
    late = [step for step in (n_steps - 3, n_steps - 2, n_steps - 1) if step >= 0]
    table: list[dict[str, float | int]] = []
    for sample, noise in keys:
        early_mean = _condition_metric(rows, sample, noise, early, "arm_mean_shift_rmse")
        late_mean = _condition_metric(rows, sample, noise, late, "arm_mean_shift_rmse")
        s0_mean = _condition_metric(rows, sample, noise, [0], "arm_mean_shift_rmse")
        s0_end = _condition_metric(rows, sample, noise, [0], "arm_endpoint_rmse")
        table.append(
            {
                "sample": sample,
                "noise": noise,
                "early_mean_shift": early_mean,
                "late_mean_shift": late_mean,
                "s0_mean_shift": s0_mean,
                "s0_endpoint": s0_end,
                "early_over_late": early_mean / late_mean if late_mean > 0.0 else float("inf"),
            }
        )
    table.sort(key=lambda item: float(item["early_mean_shift"]), reverse=True)
    return table


def _write_layout_rank(table: list[dict[str, float | int]], output: Path) -> None:
    fieldnames = [
        "sample",
        "noise",
        "early_mean_shift",
        "late_mean_shift",
        "s0_mean_shift",
        "s0_endpoint",
        "early_over_late",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table)
    print("\n=== early path-change rank (s=0,1,2 mean_shift) ===")
    for item in table[:12]:
        print(
            f"  sample={item['sample']} noise={item['noise']}  "
            f"early={item['early_mean_shift']:.4e}  "
            f"s0_end={item['s0_endpoint']:.4e}  "
            f"late={item['late_mean_shift']:.4e}"
        )


def _records_by_key(records: list[TrajRecord]) -> dict[tuple[int, int], TrajRecord]:
    keyed: dict[tuple[int, int], TrajRecord] = {}
    for record in records:
        keyed[(record.sample, record.noise)] = record
    return keyed


def _top_layout_records(
    records: list[TrajRecord],
    table: list[dict[str, float | int]],
    *,
    top_k: int = 8,
) -> list[TrajRecord]:
    keyed = _records_by_key(records)
    picked: list[TrajRecord] = []
    for item in table:
        key = (int(item["sample"]), int(item["noise"]))
        if key in keyed:
            picked.append(keyed[key])
        if len(picked) >= top_k:
            break
    if not picked:
        raise RuntimeError("Layout rank did not match any trajectory records.")
    return picked


def _write_traj_npz(record: TrajRecord, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        sample=record.sample,
        noise=record.noise,
        n_steps=record.n_steps,
        time_t=record.time_t,
        baseline=record.baseline,
        x0_pred=record.x0_pred,
        edited=record.edited,
    )
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Failed to write {output}.")


def _plot_trajectories(
    records: list[TrajRecord],
    output_dir: Path,
    *,
    per_condition: bool = True,
) -> list[Path]:
    if not records:
        raise ValueError("Cannot plot an empty trajectory list.")
    written: list[Path] = []
    pages = _chunks(records, 6)
    for index, page in enumerate(pages):
        name = (
            "sdedit_fan_overview.png"
            if index == 0
            else f"sdedit_fan_overview_p{index + 1}.png"
        )
        overview = output_dir / name
        _plot_overview(page, overview)
        written.append(overview)
    for record in records:
        npz = output_dir / f"sample{record.sample}_noise{record.noise}.npz"
        _write_traj_npz(record, npz)
        written.append(npz)
        if not per_condition:
            continue
        stem = f"sample{record.sample}_noise{record.noise}"
        fan = output_dir / f"{stem}_sdedit_fan.png"
        x0 = output_dir / f"{stem}_x0_pred.png"
        _plot_one_grid(record, fan, kind="sdedit")
        _plot_one_grid(record, x0, kind="x0")
        written.extend([fan, x0])
    return written


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
    parser.add_argument("--random-seed", type=int, default=1000)
    parser.add_argument("--random-trials", type=int, default=4)
    parser.add_argument(
        "--per-condition-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write the 3-row fan/x0 grids for every sample. Disable for a scan.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--params-dtype", default="bfloat16")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tools/img/dit_denoise_sdedit_traj"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.model != "pi05":
        raise RuntimeError("SDEdit resume is implemented for pi05 only.")
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(args.checkpoint)
    if not args.calibration_data.is_file():
        raise FileNotFoundError(args.calibration_data)
    if args.sample_index < 0:
        raise ValueError("--sample-index must be >= 0.")
    if args.noise_seed < 0:
        raise ValueError("--noise-seed must be >= 0.")
    if args.random_trials < 1:
        raise ValueError("--random-trials must be >= 1.")
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
        f"random_trials={args.random_trials}, mode=sdedit-resume-from-s"
    )

    rows: list[EditRow] = []
    trajs: list[TrajRecord] = []
    for index, (sample, seed) in enumerate(conditions, start=1):
        request = matched._fixed_noise_request(
            adapter, batches[sample], runtime, noise_seed=seed
        )
        print(
            f"\n=== condition {index}/{len(conditions)} "
            f"sample={sample} noise={seed} ==="
        )
        sample_rows, traj = _experiment(
            adapter,
            request,
            runtime=runtime,
            sample=sample,
            noise=seed,
            random_trials=args.random_trials,
            random_seed=args.random_seed,
        )
        rows.extend(sample_rows)
        trajs.append(traj)

    verdict = _verdict(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "denoise_sdedit.csv"
    png = args.output_dir / "denoise_sdedit.png"
    verdict_path = args.output_dir / "hypothesis.txt"
    _write_csv(rows, csv_path)
    _plot(rows, png)
    rank_table = _layout_rank_table(rows)
    rank_csv = args.output_dir / "layout_rank.csv"
    _write_layout_rank(rank_table, rank_csv)
    traj_paths = _plot_trajectories(
        trajs, args.output_dir, per_condition=bool(args.per_condition_plots)
    )
    top_records = _top_layout_records(trajs, rank_table, top_k=8)
    top_png = args.output_dir / "sdedit_fan_layout_changed.png"
    _plot_overview(top_records, top_png)
    for record in top_records:
        stem = f"sample{record.sample}_noise{record.noise}"
        fan = args.output_dir / f"{stem}_sdedit_fan.png"
        x0 = args.output_dir / f"{stem}_x0_pred.png"
        if not fan.is_file():
            _plot_one_grid(record, fan, kind="sdedit")
        if not x0.is_file():
            _plot_one_grid(record, x0, kind="x0")
        traj_paths.extend([fan, x0])
    traj_paths.append(top_png)
    verdict_path.write_text(verdict + "\n", encoding="utf-8")
    for path in (csv_path, png, verdict_path, rank_csv, *traj_paths):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Failed to write {path}.")
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
