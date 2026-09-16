#!/usr/bin/env python
r"""Plot the predicted clean action a_hat at every Euler step of a normal generation.

This is *not* the live noisy state ``x_t``. After DiT predicts velocity ``v``,
the linear flow is inverted to the clean-action estimate:

    pi0.5:     x(t) = t ε + (1-t) a,   v = ε - a,   a_hat = x - t v
    GR00T-N1.7: x(t) = (1-t) ε + t a,  v = a - ε,   a_hat = x + (1-t) v

``t`` and ``v`` are this column's input time and DiT output. The next Euler
input is still ``x + dt v``, not ``a_hat``.

Does not modify ``analyze_dit_denoise_noisy_xt.py``.

Example::

    CUDA_VISIBLE_DEVICES=5 HF_ENDPOINT=https://hf-mirror.com uv run python \
      tools/analyze_dit_denoise_predicted_clean.py \
      --checkpoint /data/share/pi05_libero_finetuned_v044 \
      --calibration-data ../calibration_data/libero_goal_30_7_demo.npz \
      --samples 0 --noise-seeds 0,1,2,3,4 \
      --output-dir tools/img/dit_denoise_predicted_clean_sample0_5noise
"""

from __future__ import annotations

import argparse
import contextlib
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
_LAST_CLEAN_ATOL = 2e-3


@dataclass(frozen=True)
class CleanRecord:
    sample: int
    noise: int
    time_t: np.ndarray
    final_action: np.ndarray
    clean_by_step: np.ndarray


def _time_at_step(step: int, n_steps: int) -> float:
    if n_steps < 1 or not 0 <= int(step) < n_steps:
        raise ValueError(f"step={step} must be in [0, {n_steps}).")
    return 1.0 - float(step) / float(n_steps)


def _groot_time_at_step(step: int, n_steps: int) -> float:
    if n_steps < 1 or not 0 <= int(step) < n_steps:
        raise ValueError(f"step={step} must be in [0, {n_steps}).")
    return float(step) / float(n_steps)


def _predicted_clean(
    x_t: torch.Tensor, v_t: torch.Tensor, time_t: float
) -> torch.Tensor:
    """Invert pi0.5 flow: x = t ε + (1-t) a, v = ε - a."""
    if not 0.0 <= float(time_t) <= 1.0:
        raise ValueError(f"time_t must be in [0, 1], got {time_t}.")
    return x_t - float(time_t) * v_t.to(dtype=x_t.dtype)


def _predicted_clean_groot(
    x_t: torch.Tensor, v_t: torch.Tensor, time_t: float
) -> torch.Tensor:
    """Invert GR00T flow: x = (1-t) ε + t a, v = a - ε."""
    if not 0.0 <= float(time_t) <= 1.0:
        raise ValueError(f"time_t must be in [0, 1], got {time_t}.")
    return x_t + (1.0 - float(time_t)) * v_t.to(dtype=x_t.dtype)


def _euler_record_clean(
    runner, noise: torch.Tensor
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    n_steps = int(runner._num_steps)
    if n_steps < 1:
        raise RuntimeError(f"Euler loop has no steps (n_steps={n_steps}).")
    x_t = noise
    clean: list[torch.Tensor] = []
    for step in range(n_steps):
        v_t = runner._one_step(x_t, step)
        clean.append(
            _predicted_clean(x_t, v_t, _time_at_step(step, n_steps)).detach()
        )
        x_t = x_t + float(runner._dt) * v_t.to(dtype=x_t.dtype)
    return x_t, clean


@contextlib.contextmanager
def _patched_record_clean(runner):
    if getattr(runner, "graph", None) is not None:
        raise RuntimeError(
            "Recording a_hat needs the eager loop; runner has a CUDA graph."
        )
    original = runner._fwd_loop
    captured: list[list[torch.Tensor]] = []

    def _fwd_loop(*, noise: torch.Tensor) -> torch.Tensor:
        final, clean = _euler_record_clean(runner, noise)
        captured.clear()
        captured.append(clean)
        return final

    runner._fwd_loop = _fwd_loop
    try:
        yield captured
    finally:
        runner._fwd_loop = original


def _velocity_from_euler_update(
    x_t: torch.Tensor, x_next: torch.Tensor, dt: float
) -> torch.Tensor:
    step_dt = float(dt)
    if step_dt == 0.0:
        raise RuntimeError("Cannot recover velocity from an Euler step with dt=0.")
    return (x_next - x_t) / step_dt


@contextlib.contextmanager
def _patched_record_clean_groot(runner):
    if getattr(runner, "use_cuda_graph", False):
        raise RuntimeError(
            "Recording a_hat needs the eager loop; runner has a CUDA graph."
        )
    action_head = runner.model.action_head
    original = action_head.denoise_step
    n_steps = int(action_head.num_inference_timesteps)
    fallback_dt = float(runner._dt)
    cleans: list[torch.Tensor] = []
    captured: list[list[torch.Tensor]] = [cleans]
    step_box = [0]

    def _wrapper(actions, *args, **kwargs):
        step = int(step_box[0])
        if step >= n_steps:
            raise RuntimeError(
                f"denoise_step called {step + 1} times, expected {n_steps}."
            )
        time_t = _groot_time_at_step(step, n_steps)
        x_next = original(actions, *args, **kwargs)
        dt = float(kwargs["dt"]) if "dt" in kwargs else fallback_dt
        v_t = _velocity_from_euler_update(actions, x_next, dt)
        cleans.append(_predicted_clean_groot(actions, v_t, time_t).detach())
        step_box[0] = step + 1
        return x_next

    action_head.denoise_step = _wrapper
    try:
        yield captured
    finally:
        action_head.denoise_step = original


def _step_actions(adapter, request) -> torch.Tensor:
    with torch.inference_mode():
        actions = adapter.engine.step(request)
    if not torch.is_tensor(actions) or actions.ndim != 3 or actions.shape[0] != 1:
        raise RuntimeError(
            f"Expected actions (1,horizon,width), got "
            f"{type(actions)} {getattr(actions, 'shape', None)}."
        )
    detail._finite(actions.detach(), "actions")
    return actions.detach()


def _trim_action(actions: torch.Tensor, action_dim: int) -> np.ndarray:
    if int(actions.shape[-1]) < action_dim:
        raise RuntimeError(
            f"Action width {actions.shape[-1]} < action_dim={action_dim}."
        )
    return (
        actions[0, :, :action_dim]
        .detach()
        .to(dtype=torch.float32, device="cpu")
        .numpy()
        .astype(np.float64)
    )


def _valid_action_horizon(final_action: np.ndarray, *, atol: float = 1e-5) -> int:
    if final_action.ndim != 2 or final_action.shape[0] < 2:
        raise ValueError(
            f"final_action must have shape (horizon>=2, dims), got {final_action.shape}."
        )
    mag = np.max(np.abs(final_action), axis=1)
    nonzero = np.flatnonzero(mag > atol)
    if nonzero.size == 0:
        return int(final_action.shape[0])
    return int(nonzero[-1]) + 1


def _assert_last_clean_matches_final(
    clean_by_step: np.ndarray, final_action: np.ndarray
) -> None:
    keep = _valid_action_horizon(final_action)
    last_error = float(
        np.max(np.abs(clean_by_step[-1, :keep] - final_action[:keep]))
    )
    if last_error > _LAST_CLEAN_ATOL:
        raise RuntimeError(
            "Last predicted clean action should match Euler output on the "
            f"valid horizon; max error {last_error:.4e} is too large."
        )


def _record_condition(
    adapter,
    request,
    *,
    sample: int,
    noise: int,
    n_steps: int,
    action_dim: int,
) -> CleanRecord:
    if adapter.model_kind == "groot_n17":
        from qvla.adapters.groot.step_hook import find_action_head_runner

        runner = find_action_head_runner(adapter.engine)
        n_head = int(runner.model.action_head.num_inference_timesteps)
        if n_head != n_steps:
            raise RuntimeError(
                f"Action-head steps {n_head} != expected n_steps={n_steps}."
            )
        baseline = _step_actions(adapter, request)
        with _patched_record_clean_groot(runner) as captured:
            recorded_final = _step_actions(adapter, request)
        times = np.asarray(
            [_groot_time_at_step(step, n_steps) for step in range(n_steps)],
            dtype=np.float64,
        )
    elif adapter.model_kind == "pi05":
        from qvla.adapters.pi05.step_hook import find_expert_runner

        runner = find_expert_runner(adapter.engine)
        if int(runner._num_steps) != n_steps:
            raise RuntimeError(
                f"Runner steps {runner._num_steps} != expected n_steps={n_steps}."
            )
        baseline = _step_actions(adapter, request)
        with _patched_record_clean(runner) as captured:
            recorded_final = _step_actions(adapter, request)
        times = np.asarray(
            [_time_at_step(step, n_steps) for step in range(n_steps)],
            dtype=np.float64,
        )
    else:
        raise RuntimeError(f"Unsupported model_kind={adapter.model_kind!r}.")
    detail._assert_actions_equal(
        recorded_final.cpu(), baseline.cpu(), name="record-a_hat Euler patch"
    )
    if len(captured) != 1 or len(captured[0]) != n_steps:
        raise RuntimeError(f"Expected {n_steps} recorded clean predictions.")
    final_action = _trim_action(baseline, action_dim)
    clean_by_step = np.stack(
        [_trim_action(action, action_dim) for action in captured[0]], axis=0
    )
    _assert_last_clean_matches_final(clean_by_step, final_action)
    return CleanRecord(
        sample=sample,
        noise=noise,
        time_t=times,
        final_action=final_action,
        clean_by_step=clean_by_step,
    )


def _crop_record(record: CleanRecord, keep: int) -> CleanRecord:
    horizon = int(record.final_action.shape[0])
    if keep < 2 or keep > horizon:
        raise ValueError(f"keep-horizon={keep} must be in [2, {horizon}].")
    if keep == horizon:
        return record
    return CleanRecord(
        sample=record.sample,
        noise=record.noise,
        time_t=record.time_t,
        final_action=record.final_action[:keep],
        clean_by_step=record.clean_by_step[:, :keep, :],
    )


def _records_for_plots(
    records: list[CleanRecord], keep_horizon: int | None
) -> list[CleanRecord]:
    if not records:
        return []
    if keep_horizon is not None:
        keep = int(keep_horizon)
    else:
        keep = min(_valid_action_horizon(record.final_action) for record in records)
    plotted = [_crop_record(record, keep) for record in records]
    full = int(records[0].final_action.shape[0])
    if keep < full:
        print(f"Plotting first {keep} of {full} action steps (padding dropped).")
    return plotted


def _span(values: np.ndarray, pad: float = 0.08) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise RuntimeError("Cannot set axis limits on empty values.")
    lo = float(finite.min())
    hi = float(finite.max())
    if hi == lo:
        delta = 1.0 if hi == 0.0 else abs(hi) * 0.05
        return lo - delta, hi + delta
    extra = (hi - lo) * pad
    return lo - extra, hi + extra


def _ylim_from_final(*finals: np.ndarray) -> tuple[float, float]:
    if not finals:
        raise ValueError("Need at least one series for y-limits.")
    return _span(
        np.concatenate(
            [np.asarray(values, dtype=np.float64).ravel() for values in finals]
        )
    )


def _rmse(delta: np.ndarray) -> float:
    values = np.asarray(delta, dtype=np.float64)
    if values.size == 0:
        raise ValueError("Cannot compute RMSE on empty values.")
    return float(np.sqrt(np.mean(np.square(values))))


def _rmse_to_final(record: CleanRecord) -> np.ndarray:
    delta = record.clean_by_step - record.final_action[None, :, :]
    n_steps = int(delta.shape[0])
    return np.asarray(
        [_rmse(delta[step]) for step in range(n_steps)], dtype=np.float64
    )


def _rmse_to_final_per_dim(record: CleanRecord) -> np.ndarray:
    delta = record.clean_by_step - record.final_action[None, :, :]
    n_steps, _, n_dims = delta.shape
    out = np.empty((n_steps, n_dims), dtype=np.float64)
    for step in range(n_steps):
        for dim in range(n_dims):
            out[step, dim] = _rmse(delta[step, :, dim])
    return out


def _pairwise_rmse(records: list[CleanRecord]) -> np.ndarray:
    if len(records) < 2:
        raise ValueError("Need at least two records for pairwise RMSE.")
    n_steps = int(records[0].clean_by_step.shape[0])
    out = np.empty(n_steps, dtype=np.float64)
    for step in range(n_steps):
        errors: list[float] = []
        for i, left in enumerate(records):
            for right in records[i + 1 :]:
                errors.append(
                    _rmse(left.clean_by_step[step] - right.clean_by_step[step])
                )
        out[step] = float(np.mean(errors))
    return out


def _dim_label(index: int) -> str:
    if 0 <= index < len(DIM_LABELS):
        return DIM_LABELS[index]
    return f"dim {index}"


def _plot_one(record: CleanRecord, output: Path) -> None:
    import matplotlib.pyplot as plt

    n_steps = int(record.clean_by_step.shape[0])
    horizon = int(record.final_action.shape[0])
    xy_all = np.concatenate(
        [record.clean_by_step[:, :, :2].reshape(-1, 2), record.final_action[:, :2]],
        axis=0,
    )
    xlim = _span(xy_all[:, 0])
    ylim = _span(xy_all[:, 1])
    y0_lim = _ylim_from_final(record.final_action[:, 0], record.clean_by_step[:, :, 0])
    fig, axes = plt.subplots(2, n_steps, figsize=(2.05 * n_steps, 5.4), squeeze=False)
    fig.suptitle(
        f"Predicted clean a_hat=x-tv  sample={record.sample} noise={record.noise}  "
        "black=final action; time-row ylim fixed for the row",
        fontsize=11,
    )
    for step in range(n_steps):
        ax_xy, ax_t = axes[0, step], axes[1, step]
        clean = record.clean_by_step[step]
        ax_xy.plot(
            record.final_action[:, 0],
            record.final_action[:, 1],
            color="black",
            lw=2.0,
            zorder=5,
            label="final" if step == 0 else None,
        )
        ax_xy.plot(
            clean[:, 0],
            clean[:, 1],
            color="C0",
            lw=1.0,
            alpha=0.85,
            zorder=3,
            label="a_hat" if step == 0 else None,
        )
        ax_xy.scatter([clean[0, 0]], [clean[0, 1]], s=16, color="C0", zorder=4)
        ax_xy.scatter(
            [clean[-1, 0]], [clean[-1, 1]], s=28, marker="*", color="C0", zorder=4
        )
        ax_xy.set_xlim(*xlim)
        ax_xy.set_ylim(*ylim)
        ax_xy.set_aspect("equal", adjustable="box")
        ax_xy.grid(alpha=0.2)
        ax_xy.set_title(f"s={step}  t={record.time_t[step]:.2f}", fontsize=9)
        ax_t.plot(
            np.arange(horizon),
            record.final_action[:, 0],
            color="black",
            lw=1.8,
            zorder=5,
        )
        ax_t.plot(
            np.arange(horizon), clean[:, 0], color="C0", lw=1.0, alpha=0.85, zorder=3
        )
        ax_t.set_xlim(0, horizon - 1)
        ax_t.set_ylim(*y0_lim)
        ax_t.grid(alpha=0.2)
        if step == 0:
            ax_xy.set_ylabel("action[0] vs [1]")
            ax_t.set_ylabel("action[0] vs time")
            ax_xy.legend(fontsize=7, loc="best")
        else:
            ax_xy.set_yticklabels([])
            ax_t.set_yticklabels([])
        if step == n_steps // 2:
            ax_t.set_xlabel("chunk time")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Failed to write {output}.")


def _plot_all_dims(record: CleanRecord, output: Path) -> None:
    import matplotlib.pyplot as plt

    n_steps = int(record.clean_by_step.shape[0])
    n_dims = int(record.final_action.shape[1])
    horizon = int(record.final_action.shape[0])
    fig, axes = plt.subplots(
        n_dims,
        n_steps,
        figsize=(2.05 * n_steps, 1.55 * n_dims),
        squeeze=False,
    )
    fig.suptitle(
        f"All action dims of a_hat=x-tv  sample={record.sample} noise={record.noise}  "
        "black=final, blue=a_hat; ylim fixed per dim",
        fontsize=11,
    )
    time = np.arange(horizon)
    for dim in range(n_dims):
        ylim = _ylim_from_final(
            record.final_action[:, dim],
            record.clean_by_step[:, :, dim],
        )
        for step in range(n_steps):
            ax = axes[dim, step]
            ax.plot(
                time,
                record.final_action[:, dim],
                color="black",
                lw=1.8,
                zorder=5,
                label="final" if dim == 0 and step == 0 else None,
            )
            ax.plot(
                time,
                record.clean_by_step[step, :, dim],
                color="C0",
                lw=1.0,
                alpha=0.85,
                zorder=3,
                label="a_hat" if dim == 0 and step == 0 else None,
            )
            ax.set_xlim(0, horizon - 1)
            ax.set_ylim(*ylim)
            ax.grid(alpha=0.2)
            if dim == 0:
                ax.set_title(f"s={step}  t={record.time_t[step]:.2f}", fontsize=9)
            if step == 0:
                ax.set_ylabel(_dim_label(dim), fontsize=8)
            else:
                ax.set_yticklabels([])
            if dim != n_dims - 1:
                ax.set_xticklabels([])
            elif step == n_steps // 2:
                ax.set_xlabel("chunk time")
            if dim == 0 and step == 0:
                ax.legend(fontsize=7, loc="best")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Failed to write {output}.")


def _plot_all_dims_overlay(records: list[CleanRecord], output: Path) -> None:
    import matplotlib.pyplot as plt

    if len(records) < 2:
        raise ValueError("Need at least two noise records to overlay.")
    samples = {int(record.sample) for record in records}
    if len(samples) != 1:
        raise ValueError(f"Overlay requires a single sample, got {sorted(samples)}.")
    n_steps = int(records[0].clean_by_step.shape[0])
    n_dims = int(records[0].final_action.shape[1])
    horizon = int(records[0].final_action.shape[0])
    for record in records:
        if record.clean_by_step.shape != (n_steps, horizon, n_dims):
            raise ValueError("All overlay records must share clean_by_step shape.")
        if record.final_action.shape != (horizon, n_dims):
            raise ValueError("All overlay records must share final_action shape.")
    fig, axes = plt.subplots(
        n_dims,
        n_steps,
        figsize=(2.05 * n_steps, 1.55 * n_dims),
        squeeze=False,
    )
    fig.suptitle(
        f"All action dims of a_hat=x-tv  sample={records[0].sample}  "
        "solid=a_hat, dashed=final, color=noise seed; ylim fixed per dim",
        fontsize=11,
    )
    time = np.arange(horizon)
    for dim in range(n_dims):
        ylim = _ylim_from_final(
            *(record.final_action[:, dim] for record in records),
            *(record.clean_by_step[:, :, dim] for record in records),
        )
        for step in range(n_steps):
            ax = axes[dim, step]
            for index, record in enumerate(records):
                color = f"C{index % 10}"
                label = f"n={record.noise}" if dim == 0 and step == 0 else None
                ax.plot(
                    time,
                    record.final_action[:, dim],
                    color=color,
                    lw=1.2,
                    ls="--",
                    alpha=0.7,
                    zorder=4,
                )
                ax.plot(
                    time,
                    record.clean_by_step[step, :, dim],
                    color=color,
                    lw=1.0,
                    alpha=0.9,
                    zorder=5,
                    label=label,
                )
            ax.set_xlim(0, horizon - 1)
            ax.set_ylim(*ylim)
            ax.grid(alpha=0.2)
            if dim == 0:
                ax.set_title(f"s={step}  t={records[0].time_t[step]:.2f}", fontsize=9)
            if step == 0:
                ax.set_ylabel(_dim_label(dim), fontsize=8)
            else:
                ax.set_yticklabels([])
            if dim != n_dims - 1:
                ax.set_xticklabels([])
            elif step == n_steps // 2:
                ax.set_xlabel("chunk time")
            if dim == 0 and step == 0:
                ax.legend(fontsize=7, loc="best", ncol=min(3, len(records)))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Failed to write {output}.")


def _plot_finals_across_noises(records: list[CleanRecord], output: Path) -> None:
    import matplotlib.pyplot as plt

    if len(records) < 2:
        raise ValueError("Need at least two noise records to overlay.")
    samples = {int(record.sample) for record in records}
    if len(samples) != 1:
        raise ValueError(f"Overlay requires a single sample, got {sorted(samples)}.")
    n_dims = int(records[0].final_action.shape[1])
    horizon = int(records[0].final_action.shape[0])
    fig, axes = plt.subplots(n_dims, 1, figsize=(8.5, 1.55 * n_dims), sharex=True)
    fig.suptitle(
        f"Final actions across noise seeds  sample={records[0].sample}",
        fontsize=11,
    )
    time = np.arange(horizon)
    for dim, ax in enumerate(np.atleast_1d(axes)):
        for index, record in enumerate(records):
            ax.plot(
                time,
                record.final_action[:, dim],
                color=f"C{index % 10}",
                lw=1.6,
                label=f"n={record.noise}" if dim == 0 else None,
            )
        ax.set_ylabel(_dim_label(dim), fontsize=8)
        ax.grid(alpha=0.2)
        if dim == 0:
            ax.legend(fontsize=8, loc="best", ncol=min(5, len(records)))
    np.atleast_1d(axes)[-1].set_xlabel("chunk time")
    np.atleast_1d(axes)[-1].set_xlim(0, horizon - 1)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Failed to write {output}.")


def _plot_rmse(records: list[CleanRecord], output: Path) -> None:
    import matplotlib.pyplot as plt

    if not records:
        raise ValueError("Cannot plot RMSE for an empty record list.")
    n_steps = int(records[0].clean_by_step.shape[0])
    n_dims = int(records[0].final_action.shape[1])
    times = np.asarray(records[0].time_t, dtype=np.float64)
    rmse_all = np.stack([_rmse_to_final(record) for record in records], axis=0)
    rmse_dim = np.stack(
        [_rmse_to_final_per_dim(record) for record in records], axis=0
    )
    steps = np.arange(n_steps)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    ax_all, ax_dim = axes
    mean = rmse_all.mean(axis=0)
    lo = rmse_all.min(axis=0)
    hi = rmse_all.max(axis=0)
    ax_all.fill_between(steps, lo, hi, color="C0", alpha=0.18, label="min–max")
    ax_all.plot(steps, mean, color="C0", lw=2.0, marker="o", label="mean over noise")
    for record, curve in zip(records, rmse_all, strict=True):
        ax_all.plot(
            steps,
            curve,
            color="0.55",
            lw=0.8,
            alpha=0.7,
            label=f"n={record.noise}" if len(records) <= 5 else None,
        )
    ax_all.set_xlabel("denoise step s")
    ax_all.set_ylabel("RMSE(a_hat_s, a_final)")
    ax_all.set_title("Overall")
    ax_all.set_xlim(0, n_steps - 1)
    ax_all.grid(alpha=0.2)
    ax_all.legend(fontsize=7, loc="best")
    mean_dim = rmse_dim.mean(axis=0)
    for dim in range(n_dims):
        ax_dim.plot(
            steps,
            mean_dim[:, dim],
            lw=1.6,
            marker="o",
            ms=3.5,
            label=_dim_label(dim),
        )
    ax_dim.set_xlabel("denoise step s")
    ax_dim.set_ylabel("RMSE(a_hat_s, a_final)")
    ax_dim.set_title("Per dimension (mean over noise)")
    ax_dim.set_xlim(0, n_steps - 1)
    ax_dim.grid(alpha=0.2)
    ax_dim.legend(fontsize=7, loc="best", ncol=2)
    t_note = ", ".join(f"s{s}:t={times[s]:.2f}" for s in (0, n_steps - 1))
    fig.suptitle(
        f"Predicted-clean error vs final  sample={records[0].sample}  ({t_note})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Failed to write {output}.")


def _plot_overview(records: list[CleanRecord], output: Path) -> None:
    import matplotlib.pyplot as plt

    if not records:
        raise ValueError("Cannot plot an empty record list.")
    n_steps = records[0].clean_by_step.shape[0]
    steps = [0, n_steps // 4, n_steps // 2, (3 * n_steps) // 4, n_steps - 1]
    steps = sorted(dict.fromkeys(step for step in steps if 0 <= step < n_steps))
    fig, axes = plt.subplots(
        len(records),
        len(steps),
        figsize=(2.4 * len(steps), 2.3 * len(records)),
        squeeze=False,
    )
    fig.suptitle(
        "Predicted clean a_hat=x-tv at selected denoise steps (black = final)",
        fontsize=11,
    )
    for row, record in enumerate(records):
        xy_all = np.concatenate(
            [record.clean_by_step[:, :, :2].reshape(-1, 2), record.final_action[:, :2]],
            axis=0,
        )
        xlim = _span(xy_all[:, 0])
        ylim = _span(xy_all[:, 1])
        for col, step in enumerate(steps):
            ax = axes[row, col]
            clean = record.clean_by_step[step]
            ax.plot(
                record.final_action[:, 0],
                record.final_action[:, 1],
                color="black",
                lw=2.0,
                zorder=5,
            )
            ax.plot(clean[:, 0], clean[:, 1], color="C0", lw=1.0, alpha=0.85, zorder=3)
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


def _compare_paths_for_sample(sample: int, output_dir: Path) -> tuple[Path, Path, Path]:
    return (
        output_dir / f"sample{sample}_noises_ahat_dims.png",
        output_dir / f"sample{sample}_noises_finals.png",
        output_dir / f"sample{sample}_noises_ahat_rmse.png",
    )


def _plot_same_sample_noise_compare(
    records: list[CleanRecord], output_dir: Path
) -> list[Path]:
    by_sample: dict[int, list[CleanRecord]] = {}
    for record in records:
        by_sample.setdefault(int(record.sample), []).append(record)
    written: list[Path] = []
    for sample, group in sorted(by_sample.items()):
        if len(group) < 2:
            continue
        overlay, finals, rmse = _compare_paths_for_sample(sample, output_dir)
        _plot_all_dims_overlay(group, overlay)
        _plot_finals_across_noises(group, finals)
        _plot_rmse(group, rmse)
        written.extend([overlay, finals, rmse])
    return written


def _load_npz(path: Path) -> CleanRecord:
    with np.load(path) as data:
        return CleanRecord(
            sample=int(data["sample"]),
            noise=int(data["noise"]),
            time_t=np.asarray(data["time_t"], dtype=np.float64),
            final_action=np.asarray(data["final_action"], dtype=np.float64),
            clean_by_step=np.asarray(data["clean_by_step"], dtype=np.float64),
        )


def _write_npz(record: CleanRecord, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        sample=record.sample,
        noise=record.noise,
        time_t=record.time_t,
        final_action=record.final_action,
        clean_by_step=record.clean_by_step,
    )
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Failed to write {output}.")


def _rmse_rows(records: list[CleanRecord]) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for record in records:
        overall = _rmse_to_final(record)
        per_dim = _rmse_to_final_per_dim(record)
        for step, time_t in enumerate(record.time_t):
            row: dict[str, float | int] = {
                "sample": int(record.sample),
                "noise": int(record.noise),
                "step": int(step),
                "time_t": float(time_t),
                "rmse_all": float(overall[step]),
            }
            for dim in range(per_dim.shape[1]):
                row[f"rmse_d{dim}"] = float(per_dim[step, dim])
            rows.append(row)
    return rows


def _write_rmse_csv(rows: list[dict[str, float | int]], output: Path) -> None:
    if not rows:
        raise ValueError("Cannot write an empty RMSE table.")
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Failed to write {output}.")


def _write_summary(records: list[CleanRecord], output: Path) -> None:
    lines = [
        "Predicted clean a_hat vs final action.",
        "pi0.5: a_hat = x - t v.  GR00T-N1.7: a_hat = x + (1-t) v.",
        "",
    ]
    by_sample: dict[int, list[CleanRecord]] = {}
    for record in records:
        by_sample.setdefault(int(record.sample), []).append(record)
    for sample, group in sorted(by_sample.items()):
        stacked = np.stack([_rmse_to_final(record) for record in group], axis=0)
        mean = stacked.mean(axis=0)
        lines.append(f"sample {sample}  n_noise={len(group)}")
        for step, time_t in enumerate(group[0].time_t):
            lines.append(
                f"  s={step:02d}  t={time_t:.3f}  "
                f"RMSE(a_hat,final) mean={mean[step]:.4f}  "
                f"min={stacked[:, step].min():.4f}  max={stacked[:, step].max():.4f}"
            )
        if len(group) >= 2:
            pair = _pairwise_rmse(group)
            lines.append("  pairwise RMSE of a_hat across noise seeds:")
            for step, value in enumerate(pair):
                lines.append(f"    s={step:02d}  {value:.4f}")
        lines.append("")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Failed to write {output}.")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--calibration-data", type=Path, default=None)
    parser.add_argument(
        "--from-npz-dir",
        type=Path,
        default=None,
        help="Plot from saved *.npz instead of running the model.",
    )
    matched.add_model_cli(parser)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--samples", default=None)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--noise-seeds", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--params-dtype", default="bfloat16")
    parser.add_argument(
        "--keep-horizon",
        type=int,
        default=None,
        help="Plot only the first N chunk steps. Default: drop a trailing all-zero pad.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tools/img/dit_denoise_predicted_clean"),
    )
    return parser


def _write_record_plots(
    record: CleanRecord,
    output_dir: Path,
    *,
    plot_record: CleanRecord | None = None,
) -> list[Path]:
    plotted = record if plot_record is None else plot_record
    stem = f"sample{record.sample}_noise{record.noise}"
    grid = output_dir / f"{stem}_predicted_clean.png"
    dims = output_dir / f"{stem}_predicted_clean_dims.png"
    npz = output_dir / f"{stem}.npz"
    _plot_one(plotted, grid)
    _plot_all_dims(plotted, dims)
    if not npz.is_file():
        _write_npz(record, npz)
    return [grid, dims, npz]


def _write_group_stats(
    records: list[CleanRecord], plotted: list[CleanRecord], output_dir: Path
) -> list[Path]:
    csv_path = output_dir / "predicted_clean_rmse.csv"
    summary_path = output_dir / "summary.txt"
    _write_rmse_csv(_rmse_rows(plotted), csv_path)
    _write_summary(plotted, summary_path)
    written = [csv_path, summary_path]
    by_sample: dict[int, list[CleanRecord]] = {}
    for record in plotted:
        by_sample.setdefault(int(record.sample), []).append(record)
    for sample, group in sorted(by_sample.items()):
        if len(group) == 1:
            rmse_path = output_dir / f"sample{sample}_noises_ahat_rmse.png"
            _plot_rmse(group, rmse_path)
            written.append(rmse_path)
    return written


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.from_npz_dir is not None:
        npz_dir = args.from_npz_dir
        if not npz_dir.is_dir():
            raise FileNotFoundError(npz_dir)
        records = [_load_npz(path) for path in sorted(npz_dir.glob("sample*_noise*.npz"))]
        if not records:
            raise RuntimeError(f"No sample*_noise*.npz files in {npz_dir}.")
        plotted = _records_for_plots(records, args.keep_horizon)
        npz_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for record, plot_record in zip(records, plotted, strict=True):
            written.extend(
                _write_record_plots(record, npz_dir, plot_record=plot_record)
            )
        written.extend(_plot_same_sample_noise_compare(plotted, npz_dir))
        written.extend(_write_group_stats(records, plotted, npz_dir))
        for path in written:
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Failed to write {path}.")
            print(f"Wrote {path}")
        return 0
    if args.model not in ("pi05", "groot_n17"):
        raise RuntimeError(f"Unsupported --model {args.model}.")
    if args.checkpoint is None or args.calibration_data is None:
        raise RuntimeError("Need --checkpoint and --calibration-data, or --from-npz-dir.")
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
    runtime = matched.clip_runtime(adapter, QVLAConfig.for_model_kind(args.model))
    need = max(sample_ids) + 1
    batches = list(adapter.iter_calibration_batches(need))
    if len(batches) != need:
        raise RuntimeError(f"Calibration yielded {len(batches)}, need {need}.")
    conditions = [(sample, noise) for sample in sample_ids for noise in noise_ids]
    print(
        f"model={args.model} conditions={len(conditions)} steps={runtime.num_steps} "
        f"horizon={runtime.action_horizon} action_dim={runtime.action_dim}"
    )
    records: list[CleanRecord] = []
    for index, (sample, noise) in enumerate(conditions, start=1):
        request = matched._fixed_noise_request(
            adapter, batches[sample], runtime, noise_seed=noise
        )
        record = _record_condition(
            adapter,
            request,
            sample=sample,
            noise=noise,
            n_steps=runtime.num_steps,
            action_dim=runtime.action_dim,
        )
        records.append(record)
        print(f"  [{index}/{len(conditions)}] sample={sample} noise={noise}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    plotted = _records_for_plots(records, args.keep_horizon)
    written = [output_dir / "predicted_clean_overview.png"]
    _plot_overview(plotted, written[0])
    for record, plot_record in zip(records, plotted, strict=True):
        written.extend(
            _write_record_plots(record, output_dir, plot_record=plot_record)
        )
    written.extend(_plot_same_sample_noise_compare(plotted, output_dir))
    written.extend(_write_group_stats(records, plotted, output_dir))
    for path in written:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Failed to write {path}.")
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
