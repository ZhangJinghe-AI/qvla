#!/usr/bin/env python
r"""Plot the live noisy action x_t at every Euler step of a normal generation.

This is the raw state of the denoiser, not the predicted clean action
``a_hat = x - t v``. pi0.5 uses ``x <- x + dt * v`` with ``t = 1 - s/N``.
The panel for step s shows ``x`` *before* that step's update.

pi0.5: ``t = 1 - s/N``, ``dt = -1/N`` (noise at t=1).
GR00T-N1.7: ``t = s/N``, ``dt = +1/N`` (noise at t=0). Both start from
Gaussian noise and Euler-integrate the flow.

Example::

    CUDA_VISIBLE_DEVICES=5 HF_ENDPOINT=https://hf-mirror.com uv run python \
      tools/analyze_dit_denoise_noisy_xt.py \
      --checkpoint /data/share/pi05_libero_finetuned_v044 \
      --calibration-data ../calibration_data/libero_goal_30_7_demo.npz \
      --samples 0,1,2 --noise-seeds 0 \
      --output-dir tools/img/dit_denoise_noisy_xt

    CUDA_VISIBLE_DEVICES=5 HF_ENDPOINT=https://hf-mirror.com uv run python \
      tools/analyze_dit_denoise_noisy_xt.py \
      --model groot_n17 \
      --checkpoint /data/share/GR00T-N1.7-LIBERO/libero_goal \
      --calibration-data ../calibration_data/libero_goal_30_7_demo.npz \
      --samples 0 --noise-seeds 0,1,2,3,4 \
      --output-dir tools/img/dit_denoise_noisy_xt_groot_sample0_5noise
"""

from __future__ import annotations

import argparse
import contextlib
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


@dataclass(frozen=True)
class NoisyRecord:
    sample: int
    noise: int
    time_t: np.ndarray
    final_action: np.ndarray
    noisy_by_step: np.ndarray


def _time_at_step(step: int, n_steps: int) -> float:
    if n_steps < 1 or not 0 <= int(step) < n_steps:
        raise ValueError(f"step={step} must be in [0, {n_steps}).")
    return 1.0 - float(step) / float(n_steps)


def _groot_time_at_step(step: int, n_steps: int) -> float:
    if n_steps < 1 or not 0 <= int(step) < n_steps:
        raise ValueError(f"step={step} must be in [0, {n_steps}).")
    return float(step) / float(n_steps)


def _euler_record_noisy(
    runner, noise: torch.Tensor
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    n_steps = int(runner._num_steps)
    if n_steps < 1:
        raise RuntimeError(f"Euler loop has no steps (n_steps={n_steps}).")
    x_t = noise
    states: list[torch.Tensor] = []
    for step in range(n_steps):
        states.append(x_t.detach())
        v_t = runner._one_step(x_t, step)
        x_t = x_t + float(runner._dt) * v_t.to(dtype=x_t.dtype)
    return x_t, states


@contextlib.contextmanager
def _patched_record_noisy(runner):
    if getattr(runner, "graph", None) is not None:
        raise RuntimeError(
            "Recording x_t needs the eager loop; runner has a CUDA graph."
        )
    original = runner._fwd_loop
    captured: list[list[torch.Tensor]] = []

    def _fwd_loop(*, noise: torch.Tensor) -> torch.Tensor:
        final, states = _euler_record_noisy(runner, noise)
        captured.clear()
        captured.append(states)
        return final

    runner._fwd_loop = _fwd_loop
    try:
        yield captured
    finally:
        runner._fwd_loop = original


@contextlib.contextmanager
def _patched_record_noisy_groot(runner):
    if getattr(runner, "use_cuda_graph", False):
        raise RuntimeError(
            "Recording x_t needs the eager loop; runner has a CUDA graph."
        )
    action_head = runner.model.action_head
    original = action_head.denoise_step
    states: list[torch.Tensor] = []
    captured: list[list[torch.Tensor]] = [states]

    def _wrapper(actions, *args, **kwargs):
        states.append(actions.detach().clone())
        return original(actions, *args, **kwargs)

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


def _record_condition(
    adapter,
    request,
    *,
    sample: int,
    noise: int,
    n_steps: int,
    action_dim: int,
) -> NoisyRecord:
    if adapter.model_kind == "groot_n17":
        from qvla.adapters.groot.step_hook import find_action_head_runner

        runner = find_action_head_runner(adapter.engine)
        n_head = int(runner.model.action_head.num_inference_timesteps)
        if n_head != n_steps:
            raise RuntimeError(
                f"Action-head steps {n_head} != expected n_steps={n_steps}."
            )
        baseline = _step_actions(adapter, request)
        with _patched_record_noisy_groot(runner) as captured:
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
        with _patched_record_noisy(runner) as captured:
            recorded_final = _step_actions(adapter, request)
        times = np.asarray(
            [_time_at_step(step, n_steps) for step in range(n_steps)],
            dtype=np.float64,
        )
    else:
        raise RuntimeError(f"Unsupported model_kind={adapter.model_kind!r}.")
    detail._assert_actions_equal(
        recorded_final.cpu(), baseline.cpu(), name="record-x_t Euler patch"
    )
    if len(captured) != 1 or len(captured[0]) != n_steps:
        raise RuntimeError(f"Expected {n_steps} recorded noisy states.")
    final_action = _trim_action(baseline, action_dim)
    noisy_by_step = np.stack(
        [_trim_action(state, action_dim) for state in captured[0]], axis=0
    )
    return NoisyRecord(
        sample=sample,
        noise=noise,
        time_t=times,
        final_action=final_action,
        noisy_by_step=noisy_by_step,
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


def _crop_record(record: NoisyRecord, keep: int) -> NoisyRecord:
    horizon = int(record.final_action.shape[0])
    if keep < 2 or keep > horizon:
        raise ValueError(f"keep-horizon={keep} must be in [2, {horizon}].")
    if keep == horizon:
        return record
    return NoisyRecord(
        sample=record.sample,
        noise=record.noise,
        time_t=record.time_t,
        final_action=record.final_action[:keep],
        noisy_by_step=record.noisy_by_step[:, :keep, :],
    )


def _records_for_plots(
    records: list[NoisyRecord], keep_horizon: int | None
) -> list[NoisyRecord]:
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


def _plot_one(record: NoisyRecord, output: Path) -> None:
    import matplotlib.pyplot as plt

    n_steps = int(record.noisy_by_step.shape[0])
    horizon = int(record.final_action.shape[0])
    xy_all = np.concatenate(
        [record.noisy_by_step[:, :, :2].reshape(-1, 2), record.final_action[:, :2]],
        axis=0,
    )
    xlim = _span(xy_all[:, 0])
    ylim = _span(xy_all[:, 1])
    y0_lim = _ylim_from_final(record.final_action[:, 0], record.noisy_by_step[:, :, 0])
    fig, axes = plt.subplots(2, n_steps, figsize=(2.05 * n_steps, 5.4), squeeze=False)
    fig.suptitle(
        f"Raw noisy x_t  sample={record.sample} noise={record.noise}  "
        "black=final action; time-row ylim fixed for the row",
        fontsize=11,
    )
    for step in range(n_steps):
        ax_xy, ax_t = axes[0, step], axes[1, step]
        noisy = record.noisy_by_step[step]
        ax_xy.plot(
            record.final_action[:, 0],
            record.final_action[:, 1],
            color="black",
            lw=2.0,
            zorder=5,
            label="final" if step == 0 else None,
        )
        ax_xy.plot(
            noisy[:, 0],
            noisy[:, 1],
            color="C3",
            lw=1.0,
            alpha=0.85,
            zorder=3,
            label="x_t" if step == 0 else None,
        )
        ax_xy.scatter([noisy[0, 0]], [noisy[0, 1]], s=16, color="C3", zorder=4)
        ax_xy.scatter(
            [noisy[-1, 0]], [noisy[-1, 1]], s=28, marker="*", color="C3", zorder=4
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
            np.arange(horizon), noisy[:, 0], color="C3", lw=1.0, alpha=0.85, zorder=3
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


def _dim_label(index: int) -> str:
    if 0 <= index < len(DIM_LABELS):
        return DIM_LABELS[index]
    return f"dim {index}"


def _plot_all_dims(record: NoisyRecord, output: Path) -> None:
    import matplotlib.pyplot as plt

    n_steps = int(record.noisy_by_step.shape[0])
    n_dims = int(record.final_action.shape[1])
    horizon = int(record.final_action.shape[0])
    fig, axes = plt.subplots(
        n_dims,
        n_steps,
        figsize=(2.05 * n_steps, 1.55 * n_dims),
        squeeze=False,
    )
    fig.suptitle(
        f"All action dims of raw x_t  sample={record.sample} noise={record.noise}  "
        "black=final, red=x_t; ylim fixed per dim",
        fontsize=11,
    )
    time = np.arange(horizon)
    for dim in range(n_dims):
        ylim = _ylim_from_final(
            record.final_action[:, dim],
            record.noisy_by_step[:, :, dim],
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
                record.noisy_by_step[step, :, dim],
                color="C3",
                lw=1.0,
                alpha=0.85,
                zorder=3,
                label="x_t" if dim == 0 and step == 0 else None,
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


def _plot_all_dims_overlay(records: list[NoisyRecord], output: Path) -> None:
    import matplotlib.pyplot as plt

    if len(records) < 2:
        raise ValueError("Need at least two noise records to overlay.")
    samples = {int(record.sample) for record in records}
    if len(samples) != 1:
        raise ValueError(f"Overlay requires a single sample, got {sorted(samples)}.")
    n_steps = int(records[0].noisy_by_step.shape[0])
    n_dims = int(records[0].final_action.shape[1])
    horizon = int(records[0].final_action.shape[0])
    for record in records:
        if record.noisy_by_step.shape != (n_steps, horizon, n_dims):
            raise ValueError("All overlay records must share noisy_by_step shape.")
        if record.final_action.shape != (horizon, n_dims):
            raise ValueError("All overlay records must share final_action shape.")
    fig, axes = plt.subplots(
        n_dims,
        n_steps,
        figsize=(2.05 * n_steps, 1.55 * n_dims),
        squeeze=False,
    )
    fig.suptitle(
        f"All action dims of raw x_t  sample={records[0].sample}  "
        "solid=x_t, dashed=final, color=noise seed; ylim fixed per dim",
        fontsize=11,
    )
    time = np.arange(horizon)
    for dim in range(n_dims):
        ylim = _ylim_from_final(
            *(record.final_action[:, dim] for record in records),
            *(record.noisy_by_step[:, :, dim] for record in records),
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
                    record.noisy_by_step[step, :, dim],
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


def _plot_finals_across_noises(records: list[NoisyRecord], output: Path) -> None:
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


def _compare_paths_for_sample(sample: int, output_dir: Path) -> tuple[Path, Path]:
    return (
        output_dir / f"sample{sample}_noises_xt_dims.png",
        output_dir / f"sample{sample}_noises_finals.png",
    )


def _plot_same_sample_noise_compare(
    records: list[NoisyRecord], output_dir: Path
) -> list[Path]:
    by_sample: dict[int, list[NoisyRecord]] = {}
    for record in records:
        by_sample.setdefault(int(record.sample), []).append(record)
    written: list[Path] = []
    for sample, group in sorted(by_sample.items()):
        if len(group) < 2:
            continue
        overlay, finals = _compare_paths_for_sample(sample, output_dir)
        _plot_all_dims_overlay(group, overlay)
        _plot_finals_across_noises(group, finals)
        written.extend([overlay, finals])
    return written


def _load_npz(path: Path) -> NoisyRecord:
    with np.load(path) as data:
        return NoisyRecord(
            sample=int(data["sample"]),
            noise=int(data["noise"]),
            time_t=np.asarray(data["time_t"], dtype=np.float64),
            final_action=np.asarray(data["final_action"], dtype=np.float64),
            noisy_by_step=np.asarray(data["noisy_by_step"], dtype=np.float64),
        )


def _plot_overview(records: list[NoisyRecord], output: Path) -> None:
    import matplotlib.pyplot as plt

    if not records:
        raise ValueError("Cannot plot an empty record list.")
    n_steps = records[0].noisy_by_step.shape[0]
    steps = [0, n_steps // 4, n_steps // 2, (3 * n_steps) // 4, n_steps - 1]
    steps = sorted(dict.fromkeys(step for step in steps if 0 <= step < n_steps))
    fig, axes = plt.subplots(
        len(records),
        len(steps),
        figsize=(2.4 * len(steps), 2.3 * len(records)),
        squeeze=False,
    )
    fig.suptitle("Raw noisy x_t at selected denoise steps (black = final)", fontsize=11)
    for row, record in enumerate(records):
        xy_all = np.concatenate(
            [record.noisy_by_step[:, :, :2].reshape(-1, 2), record.final_action[:, :2]],
            axis=0,
        )
        xlim = _span(xy_all[:, 0])
        ylim = _span(xy_all[:, 1])
        for col, step in enumerate(steps):
            ax = axes[row, col]
            noisy = record.noisy_by_step[step]
            ax.plot(
                record.final_action[:, 0],
                record.final_action[:, 1],
                color="black",
                lw=2.0,
                zorder=5,
            )
            ax.plot(noisy[:, 0], noisy[:, 1], color="C3", lw=1.0, alpha=0.85, zorder=3)
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


def _write_npz(record: NoisyRecord, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        sample=record.sample,
        noise=record.noise,
        time_t=record.time_t,
        final_action=record.final_action,
        noisy_by_step=record.noisy_by_step,
    )
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
        default=Path("tools/img/dit_denoise_noisy_xt"),
    )
    return parser


def _write_record_plots(
    record: NoisyRecord,
    output_dir: Path,
    *,
    plot_record: NoisyRecord | None = None,
) -> list[Path]:
    plotted = record if plot_record is None else plot_record
    stem = f"sample{record.sample}_noise{record.noise}"
    grid = output_dir / f"{stem}_noisy_xt.png"
    dims = output_dir / f"{stem}_noisy_xt_dims.png"
    npz = output_dir / f"{stem}.npz"
    _plot_one(plotted, grid)
    _plot_all_dims(plotted, dims)
    if not npz.is_file():
        _write_npz(record, npz)
    return [grid, dims, npz]


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
    records: list[NoisyRecord] = []
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
    written = [output_dir / "noisy_xt_overview.png"]
    _plot_overview(plotted, written[0])
    for record, plot_record in zip(records, plotted, strict=True):
        written.extend(
            _write_record_plots(record, output_dir, plot_record=plot_record)
        )
    written.extend(_plot_same_sample_noise_compare(plotted, output_dir))
    for path in written:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Failed to write {path}.")
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
