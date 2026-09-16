#!/usr/bin/env python
r"""Observe coarse-to-fine clean-action emergence using moving averages only.

For each normal pi0.5 generation, record the clean action predicted at every
Euler step:

    a_hat_s = x_s - t_s * v_theta(x_s, t_s)

For each arm dimension, LP is a centered moving average along chunk time and
HP(a) = a - LP(a). The experiment compares LP(a_hat_s) with LP(a_final), and
HP(a_hat_s) with HP(a_final). Each error curve is divided by its own step-0
error before coarse and detail convergence speeds are compared. Gripper is
reported separately.
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
from qvla.adapters.pi05.step_hook import find_expert_runner  # noqa: E402
from qvla.config import QVLAConfig  # noqa: E402


@dataclass(frozen=True)
class ConditionRecord:
    sample: int
    noise: int
    final_action: np.ndarray
    clean_by_step: np.ndarray


@dataclass(frozen=True)
class ErrorRow:
    sample: int
    noise: int
    window: int
    step: int
    time_t: float
    coarse_rmse: float
    detail_rmse: float
    gripper_rmse: float
    coarse_relative: float
    detail_relative: float
    gripper_relative: float


def _time_at_step(step: int, n_steps: int) -> float:
    if n_steps < 1 or not 0 <= int(step) < n_steps:
        raise ValueError(f"step={step} must be in [0, {n_steps}).")
    return 1.0 - float(step) / float(n_steps)


def _predicted_clean(
    x_t: torch.Tensor, v_t: torch.Tensor, time_t: float
) -> torch.Tensor:
    if not 0.0 <= float(time_t) <= 1.0:
        raise ValueError(f"time_t must be in [0, 1], got {time_t}.")
    return x_t - float(time_t) * v_t.to(dtype=x_t.dtype)


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
            "Recording clean actions needs the eager loop; runner has a CUDA graph."
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
) -> ConditionRecord:
    runner = find_expert_runner(adapter.engine)
    baseline = _step_actions(adapter, request)
    with _patched_record_clean(runner) as captured:
        recorded_final = _step_actions(adapter, request)
    detail._assert_actions_equal(
        recorded_final.cpu(), baseline.cpu(), name="record-clean Euler patch"
    )
    if len(captured) != 1 or len(captured[0]) != n_steps:
        raise RuntimeError(f"Expected {n_steps} recorded clean actions.")
    final_action = _trim_action(baseline, action_dim)
    clean_by_step = np.stack(
        [_trim_action(action, action_dim) for action in captured[0]], axis=0
    )
    if np.max(np.abs(clean_by_step[-1] - final_action)) > 2e-3:
        raise RuntimeError("Last clean prediction does not match final Euler action.")
    return ConditionRecord(sample, noise, final_action, clean_by_step)


def _moving_average(action: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with reflected boundaries."""
    action = np.asarray(action, dtype=np.float64)
    if action.ndim != 2:
        raise ValueError(f"Expected action (horizon,dims), got {action.shape}.")
    if window < 1 or window % 2 == 0 or window > action.shape[0]:
        raise ValueError(
            f"window must be odd and in [1, {action.shape[0]}], got {window}."
        )
    if window == 1:
        return action.copy()
    pad = window // 2
    padded = np.pad(action, ((pad, pad), (0, 0)), mode="reflect")
    cumulative = np.concatenate(
        [np.zeros((1, action.shape[1])), np.cumsum(padded, axis=0)], axis=0
    )
    return (cumulative[window:] - cumulative[:-window]) / float(window)


def _split_smooth(action: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    coarse = _moving_average(action, window)
    return coarse, np.asarray(action, dtype=np.float64) - coarse


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(values, dtype=np.float64)))))


def _relative(errors: np.ndarray) -> np.ndarray:
    errors = np.asarray(errors, dtype=np.float64)
    if errors.ndim != 1 or errors.size < 2:
        raise ValueError(f"Expected 1-D errors, got {errors.shape}.")
    if errors[0] <= 1e-12:
        return np.zeros_like(errors)
    return errors / errors[0]


def _error_rows(record: ConditionRecord, windows: list[int]) -> list[ErrorRow]:
    n_steps, horizon, action_dim = record.clean_by_step.shape
    if record.final_action.shape != (horizon, action_dim) or action_dim < 2:
        raise ValueError("Record has inconsistent action shapes.")
    final_arm = record.final_action[:, :-1]
    final_gripper = record.final_action[:, -1]
    rows: list[ErrorRow] = []
    for window in windows:
        final_coarse, final_detail = _split_smooth(final_arm, window)
        coarse = np.empty(n_steps)
        fine = np.empty(n_steps)
        gripper = np.empty(n_steps)
        for step in range(n_steps):
            arm = record.clean_by_step[step, :, :-1]
            step_coarse, step_detail = _split_smooth(arm, window)
            coarse[step] = _rmse(step_coarse - final_coarse)
            fine[step] = _rmse(step_detail - final_detail)
            gripper[step] = _rmse(
                record.clean_by_step[step, :, -1] - final_gripper
            )
        coarse_rel = _relative(coarse)
        fine_rel = _relative(fine)
        gripper_rel = _relative(gripper)
        for step in range(n_steps):
            rows.append(
                ErrorRow(
                    sample=record.sample,
                    noise=record.noise,
                    window=window,
                    step=step,
                    time_t=_time_at_step(step, n_steps),
                    coarse_rmse=float(coarse[step]),
                    detail_rmse=float(fine[step]),
                    gripper_rmse=float(gripper[step]),
                    coarse_relative=float(coarse_rel[step]),
                    detail_relative=float(fine_rel[step]),
                    gripper_relative=float(gripper_rel[step]),
                )
            )
    return rows


def _settle_step(values: np.ndarray, threshold: float = 0.1) -> int:
    values = np.asarray(values, dtype=np.float64)
    for step in range(values.size):
        if np.all(values[step:] <= threshold):
            return step
    return int(values.size)


def _summary(
    rows: list[ErrorRow], window: int
) -> tuple[str, dict[str, float]]:
    keys = sorted({(row.sample, row.noise) for row in rows if row.window == window})
    coarse_auc: list[float] = []
    detail_auc: list[float] = []
    coarse_settle: list[int] = []
    detail_settle: list[int] = []
    for key in keys:
        selected = sorted(
            [
                row
                for row in rows
                if row.window == window and (row.sample, row.noise) == key
            ],
            key=lambda row: row.step,
        )
        coarse = np.asarray([row.coarse_relative for row in selected])
        fine = np.asarray([row.detail_relative for row in selected])
        coarse_auc.append(float(coarse.mean()))
        detail_auc.append(float(fine.mean()))
        coarse_settle.append(_settle_step(coarse))
        detail_settle.append(_settle_step(fine))
    coarse_auc_array = np.asarray(coarse_auc)
    detail_auc_array = np.asarray(detail_auc)
    stats = {
        "coarse_auc": float(coarse_auc_array.mean()),
        "detail_auc": float(detail_auc_array.mean()),
        "coarse_faster": float(np.mean(coarse_auc_array < detail_auc_array)),
        "coarse_settle": float(np.median(coarse_settle)),
        "detail_settle": float(np.median(detail_settle)),
    }
    if (
        stats["coarse_auc"] < stats["detail_auc"]
        and stats["coarse_faster"] >= 0.65
        and stats["coarse_settle"] < stats["detail_settle"]
    ):
        verdict = "HOLDS"
    elif (
        stats["coarse_auc"] > stats["detail_auc"]
        and stats["coarse_faster"] <= 0.35
    ):
        verdict = "FAILS"
    else:
        verdict = "MIXED"
    return verdict, stats


def _curves(
    rows: list[ErrorRow], window: int, attr: str
) -> tuple[np.ndarray, np.ndarray]:
    n_steps = max(row.step for row in rows) + 1
    values = [
        np.asarray(
            [
                float(getattr(row, attr))
                for row in rows
                if row.window == window and row.step == step
            ]
        )
        for step in range(n_steps)
    ]
    mean = np.asarray([item.mean() for item in values])
    sem = np.asarray(
        [item.std(ddof=0) / np.sqrt(item.size) for item in values]
    )
    return mean, sem


def _plot_curves(rows: list[ErrorRow], windows: list[int], output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, len(windows), figsize=(5.1 * len(windows), 4.4), squeeze=False
    )
    for ax, window in zip(axes[0], windows):
        for attr, label, color in (
            ("coarse_relative", "coarse: smooth(a)", "C0"),
            ("detail_relative", "detail: a - smooth(a)", "C1"),
            ("gripper_relative", "gripper", "C2"),
        ):
            mean, sem = _curves(rows, window, attr)
            steps = np.arange(mean.size)
            ax.plot(steps, mean, marker="o", color=color, label=label)
            ax.fill_between(steps, mean - sem, mean + sem, color=color, alpha=0.18)
        ax.set_title(f"Moving-average window={window}")
        ax.set_xlabel("denoise step s")
        ax.set_ylabel("error to final / step-0 error")
        ax.set_xticks(steps)
        ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(
        "LP/HP convergence of predicted clean actions (mean ± SEM)", fontsize=12
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_examples(
    records: list[ConditionRecord], window: int, output: Path, count: int = 6
) -> None:
    import matplotlib.pyplot as plt

    records = records[: min(count, len(records))]
    candidate_steps = (0, 2, 5, 7, 9)
    steps = [
        step for step in candidate_steps if step < records[0].clean_by_step.shape[0]
    ]
    fig, axes = plt.subplots(
        len(records),
        len(steps),
        figsize=(2.45 * len(steps), 2.35 * len(records)),
        squeeze=False,
    )
    for row, record in enumerate(records):
        final = record.final_action[:, :2]
        final_low = _moving_average(final, window)
        all_xy = np.concatenate(
            [record.clean_by_step[:, :, :2].reshape(-1, 2), final], axis=0
        )
        xpad = max(float(np.ptp(all_xy[:, 0])) * 0.08, 1e-3)
        ypad = max(float(np.ptp(all_xy[:, 1])) * 0.08, 1e-3)
        xlim = (float(all_xy[:, 0].min() - xpad), float(all_xy[:, 0].max() + xpad))
        ylim = (float(all_xy[:, 1].min() - ypad), float(all_xy[:, 1].max() + ypad))
        for col, step in enumerate(steps):
            ax = axes[row, col]
            predicted = record.clean_by_step[step, :, :2]
            predicted_low = _moving_average(predicted, window)
            ax.plot(final[:, 0], final[:, 1], color="black", lw=1.8, label="final")
            ax.plot(
                predicted[:, 0], predicted[:, 1], color="C1", alpha=0.55,
                lw=0.9, label="a_hat",
            )
            ax.plot(
                final_low[:, 0], final_low[:, 1], color="black", lw=2.3,
                linestyle="--", label="LP(final)",
            )
            ax.plot(
                predicted_low[:, 0], predicted_low[:, 1], color="C0", lw=1.8,
                label="LP(a_hat)",
            )
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.2)
            if row == 0:
                ax.set_title(f"s={step}")
            if col == 0:
                ax.set_ylabel(f"sample {record.sample}\nnoise {record.noise}")
            if row == 0 and col == 0:
                ax.legend(fontsize=6)
    fig.suptitle(
        f"Predicted clean action and moving-average LP (window={window})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _write_rows(rows: list[ErrorRow], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(ErrorRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-data", type=Path, required=True)
    matched.add_model_cli(parser)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--samples", default=None)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--noise-seeds", default=None)
    parser.add_argument("--windows", default="5,9,15")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--params-dtype", default="bfloat16")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tools/img/dit_denoise_smooth_emergence"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.model != "pi05":
        raise RuntimeError("Smooth-emergence analysis is implemented for pi05.")
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
    windows = detail._parse_nonneg_ints(args.windows, name="--windows")
    if any(window < 1 or window % 2 == 0 for window in windows):
        raise ValueError("--windows must contain positive odd integers.")

    adapter = matched.adapter_from_args(args)
    model = adapter.build_model()
    model.eval()
    adapter.warmup_for_calibration(model)
    runtime = matched.clip_runtime(adapter, QVLAConfig.for_model_kind(args.model))
    if any(window > runtime.action_horizon for window in windows):
        raise ValueError(f"Windows exceed horizon={runtime.action_horizon}.")
    need = max(sample_ids) + 1
    batches = list(adapter.iter_calibration_batches(need))
    if len(batches) != need:
        raise RuntimeError(f"Calibration yielded {len(batches)}, need {need}.")

    conditions = [(sample, noise) for sample in sample_ids for noise in noise_ids]
    print(
        f"model=pi05 conditions={len(conditions)} steps={runtime.num_steps} "
        f"horizon={runtime.action_horizon} windows={windows}"
    )
    records: list[ConditionRecord] = []
    rows: list[ErrorRow] = []
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
        rows.extend(_error_rows(record, windows))
        print(f"  [{index}/{len(conditions)}] sample={sample} noise={noise}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "smooth_emergence.csv"
    plot_path = output_dir / "smooth_convergence.png"
    examples_path = output_dir / "smooth_action_examples.png"
    summary_path = output_dir / "summary.txt"
    _write_rows(rows, csv_path)
    _plot_curves(rows, windows, plot_path)
    _plot_examples(records, 9 if 9 in windows else windows[0], examples_path)
    lines: list[str] = []
    for window in windows:
        verdict, stats = _summary(rows, window)
        line = (
            f"window={window}: {verdict}; "
            f"coarse_auc={stats['coarse_auc']:.4f}, "
            f"detail_auc={stats['detail_auc']:.4f}, "
            f"coarse_faster={stats['coarse_faster']:.3f}, "
            f"settle10 coarse/detail={stats['coarse_settle']:.1f}/"
            f"{stats['detail_settle']:.1f}"
        )
        print(line)
        lines.append(line)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for path in (csv_path, plot_path, examples_path, summary_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Failed to write {path}.")
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
