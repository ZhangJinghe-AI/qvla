#!/usr/bin/env python
r"""Observe whether clean-action predictions emerge from coarse to fine.

This experiment does not perturb, skip, or restart denoising. For each normal
pi0.5 generation it records the clean action predicted at every Euler step:

    a_hat_s = x_s - t_s * v_theta(x_s, t_s)

The arm trajectory is split along chunk time with an orthonormal DCT. The first
K modes are the coarse path and the remaining modes are local detail. Their
errors to the final generated action are normalized independently by step 0,
so the comparison is about convergence speed rather than raw scale.

Example:
    CUDA_VISIBLE_DEVICES=5 HF_ENDPOINT=https://hf-mirror.com uv run python \
      tools/analyze_dit_denoise_frequency_emergence.py \
      --checkpoint /data/share/pi05_libero_finetuned_v044 \
      --calibration-data ../calibration_data/libero_goal_30_7_demo.npz \
      --samples 0,1,2,3 --noise-seeds 0,1 \
      --output-dir tools/img/dit_denoise_frequency_emergence
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
    cutoff: int
    step: int
    time_t: float
    coarse_rmse: float
    detail_rmse: float
    gripper_rmse: float
    coarse_relative: float
    detail_relative: float
    gripper_relative: float


def _time_at_step(step: int, n_steps: int) -> float:
    if n_steps < 1 or not (0 <= int(step) < n_steps):
        raise ValueError(f"step={step} must be in [0, {n_steps}).")
    return 1.0 - float(step) / float(n_steps)


def _predicted_clean(
    x_t: torch.Tensor, v_t: torch.Tensor, time_t: float
) -> torch.Tensor:
    """Invert x=t*eps+(1-t)*a and v=eps-a."""
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
            f"Expected engine action tensor (1,horizon,width), got "
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
    if int(runner._num_steps) != n_steps:
        raise RuntimeError(
            f"Runner steps {runner._num_steps} != expected n_steps={n_steps}."
        )
    baseline = _step_actions(adapter, request)
    with _patched_record_clean(runner) as captured:
        recorded_final = _step_actions(adapter, request)
    detail._assert_actions_equal(
        recorded_final.cpu(), baseline.cpu(), name="record-clean Euler patch"
    )
    if len(captured) != 1 or len(captured[0]) != n_steps:
        raise RuntimeError(
            f"Expected {n_steps} clean predictions, captured "
            f"{[len(item) for item in captured]}."
        )
    final_action = _trim_action(baseline, action_dim)
    clean_by_step = np.stack(
        [_trim_action(action, action_dim) for action in captured[0]], axis=0
    )
    last_error = float(np.max(np.abs(clean_by_step[-1] - final_action)))
    if last_error > 2e-3:
        raise RuntimeError(
            f"Last predicted clean action should match Euler output; max error "
            f"{last_error:.4e} is too large."
        )
    return ConditionRecord(
        sample=sample,
        noise=noise,
        final_action=final_action,
        clean_by_step=clean_by_step,
    )


def _dct_matrix(horizon: int) -> np.ndarray:
    """Return the orthonormal DCT-II analysis matrix (frequency, time)."""
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
    if not 1 <= cutoff < action.shape[0]:
        raise ValueError(f"cutoff must be in [1, {action.shape[0]}), got {cutoff}.")
    coeff = dct @ action
    coarse = dct[:cutoff].T @ coeff[:cutoff]
    return coarse, action - coarse


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(values, dtype=np.float64)))))


def _safe_relative(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError(f"Expected a 1-D error curve, got {values.shape}.")
    scale = float(values[0])
    if scale <= 1e-12:
        return np.zeros_like(values)
    return values / scale


def _error_rows(
    record: ConditionRecord, cutoffs: list[int]
) -> tuple[list[ErrorRow], np.ndarray]:
    n_steps, horizon, action_dim = record.clean_by_step.shape
    if record.final_action.shape != (horizon, action_dim) or action_dim < 2:
        raise ValueError("Record has inconsistent action shapes.")
    dct = _dct_matrix(horizon)
    arm_final = record.final_action[:, :-1]
    gripper_final = record.final_action[:, -1]
    rows: list[ErrorRow] = []
    for cutoff in cutoffs:
        final_coarse, final_detail = _split_dct(arm_final, dct, cutoff)
        coarse = np.empty(n_steps)
        fine = np.empty(n_steps)
        gripper = np.empty(n_steps)
        for step in range(n_steps):
            arm = record.clean_by_step[step, :, :-1]
            step_coarse, step_detail = _split_dct(arm, dct, cutoff)
            coarse[step] = _rmse(step_coarse - final_coarse)
            fine[step] = _rmse(step_detail - final_detail)
            gripper[step] = _rmse(
                record.clean_by_step[step, :, -1] - gripper_final
            )
        coarse_rel = _safe_relative(coarse)
        fine_rel = _safe_relative(fine)
        gripper_rel = _safe_relative(gripper)
        for step in range(n_steps):
            rows.append(
                ErrorRow(
                    sample=record.sample,
                    noise=record.noise,
                    cutoff=cutoff,
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
    arm_error = record.clean_by_step[:, :, :-1] - arm_final[None, :, :]
    coeff_error = np.einsum("kh,shd->skd", dct, arm_error)
    return rows, coeff_error


def _auc(values: np.ndarray) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _settle_step(values: np.ndarray, threshold: float = 0.1) -> int:
    values = np.asarray(values, dtype=np.float64)
    for step in range(values.size):
        if np.all(values[step:] <= threshold):
            return step
    return int(values.size)


def _summary(
    rows: list[ErrorRow], *, cutoff: int
) -> tuple[str, dict[str, float]]:
    keys = sorted({(row.sample, row.noise) for row in rows if row.cutoff == cutoff})
    coarse_auc: list[float] = []
    detail_auc: list[float] = []
    coarse_settle: list[int] = []
    detail_settle: list[int] = []
    for key in keys:
        selected = sorted(
            [
                row
                for row in rows
                if row.cutoff == cutoff and (row.sample, row.noise) == key
            ],
            key=lambda row: row.step,
        )
        coarse = np.asarray([row.coarse_relative for row in selected])
        fine = np.asarray([row.detail_relative for row in selected])
        coarse_auc.append(_auc(coarse))
        detail_auc.append(_auc(fine))
        coarse_settle.append(_settle_step(coarse))
        detail_settle.append(_settle_step(fine))
    coarse_auc_array = np.asarray(coarse_auc)
    detail_auc_array = np.asarray(detail_auc)
    stats = {
        "conditions": float(len(keys)),
        "coarse_auc_mean": float(coarse_auc_array.mean()),
        "detail_auc_mean": float(detail_auc_array.mean()),
        "coarse_faster_fraction": float(
            np.mean(coarse_auc_array < detail_auc_array)
        ),
        "coarse_settle_median": float(np.median(coarse_settle)),
        "detail_settle_median": float(np.median(detail_settle)),
    }
    if (
        stats["coarse_auc_mean"] < stats["detail_auc_mean"]
        and stats["coarse_faster_fraction"] >= 0.65
        and stats["coarse_settle_median"] < stats["detail_settle_median"]
    ):
        verdict = "HOLDS"
    elif (
        stats["coarse_auc_mean"] > stats["detail_auc_mean"]
        and stats["coarse_faster_fraction"] <= 0.35
    ):
        verdict = "FAILS"
    else:
        verdict = "MIXED"
    return verdict, stats


def _curves(
    rows: list[ErrorRow], cutoff: int, attr: str
) -> tuple[np.ndarray, np.ndarray]:
    n_steps = max(row.step for row in rows) + 1
    per_step = [
        np.asarray(
            [
                float(getattr(row, attr))
                for row in rows
                if row.cutoff == cutoff and row.step == step
            ],
            dtype=np.float64,
        )
        for step in range(n_steps)
    ]
    mean = np.asarray([values.mean() for values in per_step])
    sem = np.asarray(
        [values.std(ddof=0) / np.sqrt(values.size) for values in per_step]
    )
    return mean, sem


def _plot_curves(
    rows: list[ErrorRow], cutoffs: list[int], output: Path
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, len(cutoffs), figsize=(5.1 * len(cutoffs), 4.4), squeeze=False
    )
    for ax, cutoff in zip(axes[0], cutoffs):
        for attr, label, color in (
            ("coarse_relative", "coarse / low DCT", "C0"),
            ("detail_relative", "detail / high DCT", "C1"),
            ("gripper_relative", "gripper", "C2"),
        ):
            mean, sem = _curves(rows, cutoff, attr)
            steps = np.arange(mean.size)
            ax.plot(steps, mean, marker="o", label=label, color=color)
            ax.fill_between(steps, mean - sem, mean + sem, color=color, alpha=0.18)
        ax.set_title(f"Low-frequency modes K={cutoff}")
        ax.set_xlabel("denoise step s")
        ax.set_ylabel("error to final / step-0 error")
        ax.set_xticks(steps)
        ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Convergence of per-step predicted clean action (mean ± SEM)", fontsize=12
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_frequency_heatmap(coeff_errors: list[np.ndarray], output: Path) -> None:
    import matplotlib.pyplot as plt

    squared = np.concatenate(
        [np.square(error).mean(axis=2, keepdims=True) for error in coeff_errors],
        axis=2,
    )
    rms = np.sqrt(squared.mean(axis=2))
    denominator = np.maximum(rms[0:1], 1e-12)
    relative = rms / denominator
    display = np.log10(np.clip(relative, 1e-3, 2.0)).T
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    image = ax.imshow(
        display,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=-3.0,
        vmax=np.log10(2.0),
    )
    ax.set_xlabel("denoise step s")
    ax.set_ylabel("DCT mode k (low → high)")
    ax.set_title("Arm frequency error to final action, log10(relative to step 0)")
    ax.set_xticks(np.arange(display.shape[1]))
    fig.colorbar(image, ax=ax, label="log10 relative error")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _lowpass(action: np.ndarray, cutoff: int) -> np.ndarray:
    coarse, _ = _split_dct(action, _dct_matrix(action.shape[0]), cutoff)
    return coarse


def _plot_examples(
    records: list[ConditionRecord], cutoff: int, output: Path, count: int = 6
) -> None:
    import matplotlib.pyplot as plt

    chosen = records[: min(count, len(records))]
    steps = [0, 2, 5, 7, 9]
    n_steps = chosen[0].clean_by_step.shape[0]
    steps = [step for step in steps if step < n_steps]
    fig, axes = plt.subplots(
        len(chosen), len(steps), figsize=(2.45 * len(steps), 2.35 * len(chosen)),
        squeeze=False,
    )
    for row, record in enumerate(chosen):
        final = record.final_action[:, :2]
        final_low = _lowpass(final, cutoff)
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
            predicted_low = _lowpass(predicted, cutoff)
            ax.plot(final[:, 0], final[:, 1], color="black", lw=2.0, label="final")
            ax.plot(
                predicted[:, 0], predicted[:, 1], color="C1", alpha=0.55,
                lw=0.9, label="a_hat",
            )
            ax.plot(
                final_low[:, 0], final_low[:, 1], color="black", lw=2.4,
                linestyle="--", label="final low",
            )
            ax.plot(
                predicted_low[:, 0], predicted_low[:, 1], color="C0", lw=1.8,
                label="a_hat low",
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
        f"Predicted clean action: full path and coarse DCT path (K={cutoff})",
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
    parser.add_argument("--cutoffs", default="3,5,8")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--params-dtype", default="bfloat16")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tools/img/dit_denoise_frequency_emergence"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.model != "pi05":
        raise RuntimeError("Frequency-emergence analysis is implemented for pi05.")
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
    cutoffs = detail._parse_nonneg_ints(args.cutoffs, name="--cutoffs")
    if any(cutoff < 1 for cutoff in cutoffs):
        raise ValueError("--cutoffs must contain positive integers.")

    adapter = matched.adapter_from_args(args)
    model = adapter.build_model()
    model.eval()
    adapter.warmup_for_calibration(model)
    runtime = matched.clip_runtime(adapter, QVLAConfig.for_model_kind(args.model))
    if any(cutoff >= runtime.action_horizon for cutoff in cutoffs):
        raise ValueError(
            f"All cutoffs must be below horizon={runtime.action_horizon}."
        )
    need = max(sample_ids) + 1
    batches = list(adapter.iter_calibration_batches(need))
    if len(batches) != need:
        raise RuntimeError(f"Calibration yielded {len(batches)}, need {need}.")

    conditions = [(sample, noise) for sample in sample_ids for noise in noise_ids]
    print(
        f"model=pi05 conditions={len(conditions)} samples={sample_ids} "
        f"noise_seeds={noise_ids} steps={runtime.num_steps} "
        f"horizon={runtime.action_horizon} action_dim={runtime.action_dim} "
        f"cutoffs={cutoffs}"
    )
    records: list[ConditionRecord] = []
    all_rows: list[ErrorRow] = []
    coeff_errors: list[np.ndarray] = []
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
        rows, coeff_error = _error_rows(record, cutoffs)
        records.append(record)
        all_rows.extend(rows)
        coeff_errors.append(coeff_error)
        print(f"  [{index}/{len(conditions)}] sample={sample} noise={noise}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "frequency_emergence.csv"
    curves_path = output_dir / "frequency_convergence.png"
    heatmap_path = output_dir / "dct_convergence_heatmap.png"
    examples_path = output_dir / "clean_action_examples.png"
    summary_path = output_dir / "summary.txt"
    _write_rows(all_rows, csv_path)
    _plot_curves(all_rows, cutoffs, curves_path)
    _plot_frequency_heatmap(coeff_errors, heatmap_path)
    _plot_examples(records, 5 if 5 in cutoffs else cutoffs[0], examples_path)

    lines: list[str] = []
    for cutoff in cutoffs:
        verdict, stats = _summary(all_rows, cutoff=cutoff)
        line = (
            f"K={cutoff}: {verdict}; coarse_auc={stats['coarse_auc_mean']:.4f}, "
            f"detail_auc={stats['detail_auc_mean']:.4f}, "
            f"coarse_faster={stats['coarse_faster_fraction']:.3f}, "
            f"settle10 coarse/detail={stats['coarse_settle_median']:.1f}/"
            f"{stats['detail_settle_median']:.1f}"
        )
        print(line)
        lines.append(line)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for path in (csv_path, curves_path, heatmap_path, examples_path, summary_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Failed to write {path}.")
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
