"""Tests for SDEdit resume-from-step denoise analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import analyze_dit_denoise_sdedit as analysis  # noqa: E402
import analyze_dit_outlier_action_detail as detail  # noqa: E402


class _FakeRunner:
    def __init__(self, num_steps: int = 3, dt: float = 0.5):
        self._num_steps = num_steps
        self._dt = dt
        self.graph = None
        self.calls: list[int] = []

    def _one_step(self, x_t: torch.Tensor, step: int) -> torch.Tensor:
        self.calls.append(int(step))
        return torch.ones_like(x_t) * 2.0 * float(step + 1)

    def _fwd_loop(self, *, noise: torch.Tensor) -> torch.Tensor:
        x_t = noise
        for step in range(self._num_steps):
            v_t = self._one_step(x_t, step)
            x_t = x_t + self._dt * v_t
        return x_t


def test_time_at_step_matches_pi05_schedule():
    assert analysis._time_at_step(0, 10) == pytest.approx(1.0)
    assert analysis._time_at_step(9, 10) == pytest.approx(0.1)
    assert analysis._time_at_step(10, 10) == pytest.approx(0.0)


def test_mix_recovers_noise_at_t1_and_action_at_t0():
    action = torch.ones(1, 2, 2)
    eps = torch.full((1, 2, 2), 3.0)
    torch.testing.assert_close(analysis._mix_action(action, eps, 1.0), eps)
    torch.testing.assert_close(analysis._mix_action(action, eps, 0.0), action)
    mid = analysis._mix_action(action, eps, 0.25)
    torch.testing.assert_close(mid, 0.25 * eps + 0.75 * action)


def test_resume_from_step_skips_prefix_updates():
    runner = _FakeRunner()
    noise = torch.zeros(1, 2, 2)
    full = analysis._euler_from_step(runner, noise, start_step=0)
    # deltas 1+2+3
    torch.testing.assert_close(full, torch.full((1, 2, 2), 6.0))
    runner.calls.clear()
    resumed = analysis._euler_from_step(runner, torch.full((1, 2, 2), 10.0), start_step=1)
    assert runner.calls == [1, 2]
    torch.testing.assert_close(resumed, torch.full((1, 2, 2), 15.0))


def test_patched_resume_from_zero_matches_original():
    runner = _FakeRunner()
    noise = torch.zeros(1, 2, 2)
    original = runner._fwd_loop(noise=noise)
    with analysis._patched_euler_from_step(runner, start_step=0):
        patched = runner._fwd_loop(noise=noise)
    torch.testing.assert_close(patched, original)


def test_early_late_excludes_step_zero():
    early, late = analysis._early_late_steps(10)
    assert early == [1, 2, 3, 4]
    assert late == [5, 6, 7, 8, 9]


def _row(step: int, mean_shift: float, local: float) -> analysis.EditRow:
    metrics = detail.ActionMetrics(
        total_rmse=0.0,
        arm_mean_shift_rmse=mean_shift,
        arm_endpoint_rmse=0.0,
        arm_net_disp_rmse=0.0,
        arm_local_rmse=local,
        arm_step_rmse=0.0,
        gripper_rmse=0.0,
        gripper_switch_shift=0.0,
    )
    return analysis.EditRow(
        sample=0,
        noise=0,
        trial=0,
        step=step,
        n_steps=10,
        time_t=1.0 - step / 10.0,
        metrics=metrics,
    )


def test_verdict_holds_when_early_is_coarse_and_late_is_local():
    rows = [_row(0, 9.0, 9.0)]
    for step in range(1, 5):
        rows.append(_row(step, 4.0, 1.0))
    for step in range(5, 10):
        rows.append(_row(step, 0.5, 2.0))
    text = analysis._verdict(rows)
    assert text.startswith("HOLDS")


def test_verdict_ignores_full_redraw_at_step_zero():
    rows = [_row(0, 100.0, 0.1)]
    for step in range(1, 5):
        rows.append(_row(step, 4.0, 1.0))
    for step in range(5, 10):
        rows.append(_row(step, 0.5, 2.0))
    text = analysis._verdict(rows)
    assert text.startswith("HOLDS")


def test_predicted_clean_recovers_action():
    t = 0.4
    action = torch.full((1, 4, 3), 2.0)
    eps = torch.full((1, 4, 3), 5.0)
    x_t = t * eps + (1.0 - t) * action
    v_t = eps - action
    torch.testing.assert_close(analysis._predicted_clean(x_t, v_t, t), action)


def test_last_x0_matches_final_state_on_pi05_schedule():
    runner = _FakeRunner(num_steps=4, dt=-0.25)
    noise = torch.zeros(1, 3, 3)
    final, predicted = analysis._euler_record_clean(runner, noise)
    assert len(predicted) == 4
    torch.testing.assert_close(predicted[-1], final)


def test_record_clean_patch_matches_original_loop():
    runner = _FakeRunner(num_steps=4, dt=-0.25)
    noise = torch.zeros(1, 3, 3)
    original = runner._fwd_loop(noise=noise)
    with analysis._patched_euler_record_clean(runner) as recorded:
        patched = runner._fwd_loop(noise=noise)
    torch.testing.assert_close(patched, original)
    assert len(recorded) == 1
    assert len(recorded[0]) == 4


def test_overview_steps_picks_early_mid_late():
    assert analysis._overview_steps(10) == [0, 2, 5, 7, 9]
    assert analysis._overview_steps(3) == [0, 1, 2]


def test_chunks_paginates_records():
    records = [_toy_record(), _toy_record(), _toy_record()]
    pages = analysis._chunks(records, 2)
    assert [len(page) for page in pages] == [2, 1]


def test_layout_rank_puts_large_early_shift_first():
    rows = []
    for step in range(10):
        rows.append(_row(step, 0.02, 0.05))
        rows[-1] = analysis.EditRow(
            sample=1,
            noise=0,
            trial=0,
            step=step,
            n_steps=10,
            time_t=1.0 - step / 10.0,
            metrics=rows[-1].metrics,
        )
    big = []
    for step in range(10):
        mean_shift = 0.2 if step <= 2 else 0.04
        big.append(
            analysis.EditRow(
                sample=7,
                noise=0,
                trial=0,
                step=step,
                n_steps=10,
                time_t=1.0 - step / 10.0,
                metrics=detail.ActionMetrics(
                    total_rmse=0.0,
                    arm_mean_shift_rmse=mean_shift,
                    arm_endpoint_rmse=mean_shift * 2,
                    arm_net_disp_rmse=0.0,
                    arm_local_rmse=0.05,
                    arm_step_rmse=0.0,
                    gripper_rmse=0.0,
                    gripper_switch_shift=0.0,
                ),
            )
        )
    table = analysis._layout_rank_table(rows + big)
    assert table[0]["sample"] == 7
    assert table[0]["early_mean_shift"] > table[1]["early_mean_shift"]


def _toy_record() -> analysis.TrajRecord:
    horizon = 6
    dim = 3
    n_steps = 2
    n_trials = 2
    t = np.linspace(0.0, 1.0, horizon, dtype=np.float32)
    baseline = np.stack([t, 0.2 * t, np.linspace(-1.0, 1.0, horizon, dtype=np.float32)], axis=1)
    edited = np.zeros((n_steps, n_trials, horizon, dim), dtype=np.float32)
    x0 = np.zeros((n_steps, horizon, dim), dtype=np.float32)
    for step in range(n_steps):
        x0[step] = baseline + 0.05 * (step + 1)
        for trial in range(n_trials):
            edited[step, trial] = baseline + 0.1 * (trial + 1) * (1.0 - 0.5 * step)
    return analysis.TrajRecord(
        sample=0,
        noise=0,
        n_steps=n_steps,
        time_t=np.array([1.0, 0.5], dtype=np.float64),
        baseline=baseline,
        x0_pred=x0,
        edited=edited,
    )


def test_trajectory_plots_write_pngs(tmp_path: Path):
    record = _toy_record()
    written = analysis._plot_trajectories([record], tmp_path)
    assert written
    for path in written:
        assert path.is_file() and path.stat().st_size > 0
