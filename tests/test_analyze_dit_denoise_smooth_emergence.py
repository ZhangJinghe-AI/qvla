"""Tests for moving-average LP/HP clean-action analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import analyze_dit_denoise_smooth_emergence as analysis  # noqa: E402


class _FakeRunner:
    def __init__(self, num_steps: int = 4):
        self._num_steps = num_steps
        self._dt = -1.0 / num_steps
        self.graph = None

    def _one_step(self, x_t: torch.Tensor, step: int) -> torch.Tensor:
        return torch.ones_like(x_t) * float(step + 1)

    def _fwd_loop(self, *, noise: torch.Tensor) -> torch.Tensor:
        x_t = noise
        for step in range(self._num_steps):
            x_t = x_t + self._dt * self._one_step(x_t, step)
        return x_t


def test_predicted_clean_inverts_linear_flow():
    action = torch.full((1, 5, 3), 2.0)
    eps = torch.full((1, 5, 3), 5.0)
    t = 0.4
    x_t = t * eps + (1.0 - t) * action
    velocity = eps - action
    torch.testing.assert_close(
        analysis._predicted_clean(x_t, velocity, t), action
    )


def test_record_loop_preserves_final_and_last_clean():
    runner = _FakeRunner()
    noise = torch.zeros(1, 5, 3)
    expected = runner._fwd_loop(noise=noise)
    final, clean = analysis._euler_record_clean(runner, noise)
    torch.testing.assert_close(final, expected)
    torch.testing.assert_close(clean[-1], final)


def test_moving_average_preserves_constant():
    action = np.full((20, 3), 2.5)
    np.testing.assert_allclose(analysis._moving_average(action, 5), action)


def test_lp_plus_hp_recovers_action_exactly():
    action = np.random.default_rng(0).normal(size=(50, 6))
    low, high = analysis._split_smooth(action, 9)
    np.testing.assert_allclose(low + high, action, atol=1e-14)


def test_moving_average_removes_fast_alternation():
    action = np.tile(np.array([-1.0, 1.0]), 25)[:, None]
    low = analysis._moving_average(action, 9)
    assert analysis._rmse(low) < analysis._rmse(action) * 0.25


def test_even_window_is_rejected():
    with pytest.raises(ValueError, match="odd"):
        analysis._moving_average(np.zeros((50, 2)), 8)


def _record(coarse_first: bool) -> analysis.ConditionRecord:
    steps, horizon, dims = 10, 50, 7
    time = np.linspace(0.0, 1.0, horizon)
    slow = time[:, None]
    fast = np.tile(np.array([-1.0, 1.0]), 25)[:, None]
    final = np.zeros((horizon, dims))
    clean = np.zeros((steps, horizon, dims))
    for step in range(steps):
        slow_error = max(0.0, 1.0 - step / (3.0 if coarse_first else 9.0))
        fast_error = max(0.0, 1.0 - step / (9.0 if coarse_first else 3.0))
        clean[step, :, :6] = slow_error * slow + fast_error * fast
        clean[step, :, -1] = fast_error * fast[:, 0]
    clean[-1] = final
    return analysis.ConditionRecord(0, 0, final, clean)


def test_summary_detects_coarse_first():
    rows = analysis._error_rows(_record(coarse_first=True), [9])
    verdict, stats = analysis._summary(rows, 9)
    assert verdict == "HOLDS"
    assert stats["coarse_auc"] < stats["detail_auc"]


def test_summary_rejects_detail_first():
    rows = analysis._error_rows(_record(coarse_first=False), [9])
    verdict, stats = analysis._summary(rows, 9)
    assert verdict == "FAILS"
    assert stats["coarse_auc"] > stats["detail_auc"]


def test_settle_step_must_remain_below_threshold():
    assert analysis._settle_step(np.array([1.0, 0.05, 0.2, 0.0])) == 3
