"""Tests for clean-action frequency-emergence analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import analyze_dit_denoise_frequency_emergence as analysis  # noqa: E402


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


def test_record_loop_preserves_final_and_last_prediction():
    runner = _FakeRunner()
    noise = torch.zeros(1, 5, 3)
    expected = runner._fwd_loop(noise=noise)
    final, clean = analysis._euler_record_clean(runner, noise)
    torch.testing.assert_close(final, expected)
    torch.testing.assert_close(clean[-1], final)
    assert len(clean) == runner._num_steps


def test_patch_restores_original_loop():
    runner = _FakeRunner()
    original_method = runner._fwd_loop
    noise = torch.zeros(1, 5, 3)
    expected = original_method(noise=noise)
    with analysis._patched_record_clean(runner) as captured:
        actual = runner._fwd_loop(noise=noise)
    torch.testing.assert_close(actual, expected)
    assert len(captured[0]) == runner._num_steps
    torch.testing.assert_close(runner._fwd_loop(noise=noise), expected)


def test_dct_is_orthonormal_and_split_reconstructs():
    rng = np.random.default_rng(0)
    action = rng.normal(size=(50, 6))
    dct = analysis._dct_matrix(50)
    np.testing.assert_allclose(dct @ dct.T, np.eye(50), atol=1e-12)
    coarse, detail = analysis._split_dct(action, dct, cutoff=5)
    np.testing.assert_allclose(coarse + detail, action, atol=1e-12)


def test_low_modes_capture_slow_signal():
    horizon = 50
    time = np.arange(horizon)
    dct = analysis._dct_matrix(horizon)
    slow = np.cos(np.pi / horizon * (time + 0.5))[:, None]
    fast = np.cos(20 * np.pi / horizon * (time + 0.5))[:, None]
    slow_coarse, slow_detail = analysis._split_dct(slow, dct, cutoff=5)
    fast_coarse, fast_detail = analysis._split_dct(fast, dct, cutoff=5)
    assert analysis._rmse(slow_detail) < 1e-12
    assert analysis._rmse(slow_coarse) > 0.5
    assert analysis._rmse(fast_coarse) < 1e-12
    assert analysis._rmse(fast_detail) > 0.5


def _record(coarse_first: bool) -> analysis.ConditionRecord:
    steps, horizon, dims = 10, 50, 7
    dct = analysis._dct_matrix(horizon)
    final = np.zeros((horizon, dims), dtype=np.float64)
    clean = np.zeros((steps, horizon, dims), dtype=np.float64)
    low = dct[1]
    high = dct[20]
    for step in range(steps):
        low_error = max(0.0, 1.0 - step / (3.0 if coarse_first else 9.0))
        high_error = max(0.0, 1.0 - step / (9.0 if coarse_first else 3.0))
        clean[step, :, :6] = (
            low_error * low[:, None] + high_error * high[:, None]
        )
        clean[step, :, -1] = high_error * high
    clean[-1] = final
    return analysis.ConditionRecord(
        sample=0, noise=0, final_action=final, clean_by_step=clean
    )


def test_error_rows_detect_coarse_first_convergence():
    rows, coeff = analysis._error_rows(_record(coarse_first=True), [5])
    assert coeff.shape == (10, 50, 6)
    verdict, stats = analysis._summary(rows, cutoff=5)
    assert verdict == "HOLDS"
    assert stats["coarse_auc_mean"] < stats["detail_auc_mean"]
    assert stats["coarse_settle_median"] < stats["detail_settle_median"]


def test_error_rows_reject_detail_first_convergence():
    rows, _ = analysis._error_rows(_record(coarse_first=False), [5])
    verdict, stats = analysis._summary(rows, cutoff=5)
    assert verdict == "FAILS"
    assert stats["coarse_auc_mean"] > stats["detail_auc_mean"]


def test_settle_requires_remaining_steps_below_threshold():
    assert analysis._settle_step(np.array([1.0, 0.05, 0.2, 0.0])) == 3
    assert analysis._settle_step(np.array([1.0, 0.2, 0.05, 0.0])) == 2


def test_invalid_cutoff_rejected():
    with pytest.raises(ValueError, match="cutoff"):
        analysis._split_dct(np.zeros((5, 2)), analysis._dct_matrix(5), 5)
