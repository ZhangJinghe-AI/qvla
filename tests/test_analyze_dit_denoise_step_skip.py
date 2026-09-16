"""Tests for the single Euler-step skip experiment."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import analyze_dit_denoise_step_skip as analysis  # noqa: E402
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


def test_skip_drops_only_that_euler_update():
    runner = _FakeRunner()
    noise = torch.zeros(1, 2, 2)
    full, full_rms = analysis._euler_with_optional_skip(runner, noise, skip_step=None)
    skipped, skip_rms = analysis._euler_with_optional_skip(runner, noise, skip_step=1)
    # dt=0.5, v=2*(step+1) → delta = step+1. Full: 0+1+2+3=6.
    torch.testing.assert_close(full, torch.full((1, 2, 2), 6.0))
    # Skip step 1 (delta 2): 0+1+3=4.
    torch.testing.assert_close(skipped, torch.full((1, 2, 2), 4.0))
    assert full_rms[1] == pytest.approx(skip_rms[1])
    assert full_rms[1] > skip_rms[0]


def test_patched_no_skip_matches_original_loop():
    runner = _FakeRunner()
    noise = torch.zeros(1, 2, 2)
    original = runner._fwd_loop(noise=noise)
    with analysis._patched_euler_skip(runner, skip_step=None) as recorded:
        patched = runner._fwd_loop(noise=noise)
    torch.testing.assert_close(patched, original)
    assert len(recorded) == 3


def test_patched_skip_rejects_cuda_graph():
    runner = _FakeRunner()
    runner.graph = object()
    with pytest.raises(RuntimeError, match="CUDA graph"):
        with analysis._patched_euler_skip(runner, skip_step=0):
            pass


def test_early_late_split_for_ten_steps():
    early, late = analysis._early_late_steps(10)
    assert early == list(range(5))
    assert late == list(range(5, 10))


def _row(step: int, mean_shift: float, local: float) -> analysis.SkipRow:
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
    return analysis.SkipRow(
        sample=0,
        noise=0,
        step=step,
        n_steps=4,
        update_rms=1.0,
        remove_rms=1.0,
        metrics=metrics,
    )


def test_equal_shrink_removes_matched_rms():
    runner = _FakeRunner()
    noise = torch.zeros(1, 2, 2)
    full, full_rms = analysis._euler_with_optional_skip(runner, noise, skip_step=None)
    remove = min(full_rms)
    assert remove == pytest.approx(1.0)
    # Step 1 delta RMS=2 → keep=0.5, path 0+1+1+3=5.
    shrunk, _ = analysis._euler_with_optional_skip(
        runner, noise, skip_step=1, remove_rms=remove
    )
    torch.testing.assert_close(full, torch.full((1, 2, 2), 6.0))
    torch.testing.assert_close(shrunk, torch.full((1, 2, 2), 5.0))
    # Smallest step is fully dropped: 0+0+2+3=5.
    dropped, _ = analysis._euler_with_optional_skip(
        runner, noise, skip_step=0, remove_rms=remove
    )
    torch.testing.assert_close(dropped, torch.full((1, 2, 2), 5.0))
    assert analysis._keep_scale(2.0, 1.0) == pytest.approx(0.5)


def test_verdict_holds_when_early_is_coarse_and_late_is_local():
    rows = [
        _row(0, 4.0, 1.0),
        _row(1, 3.0, 1.2),
        _row(2, 1.0, 3.0),
        _row(3, 0.5, 4.0),
    ]
    text = analysis._verdict(rows)
    assert text.startswith("HOLDS")


def test_verdict_fails_when_both_directions_are_wrong():
    rows = [
        _row(0, 0.5, 4.0),
        _row(1, 0.4, 3.0),
        _row(2, 3.0, 1.0),
        _row(3, 4.0, 0.5),
    ]
    text = analysis._verdict(rows)
    assert text.startswith("FAILS")
