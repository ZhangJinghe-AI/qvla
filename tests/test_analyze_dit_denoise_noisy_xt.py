"""Tests for raw noisy x_t visualization."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import analyze_dit_denoise_noisy_xt as analysis  # noqa: E402


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


def test_crop_drops_trailing_zero_actions():
    final = np.zeros((40, 7))
    final[:16] = 1.0
    assert analysis._valid_action_horizon(final) == 16
    record = analysis.NoisyRecord(
        sample=0,
        noise=0,
        time_t=np.linspace(0.0, 0.75, 4),
        final_action=final,
        noisy_by_step=np.ones((4, 40, 7)),
    )
    cropped = analysis._crop_record(record, 16)
    assert cropped.final_action.shape == (16, 7)
    assert cropped.noisy_by_step.shape == (4, 16, 7)
    plotted = analysis._records_for_plots([record], None)
    assert plotted[0].final_action.shape[0] == 16


def test_row_ylim_is_fixed_and_covers_all_xt():
    final = np.linspace(0.0, 1.0, 8)
    xt_all = np.array([[-3.0, 4.0], [-20.0, 20.0]])
    lo, hi = analysis._ylim_from_final(final, xt_all)
    assert lo < -20.0
    assert hi > 20.0
    same_lo, same_hi = analysis._ylim_from_final(final, xt_all)
    assert (lo, hi) == (same_lo, same_hi)


def test_groot_time_starts_at_noise_zero():
    assert analysis._groot_time_at_step(0, 4) == 0.0
    assert analysis._groot_time_at_step(3, 4) == 0.75
    assert analysis._time_at_step(0, 4) == 1.0


class _FakeGrootHead:
    def __init__(self, num_steps: int = 4):
        self.num_inference_timesteps = num_steps
        self._dt = 1.0 / num_steps

    def denoise_step(self, actions, **kwargs):
        del kwargs
        return actions + self._dt * torch.ones_like(actions)


class _FakeGrootRunner:
    def __init__(self, num_steps: int = 4):
        self.use_cuda_graph = False
        self.model = type("M", (), {})()
        self.model.action_head = _FakeGrootHead(num_steps)

    def _fwd_loop(self, *, noise: torch.Tensor) -> torch.Tensor:
        actions = noise
        for _ in range(self.model.action_head.num_inference_timesteps):
            actions = self.model.action_head.denoise_step(actions)
        return actions


def test_groot_patch_records_input_and_matches_original():
    runner = _FakeGrootRunner()
    noise = torch.zeros(1, 5, 7)
    expected = runner._fwd_loop(noise=noise)
    with analysis._patched_record_noisy_groot(runner) as captured:
        actual = runner._fwd_loop(noise=noise)
    torch.testing.assert_close(actual, expected)
    assert len(captured[0]) == runner.model.action_head.num_inference_timesteps
    torch.testing.assert_close(captured[0][0], noise)
    last = captured[0][-1]
    torch.testing.assert_close(
        last + runner.model.action_head._dt * torch.ones_like(last),
        actual,
    )


def test_record_noisy_starts_from_input_and_ends_at_final():
    runner = _FakeRunner()
    noise = torch.zeros(1, 5, 3)
    expected = runner._fwd_loop(noise=noise)
    final, states = analysis._euler_record_noisy(runner, noise)
    torch.testing.assert_close(final, expected)
    torch.testing.assert_close(states[0], noise)
    assert len(states) == runner._num_steps
    torch.testing.assert_close(
        states[-1] + runner._dt * runner._one_step(states[-1], runner._num_steps - 1),
        final,
    )


def test_patch_matches_original_loop(tmp_path: Path):
    runner = _FakeRunner()
    noise = torch.zeros(1, 6, 3)
    expected = runner._fwd_loop(noise=noise)
    with analysis._patched_record_noisy(runner) as captured:
        actual = runner._fwd_loop(noise=noise)
    torch.testing.assert_close(actual, expected)
    assert len(captured[0]) == runner._num_steps
    record = analysis.NoisyRecord(
        sample=0,
        noise=0,
        time_t=np.linspace(1.0, 0.1, 4),
        final_action=np.linspace(0.0, 1.0, 42).reshape(6, 7),
        noisy_by_step=np.random.default_rng(0).normal(size=(4, 6, 7)),
    )
    out = tmp_path / "one.png"
    dims = tmp_path / "dims.png"
    analysis._plot_one(record, out)
    analysis._plot_all_dims(record, dims)
    assert out.is_file() and out.stat().st_size > 0
    assert dims.is_file() and dims.stat().st_size > 0
    other = analysis.NoisyRecord(
        sample=0,
        noise=1,
        time_t=record.time_t,
        final_action=record.final_action + 0.2,
        noisy_by_step=record.noisy_by_step + 0.3,
    )
    overlay = tmp_path / "overlay.png"
    finals = tmp_path / "finals.png"
    analysis._plot_all_dims_overlay([record, other], overlay)
    analysis._plot_finals_across_noises([record, other], finals)
    assert overlay.is_file() and overlay.stat().st_size > 0
    assert finals.is_file() and finals.stat().st_size > 0
    npz = tmp_path / "sample0_noise0.npz"
    analysis._write_npz(record, npz)
    loaded = analysis._load_npz(npz)
    np.testing.assert_allclose(loaded.final_action, record.final_action)
    assert (
        analysis.main(["--from-npz-dir", str(tmp_path)]) == 0
    )
    assert (tmp_path / "sample0_noise0_noisy_xt_dims.png").is_file()
