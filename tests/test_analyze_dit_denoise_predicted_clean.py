"""Tests for predicted-clean a_hat = x - t v visualization."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import analyze_dit_denoise_predicted_clean as analysis  # noqa: E402


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


class _FakeGrootHead:
    def __init__(self, num_steps: int = 4):
        self.num_inference_timesteps = num_steps
        self._dt = 1.0 / num_steps

    def denoise_step(self, actions, **kwargs):
        dt = float(kwargs.get("dt", self._dt))
        return actions + dt * torch.ones_like(actions)


class _FakeGrootRunner:
    def __init__(self, num_steps: int = 4):
        self.use_cuda_graph = False
        self._dt = 1.0 / num_steps
        self.model = type("M", (), {})()
        self.model.action_head = _FakeGrootHead(num_steps)

    def _fwd_loop(self, *, noise: torch.Tensor) -> torch.Tensor:
        actions = noise
        for step in range(self.model.action_head.num_inference_timesteps):
            actions = self.model.action_head.denoise_step(actions, dt=self._dt)
            del step
        return actions


def _record(**kwargs) -> analysis.CleanRecord:
    defaults = dict(
        sample=0,
        noise=0,
        time_t=np.linspace(1.0, 0.1, 4),
        final_action=np.linspace(0.0, 1.0, 42).reshape(6, 7),
        clean_by_step=np.random.default_rng(0).normal(size=(4, 6, 7)),
    )
    defaults.update(kwargs)
    return analysis.CleanRecord(**defaults)


def test_predicted_clean_inverts_linear_flow():
    action = torch.full((1, 5, 3), 2.0)
    eps = torch.full((1, 5, 3), 5.0)
    t = 0.4
    x_t = t * eps + (1.0 - t) * action
    velocity = eps - action
    torch.testing.assert_close(
        analysis._predicted_clean(x_t, velocity, t), action
    )


def test_groot_predicted_clean_inverts_linear_flow():
    action = torch.full((1, 5, 3), 2.0)
    eps = torch.full((1, 5, 3), 5.0)
    t = 0.4
    x_t = (1.0 - t) * eps + t * action
    velocity = action - eps
    torch.testing.assert_close(
        analysis._predicted_clean_groot(x_t, velocity, t), action
    )


def test_last_clean_equals_euler_output_on_pi05_schedule():
    runner = _FakeRunner()
    noise = torch.zeros(1, 5, 3)
    expected = runner._fwd_loop(noise=noise)
    final, clean = analysis._euler_record_clean(runner, noise)
    torch.testing.assert_close(final, expected)
    torch.testing.assert_close(clean[-1], final)
    torch.testing.assert_close(
        clean[0],
        noise - 1.0 * runner._one_step(noise, 0),
    )


def test_groot_patch_records_clean_and_matches_original():
    runner = _FakeGrootRunner()
    noise = torch.zeros(1, 5, 7)
    expected = runner._fwd_loop(noise=noise)
    with analysis._patched_record_clean_groot(runner) as captured:
        actual = runner._fwd_loop(noise=noise)
    torch.testing.assert_close(actual, expected)
    n_steps = runner.model.action_head.num_inference_timesteps
    assert len(captured[0]) == n_steps
    t0 = analysis._groot_time_at_step(0, n_steps)
    torch.testing.assert_close(
        captured[0][0],
        analysis._predicted_clean_groot(noise, torch.ones_like(noise), t0),
    )
    torch.testing.assert_close(captured[0][-1], actual)


def test_crop_drops_trailing_zero_actions():
    final = np.zeros((40, 7))
    final[:16] = 1.0
    assert analysis._valid_action_horizon(final) == 16
    record = _record(
        time_t=np.linspace(0.0, 0.75, 4),
        final_action=final,
        clean_by_step=np.ones((4, 40, 7)),
    )
    cropped = analysis._crop_record(record, 16)
    assert cropped.final_action.shape == (16, 7)
    assert cropped.clean_by_step.shape == (4, 16, 7)
    plotted = analysis._records_for_plots([record], None)
    assert plotted[0].final_action.shape[0] == 16


def test_last_clean_check_ignores_masked_tail():
    final = np.zeros((8, 7))
    final[:4] = 0.5
    clean = np.zeros((3, 8, 7))
    clean[-1, :4] = 0.5
    clean[-1, 4:] = 9.0
    analysis._assert_last_clean_matches_final(clean, final)


def test_row_ylim_is_fixed_and_covers_all_ahat():
    final = np.linspace(0.0, 1.0, 8)
    ahat_all = np.array([[-3.0, 4.0], [-20.0, 20.0]])
    lo, hi = analysis._ylim_from_final(final, ahat_all)
    assert lo < -20.0
    assert hi > 20.0


def test_rmse_and_pairwise(tmp_path: Path):
    final = np.zeros((6, 7))
    left = _record(
        noise=0,
        final_action=final,
        clean_by_step=np.zeros((4, 6, 7)),
    )
    right = _record(
        noise=1,
        final_action=final,
        clean_by_step=np.full((4, 6, 7), 2.0),
    )
    np.testing.assert_allclose(analysis._rmse_to_final(left), 0.0)
    np.testing.assert_allclose(analysis._rmse_to_final(right), 2.0)
    pair = analysis._pairwise_rmse([left, right])
    np.testing.assert_allclose(pair, 2.0)
    out = tmp_path / "rmse.png"
    analysis._plot_rmse([left, right], out)
    assert out.is_file() and out.stat().st_size > 0


def test_plots_and_from_npz_dir(tmp_path: Path):
    record = _record()
    out = tmp_path / "one.png"
    dims = tmp_path / "dims.png"
    analysis._plot_one(record, out)
    analysis._plot_all_dims(record, dims)
    assert out.is_file() and out.stat().st_size > 0
    assert dims.is_file() and dims.stat().st_size > 0
    other = _record(
        noise=1,
        final_action=record.final_action + 0.2,
        clean_by_step=record.clean_by_step + 0.3,
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
    np.testing.assert_allclose(loaded.clean_by_step, record.clean_by_step)
    other_npz = tmp_path / "sample0_noise1.npz"
    analysis._write_npz(other, other_npz)
    assert analysis.main(["--from-npz-dir", str(tmp_path)]) == 0
    assert (tmp_path / "sample0_noise0_predicted_clean_dims.png").is_file()
    assert (tmp_path / "sample0_noises_ahat_rmse.png").is_file()
    assert (tmp_path / "summary.txt").is_file()
