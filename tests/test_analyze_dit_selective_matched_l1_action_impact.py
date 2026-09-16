"""Tests for selective mean+std clip with matched L1 energy."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

from qvla.core.clip import select_channels_by_robust_amax

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import analyze_dit_selective_matched_l1_action_impact as analysis  # noqa: E402


def _varied_outliers() -> torch.Tensor:
    values = torch.ones(50, 8)
    for channel in range(8):
        values[:, channel] = 1.0 + 0.2 * float(channel)
        values[:, channel] += 0.05 * torch.linspace(0.0, 1.0, 50)
    values[0, 0] = 40.0
    values[:, 1] = torch.linspace(1.0, 8.0, 50)
    return values


def test_hits_target_l1_and_leaves_unselected_untouched():
    x = _varied_outliers()
    clipped, stats = analysis._clip_selective_matched_k(x, 0.005)
    assert math.isclose(stats.removed_l1_frac, 0.005, abs_tol=1e-6)
    assert stats.n_selected_channels >= 1
    assert stats.token_clip_frac > 0.0
    amax = x.amax(dim=0)
    selected, _, _, _ = select_channels_by_robust_amax(amax)
    unselected = ~selected
    if bool(unselected.any().item()):
        torch.testing.assert_close(clipped[:, unselected], x[:, unselected])


def test_impossible_alpha_raises():
    x = _varied_outliers()
    with pytest.raises(RuntimeError, match="cannot hit"):
        analysis._clip_selective_matched_k(x, 0.9)


def test_equal_amax_raises_mad():
    with pytest.raises(RuntimeError, match="MAD"):
        analysis._clip_selective_matched_k(torch.ones(16, 4), 0.005)


def test_k_is_scale_equivariant():
    x = _varied_outliers()
    _, a = analysis._clip_selective_matched_k(x, 0.005)
    _, b = analysis._clip_selective_matched_k(x * 25.0, 0.005)
    assert math.isclose(a.k, b.k, rel_tol=1e-4, abs_tol=1e-4)
    assert math.isclose(a.removed_l1_frac, b.removed_l1_frac, abs_tol=1e-6)
    assert a.n_selected_channels == b.n_selected_channels


def _dummy_row(name: str, idx: int, rmse: float) -> analysis.LayerResult:
    metrics = analysis.detail.ActionMetrics(
        total_rmse=rmse,
        arm_mean_shift_rmse=0.0,
        arm_endpoint_rmse=0.0,
        arm_net_disp_rmse=0.0,
        arm_local_rmse=0.0,
        arm_step_rmse=0.0,
        gripper_rmse=0.0,
        gripper_switch_shift=0.0,
    )
    return analysis.LayerResult(
        layer_name=name,
        layer_idx=idx,
        kind="down_proj",
        mean_k=1.0,
        n_selected_channels=4,
        mean_removed_l1_frac=0.005,
        mean_token_clip_frac=0.01,
        metrics=metrics,
    )


def test_parse_nonneg_ints_unique_order():
    assert analysis._parse_nonneg_ints("0,2,2,1", name="--samples") == [0, 2, 1]


def test_parse_nonneg_ints_rejects_empty_and_negative():
    with pytest.raises(ValueError, match="at least one"):
        analysis._parse_nonneg_ints(" , ", name="--samples")
    with pytest.raises(ValueError, match=">= 0"):
        analysis._parse_nonneg_ints("0,-1", name="--noise-seeds")


def test_mean_layer_results_averages_rmse():
    a = [_dummy_row("L0", 0, 1.0), _dummy_row("L1", 1, 3.0)]
    b = [_dummy_row("L0", 0, 3.0), _dummy_row("L1", 1, 5.0)]
    mean = analysis._mean_layer_results([a, b])
    assert math.isclose(mean[0].metrics.total_rmse, 2.0)
    assert math.isclose(mean[1].metrics.total_rmse, 4.0)


def test_depth_corr_detects_monotone_depth():
    rows = [_dummy_row(f"L{i}", i, 0.1 * float(i)) for i in range(6)]
    pearson, spearman = analysis._depth_corr(rows)
    assert pearson > 0.99
    assert spearman > 0.99
