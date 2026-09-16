"""Tests for amax-ordered complementary 3σ tail clipping."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import analyze_dit_outlier_action_detail as analysis  # noqa: E402


def test_amax_order_clips_larger_amax_remaining_channel_first():
    tokens, channels = 24, 6
    x = torch.ones(tokens, channels)
    x[0, 0] = 12.0
    x[0, 1] = 9.0
    x[0, 2] = 6.0
    x[0, 3] = 8.5
    selected = torch.zeros(channels, dtype=torch.bool)
    selected[0] = True
    plan = analysis._complement_amax_plan(
        x.abs(), selected, std_k=3.0, skip_first_token=False
    )
    assert bool(plan.matchable.item())
    used = plan.channel_frac > 0
    assert bool(used[1].item())
    if float(plan.channel_frac[1].item()) < 1.0:
        assert not bool(used[2].item())
        assert not bool(used[3].item())
    target = float(
        ((x.abs()[:, 0] - plan.bounds[0]).clamp(min=0.0) * plan.over_use[:, 0]).sum()
    )
    removed = float(
        ((x.abs() - plan.bounds.view(1, -1)).clamp(min=0.0)
         * plan.over_rem.to(x.dtype)
         * plan.channel_frac.view(1, -1)).sum()
    )
    assert abs(removed - target) / target < analysis.L1_MATCH_RTOL
    massive = analysis._apply_outlier(x, plan.bounds, plan.over_use)
    shoulder = analysis._apply_channel_excess_frac(
        x, plan.bounds, plan.over_rem, plan.channel_frac
    )
    torch.testing.assert_close(massive[:, 1:], x[:, 1:])
    torch.testing.assert_close(shoulder[:, 0], x[:, 0])


def test_amax_insufficient_skips_both_sides():
    x = torch.ones(16, 4) * 0.01
    x[0, 0] = 80.0
    x[1:, 0] = 0.2
    selected = torch.zeros(4, dtype=torch.bool)
    selected[0] = True
    plan = analysis._complement_amax_plan(
        x.abs(), selected, std_k=3.0, skip_first_token=False
    )
    assert int(plan.n_insufficient.item()) == 1
    assert not bool(plan.matchable.item())
    massive = analysis._apply_outlier(x, plan.bounds, plan.over_use)
    shoulder = analysis._apply_channel_excess_frac(
        x, plan.bounds, plan.over_rem, plan.channel_frac
    )
    torch.testing.assert_close(massive, x)
    torch.testing.assert_close(shoulder, x)


def test_amax_skip_first_token_leaves_state_row():
    x = torch.ones(17, 8) * 0.8
    x[0, 0] = 50.0
    x[1, 0] = 20.0
    x[2, 5] = 8.0
    selected = torch.zeros(8, dtype=torch.bool)
    selected[0] = True
    plan = analysis._complement_amax_plan(
        x.abs(), selected, std_k=3.0, skip_first_token=True
    )
    assert not bool(plan.over_use[0].any().item())
    assert not bool(plan.over_rem[0].any().item())
    massive = analysis._apply_outlier(x, plan.bounds, plan.over_use)
    shoulder = analysis._apply_channel_excess_frac(
        x, plan.bounds, plan.over_rem, plan.channel_frac
    )
    torch.testing.assert_close(massive[0], x[0])
    torch.testing.assert_close(shoulder[0], x[0])
