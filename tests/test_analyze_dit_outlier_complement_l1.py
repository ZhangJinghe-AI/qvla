"""Tests for complementary-channel 3σ shoulder L1 matching."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import analyze_dit_outlier_action_detail as analysis  # noqa: E402


def test_complement_matches_l1_on_remaining_tails_only():
    tokens, channels = 32, 8
    x = torch.ones(tokens, channels)
    x[0, 0] = 40.0
    x[1:, 0] = 1.0
    x[0, 4] = 6.0
    x[1:, 4] = 1.0
    selected = torch.zeros(channels, dtype=torch.bool)
    selected[0] = True
    values = x.abs()
    plan = analysis._complement_channel_plan(
        values, selected, std_k=3.0, skip_first_token=False
    )
    assert bool(plan.matchable.item())
    assert bool(plan.over_use[:, 0].any().item())
    assert not bool(plan.over_use[:, 4].any().item())
    assert not bool(plan.body[:, 0].any().item())
    assert bool(plan.body[:, 4].any().item())
    target = float(
        ((values[:, 0] - plan.bounds[0]).clamp(min=0.0) * plan.over_use[:, 0]).sum()
    )
    removed = float((values * plan.body * plan.frac).sum())
    assert abs(removed - target) / target < analysis.L1_MATCH_RTOL
    massive = analysis._apply_outlier(x, plan.bounds, plan.over_use)
    shoulder = analysis._apply_bulk(x, plan.body, plan.frac)
    torch.testing.assert_close(massive[:, 1:], x[:, 1:])
    torch.testing.assert_close(shoulder[:, 0], x[:, 0])


def test_complement_insufficient_skips_both_sides():
    x = torch.ones(16, 4) * 0.01
    x[0, 0] = 80.0
    x[1:, 0] = 0.2
    selected = torch.zeros(4, dtype=torch.bool)
    selected[0] = True
    plan = analysis._complement_channel_plan(
        x.abs(), selected, std_k=3.0, skip_first_token=False
    )
    leftover = float(x[:, 1:].abs().sum())
    target = float(
        ((x.abs()[:, 0] - plan.bounds[0]).clamp(min=0.0) * (x.abs()[:, 0] > plan.bounds[0])).sum()
    )
    assert leftover <= target
    assert int(plan.n_insufficient.item()) == 1
    assert not bool(plan.matchable.item())
    assert not bool(plan.over_use.any().item())
    assert not bool(plan.body.any().item())
    massive = analysis._apply_outlier(x, plan.bounds, plan.over_use)
    shoulder = analysis._apply_bulk(x, plan.body, plan.frac)
    torch.testing.assert_close(massive, x)
    torch.testing.assert_close(shoulder, x)


def test_complement_skip_first_token_leaves_state_row():
    x = torch.ones(17, 8) * 0.8
    x[0, 0] = 50.0
    x[1, 0] = 20.0
    x[2, 5] = 8.0
    selected = torch.zeros(8, dtype=torch.bool)
    selected[0] = True
    plan = analysis._complement_channel_plan(
        x.abs(), selected, std_k=3.0, skip_first_token=True
    )
    assert not bool(plan.over_use[0].any().item())
    assert not bool(plan.body[0].any().item())
    massive = analysis._apply_outlier(x, plan.bounds, plan.over_use)
    shoulder = analysis._apply_bulk(x, plan.body, plan.frac)
    torch.testing.assert_close(massive[0], x[0])
    torch.testing.assert_close(shoulder[0], x[0])
