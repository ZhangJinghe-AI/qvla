"""Tests for per-channel matched L1 massive vs shoulder clip."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import analyze_dit_outlier_action_detail as analysis  # noqa: E402


def test_per_channel_prefix_takes_largest_until_extra():
    values = torch.tensor(
        [
            [9.0, 1.0],
            [4.0, 8.0],
            [3.0, 2.0],
            [1.0, 7.0],
        ]
    )
    candidate = torch.tensor(
        [
            [True, True],
            [True, True],
            [True, False],
            [True, True],
        ]
    )
    extra = torch.tensor([11.0, 8.5])
    active = torch.tensor([True, True])
    picked = analysis._largest_prefix_mask_per_channel(
        values, candidate, extra, active=active
    )
    assert bool(picked[0, 0].item()) and bool(picked[1, 0].item())
    assert not bool(picked[2, 0].item()) and not bool(picked[3, 0].item())
    assert bool(picked[1, 1].item()) and bool(picked[3, 1].item())
    assert not bool(picked[0, 1].item())


def test_per_channel_l1_matches_and_skips_starved_column():
    tokens, channels = 20, 6
    x = torch.ones(tokens, channels) * 0.4
    x[0, 0] = 12.0
    x[1:, 0] = 1.2
    x[0, 1] = 40.0
    x[1:, 1] = 0.05
    values, selected, bounds, over = analysis._outlier_over_and_bounds(x, 3.0)
    assert bool(selected[0].item()) and bool(selected[1].item())
    plan = analysis._per_channel_body_and_frac(
        values, bounds, over, selected, skip_first_token=False
    )
    assert bool(plan.matchable[0].item())
    assert not bool(plan.matchable[1].item())
    assert int(plan.n_insufficient.item()) >= 1
    assert not bool(plan.over_use[:, 1].any().item())
    assert not bool(plan.body[:, 1].any().item())
    assert not bool((plan.body & over).any().item())
    target0 = float(((values[:, 0] - bounds[0]).clamp(min=0.0) * over[:, 0]).sum())
    removed0 = float((values[:, 0] * plan.body[:, 0] * plan.frac[:, 0]).sum())
    assert abs(removed0 - target0) / target0 < analysis.L1_MATCH_RTOL
    massive = analysis._apply_outlier(x, bounds, plan.over_use)
    torch.testing.assert_close(massive[:, 1], x[:, 1])
    assert not torch.equal(massive[:, 0], x[:, 0])


def test_per_channel_skip_first_token_leaves_state_row():
    x = torch.ones(17, 16) * 0.2
    x[0, -1] = 400.0
    x[1, -1] = 80.0
    x[2:, -1] = 1.5
    values, selected, bounds, over = analysis._outlier_over_and_bounds(
        x, 3.0, skip_first_token=True
    )
    plan = analysis._per_channel_body_and_frac(
        values, bounds, over, selected, skip_first_token=True
    )
    assert not bool(over[0].any().item())
    assert not bool(plan.over_use[0].any().item())
    assert not bool(plan.body[0].any().item())
    clipped = analysis._apply_outlier(x, bounds, plan.over_use)
    bulk = analysis._apply_bulk(x, plan.body, plan.frac)
    torch.testing.assert_close(clipped[0], x[0])
    torch.testing.assert_close(bulk[0], x[0])
