"""Tests for unselected-channel μ+3σ normal-outlier occupancy."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import analyze_dit_outlier_action_detail as analysis  # noqa: E402


def test_spike_only_selected_channel_has_empty_normal():
    x = torch.ones(32, 4)
    x[0, 0] = 40.0
    split = analysis._normal_outlier_split(x, std_k=3.0, skip_first_token=False)
    assert int(split.n_selected_ch.item()) >= 1
    assert int(split.n_massive.item()) >= 1
    assert int(split.n_normal.item()) == 0
    assert not bool(split.normal.any().item())
    assert bool(split.massive[0, 0].item())
    assert not bool(split.normal[:, 1:].any().item())


def test_elevated_selected_body_counts_as_normal():
    x = torch.ones(32, 4)
    x[0, 0] = 40.0
    x[1:, 0] = 5.0
    split = analysis._normal_outlier_split(x, std_k=3.0, skip_first_token=False)
    assert bool(split.massive[0, 0].item())
    assert not bool(split.normal[0, 0].item())
    assert int(split.n_normal.item()) == 31
    assert bool(split.normal[1:, 0].all().item())
    assert not bool(split.normal[:, 1:].any().item())
    assert float(split.t_ref.item()) < 5.0
    assert float(split.excess_normal.item()) > 0.0


def test_normal_stays_inside_selected_non_massive():
    torch.manual_seed(0)
    x = torch.randn(24, 32).abs() + 0.4
    x[0, 0] = float(x.amax().item()) * 8.0
    split = analysis._normal_outlier_split(x, std_k=3.0, skip_first_token=False)
    _, selected, _, _ = analysis._outlier_over_and_bounds(x, 3.0)
    assert not bool((split.normal & split.massive).any().item())
    assert not bool((split.normal & (~selected.view(1, -1))).any().item())
    assert not bool((split.normal & split.ref_mask).any().item())


def test_normal_match_plan_matches_excess_l1():
    x = torch.ones(32, 4)
    x[0, 0] = 40.0
    x[1:, 0] = 5.0
    plan = analysis._normal_outlier_match_plan(x, std_k=3.0, skip_first_token=False)
    assert bool(plan.matchable.item())
    assert int(plan.n_insufficient.item()) == 0
    target = float(
        ((x.abs() - plan.bounds.view(1, -1)).clamp(min=0.0) * plan.over_use.to(x.dtype)).sum()
    )
    removed = float(
        ((x.abs() - plan.t_ref).clamp(min=0.0) * plan.normal.to(x.dtype) * plan.frac).sum()
    )
    assert abs(removed - target) / target < analysis.L1_MATCH_RTOL
    massive = analysis._apply_outlier(x, plan.bounds, plan.over_use)
    shoulder = analysis._apply_ref_excess_frac(x, plan.t_ref, plan.normal, plan.frac)
    torch.testing.assert_close(massive[1:, 0], x[1:, 0])
    torch.testing.assert_close(shoulder[0, 0], x[0, 0])


def test_normal_match_plan_skips_when_normal_empty():
    x = torch.ones(32, 4)
    x[0, 0] = 40.0
    plan = analysis._normal_outlier_match_plan(x, std_k=3.0, skip_first_token=False)
    assert int(plan.n_insufficient.item()) == 1
    assert not bool(plan.matchable.item())
    massive = analysis._apply_outlier(x, plan.bounds, plan.over_use)
    shoulder = analysis._apply_ref_excess_frac(x, plan.t_ref, plan.normal, plan.frac)
    torch.testing.assert_close(massive, x)
    torch.testing.assert_close(shoulder, x)


def test_normal_skip_first_token_leaves_state_row():
    x = torch.ones(17, 8) * 0.8
    x[0, 0] = 50.0
    x[1, 0] = 20.0
    x[2:, 0] = 5.0
    split = analysis._normal_outlier_split(x, std_k=3.0, skip_first_token=True)
    assert not bool(split.massive[0].any().item())
    assert not bool(split.normal[0].any().item())
    assert not bool(split.ref_mask[0].any().item())


def test_next_largest_matches_massive_token_count():
    x = torch.ones(32, 4)
    x[0, 0] = 40.0
    x[1:4, 0] = 20.0
    split = analysis._normal_outlier_split(
        x, std_k=3.0, skip_first_token=False, ref_mode="next_largest"
    )
    assert int(split.n_massive.item()) == 1
    assert int(split.n_normal.item()) == 1
    assert bool(split.normal[1:4, 0].any().item())
    assert not bool(split.normal[4:, 0].any().item())
    plan = analysis._normal_outlier_match_plan(
        x, std_k=3.0, skip_first_token=False, ref_mode="next_largest"
    )
    assert bool(plan.matchable.item())
    target = float(
        ((x.abs() - plan.bounds.view(1, -1)).clamp(min=0.0) * plan.over_use.to(x.dtype)).sum()
    )
    removed = float(
        ((x.abs() - plan.t_ref).clamp(min=0.0) * plan.normal.to(x.dtype) * plan.frac).sum()
    )
    assert abs(removed - target) / target < analysis.L1_MATCH_RTOL
    massive = analysis._apply_outlier(x, plan.bounds, plan.over_use)
    shoulder = analysis._apply_ref_excess_frac(x, plan.t_ref, plan.normal, plan.frac)
    torch.testing.assert_close(massive[1:, 0], x[1:, 0])
    torch.testing.assert_close(shoulder[0, 0], x[0, 0])


def test_leftover_mad_keeps_column_bulk_ordinary():
    x = torch.ones(32, 4)
    x[0, 0] = 40.0
    x[1, 0] = 12.0
    x[2, 0] = 11.0
    x[3, 0] = 10.0
    split = analysis._normal_outlier_split(
        x, std_k=3.0, skip_first_token=False, ref_mode="leftover_mad"
    )
    assert bool(split.massive[0, 0].item())
    assert not bool(split.massive[1:4, 0].any().item())
    assert bool(split.normal[1:4, 0].any().item())
    assert not bool(split.normal[4:, 0].any().item())
    assert not bool(split.normal[:, 1:].any().item())
    plan = analysis._normal_outlier_match_plan(
        x, std_k=3.0, skip_first_token=False, ref_mode="leftover_mad"
    )
    assert bool(plan.matchable.item())
    target = float(
        ((x.abs() - plan.bounds.view(1, -1)).clamp(min=0.0) * plan.over_use.to(x.dtype)).sum()
    )
    removed = float(
        ((x.abs() - analysis._bound_view(plan.t_ref)).clamp(min=0.0)
         * plan.normal.to(x.dtype) * plan.frac).sum()
    )
    assert abs(removed - target) / target < analysis.L1_MATCH_RTOL
    massive = analysis._apply_outlier(x, plan.bounds, plan.over_use)
    shoulder = analysis._apply_ref_excess_frac(x, plan.t_ref, plan.normal, plan.frac)
    torch.testing.assert_close(massive[1:, 0], x[1:, 0])
    torch.testing.assert_close(shoulder[0, 0], x[0, 0])
    torch.testing.assert_close(shoulder[4:, 0], x[4:, 0])


def test_leftover_mad_empty_when_leftover_is_flat():
    x = torch.ones(32, 4)
    x[0, 0] = 40.0
    split = analysis._normal_outlier_split(
        x, std_k=3.0, skip_first_token=False, ref_mode="leftover_mad"
    )
    assert int(split.n_normal.item()) == 0
    plan = analysis._normal_outlier_match_plan(
        x, std_k=3.0, skip_first_token=False, ref_mode="leftover_mad"
    )
    assert int(plan.n_insufficient.item()) == 1
    assert not bool(plan.matchable.item())


def test_channel_mad_partitions_selected_column_by_all_token_floor():
    x = torch.ones(32, 4)
    x[0, 0] = 40.0
    x[1:4, 0] = 12.0
    split = analysis._normal_outlier_split(
        x, std_k=3.0, skip_first_token=False, ref_mode="channel_mad"
    )
    _, selected, _bounds, over = analysis._outlier_over_and_bounds(x, 3.0)
    t_mad = analysis._channel_mad_threshold(x.abs())
    assert bool(selected[0].item())
    assert bool(split.massive[0, 0].item())
    assert not bool(split.normal[0, 0].item())
    torch.testing.assert_close(split.t_ref, t_mad)
    expected = (
        (~over)
        & selected.view(1, -1)
        & (x.abs() > t_mad.view(1, -1))
    )
    assert torch.equal(split.normal, expected)
    ordinary = selected.view(1, -1) & (~split.massive) & (~split.normal)
    assert not bool((ordinary & (x.abs() > t_mad.view(1, -1))).any().item())
    assert not bool((split.normal & (~selected.view(1, -1))).any().item())
    plan = analysis._normal_outlier_match_plan(
        x, std_k=3.0, skip_first_token=False, ref_mode="channel_mad"
    )
    assert bool(plan.matchable.item())
    target = float(
        ((x.abs() - plan.bounds.view(1, -1)).clamp(min=0.0) * plan.over_use.to(x.dtype)).sum()
    )
    removed = float(
        ((x.abs() - analysis._bound_view(plan.t_ref)).clamp(min=0.0)
         * plan.normal.to(x.dtype) * plan.frac).sum()
    )
    assert abs(removed - target) / target < analysis.L1_MATCH_RTOL
    massive = analysis._apply_outlier(x, plan.bounds, plan.over_use)
    shoulder = analysis._apply_ref_excess_frac(x, plan.t_ref, plan.normal, plan.frac)
    torch.testing.assert_close(massive[1:, 0], x[1:, 0])
    torch.testing.assert_close(shoulder[0, 0], x[0, 0])
    torch.testing.assert_close(shoulder[4:, 0], x[4:, 0])


def test_channel_mad_skip_first_fits_mad_on_action_tokens():
    x = torch.ones(17, 8) * 0.8
    x[0, 0] = 50.0
    x[1, 0] = 20.0
    x[2:, 0] = 5.0
    split = analysis._normal_outlier_split(
        x, std_k=3.0, skip_first_token=True, ref_mode="channel_mad"
    )
    t_mad = analysis._channel_mad_threshold(x[1:].abs())
    torch.testing.assert_close(split.t_ref, t_mad)
    assert not bool(split.massive[0].any().item())
    assert not bool(split.normal[0].any().item())


def test_channel_mad_floor_uses_all_tokens_not_leftover_only():
    x = torch.ones(32, 4)
    x[0, 0] = 1000.0
    x[1:16, 0] = 20.0
    all_split = analysis._normal_outlier_split(
        x, std_k=3.0, skip_first_token=False, ref_mode="channel_mad"
    )
    leftover_split = analysis._normal_outlier_split(
        x, std_k=3.0, skip_first_token=False, ref_mode="leftover_mad"
    )
    assert bool(all_split.massive[0, 0].item())
    assert float(all_split.t_ref[0].item()) != float(leftover_split.t_ref[0].item())
    assert not bool((all_split.normal & all_split.massive).any().item())


def test_channel_mad_by_step_script_wires_channel_mad():
    text = (TOOLS / "analyze_dit_outlier_lp_hp_channel_mad_by_step.py").read_text(
        encoding="utf-8"
    )
    assert 'normal_ref_mode="channel_mad"' in text
    assert "leftover_mad" not in text


def test_channel_mad_full_by_step_script_wires_full_clip():
    text = (TOOLS / "analyze_dit_outlier_lp_hp_channel_mad_full_by_step.py").read_text(
        encoding="utf-8"
    )
    assert 'normal_ref_mode="channel_mad"' in text
    assert "normal_full_clip=True" in text
    assert "allow_empty=True" in text


def test_channel_mad_full_clip_floors_shoulder_and_leaves_massive():
    x = torch.ones(32, 4)
    x[0, 0] = 40.0
    x[1:4, 0] = 12.0
    split = analysis._normal_outlier_split(
        x, std_k=3.0, skip_first_token=False, ref_mode="channel_mad"
    )
    plan = analysis._normal_outlier_full_plan(
        x, std_k=3.0, skip_first_token=False, ref_mode="channel_mad"
    )
    assert float(plan.frac.item()) == 1.0
    assert int(plan.n_insufficient.item()) == 0
    assert torch.equal(plan.over_use, split.massive)
    assert torch.equal(plan.normal, split.normal)
    massive = analysis._apply_outlier(x, plan.bounds, plan.over_use)
    shoulder = analysis._apply_ref_excess_frac(x, plan.t_ref, plan.normal, plan.frac)
    torch.testing.assert_close(massive[1:, 0], x[1:, 0])
    torch.testing.assert_close(shoulder[0, 0], x[0, 0])
    torch.testing.assert_close(shoulder[4:, 0], x[4:, 0])
    if bool(split.normal[1, 0].item()):
        torch.testing.assert_close(
            shoulder[1, 0].abs(),
            plan.t_ref[0],
            atol=1e-5,
            rtol=1e-5,
        )
    assert float(massive[0, 0].abs().item()) <= float(plan.bounds[0].item()) + 1e-5


def test_channel_mad_full_clip_does_not_skip_when_matching_would():
    x = torch.ones(32, 4)
    x[0, 0] = 40.0
    matched = analysis._normal_outlier_match_plan(
        x, std_k=3.0, skip_first_token=False, ref_mode="channel_mad"
    )
    full = analysis._normal_outlier_full_plan(
        x, std_k=3.0, skip_first_token=False, ref_mode="channel_mad"
    )
    assert int(matched.n_insufficient.item()) == 1
    assert not bool(matched.matchable.item())
    assert int(full.n_insufficient.item()) == 0
    assert bool(full.over_use[0, 0].item())
    massive = analysis._apply_outlier(x, full.bounds, full.over_use)
    assert float(massive[0, 0].abs().item()) < 40.0
    torch.testing.assert_close(massive[1:, 0], x[1:, 0])
