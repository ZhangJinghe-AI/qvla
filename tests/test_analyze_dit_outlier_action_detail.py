"""Tests for the DiT outlier action-detail experiment."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import analyze_dit_outlier_action_detail as analysis  # noqa: E402


def test_constant_arm_offset_is_overall_not_local():
    baseline = torch.zeros(16, 7)
    changed = baseline.clone()
    changed[:, :6] += 0.2
    metrics = analysis._action_metrics(changed, baseline)
    assert math.isclose(metrics.arm_mean_shift_rmse, 0.2, abs_tol=1e-6)
    assert math.isclose(metrics.arm_endpoint_rmse, 0.2, abs_tol=1e-6)
    assert metrics.arm_net_disp_rmse < 1e-10
    assert metrics.arm_local_rmse < 1e-10
    assert metrics.arm_step_rmse < 1e-10
    assert metrics.gripper_rmse < 1e-10
    assert metrics.gripper_switch_shift == 0.0


def test_single_timestep_arm_spike_is_local_not_path_shift():
    baseline = torch.zeros(16, 7)
    changed = baseline.clone()
    changed[8, :6] += 1.0
    metrics = analysis._action_metrics(changed, baseline)
    assert metrics.arm_local_rmse > metrics.arm_mean_shift_rmse
    assert metrics.arm_net_disp_rmse < 1e-10
    assert metrics.arm_endpoint_rmse < 1e-10
    assert metrics.arm_step_rmse > 0.0
    assert metrics.gripper_rmse < 1e-10


def test_gripper_switch_shift_counts_delay():
    baseline = torch.zeros(16, 7)
    baseline[:, 6] = -1.0
    baseline[8:, 6] = 1.0
    changed = baseline.clone()
    changed[:, 6] = -1.0
    changed[10:, 6] = 1.0
    metrics = analysis._action_metrics(changed, baseline)
    assert metrics.gripper_switch_shift == 2.0
    assert metrics.arm_mean_shift_rmse < 1e-10
    assert metrics.gripper_rmse > 0.0


def test_bulk_plan_matches_l1_on_same_channels_and_skips_outlier_values():
    scales = torch.tensor(
        [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 100.0, 10000.0]
    )
    x = scales.view(1, -1).expand(50, -1).clone() * 0.2
    x[0] = scales
    selective = analysis._clip_plan(x, 3.0)
    bulk = analysis._same_channel_bulk_plan(x, selective, seed=11)
    outlier_channels = selective.mask.any(dim=0)
    target = analysis._removed_l1(x, selective)

    assert target > 0.0
    analysis._assert_l1_match(
        analysis._removed_l1(x, bulk),
        target,
        name="unit bulk",
    )
    assert not bool((bulk.mask & selective.mask).any().item())
    assert not bool((bulk.mask & ~outlier_channels.view(1, -1)).any().item())
    assert bool((bulk.mask.any(dim=0) <= outlier_channels).all().item())
    bulk_changed = analysis._apply_plan(x, bulk, kind="random")
    untouched = ~outlier_channels
    torch.testing.assert_close(bulk_changed[:, untouched], x[:, untouched])
    original = x.clone()
    analysis._apply_plan(x, bulk, kind="random")
    torch.testing.assert_close(x, original)


def test_apply_plan_changes_exactly_the_selected_action_values():
    scales = torch.tensor(
        [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 100.0, 10000.0]
    )
    action = scales.view(1, -1).expand(50, -1).clone() * 0.2
    action[0] = scales
    plan = analysis._clip_plan(action, 3.0)
    changed = analysis._apply_plan(action, plan, kind="selective")
    assert int((changed != action).sum()) == int(plan.mask.sum())


def test_same_channel_bulk_rejects_too_few_leftover_values():
    mask = torch.zeros(4, 10, dtype=torch.bool)
    mask[:, 8] = True
    selective = analysis.ClipPlan(
        mask=mask,
        bounds=torch.ones(10),
        ratios=torch.tensor([0.5, 0.5, 0.5, 0.5]),
    )
    with pytest.raises(RuntimeError, match="leftover L1"):
        analysis._same_channel_bulk_plan(torch.ones(4, 10), selective, seed=0)


def test_same_channel_bulk_rejects_when_body_l1_cannot_match():
    values = torch.tensor([[1.0, 100.0], [1.0, 0.1]])
    mask = torch.tensor([[False, True], [False, False]])
    selective = analysis.ClipPlan(
        mask=mask,
        bounds=torch.tensor([1.0, 50.0]),
        ratios=torch.tensor([0.5]),
    )
    with pytest.raises(RuntimeError, match="leftover L1"):
        analysis._same_channel_bulk_plan(values, selective, seed=0)


def test_apply_plan_does_not_mutate_input():
    scales = torch.tensor(
        [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 100.0, 10000.0]
    )
    action = scales.view(1, -1).expand(50, -1).clone() * 0.2
    action[0] = scales
    plan = analysis._clip_plan(action, 3.0)
    original = action.clone()
    changed = analysis._apply_plan(action, plan, kind="selective")
    torch.testing.assert_close(action, original)
    assert int((changed != action).sum()) == int(plan.mask.sum())


def test_exact_fp32_rejects_fp16_and_roundtrips_bf16():
    torch.manual_seed(0)
    x = torch.randn(8, 8, dtype=torch.bfloat16)
    assert torch.equal(analysis._exact_fp32(x).to(torch.bfloat16), x)
    with pytest.raises(RuntimeError, match="fp16"):
        analysis._exact_fp32(torch.randn(4, 4, dtype=torch.float16))


def test_parse_nonneg_ints_dedups_and_rejects_negative():
    assert analysis._parse_nonneg_ints("0,2,2,3", name="--samples") == [0, 2, 3]
    with pytest.raises(ValueError, match=">= 0"):
        analysis._parse_nonneg_ints("0,-1", name="--noise-seeds")


def test_clip_plan_allow_empty_when_nothing_exceeds_bound():
    scales = torch.tensor([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 2.0])
    x = scales.view(1, -1).expand(50, -1).clone()
    assert analysis._clip_plan_maybe(x, 3.0) is None
    with pytest.raises(RuntimeError, match="no action-token outliers"):
        analysis._clip_plan(x, 3.0)


def test_device_outlier_masks_match_clip_plan():
    scales = torch.tensor(
        [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 100.0, 10000.0]
    )
    x = scales.view(1, -1).expand(50, -1).clone() * 0.2
    x[0] = scales
    plan = analysis._clip_plan(x, 3.0)
    assert plan is not None
    values, _selected, bounds, over = analysis._outlier_over_and_bounds(x, 3.0)
    assert over.device == x.device
    assert torch.equal(over, plan.mask)
    torch.testing.assert_close(bounds, plan.bounds)
    clipped = analysis._apply_outlier(x, bounds, over)
    planned = analysis._apply_plan(x, plan, kind="selective")
    torch.testing.assert_close(clipped, planned)


def test_device_bulk_matches_outlier_removed_l1():
    x = torch.ones(50, 16)
    x[0, -1] = 80.0
    x[0, -2] = 60.0
    values, selected, bounds, over = analysis._outlier_over_and_bounds(x, 3.0)
    assert bool(selected.any().item())
    body, frac, target, insufficient, expanded = analysis._bulk_body_and_frac(
        values, bounds, over, selected, std_k=3.0, seed=11, drop_p=0.0
    )
    assert not bool(insufficient.item())
    assert not bool(expanded.item())
    assert frac.shape == values.shape
    removed = (values * body.to(values.dtype) * frac).sum()
    torch.testing.assert_close(removed, target, rtol=1e-5, atol=0.0)
    assert not bool((body & over).any().item())
    assert bool((frac[body] > 0.0).all().item())
    assert bool((frac[body] < 1.0).all().item())
    assert not bool((body & ~selected.view(1, -1)).any().item())
    changed = analysis._apply_bulk(x, body, frac)
    torch.testing.assert_close(changed[:, ~selected], x[:, ~selected])


def test_device_bulk_clips_mad_to_std_band():
    x = torch.ones(20, 16) * 0.2
    x[0, -1] = 80.0
    x[1, -1] = 9.0
    x[2, -1] = 8.0
    x[3, -1] = 7.0
    x[4, -1] = 1.0
    values, selected, bounds, over = analysis._outlier_over_and_bounds(x, 3.0)
    body, frac, target, insufficient, expanded = analysis._bulk_body_and_frac(
        values, bounds, over, selected, std_k=3.0, seed=0
    )
    assert bool(over[0, -1].item())
    assert not bool(insufficient.item())
    assert not bool(expanded.item())
    mad_lo = analysis._channel_mad_threshold(values)
    leftover_mask = (~over) & selected.view(1, -1) & (values > 0.0)
    below = leftover_mask & (values < mad_lo.view(1, -1))
    assert not bool((body & below).any().item())
    body_frac = frac[body]
    assert bool((body_frac > 0.0).all().item())
    assert bool((body_frac < 1.0).all().item())
    torch.testing.assert_close(
        body_frac, body_frac[0].expand_as(body_frac), rtol=1e-5, atol=0.0
    )
    removed = (values * body.to(values.dtype) * frac).sum()
    torch.testing.assert_close(removed, target, rtol=1e-5, atol=0.0)
    changed = analysis._apply_bulk(x, body, frac)
    torch.testing.assert_close(changed[over], x[over])


def test_device_bulk_expands_largest_outside_band_when_short():
    x = torch.zeros(34, 8)
    x[:16] = 0.1
    x[16:] = 5.0
    x[-2, -1] = 8.0
    x[-1, -1] = 100.0
    values, selected, bounds, over = analysis._outlier_over_and_bounds(x, 3.0)
    mad_lo = analysis._channel_mad_threshold(values)
    leftover = (~over) & selected.view(1, -1) & (values > 0.0)
    band = leftover & (values >= mad_lo.view(1, -1)) & (values <= bounds.view(1, -1))
    target = analysis._removed_l1_from_over(values, bounds, over)
    band_l1 = (values * band.to(values.dtype)).sum()
    leftover_l1 = (values * leftover.to(values.dtype)).sum()
    assert bool(over.any().item())
    assert float(target.item()) > 0.0
    assert float(band_l1.item()) <= float(target.item())
    assert float(leftover_l1.item()) > float(target.item())

    body, frac, got_target, insufficient, expanded = analysis._bulk_body_and_frac(
        values, bounds, over, selected, std_k=3.0, seed=0
    )
    assert bool(expanded.item())
    assert not bool(insufficient.item())
    torch.testing.assert_close(got_target, target)
    assert bool((band <= body).all().item())
    recruited = body & ~band
    assert bool(recruited.any().item())
    assert not bool((body & ~leftover).any().item())
    min_recruited = values[recruited].min()
    bigger = leftover & ~band & (values > min_recruited)
    assert bool((bigger <= body).all().item())
    smaller = leftover & ~band & (values < min_recruited)
    assert not bool((body & smaller).any().item())
    drop = recruited & (values == min_recruited)
    drop_one = torch.zeros_like(drop)
    drop_one.view(-1)[drop.view(-1).nonzero()[0]] = True
    without = body & ~drop_one
    without_l1 = (values * without.to(values.dtype)).sum()
    assert float(without_l1.item()) <= float(target.item())
    body_frac = frac[body]
    assert bool((body_frac > 0.0).all().item())
    assert bool((body_frac < 1.0).all().item())
    torch.testing.assert_close(
        body_frac, body_frac[0].expand_as(body_frac), rtol=1e-5, atol=0.0
    )
    removed = (values * body.to(values.dtype) * frac).sum()
    torch.testing.assert_close(removed, target, rtol=1e-5, atol=0.0)
    changed = analysis._apply_bulk(x, body, frac)
    torch.testing.assert_close(changed[over], x[over])
    torch.testing.assert_close(changed[:, ~selected], x[:, ~selected])


def test_device_bulk_insufficient_leftover_is_noop():
    x = torch.full((20, 16), 0.01)
    x[0, -1] = 80.0
    values, selected, bounds, over = analysis._outlier_over_and_bounds(x, 3.0)
    body, frac, target, insufficient, expanded = analysis._bulk_body_and_frac(
        values, bounds, over, selected, std_k=3.0, seed=0
    )
    assert bool(over.any().item())
    assert float(target.item()) > 0.0
    assert bool(insufficient.item())
    assert not bool(expanded.item())
    assert int(body.sum().item()) == 0
    assert float(frac.max().item()) == 0.0
    changed = analysis._apply_bulk(x, body, frac)
    torch.testing.assert_close(changed, x)


def test_empty_outlier_site_is_not_insufficient():
    scales = torch.tensor([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 2.0])
    x = scales.view(1, -1).expand(50, -1).clone()
    values, selected, bounds, over = analysis._outlier_over_and_bounds(x, 3.0)
    assert not bool(over.any().item())
    _body, _frac, target, insufficient, expanded = analysis._bulk_body_and_frac(
        values, bounds, over, selected, std_k=3.0, seed=0, drop_p=0.0
    )
    assert float(target.item()) == 0.0
    assert not bool(insufficient.item())
    assert not bool(expanded.item())


def test_identity_actions_helper_accepts_equal_and_rejects_shift():
    baseline = torch.zeros(2, 7)
    analysis._assert_actions_equal(baseline.clone(), baseline, name="identity")
    changed = baseline.clone()
    changed[0, 0] = 1e-3
    with pytest.raises(RuntimeError, match="hook noise"):
        analysis._assert_actions_equal(changed, baseline, name="identity write-back")


def test_all_layers_expected_applied_is_one_for_a_single_denoise_step():
    import analyze_dit_outlier_action_detail_all as all_exp

    assert (
        all_exp._expected_applied(
            10,
            capture=True,
            identity_writeback=False,
            restore_after_clip=False,
            target_step=None,
        )
        == 0
    )
    assert (
        all_exp._expected_applied(
            10,
            capture=False,
            identity_writeback=True,
            restore_after_clip=False,
            target_step=None,
        )
        == 10
    )
    assert (
        all_exp._expected_applied(
            10,
            capture=False,
            identity_writeback=False,
            restore_after_clip=True,
            target_step=None,
        )
        == 10
    )
    assert (
        all_exp._expected_applied(
            10,
            capture=False,
            identity_writeback=False,
            restore_after_clip=False,
            target_step=3,
        )
        == 1
    )
    assert (
        all_exp._expected_applied(
            10,
            capture=False,
            identity_writeback=False,
            restore_after_clip=False,
            target_step=None,
        )
        == 10
    )


def test_skip_first_token_never_clips_or_scales_state_row():
    x = torch.ones(17, 16) * 0.2
    x[0, -1] = 400.0
    x[1, -1] = 80.0
    x[2, -1] = 9.0
    x[3, -1] = 8.0
    x[4, -1] = 7.0
    values, selected, bounds, over = analysis._outlier_over_and_bounds(
        x, 3.0, skip_first_token=True
    )
    assert not bool(over[0].any().item())
    assert bool(over[1, -1].item())
    clipped = analysis._apply_outlier(x, bounds, over)
    torch.testing.assert_close(clipped[0], x[0])
    body, frac, target, insufficient, _expanded = analysis._bulk_body_and_frac(
        values,
        bounds,
        over,
        selected,
        std_k=3.0,
        seed=0,
        skip_first_token=True,
    )
    assert not bool(insufficient.item())
    assert not bool(body[0].any().item())
    assert float(target.item()) > 0.0
    bulk = analysis._apply_bulk(x, body, frac)
    torch.testing.assert_close(bulk[0], x[0])


def test_skip_first_token_ignores_state_when_selecting_channels():
    x = torch.ones(17, 16) * 0.2
    x[0, 0] = 1.0e4
    values, selected, bounds, over = analysis._outlier_over_and_bounds(
        x, 3.0, skip_first_token=True
    )
    del values, bounds
    assert not bool(over.any().item())
    assert not bool(selected[0].item())


def test_valid_action_horizon_drops_trailing_zeros():
    import analyze_dit_outlier_action_detail_all as all_exp

    final = torch.zeros(40, 7)
    final[:16] = 1.0
    assert all_exp._valid_action_horizon(final) == 16

