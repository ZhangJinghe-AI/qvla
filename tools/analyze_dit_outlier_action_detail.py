#!/usr/bin/env python
r"""Causally test whether pi0.5 DiT action-token outliers are negligible detail.

Fix the same calibration sample and diffusion noise, then compare three actions:

1. original model (no intervention)
2. outliers only: on MAD-selected channels, clamp values above μ+kσ
3. matched bulk control: same channels, same removed L1, leftover in
   [median+3×1.4826×MAD, μ+3σ] scaled by one shared fraction. If that
   band is too small, keep recruiting the largest leftover below the
   MAD floor until the set can match L1, then scale the whole set.

Score ``arm_mean_shift`` (coarse path) and ``arm_local`` (residual detail).
The hypothesis holds only if (2) barely moves the coarse path and local detail
stays small, while (3) shifts the coarse path. If (2) also shifts the path, or
(2) and (3) look the same, the spikes are not ignorable detail.

The bulk control must match removed L1 on the same channels or the comparison
is invalid. Several sample/noise pairs are required. Identity write-back
(clip, then write the original values back) must leave actions bit-identical
to the no-hook baseline; otherwise the measurement is hook noise.

Example:

    CUDA_VISIBLE_DEVICES=5 uv run python \
      tools/analyze_dit_outlier_action_detail.py \
      --checkpoint /data/share/pi05_libero_finetuned_v044 \
      --calibration-data ../calibration_data/libero_goal_30_7_demo.npz \
      --samples 0,1,2,3 --noise-seeds 0,1 --random-trials 4 \
      --layer-regex 'expert_stack\.layers\.8\.mlp\.down_proj$' \
      --output-dir tools/img/dit_outlier_action_detail_matched_l1_s4n2
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np
import torch

L1_MATCH_RTOL = 1e-5

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from qvla.adapters import get_adapter  # noqa: E402
from qvla.adapters.pi05.obs import build_pi05_request  # noqa: E402
from qvla.adapters.pi05.step_hook import find_expert_runner, patched_one_step  # noqa: E402
from qvla.config import QVLAConfig  # noqa: E402
from qvla.core.clip import (  # noqa: E402
    MAD_NORMAL_SCALE,
    SELECTIVE_CHANNEL_MAD_K,
    channel_outlier_mean_std_amax,
    select_channels_by_robust_amax,
)
from qvla.runtime import list_target_modules  # noqa: E402


@dataclass(frozen=True)
class ClipPlan:
    mask: torch.Tensor
    bounds: torch.Tensor
    ratios: torch.Tensor


@dataclass(frozen=True)
class ActionMetrics:
    total_rmse: float
    arm_mean_shift_rmse: float
    arm_endpoint_rmse: float
    arm_net_disp_rmse: float
    arm_local_rmse: float
    arm_step_rmse: float
    gripper_rmse: float
    gripper_switch_shift: float


@dataclass(frozen=True)
class StepResult:
    step: int
    outlier_count: int
    selected_channels: int
    bulk_count: int
    removed_l1_pct: float
    bulk_removed_l1_pct: float
    selective: ActionMetrics
    random_mean: ActionMetrics
    random_std: ActionMetrics


@dataclass(frozen=True)
class ConditionResult:
    sample: int
    noise: int
    steps: list[StepResult]


def _finite(x: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(x).all().item()):
        raise RuntimeError(f"{name} contains NaN/Inf.")


def _exact_fp32(x: torch.Tensor) -> torch.Tensor:
    """Promote activations to fp32 without extra rounding.

    bf16→fp32 is exact. fp16→fp32 is not, and writing that snapshot back was
    the measurement floor that hid the clip effect.
    """
    if x.dtype == torch.float16:
        raise RuntimeError(
            "Refusing to snapshot fp16 activations; that rounding is the "
            "measurement floor this script removes. Use bf16 or fp32."
        )
    return x.detach().to(torch.float32)


def _parse_nonneg_ints(text: str, *, name: str) -> list[int]:
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if not parts:
        raise ValueError(f"{name} must list at least one integer.")
    values: list[int] = []
    seen: set[int] = set()
    for part in parts:
        value = int(part)
        if value < 0:
            raise ValueError(f"{name} values must be >= 0, got {value}.")
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def _mean(xs: list[float]) -> float:
    if not xs:
        raise ValueError("Cannot average an empty list.")
    return sum(xs) / len(xs)


def _removed_l1(action_tokens: torch.Tensor, plan: ClipPlan) -> float:
    values = action_tokens.abs().to(torch.float32)
    if int(plan.mask.sum().item()) != int(plan.ratios.numel()):
        raise RuntimeError(
            f"Plan mask has {int(plan.mask.sum().item())} values but "
            f"{int(plan.ratios.numel())} ratios."
        )
    if int(plan.mask.sum().item()) < 1:
        raise RuntimeError("Clip plan is empty.")
    return float(
        (values[plan.mask] * (1.0 - plan.ratios.to(dtype=values.dtype))).sum().item()
    )


def _assert_l1_match(actual: float, target: float, *, name: str) -> None:
    if target <= 0.0:
        raise RuntimeError(f"{name} target removed L1 must be > 0, got {target}.")
    rel = abs(actual - target) / target
    if rel > L1_MATCH_RTOL:
        raise RuntimeError(
            f"{name} removed L1 {actual:.8g} != target {target:.8g} "
            f"(rel={rel:.3g} > {L1_MATCH_RTOL})."
        )


def _assert_actions_equal(
    changed: torch.Tensor,
    baseline: torch.Tensor,
    *,
    name: str,
) -> None:
    if changed.shape != baseline.shape:
        raise RuntimeError(
            f"{name} action shape {tuple(changed.shape)} != "
            f"baseline {tuple(baseline.shape)}."
        )
    if torch.equal(changed, baseline):
        return
    delta = (changed - baseline).to(torch.float64)
    rmse = float(delta.square().mean().sqrt().item())
    max_abs = float(delta.abs().max().item())
    raise RuntimeError(
        f"{name} moved actions: RMSE={rmse:.6e}, max_abs={max_abs:.6e}. "
        "Measurements would still include hook noise."
    )


def _one_dit_layer(model: torch.nn.Module, regex: str) -> tuple[str, torch.nn.Module]:
    pattern = re.compile(regex)
    config = QVLAConfig.pi05_default()
    hits = [
        (name, module)
        for name, scope, module in list_target_modules(model, config)
        if scope == "dit" and pattern.search(name)
    ]
    if len(hits) != 1:
        raise RuntimeError(
            f"--layer-regex must match exactly one DiT linear, got {len(hits)}: "
            f"{[name for name, _ in hits]}"
        )
    return hits[0]


def _rmse(x: torch.Tensor) -> float:
    return float(x.square().mean().sqrt().item())


def _first_sign_flip(series: torch.Tensor) -> int:
    """First index where ``series`` flips sign; ``len(series)`` if it never does."""
    values = series.reshape(-1).to(torch.float64)
    if values.numel() < 1:
        raise ValueError("gripper series is empty.")
    signs = torch.sign(values)
    nonzero = torch.nonzero(signs != 0, as_tuple=False).flatten()
    if nonzero.numel() == 0:
        return int(values.numel())
    start = signs[int(nonzero[0].item())]
    flipped = torch.nonzero((signs * start) < 0, as_tuple=False).flatten()
    if flipped.numel() == 0:
        return int(values.numel())
    return int(flipped[0].item())


def _action_metrics(
    changed: torch.Tensor,
    baseline: torch.Tensor,
) -> ActionMetrics:
    if changed.shape != baseline.shape or changed.ndim != 2:
        raise ValueError(
            f"Actions must be matching (horizon, dims), got "
            f"{tuple(changed.shape)} and {tuple(baseline.shape)}."
        )
    if int(changed.shape[0]) < 2 or int(changed.shape[1]) < 2:
        raise ValueError(
            "Need horizon>=2 and at least one arm dim plus gripper, got "
            f"{tuple(changed.shape)}."
        )
    delta = (changed - baseline).to(torch.float64)
    _finite(delta, "action delta")
    arm = delta[:, :-1]
    gripper = delta[:, -1]
    return ActionMetrics(
        total_rmse=_rmse(delta),
        arm_mean_shift_rmse=_rmse(arm.mean(dim=0)),
        arm_endpoint_rmse=_rmse(arm[-1]),
        arm_net_disp_rmse=_rmse(arm[-1] - arm[0]),
        arm_local_rmse=_rmse(arm - arm.mean(dim=0)),
        arm_step_rmse=_rmse(torch.diff(arm, n=1, dim=0)),
        gripper_rmse=_rmse(gripper),
        gripper_switch_shift=float(
            abs(
                _first_sign_flip(changed[:, -1])
                - _first_sign_flip(baseline[:, -1])
            )
        ),
    )


def _clip_plan(
    action_tokens: torch.Tensor,
    std_k: float,
    *,
    allow_empty: bool = False,
) -> ClipPlan | None:
    if action_tokens.ndim != 2 or min(action_tokens.shape) < 2:
        raise ValueError(
            "action_tokens must be non-empty (tokens, channels), got "
            f"{tuple(action_tokens.shape)}."
        )
    values = action_tokens.abs().to(torch.float32)
    _finite(values, "action-token activation")
    hard_amax = values.amax(dim=0)
    selected, _median, _mad, _threshold = select_channels_by_robust_amax(
        hard_amax
    )
    all_bounds = channel_outlier_mean_std_amax(values, std_k=std_k)
    bounds = torch.where(selected, all_bounds, hard_amax)
    mask = values > bounds.view(1, -1)
    if bool((mask & ~selected.view(1, -1)).any().item()):
        raise RuntimeError("Unselected channels contain clip candidates.")
    if int(mask.sum().item()) < 1:
        if allow_empty:
            return None
        raise RuntimeError("Selective rule found no action-token outliers.")
    ratios = bounds.view(1, -1).expand_as(values)[mask] / values[mask]
    if not bool(((ratios > 0.0) & (ratios < 1.0)).all().item()):
        raise RuntimeError("Clip ratios must lie strictly inside (0, 1).")
    return ClipPlan(mask=mask, bounds=bounds, ratios=ratios)


def _clip_plan_maybe(
    action_tokens: torch.Tensor,
    std_k: float,
) -> ClipPlan | None:
    return _clip_plan(action_tokens, std_k, allow_empty=True)


def _select_channels_device(channel_amax: torch.Tensor) -> torch.Tensor:
    """MAD channel mask on ``channel_amax``'s device. No host sync."""
    values = channel_amax.detach().to(torch.float32)
    center = torch.quantile(values, 0.5)
    mad = torch.quantile((values - center).abs(), 0.5)
    threshold = center + SELECTIVE_CHANNEL_MAD_K * MAD_NORMAL_SCALE * mad
    return values > threshold


def _outlier_over_and_bounds(
    action_tokens: torch.Tensor,
    std_k: float,
    *,
    skip_first_token: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(abs_values, selected, bounds, over)`` on the input device.

    ``skip_first_token`` (GR00T) fits MAD / μ+kσ on tokens after row 0 and
    never marks that leading state token for clipping.
    """
    if action_tokens.ndim != 2 or min(action_tokens.shape) < 2:
        raise ValueError(
            "action_tokens must be non-empty (tokens, channels), got "
            f"{tuple(action_tokens.shape)}."
        )
    if skip_first_token and int(action_tokens.shape[0]) < 3:
        raise ValueError(
            "skip_first_token needs a state row plus >= 2 action tokens, "
            f"got {tuple(action_tokens.shape)}."
        )
    values = action_tokens.abs().to(torch.float32)
    stat = values[1:] if skip_first_token else values
    hard_amax = stat.amax(dim=0)
    selected = _select_channels_device(hard_amax)
    bounds = torch.where(
        selected,
        channel_outlier_mean_std_amax(stat, std_k=std_k),
        hard_amax,
    )
    over = (values > bounds.view(1, -1)) & selected.view(1, -1)
    if skip_first_token:
        over = over.clone()
        over[0] = False
    return values, selected, bounds, over


def _apply_outlier(
    x: torch.Tensor,
    bounds: torch.Tensor,
    over: torch.Tensor,
) -> torch.Tensor:
    mag = torch.where(over, bounds.view(1, -1).to(dtype=x.dtype), x.abs())
    return x.sign() * mag


def _removed_l1_from_over(
    values: torch.Tensor,
    bounds: torch.Tensor,
    over: torch.Tensor,
) -> torch.Tensor:
    return ((values - bounds.view(1, -1)).clamp(min=0.0) * over.to(values.dtype)).sum()


def _channel_mad_threshold(values: torch.Tensor) -> torch.Tensor:
    """Per-channel ``median + 3 · 1.4826 · MAD`` of token magnitudes."""
    center = torch.quantile(values, 0.5, dim=0)
    mad = torch.quantile((values - center.view(1, -1)).abs(), 0.5, dim=0)
    return center + SELECTIVE_CHANNEL_MAD_K * MAD_NORMAL_SCALE * mad


def _largest_prefix_mask(
    values: torch.Tensor,
    candidate: torch.Tensor,
    extra: torch.Tensor,
    *,
    active: torch.Tensor,
) -> torch.Tensor:
    """Shortest largest-first prefix of ``candidate`` whose L1 exceeds ``extra``.

    ``active`` gates the result. No host sync.
    """
    flat_values = values.reshape(-1)
    flat_cand = candidate.reshape(-1)
    neg = torch.tensor(-1.0, device=values.device, dtype=values.dtype)
    keys = torch.where(flat_cand, flat_values, neg)
    order = torch.argsort(keys, descending=True)
    sorted_pos = keys.index_select(0, order).clamp(min=0.0)
    hits = sorted_pos.cumsum(dim=0) > extra
    has_hit = hits.any()
    cutoff = hits.to(torch.int64).argmax()
    rank = torch.empty_like(order)
    rank[order] = torch.arange(
        order.numel(), device=values.device, dtype=torch.int64
    )
    picked = active & has_hit & flat_cand & (rank <= cutoff)
    return picked.view_as(values)


def _largest_n_mask(
    values: torch.Tensor,
    candidate: torch.Tensor,
    n: torch.Tensor,
    *,
    active: torch.Tensor,
) -> torch.Tensor:
    """Largest ``n`` tokens in ``candidate``. ``active`` gates the result."""
    flat_values = values.reshape(-1)
    flat_cand = candidate.reshape(-1)
    neg = torch.tensor(-1.0, device=values.device, dtype=values.dtype)
    keys = torch.where(flat_cand, flat_values, neg)
    order = torch.argsort(keys, descending=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(
        order.numel(), device=values.device, dtype=torch.int64
    )
    picked = active & flat_cand & (rank < n.to(dtype=torch.int64))
    return picked.view_as(values)


def _bulk_body_and_frac(
    values: torch.Tensor,
    bounds: torch.Tensor,
    over: torch.Tensor,
    selected: torch.Tensor,
    *,
    std_k: float,
    seed: int,
    drop_p: float = 0.5,
    skip_first_token: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scale leftover starting from the MAD-to-μ+3σ band by one shared fraction.

    On MAD-selected channels, first take leftover in
    ``[median + 3·1.4826·MAD, μ+kσ]``. If that band's L1 is not strictly
    larger than the outlier target, recruit leftover below the MAD floor
    largest-first until the combined set can match. Then multiply every
    chosen value by ``1 - R / sum(|x|)``. Outlier positions and unselected
    channels are not touched. ``seed`` and ``drop_p`` are ignored.

    Returns ``(body, frac, target, insufficient, expanded)`` on ``values``'s
    device with no host sync. ``insufficient`` is True only when all leftover
    still cannot match; empty outlier sites are not counted insufficient.
    """
    del std_k, seed, drop_p
    leftover = (~over) & selected.view(1, -1) & (values > 0.0)
    if skip_first_token:
        leftover = leftover.clone()
        leftover[0] = False
        mad_src = values[1:]
    else:
        mad_src = values
    mad_lo = _channel_mad_threshold(mad_src)
    band = leftover & (values >= mad_lo.view(1, -1)) & (values <= bounds.view(1, -1))
    target = _removed_l1_from_over(values, bounds, over)
    band_l1 = (values * band.to(values.dtype)).sum()
    leftover_l1 = (values * leftover.to(values.dtype)).sum()
    insufficient = (target > 0.0) & (leftover_l1 <= target)
    need_expand = (target > 0.0) & (band_l1 <= target) & (~insufficient)
    recruited = _largest_prefix_mask(
        values,
        leftover & ~band,
        target - band_l1,
        active=need_expand,
    )
    body_mask = band | recruited
    tiny = torch.tensor(
        torch.finfo(torch.float32).tiny,
        device=values.device,
        dtype=values.dtype,
    )
    combined_l1 = (values * body_mask.to(values.dtype)).sum()
    frac_u = target / combined_l1.clamp(min=tiny)
    frac = torch.where(body_mask, frac_u.expand_as(values), torch.zeros_like(values))
    frac = torch.where(insufficient, torch.zeros_like(frac), frac)
    body = body_mask & (frac > 0.0) & (frac < 1.0)
    return body, frac, target, insufficient, need_expand


def _apply_bulk(x: torch.Tensor, body: torch.Tensor, frac: torch.Tensor) -> torch.Tensor:
    scale = (1.0 - frac).to(dtype=x.dtype)
    return torch.where(body, x * scale, x)


@dataclass(frozen=True)
class PerChannelPlan:
    body: torch.Tensor
    frac: torch.Tensor
    over_use: torch.Tensor
    matchable: torch.Tensor
    n_target: torch.Tensor
    n_expand_enough: torch.Tensor
    n_insufficient: torch.Tensor
    max_rel: torch.Tensor
    target_sum: torch.Tensor


def _largest_prefix_mask_per_channel(
    values: torch.Tensor,
    candidate: torch.Tensor,
    extra: torch.Tensor,
    *,
    active: torch.Tensor,
) -> torch.Tensor:
    """Per-column shortest largest-first prefix whose L1 exceeds ``extra``."""
    if values.shape != candidate.shape or values.ndim != 2:
        raise ValueError(
            f"values/candidate must be (T,C), got {tuple(values.shape)} vs "
            f"{tuple(candidate.shape)}."
        )
    if extra.shape != active.shape or extra.ndim != 1 or extra.shape[0] != values.shape[1]:
        raise ValueError(
            f"extra/active must be (C,) with C={values.shape[1]}, got "
            f"{tuple(extra.shape)} vs {tuple(active.shape)}."
        )
    tokens = int(values.shape[0])
    extra = extra.to(dtype=values.dtype).clamp(min=0.0)
    neg = torch.tensor(-1.0, device=values.device, dtype=values.dtype)
    keys = torch.where(candidate, values, neg)
    order = torch.argsort(keys, dim=0, descending=True)
    sorted_pos = keys.gather(0, order).clamp(min=0.0)
    hits = sorted_pos.cumsum(dim=0) > extra.view(1, -1)
    has_hit = hits.any(dim=0)
    cutoff = hits.to(torch.int64).argmax(dim=0)
    rank = torch.empty_like(order)
    token_rank = torch.arange(tokens, device=values.device, dtype=torch.int64)
    rank.scatter_(0, order, token_rank.unsqueeze(1).expand_as(order))
    return (
        active.view(1, -1)
        & has_hit.view(1, -1)
        & candidate
        & (rank <= cutoff.view(1, -1))
    )


def _per_channel_body_and_frac(
    values: torch.Tensor,
    bounds: torch.Tensor,
    over: torch.Tensor,
    selected: torch.Tensor,
    *,
    skip_first_token: bool = False,
) -> PerChannelPlan:
    """Match removed L1 independently on each MAD-selected channel.

    Shoulder leftover starts in ``[median + 3·1.4826·MAD, μ+kσ]``. If that
    band is not strictly larger than the channel's massive L1, recruit leftover
    below the MAD floor largest-first. Channels whose entire leftover still
    cannot exceed the massive L1 are skipped on both sides.
    """
    leftover = (~over) & selected.view(1, -1) & (values > 0.0)
    if skip_first_token:
        leftover = leftover.clone()
        leftover[0] = False
        mad_src = values[1:]
    else:
        mad_src = values
    mad_lo = _channel_mad_threshold(mad_src)
    band = leftover & (values >= mad_lo.view(1, -1)) & (values <= bounds.view(1, -1))
    target_j = ((values - bounds.view(1, -1)).clamp(min=0.0) * over.to(values.dtype)).sum(0)
    band_l1_j = (values * band.to(values.dtype)).sum(0)
    leftover_l1_j = (values * leftover.to(values.dtype)).sum(0)
    has_target = selected & (target_j > 0.0)
    insufficient_j = has_target & (leftover_l1_j <= target_j)
    matchable = has_target & ~insufficient_j
    need_expand_j = matchable & (band_l1_j <= target_j)
    recruited = _largest_prefix_mask_per_channel(
        values,
        leftover & ~band,
        target_j - band_l1_j,
        active=need_expand_j,
    )
    body_mask = (band | recruited) & matchable.view(1, -1)
    tiny = torch.tensor(
        torch.finfo(torch.float32).tiny,
        device=values.device,
        dtype=values.dtype,
    )
    combined_l1_j = (values * body_mask.to(values.dtype)).sum(0)
    frac_j = torch.where(matchable, target_j / combined_l1_j.clamp(min=tiny), torch.zeros_like(target_j))
    frac = torch.where(body_mask, frac_j.view(1, -1).expand_as(values), torch.zeros_like(values))
    body = body_mask & (frac > 0.0) & (frac < 1.0)
    over_use = over & matchable.view(1, -1)
    removed_j = (values * body.to(values.dtype) * frac).sum(0)
    rel_j = torch.where(
        matchable,
        (removed_j - target_j).abs() / target_j.clamp(min=tiny),
        torch.zeros_like(target_j),
    )
    return PerChannelPlan(
        body=body,
        frac=frac,
        over_use=over_use,
        matchable=matchable,
        n_target=has_target.sum(),
        n_expand_enough=need_expand_j.sum(),
        n_insufficient=insufficient_j.sum(),
        max_rel=rel_j.max(),
        target_sum=(target_j * matchable.to(target_j.dtype)).sum(),
    )


@dataclass(frozen=True)
class ComplementPlan:
    bounds: torch.Tensor
    over_use: torch.Tensor
    body: torch.Tensor
    frac: torch.Tensor
    matchable: torch.Tensor
    n_target: torch.Tensor
    n_expand_enough: torch.Tensor
    n_insufficient: torch.Tensor
    max_rel: torch.Tensor


def _complement_channel_plan(
    values: torch.Tensor,
    selected: torch.Tensor,
    *,
    std_k: float,
    skip_first_token: bool = False,
) -> ComplementPlan:
    """Match massive L1 on MAD channels to 3σ tails on the complementary channels.

    Massive clip uses ``μ+kσ`` on ``selected`` channels. Shoulder clip uses
    ``μ+kσ`` tails on the remaining channels, then expands largest-first into
    those channels' bodies if the tails cannot strictly exceed the massive L1.
    If all remaining leftover still cannot match, both sides skip.
    """
    if selected.ndim != 1 or int(selected.shape[0]) != int(values.shape[1]):
        raise ValueError(
            f"selected must be (C,) with C={values.shape[1]}, got {tuple(selected.shape)}."
        )
    stat = values[1:] if skip_first_token else values
    all_bounds = channel_outlier_mean_std_amax(stat, std_k=std_k)
    remaining = ~selected
    over_sel = (values > all_bounds.view(1, -1)) & selected.view(1, -1)
    leftover = remaining.view(1, -1) & (values > 0.0)
    if skip_first_token:
        over_sel = over_sel.clone()
        over_sel[0] = False
        leftover = leftover.clone()
        leftover[0] = False
    over_rem = leftover & (values > all_bounds.view(1, -1))
    target = _removed_l1_from_over(values, all_bounds, over_sel)
    band_l1 = (values * over_rem.to(values.dtype)).sum()
    leftover_l1 = (values * leftover.to(values.dtype)).sum()
    has_target = target > 0.0
    insufficient = has_target & (leftover_l1 <= target)
    matchable = has_target & ~insufficient
    need_expand = matchable & (band_l1 <= target)
    recruited = _largest_prefix_mask(
        values,
        leftover & ~over_rem,
        target - band_l1,
        active=need_expand,
    )
    body_mask = (over_rem | recruited) & matchable
    tiny = torch.tensor(
        torch.finfo(torch.float32).tiny,
        device=values.device,
        dtype=values.dtype,
    )
    combined_l1 = (values * body_mask.to(values.dtype)).sum()
    frac_u = target / combined_l1.clamp(min=tiny)
    frac = torch.where(body_mask, frac_u.expand_as(values), torch.zeros_like(values))
    body = body_mask & (frac > 0.0) & (frac < 1.0)
    over_use = over_sel & matchable
    removed = (values * body.to(values.dtype) * frac).sum()
    rel = torch.where(
        matchable,
        (removed - target).abs() / target.clamp(min=tiny),
        torch.zeros((), device=values.device, dtype=values.dtype),
    )
    one = torch.ones((), dtype=torch.int64, device=values.device)
    zero = torch.zeros((), dtype=torch.int64, device=values.device)
    return ComplementPlan(
        bounds=all_bounds,
        over_use=over_use,
        body=body,
        frac=frac,
        matchable=matchable,
        n_target=torch.where(has_target, one, zero),
        n_expand_enough=torch.where(need_expand, one, zero),
        n_insufficient=torch.where(insufficient, one, zero),
        max_rel=rel,
    )


@dataclass(frozen=True)
class AmaxGreedyPlan:
    bounds: torch.Tensor
    over_use: torch.Tensor
    over_rem: torch.Tensor
    channel_frac: torch.Tensor
    matchable: torch.Tensor
    n_target: torch.Tensor
    n_full: torch.Tensor
    n_insufficient: torch.Tensor
    max_rel: torch.Tensor


def _apply_channel_excess_frac(
    x: torch.Tensor,
    bounds: torch.Tensor,
    over: torch.Tensor,
    channel_frac: torch.Tensor,
) -> torch.Tensor:
    """Move over-threshold magnitudes a fraction of the way to ``bounds``."""
    values = x.abs()
    frac = channel_frac.view(1, -1).to(dtype=values.dtype)
    excess = (values - bounds.view(1, -1)).clamp(min=0.0) * over.to(values.dtype)
    new_mag = values - excess * frac
    return x.sign() * new_mag.to(dtype=x.dtype)


def _complement_amax_plan(
    values: torch.Tensor,
    selected: torch.Tensor,
    *,
    std_k: float,
    skip_first_token: bool = False,
) -> AmaxGreedyPlan:
    """Clip remaining-channel 3σ tails in descending amax order until L1 matches.

    Unselected channels are sorted by amax. Each contributes a full ``μ+kσ``
    clamp of its tails until the massive removed L1 is reached; the last
    channel is scaled so the total matches exactly. If all remaining tails
    still cannot reach the target, both sides skip.
    """
    if selected.ndim != 1 or int(selected.shape[0]) != int(values.shape[1]):
        raise ValueError(
            f"selected must be (C,) with C={values.shape[1]}, got {tuple(selected.shape)}."
        )
    channels = int(values.shape[1])
    stat = values[1:] if skip_first_token else values
    all_bounds = channel_outlier_mean_std_amax(stat, std_k=std_k)
    over_sel = (values > all_bounds.view(1, -1)) & selected.view(1, -1)
    over_rem = (values > all_bounds.view(1, -1)) & (~selected).view(1, -1)
    if skip_first_token:
        over_sel = over_sel.clone()
        over_sel[0] = False
        over_rem = over_rem.clone()
        over_rem[0] = False
    target = _removed_l1_from_over(values, all_bounds, over_sel)
    rem_l1 = ((values - all_bounds.view(1, -1)).clamp(min=0.0) * over_rem.to(values.dtype)).sum(0)
    amax = stat.amax(dim=0)
    neg = torch.tensor(-1.0, device=values.device, dtype=amax.dtype)
    keys = torch.where((~selected) & (rem_l1 > 0.0), amax, neg)
    order = torch.argsort(keys, descending=True)
    sorted_l1 = rem_l1.index_select(0, order)
    cs = sorted_l1.cumsum(dim=0)
    leftover_l1 = rem_l1.sum()
    has_target = target > 0.0
    insufficient = has_target & (leftover_l1 < target)
    matchable = has_target & (leftover_l1 >= target)
    hits = cs >= target
    cutoff = hits.to(torch.int64).argmax()
    rank = torch.empty(channels, dtype=torch.int64, device=values.device)
    rank[order] = torch.arange(channels, device=values.device, dtype=torch.int64)
    usable = (~selected) & (rem_l1 > 0.0)
    full = matchable & usable & (rank < cutoff)
    last = matchable & usable & (rank == cutoff)
    tiny = torch.tensor(
        torch.finfo(torch.float32).tiny,
        device=values.device,
        dtype=values.dtype,
    )
    last_l1 = sorted_l1[cutoff]
    prefix_l1 = cs[cutoff] - last_l1
    f_last = (target - prefix_l1) / last_l1.clamp(min=tiny)
    ones = torch.ones_like(rem_l1)
    zeros = torch.zeros_like(rem_l1)
    channel_frac = torch.where(full, ones, zeros)
    channel_frac = torch.where(last, f_last.expand_as(channel_frac), channel_frac)
    channel_frac = torch.where(matchable, channel_frac, zeros)
    over_use = over_sel & matchable
    removed = (rem_l1 * channel_frac).sum()
    rel = torch.where(
        matchable,
        (removed - target).abs() / target.clamp(min=tiny),
        torch.zeros((), device=values.device, dtype=values.dtype),
    )
    one = torch.ones((), dtype=torch.int64, device=values.device)
    zero = torch.zeros((), dtype=torch.int64, device=values.device)
    return AmaxGreedyPlan(
        bounds=all_bounds,
        over_use=over_use,
        over_rem=over_rem,
        channel_frac=channel_frac,
        matchable=matchable,
        n_target=torch.where(has_target, one, zero),
        n_full=full.sum(),
        n_insufficient=torch.where(insufficient, one, zero),
        max_rel=rel,
    )


def _leftover_pool_plan(
    values: torch.Tensor,
    selected: torch.Tensor,
    *,
    std_k: float,
    skip_first_token: bool = False,
) -> ComplementPlan:
    """Match massive L1 using a global μ+kσ tail on leftover activations.

    Leftover is every activation except massive sites on MAD-selected
    channels. Shoulder first takes leftover values above the leftover-wide
    ``μ+kσ``, then expands largest-first below that floor if needed. If all
    leftover still cannot match, both sides skip.
    """
    if selected.ndim != 1 or int(selected.shape[0]) != int(values.shape[1]):
        raise ValueError(
            f"selected must be (C,) with C={values.shape[1]}, got {tuple(selected.shape)}."
        )
    stat = values[1:] if skip_first_token else values
    all_bounds = channel_outlier_mean_std_amax(stat, std_k=std_k)
    over_sel = (values > all_bounds.view(1, -1)) & selected.view(1, -1)
    leftover = (~over_sel) & (values > 0.0)
    if skip_first_token:
        over_sel = over_sel.clone()
        over_sel[0] = False
        leftover = leftover.clone()
        leftover[0] = False
    weight = leftover.to(values.dtype)
    n_left = weight.sum()
    tiny = torch.tensor(
        torch.finfo(torch.float32).tiny,
        device=values.device,
        dtype=values.dtype,
    )
    mu = (values * weight).sum() / n_left.clamp(min=tiny)
    var = ((values - mu).square() * weight).sum() / n_left.clamp(min=tiny)
    bound = mu + float(std_k) * var.clamp(min=0.0).sqrt()
    over_pool = leftover & (values > bound)
    target = _removed_l1_from_over(values, all_bounds, over_sel)
    band_l1 = (values * over_pool.to(values.dtype)).sum()
    leftover_l1 = (values * weight).sum()
    has_target = target > 0.0
    enough_stats = n_left >= 2
    insufficient = has_target & ((leftover_l1 <= target) | (~enough_stats))
    matchable = has_target & ~insufficient
    need_expand = matchable & (band_l1 <= target)
    recruited = _largest_prefix_mask(
        values,
        leftover & ~over_pool,
        target - band_l1,
        active=need_expand,
    )
    body_mask = (over_pool | recruited) & matchable
    combined_l1 = (values * body_mask.to(values.dtype)).sum()
    frac_u = target / combined_l1.clamp(min=tiny)
    frac = torch.where(body_mask, frac_u.expand_as(values), torch.zeros_like(values))
    body = body_mask & (frac > 0.0) & (frac < 1.0)
    over_use = over_sel & matchable
    removed = (values * body.to(values.dtype) * frac).sum()
    rel = torch.where(
        matchable,
        (removed - target).abs() / target.clamp(min=tiny),
        torch.zeros((), device=values.device, dtype=values.dtype),
    )
    one = torch.ones((), dtype=torch.int64, device=values.device)
    zero = torch.zeros((), dtype=torch.int64, device=values.device)
    return ComplementPlan(
        bounds=all_bounds,
        over_use=over_use,
        body=body,
        frac=frac,
        matchable=matchable,
        n_target=torch.where(has_target, one, zero),
        n_expand_enough=torch.where(need_expand, one, zero),
        n_insufficient=torch.where(insufficient, one, zero),
        max_rel=rel,
    )


@dataclass(frozen=True)
class NormalOutlierSplit:
    """Token split using unselected activations as the normal-outlier ruler."""

    bounds: torch.Tensor
    t_ref: torch.Tensor
    massive: torch.Tensor
    normal: torch.Tensor
    selected_bulk: torch.Tensor
    ref_mask: torch.Tensor
    n_selected_ch: torch.Tensor
    n_ref: torch.Tensor
    n_massive: torch.Tensor
    n_normal: torch.Tensor
    n_selected_bulk: torch.Tensor
    l1_massive: torch.Tensor
    l1_normal: torch.Tensor
    l1_selected_bulk: torch.Tensor
    excess_massive: torch.Tensor
    excess_normal: torch.Tensor


def _masked_channel_median_mad(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-column median and MAD of ``mask``ed magnitudes. ``ok`` is n>=2."""
    n = mask.sum(dim=0)
    ok = n >= 2
    nan = torch.tensor(float("nan"), device=values.device, dtype=values.dtype)
    filled = torch.where(mask, values, nan)
    filled = torch.where(ok.view(1, -1), filled, torch.zeros_like(filled))
    center = torch.nanquantile(filled, 0.5, dim=0)
    mad = torch.nanquantile((filled - center.view(1, -1)).abs(), 0.5, dim=0)
    return center, mad, ok


def _bound_view(t_ref: torch.Tensor) -> torch.Tensor:
    if t_ref.ndim == 0:
        return t_ref
    if t_ref.ndim != 1:
        raise ValueError(f"t_ref must be scalar or (C,), got {tuple(t_ref.shape)}.")
    return t_ref.view(1, -1)


def _normal_outlier_split(
    action_tokens: torch.Tensor,
    *,
    std_k: float,
    skip_first_token: bool = False,
    ref_mode: str = "unselected_std",
    ref_std_k: float | None = None,
) -> NormalOutlierSplit:
    """Split tokens into massive / normal-outlier / selected bulk.

    Massive is unchanged: MAD-selected channels, ``|x| > μ_j+kσ_j``.

    ``unselected_std``: leftover above unselected ``μ+kσ``.
    ``next_largest``: the next ``n_massive`` leftover tokens.
    ``leftover_mad``: leftover above that column's leftover median+3MAD.
    ``channel_mad``: leftover above that column's all-token median+3MAD
    (same formula as MAD channel selection, fit on every token).
    """
    allowed = {"unselected_std", "next_largest", "leftover_mad", "channel_mad"}
    if ref_mode not in allowed:
        raise ValueError(f"ref_mode must be one of {sorted(allowed)}, got {ref_mode!r}.")
    values, selected, bounds, over = _outlier_over_and_bounds(
        action_tokens, std_k, skip_first_token=skip_first_token
    )
    valid = torch.ones_like(values, dtype=torch.bool)
    if skip_first_token:
        valid[0] = False
    ref_mask = valid & (~selected.view(1, -1))
    massive = over & valid
    leftover_sel = valid & selected.view(1, -1) & (~massive)
    tiny = torch.tensor(
        torch.finfo(torch.float32).tiny,
        device=values.device,
        dtype=values.dtype,
    )
    n_massive = massive.sum()
    if ref_mode == "next_largest":
        t_ref = torch.zeros((), device=values.device, dtype=values.dtype)
        normal = _largest_n_mask(
            values, leftover_sel, n_massive, active=n_massive > 0
        )
        excess_normal = (values * normal.to(values.dtype)).sum()
    elif ref_mode == "leftover_mad":
        center, mad, ok = _masked_channel_median_mad(values, leftover_sel)
        t_ref = center + SELECTIVE_CHANNEL_MAD_K * MAD_NORMAL_SCALE * mad
        t_ref = torch.where(ok, t_ref, torch.full_like(t_ref, float("inf")))
        normal = leftover_sel & ok.view(1, -1) & (values > t_ref.view(1, -1))
        excess_normal = (
            (values - t_ref.view(1, -1)).clamp(min=0.0) * normal.to(values.dtype)
        ).sum()
        ref_mask = leftover_sel
    elif ref_mode == "channel_mad":
        # Same median+3×1.4826×MAD formula as channel selection, but along
        # tokens of this column (including massive). Band is (MAD, μ+kσ].
        stat = values[1:] if skip_first_token else values
        t_ref = _channel_mad_threshold(stat)
        normal = leftover_sel & (values > t_ref.view(1, -1))
        excess_normal = (
            (values - t_ref.view(1, -1)).clamp(min=0.0) * normal.to(values.dtype)
        ).sum()
        ref_mask = leftover_sel
    else:
        ruler_k = float(std_k if ref_std_k is None else ref_std_k)
        weight = ref_mask.to(values.dtype)
        n_pool = weight.sum()
        mu = (values * weight).sum() / n_pool.clamp(min=tiny)
        var = ((values - mu).square() * weight).sum() / n_pool.clamp(min=tiny)
        t_ref = mu + ruler_k * var.clamp(min=0.0).sqrt()
        ok_ref = n_pool >= 2
        normal = leftover_sel & ok_ref & (values > t_ref)
        excess_normal = ((values - t_ref).clamp(min=0.0) * normal.to(values.dtype)).sum()
    selected_bulk = leftover_sel & ~normal
    excess_massive = (
        (values - bounds.view(1, -1)).clamp(min=0.0) * massive.to(values.dtype)
    ).sum()
    n_ref = ref_mask.to(values.dtype).sum()
    return NormalOutlierSplit(
        bounds=bounds,
        t_ref=t_ref,
        massive=massive,
        normal=normal,
        selected_bulk=selected_bulk,
        ref_mask=ref_mask,
        n_selected_ch=selected.sum(),
        n_ref=n_ref,
        n_massive=n_massive,
        n_normal=normal.sum(),
        n_selected_bulk=selected_bulk.sum(),
        l1_massive=(values * massive.to(values.dtype)).sum(),
        l1_normal=(values * normal.to(values.dtype)).sum(),
        l1_selected_bulk=(values * selected_bulk.to(values.dtype)).sum(),
        excess_massive=excess_massive,
        excess_normal=excess_normal,
    )


def _apply_ref_excess_frac(
    x: torch.Tensor,
    t_ref: torch.Tensor,
    mask: torch.Tensor,
    frac: torch.Tensor,
) -> torch.Tensor:
    """Move masked magnitudes a fraction of the way to ``t_ref``."""
    values = x.abs()
    excess = (values - _bound_view(t_ref)).clamp(min=0.0) * mask.to(values.dtype)
    new_mag = values - excess * frac.to(dtype=values.dtype)
    return x.sign() * new_mag.to(dtype=x.dtype)


@dataclass(frozen=True)
class NormalMatchPlan:
    bounds: torch.Tensor
    t_ref: torch.Tensor
    over_use: torch.Tensor
    normal: torch.Tensor
    frac: torch.Tensor
    matchable: torch.Tensor
    n_target: torch.Tensor
    n_insufficient: torch.Tensor
    max_rel: torch.Tensor


def _normal_outlier_match_plan(
    action_tokens: torch.Tensor,
    *,
    std_k: float,
    skip_first_token: bool = False,
    ref_mode: str = "unselected_std",
    ref_std_k: float | None = None,
) -> NormalMatchPlan:
    """Match massive excess L1 using the chosen normal-outlier set.

    ``unselected_std`` clamps that set toward ``t_ref``. ``leftover_mad``
    clamps leftover tails toward each column's leftover median+3MAD.
    ``channel_mad`` clamps leftover tails toward each column's all-token
    median+3MAD. ``next_largest`` scales the next ``n_massive`` leftover
    tokens toward zero. If the set cannot strictly exceed the massive
    excess, both sides skip. Does not expand into selected bulk.
    """
    split = _normal_outlier_split(
        action_tokens,
        std_k=std_k,
        skip_first_token=skip_first_token,
        ref_mode=ref_mode,
        ref_std_k=ref_std_k,
    )
    tiny = torch.tensor(
        torch.finfo(torch.float32).tiny,
        device=action_tokens.device,
        dtype=torch.float32,
    )
    target = split.excess_massive
    has_target = target > 0.0
    insufficient = has_target & (split.excess_normal <= target)
    matchable = has_target & ~insufficient
    frac_u = target / split.excess_normal.clamp(min=tiny)
    frac = torch.where(matchable, frac_u, torch.zeros((), device=action_tokens.device, dtype=torch.float32))
    over_use = split.massive & matchable
    normal = split.normal & matchable
    removed = split.excess_normal * frac
    rel = torch.where(
        matchable,
        (removed - target).abs() / target.clamp(min=tiny),
        torch.zeros((), device=action_tokens.device, dtype=torch.float32),
    )
    one = torch.ones((), dtype=torch.int64, device=action_tokens.device)
    zero = torch.zeros((), dtype=torch.int64, device=action_tokens.device)
    return NormalMatchPlan(
        bounds=split.bounds,
        t_ref=split.t_ref,
        over_use=over_use,
        normal=normal,
        frac=frac,
        matchable=matchable,
        n_target=torch.where(has_target, one, zero),
        n_insufficient=torch.where(insufficient, one, zero),
        max_rel=rel,
    )


def _normal_outlier_full_plan(
    action_tokens: torch.Tensor,
    *,
    std_k: float,
    skip_first_token: bool = False,
    ref_mode: str = "channel_mad",
    ref_std_k: float | None = None,
) -> NormalMatchPlan:
    """Clip massive and normal independently, no excess-L1 matching.

    Massive goes all the way to ``μ+kσ``. Normal leftover in
    ``(t_ref, μ+kσ]`` goes all the way to ``t_ref``. Tokens above ``μ+kσ``
    are left unchanged on the normal side.
    """
    split = _normal_outlier_split(
        action_tokens,
        std_k=std_k,
        skip_first_token=skip_first_token,
        ref_mode=ref_mode,
        ref_std_k=ref_std_k,
    )
    device = action_tokens.device
    one = torch.ones((), dtype=torch.float32, device=device)
    one_i = torch.ones((), dtype=torch.int64, device=device)
    zero_i = torch.zeros((), dtype=torch.int64, device=device)
    zero_f = torch.zeros((), dtype=torch.float32, device=device)
    has_massive = split.massive.any()
    return NormalMatchPlan(
        bounds=split.bounds,
        t_ref=split.t_ref,
        over_use=split.massive,
        normal=split.normal,
        frac=one,
        matchable=has_massive | split.normal.any(),
        n_target=torch.where(has_massive, one_i, zero_i),
        n_insufficient=zero_i,
        max_rel=zero_f,
    )


def _same_channel_bulk_plan(
    action_tokens: torch.Tensor,
    selective: ClipPlan,
    *,
    seed: int,
) -> ClipPlan:
    """Scale leftover from the MAD-to-μ+3σ band, expanding largest-first if needed.

    ``seed`` is ignored. If all same-channel leftover still cannot supply
    that L1 without being zeroed, this raises instead of falling back.
    """
    del seed
    values = action_tokens.abs().to(torch.float32)
    _finite(values, "action-token activation")
    outlier_channels = selective.mask.any(dim=0)
    if not bool(outlier_channels.any().item()):
        raise RuntimeError("Selective plan has no outlier channels.")
    target = _removed_l1(action_tokens, selective)
    leftover_mask = (
        (~selective.mask) & outlier_channels.view(1, -1) & (values > 0.0)
    )
    leftover_l1 = float(values[leftover_mask].sum().item())
    if leftover_l1 <= target:
        raise RuntimeError(
            f"Same-channel leftover L1={leftover_l1:.6g} is not strictly "
            f"larger than outlier removed L1={target:.6g}; cannot match L1 "
            "without zeroing the body."
        )
    body, frac, _target, insufficient, _expanded = _bulk_body_and_frac(
        values,
        selective.bounds,
        selective.mask,
        outlier_channels,
        std_k=1.0,
        seed=0,
    )
    if bool(insufficient.item()) or int(body.sum().item()) < 1:
        raise RuntimeError(
            f"Same-channel leftover never exceeded target L1={target:.6g} "
            "after expanding below the MAD floor."
        )
    if bool((body & selective.mask).any().item()):
        raise RuntimeError("Bulk control overlapped outlier positions.")
    if bool((body & ~outlier_channels.view(1, -1)).any().item()):
        raise RuntimeError("Bulk control touched a non-outlier channel.")
    keep = (1.0 - frac)[body].to(torch.float32)
    if not bool(((keep > 0.0) & (keep < 1.0)).all().item()):
        raise RuntimeError("Bulk keep ratios must lie strictly inside (0, 1).")
    plan = ClipPlan(
        mask=body,
        bounds=torch.full_like(selective.bounds, torch.nan),
        ratios=keep,
    )
    _assert_l1_match(
        _removed_l1(action_tokens, plan),
        target,
        name="same-channel bulk",
    )
    return plan


def _apply_plan(
    flat: torch.Tensor,
    plan: ClipPlan,
    *,
    kind: str,
) -> torch.Tensor:
    if flat.shape != plan.mask.shape:
        raise RuntimeError(
            f"Activation shape {tuple(flat.shape)} != plan mask "
            f"{tuple(plan.mask.shape)}."
        )
    changed = flat.clone()
    mask = plan.mask.to(changed.device)
    if kind == "selective":
        bound = plan.bounds.to(device=changed.device, dtype=changed.dtype)
        expanded = bound.view(1, -1).expand_as(changed)
        changed[mask] = changed[mask].sign() * expanded[mask]
    elif kind == "random":
        ratio = plan.ratios.to(device=changed.device, dtype=changed.dtype)
        changed[mask] = changed[mask] * ratio
    else:
        raise ValueError(f"Unknown intervention kind {kind!r}.")
    return changed


def _run(
    adapter,
    request,
    layer: torch.nn.Module,
    *,
    num_steps: int,
    horizon: int,
    baseline_activations: torch.Tensor | None = None,
    target_step: int | None = None,
    plan: ClipPlan | None = None,
    kind: str | None = None,
    identity_writeback: bool = False,
    restore_plans: dict[int, ClipPlan] | None = None,
    restore_kind: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    capture = baseline_activations is None
    n_flags = (
        int(identity_writeback)
        + int(restore_plans is not None)
        + int(plan is not None)
    )
    if capture:
        if (
            n_flags
            or target_step is not None
            or kind is not None
            or restore_kind is not None
        ):
            raise ValueError("Capture must not set an intervention.")
    elif identity_writeback:
        if (
            n_flags != 1
            or target_step is not None
            or kind is not None
            or restore_kind is not None
        ):
            raise ValueError(
                "Identity write-back must only write captured activations."
            )
    elif restore_plans is not None:
        if (
            n_flags != 1
            or target_step is not None
            or plan is not None
            or kind is not None
        ):
            raise ValueError(
                "Restore write-back needs per-step plans, not a single intervention."
            )
        if restore_kind is None:
            raise ValueError("Restore write-back needs restore_kind.")
        if sorted(restore_plans) != list(range(num_steps)):
            raise ValueError("Restore plans must cover every denoise step.")
    else:
        if plan is None or kind is None or target_step is None:
            raise ValueError("Intervention needs target_step, plan, and kind.")
        if restore_kind is not None:
            raise ValueError("Intervention must not set restore_kind.")
        if not (0 <= int(target_step) < num_steps):
            raise ValueError(
                f"target_step must be in [0, {num_steps}), got {target_step}."
            )
    weight = getattr(layer, "weight", None)
    if not torch.is_tensor(weight) or weight.ndim != 2:
        raise RuntimeError("Target layer must have a 2-D tensor weight.")
    in_features = int(weight.shape[1])
    current_step: list[int | None] = [None]
    callbacks: list[int | None] = []
    captured: dict[int, torch.Tensor] = {}
    applied = 0

    def hook(_module, inputs):
        nonlocal applied
        step = current_step[0]
        if step is None:
            raise RuntimeError(
                "Target linear ran outside the denoise loop; choose a per-step "
                "DiT action linear, not a cached cross-attention K/V linear."
            )
        if len(inputs) != 1 or not torch.is_tensor(inputs[0]):
            raise RuntimeError("Target linear must receive exactly one tensor input.")
        if step in captured:
            raise RuntimeError(f"Target linear ran more than once at step {step}.")
        x = inputs[0]
        if int(x.shape[-1]) != in_features:
            raise RuntimeError(
                f"Input width {x.shape[-1]} != in_features={in_features}."
            )
        flat = x.reshape(-1, in_features)
        if int(flat.shape[0]) != horizon:
            raise RuntimeError(
                f"Expected {horizon} pi0.5 action tokens, got {flat.shape[0]}."
            )
        live = _exact_fp32(flat).clone()
        snapshot = live.cpu()
        _finite(snapshot, f"activation at step {step}")
        captured[step] = snapshot
        if capture:
            return None
        assert baseline_activations is not None
        if identity_writeback:
            if not torch.equal(snapshot, baseline_activations[step]):
                raise RuntimeError(
                    f"Identity forward diverged at step {step} before write-back."
                )
            applied += 1
            replaced = live.to(device=x.device, dtype=x.dtype)
            return (replaced.reshape_as(x),)
        if restore_plans is not None:
            assert restore_kind is not None
            if not torch.equal(snapshot, baseline_activations[step]):
                raise RuntimeError(
                    f"Restore forward diverged at step {step} before clip."
                )
            before = live.clone()
            _apply_plan(live, restore_plans[step], kind=restore_kind)
            if not torch.equal(live, before):
                raise RuntimeError("_apply_plan mutated the activation snapshot.")
            applied += 1
            replaced = live.to(device=x.device, dtype=x.dtype)
            return (replaced.reshape_as(x),)
        assert target_step is not None and plan is not None and kind is not None
        if step <= target_step and not torch.equal(
            snapshot, baseline_activations[step]
        ):
            raise RuntimeError(
                f"Repeated fixed-noise forward diverged before intervention: "
                f"target={target_step}, observed={step}."
            )
        if step != target_step:
            return None
        applied += 1
        clipped = _apply_plan(live, plan, kind=kind)
        return (clipped.to(device=x.device, dtype=x.dtype).reshape_as(x),)

    def step_callback(step: int | None) -> None:
        value = None if step is None else int(step)
        current_step[0] = value
        callbacks.append(value)

    handle = layer.register_forward_pre_hook(hook)
    try:
        runner = find_expert_runner(adapter.engine)
        with patched_one_step(runner, step_callback):
            step_callback(None)
            with torch.inference_mode():
                actions = adapter.engine.step(request)
    finally:
        handle.remove()

    expected = [None, *range(num_steps)]
    if callbacks != expected:
        raise RuntimeError(f"Denoise callback order {callbacks} != {expected}.")
    if sorted(captured) != list(range(num_steps)):
        raise RuntimeError(
            f"Captured steps {sorted(captured)} != {list(range(num_steps))}."
        )
    if capture:
        expected_applied = 0
    elif identity_writeback or restore_plans is not None:
        expected_applied = num_steps
    else:
        expected_applied = 1
    if applied != expected_applied:
        raise RuntimeError(
            f"Hook applied {applied} times, expected {expected_applied}."
        )
    if not torch.is_tensor(actions):
        raise RuntimeError(f"Engine returned {type(actions)}, expected tensor.")
    actions = actions.detach().to(torch.float32).cpu()
    _finite(actions, "predicted actions")
    activations = (
        torch.stack([captured[step] for step in range(num_steps)])
        if capture
        else None
    )
    return actions, activations


def _mean_std(rows: list[ActionMetrics]) -> tuple[ActionMetrics, ActionMetrics]:
    if not rows:
        raise ValueError("Cannot aggregate an empty metric list.")
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for field in fields(ActionMetrics):
        values = torch.tensor(
            [getattr(row, field.name) for row in rows], dtype=torch.float64
        )
        means[field.name] = float(values.mean().item())
        stds[field.name] = float(values.std(unbiased=False).item())
    return ActionMetrics(**means), ActionMetrics(**stds)


def _removed_l1_pct(action_tokens: torch.Tensor, plan: ClipPlan) -> float:
    values = action_tokens.abs().to(torch.float32)
    before = float(values.sum().item())
    if before <= 0.0:
        raise RuntimeError("Action-token activation L1 norm is non-positive.")
    return 100.0 * _removed_l1(action_tokens, plan) / before


def _experiment(
    adapter,
    request,
    layer: torch.nn.Module,
    *,
    num_steps: int,
    horizon: int,
    action_dim: int,
    std_k: float,
    random_trials: int,
    random_seed: int,
) -> tuple[list[StepResult], dict[str, torch.Tensor]]:
    baseline_actions, baseline_activations = _run(
        adapter,
        request,
        layer,
        num_steps=num_steps,
        horizon=horizon,
    )
    assert baseline_activations is not None
    if baseline_actions.ndim != 3 or baseline_actions.shape[:2] != (1, horizon):
        raise RuntimeError(
            f"Expected actions (1,{horizon},width), got {baseline_actions.shape}."
        )
    if int(baseline_actions.shape[2]) < action_dim:
        raise RuntimeError(
            f"Action output width {baseline_actions.shape[2]} < action_dim={action_dim}."
        )
    baseline = baseline_actions[0, :, :action_dim]
    identity_actions, _ = _run(
        adapter,
        request,
        layer,
        num_steps=num_steps,
        horizon=horizon,
        baseline_activations=baseline_activations,
        identity_writeback=True,
    )
    _assert_actions_equal(
        identity_actions,
        baseline_actions,
        name="identity write-back",
    )
    print("identity write-back: actions unchanged")

    restore_plans = {
        step: _clip_plan(baseline_activations[step], std_k)
        for step in range(num_steps)
    }
    restore_actions, _ = _run(
        adapter,
        request,
        layer,
        num_steps=num_steps,
        horizon=horizon,
        baseline_activations=baseline_activations,
        restore_plans=restore_plans,
        restore_kind="selective",
    )
    _assert_actions_equal(
        restore_actions,
        baseline_actions,
        name="clip-then-restore write-back",
    )
    print("clip-then-restore write-back: actions unchanged")

    selective_actions: list[torch.Tensor] = []
    random_actions: list[torch.Tensor] = []
    results: list[StepResult] = []

    for step in range(num_steps):
        action_tokens = baseline_activations[step]
        plan = restore_plans[step]
        target_l1 = _removed_l1(action_tokens, plan)
        changed, _ = _run(
            adapter,
            request,
            layer,
            num_steps=num_steps,
            horizon=horizon,
            baseline_activations=baseline_activations,
            target_step=step,
            plan=plan,
            kind="selective",
        )
        changed_trimmed = changed[0, :, :action_dim]
        selective_actions.append(changed_trimmed)
        selective_metrics = _action_metrics(changed_trimmed, baseline)

        trial_metrics: list[ActionMetrics] = []
        trial_actions: list[torch.Tensor] = []
        trial_l1_pct: list[float] = []
        trial_counts: list[int] = []
        for trial in range(random_trials):
            control = _same_channel_bulk_plan(
                action_tokens,
                plan,
                seed=random_seed + step * random_trials + trial,
            )
            _assert_l1_match(
                _removed_l1(action_tokens, control),
                target_l1,
                name=f"bulk step={step} trial={trial}",
            )
            random_changed, _ = _run(
                adapter,
                request,
                layer,
                num_steps=num_steps,
                horizon=horizon,
                baseline_activations=baseline_activations,
                target_step=step,
                plan=control,
                kind="random",
            )
            random_trimmed = random_changed[0, :, :action_dim]
            trial_actions.append(random_trimmed)
            trial_metrics.append(_action_metrics(random_trimmed, baseline))
            trial_l1_pct.append(_removed_l1_pct(action_tokens, control))
            trial_counts.append(int(control.mask.sum().item()))
        random_mean, random_std = _mean_std(trial_metrics)
        random_actions.append(torch.stack(trial_actions))
        result = StepResult(
            step=step,
            outlier_count=int(plan.mask.sum().item()),
            selected_channels=int(plan.mask.any(dim=0).sum().item()),
            bulk_count=int(round(_mean([float(n) for n in trial_counts]))),
            removed_l1_pct=_removed_l1_pct(action_tokens, plan),
            bulk_removed_l1_pct=_mean(trial_l1_pct),
            selective=selective_metrics,
            random_mean=random_mean,
            random_std=random_std,
        )
        results.append(result)
        print(
            f"step={step:2d} outliers={result.outlier_count:4d} "
            f"channels={result.selected_channels:3d} "
            f"bulk_values={result.bulk_count:4d} "
            f"removed_L1={result.removed_l1_pct:.4f}% "
            f"bulk_L1={result.bulk_removed_l1_pct:.4f}% "
            f"selective(mean={selective_metrics.arm_mean_shift_rmse:.3e}, "
            f"end={selective_metrics.arm_endpoint_rmse:.3e}, "
            f"disp={selective_metrics.arm_net_disp_rmse:.3e}, "
            f"local={selective_metrics.arm_local_rmse:.3e}, "
            f"step={selective_metrics.arm_step_rmse:.3e}, "
            f"grip={selective_metrics.gripper_rmse:.3e}, "
            f"switch={selective_metrics.gripper_switch_shift:.1f}) "
            f"bulk(mean={random_mean.arm_mean_shift_rmse:.3e}, "
            f"local={random_mean.arm_local_rmse:.3e}, "
            f"grip={random_mean.gripper_rmse:.3e})"
        )

    arrays = {
        "baseline_activations": baseline_activations,
        "baseline_actions": baseline,
        "selective_actions": torch.stack(selective_actions),
        "random_actions": torch.stack(random_actions),
    }
    return results, arrays


def _write_csv(results: list[StepResult], output: Path) -> None:
    metric_names = [field.name for field in fields(ActionMetrics)]
    fieldnames = [
        "step",
        "outlier_count",
        "selected_channels",
        "bulk_count",
        "removed_l1_pct",
        "bulk_removed_l1_pct",
        *[f"selective_{name}" for name in metric_names],
        *[f"random_mean_{name}" for name in metric_names],
        *[f"random_std_{name}" for name in metric_names],
    ]
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {
                "step": result.step,
                "outlier_count": result.outlier_count,
                "selected_channels": result.selected_channels,
                "bulk_count": result.bulk_count,
                "removed_l1_pct": result.removed_l1_pct,
                "bulk_removed_l1_pct": result.bulk_removed_l1_pct,
            }
            for prefix, metrics in (
                ("selective", result.selective),
                ("random_mean", result.random_mean),
                ("random_std", result.random_std),
            ):
                for name in metric_names:
                    row[f"{prefix}_{name}"] = getattr(metrics, name)
            writer.writerow(row)


def _write_long_csv(
    rows: list[tuple[int, int, StepResult]],
    output: Path,
) -> None:
    metric_names = [field.name for field in fields(ActionMetrics)]
    fieldnames = [
        "sample",
        "noise",
        "step",
        "outlier_count",
        "selected_channels",
        "bulk_count",
        "removed_l1_pct",
        "bulk_removed_l1_pct",
        *[f"selective_{name}" for name in metric_names],
        *[f"random_mean_{name}" for name in metric_names],
        *[f"random_std_{name}" for name in metric_names],
    ]
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for sample, noise, result in rows:
            row = {
                "sample": sample,
                "noise": noise,
                "step": result.step,
                "outlier_count": result.outlier_count,
                "selected_channels": result.selected_channels,
                "bulk_count": result.bulk_count,
                "removed_l1_pct": result.removed_l1_pct,
                "bulk_removed_l1_pct": result.bulk_removed_l1_pct,
            }
            for prefix, metrics in (
                ("selective", result.selective),
                ("random_mean", result.random_mean),
                ("random_std", result.random_std),
            ):
                for name in metric_names:
                    row[f"{prefix}_{name}"] = getattr(metrics, name)
            writer.writerow(row)


def _mean_step_results(groups: list[list[StepResult]]) -> list[StepResult]:
    if not groups:
        raise ValueError("Need at least one condition to average.")
    n_steps = len(groups[0])
    if n_steps < 1:
        raise ValueError("Each condition must contain at least one step.")
    if any(len(group) != n_steps for group in groups):
        raise RuntimeError("Condition step counts do not match.")
    averaged: list[StepResult] = []
    for index in range(n_steps):
        rows = [group[index] for group in groups]
        steps = {row.step for row in rows}
        if len(steps) != 1:
            raise RuntimeError(f"Step order mismatch at index {index}: {sorted(steps)}.")
        selective_mean, _selective_std = _mean_std([row.selective for row in rows])
        random_mean, random_std = _mean_std([row.random_mean for row in rows])
        averaged.append(
            StepResult(
                step=rows[0].step,
                outlier_count=int(
                    round(_mean([float(row.outlier_count) for row in rows]))
                ),
                selected_channels=int(
                    round(_mean([float(row.selected_channels) for row in rows]))
                ),
                bulk_count=int(round(_mean([float(row.bulk_count) for row in rows]))),
                removed_l1_pct=_mean([row.removed_l1_pct for row in rows]),
                bulk_removed_l1_pct=_mean([row.bulk_removed_l1_pct for row in rows]),
                selective=selective_mean,
                random_mean=random_mean,
                random_std=random_std,
            )
        )
    return averaged


def _verdict(rows: list[tuple[int, int, StepResult]]) -> str:
    if not rows:
        raise ValueError("Cannot judge an empty result list.")
    sel_mean = [row.selective.arm_mean_shift_rmse for _, _, row in rows]
    sel_local = [row.selective.arm_local_rmse for _, _, row in rows]
    bulk_mean = [row.random_mean.arm_mean_shift_rmse for _, _, row in rows]
    bulk_local = [row.random_mean.arm_local_rmse for _, _, row in rows]
    sel_mean_avg = _mean(sel_mean)
    sel_local_avg = _mean(sel_local)
    bulk_mean_avg = _mean(bulk_mean)
    bulk_local_avg = _mean(bulk_local)
    if bulk_mean_avg <= 0.0:
        raise RuntimeError("Bulk mean-shift is non-positive; cannot form a ratio.")
    mean_ratio = sel_mean_avg / bulk_mean_avg
    local_ratio = (
        sel_local_avg / bulk_local_avg if bulk_local_avg > 0.0 else float("inf")
    )
    print(
        "\n=== hypothesis summary "
        f"(n={len(rows)} sample×noise×step) ==="
    )
    print(
        f"outlier  arm_mean_shift={sel_mean_avg:.4e}  "
        f"arm_local={sel_local_avg:.4e}"
    )
    print(
        f"bulk     arm_mean_shift={bulk_mean_avg:.4e}  "
        f"arm_local={bulk_local_avg:.4e}"
    )
    print(
        f"ratio outlier/bulk  mean_shift={mean_ratio:.3f}  "
        f"local={local_ratio:.3f}"
    )
    # Holds: outliers barely move the coarse path, local is small, bulk moves it.
    if mean_ratio < 0.25 and sel_mean_avg < bulk_mean_avg and local_ratio < 0.5:
        verdict = (
            "HOLDS: clipping outliers barely moves the coarse path and local "
            "detail stays smaller than the matched bulk control, while shrinking "
            "ordinary non-outlier values by the same L1 shifts the path."
        )
    elif mean_ratio > 0.5 or sel_mean_avg >= bulk_mean_avg:
        verdict = (
            "FAILS: clipping outliers also shifts the coarse path, or the "
            "outlier and bulk interventions are comparable. The spikes are not "
            "ignorable detail."
        )
    else:
        verdict = (
            "MIXED: outliers move the coarse path less than the matched bulk "
            "control, but not enough to call them negligible detail."
        )
    print(verdict)
    return verdict


def _plot(
    results: list[StepResult],
    arrays: dict[str, torch.Tensor],
    *,
    layer_name: str,
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    steps = np.array([row.step for row in results])
    selective = [row.selective for row in results]
    random = [row.random_mean for row in results]

    def vals(rows: list[ActionMetrics], name: str) -> np.ndarray:
        return np.array([getattr(row, name) for row in rows])

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    ax = axes[0, 0]
    for name, label in (
        ("arm_mean_shift_rmse", "mean shift"),
        ("arm_endpoint_rmse", "endpoint"),
        ("arm_net_disp_rmse", "start-to-end"),
    ):
        ax.plot(steps, vals(selective, name), "o-", label=f"outlier {label}")
        ax.plot(steps, vals(random, name), "--", label=f"same-ch bulk {label}")
    ax.set_title("Overall arm change")
    ax.set_ylabel("RMSE")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[0, 1]
    ax.plot(steps, vals(selective, "arm_local_rmse"), "o-", label="outlier local")
    ax.plot(steps, vals(selective, "arm_step_rmse"), "o-", label="outlier step")
    ax.plot(steps, vals(random, "arm_local_rmse"), "--", label="same-ch bulk local")
    ax.plot(steps, vals(random, "arm_step_rmse"), "--", label="same-ch bulk step")
    ax.set_title("Local arm change")
    ax.set_ylabel("RMSE")
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    ax.plot(steps, vals(selective, "gripper_rmse"), "o-", label="outlier gripper")
    ax.plot(steps, vals(random, "gripper_rmse"), "--", label="same-ch bulk gripper")
    ax.set_ylabel("gripper RMSE")
    ax2 = ax.twinx()
    ax2.plot(
        steps,
        vals(selective, "gripper_switch_shift"),
        "o-",
        color="#F58518",
        label="outlier switch",
    )
    ax2.plot(
        steps,
        vals(random, "gripper_switch_shift"),
        "--",
        color="#F58518",
        label="same-ch bulk switch",
    )
    ax2.set_ylabel("gripper switch shift (steps)")
    ax.set_title("Gripper change")
    lines = ax.get_legend_handles_labels()
    lines2 = ax2.get_legend_handles_labels()
    ax.legend(lines[0] + lines2[0], lines[1] + lines2[1], fontsize=8)

    ax = axes[1, 0]
    plotted = 0
    for name, label in (
        ("arm_mean_shift_rmse", "mean"),
        ("arm_endpoint_rmse", "end"),
        ("arm_local_rmse", "local"),
        ("arm_step_rmse", "step"),
        ("gripper_rmse", "gripper"),
    ):
        sel = vals(selective, name)
        rnd = vals(random, name)
        if np.any(rnd <= 0.0):
            print(f"skip ratio for {name}: same-channel bulk control is non-positive.")
            continue
        ax.plot(steps, sel / rnd, "o-", label=label)
        plotted += 1
    if plotted < 1:
        raise RuntimeError("No positive same-channel bulk metrics for ratio panel.")
    ax.axhline(1.0, color="black", linewidth=1.0)
    ax.set_title("Outlier / same-channel bulk ratio")
    ax.set_xlabel("intervened denoise step")
    ax.set_ylabel("ratio")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.bar(
        steps - 0.18,
        [row.outlier_count for row in results],
        width=0.36,
        label="outlier values",
    )
    ax2 = ax.twinx()
    ax2.bar(
        steps + 0.18,
        [row.removed_l1_pct for row in results],
        width=0.36,
        color="#F58518",
        label="removed L1 %",
    )
    ax.set_title("Intervention size (matched L1)")
    ax.set_xlabel("intervened denoise step")
    ax.set_ylabel("outlier values")
    ax2.set_ylabel("removed activation L1 (%)")
    lines = ax.get_legend_handles_labels()
    lines2 = ax2.get_legend_handles_labels()
    ax.legend(lines[0] + lines2[0], lines[1] + lines2[1], fontsize=8)

    baseline = arrays["baseline_actions"]
    selective_actions = arrays["selective_actions"]
    dim_rmse = (selective_actions - baseline.unsqueeze(0)).square().mean(1).sqrt()
    image = axes[1, 2].imshow(
        dim_rmse.numpy(),
        aspect="auto",
        origin="lower",
        cmap="magma",
        interpolation="nearest",
    )
    axes[1, 2].set_title("Change by action dim (0-5 arm, 6 gripper)")
    axes[1, 2].set_xlabel("action dimension")
    axes[1, 2].set_ylabel("intervened denoise step")
    fig.colorbar(image, ax=axes[1, 2], label="delta RMSE")

    for ax in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]):
        ax.grid(alpha=0.25)
        ax.set_xticks(steps)
    fig.suptitle(
        f"{layer_name}\n"
        "outlier clamp vs same-channel matched-L1 bulk; "
        "overall=mean/endpoint/start-to-end; "
        "detail=local residual/step motion/gripper",
        fontsize=11,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Failed to write {output}.")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-data", type=Path, required=True)
    parser.add_argument("--layer-regex", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--samples",
        default=None,
        help="Comma-separated sample indices. Default: --sample-index.",
    )
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument(
        "--noise-seeds",
        default=None,
        help="Comma-separated noise seeds. Default: --noise-seed.",
    )
    parser.add_argument("--random-seed", type=int, default=1000)
    parser.add_argument("--random-trials", type=int, default=4)
    parser.add_argument("--outlier-std-k", type=float, default=3.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--params-dtype", default="bfloat16")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tools/img/dit_outlier_action_detail_matched_l1"),
    )
    return parser


def _fixed_noise_request(
    adapter,
    batch,
    *,
    noise_seed: int,
    horizon: int,
    width: int,
    scheduler,
):
    adapter._ensure_processor()
    request = build_pi05_request(
        adapter._processor,
        batch,
        state_dim=adapter.cfg.state_dim,
    )
    generator = torch.Generator(device=scheduler.device)
    generator.manual_seed(int(noise_seed))
    noise = torch.randn(
        1,
        horizon,
        width,
        generator=generator,
        device=scheduler.device,
        dtype=scheduler.params_dtype,
    )
    return replace(request, noise=noise)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(args.checkpoint)
    if not args.calibration_data.is_file():
        raise FileNotFoundError(args.calibration_data)
    if args.sample_index < 0:
        raise ValueError("--sample-index must be >= 0.")
    if args.noise_seed < 0:
        raise ValueError("--noise-seed must be >= 0.")
    if args.random_trials < 1:
        raise ValueError("--random-trials must be >= 1.")
    if args.outlier_std_k <= 0.0:
        raise ValueError("--outlier-std-k must be > 0.")
    sample_ids = (
        _parse_nonneg_ints(args.samples, name="--samples")
        if args.samples is not None
        else [int(args.sample_index)]
    )
    noise_ids = (
        _parse_nonneg_ints(args.noise_seeds, name="--noise-seeds")
        if args.noise_seeds is not None
        else [int(args.noise_seed)]
    )

    adapter = get_adapter(
        "pi05",
        checkpoint_path=args.checkpoint,
        calibration_source="file",
        calibration_data_path=args.calibration_data,
        device=args.device,
        params_dtype=args.params_dtype,
    )
    model = adapter.build_model()
    model.eval()
    adapter.warmup_for_calibration(model)
    layer_name, layer = _one_dit_layer(model, args.layer_regex)
    config = QVLAConfig.pi05_default()
    num_steps = adapter.dit_step_count(config)
    scheduler = adapter.engine.entry.scheduler
    horizon = int(scheduler.cfg.chunk_size)
    width = int(scheduler.cfg.max_action_dim)
    action_dim = int(adapter.cfg.action_dim)
    if action_dim < 2:
        raise RuntimeError(f"Need arm+gripper action_dim>=2, got {action_dim}.")

    need = max(sample_ids) + 1
    batches = list(adapter.iter_calibration_batches(need))
    if len(batches) != need:
        raise RuntimeError(
            f"Requested samples {sample_ids}, calibration yielded "
            f"{len(batches)} samples."
        )
    conditions = [(sample, seed) for sample in sample_ids for seed in noise_ids]
    print(
        f"layer={layer_name}, samples={sample_ids}, noise_seeds={noise_ids}, "
        f"n_conditions={len(conditions)}, steps={num_steps}, "
        f"horizon={horizon}, action_dim={action_dim}, "
        f"random_trials={args.random_trials}, std_k={args.outlier_std_k}"
    )

    grouped: list[list[StepResult]] = []
    long_rows: list[tuple[int, int, StepResult]] = []
    last_arrays: dict[str, torch.Tensor] | None = None
    for index, (sample, seed) in enumerate(conditions, start=1):
        request = _fixed_noise_request(
            adapter,
            batches[sample],
            noise_seed=seed,
            horizon=horizon,
            width=width,
            scheduler=scheduler,
        )
        print(
            f"\n=== condition {index}/{len(conditions)} "
            f"sample={sample} noise={seed} ==="
        )
        results, arrays = _experiment(
            adapter,
            request,
            layer,
            num_steps=num_steps,
            horizon=horizon,
            action_dim=action_dim,
            std_k=args.outlier_std_k,
            random_trials=args.random_trials,
            random_seed=args.random_seed,
        )
        grouped.append(results)
        long_rows.extend((sample, seed, row) for row in results)
        last_arrays = arrays

    if last_arrays is None:
        raise RuntimeError("No conditions were evaluated.")
    mean_results = _mean_step_results(grouped)
    verdict = _verdict(long_rows)

    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", layer_name).strip("_")
    if not stem:
        raise RuntimeError(f"Cannot make filename from {layer_name!r}.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    png = args.output_dir / f"{stem}_action_detail.png"
    csv_path = args.output_dir / f"{stem}_action_detail.csv"
    long_path = args.output_dir / f"{stem}_action_detail_per_condition.csv"
    npz = args.output_dir / f"{stem}_action_detail_last_condition.npz"
    verdict_path = args.output_dir / f"{stem}_hypothesis.txt"
    _plot(
        mean_results,
        last_arrays,
        layer_name=(
            f"{layer_name} (mean over {len(sample_ids)} sample(s) × "
            f"{len(noise_ids)} noise seed(s))"
        ),
        output=png,
    )
    _write_csv(mean_results, csv_path)
    _write_long_csv(long_rows, long_path)
    np.savez_compressed(
        npz,
        **{name: value.numpy() for name, value in last_arrays.items()},
    )
    verdict_path.write_text(verdict + "\n", encoding="utf-8")
    for path in (png, csv_path, long_path, npz, verdict_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Failed to write {path}.")
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
