#!/usr/bin/env python
r"""Clip every per-step DiT linear, once per denoise step, then test the outlier hypothesis.

Each trial still clips **all** per-step DiT linears together (prefix-only K/V
linears are skipped). The denoise loop is tested separately: one forward
intervenes at a single step, the others pass through. Repeat for every step.

On a fixed sample and noise the three actions are:

1. original model
2. outliers only: MAD channels, clamp values above μ+kσ
3. matched bulk control: leftover in-body values on the same MAD-selected
   channels, in [median+3×1.4826×MAD, μ+3σ], one shared scale, same removed L1.
   If the band is too small, recruit the largest leftover below the MAD floor
   until L1 can match, then scale the combined set.

Plans are computed on the live activation at the clipped step. Identity
write-back and clip-then-restore must leave actions bit-identical to the
no-hook baseline. Several sample/noise pairs are required.

GR00T-N1.7 DiT tokens are ``cat(state, action_chunk)``. That run never
clips row 0 (state); MAD / μ+kσ use only the action tokens. pi0.5 is
unchanged (every token is an action step).

Example:
    CUDA_VISIBLE_DEVICES=5 uv run python \
      tools/analyze_dit_outlier_action_detail_all.py \
      --checkpoint /data/share/pi05_libero_finetuned_v044 \
      --calibration-data ../calibration_data/libero_goal_30_7_demo.npz \
      --samples 0,1,2,3 --noise-seeds 0,1 --random-trials 4 \
      --output-dir tools/img/dit_outlier_vs_samech_mad_band_expand_all_by_step

    CUDA_VISIBLE_DEVICES=5 uv run python \
      tools/analyze_dit_outlier_action_detail_all.py \
      --model groot_n17 \
      --checkpoint /data/share/GR00T-N1.7-LIBERO/libero_goal \
      --calibration-data ../calibration_data/libero_goal_30_7_demo.npz \
      --samples 0,1,2,3 --noise-seeds 0,1 --random-trials 4 \
      --output-dir tools/img/dit_outlier_vs_samech_mad_band_expand_all_by_step_groot
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
_TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_TOOLS))

import analyze_dit_matched_clip_action_impact as matched  # noqa: E402
import analyze_dit_outlier_action_detail as detail  # noqa: E402
from qvla.config import QVLAConfig  # noqa: E402


@dataclass(frozen=True)
class SiteStats:
    n_sites: int
    n_clipped: int
    n_empty: int
    n_expanded: int
    outlier_count: int
    bulk_count: int
    selected_channels: int
    mean_removed_l1_pct: float
    mean_bulk_removed_l1_pct: float
    n_ch_target: int = 0
    n_ch_expand_enough: int = 0
    n_ch_insufficient: int = 0


@dataclass(frozen=True)
class ConditionResult:
    sample: int
    noise: int
    step: int
    n_layers: int
    n_steps: int
    selective_stats: SiteStats
    bulk_stats: SiteStats
    selective: detail.ActionMetrics
    random_mean: detail.ActionMetrics
    random_std: detail.ActionMetrics


def _expected_applied(
    num_steps: int,
    *,
    capture: bool,
    identity_writeback: bool,
    restore_after_clip: bool,
    target_step: int | None,
) -> int:
    if capture:
        return 0
    if identity_writeback or restore_after_clip:
        return int(num_steps)
    if target_step is None:
        return int(num_steps)
    return 1


def _empty_site_stats() -> SiteStats:
    return SiteStats(
        n_sites=0,
        n_clipped=0,
        n_empty=0,
        n_expanded=0,
        outlier_count=0,
        bulk_count=0,
        selected_channels=0,
        mean_removed_l1_pct=0.0,
        mean_bulk_removed_l1_pct=0.0,
        n_ch_target=0,
        n_ch_expand_enough=0,
        n_ch_insufficient=0,
    )


def _valid_action_horizon(final_action: torch.Tensor, *, atol: float = 1e-5) -> int:
    if final_action.ndim != 2 or int(final_action.shape[0]) < 2:
        raise ValueError(
            f"final_action must have shape (horizon>=2, dims), got {tuple(final_action.shape)}."
        )
    mag = final_action.abs().amax(dim=1)
    nonzero = torch.nonzero(mag > atol, as_tuple=False).flatten()
    if int(nonzero.numel()) == 0:
        return int(final_action.shape[0])
    return int(nonzero[-1].item()) + 1


def _run_all(
    adapter,
    request,
    layers: list[tuple[str, torch.nn.Module]],
    *,
    num_steps: int,
    n_tokens: int,
    std_k: float,
    baseline_activations: dict[str, torch.Tensor] | None = None,
    identity_writeback: bool = False,
    restore_after_clip: bool = False,
    kind: str | None = None,
    bulk_seed: int | None = None,
    layer_index: dict[str, int] | None = None,
    target_step: int | None = None,
    skip_first_token: bool = False,
    per_channel_l1: bool = False,
    complement_l1: bool = False,
    complement_amax_l1: bool = False,
    leftover_pool_l1: bool = False,
    normal_ref_l1: bool = False,
    normal_ref_mode: str = "unselected_std",
    normal_full_clip: bool = False,
    allow_empty: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None, SiteStats]:
    capture = baseline_activations is None
    n_flags = int(identity_writeback) + int(restore_after_clip) + int(kind is not None)
    l1_modes = (
        int(per_channel_l1)
        + int(complement_l1)
        + int(complement_amax_l1)
        + int(leftover_pool_l1)
        + int(normal_ref_l1)
    )
    if l1_modes > 1:
        raise ValueError(
            "Choose at most one of per_channel_l1, complement_l1, "
            "complement_amax_l1, leftover_pool_l1, normal_ref_l1."
        )
    extra_l1 = l1_modes > 0
    if normal_full_clip and not normal_ref_l1:
        raise ValueError("normal_full_clip requires normal_ref_l1.")
    if target_step is not None and not (0 <= int(target_step) < num_steps):
        raise ValueError(
            f"target_step must be in [0, {num_steps}), got {target_step}."
        )
    if capture:
        if n_flags or bulk_seed is not None or target_step is not None or extra_l1:
            raise ValueError("Capture must not set an intervention.")
    elif identity_writeback:
        if (
            n_flags != 1
            or kind is not None
            or bulk_seed is not None
            or target_step is not None
            or extra_l1
        ):
            raise ValueError("Identity write-back must only write captured activations.")
    elif restore_after_clip:
        if (
            n_flags != 1
            or kind is not None
            or bulk_seed is not None
            or target_step is not None
            or extra_l1
        ):
            raise ValueError("Restore write-back must clip then write original values.")
    else:
        # target_step=None clips every denoise step on the live activation.
        if kind not in {"selective", "random"}:
            raise ValueError(f"Intervention kind must be selective or random, got {kind!r}.")
        if kind == "random" and bulk_seed is None:
            raise ValueError("Bulk intervention needs bulk_seed.")
        if kind == "selective" and bulk_seed is not None:
            raise ValueError("Outlier intervention must not set bulk_seed.")
        if layer_index is None:
            raise ValueError("Intervention needs layer_index.")

    in_features = {}
    for name, layer in layers:
        weight = getattr(layer, "weight", None)
        if not torch.is_tensor(weight) or weight.ndim != 2:
            raise RuntimeError(f"{name} must have a 2-D tensor weight.")
        in_features[name] = int(weight.shape[1])
    device = layers[0][1].weight.device

    current_step: list[int | None] = [None]
    callbacks: list[int | None] = []
    captured: dict[str, dict[int, torch.Tensor]] = {name: {} for name, _ in layers}
    seen: dict[str, set[int]] = {name: set() for name, _ in layers}
    applied = {name: 0 for name, _ in layers}
    n_sites_t = torch.zeros((), dtype=torch.int64, device=device)
    n_clipped_t = torch.zeros((), dtype=torch.int64, device=device)
    outlier_count_t = torch.zeros((), dtype=torch.int64, device=device)
    selected_channels_t = torch.zeros((), dtype=torch.int64, device=device)
    bulk_count_t = torch.zeros((), dtype=torch.int64, device=device)
    removed_pct_sum = torch.zeros((), dtype=torch.float64, device=device)
    bulk_pct_sum = torch.zeros((), dtype=torch.float64, device=device)
    l1_rel_err_max = torch.zeros((), dtype=torch.float64, device=device)
    n_expanded_t = torch.zeros((), dtype=torch.int64, device=device)
    n_leftover_short_t = torch.zeros((), dtype=torch.int64, device=device)
    n_ch_target_t = torch.zeros((), dtype=torch.int64, device=device)
    n_ch_expand_enough_t = torch.zeros((), dtype=torch.int64, device=device)
    n_ch_insufficient_t = torch.zeros((), dtype=torch.int64, device=device)
    diverged = torch.zeros((), dtype=torch.bool, device=device)
    tiny = torch.tensor(torch.finfo(torch.float32).tiny, device=device, dtype=torch.float32)

    def make_hook(name: str):
        width = in_features[name]

        def hook(_module, inputs):
            step = current_step[0]
            if step is None:
                raise RuntimeError(
                    f"{name} ran outside the denoise loop; prefix-only K/V "
                    "linears must be skipped."
                )
            if len(inputs) != 1 or not torch.is_tensor(inputs[0]):
                raise RuntimeError(f"{name} must receive exactly one tensor input.")
            if step in seen[name]:
                raise RuntimeError(f"{name} ran more than once at step {step}.")
            seen[name].add(int(step))
            x = inputs[0]
            if int(x.shape[-1]) != width:
                raise RuntimeError(
                    f"{name} input width {x.shape[-1]} != in_features={width}."
                )
            flat = x.reshape(-1, width)
            if int(flat.shape[0]) != n_tokens:
                raise RuntimeError(
                    f"{name}: expected {n_tokens} DiT tokens, got {flat.shape[0]}."
                )
            live = detail._exact_fp32(flat)
            if capture:
                captured[name][step] = live.detach().clone()
                return None
            assert baseline_activations is not None
            if identity_writeback or restore_after_clip:
                expected = baseline_activations[name][step]
                if expected.device != live.device or expected.dtype != live.dtype:
                    expected = expected.to(device=live.device, dtype=live.dtype)
                diverged.logical_or_((live != expected).any())
                if restore_after_clip:
                    _values, _selected, bounds, over = detail._outlier_over_and_bounds(
                        live, std_k
                    )
                    _ = detail._apply_outlier(live, bounds, over)
                applied[name] += 1
                replaced = live.to(device=x.device, dtype=x.dtype)
                return (replaced.reshape_as(x),)

            assert kind is not None and layer_index is not None
            if not matched._clip_this_step(int(step), target_step):
                if int(step) < int(target_step):
                    expected = baseline_activations[name][step]
                    if expected.device != live.device or expected.dtype != live.dtype:
                        expected = expected.to(device=live.device, dtype=live.dtype)
                    diverged.logical_or_((live != expected).any())
                return None

            values, selected, bounds, over = detail._outlier_over_and_bounds(
                live, std_k, skip_first_token=skip_first_token
            )
            n_sites_t.add_(1)
            site_l1 = values.sum().clamp(min=tiny)
            if per_channel_l1:
                plan = detail._per_channel_body_and_frac(
                    values,
                    bounds,
                    over,
                    selected,
                    skip_first_token=skip_first_token,
                )
                n_ch_target_t.add_(plan.n_target)
                n_ch_expand_enough_t.add_(plan.n_expand_enough)
                n_ch_insufficient_t.add_(plan.n_insufficient)
                n_expanded_t.add_(plan.n_expand_enough.clamp(max=1))
                has_clip = plan.matchable.any()
                n_clipped_t.add_(has_clip.to(torch.int64))
                outlier_count_t.add_(plan.over_use.sum())
                selected_channels_t.add_(plan.matchable.sum())
                removed = detail._removed_l1_from_over(values, bounds, plan.over_use)
                removed_pct_sum.add_((100.0 * removed / site_l1).to(torch.float64))
                if kind == "selective":
                    clipped = detail._apply_outlier(live, bounds, plan.over_use)
                else:
                    bulk_removed = (values * plan.body.to(values.dtype) * plan.frac).sum()
                    bulk_count_t.add_(plan.body.sum())
                    bulk_pct_sum.add_((100.0 * bulk_removed / site_l1).to(torch.float64))
                    l1_rel_err_max.copy_(
                        torch.maximum(l1_rel_err_max, plan.max_rel.to(torch.float64))
                    )
                    clipped = detail._apply_bulk(live, plan.body, plan.frac)
            elif complement_l1 or leftover_pool_l1:
                plan_fn = (
                    detail._leftover_pool_plan
                    if leftover_pool_l1
                    else detail._complement_channel_plan
                )
                plan = plan_fn(
                    values,
                    selected,
                    std_k=std_k,
                    skip_first_token=skip_first_token,
                )
                n_ch_target_t.add_(plan.n_target)
                n_ch_expand_enough_t.add_(plan.n_expand_enough)
                n_ch_insufficient_t.add_(plan.n_insufficient)
                n_expanded_t.add_(plan.n_expand_enough)
                has_clip = plan.matchable
                n_clipped_t.add_(has_clip.to(torch.int64))
                outlier_count_t.add_(plan.over_use.sum())
                selected_channels_t.add_(selected.sum())
                removed = detail._removed_l1_from_over(values, plan.bounds, plan.over_use)
                removed_pct_sum.add_((100.0 * removed / site_l1).to(torch.float64))
                if kind == "selective":
                    clipped = detail._apply_outlier(live, plan.bounds, plan.over_use)
                else:
                    bulk_removed = (values * plan.body.to(values.dtype) * plan.frac).sum()
                    bulk_count_t.add_(plan.body.sum())
                    bulk_pct_sum.add_((100.0 * bulk_removed / site_l1).to(torch.float64))
                    l1_rel_err_max.copy_(
                        torch.maximum(l1_rel_err_max, plan.max_rel.to(torch.float64))
                    )
                    clipped = detail._apply_bulk(live, plan.body, plan.frac)
            elif normal_ref_l1:
                plan_fn = (
                    detail._normal_outlier_full_plan
                    if normal_full_clip
                    else detail._normal_outlier_match_plan
                )
                plan = plan_fn(
                    live,
                    std_k=std_k,
                    skip_first_token=skip_first_token,
                    ref_mode=normal_ref_mode,
                )
                n_ch_target_t.add_(plan.n_target)
                n_ch_insufficient_t.add_(plan.n_insufficient)
                if normal_full_clip:
                    has_clip = (
                        plan.over_use.any() if kind == "selective" else plan.normal.any()
                    )
                else:
                    has_clip = plan.matchable
                n_clipped_t.add_(has_clip.to(torch.int64))
                outlier_count_t.add_(plan.over_use.sum())
                selected_channels_t.add_(plan.normal.any(dim=0).sum())
                removed = detail._removed_l1_from_over(values, plan.bounds, plan.over_use)
                removed_pct_sum.add_((100.0 * removed / site_l1).to(torch.float64))
                if kind == "selective":
                    clipped = detail._apply_outlier(live, plan.bounds, plan.over_use)
                else:
                    bulk_removed = (
                        (values - detail._bound_view(plan.t_ref)).clamp(min=0.0)
                        * plan.normal.to(values.dtype)
                        * plan.frac
                    ).sum()
                    bulk_count_t.add_(plan.normal.sum())
                    bulk_pct_sum.add_((100.0 * bulk_removed / site_l1).to(torch.float64))
                    l1_rel_err_max.copy_(
                        torch.maximum(l1_rel_err_max, plan.max_rel.to(torch.float64))
                    )
                    clipped = detail._apply_ref_excess_frac(
                        live, plan.t_ref, plan.normal, plan.frac
                    )
            elif complement_amax_l1:
                plan = detail._complement_amax_plan(
                    values,
                    selected,
                    std_k=std_k,
                    skip_first_token=skip_first_token,
                )
                n_ch_target_t.add_(plan.n_target)
                n_ch_expand_enough_t.add_(plan.n_full)
                n_ch_insufficient_t.add_(plan.n_insufficient)
                n_expanded_t.add_((plan.n_full > 0).to(torch.int64))
                has_clip = plan.matchable
                n_clipped_t.add_(has_clip.to(torch.int64))
                outlier_count_t.add_(plan.over_use.sum())
                selected_channels_t.add_((plan.channel_frac > 0).sum())
                removed = detail._removed_l1_from_over(values, plan.bounds, plan.over_use)
                removed_pct_sum.add_((100.0 * removed / site_l1).to(torch.float64))
                if kind == "selective":
                    clipped = detail._apply_outlier(live, plan.bounds, plan.over_use)
                else:
                    bulk_removed = (
                        (values - plan.bounds.view(1, -1)).clamp(min=0.0)
                        * plan.over_rem.to(values.dtype)
                        * plan.channel_frac.view(1, -1)
                    ).sum()
                    bulk_count_t.add_(((plan.channel_frac > 0).view(1, -1) & plan.over_rem).sum())
                    bulk_pct_sum.add_((100.0 * bulk_removed / site_l1).to(torch.float64))
                    l1_rel_err_max.copy_(
                        torch.maximum(l1_rel_err_max, plan.max_rel.to(torch.float64))
                    )
                    clipped = detail._apply_channel_excess_frac(
                        live, plan.bounds, plan.over_rem, plan.channel_frac
                    )
            else:
                has_clip = over.any()
                n_clipped_t.add_(has_clip.to(torch.int64))
                outlier_count_t.add_(over.sum())
                selected_channels_t.add_(over.any(dim=0).sum())
                removed = detail._removed_l1_from_over(values, bounds, over)
                removed_pct_sum.add_((100.0 * removed / site_l1).to(torch.float64))
                if kind == "selective":
                    clipped = detail._apply_outlier(live, bounds, over)
                else:
                    assert bulk_seed is not None
                    body, frac, target, insufficient, expanded = detail._bulk_body_and_frac(
                        values,
                        bounds,
                        over,
                        selected,
                        std_k=std_k,
                        seed=int(bulk_seed) + int(layer_index[name]) * 97 + int(step),
                        skip_first_token=skip_first_token,
                    )
                    n_expanded_t.add_(expanded.to(torch.int64))
                    n_leftover_short_t.add_(insufficient.to(torch.int64))
                    bulk_count_t.add_(body.sum())
                    bulk_removed = (values * body.to(values.dtype) * frac).sum()
                    bulk_pct_sum.add_((100.0 * bulk_removed / site_l1).to(torch.float64))
                    rel = (bulk_removed - target).abs() / target.clamp(min=tiny)
                    l1_rel_err_max.copy_(
                        torch.maximum(l1_rel_err_max, rel.to(torch.float64))
                    )
                    clipped = detail._apply_bulk(live, body, frac)
            if skip_first_token and not torch.equal(clipped[0], live[0]):
                raise RuntimeError(
                    f"{name} changed the GR00T state token (row 0) at step {step}."
                )
            applied[name] += 1
            return (clipped.to(device=x.device, dtype=x.dtype).reshape_as(x),)

        return hook

    def step_callback(step: int | None) -> None:
        value = None if step is None else int(step)
        current_step[0] = value
        callbacks.append(value)

    handles = [
        layer.register_forward_pre_hook(make_hook(name))
        for name, layer in layers
    ]
    try:
        with matched._with_denoise_callback(adapter, step_callback):
            step_callback(None)
            with torch.inference_mode():
                actions = adapter.engine.step(request)
    finally:
        for handle in handles:
            handle.remove()

    expected = [None, *range(num_steps)]
    if callbacks != expected:
        raise RuntimeError(f"Denoise callback order {callbacks} != {expected}.")
    expected_applied = _expected_applied(
        num_steps,
        capture=capture,
        identity_writeback=identity_writeback,
        restore_after_clip=restore_after_clip,
        target_step=target_step,
    )
    for name, _layer in layers:
        if capture:
            if sorted(captured[name]) != list(range(num_steps)):
                raise RuntimeError(
                    f"{name} captured steps {sorted(captured[name])} != "
                    f"{list(range(num_steps))}."
                )
        elif seen[name] != set(range(num_steps)):
            raise RuntimeError(
                f"{name} saw steps {sorted(seen[name])} != {list(range(num_steps))}."
            )
        if applied[name] != expected_applied:
            raise RuntimeError(
                f"{name} hook applied {applied[name]} times, expected {expected_applied}."
            )
    if bool(diverged.item()):
        if identity_writeback or restore_after_clip:
            label = (
                "identity write-back"
                if identity_writeback
                else "clip-then-restore write-back"
            )
            raise RuntimeError(
                f"{label} diverged from the captured baseline before writing "
                "values back. Measurements would still include hook noise."
            )
        raise RuntimeError(
            f"Activations diverged before clip step {target_step}; "
            "later-step measurements would mix earlier hook noise."
        )
    if not torch.is_tensor(actions):
        raise RuntimeError(f"Engine returned {type(actions)}, expected tensor.")
    actions = actions.detach().to(torch.float32).cpu()
    detail._finite(actions, "predicted actions")
    activations = None
    if capture:
        activations = {}
        for name, _layer in layers:
            activations[name] = torch.stack(
                [captured[name][step] for step in range(num_steps)]
            )
    if capture or kind is None:
        return actions, activations, _empty_site_stats()
    n_clipped = int(n_clipped_t.item())
    n_sites = int(n_sites_t.item())
    n_expanded = int(n_expanded_t.item())
    n_leftover_short = int(n_leftover_short_t.item())
    expected_sites = (
        len(layers) if target_step is not None else len(layers) * int(num_steps)
    )
    step_label = (
        f"at step {target_step}" if target_step is not None else "across all denoise steps"
    )
    if n_sites != expected_sites:
        raise RuntimeError(
            f"Clipped {n_sites} linears {step_label}, expected {expected_sites}."
        )
    if n_clipped < 1 and not allow_empty:
        raise RuntimeError(
            f"No DiT linear produced an outlier clip {step_label}."
        )
    if n_leftover_short > 0 and not extra_l1:
        raise RuntimeError(
            f"{n_leftover_short} layer sites {step_label} still cannot "
            "match outlier L1 after expanding leftover below the MAD floor."
        )
    rel_err = float(l1_rel_err_max.item())
    if kind == "random" and not normal_full_clip and rel_err > detail.L1_MATCH_RTOL:
        raise RuntimeError(
            f"Bulk removed L1 missed the outlier target {step_label} "
            f"(max rel={rel_err:.3g})."
        )
    stats = SiteStats(
        n_sites=n_sites,
        n_clipped=n_clipped,
        n_empty=n_sites - n_clipped,
        n_expanded=n_expanded,
        outlier_count=int(outlier_count_t.item()),
        bulk_count=int(bulk_count_t.item()),
        selected_channels=int(selected_channels_t.item()),
        mean_removed_l1_pct=(
            float(removed_pct_sum.item()) / n_clipped if n_clipped else 0.0
        ),
        mean_bulk_removed_l1_pct=(
            float(bulk_pct_sum.item()) / n_clipped
            if kind == "random" and n_clipped
            else 0.0
        ),
        n_ch_target=int(n_ch_target_t.item()),
        n_ch_expand_enough=int(n_ch_expand_enough_t.item()),
        n_ch_insufficient=int(n_ch_insufficient_t.item()),
    )
    return actions, activations, stats


def _mean_site_stats(rows: list[SiteStats]) -> SiteStats:
    if not rows:
        raise ValueError("Cannot average empty site stats.")
    return SiteStats(
        n_sites=int(round(detail._mean([float(row.n_sites) for row in rows]))),
        n_clipped=int(round(detail._mean([float(row.n_clipped) for row in rows]))),
        n_empty=int(round(detail._mean([float(row.n_empty) for row in rows]))),
        n_expanded=int(
            round(detail._mean([float(row.n_expanded) for row in rows]))
        ),
        outlier_count=int(round(detail._mean([float(row.outlier_count) for row in rows]))),
        bulk_count=int(round(detail._mean([float(row.bulk_count) for row in rows]))),
        selected_channels=int(
            round(detail._mean([float(row.selected_channels) for row in rows]))
        ),
        mean_removed_l1_pct=detail._mean([row.mean_removed_l1_pct for row in rows]),
        mean_bulk_removed_l1_pct=detail._mean(
            [row.mean_bulk_removed_l1_pct for row in rows]
        ),
        n_ch_target=int(round(detail._mean([float(row.n_ch_target) for row in rows]))),
        n_ch_expand_enough=int(
            round(detail._mean([float(row.n_ch_expand_enough) for row in rows]))
        ),
        n_ch_insufficient=int(
            round(detail._mean([float(row.n_ch_insufficient) for row in rows]))
        ),
    )


def _experiment(
    adapter,
    request,
    layers: list[tuple[str, torch.nn.Module]],
    *,
    runtime: matched.ClipRuntime,
    std_k: float,
    random_trials: int,
    random_seed: int,
    sample: int,
    noise: int,
    skip_first_token: bool = False,
) -> list[ConditionResult]:
    num_steps = runtime.num_steps
    n_tokens = runtime.n_tokens
    action_dim = runtime.action_dim
    horizon = runtime.action_horizon
    layer_index = {name: index for index, (name, _layer) in enumerate(layers)}

    baseline_actions, baseline_activations, _ = _run_all(
        adapter,
        request,
        layers,
        num_steps=num_steps,
        n_tokens=n_tokens,
        std_k=std_k,
        skip_first_token=skip_first_token,
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
    baseline_full = baseline_actions[0, :, :action_dim]
    keep = (
        _valid_action_horizon(baseline_full) if skip_first_token else int(horizon)
    )
    if keep < horizon:
        print(f"Scoring first {keep} of {horizon} action steps (padding dropped).")
    baseline = baseline_full[:keep]

    identity_actions, _, _ = _run_all(
        adapter,
        request,
        layers,
        num_steps=num_steps,
        n_tokens=n_tokens,
        std_k=std_k,
        baseline_activations=baseline_activations,
        identity_writeback=True,
    )
    detail._assert_actions_equal(
        identity_actions,
        baseline_actions,
        name="identity write-back",
    )
    print("identity write-back (all layers, all steps): actions unchanged")

    restore_actions, _, _ = _run_all(
        adapter,
        request,
        layers,
        num_steps=num_steps,
        n_tokens=n_tokens,
        std_k=std_k,
        baseline_activations=baseline_activations,
        restore_after_clip=True,
    )
    detail._assert_actions_equal(
        restore_actions,
        baseline_actions,
        name="clip-then-restore write-back",
    )
    print("clip-then-restore write-back (all layers, all steps): actions unchanged")

    rows: list[ConditionResult] = []
    for step in range(num_steps):
        print(f"--- denoise step {step}/{num_steps - 1} ---")
        selective_actions, _, selective_stats = _run_all(
            adapter,
            request,
            layers,
            num_steps=num_steps,
            n_tokens=n_tokens,
            std_k=std_k,
            baseline_activations=baseline_activations,
            kind="selective",
            layer_index=layer_index,
            target_step=step,
            skip_first_token=skip_first_token,
        )
        selective_metrics = detail._action_metrics(
            selective_actions[0, :keep, :action_dim],
            baseline,
        )
        print(
            f"outlier sites={selective_stats.n_clipped}/{selective_stats.n_sites} "
            f"empty={selective_stats.n_empty} "
            f"values={selective_stats.outlier_count} "
            f"mean_L1={selective_stats.mean_removed_l1_pct:.4f}% "
            f"mean={selective_metrics.arm_mean_shift_rmse:.3e} "
            f"local={selective_metrics.arm_local_rmse:.3e}"
        )

        trial_metrics: list[detail.ActionMetrics] = []
        trial_stats: list[SiteStats] = []
        for trial in range(random_trials):
            bulk_actions, _, bulk_stats = _run_all(
                adapter,
                request,
                layers,
                num_steps=num_steps,
                n_tokens=n_tokens,
                std_k=std_k,
                baseline_activations=baseline_activations,
                kind="random",
                bulk_seed=random_seed + trial * 1_000_003 + step * 17,
                layer_index=layer_index,
                target_step=step,
                skip_first_token=skip_first_token,
            )
            trial_metrics.append(
                detail._action_metrics(
                    bulk_actions[0, :keep, :action_dim], baseline
                )
            )
            trial_stats.append(bulk_stats)
            print(
                f"  bulk trial={trial} sites={bulk_stats.n_clipped}/{bulk_stats.n_sites} "
                f"expand={bulk_stats.n_expanded} "
                f"mean_L1={bulk_stats.mean_bulk_removed_l1_pct:.4f}% "
                f"mean={trial_metrics[-1].arm_mean_shift_rmse:.3e} "
                f"local={trial_metrics[-1].arm_local_rmse:.3e}"
            )
        random_mean, random_std = detail._mean_std(trial_metrics)
        rows.append(
            ConditionResult(
                sample=sample,
                noise=noise,
                step=step,
                n_layers=len(layers),
                n_steps=num_steps,
                selective_stats=selective_stats,
                bulk_stats=_mean_site_stats(trial_stats),
                selective=selective_metrics,
                random_mean=random_mean,
                random_std=random_std,
            )
        )
    return rows


def _as_labeled(
    rows: list[ConditionResult],
) -> list[tuple[int, int, detail.StepResult]]:
    return [
        (
            row.sample,
            row.noise,
            detail.StepResult(
                step=row.step,
                outlier_count=row.selective_stats.outlier_count,
                selected_channels=row.selective_stats.selected_channels,
                bulk_count=row.bulk_stats.bulk_count,
                removed_l1_pct=row.selective_stats.mean_removed_l1_pct,
                bulk_removed_l1_pct=row.bulk_stats.mean_bulk_removed_l1_pct,
                selective=row.selective,
                random_mean=row.random_mean,
                random_std=row.random_std,
            ),
        )
        for row in rows
    ]


def _rows_by_step(rows: list[ConditionResult]) -> dict[int, list[ConditionResult]]:
    grouped: dict[int, list[ConditionResult]] = defaultdict(list)
    for row in rows:
        grouped[row.step].append(row)
    return dict(grouped)


def _verdict(rows: list[ConditionResult]) -> str:
    if not rows:
        raise ValueError("Cannot judge an empty result list.")
    grouped = _rows_by_step(rows)
    lines: list[str] = []
    for step in sorted(grouped):
        print(f"\n=== denoise step {step} (n={len(grouped[step])} sample×noise) ===")
        text = detail._verdict(_as_labeled(grouped[step]))
        lines.append(f"step {step}: {text}")
    print(f"\n=== overall (n={len(rows)} sample×noise×step) ===")
    overall = detail._verdict(_as_labeled(rows))
    lines.append(f"overall: {overall}")
    expanded = [float(row.bulk_stats.n_expanded) for row in rows]
    expand_mean = detail._mean(expanded)
    expand_max = int(max(expanded))
    if expand_max > 0:
        note = (
            f"note: MAD-to-μ+3σ band was short at mean {expand_mean:.2f} "
            f"sites/step (max {expand_max}); recruited largest leftover "
            "below the MAD floor until L1 matched."
        )
        print(note)
        lines.append(note)
    return "\n".join(lines)


def _write_csv(rows: list[ConditionResult], output: Path) -> None:
    metric_names = [field.name for field in fields(detail.ActionMetrics)]
    fieldnames = [
        "sample",
        "noise",
        "step",
        "n_layers",
        "n_steps",
        "selective_n_clipped",
        "selective_n_empty",
        "bulk_n_expanded",
        "selective_outlier_count",
        "selective_mean_removed_l1_pct",
        "bulk_n_clipped",
        "bulk_count",
        "bulk_mean_removed_l1_pct",
        *[f"selective_{name}" for name in metric_names],
        *[f"random_mean_{name}" for name in metric_names],
        *[f"random_std_{name}" for name in metric_names],
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            item = {
                "sample": row.sample,
                "noise": row.noise,
                "step": row.step,
                "n_layers": row.n_layers,
                "n_steps": row.n_steps,
                "selective_n_clipped": row.selective_stats.n_clipped,
                "selective_n_empty": row.selective_stats.n_empty,
                "bulk_n_expanded": row.bulk_stats.n_expanded,
                "selective_outlier_count": row.selective_stats.outlier_count,
                "selective_mean_removed_l1_pct": row.selective_stats.mean_removed_l1_pct,
                "bulk_n_clipped": row.bulk_stats.n_clipped,
                "bulk_count": row.bulk_stats.bulk_count,
                "bulk_mean_removed_l1_pct": row.bulk_stats.mean_bulk_removed_l1_pct,
            }
            for prefix, metrics in (
                ("selective", row.selective),
                ("random_mean", row.random_mean),
                ("random_std", row.random_std),
            ):
                for name in metric_names:
                    item[f"{prefix}_{name}"] = getattr(metrics, name)
            writer.writerow(item)


def _mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot reduce empty metric list.")
    return float(array.mean()), float(array.std(ddof=0))


def _plot(
    rows: list[ConditionResult],
    output: Path,
    *,
    skip_first_token: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    grouped = _rows_by_step(rows)
    steps = sorted(grouped)
    xs = np.asarray(steps, dtype=np.float64)
    sel_mean, sel_mean_std = zip(
        *[
            _mean_std([row.selective.arm_mean_shift_rmse for row in grouped[step]])
            for step in steps
        ]
    )
    bulk_mean, bulk_mean_std = zip(
        *[
            _mean_std([row.random_mean.arm_mean_shift_rmse for row in grouped[step]])
            for step in steps
        ]
    )
    sel_local, sel_local_std = zip(
        *[
            _mean_std([row.selective.arm_local_rmse for row in grouped[step]])
            for step in steps
        ]
    )
    bulk_local, bulk_local_std = zip(
        *[
            _mean_std([row.random_mean.arm_local_rmse for row in grouped[step]])
            for step in steps
        ]
    )
    mean_ratio = [
        grouped_mean / bulk if bulk > 0.0 else float("inf")
        for grouped_mean, bulk in zip(sel_mean, bulk_mean)
    ]
    local_ratio = [
        grouped_local / bulk if bulk > 0.0 else float("inf")
        for grouped_local, bulk in zip(sel_local, bulk_local)
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), constrained_layout=True)
    axes[0].errorbar(
        xs, sel_mean, yerr=sel_mean_std, marker="o", capsize=3, label="outlier"
    )
    axes[0].errorbar(
        xs,
        bulk_mean,
        yerr=bulk_mean_std,
        marker="s",
        capsize=3,
        label="MAD band + expand",
    )
    axes[0].set_title("Arm mean shift (coarse path)")
    axes[0].set_ylabel("RMSE")
    axes[1].errorbar(
        xs, sel_local, yerr=sel_local_std, marker="o", capsize=3, label="outlier"
    )
    axes[1].errorbar(
        xs,
        bulk_local,
        yerr=bulk_local_std,
        marker="s",
        capsize=3,
        label="MAD band + expand",
    )
    axes[1].set_title("Arm local residual (detail)")
    axes[1].set_ylabel("RMSE")
    axes[2].plot(xs, mean_ratio, "o-", label="mean_shift")
    axes[2].plot(xs, local_ratio, "s--", label="local")
    axes[2].axhline(0.25, color="0.4", linewidth=0.8, linestyle=":")
    axes[2].axhline(0.5, color="0.4", linewidth=0.8, linestyle="--")
    axes[2].set_title("Outlier / bulk ratio")
    axes[2].set_ylabel("ratio")
    for ax in axes:
        ax.set_xlabel("denoise step")
        ax.set_xticks(list(steps))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    state_note = (
        "GR00T: state token (row 0) not clipped; "
        if skip_first_token
        else ""
    )
    fig.suptitle(
        state_note
        + "All per-step DiT linears, one denoise step at a time; "
        "outlier vs MAD-to-μ+3σ band, largest-expand if short (matched L1)",
        fontsize=12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
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
    matched.add_model_cli(parser)
    parser.add_argument("--layer-regex", default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--samples", default=None)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--noise-seeds", default=None)
    parser.add_argument("--random-seed", type=int, default=1000)
    parser.add_argument("--random-trials", type=int, default=4)
    parser.add_argument("--outlier-std-k", type=float, default=3.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--params-dtype", default="bfloat16")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tools/img/dit_outlier_vs_samech_mad_band_expand_all_by_step"),
    )
    return parser


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
        detail._parse_nonneg_ints(args.samples, name="--samples")
        if args.samples is not None
        else [int(args.sample_index)]
    )
    noise_ids = (
        detail._parse_nonneg_ints(args.noise_seeds, name="--noise-seeds")
        if args.noise_seeds is not None
        else [int(args.noise_seed)]
    )

    adapter = matched.adapter_from_args(args)
    model = adapter.build_model()
    model.eval()
    adapter.warmup_for_calibration(model)
    config = QVLAConfig.for_model_kind(args.model)
    layers = matched._dit_layers(model, args.layer_regex, config=config)
    runtime = matched.clip_runtime(adapter, config)
    need = max(sample_ids) + 1
    batches = list(adapter.iter_calibration_batches(need))
    if len(batches) != need:
        raise RuntimeError(
            f"Requested samples {sample_ids}, calibration yielded "
            f"{len(batches)} samples."
        )
    conditions = [(sample, seed) for sample in sample_ids for seed in noise_ids]
    skip_first_token = args.model == "groot_n17"
    print(
        f"model={args.model}, layers={len(layers)}, samples={sample_ids}, "
        f"noise_seeds={noise_ids}, n_conditions={len(conditions)}, "
        f"steps={runtime.num_steps}, n_tokens={runtime.n_tokens}, "
        f"action_dim={runtime.action_dim}, random_trials={args.random_trials}, "
        f"std_k={args.outlier_std_k}, mode=all-linears-one-step, "
        f"skip_first_token={skip_first_token}"
    )

    rows: list[ConditionResult] = []
    for index, (sample, seed) in enumerate(conditions, start=1):
        request = matched._fixed_noise_request(
            adapter, batches[sample], runtime, noise_seed=seed
        )
        print(
            f"\n=== condition {index}/{len(conditions)} "
            f"sample={sample} noise={seed} ==="
        )
        rows.extend(
            _experiment(
                adapter,
                request,
                layers,
                runtime=runtime,
                std_k=args.outlier_std_k,
                random_trials=args.random_trials,
                random_seed=args.random_seed,
                sample=sample,
                noise=seed,
                skip_first_token=skip_first_token,
            )
        )

    verdict = _verdict(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "outlier_vs_bulk_all_layers_by_step.csv"
    png = args.output_dir / "outlier_vs_bulk_all_layers_by_step.png"
    verdict_path = args.output_dir / "hypothesis.txt"
    _write_csv(rows, csv_path)
    _plot(rows, png, skip_first_token=skip_first_token)
    verdict_path.write_text(verdict + "\n", encoding="utf-8")
    for path in (csv_path, png, verdict_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Failed to write {path}.")
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
