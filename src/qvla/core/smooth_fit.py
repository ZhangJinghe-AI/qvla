"""Fit per-input-channel SmoothQuant scale ``s`` from calibration stats.

Given a linear layer with weight ``W ∈ R^{N×K}`` and per-input-channel
activation amax ``a ∈ R^K``, the SmoothQuant scale is ::

    w_j = max_i |W_{i,j}|
    F̃_j = F_j / max_k F_k              # ∈ (0, 1]
    g_j  = (1 + beta · F̃_j) / mean(1 + beta · F̃)
    ã_j  = a_j · g_j                    # beta=0 → ã=a
    s_j  = ã_j^alpha / w_j^{1-alpha}

``alpha`` still owns the act↔weight migration (SmoothQuant paper default
``0.5``). Fisher only reshapes the *effective* activation amax before that
power:

* larger ``a`` → larger ``s``
* larger ``F̃`` → larger ``s`` (boost)
* ``mean(g) = 1`` so Fisher redistributes mass instead of globally
  inflating/deflating ``s``
* flat Fisher (all equal) → ``g ≡ 1`` → identical to plain SmoothQuant
* peaked Fisher → only the salient tail is boosted; low-F channels stay
  near ``a`` (unlike the old ``ã = a · F̃^beta``, which crushed almost
  every channel when ``F`` was sparse and caused W4A4 collapse)

``alpha=0`` ignores Fisher entirely because ``ã^0 ≡ 1``.
"""

from __future__ import annotations

import logging
import math

import torch

logger = logging.getLogger(__name__)


def pmean_per_step_channel_amax(per_step: torch.Tensor, p: float) -> torch.Tensor:
    """``a_j = (mean_t a_{j,t}^p)^{1/p}`` over denoise steps.

    ``per_step`` is ``(num_steps, in_features)`` non-negative channel absmax.
    """
    if per_step.ndim != 2 or per_step.shape[0] < 1 or per_step.shape[1] < 1:
        raise ValueError(
            "per_step amax must have shape (num_steps>=1, in_features>=1), "
            f"got {tuple(per_step.shape)}."
        )
    p_val = float(p)
    if not math.isfinite(p_val) or p_val <= 0.0:
        raise ValueError(f"p-mean p must be finite and > 0, got {p}.")
    table = per_step.detach().to(torch.float32)
    if not bool(torch.isfinite(table).all().item()):
        raise RuntimeError("per-step channel amax contains non-finite values.")
    if bool((table < 0).any().item()):
        raise RuntimeError("per-step channel amax must be non-negative.")
    out = table.pow(p_val).mean(dim=0).pow(1.0 / p_val)
    if not bool(torch.isfinite(out).all().item()):
        raise RuntimeError(
            f"p-mean(p={p_val}) produced non-finite channel amax."
        )
    return out


def fit_smooth_scale(
    *,
    layer_name: str | None,
    weight: torch.Tensor,
    act_channel_amax: torch.Tensor,
    alpha: float,
    epsilon: float = 1e-5,
    fisher: torch.Tensor | None = None,
    fisher_beta: float = 0.0,
    s_min: float = 1e-4,
    s_max: float = 1e4,
) -> torch.Tensor:
    """Fit :math:`s` for one layer.

    Parameters
    ----------
    weight:
        ``(out_features, in_features)`` weight tensor (already flattened for
        Conv2d).
    act_channel_amax:
        ``(in_features,)`` positive tensor used as the SmoothQuant numerator
        ``a_j``. Typically hard absmax ``max_t |x_{t,j}|``, or a clipped
        percentile statistic (see ``ScopeConfig.smooth_act_percentile``).
    alpha:
        Migration knob in ``[0, 1]``. ``0.5`` is a reasonable default.
    epsilon:
        Floor added to ``a`` / ``w`` / ``F̃`` before powers to avoid ``0`` on
        near-dead channels.
    fisher:
        Optional ``(in_features,)`` per-channel Fisher sensitivity. Ignored
        when ``fisher_beta == 0``. Max-normalised to ``F̃ ∈ (0, 1]`` then
        converted to a mean-1 boost ``g = (1 + beta·F̃) / mean(...)``.
    fisher_beta:
        Non-negative boost strength on ``F̃`` (see module docstring). ``0``
        disables Fisher reweighting.
    s_min, s_max:
        Hard clamps on ``s`` to bound the weight-column blow-up.

    Raises
    ------
    ValueError
        On invalid arguments or degenerate stats (non-finite ``s``, wrong
        shapes, negative alpha/beta, missing Fisher when ``fisher_beta != 0``,
        etc.). All-zero Fisher is *not* an error: the layer is assumed off
        the action graph and Fisher reweighting is skipped (plain SQ).
    """
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2-D, got {tuple(weight.shape)}.")
    N, K = int(weight.shape[0]), int(weight.shape[1])
    del N
    if act_channel_amax.ndim != 1 or int(act_channel_amax.shape[0]) != K:
        raise ValueError(
            f"act_channel_amax must have shape ({K},), got "
            f"{tuple(act_channel_amax.shape)}."
        )
    if not (0.0 <= float(alpha) <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}.")
    if float(fisher_beta) < 0.0:
        raise ValueError(
            f"fisher_beta must be >= 0 (boost strength), got {fisher_beta}."
        )
    if float(epsilon) <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}.")
    if not (float(s_min) > 0.0 and float(s_max) > float(s_min)):
        raise ValueError(
            f"s_min / s_max invalid: got s_min={s_min}, s_max={s_max}."
        )

    device = weight.device
    W = weight.detach().to(device, dtype=torch.float32)
    a = act_channel_amax.detach().to(device, dtype=torch.float32).abs()
    w = W.abs().amax(dim=0)

    a_c = a.clamp_min(float(epsilon))
    w_c = w.clamp_min(float(epsilon))

    # Fold Fisher into activation amax *before* the SmoothQuant power so
    # alpha still owns act↔weight migration.
    a_eff = a_c
    used_fisher = False
    if float(fisher_beta) != 0.0:
        if fisher is None:
            raise ValueError(
                f"fisher_beta={fisher_beta} requires a Fisher tensor for "
                f"layer {layer_name!r}, got None."
            )
        if fisher.ndim != 1 or int(fisher.shape[0]) != K:
            raise ValueError(
                f"fisher must have shape ({K},), got {tuple(fisher.shape)}."
            )
        f = fisher.detach().to(device, dtype=torch.float32).clamp_min(0.0)
        f_max = f.max()
        if float(f_max.item()) <= 0.0:
            # Layers off the action graph (e.g. pi05 LLM final o_proj / MLP)
            # legitimately report all-zero Fisher. Keep plain SmoothQuant.
            logger.warning(
                "SmoothQuant %s: fisher_beta=%.2f but Fisher is all-zero "
                "(layer not on the action graph); skipping Fisher reweighting.",
                layer_name,
                float(fisher_beta),
            )
        else:
            # F̃ ∈ (0, 1]; boost-only gate with mean(g)=1 so peaked Fisher
            # raises the salient tail without crushing the bulk of channels.
            f_norm = (f / f_max).clamp_min(float(epsilon))
            g = 1.0 + float(fisher_beta) * f_norm
            g = g / g.mean().clamp_min(float(epsilon))
            a_eff = a_c * g
            used_fisher = True

    s = a_eff.pow(float(alpha)) / w_c.pow(1.0 - float(alpha))

    # Dead channels (both activation and weight zero) contribute nothing to y;
    # pin s=1 so the runtime division is a no-op there.
    dead_mask = (a == 0) & (w == 0)
    if bool(dead_mask.any().item()):
        s = torch.where(dead_mask, torch.ones_like(s), s)

    if not torch.isfinite(s).all():
        raise ValueError(
            f"SmoothQuant scale for layer {layer_name!r} contains non-finite "
            "entries after fit; inspect calibration stats."
        )

    s = s.clamp(min=float(s_min), max=float(s_max))

    logger.info(
        "SmoothQuant fit %s: alpha=%.2f fisher_beta=%.2f "
        "s in [%.3e, %.3e] log10-range=%.2f dead=%d/%d used_fisher=%s",
        layer_name,
        float(alpha),
        float(fisher_beta),
        float(s.min().item()),
        float(s.max().item()),
        float((s.max() / s.min()).log10().item()),
        int(dead_mask.sum().item()),
        K,
        used_fisher,
    )

    return s.cpu().to(torch.float32).contiguous()


__all__ = ["fit_smooth_scale", "pmean_per_step_channel_amax"]
