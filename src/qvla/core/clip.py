"""Activation clipping primitives — outlier-aware per-channel amax estimators.

These functions compute per-channel activation maxima using various strategies
to handle outlier tokens:

* :func:`channel_outlier_bulk_kappa_amax` — clip at ``κ · percentile(|x|)``
* :func:`channel_outlier_mean_std_amax` — clip at ``μ + k·σ``
* :func:`layer_outlier_mean_std_amax` — one layer-wide ``μ + k·σ`` shared
  by every channel
* :func:`selective_channel_outlier_mean_std_amax` — first select unusually
  large-amax channels with robust cross-channel MAD, then apply ``μ + k·σ``
  only to those channels
"""

from __future__ import annotations

import math

import torch


MAD_NORMAL_SCALE = 1.4826
SELECTIVE_CHANNEL_MAD_K = 3.0


def scheduled_outlier_std_k(
    k_start: float,
    k_end: float,
    *,
    step: int,
    num_steps: int,
) -> float:
    """Linear ``k`` from ``k_start`` at step 0 to ``k_end`` at the last step."""
    start = float(k_start)
    end = float(k_end)
    if not math.isfinite(start) or start <= 0.0:
        raise ValueError(f"k_start must be finite and > 0, got {k_start}.")
    if not math.isfinite(end) or end <= 0.0:
        raise ValueError(f"k_end must be finite and > 0, got {k_end}.")
    n = int(num_steps)
    t = int(step)
    if n < 2:
        raise ValueError(
            f"std_k schedule requires num_steps>=2, got num_steps={num_steps}."
        )
    if t < 0 or t >= n:
        raise ValueError(
            f"step must be in [0, {n}), got {step}."
        )
    return start + (end - start) * (t / (n - 1))


def std_k_schedule_bounds(
    std_k: float, down: float, up: float
) -> tuple[float, float]:
    """Return ``(std_k - down, std_k + up)``; ``down=up=0`` keeps ``(std_k, std_k)``."""
    k = float(std_k)
    d = float(down)
    u = float(up)
    if not math.isfinite(k) or k <= 0.0:
        raise ValueError(f"std_k must be finite and > 0, got {std_k}.")
    if not math.isfinite(d) or d < 0.0:
        raise ValueError(f"down must be finite and >= 0, got {down}.")
    if not math.isfinite(u) or u < 0.0:
        raise ValueError(f"up must be finite and >= 0, got {up}.")
    if d == 0.0 and u == 0.0:
        return k, k
    start = k - d
    end = k + u
    if not math.isfinite(start) or start <= 0.0:
        raise ValueError(
            f"std_k - down must be > 0, got std_k={std_k}, down={down}."
        )
    if not math.isfinite(end) or end <= 0.0:
        raise ValueError(
            f"std_k + up must be > 0, got std_k={std_k}, up={up}."
        )
    return start, end


def select_channels_by_robust_amax(
    channel_amax: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select large channels with ``median + 3*1.4826*MAD``.

    Returns ``(selected, median, mad, threshold)``. Invalid or degenerate
    statistics raise; this helper never falls back to another selector.
    """
    if channel_amax.ndim != 1 or channel_amax.numel() < 2:
        raise ValueError(
            "select_channels_by_robust_amax expects a 1D tensor with at "
            f"least two channels, got shape={tuple(channel_amax.shape)}."
        )
    values = channel_amax.detach().to(torch.float32)
    if not bool(torch.isfinite(values).all().item()):
        raise RuntimeError(
            "selective channel clip requires finite per-channel activation amax."
        )

    center = torch.quantile(values, 0.5)
    mad = torch.quantile((values - center).abs(), 0.5)
    if not bool(torch.isfinite(center).item()) or not bool(torch.isfinite(mad).item()):
        raise RuntimeError(
            "selective channel clip produced non-finite median/MAD statistics."
        )
    if float(mad.item()) <= 0.0:
        raise RuntimeError(
            "selective channel clip requires positive cross-channel MAD; "
            f"got MAD={float(mad.item()):.6g}."
        )

    threshold = center + SELECTIVE_CHANNEL_MAD_K * MAD_NORMAL_SCALE * mad
    return values > threshold, center, mad, threshold


def channel_outlier_bulk_kappa_amax(
    abs_activations: torch.Tensor,
    *,
    kappa: float,
    bulk_percentile: float = 95.0,
) -> torch.Tensor:
    """Per-channel amax via bulk tip clip: ``a_j = min(max_j, κ · P_β(|x|))``.

    ``abs_activations`` is ``(num_tokens, in_features)``, typically **all
    calibration tokens concatenated**.

    For channel ``j``::

        ref_j = percentile_β(|x_{:,j}|)
        a_j   = min( max_t |x_{t,j}|, κ · ref_j )

    So rare tips above the bulk are clipped; heavy / structured mass near the
    bulk keeps a high ``a_j`` (near hard max when ``max ≤ κ·P_β``).

    Raises if ``T<1``, ``kappa<=0``, or ``bulk_percentile`` not in ``(0, 100)``.
    """
    if abs_activations.ndim != 2:
        raise ValueError(
            f"channel_outlier_bulk_kappa_amax expects (num_tokens, in_features), "
            f"got {tuple(abs_activations.shape)}."
        )
    if float(kappa) <= 0.0:
        raise ValueError(f"kappa must be > 0, got {kappa}.")
    if not (0.0 < float(bulk_percentile) < 100.0):
        raise ValueError(
            f"bulk_percentile must be in (0, 100), got {bulk_percentile}."
        )
    x = abs_activations.detach().to(torch.float32)
    t = int(x.shape[0])
    if t < 1:
        raise ValueError(
            f"channel_outlier_bulk_kappa_amax needs >= 1 token, got T={t}."
        )
    ref = torch.quantile(x, float(bulk_percentile) / 100.0, dim=0)
    thr = float(kappa) * ref.clamp_min(1e-30)
    return torch.minimum(x.amax(dim=0), thr)


def channel_outlier_mean_std_amax(
    abs_activations: torch.Tensor,
    *,
    std_k: float,
) -> torch.Tensor:
    """Per-channel amax via mean + k·std tip clip: ``a_j = min(max_j, μ_j + k·σ_j)``.

    ``abs_activations`` is ``(num_tokens, in_features)``. Uses population std
    (``unbiased=False``). Raises if ``T<2`` or ``std_k<=0``.
    """
    if abs_activations.ndim != 2:
        raise ValueError(
            f"channel_outlier_mean_std_amax expects (num_tokens, in_features), "
            f"got {tuple(abs_activations.shape)}."
        )
    if float(std_k) <= 0.0:
        raise ValueError(f"std_k must be > 0, got {std_k}.")
    x = abs_activations.detach().to(torch.float32)
    t = int(x.shape[0])
    if t < 2:
        raise ValueError(
            f"channel_outlier_mean_std_amax needs >= 2 tokens, got T={t}."
        )
    thr = x.mean(dim=0) + float(std_k) * x.std(dim=0, unbiased=False)
    return torch.minimum(x.amax(dim=0), thr)


def layer_outlier_mean_std_amax(
    abs_activations: torch.Tensor,
    *,
    std_k: float,
) -> torch.Tensor:
    """One layer-wide mean+k·std threshold shared by every channel.

    ``abs_activations`` is ``(num_tokens, in_features)``. Flatten all tip
    ``|x|`` values, set ``c = μ + k·σ`` (population std), then ::

        a_j = min(max_t |x_{t,j}|, c)

    This is the naive global-clip baseline: channels whose hard amax is
    already ``≤ c`` are untouched; larger channels share the same cap.

    Raises if the flattened length is ``< 2``, ``std_k<=0``, or ``c`` is
    not finite. Does not fall back to per-channel statistics.
    """
    if abs_activations.ndim != 2:
        raise ValueError(
            f"layer_outlier_mean_std_amax expects (num_tokens, in_features), "
            f"got {tuple(abs_activations.shape)}."
        )
    if float(std_k) <= 0.0:
        raise ValueError(f"std_k must be > 0, got {std_k}.")
    x = abs_activations.detach().to(torch.float32)
    n = int(x.numel())
    if n < 2:
        raise ValueError(
            f"layer_outlier_mean_std_amax needs >= 2 |x| values, got {n}."
        )
    flat = x.reshape(-1)
    if not bool(torch.isfinite(flat).all().item()):
        raise RuntimeError("layer-global clip requires finite |x| values.")
    thr = flat.mean() + float(std_k) * flat.std(unbiased=False)
    if not bool(torch.isfinite(thr).item()):
        raise RuntimeError(
            f"layer-global clip threshold is not finite: {float(thr.item()):.6g}."
        )
    return torch.minimum(x.amax(dim=0), thr)


def selective_channel_outlier_mean_std_amax(
    abs_activations: torch.Tensor,
    *,
    std_k: float,
) -> torch.Tensor:
    """Apply mean+std clipping only to robustly large-amax channels.

    The first stage compares per-channel hard amax values and selects

    ``amax_j > median(amax) + 3 * 1.4826 * MAD(amax)``.

    The selected channels use the existing per-channel ``mean + std_k*std``
    threshold. Every unselected channel keeps its hard amax, so runtime
    clipping is an exact no-op for that channel.

    This mode deliberately raises when MAD is zero or any derived statistic is
    non-finite. It does not silently fall back to clipping every channel.
    """
    clipped = channel_outlier_mean_std_amax(
        abs_activations,
        std_k=std_k,
    )
    x = abs_activations.detach().to(torch.float32)
    hard_amax = x.amax(dim=0)
    selected, _, _, _ = select_channels_by_robust_amax(hard_amax)
    result = torch.where(selected, clipped, hard_amax)
    if not bool(torch.isfinite(result).all().item()):
        raise RuntimeError("selective channel clip produced non-finite thresholds.")
    return result


__all__ = [
    "MAD_NORMAL_SCALE",
    "SELECTIVE_CHANNEL_MAD_K",
    "scheduled_outlier_std_k",
    "std_k_schedule_bounds",
    "channel_outlier_bulk_kappa_amax",
    "channel_outlier_mean_std_amax",
    "layer_outlier_mean_std_amax",
    "select_channels_by_robust_amax",
    "selective_channel_outlier_mean_std_amax",
]
