"""Offline weight quantizers — RTN, GPTQ, and the RTN-residual variant.

All three operate on a **rotated** weight matrix (rotation applied upstream by
the builder). They produce the same output schema so the pack format doesn't
care which quantizer was used:

* ``qweight``      : ``int8`` tensor holding signed symmetric int values in
                     ``[qmin, qmax]`` for ``weight_bits`` (one value per int8
                     byte — packing into sub-byte storage is deferred to the
                     on-disk pack format).
* ``weight_scale`` : per-group float32 scales such that
                     ``W_fp ≈ qweight * weight_scale``. For ``group_size=-1``
                     the scale is per-output-channel ``(N,)``; otherwise it is
                     ``(N, K/group_size)``.

The GPTQ implementation here is a clean from-scratch port of Frantar et al.
(2022) — Hessian-based error compensation with a Cholesky factorization for
numerical stability. It's faster than the original implementation when the
group size matches the block size (the common case).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

# ``weight_bits`` / ``act_bits`` == 16 means "keep full precision" (no quant).
NO_QUANT_BITS = 16


def is_no_quant(bits: int) -> bool:
    """True when the configured bit width disables quantization."""
    return bits >= NO_QUANT_BITS


def symmetric_quant_range(bits: int) -> tuple[int, int]:
    """Signed symmetric quant grid for ``bits`` (e.g. 4 → ``(-8, 7)``)."""
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    return qmin, qmax


def percentile_amax(amax_row: torch.Tensor, percentile: float) -> torch.Tensor:
    """Trim per-channel amax to a global cross-channel ``percentile`` cap.

    Legacy helper: takes one max per channel, computes the percentile *across
    channels*, then clamps. Prefer :func:`channel_percentile_amax` for act scales.
    """
    if percentile >= 100.0:
        return amax_row.clone()
    q = torch.quantile(amax_row.to(torch.float32), percentile / 100.0)
    return amax_row.clamp_max(q)


def _topk_percentile_along(
    x: torch.Tensor, percentile: float, *, dim: int
) -> torch.Tensor:
    """Approximate ``percentile`` of ``x`` along ``dim`` via upper-tail ``topk``.

    Cheaper than ``torch.quantile`` (no full sort / interpolation). For high
    thresholds (e.g. 99.9) this is usually close enough for act-scale clipping.
    """
    n = int(x.shape[dim])
    # k-th largest ≈ upper-tail percentile. For 99.9 with n=1000 → k=1 (max).
    k = max(1, min(n, int(math.ceil((1.0 - percentile / 100.0) * n))))
    return torch.topk(x, k=k, dim=dim, largest=True).values.select(dim, k - 1)


def token_percentile_amax(
    abs_activations: torch.Tensor, percentile: float
) -> torch.Tensor:
    """Per-token ``percentile`` of ``|activations|`` along the batch×channel axes."""
    if abs_activations.ndim != 3:
        raise ValueError(
            f"token_percentile_amax expects (batch, num_tokens, in_features), "
            f"got {tuple(abs_activations.shape)}."
        )
    # Fold batch×channel into one axis: (T, B*D).
    x = abs_activations.to(torch.float32).permute(1, 0, 2).reshape(
        abs_activations.shape[1], -1
    )
    return _topk_percentile_along(x, percentile, dim=1)


def channel_percentile_amax(
    abs_activations: torch.Tensor, percentile: float
) -> torch.Tensor:
    """Per-channel ``percentile`` of ``|activations|`` along the token axis."""
    if abs_activations.ndim != 2:
        raise ValueError(
            f"channel_percentile_amax expects (num_tokens, in_features), "
            f"got {tuple(abs_activations.shape)}."
        )
    return _topk_percentile_along(
        abs_activations.to(torch.float32), percentile, dim=0
    )



# --------------------------------------------------------------------------- #
# FP4 (E2M1) grid — micro-float quantization.                                 #
# --------------------------------------------------------------------------- #

# E2M1 representable magnitudes (bias=1): ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}.
# 4-bit signed: 1 sign + 2 exponent + 1 mantissa → 16 codes total.
_FP4_E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)
_FP4_E2M1_VALUES_CACHE: dict[str, torch.Tensor] = {}
_FP4_E2M1_GRID_CACHE: dict[str, torch.Tensor] = {}


def warmup_fp4_cuda_tensors(device: torch.device | str) -> None:
    """Pre-allocate FP4 lookup tables on ``device`` before CUDA graph capture.

    CUDA graph capture forbids CPU→GPU copies unless the source tensor is
    pinned. NVFP4 activation quant reads module-level CPU constants; warm
    them up outside capture so graph replay only touches device-resident
    tensors.
    """
    dev = torch.device(device)
    if dev.type != "cuda":
        return
    _fp4_e2m1_grid(dev)
    _fp4_e2m1_values_on(dev)


def _fp4_e2m1_values_on(device: torch.device) -> torch.Tensor:
    key = str(device)
    cached = _FP4_E2M1_VALUES_CACHE.get(key)
    if cached is None:
        cached = _FP4_E2M1_VALUES.to(device=device)
        _FP4_E2M1_VALUES_CACHE[key] = cached
    return cached


def _fp4_e2m1_grid(device: torch.device | None = None) -> torch.Tensor:
    """Full signed E2M1 grid: [-6, ..., -0.5, 0, 0.5, ..., 6] (15 unique)."""
    dev = device or torch.device("cpu")
    key = str(dev)
    cached = _FP4_E2M1_GRID_CACHE.get(key)
    if cached is None:
        pos = _fp4_e2m1_values_on(dev)
        neg = -pos[1:].flip(0)
        cached = torch.cat([neg, pos])
        _FP4_E2M1_GRID_CACHE[key] = cached
    return cached


def _nearest_on_sorted_grid(
    values: torch.Tensor, sorted_grid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest-neighbour on a 1-D sorted grid with O(N) memory."""
    n = sorted_grid.numel()
    if n == 0:
        raise ValueError("sorted_grid must be non-empty")
    if n == 1:
        idx = torch.zeros_like(values, dtype=torch.long)
        return sorted_grid[idx], idx

    idx = torch.searchsorted(sorted_grid, values, right=False).clamp(1, n - 1)
    lower = sorted_grid[idx - 1]
    upper = sorted_grid[idx]
    dist_lower = (values - lower).abs()
    dist_upper = (values - upper).abs()
    pick_upper = dist_upper < dist_lower
    out_idx = torch.where(pick_upper, idx, idx - 1)
    return sorted_grid[out_idx], out_idx


def _fp4_e2m1_quantize(
    scaled: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest-neighbour round onto the signed E2M1 grid.

    ``scaled`` is already ``w / scale`` (any shape). Returns
    ``(codes_int8, q_vals)`` where ``q_vals`` has the same shape as
    ``scaled`` and ``codes`` encodes ``sign * index`` into
    ``_FP4_E2M1_VALUES``.
    """
    pos = _fp4_e2m1_values_on(scaled.device)
    orig_shape = scaled.shape
    flat = scaled.reshape(-1)
    signs = flat.sign()
    q_abs, code_idx = _nearest_on_sorted_grid(flat.abs(), pos)
    q_vals = (q_abs * signs).reshape(orig_shape)
    codes = (code_idx * signs.to(torch.int8)).to(torch.int8).reshape(orig_shape)
    return codes, q_vals


def _fp4_e2m1_dequantize(codes: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Convert int8 E2M1 codes back to float values."""
    pos = _fp4_e2m1_values_on(device)
    abs_codes = codes.abs().to(torch.long).clamp(max=7)
    magnitudes = pos[abs_codes]
    return magnitudes * codes.sign().to(magnitudes.dtype)


# --------------------------------------------------------------------------- #
# NVFP4 — official NVIDIA hierarchical scaling (Transformer Engine).          #
#                                                                             #
#   x ≈ x_e2m1 * s_block * s_global                                           #
#                                                                             #
#   s_global = global_amax / (FP8_E4M3_MAX * FP4_E2M1_MAX)   # FP32           #
#   s_block  = (block_amax / FP4_E2M1_MAX) / s_global        # FP8 E4M3       #
#   block size = 16                                                           #
# --------------------------------------------------------------------------- #

NVFP4_BLOCK_SIZE = 16
FP8_E4M3_MAX = 448.0
FP4_E2M1_MAX = 6.0


def _validate_nvfp4_group_size(group_size: int) -> None:
    if group_size != NVFP4_BLOCK_SIZE:
        raise ValueError(
            f"NVFP4 requires group_size={NVFP4_BLOCK_SIZE}, got {group_size}."
        )


def _cast_fp8_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Round-trip through FP8 E4M3 (hardware storage dtype for NVFP4 block scales).

    Clamp to the E4M3 finite range *before* casting — oversized values become
    NaN under ``float8_e4m3fn``, and a post-cast ``clamp_min`` cannot recover.
    """
    return (
        x.clamp(min=0.0, max=FP8_E4M3_MAX)
        .to(torch.float8_e4m3fn)
        .to(torch.float32)
    )


def _nvfp4_compute_scales(
    w: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Official NVFP4 two-level scales for a ``(N, K)`` weight.

    Returns ``(s_global, s_block)`` where ``s_global`` is shape ``(1,)`` FP32
    and ``s_block`` is shape ``(N, K/group_size)`` after FP8 E4M3 cast.
    """
    _validate_nvfp4_group_size(group_size)
    N, K = w.shape
    if K % group_size != 0:
        raise ValueError(
            f"NVFP4 requires K divisible by group_size={group_size}, got K={K}."
        )
    global_amax = w.abs().amax().clamp_min(1e-12)
    s_global = (global_amax / (FP8_E4M3_MAX * FP4_E2M1_MAX)).reshape(1)

    w_blocks = w.reshape(N, K // group_size, group_size)
    block_amax = w_blocks.abs().amax(dim=2).clamp_min(1e-12)
    s_block = _cast_fp8_e4m3((block_amax / FP4_E2M1_MAX) / s_global)
    s_block = s_block.clamp_min(1e-12)
    return s_global.to(torch.float32), s_block.contiguous()


def _nvfp4_effective_scale(
    s_global: torch.Tensor, s_block: torch.Tensor
) -> torch.Tensor:
    """``s_global * s_block`` broadcast to ``s_block``'s shape."""
    return s_global.reshape(1, 1) * s_block


def nvfp4_quantize(
    w: torch.Tensor, *, group_size: int = NVFP4_BLOCK_SIZE
) -> QuantizedWeight:
    """Official NVFP4 RTN (Transformer Engine hierarchical scaling).

    ``group_size`` controls the block dimension and must be 16 for NVFP4.
    Stores ``weight_scale = s_global`` ``(1,)`` and
    ``weight_scale_2 = s_block`` ``(N, K/group_size)``.
    """
    _validate_nvfp4_group_size(group_size)
    w_fp32 = w.detach().to(torch.float32)
    N, K = w_fp32.shape
    s_global, s_block = _nvfp4_compute_scales(w_fp32, group_size)
    effective = _nvfp4_effective_scale(s_global, s_block)  # (N, num_blocks)
    scaled = (
        w_fp32.reshape(N, K // group_size, group_size)
        / effective.unsqueeze(2)
    ).reshape(N, K)
    codes, _ = _fp4_e2m1_quantize(scaled)
    return QuantizedWeight(
        qweight=codes,
        weight_scale=s_global.contiguous(),
        group_size=group_size,
        weight_bits=4,
        weight_format="nvfp",
        weight_scale_2=s_block,
    )


def nvfp4_dequantize(
    codes: torch.Tensor,
    s_global: torch.Tensor,
    s_block: torch.Tensor,
    group_size: int = NVFP4_BLOCK_SIZE,
) -> torch.Tensor:
    """``x = e2m1(codes) * s_block * s_global``."""
    _validate_nvfp4_group_size(group_size)
    N, K = codes.shape
    if K % group_size != 0 or s_block.shape != (N, K // group_size):
        raise ValueError(
            "NVFP4 scale shape mismatch: expected "
            f"{(N, K // group_size)} for codes shape {(N, K)}, "
            f"got {tuple(s_block.shape)}."
        )
    vals = _fp4_e2m1_dequantize(codes, device=codes.device)
    return (
        vals.reshape(N, K // group_size, group_size)
        * s_block.unsqueeze(2)
        * s_global.reshape(1, 1, 1)
    ).reshape(N, K)


def nvfp4_quantize_activation(
    x: torch.Tensor,
    *,
    group_size: int = NVFP4_BLOCK_SIZE,
    num_samples: int,
) -> torch.Tensor:
    """Online official NVFP4 for activations (software emulation).

    Applies one ``s_global`` per sample and per-token per-16-element FP8
    ``s_block`` along the last dimension, then rounds to E2M1 and dequantizes
    back to float. Leading rows must contain equally sized, contiguous samples.
    """
    _validate_nvfp4_group_size(group_size)
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}.")
    if x.shape[-1] % group_size != 0:
        raise ValueError(
            f"NVFP4 activations require last dim divisible by "
            f"group_size={group_size}, got {x.shape[-1]}."
        )
    x32 = x.to(torch.float32)
    orig_shape = x32.shape
    D = orig_shape[-1]
    flat = x32.reshape(-1, D)
    M = flat.shape[0]
    if M % num_samples != 0:
        raise ValueError(
            "NVFP4 activation rows must divide evenly into samples: "
            f"rows={M}, num_samples={num_samples}."
        )
    num_blocks = D // group_size
    rows_per_sample = M // num_samples
    samples = flat.reshape(num_samples, rows_per_sample, D)

    global_amax = samples.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    s_global = global_amax / (FP8_E4M3_MAX * FP4_E2M1_MAX)

    blocks = samples.reshape(
        num_samples, rows_per_sample, num_blocks, group_size
    )
    block_amax = blocks.abs().amax(dim=3).clamp_min(1e-12)
    s_block = _cast_fp8_e4m3(
        (block_amax / FP4_E2M1_MAX) / s_global
    ).clamp_min(1e-12)

    effective_scale = s_global * s_block
    scaled = blocks / effective_scale.unsqueeze(3)
    _, q_vals = _fp4_e2m1_quantize(scaled)
    dequant = q_vals * effective_scale.unsqueeze(3)
    return dequant.reshape(orig_shape)


# --------------------------------------------------------------------------- #
# QuantizedWeight + format-aware scale / round helpers.                        #
# --------------------------------------------------------------------------- #


@dataclass
class QuantizedWeight:
    """Single result type for every quantizer; mirrors the pack on-disk layout."""

    qweight: torch.Tensor       # (N, K) int8 codes
    weight_scale: torch.Tensor  # (N,) / (N, num_groups) / NVFP4 s_global (1,)
    group_size: int             # -1 ⇒ per-channel; NVFP4 uses 16
    weight_bits: int = 4
    # "int" = signed symmetric integer grid
    # "fp"  = E2M1 + single-level per-group scale
    # "nvfp" = official NVFP4 (E2M1 + s_block FP8 + s_global FP32)
    weight_format: str = "int"
    residual: torch.Tensor | None = None  # (N, K) or None
    fp_weight: torch.Tensor | None = None  # (N, K) float32
    # NVFP4 per-block FP8 scales ``(N, K/16)``; None for int/fp.
    weight_scale_2: torch.Tensor | None = None


def _int_fp_grid_max(weight_format: str, weight_bits: int) -> float:
    """Max representable value on the single-level quantization grid.

    Used by int / plain-fp scale fitting. NVFP4 uses two-level scales and
    must not go through this helper.
    """
    if weight_format == "fp":
        return FP4_E2M1_MAX
    if weight_format == "int":
        _, qmax = symmetric_quant_range(weight_bits)
        return float(qmax)
    if weight_format == "nvfp":
        raise ValueError(
            "_int_fp_grid_max does not apply to weight_format='nvfp'; "
            "use NVFP4 two-level scales (s_global / s_block) instead."
        )
    raise ValueError(
        f"weight_format must be 'int', 'fp', or 'nvfp', got {weight_format!r}."
    )


def _int_fp_compute_scale(
    w: torch.Tensor,
    group_size: int,
    *,
    weight_bits: int,
    weight_format: str = "int",
) -> torch.Tensor:
    """Per-group amax-based scale for int / plain-fp formats."""
    gmax = _int_fp_grid_max(weight_format, weight_bits)
    N, K = w.shape
    if group_size <= 0 or group_size >= K:
        amax = w.abs().amax(dim=1).clamp_min_(1e-12)
        return (amax / gmax).contiguous()
    if K % group_size != 0:
        raise ValueError(
            f"K={K} not divisible by group_size={group_size}; "
            "either pick a divisor or set group_size=-1 (per-channel)."
        )
    num_groups = K // group_size
    grouped = w.reshape(N, num_groups, group_size)
    amax = grouped.abs().amax(dim=2).clamp_min_(1e-12)
    return (amax / gmax).contiguous()


def _int_fp_quantize(
    w: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
    *,
    weight_bits: int,
    weight_format: str = "int",
) -> torch.Tensor:
    """Round ``w / scale`` to the nearest grid point. Returns int8 codes.

    Single-level formats only (``int`` / ``fp``). NVFP4 uses
    :func:`nvfp4_quantize` instead.
    """
    del group_size  # scale already encodes grouping; kept for call-site symmetry.
    N, K = w.shape
    if scale.ndim == 1:
        scaled = w / scale.unsqueeze(1)
    else:
        num_groups = scale.shape[1]
        gs = K // num_groups
        scaled = (w.reshape(N, num_groups, gs) / scale.unsqueeze(2)).reshape(N, K)

    if weight_format == "fp":
        codes, _ = _fp4_e2m1_quantize(scaled)
        return codes
    if weight_format == "int":
        qmin, qmax = symmetric_quant_range(weight_bits)
        return torch.round(scaled).clamp_(qmin, qmax).to(torch.int8)
    if weight_format == "nvfp":
        raise ValueError(
            "_int_fp_quantize does not apply to weight_format='nvfp'; "
            "use nvfp4_quantize instead."
        )
    raise ValueError(
        f"weight_format must be 'int', 'fp', or 'nvfp', got {weight_format!r}."
    )


def _int_fp_dequantize(
    codes: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
    *,
    weight_format: str = "int",
) -> torch.Tensor:
    """Inverse of :func:`_int_fp_quantize`. Returns float32.

    Single-level formats only (``int`` / ``fp``). NVFP4 uses
    :func:`nvfp4_dequantize` instead.
    """
    del group_size
    N, K = codes.shape
    if weight_format == "fp":
        vals = _fp4_e2m1_dequantize(codes, device=codes.device)
    elif weight_format == "int":
        vals = codes.to(torch.float32)
    elif weight_format == "nvfp":
        raise ValueError(
            "_int_fp_dequantize does not apply to weight_format='nvfp'; "
            "use nvfp4_dequantize instead."
        )
    else:
        raise ValueError(
            f"weight_format must be 'int', 'fp', or 'nvfp', got {weight_format!r}."
        )

    if scale.ndim == 1:
        return vals * scale.unsqueeze(1)
    num_groups = scale.shape[1]
    gs = K // num_groups
    return (vals.reshape(N, num_groups, gs) * scale.unsqueeze(2)).reshape(N, K)


# --------------------------------------------------------------------------- #
# RTN — round-to-nearest.                                                     #
# --------------------------------------------------------------------------- #


def rtn_quantize(
    w: torch.Tensor,
    *,
    group_size: int = 128,
    weight_bits: int = 4,
    weight_format: str = "int",
) -> QuantizedWeight:
    """Plain round-to-nearest. The fastest baseline."""
    if weight_format == "nvfp":
        return nvfp4_quantize(w, group_size=group_size)
    w_fp32 = w.detach().to(torch.float32)
    scale = _int_fp_compute_scale(
        w_fp32, group_size, weight_bits=weight_bits, weight_format=weight_format,
    )
    qweight = _int_fp_quantize(
        w_fp32, scale, group_size,
        weight_bits=weight_bits, weight_format=weight_format,
    )
    return QuantizedWeight(
        qweight=qweight,
        weight_scale=scale,
        group_size=group_size,
        weight_bits=weight_bits,
        weight_format=weight_format,
    )


def rtn_residual_quantize(
    w: torch.Tensor,
    *,
    group_size: int = 128,
    keep_top_k_outlier_cols: int = 0,
    weight_bits: int = 4,
    weight_format: str = "int",
) -> QuantizedWeight:
    """RTN with a low-rank-ish residual to absorb the brightest outliers.

      W ≈ dequant(RTN(W - Δ)) + Δ

    where ``Δ`` stores the ``keep_top_k`` columns of ``W`` with the largest
    L2 norm at full precision. Passing ``keep_top_k_outlier_cols=0`` (the
    default) makes this identical to plain RTN.
    """
    w_fp32 = w.detach().to(torch.float32)
    N, K = w_fp32.shape

    residual = torch.zeros_like(w_fp32) if keep_top_k_outlier_cols > 0 else None
    if keep_top_k_outlier_cols > 0:
        col_norms = w_fp32.pow(2).sum(dim=0)
        top_k = min(keep_top_k_outlier_cols, K)
        idx = torch.topk(col_norms, top_k).indices
        residual[:, idx] = w_fp32[:, idx]
        w_quantizable = w_fp32.clone()
        w_quantizable[:, idx] = 0.0
    else:
        w_quantizable = w_fp32

    if weight_format == "nvfp":
        qw = nvfp4_quantize(w_quantizable, group_size=group_size)
        return QuantizedWeight(
            qweight=qw.qweight,
            weight_scale=qw.weight_scale,
            group_size=qw.group_size,
            weight_bits=4,
            weight_format="nvfp",
            residual=residual.to(torch.float16) if residual is not None else None,
            weight_scale_2=qw.weight_scale_2,
        )

    scale = _int_fp_compute_scale(
        w_quantizable, group_size, weight_bits=weight_bits, weight_format=weight_format,
    )
    qweight = _int_fp_quantize(
        w_quantizable, scale, group_size,
        weight_bits=weight_bits, weight_format=weight_format,
    )
    return QuantizedWeight(
        qweight=qweight,
        weight_scale=scale,
        group_size=group_size,
        weight_bits=weight_bits,
        weight_format=weight_format,
        residual=residual.to(torch.float16) if residual is not None else None,
    )


# --------------------------------------------------------------------------- #
# GPTQ helpers                                                              #
# --------------------------------------------------------------------------- #


def _hessian_inv_chol(
    H: torch.Tensor,
    *,
    damp_percent: float,
    max_retries: int = 6,
) -> torch.Tensor:
    """Upper-triangular Cholesky factor of ``H^{-1}`` for the GPTQ inner loop.

    Calibration Hessians are only numerically PSD; symmetrize, damp the
    diagonal, and retry with stronger damping before falling back to an
    eigenvalue floor so Cholesky never aborts the whole build.
    """
    device = H.device
    K = H.shape[0]
    H = 0.5 * (H + H.T)
    diag_mean = float(torch.diag(H).mean().clamp_min(1e-12))

    for attempt in range(max_retries):
        damp = diag_mean * damp_percent * (10.0 ** attempt)
        H_try = H + damp * torch.eye(K, device=device, dtype=H.dtype)
        try:
            L = torch.linalg.cholesky(H_try)
            Hinv = torch.cholesky_inverse(L)
            return torch.linalg.cholesky(Hinv, upper=True)
        except RuntimeError:
            continue

    eigvals, eigvecs = torch.linalg.eigh(H)
    floor = diag_mean * damp_percent
    eigvals = eigvals.clamp_min(floor)
    H_psd = (eigvecs * eigvals.unsqueeze(0)) @ eigvecs.T
    L = torch.linalg.cholesky(H_psd)
    Hinv = torch.cholesky_inverse(L)
    return torch.linalg.cholesky(Hinv, upper=True)


# --------------------------------------------------------------------------- #
# GPTQ — Hessian error-compensated quantization.                              #
# --------------------------------------------------------------------------- #


def _quantize_col(
    w_col: torch.Tensor,
    s: torch.Tensor,
    weight_format: str,
    weight_bits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a single column ``(N,)`` and return ``(codes_int8, dequant_fp32)``.

    ``s`` is the per-row scale already applied to this column:
    - ``int`` / ``fp``: the single-level group scale
    - ``nvfp``: the effective scale ``s_global * s_block``
    """
    scaled = w_col / s
    if weight_format in ("fp", "nvfp"):
        codes, q_vals = _fp4_e2m1_quantize(scaled)
        return codes, q_vals * s
    if weight_format == "int":
        qmin, qmax = symmetric_quant_range(weight_bits)
        q = torch.round(scaled).clamp_(qmin, qmax)
        return q.to(torch.int8), q * s
    raise ValueError(
        f"weight_format must be 'int', 'fp', or 'nvfp', got {weight_format!r}."
    )


def gptq_quantize(
    w: torch.Tensor,
    hessian: torch.Tensor,
    *,
    group_size: int = 128,
    block_size: int = 128,
    damp_percent: float = 0.01,
    weight_bits: int = 4,
    weight_format: str = "int",
) -> QuantizedWeight:
    """GPTQ from Frantar et al. 2022.

    Supports ``int``, plain ``fp`` (E2M1 + per-group scale), and official
    ``nvfp`` (E2M1 + FP8 block scale + FP32 global scale). The error-
    compensation loop is grid-agnostic.
    """
    if w.ndim != 2:
        raise ValueError(f"gptq_quantize expects (N, K), got {tuple(w.shape)}.")
    if hessian.shape != (w.shape[1], w.shape[1]):
        raise ValueError(
            f"hessian shape {tuple(hessian.shape)} doesn't match "
            f"weight K={w.shape[1]}."
        )
    device = w.device
    N, K = w.shape
    use_nvfp = weight_format == "nvfp"
    if use_nvfp:
        _validate_nvfp4_group_size(group_size)
    elif weight_format in ("int", "fp"):
        gmax = _int_fp_grid_max(weight_format, weight_bits)
    else:
        raise ValueError(
            f"weight_format must be 'int', 'fp', or 'nvfp', got {weight_format!r}."
        )

    W = w.to(torch.float32).clone()
    H = hessian.to(torch.float32).clone()

    dead = torch.diag(H) == 0.0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0

    Hinv_chol = _hessian_inv_chol(H, damp_percent=damp_percent)

    if group_size <= 0:
        group_size = K
    if K % group_size != 0:
        raise ValueError(f"K={K} not divisible by group_size={group_size}.")
    num_groups = K // group_size

    if use_nvfp:
        global_amax = W.abs().amax().clamp_min(1e-12)
        s_global = (global_amax / (FP8_E4M3_MAX * FP4_E2M1_MAX)).reshape(1)
        s_block = torch.zeros(N, num_groups, dtype=torch.float32, device=device)
        scales = None
    else:
        s_global = None
        s_block = None
        scales = torch.zeros(N, num_groups, dtype=torch.float32, device=device)

    qweight = torch.zeros(N, K, dtype=torch.int8, device=device)

    for col_start in range(0, K, block_size):
        col_end = min(col_start + block_size, K)
        block_W = W[:, col_start:col_end].clone()
        block_Q = torch.zeros_like(block_W, dtype=torch.int8)
        block_err = torch.zeros_like(block_W)
        block_Hinv = Hinv_chol[col_start:col_end, col_start:col_end].clone()

        for j in range(col_end - col_start):
            col_idx = col_start + j
            w_col = block_W[:, j].clone()
            d_jj = block_Hinv[j, j].clamp_min(1e-12)
            grp = col_idx // group_size

            if col_idx % group_size == 0:
                g_end = min(col_idx + group_size, K)
                grp_view = W[:, col_idx:g_end]
                if use_nvfp:
                    # s_global is frozen from the initial W (codes already
                    # emitted for prior groups use it). Clamp the block scale
                    # argument so GPTQ error growth cannot overflow FP8.
                    block_amax = grp_view.abs().amax(dim=1).clamp_min(1e-12)
                    raw = (block_amax / FP4_E2M1_MAX) / s_global
                    s_block[:, grp] = _cast_fp8_e4m3(raw).clamp_min(1e-12)
                else:
                    amax = grp_view.abs().amax(dim=1).clamp_min_(1e-12)
                    scales[:, grp] = amax / gmax

            if use_nvfp:
                s = (s_global.reshape(1) * s_block[:, grp]).reshape(N)
            else:
                s = scales[:, grp]

            codes, w_dequant = _quantize_col(w_col, s, weight_format, weight_bits)
            block_Q[:, j] = codes

            err_col = (w_col - w_dequant) / d_jj
            block_err[:, j] = err_col

            if j + 1 < (col_end - col_start):
                block_W[:, j + 1:] -= (
                    err_col.unsqueeze(1) * block_Hinv[j, j + 1:].unsqueeze(0)
                )

        qweight[:, col_start:col_end] = block_Q

        if col_end < K:
            W[:, col_end:] -= block_err @ Hinv_chol[col_start:col_end, col_end:]

    if use_nvfp:
        return QuantizedWeight(
            qweight=qweight.contiguous(),
            weight_scale=s_global.contiguous(),
            group_size=group_size,
            weight_bits=weight_bits,
            weight_format="nvfp",
            weight_scale_2=s_block.contiguous(),
        )
    return QuantizedWeight(
        qweight=qweight.contiguous(),
        weight_scale=scales.contiguous(),
        group_size=group_size,
        weight_bits=weight_bits,
        weight_format=weight_format,
    )




# --------------------------------------------------------------------------- #
# Convenience facade used by the builder.                                     #
# --------------------------------------------------------------------------- #


def no_quantize(w: torch.Tensor) -> QuantizedWeight:
    """Store a rotated weight at full precision (``weight_bits == 16``)."""
    w_fp32 = w.detach().to(torch.float32)
    n, k = w_fp32.shape
    return QuantizedWeight(
        qweight=torch.zeros(n, k, dtype=torch.int8),
        weight_scale=torch.ones(n, dtype=torch.float32),
        group_size=-1,
        weight_bits=NO_QUANT_BITS,
        residual=None,
        fp_weight=w_fp32.contiguous(),
    )


def quantize_weight(
    w: torch.Tensor,
    quantizer: str,
    *,
    group_size: int,
    weight_bits: int = 4,
    weight_format: str = "int",
    hessian: torch.Tensor | None = None,
    gptq_block_size: int = 128,
    gptq_damp_percent: float = 0.01,
    rtn_residual_top_k: int = 8,
) -> QuantizedWeight:
    """Dispatch to the requested quantizer.

    ``quantizer`` is one of ``"rtn"``, ``"rtn_residual"``, ``"gptq"``.
    ``weight_format`` selects the number grid (``"int"``/``"fp"``/``"nvfp"``).
    All three quantizers support all formats — GPTQ's error-compensation loop
    is grid-agnostic.
    """
    if is_no_quant(weight_bits):
        return no_quantize(w)

    if quantizer == "rtn":
        return rtn_quantize(
            w, group_size=group_size, weight_bits=weight_bits,
            weight_format=weight_format,
        )
    if quantizer == "rtn_residual":
        return rtn_residual_quantize(
            w,
            group_size=group_size,
            keep_top_k_outlier_cols=rtn_residual_top_k,
            weight_bits=weight_bits,
            weight_format=weight_format,
        )
    if quantizer == "gptq":
        if hessian is None:
            raise ValueError("gptq quantizer requires a `hessian` matrix.")
        return gptq_quantize(
            w, hessian,
            group_size=group_size,
            block_size=gptq_block_size,
            damp_percent=gptq_damp_percent,
            weight_bits=weight_bits,
            weight_format=weight_format,
        )
    raise ValueError(f"Unknown quantizer: {quantizer!r}")


def dequantize(qw: QuantizedWeight) -> torch.Tensor:
    """Bring a :class:`QuantizedWeight` back into a dense float32 tensor.

    Used by the runtime layer when it falls back to bf16 matmul, and by tests
    to compute reconstruction error.
    """
    if is_no_quant(qw.weight_bits):
        if qw.fp_weight is None:
            raise ValueError(
                f"weight_bits={qw.weight_bits} requires fp_weight but none was stored."
            )
        return qw.fp_weight.to(torch.float32)

    fmt = getattr(qw, "weight_format", "int")
    if fmt == "nvfp":
        if qw.weight_scale_2 is None:
            raise ValueError("NVFP4 dequant requires weight_scale_2 (s_block).")
        out = nvfp4_dequantize(
            qw.qweight,
            qw.weight_scale,
            qw.weight_scale_2,
            group_size=qw.group_size,
        )
    elif fmt in ("int", "fp"):
        out = _int_fp_dequantize(
            qw.qweight,
            qw.weight_scale,
            qw.group_size,
            weight_format=fmt,
        )
    else:
        raise ValueError(
            f"weight_format must be 'int', 'fp', or 'nvfp', got {fmt!r}."
        )
    if qw.residual is not None:
        out = out + qw.residual.to(torch.float32)
    return out


__all__ = [
    "FP4_E2M1_MAX",
    "FP8_E4M3_MAX",
    "NO_QUANT_BITS",
    "NVFP4_BLOCK_SIZE",
    "QuantizedWeight",
    "dequantize",
    "gptq_quantize",
    "is_no_quant",
    "no_quantize",
    "nvfp4_dequantize",
    "nvfp4_quantize",
    "nvfp4_quantize_activation",
    "channel_percentile_amax",
    "token_percentile_amax",
    "percentile_amax",
    "quantize_weight",
    "rtn_quantize",
    "rtn_residual_quantize",
    "symmetric_quant_range",
]
