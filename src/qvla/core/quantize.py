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


def channel_percentile_amax(
    abs_activations: torch.Tensor, percentile: float
) -> torch.Tensor:
    """Per-channel ``percentile`` of ``|activations|`` along the token axis."""
    if abs_activations.ndim != 2:
        raise ValueError(
            f"channel_percentile_amax expects (num_tokens, in_features), "
            f"got {tuple(abs_activations.shape)}."
        )
    if percentile >= 100.0:
        return abs_activations.abs().amax(dim=0)
    return torch.quantile(
        abs_activations.to(torch.float32), percentile / 100.0, dim=0
    )


# --------------------------------------------------------------------------- #
# RTN — round-to-nearest.                                                     #
# --------------------------------------------------------------------------- #


@dataclass
class QuantizedWeight:
    """Single result type for every quantizer; mirrors the pack on-disk layout."""

    qweight: torch.Tensor       # (N, K) int8 in [qmin, qmax] for weight_bits
    weight_scale: torch.Tensor  # (N,) or (N, num_groups) float32
    group_size: int             # -1 ⇒ per-channel
    weight_bits: int = 4
    # Optional residual term used by RTN-residual: a low-rank correction
    # holding the part of the weight that the int4 grid couldn't capture.
    # Stored at bf16 / fp32; cheap because rank is tiny.
    residual: torch.Tensor | None = None  # (N, K) or None
    # Full-precision rotated weight when ``weight_bits == NO_QUANT_BITS``.
    fp_weight: torch.Tensor | None = None  # (N, K) float32


def _compute_scale(
    w: torch.Tensor, group_size: int, *, weight_bits: int
) -> torch.Tensor:
    """Per-row amax-based scale; reshape into groups if requested.

    For symmetric quantization the scale is just ``amax / qmax`` so that the
    largest magnitude in each group lands exactly on the quant grid edge.
    """
    _, qmax = symmetric_quant_range(weight_bits)
    N, K = w.shape
    if group_size <= 0 or group_size >= K:
        amax = w.abs().amax(dim=1).clamp_min_(1e-12)
        return (amax / qmax).contiguous()  # (N,)
    if K % group_size != 0:
        raise ValueError(
            f"K={K} not divisible by group_size={group_size}; "
            "either pick a divisor or set group_size=-1 (per-channel)."
        )
    num_groups = K // group_size
    grouped = w.reshape(N, num_groups, group_size)
    amax = grouped.abs().amax(dim=2).clamp_min_(1e-12)
    return (amax / qmax).contiguous()  # (N, num_groups)


def _quantize(
    w: torch.Tensor, scale: torch.Tensor, group_size: int, *, weight_bits: int
) -> torch.Tensor:
    """Apply a precomputed scale and round to the symmetric ``weight_bits`` grid."""
    qmin, qmax = symmetric_quant_range(weight_bits)
    N, K = w.shape
    if scale.ndim == 1:  # per-channel
        s = scale.unsqueeze(1)  # (N, 1)
        q = torch.round(w / s).clamp_(qmin, qmax).to(torch.int8)
        return q
    # per-group
    num_groups = scale.shape[1]
    assert num_groups * group_size == K
    grouped = w.reshape(N, num_groups, group_size)
    q = torch.round(grouped / scale.unsqueeze(2)).clamp_(qmin, qmax).to(torch.int8)
    return q.reshape(N, K)


def rtn_quantize(
    w: torch.Tensor, *, group_size: int = 128, weight_bits: int = 4
) -> QuantizedWeight:
    """Plain round-to-nearest. The fastest baseline."""
    w_fp32 = w.detach().to(torch.float32)
    scale = _compute_scale(w_fp32, group_size, weight_bits=weight_bits)
    qweight = _quantize(
        w_fp32, scale, group_size, weight_bits=weight_bits
    )
    return QuantizedWeight(
        qweight=qweight,
        weight_scale=scale,
        group_size=group_size,
        weight_bits=weight_bits,
        residual=None,
    )


def rtn_residual_quantize(
    w: torch.Tensor,
    *,
    group_size: int = 128,
    keep_top_k_outlier_cols: int = 0,
    weight_bits: int = 4,
) -> QuantizedWeight:
    """RTN with a low-rank-ish residual to absorb the brightest outliers.

    The QVLA README describes the DiT-side quantizer as "RTN residual".
    The minimal interpretation we adopt:

      W ≈ dequant(RTN(W - Δ)) + Δ

    where ``Δ`` is a sparse correction that stores the ``keep_top_k`` columns
    of ``W`` with the largest L2 norm at full precision (those are the columns
    that the rotation couldn't tame). The residual stores ``Δ`` densely
    (``(N, K)`` with zeros elsewhere); only the picked columns are non-zero so
    bf16 storage is cheap relative to a full bf16 matrix on disk.

    Passing ``keep_top_k_outlier_cols=0`` (the default) makes this identical to
    plain RTN; raise it to absorb whatever fraction of outliers your model
    needs. Setting it to a small positive integer like 8 is usually enough.
    """
    w_fp32 = w.detach().to(torch.float32)
    N, K = w_fp32.shape

    residual = torch.zeros_like(w_fp32) if keep_top_k_outlier_cols > 0 else None
    if keep_top_k_outlier_cols > 0:
        col_norms = w_fp32.pow(2).sum(dim=0)
        top_k = min(keep_top_k_outlier_cols, K)
        # `top_k` indices of largest-magnitude columns.
        idx = torch.topk(col_norms, top_k).indices
        residual[:, idx] = w_fp32[:, idx]
        w_quantizable = w_fp32.clone()
        w_quantizable[:, idx] = 0.0
    else:
        w_quantizable = w_fp32

    scale = _compute_scale(w_quantizable, group_size, weight_bits=weight_bits)
    qweight = _quantize(
        w_quantizable, scale, group_size, weight_bits=weight_bits
    )
    return QuantizedWeight(
        qweight=qweight,
        weight_scale=scale,
        group_size=group_size,
        weight_bits=weight_bits,
        # Stored as fp16 to halve disk size; pi05 runs at bf16 anyway.
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


def gptq_quantize(
    w: torch.Tensor,
    hessian: torch.Tensor,
    *,
    group_size: int = 128,
    block_size: int = 128,
    damp_percent: float = 0.01,
    weight_bits: int = 4,
) -> QuantizedWeight:
    """GPTQ from Frantar et al. 2022, written from scratch.

    Arguments:
        w: ``(N, K)`` float matrix — the rotated, full-precision weight.
        hessian: ``(K, K)`` symmetric PSD matrix — typically
            ``XᵀX / num_calibration_tokens`` of the post-rotation activations.
            Re-using the same matrix that fed the rotation fit is fine.
        group_size: width along K of each scale group. Use ``-1`` for
            per-channel.
        block_size: column-block size for the GPTQ inner loop. Smaller =
            slower but slightly more accurate; ``block_size == group_size`` is
            the standard sweet spot.
        damp_percent: diagonal damping fraction. ``0.01`` matches the QVLA
            paper default.

    Returns:
        :class:`QuantizedWeight`. The output dequantizes (``q · scale``) to a
        weight that is closer to the original than RTN in a Hessian-weighted
        sense.

    Implementation:
        1. Damp the Hessian and Cholesky-factor `H⁻¹`.
        2. Walk columns in blocks. For each block we maintain the in-block
           quantization error and propagate it to the unquantized columns
           on the right via the Cholesky-decomposed inverse Hessian.
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
    qmin, qmax = symmetric_quant_range(weight_bits)

    W = w.to(torch.float32).clone()
    H = hessian.to(torch.float32).clone()

    # 1. Dead columns: any K-th feature with zero diagonal Hessian carries no
    #    gradient information; set its weight column to zero so it doesn't
    #    blow up the inverse below.
    dead = torch.diag(H) == 0.0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0

    # 2. Damp the diagonal so the Cholesky is well-conditioned.
    # 3. Inverse-Cholesky for "look ahead and propagate quant error".
    Hinv_chol = _hessian_inv_chol(H, damp_percent=damp_percent)

    # 4. Quantize block by block.
    if group_size <= 0:
        group_size = K
    if K % group_size != 0:
        raise ValueError(f"K={K} not divisible by group_size={group_size}.")
    num_groups = K // group_size
    scales = torch.zeros(N, num_groups, dtype=torch.float32, device=device)
    qweight = torch.zeros(N, K, dtype=torch.int8, device=device)

    for col_start in range(0, K, block_size):
        col_end = min(col_start + block_size, K)
        block_W = W[:, col_start:col_end].clone()           # (N, b)
        block_Q = torch.zeros_like(block_W, dtype=torch.int8)
        block_err = torch.zeros_like(block_W)               # (N, b)
        block_Hinv = Hinv_chol[col_start:col_end, col_start:col_end].clone()  # (b, b)

        for j in range(col_end - col_start):
            col_idx = col_start + j
            w_col = block_W[:, j].clone()
            d_jj = block_Hinv[j, j].clamp_min(1e-12)

            # Pick this column's group scale on the first column of each group.
            if col_idx % group_size == 0:
                grp = col_idx // group_size
                g_start = col_idx
                g_end = min(col_idx + group_size, K)
                # The fitted scale uses the *current* (error-compensated) view
                # of the upcoming group — that's the GPTQ trick: by the time we
                # reach this column we've already absorbed the rounding error of
                # the previous columns into W[:, col_idx:].
                grp_view = W[:, g_start:g_end]
                amax = grp_view.abs().amax(dim=1).clamp_min_(1e-12)
                scales[:, grp] = amax / qmax
            s = scales[:, col_idx // group_size]  # (N,)

            q = torch.round(w_col / s).clamp_(qmin, qmax)
            block_Q[:, j] = q.to(torch.int8)

            # De-quant error for this column.
            err_col = (w_col - q * s) / d_jj  # (N,)
            block_err[:, j] = err_col

            # Propagate to remaining columns inside this block.
            if j + 1 < (col_end - col_start):
                block_W[:, j + 1 :] -= err_col.unsqueeze(1) * block_Hinv[j, j + 1 :].unsqueeze(0)

        qweight[:, col_start:col_end] = block_Q

        # Propagate the block's accumulated error to the *remaining* columns
        # outside this block (the look-ahead step).
        if col_end < K:
            W[:, col_end:] -= block_err @ Hinv_chol[col_start:col_end, col_end:]

    return QuantizedWeight(
        qweight=qweight.contiguous(),
        weight_scale=scales.contiguous(),
        group_size=group_size,
        weight_bits=weight_bits,
        residual=None,
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
    hessian: torch.Tensor | None = None,
    gptq_block_size: int = 128,
    gptq_damp_percent: float = 0.01,
    rtn_residual_top_k: int = 8,
) -> QuantizedWeight:
    """Dispatch to the requested quantizer.

    ``quantizer`` is one of ``"rtn"``, ``"rtn_residual"``, ``"gptq"``. ``gptq``
    requires ``hessian``; the others ignore it.

    When ``weight_bits >= NO_QUANT_BITS`` (16), the weight is stored at full
    precision and no quantizer is run.
    """
    if is_no_quant(weight_bits):
        return no_quantize(w)
    if quantizer == "rtn":
        return rtn_quantize(w, group_size=group_size, weight_bits=weight_bits)
    if quantizer == "rtn_residual":
        return rtn_residual_quantize(
            w,
            group_size=group_size,
            keep_top_k_outlier_cols=rtn_residual_top_k,
            weight_bits=weight_bits,
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
    q = qw.qweight.to(torch.float32)
    N, K = q.shape
    if qw.weight_scale.ndim == 1:
        out = q * qw.weight_scale.unsqueeze(1)
    else:
        num_groups = qw.weight_scale.shape[1]
        gs = K // num_groups
        out = (q.reshape(N, num_groups, gs) * qw.weight_scale.unsqueeze(2)).reshape(N, K)
    if qw.residual is not None:
        out = out + qw.residual.to(torch.float32)
    return out


__all__ = [
    "NO_QUANT_BITS",
    "QuantizedWeight",
    "dequantize",
    "gptq_quantize",
    "is_no_quant",
    "no_quantize",
    "channel_percentile_amax",
    "percentile_amax",
    "quantize_weight",
    "rtn_quantize",
    "rtn_residual_quantize",
    "symmetric_quant_range",
]
