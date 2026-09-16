"""Low-level rotation / transform math primitives.

Hadamard matrices, SVD block rotations, zigzag permutations, and the
block-diagonal matmul that underlies all per-block transforms.

Higher-level pipeline orchestration lives in :mod:`qvla.core.pipeline`.
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Literal

import torch

PermScore = Literal["weight", "activation", "activation_weight", "fisher"]
SvdSource = Literal["weight", "activation"]
PipelineStep = Literal[
    "clip", "smooth", "perm", "svd", "hadamard", "random_hadamard"
]

logger = logging.getLogger(__name__)


def is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def validate_block(d: int, block_size: int) -> int:
    if not is_pow2(block_size):
        raise ValueError(f"block_size={block_size} must be a power of two.")
    if d % block_size != 0:
        raise ValueError(
            f"d={d} not divisible by block_size={block_size}; pick a smaller "
            "block (QVLA padding logic is intentionally not implemented "
            "here to keep the math obvious; channel-pad upstream if you need it)."
        )
    return d // block_size


def hadamard_matrix(n: int, *, dtype: torch.dtype = torch.float32, device=None) -> torch.Tensor:
    """Sylvester construction of an `n×n` Hadamard matrix, normalized to be orthonormal."""
    if not is_pow2(n):
        raise ValueError(f"hadamard_matrix(n={n}): n must be a power of two.")
    h = torch.ones(1, 1, dtype=dtype, device=device)
    while h.shape[0] < n:
        h = torch.cat(
            [torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)],
            dim=0,
        )
    return h / math.sqrt(n)


def random_hadamard_matrix(
    n: int,
    *,
    dtype: torch.dtype = torch.float32,
    device=None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Random orthogonal Hadamard: ``H @ diag(±1)`` (Sylvester ``H`` + random sign flips)."""
    h = hadamard_matrix(n, dtype=dtype, device=device)
    signs = torch.empty(n, dtype=dtype, device=device)
    signs.bernoulli_(0.5, generator=generator)
    signs = signs.mul(2).sub(1).to(dtype=dtype)
    return h * signs.unsqueeze(0)


def random_hadamard_seed(layer_name: str, build_seed: int) -> int:
    """Per-layer seed for random Hadamard (stable across runs and RNG draw order)."""
    digest = hashlib.sha256(f"{build_seed}:{layer_name}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def standard_hadamard_blocks(
    num_blocks: int,
    block_size: int,
    *,
    device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    h = hadamard_matrix(block_size, dtype=dtype, device=device)
    return h.unsqueeze(0).expand(num_blocks, -1, -1).contiguous()


def random_hadamard_blocks(
    num_blocks: int,
    block_size: int,
    *,
    device,
    dtype: torch.dtype = torch.float32,
    seed: int | None = None,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    h = random_hadamard_matrix(
        block_size, dtype=dtype, device=device, generator=generator
    )
    return h.unsqueeze(0).expand(num_blocks, -1, -1).contiguous()


def apply_block_matmul(
    x: torch.Tensor,
    blocks: torch.Tensor,
    *,
    block_size: int,
    d: int,
) -> torch.Tensor:
    """Block-diagonal matmul along the last axis: ``x @ block_diag(blocks)``.

    Works for activations ``(..., d)`` and weights ``(out_features, d)``.
    """
    b = block_size
    nb = d // b
    if x.shape[-1] != d:
        raise ValueError(f"Expected last dim {d}, got {x.shape[-1]}.")
    lead = x.shape[:-1]
    flat = x.reshape(-1, nb, b)
    blk = blocks.to(device=flat.device, dtype=flat.dtype)
    rotated = torch.einsum("tnb,nbc->tnc", flat, blk)
    return rotated.reshape(*lead, d)


def zigzag_permutation_omega_qvla(energy: torch.Tensor) -> torch.Tensor:
    """Omega-QVLA global interleave permutation (debug / legacy reference).

    Matches ``zigzag_permutation`` in Omega-QVLA ``gr00t/quantization/duquant_preprocess.py``:
    sort channels by energy descending, then alternate picking from the largest
    and smallest ends of the sorted list (slot 0 = max, slot 1 = min, …).

    Not wired into production pipelines; kept for debugging against Omega-QVLA packs.
    """
    if energy.ndim != 1:
        raise ValueError(f"energy must be 1-D, got shape {tuple(energy.shape)}.")
    order = torch.argsort(energy, descending=True)
    n = order.numel()
    perm: list[int] = []
    left, right = 0, n - 1
    toggle = True
    while left <= right:
        if toggle:
            perm.append(int(order[left].item()))
            left += 1
        else:
            perm.append(int(order[right].item()))
            right -= 1
        toggle = not toggle
    return torch.tensor(perm, dtype=torch.int64, device=energy.device)


def zigzag_permutation(energy: torch.Tensor, *, block_size: int) -> torch.Tensor:
    """DuQuant zigzag channel reorder from per-channel energy (descending sort).

    Sorted channels are assigned to blocks in a back-and-forth pattern (block 0,
    1, …, K-1, K-2, …); within each block, higher-energy channels occupy earlier
    slots.  Matches ``Quantizer.permutation_zigzag`` in the official DuQuant repo.
    """
    if energy.ndim != 1:
        raise ValueError(f"energy must be 1-D, got shape {tuple(energy.shape)}.")
    n = energy.numel()
    if n % block_size != 0:
        raise ValueError(
            f"energy length {n} must be divisible by block_size {block_size}."
        )
    num_blocks = n // block_size

    order = torch.argsort(energy, descending=True)
    sorted_pairs: list[tuple[int, float]] = [
        (int(order[i].item()), float(energy[order[i]].item())) for i in range(n)
    ]

    blocks: list[list[tuple[int, float]]] = [[] for _ in range(num_blocks)]
    cur = 0
    up = True
    for pair in sorted_pairs:
        blocks[cur].append(pair)
        if up:
            cur += 1
            if cur == num_blocks:
                cur -= 1
                up = False
        else:
            cur -= 1
            if cur == -1:
                cur += 1
                up = True

    perm = torch.empty(n, dtype=torch.int64, device=energy.device)
    for block_idx, block_pairs in enumerate(blocks):
        block_pairs.sort(key=lambda item: item[1], reverse=True)
        start = block_idx * block_size
        perm[start : start + block_size] = torch.tensor(
            [ch for ch, _ in block_pairs],
            dtype=torch.int64,
            device=energy.device,
        )
    return perm


def compute_perm_energy(
    weight: torch.Tensor,
    *,
    perm_score: PermScore,
    activation_amax: torch.Tensor | None = None,
    sensitivity: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, str]:
    """Per-input-channel energy driving zigzag permutation."""
    if perm_score == "fisher":
        if sensitivity is None:
            raise ValueError(
                "perm_score='fisher' requires Fisher sensitivity from calibration."
            )
        return sensitivity.detach().to(torch.float32).clamp_min(eps), "fisher"
    if perm_score not in ("weight", "activation", "activation_weight"):
        raise ValueError(
            f"perm_score must be weight|activation|activation_weight|fisher, got {perm_score!r}."
        )
    score_used = perm_score
    if perm_score in ("activation", "activation_weight") and activation_amax is None:
        raise ValueError(
            f"perm_score={perm_score!r} requires activation_amax from calibration stats."
        )

    w = weight.detach().to(torch.float32)
    if perm_score == "weight":
        energy = (w * w).mean(dim=0)
    elif perm_score == "activation":
        energy = activation_amax.detach().to(torch.float32).pow(2)
    else:
        a = activation_amax.detach().to(torch.float32).pow(2)
        energy = a * (w * w).mean(dim=0)

    return energy, score_used


def compute_block_rotation_from_weight(
    weight_block: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Orthonormal ``U`` from SVD of ``weight_block.T`` (DuQuant default).

    ``weight_block`` has shape ``(out_features, block_size)``.
    """
    x = weight_block.T.to(torch.float32)  # (B, out)
    b = x.shape[0]
    u, _, _ = torch.linalg.svd(x, full_matrices=False)
    if u.shape[1] < b:
        pad = torch.zeros(b, b - u.shape[1], dtype=u.dtype, device=u.device)
        u = torch.cat([u, pad], dim=1)
    return u[:, :b].contiguous()


def compute_block_rotation_from_activation_cov(
    activation_cov_block: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Orthonormal eigenbasis from an activation covariance block (ablation path).

    ``activation_cov_block`` has shape ``(block_size, block_size)`` (symmetric PSD).
    """
    block_size = activation_cov_block.shape[0]
    _, eigvecs = torch.linalg.eigh(
        activation_cov_block.to(torch.float32)
        + eps * torch.eye(block_size, device=activation_cov_block.device)
    )
    return eigvecs.contiguous()


def u_for_block(
    weight_block: torch.Tensor | None,
    sigma_b: torch.Tensor | None,
    *,
    svd_source: SvdSource,
    eps: float,
) -> torch.Tensor:
    """One block ``U`` matrix ``(block_size, block_size)`` — Hadamard applied later."""
    if svd_source == "weight":
        if weight_block is None:
            raise ValueError("weight SVD requires weight_block.")
        return compute_block_rotation_from_weight(weight_block, eps=eps)
    if sigma_b is None:
        raise ValueError("activation SVD requires activation covariance block.")
    return compute_block_rotation_from_activation_cov(sigma_b, eps=eps)


def fit_u_blocks(
    *,
    d: int,
    block_size: int,
    weight: torch.Tensor,
    activation_cov: torch.Tensor | None,
    svd_source: SvdSource,
    perm: torch.Tensor | None,
    sensitivity: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    """Fit per-block ``U`` (no Hadamard) on the current column layout.

    ``W`` is always in original channel order. When ``perm`` is set, weight
    blocks use ``W[:, perm[block]]``.

    ``activation_cov`` must be in the **SVD step input layout** — i.e. collected
    on ``prefix.apply(x)`` where ``prefix`` is the pipeline prefix before SVD.
    Block ``i`` uses ``cov[i*bs:(i+1)*bs, i*bs:(i+1)*bs]``. The builder
    guarantees this via pipeline-wise calibration passes.
    """
    num_blocks = validate_block(d, block_size)
    w = weight.detach().to(torch.float32)
    cov_fp32 = (
        activation_cov.to(torch.float32)
        if activation_cov is not None and svd_source == "activation"
        else None
    )

    u_blocks = torch.empty(
        num_blocks, block_size, block_size, dtype=torch.float32, device=w.device
    )
    for i in range(num_blocks):
        start = i * block_size
        end = (i + 1) * block_size
        slot = torch.arange(start, end, device=w.device)
        w_cols = perm[slot] if perm is not None else slot
        w_block = w[:, w_cols]

        sigma_b = None
        if cov_fp32 is not None:
            cov_cols = slot
            sigma_b = cov_fp32[cov_cols][:, cov_cols]
            if sensitivity is not None:
                sens = sensitivity.detach().to(torch.float32).clamp_min(eps)[w_cols]
                d_sqrt = sens.sqrt()
                sigma_b = sigma_b * d_sqrt.unsqueeze(1) * d_sqrt.unsqueeze(0)

        u_blocks[i] = u_for_block(
            w_block,
            sigma_b,
            svd_source=svd_source,
            eps=eps,
        )

    return u_blocks.contiguous()


def zigzag_from_stats(
    weight: torch.Tensor,
    *,
    u_blocks: torch.Tensor | None,
    pre_perm_pipeline: tuple[PipelineStep, ...],
    block_size: int,
    d: int,
    perm_score: PermScore,
    activation_amax: torch.Tensor | None,
    sensitivity: torch.Tensor | None,
    eps: float,
    random_hadamard_blocks: torch.Tensor | None = None,
    act_clip: torch.Tensor | None = None,
    smooth_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, str]:
    from qvla.core.pipeline import apply_input_pipeline

    del act_clip  # activation-only; never applied to weight for perm scoring

    w_for_perm = weight
    # Weight-side only: skip activation-only ``clip``. (Smooth never appears
    # before ``perm`` in ALLOWED_PIPELINES; orthogonal prefix steps are shared.)
    weight_pipeline = tuple(s for s in pre_perm_pipeline if s != "clip")
    if weight_pipeline and perm_score in ("weight", "activation_weight"):
        w_for_perm = apply_input_pipeline(
            weight.to(torch.float32),
            pipeline=weight_pipeline,
            u_blocks=u_blocks,
            perm=None,
            block_size=block_size,
            d=d,
            random_hadamard_blocks=random_hadamard_blocks,
            act_clip=None,
            smooth_scale=smooth_scale,
        ).contiguous()
    energy, score_used = compute_perm_energy(
        w_for_perm,
        perm_score=perm_score,
        activation_amax=activation_amax,
        sensitivity=sensitivity,
        eps=eps,
    )
    return zigzag_permutation(energy, block_size=block_size), score_used


__all__ = [
    "PermScore",
    "PipelineStep",
    "SvdSource",
    "apply_block_matmul",
    "compute_block_rotation_from_activation_cov",
    "compute_block_rotation_from_weight",
    "compute_perm_energy",
    "fit_u_blocks",
    "hadamard_matrix",
    "is_pow2",
    "random_hadamard_blocks",
    "random_hadamard_matrix",
    "random_hadamard_seed",
    "standard_hadamard_blocks",
    "u_for_block",
    "validate_block",
    "zigzag_from_stats",
    "zigzag_permutation",
    "zigzag_permutation_omega_qvla",
]
