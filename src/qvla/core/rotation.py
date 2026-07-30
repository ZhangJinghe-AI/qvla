"""DuQuant-style rotation matrices (QVLA / DuQuant composite transform).

The input-side transform is a configurable **pipeline** of steps applied
left-to-right at runtime:

* ``perm`` — zigzag channel reorder (optional)
* ``svd`` — per-block orthonormal ``U`` from weight / activation stats
* ``hadamard`` — standard Sylvester block Hadamard (deterministic, not stored)
* ``random_hadamard`` — random sign-flipped Hadamard ``H @ diag(±1)`` (fitted once, stored in pack)

``U`` is stored in ``u_blocks``. Calibration and runtime both follow the same
pipeline: steps not listed are skipped entirely.

All transforms preserve ``y = x Wᵀ`` when the same pipeline is applied to
``W`` offline (via :func:`apply_input_pipeline` / :meth:`Rotation.apply`).
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from qvla.build.collector import LayerStats

PermScore = Literal["weight", "activation", "activation_weight", "fisher"]
SvdSource = Literal["weight", "activation"]
PipelineStep = Literal["perm", "svd", "hadamard", "random_hadamard"]

_VALID_STEPS = frozenset({"perm", "svd", "hadamard", "random_hadamard"})

logger = logging.getLogger(__name__)


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def validate_pipeline(
    pipeline: tuple[PipelineStep, ...],
) -> tuple[PipelineStep, ...]:
    for step in pipeline:
        if step not in _VALID_STEPS:
            raise ValueError(
                f"Invalid pipeline step {step!r}; expected perm|svd|hadamard|random_hadamard."
            )
    return pipeline


def parse_pipeline_string(value: str) -> tuple[PipelineStep, ...]:
    """Parse CLI / JSON pipeline strings like ``perm,svd,hadamard``."""
    text = value.strip().lower()
    if text in ("", "none", "identity"):
        return ()
    steps = tuple(s.strip() for s in text.split(",") if s.strip())
    return validate_pipeline(steps)  # type: ignore[arg-type]


def hadamard_matrix(n: int, *, dtype: torch.dtype = torch.float32, device=None) -> torch.Tensor:
    """Sylvester construction of an `n×n` Hadamard matrix, normalized to be orthonormal."""
    if not _is_pow2(n):
        raise ValueError(f"hadamard_matrix(n={n}): n must be a power of two.")
    # torch.ones allocates directly on `device`; torch.tensor([[1.0]]) would
    # stage on CPU first and break CUDA graph capture (implicit H2D copy).
    h = torch.ones(1, 1, dtype=dtype, device=device)
    while h.shape[0] < n:
        h = torch.cat(
            [torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)],
            dim=0,
        )
    return h / math.sqrt(n)


def _random_hadamard_seed(layer_name: str, build_seed: int) -> int:
    """Per-layer seed for random Hadamard (stable across runs and RNG draw order)."""
    digest = hashlib.sha256(f"{build_seed}:{layer_name}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


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


def _validate_block(d: int, block_size: int) -> int:
    if not _is_pow2(block_size):
        raise ValueError(f"block_size={block_size} must be a power of two.")
    if d % block_size != 0:
        raise ValueError(
            f"d={d} not divisible by block_size={block_size}; pick a smaller "
            "block (QVLA padding logic is intentionally not implemented "
            "here to keep the math obvious; channel-pad upstream if you need it)."
        )
    return d // block_size


def _standard_hadamard_blocks(
    num_blocks: int,
    block_size: int,
    *,
    device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    h = hadamard_matrix(block_size, dtype=dtype, device=device)
    return h.unsqueeze(0).expand(num_blocks, -1, -1).contiguous()


def _random_hadamard_blocks(
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


def _apply_block_matmul(
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


def apply_input_pipeline(
    x: torch.Tensor,
    *,
    pipeline: tuple[PipelineStep, ...],
    u_blocks: torch.Tensor | None,
    perm: torch.Tensor | None,
    block_size: int,
    d: int,
    random_hadamard_blocks: torch.Tensor | None = None,
) -> torch.Tensor:
    out = x
    num_blocks = d // block_size
    for step in pipeline:
        if step == "perm":
            if perm is None:
                continue
            out = out.index_select(dim=-1, index=perm.to(out.device))
        elif step == "svd":
            if u_blocks is not None:
                out = _apply_block_matmul(
                    out, u_blocks, block_size=block_size, d=d
                )
        elif step == "hadamard":
            h_blocks = _standard_hadamard_blocks(
                num_blocks, block_size, device=out.device, dtype=out.dtype
            )
            out = _apply_block_matmul(out, h_blocks, block_size=block_size, d=d)
        elif step == "random_hadamard":
            if random_hadamard_blocks is not None:
                h_blocks = random_hadamard_blocks.to(
                    device=out.device, dtype=out.dtype
                )
            else:
                h_blocks = _random_hadamard_blocks(
                    num_blocks, block_size, device=out.device, dtype=out.dtype
                )
            out = _apply_block_matmul(out, h_blocks, block_size=block_size, d=d)
        else:
            raise ValueError(f"Unknown pipeline step {step!r}.")
    return out


@dataclass
class Rotation:
    """Block-diagonal rotation plus optional zigzag permutation.

    ``perm`` maps output slot → original channel index:
    ``x_perm[..., k] = x[..., perm[k]]`` (same convention as DuQuant /
    ``index_select(-1, perm)``).
    """

    mode: str
    block_size: int
    d: int
    u_blocks: torch.Tensor | None = None
    perm: torch.Tensor | None = None
    random_hadamard_blocks: torch.Tensor | None = None
    pipeline: tuple[PipelineStep, ...] = ("perm", "svd", "hadamard")
    meta: dict | None = None

    @property
    def is_identity(self) -> bool:
        return not self.pipeline
            

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the configured pipeline along the last axis."""
        if self.is_identity:
            return x
        return apply_input_pipeline(
            x,
            pipeline=self.pipeline,
            u_blocks=self.u_blocks,
            perm=self.perm,
            block_size=self.block_size,
            d=self.d,
            random_hadamard_blocks=self.random_hadamard_blocks,
        )

    def state_dict(self) -> dict:
        sd: dict = {
            "mode": self.mode,
            "block_size": self.block_size,
            "d": self.d,
            "pipeline": list(self.pipeline),
            "perm": (
                self.perm.detach().to(torch.int64).cpu()
                if self.perm is not None
                else None
            ),
            "meta": dict(self.meta or {}),
        }
        if self.u_blocks is not None:
            sd["u_blocks"] = self.u_blocks.detach().to(torch.float32).cpu()
        if self.random_hadamard_blocks is not None:
            sd["random_hadamard_blocks"] = (
                self.random_hadamard_blocks.detach().to(torch.float32).cpu()
            )
        return sd

    @classmethod
    def from_state_dict(cls, sd: dict) -> "Rotation":
        perm = sd.get("perm")
        pipeline_raw = sd.get("pipeline")
        u_blocks = sd.get("u_blocks")
        random_hadamard_blocks = sd.get("random_hadamard_blocks")
        pipeline = tuple(pipeline_raw) if pipeline_raw is not None else ()

        return cls(
            mode=sd["mode"],
            block_size=int(sd["block_size"]),
            d=int(sd["d"]),
            u_blocks=u_blocks.clone() if u_blocks is not None else None,
            perm=perm.clone() if perm is not None else None,
            random_hadamard_blocks=(
                random_hadamard_blocks.clone()
                if random_hadamard_blocks is not None
                else None
            ),
            pipeline=pipeline,
            meta=dict(sd.get("meta") or {}),
        )


def identity_rotation(d: int, block_size: int) -> Rotation:
    return Rotation(
        mode="none",
        block_size=block_size,
        d=d,
        u_blocks=None,
        perm=None,
        pipeline=(),
    )


def hadamard_rotation(d: int, block_size: int, *, device=None) -> Rotation:
    """Pure standard block-Hadamard rotation. No data needed."""
    _validate_block(d, block_size)
    return Rotation(
        mode="hadamard",
        block_size=block_size,
        d=d,
        u_blocks=None,
        perm=None,
        pipeline=("hadamard",),
    )


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


def _u_for_block(
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


def _fit_u_blocks(
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
    num_blocks = _validate_block(d, block_size)
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

        u_blocks[i] = _u_for_block(
            w_block,
            sigma_b,
            svd_source=svd_source,
            eps=eps,
        )

    return u_blocks.contiguous()


def _zigzag_from_stats(
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
) -> tuple[torch.Tensor, str]:
    w_for_perm = weight
    if pre_perm_pipeline and perm_score in ("weight", "activation_weight"):
        w_for_perm = apply_input_pipeline(
            weight.to(torch.float32),
            pipeline=pre_perm_pipeline,
            u_blocks=u_blocks,
            perm=None,
            block_size=block_size,
            d=d,
            random_hadamard_blocks=random_hadamard_blocks,
        ).contiguous()
    energy, score_used = compute_perm_energy(
        w_for_perm,
        perm_score=perm_score,
        activation_amax=activation_amax,
        sensitivity=sensitivity,
        eps=eps,
    )
    return zigzag_permutation(energy, block_size=block_size), score_used


def step_needs_activation_calibration(
    step: PipelineStep,
    *,
    perm_score: PermScore,
    svd_source: SvdSource,
) -> bool:
    """True when a pipeline step needs activation stats at its input."""
    if step == "perm":
        return perm_score in ("activation", "activation_weight")
    if step == "svd":
        return svd_source == "activation"
    return False


@dataclass
class PipelineRotationBuild:
    """Incrementally fit a :class:`Rotation` one pipeline step at a time."""

    d: int
    block_size: int
    weight: torch.Tensor
    pipeline: tuple[PipelineStep, ...]
    perm_score: PermScore
    svd_source: SvdSource
    sensitivity: torch.Tensor | None = None
    eps: float = 1e-6
    layer_name: str | None = None
    build_seed: int = 0
    perm: torch.Tensor | None = None
    u_blocks: torch.Tensor | None = None
    random_hadamard_blocks: torch.Tensor | None = None
    meta: dict | None = None

    def __post_init__(self) -> None:
        if self.meta is None:
            self.meta = {
                "svd_source": self.svd_source,
                "perm_score": self.perm_score,
                "pipeline": list(self.pipeline),
            }

    def prefix_rotation(self, step_index: int) -> Rotation:
        """Rotation applying ``pipeline[:step_index]`` with components fit so far."""
        prefix = self.pipeline[:step_index]
        if not prefix:
            return identity_rotation(self.d, self.block_size)
        perm = None
        u_blocks = None
        random_hadamard_blocks = None
        if "perm" in prefix:
            perm_idx = self.pipeline.index("perm")
            if perm_idx < step_index:
                perm = self.perm
        if "svd" in prefix:
            svd_idx = self.pipeline.index("svd")
            if svd_idx < step_index:
                u_blocks = self.u_blocks
        if "random_hadamard" in prefix:
            rh_idx = self.pipeline.index("random_hadamard")
            if rh_idx < step_index:
                random_hadamard_blocks = self.random_hadamard_blocks
        return Rotation(
            mode="+".join(prefix),
            block_size=self.block_size,
            d=self.d,
            u_blocks=u_blocks,
            perm=perm,
            random_hadamard_blocks=random_hadamard_blocks,
            pipeline=prefix,
            meta=dict(self.meta or {}),
        )

    def fit_step(
        self,
        step_index: int,
        *,
        stats: LayerStats | None = None,
    ) -> None:
        """Fit ``pipeline[step_index]`` using optional activation stats at its input."""
        step = self.pipeline[step_index]
        pre = self.pipeline[:step_index]
        w = self.weight.detach().to(torch.float32)
        device = w.device

        if step == "perm":
            amax = (
                stats.static_cross_channel_amax.to(device)
                if stats is not None and stats.n_tokens > 0
                else None
            )
            sens = (
                self.sensitivity.to(device)
                if self.sensitivity is not None
                else None
            )
            self.perm, score_used = _zigzag_from_stats(
                w,
                u_blocks=self.u_blocks,
                pre_perm_pipeline=pre,
                block_size=self.block_size,
                d=self.d,
                perm_score=self.perm_score,
                activation_amax=amax,
                sensitivity=sens,
                eps=self.eps,
                random_hadamard_blocks=self.random_hadamard_blocks,
            )
            self.meta["perm_score_used"] = score_used
        elif step == "svd":
            cov = (
                stats.covariance().to(device)
                if stats is not None and stats.n_tokens > 0
                else None
            )
            sens = (
                self.sensitivity.to(device)
                if self.sensitivity is not None
                else None
            )
            self.u_blocks = _fit_u_blocks(
                d=self.d,
                block_size=self.block_size,
                weight=w,
                activation_cov=cov,
                svd_source=self.svd_source,
                perm=self.perm,
                sensitivity=sens,
                eps=self.eps,
            )
        elif step == "hadamard":
            pass
        elif step == "random_hadamard":
            if self.random_hadamard_blocks is None:
                num_blocks = _validate_block(self.d, self.block_size)
                layer = self.layer_name or f"d{self.d}_bs{self.block_size}"
                seed = _random_hadamard_seed(layer, self.build_seed)
                self.random_hadamard_blocks = _random_hadamard_blocks(
                    num_blocks,
                    self.block_size,
                    device=device,
                    seed=seed,
                )
        else:
            raise ValueError(f"Unknown pipeline step {step!r}.")

    def finish(self) -> Rotation:
        """Return the fully fitted rotation."""
        if not self.pipeline:
            return identity_rotation(self.d, self.block_size)
        if self.pipeline == ("hadamard",):
            try:
                return hadamard_rotation(self.d, self.block_size)
            except ValueError as e:
                label = self.layer_name or "layer"
                logger.warning(
                    "Skipping rotation for %s (K=%d, block_size=%d): %s",
                    label,
                    self.d,
                    self.block_size,
                    e,
                )
                return identity_rotation(self.d, self.block_size)
        try:
            return Rotation(
                mode="+".join(self.pipeline),
                block_size=self.block_size,
                d=self.d,
                u_blocks=self.u_blocks,
                perm=self.perm,
                random_hadamard_blocks=self.random_hadamard_blocks,
                pipeline=self.pipeline,
                meta=dict(self.meta or {}),
            )
        except ValueError as e:
            label = self.layer_name or "layer"
            logger.warning(
                "Skipping rotation for %s (K=%d, block_size=%d): %s",
                label,
                self.d,
                self.block_size,
                e,
            )
            return identity_rotation(self.d, self.block_size)


__all__ = [
    "PermScore",
    "PipelineStep",
    "PipelineRotationBuild",
    "Rotation",
    "SvdSource",
    "hadamard_matrix",
    "identity_rotation",
    "parse_pipeline_string",
    "random_hadamard_matrix",
    "step_needs_activation_calibration",
    "validate_pipeline",
]
