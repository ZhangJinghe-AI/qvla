"""Input-side transform pipeline.

The scope ``pipeline`` is an ordered list of steps applied left-to-right.
All steps are handled uniformly by :func:`apply_input_pipeline` /
:class:`Transform`. Each step has an optional data tensor; when ``None``
the step is a no-op:

* ``clip`` — per-channel ``clamp(x, ±c)`` via ``act_clip``
* ``smooth`` — SmoothQuant ``x / s`` via ``smooth_scale``
* ``perm`` — zigzag channel reorder via ``perm``
* ``svd`` — per-block orthonormal ``U`` via ``u_blocks``
* ``hadamard`` — standard Sylvester block Hadamard (deterministic)
* ``random_hadamard`` — random sign-flipped Hadamard via ``random_hadamard_blocks``

Only pipelines in :data:`ALLOWED_PIPELINES` are accepted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

from qvla.core.rotation import (
    apply_block_matmul,
    fit_u_blocks,
    random_hadamard_blocks as make_random_hadamard_blocks,
    random_hadamard_seed,
    standard_hadamard_blocks,
    validate_block,
    zigzag_from_stats,
)

if TYPE_CHECKING:
    from qvla.build.collector import AmaxCollectPlan, LayerStats

PermScore = Literal["weight", "activation", "activation_weight", "fisher"]
SvdSource = Literal["weight", "activation"]
PipelineStep = Literal[
    "clip", "smooth", "perm", "svd", "hadamard", "random_hadamard"
]

_VALID_STEPS = frozenset(
    {"clip", "smooth", "perm", "svd", "hadamard", "random_hadamard"}
)

ALLOWED_PIPELINES: frozenset[tuple[PipelineStep, ...]] = frozenset(
    {
        (),
        # Orthogonal / DuQuant-style
        ("hadamard",),
        ("random_hadamard",),
        ("svd", "hadamard"),
        ("svd", "random_hadamard"),
        ("perm", "svd", "hadamard"),
        ("perm", "svd", "random_hadamard"),
        # Adaptive act_clip (optional + orthogonal)
        ("clip",),
        ("clip", "hadamard"),
        ("clip", "random_hadamard"),
        ("clip", "svd", "hadamard"),
        ("clip", "svd", "random_hadamard"),
        ("clip", "perm", "svd", "hadamard"),
        ("clip", "perm", "svd", "random_hadamard"),
        # SmoothQuant (optional + Hadamard). s is applied in pre-Hadamard
        # channel space; R is the same on (x/s) and (W*s). perm/svd after
        # smooth are not allowlisted (weight-side perm scoring would apply
        # x/s instead of W*s).
        ("smooth",),
        ("clip", "smooth"),
        ("smooth", "hadamard"),
        ("smooth", "random_hadamard"),
        ("clip", "smooth", "hadamard"),
        ("clip", "smooth", "random_hadamard"),
    }
)

DEFAULT_PIPELINE: tuple[PipelineStep, ...] = ()

logger = logging.getLogger(__name__)


def pipeline_has_clip(pipeline: tuple[PipelineStep, ...]) -> bool:
    return "clip" in pipeline


def pipeline_has_smooth(pipeline: tuple[PipelineStep, ...]) -> bool:
    return "smooth" in pipeline


def validate_pipeline(
    pipeline: tuple[PipelineStep, ...],
) -> tuple[PipelineStep, ...]:
    for step in pipeline:
        if step not in _VALID_STEPS:
            raise ValueError(
                f"Invalid pipeline step {step!r}; expected "
                "clip|smooth|perm|svd|hadamard|random_hadamard."
            )
    key = tuple(pipeline)
    if key not in ALLOWED_PIPELINES:
        allowed = ", ".join(
            ("none" if not p else ",".join(p)) for p in sorted(ALLOWED_PIPELINES)
        )
        got = "none" if not pipeline else ",".join(pipeline)
        raise ValueError(
            f"Unsupported pipeline {got!r}. Allowed pipelines: {allowed}."
        )
    return pipeline


def parse_pipeline_string(value: str) -> tuple[PipelineStep, ...]:
    """Parse CLI / JSON pipeline strings like ``clip,hadamard`` or ``smooth``."""
    text = value.strip().lower()
    if text in ("", "none", "identity"):
        return ()
    steps = tuple(s.strip() for s in text.split(",") if s.strip())
    return validate_pipeline(steps)  # type: ignore[arg-type]


def apply_input_pipeline(
    x: torch.Tensor,
    *,
    pipeline: tuple[PipelineStep, ...],
    u_blocks: torch.Tensor | None,
    perm: torch.Tensor | None,
    block_size: int,
    d: int,
    random_hadamard_blocks: torch.Tensor | None = None,
    act_clip: torch.Tensor | None = None,
    smooth_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply pipeline steps along the last axis in order."""
    out = x
    num_blocks = d // block_size
    for pipe_step in pipeline:
        if pipe_step == "clip":
            if act_clip is not None:
                c = act_clip.to(device=out.device, dtype=out.dtype)
                out = out.clamp(min=-c, max=c)
        elif pipe_step == "smooth":
            if smooth_scale is not None:
                s = smooth_scale.to(device=out.device, dtype=out.dtype)
                out = out / s
        elif pipe_step == "perm":
            if perm is None:
                continue
            out = out.index_select(dim=-1, index=perm.to(out.device))
        elif pipe_step == "svd":
            if u_blocks is not None:
                out = apply_block_matmul(
                    out, u_blocks, block_size=block_size, d=d
                )
        elif pipe_step == "hadamard":
            h_blocks = standard_hadamard_blocks(
                num_blocks, block_size, device=out.device, dtype=out.dtype
            )
            out = apply_block_matmul(out, h_blocks, block_size=block_size, d=d)
        elif pipe_step == "random_hadamard":
            if random_hadamard_blocks is None:
                raise ValueError(
                    "Pipeline contains 'random_hadamard' but "
                    "random_hadamard_blocks was not provided."
                )
            h_blocks = random_hadamard_blocks.to(
                device=out.device, dtype=out.dtype
            )
            out = apply_block_matmul(out, h_blocks, block_size=block_size, d=d)
        else:
            raise ValueError(f"Unknown pipeline step {pipe_step!r}.")
    return out


@dataclass
class Transform:
    """Input-side transform pipeline applied in order via :func:`apply_input_pipeline`.

    Each step has an optional data tensor (no-op when the tensor is ``None``):

    * ``clip`` → :attr:`act_clip` (per-channel clamp)
    * ``smooth`` → :attr:`smooth_scale` (per-channel ``x / s``)
    * ``perm`` → :attr:`perm` (zigzag channel reorder)
    * ``svd`` → :attr:`u_blocks` (per-block orthonormal)
    * ``hadamard`` → deterministic Sylvester block Hadamard
    * ``random_hadamard`` → :attr:`random_hadamard_blocks`
    """

    mode: str
    block_size: int
    d: int
    u_blocks: torch.Tensor | None = None
    perm: torch.Tensor | None = None
    random_hadamard_blocks: torch.Tensor | None = None
    act_clip: torch.Tensor | None = None
    smooth_scale: torch.Tensor | None = None
    pipeline: tuple[PipelineStep, ...] = ("perm", "svd", "hadamard")
    perm_score_used: str | None = None

    @property
    def is_identity(self) -> bool:
        return not self.pipeline

    def apply(self, x: torch.Tensor, *, step: int | None = None) -> torch.Tensor:
        """Apply the configured pipeline along the last axis."""
        if self.is_identity:
            return x
        act_clip = self.act_clip
        if act_clip is not None and act_clip.ndim == 2:
            if step is None:
                raise RuntimeError(
                    "per-step act_clip requires the current denoise step."
                )
            act_clip = act_clip[step]
        return apply_input_pipeline(
            x,
            pipeline=self.pipeline,
            u_blocks=self.u_blocks,
            perm=self.perm,
            block_size=self.block_size,
            d=self.d,
            random_hadamard_blocks=self.random_hadamard_blocks,
            act_clip=act_clip,
            smooth_scale=self.smooth_scale,
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
        }
        if self.u_blocks is not None:
            sd["u_blocks"] = self.u_blocks.detach().to(torch.float32).cpu()
        if self.random_hadamard_blocks is not None:
            sd["random_hadamard_blocks"] = (
                self.random_hadamard_blocks.detach().to(torch.float32).cpu()
            )
        if self.act_clip is not None:
            sd["act_clip"] = self.act_clip.detach().to(torch.float32).cpu()
        if self.smooth_scale is not None:
            sd["smooth_scale"] = self.smooth_scale.detach().to(torch.float32).cpu()
        return sd

    @classmethod
    def from_state_dict(cls, sd: dict) -> "Transform":
        perm = sd.get("perm")
        pipeline_raw = sd.get("pipeline")
        u_blocks = sd.get("u_blocks")
        random_hadamard_blocks = sd.get("random_hadamard_blocks")
        act_clip = sd.get("act_clip")
        smooth_scale = sd.get("smooth_scale")
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
            act_clip=act_clip.clone() if act_clip is not None else None,
            smooth_scale=smooth_scale.clone() if smooth_scale is not None else None,
            pipeline=pipeline,
        )


def identity_transform(d: int, block_size: int) -> Transform:
    return Transform(
        mode="none",
        block_size=block_size,
        d=d,
        pipeline=(),
    )


def hadamard_transform(d: int, block_size: int, *, device=None) -> Transform:
    """Pure standard block-Hadamard transform. No data needed."""
    validate_block(d, block_size)
    return Transform(
        mode="hadamard",
        block_size=block_size,
        d=d,
        pipeline=("hadamard",),
    )


def step_needs_activation_calibration(
    step: PipelineStep,
    *,
    perm_score: PermScore = "weight",
    svd_source: SvdSource = "weight",
) -> bool:
    """True when a pipeline step needs activation stats at its input."""
    if step in ("clip", "smooth"):
        return True
    if step == "perm":
        return perm_score in ("activation", "activation_weight")
    if step == "svd":
        return svd_source == "activation"
    return False


@dataclass
class PipelineBuild:
    """Incrementally fit a :class:`Transform` one pipeline step at a time."""

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
    # clip config
    clip_epsilon: float = 1e-5
    clip_kappa: float = 0.0
    clip_bulk_percentile: float = 95.0
    clip_std_k: float = 0.0
    clip_std_k_down: float = 0.0
    clip_std_k_up: float = 0.0
    clip_selective_channels: bool = False
    clip_global: bool = False
    clip_skip_first_token: bool = False
    # smooth config
    smooth_alpha: float = 0.5
    smooth_epsilon: float = 1e-5
    smooth_act_percentile: float = 100.0
    smooth_fisher: torch.Tensor | None = None
    smooth_fisher_beta: float = 0.0
    # None: hard max over all tokens/steps (original). Positive p: DiT
    # per-step absmax then p-mean. See ``ScopeConfig.smooth_step_pmean_p``.
    smooth_step_pmean_p: float | None = None
    # fitted data (set by fit_step or externally)
    act_clip: torch.Tensor | None = None
    smooth_scale: torch.Tensor | None = None
    perm: torch.Tensor | None = None
    u_blocks: torch.Tensor | None = None
    random_hadamard_blocks: torch.Tensor | None = None
    perm_score_used: str | None = None

    def prefix_transform(self, step_index: int) -> Transform:
        """Transform applying ``pipeline[:step_index]`` with components fit so far."""
        prefix = self.pipeline[:step_index]
        if not prefix:
            return identity_transform(self.d, self.block_size)
        return Transform(
            mode="+".join(prefix),
            block_size=self.block_size,
            d=self.d,
            act_clip=self.act_clip if "clip" in prefix else None,
            smooth_scale=self.smooth_scale if "smooth" in prefix else None,
            perm=self.perm if "perm" in prefix else None,
            u_blocks=self.u_blocks if "svd" in prefix else None,
            random_hadamard_blocks=self.random_hadamard_blocks if "random_hadamard" in prefix else None,
            pipeline=prefix,
        )

    def step_amax_plan(self, step_index: int) -> AmaxCollectPlan:
        """Return the :class:`AmaxCollectPlan` for collecting stats at *step_index*."""
        from qvla.build.collector import AmaxCollectPlan

        step = self.pipeline[step_index]
        if step == "clip":
            return AmaxCollectPlan(
                collect_hessian=False,
                collect_adaptive_inner_channel=True,
                outlier_kappa=self.clip_kappa,
                outlier_bulk_percentile=self.clip_bulk_percentile,
                outlier_std_k=self.clip_std_k,
                outlier_std_k_down=self.clip_std_k_down,
                outlier_std_k_up=self.clip_std_k_up,
                outlier_selective_channels=self.clip_selective_channels,
                outlier_global=self.clip_global,
                outlier_skip_first_token=self.clip_skip_first_token,
            )
        if step == "smooth":
            if self.smooth_act_percentile < 100.0:
                return AmaxCollectPlan(
                    collect_hessian=False,
                    collect_inner_channel=True,
                    inner_percentile=self.smooth_act_percentile,
                )
            return AmaxCollectPlan(collect_hessian=False, collect_cross_channel=True)
        if step == "perm":
            return AmaxCollectPlan(collect_hessian=False, collect_cross_channel=True)
        if step == "svd":
            return AmaxCollectPlan(collect_hessian=True)
        return AmaxCollectPlan(collect_hessian=False)

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
            self.perm, score_used = zigzag_from_stats(
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
                act_clip=self.act_clip,
                smooth_scale=self.smooth_scale,
            )
            self.perm_score_used = score_used
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
            self.u_blocks = fit_u_blocks(
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
                num_blocks = validate_block(self.d, self.block_size)
                layer = self.layer_name or f"d{self.d}_bs{self.block_size}"
                seed = random_hadamard_seed(layer, self.build_seed)
                self.random_hadamard_blocks = make_random_hadamard_blocks(
                    num_blocks,
                    self.block_size,
                    device=device,
                    seed=seed,
                )
        elif step == "clip":
            if stats is not None and stats.n_tokens > 0:
                amax = stats.act_channel_inner_amax().to(device)
                self.act_clip = amax.abs().clamp_min(self.clip_epsilon).to(torch.float32)
        elif step == "smooth":
            if stats is not None and stats.n_tokens > 0:
                from qvla.core.smooth_fit import fit_smooth_scale

                if self.smooth_step_pmean_p is not None:
                    if self.smooth_act_percentile != 100.0:
                        raise RuntimeError(
                            "smooth_step_pmean_p requires "
                            "smooth_act_percentile==100, got "
                            f"{self.smooth_act_percentile}."
                        )
                    act_a = stats.act_channel_cross_amax_pmean(
                        p=float(self.smooth_step_pmean_p)
                    ).to(device)
                elif self.smooth_act_percentile < 100.0:
                    act_a = stats.act_channel_inner_amax().to(device)
                else:
                    act_a = stats.act_channel_cross_amax().to(device)
                self.smooth_scale = fit_smooth_scale(
                    layer_name=self.layer_name,
                    weight=w,
                    act_channel_amax=act_a,
                    alpha=self.smooth_alpha,
                    epsilon=self.smooth_epsilon,
                    fisher=self.smooth_fisher,
                    fisher_beta=self.smooth_fisher_beta,
                )
        else:
            raise ValueError(f"Unknown pipeline step {step!r}.")

    def finish(self) -> Transform:
        """Return the fully fitted transform."""
        if not self.pipeline:
            return identity_transform(self.d, self.block_size)
        return Transform(
            mode="+".join(self.pipeline),
            block_size=self.block_size,
            d=self.d,
            act_clip=self.act_clip,
            smooth_scale=self.smooth_scale,
            u_blocks=self.u_blocks,
            perm=self.perm,
            random_hadamard_blocks=self.random_hadamard_blocks,
            pipeline=self.pipeline,
            perm_score_used=self.perm_score_used,
        )



__all__ = [
    "ALLOWED_PIPELINES",
    "DEFAULT_PIPELINE",
    "PermScore",
    "PipelineBuild",
    "PipelineStep",
    "SvdSource",
    "Transform",
    "apply_input_pipeline",
    "hadamard_transform",
    "identity_transform",
    "parse_pipeline_string",
    "pipeline_has_clip",
    "pipeline_has_smooth",
    "step_needs_activation_calibration",
    "validate_pipeline",
]
