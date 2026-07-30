"""Test-only helpers — not part of the production API."""

from __future__ import annotations

import logging

import torch

from qvla.build.collector import LayerStats
from qvla.core.rotation import (
    PermScore,
    PipelineRotationBuild,
    PipelineStep,
    SvdSource,
    identity_rotation,
    step_needs_activation_calibration,
    validate_pipeline,
)

logger = logging.getLogger(__name__)


def fit_rotation(
    *,
    d: int,
    block_size: int,
    weight: torch.Tensor,
    pipeline: tuple[PipelineStep, ...],
    activation_cov: torch.Tensor | None = None,
    activation_amax: torch.Tensor | None = None,
    sensitivity: torch.Tensor | None = None,
    svd_source: SvdSource = "weight",
    perm_score: PermScore = "weight",
    device=None,
    eps: float = 1e-6,
    layer_name: str | None = None,
    build_seed: int = 0,
):
    """Fit a rotation offline when stats are already available (tests only)."""
    pipeline = validate_pipeline(pipeline)
    if device is not None:
        weight = weight.to(device)
        if activation_cov is not None:
            activation_cov = activation_cov.to(device)
        if activation_amax is not None:
            activation_amax = activation_amax.to(device)
        if sensitivity is not None:
            sensitivity = sensitivity.to(device)

    builder = PipelineRotationBuild(
        d=d,
        block_size=block_size,
        weight=weight,
        pipeline=pipeline,
        perm_score=perm_score,
        svd_source=svd_source,
        sensitivity=sensitivity,
        eps=eps,
        layer_name=layer_name,
        build_seed=build_seed,
    )
    try:
        for i, step in enumerate(pipeline):
            stats = None
            if step_needs_activation_calibration(
                step, perm_score=perm_score, svd_source=svd_source
            ):
                stats = LayerStats(in_features=d)
                if step == "perm" and activation_amax is not None:
                    stats.static_cross_channel_amax = activation_amax.to(torch.float32)
                    stats.n_tokens = 1
                elif step == "svd" and activation_cov is not None:
                    stats.xtx = activation_cov.to(torch.float64)
                    stats.n_tokens = 1
            builder.fit_step(i, stats=stats)
        return builder.finish()
    except ValueError as e:
        label = layer_name or "layer"
        logger.warning(
            "Skipping rotation for %s (K=%d, block_size=%d): %s",
            label,
            d,
            block_size,
            e,
        )
        return identity_rotation(d, block_size)
