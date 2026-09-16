"""Pure helpers shared by every Fisher collector and the driver.

Kept free of torch.nn hooks / adapter access so unit tests can exercise
these functions in isolation. Enum Literals live in :mod:`qvla.config`
(single source of truth for CLI choices and typing).
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from qvla.config import FisherMethod, FisherType, StepAggregation

logger = logging.getLogger(__name__)


def normalize_fisher_sensitivity(fisher: torch.Tensor) -> torch.Tensor:
    """Scale Fisher so ``mean(F) == 1``; preserves per-channel relative ratios."""
    f = fisher.to(torch.float32)
    if f.numel() == 0:
        raise ValueError("Cannot normalize empty Fisher sensitivity.")
    if f.sum().item() <= 0:
        logger.warning("All-zero Fisher sensitivity; skip normalization.")
        return f.clamp_min(1e-8)
    mean = f.mean()
    if mean.item() <= 0:
        raise ValueError("Fisher sensitivity mean must be positive.")
    return f / mean


def select_fisher_actions(
    actions: torch.Tensor,
    *,
    timestep: str = "all",
    action_dim: int,
) -> torch.Tensor:
    """Select action-chunk timesteps, then keep the first ``action_dim`` DoFs.

    ``actions`` must be ``(B, T, A)`` where ``A`` is the model's native
    action width (e.g. ``max_action_dim``). Only ``[..., :action_dim]`` is
    kept — the executed task DoFs (e.g. 7 for LIBERO), not padding dims.

    ``timestep``:
      * ``"all"``    — keep the full ``(B, T, action_dim)`` tensor.
      * ``"i"``      — one chunk index (0-based) → ``(B, action_dim)``.
      * ``"i,j,k"`` — comma-separated indices → ``(B, K, action_dim)``.
    """
    if action_dim <= 0:
        raise ValueError(f"action_dim must be positive, got {action_dim}.")
    if actions.ndim != 3:
        raise ValueError(
            f"Fisher actions must be (B, T, A), got shape {tuple(actions.shape)}."
        )
    native_a = actions.shape[-1]
    if action_dim > native_a:
        raise ValueError(
            f"action_dim={action_dim} exceeds native action width {native_a}."
        )
    spec = timestep.strip().lower()
    t_len = actions.shape[1]
    if spec == "all":
        selected = actions
    else:
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        if not parts:
            raise ValueError("fisher_action_timestep cannot be empty.")
        idxs: list[int] = []
        for part in parts:
            try:
                idx = int(part)
            except ValueError as e:
                raise ValueError(
                    "fisher_action_timestep must be 'all' or integer indices "
                    f"(e.g. '0,29,50'), got {timestep!r}."
                ) from e
            if idx < 0 or idx >= t_len:
                raise ValueError(
                    f"fisher_action_timestep index {idx} out of range for "
                    f"chunk length {t_len}."
                )
            idxs.append(idx)
        if len(idxs) == 1:
            selected = actions[:, idxs[0], :]
        else:
            selected = actions[:, idxs, :]
    return selected[..., :action_dim].contiguous()


def resolve_fisher_action_dim(adapter: Any) -> int:
    """Task action DoF count from the model adapter (e.g. LIBERO ``action_dim=7``)."""
    cfg = getattr(adapter, "cfg", None)
    if cfg is None:
        raise ValueError(
            f"Fisher requires adapter.cfg.action_dim; "
            f"{type(adapter).__name__} has no cfg."
        )
    if not hasattr(cfg, "action_dim"):
        raise ValueError(
            f"Fisher requires adapter.cfg.action_dim; "
            f"{type(cfg).__name__} has no action_dim field."
        )
    dim = int(cfg.action_dim)
    if dim <= 0:
        raise ValueError(f"adapter.cfg.action_dim must be positive, got {dim}.")
    return dim


# Re-exported so ``from qvla.build.fisher.common import FisherMethod`` keeps
# working; definitions live in ``qvla.config``.
__all__ = [
    "FisherMethod",
    "FisherType",
    "StepAggregation",
    "normalize_fisher_sensitivity",
    "resolve_fisher_action_dim",
    "select_fisher_actions",
]
