"""GR00T-N1.7 autograd context for Fisher / policy-aware rotation.

Composes the model-agnostic patches from :mod:`qvla.build.grad_patches`:
GR00T's backbone (Qwen3-VL text) needs Linear + RMSNorm patches, its DiT
action head needs Linear + LayerNorm patches. Attention uses SDPA (autograd
OK by default); the fused GeGLU / RoPE kernels are not used by the layers
QVLA targets here.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator

from qvla.build.grad_patches import (
    patch_layernorms_for_grad,
    patch_linears_for_grad,
    patch_rmsnorms_for_grad,
    restore_layernorms_for_grad,
    restore_linears_for_grad,
    restore_rmsnorms_for_grad,
)


@contextlib.contextmanager
def differentiable_inference_context(scheduler: Any) -> Iterator[None]:
    """Patch GR00T-N1.7 backbone + action head for one Fisher pass."""
    model = scheduler.model
    backbone = getattr(model, "backbone", None)
    action_head = getattr(model, "action_head", None)
    if backbone is None or action_head is None:
        raise RuntimeError(
            "GR00T-N1.7 scheduler.model is missing backbone/action_head; "
            "cannot install Fisher autograd patches."
        )

    linears = patch_linears_for_grad(backbone, log_label="backbone")
    linears += patch_linears_for_grad(action_head, log_label="action_head")
    norms = patch_layernorms_for_grad(action_head, log_label="action_head")
    rms = patch_rmsnorms_for_grad(backbone, log_label="backbone")
    try:
        yield
    finally:
        restore_linears_for_grad(linears)
        restore_layernorms_for_grad(norms)
        restore_rmsnorms_for_grad(rms)


__all__ = ["differentiable_inference_context"]
