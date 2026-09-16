"""Model-agnostic autograd patches for phyai layers (Fisher / sensitivity passes).

phyai's default inference kernels (FlashInfer GEMM, phyai-kernel Triton norms,
fused GeGLU, fused RoPE) have no backward. These helpers swap each layer to a
torch-native reference implementation **only while grad is enabled**, so
regular inference forwards keep the fast kernels.

All patchers walk a caller-supplied ``root`` module — which sub-tree of the
model to patch (expert stack, backbone, action head, ...) is the model
adapter's decision. Each ``patch_*`` returns the list of patched modules;
pass it to the matching ``restore_*`` to undo.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from qvla.runtime.wrap import _is_linear_like


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Linears -> F.linear                                                          #
# --------------------------------------------------------------------------- #


def linear_with_bias(layer: Any, x: torch.Tensor) -> torch.Tensor:
    """Autograd-safe matmul for phyai linear weights (respects ``skip_bias_add``)."""
    bias = layer.bias if not getattr(layer, "skip_bias_add", False) else None
    return F.linear(x, layer.weight, bias)


def _returns_tuple(mod: nn.Module) -> bool:
    """phyai ``LinearBase`` returns ``(y, bias)``; ``GR00TN17Linear``/``nn.Linear`` return ``y``."""
    if type(mod).__name__ == "GR00TN17Linear" or isinstance(mod, nn.Linear):
        return False
    if hasattr(mod, "skip_bias_add"):
        return True
    cls_name = type(mod).__name__
    return cls_name.endswith("Linear") and cls_name != "Linear"


def patch_linears_for_grad(root: nn.Module, *, log_label: str) -> list[Any]:
    """Route every linear under ``root`` through ``F.linear`` when grad is needed.

    FlashInfer GEMM (the default phyai linear backend) has no autograd; without
    this patch gradients stop at the first projection and every downstream layer
    reports zero Fisher sensitivity. ``QuantLinear`` modules are skipped.
    """
    from qvla.runtime.quant_linear import QuantLinear

    patched: list[Any] = []
    for mod in root.modules():
        if isinstance(mod, QuantLinear):
            continue
        if not _is_linear_like(mod):
            continue
        if getattr(mod, "_qvla_linear_grad_patched", False):
            continue
        orig_forward = mod.forward
        as_tuple = _returns_tuple(mod)
        skip_bias_add = bool(getattr(mod, "skip_bias_add", False))

        def make_forward(
            layer: Any,
            original: Any,
            *,
            return_tuple: bool,
            skip_bias: bool,
        ):
            def forward(x: torch.Tensor, *args: Any, **kwargs: Any):
                if not torch.is_grad_enabled():
                    return original(x, *args, **kwargs)
                bias = layer.bias if not skip_bias else None
                y = F.linear(x, layer.weight, bias)
                if return_tuple:
                    bias_out = layer.bias if skip_bias else None
                    return y, bias_out
                return y

            return forward

        mod._qvla_orig_forward = orig_forward  # type: ignore[attr-defined]
        mod.forward = make_forward(  # type: ignore[method-assign]
            mod, orig_forward, return_tuple=as_tuple, skip_bias=skip_bias_add
        )
        mod._qvla_linear_grad_patched = True
        patched.append(mod)
    if patched:
        logger.info(
            "Patched %d %s Linear(s) to F.linear for autograd.",
            len(patched),
            log_label,
        )
    return patched


def restore_linears_for_grad(patched: list[Any]) -> None:
    for mod in patched:
        if getattr(mod, "_qvla_linear_grad_patched", False):
            mod.forward = mod._qvla_orig_forward  # type: ignore[method-assign]
            mod._qvla_linear_grad_patched = False


# --------------------------------------------------------------------------- #
# phyai LayerNorm -> F.layer_norm                                              #
# --------------------------------------------------------------------------- #


def _torch_layernorm(mod: Any, x: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(
        x,
        (mod.hidden_size,),
        weight=mod.weight if mod.weight is not None else None,
        bias=mod.bias if getattr(mod, "has_bias", False) else None,
        eps=float(mod.variance_epsilon),
    )


def patch_layernorms_for_grad(root: nn.Module, *, log_label: str) -> list[Any]:
    """Route phyai ``LayerNorm`` under ``root`` through ``F.layer_norm`` under grad."""
    from phyai.layers.layer_norm import LayerNorm

    patched: list[Any] = []
    for mod in root.modules():
        if not isinstance(mod, LayerNorm):
            continue
        if getattr(mod, "_qvla_torch_layernorm_patched", False):
            continue
        orig_forward = mod.forward

        def make_forward(layer: Any, original: Any):
            def forward(x: torch.Tensor, *args: Any, **kwargs: Any):
                if torch.is_grad_enabled():
                    return _torch_layernorm(layer, x)
                return original(x, *args, **kwargs)

            return forward

        mod._qvla_orig_forward = orig_forward  # type: ignore[attr-defined]
        mod.forward = make_forward(mod, orig_forward)  # type: ignore[method-assign]
        mod._qvla_torch_layernorm_patched = True
        patched.append(mod)
    if patched:
        logger.info(
            "Patched %d %s LayerNorm(s) to F.layer_norm for autograd.",
            len(patched),
            log_label,
        )
    return patched


def restore_layernorms_for_grad(patched: list[Any]) -> None:
    for mod in patched:
        if getattr(mod, "_qvla_torch_layernorm_patched", False):
            mod.forward = mod._qvla_orig_forward  # type: ignore[method-assign]
            mod._qvla_torch_layernorm_patched = False


# --------------------------------------------------------------------------- #
# phyai plain RMSNorm -> torch reference (forward patch, incl. residual path)  #
# --------------------------------------------------------------------------- #


def _torch_rmsnorm(
    mod: Any, x: torch.Tensor, residual: torch.Tensor | None = None
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Autograd-safe RMSNorm matching phyai's plain ``RMSNorm`` math (not Gemma)."""
    if residual is not None:
        x = x + residual
    orig_dtype = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    y = (xf * torch.rsqrt(var + float(mod.variance_epsilon))) * mod.weight.float()
    y = y.to(orig_dtype)
    if residual is not None:
        return y, x
    return y


def patch_rmsnorms_for_grad(root: nn.Module, *, log_label: str) -> list[Any]:
    """Route plain phyai ``RMSNorm`` under ``root`` through torch under grad.

    ``GemmaRMSNorm`` is intentionally excluded — its ``(1 + w)`` kernel is
    handled by :func:`patch_gemma_rmsnorm_kernels_for_grad`.
    """
    from phyai.layers.layer_norm import GemmaRMSNorm, RMSNorm

    patched: list[Any] = []
    for mod in root.modules():
        if not isinstance(mod, RMSNorm) or isinstance(mod, GemmaRMSNorm):
            continue
        if getattr(mod, "_qvla_torch_rmsnorm_patched", False):
            continue
        orig_forward = mod.forward

        def make_forward(layer: Any, original: Any):
            def forward(
                x: torch.Tensor,
                residual: torch.Tensor | None = None,
                *args: Any,
                **kwargs: Any,
            ):
                if torch.is_grad_enabled():
                    return _torch_rmsnorm(layer, x, residual)
                return original(x, residual, *args, **kwargs)

            return forward

        mod._qvla_orig_forward = orig_forward  # type: ignore[attr-defined]
        mod.forward = make_forward(mod, orig_forward)  # type: ignore[method-assign]
        mod._qvla_torch_rmsnorm_patched = True
        patched.append(mod)
    if patched:
        logger.info(
            "Patched %d %s RMSNorm(s) to torch for autograd.",
            len(patched),
            log_label,
        )
    return patched


def restore_rmsnorms_for_grad(patched: list[Any]) -> None:
    for mod in patched:
        if getattr(mod, "_qvla_torch_rmsnorm_patched", False):
            mod.forward = mod._qvla_orig_forward  # type: ignore[method-assign]
            mod._qvla_torch_rmsnorm_patched = False


# --------------------------------------------------------------------------- #
# phyai GemmaRMSNorm -> torch (1 + w) kernel swap                              #
# --------------------------------------------------------------------------- #


def _torch_gemma_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    needs_reshape = x.dim() != 2
    if needs_reshape:
        orig_shape = x.shape
        x = x.contiguous().reshape(-1, orig_shape[-1])
    var = x.pow(2).mean(dim=-1, keepdim=True)
    out = x * torch.rsqrt(var + eps) * (1.0 + weight)
    if needs_reshape:
        out = out.reshape(orig_shape)
    return out


def patch_gemma_rmsnorm_kernels_for_grad(
    root: nn.Module, *, log_label: str
) -> list[Any]:
    """Swap ``GemmaRMSNorm``'s non-residual kernel to the torch reference."""
    from phyai.layers.layer_norm import GemmaRMSNorm

    patched: list[Any] = []
    for mod in root.modules():
        if not isinstance(mod, GemmaRMSNorm):
            continue
        if getattr(mod, "_qvla_torch_rms_patched", False):
            continue
        mod._qvla_orig_rmsnorm = mod._rmsnorm  # type: ignore[attr-defined]
        mod._rmsnorm = _torch_gemma_rmsnorm
        mod._qvla_torch_rms_patched = True
        patched.append(mod)
    if patched:
        logger.info(
            "Patched %d %s GemmaRMSNorm module(s) to torch backend.",
            len(patched),
            log_label,
        )
    return patched


def restore_gemma_rmsnorm_kernels_for_grad(patched: list[Any]) -> None:
    for mod in patched:
        if getattr(mod, "_qvla_torch_rms_patched", False):
            mod._rmsnorm = mod._qvla_orig_rmsnorm  # type: ignore[attr-defined]
            mod._qvla_torch_rms_patched = False


# --------------------------------------------------------------------------- #
# phyai AdaRMSNorm -> torch kernel swap                                        #
# --------------------------------------------------------------------------- #


def patch_adarmsnorm_kernels_for_grad(
    root: nn.Module, *, log_label: str
) -> list[Any]:
    """Swap ``AdaRMSNorm``'s Triton kernel to the torch reference under grad.

    The default ``phyai-kernel`` Triton path has no autograd, which disconnects
    adaptive-norm DiT stacks even when linears and attention are patched.
    """
    from phyai.layers.layer_norm import AdaRMSNorm, _torch_adarmsnorm

    patched: list[Any] = []
    for mod in root.modules():
        if not isinstance(mod, AdaRMSNorm):
            continue
        if getattr(mod, "_qvla_torch_adarms_patched", False):
            continue
        mod._qvla_orig_adarms_kernel = mod._adarms_kernel  # type: ignore[attr-defined]
        mod._adarms_kernel = _torch_adarmsnorm
        mod._qvla_torch_adarms_patched = True
        patched.append(mod)
    if patched:
        logger.info(
            "Patched %d %s AdaRMSNorm module(s) to torch backend.",
            len(patched),
            log_label,
        )
    return patched


def restore_adarmsnorm_kernels_for_grad(patched: list[Any]) -> None:
    for mod in patched:
        if getattr(mod, "_qvla_torch_adarms_patched", False):
            mod._adarms_kernel = mod._qvla_orig_adarms_kernel  # type: ignore[attr-defined]
            mod._qvla_torch_adarms_patched = False


# --------------------------------------------------------------------------- #
# phyai DenseMLP fused GeGLU -> torch                                          #
# --------------------------------------------------------------------------- #


def _gelu_tanh_and_mul_torch(fused: torch.Tensor) -> torch.Tensor:
    gate, up = fused.chunk(2, dim=-1)
    return F.gelu(gate, approximate="tanh") * up


def patch_mlp_activations_for_grad(root: nn.Module, *, log_label: str) -> list[Any]:
    """Route gated ``DenseMLP`` GeGLU through torch when grad is needed."""
    from phyai.layers.mlp.dense_mlp import DenseMLP

    patched: list[Any] = []
    for mod in root.modules():
        if not isinstance(mod, DenseMLP) or not mod.gated or mod._act_and_mul is None:
            continue
        if getattr(mod, "_qvla_act_patched", False):
            continue
        orig_act = mod._act_and_mul

        def make_act(original):
            def act_and_mul(fused: torch.Tensor) -> torch.Tensor:
                if torch.is_grad_enabled():
                    return _gelu_tanh_and_mul_torch(fused)
                return original(fused)

            return act_and_mul

        mod._qvla_orig_act_and_mul = orig_act  # type: ignore[attr-defined]
        mod._act_and_mul = make_act(orig_act)
        mod._qvla_act_patched = True
        patched.append(mod)
    if patched:
        logger.info(
            "Patched %d %s DenseMLP activation(s) to torch backend.",
            len(patched),
            log_label,
        )
    return patched


def restore_mlp_activations_for_grad(patched: list[Any]) -> None:
    for mod in patched:
        if getattr(mod, "_qvla_act_patched", False):
            mod._act_and_mul = mod._qvla_orig_act_and_mul  # type: ignore[attr-defined]
            mod._qvla_act_patched = False


# --------------------------------------------------------------------------- #
# phyai RotaryEmbedding -> eager RoPE                                          #
# --------------------------------------------------------------------------- #


def patch_rope_for_grad(rope: Any) -> Any | None:
    """Use eager RoPE when autograd is needed (flashinfer has no backward)."""
    if getattr(rope, "_qvla_rope_grad_patched", False):
        return None
    orig_forward = rope.forward

    def forward(positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor):
        if torch.is_grad_enabled():
            return rope._forward_eager(positions, q, k)
        return orig_forward(positions, q, k)

    rope._qvla_orig_forward = orig_forward  # type: ignore[attr-defined]
    rope.forward = forward  # type: ignore[method-assign]
    rope._qvla_rope_grad_patched = True
    logger.info("Patched RotaryEmbedding for autograd (eager RoPE fallback).")
    return rope


def restore_rope_for_grad(rope: Any | None) -> None:
    if rope is None or not getattr(rope, "_qvla_rope_grad_patched", False):
        return
    rope.forward = rope._qvla_orig_forward  # type: ignore[method-assign]
    rope._qvla_rope_grad_patched = False


__all__ = [
    "linear_with_bias",
    "patch_adarmsnorm_kernels_for_grad",
    "patch_gemma_rmsnorm_kernels_for_grad",
    "patch_layernorms_for_grad",
    "patch_linears_for_grad",
    "patch_mlp_activations_for_grad",
    "patch_rmsnorms_for_grad",
    "patch_rope_for_grad",
    "restore_adarmsnorm_kernels_for_grad",
    "restore_gemma_rmsnorm_kernels_for_grad",
    "restore_layernorms_for_grad",
    "restore_linears_for_grad",
    "restore_mlp_activations_for_grad",
    "restore_rmsnorms_for_grad",
    "restore_rope_for_grad",
]
