"""Autograd-friendly pi0.5 forward path for the Fisher (policy-aware rotation) pass.

phyai's ``PI05WS1Scheduler.step`` is inference-only:

* ``@torch.no_grad()`` on the whole step.
* Vision embeddings are gathered with in-place writes into a ``torch.zeros``
  buffer — that breaks the graph from ``pixel_values`` to downstream layers.
* ``_pack_prefix`` scatters into another zero buffer the same way.

CUDA-graph replay also blocks autograd. This module provides eager,
graph-preserving equivalents without editing ``phyai/``.
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from phyai.payload import LLMForwardBatch, VisionForwardBatch

from qvla.build.attention_ste import (
    attach_sensitivity_metadata,
    patch_attention_ste,
    restore_attention_ste,
)

if TYPE_CHECKING:
    from phyai.models.pi05.scheduler_ws1_pi05 import PI05Request, PI05WS1Scheduler


logger = logging.getLogger(__name__)


def _tensor_in_autograd_graph(t: torch.Tensor) -> bool:
    """True when ``t`` participates in the current autograd graph."""
    return torch.is_grad_enabled() and (t.requires_grad or t.grad_fn is not None)


def _linear_with_bias(layer: Any, x: torch.Tensor) -> torch.Tensor:
    """Autograd-safe matmul for phyai ``ReplicatedLinear`` weights."""
    bias = layer.bias if not getattr(layer, "skip_bias_add", False) else None
    return F.linear(x, layer.weight, bias)


def _linear_forward_with_bias_tuple(layer: Any, x: torch.Tensor) -> tuple[torch.Tensor, Any]:
    """``F.linear`` forward matching phyai ``LinearBase``'s ``(y, bias)`` contract."""
    y = _linear_with_bias(layer, x)
    bias_out = layer.bias if getattr(layer, "skip_bias_add", False) else None
    return y, bias_out


def _is_expert_stack_linear(mod: nn.Module) -> bool:
    """True for phyai linears inside the expert stack (not ``QuantLinear``)."""
    from qvla.runtime.quant_linear import QuantLinear
    from qvla.runtime.wrap import _is_linear_like

    if isinstance(mod, QuantLinear):
        return False
    return _is_linear_like(mod)


def patch_expert_linears_for_grad(expert_stack: Any) -> list[Any]:
    """Route expert-stack linears through ``F.linear`` when grad is needed.

    FlashInfer GEMM (the default phyai linear backend) has no autograd.  During
    the Fisher / sensitivity passes the expert stack is still full-precision, so
    without this patch gradients stop at the first ``qkv_proj`` and every DiT
    layer reports zero Fisher sensitivity.
    """
    patched: list[Any] = []
    for mod in expert_stack.modules():
        if not _is_expert_stack_linear(mod):
            continue
        if getattr(mod, "_qvla_linear_grad_patched", False):
            continue
        orig_forward = mod.forward

        def make_forward(layer: Any, original: Any):
            def forward(x: torch.Tensor, *args: Any, **kwargs: Any):
                if torch.is_grad_enabled():
                    return _linear_forward_with_bias_tuple(layer, x)
                return original(x, *args, **kwargs)

            return forward

        mod._qvla_orig_forward = orig_forward  # type: ignore[attr-defined]
        mod.forward = make_forward(mod, orig_forward)  # type: ignore[method-assign]
        mod._qvla_linear_grad_patched = True
        patched.append(mod)
    if patched:
        logger.info(
            "Patched %d expert Linear(s) to F.linear for autograd.",
            len(patched),
        )
    return patched


def restore_expert_linears_for_grad(patched: list[Any]) -> None:
    for mod in patched:
        if getattr(mod, "_qvla_linear_grad_patched", False):
            mod.forward = mod._qvla_orig_forward  # type: ignore[method-assign]
            mod._qvla_linear_grad_patched = False


def patch_expert_norms_for_grad(scheduler: Any) -> list[Any]:
    """Switch expert AdaRMSNorm kernels to the torch reference when grad is needed.

    The default ``phyai-kernel`` Triton path has no autograd, which disconnects
    the DiT stack even when attention STE and action-head patches are active.
    """
    from phyai.layers.layer_norm import AdaRMSNorm, _torch_adarmsnorm

    patched: list[Any] = []
    stack = scheduler.expert_runner.expert_stack
    for mod in stack.modules():
        if not isinstance(mod, AdaRMSNorm):
            continue
        if getattr(mod, "_qvla_torch_adarms_patched", False):
            continue
        mod._qvla_orig_adarms_kernel = mod._adarms_kernel  # type: ignore[attr-defined]
        mod._adarms_kernel = _torch_adarmsnorm
        mod._qvla_torch_adarms_patched = True
        patched.append(mod)
    if patched:
        logger.info("Patched %d expert AdaRMSNorm module(s) to torch backend.", len(patched))
    return patched


def restore_expert_norms_for_grad(patched: list[Any]) -> None:
    for mod in patched:
        if getattr(mod, "_qvla_torch_adarms_patched", False):
            mod._adarms_kernel = mod._qvla_orig_adarms_kernel  # type: ignore[attr-defined]
            mod._qvla_torch_adarms_patched = False


def _gelu_tanh_and_mul_torch(fused: torch.Tensor) -> torch.Tensor:
    gate, up = fused.chunk(2, dim=-1)
    return F.gelu(gate, approximate="tanh") * up


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


def patch_expert_mlp_activations_for_grad(expert_stack: Any) -> list[Any]:
    """Route GeGLU through torch when grad is needed (flashinfer has no backward)."""
    from phyai.layers.mlp.dense_mlp import DenseMLP

    patched: list[Any] = []
    for mod in expert_stack.modules():
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
            "Patched %d expert DenseMLP activation(s) to torch backend.",
            len(patched),
        )
    return patched


def restore_expert_mlp_activations_for_grad(patched: list[Any]) -> None:
    for mod in patched:
        if getattr(mod, "_qvla_act_patched", False):
            mod._act_and_mul = mod._qvla_orig_act_and_mul  # type: ignore[attr-defined]
            mod._qvla_act_patched = False


def _gated_residual(residual: torch.Tensor, out: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """``residual + out * gate`` with the skip path stopped for autograd."""
    if _tensor_in_autograd_graph(residual) or _tensor_in_autograd_graph(out):
        return torch.addcmul(residual.detach(), out, gate)
    return torch.addcmul(residual, out, gate)


def patch_expert_gated_residuals_for_grad(expert_stack: Any) -> list[Any]:
    """Stop gated residuals from carrying gradients around quantized sublayers."""
    from phyai.models.pi05.modeling_pi05 import PI05ExpertLayer

    patched: list[Any] = []
    for layer in expert_stack.layers:
        if not isinstance(layer, PI05ExpertLayer):
            continue
        if getattr(layer, "_qvla_residual_patched", False):
            continue
        orig_forward = layer.forward

        def make_forward(lyr: PI05ExpertLayer):
            def forward(
                h: torch.Tensor,
                position_ids: torch.Tensor,
                cond: torch.Tensor | None,
                rope: Any,
                attn_ctx: Any,
                *,
                modulation: Any | None = None,
            ) -> torch.Tensor:
                residual = h
                if modulation is None:
                    n, gate_attn = lyr.input_layernorm(h, cond)
                else:
                    n, gate_attn = lyr.input_layernorm(
                        h, modulation=modulation.input_ln
                    )
                fused, _ = lyr.qkv_proj(n)
                q, k, v = lyr._split_qkv(fused, h.shape[:-1])
                q, k = rope(position_ids, q, k)
                attn_out = lyr.attn(q, k, v, attn_ctx)
                attn_flat = attn_out.reshape(*attn_out.shape[:-2], -1)
                out, _ = lyr.o_proj(attn_flat)
                h = _gated_residual(residual, out, gate_attn)
                residual = h
                if modulation is None:
                    m, gate_mlp = lyr.post_attention_layernorm(h, cond)
                else:
                    m, gate_mlp = lyr.post_attention_layernorm(
                        h, modulation=modulation.post_attention_ln
                    )
                m = lyr.mlp(m)
                return _gated_residual(residual, m, gate_mlp)

            return forward

        layer._qvla_orig_forward = orig_forward  # type: ignore[attr-defined]
        layer.forward = make_forward(layer)  # type: ignore[method-assign]
        layer._qvla_residual_patched = True
        patched.append(layer)
    if patched:
        logger.info(
            "Patched %d expert layer(s) to block gated-residual grad shortcuts.",
            len(patched),
        )
    return patched


def restore_expert_gated_residuals_for_grad(patched: list[Any]) -> None:
    for layer in patched:
        if getattr(layer, "_qvla_residual_patched", False):
            layer.forward = layer._qvla_orig_forward  # type: ignore[method-assign]
            layer._qvla_residual_patched = False


def patch_expert_ops_for_grad(
    scheduler: Any,
) -> tuple[Any | None, list[Any], list[Any], list[Any], list[Any]]:
    """Patch RoPE, linears, norms, MLP activations, and gated residuals for autograd."""
    stack = scheduler.expert_runner.expert_stack
    rope = patch_rope_for_grad(scheduler.expert_runner.rope)
    norms = patch_expert_norms_for_grad(scheduler)
    linears = patch_expert_linears_for_grad(stack)
    mlps = patch_expert_mlp_activations_for_grad(stack)
    residuals = patch_expert_gated_residuals_for_grad(stack)
    return rope, norms, mlps, residuals, linears


def restore_expert_ops_for_grad(
    rope: Any | None,
    norms: list[Any],
    mlps: list[Any],
    residuals: list[Any],
    linears: list[Any] | None = None,
) -> None:
    restore_rope_for_grad(rope)
    restore_expert_norms_for_grad(norms)
    restore_expert_mlp_activations_for_grad(mlps)
    restore_expert_gated_residuals_for_grad(residuals)
    restore_expert_linears_for_grad(linears or [])


@contextlib.contextmanager
def differentiable_inference_context(scheduler: Any) -> Iterator[None]:
    """Patch pi0.5 runners for one Fisher / differentiable calibration pass.

    Installs attention STE, ``F.linear`` fallbacks on expert linears, and the
    other autograd shims required by :func:`differentiable_step`. Restores all
    patches on exit.
    """
    attn = patch_attention_ste(scheduler)
    heads = patch_action_heads_for_grad(scheduler)
    ops = patch_expert_ops_for_grad(scheduler)
    try:
        yield
    finally:
        restore_attention_ste(attn)
        restore_action_heads_for_grad(heads)
        restore_expert_ops_for_grad(*ops)


def patch_action_heads_for_grad(scheduler: Any) -> tuple[Any, Any, Any] | None:
    """Route action in/out projections through ``F.linear`` when grad is needed.

    QVLA packs quantize expert-stack linears only. ``action_in_proj`` and
    ``action_out_proj`` remain phyai ``ReplicatedLinear`` modules whose default
    flashinfer GEMM has no autograd, which disconnects the noise leaf from the
    DiT stack. Returns ``(heads, orig_embed, orig_project)`` on first patch,
    else ``None``.
    """
    heads = scheduler.expert_runner.heads
    if getattr(heads, "_qvla_grad_patched", False):
        return None

    orig_embed = heads.embed_action
    orig_project = heads.project_action

    def embed_action(x: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():
            return _linear_with_bias(heads.action_in_proj, x)
        return orig_embed(x)

    def project_action(x: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():
            return _linear_with_bias(heads.action_out_proj, x)
        return orig_project(x)

    heads.embed_action = embed_action  # type: ignore[method-assign]
    heads.project_action = project_action  # type: ignore[method-assign]
    heads._qvla_grad_patched = True  # type: ignore[attr-defined]
    logger.info("Patched action in/out projections for autograd (F.linear fallback).")
    return heads, orig_embed, orig_project


def restore_action_heads_for_grad(
    patched: tuple[Any, Any, Any] | None,
) -> None:
    if patched is None:
        return
    heads, orig_embed, orig_project = patched
    heads.embed_action = orig_embed  # type: ignore[method-assign]
    heads.project_action = orig_project  # type: ignore[method-assign]
    heads._qvla_grad_patched = False  # type: ignore[attr-defined]


def force_eager_runners(scheduler: PI05WS1Scheduler) -> None:
    """Disable CUDA-graph capture/replay on every pi0.5 runner."""
    for runner_name in ("vision_runner", "llm_runner", "expert_runner"):
        runner = getattr(scheduler, runner_name, None)
        if runner is None:
            continue
        runner.use_cuda_graph = False
        graph = getattr(runner, "graph", None)
        if graph is not None:
            runner.graph = None
        graphs = getattr(runner, "graphs", None)
        if isinstance(graphs, dict):
            graphs.clear()


def _pack_prefix_diff(
    scheduler: PI05WS1Scheduler,
    image_embs: torch.Tensor,
    lang_embs: torch.Tensor,
    layout,
) -> torch.Tensor:
    """Same layout as ``PI05WS1Scheduler._pack_prefix`` but autograd-safe."""
    max_B = scheduler.max_batch_size
    n_ps = layout.n_per_sample
    n_img = scheduler.image_token_count
    lang_bucket = n_ps - n_img
    mask = layout.lang_mask.to(lang_embs.dtype)[..., None]
    lang_block = lang_embs[:, :lang_bucket] * mask
    packed_rows = torch.cat([image_embs, lang_block], dim=1)
    return packed_rows.reshape(max_B * n_ps, packed_rows.shape[-1])


def _gather_image_embs(
    scheduler: PI05WS1Scheduler,
    pixel_values: torch.Tensor,
    *,
    actual_B: int,
    max_B: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Stack per-sample vision outputs instead of in-place buffer writes."""
    pixel_values = pixel_values.to(device=device, dtype=dtype)
    chunks: list[torch.Tensor] = []
    for b in range(actual_B):
        vision_out = scheduler.vision_runner.forward(
            VisionForwardBatch(pixel_values=pixel_values[b])
        )
        chunks.append(vision_out.flatten(0, 1))
    image_embs = torch.stack(chunks, dim=0)
    if actual_B < max_B:
        pad = torch.zeros(
            max_B - actual_B,
            *image_embs.shape[1:],
            dtype=image_embs.dtype,
            device=image_embs.device,
        )
        image_embs = torch.cat([image_embs, pad], dim=0)
    return image_embs


def differentiable_step(
    scheduler: PI05WS1Scheduler,
    request: PI05Request,
) -> torch.Tensor:
    """Run one pi0.5 inference with autograd preserved.

    Mirrors ``PI05WS1Scheduler.step`` but replaces graph-breaking buffer
    staging with ``torch.stack`` / ``torch.cat``. Call
    :func:`force_eager_runners` before ``scheduler.setup()`` so runners stay
    in eager mode.
    """
    cfg = scheduler.cfg
    device = scheduler.device
    dtype = scheduler.params_dtype
    scheduler._validate(request)

    actual_B = int(request.pixel_values.shape[0])
    max_B = scheduler.max_batch_size

    lang_lens_cpu = tuple(int(x) for x in request.lang_lens.tolist())
    n_per_sample = scheduler._bucket_n_per_sample(max(lang_lens_cpu, default=0))
    key = (actual_B, lang_lens_cpu)
    layout = scheduler._layout_cache.get(key)
    if layout is None:
        layout = scheduler._build_layout(actual_B, lang_lens_cpu, n_per_sample)
        scheduler._layout_cache[key] = layout
    plan_changed = key != scheduler._last_layout_key

    image_embs = _gather_image_embs(
        scheduler,
        request.pixel_values,
        actual_B=actual_B,
        max_B=max_B,
        device=device,
        dtype=dtype,
    )

    if actual_B < max_B:
        input_ids_padded = torch.zeros(
            max_B, cfg.tokenizer_max_length, dtype=torch.int64, device=device
        )
        input_ids_padded[:actual_B] = request.input_ids.to(
            device=device, dtype=torch.int64
        )
    else:
        input_ids_padded = request.input_ids.to(device=device, dtype=torch.int64)

    lang_embs = scheduler.model.paligemma_lm.embed_lang(input_ids_padded)
    if lang_embs.dtype != dtype:
        lang_embs = lang_embs.to(dtype)
    packed = _pack_prefix_diff(scheduler, image_embs, lang_embs, layout)
    attach_sensitivity_metadata(
        scheduler,
        prefix_meta=layout.prefix_meta,
        joint_meta=layout.joint_meta,
    )

    if plan_changed:
        scheduler.prefix_static.reset()
        scheduler.suffix_static.reset()
        _ = scheduler.prefix_static.allocate(layout.n_real_total)
        _ = scheduler.suffix_static.allocate(max_B * cfg.chunk_size)
        scheduler.llm_runner.plan_inference(layout.prefix_meta)

    llm_batch = LLMForwardBatch(
        hidden_states=packed,
        position_ids=layout.position_ids,
        write_indices=layout.write_indices,
    )
    scheduler.llm_runner.forward(llm_batch, n_per_sample=layout.n_per_sample)

    if plan_changed:
        scheduler.expert_runner.plan_inference(layout.joint_meta)
    scheduler._last_layout_key = key

    if scheduler.time_emb_table is None:
        raise RuntimeError(
            "differentiable_step() called before scheduler.setup(); "
            "time_emb_table is missing."
        )

    if request.noise is None:
        noise = torch.randn(
            max_B,
            cfg.chunk_size,
            cfg.max_action_dim,
            dtype=dtype,
            device=device,
        )
    else:
        noise = torch.zeros(
            max_B,
            cfg.chunk_size,
            cfg.max_action_dim,
            dtype=dtype,
            device=device,
        )
        noise[:actual_B] = request.noise.to(device=device, dtype=dtype)

    # Flow-matching starts from noise; making it a leaf with ``requires_grad``
    # gives a clean autograd path through the DiT expert (suffix tokens) even
    # when prefix KV writes use in-place cache scatter (non-differentiable).
    noise = noise.detach().requires_grad_(True)
    scheduler._qvla_sensitivity_noise = noise  # type: ignore[attr-defined]

    x_t = scheduler.expert_runner.forward(noise)
    return x_t[:actual_B].clone()


def reset_differentiable_state(scheduler: Any) -> None:
    """Clear shared autograd state between sensitivity forward+backward passes.

    The attention STE surrogate scatters K/V into the scheduler's
    :class:`~phyai.cache.kv_cache_pool.KVCachePool` via in-place ``index_put_``.
    After ``loss.backward()`` those buffers can retain freed-graph references;
    reusing them on the next sample triggers "backward through the graph a
    second time". Replacing each buffer with a detached view breaks the link.
    """
    pool = scheduler.kv_pool
    for layer_id in range(pool.num_layers):
        pool.k_buffers[layer_id] = pool.k_buffers[layer_id].detach()
        pool.v_buffers[layer_id] = pool.v_buffers[layer_id].detach()


__all__ = [
    "differentiable_inference_context",
    "differentiable_step",
    "force_eager_runners",
    "patch_action_heads_for_grad",
    "patch_expert_linears_for_grad",
    "patch_expert_mlp_activations_for_grad",
    "patch_expert_norms_for_grad",
    "patch_expert_ops_for_grad",
    "patch_rope_for_grad",
    "reset_differentiable_state",
    "restore_action_heads_for_grad",
    "restore_expert_linears_for_grad",
    "restore_expert_mlp_activations_for_grad",
    "restore_expert_norms_for_grad",
    "restore_expert_ops_for_grad",
    "restore_rope_for_grad",
]
