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

from phyai.payload import LLMForwardBatch, VisionForwardBatch

from qvla.adapters.pi05.attention_grad import (
    attach_attention_metadata,
    clear_prefix_kv_tape,
    patch_attention_for_grad,
    restore_attention_for_grad,
)
from qvla.build.grad_patches import (
    linear_with_bias,
    patch_adarmsnorm_kernels_for_grad,
    patch_gemma_rmsnorm_kernels_for_grad,
    patch_linears_for_grad,
    patch_mlp_activations_for_grad,
    patch_rope_for_grad,
    restore_adarmsnorm_kernels_for_grad,
    restore_gemma_rmsnorm_kernels_for_grad,
    restore_linears_for_grad,
    restore_mlp_activations_for_grad,
    restore_rope_for_grad,
)

if TYPE_CHECKING:
    from phyai.models.pi05.scheduler_ws1_pi05 import PI05Request, PI05WS1Scheduler


logger = logging.getLogger(__name__)


def patch_llm_ops_for_grad(scheduler: Any) -> tuple[Any | None, list[Any], list[Any], list[Any]]:
    """Patch paligemma prefix linears, norms, MLP activations, and RoPE for autograd."""
    lm = scheduler.llm_runner.paligemma_lm
    return (
        patch_rope_for_grad(scheduler.llm_runner.rope),
        patch_linears_for_grad(lm, log_label="LLM"),
        patch_gemma_rmsnorm_kernels_for_grad(lm, log_label="LLM"),
        patch_mlp_activations_for_grad(lm, log_label="LLM"),
    )


def restore_llm_ops_for_grad(
    rope: Any | None,
    linears: list[Any],
    norms: list[Any],
    mlps: list[Any],
) -> None:
    restore_rope_for_grad(rope)
    restore_linears_for_grad(linears)
    restore_gemma_rmsnorm_kernels_for_grad(norms)
    restore_mlp_activations_for_grad(mlps)


def patch_expert_ops_for_grad(
    scheduler: Any,
) -> tuple[Any | None, list[Any], list[Any], list[Any]]:
    """Patch RoPE, linears, norms, and MLP activations for autograd."""
    stack = scheduler.expert_runner.expert_stack
    rope = patch_rope_for_grad(scheduler.expert_runner.rope)
    norms = patch_adarmsnorm_kernels_for_grad(stack, log_label="expert")
    linears = patch_linears_for_grad(stack, log_label="expert")
    mlps = patch_mlp_activations_for_grad(stack, log_label="expert")
    return rope, norms, mlps, linears


def restore_expert_ops_for_grad(
    rope: Any | None,
    norms: list[Any],
    mlps: list[Any],
    linears: list[Any] | None = None,
) -> None:
    restore_rope_for_grad(rope)
    restore_adarmsnorm_kernels_for_grad(norms)
    restore_mlp_activations_for_grad(mlps)
    restore_linears_for_grad(linears or [])


@contextlib.contextmanager
def differentiable_inference_context(scheduler: Any) -> Iterator[None]:
    """Patch pi0.5 runners for one Fisher / differentiable calibration pass.

    Installs eager attention on LLM + expert, ``F.linear`` fallbacks, and the
    other autograd shims required by :func:`differentiable_step`. Restores all
    patches on exit.
    """
    attn = patch_attention_for_grad(scheduler, llm=True, expert=True)
    heads = patch_action_heads_for_grad(scheduler)
    llm_ops = patch_llm_ops_for_grad(scheduler)
    expert_ops = patch_expert_ops_for_grad(scheduler)
    try:
        yield
    finally:
        restore_attention_for_grad(attn)
        restore_action_heads_for_grad(heads)
        restore_llm_ops_for_grad(*llm_ops)
        restore_expert_ops_for_grad(*expert_ops)


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
            return linear_with_bias(heads.action_in_proj, x)
        return orig_embed(x)

    def project_action(x: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():
            return linear_with_bias(heads.action_out_proj, x)
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
    if torch.is_grad_enabled():
        # Prefix embeddings are built outside the noise leaf; reconnect the LLM
        # prefill stack so Fisher can backprop from actions through prefix KV tape.
        packed = packed.detach().requires_grad_(True)
    attach_attention_metadata(
        scheduler,
        prefix_meta=layout.prefix_meta,
        joint_meta=layout.joint_meta,
    )
    clear_prefix_kv_tape(scheduler.kv_pool)

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

    Expert attention writes K/V into the scheduler's
    :class:`~phyai.cache.kv_cache_pool.KVCachePool` via in-place ``index_put_``.
    After ``loss.backward()`` those buffers can retain freed-graph references;
    reusing them on the next sample triggers "backward through the graph a
    second time". Replacing each buffer with a detached view breaks the link.
    """
    pool = scheduler.kv_pool
    clear_prefix_kv_tape(pool)
    for layer_id in range(pool.num_layers):
        pool.k_buffers[layer_id] = pool.k_buffers[layer_id].detach()
        pool.v_buffers[layer_id] = pool.v_buffers[layer_id].detach()


__all__ = [
    "differentiable_inference_context",
    "differentiable_step",
    "force_eager_runners",
    "patch_action_heads_for_grad",
    "patch_expert_ops_for_grad",
    "patch_llm_ops_for_grad",
    "reset_differentiable_state",
    "restore_action_heads_for_grad",
    "restore_expert_ops_for_grad",
    "restore_llm_ops_for_grad",
]
