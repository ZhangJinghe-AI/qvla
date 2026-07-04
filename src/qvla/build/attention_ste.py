"""Straight-through-estimator wrappers for flashinfer paged-KV attention.

FlashInfer kernels do not implement autograd, and phyai's ``eager`` paged
backend only supports *contiguous* KV slabs — pi0.5's prefix layout is
non-contiguous (image block + lang block), so ``attn=eager`` cannot run
inference at all.

For the Fisher pass we keep **flashinfer in the forward** (deployment match)
and route backward through a pure-PyTorch gather +
:func:`~phyai.layers.attention.common.eager_attn` surrogate. Patching the
expert runner alone is sufficient for DiT-only analysis.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import torch

from phyai.layers.attention.common import eager_attn, repeat_kv


logger = logging.getLogger(__name__)

_OriginalForward = Callable[..., torch.Tensor]


def _gather_joint_kv_segment(
    idx: torch.Tensor,
    k_suffix: torch.Tensor,
    v_suffix: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    *,
    suffix_slot_base: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather one ragged KV segment with autograd through suffix ``k``/``v``.

    Joint indices interleave prefix slots (LLM cache, no grad needed) and
  suffix slots (current expert step). Reading suffix rows from the in-place
    KV pool breaks the graph from ``qkv_proj``; route suffix gathers through
    the live ``k``/``v`` tensors instead.
    """
    idx_long = idx.long()
    n = int(idx_long.numel())
    if n == 0:
        empty = k_suffix.new_zeros(0, *k_suffix.shape[1:])
        return empty, empty.clone()

    is_suffix = idx_long >= suffix_slot_base
    k_seg = torch.empty((n, *k_suffix.shape[1:]), device=k_suffix.device, dtype=k_suffix.dtype)
    v_seg = torch.empty_like(k_seg)

    if (~is_suffix).any():
        prefix_idx = idx_long[~is_suffix]
        k_seg[~is_suffix] = k_pool.index_select(0, prefix_idx).detach()
        v_seg[~is_suffix] = v_pool.index_select(0, prefix_idx).detach()
    if is_suffix.any():
        suffix_tok = idx_long[is_suffix] - suffix_slot_base
        k_seg[is_suffix] = k_suffix.index_select(0, suffix_tok)
        v_seg[is_suffix] = v_suffix.index_select(0, suffix_tok)
    return k_seg, v_seg


def _surrogate_ragged_paged_attn(
    layer: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ctx: Any,
    meta: Any,
    *,
    suffix_slot_base: int,
    write_kv: bool = True,
) -> torch.Tensor:
    """Differentiable paged-KV attention via index-gather + eager_attn."""
    if meta is None:
        raise RuntimeError(
            "Sensitivity attention surrogate missing metadata. "
            "Call attach_sensitivity_metadata() before forward."
        )
    if (
        meta.cu_seqlens_q is None
        or meta.paged_kv_indptr is None
        or meta.paged_kv_indices is None
    ):
        raise ValueError("Attention metadata is missing paged-KV fields.")

    if write_kv:
        ctx.kv_pool.write_kv(layer.layer_id, ctx.write_indices, k, v)
    k_cache, v_cache = ctx.kv_pool.kv_buffer(layer.layer_id)
    # page_size == 1 → squeeze the singleton page axis.
    k_pool = k_cache.squeeze(1)
    v_pool = v_cache.squeeze(1)

    cu_q = meta.cu_seqlens_q.to(torch.int64)
    indptr = meta.paged_kv_indptr.to(torch.int64)
    indices = meta.paged_kv_indices.to(torch.int64)

    segments: list[torch.Tensor] = []
    for b in range(indptr.numel() - 1):
        q_start, q_end = int(cu_q[b].item()), int(cu_q[b + 1].item())
        n_tok = q_end - q_start
        if n_tok == 0:
            continue
        kv_start, kv_end = int(indptr[b].item()), int(indptr[b + 1].item())
        if kv_end == kv_start:
            segments.append(q.new_zeros(n_tok, q.shape[1], q.shape[2]))
            continue
        idx = indices[kv_start:kv_end]
        k_seg, v_seg = _gather_joint_kv_segment(
            idx,
            k,
            v,
            k_pool,
            v_pool,
            suffix_slot_base=suffix_slot_base,
        )
        qi = q[q_start:q_end].transpose(0, 1).unsqueeze(0)
        ki = repeat_kv(
            k_seg.transpose(0, 1).unsqueeze(0),
            layer.num_heads,
            layer.num_kv_heads,
        )
        vi = repeat_kv(
            v_seg.transpose(0, 1).unsqueeze(0),
            layer.num_heads,
            layer.num_kv_heads,
        )
        oi = eager_attn(
            qi,
            ki,
            vi,
            scale=layer.scale,
            causal=layer.causal,
            sliding_window=None,
            logits_soft_cap=getattr(layer, "logits_soft_cap", None),
        )
        segments.append(oi.squeeze(0).transpose(0, 1))
    if not segments:
        return q.new_zeros(q.shape)
    return torch.cat(segments, dim=0)


def _make_ste_forward(
    original: _OriginalForward,
    meta_attr: str,
    runner: Any,
) -> _OriginalForward:
    def ste_forward(layer, q, k, v, ctx):
        y_deploy = original(layer, q, k, v, ctx)
        if not torch.is_grad_enabled():
            return y_deploy
        meta = getattr(runner, meta_attr, None)
        suffix_slot_base = int(getattr(runner, "_sensitivity_suffix_slot_base", 0))
        y_sur = _surrogate_ragged_paged_attn(
            layer,
            q,
            k,
            v,
            ctx,
            meta,
            suffix_slot_base=suffix_slot_base,
            write_kv=True,
        )
        return y_deploy + (y_sur - y_deploy.detach())

    return ste_forward


def attach_sensitivity_metadata(
    scheduler: Any,
    *,
    prefix_meta: Any,
    joint_meta: Any,
) -> None:
    """Stash per-inference attention metadata for the STE surrogate."""
    scheduler.llm_runner._sensitivity_prefix_meta = prefix_meta
    scheduler.expert_runner._sensitivity_joint_meta = joint_meta
    scheduler.expert_runner._sensitivity_suffix_slot_base = int(scheduler.suffix_base)


def patch_attention_ste(
    scheduler: Any,
    *,
    llm: bool = False,
    expert: bool = True,
) -> list[tuple[Any, _OriginalForward]]:
    """Wrap flashinfer forwards with STE surrogates.

    For DiT-only action sensitivity (noise leaf), patching the expert runner
    alone is sufficient and faster.
    """
    patched: list[tuple[Any, _OriginalForward]] = []

    if llm:
        llm_be = scheduler.llm_runner.attn_backend
        llm_orig = llm_be.forward
        llm_be.forward = _make_ste_forward(  # type: ignore[method-assign]
            llm_orig, "_sensitivity_prefix_meta", scheduler.llm_runner
        )
        patched.append((llm_be, llm_orig))

    if expert:
        exp_be = scheduler.expert_runner.attn_backend
        exp_orig = exp_be.forward
        exp_be.forward = _make_ste_forward(  # type: ignore[method-assign]
            exp_orig, "_sensitivity_joint_meta", scheduler.expert_runner
        )
        patched.append((exp_be, exp_orig))

    if patched:
        parts: list[str] = []
        if llm:
            parts.append("llm")
        if expert:
            parts.append("expert")
        logger.info("Patched attention STE on: %s.", ", ".join(parts))
    return patched


def restore_attention_ste(patched: list[tuple[Any, _OriginalForward]]) -> None:
    for backend, original in patched:
        backend.forward = original  # type: ignore[method-assign]


__all__ = [
    "attach_sensitivity_metadata",
    "patch_attention_ste",
    "restore_attention_ste",
]
