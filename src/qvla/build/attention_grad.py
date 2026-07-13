"""Autograd-friendly paged-KV attention for the Fisher / differentiable pass.

FlashInfer has no autograd, and phyai's ``eager`` paged backend only supports
contiguous KV slabs — pi0.5's prefix layout is non-contiguous. When grad is
needed we route attention through a pure-PyTorch gather +
:func:`~phyai.layers.attention.common.eager_attn` implementation; inference
keeps the original flashinfer forward.

For Fisher sensitivity on LLM quant targets, prefix K/V written during the
paligemma prefill are recorded in a per-layer *tape* so expert joint attention
can read them with autograd instead of ``detach``ing the KV pool.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import torch

from phyai.layers.attention.common import eager_attn, repeat_kv


logger = logging.getLogger(__name__)

_OriginalForward = Callable[..., torch.Tensor]

# layer_id -> (write_indices, k, v) from the LLM prefix prefill (grad-enabled).
PrefixKVTape = dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


def clear_prefix_kv_tape(kv_pool: Any) -> None:
    """Drop recorded prefix K/V before the next Fisher sample."""
    tape = getattr(kv_pool, "_qvla_prefix_kv_tape", None)
    if tape is not None:
        tape.clear()


def _record_prefix_kv(
    kv_pool: Any,
    layer_id: int,
    write_indices: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    if not torch.is_grad_enabled():
        return
    tape: PrefixKVTape = getattr(kv_pool, "_qvla_prefix_kv_tape", None)
    if tape is None:
        tape = {}
        kv_pool._qvla_prefix_kv_tape = tape  # type: ignore[attr-defined]
    tape[int(layer_id)] = (write_indices, k, v)


def _rows_for_pool_slots(
    write_indices: torch.Tensor,
    pool_slots: torch.Tensor,
) -> torch.Tensor:
    """Map global KV-pool slot ids to rows in a live ``(N, H, D)`` K/V tensor."""
    slots = pool_slots.long()
    wi = write_indices.long()
    size = int(max(slots.max().item(), wi.max().item())) + 1
    inv = pool_slots.new_full((size,), -1)
    inv.scatter_(
        0,
        wi,
        torch.arange(wi.numel(), device=wi.device, dtype=inv.dtype),
    )
    rows = inv[slots]
    if (rows < 0).any():
        missing = slots[rows < 0]
        raise RuntimeError(
            "Prefix KV tape is missing pool slot(s) required by joint attention: "
            f"{missing.tolist()[:8]}{'...' if missing.numel() > 8 else ''}. "
            "Ensure LLM prefix forward ran with autograd before the expert loop."
        )
    return rows


def _gather_joint_kv_segment(
    idx: torch.Tensor,
    k_live: torch.Tensor,
    v_live: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    *,
    suffix_slot_base: int,
    prefix_tape: PrefixKVTape | None,
    layer_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather one ragged KV segment; keep prefix rows on the autograd graph."""
    idx_long = idx.long()
    n = int(idx_long.numel())
    if n == 0:
        empty = k_live.new_zeros(0, *k_live.shape[1:])
        return empty, empty.clone()

    is_suffix = idx_long >= suffix_slot_base
    k_seg = torch.empty((n, *k_live.shape[1:]), device=k_live.device, dtype=k_live.dtype)
    v_seg = torch.empty_like(k_seg)

    if (~is_suffix).any():
        prefix_idx = idx_long[~is_suffix]
        if torch.is_grad_enabled():
            tape_entry = (prefix_tape or {}).get(int(layer_id))
            if tape_entry is None:
                raise RuntimeError(
                    f"Prefix KV tape missing for layer {layer_id} during autograd "
                    "gather. Run LLM prefix prefill with record_prefix_kv=True "
                    "before the expert denoise loop."
                )
            wi, k_pre, v_pre = tape_entry
            rows = _rows_for_pool_slots(wi, prefix_idx)
            k_seg[~is_suffix] = k_pre.index_select(0, rows)
            v_seg[~is_suffix] = v_pre.index_select(0, rows)
        else:
            k_seg[~is_suffix] = k_pool.index_select(0, prefix_idx)
            v_seg[~is_suffix] = v_pool.index_select(0, prefix_idx)

    if is_suffix.any():
        suffix_tok = idx_long[is_suffix] - suffix_slot_base
        k_seg[is_suffix] = k_live.index_select(0, suffix_tok)
        v_seg[is_suffix] = v_live.index_select(0, suffix_tok)
    return k_seg, v_seg


def _ragged_paged_attn_eager(
    layer: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ctx: Any,
    meta: Any,
    *,
    suffix_slot_base: int,
    write_kv: bool = True,
    record_prefix_kv: bool = False,
) -> torch.Tensor:
    """Differentiable paged-KV attention via index-gather + eager_attn."""
    if meta is None:
        raise RuntimeError(
            "Differentiable attention missing metadata. "
            "Call attach_attention_metadata() before forward."
        )
    if (
        meta.cu_seqlens_q is None
        or meta.paged_kv_indptr is None
        or meta.paged_kv_indices is None
    ):
        raise ValueError("Attention metadata is missing paged-KV fields.")

    if write_kv:
        ctx.kv_pool.write_kv(layer.layer_id, ctx.write_indices, k, v)
        if record_prefix_kv:
            _record_prefix_kv(ctx.kv_pool, layer.layer_id, ctx.write_indices, k, v)

    prefix_tape: PrefixKVTape | None = getattr(ctx.kv_pool, "_qvla_prefix_kv_tape", None)
    k_cache, v_cache = ctx.kv_pool.kv_buffer(layer.layer_id)
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
            prefix_tape=prefix_tape,
            layer_id=int(layer.layer_id),
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


def _make_grad_forward(
    original: _OriginalForward,
    meta_attr: str,
    runner: Any,
    *,
    record_prefix_kv: bool,
) -> _OriginalForward:
    def grad_forward(layer, q, k, v, ctx):
        if not torch.is_grad_enabled():
            return original(layer, q, k, v, ctx)
        meta = getattr(runner, meta_attr, None)
        suffix_slot_base = int(getattr(runner, "_sensitivity_suffix_slot_base", 0))
        return _ragged_paged_attn_eager(
            layer,
            q,
            k,
            v,
            ctx,
            meta,
            suffix_slot_base=suffix_slot_base,
            write_kv=True,
            record_prefix_kv=record_prefix_kv,
        )

    return grad_forward


def attach_attention_metadata(
    scheduler: Any,
    *,
    prefix_meta: Any,
    joint_meta: Any,
) -> None:
    """Stash per-inference attention metadata for the eager grad path."""
    suffix_base = int(scheduler.suffix_base)
    scheduler.llm_runner._sensitivity_prefix_meta = prefix_meta
    scheduler.llm_runner._sensitivity_suffix_slot_base = suffix_base
    scheduler.expert_runner._sensitivity_joint_meta = joint_meta
    scheduler.expert_runner._sensitivity_suffix_slot_base = suffix_base


def patch_attention_for_grad(
    scheduler: Any,
    *,
    llm: bool = True,
    expert: bool = True,
) -> list[tuple[Any, _OriginalForward]]:
    """Use eager paged attention when autograd is enabled."""
    patched: list[tuple[Any, _OriginalForward]] = []

    if llm:
        llm_be = scheduler.llm_runner.attn_backend
        llm_orig = llm_be.forward
        llm_be.forward = _make_grad_forward(  # type: ignore[method-assign]
            llm_orig, "_sensitivity_prefix_meta", scheduler.llm_runner, record_prefix_kv=True
        )
        patched.append((llm_be, llm_orig))

    if expert:
        exp_be = scheduler.expert_runner.attn_backend
        exp_orig = exp_be.forward
        exp_be.forward = _make_grad_forward(  # type: ignore[method-assign]
            exp_orig, "_sensitivity_joint_meta", scheduler.expert_runner, record_prefix_kv=False
        )
        patched.append((exp_be, exp_orig))

    if patched:
        parts: list[str] = []
        if llm:
            parts.append("llm")
        if expert:
            parts.append("expert")
        logger.info("Patched attention for autograd on: %s.", ", ".join(parts))
    return patched


def restore_attention_for_grad(patched: list[tuple[Any, _OriginalForward]]) -> None:
    for backend, original in patched:
        backend.forward = original  # type: ignore[method-assign]


__all__ = [
    "attach_attention_metadata",
    "clear_prefix_kv_tape",
    "patch_attention_for_grad",
    "restore_attention_for_grad",
]
