"""Paged-KV attention helpers for Fisher / differentiable / deterministic eval.

FlashInfer has no autograd, and phyai's ``eager`` paged backend only supports
contiguous KV slabs — pi0.5's prefix layout is non-contiguous. When grad is
needed we route attention through a pure-PyTorch gather +
:func:`~phyai.layers.attention.common.eager_attn` implementation; the default
inference path keeps the original flashinfer forward.

For Fisher sensitivity on LLM quant targets, prefix K/V written during the
paligemma prefill are recorded in a per-layer *tape* so expert joint attention
can read them with autograd instead of ``detach``ing the KV pool. The Fisher
grad path uses :func:`ragged_paged_attn_batched_grad` (padded batch matmul +
tape gather); non-uniform Q lengths raise instead of silently falling back to
the per-sample loop.

For deterministic eval without FlashInfer, :func:`patch_attention_for_inference`
replaces both runners with torch attention
(:func:`ragged_paged_attn_batched` by default, or per-sample
:func:`_ragged_paged_attn_eager`) and stashes plan metadata on the runners
(including expert ``pos_ids_suffix_buf`` for RoPE).
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import torch
import torch.nn.functional as F

from phyai.layers.attention.common import eager_attn, repeat_kv


logger = logging.getLogger(__name__)

_OriginalForward = Callable[..., torch.Tensor]
_OriginalPlanInference = Callable[..., None]
# (backend, original_forward, runner, original_plan_inference)
_InferencePatch = tuple[Any, _OriginalForward, Any, _OriginalPlanInference]

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


def _require_batched_attn_meta(meta: Any, *, what: str) -> None:
    if meta is None:
        raise RuntimeError(
            f"{what} missing metadata; plan_inference / attach_attention_metadata "
            "must stash it first."
        )
    if (
        meta.cu_seqlens_q is None
        or meta.paged_kv_indptr is None
        or meta.paged_kv_indices is None
    ):
        raise ValueError("Attention metadata is missing paged-KV fields.")


def _padded_kv_layout(
    meta: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
    """Return ``(idx_pad, key_valid, q_lens, B, S_q, S_kv)`` for batched attn.

    Requires uniform per-sample Q lengths; raises otherwise (no silent fallback).
    """
    cu_q = meta.cu_seqlens_q.to(torch.int64)
    indptr = meta.paged_kv_indptr.to(torch.int64)
    indices = meta.paged_kv_indices.to(torch.int64)
    B = int(indptr.numel() - 1)
    if B <= 0:
        raise RuntimeError(f"batched eager attention got empty batch (B={B}).")

    q_lens = cu_q[1:] - cu_q[:-1]
    kv_lens = indptr[1:] - indptr[:-1]
    if not torch.equal(q_lens, q_lens.new_full(q_lens.shape, int(q_lens[0].item()))):
        raise RuntimeError(
            "batched eager attention requires uniform per-sample Q lengths; "
            f"got q_lens={q_lens.tolist()}."
        )
    S_q = int(q_lens[0].item())
    S_kv = int(kv_lens.max().item()) if kv_lens.numel() else 0
    device = indices.device
    arange_kv = torch.arange(S_kv, device=device, dtype=torch.int64)
    key_valid = arange_kv.unsqueeze(0) < kv_lens.unsqueeze(1)
    starts = indptr[:-1]
    gather_pos = starts.unsqueeze(1) + arange_kv.unsqueeze(0)
    gather_pos = torch.where(key_valid, gather_pos, gather_pos.new_zeros(()))
    gather_pos = gather_pos.clamp(0, max(indices.numel() - 1, 0))
    idx_pad = indices[gather_pos]
    idx_pad = torch.where(key_valid, idx_pad, idx_pad.new_zeros(()))
    return idx_pad, key_valid, q_lens, B, S_q, S_kv


def _padded_batched_attn(
    layer: Any,
    q: torch.Tensor,
    k_pad: torch.Tensor,
    v_pad: torch.Tensor,
    key_valid: torch.Tensor,
    *,
    B: int,
    S_q: int,
    S_kv: int,
) -> torch.Tensor:
    """Batched QK^T / softmax / AV on padded KV (fp32 softmax)."""
    qi = q.view(B, S_q, q.shape[1], q.shape[2]).transpose(1, 2)
    ki = repeat_kv(
        k_pad.transpose(1, 2),
        layer.num_heads,
        layer.num_kv_heads,
    )
    vi = repeat_kv(
        v_pad.transpose(1, 2),
        layer.num_heads,
        layer.num_kv_heads,
    )

    attn = torch.matmul(qi, ki.transpose(-2, -1)) * layer.scale
    soft_cap = getattr(layer, "logits_soft_cap", None)
    if soft_cap is not None:
        attn = soft_cap * torch.tanh(attn / soft_cap)
    if layer.causal:
        i = torch.arange(S_q, device=q.device).unsqueeze(1)
        j = torch.arange(S_kv, device=q.device).unsqueeze(0)
        causal_ok = (i + (S_kv - S_q)) >= j
        attn = attn.masked_fill(~causal_ok.view(1, 1, S_q, S_kv), float("-inf"))
    attn = attn.masked_fill(~key_valid[:, None, None, :], float("-inf"))
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    attn = torch.nan_to_num(attn, nan=0.0)
    out = torch.matmul(attn, vi)
    return out.transpose(1, 2).reshape(q.shape)


def _gather_joint_kv_padded(
    idx_pad: torch.Tensor,
    key_valid: torch.Tensor,
    k_live: torch.Tensor,
    v_live: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    *,
    suffix_slot_base: int,
    prefix_tape: PrefixKVTape | None,
    layer_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather padded ``(B, S_kv, ...)`` KV using the prefix tape under autograd.

    Invalid (padding) slots stay zero and must be masked by ``key_valid`` in
    attention — they are never looked up in the tape (avoids false misses).
    """
    B, S_kv = int(idx_pad.shape[0]), int(idx_pad.shape[1])
    tail = k_live.shape[1:]
    k_pad = k_live.new_zeros((B, S_kv, *tail))
    v_pad = v_live.new_zeros((B, S_kv, *v_live.shape[1:]))
    flat_valid = key_valid.reshape(-1)
    if not bool(flat_valid.any().item()):
        return k_pad, v_pad

    valid_idx = idx_pad.reshape(-1)[flat_valid]
    k_valid, v_valid = _gather_joint_kv_segment(
        valid_idx,
        k_live,
        v_live,
        k_pool,
        v_pool,
        suffix_slot_base=suffix_slot_base,
        prefix_tape=prefix_tape,
        layer_id=layer_id,
    )
    k_flat = k_pad.reshape(B * S_kv, *tail)
    v_flat = v_pad.reshape(B * S_kv, *v_live.shape[1:])
    k_flat[flat_valid] = k_valid
    v_flat[flat_valid] = v_valid
    return k_pad, v_pad


def ragged_paged_attn_batched(
    layer: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ctx: Any,
    meta: Any,
    *,
    write_kv: bool = True,
) -> torch.Tensor:
    """Paged-KV attention via padded batch matmul (deterministic, no FlashInfer).

    Faster than :func:`_ragged_paged_attn_eager`'s per-sample Python loop:
    gather KV into ``(B, S_kv, ...)``, then one batched QK^T / softmax / AV.
    Uses the same pure-torch math path as ``eager_attn`` (fp32 softmax).

    Requires uniform per-sample Q lengths (pi0.5 bucket / chunk layouts).
    After ``write_kv``, both prefix and current-step suffix are read from the
    KV pool (inference-only; not for the Fisher tape path).
    """
    _require_batched_attn_meta(meta, what="Eager attention")

    if write_kv:
        ctx.kv_pool.write_kv(layer.layer_id, ctx.write_indices, k, v)

    k_cache, v_cache = ctx.kv_pool.kv_buffer(layer.layer_id)
    k_pool = k_cache.squeeze(1)
    v_pool = v_cache.squeeze(1)

    idx_pad, key_valid, _q_lens, B, S_q, S_kv = _padded_kv_layout(meta)
    if S_q == 0 or S_kv == 0:
        return q.new_zeros(q.shape)

    flat = idx_pad.reshape(-1)
    k_pad = k_pool.index_select(0, flat).view(B, S_kv, k_pool.shape[1], k_pool.shape[2])
    v_pad = v_pool.index_select(0, flat).view(B, S_kv, v_pool.shape[1], v_pool.shape[2])
    return _padded_batched_attn(
        layer, q, k_pad, v_pad, key_valid, B=B, S_q=S_q, S_kv=S_kv
    )


def ragged_paged_attn_batched_grad(
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
    """Batched differentiable paged-KV attention for the Fisher path.

    Same padded-batch math as :func:`ragged_paged_attn_batched`, but gathers
    prefix rows from the autograd prefix-KV tape and suffix rows from the live
    ``k``/``v`` tensors. Requires uniform per-sample Q lengths; raises if not.
    """
    _require_batched_attn_meta(meta, what="Differentiable batched attention")

    if write_kv:
        ctx.kv_pool.write_kv(layer.layer_id, ctx.write_indices, k, v)
        if record_prefix_kv:
            _record_prefix_kv(ctx.kv_pool, layer.layer_id, ctx.write_indices, k, v)

    prefix_tape: PrefixKVTape | None = getattr(ctx.kv_pool, "_qvla_prefix_kv_tape", None)
    k_cache, v_cache = ctx.kv_pool.kv_buffer(layer.layer_id)
    k_pool = k_cache.squeeze(1)
    v_pool = v_cache.squeeze(1)

    idx_pad, key_valid, _q_lens, B, S_q, S_kv = _padded_kv_layout(meta)
    if S_q == 0 or S_kv == 0:
        return q.new_zeros(q.shape)

    k_pad, v_pad = _gather_joint_kv_padded(
        idx_pad,
        key_valid,
        k,
        v,
        k_pool,
        v_pool,
        suffix_slot_base=suffix_slot_base,
        prefix_tape=prefix_tape,
        layer_id=int(layer.layer_id),
    )
    return _padded_batched_attn(
        layer, q, k_pad, v_pad, key_valid, B=B, S_q=S_q, S_kv=S_kv
    )


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
        return ragged_paged_attn_batched_grad(
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
        logger.info(
            "Patched attention for autograd on: %s (batched eager + prefix KV tape).",
            ", ".join(parts),
        )
    return patched


def restore_attention_for_grad(patched: list[tuple[Any, _OriginalForward]]) -> None:
    for backend, original in patched:
        backend.forward = original  # type: ignore[method-assign]


def _make_inference_forward(
    meta_attr: str,
    runner: Any,
    scheduler: Any,
    *,
    batched: bool,
) -> _OriginalForward:
    suffix_slot_base = int(scheduler.suffix_base)

    def infer_forward(layer, q, k, v, ctx):
        meta = getattr(runner, meta_attr, None)
        if batched:
            return ragged_paged_attn_batched(
                layer, q, k, v, ctx, meta, write_kv=True
            )
        return _ragged_paged_attn_eager(
            layer,
            q,
            k,
            v,
            ctx,
            meta,
            suffix_slot_base=suffix_slot_base,
            write_kv=True,
            record_prefix_kv=False,
        )

    return infer_forward


def _make_inference_plan(
    meta_attr: str,
    runner: Any,
    *,
    is_expert: bool,
) -> _OriginalPlanInference:
    def plan_inference(meta: Any) -> None:
        # Expert RoPE reads ``pos_ids_suffix_buf``; original plan_inference
        # copies meta.position_ids into it before FlashInfer planning.
        if is_expert:
            if meta.position_ids is None:
                raise ValueError(
                    "expert plan_inference requires meta.position_ids "
                    "(feeds pos_ids_suffix_buf for RoPE)."
                )
            runner.pos_ids_suffix_buf.copy_(meta.position_ids.to(torch.int32))
        setattr(runner, meta_attr, meta)
        runner._capture_plan = None

    return plan_inference


def patch_attention_for_inference(
    scheduler: Any,
    *,
    llm: bool = True,
    expert: bool = True,
    batched: bool = True,
) -> list[_InferencePatch]:
    """Always-on torch attention (no FlashInfer).

    Replaces ``attn_backend.forward`` and ``runner.plan_inference`` so the
    normal ``scheduler.step`` path uses :func:`ragged_paged_attn_batched`
    (``batched=True``, default) or per-sample :func:`_ragged_paged_attn_eager`.
    Expert ``plan_inference`` still refreshes ``pos_ids_suffix_buf`` for RoPE.
    """
    patched: list[_InferencePatch] = []

    if llm:
        runner = scheduler.llm_runner
        backend = runner.attn_backend
        orig_fwd = backend.forward
        orig_plan = runner.plan_inference
        backend.forward = _make_inference_forward(  # type: ignore[method-assign]
            "_qvla_infer_prefix_meta", runner, scheduler, batched=batched
        )
        runner.plan_inference = _make_inference_plan(  # type: ignore[method-assign]
            "_qvla_infer_prefix_meta", runner, is_expert=False
        )
        patched.append((backend, orig_fwd, runner, orig_plan))

    if expert:
        runner = scheduler.expert_runner
        backend = runner.attn_backend
        orig_fwd = backend.forward
        orig_plan = runner.plan_inference
        backend.forward = _make_inference_forward(  # type: ignore[method-assign]
            "_qvla_infer_joint_meta", runner, scheduler, batched=batched
        )
        runner.plan_inference = _make_inference_plan(  # type: ignore[method-assign]
            "_qvla_infer_joint_meta", runner, is_expert=True
        )
        patched.append((backend, orig_fwd, runner, orig_plan))

    if patched:
        parts: list[str] = []
        if llm:
            parts.append("llm")
        if expert:
            parts.append("expert")
        kind = "batched" if batched else "per-sample"
        logger.info(
            "Patched attention for inference on: %s "
            "(%s torch math, no FlashInfer).",
            ", ".join(parts),
            kind,
        )
    return patched


def restore_attention_for_inference(patched: list[_InferencePatch]) -> None:
    for backend, orig_fwd, runner, orig_plan in patched:
        backend.forward = orig_fwd  # type: ignore[method-assign]
        runner.plan_inference = orig_plan  # type: ignore[method-assign]


__all__ = [
    "attach_attention_metadata",
    "clear_prefix_kv_tape",
    "patch_attention_for_grad",
    "patch_attention_for_inference",
    "ragged_paged_attn_batched",
    "ragged_paged_attn_batched_grad",
    "restore_attention_for_grad",
    "restore_attention_for_inference",
]
