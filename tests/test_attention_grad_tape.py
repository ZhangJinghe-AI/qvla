"""Tests for prefix KV tape used in Fisher LLM backprop."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_PKG_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from qvla.build.attention_grad import (  # noqa: E402
    _gather_joint_kv_segment,
    _record_prefix_kv,
    _rows_for_pool_slots,
    clear_prefix_kv_tape,
)


class _FakePool:
    pass


def test_rows_for_pool_slots_round_trip():
    write_indices = torch.tensor([10, 11, 12], dtype=torch.int64)
    pool_slots = torch.tensor([11, 10], dtype=torch.int64)
    rows = _rows_for_pool_slots(write_indices, pool_slots)
    assert rows.tolist() == [1, 0]


def test_prefix_kv_tape_missing_raises():
    k_suffix = torch.tensor([[9.0]], requires_grad=True)
    v_suffix = k_suffix.clone()
    k_pool = torch.zeros(200, 1)
    v_pool = torch.zeros(200, 1)
    idx = torch.tensor([101], dtype=torch.int64)

    try:
        _gather_joint_kv_segment(
            idx,
            k_suffix,
            v_suffix,
            k_pool,
            v_pool,
            suffix_slot_base=200,
            prefix_tape={},
            layer_id=0,
        )
    except RuntimeError as exc:
        assert "Prefix KV tape missing" in str(exc)
    else:
        raise AssertionError("expected RuntimeError when prefix tape entry is missing")


def test_prefix_kv_tape_connects_expert_to_llm_graph():
    k_pre = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
    v_pre = k_pre.clone()
    write_indices = torch.tensor([100, 101, 102], dtype=torch.int64)

    pool = _FakePool()
    _record_prefix_kv(pool, layer_id=0, write_indices=write_indices, k=k_pre, v=v_pre)
    tape = pool._qvla_prefix_kv_tape

    k_suffix = torch.tensor([[9.0]], requires_grad=True)
    v_suffix = k_suffix.clone()
    k_pool = torch.zeros(200, 1)
    v_pool = torch.zeros(200, 1)
    idx = torch.tensor([101, 102, 200], dtype=torch.int64)

    k_seg, v_seg = _gather_joint_kv_segment(
        idx,
        k_suffix,
        v_suffix,
        k_pool,
        v_pool,
        suffix_slot_base=200,
        prefix_tape=tape,
        layer_id=0,
    )
    out = (k_seg * v_seg).sum()
    out.backward()

    assert k_pre.grad is not None
    assert k_pre.grad.abs().sum().item() > 0
    assert k_suffix.grad is not None
    clear_prefix_kv_tape(pool)
    assert pool._qvla_prefix_kv_tape == {}
