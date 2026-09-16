"""Adaptive-clip collector VRAM estimate is model/scope aware."""

from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn

_PKG_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from qvla.build.utils import estimate_adaptive_clip_bytes


def _targets(*items: tuple[str, str]) -> list[tuple[str, str, nn.Module]]:
    return [(name, scope, nn.Linear(8, 8)) for name, scope in items]


def test_estimate_without_model_kind_keeps_legacy_512():
    targets = _targets(("layer", "dit"))
    assert estimate_adaptive_clip_bytes(targets, 2) == 2 * 512 * 8 * 2


def test_estimate_groot_dit_loop_uses_state_plus_chunk_times_steps():
    targets = _targets(("action_head.model.transformer_blocks.0.ff.fc1", "dit"))
    got = estimate_adaptive_clip_bytes(
        targets,
        2,
        model_kind="groot_n17",
        chunk_size=16,
        num_steps_by_scope={"dit": 4},
    )
    assert got == 2 * (16 + 1) * 4 * 8 * 2


def test_estimate_groot_dit_kv_uses_prefix_not_denoise_steps():
    targets = _targets(("action_head.model.transformer_blocks.0.attn1.to_k", "dit"))
    got = estimate_adaptive_clip_bytes(
        targets,
        2,
        model_kind="groot_n17",
        chunk_size=16,
        num_steps_by_scope={"dit": 4},
    )
    assert got == 2 * 512 * 8 * 2


def test_estimate_pi05_dit_uses_chunk_times_steps():
    targets = _targets(("expert_stack.layers.0.qkv_proj", "dit"))
    got = estimate_adaptive_clip_bytes(
        targets,
        3,
        model_kind="pi05",
        chunk_size=50,
        num_steps_by_scope={"dit": 10},
    )
    assert got == 3 * 50 * 10 * 8 * 2


def test_estimate_llm_prefix_differs_by_model():
    targets = _targets(("layers.0.qkv_proj", "llm"))
    pi05 = estimate_adaptive_clip_bytes(
        targets, 1, model_kind="pi05", chunk_size=50, num_steps_by_scope={"llm": 1}
    )
    groot = estimate_adaptive_clip_bytes(
        targets,
        1,
        model_kind="groot_n17",
        chunk_size=16,
        num_steps_by_scope={"llm": 1},
    )
    assert pi05 == 384 * 8 * 2
    assert groot == 512 * 8 * 2
    assert pi05 < groot
