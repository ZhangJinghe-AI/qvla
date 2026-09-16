"""Tests for matched-L1 DiT clip action-impact analysis."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import analyze_dit_matched_clip_action_impact as analysis  # noqa: E402


def test_matched_l1_threshold_hits_target_and_clips_the_tail():
    values = torch.tensor([1.0, 1.0, 1.0, 1.0, 10.0])
    tau, removed, token_frac = analysis._matched_l1_threshold(values, 0.25)
    total = float(values.sum().item())
    assert math.isclose(removed, 0.25, abs_tol=1e-10)
    assert math.isclose(token_frac, 0.2, abs_tol=1e-10)
    assert tau < 10.0
    clipped = values.clamp(max=tau)
    got = float((values - clipped).sum().item()) / total
    assert math.isclose(got, 0.25, abs_tol=1e-10)


def test_matched_l1_threshold_is_scale_equivariant():
    base = torch.tensor([0.2, 0.5, 0.7, 1.0, 8.0, 12.0])
    scaled = base * 40.0
    tau_a, rem_a, tok_a = analysis._matched_l1_threshold(base, 0.1)
    tau_b, rem_b, tok_b = analysis._matched_l1_threshold(scaled, 0.1)
    assert math.isclose(tau_b / tau_a, 40.0, rel_tol=1e-8)
    assert math.isclose(rem_a, rem_b, abs_tol=1e-10)
    assert math.isclose(tok_a, tok_b, abs_tol=1e-10)


def test_clip_preserves_sign_and_does_not_enlarge_values():
    x = torch.tensor([[-12.0, 0.5], [1.0, 9.0]])
    clipped, stats = analysis._clip_matched_l1(x, 0.2)
    assert stats.removed_l1_frac > 0.0
    torch.testing.assert_close(clipped.sign(), x.sign())
    assert bool((clipped.abs() <= x.abs() + 1e-12).all().item())
    assert bool((clipped.abs() <= stats.tau + 1e-12).all().item())


def test_constant_activation_shrinks_every_value():
    tau, removed, token_frac = analysis._matched_l1_threshold(torch.ones(8), 0.01)
    assert math.isclose(tau, 0.99, abs_tol=1e-10)
    assert math.isclose(removed, 0.01, abs_tol=1e-10)
    assert math.isclose(token_frac, 1.0, abs_tol=1e-10)


def test_nonpositive_l1_raises():
    with pytest.raises(RuntimeError, match="non-positive"):
        analysis._matched_l1_threshold(torch.zeros(8), 0.01)


def test_all_steps_only_require_step0_equality():
    assert analysis._must_match_baseline(0, None)
    assert not analysis._must_match_baseline(1, None)
    assert analysis._clip_this_step(0, None)
    assert analysis._clip_this_step(9, None)


def test_single_step_clips_only_the_target():
    assert analysis._must_match_baseline(0, 4)
    assert analysis._must_match_baseline(4, 4)
    assert not analysis._must_match_baseline(5, 4)
    assert not analysis._clip_this_step(3, 4)
    assert analysis._clip_this_step(4, 4)
    assert not analysis._clip_this_step(5, 4)


def test_bf16_fp32_roundtrip_is_identity():
    torch.manual_seed(0)
    x = torch.randn(32, 64, dtype=torch.bfloat16)
    assert torch.equal(analysis._exact_fp32(x).to(torch.bfloat16), x)


def test_fp16_snapshot_is_rejected():
    with pytest.raises(RuntimeError, match="fp16"):
        analysis._exact_fp32(torch.randn(4, 4, dtype=torch.float16))


def test_unclipped_bf16_values_survive_writeback():
    torch.manual_seed(0)
    x = torch.randn(32, 64, dtype=torch.bfloat16)
    live = analysis._exact_fp32(x)
    tau = float(live.abs().max().item()) + 1.0
    clipped = live.sign() * live.abs().clamp(max=tau)
    assert torch.equal(clipped.to(torch.bfloat16), x)


def test_layer_idx_accepts_pi05_and_groot_names():
    assert analysis._layer_idx("expert_stack.layers.8.mlp.down_proj") == 8
    assert (
        analysis._layer_idx("action_head.model.transformer_blocks.31.attn1.to_q") == 31
    )


def test_layer_kind_accepts_groot_linears():
    assert analysis._layer_kind("action_head.model.transformer_blocks.0.attn1.to_q") == "to_q"
    assert analysis._layer_kind("action_head.model.transformer_blocks.1.ff.fc2") == "fc2"
    assert analysis._layer_kind("expert_stack.layers.0.mlp.down_proj") == "down_proj"


def test_prefix_only_kv_uses_cross_attention_flag():
    class Block(torch.nn.Module):
        def __init__(self, cross: bool):
            super().__init__()
            self.is_cross_attention = cross
            self.attn1 = torch.nn.Module()
            self.attn1.to_k = torch.nn.Linear(4, 4)
            self.attn1.to_v = torch.nn.Linear(4, 4)
            self.attn1.to_q = torch.nn.Linear(4, 4)

    class Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.action_head = torch.nn.Module()
            self.action_head.model = torch.nn.Module()
            self.action_head.model.transformer_blocks = torch.nn.ModuleList(
                [Block(True), Block(False)]
            )

    model = Dummy()
    cross_k = "action_head.model.transformer_blocks.0.attn1.to_k"
    self_k = "action_head.model.transformer_blocks.1.attn1.to_k"
    query = "action_head.model.transformer_blocks.0.attn1.to_q"
    assert analysis._is_prefix_only_kv(model, cross_k)
    assert not analysis._is_prefix_only_kv(model, self_k)
    assert not analysis._is_prefix_only_kv(model, query)


def test_index_tertile_bands_match_pi05_and_groot():
    assert analysis._index_tertile_bands(17) == [
        ("early L0-5", 0, 5),
        ("mid L6-11", 6, 11),
        ("late L12-17", 12, 17),
    ]
    assert analysis._index_tertile_bands(31) == [
        ("early L0-9", 0, 9),
        ("mid L10-20", 10, 20),
        ("late L21-31", 21, 31),
    ]


def test_step_tertile_slices_match_ten_and_four_steps():
    assert analysis._step_tertile_slices(10) == (slice(0, 3), slice(3, 7), slice(7, 10))
    assert analysis._step_tertile_slices(4) == (slice(0, 1), slice(1, 3), slice(3, 4))
