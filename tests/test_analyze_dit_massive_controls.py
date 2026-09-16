"""Tests for massive-outlier control diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import analyze_dit_massive_controls as analysis  # noqa: E402


def test_token_massive_counts_mark_the_spike_row():
    x = torch.ones(32, 4)
    x[2, 0] = 40.0
    counts = analysis._token_massive_counts(x, std_k=3.0, skip_first_token=False)
    assert tuple(counts.shape) == (32,)
    assert int(counts[2].item()) >= 1
    assert int(counts.sum().item()) >= 1
    assert int(counts[0].item()) == 0


def test_occupancy_sums_layers_of_the_same_kind():
    spike = torch.ones(10, 32, 4)
    spike[0, 3, 0] = 50.0
    activations = {
        "expert_stack.layers.0.mlp.down_proj": spike,
        "expert_stack.layers.1.mlp.down_proj": spike.clone(),
        "expert_stack.layers.0.qkv_proj": torch.ones(10, 32, 4),
    }
    rows = analysis._occupancy_from_activations(
        activations, sample=2, std_k=3.0, skip_first_token=False
    )
    down = [row for row in rows if row.kind == "down_proj" and row.step == 0 and row.token == 3]
    qkv = [row for row in rows if row.kind == "qkv_proj" and row.step == 0]
    assert len(down) == 1
    assert down[0].sample == 2
    assert down[0].n_massive >= 2
    assert all(row.n_massive == 0 for row in qkv)


def test_clip_kwargs_last_step_is_the_final_index():
    last = analysis._clip_kwargs("last", 10)
    assert last["kind"] == "selective"
    assert last["target_step"] == 9
    assert last["allow_empty"] is True
    assert "normal_ref_l1" not in last
    all_steps = analysis._clip_kwargs("all", 10)
    assert all_steps["target_step"] is None
