"""Tests for DiT massive vs shoulder outlier classification."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import analyze_dit_outlier_types as analysis  # noqa: E402


def test_token_groups_groot_split_state_valid_pad():
    names = analysis._token_group_names(41, skip_first=True, valid_action_horizon=16)
    assert names[0] == "state"
    assert names[1:17] == ["action_valid"] * 16
    assert names[17:] == ["action_pad"] * 24


def test_token_groups_pi05_are_all_action():
    names = analysis._token_group_names(50, skip_first=False, valid_action_horizon=None)
    assert names == ["action"] * 50


def test_skip_first_massive_is_action_tip_not_state():
    x = torch.ones(17, 16) * 0.2
    x[0, -1] = 400.0
    x[1, -1] = 80.0
    x[2, -1] = 9.0
    x[3, -1] = 8.0
    x[4, -1] = 7.0
    values, selected, massive, shoulder, state_would = analysis._outlier_masks(
        x, 3.0, skip_first_token=True
    )
    assert not bool(massive[0].any().item())
    assert not bool(shoulder[0].any().item())
    assert bool(massive[1, -1].item())
    assert bool(state_would[-1].item())
    assert bool(selected[-1].item())
    assert float(values[massive].min().item()) > float(values[shoulder].max().item())


def test_token_acc_sums_per_index(tmp_path: Path):
    acc = analysis.TokenIndexAcc.zeros(4, 2)
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    massive = torch.zeros_like(values, dtype=torch.bool)
    massive[1, 1] = True
    shoulder = torch.zeros_like(values, dtype=torch.bool)
    shoulder[2, 0] = True
    acc.add(0, values, massive, shoulder)
    np.testing.assert_allclose(acc.massive_l1[0], [0.0, 4.0, 0.0, 0.0])
    np.testing.assert_allclose(acc.shoulder_l1[0], [0.0, 0.0, 5.0, 0.0])
    np.testing.assert_allclose(acc.massive_n[0], [0.0, 1.0, 0.0, 0.0])
    names = analysis._token_group_names(4, skip_first=True, valid_action_horizon=2)
    grouped = analysis._group_l1(acc, names)
    assert grouped["state"][0] == 0.0
    assert grouped["action_valid"][0] == 4.0
    assert grouped["action_valid"][1] == 5.0
    assert grouped["action_pad"][0] == 0.0
    row, _v, m, s, state_would = analysis._site_row(
        x := torch.ones(17, 16) * 0.2,
        sample=0,
        noise=0,
        step=0,
        layer="fake",
        kind="fc1",
        std_k=3.0,
        skip_first_token=True,
    )
    del x, m, s, state_would
    assert row.n_massive == 0
    assert row.state_would_clip_n == 0
    out = tmp_path / "summary.txt"
    analysis._write_summary(
        [row],
        out,
        skip_first=True,
        valid_action_horizon=16,
        token_acc=acc,
    )
    text = out.read_text()
    assert "action_valid" in text
    assert "first action" in text
    token_rows = analysis._token_rows(acc, names)
    assert token_rows[1]["massive_n"] == 1.0
    csv_path = tmp_path / "tokens.csv"
    analysis._write_token_csv(token_rows, csv_path)
    assert csv_path.is_file()


def test_raw_over_keeps_state_spike_that_clip_skips():
    acc = analysis.TokenIndexAcc.zeros(4, 1)
    values = torch.tensor([[9.0, 1.0], [1.0, 8.0], [1.0, 1.0], [1.0, 1.0]])
    massive = torch.tensor(
        [[False, False], [False, True], [False, False], [False, False]]
    )
    shoulder = torch.zeros_like(massive)
    state_would = torch.tensor([True, False])
    acc.add(0, values, massive, shoulder, state_would=state_would)
    np.testing.assert_allclose(acc.massive_n[0], [0.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(acc.raw_over_n[0], [1.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(acc.raw_over_l1[0], [9.0, 8.0, 0.0, 0.0])
