"""Tests for existing-rule L1 energy measurement."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import analyze_dit_outlier_rule_l1 as analysis  # noqa: E402


def _outlier_tokens() -> torch.Tensor:
    values = torch.ones(50, 8)
    for channel in range(8):
        values[:, channel] = 1.0 + 0.15 * float(channel)
    values[0, 0] = 40.0
    values[:, 1] = torch.linspace(1.0, 4.0, 50)
    return values


def test_per_channel_clips_only_the_heavy_channel():
    abs_x = _outlier_tokens()
    measured = analysis._measure_one(abs_x, std_k=3.0)
    per = measured["per_channel"]
    assert per["status"] == "ok"
    assert per["removed_l1_frac"] > 0.0
    assert per["n_clipped"] >= 1
    assert per["n_channels_capped"] >= 1


def test_selective_l1_is_at_most_per_channel():
    abs_x = _outlier_tokens()
    measured = analysis._measure_one(abs_x, std_k=3.0)
    sel = measured["selective"]
    per = measured["per_channel"]
    assert sel["status"] == "ok"
    assert sel["n_selected_channels"] < int(abs_x.shape[1])
    assert sel["removed_l1_frac"] <= per["removed_l1_frac"] + 1e-12


def test_l1_from_bounds_matches_explicit_sum():
    abs_x = torch.tensor([[1.0, 10.0], [1.0, 2.0], [1.0, 2.0]])
    bounds = torch.tensor([1.0, 4.0])
    frac, token_frac, n_capped, n_clipped = analysis._l1_from_bounds(abs_x, bounds)
    removed = 6.0
    total = 17.0
    assert math.isclose(frac, removed / total, abs_tol=1e-12)
    assert n_clipped == 1
    assert n_capped == 1
    assert math.isclose(token_frac, 1.0 / 6.0, abs_tol=1e-12)


def test_equal_amax_selective_records_mad_failure():
    abs_x = torch.ones(16, 4)
    measured = analysis._measure_one(abs_x, std_k=3.0)
    assert measured["selective"]["status"] == "mad_failed"
    assert math.isnan(measured["selective"]["removed_l1_frac"])
    assert measured["per_channel"]["status"] == "ok"
    assert measured["per_channel"]["removed_l1_frac"] == 0.0


def test_std_k_must_be_positive():
    with pytest.raises(ValueError, match="std_k"):
        analysis._measure_one(torch.ones(8, 4), std_k=0.0)
