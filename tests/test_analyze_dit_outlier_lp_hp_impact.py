"""Tests for massive vs shoulder DCT coarse/detail scoring."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import analyze_dit_outlier_lp_hp_impact as analysis  # noqa: E402


def test_dct_split_recovers_action():
    dct = analysis._dct_matrix(50)
    action = np.linspace(0.0, 1.0, 50)[:, None] + np.sin(np.linspace(0, 12, 50))[:, None]
    low, high = analysis._split_dct(action, dct, 5)
    np.testing.assert_allclose(low + high, action, atol=1e-12)


def test_slow_curve_is_mostly_lp():
    dct = analysis._dct_matrix(50)
    time = np.linspace(0.0, 1.0, 50)[:, None]
    baseline = np.concatenate([time * 0.0, np.zeros((50, 1))], axis=1)
    changed = np.concatenate([time, np.zeros((50, 1))], axis=1)
    metrics = analysis._freq_metrics(changed, baseline, dct, 5)
    assert metrics.lp_rmse > 3.0 * metrics.hp_rmse


def test_single_spike_is_mostly_hp():
    dct = analysis._dct_matrix(50)
    baseline = np.zeros((50, 7))
    changed = baseline.copy()
    changed[24, :6] = 1.0
    metrics = analysis._freq_metrics(changed, baseline, dct, 5)
    assert metrics.hp_rmse > 3.0 * metrics.lp_rmse


def _row(step: int, lp_m: float, hp_m: float, lp_s: float, hp_s: float) -> analysis.StepRow:
    return analysis.StepRow(
        sample=0,
        noise=0,
        step=step,
        cutoff=5,
        massive=analysis.FreqMetrics(5, lp_m, hp_m, 0.0),
        shoulder=analysis.FreqMetrics(5, lp_s, hp_s, 0.0),
    )


def test_verdict_holds_when_shoulder_owns_lp_and_massive_owns_hp():
    rows = [_row(0, 1.0, 3.0, 2.0, 1.0), _row(1, 1.0, 3.0, 2.0, 1.0)]
    text = analysis._verdict_for_cutoff(rows, 5)
    assert text.startswith("HOLDS")


def test_verdict_fails_when_ratios_flip():
    rows = [_row(0, 2.0, 1.0, 1.0, 3.0)]
    text = analysis._verdict_for_cutoff(rows, 5)
    assert text.startswith("FAILS")
