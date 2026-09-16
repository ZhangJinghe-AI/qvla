"""Tests for all-denoise-step massive vs shoulder DCT scoring."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import analyze_dit_outlier_lp_hp_all_steps as analysis  # noqa: E402
import analyze_dit_outlier_lp_hp_impact as lp_hp  # noqa: E402


def test_arm_overall_matches_parseval_lp_hp():
    dct = lp_hp._dct_matrix(50)
    baseline = np.zeros((50, 7))
    changed = baseline.copy()
    changed[:, 0] = np.linspace(0.0, 1.0, 50)
    changed[24, 1] = 0.4
    overall = analysis._arm_overall(changed, baseline)
    metrics = lp_hp._freq_metrics(changed, baseline, dct, 5)
    np.testing.assert_allclose(
        overall**2, metrics.lp_rmse**2 + metrics.hp_rmse**2, atol=1e-12
    )


def _row(
    *,
    overall_m: float,
    overall_s: float,
    lp_m: float,
    hp_m: float,
    lp_s: float,
    hp_s: float,
    cutoff: int = 5,
) -> analysis.CondRow:
    return analysis.CondRow(
        sample=0,
        noise=0,
        cutoff=cutoff,
        massive_overall=overall_m,
        shoulder_overall=overall_s,
        massive=lp_hp.FreqMetrics(cutoff, lp_m, hp_m, 0.0),
        shoulder=lp_hp.FreqMetrics(cutoff, lp_s, hp_s, 0.0),
        massive_n_sites=720,
        massive_n_clipped=640,
        shoulder_n_expanded=0,
    )


def test_verdict_holds_when_shoulder_owns_lp_and_massive_owns_hp():
    text = analysis._verdict_for_cutoff(
        [_row(overall_m=3.2, overall_s=2.2, lp_m=1.0, hp_m=3.0, lp_s=2.0, hp_s=1.0)],
        5,
    )
    assert text.startswith("HOLDS")


def test_verdict_fails_when_ratios_flip():
    text = analysis._verdict_for_cutoff(
        [_row(overall_m=2.2, overall_s=3.2, lp_m=2.0, hp_m=1.0, lp_s=1.0, hp_s=3.0)],
        5,
    )
    assert text.startswith("FAILS")
