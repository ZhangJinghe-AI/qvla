"""Tests for all-linear, one-denoise-step clip analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import analyze_dit_selective_clip_by_denoise_step as analysis  # noqa: E402


def test_capture_rejects_clip_flags():
    with pytest.raises(ValueError, match="Capture"):
        analysis._run_all_layers(
            adapter=None,
            request=None,
            layers=[],
            num_steps=10,
            horizon=50,
            alpha=0.005,
            target_step=None,
        )


def test_intervention_requires_alpha_and_step():
    with pytest.raises(ValueError, match="alpha"):
        analysis._run_all_layers(
            adapter=None,
            request=None,
            layers=[],
            num_steps=10,
            horizon=50,
            alpha=None,
            target_step=3,
            baseline_activations={"x": None},
        )


def test_identity_requires_target_step():
    with pytest.raises(ValueError, match="Identity"):
        analysis._run_all_layers(
            adapter=None,
            request=None,
            layers=[],
            num_steps=10,
            horizon=50,
            alpha=None,
            target_step=None,
            baseline_activations={"x": None},
            identity_writeback=True,
        )
