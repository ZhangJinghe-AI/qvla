"""Tests for the channel-amax tail selection visualization."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import visualize_channel_amax_tail_selection as viz  # noqa: E402


def test_robust_selector_exposes_long_tail_hidden_by_mean_std():
    hard_amax = torch.tensor(
        [[1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 100.0, 10000.0]]
    )
    stats = viz._selection_stats(hard_amax)

    assert not bool(stats["classic_selected"].any().item())
    assert stats["robust_selected"].tolist() == [
        False, False, False, False, False, False, False, False, True, True
    ]


def test_selection_stats_rejects_zero_mad_without_fallback():
    with pytest.raises(RuntimeError, match="positive cross-channel MAD"):
        viz._selection_stats(torch.ones(4, 8))


def test_plot_writes_comparison(tmp_path):
    stats = viz._selection_stats(
        torch.tensor(
            [[1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 100.0, 10000.0]]
        )
    )
    output = tmp_path / "comparison.png"
    viz._plot(stats, title="test", output=output)
    assert output.is_file()
    assert output.stat().st_size > 0


def _tip_with_two_outlier_channels() -> torch.Tensor:
    """T=32 activations whose per-channel amax matches the selector fixture."""
    scale = torch.tensor(
        [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 100.0, 10000.0]
    )
    values = scale.unsqueeze(0).expand(32, -1).clone()
    values[1:] = scale.unsqueeze(0) * 0.2
    return values


def test_stage1_threshold_rank_interpolates_between_neighbors():
    x = viz._stage1_threshold_rank(np.array([10.0, 6.0, 2.0]), 4.0)
    assert x == 1.5


def test_stage1_threshold_rank_none_and_all_selected():
    ranked = np.array([5.0, 4.0, 3.0])
    assert viz._stage1_threshold_rank(ranked, 10.0) == 0.0
    assert viz._stage1_threshold_rank(ranked, 2.0) == 3.0


def test_stage1_threshold_rank_rejects_unsorted():
    with pytest.raises(RuntimeError, match="sorted descending"):
        viz._stage1_threshold_rank(np.array([1.0, 3.0, 2.0]), 2.0)


def test_clip_fraction_is_clipped_token_count_over_channel_tokens():
    values = _tip_with_two_outlier_channels()
    stats = viz._clip_fraction_stats(values)
    assert not bool(stats["classic_selected"].any().item())
    assert stats["robust_selected"].tolist() == [
        False, False, False, False, False, False, False, False, True, True
    ]
    expected_all = torch.full((10,), 1.0 / float(values.shape[0]))
    torch.testing.assert_close(stats["clip_frac_all"], expected_all)
    expected_sel = expected_all.clone()
    expected_sel[:8] = 0.0
    torch.testing.assert_close(stats["clip_frac_selective"], expected_sel)


def test_token_clip_fraction_counts_tokens_above_bound():
    values = torch.tensor([[10.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    bound = torch.tensor([2.0, 1.0])
    frac = viz._token_clip_fraction(values, bound, name="test")
    torch.testing.assert_close(frac, torch.tensor([0.25, 0.0]))


def test_token_clip_fraction_rejects_bound_above_amax():
    values = torch.ones(4, 2)
    bound = torch.tensor([1.0, 2.0])
    with pytest.raises(RuntimeError, match="exceeded hard channel amax"):
        viz._token_clip_fraction(values, bound, name="test")


def test_clip_fraction_rejects_too_few_tokens():
    with pytest.raises(RuntimeError, match=">= 2 tip tokens"):
        viz._clip_fraction_stats(
            torch.tensor([[1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 100.0, 10000.0]])
        )


def test_clip_fraction_rejects_nonpositive_amax():
    values = _tip_with_two_outlier_channels()
    values[:, 0] = 0.0
    with pytest.raises(RuntimeError, match="strictly positive channel amax"):
        viz._clip_fraction_stats(values)


def test_plot_clip_percent_writes(tmp_path):
    stats = viz._clip_fraction_stats(_tip_with_two_outlier_channels())
    output = tmp_path / "clip_percent.png"
    viz._plot_clip_percent(stats, title="test", output=output)
    assert output.is_file()
    assert output.stat().st_size > 0
