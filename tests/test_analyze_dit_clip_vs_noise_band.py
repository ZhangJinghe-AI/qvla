"""Tests for medoid-referenced clip vs noise-band analysis."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import analyze_dit_clip_vs_noise_band as analysis  # noqa: E402
import analyze_dit_outlier_action_detail as detail  # noqa: E402
import analyze_dit_outlier_action_detail_all as all_exp  # noqa: E402


def _chunk(*values: float, horizon: int = 4, dims: int = 7) -> torch.Tensor:
    action = torch.zeros(horizon, dims)
    action[:, 0] = torch.tensor(values, dtype=torch.float32)
    return action


def _metrics(total: float) -> detail.ActionMetrics:
    return detail.ActionMetrics(
        total_rmse=total,
        arm_mean_shift_rmse=total,
        arm_endpoint_rmse=total * 0.8,
        arm_net_disp_rmse=0.0,
        arm_local_rmse=total * 0.2,
        arm_step_rmse=0.0,
        gripper_rmse=0.0,
        gripper_switch_shift=0.0,
    )


def _clip(
    sample: int,
    noise: int,
    kind: str,
    step: int | None,
    total: float,
) -> analysis.ClipRow:
    return analysis.ClipRow(
        sample=sample,
        medoid_noise=noise,
        kind=kind,
        step=step,
        n_sites=72,
        n_clipped=40,
        n_empty=32,
        n_insufficient=3,
        outlier_count=80,
        selected_channels=12,
        mean_removed_l1_pct=0.4,
        metrics=_metrics(total),
    )


def test_medoid_is_the_cluster_center_not_the_outlier():
    chunks = torch.zeros(6, 8, 7)
    chunks[:5] += torch.randn(5, 8, 7) * 0.01
    chunks[5] += 40.0
    index = analysis._medoid_index(chunks)
    assert index in range(5)


def test_medoid_of_a_line_is_the_middle_sample_not_the_origin():
    chunks = torch.stack(
        [_chunk(0.0, 0.0, 0.0, 0.0), _chunk(0.1, 0.1, 0.1, 0.1), _chunk(10.0, 10.0, 10.0, 10.0)]
    )
    assert analysis._medoid_index(chunks) == 1


def test_identical_actions_break_ties_at_index_zero():
    chunks = torch.ones(4, 5, 7)
    assert analysis._medoid_index(chunks) == 0


def test_medoid_stays_in_the_dense_cluster_when_one_action_is_far():
    chunks = torch.stack(
        [
            _chunk(0.0, 0.0, 0.0, 0.0),
            _chunk(0.0, 0.0, 0.0, 0.0),
            _chunk(0.0, 0.0, 0.0, 0.0),
            _chunk(1.0, 1.0, 1.0, 1.0),
            _chunk(30.0, 30.0, 30.0, 30.0),
        ]
    )
    choice = analysis._choose_medoid(3, [10, 11, 12, 13, 14], chunks)
    assert choice.noise in {10, 11, 12}
    assert choice.mean_rmse_to_others < 20.0


def test_quantile_band_matches_numpy():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    band = analysis._quantile_band(values)
    assert band.n == 10
    assert band.minimum == 1.0
    assert band.maximum == 10.0
    assert band.median == 5.5
    np.testing.assert_allclose(band.p10, np.quantile(values, 0.10))
    np.testing.assert_allclose(band.p90, np.quantile(values, 0.90))


def test_metric_values_skip_the_medoid():
    rows = [
        analysis.NoiseRow(0, 1, True, _metrics(0.0)),
        analysis.NoiseRow(0, 2, False, _metrics(0.4)),
        analysis.NoiseRow(0, 3, False, _metrics(0.6)),
    ]
    assert analysis._metric_values(rows, "total_rmse", skip_medoid=True) == [0.4, 0.6]


def test_placement_thresholds():
    band = analysis.QuantileBand(0.1, 0.2, 0.3, 0.4, 0.5, n=7)
    assert analysis._placement(0.15, band) == "below_p10"
    assert analysis._placement(0.3, band) == "inside_p10_p90"
    assert analysis._placement(0.45, band) == "between_p90_and_max"
    assert analysis._placement(0.9, band) == "above_max"


def test_pairwise_rejects_a_single_action():
    with pytest.raises(ValueError, match=">=2"):
        analysis._pairwise_total_rmse(torch.zeros(1, 4, 7))


def test_clip_kwargs_full_unmatched_channel_mad():
    massive = analysis._matched_clip_kwargs("massive", 3)
    assert massive["kind"] == "selective"
    assert massive.get("normal_ref_l1") is not True
    assert massive.get("normal_full_clip") is not True
    assert massive["allow_empty"] is True
    assert massive["target_step"] == 3
    assert "bulk_seed" not in massive
    normal = analysis._matched_clip_kwargs("normal", 3)
    assert normal["kind"] == "random"
    assert normal["normal_ref_l1"] is True
    assert normal["normal_ref_mode"] == "channel_mad"
    assert normal["normal_full_clip"] is True
    assert normal["bulk_seed"] == 1000 + 3 * 17
    all_steps = analysis._matched_clip_kwargs("normal", None)
    assert all_steps["target_step"] is None
    assert all_steps["bulk_seed"] == 1000
    assert all_steps["normal_full_clip"] is True


def test_run_all_accepts_allow_empty():
    sig = inspect.signature(all_exp._run_all).parameters
    assert "allow_empty" in sig
    assert "normal_full_clip" in sig


def test_summary_and_plot(tmp_path: Path):
    medoids = [
        analysis.MedoidChoice(
            sample=0,
            noise=2,
            index=2,
            mean_rmse_to_others=0.05,
        ),
        analysis.MedoidChoice(
            sample=1,
            noise=0,
            index=0,
            mean_rmse_to_others=0.06,
        ),
    ]
    noise_rows = [
        analysis.NoiseRow(0, 2, True, _metrics(0.0)),
        analysis.NoiseRow(0, 0, False, _metrics(0.05)),
        analysis.NoiseRow(0, 1, False, _metrics(0.07)),
        analysis.NoiseRow(1, 0, True, _metrics(0.0)),
        analysis.NoiseRow(1, 1, False, _metrics(0.04)),
        analysis.NoiseRow(1, 2, False, _metrics(0.08)),
    ]
    clip_rows: list[analysis.ClipRow] = []
    for sample, noise in ((0, 2), (1, 0)):
        for kind, scale in (("massive", 1.0), ("normal", 1.4)):
            for step in (0, 1, None):
                total = (0.03 if step == 0 else 0.09 if step == 1 else 0.12) * scale
                clip_rows.append(_clip(sample, noise, kind, step, total))
    text = analysis._summary(medoids, noise_rows, clip_rows)
    assert "per-sample medoid" in text
    assert "channel_mad full clip" in text
    assert "No excess-L1 matching" in text
    assert "massive:" in text
    assert "normal:" in text
    assert "closest-to-mean" not in text
    assert "Pairwise total RMSE" not in text
    assert "step=  0" in text
    assert "step=all" in text
    png = tmp_path / "clip_vs_noise_band.png"
    analysis._plot(clip_rows, noise_rows, png, num_steps=2)
    assert png.is_file() and png.stat().st_size > 0


def test_replot_from_saved_csv(tmp_path: Path):
    medoids = [
        analysis.MedoidChoice(sample=0, noise=2, index=2, mean_rmse_to_others=0.05),
    ]
    noise_rows = [
        analysis.NoiseRow(0, 2, True, _metrics(0.0)),
        analysis.NoiseRow(0, 0, False, _metrics(0.05)),
        analysis.NoiseRow(0, 1, False, _metrics(0.07)),
    ]
    clip_rows = [
        _clip(0, 2, kind, step, 0.03 if step == 0 else 0.12)
        for kind in ("massive", "normal")
        for step in (0, None)
    ]
    analysis._write_csv(
        medoids,
        tmp_path / "medoids.csv",
        fieldnames=["sample", "medoid_noise", "medoid_index", "mean_rmse_to_others"],
    )
    analysis._write_csv(
        noise_rows,
        tmp_path / "noise_vs_medoid.csv",
        fieldnames=["sample", "noise", "is_medoid", *analysis.METRIC_NAMES],
    )
    analysis._write_csv(
        clip_rows,
        tmp_path / "clip_by_step.csv",
        fieldnames=analysis.CLIP_CSV_FIELDS,
    )
    assert analysis._replot_from_dir(tmp_path) == 0
    assert (tmp_path / "clip_vs_noise_band.png").is_file()
    text = (tmp_path / "summary.txt").read_text(encoding="utf-8")
    assert "per-sample medoid" in text
    assert "channel_mad full clip" in text
    assert "No excess-L1 matching" in text
    assert "closest-to-mean" not in text
    loaded = analysis._load_clip_rows(tmp_path / "clip_by_step.csv")
    assert {row.kind for row in loaded} == {"massive", "normal"}
    assert all(row.n_insufficient == 3 for row in loaded)
