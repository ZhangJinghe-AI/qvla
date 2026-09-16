"""Smoke tests for the SmoothQuant integration.

These tests are CPU-only and self-contained. They cover:

* :func:`fit_smooth_scale` produces reasonable ``s`` and errors loudly.
* :class:`~qvla.config.ScopeConfig` rejects illegal pipelines (e.g.
  ``hadamard`` before ``clip``; ``smooth`` before ``perm``).
  ``clip,smooth,hadamard`` is allowed.
* Pack save / load round-trips ``smooth_scale`` on :class:`Transform`.
* Forward through :class:`QuantLinear` with SmoothQuant matches the
  bit-identical baseline before any transform (weight_bits=16, act_bits=16).
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

_PKG_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from qvla.core.smooth_fit import fit_smooth_scale
from qvla.config import QVLAConfig, ScopeConfig
from qvla.core.clip import (
    channel_outlier_mean_std_amax,
    layer_outlier_mean_std_amax,
    selective_channel_outlier_mean_std_amax,
)
from qvla.core.pack import LayerPack, Pack
from qvla.core.quantize import no_quantize
from qvla.core.pipeline import PipelineBuild, Transform, identity_transform
from qvla.runtime import QuantLinear


# --------------------------------------------------------------------------- #
# Fit                                                                         #
# --------------------------------------------------------------------------- #


def test_fit_smooth_scale_shifts_outliers():
    """A channel with a huge activation outlier should get a large ``s``."""
    torch.manual_seed(2)
    K, N = 16, 8
    W = torch.randn(N, K) * 0.05
    amax = torch.ones(K)
    amax[5] = 100.0  # outlier channel
    s = fit_smooth_scale(
        layer_name="test",
        weight=W,
        act_channel_amax=amax,
        alpha=0.5,
    )
    assert s[5] > 5.0 * s.median()


def test_fit_smooth_scale_alpha_extremes():
    """alpha=0 → s = w^(-1); alpha=1 → s = a. Both should error out cleanly on
    non-finite results, which cannot happen with epsilon floor."""
    K = 8
    W = torch.randn(4, K)
    a = torch.rand(K) + 0.1

    s0 = fit_smooth_scale(
        layer_name="a0", weight=W, act_channel_amax=a, alpha=0.0
    )
    w = W.abs().amax(dim=0).clamp_min(1e-5)
    assert torch.allclose(s0, 1.0 / w, rtol=1e-4)

    s1 = fit_smooth_scale(
        layer_name="a1", weight=W, act_channel_amax=a, alpha=1.0
    )
    assert torch.allclose(s1, a.clamp_min(1e-5), rtol=1e-4)


def test_fit_smooth_scale_fisher_beta_requires_fisher():
    with pytest.raises(ValueError, match="fisher_beta"):
        fit_smooth_scale(
            layer_name="test",
            weight=torch.randn(4, 8),
            act_channel_amax=torch.ones(8),
            alpha=0.5,
            fisher=None,
            fisher_beta=0.5,
        )


def test_fit_smooth_scale_rejects_negative_fisher_beta():
    with pytest.raises(ValueError, match="fisher_beta"):
        fit_smooth_scale(
            layer_name="test",
            weight=torch.randn(4, 8),
            act_channel_amax=torch.ones(8),
            alpha=0.5,
            fisher=torch.ones(8),
            fisher_beta=-0.5,
        )


def test_fit_smooth_scale_all_zero_fisher_skips_reweighting():
    """Off-graph layers (all-zero Fisher) fall back to plain SmoothQuant."""
    K = 8
    W = torch.randn(4, K)
    a = torch.rand(K) + 0.1
    s = fit_smooth_scale(
        layer_name="paligemma_lm.layers.17.o_proj",
        weight=W,
        act_channel_amax=a,
        alpha=0.5,
        fisher=torch.zeros(K),
        fisher_beta=0.5,
    )
    s_plain = fit_smooth_scale(
        layer_name="plain",
        weight=W,
        act_channel_amax=a,
        alpha=0.5,
        fisher_beta=0.0,
    )
    assert torch.allclose(s, s_plain, rtol=1e-5)


def test_fit_smooth_scale_fisher_weighting():
    """Higher Fisher → larger s when a and w are uniform."""
    K = 8
    W = torch.ones(4, K)
    a = torch.ones(K)
    fisher = torch.ones(K)
    fisher[3] = 8.0
    s = fit_smooth_scale(
        layer_name="test",
        weight=W,
        act_channel_amax=a,
        alpha=0.5,
        fisher=fisher,
        fisher_beta=1.0,
    )
    assert int(s.argmax().item()) == 3
    assert s[3] > s.median()


def test_fit_smooth_scale_flat_fisher_is_noop():
    """Uniform Fisher → g≡1 → identical to plain SmoothQuant."""
    K = 16
    W = torch.randn(4, K)
    a = torch.rand(K) + 0.1
    s = fit_smooth_scale(
        layer_name="flat",
        weight=W,
        act_channel_amax=a,
        alpha=0.5,
        fisher=torch.ones(K) * 3.7,
        fisher_beta=1.0,
    )
    plain = fit_smooth_scale(
        layer_name="plain", weight=W, act_channel_amax=a, alpha=0.5
    )
    assert torch.allclose(s, plain, rtol=1e-5)


def test_fit_smooth_scale_peaked_fisher_does_not_crush_bulk():
    """Sparse Fisher must boost the peak without collapsing median s.

    Regression for the old ``ã = a · F̃^beta`` path, which with max-norm
    F̃ drove ~99% of channels toward ``s_min`` and zeroed W4A4 success.
    """
    K = 1024
    W = torch.ones(4, K)
    a = torch.ones(K) * 10.0
    fisher = torch.full((K,), 1e-6)
    fisher[0] = 1.0  # single salient channel

    plain = fit_smooth_scale(
        layer_name="plain",
        weight=W,
        act_channel_amax=a,
        alpha=1.0,
        s_min=1e-12,
        s_max=1e12,
    )
    boosted = fit_smooth_scale(
        layer_name="boost",
        weight=W,
        act_channel_amax=a,
        alpha=1.0,
        fisher=fisher,
        fisher_beta=1.0,
        s_min=1e-12,
        s_max=1e12,
    )

    # Peak rises; bulk stays near plain (mean-1 boost).
    assert float(boosted[0].item()) > float(plain[0].item())
    assert float(boosted.median().item()) > 0.5 * float(plain.median().item())
    assert float(boosted.median().item()) > 1.0


def test_fit_smooth_scale_fisher_gated_by_alpha():
    """Fisher folds into ``a`` then ``^alpha``; alpha=0 must ignore Fisher."""
    K = 8
    W = torch.randn(4, K)
    a = torch.ones(K)
    fisher = torch.ones(K)
    fisher[3] = 16.0

    s_alpha0 = fit_smooth_scale(
        layer_name="a0",
        weight=W,
        act_channel_amax=a,
        alpha=0.0,
        fisher=fisher,
        fisher_beta=1.0,
    )
    s_plain0 = fit_smooth_scale(
        layer_name="p0", weight=W, act_channel_amax=a, alpha=0.0
    )
    assert torch.allclose(s_alpha0, s_plain0, rtol=1e-5)

    s_alpha1 = fit_smooth_scale(
        layer_name="a1",
        weight=W,
        act_channel_amax=a,
        alpha=1.0,
        fisher=fisher,
        fisher_beta=1.0,
    )
    assert int(s_alpha1.argmax().item()) == 3


# --------------------------------------------------------------------------- #
# Config validation                                                           #
# --------------------------------------------------------------------------- #


def test_scope_config_allows_smooth_plus_hadamard():
    cfg = ScopeConfig(pipeline=("smooth", "hadamard"))
    assert cfg.smooth_enabled
    cfg = ScopeConfig(
        pipeline=("clip", "smooth", "hadamard"),
        act_outlier_std_k=3.0,
    )
    assert cfg.clip_enabled and cfg.smooth_enabled


def test_scope_config_rejects_smooth_plus_perm():
    with pytest.raises(ValueError, match="Unsupported pipeline"):
        ScopeConfig(pipeline=("smooth", "perm", "svd", "hadamard"))


def test_scope_config_allows_clip_only_plus_hadamard():
    cfg = ScopeConfig(
        pipeline=("clip", "hadamard"),
        act_outlier_kappa=2.0,
    )
    assert cfg.clip_enabled


def test_amax_collect_plan_never_collects_adaptive():
    """Final calibration never collects adaptive stats — clip is fitted in pipeline build."""
    from qvla.build.builder import _amax_collect_plan

    for pipeline in [
        ("clip",),
        ("clip", "hadamard"),
        ("clip", "smooth"),
        ("clip", "smooth", "hadamard"),
    ]:
        cfg = ScopeConfig(
            pipeline=pipeline,
            act_outlier_kappa=2.0,
            act_scale_mode="dynamic",
        )
        plan = _amax_collect_plan(cfg)
        assert plan.collect_adaptive_inner_channel is False


def test_scope_config_rejects_unsupported_pipeline_order():
    with pytest.raises(ValueError, match="Unsupported pipeline"):
        ScopeConfig(pipeline=("hadamard", "clip"), act_outlier_kappa=2.0)


def test_scope_config_from_dict_no_legacy_rewrite():
    """Current fields only; no auto-insert of clip / smooth from old keys."""
    cfg = QVLAConfig.from_dict(
        {
            "model_kind": "pi05",
            "llm": {
                "pipeline": ["clip", "smooth"],
                "act_outlier_kappa": 2.0,
                "smooth_alpha": 0.5,
            },
        }
    )
    assert cfg.llm.pipeline == ("clip", "smooth")
    assert cfg.llm.smooth_enabled
    assert cfg.llm.clip_enabled


def test_scope_config_allows_smooth_plus_per_token():
    cfg = ScopeConfig(pipeline=("smooth",), smooth_alpha=0.5)
    assert cfg.smooth_enabled


def test_scope_config_rejects_bad_alpha():
    with pytest.raises(ValueError, match="smooth_alpha"):
        ScopeConfig(pipeline=("smooth",), smooth_alpha=2.0)


def test_scope_config_rejects_bad_act_outlier_fit_tokens():
    with pytest.raises(ValueError, match="act_outlier_fit_tokens"):
        ScopeConfig(
            pipeline=("clip",),
            act_outlier_kappa=2.0,
            act_outlier_fit_tokens="invalid_token_string",
        )


def test_scope_config_rejects_bad_smooth_act_percentile():
    with pytest.raises(ValueError, match="smooth_act_percentile"):
        ScopeConfig(
            pipeline=("smooth",),
            smooth_alpha=0.5,
            smooth_act_percentile=-1.0,
        )


def test_scope_config_rejects_outlier_kappa_with_fixed_percentile():
    with pytest.raises(ValueError, match="mutually exclusive"):
        ScopeConfig(
            pipeline=("clip",),
            act_outlier_kappa=2.0,
            smooth_act_percentile=99.5,
        )


def test_scope_config_rejects_outlier_std_k_with_fixed_percentile():
    with pytest.raises(ValueError, match="mutually exclusive"):
        ScopeConfig(
            pipeline=("clip",),
            act_outlier_std_k=2.0,
            smooth_act_percentile=99.5,
        )


def test_scope_config_rejects_kappa_and_std_k_together():
    with pytest.raises(ValueError, match="act_outlier_kappa.*act_outlier_std_k"):
        ScopeConfig(
            pipeline=("clip",),
            act_outlier_kappa=2.0,
            act_outlier_std_k=2.0,
        )


# --------------------------------------------------------------------------- #
# AmaxCollectPlan                                                             #
# --------------------------------------------------------------------------- #




# --------------------------------------------------------------------------- #
# Adaptive inner amax                                                         #
# --------------------------------------------------------------------------- #


def test_adaptive_inner_uses_concatenated_tokens():
    """Adaptive inner amax should come from concatenated token data."""
    from qvla.build.collector import AmaxCollectPlan, LayerStats

    K = 4
    plan = AmaxCollectPlan(
        collect_adaptive_inner_channel=True,
        outlier_kappa=2.0,
        outlier_bulk_percentile=50.0,
    )
    stats = LayerStats(in_features=K)
    stats.init(K, num_steps=1, device="cpu", amax_plan=plan)
    x = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    stats.update(x, step=0)
    amax = stats.act_channel_inner_amax()
    assert amax.shape == (K,)


def test_adaptive_inner_rejects_empty_collection():
    from qvla.build.collector import AmaxCollectPlan, LayerStats

    K = 4
    plan = AmaxCollectPlan()
    stats = LayerStats(in_features=K)
    stats.init(K, num_steps=1, device="cpu", amax_plan=plan)
    x = torch.randn(2, K)
    stats.update(x, step=0)
    with pytest.raises(RuntimeError, match="inner-channel"):
        stats.act_channel_inner_amax()


def test_channel_outlier_bulk_kappa_amax_clips_sparse_spike():
    """A single huge spike in a mostly-normal channel should be clipped."""
    from qvla.build.collector import AmaxCollectPlan, LayerStats

    K = 4
    plan = AmaxCollectPlan(
        collect_adaptive_inner_channel=True,
        outlier_kappa=2.0,
        outlier_bulk_percentile=80.0,
    )
    stats = LayerStats(in_features=K)
    stats.init(K, num_steps=1, device="cpu", amax_plan=plan)
    x = torch.randn(100, K) * 0.1
    x[0, 0] = 100.0  # spike
    stats.update(x, step=0)

    amax = stats.act_channel_inner_amax()
    assert amax[0] < 100.0


def test_channel_outlier_bulk_kappa_amax_negative_kappa():
    from qvla.build.collector import AmaxCollectPlan

    plan = AmaxCollectPlan(
        collect_adaptive_inner_channel=True,
        outlier_kappa=-1.0,
    )
    assert plan.outlier_kappa == -1.0


def test_channel_outlier_mean_std_amax_clips_sparse_spike():
    from qvla.build.collector import AmaxCollectPlan, LayerStats

    K = 4
    plan = AmaxCollectPlan(
        collect_adaptive_inner_channel=True,
        outlier_std_k=2.0,
    )
    stats = LayerStats(in_features=K)
    stats.init(K, num_steps=1, device="cpu", amax_plan=plan)
    x = torch.randn(100, K) * 0.1
    x[0, 0] = 100.0
    stats.update(x, step=0)
    amax = stats.act_channel_inner_amax()
    assert amax[0] < 100.0


def test_channel_outlier_mean_std_amax_negative_std_k():
    from qvla.build.collector import AmaxCollectPlan

    plan = AmaxCollectPlan(
        collect_adaptive_inner_channel=True,
        outlier_std_k=-1.0,
    )
    assert plan.outlier_std_k == -1.0


def test_original_mean_std_path_is_bit_identical_when_selective_is_disabled():
    from qvla.build.collector import AmaxCollectPlan, LayerStats

    torch.manual_seed(7)
    x = torch.randn(100, 8)
    plan = AmaxCollectPlan(
        collect_adaptive_inner_channel=True,
        outlier_std_k=3.0,
        outlier_selective_channels=False,
    )
    stats = LayerStats(in_features=x.shape[1])
    stats.init(x.shape[1], num_steps=1, device="cpu", amax_plan=plan)
    stats.update(x, step=0)

    # Adaptive chunks are intentionally buffered as fp16 by the existing path.
    expected = channel_outlier_mean_std_amax(
        x.abs().to(torch.float16),
        std_k=3.0,
    )
    assert torch.equal(stats.act_channel_inner_amax(), expected)


def test_selective_mean_std_clips_only_robustly_large_amax_channels():
    maxima = torch.tensor([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    normal = torch.stack(
        [torch.linspace(0.0, float(v), 100) for v in maxima],
        dim=1,
    )
    outlier = torch.full((100, 1), 0.1)
    outlier[0, 0] = 100.0
    x = torch.cat((normal, outlier), dim=1)

    result = selective_channel_outlier_mean_std_amax(x, std_k=3.0)
    hard_amax = x.amax(dim=0)
    assert torch.equal(result[:-1], hard_amax[:-1])
    assert result[-1] < hard_amax[-1]


def test_selective_mean_std_rejects_zero_cross_channel_mad():
    x = torch.ones(16, 8)
    with pytest.raises(RuntimeError, match="positive cross-channel MAD"):
        selective_channel_outlier_mean_std_amax(x, std_k=3.0)


def test_pipeline_build_propagates_selective_channel_clip_to_plan():
    build = PipelineBuild(
        d=8,
        block_size=8,
        weight=torch.randn(4, 8),
        pipeline=("clip",),
        perm_score="weight",
        svd_source="weight",
        clip_std_k=3.0,
        clip_selective_channels=True,
    )
    plan = build.step_amax_plan(0)
    assert plan.outlier_selective_channels is True


def test_scope_config_rejects_selective_channels_without_mean_std_clip():
    with pytest.raises(ValueError, match="requires act_outlier_std_k>0"):
        ScopeConfig(
            pipeline=("clip",),
            act_outlier_kappa=2.0,
            act_outlier_selective_channels=True,
        )


def test_scope_config_rejects_global_clip_without_mean_std():
    with pytest.raises(ValueError, match="act_outlier_global=True requires"):
        ScopeConfig(
            pipeline=("clip",),
            act_outlier_kappa=2.0,
            act_outlier_global=True,
        )


def test_scope_config_rejects_global_clip_with_selective():
    with pytest.raises(ValueError, match="mutually exclusive"):
        ScopeConfig(
            pipeline=("clip",),
            act_outlier_std_k=3.0,
            act_outlier_global=True,
            act_outlier_selective_channels=True,
        )


def test_layer_outlier_mean_std_uses_one_shared_threshold():
    values = torch.tensor(
        [
            [1.0, 1.0, 100.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    result = layer_outlier_mean_std_amax(values, std_k=3.0)
    flat = values.reshape(-1)
    thr = flat.mean() + 3.0 * flat.std(unbiased=False)
    expected = torch.minimum(values.amax(dim=0), thr)
    torch.testing.assert_close(result, expected)
    assert result[0] == values[:, 0].amax()
    assert result[2] < values[:, 2].amax()
    assert result[0] == result[1]


def test_layer_outlier_mean_std_rejects_too_few_values():
    with pytest.raises(ValueError, match=">= 2"):
        layer_outlier_mean_std_amax(torch.tensor([[1.0]]), std_k=3.0)


def test_pipeline_build_propagates_global_clip_to_plan():
    build = PipelineBuild(
        d=8,
        block_size=8,
        weight=torch.randn(4, 8),
        pipeline=("clip",),
        perm_score="weight",
        svd_source="weight",
        clip_std_k=3.0,
        clip_global=True,
    )
    plan = build.step_amax_plan(0)
    assert plan.outlier_global is True
    assert plan.outlier_selective_channels is False


def test_adaptive_inner_global_clip_matches_layer_helper():
    from qvla.build.collector import AmaxCollectPlan, LayerStats

    torch.manual_seed(3)
    x = torch.randn(32, 6)
    x[0, 5] = 80.0
    plan = AmaxCollectPlan(
        collect_adaptive_inner_channel=True,
        outlier_std_k=3.0,
        outlier_global=True,
    )
    stats = LayerStats(in_features=x.shape[1])
    stats.init(x.shape[1], num_steps=1, device="cpu", amax_plan=plan)
    stats.update(x, step=0)
    expected = layer_outlier_mean_std_amax(
        x.abs().to(torch.float16),
        std_k=3.0,
    )
    torch.testing.assert_close(stats.act_channel_inner_amax(), expected)


def test_adaptive_inner_mean_std_uses_concatenated_tokens():
    from qvla.build.collector import AmaxCollectPlan, LayerStats

    K = 4
    plan = AmaxCollectPlan(
        collect_adaptive_inner_channel=True,
        outlier_std_k=2.0,
    )
    stats = LayerStats(in_features=K)
    stats.init(K, num_steps=1, device="cpu", amax_plan=plan)
    x = torch.randn(20, K)
    stats.update(x, step=0)
    amax = stats.act_channel_inner_amax()
    assert amax.shape == (K,)


def test_adaptive_inner_tip_clip_floored_by_rest_max():
    """When kappa selects many tokens as tip, floor the clip at rest-max."""
    from qvla.build.collector import AmaxCollectPlan, LayerStats

    K = 2
    plan = AmaxCollectPlan(
        collect_adaptive_inner_channel=True,
        outlier_kappa=0.01,
        outlier_bulk_percentile=95.0,
    )
    stats = LayerStats(in_features=K)
    stats.init(K, num_steps=1, device="cpu", amax_plan=plan)
    x = torch.randn(200, K)
    stats.update(x, step=0)
    amax = stats.act_channel_inner_amax()
    rest_max = x.abs().amax(dim=0)
    for j in range(K):
        assert amax[j].item() >= 0


# --------------------------------------------------------------------------- #
# Needs-fisher helper                                                         #
# --------------------------------------------------------------------------- #


def test_needs_fisher_reflects_smooth_beta():
    cfg = QVLAConfig.pi05_default()
    cfg_with = replace(
        cfg, llm=replace(cfg.llm, pipeline=("smooth",), smooth_fisher_beta=0.5)
    )
    assert cfg_with.needs_fisher is True
    cfg_without = replace(
        cfg, llm=replace(cfg.llm, smooth_fisher_beta=0.0)
    )
    no_perm = replace(
        cfg_without,
        llm=replace(cfg_without.llm, perm_score="weight"),
        dit=replace(cfg_without.dit, perm_score="weight"),
    )
    assert no_perm.needs_fisher is False


# --------------------------------------------------------------------------- #
# Pack save / load                                                            #
# --------------------------------------------------------------------------- #


def test_pack_save_load_preserves_smooth(tmp_path):
    cfg = QVLAConfig.pi05_default()
    K, N = 32, 16
    W = torch.randn(N, K) * 0.05
    qw = no_quantize(W.to(torch.bfloat16))
    s = torch.rand(K) + 0.1
    rot = Transform(
        mode="smooth",
        block_size=16,
        d=K,
        smooth_scale=s,
        pipeline=("smooth",),
    )
    layer = LayerPack(
        name="dit.layer",
        scope="dit",
        in_features=K,
        out_features=N,
        bias_present=False,
        qweight=qw.qweight,
        weight_scale=qw.weight_scale,
        group_size=qw.group_size,
        weight_bits=qw.weight_bits,
        rotation=rot,
        act_bits=16,
        act_scale_mode="dynamic",
        act_scale_table=None,
        bias=None,
        residual=None,
        extras={"fp_weight": W.to(torch.bfloat16)},
    )
    pack = Pack(config=cfg, layers={"dit.layer": layer})
    path = tmp_path / "pack.pt"
    pack.save(path)
    loaded = Pack.load(path)
    lp2 = loaded.layers["dit.layer"]
    assert lp2.rotation.smooth_scale is not None
    assert torch.allclose(lp2.rotation.smooth_scale, s.to(torch.float32))


def test_pack_save_load_preserves_act_clip(tmp_path):
    cfg = QVLAConfig.pi05_default()
    K, N = 32, 16
    W = torch.randn(N, K) * 0.05
    qw = no_quantize(W.to(torch.bfloat16))
    clip = torch.rand(K) + 0.5
    rot = Transform(
        mode="clip",
        block_size=16,
        d=K,
        act_clip=clip,
        pipeline=("clip",),
    )
    layer = LayerPack(
        name="llm.layer",
        scope="llm",
        in_features=K,
        out_features=N,
        bias_present=False,
        qweight=qw.qweight,
        weight_scale=qw.weight_scale,
        group_size=qw.group_size,
        weight_bits=qw.weight_bits,
        rotation=rot,
        act_bits=16,
        act_scale_mode="dynamic",
        act_scale_table=None,
        bias=None,
        residual=None,
        extras={"fp_weight": W.to(torch.bfloat16)},
    )
    path = tmp_path / "pack_clip.pt"
    Pack(config=cfg, layers={"llm.layer": layer}).save(path)
    lp2 = Pack.load(path).layers["llm.layer"]
    assert lp2.rotation.act_clip is not None
    assert torch.allclose(lp2.rotation.act_clip, clip.to(torch.float32))


def test_pack_save_load_identity_smooth_default(tmp_path):
    """LayerPacks constructed without smooth survive save/load."""
    cfg = QVLAConfig.pi05_default()
    K, N = 32, 16
    W = torch.randn(N, K) * 0.05
    qw = no_quantize(W.to(torch.bfloat16))
    layer = LayerPack(
        name="dit.layer", scope="dit",
        in_features=K, out_features=N, bias_present=False,
        qweight=qw.qweight, weight_scale=qw.weight_scale,
        group_size=qw.group_size, weight_bits=qw.weight_bits,
        rotation=identity_transform(K, 16),
        act_bits=16, act_scale_mode="dynamic",
        act_scale_table=None, bias=None, residual=None,
        extras={"fp_weight": W.to(torch.bfloat16)},
    )
    pack = Pack(config=cfg, layers={"dit.layer": layer})
    path = tmp_path / "pack.pt"
    pack.save(path)
    loaded = Pack.load(path)
    assert loaded.layers["dit.layer"].rotation.smooth_scale is None


# --------------------------------------------------------------------------- #
# Runtime forward equivalence                                                 #
# --------------------------------------------------------------------------- #


def test_quant_linear_smooth_bf16_is_identity_forward():
    """With weight_bits=16 / act_bits=16 (no quantisation), the SmoothQuant
    transform must be a mathematical identity: ``y = (x/s) · (W·diag(s))ᵀ``
    equals ``y = x · Wᵀ`` up to bf16 rounding."""
    torch.manual_seed(3)
    K, N = 64, 32
    W = torch.randn(N, K) * 0.05
    s = torch.rand(K) + 0.1
    W_smooth = W * s.unsqueeze(0)  # W · diag(s)
    qw = no_quantize(W_smooth.to(torch.bfloat16))
    rot = Transform(
        mode="smooth",
        block_size=32,
        d=K,
        smooth_scale=s,
        pipeline=("smooth",),
    )
    layer_pack = LayerPack(
        name="dit.layer",
        scope="dit",
        in_features=K,
        out_features=N,
        bias_present=False,
        qweight=qw.qweight,
        weight_scale=qw.weight_scale,
        group_size=qw.group_size,
        weight_bits=qw.weight_bits,
        rotation=rot,
        act_bits=16,
        act_scale_mode="dynamic",
        act_scale_table=None,
        bias=None,
        residual=None,
        extras={"fp_weight": W_smooth.to(torch.bfloat16)},
    )
    layer = QuantLinear(
        layer_pack, return_tuple=False, skip_bias_add=False,
        output_dtype=torch.float32, device="cpu",
    )
    x = torch.randn(4, K)
    y_ref = x @ W.T
    y_q = layer(x)
    assert torch.allclose(y_q, y_ref, rtol=5e-2, atol=5e-2)


def test_quant_linear_without_smooth_is_bit_identical():
    """LayerPack without smooth must give exactly the same forward as
    before the SmoothQuant feature landed."""
    torch.manual_seed(4)
    K, N = 64, 32
    W = torch.randn(N, K) * 0.05
    qw = no_quantize(W.to(torch.bfloat16))
    layer_pack = LayerPack(
        name="dit.layer", scope="dit",
        in_features=K, out_features=N, bias_present=False,
        qweight=qw.qweight, weight_scale=qw.weight_scale,
        group_size=qw.group_size, weight_bits=qw.weight_bits,
        rotation=identity_transform(K, 32),
        act_bits=16, act_scale_mode="dynamic",
        act_scale_table=None, bias=None, residual=None,
        extras={"fp_weight": W.to(torch.bfloat16)},
    )
    layer = QuantLinear(
        layer_pack, return_tuple=False, skip_bias_add=False,
        output_dtype=torch.float32, device="cpu",
    )
    x = torch.randn(4, K)
    W_bf = W.to(torch.bfloat16).to(torch.float32)
    y_ref = x @ W_bf.T
    y_q = layer(x)
    assert torch.allclose(y_q, y_ref, rtol=1e-5, atol=1e-5)
    assert layer._has_smooth is False


def test_quant_linear_runtime_act_clip_before_smooth():
    """Runtime must clamp each channel to act_clip before x/s."""
    torch.manual_seed(5)
    K, N = 8, 4
    s = torch.ones(K)
    clip = torch.full((K,), 0.5)
    W = torch.randn(N, K) * 0.1
    W_smooth = W * s.unsqueeze(0)
    qw = no_quantize(W_smooth.to(torch.bfloat16))
    rot = Transform(
        mode="clip+smooth",
        block_size=K,
        d=K,
        act_clip=clip,
        smooth_scale=s,
        pipeline=("clip", "smooth"),
    )
    layer_pack = LayerPack(
        name="llm.layer",
        scope="llm",
        in_features=K,
        out_features=N,
        bias_present=False,
        qweight=qw.qweight,
        weight_scale=qw.weight_scale,
        group_size=qw.group_size,
        weight_bits=qw.weight_bits,
        rotation=rot,
        act_bits=16,
        act_scale_mode="dynamic",
        act_scale_table=None,
        bias=None,
        residual=None,
        extras={"fp_weight": W_smooth.to(torch.bfloat16)},
    )
    layer = QuantLinear(
        layer_pack,
        return_tuple=False,
        skip_bias_add=False,
        output_dtype=torch.float32,
        device="cpu",
    )
    x = torch.randn(3, K) * 5.0
    x_clip = x.clamp(min=-0.5, max=0.5)
    y_ref = x_clip @ W.T
    y_q = layer(x)
    assert torch.allclose(y_q, y_ref, rtol=5e-2, atol=5e-2)


def test_quant_linear_clip_only_without_smooth():
    """Clip-only pack: no s divide, weight unchanged, activations clamped."""
    torch.manual_seed(6)
    K, N = 8, 4
    clip = torch.full((K,), 0.5)
    W = torch.randn(N, K) * 0.1
    qw = no_quantize(W.to(torch.bfloat16))
    rot = Transform(
        mode="clip",
        block_size=K,
        d=K,
        act_clip=clip,
        pipeline=("clip",),
    )
    layer_pack = LayerPack(
        name="llm.layer",
        scope="llm",
        in_features=K,
        out_features=N,
        bias_present=False,
        qweight=qw.qweight,
        weight_scale=qw.weight_scale,
        group_size=qw.group_size,
        weight_bits=qw.weight_bits,
        rotation=rot,
        act_bits=16,
        act_scale_mode="dynamic",
        act_scale_table=None,
        bias=None,
        residual=None,
        extras={"fp_weight": W.to(torch.bfloat16)},
    )
    layer = QuantLinear(
        layer_pack,
        return_tuple=False,
        skip_bias_add=False,
        output_dtype=torch.float32,
        device="cpu",
    )
    assert layer._has_smooth is False
    x = torch.randn(3, K) * 5.0
    y_ref = x.clamp(min=-0.5, max=0.5) @ W.to(torch.bfloat16).float().T
    y_q = layer(x)
    assert torch.allclose(y_q, y_ref, rtol=5e-2, atol=5e-2)


def test_pmean_per_step_channel_amax_matches_formula():
    from qvla.core.smooth_fit import pmean_per_step_channel_amax

    table = torch.tensor([[1.0, 8.0], [3.0, 2.0]], dtype=torch.float32)
    got = pmean_per_step_channel_amax(table, 4.0)
    expected = (table.pow(4).mean(dim=0)).pow(0.25)
    torch.testing.assert_close(got, expected)


def test_pmean_per_step_rejects_nonpositive_p():
    from qvla.core.smooth_fit import pmean_per_step_channel_amax

    table = torch.ones(2, 3)
    with pytest.raises(ValueError, match="p-mean p must be finite"):
        pmean_per_step_channel_amax(table, 0.0)


def test_scope_config_rejects_pmean_without_smooth():
    with pytest.raises(ValueError, match="requires 'smooth' in pipeline"):
        ScopeConfig(num_steps=4, smooth_step_pmean_p=4.0)


def test_scope_config_rejects_pmean_with_num_steps_one():
    with pytest.raises(ValueError, match="requires num_steps>1"):
        ScopeConfig(pipeline=("smooth",), smooth_step_pmean_p=4.0)


def test_scope_config_rejects_pmean_with_inner_percentile():
    with pytest.raises(ValueError, match="smooth_act_percentile==100"):
        ScopeConfig(
            pipeline=("smooth",),
            num_steps=4,
            smooth_step_pmean_p=4.0,
            smooth_act_percentile=99.9,
        )


def test_act_channel_cross_amax_pmean_full_steps():
    from qvla.build.collector import AmaxCollectPlan, LayerStats
    from qvla.core.smooth_fit import pmean_per_step_channel_amax

    stats = LayerStats(in_features=2)
    stats.init(
        2,
        2,
        "cpu",
        amax_plan=AmaxCollectPlan(collect_hessian=False, collect_cross_channel=True),
    )
    stats.update(torch.tensor([[1.0, 8.0]]), step=0)
    stats.update(torch.tensor([[3.0, 2.0]]), step=1)
    got = stats.act_channel_cross_amax_pmean(p=4.0)
    expected = pmean_per_step_channel_amax(stats.per_step_cross_channel_amax, 4.0)
    torch.testing.assert_close(got, expected)
    assert not torch.allclose(got, stats.static_cross_channel_amax)


def test_act_channel_cross_amax_pmean_prefix_only_uses_static():
    from qvla.build.collector import AmaxCollectPlan, LayerStats

    stats = LayerStats(in_features=2)
    stats.init(
        2,
        2,
        "cpu",
        amax_plan=AmaxCollectPlan(collect_hessian=False, collect_cross_channel=True),
    )
    stats.update(torch.tensor([[1.0, 4.0], [2.0, 3.0]]), step=None)
    got = stats.act_channel_cross_amax_pmean(p=4.0)
    torch.testing.assert_close(got, stats.static_cross_channel_amax)


def test_act_channel_cross_amax_pmean_partial_coverage_raises():
    from qvla.build.collector import AmaxCollectPlan, LayerStats

    stats = LayerStats(in_features=2)
    stats.init(
        2,
        2,
        "cpu",
        amax_plan=AmaxCollectPlan(collect_hessian=False, collect_cross_channel=True),
    )
    stats.update(torch.tensor([[1.0, 2.0]]), step=0)
    with pytest.raises(RuntimeError, match="every denoise step or prefix-only"):
        stats.act_channel_cross_amax_pmean(p=4.0)


def test_assert_smooth_pmean_step_coverage_requires_one_full_layer():
    from qvla.build.collector import (
        AmaxCollectPlan,
        LayerStats,
        assert_smooth_pmean_step_coverage,
    )

    prefix = LayerStats(in_features=2)
    prefix.init(
        2,
        2,
        "cpu",
        amax_plan=AmaxCollectPlan(collect_hessian=False, collect_cross_channel=True),
    )
    prefix.update(torch.tensor([[1.0, 1.0]]), step=None)
    with pytest.raises(RuntimeError, match="at least one layer observed"):
        assert_smooth_pmean_step_coverage({"kv": prefix})

    full = LayerStats(in_features=2)
    full.init(
        2,
        2,
        "cpu",
        amax_plan=AmaxCollectPlan(collect_hessian=False, collect_cross_channel=True),
    )
    full.update(torch.tensor([[1.0, 1.0]]), step=0)
    full.update(torch.tensor([[2.0, 2.0]]), step=1)
    assert_smooth_pmean_step_coverage({"kv": prefix, "ff": full})


def test_pipeline_smooth_pmean_differs_from_static_max():
    from qvla.build.collector import AmaxCollectPlan, LayerStats
    from qvla.core.smooth_fit import fit_smooth_scale, pmean_per_step_channel_amax

    stats = LayerStats(in_features=2)
    stats.init(
        2,
        2,
        "cpu",
        amax_plan=AmaxCollectPlan(collect_hessian=False, collect_cross_channel=True),
    )
    stats.update(torch.tensor([[1.0, 8.0]]), step=0)
    stats.update(torch.tensor([[3.0, 2.0]]), step=1)
    weight = torch.ones(3, 2)
    build = PipelineBuild(
        d=2,
        block_size=2,
        weight=weight,
        pipeline=("smooth",),
        perm_score="weight",
        svd_source="weight",
        smooth_step_pmean_p=4.0,
    )
    build.fit_step(0, stats=stats)
    expected = fit_smooth_scale(
        layer_name=None,
        weight=weight,
        act_channel_amax=pmean_per_step_channel_amax(
            stats.per_step_cross_channel_amax, 4.0
        ),
        alpha=0.5,
    )
    torch.testing.assert_close(build.smooth_scale, expected)
    legacy = fit_smooth_scale(
        layer_name=None,
        weight=weight,
        act_channel_amax=stats.static_cross_channel_amax,
        alpha=0.5,
    )
    assert not torch.allclose(build.smooth_scale, legacy)
