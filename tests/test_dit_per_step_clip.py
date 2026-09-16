"""Per-step DiT adaptive clip: fit, coverage errors, and runtime indexing."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_PKG_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from qvla.build.collector import (
    AmaxCollectPlan,
    LayerStats,
    assert_dit_clip_has_per_step_table,
)
from qvla.config import QVLAConfig, ScopeConfig
from qvla.core.clip import (
    scheduled_outlier_std_k,
    std_k_schedule_bounds,
    selective_channel_outlier_mean_std_amax,
)
from qvla.core.pack import LayerPack
from qvla.core.pipeline import PipelineBuild, Transform
from qvla.core.quantize import no_quantize
from qvla.runtime.quant_linear import QuantLinear


def _adaptive_plan(**kwargs) -> AmaxCollectPlan:
    return AmaxCollectPlan(collect_hessian=False, collect_adaptive_inner_channel=True, **kwargs)


def _selective_tokens(
    n_tokens: int, n_channels: int, *, spike: float
) -> torch.Tensor:
    """Varied-amax channels plus one MAD outlier so selective clip is defined."""
    maxima = torch.linspace(1.0, 1.5, n_channels - 1)
    normal = torch.stack(
        [torch.linspace(0.0, float(v), n_tokens) for v in maxima],
        dim=1,
    )
    outlier = torch.full((n_tokens, 1), 0.1)
    outlier[0, 0] = spike
    return torch.cat((normal, outlier), dim=1)


def test_qvla_config_dit_clip_requires_num_steps():
    with pytest.raises(ValueError, match="num_steps>1"):
        QVLAConfig(
            dit=ScopeConfig(
                pipeline=("clip",),
                num_steps=1,
                act_outlier_std_k=3.0,
                act_outlier_selective_channels=True,
                act_outlier_fit_tokens="all",
            )
        )


def test_qvla_config_dit_clip_requires_selective_mean_std():
    with pytest.raises(ValueError, match="act_outlier_selective_channels"):
        QVLAConfig(
            dit=ScopeConfig(
                pipeline=("clip",),
                num_steps=10,
                act_outlier_std_k=3.0,
                act_outlier_fit_tokens="all",
            )
        )
    with pytest.raises(ValueError, match="act_outlier_kappa"):
        QVLAConfig(
            dit=ScopeConfig(
                pipeline=("clip",),
                num_steps=10,
                act_outlier_kappa=2.0,
                act_outlier_fit_tokens="all",
            )
        )
    with pytest.raises(ValueError, match="act_outlier_fit_tokens"):
        QVLAConfig(
            dit=ScopeConfig(
                pipeline=("clip",),
                num_steps=10,
                act_outlier_std_k=3.0,
                act_outlier_selective_channels=True,
                act_outlier_fit_tokens="image_lang_pad",
            )
        )


def test_std_k_down_up_require_std_k_and_dit_steps():
    with pytest.raises(ValueError, match="act_outlier_std_k_down/up require act_outlier_std_k>0"):
        ScopeConfig(
            pipeline=("clip",),
            num_steps=4,
            act_outlier_kappa=2.0,
            act_outlier_std_k_down=0.5,
        )
    with pytest.raises(ValueError, match="act_outlier_std_k_down/up require num_steps>1"):
        ScopeConfig(
            pipeline=("clip",),
            act_outlier_std_k=3.0,
            act_outlier_std_k_up=0.5,
        )
    with pytest.raises(ValueError, match="std_k - act_outlier_std_k_down must be > 0"):
        ScopeConfig(
            pipeline=("clip",),
            num_steps=4,
            act_outlier_std_k=3.0,
            act_outlier_std_k_down=3.0,
            act_outlier_selective_channels=True,
            act_outlier_fit_tokens="all",
        )
    with pytest.raises(ValueError, match="DiT-only"):
        QVLAConfig(
            llm=ScopeConfig(
                pipeline=("clip",),
                num_steps=4,
                act_outlier_std_k=3.0,
                act_outlier_std_k_down=0.5,
                act_outlier_fit_tokens="image_lang_pad",
            )
        )


def test_qvla_config_dit_clip_accepts_selective_all_tokens():
    cfg = QVLAConfig(
        dit=ScopeConfig(
            pipeline=("clip",),
            num_steps=10,
            act_outlier_std_k=3.0,
            act_outlier_selective_channels=True,
            act_outlier_fit_tokens="all",
        )
    )
    assert cfg.dit.clip_enabled
    assert cfg.dit.num_steps == 10


def test_qvla_config_dit_clip_accepts_skip_first():
    cfg = QVLAConfig(
        model_kind="groot_n17",
        dit=ScopeConfig(
            pipeline=("clip",),
            num_steps=4,
            act_outlier_std_k=3.0,
            act_outlier_selective_channels=True,
            act_outlier_fit_tokens="skip_first",
        )
    )
    assert cfg.dit.act_outlier_fit_tokens == "skip_first"


def test_qvla_config_pi05_rejects_skip_first():
    with pytest.raises(ValueError, match="GR00T-only"):
        QVLAConfig(
            model_kind="pi05",
            dit=ScopeConfig(
                pipeline=("clip",),
                num_steps=10,
                act_outlier_std_k=3.0,
                act_outlier_selective_channels=True,
                act_outlier_fit_tokens="skip_first",
            )
        )


def test_per_step_selective_clip_is_independent():
    plan = _adaptive_plan(outlier_std_k=3.0, outlier_selective_channels=True)
    stats = LayerStats(in_features=7)
    stats.init(7, num_steps=2, device="cpu", amax_plan=plan)
    x0 = _selective_tokens(32, 7, spike=40.0)
    x1 = _selective_tokens(32, 7, spike=200.0)
    stats.update(x0, step=0)
    stats.update(x1, step=1)
    table = stats.act_channel_inner_amax()
    assert table.shape == (2, 7)
    expected0 = selective_channel_outlier_mean_std_amax(
        x0.abs().to(torch.float16), std_k=3.0
    )
    expected1 = selective_channel_outlier_mean_std_amax(
        x1.abs().to(torch.float16), std_k=3.0
    )
    assert torch.equal(table[0], expected0)
    assert torch.equal(table[1], expected1)
    assert not torch.equal(table[0], table[1])
    pooled = selective_channel_outlier_mean_std_amax(
        torch.cat((x0, x1), dim=0).abs().to(torch.float16), std_k=3.0
    )
    assert not torch.equal(table[0], pooled)
    assert not torch.equal(table[1], pooled)


def test_scheduled_outlier_std_k_endpoints():
    assert scheduled_outlier_std_k(2.5, 3.5, step=0, num_steps=5) == 2.5
    assert scheduled_outlier_std_k(2.5, 3.5, step=4, num_steps=5) == 3.5
    mid = scheduled_outlier_std_k(2.5, 3.5, step=2, num_steps=5)
    assert mid == pytest.approx(3.0)
    assert std_k_schedule_bounds(3.0, 0.5, 0.5) == (2.5, 3.5)
    assert std_k_schedule_bounds(3.0, 0.0, 0.0) == (3.0, 3.0)
    with pytest.raises(ValueError, match="num_steps>=2"):
        scheduled_outlier_std_k(2.5, 3.5, step=0, num_steps=1)
    with pytest.raises(ValueError, match="k_end"):
        scheduled_outlier_std_k(2.5, 0.0, step=0, num_steps=4)
    with pytest.raises(ValueError, match="std_k - down must be > 0"):
        std_k_schedule_bounds(3.0, 3.0, 0.5)


def test_per_step_std_k_down_up_around_3():
    plan = _adaptive_plan(
        outlier_std_k=3.0,
        outlier_std_k_down=0.5,
        outlier_std_k_up=0.5,
        outlier_selective_channels=True,
    )
    stats = LayerStats(in_features=7)
    stats.init(7, num_steps=3, device="cpu", amax_plan=plan)
    xs = [
        _selective_tokens(32, 7, spike=40.0),
        _selective_tokens(32, 7, spike=80.0),
        _selective_tokens(32, 7, spike=160.0),
    ]
    for t, x in enumerate(xs):
        stats.update(x, step=t)
    table = stats.act_channel_inner_amax()
    ks = (2.5, 3.0, 3.5)
    for t, (x, k) in enumerate(zip(xs, ks)):
        expected = selective_channel_outlier_mean_std_amax(
            x.abs().to(torch.float16), std_k=k
        )
        assert torch.equal(table[t], expected)
    unchanged = [
        selective_channel_outlier_mean_std_amax(
            x.abs().to(torch.float16), std_k=2.5
        )
        for x in xs
    ]
    assert not torch.equal(table[2], unchanged[2])


def test_per_step_clip_rejects_missing_step():
    plan = _adaptive_plan(outlier_std_k=3.0, outlier_selective_channels=True)
    stats = LayerStats(in_features=7)
    stats.init(7, num_steps=2, device="cpu", amax_plan=plan)
    stats.update(_selective_tokens(16, 7, spike=50.0), step=0)
    with pytest.raises(RuntimeError, match="every denoise step or prefix-only"):
        stats.act_channel_inner_amax()


def test_per_step_clip_rejects_mixed_prefix_and_steps():
    plan = _adaptive_plan(outlier_std_k=3.0, outlier_selective_channels=True)
    stats = LayerStats(in_features=7)
    stats.init(7, num_steps=2, device="cpu", amax_plan=plan)
    x = _selective_tokens(16, 7, spike=50.0)
    stats.update(x, step=None)
    with pytest.raises(RuntimeError, match="mixed prefix"):
        stats.update(x, step=0)

    stats_rev = LayerStats(in_features=7)
    stats_rev.init(7, num_steps=2, device="cpu", amax_plan=plan)
    stats_rev.update(x, step=0)
    with pytest.raises(RuntimeError, match="mixed prefix"):
        stats_rev.update(x, step=None)


def test_prefix_only_clip_is_1d():
    plan = _adaptive_plan(outlier_std_k=3.0, outlier_selective_channels=True)
    stats = LayerStats(in_features=7)
    stats.init(7, num_steps=4, device="cpu", amax_plan=plan)
    x = _selective_tokens(16, 7, spike=80.0)
    stats.update(x, step=None)
    clip = stats.act_channel_inner_amax()
    assert clip.ndim == 1
    assert clip.shape == (7,)
    expected = selective_channel_outlier_mean_std_amax(
        x.abs().to(torch.float16), std_k=3.0
    )
    assert torch.equal(clip, expected)


def test_prefix_only_clip_ignores_std_k_down_up():
    plan = _adaptive_plan(
        outlier_std_k=3.0,
        outlier_std_k_down=0.5,
        outlier_std_k_up=0.5,
        outlier_selective_channels=True,
    )
    stats = LayerStats(in_features=7)
    stats.init(7, num_steps=4, device="cpu", amax_plan=plan)
    x = _selective_tokens(16, 7, spike=80.0)
    stats.update(x, step=None)
    clip = stats.act_channel_inner_amax()
    expected = selective_channel_outlier_mean_std_amax(
        x.abs().to(torch.float16), std_k=3.0
    )
    assert torch.equal(clip, expected)


def test_per_step_skip_first_floors_by_token0():
    plan = _adaptive_plan(
        outlier_std_k=3.0,
        outlier_selective_channels=True,
        outlier_skip_first_token=True,
    )
    stats = LayerStats(in_features=7)
    stats.init(7, num_steps=2, device="cpu", amax_plan=plan)
    x0 = _selective_tokens(16, 7, spike=40.0)
    state0 = torch.zeros(7)
    state0[3] = 400.0
    x0 = torch.cat((state0.unsqueeze(0), x0), dim=0)
    x1 = _selective_tokens(16, 7, spike=90.0)
    state1 = torch.zeros(7)
    state1[3] = 250.0
    x1 = torch.cat((state1.unsqueeze(0), x1), dim=0)
    stats.update(x0, step=0)
    stats.update(x1, step=1)
    table = stats.act_channel_inner_amax()

    def expected(x: torch.Tensor) -> torch.Tensor:
        abs16 = x.abs().to(torch.float16)
        a = selective_channel_outlier_mean_std_amax(abs16[1:], std_k=3.0)
        return torch.maximum(a, abs16[0].to(torch.float32))

    assert torch.equal(table[0], expected(x0))
    assert torch.equal(table[1], expected(x1))
    included = selective_channel_outlier_mean_std_amax(
        x0.abs().to(torch.float16), std_k=3.0
    )
    assert not torch.equal(table[0], included)


def test_prefix_skip_first_still_uses_all_tokens():
    plan = _adaptive_plan(
        outlier_std_k=3.0,
        outlier_selective_channels=True,
        outlier_skip_first_token=True,
    )
    stats = LayerStats(in_features=7)
    stats.init(7, num_steps=4, device="cpu", amax_plan=plan)
    x = _selective_tokens(16, 7, spike=80.0)
    stats.update(x, step=None)
    clip = stats.act_channel_inner_amax()
    expected = selective_channel_outlier_mean_std_amax(
        x.abs().to(torch.float16), std_k=3.0
    )
    assert torch.equal(clip, expected)


def test_llm_num_steps_1_clip_stays_1d():
    plan = _adaptive_plan(outlier_std_k=3.0)
    stats = LayerStats(in_features=4)
    stats.init(4, num_steps=1, device="cpu", amax_plan=plan)
    x = torch.randn(20, 4)
    x[0, 0] = 50.0
    stats.update(x, step=0)
    clip = stats.act_channel_inner_amax()
    assert clip.shape == (4,)


def test_pipeline_build_stores_per_step_table():
    plan = _adaptive_plan(outlier_std_k=3.0, outlier_selective_channels=True)
    stats = LayerStats(in_features=7)
    stats.init(7, num_steps=2, device="cpu", amax_plan=plan)
    stats.update(_selective_tokens(16, 7, spike=40.0), step=0)
    stats.update(_selective_tokens(16, 7, spike=90.0), step=1)
    build = PipelineBuild(
        d=7,
        block_size=1,
        weight=torch.randn(3, 7),
        pipeline=("clip",),
        perm_score="weight",
        svd_source="weight",
        clip_std_k=3.0,
        clip_selective_channels=True,
    )
    build.fit_step(0, stats=stats)
    assert build.act_clip is not None
    assert build.act_clip.shape == (2, 7)


def test_transform_apply_indexes_per_step_clip():
    table = torch.tensor([[0.1, 0.1], [2.0, 2.0]])
    rot = Transform(
        mode="clip",
        block_size=2,
        d=2,
        act_clip=table,
        pipeline=("clip",),
    )
    x = torch.tensor([[10.0, -10.0]])
    out0 = rot.apply(x, step=0)
    out1 = rot.apply(x, step=1)
    assert torch.allclose(out0, torch.tensor([[0.1, -0.1]]))
    assert torch.allclose(out1, torch.tensor([[2.0, -2.0]]))
    with pytest.raises(RuntimeError, match="requires the current denoise step"):
        rot.apply(x)


def test_assert_dit_clip_requires_a_2d_table():
    with pytest.raises(RuntimeError, match="prefix-only"):
        assert_dit_clip_has_per_step_table(
            {"kv": torch.ones(8), "kv2": torch.ones(8)}
        )
    assert_dit_clip_has_per_step_table(
        {"kv": torch.ones(8), "ff": torch.ones(4, 8)}
    )


def _clip_layer_pack(act_clip: torch.Tensor, *, k: int = 8, n: int = 4) -> LayerPack:
    w = torch.randn(n, k) * 0.1
    qw = no_quantize(w.to(torch.bfloat16))
    rot = Transform(
        mode="clip",
        block_size=k,
        d=k,
        act_clip=act_clip,
        pipeline=("clip",),
    )
    return LayerPack(
        name="dit.layer",
        scope="dit",
        in_features=k,
        out_features=n,
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
        extras={"fp_weight": w.to(torch.bfloat16)},
    )


def test_quant_linear_per_step_clip_indexes_rows():
    k, n = 8, 4
    table = torch.stack(
        (torch.full((k,), 0.25), torch.full((k,), 1.5)),
        dim=0,
    )
    pack = _clip_layer_pack(table, k=k, n=n)
    w = pack.extras["fp_weight"].float()
    layer = QuantLinear(
        pack,
        return_tuple=False,
        skip_bias_add=False,
        output_dtype=torch.float32,
        device="cpu",
    )
    x = torch.randn(3, k) * 8.0
    y0 = layer(x)
    y1 = layer(x)
    ref0 = x.clamp(min=-0.25, max=0.25) @ w.T
    ref1 = x.clamp(min=-1.5, max=1.5) @ w.T
    assert torch.allclose(y0, ref0, rtol=5e-2, atol=5e-2)
    assert torch.allclose(y1, ref1, rtol=5e-2, atol=5e-2)
    assert not torch.allclose(y0, y1, rtol=1e-3, atol=1e-3)


def test_pack_save_load_preserves_per_step_act_clip(tmp_path):
    from qvla.core.pack import Pack

    cfg = QVLAConfig(
        dit=ScopeConfig(
            pipeline=("clip",),
            num_steps=4,
            act_outlier_std_k=3.0,
            act_outlier_selective_channels=True,
            act_outlier_fit_tokens="all",
        )
    )
    table = torch.rand(4, 32) + 0.5
    pack_layer = _clip_layer_pack(table, k=32, n=16)
    path = tmp_path / "dit_clip.pt"
    Pack(config=cfg, layers={"dit.layer": pack_layer}).save(path)
    loaded = Pack.load(path).layers["dit.layer"]
    assert loaded.rotation.act_clip is not None
    assert loaded.rotation.act_clip.shape == (4, 32)
    assert torch.allclose(loaded.rotation.act_clip, table.to(torch.float32))
