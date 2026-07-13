"""Smoke tests — exercise every public surface without needing GPU / phyai.

These tests are intentionally CPU-only and self-contained:

* Build a synthetic tiny "model" of pure ``nn.Linear`` layers.
* Run the rotation primitive in isolation.
* Round-trip a pack through quantize -> save -> load -> dequant.
* Replace layers via :func:`enable_quantization` and check the forward error.
* Verify the per-layer step counter advances correctly across simulated steps.

Run with::

    pytest /phyai_workspace/qvla/tests/ -v
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Make the package importable without `pip install`.
_PKG_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from qvla.config import QVLAConfig, ScopeConfig
from qvla.core.pack import LayerPack, Pack
from qvla.runtime import QuantLinear
from qvla.core.quantize import (
    QuantizedWeight, dequantize, gptq_quantize, is_no_quant, no_quantize,
    quantize_weight, rtn_quantize,
    rtn_residual_quantize, symmetric_quant_range,
)
from qvla.build.collector import (
    LayerStats,
    RotatedActivationCollector,
)
from qvla.core.rotation import identity_rotation
from rotation_helpers import fit_rotation
from qvla.runtime import (
    classify, enable_quantization, list_target_modules,
)


torch.manual_seed(0)


# --------------------------------------------------------------------------- #
# Rotation                                                                    #
# --------------------------------------------------------------------------- #


def test_hadamard_orthonormal():
    """Hadamard-only pipeline must be orthonormal block-wise."""
    for n in (2, 4, 8, 16, 64):
        rot = fit_rotation(
            d=n, block_size=n, weight=torch.randn(1, n), pipeline=("hadamard",)
        )
        R = rot.apply(torch.eye(n))
        assert torch.allclose(R @ R.T, torch.eye(n), atol=1e-5)


def test_identity_rotation_is_noop():
    rot = identity_rotation(d=128, block_size=64)
    x = torch.randn(3, 128)
    out = rot.apply(x)
    assert torch.equal(out, x)


def test_hadamard_rotation_invertible():
    rot = fit_rotation(
        d=128, block_size=64, weight=torch.randn(1, 128), pipeline=("hadamard",)
    )
    x = torch.randn(3, 128)
    R = rot.apply(torch.eye(128))
    y = x @ R
    y_re = y @ R.T
    assert torch.allclose(y_re, x, atol=1e-5)


def test_random_hadamard_orthonormal():
    """Random Hadamard pipeline step must be orthonormal."""
    rot = fit_rotation(
        d=64,
        block_size=16,
        weight=torch.randn(8, 64),
        pipeline=("random_hadamard",),
    )
    assert rot.random_hadamard_blocks is not None
    R = rot.apply(torch.eye(64))
    assert torch.allclose(R.T @ R, torch.eye(64), atol=1e-5)


def test_random_hadamard_preserves_linear_map():
    """Random Hadamard absorbed into W must preserve y = x W^T."""
    torch.manual_seed(1)
    n, k = 16, 64
    W = torch.randn(n, k) * 0.05
    x = torch.randn(4, k)
    y_ref = x @ W.T
    rot = fit_rotation(
        d=k,
        block_size=16,
        weight=W,
        pipeline=("perm", "svd", "random_hadamard"),
        perm_score="weight",
    )
    from qvla.build.builder import _rotate_weight

    W_t = _rotate_weight(W, rot)
    x_t = rot.apply(x)
    assert torch.allclose(x_t @ W_t.T, y_ref, atol=1e-4)


def test_random_hadamard_differs_from_standard():
    w = torch.randn(8, 64)
    std = fit_rotation(d=64, block_size=16, weight=w, pipeline=("hadamard",))
    rnd = fit_rotation(d=64, block_size=16, weight=w, pipeline=("random_hadamard",))
    x = torch.randn(4, 64)
    assert not torch.allclose(std.apply(x), rnd.apply(x), atol=1e-5)


def test_random_hadamard_reproducible_with_build_seed():
    """Same build_seed + layer name must yield identical random Hadamard blocks."""
    w = torch.randn(8, 64)
    kwargs = dict(
        d=64,
        block_size=16,
        weight=w,
        pipeline=("random_hadamard",),
        layer_name="expert_stack.layers.0.mlp.gate_up_proj",
        build_seed=42,
    )
    r1 = fit_rotation(**kwargs)
    r2 = fit_rotation(**kwargs)
    assert r1.random_hadamard_blocks is not None
    assert torch.equal(r1.random_hadamard_blocks, r2.random_hadamard_blocks)
    r3 = fit_rotation(**{**kwargs, "build_seed": 43})
    assert not torch.equal(r1.random_hadamard_blocks, r3.random_hadamard_blocks)


def test_svd_hadamard_orthonormal():
    """The SVD-Hadamard rotation matrix must be orthonormal block-wise."""
    d, b, n = 64, 16, 32
    W = torch.randn(n, d) * 0.1
    rot = fit_rotation(
        d=d, block_size=b, weight=W, pipeline=("svd", "hadamard")
    )
    R = rot.apply(torch.eye(d))
    assert torch.allclose(R.T @ R, torch.eye(d), atol=1e-4)


def test_weight_svd_rotation_preserves_linear_map():
    """Perm + rotation absorbed into W must preserve y = x W^T."""
    torch.manual_seed(1)
    n, k = 16, 64
    W = torch.randn(n, k) * 0.05
    x = torch.randn(4, k)
    y_ref = x @ W.T
    rot = fit_rotation(
        d=k, block_size=16, weight=W, pipeline=("perm", "svd", "hadamard"), perm_score="weight"
    )
    from qvla.build.builder import _rotate_weight

    W_t = _rotate_weight(W, rot)
    x_t = rot.apply(x)
    assert torch.allclose(x_t @ W_t.T, y_ref, atol=1e-4)


def test_rotated_activation_hessian_matches_congruence():
    """Phase-2 ``X'ᵀX'`` on rotation.apply(x) equals ``T.T @ H @ T``."""
    torch.manual_seed(2)
    k = 64
    x = torch.randn(32, k)
    H = x.T @ x
    W = torch.randn(16, k) * 0.05
    rot = fit_rotation(
        d=k, block_size=16, weight=W, pipeline=("perm", "svd", "hadamard"), perm_score="weight"
    )
    h_collected = rot.apply(x).T @ rot.apply(x)
    T = rot.apply(torch.eye(k))
    h_ref = T.T @ H @ T
    assert torch.allclose(h_collected, h_ref, atol=1e-3)


def test_activation_svd_after_perm_matches_direct_u_fit():
    """Pipeline-wise SVD on perm(x) cov must match _fit_u_blocks on the same cov."""
    torch.manual_seed(3)
    d, block_size, n = 64, 16, 32
    W = torch.randn(n, d) * 0.05
    X = torch.randn(256, d)
    amax = X.abs().amax(dim=0)

    from qvla.build.collector import LayerStats
    from qvla.core.rotation import (
        PipelineRotationBuild,
        Rotation,
        _fit_u_blocks,
    )

    builder = PipelineRotationBuild(
        d=d,
        block_size=block_size,
        weight=W,
        pipeline=("perm", "svd", "hadamard"),
        perm_score="activation",
        svd_source="activation",
    )
    perm_stats = LayerStats(in_features=d)
    perm_stats.static_channel_amax = amax
    perm_stats.n_tokens = X.shape[0]
    builder.fit_step(0, stats=perm_stats)

    X_perm = X[:, builder.perm]
    H_perm = (X_perm.T @ X_perm) / X.shape[0]
    svd_stats = LayerStats(in_features=d)
    svd_stats.xtx = H_perm.to(torch.float64) * X.shape[0]
    svd_stats.n_tokens = X.shape[0]
    builder.fit_step(1, stats=svd_stats)
    rot_new = builder.finish()

    u_ref = _fit_u_blocks(
        d=d,
        block_size=block_size,
        weight=W,
        activation_cov=H_perm,
        svd_source="activation",
        perm=builder.perm,
        sensitivity=None,
        eps=1e-6,
    )
    rot_ref = Rotation(
        mode="perm+svd+hadamard",
        block_size=block_size,
        d=d,
        u_blocks=u_ref,
        perm=builder.perm,
        pipeline=("perm", "svd", "hadamard"),
    )
    R_new = rot_new.apply(torch.eye(d))
    R_ref = rot_ref.apply(torch.eye(d))
    assert torch.allclose(R_new, R_ref, atol=1e-4)


# --------------------------------------------------------------------------- #
# Quantize                                                                    #
# --------------------------------------------------------------------------- #


def test_rtn_round_trip_close():
    """Dequant of a freshly RTN-quantized weight matches the original ~OK."""
    W = torch.randn(64, 128) * 0.1
    qw = rtn_quantize(W, group_size=32)
    W_dq = dequantize(qw)
    err = (W - W_dq).abs().max().item()
    # int4 with per-group scales ⇒ ≤ scale / 2 per element. With group_size=32
    # and randn*0.1, expected max scale ~0.07, so err ≤ ~0.01.
    assert err < 0.05, f"RTN max error too large: {err}"


def test_rtn_residual_keeps_outlier_columns():
    """When asked to keep top-k columns at fp, the residual must be non-zero there."""
    W = torch.randn(8, 64) * 0.05
    W[:, 3] = 5.0  # clear outlier column
    qw = rtn_residual_quantize(W, group_size=32, keep_top_k_outlier_cols=2)
    assert qw.residual is not None
    # Column 3 should land in residual; non-outlier columns should be zero.
    res = qw.residual.to(torch.float32)
    assert res[:, 3].abs().mean().item() > 0.1
    assert res[:, 0].abs().mean().item() < 1e-6


def test_gptq_runs_and_is_no_worse_than_rtn_on_average():
    """GPTQ should not be *worse* than RTN on a Hessian that matches the data.

    We don't assert "strictly better" because for random toy data RTN is often
    nearly optimal; we just check GPTQ doesn't blow up.
    """
    torch.manual_seed(0)
    K, N = 256, 64
    W = torch.randn(N, K) * 0.05
    X = torch.randn(1024, K)
    H = X.T @ X
    qw_gptq = gptq_quantize(W, H, group_size=64, block_size=64)
    qw_rtn = rtn_quantize(W, group_size=64)

    W_gptq = dequantize(qw_gptq)
    W_rtn = dequantize(qw_rtn)

    # Use the Hessian-weighted error since that's GPTQ's objective.
    def hwerr(W_q):
        d = (W - W_q) @ H @ (W - W_q).T
        return torch.diag(d).mean().item()

    assert hwerr(W_gptq) <= hwerr(W_rtn) * 1.5  # generous margin for randomness


def test_weight_bits_16_skips_quantization():
    W = torch.randn(32, 64) * 0.05
    qw = quantize_weight(W, "gptq", group_size=64, weight_bits=16, hessian=torch.eye(64))
    assert is_no_quant(qw.weight_bits)
    assert torch.allclose(dequantize(qw), W, atol=1e-6)


def test_no_quant_layer_forward_matches_fp_linear(tmp_path):
    K, N = 128, 64
    W = torch.randn(N, K) * 0.05
    W_store = W.to(torch.bfloat16)
    qw = no_quantize(W_store, weight_bits=16)
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
        rotation=identity_rotation(K, 64),
        act_bits=16,
        act_scale_mode="dynamic",
        act_scale_table=None,
        bias=None,
        residual=None,
        extras={"fp_weight": W_store},
    )
    layer = QuantLinear(
        layer_pack,
        return_tuple=False,
        skip_bias_add=False,
        output_dtype=torch.float32,
        device="cpu",
    )
    x = torch.randn(4, K)
    y_ref = x @ W_store.to(torch.float32).T
    y_q = layer(x)
    assert torch.allclose(y_q, y_ref, rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------- #
# Pack round-trip                                                             #
# --------------------------------------------------------------------------- #


def test_pack_save_load_roundtrip(tmp_path):
    cfg = QVLAConfig.pi05_default()
    cfg = cfg.with_overrides(dit=replace(cfg.dit, num_steps=3))
    K, N = 128, 64
    W = torch.randn(N, K) * 0.05
    qw = rtn_quantize(W, group_size=64)
    rot = identity_rotation(K, 64)
    layer = LayerPack(
        name="foo.bar.qkv_proj",
        scope="dit",
        in_features=K,
        out_features=N,
        bias_present=False,
        qweight=qw.qweight,
        weight_scale=qw.weight_scale,
        group_size=qw.group_size,
        weight_bits=qw.weight_bits,
        rotation=rot,
        act_bits=4,
        act_scale_mode="per_step",
        act_scale_table=torch.ones(3, K) * 0.01,
        bias=None,
        residual=None,
    )
    pack = Pack(config=cfg, layers={"foo.bar.qkv_proj": layer})
    path = tmp_path / "pack.pt"
    pack.save(path)

    loaded = Pack.load(path)
    assert set(loaded.layers) == {"foo.bar.qkv_proj"}
    lp2 = loaded.layers["foo.bar.qkv_proj"]
    assert torch.equal(lp2.qweight, qw.qweight)
    assert torch.allclose(lp2.weight_scale, qw.weight_scale)
    assert lp2.act_scale_table.shape == (3, K)


# --------------------------------------------------------------------------- #
# Wrap / replace                                                              #
# --------------------------------------------------------------------------- #


class _TinyModel(nn.Module):
    """Minimal model that mimics phyai PI05Model module naming."""

    def __init__(self):
        super().__init__()
        self.paligemma_lm = nn.Module()
        self.paligemma_lm.layers = nn.ModuleList(
            [self._make_layer() for _ in range(2)]
        )
        self.expert_stack = nn.Module()
        self.expert_stack.layers = nn.ModuleList(
            [self._make_layer() for _ in range(2)]
        )

    @staticmethod
    def _make_layer():
        b = nn.Module()
        b.qkv_proj = nn.Linear(128, 128, bias=False)
        b.o_proj = nn.Linear(128, 128, bias=False)
        b.mlp = nn.Module()
        b.mlp.gate_up_proj = nn.Linear(128, 256, bias=False)
        b.mlp.down_proj = nn.Linear(256, 128, bias=False)
        return b


def test_classify_pi05_regexes():
    cfg = QVLAConfig.pi05_default()
    llm_name = "paligemma_lm.layers.0.qkv_proj"
    dit_name = "expert_stack.layers.0.mlp.down_proj"
    assert classify(llm_name, cfg) == "llm"
    assert classify(dit_name, cfg) == "dit"
    assert classify("paligemma_lm.lm_head", cfg) is None


def test_list_target_modules_partitions_correctly():
    model = _TinyModel()
    cfg = QVLAConfig.pi05_default()
    targets = list_target_modules(model, cfg)
    llm_n = sum(1 for _, s, _ in targets if s == "llm")
    dit_n = sum(1 for _, s, _ in targets if s == "dit")
    assert llm_n == 2 * 4  # 2 layers × 4 linears per block
    assert dit_n == 2 * 4


def _fake_pack_for_module(model, cfg) -> Pack:
    """Build a pack matching the _TinyModel: rotation=identity, RTN, dynamic act."""
    layers: dict[str, LayerPack] = {}
    for name, scope, mod in list_target_modules(model, cfg):
        W = mod.weight.detach()
        N, K = W.shape
        qw = rtn_quantize(W, group_size=64)
        layers[name] = LayerPack(
            name=name, scope=scope,
            in_features=K, out_features=N, bias_present=False,
            qweight=qw.qweight, weight_scale=qw.weight_scale,
            group_size=qw.group_size, weight_bits=qw.weight_bits,
            rotation=identity_rotation(K, 64),
            act_bits=4, act_scale_mode="dynamic",
            act_scale_table=None,
            bias=None, residual=None,
        )
    return Pack(config=cfg, layers=layers)


def test_enable_quantization_replaces_and_forwards():
    model = _TinyModel()
    cfg = QVLAConfig.pi05_default()
    pack = _fake_pack_for_module(model, cfg)

    # Reference forward on a single block before swap.
    block = model.expert_stack.layers[0]
    x = torch.randn(2, 128)
    y_fp_q = block.qkv_proj(x)

    replaced = enable_quantization(
        model, pack, device="cpu", output_dtype=torch.float32,
    )
    assert len(replaced) == 2 * 2 * 4  # full coverage

    # The replaced layer should still be callable with the same shape.
    new_block = model.expert_stack.layers[0]
    y_q = new_block.qkv_proj(x)
    if isinstance(y_q, tuple):
        y_q = y_q[0]
    assert y_q.shape == y_fp_q.shape
    # int4 quant error should still preserve coarse direction.
    rel = (y_q - y_fp_q).norm() / y_fp_q.norm().clamp_min(1e-6)
    assert rel < 0.5, f"replacement forward diverged: rel error {rel:.3f}"


# --------------------------------------------------------------------------- #
# Rotated activation amax collector                                           #
# --------------------------------------------------------------------------- #


def test_act_channel_percentile_amax_is_inner_not_cross():
    """Act scales use each channel's own value distribution, not a global cap."""
    from qvla.build.collector import LayerStats
    from qvla.core.quantize import (
        channel_percentile_amax,
        percentile_amax,
    )

    # Channel 0: mostly small with one huge outlier; channel 1: uniformly moderate.
    x = torch.tensor(
        [
            [0.1, 1.0],
            [0.2, 1.1],
            [0.1, 0.9],
            [10.0, 1.0],
        ],
        dtype=torch.float32,
    )
    stats = LayerStats(in_features=2)
    stats.static_abs_samples = [x.abs()]

    per_ch = stats.act_channel_percentile_amax(99.9)
    expected = channel_percentile_amax(x.abs(), 99.9)
    assert torch.allclose(per_ch, expected, rtol=1e-5, atol=1e-5)

    # Legacy cross cap would clamp both channels to the same global q.
    legacy = percentile_amax(x.abs().amax(dim=0), 99.9)
    assert not torch.allclose(per_ch, legacy)


def test_rotated_act_scale_collector_matches_rotation_apply():
    """RotatedActivationCollector stats must match manual rotation.apply amax."""
    K = 64
    lin = nn.Linear(K, 16, bias=False)
    x = torch.randn(8, K)
    rot = fit_rotation(
        d=K,
        block_size=32,
        weight=lin.weight.detach(),
        pipeline=("perm", "svd", "hadamard"),
        perm_score="weight",
        device="cpu",
    )
    manual = rot.apply(x).abs().amax(dim=0)

    captured: dict[str, torch.Tensor] = {}

    def pre_hook(_m, inputs):
        captured["x"] = inputs[0]

    h = lin.register_forward_pre_hook(pre_hook)
    with RotatedActivationCollector(
        [("layer", "dit", lin)],
        {"layer": rot},
        num_steps_by_scope={"dit": 3},
        device="cpu",
    ) as col:
        col.set_current_step(1)
        lin(x)
    h.remove()

    collected = col.stats["layer"].per_step_channel_amax[1]
    assert torch.allclose(collected, manual, rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------- #
# Step counter                                                                #
# --------------------------------------------------------------------------- #


def test_per_step_counter_advances():
    """Calling the layer N times should index slots 0..N-1 % num_steps."""
    K, N = 128, 64
    W = torch.randn(N, K) * 0.05
    qw = rtn_quantize(W, group_size=64)
    table = torch.arange(3 * K, dtype=torch.float32).reshape(3, K) * 0.001 + 0.01
    layer_pack = LayerPack(
        name="dit.layer", scope="dit",
        in_features=K, out_features=N, bias_present=False,
        qweight=qw.qweight, weight_scale=qw.weight_scale,
        group_size=qw.group_size, weight_bits=qw.weight_bits,
        rotation=identity_rotation(K, 64),
        act_bits=4, act_scale_mode="per_step",
        act_scale_table=table, bias=None, residual=None,
    )
    layer = QuantLinear(
        layer_pack, return_tuple=False, skip_bias_add=False,
        output_dtype=torch.float32, device="cpu",
    )
    x = torch.randn(1, K)
    for expected_step in (0, 1, 2, 0, 1):
        # Capture the pre-call counter; the layer reads it then increments.
        pre = layer._step_counter % layer._num_steps
        assert pre == expected_step
        _ = layer(x)


def test_static_per_token_uses_offline_table():
    """per_token static reads per-token scales from act_scale_table."""
    K, N = 64, 32
    W = torch.randn(N, K) * 0.05
    qw = rtn_quantize(W, group_size=64)
    table = torch.tensor([0.5, 0.1], dtype=torch.float32)
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
        rotation=identity_rotation(K, 64),
        act_bits=4,
        act_scale_mode="static",
        act_scale_granularity="per_token",
        act_scale_table=table,
        bias=None,
        residual=None,
    )
    layer = QuantLinear(
        layer_pack,
        return_tuple=False,
        skip_bias_add=False,
        output_dtype=torch.float32,
        device="cpu",
    )
    x = torch.randn(2, K)
    scale = layer._activation_scale(x)
    assert scale.shape == (2, 1)
    assert torch.allclose(scale, table.view(2, 1))


def test_per_token_scales_tile_across_batch():
    """Calibrated chunk_size scales tile when expert flattens B*chunk tokens."""
    K, N, chunk, batch = 64, 32, 50, 16
    W = torch.randn(N, K) * 0.05
    qw = rtn_quantize(W, group_size=64)
    table = torch.linspace(0.01, 0.5, chunk, dtype=torch.float32).unsqueeze(0).repeat(10, 1)
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
        rotation=identity_rotation(K, 64),
        act_bits=4,
        act_scale_mode="per_step",
        act_scale_granularity="per_token",
        act_scale_table=table,
        bias=None,
        residual=None,
    )
    layer = QuantLinear(
        layer_pack,
        return_tuple=False,
        skip_bias_add=False,
        output_dtype=torch.float32,
        device="cpu",
    )
    x = torch.randn(batch * chunk, K)
    scale = layer._activation_scale(x)
    assert scale.shape == (batch * chunk, 1)
    assert torch.allclose(scale, table[0].repeat(batch).view(-1, 1))
    _ = layer(x)  # full forward must not raise
