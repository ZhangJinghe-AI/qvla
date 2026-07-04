"""Unit tests for tools/weight_row_distribution.py (CPU-only)."""

from __future__ import annotations

import pytest
import torch

from weight_row_distribution import (
    VARIANT_ORDER,
    _apply_block_matmul,
    _hadamard_blocks,
    apply_activation_pipeline,
    apply_weight_pipeline,
    compare_activation_variants,
    compare_weight_variants,
    plot_activation_distributions,
    plot_weight_distributions,
)


def _skewed_weight(out: int = 32, inp: int = 64) -> torch.Tensor:
    torch.manual_seed(0)
    w = torch.randn(out, inp) * 0.2
    w[:4] *= 10.0
    return w


def test_nine_variants_returned():
    w = _skewed_weight()
    dists = compare_weight_variants(w, block_size=16)
    assert set(dists.keys()) == set(VARIANT_ORDER)
    assert len(dists) == 9


def test_input_col_l2_changes_under_hadamard():
    w = _skewed_weight()
    dists = compare_weight_variants(w, block_size=16, metric="l2")
    assert not torch.allclose(dists["hadamard"].values, dists["original"].values, atol=1e-5)


def test_row_max_abs_changes_under_hadamard():
    w = _skewed_weight()
    dists = compare_weight_variants(w, block_size=16, metric="max_abs")
    assert dists["hadamard"].max_min_ratio < dists["original"].max_min_ratio


def test_perm_hadamard_changes_distribution():
    w = _skewed_weight()
    dists = compare_weight_variants(w, block_size=16, metric="max_abs")
    assert not torch.allclose(dists["original"].values, dists["perm_hadamard"].values, atol=1e-5)


def test_svd_hadamard_preserves_forward():
    w = _skewed_weight(out=16, inp=64)
    x = torch.randn(3, 64)
    y_ref = x @ w.T
    steps = ("perm", "svd", "hadamard")
    w_rot = apply_weight_pipeline(w, steps, block_size=16)
    x_rot = apply_activation_pipeline(x, w, steps, block_size=16)
    assert torch.allclose(x_rot @ w_rot.T, y_ref, atol=1e-4)


def test_random_hadamard_pipeline_differs_from_standard():
    w = _skewed_weight()
    std = apply_weight_pipeline(w, ("hadamard",), block_size=16)
    rnd = apply_weight_pipeline(
        w,
        ("random_hadamard",),
        block_size=16,
        layer_name="test.layer",
        build_seed=42,
    )
    assert not torch.allclose(std, rnd, atol=1e-5)


def test_plot_requires_matplotlib():
    w = _skewed_weight(out=8, inp=16)
    dists = compare_weight_variants(w, block_size=8)
    pytest.importorskip("matplotlib")
    fig = plot_weight_distributions(dists)
    assert fig is not None


def test_perm_after_rotation_uses_rotated_weight():
    """Perm following a rotation must not reuse the original W energy."""
    from qvla.core.rotation import compute_perm_energy, zigzag_permutation

    w = _skewed_weight()
    w_after_svd = apply_weight_pipeline(w, ("svd",), block_size=16)

    _, score = compute_perm_energy(w, perm_score="weight")
    assert score == "weight"
    perm_original = zigzag_permutation((w * w).mean(dim=0))
    perm_after_svd = zigzag_permutation((w_after_svd * w_after_svd).mean(dim=0))
    assert not torch.equal(perm_original, perm_after_svd)

    via_pipeline = apply_weight_pipeline(w, ("svd", "perm", "hadamard"), block_size=16)
    w_wrong = w_after_svd[:, perm_original]
    blocks = _hadamard_blocks(w_wrong.shape[1], 16, device=w_wrong.device)
    w_wrong = _apply_block_matmul(w_wrong, blocks, 16)
    assert not torch.allclose(via_pipeline, w_wrong, atol=1e-4)


def test_act_perm_after_svd_uses_rotated_activation_amax():
    """Activation perm after SVD must use amax on SVD-rotated activations."""
    from qvla.core.rotation import (
        compute_perm_energy,
        zigzag_permutation,
    )

    w = _skewed_weight()
    x = torch.randn(128, w.shape[1]) * 0.2
    x[:, :4] *= 6.0
    amax_raw = x.abs().amax(dim=0)

    w_after_svd = apply_weight_pipeline(w, ("svd",), block_size=16)
    x_after_svd = apply_activation_pipeline(x, w, ("svd",), block_size=16)
    amax_after_svd = x_after_svd.abs().amax(dim=0)

    perm_wrong = zigzag_permutation(
        compute_perm_energy(w_after_svd, perm_score="activation", activation_amax=amax_raw)[0]
    )
    perm_right = zigzag_permutation(
        compute_perm_energy(w_after_svd, perm_score="activation", activation_amax=amax_after_svd)[0]
    )
    assert not torch.equal(perm_wrong, perm_right)

    via_pipeline = apply_weight_pipeline(
        w,
        ("svd", "perm", "hadamard"),
        block_size=16,
        perm_score="activation",
        activation_tokens=x,
    )
    w_wrong = w_after_svd[:, perm_wrong]
    blocks_h = _hadamard_blocks(w_wrong.shape[1], 16, device=w_wrong.device)
    w_wrong = _apply_block_matmul(w_wrong, blocks_h, 16)
    assert not torch.allclose(via_pipeline, w_wrong, atol=1e-4)


def test_perm_svd_h_reduces_outliers():
    w = _skewed_weight()
    dists = compare_weight_variants(w, block_size=16, metric="max_abs")
    assert dists["perm_svd_h"].max_min_ratio < dists["original"].max_min_ratio
    assert dists["perm_svd_h"].std < dists["original"].std


def test_activation_svd_accepts_cpu_tokens_with_cuda_weight():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    w = _skewed_weight().cuda()
    x = torch.randn(64, w.shape[1])
    dists = compare_weight_variants(
        w,
        block_size=16,
        svd_source="activation",
        activation_tokens=x,
    )
    assert set(dists.keys()) == set(VARIANT_ORDER)


def test_activation_svd_requires_tokens():
    w = _skewed_weight()
    with pytest.raises(ValueError, match="activation_tokens"):
        compare_weight_variants(w, block_size=16, svd_source="activation")


def test_compare_activation_variants_nine():
    w = _skewed_weight()
    x = torch.randn(64, w.shape[1]) * 0.2
    dists = compare_activation_variants(x, w, block_size=16)
    assert set(dists.keys()) == set(VARIANT_ORDER)
    assert len(dists) == 9


def test_activation_pipeline_preserves_forward():
    from weight_row_distribution import VARIANT_PIPELINES

    w = _skewed_weight(out=16, inp=64)
    x = torch.randn(8, 64)
    y_ref = x @ w.T
    steps = VARIANT_PIPELINES["perm_svd_h"]
    w_rot = apply_weight_pipeline(w, steps, block_size=16)
    x_rot = apply_activation_pipeline(x, w, steps, block_size=16)
    assert torch.allclose(x_rot @ w_rot.T, y_ref, atol=1e-4)


def test_plot_activation_requires_matplotlib():
    w = _skewed_weight(out=8, inp=16)
    x = torch.randn(32, 16)
    dists = compare_activation_variants(x, w, block_size=8)
    pytest.importorskip("matplotlib")
    fig = plot_activation_distributions(dists)
    assert fig is not None
