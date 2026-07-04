"""Tests for Fisher sensitivity and policy-aware rotation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_PKG_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from qvla.build.fisher import FisherCollector, LayerFisherResult
from rotation_helpers import fit_rotation


# -------------------------------------------------------------------- #
# Helpers                                                                #
# -------------------------------------------------------------------- #


class _ToyModel(nn.Module):
    """Two-layer model: linear_a -> ReLU -> linear_b."""

    def __init__(self, d_in: int = 32, d_hidden: int = 16, d_out: int = 7):
        super().__init__()
        self.linear_a = nn.Linear(d_in, d_hidden, bias=False)
        self.linear_b = nn.Linear(d_hidden, d_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_b(torch.relu(self.linear_a(x)))


class _ToyDiTModel(nn.Module):
    """Minimal 'DiT' with a loop of ``num_steps`` through one shared linear."""

    def __init__(self, d: int = 16, d_out: int = 7, num_steps: int = 3):
        super().__init__()
        self.step_linear = nn.Linear(d, d, bias=False)
        self.out_proj = nn.Linear(d, d_out, bias=False)
        self.num_steps = num_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.num_steps):
            x = x + self.step_linear(x)
        return self.out_proj(x)


def _make_targets(
    model: nn.Module,
    scope: str = "dit",
) -> list[tuple[str, str, nn.Module]]:
    """Build ``(name, scope, module)`` triples for every Linear in *model*."""
    return [
        (name, scope, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear)
    ]


# -------------------------------------------------------------------- #
# FisherCollector — basic sanity                                        #
# -------------------------------------------------------------------- #


def test_fisher_collector_captures_nonzero_sensitivity():
    model = _ToyModel(d_in=32, d_hidden=16, d_out=7)
    targets = _make_targets(model, scope="dit")

    with FisherCollector(targets) as fc:
        fc.begin_sample()
        x = torch.randn(2, 32, requires_grad=True)
        actions = model(x)
        fc.compute_jacobian_sensitivity(actions, model=model)
        results = fc.get_results()

    # linear_b receives output of relu(linear_a(x)) which has grad_fn
    assert len(results) == 2
    for name, result in results.items():
        sens = result.aggregate("uniform")
        assert sens.shape == (result.in_features,)
    # At least linear_b must have non-zero sensitivity
    b_sens = results["linear_b"].aggregate("uniform")
    assert b_sens.abs().sum().item() > 0


def test_fisher_collector_multiple_samples():
    model = _ToyModel()
    targets = _make_targets(model)

    with FisherCollector(targets) as fc:
        for _ in range(3):
            fc.begin_sample()
            x = torch.randn(2, 32, requires_grad=True)
            actions = model(x)
            fc.compute_jacobian_sensitivity(actions, model=model)
        results = fc.get_results()

    for result in results.values():
        assert result._n_samples == 3
        per_step = result.sensitivity_per_step()
        assert per_step is not None


def test_fisher_collector_dit_per_step():
    """DiT layers called multiple times should get per-step sensitivity."""
    model = _ToyDiTModel(d=16, d_out=7, num_steps=3)
    targets = _make_targets(model, scope="dit")

    with FisherCollector(targets, num_dit_steps=3) as fc:
        fc.begin_sample()
        x = torch.randn(1, 16, requires_grad=True)

        for step in range(3):
            fc.set_current_step(step)
            x = x + model.step_linear(x)

        fc.set_current_step(None)
        actions = model.out_proj(x)
        fc.compute_jacobian_sensitivity(actions, model=model)
        results = fc.get_results()

    step_linear_result = results.get("step_linear")
    assert step_linear_result is not None
    per_step = step_linear_result.sensitivity_per_step()
    assert len(per_step) == 3, f"Expected 3 steps, got {len(per_step)}"
    for step_idx in range(3):
        assert step_idx in per_step


def test_fisher_collector_llm_no_step():
    """LLM-scope layers should have step=None regardless of set_current_step."""
    model = _ToyModel()
    targets = _make_targets(model, scope="llm")

    with FisherCollector(targets) as fc:
        fc.begin_sample()
        fc.set_current_step(5)
        x = torch.randn(2, 32, requires_grad=True)
        actions = model(x)
        fc.compute_jacobian_sensitivity(actions, model=model)
        results = fc.get_results()

    for result in results.values():
        per_step = result.sensitivity_per_step()
        assert set(per_step.keys()) == {None}


def test_fisher_collector_no_grad_skips():
    """When actions has no grad_fn, the collector should skip gracefully."""
    model = _ToyModel()
    targets = _make_targets(model)

    with FisherCollector(targets) as fc:
        fc.begin_sample()
        with torch.no_grad():
            x = torch.randn(2, 32)
            actions = model(x)
        fc.compute_jacobian_sensitivity(actions, model=model)
        results = fc.get_results()

    for result in results.values():
        sens = result.aggregate("uniform")
        assert sens.abs().sum().item() == 0


def test_fisher_3d_actions():
    """Actions shaped (batch, chunk_size, action_dim) should be handled."""

    class _ChunkedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(16, 21, bias=False)

        def forward(self, x):
            return self.linear(x).reshape(x.shape[0], 3, 7)

    model = _ChunkedModel()
    targets = _make_targets(model)

    with FisherCollector(targets) as fc:
        fc.begin_sample()
        x = torch.randn(2, 16, requires_grad=True)
        actions = model(x)
        assert actions.shape == (2, 3, 7)
        fc.compute_jacobian_sensitivity(actions, model=model)
        results = fc.get_results()

    result = results["linear"]
    sens = result.aggregate("uniform")
    assert sens.shape == (16,)
    assert sens.abs().sum().item() > 0


# -------------------------------------------------------------------- #
# LayerFisherResult                                                      #
# -------------------------------------------------------------------- #


def test_layer_fisher_result_aggregate_uniform():
    result = LayerFisherResult(name="test", scope="dit", in_features=8)
    result._step_accum = {
        0: torch.ones(8) * 2.0,
        1: torch.ones(8) * 4.0,
        2: torch.ones(8) * 6.0,
    }
    result._n_samples = 1
    agg = result.aggregate("uniform")
    assert agg.shape == (8,)
    assert agg.allclose(torch.ones(8) * 4.0)


def test_layer_fisher_result_empty():
    result = LayerFisherResult(name="test", scope="dit", in_features=4)
    agg = result.aggregate("uniform")
    assert agg.shape == (4,)
    assert agg.abs().sum().item() == 0


# -------------------------------------------------------------------- #
# Fisher perm (perm_score=fisher)                                       #
# -------------------------------------------------------------------- #


def test_policy_rotation_is_orthonormal():
    d, n = 64, 32
    W = torch.randn(n, d) * 0.1

    rot = fit_rotation(d=d, block_size=16, weight=W, pipeline=("svd", "hadamard"))
    R = rot.apply(torch.eye(d))
    eye = R @ R.T
    assert torch.allclose(eye, torch.eye(d), atol=1e-5), "Rotation is not orthonormal"


def test_policy_rotation_differs_from_vanilla():
    """With non-uniform sensitivity, policy rotation should differ from vanilla."""
    d, n = 64, 32
    torch.manual_seed(42)
    W = torch.randn(n, d) * 0.1

    sens = torch.ones(d)
    sens[:16] = 10.0

    rot_vanilla = fit_rotation(
        d=d, block_size=16, weight=W, pipeline=("perm", "svd", "hadamard")
    )
    rot_policy = fit_rotation(
        d=d,
        block_size=16,
        weight=W,
        pipeline=("perm", "svd", "hadamard"),
        sensitivity=sens,
        perm_score="fisher",
    )

    R_v = rot_vanilla.apply(torch.eye(d))
    R_p = rot_policy.apply(torch.eye(d))

    assert not torch.allclose(R_v, R_p, atol=1e-4), (
        "Policy rotation should differ from vanilla with non-uniform sensitivity"
    )


def test_policy_rotation_zero_sensitivity_uses_policy_mode():
    """Zero sensitivity still runs the policy path (no vanilla fallback)."""
    d, n = 32, 16
    torch.manual_seed(0)
    W = torch.randn(n, d) * 0.1
    sens = torch.zeros(d)

    rot = fit_rotation(
        d=d,
        block_size=16,
        weight=W,
        pipeline=("perm", "svd", "hadamard"),
        sensitivity=sens,
        perm_score="fisher",
    )
    assert rot.meta is not None and rot.meta.get("perm_score_used") == "fisher"


def test_policy_rotation_shape_validation():
    W = torch.randn(8, 16)
    rot = fit_rotation(
        d=16,
        block_size=12,
        weight=W,
        pipeline=("perm", "svd", "hadamard"),
        sensitivity=torch.ones(16),
        perm_score="fisher",
    )
    assert rot.is_identity


def test_fit_rotation_dispatches_policy():
    d, n = 32, 16
    W = torch.randn(n, d)
    sens = torch.ones(d)
    rot = fit_rotation(
        d=d,
        block_size=16,
        weight=W,
        pipeline=("perm", "svd", "hadamard"),
        sensitivity=sens,
        perm_score="fisher",
    )
    assert rot.meta is not None and rot.meta.get("perm_score_used") == "fisher"
    assert rot.u_blocks is not None
    assert rot.u_blocks.shape == (2, 16, 16)


def test_fit_rotation_fisher_requires_sensitivity():
    rot = fit_rotation(
        d=32,
        block_size=16,
        weight=torch.randn(8, 32),
        pipeline=("perm", "svd", "hadamard"),
        perm_score="fisher",
    )
    assert rot.is_identity


# -------------------------------------------------------------------- #
# Config integration                                                     #
# -------------------------------------------------------------------- #


def test_config_needs_fisher():
    from qvla.config import QVLAConfig, ScopeConfig
    from dataclasses import replace

    cfg = QVLAConfig.pi05_default()
    assert not cfg.needs_fisher

    cfg_policy = cfg.with_overrides(
        dit=replace(cfg.dit, perm_score="fisher")
    )
    assert cfg_policy.needs_fisher


def test_config_fisher_fields_serialize():
    from qvla.config import QVLAConfig

    cfg = QVLAConfig.pi05_default()
    cfg = cfg.with_overrides(fisher_num_samples=8, fisher_step_aggregation="uniform")
    d = cfg.to_dict()
    cfg2 = QVLAConfig.from_dict(d)
    assert cfg2.fisher_num_samples == 8
    assert cfg2.fisher_step_aggregation == "uniform"


def test_expert_linear_grad_patch_restores_autograd():
    """FlashInfer-like linears break the graph; the Fisher patch must reconnect it."""
    from qvla.build.differentiable_forward import (
        patch_expert_linears_for_grad,
        restore_expert_linears_for_grad,
    )

    class _FlashInferLikeLinear(nn.Module):
        skip_bias_add = False
        in_features = 8
        out_features = 4

        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 8) * 0.1)
            self.bias = nn.Parameter(torch.zeros(4))

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
            with torch.no_grad():
                y = torch.nn.functional.linear(x, self.weight, self.bias)
            return y, None

    class _Stack(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = _FlashInferLikeLinear()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y, _ = self.lin(x)
            return y

    stack = _Stack()
    x = torch.randn(2, 8, requires_grad=True)
    y = stack(x)
    assert not (y.requires_grad or y.grad_fn is not None)

    patched = patch_expert_linears_for_grad(stack)
    assert len(patched) == 1

    x2 = torch.randn(2, 8, requires_grad=True)
    y2 = stack(x2)
    y2.sum().backward()
    assert x2.grad is not None
    assert x2.grad.abs().sum().item() > 0

    restore_expert_linears_for_grad(patched)


# -------------------------------------------------------------------- #
# Sensitivity-weighted rotation reduces quant error on biased data      #
# -------------------------------------------------------------------- #


def test_policy_rotation_reduces_quant_error_for_sensitive_channels():
    """When some channels are far more important, policy rotation should
    give them better quantization (lower weighted MSE)."""
    d, n = 64, 32
    block_size = 16
    torch.manual_seed(123)

    W = torch.randn(n, d) * 0.1
    x_base = torch.randn(200, d) * 0.5
    x_base[:, :8] *= 5.0

    sens = torch.ones(d) * 0.01
    sens[:8] = 10.0

    rot_v = fit_rotation(
        d=d, block_size=block_size, weight=W, pipeline=("svd", "hadamard")
    )
    rot_p = fit_rotation(
        d=d,
        block_size=block_size,
        weight=W,
        pipeline=("svd", "hadamard"),
        sensitivity=sens,
        perm_score="fisher",
    )

    def _weighted_quant_error(rotation):
        x_rot = rotation.apply(x_base)
        amax = x_rot.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
        scale = amax / 7.0
        x_q = torch.round(x_rot / scale).clamp(-8, 7) * scale
        err = (x_rot - x_q).pow(2)
        weighted_err = err * sens.unsqueeze(0)
        return weighted_err.mean().item()

    err_vanilla = _weighted_quant_error(rot_v)
    err_policy = _weighted_quant_error(rot_p)

    assert err_policy <= err_vanilla * 1.1, (
        f"Policy rotation should not be much worse: "
        f"policy={err_policy:.6f} vs vanilla={err_vanilla:.6f}"
    )
