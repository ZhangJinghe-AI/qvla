"""Tests for the DiT outlier causal-intervention visualization."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import visualize_dit_outlier_causal_intervention as viz  # noqa: E402


def _activation() -> torch.Tensor:
    scale = torch.tensor(
        [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 100.0, 10000.0]
    )
    x = scale.view(1, -1).expand(41, -1).clone() * 0.2
    x[0] = scale
    return x


def test_clip_and_random_plans_are_strictly_matched():
    x = _activation()
    selective = viz._clip_plan(x, 3.0)
    control = viz._random_plan(x, selective, seed=7)

    assert selective.mask[:, :8].sum().item() == 0
    assert selective.mask[:, 8:].sum().item() == 2
    assert control.mask.sum() == selective.mask.sum()
    torch.testing.assert_close(
        control.ratios.sort().values,
        selective.ratios.sort().values,
    )
    assert not bool((control.mask & selective.mask).any().item())

    clipped = viz._apply_plan(x, selective, "selective")
    random_changed = viz._apply_plan(x, control, "random")
    assert bool((clipped.abs() <= x.abs()).all().item())
    assert bool((random_changed.abs() <= x.abs()).all().item())
    assert int((clipped != x).sum()) == int(selective.mask.sum())
    assert int((random_changed != x).sum()) == int(control.mask.sum())


def test_binary_mask_expands_rectangular_prefix():
    mask = torch.tensor(
        [[[1, 1, 0], [1, 1, 0], [0, 0, 0], [0, 0, 0]]]
    )
    expanded = viz._binary_mask(mask, horizon=4, width=3)
    assert expanded.tolist() == [
        [True, True, False],
        [True, True, False],
        [False, False, False],
        [False, False, False],
    ]


def test_binary_mask_rejects_non_rectangular_mask():
    mask = torch.tensor([[[1, 1], [1, 0]]])
    with pytest.raises(RuntimeError, match="not a rectangular"):
        viz._binary_mask(mask, horizon=2, width=2)


def test_task_action_dim_uses_processor_decode_layout():
    action_config = SimpleNamespace(modality_keys=("arm", "gripper"))
    processor = SimpleNamespace(
        modality_config={"action": action_config},
        embodiment_tag="LIBERO",
        norm_params={
            "LIBERO": {
                "action": {"arm": {"dim": 6}, "gripper": {"dim": 1}}
            }
        },
    )
    assert viz._task_action_dim(processor, runtime_width=132) == 7


def test_action_error_uses_only_active_rectangle():
    baseline = torch.tensor([[[3.0, 4.0], [9.0, 9.0]]])
    changed = torch.tensor([[[0.0, 0.0], [100.0, 100.0]]])
    active = torch.tensor([[True, True], [False, False]])
    per_token, relative = viz._action_error(changed, baseline, active)
    torch.testing.assert_close(per_token, torch.tensor([5.0]))
    assert relative == 1.0


def test_plot_writes_png(tmp_path):
    result = viz.Result(
        clip_pct=torch.ones(2, 5),
        removed_l1_pct=torch.ones(2, 5) * 0.5,
        selective_action_error=torch.ones(2, 3) * 0.1,
        random_action_error=torch.ones(2, 3) * 0.2,
        selective_rel_l2=torch.tensor([0.01, 0.02]),
        random_rel_l2=torch.tensor([0.03, 0.04]),
        random_rel_l2_std=torch.tensor([0.001, 0.002]),
        outlier_counts=torch.tensor([4, 5]),
    )
    output = tmp_path / "causal.png"
    viz._plot(result, layer_name="action_head.model.block.8.ff.fc2", output=output)
    assert output.is_file()
    assert output.stat().st_size > 0
