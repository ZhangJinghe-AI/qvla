"""Tests for GR00T calibration file → obs conversion."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from qvla.adapters.groot.calibration import (
    file_batch_to_groot_obs,
    groot_outlier_token_keep_mask,
)


def _libero_processor_stub():
    video = SimpleNamespace(modality_keys=["image", "wrist_image"], delta_indices=[0])
    state = SimpleNamespace(
        modality_keys=["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        delta_indices=[0],
    )
    language = SimpleNamespace(
        modality_keys=["annotation.human.action.task_description"],
        delta_indices=[0],
    )
    dims = {"x": 1, "y": 1, "z": 1, "roll": 1, "pitch": 1, "yaw": 1, "gripper": 2}
    norm_state = {k: {"dim": d} for k, d in dims.items()}
    return SimpleNamespace(
        embodiment_tag="libero_sim",
        modality_config={"video": video, "state": state, "language": language},
        norm_params={"libero_sim": {"state": norm_state}},
    )


def test_file_batch_to_groot_obs_libero_shape():
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    batch = {
        "images": {"image": img, "wrist_image": img + 1},
        "states": np.arange(8, dtype=np.float32),
        "task_description": "pick up the bowl",
    }
    obs = file_batch_to_groot_obs(batch, _libero_processor_stub())
    assert set(obs) == {"video", "state", "language"}
    assert obs["video"]["image"].shape == (1, 1, 256, 256, 3)
    assert obs["video"]["wrist_image"].shape == (1, 1, 256, 256, 3)
    assert obs["state"]["x"].shape == (1, 1, 1)
    assert obs["state"]["gripper"].shape == (1, 1, 2)
    np.testing.assert_array_equal(obs["state"]["gripper"][0, 0], [6.0, 7.0])
    assert obs["language"]["annotation.human.action.task_description"] == [
        ["pick up the bowl"]
    ]


def test_file_batch_to_groot_obs_passthrough():
    ready = {"video": {"image": 1}, "state": {"x": 2}, "language": {"t": [["a"]]}}
    assert file_batch_to_groot_obs(ready, _libero_processor_stub()) is ready


def test_groot_outlier_token_keep_mask_image_and_pad():
    # left pad | img img | task | img | task | pad
    ids = torch.tensor([0, 10, 10, 5, 10, 6, 0])
    attn = torch.tensor([0, 1, 1, 1, 1, 1, 0])
    image = groot_outlier_token_keep_mask(
        ids, attn, token_scope="image", image_token_id=10
    )
    both = groot_outlier_token_keep_mask(
        ids, attn, token_scope="image_lang_pad", image_token_id=10
    )
    all_keep = groot_outlier_token_keep_mask(
        ids, attn, token_scope="all", image_token_id=10
    )
    assert image.tolist() == [False, True, True, False, True, False, False]
    assert both.tolist() == [True, True, True, False, True, False, True]
    assert all_keep.tolist() == [True] * 7


def test_groot_outlier_token_keep_mask_rejects_bad_scope():
    with pytest.raises(ValueError, match="token_scope"):
        groot_outlier_token_keep_mask(
            torch.tensor([1]),
            torch.tensor([1]),
            token_scope="task",
            image_token_id=10,
        )
