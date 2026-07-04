"""Observation dict → phyai PI05Request preprocessing."""

from __future__ import annotations

import numpy as np
import torch


def obs_to_cameras(obs: dict) -> list[torch.Tensor]:
    """Convert ``obs['images']`` values to NCHW float tensors in [0, 1]."""
    cameras: list[torch.Tensor] = []
    for img in obs.get("images", {}).values():
        arr = np.ascontiguousarray(img)
        cameras.append(
            torch.as_tensor(arr, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        )
    return cameras


def obs_to_transition(obs: dict, *, state_dim: int) -> dict:
    from phyai_utils_tools.processing.transition import IMAGES, STATE, TASK

    return {
        IMAGES: obs_to_cameras(obs),
        STATE: torch.as_tensor(np.asarray(obs["states"], dtype=np.float32)),
        TASK: str(obs.get("task_description", "")),
    }


def build_pi05_request(processor, obs: dict, *, state_dim: int):
    from phyai.models.pi05.scheduler_ws1_pi05 import PI05Request

    out = processor.preprocess(obs_to_transition(obs, state_dim=state_dim))
    return PI05Request(
        pixel_values=out.pixel_values,
        input_ids=out.input_ids,
        lang_lens=out.lang_lens,
    )
