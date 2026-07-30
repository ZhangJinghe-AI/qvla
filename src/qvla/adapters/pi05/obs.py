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


def build_pi05_request(processor, obs: dict | list[dict], *, state_dim: int):
    """Build a ``PI05Request`` from one observation or a list (batch dim 0)."""
    from phyai.models.pi05.scheduler_ws1_pi05 import PI05Request

    if isinstance(obs, dict):
        out = processor.preprocess(obs_to_transition(obs, state_dim=state_dim))
        return PI05Request(
            pixel_values=out.pixel_values,
            input_ids=out.input_ids,
            lang_lens=out.lang_lens,
        )

    if not obs:
        raise ValueError("build_pi05_request() requires a non-empty observation list.")
    processed = [
        processor.preprocess(obs_to_transition(item, state_dim=state_dim)) for item in obs
    ]
    return PI05Request(
        pixel_values=torch.cat([p.pixel_values for p in processed], dim=0),
        input_ids=torch.cat([p.input_ids for p in processed], dim=0),
        lang_lens=torch.cat([p.lang_lens for p in processed], dim=0),
    )
