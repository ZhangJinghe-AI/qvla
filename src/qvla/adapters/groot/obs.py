"""Observation dict → GR00T-N1.7 request preprocessing."""

from __future__ import annotations

import torch


def build_groot_request(processor, obs: dict | list[dict], *, device: str = "cuda"):
    """Build a GR00TN17Request from one observation dict or a list (batch).

    Each observation dict should have:
      - video: dict[str, np.ndarray] shaped (B, T, H, W, 3) uint8
      - state: dict[str, np.ndarray] shaped (B, T, D) float32
      - language: dict[str, list[list[str]]] shaped (B, T)
    """
    from phyai.models.gr00t_n17.scheduler_ws1_gr00t_n17 import GR00TN17Request
    from phyai_utils_tools.models.gr00t import GR00TObservation

    if isinstance(obs, dict):
        obs_list = [obs]
    else:
        obs_list = list(obs)

    if not obs_list:
        raise ValueError("build_groot_request() requires a non-empty observation list.")

    all_tensors: list[dict[str, torch.Tensor]] = []
    for item in obs_list:
        observation = GR00TObservation(
            video=item["video"],
            state=item["state"],
            language=item["language"],
        )
        prepared = processor.process_observation(observation)
        all_tensors.append(prepared.tensors)

    if len(all_tensors) == 1:
        tensors = {
            key: value.to(device=device) if isinstance(value, torch.Tensor) else value
            for key, value in all_tensors[0].items()
        }
        return GR00TN17Request(tensors=tensors)

    # Batch multiple observations
    batched: dict[str, torch.Tensor] = {}
    for key in all_tensors[0]:
        values = [t[key] for t in all_tensors]
        if isinstance(values[0], torch.Tensor):
            if values[0].dim() == 0:
                batched[key] = torch.stack(values).to(device=device)
            else:
                batched[key] = torch.cat(values, dim=0).to(device=device)
        else:
            batched[key] = values[0]
    return GR00TN17Request(tensors=batched)
