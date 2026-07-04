"""PI05 adapter configuration and shared dtype helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

CalibrationSource = Literal["synthetic", "file"]

DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def resolve_dtype(name: str) -> torch.dtype:
    try:
        return DTYPE_MAP[name]
    except KeyError as exc:
        raise ValueError(f"Unknown dtype {name!r}; expected one of {sorted(DTYPE_MAP)}.") from exc


def read_checkpoint_config(checkpoint_path: Path) -> dict:
    json_path = checkpoint_path / "config.json"
    if not json_path.is_file():
        raise FileNotFoundError(
            f"PI05 checkpoint config not found: {json_path}. "
            "Expected a HuggingFace-style checkpoint directory."
        )
    with json_path.open(encoding="utf-8") as f:
        return json.load(f)


def lerobot_weight_remap(key: str) -> str | None:
    """Drop duplicate embed_tokens / lm_head keys (lerobot checkpoint layout)."""
    if key.startswith("model."):
        key = key[len("model.") :]
    if "embed_tokens" in key or "gemma_expert.lm_head" in key:
        return None
    return key


@dataclass
class PI05AdapterConfig:
    checkpoint_path: Path
    device: str = "cuda"
    params_dtype: str = "bfloat16"
    vision_params_dtype: str = "float16"
    attn_backend: str = "flashinfer"
    norm_backend: str = "phyai-kernel"
    calibration_source: CalibrationSource = "synthetic"
    calibration_data_path: Path | None = None
    seed: int = 0
    image_size: int = 224
    num_real_cameras: int = 2
    action_dim: int = 7
    state_dim: int = 8
    tokenizer_max_length: int = 200
    tokenizer_name: str = "google/paligemma-3b-pt-224"

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, **kwargs) -> PI05AdapterConfig:
        """Build config from a checkpoint dir and its ``config.json``."""
        checkpoint_path = Path(checkpoint_path)
        if kwargs.get("calibration_data_path") is not None:
            kwargs["calibration_data_path"] = Path(kwargs["calibration_data_path"])
        raw = read_checkpoint_config(checkpoint_path)
        empty_cameras = int(raw.get("empty_cameras", 1))
        return cls(
            checkpoint_path=checkpoint_path,
            image_size=int(raw.get("image_resolution", [224, 224])[0]),
            num_real_cameras=max(1, 3 - empty_cameras),
            action_dim=int(
                raw.get("output_features", {}).get("action", {}).get("shape", [7])[0]
            ),
            state_dim=int(
                raw.get("input_features", {})
                .get("observation.state", {})
                .get("shape", [8])[0]
            ),
            tokenizer_max_length=int(raw.get("tokenizer_max_length", 200)),
            tokenizer_name=str(raw.get("tokenizer_name", "google/paligemma-3b-pt-224")),
            **kwargs,
        )
