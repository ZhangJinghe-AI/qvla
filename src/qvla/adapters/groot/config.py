"""GR00T-N1.7 adapter configuration and shared helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

CalibrationSource = Literal["synthetic", "file"]

DEFAULT_PROCESSOR_MODEL_NAME_OR_PATH = "/data/share/Cosmos-Reason2-2B"

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
            f"GR00T-N1.7 checkpoint config not found: {json_path}. "
            "Expected a checkpoint directory with config.json."
        )
    with json_path.open(encoding="utf-8") as f:
        return json.load(f)


@dataclass
class GR00TAdapterConfig:
    checkpoint_path: Path
    device: str = "cuda"
    params_dtype: str = "bfloat16"
    calibration_source: CalibrationSource = "synthetic"
    calibration_data_path: Path | None = None
    seed: int = 0
    embodiment_tag: str = "LIBERO_PANDA"
    processor_model_name_or_path: str = DEFAULT_PROCESSOR_MODEL_NAME_OR_PATH
    image_size: int = 256
    max_batch_size: int = 1
    action_dim: int = 7

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, **kwargs) -> GR00TAdapterConfig:
        """Build config from a checkpoint directory."""
        checkpoint_path = Path(checkpoint_path)
        if kwargs.get("calibration_data_path") is not None:
            kwargs["calibration_data_path"] = Path(kwargs["calibration_data_path"])
        raw = read_checkpoint_config(checkpoint_path)
        embodiment_tag = kwargs.pop("embodiment_tag", None)
        if embodiment_tag is None:
            embodiment_tag = raw.get("embodiment_tag", "LIBERO_PANDA")
        action_dim = kwargs.pop("action_dim", None)
        if action_dim is None:
            action_head_cfg = raw.get("action_head", {})
            action_dim = int(action_head_cfg.get("max_action_dim", 132))
        processor_model_name_or_path = (
            kwargs.pop("processor_model_name_or_path", None)
            or DEFAULT_PROCESSOR_MODEL_NAME_OR_PATH
        )
        return cls(
            checkpoint_path=checkpoint_path,
            embodiment_tag=embodiment_tag,
            action_dim=int(action_dim),
            processor_model_name_or_path=str(processor_model_name_or_path),
            **kwargs,
        )
