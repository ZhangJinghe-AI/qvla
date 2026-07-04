"""Calibration batch iterators for PI05."""

from __future__ import annotations

import logging
from typing import Iterator

import numpy as np

from qvla.adapters.pi05.config import PI05AdapterConfig


logger = logging.getLogger(__name__)

_TASK_POOL = (
    "pick up the object and place it in the basket",
    "open the drawer and put the cup inside",
    "move the red block to the left side of the table",
    "stack the small box on top of the large box",
    "close the door of the microwave gently",
)


def iter_calibration_batches(cfg: PI05AdapterConfig, num_samples: int, *, state_dim: int) -> Iterator[dict]:
    if cfg.calibration_source == "file":
        yield from _iter_file_batches(cfg, num_samples)
        return
    yield from _iter_synthetic_batches(num_samples, state_dim=state_dim, seed=cfg.seed)


def _iter_synthetic_batches(num_samples: int, *, state_dim: int, seed: int) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    for i in range(num_samples):
        yield {
            "images": {
                "image": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
                "wrist_image": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
            },
            "states": rng.standard_normal(state_dim).astype(np.float32),
            "task_description": _TASK_POOL[i % len(_TASK_POOL)],
        }


def _iter_file_batches(cfg: PI05AdapterConfig, num_samples: int) -> Iterator[dict]:
    from qvla.calibration.file import make_file_calibration_provider

    if cfg.calibration_data_path is None:
        raise ValueError("calibration_source='file' requires calibration_data_path.")
    provider = make_file_calibration_provider(cfg.calibration_data_path)
    meta = getattr(provider, "__calibration_meta__", {})
    logger.info(
        "PI05Adapter: file calibration (%s, source=%s, suite=%s).",
        cfg.calibration_data_path,
        meta.get("source", "unknown"),
        meta.get("suite", "n/a"),
    )
    yield from provider(num_samples)
