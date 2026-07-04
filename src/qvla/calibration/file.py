"""Offline calibration datasets — export in Docker, load during pack build.

File format (``.npz``, version 1)::

    images          uint8  (N, H, W, 3)
    wrist_images    uint8  (N, H, W, 3)
    states          float32 (N, D)
    task_descriptions  object/str array length N
    meta_json       str — provenance (suite, seed, source, …)
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

CALIBRATION_FILE_VERSION = 1


def batches_to_arrays(batches: list[dict]) -> dict[str, Any]:
    """Stack PI05Adapter-style calibration batches for ``np.savez_compressed``."""
    if not batches:
        raise ValueError("Cannot save an empty calibration dataset.")
    images = []
    wrists = []
    states = []
    tasks: list[str] = []
    for batch in batches:
        imgs = batch.get("images", {})
        images.append(np.asarray(imgs["image"], dtype=np.uint8))
        wrists.append(np.asarray(imgs["wrist_image"], dtype=np.uint8))
        states.append(np.asarray(batch["states"], dtype=np.float32))
        tasks.append(str(batch.get("task_description", "")))
    return {
        "images": np.stack(images, axis=0),
        "wrist_images": np.stack(wrists, axis=0),
        "states": np.stack(states, axis=0),
        "task_descriptions": np.asarray(tasks, dtype=object),
    }


def arrays_to_batches(arrays: dict[str, Any]) -> list[dict]:
    """Expand stacked arrays back into PI05Adapter calibration batches."""
    n = int(arrays["images"].shape[0])
    tasks = arrays["task_descriptions"]
    batches: list[dict] = []
    for i in range(n):
        batches.append(
            {
                "images": {
                    "image": np.asarray(arrays["images"][i], dtype=np.uint8),
                    "wrist_image": np.asarray(arrays["wrist_images"][i], dtype=np.uint8),
                },
                "states": np.asarray(arrays["states"][i], dtype=np.float32),
                "task_description": str(tasks[i]),
            }
        )
    return batches


def save_calibration_dataset(
    path: str | Path,
    batches: list[dict],
    *,
    meta: dict[str, Any] | None = None,
) -> None:
    """Write a version-1 calibration ``.npz`` file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = batches_to_arrays(batches)
    header = {"version": CALIBRATION_FILE_VERSION, "num_samples": len(batches)}
    if meta:
        header.update(meta)
    np.savez_compressed(path, **arrays, meta_json=json.dumps(header))
    logger.info("Wrote %d calibration samples to %s", len(batches), path)


def load_calibration_dataset(path: str | Path) -> tuple[list[dict], dict[str, Any]]:
    """Load a calibration ``.npz`` and return ``(batches, meta)``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Calibration dataset not found: {path}")
    with np.load(path, allow_pickle=True) as data:
        required = ("images", "wrist_images", "states", "task_descriptions", "meta_json")
        missing = [k for k in required if k not in data.files]
        if missing:
            raise ValueError(
                f"Calibration file {path} is missing keys: {missing}. "
                f"Found: {list(data.files)}"
            )
        meta = json.loads(str(data["meta_json"]))
        version = int(meta.get("version", 0))
        if version != CALIBRATION_FILE_VERSION:
            raise ValueError(
                f"Unsupported calibration file version {version} "
                f"(expected {CALIBRATION_FILE_VERSION})."
            )
        arrays = {k: data[k] for k in ("images", "wrist_images", "states", "task_descriptions")}
        batches = arrays_to_batches(arrays)
    logger.info("Loaded %d calibration samples from %s", len(batches), path)
    return batches, meta


def make_file_calibration_provider(path: str | Path) -> Callable[[int], Iterator[dict]]:
    """Return a provider that cycles through samples stored in ``path``."""
    batches, meta = load_calibration_dataset(path)
    stored = len(batches)

    def provider(num_samples: int) -> Iterator[dict]:
        if num_samples > stored:
            logger.warning(
                "Requested %d calibration samples but file %s only has %d; "
                "cycling through stored samples.",
                num_samples,
                path,
                stored,
            )
        for i in range(num_samples):
            yield batches[i % stored]

    provider.__calibration_meta__ = meta  # type: ignore[attr-defined]
    return provider


__all__ = [
    "CALIBRATION_FILE_VERSION",
    "arrays_to_batches",
    "batches_to_arrays",
    "load_calibration_dataset",
    "make_file_calibration_provider",
    "save_calibration_dataset",
]
