"""Tests for offline calibration file I/O (path B)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_PKG_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from qvla.calibration.file import (
    load_calibration_dataset,
    make_file_calibration_provider,
    save_calibration_dataset,
)
from qvla.adapters.pi05 import PI05Adapter, PI05AdapterConfig


def _write_min_pi05_checkpoint_config(checkpoint_dir: Path) -> None:
    (checkpoint_dir / "config.json").write_text(
        json.dumps(
            {
                "input_features": {"observation.state": {"shape": [8]}},
                "output_features": {"action": {"shape": [7]}},
                "image_resolution": [224, 224],
                "empty_cameras": 1,
                "tokenizer_max_length": 200,
            }
        ),
        encoding="utf-8",
    )


def _sample_batch(i: int) -> dict:
    rng = np.random.default_rng(i)
    img = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
    return {
        "images": {"image": img, "wrist_image": img.copy()},
        "states": rng.standard_normal(8).astype(np.float32),
        "task_description": f"task {i}",
    }


def test_save_load_roundtrip(tmp_path):
    batches = [_sample_batch(i) for i in range(3)]
    path = tmp_path / "calib.npz"
    save_calibration_dataset(path, batches, meta={"source": "test", "suite": "libero_object"})
    loaded, meta = load_calibration_dataset(path)
    assert meta["source"] == "test"
    assert meta["num_samples"] == 3
    assert len(loaded) == 3
    assert loaded[1]["task_description"] == "task 1"
    assert loaded[1]["images"]["image"].shape == (256, 256, 3)


def test_file_provider_cycles_when_requesting_more(tmp_path):
    path = tmp_path / "calib.npz"
    save_calibration_dataset(path, [_sample_batch(0), _sample_batch(1)])
    provider = make_file_calibration_provider(path)
    out = list(provider(5))
    assert len(out) == 5
    assert out[0]["task_description"] == "task 0"
    assert out[2]["task_description"] == "task 0"


def test_pi05_adapter_file_source(tmp_path):
    ckpt = tmp_path / "checkpoint"
    ckpt.mkdir()
    _write_min_pi05_checkpoint_config(ckpt)
    path = tmp_path / "calib.npz"
    save_calibration_dataset(path, [_sample_batch(0)])
    adapter = PI05Adapter(
        PI05AdapterConfig.from_checkpoint(
            ckpt,
            calibration_source="file",
            calibration_data_path=path,
        )
    )
    batches = list(adapter.iter_calibration_batches(1))
    assert batches[0]["task_description"] == "task 0"


def test_pi05_adapter_file_source_requires_path(tmp_path):
    ckpt = tmp_path / "checkpoint"
    ckpt.mkdir()
    _write_min_pi05_checkpoint_config(ckpt)
    adapter = PI05Adapter(
        PI05AdapterConfig.from_checkpoint(ckpt, calibration_source="file")
    )
    with pytest.raises(ValueError, match="calibration_data_path"):
        list(adapter.iter_calibration_batches(1))
