"""Tests for PI05 calibration batch helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PKG_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))


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


def test_pi05_adapter_defaults_to_synthetic(tmp_path):
    from qvla.adapters.pi05 import PI05Adapter, PI05AdapterConfig

    _write_min_pi05_checkpoint_config(tmp_path)
    adapter = PI05Adapter(PI05AdapterConfig.from_checkpoint(tmp_path))
    assert adapter.calibration_source == "synthetic"
    batches = list(adapter.iter_calibration_batches(2))
    assert len(batches) == 2
    assert "image" in batches[0]["images"]
