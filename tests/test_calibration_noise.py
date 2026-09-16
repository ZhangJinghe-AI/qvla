"""Unit tests for selectable calibration-noise policies."""

from __future__ import annotations

import torch

import qvla.build.calibration_noise as calibration_noise_module
from qvla.build.calibration_noise import (
    CalibrationNoiseSpec,
    _STATE,
    calibration_noise_for_sample_if_enabled,
    global_calibration_noise_seed,
    global_noise_dump_path,
    per_sample_calibration_noise_seed,
    install_per_sample_calibration_noise,
    new_calibration_noise_run_id,
)


class _FakeAdapter:
    model_kind = "pi05"

    def calibration_noise_spec(self) -> CalibrationNoiseSpec:
        return CalibrationNoiseSpec(
            chunk_size=4,
            action_dim=7,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


class _FakeGrootAdapter:
    model_kind = "groot_n17"

    def calibration_noise_spec(self) -> CalibrationNoiseSpec:
        return CalibrationNoiseSpec(
            chunk_size=16,
            action_dim=32,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def _clear(adapter) -> None:
    _STATE.pop(id(adapter), None)


def test_per_sample_seed_includes_noise_index():
    s0 = per_sample_calibration_noise_seed(run_id=1, sample_index=0, noise_index=0)
    s1 = per_sample_calibration_noise_seed(run_id=1, sample_index=0, noise_index=1)
    assert s0 != s1


def test_per_sample_mode_preserves_original_distinct_noises():
    adapter = _FakeAdapter()
    _clear(adapter)
    install_per_sample_calibration_noise(
        adapter, build_seed=0, noise_ensemble_k=2
    )
    n0 = calibration_noise_for_sample_if_enabled(adapter, 0, noise_index=0)
    n1 = calibration_noise_for_sample_if_enabled(adapter, 1, noise_index=0)
    nk = calibration_noise_for_sample_if_enabled(adapter, 0, noise_index=1)
    assert n0 is not None and n1 is not None and nk is not None
    assert n0.shape == (1, 4, 7)
    assert not torch.equal(n0, n1)
    assert not torch.equal(n0, nk)
    _clear(adapter)


def test_groot_adapter_supports_noise_ensemble():
    adapter = _FakeGrootAdapter()
    _clear(adapter)
    install_per_sample_calibration_noise(
        adapter, build_seed=0, noise_ensemble_k=2
    )
    n0 = calibration_noise_for_sample_if_enabled(adapter, 0, noise_index=0)
    nk = calibration_noise_for_sample_if_enabled(adapter, 0, noise_index=1)
    assert n0 is not None and nk is not None
    assert n0.shape == (1, 16, 32)
    assert not torch.equal(n0, nk)
    _clear(adapter)


def test_global_noise_matches_model_server_protocol(tmp_path, monkeypatch):
    monkeypatch.setattr(
        calibration_noise_module, "GLOBAL_NOISE_DUMP_DIR", tmp_path
    )
    adapter = _FakeAdapter()
    _clear(adapter)
    install_per_sample_calibration_noise(
        adapter, build_seed=0, noise_ensemble_k=1, mode="global"
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    expected = torch.randn(
        1,
        4,
        7,
        generator=generator,
        dtype=torch.float32,
    )
    actual = calibration_noise_for_sample_if_enabled(adapter, 0)
    assert actual is not None
    assert torch.equal(actual, expected)
    dump_path = global_noise_dump_path(model="pi05", seed=0)
    assert dump_path == tmp_path / "qvla_global_noise_pi05_seed0.pt"
    assert torch.equal(torch.load(dump_path, weights_only=True), expected)
    _clear(adapter)


def test_global_noise_is_shared_and_ignores_global_rng(tmp_path, monkeypatch):
    monkeypatch.setattr(
        calibration_noise_module, "GLOBAL_NOISE_DUMP_DIR", tmp_path
    )
    adapter = _FakeAdapter()
    _clear(adapter)
    install_per_sample_calibration_noise(
        adapter, build_seed=17, noise_ensemble_k=1, mode="global"
    )
    n0 = calibration_noise_for_sample_if_enabled(adapter, 0)
    _ = torch.randn(100)
    n1 = calibration_noise_for_sample_if_enabled(adapter, 1)
    assert n0 is not None and n1 is not None
    assert torch.equal(n0, n1)
    _clear(adapter)


def test_global_noise_dump_reuses_matching_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        calibration_noise_module, "GLOBAL_NOISE_DUMP_DIR", tmp_path
    )
    adapter = _FakeAdapter()
    _clear(adapter)
    install_per_sample_calibration_noise(
        adapter, build_seed=0, noise_ensemble_k=1, mode="global"
    )
    first = calibration_noise_for_sample_if_enabled(adapter, 0)
    assert first is not None
    dump_path = global_noise_dump_path(model="pi05", seed=0)
    mtime = dump_path.stat().st_mtime_ns

    _clear(adapter)
    install_per_sample_calibration_noise(
        adapter, build_seed=0, noise_ensemble_k=1, mode="global"
    )
    second = calibration_noise_for_sample_if_enabled(adapter, 0)
    assert second is not None
    assert torch.equal(first, second)
    assert dump_path.stat().st_mtime_ns == mtime
    _clear(adapter)


def test_global_noise_dump_raises_on_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(
        calibration_noise_module, "GLOBAL_NOISE_DUMP_DIR", tmp_path
    )
    dump_path = global_noise_dump_path(model="pi05", seed=0)
    torch.save(torch.ones(1, 4, 7), dump_path)

    adapter = _FakeAdapter()
    _clear(adapter)
    install_per_sample_calibration_noise(
        adapter, build_seed=0, noise_ensemble_k=1, mode="global"
    )
    try:
        calibration_noise_for_sample_if_enabled(adapter, 0)
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "differs from existing dump" in str(exc)
    _clear(adapter)


def test_global_mode_rejects_noise_ensemble():
    adapter = _FakeAdapter()
    _clear(adapter)
    try:
        install_per_sample_calibration_noise(
            adapter, build_seed=0, noise_ensemble_k=2, mode="global"
        )
        assert False, "expected ValueError"
    except ValueError:
        pass
    assert global_calibration_noise_seed(base_seed=0) == 0
    _clear(adapter)


def test_missing_sample_index_raises_when_installed():
    adapter = _FakeAdapter()
    _clear(adapter)
    install_per_sample_calibration_noise(adapter, noise_ensemble_k=1)
    try:
        calibration_noise_for_sample_if_enabled(adapter, None)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
    _clear(adapter)


def test_noise_index_out_of_range_raises():
    adapter = _FakeAdapter()
    _clear(adapter)
    install_per_sample_calibration_noise(adapter, noise_ensemble_k=2)
    try:
        calibration_noise_for_sample_if_enabled(adapter, 0, noise_index=2)
        assert False, "expected ValueError"
    except ValueError:
        pass
    _clear(adapter)


def test_run_ids_differ_across_installs():
    a = new_calibration_noise_run_id(build_seed=0)
    b = new_calibration_noise_run_id(build_seed=0)
    assert a != b


def test_install_requires_calibration_noise_spec():
    class _NoSpec:
        model_kind = "unknown"

    adapter = _NoSpec()
    _clear(adapter)
    try:
        install_per_sample_calibration_noise(adapter, noise_ensemble_k=1)
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "calibration_noise_spec" in str(exc)
    _clear(adapter)
