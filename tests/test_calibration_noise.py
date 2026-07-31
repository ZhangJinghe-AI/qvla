"""Unit tests for selectable calibration-noise policies."""

from __future__ import annotations

import torch

from qvla.build.calibration_noise import (
    _STATE,
    calibration_noise_for_sample_if_enabled,
    global_calibration_noise_seed,
    per_sample_calibration_noise_seed,
    install_per_sample_calibration_noise,
    new_calibration_noise_run_id,
)


class _FakeSchedCfg:
    chunk_size = 4
    max_action_dim = 7


class _FakeSched:
    device = torch.device("cpu")
    params_dtype = torch.float32
    cfg = _FakeSchedCfg()


class _FakeEntry:
    scheduler = _FakeSched()


class _FakeEngine:
    entry = _FakeEntry()


class _FakeAdapter:
    model_kind = "pi05"
    _engine = _FakeEngine()


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
    assert not torch.equal(n0, n1)
    assert not torch.equal(n0, nk)
    _clear(adapter)


def test_global_noise_matches_model_server_protocol():
    adapter = _FakeAdapter()
    _clear(adapter)
    install_per_sample_calibration_noise(
        adapter, build_seed=0, noise_ensemble_k=1, mode="global"
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    expected = torch.randn(
        1,
        _FakeSchedCfg.chunk_size,
        _FakeSchedCfg.max_action_dim,
        generator=generator,
        dtype=torch.float32,
    )
    actual = calibration_noise_for_sample_if_enabled(adapter, 0)
    assert actual is not None
    assert torch.equal(actual, expected)
    _clear(adapter)


def test_global_noise_is_shared_and_ignores_global_rng():
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
