"""Unit tests for per-sample calibration noise + noise ensemble."""

from __future__ import annotations

import torch

from qvla.build.calibration_noise import (
    _STATE,
    calibration_noise_for_sample_if_enabled,
    calibration_noise_seed,
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


def test_noise_seed_includes_noise_index():
    s0 = calibration_noise_seed(run_id=1, sample_index=0, noise_index=0)
    s1 = calibration_noise_seed(run_id=1, sample_index=0, noise_index=1)
    assert s0 != s1


def test_noise_ensemble_k_draws_distinct_noises():
    adapter = _FakeAdapter()
    _STATE.pop(id(adapter), None)
    install_per_sample_calibration_noise(
        adapter, build_seed=0, noise_ensemble_k=3
    )
    n0 = calibration_noise_for_sample_if_enabled(adapter, 0, noise_index=0)
    n1 = calibration_noise_for_sample_if_enabled(adapter, 0, noise_index=1)
    n0b = calibration_noise_for_sample_if_enabled(adapter, 0, noise_index=0)
    assert n0 is not None and n1 is not None
    assert torch.equal(n0, n0b)
    assert not torch.equal(n0, n1)
    _STATE.pop(id(adapter), None)


def test_missing_sample_index_raises_when_installed():
    adapter = _FakeAdapter()
    _STATE.pop(id(adapter), None)
    install_per_sample_calibration_noise(adapter, noise_ensemble_k=1)
    try:
        calibration_noise_for_sample_if_enabled(adapter, None)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
    _STATE.pop(id(adapter), None)


def test_noise_index_out_of_range_raises():
    adapter = _FakeAdapter()
    _STATE.pop(id(adapter), None)
    install_per_sample_calibration_noise(adapter, noise_ensemble_k=2)
    try:
        calibration_noise_for_sample_if_enabled(adapter, 0, noise_index=2)
        assert False, "expected ValueError"
    except ValueError:
        pass
    _STATE.pop(id(adapter), None)


def test_run_ids_differ_across_installs():
    a = new_calibration_noise_run_id(build_seed=0)
    b = new_calibration_noise_run_id(build_seed=0)
    assert a != b
