"""Per-sample diffusion noise for pack-build calibration (pi0.5)."""

from __future__ import annotations

import hashlib
import logging
import secrets
from dataclasses import dataclass

import torch


logger = logging.getLogger(__name__)


@dataclass
class _CalibrationNoiseState:
    run_id: int
    cache: dict[int, torch.Tensor]


_STATE: dict[int, _CalibrationNoiseState] = {}


def new_calibration_noise_run_id(*, build_seed: int) -> int:
    """One random run id per pack build (differs across rebuilds)."""
    nonce = secrets.randbits(63)
    digest = hashlib.sha256(f"{build_seed}:{nonce}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def calibration_noise_seed(*, run_id: int, sample_index: int) -> int:
    """Stable seed for ``sample_index`` within a single pack build."""
    digest = hashlib.sha256(f"{run_id}:{sample_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def install_per_sample_calibration_noise(adapter, *, build_seed: int = 0) -> None:
    """Enable fixed per-sample diffusion noise on a pi0.5 adapter."""
    if getattr(adapter, "model_kind", None) != "pi05":
        return
    _STATE[id(adapter)] = _CalibrationNoiseState(
        run_id=new_calibration_noise_run_id(build_seed=build_seed),
        cache={},
    )
    logger.info(
        "Per-sample calibration diffusion noise enabled (stable within build)."
    )


def calibration_noise_for_sample_if_enabled(
    adapter, sample_index: int | None
) -> torch.Tensor | None:
    """Return cached noise for *sample_index*, or ``None`` if not enabled."""
    if sample_index is None:
        return None
    state = _STATE.get(id(adapter))
    if state is None:
        return None

    cached = state.cache.get(sample_index)
    if cached is not None:
        return cached

    sched = adapter._engine.entry.scheduler
    cfg = sched.cfg
    generator = torch.Generator(device=sched.device)
    generator.manual_seed(
        calibration_noise_seed(run_id=state.run_id, sample_index=sample_index)
    )
    noise = torch.randn(
        1,
        cfg.chunk_size,
        cfg.max_action_dim,
        generator=generator,
        device=sched.device,
        dtype=sched.params_dtype,
    )
    state.cache[sample_index] = noise
    return noise
