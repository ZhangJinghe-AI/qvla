"""Per-sample diffusion noise for pack-build calibration (pi0.5).

Each pack build draws a random ``run_id``. Within a build, noise for
``(sample_index, noise_index)`` is deterministic and cached so every
calibration pass (pipeline steps, final Hessian / act-scale) reuses the
same tensors.

``noise_ensemble_k > 1`` draws K independent noises per sample; the builder
runs K forwards and the collector aggregates (max for amax, sum for XᵀX).
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from dataclasses import dataclass, field

import torch


logger = logging.getLogger(__name__)


@dataclass
class _CalibrationNoiseState:
    run_id: int
    noise_ensemble_k: int
    cache: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)


_STATE: dict[int, _CalibrationNoiseState] = {}


def new_calibration_noise_run_id(*, build_seed: int) -> int:
    """One random run id per pack build (differs across rebuilds)."""
    nonce = secrets.randbits(63)
    digest = hashlib.sha256(f"{build_seed}:{nonce}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def calibration_noise_seed(
    *, run_id: int, sample_index: int, noise_index: int
) -> int:
    """Stable seed for ``(sample_index, noise_index)`` within one pack build."""
    digest = hashlib.sha256(
        f"{run_id}:{sample_index}:{noise_index}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def install_per_sample_calibration_noise(
    adapter,
    *,
    build_seed: int = 0,
    noise_ensemble_k: int = 1,
) -> None:
    """Enable fixed per-sample diffusion noise on a pi0.5 adapter."""
    if noise_ensemble_k < 1:
        raise ValueError(f"noise_ensemble_k must be >= 1, got {noise_ensemble_k}.")
    if getattr(adapter, "model_kind", None) != "pi05":
        if noise_ensemble_k != 1:
            raise RuntimeError(
                f"noise_ensemble_k={noise_ensemble_k} requires pi05 calibration "
                f"noise; got model_kind={getattr(adapter, 'model_kind', None)!r}."
            )
        return
    _STATE[id(adapter)] = _CalibrationNoiseState(
        run_id=new_calibration_noise_run_id(build_seed=build_seed),
        noise_ensemble_k=noise_ensemble_k,
    )
    logger.info(
        "Per-sample calibration diffusion noise enabled "
        "(noise_ensemble_k=%d, stable within build).",
        noise_ensemble_k,
    )


def calibration_noise_for_sample_if_enabled(
    adapter,
    sample_index: int | None,
    *,
    noise_index: int = 0,
) -> torch.Tensor | None:
    """Return cached noise for ``(sample_index, noise_index)``, or ``None``."""
    state = _STATE.get(id(adapter))
    if state is None:
        return None
    if sample_index is None:
        raise RuntimeError(
            "Calibration noise is installed but sample_index is None; "
            "pass sample_index from the builder calibration loop."
        )
    if noise_index < 0 or noise_index >= state.noise_ensemble_k:
        raise ValueError(
            f"noise_index={noise_index} out of range for "
            f"noise_ensemble_k={state.noise_ensemble_k}."
        )

    key = (sample_index, noise_index)
    cached = state.cache.get(key)
    if cached is not None:
        return cached

    sched = adapter._engine.entry.scheduler
    cfg = sched.cfg
    generator = torch.Generator(device=sched.device)
    generator.manual_seed(
        calibration_noise_seed(
            run_id=state.run_id,
            sample_index=sample_index,
            noise_index=noise_index,
        )
    )
    noise = torch.randn(
        1,
        cfg.chunk_size,
        cfg.max_action_dim,
        generator=generator,
        device=sched.device,
        dtype=sched.params_dtype,
    )
    state.cache[key] = noise
    return noise
