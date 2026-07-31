"""Diffusion noise policies for pack-build calibration (pi0.5).

``per_sample`` preserves the original behaviour: every pack build gets a
random run id and each ``(sample_index, noise_index)`` receives deterministic
noise within that build.

``global`` matches the model server's global-noise protocol: all calibration
samples share one float32 noise drawn by a freshly seeded CPU generator.  It
therefore exactly matches ``inference_seed=build_seed`` and is unaffected by
global RNG consumption.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from dataclasses import dataclass, field
from typing import Literal

import torch


logger = logging.getLogger(__name__)

CalibrationNoiseMode = Literal["per_sample", "global"]


@dataclass
class _CalibrationNoiseState:
    mode: CalibrationNoiseMode
    run_id: int
    base_seed: int
    noise_ensemble_k: int
    cache: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)
    global_noise: torch.Tensor | None = None


_STATE: dict[int, _CalibrationNoiseState] = {}


def new_calibration_noise_run_id(*, build_seed: int) -> int:
    """One random run id per pack build for ``per_sample`` mode."""
    nonce = secrets.randbits(63)
    digest = hashlib.sha256(f"{build_seed}:{nonce}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def per_sample_calibration_noise_seed(
    *, run_id: int, sample_index: int, noise_index: int
) -> int:
    """Stable per-sample seed within one pack build."""
    digest = hashlib.sha256(
        f"{run_id}:{sample_index}:{noise_index}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def global_calibration_noise_seed(*, base_seed: int) -> int:
    """Model-server-compatible global calibration-noise seed."""
    return int(base_seed) % (2**31)


def _sample_noise(
    *,
    seed: int,
    chunk_size: int,
    max_action_dim: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Draw one ``(1, chunk_size, max_action_dim)`` noise from a local generator."""
    device = torch.device(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return torch.randn(
        1,
        chunk_size,
        max_action_dim,
        generator=generator,
        device=device,
        dtype=dtype,
    )


def install_per_sample_calibration_noise(
    adapter,
    *,
    build_seed: int = 0,
    noise_ensemble_k: int = 1,
    mode: CalibrationNoiseMode = "per_sample",
) -> None:
    """Enable the selected fixed calibration-noise policy."""
    if noise_ensemble_k < 1:
        raise ValueError(f"noise_ensemble_k must be >= 1, got {noise_ensemble_k}.")
    if mode not in ("per_sample", "global"):
        raise ValueError(f"Unknown calibration noise mode: {mode!r}.")
    if mode == "global" and noise_ensemble_k != 1:
        raise ValueError(
            "calibration noise mode 'global' shares one noise across all samples; "
            f"noise_ensemble_k must be 1, got {noise_ensemble_k}."
        )
    if getattr(adapter, "model_kind", None) != "pi05":
        if noise_ensemble_k != 1:
            raise RuntimeError(
                f"noise_ensemble_k={noise_ensemble_k} requires pi05 calibration "
                f"noise; got model_kind={getattr(adapter, 'model_kind', None)!r}."
            )
        return
    _STATE[id(adapter)] = _CalibrationNoiseState(
        mode=mode,
        run_id=new_calibration_noise_run_id(build_seed=build_seed),
        base_seed=build_seed,
        noise_ensemble_k=noise_ensemble_k,
    )
    logger.info(
        "Calibration diffusion noise enabled "
        "(mode=%s, build_seed=%d, noise_ensemble_k=%d).",
        mode,
        build_seed,
        noise_ensemble_k,
    )


def calibration_noise_for_sample_if_enabled(
    adapter,
    sample_index: int | None,
    *,
    noise_index: int = 0,
) -> torch.Tensor | None:
    """Return cached calibration noise, or ``None`` when not installed."""
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

    sched = adapter._engine.entry.scheduler
    cfg = sched.cfg
    if state.mode == "global":
        if state.global_noise is not None:
            return state.global_noise
        noise = _sample_noise(
            seed=global_calibration_noise_seed(base_seed=state.base_seed),
            chunk_size=cfg.chunk_size,
            max_action_dim=cfg.max_action_dim,
            device="cpu",
            dtype=torch.float32,
        )
        state.global_noise = noise
        return noise
    elif state.mode == "per_sample":
        key = (sample_index, noise_index)
        cached = state.cache.get(key)
        if cached is not None:
            return cached
        noise = _sample_noise(
            seed=per_sample_calibration_noise_seed(
                run_id=state.run_id,
                sample_index=sample_index,
                noise_index=noise_index,
            ),
            chunk_size=cfg.chunk_size,
            max_action_dim=cfg.max_action_dim,
            device=sched.device,
            dtype=sched.params_dtype,
        )
        state.cache[key] = noise
        return noise
    else:
        raise ValueError(f"Unknown calibration noise mode: {state.mode!r}.")
