"""Diffusion noise policies for pack-build calibration.

``per_sample`` preserves the original behaviour: every pack build gets a
random run id and each ``(sample_index, noise_index)`` receives deterministic
noise within that build.

``global`` matches the model server's global-noise protocol: all calibration
samples share one float32 noise drawn by a freshly seeded CPU generator.  It
therefore exactly matches ``inference_seed=build_seed`` and is unaffected by
global RNG consumption.

The noise shape/placement is model-specific and comes from
:meth:`ModelAdapter.calibration_noise_spec`.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch


logger = logging.getLogger(__name__)

CalibrationNoiseMode = Literal["per_sample", "global"]
# Temporary cross-process validation artifacts for pi05-phyai.py.
# Resolved relative to this file: ../../../.. / calibration_data / noise
GLOBAL_NOISE_DUMP_DIR = (
    Path(__file__).resolve().parents[4] / "calibration_data" / "noise"
)


def _model_slug(model: str) -> str:
    slug = model.strip().lower()
    if not slug or any(c != "_" and not c.isalnum() for c in slug):
        raise ValueError(
            f"model must be a non-empty alphanumeric/underscore slug, got {model!r}."
        )
    return slug


def global_noise_dump_path(*, model: str, seed: int) -> Path:
    """Path for the temporary global-noise dump keyed by model and seed."""
    return (
        GLOBAL_NOISE_DUMP_DIR
        / f"qvla_global_noise_{_model_slug(model)}_seed{int(seed)}.pt"
    )


def _save_or_verify_global_noise(
    noise: torch.Tensor, *, model: str, seed: int
) -> Path:
    """Save ``noise`` once; if the dump exists, require an exact match."""
    path = global_noise_dump_path(model=model, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = torch.load(path, map_location="cpu", weights_only=True)
        if not torch.equal(noise, existing):
            raise RuntimeError(
                "Generated global calibration noise differs from existing dump "
                f"at {path} (new={tuple(noise.shape)}/{noise.dtype}, "
                f"existing={tuple(existing.shape)}/{existing.dtype})."
            )
        logger.info(
            "Existing global calibration noise matches dump: %s", path
        )
        return path
    torch.save(noise, path)
    logger.info("Saved global calibration noise to %s.", path)
    return path


@dataclass(frozen=True)
class CalibrationNoiseSpec:
    """Shape/placement of one diffusion-noise draw: ``(1, chunk_size, action_dim)``."""

    chunk_size: int
    action_dim: int
    device: torch.device
    dtype: torch.dtype


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
    action_dim: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Draw one ``(1, chunk_size, action_dim)`` noise from a local generator."""
    device = torch.device(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return torch.randn(
        1,
        chunk_size,
        action_dim,
        generator=generator,
        device=device,
        dtype=dtype,
    )


def _noise_spec(adapter) -> CalibrationNoiseSpec:
    fn = getattr(adapter, "calibration_noise_spec", None)
    if fn is None:
        raise RuntimeError(
            f"{type(adapter).__name__} does not implement calibration_noise_spec(); "
            "fixed calibration diffusion noise requires a noise shape/placement."
        )
    spec = fn()
    if not isinstance(spec, CalibrationNoiseSpec):
        raise TypeError(
            f"{type(adapter).__name__}.calibration_noise_spec() must return "
            f"CalibrationNoiseSpec, got {type(spec).__name__}."
        )
    if spec.chunk_size < 1 or spec.action_dim < 1:
        raise ValueError(
            f"Invalid CalibrationNoiseSpec from {type(adapter).__name__}: "
            f"chunk_size={spec.chunk_size}, action_dim={spec.action_dim}."
        )
    return spec


def install_per_sample_calibration_noise(
    adapter,
    *,
    build_seed: int = 0,
    noise_ensemble_k: int = 1,
    mode: CalibrationNoiseMode = "per_sample",
) -> None:
    """Enable the selected fixed calibration-noise policy for any adapter."""
    if noise_ensemble_k < 1:
        raise ValueError(f"noise_ensemble_k must be >= 1, got {noise_ensemble_k}.")
    if mode not in ("per_sample", "global"):
        raise ValueError(f"Unknown calibration noise mode: {mode!r}.")
    if mode == "global" and noise_ensemble_k != 1:
        raise ValueError(
            "calibration noise mode 'global' shares one noise across all samples; "
            f"noise_ensemble_k must be 1, got {noise_ensemble_k}."
        )
    # Resolve shape eagerly so unsupported adapters fail at install time.
    _noise_spec(adapter)
    _STATE[id(adapter)] = _CalibrationNoiseState(
        mode=mode,
        run_id=new_calibration_noise_run_id(build_seed=build_seed),
        base_seed=build_seed,
        noise_ensemble_k=noise_ensemble_k,
    )
    logger.info(
        "Calibration diffusion noise enabled "
        "(model=%s, mode=%s, build_seed=%d, noise_ensemble_k=%d).",
        getattr(adapter, "model_kind", type(adapter).__name__),
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

    spec = _noise_spec(adapter)
    if state.mode == "global":
        if state.global_noise is not None:
            return state.global_noise
        noise = _sample_noise(
            seed=global_calibration_noise_seed(base_seed=state.base_seed),
            chunk_size=spec.chunk_size,
            action_dim=spec.action_dim,
            device="cpu",
            dtype=torch.float32,
        )
        # Temporary cross-process validation artifact for model servers.
        model = getattr(adapter, "model_kind", type(adapter).__name__)
        _save_or_verify_global_noise(
            noise, model=str(model), seed=state.base_seed
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
            chunk_size=spec.chunk_size,
            action_dim=spec.action_dim,
            device=spec.device,
            dtype=spec.dtype,
        )
        state.cache[key] = noise
        return noise
    else:
        raise ValueError(f"Unknown calibration noise mode: {state.mode!r}.")
