"""High-level Fisher driver used by the builder.

The driver's job is exactly:

1. Validate config, look up the requested collector class, resolve
   engine state on the adapter.
2. Iterate over calibration samples in batches, running one differentiable
   forward + backward estimator per (batch, noise) pair through a
   :class:`~qvla.build.fisher.collectors.BaseFisherCollector`.
3. Aggregate per-step sensitivity into the ``(in_features,)`` vector each
   consumer (perm-score / Fisher-SVD / Fisher-GPTQ / SmoothQuant) expects.

The dispatch on ``fisher_type`` lives here, not inside the collector — the
collectors don't know about calibration batches, and the driver doesn't
know how a collector extracts channel-level numbers from a backward.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from qvla.build.fisher.collectors import (
    BaseFisherCollector,
    InputGradFisherCollector,
    OutputHessianFisherCollector,
)
from qvla.build.fisher.common import (
    FisherMethod,
    FisherType,
    StepAggregation,
)

if TYPE_CHECKING:
    from qvla.core.pipeline import Transform

logger = logging.getLogger(__name__)


_COLLECTOR_REGISTRY: dict[str, type[BaseFisherCollector]] = {
    "input_grad": InputGradFisherCollector,
    "output_hessian": OutputHessianFisherCollector,
}


def _resolve_collector_class(fisher_type: str) -> type[BaseFisherCollector]:
    key = str(fisher_type).strip().lower()
    if key not in _COLLECTOR_REGISTRY:
        raise ValueError(
            f"Unknown fisher_type {fisher_type!r}; expected one of "
            f"{sorted(_COLLECTOR_REGISTRY)}."
        )
    return _COLLECTOR_REGISTRY[key]


def compute_fisher_sensitivity(
    adapter: Any,
    target_modules: list[tuple[str, str, nn.Module]],
    *,
    action_dim: int,
    num_samples: int = 4,
    num_dit_steps: int = 10,
    step_aggregation: StepAggregation = "uniform",
    noise_ensemble_k: int = 1,
    action_timestep: str = "all",
    batch_size: int = 1,
    fisher_type: FisherType = "input_grad",
    method: FisherMethod = "exact",
    hutchinson_probes: int = 8,
    rotations: dict[str, "Transform"] | None = None,
    progress: Any | None = None,
) -> dict[str, torch.Tensor]:
    """End-to-end Fisher sensitivity on the FP model.

    ``fisher_type`` selects *what* is measured
    (:class:`~qvla.build.fisher.collectors.InputGradFisherCollector` vs
    :class:`~qvla.build.fisher.collectors.OutputHessianFisherCollector`);
    ``method`` selects the outer expectation estimator
    (``"exact"`` — one backward per action DoF, or ``"hutchinson"`` —
    ``hutchinson_probes`` Rademacher probes).

    Runs
    ``ceil(num_samples / batch_size) × noise_ensemble_k`` forward+backward
    loops inside ``adapter.fisher_forward_context()`` and returns a dict
    mapping each layer name to its aggregated ``(in_features,)`` vector.

    When ``rotations`` is supplied, sensitivity is reported in the rotated
    input-channel layout — required by Fisher-GPTQ so Hadamard blocks
    don't get uniformised by an input-space diagonal.
    """
    from qvla.runtime.step_context import reset_step_counters

    _validate_driver_inputs(
        noise_ensemble_k=noise_ensemble_k,
        action_dim=action_dim,
        batch_size=batch_size,
        num_samples=num_samples,
        method=method,
        hutchinson_probes=hutchinson_probes,
    )
    collector_cls = _resolve_collector_class(fisher_type)

    if not target_modules:
        logger.info("Fisher: no target modules; skipping.")
        return {}

    engine = adapter.engine
    if engine is None:
        raise RuntimeError(
            "Adapter has no engine — call build_model() and "
            "warmup_for_calibration() first."
        )
    sched = engine.entry.scheduler
    model = engine.entry.model
    max_B = int(getattr(sched, "max_batch_size", 1))
    if batch_size > max_B:
        raise RuntimeError(
            f"fisher batch_size={batch_size} exceeds engine max_batch_size={max_B}. "
            "Set adapter.cfg.max_batch_size before build_model() "
            "(build_pack does this from config.fisher_batch_size)."
        )

    samples = list(adapter.iter_calibration_batches(num_samples))
    if len(samples) != num_samples:
        raise RuntimeError(
            f"Fisher expected {num_samples} calibration samples, "
            f"got {len(samples)} from iter_calibration_batches()."
        )

    n_forwards = (num_samples + batch_size - 1) // batch_size
    total = n_forwards * noise_ensemble_k
    logger.info(
        "Fisher: fisher_type=%s method=%s%s; %d samples, batch_size=%d "
        "(%d forwards × %d noise).",
        fisher_type,
        method,
        f", probes={hutchinson_probes}" if method == "hutchinson" else "",
        num_samples,
        batch_size,
        n_forwards,
        noise_ensemble_k,
    )

    done = 0

    def _run_fisher_loop(fc: BaseFisherCollector) -> None:
        nonlocal done
        for start in range(0, num_samples, batch_size):
            chunk = samples[start : start + batch_size]
            indices = list(range(start, start + len(chunk)))
            for noise_index in range(noise_ensemble_k):
                logger.info(
                    "Fisher samples [%d:%d] / %d noise %d / %d",
                    start,
                    start + len(chunk),
                    num_samples,
                    noise_index + 1,
                    noise_ensemble_k,
                )
                fc.begin_sample()
                reset_step_counters(model)
                model.zero_grad(set_to_none=True)

                fc.set_current_step(None)
                actions = adapter.forward_differentiable(
                    chunk,
                    sample_indices=indices,
                    noise_index=noise_index,
                )
                if int(actions.shape[0]) != len(chunk):
                    raise RuntimeError(
                        f"Fisher forward returned batch {int(actions.shape[0])} "
                        f"but requested {len(chunk)} observations."
                    )

                fc.compute_jacobian_sensitivity(
                    actions,
                    model=model,
                    action_timestep=action_timestep,
                    method=method,
                    hutchinson_probes=hutchinson_probes,
                )
                done += 1
                if progress is not None:
                    progress("fisher", 0.05 + 0.95 * done / max(1, total))

    with collector_cls(
        target_modules,
        num_dit_steps,
        rotations=rotations,
        action_dim=action_dim,
    ) as fc:
        def _set_step(step: int) -> None:
            fc.set_current_step(int(step))

        with adapter.fisher_forward_context(_set_step):
            _run_fisher_loop(fc)
        results = fc.get_results()

    aggregated: dict[str, torch.Tensor] = {}
    for name, result in results.items():
        aggregated[name] = result.aggregate(step_aggregation)

    _log_summary(target_modules, aggregated)
    return aggregated


def _validate_driver_inputs(
    *,
    noise_ensemble_k: int,
    action_dim: int,
    batch_size: int,
    num_samples: int,
    method: str,
    hutchinson_probes: int,
) -> None:
    if noise_ensemble_k < 1:
        raise ValueError(f"noise_ensemble_k must be >= 1, got {noise_ensemble_k}.")
    if action_dim <= 0:
        raise ValueError(f"action_dim must be positive, got {action_dim}.")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}.")
    if batch_size > num_samples:
        raise ValueError(
            f"Fisher batch_size={batch_size} cannot exceed "
            f"num_samples={num_samples}."
        )
    m = str(method).strip().lower()
    if m not in ("exact", "hutchinson"):
        raise ValueError(
            f"Unknown fisher method {method!r}; expected 'exact' or 'hutchinson'."
        )
    if hutchinson_probes < 1:
        raise ValueError(
            f"hutchinson_probes must be >= 1, got {hutchinson_probes}."
        )


def _log_summary(
    target_modules: list[tuple[str, str, nn.Module]],
    aggregated: dict[str, torch.Tensor],
) -> None:
    n_nonzero = sum(1 for v in aggregated.values() if v.abs().sum().item() > 0)
    n_llm = sum(1 for _, s, _ in target_modules if s == "llm")
    n_dit = sum(1 for _, s, _ in target_modules if s == "dit")
    n_llm_nz = sum(
        1
        for name, scope, _ in target_modules
        if scope == "llm" and aggregated[name].abs().sum().item() > 0
    )
    n_dit_nz = sum(
        1
        for name, scope, _ in target_modules
        if scope == "dit" and aggregated[name].abs().sum().item() > 0
    )
    logger.info(
        "Fisher sensitivity: %d / %d layers have non-zero sensitivity "
        "(LLM %d/%d, DiT %d/%d).",
        n_nonzero,
        len(aggregated),
        n_llm_nz,
        n_llm,
        n_dit_nz,
        n_dit,
    )


__all__ = ["compute_fisher_sensitivity"]
