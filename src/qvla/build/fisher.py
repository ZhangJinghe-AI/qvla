"""Fisher sensitivity analysis for policy-aware rotation.

Computes the per-layer per-channel Fisher information diagonal by constructing
the *exact* Jacobian of the action output w.r.t. each target linear layer's
input activation.  For each action dimension ``a_i`` and hidden channel ``c``:

    p_l[c] = Σ_i  E[ (∂a_i / ∂h_l[:, c])² ]

Because the action space is low-dimensional (typically 7-DoF), we loop over
action dimensions and do one backward pass each — no random projections.

For DiT layers the sensitivity is collected **per denoising step** so that
downstream aggregation can weight steps differently.  For LLM layers a single
(step-agnostic) sensitivity vector is stored.

Step-wise aggregation
---------------------
``StepAggregation`` controls how per-step DiT sensitivities are collapsed
into a single vector consumed by the rotation builder.  Currently supported:

* ``"uniform"`` — equal weight for every step.

The interface is designed for easy extension (e.g. noise-schedule weighting).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


StepAggregation = Literal["uniform"]


@dataclass
class LayerFisherResult:
    """Accumulated Fisher sensitivity for one layer across calibration samples."""

    name: str
    scope: str  # "llm" | "dit"
    in_features: int

    # step_key -> (in_features,) accumulated squared Jacobian norm.
    # LLM layers use key ``None``; DiT layers use integer step indices.
    _step_accum: dict[int | None, torch.Tensor] = field(
        default_factory=dict, repr=False
    )
    _n_samples: int = 0

    def sensitivity_per_step(self) -> dict[int | None, torch.Tensor]:
        """Return per-step sensitivity averaged over calibration samples."""
        n = max(1, self._n_samples)
        return {step: s / n for step, s in self._step_accum.items()}

    def aggregate(self, method: StepAggregation = "uniform") -> torch.Tensor:
        """Collapse the step axis into a single ``(in_features,)`` vector."""
        per_step = self.sensitivity_per_step()
        if not per_step:
            return torch.zeros(self.in_features)

        if method == "uniform":
            stacked = torch.stack(list(per_step.values()), dim=0)
            return stacked.mean(dim=0)

        raise ValueError(f"Unknown step aggregation: {method!r}")


class FisherCollector:
    """Capture activations and compute exact Jacobian sensitivity.

    Typical usage inside the builder pipeline::

        with FisherCollector(target_modules, num_dit_steps=10) as fc:
            for batch in batches:
                fc.begin_sample()
                # ... patch expert runner to call fc.set_current_step(step) ...
                actions = adapter.forward_differentiable(batch)
                fc.compute_jacobian_sensitivity(actions, model)
            results = fc.get_results()

    Parameters
    ----------
    targets : list of ``(name, scope, module)``
        Same format returned by :func:`wrap.list_target_modules`.
    num_dit_steps : int
        Expected number of denoising steps (informational only; the
        collector records whatever steps actually arrive).
    """

    def __init__(
        self,
        targets: list[tuple[str, str, nn.Module]],
        num_dit_steps: int = 10,
    ) -> None:
        self._targets = targets
        self._num_dit_steps = num_dit_steps

        self._scopes: dict[str, str] = {}
        self._results: dict[str, LayerFisherResult] = {}
        self._hooks: list[Any] = []
        self._current_step: int | None = None

        # name -> [(step_key, activation_tensor)] for the current sample
        self._sample_acts: dict[str, list[tuple[int | None, torch.Tensor]]] = {}

        for name, scope, mod in targets:
            self._scopes[name] = scope
            in_f = int(getattr(mod, "in_features", 0))
            if in_f == 0:
                w = getattr(mod, "weight", None)
                in_f = int(w.shape[-1]) if w is not None else 0
            self._results[name] = LayerFisherResult(
                name=name, scope=scope, in_features=in_f
            )
            self._sample_acts[name] = []
            handle = mod.register_forward_pre_hook(self._make_hook(name))
            self._hooks.append(handle)

    # -- hooks ---------------------------------------------------------

    def _make_hook(self, name: str):
        def hook(_mod: nn.Module, inputs: tuple):
            if not inputs:
                return
            x = inputs[0]
            if not isinstance(x, torch.Tensor):
                return
            if not torch.is_grad_enabled():
                return
            # Only capture tensors in the autograd graph
            if not (x.requires_grad or x.grad_fn is not None):
                return
            x.retain_grad()
            step = self._current_step if self._scopes[name] == "dit" else None
            self._sample_acts[name].append((step, x))

        return hook

    # -- public API ----------------------------------------------------

    def set_current_step(self, step: int | None) -> None:
        """Tell the collector which Euler step the next forward belongs to."""
        self._current_step = step

    def begin_sample(self) -> None:
        """Reset per-sample activation state before a new forward pass."""
        for name in self._sample_acts:
            self._sample_acts[name] = []

    def compute_jacobian_sensitivity(
        self,
        actions: torch.Tensor,
        model: nn.Module | None = None,
    ) -> None:
        """Compute the exact diagonal of ``J^T J`` via looped backward.

        For a ``(B, ...)``-shaped *actions* tensor, all trailing dimensions
        are flattened into independent action outputs.  The method performs
        ``Π(trailing dims)`` backward passes — typically 7 for a 7-DoF arm.

        Parameters
        ----------
        actions : Tensor
            Differentiable output of ``forward_differentiable``.
        model : nn.Module, optional
            If provided, ``model.zero_grad(set_to_none=True)`` is called
            before each backward to avoid accumulating parameter gradients
            (they are not needed for Fisher sensitivity and waste memory).
        """
        if not (actions.requires_grad or actions.grad_fn is not None):
            logger.warning(
                "actions has no grad_fn; Fisher computation skipped for this "
                "sample.  Check that the differentiable forward path is active."
            )
            return

        batch_size = actions.shape[0]
        actions_flat = actions.reshape(batch_size, -1)
        n_action_dims = actions_flat.shape[1]

        n_layers_active = sum(1 for v in self._sample_acts.values() if v)
        logger.info(
            "Exact Jacobian: %d action dims × %d active layers.",
            n_action_dims,
            n_layers_active,
        )

        for i in range(n_action_dims):
            # Zero captured activation grads
            for step_acts in self._sample_acts.values():
                for _step, x in step_acts:
                    if x.grad is not None:
                        x.grad.zero_()

            if model is not None:
                model.zero_grad(set_to_none=True)

            grad_output = torch.zeros_like(actions_flat)
            grad_output[:, i] = 1.0
            retain = i < n_action_dims - 1
            actions_flat.backward(grad_output, retain_graph=retain)

            for name, step_acts in self._sample_acts.items():
                result = self._results[name]
                for step, x in step_acts:
                    if x.grad is None:
                        continue
                    grad_sq = x.grad.detach().float().pow(2)
                    # Average over all dims except the channel (last) dim
                    reduce = list(range(grad_sq.ndim - 1))
                    ch_sens = grad_sq.mean(dim=reduce) if reduce else grad_sq
                    if step not in result._step_accum:
                        result._step_accum[step] = torch.zeros(
                            result.in_features,
                            device=ch_sens.device,
                            dtype=ch_sens.dtype,
                        )
                    result._step_accum[step].add_(ch_sens)

        for result in self._results.values():
            result._n_samples += 1

    def get_results(self) -> dict[str, LayerFisherResult]:
        """Return the accumulated Fisher results keyed by layer name."""
        return dict(self._results)

    # -- lifecycle -----------------------------------------------------

    def detach(self) -> None:
        """Remove all forward hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __enter__(self) -> FisherCollector:
        return self

    def __exit__(self, *args) -> None:
        self.detach()


# ---------------------------------------------------------------------- #
# High-level driver used by the builder                                    #
# ---------------------------------------------------------------------- #


def compute_fisher_sensitivity(
    adapter: Any,
    target_modules: list[tuple[str, str, nn.Module]],
    *,
    num_samples: int = 4,
    num_dit_steps: int = 10,
    step_aggregation: StepAggregation = "uniform",
    progress: Any | None = None,
) -> dict[str, torch.Tensor]:
    """End-to-end Fisher sensitivity on the FP model.

    Patches the scheduler for differentiable inference, runs
    ``num_samples`` forward+backward loops, and returns a dict mapping
    each layer name to its aggregated ``(in_features,)`` sensitivity
    vector.

    Only layers whose activations participate in the autograd graph
    (typically DiT layers connected through the noise leaf) will have
    non-zero sensitivity.  LLM layers whose prefix path is detached
    will have zero sensitivity — they fall back to vanilla
    ``svd_hadamard`` rotation at build time.
    """
    from qvla.build.differentiable_forward import (
        force_eager_runners,
        differentiable_inference_context,
        reset_differentiable_state,
    )
    from qvla.runtime.step_context import reset_step_counters

    engine = adapter.engine
    if engine is None:
        raise RuntimeError(
            "Adapter has no engine — call build_model() and "
            "warmup_for_calibration() first."
        )
    sched = engine.entry.scheduler
    model = engine.entry.model

    force_eager_runners(sched)

    from qvla.adapters.pi05.step_hook import patched_one_step

    with differentiable_inference_context(sched):
        with FisherCollector(target_modules, num_dit_steps) as fc:
            for i, batch in enumerate(
                adapter.iter_calibration_batches(num_samples)
            ):
                logger.info("Fisher sample %d / %d", i + 1, num_samples)
                fc.begin_sample()
                reset_differentiable_state(sched)
                reset_step_counters(model)
                model.zero_grad(set_to_none=True)

                runner = sched.expert_runner

                def _step_cb(step, _fc=fc):
                    _fc.set_current_step(int(step))

                with patched_one_step(runner, _step_cb):
                    fc.set_current_step(None)
                    actions = adapter.forward_differentiable(batch)

                fc.compute_jacobian_sensitivity(actions, model=model)

                if progress is not None:
                    progress(
                        "fisher", 0.05 + 0.95 * (i + 1) / max(1, num_samples)
                    )

            results = fc.get_results()

    aggregated: dict[str, torch.Tensor] = {}
    for name, result in results.items():
        aggregated[name] = result.aggregate(step_aggregation)

    n_nonzero = sum(
        1 for v in aggregated.values() if v.abs().sum().item() > 0
    )
    logger.info(
        "Fisher sensitivity: %d / %d layers have non-zero sensitivity.",
        n_nonzero,
        len(aggregated),
    )
    return aggregated


__all__ = [
    "FisherCollector",
    "LayerFisherResult",
    "StepAggregation",
    "compute_fisher_sensitivity",
]
