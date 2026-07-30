"""Fisher sensitivity analysis for policy-aware rotation.

Computes the per-layer per-channel Fisher information diagonal by constructing
the *exact* Jacobian of the action output w.r.t. each target linear layer's
input activation.  For each action dimension ``a_i`` and hidden channel ``c``:

    p_l[c] = Σ_i  E[ (∂a_i / ∂h_l[:, c])² ]

Because the action space is low-dimensional (typically 7-DoF), we loop over
action dimensions and do one backward pass each — no random projections.
For chunked actions ``(B, T, A)``, ``fisher_action_timestep`` can restrict the
Jacobian to one timestep (``last`` / index) so the loop is ``A`` instead of
``T * A``.

For DiT layers the sensitivity is collected **per denoising step** so that
downstream aggregation can weight steps differently.  For LLM layers a single
(step-agnostic) sensitivity vector is stored.

Step-wise aggregation
---------------------
``StepAggregation`` controls how per-step DiT sensitivities are collapsed
into a single vector consumed by the rotation builder.  Currently supported:

* ``"uniform"`` — equal weight for every step.
* ``"max"`` — per-channel max across steps (conservative for step-shared weights).
* ``"late_mean"`` — mean over the second half of denoise steps.
* ``"very_late_mean"`` — mean over the last fifth of denoise steps.
* ``"weighted_linear"`` — weighted mean with ``w(s) ∝ (s+1)``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from qvla.core.rotation import Rotation

logger = logging.getLogger(__name__)


StepAggregation = Literal[
    "uniform", "max", "late_mean", "very_late_mean", "weighted_linear"
]
FisherMethod = Literal["exact", "hutchinson"]


def normalize_fisher_sensitivity(fisher: torch.Tensor) -> torch.Tensor:
    """Scale Fisher so ``mean(F) == 1``; preserves per-channel relative ratios."""
    f = fisher.to(torch.float32)
    if f.numel() == 0:
        raise ValueError("Cannot normalize empty Fisher sensitivity.")
    if f.sum().item() <= 0:
        logger.warning("All-zero Fisher sensitivity; skip normalization.")
        return f.clamp_min(1e-8)
    mean = f.mean()
    if mean.item() <= 0:
        raise ValueError("Fisher sensitivity mean must be positive.")
    return f / mean


def select_fisher_actions(
    actions: torch.Tensor,
    *,
    timestep: str = "all",
    action_dim: int,
) -> torch.Tensor:
    """Select action-chunk timesteps, then keep the first ``action_dim`` DoFs.

    ``actions`` must be ``(B, T, A)`` where ``A`` is the model's native
    action width (e.g. ``max_action_dim``).  Only ``[..., :action_dim]`` is
    kept — the executed task DoFs (e.g. 7 for LIBERO), not padding dims.

    ``timestep``:
      * ``"all"`` — keep full ``(B, T, action_dim)``
      * ``"i"`` — one chunk index (0-based) → ``(B, action_dim)``
      * ``"i,j,k"`` — comma-separated chunk indices → ``(B, K, action_dim)``

        """
    if action_dim <= 0:
        raise ValueError(f"action_dim must be positive, got {action_dim}.")
    if actions.ndim != 3:
        raise ValueError(
            f"Fisher actions must be (B, T, A), got shape {tuple(actions.shape)}."
        )
    native_a = actions.shape[-1]
    if action_dim > native_a:
        raise ValueError(
            f"action_dim={action_dim} exceeds native action width {native_a}."
        )
    spec = timestep.strip().lower()
    t_len = actions.shape[1]
    if spec == "all":
        selected = actions
    else:
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        if not parts:
            raise ValueError("fisher_action_timestep cannot be empty.")
        idxs: list[int] = []
        for part in parts:
            try:
                idx = int(part)
            except ValueError as e:
                raise ValueError(
                    "fisher_action_timestep must be 'all' or integer indices "
                    f"(e.g. '0,29,50'), got {timestep!r}."
                ) from e
            if idx < 0 or idx >= t_len:
                raise ValueError(
                    f"fisher_action_timestep index {idx} out of range for "
                    f"chunk length {t_len}."
                )
            idxs.append(idx)
        if len(idxs) == 1:
            selected = actions[:, idxs[0], :]
        else:
            selected = actions[:, idxs, :]
    return selected[..., :action_dim].contiguous()


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

        # LLM has a single ``None`` key — no step axis to aggregate.
        if self.scope == "llm":
            return per_step[None]

        if method == "uniform":
            stacked = torch.stack(list(per_step.values()), dim=0)
            return stacked.mean(dim=0)
        if method == "max":
            stacked = torch.stack(list(per_step.values()), dim=0)
            return stacked.max(dim=0).values

        step_keys = sorted(k for k in per_step if k is not None)
        if not step_keys:
            raise ValueError(
                f"DiT layer {self.name!r} has no integer step keys to aggregate "
                f"with method={method!r}; keys={list(per_step)}"
            )

        if method == "late_mean":
            start = len(step_keys) // 2
            stacked = torch.stack([per_step[k] for k in step_keys[start:]], dim=0)
            return stacked.mean(dim=0)
        if method == "very_late_mean":
            start = len(step_keys) - max(1, len(step_keys) // 5)
            stacked = torch.stack([per_step[k] for k in step_keys[start:]], dim=0)
            return stacked.mean(dim=0)
        if method == "weighted_linear":
            weights = [float(k + 1) for k in step_keys]
            total = sum(weights)
            device = per_step[step_keys[0]].device
            result = torch.zeros(self.in_features, device=device, dtype=torch.float32)
            for k, w in zip(step_keys, weights):
                result.add_(per_step[k], alpha=w / total)
            return result

        raise ValueError(f"Unknown step aggregation: {method!r}")


class FisherCollector:
    """Capture activations and compute exact Jacobian sensitivity.

    Typical usage inside the builder pipeline::

        with FisherCollector(target_modules, num_dit_steps=10) as fc:
            for batch in batches:
                fc.begin_sample()
                # ... patch expert runner to call fc.set_current_step(step) ...
                actions = adapter.forward_differentiable([batch])
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
        rotations: dict[str, Rotation] | None = None,
        *,
        action_dim: int,
    ) -> None:
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}.")
        self._action_dim = action_dim
        self._targets = targets
        self._num_dit_steps = num_dit_steps
        self._rotations: dict[str, Any] = rotations or {}

        self._scopes: dict[str, str] = {}
        self._results: dict[str, LayerFisherResult] = {}
        self._hooks: list[Any] = []
        self._current_step: int | None = None

        # name -> [(step_key, activation_tensor)] for the current sample
        self._sample_acts: dict[str, list[tuple[int | None, torch.Tensor]]] = {}
        self._warned_no_grad: set[str] = set()

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
        *,
        action_timestep: str = "all",
        method: FisherMethod = "exact",
        hutchinson_probes: int = 8,
    ) -> None:
        """Compute the exact diagonal of ``J^T J`` via looped backward.

        For a ``(B, ...)``-shaped *actions* tensor, trailing dimensions are
        flattened into independent action outputs.  With ``action_timestep``
        other than ``"all"``, a ``(B, T, A)`` chunk is reduced to one
        timestep first so only ``A`` backwards run.

        Parameters
        ----------
        actions : Tensor
            Differentiable output of ``forward_differentiable``.
        model : nn.Module, optional
            If provided, ``model.zero_grad(set_to_none=True)`` is called
            before each backward to avoid accumulating parameter gradients
            (they are not needed for Fisher sensitivity and waste memory).
        action_timestep : str
            ``"all"`` | integer index | comma-separated integer indices.
        method : {"exact", "hutchinson"}
            Fisher estimator. ``exact`` loops over action dimensions (legacy
            behaviour). ``hutchinson`` uses random Rademacher projections.
        hutchinson_probes : int
            Number of random projections when ``method="hutchinson"``.
        """
        method = str(method).strip().lower()
        if method not in ("exact", "hutchinson"):
            raise ValueError(
                f"Unknown fisher method {method!r}; expected 'exact' or 'hutchinson'."
            )
        if hutchinson_probes < 1:
            raise ValueError(
                f"hutchinson_probes must be >= 1, got {hutchinson_probes}."
            )
        actions = select_fisher_actions(
            actions,
            timestep=action_timestep,
            action_dim=self._action_dim,
        )
        if not (actions.requires_grad or actions.grad_fn is not None):
            raise RuntimeError(
                "actions has no grad_fn; Fisher cannot run. "
                "Check that the differentiable forward path is active."
            )
        batch_size = actions.shape[0]
        actions_flat = actions.reshape(batch_size, -1)
        n_action_dims = actions_flat.shape[1]

        n_layers_active = sum(1 for v in self._sample_acts.values() if v)
        logger.info(
            "Fisher Jacobian (%s): %d action dims × %d active layers "
            "(action_timestep=%s, actions_shape=%s).",
            method,
            n_action_dims,
            n_layers_active,
            action_timestep,
            tuple(actions.shape),
        )
        def _accumulate_from_current_grads(scale: float = 1.0) -> None:
            for name, step_acts in self._sample_acts.items():
                result = self._results[name]
                for step, x in step_acts:
                    grad_x = x.grad
                    if grad_x is None:
                        # e.g. pi0.5 last LLM layer o_proj/MLP: prefix prefill
                        # writes K/V before them and discards the final hidden state.
                        if name not in self._warned_no_grad:
                            logger.info(
                                "Layer %r (step=%r) not on the action graph; "
                                "Fisher sensitivity stays zero.",
                                name,
                                step,
                            )
                            self._warned_no_grad.add(name)
                        if step not in result._step_accum:
                            result._step_accum[step] = torch.zeros(
                                result.in_features,
                                device=x.device,
                                dtype=torch.float32,
                            )
                        continue
                    grad_f = grad_x.detach().float()
                    rotation = self._rotations.get(name)
                    if rotation is not None and not rotation.is_identity:
                        grad_f = rotation.apply(grad_f)
                    grad_sq = grad_f.pow(2)
                    reduce = list(range(grad_sq.ndim - 1))
                    ch_sens = grad_sq.mean(dim=reduce) if reduce else grad_sq
                    if step not in result._step_accum:
                        result._step_accum[step] = torch.zeros(
                            result.in_features,
                            device=ch_sens.device,
                            dtype=ch_sens.dtype,
                        )
                    # Weight by batch size so uneven trailing batches match
                    # the per-sample mean (B * mean_tokens == sum of means).
                    result._step_accum[step].add_(
                        ch_sens, alpha=float(batch_size) * float(scale)
                    )

        def _clear_grads() -> None:
            for step_acts in self._sample_acts.values():
                for _step, x in step_acts:
                    if x.grad is not None:
                        x.grad.zero_()
            if model is not None:
                model.zero_grad(set_to_none=True)

        if method == "exact":
            for i in range(n_action_dims):
                _clear_grads()

                grad_output = torch.zeros_like(actions_flat)
                grad_output[:, i] = 1.0
                retain = i < n_action_dims - 1
                actions_flat.backward(grad_output, retain_graph=retain)
                _accumulate_from_current_grads()
        else:
            # Hutchinson: E[(J^T v)^2] = diag(J^T J), v_i in {-1, +1}.
            for probe_idx in range(hutchinson_probes):
                _clear_grads()

                signs = torch.randint(
                    0,
                    2,
                    (n_action_dims,),
                    device=actions_flat.device,
                    dtype=torch.int64,
                ).to(actions_flat.dtype)
                signs = signs.mul_(2.0).sub_(1.0)
                grad_output = signs.unsqueeze(0).expand(batch_size, -1)
                retain = probe_idx < hutchinson_probes - 1
                actions_flat.backward(grad_output, retain_graph=retain)
                _accumulate_from_current_grads(scale=1.0 / hutchinson_probes)

        for name, result in self._results.items():
            if self._sample_acts.get(name):
                result._n_samples += batch_size

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


def resolve_fisher_action_dim(adapter: Any) -> int:
    """Task action DoF count from the model adapter (e.g. LIBERO ``action_dim=7``)."""
    cfg = getattr(adapter, "cfg", None)
    if cfg is None:
        raise ValueError(
            f"Fisher requires adapter.cfg.action_dim; "
            f"{type(adapter).__name__} has no cfg."
        )
    if not hasattr(cfg, "action_dim"):
        raise ValueError(
            f"Fisher requires adapter.cfg.action_dim; "
            f"{type(cfg).__name__} has no action_dim field."
        )
    dim = int(cfg.action_dim)
    if dim <= 0:
        raise ValueError(f"adapter.cfg.action_dim must be positive, got {dim}.")
    return dim


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
    method: FisherMethod = "exact",
    hutchinson_probes: int = 8,
    rotations: dict[str, Rotation] | None = None,
    progress: Any | None = None,
) -> dict[str, torch.Tensor]:
    """End-to-end Fisher sensitivity on the FP model.

    Patches the scheduler for differentiable inference, runs
    ``ceil(num_samples / batch_size) × noise_ensemble_k`` forward+backward
    loops (each forward may contain up to ``batch_size`` observations), and
    returns a dict mapping each layer name to its aggregated ``(in_features,)``
    sensitivity vector. ``method="exact"`` uses one backward per action dim;
    ``method="hutchinson"`` uses ``hutchinson_probes`` random projections.

    When ``rotations`` is provided, each layer's input gradient is transformed
    by ``rotation.apply(grad)`` before computing per-channel sensitivity.
    This yields Fisher in the **rotated** channel layout — needed for
    Fisher-GPTQ to avoid the diagonal-approximation uniformisation caused
    by Hadamard.

    Layers whose inputs are not connected to actions (e.g. the last LLM layer's
    ``o_proj`` and MLP in pi0.5 prefix prefill) receive zero sensitivity.
    Other LLM and DiT linears are patched for autograd; prefix K/V from the
    paligemma prefill are replayed from a per-layer tape so expert joint
    attention can backprop into earlier LLM activations.
    """
    from qvla.build.differentiable_forward import (
        force_eager_runners,
        differentiable_inference_context,
        reset_differentiable_state,
    )
    from qvla.runtime.step_context import reset_step_counters

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
    method = str(method).strip().lower()
    if method not in ("exact", "hutchinson"):
        raise ValueError(
            f"Unknown fisher method {method!r}; expected 'exact' or 'hutchinson'."
        )
    if hutchinson_probes < 1:
        raise ValueError(
            f"hutchinson_probes must be >= 1, got {hutchinson_probes}."
        )
    if not target_modules:
        logger.info("Fisher: no target modules; skipping.")
        return {}

    k = noise_ensemble_k

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

    force_eager_runners(sched)

    from qvla.adapters.pi05.step_hook import patched_one_step

    samples = list(adapter.iter_calibration_batches(num_samples))
    if len(samples) != num_samples:
        raise RuntimeError(
            f"Fisher expected {num_samples} calibration samples, "
            f"got {len(samples)} from iter_calibration_batches()."
        )

    n_forwards = (num_samples + batch_size - 1) // batch_size
    total = n_forwards * k
    done = 0
    logger.info(
        "Fisher: %d samples, batch_size=%d (%d forwards × %d noise), "
        "method=%s%s.",
        num_samples,
        batch_size,
        n_forwards,
        k,
        method,
        (
            f", probes={hutchinson_probes}"
            if method == "hutchinson"
            else ""
        ),
    )
    with differentiable_inference_context(sched):
        with FisherCollector(
            target_modules,
            num_dit_steps,
            rotations=rotations,
            action_dim=action_dim,
        ) as fc:
            for start in range(0, num_samples, batch_size):
                chunk = samples[start : start + batch_size]
                indices = list(range(start, start + len(chunk)))
                for noise_index in range(k):
                    logger.info(
                        "Fisher samples [%d:%d] / %d noise %d / %d",
                        start,
                        start + len(chunk),
                        num_samples,
                        noise_index + 1,
                        k,
                    )
                    fc.begin_sample()
                    reset_differentiable_state(sched)
                    reset_step_counters(model)
                    model.zero_grad(set_to_none=True)

                    runner = sched.expert_runner

                    def _step_cb(step, _fc=fc):
                        _fc.set_current_step(int(step))

                    with patched_one_step(runner, _step_cb):
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

            results = fc.get_results()

    aggregated: dict[str, torch.Tensor] = {}
    for name, result in results.items():
        aggregated[name] = result.aggregate(step_aggregation)

    n_nonzero = sum(
        1 for v in aggregated.values() if v.abs().sum().item() > 0
    )
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
    return aggregated


__all__ = [
    "FisherCollector",
    "FisherMethod",
    "LayerFisherResult",
    "StepAggregation",
    "compute_fisher_sensitivity",
    "normalize_fisher_sensitivity",
    "resolve_fisher_action_dim",
    "select_fisher_actions",
]
