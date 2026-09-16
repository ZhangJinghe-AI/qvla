"""Fisher collectors: capture activations, run backward, accumulate sensitivity.

The class hierarchy factors the two dimensions cleanly:

* :class:`BaseFisherCollector` owns everything that is independent of
  *how* per-channel sensitivity is extracted from a forward pass —
  target registry, step tagging, per-sample bookkeeping, the outer
  ``exact`` / ``hutchinson`` backward loop, and the final
  :class:`~qvla.build.fisher.result.LayerFisherResult` output.
* :class:`InputGradFisherCollector` and
  :class:`OutputHessianFisherCollector` supply exactly two methods each:

    - ``_install_hook(name, module)`` — register the forward / pre-forward
      hook that captures the tensors the collector needs.
    - ``_extract_channel(name, capture)`` — turn one capture (plus the
      autograd state that ``backward()`` just populated) into the
      ``(in_features,)`` per-channel contribution to add into that
      layer's step bucket.

Everything is fail-loud: bad shapes, missing gradients on layers that
should have them, non-differentiable actions, etc. are surfaced as
:class:`RuntimeError` / :class:`ValueError`, never silently zeroed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from qvla.build.fisher.common import FisherMethod, select_fisher_actions
from qvla.build.fisher.result import LayerFisherResult

if TYPE_CHECKING:
    from qvla.core.pipeline import Transform

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Base class                                                                  #
# --------------------------------------------------------------------------- #


class BaseFisherCollector:
    """Common bookkeeping for every Fisher collector.

    Subclasses implement :meth:`_install_hook` (what to retain from a
    forward pass) and :meth:`_extract_channel` (how to turn one capture
    + the backward it just witnessed into a ``(in_features,)`` vector).
    They do *not* touch the accumulator, the step key, or the backward
    loop — that keeps InputGrad and OutputHessian bit-comparable modulo
    their intrinsic mathematical differences.

    Parameters
    ----------
    targets:
        ``list of (name, scope, module)`` — matches
        :func:`qvla.runtime.wrap.list_target_modules`.
    num_dit_steps:
        Informational upper bound on denoising steps (buckets are created
        lazily as steps arrive, so a smaller-than-``num_dit_steps`` run
        is still well-defined).
    action_dim:
        Number of task DoFs the calling adapter exposes.
    rotations:
        Optional per-layer rotation. When supplied, the sensitivity is
        reported in the rotated input-channel layout — needed for
        Fisher-GPTQ so Hadamard blocks don't get uniformised by an
        input-space diagonal.
    """

    def __init__(
        self,
        targets: list[tuple[str, str, nn.Module]],
        num_dit_steps: int = 10,
        rotations: dict[str, "Transform"] | None = None,
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
        self._in_features: dict[str, int] = {}
        self._results: dict[str, LayerFisherResult] = {}
        self._hooks: list[Any] = []
        self._current_step: int | None = None

        # name -> [(step_key, capture)] for the current sample.
        # ``capture`` is whatever :meth:`_install_hook` chooses to stash;
        # the base class treats it as opaque.
        self._sample_captures: dict[str, list[tuple[int | None, Any]]] = {}
        self._warned_no_grad: set[str] = set()
        self._current_batch_size: int = 0

        for name, scope, mod in targets:
            self._scopes[name] = scope
            in_f = int(getattr(mod, "in_features", 0))
            if in_f == 0:
                w = getattr(mod, "weight", None)
                in_f = int(w.shape[-1]) if w is not None else 0
            self._in_features[name] = in_f
            self._results[name] = LayerFisherResult(
                name=name, scope=scope, in_features=in_f
            )
            self._sample_captures[name] = []
            self._hooks.append(self._install_hook(name, mod))

    # -- subclass contract -------------------------------------------------

    def _install_hook(self, name: str, module: nn.Module) -> Any:
        raise NotImplementedError

    def _extract_channel(
        self, name: str, capture: Any
    ) -> torch.Tensor | None:
        """Return this layer's per-channel contribution after ``backward()``.

        Return ``None`` (never a zero tensor and never raise) if the
        layer is not on the action graph so the base class can log-once
        and skip it uniformly across collectors.
        """
        raise NotImplementedError

    # -- step / sample lifecycle -------------------------------------------

    def set_current_step(self, step: int | None) -> None:
        """Tell the collector which Euler step the next forward belongs to."""
        self._current_step = step

    def begin_sample(self) -> None:
        """Reset per-sample activation state before a new forward pass."""
        for name in self._sample_captures:
            self._sample_captures[name] = []

    def _step_key_for(self, name: str) -> int | None:
        return self._current_step if self._scopes[name] == "dit" else None

    def _record_capture(self, name: str, capture: Any) -> None:
        """Subclass hook helper: append a capture under the current step key."""
        self._sample_captures[name].append((self._step_key_for(name), capture))

    @staticmethod
    def _capture_tensor_with_grad(capture: Any) -> torch.Tensor | None:
        """Return the captured tensor whose ``.grad`` must be reset between backwards.

        Both concrete collectors stash a small dict; the tensor that
        received ``retain_grad()`` lives under key ``"grad_target"``.
        """
        if isinstance(capture, dict):
            return capture.get("grad_target")
        return None

    # -- main backward loop ------------------------------------------------

    def compute_jacobian_sensitivity(
        self,
        actions: torch.Tensor,
        model: nn.Module | None = None,
        *,
        action_timestep: str = "all",
        method: FisherMethod = "exact",
        hutchinson_probes: int = 8,
    ) -> None:
        """Run the backward estimator over ``actions`` and accumulate.

        ``method="exact"`` costs one backward per action DoF and yields
        the true diagonal of the layer-local sensitivity. ``"hutchinson"``
        uses ``hutchinson_probes`` Rademacher vectors — same expectation,
        much cheaper when the action space is wide.
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
        batch_size = int(actions.shape[0])
        actions_flat = actions.reshape(batch_size, -1)
        n_action_dims = int(actions_flat.shape[1])
        # Some subclasses (InputGrad) need the outer batch size at
        # per-capture extraction time to reproduce the historical
        # ``alpha = batch_size × scale`` weighting.
        self._current_batch_size = batch_size

        n_layers_active = sum(1 for v in self._sample_captures.values() if v)
        if n_layers_active == 0:
            raise RuntimeError(
                "Fisher backward: no target layer captured activations with a "
                f"grad path ({len(self._sample_captures)} targets registered). "
                "Hooks must see differentiable Linear outputs during "
                "forward_differentiable()."
            )
        logger.info(
            "Fisher backward (%s, %s): %d action dims × %d active layers "
            "(action_timestep=%s, actions_shape=%s).",
            self.__class__.__name__,
            method,
            n_action_dims,
            n_layers_active,
            action_timestep,
            tuple(actions.shape),
        )

        def _accumulate_all(scale: float) -> None:
            for name, captures in self._sample_captures.items():
                if not captures:
                    continue
                result = self._results[name]
                for step, cap in captures:
                    contrib = self._extract_channel(name, cap)
                    if contrib is None:
                        if name not in self._warned_no_grad:
                            logger.info(
                                "Layer %r (step=%r) not on the action graph; "
                                "Fisher sensitivity stays zero.",
                                name,
                                step,
                            )
                            self._warned_no_grad.add(name)
                        # Materialise a zero bucket so downstream aggregation
                        # sees a consistent set of keys.
                        if step not in result._step_accum:
                            result._step_accum[step] = torch.zeros(
                                result.in_features, dtype=torch.float32
                            )
                        continue
                    if step not in result._step_accum:
                        result._step_accum[step] = torch.zeros(
                            result.in_features,
                            device=contrib.device,
                            dtype=contrib.dtype,
                        )
                    result._step_accum[step].add_(contrib, alpha=float(scale))

        def _clear_autograd_state() -> None:
            # Clear x.grad / y.grad on every captured tensor so the next
            # backward accumulates a clean vector, and drop parameter grads
            # (we only need activation gradients).
            for captures in self._sample_captures.values():
                for _step, cap in captures:
                    tensor = self._capture_tensor_with_grad(cap)
                    if tensor is not None and tensor.grad is not None:
                        tensor.grad.zero_()
            if model is not None:
                model.zero_grad(set_to_none=True)

        if method == "exact":
            for i in range(n_action_dims):
                _clear_autograd_state()
                grad_output = torch.zeros_like(actions_flat)
                grad_output[:, i] = 1.0
                retain = i < n_action_dims - 1
                actions_flat.backward(grad_output, retain_graph=retain)
                _accumulate_all(scale=1.0)
        else:
            # Hutchinson: E[(v^T J)^2] = diag(J^T J), v_i in {-1, +1}.
            for probe_idx in range(hutchinson_probes):
                _clear_autograd_state()
                signs = torch.randint(
                    0, 2, (n_action_dims,),
                    device=actions_flat.device, dtype=torch.int64,
                ).to(actions_flat.dtype)
                signs = signs.mul_(2.0).sub_(1.0)
                grad_output = signs.unsqueeze(0).expand(batch_size, -1)
                retain = probe_idx < hutchinson_probes - 1
                actions_flat.backward(grad_output, retain_graph=retain)
                _accumulate_all(scale=1.0 / hutchinson_probes)

        for name, result in self._results.items():
            if self._sample_captures.get(name):
                result._n_samples += batch_size

    # -- results / lifecycle -----------------------------------------------

    def get_results(self) -> dict[str, LayerFisherResult]:
        return dict(self._results)

    def detach(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __enter__(self) -> "BaseFisherCollector":
        return self

    def __exit__(self, *args) -> None:
        self.detach()




# --------------------------------------------------------------------------- #
# Input-gradient collector — historical QVLA behaviour                        #
# --------------------------------------------------------------------------- #


class InputGradFisherCollector(BaseFisherCollector):
    """Per-input-channel Fisher information via ``(∂a/∂x)²``.

    For layer :math:`l` with input activation :math:`x_l`:

    .. math::

        F_c \\;=\\; \\sum_i \\mathbb{E}
                    \\bigl[(\\partial a_i / \\partial x_{l,c})^2\\bigr]

    which is exactly the QVLA-paper diagonal Fisher of the action-space
    Jacobian, in the input channel layout. Optional rotation is applied
    to the *gradient* prior to the per-channel reduction — that reports
    sensitivity in the rotated channel space (required by Fisher-GPTQ so
    Hadamard blocks stay informative).
    """

    def _install_hook(self, name: str, module: nn.Module) -> Any:
        def hook(_mod: nn.Module, inputs: tuple) -> None:
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
            # ``grad_target`` names the tensor whose .grad we need to reset
            # between backwards (see BaseFisherCollector._clear_autograd_state).
            self._record_capture(name, {"x": x, "grad_target": x})

        return module.register_forward_pre_hook(hook)

    def _extract_channel(
        self, name: str, capture: Any
    ) -> torch.Tensor | None:
        x = capture["x"]
        grad_x = x.grad
        if grad_x is None:
            return None
        grad_f = grad_x.detach().float()
        rotation = self._rotations.get(name)
        if rotation is not None and not rotation.is_identity:
            grad_f = rotation.apply(grad_f)
        grad_sq = grad_f.pow(2)
        reduce = list(range(grad_sq.ndim - 1))
        ch = grad_sq.mean(dim=reduce) if reduce else grad_sq
        # ``grad.mean(non-channel)`` collapses the outer batch axis to a
        # per-sample-per-token mean. Multiplying by the *outer* batch size
        # (from actions.shape[0], not x.shape[0], because some layers see
        # flattened tokens) keeps the driver's ``_n_samples += batch_size``
        # convention numerically identical to the pre-refactor collector.
        return ch.mul_(float(self._current_batch_size))


# --------------------------------------------------------------------------- #
# Output-gradient collector — token-weighted input Hessian diagonal           #
# --------------------------------------------------------------------------- #


class OutputHessianFisherCollector(BaseFisherCollector):
    """Policy-aware rectified Hessian, HBVLA-style.

    For layer :math:`l` with input :math:`x_l\\in\\mathbb{R}^{B\\times T\\times K}`
    and output :math:`y_l=x_l W^\\top\\in\\mathbb{R}^{B\\times T\\times N}`, the
    backward populates :math:`\\partial a_i / \\partial y_l`. We form a
    non-negative *token importance*

    .. math::

        s_{b,t} \\;=\\; \\sum_i (\\partial a_i / \\partial y_{b,t})^2

    (summed over output channels and, in ``exact`` mode, over action
    dims across the outer loop). The channel-level importance is the
    token-weighted diagonal of the input-side Hessian:

    .. math::

        F_c \\;=\\; \\sum_{b,t} s_{b,t}\\; x_{b,t,c}^2

    Interpretation: channels where the model actually places
    action-critical energy at critical token positions score highest —
    which is exactly what SmoothQuant / Fisher-GPTQ want to protect.
    Transform, when supplied, rotates :math:`x` before the squared sum so
    the result lives in the rotated channel layout, matching the
    contract of :class:`InputGradFisherCollector`.
    """

    @staticmethod
    def _unwrap_linear_output(name: str, output: Any) -> torch.Tensor:
        """phyai ``LinearBase`` returns ``(y, bias)``; ``nn.Linear`` returns ``y``."""
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
            return output[0]
        raise RuntimeError(
            f"OutputHessianFisherCollector: layer {name!r} unexpected forward "
            f"output type {type(output).__name__}; expected Tensor or "
            f"(Tensor, ...)."
        )

    def _install_hook(self, name: str, module: nn.Module) -> Any:
        def hook(_mod: nn.Module, inputs: tuple, output: Any) -> None:
            if not inputs:
                raise RuntimeError(
                    f"OutputHessianFisherCollector: layer {name!r} forward "
                    "hook received empty inputs."
                )
            x = inputs[0]
            if not isinstance(x, torch.Tensor):
                raise RuntimeError(
                    f"OutputHessianFisherCollector: layer {name!r} expected "
                    f"Tensor input, got {type(x).__name__}."
                )
            y = self._unwrap_linear_output(name, output)
            if not torch.is_grad_enabled():
                return
            if not (y.requires_grad or y.grad_fn is not None):
                return
            y.retain_grad()
            # ``x`` is kept only for its values (we don't need its grad),
            # ``y`` owns the retained grad that backward will populate.
            self._record_capture(
                name, {"x": x, "y": y, "grad_target": y}
            )

        return module.register_forward_hook(hook)

    def _extract_channel(
        self, name: str, capture: Any
    ) -> torch.Tensor | None:
        y = capture["y"]
        x = capture["x"]
        grad_y = y.grad
        if grad_y is None:
            return None
        expected_k = self._in_features[name]
        if int(x.shape[-1]) != expected_k:
            raise RuntimeError(
                f"OutputHessianFisherCollector: layer {name!r} captured x with "
                f"last dim {int(x.shape[-1])} but expected in_features={expected_k}."
            )
        if int(y.shape[-1]) == 0:
            raise RuntimeError(
                f"OutputHessianFisherCollector: layer {name!r} captured y with "
                "zero output channels."
            )
        # (..., N) → (..., ) after summing over output channels.
        s_t = grad_y.detach().float().pow(2).sum(dim=-1)          # (..., T) or (...)
        s_flat = s_t.reshape(-1)                                  # (BT,)
        x_flat = x.detach().to(torch.float32).reshape(-1, expected_k)
        if int(x_flat.shape[0]) != int(s_flat.shape[0]):
            raise RuntimeError(
                f"OutputHessianFisherCollector: layer {name!r} x/y token counts "
                f"disagree — x has {int(x_flat.shape[0])} tokens, y has "
                f"{int(s_flat.shape[0])}. This means the forward hook fired "
                "on tensors from different iterations; check the adapter's "
                "differentiable forward for in-place aliasing."
            )
        rotation = self._rotations.get(name)
        if rotation is not None and not rotation.is_identity:
            x_flat = rotation.apply(x_flat)
        return (s_flat.unsqueeze(1) * x_flat.pow(2)).sum(dim=0)   # (K,)


__all__ = [
    "BaseFisherCollector",
    "InputGradFisherCollector",
    "OutputHessianFisherCollector",
]
