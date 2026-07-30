"""Activation capture for calibration.

The builder needs three statistics per target layer:

1. **Per-channel mean activation covariance** ``X.T @ X / N`` — fed into the
   svd-Hadamard rotation fit.
2. **Hessian** ``X.T @ X`` — the same matrix unnormalized, fed into GPTQ.
3. **Per-step activation amax** — for DiT ``per_step`` / ``static`` modes, a
   second pass records amax on ``rotation.apply(x)`` (see
   :class:`RotatedActivationCollector`).

:class:`RotatedActivationCollector` owns forward pre-hooks on each target layer;
the hook streams ``XᵀX`` / amax. Offline act-scale modes may also stream
cross/inner channel or token amax per :class:`AmaxCollectPlan`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from qvla.core.quantize import (
    channel_percentile_amax,
    token_percentile_amax,
)

if TYPE_CHECKING:
    from qvla.core.rotation import Rotation

logger = logging.getLogger(__name__)


def _update_static_token_amax(buf: torch.Tensor, row_amax: torch.Tensor) -> torch.Tensor:
    """Streaming max for ``static_token_amax``, growing along the token axis."""
    n = int(row_amax.shape[0])
    if buf.numel() == 0:
        return row_amax.clone()
    if n > buf.numel():
        buf = torch.cat(
            [buf, torch.zeros(n - buf.numel(), device=buf.device, dtype=buf.dtype)]
        )
    torch.maximum(buf[:n], row_amax, out=buf[:n])
    return buf


def _update_per_step_token_amax(
    table: torch.Tensor, step: int, row_amax: torch.Tensor
) -> torch.Tensor:
    """Streaming max for ``per_step_token_amax``, growing along the token axis."""
    n = int(row_amax.shape[0])
    if table.shape[1] < n:
        pad = torch.zeros(
            table.shape[0],
            n - table.shape[1],
            device=table.device,
            dtype=table.dtype,
        )
        table = torch.cat([table, pad], dim=1)
    torch.maximum(table[step, :n], row_amax, out=table[step, :n])
    return table


@dataclass(frozen=True)
class AmaxCollectPlan:
    """What phase-2 must accumulate beyond the GPTQ Hessian ``XᵀX``.

    Offline act-scale builds enable the matching cross/inner + channel/token flags.
    ``dynamic`` keeps all False (Hessian only).
    """

    collect_cross_channel: bool = False
    collect_cross_token: bool = False
    collect_inner_channel: bool = False
    collect_inner_token: bool = False
    inner_percentile: float = 99.9

    @property
    def collect_cross(self) -> bool:
        return self.collect_cross_channel or self.collect_cross_token

    @property
    def collect_inner(self) -> bool:
        return self.collect_inner_channel or self.collect_inner_token


@dataclass
class LayerStats:
    """Streaming statistics for one linear layer."""

    in_features: int

    # X.T @ X accumulated; divided by ``n_tokens`` on demand.
    xtx: torch.Tensor = field(default_factory=lambda: torch.zeros(0))
    n_tokens: float = 0

    # Per-step amax over channels. Only populated when ``num_steps > 1``.
    # Shape: (num_steps, in_features). Updated with `max` in-place.
    per_step_cross_channel_amax: torch.Tensor | None = None
    # Static (cross-step) amax — always populated, used for rotation / perm fit.
    static_cross_channel_amax: torch.Tensor = field(default_factory=lambda: torch.zeros(0))
    # Static per-token amax — shape (num_tokens,), max over channels per position.
    static_cross_token_amax: torch.Tensor = field(default_factory=lambda: torch.zeros(0))
    # Per-step per-token amax. Only populated when ``num_steps > 1``.
    # Shape: (num_steps, num_tokens). Updated with `max` in-place.
    per_step_cross_token_amax: torch.Tensor | None = None
    # Per-forward token/channel percentile, then cross-forward ``max``.
    inner_percentile: float = 99.9
    static_inner_channel_amax: torch.Tensor = field(default_factory=lambda: torch.zeros(0))
    static_inner_token_amax: torch.Tensor = field(default_factory=lambda: torch.zeros(0))
    per_step_inner_channel_amax: torch.Tensor | None = None
    per_step_inner_token_amax: torch.Tensor | None = None
    amax_plan: AmaxCollectPlan = field(default_factory=AmaxCollectPlan)

    def init_phase1(self, in_features: int, num_steps: int, device, dtype) -> None:
        self.in_features = in_features
        self.xtx = torch.zeros(in_features, in_features, device=device, dtype=torch.float32)
        self.static_cross_channel_amax = torch.zeros(in_features, device=device, dtype=torch.float32)
        self.static_cross_token_amax = torch.zeros(0, device=device, dtype=torch.float32)
        self.per_step_cross_token_amax = None
        self.per_step_cross_channel_amax = None
        self.static_inner_channel_amax = torch.zeros(0, device=device, dtype=torch.float32)
        self.static_inner_token_amax = torch.zeros(0, device=device, dtype=torch.float32)
        self.per_step_inner_channel_amax = None
        self.per_step_inner_token_amax = None
        self.amax_plan = AmaxCollectPlan()
        self.n_tokens = 0

    def update_phase1(self, x: torch.Tensor, *, step: int | None, xtx_weight: float = 1.0) -> None:
        """Accumulate ``xtx`` and ``static_cross_channel_amax`` from one mini-batch of activations."""
        flat = x.reshape(-1, self.in_features).detach()
        if flat.shape[0] == 0:
            return
        # Zero-weight steps (late_mean / very_late_mean) skip the expensive gram
        # matmul entirely; the result would be a no-op anyway.
        if xtx_weight > 0:
            flat32 = flat.to(dtype=torch.float32)
            gram = (flat32.T @ flat32).to(device=self.xtx.device, dtype=self.xtx.dtype)
            self.xtx.add_(gram, alpha=xtx_weight)
            self.n_tokens += xtx_weight * flat.shape[0]
        amax_now = flat.abs().amax(dim=0).to(device=self.static_cross_channel_amax.device)
        torch.maximum(self.static_cross_channel_amax, amax_now, out=self.static_cross_channel_amax)

    def init_phase2(
        self,
        in_features: int,
        num_steps: int,
        device,
        *,
        amax_plan: AmaxCollectPlan,
    ) -> None:
        """Allocate buffers for phase-2 collection (``xtx`` + optional act amax)."""
        if amax_plan.collect_inner:
            if not (0.0 < float(amax_plan.inner_percentile) <= 100.0):
                raise ValueError(
                    f"inner_percentile must be in (0, 100], got {amax_plan.inner_percentile}."
                )
            self.inner_percentile = float(amax_plan.inner_percentile)
        self.in_features = in_features
        self.amax_plan = amax_plan
        self.xtx = torch.zeros(in_features, in_features, device=device, dtype=torch.float32)
        self.n_tokens = 0

        self.static_cross_channel_amax = torch.zeros(0, device=device, dtype=torch.float32)
        self.static_cross_token_amax = torch.zeros(0, device=device, dtype=torch.float32)
        self.static_inner_channel_amax = torch.zeros(0, device=device, dtype=torch.float32)
        self.static_inner_token_amax = torch.zeros(0, device=device, dtype=torch.float32)
        self.per_step_cross_channel_amax = None
        self.per_step_cross_token_amax = None
        self.per_step_inner_channel_amax = None
        self.per_step_inner_token_amax = None

        if amax_plan.collect_cross_channel:
            self.static_cross_channel_amax = torch.zeros(
                in_features, device=device, dtype=torch.float32
            )
            if num_steps > 1:
                self.per_step_cross_channel_amax = torch.zeros(
                    num_steps, in_features, device=device, dtype=torch.float32
                )
        if amax_plan.collect_cross_token:
            self.static_cross_token_amax = torch.zeros(0, device=device, dtype=torch.float32)
            if num_steps > 1:
                self.per_step_cross_token_amax = torch.zeros(
                    num_steps, 0, device=device, dtype=torch.float32
                )
        if amax_plan.collect_inner_channel:
            self.static_inner_channel_amax = torch.zeros(
                in_features, device=device, dtype=torch.float32
            )
            if num_steps > 1:
                self.per_step_inner_channel_amax = torch.zeros(
                    num_steps, in_features, device=device, dtype=torch.float32
                )
        if amax_plan.collect_inner_token:
            self.static_inner_token_amax = torch.zeros(0, device=device, dtype=torch.float32)
            if num_steps > 1:
                self.per_step_inner_token_amax = torch.zeros(
                    num_steps, 0, device=device, dtype=torch.float32
                )

    def update_phase2(self, x: torch.Tensor, *, step: int | None, xtx_weight: float = 1.0) -> None:
        """Accumulate ``xtx`` and any requested cross/inner amax tables."""
        flat = x.reshape(-1, self.in_features).detach()
        if flat.shape[0] == 0:
            return
        # Zero-weight steps (late_mean / very_late_mean) skip the expensive gram
        # matmul entirely; the result would be a no-op anyway.
        if xtx_weight > 0:
            flat32 = flat.to(dtype=torch.float32)
            gram = (flat32.T @ flat32).to(device=self.xtx.device, dtype=self.xtx.dtype)
            self.xtx.add_(gram, alpha=xtx_weight)
            self.n_tokens += xtx_weight * flat.shape[0]
        if not (self.amax_plan.collect_cross or self.amax_plan.collect_inner):
            return
        abs_flat = flat.abs()
        if self.amax_plan.collect_cross:
            self._update_cross_amax(abs_flat, step=step)
        if self.amax_plan.collect_inner:
            self._update_inner_amax(abs_flat, step=step)

    def _update_cross_amax(self, abs_flat: torch.Tensor, *, step: int | None) -> None:
        """Streaming true max over forwards (feeds ``act_percentile_mode=cross``)."""
        plan = self.amax_plan
        stats_device = self.xtx.device
        amax_now = None
        row_amax = None
        if plan.collect_cross_channel:
            amax_now = abs_flat.amax(dim=0).to(device=stats_device, dtype=torch.float32)
            torch.maximum(
                self.static_cross_channel_amax, amax_now, out=self.static_cross_channel_amax
            )
        if plan.collect_cross_token:
            row_amax = abs_flat.amax(dim=1).to(device=stats_device, dtype=torch.float32)
            self.static_cross_token_amax = _update_static_token_amax(
                self.static_cross_token_amax, row_amax
            )
        if self.per_step_cross_channel_amax is None and self.per_step_cross_token_amax is None:
            return
        if step is None:
            raise ValueError(
                "DiT per-step cross amax requires step_callback / set_current_step; "
                "got step=None."
            )
        num_steps = (
            self.per_step_cross_channel_amax.shape[0]
            if self.per_step_cross_channel_amax is not None
            else self.per_step_cross_token_amax.shape[0]
        )
        if step < 0 or step >= num_steps:
            raise IndexError(
                f"step={step} out of range for per-step cross amax (num_steps={num_steps})."
            )
        if self.per_step_cross_channel_amax is not None:
            assert amax_now is not None
            torch.maximum(
                self.per_step_cross_channel_amax[step],
                amax_now,
                out=self.per_step_cross_channel_amax[step],
            )
        if self.per_step_cross_token_amax is not None:
            assert row_amax is not None
            self.per_step_cross_token_amax = _update_per_step_token_amax(
                self.per_step_cross_token_amax, step, row_amax
            )

    def _update_inner_amax(self, abs_flat: torch.Tensor, *, step: int | None) -> None:
        """Per-forward channel/token percentile, then cross-forward ``max`` (inner)."""
        plan = self.amax_plan
        p = self.inner_percentile
        stats_device = self.xtx.device
        ch_q = None
        tok_q = None
        if plan.collect_inner_channel:
            ch_q = channel_percentile_amax(abs_flat, p).to(
                device=stats_device, dtype=torch.float32
            )
            torch.maximum(
                self.static_inner_channel_amax, ch_q, out=self.static_inner_channel_amax
            )
        if plan.collect_inner_token:
            # One forward → batch=1; percentile over channels at each token position.
            tok_q = token_percentile_amax(abs_flat.unsqueeze(0), p).to(
                device=stats_device, dtype=torch.float32
            )
            self.static_inner_token_amax = _update_static_token_amax(
                self.static_inner_token_amax, tok_q
            )
        if self.per_step_inner_channel_amax is None and self.per_step_inner_token_amax is None:
            return
        if step is None:
            raise ValueError(
                "DiT per-step inner amax requires step_callback / set_current_step; "
                "got step=None."
            )
        num_steps = (
            self.per_step_inner_channel_amax.shape[0]
            if self.per_step_inner_channel_amax is not None
            else self.per_step_inner_token_amax.shape[0]
        )
        if step < 0 or step >= num_steps:
            raise IndexError(
                f"step={step} out of range for per-step inner amax (num_steps={num_steps})."
            )
        if self.per_step_inner_channel_amax is not None:
            assert ch_q is not None
            torch.maximum(
                self.per_step_inner_channel_amax[step],
                ch_q,
                out=self.per_step_inner_channel_amax[step],
            )
        if self.per_step_inner_token_amax is not None:
            assert tok_q is not None
            self.per_step_inner_token_amax = _update_per_step_token_amax(
                self.per_step_inner_token_amax, step, tok_q
            )

    def act_channel_cross_amax(self, *, step: int | None = None) -> torch.Tensor:
        """Hard max over |activation| (cross mode raw amax before percentile cap)."""
        if not self.amax_plan.collect_cross_channel:
            raise RuntimeError(
                "act_channel_cross_amax requested but cross-channel amax was not collected."
            )
        if step is not None:
            if self.per_step_cross_channel_amax is None:
                raise ValueError(
                    "Requested per-step channel cross amax but num_steps==1 "
                    "(no per_step_cross_channel_amax)."
                )
            return self.per_step_cross_channel_amax[step]
        return self.static_cross_channel_amax

    def act_token_cross_amax(self, *, step: int | None = None) -> torch.Tensor:
        """Hard max over channels per token (cross mode raw amax before percentile cap)."""
        if not self.amax_plan.collect_cross_token:
            raise RuntimeError(
                "act_token_cross_amax requested but cross-token amax was not collected."
            )
        if step is not None:
            if self.per_step_cross_token_amax is None:
                raise ValueError(
                    "Requested per-step token cross amax but num_steps==1 "
                    "(no per_step_cross_token_amax)."
                )
            return self.per_step_cross_token_amax[step]
        return self.static_cross_token_amax

    def act_channel_inner_amax(self, *, step: int | None = None) -> torch.Tensor:
        """Per-forward channel percentile, then cross-forward max (inner)."""
        if not self.amax_plan.collect_inner_channel:
            raise RuntimeError(
                "act_channel_inner_amax requested but inner-channel amax was not collected."
            )
        if step is not None:
            if self.per_step_inner_channel_amax is None:
                raise ValueError(
                    "Requested per-step channel inner amax but num_steps==1 "
                    "(no per_step_inner_channel_amax)."
                )
            return self.per_step_inner_channel_amax[step]
        return self.static_inner_channel_amax

    def act_token_inner_amax(self, *, step: int | None = None) -> torch.Tensor:
        """Per-forward token percentile, then cross-forward max (inner)."""
        if not self.amax_plan.collect_inner_token:
            raise RuntimeError(
                "act_token_inner_amax requested but inner-token amax was not collected."
            )
        if step is not None:
            if self.per_step_inner_token_amax is None:
                raise ValueError(
                    "Requested per-step token inner amax but num_steps==1 "
                    "(no per_step_inner_token_amax)."
                )
            return self.per_step_inner_token_amax[step]
        return self.static_inner_token_amax

    def covariance(self) -> torch.Tensor:
        """Return ``X.T @ X / n_tokens`` (float32)."""
        if self.n_tokens == 0:
            return torch.zeros(self.in_features, self.in_features, dtype=torch.float32)
        return (self.xtx / max(1, self.n_tokens)).to(torch.float32)

    def hessian(self) -> torch.Tensor:
        """Return ``X.T @ X`` (float32) — same matrix as covariance, unnormalized."""
        return self.xtx.to(torch.float32)

    @staticmethod
    def empty(in_features: int) -> "LayerStats":
        """Placeholder when phase 1 did not run for a layer."""
        ls = LayerStats(in_features=in_features)
        ls.xtx = torch.zeros(0)
        ls.static_cross_channel_amax = torch.zeros(in_features, dtype=torch.float32)
        ls.static_cross_token_amax = torch.zeros(0, dtype=torch.float32)
        ls.static_inner_channel_amax = torch.zeros(0, dtype=torch.float32)
        ls.static_inner_token_amax = torch.zeros(0, dtype=torch.float32)
        ls.per_step_inner_channel_amax = None
        ls.per_step_inner_token_amax = None
        ls.per_step_cross_token_amax = None
        ls.per_step_cross_channel_amax = None
        ls.amax_plan = AmaxCollectPlan()
        ls.n_tokens = 0
        return ls


class RotatedActivationCollector:
    """Collector on ``rotation.apply(x)``.

    * ``quant_stats=True``: Hessian, plus optional cross/inner amax per
      ``amax_plan_by_scope``. Requires a plan for every target scope.
    * ``quant_stats=False``: covariance + amax for intermediate pipeline-step fitting.

    When ``noise_ensemble_k > 1``, LLM layer hooks are skipped for
    ``noise_index > 0`` because LLM activations are diffusion-noise-invariant;
    collecting once is enough.
    """

    def __init__(
        self,
        targets: list[tuple[str, str, nn.Module]],
        rotations: dict[str, "Rotation"],
        *,
        num_steps_by_scope: dict[str, int],
        device: str | torch.device = "cpu",
        quant_stats: bool = True,
        amax_plan_by_scope: dict[str, AmaxCollectPlan] | None = None,
        noise_ensemble_k: int = 1,
        dit_step_weights: dict[int, float] | None = None,
    ) -> None:
        self.stats: dict[str, LayerStats] = {}
        self.scopes: dict[str, str] = {}
        self.rotations = rotations
        self.handles: list = []
        self._current_step: int | None = None
        self._noise_index: int = 0
        self._noise_ensemble_k: int = max(1, noise_ensemble_k)
        self._dit_step_weights: dict[int, float] | None = dit_step_weights
        self.device = torch.device(device)
        self.quant_stats = quant_stats

        if quant_stats:
            if amax_plan_by_scope is None:
                raise ValueError(
                    "quant_stats=True requires amax_plan_by_scope "
                    "(scope -> AmaxCollectPlan)."
                )

        for name, scope, mod in targets:
            if name not in rotations:
                raise KeyError(f"Missing rotation for layer {name!r}.")
            in_f = int(getattr(mod, "in_features"))
            ls = LayerStats(in_features=in_f)
            nsteps = num_steps_by_scope.get(scope, 1)
            if quant_stats:
                ls.init_phase2(
                    in_f,
                    nsteps,
                    self.device,
                    amax_plan=amax_plan_by_scope[scope],
                )
            else:
                ls.init_phase1(in_f, nsteps, self.device, torch.float32)
            self.stats[name] = ls
            self.scopes[name] = scope
            handle = mod.register_forward_pre_hook(
                self._make_hook(name), with_kwargs=False
            )
            self.handles.append(handle)

    def _make_hook(self, name: str):
        rotation = self.rotations[name]

        def hook(_mod: nn.Module, inputs: tuple):
            if not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x):
                return
            scope = self.scopes[name]
            # LLM activations are diffusion-noise-invariant; collect once.
            if scope != "dit" and self._noise_ensemble_k > 1 and self._noise_index > 0:
                return
            step = self._current_step if scope == "dit" else None
            xtx_weight = 1.0
            if scope == "dit" and self._dit_step_weights is not None and step is not None:
                xtx_weight = self._dit_step_weights.get(step, 1.0)
            x_rot = rotation.apply(x)
            if self.quant_stats:
                self.stats[name].update_phase2(x_rot, step=step, xtx_weight=xtx_weight)
            else:
                self.stats[name].update_phase1(x_rot, step=step, xtx_weight=xtx_weight)

        return hook

    def set_current_step(self, step: int | None) -> None:
        self._current_step = step

    def set_noise_index(self, idx: int) -> None:
        self._noise_index = idx

    def detach(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def __enter__(self) -> "RotatedActivationCollector":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.detach()


__all__ = ["AmaxCollectPlan", "LayerStats", "RotatedActivationCollector"]
