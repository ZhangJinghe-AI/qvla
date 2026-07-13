"""Activation capture for calibration.

The builder needs three statistics per target layer:

1. **Per-channel mean activation covariance** ``X.T @ X / N`` — fed into the
   svd-Hadamard rotation fit.
2. **Hessian** ``X.T @ X`` — the same matrix unnormalized, fed into GPTQ.
3. **Per-step activation amax** — for DiT ``per_step`` / ``static`` modes, a
   second pass records amax on ``rotation.apply(x)`` (see
   :class:`RotatedActivationCollector`).

:class:`RotatedActivationCollector` owns forward pre-hooks on each target layer;
the hook keeps a streaming average so calibration RAM stays bounded regardless
of how many tokens the loop sees.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from qvla.core.quantize import (
    channel_percentile_amax,
    token_position_percentile_amax,
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


@dataclass
class LayerStats:
    """Streaming statistics for one linear layer."""

    in_features: int

    # X.T @ X accumulated; divided by ``n_tokens`` on demand.
    xtx: torch.Tensor = field(default_factory=lambda: torch.zeros(0))
    n_tokens: int = 0

    # Per-step amax over channels. Only populated when ``num_steps > 1``.
    # Shape: (num_steps, in_features). Updated with `max` in-place.
    per_step_channel_amax: torch.Tensor | None = None
    # Static (cross-step) amax — always populated, used for rotation / perm fit.
    static_channel_amax: torch.Tensor = field(default_factory=lambda: torch.zeros(0))
    # Static per-token amax — shape (num_tokens,), max over channels per position.
    static_token_amax: torch.Tensor = field(default_factory=lambda: torch.zeros(0))
    # Per-step per-token amax. Only populated when ``num_steps > 1``.
    # Shape: (num_steps, num_tokens). Updated with `max` in-place.
    per_step_token_amax: torch.Tensor | None = None
    # Raw |activation| samples (CPU) for inner percentile (channel + token).
    static_abs_samples: list[torch.Tensor] = field(default_factory=list)
    per_step_abs_samples: list[list[torch.Tensor]] | None = None

    def init_phase1(self, in_features: int, num_steps: int, device, dtype) -> None:
        self.in_features = in_features
        self.xtx = torch.zeros(in_features, in_features, device=device, dtype=torch.float64)
        self.static_channel_amax = torch.zeros(in_features, device=device, dtype=torch.float32)
        self.static_token_amax = torch.zeros(0, device=device, dtype=torch.float32)
        self.static_abs_samples = []
        self.per_step_abs_samples = None
        self.per_step_token_amax = None
        self.per_step_channel_amax = None
        self.n_tokens = 0

    def update_phase1(self, x: torch.Tensor, *, step: int | None) -> None:
        """Accumulate ``xtx`` and ``static_channel_amax`` from one mini-batch of activations."""
        flat = (
            x.reshape(-1, self.in_features)
            .detach()
            .to(device=self.xtx.device, dtype=torch.float32)
        )
        if flat.shape[0] == 0:
            return
        self.xtx.add_(flat.T.to(torch.float64) @ flat.to(torch.float64))
        self.n_tokens += int(flat.shape[0])
        amax_now = flat.abs().amax(dim=0)
        torch.maximum(self.static_channel_amax, amax_now, out=self.static_channel_amax)

    def init_phase2(self, in_features: int, num_steps: int, device) -> None:
        """Allocate buffers for phase-2 collection (``xtx`` + act amax tables)."""
        self.in_features = in_features
        self.xtx = torch.zeros(in_features, in_features, device=device, dtype=torch.float64)
        self.n_tokens = 0
        self.static_channel_amax = torch.zeros(in_features, device=device, dtype=torch.float32)
        self.static_token_amax = torch.zeros(0, device=device, dtype=torch.float32)
        self.static_abs_samples = []
        if num_steps > 1:
            self.per_step_channel_amax = torch.zeros(
                num_steps, in_features, device=device, dtype=torch.float32
            )
            self.per_step_abs_samples = [[] for _ in range(num_steps)]
            self.per_step_token_amax = torch.zeros(
                num_steps, 0, device=device, dtype=torch.float32
            )
        else:
            self.per_step_channel_amax = None
            self.per_step_abs_samples = None
            self.per_step_token_amax = None

    def update_phase2(self, x: torch.Tensor, *, step: int | None) -> None:
        """Accumulate ``xtx``, ``static_channel_amax``, and ``per_step_channel_amax``."""
        flat = (
            x.reshape(-1, self.in_features)
            .detach()
            .to(device=self.xtx.device, dtype=torch.float32)
        )
        if flat.shape[0] == 0:
            return
        self.xtx.add_(flat.T.to(torch.float64) @ flat.to(torch.float64))
        self.n_tokens += int(flat.shape[0])
        abs_flat = flat.abs()
        amax_now = abs_flat.amax(dim=0)
        torch.maximum(self.static_channel_amax, amax_now, out=self.static_channel_amax)
        self.static_abs_samples.append(abs_flat.detach().cpu())
        row_amax = abs_flat.amax(dim=1)
        self.static_token_amax = _update_static_token_amax(self.static_token_amax, row_amax)
        if self.per_step_channel_amax is not None and step is not None:
            row = self.per_step_channel_amax[step]
            torch.maximum(row, amax_now, out=row)
            assert self.per_step_abs_samples is not None
            self.per_step_abs_samples[step].append(abs_flat.detach().cpu())
            self.per_step_token_amax = _update_per_step_token_amax(
                self.per_step_token_amax, step, row_amax
            )

    def act_channel_percentile_amax(
        self, percentile: float, *, step: int | None = None
    ) -> torch.Tensor:
        """Per-channel ``percentile`` of |activation| from collected samples."""
        if step is not None and self.per_step_abs_samples is not None:
            chunks = self.per_step_abs_samples[step]
        else:
            chunks = self.static_abs_samples
        if not chunks:
            raise ValueError("No activation samples collected for percentile amax.")
        data = torch.cat(chunks, dim=0)
        return channel_percentile_amax(data, percentile)

    def act_token_percentile_amax(
        self, percentile: float, *, step: int | None = None
    ) -> torch.Tensor:
        """Per-token-position ``percentile`` of |activation| from collected samples."""
        if step is not None and self.per_step_abs_samples is not None:
            chunks = self.per_step_abs_samples[step]
        else:
            chunks = self.static_abs_samples
        if not chunks:
            raise ValueError("No activation samples collected for token percentile amax.")
        lengths = {int(chunk.shape[0]) for chunk in chunks}
        if len(lengths) != 1:
            raise ValueError(
                "Per-token act scales require a fixed token count across calibration "
                f"samples; saw lengths {sorted(lengths)}."
            )
        data = torch.stack(chunks, dim=0)
        return token_position_percentile_amax(data, percentile)

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
        ls.static_channel_amax = torch.zeros(in_features, dtype=torch.float32)
        ls.static_token_amax = torch.zeros(0, dtype=torch.float32)
        ls.static_abs_samples = []
        ls.per_step_abs_samples = None
        ls.per_step_token_amax = None
        ls.per_step_channel_amax = None
        ls.n_tokens = 0
        return ls


class RotatedActivationCollector:
    """Collector on ``rotation.apply(x)``.

    * ``quant_stats=True`` (default): Hessian + per-step amax for GPTQ / act scales.
    * ``quant_stats=False``: covariance + amax for intermediate pipeline-step fitting.
    """

    def __init__(
        self,
        targets: list[tuple[str, str, nn.Module]],
        rotations: dict[str, "Rotation"],
        *,
        num_steps_by_scope: dict[str, int],
        device: str | torch.device = "cpu",
        quant_stats: bool = True,
    ) -> None:
        self.stats: dict[str, LayerStats] = {}
        self.scopes: dict[str, str] = {}
        self.rotations = rotations
        self.handles: list = []
        self._current_step: int | None = None
        self.device = torch.device(device)
        self.quant_stats = quant_stats

        for name, scope, mod in targets:
            if name not in rotations:
                raise KeyError(f"Missing rotation for layer {name!r}.")
            in_f = int(getattr(mod, "in_features"))
            ls = LayerStats(in_features=in_f)
            nsteps = num_steps_by_scope.get(scope, 1)
            if quant_stats:
                ls.init_phase2(in_f, nsteps, self.device)
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
            step = self._current_step if scope == "dit" else None
            x_rot = rotation.apply(x)
            if self.quant_stats:
                self.stats[name].update_phase2(x_rot, step=step)
            else:
                self.stats[name].update_phase1(x_rot, step=step)

        return hook

    def set_current_step(self, step: int | None) -> None:
        self._current_step = step

    def detach(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def __enter__(self) -> "RotatedActivationCollector":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.detach()


__all__ = ["LayerStats", "RotatedActivationCollector"]
