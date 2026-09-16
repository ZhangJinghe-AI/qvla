"""Activation capture for calibration.

The builder needs three statistics per target layer:

1. **Per-channel mean activation covariance** ``X.T @ X / N`` — fed into the
   svd-Hadamard rotation fit.
2. **Hessian** ``X.T @ X`` — the same matrix unnormalized, fed into GPTQ.
3. **Per-step activation amax** — for DiT ``per_step`` / ``static`` modes.

:class:`RotatedActivationCollector` owns forward pre-hooks on each target
layer. Each hook applies ``transform.apply(x)`` (the full pipeline including
optional clip) then streams ``XᵀX`` / amax per :class:`AmaxCollectPlan`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from qvla.core.clip import (
    channel_outlier_bulk_kappa_amax,
    channel_outlier_mean_std_amax,
    layer_outlier_mean_std_amax,
    scheduled_outlier_std_k,
    std_k_schedule_bounds,
    selective_channel_outlier_mean_std_amax,
)
from qvla.core.quantize import (
    channel_percentile_amax,
    token_percentile_amax,
)

if TYPE_CHECKING:
    from qvla.core.pipeline import Transform

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
    """Complete specification of what a calibration pass must collect.

    Each flag controls one data stream:

    * ``collect_hessian`` — allocate and accumulate the K×K ``XᵀX`` matrix
      (needed for GPTQ weight quantisation in the final pass, not needed
      for pipeline-build passes that only fit clip/smooth/perm).
    * ``collect_cross_*`` / ``collect_inner_*`` — per-channel or per-token
      activation magnitude statistics for offline act-scale tables.
    * ``collect_adaptive_inner_channel`` — buffer |x| tokens for one-shot
      tip-clip amax (κ×P_β or mean+k·std).
    """

    collect_hessian: bool = True
    collect_cross_channel: bool = False
    collect_cross_token: bool = False
    collect_inner_channel: bool = False
    collect_inner_token: bool = False
    collect_adaptive_inner_channel: bool = False
    inner_percentile: float = 99.9
    outlier_kappa: float = 0.0
    outlier_bulk_percentile: float = 95.0
    outlier_std_k: float = 0.0
    outlier_std_k_down: float = 0.0
    outlier_std_k_up: float = 0.0
    outlier_selective_channels: bool = False
    outlier_global: bool = False
    # DiT denoise: treat sequence position 0 as rest (GR00T state token).
    outlier_skip_first_token: bool = False

    @property
    def collect_cross(self) -> bool:
        return self.collect_cross_channel or self.collect_cross_token

    @property
    def collect_inner(self) -> bool:
        return (
            self.collect_inner_channel
            or self.collect_inner_token
            or self.collect_adaptive_inner_channel
        )


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
    # Fixed-percentile inner: per-forward percentile then streaming max.
    # Adaptive: buffer all |x| chunks, one global adaptive amax on read.
    inner_percentile: float = 99.9
    static_inner_channel_amax: torch.Tensor = field(default_factory=lambda: torch.zeros(0))
    static_inner_token_amax: torch.Tensor = field(default_factory=lambda: torch.zeros(0))
    per_step_inner_channel_amax: torch.Tensor | None = None
    per_step_inner_token_amax: torch.Tensor | None = None
    # Which denoise steps actually ran this layer (integer step index).
    # Prefix forwards (step=None) copy amax into every row but do not
    # set these bits, so missing step_callback cannot look like full coverage.
    # Shape: (num_steps,) bool. Only allocated with per-step cross buffers.
    per_step_seen: torch.Tensor | None = None
    amax_plan: AmaxCollectPlan = field(default_factory=AmaxCollectPlan)
    # Adaptive tip-clip: concatenated |x| (LLM, or DiT prefix-only ``step=None``).
    # Denoise-loop layers with ``num_steps>1`` use ``_adaptive_tip_chunks_by_step``.
    _adaptive_tip_chunks: list[torch.Tensor] = field(default_factory=list)
    _adaptive_rest_amax: torch.Tensor | None = None
    _adaptive_tip_chunks_by_step: list[list[torch.Tensor]] = field(default_factory=list)
    _adaptive_rest_amax_by_step: list[torch.Tensor | None] = field(default_factory=list)
    _adaptive_channel_finalized: bool = False
    _collect_hessian: bool = field(default=True, repr=False)
    _adaptive_token_keep: torch.Tensor | None = field(default=None, repr=False)

    def init(
        self,
        in_features: int,
        num_steps: int,
        device,
        *,
        amax_plan: AmaxCollectPlan,
    ) -> None:
        """Allocate buffers based on ``amax_plan``."""
        if amax_plan.collect_inner:
            if (
                not amax_plan.collect_adaptive_inner_channel
                and not (0.0 < float(amax_plan.inner_percentile) <= 100.0)
            ):
                raise ValueError(
                    f"inner_percentile must be in (0, 100], got {amax_plan.inner_percentile}."
                )
            self.inner_percentile = float(amax_plan.inner_percentile)
        self.in_features = in_features
        self.amax_plan = amax_plan

        self._collect_hessian = bool(amax_plan.collect_hessian)
        if self._collect_hessian:
            self.xtx = torch.zeros(
                in_features, in_features, device=device, dtype=torch.float32
            )
        else:
            self.xtx = torch.zeros(0, device=device, dtype=torch.float32)
        self.n_tokens = 0

        self.static_cross_channel_amax = torch.zeros(0, device=device, dtype=torch.float32)
        self.static_cross_token_amax = torch.zeros(0, device=device, dtype=torch.float32)
        self.static_inner_channel_amax = torch.zeros(0, device=device, dtype=torch.float32)
        self.static_inner_token_amax = torch.zeros(0, device=device, dtype=torch.float32)
        self.per_step_cross_channel_amax = None
        self.per_step_cross_token_amax = None
        self.per_step_inner_channel_amax = None
        self.per_step_inner_token_amax = None
        self.per_step_seen = None
        self._adaptive_tip_chunks = []
        self._adaptive_rest_amax = None
        self._adaptive_tip_chunks_by_step = []
        self._adaptive_rest_amax_by_step = []
        self._adaptive_channel_finalized = False
        self._adaptive_token_keep: torch.Tensor | None = None

        if amax_plan.collect_cross_channel:
            self.static_cross_channel_amax = torch.zeros(
                in_features, device=device, dtype=torch.float32
            )
            if num_steps > 1:
                self.per_step_cross_channel_amax = torch.zeros(
                    num_steps, in_features, device=device, dtype=torch.float32
                )
                self.per_step_seen = torch.zeros(
                    num_steps, device=device, dtype=torch.bool
                )
        if amax_plan.collect_cross_token:
            self.static_cross_token_amax = torch.zeros(0, device=device, dtype=torch.float32)
            if num_steps > 1:
                self.per_step_cross_token_amax = torch.zeros(
                    num_steps, 0, device=device, dtype=torch.float32
                )
                if self.per_step_seen is None:
                    self.per_step_seen = torch.zeros(
                        num_steps, device=device, dtype=torch.bool
                    )
        if amax_plan.collect_inner_channel or amax_plan.collect_adaptive_inner_channel:
            self.static_inner_channel_amax = torch.zeros(
                in_features, device=device, dtype=torch.float32
            )
            if amax_plan.collect_adaptive_inner_channel and num_steps > 1:
                self._adaptive_tip_chunks_by_step = [[] for _ in range(num_steps)]
                self._adaptive_rest_amax_by_step = [None] * num_steps
            elif num_steps > 1:
                self.per_step_inner_channel_amax = torch.zeros(
                    num_steps, in_features, device=device, dtype=torch.float32
                )
        if amax_plan.collect_inner_token:
            self.static_inner_token_amax = torch.zeros(0, device=device, dtype=torch.float32)
            if num_steps > 1:
                self.per_step_inner_token_amax = torch.zeros(
                    num_steps, 0, device=device, dtype=torch.float32
                )

    def update(self, x: torch.Tensor, *, step: int | None, xtx_weight: float = 1.0) -> None:
        """Accumulate ``xtx`` and any requested cross/inner amax tables."""
        flat = x.reshape(-1, self.in_features).detach()
        if flat.shape[0] == 0:
            return
        if self._collect_hessian:
            if xtx_weight > 0:
                flat32 = flat.to(dtype=torch.float32)
                gram = (flat32.T @ flat32).to(device=self.xtx.device, dtype=self.xtx.dtype)
                self.xtx.add_(gram, alpha=xtx_weight)
                self.n_tokens += xtx_weight * flat.shape[0]
        else:
            self.n_tokens += flat.shape[0]
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
        if step is None:
            # Cached cross-attn K/V is reused at every denoise step, so write
            # the same amax into every row. p-mean of identical rows equals
            # the prefix absmax. Do not mark per_step_seen: that flag is only
            # for real denoise-step callbacks.
            if self.per_step_cross_channel_amax is not None:
                if amax_now is None:
                    raise RuntimeError(
                        "prefix forward has per-step channel amax but no amax_now."
                    )
                torch.maximum(
                    self.per_step_cross_channel_amax,
                    amax_now.unsqueeze(0),
                    out=self.per_step_cross_channel_amax,
                )
            return
        if self.per_step_cross_channel_amax is None and self.per_step_cross_token_amax is None:
            return
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
        if self.per_step_seen is None:
            raise RuntimeError(
                "per-step cross amax updated but per_step_seen was not allocated."
            )
        self.per_step_seen[step] = True

    def _update_inner_amax(self, abs_flat: torch.Tensor, *, step: int | None) -> None:
        """Fixed-percentile inner (per-forward then max), or buffer |x| for adaptive."""
        plan = self.amax_plan
        p = self.inner_percentile
        stats_device = self.xtx.device
        ch_q = None
        tok_q = None
        if plan.collect_adaptive_inner_channel:
            if self._adaptive_channel_finalized:
                raise RuntimeError(
                    "adaptive inner-channel amax already finalized; cannot add forwards."
                )
            if abs_flat.shape[-1] != self.in_features:
                raise ValueError(
                    f"adaptive abs chunk last dim {abs_flat.shape[-1]} != "
                    f"in_features={self.in_features}."
                )
            abs16 = abs_flat.detach().to(
                device=self.xtx.device, dtype=torch.float16
            ).clone()
            self._buffer_adaptive_abs(abs16, step=step)
            return
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
            # Same prefix path as cross amax: encoder KV / LLM-scope forwards
            # have no denoise index. Static inner amax is already updated.
            return
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

    def _split_tip_rest(
        self, abs_act: torch.Tensor, *, step: int | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Split tip/rest by keep-mask. Rest amax is None when keep is unset."""
        keep = self._adaptive_token_keep
        if (
            keep is None
            and self.amax_plan.outlier_skip_first_token
            and step is not None
        ):
            n = int(abs_act.shape[0])
            if n < 2:
                raise RuntimeError(
                    "outlier_skip_first_token requires >= 2 tokens "
                    f"(state + action), got {n}."
                )
            keep = torch.ones(n, dtype=torch.bool, device=abs_act.device)
            keep[0] = False
        if keep is None:
            return abs_act, None
        if keep.ndim != 1 or int(keep.numel()) != int(abs_act.shape[0]):
            raise ValueError(
                f"adaptive tip_keep length {int(keep.numel())} != "
                f"num_tokens={int(abs_act.shape[0])}."
            )
        keep_dev = keep.detach().to(device=abs_act.device, dtype=torch.bool)
        tip = abs_act[keep_dev]
        if tip.shape[0] == 0:
            raise RuntimeError(
                "adaptive tip_keep selected zero tokens for this forward."
            )
        rest = abs_act[~keep_dev]
        if rest.shape[0] == 0:
            return tip, None
        rest_amax = rest.amax(dim=0).to(device=self.xtx.device, dtype=torch.float32)
        return tip, rest_amax

    @staticmethod
    def _merge_rest_amax(
        current: torch.Tensor | None, rest: torch.Tensor | None
    ) -> torch.Tensor | None:
        if rest is None:
            return current
        if current is None:
            return rest
        torch.maximum(current, rest, out=current)
        return current

    def _buffer_adaptive_abs(
        self, abs_act: torch.Tensor, *, step: int | None
    ) -> None:
        """Buffer one forward of |x|. Integer ``step`` goes to that denoise
        bucket; ``step=None`` uses the original concatenated list (LLM / prefix).
        """
        tip, rest = self._split_tip_rest(abs_act, step=step)
        buckets = self._adaptive_tip_chunks_by_step
        if buckets and step is not None:
            if self._adaptive_tip_chunks:
                raise RuntimeError(
                    "adaptive clip mixed prefix (step=None) activations with "
                    f"denoise-step hit {step}; a layer must be prefix-only "
                    "or observed at every denoise step, not both."
                )
            if step < 0 or step >= len(buckets):
                raise IndexError(
                    f"step={step} out of range for per-step clip "
                    f"(num_steps={len(buckets)})."
                )
            buckets[step].append(tip)
            self._adaptive_rest_amax_by_step[step] = self._merge_rest_amax(
                self._adaptive_rest_amax_by_step[step], rest
            )
            return
        if buckets and any(buckets):
            raise RuntimeError(
                "adaptive clip mixed prefix (step=None) activations with "
                "denoise-step hits; a layer must be prefix-only "
                "or observed at every denoise step, not both."
            )
        self._adaptive_tip_chunks.append(tip)
        self._adaptive_rest_amax = self._merge_rest_amax(
            self._adaptive_rest_amax, rest
        )

    def _fit_adaptive_channel_amax(
        self,
        tip_abs: torch.Tensor,
        rest_amax: torch.Tensor | None,
        *,
        std_k: float,
    ) -> torch.Tensor:
        """Fit one (C,) clip vector from concatenated tip |x|."""
        kappa = float(self.amax_plan.outlier_kappa)
        std_k = float(std_k)
        bulk_p = float(self.amax_plan.outlier_bulk_percentile)
        if self.amax_plan.outlier_selective_channels and std_k <= 0.0:
            raise RuntimeError(
                "selective channel clip requires outlier_std_k>0; "
                "no fallback to the kappa method is allowed."
            )
        if self.amax_plan.outlier_global and std_k <= 0.0:
            raise RuntimeError(
                "layer-global clip requires outlier_std_k>0; "
                "no fallback to per-channel or kappa clip is allowed."
            )
        if self.amax_plan.outlier_global and self.amax_plan.outlier_selective_channels:
            raise RuntimeError(
                "layer-global clip and selective channel clip cannot both be set."
            )
        if std_k > 0.0:
            if kappa > 0.0:
                raise RuntimeError(
                    "adaptive finalize: outlier_kappa and outlier_std_k both set."
                )
            if self.amax_plan.outlier_global:
                a = layer_outlier_mean_std_amax(tip_abs, std_k=std_k)
            elif self.amax_plan.outlier_selective_channels:
                a = selective_channel_outlier_mean_std_amax(
                    tip_abs,
                    std_k=std_k,
                )
            else:
                a = channel_outlier_mean_std_amax(tip_abs, std_k=std_k)
        elif kappa > 0.0:
            a = channel_outlier_bulk_kappa_amax(
                tip_abs, kappa=kappa, bulk_percentile=bulk_p
            )
        else:
            raise RuntimeError(
                "adaptive finalize requires outlier_kappa>0 or outlier_std_k>0."
            )
        if rest_amax is not None:
            a = torch.maximum(a, rest_amax)
        return a.to(device=self.xtx.device, dtype=torch.float32)

    def _clear_adaptive_buffers(self) -> None:
        self._adaptive_tip_chunks = []
        self._adaptive_rest_amax = None
        self._adaptive_tip_chunks_by_step = [
            [] for _ in self._adaptive_tip_chunks_by_step
        ]
        self._adaptive_rest_amax_by_step = [
            None for _ in self._adaptive_rest_amax_by_step
        ]
        self._adaptive_channel_finalized = True

    def _finalize_adaptive_inner_channel(self) -> None:
        """Tip-clip on buffered tip tokens; floor by streamed rest-channel amax."""
        if self._adaptive_channel_finalized:
            return
        if not self.amax_plan.collect_adaptive_inner_channel:
            raise RuntimeError(
                "_finalize_adaptive_inner_channel called without adaptive collection."
            )
        buckets = self._adaptive_tip_chunks_by_step
        if buckets and any(buckets):
            if self._adaptive_tip_chunks:
                raise RuntimeError(
                    "adaptive clip mixed prefix (step=None) activations with "
                    "denoise-step hits; a layer must be prefix-only "
                    "or observed at every denoise step, not both."
                )
            rows: list[torch.Tensor] = []
            n_steps = len(buckets)
            k_start, k_end = std_k_schedule_bounds(
                float(self.amax_plan.outlier_std_k),
                float(self.amax_plan.outlier_std_k_down),
                float(self.amax_plan.outlier_std_k_up),
            )
            for t, chunks in enumerate(buckets):
                if not chunks:
                    raise RuntimeError(
                        "adaptive clip requires every denoise step or prefix-only "
                        f"coverage; denoise step {t} collected no tokens."
                    )
                tip_abs = torch.cat(chunks, dim=0).to(torch.float32)
                rows.append(
                    self._fit_adaptive_channel_amax(
                        tip_abs,
                        self._adaptive_rest_amax_by_step[t],
                        std_k=scheduled_outlier_std_k(
                            k_start, k_end, step=t, num_steps=n_steps
                        ),
                    )
                )
            self.per_step_inner_channel_amax = torch.stack(rows, dim=0)
            self.static_inner_channel_amax = torch.zeros(
                0, device=self.xtx.device, dtype=torch.float32
            )
            self._clear_adaptive_buffers()
            return
        if not self._adaptive_tip_chunks:
            raise RuntimeError(
                "adaptive inner-channel amax: no activation chunks were collected."
            )
        tip_abs = torch.cat(self._adaptive_tip_chunks, dim=0).to(torch.float32)
        self.static_inner_channel_amax = self._fit_adaptive_channel_amax(
            tip_abs,
            self._adaptive_rest_amax,
            std_k=float(self.amax_plan.outlier_std_k),
        )
        self.per_step_inner_channel_amax = None
        self._clear_adaptive_buffers()

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

    def act_channel_cross_amax_pmean(self, *, p: float) -> torch.Tensor:
        """``(mean_t a_{j,t}^p)^{1/p}`` over the per-step channel absmax table.

        Prefix-only layers (GR00T cross-attn K/V) have the same amax in every
        row, so the p-mean equals that absmax. Partial denoise coverage raises.
        """
        from qvla.core.smooth_fit import pmean_per_step_channel_amax

        if not self.amax_plan.collect_cross_channel:
            raise RuntimeError(
                "act_channel_cross_amax_pmean requested but cross-channel "
                "amax was not collected."
            )
        if self.per_step_cross_channel_amax is None or self.per_step_seen is None:
            raise RuntimeError(
                "act_channel_cross_amax_pmean requires per-step cross "
                "channel amax (num_steps>1)."
            )
        n_seen = int(self.per_step_seen.sum().item())
        num_steps = int(self.per_step_seen.numel())
        table = self.per_step_cross_channel_amax
        if n_seen == num_steps:
            return pmean_per_step_channel_amax(table, p)
        if n_seen == 0:
            if self.n_tokens <= 0:
                raise RuntimeError(
                    "smooth p-mean has no activations for this layer."
                )
            if not torch.equal(table, table[:1].expand_as(table)):
                raise RuntimeError(
                    "layer has no denoise-step hits but per-step amax is not "
                    "constant across steps."
                )
            return pmean_per_step_channel_amax(table, p)
        seen_idx = torch.nonzero(self.per_step_seen, as_tuple=False).view(-1).tolist()
        raise RuntimeError(
            "smooth p-mean requires every denoise step or prefix-only "
            f"coverage; got {n_seen}/{num_steps} steps {seen_idx}."
        )

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
        """Inner-channel amax: fixed percentile (max over forwards) or adaptive clip.

        Adaptive denoise-loop layers return ``(num_steps, C)``; LLM and
        prefix-only DiT layers return ``(C,)``.
        """
        if not (
            self.amax_plan.collect_inner_channel
            or self.amax_plan.collect_adaptive_inner_channel
        ):
            raise RuntimeError(
                "act_channel_inner_amax requested but inner-channel amax was not collected."
            )
        if self.amax_plan.collect_adaptive_inner_channel:
            self._finalize_adaptive_inner_channel()
        if step is not None:
            if self.per_step_inner_channel_amax is None:
                raise ValueError(
                    "Requested per-step channel inner amax but this layer has "
                    "no per-step table (num_steps==1 or prefix-only clip)."
                )
            return self.per_step_inner_channel_amax[step]
        if self.amax_plan.collect_adaptive_inner_channel:
            if self.per_step_inner_channel_amax is not None:
                return self.per_step_inner_channel_amax
            if self.static_inner_channel_amax.numel() == 0:
                raise RuntimeError(
                    "adaptive clip finalize produced neither a per-step table "
                    "nor a 1D prefix act_clip."
                )
            return self.static_inner_channel_amax
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
        if not self._collect_hessian:
            raise RuntimeError(
                "LayerStats.hessian() called but collect_hessian=False for this pass."
            )
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
        ls.per_step_seen = None
        ls.amax_plan = AmaxCollectPlan()
        ls._adaptive_tip_chunks = []
        ls._adaptive_rest_amax = None
        ls._adaptive_tip_chunks_by_step = []
        ls._adaptive_rest_amax_by_step = []
        ls._adaptive_channel_finalized = False
        ls._collect_hessian = False
        ls.n_tokens = 0
        return ls


def assert_smooth_pmean_step_coverage(stats: dict[str, LayerStats]) -> None:
    """Require at least one layer with full denoise-step amax coverage.

    Prefix-only layers (cross-attn K/V) have ``per_step_seen`` all False
    because prefix forwards do not count as denoise callbacks; they are
    allowed. Partial coverage or zero fully-covered layers raise — missing
    ``step_callback`` must not silently look like a complete denoise loop.
    """
    if not stats:
        raise RuntimeError("smooth p-mean coverage check got an empty stats dict.")
    n_full = 0
    incomplete: list[str] = []
    for name, ls in stats.items():
        if ls.per_step_seen is None:
            raise RuntimeError(
                f"smooth p-mean layer {name!r} has no per_step_seen; "
                "num_steps>1 is required."
            )
        n_seen = int(ls.per_step_seen.sum().item())
        num_steps = int(ls.per_step_seen.numel())
        if n_seen == num_steps:
            n_full += 1
        elif n_seen != 0:
            seen_idx = torch.nonzero(ls.per_step_seen, as_tuple=False).view(-1).tolist()
            incomplete.append(f"{name}: {n_seen}/{num_steps} steps {seen_idx}")
    if incomplete:
        raise RuntimeError(
            "smooth p-mean has partial denoise-step coverage: "
            + "; ".join(incomplete)
        )
    if n_full == 0:
        raise RuntimeError(
            "smooth p-mean requires at least one layer observed at every "
            "denoise step; got none (step_callback missing, or only "
            "prefix-only layers were collected)."
        )


def assert_dit_clip_has_per_step_table(
    act_clips: dict[str, torch.Tensor],
) -> None:
    """Require at least one DiT layer with ``(num_steps, C)`` act_clip.

    Prefix-only layers store ``(C,)``. If every layer is 1D, the denoise
    callback never hit a loop linear and per-step clip did not run.
    """
    if not act_clips:
        raise RuntimeError("DiT clip coverage check got an empty act_clip dict.")
    n_per_step = 0
    missing: list[str] = []
    for name, clip in act_clips.items():
        if clip is None:
            missing.append(name)
            continue
        if clip.ndim == 2:
            if int(clip.shape[0]) < 2:
                raise RuntimeError(
                    f"DiT clip layer {name!r} per-step act_clip first dim "
                    f"must be num_steps>=2, got shape={tuple(clip.shape)}."
                )
            n_per_step += 1
        elif clip.ndim != 1:
            raise RuntimeError(
                f"DiT clip layer {name!r} act_clip must be 1D or 2D, "
                f"got shape={tuple(clip.shape)}."
            )
    if missing:
        raise RuntimeError(
            "DiT clip fitted no act_clip for: " + ", ".join(missing)
        )
    if n_per_step == 0:
        raise RuntimeError(
            "DiT clip requires at least one layer with per-step act_clip "
            "of shape (num_steps, C); every layer was prefix-only 1D. "
            "step_callback is missing or no denoise-loop linear was collected."
        )


def _keep_mask_skip_first_token(x: torch.Tensor) -> torch.Tensor | None:
    """False at sequence position 0 (GR00T state token); True elsewhere."""
    if x.ndim < 2:
        return None
    n = int(x.reshape(-1, x.shape[-1]).shape[0])
    if n < 2:
        return None
    seq = int(x.shape[-2]) if x.ndim >= 3 else n
    if seq < 2 or n % seq != 0:
        raise RuntimeError(
            "skip_first keep mask requires each sequence to have >= 2 "
            f"tokens and numel/C to divide seq; got shape={tuple(x.shape)} "
            f"n={n} seq={seq}."
        )
    keep = torch.ones(n // seq, seq, dtype=torch.bool, device=x.device)
    keep[:, 0] = False
    return keep.reshape(-1)


class RotatedActivationCollector:
    """Collect stats on ``transform.apply(x)`` for each target layer.

    Each layer's transform (a :class:`Transform`) encapsulates the full
    pipeline (clip → perm → svd → hadamard, etc.) as a single ``apply()``.

    When ``noise_ensemble_k > 1``, LLM layer hooks are skipped for
    ``noise_index > 0`` because LLM activations are diffusion-noise-invariant.
    """

    def __init__(
        self,
        targets: list[tuple[str, str, nn.Module]],
        transforms: dict[str, "Transform"],
        *,
        num_steps_by_scope: dict[str, int],
        device: str | torch.device = "cpu",
        amax_plan_by_scope: dict[str, AmaxCollectPlan],
        noise_ensemble_k: int = 1,
        dit_step_weights: dict[int, float] | None = None,
    ) -> None:
        self.stats: dict[str, LayerStats] = {}
        self.scopes: dict[str, str] = {}
        self.transforms = transforms
        self.handles: list = []
        self._current_step: int | None = None
        self._noise_index: int = 0
        self._noise_ensemble_k: int = max(1, noise_ensemble_k)
        self._dit_step_weights: dict[int, float] = dict(dit_step_weights or {})
        self._adaptive_token_keep: torch.Tensor | None = None
        self.device = torch.device(device)

        if not amax_plan_by_scope:
            raise ValueError("amax_plan_by_scope must be a non-empty dict.")

        for name, scope, mod in targets:
            if name not in transforms:
                raise KeyError(f"Missing rotation for layer {name!r}.")
            in_f = int(getattr(mod, "in_features"))
            ls = LayerStats(in_features=in_f)
            nsteps = num_steps_by_scope.get(scope, 1)
            amax_plan = amax_plan_by_scope[scope]
            ls.init(in_f, nsteps, self.device, amax_plan=amax_plan)
            self.stats[name] = ls
            self.scopes[name] = scope
            handle = mod.register_forward_pre_hook(
                self._make_hook(name), with_kwargs=False
            )
            self.handles.append(handle)

    def _make_hook(self, name: str):
        transform = self.transforms[name]

        def hook(_mod: nn.Module, inputs: tuple):
            if not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x):
                return
            scope = self.scopes[name]
            if scope != "dit" and self._noise_ensemble_k > 1 and self._noise_index > 0:
                return
            step = self._current_step if scope == "dit" else None
            xtx_weight = 1.0
            if scope == "dit" and self._dit_step_weights and step is not None:
                xtx_weight = self._dit_step_weights.get(step, 1.0)
            x_transformed = transform.apply(x, step=step)
            keep = self._adaptive_token_keep if scope != "dit" else None
            if (
                scope == "dit"
                and step is not None
                and self.stats[name].amax_plan.outlier_skip_first_token
            ):
                keep = _keep_mask_skip_first_token(x_transformed)
            self.stats[name]._adaptive_token_keep = keep
            self.stats[name].update(x_transformed, step=step, xtx_weight=xtx_weight)

        return hook

    def set_current_step(self, step: int | None) -> None:
        self._current_step = step

    def set_noise_index(self, idx: int) -> None:
        self._noise_index = idx

    def set_adaptive_token_keep_mask(self, mask: torch.Tensor | None) -> None:
        """Bool mask over flat tokens for the next LLM adaptive abs update."""
        if mask is not None:
            if mask.ndim != 1 or mask.dtype != torch.bool:
                raise ValueError(
                    f"adaptive keep mask must be 1-D bool, got shape={tuple(mask.shape)} "
                    f"dtype={mask.dtype}."
                )
        self._adaptive_token_keep = mask

    def detach(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def __enter__(self) -> "RotatedActivationCollector":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.detach()


__all__ = [
    "AmaxCollectPlan",
    "LayerStats",
    "RotatedActivationCollector",
    "assert_dit_clip_has_per_step_table",
    "assert_smooth_pmean_step_coverage",
]
