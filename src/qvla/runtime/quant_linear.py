"""``QuantLinear`` — the runtime nn.Module that does the W4A4 forward pass.

The layer is intentionally a *replacement* for any of:

* ``torch.nn.Linear``,
* ``phyai.layers.linear.ReplicatedLinear``,
* ``phyai.layers.linear.RowParallelLinear``,
* ``phyai.layers.linear.ColumnParallelLinear``,
* ``phyai.layers.linear.QKVParallelLinear``.

Output shape matches the original — for phyai linears we return the
``(y, optional_bias)`` tuple, for ``torch.nn.Linear`` we return ``y``. The
wrapper records the right convention at construction time.

Forward pass
------------
::

    1.  x_rot  = R(x)                              # block-diag rotation
    2.  x_q, s_x = quantize_activation(x_rot)      # int4 + per-token / per-step scale
    3.  W_fp = dequant(qweight, weight_scale)      # int4 -> bf16 dense
    4.  y    = matmul(x_q.float() * s_x, W_fp.T)   # bf16 GEMM
    4a. y   += matmul(x_rot, residual.T)           # only if residual is present
    5.  return y + bias

We do *not* yet ship a fused int4 × int4 GEMM kernel — step 3 materializes a
dense bf16 weight on the fly and step 4 uses the regular bf16 matmul. This is
what loses most of the "compute" win but still recovers ~75 % of the memory
bandwidth win because:

* The on-disk weight is int8 (4 bits + 4 bits per byte effectively, since we
  haven't packed yet — even better when we do pack), so wall-time download
  cost is halved.
* The dequant-on-the-fly is fused with the next GEMM by torch.compile in many
  cases (when the model server enables it).

To plug in a real int4 GEMM, override :meth:`QuantLinear._matmul_kernel`
or monkey-patch the class.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

from qvla.core.pack import LayerPack
from qvla.core.quantize import is_no_quant, symmetric_quant_range
from qvla.core.rotation import apply_input_pipeline


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _RuntimeFlags:
    """Knobs that don't live on the pack but affect runtime behaviour."""

    return_tuple: bool        # phyai linears return ``(y, bias_or_None)``
    skip_bias_add: bool       # phyai's `skip_bias_add` flag
    output_dtype: torch.dtype # cast the GEMM output to this before returning


class QuantLinear(nn.Module):
    """Drop-in replacement for a bf16 Linear that runs a W4A4 forward.

    Parameters are *not* registered as ``nn.Parameter`` because we never train
    them and we want to keep them out of any ``state_dict`` walks the host
    model performs (phyai's weight loader, for instance, would otherwise try to
    overwrite them from disk). Instead they live as buffers — still moved by
    ``.to(device)`` calls, still serialized if you ever choose to, but invisible
    to ``model.named_parameters()``.
    """

    def __init__(
        self,
        pack: LayerPack,
        *,
        return_tuple: bool,
        skip_bias_add: bool,
        output_dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cuda",
    ) -> None:
        super().__init__()
        self.in_features = pack.in_features
        self.out_features = pack.out_features
        self.weight_bits = pack.weight_bits
        self.act_bits = pack.act_bits
        self._weight_no_quant = is_no_quant(self.weight_bits)
        self._act_no_quant = is_no_quant(self.act_bits)
        self._act_qmin, self._act_qmax = symmetric_quant_range(self.act_bits)
        self.group_size = pack.group_size
        self.scope = pack.scope
        self.act_scale_mode = pack.act_scale_mode
        self.act_scale_granularity = pack.act_scale_granularity
        self.name = pack.name

        self._flags = _RuntimeFlags(
            return_tuple=return_tuple,
            skip_bias_add=skip_bias_add,
            output_dtype=output_dtype,
        )

        device_t = torch.device(device)

        # ---- weight: int8-holding-int4 codes + per-group scale --------
        # We keep ``qweight`` int8 so the GEMM dequant step is a single mul.
        # When (and only when) someone plugs in a packed int4 kernel,
        # qweight can be re-packed to (N, K//2) uint8 — the pack file already
        # stores a separate bit indicating whether the codes are packed.
        self.register_buffer("qweight", pack.qweight.to(device_t).contiguous(), persistent=False)
        self.register_buffer(
            "weight_scale",
            pack.weight_scale.to(device_t, dtype=torch.float32).contiguous(),
            persistent=False,
        )

        if self._weight_no_quant:
            fp_weight = pack.extras.get("fp_weight")
            if fp_weight is None:
                raise ValueError(
                    f"Layer {pack.name!r} has weight_bits={self.weight_bits} "
                    "but pack extras contain no fp_weight."
                )
            self.register_buffer(
                "fp_weight",
                fp_weight.to(device_t, dtype=output_dtype).contiguous(),
                persistent=False,
            )

        if pack.bias is not None:
            self.register_buffer("bias", pack.bias.to(device_t, dtype=output_dtype), persistent=False)
        else:
            self.bias = None  # type: ignore[assignment]

        # ---- rotation: U blocks + optional zigzag perm + pipeline -----------
        rot = pack.rotation
        self._rotation_pipeline = rot.pipeline
        if rot.is_identity:
            self._has_perm = False
            self._rot_block_size = 0
            self.register_buffer("rotation_u_blocks", torch.empty(0, device=device_t), persistent=False)
            self.register_buffer("perm", torch.empty(0, dtype=torch.int64, device=device_t), persistent=False)
            self.register_buffer(
                "random_hadamard_blocks", torch.empty(0, device=device_t), persistent=False
            )
        else:
            self._has_perm = rot.perm is not None
            self._rot_block_size = rot.block_size
            if rot.perm is not None:
                self.register_buffer(
                    "perm",
                    rot.perm.to(device_t, dtype=torch.int64).contiguous(),
                    persistent=False,
                )
            else:
                self.register_buffer(
                    "perm",
                    torch.empty(0, dtype=torch.int64, device=device_t),
                    persistent=False,
                )
            if rot.u_blocks is not None:
                self.register_buffer(
                    "rotation_u_blocks",
                    rot.u_blocks.to(device_t, dtype=output_dtype).contiguous(),
                    persistent=False,
                )
            else:
                self.register_buffer(
                    "rotation_u_blocks",
                    torch.empty(0, device=device_t),
                    persistent=False,
                )
            if rot.random_hadamard_blocks is not None:
                self.register_buffer(
                    "random_hadamard_blocks",
                    rot.random_hadamard_blocks.to(
                        device_t, dtype=output_dtype
                    ).contiguous(),
                    persistent=False,
                )
            else:
                self.register_buffer(
                    "random_hadamard_blocks",
                    torch.empty(0, device=device_t),
                    persistent=False,
                )

        # ---- residual (optional) --------------------------------------
        if pack.residual is not None:
            self.register_buffer(
                "residual",
                pack.residual.to(device_t, dtype=output_dtype).contiguous(),
                persistent=False,
            )
            self._has_residual = True
        else:
            self.residual = None  # type: ignore[assignment]
            self._has_residual = False

        # ---- activation scale table -----------------------------------
        if pack.act_scale_table is not None:
            self.register_buffer(
                "act_scale_table",
                pack.act_scale_table.to(device_t, dtype=torch.float32).contiguous(),
                persistent=False,
            )
            self._has_scale_table = True
            self._num_steps = (
                self.act_scale_table.shape[0]
                if self.act_scale_table.ndim == 2
                else 1
            )
        else:
            self.act_scale_table = None  # type: ignore[assignment]
            self._has_scale_table = False
            self._num_steps = 1

        # Python int counter — see step_context.py for the rationale on why
        # this is safe even under CUDA graph capture.
        self._step_counter: int = 0

    # ------------------------------------------------------------------
    # Counter management
    # ------------------------------------------------------------------

    def reset_step_counter(self) -> None:
        self._step_counter = 0

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"weight_bits={self.weight_bits}, act_bits={self.act_bits}, "
            f"group_size={self.group_size}, scope={self.scope}, "
            f"act_scale_mode={self.act_scale_mode}, "
            f"act_scale_granularity={self.act_scale_granularity}, "
            f"rotation_pipeline={','.join(self._rotation_pipeline) or 'none'}, "
            f"has_perm={self._has_perm}, "
            f"has_residual={self._has_residual}"
        )

    def _apply_rotation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the configured perm / SVD / Hadamard pipeline along the last axis."""
        if not self._rotation_pipeline:
            return x
        u_blocks = (
            self.rotation_u_blocks
            if self.rotation_u_blocks.numel() > 0
            else None
        )
        perm = self.perm if self._has_perm else None
        random_hadamard_blocks = (
            self.random_hadamard_blocks
            if self.random_hadamard_blocks.numel() > 0
            else None
        )
        return apply_input_pipeline(
            x,
            pipeline=self._rotation_pipeline,
            u_blocks=u_blocks,
            perm=perm,
            block_size=self._rot_block_size,
            d=self.in_features,
            random_hadamard_blocks=random_hadamard_blocks,
        )

    def _broadcast_per_token_scale(
        self, calibrated: torch.Tensor, x_rot: torch.Tensor
    ) -> torch.Tensor:
        """Expand calibrated per-token scales to ``x_rot``'s token axis.

        Packs are typically calibrated at ``B=1`` so the table length equals
        ``chunk_size``. PhyAI's expert runner flattens to ``(B * chunk_size, K)``
        in sample-major order, so when the runtime token count is an integer
        multiple of the table length we repeat the calibrated scales once per
        batch item (token ``i`` in every sample shares ``calibrated[i]``).
        """
        t = int(x_rot.shape[-2])
        calibrated = calibrated.reshape(-1)
        n = int(calibrated.numel())
        if t == n:
            scales = calibrated
        elif t > n and t % n == 0:
            scales = calibrated.repeat(t // n)
        elif t < n:
            scales = calibrated[:t]
        else:
            raise RuntimeError(
                f"QuantLinear({self.name}): per_token act_scale_table length "
                f"{n} cannot broadcast to runtime tokens {t}. Use "
                f"max_batch_size so B*chunk_size is {n} or a multiple of {n}, "
                f"rebuild with act_scale_granularity=per_channel, or rebuild "
                f"the pack at the serve batch size."
            )
        return scales.view(*((1,) * (x_rot.ndim - 2)), -1, 1)

    def _activation_scale(self, x_rot: torch.Tensor) -> torch.Tensor:
        """Return activation scale in float32.

        ``dynamic`` uses runtime per-token amax. ``static`` / ``per_step`` use
        the offline table: per-channel rows are ``(1, in_features)``;
        per-token rows are ``(num_tokens, 1)``.
        """
        if self.act_scale_mode == "dynamic" or not self._has_scale_table:
            amax = x_rot.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
            return (amax / self._act_qmax).to(torch.float32)

        if self.act_scale_mode == "static":
            if self.act_scale_granularity == "per_token":
                return self._broadcast_per_token_scale(self.act_scale_table, x_rot)
            return self.act_scale_table.view(1, -1)

        step = self._step_counter % self._num_steps
        self._step_counter += 1
        if self.act_scale_granularity == "per_token":
            return self._broadcast_per_token_scale(
                self.act_scale_table[step], x_rot
            )
        return self.act_scale_table[step].view(1, -1)

    def _quantize_activation(
        self, x_rot: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Symmetric int4 activation quant. Returns ``(x_dequant, _)`` for now.

        We dequantize back to bf16 here rather than carrying int4 through the
        GEMM, because the GEMM kernel is bf16. The int4 quant is still
        *information-lossy* in the same way the int4 path would be, so the
        end-to-end accuracy matches.
        """
        if self._act_no_quant:
            return x_rot, torch.ones(1, 1, device=x_rot.device, dtype=torch.float32)
        scale = self._activation_scale(x_rot)
        # x_rot is at output_dtype (bf16). Promote to float32 for the round.
        x_fp32 = x_rot.to(torch.float32)
        q = torch.round(x_fp32 / scale).clamp(self._act_qmin, self._act_qmax)
        x_dequant = (q * scale).to(self._flags.output_dtype)
        return x_dequant, scale

    def _dequantize_weight(self) -> torch.Tensor:
        """Bring the int4 weight back to ``output_dtype`` for the GEMM.

        We compute this on demand and cache it for the lifetime of one forward
        pass. Trivially fast vs the GEMM itself but it'd be nice to fuse;
        torch.compile happily does so when wrapped.
        """
        if self._weight_no_quant:
            return self.fp_weight
        q = self.qweight  # int8 in [-8, 7], shape (N, K)
        scale = self.weight_scale  # (N,) or (N, K/group_size) fp32
        if scale.ndim == 1:
            W_fp = q.to(torch.float32) * scale.unsqueeze(1)
        else:
            N, K = q.shape
            num_groups = scale.shape[1]
            gs = K // num_groups
            W_fp = (
                q.to(torch.float32).reshape(N, num_groups, gs) * scale.unsqueeze(2)
            ).reshape(N, K)
        return W_fp.to(self._flags.output_dtype)

    def _matmul_kernel(self, x: torch.Tensor) -> torch.Tensor:
        """The matmul itself. Plug a real int4 kernel here if you have one.

        Default: dequantize weight to ``output_dtype`` and run a normal matmul.
        """
        W_fp = self._dequantize_weight()           # (N, K) bf16
        # Standard linear: y = x @ W.T
        return torch.nn.functional.linear(x, W_fp, bias=None)

    def forward(self, x: torch.Tensor):
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"QuantLinear({self.name}): expected last dim "
                f"{self.in_features}, got {x.shape[-1]}."
            )

        # 1. Rotation along input axis (block-diag).
        x_rot = self._apply_rotation(x.to(self._flags.output_dtype))

        # 2. Activation quant -> dequant (lossy round to int4).
        x_q, _scale = self._quantize_activation(x_rot)

        # 3. Main GEMM.
        y = self._matmul_kernel(x_q)

        # 4. Optional residual correction (rotation-tamed outliers).
        if self._has_residual:
            # Residual is stored at fp16; promote to output dtype for the add.
            res = self.residual.to(self._flags.output_dtype)
            y = y + torch.nn.functional.linear(x_rot, res, bias=None)

        # 5. Bias.
        add_bias = self.bias if (self.bias is not None and not self._flags.skip_bias_add) else None
        if add_bias is not None:
            y = y + add_bias

        if self._flags.return_tuple:
            unused_bias = self.bias if self._flags.skip_bias_add else None
            return y, unused_bias
        return y


__all__ = ["QuantLinear"]
