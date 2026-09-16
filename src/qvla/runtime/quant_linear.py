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

    1.  x_t   = pipeline(x)                        # clip → smooth → rotation
    2.  x_q, s_x = quantize_activation(x_t)       # int4 + per-token / per-step scale
    3.  W_fp = dequant(qweight, weight_scale)      # int4 -> bf16 dense
    4.  y    = matmul(x_q.float() * s_x, W_fp.T)  # bf16 GEMM
    4a. y   += matmul(x_t, residual.T)             # only if residual is present
    5.  return y + bias

All input-side transforms (clip, smooth, rotation) are unified in the
pipeline stored on :class:`~qvla.core.pipeline.Transform`. Steps that are
absent from the pipeline are skipped.

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
from qvla.core.quantize import (
    is_no_quant,
    symmetric_quant_range,
    FP4_E2M1_MAX,
    _fp4_e2m1_dequantize,
    _fp4_e2m1_quantize,
    nvfp4_dequantize,
    nvfp4_quantize_activation,
)
from qvla.core.pipeline import apply_input_pipeline


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _RuntimeFlags:
    """Knobs that don't live on the pack but affect runtime behaviour."""

    return_tuple: bool        # phyai linears return ``(y, bias_or_None)``
    skip_bias_add: bool       # phyai's `skip_bias_add` flag
    output_dtype: torch.dtype # cast the GEMM output to this before returning
    nvfp_activation_num_samples: int | None


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
        nvfp_activation_num_samples: int | None = None,
    ) -> None:
        super().__init__()
        self.in_features = pack.in_features
        self.out_features = pack.out_features
        self.weight_bits = pack.weight_bits
        self.weight_format = getattr(pack, "weight_format", "int")
        self.act_bits = pack.act_bits
        self.act_format = getattr(pack, "act_format", "int")
        if self.weight_format not in ("int", "fp", "nvfp"):
            raise ValueError(
                f"QuantLinear({pack.name!r}): weight_format must be "
                f"'int', 'fp', or 'nvfp', got {self.weight_format!r}."
            )
        if self.act_format not in ("int", "fp", "nvfp"):
            raise ValueError(
                f"QuantLinear({pack.name!r}): act_format must be "
                f"'int', 'fp', or 'nvfp', got {self.act_format!r}."
            )
        self._weight_no_quant = is_no_quant(self.weight_bits)
        self._act_no_quant = is_no_quant(self.act_bits)
        self._act_qmin, self._act_qmax = symmetric_quant_range(self.act_bits)
        self._act_is_fp = self.act_format == "fp"
        self._act_is_nvfp = self.act_format == "nvfp"
        self.group_size = pack.group_size
        self.scope = pack.scope
        self.act_scale_mode = pack.act_scale_mode
        self.act_scale_granularity = pack.act_scale_granularity
        self.name = pack.name
        if self.weight_format == "nvfp" and self.group_size != 16:
            raise ValueError(
                f"QuantLinear({pack.name!r}): NVFP4 weight requires group_size=16, "
                f"got {self.group_size}."
            )
        if self.act_format == "nvfp" and (
            self.act_scale_mode != "dynamic"
            or self.act_scale_granularity != "per_block"
        ):
            raise ValueError(
                f"QuantLinear({pack.name!r}): NVFP4 activation requires "
                "act_scale_mode='dynamic' and "
                "act_scale_granularity='per_block'."
            )
        if self.act_scale_granularity == "per_block":
            if self.act_scale_mode != "dynamic":
                raise ValueError(
                    f"QuantLinear({pack.name!r}): per_block activation scaling "
                    f"requires dynamic mode, got {self.act_scale_mode!r}."
                )
            if self.group_size <= 0 or self.in_features % self.group_size != 0:
                raise ValueError(
                    f"QuantLinear({pack.name!r}): in_features={self.in_features} "
                    f"must be divisible by positive group_size={self.group_size} "
                    "for per_block activation scaling."
                )
        if self.act_format == "nvfp" and self.group_size != 16:
            raise ValueError(
                f"QuantLinear({pack.name!r}): NVFP4 activation requires "
                f"group_size=16, got {self.group_size}."
            )
        if self._act_is_nvfp and (
            nvfp_activation_num_samples is None
            or nvfp_activation_num_samples <= 0
        ):
            raise ValueError(
                f"QuantLinear({pack.name!r}): NVFP4 activation requires an "
                "explicit positive nvfp_activation_num_samples."
            )
        self._flags = _RuntimeFlags(
            return_tuple=return_tuple,
            skip_bias_add=skip_bias_add,
            output_dtype=output_dtype,
            nvfp_activation_num_samples=nvfp_activation_num_samples,
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

        if getattr(pack, "weight_scale_2", None) is not None:
            self.register_buffer(
                "weight_scale_2",
                pack.weight_scale_2.to(device_t, dtype=torch.float32).contiguous(),
                persistent=False,
            )
            self._has_weight_scale_2 = True
        else:
            self._has_weight_scale_2 = False

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

        # ---- transform pipeline (clip → smooth → rotation) ------------------
        # All input-side transforms live on the Transform object.
        rot = pack.rotation
        self._transform_pipeline = rot.pipeline
        self._has_smooth = rot.smooth_scale is not None
        if self._has_smooth:
            self.register_buffer(
                "smooth_s",
                rot.smooth_scale.to(device_t, dtype=output_dtype).contiguous(),
                persistent=False,
            )
        else:
            self.register_buffer(
                "smooth_s",
                torch.empty(0, device=device_t, dtype=output_dtype),
                persistent=False,
            )
        if rot.act_clip is not None:
            self.register_buffer(
                "act_clip",
                rot.act_clip.to(device_t, dtype=output_dtype).contiguous(),
                persistent=False,
            )
        else:
            self.register_buffer(
                "act_clip",
                torch.empty(0, device=device_t, dtype=output_dtype),
                persistent=False,
            )
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
        if self.act_scale_mode != "dynamic" and not self._has_scale_table:
            raise ValueError(
                f"QuantLinear({self.name!r}): act_scale_mode="
                f"{self.act_scale_mode!r} requires act_scale_table."
            )

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
            f"pipeline={','.join(self._transform_pipeline) or 'none'}, "
            f"has_perm={self._has_perm}, "
            f"has_smooth={self._has_smooth}, "
            f"has_residual={self._has_residual}"
        )

    def _apply_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the full pipeline (clip → smooth → rotation) along the last axis."""
        if not self._transform_pipeline:
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
        act_clip = self.act_clip if self.act_clip.numel() > 0 else None
        if act_clip is not None and act_clip.ndim == 2:
            act_clip = act_clip[self._step_counter % int(act_clip.shape[0])]
        return apply_input_pipeline(
            x,
            pipeline=self._transform_pipeline,
            u_blocks=u_blocks,
            perm=perm,
            block_size=self._rot_block_size,
            d=self.in_features,
            random_hadamard_blocks=random_hadamard_blocks,
            act_clip=act_clip,
            smooth_scale=self.smooth_s if self._has_smooth else None,
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
        if self.act_scale_mode == "dynamic":
            if self.act_scale_granularity == "per_block":
                blocks = x_rot.reshape(
                    *x_rot.shape[:-1],
                    self.in_features // self.group_size,
                    self.group_size,
                )
                amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
                grid_max = (
                    FP4_E2M1_MAX if self._act_is_fp else float(self._act_qmax)
                )
                return (amax / grid_max).to(torch.float32)
            amax = x_rot.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
            grid_max = FP4_E2M1_MAX if self._act_is_fp else float(self._act_qmax)
            return (amax / grid_max).to(torch.float32)

        if self.act_scale_mode == "static":
            if self.act_scale_granularity == "per_token":
                return self._broadcast_per_token_scale(self.act_scale_table, x_rot)
            return self.act_scale_table.view(1, -1)

        step = self._step_counter % self._num_steps
        if self.act_scale_granularity == "per_token":
            return self._broadcast_per_token_scale(
                self.act_scale_table[step], x_rot
            )
        return self.act_scale_table[step].view(1, -1)

    def _quantize_activation(
        self, x_rot: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize activations. Returns ``(x_dequant, scale)``.

        Supports signed symmetric int4 and E2M1 fp4 grids. We dequantize back
        to bf16 here rather than carrying 4-bit codes through the GEMM, because
        the GEMM kernel is bf16. The quant is still information-lossy in the
        same way the native 4-bit path would be.
        """
        if self._act_no_quant:
            return x_rot, torch.ones(1, 1, device=x_rot.device, dtype=torch.float32)

        if self._act_is_nvfp:
            num_samples = self._flags.nvfp_activation_num_samples
            assert num_samples is not None
            x_dequant = nvfp4_quantize_activation(
                x_rot,
                group_size=self.group_size,
                num_samples=num_samples,
            ).to(self._flags.output_dtype)
            return x_dequant, torch.ones(1, 1, device=x_rot.device, dtype=torch.float32)

        scale = self._activation_scale(x_rot)
        x_fp32 = x_rot.to(torch.float32)
        quant_shape = x_fp32.shape
        if self.act_scale_granularity == "per_block":
            x_fp32 = x_fp32.reshape(
                *x_fp32.shape[:-1],
                self.in_features // self.group_size,
                self.group_size,
            )
        scaled = x_fp32 / scale

        if self._act_is_fp:
            _, q_vals = _fp4_e2m1_quantize(scaled)
            x_dequant = (q_vals * scale).to(self._flags.output_dtype)
        else:
            q = torch.round(scaled).clamp(self._act_qmin, self._act_qmax)
            x_dequant = (q * scale).to(self._flags.output_dtype)

        return x_dequant.reshape(quant_shape), scale

    def _dequantize_weight(self) -> torch.Tensor:
        """Bring the quantized weight back to ``output_dtype`` for the GEMM.

        Supports int4 (signed symmetric), fp4 (E2M1 single-level), and
        official NVFP4 (E2M1 + FP8 block + FP32 global). We compute this on
        demand; torch.compile happily fuses dequant + GEMM when wrapped.
        """
        if self._weight_no_quant:
            return self.fp_weight
        q = self.qweight  # int8, shape (N, K)
        scale = self.weight_scale  # (N,) or (N, K/group_size) or NVFP4 (1,)
        N, K = q.shape

        if self.weight_format == "nvfp":
            if not self._has_weight_scale_2:
                raise RuntimeError(
                    f"QuantLinear({self.name}): weight_format=nvfp requires weight_scale_2."
                )
            W_fp = nvfp4_dequantize(
                q, scale, self.weight_scale_2, group_size=self.group_size
            )
        elif self.weight_format == "fp":
            vals = _fp4_e2m1_dequantize(q, device=q.device)
            if scale.ndim == 1:
                W_fp = vals * scale.unsqueeze(1)
            else:
                num_groups = scale.shape[1]
                gs = K // num_groups
                W_fp = (vals.reshape(N, num_groups, gs) * scale.unsqueeze(2)).reshape(N, K)
        else:
            if scale.ndim == 1:
                W_fp = q.to(torch.float32) * scale.unsqueeze(1)
            else:
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

        # 1. Full input-side pipeline: clip → smooth → rotation.
        x_cast = x.to(self._flags.output_dtype)
        x_rot = self._apply_transform(x_cast)

        # 2. Activation quant -> dequant (lossy round to int4).
        x_q, _scale = self._quantize_activation(x_rot)
        if self.act_clip.ndim == 2 or self.act_scale_mode == "per_step":
            self._step_counter += 1

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
