"""On-disk pack file format.

A pack is a single ``.pt`` (torch.save) blob containing one
:class:`LayerPack` per quantized layer plus a header with the
:class:`QVLAConfig` that produced it. The layout is:

    {
        "format_version": 1,
        "config": <serialized QVLAConfig.to_dict()>,
        "layers": {
            "<qualified_name>": <LayerPack-as-dict>,
            ...
        },
        # Optional opaque metadata for diagnostics / provenance.
        "meta": {
            "created_at": "<ISO timestamp>",
            "model_kind": "pi05",
            "num_samples": <int>,
            "git_sha": "<str or None>",
        },
    }

The file is self-describing: the runtime loader reads the embedded config and
validates that the user-supplied config matches before any layer replacement.

We deliberately keep the format ``torch.save``-based rather than custom binary:
the slowdown vs a hand-rolled binary is irrelevant (packs are <1 GB and load
once) and the format is trivially debug-inspectable from a Python REPL.
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from qvla.config import QVLAConfig
from qvla.core.rotation import Rotation


logger = logging.getLogger(__name__)

PACK_FORMAT_VERSION = 1


@dataclass
class LayerPack:
    """Everything needed to reconstruct one ``QuantLinear``.

    Stored shapes are *unrotated* in the sense that the rotation is recorded
    separately — the runtime layer applies the rotation to the activation, so
    the stored ``qweight`` is the already-rotated weight matrix
    (``W ← Rᵀ W`` baked in offline by the builder).
    """

    name: str                      # qualified module name (e.g. "model.paligemma_lm.layers.0.qkv_proj")
    scope: str                     # "llm" | "dit"

    in_features: int
    out_features: int
    bias_present: bool

    # int4-as-int8 representation. Storage: (out_features, in_features) int8 in [-8, 7].
    qweight: torch.Tensor
    weight_scale: torch.Tensor     # (N,) per-channel or (N, K/group_size)
    group_size: int
    weight_bits: int

    # Rotation applied along the input axis of the *original* weight.
    rotation: Rotation

    # Activation quant config (mode + table). When ``act_scale_table is None``
    # the runtime falls back to dynamic per-token quant.
    act_bits: int
    act_scale_mode: str            # "per_step" | "static" | "dynamic"
    act_scale_table: torch.Tensor | None  # per_channel: (in_features,) or (num_steps, in_features);
                                          # per_token: (num_tokens,) or (num_steps, num_tokens)

    # Optional bias (kept full-precision; tiny memory).
    bias: torch.Tensor | None

    # Optional sparse residual (only set when ``rtn_residual`` keeps outliers).
    residual: torch.Tensor | None

    # Free-form fork for future extensions without bumping format_version.
    extras: dict[str, Any] = field(default_factory=dict)
    act_scale_granularity: str = "per_channel"  # "per_channel" | "per_token"

    def state_dict(self) -> dict:
        return {
            "name": self.name,
            "scope": self.scope,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "bias_present": self.bias_present,
            "qweight": self.qweight.detach().contiguous().cpu(),
            "weight_scale": self.weight_scale.detach().contiguous().cpu(),
            "group_size": self.group_size,
            "weight_bits": self.weight_bits,
            "rotation": self.rotation.state_dict(),
            "act_bits": self.act_bits,
            "act_scale_mode": self.act_scale_mode,
            "act_scale_granularity": self.act_scale_granularity,
            "act_scale_table": (
                self.act_scale_table.detach().contiguous().cpu()
                if self.act_scale_table is not None
                else None
            ),
            "bias": (
                self.bias.detach().contiguous().cpu() if self.bias is not None else None
            ),
            "residual": (
                self.residual.detach().contiguous().cpu()
                if self.residual is not None
                else None
            ),
            "extras": dict(self.extras),
        }

    @classmethod
    def from_state_dict(cls, sd: dict) -> "LayerPack":
        return cls(
            name=sd["name"],
            scope=sd["scope"],
            in_features=int(sd["in_features"]),
            out_features=int(sd["out_features"]),
            bias_present=bool(sd["bias_present"]),
            qweight=sd["qweight"],
            weight_scale=sd["weight_scale"],
            group_size=int(sd["group_size"]),
            weight_bits=int(sd["weight_bits"]),
            rotation=Rotation.from_state_dict(sd["rotation"]),
            act_bits=int(sd["act_bits"]),
            act_scale_mode=sd["act_scale_mode"],
            act_scale_granularity=sd.get("act_scale_granularity", "per_channel"),
            act_scale_table=sd.get("act_scale_table"),
            bias=sd.get("bias"),
            residual=sd.get("residual"),
            extras=dict(sd.get("extras", {})),
        )


@dataclass
class Pack:
    """Top-level pack object — header + layer table."""

    config: QVLAConfig
    layers: dict[str, LayerPack]
    meta: dict[str, Any] = field(default_factory=dict)
    format_version: int = PACK_FORMAT_VERSION

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "format_version": self.format_version,
            "config": self.config.to_dict(),
            "layers": {name: lp.state_dict() for name, lp in self.layers.items()},
            "meta": {
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                **self.meta,
            },
        }
        torch.save(blob, path)
        logger.info("Saved QVLA pack to %s (%d layers).", path, len(self.layers))

    @classmethod
    def load(cls, path: str | Path, *, map_location: str | torch.device = "cpu") -> "Pack":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Pack file not found: {path}")
        # weights_only=False intentionally — packs contain pickled dicts of dataclasses.
        # We trust the input path (always points to a user-owned file).
        blob = torch.load(path, map_location=map_location, weights_only=False)
        fmt = int(blob.get("format_version", 0))
        if fmt != PACK_FORMAT_VERSION:
            raise ValueError(
                f"Pack file {path} has format_version={fmt}, expected "
                f"{PACK_FORMAT_VERSION}. Rebuild with the current builder."
            )
        config = QVLAConfig.from_dict(blob["config"])
        layers = {
            name: LayerPack.from_state_dict(sd)
            for name, sd in blob["layers"].items()
        }
        return cls(
            config=config,
            layers=layers,
            meta=dict(blob.get("meta", {})),
            format_version=fmt,
        )

    def summary(self) -> str:
        lines = [
            f"QVLA pack — model_kind={self.config.model_kind} layers={len(self.layers)}",
            f"  LLM scope: weight_quantizer={self.config.llm.weight_quantizer} "
            f"pipeline={','.join(self.config.llm.pipeline) or 'none'}",
            f"  DiT scope: weight_quantizer={self.config.dit.weight_quantizer} "
            f"pipeline={','.join(self.config.dit.pipeline) or 'none'} "
            f"act_scale_mode={self.config.dit.act_scale_mode}",
        ]
        return "\n".join(lines)


__all__ = ["LayerPack", "Pack", "PACK_FORMAT_VERSION"]
