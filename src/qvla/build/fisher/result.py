"""Per-layer sensitivity accumulator + step-axis aggregation.

Deliberately format-agnostic: whether the numbers come from
:class:`InputGradFisherCollector` or :class:`OutputHessianFisherCollector`
the container is identical. Callers use :meth:`aggregate` to collapse
the per-step axis into the single ``(in_features,)`` vector consumed by
rotation-fit / GPTQ / SmoothQuant.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from qvla.build.fisher.common import StepAggregation


@dataclass
class LayerFisherResult:
    """Accumulated per-input-channel sensitivity for one layer.

    Attributes
    ----------
    name, scope:
        Layer identity for logging and step-aggregation dispatch.
    in_features:
        Number of input channels (the length of each per-step vector).
    _step_accum:
        Streaming sum. Key ``None`` means step-agnostic (LLM); integer
        keys are diffusion / denoising step indices (DiT).
    _n_samples:
        Total observations mixed in — used as the divisor when a caller
        asks for a per-sample mean.
    """

    name: str
    scope: str  # "llm" | "dit"
    in_features: int

    _step_accum: dict[int | None, torch.Tensor] = field(
        default_factory=dict, repr=False
    )
    _n_samples: int = 0

    def sensitivity_per_step(self) -> dict[int | None, torch.Tensor]:
        """Per-step sensitivity averaged over calibration samples."""
        n = max(1, self._n_samples)
        return {step: s / n for step, s in self._step_accum.items()}

    def aggregate(self, method: StepAggregation = "uniform") -> torch.Tensor:
        """Collapse the step axis into one ``(in_features,)`` vector."""
        per_step = self.sensitivity_per_step()
        if not per_step:
            return torch.zeros(self.in_features)

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


__all__ = ["LayerFisherResult"]
