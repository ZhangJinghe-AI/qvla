"""Per-model adapter interface.

The core quantization machinery (rotation, GPTQ/RTN, pack format, runtime
layer) is model-blind. Anything that depends on a specific VLA's runner topology
— how to build the model, how to run a calibration step, how many denoise
steps the DiT does, where to get observation samples from — is delegated to a
:class:`ModelAdapter`.

Adapters are small: they declare the regexes that classify each linear into
``llm`` or ``dit``, they yield calibration batches, they run *one* inference
of the model on a batch (and tell the calibration collector which step they're
on), and they answer ``dit_step_count()`` for the per-step act-scale table.

Concrete adapters live in :mod:`qvla.adapters.pi05` and
:mod:`qvla.adapters.groot`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import replace
from typing import Callable, Iterator

import torch
import torch.nn as nn

from qvla.config import QVLAConfig


class ModelAdapter(ABC):
    """Abstract VLA-specific glue used by the builder."""

    #: Short id ("pi05", "groot_n17", ...). Recorded in pack header.
    model_kind: str = "unknown"

    def prepare_model(
        self, config: QVLAConfig
    ) -> tuple[nn.Module, QVLAConfig]:
        """Build the model and sync runtime-derived knobs into *config*."""
        model = self.build_model()
        model.eval()
        dit_steps = self.dit_step_count(config)
        config = config.with_overrides(
            dit=replace(config.dit, num_steps=dit_steps)
        )
        return model, config

    @abstractmethod
    def build_model(self) -> nn.Module:
        """Construct the full-precision model and load weights.

        After this returns, ``model.named_modules()`` must contain every Linear
        the regexes target. CUDA-graph capture (if any) should be **off**
        here — the builder needs to run a Python loop with hooks attached.
        """

    @abstractmethod
    def iter_calibration_batches(self, num_samples: int) -> Iterator[dict]:
        """Yield ``num_samples`` calibration batches.

        Each batch is whatever the adapter's ``forward_for_calibration`` knows
        how to consume — typically a dict mirroring the model server's
        observation shape.
        """

    @abstractmethod
    def forward_for_calibration(
        self,
        model: nn.Module,
        batch: dict,
        *,
        step_callback,
        sample_index: int | None = None,
        noise_index: int = 0,
    ) -> None:
        """Run one calibration inference end-to-end.

        ``step_callback(step: int | None)`` must be invoked with the current
        Euler step index right before each DiT pass, and with ``None`` for
        any LLM-only / pre-prefix work. The builder hooks the callback up to
        the :class:`RotatedActivationCollector` so per-step stats land in the right
        bin.

        ``noise_index`` selects which of the ``noise_ensemble_k`` diffusion
        noises to use for this sample (installed calibration noise).
        """

    def calibration_outlier_token_keep_mask(
        self,
        batch: dict,
        *,
        token_scope: str,
    ) -> torch.Tensor | None:
        """Bool mask over packed prefix tokens for adaptive act-clip tip fit.

        Tip-clip runs on ``True`` positions; the complement floors ``act_clip``
        so those tokens are never clipped below their calib max.
        ``token_scope='all'`` may return ``None`` (collector keeps every token)
        or an all-True mask.

        Default: ``all`` → ``None``; other scopes raise — override in adapters
        that know image/lang layout (e.g. pi0.5).
        """
        del batch
        if token_scope == "all":
            return None
        raise NotImplementedError(
            f"{type(self).__name__} does not support calibration outlier "
            f"token_scope={token_scope!r}; implement "
            "calibration_outlier_token_keep_mask (or use token_scope='all')."
        )

    def calibration_noise_spec(self):
        """Return diffusion-noise shape/placement for pack-build calibration.

        Required when fixed calibration noise is installed (the default pack
        build path). Override in concrete adapters.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement calibration_noise_spec(); "
            "fixed calibration diffusion noise requires a noise shape/placement."
        )

    def dit_step_count(self, config: QVLAConfig) -> int:
        """How many denoise steps the DiT runs per inference.

        Called from :meth:`prepare_model` after :meth:`build_model` to size
        the per-step activation scale table.
        Override when the loaded model exposes a scheduler step count.
        """
        return max(1, int(config.dit.num_steps))

    def warmup_for_calibration(self, model: nn.Module) -> None:
        """Optional pre-calibration priming (e.g. FlashInfer max-length plan).

        Called after :meth:`build_model` and **before** activation hooks are
        attached, so dummy forwards here do not affect quantization stats.
        """
        del model

    @property
    def engine(self):
        """Underlying inference engine, or ``None`` before :meth:`build_model`."""
        return getattr(self, "_engine", None)

    @contextmanager
    def fisher_forward_context(
        self, set_step: Callable[[int], None]
    ) -> Iterator[None]:
        """Model-specific setup around the Fisher sample loop.

        Default is a no-op. Adapters that need autograd patches, CUDA-graph
        disable, or DiT step-index hooks (e.g. pi0.5) override this.
        ``set_step(step)`` should be called before each DiT denoise pass so
        the Fisher collector bins sensitivity by Euler step.
        """
        del set_step
        yield

    def forward_differentiable(
        self,
        batch: list[dict],
        *,
        sample_indices: list[int] | None = None,
        noise_index: int = 0,
    ):
        """Run one inference with autograd enabled; return differentiable actions.

        Required for Fisher sensitivity (``perm_score=fisher``). Subclasses that
        don't support differentiable inference should leave this unimplemented —
        the builder will raise when Fisher perm is requested.

        ``batch`` is a non-empty list of observation dicts (Fisher micro-batch).
        ``sample_indices`` / ``noise_index``
        select the installed calibration diffusion noise when noise-ensemble
        calibration is enabled.

        Returns:
            A :class:`torch.Tensor` connected to the autograd graph so that
            :meth:`~torch.Tensor.backward` reaches the model's intermediate
            activations.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement forward_differentiable(); "
            "policy-aware rotation (perm_score=fisher) requires a "
            "differentiable forward path."
        )


__all__ = ["ModelAdapter"]
