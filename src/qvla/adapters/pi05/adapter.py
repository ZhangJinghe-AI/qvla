"""PI05 :class:`ModelAdapter` implementation."""

from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np
import torch
import torch.nn as nn

from qvla.adapters.base import ModelAdapter
from qvla.adapters.pi05.calibration import iter_calibration_batches
from qvla.adapters.pi05.config import PI05AdapterConfig
from qvla.adapters.pi05.engine import build_pi05_engine, dit_step_count
from qvla.adapters.pi05.obs import build_pi05_request
from qvla.adapters.pi05.processor import create_pi05_processor
from qvla.adapters.pi05.step_hook import find_expert_runner, patched_one_step
from qvla.config import QVLAConfig


logger = logging.getLogger(__name__)


class PI05Adapter(ModelAdapter):
    """phyai pi0.5 adapter — see :mod:`qvla.adapters.pi05`."""

    model_kind: str = "pi05"

    def __init__(self, config: PI05AdapterConfig) -> None:
        self.cfg = config

        self._engine = None
        self._processor = None

    # Expose common knobs for tests / scripts --------------------------------

    @property
    def checkpoint_path(self):
        return self.cfg.checkpoint_path

    @property
    def calibration_source(self):
        return self.cfg.calibration_source

    @property
    def device(self):
        return self.cfg.device

    # ModelAdapter -----------------------------------------------------------

    def build_model(self) -> nn.Module:
        self._engine = build_pi05_engine(self.cfg)
        return self._engine.entry.model

    def warmup_for_calibration(self, model: nn.Module) -> None:
        del model
        if self._engine is None:
            raise RuntimeError("warmup_for_calibration() requires build_model() first.")
        self._ensure_processor()
        dummy_obs = {
            "images": {
                "image": np.zeros((256, 256, 3), dtype=np.uint8),
                "wrist_image": np.zeros((256, 256, 3), dtype=np.uint8),
            },
            "states": np.zeros(self.cfg.state_dim, dtype=np.float32),
            "task_description": "pad " * self.cfg.tokenizer_max_length,
        }
        request = build_pi05_request(self._processor, dummy_obs, state_dim=self.cfg.state_dim)
        logger.info("PI05Adapter: FlashInfer warmup (max-length prompt, 3 steps) ...")
        with torch.inference_mode():
            for _ in range(3):
                _ = self._engine.step(request)
            torch.cuda.synchronize()
        logger.info("PI05Adapter: FlashInfer warmup complete.")

    def iter_calibration_batches(self, num_samples: int):
        yield from iter_calibration_batches(
            self.cfg, num_samples, state_dim=self.cfg.state_dim
        )

    def forward_for_calibration(
        self,
        model: nn.Module,
        batch: dict,
        *,
        step_callback,
        sample_index: int | None = None,
        noise_index: int = 0,
    ) -> None:
        del model
        self._ensure_processor()
        assert self._engine is not None
        request = build_pi05_request(self._processor, batch, state_dim=self.cfg.state_dim)
        from qvla.build.calibration_noise import calibration_noise_for_sample_if_enabled

        noise = calibration_noise_for_sample_if_enabled(
            self, sample_index, noise_index=noise_index
        )
        if noise is not None:
            request = replace(request, noise=noise)
        runner = find_expert_runner(self._engine)
        with patched_one_step(runner, step_callback):
            step_callback(None)
            _ = self._engine.step(request)

    def forward_differentiable(
        self,
        batch: list[dict],
        *,
        sample_indices: list[int] | None = None,
        noise_index: int = 0,
    ) -> torch.Tensor:
        from qvla.build.calibration_noise import calibration_noise_for_sample_if_enabled
        from qvla.build.differentiable_forward import differentiable_step
        from qvla.runtime.step_context import reset_step_counters

        self._ensure_processor()
        assert self._engine is not None
        sched = getattr(self._engine.entry, "scheduler", None)
        if sched is None:
            raise RuntimeError("pi05 scheduler not built; call build_model() first.")

        obs_list = list(batch)
        if not obs_list:
            raise ValueError("forward_differentiable() got an empty batch list.")
        if sample_indices is not None:
            indices: list[int | None] = list(sample_indices)
        else:
            indices = [None] * len(obs_list)

        if len(indices) != len(obs_list):
            raise ValueError(
                f"sample_indices length {len(indices)} != batch size {len(obs_list)}."
            )
        max_B = int(sched.max_batch_size)
        if len(obs_list) > max_B:
            raise RuntimeError(
                f"Fisher/differentiable batch size {len(obs_list)} exceeds engine "
                f"max_batch_size={max_B}. Rebuild with a larger max_batch_size "
                "(pack build sets this from fisher_batch_size)."
            )

        reset_step_counters(self._engine.entry.model)
        request = build_pi05_request(
            self._processor, obs_list, state_dim=self.cfg.state_dim
        )
        request.pixel_values = request.pixel_values.detach()

        noises = [
            calibration_noise_for_sample_if_enabled(
                self, idx, noise_index=noise_index
            )
            for idx in indices
        ]
        if any(n is not None for n in noises):
            if any(n is None for n in noises):
                raise RuntimeError(
                    "Calibration noise is installed but some sample_indices values "
                    "in the batch are None; pass sample_indices for every slot."
                )
            request = replace(request, noise=torch.cat(noises, dim=0))
        return differentiable_step(sched, request)

    @property
    def engine(self):
        return self._engine

    @property
    def model(self) -> nn.Module | None:
        if self._engine is None:
            return None
        return self._engine.entry.model

    def dit_step_count(self, config: QVLAConfig) -> int:
        if self._engine is None:
            raise RuntimeError("dit_step_count() requires build_model() first.")
        return dit_step_count(self._engine, fallback=config.dit.num_steps)

    # Internal ---------------------------------------------------------------

    def _ensure_processor(self) -> None:
        if self._processor is not None:
            return
        self._processor = create_pi05_processor(self.cfg)
