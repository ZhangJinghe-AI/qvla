"""GR00T-N1.7 :class:`ModelAdapter` implementation."""

from __future__ import annotations

import logging
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn

from qvla.adapters.base import ModelAdapter
from qvla.adapters.groot.calibration import (
    groot_outlier_token_keep_mask,
    iter_calibration_batches,
)
from qvla.adapters.groot.config import GR00TAdapterConfig
from qvla.adapters.groot.engine import build_groot_engine, dit_step_count
from qvla.adapters.groot.obs import build_groot_request
from qvla.adapters.groot.processor import create_groot_processor
from qvla.adapters.groot.step_hook import find_action_head_runner, patched_action_head_denoise
from qvla.config import QVLAConfig


logger = logging.getLogger(__name__)


class GR00TAdapter(ModelAdapter):
    """phyai GR00T-N1.7 adapter — see :mod:`qvla.adapters.groot`."""

    model_kind: str = "groot_n17"

    def __init__(self, config: GR00TAdapterConfig) -> None:
        self.cfg = config

        self._engine = None
        self._processor = None

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
        self._engine = build_groot_engine(self.cfg)
        return self._engine.entry.model

    def warmup_for_calibration(self, model: nn.Module) -> None:
        del model
        if self._engine is None:
            raise RuntimeError("warmup_for_calibration() requires build_model() first.")
        self._ensure_processor()
        from phyai_utils_tools.models.gr00t import GR00TObservation

        modality_config = self._processor.modality_config
        tag = self._processor.embodiment_tag
        video_keys = modality_config["video"].modality_keys
        state_keys = modality_config["state"].modality_keys
        language_key = modality_config["language"].modality_keys[0]
        video_time = len(modality_config["video"].delta_indices)
        state_time = len(modality_config["state"].delta_indices)
        language_time = len(modality_config["language"].delta_indices)

        video = {
            key: np.zeros((1, video_time, self.cfg.image_size, self.cfg.image_size, 3), dtype=np.uint8)
            for key in video_keys
        }
        state = {
            key: np.zeros((1, state_time, int(self._processor.norm_params[tag]["state"][key]["dim"])), dtype=np.float32)
            for key in state_keys
        }
        language = {
            language_key: [["pad " * 50 for _ in range(language_time)]]
        }
        dummy_obs = {"video": video, "state": state, "language": language}
        request = build_groot_request(self._processor, dummy_obs, device=self.cfg.device)

        logger.info("GR00TAdapter: warmup (3 steps) ...")
        with torch.inference_mode():
            for _ in range(3):
                _ = self._engine.step(request)
            torch.cuda.synchronize()
        logger.info("GR00TAdapter: warmup complete.")

    def iter_calibration_batches(self, num_samples: int):
        self._ensure_processor()
        yield from iter_calibration_batches(self.cfg, self._processor, num_samples)

    def calibration_outlier_token_keep_mask(
        self,
        batch: dict,
        *,
        token_scope: str,
    ) -> torch.Tensor:
        """Keep image (and optionally lang_pad) tokens for adaptive tip-clip fit."""
        self._ensure_processor()
        if self._engine is None:
            raise RuntimeError(
                "calibration_outlier_token_keep_mask() requires build_model() first."
            )
        request = build_groot_request(
            self._processor, batch, device=self.cfg.device
        )
        input_ids = request.tensors["input_ids"]
        attention_mask = request.tensors["attention_mask"]
        if int(input_ids.shape[0]) != 1:
            raise RuntimeError(
                "adaptive outlier token mask expects batch size 1, got "
                f"input_ids shape {tuple(input_ids.shape)}."
            )
        qwen = self._engine.entry.model.backbone.qwen3vl_model
        video_token_id = getattr(qwen.config, "video_token_id", None)
        return groot_outlier_token_keep_mask(
            input_ids[0],
            attention_mask[0],
            token_scope=token_scope,
            image_token_id=int(qwen.config.image_token_id),
            video_token_id=(
                None if video_token_id is None else int(video_token_id)
            ),
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
        request = build_groot_request(self._processor, batch, device=self.cfg.device)

        from qvla.build.calibration_noise import calibration_noise_for_sample_if_enabled
        from phyai.models.gr00t_n17.scheduler_ws1_gr00t_n17 import GR00TN17Request

        noise = calibration_noise_for_sample_if_enabled(
            self, sample_index, noise_index=noise_index
        )
        if noise is not None:
            request = GR00TN17Request(tensors=request.tensors, noise=noise)

        runner = find_action_head_runner(self._engine)
        with patched_action_head_denoise(runner, step_callback) as _step_state:
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
        from phyai.models.gr00t_n17.scheduler_ws1_gr00t_n17 import GR00TN17Request

        self._ensure_processor()
        assert self._engine is not None
        sched = getattr(self._engine.entry, "scheduler", None)
        if sched is None:
            raise RuntimeError("GR00T-N1.7 scheduler not built; call build_model() first.")

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

        request = build_groot_request(self._processor, obs_list, device=self.cfg.device)

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
            request = GR00TN17Request(
                tensors=request.tensors,
                noise=torch.cat(noises, dim=0),
            )

        # Run the scheduler step with grad enabled for Fisher sensitivity.
        # GR00T-N1.7 uses @torch.no_grad in its step; bypass via direct forward.
        backbone_inputs, action_inputs = sched._prepare_request(request)
        backbone_output = sched.backbone_runner.forward(backbone_inputs)
        noise_tensor = sched.action_head_runner._prepare_noise(backbone_output, request.noise)
        noise_tensor = noise_tensor.detach().requires_grad_(True)

        inputs = sched.action_head_runner._graph_inputs(backbone_output, action_inputs, noise_tensor)
        actions = sched.action_head_runner._fwd_loop(**inputs)
        return actions

    @contextmanager
    def fisher_forward_context(self, set_step):
        from qvla.adapters.groot.differentiable import differentiable_inference_context

        if self._engine is None:
            raise RuntimeError("fisher_forward_context() requires build_model() first.")
        sched = self._engine.entry.scheduler
        runner = find_action_head_runner(self._engine)
        with differentiable_inference_context(sched):
            with patched_action_head_denoise(runner, set_step):
                yield

    @property
    def engine(self):
        return self._engine

    @property
    def model(self) -> nn.Module | None:
        if self._engine is None:
            return None
        return self._engine.entry.model

    def calibration_noise_spec(self):
        from qvla.build.calibration_noise import CalibrationNoiseSpec

        if self._engine is None:
            raise RuntimeError("calibration_noise_spec() requires build_model() first.")
        sched = self._engine.entry.scheduler
        action_head = sched.model.action_head
        return CalibrationNoiseSpec(
            chunk_size=int(action_head.action_horizon),
            action_dim=int(action_head.action_dim),
            device=torch.device(sched.device),
            dtype=sched.model.params_dtype,
        )

    def dit_step_count(self, config: QVLAConfig) -> int:
        if self._engine is None:
            raise RuntimeError("dit_step_count() requires build_model() first.")
        return dit_step_count(self._engine, fallback=config.dit.num_steps)

    # Internal ---------------------------------------------------------------

    def _ensure_processor(self) -> None:
        if self._processor is not None:
            return
        self._processor = create_groot_processor(self.cfg)
