"""phyai Engine construction for GR00T-N1.7 pack build / calibration."""

from __future__ import annotations

import logging

from qvla.adapters.groot.config import GR00TAdapterConfig, resolve_dtype


logger = logging.getLogger(__name__)


def build_groot_engine(cfg: GR00TAdapterConfig):
    from phyai.engine import Engine, EngineArgs
    from phyai.engine_config import DeviceConfig, EngineConfig, RuntimeConfig
    from phyai.models.gr00t_n17.main_gr00t_n17 import GR00TN17Args

    params_dtype = resolve_dtype(cfg.params_dtype)

    logger.info(
        "GR00TAdapter: building model from %s (eager mode, no CUDA graph).",
        cfg.checkpoint_path,
    )
    return Engine(
        EngineArgs(
            plugin="gr00t_n17",
            plugin_args=GR00TN17Args(
                checkpoint_dir=cfg.checkpoint_path,
                max_batch_size=cfg.max_batch_size,
            ),
            config=EngineConfig(
                device=DeviceConfig(target=cfg.device, params_dtype=params_dtype),
                runtime=RuntimeConfig(use_cuda_graph=False),
            ),
        )
    )


def dit_step_count(engine, *, fallback: int) -> int:
    """Extract the number of action-head inference timesteps from the engine."""
    sched = getattr(engine.entry, "scheduler", None)
    if sched is None:
        return max(1, int(fallback))
    model = getattr(sched, "model", None)
    if model is None:
        return max(1, int(fallback))
    action_head = getattr(model, "action_head", None)
    if action_head is None:
        return max(1, int(fallback))
    num_steps = getattr(action_head, "num_inference_timesteps", None)
    if num_steps is not None and int(num_steps) > 0:
        return int(num_steps)
    return max(1, int(fallback))
