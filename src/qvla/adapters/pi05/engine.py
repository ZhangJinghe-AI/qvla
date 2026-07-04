"""phyai Engine construction for pack build / calibration."""

from __future__ import annotations

import logging

from qvla.adapters.pi05.config import PI05AdapterConfig, lerobot_weight_remap, resolve_dtype


logger = logging.getLogger(__name__)


def build_pi05_engine(cfg: PI05AdapterConfig):
    from phyai.engine import Engine, EngineArgs
    from phyai.engine_config import BackendConfig, DeviceConfig, EngineConfig, RuntimeConfig
    from phyai.models.pi05.main_pi05 import PI05Args

    params_dtype = resolve_dtype(cfg.params_dtype)
    vision_params_dtype = resolve_dtype(cfg.vision_params_dtype)

    logger.info(
        "PI05Adapter: building model from %s (eager mode, attn=%s).",
        cfg.checkpoint_path,
        cfg.attn_backend,
    )
    return Engine(
        EngineArgs(
            plugin="pi05",
            plugin_args=PI05Args(
                checkpoint_dir=cfg.checkpoint_path,
                max_batch_size=1,
                vision_params_dtype=vision_params_dtype,
                weight_remap=lerobot_weight_remap,
                inputs_image_shape=[
                    [cfg.image_size, cfg.image_size, 3]
                    for _ in range(cfg.num_real_cameras)
                ],
            ),
            config=EngineConfig(
                backends=BackendConfig(
                    attn=cfg.attn_backend,
                    norm=cfg.norm_backend,
                    linear="torch",
                ),
                device=DeviceConfig(target=cfg.device, params_dtype=params_dtype),
                runtime=RuntimeConfig(use_cuda_graph=False),
            ),
        )
    )


def dit_step_count(engine, *, fallback: int) -> int:
    sched = getattr(engine.entry, "scheduler", None)
    if sched is not None and hasattr(sched, "_num_steps"):
        ns = int(getattr(sched, "_num_steps"))
        if ns > 0:
            return ns
    return max(1, int(fallback))
