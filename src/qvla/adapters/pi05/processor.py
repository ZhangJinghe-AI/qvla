"""PI05Processor construction."""

from __future__ import annotations

from qvla.adapters.pi05.config import PI05AdapterConfig, resolve_dtype


def reorder_postprocessor_slice_first(processor) -> None:
    from phyai_utils_tools.processing.steps.action_steps import SliceActionStep

    steps = list(processor.postprocessor.steps)
    slice_steps = [s for s in steps if isinstance(s, SliceActionStep)]
    core_steps = [s for s in steps if not isinstance(s, SliceActionStep)]
    processor._postprocessor.steps = [*slice_steps, *core_steps]


def create_pi05_processor(cfg: PI05AdapterConfig):
    import phyai_utils_tools.models.pi05.steps_pi05  # noqa: F401 — register steps
    from phyai_utils_tools.models.pi05.processor_pi05 import PI05Processor
    from phyai_utils_tools.processing.steps.device_steps import DeviceStep

    params_dtype = resolve_dtype(cfg.params_dtype)
    processor = PI05Processor.from_pretrained(
        cfg.checkpoint_path,
        tokenizer_name=cfg.tokenizer_name,
        image_size=cfg.image_size,
        num_images=cfg.num_real_cameras,
        action_dim=cfg.action_dim,
        normalize_pixels=True,
        device=cfg.device,
        params_dtype=params_dtype,
    )
    reorder_postprocessor_slice_first(processor)
    for step in processor.preprocessor.steps:
        if isinstance(step, DeviceStep):
            step.float_dtype = params_dtype
            step._float_dtype = params_dtype
    return processor
