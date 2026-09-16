"""GR00TProcessor construction for calibration."""

from __future__ import annotations

from qvla.adapters.groot.config import GR00TAdapterConfig


def create_groot_processor(cfg: GR00TAdapterConfig):
    from phyai_utils_tools.models.gr00t import GR00TProcessor

    return GR00TProcessor.from_pretrained(
        cfg.checkpoint_path,
        embodiment_tag=cfg.embodiment_tag,
        model_name=cfg.processor_model_name_or_path,
        transformers_loading_kwargs={"local_files_only": True},
    )
