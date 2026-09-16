"""Calibration batch iterators for GR00T-N1.7."""

from __future__ import annotations

import logging
from typing import Any, Iterator

import numpy as np
import torch

from qvla.adapters.groot.config import GR00TAdapterConfig


logger = logging.getLogger(__name__)

_TOKEN_SCOPES = ("all", "image", "image_lang_pad")


def groot_outlier_token_keep_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    token_scope: str,
    image_token_id: int,
    video_token_id: int | None = None,
) -> torch.Tensor:
    """Bool mask over one GR00T/Qwen3-VL sequence for adaptive tip-clip.

    ``True`` = tip (used to fit ``a_j``). Matches the visualization labels:
    ``attention_mask==0`` → ``lang_pad``; ``input_ids==image_token_id`` (valid)
    → image; remaining valid tokens → task (rest / floor).
    """
    if token_scope not in _TOKEN_SCOPES:
        raise ValueError(
            "token_scope must be 'all', 'image' or 'image_lang_pad', "
            f"got {token_scope!r}."
        )
    ids = input_ids.detach()
    attn = attention_mask.detach()
    if ids.ndim == 2 and ids.shape[0] == 1:
        ids = ids[0]
        attn = attn[0] if attn.ndim == 2 else attn
    if ids.ndim != 1 or attn.ndim != 1:
        raise ValueError(
            "input_ids/attention_mask must be 1-D (or batch-1 2-D); "
            f"got {tuple(input_ids.shape)} / {tuple(attention_mask.shape)}."
        )
    if ids.numel() != attn.numel():
        raise ValueError(
            f"input_ids len {ids.numel()} != attention_mask len {attn.numel()}."
        )
    n = int(ids.numel())
    if token_scope == "all":
        return torch.ones(n, dtype=torch.bool)
    valid = attn.to(torch.long) != 0
    image = valid & (ids.to(torch.long) == int(image_token_id))
    if video_token_id is not None:
        image = image | (valid & (ids.to(torch.long) == int(video_token_id)))
    if token_scope == "image":
        return image.cpu()
    return (image | ~valid).cpu()


_TASK_POOL = (
    "pick up the object and place it in the basket",
    "open the drawer and put the cup inside",
    "move the red block to the left side of the table",
    "stack the small box on top of the large box",
    "close the door of the microwave gently",
)


def iter_calibration_batches(
    cfg: GR00TAdapterConfig, processor, num_samples: int
) -> Iterator[dict]:
    if cfg.calibration_source == "file":
        yield from _iter_file_batches(cfg, processor, num_samples)
        return
    yield from _iter_synthetic_batches(cfg, processor, num_samples, seed=cfg.seed)


def _iter_synthetic_batches(
    cfg: GR00TAdapterConfig, processor, num_samples: int, *, seed: int
) -> Iterator[dict]:
    """Generate synthetic calibration observations matching the processor modality config."""
    rng = np.random.default_rng(seed)
    modality_config = processor.modality_config
    tag = processor.embodiment_tag

    video_keys = modality_config["video"].modality_keys
    state_keys = modality_config["state"].modality_keys
    language_key = modality_config["language"].modality_keys[0]
    video_time = len(modality_config["video"].delta_indices)
    state_time = len(modality_config["state"].delta_indices)
    language_time = len(modality_config["language"].delta_indices)

    for i in range(num_samples):
        video = {}
        for key in video_keys:
            video[key] = rng.integers(
                0, 256,
                size=(1, video_time, cfg.image_size, cfg.image_size, 3),
                dtype=np.uint8,
            )
        state = {}
        for key in state_keys:
            dim = int(processor.norm_params[tag]["state"][key]["dim"])
            state[key] = (rng.random((1, state_time, dim)) * 2 - 1).astype(np.float32)
        language = {
            language_key: [[_TASK_POOL[i % len(_TASK_POOL)] for _ in range(language_time)]]
        }
        yield {"video": video, "state": state, "language": language}


def _expand_video_frame(img: np.ndarray, *, t_video: int) -> np.ndarray:
    """``(H,W,3)`` or ``(T,H,W,3)`` → ``(1,T,H,W,3)`` uint8."""
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim == 3:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 4:
        raise ValueError(f"Expected image shape (H,W,3) or (T,H,W,3); got {arr.shape}.")
    if arr.shape[0] < t_video:
        arr = np.repeat(arr, t_video, axis=0)[:t_video]
    elif arr.shape[0] > t_video:
        arr = arr[:t_video]
    return arr[np.newaxis, ...]


def file_batch_to_groot_obs(batch: dict[str, Any], processor) -> dict[str, Any]:
    """Convert a version-1 calibration file batch (PI05-shaped) to GR00T obs.

    File batches use ``images`` / ``states`` / ``task_description``. GR00T
    expects ``video`` / ``state`` / ``language`` keyed by the processor modality
    config (e.g. libero_sim ``image`` + ``wrist_image``).
    """
    if "video" in batch and "state" in batch and "language" in batch:
        return batch

    modality_config = processor.modality_config
    tag = processor.embodiment_tag
    video_keys = modality_config["video"].modality_keys
    state_keys = modality_config["state"].modality_keys
    language_key = modality_config["language"].modality_keys[0]
    t_video = len(modality_config["video"].delta_indices)
    t_state = len(modality_config["state"].delta_indices)
    t_lang = len(modality_config["language"].delta_indices)

    images = batch.get("images", {})
    if not isinstance(images, dict) or not images:
        raise ValueError(
            "Calibration file batch must contain an 'images' dict "
            f"(or a ready GR00T 'video' dict); got keys {sorted(batch)}."
        )

    video: dict[str, np.ndarray] = {}
    img_items = list(images.items())
    for idx, key in enumerate(video_keys):
        if key in images:
            frame = images[key]
        elif idx < len(img_items):
            frame = img_items[idx][1]
        else:
            raise ValueError(
                f"Missing video key {key!r} in calibration images "
                f"(available: {sorted(images)})."
            )
        video[key] = _expand_video_frame(frame, t_video=t_video)

    if "states" not in batch:
        raise ValueError(
            "Calibration file batch requires 'states' "
            f"(or a ready GR00T 'state' dict); got keys {sorted(batch)}."
        )
    state_arr = np.asarray(batch["states"], dtype=np.float32).reshape(-1)
    state: dict[str, np.ndarray] = {}
    offset = 0
    for key in state_keys:
        dim = int(processor.norm_params[tag]["state"][key]["dim"])
        if offset + dim > state_arr.size:
            raise ValueError(
                f"State vector length {state_arr.size} is too short for modality "
                f"keys {state_keys} (need >= {offset + dim} at key {key!r})."
            )
        val = state_arr[offset : offset + dim]
        state[key] = np.broadcast_to(val.reshape(1, 1, dim), (1, t_state, dim)).copy()
        offset += dim

    task = str(batch.get("task_description", ""))
    language = {language_key: [[task for _ in range(t_lang)]]}
    return {"video": video, "state": state, "language": language}


def _iter_file_batches(
    cfg: GR00TAdapterConfig, processor, num_samples: int
) -> Iterator[dict]:
    from qvla.calibration.file import make_file_calibration_provider

    if cfg.calibration_data_path is None:
        raise ValueError("calibration_source='file' requires calibration_data_path.")
    provider = make_file_calibration_provider(cfg.calibration_data_path)
    meta = getattr(provider, "__calibration_meta__", {})
    logger.info(
        "GR00TAdapter: file calibration (%s, source=%s, suite=%s).",
        cfg.calibration_data_path,
        meta.get("source", "unknown"),
        meta.get("suite", "n/a"),
    )
    for batch in provider(num_samples):
        yield file_batch_to_groot_obs(batch, processor)
