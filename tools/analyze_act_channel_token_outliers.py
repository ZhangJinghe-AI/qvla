#!/usr/bin/env python
"""Locate activation outlier channels and classify which tokens cause them.

For target LLM linears matching ``--layer-regex`` (one or many), collects
pre-linear activations on calibration samples, then for each linear:

* writes a scatter of per-channel outlier clip % (all channels)
* for the worst ``--top-channels`` (by ``--top-by amax`` or ``clip``), prints
  a report and keeps the existing per-channel scatter / image-patch plots

Token labels::

    image[<cam>] | task | state | action_prompt | lang_pad

Outlier rule (matches QVLA adaptive act_clip; pick exactly one method)::

    tip = tokens selected by --fit-tokens
          (default image_lang_pad; choices: image|image_lang_pad|all)
    A) κ × P_β:  tip_thr = κ · P_β(|x_tip|)
    B) mean+k·std: tip_thr = μ + k · σ  (|x_tip|; population std)
    a_j = min(max(|x_tip|), tip_thr)
    a_j = max(a_j, max(|x_rest|))   # floor so task/state/action are not clipped
    outlier iff |x| > a_j

Default: ``--outlier-kappa/--outlier-std-k/--outlier-bulk-percentile`` are all
``0`` (off). Enable exactly one method by setting its args ``>0``:

* κ × P_β: ``--outlier-kappa 2 --outlier-bulk-percentile 95``
* mean+k·std: ``--outlier-std-k 3``

Tip scope is set by ``--fit-tokens`` (default ``image_lang_pad``).

Supports ``--model pi05`` and ``--model groot_n17``.

pi0.5 prefix layout (per sample)::

    [ image tokens | language tokens (task + state bins + Action: ...) | pad ]

GR00T-N1.7 VLM layout (per sample, Qwen3-VL)::

    left pad | chat/image tokens (interleaved) | task text
    image tokens are labeled from ``input_ids == image_token_id``;
    non-image content tokens are labeled ``task`` (no State:/Action: split).

Example::

    # pi0.5 — κ × P95
    CUDA_VISIBLE_DEVICES=4 uv run python tools/analyze_act_channel_token_outliers.py \\
        --model pi05 \\
        --checkpoint /data/share/pi05_libero_finetuned_v044 \\
        --calibration-data ../calibration_data/libero_goal_30_7_demo.npz \\
        --num-samples 8 \\
        --layer-regex 'paligemma_lm\\.layers\\.\\d+\\.mlp\\.down_proj$' \\
        --outlier-kappa 2 --outlier-bulk-percentile 95

    # GR00T-N1.7 — mean + 3·std
    CUDA_VISIBLE_DEVICES=4 uv run python tools/analyze_act_channel_token_outliers.py \\
        --model groot_n17 \\
        --checkpoint /data/share/GR00T-N1.7-LIBERO/libero_goal \\
        --embodiment-tag LIBERO_PANDA \\
        --calibration-data ../calibration_data/libero_goal_30_7_demo.npz \\
        --num-samples 8 \\
        --layer-regex 'backbone\\.qwen3vl_model\\.model\\.language_model\\.layers\\.\\d+\\.mlp\\.down_proj$' \\
        --outlier-std-k 3
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from qvla.adapters import get_adapter  # noqa: E402
from qvla.adapters.groot.config import (  # noqa: E402
    DEFAULT_PROCESSOR_MODEL_NAME_OR_PATH,
)
from qvla.adapters.groot.obs import build_groot_request  # noqa: E402
from qvla.adapters.pi05.obs import build_pi05_request  # noqa: E402
from qvla.config import QVLAConfig  # noqa: E402
from qvla.core.clip import (  # noqa: E402
    channel_outlier_bulk_kappa_amax,
    channel_outlier_mean_std_amax,
)
from qvla.runtime import list_target_modules  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_FIT_TOKENS = "image_lang_pad"
MODEL_CHOICES = ("pi05", "groot_n17")
DEFAULT_LAYER_REGEX = {
    "pi05": r"paligemma_lm\.layers\.15\.mlp\.down_proj$",
    "groot_n17": (
        r"backbone\.qwen3vl_model\.model\.language_model\.layers\.15"
        r"\.mlp\.down_proj$"
    ),
}

TOKEN_TYPES = (
    "image",
    "task",
    "state",
    "action_prompt",
    "lang_pad",
)
TOKEN_COLORS = {
    "image": "#4C78A8",
    "task": "#F58518",
    "state": "#E45756",
    "action_prompt": "#72B7B2",
    "lang_pad": "#B8B8B8",
}


@dataclass(frozen=True)
class SampleMeta:
    sample_index: int
    n_per_sample: int
    n_img: int
    num_patches: int
    num_images: int
    lang_len: int
    lang_bucket: int
    prompt: str
    labels: tuple[str, ...]  # length == n_per_sample
    # Display images (num_images, H, W, 3) uint8 — model-facing for pi05,
    # raw video frames for groot (overlay may be approximate / skipped).
    images: np.ndarray
    # Per-image merged patch counts; empty means uniform ``num_patches``.
    patches_per_image: tuple[int, ...] = ()
    # Per-image merged (grid_h, grid_w); empty → derive square grid from patches.
    image_grids: tuple[tuple[int, int], ...] = ()


def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING - 10 * min(verbosity, 2)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _prompt_char_spans(prompt: str) -> dict[str, tuple[int, int]]:
    """Character spans for task / state / action_prompt inside a pi0.5 prompt.

    ``"State: "`` is included in ``task`` so every character maps to exactly one
    span; ``state`` covers only the bin digits; ``action_prompt`` includes
    ``"Action: "``.
    """
    state_marker = "State: "
    action_marker = "Action: "
    state_at = prompt.find(state_marker)
    action_at = prompt.find(action_marker)
    if state_at < 0 or action_at < 0 or action_at < state_at:
        raise ValueError(
            "Prompt missing expected 'State: ' / 'Action: ' layout: "
            f"state_at={state_at} action_at={action_at} prompt={prompt!r}"
        )
    state_body = state_at + len(state_marker)
    return {
        "task": (0, state_body),
        "state": (state_body, action_at),
        "action_prompt": (action_at, len(prompt)),
    }


def _classify_lang_token(
    start: int,
    end: int,
    spans: dict[str, tuple[int, int]],
) -> str:
    """Map one token's character span to task / state / action_prompt."""
    # PaliGemma BOS (and similar specials) use an empty offset mapping (0, 0);
    # they sit at the start of the language prefix, so count them as task.
    if end <= start:
        return "task"
    mid = (start + end) // 2
    for name in ("state", "action_prompt", "task"):
        a, b = spans[name]
        if a <= mid < b:
            return name
    for name in ("state", "action_prompt", "task"):
        a, b = spans[name]
        if start < b and end > a:
            return name
    raise ValueError(
        f"Lang token offsets [{start}, {end}) fall outside prompt spans {spans}."
    )


def _label_prefix_tokens(
    *,
    tokenizer,
    prompt: str,
    n_img: int,
    num_patches: int,
    num_images: int,
    lang_len: int,
    n_per_sample: int,
) -> list[str]:
    """Build per-position labels for one packed prefix of length ``n_per_sample``."""
    if n_per_sample < n_img:
        raise ValueError(f"n_per_sample={n_per_sample} < n_img={n_img}.")
    if lang_len < 0 or lang_len > n_per_sample - n_img:
        raise ValueError(
            f"lang_len={lang_len} incompatible with n_img={n_img}, "
            f"n_per_sample={n_per_sample}."
        )

    if num_patches <= 0:
        raise ValueError(f"num_patches must be > 0, got {num_patches}.")
    if n_img != num_patches * num_images:
        raise ValueError(
            f"n_img={n_img} != num_patches*num_images="
            f"{num_patches * num_images}."
        )

    labels: list[str] = []
    for i in range(n_img):
        cam = i // num_patches
        labels.append(f"image[{cam}]")

    enc = tokenizer(
        prompt,
        add_special_tokens=True,
        return_offsets_mapping=True,
        return_attention_mask=False,
        truncation=False,
    )
    offsets = list(enc["offset_mapping"])
    # Match the processor path: right-pad / truncate is applied separately;
    # real lang tokens are the first ``lang_len`` ids.
    if len(offsets) < lang_len:
        raise RuntimeError(
            f"Tokenizer produced {len(offsets)} tokens but lang_len={lang_len}. "
            f"prompt[:80]={prompt[:80]!r}"
        )
    spans = _prompt_char_spans(prompt)
    for tok_i in range(lang_len):
        start, end = offsets[tok_i]
        labels.append(_classify_lang_token(int(start), int(end), spans))

    lang_bucket = n_per_sample - n_img
    for _ in range(lang_len, lang_bucket):
        labels.append("lang_pad")
    if len(labels) != n_per_sample:
        raise RuntimeError(
            f"Label length {len(labels)} != n_per_sample={n_per_sample}."
        )
    return labels


def _patches_per_image(meta: SampleMeta) -> tuple[int, ...]:
    if meta.patches_per_image:
        return meta.patches_per_image
    if meta.num_images <= 0:
        return ()
    return tuple(meta.num_patches for _ in range(meta.num_images))


def _label_groot_tokens(
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_token_id: int,
    patches_per_image: list[int] | tuple[int, ...],
) -> list[str]:
    """Label one GR00T/Qwen3-VL sequence (left-pad + interleaved image tokens)."""
    if input_ids.ndim != 1 or attention_mask.ndim != 1:
        raise ValueError(
            "input_ids/attention_mask must be 1-D; "
            f"got {tuple(input_ids.shape)} / {tuple(attention_mask.shape)}."
        )
    if input_ids.numel() != attention_mask.numel():
        raise ValueError(
            f"input_ids len {input_ids.numel()} != attention_mask "
            f"len {attention_mask.numel()}."
        )
    ids = input_ids.detach().to(torch.long).cpu()
    attn = attention_mask.detach().to(torch.long).cpu()
    ppi = [int(p) for p in patches_per_image]
    if any(p <= 0 for p in ppi):
        raise ValueError(f"patches_per_image must be >0, got {ppi}.")

    n_img_expected = int(sum(ppi))
    n_img_ids = int((ids == int(image_token_id)).sum().item())
    if n_img_ids != n_img_expected:
        raise RuntimeError(
            f"image token count from input_ids={n_img_ids} != "
            f"sum(patches_per_image)={n_img_expected} (ppi={ppi})."
        )

    # Cumulative ends so image token i maps to camera cam.
    ends: list[int] = []
    acc = 0
    for p in ppi:
        acc += p
        ends.append(acc)

    labels: list[str] = []
    img_seen = 0
    for pos in range(ids.numel()):
        if int(attn[pos].item()) == 0:
            labels.append("lang_pad")
            continue
        if int(ids[pos].item()) == int(image_token_id):
            cam = 0
            while cam < len(ends) and img_seen >= ends[cam]:
                cam += 1
            if cam >= len(ends):
                raise RuntimeError(
                    f"Ran out of cameras while labeling image token "
                    f"{img_seen}/{n_img_expected}."
                )
            labels.append(f"image[{cam}]")
            img_seen += 1
        else:
            labels.append("task")
    if img_seen != n_img_expected:
        raise RuntimeError(
            f"Labeled {img_seen} image tokens, expected {n_img_expected}."
        )
    return labels


def _extend_labels_to_length(labels: list[str], n: int) -> list[str]:
    """Pad trailing bucket / CUDA-graph right-pad tokens as ``lang_pad``."""
    if n < len(labels):
        raise RuntimeError(
            f"Activation length {n} < labeled tokens {len(labels)}."
        )
    if n == len(labels):
        return labels
    return labels + ["lang_pad"] * (n - len(labels))


def _groot_patches_and_grids(
    image_grid_thw: torch.Tensor,
    *,
    merge_size: int,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Merged patch counts and (H,W) grids from Qwen3-VL ``image_grid_thw``."""
    if image_grid_thw.ndim != 2 or image_grid_thw.shape[-1] != 3:
        raise ValueError(
            f"image_grid_thw must be (N,3), got {tuple(image_grid_thw.shape)}."
        )
    if merge_size <= 0:
        raise ValueError(f"merge_size must be >0, got {merge_size}.")
    merge_area = int(merge_size) ** 2
    patches: list[int] = []
    grids: list[tuple[int, int]] = []
    for row in image_grid_thw.detach().to(torch.long).cpu():
        t, h, w = (int(row[0]), int(row[1]), int(row[2]))
        if t <= 0 or h <= 0 or w <= 0:
            raise ValueError(f"bad image_grid_thw row={(t, h, w)}.")
        if h % merge_size != 0 or w % merge_size != 0:
            raise ValueError(
                f"grid_thw H/W={(h, w)} not divisible by merge_size={merge_size}."
            )
        gh, gw = h // merge_size, w // merge_size
        n = (t * h * w) // merge_area
        if n != t * gh * gw:
            raise RuntimeError(
                f"patch count mismatch: n={n} vs t*gh*gw={t * gh * gw}."
            )
        # Temporal>1 is flattened into one logical "image" stream for overlay;
        # use spatial grid of one frame (t copies of tokens still share cam id).
        patches.append(n)
        grids.append((gh * t, gw) if t > 1 else (gh, gw))
    return patches, grids


def _groot_display_images(batch: dict, processor) -> np.ndarray:
    """Stack video frames as (num_images, H, W, 3) uint8 in VLM image order."""
    keys = list(processor.modality_config["video"].modality_keys)
    frames: list[np.ndarray] = []
    for key in keys:
        arr = batch["video"][key]
        if not isinstance(arr, np.ndarray) or arr.ndim != 5:
            raise RuntimeError(
                f"video[{key!r}] must be (B,T,H,W,3) ndarray, got "
                f"{type(arr)} / {getattr(arr, 'shape', None)}."
            )
        for t in range(arr.shape[1]):
            frames.append(np.asarray(arr[0, t], dtype=np.uint8))
    if not frames:
        raise RuntimeError("No video frames found for GR00T display images.")
    return np.stack(frames, axis=0)


def _extract_groot_prompt(batch: dict) -> str:
    language = batch.get("language")
    if not isinstance(language, dict) or not language:
        return ""
    key = next(iter(language))
    hist = language[key]
    if isinstance(hist, list) and hist:
        first = hist[0]
        if isinstance(first, list) and first:
            return str(first[0])
        return str(first)
    return str(hist)


def _coarse_type(label: str) -> str:
    if label.startswith("image"):
        return "image"
    if label in TOKEN_TYPES:
        return label
    raise ValueError(f"Unexpected token label {label!r}.")


def _find_layers(
    model: torch.nn.Module,
    config: QVLAConfig,
    layer_regex: str,
) -> list[tuple[str, str, torch.nn.Module]]:
    pat = re.compile(layer_regex)
    hits = [
        (name, scope, mod)
        for name, scope, mod in list_target_modules(model, config)
        if pat.search(name)
    ]
    if not hits:
        raise SystemExit(f"No target layer matched --layer-regex={layer_regex!r}.")
    return hits


def _down_layer_index(layer_name: str) -> int:
    """Parse ``layers.<i>`` from a module name like ``...layers.15.mlp.down_proj``."""
    m = re.search(r"\.layers\.(\d+)\.", layer_name)
    if m is None:
        raise ValueError(
            f"Cannot parse down-layer index from layer name {layer_name!r}."
        )
    return int(m.group(1))


def _layer_out_dir(output_dir: Path, layer_name: str, *, disambiguate: bool) -> Path:
    """``output_dir/<layer_idx>/`` (or ``<layer_idx>_<leaf>/`` if needed)."""
    idx = _down_layer_index(layer_name)
    if disambiguate:
        leaf = layer_name.rsplit(".", 1)[-1]
        return output_dir / f"{idx}_{leaf}"
    return output_dir / str(idx)


def _pixel_values_to_uint8(pixel_values: torch.Tensor) -> np.ndarray:
    """Convert one sample ``(num_images, C, H, W)`` model pixels to uint8 RGB."""
    if pixel_values.ndim != 4:
        raise ValueError(
            f"Expected pixel_values (num_images, C, H, W), got {tuple(pixel_values.shape)}."
        )
    x = pixel_values.detach().to(dtype=torch.float32).cpu()
    # SigLIP path uses [-1, 1]; raw [0, 1] is left unchanged.
    if float(x.min().item()) < -1e-3:
        x = (x + 1.0) * 0.5
    x = x.clamp(0.0, 1.0).permute(0, 2, 3, 1).numpy()
    return np.rint(x * 255.0).astype(np.uint8)


def _install_act_hooks(
    layers: list[tuple[str, torch.nn.Module]],
    chunks: dict[str, list[torch.Tensor]],
):
    def _make_hook(layer_name: str):
        def hook(_mod, inputs):
            if not inputs or not torch.is_tensor(inputs[0]):
                raise RuntimeError(f"{layer_name}: expected tensor input.")
            x = inputs[0]
            if x.ndim < 2:
                raise RuntimeError(
                    f"{layer_name}: expected ndim>=2, got {tuple(x.shape)}."
                )
            chunks[layer_name].append(
                x.reshape(-1, x.shape[-1]).detach().to(torch.float32).cpu()
            )

        return hook

    return [
        module.register_forward_pre_hook(_make_hook(name)) for name, module in layers
    ]


def _collect_layers(
    adapter,
    model: torch.nn.Module,
    layers: list[tuple[str, torch.nn.Module]],
    *,
    num_samples: int,
    model_kind: str,
) -> tuple[dict[str, torch.Tensor], list[SampleMeta]]:
    """One calib pass; return ``{name: activation[T,K]}`` and shared sample metas."""
    if model_kind == "pi05":
        return _collect_layers_pi05(
            adapter, model, layers, num_samples=num_samples
        )
    if model_kind == "groot_n17":
        return _collect_layers_groot(
            adapter, model, layers, num_samples=num_samples
        )
    raise ValueError(f"Unsupported model_kind={model_kind!r}.")


def _collect_layers_pi05(
    adapter,
    model: torch.nn.Module,
    layers: list[tuple[str, torch.nn.Module]],
    *,
    num_samples: int,
) -> tuple[dict[str, torch.Tensor], list[SampleMeta]]:
    """pi0.5: contiguous ``[image | lang | pad]`` prefix."""
    if not layers:
        raise ValueError("layers must be non-empty.")
    sched = adapter.engine.entry.scheduler
    tokenizer = adapter._processor.tokenizer
    n_img = int(sched.image_token_count)
    num_images = int(sched.num_images)
    num_patches = int(sched.cfg.vision.num_patches)
    chunks: dict[str, list[torch.Tensor]] = {name: [] for name, _ in layers}
    metas: list[SampleMeta] = []

    handles = _install_act_hooks(layers, chunks)
    try:
        n_seen = 0
        for batch in adapter.iter_calibration_batches(num_samples):
            before = {name: len(chunks[name]) for name, _ in layers}
            request = build_pi05_request(
                adapter._processor, batch, state_dim=adapter.cfg.state_dim
            )
            if request.pixel_values.shape[0] != 1:
                raise RuntimeError(
                    "Expected batch size 1 for image overlay; got "
                    f"pixel_values shape {tuple(request.pixel_values.shape)}."
                )
            lang_len = int(request.lang_lens[0].item())
            n_ps = int(sched._bucket_n_per_sample(lang_len))
            prompt = _extract_prompt(adapter, batch)
            images = _pixel_values_to_uint8(request.pixel_values[0])
            if images.shape[0] != num_images:
                raise RuntimeError(
                    f"Sample {n_seen}: images={images.shape[0]} != num_images={num_images}."
                )

            adapter.forward_for_calibration(
                model, batch, step_callback=lambda _s: None
            )
            for name, _ in layers:
                got = len(chunks[name]) - before[name]
                if got != 1:
                    raise RuntimeError(
                        f"{name}: expected 1 activation chunk for sample "
                        f"{n_seen}, got {got}."
                    )
                act = chunks[name][-1]
                if act.shape[0] != n_ps:
                    raise RuntimeError(
                        f"{name} sample {n_seen}: activation tokens={act.shape[0]} != "
                        f"n_per_sample={n_ps} (lang_len={lang_len}, n_img={n_img})."
                    )
            labels = _label_prefix_tokens(
                tokenizer=tokenizer,
                prompt=prompt,
                n_img=n_img,
                num_patches=num_patches,
                num_images=num_images,
                lang_len=lang_len,
                n_per_sample=n_ps,
            )
            metas.append(
                SampleMeta(
                    sample_index=n_seen,
                    n_per_sample=n_ps,
                    n_img=n_img,
                    num_patches=num_patches,
                    num_images=num_images,
                    lang_len=lang_len,
                    lang_bucket=n_ps - n_img,
                    prompt=prompt,
                    labels=tuple(labels),
                    images=images,
                    patches_per_image=tuple(
                        num_patches for _ in range(num_images)
                    ),
                )
            )
            n_seen += 1
        if n_seen != num_samples:
            raise RuntimeError(f"Expected {num_samples} samples, got {n_seen}.")
    finally:
        for handle in handles:
            handle.remove()

    activations = {
        name: torch.cat(chunks[name], dim=0) for name, _ in layers
    }
    return activations, metas


def _collect_layers_groot(
    adapter,
    model: torch.nn.Module,
    layers: list[tuple[str, torch.nn.Module]],
    *,
    num_samples: int,
) -> tuple[dict[str, torch.Tensor], list[SampleMeta]]:
    """GR00T-N1.7: Qwen3-VL interleaved image tokens + left/right pad."""
    if not layers:
        raise ValueError("layers must be non-empty.")
    qwen = model.backbone.qwen3vl_model
    image_token_id = int(qwen.config.image_token_id)
    merge_size = int(qwen.model.visual.spatial_merge_size)
    chunks: dict[str, list[torch.Tensor]] = {name: [] for name, _ in layers}
    metas: list[SampleMeta] = []

    handles = _install_act_hooks(layers, chunks)
    try:
        n_seen = 0
        for batch in adapter.iter_calibration_batches(num_samples):
            before = {name: len(chunks[name]) for name, _ in layers}
            request = build_groot_request(
                adapter._processor, batch, device=adapter.cfg.device
            )
            tensors = request.tensors
            input_ids = tensors["input_ids"]
            attention_mask = tensors["attention_mask"]
            image_grid_thw = tensors["image_grid_thw"]
            if input_ids.shape[0] != 1:
                raise RuntimeError(
                    "Expected batch size 1 for GR00T token labeling; got "
                    f"input_ids shape {tuple(input_ids.shape)}."
                )
            ppi, grids = _groot_patches_and_grids(
                image_grid_thw, merge_size=merge_size
            )
            num_images = len(ppi)
            n_img = int(sum(ppi))
            num_patches = int(ppi[0]) if ppi else 0
            prompt = _extract_groot_prompt(batch)
            images = _groot_display_images(batch, adapter._processor)
            if images.shape[0] != num_images:
                raise RuntimeError(
                    f"Sample {n_seen}: display images={images.shape[0]} != "
                    f"num_images={num_images} from image_grid_thw."
                )

            labels = _label_groot_tokens(
                input_ids=input_ids[0],
                attention_mask=attention_mask[0],
                image_token_id=image_token_id,
                patches_per_image=ppi,
            )
            lang_len = int((attention_mask[0] != 0).sum().item())

            adapter.forward_for_calibration(
                model, batch, step_callback=lambda _s: None
            )
            n_ps = None
            for name, _ in layers:
                got = len(chunks[name]) - before[name]
                if got != 1:
                    raise RuntimeError(
                        f"{name}: expected 1 activation chunk for sample "
                        f"{n_seen}, got {got}."
                    )
                act = chunks[name][-1]
                if n_ps is None:
                    n_ps = int(act.shape[0])
                elif int(act.shape[0]) != n_ps:
                    raise RuntimeError(
                        f"{name} sample {n_seen}: activation tokens="
                        f"{act.shape[0]} != {n_ps} from sibling layer."
                    )
            assert n_ps is not None
            labels = _extend_labels_to_length(labels, n_ps)
            metas.append(
                SampleMeta(
                    sample_index=n_seen,
                    n_per_sample=n_ps,
                    n_img=n_img,
                    num_patches=num_patches,
                    num_images=num_images,
                    lang_len=lang_len,
                    lang_bucket=n_ps - lang_len,
                    prompt=prompt,
                    labels=tuple(labels),
                    images=images,
                    patches_per_image=tuple(ppi),
                    image_grids=tuple(grids),
                )
            )
            n_seen += 1
        if n_seen != num_samples:
            raise RuntimeError(f"Expected {num_samples} samples, got {n_seen}.")
    finally:
        for handle in handles:
            handle.remove()

    activations = {
        name: torch.cat(chunks[name], dim=0) for name, _ in layers
    }
    return activations, metas


def _extract_prompt(adapter, batch: dict) -> str:
    """Re-run the preprocessor's prompt assembly for one observation."""
    from qvla.adapters.pi05.obs import obs_to_transition
    from phyai_utils_tools.processing.transition import PROMPT, TASK

    transition = obs_to_transition(batch, state_dim=adapter.cfg.state_dim)
    # Walk preprocessor steps until PROMPT exists (StateTokenizerPrepareStep).
    cur = transition
    for step in adapter._processor.preprocessor.steps:
        cur = step(cur)
        if PROMPT in cur:
            prompts = cur[PROMPT]
            if isinstance(prompts, list):
                return str(prompts[0])
            return str(prompts)
    tasks = transition.get(TASK, "")
    if isinstance(tasks, list):
        return str(tasks[0])
    return str(tasks)


def _all_labels(metas: list[SampleMeta]) -> list[str]:
    out: list[str] = []
    for m in metas:
        out.extend(m.labels)
    return out


def _channel_outliers(
    activation: torch.Tensor,
    *,
    top_k: int,
) -> list[tuple[int, float, float, float]]:
    """Return ``(channel, amax, median_over_ch, amax/median)`` for top channels."""
    if activation.ndim != 2:
        raise ValueError(f"Expected (T,K), got {tuple(activation.shape)}.")
    abs_x = activation.abs()
    ch_amax = abs_x.amax(dim=0)
    median = float(ch_amax.median().item())
    if median <= 0:
        raise RuntimeError("Channel-amax median is non-positive.")
    top_vals, top_idx = torch.topk(ch_amax, k=min(top_k, ch_amax.numel()))
    return [
        (int(i.item()), float(v.item()), median, float(v.item()) / median)
        for i, v in zip(top_idx, top_vals)
    ]


def _channel_outliers_by_clip(
    clip_pct: np.ndarray,
    activation: torch.Tensor,
    *,
    top_k: int,
) -> list[tuple[int, float, float, float]]:
    """Like ``_channel_outliers`` but ranked by clip% instead of amax."""
    if activation.ndim != 2:
        raise ValueError(f"Expected (T,K), got {tuple(activation.shape)}.")
    abs_x = activation.abs()
    ch_amax = abs_x.amax(dim=0)
    median = float(ch_amax.median().item())
    if median <= 0:
        raise RuntimeError("Channel-amax median is non-positive.")
    order = np.argsort(-clip_pct)[: min(top_k, len(clip_pct))]
    return [
        (
            int(i),
            float(ch_amax[i].item()),
            median,
            float(ch_amax[i].item()) / median,
        )
        for i in order
    ]


def _tip_token_keep(
    labels: list[str],
    *,
    fit_tokens: str = DEFAULT_FIT_TOKENS,
) -> torch.Tensor:
    """Bool mask over tokens used for tip-clip (same scopes as QVLA collector)."""
    if fit_tokens == "all":
        return torch.ones(len(labels), dtype=torch.bool)
    if fit_tokens == "image":
        tip_types = {"image"}
    elif fit_tokens == "image_lang_pad":
        tip_types = {"image", "lang_pad"}
    else:
        raise ValueError(
            "fit_tokens must be 'all'|'image'|'image_lang_pad', "
            f"got {fit_tokens!r}."
        )
    return torch.tensor(
        [_coarse_type(lab) in tip_types for lab in labels], dtype=torch.bool
    )


def _resolve_outlier_rule(
    *,
    kappa: float,
    bulk_percentile: float,
    std_k: float,
) -> tuple[float, float | None, float]:
    """Return ``(kappa, bulk_percentile, std_k)`` with exactly one method on.

    Defaults are all ``0`` (off). Exactly one of ``kappa`` / ``std_k`` must be
    ``>0``. When ``kappa>0``, ``bulk_percentile`` must be in ``(0, 100)``; when
    ``std_k>0``, ``bulk_percentile`` must stay ``0``.
    """
    kappa_f = float(kappa)
    std_k_f = float(std_k)
    bulk_f = float(bulk_percentile)
    if kappa_f < 0.0:
        raise SystemExit(f"--outlier-kappa must be >= 0, got {kappa_f}.")
    if std_k_f < 0.0:
        raise SystemExit(f"--outlier-std-k must be >= 0, got {std_k_f}.")
    if bulk_f < 0.0:
        raise SystemExit(
            f"--outlier-bulk-percentile must be >= 0, got {bulk_f}."
        )
    if kappa_f > 0.0 and std_k_f > 0.0:
        raise SystemExit(
            "--outlier-kappa>0 and --outlier-std-k>0 are mutually exclusive "
            f"(got kappa={kappa_f}, std_k={std_k_f})."
        )
    if kappa_f <= 0.0 and std_k_f <= 0.0:
        raise SystemExit(
            "Enable exactly one outlier method: set --outlier-kappa>0 or "
            f"--outlier-std-k>0 (got kappa={kappa_f}, std_k={std_k_f})."
        )
    if kappa_f > 0.0:
        if not (0.0 < bulk_f < 100.0):
            raise SystemExit(
                "--outlier-bulk-percentile must be in (0, 100) when "
                f"--outlier-kappa>0, got {bulk_f}."
            )
        return kappa_f, bulk_f, std_k_f
    if bulk_f != 0.0:
        raise SystemExit(
            "--outlier-bulk-percentile must be 0 when using --outlier-std-k "
            f"(got bulk={bulk_f} with std_k={std_k_f})."
        )
    return kappa_f, None, std_k_f


def _outlier_rule_label(
    *, kappa: float, bulk_percentile: float | None, std_k: float
) -> str:
    if std_k > 0.0 and kappa > 0.0:
        raise ValueError(
            f"both methods on: kappa={kappa}, std_k={std_k}."
        )
    if std_k > 0.0:
        return f"mean+{std_k:g}·std"
    if kappa > 0.0:
        if bulk_percentile is None:
            raise ValueError("kappa>0 requires bulk_percentile.")
        return f"κ={kappa:g}×P{bulk_percentile:g}"
    raise ValueError(
        f"no outlier method enabled (kappa={kappa}, std_k={std_k})."
    )


def _tip_clip_levels(
    tip: torch.Tensor,
    *,
    kappa: float,
    bulk_percentile: float | None,
    std_k: float,
) -> tuple[torch.Tensor, str]:
    """Per-channel tip clip + one-line detail (from channel 0 if K>1)."""
    if tip.ndim != 2:
        raise ValueError(f"tip must be (T,K), got {tuple(tip.shape)}.")
    if std_k > 0.0 and kappa > 0.0:
        raise ValueError(
            f"both methods on: kappa={kappa}, std_k={std_k}."
        )
    tip_f = tip.to(torch.float32)
    if std_k > 0.0:
        if tip_f.shape[0] < 2:
            raise RuntimeError(
                f"mean+std tip-clip needs >= 2 tip tokens, got T={int(tip_f.shape[0])}."
            )
        clip = channel_outlier_mean_std_amax(tip_f, std_k=std_k)
        mean0 = float(tip_f[:, 0].mean().item())
        std0 = float(tip_f[:, 0].std(unbiased=False).item())
        thr0 = mean0 + float(std_k) * std0
        detail = (
            f"mean={mean0:.4g} std={std0:.4g} → mean+{std_k:g}·std={thr0:.4g}"
        )
        return clip, detail
    if kappa > 0.0:
        if bulk_percentile is None:
            raise ValueError("kappa>0 requires bulk_percentile.")
        if tip_f.shape[0] < 1:
            raise RuntimeError("κ×P_β tip-clip needs >= 1 tip token.")
        clip = channel_outlier_bulk_kappa_amax(
            tip_f, kappa=kappa, bulk_percentile=bulk_percentile
        )
        ref0 = float(
            torch.quantile(tip_f[:, 0], float(bulk_percentile) / 100.0).item()
        )
        thr0 = float(kappa) * max(ref0, 1e-30)
        detail = (
            f"κ={kappa:g}×P{bulk_percentile:g}={ref0:.4g} → tip_thr={thr0:.4g}"
        )
        return clip, detail
    raise ValueError(
        f"no outlier method enabled (kappa={kappa}, std_k={std_k})."
    )


def _token_outlier_mask(
    col: torch.Tensor,
    labels: list[str],
    *,
    kappa: float,
    bulk_percentile: float | None,
    std_k: float,
    fit_tokens: str = DEFAULT_FIT_TOKENS,
) -> tuple[torch.Tensor, float, float, str]:
    """Mask ``|x| > a_j`` with tip-clip + rest-floor.

    Returns ``(mask, clip_level, rest_max, tip_detail)``.
    """
    if int(col.numel()) != len(labels):
        raise ValueError(
            f"col length {int(col.numel())} != labels={len(labels)}."
        )
    v = col.abs().to(torch.float64)
    if v.numel() < 1:
        raise ValueError("token column is empty.")
    tip_keep = _tip_token_keep(labels, fit_tokens=fit_tokens)
    tip = v[tip_keep]
    if tip.numel() < 1:
        raise RuntimeError(
            f"fit_tokens={fit_tokens!r} selected zero tip tokens."
        )
    tip_clip_t, tip_detail = _tip_clip_levels(
        tip.unsqueeze(1),
        kappa=kappa,
        bulk_percentile=bulk_percentile,
        std_k=std_k,
    )
    tip_clip = float(tip_clip_t[0].item())
    rest = v[~tip_keep]
    rest_max = float(rest.max().item()) if rest.numel() else 0.0
    clip_level = max(tip_clip, rest_max) if rest.numel() else tip_clip
    return v > clip_level, clip_level, rest_max, tip_detail


def _channel_outlier_clip_pct(
    activation: torch.Tensor,
    labels: list[str],
    *,
    kappa: float,
    bulk_percentile: float | None,
    std_k: float,
    fit_tokens: str = DEFAULT_FIT_TOKENS,
) -> np.ndarray:
    """Per-channel outlier fraction (%) under tip-clip + rest-floor."""
    if activation.ndim != 2:
        raise ValueError(f"Expected (T,K), got {tuple(activation.shape)}.")
    if int(activation.shape[0]) != len(labels):
        raise ValueError(
            f"activation tokens={activation.shape[0]} != labels={len(labels)}."
        )
    abs_x = activation.abs().to(torch.float64)
    if abs_x.shape[0] < 1:
        raise ValueError("activation has zero tokens.")
    tip_keep = _tip_token_keep(labels, fit_tokens=fit_tokens)
    tip = abs_x[tip_keep]
    if tip.shape[0] < 1:
        raise RuntimeError(
            f"fit_tokens={fit_tokens!r} selected zero tip tokens."
        )
    tip_clip, _ = _tip_clip_levels(
        tip,
        kappa=kappa,
        bulk_percentile=bulk_percentile,
        std_k=std_k,
    )
    tip_clip = tip_clip.to(dtype=abs_x.dtype)
    rest = abs_x[~tip_keep]
    if rest.shape[0] > 0:
        clip_level = torch.maximum(tip_clip, rest.amax(dim=0))
    else:
        clip_level = tip_clip
    frac = (abs_x > clip_level.unsqueeze(0)).to(torch.float64).mean(dim=0)
    return (frac * 100.0).cpu().numpy()


def _plot_clip_pct_by_channel(
    pct: np.ndarray,
    *,
    output_path: Path,
    title: str,
) -> None:
    """Scatter: channel idx → outlier clip % for one linear."""
    if pct.ndim != 1:
        raise ValueError(f"Expected 1-D pct, got shape {pct.shape}.")
    xs = np.arange(int(pct.shape[0]))
    fig, ax = plt.subplots(figsize=(14, 4.5), constrained_layout=True)
    ax.scatter(xs, pct, s=6, c="#4C78A8", alpha=0.85, linewidths=0)
    ax.set_xlabel("channel idx")
    ax.set_ylabel("outlier clip %")
    ax.set_ylim(bottom=0.0)
    mean_pct = float(pct.mean()) if pct.size else 0.0
    max_pct = float(pct.max()) if pct.size else 0.0
    n_pos = int((pct > 0.0).sum()) if pct.size else 0
    ax.set_title(
        f"{title}\n"
        f"channels={pct.shape[0]}  clip%>0: {n_pos}  "
        f"mean={mean_pct:.2f}%  max={max_pct:.2f}%",
        fontsize=10,
    )
    ax.grid(True, alpha=0.3)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {output_path}")


def _summarize_by_type(
    labels: list[str],
    mask: torch.Tensor,
    values: torch.Tensor,
) -> list[tuple[str, int, int, float, float]]:
    """Per coarse type: (type, n_outlier, n_total, max_abs, mean_abs_outliers)."""
    abs_v = values.abs().cpu().numpy()
    m = mask.cpu().numpy().astype(bool)
    rows: list[tuple[str, int, int, float, float]] = []
    for t in TOKEN_TYPES:
        idxs = [i for i, lab in enumerate(labels) if _coarse_type(lab) == t]
        if not idxs:
            continue
        sub_m = m[idxs]
        n_out = int(sub_m.sum())
        max_abs = float(abs_v[idxs].max()) if idxs else 0.0
        mean_out = float(abs_v[idxs][sub_m].mean()) if n_out else 0.0
        rows.append((t, n_out, len(idxs), max_abs, mean_out))
    return rows


def _print_channel_report(
    *,
    channel: int,
    ch_amax: float,
    ch_med: float,
    ratio: float,
    labels: list[str],
    col: torch.Tensor,
    kappa: float,
    bulk_percentile: float | None,
    std_k: float,
    metas: list[SampleMeta],
    max_list: int,
    fit_tokens: str = DEFAULT_FIT_TOKENS,
) -> torch.Tensor:
    mask, thr, rest_max, tip_detail = _token_outlier_mask(
        col,
        labels,
        kappa=kappa,
        bulk_percentile=bulk_percentile,
        std_k=std_k,
        fit_tokens=fit_tokens,
    )
    n_out = int(mask.sum().item())
    n_tok = int(col.numel())
    abs_col = col.abs()
    print("\n" + "=" * 72)
    print(
        f"channel={channel}  channel_amax={ch_amax:.4g}  "
        f"ch_amax/median={ratio:.2f}  (median_ch_amax={ch_med:.4g})"
    )
    print(
        f"token outliers: {n_out}/{n_tok}  "
        f"(tip={fit_tokens}: {tip_detail}; "
        f"rest_max={rest_max:.4g} → clip={thr:.4g})"
    )
    print(
        f"token |x| stats: min={float(abs_col.min()):.4g}  "
        f"median={float(abs_col.median()):.4g}  "
        f"mean={float(abs_col.mean()):.4g}  "
        f"max={float(abs_col.max()):.4g}"
    )
    if n_out == n_tok:
        print(">>> ALL tokens are outliers on this channel.")
    elif n_out == 0:
        print(">>> No tokens exceed the token-level threshold.")
    else:
        print(">>> Only a SUBSET of tokens are outliers on this channel.")

    print("\nBy token type:")
    print(
        f"{'type':16s}  {'out':>6s}  {'total':>6s}  {'out%':>7s}  "
        f"{'max|x|':>10s}  {'mean|x|_out':>12s}"
    )
    for t, n_o, n_t, mx, mean_o in _summarize_by_type(labels, mask, col):
        pct = 100.0 * n_o / n_t if n_t else 0.0
        print(
            f"{t:16s}  {n_o:6d}  {n_t:6d}  {pct:6.1f}%  "
            f"{mx:10.4g}  {mean_o:12.4g}"
        )

    # Per-sample breakdown for image vs state vs task.
    print("\nPer-sample outlier counts (coarse types):")
    offset = 0
    for meta in metas:
        sl = slice(offset, offset + meta.n_per_sample)
        m_s = mask[sl]
        labs = list(meta.labels)
        counts: dict[str, int] = {}
        for i, flag in enumerate(m_s.tolist()):
            if not flag:
                continue
            counts[_coarse_type(labs[i])] = counts.get(_coarse_type(labs[i]), 0) + 1
        parts = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "(none)"
        print(
            f"  sample {meta.sample_index:02d}: lang_len={meta.lang_len}  "
            f"n_ps={meta.n_per_sample}  outliers={int(m_s.sum())}  [{parts}]"
        )
        print(f"    prompt: {meta.prompt[:100]!r}...")
        offset += meta.n_per_sample

    # List top individual outlier tokens.
    out_idx = torch.nonzero(mask, as_tuple=False).flatten()
    if out_idx.numel() > 0:
        scores = abs_col[out_idx]
        order = torch.argsort(scores, descending=True)
        show = order[:max_list]
        print(f"\nTop-{min(max_list, int(out_idx.numel()))} outlier tokens:")
        print(
            f"{'rank':>4s}  {'tok':>6s}  {'sample':>6s}  {'local':>6s}  "
            f"{'|x|':>10s}  {'type':16s}  detail"
        )
        for rank, oi in enumerate(show.tolist(), start=1):
            global_i = int(out_idx[oi].item())
            sample_i, local_i, detail = _locate_token(global_i, metas)
            lab = labels[global_i]
            print(
                f"{rank:4d}  {global_i:6d}  {sample_i:6d}  {local_i:6d}  "
                f"{float(abs_col[global_i]):10.4g}  {lab:16s}  {detail}"
            )
    return mask


def _locate_token(
    global_i: int, metas: list[SampleMeta]
) -> tuple[int, int, str]:
    offset = 0
    for meta in metas:
        if offset <= global_i < offset + meta.n_per_sample:
            local = global_i - offset
            label = meta.labels[local] if local < len(meta.labels) else "?"
            if label.startswith("image["):
                try:
                    cam = int(label[len("image[") : -1])
                except ValueError:
                    cam = -1
                ppi = _patches_per_image(meta)
                cam_locals = [
                    i
                    for i, lab in enumerate(meta.labels)
                    if lab == f"image[{cam}]"
                ]
                try:
                    patch = cam_locals.index(local)
                except ValueError:
                    patch = -1
                detail = (
                    f"{label} cam={cam} patch={patch}"
                    f"/{ppi[cam] if 0 <= cam < len(ppi) else '?'}"
                )
            else:
                detail = f"{label} local={local}/{meta.n_per_sample}"
            return meta.sample_index, local, detail
        offset += meta.n_per_sample
    return -1, -1, "?"


def _patch_grid(num_patches: int, image_hw: tuple[int, int]) -> tuple[int, int]:
    """Return ``(grid_side, patch_size)`` for a square patch grid on ``(H, W)``."""
    h, w = image_hw
    if h != w:
        raise ValueError(f"Expected square image for patch overlay, got H={h} W={w}.")
    grid = int(round(num_patches**0.5))
    if grid * grid != num_patches:
        raise ValueError(f"num_patches={num_patches} is not a square grid.")
    if h % grid != 0:
        raise ValueError(f"image size {h} not divisible by grid {grid}.")
    return grid, h // grid


def _image_grid_for_cam(meta: SampleMeta, cam: int) -> tuple[int, int] | None:
    """Merged (grid_h, grid_w) for one camera, or None if unknown/non-rect."""
    if 0 <= cam < len(meta.image_grids):
        return meta.image_grids[cam]
    ppi = _patches_per_image(meta)
    if not (0 <= cam < len(ppi)):
        return None
    n = ppi[cam]
    g = int(round(n**0.5))
    if g * g != n:
        return None
    return g, g


def _can_overlay_image_patches(metas: list[SampleMeta]) -> bool:
    """True when every sample/cam has a rectangular grid that tiles the display."""
    if not metas:
        return False
    for meta in metas:
        if meta.images.ndim != 4 or meta.images.shape[0] != meta.num_images:
            return False
        for cam in range(meta.num_images):
            grid = _image_grid_for_cam(meta, cam)
            if grid is None:
                return False
            gh, gw = grid
            h, w = int(meta.images.shape[1]), int(meta.images.shape[2])
            if gh <= 0 or gw <= 0 or h % gh != 0 or w % gw != 0:
                return False
            ppi = _patches_per_image(meta)
            if cam >= len(ppi) or ppi[cam] != gh * gw:
                return False
    return True


def _prompt_for_display(prompt: str, *, width: int = 42) -> str:
    """Wrapped prompt text for the per-sample label column."""
    text = " ".join(prompt.split())
    return textwrap.fill(text, width=width)


def _plot_image_outlier_patches(
    *,
    col: torch.Tensor,
    mask: torch.Tensor,
    metas: list[SampleMeta],
    channel: int,
    output_path: Path,
    title: str,
) -> None:
    """Overlay outlier image-patch boxes on the model-facing camera frames."""
    del channel  # reserved for callers / future per-channel styling
    if not metas:
        raise ValueError("metas is empty.")
    if not _can_overlay_image_patches(metas):
        print(
            f"Skip image patch overlay ({output_path}): non-square / "
            "non-tiling GR00T grids or missing display images; "
            "by_token.png is still written."
        )
        return
    num_images = metas[0].num_images
    n_samples = len(metas)
    abs_col = col.abs().cpu()
    # Extra left column holds the per-sample prompt text.
    fig, axes = plt.subplots(
        n_samples,
        num_images + 1,
        figsize=(3.2 + 4.2 * num_images, 4.0 * n_samples),
        squeeze=False,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.15] + [1.0] * num_images},
    )
    fig.suptitle(title, fontsize=12)
    cam_names = {0: "cam0/image", 1: "cam1/wrist"}

    offset = 0
    for row, meta in enumerate(metas):
        if meta.images.ndim != 4 or meta.images.shape[0] != num_images:
            raise RuntimeError(
                f"sample {meta.sample_index}: bad images shape {meta.images.shape}."
            )
        h, w = int(meta.images.shape[1]), int(meta.images.shape[2])
        sample_mask = mask[offset : offset + meta.n_per_sample]
        sample_abs = abs_col[offset : offset + meta.n_per_sample]

        text_ax = axes[row, 0]
        text_ax.set_axis_off()
        text_ax.text(
            0.0,
            0.5,
            f"s{meta.sample_index:02d}\n{_prompt_for_display(meta.prompt)}",
            transform=text_ax.transAxes,
            va="center",
            ha="left",
            fontsize=8,
            wrap=True,
        )

        for cam in range(num_images):
            ax = axes[row, cam + 1]
            ax.imshow(meta.images[cam])
            ax.set_axis_off()
            cam_locals = [
                i
                for i, lab in enumerate(meta.labels)
                if lab == f"image[{cam}]"
            ]
            cam_out = [i for i in cam_locals if bool(sample_mask[i].item())]
            grid = _image_grid_for_cam(meta, cam)
            assert grid is not None
            gh, gw = grid
            patch_h, patch_w = h // gh, w // gw
            if cam_out:
                vmax = float(sample_abs[cam_out].max().item())
            else:
                vmax = 1.0
            for local_i in cam_out:
                patch_i = cam_locals.index(local_i)
                r, c = divmod(patch_i, gw)
                mag = float(sample_abs[local_i].item())
                # Hotter color = larger |activation|.
                t = mag / max(vmax, 1e-30)
                color = (1.0, 1.0 - 0.7 * t, 0.0)
                ax.add_patch(
                    Rectangle(
                        (c * patch_w, r * patch_h),
                        patch_w,
                        patch_h,
                        fill=False,
                        edgecolor=color,
                        linewidth=1.6,
                    )
                )
            name = cam_names.get(cam, f"cam{cam}")
            ax.set_title(
                f"{name}  out={len(cam_out)}",
                fontsize=9,
            )
        offset += meta.n_per_sample

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    print(f"Wrote {output_path}")


def _plot_channel(
    *,
    col: torch.Tensor,
    labels: list[str],
    mask: torch.Tensor,
    channel: int,
    thr: float,
    output_path: Path,
    title: str,
) -> None:
    abs_v = col.abs().cpu().numpy()
    xs = np.arange(abs_v.shape[0])
    fig, ax = plt.subplots(figsize=(14, 5.2), constrained_layout=True)
    plotted = set()
    for t in TOKEN_TYPES:
        idxs = [i for i, lab in enumerate(labels) if _coarse_type(lab) == t]
        if not idxs:
            continue
        ax.scatter(
            xs[idxs],
            abs_v[idxs],
            s=8,
            c=TOKEN_COLORS[t],
            label=t,
            alpha=0.85,
            linewidths=0,
        )
        plotted.add(t)
    # Highlight outliers with a black edge.
    out_i = torch.nonzero(mask, as_tuple=False).flatten().cpu().numpy()
    n_tok = int(mask.numel())
    n_out = int(out_i.size)
    n_norm = n_tok - n_out
    out_pct = 100.0 * n_out / n_tok if n_tok else 0.0
    norm_pct = 100.0 * n_norm / n_tok if n_tok else 0.0
    if out_i.size:
        ax.scatter(
            xs[out_i],
            abs_v[out_i],
            s=28,
            facecolors="none",
            edgecolors="black",
            linewidths=0.6,
            label="outlier",
        )

    # Reference lines: outlier threshold + channel |x| percentiles over all tokens.
    # (SmoothQuant's smooth_act_percentile is per-forward then max; these global
    # lines are still a useful visual of how aggressive each cut is on this plot.)
    pct_specs = (
        (50.0, "#7f7f7f", ":"),
        (95.0, "#2ca02c", "-"),
        (99.0, "#1f77b4", "-"),
        (99.5, "#9467bd", "-"),
        (99.9, "#ff7f0e", "-"),
    )
    pct_vals = {
        p: float(np.percentile(abs_v, p)) for p, _, _ in pct_specs
    }
    amax = float(abs_v.max()) if abs_v.size else 0.0
    ax.axhline(thr, color="red", linestyle="--", linewidth=1.0, label=f"thr={thr:.3g}")
    for p, color, ls in pct_specs:
        v = pct_vals[p]
        ax.axhline(
            v,
            color=color,
            linestyle=ls,
            linewidth=1.1,
            label=f"p{p:g}={v:.3g}",
        )
    ax.axhline(
        amax,
        color="black",
        linestyle=":",
        linewidth=1.0,
        label=f"max={amax:.3g}",
    )
    ax.set_xlabel("token index (concatenated samples)")
    ax.set_ylabel(f"|activation[:, {channel}]|")
    ax.set_title(
        f"{title}\n"
        f"outlier={n_out}/{n_tok} ({out_pct:.1f}%)  |  "
        f"normal={n_norm}/{n_tok} ({norm_pct:.1f}%)  |  "
        f"p95={pct_vals[95.0]:.3g}  p99={pct_vals[99.0]:.3g}  "
        f"p99.5={pct_vals[99.5]:.3g}  p99.9={pct_vals[99.9]:.3g}  max={amax:.3g}",
        fontsize=10,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncols=3, loc="upper right")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {output_path}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default="pi05",
        help="Model kind: selects adapter, default config, and token layout.",
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--calibration-data", required=True)
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--params-dtype", default="bfloat16")
    p.add_argument(
        "--embodiment-tag",
        default=None,
        dest="embodiment_tag",
        help="GR00T processor embodiment tag (e.g. LIBERO_PANDA). Only with --model groot_n17.",
    )
    p.add_argument(
        "--processor-model-name-or-path",
        default=None,
        dest="processor_model_name_or_path",
        help=(
            "Local Cosmos/VLM path for GR00T tokenizer + image preprocessor "
            f"(default: {DEFAULT_PROCESSOR_MODEL_NAME_OR_PATH}). "
            "Only with --model groot_n17."
        ),
    )
    p.add_argument(
        "--layer-regex",
        default=None,
        help=(
            "Regex over target module names; may match multiple linears. "
            "Default depends on --model "
            f"(pi05: {DEFAULT_LAYER_REGEX['pi05']!r}; "
            f"groot_n17: {DEFAULT_LAYER_REGEX['groot_n17']!r})."
        ),
    )
    p.add_argument(
        "--top-channels",
        type=int,
        default=3,
        help="How many worst channels to inspect per linear.",
    )
    p.add_argument(
        "--top-by",
        choices=("amax", "clip"),
        default="amax",
        help=(
            "Rank channels by 'amax' (channel abs-max, default) or "
            "'clip' (outlier clip %%). 'clip' picks the channels where "
            "tip-clip removes the most tokens."
        ),
    )
    p.add_argument(
        "--outlier-kappa",
        type=float,
        default=0.0,
        help=(
            "κ for tip-clip a_j=min(max, κ·P_β(|tip|)); 0=off (default). "
            "With --outlier-bulk-percentile. Mutually exclusive with "
            "--outlier-std-k>0."
        ),
    )
    p.add_argument(
        "--outlier-bulk-percentile",
        type=float,
        default=0.0,
        help=(
            "Bulk percentile β for κ×P_β; 0=off (default). Required in "
            "(0, 100) when --outlier-kappa>0; must stay 0 with --outlier-std-k."
        ),
    )
    p.add_argument(
        "--outlier-std-k",
        type=float,
        default=0.0,
        help=(
            "k for tip-clip a_j=min(max, mean+k·std); 0=off (default). "
            "Mutually exclusive with --outlier-kappa>0."
        ),
    )
    p.add_argument(
        "--fit-tokens",
        choices=("image", "image_lang_pad", "all"),
        default=DEFAULT_FIT_TOKENS,
        help=(
            "Tip-token scope for outlier clip "
            f"(default {DEFAULT_FIT_TOKENS})."
        ),
    )
    p.add_argument("--max-list", type=int, default=40)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tools/img/act_channel_token_outliers"),
    )
    p.add_argument("-v", "--verbose", action="count", default=1)
    return p


def _adapter_kwargs(args: argparse.Namespace) -> dict:
    kwargs: dict = {
        "checkpoint_path": args.checkpoint,
        "device": args.device,
        "params_dtype": args.params_dtype,
        "calibration_source": "file",
        "calibration_data_path": args.calibration_data,
    }
    if args.model == "groot_n17":
        if args.embodiment_tag is not None:
            kwargs["embodiment_tag"] = args.embodiment_tag
        kwargs["processor_model_name_or_path"] = (
            args.processor_model_name_or_path
            or DEFAULT_PROCESSOR_MODEL_NAME_OR_PATH
        )
    else:
        if args.embodiment_tag is not None:
            raise SystemExit(
                "--embodiment-tag is only valid with --model groot_n17."
            )
        if args.processor_model_name_or_path is not None:
            raise SystemExit(
                "--processor-model-name-or-path is only valid with "
                "--model groot_n17."
            )
    return kwargs


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    if args.num_samples < 1:
        raise SystemExit("--num-samples must be >= 1.")
    if args.layer_regex is None:
        args.layer_regex = DEFAULT_LAYER_REGEX[args.model]
    kappa, bulk_p, std_k = _resolve_outlier_rule(
        kappa=args.outlier_kappa,
        bulk_percentile=args.outlier_bulk_percentile,
        std_k=args.outlier_std_k,
    )
    rule_label = _outlier_rule_label(
        kappa=kappa, bulk_percentile=bulk_p, std_k=std_k
    )
    print(
        f"model={args.model}  outlier rule: tip={args.fit_tokens}, "
        f"{rule_label}, floor=max(rest)"
    )

    adapter = get_adapter(args.model, **_adapter_kwargs(args))
    model = adapter.build_model()
    adapter.warmup_for_calibration(model)
    config = QVLAConfig.for_model_kind(args.model)
    hits = _find_layers(model, config, args.layer_regex)
    layer_idxs = [_down_layer_index(name) for name, _, _ in hits]
    disambiguate = len(layer_idxs) != len(set(layer_idxs))
    print(f"matched {len(hits)} linear(s):")
    for name, scope, _ in hits:
        print(f"  {name}  (scope={scope})")

    activations, metas = _collect_layers(
        adapter,
        model,
        [(name, mod) for name, _, mod in hits],
        num_samples=args.num_samples,
        model_kind=args.model,
    )
    labels = _all_labels(metas)
    type_counts: dict[str, int] = {}
    for lab in labels:
        type_counts[_coarse_type(lab)] = type_counts.get(_coarse_type(lab), 0) + 1
    print(
        f"samples={len(metas)}  n_img={metas[0].n_img}  "
        f"num_images={metas[0].num_images}  "
        f"patches/image={_patches_per_image(metas[0])}"
    )
    print("token type counts:", dict(sorted(type_counts.items())))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for layer_name, _scope, _mod in hits:
        activation = activations[layer_name]
        if len(labels) != activation.shape[0]:
            raise RuntimeError(
                f"{layer_name}: labels={len(labels)} != "
                f"activation tokens={activation.shape[0]}."
            )
        print(f"\nlayer={layer_name}  activation shape={tuple(activation.shape)}")

        layer_dir = _layer_out_dir(
            args.output_dir, layer_name, disambiguate=disambiguate
        )
        layer_dir.mkdir(parents=True, exist_ok=True)

        clip_pct = _channel_outlier_clip_pct(
            activation,
            labels,
            kappa=kappa,
            bulk_percentile=bulk_p,
            std_k=std_k,
            fit_tokens=args.fit_tokens,
        )
        _plot_clip_pct_by_channel(
            clip_pct,
            output_path=layer_dir / "clip_pct_by_channel.png",
            title=(
                f"{layer_name}  outlier clip % by channel  "
                f"(tip={args.fit_tokens}, {rule_label}, floor=max(rest))"
            ),
        )

        if args.top_by == "clip":
            channels = _channel_outliers_by_clip(
                clip_pct, activation, top_k=args.top_channels
            )
        else:
            channels = _channel_outliers(activation, top_k=args.top_channels)
        for ch, ch_amax, ch_med, ratio in channels:
            col = activation[:, ch]
            mask = _print_channel_report(
                channel=ch,
                ch_amax=ch_amax,
                ch_med=ch_med,
                ratio=ratio,
                labels=labels,
                col=col,
                kappa=kappa,
                bulk_percentile=bulk_p,
                std_k=std_k,
                metas=metas,
                max_list=args.max_list,
                fit_tokens=args.fit_tokens,
            )
            _, thr, _, _ = _token_outlier_mask(
                col,
                labels,
                kappa=kappa,
                bulk_percentile=bulk_p,
                std_k=std_k,
                fit_tokens=args.fit_tokens,
            )
            ch_dir = layer_dir / str(ch)
            ch_dir.mkdir(parents=True, exist_ok=True)
            title = (
                f"{layer_name}  ch={ch}  "
                f"|amax|={ch_amax:.3g} ({ratio:.1f}×med)  "
                f"token_out={int(mask.sum())}/{mask.numel()}"
            )
            _plot_channel(
                col=col,
                labels=labels,
                mask=mask,
                channel=ch,
                thr=float(thr),
                output_path=ch_dir / "by_token.png",
                title=title,
            )
            _plot_image_outlier_patches(
                col=col,
                mask=mask,
                metas=metas,
                channel=ch,
                output_path=ch_dir / "image_outliers.png",
                title=title + "  |  image patch boxes (yellow→red = larger |x|)",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
