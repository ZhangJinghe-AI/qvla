#!/usr/bin/env python
r"""Compare classic mean+3std and robust MAD channel selection on real activations.

The comparison is only for the first, cross-channel stage that decides which
channels may be clipped. The later per-channel clip amount remains the existing
mean+k*std over selected token values.

A second figure per layer plots, for each channel, the percent of tip
tokens that exceed the clip bound: ``100 * |{t : |x_{t,j}| > a_j}| / T``.
Channels are ranked by amax. Vertical lines mark where each stage-1
amax threshold sits on that ranking.

Example:

    CUDA_VISIBLE_DEVICES=4 uv run python \
      tools/visualize_channel_amax_tail_selection.py \
      --model groot_n17 \
      --checkpoint /data/share/GR00T-N1.7-LIBERO/libero_goal \
      --embodiment-tag LIBERO_PANDA \
      --processor-model-name-or-path /data/share/Cosmos-Reason2-2B \
      --calibration-data ../calibration_data/libero_goal_30_7_demo.npz \
      --num-samples 8 \
      --layer-regex \
      'backbone\.qwen3vl_model\.model\.language_model\.layers\.11\.mlp\.down_proj$'
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import analyze_act_channel_token_outliers as shared
from qvla.adapters import get_adapter
from qvla.adapters.groot.config import DEFAULT_PROCESSOR_MODEL_NAME_OR_PATH
from qvla.config import QVLAConfig
from qvla.core.clip import (
    channel_outlier_mean_std_amax,
    select_channels_by_robust_amax,
)


CLASSIC_STD_K = 3.0


def _selection_stats(abs_tip: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return exact build-like hard amax and both cross-channel selectors."""
    if abs_tip.ndim != 2 or min(abs_tip.shape) < 1:
        raise ValueError(
            f"abs_tip must be a non-empty (tokens, channels) tensor, got {tuple(abs_tip.shape)}."
        )
    # The pack collector stores adaptive activation chunks as fp16.
    values = abs_tip.detach().to(torch.float16).to(torch.float32)
    hard_amax = values.amax(dim=0)
    if not bool(torch.isfinite(hard_amax).all().item()):
        raise RuntimeError("Channel amax contains NaN/Inf after fp16 buffering.")

    mean = hard_amax.mean()
    std = hard_amax.std(unbiased=False)
    classic_threshold = mean + CLASSIC_STD_K * std
    if not bool(torch.isfinite(classic_threshold).item()):
        raise RuntimeError("mean+3std channel-selection threshold is not finite.")

    robust_selected, median, mad, robust_threshold = (
        select_channels_by_robust_amax(hard_amax)
    )
    return {
        "amax": hard_amax,
        "classic_selected": hard_amax > classic_threshold,
        "robust_selected": robust_selected,
        "mean": mean,
        "std": std,
        "classic_threshold": classic_threshold,
        "median": median,
        "mad": mad,
        "robust_threshold": robust_threshold,
    }


def _stage1_threshold_rank(ranked_amax: np.ndarray, threshold: float) -> float:
    """Rank coordinate where ``amax`` crosses a stage-1 threshold.

    ``ranked_amax`` must be finite and sorted descending. Channels with
    ``amax > threshold`` are selected. The returned x is on the same axis as
    ``np.arange(len(ranked_amax))``:

    * all selected → ``n`` (right of the last channel)
    * none selected → ``0`` (left of the first channel)
    * otherwise linear interpolation between the last selected rank and the
      first unselected rank
    """
    ranked = np.asarray(ranked_amax, dtype=np.float64)
    if ranked.ndim != 1 or ranked.size < 1:
        raise ValueError(
            f"ranked_amax must be a 1-D non-empty array, got shape={ranked.shape}."
        )
    if not np.isfinite(ranked).all():
        raise RuntimeError("ranked amax contains NaN/Inf.")
    if not np.isfinite(threshold):
        raise RuntimeError(f"stage-1 threshold is not finite: {threshold}.")
    if np.any(np.diff(ranked) > 0.0):
        raise RuntimeError("ranked_amax must be sorted descending.")

    n = int(ranked.size)
    if float(ranked[-1]) > threshold:
        return float(n)
    if float(ranked[0]) <= threshold:
        return 0.0
    after = int(np.argmax(ranked <= threshold))
    if after < 1:
        raise RuntimeError(
            f"internal rank split failed: after={after}, n={n}, threshold={threshold}."
        )
    left = float(ranked[after - 1])
    right = float(ranked[after])
    span = left - right
    if span <= 0.0:
        raise RuntimeError(
            "cannot interpolate stage-1 threshold through a non-positive "
            f"amax span: left={left}, right={right}, threshold={threshold}."
        )
    frac = (left - float(threshold)) / span
    if not np.isfinite(frac) or frac < 0.0 or frac > 1.0:
        raise RuntimeError(
            f"stage-1 threshold interpolation left the unit interval: frac={frac}."
        )
    return float(after - 1) + frac


def _token_clip_fraction(
    values: torch.Tensor, bound: torch.Tensor, *, name: str
) -> torch.Tensor:
    """Fraction of tokens with ``|x| > bound`` on each channel."""
    if bound.shape != (values.shape[1],):
        raise RuntimeError(
            f"{name}: clip bound shape {tuple(bound.shape)} does not match "
            f"C={int(values.shape[1])}."
        )
    if not bool(torch.isfinite(bound).all().item()):
        raise RuntimeError(f"{name}: clip bound is not finite.")
    amax = values.amax(dim=0)
    if bool((bound > amax).any().item()):
        raise RuntimeError(f"{name}: clip bound exceeded hard channel amax.")
    num_tokens = int(values.shape[0])
    counts = (values > bound.unsqueeze(0)).sum(dim=0)
    frac = counts.to(torch.float64) / float(num_tokens)
    if not bool(torch.isfinite(frac).all().item()):
        raise RuntimeError(f"{name}: token clip fraction is not finite.")
    if bool((frac < 0.0).any().item()) or bool((frac > 1.0).any().item()):
        raise RuntimeError(
            f"{name}: token clip fraction left [0, 1]; "
            f"min={float(frac.min().item()):.6g}, max={float(frac.max().item()):.6g}."
        )
    bound_below_amax = bound < amax
    frac_positive = frac > 0.0
    mismatched = bound_below_amax != frac_positive
    if bool(mismatched.any().item()):
        bad = torch.nonzero(mismatched, as_tuple=False).flatten()[:20].tolist()
        raise RuntimeError(
            f"{name}: bound < amax must clip at least one token, and bound "
            f"== amax must clip none; mismatched channels={bad}."
        )
    return frac.to(torch.float32)


def _clip_fraction_stats(abs_tip: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-channel token clip fractions for all-channel and selective MAD clip."""
    stats = _selection_stats(abs_tip)
    values = abs_tip.detach().to(torch.float16).to(torch.float32)
    if int(values.shape[0]) < 2:
        raise RuntimeError(
            "clip-percent visualization needs >= 2 tip tokens for mean+k*std, "
            f"got T={int(values.shape[0])}."
        )
    amax = stats["amax"]
    nonpositive = torch.nonzero(amax <= 0.0, as_tuple=False).flatten()
    if int(nonpositive.numel()) > 0:
        raise RuntimeError(
            "clip percent requires strictly positive channel amax; "
            f"non-positive channels={nonpositive[:20].tolist()}."
        )

    all_clipped = channel_outlier_mean_std_amax(values, std_k=CLASSIC_STD_K)
    selected = stats["robust_selected"]
    selective_clipped = torch.where(selected, all_clipped, amax)
    if not bool(torch.isfinite(selective_clipped).all().item()):
        raise RuntimeError("selective clip produced non-finite channel amax.")

    frac_all = _token_clip_fraction(values, all_clipped, name="all-channel")
    frac_sel = _token_clip_fraction(values, selective_clipped, name="selective")
    leaked = (~selected) & (frac_sel > 0.0)
    if bool(leaked.any().item()):
        raise RuntimeError(
            "unselected channels must have zero clip fraction; "
            f"leaked={int(leaked.sum().item())}."
        )

    stats["clip_frac_all"] = frac_all
    stats["clip_frac_selective"] = frac_sel
    return stats


def _plot(stats: dict[str, torch.Tensor], *, title: str, output: Path) -> None:
    amax = stats["amax"].cpu().numpy()
    classic = stats["classic_selected"].cpu().numpy().astype(bool)
    robust = stats["robust_selected"].cpu().numpy().astype(bool)
    classic_thr = float(stats["classic_threshold"].item())
    robust_thr = float(stats["robust_threshold"].item())

    order = np.argsort(-amax)
    ranked = amax[order]
    rank = np.arange(len(order))
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)

    ax = axes[0]
    ax.plot(rank, ranked, color="#777777", linewidth=1.0, label="channel amax")
    ax.scatter(
        rank[classic[order]],
        ranked[classic[order]],
        s=24,
        marker="x",
        color="#F58518",
        label=f"mean+3std selected ({int(classic.sum())})",
    )
    ax.scatter(
        rank[robust[order]],
        ranked[robust[order]],
        s=30,
        facecolors="none",
        edgecolors="#4C78A8",
        label=f"median+3×1.4826×MAD selected ({int(robust.sum())})",
    )
    ax.axhline(classic_thr, color="#F58518", linestyle="--", linewidth=1.4)
    ax.axhline(robust_thr, color="#4C78A8", linestyle="--", linewidth=1.4)
    ax.set_xlabel("channel rank (largest first)")
    ax.set_ylabel("channel amax")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend()

    ax = axes[1]
    ax.hist(amax, bins=60, color="#B8B8B8", edgecolor="white")
    ax.axvline(
        classic_thr,
        color="#F58518",
        linestyle="--",
        label=f"mean+3std={classic_thr:.4g}",
    )
    ax.axvline(
        robust_thr,
        color="#4C78A8",
        linestyle="--",
        label=f"robust MAD={robust_thr:.4g}",
    )
    ax.set_xlabel("channel amax")
    ax.set_ylabel("number of channels")
    ax.set_title(
        "Channel-amax distribution; high-tail selection uses the upper bound only"
    )
    ax.grid(alpha=0.2)
    ax.legend()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def _plot_clip_percent(
    stats: dict[str, torch.Tensor], *, title: str, output: Path
) -> None:
    amax = stats["amax"].cpu().numpy()
    frac_all = stats["clip_frac_all"].cpu().numpy() * 100.0
    frac_sel = stats["clip_frac_selective"].cpu().numpy() * 100.0
    classic_thr = float(stats["classic_threshold"].item())
    robust_thr = float(stats["robust_threshold"].item())

    order = np.argsort(-amax)
    ranked = amax[order]
    rank = np.arange(len(order))
    classic_x = _stage1_threshold_rank(ranked, classic_thr)
    robust_x = _stage1_threshold_rank(ranked, robust_thr)

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    ax.plot(
        rank,
        frac_all[order],
        color="#F58518",
        linewidth=1.2,
        label="all-channel mean+3std clip %",
    )
    ax.plot(
        rank,
        frac_sel[order],
        color="#4C78A8",
        linewidth=1.2,
        label="selective MAD clip %",
    )
    ax.axvline(
        classic_x,
        color="#F58518",
        linestyle="--",
        linewidth=1.4,
        label=f"stage-1 mean+3std @ rank {classic_x:.1f}",
    )
    ax.axvline(
        robust_x,
        color="#4C78A8",
        linestyle="--",
        linewidth=1.4,
        label=f"stage-1 robust MAD @ rank {robust_x:.1f}",
    )
    ymax = float(np.max(np.concatenate([frac_all, frac_sel])))
    if not np.isfinite(ymax):
        raise RuntimeError("clip-percent y-axis max is not finite.")
    if ymax < 0.0 or ymax > 100.0:
        raise RuntimeError(f"clip-percent y-axis max left [0, 100]: {ymax}.")
    ax.set_xlabel("channel rank (largest amax first)")
    ax.set_ylabel("clipped tokens / channel tokens (%)")
    ax.set_ylim(0.0, 100.0 if ymax == 0.0 else min(100.0, ymax * 1.05))
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


_GROOT_NAME_PREFIX = "backbone.qwen3vl_model.model."


def _safe_layer_name(name: str) -> str:
    if name.startswith(_GROOT_NAME_PREFIX):
        name = name[len(_GROOT_NAME_PREFIX) :]
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
    if not safe:
        raise ValueError(f"Cannot derive output name from layer {name!r}.")
    return safe


def _adapter_kwargs(args: argparse.Namespace) -> dict:
    kwargs = {
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
            raise ValueError("--embodiment-tag is only valid for groot_n17.")
        if args.processor_model_name_or_path is not None:
            raise ValueError(
                "--processor-model-name-or-path is only valid for groot_n17."
            )
    return kwargs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", choices=shared.MODEL_CHOICES, default="pi05")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration-data", required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--params-dtype", default="bfloat16")
    parser.add_argument("--embodiment-tag")
    parser.add_argument("--processor-model-name-or-path")
    parser.add_argument("--layer-regex")
    parser.add_argument(
        "--fit-tokens",
        choices=("image", "image_lang_pad", "all"),
        default="image_lang_pad",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tools/img/channel_amax_tail_selection"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.num_samples < 1:
        raise ValueError("--num-samples must be >= 1.")
    layer_regex = args.layer_regex or shared.DEFAULT_LAYER_REGEX[args.model]

    adapter = get_adapter(args.model, **_adapter_kwargs(args))
    model = adapter.build_model()
    adapter.warmup_for_calibration(model)
    config = QVLAConfig.for_model_kind(args.model)
    hits = shared._find_layers(model, config, layer_regex)
    activations, metas = shared._collect_layers(
        adapter,
        model,
        [(name, module) for name, _scope, module in hits],
        num_samples=args.num_samples,
        model_kind=args.model,
    )
    labels = shared._all_labels(metas)
    tip_keep = shared._tip_token_keep(labels, fit_tokens=args.fit_tokens)
    if not bool(tip_keep.any().item()):
        raise RuntimeError(f"fit_tokens={args.fit_tokens!r} selected no tokens.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for layer_name, _scope, _module in hits:
        activation = activations[layer_name]
        if activation.shape[0] != len(labels):
            raise RuntimeError(
                f"{layer_name}: activation tokens={activation.shape[0]} "
                f"!= labels={len(labels)}."
            )
        stats = _clip_fraction_stats(activation.abs()[tip_keep])
        classic_n = int(stats["classic_selected"].sum().item())
        robust_n = int(stats["robust_selected"].sum().item())
        title = (
            f"{layer_name}\n"
            f"tip={args.fit_tokens}, T={int(tip_keep.sum())}, "
            f"mean+3std={classic_n}, robust MAD={robust_n}"
        )
        stem = _safe_layer_name(layer_name)
        output = args.output_dir / f"{stem}_channel_amax_tail_selection.png"
        clip_output = args.output_dir / f"{stem}_channel_clip_percent.png"
        _plot(stats, title=title, output=output)
        _plot_clip_percent(stats, title=title, output=clip_output)
        print(f"{layer_name}: mean+3std={classic_n}, robust_MAD={robust_n}")
        print(f"Wrote {output}")
        print(f"Wrote {clip_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
