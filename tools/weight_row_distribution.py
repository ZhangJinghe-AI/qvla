"""Weight row-norm distribution under DuQuant-style input rotations.

Compares nine sequential transform pipelines on ``W`` (input-side / column ops):

1. Original ``W``
2. Hadamard
3. SVD (``U`` only)
4. zigzag perm → Hadamard
5. SVD → Hadamard
6. Hadamard → zigzag perm → Hadamard
7. SVD → zigzag perm → Hadamard
8. zigzag perm → SVD → Hadamard (QVLA ``svd_hadamard`` default)
9. zigzag perm → Hadamard → Hadamard

Each step is applied left-to-right on the current weight.  SVD blocks are refit
from the current ``W`` layout at that step; Hadamard is the fixed block matrix
``H``; zigzag perm uses ``perm_score`` (default: weight column energy).

Row metrics default to per-input-channel **max absolute value** (see ``metric``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from qvla.build.collector import LayerStats
from qvla.core.rotation import (
    PermScore,
    PipelineRotationBuild,
    SvdSource,
    _standard_hadamard_blocks,
    step_needs_activation_calibration,
    validate_pipeline,
)

ChannelMetric = Literal["max_abs", "l2"]
PipelineStep = Literal["perm", "svd", "hadamard", "random_hadamard"]
VariantName = Literal[
    "original",
    "hadamard",
    "svd",
    "perm_hadamard",
    "svd_hadamard",
    "h_perm_h",
    "svd_perm_h",
    "perm_svd_h",
    "perm_h2",
]

VARIANT_ORDER: tuple[VariantName, ...] = (
    "original",
    "hadamard",
    "svd",
    "perm_hadamard",
    "svd_hadamard",
    "h_perm_h",
    "svd_perm_h",
    "perm_svd_h",
    "perm_h2",
)

VARIANT_PIPELINES: dict[VariantName, tuple[PipelineStep, ...]] = {
    "original": (),
    "hadamard": ("hadamard",),
    "svd": ("svd",),
    "perm_hadamard": ("perm", "hadamard"),
    "svd_hadamard": ("svd", "hadamard"),
    "h_perm_h": ("hadamard", "perm", "hadamard"),
    "svd_perm_h": ("svd", "perm", "hadamard"),
    "perm_svd_h": ("perm", "svd", "hadamard"),
    "perm_h2": ("perm", "hadamard", "hadamard"),
}

VARIANT_WEIGHT_TITLES: dict[VariantName, str] = {
    "original": "Original $W$",
    "hadamard": r"$H^T W$",
    "svd": r"$R_{SVD}^T W$",
    "perm_hadamard": r"$P \rightarrow H$",
    "svd_hadamard": r"$(R_{SVD}H)^T W$",
    "h_perm_h": r"$H \rightarrow P \rightarrow H$",
    "svd_perm_h": r"$R_{SVD} \rightarrow P \rightarrow H$",
    "perm_svd_h": r"$P \rightarrow R_{SVD} \rightarrow H$",
    "perm_h2": r"$P \rightarrow HH$",
}

VARIANT_ACTIVATION_TITLES: dict[VariantName, str] = {
    "original": "Original $X$",
    "hadamard": r"$H^T X$",
    "svd": r"$R_{SVD}^T X$",
    "perm_hadamard": r"$P \rightarrow H$",
    "svd_hadamard": r"$(R_{SVD}H)^T X$",
    "h_perm_h": r"$H \rightarrow P \rightarrow H$",
    "svd_perm_h": r"$R_{SVD} \rightarrow P \rightarrow H$",
    "perm_svd_h": r"$P \rightarrow R_{SVD} \rightarrow H$",
    "perm_h2": r"$P \rightarrow HH$",
}

VARIANT_COLORS: dict[VariantName, str] = {
    "original": "#d62728",
    "hadamard": "#ff7f0e",
    "svd": "#2ca02c",
    "perm_hadamard": "#bcbd22",
    "svd_hadamard": "#1f77b4",
    "h_perm_h": "#9467bd",
    "svd_perm_h": "#8c564b",
    "perm_svd_h": "#e377c2",
    "perm_h2": "#17becf",
}


@dataclass(frozen=True)
class ChannelDistribution:
    """Per-input-channel values plus summary statistics."""

    name: str
    values: torch.Tensor
    max_min_ratio: float
    std: float

    @property
    def num_channels(self) -> int:
        return int(self.values.numel())


def weight_metric_values(weight: torch.Tensor, *, metric: ChannelMetric = "max_abs") -> torch.Tensor:
    """Per-input-channel metric on ``weight`` ``(out, in)``."""
    return channel_metric_values(weight, metric=metric)


def channel_metric_values(
    values: torch.Tensor,
    *,
    metric: ChannelMetric = "max_abs",
) -> torch.Tensor:
    """One scalar per input channel; ``values`` is ``(..., in_features)``."""
    v = values.detach().to(torch.float32)
    if v.ndim == 1:
        raise ValueError(f"Expected at least 2-D (..., in_features), got {tuple(v.shape)}.")
    if metric == "max_abs":
        return v.abs().amax(dim=0)
    if metric == "l2":
        return v.norm(dim=0, p=2)
    raise ValueError(f"Unknown metric {metric!r}; expected max_abs|l2.")


def activation_metric_values(
    activation: torch.Tensor,
    *,
    metric: ChannelMetric = "max_abs",
) -> torch.Tensor:
    """Per-input-channel metric on layer inputs ``(num_tokens, in_features)``."""
    return channel_metric_values(activation, metric=metric)


def distribution_stats(values: torch.Tensor, *, eps: float = 1e-12) -> tuple[float, float]:
    """Return ``(max/min ratio, std)`` for a 1-D tensor."""
    v = values.detach().to(torch.float64)
    vmax = float(v.max().item())
    vmin = max(float(v.min().item()), eps)
    return vmax / vmin, float(v.std(unbiased=False).item())


def summarize_weight_channels(
    weight: torch.Tensor,
    *,
    name: str,
    metric: ChannelMetric = "max_abs",
) -> ChannelDistribution:
    values = weight_metric_values(weight, metric=metric)
    ratio, std = distribution_stats(values)
    return ChannelDistribution(name=name, values=values.cpu(), max_min_ratio=ratio, std=std)


def summarize_activation_channels(
    activation: torch.Tensor,
    *,
    name: str,
    metric: ChannelMetric = "max_abs",
) -> ChannelDistribution:
    values = activation_metric_values(activation, metric=metric)
    ratio, std = distribution_stats(values)
    return ChannelDistribution(name=name, values=values.cpu(), max_min_ratio=ratio, std=std)


def _validate_block(d: int, block_size: int) -> int:
    if block_size <= 0 or (block_size & (block_size - 1)) != 0:
        raise ValueError(f"block_size={block_size} must be a power of two.")
    if d % block_size != 0:
        raise ValueError(f"d={d} not divisible by block_size={block_size}.")
    return d // block_size


def _hadamard_blocks(d: int, block_size: int, *, device) -> torch.Tensor:
    num_blocks = _validate_block(d, block_size)
    return _standard_hadamard_blocks(
        num_blocks, block_size, device=device, dtype=torch.float32
    )


def _apply_block_matmul(weight: torch.Tensor, blocks: torch.Tensor, block_size: int) -> torch.Tensor:
    """``W[:, block] ← W[:, block] @ blocks[i]`` without aliasing bugs."""
    w_out = weight.clone()
    num_blocks = w_out.shape[1] // block_size
    for i in range(num_blocks):
        s = slice(i * block_size, (i + 1) * block_size)
        w_out[:, s] = (w_out[:, s] @ blocks[i].to(w_out.device)).clone()
    return w_out


def _layer_stats_from_tokens(
    x: torch.Tensor,
    step: PipelineStep,
    *,
    in_features: int,
) -> LayerStats:
    """Activation stats at a pipeline step (prefix layout), matching builder calibration."""
    stats = LayerStats(in_features=in_features)
    flat = x.reshape(-1, in_features)
    stats.n_tokens = int(flat.shape[0])
    if stats.n_tokens == 0:
        raise ValueError("activation_tokens must contain at least one token.")
    if step == "perm":
        stats.static_channel_amax = flat.abs().amax(dim=0).to(torch.float32)
    elif step == "svd":
        stats.xtx = (flat.T @ flat).to(torch.float64)
    return stats


def _apply_input_step(
    tensor: torch.Tensor,
    step: PipelineStep,
    *,
    perm: torch.Tensor | None,
    u_blocks: torch.Tensor | None,
    random_hadamard_blocks: torch.Tensor | None,
    block_size: int,
    d: int,
) -> torch.Tensor:
    """Apply one input-side pipeline step along the last axis (``W`` or ``X``)."""
    if step == "perm":
        if perm is None:
            raise ValueError("perm step requires fitted permutation indices.")
        return tensor[:, perm.to(tensor.device)]
    if step == "svd":
        if u_blocks is None:
            raise ValueError("svd step requires fitted U blocks.")
        return _apply_block_matmul(tensor, u_blocks, block_size)
    if step == "hadamard":
        h_blocks = _hadamard_blocks(d, block_size, device=tensor.device)
        return _apply_block_matmul(tensor, h_blocks, block_size)
    if step == "random_hadamard":
        if random_hadamard_blocks is None:
            raise ValueError("random_hadamard step requires fitted random_hadamard blocks.")
        return _apply_block_matmul(tensor, random_hadamard_blocks, block_size)
    raise ValueError(f"Unknown pipeline step {step!r}.")


def _needs_activation_tokens(*, perm_score: PermScore, svd_source: SvdSource) -> bool:
    return perm_score in ("activation", "activation_weight") or svd_source == "activation"


def _apply_pipeline_core(
    weight: torch.Tensor,
    steps: tuple[PipelineStep, ...],
    *,
    activation_tokens: torch.Tensor | None = None,
    block_size: int,
    perm_score: PermScore = "weight",
    svd_source: SvdSource = "weight",
    eps: float = 1e-6,
    build_seed: int = 0,
    layer_name: str | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Apply pipeline to ``W``; optionally mirror the same ops on ``activation_tokens``.

    Activation-driven perm / SVD steps require ``activation_tokens`` in the prefix
    layout at each step (same contract as builder pipeline-wise calibration).
    """
    w = weight.detach().to(torch.float32).clone()
    x_out = (
        activation_tokens.detach().to(torch.float32).clone()
        if activation_tokens is not None
        else None
    )
    d = int(w.shape[1])
    pipeline = validate_pipeline(steps)  # type: ignore[arg-type]

    if _needs_activation_tokens(perm_score=perm_score, svd_source=svd_source):
        if x_out is None:
            raise ValueError(
                "activation_tokens is required when perm_score uses activations "
                f"({perm_score!r}) or svd_source='activation'."
            )
        if x_out.shape[-1] != d:
            raise ValueError(
                f"activation_tokens in_features={x_out.shape[-1]} != weight in_features={d}."
            )

    builder = PipelineRotationBuild(
        d=d,
        block_size=block_size,
        weight=weight,
        pipeline=pipeline,  # type: ignore[arg-type]
        perm_score=perm_score,
        svd_source=svd_source,
        eps=eps,
        layer_name=layer_name,
        build_seed=build_seed,
    )

    for i, step in enumerate(pipeline):
        stats = None
        if step_needs_activation_calibration(
            step, perm_score=perm_score, svd_source=svd_source
        ):
            assert x_out is not None
            stats = _layer_stats_from_tokens(x_out, step, in_features=d)
        builder.fit_step(i, stats=stats)

        step_kwargs = {
            "perm": builder.perm if step == "perm" else None,
            "u_blocks": builder.u_blocks if step == "svd" else None,
            "random_hadamard_blocks": (
                builder.random_hadamard_blocks if step == "random_hadamard" else None
            ),
            "block_size": block_size,
            "d": d,
        }
        w = _apply_input_step(w, step, **step_kwargs)
        if x_out is not None:
            x_out = _apply_input_step(x_out, step, **step_kwargs)

    w = w.contiguous()
    if x_out is None:
        return w
    return w, x_out.contiguous()


def apply_weight_pipeline(
    weight: torch.Tensor,
    steps: tuple[PipelineStep, ...],
    *,
    block_size: int,
    perm_score: PermScore = "weight",
    svd_source: SvdSource = "weight",
    activation_tokens: torch.Tensor | None = None,
    eps: float = 1e-6,
    build_seed: int = 0,
    layer_name: str | None = None,
) -> torch.Tensor:
    """Apply a left-to-right pipeline of column transforms to ``W``.

    Pass ``activation_tokens`` when ``perm_score`` or ``svd_source`` use activations.
    """
    out = _apply_pipeline_core(
        weight,
        steps,
        activation_tokens=activation_tokens,
        block_size=block_size,
        perm_score=perm_score,
        svd_source=svd_source,
        eps=eps,
        build_seed=build_seed,
        layer_name=layer_name,
    )
    if isinstance(out, tuple):
        return out[0]
    return out


def apply_activation_pipeline(
    activation: torch.Tensor,
    weight: torch.Tensor,
    steps: tuple[PipelineStep, ...],
    *,
    block_size: int,
    perm_score: PermScore = "weight",
    svd_source: SvdSource = "weight",
    eps: float = 1e-6,
    build_seed: int = 0,
    layer_name: str | None = None,
) -> torch.Tensor:
    """Apply the same input-side pipeline to layer activations ``X``.

    ``X`` is used both as the transform target and as calibration tokens for
    activation-driven perm / SVD steps at each prefix layout.
    """
    if activation.shape[-1] != weight.shape[1]:
        raise ValueError(
            f"activation in_features={activation.shape[-1]} != weight in_features={weight.shape[1]}."
        )
    flat = activation.reshape(-1, activation.shape[-1])
    _, x_out = _apply_pipeline_core(
        weight,
        steps,
        activation_tokens=flat,
        block_size=block_size,
        perm_score=perm_score,
        svd_source=svd_source,
        eps=eps,
        build_seed=build_seed,
        layer_name=layer_name,
    )
    return x_out.reshape_as(activation)


def compare_weight_variants(
    weight: torch.Tensor,
    *,
    block_size: int = 64,
    metric: ChannelMetric = "max_abs",
    perm_score: PermScore = "weight",
    svd_source: SvdSource = "weight",
    activation_tokens: torch.Tensor | None = None,
    build_seed: int = 0,
    layer_name: str | None = None,
) -> dict[VariantName, ChannelDistribution]:
    """Build nine transformed weights and summarize row metrics for each."""
    if _needs_activation_tokens(perm_score=perm_score, svd_source=svd_source):
        if activation_tokens is None:
            raise ValueError(
                "activation_tokens is required when perm_score uses activations "
                f"({perm_score!r}) or svd_source='activation'."
            )

    w_base = weight.detach().to(torch.float32).clone()
    pipeline_kwargs = {
        "block_size": block_size,
        "perm_score": perm_score,
        "svd_source": svd_source,
        "activation_tokens": activation_tokens,
        "build_seed": build_seed,
        "layer_name": layer_name,
    }

    out: dict[VariantName, ChannelDistribution] = {}
    for key in VARIANT_ORDER:
        steps = VARIANT_PIPELINES[key]
        if not steps:
            w_rot = w_base
        else:
            w_rot = apply_weight_pipeline(w_base, steps, **pipeline_kwargs)
        out[key] = summarize_weight_channels(w_rot, name=VARIANT_WEIGHT_TITLES[key], metric=metric)
    return out


def compare_activation_variants(
    activation: torch.Tensor,
    weight: torch.Tensor,
    *,
    block_size: int = 64,
    metric: ChannelMetric = "max_abs",
    perm_score: PermScore = "weight",
    svd_source: SvdSource = "weight",
    build_seed: int = 0,
    layer_name: str | None = None,
) -> dict[VariantName, ChannelDistribution]:
    """Apply the nine pipelines to layer inputs and summarize per-channel metrics."""
    x_base = activation.detach().to(torch.float32).clone()
    w_base = weight.detach().to(torch.float32).clone()
    pipeline_kwargs = {
        "block_size": block_size,
        "perm_score": perm_score,
        "svd_source": svd_source,
        "build_seed": build_seed,
        "layer_name": layer_name,
    }

    out: dict[VariantName, ChannelDistribution] = {}
    for key in VARIANT_ORDER:
        steps = VARIANT_PIPELINES[key]
        if not steps:
            x_rot = x_base
        else:
            x_rot = apply_activation_pipeline(x_base, w_base, steps, **pipeline_kwargs)
        out[key] = summarize_activation_channels(
            x_rot,
            name=VARIANT_ACTIVATION_TITLES[key],
            metric=metric,
        )
    return out


def plot_weight_distributions(
    distributions: dict[VariantName, ChannelDistribution],
    *,
    output_path: str | None = None,
    highlight_top_k: int = 0,
    figsize: tuple[float, float] = (14.0, 14.0),
    ylabel: str = "Weights Input channel",
    xlabel: str = "Column norm",
):
    """Render a 3×3 grid of horizontal bar charts."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "plot_weight_distributions requires matplotlib (uv sync --group dev)."
        ) from e

    fig, axes = plt.subplots(3, 3, figsize=figsize, sharey=True)
    axes_flat = axes.flatten()
    n_rows = distributions["original"].num_channels
    y = torch.arange(n_rows)

    for ax, key in zip(axes_flat, VARIANT_ORDER):
        dist = distributions[key]
        color = VARIANT_COLORS[key]
        vals = dist.values.numpy()
        if highlight_top_k > 0:
            bar_colors = ["#cccccc"] * n_rows
            top_idx = torch.topk(dist.values, min(highlight_top_k, n_rows)).indices.tolist()
            for idx in top_idx:
                bar_colors[idx] = color
        else:
            bar_colors = [color] * n_rows

        ax.barh(y, vals, color=bar_colors, height=0.85)
        ax.set_title(
            f"{dist.name}\nmax/min = {dist.max_min_ratio:.0f}$\\times$, "
            f"$\\sigma$ = {dist.std:.1f}",
            fontsize=9,
        )
        ax.set_xlabel(xlabel, fontsize=8)
        ax.invert_yaxis()

    axes[0, 0].set_ylabel(ylabel)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_activation_distributions(
    distributions: dict[VariantName, ChannelDistribution],
    *,
    output_path: str | None = None,
    highlight_top_k: int = 0,
    figsize: tuple[float, float] = (14.0, 14.0),
):
    """Render a 3×3 grid for activation input-channel metrics."""
    return plot_weight_distributions(
        distributions,
        output_path=output_path,
        highlight_top_k=highlight_top_k,
        figsize=figsize,
        ylabel="Activation Input channel",
        xlabel="Column norm",
    )


__all__ = [
    "VARIANT_ACTIVATION_TITLES",
    "VARIANT_WEIGHT_TITLES",
    "VARIANT_COLORS",
    "VARIANT_ORDER",
    "VARIANT_PIPELINES",
    "ChannelDistribution",
    "ChannelMetric",
    "PipelineStep",
    "VariantName",
    "activation_metric_values",
    "apply_activation_pipeline",
    "apply_weight_pipeline",
    "channel_metric_values",
    "compare_activation_variants",
    "compare_weight_variants",
    "distribution_stats",
    "plot_activation_distributions",
    "plot_weight_distributions",
    "summarize_activation_channels",
    "summarize_weight_channels",
    "weight_metric_values",
]
