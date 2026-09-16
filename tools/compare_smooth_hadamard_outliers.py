#!/usr/bin/env python
"""Compare SmoothQuant vs block-Hadamard on activation / weight outliers.

Produces two figures per layer under ``--output-dir``:

1. **Summary** (``{stem}.png``) — sorted channel-amax curves and
   outlier metric bars for original / SmoothQuant / Hadamard.
2. **Distributions** (``dist_{stem}.png``) — ``4×3`` grid::

       rows = act-2D | act-3D | weight-2D | weight-3D
       cols = Original | SmoothQuant | Hadamard

   * **2D**: x = per-input-channel max-abs, y = channel index (linear axes).
   * **3D**: surface with z = |value| (act: token×in; weight: out×in; linear axes).

Transforms (SmoothQuant / Hadamard / amax) run on ``--device``; tensors are
copied to host only once when matplotlib draws.

No silent fallbacks: incompatible shapes / missing inputs raise.

Examples::

    uv run python tools/compare_smooth_hadamard_outliers.py \\
        --checkpoint /data/share/pi05_libero_finetuned_v044 \\
        --calibration-data ../calibration_data/libero_goal_30_7_demo.npz \\
        --num-samples 8 \\
        --output-dir tools/img/smooth_vs_hadamard_all
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from qvla.adapters import get_adapter  # noqa: E402
from qvla.core.smooth_fit import fit_smooth_scale  # noqa: E402
from qvla.config import QVLAConfig  # noqa: E402
from qvla.core.pipeline import hadamard_transform  # noqa: E402
from qvla.runtime import list_target_modules  # noqa: E402

logger = logging.getLogger(__name__)

VARIANT_ORDER = ("original", "smoothquant", "hadamard")
VARIANT_COLORS = {
    "original": "#d62728",
    "smoothquant": "#1f77b4",
    "hadamard": "#ff7f0e",
}
VARIANT_LABELS = {
    "original": "Original",
    "smoothquant": "SmoothQuant",
    "hadamard": "Hadamard",
}


@dataclass(frozen=True)
class OutlierMetrics:
    max_min_ratio: float
    max_median_ratio: float
    outlier_frac: float
    log_std: float


@dataclass(frozen=True)
class VariantBundle:
    """Transformed tensors plus per-channel summary stats for one variant.

    Tensors stay on the compute device (typically CUDA); only plotting
    materializes NumPy on CPU once.
    """

    name: str
    weight: torch.Tensor  # (N, K) float32
    activation: torch.Tensor  # (T, K) float32
    act_channel: torch.Tensor  # (K,)
    weight_channel: torch.Tensor  # (K,)
    act_metrics: OutlierMetrics
    weight_metrics: OutlierMetrics


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    """Single device→host copy for matplotlib (no intermediate round-trips)."""
    return t.detach().to(dtype=torch.float32).cpu().numpy()


def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING - 10 * min(verbosity, 2)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _channel_amax(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2-D tensor, got shape {tuple(matrix.shape)}.")
    return matrix.detach().to(torch.float32).abs().amax(dim=0)


def _outlier_metrics(values: torch.Tensor, *, outlier_factor: float) -> OutlierMetrics:
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError(f"Expected non-empty 1-D tensor, got shape {tuple(values.shape)}.")
    if float(outlier_factor) <= 0.0:
        raise ValueError(f"outlier_factor must be > 0, got {outlier_factor}.")
    v = values.detach().to(torch.float64)
    if not torch.isfinite(v).all():
        raise ValueError("Channel metric contains non-finite values.")
    if float(v.min().item()) < 0.0:
        raise ValueError("Channel metric must be non-negative.")
    v_pos = v.clamp_min(1e-30)
    vmax = float(v_pos.max().item())
    vmin = float(v_pos.min().item())
    median = float(v_pos.median().item())
    return OutlierMetrics(
        max_min_ratio=vmax / vmin,
        max_median_ratio=vmax / median,
        outlier_frac=float((v_pos > (outlier_factor * median)).float().mean().item()),
        log_std=float(torch.log10(v_pos).std(unbiased=False).item()),
    )


def _apply_hadamard_pair(
    weight: torch.Tensor,
    activation: torch.Tensor,
    *,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    d = int(weight.shape[1])
    if block_size <= 0 or (block_size & (block_size - 1)) != 0:
        raise ValueError(f"block_size must be a power of two, got {block_size}.")
    if d % block_size != 0:
        raise ValueError(
            f"in_features={d} is not divisible by --block-size={block_size}."
        )
    if activation.shape[-1] != d:
        raise ValueError(
            f"activation in_features={activation.shape[-1]} != weight in_features={d}."
        )
    rot = hadamard_transform(d, block_size, device=weight.device)
    return rot.apply(weight), rot.apply(activation)


def _apply_smooth_pair(
    weight: torch.Tensor,
    activation: torch.Tensor,
    *,
    alpha: float,
    epsilon: float,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    act_amax = _channel_amax(activation)
    s = fit_smooth_scale(
        layer_name=layer_name,
        weight=weight,
        act_channel_amax=act_amax,
        alpha=alpha,
        epsilon=epsilon,
    )
    s_dev = s.to(device=weight.device, dtype=weight.dtype)
    return weight * s_dev.unsqueeze(0), activation / s_dev


def build_variants(
    weight: torch.Tensor,
    activation: torch.Tensor,
    *,
    alpha: float,
    epsilon: float,
    block_size: int,
    outlier_factor: float,
    layer_name: str,
) -> dict[str, VariantBundle]:
    if weight.ndim != 2 or activation.ndim != 2:
        raise ValueError(
            f"weight and activation must be 2-D, got "
            f"weight={tuple(weight.shape)} activation={tuple(activation.shape)}."
        )
    if activation.shape[-1] != weight.shape[1]:
        raise ValueError(
            f"activation in_features={activation.shape[-1]} != "
            f"weight in_features={weight.shape[1]}."
        )
    if activation.shape[0] == 0:
        raise ValueError("activation has zero tokens.")

    if weight.device != activation.device:
        raise ValueError(
            f"weight device {weight.device} != activation device {activation.device}."
        )
    # Keep transforms on the same device as inputs (no CPU↔GPU bounce).
    w0 = weight.detach().to(dtype=torch.float32).contiguous()
    x0 = activation.detach().to(dtype=torch.float32).contiguous()
    w_s, x_s = _apply_smooth_pair(
        w0, x0, alpha=alpha, epsilon=epsilon, layer_name=layer_name
    )
    w_h, x_h = _apply_hadamard_pair(w0, x0, block_size=block_size)

    pairs = {
        "original": (w0, x0),
        "smoothquant": (w_s, x_s),
        "hadamard": (w_h, x_h),
    }
    out: dict[str, VariantBundle] = {}
    for name, (w, x) in pairs.items():
        act_ch = _channel_amax(x)
        w_ch = _channel_amax(w)
        out[name] = VariantBundle(
            name=name,
            weight=w,
            activation=x,
            act_channel=act_ch,
            weight_channel=w_ch,
            act_metrics=_outlier_metrics(act_ch, outlier_factor=outlier_factor),
            weight_metrics=_outlier_metrics(w_ch, outlier_factor=outlier_factor),
        )
    return out


def _sorted_desc(values: torch.Tensor) -> np.ndarray:
    return np.sort(_to_numpy(values).astype(np.float64, copy=False))[::-1]


def plot_summary(
    variants: dict[str, VariantBundle],
    *,
    output_path: Path,
    title: str,
    outlier_factor: float,
) -> None:
    """Sorted channel-amax curves + outlier metric bars (act / weight only)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(title, fontsize=13)

    for ax, attr, ylabel in (
        (axes[0, 0], "act_channel", "Activation channel amax (sorted)"),
        (axes[0, 1], "weight_channel", "Weight column amax (sorted)"),
    ):
        for name in VARIANT_ORDER:
            y = _sorted_desc(getattr(variants[name], attr))
            ax.plot(
                np.arange(1, len(y) + 1),
                y,
                color=VARIANT_COLORS[name],
                label=VARIANT_LABELS[name],
                linewidth=1.6,
            )
        ax.set_xlabel("channel rank (desc)")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="major", alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    metric_keys = ("max_median_ratio", "max_min_ratio", "outlier_frac", "log_std")
    metric_labels = ("max/med", "max/min", "outlier%", "log10-std")
    x_pos = np.arange(len(metric_keys))
    width = 0.22
    n_var = len(VARIANT_ORDER)

    for ax, metrics_attr, subtitle in (
        (axes[1, 0], "act_metrics", f"Activation outliers (>{outlier_factor:g}× med)"),
        (axes[1, 1], "weight_metrics", f"Weight outliers (>{outlier_factor:g}× med)"),
    ):
        for i, name in enumerate(VARIANT_ORDER):
            m: OutlierMetrics = getattr(variants[name], metrics_attr)
            vals = [
                m.max_median_ratio,
                m.max_min_ratio,
                100.0 * m.outlier_frac,
                m.log_std,
            ]
            ax.bar(
                x_pos + (i - (n_var - 1) / 2) * width,
                vals,
                width=width,
                color=VARIANT_COLORS[name],
                label=VARIANT_LABELS[name],
            )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metric_labels)
        ax.set_title(subtitle, fontsize=10)
        ax.grid(True, axis="y", which="major", alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")

    output_path = Path(output_path)
    if output_path.parent.exists() and not output_path.parent.is_dir():
        raise RuntimeError(f"Output parent is not a directory: {output_path.parent}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_channel_amax_2d(ax, channel_amax: torch.Tensor, *, color: str, title: str) -> None:
    """2D: x = magnitude, y = channel index (linear axes)."""
    vals = _to_numpy(channel_amax).astype(np.float64, copy=False)
    channels = np.arange(vals.shape[0], dtype=np.float64)
    ax.plot(vals, channels, color=color, linewidth=0.8)
    ax.set_xlabel("|amax|")
    ax.set_ylabel("channel")
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()


def _plot_matrix_abs_3d(
    ax,
    matrix: torch.Tensor,
    *,
    x_label: str,
    y_label: str,
    title: str,
) -> None:
    """3D surface: z = |matrix[y, x]| (same stride policy as fisher 3D tools)."""
    if matrix.ndim != 2:
        raise ValueError(f"3D plot expects 2-D matrix, got {tuple(matrix.shape)}.")
    n_y, n_x = int(matrix.shape[0]), int(matrix.shape[1])
    # abs on device, then one host copy for matplotlib.
    data = _to_numpy(matrix.abs())
    if not np.isfinite(data).all():
        raise ValueError(f"Non-finite values in 3D matrix for title={title!r}.")
    xx, yy = np.meshgrid(
        np.arange(n_x, dtype=np.float64),
        np.arange(n_y, dtype=np.float64),
    )
    # Match plot_fisher_*_structure_3d.py: ~60 samples/axis, not every cell.
    ax.plot_surface(
        xx,
        yy,
        data,
        cmap="viridis",
        linewidth=0,
        antialiased=False,
        rstride=max(1, n_y // 60),
        cstride=max(1, n_x // 60),
    )
    ax.set_xlabel(x_label, fontsize=7)
    ax.set_ylabel(y_label, fontsize=7)
    ax.set_zlabel("|value|", fontsize=7)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=6)


def plot_distributions_4x3(
    variants: dict[str, VariantBundle],
    *,
    output_path: Path,
    title: str,
) -> None:
    """4×3 grid: act-2D / act-3D / weight-2D / weight-3D × 3 variants."""
    for name in VARIANT_ORDER:
        if name not in variants:
            raise KeyError(f"Missing variant {name!r} in distribution plot.")

    fig = plt.figure(figsize=(14, 16), constrained_layout=True)
    fig.suptitle(title, fontsize=13)

    row_specs = (
        ("act_2d", "Activation 2D (x=|amax|, y=channel)"),
        ("act_3d", "Activation 3D (x=in, y=token, z=|x|)"),
        ("weight_2d", "Weight 2D (x=|col-amax|, y=in-channel)"),
        ("weight_3d", "Weight 3D (x=in, y=out, z=|W|)"),
    )

    for row, (kind, _row_title) in enumerate(row_specs):
        for col, name in enumerate(VARIANT_ORDER):
            vb = variants[name]
            label = VARIANT_LABELS[name]
            idx = row * 3 + col + 1
            if kind.endswith("3d"):
                ax = fig.add_subplot(4, 3, idx, projection="3d")
            else:
                ax = fig.add_subplot(4, 3, idx)

            if kind == "act_2d":
                _plot_channel_amax_2d(
                    ax,
                    vb.act_channel,
                    color=VARIANT_COLORS[name],
                    title=f"{label} · act 2D",
                )
            elif kind == "act_3d":
                _plot_matrix_abs_3d(
                    ax,
                    vb.activation,
                    x_label="in",
                    y_label="token",
                    title=f"{label} · act 3D",
                )
            elif kind == "weight_2d":
                _plot_channel_amax_2d(
                    ax,
                    vb.weight_channel,
                    color=VARIANT_COLORS[name],
                    title=f"{label} · weight 2D",
                )
            elif kind == "weight_3d":
                _plot_matrix_abs_3d(
                    ax,
                    vb.weight,
                    x_label="in",
                    y_label="out",
                    title=f"{label} · weight 3D",
                )
            else:
                raise RuntimeError(f"Unknown row kind {kind!r}.")

    output_path = Path(output_path)
    if output_path.parent.exists() and not output_path.parent.is_dir():
        raise RuntimeError(f"Output parent is not a directory: {output_path.parent}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def print_table(variants: dict[str, VariantBundle]) -> None:
    header = (
        f"{'variant':18s}  "
        f"{'act max/med':>11s}  {'act max/min':>11s}  {'act out%':>8s}  "
        f"{'w max/med':>10s}  {'w max/min':>10s}  {'w out%':>7s}"
    )
    print(header)
    print("-" * len(header))
    for name in VARIANT_ORDER:
        vs = variants[name]
        a, w = vs.act_metrics, vs.weight_metrics
        print(
            f"{VARIANT_LABELS[name]:18s}  "
            f"{a.max_median_ratio:11.2f}  {a.max_min_ratio:11.2f}  "
            f"{100 * a.outlier_frac:7.1f}%  "
            f"{w.max_median_ratio:10.2f}  {w.max_min_ratio:10.2f}  "
            f"{100 * w.outlier_frac:6.1f}%"
        )


def _layer_file_stem(name: str, scope: str | None = None) -> str:
    m = re.search(r"layers\.(\d+)\.", name)
    if m:
        idx = int(m.group(1))
        for kind in ("qkv_proj", "o_proj", "gate_up_proj", "down_proj"):
            if name.endswith(kind):
                prefix = f"{scope}_" if scope else ""
                return f"{prefix}layer{idx:02d}_{kind}"
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_")
    return safe


def _list_target_layers(
    model: torch.nn.Module,
    config: QVLAConfig,
) -> list[tuple[str, str, torch.nn.Module]]:
    targets = list_target_modules(model, config)
    if not targets:
        raise SystemExit("No quant target layers found.")
    return sorted(targets, key=lambda t: (t[1], t[0]))


def _collect_activations(
    adapter,
    model: torch.nn.Module,
    layers: list[tuple[str, str, torch.nn.Module]],
    *,
    num_samples: int,
) -> dict[str, torch.Tensor]:
    if num_samples < 1:
        raise ValueError(f"--num-samples must be >= 1, got {num_samples}.")
    stores: dict[str, list[torch.Tensor]] = {name: [] for name, _, _ in layers}

    def make_hook(layer_name: str):
        def hook(_mod, inputs):
            if not inputs:
                raise RuntimeError(f"Layer {layer_name!r}: empty forward inputs.")
            x = inputs[0]
            if not torch.is_tensor(x):
                raise RuntimeError(
                    f"Layer {layer_name!r}: expected tensor input, got {type(x)}."
                )
            if x.ndim < 2:
                raise RuntimeError(
                    f"Layer {layer_name!r}: expected input ndim>=2, got {tuple(x.shape)}."
                )
            # Stay on the activation device; no GPU↔CPU bounce in the hook.
            stores[layer_name].append(
                x.reshape(-1, x.shape[-1]).detach().contiguous()
            )

        return hook

    handles = [
        module.register_forward_pre_hook(make_hook(name)) for name, _, module in layers
    ]
    try:
        n_seen = 0
        for batch in adapter.iter_calibration_batches(num_samples):
            adapter.forward_for_calibration(model, batch, step_callback=lambda _s: None)
            n_seen += 1
        if n_seen != num_samples:
            raise RuntimeError(
                f"Expected {num_samples} calibration batches, got {n_seen}."
            )
    finally:
        for handle in handles:
            handle.remove()

    out: dict[str, torch.Tensor] = {}
    for name, _, module in layers:
        chunks = stores[name]
        if not chunks:
            raise RuntimeError(f"No activations captured for {name!r}.")
        act = torch.cat(chunks, dim=0).to(torch.float32)
        w = module.weight
        if act.shape[-1] != int(w.shape[1]):
            raise RuntimeError(
                f"Layer {name!r}: activation K={act.shape[-1]} != weight K={w.shape[1]}."
            )
        out[name] = act
    return out


def _process_one_layer(
    *,
    layer_name: str,
    scope: str | None,
    weight: torch.Tensor,
    activation: torch.Tensor,
    alpha: float,
    epsilon: float,
    block_size: int,
    outlier_factor: float,
    summary_path: Path,
    dist_path: Path,
) -> None:
    d = int(weight.shape[1])
    if d % block_size != 0:
        raise ValueError(
            f"Layer {layer_name!r}: in_features={d} not divisible by "
            f"--block-size={block_size}."
        )

    variants = build_variants(
        weight,
        activation,
        alpha=alpha,
        epsilon=epsilon,
        block_size=block_size,
        outlier_factor=outlier_factor,
        layer_name=layer_name,
    )
    title = (
        f"{layer_name}  |  tokens={activation.shape[0]}  "
        f"α={alpha:g}  block={block_size}"
    )
    print(f"\n=== {layer_name} ===")
    print_table(variants)
    plot_summary(
        variants,
        output_path=summary_path,
        title=title,
        outlier_factor=outlier_factor,
    )
    print(f"Wrote {summary_path}")
    plot_distributions_4x3(
        variants,
        output_path=dist_path,
        title=title,
    )
    print(f"Wrote {dist_path}")
    del variants


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True, help="pi0.5 checkpoint path.")
    p.add_argument(
        "--calibration-data",
        required=True,
        help="Calibration .npz path.",
    )
    p.add_argument("--num-samples", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--params-dtype", default="bfloat16")
    p.add_argument("--smooth-alpha", type=float, default=0.5)
    p.add_argument("--smooth-epsilon", type=float, default=1e-5)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument(
        "--outlier-factor",
        type=float,
        default=10.0,
        help="Channel is an outlier if value > factor × median.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for per-layer summary and distribution PNGs.",
    )
    p.add_argument("-v", "--verbose", action="count", default=1)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _setup_logging(args.verbose)

    if args.block_size <= 0 or (args.block_size & (args.block_size - 1)) != 0:
        raise SystemExit(
            f"--block-size must be a power of two, got {args.block_size}."
        )

    adapter = get_adapter(
        "pi05",
        checkpoint_path=args.checkpoint,
        device=args.device,
        params_dtype=args.params_dtype,
        calibration_source="file",
        calibration_data_path=args.calibration_data,
    )
    model = adapter.build_model()
    adapter.warmup_for_calibration(model)
    config = QVLAConfig.pi05_default()
    layers = _list_target_layers(model, config)
    logger.info("Collecting activations for %d layer(s) ...", len(layers))
    acts = _collect_activations(
        adapter, model, layers, num_samples=args.num_samples
    )

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    compute_device = torch.device(args.device)
    for layer_name, scope, module in layers:
        activation = acts.pop(layer_name).to(device=compute_device, dtype=torch.float32)
        weight = module.weight.detach().to(device=compute_device, dtype=torch.float32)
        stem = _layer_file_stem(layer_name, scope)
        summary_path = out_dir / f"{stem}.png"
        dist_path = out_dir / f"{stem}_dist.png"
        _process_one_layer(
            layer_name=layer_name,
            scope=scope,
            weight=weight,
            activation=activation,
            alpha=args.smooth_alpha,
            epsilon=args.smooth_epsilon,
            block_size=args.block_size,
            outlier_factor=args.outlier_factor,
            summary_path=summary_path,
            dist_path=dist_path,
        )
        del weight, activation
        if compute_device.type == "cuda":
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
