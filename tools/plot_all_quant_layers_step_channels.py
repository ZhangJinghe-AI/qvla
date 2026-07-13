#!/usr/bin/env python
"""Batch per-step channel plots for every DiT (expert_stack) quant linear layer.

One calibration pass collects per-Euler-step activations for all 72 DiT layers;
each layer gets one combined figure (4 panels).

Outputs land flat under ``--output-dir``::

    dit_layer00_qkv_proj_step_activation_analysis.png
    ...  (72 layers × 1 figure)

Example::

    uv run python tools/plot_all_quant_layers_step_channels.py \\
        --checkpoint /data/share/pi05_libero_finetuned_v044 \\
        --output-dir tools/img/dit_all_layers_per_step_channels \\
        --calibration-source file \\
        --calibration-data ../calibration_data/libero_object_16_7.npz \\
        --num-samples 8
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent
_ROOT = _TOOLS.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_TOOLS))

import analyze_dit_step_activations as dsa  # noqa: E402

from qvla.adapters import get_adapter  # noqa: E402
from qvla.config import QVLAConfig  # noqa: E402


def _layer_file_stem(name: str) -> str:
    """``expert_stack.layers.0.qkv_proj`` -> ``dit_layer00_qkv_proj``."""
    idx = dsa._layer_idx(name)
    kind = dsa._layer_kind(name)
    if idx >= 0:
        return f"dit_layer{idx:02d}_{kind}"
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", name)
    return f"dit_{safe}"


def _plot_layer_figures(
    layer_name: str,
    store: dict[str, dict[int, dsa._PerStepAccumulator]],
    out_dir: Path,
    *,
    num_steps: int,
    act_metric: dsa.ActMetric,
    top_k: int = 32,
) -> Path | None:
    """Write one combined PNG: original 3 panels + top-k outlier magnitude."""
    import matplotlib.pyplot as plt

    step_dict = store[layer_name]
    if not step_dict:
        return None

    highlight = (0, num_steps // 2, num_steps - 1)
    vectors = dsa._channel_vectors_by_step(store, layer_name, num_steps)
    n_steps = len(vectors)
    if n_steps < 2:
        return None

    short = dsa._short_name(layer_name)
    stem = _layer_file_stem(layer_name)
    path = out_dir / f"{stem}_step_activation_analysis.png"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2×2 layout; left column enlarged for channel curves + dual heatmaps.
    fig = plt.figure(figsize=(28, 15))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[14, 8],
        height_ratios=[6, 7],
        hspace=0.32,
        wspace=0.24,
    )

    # TL: channels_by_denoise_step — was figsize=(10, 5)
    ax_ch = fig.add_subplot(gs[0, 0])
    cmap = plt.cm.plasma
    for step in highlight:
        if step not in step_dict:
            continue
        vals = step_dict[step].channel_scores().numpy()
        color = cmap(step / max(1, num_steps - 1))
        ax_ch.plot(vals, color=color, alpha=0.9, lw=1.5, label=f"step {step}")
    ax_ch.set_xlabel("Input channel")
    ax_ch.set_ylabel(f"Per-channel {act_metric}")
    ax_ch.set_title(short, fontsize=11)
    ax_ch.legend(fontsize=9, loc="upper right")
    ax_ch.tick_params(labelsize=9)
    ax_ch.grid(True, alpha=0.2)

    steps = list(range(n_steps))

    # TR: top-k outlier magnitude — was figsize=(7, 4)
    ax_topk = fig.add_subplot(gs[0, 1])
    k = min(top_k, vectors[0].numel())
    topk_means = [float(v.topk(k).values.mean().item()) for v in vectors]
    ax_topk.plot(steps, topk_means, marker="o", ms=6, lw=2.0, color="#d62728")
    ax_topk.set_xlabel("Denoise step")
    ax_topk.set_ylabel(f"Mean top-{k} {act_metric}")
    ax_topk.set_title(f"Outlier magnitude (top-{k} mean) — {short}", fontsize=10)
    ax_topk.tick_params(labelsize=9)
    ax_topk.grid(True, alpha=0.25)

    # BL: step_curve_similarity_heatmap — was figsize=(11, 4.5)
    pearson = dsa._step_similarity_matrix(vectors, metric="pearson").numpy()
    spearman = dsa._step_similarity_matrix(vectors, metric="spearman").numpy()
    sub_sim = gs[1, 0].subgridspec(1, 2, wspace=0.2)
    ax_sim_left = None
    ax_sim_right = None
    for col, (mat, title) in enumerate(
        (
            (pearson, "Pearson r (shape + scale)"),
            (spearman, "Spearman rho (rank / peak location)"),
        )
    ):
        ax = fig.add_subplot(sub_sim[col])
        if col == 0:
            ax_sim_left = ax
        elif col == 1:
            ax_sim_right = ax
        im = ax.imshow(
            mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal", interpolation="nearest"
        )
        ax.set_xticks(steps)
        ax.set_yticks(steps)
        ax.set_xlabel("Step")
        ax.set_ylabel("Step")
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=9)
        for i in range(n_steps):
            for j in range(n_steps):
                ax.text(
                    j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=11, color="black"
                )
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)

    # BR: step_curve_corr_vs_step0 — was figsize=(7, 4)
    ax_ref = fig.add_subplot(gs[1, 1])
    ref = vectors[0]
    ax_ref.plot(steps, [dsa._pearson(ref, v) for v in vectors], marker="o", ms=6, lw=2.0, label="Pearson r")
    ax_ref.plot(steps, [dsa._spearman(ref, v) for v in vectors], marker="s", ms=5, lw=1.8, label="Spearman rho")
    ax_ref.plot(steps, [dsa._cosine(ref, v) for v in vectors], marker="^", ms=5, lw=1.8, label="Cosine")
    ax_ref.axhline(1.0, color="#cccccc", lw=0.6)
    ax_ref.set_xlabel("Step")
    ax_ref.set_ylabel("Similarity vs step 0")
    ax_ref.set_ylim(-0.05, 1.05)
    ax_ref.set_title(f"Curve drift vs step 0 — {short}", fontsize=10)
    ax_ref.legend(fontsize=9)
    ax_ref.tick_params(labelsize=9)
    ax_ref.grid(True, alpha=0.25)

    if ax_sim_left is not None and ax_sim_right is not None:
        pos_l = ax_sim_left.get_position()
        pos_r = ax_sim_right.get_position()
        fig.text(
            (pos_l.x0 + pos_r.x1) / 2,
            max(pos_l.y1, pos_r.y1) + 0.015,
            f"Step curve similarity — {short} ({act_metric})",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--calibration-data", type=Path)
    p.add_argument(
        "--calibration-source",
        choices=("file", "synthetic"),
        required=True,
    )
    p.add_argument("--num-samples", type=int, required=True)
    p.add_argument(
        "--layer-regex",
        help="Optional regex filter on expert_stack layer names.",
    )
    p.add_argument(
        "--act-metric",
        choices=("max_abs", "l2"),
        default="l2",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Flat output directory for all PNG figures.",
    )
    args = p.parse_args(argv)

    if args.calibration_source == "file" and args.calibration_data is None:
        p.error("--calibration-data is required when --calibration-source=file.")

    config = QVLAConfig.pi05_default()
    adapter_kwargs: dict = {
        "checkpoint_path": args.checkpoint,
        "calibration_source": args.calibration_source,
    }
    if args.calibration_source == "file":
        adapter_kwargs["calibration_data_path"] = args.calibration_data

    adapter = get_adapter("pi05", **adapter_kwargs)
    model = adapter.build_model()
    adapter.warmup_for_calibration(model)

    num_steps = adapter.dit_step_count(config)
    targets = dsa._resolve_dit_targets(model, config, layer_regex=args.layer_regex)
    print(
        f"Collecting DiT activations for {len(targets)} layer(s), "
        f"{num_steps} step(s), {args.num_samples} sample(s)..."
    )

    store = dsa._collect_per_step_activations(
        adapter,
        model,
        targets,
        num_samples=args.num_samples,
        num_steps=num_steps,
        act_metric=args.act_metric,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_files = 0
    try:
        for i, (name, _, _) in enumerate(targets, start=1):
            path = _plot_layer_figures(
                name,
                store,
                args.output_dir,
                num_steps=num_steps,
                act_metric=args.act_metric,
            )
            if path is not None:
                total_files += 1
            print(f"[{i}/{len(targets)}] {_layer_file_stem(name)}")
    except ImportError as e:
        raise SystemExit("Install matplotlib (uv sync --group dev) for plots.") from e

    print(f"\nDone. Wrote {total_files} figure(s) to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
