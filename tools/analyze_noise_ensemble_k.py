#!/usr/bin/env python
"""Noise-ensemble K stability for DiT calibration stats used by the builder.

Stats match what pack-build actually consumes:

* ``perm_amax``       — ``static_channel_amax`` path (perm_score=activation*)
* ``act_scale_amax``  — ``per_step_channel_amax`` (DiT per_step act scales)
* ``svd_gptq_cov``    — ``X.T @ X / n`` (SVD activation cov; GPTQ Hessian is
                        the same matrix unnormalized, so seed-similarity matches)

Protocol: fixed samples, two run seeds, each sample uses K noises; aggregate
over (sample x K) with ``max`` for amax stats and ``mean`` for cov, then
compare seed A vs B with Spearman / cosine.

Outputs under ``--output-dir``::

    heatmap_k{K}_{stat}_spearman.png   # layer x step (one per K, per stat)
    layer{idx}_{kind}.png              # per linear: one subplot per stat
    summary.csv

Example::

    CUDA_VISIBLE_DEVICES=2 uv run python tools/analyze_noise_ensemble_k.py \\
        --checkpoint /data/share/pi05_libero_finetuned_v044 \\
        --calibration-source file \\
        --calibration-data ../calibration_data/libero_object_16_7.npz \\
        --num-samples 4 \\
        --k-list 1,2,4,8,16 \\
        --seed-a 0 --seed-b 1 \\
        --output-dir tools/img/noise_ensemble_k
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

_TOOLS = Path(__file__).resolve().parent
_ROOT = _TOOLS.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_TOOLS))

import analyze_dit_noise_sensitivity as dns  # noqa: E402
import analyze_dit_step_activations as dsa  # noqa: E402

from qvla.adapters import get_adapter  # noqa: E402
from qvla.config import QVLAConfig  # noqa: E402

# Names used in plots / CSV. All three are activation-side builder inputs.
STAT_NAMES = ("perm_amax", "act_scale_amax", "svd_gptq_cov")
_AMAX_STATS = frozenset({"perm_amax", "act_scale_amax"})


def _parse_k_list(text: str) -> list[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals or any(k < 1 for k in vals):
        raise SystemExit(f"Invalid --k-list {text!r}; need positive ints.")
    return vals


def _noise_seed(run_seed: int, sample_index: int, noise_index: int) -> int:
    return run_seed * 1_000_003 + sample_index * 1_009 + noise_index


def _upper_tri_vec(mat: torch.Tensor) -> torch.Tensor:
    """Flatten upper triangle (incl. diagonal) of a square matrix."""
    idx = torch.triu_indices(mat.shape[0], mat.shape[1])
    return mat[idx[0], idx[1]].contiguous()


def _collect_one_forward(
    adapter,
    batch: dict,
    targets: list[tuple[str, str, torch.nn.Module]],
    *,
    noise: torch.Tensor,
    num_steps: int,
) -> dict[str, dict[int, dict[str, torch.Tensor]]]:
    """One forward → per (layer, step) builder-aligned channel stats.

    * ``act_scale_amax``: per-step channel amax (same as collector per_step table)
    * ``perm_amax``: static amax broadcast to every step (max over steps in this
      forward) so heatmaps stay layer×step; comparing any step column equals
      comparing the static vector.
    * ``svd_gptq_cov``: per-step ``X.T@X/n`` upper-tri (SVD/GPTQ use the pooled
      matrix; per-step view shows which steps are unstable).
    """
    in_features = {name: int(mod.weight.shape[1]) for name, _, mod in targets}
    # layer -> step -> (sum_sq_mat, max_abs, n_tokens)
    store: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor, int]]] = {
        name: {} for name, _, _ in targets
    }
    current_step: list[int | None] = [None]
    handles = []

    def make_hook(layer_name: str):
        d = in_features[layer_name]

        def hook(_mod, inputs):
            step = current_step[0]
            if step is None or not inputs or not torch.is_tensor(inputs[0]):
                return
            flat = inputs[0].reshape(-1, d).detach().to(device="cpu", dtype=torch.float32)
            if flat.shape[0] == 0:
                return
            step_dict = store[layer_name]
            if step not in step_dict:
                step_dict[step] = (
                    torch.zeros(d, d, dtype=torch.float64),
                    torch.zeros(d, dtype=torch.float32),
                    0,
                )
            xtx, max_abs, n = step_dict[step]
            x64 = flat.to(torch.float64)
            xtx = xtx + x64.T @ x64
            max_abs = torch.maximum(max_abs, flat.abs().amax(dim=0))
            step_dict[step] = (xtx, max_abs, n + flat.shape[0])

        return hook

    for name, _, mod in targets:
        handles.append(mod.register_forward_pre_hook(make_hook(name)))

    def step_cb(step: int | None) -> None:
        current_step[0] = step

    try:
        dns._forward_with_noise(adapter, batch, noise, step_cb)
    finally:
        for h in handles:
            h.remove()

    out: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    for name, step_dict in store.items():
        if not step_dict:
            raise RuntimeError(f"No activations captured for {name}.")
        # static amax = max over steps (builder perm path).
        static_amax = None
        for _step, (_xtx, max_abs, _n) in step_dict.items():
            static_amax = max_abs if static_amax is None else torch.maximum(static_amax, max_abs)
        assert static_amax is not None

        out[name] = {}
        for step in range(num_steps):
            if step not in step_dict:
                raise RuntimeError(f"Missing step {step} for {name}.")
            xtx, max_abs, n = step_dict[step]
            cov = (xtx / max(n, 1)).to(torch.float32)
            out[name][step] = {
                "perm_amax": static_amax.clone(),
                "act_scale_amax": max_abs.clone(),
                "svd_gptq_cov": _upper_tri_vec(cov),
            }
    return out





def _ensemble_stats(
    adapter,
    targets: list[tuple[str, str, torch.nn.Module]],
    batches: list[dict],
    *,
    run_seed: int,
    k: int,
    num_steps: int,
) -> dict[str, dict[int, dict[str, torch.Tensor]]]:
    """Aggregate over (sample x K noises): max for amax, mean for cov."""
    out: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    count = 0
    for sample_index, batch in enumerate(batches):
        for noise_index in range(k):
            seed = _noise_seed(run_seed, sample_index, noise_index)
            noise = dns._make_noise(adapter, seed)
            one = _collect_one_forward(
                adapter, batch, targets, noise=noise, num_steps=num_steps
            )
            count += 1
            for name, step_dict in one.items():
                if name not in out:
                    out[name] = {
                        step: {s: v.clone() for s, v in stats.items()}
                        for step, stats in step_dict.items()
                    }
                    continue
                for step, stats in step_dict.items():
                    for stat_name, vec in stats.items():
                        if stat_name in _AMAX_STATS:
                            torch.maximum(
                                out[name][step][stat_name],
                                vec,
                                out=out[name][step][stat_name],
                            )
                        else:
                            out[name][step][stat_name] += vec
            print(
                f"  run_seed={run_seed} K={k}: "
                f"sample {sample_index + 1}/{len(batches)} "
                f"noise {noise_index + 1}/{k}",
                flush=True,
            )

    if count == 0:
        raise RuntimeError("No forwards ran.")
    for name in out:
        for step in out[name]:
            out[name][step]["svd_gptq_cov"] /= count
    return out


def _layer_labels(names: list[str]) -> list[tuple[str, str]]:
    kind_order = {"qkv_proj": 0, "o_proj": 1, "gate_up_proj": 2, "down_proj": 3, "other": 4}
    ordered = sorted(
        names,
        key=lambda n: (dsa._layer_idx(n), kind_order.get(dsa._layer_kind(n), 9), n),
    )
    labels = []
    for name in ordered:
        idx = dsa._layer_idx(name)
        kind = dsa._layer_kind(name)
        short = kind.replace("_proj", "").replace("gate_up", "gu")
        labels.append((name, f"L{idx:02d}.{short}"))
    return labels


def _plot_heatmap(mat, *, ylabels: list[str], path: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, max(8.0, len(ylabels) * 0.16)))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-1.0, vmax=1.0, interpolation="nearest")
    ax.set_xlabel("Denoise step")
    ax.set_ylabel("DiT linear layer")
    ax.set_title(title)
    ax.set_xticks(range(mat.shape[1]))
    y_stride = max(1, len(ylabels) // 18)
    ax.set_yticks(range(0, len(ylabels), y_stride))
    ax.set_yticklabels([ylabels[i] for i in range(0, len(ylabels), y_stride)], fontsize=7)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isnan(val):
                continue
            ax.text(
                j, i, f"{val:.2f}", ha="center", va="center", fontsize=5,
                color="white" if abs(val) > 0.55 else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def _plot_layer_curves(
    *,
    layer_name: str,
    k_list: list[int],
    curves: dict[str, dict[str, list[float]]],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    n_stats = len(STAT_NAMES)
    fig, axes = plt.subplots(1, n_stats, figsize=(5.2 * n_stats, 4.2), sharey=True)
    if n_stats == 1:
        axes = [axes]

    titles = {
        "perm_amax": "perm_amax\n(static channel amax)",
        "act_scale_amax": "act_scale_amax\n(per-step channel amax)",
        "svd_gptq_cov": "svd_gptq_cov\n(upper-tri of XᵀX/n)",
    }
    for ax, stat_name in zip(axes, STAT_NAMES):
        ax.plot(k_list, curves[stat_name]["spearman"], marker="o", label="Spearman")
        ax.plot(k_list, curves[stat_name]["cosine"], marker="s", label="Cosine")
        ax.set_xlabel("K (noises per sample)")
        ax.set_title(titles[stat_name], fontsize=9)
        ax.set_xticks(k_list)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Similarity (seed A vs seed B)")
    fig.suptitle(layer_name, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def main(argv: list[str] | None = None) -> None:
    import numpy as np

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--calibration-source", choices=("file", "synthetic"), required=True)
    p.add_argument("--calibration-data", type=Path)
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--sample-start", type=int, default=0)
    p.add_argument("--k-list", type=str, default="1,2,4,8,16")
    p.add_argument("--seed-a", type=int, default=0)
    p.add_argument("--seed-b", type=int, default=1)
    p.add_argument("--layer-regex", default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(argv)

    if args.calibration_source == "file" and args.calibration_data is None:
        p.error("--calibration-data is required when --calibration-source=file.")
    if args.seed_a == args.seed_b:
        p.error("--seed-a and --seed-b must differ.")

    k_list = _parse_k_list(args.k_list)
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
    all_batches = list(adapter.iter_calibration_batches(args.sample_start + args.num_samples))
    batches = all_batches[args.sample_start : args.sample_start + args.num_samples]
    if len(batches) < args.num_samples:
        raise SystemExit(
            f"Need {args.num_samples} samples from index {args.sample_start}, "
            f"got {len(batches)}."
        )

    layer_axis = _layer_labels([n for n, _, _ in targets])
    ylabels = [lab for _, lab in layer_axis]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[int, dict[str, dict[str, dict[int, tuple[float, float]]]]] = {}
    csv_rows: list[dict] = []

    for k in k_list:
        print(f"\n=== K={k}  seed_a={args.seed_a} vs seed_b={args.seed_b} ===", flush=True)
        stats_a = _ensemble_stats(
            adapter, targets, batches,
            run_seed=args.seed_a, k=k, num_steps=num_steps,
        )
        stats_b = _ensemble_stats(
            adapter, targets, batches,
            run_seed=args.seed_b, k=k, num_steps=num_steps,
        )

        results[k] = {s: {} for s in STAT_NAMES}
        for name, _label in layer_axis:
            for stat_name in STAT_NAMES:
                results[k][stat_name][name] = {}
                for step in range(num_steps):
                    va = stats_a[name][step][stat_name]
                    vb = stats_b[name][step][stat_name]
                    sp = dsa._spearman(va, vb)
                    cos = dsa._cosine(va, vb)
                    results[k][stat_name][name][step] = (sp, cos)
                    csv_rows.append(
                        {
                            "k": k,
                            "stat": stat_name,
                            "layer": name,
                            "step": step,
                            "spearman": sp,
                            "cosine": cos,
                        }
                    )

        for stat_name in STAT_NAMES:
            mat = np.full((len(layer_axis), num_steps), np.nan)
            for i, (name, _) in enumerate(layer_axis):
                for step in range(num_steps):
                    mat[i, step] = results[k][stat_name][name][step][0]
            _plot_heatmap(
                mat,
                ylabels=ylabels,
                path=args.output_dir / f"heatmap_k{k}_{stat_name}_spearman.png",
                title=(
                    f"Spearman seed{args.seed_a} vs seed{args.seed_b} | "
                    f"K={k} | {stat_name} | samples={args.num_samples}"
                ),
            )

    for name, label in layer_axis:
        curves: dict[str, dict[str, list[float]]] = {
            s: {"spearman": [], "cosine": []} for s in STAT_NAMES
        }
        for k in k_list:
            for stat_name in STAT_NAMES:
                sps = [results[k][stat_name][name][step][0] for step in range(num_steps)]
                cos = [results[k][stat_name][name][step][1] for step in range(num_steps)]
                curves[stat_name]["spearman"].append(sum(sps) / len(sps))
                curves[stat_name]["cosine"].append(sum(cos) / len(cos))
        _plot_layer_curves(
            layer_name=f"{label}  ({name})",
            k_list=k_list,
            curves=curves,
            path=args.output_dir / f"{dsa._layer_file_suffix(name)}.png",
        )

    csv_path = args.output_dir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["k", "stat", "layer", "step", "spearman", "cosine"]
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
