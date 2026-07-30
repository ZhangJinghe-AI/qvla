#!/usr/bin/env python
"""DiT activation-covariance stability across noise seeds (relF + Spearman).

Fixed calibration inputs; two run seeds. For each denoise step independently:

  * accumulate ``Σ = XᵀX / n`` on GPU only (one matrix per layer),
  * compare seed A vs B (relF / Spearman),
  * keep the scalar metrics, then free the GPU covariances.

Two sweep modes (``--sweep``)::

  k        (default) fixed ``--num-samples``, sweep ``--k-list``
  samples  fixed K (``--k-list`` must be a single value, e.g. 1),
           sweep ``--num-samples-list``

Outputs under ``--output-dir``::

    heatmap_n{N}_k{K}_cov_relF_spearman.png

Examples::

    # Sweep noise-ensemble K (original)
    CUDA_VISIBLE_DEVICES=2 uv run python tools/analyze_dit_cov_stability.py \\
        --checkpoint /data/share/pi05_libero_finetuned_v044 \\
        --calibration-source file \\
        --calibration-data ../calibration_data/libero_goal_16_7.npz \\
        --num-samples 4 \\
        --k-list 1,2,4,8 \\
        --seed-a 0 --seed-b 1 \\
        --output-dir tools/img/dit_cov_stability

    # Single setting: N=10, K=8
    CUDA_VISIBLE_DEVICES=2 uv run python tools/analyze_dit_cov_stability.py \\
        --checkpoint /data/share/pi05_libero_finetuned_v044 \\
        --calibration-source file \\
        --calibration-data ../calibration_data/libero_goal_64_7.npz \\
        --num-samples 10 \\
        --k-list 8 \\
        --seed-a 0 --seed-b 1 \\
        --output-dir tools/img/dit_cov_stability_n10_k8

    # Sweep calibration sample count N with fixed K
    CUDA_VISIBLE_DEVICES=2 uv run python tools/analyze_dit_cov_stability.py \\
        --checkpoint /data/share/pi05_libero_finetuned_v044 \\
        --calibration-source file \\
        --calibration-data ../calibration_data/libero_goal_16_7.npz \\
        --sweep samples \\
        --k-list 1 \\
        --num-samples-list 1,2,4,8,16 \\
        --seed-a 0 --seed-b 1 \\
        --output-dir tools/img/dit_cov_stability_nsweep
"""

from __future__ import annotations

import argparse
import gc
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


def _parse_int_list(text: str, *, name: str) -> list[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals or any(v < 1 for v in vals):
        raise SystemExit(f"Invalid {name} {text!r}; need positive ints.")
    return vals


def _noise_seed(run_seed: int, sample_index: int, noise_index: int) -> int:
    return run_seed * 1_000_003 + sample_index * 1_009 + noise_index


def _upper_tri_vec(mat: torch.Tensor) -> torch.Tensor:
    idx = torch.triu_indices(mat.shape[0], mat.shape[1], device=mat.device)
    return mat[idx[0], idx[1]].contiguous()


def _rel_frobenius(a: torch.Tensor, b: torch.Tensor, *, eps: float = 1e-12) -> float:
    """‖A − B‖_F / ‖A‖_F."""
    a64 = a.to(torch.float64)
    b64 = b.to(torch.float64)
    denom = float(a64.norm().item())
    if denom < eps:
        denom = eps
    return float((a64 - b64).norm().item() / denom)


def _stats_device(targets: list[tuple[str, str, torch.nn.Module]]) -> torch.device:
    return targets[0][2].weight.device


def _free_cov(cov: dict[str, torch.Tensor] | None) -> None:
    if cov is None:
        return
    cov.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _collect_one_forward_cov_step(
    adapter,
    batch: dict,
    targets: list[tuple[str, str, torch.nn.Module]],
    *,
    noise: torch.Tensor,
    target_step: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """One forward → per-layer covariance for ``target_step`` only (GPU)."""
    in_features = {name: int(mod.weight.shape[1]) for name, _, mod in targets}
    store: dict[str, tuple[torch.Tensor, int]] = {}
    current_step: list[int | None] = [None]
    handles = []

    def make_hook(layer_name: str):
        d = in_features[layer_name]

        def hook(_mod, inputs):
            step = current_step[0]
            if step != target_step or not inputs or not torch.is_tensor(inputs[0]):
                return
            flat = inputs[0].reshape(-1, d).detach()
            if flat.shape[0] == 0:
                return
            flat_f = flat.float()
            gram = flat_f.T @ flat_f
            if layer_name not in store:
                store[layer_name] = (
                    torch.zeros(d, d, device=device, dtype=torch.float32),
                    0,
                )
            xtx, n = store[layer_name]
            xtx.add_(gram)
            store[layer_name] = (xtx, n + int(flat.shape[0]))

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

    out: dict[str, torch.Tensor] = {}
    for name, _, _ in targets:
        if name not in store:
            raise RuntimeError(
                f"No activations at step={target_step} for layer {name!r}."
            )
        xtx, n = store[name]
        out[name] = xtx / max(n, 1)
    store.clear()
    return out


def _ensemble_cov_for_step(
    adapter,
    targets: list[tuple[str, str, torch.nn.Module]],
    batches: list[dict],
    *,
    run_seed: int,
    k: int,
    target_step: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Mean covariance at one denoise step over (sample × K), on GPU."""
    out: dict[str, torch.Tensor] = {}
    count = 0
    for sample_index, batch in enumerate(batches):
        for noise_index in range(k):
            seed = _noise_seed(run_seed, sample_index, noise_index)
            noise = dns._make_noise(adapter, seed)
            one = _collect_one_forward_cov_step(
                adapter,
                batch,
                targets,
                noise=noise,
                target_step=target_step,
                device=device,
            )
            count += 1
            if not out:
                out = {name: cov.clone() for name, cov in one.items()}
            else:
                for name, cov in one.items():
                    out[name].add_(cov)
            del one
            print(
                f"  run_seed={run_seed} step={target_step} K={k}: "
                f"sample {sample_index + 1}/{len(batches)} "
                f"noise {noise_index + 1}/{k}",
                flush=True,
            )
    if count == 0:
        raise RuntimeError("No forwards ran.")
    for name in out:
        out[name].div_(count)
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


def _shared_relf_vmax(results: dict) -> float:
    import numpy as np

    finite_vals = [
        float(v)
        for relf_mat, _ in results.values()
        for v in relf_mat.ravel()
        if np.isfinite(v)
    ]
    relf_vmax = max(finite_vals) if finite_vals else 1.0
    return max(0.05, float(np.ceil(relf_vmax * 20.0) / 20.0))


def _plot_heatmap_figure(
    *,
    relf_mat,
    spearman_mat,
    ylabels: list[str],
    path: Path,
    title: str,
    relf_vmax: float,
) -> None:
    """Two panels: relF + Spearman, shared magma cmap; Spearman stays in [-1, 1]."""
    import matplotlib.pyplot as plt
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1, 2, figsize=(16, max(8.0, len(ylabels) * 0.16)), sharey=True
    )

    panels = [
        (axes[0], relf_mat, "relF  ‖Σa−Σb‖_F / ‖Σa‖_F", 0.0, relf_vmax),
        (axes[1], spearman_mat, "Spearman (upper-tri Σ)", -1.0, 1.0),
    ]
    for ax, mat, panel_title, vmin, vmax in panels:
        im = ax.imshow(
            mat,
            aspect="auto",
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_xlabel("Denoise step")
        ax.set_title(panel_title)
        ax.set_xticks(range(mat.shape[1]))
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if np.isnan(val):
                    continue
                dark = (val - vmin) > 0.55 * (vmax - vmin)
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="black" if dark else "white",
                )
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    axes[0].set_ylabel("DiT linear layer")
    y_stride = max(1, len(ylabels) // 18)
    axes[0].set_yticks(range(0, len(ylabels), y_stride))
    axes[0].set_yticklabels(
        [ylabels[i] for i in range(0, len(ylabels), y_stride)], fontsize=7
    )

    fig.suptitle(title, fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def _run_one_setting(
    *,
    adapter,
    targets,
    batches: list[dict],
    layer_axis: list[tuple[str, str]],
    num_steps: int,
    device: torch.device,
    run_seed_a: int,
    run_seed_b: int,
    k: int,
) -> tuple:
    import numpy as np

    relf_mat = np.full((len(layer_axis), num_steps), np.nan)
    spearman_mat = np.full((len(layer_axis), num_steps), np.nan)
    for step in range(num_steps):
        print(f"\n--- denoise step {step}/{num_steps - 1} ---", flush=True)
        cov_a = _ensemble_cov_for_step(
            adapter,
            targets,
            batches,
            run_seed=run_seed_a,
            k=k,
            target_step=step,
            device=device,
        )
        cov_b = _ensemble_cov_for_step(
            adapter,
            targets,
            batches,
            run_seed=run_seed_b,
            k=k,
            target_step=step,
            device=device,
        )
        for i, (name, _) in enumerate(layer_axis):
            sa = cov_a[name]
            sb = cov_b[name]
            relf_mat[i, step] = _rel_frobenius(sa, sb)
            spearman_mat[i, step] = dsa._spearman(
                _upper_tri_vec(sa), _upper_tri_vec(sb)
            )
        _free_cov(cov_a)
        _free_cov(cov_b)
    return relf_mat, spearman_mat


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--calibration-source", choices=("file", "synthetic"), required=True)
    p.add_argument("--calibration-data", type=Path)
    p.add_argument(
        "--sweep",
        choices=("k", "samples"),
        default="k",
        help="Sweep noise-ensemble K (default) or calibration sample count N.",
    )
    p.add_argument("--num-samples", type=int, default=4, help="Used when --sweep=k.")
    p.add_argument(
        "--num-samples-list",
        type=str,
        default="1,2,4,8,16,32,64",
        help="Used when --sweep=samples.",
    )
    p.add_argument("--sample-start", type=int, default=0)
    p.add_argument(
        "--k-list",
        type=str,
        default="1,2,4,8",
        help="K values when --sweep=k; must be a single K when --sweep=samples.",
    )
    p.add_argument("--seed-a", type=int, default=0)
    p.add_argument("--seed-b", type=int, default=1)
    p.add_argument("--layer-regex", default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(argv)

    if args.calibration_source == "file" and args.calibration_data is None:
        p.error("--calibration-data is required when --calibration-source=file.")
    if args.seed_a == args.seed_b:
        p.error("--seed-a and --seed-b must differ.")

    k_list = _parse_int_list(args.k_list, name="--k-list")
    if args.sweep == "samples":
        if len(k_list) != 1:
            p.error("--sweep=samples requires a single K in --k-list (e.g. --k-list 1).")
        n_list = _parse_int_list(args.num_samples_list, name="--num-samples-list")
        fixed_k = k_list[0]
        max_n = max(n_list)
    else:
        n_list = [args.num_samples]
        fixed_k = None
        max_n = args.num_samples
        if args.num_samples < 1:
            p.error("--num-samples must be >= 1.")

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
    device = _stats_device(targets)
    all_batches = list(adapter.iter_calibration_batches(args.sample_start + max_n))
    pool = all_batches[args.sample_start : args.sample_start + max_n]
    if len(pool) < max_n:
        raise SystemExit(
            f"Need {max_n} samples from index {args.sample_start}, "
            f"got {len(pool)}. (Current calibration file may be smaller than "
            f"--num-samples-list max; e.g. libero_goal_16_7.npz has 16.)"
        )

    layer_axis = _layer_labels([n for n, _, _ in targets])
    ylabels = [lab for _, lab in layer_axis]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[object, tuple[object, object]] = {}
    plot_jobs: list[tuple[object, str, str]] = []

    if args.sweep == "k":
        batches = pool
        for k in k_list:
            print(
                f"\n=== K={k}  seed_a={args.seed_a} vs seed_b={args.seed_b} "
                f"N={args.num_samples} (per-step GPU) ===",
                flush=True,
            )
            results[k] = _run_one_setting(
                adapter=adapter,
                targets=targets,
                batches=batches,
                layer_axis=layer_axis,
                num_steps=num_steps,
                device=device,
                run_seed_a=args.seed_a,
                run_seed_b=args.seed_b,
                k=k,
            )
            plot_jobs.append(
                (
                    k,
                    f"heatmap_n{args.num_samples}_k{k}_cov_relF_spearman.png",
                    f"K={k} | samples={args.num_samples}",
                )
            )
    else:
        assert fixed_k is not None
        for n in n_list:
            batches = pool[:n]
            print(
                f"\n=== N={n}  K={fixed_k}  seed_a={args.seed_a} vs seed_b={args.seed_b} "
                f"(per-step GPU) ===",
                flush=True,
            )
            results[n] = _run_one_setting(
                adapter=adapter,
                targets=targets,
                batches=batches,
                layer_axis=layer_axis,
                num_steps=num_steps,
                device=device,
                run_seed_a=args.seed_a,
                run_seed_b=args.seed_b,
                k=fixed_k,
            )
            plot_jobs.append(
                (
                    n,
                    f"heatmap_n{n}_k{fixed_k}_cov_relF_spearman.png",
                    f"K={fixed_k} | samples={n}",
                )
            )

    relf_vmax = _shared_relf_vmax(results)
    print(f"Shared relF color scale: [0, {relf_vmax:.4f}]", flush=True)

    for key, filename, title_extra in plot_jobs:
        relf_mat, spearman_mat = results[key]
        _plot_heatmap_figure(
            relf_mat=relf_mat,
            spearman_mat=spearman_mat,
            ylabels=ylabels,
            path=args.output_dir / filename,
            title=(
                f"DiT cov seed{args.seed_a} vs seed{args.seed_b} | {title_extra}"
            ),
            relf_vmax=relf_vmax,
        )


if __name__ == "__main__":
    main()
