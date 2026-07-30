#!/usr/bin/env python
"""Compare denoise-step aggregation schemes for DiT activation covariance.

For each aggregation scheme separately (GPU-only, one ``Σ`` per layer):

  * run calibration forwards under two noise seeds,
  * online-pool denoise steps into a single covariance per layer,
  * compare seed A vs B (relF / Spearman),
  * free GPU covariances before the next scheme.

Schemes (defaults assume 10 Euler steps)::

    early      mean over steps 0..4
    late       mean over steps 5..9
    very_late  mean over steps 8..9
    weighted   weighted mean over 0..9 with w(s) ∝ (s+1)  (early light, late heavy)

Plot: one figure, two panels (relF | Spearman); each panel has four scheme
lines vs linear layer.

Outputs under ``--output-dir``::

    cov_step_agg_lines.png

Example::

    CUDA_VISIBLE_DEVICES=2 uv run python tools/analyze_dit_cov_step_agg.py \\
        --checkpoint /data/share/pi05_libero_finetuned_v044 \\
        --calibration-source file \\
        --calibration-data ../calibration_data/libero_object_16_7.npz \\
        --num-samples 4 \\
        --noise-ensemble-k 1 \\
        --seed-a 0 --seed-b 1 \\
        --output-dir tools/img/dit_cov_step_agg
"""

from __future__ import annotations

import argparse
import gc
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

_TOOLS = Path(__file__).resolve().parent
_ROOT = _TOOLS.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_TOOLS))

import analyze_dit_cov_stability as covstab  # noqa: E402
import analyze_dit_noise_sensitivity as dns  # noqa: E402
import analyze_dit_step_activations as dsa  # noqa: E402

from qvla.adapters import get_adapter  # noqa: E402
from qvla.config import QVLAConfig  # noqa: E402


@dataclass(frozen=True)
class AggScheme:
    key: str
    label: str
    steps: tuple[int, ...]
    weights: tuple[float, ...]  # same length as steps; normalized at use time

    def weight_map(self) -> dict[int, float]:
        total = sum(self.weights)
        if total <= 0:
            raise ValueError(f"Scheme {self.key!r} has non-positive weight sum.")
        return {s: w / total for s, w in zip(self.steps, self.weights)}


def _parse_step_range(text: str, *, num_steps: int, name: str) -> tuple[int, ...]:
    """Parse ``lo-hi`` or comma list into sorted unique step indices."""
    text = text.strip()
    if "-" in text and "," not in text:
        lo_s, hi_s = text.split("-", 1)
        lo, hi = int(lo_s), int(hi_s)
        steps = tuple(range(lo, hi + 1))
    else:
        steps = tuple(sorted({int(x.strip()) for x in text.split(",") if x.strip()}))
    if not steps:
        raise SystemExit(f"Empty step range for {name}: {text!r}")
    for s in steps:
        if s < 0 or s >= num_steps:
            raise SystemExit(
                f"{name} step {s} out of range for num_steps={num_steps}."
            )
    return steps


def _default_schemes(num_steps: int, args) -> list[AggScheme]:
    early = _parse_step_range(args.early_steps, num_steps=num_steps, name="early")
    late = _parse_step_range(args.late_steps, num_steps=num_steps, name="late")
    very_late = _parse_step_range(
        args.very_late_steps, num_steps=num_steps, name="very_late"
    )
    all_steps = tuple(range(num_steps))
    w = tuple(float(s + 1) for s in all_steps)
    return [
        AggScheme("early", f"early mean [{early[0]}-{early[-1]}]", early, (1.0,) * len(early)),
        AggScheme("late", f"late mean [{late[0]}-{late[-1]}]", late, (1.0,) * len(late)),
        AggScheme(
            "very_late",
            f"very_late mean [{very_late[0]}-{very_late[-1]}]",
            very_late,
            (1.0,) * len(very_late),
        ),
        AggScheme(
            "weighted",
            f"weighted w∝(s+1) [0-{num_steps - 1}]",
            all_steps,
            w,
        ),
    ]


def _free_cov(cov: dict[str, torch.Tensor] | None) -> None:
    if cov is None:
        return
    cov.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _collect_one_forward_scheme(
    adapter,
    batch: dict,
    targets: list[tuple[str, str, torch.nn.Module]],
    *,
    noise: torch.Tensor,
    scheme: AggScheme,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """One forward → per-layer scheme-pooled covariance on GPU.

    Online: when a denoise step finishes, if it belongs to the scheme, fold
    ``w_s * (XᵀX / n)`` into the single per-layer accumulator.
    """
    in_features = {name: int(mod.weight.shape[1]) for name, _, mod in targets}
    weight_map = scheme.weight_map()
    scheme_steps = frozenset(weight_map)

    # Running pooled cov (GPU).
    pooled: dict[str, torch.Tensor] = {
        name: torch.zeros(d, d, device=device, dtype=torch.float32)
        for name, d in ((n, in_features[n]) for n, _, _ in targets)
    }
    # Per-layer buffer for the active denoise step only.
    step_buf: dict[str, tuple[torch.Tensor, int] | None] = {
        name: None for name, _, _ in targets
    }
    active_step: list[int | None] = [None]
    handles = []

    def _finalize_step(step: int) -> None:
        w = weight_map[step]
        for name, _, _ in targets:
            buf = step_buf[name]
            if buf is None:
                raise RuntimeError(
                    f"Scheme {scheme.key!r}: missing activations at step={step} "
                    f"for layer {name!r}."
                )
            xtx, n = buf
            pooled[name].add_(xtx, alpha=w / max(n, 1))
            step_buf[name] = None

    def make_hook(layer_name: str):
        d = in_features[layer_name]

        def hook(_mod, inputs):
            step = active_step[0]
            if step is None or step not in scheme_steps:
                return
            if not inputs or not torch.is_tensor(inputs[0]):
                return
            flat = inputs[0].reshape(-1, d).detach()
            if flat.shape[0] == 0:
                return
            flat_f = flat.float()
            gram = flat_f.T @ flat_f
            buf = step_buf[layer_name]
            if buf is None:
                xtx = torch.zeros(d, d, device=device, dtype=torch.float32)
                step_buf[layer_name] = (xtx, 0)
                buf = step_buf[layer_name]
            xtx, n = buf
            xtx.add_(gram)
            step_buf[layer_name] = (xtx, n + int(flat.shape[0]))

        return hook

    for name, _, mod in targets:
        handles.append(mod.register_forward_pre_hook(make_hook(name)))

    def step_cb(step: int | None) -> None:
        prev = active_step[0]
        # Transition finalize: covers steps 0..N-2 when the next step starts.
        if prev is not None and step != prev and prev in scheme_steps:
            _finalize_step(prev)
        active_step[0] = step

    try:
        dns._forward_with_noise(adapter, batch, noise, step_cb)
        # Last denoise step has no successor callback, so finalize it here.
        last = active_step[0]
        if last is not None and last in scheme_steps:
            _finalize_step(last)
    finally:
        for h in handles:
            h.remove()

    return pooled


def _ensemble_cov_for_scheme(
    adapter,
    targets: list[tuple[str, str, torch.nn.Module]],
    batches: list[dict],
    *,
    run_seed: int,
    k: int,
    scheme: AggScheme,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Mean scheme-pooled covariance over (sample × K), on GPU."""
    out: dict[str, torch.Tensor] = {}
    count = 0
    for sample_index, batch in enumerate(batches):
        for noise_index in range(k):
            seed = covstab._noise_seed(run_seed, sample_index, noise_index)
            noise = dns._make_noise(adapter, seed)
            one = _collect_one_forward_scheme(
                adapter,
                batch,
                targets,
                noise=noise,
                scheme=scheme,
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
                f"  scheme={scheme.key} run_seed={run_seed} K={k}: "
                f"sample {sample_index + 1}/{len(batches)} "
                f"noise {noise_index + 1}/{k}",
                flush=True,
            )
    if count == 0:
        raise RuntimeError("No forwards ran.")
    for name in out:
        out[name].div_(count)
    return out


def _plot_scheme_lines(
    *,
    schemes: list[AggScheme],
    layer_labels: list[str],
    relf: dict[str, list[float]],
    spearman: dict[str, list[float]],
    path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    x = list(range(len(layer_labels)))
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharex=True)

    colors = {
        "early": "#C44E52",
        "late": "#4C72B0",
        "very_late": "#55A868",
        "weighted": "#DD8452",
    }
    markers = {"early": "o", "late": "s", "very_late": "^", "weighted": "D"}

    for ax, metric_name, series in (
        (axes[0], "relF  ‖Σa−Σb‖_F / ‖Σa‖_F  (lower better)", relf),
        (axes[1], "Spearman (upper-tri Σ)  (higher better)", spearman),
    ):
        for scheme in schemes:
            ax.plot(
                x,
                series[scheme.key],
                marker=markers.get(scheme.key, "o"),
                markersize=3.5,
                linewidth=1.2,
                color=colors.get(scheme.key),
                label=scheme.label,
            )
        ax.set_title(metric_name, fontsize=10)
        ax.set_xlabel("DiT linear layer")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    axes[0].set_ylabel("Metric")
    stride = max(1, len(layer_labels) // 16)
    tick_pos = list(range(0, len(layer_labels), stride))
    for ax in axes:
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(
            [layer_labels[i] for i in tick_pos], rotation=60, ha="right", fontsize=7
        )
    axes[1].set_ylim(-0.05, 1.05)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--calibration-source", choices=("file", "synthetic"), required=True)
    p.add_argument("--calibration-data", type=Path)
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--sample-start", type=int, default=0)
    p.add_argument(
        "--noise-ensemble-k",
        type=int,
        default=1,
        help="Noises per sample within each seed (default: 1).",
    )
    p.add_argument("--seed-a", type=int, default=0)
    p.add_argument("--seed-b", type=int, default=1)
    p.add_argument("--early-steps", type=str, default="0-4")
    p.add_argument("--late-steps", type=str, default="5-9")
    p.add_argument("--very-late-steps", type=str, default="8-9")
    p.add_argument("--layer-regex", default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(argv)

    if args.calibration_source == "file" and args.calibration_data is None:
        p.error("--calibration-data is required when --calibration-source=file.")
    if args.seed_a == args.seed_b:
        p.error("--seed-a and --seed-b must differ.")
    if args.noise_ensemble_k < 1:
        p.error("--noise-ensemble-k must be >= 1.")

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
    schemes = _default_schemes(num_steps, args)
    targets = dsa._resolve_dit_targets(model, config, layer_regex=args.layer_regex)
    device = covstab._stats_device(targets)
    all_batches = list(
        adapter.iter_calibration_batches(args.sample_start + args.num_samples)
    )
    batches = all_batches[args.sample_start : args.sample_start + args.num_samples]
    if len(batches) < args.num_samples:
        raise SystemExit(
            f"Need {args.num_samples} samples from index {args.sample_start}, "
            f"got {len(batches)}."
        )

    layer_axis = covstab._layer_labels([n for n, _, _ in targets])
    labels = [lab for _, lab in layer_axis]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    k = args.noise_ensemble_k
    relf: dict[str, list[float]] = {s.key: [] for s in schemes}
    spearman: dict[str, list[float]] = {s.key: [] for s in schemes}

    for scheme in schemes:
        print(
            f"\n=== scheme={scheme.key} ({scheme.label}) | "
            f"seed {args.seed_a} vs {args.seed_b} | K={k} ===",
            flush=True,
        )
        cov_a = _ensemble_cov_for_scheme(
            adapter,
            targets,
            batches,
            run_seed=args.seed_a,
            k=k,
            scheme=scheme,
            device=device,
        )
        cov_b = _ensemble_cov_for_scheme(
            adapter,
            targets,
            batches,
            run_seed=args.seed_b,
            k=k,
            scheme=scheme,
            device=device,
        )
        for name, _label in layer_axis:
            sa = cov_a[name]
            sb = cov_b[name]
            relf[scheme.key].append(covstab._rel_frobenius(sa, sb))
            spearman[scheme.key].append(
                dsa._spearman(covstab._upper_tri_vec(sa), covstab._upper_tri_vec(sb))
            )
        _free_cov(cov_a)
        _free_cov(cov_b)

    _plot_scheme_lines(
        schemes=schemes,
        layer_labels=labels,
        relf=relf,
        spearman=spearman,
        path=args.output_dir / "cov_step_agg_lines.png",
        title=(
            f"DiT cov step-agg | seed{args.seed_a} vs seed{args.seed_b} | "
            f"K={k} | samples={args.num_samples}"
        ),
    )

    print("\nMean over layers:")
    for scheme in schemes:
        print(
            f"  {scheme.key:10s}  relF={sum(relf[scheme.key]) / len(relf[scheme.key]):.4f}"
            f"  spearman={sum(spearman[scheme.key]) / len(spearman[scheme.key]):.4f}"
            f"  ({scheme.label})"
        )


if __name__ == "__main__":
    main()
