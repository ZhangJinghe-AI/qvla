#!/usr/bin/env python
"""Plot per-channel Fisher sensitivity vs activation energy for quant layer(s).

Fisher sensitivity comes from the differentiable forward + exact action Jacobian.
Activation *energy* uses the same per-channel metric as the other DiT analysis
tools (default: token-aggregated L2 norm per input channel).

Single layer::

    CUDA_VISIBLE_DEVICES=2 uv run python tools/plot_fisher_sensitivity.py \\
        --checkpoint /data/share/pi05_libero_finetuned_v044 \\
        --calibration-source file \\
        --calibration-data ../calibration_data/libero_object_16_7.npz \\
        --layer-regex 'expert_stack\\.layers\\.0\\.qkv_proj$' \\
        --fisher-num-samples 4 \\
        --output tools/img/fisher/dit_layer00_qkv_fisher.png

All quant target linears::

    CUDA_VISIBLE_DEVICES=2 uv run python tools/plot_fisher_sensitivity.py \\
        --checkpoint /data/share/pi05_libero_finetuned_v044 \\
        --calibration-source file \\
        --calibration-data ../calibration_data/libero_object_16_7.npz \\
        --all-layers \\
        --fisher-num-samples 4 \\
        --output-dir tools/img/fisher
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Literal

import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from qvla.adapters import get_adapter  # noqa: E402
from qvla.build.fisher import FisherCollector  # noqa: E402
from qvla.config import QVLAConfig  # noqa: E402
from qvla.runtime import list_target_modules  # noqa: E402

ActMetric = Literal["max_abs", "l2"]


class _ChannelActAccumulator:
    """Streaming per-channel activation energy (same definition as DiT act tools)."""

    __slots__ = ("metric", "in_features", "n_tokens", "sum_sq", "max_abs")

    def __init__(self, metric: ActMetric, in_features: int) -> None:
        self.metric = metric
        self.in_features = in_features
        self.n_tokens = 0
        self.sum_sq = torch.zeros(in_features, dtype=torch.float64)
        self.max_abs = torch.zeros(in_features, dtype=torch.float32)

    def update(self, x: torch.Tensor) -> None:
        flat = x.reshape(-1, self.in_features).detach().to(device="cpu", dtype=torch.float32)
        if flat.shape[0] == 0:
            return
        self.n_tokens += flat.shape[0]
        self.sum_sq += (flat.double() * flat.double()).sum(dim=0)
        self.max_abs = torch.maximum(self.max_abs, flat.abs().amax(dim=0))

    def channel_scores(self) -> torch.Tensor:
        if self.n_tokens == 0:
            raise RuntimeError("No tokens accumulated for activation energy.")
        if self.metric == "max_abs":
            return self.max_abs
        return self.sum_sq.sqrt().to(torch.float32)


def _resolve_layers(
    model: torch.nn.Module,
    config: QVLAConfig,
    layer_regex: str | None,
) -> list[tuple[str, str, torch.nn.Module]]:
    targets = list_target_modules(model, config)
    if layer_regex:
        pattern = re.compile(layer_regex)
        targets = [(n, s, m) for n, s, m in targets if pattern.search(n)]
    if not targets:
        hint = f" matching {layer_regex!r}" if layer_regex else ""
        raise SystemExit(f"No quant target layers found{hint}.")
    if layer_regex and len(targets) > 1:
        names = [n for n, _, _ in targets]
        raise SystemExit(
            f"--layer-regex matched {len(targets)} layers; use --all-layers or narrow the pattern.\n"
            f"Matches: {names[:10]}{'...' if len(names) > 10 else ''}"
        )
    return sorted(targets, key=lambda t: (t[1], t[0]))


def _layer_file_stem(name: str, scope: str) -> str:
    m = re.search(r"layers\.(\d+)\.", name)
    if m:
        idx = int(m.group(1))
        for kind in ("qkv_proj", "o_proj", "gate_up_proj", "down_proj"):
            if name.endswith(kind):
                return f"{scope}_layer{idx:02d}_{kind}"
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", name)
    return f"{scope}_{safe}"


def _resolve_layer(
    model: torch.nn.Module,
    config: QVLAConfig,
    layer_regex: str,
) -> tuple[str, str, torch.nn.Module]:
    return _resolve_layers(model, config, layer_regex)[0]


def _pick_per_step(
    per_step: dict[int | None, torch.Tensor],
    *,
    step: int | None,
    step_aggregation: str,
    label: str,
) -> tuple[torch.Tensor, str]:
    if step is not None:
        if step not in per_step:
            available = sorted(k for k in per_step if k is not None)
            raise SystemExit(
                f"Step {step} not found; available DiT steps: {available[:20]}"
                f"{'...' if len(available) > 20 else ''}"
            )
        return per_step[step].detach().cpu(), f"{label} @ step {step}"

    if not per_step:
        raise SystemExit(f"No {label} collected for this layer.")

    if step_aggregation == "uniform":
        stacked = torch.stack(list(per_step.values()), dim=0)
        agg = "uniform mean over steps" if len(per_step) > 1 else "single step"
        return stacked.mean(dim=0).detach().cpu(), f"{label} ({agg})"

    raise ValueError(f"Unknown step aggregation: {step_aggregation!r}")


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.corrcoef(torch.stack([a.float(), b.float()]))[0, 1].item())


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra = a.argsort().argsort().to(torch.float64)
    rb = b.argsort().argsort().to(torch.float64)
    return _pearson(ra, rb)


def _collect_fisher_and_activation(
    adapter,
    targets: list[tuple[str, str, torch.nn.Module]],
    *,
    num_samples: int,
    num_dit_steps: int,
    act_metric: ActMetric,
) -> tuple[dict[str, dict[int | None, torch.Tensor]], dict[str, dict[int | None, torch.Tensor]]]:
    """Run Fisher pass; also accumulate per-channel activation energy on the same forwards."""
    from qvla.adapters.pi05.step_hook import patched_one_step
    from qvla.build.differentiable_forward import (
        differentiable_inference_context,
        force_eager_runners,
        reset_differentiable_state,
    )
    from qvla.runtime.step_context import reset_step_counters

    engine = adapter.engine
    if engine is None:
        raise RuntimeError("Adapter has no engine — call build_model() first.")
    sched = engine.entry.scheduler
    model = engine.entry.model
    force_eager_runners(sched)

    scopes = {name: scope for name, scope, _ in targets}
    in_features = {name: int(mod.weight.shape[1]) for name, _, mod in targets}
    act_store: dict[str, dict[int | None, _ChannelActAccumulator]] = {
        name: {} for name, _, _ in targets
    }
    current_step: list[int | None] = [None]
    act_handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_act_hook(layer_name: str):
        scope = scopes[layer_name]

        def hook(_mod, inputs):
            if not inputs or not torch.is_tensor(inputs[0]):
                return
            step_key: int | None = current_step[0] if scope == "dit" else None
            if scope == "dit" and step_key is None:
                return
            step_dict = act_store[layer_name]
            if step_key not in step_dict:
                step_dict[step_key] = _ChannelActAccumulator(act_metric, in_features[layer_name])
            step_dict[step_key].update(inputs[0])

        return hook

    for name, _, mod in targets:
        act_handles.append(mod.register_forward_pre_hook(make_act_hook(name)))

    try:
        with differentiable_inference_context(sched):
            with FisherCollector(targets, num_dit_steps) as fc:
                for i, batch in enumerate(adapter.iter_calibration_batches(num_samples)):
                    print(f"Fisher sample {i + 1}/{num_samples}")
                    fc.begin_sample()
                    reset_differentiable_state(sched)
                    reset_step_counters(model)
                    model.zero_grad(set_to_none=True)

                    runner = sched.expert_runner

                    def _step_cb(step, _fc=fc):
                        if step is None:
                            current_step[0] = None
                            return
                        current_step[0] = int(step)
                        _fc.set_current_step(int(step))

                    with patched_one_step(runner, _step_cb):
                        fc.set_current_step(None)
                        actions = adapter.forward_differentiable(batch)
                    fc.compute_jacobian_sensitivity(actions, model=model)

                fisher_results = fc.get_results()
    finally:
        for handle in act_handles:
            handle.remove()

    fisher_map = {name: result.sensitivity_per_step() for name, result in fisher_results.items()}
    act_map = {
        name: {step: acc.channel_scores() for step, acc in step_dict.items()}
        for name, step_dict in act_store.items()
    }
    if not any(step_dict for step_dict in act_map.values()):
        raise RuntimeError("No activation energy captured; check hooks and forward path.")
    return fisher_map, act_map


def _print_stats(
    fisher: torch.Tensor,
    energy: torch.Tensor,
    *,
    layer_name: str,
    label: str,
    act_metric: ActMetric,
) -> None:
    import statistics

    f = fisher.float()
    e = energy.float()
    print(f"\nLayer: {layer_name}")
    print(f"  aggregation: {label}")
    print(f"  channels:    {f.numel()}")
    print(f"  Fisher  min/mean/max: {f.min():.4e} / {f.mean():.4e} / {f.max():.4e}")
    print(
        f"  Energy ({act_metric}) min/mean/max: "
        f"{e.min():.4e} / {e.mean():.4e} / {e.max():.4e}"
    )
    print(f"  Pearson r:   {_pearson(f, e):+.6f}")
    print(f"  Spearman rho:{_spearman(f, e):+.6f}")
    pos = f[f > 0]
    if pos.numel():
        print(f"  Fisher nonzero: {int(pos.numel())}/{f.numel()}")
        print(f"  Fisher median (pos): {statistics.median(pos.tolist()):.4e}")


def _plot_fisher_vs_energy(
    fisher: torch.Tensor,
    energy: torch.Tensor,
    *,
    layer_name: str,
    label: str,
    act_metric: ActMetric,
    output: Path,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    f = fisher.float().numpy()
    e = energy.float().numpy()
    channels = np.arange(f.size)
    pearson = _pearson(fisher, energy)
    spearman = _spearman(fisher, energy)

    fig, (ax_top, ax_mid, ax_scatter) = plt.subplots(
        3,
        1,
        figsize=(12, 9),
        gridspec_kw={"height_ratios": [1, 1, 1.1], "hspace": 0.35},
    )

    ax_top.plot(channels, f, color="#4C72B0", linewidth=0.8, alpha=0.9)
    ax_top.set_ylabel("Fisher sensitivity")
    ax_top.grid(True, alpha=0.25)
    ax_top.set_title(
        f"{layer_name}\n{label}  |  "
        f"Pearson r={pearson:+.6f}, Spearman rho={spearman:+.6f}"
    )

    ax_mid.plot(channels, e, color="#DD8452", linewidth=0.8, alpha=0.9)
    ax_mid.set_ylabel(f"Activation energy ({act_metric})")
    ax_mid.set_xlabel("input channel")
    ax_mid.grid(True, alpha=0.25)

    ax_scatter.scatter(e, f, s=10, alpha=0.45, color="#4C72B0", edgecolors="none")
    slope, intercept = np.polyfit(e, f, 1)
    x_line = np.linspace(e.min(), e.max(), 100)
    ax_scatter.plot(
        x_line,
        slope * x_line + intercept,
        color="#C44E52",
        linewidth=1.5,
        label=f"linear fit (slope={slope:.2e})",
    )
    ax_scatter.set_xlabel(f"Activation energy ({act_metric})")
    ax_scatter.set_ylabel("Fisher sensitivity")
    ax_scatter.set_title("Fisher vs energy (one point per channel)")
    ax_scatter.legend(loc="upper left", fontsize=9)
    ax_scatter.grid(True, alpha=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--calibration-data", type=Path)
    p.add_argument("--calibration-source", choices=("file", "synthetic"), required=True)
    p.add_argument(
        "--layer-regex",
        help="Regex matching exactly one layer (single-layer mode). Optional filter with --all-layers.",
    )
    p.add_argument(
        "--all-layers",
        action="store_true",
        help="Plot every quant target linear (one PNG per layer in --output-dir).",
    )
    p.add_argument("--fisher-num-samples", type=int, default=1)
    p.add_argument(
        "--fisher-step-aggregation",
        choices=("uniform",),
        default="uniform",
        help="How to collapse DiT step-wise Fisher (default: uniform mean).",
    )
    p.add_argument(
        "--step",
        type=int,
        default=None,
        help="DiT denoise step index; omit to use --fisher-step-aggregation.",
    )
    p.add_argument(
        "--act-metric",
        choices=("max_abs", "l2"),
        default="l2",
        help="Per-channel activation energy metric (default: l2).",
    )
    p.add_argument("--output", type=Path, help="PNG path for single-layer mode.")
    p.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for --all-layers (auto-named PNGs per layer).",
    )
    args = p.parse_args(argv)

    if args.calibration_source == "file" and args.calibration_data is None:
        p.error("--calibration-data is required when --calibration-source=file.")
    if args.fisher_num_samples < 1:
        p.error("--fisher-num-samples must be >= 1.")
    if args.all_layers:
        if args.output_dir is None:
            p.error("--output-dir is required with --all-layers.")
    else:
        if args.layer_regex is None:
            p.error("Single-layer mode requires --layer-regex (or pass --all-layers).")
        if args.output is None:
            p.error("Single-layer mode requires --output.")

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

    targets = (
        _resolve_layers(model, config, args.layer_regex)
        if args.all_layers
        else [_resolve_layer(model, config, args.layer_regex)]
    )
    num_dit_steps = adapter.dit_step_count(config)

    print(f"Collecting Fisher + energy for {len(targets)} layer(s)...")
    fisher_map, act_map = _collect_fisher_and_activation(
        adapter,
        targets,
        num_samples=args.fisher_num_samples,
        num_dit_steps=num_dit_steps,
        act_metric=args.act_metric,
    )

    try:
        for name, scope, _module in targets:
            fisher, fisher_label = _pick_per_step(
                fisher_map[name],
                step=args.step,
                step_aggregation=args.fisher_step_aggregation,
                label="Fisher",
            )
            energy, energy_label = _pick_per_step(
                act_map[name],
                step=args.step,
                step_aggregation=args.fisher_step_aggregation,
                label="Energy",
            )
            if fisher.shape != energy.shape:
                print(f"Skipping {name}: shape mismatch Fisher {tuple(fisher.shape)} vs energy {tuple(energy.shape)}")
                continue

            agg_label = (
                fisher_label if fisher_label == energy_label else f"{fisher_label}; {energy_label}"
            )
            _print_stats(fisher, energy, layer_name=name, label=agg_label, act_metric=args.act_metric)

            if args.all_layers:
                assert args.output_dir is not None
                out_path = args.output_dir / f"{_layer_file_stem(name, scope)}_fisher.png"
            else:
                assert args.output is not None
                out_path = args.output

            _plot_fisher_vs_energy(
                fisher,
                energy,
                layer_name=name,
                label=agg_label,
                act_metric=args.act_metric,
                output=out_path,
            )
    except ImportError as e:
        raise SystemExit("Install matplotlib (uv sync --group dev) for plots.") from e
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
