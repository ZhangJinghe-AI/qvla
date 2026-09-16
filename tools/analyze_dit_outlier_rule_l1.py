#!/usr/bin/env python
r"""Measure L1 energy removed by the repo's existing DiT outlier-clip rules.

The matched-L1 experiment fixes the energy fraction α and solves for τ.
This script does the opposite: keep the existing clip *rules*, and report
how much L1 they actually remove on real pi0.5 DiT action-token activations.

Rules (same primitives as quantization / the causal outlier scripts):

* ``per_channel`` — every channel ``min(amax, μ + k·σ)``
* ``selective`` — MAD-select large-amax channels, then per-channel ``μ + k·σ``
  only there; other channels keep hard amax
* ``global`` — one layer-wide ``μ + k·σ`` shared by every channel

One fixed sample and noise, one forward, every DiT linear, every denoise step.

Example:

    CUDA_VISIBLE_DEVICES=5 uv run python \
      tools/analyze_dit_outlier_rule_l1.py \
      --checkpoint /data/share/pi05_libero_finetuned_v044 \
      --calibration-data ../calibration_data/libero_goal_30_7_demo.npz \
      --sample-index 0 --noise-seed 0 --std-k 3 \
      --output-dir tools/img/dit_outlier_rule_l1
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
_TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_TOOLS))

import analyze_dit_matched_clip_action_impact as matched  # noqa: E402
from qvla.adapters import get_adapter  # noqa: E402
from qvla.adapters.pi05.obs import build_pi05_request  # noqa: E402
from qvla.adapters.pi05.step_hook import find_expert_runner, patched_one_step  # noqa: E402
from qvla.config import QVLAConfig  # noqa: E402
from qvla.core.clip import (  # noqa: E402
    channel_outlier_mean_std_amax,
    layer_outlier_mean_std_amax,
    select_channels_by_robust_amax,
    selective_channel_outlier_mean_std_amax,
)

METHODS = ("per_channel", "selective", "global")


@dataclass(frozen=True)
class RuleHit:
    method: str
    layer_name: str
    layer_idx: int
    kind: str
    step: int
    status: str
    removed_l1_frac: float
    token_clip_frac: float
    n_channels_capped: int
    n_selected_channels: int
    n_clipped: int


def _l1_from_bounds(abs_x: torch.Tensor, bounds: torch.Tensor) -> tuple[float, float, int, int]:
    if abs_x.ndim != 2 or bounds.ndim != 1 or int(bounds.numel()) != int(abs_x.shape[1]):
        raise ValueError(
            f"Need |x| (T,C) and bounds (C,), got {tuple(abs_x.shape)} and "
            f"{tuple(bounds.shape)}."
        )
    values = abs_x.detach().to(torch.float64)
    cap = bounds.detach().to(torch.float64).view(1, -1)
    total = float(values.sum().item())
    if total <= 0.0:
        raise RuntimeError("Activation L1 norm is non-positive.")
    clipped = values.clamp(max=cap)
    removed = float((values - clipped).sum().item())
    mask = values > cap
    n_clipped = int(mask.sum().item())
    n_capped = int((bounds.to(torch.float64) < values.amax(dim=0)).sum().item())
    token_frac = float(mask.to(torch.float64).mean().item())
    return removed / total, token_frac, n_capped, n_clipped


def _measure_one(abs_x: torch.Tensor, std_k: float) -> dict[str, dict]:
    hard = abs_x.amax(dim=0)
    out: dict[str, dict] = {}

    per_bounds = channel_outlier_mean_std_amax(abs_x, std_k=std_k)
    l1, tok, n_cap, n_clip = _l1_from_bounds(abs_x, per_bounds)
    out["per_channel"] = {
        "status": "ok",
        "removed_l1_frac": l1,
        "token_clip_frac": tok,
        "n_channels_capped": n_cap,
        "n_selected_channels": int(abs_x.shape[1]),
        "n_clipped": n_clip,
    }

    glob_bounds = layer_outlier_mean_std_amax(abs_x, std_k=std_k)
    l1, tok, n_cap, n_clip = _l1_from_bounds(abs_x, glob_bounds)
    out["global"] = {
        "status": "ok",
        "removed_l1_frac": l1,
        "token_clip_frac": tok,
        "n_channels_capped": n_cap,
        "n_selected_channels": int(abs_x.shape[1]),
        "n_clipped": n_clip,
    }

    try:
        selected, _, _, _ = select_channels_by_robust_amax(hard)
        sel_bounds = selective_channel_outlier_mean_std_amax(abs_x, std_k=std_k)
        l1, tok, n_cap, n_clip = _l1_from_bounds(abs_x, sel_bounds)
        out["selective"] = {
            "status": "ok",
            "removed_l1_frac": l1,
            "token_clip_frac": tok,
            "n_channels_capped": n_cap,
            "n_selected_channels": int(selected.sum().item()),
            "n_clipped": n_clip,
        }
    except RuntimeError as exc:
        message = str(exc)
        if "MAD" not in message and "positive cross-channel" not in message:
            raise
        out["selective"] = {
            "status": "mad_failed",
            "removed_l1_frac": float("nan"),
            "token_clip_frac": float("nan"),
            "n_channels_capped": 0,
            "n_selected_channels": 0,
            "n_clipped": 0,
        }
    return out


def _capture_all(
    adapter,
    request,
    layers: list[tuple[str, torch.nn.Module]],
    *,
    num_steps: int,
    horizon: int,
) -> dict[str, torch.Tensor]:
    current_step: list[int | None] = [None]
    callbacks: list[int | None] = []
    captured: dict[str, dict[int, torch.Tensor]] = {name: {} for name, _ in layers}

    def make_hook(name: str, layer: torch.nn.Module):
        weight = getattr(layer, "weight", None)
        if not torch.is_tensor(weight) or weight.ndim != 2:
            raise RuntimeError(f"{name} must have a 2-D tensor weight.")
        in_features = int(weight.shape[1])

        def hook(_module, inputs):
            step = current_step[0]
            if step is None:
                raise RuntimeError(
                    f"{name} ran outside the denoise loop; choose a per-step "
                    "DiT action linear."
                )
            if len(inputs) != 1 or not torch.is_tensor(inputs[0]):
                raise RuntimeError(f"{name} must receive exactly one tensor input.")
            if step in captured[name]:
                raise RuntimeError(f"{name} ran more than once at step {step}.")
            x = inputs[0]
            if int(x.shape[-1]) != in_features:
                raise RuntimeError(
                    f"{name} input width {x.shape[-1]} != in_features={in_features}."
                )
            flat = x.reshape(-1, in_features)
            if int(flat.shape[0]) != horizon:
                raise RuntimeError(
                    f"{name}: expected {horizon} action tokens, got {flat.shape[0]}."
                )
            snapshot = matched._exact_fp32(flat).clone()
            captured[name][step] = snapshot
            return None

        return hook

    def step_callback(step: int | None) -> None:
        value = None if step is None else int(step)
        current_step[0] = value
        callbacks.append(value)

    handles = [
        layer.register_forward_pre_hook(make_hook(name, layer))
        for name, layer in layers
    ]
    try:
        runner = find_expert_runner(adapter.engine)
        with patched_one_step(runner, step_callback):
            step_callback(None)
            with torch.inference_mode():
                adapter.engine.step(request)
    finally:
        for handle in handles:
            handle.remove()

    expected = [None, *range(num_steps)]
    if callbacks != expected:
        raise RuntimeError(f"Denoise callback order {callbacks} != {expected}.")
    stacked: dict[str, torch.Tensor] = {}
    for name, _layer in layers:
        steps = captured[name]
        if sorted(steps) != list(range(num_steps)):
            raise RuntimeError(
                f"{name} captured steps {sorted(steps)} != {list(range(num_steps))}."
            )
        stacked[name] = torch.stack(
            [steps[step].detach().cpu() for step in range(num_steps)]
        )
        if not bool(torch.isfinite(stacked[name]).all().item()):
            raise RuntimeError(f"Non-finite activation captured for {name}.")
        del steps
    return stacked


def _summarize(hits: list[RuleHit], method: str) -> str:
    rows = [row for row in hits if row.method == method]
    failed = [row for row in rows if row.status != "ok"]
    ok = [row for row in rows if row.status == "ok"]
    if not ok:
        return f"{method}: no successful fits ({len(failed)} failed)"
    vals = np.array([row.removed_l1_frac for row in ok], dtype=np.float64)
    zero = int((vals <= 0.0).sum())
    return (
        f"{method}: n_ok={len(ok)} n_fail={len(failed)}  "
        f"L1% median={100 * float(np.median(vals)):.4g}  "
        f"mean={100 * float(vals.mean()):.4g}  "
        f"p95={100 * float(np.percentile(vals, 95)):.4g}  "
        f"max={100 * float(vals.max()):.4g}  "
        f"zero_clip={zero}/{len(ok)}"
    )


def _plot(hits: list[RuleHit], output: Path, *, std_k: float) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    colors = {
        "per_channel": "#4C78A8",
        "selective": "#F58518",
        "global": "#54A24B",
    }
    boxes = []
    labels = []
    for method in METHODS:
        vals = [
            100.0 * row.removed_l1_frac
            for row in hits
            if row.method == method and row.status == "ok"
        ]
        boxes.append(vals)
        labels.append(method)
    axes[0].boxplot(boxes, showfliers=True)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("removed L1 %")
    axes[0].set_title("All layer × denoise-step")
    axes[0].grid(alpha=0.25, axis="y")

    layers = sorted({row.layer_idx for row in hits})
    xs = np.arange(len(layers))
    for method in METHODS:
        means = []
        for layer_idx in layers:
            vals = [
                row.removed_l1_frac
                for row in hits
                if row.method == method
                and row.status == "ok"
                and row.layer_idx == layer_idx
            ]
            means.append(100.0 * (sum(vals) / len(vals)) if vals else float("nan"))
        axes[1].plot(xs, means, "o-", color=colors[method], label=method)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([f"L{i}" for i in layers], fontsize=8)
    axes[1].set_ylabel("mean removed L1 %")
    axes[1].set_title("Mean over steps and kinds")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
    fig.suptitle(f"Existing outlier-clip rules, std_k={std_k:g}", fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Failed to write {output}.")


def _write_csv(hits: list[RuleHit], output: Path) -> None:
    fieldnames = [
        "method",
        "layer",
        "layer_idx",
        "kind",
        "step",
        "status",
        "removed_l1_frac",
        "token_clip_frac",
        "n_channels_capped",
        "n_selected_channels",
        "n_clipped",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in hits:
            writer.writerow(
                {
                    "method": row.method,
                    "layer": row.layer_name,
                    "layer_idx": row.layer_idx,
                    "kind": row.kind,
                    "step": row.step,
                    "status": row.status,
                    "removed_l1_frac": row.removed_l1_frac,
                    "token_clip_frac": row.token_clip_frac,
                    "n_channels_capped": row.n_channels_capped,
                    "n_selected_channels": row.n_selected_channels,
                    "n_clipped": row.n_clipped,
                }
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-data", type=Path, required=True)
    parser.add_argument("--layer-regex", default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--std-k", type=float, default=3.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--params-dtype", default="bfloat16")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tools/img/dit_outlier_rule_l1"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(args.checkpoint)
    if not args.calibration_data.is_file():
        raise FileNotFoundError(args.calibration_data)
    if args.sample_index < 0:
        raise ValueError("--sample-index must be >= 0.")
    if args.std_k <= 0.0:
        raise ValueError(f"--std-k must be > 0, got {args.std_k}.")

    adapter = get_adapter(
        "pi05",
        checkpoint_path=args.checkpoint,
        calibration_source="file",
        calibration_data_path=args.calibration_data,
        device=args.device,
        params_dtype=args.params_dtype,
    )
    model = adapter.build_model()
    model.eval()
    adapter.warmup_for_calibration(model)
    layers = matched._dit_layers(model, args.layer_regex)
    config = QVLAConfig.pi05_default()
    num_steps = adapter.dit_step_count(config)
    scheduler = adapter.engine.entry.scheduler
    horizon = int(scheduler.cfg.chunk_size)
    width = int(scheduler.cfg.max_action_dim)

    batches = list(adapter.iter_calibration_batches(args.sample_index + 1))
    if len(batches) != args.sample_index + 1:
        raise RuntimeError(
            f"Requested sample {args.sample_index}, calibration yielded "
            f"{len(batches)} samples."
        )
    adapter._ensure_processor()
    request = build_pi05_request(
        adapter._processor,
        batches[args.sample_index],
        state_dim=adapter.cfg.state_dim,
    )
    generator = torch.Generator(device=scheduler.device)
    generator.manual_seed(args.noise_seed)
    noise = torch.randn(
        1,
        horizon,
        width,
        generator=generator,
        device=scheduler.device,
        dtype=scheduler.params_dtype,
    )
    request = replace(request, noise=noise)
    print(
        f"layers={len(layers)}, sample={args.sample_index}, noise={args.noise_seed}, "
        f"steps={num_steps}, horizon={horizon}, std_k={args.std_k}"
    )

    print("capturing activations in one forward...")
    stacked = _capture_all(
        adapter,
        request,
        layers,
        num_steps=num_steps,
        horizon=horizon,
    )
    print(f"captured {len(stacked)} layers")
    hits: list[RuleHit] = []
    for index, (name, _layer) in enumerate(layers, start=1):
        acts = stacked[name]
        if acts.ndim != 3 or int(acts.shape[0]) != num_steps:
            raise RuntimeError(f"{name} activations {tuple(acts.shape)} unexpected.")
        for step in range(num_steps):
            measured = _measure_one(acts[step].abs(), args.std_k)
            for method in METHODS:
                item = measured[method]
                hits.append(
                    RuleHit(
                        method=method,
                        layer_name=name,
                        layer_idx=matched._layer_idx(name),
                        kind=matched._layer_kind(name),
                        step=step,
                        status=str(item["status"]),
                        removed_l1_frac=float(item["removed_l1_frac"]),
                        token_clip_frac=float(item["token_clip_frac"]),
                        n_channels_capped=int(item["n_channels_capped"]),
                        n_selected_channels=int(item["n_selected_channels"]),
                        n_clipped=int(item["n_clipped"]),
                    )
                )
        print(f"[{index}/{len(layers)}] {name}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png = args.output_dir / "outlier_rule_l1.png"
    csv_path = args.output_dir / "outlier_rule_l1.csv"
    _write_csv(hits, csv_path)
    print(f"Wrote {csv_path}")
    _plot(hits, png, std_k=args.std_k)
    print(f"Wrote {png}")

    print("\nAll denoise steps:")
    for method in METHODS:
        print("  " + _summarize(hits, method))
    step4 = [row for row in hits if row.step == 4]
    print("\nDenoise step 4 only:")
    for method in METHODS:
        print("  " + _summarize(step4, method))
    print("\nBy kind (all steps, L1% mean of successful fits):")
    for method in METHODS:
        parts = []
        for kind in matched.KIND_ORDER:
            vals = [
                row.removed_l1_frac
                for row in hits
                if row.method == method and row.kind == kind and row.status == "ok"
            ]
            if vals:
                parts.append(f"{kind}={100 * (sum(vals) / len(vals)):.4g}%")
        print(f"  {method}: " + ", ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
