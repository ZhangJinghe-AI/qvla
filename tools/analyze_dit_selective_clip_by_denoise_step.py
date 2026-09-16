#!/usr/bin/env python
r"""Clip every DiT linear at one denoise step and compare action impact across steps.

At replay ``t``, all 72 DiT linears are clipped, but only at denoise step ``t``.
Other steps are left unchanged. The clip rule is selective MAD channels plus
per-channel ``μ + k·σ``, with ``k`` matched to a fixed L1 fraction ``α``.

This isolates *which denoise step* carries action sensitivity, not which layer.

Example:

    CUDA_VISIBLE_DEVICES=5 uv run python \
      tools/analyze_dit_selective_clip_by_denoise_step.py \
      --checkpoint /data/share/pi05_libero_finetuned_v044 \
      --calibration-data ../calibration_data/libero_goal_30_7_demo.npz \
      --sample-index 0 --noise-seed 0 --l1-remove 0.005 \
      --output-dir tools/img/dit_selective_clip_by_denoise_step_a005
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
_TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_TOOLS))

import analyze_dit_matched_clip_action_impact as matched  # noqa: E402
import analyze_dit_outlier_action_detail as detail  # noqa: E402
import analyze_dit_selective_matched_l1_action_impact as selective  # noqa: E402
from qvla.config import QVLAConfig  # noqa: E402


@dataclass(frozen=True)
class StepResult:
    step: int
    n_linears_clipped: int
    mean_k: float
    mean_removed_l1_frac: float
    mean_token_clip_frac: float
    metrics: detail.ActionMetrics


def _run_all_layers(
    adapter,
    request,
    layers: list[tuple[str, torch.nn.Module]],
    *,
    num_steps: int,
    horizon: int,
    alpha: float | None,
    target_step: int | None,
    baseline_activations: dict[str, torch.Tensor] | None = None,
    identity_writeback: bool = False,
    n_tokens: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None, dict[str, selective.SelectiveClipStats]]:
    capture = baseline_activations is None
    if capture:
        if alpha is not None or identity_writeback or target_step is not None:
            raise ValueError("Capture must not clip or restrict a denoise step.")
    elif identity_writeback:
        if alpha is not None or target_step is None:
            raise ValueError("Identity write-back needs a target step and no alpha.")
    else:
        if alpha is None or target_step is None:
            raise ValueError("Intervention needs alpha and a target step.")
        if not (0 <= int(target_step) < num_steps):
            raise ValueError(f"target_step must be in [0, {num_steps}), got {target_step}.")

    current_step: list[int | None] = [None]
    callbacks: list[int | None] = []
    captured: dict[str, dict[int, torch.Tensor]] = {name: {} for name, _ in layers}
    stats: dict[str, selective.SelectiveClipStats] = {}
    applied = {name: 0 for name, _ in layers}
    in_features = {}
    for name, layer in layers:
        weight = getattr(layer, "weight", None)
        if not torch.is_tensor(weight) or weight.ndim != 2:
            raise RuntimeError(f"{name} must have a 2-D tensor weight.")
        in_features[name] = int(weight.shape[1])

    def make_hook(name: str, layer: torch.nn.Module):
        width = in_features[name]

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
            if int(x.shape[-1]) != width:
                raise RuntimeError(
                    f"{name} input width {x.shape[-1]} != in_features={width}."
                )
            flat = x.reshape(-1, width)
            token_count = horizon if n_tokens is None else n_tokens
            if int(flat.shape[0]) != token_count:
                raise RuntimeError(
                    f"{name}: expected {token_count} DiT tokens, got {flat.shape[0]}."
                )
            live = matched._exact_fp32(flat).clone()
            if capture:
                captured[name][step] = live
                return None
            assert baseline_activations is not None
            if step < int(target_step):
                expected = baseline_activations[name][step]
                if expected.device != live.device or expected.dtype != live.dtype:
                    expected = expected.to(device=live.device, dtype=live.dtype)
                if not torch.equal(live, expected):
                    raise RuntimeError(
                        f"{name} diverged at step {step} before the clip step "
                        f"{target_step}."
                    )
                captured[name][step] = live
                return None
            captured[name][step] = live
            if step != int(target_step):
                return None
            applied[name] += 1
            if identity_writeback:
                replaced = live.to(device=x.device, dtype=x.dtype)
            else:
                assert alpha is not None
                clipped, clip_stats = selective._clip_selective_matched_k(live, alpha)
                stats[name] = clip_stats
                replaced = clipped.to(device=x.device, dtype=x.dtype)
            return (replaced.reshape_as(x),)

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
        with matched._with_denoise_callback(adapter, step_callback):
            step_callback(None)
            with torch.inference_mode():
                actions = adapter.engine.step(request)
    finally:
        for handle in handles:
            handle.remove()

    expected = [None, *range(num_steps)]
    if callbacks != expected:
        raise RuntimeError(f"Denoise callback order {callbacks} != {expected}.")
    for name, _layer in layers:
        if sorted(captured[name]) != list(range(num_steps)):
            raise RuntimeError(
                f"{name} captured steps {sorted(captured[name])} != "
                f"{list(range(num_steps))}."
            )
        if not capture and applied[name] != 1:
            raise RuntimeError(
                f"{name} clip applied {applied[name]} times, expected 1."
            )
    if not capture and not identity_writeback:
        if sorted(stats) != sorted(name for name, _layer in layers):
            raise RuntimeError("Missing clip stats for some DiT linears.")
    if not torch.is_tensor(actions):
        raise RuntimeError(f"Engine returned {type(actions)}, expected tensor.")
    actions = actions.detach().to(torch.float32).cpu()
    detail._finite(actions, "predicted actions")
    activations = None
    if capture:
        activations = {}
        for name, _layer in layers:
            stacked = torch.stack(
                [captured[name][step].detach().cpu() for step in range(num_steps)]
            )
            if not bool(torch.isfinite(stacked).all().item()):
                raise RuntimeError(f"Non-finite activation captured for {name}.")
            activations[name] = stacked
    return actions, activations, stats


def _plot(results: list[StepResult], output: Path, *, alpha: float) -> None:
    import matplotlib.pyplot as plt

    xs = np.array([row.step for row in results], dtype=np.int64)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(xs, [row.metrics.total_rmse for row in results], "o-", label="total")
    axes[0].plot(xs, [row.metrics.arm_endpoint_rmse for row in results], "s--", label="endpoint")
    axes[0].plot(xs, [row.metrics.arm_local_rmse for row in results], "^-.", label="local")
    axes[0].plot(xs, [row.metrics.gripper_rmse for row in results], "x:", label="gripper")
    axes[0].set_xlabel("denoise step")
    axes[0].set_ylabel("action RMSE vs unclipped")
    axes[0].set_title("Action change vs clip step")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[0].set_xticks(list(xs))

    axes[1].plot(xs, [100.0 * row.mean_removed_l1_frac for row in results], "o-", label="L1 %")
    axes[1].axhline(100.0 * alpha, color="black", linestyle="--", label=f"target {100 * alpha:.2f}%")
    axes[1].plot(xs, [100.0 * row.mean_token_clip_frac for row in results], "s--", label="token %")
    axes[1].set_xlabel("denoise step")
    axes[1].set_ylabel("percent")
    axes[1].set_title("Mean clip intensity over all DiT linears")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
    axes[1].set_xticks(list(xs))
    fig.suptitle(
        f"All DiT linears clipped together, one denoise step at a time, α={alpha:g}",
        fontsize=12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Failed to write {output}.")


def _write_csv(results: list[StepResult], output: Path) -> None:
    metric_names = [field.name for field in fields(detail.ActionMetrics)]
    fieldnames = [
        "step",
        "n_linears_clipped",
        "mean_k",
        "mean_removed_l1_frac",
        "mean_token_clip_frac",
        *metric_names,
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            item = {
                "step": row.step,
                "n_linears_clipped": row.n_linears_clipped,
                "mean_k": row.mean_k,
                "mean_removed_l1_frac": row.mean_removed_l1_frac,
                "mean_token_clip_frac": row.mean_token_clip_frac,
            }
            for name in metric_names:
                item[name] = getattr(row.metrics, name)
            writer.writerow(item)


def _print_step_trend(results: list[StepResult]) -> None:
    steps = np.array([row.step for row in results], dtype=np.float64)
    rmse = np.array([row.metrics.total_rmse for row in results], dtype=np.float64)
    pearson = float(np.corrcoef(steps, rmse)[0, 1])
    spearman = float(
        np.corrcoef(steps.argsort().argsort(), rmse.argsort().argsort())[0, 1]
    )
    print("\nDenoise-step trend of total action RMSE:")
    print(f"  Pearson(step, RMSE)={pearson:.3f}  Spearman={spearman:.3f}")
    early_s, mid_s, late_s = matched._step_tertile_slices(int(len(rmse)))
    print(f"  early steps {results[early_s][0].step}-{results[early_s][-1].step}: "
          f"mean={rmse[early_s].mean():.3e}")
    if mid_s.start < mid_s.stop:
        print(f"  mid steps {results[mid_s][0].step}-{results[mid_s][-1].step}: "
              f"mean={rmse[mid_s].mean():.3e}")
    print(f"  late steps {results[late_s][0].step}-{results[late_s][-1].step}: "
          f"mean={rmse[late_s].mean():.3e}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-data", type=Path, required=True)
    matched.add_model_cli(parser)
    parser.add_argument("--layer-regex", default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--l1-remove", type=float, default=0.005)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--params-dtype", default="bfloat16")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tools/img/dit_selective_clip_by_denoise_step_a005"),
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
    if not (0.0 < args.l1_remove < 1.0):
        raise ValueError(f"--l1-remove must be in (0, 1), got {args.l1_remove}.")

    adapter = matched.adapter_from_args(args)
    model = adapter.build_model()
    model.eval()
    adapter.warmup_for_calibration(model)
    config = QVLAConfig.for_model_kind(args.model)
    layers = matched._dit_layers(model, args.layer_regex, config=config)
    runtime = matched.clip_runtime(adapter, config)
    num_steps = runtime.num_steps
    horizon = runtime.action_horizon
    action_dim = runtime.action_dim

    batches = list(adapter.iter_calibration_batches(args.sample_index + 1))
    if len(batches) != args.sample_index + 1:
        raise RuntimeError(
            f"Requested sample {args.sample_index}, calibration yielded "
            f"{len(batches)} samples."
        )
    request = matched._fixed_noise_request(
        adapter, batches[args.sample_index], runtime, noise_seed=args.noise_seed
    )
    print(
        f"model={args.model}, layers={len(layers)}, sample={args.sample_index}, "
        f"noise={args.noise_seed}, steps={num_steps}, horizon={horizon}, "
        f"n_tokens={runtime.n_tokens}, action_dim={action_dim}, "
        f"l1_remove={args.l1_remove}, clip=all_linears_one_step"
    )

    baseline_actions, baseline_activations, _ = _run_all_layers(
        adapter,
        request,
        layers,
        num_steps=num_steps,
        horizon=horizon,
        n_tokens=runtime.n_tokens,
        alpha=None,
        target_step=None,
    )
    assert baseline_activations is not None
    if baseline_actions.shape[:2] != (1, horizon):
        raise RuntimeError(
            f"Expected actions (1,{horizon},*), got {baseline_actions.shape}."
        )
    if int(baseline_actions.shape[2]) < action_dim:
        raise RuntimeError(
            f"Action width {baseline_actions.shape[2]} < {action_dim}."
        )
    identity_actions, _, _ = _run_all_layers(
        adapter,
        request,
        layers,
        num_steps=num_steps,
        horizon=horizon,
        n_tokens=runtime.n_tokens,
        alpha=None,
        target_step=0,
        baseline_activations=baseline_activations,
        identity_writeback=True,
    )
    if not torch.equal(identity_actions, baseline_actions):
        delta = (identity_actions - baseline_actions).to(torch.float64)
        rmse = float(delta.square().mean().sqrt().item())
        raise RuntimeError(f"Identity write-back moved actions: RMSE={rmse:.6e}.")
    print("identity write-back at step 0 on all DiT linears: actions unchanged")

    results: list[StepResult] = []
    for step in range(num_steps):
        changed_actions, _, stats = _run_all_layers(
            adapter,
            request,
            layers,
            num_steps=num_steps,
            horizon=horizon,
        n_tokens=runtime.n_tokens,
            alpha=args.l1_remove,
            target_step=step,
            baseline_activations=baseline_activations,
        )
        metrics = detail._action_metrics(
            changed_actions[0, :, :action_dim],
            baseline_actions[0, :, :action_dim],
        )
        ks = [item.k for item in stats.values()]
        l1s = [item.removed_l1_frac for item in stats.values()]
        toks = [item.token_clip_frac for item in stats.values()]
        row = StepResult(
            step=step,
            n_linears_clipped=len(stats),
            mean_k=sum(ks) / len(ks),
            mean_removed_l1_frac=sum(l1s) / len(l1s),
            mean_token_clip_frac=sum(toks) / len(toks),
            metrics=metrics,
        )
        results.append(row)
        print(
            f"[step {step}/{num_steps - 1}] linears={row.n_linears_clipped}  "
            f"L1={100 * row.mean_removed_l1_frac:.3f}%  "
            f"k={row.mean_k:.4g}  "
            f"tokens={100 * row.mean_token_clip_frac:.3f}%  "
            f"total={metrics.total_rmse:.3e}  "
            f"end={metrics.arm_endpoint_rmse:.3e}  "
            f"local={metrics.arm_local_rmse:.3e}  "
            f"grip={metrics.gripper_rmse:.3e}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "selective_clip_by_denoise_step.csv"
    png = args.output_dir / "selective_clip_by_denoise_step.png"
    _write_csv(results, csv_path)
    print(f"Wrote {csv_path}")
    _plot(results, png, alpha=args.l1_remove)
    print(f"Wrote {png}")
    ranked = sorted(results, key=lambda row: row.metrics.total_rmse, reverse=True)
    print("\nDenoise steps ranked by total action RMSE:")
    for row in ranked:
        print(
            f"  step={row.step}  total={row.metrics.total_rmse:.4e}  "
            f"end={row.metrics.arm_endpoint_rmse:.4e}  "
            f"local={row.metrics.arm_local_rmse:.4e}"
        )
    _print_step_trend(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
