#!/usr/bin/env python
r"""Clip every DiT linear by the same *relative amount* and rank action impact.

Different layers have different activation scales and tail shapes, so a shared
absolute threshold would clip some layers almost fully and others not at all.
This script instead removes a fixed fraction ``α`` of each linear's input L1
mass, independently at every denoise step::

    choose τ such that  sum max(|x| − τ, 0)  /  sum |x|  = α
    x ← sign(x) * min(|x|, τ)

That τ is scale-equivariant (10× activations get 10× τ) and distribution-aware
(a heavier tail gets a higher τ, but still loses the same energy fraction).

Each linear is intervened in isolation. By default every denoise step of that
linear is clipped; ``--target-step`` clips only one step. Other linears are
untouched. Activations are clipped in fp32 on the live tensor (bf16→fp32 is
exact); unclipped values are written back unchanged. The script aborts if an
identity write-back already moves the actions.

Action change uses the same overall/detail metrics as
``analyze_dit_outlier_action_detail.py``.

Example:

    CUDA_VISIBLE_DEVICES=5 uv run python \
      tools/analyze_dit_matched_clip_action_impact.py \
      --checkpoint /data/share/pi05_libero_finetuned_v044 \
      --calibration-data ../calibration_data/libero_goal_30_7_demo.npz \
      --sample-index 0 --noise-seed 0 --l1-remove 0.05 --target-step 4 \
      --output-dir tools/img/dit_matched_clip_action_impact_a05_step4_exact
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import re
import sys
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
_TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_TOOLS))

import analyze_dit_outlier_action_detail as detail  # noqa: E402
from qvla.adapters import get_adapter  # noqa: E402
from qvla.adapters.pi05.obs import build_pi05_request  # noqa: E402
from qvla.adapters.pi05.step_hook import find_expert_runner, patched_one_step  # noqa: E402
from qvla.config import QVLAConfig  # noqa: E402
from qvla.runtime import list_target_modules  # noqa: E402

KIND_ORDER = (
    "qkv_proj",
    "o_proj",
    "gate_up_proj",
    "down_proj",
    "to_q",
    "to_k",
    "to_v",
    "to_out",
    "fc1",
    "fc2",
)
KIND_COLORS = {
    "qkv_proj": "#4C78A8",
    "o_proj": "#F58518",
    "gate_up_proj": "#54A24B",
    "down_proj": "#E45756",
    "to_q": "#4C78A8",
    "to_k": "#72B7B2",
    "to_v": "#54A24B",
    "to_out": "#F58518",
    "fc1": "#E45756",
    "fc2": "#B279A2",
}
L1_MATCH_ATOL = 1e-8
_LAYER_IDX_RE = re.compile(r"(?:layers|transformer_blocks)\.(\d+)\.")
_ATTN_KV_RE = re.compile(r"attn1\.to_[kv]$")


@dataclass(frozen=True)
class ClipRuntime:
    num_steps: int
    action_horizon: int
    n_tokens: int
    noise_width: int
    action_dim: int
    device: torch.device
    dtype: torch.dtype


def _must_match_baseline(step: int, target_step: int | None) -> bool:
    """Capture equality holds until (and including) the first clipped step."""
    if target_step is None:
        return step == 0
    return step <= target_step


def _clip_this_step(step: int, target_step: int | None) -> bool:
    if target_step is None:
        return True
    return step == target_step


@dataclass(frozen=True)
class ClipStats:
    tau: float
    removed_l1_frac: float
    token_clip_frac: float


@dataclass(frozen=True)
class LayerResult:
    layer_name: str
    layer_idx: int
    kind: str
    mean_tau: float
    mean_removed_l1_frac: float
    mean_token_clip_frac: float
    metrics: detail.ActionMetrics


def _layer_idx(name: str) -> int:
    match = _LAYER_IDX_RE.search(name)
    if match is None:
        raise RuntimeError(f"Cannot parse layer index from {name!r}.")
    return int(match.group(1))


def _layer_kind(name: str) -> str:
    for kind in KIND_ORDER:
        if name.endswith(kind):
            return kind
    raise RuntimeError(f"Unknown DiT linear kind in {name!r}.")


def _is_prefix_only_kv(model: torch.nn.Module, name: str) -> bool:
    """True for GR00T cross-attn K/V, which run once before the denoise loop."""
    if _ATTN_KV_RE.search(name) is None or ".attn1." not in name:
        return False
    block_name = name.rsplit(".attn1.", 1)[0]
    try:
        block = model.get_submodule(block_name)
    except AttributeError as exc:
        raise RuntimeError(f"Cannot resolve parent block for {name!r}.") from exc
    return bool(getattr(block, "is_cross_attention", False))


def _dit_layers(
    model: torch.nn.Module,
    regex: str | None,
    *,
    config: QVLAConfig | None = None,
    skip_prefix_kv: bool = True,
) -> list[tuple[str, torch.nn.Module]]:
    config = QVLAConfig.pi05_default() if config is None else config
    hits = [
        (name, module)
        for name, scope, module in list_target_modules(model, config)
        if scope == "dit"
    ]
    if regex:
        pattern = re.compile(regex)
        hits = [(name, module) for name, module in hits if pattern.search(name)]
    if skip_prefix_kv:
        hits = [
            (name, module)
            for name, module in hits
            if not _is_prefix_only_kv(model, name)
        ]
    if not hits:
        raise RuntimeError("No per-step DiT linears matched.")
    return sorted(
        hits,
        key=lambda item: (_layer_idx(item[0]), KIND_ORDER.index(_layer_kind(item[0]))),
    )


def add_model_cli(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", choices=("pi05", "groot_n17"), default="pi05")
    parser.add_argument("--embodiment-tag", default=None)
    parser.add_argument("--processor-model-name-or-path", default=None)
    parser.add_argument(
        "--action-dim",
        type=int,
        default=None,
        help="Action dims used for RMSE (LIBERO=7). Default: adapter config.",
    )


def adapter_from_args(args: argparse.Namespace):
    kwargs: dict = {
        "checkpoint_path": args.checkpoint,
        "calibration_source": "file",
        "calibration_data_path": args.calibration_data,
        "device": args.device,
        "params_dtype": args.params_dtype,
    }
    if args.model == "groot_n17":
        from qvla.adapters.groot.config import DEFAULT_PROCESSOR_MODEL_NAME_OR_PATH

        kwargs["embodiment_tag"] = args.embodiment_tag or "LIBERO_PANDA"
        kwargs["processor_model_name_or_path"] = (
            args.processor_model_name_or_path or DEFAULT_PROCESSOR_MODEL_NAME_OR_PATH
        )
        kwargs["action_dim"] = (
            int(args.action_dim) if args.action_dim is not None else 7
        )
    else:
        if args.embodiment_tag is not None:
            raise ValueError("--embodiment-tag is only valid with --model groot_n17.")
        if args.processor_model_name_or_path is not None:
            raise ValueError(
                "--processor-model-name-or-path is only valid with --model groot_n17."
            )
        if args.action_dim is not None:
            kwargs["action_dim"] = int(args.action_dim)
    return get_adapter(args.model, **kwargs)


def clip_runtime(adapter, config: QVLAConfig) -> ClipRuntime:
    sched = adapter.engine.entry.scheduler
    num_steps = adapter.dit_step_count(config)
    if adapter.model_kind == "groot_n17":
        head = sched.model.action_head
        action_horizon = int(head.action_horizon)
        # DiT tokens are cat(state_token, action_tokens).
        n_tokens = 1 + action_horizon
        noise_width = int(head.action_dim)
        dtype = sched.model.params_dtype
        device = sched.device
    elif adapter.model_kind == "pi05":
        action_horizon = int(sched.cfg.chunk_size)
        n_tokens = action_horizon
        noise_width = int(sched.cfg.max_action_dim)
        dtype = sched.params_dtype
        device = sched.device
    else:
        raise RuntimeError(f"Unsupported model_kind={adapter.model_kind!r}.")
    action_dim = int(adapter.cfg.action_dim)
    if action_dim < 2:
        raise RuntimeError(f"Need arm+gripper action_dim>=2, got {action_dim}.")
    if num_steps < 1 or action_horizon < 2 or n_tokens < 2 or noise_width < 1:
        raise RuntimeError(
            f"Invalid clip runtime: steps={num_steps}, horizon={action_horizon}, "
            f"n_tokens={n_tokens}, noise_width={noise_width}."
        )
    return ClipRuntime(
        num_steps=num_steps,
        action_horizon=action_horizon,
        n_tokens=n_tokens,
        noise_width=noise_width,
        action_dim=action_dim,
        device=torch.device(device),
        dtype=dtype,
    )


def _fixed_noise_request(adapter, batch, runtime: ClipRuntime, *, noise_seed: int):
    adapter._ensure_processor()
    generator = torch.Generator(device=runtime.device)
    generator.manual_seed(int(noise_seed))
    noise = torch.randn(
        1,
        runtime.action_horizon,
        runtime.noise_width,
        generator=generator,
        device=runtime.device,
        dtype=runtime.dtype,
    )
    if adapter.model_kind == "groot_n17":
        from phyai.models.gr00t_n17.scheduler_ws1_gr00t_n17 import GR00TN17Request
        from qvla.adapters.groot.obs import build_groot_request

        raw = build_groot_request(
            adapter._processor,
            batch,
            device=str(runtime.device),
        )
        return GR00TN17Request(tensors=raw.tensors, noise=noise)
    request = build_pi05_request(
        adapter._processor,
        batch,
        state_dim=adapter.cfg.state_dim,
    )
    return replace(request, noise=noise)


@contextlib.contextmanager
def _with_denoise_callback(adapter, step_callback):
    if adapter.model_kind == "groot_n17":
        from qvla.adapters.groot.step_hook import (
            find_action_head_runner,
            patched_action_head_denoise,
        )

        runner = find_action_head_runner(adapter.engine)
        with patched_action_head_denoise(runner, step_callback):
            yield
        return
    if adapter.model_kind != "pi05":
        raise RuntimeError(f"Unsupported model_kind={adapter.model_kind!r}.")
    runner = find_expert_runner(adapter.engine)
    with patched_one_step(runner, step_callback):
        yield


def _index_tertile_bands(max_idx: int) -> list[tuple[str, int, int]]:
    if max_idx < 0:
        raise ValueError(f"max_idx must be >= 0, got {max_idx}.")
    n = max_idx + 1
    a = n // 3
    b = (2 * n) // 3
    if a < 1 or b <= a:
        return [(f"L0-{max_idx}", 0, max_idx)]
    return [
        (f"early L0-{a - 1}", 0, a - 1),
        (f"mid L{a}-{b - 1}", a, b - 1),
        (f"late L{b}-{max_idx}", b, max_idx),
    ]


def _step_tertile_slices(n_steps: int) -> tuple[slice, slice, slice]:
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}.")
    side = max(1, n_steps // 3)
    if 2 * side >= n_steps:
        return slice(0, 1), slice(1, max(1, n_steps - 1)), slice(n_steps - 1, n_steps)
    return slice(0, side), slice(side, n_steps - side), slice(n_steps - side, n_steps)


def _matched_l1_threshold(
    abs_flat: torch.Tensor, alpha: float
) -> tuple[float, float, float]:
    """Return ``(τ, removed_L1_frac, token_clip_frac)`` for one activation tensor."""
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    values = abs_flat.detach().to(torch.float64).reshape(-1)
    if int(values.numel()) < 2:
        raise ValueError(f"Need >= 2 activation values, got {int(values.numel())}.")
    if not bool(torch.isfinite(values).all().item()):
        raise RuntimeError("Activation L1 clip requires finite |x|.")
    total = float(values.sum().item())
    if total <= 0.0:
        raise RuntimeError("Activation L1 norm is non-positive.")
    target = float(alpha) * total
    descending, _ = torch.sort(values, descending=True)
    prefix = torch.cumsum(descending, dim=0)
    count = int(descending.numel())
    tau: float | None = None
    for kept in range(1, count + 1):
        candidate = (float(prefix[kept - 1].item()) - target) / float(kept)
        high = float(descending[kept - 1].item())
        low = float(descending[kept].item()) if kept < count else 0.0
        if low <= candidate < high:
            tau = candidate
            break
    if tau is None:
        raise RuntimeError(
            f"Cannot remove L1 fraction {alpha:.6g} from this activation "
            f"(total={total:.6g}, max={float(descending[0].item()):.6g})."
        )
    removed = float((values - tau).clamp(min=0.0).sum().item())
    removed_frac = removed / total
    if abs(removed_frac - float(alpha)) > L1_MATCH_ATOL:
        raise RuntimeError(
            f"L1 clip missed the target: want {alpha:.8g}, got {removed_frac:.8g}."
        )
    token_frac = float((values > tau).to(torch.float64).mean().item())
    if token_frac <= 0.0:
        raise RuntimeError("L1 clip selected no elements.")
    return tau, removed_frac, token_frac


def _exact_fp32(x: torch.Tensor) -> torch.Tensor:
    """Promote activations to fp32 without extra rounding.

    bf16→fp32 is exact. fp16→fp32 is not, and writing that snapshot back was
    the measurement floor that hid the clip effect.
    """
    if x.dtype == torch.float16:
        raise RuntimeError(
            "Refusing to snapshot fp16 activations; that rounding is the "
            "measurement floor this script removes. Use bf16 or fp32."
        )
    return x.detach().to(torch.float32)


def _clip_matched_l1(x: torch.Tensor, alpha: float) -> tuple[torch.Tensor, ClipStats]:
    tau, removed_frac, token_frac = _matched_l1_threshold(
        x.detach().abs().to(torch.float32).cpu(), alpha
    )
    clipped = x.sign() * x.abs().clamp(max=tau)
    return clipped, ClipStats(
        tau=tau, removed_l1_frac=removed_frac, token_clip_frac=token_frac
    )


def _run_layer(
    adapter,
    request,
    layer: torch.nn.Module,
    *,
    num_steps: int,
    horizon: int,
    alpha: float | None,
    baseline_activations: torch.Tensor | None = None,
    target_step: int | None = None,
    identity_writeback: bool = False,
    clip_fn=None,
    n_tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, list]:
    capture = baseline_activations is None
    if capture:
        if alpha is not None or identity_writeback:
            raise ValueError("Capture must use alpha=None without write-back.")
    elif identity_writeback:
        if alpha is not None:
            raise ValueError("Identity write-back must not clip.")
    elif alpha is None:
        raise ValueError("Intervention requires a clip fraction.")
    if target_step is not None and not (0 <= int(target_step) < num_steps):
        raise ValueError(
            f"target_step must be in [0, {num_steps}), got {target_step}."
        )
    weight = getattr(layer, "weight", None)
    if not torch.is_tensor(weight) or weight.ndim != 2:
        raise RuntimeError("Target layer must have a 2-D tensor weight.")
    in_features = int(weight.shape[1])
    current_step: list[int | None] = [None]
    callbacks: list[int | None] = []
    captured: dict[int, torch.Tensor] = {}
    stats: dict[int, object] = {}
    applied = 0

    def hook(_module, inputs):
        nonlocal applied
        step = current_step[0]
        if step is None:
            raise RuntimeError(
                "Target linear ran outside the denoise loop; choose a per-step "
                "DiT action linear."
            )
        if len(inputs) != 1 or not torch.is_tensor(inputs[0]):
            raise RuntimeError("Target linear must receive exactly one tensor input.")
        if step in captured:
            raise RuntimeError(f"Target linear ran more than once at step {step}.")
        x = inputs[0]
        if int(x.shape[-1]) != in_features:
            raise RuntimeError(
                f"Input width {x.shape[-1]} != in_features={in_features}."
            )
        flat = x.reshape(-1, in_features)
        token_count = horizon if n_tokens is None else n_tokens
        if int(flat.shape[0]) != token_count:
            raise RuntimeError(
                f"Expected {token_count} DiT tokens, got {flat.shape[0]}."
            )
        live = _exact_fp32(flat)
        snapshot = live.cpu()
        detail._finite(snapshot, f"activation at step {step}")
        captured[step] = snapshot
        if capture:
            return None
        assert baseline_activations is not None
        if _must_match_baseline(step, target_step) and not torch.equal(
            snapshot, baseline_activations[step]
        ):
            raise RuntimeError(
                f"Repeated fixed-noise forward diverged at step {step}, "
                "before clipping."
            )
        if not _clip_this_step(step, target_step):
            return None
        applied += 1
        if identity_writeback:
            replaced = live.to(device=x.device, dtype=x.dtype)
        else:
            assert alpha is not None
            apply = _clip_matched_l1 if clip_fn is None else clip_fn
            clipped, clip_stats = apply(live, alpha)
            stats[step] = clip_stats
            replaced = clipped.to(device=x.device, dtype=x.dtype)
        return (replaced.reshape_as(x),)

    def step_callback(step: int | None) -> None:
        value = None if step is None else int(step)
        current_step[0] = value
        callbacks.append(value)

    handle = layer.register_forward_pre_hook(hook)
    try:
        with _with_denoise_callback(adapter, step_callback):
            step_callback(None)
            with torch.inference_mode():
                actions = adapter.engine.step(request)
    finally:
        handle.remove()

    expected = [None, *range(num_steps)]
    if callbacks != expected:
        raise RuntimeError(f"Denoise callback order {callbacks} != {expected}.")
    if sorted(captured) != list(range(num_steps)):
        raise RuntimeError(
            f"Captured steps {sorted(captured)} != {list(range(num_steps))}."
        )
    expected_applied = num_steps if target_step is None else 1
    if not capture and applied != expected_applied:
        raise RuntimeError(
            f"Clip applied {applied} times, expected {expected_applied}."
        )
    if not torch.is_tensor(actions):
        raise RuntimeError(f"Engine returned {type(actions)}, expected tensor.")
    actions = actions.detach().to(torch.float32).cpu()
    detail._finite(actions, "predicted actions")
    activations = (
        torch.stack([captured[step] for step in range(num_steps)])
        if capture
        else None
    )
    if capture or identity_writeback:
        clip_stats = []
    elif target_step is None:
        clip_stats = [stats[step] for step in range(num_steps)]
    else:
        clip_stats = [stats[target_step]]
    return actions, activations, clip_stats


def _mean(xs: list[float]) -> float:
    if not xs:
        raise ValueError("Cannot average an empty list.")
    return sum(xs) / len(xs)


def _plot(
    results: list[LayerResult],
    output: Path,
    *,
    alpha: float,
    target_step: int | None,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    xs = np.arange(len(results))
    labels = [f"L{row.layer_idx}.{row.kind.replace('_proj', '')}" for row in results]
    bar_colors = [KIND_COLORS.get(row.kind, "#9D755D") for row in results]

    ax = axes[0, 0]
    ax.bar(xs, [row.metrics.total_rmse for row in results], color=bar_colors)
    ax.set_title("Total action RMSE after matched-L1 clip")
    ax.set_ylabel("action RMSE")

    ax = axes[0, 1]
    ax.plot(xs, [row.metrics.arm_mean_shift_rmse for row in results], "o-", label="mean")
    ax.plot(xs, [row.metrics.arm_endpoint_rmse for row in results], "o-", label="endpoint")
    ax.plot(xs, [row.metrics.arm_net_disp_rmse for row in results], "o-", label="start-to-end")
    ax.set_title("Overall arm change")
    ax.set_ylabel("RMSE")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(xs, [row.metrics.arm_local_rmse for row in results], "o-", label="local")
    ax.plot(xs, [row.metrics.arm_step_rmse for row in results], "o-", label="step")
    ax.plot(xs, [row.metrics.gripper_rmse for row in results], "o-", label="gripper")
    ax.set_title("Local / gripper change")
    ax.set_ylabel("RMSE")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(xs, [100.0 * row.mean_token_clip_frac for row in results], "o-", label="token clip %")
    ax.axhline(100.0 * alpha, color="black", linestyle="--", label=f"L1 target {100 * alpha:.2f}%")
    ax.plot(
        xs,
        [100.0 * row.mean_removed_l1_frac for row in results],
        "s--",
        label="removed L1 %",
    )
    ax.set_title("Clip intensity (should share L1 %, not token %)")
    ax.set_ylabel("percent")
    ax.legend(fontsize=8)

    stride = max(1, len(labels) // 18)
    ticks = list(range(0, len(labels), stride))
    for ax in axes.flat:
        ax.set_xticks(ticks)
        ax.set_xticklabels([labels[i] for i in ticks], rotation=75, fontsize=7)
        ax.grid(alpha=0.25, axis="y")
        ax.set_xlabel("DiT linear")
    step_label = (
        "every denoise step"
        if target_step is None
        else f"denoise step {target_step} only"
    )
    fig.suptitle(
        f"Matched L1 clip α={alpha:g} on {step_label}, one linear at a time",
        fontsize=12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Failed to write {output}.")


def _write_csv(results: list[LayerResult], output: Path) -> None:
    metric_names = [field.name for field in fields(detail.ActionMetrics)]
    fieldnames = [
        "layer",
        "layer_idx",
        "kind",
        "mean_tau",
        "mean_removed_l1_frac",
        "mean_token_clip_frac",
        *metric_names,
    ]
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            item = {
                "layer": row.layer_name,
                "layer_idx": row.layer_idx,
                "kind": row.kind,
                "mean_tau": row.mean_tau,
                "mean_removed_l1_frac": row.mean_removed_l1_frac,
                "mean_token_clip_frac": row.mean_token_clip_frac,
            }
            for name in metric_names:
                item[name] = getattr(row.metrics, name)
            writer.writerow(item)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-data", type=Path, required=True)
    add_model_cli(parser)
    parser.add_argument("--layer-regex", default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument(
        "--l1-remove",
        type=float,
        default=0.01,
        help="Fraction of each linear's |x| L1 mass to clip away (default: 0.01).",
    )
    parser.add_argument(
        "--target-step",
        type=int,
        default=None,
        help=(
            "Clip only this 0-indexed denoise step. Default: clip every step."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--params-dtype", default="bfloat16")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tools/img/dit_matched_clip_action_impact"),
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
    if args.target_step is not None and args.target_step < 0:
        raise ValueError(f"--target-step must be >= 0, got {args.target_step}.")

    adapter = adapter_from_args(args)
    model = adapter.build_model()
    model.eval()
    adapter.warmup_for_calibration(model)
    config = QVLAConfig.for_model_kind(args.model)
    layers = _dit_layers(model, args.layer_regex, config=config)
    runtime = clip_runtime(adapter, config)
    num_steps = runtime.num_steps
    horizon = runtime.action_horizon
    action_dim = runtime.action_dim

    batches = list(adapter.iter_calibration_batches(args.sample_index + 1))
    if len(batches) != args.sample_index + 1:
        raise RuntimeError(
            f"Requested sample {args.sample_index}, calibration yielded "
            f"{len(batches)} samples."
        )
    request = _fixed_noise_request(
        adapter, batches[args.sample_index], runtime, noise_seed=args.noise_seed
    )
    if args.target_step is not None and args.target_step >= num_steps:
        raise ValueError(
            f"--target-step must be in [0, {num_steps}), got {args.target_step}."
        )
    print(
        f"model={args.model}, layers={len(layers)}, sample={args.sample_index}, "
        f"noise={args.noise_seed}, steps={num_steps}, horizon={horizon}, "
        f"n_tokens={runtime.n_tokens}, action_dim={action_dim}, "
        f"l1_remove={args.l1_remove}, target_step={args.target_step}"
    )

    probe_name, probe_layer = layers[0]
    baseline_actions, baseline_activations, _ = _run_layer(
        adapter,
        request,
        probe_layer,
        num_steps=num_steps,
        horizon=horizon,
        n_tokens=runtime.n_tokens,
        alpha=None,
    )
    assert baseline_activations is not None
    identity_actions, _, _ = _run_layer(
        adapter,
        request,
        probe_layer,
        num_steps=num_steps,
        horizon=horizon,
        n_tokens=runtime.n_tokens,
        alpha=None,
        baseline_activations=baseline_activations,
        target_step=args.target_step,
        identity_writeback=True,
    )
    if not torch.equal(identity_actions, baseline_actions):
        delta = (identity_actions - baseline_actions).to(torch.float64)
        rmse = float(delta.square().mean().sqrt().item())
        raise RuntimeError(
            f"Identity write-back moved actions on {probe_name}: RMSE={rmse:.6e}. "
            "Clip measurements would still include hook noise."
        )
    print(f"identity write-back on {probe_name}: actions unchanged")

    results: list[LayerResult] = []
    for index, (layer_name, layer) in enumerate(layers, start=1):
        baseline_actions, baseline_activations, _ = _run_layer(
            adapter,
            request,
            layer,
            num_steps=num_steps,
            horizon=horizon,
            n_tokens=runtime.n_tokens,
            alpha=None,
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
        changed_actions, _, clip_stats = _run_layer(
            adapter,
            request,
            layer,
            num_steps=num_steps,
            horizon=horizon,
            n_tokens=runtime.n_tokens,
            alpha=args.l1_remove,
            baseline_activations=baseline_activations,
            target_step=args.target_step,
        )
        metrics = detail._action_metrics(
            changed_actions[0, :, :action_dim],
            baseline_actions[0, :, :action_dim],
        )
        row = LayerResult(
            layer_name=layer_name,
            layer_idx=_layer_idx(layer_name),
            kind=_layer_kind(layer_name),
            mean_tau=_mean([item.tau for item in clip_stats]),
            mean_removed_l1_frac=_mean(
                [item.removed_l1_frac for item in clip_stats]
            ),
            mean_token_clip_frac=_mean(
                [item.token_clip_frac for item in clip_stats]
            ),
            metrics=metrics,
        )
        results.append(row)
        print(
            f"[{index}/{len(layers)}] {layer_name}  "
            f"L1={100 * row.mean_removed_l1_frac:.3f}%  "
            f"tokens={100 * row.mean_token_clip_frac:.3f}%  "
            f"tau={row.mean_tau:.4g}  "
            f"total={metrics.total_rmse:.3e}  "
            f"end={metrics.arm_endpoint_rmse:.3e}  "
            f"local={metrics.arm_local_rmse:.3e}  "
            f"grip={metrics.gripper_rmse:.3e}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png = args.output_dir / "matched_l1_clip_action_impact.png"
    csv_path = args.output_dir / "matched_l1_clip_action_impact.csv"
    _plot(results, png, alpha=args.l1_remove, target_step=args.target_step)
    _write_csv(results, csv_path)
    for path in (png, csv_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Failed to write {path}.")
        print(f"Wrote {path}")
    ranked = sorted(results, key=lambda row: row.metrics.total_rmse, reverse=True)
    print("\nTop-5 layers by total action RMSE:")
    for row in ranked[:5]:
        print(
            f"  {row.layer_name}  total={row.metrics.total_rmse:.4e}  "
            f"end={row.metrics.arm_endpoint_rmse:.4e}  "
            f"local={row.metrics.arm_local_rmse:.4e}"
        )
    _print_depth_trend(results)
    return 0


def _print_depth_trend(results: list[LayerResult]) -> None:
    if len(results) < 3:
        return
    layers = np.array([row.layer_idx for row in results], dtype=np.float64)
    rmse = np.array([row.metrics.total_rmse for row in results], dtype=np.float64)
    pearson = float(np.corrcoef(layers, rmse)[0, 1])
    spearman = float(
        np.corrcoef(layers.argsort().argsort(), rmse.argsort().argsort())[0, 1]
    )
    bands = _index_tertile_bands(int(max(row.layer_idx for row in results)))
    print("\nDepth trend of total action RMSE:")
    print(f"  Pearson(layer, RMSE)={pearson:.3f}  Spearman={spearman:.3f}")
    for label, lo, hi in bands:
        vals = [row.metrics.total_rmse for row in results if lo <= row.layer_idx <= hi]
        if vals:
            print(f"  {label}: mean={sum(vals) / len(vals):.3e}  n={len(vals)}")
    print("  by kind:")
    for kind in KIND_ORDER:
        subset = [row for row in results if row.kind == kind]
        if len(subset) < 3:
            continue
        xs = np.array([row.layer_idx for row in subset], dtype=np.float64)
        ys = np.array([row.metrics.total_rmse for row in subset], dtype=np.float64)
        r = float(np.corrcoef(xs, ys)[0, 1])
        print(f"    {kind}: Pearson={r:.3f}  mean={float(ys.mean()):.3e}")


if __name__ == "__main__":
    raise SystemExit(main())
