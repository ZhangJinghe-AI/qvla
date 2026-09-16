#!/usr/bin/env python
"""Compare pi0.5 actions: original ``engine.step`` vs differentiable forward.

Loads one FP engine, runs the standard (non-differentiable) inference path and
the autograd-safe ``differentiable_step`` replacement on the same calibration
input and fixed diffusion noise, then reports action-level error metrics.

Use this to verify that ``adapters/pi05/differentiable_forward.py`` faithfully mirrors
``PI05WS1Scheduler.step`` before trusting Fisher / policy-aware rotation.

Example::

    uv run python tools/compare_differentiable_actions.py \\
        --checkpoint /data/share/pi05_libero_finetuned_v044 \\
        --calibration-source file \\
        --calibration-data ../calibration_data/libero_object_16_7.npz \\
        --num-samples 4 \\
        --noise-seed 0 \\
        --atol 0.05
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from qvla.adapters import get_adapter  # noqa: E402
from qvla.adapters.pi05.obs import build_pi05_request  # noqa: E402


@dataclass(frozen=True)
class ActionDiffMetrics:
    max_abs: float
    mean_abs: float
    abs_l2: float
    rel_l2: float
    cosine: float

    @classmethod
    def compute(cls, original: torch.Tensor, replaced: torch.Tensor) -> ActionDiffMetrics:
        a = original.detach().float().reshape(-1)
        b = replaced.detach().float().reshape(-1)
        if a.shape != b.shape:
            raise ValueError(f"shape mismatch: original {tuple(original.shape)} vs replaced {tuple(replaced.shape)}")
        diff = a - b
        abs_l2 = float(diff.norm().item())
        denom = b.norm().clamp_min(1e-12)
        return cls(
            max_abs=float(diff.abs().max().item()),
            mean_abs=float(diff.abs().mean().item()),
            abs_l2=abs_l2,
            rel_l2=float(diff.norm() / denom),
            cosine=float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()),
        )


@dataclass
class SampleResult:
    sample_index: int
    noise_seed: int
    shape: list[int]
    raw: ActionDiffMetrics
    trimmed: ActionDiffMetrics


def _make_noise(adapter, seed: int) -> torch.Tensor:
    sched = adapter._engine.entry.scheduler
    cfg = sched.cfg
    generator = torch.Generator(device=sched.device)
    generator.manual_seed(seed)
    return torch.randn(
        1,
        cfg.chunk_size,
        cfg.max_action_dim,
        generator=generator,
        device=sched.device,
        dtype=sched.params_dtype,
    )


def _build_request(adapter, batch: dict, noise: torch.Tensor):
    adapter._ensure_processor()
    request = build_pi05_request(
        adapter._processor, batch, state_dim=adapter.cfg.state_dim
    )
    return replace(request, noise=noise)


def _forward_original(adapter, request) -> torch.Tensor:
    # no_grad (not inference_mode): inference_mode marks KV buffers as
    # inference tensors and breaks the following autograd-enabled
    # differentiable_step.
    with torch.no_grad():
        out = adapter._engine.step(request)
    return out.clone()


def _forward_differentiable(adapter, request) -> torch.Tensor:
    from qvla.adapters.pi05.differentiable_forward import (
        differentiable_inference_context,
        differentiable_step,
        force_eager_runners,
        reset_differentiable_state,
    )
    from qvla.runtime.step_context import reset_step_counters

    engine = adapter._engine
    assert engine is not None
    sched = engine.entry.scheduler
    model = engine.entry.model
    force_eager_runners(sched)

    with differentiable_inference_context(sched):
        reset_step_counters(model)
        reset_differentiable_state(sched)
        # with torch.inference_mode():
        out = differentiable_step(sched, request)
        return out.clone()


def _trim_actions(actions: torch.Tensor, action_dim: int) -> torch.Tensor:
    return actions[..., :action_dim]


def _compare_sample(
    adapter,
    batch: dict,
    *,
    sample_index: int,
    noise_seed: int,
) -> SampleResult:
    noise = _make_noise(adapter, noise_seed)
    request = _build_request(adapter, batch, noise)

    original = _forward_original(adapter, request)
    replaced = _forward_differentiable(adapter, request)

    action_dim = adapter.cfg.action_dim
    return SampleResult(
        sample_index=sample_index,
        noise_seed=noise_seed,
        shape=list(original.shape),
        raw=ActionDiffMetrics.compute(original, replaced),
        trimmed=ActionDiffMetrics.compute(
            _trim_actions(original, action_dim),
            _trim_actions(replaced, action_dim),
        ),
    )


def _print_metrics(label: str, metrics: ActionDiffMetrics) -> None:
    print(
        f"  {label:8s}  max_abs={metrics.max_abs:.6e}  "
        f"mean_abs={metrics.mean_abs:.6e}  "
        f"abs_l2={metrics.abs_l2:.6e}  "
        f"rel_l2={metrics.rel_l2:.6e}  cosine={metrics.cosine:.8f}"
    )


def _print_sample(result: SampleResult) -> None:
    print(f"\nSample {result.sample_index}  noise_seed={result.noise_seed}  shape={tuple(result.shape)}")
    _print_metrics("raw", result.raw)
    _print_metrics("trimmed", result.trimmed)


def _aggregate(results: list[SampleResult], field: str) -> ActionDiffMetrics:
    rows = [getattr(r, field) for r in results]
    return ActionDiffMetrics(
        max_abs=max(m.max_abs for m in rows),
        mean_abs=sum(m.mean_abs for m in rows) / len(rows),
        abs_l2=sum(m.abs_l2 for m in rows) / len(rows),
        rel_l2=sum(m.rel_l2 for m in rows) / len(rows),
        cosine=sum(m.cosine for m in rows) / len(rows),
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--calibration-data", type=Path)
    p.add_argument("--calibration-source", choices=("file", "synthetic"), required=True)
    p.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of calibration batches to compare (default: 1).",
    )
    p.add_argument(
        "--sample-start",
        type=int,
        default=0,
        help="First calibration index (default: 0).",
    )
    p.add_argument(
        "--noise-seed",
        type=int,
        default=0,
        help="Diffusion noise seed shared by both forwards (default: 0).",
    )
    p.add_argument(
        "--atol",
        type=float,
        default=None,
        help="If set, exit 1 when trimmed max_abs exceeds this threshold.",
    )
    args = p.parse_args(argv)

    if args.calibration_source == "file" and args.calibration_data is None:
        p.error("--calibration-data is required when --calibration-source=file.")
    if args.num_samples < 1:
        p.error("--num-samples must be >= 1.")

    adapter_kwargs: dict = {
        "checkpoint_path": args.checkpoint,
        "calibration_source": args.calibration_source,
    }
    if args.calibration_source == "file":
        adapter_kwargs["calibration_data_path"] = args.calibration_data

    adapter = get_adapter("pi05", **adapter_kwargs)
    model = adapter.build_model()
    adapter.warmup_for_calibration(model)

    batches = list(adapter.iter_calibration_batches(args.sample_start + args.num_samples))
    if args.sample_start + args.num_samples > len(batches):
        p.error(
            f"Need calibration samples [{args.sample_start}, "
            f"{args.sample_start + args.num_samples}), but only {len(batches)} available."
        )

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Samples: {args.sample_start}..{args.sample_start + args.num_samples - 1}")
    print(f"Noise seed: {args.noise_seed}")

    results: list[SampleResult] = []
    for offset in range(args.num_samples):
        sample_index = args.sample_start + offset
        batch = batches[sample_index]
        result = _compare_sample(
            adapter,
            batch,
            sample_index=sample_index,
            noise_seed=args.noise_seed,
        )
        results.append(result)
        _print_sample(result)

    if len(results) > 1:
        print("\nAggregate (trimmed):")
        agg = _aggregate(results, "trimmed")
        _print_metrics("mean", agg)
        worst = max(results, key=lambda r: r.trimmed.max_abs)
        print(f"  worst sample: {worst.sample_index} (max_abs={worst.trimmed.max_abs:.6e})")

    if args.atol is not None:
        worst_max = max(r.trimmed.max_abs for r in results)
        if worst_max > args.atol:
            print(f"\nFAIL: trimmed max_abs {worst_max:.6e} > atol {args.atol:.6e}")
            return 1
        print(f"\nPASS: trimmed max_abs {worst_max:.6e} <= atol {args.atol:.6e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
