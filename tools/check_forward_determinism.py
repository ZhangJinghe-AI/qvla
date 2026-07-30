#!/usr/bin/env python
"""Repeat pi0.5 forward on a fixed request; find where outputs start to diverge.

Compares repeated ``engine.step`` outputs on the same input + fixed noise.
Optionally compares bs=1 vs bs=N for the same target sample (slot 0 in the batch).

Example::

    cd /path/to/qvla
    CUDA_VISIBLE_DEVICES=0 uv run python tools/check_forward_determinism.py \\
        --checkpoint /data/share/pi05_libero_finetuned_v044 \\
        --runs 5

    uv run python tools/check_forward_determinism.py \\
        --checkpoint /data/share/pi05_libero_finetuned_v044 \\
        --calibration-source file \\
        --calibration-data ../calibration_data/libero_object_16_7.npz \\
        --sample-index 0

    uv run python tools/check_forward_determinism.py \\
        --checkpoint /data/share/pi05_libero_finetuned_v044 \\
        --calibration-source file \\
        --calibration-data ../calibration_data/libero_object_16_7.npz \\
        --sample-index 0 \\
        --batch-size 8
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from qvla.adapters import get_adapter  # noqa: E402
from qvla.adapters.pi05.config import PI05AdapterConfig, read_checkpoint_config  # noqa: E402
from qvla.adapters.pi05.obs import build_pi05_request, obs_to_transition  # noqa: E402

# pi05-phyai.py LIBERO defaults (FP inference path).
_DEVICE = "cuda"
_PARAMS_DTYPE = "bfloat16"
_VISION_PARAMS_DTYPE = "float16"
_USE_CUDA_GRAPH = False
_ATTN_BACKEND = "flashinfer"
_NORM_BACKEND = "phyai-kernel"
_LINEAR_BACKEND = "torch"
_INFERENCE_SEED = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--calibration-data", type=Path)
    p.add_argument(
        "--calibration-source",
        choices=("file", "synthetic"),
        default="synthetic",
        help="Use real LIBERO samples from .npz (file) or fixed random obs (synthetic).",
    )
    p.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Target calibration sample; placed at batch slot 0 when batch-size > 1.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="If > 1, compare bs=1 vs this batch size for the target sample (default: 1).",
    )
    p.add_argument(
        "--filler",
        choices=("distinct", "same"),
        default="distinct",
        help=(
            "Batch slots 1..N-1: 'distinct' uses other calibration samples; "
            "'same' replicates the target so lang_lens and lang bucket match bs=1."
        ),
    )
    p.add_argument(
        "--disable-split-kv",
        action="store_true",
        help="Pass flashinfer disable_split_kv=True for batch-size-invariant attention.",
    )
    return p.parse_args()


def _make_noise(sched, batch_size: int = 1) -> torch.Tensor:
    cfg = sched.cfg
    gen = torch.Generator(device="cpu")
    gen.manual_seed(_INFERENCE_SEED)
    noise = torch.randn(
        1,
        cfg.chunk_size,
        cfg.max_action_dim,
        generator=gen,
        dtype=torch.float32,
    )
    if batch_size == 1:
        return noise
    # Match pi05-phyai: one seeded draw, expand so slot 0 noise equals bs=1.
    return noise.expand(batch_size, -1, -1).contiguous()


def _load_calibration_samples(args: argparse.Namespace, adapter) -> list[dict]:
    if args.calibration_source == "file":
        from qvla.calibration.file import load_calibration_dataset

        batches, _meta = load_calibration_dataset(args.calibration_data)
        return batches
    # Mixed-companion stage needs two disjoint filler sets after the target.
    need = max(args.sample_index + 1, args.batch_size, args.sample_index + 2 * args.batch_size - 1)
    return list(adapter.iter_calibration_batches(need))


def _batch_obs_list(
    samples: list[dict],
    sample_index: int,
    batch_size: int,
    filler: str,
) -> list[dict]:
    n = len(samples)
    if sample_index >= n:
        raise SystemExit(f"--sample-index {sample_index} out of range ({n} samples available).")
    if filler == "same":
        return [samples[sample_index]] * batch_size
    return [samples[(sample_index + i) % n] for i in range(batch_size)]


def _report_lang_layout(sched, name: str, request) -> None:
    """Print language lengths and their resulting prefix bucket."""
    lens = [int(x) for x in request.lang_lens.tolist()]
    n_per_sample = sched._bucket_n_per_sample(max(lens))
    lang_bucket = n_per_sample - sched.image_token_count
    print(
        f"  {name}: lang_lens={lens}  max_lang={max(lens)}  "
        f"lang_bucket={lang_bucket}  n_per_sample={n_per_sample}"
    )


def _build_batch_request(processor, obs_list: list[dict], *, state_dim: int):
    from phyai.models.pi05.scheduler_ws1_pi05 import PI05Request

    processed = [
        processor.preprocess(obs_to_transition(obs, state_dim=state_dim)) for obs in obs_list
    ]
    return PI05Request(
        pixel_values=torch.cat([p.pixel_values for p in processed], dim=0),
        input_ids=torch.cat([p.input_ids for p in processed], dim=0),
        lang_lens=torch.cat([p.lang_lens for p in processed], dim=0),
    )


def _report_diff(name: str, ref: torch.Tensor, cur: torch.Tensor) -> float:
    diff = (cur.float().cpu() - ref.float().cpu()).abs()
    max_d = float(diff.max())
    min_d = float(diff.min())
    print(f"  {name}: max={max_d:.6e} min={min_d:.6e}")
    return max_d


def _build_engine(cfg: PI05AdapterConfig, *, max_batch_size: int, disable_split_kv: bool):
    """Match vla_eval ``pi05-phyai.py`` FP engine wiring."""
    from phyai.engine import Engine, EngineArgs
    from phyai.engine_config import BackendConfig, DeviceConfig, EngineConfig, RuntimeConfig
    from phyai.models.pi05.main_pi05 import PI05Args
    from qvla.adapters.pi05.config import lerobot_weight_remap, resolve_dtype

    # flashinfer 0.6.x: disable_split_kv under CUDA-graph mode pads the
    # launch grid but skips allocating block_valid_mask → IMA. Refuse the combo.
    if disable_split_kv and _USE_CUDA_GRAPH:
        raise SystemExit(
            "--disable-split-kv requires _USE_CUDA_GRAPH=False "
            "(flashinfer 0.6.x IMAs when both are enabled); "
            "edit _USE_CUDA_GRAPH in this script and rerun."
        )
    params_dtype = resolve_dtype(_PARAMS_DTYPE)
    vision_params_dtype = resolve_dtype(_VISION_PARAMS_DTYPE)
    return Engine(
        EngineArgs(
            plugin="pi05",
            plugin_args=PI05Args(
                checkpoint_dir=cfg.checkpoint_path,
                max_batch_size=max_batch_size,
                vision_params_dtype=vision_params_dtype,
                weight_remap=lerobot_weight_remap,
                inputs_image_shape=[
                    [cfg.image_size, cfg.image_size, 3]
                    for _ in range(cfg.num_real_cameras)
                ],
            ),
            config=EngineConfig(
                backends=BackendConfig(
                    attn=_ATTN_BACKEND,
                    norm=_NORM_BACKEND,
                    linear=_LINEAR_BACKEND,
                ),
                device=DeviceConfig(target=_DEVICE, params_dtype=params_dtype),
                runtime=RuntimeConfig(
                    use_cuda_graph=_USE_CUDA_GRAPH,
                    force_linear_kernel=_LINEAR_BACKEND,
                    flashinfer_workspace_bytes=max(
                        256 * 1024 * 1024,
                        64 * 1024 * 1024 * max_batch_size,
                    ),
                    flashinfer_disable_split_kv=disable_split_kv,
                ),
            ),
        )
    )


def main() -> int:
    args = parse_args()
    if args.calibration_source == "file" and args.calibration_data is None:
        raise SystemExit("--calibration-data is required when --calibration-source=file.")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1.")
    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        return 1

    raw = read_checkpoint_config(args.checkpoint)
    print("=== checkpoint ===")
    print(f"path: {args.checkpoint}")
    print(f"n_action_steps={raw.get('n_action_steps')}  chunk_size={raw.get('chunk_size')}")
    engine_bs = max(1, args.batch_size)
    print(
        f"attn={_ATTN_BACKEND}  cuda_graph={_USE_CUDA_GRAPH}  "
        f"disable_split_kv={args.disable_split_kv}  "
        f"inference_seed={_INFERENCE_SEED}  calibration={args.calibration_source}  "
        f"max_batch_size={engine_bs}"
    )

    adapter_kwargs: dict = {
        "checkpoint_path": args.checkpoint,
        "calibration_source": args.calibration_source,
    }
    if args.calibration_source == "file":
        adapter_kwargs["calibration_data_path"] = args.calibration_data

    adapter = get_adapter("pi05", **adapter_kwargs)
    adapter._engine = _build_engine(
        adapter.cfg,
        max_batch_size=engine_bs,
        disable_split_kv=args.disable_split_kv,
    )
    adapter._ensure_processor()

    sched = adapter._engine.entry.scheduler
    samples = _load_calibration_samples(args, adapter)
    target_obs = samples[args.sample_index]
    single_request = replace(
        build_pi05_request(adapter._processor, target_obs, state_dim=adapter.cfg.state_dim),
        noise=_make_noise(sched, batch_size=1),
    )

    batch_request = None
    if args.batch_size > 1:
        if (
            args.filler == "distinct"
            and len(samples) < args.batch_size
            and args.calibration_source == "file"
        ):
            print(
                f"note: batch-size={args.batch_size} > {len(samples)} stored samples; "
                "cycling fillers from the calibration file."
            )
        obs_list = _batch_obs_list(
            samples,
            args.sample_index,
            args.batch_size,
            args.filler,
        )
        batch_request = replace(
            _build_batch_request(adapter._processor, obs_list, state_dim=adapter.cfg.state_dim),
            noise=_make_noise(sched, batch_size=args.batch_size),
        )

    print("\n=== lang bucket ===")
    _report_lang_layout(sched, "bs=1", single_request)
    if batch_request is not None:
        _report_lang_layout(
            sched,
            f"bs={args.batch_size} ({args.filler})",
            batch_request,
        )

    print("\n=== warmup ===")
    for _ in range(args.warmup):
        adapter._engine.step(single_request)
    if batch_request is not None:
        for _ in range(args.warmup):
            adapter._engine.step(batch_request)

    print("\n[stage] bs=1 repeated engine.step (raw actions)")
    outs: list[torch.Tensor] = []
    for i in range(args.runs):
        torch.cuda.synchronize()
        out = adapter._engine.step(single_request).detach()
        outs.append(out.float().cpu())
        print(f"  run {i}: shape={tuple(out.shape)}")

    ref = outs[0]
    bs1_repeat_diff = False
    for i in range(1, args.runs):
        max_d = _report_diff(f"run0 vs run{i}", ref, outs[i])
        bs1_repeat_diff = bs1_repeat_diff or max_d > 0

    bs1_vs_batch_diff = False
    batch_repeat_diff = False
    if batch_request is not None:
        print(
            f"\n[stage] bs=1 vs bs={args.batch_size} "
            f"(filler={args.filler}, target at slot 0, same inference_seed noise)"
        )
        torch.cuda.synchronize()
        bs1_out = adapter._engine.step(single_request).detach().float().cpu()[0]
        torch.cuda.synchronize()
        batch_out = adapter._engine.step(batch_request).detach().float().cpu()[0]
        bs1_vs_batch_diff = _report_diff("bs=1 vs bs=N slot0", bs1_out, batch_out) > 0

        print(f"\n[stage] bs={args.batch_size} repeated engine.step (slot 0 raw actions)")
        batch_outs: list[torch.Tensor] = []
        for i in range(args.runs):
            torch.cuda.synchronize()
            out = adapter._engine.step(batch_request).detach().float().cpu()[0]
            batch_outs.append(out)
            print(f"  run {i}: shape={tuple(out.shape)}")
        batch_ref = batch_outs[0]
        for i in range(1, args.runs):
            max_d = _report_diff(f"run0 vs run{i}", batch_ref, batch_outs[i])
            batch_repeat_diff = batch_repeat_diff or max_d > 0

    print("\n[stage] postprocessed actions (env-facing, bs=1)")
    posts: list[np.ndarray] = []
    for _ in range(args.runs):
        raw_out = adapter._engine.step(single_request).detach()
        posts.append(
            adapter._processor.postprocess(raw_out).to(torch.float32).cpu().numpy()
        )
    gripper_flips = 0
    for i in range(1, len(posts)):
        d = np.abs(posts[i] - posts[0])
        print(f"  run0 vs run{i}: max={d.max():.6e} min={d.min():.6e}")
        g0 = posts[0][..., -1] < 0
        g1 = posts[i][..., -1] < 0
        flips = int(np.sum(g0 != g1))
        gripper_flips += flips
        if flips:
            print(f"    gripper sign would flip @ threshold 0: {flips} values")

    mixed_ab_diff = False
    mixed_short_diff = False
    mixed_vs_bs1_diff = False
    if args.batch_size > 1:
        bs = args.batch_size
        short_bs = max(2, bs - 3)
        print(
            f"\n[stage] bs={bs} mixed companions "
            f"(target fixed at slot 0; run A/B use disjoint fillers, run C uses bs={short_bs})"
        )
        n = len(samples)
        fillers_a = [(args.sample_index + 1 + i) % n for i in range(bs - 1)]
        fillers_b = [(args.sample_index + bs + i) % n for i in range(bs - 1)]
        if set(fillers_a) & set(fillers_b):
            print(
                f"  note: only {n} samples available; filler sets A/B overlap "
                "(cannot make companions fully disjoint)."
            )

        def _mk_mixed_request(filler_idx: list[int]):
            obs_list = [samples[args.sample_index]] + [samples[j] for j in filler_idx]
            return replace(
                _build_batch_request(
                    adapter._processor, obs_list, state_dim=adapter.cfg.state_dim
                ),
                noise=_make_noise(sched, batch_size=len(obs_list)),
            )

        mixed_requests = [
            ("A", _mk_mixed_request(fillers_a)),
            ("B", _mk_mixed_request(fillers_b)),
            ("C (short)", _mk_mixed_request(fillers_a[: short_bs - 1])),
        ]
        for name, req in mixed_requests:
            _report_lang_layout(sched, f"run {name} bs={req.lang_lens.numel()}", req)
            for _ in range(args.warmup):
                adapter._engine.step(req)

        mixed_outs: dict[str, torch.Tensor] = {}
        for name, req in mixed_requests:
            torch.cuda.synchronize()
            out = adapter._engine.step(req).detach().float().cpu()[0]
            mixed_outs[name] = out
            print(f"  run {name}: bs={req.lang_lens.numel()}  slot0 shape={tuple(out.shape)}")

        torch.cuda.synchronize()
        bs1_ref = adapter._engine.step(single_request).detach().float().cpu()[0]
        mixed_ab_diff = _report_diff("run A vs run B (slot 0)", mixed_outs["A"], mixed_outs["B"]) > 0
        mixed_short_diff = (
            _report_diff("run A vs run C short (slot 0)", mixed_outs["A"], mixed_outs["C (short)"])
            > 0
        )
        mixed_vs_bs1_diff = _report_diff("bs=1 vs run A (slot 0)", bs1_ref, mixed_outs["A"]) > 0

    slot_diff = False
    if args.batch_size > 1:
        bs = args.batch_size
        slots = sorted({0, 1, bs // 2, bs - 1})
        print(
            f"\n[stage] bs={bs} slot sensitivity "
            f"(same batch content as run A; target moved to slots {slots})"
        )

        def _mk_slot_request(slot: int):
            # Same 8 obs as run A, but the target sits at `slot` and the
            # fillers keep their relative order in the remaining slots.
            obs_list = [samples[j] for j in fillers_a]
            obs_list.insert(slot, samples[args.sample_index])
            return replace(
                _build_batch_request(
                    adapter._processor, obs_list, state_dim=adapter.cfg.state_dim
                ),
                noise=_make_noise(sched, batch_size=len(obs_list)),
            )

        slot_requests = [(s, _mk_slot_request(s)) for s in slots]
        for s, req in slot_requests:
            _report_lang_layout(sched, f"target@slot{s}", req)
            for _ in range(args.warmup):
                adapter._engine.step(req)

        slot_outs: dict[int, torch.Tensor] = {}
        for s, req in slot_requests:
            torch.cuda.synchronize()
            out = adapter._engine.step(req).detach().float().cpu()[s]
            slot_outs[s] = out
            print(f"  target@slot{s}: shape={tuple(out.shape)}")

        slot_ref = slot_outs[slots[0]]
        for s in slots[1:]:
            max_d = _report_diff(f"slot{slots[0]} vs slot{s} (target row)", slot_ref, slot_outs[s])
            slot_diff = slot_diff or max_d > 0

    print("\n=== summary ===")
    if not bs1_repeat_diff:
        print("bs=1 repeat: identical across runs")
    else:
        print("bs=1 repeat: DIFFERS across runs")
    if batch_request is not None:
        if not bs1_vs_batch_diff:
            print(f"bs=1 vs bs={args.batch_size}: slot-0 output matches bs=1")
        else:
            print(f"bs=1 vs bs={args.batch_size}: slot-0 output DIFFERS from bs=1")
        if not batch_repeat_diff:
            print(f"bs={args.batch_size} repeat: slot-0 identical across runs")
        else:
            print(f"bs={args.batch_size} repeat: slot-0 DIFFERS across runs")
        if not mixed_ab_diff:
            print("mixed companions A vs B: slot-0 identical despite different fillers")
        else:
            print("mixed companions A vs B: slot-0 DIFFERS when fillers change")
        if not mixed_short_diff:
            print("mixed companions A vs C (short batch): slot-0 identical")
        else:
            print("mixed companions A vs C (short batch): slot-0 DIFFERS")
        if not mixed_vs_bs1_diff:
            print("mixed companions vs bs=1: slot-0 identical")
        else:
            print("mixed companions vs bs=1: slot-0 DIFFERS from bs=1")
        if not slot_diff:
            print("slot sensitivity: target output identical at every slot position")
        else:
            print("slot sensitivity: target output DIFFERS across slot positions")
    if gripper_flips:
        print("gripper threshold can amplify tiny action noise in LIBERO eval")

    adapter._engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
