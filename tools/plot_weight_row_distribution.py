#!/usr/bin/env python
"""Plot weight row-norm distributions under DuQuant-style rotations.

Examples::

    # Synthetic sanity check (no checkpoint):
    uv run python tools/plot_weight_row_distribution.py \\
        --synthetic --weight-output /tmp/row_dist.png

    # Real DiT layer (pi0.5 uses expert_stack + qkv_proj, not dit/attn/q_proj):
    uv run python tools/plot_weight_row_distribution.py \\
        --checkpoint /data/share/pi05-libero \\
        --layer-regex 'expert_stack\\.layers\\.0\\.qkv_proj$' \\
        --block-size 64 \\
        --weight-output ./figures/dit_q_row_dist.png

    # Activation input-channel distribution (same nine pipelines as weight):
    uv run python tools/plot_weight_row_distribution.py \\
        --checkpoint /data/share/pi05-libero \\
        --layer-regex 'paligemma_lm\\.layers\\.0\\.qkv_proj$' \\
        --calibration-data ../calibration_data/libero_object_16_7.npz \\
        --eval-samples 10 \\
        --activation-output ./figures/qkv_act_dist.png
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "tools"))

from qvla.adapters import get_adapter  # noqa: E402
from qvla.config import QVLAConfig  # noqa: E402
from qvla.runtime import list_target_modules  # noqa: E402
from weight_row_distribution import (  # noqa: E402
    VARIANT_ORDER,
    compare_activation_variants,
    compare_weight_variants,
    plot_activation_distributions,
    plot_weight_distributions,
)


logger = logging.getLogger(__name__)


def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING - 10 * min(verbosity, 2)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _make_synthetic_weight(out: int = 33, inp: int = 128) -> torch.Tensor:
    """Skewed weight matrix resembling outlier-heavy layers in the reference figure."""
    torch.manual_seed(0)
    w = torch.randn(out, inp) * 0.3
    w[:5] *= 8.0
    return w


def _make_synthetic_activation(num_tokens: int = 256, inp: int = 128) -> torch.Tensor:
    torch.manual_seed(1)
    x = torch.randn(num_tokens, inp) * 0.3
    x[:, :5] *= 8.0
    return x


def _resolve_layer(
    model: torch.nn.Module,
    config: QVLAConfig,
    layer_regex: str,
) -> tuple[str, str, torch.nn.Module]:
    pattern = re.compile(layer_regex)
    matches = [
        (name, scope, mod)
        for name, scope, mod in list_target_modules(model, config)
        if pattern.search(name)
    ]
    if not matches:
        raise SystemExit(f"No layer matched --layer-regex={layer_regex!r}.")
    if len(matches) > 1:
        names = [n for n, _, _ in matches]
        raise SystemExit(
            f"--layer-regex matched {len(matches)} layers; narrow the pattern.\n"
            f"Matches: {names[:10]}{'...' if len(names) > 10 else ''}"
        )
    name, scope, mod = matches[0]
    return name, scope, mod


def _needs_activation_calibration(perm_score: str, svd_source: str) -> bool:
    return perm_score in ("activation", "activation_weight") or svd_source == "activation"


def _calibration_sample_count(path: Path) -> int:
    with np.load(path, allow_pickle=True) as data:
        meta = json.loads(str(data["meta_json"]))
        return int(meta.get("num_samples", data["images"].shape[0]))


def _validate_fit_eval_split(
    calibration_data: Path,
    *,
    fit_samples: int,
    eval_samples: int,
) -> None:
    stored = _calibration_sample_count(calibration_data)
    needed = fit_samples + eval_samples
    if needed > stored:
        raise SystemExit(
            f"fit-samples ({fit_samples}) + eval-samples ({eval_samples}) = {needed} "
            f"exceeds {stored} samples in {calibration_data}."
        )


def _collect_layer_input_activations(
    adapter,
    model: torch.nn.Module,
    module: torch.nn.Module,
    *,
    num_samples: int,
    start: int = 0,
) -> torch.Tensor:
    """Capture pre-rotation inputs to one linear layer across calibration samples."""
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}.")
    if start < 0:
        raise ValueError(f"start must be non-negative, got {start}.")

    chunks: list[torch.Tensor] = []

    def hook(_mod, inputs):
        if not inputs:
            return
        x = inputs[0]
        if torch.is_tensor(x):
            chunks.append(x.reshape(-1, x.shape[-1]).detach().cpu())

    handle = module.register_forward_pre_hook(hook)
    try:
        def step_cb(step):
            pass

        total = start + num_samples
        for i, batch in enumerate(adapter.iter_calibration_batches(total)):
            if i < start:
                continue
            adapter.forward_for_calibration(model, batch, step_callback=step_cb)
    finally:
        handle.remove()

    if not chunks:
        raise RuntimeError("No activations captured; did the target layer run during calibration?")
    return torch.cat(chunks, dim=0).to(torch.float32)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument(
        "--perm-score",
        choices=("weight", "activation", "activation_weight"),
        default="weight",
    )
    parser.add_argument(
        "--svd-source",
        choices=("weight", "activation"),
        default="weight",
    )
    parser.add_argument(
        "--metric",
        choices=("max_abs", "l2"),
        default="l2",
        help="Input-channel metric: max_abs (default) or column L2 norm.",
    )
    parser.add_argument("--highlight-top-k", type=int, default=0)
    parser.add_argument("-v", "--verbose", action="count", default=0)

    src = parser.add_argument_group("weight source")
    src.add_argument("--synthetic", action="store_true", help="Use a synthetic skewed matrix.")
    src.add_argument("--checkpoint", type=Path, help="pi0.5 checkpoint for real weights.")
    src.add_argument(
        "--layer-regex",
        help="Regex selecting exactly one target Linear (required with --checkpoint).",
    )

    out = parser.add_argument_group("output")
    out.add_argument(
        "--weight-output",
        type=Path,
        help="PNG path for weight input-channel distribution (9 pipelines).",
    )
    out.add_argument(
        "--activation-output",
        type=Path,
        help="PNG path for activation input-channel distribution (9 pipelines).",
    )

    cal = parser.add_argument_group(
        "calibration (act-based perm/SVD fit, or --activation-output capture)"
    )
    cal.add_argument("--calibration-data", type=Path, help=".npz calibration file.")
    cal.add_argument(
        "--fit-samples",
        type=int,
        default=10,
        help="First N calibration samples for perm/SVD fit (indices 0..N-1).",
    )
    cal.add_argument(
        "--eval-samples",
        type=int,
        default=4,
        help=(
            "Calibration samples for activation plot. When fit samples are also "
            "used, eval takes the next M samples (indices N..N+M-1), non-overlapping."
        ),
    )

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    if not args.synthetic and not args.checkpoint:
        parser.error("Provide --synthetic or --checkpoint.")
    if args.weight_output is None and args.activation_output is None:
        parser.error("--weight-output and/or --activation-output is required.")

    activation_tokens: torch.Tensor | None = None
    activation_inputs: torch.Tensor | None = None
    adapter = None
    model = None
    weight: torch.Tensor | None = None
    layer_label = "synthetic"

    if args.synthetic:
        weight = _make_synthetic_weight()
        if _needs_activation_calibration(args.perm_score, args.svd_source):
            activation_tokens = _make_synthetic_activation(inp=weight.shape[1])
        if args.activation_output is not None:
            activation_inputs = _make_synthetic_activation(inp=weight.shape[1])
    elif args.checkpoint:
        if not args.layer_regex:
            parser.error("--layer-regex is required with --checkpoint.")
        config = QVLAConfig.pi05_default()
        adapter_kwargs: dict = {"checkpoint_path": args.checkpoint}
        if args.calibration_data:
            adapter_kwargs["calibration_source"] = "file"
            adapter_kwargs["calibration_data_path"] = args.calibration_data
        adapter = get_adapter("pi05", **adapter_kwargs)
        model = adapter.build_model()
        adapter.warmup_for_calibration(model)

        layer_name, scope, module = _resolve_layer(model, config, args.layer_regex)
        weight = getattr(module, "weight").detach().to(torch.float32)
        layer_label = layer_name

        needs_cal = _needs_activation_calibration(args.perm_score, args.svd_source)
        needs_eval = args.activation_output is not None

        if needs_cal and needs_eval and args.calibration_data is not None:
            _validate_fit_eval_split(
                args.calibration_data,
                fit_samples=args.fit_samples,
                eval_samples=args.eval_samples,
            )

        if needs_cal:
            if args.calibration_data is None:
                parser.error(
                    f"--perm-score={args.perm_score} / --svd-source={args.svd_source} "
                    "require --calibration-data."
                )
            activation_tokens = _collect_layer_input_activations(
                adapter,
                model,
                module,
                num_samples=args.fit_samples,
                start=0,
            )
            logger.info(
                "Fit calibration (samples 0..%d): %d tokens x %d channels.",
                args.fit_samples - 1,
                activation_tokens.shape[0],
                activation_tokens.shape[1],
            )
        elif (
            args.calibration_data
            and args.weight_output is not None
            and args.activation_output is None
        ):
            logger.warning(
                "Calibration file provided but perm_score=weight and svd_source=weight; "
                "rotation fit does not use calibration — weight plot depends on W only."
            )

        if needs_eval:
            if args.calibration_data is None:
                parser.error("--activation-output requires --calibration-data.")
            eval_start = args.fit_samples if needs_cal else 0
            activation_inputs = _collect_layer_input_activations(
                adapter,
                model,
                module,
                num_samples=args.eval_samples,
                start=eval_start,
            )
            logger.info(
                "Eval calibration (samples %d..%d): %d tokens x %d channels.",
                eval_start,
                eval_start + args.eval_samples - 1,
                activation_inputs.shape[0],
                activation_inputs.shape[1],
            )
    else:
        parser.error("Provide --synthetic or --checkpoint.")

    assert weight is not None

    if args.weight_output is not None:
        distributions = compare_weight_variants(
            weight,
            block_size=args.block_size,
            metric=args.metric,
            perm_score=args.perm_score,
            svd_source=args.svd_source,
            activation_tokens=activation_tokens,
        )

        args.weight_output.parent.mkdir(parents=True, exist_ok=True)
        plot_weight_distributions(
            distributions,
            output_path=str(args.weight_output),
            highlight_top_k=args.highlight_top_k,
        )

        print(f"Layer: {layer_label} (weight)")
        for key in VARIANT_ORDER:
            d = distributions[key]
            print(f"  {key:14s}  max/min={d.max_min_ratio:6.1f}x  sigma={d.std:.2f}")
        print(f"Wrote {args.weight_output}")

    if args.activation_output is not None:
        assert activation_inputs is not None
        act_distributions = compare_activation_variants(
            activation_inputs,
            weight,
            block_size=args.block_size,
            metric=args.metric,
            perm_score=args.perm_score,
            svd_source=args.svd_source,
        )

        args.activation_output.parent.mkdir(parents=True, exist_ok=True)
        plot_activation_distributions(
            act_distributions,
            output_path=str(args.activation_output),
            highlight_top_k=args.highlight_top_k,
        )

        print(f"Layer: {layer_label} (activation, {activation_inputs.shape[0]} tokens)")
        for key in VARIANT_ORDER:
            d = act_distributions[key]
            print(f"  {key:14s}  max/min={d.max_min_ratio:6.1f}x  sigma={d.std:.2f}")
        print(f"Wrote {args.activation_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
