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

    # All quant target linear layers (flat under --output-dir):
    uv run python tools/plot_weight_row_distribution.py \\
        --checkpoint /data/share/pi05_libero_finetuned_v044 \\
        --all-layers \\
        --calibration-data ../calibration_data/libero_object_16_7.npz \\
        --fit-samples 10 \\
        --eval-samples 4 \\
        --perm-score activation \\
        --svd-source activation \\
        --output-dir tools/img/row_dist_all_layers
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
    matches = _list_layers(model, config, layer_regex)
    if len(matches) > 1:
        names = [n for n, _, _ in matches]
        raise SystemExit(
            f"--layer-regex matched {len(matches)} layers; narrow the pattern.\n"
            f"Matches: {names[:10]}{'...' if len(names) > 10 else ''}"
        )
    return matches[0]


def _list_layers(
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


def _collect_layer_activations(
    adapter,
    model: torch.nn.Module,
    layers: list[tuple[str, str, torch.nn.Module]],
    *,
    num_samples: int,
    start: int = 0,
) -> dict[str, torch.Tensor]:
    """Capture linear inputs across calibration samples (one forward pass for all layers)."""
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}.")
    if start < 0:
        raise ValueError(f"start must be non-negative, got {start}.")

    stores: dict[str, list[torch.Tensor]] = {name: [] for name, _, _ in layers}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    for name, _, module in layers:

        def make_hook(layer_name: str):
            def hook(_mod, inputs):
                if not inputs:
                    return
                x = inputs[0]
                if torch.is_tensor(x):
                    stores[layer_name].append(
                        x.reshape(-1, x.shape[-1]).detach().cpu()
                    )

            return hook

        handles.append(module.register_forward_pre_hook(make_hook(name)))

    try:
        def step_cb(step):
            pass

        total = start + num_samples
        for i, batch in enumerate(adapter.iter_calibration_batches(total)):
            if i < start:
                continue
            adapter.forward_for_calibration(model, batch, step_callback=step_cb)
    finally:
        for handle in handles:
            handle.remove()

    out: dict[str, torch.Tensor] = {}
    for name, _, _ in layers:
        chunks = stores[name]
        if not chunks:
            raise RuntimeError(
                f"No activations captured for {name}; did the layer run during calibration?"
            )
        out[name] = torch.cat(chunks, dim=0).to(torch.float32)
    return out


def _process_layer(
    args,
    adapter,
    model: torch.nn.Module,
    layer_name: str,
    module: torch.nn.Module,
    *,
    weight_output: Path | None,
    activation_output: Path | None,
    cached_fit_activations: torch.Tensor | None = None,
    cached_eval_activations: torch.Tensor | None = None,
) -> None:
    weight = getattr(module, "weight").detach().to(torch.float32)
    activation_tokens: torch.Tensor | None = None
    activation_inputs: torch.Tensor | None = None

    needs_cal = _needs_activation_calibration(args.perm_score, args.svd_source)
    needs_eval = activation_output is not None

    if needs_cal and needs_eval and args.calibration_data is not None:
        _validate_fit_eval_split(
            args.calibration_data,
            fit_samples=args.fit_samples,
            eval_samples=args.eval_samples,
        )

    if needs_cal:
        if args.calibration_data is None:
            raise SystemExit(
                f"--perm-score={args.perm_score} / --svd-source={args.svd_source} "
                "require --calibration-data."
            )
        activation_tokens = cached_fit_activations
        if activation_tokens is None:
            activation_tokens = _collect_layer_activations(
                adapter, model, [(layer_name, "", module)],
                num_samples=args.fit_samples, start=0,
            )[layer_name]
        logger.info(
            "%s fit calibration (samples 0..%d): %d tokens x %d channels.",
            layer_name,
            args.fit_samples - 1,
            activation_tokens.shape[0],
            activation_tokens.shape[1],
        )
    elif (
        args.calibration_data
        and weight_output is not None
        and activation_output is None
    ):
        logger.warning(
            "%s: calibration file provided but perm_score=weight and svd_source=weight; "
            "rotation fit does not use calibration.",
            layer_name,
        )

    if needs_eval:
        if args.calibration_data is None:
            raise SystemExit("Activation plots require --calibration-data.")
        eval_start = args.fit_samples if needs_cal else 0
        activation_inputs = cached_eval_activations
        if activation_inputs is None:
            activation_inputs = _collect_layer_activations(
                adapter, model, [(layer_name, "", module)],
                num_samples=args.eval_samples, start=eval_start,
            )[layer_name]
        logger.info(
            "%s eval calibration (samples %d..%d): %d tokens x %d channels.",
            layer_name,
            eval_start,
            eval_start + args.eval_samples - 1,
            activation_inputs.shape[0],
            activation_inputs.shape[1],
        )


    if weight_output is not None:
        distributions = compare_weight_variants(
            weight,
            block_size=args.block_size,
            metric=args.metric,
            perm_score=args.perm_score,
            svd_source=args.svd_source,
            activation_tokens=activation_tokens,
        )
        weight_output.parent.mkdir(parents=True, exist_ok=True)
        plot_weight_distributions(
            distributions,
            output_path=str(weight_output),
            highlight_top_k=args.highlight_top_k,
        )
        print(f"Layer: {layer_name} (weight)")
        for key in VARIANT_ORDER:
            d = distributions[key]
            print(f"  {key:14s}  max/min={d.max_min_ratio:6.1f}x  sigma={d.std:.2f}")
        print(f"Wrote {weight_output}")

    if activation_output is not None:
        assert activation_inputs is not None
        act_distributions = compare_activation_variants(
            activation_inputs,
            weight,
            block_size=args.block_size,
            metric=args.metric,
            perm_score=args.perm_score,
            svd_source=args.svd_source,
        )
        activation_output.parent.mkdir(parents=True, exist_ok=True)
        plot_activation_distributions(
            act_distributions,
            output_path=str(activation_output),
            highlight_top_k=args.highlight_top_k,
        )
        print(f"Layer: {layer_name} (activation, {activation_inputs.shape[0]} tokens)")
        for key in VARIANT_ORDER:
            d = act_distributions[key]
            print(f"  {key:14s}  max/min={d.max_min_ratio:6.1f}x  sigma={d.std:.2f}")
        print(f"Wrote {activation_output}")


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
        help="Regex for one layer, or an optional filter with --all-layers.",
    )
    src.add_argument(
        "--all-layers",
        action="store_true",
        help="Plot every quant target linear layer into --output-dir.",
    )

    out = parser.add_argument_group("output")
    out.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for --all-layers (auto-named PNGs per layer).",
    )
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
    if args.all_layers:
        if args.synthetic:
            parser.error("--all-layers requires --checkpoint.")
        if args.output_dir is None:
            parser.error("--all-layers requires --output-dir.")
        if args.weight_output or args.activation_output:
            parser.error(
                "--all-layers uses --output-dir; omit --weight-output / --activation-output."
            )
    elif args.weight_output is None and args.activation_output is None:
        parser.error("--weight-output and/or --activation-output is required.")

    if args.synthetic:
        weight = _make_synthetic_weight()
        layer_label = "synthetic"
        activation_tokens = None
        activation_inputs = None
        if _needs_activation_calibration(args.perm_score, args.svd_source):
            activation_tokens = _make_synthetic_activation(inp=weight.shape[1])
        if args.activation_output is not None:
            activation_inputs = _make_synthetic_activation(inp=weight.shape[1])

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

    assert args.checkpoint is not None
    config = QVLAConfig.pi05_default()
    adapter_kwargs: dict = {"checkpoint_path": args.checkpoint}
    if args.calibration_data:
        adapter_kwargs["calibration_source"] = "file"
        adapter_kwargs["calibration_data_path"] = args.calibration_data
    adapter = get_adapter("pi05", **adapter_kwargs)
    model = adapter.build_model()
    adapter.warmup_for_calibration(model)

    if args.all_layers:
        layers = _list_layers(model, config, args.layer_regex)
        needs_cal = _needs_activation_calibration(args.perm_score, args.svd_source)
        needs_eval = args.calibration_data is not None
        if needs_cal and needs_eval:
            _validate_fit_eval_split(
                args.calibration_data,
                fit_samples=args.fit_samples,
                eval_samples=args.eval_samples,
            )

        fit_cache: dict[str, torch.Tensor] = {}
        eval_cache: dict[str, torch.Tensor] = {}
        if needs_cal:
            print(
                f"Collecting fit activations for {len(layers)} layers "
                f"({args.fit_samples} calibration samples)..."
            )
            fit_cache = _collect_layer_activations(
                adapter,
                model,
                layers,
                num_samples=args.fit_samples,
                start=0,
            )
        if needs_eval:
            eval_start = args.fit_samples if needs_cal else 0
            print(
                f"Collecting eval activations for {len(layers)} layers "
                f"({args.eval_samples} calibration samples, start={eval_start})..."
            )
            eval_cache = _collect_layer_activations(
                adapter,
                model,
                layers,
                num_samples=args.eval_samples,
                start=eval_start,
            )

        print(f"Plotting {len(layers)} layers into {args.output_dir}")
        for i, (layer_name, scope, module) in enumerate(layers, 1):
            stem = _layer_file_stem(layer_name, scope)
            print(f"\n[{i}/{len(layers)}] {layer_name}")
            _process_layer(
                args,
                adapter,
                model,
                layer_name,
                module,
                weight_output=args.output_dir / f"{stem}_weight_dist.png",
                activation_output=(
                    args.output_dir / f"{stem}_act_dist.png"
                    if args.calibration_data is not None
                    else None
                ),
                cached_fit_activations=fit_cache.get(layer_name),
                cached_eval_activations=eval_cache.get(layer_name),
            )
        return 0

    if not args.layer_regex:
        parser.error("--layer-regex is required with --checkpoint (unless --all-layers).")
    layer_name, _scope, module = _resolve_layer(model, config, args.layer_regex)
    _process_layer(
        args,
        adapter,
        model,
        layer_name,
        module,
        weight_output=args.weight_output,
        activation_output=args.activation_output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
