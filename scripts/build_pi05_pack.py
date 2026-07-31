#!/usr/bin/env python
"""Build and debug pi0.5 QVLA W4A4 packs.

Subcommands::

    build          Build a W4A4 pack (default when omitted)
    debug-regex    Dry-run layer include/exclude regex matching

Examples::

    # Standard W4A4 pack
    uv run python scripts/build_pi05_pack.py \\
        --checkpoint /data/share/pi05-libero \\
        --output ./packs/pi05_libero_W4A4.pt \\
        --calibration-source file \\
        --calibration-data ../calibration_data/libero_object_16_7.npz \\
        -vv

    # Fisher-driven perm on DiT (runs a Fisher pass during the build)
    uv run python scripts/build_pi05_pack.py \\
        --checkpoint /data/share/pi05-libero \\
        --output ./packs/pi05_libero_policy_W4A4.pt \\
        --calibration-source file \\
        --calibration-data ../calibration_data/libero_object_16_7.npz \\
        --dit-pipeline perm,svd,hadamard \\
        --dit-perm-score fisher \\
        --fisher-num-samples 4 \\
        --fisher-batch-size 4 \\
        --fisher-step-aggregation uniform \\
        --calibration-step-aggregation uniform \\
        -vv

    uv run python scripts/build_pi05_pack.py debug-regex \\
        --checkpoint /data/share/pi05-libero
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

# Allow running from a fresh checkout without `pip install -e .`.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from qvla.adapters import get_adapter  # noqa: E402
from qvla.build import build_pack  # noqa: E402
from qvla.config import (  # noqa: E402
    ActPercentileMode,
    ActScaleGranularity,
    ActScaleMode,
    DEFAULT_PIPELINE,
    QVLAConfig,
    PermScore,
    SvdSource,
    WeightQuantizer,
)
from qvla.core.rotation import parse_pipeline_string  # noqa: E402
from qvla.runtime import list_target_modules  # noqa: E402

_SUBCOMMANDS = frozenset({"build", "debug-regex"})
_WEIGHT_QUANT_CHOICES: tuple[WeightQuantizer, ...] = ("gptq", "rtn", "rtn_residual")
_ACT_SCALE_CHOICES: tuple[ActScaleMode, ...] = ("per_step", "static", "dynamic")
_ACT_SCALE_GRANULARITY_CHOICES: tuple[ActScaleGranularity, ...] = (
    "per_channel",
    "per_token",
)
_ACT_PERCENTILE_MODE_CHOICES: tuple[ActPercentileMode, ...] = (
    "inner",
    "cross",
)
_FISHER_STEP_AGG_CHOICES = (
    "uniform", "max", "late_mean", "very_late_mean", "weighted_linear"
)
_FISHER_METHOD_CHOICES = ("exact", "hutchinson")
_CALIB_STEP_AGG_CHOICES = (
    "uniform", "late_mean", "very_late_mean", "weighted_linear"
)
_PERM_SCORE_CHOICES: tuple[PermScore, ...] = (
    "weight",
    "activation",
    "activation_weight",
    "fisher",
)
_SVD_SOURCE_CHOICES: tuple[SvdSource, ...] = ("weight", "activation")


def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING - 10 * min(verbosity, 2)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _normalize_argv(argv: Sequence[str] | None) -> list[str]:
    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)
    if not argv or argv in (["-h"], ["--help"]):
        return argv
    if argv[0] not in _SUBCOMMANDS:
        return ["build", *argv]
    return argv


def _scope_overrides(prefix: str, args: argparse.Namespace) -> dict:
    mapping = {
        f"{prefix}_quant": "weight_quantizer",
        f"{prefix}_pipeline": "pipeline",
        f"{prefix}_weight_bits": "weight_bits",
        f"{prefix}_group_size": "group_size",
        f"{prefix}_act_bits": "act_bits",
        f"{prefix}_act_scale_mode": "act_scale_mode",
        f"{prefix}_act_scale_granularity": "act_scale_granularity",
        f"{prefix}_act_percentile": "act_percentile",
        f"{prefix}_act_percentile_mode": "act_percentile_mode",
        f"{prefix}_rotation_block_size": "rotation_block_size",
        f"{prefix}_perm_score": "perm_score",
        f"{prefix}_svd_source": "svd_source",
        f"{prefix}_gptq_damp_percent": "gptq_damp_percent",
        f"{prefix}_gptq_block_size": "gptq_block_size",
        f"{prefix}_fisher_gptq": "fisher_gptq",
        f"{prefix}_include_regex": "include_regex",
        f"{prefix}_exclude_regex": "exclude_regex",
    }
    updates: dict = {}
    for arg_name, field_name in mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            updates[field_name] = value
    return updates


def build_config_from_args(args: argparse.Namespace) -> QVLAConfig:
    if args.config_json:
        config = QVLAConfig.from_json(args.config_json)
    else:
        config = QVLAConfig.pi05_default()

    llm = config.llm
    dit = config.dit

    if getattr(args, "weight_bits", None) is not None:
        llm = replace(llm, weight_bits=args.weight_bits)
        dit = replace(dit, weight_bits=args.weight_bits)
    if getattr(args, "act_bits", None) is not None:
        llm = replace(llm, act_bits=args.act_bits)
        dit = replace(dit, act_bits=args.act_bits)

    llm_updates = _scope_overrides("llm", args)
    if llm_updates:
        llm = replace(llm, **llm_updates)
    dit_updates = _scope_overrides("dit", args)
    if dit_updates:
        dit = replace(dit, **dit_updates)

    top: dict = {"llm": llm, "dit": dit}
    if getattr(args, "skip_incompatible", None) is not None:
        top["skip_incompatible"] = args.skip_incompatible
    if getattr(args, "fisher_num_samples", None) is not None:
        top["fisher_num_samples"] = args.fisher_num_samples
    if getattr(args, "fisher_step_aggregation", None) is not None:
        top["fisher_step_aggregation"] = args.fisher_step_aggregation
    if getattr(args, "calibration_step_aggregation", None) is not None:
        top["calibration_step_aggregation"] = args.calibration_step_aggregation
    if getattr(args, "fisher_action_timestep", None) is not None:
        top["fisher_action_timestep"] = args.fisher_action_timestep
    if getattr(args, "fisher_method", None) is not None:
        top["fisher_method"] = args.fisher_method
    if getattr(args, "fisher_hutchinson_probes", None) is not None:
        top["fisher_hutchinson_probes"] = args.fisher_hutchinson_probes
    if getattr(args, "fisher_batch_size", None) is not None:
        top["fisher_batch_size"] = args.fisher_batch_size
    if getattr(args, "build_seed", None) is not None:
        top["build_seed"] = args.build_seed
    if getattr(args, "calibration_noise_mode", None) is not None:
        top["calibration_noise_mode"] = args.calibration_noise_mode
    if getattr(args, "noise_ensemble_k", None) is not None:
        top["noise_ensemble_k"] = args.noise_ensemble_k
    return config.with_overrides(**top)


def _pack_bitwidth_label(config: QVLAConfig) -> str:
    """``W4A4``-style suffix from per-scope weight/act bit widths."""
    lw, la = config.llm.weight_bits, config.llm.act_bits
    dw, da = config.dit.weight_bits, config.dit.act_bits
    if lw == dw and la == da:
        return f"W{lw}A{la}"
    return f"Wllm{lw}Wditt{dw}Allm{la}Adit{da}"


def _auto_output_path(args: argparse.Namespace, config: QVLAConfig) -> Path:
    if args.output:
        return Path(args.output)
    ckpt = Path(args.checkpoint).resolve()
    out_dir = ckpt.parent / f"{ckpt.name}-packs"
    name = f"{ckpt.name}-{_pack_bitwidth_label(config)}"
    base = QVLAConfig.pi05_default()
    tags: list[str] = []
    skip_tag_keys = frozenset({"weight_bits", "act_bits"})

    def _walk(b: dict, f: dict, prefix: str = "") -> None:
        for key, val in f.items():
            if key in skip_tag_keys:
                continue
            if b.get(key) == val:
                continue
            if isinstance(val, dict) and isinstance(b.get(key), dict):
                _walk(b[key], val, f"{prefix}{key}_")
            else:
                val_part = str(val).replace(".", "p")
                tags.append(f"{prefix}{key}_{val_part}")

    _walk(base.to_dict(), config.to_dict())
    if args.num_samples != 10:
        tags.append(f"ns{args.num_samples}")
    if getattr(args, "noise_ensemble_k", None) not in (None, 1):
        tags.append(f"nk{args.noise_ensemble_k}")
    if args.calibration_source != "synthetic":
        tags.append(f"cal{args.calibration_source}")
    if tags:
        name += "-" + "-".join(tags)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{name}.pt"


def build_adapter_kwargs(args: argparse.Namespace) -> dict:
    kwargs = {
        "checkpoint_path": args.checkpoint,
        "device": args.device,
        "params_dtype": args.params_dtype,
        "calibration_source": args.calibration_source,
    }
    if args.calibration_source == "file":
        if not args.calibration_data:
            raise ValueError(
                "--calibration-data is required when --calibration-source=file"
            )
        kwargs["calibration_data_path"] = args.calibration_data
    return kwargs


def _add_scope_args(
    parser: argparse.ArgumentParser,
    prefix: str,
    *,
    help_prefix: str,
) -> None:
    g = parser.add_argument_group(f"{help_prefix} scope")
    g.add_argument(
        f"--{prefix}-quant",
        choices=_WEIGHT_QUANT_CHOICES,
        default=None,
        dest=f"{prefix}_quant",
        help=f"{help_prefix} weight quantizer.",
    )
    g.add_argument(
        f"--{prefix}-pipeline",
        type=parse_pipeline_string,
        default=None,
        dest=f"{prefix}_pipeline",
        metavar="STEPS",
        help=(
            f"{help_prefix} input transform pipeline (comma-separated). "
            "Steps: perm, svd, hadamard, random_hadamard. Use 'none' to disable. "
            "Default: perm,svd,hadamard."
        ),
    )
    g.add_argument(
        f"--{prefix}-weight-bits",
        type=int,
        default=None,
        dest=f"{prefix}_weight_bits",
    )
    g.add_argument(
        f"--{prefix}-group-size",
        type=int,
        default=None,
        dest=f"{prefix}_group_size",
        help="Weight scale group size along K; -1 means per-channel.",
    )
    g.add_argument(
        f"--{prefix}-act-bits",
        type=int,
        default=None,
        dest=f"{prefix}_act_bits",
    )
    g.add_argument(
        f"--{prefix}-act-scale-mode",
        choices=_ACT_SCALE_CHOICES,
        default=None,
        dest=f"{prefix}_act_scale_mode",
    )
    g.add_argument(
        f"--{prefix}-act-scale-granularity",
        choices=_ACT_SCALE_GRANULARITY_CHOICES,
        default=None,
        dest=f"{prefix}_act_scale_granularity",
        help=(
            "Activation scale granularity for static / per_step: "
            "per_channel (default) or per_token."
        ),
    )
    g.add_argument(
        f"--{prefix}-act-percentile",
        type=float,
        default=None,
        dest=f"{prefix}_act_percentile",
        help="Percentile clip for static / per-step activation scales.",
    )
    g.add_argument(
        f"--{prefix}-act-percentile-mode",
        choices=_ACT_PERCENTILE_MODE_CHOICES,
        default=None,
        dest=f"{prefix}_act_percentile_mode",
        help=(
            "How to apply act_percentile for static / per_step scales: "
            "inner (per-dimension sample percentile) or "
            "cross (global cap on per-dimension maxes)."
        ),
    )
    g.add_argument(
        f"--{prefix}-rotation-block-size",
        type=int,
        default=None,
        dest=f"{prefix}_rotation_block_size",
    )
    g.add_argument(
        f"--{prefix}-perm-score",
        choices=_PERM_SCORE_CHOICES,
        default=None,
        dest=f"{prefix}_perm_score",
        help="Energy for zigzag perm when pipeline includes perm: weight | activation | activation_weight | fisher.",
    )
    g.add_argument(
        f"--{prefix}-svd-source",
        choices=_SVD_SOURCE_CHOICES,
        default=None,
        dest=f"{prefix}_svd_source",
        help="SVD basis: weight (paper) or activation (legacy ablation).",
    )
    g.add_argument(
        f"--{prefix}-gptq-damp-percent",
        type=float,
        default=None,
        dest=f"{prefix}_gptq_damp_percent",
    )
    g.add_argument(
        f"--{prefix}-gptq-block-size",
        type=int,
        default=None,
        dest=f"{prefix}_gptq_block_size",
    )
    g.add_argument(
        f"--{prefix}-fisher-gptq",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest=f"{prefix}_fisher_gptq",
        help=(
            "Weight GPTQ Hessian by Fisher sensitivity so GPTQ protects "
            "action-sensitive input columns. Requires --{}-quant gptq.".format(prefix)
        ),
    )
    g.add_argument(
        f"--{prefix}-include-regex",
        default=None,
        dest=f"{prefix}_include_regex",
        help="Override the layer include regex for this scope.",
    )
    g.add_argument(
        f"--{prefix}-exclude-regex",
        default=None,
        dest=f"{prefix}_exclude_regex",
        help="Override the layer exclude regex for this scope.",
    )


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    shared = parser.add_argument_group("shared quant (both scopes)")
    shared.add_argument(
        "--weight-bits",
        type=int,
        default=None,
        help="Shortcut: set weight_bits on LLM and DiT (16 = no weight quant).",
    )
    shared.add_argument(
        "--act-bits",
        type=int,
        default=None,
        help="Shortcut: set act_bits on LLM and DiT (16 = no activation quant).",
    )
    _add_scope_args(parser, "llm", help_prefix="LLM")
    _add_scope_args(parser, "dit", help_prefix="DiT")
    top = parser.add_argument_group("top-level config")
    top.add_argument(
        "--config-json",
        default=None,
        help="Base QVLAConfig JSON; explicit flags override it.",
    )
    top.add_argument(
        "--skip-incompatible",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip layers whose shapes are incompatible with group_size / rotation.",
    )
    fisher = parser.add_argument_group("Fisher sensitivity (perm_score=fisher)")
    fisher.add_argument(
        "--fisher-num-samples",
        type=int,
        default=None,
        dest="fisher_num_samples",
        help="Calibration samples for the Fisher pass (default: 4).",
    )
    fisher.add_argument(
        "--fisher-step-aggregation",
        choices=_FISHER_STEP_AGG_CHOICES,
        default=None,
        dest="fisher_step_aggregation",
        help="How to aggregate per-step DiT Fisher sensitivity.",
    )
    top.add_argument(
        "--calibration-step-aggregation",
        choices=_CALIB_STEP_AGG_CHOICES,
        default=None,
        dest="calibration_step_aggregation",
        help=(
            "How to aggregate per-step DiT XᵀX (covariance / Hessian). "
            "uniform: all steps equal (default). "
            "late_mean: second half of denoise steps only. "
            "very_late_mean: last fifth of denoise steps only. "
            "weighted_linear: w(s) ∝ (s+1), later steps heavier."
        ),
    )
    fisher.add_argument(
        "--fisher-action-timestep",
        default=None,
        dest="fisher_action_timestep",
        help=(
            "Action-chunk timesteps for the Fisher Jacobian: "
            "'all', one index (e.g. '50' as last for T=50), or comma-separated "
            "indices (e.g. '0,29,50')."
        ),
    )
    fisher.add_argument(
        "--fisher-method",
        choices=_FISHER_METHOD_CHOICES,
        default=None,
        dest="fisher_method",
        help=(
            "Fisher estimator: exact (one backward per action dim) or "
            "hutchinson (random-projection approximation)."
        ),
    )
    fisher.add_argument(
        "--fisher-hutchinson-probes",
        type=int,
        default=None,
        dest="fisher_hutchinson_probes",
        help=(
            "Number of random projections when --fisher-method=hutchinson "
            "(default: 8)."
        ),
    )
    fisher.add_argument(
        "--fisher-batch-size",
        type=int,
        default=None,
        dest="fisher_batch_size",
        help=(
            "Observations per Fisher differentiable forward (default: 4). "
            "Raises the pi0.5 engine max_batch_size to at least this value."
        ),
    )
    top.add_argument(
        "--build-seed",
        type=int,
        default=None,
        dest="build_seed",
        help=(
            "Seed for reproducible random Hadamard and pi0.5 calibration noise "
            "(default: 0)."
        ),
    )
    top.add_argument(
        "--calibration-noise-mode",
        choices=("per_sample", "global"),
        default=None,
        help=(
            "Calibration diffusion-noise policy: per_sample preserves the original "
            "independent-noise behaviour; global matches model-server global mode "
            "when build_seed equals inference_seed (default: per_sample)."
        ),
    )
    top.add_argument(
        "--noise-ensemble-k",
        type=int,
        default=None,
        dest="noise_ensemble_k",
        help=(
            "Diffusion noises used for calibration (default: 1). "
            "Global noise mode requires K=1. "
            "Applied when calibrating DiT layers; LLM-only pipeline passes stay "
            "at K=1. amax uses max, XᵀX / Hessian accumulate across noises."
        ),
    )


def _add_run_args(parser: argparse.ArgumentParser, *, verbose_default: int) -> None:
    run = parser.add_argument_group("run")
    run.add_argument("--checkpoint", required=True)
    run.add_argument("--device", default="cuda")
    run.add_argument("--params-dtype", default="bfloat16")
    run.add_argument("-v", "--verbose", action="count", default=verbose_default)


def _build_build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build-pi05-pack build",
        description="Build a pi0.5 QVLA W4A4 pack.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="help")
    _add_run_args(parser, verbose_default=1)
    run = parser.add_argument_group("run")
    run.add_argument(
        "--output",
        default=None,
        help="Output .pt path (default: auto under <checkpoint>-packs/).",
    )
    run.add_argument("--num-samples", type=int, default=10)

    _add_config_args(parser)

    cal = parser.add_argument_group("calibration")
    cal.add_argument(
        "--calibration-source",
        choices=("synthetic", "file"),
        default="synthetic",
    )
    cal.add_argument(
        "--calibration-data",
        default=None,
        help="Path to a .npz file when --calibration-source=file.",
    )
    return parser


def _build_debug_regex_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build-pi05-pack debug-regex",
        description="Print linear modules matched by the include/exclude regexes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="help")
    _add_run_args(parser, verbose_default=0)
    _add_config_args(parser)
    return parser


def _make_root_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build-pi05-pack",
        description="Build and debug pi0.5 QVLA W4A4 packs.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_build = sub.add_parser(
        "build",
        help="Build a pack from a checkpoint.",
        parents=[_build_build_parser()],
        conflict_handler="resolve",
    )
    p_build.set_defaults(func=run_build)
    p_dbg = sub.add_parser(
        "debug-regex",
        help="Dry-run layer include/exclude regex matching.",
        parents=[_build_debug_regex_parser()],
        conflict_handler="resolve",
    )
    p_dbg.set_defaults(func=run_debug_regex)
    return parser


def run_build(args: argparse.Namespace) -> int:
    if args.noise_ensemble_k is not None and args.noise_ensemble_k < 1:
        raise ValueError(f"--noise-ensemble-k must be >= 1, got {args.noise_ensemble_k}.")
    if args.fisher_batch_size is not None and args.fisher_batch_size < 1:
        raise ValueError(
            f"--fisher-batch-size must be >= 1, got {args.fisher_batch_size}."
        )
    if (
        args.fisher_hutchinson_probes is not None
        and args.fisher_hutchinson_probes < 1
    ):
        raise ValueError(
            "--fisher-hutchinson-probes must be >= 1, "
            f"got {args.fisher_hutchinson_probes}."
        )
    config = build_config_from_args(args)
    if config.calibration_noise_mode == "global" and config.noise_ensemble_k != 1:
        raise ValueError(
            "--calibration-noise-mode global requires --noise-ensemble-k 1, "
            f"got {config.noise_ensemble_k}."
        )
    output_path = _auto_output_path(args, config)
    adapter = get_adapter("pi05", **build_adapter_kwargs(args))

    def _progress(stage: str, frac: float) -> None:
        sys.stderr.write(f"[{stage:>10}] {frac * 100:5.1f}%\r")
        sys.stderr.flush()

    build_pack(
        adapter,
        output_path=output_path,
        config=config,
        num_samples=args.num_samples,
        progress=_progress,
    )
    sys.stderr.write("\n")
    print(f"Wrote pack to {output_path}")
    return 0


def run_debug_regex(args: argparse.Namespace) -> int:
    config = build_config_from_args(args)
    adapter = get_adapter(
        "pi05",
        checkpoint_path=args.checkpoint,
        device=args.device,
        params_dtype=args.params_dtype,
    )
    model = adapter.build_model()
    targets = list_target_modules(model, config)

    by_scope: dict[str, list[str]] = {"llm": [], "dit": []}
    for name, scope, _mod in targets:
        by_scope[scope].append(name)

    print(f"LLM matches ({len(by_scope['llm'])}):")
    for name in by_scope["llm"][:20]:
        print(f"  {name}")
    if len(by_scope["llm"]) > 20:
        print(f"  ... +{len(by_scope['llm']) - 20} more")
    print(f"DiT matches ({len(by_scope['dit'])}):")
    for name in by_scope["dit"][:20]:
        print(f"  {name}")
    if len(by_scope["dit"]) > 20:
        print(f"  ... +{len(by_scope['dit']) - 20} more")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    argv = _normalize_argv(argv)
    parser = _make_root_parser()
    if not argv or argv in (["-h"], ["--help"]):
        parser.print_help()
        return 0
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return int(args.func(args))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    sys.exit(main())
