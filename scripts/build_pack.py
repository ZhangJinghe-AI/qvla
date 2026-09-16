#!/usr/bin/env python
"""Build and debug QVLA W4A4 packs for any supported model.

Subcommands::

    build          Build a W4A4 pack (default when omitted)
    debug-regex    Dry-run layer include/exclude regex matching

Examples::

    # pi0.5
    uv run python scripts/build_pack.py \\
        --model pi05 \\
        --checkpoint /data/share/pi05-libero \\
        --output ./packs/pi05_libero_W4A4.pt \\
        --calibration-source file \\
        --calibration-data ../calibration_data/libero_object_16_7.npz \\
        -vv

    # GR00T-N1.7
    uv run python scripts/build_pack.py \\
        --model groot_n17 \\
        --checkpoint /data/share/gr00t-n17-libero \\
        --embodiment-tag LIBERO_PANDA \\
        --processor-model-name-or-path /data/share/Cosmos-Reason2-2B \\
        --calibration-source synthetic \\
        -vv

    # Fisher-driven perm on DiT (opt into rotation pipeline)
    uv run python scripts/build_pack.py \\
        --model pi05 \\
        --checkpoint /data/share/pi05-libero \\
        --dit-pipeline perm,svd,hadamard \\
        --dit-perm-score fisher \\
        --fisher-num-samples 4 \\
        --fisher-batch-size 4 \\
        -vv

    uv run python scripts/build_pack.py debug-regex \\
        --model pi05 \\
        --checkpoint /data/share/pi05-libero
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence, get_args

# Allow running from a fresh checkout without `pip install -e .`.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from qvla.adapters import get_adapter  # noqa: E402
from qvla.adapters.groot.config import DEFAULT_PROCESSOR_MODEL_NAME_OR_PATH  # noqa: E402
from qvla.build import build_pack  # noqa: E402
from qvla.config import (  # noqa: E402
    ActOutlierFitTokens,
    ActPercentileMode,
    ActScaleGranularity,
    ActScaleMode,
    CalibrationStepAggregation,
    FisherMethod,
    FisherType,
    PermScore,
    QVLAConfig,
    StepAggregation,
    SvdSource,
    QuantFormat,
    WeightQuantizer,
)
from qvla.core.pipeline import parse_pipeline_string  # noqa: E402
from qvla.runtime import list_target_modules  # noqa: E402

_SUBCOMMANDS = frozenset({"build", "debug-regex"})
_MODEL_CHOICES = ("pi05", "groot_n17")
# Argparse choices — derived from config Literals so typing and CLI stay synced.
_WEIGHT_QUANT_CHOICES: tuple[WeightQuantizer, ...] = get_args(WeightQuantizer)
_QUANT_FORMAT_CHOICES: tuple[QuantFormat, ...] = get_args(QuantFormat)
_ACT_SCALE_CHOICES: tuple[ActScaleMode, ...] = get_args(ActScaleMode)
_ACT_SCALE_GRANULARITY_CHOICES: tuple[ActScaleGranularity, ...] = get_args(
    ActScaleGranularity
)
_ACT_PERCENTILE_MODE_CHOICES: tuple[ActPercentileMode, ...] = get_args(ActPercentileMode)
_ACT_OUTLIER_FIT_TOKENS_CHOICES: tuple[ActOutlierFitTokens, ...] = get_args(
    ActOutlierFitTokens
)
_FISHER_STEP_AGG_CHOICES: tuple[StepAggregation, ...] = get_args(StepAggregation)
_FISHER_METHOD_CHOICES: tuple[FisherMethod, ...] = get_args(FisherMethod)
_FISHER_TYPE_CHOICES: tuple[FisherType, ...] = get_args(FisherType)
_CALIB_STEP_AGG_CHOICES: tuple[CalibrationStepAggregation, ...] = get_args(
    CalibrationStepAggregation
)
_PERM_SCORE_CHOICES: tuple[PermScore, ...] = get_args(PermScore)
_SVD_SOURCE_CHOICES: tuple[SvdSource, ...] = get_args(SvdSource)


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
        f"{prefix}_weight_format": "weight_format",
        f"{prefix}_act_format": "act_format",
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
        f"{prefix}_smooth_alpha": "smooth_alpha",
        f"{prefix}_smooth_epsilon": "smooth_epsilon",
        f"{prefix}_smooth_fisher_beta": "smooth_fisher_beta",
        f"{prefix}_smooth_act_percentile": "smooth_act_percentile",
        f"{prefix}_smooth_step_pmean_p": "smooth_step_pmean_p",
        f"{prefix}_act_outlier_kappa": "act_outlier_kappa",
        f"{prefix}_act_outlier_bulk_percentile": (
            "act_outlier_bulk_percentile"
        ),
        f"{prefix}_act_outlier_std_k": "act_outlier_std_k",
        f"{prefix}_act_outlier_std_k_down": "act_outlier_std_k_down",
        f"{prefix}_act_outlier_std_k_up": "act_outlier_std_k_up",
        f"{prefix}_act_outlier_fit_tokens": "act_outlier_fit_tokens",
        f"{prefix}_act_outlier_selective_channels": (
            "act_outlier_selective_channels"
        ),
        f"{prefix}_act_outlier_global": "act_outlier_global",
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
    model = args.model
    config = QVLAConfig.for_model_kind(model)

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
    if getattr(args, "fisher_type", None) is not None:
        top["fisher_type"] = args.fisher_type
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
    return f"Wllm{lw}Wdit{dw}Allm{la}Adit{da}"


def _auto_output_path(args: argparse.Namespace, config: QVLAConfig) -> Path:
    if args.output:
        return Path(args.output)
    ckpt = Path(args.checkpoint).resolve()
    out_dir = ckpt.parent / f"{ckpt.name}-packs"
    name = f"{ckpt.name}-{_pack_bitwidth_label(config)}"
    base = QVLAConfig.for_model_kind(args.model)
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
    calibration_source = getattr(args, "calibration_source", "synthetic")
    kwargs = {
        "checkpoint_path": args.checkpoint,
        "device": args.device,
        "params_dtype": args.params_dtype,
        "calibration_source": calibration_source,
    }
    if calibration_source == "file":
        calibration_data = getattr(args, "calibration_data", None)
        if not calibration_data:
            raise ValueError(
                "--calibration-data is required when --calibration-source=file"
            )
        kwargs["calibration_data_path"] = calibration_data

    embodiment_tag = getattr(args, "embodiment_tag", None)
    processor_model_name_or_path = getattr(args, "processor_model_name_or_path", None)
    if args.model == "groot_n17":
        if embodiment_tag is not None:
            kwargs["embodiment_tag"] = embodiment_tag
        # Default local Cosmos path when CLI omits the flag.
        kwargs["processor_model_name_or_path"] = (
            processor_model_name_or_path or DEFAULT_PROCESSOR_MODEL_NAME_OR_PATH
        )
    else:
        if embodiment_tag is not None:
            raise ValueError(
                f"--embodiment-tag is only valid with --model groot_n17; "
                f"got --model {args.model}."
            )
        if processor_model_name_or_path is not None:
            raise ValueError(
                f"--processor-model-name-or-path is only valid with "
                f"--model groot_n17; got --model {args.model}."
            )
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
        f"--{prefix}-weight-format",
        choices=_QUANT_FORMAT_CHOICES,
        default=None,
        dest=f"{prefix}_weight_format",
        help=(
            f"{help_prefix} weight number format: int (signed symmetric "
            "integer, default), fp (E2M1 micro-float), nvfp (NVFP4 "
            "two-level scaling). fp/nvfp only support weight_bits=4."
        ),
    )
    g.add_argument(
        f"--{prefix}-act-format",
        choices=_QUANT_FORMAT_CHOICES,
        default=None,
        dest=f"{prefix}_act_format",
        help=(
            f"{help_prefix} activation number format: int (signed symmetric "
            "integer, default), fp (E2M1 micro-float), nvfp (official NVFP4 "
            "online two-level; requires act_scale_mode=dynamic). "
            "fp/nvfp only support act_bits=4."
        ),
    )
    g.add_argument(
        f"--{prefix}-pipeline",
        type=parse_pipeline_string,
        default=None,
        dest=f"{prefix}_pipeline",
        metavar="STEPS",
        help=(
            f"{help_prefix} input transform pipeline (comma-separated). "
            "Steps: clip, smooth, perm, svd, hadamard, random_hadamard. "
            "Only allowlisted sequences are accepted (see "
            "qvla.core.pipeline.ALLOWED_PIPELINES). Examples: none; "
            "hadamard; clip; clip,hadamard; smooth; clip,smooth; "
            "clip,smooth,hadamard; perm,svd,hadamard. Default: none."
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
        help=(
            "Weight scale group size along K; -1 means per-channel. Also used "
            "as the dynamic activation block size when "
            "act_scale_granularity=per_block. NVFP4 requires 16."
        ),
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
            "per_token (default), per_channel, or per_block (dynamic; "
            "block size comes from group_size)."
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
        help=(
            "Energy for zigzag perm when pipeline includes perm: "
            "weight | activation | activation_weight | fisher."
        ),
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
        f"--{prefix}-smooth-alpha",
        type=float,
        default=None,
        dest=f"{prefix}_smooth_alpha",
        help=(
            f"{help_prefix} SmoothQuant migration exponent alpha (in [0, 1]; "
            "0.5 default). 0 -> keep activation, all difficulty to weight; "
            "1 -> keep weight, all difficulty to activation."
        ),
    )
    g.add_argument(
        f"--{prefix}-smooth-epsilon",
        type=float,
        default=None,
        dest=f"{prefix}_smooth_epsilon",
        help=(
            f"{help_prefix} SmoothQuant epsilon floor on per-channel amax "
            "before the fractional powers (default 1e-5)."
        ),
    )
    g.add_argument(
        f"--{prefix}-smooth-fisher-beta",
        type=float,
        default=None,
        dest=f"{prefix}_smooth_fisher_beta",
        help=(
            f"{help_prefix} Fisher boost on effective activation amax: "
            "F̃=F/max(F)∈(0,1], g=(1+beta·F̃)/mean(...), "
            "ã=a·g, s=ã^alpha/w^{1-alpha} "
            "(0 disables; >=0). Requires Fisher sensitivities."
        ),
    )
    g.add_argument(
        f"--{prefix}-smooth-act-percentile",
        type=float,
        default=None,
        dest=f"{prefix}_smooth_act_percentile",
        help=(
            f"{help_prefix} Percentile for SmoothQuant numerator a_j: "
            "100 (default) = hard per-channel absmax; "
            "e.g. 99.9 = per-forward channel percentile then max "
            "across forwards (clips rare outliers from dominating s). "
            "Incompatible with adaptive outlier clip "
            f"(--{prefix}-act-outlier-kappa > 0 or "
            f"--{prefix}-act-outlier-std-k > 0)."
        ),
    )
    g.add_argument(
        f"--{prefix}-smooth-step-pmean-p",
        type=float,
        default=None,
        dest=f"{prefix}_smooth_step_pmean_p",
        help=(
            f"{help_prefix} SmoothQuant a_j aggregation over denoise steps: "
            "omit for the original hard max over all tokens and steps; "
            "set to p>0 (typically 4) for "
            "a_j=(mean_t a_j,t^p)^(1/p) from per-step channel absmax. "
            f"Requires 'smooth' in --{prefix}-pipeline, num_steps>1, and "
            f"--{prefix}-smooth-act-percentile 100."
        ),
    )
    g.add_argument(
        f"--{prefix}-act-outlier-kappa",
        type=float,
        default=None,
        dest=f"{prefix}_act_outlier_kappa",
        help=(
            f"{help_prefix} Adaptive tip-clip κ for a_j / runtime act_clip "
            f"(0 disables). Requires 'clip' in --{prefix}-pipeline (e.g. "
            f"'clip' or 'clip,hadamard'). Concatenate calibration tokens, then "
            "a_j=min(max_j, κ·P_β(|x|)) with β from "
            f"--{prefix}-act-outlier-bulk-percentile. With 'smooth' in "
            f"--{prefix}-pipeline, a_j also fits s. Requires "
            f"--{prefix}-smooth-act-percentile=100 (default). Mutually "
            f"exclusive with --{prefix}-act-outlier-std-k > 0."
        ),
    )
    g.add_argument(
        f"--{prefix}-act-outlier-bulk-percentile",
        type=float,
        default=None,
        dest=f"{prefix}_act_outlier_bulk_percentile",
        help=(
            f"{help_prefix} Bulk percentile β ∈ (0, 100) for adaptive tip-clip "
            "(default 95). Used only when "
            f"--{prefix}-act-outlier-kappa > 0."
        ),
    )
    g.add_argument(
        f"--{prefix}-act-outlier-std-k",
        type=float,
        default=None,
        dest=f"{prefix}_act_outlier_std_k",
        help=(
            f"{help_prefix} Adaptive tip-clip via mean+k·std for a_j / "
            "runtime act_clip (0 disables). a_j=min(max_j, μ_j+k·σ_j). "
            f"Requires 'clip' in --{prefix}-pipeline. Same SmoothQuant wiring "
            "as kappa. Mutually exclusive "
            f"with --{prefix}-act-outlier-kappa > 0. Requires "
            f"--{prefix}-smooth-act-percentile=100 (default). "
            "DiT fits this independently at every denoise step "
            "(requires --dit-act-outlier-selective-channels and "
            "--dit-act-outlier-fit-tokens all|skip_first)."
        ),
    )
    g.add_argument(
        f"--{prefix}-act-outlier-std-k-down",
        type=float,
        default=None,
        dest=f"{prefix}_act_outlier_std_k_down",
        help=(
            f"{help_prefix} DiT-only. Subtracted from "
            f"--{prefix}-act-outlier-std-k at denoise step 0. "
            "0 (default) and --*-std-k-up 0 keep the same k at every step. "
            "Prefix-only layers still use std-k. LLM must leave this unset."
        ),
    )
    g.add_argument(
        f"--{prefix}-act-outlier-std-k-up",
        type=float,
        default=None,
        dest=f"{prefix}_act_outlier_std_k_up",
        help=(
            f"{help_prefix} DiT-only. Added to "
            f"--{prefix}-act-outlier-std-k at the last denoise step. "
            "k interpolates from std-k-down to std-k+up. "
            "Requires std-k - down > 0 and num_steps>=2. "
            "LLM must leave this unset."
        ),
    )
    g.add_argument(
        f"--{prefix}-act-outlier-fit-tokens",
        choices=_ACT_OUTLIER_FIT_TOKENS_CHOICES,
        default=None,
        dest=f"{prefix}_act_outlier_fit_tokens",
        help=(
            f"{help_prefix} Which tokens enter adaptive tip-clip (default "
            "image_lang_pad): 'image' or 'image_lang_pad' tip-clip those "
            "tokens and floor act_clip by max(|rest|); 'all' tip-clips every "
            "token (no rest floor); 'skip_first' tip-clips tokens after "
            "position 0 and floors by that token (GR00T DiT state only). "
            "LLM clip typically uses image_lang_pad. "
            "DiT clip: pi0.5 requires 'all'; GR00T may use 'skip_first'."
        ),
    )
    g.add_argument(
        f"--{prefix}-act-outlier-selective-channels",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest=f"{prefix}_act_outlier_selective_channels",
        help=(
            f"{help_prefix} temporary old/new clip ablation switch. When enabled, "
            "select channels whose hard amax exceeds "
            "median+3*1.4826*MAD across channels, then apply the existing "
            "mean+k*std threshold only to those channels. Requires "
            f"--{prefix}-act-outlier-std-k>0; no fallback is used. "
            "Required for DiT clip (per denoise step)."
        ),
    )
    g.add_argument(
        f"--{prefix}-act-outlier-global",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest=f"{prefix}_act_outlier_global",
        help=(
            f"{help_prefix} naive layer-global clip baseline. When enabled, "
            "fit one mean+k*std threshold over all tip |x| in the layer, "
            "then a_j=min(amax_j, c). Requires "
            f"--{prefix}-act-outlier-std-k>0. Mutually exclusive with "
            f"--{prefix}-act-outlier-selective-channels; no fallback is used."
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
            "'all' (default), one index (e.g. '50' as last for T=50), or "
            "comma-separated indices (e.g. '0,29,50')."
        ),
    )
    fisher.add_argument(
        "--fisher-type",
        choices=_FISHER_TYPE_CHOICES,
        default=None,
        dest="fisher_type",
        help=(
            "Fisher measurement target: "
            "'input_grad' uses (∂a/∂x)² collapsed to per-input-channel "
            "(QVLA paper diagonal); "
            "'output_hessian' back-props to the output activation to get "
            "per-token importance, then forms the input-side Hessian "
            "diagonal weighted by those token importances (HBVLA-style)."
        ),
    )
    fisher.add_argument(
        "--fisher-method",
        choices=_FISHER_METHOD_CHOICES,
        default=None,
        dest="fisher_method",
        help=(
            "Fisher estimator: exact (one backward per action dim) or "
            "hutchinson (random-projection approximation). Applies to any "
            "--fisher-type."
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
            "Observations per Fisher differentiable forward (default: 1). "
            "Raises the engine max_batch_size to at least this value."
        ),
    )
    top.add_argument(
        "--build-seed",
        type=int,
        default=None,
        dest="build_seed",
        help=(
            "Seed for reproducible random Hadamard and calibration noise "
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
    run.add_argument(
        "--model",
        required=True,
        choices=_MODEL_CHOICES,
        help="Model kind: selects adapter and default QVLA recipe.",
    )
    run.add_argument("--checkpoint", required=True)
    run.add_argument("--device", default="cuda")
    run.add_argument("--params-dtype", default="bfloat16")
    run.add_argument(
        "--embodiment-tag",
        default=None,
        dest="embodiment_tag",
        help="GR00T processor embodiment tag (e.g. LIBERO_PANDA). Only valid with --model groot_n17.",
    )
    run.add_argument(
        "--processor-model-name-or-path",
        default=None,
        dest="processor_model_name_or_path",
        help=(
            "Local Cosmos/VLM path for GR00T tokenizer + image preprocessor "
            "(default: /data/share/Cosmos-Reason2-2B). Only valid with --model groot_n17."
        ),
    )
    run.add_argument("-v", "--verbose", action="count", default=verbose_default)


def _build_build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build-pack build",
        description="Build a QVLA W4A4 pack.",
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
        prog="build-pack debug-regex",
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
        prog="build-pack",
        description="Build and debug QVLA W4A4 packs (pi05, groot, ...).",
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
    adapter = get_adapter(args.model, **build_adapter_kwargs(args))

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
    adapter = get_adapter(args.model, **build_adapter_kwargs(args))
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
