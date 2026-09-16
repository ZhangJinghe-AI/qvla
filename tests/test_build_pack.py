"""Tests for scripts/build_pack.py config assembly."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS.parent / "src"))
_spec = importlib.util.spec_from_file_location(
    "build_pack", _SCRIPTS / "build_pack.py"
)
assert _spec and _spec.loader
build_pack = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_pack)


def _parse(argv: list[str]) -> argparse.Namespace:
    return build_pack._build_build_parser().parse_args(argv)


def test_default_config_matches_pi05_recipe():
    args = _parse(["--model", "pi05", "--checkpoint", "/ckpt", "--output", "/out.pt"])
    cfg = build_pack.build_config_from_args(args)
    assert cfg.model_kind == "pi05"
    assert cfg.llm.weight_quantizer == "gptq"
    assert cfg.dit.weight_quantizer == "gptq"
    assert cfg.llm.act_scale_mode == "dynamic"
    assert cfg.dit.act_scale_mode == "dynamic"
    assert cfg.llm.act_scale_granularity == "per_token"
    assert cfg.dit.act_scale_granularity == "per_token"


def test_default_config_matches_groot_recipe():
    args = _parse(["--model", "groot_n17", "--checkpoint", "/ckpt", "--output", "/out.pt"])
    cfg = build_pack.build_config_from_args(args)
    assert cfg.model_kind == "groot_n17"
    assert cfg.llm.weight_quantizer == "gptq"
    assert cfg.dit.weight_quantizer == "gptq"
    assert cfg.dit.num_steps == 4
    assert cfg.dit.act_scale_mode == "dynamic"
    assert cfg.dit.act_scale_granularity == "per_token"


def test_dit_quant_override():
    args = _parse(
        [
            "--model", "pi05",
            "--checkpoint", "/ckpt",
            "--output", "/out.pt",
            "--dit-quant", "rtn",
            "--dit-act-scale-mode", "per_step",
        ]
    )
    cfg = build_pack.build_config_from_args(args)
    assert cfg.dit.weight_quantizer == "rtn"
    assert cfg.dit.act_scale_mode == "per_step"


def test_shared_and_per_scope_weight_bits():
    args = _parse(
        [
            "--model", "pi05",
            "--checkpoint", "/ckpt",
            "--output", "/out.pt",
            "--weight-bits", "4",
            "--llm-group-size", "64",
        ]
    )
    cfg = build_pack.build_config_from_args(args)
    assert cfg.llm.weight_bits == 4
    assert cfg.dit.weight_bits == 4
    assert cfg.llm.group_size == 64
    assert cfg.dit.group_size == 128


def test_weight_bits_16_means_no_quant():
    args = _parse(
        [
            "--model", "pi05",
            "--checkpoint", "/ckpt",
            "--output", "/out.pt",
            "--weight-bits", "16",
            "--act-bits", "16",
        ]
    )
    cfg = build_pack.build_config_from_args(args)
    assert cfg.llm.weight_bits == 16
    assert cfg.dit.weight_bits == 16
    assert cfg.llm.act_bits == 16
    assert cfg.dit.act_bits == 16


def test_default_config_duquant_knobs():
    args = _parse(["--model", "pi05", "--checkpoint", "/ckpt", "--output", "/out.pt"])
    cfg = build_pack.build_config_from_args(args)
    assert cfg.llm.pipeline == ()
    assert cfg.llm.svd_source == "weight"
    assert cfg.llm.perm_score == "weight"
    assert cfg.dit.pipeline == ()


def test_pipeline_override():
    args = _parse(
        [
            "--model", "pi05",
            "--checkpoint", "/ckpt",
            "--output", "/out.pt",
            "--llm-pipeline", "perm,svd,hadamard",
            "--dit-pipeline", "svd,hadamard",
        ]
    )
    cfg = build_pack.build_config_from_args(args)
    assert cfg.llm.pipeline == ("perm", "svd", "hadamard")
    assert cfg.dit.pipeline == ("svd", "hadamard")


def test_dit_clip_override():
    args = _parse(
        [
            "--model", "pi05",
            "--checkpoint", "/ckpt",
            "--output", "/out.pt",
            "--dit-pipeline", "clip",
            "--dit-act-outlier-std-k", "3",
            "--dit-act-outlier-selective-channels",
            "--dit-act-outlier-fit-tokens", "all",
        ]
    )
    cfg = build_pack.build_config_from_args(args)
    assert cfg.dit.pipeline == ("clip",)
    assert cfg.dit.act_outlier_std_k == 3.0
    assert cfg.dit.act_outlier_selective_channels is True
    assert cfg.dit.act_outlier_fit_tokens == "all"


def test_dit_clip_std_k_down_up_override():
    args = _parse(
        [
            "--model", "pi05",
            "--checkpoint", "/ckpt",
            "--output", "/out.pt",
            "--dit-pipeline", "clip",
            "--dit-act-outlier-std-k", "3",
            "--dit-act-outlier-std-k-down", "0.5",
            "--dit-act-outlier-std-k-up", "0.5",
            "--dit-act-outlier-selective-channels",
            "--dit-act-outlier-fit-tokens", "all",
        ]
    )
    cfg = build_pack.build_config_from_args(args)
    assert cfg.dit.act_outlier_std_k == 3.0
    assert cfg.dit.act_outlier_std_k_down == 0.5
    assert cfg.dit.act_outlier_std_k_up == 0.5
    assert cfg.llm.act_outlier_std_k_down == 0.0
    assert cfg.llm.act_outlier_std_k_up == 0.0


def test_llm_clip_std_k_down_up_raises():
    args = _parse(
        [
            "--model", "pi05",
            "--checkpoint", "/ckpt",
            "--output", "/out.pt",
            "--llm-pipeline", "clip",
            "--llm-act-outlier-std-k", "3",
            "--llm-act-outlier-std-k-down", "0.5",
        ]
    )
    with pytest.raises(ValueError, match="DiT-only|num_steps>1"):
        build_pack.build_config_from_args(args)


def test_dit_clip_skip_first_override():
    args = _parse(
        [
            "--model", "groot_n17",
            "--checkpoint", "/ckpt",
            "--output", "/out.pt",
            "--dit-pipeline", "clip",
            "--dit-act-outlier-std-k", "3",
            "--dit-act-outlier-selective-channels",
            "--dit-act-outlier-fit-tokens", "skip_first",
        ]
    )
    cfg = build_pack.build_config_from_args(args)
    assert cfg.dit.act_outlier_fit_tokens == "skip_first"


def test_pi05_dit_clip_skip_first_raises():
    args = _parse(
        [
            "--model", "pi05",
            "--checkpoint", "/ckpt",
            "--output", "/out.pt",
            "--dit-pipeline", "clip",
            "--dit-act-outlier-std-k", "3",
            "--dit-act-outlier-selective-channels",
            "--dit-act-outlier-fit-tokens", "skip_first",
        ]
    )
    with pytest.raises(ValueError, match="GR00T-only"):
        build_pack.build_config_from_args(args)


def test_dit_clip_without_all_tokens_raises():
    args = _parse(
        [
            "--model", "pi05",
            "--checkpoint", "/ckpt",
            "--output", "/out.pt",
            "--dit-pipeline", "clip",
            "--dit-act-outlier-std-k", "3",
            "--dit-act-outlier-selective-channels",
        ]
    )
    with pytest.raises(ValueError, match="act_outlier_fit_tokens"):
        build_pack.build_config_from_args(args)


def test_selective_channel_clip_override():
    args = _parse(
        [
            "--model", "pi05",
            "--checkpoint", "/ckpt",
            "--output", "/out.pt",
            "--llm-pipeline", "clip",
            "--llm-act-outlier-std-k", "3",
            "--llm-act-outlier-selective-channels",
        ]
    )
    cfg = build_pack.build_config_from_args(args)
    assert cfg.llm.act_outlier_selective_channels is True
    assert cfg.dit.act_outlier_selective_channels is False


def test_global_clip_override():
    args = _parse(
        [
            "--model", "pi05",
            "--checkpoint", "/ckpt",
            "--output", "/out.pt",
            "--llm-pipeline", "clip",
            "--llm-act-outlier-std-k", "3",
            "--llm-act-outlier-global",
        ]
    )
    cfg = build_pack.build_config_from_args(args)
    assert cfg.llm.act_outlier_global is True
    assert cfg.dit.act_outlier_global is False
    assert cfg.llm.act_outlier_selective_channels is False


def test_act_percentile_mode_override():
    args = _parse(
        [
            "--model", "pi05",
            "--checkpoint", "/ckpt",
            "--output", "/out.pt",
            "--dit-act-percentile-mode", "inner",
        ]
    )
    cfg = build_pack.build_config_from_args(args)
    assert cfg.dit.act_percentile_mode == "inner"
    assert cfg.llm.act_percentile_mode == "cross"


def test_svd_source_activation_override():
    args = _parse(
        [
            "--model", "pi05",
            "--checkpoint", "/ckpt",
            "--output", "/out.pt",
            "--llm-svd-source", "activation",
            "--llm-pipeline", "svd,hadamard",
        ]
    )
    cfg = build_pack.build_config_from_args(args)
    assert cfg.llm.svd_source == "activation"
    assert cfg.llm.pipeline == ("svd", "hadamard")


def test_debug_regex_subcommand_registered():
    parser = build_pack._make_root_parser()
    sub_action = next(
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    )
    assert "build" in sub_action.choices
    assert "debug-regex" in sub_action.choices


def test_auto_output_path():
    args = _parse(["--model", "pi05", "--checkpoint", "/data/share/pi05-libero"])
    path = build_pack._auto_output_path(
        args, build_pack.build_config_from_args(args)
    )
    assert path == Path("/data/share/pi05-libero-packs/pi05-libero-W4A4.pt")
    args2 = _parse(
        [
            "--model", "pi05",
            "--checkpoint", "/data/share/pi05-libero",
            "--dit-quant", "rtn",
        ]
    )
    path2 = build_pack._auto_output_path(
        args2, build_pack.build_config_from_args(args2)
    )
    assert "dit_weight_quantizer_rtn" in path2.name


def test_auto_output_path_w4a16():
    args = _parse(
        [
            "--model", "pi05",
            "--checkpoint", "/data/share/pi05-libero",
            "--weight-bits", "4",
            "--act-bits", "16",
        ]
    )
    path = build_pack._auto_output_path(
        args, build_pack.build_config_from_args(args)
    )
    assert path.name == "pi05-libero-W4A16.pt"


def test_embodiment_tag_rejected_for_pi05():
    args = _parse(
        [
            "--model", "pi05",
            "--checkpoint", "/ckpt",
            "--output", "/out.pt",
            "--embodiment-tag", "LIBERO_PANDA",
        ]
    )
    try:
        build_pack.build_adapter_kwargs(args)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "--embodiment-tag" in str(exc)


def test_embodiment_tag_accepted_for_groot():
    args = _parse(
        [
            "--model", "groot_n17",
            "--checkpoint", "/ckpt",
            "--output", "/out.pt",
            "--embodiment-tag", "LIBERO_PANDA",
        ]
    )
    kwargs = build_pack.build_adapter_kwargs(args)
    assert kwargs["embodiment_tag"] == "LIBERO_PANDA"
    assert kwargs["processor_model_name_or_path"] == "/data/share/Cosmos-Reason2-2B"


def test_processor_model_name_or_path_rejected_for_pi05():
    args = _parse(
        [
            "--model", "pi05",
            "--checkpoint", "/ckpt",
            "--output", "/out.pt",
            "--processor-model-name-or-path", "/data/share/Cosmos-Reason2-2B",
        ]
    )
    try:
        build_pack.build_adapter_kwargs(args)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "--processor-model-name-or-path" in str(exc)


def test_processor_model_name_or_path_override_for_groot():
    args = _parse(
        [
            "--model", "groot_n17",
            "--checkpoint", "/ckpt",
            "--output", "/out.pt",
            "--processor-model-name-or-path", "/other/cosmos",
        ]
    )
    kwargs = build_pack.build_adapter_kwargs(args)
    assert kwargs["processor_model_name_or_path"] == "/other/cosmos"


def test_dit_smooth_step_pmean_p_override():
    args = _parse(
        [
            "--model", "groot_n17",
            "--checkpoint", "/ckpt",
            "--output", "/out.pt",
            "--dit-pipeline", "smooth",
            "--dit-smooth-step-pmean-p", "4",
        ]
    )
    cfg = build_pack.build_config_from_args(args)
    assert cfg.dit.smooth_step_pmean_p == 4.0
    assert cfg.dit.pipeline == ("smooth",)
    assert cfg.llm.smooth_step_pmean_p is None
