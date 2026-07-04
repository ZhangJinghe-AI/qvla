"""Tests for scripts/build_pi05_pack.py config assembly."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS.parent / "src"))
_spec = importlib.util.spec_from_file_location(
    "build_pi05_pack", _SCRIPTS / "build_pi05_pack.py"
)
assert _spec and _spec.loader
build_pi05_pack = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_pi05_pack)


def _parse(argv: list[str]) -> argparse.Namespace:
    return build_pi05_pack._build_build_parser().parse_args(argv)


def test_default_config_matches_pi05_recipe():
    args = _parse(["--checkpoint", "/ckpt", "--output", "/out.pt"])
    cfg = build_pi05_pack.build_config_from_args(args)
    assert cfg.model_kind == "pi05"
    assert cfg.llm.weight_quantizer == "gptq"
    assert cfg.dit.weight_quantizer == "rtn"
    assert cfg.llm.act_scale_mode == "dynamic"
    assert cfg.dit.act_scale_mode == "per_step"


def test_dit_quant_override():
    args = _parse(
        [
            "--checkpoint", "/ckpt",
            "--output", "/out.pt",
            "--dit-quant", "gptq",
            "--dit-act-scale-mode", "dynamic",
        ]
    )
    cfg = build_pi05_pack.build_config_from_args(args)
    assert cfg.dit.weight_quantizer == "gptq"
    assert cfg.dit.act_scale_mode == "dynamic"


def test_shared_and_per_scope_weight_bits():
    args = _parse(
        [
            "--checkpoint", "/ckpt",
            "--output", "/out.pt",
            "--weight-bits", "4",
            "--llm-group-size", "64",
        ]
    )
    cfg = build_pi05_pack.build_config_from_args(args)
    assert cfg.llm.weight_bits == 4
    assert cfg.dit.weight_bits == 4
    assert cfg.llm.group_size == 64
    assert cfg.dit.group_size == 128


def test_weight_bits_16_means_no_quant():
    args = _parse(
        [
            "--checkpoint", "/ckpt",
            "--output", "/out.pt",
            "--weight-bits", "16",
            "--act-bits", "16",
        ]
    )
    cfg = build_pi05_pack.build_config_from_args(args)
    assert cfg.llm.weight_bits == 16
    assert cfg.dit.weight_bits == 16
    assert cfg.llm.act_bits == 16
    assert cfg.dit.act_bits == 16


def test_default_config_duquant_knobs():
    args = _parse(["--checkpoint", "/ckpt", "--output", "/out.pt"])
    cfg = build_pi05_pack.build_config_from_args(args)
    assert cfg.llm.pipeline == ("perm", "svd", "random_hadamard")
    assert cfg.llm.svd_source == "weight"
    assert cfg.llm.perm_score == "weight"
    assert cfg.dit.pipeline == ("perm", "svd", "random_hadamard")


def test_pipeline_override():
    args = _parse(
        [
            "--checkpoint", "/ckpt",
            "--output", "/out.pt",
            "--llm-pipeline", "none",
            "--dit-pipeline", "svd,hadamard",
        ]
    )
    cfg = build_pi05_pack.build_config_from_args(args)
    assert cfg.llm.pipeline == ()
    assert cfg.dit.pipeline == ("svd", "hadamard")


def test_act_percentile_mode_override():
    args = _parse(
        [
            "--checkpoint", "/ckpt",
            "--output", "/out.pt",
            "--dit-act-percentile-mode", "inner_channel",
        ]
    )
    cfg = build_pi05_pack.build_config_from_args(args)
    assert cfg.dit.act_percentile_mode == "inner_channel"
    assert cfg.llm.act_percentile_mode == "cross_channel"


def test_svd_source_activation_override():
    args = _parse(
        [
            "--checkpoint", "/ckpt",
            "--output", "/out.pt",
            "--llm-svd-source", "activation",
            "--llm-pipeline", "svd,hadamard",
        ]
    )
    cfg = build_pi05_pack.build_config_from_args(args)
    assert cfg.llm.svd_source == "activation"
    assert cfg.llm.pipeline == ("svd", "hadamard")


def test_debug_regex_subcommand_registered():
    parser = build_pi05_pack._make_root_parser()
    sub_action = next(
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    )
    assert "build" in sub_action.choices
    assert "debug-regex" in sub_action.choices


def test_auto_output_path():
    args = _parse(["--checkpoint", "/data/share/pi05-libero"])
    path = build_pi05_pack._auto_output_path(
        args, build_pi05_pack.build_config_from_args(args)
    )
    assert path == Path("/data/share/pi05-libero-packs/pi05-libero-W4A4.pt")
    args2 = _parse(
        [
            "--checkpoint", "/data/share/pi05-libero",
            "--dit-quant", "gptq",
        ]
    )
    path2 = build_pi05_pack._auto_output_path(
        args2, build_pi05_pack.build_config_from_args(args2)
    )
    assert "dit_weight_quantizer_gptq" in path2.name


def test_auto_output_path_w4a16():
    args = _parse(
        [
            "--checkpoint", "/data/share/pi05-libero",
            "--weight-bits", "4",
            "--act-bits", "16",
        ]
    )
    path = build_pi05_pack._auto_output_path(
        args, build_pi05_pack.build_config_from_args(args)
    )
    assert path.name == "pi05-libero-W4A16.pt"
