"""Single configuration dataclass for the whole QVLA recipe.

One config drives both the offline builder and the runtime loader so the two
can never drift. It's serialized into the pack header itself, so the pack file
is self-describing — the runtime side validates the loaded config against the
user-supplied one (any mismatch raises before any compute happens).

Design choices kept deliberately narrow:

* `weight_bits` / `act_bits` support `4` (int4 quant) and `16` (no quant —
  full precision; rotation / perm still apply). Wider int quant bit-widths
  would need a different storage layout.
* `weight_quantizer` ∈ {"gptq", "rtn"} on a *per-scope* basis (LLM vs DiT) —
  the recipe in the QVLA README uses gptq for LLM and rtn-residual for
  DiT, but the knob is free.
* `pipeline` — comma-ordered input transform steps ``perm``, ``svd``,
  ``hadamard``, ``random_hadamard`` (empty = no rotation).
* `act_scale_mode` ∈ {"per_step", "static", "dynamic"} — only the DiT side
  has a meaningful step axis; the LLM side coerces to "static" / "dynamic".
* `act_scale_granularity` ∈ {"per_channel", "per_token"} — for static /
  per_step only; dynamic is always per-token.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Literal

WeightQuantizer = Literal["gptq", "rtn", "rtn_residual"]
ActScaleMode = Literal["per_step", "static", "dynamic"]
ActScaleGranularity = Literal["per_channel", "per_token"]
ActPercentileMode = Literal["inner", "cross"]
StepAggregation = Literal["uniform"]
PermScore = Literal["weight", "activation", "activation_weight", "fisher"]
SvdSource = Literal["weight", "activation"]
PipelineStep = Literal["perm", "svd", "hadamard", "random_hadamard"]

DEFAULT_PIPELINE: tuple[PipelineStep, ...] = ("perm", "svd", "hadamard")


def _normalize_scope_dict(d: dict) -> dict:
    """Coerce scope dict fields when loading from JSON (``pipeline`` list/str → tuple)."""
    d = dict(d)
    if "pipeline" not in d:
        return d
    raw = d["pipeline"]
    if isinstance(raw, str):
        from qvla.core.rotation import parse_pipeline_string

        d["pipeline"] = parse_pipeline_string(raw)
    elif isinstance(raw, list):
        d["pipeline"] = tuple(raw)
    return d


@dataclass(frozen=True)
class ScopeConfig:
    """Per-scope knobs (one for the LLM backbone, one for the DiT action head).

    A scope is just "the set of layers whose qualified name matches
    ``include_regex`` and *not* ``exclude_regex``". Layers that match neither
    scope are left at full precision.
    """

    # Regex matched against `module.qualified_name`. Both empty disables this scope.
    include_regex: str = ""
    exclude_regex: str = ""

    # Per-scope quantizer choice.
    weight_quantizer: WeightQuantizer = "rtn"

    # int4 by default; 16 means keep full precision (no weight quant).
    weight_bits: int = 4
    group_size: int = 128

    # int4 by default; 16 means keep full precision (no activation quant).
    act_bits: int = 4
    act_scale_mode: ActScaleMode = "dynamic"
    act_scale_granularity: ActScaleGranularity = "per_channel"
    act_percentile: float = 99.9  # for static / per-step
    # ``inner``: per-dimension percentile over activation samples;
    # ``cross``: global cap on per-dimension maxes.
    act_percentile_mode: ActPercentileMode = "cross"

    # Number of denoise steps; only relevant for `per_step`. 1 collapses to static.
    num_steps: int = 1

    # Block size for block SVD / Hadamard (power of two, divides in_features).
    rotation_block_size: int = 64

    # Input-side transform pipeline, applied left-to-right at runtime.
    pipeline: tuple[PipelineStep, ...] = DEFAULT_PIPELINE

    # DuQuant zigzag permutation energy (only used when ``perm`` is in pipeline).
    perm_score: PermScore = "weight"

    # SVD basis for block rotation: ``weight`` (default) or ``activation``.
    svd_source: SvdSource = "weight"

    # GPTQ knobs.
    gptq_damp_percent: float = 0.01
    gptq_block_size: int = 128


@dataclass(frozen=True)
class QVLAConfig:
    """Top-level QVLA config.

    `model_kind` is purely advisory — it's recorded in the pack header so a
    `pi05` pack can refuse to load on a `groot` model with a clear error.
    """

    model_kind: str = "pi05"

    llm: ScopeConfig = field(default_factory=ScopeConfig)
    dit: ScopeConfig = field(default_factory=ScopeConfig)

    # If True, layers that match include but whose shape is incompatible with
    # the chosen group_size are silently skipped (with a log line). If False,
    # the builder raises so the user can fix the regex.
    skip_incompatible: bool = False

    # ---- Fisher sensitivity (``perm_score=fisher``) --------------------

    fisher_num_samples: int = 4
    fisher_step_aggregation: StepAggregation = "uniform"

    # Per-layer random Hadamard seeds are derived from this + qualified layer name.
    build_seed: int = 0

    # ---- serialization ---------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "QVLAConfig":
        llm_raw = _normalize_scope_dict(d.get("llm", {})) if "llm" in d else {}
        dit_raw = _normalize_scope_dict(d.get("dit", {})) if "dit" in d else {}
        llm = ScopeConfig(**llm_raw) if llm_raw else ScopeConfig()
        dit = ScopeConfig(**dit_raw) if dit_raw else ScopeConfig()
        return cls(
            model_kind=d.get("model_kind", "pi05"),
            llm=llm,
            dit=dit,
            skip_incompatible=d.get("skip_incompatible", True),
            fisher_num_samples=d.get("fisher_num_samples", 4),
            fisher_step_aggregation=d.get("fisher_step_aggregation", "uniform"),
            build_seed=d.get("build_seed", 0),
        )

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "QVLAConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    # ---- factories -------------------------------------------------------

    @classmethod
    def pi05_default(cls) -> "QVLAConfig":
        """QVLA's pi0.5 W4A4 recipe (QVLA README §5)."""
        llm = ScopeConfig(
            include_regex=(
                r"paligemma_lm\.layers\.\d+\."
                r"((qkv_proj|o_proj)|mlp\.(gate_up_proj|down_proj))$"
            ),
            exclude_regex=(
                r".*(vision|embed|lm_head|norm|layernorm|"
                r"action_in_proj|action_out_proj|time_mlp).*"
            ),
            weight_quantizer="gptq",
            weight_bits=4,
            group_size=128,
            act_bits=4,
            act_scale_mode="dynamic",
            act_percentile=99.9,
            num_steps=1,
            rotation_block_size=64,
            pipeline=DEFAULT_PIPELINE,
            perm_score="weight",
            svd_source="weight",
            gptq_damp_percent=0.01,
            gptq_block_size=128,
        )
        dit = ScopeConfig(
            include_regex=(
                r"expert_stack\.layers\.\d+\."
                r"((qkv_proj|o_proj)|mlp\.(gate_up_proj|down_proj))$"
            ),
            exclude_regex=(
                r".*(vision|embed|lm_head|norm|layernorm|"
                r"action_in_proj|action_out_proj|time_mlp).*"
            ),
            weight_quantizer="rtn",
            weight_bits=4,
            group_size=128,
            act_bits=4,
            act_scale_mode="per_step",
            act_percentile=99.9,
            num_steps=10,
            rotation_block_size=64,
            pipeline=DEFAULT_PIPELINE,
            perm_score="weight",
            svd_source="weight",
            gptq_damp_percent=0.01,
            gptq_block_size=128,
        )
        return cls(model_kind="pi05", llm=llm, dit=dit)

    @classmethod
    def groot_default(cls) -> "QVLAConfig":
        """Placeholder Groot N1.5/N1.7 recipe."""
        llm = ScopeConfig(
            include_regex=(
                r".*backbone\.eagle_model\.language_model\..*\."
                r"(q_proj|k_proj|v_proj|o_proj|"
                r"gate_proj|up_proj|down_proj)$"
            ),
            exclude_regex=(
                r"(?:^|\.)(vision|radio|norm|ln|layernorm|embed|lm_head|"
                r"timestep_encoder|state_encoder|action_encoder|action_decoder|"
                r"pos_embed|vl_self_attention|vlln|future_tokens)(?:\.|$)"
            ),
            weight_quantizer="gptq",
            weight_bits=4,
            group_size=128,
            act_bits=4,
            act_scale_mode="dynamic",
            num_steps=1,
            rotation_block_size=64,
            pipeline=DEFAULT_PIPELINE,
        )
        dit = ScopeConfig(
            include_regex=(
                r".*action_head\.model\.transformer_blocks\.\d+\."
                r"(attn1\.(to_q|to_k|to_v|to_out\.0)|ff\.net\.(0\.proj|2))$"
            ),
            exclude_regex=r"^$",
            weight_quantizer="rtn_residual",
            weight_bits=4,
            group_size=128,
            act_bits=4,
            act_scale_mode="per_step",
            num_steps=8,
            rotation_block_size=64,
            pipeline=DEFAULT_PIPELINE,
        )
        return cls(model_kind="groot", llm=llm, dit=dit)

    @classmethod
    def for_model_kind(cls, model_kind: str) -> "QVLAConfig":
        """Recipe factory keyed by :attr:`~ModelAdapter.model_kind`."""
        if model_kind == "pi05":
            return cls.pi05_default()
        if model_kind == "groot":
            return cls.groot_default()
        raise ValueError(
            f"Unknown model_kind={model_kind!r}. "
            "Built-in recipes: 'pi05', 'groot'."
        )

    def with_overrides(self, **kwargs) -> "QVLAConfig":
        return replace(self, **kwargs)

    @property
    def needs_fisher(self) -> bool:
        """True when any scope uses Fisher-driven zigzag perm."""
        return (
            "perm" in self.llm.pipeline and self.llm.perm_score == "fisher"
        ) or (
            "perm" in self.dit.pipeline and self.dit.perm_score == "fisher"
        )


__all__ = [
    "ActPercentileMode",
    "ActScaleGranularity",
    "ActScaleMode",
    "DEFAULT_PIPELINE",
    "QVLAConfig",
    "PermScore",
    "PipelineStep",
    "ScopeConfig",
    "StepAggregation",
    "SvdSource",
    "WeightQuantizer",
]
