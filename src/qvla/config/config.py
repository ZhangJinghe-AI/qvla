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
  built-in recipes default both scopes to gptq; rtn / rtn_residual remain
  available via override.
* `pipeline` — comma-ordered input transforms from
  :data:`qvla.core.pipeline.ALLOWED_PIPELINES` (``clip``, ``smooth``,
  ``perm``, ``svd``, ``hadamard``, ``random_hadamard``; empty = none).
* `act_scale_mode` ∈ {"per_step", "static", "dynamic"} — DiT may use any;
  LLM must be ``dynamic`` (variable sequence length). ``per_step`` also
  requires ``num_steps > 1``.
* `act_scale_granularity` ∈ {"per_channel", "per_token", "per_block"} —
  for static / per_step (default ``per_token``). Dynamic ``per_block`` uses
  ``group_size`` as the activation block size. NVFP4 requires block size 16.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Literal

WeightQuantizer = Literal["gptq", "rtn", "rtn_residual"]
QuantFormat = Literal["int", "fp", "nvfp"]
ActScaleMode = Literal["per_step", "static", "dynamic"]
ActScaleGranularity = Literal["per_channel", "per_token", "per_block"]
ActPercentileMode = Literal["inner", "cross"]
StepAggregation = Literal[
    "uniform", "max", "late_mean", "very_late_mean", "weighted_linear"
]
# Calibration XᵀX aggregation does not support ``max`` (non-linear over Gram
# matrices); keep a dedicated Literal so config typing stays precise.
CalibrationStepAggregation = Literal[
    "uniform", "late_mean", "very_late_mean", "weighted_linear"
]
FisherMethod = Literal["exact", "hutchinson"]
FisherType = Literal["input_grad", "output_hessian"]
PermScore = Literal["weight", "activation", "activation_weight", "fisher"]
SvdSource = Literal["weight", "activation"]
PipelineStep = Literal[
    "clip", "smooth", "perm", "svd", "hadamard", "random_hadamard"
]
ActOutlierFitTokens = Literal["image", "image_lang_pad", "all", "skip_first"]

# Empty = no input transform. Opt into e.g. clip,hadamard or smooth via CLI.
DEFAULT_PIPELINE: tuple[PipelineStep, ...] = ()


def _validate_scope_pipeline(
    pipeline: tuple[PipelineStep, ...],
) -> tuple[PipelineStep, ...]:
    """Lazy import avoids config ↔ core.pack circular import via pipeline."""
    from qvla.core.pipeline import validate_pipeline

    return validate_pipeline(pipeline)



def _normalize_scope_dict(d: dict) -> dict:
    """Coerce scope dict fields when loading from JSON (``pipeline`` list/str → tuple)."""
    d = dict(d)
    raw = d.get("pipeline", ())
    if isinstance(raw, list):
        pipe: tuple = tuple(raw)
    elif isinstance(raw, tuple):
        pipe = raw
    elif isinstance(raw, str):
        from qvla.core.pipeline import parse_pipeline_string

        pipe = parse_pipeline_string(raw)
    elif raw is None:
        pipe = ()
    else:
        raise TypeError(
            f"scope pipeline must be list/tuple/str, got {type(raw).__name__}."
        )
    d["pipeline"] = _validate_scope_pipeline(pipe)  # type: ignore[arg-type]
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
    # Number format for quantized weights: "int" (signed symmetric integer
    # grid, default), "fp" (E2M1 micro-float), "nvfp" (NVFP4 — E2M1 with
    # per-block + per-tensor two-level scaling).
    weight_format: QuantFormat = "int"
    # Weight scale group size; also the dynamic activation block size when
    # act_scale_granularity="per_block". NVFP4 requires 16.
    group_size: int = 128

    # int4 by default; 16 means keep full precision (no activation quant).
    act_bits: int = 4
    # Number format for quantized activations: "int" (signed symmetric
    # integer, default), "fp" (E2M1 micro-float, single-level scale), or
    # "nvfp" (official NVFP4 online two-level; requires act_scale_mode=dynamic).
    act_format: QuantFormat = "int"
    act_scale_mode: ActScaleMode = "dynamic"
    act_scale_granularity: ActScaleGranularity = "per_token"
    act_percentile: float = 99.9  # for static / per-step
    # ``inner``: per-dimension percentile over activation samples;
    # ``cross``: global cap on per-dimension maxes.
    act_percentile_mode: ActPercentileMode = "cross"

    # Number of denoise steps; used by ``per_step`` act scales, DiT
    # per-step ``act_clip``, and SmoothQuant ``smooth_step_pmean_p``.
    # 1 collapses to static.
    num_steps: int = 1

    # Block size for block SVD / Hadamard (power of two, divides in_features).
    rotation_block_size: int = 64

    # Full input-side pipeline (clip / smooth / orthogonal). Must be in
    # :data:`~qvla.core.pipeline.ALLOWED_PIPELINES`.
    pipeline: tuple[PipelineStep, ...] = DEFAULT_PIPELINE

    # DuQuant zigzag permutation energy (only used when ``perm`` is in pipeline).
    perm_score: PermScore = "weight"

    # SVD basis for block rotation: ``weight`` (default) or ``activation``.
    svd_source: SvdSource = "weight"

    # GPTQ knobs.
    gptq_damp_percent: float = 0.01
    gptq_block_size: int = 128

    # When True, weight the GPTQ Hessian by per-channel Fisher sensitivity
    # so that GPTQ protects action-sensitive input columns more aggressively.
    fisher_gptq: bool = False

    # ---- SmoothQuant / act_clip knobs (enabled by pipeline steps) --------
    # ``smooth`` in pipeline → fit per-channel ``s``, fold into ``W``, runtime
    # ``x / s``. ``clip`` in pipeline → adaptive tip-clip ``act_clip`` (requires
    # kappa>0 or std_k>0). Runtime order: clip → smooth → orthogonal.
    # Migration exponent. SmoothQuant paper starts at 0.5. Must lie in [0, 1].
    smooth_alpha: float = 0.5
    # Floor added to per-channel amax before the fractional powers, to keep
    # ``s`` finite when a channel is dead on one side (all-zero activation or
    # weight column).
    smooth_epsilon: float = 1e-5
    # Fisher boost on effective activation amax before SmoothQuant:
    # ``F̃=F/max(F)∈(0,1], g=(1+beta·F̃)/mean(...), ã=a·g``,
    # then ``s=ã^alpha/w^{1-alpha}`` (see ``fit_smooth_scale``).
    # ``0.0`` disables; non-zero requires Fisher at build time.
    smooth_fisher_beta: float = 0.0
    # Activation statistic used as the SmoothQuant numerator ``a_j``.
    # ``100`` (default): hard per-channel absmax over calibration.
    # ``<100`` (e.g. ``99.9``): per-forward channel percentile of ``|x|``,
    # then max across forwards (same ``inner`` stream as act scales). Clips
    # rare outliers from dominating ``s`` / weight-column blow-up.
    # Mutually exclusive with adaptive outlier clip (kappa>0 or std_k>0).
    smooth_act_percentile: float = 100.0
    # DiT SmoothQuant numerator aggregation over denoise steps. ``None``
    # (default) keeps the original hard max over all tokens and steps.
    # A positive ``p`` uses per-step channel absmax then
    # ``a_j = (mean_t a_{j,t}^p)^{1/p}``. Requires ``smooth`` in pipeline,
    # ``num_steps>1``, and ``smooth_act_percentile==100``. Cross-attn K/V
    # is cached and reused every denoise step, so its per-step table is
    # that absmax repeated; the p-mean equals the cached value.
    smooth_step_pmean_p: float | None = None
    # Adaptive tip-clip method A: κ × P_β. ``0`` disables this method.
    # ``>0`` (e.g. ``2``): ``a_j = min(max_j, κ · P_β(|x|))`` with
    # ``β = act_outlier_bulk_percentile``. Requires ``clip`` in pipeline.
    # Mutually exclusive with ``act_outlier_std_k > 0``.
    # Requires ``smooth_act_percentile==100`` when SmoothQuant also runs.
    act_outlier_kappa: float = 0.0
    # Bulk reference percentile β ∈ (0, 100) for the κ × P_β tip-clip rule.
    act_outlier_bulk_percentile: float = 95.0
    # Adaptive tip-clip method B: mean + k·std. ``0`` disables this method.
    # ``>0`` (e.g. ``3``): ``a_j = min(max_j, μ_j + k·σ_j)``. Requires
    # ``clip`` in pipeline. Mutually exclusive with kappa; requires
    # ``smooth_act_percentile==100`` when SmoothQuant also runs.
    act_outlier_std_k: float = 0.0
    # DiT denoise-loop only. ``0`` / ``0`` keeps ``act_outlier_std_k`` at every
    # step (original). If either is ``>0``, k goes from ``std_k - down`` at
    # step 0 to ``std_k + up`` at the last step. Prefix-only layers still
    # use ``act_outlier_std_k``. Requires ``std_k>0``, ``num_steps>1``, and
    # ``std_k - down > 0``. LLM must leave both at 0.
    act_outlier_std_k_down: float = 0.0
    act_outlier_std_k_up: float = 0.0
    # Which tokens enter the adaptive tip-clip statistic:
    # ``image`` / ``image_lang_pad`` (LLM) tip-clip those tokens and floor by
    # max(|rest|); ``all`` tip-clips every token (no rest floor);
    # ``skip_first`` (DiT) tip-clips tokens after position 0 and floors by
    # that first token — GR00T ``cat(state, action)``.
    act_outlier_fit_tokens: ActOutlierFitTokens = "image_lang_pad"
    # Temporary old/new ablation switch. False preserves the original behavior
    # (mean+k·std on every channel). True first selects unusually large-amax
    # channels with median+3*1.4826*MAD, then applies mean+k·std only there.
    act_outlier_selective_channels: bool = False
    # Naive reviewer baseline. False keeps per-channel mean+k·std. True uses
    # one layer-wide mean+k·std over all tip |x|, then a_j=min(amax_j, c).
    # Requires std_k>0. Mutually exclusive with selective-channel clip.
    act_outlier_global: bool = False

    def __post_init__(self) -> None:
        # Frozen dataclass — validation only, no mutation.
        _validate_scope_pipeline(self.pipeline)
        if not (0.0 <= float(self.smooth_alpha) <= 1.0):
            raise ValueError(
                f"smooth_alpha must be in [0, 1], got {self.smooth_alpha}."
            )
        if float(self.smooth_epsilon) <= 0.0:
            raise ValueError(
                f"smooth_epsilon must be > 0, got {self.smooth_epsilon}."
            )
        if not (0.0 < float(self.smooth_act_percentile) <= 100.0):
            raise ValueError(
                "smooth_act_percentile must be in (0, 100], got "
                f"{self.smooth_act_percentile}."
            )
        if self.smooth_step_pmean_p is not None:
            pmean_p = float(self.smooth_step_pmean_p)
            if not math.isfinite(pmean_p) or pmean_p <= 0.0:
                raise ValueError(
                    "smooth_step_pmean_p must be a finite value > 0, "
                    f"got {self.smooth_step_pmean_p}."
                )
            if "smooth" not in self.pipeline:
                raise ValueError(
                    "smooth_step_pmean_p requires 'smooth' in pipeline "
                    f"(got pipeline={self.pipeline!r})."
                )
            if int(self.num_steps) <= 1:
                raise ValueError(
                    "smooth_step_pmean_p requires num_steps>1 to aggregate "
                    f"denoise-step amax, got num_steps={self.num_steps}."
                )
            if float(self.smooth_act_percentile) != 100.0:
                raise ValueError(
                    "smooth_step_pmean_p requires smooth_act_percentile==100 "
                    "(hard per-step absmax); got "
                    f"{self.smooth_act_percentile}."
                )
        if float(self.act_outlier_kappa) < 0.0:
            raise ValueError(
                "act_outlier_kappa must be >= 0, got "
                f"{self.act_outlier_kappa}."
            )
        if float(self.act_outlier_std_k) < 0.0:
            raise ValueError(
                "act_outlier_std_k must be >= 0, got "
                f"{self.act_outlier_std_k}."
            )
        if float(self.act_outlier_std_k_down) < 0.0:
            raise ValueError(
                "act_outlier_std_k_down must be >= 0, got "
                f"{self.act_outlier_std_k_down}."
            )
        if float(self.act_outlier_std_k_up) < 0.0:
            raise ValueError(
                "act_outlier_std_k_up must be >= 0, got "
                f"{self.act_outlier_std_k_up}."
            )
        if not (0.0 < float(self.act_outlier_bulk_percentile) < 100.0):
            raise ValueError(
                "act_outlier_bulk_percentile must be in (0, 100), got "
                f"{self.act_outlier_bulk_percentile}."
            )
        kappa = float(self.act_outlier_kappa)
        std_k = float(self.act_outlier_std_k)
        if kappa > 0.0 and std_k > 0.0:
            raise ValueError(
                "act_outlier_kappa>0 and act_outlier_std_k>0 "
                f"are mutually exclusive (got kappa={kappa}, std_k={std_k})."
            )
        if (kappa > 0.0 or std_k > 0.0) and float(self.smooth_act_percentile) < 100.0:
            raise ValueError(
                "adaptive act outlier clip (kappa>0 or std_k>0) is mutually "
                "exclusive with smooth_act_percentile<100 (got kappa="
                f"{kappa}, std_k={std_k}, percentile="
                f"{self.smooth_act_percentile})."
            )
        has_clip = "clip" in self.pipeline
        has_adaptive = kappa > 0.0 or std_k > 0.0
        if has_clip and not has_adaptive:
            raise ValueError(
                "pipeline includes 'clip' but act_outlier_kappa and "
                "act_outlier_std_k are both 0; set one tip-clip method."
            )
        if has_adaptive and not has_clip:
            raise ValueError(
                "act_outlier_kappa>0 or act_outlier_std_k>0 "
                "requires 'clip' in pipeline "
                f"(got pipeline={self.pipeline!r})."
            )
        if self.act_outlier_fit_tokens not in (
            "image",
            "image_lang_pad",
            "all",
            "skip_first",
        ):
            raise ValueError(
                "act_outlier_fit_tokens must be 'image', 'image_lang_pad', "
                f"'all', or 'skip_first', got {self.act_outlier_fit_tokens!r}."
            )
        if self.act_outlier_selective_channels and std_k <= 0.0:
            raise ValueError(
                "act_outlier_selective_channels=True requires "
                "act_outlier_std_k>0; selective channel filtering is defined "
                "only for the mean+k*std clip method."
            )
        if self.act_outlier_global and std_k <= 0.0:
            raise ValueError(
                "act_outlier_global=True requires act_outlier_std_k>0; "
                "layer-global clip is defined only for the mean+k*std method."
            )
        if self.act_outlier_global and self.act_outlier_selective_channels:
            raise ValueError(
                "act_outlier_global and act_outlier_selective_channels "
                "are mutually exclusive."
            )
        std_k_down = float(self.act_outlier_std_k_down)
        std_k_up = float(self.act_outlier_std_k_up)
        if std_k_down > 0.0 or std_k_up > 0.0:
            if std_k <= 0.0:
                raise ValueError(
                    "act_outlier_std_k_down/up require act_outlier_std_k>0; "
                    f"got std_k={std_k}, down={std_k_down}, up={std_k_up}."
                )
            if int(self.num_steps) <= 1:
                raise ValueError(
                    "act_outlier_std_k_down/up require num_steps>1; "
                    f"got num_steps={self.num_steps}."
                )
            if std_k - std_k_down <= 0.0:
                raise ValueError(
                    "act_outlier_std_k - act_outlier_std_k_down must be > 0; "
                    f"got std_k={std_k}, down={std_k_down}."
                )
        if self.act_scale_mode not in ("per_step", "static", "dynamic"):
            raise ValueError(
                "act_scale_mode must be 'per_step', 'static', or 'dynamic', "
                f"got {self.act_scale_mode!r}."
            )
        if self.act_scale_mode == "per_step" and self.num_steps <= 1:
            raise ValueError(
                "act_scale_mode='per_step' requires num_steps > 1 "
                f"(got num_steps={self.num_steps})."
            )
        if self.weight_format not in ("int", "fp", "nvfp"):
            raise ValueError(
                f"weight_format must be 'int', 'fp', or 'nvfp', "
                f"got {self.weight_format!r}."
            )
        if self.weight_format in ("fp", "nvfp") and self.weight_bits != 4:
            raise ValueError(
                f"weight_format={self.weight_format!r} only supports "
                f"weight_bits=4, got {self.weight_bits}."
            )
        if self.weight_format == "nvfp" and self.group_size != 16:
            raise ValueError(
                "weight_format='nvfp' requires group_size=16, "
                f"got {self.group_size}."
            )
        if self.act_format not in ("int", "fp", "nvfp"):
            raise ValueError(
                f"act_format must be 'int', 'fp', or 'nvfp', "
                f"got {self.act_format!r}."
            )
        if self.act_format in ("fp", "nvfp") and self.act_bits != 4:
            raise ValueError(
                f"act_format={self.act_format!r} only supports "
                f"act_bits=4, got {self.act_bits}."
            )
        if self.act_scale_granularity not in ("per_channel", "per_token", "per_block"):
            raise ValueError(
                "act_scale_granularity must be 'per_channel', 'per_token', "
                f"or 'per_block', got {self.act_scale_granularity!r}."
            )
        # Per-block scaling is a granularity choice shared by all number
        # formats. It is computed online and uses group_size as the block size.
        if self.act_scale_granularity == "per_block":
            if self.act_scale_mode != "dynamic":
                raise ValueError(
                    "act_scale_granularity='per_block' requires "
                    f"act_scale_mode='dynamic', got {self.act_scale_mode!r}."
                )
            if self.group_size <= 0:
                raise ValueError(
                    "act_scale_granularity='per_block' requires "
                    f"group_size > 0, got {self.group_size}."
                )

        # NVFP4 adds stricter hardware-format constraints.
        if self.act_format == "nvfp":
            if self.act_scale_mode != "dynamic":
                raise ValueError(
                    "act_format='nvfp' only supports act_scale_mode='dynamic' "
                    f"(online per-token × block-16); got {self.act_scale_mode!r}."
                )
            if self.act_scale_granularity != "per_block":
                raise ValueError(
                    "act_format='nvfp' requires act_scale_granularity='per_block' "
                    f"(got {self.act_scale_granularity!r})."
                )
            if self.group_size != 16:
                raise ValueError(
                    "act_format='nvfp' requires group_size=16, "
                    f"got {self.group_size}."
                )

    @property
    def smooth_enabled(self) -> bool:
        """True when ``smooth`` is in :attr:`pipeline`."""
        return "smooth" in self.pipeline

    @property
    def clip_enabled(self) -> bool:
        """True when ``clip`` is in :attr:`pipeline`."""
        return "clip" in self.pipeline



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
    # How to aggregate per-step DiT XᵀX (covariance / Hessian) into a single
    # matrix per layer.  Applies to rotation fitting (SVD/perm) and GPTQ Hessian.
    # "uniform"         all steps equal weight (default, matches legacy behaviour).
    # "late_mean"       only the second half of denoise steps.
    # "very_late_mean"  only the last fifth of denoise steps (e.g. 8-9 for 10 steps).
    # "weighted_linear" w(s) ∝ (s+1), later steps contribute more.
    calibration_step_aggregation: CalibrationStepAggregation = "uniform"
    # Action-chunk timesteps for Fisher Jacobian:
    # "all" | integer index | comma-separated indices (e.g. "0,29,50").
    # Default "all" uses every action-chunk timestep.
    fisher_action_timestep: str = "all"
    # Fisher target — *what* per-channel importance measures:
    #   "input_grad"    : Σ_i E[(∂a_i/∂x_c)²] (per-input-channel Jacobian).
    #   "output_hessian": token importance from ∂a/∂y then weight the
    #                     input-side Hessian diagonal (HBVLA-style rectified
    #                     policy-aware Hessian).
    # Orthogonal to fisher_method: either can be estimated with "exact" or
    # "hutchinson".
    fisher_type: FisherType = "input_grad"
    # Fisher estimator: exact Jacobian loop or Hutchinson random projections.
    fisher_method: FisherMethod = "exact"
    # Number of random projections when fisher_method="hutchinson".
    fisher_hutchinson_probes: int = 8
    # Observations per differentiable Fisher forward. Pack build raises the
    # pi0.5 engine max_batch_size to at least this before build_model().
    fisher_batch_size: int = 1

    # Shared RNG seed for pack build (calibration noise + random Hadamard).
    build_seed: int = 0
    # How many independent noise draws to ensemble during calibration / Fisher.
    # 1 = single noise (legacy). Only DiT layers re-collect across noises;
    # LLM activations are noise-invariant and are hooked once.
    noise_ensemble_k: int = 1
    # ``per_sample``: fresh noise per calibration sample; ``global``: one shared noise.
    calibration_noise_mode: Literal["per_sample", "global"] = "per_sample"

    def __post_init__(self) -> None:
        # Cross-scope constraints (ScopeConfig cannot know llm vs dit).
        if self.llm.act_scale_mode != "dynamic":
            raise ValueError(
                "LLM act_scale_mode must be 'dynamic' (variable sequence "
                f"length; no denoise step axis); got {self.llm.act_scale_mode!r}."
            )
        dit = self.dit
        if dit.clip_enabled:
            if int(dit.num_steps) <= 1:
                raise ValueError(
                    "DiT clip requires num_steps>1 so each denoise step "
                    f"fits its own act_clip; got num_steps={dit.num_steps}."
                )
            if float(dit.act_outlier_kappa) > 0.0:
                raise ValueError(
                    "DiT clip does not allow act_outlier_kappa; use "
                    "selective mean+k*std only."
                )
            if dit.act_outlier_global:
                raise ValueError(
                    "DiT clip does not allow act_outlier_global; use "
                    "selective mean+k*std only."
                )
            if float(dit.act_outlier_std_k) <= 0.0:
                raise ValueError(
                    "DiT clip requires act_outlier_std_k>0 (selective "
                    f"mean+k*std); got {dit.act_outlier_std_k}."
                )
            if not dit.act_outlier_selective_channels:
                raise ValueError(
                    "DiT clip requires act_outlier_selective_channels=True; "
                    "unselected channels must keep hard amax."
                )
            if dit.act_outlier_fit_tokens not in ("all", "skip_first"):
                raise ValueError(
                    "DiT clip requires act_outlier_fit_tokens='all' "
                    "(pi0.5) or 'skip_first' (GR00T state token as rest); "
                    f"got {dit.act_outlier_fit_tokens!r}."
                )
            if (
                dit.act_outlier_fit_tokens == "skip_first"
                and self.model_kind != "groot_n17"
            ):
                raise ValueError(
                    "act_outlier_fit_tokens='skip_first' is GR00T-only "
                    "(leading state token); "
                    f"got model_kind={self.model_kind!r}."
                )
        if self.llm.clip_enabled and self.llm.act_outlier_fit_tokens == "skip_first":
            raise ValueError(
                "act_outlier_fit_tokens='skip_first' is DiT-only."
            )
        if (
            float(self.llm.act_outlier_std_k_down) > 0.0
            or float(self.llm.act_outlier_std_k_up) > 0.0
        ):
            raise ValueError(
                "act_outlier_std_k_down/up are DiT-only "
                f"(got llm down={self.llm.act_outlier_std_k_down}, "
                f"up={self.llm.act_outlier_std_k_up})."
            )

    @classmethod
    def for_model_kind(cls, model_kind: str) -> "QVLAConfig":
        """Recipe factory keyed by model kind."""
        if model_kind == "pi05":
            return cls.pi05_default()
        if model_kind == "groot_n17":
            return cls.groot_default()
        raise ValueError(
            f"Unknown model_kind={model_kind!r}. "
            "Built-in recipes: 'pi05', 'groot_n17'."
        )

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
            weight_quantizer="gptq",
            weight_bits=4,
            group_size=128,
            act_bits=4,
            act_scale_mode="dynamic",
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
        """GR00T-N1.7 W4A4 recipe.

        Backbone: Qwen3-VL text layers (12 layers after truncation).
        Action head: DiT transformer blocks (16 layers, 4 denoise steps).
        """
        llm = ScopeConfig(
            include_regex=(
                r"backbone\.qwen3vl_model\.model\.language_model\.layers\.\d+\."
                r"(qkv_proj|o_proj|mlp\.(gate_up_proj|down_proj))$"
            ),
            exclude_regex=r".*(visual|embed|norm|lm_head|rotary).*",
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
                r"action_head\.model\.transformer_blocks\.\d+\."
                r"(attn1\.(to_q|to_k|to_v|to_out)|ff\.(fc1|fc2))$"
            ),
            exclude_regex=(
                r".*(norm|timestep_encoder|action_encoder|state_encoder|"
                r"action_decoder|pos_embed|vl_self_attention).*"
            ),
            weight_quantizer="gptq",
            weight_bits=4,
            group_size=128,
            act_bits=4,
            act_scale_mode="dynamic",
            act_percentile=99.9,
            num_steps=4,
            rotation_block_size=64,
            pipeline=DEFAULT_PIPELINE,
            perm_score="weight",
            svd_source="weight",
            gptq_damp_percent=0.01,
            gptq_block_size=128,
        )
        return cls(model_kind="groot_n17", llm=llm, dit=dit)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "QVLAConfig":
        llm_raw = _normalize_scope_dict(dict(d.get("llm", {})))
        dit_raw = _normalize_scope_dict(dict(d.get("dit", {})))
        llm_fields = {f.name for f in ScopeConfig.__dataclass_fields__.values()}
        for label, raw in (("llm", llm_raw), ("dit", dit_raw)):
            unknown = set(raw) - llm_fields
            if unknown:
                raise ValueError(
                    f"Unknown {label} ScopeConfig field(s): {sorted(unknown)}. "
                    "Re-export the config with the current schema."
                )
        llm = ScopeConfig(**{k: v for k, v in llm_raw.items() if k in llm_fields})
        dit = ScopeConfig(**{k: v for k, v in dit_raw.items() if k in llm_fields})
        top = {k: v for k, v in d.items() if k not in ("llm", "dit")}
        known = {f.name for f in QVLAConfig.__dataclass_fields__.values()}
        unknown_top = set(top) - known
        if unknown_top:
            raise ValueError(
                f"Unknown QVLAConfig field(s): {sorted(unknown_top)}. "
                "Re-export the config with the current schema."
            )
        return QVLAConfig(
            llm=llm,
            dit=dit,
            **{k: v for k, v in top.items() if k in known},
        )

    def with_overrides(self, **kwargs) -> "QVLAConfig":
        return replace(self, **kwargs)

    @property
    def needs_fisher(self) -> bool:
        """True when any scope needs original-space Fisher for rotation fitting.

        Covers ``perm_score=fisher`` (and any Fisher-weighted SVD that reuses
        that same early pass). Fisher-GPTQ is separate: it runs a second pass
        after rotations are fitted, gated by ``ScopeConfig.fisher_gptq``.
        """
        return (
            ("perm" in self.llm.pipeline and self.llm.perm_score == "fisher")
            or ("perm" in self.dit.pipeline and self.dit.perm_score == "fisher")
            or (self.llm.smooth_enabled and float(self.llm.smooth_fisher_beta) != 0.0)
            or (self.dit.smooth_enabled and float(self.dit.smooth_fisher_beta) != 0.0)
        )


__all__ = [
    "ActOutlierFitTokens",
    "ActPercentileMode",
    "ActScaleGranularity",
    "ActScaleMode",
    "CalibrationStepAggregation",
    "DEFAULT_PIPELINE",
    "FisherMethod",
    "FisherType",
    "PermScore",
    "PipelineStep",
    "QVLAConfig",
    "ScopeConfig",
    "StepAggregation",
    "SvdSource",
    "QuantFormat",
    "WeightQuantizer",
]
