"""Walk-and-replace logic.

The public entrypoint is :func:`enable_quantization`. It:

1. Walks ``model.named_modules()`` and finds every Linear-shaped module whose
   qualified name matches one of the pack's layer entries (using the pack's
   embedded :class:`QVLAConfig`).
2. Replaces the original module with an :class:`QuantLinear` carrying that
   layer's quant payload.
3. Returns the list of replaced names so the caller can log / verify them.

The replacement is *framework-agnostic*: we recognize both
``torch.nn.Linear`` and any of phyai's ``LinearBase`` subclasses. The
``return_tuple`` flag — which controls whether the layer returns ``y`` or
``(y, bias)`` — is auto-detected from the original module's class.

We intentionally do not touch any state the host framework cares about — no
hooks added, no parameter renaming, no class-replacement. The result is that
the host model's forward graph topology is identical; only the leaf module's
implementation changed.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable

import torch.nn as nn

from qvla.config import QVLAConfig, ScopeConfig
from qvla.core.pack import LayerPack, Pack
from qvla.runtime.quant_linear import QuantLinear


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Auto-detection helpers                                                      #
# --------------------------------------------------------------------------- #


def _is_linear_like(mod: nn.Module) -> bool:
    """True for ``nn.Linear`` and any phyai linear (``LinearBase`` subclass).

    We deliberately don't try to import phyai here — that would couple this
    package to phyai's specific layout. Instead we duck-type on the attributes
    every linear-like layer exposes.
    """
    if isinstance(mod, nn.Linear):
        return True
    has_shapes = (
        hasattr(mod, "in_features") and hasattr(mod, "out_features")
        and hasattr(mod, "weight")
    )
    # phyai linears also expose a ``spec`` attribute and a ``forward``
    # returning a 2-tuple; ``in_features``/``out_features`` plus a 2-D
    # ``weight`` parameter is the safer minimal contract.
    if not has_shapes:
        return False
    w = getattr(mod, "weight", None)
    return w is not None and getattr(w, "ndim", 0) == 2


def _returns_tuple(mod: nn.Module) -> bool:
    """phyai's ``LinearBase`` subclasses return ``(y, bias)``; nn.Linear returns ``y``.

    ``GR00TN17Linear`` subclasses ``ReplicatedLinear`` (so it has
    ``skip_bias_add``) but overrides ``forward`` to return a plain tensor —
    same contract as ``nn.Linear``. Must be checked before the phyai heuristic.
    """
    if type(mod).__name__ == "GR00TN17Linear" or isinstance(mod, nn.Linear):
        return False
    if hasattr(mod, "skip_bias_add"):
        return True
    cls_name = type(mod).__name__
    return cls_name.endswith("Linear") and cls_name != "Linear"


def _skip_bias_add(mod: nn.Module) -> bool:
    return bool(getattr(mod, "skip_bias_add", False))


def _set_submodule(root: nn.Module, name: str, new: nn.Module) -> None:
    """Set ``root.<qualified_name> = new`` without touching the rest of the tree."""
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


# --------------------------------------------------------------------------- #
# Regex matching                                                              #
# --------------------------------------------------------------------------- #


def _matches_scope(name: str, scope: ScopeConfig) -> bool:
    if not scope.include_regex:
        return False
    if not re.search(scope.include_regex, name):
        return False
    if scope.exclude_regex and re.search(scope.exclude_regex, name):
        return False
    return True


def classify(name: str, config: QVLAConfig) -> str | None:
    """Return ``"llm"`` / ``"dit"`` / ``None`` depending on which scope claims ``name``.

    LLM takes precedence (we check it first). The pi05 / Groot regexes are
    written to be disjoint, but this guards against an accidentally
    overlapping user config.
    """
    if _matches_scope(name, config.llm):
        return "llm"
    if _matches_scope(name, config.dit):
        return "dit"
    return None


def list_target_modules(
    model: nn.Module, config: QVLAConfig
) -> list[tuple[str, str, nn.Module]]:
    """Walk ``model`` and return ``(name, scope, module)`` for every match.

    Exposed publicly so the builder can iterate the same set without
    re-running the regex matcher.
    """
    out: list[tuple[str, str, nn.Module]] = []
    for name, mod in model.named_modules():
        if not _is_linear_like(mod):
            continue
        scope = classify(name, config)
        if scope is None:
            continue
        out.append((name, scope, mod))
    return out


# --------------------------------------------------------------------------- #
# Public entrypoint                                                           #
# --------------------------------------------------------------------------- #


def _build_replacement(
    pack_entry: LayerPack,
    original: nn.Module,
    device,
    output_dtype,
    nvfp_activation_num_samples: int | None,
) -> QuantLinear:
    return QuantLinear(
        pack_entry,
        return_tuple=_returns_tuple(original),
        skip_bias_add=_skip_bias_add(original),
        output_dtype=output_dtype,
        device=device,
        nvfp_activation_num_samples=nvfp_activation_num_samples,
    )


def enable_quantization(
    model: nn.Module,
    pack: Pack,
    *,
    device: str | None = None,
    output_dtype=None,
    strict: bool = True,
    nvfp_activation_num_samples: int | None = None,
) -> list[str]:
    """Swap every matched linear in ``model`` with an :class:`QuantLinear`.

    Args:
        model: the host model. We walk ``model.named_modules()`` for matches.
        pack: a loaded :class:`Pack` (call :meth:`Pack.load` upstream).
        device: where the new buffers live. ``None`` defaults to the device of
            the first replaced module's weight.
        output_dtype: dtype of the GEMM output. ``None`` defaults to the
            host's params dtype (read from the first replaced module's weight).
        strict: if True, raise when a pack layer has no matching module in the
            model (or vice versa, when a model module matches the regex but is
            absent from the pack).
        nvfp_activation_num_samples: Number of equal, contiguous samples in
            every NVFP4 activation tensor. Required when the pack quantizes
            activations as NVFP4.

    Returns:
        The list of qualified names that were replaced.
    """
    assert isinstance(pack, Pack)
    cfg = pack.config

    # Collect target modules from the host model.
    target_modules = list_target_modules(model, cfg)
    target_by_name = {name: (scope, m) for name, scope, m in target_modules}

    # Cross-check against the pack.
    pack_names = set(pack.layers.keys())
    model_names = set(target_by_name.keys())
    missing_in_pack = sorted(model_names - pack_names)
    missing_in_model = sorted(pack_names - model_names)
    if missing_in_pack and strict:
        raise RuntimeError(
            f"{len(missing_in_pack)} module(s) matched the config regex but "
            f"are absent from the pack. First few: {missing_in_pack[:5]}. "
            "Either rebuild the pack or pass strict=False."
        )
    if missing_in_model and strict:
        raise RuntimeError(
            f"{len(missing_in_model)} pack layer(s) have no matching module "
            f"in this model. First few: {missing_in_model[:5]}. "
            "Either rebuild the pack against this model or pass strict=False."
        )

    # Pick a device / dtype from the first match if none supplied.
    if target_modules:
        ref_weight = getattr(target_modules[0][2], "weight")
        if device is None:
            device = ref_weight.device
        if output_dtype is None:
            output_dtype = ref_weight.dtype

    replaced: list[str] = []
    for name, scope, original in target_modules:
        if name not in pack.layers:
            logger.debug("Skipping %s (no pack entry, strict=False).", name)
            continue
        entry = pack.layers[name]
        if entry.scope != scope:
            logger.warning(
                "Pack entry %s has scope=%s but model regex classified it as %s. "
                "Using the pack's scope.", name, entry.scope, scope,
            )
        # Build replacement and slot it in.
        new_mod = _build_replacement(
            entry,
            original,
            device,
            output_dtype,
            nvfp_activation_num_samples,
        )
        _set_submodule(model, name, new_mod)
        replaced.append(name)

    logger.info(
        "QVLA: replaced %d / %d linear layers (pack=%s, model_kind=%s).",
        len(replaced), len(target_modules), pack.summary().splitlines()[0], cfg.model_kind,
    )
    return replaced


def list_replaced(model: nn.Module) -> Iterable[tuple[str, QuantLinear]]:
    """Iterate ``(name, QuantLinear)`` for every quantized layer in ``model``."""
    for name, mod in model.named_modules():
        if isinstance(mod, QuantLinear):
            yield name, mod


__all__ = ["classify", "enable_quantization", "list_replaced", "list_target_modules"]
