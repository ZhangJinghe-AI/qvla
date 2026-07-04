"""Per-model adapters.

Lazy imports — touching an adapter pulls in that model's specific dependencies
(e.g. ``phyai``, ``transformers``). Top-level ``from qvla
import QVLAConfig`` stays dependency-free.
"""

from __future__ import annotations

from typing import Callable

from qvla.adapters.base import ModelAdapter

_ADAPTER_REGISTRY: dict[str, Callable[..., ModelAdapter]] = {}


def register_adapter(
    model_kind: str,
    factory: Callable[..., ModelAdapter] | None = None,
) -> Callable[..., ModelAdapter] | Callable[[Callable[..., ModelAdapter]], Callable[..., ModelAdapter]]:
    """Register an adapter factory: ``@register_adapter("pi05")`` or direct call."""

    def _register(fn: Callable[..., ModelAdapter]) -> Callable[..., ModelAdapter]:
        _ADAPTER_REGISTRY[model_kind] = fn
        return fn

    if factory is not None:
        return _register(factory)
    return _register


@register_adapter("pi05")
def _load_pi05(**kwargs) -> ModelAdapter:
    from qvla.adapters.pi05 import PI05Adapter, PI05AdapterConfig

    return PI05Adapter(PI05AdapterConfig.from_checkpoint(**kwargs))


def get_adapter(model_kind: str, **kwargs) -> ModelAdapter:
    """Factory: lazy-import and instantiate the named adapter."""
    try:
        factory = _ADAPTER_REGISTRY[model_kind]
    except KeyError as exc:
        known = ", ".join(sorted(_ADAPTER_REGISTRY))
        raise ValueError(
            f"Unknown model_kind={model_kind!r}. Built-in adapters: {known}. "
            "Register a custom one with @register_adapter(...)."
        ) from exc
    return factory(**kwargs)


__all__ = ["ModelAdapter", "get_adapter", "register_adapter"]
