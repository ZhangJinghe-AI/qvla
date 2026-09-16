"""QVLA — W4A4 post-training quantization for VLA models.

Public surface kept small and stable:

* :class:`QVLAConfig`           — the one config dataclass.
* :func:`enable_quantization`        — swap layers in-place from a loaded pack.
* :func:`build_pack`                 — offline calibration + pack builder.
* :class:`QuantLinear`          — the runtime layer (rarely imported by users).
* :class:`ModelAdapter`              — interface for adding new model families.
* :mod:`qvla.adapters.pi05`    — concrete pi0.5 adapter.
* :mod:`qvla.adapters.groot`   — Groot stub.

Everything else is internal. Importing this top-level module is side-effect free
(no CUDA, no torch.compile, no fixture downloads).
"""

from __future__ import annotations

from qvla.build import build_pack, compute_fisher_sensitivity
from qvla.build.fisher import InputGradFisherCollector
from qvla.config import QVLAConfig
from qvla.runtime import QuantLinear, enable_quantization

__all__ = [
    "InputGradFisherCollector",
    "QVLAConfig",
    "QuantLinear",
    "build_pack",
    "compute_fisher_sensitivity",
    "enable_quantization",
]

__version__ = "0.1.0"
