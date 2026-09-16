"""GR00T-N1.7 model adapter (phyai Engine + calibration hooks).

Submodules:

* :mod:`config` — :class:`GR00TAdapterConfig`, checkpoint I/O, dtype helpers
* :mod:`obs` — observation dict → :class:`GR00TN17Request`
* :mod:`processor` — :class:`GR00TProcessor` setup
* :mod:`engine` — eager :class:`Engine` for pack build
* :mod:`calibration` — synthetic / file batch iterators
* :mod:`step_hook` — per-denoise-step callback for the collector
"""

from qvla.adapters.groot.adapter import GR00TAdapter
from qvla.adapters.groot.config import (
    CalibrationSource,
    GR00TAdapterConfig,
)

__all__ = [
    "CalibrationSource",
    "GR00TAdapter",
    "GR00TAdapterConfig",
]
