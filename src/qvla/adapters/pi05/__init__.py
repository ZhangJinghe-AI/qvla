"""pi0.5 model adapter (phyai Engine + calibration hooks).

Submodules:

* :mod:`config` — :class:`PI05AdapterConfig`, checkpoint I/O, dtype helpers
* :mod:`obs` — observation dict → :class:`PI05Request`
* :mod:`processor` — :class:`PI05Processor` setup
* :mod:`engine` — eager :class:`Engine` for pack build
* :mod:`calibration` — synthetic / file batch iterators
* :mod:`step_hook` — per-Euler-step callback for the collector
"""

from qvla.adapters.pi05.adapter import PI05Adapter
from qvla.adapters.pi05.config import (
    CalibrationSource,
    PI05AdapterConfig,
)

__all__ = [
    "CalibrationSource",
    "PI05Adapter",
    "PI05AdapterConfig",
]
