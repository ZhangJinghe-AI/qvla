"""Per-layer auto-incrementing step counter for the DiT per-step act-scale table.

Why a counter at all
--------------------
QVLA's per-step recipe stores one activation scale row per Euler denoise
step. The runtime layer needs to know "which step am I in?" to index its
``act_scale_table``. The cleanest way to bridge that without touching phyai's
runner is to let each layer count its own calls — since each linear in the
expert is called *exactly once* per Euler step, the counter modulo
``num_steps`` *is* the step index.

Why this also works under CUDA graphs
-------------------------------------
phyai's :class:`PI05ExpertRunner._fwd_loop` unrolls the entire Euler loop
(``for step in range(num_steps)``) into a single captured CUDA graph. At
**graph capture time** the Python interpreter walks the loop, so each layer's
forward is actually invoked ``num_steps`` times, with the counter going
``0, 1, 2, ..., num_steps-1``. Each step's traced operations therefore index a
*different fixed slice* into ``act_scale_table``. The slice indices are baked
into the graph; replay just rewires the input data.

At replay time the Python counter doesn't increment (the Python forward is
skipped — only CUDA ops replay), but that's fine: the captured ops already
encode the right per-step scales.

Eager mode (cuda_graph=False) also works: the counter increments on every
forward call, wraps modulo ``num_steps``, and stays correct as long as each
inference runs an integer number of full Euler loops.

Reset utility
-------------
For belt-and-braces correctness (e.g. during pack-builder calibration where
you might call the layer outside of an Euler loop) call :func:`reset_step_counters`
before the loop. It walks a model and resets every ``QuantLinear`` in it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:  # avoid cycle at import time
    from qvla.runtime.quant_linear import QuantLinear


def reset_step_counters(model: nn.Module) -> int:
    """Reset every :class:`QuantLinear`'s call counter to 0.

    Returns the number of layers reset so the caller can sanity-check that the
    walk hit something.
    """
    # Local import avoids the cycle at top level.
    from qvla.runtime.quant_linear import QuantLinear

    n = 0
    for m in model.modules():
        if isinstance(m, QuantLinear):
            m.reset_step_counter()
            n += 1
    return n


__all__ = ["reset_step_counters"]
