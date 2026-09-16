"""Fisher / policy-aware sensitivity subsystem.

Two orthogonal knobs drive the layout of this subpackage:

* **``fisher_type``** — *what* the sensitivity is computed *from*:

  - ``"input_grad"``  — the historical QVLA behaviour. For each action
    dimension :math:`a_i`, aggregate :math:`(\\partial a_i/\\partial x)^2`
    over the non-channel axes to a per-input-channel score. Implemented
    by :class:`InputGradFisherCollector`.
  - ``"output_hessian"`` — the HBVLA-style policy-aware rectified
    Hessian. For each action dimension, back-prop to the *output*
    activation ``y = Wx``, aggregate over output channels to get a
    per-token importance ``s_t``, then form the input-side Hessian
    diagonal ``F_c = Σ_t s_t · x_{t,c}^2`` — a token-weighted
    energy of each input channel. Implemented by
    :class:`OutputHessianFisherCollector`.

* **``fisher_method``** — *how* the outer expectation over action
  dimensions is approximated:

  - ``"exact"``       — one backward pass per action dim (default).
  - ``"hutchinson"``  — Rademacher probes, unbiased estimator of the
    same quantity in fewer backwards.

The two knobs are independent — a collector implementation is free to
support either estimator or explicitly refuse one. Both current
implementations support both.
"""

from qvla.build.fisher.collectors import (
    BaseFisherCollector,
    InputGradFisherCollector,
    OutputHessianFisherCollector,
)
from qvla.build.fisher.common import (
    FisherMethod,
    FisherType,
    StepAggregation,
    normalize_fisher_sensitivity,
    resolve_fisher_action_dim,
    select_fisher_actions,
)
from qvla.build.fisher.driver import compute_fisher_sensitivity
from qvla.build.fisher.result import LayerFisherResult

__all__ = [
    "BaseFisherCollector",
    "FisherMethod",
    "FisherType",
    "InputGradFisherCollector",
    "LayerFisherResult",
    "OutputHessianFisherCollector",
    "StepAggregation",
    "compute_fisher_sensitivity",
    "normalize_fisher_sensitivity",
    "resolve_fisher_action_dim",
    "select_fisher_actions",
]
