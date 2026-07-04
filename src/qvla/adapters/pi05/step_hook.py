"""Euler-step callback wiring for activation collection."""

from __future__ import annotations

import contextlib


@contextlib.contextmanager
def patched_one_step(runner, step_callback):
    """Wrap ``runner._one_step`` to invoke ``step_callback(step)`` each Euler step."""
    original = runner._one_step

    def _wrapper(x_t, step):
        step_callback(int(step))
        return original(x_t, step)

    runner._one_step = _wrapper
    try:
        yield
    finally:
        runner._one_step = original


def find_expert_runner(engine):
    sched = getattr(engine.entry, "scheduler", None)
    if sched is None:
        raise RuntimeError("pi05 scheduler not built; call build_model() first.")
    for attr in ("expert_runner", "_expert_runner", "expert", "_expert"):
        runner = getattr(sched, attr, None)
        if runner is not None:
            return runner
    for name in dir(sched):
        value = getattr(sched, name, None)
        if value is not None and type(value).__name__ == "PI05ExpertRunner":
            return value
    raise RuntimeError(
        "Could not locate PI05ExpertRunner on the scheduler. "
        "The phyai scheduler API changed; update adapters.pi05.step_hook."
    )
