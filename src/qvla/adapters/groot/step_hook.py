"""Denoising-step callback wiring for GR00T-N1.7 action head activation collection."""

from __future__ import annotations

import contextlib


@contextlib.contextmanager
def patched_action_head_denoise(action_head_runner, step_callback):
    """Wrap the action head forward loop to invoke ``step_callback(step)`` each denoise step.

    GR00T-N1.7 denoises in a loop inside the action head runner's ``_fwd_loop`` method.
    We intercept the ``denoise_step`` method on the action_head to fire the callback.

    The call counter wraps modulo ``num_inference_timesteps`` so one context can
    span multiple forwards (e.g. the Fisher sample loop): each full denoise loop
    yields step indices ``0 .. num_steps-1`` again.
    """
    action_head = action_head_runner.model.action_head
    original_denoise_step = action_head.denoise_step
    num_steps = int(action_head.num_inference_timesteps)
    if num_steps < 1:
        raise RuntimeError(
            f"GR00T-N1.7 num_inference_timesteps must be >= 1, got {num_steps}."
        )

    _calls = [0]

    def _wrapper(*args, **kwargs):
        step_callback(int(_calls[0] % num_steps))
        result = original_denoise_step(*args, **kwargs)
        _calls[0] += 1
        return result

    action_head.denoise_step = _wrapper
    try:
        yield _calls
    finally:
        action_head.denoise_step = original_denoise_step


def find_action_head_runner(engine):
    """Locate the GR00TN17ActionHeadRunner on the scheduler."""
    sched = getattr(engine.entry, "scheduler", None)
    if sched is None:
        raise RuntimeError("GR00T-N1.7 scheduler not built; call build_model() first.")
    runner = getattr(sched, "action_head_runner", None)
    if runner is not None:
        return runner
    raise RuntimeError(
        "Could not locate GR00TN17ActionHeadRunner on the scheduler. "
        "The phyai scheduler API changed; update adapters.groot.step_hook."
    )
