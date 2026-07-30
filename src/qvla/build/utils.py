"""Build-time helpers: process memory, crash diagnostics, collector placement."""

from __future__ import annotations

import logging
import os
import signal

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ── Process memory / OOM diagnostics ─────────────────────────────────────────

def rss_gb() -> float:
    """Current process RSS in GiB (Linux). Returns -1.0 if unavailable."""
    try:
        with open(f"/proc/{os.getpid()}/statm") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / (1 << 30)
    except Exception:
        return -1.0


def log_memory(label: str) -> None:
    """Log process RSS; WARNING when above ~60 GiB."""
    rss = rss_gb()
    if rss < 0:
        return
    level = logging.WARNING if rss > 60.0 else logging.INFO
    logger.log(level, "%s — process RSS: %.1f GiB", label, rss)


def install_crash_handler() -> None:
    """Install SIGTERM/SIGABRT handlers that log RSS before exiting.

    Helps diagnose silent OOM kills (Linux OOM killer sends SIGKILL which
    cannot be caught; SIGTERM may still appear in some environments).
    """
    def _handler(signum, frame):
        sig_name = signal.Signals(signum).name
        rss = rss_gb()
        logger.critical(
            "Received %s (RSS=%.1f GiB). Likely OOM-killed by the OS. "
            "Try reducing --num-samples or --noise-ensemble-k, or free system memory.",
            sig_name,
            rss,
        )
        raise SystemExit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGABRT):
        try:
            signal.signal(sig, _handler)
        except (OSError, ValueError):
            pass


# ── Collector device selection (GPU vs CPU for XᵀX) ──────────────────────────

def estimate_xtx_bytes(
    targets: list[tuple[str, str, nn.Module]],
) -> int:
    """Total bytes for all fp32 ``xtx`` matrices."""
    total = 0
    for _name, _scope, mod in targets:
        d = int(getattr(mod, "in_features"))
        total += d * d * 4
    return total


def choose_collector_device(
    targets: list[tuple[str, str, nn.Module]],
    model_device: torch.device,
    *,
    headroom_gib: float = 5.0,
) -> torch.device:
    """Prefer GPU for Hessian accumulation when free VRAM allows; else CPU."""
    if model_device.type != "cuda":
        return torch.device("cpu")
    xtx_gib = estimate_xtx_bytes(targets) / (1 << 30)
    try:
        # Reclaim caching-allocator pools from prior passes before measuring.
        torch.cuda.empty_cache()
        free_bytes, _ = torch.cuda.mem_get_info(model_device)
    except Exception:
        logger.info("Cannot query GPU memory; falling back to CPU collector.")
        return torch.device("cpu")
    free_gib = free_bytes / (1 << 30)
    if xtx_gib + headroom_gib < free_gib:
        logger.info(
            "Collector on GPU (xtx ≈ %.1f GiB, free ≈ %.1f GiB, headroom %.1f GiB).",
            xtx_gib, free_gib, headroom_gib,
        )
        return model_device
    logger.info(
        "xtx %.1f GiB + headroom %.1f GiB > GPU free %.1f GiB; using CPU collector.",
        xtx_gib, headroom_gib, free_gib,
    )
    return torch.device("cpu")


__all__ = [
    "rss_gb",
    "log_memory",
    "install_crash_handler",
    "estimate_xtx_bytes",
    "choose_collector_device",
]
