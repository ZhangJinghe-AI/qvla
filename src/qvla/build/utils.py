"""Build-time helpers: process memory, crash diagnostics, collector placement."""

from __future__ import annotations

import logging
import os
import signal

import torch
import torch.nn as nn

from qvla.build.collector import AmaxCollectPlan

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


# Packed LLM / prefix-KV length when the engine layout is not available.
# pi0.5: ~256 PaliGemma image tokens + short language/pad. GR00T: Qwen3-VL prefix.
_LLM_CLIP_TOKENS = {"pi05": 384, "groot_n17": 512}
# GR00T cross-attn K/V keep one concat of VL tokens (not per denoise step).
_DIT_PREFIX_KV_TOKENS = {"pi05": 384, "groot_n17": 512}


def clip_buffer_tokens_per_sample(
    *,
    model_kind: str | None,
    name: str,
    scope: str,
    chunk_size: int | None,
    num_steps: int,
    fallback: int = 512,
) -> int:
    """How many fp16 rows one layer keeps for adaptive clip, per calib sample.

    DiT denoise-loop linears buffer each Euler step separately, so this is
    ``seq * num_steps``. Prefix-only K/V and LLM prefixes are a single concat.
    Unknown ``model_kind`` keeps ``fallback`` (legacy 512).
    """
    if model_kind not in _LLM_CLIP_TOKENS:
        return fallback
    if scope == "llm":
        return _LLM_CLIP_TOKENS[model_kind]
    steps = max(int(num_steps), 1)
    if name.endswith("to_k") or name.endswith("to_v"):
        return _DIT_PREFIX_KV_TOKENS[model_kind]
    if chunk_size is None or int(chunk_size) < 1:
        return fallback * steps
    seq = int(chunk_size) + (1 if model_kind == "groot_n17" else 0)
    return seq * steps


def estimate_adaptive_clip_bytes(
    targets: list[tuple[str, str, nn.Module]],
    num_samples: int,
    tokens_per_sample: int = 512,
    *,
    model_kind: str | None = None,
    chunk_size: int | None = None,
    num_steps_by_scope: dict[str, int] | None = None,
) -> int:
    """Estimate bytes for adaptive clip tip buffers (fp16, all tokens kept)."""
    total = 0
    steps_map = num_steps_by_scope or {}
    for name, scope, mod in targets:
        d = int(getattr(mod, "in_features"))
        n_tok = clip_buffer_tokens_per_sample(
            model_kind=model_kind,
            name=name,
            scope=scope,
            chunk_size=chunk_size,
            num_steps=int(steps_map.get(scope, 1)),
            fallback=tokens_per_sample,
        )
        total += num_samples * n_tok * d * 2
    return total


def choose_collector_device(
    targets: list[tuple[str, str, nn.Module]],
    model_device: torch.device,
    *,
    amax_plan: AmaxCollectPlan,
    num_samples: int = 1,
    tokens_per_sample: int = 512,
    headroom_gib: float = 5.0,
    model_kind: str | None = None,
    chunk_size: int | None = None,
    num_steps_by_scope: dict[str, int] | None = None,
) -> torch.device:
    """Prefer GPU when free VRAM can hold collector buffers; else CPU."""
    if model_device.type != "cuda":
        return torch.device("cpu")

    buffer_bytes = 0
    if amax_plan.collect_hessian:
        buffer_bytes += estimate_xtx_bytes(targets)
    if amax_plan.collect_adaptive_inner_channel:
        buffer_bytes += estimate_adaptive_clip_bytes(
            targets,
            num_samples,
            tokens_per_sample,
            model_kind=model_kind,
            chunk_size=chunk_size,
            num_steps_by_scope=num_steps_by_scope,
        )
    buffer_gib = buffer_bytes / (1 << 30)

    try:
        torch.cuda.empty_cache()
        free_bytes, _ = torch.cuda.mem_get_info(model_device)
    except Exception:
        logger.info("Cannot query GPU memory; falling back to CPU collector.")
        return torch.device("cpu")
    free_gib = free_bytes / (1 << 30)
    if buffer_gib + headroom_gib < free_gib:
        logger.info(
            "Collector on GPU (buffers ≈ %.1f GiB, free ≈ %.1f GiB, headroom %.1f GiB).",
            buffer_gib, free_gib, headroom_gib,
        )
        return model_device
    logger.info(
        "Buffers %.1f GiB + headroom %.1f GiB > GPU free %.1f GiB; using CPU collector.",
        buffer_gib, headroom_gib, free_gib,
    )
    return torch.device("cpu")


__all__ = [
    "rss_gb",
    "log_memory",
    "install_crash_handler",
    "estimate_xtx_bytes",
    "clip_buffer_tokens_per_sample",
    "estimate_adaptive_clip_bytes",
    "choose_collector_device",
]
