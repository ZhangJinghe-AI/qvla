#!/usr/bin/env python
r"""Characterize massive vs normal (shoulder) outliers on DiT activations.

Uses the same selective clip rule as quantization / the causal MAD-band
experiment:

    MAD-select large-amax channels
    massive        = |x| > μ+kσ on those channels   (what clip removes)
    normal-outlier = same channels, in [MAD floor, μ+kσ]  (shoulder)
    rest           = everything else

GR00T-N1.7 fits the rule on the 40 action tokens and never counts the
leading state token as clip-massive/shoulder (skip_first). Token-index
plots still show state / valid-16 / pad-24 so we can see where the mass
sits. pi0.5 uses every action token.

Example::

    CUDA_VISIBLE_DEVICES=5 HF_ENDPOINT=https://hf-mirror.com uv run python \
      tools/analyze_dit_outlier_types.py \
      --model groot_n17 \
      --checkpoint /data/share/GR00T-N1.7-LIBERO/libero_goal \
      --calibration-data ../calibration_data/libero_goal_30_7_demo.npz \
      --samples 0,1,2,3 --noise-seeds 0,1 \
      --output-dir tools/img/dit_outlier_types_groot
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
_TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_TOOLS))

import analyze_dit_matched_clip_action_impact as matched  # noqa: E402
import analyze_dit_outlier_action_detail as detail  # noqa: E402
from qvla.config import QVLAConfig  # noqa: E402


TOKEN_GROUPS = ("state", "action_valid", "action_pad", "action")


@dataclass(frozen=True)
class SiteRow:
    sample: int
    noise: int
    step: int
    layer: str
    kind: str
    n_tokens: int
    n_channels: int
    n_selected: int
    l1_all: float
    l1_massive: float
    l1_shoulder: float
    n_massive: int
    n_shoulder: int
    n_tokens_with_massive: int
    n_tokens_with_shoulder: int
    median_selected: float
    mean_massive: float
    mean_shoulder: float
    state_would_clip_l1: float
    state_would_clip_n: int


@dataclass
class TokenIndexAcc:
    n_tokens: int
    n_steps: int
    massive_l1: np.ndarray
    shoulder_l1: np.ndarray
    all_l1: np.ndarray
    massive_n: np.ndarray
    raw_over_l1: np.ndarray
    raw_over_n: np.ndarray

    @classmethod
    def zeros(cls, n_tokens: int, n_steps: int) -> "TokenIndexAcc":
        shape = (n_steps, n_tokens)
        return cls(
            n_tokens=n_tokens,
            n_steps=n_steps,
            massive_l1=np.zeros(shape, dtype=np.float64),
            shoulder_l1=np.zeros(shape, dtype=np.float64),
            all_l1=np.zeros(shape, dtype=np.float64),
            massive_n=np.zeros(shape, dtype=np.float64),
            raw_over_l1=np.zeros(shape, dtype=np.float64),
            raw_over_n=np.zeros(shape, dtype=np.float64),
        )

    def add(
        self,
        step: int,
        values: torch.Tensor,
        massive: torch.Tensor,
        shoulder: torch.Tensor,
        state_would: torch.Tensor | None = None,
    ) -> None:
        vals = values.detach().to(dtype=torch.float64, device="cpu").numpy()
        m = massive.detach().to(dtype=torch.bool, device="cpu").numpy()
        s = shoulder.detach().to(dtype=torch.bool, device="cpu").numpy()
        raw = m.copy()
        if state_would is not None:
            raw[0] = state_would.detach().to(dtype=torch.bool, device="cpu").numpy()
        self.all_l1[step] += vals.sum(axis=1)
        self.massive_l1[step] += np.where(m, vals, 0.0).sum(axis=1)
        self.shoulder_l1[step] += np.where(s, vals, 0.0).sum(axis=1)
        self.massive_n[step] += m.sum(axis=1)
        self.raw_over_l1[step] += np.where(raw, vals, 0.0).sum(axis=1)
        self.raw_over_n[step] += raw.sum(axis=1)


def _token_group_names(
    n_tokens: int, *, skip_first: bool, valid_action_horizon: int | None
) -> list[str]:
    if n_tokens < 1:
        raise ValueError(f"n_tokens must be >= 1, got {n_tokens}.")
    names: list[str] = []
    for index in range(n_tokens):
        if skip_first and index == 0:
            names.append("state")
            continue
        if skip_first and valid_action_horizon is not None:
            action_index = index - 1
            names.append(
                "action_valid" if action_index < int(valid_action_horizon) else "action_pad"
            )
            continue
        names.append("action")
    return names


def _outlier_masks(
    live: torch.Tensor,
    std_k: float,
    *,
    skip_first_token: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(values, selected, massive, shoulder, state_would_clip)``."""
    values, selected, bounds, massive = detail._outlier_over_and_bounds(
        live, std_k, skip_first_token=skip_first_token
    )
    leftover = (~massive) & selected.view(1, -1) & (values > 0.0)
    if skip_first_token:
        leftover = leftover.clone()
        leftover[0] = False
        mad_src = values[1:]
        state_would = (values[0] > bounds) & selected
    else:
        mad_src = values
        state_would = torch.zeros_like(selected)
    mad_lo = detail._channel_mad_threshold(mad_src)
    shoulder = leftover & (values >= mad_lo.view(1, -1)) & (values <= bounds.view(1, -1))
    return values, selected, massive, shoulder, state_would


def _site_row(
    live: torch.Tensor,
    *,
    sample: int,
    noise: int,
    step: int,
    layer: str,
    kind: str,
    std_k: float,
    skip_first_token: bool,
) -> tuple[SiteRow, torch.Tensor, torch.Tensor, torch.Tensor]:
    values, selected, massive, shoulder, state_would = _outlier_masks(
        live, std_k, skip_first_token=skip_first_token
    )
    l1_all = float(values.sum().item())
    selected_vals = values[:, selected]
    median_selected = (
        float(selected_vals.median().item()) if int(selected_vals.numel()) > 0 else 0.0
    )
    massive_vals = values[massive]
    shoulder_vals = values[shoulder]
    row = SiteRow(
        sample=sample,
        noise=noise,
        step=step,
        layer=layer,
        kind=kind,
        n_tokens=int(values.shape[0]),
        n_channels=int(values.shape[1]),
        n_selected=int(selected.sum().item()),
        l1_all=l1_all,
        l1_massive=float(massive_vals.sum().item()) if massive_vals.numel() else 0.0,
        l1_shoulder=float(shoulder_vals.sum().item()) if shoulder_vals.numel() else 0.0,
        n_massive=int(massive.sum().item()),
        n_shoulder=int(shoulder.sum().item()),
        n_tokens_with_massive=int(massive.any(dim=1).sum().item()),
        n_tokens_with_shoulder=int(shoulder.any(dim=1).sum().item()),
        median_selected=median_selected,
        mean_massive=float(massive_vals.mean().item()) if massive_vals.numel() else 0.0,
        mean_shoulder=float(shoulder_vals.mean().item()) if shoulder_vals.numel() else 0.0,
        state_would_clip_l1=(
            float((values[0, state_would]).sum().item()) if skip_first_token else 0.0
        ),
        state_would_clip_n=int(state_would.sum().item()) if skip_first_token else 0,
    )
    return row, values, massive, shoulder, state_would


def _capture_and_classify(
    adapter,
    request,
    layers: list[tuple[str, torch.nn.Module]],
    *,
    num_steps: int,
    n_tokens: int,
    std_k: float,
    skip_first_token: bool,
    sample: int,
    noise: int,
    token_acc: TokenIndexAcc | None = None,
) -> list[SiteRow]:
    in_features = {}
    for name, layer in layers:
        weight = getattr(layer, "weight", None)
        if not torch.is_tensor(weight) or weight.ndim != 2:
            raise RuntimeError(f"{name} must have a 2-D tensor weight.")
        in_features[name] = int(weight.shape[1])
    current_step: list[int | None] = [None]
    callbacks: list[int | None] = []
    rows: list[SiteRow] = []
    seen: dict[str, set[int]] = {name: set() for name, _ in layers}

    def make_hook(name: str):
        width = in_features[name]
        kind = matched._layer_kind(name)

        def hook(_module, inputs):
            step = current_step[0]
            if step is None:
                raise RuntimeError(
                    f"{name} ran outside the denoise loop; prefix-only K/V "
                    "linears must be skipped."
                )
            if len(inputs) != 1 or not torch.is_tensor(inputs[0]):
                raise RuntimeError(f"{name} must receive exactly one tensor input.")
            if step in seen[name]:
                raise RuntimeError(f"{name} ran more than once at step {step}.")
            seen[name].add(int(step))
            x = inputs[0]
            if int(x.shape[-1]) != width:
                raise RuntimeError(
                    f"{name} input width {x.shape[-1]} != in_features={width}."
                )
            flat = x.reshape(-1, width)
            if int(flat.shape[0]) != n_tokens:
                raise RuntimeError(
                    f"{name}: expected {n_tokens} DiT tokens, got {flat.shape[0]}."
                )
            live = detail._exact_fp32(flat)
            row, values, massive, shoulder, state_would = _site_row(
                live,
                sample=sample,
                noise=noise,
                step=int(step),
                layer=name,
                kind=kind,
                std_k=std_k,
                skip_first_token=skip_first_token,
            )
            if token_acc is not None:
                token_acc.add(
                    int(step),
                    values,
                    massive,
                    shoulder,
                    state_would=state_would,
                )
            rows.append(row)
            return None

        return hook

    def step_callback(step: int | None) -> None:
        value = None if step is None else int(step)
        current_step[0] = value
        callbacks.append(value)

    handles = [layer.register_forward_pre_hook(make_hook(name)) for name, layer in layers]
    try:
        with matched._with_denoise_callback(adapter, step_callback):
            step_callback(None)
            with torch.inference_mode():
                actions = adapter.engine.step(request)
        del actions
    finally:
        for handle in handles:
            handle.remove()
    expected = [None, *range(num_steps)]
    if callbacks != expected:
        raise RuntimeError(f"Denoise callback order {callbacks} != {expected}.")
    for name, _layer in layers:
        if seen[name] != set(range(num_steps)):
            raise RuntimeError(
                f"{name} saw steps {sorted(seen[name])} != {list(range(num_steps))}."
            )
    if len(rows) != len(layers) * num_steps:
        raise RuntimeError(
            f"Expected {len(layers) * num_steps} site rows, got {len(rows)}."
        )
    return rows


def _write_site_csv(rows: list[SiteRow], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(SiteRow.__dataclass_fields__.keys())
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: getattr(row, name) for name in fieldnames})
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Failed to write {output}.")


def _agg_by_step(rows: list[SiteRow]) -> dict[int, dict[str, float]]:
    grouped: dict[int, list[SiteRow]] = defaultdict(list)
    for row in rows:
        grouped[row.step].append(row)
    out: dict[int, dict[str, float]] = {}
    for step, items in grouped.items():
        l1_all = sum(item.l1_all for item in items)
        l1_m = sum(item.l1_massive for item in items)
        l1_s = sum(item.l1_shoulder for item in items)
        n_sel = sum(item.n_selected for item in items)
        n_ch = sum(item.n_channels for item in items)
        out[step] = {
            "l1_all": l1_all,
            "l1_massive": l1_m,
            "l1_shoulder": l1_s,
            "frac_massive": l1_m / l1_all if l1_all else 0.0,
            "frac_shoulder": l1_s / l1_all if l1_all else 0.0,
            "n_massive": float(sum(item.n_massive for item in items)),
            "n_shoulder": float(sum(item.n_shoulder for item in items)),
            "n_values": float(sum(item.n_tokens * item.n_channels for item in items)),
            "frac_channels": n_sel / n_ch if n_ch else 0.0,
            "mean_tokens_with_massive": float(
                np.mean([item.n_tokens_with_massive for item in items])
            ),
            "mean_massive": float(
                np.mean([item.mean_massive for item in items if item.n_massive > 0] or [0.0])
            ),
            "mean_shoulder": float(
                np.mean(
                    [item.mean_shoulder for item in items if item.n_shoulder > 0] or [0.0]
                )
            ),
            "median_selected": float(
                np.mean([item.median_selected for item in items if item.n_selected > 0] or [0.0])
            ),
            "state_would_l1": sum(item.state_would_clip_l1 for item in items),
            "state_would_n": float(sum(item.state_would_clip_n for item in items)),
        }
    return out


def _reduce(items: list[SiteRow]) -> dict[str, float]:
    fake = [SiteRow(**{**item.__dict__, "step": 0}) for item in items]
    return _agg_by_step(fake)[0]


def _group_l1(acc: TokenIndexAcc, names: list[str]) -> dict[str, tuple[float, float, float]]:
    out: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for step in range(acc.n_steps):
        for index, name in enumerate(names):
            out[name][0] += float(acc.massive_l1[step, index])
            out[name][1] += float(acc.shoulder_l1[step, index])
            out[name][2] += float(acc.all_l1[step, index])
    return {key: (vals[0], vals[1], vals[2]) for key, vals in out.items()}


def _token_rows(
    acc: TokenIndexAcc, names: list[str]
) -> list[dict[str, float | int | str]]:
    massive_n = acc.massive_n.sum(axis=0)
    raw_n = acc.raw_over_n.sum(axis=0)
    massive_l1 = acc.massive_l1.sum(axis=0)
    raw_l1 = acc.raw_over_l1.sum(axis=0)
    shoulder_l1 = acc.shoulder_l1.sum(axis=0)
    all_l1 = acc.all_l1.sum(axis=0)
    rows = []
    for token, name in enumerate(names):
        rows.append(
            {
                "token": int(token),
                "group": name,
                "massive_n": float(massive_n[token]),
                "raw_over_n": float(raw_n[token]),
                "massive_l1": float(massive_l1[token]),
                "raw_over_l1": float(raw_l1[token]),
                "shoulder_l1": float(shoulder_l1[token]),
                "all_l1": float(all_l1[token]),
            }
        )
    return rows


def _share(part: float, total: float) -> float:
    return 0.0 if total <= 0 else 100.0 * part / total


def _edge_lines(rows: list[dict[str, float | int | str]]) -> list[str]:
    action = [row for row in rows if str(row["group"]) != "state"]
    valid = [row for row in rows if str(row["group"]) == "action_valid"]
    pad = [row for row in rows if str(row["group"]) == "action_pad"]
    if not valid:
        valid = action
    action_n = float(sum(float(row["massive_n"]) for row in action))
    raw_n = float(sum(float(row["raw_over_n"]) for row in rows))
    lines = [
        "",
        "edge shares of clip-aligned massive count (action tokens only):",
    ]
    if action:
        first = action[0]
        last = action[-1]
        first5 = action[: min(5, len(action))]
        last5 = action[-min(5, len(action)) :]
        mid = action[len(action) // 2]
        lines.extend(
            [
                f"  first action t={first['token']}  n={float(first['massive_n']):.0f}  "
                f"share={_share(float(first['massive_n']), action_n):.1f}%",
                f"  last action  t={last['token']}  n={float(last['massive_n']):.0f}  "
                f"share={_share(float(last['massive_n']), action_n):.1f}%",
                f"  first 5 actions  share={_share(sum(float(r['massive_n']) for r in first5), action_n):.1f}%",
                f"  last 5 actions   share={_share(sum(float(r['massive_n']) for r in last5), action_n):.1f}%",
                f"  middle action t={mid['token']}  n={float(mid['massive_n']):.0f}  "
                f"share={_share(float(mid['massive_n']), action_n):.1f}%",
            ]
        )
    if valid and pad:
        last_valid = valid[-1]
        last_pad = pad[-1]
        lines.extend(
            [
                f"  last valid t={last_valid['token']}  n={float(last_valid['massive_n']):.0f}  "
                f"share={_share(float(last_valid['massive_n']), action_n):.1f}%",
                f"  last pad   t={last_pad['token']}  n={float(last_pad['massive_n']):.0f}  "
                f"share={_share(float(last_pad['massive_n']), action_n):.1f}%",
                f"  valid total share={_share(sum(float(r['massive_n']) for r in valid), action_n):.1f}%",
                f"  pad total share={_share(sum(float(r['massive_n']) for r in pad), action_n):.1f}%",
            ]
        )
    state = next((row for row in rows if str(row["group"]) == "state"), None)
    if state is not None:
        lines.extend(
            [
                "",
                "state token vs μ+3σ (excluded from clip-aligned massive):",
                f"  clip-aligned massive_n={float(state['massive_n']):.0f}",
                f"  unmasked |x|>μ+3σ n={float(state['raw_over_n']):.0f}  "
                f"share_of_all_raw={_share(float(state['raw_over_n']), raw_n):.1f}%  "
                f"L1={float(state['raw_over_l1']):.4g}",
            ]
        )
    return lines


def _write_token_csv(
    rows: list[dict[str, float | int | str]], output: Path
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "token",
        "group",
        "massive_n",
        "raw_over_n",
        "massive_l1",
        "raw_over_l1",
        "shoulder_l1",
        "all_l1",
    ]
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot(
    rows: list[SiteRow],
    output_dir: Path,
    *,
    skip_first: bool,
    token_acc: TokenIndexAcc | None = None,
    valid_action_horizon: int | None = None,
) -> list[Path]:
    import matplotlib.pyplot as plt

    by_step = _agg_by_step(rows)
    steps = sorted(by_step)
    xs = np.asarray(steps, dtype=np.float64)
    mass = np.array([by_step[s]["frac_massive"] for s in steps])
    shol = np.array([by_step[s]["frac_shoulder"] for s in steps])
    rest = 1.0 - mass - shol
    written: list[Path] = []

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.1), constrained_layout=True)
    axes[0].stackplot(
        xs, mass, shol, rest, labels=("massive μ+3σ", "shoulder", "rest"), alpha=0.85
    )
    axes[0].set_title("L1 share")
    axes[0].set_ylabel("fraction of |x|")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[1].plot(xs, [by_step[s]["n_massive"] / by_step[s]["n_values"] for s in steps], "o-", label="massive")
    axes[1].plot(xs, [by_step[s]["n_shoulder"] / by_step[s]["n_values"] for s in steps], "s--", label="shoulder")
    axes[1].set_title("Value count share")
    axes[1].set_ylabel("fraction of (token, channel)")
    axes[1].legend(fontsize=8)
    axes[2].plot(xs, [by_step[s]["mean_massive"] for s in steps], "o-", label="mean |massive|")
    axes[2].plot(xs, [by_step[s]["mean_shoulder"] for s in steps], "s--", label="mean |shoulder|")
    axes[2].plot(xs, [by_step[s]["median_selected"] for s in steps], "^:", label="median on MAD ch")
    axes[2].set_title("Magnitude")
    axes[2].set_ylabel("|x|")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("denoise step")
        ax.set_xticks(list(steps))
        ax.grid(alpha=0.25)
    fig.suptitle(
        "DiT activations: massive (μ+3σ tip) vs normal-outlier (MAD shoulder) "
        + ("[GR00T skip state, 40 action tokens]" if skip_first else "[pi0.5 all tokens]"),
        fontsize=11,
    )
    share_path = output_dir / "outlier_types_by_step.png"
    fig.savefig(share_path, dpi=150)
    plt.close(fig)
    written.append(share_path)

    kinds = sorted({row.kind for row in rows})
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    xk = np.arange(len(kinds))
    m_frac = []
    s_frac = []
    for kind in kinds:
        items = [row for row in rows if row.kind == kind]
        stats = _reduce(items)
        m_frac.append(stats["frac_massive"])
        s_frac.append(stats["frac_shoulder"])
    width = 0.36
    ax.bar(xk - width / 2, m_frac, width, label="massive")
    ax.bar(xk + width / 2, s_frac, width, label="shoulder")
    ax.set_xticks(list(xk))
    ax.set_xticklabels(kinds, rotation=25, ha="right")
    ax.set_ylabel("L1 fraction")
    ax.set_title("L1 share by linear kind")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    kind_path = output_dir / "outlier_types_by_kind.png"
    fig.tight_layout()
    fig.savefig(kind_path, dpi=150)
    plt.close(fig)
    written.append(kind_path)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    toks_m = [row.n_tokens_with_massive / max(row.n_tokens, 1) for row in rows]
    toks_s = [row.n_tokens_with_shoulder / max(row.n_tokens, 1) for row in rows]
    ax.hist(toks_m, bins=20, alpha=0.7, label="tokens with any massive")
    ax.hist(toks_s, bins=20, alpha=0.5, label="tokens with any shoulder")
    ax.set_xlabel("fraction of tokens in one linear×step")
    ax.set_ylabel("site count")
    ax.set_title("How many tokens carry the class (per site)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    sparse_path = output_dir / "outlier_types_token_occupancy.png"
    fig.tight_layout()
    fig.savefig(sparse_path, dpi=150)
    plt.close(fig)
    written.append(sparse_path)

    if token_acc is not None:
        fig, axes = plt.subplots(
            2, 1, figsize=(8.8, 6.4), sharex=True, constrained_layout=True
        )
        idx = np.arange(token_acc.n_tokens)
        mass_t = token_acc.massive_l1.sum(axis=0)
        shol_t = token_acc.shoulder_l1.sum(axis=0)
        mass_n = token_acc.massive_n.sum(axis=0)
        raw_n = token_acc.raw_over_n.sum(axis=0)
        axes[0].plot(idx, mass_t, "o-", ms=3.5, label="massive L1 (clip-aligned)")
        axes[0].plot(idx, shol_t, "s--", ms=3.5, label="shoulder L1")
        axes[0].set_ylabel("sum |x|")
        axes[0].set_title("Where massive / shoulder mass sits along the chunk")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.25)
        axes[1].plot(idx, mass_n, "o-", ms=3.5, label="massive count (clip-aligned)")
        axes[1].plot(
            idx, raw_n, "s--", ms=3.5, label="|x|>μ+3σ count (state unmasked)"
        )
        axes[1].set_xlabel("DiT token index (0=state if GR00T)")
        axes[1].set_ylabel("summed sites")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.25)
        if skip_first:
            for ax in axes:
                ax.axvline(0.5, color="0.4", lw=0.8, ls=":")
                if valid_action_horizon is not None:
                    split = 0.5 + float(valid_action_horizon)
                    ax.axvline(split, color="0.4", lw=0.8, ls="--")
        token_path = output_dir / "outlier_types_by_token.png"
        fig.savefig(token_path, dpi=150)
        plt.close(fig)
        written.append(token_path)

    for path in written:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Failed to write {path}.")
    return written


def _write_summary(
    rows: list[SiteRow],
    output: Path,
    *,
    skip_first: bool,
    valid_action_horizon: int | None,
    token_acc: TokenIndexAcc | None = None,
) -> None:
    overall = _reduce(rows)
    lines = [
        "Massive = MAD-selected channels, |x| > μ+3σ (clip tip).",
        "Normal-outlier / shoulder = same channels, MAD floor .. μ+3σ.",
        f"skip_first_token={skip_first}  valid_action_horizon={valid_action_horizon}",
        "",
        f"sites={len(rows)}",
        f"selected channels (mean frac)={overall['frac_channels']:.4f}",
        f"massive L1 frac={overall['frac_massive']:.4e}  "
        f"count frac={overall['n_massive'] / overall['n_values']:.4e}",
        f"shoulder L1 frac={overall['frac_shoulder']:.4e}  "
        f"count frac={overall['n_shoulder'] / overall['n_values']:.4e}",
        f"mean |massive|={overall['mean_massive']:.4g}  "
        f"mean |shoulder|={overall['mean_shoulder']:.4g}  "
        f"median on MAD ch={overall['median_selected']:.4g}",
        f"massive / shoulder magnitude="
        f"{(overall['mean_massive'] / overall['mean_shoulder']) if overall['mean_shoulder'] else float('inf'):.3g}",
        f"tokens with any massive (mean per site)={overall['mean_tokens_with_massive']:.2f}",
    ]
    if skip_first:
        lines.append(
            f"state would exceed μ+3σ (not clipped): "
            f"n={overall['state_would_n']:.0f}  L1={overall['state_would_l1']:.4g}  "
            f"vs massive L1={overall['l1_massive']:.4g}"
        )
    by_step = _agg_by_step(rows)
    lines.append("")
    lines.append("by denoise step:")
    for step in sorted(by_step):
        stats = by_step[step]
        lines.append(
            f"  s={step}  massive_L1={stats['frac_massive']:.4e}  "
            f"shoulder_L1={stats['frac_shoulder']:.4e}  "
            f"|m|/|s|="
            f"{(stats['mean_massive'] / stats['mean_shoulder']) if stats['mean_shoulder'] else float('inf'):.2f}"
        )
    by_kind = {kind: _reduce([row for row in rows if row.kind == kind])
               for kind in sorted({row.kind for row in rows})}
    lines.append("")
    lines.append("by kind:")
    for kind, stats in by_kind.items():
        lines.append(
            f"  {kind:12s}  massive_L1={stats['frac_massive']:.4e}  "
            f"shoulder_L1={stats['frac_shoulder']:.4e}  "
            f"ch_frac={stats['frac_channels']:.3f}"
        )
    if token_acc is not None:
        names = _token_group_names(
            token_acc.n_tokens,
            skip_first=skip_first,
            valid_action_horizon=valid_action_horizon,
        )
        grouped = _group_l1(token_acc, names)
        lines.append("")
        lines.append("L1 by token group (clip-aligned classes; state not in massive/shoulder):")
        for name in ("state", "action_valid", "action_pad", "action"):
            if name not in grouped:
                continue
            massive_l1, shoulder_l1, all_l1 = grouped[name]
            lines.append(
                f"  {name:13s}  all={all_l1:.4g}  massive={massive_l1:.4g}  "
                f"shoulder={shoulder_l1:.4g}  "
                f"massive/all={massive_l1 / all_l1 if all_l1 else 0.0:.4e}"
            )
        token_rows = _token_rows(token_acc, names)
        lines.append("")
        lines.append("per-token massive count (summed over conditions x steps x layers):")
        for row in token_rows:
            lines.append(
                f"  t={int(row['token']):02d} {str(row['group']):13s}  "
                f"massive_n={float(row['massive_n']):.0f}  "
                f"raw_over_n={float(row['raw_over_n']):.0f}  "
                f"massive_l1={float(row['massive_l1']):.4g}"
            )
        lines.extend(_edge_lines(token_rows))
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Failed to write {output}.")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-data", type=Path, required=True)
    matched.add_model_cli(parser)
    parser.add_argument("--layer-regex", default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--samples", default=None)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--noise-seeds", default=None)
    parser.add_argument("--outlier-std-k", type=float, default=3.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--params-dtype", default="bfloat16")
    parser.add_argument(
        "--valid-action-horizon",
        type=int,
        default=None,
        help="GR00T: number of real action tokens after state. Default 16 for groot_n17.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tools/img/dit_outlier_types"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(args.checkpoint)
    if not args.calibration_data.is_file():
        raise FileNotFoundError(args.calibration_data)
    sample_ids = (
        detail._parse_nonneg_ints(args.samples, name="--samples")
        if args.samples is not None
        else [int(args.sample_index)]
    )
    noise_ids = (
        detail._parse_nonneg_ints(args.noise_seeds, name="--noise-seeds")
        if args.noise_seeds is not None
        else [int(args.noise_seed)]
    )
    skip_first_token = args.model == "groot_n17"
    valid_horizon = args.valid_action_horizon
    if skip_first_token and valid_horizon is None:
        valid_horizon = 16
    adapter = matched.adapter_from_args(args)
    model = adapter.build_model()
    model.eval()
    adapter.warmup_for_calibration(model)
    config = QVLAConfig.for_model_kind(args.model)
    layers = matched._dit_layers(model, args.layer_regex, config=config)
    runtime = matched.clip_runtime(adapter, config)
    need = max(sample_ids) + 1
    batches = list(adapter.iter_calibration_batches(need))
    if len(batches) != need:
        raise RuntimeError(f"Calibration yielded {len(batches)}, need {need}.")
    conditions = [(sample, seed) for sample in sample_ids for seed in noise_ids]
    print(
        f"model={args.model} layers={len(layers)} steps={runtime.num_steps} "
        f"n_tokens={runtime.n_tokens} skip_first={skip_first_token} "
        f"valid_horizon={valid_horizon} conditions={len(conditions)}"
    )
    rows: list[SiteRow] = []
    token_acc = TokenIndexAcc.zeros(runtime.n_tokens, runtime.num_steps)
    for index, (sample, seed) in enumerate(conditions, start=1):
        request = matched._fixed_noise_request(
            adapter, batches[sample], runtime, noise_seed=seed
        )
        print(f"[{index}/{len(conditions)}] sample={sample} noise={seed}")
        rows.extend(
            _capture_and_classify(
                adapter,
                request,
                layers,
                num_steps=runtime.num_steps,
                n_tokens=runtime.n_tokens,
                std_k=args.outlier_std_k,
                skip_first_token=skip_first_token,
                sample=sample,
                noise=seed,
                token_acc=token_acc,
            )
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "outlier_types_sites.csv"
    token_csv = args.output_dir / "outlier_types_by_token.csv"
    summary_path = args.output_dir / "summary.txt"
    _write_site_csv(rows, csv_path)
    names = _token_group_names(
        token_acc.n_tokens,
        skip_first=skip_first_token,
        valid_action_horizon=valid_horizon,
    )
    _write_token_csv(_token_rows(token_acc, names), token_csv)
    _write_summary(
        rows,
        summary_path,
        skip_first=skip_first_token,
        valid_action_horizon=valid_horizon,
        token_acc=token_acc,
    )
    written = _plot(
        rows,
        args.output_dir,
        skip_first=skip_first_token,
        token_acc=token_acc,
        valid_action_horizon=valid_horizon,
    )
    for path in [csv_path, token_csv, summary_path, *written]:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
