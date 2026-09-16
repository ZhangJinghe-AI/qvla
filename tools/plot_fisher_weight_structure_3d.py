#!/usr/bin/env python
"""3D out×in Fisher and |W| side by side for quant-target linears.

Per denoise-step local weight Fisher
``F_s[o, i] = Σ_k (∂a_k / ∂W^{(s)}_{o,i})²`` where
``∂a/∂W^{(s)} = (∂a/∂y_s)^T @ x_s``, plotted next to ``|W[o, i]|``.

One PNG per quant-target linear under ``--output-dir``; fused
``qkv_proj`` / ``gate_up_proj`` emit one PNG per out-leg (Q/K/V, gate/up):

* **LLM** — 2 subplots: Fisher | |W|
* **DiT** — 4×5 subplots: Fisher step 0..9, then |W| (repeated per step for layout)

Example::

    CUDA_VISIBLE_DEVICES=2 uv run python tools/plot_fisher_weight_structure_3d.py \\
        --checkpoint /data/share/pi05_libero_finetuned_v044 \\
        --calibration-source file \\
        --calibration-data ../calibration_data/libero_object_16_7.npz \\
        --output-dir tools/img/fisher/weight_structure_3d
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from qvla.adapters import get_adapter  # noqa: E402
from qvla.adapters.pi05.step_hook import patched_one_step  # noqa: E402
from qvla.build.fisher import resolve_fisher_action_dim, select_fisher_actions  # noqa: E402
from qvla.config import QVLAConfig  # noqa: E402
from qvla.runtime import list_target_modules  # noqa: E402


def _resolve_targets(
    model: torch.nn.Module, config: QVLAConfig
) -> list[tuple[str, str, torch.nn.Module]]:
    targets = list_target_modules(model, config)
    if not targets:
        raise SystemExit("No quant target layers found.")
    return sorted(targets, key=lambda t: (t[1], t[0]))


def _layer_file_stem(name: str, scope: str) -> str:
    m = re.search(r"layers\.(\d+)\.", name)
    if m:
        idx = int(m.group(1))
        for kind in ("qkv_proj", "o_proj", "gate_up_proj", "down_proj"):
            if name.endswith(kind):
                return f"{scope}_layer{idx:02d}_{kind}"
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", name)
    return f"{scope}_{safe}"


def _require_weight(mod: torch.nn.Module, layer_name: str) -> torch.nn.Parameter:
    w = getattr(mod, "weight", None)
    if not isinstance(w, torch.nn.Parameter):
        raise RuntimeError(
            f"Layer {layer_name!r}: expected nn.Parameter weight, got {type(w)}."
        )
    if w.ndim != 2:
        raise RuntimeError(
            f"Layer {layer_name!r}: expected 2-D weight, got shape {tuple(w.shape)}."
        )
    return w


def _unwrap_linear_output(output: object, layer_name: str) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        return output[0]
    raise RuntimeError(
        f"Layer {layer_name!r}: unexpected forward output type {type(output)}."
    )


def _local_weight_grad(
    x: torch.Tensor, y_grad: torch.Tensor, *, out_features: int, in_features: int
) -> torch.Tensor:
    x_flat = x.detach().float().reshape(-1, in_features)
    g_flat = y_grad.detach().float().reshape(-1, out_features)
    if x_flat.shape[0] != g_flat.shape[0]:
        raise RuntimeError(
            f"Token count mismatch for weight grad: x={x_flat.shape[0]} "
            f"vs y.grad={g_flat.shape[0]}."
        )
    return g_flat.T @ x_flat


def _out_channel_splits(
    name: str, mod: torch.nn.Module, n_out: int
) -> list[tuple[str, slice]]:
    """Named out-row slices for fused projections; empty = no split."""
    sizes = getattr(mod, "output_partition_sizes", None)
    if name.endswith("qkv_proj"):
        labels = ("q", "k", "v")
        if sizes is None:
            raise RuntimeError(
                f"Layer {name!r}: qkv_proj missing output_partition_sizes; "
                "cannot split Q/K/V."
            )
        if len(sizes) != 3:
            raise RuntimeError(
                f"Layer {name!r}: expected 3 QKV partition sizes, got {sizes}."
            )
    elif name.endswith("gate_up_proj"):
        labels = ("gate", "up")
        if sizes is None:
            raise RuntimeError(
                f"Layer {name!r}: gate_up_proj missing output_partition_sizes; "
                "cannot split gate/up."
            )
        if len(sizes) != 2:
            raise RuntimeError(
                f"Layer {name!r}: expected 2 gate/up partition sizes, got {sizes}."
            )
    else:
        return []

    if sum(sizes) != n_out:
        raise RuntimeError(
            f"Layer {name!r}: partition sizes {list(sizes)} sum to {sum(sizes)} "
            f"!= out dim {n_out}."
        )
    splits: list[tuple[str, slice]] = []
    start = 0
    for label, size in zip(labels, sizes, strict=True):
        splits.append((label, slice(start, start + int(size))))
        start += int(size)
    return splits


def _collect_maps(
    adapter,
    targets: list[tuple[str, str, torch.nn.Module]],
    batches: list[dict],
    *,
    action_timestep: str,
    noise_ensemble_k: int = 1,
    fisher_method: str = "exact",
    hutchinson_probes: int = 8,
) -> tuple[
    dict[str, dict[int | None, torch.Tensor]],
    dict[str, torch.Tensor],
]:
    """Averaged per-step weight Fisher and |W| over calibration runs."""
    fisher_method = str(fisher_method).strip().lower()
    if fisher_method not in ("exact", "hutchinson"):
        raise ValueError(
            f"Unknown fisher method {fisher_method!r}; expected 'exact' or 'hutchinson'."
        )
    if hutchinson_probes < 1:
        raise ValueError(
            f"hutchinson_probes must be >= 1, got {hutchinson_probes}."
        )
    from qvla.adapters.pi05.differentiable_forward import (
        differentiable_inference_context,
        force_eager_runners,
        reset_differentiable_state,
    )
    from qvla.runtime.step_context import reset_step_counters

    engine = adapter.engine
    if engine is None:
        raise RuntimeError("Adapter has no engine — call build_model() first.")
    sched = engine.entry.scheduler
    model = engine.entry.model
    force_eager_runners(sched)

    scopes = {name: scope for name, scope, _ in targets}
    weights = {name: _require_weight(mod, name) for name, _, mod in targets}
    out_features = {name: int(w.shape[0]) for name, w in weights.items()}
    in_features = {name: int(w.shape[1]) for name, w in weights.items()}

    fisher_acc: dict[str, dict[int | None, torch.Tensor]] = {}
    on_graph: dict[str, bool] = {name: True for name, _, _ in targets}
    total_runs = len(batches) * noise_ensemble_k

    current_step: list[int | None] = [None]
    # name -> list of (step, x, y); y is filled by the matching forward hook.
    sample_ios: dict[
        str, list[tuple[int | None, torch.Tensor, torch.Tensor | None]]
    ] = {name: [] for name, _, _ in targets}

    def make_pre_hook(layer_name: str):
        scope = scopes[layer_name]

        def hook(_mod, inputs):
            if not inputs or not torch.is_tensor(inputs[0]):
                raise RuntimeError(
                    f"Layer {layer_name!r}: missing tensor input in forward_pre_hook."
                )
            x = inputs[0]
            if not (x.requires_grad or x.grad_fn is not None):
                raise RuntimeError(
                    f"Layer {layer_name!r}: activation is not on the autograd graph."
                )
            step = current_step[0] if scope == "dit" else None
            if scope == "dit" and step is None:
                raise RuntimeError(
                    f"DiT layer {layer_name!r} hooked while current_step is None."
                )
            sample_ios[layer_name].append((step, x, None))

        return hook

    def make_fwd_hook(layer_name: str):
        def hook(_mod, _inputs, output):
            y = _unwrap_linear_output(output, layer_name)
            if not (y.requires_grad or y.grad_fn is not None):
                raise RuntimeError(
                    f"Layer {layer_name!r}: output is not on the autograd graph."
                )
            y.retain_grad()
            entries = sample_ios[layer_name]
            if not entries:
                raise RuntimeError(
                    f"Layer {layer_name!r}: forward hook without a matching pre-hook."
                )
            step, x, y_prev = entries[-1]
            if y_prev is not None:
                raise RuntimeError(
                    f"Layer {layer_name!r}: forward hook saw an already-filled output."
                )
            entries[-1] = (step, x, y)

        return hook

    handles = []
    for name, _, mod in targets:
        handles.append(mod.register_forward_pre_hook(make_pre_hook(name)))
        handles.append(mod.register_forward_hook(make_fwd_hook(name)))

    try:
        with differentiable_inference_context(sched):
            for si, batch in enumerate(batches):
                for ni in range(noise_ensemble_k):
                    run_idx = si * noise_ensemble_k + ni + 1
                    print(
                        f"  run {run_idx}/{total_runs}  "
                        f"(sample={si}, noise={ni})"
                    )
                    for v in sample_ios.values():
                        v.clear()

                    reset_differentiable_state(sched)
                    reset_step_counters(model)
                    model.zero_grad(set_to_none=True)

                    def _step_cb(step):
                        current_step[0] = None if step is None else int(step)

                    with patched_one_step(sched.expert_runner, _step_cb):
                        current_step[0] = None
                        actions = adapter.forward_differentiable(
                            [batch], sample_indices=[si], noise_index=ni,
                        )

                    missing = [n for n, _, _ in targets if not sample_ios[n]]
                    if missing:
                        raise RuntimeError(f"No I/O captured for: {missing}")
                    incomplete = [
                        n
                        for n, _, _ in targets
                        if any(y is None for _s, _x, y in sample_ios[n])
                    ]
                    if incomplete:
                        raise RuntimeError(
                            f"Forward hook did not fill outputs for: {incomplete}"
                        )

                    if not fisher_acc:
                        for name, _, _ in targets:
                            shape = weights[name].shape
                            fisher_acc[name] = {}
                            for step, _x, _y in sample_ios[name]:
                                if step not in fisher_acc[name]:
                                    fisher_acc[name][step] = torch.zeros(
                                        shape,
                                        device=weights[name].device,
                                        dtype=torch.float32,
                                    )

                    actions = select_fisher_actions(
                        actions,
                        timestep=action_timestep,
                        action_dim=resolve_fisher_action_dim(adapter),
                    )
                    if not (actions.requires_grad or actions.grad_fn is not None):
                        raise RuntimeError("actions has no grad_fn; Fisher cannot run.")

                    batch_size = actions.shape[0]
                    actions_flat = actions.reshape(batch_size, -1)
                    n_action_dims = actions_flat.shape[1]
                    n_backward = (
                        n_action_dims
                        if fisher_method == "exact"
                        else hutchinson_probes
                    )

                    for i in range(n_backward):
                        for ios in sample_ios.values():
                            for _step, _x, y in ios:
                                if y is None:
                                    raise RuntimeError("Internal error: y is None before backward.")
                                if y.grad is not None:
                                    y.grad.zero_()
                        model.zero_grad(set_to_none=True)

                        if fisher_method == "exact":
                            grad_output = torch.zeros_like(actions_flat)
                            grad_output[:, i] = 1.0
                            scale = 1.0
                        else:
                            signs = torch.randint(
                                0,
                                2,
                                (n_action_dims,),
                                device=actions_flat.device,
                                dtype=torch.int64,
                            ).to(actions_flat.dtype)
                            signs = signs.mul_(2.0).sub_(1.0)
                            grad_output = signs.unsqueeze(0).expand(batch_size, -1)
                            scale = 1.0 / hutchinson_probes
                        actions_flat.backward(
                            grad_output, retain_graph=(i < n_backward - 1)
                        )

                        for name, _, _ in targets:
                            if not on_graph[name]:
                                continue
                            # Sum local ∂a/∂W over all uses of the same step, then square.
                            step_gW: dict[int | None, torch.Tensor] = {}
                            for step, x, y in sample_ios[name]:
                                if y is None:
                                    raise RuntimeError(
                                        f"Layer {name!r}: y is None before reading grad."
                                    )
                                if y.grad is None:
                                    on_graph[name] = False
                                    break
                                gW = _local_weight_grad(
                                    x,
                                    y.grad,
                                    out_features=out_features[name],
                                    in_features=in_features[name],
                                )
                                if step not in step_gW:
                                    step_gW[step] = gW
                                else:
                                    step_gW[step].add_(gW)
                            if not on_graph[name]:
                                continue
                            for step, gW in step_gW.items():
                                if step not in fisher_acc[name]:
                                    raise RuntimeError(
                                        f"Layer {name!r}: unexpected step {step!r} "
                                        f"(known={sorted(fisher_acc[name], key=lambda s: -1 if s is None else s)})."
                                    )
                                fisher_acc[name][step].add_(gW.pow(2), alpha=scale)

        disconnected = [n for n, ok in on_graph.items() if not ok]
        if disconnected:
            print(
                f"Not on action graph (no PNG): {len(disconnected)} layer(s):\n"
                + "\n".join(f"  - {n}" for n in disconnected)
            )
        kept = [n for n, _, _ in targets if on_graph[n]]
        if not kept:
            raise RuntimeError("No layers remain on the action graph.")

        fisher_maps = {
            n: {s: (v / total_runs).cpu() for s, v in fisher_acc[n].items()}
            for n in kept
        }
        weight_maps = {
            name: weights[name].detach().float().abs().cpu() for name in kept
        }
        return fisher_maps, weight_maps
    finally:
        for h in handles:
            h.remove()


def _dit_steps(
    per_step: dict[int | None, torch.Tensor],
    *,
    num_dit_steps: int,
    layer_name: str,
) -> list[tuple[int, torch.Tensor]]:
    steps = sorted(k for k in per_step if k is not None)
    if len(steps) != num_dit_steps:
        raise RuntimeError(
            f"Layer {layer_name!r}: expected {num_dit_steps} DiT steps, "
            f"got {len(steps)}: {steps}"
        )
    if steps != list(range(num_dit_steps)):
        raise RuntimeError(
            f"Layer {layer_name!r}: DiT steps are not 0..{num_dit_steps - 1}: {steps}"
        )
    return [(s, per_step[s]) for s in steps]


def _plot_panels(
    panels: list[tuple[str, torch.Tensor, str]],
    *,
    layer_name: str,
    ncols: int,
    output: Path,
    z_scale: str,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    n_panels = len(panels)
    nrows = math.ceil(n_panels / ncols)
    fig = plt.figure(figsize=(4.2 * ncols, 3.8 * nrows + 0.8))

    for idx, (subtitle, mat, cmap) in enumerate(panels):
        raw = mat.float().detach().cpu().numpy()
        if z_scale == "log10":
            data = np.log10(np.clip(raw, 1e-30, None))
            z_label = "log10"
        else:
            data = raw
            z_label = "linear"
        n_out, n_in = data.shape
        in_idx, out_idx = np.meshgrid(np.arange(n_in), np.arange(n_out))

        ax = fig.add_subplot(nrows, ncols, idx + 1, projection="3d")
        ax.plot_surface(
            in_idx,
            out_idx,
            data,
            cmap=cmap,
            linewidth=0,
            antialiased=False,
            rstride=max(1, n_out // 60),
            cstride=max(1, n_in // 60),
        )
        ax.set_xlabel("in", fontsize=7)
        ax.set_ylabel("out", fontsize=7)
        ax.set_zlabel(z_label, fontsize=7)
        ax.set_title(subtitle, fontsize=9)
        ax.tick_params(labelsize=6)

    fig.suptitle(layer_name, fontsize=11)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


def _process_layer(
    name: str,
    scope: str,
    fisher_map: dict[int | None, torch.Tensor],
    weight_abs: torch.Tensor,
    *,
    num_dit_steps: int,
    output: Path,
    z_scale: str,
) -> None:
    print(f"\n=== {name} ({scope}) ===")
    panels: list[tuple[str, torch.Tensor, str]] = []

    if scope == "llm":
        if None not in fisher_map:
            raise RuntimeError(f"LLM layer {name!r}: missing step key None.")
        if len(fisher_map) != 1:
            raise RuntimeError(
                f"LLM layer {name!r}: expected only step None, got {list(fisher_map)}."
            )
        fisher = fisher_map[None]
        if fisher.shape != weight_abs.shape:
            raise RuntimeError(
                f"Layer {name!r}: Fisher shape {tuple(fisher.shape)} "
                f"!= |W| shape {tuple(weight_abs.shape)}."
            )
        print(f"  Fisher / |W|: shape={tuple(fisher.shape)}")
        panels = [
            ("Fisher", fisher, "plasma"),
            ("|W|", weight_abs, "viridis"),
        ]
        ncols = 2
    else:
        fisher_steps = _dit_steps(
            fisher_map, num_dit_steps=num_dit_steps, layer_name=name
        )
        for step, fisher in fisher_steps:
            if fisher.shape != weight_abs.shape:
                raise RuntimeError(
                    f"Layer {name!r} step {step}: Fisher shape {tuple(fisher.shape)} "
                    f"!= |W| shape {tuple(weight_abs.shape)}."
                )
            print(f"  Fisher step {step}: shape={tuple(fisher.shape)}")
            panels.append((f"Fisher step {step}", fisher, "plasma"))
        # Repeat |W| once per step so the grid matches the Fisher row count.
        for step, _ in fisher_steps:
            panels.append((f"|W| (step {step})", weight_abs, "viridis"))
        ncols = 5

    _plot_panels(
        panels,
        layer_name=name,
        ncols=ncols,
        output=output,
        z_scale=z_scale,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--calibration-source", choices=("file", "synthetic"), required=True)
    p.add_argument("--calibration-data", type=Path)
    p.add_argument("--num-samples", type=int, default=1, help="Number of calibration samples.")
    p.add_argument("--noise-ensemble-k", type=int, default=1, help="Noise ensembles per sample.")
    p.add_argument(
        "--action-timestep",
        default="0,24,49",
        help=(
            "Fisher action chunk indices: 'all', one index, or comma-separated "
            "(default: 0,24,49)."
        ),
    )
    p.add_argument(
        "--fisher-method",
        choices=("exact", "hutchinson"),
        default="exact",
        help=(
            "Fisher estimator: exact (one backward per action dim) or "
            "hutchinson (random-projection approximation)."
        ),
    )
    p.add_argument(
        "--fisher-hutchinson-probes",
        type=int,
        default=8,
        help="Number of random projections when --fisher-method=hutchinson.",
    )
    p.add_argument(
        "--z-scale",
        choices=("log10", "linear"),
        default="log10",
        help="Z-axis scale for both Fisher and |W| plots.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for auto-named PNG per linear layer.",
    )
    args = p.parse_args(argv)

    if args.calibration_source == "file" and args.calibration_data is None:
        p.error("--calibration-data is required when --calibration-source=file.")
    if args.fisher_hutchinson_probes < 1:
        p.error("--fisher-hutchinson-probes must be >= 1.")

    config = QVLAConfig.pi05_default()
    adapter_kwargs: dict = {
        "checkpoint_path": args.checkpoint,
        "calibration_source": args.calibration_source,
    }
    if args.calibration_source == "file":
        adapter_kwargs["calibration_data_path"] = args.calibration_data

    adapter = get_adapter("pi05", **adapter_kwargs)
    model = adapter.build_model()
    adapter.warmup_for_calibration(model)
    num_dit_steps = adapter.dit_step_count(config)

    targets = _resolve_targets(model, config)
    mod_by_name = {name: mod for name, _, mod in targets}

    batches = list(adapter.iter_calibration_batches(args.num_samples))
    if len(batches) < args.num_samples:
        raise SystemExit(
            f"Requested {args.num_samples} samples but only {len(batches)} available."
        )

    print(
        f"Collecting per-step weight Fisher + |W| for {len(targets)} layer(s) "
        f"({args.num_samples} sample(s) × {args.noise_ensemble_k} noise(s), "
        f"action_timestep={args.action_timestep!r}, num_dit_steps={num_dit_steps}, "
        f"method={args.fisher_method}"
        + (
            f", probes={args.fisher_hutchinson_probes}"
            if args.fisher_method == "hutchinson"
            else ""
        )
        + ")..."
    )
    fisher_maps, weight_maps = _collect_maps(
        adapter,
        targets,
        batches,
        action_timestep=args.action_timestep,
        noise_ensemble_k=args.noise_ensemble_k,
        fisher_method=args.fisher_method,
        hutchinson_probes=args.fisher_hutchinson_probes,
    )

    scope_by_name = {name: scope for name, scope, _ in targets}
    for name in sorted(fisher_maps):
        stem = _layer_file_stem(name, scope_by_name[name])
        n_out = int(weight_maps[name].shape[0])
        splits = _out_channel_splits(name, mod_by_name[name], n_out)
        jobs: list[
            tuple[str, dict[int | None, torch.Tensor], torch.Tensor, Path]
        ]
        if splits:
            jobs = []
            for label, sl in splits:
                f_map = {s: mat[sl, :] for s, mat in fisher_maps[name].items()}
                w_abs = weight_maps[name][sl, :]
                out_path = args.output_dir / f"{stem}_{label}_fisher_weight_3d.png"
                jobs.append((f"{name} [{label}]", f_map, w_abs, out_path))
        else:
            out_path = args.output_dir / f"{stem}_fisher_weight_3d.png"
            jobs = [(name, fisher_maps[name], weight_maps[name], out_path)]

        for title, f_map, w_abs, out_path in jobs:
            _process_layer(
                title,
                scope_by_name[name],
                f_map,
                w_abs,
                num_dit_steps=num_dit_steps,
                output=out_path,
                z_scale=args.z_scale,
            )
        del fisher_maps[name], weight_maps[name]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
