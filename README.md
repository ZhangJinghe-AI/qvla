# QVLA

W4A4 post-training quantization for VLA models, based on the **QVLA** recipe
(svd-Hadamard rotation + GPTQ for the LLM backbone + RTN-residual + per-step
activation scales for the DiT action head).

This is a from-scratch, model-agnostic reimplementation organized so that
**adding a new VLA model means writing one small adapter file**.

## Why this package is separate from `phyai`

The constraint while building this was: *do not touch any file inside `phyai/`*.
Everything here is a standalone Python package that interacts with phyai purely
through public, stable surfaces:

* `phyai.layers.linear.layers.{ReplicatedLinear, ColumnParallelLinear, ...}`
  are recognized by `isinstance` and replaced wholesale at runtime — phyai's
  weight loader, kernel dispatcher, and CUDA-graph runner stay untouched.
* The pi0.5 model server in `vla-evaluation-harness/...` is forked into a
  separate file `pi05-phyai-qvla.py`; the original `pi05-phyai.py` is
  unchanged.

## Layout

```
src/qvla/
  config.py              # QVLAConfig (one source of truth)

  core/                  # model-agnostic quantization primitives
    rotation.py          # svd_hadamard rotation (DuQuant-style)
    quantize.py          # RTN, GPTQ, RTN-residual
    pack.py              # on-disk pack format (.pt)

  runtime/               # load a pack and run quantized inference
    quant_linear.py      # QuantLinear nn.Module
    step_context.py      # per-layer step counter (CUDA-graph friendly)
    wrap.py              # walk-and-replace via regex include/exclude

  build/                 # offline pack builder
    builder.py           # rotation -> GPTQ/RTN -> per-step scales
    collector.py         # activation-capture hooks during calibration
    fisher.py            # Fisher sensitivity (policy-aware DiT rotation)
    differentiable_forward.py
    attention_ste.py

  adapters/              # per-model glue (regexes, calibration loop)
    base.py
    pi05.py
    groot.py

  calibration/           # optional calibration-data providers
    file.py

scripts/build_pi05_pack.py
tests/
```

## Two-phase workflow

### 1. Build the pack (offline, one-time per checkpoint)

```bash
uv run python scripts/build_pi05_pack.py \
    --checkpoint /data/share/pi05-libero \
    --output ./packs/pi05_libero_object_W4A4.pt \
    --calibration-source file \
    --calibration-data ../calibration_data/libero_object_16_7.npz \
    --num-samples 16 \
    -vv
```

For **policy-aware DiT rotation**, add ``--dit-rotation policy_svd_hadamard``.
The builder runs a Fisher sensitivity pass (``--fisher-num-samples``, default 4)
before quantizing DiT layers; expect ~10 min per Fisher sample on pi0.5-libero.

```bash
uv run python scripts/build_pi05_pack.py \
    --checkpoint /data/share/pi05-libero \
    --output ./packs/pi05_libero_policy_W4A4.pt \
    --calibration-source file \
    --calibration-data ../calibration_data/libero_object_16_7.npz \
    --dit-rotation policy_svd_hadamard \
    --fisher-num-samples 4 \
    -vv
```

Run ``--help`` to override every field on ``QVLAConfig`` / ``ScopeConfig``
(per-scope rotation, zigzag perm, SVD source, activation scales, GPTQ knobs,
regexes, etc.). ``--config-json`` loads a base recipe; explicit flags win.

DuQuant rotation knobs (per scope):

* ``--llm-enable-permute`` / ``--no-llm-enable-permute`` — zigzag channel reorder (default: on)
* ``--llm-perm-score weight|activation|activation_weight`` — energy for zigzag (default: weight)
* ``--llm-svd-source weight|activation`` — SVD on weight blocks (paper) vs activation cov (ablation)

Internally the builder does, **per linear layer**:

1. Capture calibration activations (a few batches of real LIBERO observations).
2. Fit the DuQuant composite transform: zigzag permutation (weight energy by
   default) + per-block weight SVD + Hadamard (`R = U @ H`).
3. Apply the transform to the *weight* matrix offline: `W ← W[:, perm] @ R`.
4. Quantize the rotated weight to int4 with either GPTQ
   (Cholesky-based Hessian update) or RTN-with-residual.
5. For DiT layers, capture per-Euler-step activation maxima and store an
   `act_scale_table` of shape `(num_steps, …)`.
6. Serialize everything into one `.pt` pack file.

When ``--dit-rotation policy_svd_hadamard`` is set, step 2 for DiT layers uses
Fisher-weighted covariance (a differentiable forward + exact action Jacobian on
the full-precision model) instead of plain activation statistics alone.

### 2. Load the pack at eval time

The new model server (`pi05-phyai-qvla.py`) does:

```python
from qvla import Pack, enable_quantization

# 1. Build PhyAI model normally.
self._engine = Engine(EngineArgs(plugin="pi05", ...))

# 2. Swap matching Linear layers in-place with QuantLinear.
pack = Pack.load("./packs/pi05_libero_object_W4A4.pt")
enable_quantization(self._engine.entry.model, pack)

# 3. Recapture CUDA graphs (when use_cuda_graph=True in the server config).
#    The server builds the engine eager, swaps layers, then calls
#    scheduler.setup() again so captured graphs wrap QuantLinear.
```

`enable_quantization` walks `named_modules`, matches against the config's
include/exclude regex pair, and replaces each match with an
`QuantLinear` carrying its share of the pack. Per-step DiT activation
scales remain correct under CUDA-graph capture because the Euler loop is
unrolled at capture time (see `step_context.py`).

## Multi-model story

Adding **Groot N1.7** requires only:

1. A new include regex (`gr00t.adapters.groot.GR00T_LLM_RE`, `GR00T_DIT_RE`).
2. A `ModelAdapter` subclass exposing:
   * `iter_calibration_batches()` — yield observation dicts that the model's
     own preprocessing pipeline can consume.
   * `forward_for_calibration()` — run one inference and record activations.
   * `dit_step_count()` — how many denoise steps (for the per-step table).
3. No change to `core/` — the rotation, GPTQ/RTN solvers, pack format, and
   runtime layer are all model-blind.

If your model lacks a DiT head (pure LLM VLA), just point `dit_step_count` to
`1` and the per-step table collapses to a static scale.

## Fallback options (if the per-step table can't be wired)

Some architectures fuse the denoise loop in C++ kernels where the Python
counter can't see step boundaries. In that case any of the following are
drop-in fallbacks:

* **Static activation scale** — captured once over all steps; ~0.5 pp drop
  on libero_object versus per-step.
* **Dynamic per-token scale** — compute `amax` of the current activation at
  runtime; +5 % latency, +0.2 pp accuracy.
* **Drop the DiT-side W4A4** and keep just LLM-side W4A4 — the LLM is where
  most of the parameter weight lives.

These knobs all live on `QVLAConfig`.

## What's *not* in here

* Custom int4 matmul CUDA kernel. The runtime layer dequantizes to bf16 on the
  fly and calls the regular bf16 matmul (which on Hopper still benefits from
  the smaller activation memory traffic). If you need true int4 throughput,
  plug a kernel into `quant_linear.QuantLinear._matmul_kernel`.
* Tensor-parallel pack sharding. Single-GPU only for now; the layer falls
  back to bf16 when `tp_size > 1` (with a warning).
