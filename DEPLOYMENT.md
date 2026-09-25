# Deployment Handoff — Leak-Locator PINN on Renesas (Ethos-U55, CPU-only)

Everything the person deploying this needs to know, in one place. Written for the RA8 +
Ethos-U55 board, **NPU not used** — inference runs as plain TensorFlow Lite Micro (TFLM) on the
Cortex-M CPU.

**Two deployable forward models now exist — pick the one that matches your situation:**

- **§1–§7 below**: the ORIGINAL model, trained on and valid for exactly ONE pipe (100 m, DN50
  steel, leak at 37.25 m). Highest accuracy (0.4% flow error) but must be retrained and
  re-exported for any different pipe.
- **§8 (bottom of this file)**: the GENERALIZED model — one trained network, one `.tflite` file,
  that works across a *range* of pipes (20–300 m, DN25–DN200, steel through HDPE) without
  retraining, at the cost of noticeably lower accuracy (~15–20% field error vs. 0.4%). Use this
  one if you need to cover more than one physical pipe with the same deployed model.

## 1. What you're actually deploying

`results/05_pinn_forward_fp32.onnx` — a **forward** model: given a position and time along the
pipe, it predicts pressure head and flow rate there.

- **2,274 parameters.** Trivially small for any MCU with tens of KB of flash.
- Architecture: plain MLP — `Linear(2→32) → Tanh → Linear(32→32) → Tanh → Linear(32→32) → Tanh → Linear(32→2)`,
  plus a small fixed multiply/subtract at the output (hard constraint enforcing the boundary
  conditions — not a learned layer, just arithmetic).
- Trained on **one specific scenario** (100 m pipe, DN50 steel, leak at 37.25 m, 3.6 mm² CdA).
  It is not a general-purpose model — retrain (`scripts/02_forward_pinn.py` /
  `05_export_onnx.py`) for a different pipe or leak before deploying a new scenario.

**This is NOT the leak-locator.** This model answers "what's the pressure/flow at this point,
given I already know where the leak is." It does not find the leak itself — see §5.

## 1a. Input / output contract — read this before wiring anything up

**Input** — tensor `xi_tau`, shape `[batch, 2]`, **float32**:
| index | meaning | convert from real units with |
|---|---|---|
| 0 | `xi` — normalized position (dimensionless) | `xi = x_metres / 100.0` |
| 1 | `tau` — normalized time (dimensionless) | `tau = t_seconds * 1399.017 / 100.0` |

**Output** — tensor `h_q`, shape `[batch, 2]`, **float32**. These are NOT physical values —
they're normalized perturbations from the pre-transient steady state. Post-processing required:

```
// --- head, in metres ---
slope   = (39.780 - 37.949) / (1 - 0.1);              // = 2.034
H_ss    = 37.949 + slope * (1 - xi);                  // steady baseline at this xi
H       = H_ss + 10.0 * h;                            // h = output[0]

// --- flow, in m^3/s (leak params are FIXED/baked in for this export -- precompute as a
//     lookup table at build time instead of calling erf() at runtime if you can avoid it) ---
xiL     = 0.3725;  kappa = 1.0507;  qv0 = 131.679;  eps = 0.03;   // baked-in leak constants
H_L     = 37.949 + slope * (1 - xiL);                 // = 38.707
S       = 0.5 * (1.0 + erf((xi - xiL) / (sqrt(2.0) * eps)));
q_ss    = qv0 + kappa * sqrt(H_L) * (1.0 - S);
Q       = (q_ss + 10.0 * q) / 65901.27;                // q = output[1]
```

**Datatype — was float64, fixed to float32.** The training pipeline runs in float64 for
numerical stability, which is right for training but wrong for an MCU (most Cortex-M cores only
have hardware FPU for single precision; several TFLite/TFLM kernels don't support float64 at
all). `scripts/05_export_onnx.py` now explicitly casts the trained weights to float32 before
export. Verified cost of that cast: max output difference vs. the original float64 model is
3.5×10⁻⁶ — negligible against the physical scale (10 m head scale, ~1.5×10⁻⁵ m³/s flow scale).
If you ever regenerate this export yourself, confirm the ONNX graph's declared dtype is
`FLOAT` (not `DOUBLE`) before building — check with:
```python
import onnx; m = onnx.load("results/05_pinn_forward_fp32.onnx")
print(onnx.TensorProto.DataType.Name(m.graph.input[0].type.tensor_type.elem_type))  # expect FLOAT
```

## 2. Ops used — confirmed supported, don't second-guess this

Exported graph uses exactly: `Gemm, Tanh, Mul, Sub, Div, Concat, Slice`.

Every one of these is confirmed present in TFLM's actual kernel source
(`tensorflow/lite/micro/kernels/micro_ops.h`) as of this check — not assumed from an older
Renesas doc. If you use `MicroMutableOpResolver` with selective registration, register:

```cpp
resolver.AddFullyConnected();   // Gemm
resolver.AddTanh();
resolver.AddMul();
resolver.AddSub();
resolver.AddDiv();
resolver.AddConcatenation();
resolver.AddSlice();
```

Or just use the all-ops resolver if flash budget allows — simpler, bigger binary.

**Do not add Fourier-feature architectures (`arch="ff"/"modmlp"/"char"`) without re-checking.**
They use `sin`/`cos`, which ARE also confirmed present in TFLM (`Register_SIN`/`Register_COS`)
— so they're fine too if you want the extra accuracy — but if you switch architectures, rerun
`scripts/05_export_onnx.py` to re-verify the op list, don't assume.

**`Softplus` and `Clip`/`clamp` are NOT native TFLM ops** — but they only appear in this
project's training-time code, never in the exported forward-model graph. Confirmed by the op
list above containing neither. If a future export ever shows either, stop and flag it — it means
someone accidentally exported the training/parameter-fitting code instead of the network.

## 3. What's actually verified vs. what isn't

| Claim | Status |
|---|---|
| ONNX export is numerically identical to the PyTorch model | ✅ Verified — max diff 1.9×10⁻⁷ |
| Every op in the exported graph exists in TFLM's kernel registry | ✅ Verified against current TFLM source |
| The model correctly predicts pressure/flow (vs. a real solved-physics ground truth) | ✅ Verified — 0.42% flow error vs. MOC solver (`results/02_forward_pinn.json`) |
| The model runs correctly *on the actual board* | ❌ **Not tested.** No hardware-in-the-loop check was done — I don't have access to Renesas hardware. Op-list matching means it *should* work, not that it's been proven to. |
| Inference speed/latency/RAM footprint on this specific MCU | ❌ **Not measured.** Need to build it on the actual board and profile. |
| `.tflite` conversion (onnx2tf) | ❌ **Not run.** Only the ONNX file exists; conversion command is documented but untested. |
| ST Edge AI Developer Cloud check | ❌ **Not applicable** — that tool has no Renesas board option at all (STM32/Stellar/MEMS ISPU only), so it was never a valid check for this hardware in the first place. |

## 4. Exact next steps to actually get this running

```bash
pip install onnx2tf tensorflow
onnx2tf -i results/05_pinn_forward_fp32.onnx -o results/05_tflite_model/
```

Then import the resulting `.tflite` into Renesas FSP's TFLM integration (e² studio), register
the ops listed in §2, and build. **First real test should be a numerical sanity check on-device**
— feed it the same 200 test points used in `scripts/05_export_onnx.py`'s verification step and
confirm the on-device output matches the ONNX output in `results/05_export_report.json` to a
reasonable tolerance (float32 rounding aside). Don't just trust that it compiles.

## 5. The bigger picture — don't deploy the wrong model

There are two different things in this project, and only one is ready:

- **Forward model (this one)**: predicts pressure/flow given a known leak. Validated, tested,
  exportable, ready. This is what `05_export_onnx.py` exports.
- **Inverse model** (finding the leak location from sensor data automatically): **not working
  reliably**. Full details in the main `README.md` under "Known limitation." If anyone asks for
  "the leak detector" to be deployed, make sure they mean this forward/virtual-sensor model
  (paired with the working classical `moc_search`/`time_of_flight` baselines running on a
  bigger processor elsewhere, e.g. a gateway or PC) — not a self-contained on-device leak
  locator, because that part of the project isn't there yet.

## 6. Physical scope — when this model stops being valid

Baked into training, not configurable at inference time without retraining:
- Pipe: 100 m, DN50 Schedule 40 steel, water at 20°C
- Leak at 37.25 m, 3.6 mm² effective orifice area
- Valve closure: 20% opening change over 20 ms
- Wave speed ~1388 m/s (assumes steel pipe, no significant entrained air)

If the real pipe differs from these (different material, diameter, leak size/location, or valve
timing), **this exact model will give wrong answers** — it doesn't generalize. Either retrain via
`scripts/02_forward_pinn.py` / `05_export_onnx.py` with `leakpinn/physics.py`'s `Pipe`/`Valve`
parameters updated to match the real hardware, or switch to the generalized model in §8 below if
you need one deployed model to cover more than one physical pipe.

## 7. Who to ask if something breaks

This model and its export were built and verified in a Linux sandbox with **no access to actual
Renesas hardware, no GPU (for most of the build), and no browser/upload tooling** — everything
here is verified as far as software-only checks can go (physics validation against textbook
water-hammer theory, numerical export equivalence, op-list matching against TFLM source). The
gap between "should work" and "works on the board" is real and untested. Budget time for
on-device debugging; don't treat this handoff as a guarantee.

---

## 8. The GENERALIZED model — one deployed model, a range of pipes

### 8.1 What it is

`results/09_general_forward_fp32.tflite` (78 KB, fp32, 16,194 parameters) — same kind of forward
model as §1 (predicts head/flow given a known leak), but pipe geometry, wave speed, friction and
leak position/size are **runtime inputs** instead of constants baked in at training time. Trained
across randomly sampled pipes in the range documented in `README.md`'s "Generalized forward model"
section (length 20–300 m, DN25–DN200, wave speed 300–1400 m/s, leak 0.5–10% of design flow, leak
anywhere from 5–95% of the pipe). One `.tflite` file, no retraining, for any pipe in that range.

Produced by, in order: `scripts/07_train_general_forward.py` (train + validate against MOC on
held-out pipes) → `scripts/08_export_general_onnx.py` (ONNX export + TFLM op check) →
`scripts/09_convert_tflite.py` (TFLite fp32 conversion + numerical verification).

### 8.2 Input / output contract

Input tensor `scenario_xy`, shape `[1, 10]`, **float32**. Every column is dimensionless and is
computed from your pipe's raw physical parameters exactly the same way §1's `xi`/`tau` conversion
already required — nothing here needs a special function beyond `sqrt`, all ordinary host-side
arithmetic (see `leakpinn.general_pinn.scenario_consts`/`DeployNet` for the reference Python):

| idx | name | formula |
|---|---|---|
| 0 | `xi` | `x_metres / L` |
| 1 | `tau` | `a * t_seconds / L` |
| 2 | `xiL` | `leak_x_metres / L` |
| 3 | `kappa` | `B_nom * CdA * sqrt(2*9.80665)` |
| 4 | `phi` | `f * L * 9.80665 / (2*D*a**2)` (`f` = Darcy friction factor at design flow, Swamee-Jain) |
| 5 | `H1n` | reference head near the reservoir [m] / 100 |
| 6 | `H2n` | reference head at the valve [m] / 100 |
| 7 | `xi1` | reference-point position / `L` |
| 8 | `B_nom` | `a / (9.80665 * A)`, `A` = pipe cross-section area |
| 9 | `qv0` | `B_nom * Q_design` |

Output tensor `H_Q`, shape `[1, 2]`, float32: `[H_metres, Q_m3_per_s]` — **already physical
units**. Unlike §1's contract, there is **no separate host-side reconstruction formula to hand-code
per pipe** (no `erf`, no manually-typed constants) — the steady-profile reconstruction is baked
into the graph itself, because those constants are no longer fixed at export time. This is a
deliberate simplification over §1's contract, not an oversight: with leak position/size varying at
runtime, per-deployment hand-written reconstruction code would need updating for every pipe anyway.

### 8.3 Ops used — confirmed supported, checked twice

Same TFLM kernel registry as §2, checked at TWO levels (`scripts/08` checks the ONNX graph,
`scripts/09` re-checks the actual compiled `.tflite`, since onnx2tf's own lowering can rewrite an
op): `Gemm/MatMul → FULLY_CONNECTED` and (for the Fourier-feature matrix multiply specifically)
`BATCH_MATMUL`, `Tanh`, `Sin`, `Cos`, `Mul`, `Sub`, `Add`, `Div`, `Sqrt`, `Log`, `Relu`, `Concat`,
and `Slice`/`SPLIT` (repeated slices get fused into one `SPLIT` op by onnx2tf). All confirmed
present in `tensorflow/lite/micro/kernels/micro_ops.h` — `Register_BATCH_MATMUL()` and
`Register_SPLIT()` specifically re-checked against the live TFLM source for this model, not
assumed. `Erf` and `Clip` (no native TFLM kernels) are never used — see `leakpinn/general_pinn.py`'s
module docstring for the Tanh/Relu replacements and why they're needed here but weren't in §1.

### 8.4 What's actually verified vs. what isn't

| Claim | Status |
|---|---|
| ONNX export is numerically identical to the PyTorch model | ✅ max diff 1.5×10⁻⁵ |
| TFLite (fp32) export matches the ONNX graph, batch=1 (matching on-device usage) | ✅ max diff 1.5×10⁻⁵ over 32 random points |
| Every op in the compiled `.tflite` exists in TFLM's kernel registry | ✅ re-verified on the actual `.tflite`, not just the ONNX graph |
| Predicts pressure/flow correctly across the trained pipe range (vs. MOC ground truth) | ⚠️ **~21% head field error, ~14% flow field error** on 12 held-out pipes — usable as a rough virtual sensor, not a precision one. See README.md for what was tried to improve this and why a bigger/longer run wasn't a clear win. |
| The model runs correctly *on the actual board* | ❌ **Not tested** — same caveat as §7, no Renesas hardware access here |
| `.tflite` conversion (onnx2tf) | ✅ **Run and verified** (unlike §1, where this step was left for you) |

### 8.5 A build-environment gotcha, in case you hit it too

Exporting to ONNX (`torch.onnx.export`) **segfaulted** in the environment this was built in
whenever a CUDA context was active — reproduced independently of this specific model, so it's a
torch/CUDA-driver interaction issue, not a bug in the network. `scripts/08_export_general_onnx.py`
works around it by setting `CUDA_VISIBLE_DEVICES=""` before `torch` is even imported (export never
needs a GPU for a network this small anyway). If you hit a segfault re-running `torch.onnx.export`
yourself on a machine with a GPU, try the same thing first.

### 8.6 When this model stops being valid

Outside the trained range (README.md's table: length 20–300 m, DN25–DN200, wave speed 300–1400
m/s, leak 0.5–10% of design flow, leak position 5–95% of the pipe, valve closure fixed at 20% over
20 ms) — extrapolation is untested. Narrow the trained range in `leakpinn/domain.py` and retrain
(`scripts/07`–`09`) if your actual fleet of pipes is a tighter subset than this; a narrower range
converges to noticeably better accuracy, the same way §1's single-pipe model reaches 0.4%.
