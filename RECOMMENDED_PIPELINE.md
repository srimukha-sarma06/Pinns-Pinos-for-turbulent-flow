# Recommended pipeline

This is the end-to-end system this repo actually recommends deploying, after building and
honestly evaluating an ML-based alternative for leak localization (see §5 — it did not beat the
classical method, so it is NOT part of this recommendation). Two components, each already built,
validated, and exported in this repo. Neither is a novel invention here: both are the pieces this
project found to actually work, as opposed to the more ambitious ones that didn't pan out.

## 1. System overview

```
  [ two pressure transducers, valve actuator ]
                    |
                    v
  1 kHz pressure transient capture, both sensors, synchronized to valve-closure trigger
                    |
                    v
  ┌─────────────────────────────────────────────────────────────┐
  │  STEP A -- LEAK LOCALIZATION (gateway / PC, not the MCU)      │
  │  leakpinn.baselines.moc_search()                              │
  │  in: two sensor traces + known pipe/valve parameters          │
  │  out: leak position x_L, leak size CdA, corrected wave speed a│
  └─────────────────────────────────────────────────────────────┘
                    |
                    v
  ┌─────────────────────────────────────────────────────────────┐
  │  STEP B -- VIRTUAL SENSOR (Renesas RA8 + Ethos-U55, CPU only)│
  │  leakpinn.pinn.PINNNet  OR  leakpinn.general_pinn.GeneralNet  │
  │  in: query position + time (+ pipe context for the general   │
  │      model) -- leak params from Step A feed in as constants   │
  │  out: predicted head H(x,t) and flow Q(x,t) anywhere along    │
  │      the pipe, without needing a physical sensor there        │
  └─────────────────────────────────────────────────────────────┘
```

Step A runs once (or occasionally, whenever a transient is captured) on a gateway/PC — it is a
search algorithm around a full physics solver, not a tiny always-on model, and doesn't belong on
the MCU. Step B is the piece that actually goes on the Renesas board: a small trained network that
answers "what's the pressure/flow here" continuously, using Step A's result (or design values, if
no leak is suspected) as its leak-parameter inputs.

## 2. Step A: leak localization

**What it is**: `leakpinn.baselines.moc_search()` — brute-force grid search over candidate
`(x_L, CdA)` using the actual MOC water-hammer solver (`leakpinn/moc.py`, independently validated
against closed-form hydraulics, `scripts/01_validate_moc.py`) to simulate each candidate and score
it against the measured sensor traces, followed by Levenberg-Marquardt local refinement of
`(CdA, a)` and a local re-scan of `x_L`. Not a neural network — the "model" here is the physics
solver itself, used as the forward simulator inside an optimization loop.

**Inputs** (all realistically available — see the discussion earlier in this project about which
of these are actually knowable before a leak is found):
- Two pressure sensor traces, 1 kHz, synchronized to the valve-closure trigger (`Dataset.H_meas`,
  `Dataset.H_pre` in `leakpinn/synth.py`'s format — or the equivalent from real field logging)
- Pipe geometry (`L`, `D`), material/roughness, reservoir head, valve schedule (all installation
  constants) -- passed via `leakpinn.synth.Dataset.pipe` / `.valve`
- A wave-speed estimate (nominal, or refined by `leakpinn.baselines.estimate_wave_speed()` from
  the same sensor traces)
- Search ranges: `x_grid` (candidate positions) and `cda_grid` (candidate leak sizes) — **must be
  scaled to the actual pipe**, not left at the function's defaults, which were sized for one
  specific 100 m pipe. See `scripts/10_train_localization.py`'s comparison code for a
  scenario-appropriate way to build these from the pipe's own `Q_design`/`H_res`/`L`.

**Outputs**: `dict(x_L, CdA, a, cost, seconds)` — leak position [m], leak size [m²], corrected
wave speed [m/s], the achieved data-misfit cost, and how long the search took.

**Accuracy** (30 held-out pipes across the "moderate industrial range" — 20-300 m, DN25-DN200,
300-1400 m/s wave speed, leak 0.5-10% of design flow; `results/10_localization_comparison.json`):
mean absolute error **2.1 m**, median **0.3 m**, 90% of cases within 1 m, worst case 30 m. On the
original single fixed pipe this project started from (`results/03_baseline_comparison.json`):
sub-metre on 5 of 6 test cases, with one documented failure mode (a very small leak, ~1% of design
flow, where the search converges to the wrong location by ~21 m — a real, honest limitation, not
hidden).

**Cost**: ~7-15 s per case (scales with pipe length and search-grid resolution) — a search over a
physics simulator, not a single inference. Runs on a gateway/PC, not embedded hardware.

## 3. Step B: virtual sensor (deployed to the MCU)

Two trained variants exist. Pick based on whether you're deploying to one specific pipe or need
one model to cover a range of pipes.

### 3a. Single-pipe specialist (highest accuracy, one fixed pipe)

**Model**: `leakpinn.pinn.PINNNet` — `Linear(2->32) -> Tanh -> Linear(32->32) -> Tanh ->
Linear(32->32) -> Tanh -> Linear(32->2)`, plus a fixed output multiply enforcing the reservoir
boundary condition and zero initial condition (arithmetic, not a learned layer). 2,274 parameters.

**Input**: tensor `xi_tau`, shape `[1, 2]`, float32 — `[xi, tau]` where `xi = x_metres / L`,
`tau = a * t_seconds / L` (host computes this from the query point and the pipe's own `L`, `a`).

**Output**: tensor `h_q`, shape `[1, 2]`, float32 — dimensionless perturbations `[h, q]`, requiring
a short host-side reconstruction (steady-profile addition, documented formula) to get physical
head `H` [m] and flow `Q` [m³/s]. Full formula in `DEPLOYMENT.md` §1a.

**Trained on**: one specific scenario (100 m, DN50 steel, leak at 37.25 m, 3.6 mm² CdA) — retrain
(`scripts/02_forward_pinn.py` / `05_export_onnx.py`) for a different pipe.

**Accuracy**: 0.42% flow error, 0.59% head NRMSE vs. MOC ground truth
(`results/02_forward_pinn.json`). This is the number to trust if you only ever deploy to one pipe.

**Artifacts**: `results/05_pinn_forward_fp32.onnx`, `results/05_tflite_model/*_float32.tflite`.

### 3b. Generalized model (works across a range of pipes, lower accuracy)

**Model**: `leakpinn.general_pinn.DeployNet` wrapping `GeneralNet` — random-Fourier-feature input
transform on `(xi, tau)`, 4 hidden `Linear(64) -> Tanh` layers, then a fixed arithmetic
reconstruction (steady profile + Tanh-based leak smoothing, replacing the single-pipe model's
`erf` since that has no native TFLM kernel) baked into the same exported graph. 16,194 parameters.

**Input**: tensor `scenario_xy`, shape `[1, 10]`, float32:

| idx | name | meaning |
|---|---|---|
| 0 | `xi` | query position / `L` |
| 1 | `tau` | `a * t / L` |
| 2 | `xiL` | leak position / `L` (known, e.g. from Step A) |
| 3 | `kappa` | leak conductance, `B_nom * CdA * sqrt(2*9.80665)` |
| 4 | `phi` | friction number, `f*L*9.80665/(2*D*a**2)` |
| 5 | `H1n` | reference head near reservoir [m] / 100 |
| 6 | `H2n` | reference head at the valve [m] / 100 |
| 7 | `xi1` | reference-point position / `L` |
| 8 | `B_nom` | pipe impedance, `a/(9.80665*A)` |
| 9 | `qv0` | `B_nom * Q_design` |

**Output**: tensor `H_Q`, shape `[1, 2]`, float32 — `[H_metres, Q_m3s]`, **already physical units**,
no host-side reconstruction needed (unlike 3a — the leak parameters are runtime inputs here, so
there's no fixed formula to hand-code, and the reconstruction is in the graph instead).

**Trained on**: pipes randomly sampled from 20-300 m length, DN25-DN200, 300-1400 m/s wave speed,
20-80 m reservoir head, leak 0.5-10% of design flow anywhere from 5-95% of the pipe, fixed 20%/20ms
valve closure (`leakpinn/domain.py`).

**Accuracy**: ~21% head field NRMSE, ~14% flow field NRMSE vs. MOC ground truth on 12 held-out
pipes (`results/07_general_pinn_validation.json`) — noticeably rougher than 3a. Use this only when
one deployed model genuinely needs to cover multiple pipes; narrow `leakpinn/domain.py`'s `RANGES`
to your actual fleet and retrain if you need better accuracy than this.

**Artifacts**: `results/08_general_forward_fp32.onnx`, `results/09_general_forward_fp32.tflite`.

### Ops, both variants

Every op in both exported graphs is confirmed present in TFLM's kernel registry
(`tensorflow/lite/micro/kernels/micro_ops.h`): `Gemm/MatMul` (incl. `BATCH_MATMUL` for 3b's
Fourier-feature matmul), `Tanh`, `Sin`, `Cos`, `Mul`, `Sub`, `Add`, `Div`, `Sqrt`, `Relu`,
`Concat`, `Slice`/`SPLIT`. No `Erf`, no `Clip` (`torch.clamp`) — neither has a native TFLM kernel;
both variants avoid them by construction (Tanh-based smoothing, Relu-based flooring).

## 4. Choosing which forward variant to deploy

| | 3a: single-pipe | 3b: generalized |
|---|---|---|
| Accuracy | 0.4% flow error | ~14-21% field error |
| Covers | one exact pipe | a range of pipes, no retraining |
| Retrain needed for a new pipe? | yes | no (if within the trained range) |
| Model size | 2,274 params | 16,194 params |
| Recommendation | default choice if you deploy to one known pipe | only if one model must serve multiple pipe types |

## 5. What was tried and is NOT in this recommendation

An amortized regression network (sensor trace -> leak location/size directly, no search) plus a
physics-informed refinement stage was also built and evaluated end-to-end (training data
generator, TFLM-safe export, `.tflite` conversion — all working code, still in the repo:
`leakpinn/localize.py`, `leakpinn/localize_data.py`, `scripts/10`-`13`). It does **not** beat
`moc_search`: 13.3 m mean error vs. `moc_search`'s 2.1 m on the same 30 held-out pipes
(`results/10_localization_comparison.json`), and the physics-refinement stage specifically does
not help, because the generalized forward model's own ~21% error means its loss surface isn't
reliably minimised at the true answer (confirmed directly: an unconstrained global gradient-free
search over the same surrogate found a *better-scoring but more wrong* answer on all 10 cases
tested). The one real advantage of that approach is inference speed (~0.09 ms vs. `moc_search`'s
~10 s per case) — worth revisiting only if that speed matters more than accuracy for your use
case, or as a starting-point search reduction rather than the final answer. Full writeup:
`INVERSE_LOCALIZATION_PLAN.md` and the conversation history that built it.

## 6. File manifest

```
Step A (localization):
  leakpinn/baselines.py           moc_search(), time_of_flight(), estimate_wave_speed()
  leakpinn/moc.py                 MOC solver (validated ground truth + search's forward simulator)
  results/03_baseline_comparison.json   accuracy on the original 6 test cases

Step B (virtual sensor), single-pipe:
  leakpinn/pinn.py                PINNNet, training (fit())
  results/05_pinn_forward_fp32.onnx
  results/05_tflite_model/*_float32.tflite
  results/02_forward_pinn.json    accuracy numbers
  DEPLOYMENT.md §1-§7             full input/output contract, deployment notes

Step B (virtual sensor), generalized:
  leakpinn/domain.py              pipe-scenario sampler (the trained range)
  leakpinn/general_pinn.py        GeneralNet, DeployNet, training
  results/08_general_forward_fp32.onnx
  results/09_general_forward_fp32.tflite
  results/07_general_pinn_validation.json   accuracy numbers
  DEPLOYMENT.md §8                full input/output contract, deployment notes

Not recommended, kept for reference (see §5):
  leakpinn/localize.py, leakpinn/localize_data.py, scripts/10-13
  INVERSE_LOCALIZATION_PLAN.md
```

## 7. Known limitations, stated plainly

- Step A's search-grid ranges (`x_grid`, `cda_grid`) must be sized to the actual pipe being
  searched — the function's own defaults are sized for one specific pipe and will silently
  underperform (or, before the fix made during this project, crash) outside that scale.
- Step A fails on very small leaks (~1% of design flow) in the one documented test case — a real,
  known limitation, not a hypothetical one.
- Step B's generalized variant (3b) is meaningfully less accurate than the single-pipe variant
  (3a) — use 3a whenever the deployment target is a single known pipe.
- Wave speed is treated as known/estimable throughout (via `estimate_wave_speed()` or a design
  value) — genuine wave-speed uncertainty beyond that estimate is not separately modelled anywhere
  in this pipeline.
