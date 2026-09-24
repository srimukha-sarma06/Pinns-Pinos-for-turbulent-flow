# Physics-Informed Leak Localisation (DeepXDE + PyTorch)

Reconstructs the location and size of a leak in a pressurised pipeline from **two pressure
sensors** and a controlled valve transient, using a Physics-Informed Neural Network (PINN)
built on the 1-D transient continuity + momentum (water-hammer) equations with a
pressure-dependent leak sink term. See `PROBLEM_STATEMENT.md`-style detail in the chat history
for the full derivation; this README is about the code.

## Status — what's solid vs. what's still open

| Piece | Status |
|---|---|
| `leakpinn/physics.py` — real pipe/fluid/valve parameters | ✅ done. DN50 Sch40 steel, water at 20°C, Swamee-Jain friction, Korteweg wave speed |
| `leakpinn/moc.py` — Method-of-Characteristics ground-truth solver | ✅ done, **validated** (`scripts/01_validate_moc.py`, 10/10 checks pass) |
| `leakpinn/synth.py` — realistic synthetic sensor data | ✅ done (1 kHz, noise, 12-bit ADC quantisation) |
| `leakpinn/baselines.py` — time-of-flight & MOC-search+LM inversion | ✅ done, **tested on 6 scenarios** (`results/03_baseline_comparison.json`). Includes a fixed bug: `estimate_wave_speed` originally only searched the first 100 ms for the pulse transit, which silently failed (and fell back to a wrong default) whenever the true wave speed was much slower than typical steel-pipe values — now searches the full record |
| `leakpinn/pinn.py` — forward PINN (leak params **given**) | ✅ done, **validated**: 0.4% flow error vs MOC truth (`scripts/02_forward_pinn.py`) |
| `leakpinn/pinn.py` — **inverse** PINN (leak params **inferred**) | ⚠️ **not yet reliably convergent** — see below |

### The open issue, honestly

Letting the network *discover* the leak location by gradient descent (instead of being told it)
does not yet reliably converge to the true location from an arbitrary start. What I found while
debugging:

1. **Found and fixed a real bug**: the leak-position parameter used `torch.clamp`, which has
   zero gradient outside its bounds — once the optimiser pushed the raw value past the bound it
   froze there forever. Fixed with a smooth sigmoid reparameterisation (see `Params.get`).
2. **Confirmed it's not a genuine physics degeneracy.** I checked directly against the MOC
   ground-truth solver: a leak hypothesised near the reservoir fits the sensor data much worse
   (cost ≈ 20–30) than the true mid-pipe location (cost ≈ 6). So the correct answer really is
   identifiable from this data — the PINN's *training dynamics* are the problem, not the physics.
3. **Most likely cause**: with collocation points drawn uniformly over the whole (x, t) domain,
   very few of them land near the narrow Gaussian-smoothed leak term, so the gradient telling the
   optimiser "mass conservation is violated at the wrong location" is weak and noisy. I added
   `LeakZoomSampler`, an adaptive/importance-sampling callback that concentrates collocation
   points around the current leak-position estimate (redrawn periodically) — this is a standard
   fix for inverse-source PINNs, but I ran out of time to confirm it fully solves it (early runs
   improved but hadn't converged as of writing).

**Recommended next steps**, in order of effort:
- Let a longer run finish with `LeakZoomSampler` active (already wired into `fit()`) — this is
  the most promising fix and just needs more wall-clock time than this environment (single CPU
  core) allowed.
- Try a hard/exact junction condition at `x = x_L` (continuity of H, `Q_down = Q_up − CdA√H`)
  instead of a smoothed Gaussian sink — sharper gradient signal, more code.
- Multi-start (`multistart_fit` in `leakpinn/pinn.py`) + take the run with lowest final loss.

**For your submission**: the MOC-search baseline (`leakpinn/baselines.py::moc_search`) already
solves the inverse problem well (sub-metre accuracy on 5 of 6 test cases — see
`results/03_baseline_comparison.json`) and is fully working today. If the PINN inverse fit isn't
converged in time, present it as: forward PINN = fast differentiable surrogate (validated), leak
localisation = MOC-search (validated), PINN-based inversion = in-progress research direction with
a documented, diagnosed convergence issue. That is a defensible, honest story — and arguably more
"real research" than a demo that quietly works by luck.

## Repository layout

```
leakpinn/
  physics.py    Pipe / Valve / Leak dataclasses, all real engineering values
  moc.py        Method-of-Characteristics solver (ground truth + strong baseline)
  synth.py      Synthetic sensor data generator (noise, quantisation, sampling)
  baselines.py  time_of_flight(), moc_search() — working inverse solvers
  pinn.py       DeepXDE/PyTorch PINN: network architectures, training schedule, inverse fit
  metrics.py    Scoring against MOC truth (never fed to the model)
scripts/
  01_validate_moc.py       Sanity-checks the MOC solver against closed-form hydraulics
  02_forward_pinn.py       PINN with leak params GIVEN — validates the PINN formulation
  03_compare_baselines.py  time-of-flight & MOC-search across 6 test cases
results/                   JSON + PNG outputs from the scripts above
requirements.txt
```

## GPU

This code runs on GPU automatically if `torch.cuda.is_available()` (DeepXDE's own PyTorch backend
sets that as the default device on import — see `leakpinn/pinn.py` top of file for the
device-handling comment). Nothing to configure. The MOC solver (`moc.py`) and baselines
(`baselines.py`) are pure NumPy/SciPy and always run on CPU — that's fine, they're not the
bottleneck (the PINN training is).

## Renesas Ethos-U55 deployment (CPU-only, no NPU)

`scripts/05_export_onnx.py` trains a small forward PINN (2,274 parameters at the default
`width=32, depth=3`) and exports it to ONNX, then checks every op in the exported graph against
TensorFlow Lite Micro's actual kernel registry (`tensorflow/lite/micro/kernels/micro_ops.h`,
checked directly against current TFLM source — not guessed from memory). Since you're not using
the Ethos-U55 NPU, inference runs as plain CPU TFLM kernels, and **all four network architectures
in this project are supported this way**, including the Fourier-feature ones (`sin`/`cos` are
native TFLM ops: `Register_SIN()`, `Register_COS()`) — no need to avoid them.

```
python scripts/05_export_onnx.py
```

This produces `results/05_pinn_forward.onnx` and `results/05_export_report.json` (parameter
count, PyTorch-vs-ONNX numerical diff, and the op-support check). Next step (not run by the
script, needs a full TensorFlow install):

```bash
pip install onnx2tf tensorflow
onnx2tf -i results/05_pinn_forward.onnx -o results/05_tflite_model/
```

That `.tflite` file is what you hand to Renesas FSP's TFLM integration.

**On ST Edge AI Developer Cloud**: its platform selector only offers STM32 MCU / STM32 MPU /
Stellar MCU / MEMS ISPU — there is no Renesas option, so it cannot give real benchmark numbers
for this board. It can still load the ONNX/TFLite file for a generic sanity check, but the op
support / memory numbers it reports are for ST's own compiler, not Renesas's — treat it as a
secondary check at most, not a substitute for the TFLM op-list verification above.

## Setup

```bash
pip install -r requirements.txt
export DDE_BACKEND=pytorch
```

## Run

```bash
python scripts/01_validate_moc.py        # ~5 s   — confirms the physics/solver are correct
python scripts/03_compare_baselines.py   # ~5 min — working leak localisation, 6 scenarios
python scripts/02_forward_pinn.py        # ~8 min — PINN forward-model validation
```

Inverse PINN fitting (experimental, see above) — runs on GPU automatically if one is available:
```python
from leakpinn.synth import make_dataset
from leakpinn.pinn import PINNConfig, fit, multistart_fit

d = make_dataset(leak_x=37.3, CdA=3.6e-6, seed=0)
cfg = PINNConfig(arch="ff")          # try "modmlp" or "char" (characteristic-coordinate Fourier features) too
res = fit(d, cfg, verbose=True)      # or multistart_fit(d, cfg, n_starts=5) for several initial guesses
print(res.est)                       # {'x_L': ..., 'CdA': ..., 'a': ...}
```

## Architectures available (`PINNConfig.arch`)

- `"mlp"` — plain tanh MLP, no feature transform. Baseline; struggles with the sharp wave fronts.
- `"ff"` — random Fourier features (Tancik et al. 2020) on (x, t). **Default**, best general-purpose
  choice for the oscillatory wave field.
- `"modmlp"` — Fourier features + the Wang-Teng-Perdikaris "modified MLP" gating, which usually
  improves multi-scale PDE fits at some extra cost.
- `"char"` — Fourier features on the **characteristic coordinates** `(t−x/a, t+x/a)`, along which
  the lossless wave equation is exactly 1-D. In principle the best inductive bias for this
  problem; not extensively tuned here — worth trying first if you continue this.

I deliberately did **not** use a DeepONet: this problem has a fixed, known geometry and a single
leak per training run, so there's no family of operators to amortise over — a DeepONet would add
complexity without adding value here. (It would make sense if you wanted one network that
generalises instantly across many different pipes/valve schedules without retraining.)

## Optimisers

Training follows a staged schedule (configurable in `PINNConfig.schedule`):
1. **Adam**, leak parameters frozen — warm up the network on the background wave field.
2. **Adam**, leak parameters free, with `eps` (leak-source width) annealed from wide → physical —
   gives the leak position a large basin of attraction before sharpening it.
3. **L-BFGS** (`maxcor=60`, tight tolerances, strong-Wolfe line search) — standard second-stage
   optimiser for PINNs once Adam is in the right basin; converges the residuals much further than
   Adam alone.

I also wired up **NNCG** (Nyström-preconditioned Newton-CG, Rathore et al. 2024) as an optional
third stage (`dict(opt="nncg", iters=...)` in the schedule) — a genuine second-order method for
PINNs that can beat L-BFGS on hard residuals. I didn't end up needing it here once L-BFGS was
converging the *forward* problem well, but it's there if the inverse fit needs it.

I did not reach for a plain Newton-Raphson root-find: this is a least-squares/loss-minimisation
problem, not a root-finding one, and it's exactly what L-BFGS/NNCG are already designed for.

## Real-world validity of every number used

- Pipe: DN50 Schedule 40 carbon steel (`D`=52.5 mm, wall 3.91 mm, per ASME B36.10).
- Water at 20 °C: ρ=998.2 kg/m³, μ=1.002×10⁻³ Pa·s, bulk modulus K=2.18 GPa.
- Wave speed from the standard Korteweg formula with thin-wall elasticity: **1388 m/s** (matches
  published water-hammer references for steel pipe).
- Friction factor from Swamee-Jain (explicit Colebrook approximation, <1% error, valid for our
  Reynolds number ≈ 48,000).
- Valve closure: 20% opening change in 20 ms — achievable with a fast pneumatic/servo valve;
  produces an ≈11.4 m (1.1 bar) Joukowsky pulse, safely below the pipe's pressure rating.
- Sensor model: 1 kHz sampling, 0.10 m (≈1 kPa) noise, 12-bit ADC over a 10 bar span — typical of
  an industrial dynamic pressure transmitter.
- Leak sizes tested: CdA = 0.9–3.6 mm² (roughly 1–5% of the design flow) — physically a small
  puncture or corrosion pit, not a catastrophic rupture.

None of these were picked to make the results look good; they came from datasheet-typical values
and standard hydraulics references, and `scripts/01_validate_moc.py` checks the simulator
reproduces the textbook closed-form results before anything downstream is trusted.
