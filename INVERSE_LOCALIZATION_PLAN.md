# Turning this into a leak-localization (inverse) system — plan

Status: **planning document, nothing in this file is implemented yet.** Written before any code
changes, per the request to lay out the methodology first. Covers what exists today, what
literature says about this exact class of problem, and the step-by-step plan for what to build.

---

## 1. What exists in this repo today (the resources this plan builds on)

| Piece | What it does | Relevant because |
|---|---|---|
| `leakpinn/moc.py` | Method-of-Characteristics water-hammer solver, validated against closed-form hydraulics (`scripts/01_validate_moc.py`, 10/10 checks pass) | The trusted physics simulator / label generator for any training data this plan needs |
| `leakpinn/domain.py` | Samples random `(Pipe, Valve, Leak)` scenarios from a "moderate industrial range" (20–300 m, DN25–DN200, 300–1400 m/s, leak 0.5–10% of design flow, anywhere 5–95% of the pipe) | Already exactly the scenario generator needed to build a large labelled inverse-problem training set |
| `leakpinn/synth.py` | Turns one MOC run into realistic noisy, quantised, 1 kHz two-sensor pressure traces | Already the sensor-data synthesiser this plan needs, unchanged |
| `leakpinn/baselines.py` — `moc_search()` | Brute-force + Levenberg-Marquardt search over `(x_L, CdA, a)` using the MOC solver itself | The working reference to beat/match; also the source of the "small leak fails badly (−21 m)" honest baseline number from `results/03_baseline_comparison.json` |
| `leakpinn/pinn.py` — single-pipe **inverse** PINN (`Params`, `LeakZoomSampler`, `EpsAnneal`, `multistart_fit`) | Joint per-case gradient descent on `(x_L, CdA)` alongside the network weights, for ONE fixed pipe. Documented as **not reliably convergent** | The one prior attempt at this exact problem in this codebase — its diagnosed failure mode (see §2) is the starting point for this plan, not something to ignore |
| `leakpinn/general_pinn.py` — generalized **forward** model | One network, conditioned on `(xiL, kappa, phi, H1n, H2n, ...)`, predicts `H(x,t), Q(x,t)` for any pipe in the trained range. ~21% field NRMSE on held-out pipes | A fast, differentiable, GPU-batchable surrogate for "what would sensor data look like given a hypothesized leak" — reusable as the physics engine inside an inversion loop instead of running full MOC each time |
| `leakpinn/metrics.py` | Scores against MOC truth that is never shown to the model | The discipline this plan's validation reuses unchanged |

**What was explicitly confirmed in the previous turn of this conversation**, and drives the whole
design below: of the generalized forward model's 10 inputs, **`xiL` (leak location) and `kappa`
(leak size) are the two that are NOT available at real inference time** — everything else (pipe
geometry, reference heads, design flow, friction number) comes from as-built records or direct
sensor readings. Those two are exactly what an inverse/localization system has to produce.

## 2. Why the existing single-pipe inverse attempt struggled (repo's own diagnosis)

From `README.md`'s "Known limitation" section, three things were already found:

1. A real bug (`torch.clamp` killing gradients past the parameter bound) — fixed via smooth
   sigmoid/softplus reparameterisation, already in `Params.get()`.
2. **Confirmed not a physics degeneracy** — a leak hypothesised at the wrong location scores
   measurably worse (cost ≈ 20–30) against real MOC-simulated data than the true location
   (cost ≈ 6). The answer is identifiable from the data.
3. **The actual bottleneck is training dynamics**: with collocation points drawn uniformly over
   the whole `(x, t)` domain, very few land near the narrow leak term, so the gradient telling the
   optimiser "the leak is in the wrong place" is weak and noisy. `LeakZoomSampler` (importance
   sampling around the current estimate) was added as a partial fix but never confirmed to
   converge reliably, and gradient descent from an **uninformed starting guess** on a genuinely
   non-convex, weak-gradient landscape is inherently fragile — `multistart_fit` (several random
   starting guesses, keep the best) was the other partial mitigation.

## 3. What the literature says about this exact problem

Searched before writing this plan (not assumed from memory):

- **General inverse-PINN failure modes** match what's already diagnosed here almost exactly:
  non-convex loss landscapes, data-vs-physics loss imbalance, and noise sensitivity are the
  standard documented causes of unreliable convergence in joint per-case gradient-descent
  inversion ([Inverse PINN: Parameter Recovery & Inference](https://www.emergentmind.com/topics/inverse-pinn)).
  Documented mitigations beyond what's already in this repo: constrained optimisation via the
  Modified Differential Method of Multipliers (PINNverse reports up to 370× lower parameter error
  vs. unconstrained weighting, and robustness under 30% data noise —
  [PINNverse](https://arxiv.org/pdf/2504.05248)), and **amortized inference**: train a neural
  encoder to map observations directly to parameters instead of solving an optimisation problem
  per instance, explicitly framed in the literature as the alternative to optimization-based /
  Bayesian approaches for exactly this kind of ill-posed inversion
  ([InVAErt networks](https://arxiv.org/html/2408.08264)).
- **This exact domain (water-hammer / pressure-transient leak localization) already has published
  precedent for the amortized/direct-regression approach**, not just generic ML theory: leak
  detection formulated as **three supervised tasks — leaking-pipe classification, leak-location
  regression, and leak-size regression** — trained on simulated transients with measurement-noise
  augmentation, using CNN/LSTM/GRU/Transformer architectures over the sensor time series. Reported
  results: 95% of test cases located within 3.0 m on a 1000 m pipe, leak size mean absolute error
  0.31 mm ([ASCE J. Water Resources Planning and Management](https://ascelibrary.org/doi/10.1061/(ASCE)WR.1943-5452.0001187);
  similar numeric-data-generation + ANN approach in
  [Burst Localisation in Pressurised Pipelines](https://doi.org/10.3390/engproc2024069019)).
  A related paper explicitly studies the **simulation-to-field transfer gap** for this exact
  problem class ([Explainable simulation-to-field transfer learning for transient-based leak
  detection](https://pubmed.ncbi.nlm.nih.gov/42492208/)) — relevant because this repo's training
  data is 100% MOC-simulated, same as that paper's starting point.
- **Physics-informed variants of the same idea exist**: a multi-head PINN that simultaneously
  classifies leak state, estimates size, and localizes leaks in multiphase systems reports high
  accuracy (~96% localization) by combining a learned/data-driven head with a physics-residual
  loss term ([ScienceDirect, multi-head PINN](https://www.sciencedirect.com/science/article/pii/S0957582026001497) —
  abstract/summary only, full text paywalled, methodology details not independently confirmed).

**Conclusion driving this plan**: the published, domain-specific precedent uses **supervised
regression on simulated transients**, not per-case gradient descent on a joint PINN loss — which
lines up exactly with why this repo's one existing attempt at the latter didn't converge
reliably. The plan below is a hybrid that keeps physics-informed refinement (this project's whole
ethos) but stops relying on it to also do the job of "find the answer from a cold start."

---

## 4. If you have the other paper to compare against

You mentioned wanting me to look at a specific paper before finalizing this. I don't have it yet
— share it (link, DOI, or file) and I'll fold in a §3.5 comparing its formulation/architecture to
what's below before we start building, rather than revise course mid-implementation.

---

## 5. Proposed methodology

**Two-stage, not one.** Stage A does the hard part (get close to the right answer, reliably,
without gradient descent). Stage B is a short physics-consistent polish starting from Stage A's
answer, so it converges instead of wandering a non-convex landscape from a cold start — this is
the direct fix for the failure mode diagnosed in §2.

### Stage A — Amortized regression network (the primary mechanism)

**Input**: the two sensor pressure time series `H1(t), H2(t)` (same synthetic data
`leakpinn/synth.py` already produces), resampled onto a **fixed number of samples in
dimensionless `tau` space** (`tau = a*t/L`, same nondimensionalization `general_pinn.py` already
uses) — so the network sees a consistently-scaled signal regardless of the pipe's absolute size,
mirroring the forward model's own design. Plus the same "known at inference time" conditioning
inputs identified in the previous turn: `L, D, a` (or its uncertainty range), `H_res`, `phi`,
design flow — everything except `xiL`/`kappa`, which is what the network is solving for.

**Output**: `(xiL, log CdA)` — and optionally a wave-speed correction, mirroring
`general_pinn.py`'s treatment of wave-speed uncertainty in `Params`.

**Architecture**: a small 1D CNN or GRU/LSTM over the two-channel time series (the precedent
papers in §3 use exactly this family), with the conditioning scalars concatenated in — same
"features → dense" pattern already used throughout `leakpinn/general_pinn.py`. Not yet an
architecture decision to lock in without a quick empirical comparison (§8).

**Training**: fully supervised, using `leakpinn/domain.py` + `leakpinn/moc.py` + `leakpinn/synth.py`
exactly as they exist today to generate thousands of `(sensor_trace, true_xiL, true_CdA)` pairs
across randomized pipes — the same generation machinery already built and validated for the
forward model, pointed at a new purpose. No PDE residual, no collocation points, no non-convex
per-case optimization — ordinary regression loss (MSE or a physically-scaled variant) against
known ground truth. This is the part directly grounded in the ASCE/engproc precedent in §3.

**Why this is expected to work where the joint-PINN approach didn't**: it's not solving a fresh
non-convex optimization problem per pipe at inference time — the hard work (learning what a
leak's signature looks like) happens once, during training, across thousands of labelled
examples, with an ordinary supervised gradient (not a weak PDE-residual gradient from a handful of
collocation points near a narrow feature). At inference time it's a single forward pass.

### Stage B — Physics-informed refinement (the "PINN" part, now used correctly)

Take Stage A's `(xiL, kappa)` estimate as a **warm start**, then run a short local optimization:
freeze `leakpinn/general_pinn.py`'s already-trained `GeneralNet` weights, treat `(xiL, kappa)` as
the only free variables, forward-simulate the predicted sensor trace through the network (fast,
GPU-batchable, no MOC re-solve needed), compute a data-misfit loss against the actual measured
trace, and take a handful of gradient steps (or a couple of L-BFGS iterations).

This reuses exactly the training-loop pattern already in `leakpinn/pinn.py`'s `Params`/`fit()` —
gradient descent on physical unknowns against a data-misfit loss — except now starting from a
point already close to the true answer instead of `xiL_init=0.5` blind, which is precisely what
§2's diagnosis says the original attempt needed. Bounded in scope (a handful of gradient steps,
not thousands), so a bad local landscape has much less room to derail it.

**Why not skip Stage B and just ship Stage A alone?** Stage A's accuracy is bounded by how densely
it saw similar scenarios during training; Stage B recovers precision the discrete regression
network can't reach — the same "coarse scan → refine" pattern `leakpinn/pinn.py`'s
`multistart_fit()` already uses, just with a *learned* good starting point instead of a handful of
blind ones.

### Where this runs (deployment target)

Per `DEPLOYMENT.md`'s own existing framing (the classical baselines are described as "running on a
bigger processor elsewhere, e.g. a gateway or PC," not the MCU) — **this plan assumes the same**:
Stage A/B run on a gateway or PC after a transient event is captured, not on the Renesas MCU. The
MCU's job stays what it already is: run the tiny forward `.tflite` model as a virtual sensor, or
just log the raw transient for offline analysis. I'm treating this as the default rather than
something to ask about, since it matches this project's own established deployment split — flag
if that's wrong and the localization model needs to run on-device too, which would change the
architecture/size budget in §5 considerably.

---

## 6. Step-by-step implementation plan

1. **Build the labelled dataset generator** (`leakpinn/inverse_data.py` or similar): reuse
   `domain.sample_scenario()` + `synth.make_dataset()` unchanged to produce N random
   `(pipe, leak, sensor_trace)` triples; resample each trace onto a fixed-length dimensionless-tau
   grid; save `(features, xiL, CdA)` tuples. Validate the generator by checking label ranges and
   a handful of traces by eye before training anything (same discipline as `domain.py`'s own
   validation in the previous session).
2. **Pick and justify the Stage A architecture** empirically: compare a small 1D-CNN vs. a
   GRU/LSTM vs. a plain MLP-on-flattened-trace on a modest dataset (a few thousand examples)
   before committing to the final scale — the precedent papers don't agree on one architecture,
   so this needs a real comparison, not a guess.
3. **Train Stage A** on the full generated dataset (likely tens of thousands of scenarios, more
   than the forward model needed since this is now the primary accuracy mechanism, not a
   secondary polish) with a proper train/val/held-out split — held-out scenarios never seen during
   architecture selection either, to avoid quietly overfitting the architecture choice.
4. **Validate Stage A alone** against MOC ground truth AND against the existing `moc_search`
   baseline on the same held-out scenarios, reusing `leakpinn/metrics.py`'s discipline (truth
   never fed to the model) — report x_L error in metres and CdA relative error, directly
   comparable to `results/03_baseline_comparison.json`'s existing numbers.
5. **Implement Stage B** as a short gradient-refinement loop around the frozen `GeneralNet`,
   warm-started from Stage A's output; validate the delta it adds (Stage A alone vs. Stage A+B) on
   the same held-out set, and check it doesn't make things worse on any case (a real risk if the
   forward model's own ~21% NRMSE misleads the refinement on a hard case).
6. **Stress-test on the known-hard case**: the "small leak, 1% of Q0" scenario that broke
   `moc_search` (−21 m error) — this is the most informative single check available, since it's a
   documented, reproducible failure of the existing best baseline.
7. **Full comparison report**: Stage A alone, Stage A+B, `moc_search`, `time_of_flight`, side by
   side on the same held-out scenarios — an honest table, not just "ours wins," matching this
   project's existing documentation style.
8. **Only after that**: decide whether/how to export anything to ONNX/TFLite. Given §5's
   deployment assumption (gateway/PC, not MCU), this may not need the same TFLM-op-list
   discipline as the forward model at all — worth confirming before spending effort on it.

## 7. Validation discipline (non-negotiable, matches the rest of this repo)

- MOC ground truth used to generate labels is never fed to Stage A or Stage B as a feature.
- Held-out scenarios for final reporting are drawn with a different seed than anything used during
  architecture selection or hyperparameter tuning, not just different from the training set.
- Every accuracy number gets reported next to the corresponding `moc_search`/`time_of_flight`
  number on the *same* scenarios, not a different test set with a more favorable claim.
- Honest reporting of what doesn't work, same as the "small leak fails" and "bigger general model
  wasn't clearly better" notes already in `README.md`.

## 8. Open questions / decisions that affect scope

- **Architecture for Stage A** — CNN vs. RNN vs. Transformer vs. plain MLP; needs the empirical
  comparison in step 2, not a guess up front.
- **Dataset size** — the ASCE precedent doesn't state how many training examples it used; this
  needs a convergence check (does accuracy plateau, or is it still improving at N examples?)
  rather than picking a number blind.
- **Wave-speed uncertainty** — fold `a` into Stage A's outputs (predict a correction, like
  `general_pinn.py`'s `rho`) or keep it a fixed conditioning input? Affects whether Stage A needs
  a third output head.
- **The paper you want me to compare against** — not yet reviewed (§4).
- **On-device requirement** — confirm the gateway/PC assumption in §5 before any export work.

---

**Next step, if this plan looks right**: confirm the open questions in §8 (or share the paper
first so it can inform them), then I'll start with step 1 (the dataset generator) since everything
downstream depends on it.
