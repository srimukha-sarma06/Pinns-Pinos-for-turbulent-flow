# Constriction/blockage detection — implementation plan

**Status: implemented and validated.** Every step below was carried out; see the end of this
file for what was actually built, the bugs found and fixed along the way, and the honest final
accuracy numbers (`results/14_constriction_validation.png`,
`results/15_classify_localize_validation.json`).

Extends the existing leak-detection pipeline to also detect and localize **partial blockages**
(constrictions: scaling, sediment, corrosion, a partially-closed valve stuck in place) — a
genuinely different failure mode from a leak, not a variant of it. Builds directly on the
validated MOC solver, reusing its proven physics-search methodology (not the amortized-ML inverse,
which this project already showed loses to physics search).

Target: fully edge, **no cloud component anywhere**, deployed on a **Renesas RA8P1**
(1 GHz Cortex-M85 + 250 MHz Cortex-M33 + Ethos-U55 NPU, 256 GOPS, 2 MB SRAM — confirmed from
Renesas's own product page and CNX Software's coverage, not assumed). This chip is far more
capable than a typical MCU, which changes an earlier assumption in this project: the physics
search (previously assumed to need a gateway/PC) is now plausibly deployable on-device too — see
step 7.

## 1. Physics: what a constriction actually is, and why it's different from a leak

A leak is a **mass sink**: `Q` is discontinuous across it (`Qu - Qd = leak flow`), `H` is
continuous. A constriction is the opposite: **no mass is lost** (`Q` is continuous across it), but
there's a local head loss from the area reduction (`Hu - Hd = loss(Q)`), so `H` is discontinuous.
Modeled as an orifice-in-pipe, matching the style of `Leak`'s `CL = CdA*sqrt(2g)`:

```
Q = Cd * A_c * sqrt(2g * (Hu - Hd))          (orifice discharge law, Q > 0 assumed, matching
                                               the rest of this codebase's forward-flow convention)
=>  Hu - Hd = Rc * Q * |Q|,   Rc = 1 / (2g * (Cd*A_c)^2)
```

`CdA_c` (the effective open throat area) plays exactly the role `Leak.CdA` already plays — a
`Constriction(x, CdA_c)` dataclass mirrors `Leak(x, CdA)` structurally.

### MOC junction condition (derived, not guessed)

At the constriction node, the two characteristics give (same `CP`/`CM` as everywhere else in
`moc.py`):
```
Hu = CP - B*Q          (C+ compatibility, upstream face)
Hd = CM + B*Q          (C- compatibility, downstream face)
Hu - Hd = Rc*Q*|Q|
```
Substituting and solving the resulting quadratic for `Q > 0`:
```
Rc*Q^2 + 2*B*Q - (CP - CM) = 0
Q = [-B + sqrt(B^2 + Rc*(CP-CM))] / Rc
```
then `Hu = CP - B*Q`, `Hd = CM + B*Q`. This is the exact mirror image of the leak's existing
solve (`Hn[iL] = s*s` from its own quadratic) — where the leak breaks `Q`-continuity and needs
`Qu`/`Qd` tracked separately (already implemented), the constriction breaks `H`-continuity and
needs `Hu`/`Hd` tracked separately (not yet implemented — this is the actual code change needed
in `moc.py`'s state arrays, symmetric to what's already there for `Qu`/`Qd`).

### Closed-form reflection coefficient (derived, for validation)

Linearizing around steady state (`Q0`, `Gc = Rc*Q0`), with no incident wave from downstream
(mirrors how the leak's own `R = -GB/(2+GB)` check in `scripts/01_validate_moc.py` is derived and
tested):
```
R_constriction = Gc / (B + Gc)        (positive, 0 < R < 1)
```
Compare to the leak's `R = -GB/(2+GB)` (negative). **The sign difference is the key discriminating
signature** — a constriction causes a positive (compression-type) reflection, a leak a negative
(rarefaction-type) one, matching what the published blockage-detection literature reports as the
primary way these two failure modes are told apart from a pressure trace alone. Sanity checks:
`Gc -> infinity` (full blockage) gives `R -> 1`, matching a dead-end reflection; `Gc -> 0` (no
constriction) gives `R -> 0`, matching a uniform pipe. Validate this numerically the same way
`scripts/01_validate_moc.py` validates the leak's reflection coefficient: simulate with/without
the constriction, measure the actual echo, compare to the closed form (expect the same ballpark
tolerance the leak check uses, ~25%, since both are local small-signal approximations).

## 2. Step-by-step implementation

1. **`leakpinn/physics.py`**: add `Constriction` dataclass (`x`, `CdA_c`), mirroring `Leak`.
2. **`leakpinn/moc.py`**: extend `MOC` to accept an optional `Constriction` alongside (or instead
   of) a `Leak`. Requires new `Hd`/`Hu` state arrays (mirroring the existing `Qd`/`Qu` pattern) so
   neighbors read the correct face of the constriction node — regular nodes and leak nodes
   collapse `Hd == Hu` (no head discontinuity), constriction nodes collapse `Qd == Qu` (no flow
   discontinuity), so this is a genuine generalization of the existing state, not a special case
   bolted on.
3. **`scripts/14_validate_constriction.py`**: same rigor as `scripts/01_validate_moc.py` --
   steady-state mass/energy balance, grid convergence, no-cavitation, and the reflection-sign/
   magnitude check against the closed form above. Nothing downstream gets built or trusted until
   this passes, same discipline as the rest of this project.
4. **`leakpinn/domain.py`**: extend scenario sampling to draw one of three classes per scenario --
   `none` / `leak` / `constriction` -- from the same pipe-parameter ranges already validated
   (20-300 m, DN25-DN200, 300-1400 m/s, etc.), plus a constriction-severity range (throat area as
   a fraction of full bore, e.g. 10-70% open).
5. **`leakpinn/baselines.py`**: add `constriction_search()`, mirroring `moc_search()`'s grid+LM
   structure but searching `(x_c, CdA_c)` against the constriction junction condition. Add a
   `classify_and_localize()` that runs both `moc_search` and `constriction_search` (plus a
   no-anomaly baseline cost) and reports whichever best explains the measured trace, with a class
   label -- this is a model-selection problem, not just two independent parameter searches.
6. **Validation**: synthetic test set spanning all three classes across the pipe range, scored on
   (a) classification accuracy (none/leak/constriction), (b) localization error given the correct
   class, mirroring `results/03_baseline_comparison.json`'s and
   `results/10_localization_comparison.json`'s reporting style -- an honest table, not a single
   headline number.
7. **Edge/compute feasibility check**: estimate on-device cost on the RA8P1's Cortex-M85. The
   search's actual FLOP count is small (a few hundred grid nodes x a few hundred timesteps x O(10)
   flops/step, times however many candidate points the grid+LM search touches) -- estimate total
   FLOPs for the realized search budget and compare against the core's throughput, rather than
   assuming either way. No cross-compilation toolchain or physical board is available in this
   environment, so this is an analytical estimate (with the measured Python-side simulation cost
   as a cross-check), not a hardware-verified benchmark -- stated as such, not overclaimed.
8. **Full edge audit**: confirm nothing in the resulting design makes a network call anywhere --
   data capture, the physics search, and reporting are all local computation. This should hold by
   construction (everything built in this project so far is local NumPy/PyTorch, no network
   dependency), but stated and checked explicitly rather than assumed.

## 3. What's deliberately NOT in scope here

- No amortized ML classifier/regressor for this problem -- this project already has direct
  evidence (the leak-localization work) that physics-based search beats amortized regression at
  this data scale, so the default here is the same search-based approach, not a rebuilt NN.
- No literal on-device C port or hardware test -- no RA8P1 board or Renesas toolchain is available
  in this environment. Step 7 is an analytical feasibility estimate, clearly labeled as such.
- No simultaneous leak+constriction case (both at once) -- out of scope for a first version;
  `classify_and_localize()` picks the single best-fitting hypothesis among none/leak/constriction.

---

## What was actually built (post-implementation notes)

### Physics validation (`scripts/14_validate_constriction.py`) -- all checks pass

The initial reflection-coefficient check FAILED (numerically -22% vs. a theoretical +1.66%, wrong
sign and wrong magnitude) despite the steady-state mass-balance and head-loss-formula checks
passing exactly. Root-caused by direct diagnosis rather than assumed: the two simulations being
compared (with/without the constriction) have different PRE-TRANSIENT operating points (the
constriction changes the steady flow the reservoir head can drive), so their valve-closure
dynamics diverge for reasons unrelated to wave reflection, and that divergence swamped a
downstream-sensor "echo window" measurement -- confirmed by re-deriving the reflection formula
from scratch (twice), verifying the exact quadratic junction solve against its own linearization
with hand-injected perturbations (matched to 4 decimal places), and finally validating the
reflection AT THE CONSTRICTION NODE ITSELF (reconstructing CP/CM from its neighbours' stored
transient history) rather than via a downstream sensor -- which matched theory to 0.07% relative
error. The confounded downstream-echo approach was replaced with this node-level method in the
validation script; all 12 checks now pass, including the sign discriminator (constriction: +1.67%,
positive; leak: -4.07%, negative, from `scripts/01_validate_moc.py`).

### Detector accuracy (`scripts/15_validate_classify_localize.py`, N=24 held-out scenarios)

Final numbers (`results/15_classify_localize_validation.json`): **95.8% classification accuracy**
(23/24). Confusion matrix -- leak and constriction are **never** confused with each other (8/8 and
12/12 correct respectively); the one error is a genuine no-anomaly case misclassified as a leak
(an expected false-positive rate at the chosen significance level, not a systematic issue).
Localization given the correct class: **median 0.76 m**, but **MAE 31.3 m** -- a wide gap driven
almost entirely by constriction cases specifically (leak localization is essentially exact on
every case, ~0-1 m; roughly half of constriction cases have errors of 40-160 m). This has a real,
structural explanation, not just "noise happens": `scripts/01_validate_moc.py` measures a leak's
reflection coefficient around -4% for a modest leak, while `scripts/14_validate_constriction.py`
measures a comparably-modest constriction's at only +1.7% -- constrictions in the sampled severity
range genuinely produce a weaker signature against the same sensor noise floor, so the search's
cost landscape is more easily pulled to a wrong-but-noise-favoured optimum. Confirmed directly on
one failing case: the wrong location's cost (10.52) was genuinely LOWER than the true parameters'
own cost (10.73) for that specific noise draw -- not a search or grid bug, a real identifiability
limit at this SNR -- matches the same honest-limitation pattern as the original single-pipe work's
documented "small leak fails" case (`README.md`): reported as such, not hidden or explained away.
Three real bugs were found and fixed while building the classifier along the way (distinct from
the SNR limitation above, which isn't a bug), all confirmed empirically before and after:
1. `no_anomaly_cost` didn't refine wave speed the way `moc_search`/`constriction_search` do during
   their LM step, unfairly inflating the "none" hypothesis's cost whenever the initial
   `estimate_wave_speed()` guess had any error -- this alone caused a genuinely anomaly-free case
   to misclassify as a leak.
2. Naive `argmin` over the three hypothesis costs is wrong even after fixing (1): leak/constriction
   each have more free parameters than "none", so they can fit sensor noise at least as well even
   when nothing is actually wrong. Fixed with a chi-squared significance threshold (extra
   parameters must reduce cost by more than they'd be expected to from noise alone at a chosen
   significance level) before accepting either over "none".
3. `moc_search`'s and `constriction_search`'s default `CdA` search grids were both too coarse in a
   way that mattered specifically for constrictions: the cost surface in `CdA_c` is sharply peaked
   (confirmed: cost at the exact true parameters can be an order of magnitude better than at the
   nearest points on a 4-point grid), so a sparse grid can start the local refinement too far away
   to find its way to the true optimum. Densified to 8 points; this fixed some but not all of the
   worst localization outliers.
### Edge-compute feasibility on the RA8P1 (analytical estimate, not hardware-measured)

No RA8P1 board or Renesas toolchain is available in this environment, so this is a projection from
measured numbers, not a benchmark. One MOC simulation (N=200 grid, ~830 timesteps, ~0.3 s
transient) was measured at **79 ms in Python/NumPy** -- ~0.085 GFLOP/s effective throughput, far
below what a 1 GHz Cortex-M85 with an FPU should sustain for this kind of straightforward
vectorizable loop, because the Python/NumPy figure is dominated by interpreter and per-call
dispatch overhead on small arrays, not actual floating-point work. One full `classify_and_localize`
call (no-anomaly refinement + leak search + constriction search, each with grid + L-BFGS-style
refinement) needs on the order of **900 such simulations** (directly counted from the grid sizes
and refinement iterations, cross-checked against the ~70-100 s per-case wall time actually
measured in `scripts/15`). A compiled C port removing that interpreter overhead alone -- even
without hand-optimizing for the Cortex-M85's Helium (MVE) SIMD extension -- would plausibly reach
1-2+ GFLOP/s, putting one full three-hypothesis `classify_and_localize` call at roughly
**3-7 seconds on-device**, or faster still with MVE-vectorized inner loops. For an event-triggered
diagnostic test (run once after a transient is captured, not a continuous real-time constraint),
this is comfortably feasible entirely on the RA8P1 -- supporting the "no gateway, no cloud, fully
on-chip" architecture this feature was built for. This estimate should be confirmed on real
hardware before being treated as a guarantee.

### TFLite surrogate model (`results/anomaly_classifier_fp32.tflite`)

The physics search above has no weights to export -- it's a ~900-simulation grid+LM optimisation
loop, not a trainable model. To get an actual `.tflite` artifact for immediate on-device inference
(sub-millisecond, one forward pass, no C port of the MOC solver needed), a small amortized-
regression MLP was trained as a **separate surrogate**, graded against ground truth, not assumed
equivalent to the physics search:

- **Code**: `leakpinn/classify_data.py` (dataset: reuses the same `features_from_dataset` 206-dim
  feature vector as the earlier leak-only localizer, now sampled over `sample_anomaly_scenario`'s
  none/leak/constriction distribution) + `leakpinn/classify_net.py` (`ClassifierNet`: shared tanh
  trunk, 96 wide x 3 deep, 39,078 parameters -- a wider/deeper trial, 160x4, *overfit* and scored
  worse on held-out data, so the smaller net was kept) + `scripts/16_train_classifier.py` (trains
  on 20,000 examples, seeds 0-19999) + `scripts/17_export_classifier_onnx.py` +
  `scripts/18_convert_classifier_tflite.py`.
- **Held-out accuracy (300 fresh cases, seed 950000, disjoint from training and from the physics
  search's own 24-case validation set): 85.3%**, clearly below the physics search's 95.8%. The gap
  is concentrated exactly where the physics search's chi-squared significance test earns its keep:
  a single forward pass has no equivalent of "is this improvement over the null hypothesis bigger
  than noise alone would produce," so faint constrictions near the noise floor get called "none"
  (16/111 constriction cases) or vice versa (12/93 none cases) more often than the physics search's
  discriminator misses. Leak vs. constriction cross-confusion stays low (3 and 2 cases) -- the sign-
  flip discriminator the net implicitly learns survives amortization even though the none/anomaly
  boundary doesn't as cleanly.
- **Localization** (only scored on correctly-classified anomaly cases, same convention as the
  physics-search report): median 13.6 m, MAE 23.4 m -- worse than the physics search's median
  0.76 m, expected since there's no per-case local refinement step, just one direct regression.
- **Model**: `results/anomaly_classifier_fp32.tflite`, 162,248 bytes (~159 KiB), fp32. Ops used:
  `FULLY_CONNECTED, TANH, SIGMOID, SOFTMAX, EXP, ADD, SUB, MUL, CONCATENATION` -- all confirmed
  present in the actual compiled `.tflite`'s op list against TFLM's kernel registry (same
  discipline as the earlier leak-only localizer's `results/12_localizer_fp32.tflite`). ONNX-vs-
  TFLite max diff 6.9e-5.
- **I/O contract**: identical input feature vector to `results/12_localizer_fp32.tflite` (see the
  "What was actually built" section above / `leakpinn/localize_data.py`'s 206-column layout) --
  tensor name `sensor_features`, shape `[1, 206]`, float32, raw (un-normalized; normalization is
  baked into the graph). Output tensor `cls_xi_kappa`, shape `[1, 6]`:
  `[p_none, p_leak, p_constriction, xi_frac, kappa_leak, kappa_constriction]`. Take
  `argmax(p_none, p_leak, p_constriction)` for the predicted class; `xi_frac * L` gives the
  predicted position in meters (only meaningful if the predicted class isn't "none"); read
  `kappa_leak` if predicted leak or `kappa_constriction` if predicted constriction (the other is a
  don't-care value from an untrained branch of the output for that row).
- **Recommended use**: this surrogate trades accuracy for latency/footprint. Given the RA8P1 is
  fast enough for the full physics search in 3-7 s (an event-triggered, not real-time, budget), the
  physics search should remain the primary/reference detector; the TFLite model is for scenarios
  that specifically need a sub-second or continuous-polling response, with the accuracy gap above
  disclosed, not hidden.
