# Constriction/blockage detection — implementation plan

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
