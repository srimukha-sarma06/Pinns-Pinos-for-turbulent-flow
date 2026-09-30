# Embedded Pipeline: 2-Input PINN Pressure Field + 6-Output Anomaly Classifier

## 1. Purpose

This document describes the embedded inference pipeline built from the current `Pinns-Pinos-for-turbulent-flow` repository.

The system uses two separate TFLite models:

1. A **2-input single-pipe forward PINN** to generate a pressure/flow field over a normalized space-time grid.
2. A **206-input, 6-output anomaly classifier** to classify the event as `none`, `leak`, or `constriction`, and to estimate anomaly location and class-specific severity.

The intended displayed result is an **annotated pressure field**:

```text
Pressure sensors / event capture
            |
            +-----------------------------+
            |                             |
            v                             v
     206-feature builder          normalized grid (xi,tau)
            |                             |
            v                             v
   6-output classifier             2-input PINN TFLite
            |                             |
            |                             v
            |                      h(xi,tau), q(xi,tau)
            |                             |
            |                             v
            |                    pressure reconstruction
            |                             |
            +----------->  pressure grid P(xi,tau)
                          + anomaly overlay
                          |
                          v
                    final display map
```

**Important:** the displayed map is an anomaly-annotated pressure field. The 6-output classifier does not itself modify the pressure values produced by the 2-input PINN. If a physically anomaly-updated field is required, a forward model conditioned on the predicted anomaly must be used.

---

## 2. Models used

### 2.1 Single-pipe forward PINN

Source:

```text
leakpinn/pinn.py
```

Exported model:

```text
results/05_pinn_forward_fp32.onnx
results/05_tflite_model/<generated .tflite>
```

The deployed network is the small plain MLP export created by `scripts/05_export_onnx.py`:

```text
Linear(2 -> 32)
Tanh
Linear(32 -> 32)
Tanh
Linear(32 -> 32)
Tanh
Linear(32 -> 2)
```

It has 2,274 trainable parameters. The exported graph accepts `[xi, tau]` and returns `[h, q]`. `xi` and `tau` are the normalized coordinates used by the PINN; the model internally maps them to its network-normalized representation. The model output is **not absolute pressure**. It is the dimensionless perturbation pair `[h, q]`.

The repository states that the fixed single-pipe model is specialized to its training scenario rather than being a general pipe model.

### 2.2 Six-output anomaly classifier

Source:

```text
leakpinn/classify_net.py
```

Exported model:

```text
results/anomaly_classifier_fp32.tflite
```

Input:

```text
[1, 206] float32
```

Output:

```text
[1, 6] float32
```

Output layout:

```text
[0] p_none
[1] p_leak
[2] p_constriction
[3] xi_frac
[4] kappa_leak
[5] kappa_constriction
```

The first three values are a 3-class softmax. `xi_frac` is constrained to approximately `(0.03, 0.97)`. The two severity heads are positive. Only the severity head corresponding to the predicted anomaly class is meaningful; the other severity output is a don't-care value for that sample.

The classifier's exported wrapper contains the training-set feature mean/std normalization, so the MCU supplies the **raw 206-feature vector**.

---

## 3. Complete runtime pipeline

There are two computations after an event is captured.

### Branch A: pressure-field generation

The 2-input forward PINN is evaluated on a normalized grid:

\[
\xi \in [0,1]
\]

and

\[
\tau \in [0,\tau_{end}].
\]

At every grid point:

\[
[\xi,\tau]
\rightarrow
[h,q].
\]

The exported single-pipe network returns dimensionless `h` and `q`, so they must be reconstructed into physical head/flow using the same fixed scenario constants used by the export.

For the current fixed-pipe model:

\[
h=\frac{H-H_{ss}(\xi)}{H_s}
\]

with `Hs = 10 m`, hence

\[
H(\xi,\tau)=H_{ss}(\xi)+10h(\xi,\tau).
\]

For the fixed exported model, the steady head profile is:

\[
H_{ss}(\xi)=H_{2,0}
+\frac{H_{1,0}-H_{2,0}}{1-\xi_1}(1-\xi).
\]

Then pressure is:

\[
P(\xi,\tau)=\rho g H(\xi,\tau).
\]

For a water model, pressure can be converted to bar with:

\[
P_{bar}=\frac{\rho g H}{10^5}.
\]

The fixed model can also reconstruct physical flow `Q` from the dimensionless `q` using the fixed pipe/leak constants documented by `DEPLOYMENT.md`.

### Branch B: anomaly classification

The two pressure sensors provide synchronized pressure time series. Those traces are converted into the same 206-feature representation used during classifier training.

The classifier is then invoked **once per event**:

\[
X_{206}\rightarrow
[p_{none},p_{leak},p_{constriction},\xi,\kappa_L,\kappa_C].
\]

The anomaly class is:

```text
class = argmax(p_none, p_leak, p_constriction)
```

Interpretation:

```text
0 -> none
1 -> leak
2 -> constriction
```

---

## 4. 206-feature generation

The feature definition is implemented in:

```text
leakpinn/localize_data.py
```

and reused unchanged for the none/leak/constriction classifier through:

```text
leakpinn/classify_data.py
```

The vector is:

```text
0..99       100 resampled perturbation-head samples, sensor 1
100..199    100 resampled perturbation-head samples, sensor 2
200         log(phi)
201         H1_0 / 100
202         H2_0 / 100
203         log(B_nom)
204         log(qv0)
205         xi1
```

Thus:

\[
206=100+100+6.
\]

### 4.1 Pressure -> hydraulic head

For gauge pressure:

\[
H=\frac{P}{\rho g}+z.
\]

The exact density and elevation convention must match the training setup.

### 4.2 Steady baseline

Using the pre-transient samples:

\[
H_{1,0}=mean(H_{1,pre})
\]

\[
H_{2,0}=mean(H_{2,pre}).
\]

Then:

\[
\Delta H_1(t)=H_1(t)-H_{1,0}
\]

\[
\Delta H_2(t)=H_2(t)-H_{2,0}.
\]

### 4.3 Wave speed estimate

The repository's feature builder expects `a_est` and converts the dimensionless resampling grid back into physical time using:

\[
t_i=\frac{\tau_i L}{a_{est}}.
\]

The repository's baseline wave-speed estimator uses the relative arrival of the first pressure transient at the two sensors.

### 4.4 100-point resampling

The model uses:

\[
\tau_i=\frac{3.2i}{100},\quad i=0,\ldots,99
\]

then:

\[
t_i=\frac{\tau_iL}{a_{est}}.
\]

The original Python implementation uses linear interpolation (`np.interp`) of the baseline-subtracted traces.

The MCU implementation should therefore perform the equivalent linear interpolation directly on the uniformly sampled ADC data.

### 4.5 Six scalar features

The repository calculates:

\[
A=\frac{\pi D^2}{4}
\]

\[
B_{nom}=\frac{a_{est}}{gA}
\]

\[
f=f(Q_{design})
\]

\[
\phi=\frac{fLg}{2Da_{est}^2}
\]

\[
q_{v0}=B_{nom}Q_{design}
\]

\[
\xi_1=\frac{x_1}{L}.
\]

The six appended values are:

```text
log(phi)
H1_0 / 100
H2_0 / 100
log(B_nom)
log(qv0)
xi1
```

---

## 5. Classifier output handling

The classifier returns:

```text
[p_none, p_leak, p_constriction,
 xi_frac, kappa_leak, kappa_constriction]
```

### No anomaly

If:

```text
argmax(first three outputs) == 0
```

display:

```text
NONE
```

Ignore:

```text
xi_frac
kappa_leak
kappa_constriction
```

### Leak

If:

```text
argmax(first three outputs) == 1
```

use:

\[
\xi_{anomaly}=xi\_frac
\]

and:

\[
\kappa=\kappa_{leak}.
\]

Convert normalized location to metres:

\[
x_{anomaly}=\xi_{anomaly}L.
\]

The physical leak conductance can be recovered from the repository's leak parameter relationship:

\[
C_dA_{leak}=
\frac{\kappa_{leak}}
{B_{nom}\sqrt{2g}}.
\]

### Constriction

If:

```text
argmax(first three outputs) == 2
```

use:

\[
\xi_{anomaly}=xi\_frac
\]

and:

\[
\kappa=\kappa_{constriction}.
\]

Convert normalized location to metres:

\[
x_{anomaly}=\xi_{anomaly}L.
\]

For the constriction model, the effective throat quantity is analogously related to the dimensionless severity:

\[
C_dA_c=
\frac{\kappa_c}
{B_{nom}\sqrt{2g}}.
\]

Again, do not use the leak severity output for a constriction, and vice versa.

---

## 6. Final pressure-map visualization

The pressure grid is produced first from the 2-input PINN:

\[
P_{j,i}=P(\xi_i,\tau_j).
\]

The classifier result is then overlaid on the map.

For example, if:

```text
class       = leak
xi_frac     = 0.41
kappa_leak  = 0.018
```

and:

```text
L = 100 m
```

then:

```text
anomaly position = 0.41 * 100 = 41 m
```

The display can draw a vertical marker at `x = 41 m` or `xi = 0.41` and annotate the selected severity.

A conceptual screen is:

```text
                  POSITION ALONG PIPE
        0                                   1
        |-----------------------------------|
 t=0    |███████████████████████████████████|
        |████████████████▓▓▓▓███████████████|
        |██████████▓▓▓▓▓▓▓▓▓▓▓▓█████████████|
        |████▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓████████████|
        |███████████████████████████████████|
        |                |                  |
        |                |                  |
        |           anomaly                |
        |             x=41 m               |
        +-----------------------------------+
                       TIME

        LEAK
        Confidence: 0.92
        Severity: κ_leak = 0.018
```

The pressure values themselves are still the values generated by the 2-input forward PINN. The classifier supplies the annotation.

---

## 7. MCU execution order

### Event capture

```text
1. Trigger on the controlled valve transient.
2. Capture synchronized pressure samples from sensor 1 and sensor 2.
3. Keep a pre-transient section for H1_0/H2_0 estimation.
```

### Feature construction

```text
4. Convert pressure to hydraulic head.
5. Compute H1_0 and H2_0.
6. Subtract baselines.
7. Estimate wave speed a_est from the two traces.
8. Generate the 100-point dimensionless-time grid.
9. Linearly interpolate both traces.
10. Compute phi, B_nom, qv0, xi1.
11. Assemble the 206-element float32 vector.
```

### Classifier

```text
12. Run anomaly_classifier_fp32.tflite once.
13. Determine class from argmax(p_none, p_leak, p_constriction).
14. Use xi_frac only for leak/constriction.
15. Use kappa_leak only for leak.
16. Use kappa_constriction only for constriction.
```

### Pressure grid

```text
17. Generate a normalized grid of (xi,tau) points.
18. Run the 2-input forward TFLite model at each point.
19. Reconstruct H(xi,tau).
20. Convert H to pressure if required.
21. Store/render P_grid.
```

### Display

```text
22. Render P_grid as the base heatmap.
23. Overlay the anomaly marker only if class != NONE.
24. Display the relevant severity only for the selected anomaly class.
25. Display the class probabilities if desired.
```

---

## 8. File list from the repository

### Direct model implementation

```text
leakpinn/pinn.py
```

Single-pipe PINN architecture, normalization, hard output constraints, physical reconstruction helpers, and training problem definition.

```text
leakpinn/classify_net.py
```

The 206 -> 6 anomaly classifier, its class head, location head, two severity heads, and deployment normalization wrapper.

```text
leakpinn/localize_data.py
```

Authoritative definition of the 206-feature input vector, baseline calculation, resampling, `B_nom`, `phi`, `qv0`, and sensor-position feature.

```text
leakpinn/classify_data.py
```

Three-class (`none`/`leak`/`constriction`) training-data pipeline. Reuses the same 206-feature representation.

### Physics and training-data support

```text
leakpinn/physics.py
```

Pipe, valve, leak and fluid parameters; cross-sectional area; wave-speed model; Reynolds number; Darcy friction factor; valve relationships and leak/constriction hydraulic quantities.

```text
leakpinn/moc.py
```

Method-of-Characteristics solver used as the physics ground truth/reference for training-data generation and validation.

```text
leakpinn/synth.py
```

Synthetic sensor transient generation, sampling, noise and ADC quantisation.

```text
leakpinn/domain.py
```

Randomized pipe and anomaly scenario generation used by the generalized anomaly-classifier training data.

### Training/export scripts

```text
scripts/02_forward_pinn.py
```

Trains and validates the single-pipe forward PINN.

```text
scripts/05_export_onnx.py
```

Exports the small 2-input forward PINN to ONNX and checks the exported graph.

```text
scripts/16_train_classifier.py
```

Generates 206-feature classifier data, trains the 6-output anomaly classifier and evaluates held-out performance.

```text
scripts/17_export_classifier_onnx.py
```

Exports the classifier with raw-feature input and the six-output contract.

```text
scripts/18_convert_classifier_tflite.py
```

Converts the classifier ONNX model to fp32 TFLite and verifies the result.

### Deployment/reference documentation

```text
DEPLOYMENT.md
```

Exact single-pipe forward-model I/O contract, physical reconstruction formulas, fixed scenario constants and TFLM deployment notes.

```text
CONSTRICTION_DETECTION_PLAN.md
```

The none/leak/constriction design, 6-output classifier, constriction model semantics, validation and TFLite deployment notes.

```text
README.md
```

Repository-level description, model status, ranges, architecture and validation results.

### Model artifacts

```text
results/05_pinn_forward_fp32.onnx
results/05_tflite_model/<generated .tflite>
results/anomaly_classifier_fp32.tflite
```

The first pair corresponds to the single-pipe 2-input forward model. The classifier artifact is the 206 -> 6 anomaly model used by this pipeline.

---

## 9. Files not required at runtime on the MCU

The following are important for reproducing/training the models but are not needed by the board once the `.tflite` files are built:

```text
leakpinn/moc.py
leakpinn/synth.py
leakpinn/domain.py
scripts/01_validate_moc.py
scripts/02_forward_pinn.py
scripts/14_validate_constriction.py
scripts/15_validate_classify_localize.py
scripts/16_train_classifier.py
scripts/17_export_classifier_onnx.py
scripts/18_convert_classifier_tflite.py
```

At inference time the MCU only needs the exported TFLite models plus embedded equivalents of the feature/preprocessing arithmetic.

---

## 10. Important model compatibility constraint

The two models are not symmetric.

The single-pipe forward model is a **fixed-scenario specialist**. The repository explicitly states that the deployed 2-input model is tied to the fixed pipe/event used for its export and must be retrained/re-exported for a different pipe or leak. Its physical reconstruction constants include the fixed pipe length, sensor/reference heads, wave speed and leak parameters.

The anomaly classifier is instead trained over a distribution of none/leak/constriction cases. Therefore, if the classifier is retrained for a different pipe or a different Reynolds-number regime, the pressure-field model should be retrained/configured consistently with the same physical scenario assumptions.

Do not combine an arbitrary classifier trained on one scenario distribution with an unrelated fixed single-pipe pressure model and interpret the result as a fully self-consistent physical estimate.

---

## 11. Important interpretation of the final map

The final display is:

\[
\boxed{\text{PINN pressure field} + \text{anomaly annotation}}
\]

not:

\[
\boxed{\text{PINN pressure field re-solved using the classifier anomaly}}.
\]

For example, if the classifier says:

```text
LEAK
xi = 0.42
kappa_leak = ...
```

the current display pipeline places the leak marker at:

\[
x=0.42L
\]

and displays the leak severity. It does not recompute the pressure field with the estimated leak inserted into the forward PDE.

If the desired product behavior is an anomaly-conditioned, physically updated pressure field, the forward model must expose anomaly parameters as runtime inputs (the repository's generalized forward model is the model designed around this idea for leaks).

---

## 12. Validation and limitations

The repository reports that the fixed single-pipe forward PINN achieves approximately 0.4% flow error relative to its MOC reference case. The repository separately reports that the 206 -> 6 TFLite anomaly classifier has approximately 85.3% held-out classification accuracy and substantially larger localization error than the physics-search detector.

The physics-search none/leak/constriction detector remains the repository's accuracy reference; the TFLite classifier exists as a low-latency embedded surrogate.

The repository also notes that the actual models were not hardware-validated on the target board in the documented work, so MCU timing, memory and numerical behavior still need to be measured on the real hardware.

---

## 13. Minimal implementation contract

### Classifier TFLite

```text
Input : float32[1][206]
Output: float32[1][6]

Output:
0 p_none
1 p_leak
2 p_constriction
3 xi_frac
4 kappa_leak
5 kappa_constriction
```

### Single-pipe forward TFLite

```text
Input : float32[1][2]
        [xi, tau]

Output: float32[1][2]
        [h, q]
```

Then:

```text
H = Hss(xi) + Hs * h
P = rho * g * H
```

and the heatmap is:

```text
P_grid[time_index][space_index]
```

---

## 14. Recommended display resolution

For an MCU display, start with:

```text
Nx = 32
Nt = 16
```

which requires:

\[
32\times16=512
\]

forward-model invocations.

If inference time and memory allow it, increase to:

```text
Nx = 64
Nt = 24
```

for 1,536 points.

The normalized grid is:

```python
xi  = linspace(0, 1, Nx)
tau = linspace(0, tau_end, Nt, endpoint=False)
```

The physical axes, when required, are:

\[
x=\xi L
\]

and

\[
t=\frac{\tau L}{a_{nom}}.
\]

---

## 15. Summary

The complete embedded system is:

```text
                 TWO PRESSURE SENSORS
                         |
                         v
              206 FEATURE PREPROCESSOR
                         |
                         v
               +----------------------+
               | 6-output classifier  |
               +----------------------+
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
        NONE            LEAK       CONSTRICTION
                         |
                         | anomaly annotation
                         |
                         +-------------------+
                                             |
Normalized grid                           |
(xi,tau)                                   |
   |                                       |
   v                                       |
2-input forward PINN TFLite               |
   |                                       |
   v                                       |
[h,q]                                      |
   |                                       |
   v                                       |
H(xi,tau) -> P(xi,tau)                     |
   |                                       |
   +--------------------+------------------+
                        |
                        v
              ANNOTATED PRESSURE MAP
                        |
                        v
                     DISPLAY
```

The classifier determines **what anomaly to annotate, where to place it, and which severity value to display**. The 2-input PINN determines the **pressure/flow field background**. The two model outputs are deliberately kept separate and combined only at the presentation layer.
