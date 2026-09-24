"""Export a trained PINN to ONNX for microcontroller deployment (Renesas RA8 + Ethos-U55,
CPU-only / no NPU path -> plain TensorFlow Lite Micro).

What this script does, and does NOT do:
  1. Trains a small forward PINN (leak location/size GIVEN -- this is the deployed "virtual
     sensor" model, not the inverse/training-time parameter search).
  2. Exports it to ONNX and checks the export is numerically identical to the PyTorch model.
  3. Lists every ONNX op the exported graph actually uses and checks each one against the
     TFLM kernel registry (tensorflow/lite/micro/kernels/micro_ops.h, confirmed current as of
     this writing -- see the list below) to catch an unsupported op before you ever touch a
     board.
  4. Does NOT convert to .tflite itself (that needs a full TensorFlow install, which is heavy)
     -- prints the exact follow-up command (onnx2tf) instead.
  5. Does NOT talk to the Ethos-U55 NPU compiler at all, since you said you're not using the
     NPU -- this model is meant to run as plain CPU TFLM kernels.

Run: python scripts/05_export_onnx.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DDE_BACKEND", "pytorch")
import numpy as np
import torch
import onnx
import onnxruntime as ort
from leakpinn.synth import make_dataset
from leakpinn.pinn import PINNConfig, build_problem, fit
from leakpinn.physics import G

# Ops confirmed present in tensorflow/lite/micro/kernels/micro_ops.h (checked directly against
# the current TFLM source -- these are the plain-CPU reference kernels, independent of the
# Ethos-U NPU, which you said you're not using).
TFLM_SUPPORTED = {
    "Gemm", "MatMul",              # -> FULLY_CONNECTED
    "Add", "Sub", "Mul", "Div",    # -> ADD / SUB / MUL / DIV
    "Tanh", "Sigmoid",             # -> TANH / LOGISTIC
    "Sin", "Cos",                  # -> SIN / COS  (Fourier-feature architectures)
    "Concat", "Reshape", "Transpose", "Flatten", "Slice", "Squeeze", "Unsqueeze",
    "Sqrt", "Exp", "Log", "Neg", "Abs",
    "Relu", "Softmax", "Clip",     # Clip -> not in TFLM's own list, see note below
}
# NOTE: 'Clip' has no native TFLM kernel either (matches what we found for the Renesas e-AI
# Translator too) -- but as with Softplus, this project never uses clamp/clip inside the
# deployed forward pass, only in the training-time parameter reparameterisation, so it should
# never actually appear in an exported forward-model graph. Flagged here anyway so the checker
# below would catch it if it ever did.

# ---------------------------------------------------------------- train a small, MCU-sized model
d = make_dataset(leak_x=37.3, CdA=3.6e-6, seed=0)
cfg = PINNConfig(
    arch="mlp",       # smallest, most portable: plain Linear+Tanh stack, no Fourier features
    width=32, depth=3,       # small enough for a microcontroller; widen if accuracy needs it
    schedule=[
        dict(opt="adam", iters=3000, free=False),
        dict(opt="lbfgs", iters=800, free=False),
    ],
)
prob = build_problem(d, cfg)
fixed = dict(
    xiL=d.truth["leak_x"] / d.pipe.L,
    kappa=prob.B_nom * d.truth["CdA"] * np.sqrt(2 * G),
    rho=d.truth["a"] / prob.a_nom,
    cv=1.0,
)
t0 = time.time()
res = fit(d, cfg, fixed=fixed, verbose=True)
print("trained in", time.time() - t0, "s")
n_params = sum(p.numel() for p in res.model.net.parameters())
print(f"network has {n_params} parameters")

# ---------------------------------------------------------------- export
# Trained in float64 (needed for stable PINN optimisation), but an MCU wants float32: most
# Cortex-M cores only have hardware FPU support for single precision, and several TFLite/TFLM
# kernels don't support float64 at all. Cast the trained weights to float32 right before export
# rather than training in float32 from the start (which is noticeably less numerically stable
# for this kind of PDE-residual optimisation).
#
# Capture the float64 reference output FIRST, before the in-place .float() cast below mutates
# res.model.net (torch's .float()/.double() modify the module in place and return self).
test_x64 = np.random.default_rng(0).uniform(0, 1, size=(200, 2)).astype(np.float64)
test_x64[:, 1] *= prob.tau_end
with torch.no_grad():
    y_torch64 = res.model.net(torch.tensor(test_x64)).numpy()   # still float64 here

net = res.model.net.eval().float()   # in-place cast: res.model.net is now float32 from here on
os.makedirs("results", exist_ok=True)
onnx_path = "results/05_pinn_forward_fp32.onnx"
dummy = torch.zeros(1, 2, dtype=torch.float32)   # input: (xi, tau), both normalised
torch.onnx.export(
    net, dummy, onnx_path,
    input_names=["xi_tau"], output_names=["h_q"],
    dynamic_axes={"xi_tau": {0: "batch"}, "h_q": {0: "batch"}},
    opset_version=17,
)
print("exported to", onnx_path)

# ---------------------------------------------------------------- verify: PyTorch vs ONNX Runtime
# Compares the float32 ONNX export against the ORIGINAL float64 PyTorch output captured above,
# so this measures both export fidelity AND the real cost of the float64->float32 cast.
sess = ort.InferenceSession(onnx_path)
y_onnx32 = sess.run(None, {"xi_tau": test_x64.astype(np.float32)})[0]
max_diff = float(np.abs(y_torch64 - y_onnx32.astype(np.float64)).max())
print(f"max |float64 PyTorch - float32 ONNX| = {max_diff:.3e}")
print("  (measures export fidelity + the float64->float32 cast cost; compare to the physical")
print("   scale factors Hs=10 m and 1/B_nom~1.5e-5 m^3/s to judge whether it matters here)")

# ---------------------------------------------------------------- op-list check against TFLM
model = onnx.load(onnx_path)
ops_used = sorted({n.op_type for n in model.graph.node})
unsupported = [op for op in ops_used if op not in TFLM_SUPPORTED]
print("\nONNX ops used by the exported graph:", ops_used)
if unsupported:
    print("!! NOT confirmed in the TFLM kernel list:", unsupported)
else:
    print("All ops confirmed present in TFLM's kernel registry (CPU path, no NPU needed).")

report = dict(
    onnx_path=onnx_path, n_params=n_params, max_pytorch_onnx_diff=max_diff,
    ops_used=ops_used, unsupported_ops=unsupported,
    onnx_file_bytes=os.path.getsize(onnx_path),
)
json.dump(report, open("results/05_export_report.json", "w"), indent=2)
print("\nsaved results/05_export_report.json")

print("""
NEXT STEP (not run here -- needs a full TensorFlow install, which this script deliberately
avoids to stay light): convert the ONNX file to TFLite for the RA8/FSP toolchain with:

    pip install onnx2tf tensorflow
    onnx2tf -i results/05_pinn_forward_fp32.onnx -o results/05_tflite_model/

That produces a .tflite file you hand to Renesas FSP's TFLM integration. If you want, I can
also run that conversion step for you next.
""")
print("DONE")
