"""Export the GENERALIZED forward PINN (leakpinn/general_pinn.py, trained by
scripts/07_train_general_forward.py) to ONNX for microcontroller deployment (Renesas
RA8 + Ethos-U55, CPU-only / no NPU -> plain TensorFlow Lite Micro).

Unlike scripts/05_export_onnx.py (which exports the raw single-pipe network and leaves
the steady-profile reconstruction to hand-written, per-deployment C++ using baked-in
constants -- see DEPLOYMENT.md), this exports `leakpinn.general_pinn.DeployNet`, which
bundles the trained network AND the steady-profile reconstruction into ONE graph. That
is necessary here (leak position/size and pipe parameters are now runtime INPUTS, not
constants known at export time, so there is no longer a single fixed formula to hand-
write per pipe) and is checked against the same TFLM op registry as 05_export_onnx.py --
in particular this replaces `erf()` (used by leakpinn/pinn.py's steady-profile math,
fine there only because it ran on the host against a FIXED leak position, never inside
a graph) with a Tanh-based smooth step, since `Erf` has no native TFLM kernel, and
avoids `torch.clamp` (no native `Clip` kernel either) via a Relu-based floor -- see
leakpinn/general_pinn.py's module docstring for the full reasoning.

Run: DDE_BACKEND=pytorch python scripts/08_export_general_onnx.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DDE_BACKEND", "pytorch")
# torch.onnx.export segfaults in this environment when a CUDA context is active (a torch/driver
# interaction bug, reproduced independently of this model) -- the export itself never needs a GPU
# for a network this small, so disable CUDA for this whole process before torch is even imported.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
import onnx
import onnxruntime as ort

from leakpinn.general_pinn import GeneralConfig, GeneralNet, DeployNet, scenario_consts
from leakpinn.domain import sample_scenario

# Same TFLM kernel-registry check as scripts/05_export_onnx.py (see that file for how this
# list was derived from tensorflow/lite/micro/kernels/micro_ops.h).
TFLM_SUPPORTED = {
    "Gemm", "MatMul", "Add", "Sub", "Mul", "Div", "Tanh", "Sigmoid", "Sin", "Cos",
    "Concat", "Reshape", "Transpose", "Flatten", "Slice", "Squeeze", "Unsqueeze",
    "Sqrt", "Exp", "Log", "Neg", "Abs", "Relu", "Softmax", "Clip",
    "Constant",   # not a real runtime kernel: ONNX's way of embedding a fixed tensor (e.g. the
                  # Fourier-feature matrix, small scalar constants) into the graph. onnx2tf folds
                  # these into the .tflite's own tensor buffers -- verified below (scripts/09) that
                  # no "CONST" op survives as something the TFLM interpreter has to dispatch.
}

CKPT = "results/07_general_pinn.pt"
if not os.path.exists(CKPT):
    raise SystemExit(f"{CKPT} not found -- run scripts/07_train_general_forward.py first")

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = GeneralConfig(**ckpt["cfg"])
net = GeneralNet(cfg.width, cfg.depth, cfg.sigma_xi, cfg.sigma_tau, cfg.n_feat, cfg.seed)
net.load_state_dict(ckpt["state_dict"])
net.eval()
deploy = DeployNet(net, Hs=cfg.Hs).eval()
n_params = sum(p.numel() for p in net.parameters())
print(f"loaded {CKPT}: network has {n_params} parameters")

# ---------------------------------------------------------------- build a representative test batch
# 10-column input contract (see leakpinn.general_pinn.DeployNet docstring / DEPLOYMENT.md):
#   0 xi  1 tau  2 xiL  3 kappa  4 phi  5 H1n  6 H2n  7 xi1  8 B_nom  9 qv0
rng = np.random.default_rng(42)
rows = []
for _ in range(50):
    sc = sample_scenario(rng)
    c = scenario_consts(sc)
    xi = rng.uniform(0, 1)
    tau = rng.uniform(0, 3.0)
    rows.append([xi, tau, sc.xiL, c["kappa"], c["phi"], c["H1_0"] / 100.0, c["H2_0"] / 100.0,
                c["xi1"], c["B_nom"], c["qv0"]])
test_x64 = np.array(rows, dtype=np.float64)

with torch.no_grad():
    y_torch64 = deploy(torch.tensor(test_x64)).cpu().numpy()   # still float64 here

# ---------------------------------------------------------------- cast to float32 + export
# Same reasoning as scripts/05_export_onnx.py: trained/verified in float64, cast to float32
# right before export (most Cortex-M cores only have hardware FPU for single precision).
deploy_f32 = DeployNet(net, Hs=cfg.Hs).eval().float().cpu()   # in-place cast of the underlying net too
os.makedirs("results", exist_ok=True)
onnx_path = "results/08_general_forward_fp32.onnx"
dummy = torch.zeros(1, 10, dtype=torch.float32, device="cpu")
torch.onnx.export(
    deploy_f32, dummy, onnx_path,
    input_names=["scenario_xy"], output_names=["H_Q"],
    dynamic_axes={"scenario_xy": {0: "batch"}, "H_Q": {0: "batch"}},
    opset_version=17,
)
print("exported to", onnx_path)

# ---------------------------------------------------------------- verify: PyTorch vs ONNX Runtime
sess = ort.InferenceSession(onnx_path)
y_onnx32 = sess.run(None, {"scenario_xy": test_x64.astype(np.float32)})[0]
max_diff = float(np.abs(y_torch64 - y_onnx32.astype(np.float64)).max())
print(f"max |float64 PyTorch - float32 ONNX| = {max_diff:.3e}  (H in metres, Q in m^3/s)")

# ---------------------------------------------------------------- op-list check against TFLM
model = onnx.load(onnx_path)
ops_used = sorted({n.op_type for n in model.graph.node})
unsupported = [op for op in ops_used if op not in TFLM_SUPPORTED]
print("\nONNX ops used by the exported graph:", ops_used)
if unsupported:
    print("!! NOT confirmed in the TFLM kernel list:", unsupported)
else:
    print("All ops confirmed present in TFLM's kernel registry (CPU path, no NPU needed).")
assert "Erf" not in ops_used, "erf leaked into the exported graph -- no native TFLM kernel"
assert "Clip" not in ops_used, "clamp leaked into the exported graph -- no native TFLM kernel"

report = dict(
    onnx_path=onnx_path, n_params=n_params, max_pytorch_onnx_diff=max_diff,
    ops_used=ops_used, unsupported_ops=unsupported,
    onnx_file_bytes=os.path.getsize(onnx_path),
    input_columns=["xi", "tau", "xiL", "kappa", "phi", "H1n", "H2n", "xi1", "B_nom", "qv0"],
    output_columns=["H_m", "Q_m3s"],
    param_ranges=cfg.__dict__,
)
json.dump(report, open("results/08_export_report.json", "w"), indent=2)
print("\nsaved results/08_export_report.json")
print("DONE")
