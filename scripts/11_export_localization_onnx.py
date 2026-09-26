"""Export Stage A of the leak-localization model (leakpinn/localize.py's LocalizerNet, trained by
scripts/10_train_localization.py) to ONNX, and check every op against the TFLM kernel registry --
same discipline as scripts/08_export_general_onnx.py. Stage B (the physics-informed refinement) is
NOT exported: it's a short gradient loop around the already-exported forward model
(results/09_general_forward_fp32.tflite), meant to run on a gateway/PC, not a set of on-device
layers -- see INVERSE_LOCALIZATION_PLAN.md's deployment-target note.

Run: DDE_BACKEND=pytorch python scripts/11_export_localization_onnx.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DDE_BACKEND", "pytorch")
# Same CUDA/torch.onnx.export segfault workaround as scripts/05 and scripts/08 (see their
# comments) -- disable CUDA for this whole process before torch is even imported.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
import onnx
import onnxruntime as ort

from leakpinn.localize import LocalizerNet, DeployLocalizer, DTYPE
from leakpinn.localize_data import FEATURE_DIM, generate_dataset

TFLM_SUPPORTED = {
    "Gemm", "MatMul", "Add", "Sub", "Mul", "Div", "Tanh", "Sigmoid", "Sin", "Cos",
    "Concat", "Reshape", "Transpose", "Flatten", "Slice", "Squeeze", "Unsqueeze",
    "Sqrt", "Exp", "Log", "Neg", "Abs", "Relu", "Softmax", "Clip", "Constant",
}

CKPT = "results/10_localizer.pt"
if not os.path.exists(CKPT):
    raise SystemExit(f"{CKPT} not found -- run scripts/10_train_localization.py first")

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
net = LocalizerNet(FEATURE_DIM, width=64, depth=3)
net.load_state_dict(ckpt["state_dict"])
net.eval()
deploy = DeployLocalizer(net, ckpt["x_mean"], ckpt["x_std"]).eval()
n_params = sum(p.numel() for p in net.parameters())
print(f"loaded {CKPT}: network has {n_params} parameters")

# ---------------------------------------------------------------- test batch (raw, un-normalized features)
X_test, _, _, _ = generate_dataset(50, seed0=800000, n_workers=8)
test_x64 = X_test.astype(np.float64)

with torch.no_grad():
    y_torch64 = deploy(torch.tensor(test_x64)).numpy()

# ---------------------------------------------------------------- cast to float32 + export
deploy_f32 = DeployLocalizer(net, ckpt["x_mean"], ckpt["x_std"]).eval().float()
os.makedirs("results", exist_ok=True)
onnx_path = "results/11_localizer_fp32.onnx"
dummy = torch.zeros(1, FEATURE_DIM, dtype=torch.float32)
torch.onnx.export(
    deploy_f32, dummy, onnx_path,
    input_names=["sensor_features"], output_names=["xiL_kappa"],
    dynamic_axes={"sensor_features": {0: "batch"}, "xiL_kappa": {0: "batch"}},
    opset_version=17,
)
print("exported to", onnx_path)

# ---------------------------------------------------------------- verify: PyTorch vs ONNX Runtime
sess = ort.InferenceSession(onnx_path)
y_onnx32 = sess.run(None, {"sensor_features": test_x64.astype(np.float32)})[0]
max_diff = float(np.abs(y_torch64 - y_onnx32.astype(np.float64)).max())
print(f"max |float64 PyTorch - float32 ONNX| = {max_diff:.3e}  (xiL fraction, kappa dimensionless)")

# ---------------------------------------------------------------- op-list check against TFLM
model = onnx.load(onnx_path)
ops_used = sorted({n.op_type for n in model.graph.node})
unsupported = [op for op in ops_used if op not in TFLM_SUPPORTED]
print("\nONNX ops used by the exported graph:", ops_used)
if unsupported:
    print("!! NOT confirmed in the TFLM kernel list:", unsupported)
else:
    print("All ops confirmed present in TFLM's kernel registry (CPU path, no NPU needed).")
assert "Erf" not in ops_used and "Clip" not in ops_used

report = dict(onnx_path=onnx_path, n_params=n_params, max_pytorch_onnx_diff=max_diff,
             ops_used=ops_used, unsupported_ops=unsupported,
             onnx_file_bytes=os.path.getsize(onnx_path),
             input_dim=FEATURE_DIM, output_columns=["xiL", "kappa"])
json.dump(report, open("results/11_export_report.json", "w"), indent=2)
print("\nsaved results/11_export_report.json")
print("DONE")
