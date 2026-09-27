"""Export the trained none/leak/constriction classifier (scripts/16_train_classifier.py) to ONNX,
checking every op against the TFLM kernel registry -- same discipline as scripts/08 and
scripts/11.

Run: python scripts/17_export_classifier_onnx.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # same torch.onnx.export CUDA-context workaround as scripts/05/08/11
import numpy as np
import torch
import onnx
import onnxruntime as ort

from leakpinn.classify_net import ClassifierNet, DeployClassifier, DTYPE
from leakpinn.localize_data import FEATURE_DIM
from leakpinn.classify_data import generate_dataset

TFLM_SUPPORTED = {
    "Gemm", "MatMul", "Add", "Sub", "Mul", "Div", "Tanh", "Sigmoid", "Sin", "Cos",
    "Concat", "Reshape", "Transpose", "Flatten", "Slice", "Squeeze", "Unsqueeze",
    "Sqrt", "Exp", "Log", "Neg", "Abs", "Relu", "Softmax", "Clip", "Constant",
}

CKPT = "results/16_classifier.pt"
if not os.path.exists(CKPT):
    raise SystemExit(f"{CKPT} not found -- run scripts/16_train_classifier.py first")

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
net = ClassifierNet(FEATURE_DIM, width=96, depth=3)
net.load_state_dict(ckpt["state_dict"])
net.eval()
deploy = DeployClassifier(net, ckpt["x_mean"], ckpt["x_std"]).eval()
n_params = sum(p.numel() for p in net.parameters())
print(f"loaded {CKPT}: network has {n_params} parameters")

# ---------------------------------------------------------------- test batch (raw, un-normalized features)
X_test, label_test, xi_test, sev_test, _ = generate_dataset(50, seed0=960000, n_workers=8)
test_x64 = X_test.astype(np.float64)

with torch.no_grad():
    y_torch64 = deploy(torch.tensor(test_x64)).numpy()

# ---------------------------------------------------------------- cast to float32 + export
deploy_f32 = DeployClassifier(net, ckpt["x_mean"], ckpt["x_std"]).eval().float()
os.makedirs("results", exist_ok=True)
onnx_path = "results/17_classifier_fp32.onnx"
dummy = torch.zeros(1, FEATURE_DIM, dtype=torch.float32)
torch.onnx.export(
    deploy_f32, dummy, onnx_path,
    input_names=["sensor_features"], output_names=["cls_xi_kappa"],
    dynamic_axes={"sensor_features": {0: "batch"}, "cls_xi_kappa": {0: "batch"}},
    opset_version=17,
)
print("exported to", onnx_path)

# ---------------------------------------------------------------- verify: PyTorch vs ONNX Runtime
sess = ort.InferenceSession(onnx_path)
y_onnx32 = sess.run(None, {"sensor_features": test_x64.astype(np.float32)})[0]
max_diff = float(np.abs(y_torch64 - y_onnx32.astype(np.float64)).max())
print(f"max |float64 PyTorch - float32 ONNX| = {max_diff:.3e}")

pred_label = y_onnx32[:, 0:3].argmax(axis=1)
acc = float((pred_label == label_test).mean())
print(f"sanity check on 50 fresh cases (seed 960000): accuracy = {acc:.3f}")

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
             input_dim=FEATURE_DIM,
             output_columns=["p_none", "p_leak", "p_constriction", "xi_frac", "kappa_leak", "kappa_constriction"],
             sanity_check_acc_n50=acc)
json.dump(report, open("results/17_export_report.json", "w"), indent=2)
print("\nsaved results/17_export_report.json")
print("DONE")
