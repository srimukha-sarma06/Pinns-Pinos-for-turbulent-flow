"""Convert the exported generalized-forward-PINN ONNX model (scripts/08_export_general_onnx.py)
to TensorFlow Lite (float32) for the Renesas RA8/FSP TFLM toolchain, and sanity-check the
converted .tflite gives the same answers as the ONNX graph it came from.

This needs `onnx2tf` + `tensorflow`, which scripts/05-08 deliberately avoid depending on
(DEPLOYMENT.md documents the same onnx2tf command as a manual next step for that reason).
Install once with:

    pip install onnx2tf tensorflow

Run: python scripts/09_convert_tflite.py
     (does not need DDE_BACKEND -- it never imports leakpinn/torch/deepxde)
"""
import sys, os, io, re, json, shutil, subprocess, contextlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

ONNX_PATH = "results/08_general_forward_fp32.onnx"
OUT_DIR = "results/09_tflite_model"

if not os.path.exists(ONNX_PATH):
    raise SystemExit(f"{ONNX_PATH} not found -- run scripts/08_export_general_onnx.py first")

try:
    import onnx2tf  # noqa: F401
    import onnxruntime as ort
    import tensorflow as tf
except ImportError as e:
    raise SystemExit(f"missing dependency ({e}) -- run: pip install onnx2tf tensorflow")

# ---------------------------------------------------------------- convert
# `-osd` skips onnx2tf's own dynamic-range/int8 quantized outputs (we only want the plain
# float32 graph -- fp16/int8 variants would need a representative-data calibration pass we
# have no reason to do here) so we get exactly one .tflite: the fp32 model.
# `-ois scenario_xy:1,10` fixes the batch dimension to 1 -- exactly right for on-device
# inference (one query point at a time), which is also the only shape the TFLM interpreter
# needs to support. NOTE: onnx2tf's shape-fixing step rewrites the ONNX file it's given
# IN PLACE (confirmed empirically: the dynamic-batch axis scripts/08 exported gets replaced
# with a fixed [1, 10]) -- so it's given a scratch COPY here, never the ONNX_PATH original,
# so re-running scripts/08 isn't needed just to get the dynamic-batch file back.
os.makedirs("results", exist_ok=True)
if os.path.isdir(OUT_DIR):
    shutil.rmtree(OUT_DIR)
scratch_onnx = "results/.09_scratch_input.onnx"
shutil.copy(ONNX_PATH, scratch_onnx)
cmd = [sys.executable, "-m", "onnx2tf", "-i", scratch_onnx, "-o", OUT_DIR, "-osd", "-ois", "scenario_xy:1,10"]
print("running:", " ".join(cmd))
subprocess.run(cmd, check=True)
os.remove(scratch_onnx)

produced = [f for f in os.listdir(OUT_DIR) if f.endswith(".tflite")] if os.path.isdir(OUT_DIR) else []
print("produced .tflite files:", produced)
fp32_candidates = [f for f in produced if "float32" in f]
if not fp32_candidates:
    raise SystemExit(f"no float32 .tflite found among {produced} in {OUT_DIR} -- inspect onnx2tf's output above")
src = os.path.join(OUT_DIR, fp32_candidates[0])
final_path = "results/09_general_forward_fp32.tflite"
shutil.copy(src, final_path)
print(f"saved fp32 TFLite model to {final_path} ({os.path.getsize(final_path)} bytes)")

# ---------------------------------------------------------------- TFLM op-list check, on the ACTUAL .tflite
# scripts/08's op check is only a proxy: it checks the ONNX graph's op names, but onnx2tf's own
# lowering can rewrite an op into a DIFFERENT (still TFLM-native) op -- e.g. this model's raw
# `z @ B` Fourier-feature MatMul becomes TFLite BATCH_MATMUL, and its multi-slice indexing gets
# fused into SPLIT. Both are confirmed present in the current TFLM kernel registry
# (tensorflow/lite/micro/kernels/micro_ops.h: `Register_BATCH_MATMUL()`, `Register_SPLIT()`,
# checked directly against github.com/tensorflow/tflite-micro at export time) -- but re-verify
# THIS model's actual op list here rather than assuming, since a different network architecture
# could lower differently.
TFLM_SUPPORTED_TFLITE = {
    "ADD", "SUB", "MUL", "DIV", "FULLY_CONNECTED", "BATCH_MATMUL", "TANH", "LOGISTIC",
    "SIN", "COS", "CONCATENATION", "RESHAPE", "TRANSPOSE", "SLICE", "SPLIT", "SPLIT_V",
    "SQUEEZE", "SQRT", "EXP", "LOG", "NEG", "ABS", "RELU", "SOFTMAX",
}
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    tf.lite.experimental.Analyzer.analyze(model_path=final_path)
tflite_ops = sorted(set(re.findall(r"\b([A-Z][A-Z0-9_]+)\(", buf.getvalue())) - {"T"})
unsupported_tflite = [op for op in tflite_ops if op not in TFLM_SUPPORTED_TFLITE]
print("actual .tflite op list:", tflite_ops)
if unsupported_tflite:
    print("!! NOT confirmed in the TFLM kernel list:", unsupported_tflite)
else:
    print("All ops in the FINAL .tflite confirmed present in TFLM's kernel registry.")

# ---------------------------------------------------------------- sanity check vs the ONNX graph
# One row at a time (batch=1) -- that's what the deployed .tflite actually supports (see above)
# and also exactly matches how a microcontroller would call it: one query point per inference.
rng = np.random.default_rng(0)
N = 32
# representative-ish random inputs matching leakpinn.general_pinn.DeployNet's 10-column
# contract: xi,tau in [0,1]-ish ranges, xiL in [0,1], kappa/phi positive, H1n/H2n ~ O(0.2-0.8)
x = np.column_stack([
    rng.uniform(0, 1, N), rng.uniform(0, 3, N), rng.uniform(0.05, 0.95, N),
    rng.uniform(0.05, 5.0, N), rng.uniform(1e-5, 5e-3, N),
    rng.uniform(0.2, 0.8, N), rng.uniform(0.2, 0.8, N), rng.uniform(0.03, 0.3, N),
    rng.uniform(1e4, 2e5, N), rng.uniform(1e-3, 1.0, N),
]).astype(np.float32)

sess = ort.InferenceSession(ONNX_PATH)
interp = tf.lite.Interpreter(model_path=final_path)
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]

y_onnx = np.empty((N, 2), dtype=np.float32)
y_tflite = np.empty((N, 2), dtype=np.float32)
for i in range(N):
    row = x[i:i + 1]
    y_onnx[i] = sess.run(None, {"scenario_xy": row})[0][0]
    interp.set_tensor(inp["index"], row)
    interp.invoke()
    y_tflite[i] = interp.get_tensor(out["index"])[0]

max_diff = float(np.abs(y_onnx - y_tflite).max())
print(f"max |ONNX - TFLite fp32| over {N} random test points (batch=1 each) = {max_diff:.3e}")
report = dict(tflite_path=final_path, tflite_bytes=os.path.getsize(final_path),
             onnx_vs_tflite_max_diff=max_diff, n_test_points=N,
             tflite_ops_used=tflite_ops, tflite_unsupported_ops=unsupported_tflite)
json.dump(report, open("results/09_tflite_report.json", "w"), indent=2)
print("saved results/09_tflite_report.json")
print("DONE")
