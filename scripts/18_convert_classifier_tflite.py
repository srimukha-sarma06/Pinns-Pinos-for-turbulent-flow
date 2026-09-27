"""Convert the exported classifier ONNX model (scripts/17_export_classifier_onnx.py) to
TensorFlow Lite (float32), and verify it against both the ONNX graph and the actual TFLM op list --
same procedure as scripts/09/12 (see those files for why: onnx2tf mutating its input file in
place, batch=1 being both the deployable shape and the correct thing to test, and re-checking the
ACTUAL compiled .tflite's ops rather than trusting the ONNX graph's op names).

Run: python scripts/18_convert_classifier_tflite.py
     (does not need DDE_BACKEND -- it never imports leakpinn/torch/deepxde)
"""
import sys, os, io, re, json, shutil, subprocess, contextlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

ONNX_PATH = "results/17_classifier_fp32.onnx"
OUT_DIR = "results/18_tflite_model"
FEATURE_DIM = 206

if not os.path.exists(ONNX_PATH):
    raise SystemExit(f"{ONNX_PATH} not found -- run scripts/17_export_classifier_onnx.py first")

try:
    import onnx2tf  # noqa: F401
    import onnxruntime as ort
    import tensorflow as tf
except ImportError as e:
    raise SystemExit(f"missing dependency ({e}) -- run: pip install onnx2tf tensorflow")

os.makedirs("results", exist_ok=True)
if os.path.isdir(OUT_DIR):
    shutil.rmtree(OUT_DIR)
scratch_onnx = "results/.18_scratch_input.onnx"
shutil.copy(ONNX_PATH, scratch_onnx)
cmd = [sys.executable, "-m", "onnx2tf", "-i", scratch_onnx, "-o", OUT_DIR, "-osd", "-ois",
      f"sensor_features:1,{FEATURE_DIM}"]
print("running:", " ".join(cmd))
subprocess.run(cmd, check=True)
os.remove(scratch_onnx)

produced = [f for f in os.listdir(OUT_DIR) if f.endswith(".tflite")] if os.path.isdir(OUT_DIR) else []
print("produced .tflite files:", produced)
fp32_candidates = [f for f in produced if "float32" in f]
if not fp32_candidates:
    raise SystemExit(f"no float32 .tflite found among {produced} in {OUT_DIR}")
src = os.path.join(OUT_DIR, fp32_candidates[0])
final_path = "results/anomaly_classifier_fp32.tflite"
shutil.copy(src, final_path)
print(f"saved fp32 TFLite model to {final_path} ({os.path.getsize(final_path)} bytes)")

# ---------------------------------------------------------------- TFLM op-list check, on the ACTUAL .tflite
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

# ---------------------------------------------------------------- sanity check vs the ONNX graph (batch=1)
rng = np.random.default_rng(0)
N = 32
x = rng.normal(0, 1, (N, FEATURE_DIM)).astype(np.float32)

sess = ort.InferenceSession(ONNX_PATH)
interp = tf.lite.Interpreter(model_path=final_path)
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]

y_onnx = np.empty((N, 6), dtype=np.float32)
y_tflite = np.empty((N, 6), dtype=np.float32)
for i in range(N):
    row = x[i:i + 1]
    y_onnx[i] = sess.run(None, {"sensor_features": row})[0][0]
    interp.set_tensor(inp["index"], row)
    interp.invoke()
    y_tflite[i] = interp.get_tensor(out["index"])[0]

max_diff = float(np.abs(y_onnx - y_tflite).max())
print(f"max |ONNX - TFLite fp32| over {N} random test points (batch=1 each) = {max_diff:.3e}")
report = dict(tflite_path=final_path, tflite_bytes=os.path.getsize(final_path),
             onnx_vs_tflite_max_diff=max_diff, n_test_points=N,
             tflite_ops_used=tflite_ops, tflite_unsupported_ops=unsupported_tflite,
             output_columns=["p_none", "p_leak", "p_constriction", "xi_frac", "kappa_leak", "kappa_constriction"])
json.dump(report, open("results/18_tflite_report.json", "w"), indent=2)
print("saved results/18_tflite_report.json")
print("DONE")
