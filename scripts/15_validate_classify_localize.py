"""Held-out validation of the three-way anomaly detector (leakpinn.baselines.classify_and_localize):
draws random scenarios (none / leak / constriction) from leakpinn.domain.sample_anomaly_scenario
across the full "moderate industrial range", runs the classifier, and reports classification
accuracy (confusion matrix) and localization error given the correct class -- same honest,
side-by-side reporting style as results/03_baseline_comparison.json and
results/10_localization_comparison.json.

Run: python scripts/15_validate_classify_localize.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from leakpinn.domain import sample_anomaly_scenario
from leakpinn.synth import make_dataset
from leakpinn.baselines import classify_and_localize

N_CASES = 24
SEED = 700000   # fresh seed range, never used in any earlier training/validation in this project

rng = np.random.default_rng(SEED)
rows = []
t0 = time.time()
for i in range(N_CASES):
    sc = sample_anomaly_scenario(rng)
    p, valve, a = sc.pipe, sc.valve, sc.a
    T = 3.2 * p.L / a
    if sc.kind == "leak":
        d = make_dataset(pipe=p, leak_x=sc.leak.x, CdA=sc.leak.CdA, a_true=a, T=T, N_truth=300,
                         dtau=valve.dtau, t_close=valve.t_close, seed=SEED + i)
        truth_x = sc.leak.x
    elif sc.kind == "constriction":
        d = make_dataset(pipe=p, leak_x=1.0, CdA=None, a_true=a, T=T, N_truth=300,
                         dtau=valve.dtau, t_close=valve.t_close, seed=SEED + i,
                         constriction=sc.constriction)
        truth_x = sc.constriction.x
    else:
        d = make_dataset(pipe=p, leak_x=1.0, CdA=None, a_true=a, T=T, N_truth=300,
                         dtau=valve.dtau, t_close=valve.t_close, seed=SEED + i)
        truth_x = None

    r = classify_and_localize(d, a_nominal=a)
    pred_x = dict(leak=r["leak"]["x_L"], constriction=r["constriction"]["x_C"]).get(r["kind"])
    loc_err = None if (truth_x is None or pred_x is None or r["kind"] != sc.kind) else abs(pred_x - truth_x)
    row = dict(true_kind=sc.kind, pred_kind=r["kind"], L=p.L, D=p.D, a=a,
              truth_x=truth_x, pred_x=pred_x, loc_err_m=loc_err, seconds=r["seconds"])
    rows.append(row)
    print(f"  [{i:2d}] true={sc.kind:12s} pred={r['kind']:12s} "
          f"{'OK ' if sc.kind==r['kind'] else 'ERR'}  "
          f"truth_x={truth_x if truth_x is None else round(truth_x,1)}  pred_x={pred_x if pred_x is None else round(pred_x,1)}  "
          f"({r['seconds']:.0f}s)")

print(f"\ndone in {time.time()-t0:.0f}s")

kinds = ["none", "leak", "constriction"]
confusion = {t: {p: 0 for p in kinds} for t in kinds}
for r in rows:
    confusion[r["true_kind"]][r["pred_kind"]] += 1
accuracy = sum(r["true_kind"] == r["pred_kind"] for r in rows) / len(rows)
loc_errs = [r["loc_err_m"] for r in rows if r["loc_err_m"] is not None]
summary = dict(
    n_cases=N_CASES, accuracy=accuracy, confusion_matrix=confusion,
    localization_mae_m=(float(np.mean(loc_errs)) if loc_errs else None),
    localization_median_m=(float(np.median(loc_errs)) if loc_errs else None),
    n_localized=len(loc_errs),
)
print("\n=== confusion matrix (rows=true, cols=predicted) ===")
for t in kinds:
    print(f"  {t:12s}: " + "  ".join(f"{p}={confusion[t][p]}" for p in kinds))
print("\n=== summary ===")
print(json.dumps(summary, indent=2))

os.makedirs("results", exist_ok=True)
json.dump(dict(cases=rows, summary=summary), open("results/15_classify_localize_validation.json", "w"), indent=2)
print("saved results/15_classify_localize_validation.json")
print("DONE")
