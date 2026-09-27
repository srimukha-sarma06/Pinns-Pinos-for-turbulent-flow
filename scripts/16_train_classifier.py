"""Train the amortized none/leak/constriction classifier (leakpinn/classify_net.py) on data from
leakpinn/classify_data.py, and grade it against ground truth on a disjoint held-out seed range.
See CONSTRICTION_DETECTION_PLAN.md for how this relates to the physics-search
baselines.classify_and_localize (the accuracy reference this network is a deployable surrogate
for, not a replacement of).

Run: python scripts/16_train_classifier.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from leakpinn.classify_data import generate_dataset, LABEL_TO_KIND
from leakpinn.classify_net import fit_classifier, predict, ClassifyFitResult

N_TRAIN = 20000
N_HELDOUT = 300
SEED_TRAIN = 0
SEED_HELDOUT = 950000     # disjoint from training (0..20000), from the earlier 24-case physics
                          # search validation set (700000..700024), and from the leak-only
                          # localizer's train/heldout ranges (0..24000, 800000, 900000)

os.makedirs("results", exist_ok=True)

train_cache = "results/.16_train_cache.npz"
if os.path.exists(train_cache):
    d = np.load(train_cache)
    X, label, xi_frac, log_sev = d["X"], d["label"], d["xi_frac"], d["log_sev"]
    print(f"loaded cached training set: {len(X)} examples")
else:
    t0 = time.time()
    X, label, xi_frac, log_sev, _ = generate_dataset(N_TRAIN, seed0=SEED_TRAIN, n_workers=8)
    print(f"generated {N_TRAIN} training examples in {time.time()-t0:.0f}s")
    np.savez(train_cache, X=X, label=label, xi_frac=xi_frac, log_sev=log_sev)

print("class balance (train):", {LABEL_TO_KIND[k]: int((label == k).sum()) for k in (0, 1, 2)})

print("training classifier ...")
res = fit_classifier(X, label, xi_frac, log_sev, width=96, depth=3, iters=8000, lr=1e-3,
                     batch=256, verbose=True)
print(f"trained in {res.seconds:.0f}s")
torch.save(dict(state_dict=res.net.state_dict(), x_mean=res.x_mean, x_std=res.x_std),
          "results/16_classifier.pt")
print("saved results/16_classifier.pt")

# ---------------------------------------------------------------------------- held-out grading
heldout_cache = "results/.16_heldout_cache.npz"
if os.path.exists(heldout_cache):
    d = np.load(heldout_cache, allow_pickle=True)
    Xh, labelh, xih, sevh, metash = d["X"], d["label"], d["xi_frac"], d["log_sev"], d["metas"]
else:
    Xh, labelh, xih, sevh, metash = generate_dataset(N_HELDOUT, seed0=SEED_HELDOUT, n_workers=8)
    np.savez(heldout_cache, X=Xh, label=labelh, xi_frac=xih, log_sev=sevh, metas=metash)

pred = predict(res, Xh)
pred_label = pred[:, 0:3].argmax(axis=1)
acc = float((pred_label == labelh).mean())

conf = {LABEL_TO_KIND[t]: {LABEL_TO_KIND[p]: 0 for p in (0, 1, 2)} for t in (0, 1, 2)}
for t, p in zip(labelh, pred_label):
    conf[LABEL_TO_KIND[int(t)]][LABEL_TO_KIND[int(p)]] += 1

# localization error in METERS, only for rows correctly classified as an anomaly (matches the
# physics-search report's convention: mislocalizing a case that was already misclassified isn't a
# separate failure mode worth double-counting)
errs = []
for i in range(len(Xh)):
    t, p = int(labelh[i]), int(pred_label[i])
    if t == 0 or t != p:
        continue
    L = float(metash[i]["L"])
    errs.append(abs(pred[i, 3] - xih[i]) * L)
errs = np.array(errs)

summary = dict(
    n_train=N_TRAIN, n_heldout=N_HELDOUT, seed_heldout=SEED_HELDOUT,
    accuracy=acc, confusion_matrix=conf,
    localization_mae_m=float(errs.mean()) if len(errs) else None,
    localization_median_m=float(np.median(errs)) if len(errs) else None,
    n_localized=int(len(errs)),
)
print("\n=== held-out summary ===")
print(json.dumps(summary, indent=2))
json.dump(summary, open("results/16_classifier_validation.json", "w"), indent=2)
print("saved results/16_classifier_validation.json")
print("DONE")
