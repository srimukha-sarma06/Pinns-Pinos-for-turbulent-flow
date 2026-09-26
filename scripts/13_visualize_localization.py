"""Visualizations for the leak-localization comparison (scripts/10_train_localization.py's
output): predicted-vs-true scatter for all 4 methods, an error comparison, and two example
reconstructed pressure fields (best case and worst case for Stage A) built from the INFERRED
leak parameters vs MOC ground truth -- shows what the localization error actually looks like in
the physical field, not just as a single distance number.

Run: DDE_BACKEND=pytorch python scripts/13_visualize_localization.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DDE_BACKEND", "pytorch")
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from leakpinn.general_pinn import GeneralConfig, GeneralNet, predict_physical
from leakpinn.domain import Scenario, sample_scenario
from leakpinn.physics import Leak
from leakpinn.moc import MOC

comp = json.load(open("results/10_localization_comparison.json"))
rows = comp["cases"]
N_HELDOUT, SEED_HELDOUT = comp["summary"]["n_heldout"], 900000    # must match scripts/10's constants
os.makedirs("results", exist_ok=True)

# Re-derive the EXACT original Scenario objects (H_res, Q_design, etc. aren't stored in the JSON,
# only L/D/a) by replaying the same RNG draw sequence scripts/10 used -- sample_scenario() can
# consume a variable number of draws per call (rejection sampling), so scenario i can only be
# reproduced by replaying draws 0..i from the same seed, not by reseeding with SEED_HELDOUT+i alone.
_rng = np.random.default_rng(SEED_HELDOUT)
scenarios = [sample_scenario(_rng) for _ in range(N_HELDOUT)]

# ================================================================== FIGURE 1: predicted vs true x_L
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharex=True, sharey=True)
methods = [("stageA", "Stage A (regression)"), ("stageAB", "Stage A+B (+ refine)"),
          ("tof", "time-of-flight"), ("moc", "MOC search")]
for ax, (key, label) in zip(axes, methods):
    true = np.array([r["x_L_true"] for r in rows])
    pred = np.array([r[f"x_L_{key}"] for r in rows])
    ax.scatter(true, pred, s=25, alpha=0.7)
    lims = [0, max(true.max(), pred.max()) * 1.05]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.5, label="perfect")
    ax.set_xlabel("true leak position [m]")
    ax.set_title(label)
    ax.grid(alpha=.3)
axes[0].set_ylabel("predicted leak position [m]")
axes[0].legend(fontsize=8)
plt.suptitle("Leak-location prediction vs. truth, 30 held-out pipes never used in training "
             "(perfect predictions fall on the dashed line)")
plt.tight_layout()
plt.savefig("results/13_localization_scatter.png", dpi=130)
print("saved results/13_localization_scatter.png")

# ================================================================== FIGURE 2: error comparison
fig, ax = plt.subplots(figsize=(8, 5))
errs = {label: np.abs([r[f"err_{key}_m"] for r in rows]) for key, label in
       [("stageA", "Stage A"), ("stageAB", "Stage A+B"), ("tof", "time-of-flight"), ("moc", "MOC search")]}
ax.boxplot(errs.values(), tick_labels=errs.keys(), showmeans=True)
ax.set_ylabel("|leak-location error| [m]")
ax.set_yscale("log")
ax.set_title("Localization error distribution, 30 held-out pipes (log scale)")
ax.grid(alpha=.3, axis="y")
plt.tight_layout()
plt.savefig("results/13_localization_error_boxplot.png", dpi=130)
print("saved results/13_localization_error_boxplot.png")

# ================================================================== FIGURE 3: reconstructed field, best/worst Stage A case
ckpt = torch.load("results/07_general_pinn.pt", map_location="cpu", weights_only=False)
gcfg = GeneralConfig(**ckpt["cfg"])
gen_net = GeneralNet(gcfg.width, gcfg.depth, gcfg.sigma_xi, gcfg.sigma_tau, gcfg.n_feat, gcfg.seed)
gen_net.load_state_dict(ckpt["state_dict"])
gen_net.eval()

abs_err = np.abs([r["err_stageA_m"] for r in rows])
i_best, i_worst = int(np.argmin(abs_err)), int(np.argmax(abs_err))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, i, tag in [(axes[0], i_best, "best Stage A case"), (axes[1], i_worst, "worst Stage A case")]:
    r = rows[i]
    sc = scenarios[i]                       # exact original scenario (H_res, Q_design, etc. intact)
    pipe, valve, a = sc.pipe, sc.valve, sc.a
    L = pipe.L
    leak_true = Leak(r["x_L_true"], r["CdA_true"])
    m = MOC(pipe, valve, a, 300, leak_true)
    T = 3.2 * L / a
    res = m.run(T)
    t = np.linspace(0, T, 200)
    H_true_valve = np.interp(t, res["t"], res["H"][:, 300])

    sc_pred = Scenario(pipe, valve, Leak(r["x_L_stageA"], r["CdA_stageA"]), a,
                       r["x_L_stageA"] / L, qL_frac=float("nan"))   # qL_frac unused downstream
    H_pred, _ = predict_physical(gen_net, sc_pred, np.array([L]), t)

    ax.plot(t * 1e3, H_true_valve, lw=2, label=f"MOC truth (leak at {r['x_L_true']:.1f} m)")
    ax.plot(t * 1e3, H_pred[0], "--", lw=2, label=f"reconstructed from Stage A's guess ({r['x_L_stageA']:.1f} m)")
    ax.set_title(f"{tag}: L={L:.0f}m, error={r['err_stageA_m']:+.1f}m")
    ax.set_xlabel("t [ms]"); ax.set_ylabel("head at valve [m]")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
plt.suptitle("What the localization error looks like physically: valve pressure trace using the "
            "TRUE leak vs. the field reconstructed from Stage A's inferred leak location/size "
            "(reconstruction uses the generalized forward model, results/07_general_pinn.pt)")
plt.tight_layout()
plt.savefig("results/13_localization_reconstruction.png", dpi=130)
print("saved results/13_localization_reconstruction.png")
print("DONE")
