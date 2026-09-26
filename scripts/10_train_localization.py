"""Train Stage A (amortized regression) of the leak-localization inverse model, apply Stage B
(physics-informed refinement using the already-trained generalized forward model), and compare
against the classical baselines (leakpinn/baselines.py) on the SAME held-out scenarios.
See INVERSE_LOCALIZATION_PLAN.md for the full methodology.

Run: DDE_BACKEND=pytorch python scripts/10_train_localization.py
"""
import sys, os, json, time, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DDE_BACKEND", "pytorch")
import numpy as np
import torch

from leakpinn.localize_data import generate_dataset, build_example
from leakpinn.localize import fit_localizer, predict, refine, DTYPE
from leakpinn.domain import sample_scenario
from leakpinn.general_pinn import GeneralConfig, GeneralNet
from leakpinn.baselines import time_of_flight, moc_search
from leakpinn.synth import make_dataset
from leakpinn.physics import G

N_TRAIN = 24000
N_HELDOUT = 30           # smaller: moc_search's grid search is the expensive comparison, see below
SEED_TRAIN = 0
SEED_HELDOUT = 900000    # disjoint seed range from training, never touched before this script

os.makedirs("results", exist_ok=True)

# ---------------------------------------------------------------------- Stage A training data
train_cache = "results/.10_train_cache.npz"
if os.path.exists(train_cache):
    d = np.load(train_cache)
    X, xiL, log_kappa = d["X"], d["xiL"], d["log_kappa"]
    print(f"loaded cached training set: {len(X)} examples")
else:
    X, xiL, log_kappa, _ = generate_dataset(N_TRAIN, seed0=SEED_TRAIN, n_workers=8)
    np.savez(train_cache, X=X, xiL=xiL, log_kappa=log_kappa)

# ---------------------------------------------------------------------- Stage A training
print("training Stage A (amortized regression) ...")
res = fit_localizer(X, xiL, log_kappa, width=64, depth=3, iters=12000, lr=1e-3, batch=256, verbose=True)
print(f"Stage A trained in {res.seconds:.0f}s")
torch.save(dict(state_dict=res.net.state_dict(), x_mean=res.x_mean, x_std=res.x_std),
          "results/10_localizer.pt")
print("saved results/10_localizer.pt")

# ---------------------------------------------------------------------- load the forward model for Stage B
ckpt = torch.load("results/07_general_pinn.pt", map_location="cpu", weights_only=False)
gcfg = GeneralConfig(**ckpt["cfg"])
gen_net = GeneralNet(gcfg.width, gcfg.depth, gcfg.sigma_xi, gcfg.sigma_tau, gcfg.n_feat, gcfg.seed)
gen_net.load_state_dict(ckpt["state_dict"])
gen_net.eval()
from leakpinn.pinn import DEVICE
gen_net = gen_net.to(DEVICE)

# ---------------------------------------------------------------------- held-out comparison: Stage A, Stage A+B, moc_search, time_of_flight
rng = np.random.default_rng(SEED_HELDOUT)
rows = []
t0 = time.time()
for i in range(N_HELDOUT):
    sc = sample_scenario(rng)
    p, v, leak, a = sc.pipe, sc.valve, sc.leak, sc.a
    T = 3.2 * p.L / a
    d = make_dataset(pipe=p, leak_x=leak.x, CdA=leak.CdA, a_true=a, T=T, N_truth=300,
                     dtau=v.dtau, t_close=v.t_close, seed=int(SEED_HELDOUT + i))
    feat, _, log_k_true, extras = build_example(sc, seed=int(SEED_HELDOUT + i))
    pred = predict(res, feat[None, :])[0]
    xiL_A, kappa_A = float(pred[0]), float(pred[1])
    x_L_A = xiL_A * p.L
    CdA_A = kappa_A / (extras["B_nom"] * np.sqrt(2 * G))

    xiL_B, kappa_B = refine(gen_net, d, a, xiL_A, kappa_A, steps=80, lr=3e-2)
    x_L_B = xiL_B * p.L
    CdA_B = kappa_B / (extras["B_nom"] * np.sqrt(2 * G))

    tof = time_of_flight(d)
    # moc_search's DEFAULT cda_grid ([1,2,4,8] mm^2) is scaled for the original 100 m default pipe
    # -- across leakpinn.domain's wider pipe-size range, the CdA needed for the SAME leak_frac
    # (0.5-10% of design flow) can be orders of magnitude different, so the default grid would be
    # badly mismatched here and make this baseline look artificially worse than it is. Build a
    # scenario-appropriate grid from the same qL_frac range domain.py actually samples from, and
    # cap x_grid resolution so a 300 m pipe doesn't blow up the search cost vs. a 100 m one.
    # coarser than the naive L/150 step: a 300 m pipe blew the grid-search cost up to minutes/case
    # (740+ MOC simulations); capping to ~40 x-positions regardless of L keeps this comparable in
    # cost across the whole pipe-size range, at the cost of coarser localization for big pipes --
    # an apples-to-apples time budget matters more here than each case being moc_search's best case.
    cda_grid = np.array([0.005, 0.02, 0.10]) * p.Q_design / np.sqrt(2 * G * p.H_res)
    x_step = max(1.0, (p.L - 5.0) / 40.0)
    x_grid = np.arange(3.0, p.L - 2.0, x_step)
    moc = moc_search(d, a_nominal=a, cda_grid=cda_grid, x_grid=x_grid)

    truth_x_L = sc.xiL * p.L
    row = dict(
        L=p.L, D=p.D, a=a, xiL_true=sc.xiL, x_L_true=truth_x_L, CdA_true=leak.CdA,
        x_L_stageA=x_L_A, CdA_stageA=CdA_A, err_stageA_m=x_L_A - truth_x_L,
        x_L_stageAB=x_L_B, CdA_stageAB=CdA_B, err_stageAB_m=x_L_B - truth_x_L,
        x_L_tof=tof["x_L"], err_tof_m=tof["x_L"] - truth_x_L,
        x_L_moc=moc["x_L"], err_moc_m=moc["x_L"] - truth_x_L,
    )
    rows.append(row)
    print(f"  [{i:3d}] L={p.L:6.1f}m true_xL={truth_x_L:6.1f}  "
          f"A={x_L_A:6.1f}(err {row['err_stageA_m']:+6.1f})  "
          f"A+B={x_L_B:6.1f}(err {row['err_stageAB_m']:+6.1f})  "
          f"tof={tof['x_L']:6.1f}(err {row['err_tof_m']:+6.1f})  "
          f"moc={moc['x_L']:6.1f}(err {row['err_moc_m']:+6.1f})")

print(f"\nheld-out comparison done in {time.time()-t0:.0f}s")

def summarize(key):
    e = np.array([abs(r[key]) for r in rows])
    return dict(mae_m=float(e.mean()), median_m=float(np.median(e)), max_m=float(e.max()),
               frac_within_1m=float((e < 1.0).mean()), frac_within_5m=float((e < 5.0).mean()))

summary = dict(
    stageA=summarize("err_stageA_m"),
    stageA_plus_B=summarize("err_stageAB_m"),
    time_of_flight=summarize("err_tof_m"),
    moc_search=summarize("err_moc_m"),
    n_heldout=N_HELDOUT,
)
print("\n=== SUMMARY (mean abs error in metres, on", N_HELDOUT, "held-out unseen pipes) ===")
print(json.dumps(summary, indent=2))

json.dump(dict(cases=rows, summary=summary), open("results/10_localization_comparison.json", "w"), indent=2)
print("saved results/10_localization_comparison.json")
print("DONE")
