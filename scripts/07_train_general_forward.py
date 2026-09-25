"""Train the GENERALIZED forward PINN (leakpinn/general_pinn.py): one network across a
distribution of pipes (leakpinn/domain.py's "moderate industrial range"), instead of
leakpinn/pinn.py's single fixed pipe. Then validates it against the MOC ground-truth
solver on scenarios NEVER seen during training -- same "never feed the truth to the
model" discipline as leakpinn/metrics.py.

Run: DDE_BACKEND=pytorch python scripts/07_train_general_forward.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DDE_BACKEND", "pytorch")
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from leakpinn.general_pinn import GeneralConfig, fit_general, predict_physical
from leakpinn.domain import sample_scenario
from leakpinn.moc import MOC

cfg = GeneralConfig(
    width=64, depth=4, n_feat=24, sigma_xi=1.0, sigma_tau=4.0,
    n_scenarios=64, pts_per_scenario=150, bnd_per_scenario=32, refresh_every=40,
    adam_iters=6000, lr=2e-3, lr_decay_steps=1500, lr_decay_rate=0.6,
    lbfgs_iters=400, lbfgs_scenarios=128,
    w_pde=1.0, w_bc=4.0, seed=0, display_every=250,
)
# (A larger/longer run -- width=96, depth=5, 12k Adam + 800 L-BFGS iters, ~3400s on an RTX 3050 --
# was also tried and is NOT clearly better: H NRMSE improved slightly (0.214 -> 0.201) but Q NRMSE
# got worse (0.140 -> 0.179) on the same 12 held-out pipes, for ~7x the training time. This smaller
# config is what's actually shipped in results/ -- see README.md's generalized-model section.)

print("training the generalized forward PINN across randomized pipes ...")
t0 = time.time()
res = fit_general(cfg, verbose=True)
print(f"done in {res.seconds:.0f} s")

os.makedirs("results", exist_ok=True)
torch.save(dict(state_dict=res.net.state_dict(), cfg=cfg.__dict__), "results/07_general_pinn.pt")
print("saved results/07_general_pinn.pt")

# ---------------------------------------------------------------------- held-out validation
# Fresh scenarios drawn with a DIFFERENT seed than training -- never used for a gradient step.
N_HOLDOUT = 12
rng = np.random.default_rng(999)
rows = []
for i in range(N_HOLDOUT):
    sc = sample_scenario(rng)
    N_truth = 300
    m = MOC(sc.pipe, sc.valve, sc.a, N_truth, sc.leak)
    T = 0.30 * sc.pipe.L / 100.0 * (1388.0 / sc.a)   # scale the recording window with this pipe's own transit time
    r = m.run(T)
    nx_eval = 60
    x_eval = np.linspace(sc.pipe.L / N_truth, sc.pipe.L * (1 - 1.0 / N_truth), nx_eval)   # avoid the exact ends
    t_eval = np.linspace(0.0, T, 80)
    idx = np.array([int(round(xq / m.dx)) for xq in x_eval])
    H_true = np.stack([np.interp(t_eval, r["t"], r["H"][:, i]) for i in idx], axis=1).T   # (nx, nt)
    Q_true = np.stack([np.interp(t_eval, r["t"], r["Q"][:, i]) for i in idx], axis=1).T
    H_pred, Q_pred = predict_physical(res.net, sc, x_eval, t_eval)

    sH = np.abs(H_true - H_true[:, :1]).max()
    sQ = np.abs(Q_true - Q_true[:, :1]).max()
    H_nrmse = float(np.sqrt(np.mean((H_pred - H_true) ** 2)) / max(sH, 1e-9))
    Q_nrmse = float(np.sqrt(np.mean((Q_pred - Q_true) ** 2)) / max(sQ, 1e-9))
    H_rmse_m = float(np.sqrt(np.mean((H_pred - H_true) ** 2)))
    Q_rmse_Ls = float(np.sqrt(np.mean((Q_pred - Q_true) ** 2)) * 1e3)
    rows.append(dict(L=sc.pipe.L, D=sc.pipe.D, a=sc.a, H_res=sc.pipe.H_res, xiL=sc.xiL,
                     qL_frac=sc.qL_frac, H_nrmse=H_nrmse, Q_nrmse=Q_nrmse,
                     H_rmse_m=H_rmse_m, Q_rmse_Ls=Q_rmse_Ls))
    print(f"  holdout {i:2d}  L={sc.pipe.L:6.1f}m D={sc.pipe.D*1e3:5.1f}mm a={sc.a:5.0f} "
          f"xiL={sc.xiL:.2f}  H_nrmse={H_nrmse:.3f}  Q_nrmse={Q_nrmse:.3f}  H_rmse={H_rmse_m:.2f}m")

summary = dict(
    H_nrmse_mean=float(np.mean([r["H_nrmse"] for r in rows])),
    Q_nrmse_mean=float(np.mean([r["Q_nrmse"] for r in rows])),
    H_rmse_m_mean=float(np.mean([r["H_rmse_m"] for r in rows])),
    Q_rmse_Ls_mean=float(np.mean([r["Q_rmse_Ls"] for r in rows])),
    n_holdout=N_HOLDOUT, train_seconds=res.seconds, adam_iters=cfg.adam_iters, lbfgs_iters=cfg.lbfgs_iters,
)
print("\nheld-out summary:", json.dumps(summary, indent=2))
json.dump(dict(cases=rows, summary=summary, history=res.history), open("results/07_general_pinn_validation.json", "w"), indent=2)
print("saved results/07_general_pinn_validation.json")

# ---------------------------------------------------------------------- plot: one held-out case
sc = sample_scenario(np.random.default_rng(2024))
N_truth = 300
m = MOC(sc.pipe, sc.valve, sc.a, N_truth, sc.leak)
T = 0.30 * sc.pipe.L / 100.0 * (1388.0 / sc.a)
r = m.run(T)
t_eval = np.linspace(0.0, T, 200)
x_eval = np.array([sc.pipe.L])   # valve sensor
H_pred, Q_pred = predict_physical(res.net, sc, x_eval, t_eval)
H_true_valve = np.interp(t_eval, r["t"], r["H"][:, N_truth])

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(t_eval * 1e3, H_true_valve, label="MOC truth")
ax.plot(t_eval * 1e3, H_pred[0], "--", label="generalized PINN")
ax.set_xlabel("t [ms]"); ax.set_ylabel("head at valve [m]")
ax.set_title(f"held-out pipe: L={sc.pipe.L:.0f} m, D={sc.pipe.D*1e3:.0f} mm, a={sc.a:.0f} m/s, "
            f"leak at {sc.xiL*100:.0f}% of L")
ax.legend(); ax.grid(alpha=.3)
plt.tight_layout()
plt.savefig("results/07_general_pinn_example.png", dpi=130)
print("saved results/07_general_pinn_example.png")
print("DONE")
