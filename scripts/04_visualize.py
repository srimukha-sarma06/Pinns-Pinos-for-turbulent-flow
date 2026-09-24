"""Space-time contour maps + multi-time snapshot panels for H(x,t) and Q(x,t).

The pipe is 1-D in space, so there's no literal x-y analogue of a 2-D flow-field contour
plot. The natural equivalent for a 1-D+time PDE is a single contour map with x on one axis
and t on the other (colour = field value) -- that's Figure 1. Figure 2 recreates the
"snapshots at several times" layout directly: one column per time, PINN vs the MOC ground
truth overlaid.

Run: python scripts/04_visualize.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DDE_BACKEND", "pytorch")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from leakpinn.synth import make_dataset
from leakpinn.pinn import PINNConfig, build_problem, fit, predict_fields
from leakpinn.physics import G

# ---------------------------------------------------------------- train (forward, params given)
d = make_dataset(leak_x=37.3, CdA=3.6e-6, seed=0)
cfg = PINNConfig(arch="ff", schedule=[
    dict(opt="adam", iters=3000, free=False),
    dict(opt="lbfgs", iters=800, free=False),
])
prob = build_problem(d, cfg)
fixed = dict(
    xiL=d.truth["leak_x"] / d.pipe.L,
    kappa=prob.B_nom * d.truth["CdA"] * np.sqrt(2 * G),
    rho=d.truth["a"] / prob.a_nom,
    cv=1.0,
)
t0 = time.time()
res = fit(d, cfg, fixed=fixed, verbose=True)
print("trained in", time.time() - t0, "s")

# ---------------------------------------------------------------- dense (x,t) grid for both fields
nx, nt = 150, 300
x = np.linspace(0.0, d.pipe.L, nx)
t = np.linspace(0.0, d.t[-1], nt)
H, Q = predict_fields(res, x, t)          # PINN, shape (nx, nt)

# ground truth on the same grid, from the MOC solver used to build the dataset
tr = d.truth
Ht = np.stack([np.interp(t, d.t, np.interp(x, tr["x_grid"], tr["H_field"][k])) for k in range(len(d.t))], axis=1) \
    if False else None
# simpler: interpolate truth (defined on tr["x_grid"] x d.t) onto our (x, t) grid directly
from scipy.interpolate import RegularGridInterpolator
Hi = RegularGridInterpolator((tr["x_grid"], d.t), tr["H_field"].T, bounds_error=False, fill_value=None)
Qi = RegularGridInterpolator((tr["x_grid"], d.t), tr["Q_field"].T, bounds_error=False, fill_value=None)
XX, TT = np.meshgrid(x, t, indexing="ij")
H_true = Hi((XX, TT))
Q_true = Qi((XX, TT))

os.makedirs("results", exist_ok=True)

# ================================================================== FIGURE 1: space-time contours
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
panels = [
    ("head H [m]", H, H_true, "turbo"),
    ("flow Q [L/s]", Q * 1e3, Q_true * 1e3, "turbo"),
]
for row, (label, pinn_f, true_f, cmap) in enumerate(panels):
    vmin, vmax = true_f.min(), true_f.max()
    for col, (title, field) in enumerate([("PINN", pinn_f), ("MOC truth", true_f),
                                           ("|PINN - truth|", np.abs(pinn_f - true_f))]):
        ax = axes[row, col]
        c = ax.contourf(x, t * 1e3, field.T, levels=30,
                        cmap=cmap if col < 2 else "Reds",
                        vmin=(vmin if col < 2 else None), vmax=(vmax if col < 2 else None))
        plt.colorbar(c, ax=ax)
        ax.set_title(f"{label}, {title}" if row == 0 or col != 0 else title)
        ax.set_xlabel("x [m]")
        if col == 0:
            ax.set_ylabel("t [ms]")
        if row == 0:
            ax.axvline(d.truth["leak_x"], color="k", ls="--", lw=1, alpha=.6)
            ax.axvline(d.x_sensors[0], color="w", ls=":", lw=1, alpha=.8)
            ax.axvline(d.x_sensors[1], color="w", ls=":", lw=1, alpha=.8)
plt.tight_layout()
plt.savefig("results/04_spacetime_contours.png", dpi=140)
print("saved results/04_spacetime_contours.png")

# ================================================================== FIGURE 2: snapshot-at-times grid
snap_t = [0.02, 0.05, 0.10, 0.25]   # seconds -> matches the "columns = times" look of the reference
fig, axes = plt.subplots(2, len(snap_t), figsize=(4 * len(snap_t), 7), sharex=True)
for col, ts in enumerate(snap_t):
    k = np.argmin(np.abs(t - ts))
    axH, axQ = axes[0, col], axes[1, col]
    axH.plot(x, H_true[:, k], lw=2, label="MOC truth")
    axH.plot(x, H[:, k], "--", lw=2, label="PINN")
    axH.axvline(d.truth["leak_x"], color="gray", ls=":", lw=1, label="leak")
    axH.set_title(f"H(x), t={ts*1e3:.0f} ms")
    axH.grid(alpha=.3)
    if col == 0:
        axH.set_ylabel("head [m]")
        axH.legend(fontsize=8)

    axQ.plot(x, Q_true[:, k] * 1e3, lw=2, label="MOC truth")
    axQ.plot(x, Q[:, k] * 1e3, "--", lw=2, label="PINN")
    axQ.axvline(d.truth["leak_x"], color="gray", ls=":", lw=1)
    axQ.set_title(f"Q(x), t={ts*1e3:.0f} ms")
    axQ.set_xlabel("x [m]")
    axQ.grid(alpha=.3)
    if col == 0:
        axQ.set_ylabel("flow [L/s]")
plt.tight_layout()
plt.savefig("results/04_snapshots.png", dpi=140)
print("saved results/04_snapshots.png")

err = dict(H_rmse=float(np.sqrt(np.mean((H - H_true) ** 2))),
          Q_rmse_Ls=float(np.sqrt(np.mean((Q - Q_true) ** 2)) * 1e3))
json.dump(err, open("results/04_visualize.json", "w"), indent=2)
print("field errors:", err)
print("DONE")
