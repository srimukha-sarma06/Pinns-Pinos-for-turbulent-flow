"""Why a PINN beats a blind MOC simulation: wave-speed uncertainty.

Setup: the "real" pipe has the true (steel, no entrained air) wave speed. An engineer wrongly
assumes a lower value (as if there were unaccounted entrained air) and:
  (a) runs MOC forward BLINDLY with that wrong assumption -- MOC has no mechanism to notice or
      correct this, because it never looks at any sensor data at all.
  (b) trains a PINN against the two REAL (noisy) pressure sensors, using that same wrong value
      only as a starting point -- the PINN's wave-speed correction factor `rho` is free to move,
      so the sensor data pulls it back toward the truth.

This is the experiment that actually justifies "why use a PINN when you already have MOC":
MOC has no data-assimilation mechanism, a PINN does.

Run: python scripts/06_why_pinn_beats_moc.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DDE_BACKEND", "pytorch")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from leakpinn.moc import MOC
from leakpinn.synth import make_dataset
from leakpinn.pinn import PINNConfig, fit, predict_fields

# ---------------------------------------------------------------- the "real world"
a_true = None   # None -> synth.make_dataset uses the pipe's real Korteweg wave speed (~1388 m/s)
a_wrong = 1150.0   # an engineer's mistaken assumption -- e.g. unaccounted entrained air (~17% low)

d = make_dataset(leak_x=50.0, CdA=None, a_true=a_true, noise_std=0.10, seed=7)   # CdA=None -> no leak
a_true = d.truth["a"]
print(f"true wave speed   = {a_true:.1f} m/s")
print(f"wrong assumption  = {a_wrong:.1f} m/s  ({(a_wrong / a_true - 1) * 100:+.1f}% off)")

# ---------------------------------------------------------------- (a) naive MOC: blind forward run
m_naive = MOC(d.pipe, d.valve, a_wrong, 400, leak=None)
r_naive = m_naive.run(d.t[-1] + 2 * m_naive.dt)


def extract_grid(res_dict, x_query, t_query, key):
    """Pull a field onto a chosen (x,t) grid from a MOC result dict (nearest spatial node,
    linear interpolation in time)."""
    x_nodes, t_nodes, F = res_dict["x"], res_dict["t"], res_dict[key]
    out = np.empty((len(x_query), len(t_query)))
    for i, xq in enumerate(x_query):
        idx = int(np.argmin(np.abs(x_nodes - xq)))
        out[i] = np.interp(t_query, t_nodes, F[:, idx])
    return out


x_grid = d.truth["x_grid"]
Q_naive = extract_grid(r_naive, x_grid, d.t, "Q")
Q_true = d.truth["Q_field"].T   # (nx, nt), matches x_grid/d.t

naive_rmse = float(np.sqrt(np.mean((Q_naive - Q_true) ** 2)) * 1e3)   # L/s
print(f"naive blind MOC (wrong a, never sees sensor data): Q RMSE = {naive_rmse:.3f} L/s")

# ---------------------------------------------------------------- (b) PINN: same wrong prior, but
# trained against the real sensor data with the wave-speed correction factor free to move
cfg = PINNConfig(arch="ff", train_rho=True)
t0 = time.time()
res = fit(d, cfg, fixed=dict(xiL=0.5, kappa=0.0, cv=1.0), a_nom=a_wrong, verbose=True)
print("PINN trained in", time.time() - t0, "s")
a_hat = res.est["a"]
print(f"PINN's corrected wave speed = {a_hat:.1f} m/s  (started from {a_wrong:.1f}, true is {a_true:.1f})")

H_pinn, Q_pinn = predict_fields(res, x_grid, d.t)
pinn_rmse = float(np.sqrt(np.mean((Q_pinn - Q_true) ** 2)) * 1e3)
print(f"PINN (corrected via sensor data): Q RMSE = {pinn_rmse:.3f} L/s")
print(f"\n--> PINN is {naive_rmse / max(pinn_rmse, 1e-9):.1f}x more accurate than blind MOC, "
      f"despite starting from the SAME wrong wave-speed assumption.")

# ---------------------------------------------------------------- plot
os.makedirs("results", exist_ok=True)
mid = len(x_grid) // 2
val = -1   # valve sensor index in x_grid (last point = x=100m=valve)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, idx, label in [(axes[0], val, f"valve sensor, x={x_grid[val]:.0f} m"),
                       (axes[1], mid, f"mid-pipe (never measured), x={x_grid[mid]:.0f} m")]:
    ax.plot(d.t * 1e3, Q_true[idx] * 1e3, lw=2.5, color="black", label="truth")
    ax.plot(d.t * 1e3, Q_naive[idx] * 1e3, lw=1.8, ls="--", color="crimson",
            label=f"blind MOC (wrong a={a_wrong:.0f} m/s)")
    ax.plot(d.t * 1e3, Q_pinn[idx] * 1e3, lw=1.8, ls="--", color="tab:blue",
            label=f"PINN (corrected a={a_hat:.0f} m/s)")
    ax.set_title(f"flow Q(t) at {label}")
    ax.set_xlabel("t [ms]")
    ax.set_ylabel("flow [L/s]")
    ax.grid(alpha=.3)
    ax.legend(fontsize=9)
plt.suptitle(f"Same wrong wave-speed assumption ({(a_wrong/a_true-1)*100:+.0f}%) -- "
            f"blind MOC stays wrong, PINN corrects itself from sensor data", fontsize=11)
plt.tight_layout()
plt.savefig("results/06_why_pinn_beats_moc.png", dpi=140)
print("saved results/06_why_pinn_beats_moc.png")

report = dict(a_true=a_true, a_wrong=a_wrong, a_pinn_corrected=a_hat,
             naive_moc_Q_rmse_Ls=naive_rmse, pinn_Q_rmse_Ls=pinn_rmse,
             improvement_factor=naive_rmse / max(pinn_rmse, 1e-9))
json.dump(report, open("results/06_why_pinn_beats_moc.json", "w"), indent=2)
print("DONE")
