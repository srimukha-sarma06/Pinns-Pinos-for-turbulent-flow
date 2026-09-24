"""Forward PINN check: leak params GIVEN (not inferred) -- validates the PINN formulation itself
against the MOC ground truth. This is the piece of the pipeline that is fully working and tested.

Run: python scripts/02_forward_pinn.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DDE_BACKEND", "pytorch")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from leakpinn.synth import make_dataset
from leakpinn.pinn import PINNConfig, build_problem, fit
from leakpinn.metrics import field_errors
from leakpinn.physics import G

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
err, (Hp, Qp) = field_errors(res, d)
print("field errors (leak params given, network only):", json.dumps(err, indent=2))
os.makedirs("results", exist_ok=True)
json.dump(dict(fixed=fixed, field_err=err, seconds=time.time() - t0),
          open("results/02_forward_pinn.json", "w"), indent=2)

tr = d.truth
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(d.t * 1e3, tr["H_field"][:, -1], label="MOC truth (valve)")
ax[0].plot(d.t * 1e3, Hp[-1], "--", label="PINN")
ax[0].plot(d.t * 1e3, d.H_meas[:, 1], ".", ms=2, alpha=.3, label="noisy sensor")
ax[0].set_xlabel("t [ms]"); ax[0].set_ylabel("head [m]")
ax[0].legend(); ax[0].grid(alpha=.3); ax[0].set_title("valve sensor")

mid = len(tr["x_grid"]) // 2
ax[1].plot(d.t * 1e3, tr["Q_field"][:, mid] * 1e3, label="MOC truth Q(mid)")
ax[1].plot(d.t * 1e3, Qp[mid] * 1e3, "--", label="PINN Q(mid)")
ax[1].set_xlabel("t [ms]"); ax[1].set_ylabel("flow [L/s]")
ax[1].legend(); ax[1].grid(alpha=.3); ax[1].set_title("reconstructed flow (never directly measured)")

plt.tight_layout()
plt.savefig("results/02_forward_pinn.png", dpi=130)
print("DONE")
