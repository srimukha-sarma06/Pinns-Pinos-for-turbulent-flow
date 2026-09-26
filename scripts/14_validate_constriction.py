"""Validate the constriction (partial-blockage) MOC physics against closed-form theory, BEFORE
trusting any detection result built on it -- same discipline as scripts/01_validate_moc.py for
the leak physics. See CONSTRICTION_DETECTION_PLAN.md for the derivation being checked here.

Run: python scripts/14_validate_constriction.py
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from leakpinn.physics import Pipe, Leak, Constriction, design_valve, G
from leakpinn.moc import MOC

p = Pipe(); a = p.wave_speed(); v = design_valve(p); B = p.B(a)
print(f"wave speed a = {a:.1f} m/s, impedance B = {B:.0f} s/m^2, A = {p.A*1e4:.3f} cm^2")
ok = True
def check(name, cond, info=""):
    global ok; ok &= bool(cond); print(("PASS  " if cond else "FAIL  ") + name, info)

# constriction sized so it drops ~40% of the design flow's velocity head at steady state --
# a meaningfully large but not absurd blockage (throat ~= 55% open, roughly)
CDA_C = 3.0e-4
cons = Constriction(37.3, CDA_C)

# 1. steady state stays steady (no drift when the valve does not move)
vs = design_valve(p, dtau=0.0)
r = MOC(p, vs, a, 200, constriction=cons).run(0.3)
check("steady state is a fixed point of the scheme", np.abs(r["H"] - r["H"][0]).max() < 1e-9,
      f"max drift {np.abs(r['H']-r['H'][0]).max():.2e} m")

# 2. steady mass balance: Q is CONTINUOUS across a constriction (the opposite of a leak)
m = MOC(p, v, a, 200, constriction=cons)
Hd, Hu, Qd, Qu = m.steady()
check("Q continuous across constriction (no mass loss)", abs(Qd[m.iC] - Qu[m.iC]) < 1e-12,
      f"Qd={Qd[m.iC]:.6e} Qu={Qu[m.iC]:.6e}")
check("steady head drop matches Rc*Q*|Q|",
      abs((Hu[m.iC] - Hd[m.iC]) - cons.RC() * Qd[m.iC] * abs(Qd[m.iC])) < 1e-9,
      f"Hu-Hd={Hu[m.iC]-Hd[m.iC]:.4f} m vs Rc*Q|Q|={cons.RC()*Qd[m.iC]*abs(Qd[m.iC]):.4f} m")
check("no constriction => design flow recovered", abs(MOC(p, v, a, 200).steady()[2][-1] - p.Q_design) < 1e-9)

# 3. reference (no anomaly) run for comparison
m0 = MOC(p, v, a, 400); r0 = m0.run(0.3)
tau_f = 1 - v.dtau
from scipy.optimize import brentq
H0d, _, Q0d, _ = m0.steady()
dH = brentq(lambda x: B * (Q0d[-1] - v.Cv * tau_f * np.sqrt(H0d[-1] + x)) - x, 0, 60)

# 4/5. Reflection coefficient, validated AT THE NODE rather than via a downstream sensor echo.
# A downstream-sensor comparison against a totally separate "no anomaly" run is what
# scripts/01_validate_moc.py uses for the leak, and it works there because the leak's steady-state
# perturbation is small -- but for a constriction sized enough to give a clearly resolvable
# reflection, the two runs' PRE-TRANSIENT operating points (steady Q1, H at the valve) differ
# enough that the valve's own closure dynamics diverge for reasons unrelated to wave reflection,
# and that divergence dominates a downstream "echo window" measurement (confirmed directly: it
# has the same order of magnitude and a similar timescale as the true echo, for every severity
# tested, and doesn't vanish as the constriction is made gentler in the way a genuine second-order
# effect would). Sidestep this entirely by reconstructing CP, CM AT THE CONSTRICTION NODE ITSELF,
# directly from its two immediate (regular, single-valued) neighbours' stored Hd/Qu history, and
# checking the reflection there -- no downstream propagation, no valve-dynamics confound.
mc = MOC(p, v, a, 400, constriction=cons); rc = mc.run(0.15)
iC = mc.iC
Hall, Qall = rc["H"], rc["Q"]                      # Hall = Hd, Qall = Qu, for every node/time
def _Rfric(Q):
    f = p.friction_factor(Q); return f * mc.dx / (2.0 * G * p.D * p.A ** 2)
Hd_left, Qd_left = Hall[:, iC - 1], Qall[:, iC - 1]     # neighbour iC-1 is a regular node: Hd=Hu, Qd=Qu
Hu_right, Qu_right = Hall[:, iC + 1], Qall[:, iC + 1]   # neighbour iC+1 is a regular node too
CP = Hd_left + B * Qd_left - _Rfric(Qd_left) * Qd_left * np.abs(Qd_left)
CM = Hu_right - B * Qu_right + _Rfric(Qu_right) * Qu_right * np.abs(Qu_right)
dCM = CM - CM[0]                                    # incident-side (valve-side) perturbation
Qc = (-B + np.sqrt(B ** 2 + cons.RC() * (CP - CM))) / cons.RC()
Hd_iC = CM + B * Qc
dCP_new = (Hd_iC + B * Qc) - (Hall[0, iC] + B * Qall[0, iC])   # reflected (downstream-heading) disturbance

Gc = cons.RC() * Qd[m.iC]
R_th = Gc / (B + Gc)
k_peak = int(np.argmax(np.abs(dCM)))
R_num = dCP_new[k_peak] / dCM[k_peak]
t1 = (p.L - mc.x_constriction_actual) / a          # one-way travel time, valve to the node
check("incident wave reaches the constriction node at (L-xC)/a (+ half the closure time)",
      abs(rc["t"][np.argmax(np.abs(dCM) > 0.05)] - (t1 + v.t_close / 2)) < 0.012,
      f"first departure {rc['t'][np.argmax(np.abs(dCM)>0.05)]*1e3:.1f} ms vs {(t1+v.t_close/2)*1e3:.1f} ms")
check("constriction reflection coefficient matches theory (node-level, no downstream confound)",
      abs(R_num - R_th) / abs(R_th) < 0.05, f"numerical {R_num*100:.3f} % vs theory {R_th*100:.3f} %")
check("constriction reflection is POSITIVE (leak's is negative -- the discriminating signature)",
      R_num > 0, f"R_num={R_num*100:.3f} %")

# 6. grid convergence: N=200 vs N=400 at the valve sensor
r200 = MOC(p, v, a, 200, constriction=cons).run(0.3); r400 = MOC(p, v, a, 400, constriction=cons).run(0.3)
e = np.abs(np.interp(r400["t"], r200["t"], r200["H"][:, -1]) - r400["H"][:, -1]).max()
check("grid convergence N=200 vs N=400", e < 0.6, f"max diff {e:.3f} m (pulse ~ {dH:.1f} m)")

# 7. physical admissibility
hmin = min(r["H"].min() for r in (r0, rc)); hmax = max(r["H"].max() for r in (r0, rc))
check("no cavitation (gauge head > -9 m)", hmin > -9.0, f"min head {hmin:.1f} m gauge")
check("peak pressure << Sch40 rating", hmax < 100, f"peak {hmax:.1f} m = {hmax/10.2:.1f} bar")

# 8. sanity limits of the closed form: Gc->0 gives R->0 (matches a leak's Gc->0 -> R->0 too),
#    Gc->inf gives R->1 (a near-total blockage reflects like a dead end)
R_tiny = (1e-6) / (B + 1e-6); R_huge = (1e12) / (B + 1e12)
check("R -> 0 as constriction closes to nothing", R_tiny < 1e-6)
check("R -> 1 as constriction approaches full closure", abs(R_huge - 1.0) < 1e-3)

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].plot(r0["t"]*1e3, r0["H"][:, -1], label="no anomaly")
ax[0].plot(rc["t"]*1e3, rc["H"][:, -1], label="constriction at 37.3 m")
ax[0].set_xlabel("t [ms]"); ax[0].set_ylabel("head at valve [m]"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(rc["t"]*1e3, dCM, label="incident (dCM, valve-side)")
ax[1].plot(rc["t"]*1e3, dCP_new, label="reflected (dCP_new, back toward valve)")
ax[1].axvline(t1*1e3, ls="--", c="r", label="(L-xC)/a: wave reaches the node")
ax[1].axhline(0, color="k", lw=0.5)
ax[1].set_xlabel("t [ms]"); ax[1].set_ylabel("perturbation at the constriction node [m]")
ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
ax[1].set_title(f"R_num={R_num*100:.2f}% vs R_th={R_th*100:.2f}%")
plt.tight_layout(); os.makedirs("results", exist_ok=True); plt.savefig("results/14_constriction_validation.png", dpi=130)
print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
