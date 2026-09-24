"""Validate the ground-truth generator against closed-form hydraulics BEFORE trusting any ML result."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from leakpinn.physics import Pipe, Leak, design_valve, G
from leakpinn.moc import MOC

p = Pipe(); a = p.wave_speed(); v = design_valve(p); B = p.B(a)
print(f"wave speed a = {a:.1f} m/s, impedance B = {B:.0f} s/m^2, A = {p.A*1e4:.3f} cm^2")
ok = True
def check(name, cond, info=""):
    global ok; ok &= bool(cond); print(("PASS  " if cond else "FAIL  ") + name, info)

# 1. steady state stays steady (no drift when the valve does not move)
class Still(type(v)): pass
vs = design_valve(p, dtau=0.0)
r = MOC(p, vs, a, 200, Leak(37.3, 3.6e-6)).run(0.3)
check("steady state is a fixed point of the scheme", np.abs(r["H"] - r["H"][0]).max() < 1e-9,
      f"max drift {np.abs(r['H']-r['H'][0]).max():.2e} m")

# 2. mass balance of steady state with a leak
m = MOC(p, v, a, 200, Leak(37.3, 3.6e-6)); H, Qd, Qu = m.steady()
check("steady mass balance across leak", abs((Qu[m.iL] - Qd[m.iL]) - m.CL * np.sqrt(H[m.iL])) < 1e-12)
check("no leak => design flow recovered", abs(MOC(p, v, a, 200).steady()[1][-1] - p.Q_design) < 1e-9)

# 3. Joukowsky: first plateau at the valve equals B*(Q0-Q) with the valve law (no leak, before any reflection)
m0 = MOC(p, v, a, 400); r0 = m0.run(0.3); H0, Q0, _ = m0.steady()
tau_f = 1 - v.dtau
from scipy.optimize import brentq
dH = brentq(lambda x: B * (Q0[-1] - v.Cv * tau_f * np.sqrt(H0[-1] + x)) - x, 0, 60)
k = np.searchsorted(r0["t"], 0.04)  # after closure, before reservoir echo (2L/a = 0.144 s)
check("Joukowsky plateau at the valve", abs((r0["H"][k, -1] - H0[-1]) - dH) < 0.15,
      f"MOC {r0['H'][k,-1]-H0[-1]:.2f} m vs closed form {dH:.2f} m")

# 4. reservoir echo (R=-1) brings the valve head back down after 2L/a
kk = np.searchsorted(r0["t"], 0.05)
t_dn = r0["t"][kk + np.argmax(r0["H"][kk:, -1] - H0[-1] < 0.5 * dH)]
t_th = v.t_close / 2 + 2 * p.L / a
check("reservoir echo returns after ~2L/a", abs(t_dn - t_th) < 0.008, f"{t_dn*1e3:.1f} ms vs {t_th*1e3:.1f} ms")

# 5. reflection coefficient of the leak, R = -GB/(2+GB)
leak = Leak(37.3, 3.6e-6); ml = MOC(p, v, a, 400, leak); rl = ml.run(0.3); Hl, _, _ = ml.steady()
G_ = leak.CL() * 0.5 / np.sqrt(Hl[ml.iL]); R_th = -G_ * B / (2 + G_ * B)
diff = rl["H"][:, -1] - r0["H"][:, -1]
t_echo = 2 * (p.L - ml.x_leak_actual) / a
w = (rl["t"] > t_echo) & (rl["t"] < t_echo + 0.03)
R_num = diff[w][np.argmax(np.abs(diff[w]))] / dH
dd = diff - diff[0]                       # remove the (tiny) steady-state offset the leak causes
t_dep = rl["t"][np.argmax(np.abs(dd) > 0.10)]   # first clear departure from the leak-free trace
check("leak echo arrives at 2(L-xL)/a (+ half the closure time)", abs(t_dep - (t_echo + v.t_close / 2)) < 0.012,
      f"first departure {t_dep*1e3:.1f} ms vs {(t_echo+v.t_close/2)*1e3:.1f} ms")
check("reflection coefficient matches linear theory", abs(R_num - R_th) / abs(R_th) < 0.25,
      f"numerical {R_num*100:.2f} % vs theory {R_th*100:.2f} %  (echo {diff[w][np.argmax(np.abs(diff[w]))]:.2f} m)")

# 6. grid convergence: N=200 vs N=400 at the sensor
r200 = MOC(p, v, a, 200, Leak(37.5, 3.6e-6)).run(0.3); r400 = MOC(p, v, a, 400, Leak(37.5, 3.6e-6)).run(0.3)
e = np.abs(np.interp(r400["t"], r200["t"], r200["H"][:, -1]) - r400["H"][:, -1]).max()
check("grid convergence N=200 vs N=400", e < 0.6, f"max diff {e:.3f} m (pulse ~ {dH:.1f} m)")

# 7. physical admissibility: no column separation / cavitation, pressures below pipe rating
hmin = min(r["H"].min() for r in (r0, rl)); hmax = max(r["H"].max() for r in (r0, rl))
check("no cavitation (gauge head > -9 m)", hmin > -9.0, f"min head {hmin:.1f} m gauge")
check("peak pressure << Sch40 rating", hmax < 100, f"peak {hmax:.1f} m = {hmax/10.2:.1f} bar")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].plot(r0["t"]*1e3, r0["H"][:, -1], label="no leak"); ax[0].plot(rl["t"]*1e3, rl["H"][:, -1], label="leak 37.3 m, 5% flow")
ax[0].set_xlabel("t [ms]"); ax[0].set_ylabel("head at valve [m]"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(rl["t"]*1e3, diff); ax[1].axvline(t_echo*1e3, ls="--", c="r", label="2(L-xL)/a")
ax[1].set_xlabel("t [ms]"); ax[1].set_ylabel("leak minus no-leak [m]"); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); os.makedirs("results", exist_ok=True); plt.savefig("results/01_moc_validation.png", dpi=130)
print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
