"""Method-of-characteristics (MOC) water-hammer solver with a pressure-dependent leak and/or a
partial-blockage constriction.

This is the *ground-truth generator* and a conventional benchmark.  It is the
standard Wylie & Streeter scheme (first-order, fixed grid dx = a dt) with
    * constant-head reservoir upstream,
    * orifice-type valve downstream (known command tau(t)),
    * quasi-steady Darcy-Weisbach friction with Re-dependent f,
    * a point leak Q_L = CL sqrt(H) at a grid node (Q discontinuous, H continuous there), and/or
    * a point constriction Hu-Hd = Rc*Q*|Q| at a grid node (H discontinuous, Q continuous there --
      the mirror image of the leak; see CONSTRICTION_DETECTION_PLAN.md for the derivation).
Ground truth should be generated on a *finer grid* than the one used by any
inversion code that calls it, to avoid the "inverse crime".

State is tracked as FOUR arrays per node -- Hd/Hu (downstream-/upstream-facing head) and Qd/Qu
(downstream-/upstream-facing flow) -- rather than a single H and a single Q. For an ordinary node
(no leak, no constriction) these collapse to Hd==Hu and Qd==Qu, exactly reproducing the original
single-valued behaviour; a leak node has Hd==Hu (H continuous) but Qd!=Qu; a constriction node has
Qd==Qu (Q continuous) but Hd!=Hu. "Downstream-facing" values (Hd, Qd) are what's reported as "the"
field at a node for output/plotting -- sensors in this project are never placed exactly at an
anomaly node, so this choice doesn't affect any sensor-based detection result, it's just a
convention for the rare case something queries the field there directly.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import brentq
from .physics import Pipe, Valve, Leak, Constriction, G


class MOC:
    def __init__(self, pipe: Pipe, valve: Valve, a: float, N: int, leak: Leak | None = None,
                constriction: Constriction | None = None):
        self.p, self.v, self.a, self.N, self.leak, self.constriction = pipe, valve, a, N, leak, constriction
        self.dx = pipe.L / N
        self.dt = self.dx / a
        self.B = pipe.B(a)
        self.iL = None if leak is None else int(round(leak.x / self.dx))
        if leak is not None:
            assert 1 <= self.iL <= N - 1, "leak must be an interior node"
            self.x_leak_actual = self.iL * self.dx
        self.CL = 0.0 if leak is None else leak.CL()
        self.iC = None if constriction is None else int(round(constriction.x / self.dx))
        if constriction is not None:
            assert 1 <= self.iC <= N - 1, "constriction must be an interior node"
            assert constriction is None or leak is None or self.iC != self.iL, \
                "leak and constriction at the same node is not supported"
            self.x_constriction_actual = self.iC * self.dx
        self.RC = 0.0 if constriction is None else constriction.RC()

    # friction loss coefficient over one reach: dH = R(Q) Q|Q|
    def _R(self, Q):
        f = self.p.friction_factor(Q)
        return f * self.dx / (2.0 * G * self.p.D * self.p.A ** 2)

    def steady(self):
        """Steady state consistent with the discretisation (so no spurious start-up transient)."""
        N, iL, iC, p = self.N, self.iL, self.iC, self.p
        tau0 = float(self.v.tau(0.0))

        def march(Q1):
            Hd = np.empty(N + 1); Hu = np.empty(N + 1); Qd = np.empty(N + 1); Qu = np.empty(N + 1)
            Hd[0] = Hu[0] = p.H_res
            q = Q1
            for i in range(1, N + 1):
                h_after_friction = Hd[i - 1] - self._R(q) * q * abs(q)   # crossing reach [i-1, i]
                Qu[i] = q
                if iC is not None and i == iC:
                    Hu[i] = h_after_friction
                    Hd[i] = Hu[i] - self.RC * q * abs(q)
                else:
                    Hu[i] = Hd[i] = h_after_friction
                if iL is not None and i == iL:
                    qL = self.CL * np.sqrt(max(Hd[i], 0.0))
                    q = q - qL
                Qd[i] = q
            Hd[0] = Hu[0] = p.H_res; Qd[0] = Qu[0] = Q1
            return Hd, Hu, Qd, Qu

        def resid(Q1):
            Hd, _, Qd, _ = march(Q1)
            return Qd[N] - self.v.Cv * tau0 * np.sqrt(max(Hd[N], 0.0))

        # bracket scaled to this pipe's own design flow (not a fixed constant) so this solves
        # correctly across pipes of very different size, not just the ~2 L/s default
        qhi = max(5.0 * p.Q_design, 1e-3)
        Q1 = brentq(resid, 1e-8, qhi, xtol=1e-14, rtol=1e-13)
        return march(Q1)   # Hd, Hu, Qd, Qu

    def run(self, T: float, store_fields: bool = False):
        N, B, p, v = self.N, self.B, self.p, self.v
        nt = int(np.ceil(T / self.dt)) + 1
        Hd, Hu, Qd, Qu = self.steady()
        t_arr = np.arange(nt) * self.dt
        Hall = np.empty((nt, N + 1)); Qall = np.empty((nt, N + 1))
        Hall[0], Qall[0] = Hd, Qu
        iL, iC = self.iL, self.iC
        for k in range(1, nt):
            t = t_arr[k]
            Hdn = np.empty_like(Hd); Hun = np.empty_like(Hu)
            Qdn = np.empty_like(Qd); Qun = np.empty_like(Qu)
            # C+ characteristic arrives at node i from i-1's DOWNSTREAM-facing (Hd, Qd) values;
            # C- arrives at node i from i+1's UPSTREAM-facing (Hu, Qu) values -- see module
            # docstring / CONSTRICTION_DETECTION_PLAN.md for why these are the correct faces.
            qa = Qd[:-2]; qb = Qu[2:]
            CP = Hd[:-2] + B * qa - self._R(qa) * qa * np.abs(qa)
            CM = Hu[2:] - B * qb + self._R(qb) * qb * np.abs(qb)
            Hn_reg = 0.5 * (CP + CM)
            Qn_reg = (CP - CM) / (2 * B)
            Hdn[1:-1] = Hun[1:-1] = Hn_reg
            Qdn[1:-1] = Qun[1:-1] = Qn_reg
            if iL is not None:
                j = iL - 1
                s = (-B * self.CL + np.sqrt((B * self.CL) ** 2 + 8.0 * (CP[j] + CM[j]))) / 4.0
                Hdn[iL] = Hun[iL] = s * s
                Qun[iL] = (CP[j] - Hdn[iL]) / B
                Qdn[iL] = (Hdn[iL] - CM[j]) / B
            if iC is not None:
                j = iC - 1
                Qc = (-B + np.sqrt(B ** 2 + self.RC * (CP[j] - CM[j]))) / self.RC
                Qdn[iC] = Qun[iC] = Qc
                Hun[iC] = CP[j] - B * Qc
                Hdn[iC] = CM[j] + B * Qc
            # reservoir (node 0)
            q1 = Qu[1]
            CM0 = Hu[1] - B * q1 + self._R(q1) * q1 * abs(q1)
            Hdn[0] = Hun[0] = p.H_res; Qdn[0] = (Hdn[0] - CM0) / B; Qun[0] = Qdn[0]
            # valve (node N)
            qN = Qd[N - 1]
            CPN = Hd[N - 1] + B * qN - self._R(qN) * qN * abs(qN)
            c2 = (v.Cv * float(v.tau(t))) ** 2
            QN = 0.5 * (-c2 * B + np.sqrt((c2 * B) ** 2 + 4.0 * c2 * CPN))
            Hdn[N] = Hun[N] = CPN - B * QN; Qdn[N] = QN; Qun[N] = QN
            Hd, Hu, Qd, Qu = Hdn, Hun, Qdn, Qun
            Hall[k], Qall[k] = Hd, Qu
        out = dict(t=t_arr, H=Hall, Q=Qall, dx=self.dx, x=np.arange(N + 1) * self.dx, a=self.a)
        return out
