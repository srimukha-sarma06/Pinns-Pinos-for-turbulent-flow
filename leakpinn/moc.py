"""Method-of-characteristics (MOC) water-hammer solver with a pressure-dependent leak.

This is the *ground-truth generator* and a conventional benchmark.  It is the
standard Wylie & Streeter scheme (first-order, fixed grid dx = a dt) with
    * constant-head reservoir upstream,
    * orifice-type valve downstream (known command tau(t)),
    * quasi-steady Darcy-Weisbach friction with Re-dependent f,
    * a point leak Q_L = CL sqrt(H) at a grid node.
Ground truth should be generated on a *finer grid* than the one used by any
inversion code that calls it, to avoid the "inverse crime".
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import brentq
from .physics import Pipe, Valve, Leak, G


class MOC:
    def __init__(self, pipe: Pipe, valve: Valve, a: float, N: int, leak: Leak | None = None):
        self.p, self.v, self.a, self.N, self.leak = pipe, valve, a, N, leak
        self.dx = pipe.L / N
        self.dt = self.dx / a
        self.B = pipe.B(a)
        self.iL = None if leak is None else int(round(leak.x / self.dx))
        if leak is not None:
            assert 1 <= self.iL <= N - 1, "leak must be an interior node"
            self.x_leak_actual = self.iL * self.dx
        self.CL = 0.0 if leak is None else leak.CL()

    # friction loss coefficient over one reach: dH = R(Q) Q|Q|
    def _R(self, Q):
        f = self.p.friction_factor(Q)
        return f * self.dx / (2.0 * G * self.p.D * self.p.A ** 2)

    def steady(self):
        """Steady state consistent with the discretisation (so no spurious start-up transient)."""
        N, iL, p = self.N, self.iL, self.p
        tau0 = float(self.v.tau(0.0))

        def march(Q1):
            H = np.empty(N + 1); Q = np.empty(N + 1); Qu = None
            H[0] = p.H_res
            q = Q1
            for i in range(1, N + 1):
                H[i] = H[i - 1] - self._R(q) * q * abs(q)
                if iL is not None and i == iL:
                    qL = self.CL * np.sqrt(max(H[i], 0.0))
                    Qu = q
                    q = q - qL
                Q[i] = q
            Q[0] = Q1
            # flows on the upstream side of every node
            Qup = Q.copy()
            if iL is not None:
                Qup[iL] = Qu
            return H, Q, Qup

        def resid(Q1):
            H, Q, _ = march(Q1)
            return Q[N] - self.v.Cv * tau0 * np.sqrt(max(H[N], 0.0))

        # bracket scaled to this pipe's own design flow (not a fixed constant) so this solves
        # correctly across pipes of very different size, not just the ~2 L/s default
        qhi = max(5.0 * p.Q_design, 1e-3)
        Q1 = brentq(resid, 1e-8, qhi, xtol=1e-14, rtol=1e-13)
        H, Q, Qup = march(Q1)
        return H, Q, Qup   # Q = flow on the downstream side of each node

    def run(self, T: float, store_fields: bool = False):
        N, B, p, v = self.N, self.B, self.p, self.v
        nt = int(np.ceil(T / self.dt)) + 1
        H, Qd, Qu = self.steady()
        t_arr = np.arange(nt) * self.dt
        Hs = np.empty((nt, N + 1)) if store_fields else None
        Qs = np.empty((nt, N + 1)) if store_fields else None
        Hall = np.empty((nt, N + 1)); Qall = np.empty((nt, N + 1))
        Hall[0], Qall[0] = H, Qu
        iL = self.iL
        for k in range(1, nt):
            t = t_arr[k]
            Hn = np.empty_like(H); Qdn = np.empty_like(H); Qun = np.empty_like(H)
            # C+ characteristic arrives at node i from i-1 carrying the DOWNSTREAM flow of i-1
            qa = Qd[:-2]; qb = Qu[2:]
            CP = H[:-2] + B * qa - self._R(qa) * qa * np.abs(qa)
            CM = H[2:] - B * qb + self._R(qb) * qb * np.abs(qb)
            Hn[1:-1] = 0.5 * (CP + CM)
            Qdn[1:-1] = (CP - CM) / (2 * B)
            if iL is not None:
                j = iL - 1
                s = (-B * self.CL + np.sqrt((B * self.CL) ** 2 + 8.0 * (CP[j] + CM[j]))) / 4.0
                Hn[iL] = s * s
                Qun[iL] = (CP[j] - Hn[iL]) / B
                Qdn[iL] = (Hn[iL] - CM[j]) / B
            Qun[1:-1] = np.where(np.arange(1, N) == (iL if iL is not None else -1), Qun[1:-1], Qdn[1:-1])
            # reservoir (node 0)
            q1 = Qu[1]
            CM0 = H[1] - B * q1 + self._R(q1) * q1 * abs(q1)
            Hn[0] = p.H_res; Qdn[0] = (Hn[0] - CM0) / B; Qun[0] = Qdn[0]
            # valve (node N)
            qN = Qd[N - 1]
            CPN = H[N - 1] + B * qN - self._R(qN) * qN * abs(qN)
            c2 = (v.Cv * float(v.tau(t))) ** 2
            QN = 0.5 * (-c2 * B + np.sqrt((c2 * B) ** 2 + 4.0 * c2 * CPN))
            Hn[N] = CPN - B * QN; Qdn[N] = QN; Qun[N] = QN
            H, Qd, Qu = Hn, Qdn, Qun
            Hall[k], Qall[k] = H, Qu
        out = dict(t=t_arr, H=Hall, Q=Qall, dx=self.dx, x=np.arange(N + 1) * self.dx, a=self.a)
        return out
