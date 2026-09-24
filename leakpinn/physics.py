"""Physical constants, pipe definition and closed-form helpers.

Every default below is a real, checkable engineering value (see comments).
Nothing here is tuned to make the ML look good.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

G = 9.80665  # m/s^2


@dataclass
class Pipe:
    # --- geometry: carbon-steel pipe, DN50 Schedule 40 (ASME B36.10) ---------
    L: float = 100.0          # m, length of the monitored section (bench/test-loop scale)
    D: float = 0.0525         # m, inner diameter (DN50 Sch40: OD 60.3 mm, wall 3.91 mm)
    e: float = 0.00391        # m, wall thickness
    E: float = 200e9          # Pa, Young's modulus of carbon steel
    nu: float = 0.30          # Poisson ratio of steel
    rough: float = 4.5e-5     # m, absolute roughness of commercial steel (Moody chart)
    # --- fluid: water at 20 degC --------------------------------------------
    rho: float = 998.2        # kg/m^3
    mu: float = 1.002e-3      # Pa s
    K: float = 2.18e9         # Pa, bulk modulus of water at 20 degC
    # --- operating point ------------------------------------------------------
    H_res: float = 40.0       # m gauge head of the constant-head supply (~3.9 bar)
    Q_design: float = 2.0e-3  # m^3/s (2 L/s) design flow with valve fully open, no leak

    @property
    def A(self) -> float:
        return np.pi * self.D ** 2 / 4.0

    def wave_speed(self) -> float:
        """Korteweg/Joukowsky wave speed, thin-walled pipe anchored against axial
        movement (c1 = 1 - nu^2).  Real pipes with a little entrained air are
        slower, which is why `a` is treated as UNCERTAIN in the inverse problem."""
        c1 = 1.0 - self.nu ** 2
        return float(np.sqrt((self.K / self.rho) / (1.0 + c1 * self.K * self.D / (self.E * self.e))))

    def B(self, a: float) -> float:
        """Pipe characteristic impedance a/(g A)  [s/m^2]:  dH = +-B dQ across a wave."""
        return a / (G * self.A)

    def friction_factor(self, Q):
        """Darcy friction factor.  Swamee-Jain (turbulent, 5e3<Re<1e8, error <1% vs
        Colebrook) with a laminar 64/Re branch and smooth blending."""
        Q = np.abs(np.asarray(Q, dtype=float))
        Re = np.maximum(self.rho * (Q / self.A) * self.D / self.mu, 1.0)
        turb = 0.25 / np.log10(self.rough / (3.7 * self.D) + 5.74 / Re ** 0.9) ** 2
        lam = 64.0 / Re
        return np.where(Re < 2000.0, lam, np.where(Re > 4000.0, turb,
                        lam + (turb - lam) * (Re - 2000.0) / 2000.0))


@dataclass
class Valve:
    """Downstream control valve discharging to atmosphere: Q = Cv * tau(t) * sqrt(H_v).

    tau(t) is the *commanded* relative opening (known to us: we drive the valve).
    A partial, smooth closure (half-cosine) of `dtau` (20 % of the opening) over `t_close` seconds.
    20 ms for a 20 % closure is achievable with a fast pneumatic / servo actuator;
    a slower valve simply lowers the bandwidth of the probe (see README)."""
    Cv: float = 1.0
    dtau: float = 0.20
    t_close: float = 0.020

    def tau(self, t):
        u = np.clip(np.asarray(t, dtype=float) / self.t_close, 0.0, 1.0)
        return 1.0 - self.dtau * 0.5 * (1.0 - np.cos(np.pi * u))


@dataclass
class Leak:
    x: float                  # m from the reservoir
    CdA: float                # m^2, discharge coefficient * orifice area
    # Q_leak = CdA * sqrt(2 g H)  (Torricelli / orifice law)

    def CL(self) -> float:
        return self.CdA * np.sqrt(2.0 * G)


def design_valve(pipe: Pipe, dtau=0.20, t_close=0.020) -> Valve:
    """Choose Cv so that the leak-free pipe passes exactly Q_design at full opening."""
    # friction-only steady state (no leak):  H_v = H_res - h_f(Q)
    Q = pipe.Q_design
    f = float(pipe.friction_factor(Q))
    hf = f * pipe.L / pipe.D * (Q / pipe.A) ** 2 / (2 * G)
    Hv = pipe.H_res - hf
    return Valve(Cv=Q / np.sqrt(Hv), dtau=dtau, t_close=t_close)
