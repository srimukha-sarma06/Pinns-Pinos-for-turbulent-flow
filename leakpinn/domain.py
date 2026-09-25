"""Randomized pipe/valve/leak scenario sampling for the GENERALIZED forward PINN
(leakpinn/general_pinn.py). leakpinn/pinn.py remains the original single-pipe (and
optionally inverse) code path; this module instead defines a *distribution* of pipes
so one network can be trained across all of them.

"Moderate industrial range" (confirmed with the user):
    pipe length     L        20  - 300   m
    inner diameter  D        25  - 200   mm   (~DN25 .. DN200)
    wave speed      a        300 - 1400  m/s  (spans HDPE/PVC up to steel)
    reservoir head  H_res    20  - 80    m gauge
    design velocity v        1.0 - 2.5   m/s  (typical hydraulic design range)
    leak location   x_L      5%  - 95%   of L
    leak size       q_L      0.5%- 10%   of design flow, at the local head

Wave speed is sampled directly rather than derived from a material (E, nu) model --
`synth.make_dataset(..., a_true=...)` and `baselines.py`'s "unknown wave speed
(PVC-ish)" test case already treat wave speed as an independently-known/uncertain
quantity, so this reuses an existing, already-validated pattern instead of inventing
a materials catalogue nobody asked for.

Valve schedule (20% closure over 20 ms) is kept FIXED across every sampled scenario:
it was not part of the requested range, and varying it would add another whole
conditioning dimension to the network for no requested benefit.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .physics import Pipe, Valve, Leak, G, design_valve
from .moc import MOC

TAU_END = 3.2          # fixed dimensionless training horizon (~3 pipe wave-transit times), same for every pipe
EPS_EDGE = 0.025        # leak-smoothing width in xi (dimensionless), same convention as leakpinn/pinn.py

RANGES = dict(
    L=(20.0, 300.0),
    D=(0.025, 0.200),
    a=(300.0, 1400.0),
    H_res=(20.0, 80.0),
    v_design=(1.0, 2.5),
    xiL=(0.05, 0.95),
    qL_frac=(0.005, 0.10),
)


@dataclass
class Scenario:
    pipe: Pipe
    valve: Valve
    leak: Leak
    a: float           # true wave speed [m/s] (independent of Pipe.wave_speed())
    xiL: float
    qL_frac: float      # leak flow / design flow at H_res -- the sampling target (actual realised
                         # value differs slightly once the leak head is solved for, see scenario_consts)


def sample_scenario(rng: np.random.Generator, max_tries: int = 50) -> Scenario:
    """Rejection-samples the ranges above, re-drawing whenever the combination is not
    hydraulically feasible: a long, narrow, fast pipe on a low reservoir head can demand
    more friction headloss than the reservoir actually has to give (design_valve() would
    then need a negative valve head, i.e. sqrt of a negative number) -- reject and re-draw
    rather than silently producing an unphysical scenario."""
    for _ in range(max_tries):
        L = float(rng.uniform(*RANGES["L"]))
        D = float(rng.uniform(*RANGES["D"]))
        a = float(rng.uniform(*RANGES["a"]))
        H_res = float(rng.uniform(*RANGES["H_res"]))
        v = float(rng.uniform(*RANGES["v_design"]))
        xiL = float(rng.uniform(*RANGES["xiL"]))
        qL_frac = float(rng.uniform(*RANGES["qL_frac"]))

        A = np.pi * D ** 2 / 4.0
        Q_design = v * A
        # e (wall thickness) and roughness only matter through Pipe.friction_factor() here --
        # wave_speed() is bypassed (we pass `a` explicitly everywhere below) -- so these just
        # need to be plausible for a thin-walled pipe of this diameter, not tied to one material.
        e = D / float(rng.uniform(9.0, 25.0))
        rough = float(rng.uniform(1.5e-6, 4.5e-5))
        pipe = Pipe(L=L, D=D, e=e, rough=rough, H_res=H_res, Q_design=Q_design)
        f = float(pipe.friction_factor(Q_design))
        hf = f * pipe.L / pipe.D * (Q_design / pipe.A) ** 2 / (2 * G)
        if hf > 0.6 * H_res:      # keep a healthy valve-head margin, same spirit as design_valve()
            continue
        valve = design_valve(pipe, dtau=0.20, t_close=0.020)
        x_L = xiL * L
        CdA = qL_frac * Q_design / np.sqrt(2.0 * G * H_res)   # inverted Torricelli, H_res as the local-head estimate
        leak = Leak(x_L, float(CdA))
        return Scenario(pipe, valve, leak, a, xiL, qL_frac)
    raise RuntimeError("sample_scenario: could not find a feasible combination in max_tries")


def scenario_consts(sc: Scenario, x1_ref: float = 10.0, N: int = 60) -> dict:
    """Per-scenario scalars needed by the PDE residual / steady-profile reconstruction.

    H1_0, H2_0 (the steady baseline heads at a reference point and at the valve) come from
    the MOC solver's own `steady()` method -- the same validated steady-state solve used
    everywhere else in this repo (see scripts/01_validate_moc.py) -- rather than a hand
    re-derived closed form, to avoid introducing a second, unchecked steady-state formula.
    """
    p, v, leak, a = sc.pipe, sc.valve, sc.leak, sc.a
    B_nom = p.B(a)
    f = float(p.friction_factor(p.Q_design))
    phi = f * p.L * G / (2.0 * p.D * a ** 2)
    m = MOC(p, v, a, N, leak)
    H, _, _ = m.steady()
    xi1 = min(x1_ref, 0.3 * p.L) / p.L          # keep the reference point inside the pipe even for short pipes
    n1 = int(round(xi1 * p.L / m.dx))
    H1_0, H2_0 = float(H[n1]), float(H[N])
    kappa = B_nom * leak.CdA * np.sqrt(2.0 * G)
    qv0 = B_nom * p.Q_design
    tau_c = a * v.t_close / p.L
    return dict(B_nom=B_nom, phi=phi, H1_0=H1_0, H2_0=H2_0, xi1=xi1, kappa=kappa, qv0=qv0, tau_c=tau_c)
