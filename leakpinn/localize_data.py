"""Pure NumPy/SciPy half of leak localization (leakpinn/localize.py): scenario sampling, MOC
simulation, and the feature-vector contract shared between training-data generation and real
inference. Deliberately has NO torch/deepxde import, so `generate_dataset`'s multiprocessing pool
(default 'fork' on Linux) never forks a process with a CUDA context already open -- that's a real
crash risk (undefined behaviour re-using a forked CUDA context), not a hypothetical one, so this
split exists specifically to avoid it, not just for tidiness.
"""
from __future__ import annotations
import time
from multiprocessing import Pool
import numpy as np

from .physics import G
from .domain import Scenario, sample_scenario, TAU_END
from .synth import make_dataset, Dataset

N_SAMP = 100                  # resampled points PER CHANNEL (2 channels -> 200 trace inputs)
TAU_WINDOW = TAU_END           # same dimensionless recording horizon as the forward model
FEATURE_DIM = 2 * N_SAMP + 6


def features_from_dataset(d: Dataset, a_est: float):
    """Build the feature vector the SAME way regardless of whether `d` came from simulation
    (training) or a real field capture (deployment) -- everything used here is either a direct
    sensor reading (`d.H_meas`, `d.H_pre`) or a known pipe/valve constant (`d.pipe`, `d.valve`,
    `a_est`), never the true leak parameters. Column layout (206 total):
      [0:100]   resampled perturbation trace, sensor 1 (near reservoir)
      [100:200] resampled perturbation trace, sensor 2 (at the valve)
      [200] log(phi)      dimensionless friction number
      [201] H1n           reference head near reservoir [m] / 100 (measured baseline)
      [202] H2n           reference head at the valve [m] / 100 (measured baseline)
      [203] log(B_nom)    pipe impedance a_est/(g*A)
      [204] log(qv0)      B_nom * Q_design
      [205] xi1           reference-sensor position / L
    """
    p = d.pipe
    A = p.A
    B_nom = a_est / (G * A)
    f = float(p.friction_factor(p.Q_design))
    phi = f * p.L * G / (2.0 * p.D * a_est ** 2)
    base = d.H_pre.mean(axis=0)              # MEASURED pre-transient baseline (H1_0, H2_0)
    H1_0, H2_0 = float(base[0]), float(base[1])
    xi1 = float(d.x_sensors[0] / p.L)
    qv0 = B_nom * p.Q_design

    hm = d.H_meas - base                      # perturbation, same as leakpinn.baselines.perturbation()
    tau_grid = np.linspace(0.0, TAU_WINDOW, N_SAMP, endpoint=False)
    t_grid = tau_grid * p.L / a_est
    h1_rs = np.interp(t_grid, d.t, hm[:, 0], right=hm[-1, 0])
    h2_rs = np.interp(t_grid, d.t, hm[:, 1], right=hm[-1, 1])

    scalars = np.array([np.log(max(phi, 1e-12)), H1_0 / 100.0, H2_0 / 100.0,
                        np.log(max(B_nom, 1e-9)), np.log(max(qv0, 1e-9)), xi1])
    feat = np.concatenate([h1_rs, h2_rs, scalars]).astype(np.float64)
    extras = dict(B_nom=B_nom, phi=phi, H1_0=H1_0, H2_0=H2_0, xi1=xi1, qv0=qv0, a_est=a_est)
    return feat, extras


def build_example(sc: Scenario, seed: int, N_truth: int = 300):
    """One labelled example: simulate a scenario's sensor data, then build the feature vector.
    Wave speed `a` is used exactly as sampled (treated as accurately known/estimated -- e.g. via
    leakpinn.baselines.estimate_wave_speed on the real pulse transit). Deliberate simplification,
    flagged in INVERSE_LOCALIZATION_PLAN.md's open questions: NOT re-estimated with error injected
    here, so reported accuracy assumes a good wave-speed estimate, not robustness to a bad one."""
    p, v, leak, a = sc.pipe, sc.valve, sc.leak, sc.a
    T = TAU_WINDOW * p.L / a
    d = make_dataset(pipe=p, leak_x=leak.x, CdA=leak.CdA, a_true=a, T=T, N_truth=N_truth,
                     dtau=v.dtau, t_close=v.t_close, seed=seed)
    feat, extras = features_from_dataset(d, a)
    log_kappa = float(np.log(leak.CdA * (a / (G * p.A)) * np.sqrt(2 * G)))
    return feat, float(sc.xiL), log_kappa, extras


def _gen_one(seed: int):
    rng = np.random.default_rng(seed)
    sc = sample_scenario(rng)
    feat, xiL, log_kappa, _ = build_example(sc, seed=seed)
    return feat, xiL, log_kappa, dict(L=sc.pipe.L, D=sc.pipe.D, a=sc.a, H_res=sc.pipe.H_res,
                                       xiL=sc.xiL, qL_frac=sc.qL_frac, CdA=sc.leak.CdA)


def generate_dataset(n: int, seed0: int = 0, n_workers: int = 8):
    """Parallel dataset generation -- each example needs one MOC transient run (~100 ms serial),
    so this uses a process pool (independent, embarrassingly parallel scenarios). No torch import
    anywhere in this module -- see module docstring for why that matters here specifically."""
    seeds = list(range(seed0, seed0 + n))
    t0 = time.time()
    with Pool(n_workers) as pool:
        rows = pool.map(_gen_one, seeds)
    X = np.stack([r[0] for r in rows])
    xiL = np.array([r[1] for r in rows])
    log_kappa = np.array([r[2] for r in rows])
    meta = [r[3] for r in rows]
    print(f"generated {n} examples in {time.time() - t0:.1f}s ({n_workers} workers)")
    return X, xiL, log_kappa, meta
