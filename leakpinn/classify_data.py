"""Pure NumPy/SciPy half of the none/leak/constriction CLASSIFIER (leakpinn/classify_net.py):
scenario sampling, MOC simulation, and the feature-vector contract shared between training-data
generation and real inference. No torch/deepxde import (same fork-safety reason as
leakpinn/localize_data.py -- see that module's docstring).

Reuses `features_from_dataset` from leakpinn/localize_data.py unchanged: it already only reads
measured sensor traces + known pipe/valve constants, nothing anomaly-specific, so it works
identically whether the underlying scenario has no anomaly, a leak, or a constriction.
"""
from __future__ import annotations
from multiprocessing import Pool
import numpy as np

from .physics import G
from .domain import AnomalyScenario, sample_anomaly_scenario, TAU_END
from .synth import make_dataset, Dataset
from .localize_data import FEATURE_DIM, N_SAMP, features_from_dataset  # noqa: F401  (re-exported)

KIND_TO_LABEL = {"none": 0, "leak": 1, "constriction": 2}
LABEL_TO_KIND = {v: k for k, v in KIND_TO_LABEL.items()}


def build_example(sc: AnomalyScenario, seed: int, N_truth: int = 300):
    """One labelled example: simulate a scenario's sensor data, then build the feature vector.
    Returns (feat[206], label(0/1/2), xi_frac, log_sev) where xi_frac/log_sev are 0.0 for
    kind=='none' (never read by the training loop -- see the masked loss in classify_net.py).

    `log_sev` uses the SAME dimensionless-admittance formula for both anomaly types --
    log(CdA_eff * B_nom * sqrt(2g)) -- since both a leak orifice and a constriction throat are
    effective orifice areas scaled by the same pipe impedance B_nom; this keeps one regression
    head's target on one consistent scale across the two anomaly kinds (the head is still only
    trained/read for the matching class, see the masked loss)."""
    p, v, a, kind = sc.pipe, sc.valve, sc.a, sc.kind
    T = TAU_END * p.L / a
    B_nom = a / (G * p.A)
    if kind == "leak":
        d = make_dataset(pipe=p, leak_x=sc.leak.x, CdA=sc.leak.CdA, a_true=a, T=T, N_truth=N_truth,
                         dtau=v.dtau, t_close=v.t_close, seed=seed)
        xi_frac = float(sc.leak.x / p.L)
        log_sev = float(np.log(sc.leak.CdA * B_nom * np.sqrt(2 * G)))
    elif kind == "constriction":
        d = make_dataset(pipe=p, leak_x=0.0, CdA=None, a_true=a, T=T, N_truth=N_truth,
                         dtau=v.dtau, t_close=v.t_close, seed=seed, constriction=sc.constriction)
        xi_frac = float(sc.constriction.x / p.L)
        log_sev = float(np.log(sc.constriction.CdA_c * B_nom * np.sqrt(2 * G)))
    else:  # "none"
        d = make_dataset(pipe=p, leak_x=0.0, CdA=None, a_true=a, T=T, N_truth=N_truth,
                         dtau=v.dtau, t_close=v.t_close, seed=seed)
        xi_frac = 0.0
        log_sev = 0.0
    feat, extras = features_from_dataset(d, a)
    return feat, KIND_TO_LABEL[kind], xi_frac, log_sev


def _gen_one(seed: int):
    rng = np.random.default_rng(seed)
    sc = sample_anomaly_scenario(rng)
    feat, label, xi_frac, log_sev = build_example(sc, seed=seed)
    return feat, label, xi_frac, log_sev, dict(L=sc.pipe.L, D=sc.pipe.D, a=sc.a, H_res=sc.pipe.H_res,
                                                kind=sc.kind)


def generate_dataset(n: int, seed0: int = 0, n_workers: int = 8):
    """Returns (X[n,206], label[n] int, xi_frac[n], log_sev[n], metas[n])."""
    seeds = list(range(seed0, seed0 + n))
    if n_workers > 1:
        with Pool(n_workers) as pool:
            rows = pool.map(_gen_one, seeds)
    else:
        rows = [_gen_one(s) for s in seeds]
    X = np.stack([r[0] for r in rows])
    label = np.array([r[1] for r in rows], dtype=np.int64)
    xi_frac = np.array([r[2] for r in rows], dtype=np.float64)
    log_sev = np.array([r[3] for r in rows], dtype=np.float64)
    metas = [r[4] for r in rows]
    return X, label, xi_frac, log_sev, metas
