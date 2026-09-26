"""Conventional leak-localisation baselines the PINN must be judged against.

A. time_of_flight   - classic baseline-subtraction echo timing (+ reflection-coefficient sizing)
B. moc_search       - model-based inversion: brute-force MOC simulations over (x_L, CdA), then
                      Levenberg-Marquardt polish of (CdA, a).  This is the strong conventional competitor.
"""
from __future__ import annotations
import time
import numpy as np
from scipy.optimize import least_squares
from .physics import Pipe, Leak, Constriction, design_valve, G
from .moc import MOC
from .synth import Dataset


def perturbation(data: Dataset):
    """Measured head perturbation relative to the pre-transient steady mean (removes static offsets)."""
    base = data.H_pre.mean(axis=0)
    return data.H_meas - base, base


def _sim_perturbation(data, a, N, leak, constriction=None):
    p = data.pipe
    m = MOC(p, data.valve, a, N, leak, constriction=constriction)
    r = m.run(data.t[-1] + 2 * m.dt)
    n1 = int(round(data.x_sensors[0] / m.dx))
    H = np.stack([np.interp(data.t, r["t"], r["H"][:, n1]), np.interp(data.t, r["t"], r["H"][:, N])], axis=1)
    return H - r["H"][0, [n1, N]], m


def estimate_wave_speed(data: Dataset):
    """Wave speed from the transit time of the first pulse between the two sensors:
    a = (L - x1) / (t_rise@S1 - t_rise@S2)   (50 %-of-peak crossing, linear interpolation).

    Searches the FULL recorded window, not a fixed early slice: a fixed short window silently
    fails (and falls back to the nominal default) whenever the true wave speed is much slower
    than the pipe material's typical value -- e.g. a plastic pipe, or entrained air -- which is
    exactly the "unknown wave speed" scenario this function exists to handle."""
    p, t = data.pipe, data.t
    hm, _ = perturbation(data)
    def t50(h):
        thr = 0.5 * h.max(); k = np.argmax(h > thr)
        if k == 0:   # threshold never crossed (or crossed at the very first sample) -> unusable
            return None
        return np.interp(thr, [h[k - 1], h[k]], [t[k - 1], t[k]])
    t1, t2 = t50(hm[:, 0]), t50(hm[:, 1])
    if t1 is None or t2 is None:
        return p.wave_speed()
    dt = t1 - t2
    return (p.L - data.x_sensors[0]) / dt if dt > 0.02 else p.wave_speed()


# ------------------------------------------------------------------ A
def time_of_flight(data: Dataset, a_nominal=None):
    p, t = data.pipe, data.t
    fs = 1.0 / (t[1] - t[0])
    hm, base = perturbation(data)
    # (1) wave speed from the S1->S2 transit time of the first pulse
    a_hat = estimate_wave_speed(data)
    # (2) leak-free reference at the estimated a, subtract it at the valve sensor
    ref, _ = _sim_perturbation(data, a_hat, 200, None)
    resid = hm[:, 1] - ref[:, 1]
    # (3) matched filter: echo = s * template(t - dt), template = first pulse shape
    tmpl = ref[:, 1].copy(); tmpl[t > 0.06] = tmpl[t <= 0.06][-1]      # pulse rise then plateau (no later echoes)
    dt_max = min(2 * (p.L - 4.0) / a_hat, t[-1] - t[0] - 1.0 / fs)     # never probe past the recorded window
    dts = np.arange(0.5 * 2 * 4 / a_hat, max(dt_max, 1.0 / fs), 1.0 / fs)   # candidate delays
    best = (-1, None, None)
    for dt in dts:
        k = int(round(dt * fs))
        if k <= 0 or k >= len(tmpl):
            continue
        sh = np.zeros_like(tmpl); sh[k:] = tmpl[:len(tmpl) - k]
        sh = sh - sh[0]
        # only compare before the reservoir echo (2L/a) contaminates the record
        mask = t < 2 * p.L / a_hat
        den = sh[mask] @ sh[mask]
        if den <= 0: continue
        s = (sh[mask] @ resid[mask]) / den
        gain = s * s * den                      # explained energy for a negative-going echo
        if s < 0 and gain > best[0]:
            best = (gain, dt, s)
    _, dt, s = best
    x_L = p.L - a_hat * dt / 2
    R = float(np.clip(s, -0.5, -1e-4))
    GB = -2 * R / (1 + R)
    HL = base[0] + (base[1] - base[0]) * (x_L - data.x_sensors[0]) / (p.L - data.x_sensors[0])
    CL = 2 * (GB / p.B(a_hat)) * np.sqrt(max(HL, 1.0))
    return dict(x_L=float(x_L), CdA=float(CL / np.sqrt(2 * G)), a=float(a_hat), name="time-of-flight")


# ------------------------------------------------------------------ B
def moc_search(data: Dataset, a_nominal=None, N=200, x_grid=None, cda_grid=None, refine=True, verbose=False):
    t0 = time.time()
    p = data.pipe
    a0 = a_nominal or estimate_wave_speed(data)
    hm, _ = perturbation(data)
    x_grid = np.arange(3.0, p.L - 2.0, 1.0) if x_grid is None else x_grid
    cda_grid = np.array([1.0, 2.0, 4.0, 8.0]) * 1e-6 if cda_grid is None else cda_grid

    lo, hi = 2 * (p.L / N), p.L - 2 * (p.L / N)   # keep the leak >= 2 grid cells from either end

    def cost(xL, cda, a):
        xL = float(np.clip(xL, lo, hi))
        sim, _ = _sim_perturbation(data, a, N, Leak(xL, float(cda)))
        return float(np.sum((sim - hm) ** 2))
    best = (np.inf, None, None)
    for xL in x_grid:
        for c in cda_grid:
            J = cost(xL, c, a0)
            if J < best[0]: best = (J, xL, c)
    J, xL, cda = best
    a = a0
    # LM bounds on CdA scaled to the searched grid (not a fixed [0.05,40] mm^2 -- that range was
    # sized for the original ~100 m default pipe; a wider pipe needs a much bigger CdA for the
    # same leak fraction, and a fixed bound then puts the initial guess outside the bounds,
    # crashing least_squares) -- generous 10x margin either side so the LM step can still move.
    cda_lo = float(cda_grid.min()) * 1e6 * 0.1
    cda_hi = float(cda_grid.max()) * 1e6 * 10.0
    if refine:
        for _ in range(2):                     # alternate: (CdA, a) by LM, then x_L by local scan
            f = lambda th: (_sim_perturbation(data, th[1] * 1e3, N, Leak(float(xL), float(th[0] * 1e-6)))[0] - hm).ravel()
            cda0 = float(np.clip(cda * 1e6, cda_lo, cda_hi))
            sol = least_squares(f, [cda0, a / 1e3], x_scale=[1.0, 0.1], diff_step=1e-2,
                                bounds=([cda_lo, 0.8 * a0 / 1e3], [cda_hi, 1.2 * a0 / 1e3]))
            cda, a = sol.x[0] * 1e-6, sol.x[1] * 1e3
            xs = np.clip(np.arange(xL - 3, xL + 3.01, 0.5), lo, hi)
            cs = [cost(x, cda, a) for x in xs]
            xL, J = float(xs[int(np.argmin(cs))]), float(np.min(cs))
    return dict(x_L=float(xL), CdA=float(cda), a=float(a), cost=J, name="MOC search", seconds=time.time() - t0)


# ------------------------------------------------------------------ C -- constriction (partial blockage)
def constriction_search(data: Dataset, a_nominal=None, N=200, x_grid=None, cda_grid=None, refine=True):
    """Same grid+LM structure as moc_search(), searching (x_c, CdA_c) against the constriction
    junction condition (leakpinn.physics.Constriction) instead of the leak's. See
    CONSTRICTION_DETECTION_PLAN.md for the physics. Reuses moc_search's proven approach rather
    than a new algorithm, per this project's own evidence that physics-based search beats
    amortized ML regression at this problem's data scale (see leak-localization work)."""
    t0 = time.time()
    p = data.pipe
    a0 = a_nominal or estimate_wave_speed(data)
    hm, _ = perturbation(data)
    x_grid = np.arange(3.0, p.L - 2.0, max(1.0, (p.L - 5.0) / 100.0)) if x_grid is None else x_grid
    # cda_grid scaled to THIS pipe's own bore area (0.15-0.85 open, matching domain.py's sampling
    # range) rather than a fixed absolute constant -- same fix as moc_search's leak cda_grid needed.
    # The cost surface is sharply peaked in CdA_c (confirmed directly: cost at the true params can
    # be an order of magnitude better than at the nearest points on a coarse 4-point grid), so a
    # sparse grid can leave BOTH the grid search and the LM refinement started from a point too
    # far from the true optimum to find their way there -- 8 points, not 4.
    cda_grid = np.linspace(0.15, 0.85, 8) * 0.7 * p.A if cda_grid is None else cda_grid

    lo, hi = 2 * (p.L / N), p.L - 2 * (p.L / N)

    def cost(xC, cdac, a):
        xC = float(np.clip(xC, lo, hi))
        sim, _ = _sim_perturbation(data, a, N, None, Constriction(xC, float(cdac)))
        return float(np.sum((sim - hm) ** 2))
    best = (np.inf, None, None)
    for xC in x_grid:
        for c in cda_grid:
            J = cost(xC, c, a0)
            if J < best[0]: best = (J, xC, c)
    J, xC, cdac = best
    a = a0
    cda_lo = float(cda_grid.min()) * 0.1
    cda_hi = float(cda_grid.max()) * 10.0
    if refine:
        for _ in range(2):
            f = lambda th: (_sim_perturbation(data, th[1] * 1e3, N, None, Constriction(float(xC), float(th[0])))[0] - hm).ravel()
            cdac0 = float(np.clip(cdac, cda_lo, cda_hi))
            sol = least_squares(f, [cdac0, a / 1e3], x_scale=[cda_hi, 0.1], diff_step=1e-2,
                                bounds=([cda_lo, 0.8 * a0 / 1e3], [cda_hi, 1.2 * a0 / 1e3]))
            cdac, a = sol.x[0], sol.x[1] * 1e3
            xs = np.clip(np.arange(xC - 3, xC + 3.01, 0.5), lo, hi)
            cs = [cost(x, cdac, a) for x in xs]
            xC, J = float(xs[int(np.argmin(cs))]), float(np.min(cs))
    return dict(x_C=float(xC), CdA_c=float(cdac), a=float(a), cost=J, name="constriction search", seconds=time.time() - t0)


# ------------------------------------------------------------------ D -- three-way classifier
def no_anomaly_cost(data: Dataset, a_nominal=None, N=200):
    """Cost of the "nothing is wrong" hypothesis -- the third option classify_and_localize()
    weighs against a leak and a constriction.

    Refines `a` by a 1-D search here too (not just the fixed a0 estimate) -- leaving this out
    was a real bug: moc_search/constriction_search both get to fit a BETTER wave speed during
    their LM refinement, correcting for any error in the initial estimate_wave_speed() guess,
    while a "none" hypothesis stuck with only the raw estimate looks artificially worse purely
    from timing mismatch, not because there's actually an anomaly. Confirmed directly: without
    this fix, a genuinely anomaly-free case was misclassified as "leak" because the unrefined
    "none" cost was inflated well above the noise floor that leak/constriction could reach by
    also adjusting `a`."""
    a0 = a_nominal or estimate_wave_speed(data)
    hm, _ = perturbation(data)

    def cost(a):
        sim, _ = _sim_perturbation(data, a, N, None, None)
        return float(np.sum((sim - hm) ** 2))
    from scipy.optimize import minimize_scalar
    sol = minimize_scalar(cost, bounds=(0.8 * a0, 1.2 * a0), method="bounded",
                          options=dict(xatol=1.0))
    return float(sol.fun)


def classify_and_localize(data: Dataset, a_nominal=None, N=200, verbose=False, alpha=0.01, dof_extra=2):
    """Runs all three hypotheses (no anomaly / leak / constriction) against the measured data and
    reports whichever best explains it -- a model-selection problem, not three independent
    searches run in isolation. This is the actual "detector": given real sensor data with an
    UNKNOWN failure mode (if any), this is what decides which one it most likely is.

    Picking argmin(costs) directly is WRONG and was confirmed to misclassify a genuinely
    anomaly-free case: leak/constriction each have `dof_extra` more free parameters (position,
    size) than "none", so they can always fit sensor NOISE at least as well, and typically better,
    even when there's truly nothing there -- with a fair wave-speed refinement for all three (see
    no_anomaly_cost's fix above), a real no-anomaly case showed all three costs within ~0.1% of
    each other, sitting right at the noise floor, with argmin picking whichever lost by chance.
    Instead, require a candidate to beat "none" by more than the cost reduction a `dof_extra`
    parameter model is expected to achieve on pure noise ALONE at the `alpha` significance level
    (a chi-squared threshold, `noise_std^2 * chi2.ppf(1-alpha, df=dof_extra)`) before accepting it
    over "none" -- only then compare leak vs. constriction against each other."""
    t0 = time.time()
    p = data.pipe
    a0 = a_nominal or estimate_wave_speed(data)
    J_none = no_anomaly_cost(data, a0, N)
    # moc_search's OWN default cda_grid ([1,2,4,8] mm^2) is sized for the original ~100 m pipe --
    # unlike constriction_search's default (already scaled to p.A internally), calling moc_search
    # here without an explicit grid inherits that mismatch across this project's wider pipe range.
    # Confirmed directly: without this fix, classification accuracy was fine (95.8%, leak vs.
    # constriction never confused) but localization had large outliers (MAE 41 m despite a 1.6 m
    # median) on exactly the pipes where the default grid didn't cover the true CdA.
    x_step = max(1.0, (p.L - 5.0) / 100.0)
    x_grid = np.arange(3.0, p.L - 2.0, x_step)
    cda_grid = np.geomspace(0.005, 0.10, 8) * p.Q_design / np.sqrt(2 * G * p.H_res)
    leak_res = moc_search(data, a0, N, x_grid=x_grid, cda_grid=cda_grid)
    cons_res = constriction_search(data, a0, N)
    costs = dict(none=J_none, leak=leak_res["cost"], constriction=cons_res["cost"])

    from scipy.stats import chi2
    noise_std = data.meta.get("noise_std", 0.10)
    sig_threshold = noise_std ** 2 * chi2.ppf(1 - alpha, df=dof_extra)
    candidates = {k: v for k, v in costs.items() if k == "none" or (J_none - v) > sig_threshold}
    kind = min(candidates, key=candidates.get)
    out = dict(kind=kind, costs=costs, leak=leak_res, constriction=cons_res,
              seconds=time.time() - t0)
    if verbose:
        print(f"  classify: none={J_none:.3e} leak={leak_res['cost']:.3e} "
              f"constriction={cons_res['cost']:.3e} -> {kind}")
    return out
