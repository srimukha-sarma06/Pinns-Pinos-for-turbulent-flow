"""Conventional leak-localisation baselines the PINN must be judged against.

A. time_of_flight   - classic baseline-subtraction echo timing (+ reflection-coefficient sizing)
B. moc_search       - model-based inversion: brute-force MOC simulations over (x_L, CdA), then
                      Levenberg-Marquardt polish of (CdA, a).  This is the strong conventional competitor.
"""
from __future__ import annotations
import time
import numpy as np
from scipy.optimize import least_squares
from .physics import Pipe, Leak, design_valve, G
from .moc import MOC
from .synth import Dataset


def perturbation(data: Dataset):
    """Measured head perturbation relative to the pre-transient steady mean (removes static offsets)."""
    base = data.H_pre.mean(axis=0)
    return data.H_meas - base, base


def _sim_perturbation(data, a, N, leak):
    p = data.pipe
    m = MOC(p, data.valve, a, N, leak)
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
    if refine:
        for _ in range(2):                     # alternate: (CdA, a) by LM, then x_L by local scan
            f = lambda th: (_sim_perturbation(data, th[1] * 1e3, N, Leak(float(xL), float(th[0] * 1e-6)))[0] - hm).ravel()
            sol = least_squares(f, [cda * 1e6, a / 1e3], x_scale=[1.0, 0.1], diff_step=1e-2,
                                bounds=([0.05, 0.8 * a0 / 1e3], [40.0, 1.2 * a0 / 1e3]))
            cda, a = sol.x[0] * 1e-6, sol.x[1] * 1e3
            xs = np.clip(np.arange(xL - 3, xL + 3.01, 0.5), lo, hi)
            cs = [cost(x, cda, a) for x in xs]
            xL, J = float(xs[int(np.argmin(cs))]), float(np.min(cs))
    return dict(x_L=float(xL), CdA=float(cda), a=float(a), cost=J, name="MOC search", seconds=time.time() - t0)
