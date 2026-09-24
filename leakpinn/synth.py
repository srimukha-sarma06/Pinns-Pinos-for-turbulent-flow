"""Synthetic 'field' data: MOC ground truth (fine grid) -> realistic sensor records.

Sensor model (industrial dynamic pressure transmitter, 0-10 bar gauge):
  * 1 kHz sampling (typical for fast-transient logging),
  * additive white noise, default sigma = 0.10 m of water (~1 kPa, ~0.1 % of full scale),
  * 12-bit ADC quantisation of the 0-10 bar span (LSB ~ 2.5 cm of water),
  * a 100 ms pre-transient window of steady readings (used to remove static offsets).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .physics import Pipe, Leak, design_valve
from .moc import MOC

FS_SPAN_M = 10.0e5 / (998.2 * 9.80665)   # 10 bar in metres of water  (~102 m)


@dataclass
class Dataset:
    pipe: Pipe
    valve: object
    t: np.ndarray             # s, transient window (t>=0), sensor sample times
    t_pre: np.ndarray         # s, pre-transient window (t<0)
    x_sensors: np.ndarray     # m  [x1, L]
    H_meas: np.ndarray        # (nt, 2) noisy, quantised heads during transient
    H_pre: np.ndarray         # (npre, 2) noisy steady heads before the transient
    truth: dict = field(default_factory=dict)   # leak params + full fields, for scoring only
    meta: dict = field(default_factory=dict)


def make_dataset(pipe: Pipe | None = None, leak_x=37.3, CdA=3.6e-6, a_true=None,
                 noise_std=0.10, fs=1000.0, T=0.30, t_pre=0.10, x1=10.0,
                 N_truth=400, seed=0, dtau=0.20, t_close=0.020, quantise=True) -> Dataset:
    pipe = pipe or Pipe()
    a_true = a_true or pipe.wave_speed()
    valve = design_valve(pipe, dtau=dtau, t_close=t_close)
    leak = None if CdA is None or CdA <= 0 else Leak(leak_x, CdA)
    m = MOC(pipe, valve, a_true, N_truth, leak)
    res = m.run(T)
    tm, Hm = res["t"], res["H"]
    n1 = int(round(x1 / m.dx)); n2 = N_truth
    t = np.arange(0.0, T, 1.0 / fs)
    tp = -np.arange(int(t_pre * fs), 0, -1) / fs
    H_true = np.stack([np.interp(t, tm, Hm[:, n1]), np.interp(t, tm, Hm[:, n2])], axis=1)
    H0 = Hm[0, [n1, n2]]
    H_pre_true = np.tile(H0, (len(tp), 1))
    rng = np.random.default_rng(seed)
    def sense(Hx):
        y = Hx + rng.normal(0.0, noise_std, Hx.shape)
        if quantise:
            lsb = FS_SPAN_M / 4096.0
            y = np.round(y / lsb) * lsb
        return y
    H_meas, H_pre = sense(H_true), sense(H_pre_true)
    # truth fields on a 1 m x 1 ms grid (for scoring reconstructions)
    xi = np.arange(0, N_truth + 1, N_truth // int(pipe.L)) * m.dx
    idx = np.arange(0, N_truth + 1, N_truth // int(pipe.L))
    Hf = np.stack([np.interp(t, tm, Hm[:, i]) for i in idx], axis=1)
    Qf = np.stack([np.interp(t, tm, res["Q"][:, i]) for i in idx], axis=1)
    truth = dict(leak_x=(m.x_leak_actual if leak else np.nan), CdA=(CdA if leak else 0.0), a=a_true,
                 x_grid=xi, H_field=Hf, Q_field=Qf, H0_profile=Hm[0, idx], Q0_profile=res["Q"][0, idx],
                 H_true_sensors=H_true)
    meta = dict(noise_std=noise_std, fs=fs, T=T, x1=x1, N_truth=N_truth, seed=seed)
    return Dataset(pipe, valve, t, tp, np.array([x1, pipe.L]), H_meas, H_pre, truth, meta)
