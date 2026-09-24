"""Scoring against the (never-shown-to-the-model) MOC truth."""
from __future__ import annotations
import numpy as np
from .pinn import predict_fields, FitResult


def field_errors(res: FitResult, data):
    tr = data.truth
    Hp, Qp = predict_fields(res, tr["x_grid"], data.t)          # (nx, nt)
    Ht, Qt = tr["H_field"].T, tr["Q_field"].T
    H0, Q0 = tr["H0_profile"][:, None], tr["Q0_profile"][:, None]
    sH, sQ = np.abs(Ht - H0).max(), np.abs(Qt - Q0).max()
    mid = len(tr["x_grid"]) // 2
    out = dict(
        H_nrmse=float(np.sqrt(np.mean((Hp - Ht) ** 2)) / sH),                # normalised by transient amplitude
        Q_nrmse=float(np.sqrt(np.mean((Qp - Qt) ** 2)) / sQ),
        Q_rmse_Ls=float(np.sqrt(np.mean((Qp - Qt) ** 2)) * 1e3),             # L/s
        Q_rel_err_pct=float(100 * np.sqrt(np.mean((Qp - Qt) ** 2)) / np.abs(Q0).mean()),   # vs steady flow
        Q_mid_peak_err_Ls=float(1e3 * abs((Qp[mid] - Q0[mid]).min() - (Qt[mid] - Q0[mid]).min())),
        H_sensor_rmse_m=float(np.sqrt(np.mean((Hp[[10, -1]] - Ht[[10, -1]]) ** 2))),
    )
    return out, (Hp, Qp)


def param_errors(est: dict, data):
    tr = data.truth
    return dict(x_err_m=float(est["x_L"] - tr["leak_x"]), CdA_rel_err_pct=float(100 * (est["CdA"] / tr["CdA"] - 1)),
                a_rel_err_pct=float(100 * (est["a"] / tr["a"] - 1)))
