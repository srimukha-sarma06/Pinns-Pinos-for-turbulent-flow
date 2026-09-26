"""Leak LOCALIZATION (inverse problem): given two sensor pressure traces from an unknown-leak
pipe, infer where the leak is and how big it is. See INVERSE_LOCALIZATION_PLAN.md for the full
methodology and literature grounding. Two stages:

  Stage A (`LocalizerNet` + `fit_localizer`) -- an amortized regression network, trained on
    thousands of MOC-simulated (sensor trace -> true xiL, CdA) examples (leakpinn/localize_data.py
    generates these). No PDE residual, no per-case optimisation: ordinary supervised learning,
    because the single-pipe inverse PINN in leakpinn/pinn.py already showed gradient descent from
    a blind starting guess on this problem is unreliable (weak, noisy gradient from a narrow leak
    term -- see the plan doc). This is the primary, robust mechanism.

  Stage B (`refine`) -- a short physics-informed polish: freeze the already-trained generalized
    forward model (leakpinn.general_pinn.GeneralNet), warm-start (xiL, kappa) from Stage A's
    output, and take a handful of gradient steps against the actual measured trace. Same
    gradient-descent-on-physical-unknowns mechanism leakpinn/pinn.py already uses, just no longer
    asked to also find the right neighbourhood from a cold start.

Deployment constraint (same as leakpinn/general_pinn.py): every op in the EXPORTED Stage A network
is one of Gemm/MatMul, Tanh, Sigmoid, Exp, Concat, Slice -- all confirmed present in TFLM's kernel
registry. No erf, no clamp. Stage B is not exported at all: it is a short gradient loop around the
already-exported forward model, run on a gateway/PC (see the plan doc's deployment-target note),
not a new set of on-device layers.

NOTE: this module imports torch/deepxde (via leakpinn.pinn/general_pinn) -- dataset generation
lives in leakpinn/localize_data.py specifically so that its multiprocessing pool never forks a
process with those already imported (see that module's docstring).
"""
from __future__ import annotations
import time
from dataclasses import dataclass
import numpy as np
import torch

from .pinn import DEVICE
from .synth import Dataset
from .general_pinn import GeneralNet
from .localize_data import FEATURE_DIM, TAU_WINDOW, features_from_dataset

DTYPE = torch.float64


# ------------------------------------------------------------------------------- Stage A network
class LocalizerNet(torch.nn.Module):
    """Amortized regression: sensor traces + known pipe context -> (xiL, kappa). Plain MLP only
    (Gemm/Tanh/Sigmoid/Exp) -- deliberately NOT a 1D-CNN or RNN, even though the localization
    literature often uses those (see INVERSE_LOCALIZATION_PLAN.md sec 3): every op family here was
    already verified against the TFLM kernel registry in this project (general_pinn.py), whereas
    Conv/RNN kernel support would need separate re-verification. The traces are always aligned to
    a fixed t=0 (valve-closure start), so shift-invariance -- the usual reason to prefer a
    convolutional/recurrent encoder -- isn't needed here, removing the accuracy argument for
    taking on that extra op-support risk.

    Input: shape [batch, 206] -- see FEATURE_DIM / `leakpinn.localize_data.features_from_dataset`
    for the exact column layout.
    Output: shape [batch, 2] -- [xiL, kappa], already physical (xiL in (0.03, 0.97) of L, kappa > 0).
    """
    def __init__(self, in_dim: int = FEATURE_DIM, width: int = 128, depth: int = 4, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.hidden = torch.nn.ModuleList(
            [torch.nn.Linear(in_dim, width)] + [torch.nn.Linear(width, width) for _ in range(depth - 1)])
        self.out = torch.nn.Linear(width, 2)
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)
        self.to(DTYPE)

    def _hidden(self, x):
        h = x
        for lin in self.hidden:
            h = torch.tanh(lin(h))
        return h

    def forward(self, x):
        y = self.out(self._hidden(x))
        xiL = 0.03 + 0.94 * torch.sigmoid(y[:, 0:1])
        kappa = torch.exp(y[:, 1:2])
        return torch.cat([xiL, kappa], dim=1)


class DeployLocalizer(torch.nn.Module):
    """Export wrapper: normalizes the raw feature vector (fixed mean/std from training, baked in
    as constants -- ordinary Sub/Mul, TFLM-safe) then runs LocalizerNet. This is what gets
    exported to ONNX/TFLite, so the deployed model takes RAW features directly, no separate
    host-side normalization step to keep in sync."""
    def __init__(self, net: LocalizerNet, x_mean: np.ndarray, x_std: np.ndarray):
        super().__init__()
        self.net = net
        self.register_buffer("x_mean", torch.tensor(x_mean, dtype=DTYPE))
        self.register_buffer("x_std", torch.tensor(x_std, dtype=DTYPE))

    def forward(self, x):
        return self.net((x - self.x_mean) / self.x_std)


# ------------------------------------------------------------------------------- training
@dataclass
class LocalizeFitResult:
    net: LocalizerNet
    x_mean: np.ndarray
    x_std: np.ndarray
    history: list
    seconds: float


def fit_localizer(X, xiL, log_kappa, width=128, depth=4, iters=4000, lr=1e-3, batch=256,
                  val_frac=0.1, seed=0, verbose=True) -> LocalizeFitResult:
    t0 = time.time()
    n = len(X)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(n * val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    x_mean, x_std = X[tr_idx].mean(0), X[tr_idx].std(0) + 1e-8
    Xn = (X - x_mean) / x_std
    y = np.stack([xiL, log_kappa], axis=1)

    Xt = torch.as_tensor(Xn, dtype=DTYPE, device=DEVICE)
    yt = torch.as_tensor(y, dtype=DTYPE, device=DEVICE)
    tr_idx_t = torch.as_tensor(tr_idx, device=DEVICE)
    val_idx_t = torch.as_tensor(val_idx, device=DEVICE)

    net = LocalizerNet(X.shape[1], width, depth, seed).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(iters // 4, 1), gamma=0.5)
    def _loss(x, y):
        # compare against net's ACTUAL forward-pass output (sigmoid-bounded xiL, exp'd kappa),
        # not the raw pre-transform logits -- training the raw logit directly against a bounded
        # target and then sigmoid-transforming it again at inference silently miscalibrates the
        # whole prediction (this was a real bug here, caught by a narrow-range sanity check: MAE
        # stayed ~18% even on a near-trivial single-pipe-like range, which a correct pipeline
        # should nail).
        out = net(x)                                    # [xiL, kappa], already transformed
        return torch.mean((out[:, 0] - y[:, 0]) ** 2) + torch.mean((torch.log(out[:, 1]) - y[:, 1]) ** 2)

    history = []
    for it in range(iters):
        batch_idx = tr_idx_t[torch.randint(0, len(tr_idx_t), (min(batch, len(tr_idx_t)),), device=DEVICE)]
        xb, yb = Xt[batch_idx], yt[batch_idx]
        loss = _loss(xb, yb)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % max(iters // 20, 1) == 0 or it == iters - 1:
            with torch.no_grad():
                vloss = float(_loss(Xt[val_idx_t], yt[val_idx_t]))
            history.append(dict(it=it, train_loss=float(loss.detach()), val_loss=vloss, t=time.time() - t0))
            if verbose:
                print(f"  [{it:5d}] train={float(loss.detach()):.4e}  val={vloss:.4e}")
    return LocalizeFitResult(net, x_mean, x_std, history, time.time() - t0)


def predict(res: LocalizeFitResult, X: np.ndarray) -> np.ndarray:
    """Returns (xiL, kappa) for each row of raw (un-normalized) features X."""
    Xn = (X - res.x_mean) / res.x_std
    with torch.no_grad():
        y = res.net(torch.as_tensor(Xn, dtype=DTYPE, device=DEVICE)).cpu().numpy()
    return y  # columns: xiL, kappa


# ------------------------------------------------------------------------------- Stage B refinement
def refine(gen_net: GeneralNet, d: Dataset, a_est: float, xiL0: float, kappa0: float,
          steps: int = 60, lr: float = 3e-2, Hs: float = 10.0,
          trust_xiL: float = 0.08, trust_log_kappa: float = 0.8):
    """Physics-informed local polish: freeze `gen_net` (the trained generalized FORWARD model),
    treat (xiL, kappa) as the only free variables, and gradient-descend a data-misfit loss against
    the actual measured sensor trace -- warm-started from Stage A's (xiL0, kappa0) instead of a
    blind guess, which is the fix for the failure mode leakpinn/pinn.py's inverse PINN documented
    (weak/noisy gradient from a cold start on a non-convex landscape).

    Bounded to a TRUST REGION around (xiL0, kappa0) (+/- `trust_xiL` in xiL, a factor of
    exp(trust_log_kappa) in kappa) rather than the full (0.03, 0.97) range -- found empirically to
    be necessary, not optional: `gen_net` itself has ~21% field NRMSE (see README.md), so its own
    loss surface over (xiL, kappa) is NOT always minimised at the true values for a given pipe.
    An earlier unconstrained version of this function correctly, smoothly minimised the surrogate
    loss on every test case, while several of those cases walked the estimate 50+ m away from a
    much better Stage-A starting point -- "correctly descending a biased surrogate" is worse than
    doing nothing on those cases. Bounding to a local neighbourhood keeps this a polish instead of
    a second, less-reliable search."""
    p = d.pipe
    _, extras = features_from_dataset(d, a_est)
    phi = extras["phi"]; H1_0 = extras["H1_0"]; H2_0 = extras["H2_0"]; xi1 = extras["xi1"]
    base = d.H_pre.mean(axis=0)
    hm = (d.H_meas - base) / Hs                              # measured, dimensionless
    tau_i = d.t * a_est / p.L
    tau_col = torch.as_tensor(tau_i[:, None], dtype=DTYPE, device=DEVICE)
    xi_s1 = torch.full_like(tau_col, xi1)
    xi_s2 = torch.ones_like(tau_col)
    phi_col = torch.full_like(tau_col, phi)
    H1n_col = torch.full_like(tau_col, H1_0 / 100.0)
    H2n_col = torch.full_like(tau_col, H2_0 / 100.0)
    target = (torch.as_tensor(hm[:, 0:1], dtype=DTYPE, device=DEVICE),
             torch.as_tensor(hm[:, 1:2], dtype=DTYPE, device=DEVICE))

    xiL0_c = float(np.clip(xiL0, 0.03, 0.97))
    log_kappa0 = float(np.log(max(kappa0, 1e-6)))
    xiL_raw = torch.tensor(0.0, requires_grad=True, device=DEVICE, dtype=DTYPE)     # tanh(0)=0 -> starts exactly at xiL0
    logk_raw = torch.tensor(0.0, requires_grad=True, device=DEVICE, dtype=DTYPE)
    opt = torch.optim.Adam([xiL_raw, logk_raw], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        xiL_step = torch.clamp(xiL0_c + trust_xiL * torch.tanh(xiL_raw), 0.03, 0.97)
        logk_step = log_kappa0 + trust_log_kappa * torch.tanh(logk_raw)
        xiL_t = xiL_step.expand_as(tau_col)
        kappa_t = torch.exp(logk_step).expand_as(tau_col)
        y1 = gen_net(xi_s1, tau_col, xiL_t, kappa_t, phi_col, H1n_col, H2n_col)
        y2 = gen_net(xi_s2, tau_col, xiL_t, kappa_t, phi_col, H1n_col, H2n_col)
        loss = torch.mean((y1[:, 0:1] - target[0]) ** 2) + torch.mean((y2[:, 0:1] - target[1]) ** 2)
        loss.backward()
        opt.step()
    with torch.no_grad():
        xiL_f = float(torch.clamp(xiL0_c + trust_xiL * torch.tanh(xiL_raw), 0.03, 0.97))
        kappa_f = float(torch.exp(log_kappa0 + trust_log_kappa * torch.tanh(logk_raw)))
    return xiL_f, kappa_f
