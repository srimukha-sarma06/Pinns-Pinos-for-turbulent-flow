"""Amortized none/leak/constriction CLASSIFIER + localizer: a trainable-and-exportable surrogate
for the physics-search `classify_and_localize` in leakpinn/baselines.py.

This is intentionally a SEPARATE code path from baselines.classify_and_localize, not a
replacement -- CONSTRICTION_DETECTION_PLAN.md's physics-search detector remains the accuracy
reference (95.8% on the 24-case validation set) and is what this network is trained AND graded
against. The physics search itself has no weights to export; this module exists only because a
one-shot MLP forward pass is what can actually run as TFLite/TFLM on the RA8P1, whereas the
physics search is a ~900-simulation grid+LM optimisation loop (see the plan doc's edge-feasibility
estimate) better suited to a full C port of the MOC solver, not an NN export.

Deployment constraint (same discipline as leakpinn/localize.py): every op in the exported graph is
one of Gemm/MatMul, Tanh, Sigmoid, Exp, Softmax, Concat, Slice -- all confirmed in TFLM's kernel
registry by scripts/09 and scripts/12's actual-.tflite op-list checks.
"""
from __future__ import annotations
import time
from dataclasses import dataclass
import numpy as np
import torch

from .localize_data import FEATURE_DIM

DTYPE = torch.float64
DEVICE = "cpu"   # small MLP, no PDE residual -- CPU is plenty and keeps this module torch-only,
                  # no CUDA-fork hazard to manage (unlike leakpinn/pinn.py's forward-model training)


class ClassifierNet(torch.nn.Module):
    """Shared MLP trunk -> 3-way class softmax + a shared position head + two class-conditional
    severity heads (leak, constriction; only the matching one is trained/read for a given row --
    see the masked loss in fit_classifier). Plain MLP only, same op-family reasoning as
    leakpinn.localize.LocalizerNet.

    Input: shape [batch, 206] -- leakpinn.localize_data.features_from_dataset layout.
    Output: shape [batch, 6] -- [p_none, p_leak, p_constriction, xi_frac, kappa_leak, kappa_c].
      xi_frac in (0.03, 0.97) of L; kappa_leak/kappa_c > 0; only meaningful for the argmax class.
    """
    def __init__(self, in_dim: int = FEATURE_DIM, width: int = 96, depth: int = 3, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.hidden = torch.nn.ModuleList(
            [torch.nn.Linear(in_dim, width)] + [torch.nn.Linear(width, width) for _ in range(depth - 1)])
        self.cls_out = torch.nn.Linear(width, 3)
        self.xi_out = torch.nn.Linear(width, 1)
        self.sev_out = torch.nn.Linear(width, 2)   # [leak, constriction]
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
        h = self._hidden(x)
        p_cls = torch.softmax(self.cls_out(h), dim=-1)
        xi = 0.03 + 0.94 * torch.sigmoid(self.xi_out(h))
        sev = torch.exp(self.sev_out(h))
        return torch.cat([p_cls, xi, sev], dim=1)


class DeployClassifier(torch.nn.Module):
    """Export wrapper: bakes the fixed train-set normalization (mean/std) into the graph as
    constants, same pattern as leakpinn.localize.DeployLocalizer, so the deployed model takes
    RAW features directly."""
    def __init__(self, net: ClassifierNet, x_mean: np.ndarray, x_std: np.ndarray):
        super().__init__()
        self.net = net
        self.register_buffer("x_mean", torch.tensor(x_mean, dtype=DTYPE))
        self.register_buffer("x_std", torch.tensor(x_std, dtype=DTYPE))

    def forward(self, x):
        return self.net((x - self.x_mean) / self.x_std)


@dataclass
class ClassifyFitResult:
    net: ClassifierNet
    x_mean: np.ndarray
    x_std: np.ndarray
    history: list
    seconds: float


def fit_classifier(X, label, xi_frac, log_sev, width=96, depth=3, iters=6000, lr=1e-3, batch=256,
                   val_frac=0.1, seed=0, verbose=True) -> ClassifyFitResult:
    t0 = time.time()
    n = len(X)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(n * val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    x_mean, x_std = X[tr_idx].mean(0), X[tr_idx].std(0) + 1e-8
    Xn = (X - x_mean) / x_std

    Xt = torch.as_tensor(Xn, dtype=DTYPE, device=DEVICE)
    labelt = torch.as_tensor(label, dtype=torch.long, device=DEVICE)
    xit = torch.as_tensor(xi_frac, dtype=DTYPE, device=DEVICE)
    sevt = torch.as_tensor(log_sev, dtype=DTYPE, device=DEVICE)
    tr_idx_t = torch.as_tensor(tr_idx, device=DEVICE)
    val_idx_t = torch.as_tensor(val_idx, device=DEVICE)

    net = ClassifierNet(X.shape[1], width, depth, seed).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(iters // 4, 1), gamma=0.5)

    def _loss(idx):
        xb, lb, xib, sb = Xt[idx], labelt[idx], xit[idx], sevt[idx]
        out = net(xb)                                   # [p_none,p_leak,p_c, xi, kappa_l, kappa_c]
        ce = torch.nn.functional.nll_loss(torch.log(out[:, 0:3] + 1e-12), lb)
        is_leak = (lb == 1); is_con = (lb == 2); is_anom = is_leak | is_con
        xi_loss = torch.tensor(0.0, dtype=DTYPE)
        if is_anom.any():
            xi_loss = torch.mean((out[is_anom, 3] - xib[is_anom]) ** 2)
        sev_loss = torch.tensor(0.0, dtype=DTYPE)
        if is_leak.any():
            sev_loss = sev_loss + torch.mean((torch.log(out[is_leak, 4]) - sb[is_leak]) ** 2)
        if is_con.any():
            sev_loss = sev_loss + torch.mean((torch.log(out[is_con, 5]) - sb[is_con]) ** 2)
        return ce + xi_loss + sev_loss, ce, xi_loss, sev_loss

    history = []
    for it in range(iters):
        batch_idx = tr_idx_t[torch.randint(0, len(tr_idx_t), (min(batch, len(tr_idx_t)),), device=DEVICE)]
        loss, ce, xi_loss, sev_loss = _loss(batch_idx)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % max(iters // 20, 1) == 0 or it == iters - 1:
            with torch.no_grad():
                vloss, vce, vxi, vsev = _loss(val_idx_t)
                pred = net(Xt[val_idx_t])[:, 0:3].argmax(dim=1)
                acc = float((pred == labelt[val_idx_t]).float().mean())
            history.append(dict(it=it, train_loss=float(loss.detach()), val_loss=float(vloss),
                                val_acc=acc, t=time.time() - t0))
            if verbose:
                print(f"  [{it:5d}] train={float(loss.detach()):.4e}  val={float(vloss):.4e}"
                     f"  val_acc={acc:.3f}  (ce={float(vce):.3e} xi={float(vxi):.3e} sev={float(vsev):.3e})")
    return ClassifyFitResult(net, x_mean, x_std, history, time.time() - t0)


def predict(res: ClassifyFitResult, X: np.ndarray) -> np.ndarray:
    """Returns [p_none,p_leak,p_constriction, xi_frac, kappa_leak, kappa_c] per row of raw X."""
    Xn = (X - res.x_mean) / res.x_std
    with torch.no_grad():
        y = res.net(torch.as_tensor(Xn, dtype=DTYPE, device=DEVICE)).cpu().numpy()
    return y
