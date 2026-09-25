"""Generalized forward PINN: ONE network trained across a *distribution* of pipes
(leakpinn/domain.py), instead of leakpinn/pinn.py's single fixed pipe. Given the leak's
location/size and the pipe's own dimensionless numbers as extra inputs, it predicts the
same (h, q) perturbation field as leakpinn/pinn.py -- so, exactly like today's deployed
model, it still answers "what's the pressure/flow here, given a known leak", just for
any pipe in the trained range rather than one specific one.

Why a plain PyTorch loop instead of DeepXDE's Model/data.PDE (which the rest of this repo
uses): DeepXDE's PDE/TimePDE classes assume training points live on a `dde.geometry`
object of a fixed, small dimension (here: xi, tau), and filter boundary points by calling
that geometry's own `on_boundary(x)` on the *whole* point array. Our points additionally
carry ~8 per-scenario conditioning numbers (leak position/size, friction number, reference
heads, ...) that vary from row to row -- stacking those onto the geometry's array breaks
that dimension bookkeeping (confirmed empirically: `geom.dim` must match every points
array or DeepXDE's internal `np.vstack` calls raise). Computing the same physics residuals
directly with `torch.autograd.grad` (what `dde.grad.jacobian` does internally anyway) sidesteps
that entirely and is easier to reason about for a genuinely custom sampling scheme.

Deployment constraint (Renesas / TFLite Micro, CPU-only): every op that ends up in the
*exported* graph (leakpinn.general_pinn.DeployNet, see scripts/08_export_general_onnx.py)
is one of Gemm/MatMul, Tanh, Sin, Cos, Mul, Sub, Add, Div, Sqrt, Log, Relu, Concat, Slice --
all confirmed present in tensorflow/lite/micro/kernels/micro_ops.h (see 05_export_onnx.py's
TFLM_SUPPORTED list, reused by 08_export_general_onnx.py). In particular:
  * `erf` (used by leakpinn/pinn.py's single-pipe leak-smoothing step, harmless there since
    it was pure Python-side/host post-processing against a FIXED leak position) is NOT a
    TFLM op and is never used here -- `smoothstep()` below is a Tanh-based replacement with
    a matched slope at the transition, used identically in training and in the export path.
  * `torch.clamp` has no native TFLM kernel either (`Clip` is not in the kernel registry) --
    `floor_()` below reproduces a one-sided clamp with `Relu(x - m) + m`, again used in both
    training and the export path so the two stay numerically consistent.
"""
from __future__ import annotations
import time
from dataclasses import dataclass
import numpy as np
import torch

from .pinn import _FF, DEVICE                      # reuse the validated Fourier-feature block + device policy
from .domain import Scenario, sample_scenario, scenario_consts, TAU_END, EPS_EDGE

K_STEP = 0.626 * EPS_EDGE   # tanh width matched to the erf-step's slope at xi=xiL (see smoothstep())
DTYPE = torch.float64


# ------------------------------------------------------------------------------- shared physics helpers
def floor_(x, m=1.0):
    """clamp(x, min=m) using only Relu/Sub/Add -- torch.clamp has no native TFLM kernel."""
    return torch.relu(x - m) + m


def smoothstep(xi, xiL):
    """Tanh-based replacement for 0.5*(1+erf((xi-xiL)/(sqrt2 eps))): erf is not a TFLM op.
    K_STEP is chosen so this has the same slope at xi=xiL as the erf step it replaces."""
    return 0.5 * (1.0 + torch.tanh((xi - xiL) / K_STEP))


def dsmoothstep(xi, xiL):
    """d(smoothstep)/dxi -- replaces the Gaussian delta_eps used in leakpinn/pinn.py's pde();
    training-only (feeds the PDE residual), so exact consistency with smoothstep() matters more
    here than TFLM-op-safety."""
    s = torch.tanh((xi - xiL) / K_STEP)
    return (1.0 - s * s) / (2.0 * K_STEP)


def steady_profiles(xi, xiL, kappa, H1n, H2n, xi1, qv0, Hs=10.0):
    """Hss(xi) (linear head-loss profile through the two reference heads) and qss(xi) (smoothed
    leak step) -- the per-scenario generalisation of leakpinn.pinn._steady_profiles."""
    H1_0, H2_0 = 100.0 * H1n, 100.0 * H2n
    slope = (H1_0 - H2_0) / (1.0 - xi1)
    Hss = H2_0 + slope * (1.0 - xi)
    HL = H2_0 + slope * (1.0 - xiL)
    S = smoothstep(xi, xiL)
    qss = qv0 + kappa * torch.sqrt(floor_(HL, 1.0)) * (1.0 - S)
    return Hss, qss


# ------------------------------------------------------------------------------- network
class GeneralNet(torch.nn.Module):
    """7 dimensionless inputs -> 2 dimensionless perturbation outputs (h, q).

    Inputs (all dimensionless; see DEPLOYMENT.md for how to compute each from raw pipe
    parameters):
      xi    = x / L                                        in [0, 1]
      tau   = a t / L                                       dimensionless time
      xiL   = leak position / L                              in [0, 1]
      kappa = leak conductance, B_nom * CdA * sqrt(2g)        > 0
      phi   = friction number, f L g / (2 D a^2)              > 0
      H1n   = reference head (upstream) / 100                 [100 m is just a fixed divisor]
      H2n   = reference head (valve) / 100

    Hard constraints (same convention as leakpinn.pinn.PINNNet): h(xi=0,*)=0 (constant-head
    reservoir) and h=q=0 at tau=0 (steady initial state), built into the output via a
    multiply -- not learned, just arithmetic, and TFLM-safe (Mul only).
    """
    def __init__(self, width=64, depth=4, sigma_xi=1.0, sigma_tau=4.0, n_feat=24, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.feat = _FF(sigma_xi, sigma_tau, n_feat, seed)
        fin = self.feat.out_dim + 5      # + xiL, log(kappa), log(phi), H1n-0.5, H2n-0.5
        self.inp = torch.nn.Linear(fin, width)
        self.hidden = torch.nn.ModuleList([torch.nn.Linear(width, width) for _ in range(depth - 1)])
        self.out = torch.nn.Linear(width, 2)
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)
        self.to(DTYPE)

    def forward(self, xi, tau, xiL, kappa, phi, H1n, H2n):
        z = torch.cat([2.0 * xi - 1.0, 2.0 * tau / TAU_END - 1.0], dim=1)
        f = self.feat(z)
        cond = torch.cat([2.0 * xiL - 1.0, torch.log(kappa), torch.log(phi), H1n - 0.5, H2n - 0.5], dim=1)
        h = torch.tanh(self.inp(torch.cat([f, cond], dim=1)))
        for lin in self.hidden:
            h = torch.tanh(lin(h))
        y = self.out(h)
        return torch.cat([xi * tau * y[:, 0:1], tau * y[:, 1:2]], dim=1)


class DeployNet(torch.nn.Module):
    """Export wrapper: trained GeneralNet + the deterministic steady-profile reconstruction,
    all in TFLM-safe ops, so the exported graph outputs PHYSICAL head/flow directly instead of
    requiring hand-written per-deployment reconstruction code (which is what DEPLOYMENT.md's
    original single-pipe export required, and which does not generalise across pipes).

    Input tensor, shape [batch, 10], columns:
      0 xi   1 tau   2 xiL   3 kappa   4 phi   5 H1n   6 H2n   7 xi1   8 B_nom   9 qv0
    (0-6 feed the learned network; 7-9 are used only by the deterministic arithmetic below.)

    Output tensor, shape [batch, 2]: physical head H [m], physical flow Q [m^3/s].
    """
    def __init__(self, net: GeneralNet, Hs: float = 10.0):
        super().__init__()
        self.net = net
        self.Hs = Hs

    def forward(self, u):
        xi, tau, xiL, kappa, phi = u[:, 0:1], u[:, 1:2], u[:, 2:3], u[:, 3:4], u[:, 4:5]
        H1n, H2n, xi1, B_nom, qv0 = u[:, 5:6], u[:, 6:7], u[:, 7:8], u[:, 8:9], u[:, 9:10]
        y = self.net(xi, tau, xiL, kappa, phi, H1n, H2n)
        h, q = y[:, 0:1], y[:, 1:2]
        Hss, qss = steady_profiles(xi, xiL, kappa, H1n, H2n, xi1, qv0, self.Hs)
        H = Hss + self.Hs * h
        Q = (qss + self.Hs * q) / B_nom
        return torch.cat([H, Q], dim=1)


# ------------------------------------------------------------------------------- physics residuals (training only)
def _grad(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True)[0]


def pde_residuals(net, xi, tau, xiL, kappa, phi, H1n, H2n, xi1, qv0, Hs=10.0):
    """Continuity + momentum residuals (r1, r2) -- the per-scenario generalisation of
    leakpinn.pinn.build_model's `pde()`, with xiL/kappa/phi/H1n/H2n/xi1/qv0 varying per row
    instead of being fixed Problem-level constants."""
    xi = xi.clone().requires_grad_(True)
    tau = tau.clone().requires_grad_(True)
    y = net(xi, tau, xiL, kappa, phi, H1n, H2n)
    h, q = y[:, 0:1], y[:, 1:2]
    h_tau, h_xi = _grad(h, tau), _grad(h, xi)
    q_tau, q_xi = _grad(q, tau), _grad(q, xi)
    Hss, qss = steady_profiles(xi, xiL, kappa, H1n, H2n, xi1, qv0, Hs)
    Hloc = floor_(Hss + Hs * h, 1.0)
    bump = dsmoothstep(xi, xiL)
    dqL = kappa * (torch.sqrt(Hloc) - torch.sqrt(floor_(Hss, 1.0))) / Hs * bump
    r1 = h_tau + (q_xi + dqL)
    qt = qss + Hs * q
    r2 = q_tau + h_xi + phi * (qt * torch.abs(qt) - qss * torch.abs(qss)) / Hs
    return r1, r2


def valve_residual(net, tau, xiL, kappa, phi, H1n, H2n, xi1, qv0, tau_c, dtau_close=0.20, Hs=10.0):
    """Soft valve boundary condition at xi=1 -- training-only (never exported), so plain
    torch.clamp is fine here even though the exported path avoids it."""
    one = torch.ones_like(tau)
    y = net(one, tau, xiL, kappa, phi, H1n, H2n)
    q_out, h_out = y[:, 1:2], y[:, 0:1]
    u = torch.clamp(tau / tau_c, 0.0, 1.0)
    tv = 1.0 - dtau_close * 0.5 * (1.0 - torch.cos(np.pi * u))
    H2_0 = 100.0 * H2n
    kv = qv0 / torch.sqrt(floor_(H2_0, 1.0))
    Hv = floor_(H2_0 + Hs * h_out, 1.0)
    target = (kv * tv * torch.sqrt(Hv) - qv0) / Hs
    return q_out - target


# ------------------------------------------------------------------------------- scenario batching
def sample_batch(rng: np.random.Generator, n_scenarios: int, pts_per_scenario: int, bnd_per_scenario: int):
    """Draws `n_scenarios` random pipes (leakpinn.domain.sample_scenario) and, for each, some
    interior (xi, tau) collocation points (half uniform, half concentrated near that scenario's
    own leak location -- the leak is a narrow feature, same rationale as LeakZoomSampler in
    leakpinn/pinn.py) plus some xi=1 boundary points for the valve condition. Returns the raw
    point arrays (as columns: xi/tau, xiL, kappa, phi, H1n, H2n, xi1, qv0, tau_c) plus the list
    of sampled Scenarios (kept for held-out validation elsewhere)."""
    interior_rows, bnd_rows, scenarios = [], [], []
    for _ in range(n_scenarios):
        sc = sample_scenario(rng)
        c = scenario_consts(sc)
        const = [sc.xiL, c["kappa"], c["phi"], c["H1_0"] / 100.0, c["H2_0"] / 100.0,
                 c["xi1"], c["qv0"], c["tau_c"]]
        n_u = pts_per_scenario // 2
        n_f = pts_per_scenario - n_u
        xi = np.concatenate([rng.uniform(0.0, 1.0, n_u),
                             np.clip(rng.normal(sc.xiL, 4 * EPS_EDGE, n_f), 1e-3, 1 - 1e-3)])
        tau = rng.uniform(0.0, TAU_END, pts_per_scenario)
        interior_rows.append(np.column_stack([xi, tau] + [np.full(pts_per_scenario, v) for v in const]))
        tau_b = rng.uniform(0.0, TAU_END, bnd_per_scenario)
        bnd_rows.append(np.column_stack([tau_b] + [np.full(bnd_per_scenario, v) for v in const]))
        scenarios.append(sc)
    return np.concatenate(interior_rows, 0), np.concatenate(bnd_rows, 0), scenarios


def _cols(X, device):
    """Interior columns -> (xi, tau, xiL, kappa, phi, H1n, H2n, xi1, qv0), each (N,1) tensors."""
    t = torch.as_tensor(X, dtype=DTYPE, device=device)
    return [t[:, i:i + 1] for i in range(9)]


def _bnd_cols(X, device):
    """Boundary columns -> (tau, xiL, kappa, phi, H1n, H2n, xi1, qv0, tau_c)."""
    t = torch.as_tensor(X, dtype=DTYPE, device=device)
    return [t[:, i:i + 1] for i in range(9)]


# ------------------------------------------------------------------------------- config + fit
@dataclass
class GeneralConfig:
    width: int = 64
    depth: int = 4
    sigma_xi: float = 1.0
    sigma_tau: float = 4.0
    n_feat: int = 24
    n_scenarios: int = 48
    pts_per_scenario: int = 120
    bnd_per_scenario: int = 24
    refresh_every: int = 40        # resample a fresh batch of scenarios every N Adam steps
    adam_iters: int = 4000
    lr: float = 2e-3
    lr_decay_steps: int = 1500
    lr_decay_rate: float = 0.6
    lbfgs_iters: int = 400
    lbfgs_scenarios: int = 96      # a bigger, FIXED pool for the deterministic LBFGS polish stage
    w_pde: float = 1.0
    w_bc: float = 4.0
    Hs: float = 10.0
    seed: int = 0
    display_every: int = 200


@dataclass
class GeneralFitResult:
    cfg: GeneralConfig
    net: GeneralNet
    history: list
    seconds: float


def _batch_losses(net, Xi, Xb, cfg, device):
    xi, tau, xiL, kappa, phi, H1n, H2n, xi1, qv0 = _cols(Xi, device)
    r1, r2 = pde_residuals(net, xi, tau, xiL, kappa, phi, H1n, H2n, xi1, qv0, cfg.Hs)
    tau_b, xiL_b, kappa_b, phi_b, H1n_b, H2n_b, xi1_b, qv0_b = _bnd_cols(Xb, device)[:8]
    tau_c_b = torch.as_tensor(Xb[:, 8:9], dtype=DTYPE, device=device)
    r3 = valve_residual(net, tau_b, xiL_b, kappa_b, phi_b, H1n_b, H2n_b, xi1_b, qv0_b, tau_c_b, Hs=cfg.Hs)
    l_pde = torch.mean(r1 ** 2) + torch.mean(r2 ** 2)
    l_bc = torch.mean(r3 ** 2)
    return cfg.w_pde * l_pde + cfg.w_bc * l_bc, float(l_pde.detach()), float(l_bc.detach())


def fit_general(cfg: GeneralConfig, verbose: bool = True) -> GeneralFitResult:
    """Adam (periodically resampling a fresh pool of random pipes) followed by an L-BFGS
    polish on one large, fixed pool -- same two-stage idea as leakpinn.pinn.fit, adapted to
    a multi-scenario forward-only problem (no trainable physical unknowns here: xiL/kappa/phi
    are GIVEN per point, not fitted)."""
    t0 = time.time()
    torch.manual_seed(cfg.seed)
    net = GeneralNet(cfg.width, cfg.depth, cfg.sigma_xi, cfg.sigma_tau, cfg.n_feat, cfg.seed).to(DEVICE)
    rng = np.random.default_rng(cfg.seed)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.lr_decay_steps, gamma=cfg.lr_decay_rate)
    history = []
    Xi, Xb, _ = sample_batch(rng, cfg.n_scenarios, cfg.pts_per_scenario, cfg.bnd_per_scenario)
    for it in range(cfg.adam_iters):
        if it > 0 and it % cfg.refresh_every == 0:
            Xi, Xb, _ = sample_batch(rng, cfg.n_scenarios, cfg.pts_per_scenario, cfg.bnd_per_scenario)
        opt.zero_grad()
        loss, l_pde, l_bc = _batch_losses(net, Xi, Xb, cfg, DEVICE)
        loss.backward()
        opt.step()
        sched.step()
        if it % cfg.display_every == 0 or it == cfg.adam_iters - 1:
            history.append(dict(stage="adam", it=it, loss=float(loss.detach()), l_pde=l_pde, l_bc=l_bc,
                                t=time.time() - t0))
            if verbose:
                print(f"  [adam  {it:5d}] loss={float(loss.detach()):.4e}  pde={l_pde:.4e}  bc={l_bc:.4e}")

    if cfg.lbfgs_iters > 0:
        Xi, Xb, _ = sample_batch(rng, cfg.lbfgs_scenarios, cfg.pts_per_scenario, cfg.bnd_per_scenario)
        lopt = torch.optim.LBFGS(net.parameters(), lr=1.0, max_iter=cfg.lbfgs_iters, max_eval=int(cfg.lbfgs_iters * 1.25),
                                 tolerance_grad=1e-12, tolerance_change=1e-14, history_size=60, line_search_fn="strong_wolfe")
        state = {}

        def closure():
            lopt.zero_grad()
            loss, l_pde, l_bc = _batch_losses(net, Xi, Xb, cfg, DEVICE)
            loss.backward()
            state.update(loss=float(loss.detach()), l_pde=l_pde, l_bc=l_bc)
            return loss

        lopt.step(closure)
        history.append(dict(stage="lbfgs", it=cfg.adam_iters, **state, t=time.time() - t0))
        if verbose:
            print(f"  [lbfgs done] loss={state['loss']:.4e}  pde={state['l_pde']:.4e}  bc={state['l_bc']:.4e}")

    return GeneralFitResult(cfg, net, history, time.time() - t0)


# ------------------------------------------------------------------------------- inference / reconstruction
def predict_physical(net: GeneralNet, sc: Scenario, x_m: np.ndarray, t_s: np.ndarray, Hs: float = 10.0):
    """Physical H(x,t) [m] and Q(x,t) [m^3/s] for one scenario, from the trained net -- mirrors
    leakpinn.pinn.predict_fields but takes an explicit Scenario (any pipe, not just the one the
    Problem was built for)."""
    c = scenario_consts(sc)
    device = next(net.parameters()).device
    XX, TT = np.meshgrid(x_m / sc.pipe.L, t_s * sc.a / sc.pipe.L, indexing="ij")
    n = XX.size
    xi = torch.as_tensor(XX.reshape(-1, 1), dtype=DTYPE, device=device)
    tau = torch.as_tensor(TT.reshape(-1, 1), dtype=DTYPE, device=device)
    xiL = torch.full((n, 1), sc.xiL, dtype=DTYPE, device=device)
    kappa = torch.full((n, 1), c["kappa"], dtype=DTYPE, device=device)
    phi = torch.full((n, 1), c["phi"], dtype=DTYPE, device=device)
    H1n = torch.full((n, 1), c["H1_0"] / 100.0, dtype=DTYPE, device=device)
    H2n = torch.full((n, 1), c["H2_0"] / 100.0, dtype=DTYPE, device=device)
    xi1 = torch.full((n, 1), c["xi1"], dtype=DTYPE, device=device)
    qv0 = torch.full((n, 1), c["qv0"], dtype=DTYPE, device=device)
    with torch.no_grad():
        y = net(xi, tau, xiL, kappa, phi, H1n, H2n)
        Hss, qss = steady_profiles(xi, xiL, kappa, H1n, H2n, xi1, qv0, Hs)
        H = (Hss + Hs * y[:, 0:1]).cpu().numpy().reshape(XX.shape)
        Q = ((qss + Hs * y[:, 1:2]) / c["B_nom"]).cpu().numpy().reshape(XX.shape)
    return H, Q
