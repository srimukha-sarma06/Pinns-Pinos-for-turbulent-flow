"""DeepXDE (PyTorch backend) inverse PINN for leak localisation from valve-induced transients.

Formulation (all derived in the README, every symbol below is dimensionless unless noted)
------------------------------------------------------------------------------------------
  xi  = x / L                       tau = a_nom t / L        (waves move at ~1 in (xi, tau))
  h   = (H - H_ss(xi)) / Hs         q   = B_nom (Q - Q_ss(xi)) / Hs      Hs = 10 m
The network predicts the *perturbation* from the pre-transient steady state:

  continuity : h_tau + rho^2 [ q_xi + kappa (sqrt(H_ss + Hs h) - sqrt(H_ss)) / Hs * delta_eps(xi - xi_L) ] = 0
  momentum   : q_tau + h_xi + phi [ q_t|q_t| - q_s|q_s| ] / Hs = 0        (q_t = q_ss + Hs q)

  hard IC        : h = q = 0 at tau = 0            (factor tau in the output transform)
  hard reservoir : h(0, tau) = 0                   (factor xi)
  valve BC (soft): q(1,tau) = cv*kv*[ tv(tau) sqrt(H2_0 + Hs h) - sqrt(H2_0) ] / Hs      (known command tv)
  data           : h(xi_1, tau_i) = h1_i ,  h(1, tau_i) = h2_i            (two pressure sensors, no flow!)

Unknown physical scalars trained jointly with the network weights (external trainable variables):
  xi_L  leak position,  kappa  leak conductance (kappa = B_nom * CdA * sqrt(2g)),
  rho   wave-speed ratio a/a_nom (a is never known exactly),  cv  valve-coefficient scale (datasheet +-).
"""
from __future__ import annotations
import os, time
os.environ.setdefault("DDE_BACKEND", "pytorch")
from dataclasses import dataclass, field, asdict
import numpy as np
import torch
import deepxde as dde
# DeepXDE's pytorch backend already sets torch's default device to "cuda" on import when a GPU
# is available (see deepxde/backend/pytorch/tensor.py) -- we deliberately do NOT override that
# here, so this module runs on GPU automatically when one is present. Every tensor we create
# explicitly below either omits `device=` (so it follows that same default) or is moved off the
# GPU with `.cpu()` before `.numpy()` (numpy cannot read a CUDA tensor directly).
DEVICE = torch.get_default_device() if hasattr(torch, "get_default_device") else \
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
from .physics import G
from .synth import Dataset
from .baselines import estimate_wave_speed

dde.config.set_default_float("float64")


# ------------------------------------------------------------------------------- configuration
@dataclass
class PINNConfig:
    arch: str = "ff"               # 'mlp' | 'ff' | 'modmlp' | 'char'
    width: int = 48
    depth: int = 4
    sigma_xi: float = 1.0          # Fourier-feature scale in normalised xi
    sigma_tau: float = 4.0         # Fourier-feature scale in normalised tau
    n_feat: int = 24
    n_col: int = 6000              # PDE collocation points
    n_bnd: int = 400               # boundary points for the valve BC
    eps: float = 0.03              # leak source width in xi (3 m of 100 m); << pulse length (~28 m)
    Hs: float = 10.0               # head scale [m]
    tau_pad: float = 0.0
    # training schedule: list of stages. free=False keeps the physical unknowns frozen at their initial values
    # (lets the network first learn the leak-independent wave field); eps=(e0,e1) anneals the leak-source width.
    schedule: list = field(default_factory=lambda: [
        dict(opt="adam", iters=800, free=False),                   # warm up the field with the leak fixed at init
        dict(opt="adam", iters=3500, free=True, eps=(0.20, 0.03), lr=1e-3),  # let position/size move, wide->narrow eps
        dict(opt="lbfgs", iters=1200, free=True)])                 # sharpen
    lr: float = 2e-3
    lr_decay_steps: int = 2000
    lr_decay_rate: float = 0.5
    w_pde: float = 8.0
    w_bc: float = 10.0
    w_data: float = 60.0           # data is trusted to ~ sensor noise; kept lower than before so the PDE
                                     # residual cannot be relaxed to explain away a leak at the wrong location
    resample_every: int = 250      # periodically redraw collocation points (avoids fitting a fixed grid)
    train_rho: bool = True
    train_cv: bool = False        # valve coefficient is taken from its calibration (see README, sensitivity study)
    xiL_init: float = 0.5
    log_kappa_init: float = 0.0
    seed: int = 0
    display_every: int = 500


# ------------------------------------------------------------------------------- networks
class _FF(torch.nn.Module):
    """Random Fourier features (Tancik et al. 2020), optionally on the characteristic coordinates
    (tau - xi, tau + xi) along which the lossless wave equation is exactly 1-D."""
    def __init__(self, sig_xi, sig_tau, n_feat, seed, characteristic=False):
        super().__init__()
        g = torch.Generator(device=DEVICE).manual_seed(seed)
        Bm = torch.randn(2, n_feat, generator=g, dtype=torch.get_default_dtype(), device=DEVICE)
        Bm[0] *= sig_xi; Bm[1] *= sig_tau
        self.register_buffer("B", Bm * 2 * np.pi)
        self.characteristic = characteristic
        self.out_dim = 2 + 2 * n_feat

    def forward(self, z):                       # z in [-1,1]^2 (xi, tau normalised)
        if self.characteristic:
            z = torch.cat([0.5 * (z[:, 1:2] - z[:, 0:1]), 0.5 * (z[:, 1:2] + z[:, 0:1])], dim=1)
        p = z @ self.B
        return torch.cat([z, torch.sin(p), torch.cos(p)], dim=1)


class PINNNet(dde.nn.NN):
    def __init__(self, cfg: PINNConfig, tau_end: float):
        super().__init__()
        self.cfg, self.tau_end = cfg, tau_end
        torch.manual_seed(cfg.seed)
        W, d = cfg.width, cfg.depth
        a = cfg.arch
        if a == "mlp":
            self.feat, fin = None, 2
        elif a in ("ff", "modmlp"):
            self.feat = _FF(cfg.sigma_xi, cfg.sigma_tau, cfg.n_feat, cfg.seed); fin = self.feat.out_dim
        elif a == "char":
            self.feat = _FF(cfg.sigma_xi, cfg.sigma_tau, cfg.n_feat, cfg.seed, characteristic=True); fin = self.feat.out_dim
        else:
            raise ValueError(a)
        self.modified = a == "modmlp"
        self.inp = torch.nn.Linear(fin, W)
        self.hidden = torch.nn.ModuleList([torch.nn.Linear(W, W) for _ in range(d - 1)])
        self.out = torch.nn.Linear(W, 2)
        if self.modified:                        # Wang-Teng-Perdikaris 'modified MLP' gating
            self.U, self.V = torch.nn.Linear(fin, W), torch.nn.Linear(fin, W)
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight); torch.nn.init.zeros_(m.bias)
        self.double() if torch.get_default_dtype() == torch.float64 else None

    def forward(self, inputs):
        xi, tau = inputs[:, 0:1], inputs[:, 1:2]
        z = torch.cat([2 * xi - 1, 2 * tau / self.tau_end - 1], dim=1)
        f = z if self.feat is None else self.feat(z)
        h = torch.tanh(self.inp(f))
        if self.modified:
            u, v = torch.tanh(self.U(f)), torch.tanh(self.V(f))
            for lin in self.hidden:
                zg = torch.tanh(lin(h)); h = (1 - zg) * u + zg * v
        else:
            for lin in self.hidden:
                h = torch.tanh(lin(h))
        y = self.out(h)
        # hard constraints: h(0,tau)=0 (constant-head reservoir) and h=q=0 at tau=0 (steady initial state)
        return torch.cat([xi * tau * y[:, 0:1], tau * y[:, 1:2]], dim=1)


# ------------------------------------------------------------------------------- problem set-up
@dataclass
class Problem:
    data: Dataset
    a_nom: float
    L: float
    tau_end: float
    Hs: float
    B_nom: float
    xi1: float
    H1_0: float
    H2_0: float
    kv: float                 # B_nom * Cv
    phi: float                # dimensionless friction coefficient
    kappa0: float             # reference leak conductance (CdA_ref = 3 mm^2)
    dtau: float
    t_close: float
    pts: np.ndarray           # (2*nt, 2)  (xi, tau) of sensor samples
    vals: np.ndarray          # (2*nt, 1)  measured h
    sigma_h: float


def build_problem(data: Dataset, cfg: PINNConfig, a_nom: float | None = None) -> Problem:
    p = data.pipe
    a_nom = a_nom or estimate_wave_speed(data)
    B_nom = p.B(a_nom)
    base = data.H_pre.mean(axis=0)
    H1_0, H2_0 = float(base[0]), float(base[1])
    Q_v0 = data.valve.Cv * np.sqrt(H2_0)
    f = float(p.friction_factor(Q_v0))
    phi = f * p.L * G / (2 * p.D * a_nom ** 2)
    tau_end = a_nom * (data.t[-1] + 1e-9) / p.L + cfg.tau_pad
    xi1 = float(data.x_sensors[0] / p.L)
    tau_i = data.t * a_nom / p.L
    hm = (data.H_meas - base) / cfg.Hs
    pts = np.concatenate([np.stack([np.full_like(tau_i, xi1), tau_i], 1), np.stack([np.ones_like(tau_i), tau_i], 1)])
    vals = np.concatenate([hm[:, 0:1], hm[:, 1:2]])
    return Problem(data, a_nom, p.L, tau_end, cfg.Hs, B_nom, xi1, H1_0, H2_0, B_nom * data.valve.Cv, phi,
                   B_nom * 3e-6 * np.sqrt(2 * G), data.valve.dtau, data.valve.t_close, pts, vals,
                   data.meta["noise_std"] / cfg.Hs)


class Params:
    """Physical unknowns as DeepXDE trainable variables (with well-scaled reparametrisation).

    `fixed` is now a PARTIAL spec: any of xiL/kappa/rho/cv given in the dict is held constant;
    any name left OUT of the dict stays a trainable variable (subject to cfg.train_rho/train_cv
    for rho/cv). Pass fixed=None for "all four trainable" (the old behaviour), or e.g.
    fixed=dict(xiL=0.5, kappa=0.0, cv=1.0) to fix everything except rho (wave speed) -- used by
    the "PINN corrects a wrong wave-speed prior" comparison against plain MOC."""
    def __init__(self, cfg: PINNConfig, fixed: dict | None = None):
        self.cfg, self.fixed = cfg, (fixed or {})
        self.eps = cfg.eps
        self.vars = []
        if "xiL" not in self.fixed:
            init = float(np.log(np.clip((cfg.xiL_init - 0.03) / 0.94, 1e-3, 1 - 1e-3) /
                                (1 - np.clip((cfg.xiL_init - 0.03) / 0.94, 1e-3, 1 - 1e-3))))
            self.vx = dde.Variable(init); self.vars.append(self.vx)
        if "kappa" not in self.fixed:
            self.vk = dde.Variable(float(cfg.log_kappa_init)); self.vars.append(self.vk)
        if "rho" not in self.fixed and cfg.train_rho:
            self.vr = dde.Variable(0.0); self.vars.append(self.vr)
        if "cv" not in self.fixed and cfg.train_cv:
            self.vc = dde.Variable(0.0); self.vars.append(self.vc)

    def get(self, prob: Problem):
        t = lambda v: torch.tensor(float(v), dtype=torch.get_default_dtype())
        # smooth (never-saturating) reparameterisations so gradients never vanish at a bound:
        #   xiL   = sigmoid maps R -> (0.03, 0.97) of the pipe length
        #   kappa = softplus keeps the leak conductance positive without a hard floor at 0
        xiL = t(self.fixed["xiL"]) if "xiL" in self.fixed else 0.03 + 0.94 * torch.sigmoid(self.vx)
        kappa = t(self.fixed["kappa"]) if "kappa" in self.fixed else \
            prob.kappa0 * torch.nn.functional.softplus(1.0 + self.vk) / torch.nn.functional.softplus(torch.tensor(1.0))
        if "rho" in self.fixed:
            rho = t(self.fixed["rho"])
        elif self.cfg.train_rho:
            rho = 1.0 + 0.1 * self.vr
        else:
            rho = t(1.0)
        if "cv" in self.fixed:
            cv = t(self.fixed["cv"])
        elif self.cfg.train_cv:
            cv = 1.0 + 0.1 * self.vc
        else:
            cv = t(1.0)
        return dict(xiL=xiL, kappa=kappa, rho=rho, cv=cv)

    def numpy(self, prob: Problem):
        v = {k: float(x.detach()) for k, x in self.get(prob).items()}
        v["x_L"] = v["xiL"] * prob.L
        v["CdA"] = v["kappa"] / (prob.B_nom * np.sqrt(2 * G))
        v["a"] = v["rho"] * prob.a_nom
        return v


def _steady_profiles(xi, prm, prob: Problem, eps):
    """H_ss(xi) (linear between the two sensors' steady readings) and q_ss(xi) (smoothed leak step)."""
    slope = (prob.H1_0 - prob.H2_0) / (1.0 - prob.xi1)
    Hss = prob.H2_0 + slope * (1.0 - xi)
    HL = prob.H2_0 + slope * (1.0 - prm["xiL"])
    qv0 = prm["cv"] * prob.kv * torch.sqrt(torch.tensor(prob.H2_0, dtype=xi.dtype))
    S = 0.5 * (1 + torch.erf((xi - prm["xiL"]) / (np.sqrt(2) * eps)))
    qss = qv0 + prm["kappa"] * torch.sqrt(HL) * (1 - S)
    return Hss, qss, qv0


def build_model(prob: Problem, cfg: PINNConfig, prm: Params):
    geom = dde.geometry.GeometryXTime(dde.geometry.Interval(0.0, 1.0), dde.geometry.TimeDomain(0.0, prob.tau_end))
    Hs = prob.Hs

    def pde(x, y):
        h, q = y[:, 0:1], y[:, 1:2]
        h_t = dde.grad.jacobian(y, x, i=0, j=1); h_x = dde.grad.jacobian(y, x, i=0, j=0)
        q_t = dde.grad.jacobian(y, x, i=1, j=1); q_x = dde.grad.jacobian(y, x, i=1, j=0)
        xi = x[:, 0:1]
        P = prm.get(prob); eps = prm.eps
        Hss, qss, _ = _steady_profiles(xi, P, prob, eps)
        gauss = torch.exp(-0.5 * ((xi - P["xiL"]) / eps) ** 2) / (eps * np.sqrt(2 * np.pi))
        Hloc = torch.clamp(Hss + Hs * h, min=1.0)
        dqL = P["kappa"] * (torch.sqrt(Hloc) - torch.sqrt(Hss)) / Hs * gauss
        r1 = h_t + P["rho"] ** 2 * (q_x + dqL)
        qt = qss + Hs * q
        r2 = q_t + h_x + prob.phi * (qt * torch.abs(qt) - qss * torch.abs(qss)) / Hs
        return [r1, r2]

    def valve_bc(inputs, outputs, X):
        P = prm.get(prob)
        t = inputs[:, 1:2] * prob.L / prob.a_nom
        u = torch.clamp(t / prob.t_close, 0.0, 1.0)
        tv = 1.0 - prob.dtau * 0.5 * (1.0 - torch.cos(np.pi * u))
        Hv = torch.clamp(prob.H2_0 + Hs * outputs[:, 0:1], min=1.0)
        target = P["cv"] * prob.kv * (tv * torch.sqrt(Hv) - np.sqrt(prob.H2_0)) / Hs
        return outputs[:, 1:2] - target

    bc_valve = dde.icbc.OperatorBC(geom, valve_bc, lambda x, on: on and np.isclose(x[0], 1.0))
    n = len(prob.pts) // 2
    d1 = dde.icbc.PointSetBC(prob.pts[:n], prob.vals[:n], component=0)
    d2 = dde.icbc.PointSetBC(prob.pts[n:], prob.vals[n:], component=0)
    data = dde.data.TimePDE(geom, pde, [bc_valve, d1, d2], num_domain=cfg.n_col, num_boundary=cfg.n_bnd,
                            num_initial=0, train_distribution="Hammersley")
    net = PINNNet(cfg, prob.tau_end)
    model = dde.Model(data, net)
    return model


class Track(dde.callbacks.Callback):
    def __init__(self, prm, prob, every=200):
        super().__init__(); self.prm, self.prob, self.every, self.hist = prm, prob, every, []

    def on_epoch_end(self):
        e = self.model.train_state.epoch
        if e % self.every == 0 and self.prm.vars:
            v = self.prm.numpy(self.prob); v["epoch"] = e
            v["loss"] = float(np.sum(self.model.train_state.loss_train)); v["eps"] = self.prm.eps
            self.hist.append(v)


class LeakZoomSampler(dde.callbacks.Callback):
    """Adaptive/importance collocation sampling for the inverse source problem.

    The leak only shows up in the PDE residual as a Gaussian bump of width `eps` centred at the
    CURRENT estimate of xi_L.  With points drawn uniformly over the whole (xi, tau) rectangle, only
    a tiny fraction ever land near that bump, so the gradient telling the optimiser where the leak
    really is (i.e. where mass conservation is violated by the wrong-location hypothesis) is weak and
    noisy -- this is what let the network 'explain away' a leak in the wrong place in earlier runs.
    Every `period` epochs we redraw the collocation set as a mixture: half uniform over the pipe (for
    global physics coverage) and half concentrated in a narrow band around the current xi_L estimate
    (both a fine band matching the leak width, and a wider band for basin-of-attraction search)."""
    def __init__(self, prm, prob, cfg, period=150):
        super().__init__(); self.prm, self.prob, self.cfg, self.period = prm, prob, cfg, period
        self.since = 0

    def on_train_begin(self):
        self._resample()

    def on_epoch_end(self):
        self.since += 1
        if self.since >= self.period:
            self.since = 0
            self._resample()

    def _resample(self):
        cfg, prob, rng = self.cfg, self.prob, np.random.default_rng(self.model.train_state.epoch + 1)
        n = cfg.n_col
        with torch.no_grad():
            xiL = float(self.prm.get(prob)["xiL"])
        n_u, n_fine, n_wide = n // 2, n // 4, n - n // 2 - n // 4
        xi_u = rng.uniform(0, 1, n_u)
        xi_f = np.clip(rng.normal(xiL, 3 * self.prm.eps, n_fine), 0.001, 0.999)
        xi_w = np.clip(rng.normal(xiL, 0.15, n_wide), 0.001, 0.999)
        xi = np.concatenate([xi_u, xi_f, xi_w])
        tau = rng.uniform(0, prob.tau_end, n)
        X = np.stack([xi, tau], 1).astype(np.float64)
        self.model.data.replace_with_anchors(X)


class EpsAnneal(dde.callbacks.Callback):
    """Geometric annealing of the leak-source width: a wide source gives the leak position a long-range
    gradient (large basin of attraction); the width is then shrunk to its physical value."""
    def __init__(self, prm, e0, e1, iters):
        super().__init__(); self.prm, self.e0, self.e1, self.n = prm, e0, e1, iters

    def on_train_begin(self):
        self.start = self.model.train_state.epoch; self.prm.eps = self.e0

    def on_epoch_end(self):
        f = min((self.model.train_state.epoch - self.start) / max(self.n, 1), 1.0)
        self.prm.eps = float(self.e0 * (self.e1 / self.e0) ** f)


@dataclass
class FitResult:
    cfg: PINNConfig
    prob: Problem
    model: object
    prm: Params
    est: dict
    history: list
    losshistory: object
    seconds: float
    stage_log: list = field(default_factory=list)


def sensor_rmse(model, prob):
    """RMS misfit [m] between the network and the noisy sensor records (noise floor = sensor sigma)."""
    with torch.no_grad():
        y = model.net(torch.tensor(prob.pts, dtype=torch.get_default_dtype())).detach().cpu().numpy()[:, 0:1]
    return float(np.sqrt(np.mean((y - prob.vals) ** 2)) * prob.Hs)


def _weights(cfg):
    return [cfg.w_pde, cfg.w_pde, cfg.w_bc, cfg.w_data, cfg.w_data]


def fit(data: Dataset, cfg: PINNConfig, fixed: dict | None = None, a_nom: float | None = None,
        verbose=True, model_prm=None) -> FitResult:
    """Run the staged optimisation schedule (Adam -> L-BFGS [-> NNCG]) and return the fitted problem."""
    t0 = time.time()
    dde.config.set_random_seed(cfg.seed)
    prob = build_problem(data, cfg, a_nom)
    prm = Params(cfg, fixed)
    model = build_model(prob, cfg, prm)
    tr = Track(prm, prob, every=100)
    zoom = LeakZoomSampler(prm, prob, cfg, period=cfg.resample_every or 150)
    log = []
    for k, st in enumerate(cfg.schedule):
        vars_ = (prm.vars if st.get("free", True) else []) or None
        eps0, eps1 = st.get("eps", (cfg.eps, cfg.eps))
        prm.eps = eps0
        opt, n = st["opt"], st["iters"]
        cbs = [tr, zoom] + ([EpsAnneal(prm, eps0, eps1, n)] if eps0 != eps1 else [])
        if opt == "adam":
            model.compile("adam", lr=st.get("lr", cfg.lr), loss_weights=_weights(cfg), external_trainable_variables=vars_,
                          decay=("step", cfg.lr_decay_steps, cfg.lr_decay_rate))
            model.train(iterations=n, display_every=cfg.display_every, callbacks=cbs, verbose=int(verbose))
        elif opt == "lbfgs":
            dde.optimizers.set_LBFGS_options(maxcor=60, ftol=0.0, gtol=1e-12, maxiter=n, maxfun=int(n * 1.4), maxls=40)
            model.compile("L-BFGS", loss_weights=_weights(cfg), external_trainable_variables=vars_)
            model.train(display_every=max(n // 4, 50), callbacks=cbs, verbose=int(verbose))
        elif opt == "nncg":       # Nystrom-preconditioned Newton-CG polish (second-order), Rathore et al. 2024
            dde.optimizers.set_NNCG_options(rank=50, mu=1e-1, updatefreq=20, chunksz=1, cgmaxiter=200)
            model.compile("NNCG", loss_weights=_weights(cfg), external_trainable_variables=vars_)
            model.train(iterations=n, display_every=max(n // 4, 10), callbacks=cbs, verbose=int(verbose))
        else:
            raise ValueError(opt)
        prm.eps = eps1
        log.append(dict(stage=k, opt=opt, iters=n, free=bool(vars_), loss=float(np.sum(model.train_state.loss_train)),
                        data_rmse_m=sensor_rmse(model, prob),
                        est=(prm.numpy(prob) if prm.vars else None), t=time.time() - t0))
    est = prm.numpy(prob)   # numpy() now always resolves the correct mix of fixed + trained values
    return FitResult(cfg, prob, model, prm, est, tr.hist, model.losshistory, time.time() - t0, log)


def multistart_fit(data: Dataset, cfg: PINNConfig, n_starts: int = 5, search_iters=(600, 800, 250),
                   a_nom: float | None = None, verbose=False, seed0: int = 0) -> FitResult:
    """The leak-position loss landscape is multi-modal (any point that gets the travel-time right
    fits the data locally), so a single gradient-descent start is not reliable -- exactly the kind
    of thing a reviewer will ask about.  We therefore do a cheap coarse scan over the leak-position
    initial guess, keep the candidate with the lowest data misfit, then re-run the FULL schedule
    (with eps-annealing) from that start for the reported result."""
    starts = np.linspace(0.08, 0.92, n_starts)
    short = PINNConfig(**{**asdict(cfg), "schedule": [
        dict(opt="adam", iters=search_iters[0], free=False),
        dict(opt="adam", iters=search_iters[1], free=True, eps=(0.15, 0.05)),
        dict(opt="lbfgs", iters=search_iters[2], free=True)]})
    trials = []
    for i, x0 in enumerate(starts):
        c = PINNConfig(**{**asdict(short), "xiL_init": float(x0), "seed": seed0 + i, "display_every": 10**9})
        r = fit(data, c, a_nom=a_nom, verbose=False)
        trials.append((r.stage_log[-1]["loss"], x0, r))
        if verbose:
            e = r.est
            print(f"  start xiL0={x0:.2f} -> x_L={e['x_L']:.1f} m  CdA={e['CdA']:.2e}  loss={trials[-1][0]:.3e}")
    trials.sort(key=lambda z: z[0])
    best_x0 = trials[0][1]
    if verbose:
        print(f"  best start: xiL0={best_x0:.2f}  (loss {trials[0][0]:.3e}); refining with full schedule")
    final_cfg = PINNConfig(**{**asdict(cfg), "xiL_init": float(best_x0), "seed": seed0})
    res = fit(data, final_cfg, a_nom=a_nom, verbose=verbose)
    res.stage_log.insert(0, dict(stage=-1, opt="multistart-scan", iters=sum(search_iters) * n_starts, free=True,
                                 loss=trials[0][0], data_rmse_m=None,
                                 est=[dict(xiL0=float(x0), loss=float(l), x_L=float(t.est["x_L"])) for l, x0, t in trials], t=None))
    return res


# ------------------------------------------------------------------------------- reconstruction
def predict_fields(res: FitResult, x_m: np.ndarray, t_s: np.ndarray):
    """Physical H(x,t) [m] and Q(x,t) [m^3/s] on a tensor grid, from the trained PINN + estimated parameters."""
    prob, prm = res.prob, res.prm
    XX, TT = np.meshgrid(x_m / prob.L, t_s * prob.a_nom / prob.L, indexing="ij")
    pts = np.stack([XX.ravel(), TT.ravel()], 1)
    with torch.no_grad():
        y = res.model.net(torch.tensor(pts, dtype=torch.get_default_dtype())).cpu().numpy()
        P = prm.get(prob)
        xi = torch.tensor(XX.ravel()[:, None], dtype=torch.get_default_dtype())
        Hss, qss, _ = _steady_profiles(xi, P, prob, prm.eps)
    H = (Hss.cpu().numpy() + prob.Hs * y[:, 0:1]).reshape(XX.shape)
    Q = ((qss.cpu().numpy() + prob.Hs * y[:, 1:2]) / prob.B_nom).reshape(XX.shape)
    return H, Q
