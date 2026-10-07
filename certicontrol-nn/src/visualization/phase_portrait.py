"""Plotting utilities for certified safe sets, B&B partitions and trajectories."""
from __future__ import annotations

from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import PatchCollection
from scipy.integrate import solve_ivp

from src.verification.lirpa_verifier import METHOD_CODES

METHOD_COLORS = {"ibp": "#bdbdbd", "crown": "#3182bd", "z3": "#f16913"}
SAFE = "#31a354"
UNSAFE = "#de2d26"


# --------------------------------------------------------------------- sim
def simulate(cert, x0: Sequence[float], T: float = 10.0, n_eval: int = 400, max_step: float = 0.01):
    """Integrate the closed loop x' = f(x) + g(x) pi(x) with scipy (RK45)."""
    system = cert.system
    model = cert.double() if next(cert.parameters()).dtype != torch.float64 else cert

    def rhs(_t, x):
        xt = torch.as_tensor(x, dtype=torch.float64).reshape(1, -1)
        with torch.no_grad():
            u = model.controller(system.features(xt))
            return system.xdot(xt, u).reshape(-1).numpy()

    sol = solve_ivp(rhs, (0.0, T), np.asarray(x0, dtype=np.float64),
                    t_eval=np.linspace(0, T, n_eval), max_step=max_step, rtol=1e-7, atol=1e-9)
    return sol.t, sol.y.T


def _grid_eval(cert, fixed: np.ndarray, dims=(0, 1), lims=None, res=250):
    system = cert.system
    lo, hi = system.domain.tensor()
    lims = lims or [(float(lo[d]), float(hi[d])) for d in dims]
    a = np.linspace(*lims[0], res)
    b = np.linspace(*lims[1], res)
    A, Bm = np.meshgrid(a, b)
    X = np.tile(np.asarray(fixed, dtype=np.float64), (A.size, 1))
    X[:, dims[0]] = A.ravel()
    X[:, dims[1]] = Bm.ravel()
    xt = torch.from_numpy(X)
    model = cert.double()
    with torch.no_grad():
        out = model(xt).numpy()
    return A, Bm, out[:, 0].reshape(A.shape), out[:, 1].reshape(A.shape), system.is_unsafe(xt).numpy().reshape(A.shape)


def draw_leaves(ax, leaves_lo, leaves_hi, methods, dims=(0, 1), outside=None, alpha=0.35, lw=0.3):
    lo = leaves_lo.numpy()
    hi = leaves_hi.numpy()
    m = methods.numpy()
    for name, code in METHOD_CODES.items():
        sel = m == code
        if outside is not None:
            sel = sel & ~outside.numpy()
        if not sel.any():
            continue
        rects = [mpatches.Rectangle((l[dims[0]], l[dims[1]]), h[dims[0]] - l[dims[0]], h[dims[1]] - l[dims[1]])
                 for l, h in zip(lo[sel], hi[sel])]
        ax.add_collection(PatchCollection(rects, facecolor=METHOD_COLORS[name], edgecolor="k",
                                          linewidth=lw, alpha=alpha))
    handles = [mpatches.Patch(color=METHOD_COLORS[k], alpha=0.6, label=f"leaf proven by {k.upper()}")
               for k in METHOD_CODES if (m == METHOD_CODES[k]).any()]
    if outside is not None and outside.any():
        rects = [mpatches.Rectangle((l[dims[0]], l[dims[1]]), h[dims[0]] - l[dims[0]], h[dims[1]] - l[dims[1]])
                 for l, h in zip(lo[outside.numpy()], hi[outside.numpy()])]
        ax.add_collection(PatchCollection(rects, facecolor="none", edgecolor="0.55", linewidth=lw, alpha=0.8))
        handles.append(mpatches.Patch(facecolor="none", edgecolor="0.55", label="leaf proven outside C (B<0)"))
    return handles


# ------------------------------------------------------------- pendulum
def plot_pendulum(cert, verification: Optional[dict] = None, trajectories: int = 12, ax=None,
                  show_leaves: bool = True, seed: int = 0, T: float = 6.0):
    system = cert.system
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 5.2))
    A, Bm, Bv, Cv, uns = _grid_eval(cert, np.zeros(2))
    ax.contourf(A, Bm, uns.astype(float), levels=[0.5, 1.5], colors=[UNSAFE], alpha=0.25)
    ax.contourf(A, Bm, (Bv >= 0).astype(float), levels=[0.5, 1.5], colors=[SAFE], alpha=0.18)
    ax.contour(A, Bm, Bv, levels=[0.0], colors=[SAFE], linewidths=2.2)
    handles = [mpatches.Patch(color=SAFE, alpha=0.35, label=r"certified invariant set $\mathcal{C}=\{B\geq0\}$"),
               mpatches.Patch(color=UNSAFE, alpha=0.35, label=r"unsafe set $|\theta|\geq0.8$")]
    if verification is not None and show_leaves:
        d = verification["domain"]
        handles += draw_leaves(ax, d["leaves_lo"], d["leaves_hi"], d["leaves_method"],
                               outside=d["leaves_outside"], alpha=0.25)
    ilo, ihi = system.init_set.lo, system.init_set.hi
    ax.add_patch(mpatches.Rectangle((ilo[0], ilo[1]), ihi[0] - ilo[0], ihi[1] - ilo[1], fill=False,
                                    ls="--", lw=1.5, ec="k"))
    handles.append(mpatches.Patch(fill=False, ls="--", ec="k", label=r"initial set $X_0$"))
    rng = np.random.default_rng(seed)
    starts = _sample_in_C(cert, trajectories, rng)
    for x0 in starts:
        _, xs = simulate(cert, x0, T=T)
        ax.plot(xs[:, 0], xs[:, 1], color="#252525", lw=0.9, alpha=0.8)
        ax.plot(x0[0], x0[1], "o", color="#252525", ms=3)
    ax.set_xlabel(r"$\theta$ [rad]")
    ax.set_ylabel(r"$\dot\theta$ [rad/s]")
    lo, hi = system.domain.lo, system.domain.hi
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    return ax, handles


def _sample_in_C(cert, n, rng, max_tries=200000, fixed=None):
    system = cert.system
    lo, hi = system.domain.tensor()
    x = lo + (hi - lo) * torch.from_numpy(rng.random((max_tries, system.state_dim)))
    if fixed is not None:
        for d, v in fixed.items():
            x[:, d] = v
    with torch.no_grad():
        B = cert.double().barrier_value(x).squeeze(-1)
    x = x[B >= 0]
    idx = rng.choice(x.shape[0], size=min(n, x.shape[0]), replace=False)
    return x[idx].numpy()


# ----------------------------------------------------------------- dubins
def plot_dubins_slice(cert, theta: float, v: float, ax=None, trajectories: int = 0, seed: int = 0,
                      T: float = 8.0, verification: Optional[dict] = None):
    system = cert.system
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5.5))
    A, Bm, Bv, Cv, uns = _grid_eval(cert, np.array([0.0, 0.0, theta, v]))
    ax.contourf(A, Bm, (Bv >= 0).astype(float), levels=[0.5, 1.5], colors=[SAFE], alpha=0.2)
    ax.contour(A, Bm, Bv, levels=[0.0], colors=[SAFE], linewidths=2.0)
    ax.add_patch(mpatches.Circle(system.c, system.r, color=UNSAFE, alpha=0.4))
    if verification is not None:
        d = verification["domain"]
        lo, hi = d["leaves_lo"], d["leaves_hi"]
        sel = (lo[:, 2] <= theta) & (hi[:, 2] >= theta) & (lo[:, 3] <= v) & (hi[:, 3] >= v)
        draw_leaves(ax, lo[sel], hi[sel], d["leaves_method"][sel], outside=d["leaves_outside"][sel], alpha=0.2)
    # heading arrows
    ax.quiver(*np.meshgrid(np.linspace(-1.8, 1.8, 7), np.linspace(-1.8, 1.8, 7)),
              np.cos(theta), np.sin(theta), color="0.6", scale=30, width=0.004)
    if trajectories:
        rng = np.random.default_rng(seed)
        for x0 in _sample_in_C(cert, trajectories, rng):
            _, xs = simulate(cert, x0, T=T)
            ax.plot(xs[:, 0], xs[:, 1], color="#252525", lw=0.9)
            ax.plot(x0[0], x0[1], "o", color="#252525", ms=3)
    a = system.arena
    ax.set_xlim(-a, a)
    ax.set_ylim(-a, a)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$p_x$")
    ax.set_ylabel(r"$p_y$")
    ax.set_title(rf"$\theta={theta:.2f}$, $v={v:.2f}$", fontsize=10)
    return ax


def plot_dubins_trajectories(cert, n: int = 20, T: float = 8.0, seed: int = 0, ax=None):
    system = cert.system
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.add_patch(mpatches.Circle(system.c, system.r, color=UNSAFE, alpha=0.4, label="obstacle"))
    rng = np.random.default_rng(seed)
    starts = _sample_in_C(cert, n, rng)
    for x0 in starts:
        _, xs = simulate(cert, x0, T=T)
        ax.plot(xs[:, 0], xs[:, 1], lw=1.0)
        ax.plot(x0[0], x0[1], "ko", ms=3)
    a = system.arena
    ax.plot([-a, a, a, -a, -a], [-a, -a, a, a, -a], "k-", lw=1.5)
    ax.plot(*system.goal, "*", color="gold", ms=14, mec="k", label="nominal goal")
    ax.set_xlim(-a - 0.1, a + 0.1)
    ax.set_ylim(-a - 0.1, a + 0.1)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$p_x$")
    ax.set_ylabel(r"$p_y$")
    return ax
