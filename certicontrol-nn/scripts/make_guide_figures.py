"""Teaching figures for docs/GUIDE.md (python scripts/make_guide_figures.py)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
GREEN, RED, BLUE, ORANGE, GREY = "#31a354", "#de2d26", "#3182bd", "#f16913", "#737373"


def fig_invariance():
    """Phase plane: vector field, a barrier level set, arrows pointing inward on the boundary."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    g = np.linspace(-2, 2, 21)
    X, Y = np.meshgrid(g, g)
    # damped oscillator x1' = x2, x2' = -x1 - 0.6 x2
    U, V = Y, -X - 0.6 * Y
    B = 1.5 - (X ** 2 + 0.6 * X * Y + Y ** 2)
    for ax, title, good in [(axes[0], "Forward invariant: the flow points into C on its boundary", True),
                            (axes[1], "Not invariant: flow leaves C through part of its boundary", False)]:
        if not good:
            U, V = Y + 0.8, -X - 0.6 * Y + 0.4
        ax.streamplot(X, Y, U, V, color="0.7", density=1.1, linewidth=0.7, arrowsize=0.8)
        ax.contourf(X, Y, B, levels=[0, 10], colors=[GREEN], alpha=0.2)
        ax.contour(X, Y, B, levels=[0], colors=[GREEN], linewidths=2)
        ax.add_patch(plt.Rectangle((1.45, -2), 0.55, 4, color=RED, alpha=0.25))
        ax.text(1.55, 1.7, "unsafe", color=RED)
        ax.text(-0.35, -0.1, r"$\mathcal{C}=\{B\geq0\}$", color=GREEN)
        # boundary arrows
        th = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        for t in th:
            # point on B=0 ellipse via radial search
            d = np.array([np.cos(t), np.sin(t)])
            r = np.sqrt(1.5 / (d[0] ** 2 + 0.6 * d[0] * d[1] + d[1] ** 2))
            p = r * d
            f = np.array([p[1], -p[0] - 0.6 * p[1]]) if good else np.array([p[1] + 0.8, -p[0] - 0.6 * p[1] + 0.4])
            gradB = np.array([-(2 * p[0] + 0.6 * p[1]), -(0.6 * p[0] + 2 * p[1])])
            col = BLUE if gradB @ f >= 0 else RED
            ax.annotate("", xy=p + 0.35 * f / np.linalg.norm(f), xytext=p,
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.6))
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
    fig.text(0.5, 0.01, r"blue arrow: $\dot B=\nabla B\cdot f\geq 0$ at that boundary point;  "
                        r"red arrow: $\dot B<0$ (the state can leave $\mathcal{C}$)", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "invariance.png", dpi=170)
    plt.close(fig)


def fig_relaxations():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    # (a) interval vs linear bounds for tanh on [l,u]
    l, u = -0.5, 1.8
    x = np.linspace(l, u, 200)
    ax = axes[0]
    ax.plot(x, np.tanh(x), "k", lw=2, label=r"$\tanh(x)$")
    ax.fill_between([l, u], np.tanh(l), np.tanh(u), color=GREY, alpha=0.2, label="interval (IBP) box")
    dmin = min(1 - np.tanh(l) ** 2, 1 - np.tanh(u) ** 2)
    dmax = 1.0
    ax.plot(x, np.tanh(l) + dmin * (x - l), "--", color=BLUE, label="mean-value lines (slope $d_{min}$ / $d_{max}$)")
    ax.plot(x, np.tanh(l) + dmax * (x - l), "--", color=BLUE)
    ax.plot(x, np.tanh(u) - dmax * (u - x), ":", color=ORANGE)
    ax.plot(x, np.tanh(u) - dmin * (u - x), ":", color=ORANGE)
    ax.set_ylim(-1, 1.3)
    ax.set_title("(a) enclosing tanh on [l, u]", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    # (b) dependency problem
    ax = axes[1]
    w = np.linspace(0, 1, 50)
    ax.plot(w, 2 * w, color=RED, lw=2, label=r"IBP width of $x-x$ for $x\in[a,a+w]$")
    ax.plot(w, 0 * w, color=GREEN, lw=2, label="true width (always 0)")
    ax.plot(w, w, color=BLUE, lw=2, label=r"IBP width of $x(1-x)$, $x\in[0.5-\frac{w}{2},0.5+\frac{w}{2}]$")
    ax.plot(w, w ** 2 / 4, color=BLUE, lw=2, ls=":", label=r"true width of $x(1-x)$ (= $w^2/4$)")
    ax.set_xlabel("box width w")
    ax.set_title("(b) the dependency problem of IBP", fontsize=10)
    ax.legend(fontsize=7)
    # (c) branch and bound: bound gap shrinks with box size
    ax = axes[2]
    f = lambda z: np.sin(3 * z) * np.exp(-0.3 * z) + 0.35
    z = np.linspace(0, 3, 400)
    ax.plot(z, f(z), "k", lw=1.8, label="true function")
    ax.axhline(0, color="k", lw=0.6)
    for depth, col in [(1, GREY), (3, BLUE), (5, GREEN)]:
        edges = np.linspace(0, 3, 2 ** depth + 1)
        for a, b in zip(edges[:-1], edges[1:]):
            seg = f(np.linspace(a, b, 50))
            L = 3.4 * np.exp(-0.3 * a)  # Lipschitz-style bound
            lb = f((a + b) / 2) - L * (b - a) / 2
            ax.plot([a, b], [lb, lb], color=col, lw=1.4)
    ax.plot([], [], color=GREY, label="lower bounds, 2 boxes")
    ax.plot([], [], color=BLUE, label="8 boxes")
    ax.plot([], [], color=GREEN, label="32 boxes")
    ax.set_ylim(-3, 1.6)
    ax.set_title("(c) splitting boxes tightens lower bounds", fontsize=10)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "relaxations.png", dpi=170)
    plt.close(fig)


def fig_history():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for ax, name in zip(axes, ["inverted_pendulum", "dubins_car"]):
        meta = json.loads((ROOT / "results" / name / "certificate.json").read_text())
        hist = [h for h in meta["history"] if isinstance(h.get("round"), int)]
        r = [h["round"] for h in hist]
        ax.plot(r, [max(h["counterexamples"], 0.5) for h in hist], "o-", color=RED, label="counterexamples found (0 plotted at 0.5)")
        ax.set_yscale("log")
        ax.set_xlabel("CGIS round")
        ax.set_ylabel("counterexamples (log)")
        ax2 = ax.twinx()
        ax2.plot(r, [100 * h.get("proven_volume_fraction", np.nan) for h in hist], "s--", color=BLUE,
                 label="% of domain volume proven")
        ax2.set_ylabel("% volume proven", color=BLUE)
        ax.set_title(f"{name}: final status {meta['status']}", fontsize=10)
        ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "cgis_history.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    fig_invariance()
    fig_relaxations()
    fig_history()
    print("figures in", OUT)
