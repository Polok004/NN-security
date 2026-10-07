"""Render figures for a saved certificate.

    python scripts/plot_certified_region.py --system inverted_pendulum
    python scripts/plot_certified_region.py --system dubins_car
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.verification.proof_checker import load_certificate  # noqa: E402
from src.visualization.phase_portrait import (plot_dubins_slice, plot_dubins_trajectories,  # noqa: E402
                                              plot_pendulum)


def pendulum_figures(cert, payload, out: Path):
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax, handles = plot_pendulum(cert, payload["verification"], trajectories=14)
    plt.close(fig)
    fig = ax.figure
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.9)
    ax.set_title("Inverted pendulum: certified invariant set, B&B leaves, closed-loop rollouts", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "pendulum_certified.png", dpi=180)
    fig.savefig(out / "pendulum_certified.pdf")
    plt.close(fig)


def dubins_figures(cert, payload, out: Path):
    thetas = [0.0, np.pi / 2, np.pi, -np.pi / 2]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.2))
    for j, th in enumerate(thetas):
        plot_dubins_slice(cert, th, 0.0, ax=axes[0, j])
        plot_dubins_slice(cert, th, 0.6, ax=axes[1, j])
    status = payload["verification"]["summary"]["status"]
    what = "certified set" if status == "CERTIFIED" else f"barrier super-level set (verifier status: {status})"
    fig.suptitle(rf"Dubins car: slices of the {what} $\{{B\geq 0\}}$ (top $v=0$, bottom $v=0.6$); red = obstacle",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "dubins_slices.png", dpi=150)
    fig.savefig(out / "dubins_slices.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    plot_dubins_trajectories(cert, n=24, T=10.0, ax=ax)
    ax.set_title("Closed-loop rollouts from random states in C", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "dubins_trajectories.png", dpi=180)
    fig.savefig(out / "dubins_trajectories.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="inverted_pendulum")
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cert_path = Path(a.results) / a.system / "certificate.pt"
    cert, payload = load_certificate(cert_path)
    out = Path(a.out) if a.out else cert_path.parent / "figures"
    out.mkdir(parents=True, exist_ok=True)
    print(json.dumps(payload["verification"]["summary"], indent=1))
    if a.system == "inverted_pendulum":
        pendulum_figures(cert, payload, out)
    else:
        dubins_figures(cert, payload, out)
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
