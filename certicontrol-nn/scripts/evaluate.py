"""Ablations and empirical checks for a saved certificate.

    python scripts/evaluate.py --system inverted_pendulum

Writes results/<system>/evaluation.json with
  * verifier ablation (IBP only / IBP+CROWN / IBP+CROWN+Z3) on the final networks,
  * gamma sweep of the certificate,
  * Monte-Carlo closed-loop rollouts (scipy RK45) from X_0 and from C,
  * dense-sampling check of the certificate conditions.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.models.certificate import CertificateModule  # noqa: E402
from src.verification.lirpa_verifier import BranchAndBoundVerifier  # noqa: E402
from src.verification.proof_checker import load_certificate  # noqa: E402
from src.verification.smt_solver import is_violation  # noqa: E402
from src.visualization.phase_portrait import simulate  # noqa: E402


def ablation(cert, vcfg, budget):
    rows = []
    for name, crown, z3 in [("IBP only", False, False), ("IBP + CROWN", True, False),
                            ("IBP + CROWN + Z3 (full)", True, True)]:
        t0 = time.time()
        res = BranchAndBoundVerifier(cert, use_crown=crown, use_z3=z3, max_depth=vcfg["max_depth"],
                                     z3_depth=vcfg["z3_depth"], max_boxes=budget,
                                     batch_size=vcfg["batch_size"], z3_max_boxes=vcfg["z3_max_boxes"]).verify()
        s = res.summary()
        rows.append({"method": name, "status": s["status"], "seconds": round(time.time() - t0, 2),
                     "boxes_processed": s["domain_boxes_processed"], "leaves": s["domain_leaves"],
                     "proven_volume_fraction": s["proven_volume_fraction"], "z3_proved": s["z3_proved"],
                     "unknown_boxes": s["unknown_boxes"], "counterexamples": s["counterexamples"]})
        print(rows[-1])
    return rows


def gamma_sweep(cert, vcfg, gammas, budget):
    rows = []
    for g in gammas:
        c = CertificateModule(cert.system, cert.controller, cert.barrier, g)
        res = BranchAndBoundVerifier(c, max_depth=vcfg["max_depth"], z3_depth=vcfg["z3_depth"], max_boxes=budget,
                                     batch_size=vcfg["batch_size"], z3_max_boxes=vcfg["z3_max_boxes"]).verify()
        s = res.summary()
        rows.append({"gamma": g, "status": s["status"], "counterexamples": s["counterexamples"],
                     "proven_volume_fraction": s["proven_volume_fraction"], "seconds": s["seconds"]})
        print(rows[-1])
    return rows


def rollouts(cert, n, T, seed=0):
    s = cert.system
    rng = np.random.default_rng(seed)
    m = cert.double()
    out = {}
    for name in ["X0", "C"]:
        if name == "X0":
            xs = s.sample_init(n, torch.float64).numpy()
        else:
            cand = s.sample_domain(50 * n, torch.float64)
            with torch.no_grad():
                cand = cand[m.barrier_value(cand).squeeze(-1) >= 0]
            xs = cand[rng.choice(cand.shape[0], n, replace=False)].numpy()
        n_unsafe = n_leave_C = n_leave_D = 0
        min_B = np.inf
        for x0 in xs:
            _, traj = simulate(m, x0, T=T, n_eval=200)
            tt = torch.from_numpy(traj)
            with torch.no_grad():
                B = m.barrier_value(tt).squeeze(-1)
            n_unsafe += bool(s.is_unsafe(tt).any())
            n_leave_C += bool((B < -1e-6).any())
            n_leave_D += bool((~s.in_domain(tt)).any())
            min_B = min(min_B, float(B.min()))
        out[name] = {"rollouts": n, "horizon_s": T, "reached_unsafe": n_unsafe, "left_C": n_leave_C,
                     "left_D": n_leave_D, "min_B_along_rollouts": min_B}
        print(name, out[name])
    return out


def cex_rollouts(cert, cex, n, T):
    """Simulate from verifier counterexamples: does the violation matter dynamically?"""
    s = cert.system
    m = cert.double()
    n = min(n, cex.shape[0])
    left = unsafe = 0
    min_B = np.inf
    for x0 in cex[:n].numpy():
        _, traj = simulate(m, x0, T=T, n_eval=400, max_step=0.002)
        tt = torch.from_numpy(traj)
        with torch.no_grad():
            B = m.barrier_value(tt).squeeze(-1)
        left += bool((B < 0).any())
        unsafe += bool(s.is_unsafe(tt).any())
        min_B = min(min_B, float(B.min()))
    out = {"rollouts": n, "horizon_s": T, "left_C": left, "reached_unsafe": unsafe, "min_B": min_B}
    print("from counterexamples", out)
    return out


def dense_check(cert, n):
    s = cert.system
    m = cert.double()
    x = s.sample_domain(n, torch.float64)
    xb = s.sample_boundary(n // 10, torch.float64)
    xi = s.sample_init(n // 10, torch.float64)
    return {"samples": n,
            "domain_violations": int(is_violation(m, s, x, "domain").sum()),
            "boundary_violations": int(is_violation(m, s, xb, "boundary").sum()),
            "init_violations": int(is_violation(m, s, xi, "init").sum()),
            "fraction_of_D_in_C": float((m.barrier_value(x) >= 0).double().mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="inverted_pendulum")
    ap.add_argument("--rollouts", type=int, default=300)
    ap.add_argument("--horizon", type=float, default=10.0)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--skip-ablation", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(0)
    path = ROOT / "results" / a.system / "certificate.pt"
    cert, payload = load_certificate(path)
    vcfg = payload["config"]["verify"]
    budget = a.budget or vcfg["max_boxes"]
    ev = {"system": a.system}
    if not a.skip_ablation:
        ev["ablation"] = ablation(cert, vcfg, budget)
        ev["gamma_sweep"] = gamma_sweep(cert, vcfg, [0.25, 0.5, 1.0, 2.0, 4.0], budget)
    ev["dense_sampling"] = dense_check(cert, 2_000_000)
    print(ev["dense_sampling"])
    ev["rollouts"] = rollouts(cert, a.rollouts, a.horizon)
    cex = payload["verification"]["domain"]["counterexamples"]
    if cex.shape[0] > 0:
        ev["counterexample_rollouts"] = cex_rollouts(cert, cex, 50, 3.0)
    (path.parent / "evaluation.json").write_text(json.dumps(ev, indent=2))
    print("written", path.parent / "evaluation.json")


if __name__ == "__main__":
    main()
