"""Re-run the verifier on stored networks (e.g. with a larger budget) and update the certificate.

    python scripts/reverify.py --system dubins_car --max-boxes 3000000 --max-depth 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.verification.cgis_loop import VerifyConfig, save_certificate  # noqa: E402
from src.verification.lirpa_verifier import BranchAndBoundVerifier  # noqa: E402
from src.verification.proof_checker import ProofChecker, load_certificate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="inverted_pendulum")
    ap.add_argument("--max-boxes", type=int, default=None)
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--z3-max-boxes", type=int, default=None)
    ap.add_argument("--max-cex", type=int, default=None)
    a = ap.parse_args()
    out = ROOT / "results" / a.system
    cert, payload = load_certificate(out / "certificate.pt")
    cfg = payload["config"]
    v = dict(cfg["verify"])
    for k, val in [("max_boxes", a.max_boxes), ("max_depth", a.max_depth), ("z3_max_boxes", a.z3_max_boxes),
                   ("max_cex", a.max_cex)]:
        if val is not None:
            v[k] = val
    fields = set(VerifyConfig.__dataclass_fields__)
    kw = {k: val for k, val in v.items() if k in fields and not k.startswith("quick")}
    t0 = time.time()
    res = BranchAndBoundVerifier(cert, verbose=True, **kw).verify()
    print(json.dumps(res.summary(), indent=1), f"\n{time.time() - t0:.1f}s")
    cfg["verify"] = v
    history = payload.get("history", []) + [dict(res.summary(), round="reverify", full_verification=True)]
    save_certificate(out, cert, res, cfg, history)
    cert2, payload2 = load_certificate(out / "certificate.pt")
    rep = ProofChecker(cert2, tol=v["tol"]).check(payload2["verification"])
    (out / "proof_check.json").write_text(json.dumps({"ok": rep.ok, "seconds": rep.seconds,
                                                      "details": rep.details}, indent=2))
    print("[proof checker]", "ACCEPTED" if rep.ok else "REJECTED")


if __name__ == "__main__":
    main()
