"""Independently re-check a saved certificate (coverage + every leaf re-proven).

    python scripts/check_certificate.py results/inverted_pendulum/certificate.pt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.verification.proof_checker import ProofChecker, load_certificate  # noqa: E402


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/inverted_pendulum/certificate.pt"
    cert, payload = load_certificate(path)
    rep = ProofChecker(cert).check(payload["verification"])
    print(json.dumps(rep.details, indent=2))
    print("CERTIFICATE", "ACCEPTED" if rep.ok else "REJECTED", f"({rep.seconds:.1f}s)")
    sys.exit(0 if rep.ok else 1)


if __name__ == "__main__":
    main()
