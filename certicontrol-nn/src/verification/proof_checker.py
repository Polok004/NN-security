"""Re-checks a saved safety certificate from scratch.

A certificate (``certificate.pt``) stores the network weights and, for every
property, the list of leaf boxes of the branch-and-bound tree together with the
method (IBP / CROWN / Z3) that discharged each leaf.  The checker

1. rebuilds the networks from the stored weights (no training code involved);
2. checks *coverage*: all leaves lie inside the region they are meant to cover
   and their volumes sum to the region's volume (leaves come from recursive
   bisection, hence are interior-disjoint, so equal volume <=> full cover);
3. re-proves every leaf with its recorded method, using freshly computed bounds.

The certificate is accepted only if all three steps succeed for P0-P3.
"""
from __future__ import annotations

import gzip
import io
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List

import torch

from src.dynamics.continuous_system import make_system
from src.models.certificate import CertificateModule, build_models
from src.verification.lirpa_verifier import METHOD_CODES, BoundEngine
from src.verification.smt_solver import z3_check_linear_bounds


def load_certificate(path, map_location="cpu"):
    """Load ``certificate.pt`` (or its gzip-compressed ``certificate.pt.gz``)."""
    path = Path(path)
    if not path.exists() and path.with_name(path.name + ".gz").exists():
        path = path.with_name(path.name + ".gz")
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as fh:
            payload = torch.load(io.BytesIO(fh.read()), map_location=map_location, weights_only=False)
    else:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    cfg = payload["config"]
    system = make_system(cfg["system"])
    mc = cfg.get("model", {})
    ctrl, bar = build_models(system, tuple(mc.get("ctrl_hidden", (32, 32))),
                             tuple(mc.get("barrier_hidden", (32, 32))))
    ctrl.load_state_dict(payload["controller"])
    bar.load_state_dict(payload["barrier"])
    cert = CertificateModule(system, ctrl, bar, payload["gamma"]).eval()
    return cert, payload


@dataclass
class CheckReport:
    ok: bool
    details: Dict[str, Dict] = field(default_factory=dict)
    seconds: float = 0.0


def _volume(lo, hi, dims):
    return torch.prod((hi - lo)[:, dims], dim=1)


class ProofChecker:
    def __init__(self, cert: CertificateModule, tol: float = 1e-8, rel_vol_tol: float = 1e-9):
        self.engine = BoundEngine(cert, use_crown=True)
        self.system = self.engine.system
        self.tol = tol
        self.rel_vol_tol = rel_vol_tol

    # ------------------------------------------------------------ helpers
    def _coverage(self, lo, hi, region_lo, region_hi):
        dims = [d for d in range(lo.shape[1]) if float(region_hi[d] - region_lo[d]) > 0]
        inside = bool(((lo >= region_lo - 1e-12) & (hi <= region_hi + 1e-12)).all())
        vol = float(_volume(lo, hi, dims).sum())
        target = float(torch.prod((region_hi - region_lo)[dims]))
        covered = abs(vol - target) <= self.rel_vol_tol * target
        return inside and covered, {"inside": inside, "volume": vol, "target_volume": target}

    def _passes(self, olo, ohi, lo, hi, mode):
        B_ub, B_lb, c_lb = ohi[:, 0], olo[:, 0], olo[:, 1]
        if mode == "init":
            return (B_lb > self.tol) | self.engine.outside_init(lo, hi)
        outside = B_ub < -self.tol
        if mode == "boundary":
            return outside
        return outside | ((c_lb >= self.tol) & ~self.engine.unsafe_may_intersect(lo, hi))

    def _recheck_leaves(self, lo, hi, method, mode):
        ok = torch.zeros(lo.shape[0], dtype=torch.bool)
        m_ibp = method == METHOD_CODES["ibp"]
        if m_ibp.any():
            olo, ohi = self.engine.ibp(lo[m_ibp], hi[m_ibp])
            ok[m_ibp] = self._passes(olo, ohi, lo[m_ibp], hi[m_ibp], mode)
        m_cr = method == METHOD_CODES["crown"]
        if m_cr.any():
            olo, ohi = self.engine.ibp(lo[m_cr], hi[m_cr])
            cr = self.engine.crown(lo[m_cr], hi[m_cr])
            olo, ohi = torch.maximum(olo, cr["lb"]), torch.minimum(ohi, cr["ub"])
            ok[m_cr] = self._passes(olo, ohi, lo[m_cr], hi[m_cr], mode)
        m_z3 = torch.nonzero(method == METHOD_CODES["z3"]).flatten()
        if m_z3.numel() > 0:
            cr = self.engine.crown(lo[m_z3], hi[m_z3])
            for j, i in enumerate(m_z3.tolist()):
                r = z3_check_linear_bounds(self.system, lo[i], hi[i], cr["lA"][j], cr["lbias"][j],
                                           cr["uA"][j], cr["ubias"][j], mode, self.tol, 20000)
                ok[i] = r.status == "unsat"
        return ok

    def _check_part(self, part, region_lo, region_hi, mode):
        lo, hi = part["leaves_lo"].double(), part["leaves_hi"].double()
        cov_ok, cov = self._coverage(lo, hi, region_lo, region_hi)
        ok = self._recheck_leaves(lo, hi, part["leaves_method"], mode)
        n_fail = int((~ok).sum())
        res = {"leaves": int(lo.shape[0]), "coverage_ok": cov_ok, **cov, "failed_leaves": n_fail,
               "recorded_counterexamples": int(part["counterexamples"].shape[0]),
               "recorded_unknown": int(part["unknown_lo"].shape[0])}
        passed = cov_ok and n_fail == 0 and res["recorded_counterexamples"] == 0 and res["recorded_unknown"] == 0
        res["ok"] = passed
        return passed, res

    # --------------------------------------------------------------- main
    def check(self, verification: Dict) -> CheckReport:
        t0 = time.time()
        s = self.system
        lo, hi = s.domain.tensor()
        details = {}
        all_ok = True
        ok, details["P1+P2 domain"] = self._check_part(verification["domain"], lo, hi, "domain")
        all_ok &= ok
        faces = s.boundary_faces()
        if len(verification["faces"]) != len(faces):
            all_ok = False
            details["P3 faces"] = {"ok": False, "reason": "missing faces"}
        for (d, side, v), part in zip(faces, verification["faces"]):
            flo, fhi = lo.clone(), hi.clone()
            flo[d] = v
            fhi[d] = v
            ok, details[f"P3 face x{d}={v:+.3g}"] = self._check_part(part, flo, fhi, "boundary")
            all_ok &= ok
        if "init" not in verification:
            all_ok = False
            details["P0 init"] = {"ok": False, "reason": "missing"}
        else:
            ilo, ihi = s.init_set.tensor()
            ok, details["P0 init"] = self._check_part(verification["init"], ilo, ihi, "init")
            all_ok &= ok
        return CheckReport(bool(all_ok), details, time.time() - t0)
