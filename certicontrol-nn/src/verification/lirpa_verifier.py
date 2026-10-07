"""Sound bound computation + branch-and-bound certification of NCBF conditions.

Certified properties (for B = B_phi, cond = dB/dt + gamma B under u = pi(x)):

  (P1) Lie / invariance:  for all x in D:  B(x) >= 0  =>  cond(x) >= 0
  (P2) Safety:            for all x in D with x in X_u:  B(x) < 0
  (P3) Containment:       for all x on the (non-periodic) faces of D:  B(x) < 0
  (P0) Non-triviality:    for all x in X_0:  B(x) > 0

(P1)-(P3) imply that C = {x in D : B(x) >= 0} is forward invariant under the
neural controller and disjoint from the unsafe set; (P0) certifies that every
trajectory starting in the initial set X_0 stays safe forever (and rules out
the vacuous certificate C = {}).  All three are checked by
the same branch-and-bound engine: a box is *discharged* if sound bounds show
the property on the whole box; otherwise it is either falsified by a concrete
point, handed to Z3 (pointwise reasoning over CROWN's linear bounds and the
exact unsafe set), or split.
Bounds come from our own interval arithmetic (IBP) and from auto_LiRPA
(CROWN linear relaxation); both are sound, so their intersection is used.
"""
from __future__ import annotations

import copy
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from src.verification.interval import Interval
from src.verification.smt_solver import is_violation, z3_check_linear_bounds

METHOD_CODES = {"ibp": 0, "crown": 1, "z3": 2}
METHOD_NAMES = {v: k for k, v in METHOD_CODES.items()}


# ---------------------------------------------------------------------------
def verify_state_box(closed_loop_model: torch.nn.Module, x_center: torch.Tensor, eps: float) -> bool:
    """Spec-level helper: CROWN lower bound of every output over the L_inf box
    [x_center - eps, x_center + eps]; True iff all lower bounds are >= 0."""
    from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

    x_center = x_center.reshape(1, -1) if x_center.dim() == 1 else x_center
    bounded_model = BoundedModule(closed_loop_model, torch.empty_like(x_center),
                                  verbose=False, device=x_center.device)
    ptb = PerturbationLpNorm(norm=float("inf"), eps=eps)
    bounded_x = BoundedTensor(x_center, ptb)
    lb, ub = bounded_model.compute_bounds(x=(bounded_x,), method="CROWN")
    return bool((lb >= 0).all().item())


# ---------------------------------------------------------------------------
class BoundEngine:
    """Computes sound bounds of [B, cond] over batches of boxes."""

    def __init__(self, cert_module, use_crown: bool = True, crown_batch: int = 2048):
        self.model = copy.deepcopy(cert_module).double().eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.system = self.model.system
        self.use_crown = use_crown
        self.crown_batch = crown_batch
        self._lirpa = None
        self.crown_failed = False

    @torch.no_grad()
    def ibp(self, lo, hi):
        out = self.model(Interval(lo, hi))
        return out.lo, out.hi

    def _get_lirpa(self, n_state):
        if self._lirpa is None:
            from auto_LiRPA import BoundedModule
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._lirpa = BoundedModule(self.model, torch.zeros(1, n_state, dtype=torch.float64),
                                            verbose=False, device="cpu")
        return self._lirpa

    def crown(self, lo, hi):
        # auto_LiRPA allocates some intermediate buffers with the default
        # dtype; run it with float64 as default so the whole pass is float64.
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            return self._crown(lo, hi)
        finally:
            torch.set_default_dtype(prev)

    def _crown(self, lo, hi):
        """CROWN bounds plus the linear bounding functions (A matrices)."""
        from auto_LiRPA import BoundedTensor, PerturbationLpNorm
        lirpa = self._get_lirpa(lo.shape[1])
        out_name, in_name = lirpa.output_name[0], lirpa.input_name[0]
        res = {k: [] for k in ["lb", "ub", "lA", "lbias", "uA", "ubias"]}
        for i in range(0, lo.shape[0], self.crown_batch):
            l, h = lo[i:i + self.crown_batch], hi[i:i + self.crown_batch]
            ptb = PerturbationLpNorm(norm=float("inf"), x_L=l, x_U=h)
            x = BoundedTensor(0.5 * (l + h), ptb)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lb, ub, A = lirpa.compute_bounds(x=(x,), method="CROWN", return_A=True,
                                                 needed_A_dict={out_name: [in_name]})
            a = A[out_name][in_name]
            res["lb"].append(lb.detach())
            res["ub"].append(ub.detach())
            for k in ["lA", "lbias", "uA", "ubias"]:
                res[k].append(a[k].detach())
        return {k: torch.cat(v) for k, v in res.items()}

    @torch.no_grad()
    def outside_init(self, lo, hi):
        """True where the box provably misses X_0 (some g_j < 0 on the whole box)."""
        gs = self.system.init_constraints(Interval(lo, hi))
        if not gs:
            return torch.zeros(lo.shape[0], dtype=torch.bool)
        return torch.stack([g.hi[:, 0] < 0 for g in gs], 0).any(0)

    @torch.no_grad()
    def unsafe_may_intersect(self, lo, hi):
        pieces = self.system.unsafe_pieces(Interval(lo, hi))
        return torch.stack([p.lo[:, 0] <= 0 for p in pieces], 0).any(0)


# ---------------------------------------------------------------------------
@dataclass
class BBResult:
    mode: str
    leaves_lo: torch.Tensor
    leaves_hi: torch.Tensor
    leaves_method: torch.Tensor
    leaves_outside: torch.Tensor           # True if discharged because B<0 on whole box
    counterexamples: torch.Tensor
    unknown_lo: torch.Tensor
    unknown_hi: torch.Tensor
    n_processed: int
    n_z3_calls: int
    n_z3_proved: int
    n_z3_spurious: int
    seconds: float
    exhausted_budget: bool

    @property
    def certified(self):
        return self.counterexamples.shape[0] == 0 and self.unknown_lo.shape[0] == 0 and not self.exhausted_budget


@dataclass
class VerificationResult:
    domain: BBResult
    faces: List[BBResult]
    gamma: float
    seconds: float
    init: Optional[BBResult] = None
    _domain_volume: float = 0.0

    def parts(self) -> List[BBResult]:
        return [self.domain] + list(self.faces) + ([self.init] if self.init is not None else [])

    @property
    def certified(self) -> bool:
        return self.init is not None and all(p.certified for p in self.parts())

    @property
    def status(self) -> str:
        if self.certified:
            return "CERTIFIED"
        if self.counterexamples().shape[0] > 0:
            return "FALSIFIED"
        return "UNKNOWN"

    def counterexamples(self) -> torch.Tensor:
        return torch.cat([p.counterexamples for p in self.parts()])

    def unknown_boxes(self):
        lo = torch.cat([p.unknown_lo for p in self.parts()])
        hi = torch.cat([p.unknown_hi for p in self.parts()])
        return lo, hi

    def proven_volume_fraction(self) -> float:
        """Fraction of vol(D) covered by leaves on which P1 and P2 are proven."""
        d = self.domain
        if d.leaves_lo.shape[0] == 0:
            return 0.0
        w = (d.leaves_hi - d.leaves_lo).double()
        vol = torch.prod(w, 1).sum()
        return round(float(vol / self._domain_volume), 6) if self._domain_volume else 0.0

    def summary(self) -> Dict:
        d = self.domain
        m = d.leaves_method
        return {
            "status": self.status,
            "gamma": self.gamma,
            "seconds": round(self.seconds, 2),
            "domain_boxes_processed": d.n_processed,
            "domain_leaves": int(m.numel()),
            "proven_volume_fraction": self.proven_volume_fraction(),
            "leaves_by_method": {k: int((m == v).sum()) for k, v in METHOD_CODES.items()},
            "leaves_outside_safe_set": int(d.leaves_outside.sum()),
            "face_leaves": int(sum(f.leaves_method.numel() for f in self.faces)),
            "init_leaves": int(self.init.leaves_method.numel()) if self.init is not None else 0,
            "init_set_certified": bool(self.init is not None and self.init.certified),
            "z3_calls": sum(p.n_z3_calls for p in self.parts()),
            "z3_proved": sum(p.n_z3_proved for p in self.parts()),
            "z3_spurious": sum(p.n_z3_spurious for p in self.parts()),
            "counterexamples": int(self.counterexamples().shape[0]),
            "unknown_boxes": int(self.unknown_boxes()[0].shape[0]),
        }


# ---------------------------------------------------------------------------
class BranchAndBoundVerifier:
    def __init__(self, cert_module, use_crown: bool = True, use_z3: bool = True,
                 tol: float = 1e-8, batch_size: int = 8192, max_boxes: int = 2_000_000,
                 max_depth: int = 14, max_cex: int = 512, z3_max_boxes: int = 20000,
                 z3_timeout_ms: int = 5000, z3_depth: int = 6, cex_split_depth: int = 4,
                 smart_split: bool = False, verbose: bool = False):
        self.engine = BoundEngine(cert_module, use_crown=use_crown)
        self.system = self.engine.system
        self.use_z3 = use_z3
        self.tol = tol
        self.batch_size = batch_size
        self.max_boxes = max_boxes
        self.max_depth = max_depth
        self.max_cex = max_cex
        self.z3_max_boxes = z3_max_boxes
        self.z3_timeout_ms = z3_timeout_ms
        self.z3_depth = z3_depth
        self.cex_split_depth = cex_split_depth
        self.smart_split = smart_split
        self.verbose = verbose
        lo, hi = self.system.domain.tensor()
        self.dom_width = hi - lo

    # -------------------------------------------------------------- checks
    def _passes(self, lo_out, hi_out, may_unsafe, mode):
        B_ub = hi_out[:, 0]
        cond_lb = lo_out[:, 1]
        outside = B_ub < -self.tol
        if mode == "init":  # may_unsafe carries "provably outside X_0" here
            return (lo_out[:, 0] > self.tol) | may_unsafe, torch.zeros_like(may_unsafe)
        if mode == "boundary":
            return outside, outside
        return outside | ((cond_lb >= self.tol) & ~may_unsafe), outside

    def _split(self, lo, hi, score=None):
        """Bisect each box along one coordinate.

        Default: the widest coordinate relative to the domain.  When CROWN's
        linear bounds are available, ``score[i, d] = |A_d| * width_d`` (the
        contribution of coordinate d to the bound's looseness) is used instead,
        which concentrates splits on the influential coordinates.
        """
        rel = (hi - lo) / self.dom_width
        if score is None:
            d = rel.argmax(1)
        else:
            # never split below the resolution limit along a coordinate
            score = torch.where(rel > 2.0 ** (-self.max_depth), score, torch.full_like(score, -1.0))
            d = torch.where(score.max(1).values > 0, score.argmax(1), rel.argmax(1))
        mid = 0.5 * (lo.gather(1, d[:, None]) + hi.gather(1, d[:, None]))
        lo2, hi1 = lo.clone(), hi.clone()
        hi1.scatter_(1, d[:, None], mid)
        lo2.scatter_(1, d[:, None], mid)
        return torch.cat([lo, lo2]), torch.cat([hi1, hi])

    # ---------------------------------------------------------------- core
    def run(self, lo: torch.Tensor, hi: torch.Tensor, mode: str = "domain") -> BBResult:
        t0 = time.time()
        queue = deque([(lo.double().reshape(-1, lo.shape[-1]), hi.double().reshape(-1, hi.shape[-1]))])
        n = lo.shape[-1]
        min_rel = 2.0 ** (-self.max_depth)
        L_lo, L_hi, L_m, L_out = [], [], [], []
        cex, small_lo, small_hi = [], [], []
        n_cex = 0
        processed = 0
        z3_calls = z3_proved = z3_spur = 0
        exhausted = False
        while queue:
            # assemble a batch
            parts, size = [], 0
            while queue and size < self.batch_size:
                a, b = queue.popleft()
                take = self.batch_size - size
                if a.shape[0] > take:
                    queue.appendleft((a[take:], b[take:]))
                    a, b = a[:take], b[:take]
                parts.append((a, b))
                size += a.shape[0]
            blo = torch.cat([p[0] for p in parts])
            bhi = torch.cat([p[1] for p in parts])
            processed += blo.shape[0]

            # 1) concrete falsification at the centre and one random point
            pts = torch.cat([0.5 * (blo + bhi), blo + (bhi - blo) * torch.rand_like(blo)])
            viol = is_violation(self.engine.model, self.system, pts, mode)
            if viol.any():
                vb = viol.reshape(2, -1).any(0)
                cex.append(pts[viol])
                n_cex += int(viol.sum())
                # falsified boxes cannot be proven; large ones are still split so
                # that the rest of their volume keeps being explored (and more
                # counterexamples are harvested for CGIS)
                frel = ((bhi[vb] - blo[vb]) / self.dom_width).max(1).values
                big = frel > 2.0 ** (-self.cex_split_depth)
                if big.any() and n_cex < self.max_cex:
                    queue.append(self._split(blo[vb][big], bhi[vb][big]))
                blo, bhi = blo[~vb], bhi[~vb]
            if n_cex >= self.max_cex:
                break
            if blo.shape[0] == 0:
                continue

            # 2) IBP
            if mode == "domain":
                may_unsafe = self.engine.unsafe_may_intersect(blo, bhi)
            elif mode == "init":
                may_unsafe = self.engine.outside_init(blo, bhi)
            else:
                may_unsafe = torch.zeros(blo.shape[0], dtype=torch.bool)
            olo, ohi = self.engine.ibp(blo, bhi)
            ok, outside = self._passes(olo, ohi, may_unsafe, mode)
            self._record(L_lo, L_hi, L_m, L_out, blo[ok], bhi[ok], "ibp", outside[ok])
            rest = ~ok
            blo, bhi, may_unsafe, olo, ohi = blo[rest], bhi[rest], may_unsafe[rest], olo[rest], ohi[rest]

            # 3) CROWN (auto_LiRPA) on what IBP could not discharge
            crown = None
            if blo.shape[0] > 0 and self.engine.use_crown and not self.engine.crown_failed:
                try:
                    crown = self.engine.crown(blo, bhi)
                    olo, ohi = torch.maximum(olo, crown["lb"]), torch.minimum(ohi, crown["ub"])
                    ok, outside = self._passes(olo, ohi, may_unsafe, mode)
                    self._record(L_lo, L_hi, L_m, L_out, blo[ok], bhi[ok], "crown", outside[ok])
                    blo, bhi = blo[~ok], bhi[~ok]
                    crown = {k: v[~ok] for k, v in crown.items()}
                except Exception as e:  # pragma: no cover - library robustness
                    warnings.warn(f"auto_LiRPA CROWN failed ({e}); continuing with IBP only")
                    self.engine.crown_failed = True
                    crown = None

            if blo.shape[0] == 0:
                continue
            rel = ((bhi - blo) / self.dom_width).max(1).values
            score = None
            if crown is not None and self.smart_split:
                if mode == "domain":
                    A = crown["lA"][:, 1, :].abs() + crown["uA"][:, 0, :].abs()
                elif mode == "init":
                    A = crown["lA"][:, 0, :].abs()
                else:
                    A = crown["uA"][:, 0, :].abs()
                score = A * (bhi - blo)

            # 4) Z3 on small boxes: pointwise reasoning over CROWN's linear bounds
            if self.use_z3 and crown is not None:
                cand = torch.nonzero(rel <= 2.0 ** (-self.z3_depth)).flatten()
                keep = torch.ones(blo.shape[0], dtype=torch.bool)
                for i in cand.tolist():
                    if z3_calls >= self.z3_max_boxes:
                        break
                    r = z3_check_linear_bounds(self.system, blo[i], bhi[i], crown["lA"][i], crown["lbias"][i],
                                               crown["uA"][i], crown["ubias"][i], mode, self.tol,
                                               self.z3_timeout_ms)
                    z3_calls += 1
                    if r.status == "unsat":
                        z3_proved += 1
                        keep[i] = False
                        self._record(L_lo, L_hi, L_m, L_out, blo[i:i + 1], bhi[i:i + 1], "z3",
                                     torch.zeros(1, dtype=torch.bool))
                    elif r.status == "sat":
                        x = torch.from_numpy(r.candidate).reshape(1, -1)
                        if bool(is_violation(self.engine.model, self.system, x, mode)[0]):
                            cex.append(x)
                            n_cex += 1
                            keep[i] = False
                        else:
                            z3_spur += 1
                blo, bhi, rel = blo[keep], bhi[keep], rel[keep]
                if score is not None:
                    score = score[keep]
                if blo.shape[0] == 0:
                    continue

            # 5) refine, or give up on boxes at the resolution limit
            small = rel <= min_rel
            if small.any():
                small_lo.append(blo[small])
                small_hi.append(bhi[small])
            if (~small).any():
                if processed >= self.max_boxes:
                    small_lo.append(blo[~small])
                    small_hi.append(bhi[~small])
                    exhausted = True
                    while queue:
                        a, b = queue.popleft()
                        small_lo.append(a)
                        small_hi.append(b)
                    break
                queue.append(self._split(blo[~small], bhi[~small], None if score is None else score[~small]))
            if self.verbose and processed % (20 * self.batch_size) < self.batch_size:
                print(f"  [{mode}] processed={processed} queue={sum(q[0].shape[0] for q in queue)} "
                      f"leaves={sum(x.shape[0] for x in L_lo)} cex={n_cex}")

        if n_cex >= self.max_cex:
            # stopped early: remaining work is simply "unknown"
            while queue:
                a, b = queue.popleft()
                small_lo.append(a)
                small_hi.append(b)

        unk_lo = torch.cat(small_lo) if small_lo else torch.zeros(0, n, dtype=torch.float64)
        unk_hi = torch.cat(small_hi) if small_hi else torch.zeros(0, n, dtype=torch.float64)

        cat = lambda xs, shape: torch.cat(xs) if xs else torch.zeros(shape)
        return BBResult(
            mode=mode,
            leaves_lo=cat(L_lo, (0, n)).double(), leaves_hi=cat(L_hi, (0, n)).double(),
            leaves_method=cat(L_m, (0,)).long(), leaves_outside=cat(L_out, (0,)).bool(),
            counterexamples=cat(cex, (0, n)).double(),
            unknown_lo=unk_lo, unknown_hi=unk_hi,
            n_processed=processed, n_z3_calls=z3_calls, n_z3_proved=z3_proved,
            n_z3_spurious=z3_spur, seconds=time.time() - t0, exhausted_budget=exhausted)

    @staticmethod
    def _record(L_lo, L_hi, L_m, L_out, lo, hi, method, outside):
        if lo.shape[0] == 0:
            return
        L_lo.append(lo)
        L_hi.append(hi)
        L_m.append(torch.full((lo.shape[0],), METHOD_CODES[method], dtype=torch.long))
        L_out.append(outside.bool())

    # ---------------------------------------------------------------- API
    def verify(self) -> VerificationResult:
        t0 = time.time()
        lo, hi = self.system.domain.tensor()
        ilo, ihi = self.system.init_set.tensor()
        init = self.run(ilo, ihi, "init")
        dom = self.run(lo, hi, "domain")
        faces = []
        if dom.counterexamples.shape[0] < self.max_cex:
            for d, side, v in self.system.boundary_faces():
                flo, fhi = lo.clone(), hi.clone()
                flo[d] = v
                fhi[d] = v
                faces.append(self.run(flo, fhi, "boundary"))
        return VerificationResult(dom, faces, self.engine.model.gamma, time.time() - t0, init,
                                  float(torch.prod(hi - lo)))
