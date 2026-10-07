"""Counterexample-Guided Inductive Synthesis (CGIS) of controller + barrier.

    repeat
        train (pi_theta, B_phi) on sampled states  +  counterexample buffer
        verify with branch-and-bound (IBP -> CROWN -> Z3)
        if certified: emit proof certificate; stop
        add validated counterexamples (and their neighbourhoods) to the buffer
"""
from __future__ import annotations

import hashlib
import io
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from src.dynamics.continuous_system import ControlAffineSystem
from src.models.certificate import CertificateModule, build_models
from src.verification.lirpa_verifier import BranchAndBoundVerifier, VerificationResult


@dataclass
class TrainConfig:
    lr: float = 1e-3
    steps_initial: int = 3000
    steps_per_round: int = 800
    batch_domain: int = 4096
    batch_unsafe: int = 1024
    batch_boundary: int = 1024
    batch_init: int = 1024
    batch_cex: int = 1024
    margin: float = 0.02
    lie_band: float = 0.01
    lie_temp: float = 0.01
    w_lie: float = 1.0
    w_unsafe: float = 1.0
    w_boundary: float = 1.0
    w_init: float = 5.0
    w_cex: float = 2.0
    w_nominal: float = 0.0
    w_lipschitz: float = 1e-3      # controller smoothness (tighter bounds, less bang-bang)
    lip_delta: float = 0.01        # finite-difference step (fraction of domain width)
    w_cert_lip: float = 0.0        # penalty on local slopes of [B, cond] near C (verification-friendly)
    cert_lip_target: float = 1.0   # slopes (per unit of normalised state) above this are penalised
    focus_fraction: float = 0.5    # share of domain batch drawn from the band around C
    lr_final_ratio: float = 0.05   # cosine decay of the learning rate inside each phase
    weight_decay: float = 1e-5
    cex_jitter: float = 0.02
    cex_copies: int = 16
    max_buffer: int = 200_000
    log_every: int = 500


@dataclass
class VerifyConfig:
    use_crown: bool = True
    use_z3: bool = True
    tol: float = 1e-8
    batch_size: int = 8192
    max_boxes: int = 2_000_000
    max_depth: int = 14
    max_cex: int = 512
    z3_max_boxes: int = 20000
    z3_timeout_ms: int = 5000
    z3_depth: int = 6
    cex_split_depth: int = 4                # falsified boxes are still split down to this depth
    smart_split: bool = False               # branch on the coordinate with largest |A_d| * width_d
    quick_max_depth: Optional[int] = None   # cheaper verification for intermediate CGIS rounds
    quick_max_boxes: int = 200_000


def weights_sha256(module: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class CGISTrainer:
    def __init__(self, system: ControlAffineSystem, gamma: float, train_cfg: TrainConfig,
                 verify_cfg: VerifyConfig, ctrl_hidden=(32, 32), barrier_hidden=(32, 32), seed: int = 0):
        torch.manual_seed(seed)
        self.system = system
        self.gamma = gamma
        self.tc = train_cfg
        self.vc = verify_cfg
        ctrl, bar = build_models(system, ctrl_hidden, barrier_hidden)
        self.cert = CertificateModule(system, ctrl, bar, gamma)
        self.opt = torch.optim.Adam(self.cert.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
        n = system.state_dim
        self.cex_domain = torch.zeros(0, n)
        self.cex_boundary = torch.zeros(0, n)
        self.cex_init = torch.zeros(0, n)
        self.history: List[Dict] = []

    # ---------------------------------------------------------------- loss
    def losses(self, x_dom, x_uns, x_bd, x_init, x_cex_dom, x_cex_bd, x_cex_init=None):
        tc, sysm, cert = self.tc, self.system, self.cert
        m = tc.margin
        out = {}
        # (P1) Lie condition on the band B >= -delta
        o = cert(x_dom)
        B, cond = o[:, 0], o[:, 1]
        # smooth indicator of the band B >= -delta.  Its gradient w.r.t. B lets
        # the optimiser *shrink* C where the Lie condition cannot be met,
        # instead of only trying to raise dB/dt there.
        band = torch.sigmoid((B + tc.lie_band) / tc.lie_temp)
        out["lie"] = (F.relu(m - cond) * band).sum() / band.detach().sum().clamp(min=1.0)
        # (P2) B < 0 on unsafe states
        out["unsafe"] = F.relu(cert.barrier(sysm.features(x_uns)).squeeze(-1) + m).mean()
        # (P3) B < 0 on domain faces
        out["boundary"] = F.relu(cert.barrier(sysm.features(x_bd)).squeeze(-1) + m).mean()
        # non-triviality: desired initial set inside C
        out["init"] = F.relu(m - cert.barrier(sysm.features(x_init)).squeeze(-1)).mean()
        # CGIS counterexample loss  max(0, -(dB + gamma B))  (+ margin)
        lc = torch.zeros(())
        if x_cex_dom.shape[0] > 0:
            oc = cert(x_cex_dom)
            Bc, cc = oc[:, 0], oc[:, 1]
            bandc = torch.sigmoid((Bc + tc.lie_band) / tc.lie_temp)
            lc = lc + (F.relu(m - cc) * bandc).mean()
            uns = sysm.is_unsafe(x_cex_dom).float()
            lc = lc + (F.relu(Bc + m) * uns).mean()
        if x_cex_bd.shape[0] > 0:
            lc = lc + F.relu(cert.barrier(sysm.features(x_cex_bd)).squeeze(-1) + m).mean()
        if x_cex_init is not None and x_cex_init.shape[0] > 0:
            lc = lc + F.relu(m - cert.barrier(sysm.features(x_cex_init)).squeeze(-1)).mean()
        out["cgis"] = lc
        # controller smoothness: finite-difference Lipschitz penalty
        if tc.w_lipschitz > 0:
            lo, hi = sysm.domain.tensor(x_dom.dtype)
            d = torch.randn_like(x_dom) * tc.lip_delta * (hi - lo)
            f0 = sysm.features(x_dom)
            u0 = cert.controller(f0)
            u1 = cert.controller(sysm.features(x_dom + d))
            span = (sysm.u_max - sysm.u_min).to(x_dom)
            ratio = ((u1 - u0) / span).norm(dim=-1) / (d / (hi - lo)).norm(dim=-1).clamp(min=1e-9)
            out["lip"] = (ratio ** 2).mean()
        else:
            out["lip"] = torch.zeros(())
        # verification-friendly regulariser: local Lipschitz constant of the
        # certificate outputs [B, cond] near C.  Smaller local slopes shrink
        # the gap between CROWN bounds and true values on a box of given size.
        if tc.w_cert_lip > 0:
            lo, hi = sysm.domain.tensor(x_dom.dtype)
            d = torch.randn_like(x_dom) * tc.lip_delta * (hi - lo)
            o2 = cert(x_dom + d)
            dn = (d / (hi - lo)).norm(dim=-1).clamp(min=1e-9)
            slope = ((o2 - o).abs() / dn[:, None])
            w = band.detach()
            out["cert_lip"] = (w * (F.relu(slope[:, 0] - tc.cert_lip_target) ** 2
                                    + F.relu(slope[:, 1] - tc.cert_lip_target) ** 2)).sum() / w.sum().clamp(min=1.0)
            total_extra = tc.w_cert_lip * out["cert_lip"]
        else:
            total_extra = 0.0
        total = (tc.w_lie * out["lie"] + tc.w_unsafe * out["unsafe"] + tc.w_boundary * out["boundary"]
                 + tc.w_init * out["init"] + tc.w_cex * out["cgis"] + tc.w_lipschitz * out["lip"]
                 + total_extra)
        if tc.w_nominal > 0:
            u_nom = sysm.nominal_control(x_dom)
            if u_nom is not None:
                u = cert.controller(sysm.features(x_dom))
                span = (sysm.u_max - sysm.u_min).float()
                out["nominal"] = (((u - u_nom) / span) ** 2).mean()
                total = total + tc.w_nominal * out["nominal"]
        out["total"] = total
        return out

    def _sample_buffer(self, buf, n):
        if buf.shape[0] == 0:
            return buf
        idx = torch.randint(buf.shape[0], (n,))
        return buf[idx]

    @torch.no_grad()
    def _focused_domain_batch(self, n: int) -> torch.Tensor:
        """Uniform samples mixed with samples from the band {B >= -lie_band}."""
        s, tc = self.system, self.tc
        n_focus = int(n * tc.focus_fraction)
        x_uni = s.sample_domain(n - n_focus)
        if n_focus == 0:
            return x_uni
        cand = s.sample_domain(4 * n_focus)
        B = self.cert.barrier(s.features(cand)).squeeze(-1)
        cand = cand[B >= -tc.lie_band]
        if cand.shape[0] == 0:
            return s.sample_domain(n)
        idx = torch.randint(cand.shape[0], (n_focus,))
        return torch.cat([x_uni, cand[idx]])

    def train(self, steps: int, verbose: bool = True):
        tc, s = self.tc, self.system
        self.cert.train()
        for g in self.opt.param_groups:
            g["lr"] = tc.lr
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=max(steps, 1),
                                                           eta_min=tc.lr * tc.lr_final_ratio)
        for it in range(steps):
            L = self.losses(self._focused_domain_batch(tc.batch_domain), s.sample_unsafe_train(tc.batch_unsafe),
                            s.sample_boundary(tc.batch_boundary), s.sample_init(tc.batch_init),
                            self._sample_buffer(self.cex_domain, tc.batch_cex),
                            self._sample_buffer(self.cex_boundary, tc.batch_cex // 4),
                            self._sample_buffer(self.cex_init, tc.batch_cex // 4))
            self.opt.zero_grad()
            L["total"].backward()
            self.opt.step()
            sched.step()
            if verbose and (it % tc.log_every == 0 or it == steps - 1):
                print("    step %5d  " % it + "  ".join(f"{k}={v.item():.4f}" for k, v in L.items()))
        self.cert.eval()

    # -------------------------------------------------------- counterexamples
    def _expand(self, x: torch.Tensor, face: bool = False) -> torch.Tensor:
        if x.shape[0] == 0:
            return x.float()
        s = self.system
        lo, hi = s.domain.tensor(torch.float32)
        x = x.float()
        rep = x.repeat(self.tc.cex_copies, 1)
        noise = torch.randn_like(rep) * self.tc.cex_jitter * (hi - lo)
        noise[: x.shape[0]] = 0.0
        out = torch.max(torch.min(rep + noise, hi), lo)
        if face:  # keep points on their face
            on_lo = (rep - lo).abs() < 1e-9
            on_hi = (rep - hi).abs() < 1e-9
            out = torch.where(on_lo, lo.expand_as(out), out)
            out = torch.where(on_hi, hi.expand_as(out), out)
        return out

    def add_counterexamples(self, res: VerificationResult):
        dom = res.domain.counterexamples
        bd = torch.cat([f.counterexamples for f in res.faces]) if res.faces else torch.zeros(0, self.system.state_dim)
        # unresolved boxes: their centres are "hard" states worth training on
        ulo, uhi = res.domain.unknown_lo, res.domain.unknown_hi
        centres = 0.5 * (ulo + uhi)
        hard = centres[torch.randperm(centres.shape[0])[:2048]]
        self.cex_domain = torch.cat([self.cex_domain, self._expand(dom), self._expand(hard)])[-self.tc.max_buffer:]
        self.cex_boundary = torch.cat([self.cex_boundary, self._expand(bd, face=True)])[-self.tc.max_buffer:]
        if res.init is not None and res.init.counterexamples.shape[0] > 0:
            xi = self._expand(res.init.counterexamples)
            xi = xi[self.system.in_init(xi)]
            self.cex_init = torch.cat([self.cex_init, xi])[-self.tc.max_buffer:]

    # ---------------------------------------------------------------- verify
    def verify(self, verbose: bool = False, quick: bool = False) -> VerificationResult:
        kw = asdict(self.vc)
        qd, qb = kw.pop("quick_max_depth"), kw.pop("quick_max_boxes")
        if quick:
            kw["max_depth"] = qd
            kw["max_boxes"] = qb
            kw["z3_depth"] = min(kw["z3_depth"], qd)
        v = BranchAndBoundVerifier(self.cert, verbose=verbose, **kw)
        return v.verify()

    # ------------------------------------------------------------------ loop
    def run(self, max_rounds: int = 20, verbose: bool = True, checkpoint: Optional[str | Path] = None,
            resume: bool = False) -> Optional[VerificationResult]:
        """CGIS loop.  Intermediate rounds use a cheaper *quick* verification
        (coarser resolution / smaller box budget) whose job is to harvest
        counterexamples; once a quick round finds no counterexample the full
        verification is run, and only its verdict can certify."""
        t0 = time.time()
        start = 0
        if resume and checkpoint is not None and Path(checkpoint).exists():
            start = self.load_checkpoint(checkpoint) + 1
            if verbose:
                print(f"[CGIS] resumed from {checkpoint} at round {start}")
        else:
            if verbose:
                print(f"[CGIS] initial training ({self.tc.steps_initial} steps)")
            self.train(self.tc.steps_initial, verbose)
        res = None
        for r in range(start, max_rounds):
            quick = self.vc.quick_max_depth is not None
            res = self.verify(quick=quick)
            full_run = not quick
            if quick and res.counterexamples().shape[0] == 0:
                res = self.verify(quick=False)
                full_run = True
            summ = res.summary()
            summ["round"] = r
            summ["full_verification"] = full_run
            summ["buffer_size"] = int(self.cex_domain.shape[0] + self.cex_boundary.shape[0] + self.cex_init.shape[0])
            summ["elapsed"] = round(time.time() - t0, 1)
            self.history.append(summ)
            if verbose:
                print(f"[CGIS] round {r}: {json.dumps(summ)}")
            if (res.certified and full_run) or r == max_rounds - 1:
                break   # never train after the last verification: saved weights == verified weights
            self.add_counterexamples(res)
            self.train(self.tc.steps_per_round, verbose)
            if checkpoint is not None:
                self.save_checkpoint(checkpoint, r)
        return res

    def save_checkpoint(self, path, round_idx: int):
        torch.save({"cert": self.cert.state_dict(), "opt": self.opt.state_dict(), "round": round_idx,
                    "cex_domain": self.cex_domain, "cex_boundary": self.cex_boundary, "cex_init": self.cex_init,
                    "history": self.history}, path)

    def load_checkpoint(self, path) -> int:
        ck = torch.load(path, weights_only=False)
        self.cert.load_state_dict(ck["cert"])
        self.opt.load_state_dict(ck["opt"])
        self.cex_domain, self.cex_boundary, self.cex_init = ck["cex_domain"], ck["cex_boundary"], ck["cex_init"]
        self.history = ck["history"]
        return ck["round"]

    # ------------------------------------------------------------------ io
    def save(self, out_dir: str | Path, res: VerificationResult, cfg_dict: Dict):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_certificate(out, self.cert, res, cfg_dict, self.history)


def save_certificate(out: Path, cert: CertificateModule, res: VerificationResult, cfg_dict: Dict,
                     history: List[Dict]):
    d = res.domain
    payload = {
        "controller": cert.controller.state_dict(),
        "barrier": cert.barrier.state_dict(),
        "gamma": cert.gamma,
        "config": cfg_dict,
        "verification": {
            "summary": res.summary(),
            "domain": {k: getattr(d, k) for k in ["leaves_lo", "leaves_hi", "leaves_method", "leaves_outside",
                                                   "counterexamples", "unknown_lo", "unknown_hi"]},
            "faces": [{k: getattr(f, k) for k in ["leaves_lo", "leaves_hi", "leaves_method", "leaves_outside",
                                                  "counterexamples", "unknown_lo", "unknown_hi"]}
                      for f in res.faces],
            "init": {k: getattr(res.init, k) for k in ["leaves_lo", "leaves_hi", "leaves_method",
                                                       "counterexamples", "unknown_lo", "unknown_hi"]},
        },
        "history": history,
    }
    torch.save(payload, out / "certificate.pt")
    meta = {
        "system": cfg_dict.get("system", {}).get("name"),
        "status": res.status,
        "controller_sha256": weights_sha256(cert.controller),
        "barrier_sha256": weights_sha256(cert.barrier),
        "summary": res.summary(),
        "properties": [
            "P1 (invariance): for all x in D, B(x) >= 0 implies dB/dt + gamma*B >= 0 under u = pi(x)",
            "P2 (safety): for all x in D intersect X_unsafe, B(x) < 0",
            "P3 (containment): for all x on the non-periodic faces of D, B(x) < 0",
            "P0 (non-triviality): for all x in X_0, B(x) > 0",
        ],
        "history": history,
    }
    (out / "certificate.json").write_text(json.dumps(meta, indent=2))
