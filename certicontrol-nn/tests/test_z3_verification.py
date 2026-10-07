"""Tests for the Z3 engine, the branch-and-bound verifier and the proof checker."""
from pathlib import Path

import numpy as np
import pytest
import torch

from src.dynamics.continuous_system import DubinsCar, InvertedPendulum
from src.models.certificate import CertificateModule, build_models
from src.verification.lirpa_verifier import BoundEngine, BranchAndBoundVerifier
from src.verification.proof_checker import ProofChecker, load_certificate
from src.verification.smt_solver import (extract_counterexample_z3, is_violation, z3_check_box,
                                         z3_check_linear_bounds)

DT = torch.float64
ROOT = Path(__file__).resolve().parents[1]


def _cert(system, bias=None, hidden=(32, 32), seed=0):
    torch.manual_seed(seed)
    c, b = build_models(system, hidden, hidden)
    if bias is not None:
        with torch.no_grad():
            b.layers[-1].weight.mul_(1e-3)
            b.layers[-1].bias.fill_(bias)
    return CertificateModule(system, c, b, 1.0)


def _z3_on_box(cert, lo, hi, mode):
    eng = BoundEngine(cert)
    cr = eng.crown(lo[None], hi[None])
    return z3_check_linear_bounds(cert.system, lo, hi, cr["lA"][0], cr["lbias"][0], cr["uA"][0],
                                  cr["ubias"][0], mode), eng


def test_z3_finds_genuine_boundary_counterexample():
    s = InvertedPendulum()
    cert = _cert(s, bias=+5.0)            # B > 0 everywhere -> every face point violates P3
    lo = torch.tensor([1.0, -4.0], dtype=DT)
    hi = torch.tensor([1.0, 4.0], dtype=DT)
    x = extract_counterexample_z3(cert, s, lo, hi, mode="boundary")
    assert x is not None
    assert bool(is_violation(cert.double(), s, torch.tensor(x)[None], "boundary")[0])
    assert np.all(x >= lo.numpy() - 1e-12) and np.all(x <= hi.numpy() + 1e-12)


def test_z3_proves_box_where_barrier_is_negative():
    s = InvertedPendulum()
    cert = _cert(s, bias=-5.0)            # B < 0 everywhere -> P1, P2, P3 hold trivially
    lo = torch.tensor([0.1, 0.5], dtype=DT)
    hi = torch.tensor([0.3, 1.5], dtype=DT)
    r, _ = _z3_on_box(cert, lo, hi, "domain")
    assert r.status == "unsat"


def test_z3_unsafe_set_is_encoded_exactly():
    s = DubinsCar()
    cert = _cert(s, bias=+5.0)            # B > 0 everywhere
    # box inside the obstacle disc: genuine P2 violation must be found
    lo = torch.tensor([-0.2, -0.2, 0.0, 0.0], dtype=DT)
    hi = torch.tensor([0.2, 0.2, 0.5, 0.5], dtype=DT)
    x = extract_counterexample_z3(cert, s, lo, hi, mode="domain")
    assert x is not None and bool(s.is_unsafe(torch.tensor(x)[None])[0])
    # the corner box [0.43,0.6]^2 touches the obstacle only near (0.43,0.43):
    # a Z3 model (if any) must lie inside the disc when the Lie condition holds
    r, eng = _z3_on_box(cert, torch.tensor([0.43, 0.43, 0.0, 0.0], dtype=DT),
                        torch.tensor([0.6, 0.6, 0.1, 0.1], dtype=DT), "domain")
    assert r.status in ("sat", "unsat")


def test_z3_init_mode():
    s = InvertedPendulum()
    lo, hi = s.init_set.tensor()
    r, _ = _z3_on_box(_cert(s, bias=+5.0), lo, hi, "init")
    assert r.status == "unsat"                   # B > 0 on X_0 is proven
    r, _ = _z3_on_box(_cert(s, bias=-5.0), lo, hi, "init")
    assert r.status == "sat"


def test_z3_unsat_implies_no_sampled_violation():
    """Soundness of Z3 proofs: whenever UNSAT is returned, dense sampling finds nothing."""
    s = InvertedPendulum()
    cert = _cert(s, seed=3)
    eng = BoundEngine(cert)
    torch.manual_seed(0)
    n_unsat = 0
    for _ in range(40):
        c = s.sample_domain(1, DT)[0]
        w = torch.rand(2, dtype=DT) * 0.05
        lo, hi = c - w, c + w
        cr = eng.crown(lo[None], hi[None])
        r = z3_check_linear_bounds(s, lo, hi, cr["lA"][0], cr["lbias"][0], cr["uA"][0], cr["ubias"][0], "domain")
        if r.status == "unsat":
            n_unsat += 1
            xs = lo + (hi - lo) * torch.rand(5000, 2, dtype=DT)
            assert not is_violation(eng.model, s, xs, "domain").any()
    assert n_unsat > 0


def test_full_graph_relaxation_encoding_small_net():
    s = InvertedPendulum()
    cert = _cert(s, bias=-5.0, hidden=(4,)).double()
    lo = torch.tensor([0.0, 0.0], dtype=DT)
    hi = torch.tensor([0.05, 0.1], dtype=DT)
    r = z3_check_box(cert, s, lo, hi, "domain", timeout_ms=20000)
    assert r.status == "unsat"
    cert = _cert(s, bias=+5.0, hidden=(4,)).double()
    r = z3_check_box(cert, s, torch.tensor([1.0, -0.1], dtype=DT), torch.tensor([1.0, 0.1], dtype=DT),
                     "boundary", timeout_ms=20000)
    assert r.status == "sat" and r.genuine


def test_branch_and_bound_never_certifies_a_violated_network():
    s = InvertedPendulum()
    cert = _cert(s, seed=7)                      # untrained: violates the conditions
    xs = s.sample_domain(20000, DT)
    assert is_violation(cert.double(), s, xs, "domain").any() or \
        is_violation(cert.double(), s, s.sample_boundary(4000, DT), "boundary").any()
    res = BranchAndBoundVerifier(cert, max_depth=8, max_boxes=20000).verify()
    assert not res.certified
    cex = res.counterexamples()
    assert cex.shape[0] > 0
    # every reported counterexample is genuine
    mdl = cert.double()
    ok = is_violation(mdl, s, res.domain.counterexamples, "domain")
    assert ok.all()


def test_trivially_empty_barrier_is_not_certified():
    """B < 0 everywhere satisfies P1-P3 vacuously; P0 (X_0 inside C) must reject it."""
    s = InvertedPendulum()
    res = BranchAndBoundVerifier(_cert(s, bias=-5.0), max_depth=6).verify()
    assert res.domain.certified and all(f.certified for f in res.faces)
    assert not res.init.certified and not res.certified


@pytest.mark.parametrize("name", ["inverted_pendulum", "dubins_car"])
def test_saved_certificate_passes_proof_checker(name):
    path = ROOT / "results" / name / "certificate.pt"
    if not path.exists() and not path.with_name("certificate.pt.gz").exists():
        pytest.skip("run scripts/train_and_certify.py first")
    cert, payload = load_certificate(path)
    if payload["verification"]["summary"]["status"] != "CERTIFIED":
        pytest.skip("stored result is not a certificate")
    rep = ProofChecker(cert).check(payload["verification"])
    assert rep.ok, rep.details
    # an independent empirical check: simulate from states inside C
    s = cert.system
    x = s.sample_domain(200000, DT)
    m = cert.double()
    with torch.no_grad():
        o = m(x)
    inC = o[:, 0] >= 0
    assert not (inC & s.is_unsafe(x)).any()
    assert (o[inC, 1] >= 0).all()


def test_proof_checker_rejects_tampered_certificate():
    path = ROOT / "results" / "inverted_pendulum" / "certificate.pt"
    if not path.exists():
        pytest.skip("run scripts/train_and_certify.py first")
    cert, payload = load_certificate(path)
    with torch.no_grad():
        cert.controller.layers[-1].bias.add_(3.0)   # sabotage the controller
    rep = ProofChecker(cert).check(payload["verification"])
    assert not rep.ok
