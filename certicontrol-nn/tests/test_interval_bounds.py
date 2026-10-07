"""Soundness tests: computed bounds must never be violated by actual function values.

A lower bound that *overestimates* the true minimum (or an upper bound that
underestimates the maximum) would make the verifier unsound, so every test
here compares bounds with dense samples of the exact function.
"""
import math

import pytest
import torch

from src import ops
from src.dynamics.continuous_system import DubinsCar, InvertedPendulum
from src.models.certificate import CertificateModule, build_models
from src.models.neural_barrier import NeuralControlBarrier
from src.verification.interval import Interval
from src.verification.lirpa_verifier import BoundEngine, verify_state_box

torch.manual_seed(0)
DT = torch.float64


def _rand_box(n, d, scale=3.0, max_w=2.0):
    c = (torch.rand(n, d, dtype=DT) - 0.5) * 2 * scale
    w = torch.rand(n, d, dtype=DT) * max_w
    return c - w / 2, c + w / 2


def _samples(lo, hi, k=2000):
    u = torch.rand(lo.shape[0], k, lo.shape[1], dtype=DT)
    s = lo[:, None] + (hi - lo)[:, None] * u
    # include the corners of each box's first two coordinates
    s[:, 0], s[:, 1] = lo, hi
    return s


def _assert_encloses(lo_b, hi_b, vals):
    assert torch.all(vals >= lo_b[:, None]), (vals - lo_b[:, None]).min()
    assert torch.all(vals <= hi_b[:, None]), (hi_b[:, None] - vals).min()


@pytest.mark.parametrize("fn", ["tanh", "sin", "cos", "dtanh", "square"])
def test_elementwise_ops_are_sound(fn):
    lo, hi = _rand_box(400, 3, scale=6.0, max_w=8.0)
    I = Interval(lo, hi)
    out = getattr(I, fn)()
    s = _samples(lo, hi)
    ref = {"tanh": torch.tanh, "sin": torch.sin, "cos": torch.cos,
           "dtanh": lambda z: 1 - torch.tanh(z) ** 2, "square": lambda z: z * z}[fn](s)
    _assert_encloses(out.lo, out.hi, ref)


def test_sin_attains_extrema():
    I = Interval(torch.tensor([[1.0]], dtype=DT), torch.tensor([[2.0]], dtype=DT))  # contains pi/2
    out = I.sin()
    assert out.hi.item() >= 1.0 - 1e-15
    I = Interval(torch.tensor([[-2.0]], dtype=DT), torch.tensor([[-1.0]], dtype=DT))  # contains -pi/2
    assert I.sin().lo.item() <= -1.0 + 1e-15


def test_binary_and_linear_ops_are_sound():
    lo, hi = _rand_box(300, 4)
    lo2, hi2 = _rand_box(300, 4)
    a, b = Interval(lo, hi), Interval(lo2, hi2)
    sa, sb = _samples(lo, hi, 1500), _samples(lo2, hi2, 1500)
    _assert_encloses((a * b).lo, (a * b).hi, sa * sb)
    _assert_encloses((a + b).lo, (a + b).hi, sa + sb)
    _assert_encloses((a - b).lo, (a - b).hi, sa - sb)
    _assert_encloses((1.0 - a).lo, (1.0 - a).hi, 1.0 - sa)
    _assert_encloses((-2.5 * a).lo, (-2.5 * a).hi, -2.5 * sa)
    W, bias = torch.randn(6, 4, dtype=DT), torch.randn(6, dtype=DT)
    y = a.linear(W, bias)
    _assert_encloses(y.lo, y.hi, sa @ W.t() + bias)
    sc, sh = torch.tensor([2.0, -1.0, 0.5, -3.0], dtype=DT), torch.tensor([1.0, 0.0, -1.0, 2.0], dtype=DT)
    z = a.scale_shift(sc, sh)
    _assert_encloses(z.lo, z.hi, sa * sc + sh)


def test_point_interval_contains_exact_value():
    x = torch.randn(50, 2, dtype=DT)
    s = InvertedPendulum()
    c, b = build_models(s)
    m = CertificateModule(s, c, b, 1.0).double()
    out = m(Interval.point(x))
    y = m(x)
    assert torch.all(out.lo <= y) and torch.all(y <= out.hi)
    assert torch.all(out.hi - out.lo < 1e-9)


@pytest.mark.parametrize("system_cls", [InvertedPendulum, DubinsCar])
def test_closed_loop_ibp_is_sound(system_cls):
    s = system_cls()
    c, b = build_models(s)
    m = CertificateModule(s, c, b, 1.3).double()
    x = s.sample_domain(64, DT)
    w = torch.rand(64, s.state_dim, dtype=DT) * 0.2
    lo, hi = x - w, x + w
    out = m(Interval(lo, hi))
    samp = _samples(lo, hi, 1500)
    vals = m(samp.reshape(-1, s.state_dim)).reshape(64, 1500, 2)
    assert torch.all(vals >= out.lo[:, None] - 0.0)
    assert torch.all(vals <= out.hi[:, None] + 0.0)


@pytest.mark.parametrize("system_cls", [InvertedPendulum, DubinsCar])
def test_crown_bounds_and_linear_bounds_are_sound(system_cls):
    s = system_cls()
    c, b = build_models(s)
    m = CertificateModule(s, c, b, 1.0)
    eng = BoundEngine(m)
    x = s.sample_domain(32, DT)
    w = torch.rand(32, s.state_dim, dtype=DT) * 0.1
    lo, hi = x - w, x + w
    cr = eng.crown(lo, hi)
    samp = _samples(lo, hi, 1000)
    vals = eng.model(samp.reshape(-1, s.state_dim)).reshape(32, 1000, 2)
    # lower bounds never overestimate the true minimum (and dually for upper)
    assert torch.all(cr["lb"] <= vals.min(1).values + 1e-10)
    assert torch.all(cr["ub"] >= vals.max(1).values - 1e-10)
    lin_lo = torch.einsum("bon,bkn->bko", cr["lA"], samp) + cr["lbias"][:, None]
    lin_hi = torch.einsum("bon,bkn->bko", cr["uA"], samp) + cr["ubias"][:, None]
    assert torch.all(lin_lo <= vals + 1e-10) and torch.all(vals <= lin_hi + 1e-10)
    # CROWN should typically be tighter than IBP on small boxes
    ilo, ihi = eng.ibp(lo, hi)
    assert (cr["ub"] - cr["lb"]).mean() <= (ihi - ilo).mean()


def test_forward_mode_lie_derivative_matches_autograd():
    torch.manual_seed(1)
    bar = NeuralControlBarrier(3).double()
    x = torch.randn(20, 3, dtype=DT)
    f = torch.randn(20, 3, dtype=DT)
    g = torch.randn(20, 3, 2, dtype=DT)
    u = torch.randn(20, 2, dtype=DT)
    ref = bar.compute_lie_derivative(x, u, f, g)
    xdot = f + torch.bmm(g, u.unsqueeze(-1)).squeeze(-1)
    _, dB = bar.forward_with_tangent(x, xdot)
    assert torch.allclose(ref, dB, atol=1e-12)


@pytest.mark.parametrize("system_cls", [InvertedPendulum, DubinsCar])
def test_certificate_lie_derivative_matches_autograd(system_cls):
    s = system_cls()
    c, b = build_models(s)
    m = CertificateModule(s, c, b, 0.7).double()
    x = s.sample_domain(100, DT).requires_grad_(True)
    B = m.barrier(s.features(x))
    grad = torch.autograd.grad(B.sum(), x)[0]
    with torch.no_grad():
        u = m.controller(s.features(x))
        xd = s.f(x) + torch.bmm(s.g(x), u.unsqueeze(-1)).squeeze(-1)
        ref = (grad * xd).sum(-1) + 0.7 * B.squeeze(-1)
        out = m(x)
    assert torch.allclose(out[:, 1], ref, atol=1e-10)


def test_controller_respects_actuator_limits():
    s = DubinsCar()
    c, _ = build_models(s)
    with torch.no_grad():
        for p in c.parameters():
            p.mul_(50.0)
    u = c(s.features(s.sample_domain(5000)))
    assert torch.all(u <= s.u_max.float() + 1e-6) and torch.all(u >= s.u_min.float() - 1e-6)
    lo, hi = s.domain.tensor()
    I = c.double()(s.features(Interval(lo[None], hi[None])))
    assert torch.all(I.lo >= s.u_min - 1e-9) and torch.all(I.hi <= s.u_max + 1e-9)


def test_spec_verify_state_box_helper():
    net = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
    with torch.no_grad():
        net[2].bias.fill_(20.0)  # output >= 20 - 8*|w| > 0 everywhere
        net[2].weight.clamp_(-1, 1)
    assert verify_state_box(net, torch.zeros(1, 2), 0.5)
    with torch.no_grad():
        net[2].bias.fill_(-20.0)
    assert not verify_state_box(net, torch.zeros(1, 2), 0.5)
