"""Z3 SMT engine for the CGIS loop.

Two sound encodings of the negated safety condition on a state box are offered.

1. ``z3_check_linear_bounds`` (default, used by the branch-and-bound verifier).
   auto_LiRPA/CROWN returns, for the box, linear functions that sandwich the
   network outputs:   lA x + lb <= [B(x), cond(x)] <= uA x + ub.
   Z3 decides, over the *exact* rationals of these coefficients and with the
   unsafe set encoded *exactly* (polynomial arithmetic),

       exists x in box:  uB(x) >= -tol  and  ( lcond(x) < tol  or  x in X_u ).

   UNSAT proves the condition pointwise on the whole box -- strictly stronger
   than the box-level test (which needs max B < 0 or min cond >= 0 on the whole
   box), because Z3 reasons about the disjunction point by point.

2. ``z3_check_box`` (full graph relaxation, practical only for small nets).
   The whole closed-loop computation is encoded in QF_LRA: affine layers with
   coefficients rounded to a 2^-30 dyadic grid (each constraint loosened by the
   rounding error), smooth non-linearities y=f(x) enclosed by the mean-value
   inequalities  f(l)+dmin(x-l) <= y <= f(l)+dmax(x-l),
   f(u)-dmax(u-x) <= y <= f(u)-dmin(u-x), and products by McCormick envelopes.

In both cases a SAT model is only a *candidate*: it is validated by exact
float64 evaluation of the real network and reported as genuine or spurious.
"""
from __future__ import annotations

import math
import time
from fractions import Fraction
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import z3

from src.verification.interval import Interval

_K = 30                 # coefficients are rounded to the dyadic grid 2^-K ...
_GRID = 2 ** _K         # ... and every constraint is loosened by the rounding error
_EXTRA = 1e-12          # absorbs float rounding in pre-computed constants (tanh(l), ...)


def _qround(v: float, mode: str = "nearest"):
    """Round v to the grid 2^-K; returns (z3 rational, float value)."""
    s = float(v) * _GRID
    n = math.floor(s) if mode == "down" else math.ceil(s) if mode == "up" else round(s)
    return z3.Q(int(n), _GRID), n / _GRID


class Z3Context:
    """Owns the solver and builds *soundly loosened* linear constraints.

    Constraints are affine forms  sum_i a_i v_i + c  over bounded variables.
    Feeding Z3 raw float coefficients (exact rationals with 2^52 denominators)
    makes simplex pivoting explode, so coefficients are rounded to a 2^-30 grid
    and the constraint is loosened by  sum_i |a_i - a~_i| max|v_i| + |c - c~|,
    which keeps the encoding an over-approximation.
    """

    def __init__(self, timeout_ms: int = 10000):
        self.solver = z3.SolverFor("QF_LRA")
        self.solver.set("timeout", int(timeout_ms))
        self.vars: List = []
        self.mag: List[float] = []

    def new_var(self, lo: float, hi: float, prefix="t") -> int:
        i = len(self.vars)
        v = z3.Real(f"{prefix}_{i}")
        self.vars.append(v)
        lo_q, lo_f = _qround(lo - _EXTRA, "down")
        hi_q, hi_f = _qround(hi + _EXTRA, "up")
        self.mag.append(max(abs(lo_f), abs(hi_f)))
        self.solver.add(v >= lo_q, v <= hi_q)
        return i

    def geq0(self, aff: "Aff"):
        """Z3 formula over-approximating  aff >= 0."""
        terms, err = [], 0.0
        for i, a in aff.coef.items():
            q, af = _qround(a)
            # rounding to the grid + float error made while forming a itself
            err += (abs(a - af) + 1e-13 * abs(a)) * self.mag[i]
            if af != 0.0:
                terms.append(q * self.vars[i])
        cq, cf = _qround(aff.const)
        err += abs(aff.const - cf) + 1e-13 * abs(aff.const) + _EXTRA
        slack, _ = _qround(err * (1 + 1e-9) + 2.0 / _GRID, "up")
        lhs = z3.Sum(terms) + cq if terms else cq
        return lhs + slack >= 0

    def leq0(self, aff: "Aff"):
        return self.geq0(-aff)

    def add(self, *cons):
        self.solver.add(*cons)


class Aff:
    """Affine form  sum_i coef[i] * var_i + const  (float coefficients)."""

    __slots__ = ("coef", "const")

    def __init__(self, coef=None, const=0.0):
        self.coef = dict(coef or {})
        self.const = float(const)

    @staticmethod
    def var(i):
        return Aff({i: 1.0}, 0.0)

    def __add__(self, o):
        if isinstance(o, (int, float)):
            return Aff(self.coef, self.const + o)
        c = dict(self.coef)
        for k, v in o.coef.items():
            c[k] = c.get(k, 0.0) + v
        return Aff(c, self.const + o.const)

    __radd__ = __add__

    def __mul__(self, a: float):
        a = float(a)
        return Aff({k: a * v for k, v in self.coef.items()}, a * self.const)

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, o):
        return self + (-o if isinstance(o, Aff) else -float(o))


class Z3RelaxedVec:
    """A row vector of affine forms over Z3 variables + sound interval bounds."""

    def __init__(self, ctx: Z3Context, affs: List[Aff], box: Interval):
        assert box.lo.shape[0] == 1 and box.lo.shape[1] == len(affs)
        self.ctx = ctx
        self.exprs = list(affs)
        self.box = box

    def _new(self, affs, box):
        return Z3RelaxedVec(self.ctx, affs, box)

    def _var(self, box: Interval, i: int, prefix="t") -> Aff:
        return Aff.var(self.ctx.new_var(float(box.lo[0, i]), float(box.hi[0, i]), prefix))

    def __getitem__(self, idx):
        rows, cols = idx
        return self._new(self.exprs[cols], self.box[:, cols])

    @staticmethod
    def cat(xs):
        ctx = next(x.ctx for x in xs if isinstance(x, Z3RelaxedVec))
        affs, boxes = [], []
        for x in xs:
            affs += x.exprs
            boxes.append(x.box)
        return Z3RelaxedVec(ctx, affs, Interval.cat(boxes))

    def __len__(self):
        return len(self.exprs)

    # ---------------------------------------------------------- arithmetic
    def __add__(self, other):
        if isinstance(other, (int, float)):
            return self._new([e + float(other) for e in self.exprs], self.box + float(other))
        return self._new([a + b for a, b in zip(self.exprs, other.exprs)], self.box + other.box)

    __radd__ = __add__

    def __neg__(self):
        return self._new([-e for e in self.exprs], -self.box)

    def __sub__(self, other):
        return self + (-other if isinstance(other, Z3RelaxedVec) else -float(other))

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return self._new([e * float(other) for e in self.exprs], self.box * float(other))
        out_box = self.box * other.box
        affs = []
        for i, (a, b) in enumerate(zip(self.exprs, other.exprs)):
            al, au = float(self.box.lo[0, i]), float(self.box.hi[0, i])
            bl, bu = float(other.box.lo[0, i]), float(other.box.hi[0, i])
            w = self._var(out_box, i, "mul")
            c = self.ctx
            # McCormick envelope of w = a*b on [al,au] x [bl,bu]
            c.add(c.geq0(w - b * al - a * bl + al * bl),
                  c.geq0(w - b * au - a * bu + au * bu),
                  c.leq0(w - b * au - a * bl + au * bl),
                  c.leq0(w - b * al - a * bu + al * bu))
            affs.append(w)
        return self._new(affs, out_box)

    __rmul__ = __mul__

    def square(self):
        sq = self * self
        return self._new(sq.exprs, self.box.square())

    def linear(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        W = weight.detach().double().cpu().numpy()
        b = None if bias is None else bias.detach().double().cpu().numpy()
        out_box = self.box.linear(weight.detach().double(), None if bias is None else bias.detach().double())
        affs = []
        for j in range(W.shape[0]):
            e = Aff(const=0.0 if b is None else float(b[j]))
            for i in range(W.shape[1]):
                if W[j, i] != 0.0:
                    e = e + self.exprs[i] * float(W[j, i])
            y = self._var(out_box, j, "lin")
            self.ctx.add(self.ctx.geq0(y - e), self.ctx.leq0(y - e))
            affs.append(y)
        return self._new(affs, out_box)

    def scale_shift(self, scale: torch.Tensor, shift: torch.Tensor):
        s = scale.detach().double().cpu().numpy()
        c = shift.detach().double().cpu().numpy()
        affs = [e * float(s[i]) + float(c[i]) for i, e in enumerate(self.exprs)]
        return self._new(affs, self.box.scale_shift(scale.double(), shift.double()))

    # ------------------------------------------------------ nonlinearities
    def _c1(self, f, f_box: Interval, df_box: Interval, prefix):
        """Mean-value relaxation of y = f(x) for each coordinate."""
        affs = []
        c = self.ctx
        for i, x in enumerate(self.exprs):
            l, u = float(self.box.lo[0, i]), float(self.box.hi[0, i])
            y = self._var(f_box, i, prefix)
            fl = float(f(torch.tensor(l, dtype=torch.float64)))
            fu = float(f(torch.tensor(u, dtype=torch.float64)))
            dmin, dmax = float(df_box.lo[0, i]), float(df_box.hi[0, i])
            e = 1e-12
            #  f(l) + dmin (x-l) <= y <= f(l) + dmax (x-l)
            c.add(c.geq0(y - (x - l) * dmin - fl + e), c.leq0(y - (x - l) * dmax - fl - e))
            #  f(u) - dmax (u-x) <= y <= f(u) - dmin (u-x)
            c.add(c.geq0(y + (x * -1.0 + u) * dmax - fu + e), c.leq0(y + (x * -1.0 + u) * dmin - fu - e))
            affs.append(y)
        return self._new(affs, f_box)

    def tanh(self):
        return self._c1(torch.tanh, self.box.tanh(), self.box.dtanh(), "tanh")

    def dtanh(self):
        # f = 1 - tanh^2 ; f' = -2 tanh (1 - tanh^2)
        df = (self.box.tanh() * self.box.dtanh()) * (-2.0)
        return self._c1(lambda z: 1 - torch.tanh(z) ** 2, self.box.dtanh(), df, "dtanh")

    def sin(self):
        return self._c1(torch.sin, self.box.sin(), self.box.cos(), "sin")

    def cos(self):
        return self._c1(torch.cos, self.box.cos(), -self.box.sin(), "cos")


# =====================================================================
@dataclass
class Z3BoxResult:
    status: str                    # "unsat" (proven) | "sat" | "unknown"
    candidate: Optional[np.ndarray] = None
    genuine: bool = False          # candidate validated on the real network
    seconds: float = 0.0


def z3_check_box(cert_module, system, lo: torch.Tensor, hi: torch.Tensor, mode: str = "domain",
                 tol: float = 1e-8, timeout_ms: int = 10000) -> Z3BoxResult:
    """Decide the negated safety condition on one box with Z3.

    mode "domain":   exists x in box:  B(x) >= -tol  and  (cond(x) < tol  or  x unsafe)
    mode "boundary": exists x in box:  B(x) >= -tol
    (mode "init" is only supported by :func:`z3_check_linear_bounds`)
    ``cert_module`` must be a float64 :class:`CertificateModule`.
    """
    t0 = time.time()
    lo = lo.double().reshape(1, -1)
    hi = hi.double().reshape(1, -1)
    ctx = Z3Context(timeout_ms)
    idx = [ctx.new_var(float(lo[0, i]), float(hi[0, i]), "x") for i in range(lo.shape[1])]
    X = Z3RelaxedVec(ctx, [Aff.var(i) for i in idx], Interval(lo, hi))
    with torch.no_grad():
        out = cert_module(X)
    B, cond = out.exprs
    if mode == "domain":
        pieces = system.unsafe_pieces(X)
        unsafe = z3.Or([ctx.leq0(p.exprs[0]) for p in pieces])
        ctx.add(ctx.geq0(B + tol), z3.Or(ctx.leq0(cond - tol), unsafe))
    elif mode == "boundary":
        ctx.add(ctx.geq0(B + tol))
    else:
        raise ValueError(mode)
    res = ctx.solver.check()
    dt = time.time() - t0
    if res == z3.unsat:
        return Z3BoxResult("unsat", seconds=dt)
    if res == z3.unknown:
        return Z3BoxResult("unknown", seconds=dt)
    model = ctx.solver.model()
    cand = np.array([_model_float(model, ctx.vars[i]) for i in idx], dtype=np.float64)
    cand = np.clip(cand, lo[0].numpy(), hi[0].numpy())
    genuine = bool(is_violation(cert_module, system, torch.from_numpy(cand).reshape(1, -1), mode)[0])
    return Z3BoxResult("sat", cand, genuine, dt)


def _model_float(model, x):
    v = model.eval(x, model_completion=True)
    if z3.is_algebraic_value(v):
        v = v.approx(20)
    return float(Fraction(v.numerator_as_long(), v.denominator_as_long()))


@torch.no_grad()
def is_violation(cert_module, system, x: torch.Tensor, mode: str = "domain") -> torch.Tensor:
    """Exact pointwise check of the certificate conditions (float64)."""
    out = cert_module(x.double())
    B, cond = out[:, 0], out[:, 1]
    if mode == "boundary":
        return B >= 0
    if mode == "init":
        return (B <= 0) & system.in_init(x.double())
    return (B >= 0) & ((cond < 0) | system.is_unsafe(x.double()))


class Z3ExactVec:
    """Row vector of exact Z3 real expressions (for polynomial set definitions)."""

    def __init__(self, exprs):
        self.exprs = list(exprs)

    def __getitem__(self, idx):
        return Z3ExactVec(self.exprs[idx[1]])

    def __add__(self, o):
        if isinstance(o, (int, float)):
            return Z3ExactVec([e + _exact(o) for e in self.exprs])
        return Z3ExactVec([a + b for a, b in zip(self.exprs, o.exprs)])

    __radd__ = __add__

    def __mul__(self, o):
        if isinstance(o, (int, float)):
            return Z3ExactVec([_exact(o) * e for e in self.exprs])
        return Z3ExactVec([a * b for a, b in zip(self.exprs, o.exprs)])

    __rmul__ = __mul__

    def __neg__(self):
        return Z3ExactVec([-e for e in self.exprs])

    def __sub__(self, o):
        return self + (-o if isinstance(o, Z3ExactVec) else -float(o))

    def square(self):
        return Z3ExactVec([e * e for e in self.exprs])


def _exact(v: float):
    f = Fraction(float(v))
    return z3.Q(f.numerator, f.denominator)


def z3_check_linear_bounds(system, lo, hi, lA, lbias, uA, ubias, mode: str = "domain",
                           tol: float = 1e-8, timeout_ms: int = 5000) -> Z3BoxResult:
    """Decide the negated condition for one box using CROWN linear bounds.

    ``lA, uA``: (2, n) and ``lbias, ubias``: (2,) such that for all x in box
    lA x + lbias <= [B(x), cond(x)] <= uA x + ubias.
    Returns status "unsat" (proven), "sat" (candidate in ``candidate``) or
    "unknown".  Genuineness of the candidate is checked by the caller.
    """
    t0 = time.time()
    lo = [float(v) for v in lo.reshape(-1)]
    hi = [float(v) for v in hi.reshape(-1)]
    n = len(lo)
    solver = z3.Solver()
    solver.set("timeout", int(timeout_ms))
    xs = [z3.Real(f"x{i}") for i in range(n)]
    for i in range(n):
        solver.add(xs[i] >= _exact(lo[i]), xs[i] <= _exact(hi[i]))

    def lin(A, b):
        return z3.Sum([_exact(A[i]) * xs[i] for i in range(n)]) + _exact(b)

    uA, ubias, lA, lbias = (t.detach().double().cpu().numpy() for t in (uA, ubias, lA, lbias))
    X = Z3ExactVec(xs)
    if mode == "init":
        # exists x in box with x in X_0 and  lower-bound(B)(x) <= tol
        solver.add(lin(lA[0], lbias[0]) <= _exact(tol))
        for gj in system.init_constraints(X):
            solver.add(gj.exprs[0] >= 0)
    else:
        solver.add(lin(uA[0], ubias[0]) >= _exact(-tol))
    if mode == "domain":
        c_lo = lin(lA[1], lbias[1])
        unsafe = [p.exprs[0] <= 0 for p in system.unsafe_pieces(X)]
        solver.add(z3.Or([c_lo < _exact(tol)] + unsafe))
    res = solver.check()
    dt = time.time() - t0
    if res == z3.unsat:
        return Z3BoxResult("unsat", seconds=dt)
    if res == z3.unknown:
        return Z3BoxResult("unknown", seconds=dt)
    model = solver.model()
    cand = np.array([_model_float(model, x) for x in xs], dtype=np.float64)
    cand = np.clip(cand, lo, hi)
    return Z3BoxResult("sat", cand, False, dt)


def extract_counterexample_z3(cert_module, system, x_box_min, x_box_max, mode: str = "domain",
                              timeout_ms: int = 5000) -> Optional[np.ndarray]:
    """Return a *validated* counterexample inside the box, or None.

    CROWN linear bounds for the box are computed with auto_LiRPA and handed to
    Z3 together with the exact unsafe set.  None means UNSAT (box proven
    safe), a spurious candidate, or a timeout.
    """
    from src.verification.lirpa_verifier import BoundEngine  # local: avoid import cycle

    eng = BoundEngine(cert_module)
    lo = torch.as_tensor(x_box_min, dtype=torch.float64).reshape(1, -1)
    hi = torch.as_tensor(x_box_max, dtype=torch.float64).reshape(1, -1)
    cr = eng.crown(lo, hi)
    r = z3_check_linear_bounds(system, lo[0], hi[0], cr["lA"][0], cr["lbias"][0], cr["uA"][0],
                               cr["ubias"][0], mode=mode, timeout_ms=timeout_ms)
    if r.status != "sat":
        return None
    x = torch.from_numpy(r.candidate).reshape(1, -1)
    return r.candidate if bool(is_violation(eng.model, system, x, mode)[0]) else None
