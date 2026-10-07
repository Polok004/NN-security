"""Sound interval arithmetic backend (Interval Bound Propagation).

An :class:`Interval` holds batched lower / upper bounds ``lo <= x <= hi`` of
shape ``(N, d)``.  All operations return enclosures that contain the image of
the input box under the exact real-valued operation.

Floating-point soundness: computations are meant to run in ``float64``.  Every
operation widens its result outward by a small amount that dominates the
worst-case rounding error of that operation (a few ulps for element-wise ops,
``O(n * ulp * sum|terms|)`` for an ``n``-term dot product), so that the
enclosure remains valid for the real-arithmetic function despite
round-to-nearest.  This is a conservative engineering substitute for directed
rounding.
"""
from __future__ import annotations

import math

import torch

_U = 2.0 ** -52  # float64 unit roundoff (upper bound)
_ELEM_REL = 8 * _U
_ELEM_ABS = 1e-300
_LIB_ABS = 4 * _U  # transcendental library functions are accurate to < 2 ulp on [-1, 1]


def _widen(lo: torch.Tensor, hi: torch.Tensor, rel: float = _ELEM_REL, abs_: float = _ELEM_ABS):
    return lo - (rel * lo.abs() + abs_), hi + (rel * hi.abs() + abs_)


class Interval:
    __array_priority__ = 1000  # make numpy defer to our operators

    def __init__(self, lo: torch.Tensor, hi: torch.Tensor):
        if lo.shape != hi.shape:
            raise ValueError("lo/hi shape mismatch")
        self.lo = lo
        self.hi = hi

    # ------------------------------------------------------------------ utils
    @staticmethod
    def point(x: torch.Tensor) -> "Interval":
        return Interval(x.clone(), x.clone())

    @property
    def shape(self):
        return self.lo.shape

    def center(self):
        return 0.5 * (self.lo + self.hi)

    def radius(self):
        return 0.5 * (self.hi - self.lo)

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        return (x >= self.lo) & (x <= self.hi)

    def __getitem__(self, idx):
        return Interval(self.lo[idx], self.hi[idx])

    def __repr__(self):
        return f"Interval(lo={self.lo}, hi={self.hi})"

    @staticmethod
    def cat(xs):
        los, his = [], []
        for x in xs:
            if isinstance(x, Interval):
                los.append(x.lo)
                his.append(x.hi)
            else:
                los.append(x)
                his.append(x)
        return Interval(torch.cat(los, dim=1), torch.cat(his, dim=1))

    # ------------------------------------------------------------- arithmetic
    @staticmethod
    def _coerce(other, like: "Interval"):
        if isinstance(other, Interval):
            return other
        if isinstance(other, (int, float)):
            t = torch.full_like(like.lo, float(other))
            return Interval(t, t)
        if isinstance(other, torch.Tensor):
            t = other.to(like.lo).expand_as(like.lo)
            return Interval(t, t)
        return NotImplemented

    def __add__(self, other):
        o = self._coerce(other, self)
        return Interval(*_widen(self.lo + o.lo, self.hi + o.hi))

    __radd__ = __add__

    def __neg__(self):
        return Interval(-self.hi, -self.lo)

    def __sub__(self, other):
        o = self._coerce(other, self)
        return Interval(*_widen(self.lo - o.hi, self.hi - o.lo))

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            c = float(other)
            lo, hi = (c * self.lo, c * self.hi) if c >= 0 else (c * self.hi, c * self.lo)
            return Interval(*_widen(lo, hi))
        o = self._coerce(other, self)
        p = torch.stack([self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi])
        return Interval(*_widen(p.min(0).values, p.max(0).values))

    __rmul__ = __mul__

    def square(self):
        lo2, hi2 = self.lo * self.lo, self.hi * self.hi
        hi = torch.maximum(lo2, hi2)
        lo = torch.where((self.lo <= 0) & (self.hi >= 0), torch.zeros_like(lo2), torch.minimum(lo2, hi2))
        return Interval(*_widen(lo, hi))

    # ------------------------------------------------------------ linear ops
    def linear(self, weight: torch.Tensor, bias: torch.Tensor | None = None):
        W = weight.to(self.lo)
        c, r = self.center(), self.radius()
        yc = c @ W.t()
        yr = r @ W.abs().t()
        mag = c.abs() @ W.abs().t() + yr
        if bias is not None:
            b = bias.to(self.lo)
            yc = yc + b
            mag = mag + b.abs()
        n = W.shape[1] + 2
        err = 2 * n * _U * mag + _ELEM_ABS
        return Interval(yc - yr - err, yc + yr + err)

    def scale_shift(self, scale: torch.Tensor, shift: torch.Tensor):
        s = scale.to(self.lo)
        b = shift.to(self.lo)
        lo = torch.where(s >= 0, self.lo * s, self.hi * s) + b
        hi = torch.where(s >= 0, self.hi * s, self.lo * s) + b
        return Interval(*_widen(lo, hi, rel=4 * _ELEM_REL))

    # ------------------------------------------------------ nonlinear ops
    def tanh(self):
        lo, hi = torch.tanh(self.lo), torch.tanh(self.hi)
        lo, hi = lo - _LIB_ABS, hi + _LIB_ABS
        return Interval(lo.clamp(min=-1.0), hi.clamp(max=1.0))

    def dtanh(self):
        """Enclosure of ``1 - tanh(z)^2`` for z in self (unimodal, max at 0)."""
        def d(z):
            t = torch.tanh(z)
            return 1.0 - t * t
        dl, du = d(self.lo), d(self.hi)
        z0 = torch.clamp(torch.zeros_like(self.lo), self.lo, self.hi)
        hi = d(z0)
        lo = torch.minimum(dl, du)
        return Interval((lo - 4 * _LIB_ABS).clamp(min=0.0), (hi + 4 * _LIB_ABS).clamp(max=1.0))

    def sin(self):
        return _interval_sin(self.lo, self.hi)

    def cos(self):
        return _interval_sin(self.lo + math.pi / 2, self.hi + math.pi / 2, shift_err=True)


def _interval_sin(lo: torch.Tensor, hi: torch.Tensor, shift_err: bool = False) -> Interval:
    """Sound enclosure of sin over [lo, hi]."""
    two_pi = 2 * math.pi
    slo, shi = torch.sin(lo), torch.sin(hi)
    ymin = torch.minimum(slo, shi)
    ymax = torch.maximum(slo, shi)
    # does [lo,hi] contain pi/2 + 2k pi ?  (maximum 1)
    k = torch.floor((hi - math.pi / 2) / two_pi)
    has_max = (math.pi / 2 + two_pi * k) >= lo
    k2 = torch.floor((hi + math.pi / 2) / two_pi)
    has_min = (-math.pi / 2 + two_pi * k2) >= lo
    wide = (hi - lo) >= two_pi
    ymax = torch.where(has_max | wide, torch.ones_like(ymax), ymax)
    ymin = torch.where(has_min | wide, -torch.ones_like(ymin), ymin)
    # argument-reduction / pi/2-shift rounding error: sin is 1-Lipschitz, so an
    # absolute error e in the argument perturbs the value by <= e.
    arg_err = 4 * _U * torch.maximum(lo.abs(), hi.abs()) + _LIB_ABS
    if shift_err:
        arg_err = arg_err + 4 * _U * (math.pi / 2)
    return Interval((ymin - arg_err).clamp(min=-1.0), (ymax + arg_err).clamp(max=1.0))
