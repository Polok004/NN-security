"""Backend-agnostic primitive operations.

Every computation that must be *verified* (dynamics, feature maps, neural
networks, the Lie derivative) is written once against the functions in this
module.  The same source code is then executed by three backends:

* ``torch.Tensor``      -- concrete evaluation / training / auto_LiRPA tracing
* ``Interval``          -- sound interval bound propagation (IBP)
* ``Z3RelaxedVec``      -- sound linear relaxation encoded as Z3 constraints

Using a single source of truth guarantees that the object we train is exactly
the object we verify.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _is_tensor(x) -> bool:
    return isinstance(x, torch.Tensor)


def sin(x):
    return torch.sin(x) if _is_tensor(x) else x.sin()


def cos(x):
    return torch.cos(x) if _is_tensor(x) else x.cos()


def tanh(x):
    return torch.tanh(x) if _is_tensor(x) else x.tanh()


def dtanh(z, h):
    """Derivative of tanh evaluated at pre-activation ``z`` (``h = tanh(z)``).

    Concretely this is ``1 - h*h``.  Symbolic backends use ``z`` directly to
    obtain tighter enclosures than squaring an interval would give.
    """
    return 1.0 - h * h if _is_tensor(h) else z.dtanh()


def linear(x, weight: torch.Tensor, bias: torch.Tensor | None = None):
    return F.linear(x, weight, bias) if _is_tensor(x) else x.linear(weight, bias)


def scale_shift(x, scale: torch.Tensor, shift: torch.Tensor):
    """Element-wise ``x * scale + shift`` with constant (1-D) scale/shift."""
    if _is_tensor(x):
        return x * scale.to(x) + shift.to(x)
    return x.scale_shift(scale, shift)


def cat(xs):
    """Concatenate along the feature dimension (dim=1)."""
    if all(_is_tensor(x) for x in xs):
        return torch.cat(xs, dim=1)
    return type(next(x for x in xs if not _is_tensor(x))).cat(xs)
