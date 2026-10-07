"""Neural Control Barrier Function  B_phi : R^n -> R."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from src import ops


def build_tanh_mlp(in_dim: int, hidden: Sequence[int], out_dim: int) -> nn.ModuleList:
    dims = [in_dim, *hidden, out_dim]
    return nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))


def mlp_forward(layers: nn.ModuleList, h, dh=None):
    """Backend-agnostic forward pass of a tanh MLP.

    If a tangent ``dh`` is given the Jacobian-vector product is propagated
    alongside (forward-mode differentiation):  z = W h + b,  dz = W dh,
    h' = tanh(z),  dh' = tanh'(z) * dz.
    """
    for i, layer in enumerate(layers):
        z = ops.linear(h, layer.weight, layer.bias)
        dz = ops.linear(dh, layer.weight, None) if dh is not None else None
        if i == len(layers) - 1:
            return z, dz
        h = ops.tanh(z)
        if dh is not None:
            dh = ops.dtanh(z, h) * dz
    raise RuntimeError("empty MLP")


class NeuralControlBarrier(nn.Module):
    """B_phi(x) = MLP(features(x)) with smooth (tanh) activations.

    Smooth activations make the Lie derivative well defined everywhere, so the
    CBF condition can be verified pointwise and over boxes.
    """

    def __init__(self, feature_dim: int, hidden: Sequence[int] = (32, 32)):
        super().__init__()
        self.layers = build_tanh_mlp(feature_dim, hidden, 1)

    def forward(self, s):
        return mlp_forward(self.layers, s)[0]

    def forward_with_tangent(self, s, ds):
        return mlp_forward(self.layers, s, ds)

    # -- reference implementation via reverse-mode autograd (used in tests) --
    def compute_lie_derivative(self, s: torch.Tensor, u: torch.Tensor,
                               f_x: torch.Tensor, g_x: torch.Tensor) -> torch.Tensor:
        """dB/dt = grad B(s) . (f(s) + g(s) u) for identity feature maps."""
        s = s.detach().requires_grad_(True)
        B = self.forward(s)
        grad_B = torch.autograd.grad(B.sum(), s, create_graph=True)[0]
        dxdt = f_x + torch.bmm(g_x, u.unsqueeze(-1)).squeeze(-1)
        return (grad_B * dxdt).sum(dim=-1, keepdim=True)
