"""The closed-loop certificate function that is trained and verified.

    CertificateModule(x) = [ B(x),  dB/dt(x) + gamma * B(x) ]

where dB/dt(x) = grad B(x) . (f(x) + g(x) pi(x)) is computed by exact forward-
mode differentiation through the feature map and the barrier network.  The
module is a plain ``nn.Module`` (traceable by auto_LiRPA) but its ``forward``
also accepts :class:`Interval` / Z3 relaxation inputs through :mod:`src.ops`.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src import ops
from src.dynamics.continuous_system import ControlAffineSystem
from src.models.neural_barrier import NeuralControlBarrier
from src.models.neural_controller import NeuralController


class CertificateModule(nn.Module):
    def __init__(self, system: ControlAffineSystem, controller: NeuralController,
                 barrier: NeuralControlBarrier, gamma: float):
        super().__init__()
        self.system = system
        self.controller = controller
        self.barrier = barrier
        self.gamma = float(gamma)

    def terms(self, x):
        s = self.system.features(x)
        u = self.controller(s)
        xdot = self.system.xdot(x, u)
        ds = self.system.features_jvp(x, xdot)
        B, dB = self.barrier.forward_with_tangent(s, ds)
        return B, dB, u

    def forward(self, x):
        B, dB, _ = self.terms(x)
        cond = dB + self.gamma * B
        return ops.cat([B, cond])

    # convenience -------------------------------------------------------
    @torch.no_grad()
    def barrier_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.barrier(self.system.features(x))

    @torch.no_grad()
    def control(self, x: torch.Tensor) -> torch.Tensor:
        return self.controller(self.system.features(x))


def build_models(system: ControlAffineSystem, ctrl_hidden=(32, 32), barrier_hidden=(32, 32)):
    ctrl = NeuralController(system.feature_dim, system.u_min, system.u_max, ctrl_hidden)
    bar = NeuralControlBarrier(system.feature_dim, barrier_hidden)
    return ctrl, bar
