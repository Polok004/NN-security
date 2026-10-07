"""Neural feedback controller  u = pi_theta(x)  with hard actuator limits."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from src import ops
from src.models.neural_barrier import build_tanh_mlp, mlp_forward


class NeuralController(nn.Module):
    """pi(x) = u_mid + u_half * tanh(MLP(features(x))).

    The output squashing guarantees u_min <= pi(x) <= u_max for every x, so
    the verified closed loop respects actuator saturation by construction.
    """

    def __init__(self, feature_dim: int, u_min, u_max, hidden: Sequence[int] = (32, 32)):
        super().__init__()
        u_min = torch.as_tensor(u_min, dtype=torch.float32)
        u_max = torch.as_tensor(u_max, dtype=torch.float32)
        self.layers = build_tanh_mlp(feature_dim, hidden, len(u_min))
        self.register_buffer("u_mid", 0.5 * (u_max + u_min))
        self.register_buffer("u_half", 0.5 * (u_max - u_min))

    def forward(self, s):
        out = mlp_forward(self.layers, s)[0]
        return ops.scale_shift(ops.tanh(out), self.u_half, self.u_mid)
