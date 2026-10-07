"""Control-affine continuous-time dynamical systems  dx/dt = f(x) + g(x) u.

Each system specifies

* the dynamics (``f``, ``g`` and the backend-agnostic ``xdot``),
* a *feature map* fed to the neural networks together with its Jacobian-vector
  product (used for exact forward-mode Lie derivatives),
* the verification domain ``D`` (a hyper-rectangle, possibly with periodic
  coordinates), the unsafe set ``X_u`` (a union of sub-level sets
  ``{h_i(x) <= 0}``) and a desired initial set ``X_0``.

All methods used during verification are written with :mod:`src.ops`, so the
same code runs on tensors, intervals and Z3 relaxations.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from src import ops


@dataclass
class Box:
    lo: List[float]
    hi: List[float]

    def tensor(self, dtype=torch.float64):
        return torch.tensor(self.lo, dtype=dtype), torch.tensor(self.hi, dtype=dtype)


class ControlAffineSystem:
    name: str = "base"
    state_names: List[str] = []
    control_names: List[str] = []

    def __init__(self, domain: Box, u_min: List[float], u_max: List[float],
                 init_set: Box, periodic: Optional[List[int]] = None):
        self.domain = domain
        self.u_min = torch.tensor(u_min, dtype=torch.float64)
        self.u_max = torch.tensor(u_max, dtype=torch.float64)
        self.init_set = init_set
        self.periodic = list(periodic or [])

    # ------------------------------------------------------------ dimensions
    @property
    def state_dim(self) -> int:
        return len(self.domain.lo)

    @property
    def control_dim(self) -> int:
        return len(self.u_min)

    @property
    def feature_dim(self) -> int:
        raise NotImplementedError

    # ------------------------------------------------------- dynamics (ops)
    def xdot(self, x, u):
        """Closed-loop vector field f(x) + g(x) u, backend-agnostic."""
        raise NotImplementedError

    def features(self, x):
        raise NotImplementedError

    def features_jvp(self, x, xdot):
        """Directional derivative of ``features`` at x along ``xdot``."""
        raise NotImplementedError

    def unsafe_pieces(self, x) -> list:
        """List of expressions h_i(x); the unsafe set is  U_i {h_i(x) <= 0}."""
        raise NotImplementedError

    def init_constraints(self, x) -> list:
        """Expressions g_j(x); X_0 = init box  intersect  {g_j(x) >= 0 for all j}."""
        return []

    def in_init(self, x: torch.Tensor) -> torch.Tensor:
        lo, hi = self.init_set.tensor(x.dtype)
        ok = ((x >= lo) & (x <= hi)).all(-1)
        for gj in self.init_constraints(x):
            ok = ok & (gj.squeeze(-1) >= 0)
        return ok

    # ------------------------------------------- explicit f and g (tensors)
    def f(self, x: torch.Tensor) -> torch.Tensor:
        return self.xdot(x, torch.zeros(x.shape[0], self.control_dim, dtype=x.dtype))

    def g(self, x: torch.Tensor) -> torch.Tensor:
        """g(x) as a (N, n, m) tensor, obtained column-wise from xdot."""
        f = self.f(x)
        cols = []
        for j in range(self.control_dim):
            e = torch.zeros(x.shape[0], self.control_dim, dtype=x.dtype)
            e[:, j] = 1.0
            cols.append(self.xdot(x, e) - f)
        return torch.stack(cols, dim=-1)

    # ------------------------------------------------------------- sets
    def is_unsafe(self, x: torch.Tensor) -> torch.Tensor:
        pieces = self.unsafe_pieces(x)
        return torch.stack([p.squeeze(-1) <= 0 for p in pieces], 0).any(0)

    def in_domain(self, x: torch.Tensor) -> torch.Tensor:
        """Membership in D; periodic coordinates (angles) are always inside."""
        lo, hi = self.domain.tensor(x.dtype)
        ok = (x >= lo) & (x <= hi)
        for d in self.periodic:
            ok[..., d] = True
        return ok.all(-1)

    def sample_domain(self, n: int, dtype=torch.float32) -> torch.Tensor:
        lo, hi = self.domain.tensor(dtype)
        return lo + (hi - lo) * torch.rand(n, self.state_dim, dtype=dtype)

    def sample_box(self, box: Box, n: int, dtype=torch.float32) -> torch.Tensor:
        lo, hi = box.tensor(dtype)
        return lo + (hi - lo) * torch.rand(n, self.state_dim, dtype=dtype)

    def sample_init(self, n: int, dtype=torch.float32) -> torch.Tensor:
        return self.sample_box(self.init_set, n, dtype)

    def sample_unsafe(self, n: int, dtype=torch.float32) -> torch.Tensor:
        out, got = [], 0
        while got < n:
            x = self.sample_domain(4 * n, dtype)
            x = x[self.is_unsafe(x)]
            out.append(x)
            got += x.shape[0]
        return torch.cat(out)[:n]

    def sample_unsafe_train(self, n: int, dtype=torch.float32) -> torch.Tensor:
        """Unsafe samples used by the *training* loss (may be an inflated set).

        Training against a slightly inflated unsafe set keeps C away from the
        obstacle boundary, where the invariance condition is hardest to meet;
        verification (P2) always uses the true unsafe set.
        """
        return self.sample_unsafe(n, dtype)

    def boundary_faces(self):
        """Faces of the domain (d, side, value) over non-periodic coordinates."""
        faces = []
        for d in range(self.state_dim):
            if d in self.periodic:
                continue
            faces.append((d, "lo", self.domain.lo[d]))
            faces.append((d, "hi", self.domain.hi[d]))
        return faces

    def sample_boundary(self, n: int, dtype=torch.float32) -> torch.Tensor:
        x = self.sample_domain(n, dtype)
        faces = self.boundary_faces()
        idx = torch.randint(len(faces), (n,))
        for k, (d, _, v) in enumerate(faces):
            x[idx == k, d] = v
        return x

    # ------------------------------------------------------------ numerics
    def rhs_numpy(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        xt = torch.as_tensor(x, dtype=torch.float64).reshape(1, -1)
        ut = torch.as_tensor(u, dtype=torch.float64).reshape(1, -1)
        return self.xdot(xt, ut).reshape(-1).numpy()

    def nominal_control(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        return None


# =====================================================================
class InvertedPendulum(ControlAffineSystem):
    """theta'' = (g/l) sin(theta) - b/(m l^2) theta' + u/(m l^2), |u| <= u_max.

    theta = 0 is the upright equilibrium.  Safety: |theta| < theta_unsafe.
    With u_max < m g l sin(theta_unsafe) the torque limit -- not the unsafe set
    -- determines the largest invariant set, which makes the problem
    non-trivial.
    """

    name = "inverted_pendulum"
    state_names = ["theta", "omega"]
    control_names = ["torque"]

    def __init__(self, m=1.0, l=1.0, b=0.1, grav=9.81, u_max=6.0,
                 theta_unsafe=0.8, theta_lim=1.0, omega_lim=4.0,
                 init_theta=0.25, init_omega=0.4, omega_scale=4.0):
        super().__init__(Box([-theta_lim, -omega_lim], [theta_lim, omega_lim]),
                         [-u_max], [u_max],
                         Box([-init_theta, -init_omega], [init_theta, init_omega]))
        self.m, self.l, self.b, self.grav = m, l, b, grav
        self.theta_unsafe = theta_unsafe
        self.omega_scale = omega_scale

    @property
    def feature_dim(self):
        return 2

    def xdot(self, x, u):
        th, om = x[:, 0:1], x[:, 1:2]
        ml2 = self.m * self.l ** 2
        dom = (self.grav / self.l) * ops.sin(th) + (-self.b / ml2) * om + (1.0 / ml2) * u[:, 0:1]
        return ops.cat([om, dom])

    def features(self, x):
        return ops.cat([x[:, 0:1], (1.0 / self.omega_scale) * x[:, 1:2]])

    def features_jvp(self, x, xdot):
        return ops.cat([xdot[:, 0:1], (1.0 / self.omega_scale) * xdot[:, 1:2]])

    def unsafe_pieces(self, x):
        th = x[:, 0:1]
        return [(-1.0 * th) + self.theta_unsafe, th + self.theta_unsafe]


# =====================================================================
class DubinsCar(ControlAffineSystem):
    """4-D Dubins car with acceleration control and a circular obstacle.

    px' = v cos(theta), py' = v sin(theta), theta' = u1, v' = u2.
    State x = (px, py, theta, v); theta is periodic (features use cos/sin).
    Unsafe: the obstacle disc ||(px,py) - c|| <= r.  The domain walls (and the
    speed limits) must also never be crossed, which the verifier enforces by
    requiring B < 0 on every non-periodic face of D.
    """

    name = "dubins_car"
    state_names = ["px", "py", "theta", "v"]
    control_names = ["omega", "accel"]

    def __init__(self, arena=2.0, v_max=1.0, omega_max=2.0, a_max=2.0,
                 obstacle_center=(0.0, 0.0), obstacle_radius=0.6,
                 init_xy=1.4, init_v=0.25, goal=(1.4, 1.4), v_des=0.5, train_inflation=0.0):
        super().__init__(Box([-arena, -arena, -math.pi, -v_max], [arena, arena, math.pi, v_max]),
                         [-omega_max, -a_max], [omega_max, a_max],
                         Box([-init_xy, -init_xy, -math.pi, -init_v], [init_xy, init_xy, math.pi, init_v]),
                         periodic=[2])
        self.arena = arena
        self.c = tuple(float(c) for c in obstacle_center)
        self.r = float(obstacle_radius)
        self.goal = goal
        self.v_des = v_des
        self.init_clearance = 0.35
        self.train_inflation = float(train_inflation)

    @property
    def feature_dim(self):
        return 5

    def xdot(self, x, u):
        th, v = x[:, 2:3], x[:, 3:4]
        return ops.cat([v * ops.cos(th), v * ops.sin(th), u[:, 0:1], u[:, 1:2]])

    def features(self, x):
        s = 1.0 / self.arena
        th = x[:, 2:3]
        return ops.cat([s * x[:, 0:1], s * x[:, 1:2], ops.cos(th), ops.sin(th), x[:, 3:4]])

    def features_jvp(self, x, xdot):
        s = 1.0 / self.arena
        th = x[:, 2:3]
        dth = xdot[:, 2:3]
        return ops.cat([s * xdot[:, 0:1], s * xdot[:, 1:2],
                        (-1.0 * ops.sin(th)) * dth, ops.cos(th) * dth, xdot[:, 3:4]])

    def unsafe_pieces(self, x):
        dx = x[:, 0:1] + (-self.c[0])
        dy = x[:, 1:2] + (-self.c[1])
        return [_square(dx) + _square(dy) + (-self.r ** 2)]

    def init_constraints(self, x):
        dx = x[:, 0:1] + (-self.c[0])
        dy = x[:, 1:2] + (-self.c[1])
        return [_square(dx) + _square(dy) + (-(self.r + self.init_clearance) ** 2)]

    def sample_init(self, n: int, dtype=torch.float32) -> torch.Tensor:
        out, got = [], 0
        while got < n:
            x = self.sample_box(self.init_set, 4 * n, dtype)
            x = x[self.in_init(x)]
            out.append(x)
            got += x.shape[0]
        return torch.cat(out)[:n]

    def sample_unsafe_train(self, n: int, dtype=torch.float32) -> torch.Tensor:
        if self.train_inflation <= 0:
            return self.sample_unsafe(n, dtype)
        out, got = [], 0
        R = self.r + self.train_inflation
        while got < n:
            x = self.sample_domain(4 * n, dtype)
            d2 = (x[:, 0] - self.c[0]) ** 2 + (x[:, 1] - self.c[1]) ** 2
            x = x[d2 <= R ** 2]
            out.append(x)
            got += x.shape[0]
        return torch.cat(out)[:n]

    def nominal_control(self, x):
        gx, gy = self.goal
        desired = torch.atan2(gy - x[:, 1], gx - x[:, 0])
        err = torch.atan2(torch.sin(desired - x[:, 2]), torch.cos(desired - x[:, 2]))
        u1 = (2.0 * err).clamp(float(self.u_min[0]), float(self.u_max[0]))
        u2 = (2.0 * (self.v_des - x[:, 3])).clamp(float(self.u_min[1]), float(self.u_max[1]))
        return torch.stack([u1, u2], -1)


def _square(x):
    if isinstance(x, torch.Tensor):
        return x * x
    return x.square()


def make_system(cfg) -> ControlAffineSystem:
    """Instantiate a system from a (Hydra / dict) config."""
    cfg = dict(cfg)
    name = cfg.pop("name")
    if name == "inverted_pendulum":
        return InvertedPendulum(**cfg)
    if name == "dubins_car":
        cfg["obstacle_center"] = tuple(cfg.get("obstacle_center", (0.0, 0.0)))
        cfg["goal"] = tuple(cfg.get("goal", (1.4, 1.4)))
        return DubinsCar(**cfg)
    raise ValueError(f"unknown system {name}")
