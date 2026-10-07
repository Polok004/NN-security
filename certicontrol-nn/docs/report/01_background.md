# Part I — Background {.unnumbered}

# Motivation: from testing to proof

## The deployment problem

Neural networks trained by reinforcement learning or imitation learning are
increasingly used as **controllers**: functions that map sensor readings to
motor commands in cars, drones, robot arms and medical devices. Before
deployment, such controllers are usually validated by **testing**: run many
simulations and check that nothing bad happens.

Testing answers the question *"did anything bad happen on the cases I tried?"*.
Safety certification needs the answer to *"can anything bad ever happen?"*.
These questions differ fundamentally because:

1. **The state space is continuous.** A car's state (position, heading,
   speed) is a vector of real numbers. There are uncountably many states. Any
   finite test set covers a set of volume zero.
2. **Neural networks are not smooth in the engineering sense.** They can
   change behaviour sharply in small regions, especially regions that training
   rarely visited.
3. **Failures compound over time.** A small error at one state moves the
   system to a different state, where the controller may behave worse. Rare
   bad regions are reached by trajectories, not by independent random draws.

## A concrete demonstration from this project

The 4-dimensional Dubins car experiment (Chapter 20) produced a
controller/barrier pair with these properties:

| evidence | verdict it suggests |
|---|---|
| 2,000,000 uniformly random states: 23 violate the safety condition (≈ $10^{-5}$) | "essentially safe" |
| 100 simulated drives from safe states for 10 s: none leave the safe region | "safe" |
| the formal verifier: 524 validated violating states; certification refused | **not safe** |
| 50 simulations started *at the verifier's counterexamples*: 41 leave the safe region | the verifier is right |

This single table is the motivation for the whole project. The verifier found
failures that testing effectively never finds, and it can prove their absence
when they really are absent, as for the pendulum.

## What the project builds

CertiControl-NN is an end-to-end system that:

1. **defines** a physical system with its safety requirements (Chapter 3);
2. **trains** two neural networks jointly: a controller $\pi_\theta$ and a
   barrier function $B_\phi$ (Chapter 12);
3. **verifies**, with mathematical rigour, that $B_\phi$ certifies safety of
   the closed loop *for all states* in a domain (Chapters 6–10);
4. if verification fails, **extracts concrete counterexamples**, adds them to
   the training data and repeats (Counterexample-Guided Inductive Synthesis,
   Chapter 13);
5. **stores** the proof and re-checks it with an independent checker
   (Chapter 11);
6. **visualises** everything in a Streamlit dashboard (Chapter 18).

It runs on a laptop CPU with open-source tools only: PyTorch, auto_LiRPA, Z3,
SciPy, Hydra, Streamlit and Matplotlib.

## Summary of results

| system | state dim. | outcome |
|---|---|---|
| torque-limited inverted pendulum | 2 | **Certified** in 3 synthesis rounds; proof of 140 boxes, checked independently |
| Dubins car with obstacle and walls | 4 | **Not certified**: 99.4% of the domain volume proven, 524 genuine counterexamples found after 40 rounds |

The second outcome is reported honestly as a partial result. Chapter 20
explains exactly why the failures occur.

---

# Mathematical prerequisites

This chapter collects every piece of mathematics used later. Skim it if it is
familiar.

## Vectors, norms and boxes

* A vector $x\in\mathbb R^n$ has components $x_1,\dots,x_n$.
* **Norms** measure size: $\|x\|_2=\sqrt{\sum_i x_i^2}$ (Euclidean) and
  $\|x\|_\infty=\max_i|x_i|$ (maximum norm).
* A **box** (hyper-rectangle) is $R=[l,u]=\{x: l_i\le x_i\le u_i\ \forall i\}$.
  In centre–radius form, $c=\frac{l+u}{2}$ and $r=\frac{u-l}{2}$, so
  $R=\{x:\|x-c\|_\infty\le r\}$ when all $r_i$ are equal. This is why
  auto_LiRPA calls a box an "$L_\infty$ perturbation".
* **Volume** of a box: $\prod_i(u_i-l_i)$. The proof checker uses volumes to
  confirm that the proof boxes cover the domain.

## Minimising a linear function over a box

This small fact is used everywhere in Chapters 7–10.

**Lemma 2.1.** For $a\in\mathbb R^n$ and $R=[l,u]$:
$$\min_{x\in R} a^\top x=\sum_i \min(a_il_i,\,a_iu_i)=a^\top c-|a|^\top r .$$

*Proof.* The objective is a sum of independent one-dimensional terms. Each term
$a_ix_i$ is minimised at $x_i=l_i$ if $a_i\ge0$ and at $x_i=u_i$ otherwise.
Writing $x_i=c_i+t_ir_i$ with $t_i\in[-1,1]$ gives $a_ic_i+a_ir_it_i$, whose
minimum is $a_ic_i-|a_i|r_i$. $\square$

Similarly, $\max_{x\in R}a^\top x=a^\top c+|a|^\top r$. This operation is called
**concretization**.

## Derivatives, gradients and Jacobians

* For $h:\mathbb R^n\to\mathbb R$, the **gradient** $\nabla h(x)$ is the vector
  of partial derivatives $\partial h/\partial x_i$.
* For $F:\mathbb R^n\to\mathbb R^m$, the **Jacobian** $J_F(x)$ is the
  $m\times n$ matrix $[\partial F_i/\partial x_j]$.
* **Chain rule.** $J_{G\circ F}(x)=J_G(F(x))\,J_F(x)$. Along a curve $x(t)$:
  $\frac{d}{dt}h(x(t))=\nabla h(x(t))\cdot\dot x(t)$.
* **Directional derivative.** The derivative of $h$ at $x$ in direction $v$ is
  $\nabla h(x)\cdot v=\lim_{\epsilon\to0}\frac{h(x+\epsilon v)-h(x)}{\epsilon}$.
  For vector-valued maps it is the **Jacobian-vector product (JVP)**
  $J_F(x)\,v$.

## The mean value theorem and Lipschitz continuity

**Mean value theorem (MVT).** If $f:[l,u]\to\mathbb R$ is differentiable, then
for $x,y\in[l,u]$ there is a $\xi$ between them with
$f(y)-f(x)=f'(\xi)(y-x)$.

**Consequence used by the Z3 encoding (Chapter 9).** Suppose
$d_{\min}\le f'\le d_{\max}$ on $[l,u]$. Then for $x\in[l,u]$:
$$f(l)+d_{\min}(x-l)\le f(x)\le f(l)+d_{\max}(x-l),\qquad
f(u)-d_{\max}(u-x)\le f(x)\le f(u)-d_{\min}(u-x).$$
*Proof.* Apply the MVT to $(l,x)$: $f(x)-f(l)=f'(\xi)(x-l)$ with $x-l\ge0$, so
the difference lies between $d_{\min}(x-l)$ and $d_{\max}(x-l)$. Apply it to
$(x,u)$ in the same way. $\square$

A function is **$L$-Lipschitz** if $|h(x)-h(y)|\le L\|x-y\|$. Lipschitz
constants bound how much a function can change within a box. That is why smooth
controllers (small $L$) are easier to verify (Chapter 12).

## Convexity and concavity

$f$ is **convex** on an interval if chords lie above the graph, and **concave**
if chords lie below. For a twice differentiable $f$, convexity means $f''\ge0$.
$\tanh$ is convex for $z\le0$ and concave for $z\ge0$, because
$\tanh''(z)=-2\tanh(z)(1-\tanh^2 z)$. This decides which lines give valid
linear bounds (Chapter 8).

## Sets: interior, boundary, compactness

* The **boundary** $\partial S$ is the set of points arbitrarily close to both
  $S$ and its complement. The **interior** is $S\setminus\partial S$ (for closed
  $S$).
* A set in $\mathbb R^n$ is **compact** if it is closed and bounded. Continuous
  functions attain their minimum and maximum on compact sets, and solutions of
  ODEs that stay in a compact set exist for all time.
* The **super-level set** of $B$ is $\{x: B(x)\ge0\}$. It is closed if $B$ is
  continuous.

## Linear programming (LP) in one paragraph

An LP minimises a linear function subject to linear inequalities. LPs are
solvable in polynomial time. A set of linear inequalities is a **polyhedron**.
Chapter 9 uses the fact that deciding whether a polyhedron is empty
(*feasibility*) is exactly what an SMT solver for linear real arithmetic does.

## Floating-point arithmetic

Computers represent reals as IEEE-754 doubles (float64): 53 significant bits.
Every operation rounds to the nearest representable number, with relative
error at most the **unit roundoff** $u=2^{-53}\approx1.1\times10^{-16}$. The
code uses the more conservative $2^{-52}$. For a dot product of $n$ terms, the
computed result $\hat s$ satisfies the classic bound (Higham, *Accuracy and
Stability of Numerical Algorithms*):
$$|\hat s-s|\le\gamma_n\sum_i|a_ib_i|,\qquad\gamma_n=\frac{nu}{1-nu}\approx nu .$$
A sound verifier must not let such errors turn "almost true" into "proven".
Chapter 7 explains how the interval code accounts for them.

---

# Dynamical systems and control

## States and ordinary differential equations

The **state** $x(t)\in\mathbb R^n$ is the information needed to predict the
future. A continuous-time system evolves by an **ODE** $\dot x=F(x)$.

**Existence and uniqueness (Picard–Lindelöf).** If $F$ is locally Lipschitz, then
for each $x_0$ there is a unique solution on some time interval $[0,T)$. If the
solution stays in a compact set, it exists for all $t\ge0$. All functions in
this project (polynomials, $\sin$, $\cos$, $\tanh$ networks) are smooth,
hence locally Lipschitz.

## Controlled systems and the control-affine form

With an input $u$ the system becomes $\dot x=F(x,u)$. The project uses the
**control-affine** form
$$\dot x=f(x)+g(x)\,u,\qquad u\in\mathcal U=[u_{\min},u_{\max}]\subset\mathbb R^m .$$

*Why control-affine?*

1. It covers most mechanical systems: forces and torques enter linearly through Newton's law.
2. Barrier-function theory is simplest in this setting: $\dot B$ is affine in $u$.
3. The closed-loop computation is a small graph that bounds tightly.

**Code: one definition, two derived objects.** Each system implements only
`xdot(x, u)`. The drift and input matrix are derived (`ControlAffineSystem.f`,
`.g`):
$$f(x)=\texttt{xdot}(x,0),\qquad g(x)_{:,j}=\texttt{xdot}(x,e_j)-f(x).$$
This is exact because `xdot` is affine in $u$. It guarantees that $f$, $g$ and
`xdot` can never disagree.

## System 1: the torque-limited inverted pendulum

### Derivation from Newton's law

A point mass $m$ on a massless rod of length $l$ rotates about a pivot. Let
$\theta$ be the angle from upright. Gravity produces torque $mgl\sin\theta$,
pushing away from upright. Joint friction produces torque $-b\dot\theta$, and
the motor produces torque $u$. The moment of inertia is $I=ml^2$. Newton's law
for rotation, $I\ddot\theta=\sum\text{torques}$, gives
$$ml^2\ddot\theta=mgl\sin\theta-b\dot\theta+u
\quad\Longrightarrow\quad
\ddot\theta=\frac{g}{l}\sin\theta-\frac{b}{ml^2}\dot\theta+\frac{1}{ml^2}u .$$
With state $x=(\theta,\omega)$, $\omega=\dot\theta$:
$$f(x)=\begin{pmatrix}\omega\\ \frac gl\sin\theta-\frac b{ml^2}\omega\end{pmatrix},\qquad
g(x)=\begin{pmatrix}0\\ \frac1{ml^2}\end{pmatrix}.$$

### Parameters and sets

| symbol | value | meaning |
|---|---|---|
| $m,l$ | 1, 1 | mass, length |
| $b$ | 0.1 | friction |
| $g$ | 9.81 | gravity |
| $u_{\max}$ | 6 | torque limit, $|u|\le6$ |
| $D$ | $[-1,1]\times[-4,4]$ | verification domain |
| $X_{\rm unsafe}$ | $|\theta|\ge0.8$ | too far from upright |
| $X_0$ | $[-0.25,0.25]\times[-0.4,0.4]$ | initial states |

**Why the problem is non-trivial.** To *hold* the pendulum at angle $\theta$
takes torque $mgl\sin\theta=9.81\sin\theta$. With $|u|\le6$, holding is
impossible beyond $\sin\theta=6/9.81$, that is $\theta\approx0.66$ rad, which is
less than the unsafe threshold 0.8. Moving states are harder still: a pendulum
at $\theta=0.5$ swinging outward fast cannot be stopped in time. The largest
region from which safety can be maintained, the **viability kernel**, is
therefore a tilted, lens-shaped region that nobody wrote down in advance. The
training must discover it.

### Code

```python
class InvertedPendulum(ControlAffineSystem):
    def xdot(self, x, u):
        th, om = x[:, 0:1], x[:, 1:2]           # columns kept 2-D: shape (N,1)
        ml2 = self.m * self.l ** 2
        dom = (self.grav / self.l) * ops.sin(th) + (-self.b / ml2) * om + (1.0 / ml2) * u[:, 0:1]
        return ops.cat([om, dom])               # shape (N,2)

    def features(self, x):                      # psi(x) = (theta, omega/4)
        return ops.cat([x[:, 0:1], (1.0 / self.omega_scale) * x[:, 1:2]])

    def features_jvp(self, x, xdot):            # d/dt psi(x) = J_psi(x) xdot
        return ops.cat([xdot[:, 0:1], (1.0 / self.omega_scale) * xdot[:, 1:2]])

    def unsafe_pieces(self, x):                 # unsafe = {0.8-theta<=0} U {theta+0.8<=0}
        th = x[:, 0:1]
        return [(-1.0 * th) + self.theta_unsafe, th + self.theta_unsafe]
```

*Design notes.*

* Slicing with `0:1` instead of `0` keeps shapes `(N,1)`, so the same code works for tensors, intervals and Z3 vectors.
* The unsafe set is written as a **union of sub-level sets** $\bigcup_i\{h_i(x)\le0\}$. This format is simple to test with intervals (can $h_i\le0$ happen in this box?) and to encode exactly in Z3 (a disjunction).
* The feature $\omega/4$ rescales $\omega\in[-4,4]$ to $[-1,1]$, the same range as $\theta$. Networks train better when inputs have similar scales.

## System 2: the Dubins car with acceleration

### Kinematics

A car moves in the plane with position $(p_x,p_y)$, heading $\theta$ and
forward speed $v$ (negative $v$ means reversing). Moving at speed $v$ in
direction $\theta$ gives $\dot p_x=v\cos\theta$ and $\dot p_y=v\sin\theta$. The
inputs are the turn rate $u_1=\dot\theta$ and the acceleration $u_2=\dot v$:
$$f(x)=\begin{pmatrix}v\cos\theta\\ v\sin\theta\\0\\0\end{pmatrix},\qquad
g(x)=\begin{pmatrix}0&0\\0&0\\1&0\\0&1\end{pmatrix}.$$

| item | value |
|---|---|
| domain $D$ | $p_x,p_y\in[-2,2]$ (walls), $\theta\in[-\pi,\pi]$ (periodic), $v\in[-1,1]$ |
| inputs | $|u_1|\le2$ rad/s, $|u_2|\le2$ m/s² |
| unsafe set | obstacle disc $p_x^2+p_y^2\le0.6^2$ |
| initial set $X_0$ | $|p_x|,|p_y|\le1.4$, $|v|\le0.25$, any heading, at least 0.35 from the obstacle surface |
| nominal task | drive towards goal $(1.4,1.4)$ at speed 0.5 (a soft preference only) |

**Walls and speed limits** are not in the unsafe set. Instead, they are faces of
$D$ that the safe set must never touch (property P3, Chapter 5). The safe set
must therefore also keep the car away from walls and prevent it from exceeding
$|v|=1$.

### The periodic heading

$\theta=\pi$ and $\theta=-\pi$ describe the same physical state. If the
networks saw $\theta$ directly, they could assign different values to the two,
and the domain would have artificial faces at $\theta=\pm\pi$. The **feature
map** therefore uses $(\cos\theta,\sin\theta)$:
$$\psi(x)=\big(p_x/2,\ p_y/2,\ \cos\theta,\ \sin\theta,\ v\big),$$
and $\theta$ is declared periodic (`periodic=[2]`) so that no face is created
for it. The time derivative of the features, needed for $\dot B$, is derived
with the chain rule:
$$\frac{d}{dt}\psi=\big(\dot p_x/2,\ \dot p_y/2,\ -\sin\theta\,\dot\theta,\ \cos\theta\,\dot\theta,\ \dot v\big).$$

```python
def features_jvp(self, x, xdot):
    s = 1.0 / self.arena
    th, dth = x[:, 2:3], xdot[:, 2:3]
    return ops.cat([s * xdot[:, 0:1], s * xdot[:, 1:2],
                    (-1.0 * ops.sin(th)) * dth, ops.cos(th) * dth, xdot[:, 3:4]])
```

### Sets written as polynomials

```python
def unsafe_pieces(self, x):     # (px-cx)^2 + (py-cy)^2 - r^2 <= 0
    dx = x[:, 0:1] + (-self.c[0]); dy = x[:, 1:2] + (-self.c[1])
    return [_square(dx) + _square(dy) + (-self.r ** 2)]

def init_constraints(self, x):  # X0 also requires distance >= r + 0.35
    ...
    return [_square(dx) + _square(dy) + (-(self.r + self.init_clearance) ** 2)]
```

Z3 can decide polynomial constraints exactly, so the obstacle is encoded
**without approximation** (Chapter 9).

## Feedback, closed loop and simulation

A controller $u=\pi(x)$ produces the **closed loop**
$\dot x=F_{cl}(x)=f(x)+g(x)\pi(x)$.

**Simulation** (`src/visualization/phase_portrait.py::simulate`) integrates the
closed loop with SciPy's `solve_ivp` using **RK45** (Dormand–Prince). This
method estimates its own error by comparing a 4th- and a 5th-order Runge–Kutta
step, and adapts the step size so the error stays below the tolerances
(`rtol=1e-7`, `atol=1e-9`). Compare the simplest method, explicit Euler
$x_{k+1}=x_k+hF(x_k)$, whose error is $O(h)$. RK45 is far more accurate at the
same cost. Simulation in this project is for **illustration and
cross-checking only**; it never contributes to a proof.

---

# Neural networks and differentiation

## The multilayer perceptron

An MLP with hidden widths $n_1,\dots,n_{L-1}$ computes
$$h_0=s,\qquad z_k=W_kh_{k-1}+b_k,\quad h_k=\tanh(z_k)\ (k<L),\qquad y=z_L=W_Lh_{L-1}+b_L .$$

**Parameter counts** (hidden widths [32, 32]):

| network | input dim | parameters |
|---|---|---|
| pendulum controller | 2 | $(2\cdot32+32)+(32\cdot32+32)+(32\cdot1+1)=1185$ |
| pendulum barrier | 2 | 1185 |
| Dubins controller | 5 | $(5\cdot32+32)+(32\cdot32+32)+(32\cdot2+2)=1314$ |
| Dubins barrier | 5 | 1281 |

These are deliberately **small**. Verification cost grows quickly with network
size, and small networks already have enough capacity for these problems.

`src/models/neural_barrier.py::build_tanh_mlp` builds the layers as a
`ModuleList` of `nn.Linear`.

## Why tanh?

$\tanh(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}}$ has these properties:

1. **Smoothness.** $\tanh'(z)=1-\tanh^2z$ and $\tanh''(z)=-2\tanh z\,(1-\tanh^2z)$. The barrier condition uses $\nabla B$, which must exist everywhere.
2. **Monotone and bounded**, with range $(-1,1)$. Interval bounds are trivial: $\tanh[\underline z,\overline z]=[\tanh\underline z,\tanh\overline z]$.
3. **$\tanh'$ is bounded, positive and unimodal**, with maximum 1 at $z=0$. This gives cheap, tight bounds on derivatives.

*ReLU comparison.* $\mathrm{ReLU}(z)=\max(0,z)$ is standard in vision and simple
to verify for plain outputs. But $B$ would be piecewise linear and its gradient
would jump. $\dot B$ would then be undefined on the kinks, and invariance
arguments would need nonsmooth analysis (Clarke generalized gradients). Smooth
activations avoid all of this.

## The controller: saturation by construction

$$\pi(x)=u_{\rm mid}+u_{\rm half}\odot\tanh(\mathrm{MLP}_\theta(\psi(x))).$$
Since $\tanh\in(-1,1)$, every output component lies in
$(u_{\min},u_{\max})$ **for every weight setting and every input**. Actuator
limits therefore never need to be verified, and the verifier's interval
bounds automatically respect them. `test_controller_respects_actuator_limits`
multiplies all weights by 50 and confirms the limits still hold.

## Automatic differentiation

Automatic differentiation (AD) computes exact derivatives of programs by
applying the chain rule to every elementary operation.

### Reverse mode

Reverse mode (backpropagation) runs the program forward, then propagates
**adjoints** $\bar v=\partial y/\partial v$ backward. One backward pass gives
the full gradient of a scalar output, at a cost of a small constant times the
forward pass. PyTorch's `loss.backward()` uses it for training.

### Forward mode and dual numbers

Forward mode propagates **tangents** $\dot v=\frac{\partial v}{\partial x}\cdot d$
alongside the values, for a chosen direction $d$. Algebraically this is
arithmetic on **dual numbers** $a+b\varepsilon$ with $\varepsilon^2=0$:
$$f(a+b\varepsilon)=f(a)+f'(a)b\,\varepsilon .$$
One forward pass gives one directional derivative (JVP).

### Why forward mode fits this project exactly

The barrier condition needs
$$\dot B(x)=\nabla B(x)\cdot F_{cl}(x),$$
which is precisely the directional derivative of $B$ in direction
$F_{cl}(x)$. A single forward-mode pass with seed direction
$F_{cl}(x)$ computes it. The result is an ordinary feed-forward graph that bound
propagation tools can analyse. Reverse mode would give the same number, but
through a backward graph. Tools like auto_LiRPA are designed around forward
computation graphs, and bounding a gradient computed by backpropagation would
be far more involved.

### Derivation of the tangent recursion

Let $d\psi=J_\psi(x)F_{cl}(x)$ be the feature-space direction. For the affine
layer $z=Wh+b$, the chain rule gives $dz=W\,dh$ (the constant $b$ contributes
nothing). For $h'=\tanh(z)$, applied elementwise, it gives
$dh'=\tanh'(z)\odot dz$. So:
$$
\begin{aligned}
(h_0,\,dh_0)&=(\psi(x),\,J_\psi F_{cl})\\
(z_k,\,dz_k)&=(W_kh_{k-1}+b_k,\ W_k\,dh_{k-1})\\
(h_k,\,dh_k)&=(\tanh z_k,\ (1-h_k^2)\odot dz_k)\\
(B,\,\dot B)&=(z_L,\,dz_L)
\end{aligned}
$$

### Worked example

Take a one-neuron barrier $B(x)=2\tanh(x_1-x_2)$ and closed-loop field
$F_{cl}(x)=(x_2,-x_1)$ at $x=(1,0.5)$.

* Value: $z=0.5$, $h=\tanh0.5=0.4621$, $B=0.9242$.
* Tangent: the seed is $F_{cl}=(0.5,-1)$. Then $dz=1\cdot0.5+(-1)(-1)=1.5$, $dh=(1-0.4621^2)\cdot1.5=1.1797$, and $\dot B=2\cdot1.1797=2.3594$.
* Check by the gradient: $\nabla B=2(1-h^2)(1,-1)=(1.5729,-1.5729)$, and $\nabla B\cdot F_{cl}=0.7864+1.5729=2.3594$. ✓

### Code

```python
def mlp_forward(layers, h, dh=None):
    for i, layer in enumerate(layers):
        z = ops.linear(h, layer.weight, layer.bias)              # z = W h + b
        dz = ops.linear(dh, layer.weight, None) if dh is not None else None   # dz = W dh
        if i == len(layers) - 1:
            return z, dz                                         # output layer: no tanh
        h = ops.tanh(z)
        if dh is not None:
            dh = ops.dtanh(z, h) * dz                            # (1-h^2) * dz
```

`ops.dtanh(z, h)` returns `1 - h*h` for tensors. For intervals and Z3, it uses
$z$ directly to obtain tighter bounds (Chapter 7).

The certificate module (`src/models/certificate.py`) chains everything:

```python
def terms(self, x):
    s = self.system.features(x)                 # psi(x)
    u = self.controller(s)                      # pi(x)
    xdot = self.system.xdot(x, u)               # F_cl(x) = f + g pi
    ds = self.system.features_jvp(x, xdot)      # J_psi(x) F_cl(x)
    B, dB = self.barrier.forward_with_tangent(s, ds)
    return B, dB, u

def forward(self, x):
    B, dB, _ = self.terms(x)
    return ops.cat([B, dB + self.gamma * B])    # the two verified outputs
```

**Validation.** `test_certificate_lie_derivative_matches_autograd` computes
$\nabla B$ with reverse-mode autograd, forms $\nabla B\cdot(f+g\pi)+\gamma B$
independently, and checks agreement to $10^{-10}$ for both systems. The spec's
reference method `NeuralControlBarrier.compute_lie_derivative` (reverse mode)
is also tested against the forward-mode tangent.

## One computation, several meanings: `src/ops.py`

Every verified computation is written in terms of seven primitives plus
`+ - *` and slicing:

| primitive | meaning |
|---|---|
| `sin`, `cos`, `tanh` | elementwise functions |
| `dtanh(z, h)` | $1-\tanh^2(z)$ |
| `linear(x, W, b)` | $xW^\top+b$ |
| `scale_shift(x, a, c)` | $x\odot a+c$ with constant vectors |
| `cat(xs)` | concatenate features |

Each primitive checks the type of its argument:

```python
def tanh(x):
    return torch.tanh(x) if _is_tensor(x) else x.tanh()
```

When `x` is an `Interval`, `x.tanh()` returns a sound interval. When `x` is a
`Z3RelaxedVec`, it returns fresh Z3 variables with enclosing constraints.
Python's operator overloading (`__add__`, `__mul__`, `__getitem__`) handles
the rest.

**Why this design matters.** It guarantees *the verified function is the trained
function*. There is only one implementation of the dynamics, features,
networks and Lie derivative. A typical bug in verification projects, a formula
in the verifier that differs slightly from the model in the trainer, becomes
impossible by construction.
