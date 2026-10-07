# Part II — Safety theory {.unnumbered}

# Invariance, barrier certificates and the main theorem

## Safety as "never enter the unsafe set"

Given a closed-loop system $\dot x=F_{cl}(x)$, an **unsafe set**
$X_{\rm unsafe}$ and a set of **initial states** $X_0$, the system is *safe* if
every trajectory with $x(0)\in X_0$ satisfies $x(t)\notin X_{\rm unsafe}$ for
all $t\ge0$.

Checking this directly means reasoning about infinitely many infinitely long
trajectories. The standard trick is to reduce it to a **local** condition on a
set.

## Forward invariance

A set $\mathcal C$ is **forward invariant** if $x(0)\in\mathcal C$ implies
$x(t)\in\mathcal C$ for all $t\ge0$.

**Safety via invariance.** If some $\mathcal C$ satisfies

(a) $X_0\subseteq\mathcal C$,

(b) $\mathcal C\cap X_{\rm unsafe}=\emptyset$, and

(c) $\mathcal C$ is forward invariant,

then the system is safe. *Proof:* a trajectory starting in $X_0$ starts in
$\mathcal C$ by (a), stays in $\mathcal C$ by (c), and $\mathcal C$ contains no
unsafe state by (b). $\square$

The remaining question is how to establish (c) without simulating.

## Describing sets by functions

Let $\mathcal C=\{x:B(x)\ge0\}$ for a continuously differentiable
$B:\mathbb R^n\to\mathbb R$. Along a trajectory,
$$\dot B(x)=\frac{d}{dt}B(x(t))=\nabla B(x)\cdot F_{cl}(x)=:L_{F_{cl}}B(x),$$
the **Lie derivative** of $B$ along $F_{cl}$. It is a function of the state
only, so it can be checked pointwise, and hence over boxes, without
simulating.

## Nagumo's theorem

**Theorem (Nagumo, 1942; informal).** Let $\mathcal C$ be closed and assume
solutions are unique. Then $\mathcal C$ is forward invariant if and only if,
at every boundary point, $F_{cl}(x)$ points into $\mathcal C$ or is tangent to
it (it lies in the *tangent cone*).

For $\mathcal C=\{B\ge0\}$ with $\nabla B\ne0$ on the boundary, the condition
reads $\dot B(x)\ge0$ wherever $B(x)=0$. This is intuitive: on the boundary,
$B$ must not start to decrease. The figure below illustrates this.

![Left: the flow points into $\mathcal C$ everywhere on its boundary, so $\mathcal C$ is forward invariant. Right: on part of the boundary (red arrows) $\dot B<0$, and trajectories escape towards the unsafe region.](../figures/invariance.png)


**Why not check just the boundary?** Two reasons. First, the boundary is a
thin, lower-dimensional set: a box verifier cannot isolate it, and any box
touching the boundary also contains points just inside and just outside.
Second, the condition "$\ge0$ exactly at $B=0$" has no robustness margin: a
tiny numerical error flips it. Both problems are fixed by strengthening the
condition to all of $\mathcal C$.

## Barrier certificates and control barrier functions

**Barrier certificates** (Prajna & Jadbabaie, 2004) are functions separating
initial and unsafe sets with a sign condition on $\dot B$. **Control barrier
functions** (Ames et al., 2017/2019) generalise this to systems with inputs.
The standard *zeroing CBF* condition is
$$\dot B(x)\ \ge\ -\alpha(B(x))\qquad\text{for all }x\in\mathcal C,$$
where $\alpha$ is an (extended) **class-$\mathcal K$ function**: continuous,
strictly increasing, with $\alpha(0)=0$.

*Interpretation.* On the boundary ($B=0$) the condition becomes $\dot B\ge0$,
which is Nagumo's condition. Inside, $B$ may decrease, but no faster than
$\alpha(B)$. As the state approaches the boundary, the permitted rate of
decrease goes to zero, like a car braking harder the closer it gets to a wall.

This project uses $\alpha(y)=\gamma y$ with $\gamma=1$. The quantity checked
everywhere is
$$\boxed{\ \mathrm{cond}(x)=\nabla B(x)\cdot\big(f(x)+g(x)\pi(x)\big)+\gamma B(x)\ \ge 0\ }$$

**Neural CBF.** Here $B=B_\phi$ is a neural network and the input is fixed to
the neural controller $\pi_\theta$. Both are learned. Because the controller is
fixed inside the proof, the certificate says: *this specific closed loop is
safe*. There is no online optimisation, unlike a CBF-QP safety filter, which
solves a quadratic program at every control step.

## The comparison lemma, with proof

**Lemma 5.1.** Let $b:[0,T)\to\mathbb R$ be continuously differentiable with
$b(0)\ge0$, and suppose $\dot b(t)+\gamma b(t)\ge\tau$ whenever $b(t)\ge0$, for
some $\tau>0$ (the verified margin). Then $b(t)\ge e^{-\gamma t}b(0)\ge0$ for
all $t\in[0,T)$.

*Proof.*

**Step 1: $b$ never becomes negative.** Suppose it does, and let
$t_0=\inf\{t: b(t)<0\}$. By continuity $b(t_0)=0$, and $b$ takes negative
values at times arbitrarily close after $t_0$. But at $t_0$ the hypothesis
applies: $\dot b(t_0)\ge\tau-\gamma\cdot0=\tau>0$. So $b$ is strictly
increasing at $t_0$, and $b>0$ on some interval $(t_0,t_0+\delta)$. This
contradicts the choice of $t_0$. Hence $b(t)\ge0$ for all $t$.

**Step 2: the exponential bound.** Since $b\ge0$ everywhere, the hypothesis
holds at all times. Let $\phi(t)=e^{\gamma t}b(t)$. Then
$\dot\phi=e^{\gamma t}(\dot b+\gamma b)\ge e^{\gamma t}\tau\ge0$, so $\phi$ is
non-decreasing: $e^{\gamma t}b(t)\ge b(0)$, that is,
$b(t)\ge e^{-\gamma t}b(0)$. $\square$

*Remark.* The margin $\tau=10^{-8}$ enforced by the verifier (Chapter 7) is
not only a guard against rounding. It also makes Step 1 a one-line argument.
Without a margin ($\tau=0$), the same conclusion needs a more careful
argument (for example, Nagumo's theorem with uniqueness of solutions).

## The four certified properties

The verifier works on a bounded box $D$ (the domain). Define
$\mathcal C=\{x\in D: B(x)\ge0\}$. The certified properties are:

| | statement | why it is needed |
|---|---|---|
| **P1** invariance | $\forall x\in D:\ B(x)\ge0\Rightarrow\mathrm{cond}(x)\ge\tau$ | the CBF condition on the whole safe set |
| **P2** safety | $\forall x\in D\cap X_{\rm unsafe}:\ B(x)<0$ | ensures $\mathcal C\cap X_{\rm unsafe}=\emptyset$ |
| **P3** containment | $\forall x\in\partial D$ (non-periodic faces): $B(x)<0$ | ensures $\mathcal C$ does not reach the edge of the verified domain |
| **P0** non-triviality | $\forall x\in X_0:\ B(x)>0$ | ensures $X_0\subseteq\mathcal C$, and rules out $\mathcal C=\emptyset$ |

**What happens without each property.**

* **Without P1:** nothing prevents leaving $\mathcal C$.
* **Without P2:** $\mathcal C$ might contain unsafe states.
* **Without P3:** the proof only covers states inside $D$. A trajectory could exit $D$ through a face while $B$ is still positive, enter the unverified region, and come back anywhere. Requiring $B<0$ on the faces makes $\{B\ge0\}\cap D$ a set that cannot touch $\partial D$. By continuity, any trajectory leaving $D$ would first have to cross $B=0$ inside $D$, which P1 forbids.
* **Without P0:** the constant function $B\equiv-1$ satisfies P1 (vacuously, since $B\ge0$ never holds), P2 and P3. It "certifies" nothing. This really happened during development (Chapter 21): training converged to $B<0$ everywhere until P0 was added to the verifier and weighted strongly in the loss.

## Main theorem, with proof

**Theorem 5.2.** Let $B$ and $\pi$ be continuously differentiable, and suppose
P0–P3 hold. Then every closed-loop trajectory with $x(0)\in X_0$ exists for all
$t\ge0$, stays in $\mathcal C$, and never enters $X_{\rm unsafe}$.

*Proof.*

1. *$\mathcal C$ is compact and lies in the interior of $D$ (in non-periodic coordinates).* $\mathcal C$ is closed ($B$ is continuous) and bounded ($\subseteq D$). By P3, $B<0$ on every non-periodic face of $D$, so $\mathcal C$ contains no point of those faces. For the periodic coordinate $\theta$, the faces $\theta=\pm\pi$ are identified (the networks see $\cos\theta,\sin\theta$), so they are not a real boundary.
2. *Existence.* $F_{cl}$ is locally Lipschitz, so a unique solution exists on some $[0,T)$. As long as it stays in the compact $\mathcal C$, it can be extended. So it suffices to show that it stays in $\mathcal C$.
3. *Invariance.* $x(0)\in X_0\subseteq\mathcal C$ by P0. Let $b(t)=B(x(t))$. While $x(t)\in D$ and $b(t)\ge0$, P1 gives $\dot b+\gamma b=\mathrm{cond}\ge\tau>0$. By Lemma 5.1, $b$ never becomes negative, *provided* the trajectory stays in $D$. It cannot leave $D$ while $b\ge0$: to reach a face, $x(t)$ would have to arrive at a point where $B<0$ (P3), so $b$ would first have to cross zero inside $D$, which Lemma 5.1 excludes. Hence $x(t)\in\mathcal C$ for all $t$.
4. *Safety.* By P2, no point of $\mathcal C$ is unsafe. $\square$

**Corollary 5.3 (robustness in $\gamma$).** If P1 holds for $\gamma$, it holds for
every $\gamma'\ge\gamma$: on $\mathcal C$, $B\ge0$, so $\gamma'B\ge\gamma B$.
The dashboard exploits this. The pendulum proof was in fact re-verified
successfully for $\gamma\in\{0.25,0.5,1,2,4\}$.

## Connection to Lyapunov functions

A **Lyapunov function** $V$ proves *stability* (convergence to an
equilibrium) through $\dot V<0$. A barrier function proves *invariance*
(staying in a set) through $\dot B\ge-\alpha(B)$. Both reduce a
statement about trajectories to a pointwise inequality on the state space,
which is exactly what makes them verifiable by the same bound-propagation
machinery. Neural Lyapunov control (Chang et al., 2019) and FOSSIL (Abate et
al., 2021) apply the learn-then-verify idea to Lyapunov functions.

## Why *learn* $B$ and $\pi$?

Classical constructions (sum-of-squares programming, Hamilton–Jacobi
reachability) scale poorly or require polynomial dynamics. Neural networks
are flexible function approximators, and training can shape $B$ to fit the true
viability kernel. The price is that the result must be **verified**, because
training only enforces the conditions at sampled points. The rest of the
report is about paying that price rigorously.
