# SAT, SMT and the Z3 engines

## From SAT to SMT

**SAT.** Given a Boolean formula such as $(p\lor\lnot q)\land(q\lor r)$, is there
an assignment of true and false to the variables that makes it true? SAT is
NP-complete. Modern solvers still handle millions of variables, using
**CDCL** (conflict-driven clause learning): guess a value, propagate
consequences, and when a conflict occurs, learn a new clause explaining it and
backtrack.

**SMT** (Satisfiability Modulo Theories) allows atoms from mathematical
*theories* instead of plain Boolean variables:

| theory | example atom | decidable? | typical method |
|---|---|---|---|
| QF_LRA: linear real arithmetic | $2x-3y\le1$ | yes, in polynomial time for conjunctions | simplex |
| QF_NRA: nonlinear (polynomial) real arithmetic | $x^2+y^2\le0.36$ | yes, but expensive | cylindrical algebraic decomposition / nlsat |
| reals with $\sin$, $\tanh$ | $\tanh(x)\ge0.3$ | **no** (in general) | only δ-complete approximations (dReal) |

An SMT solver combines a SAT engine for the Boolean structure (`Or`, `And`)
with theory solvers for the atoms. This architecture is called **DPLL(T)**.

## Z3 basics

Z3 (de Moura and Bjørner, 2008) is Microsoft Research's SMT solver. Its Python
API:

```python
import z3
x = z3.Real("x")
s = z3.Solver()
s.add(x >= 0, x <= 1)
s.add(z3.RealVal("1.4999") * x + z3.RealVal("-1.0435") < 0)
print(s.check())   # sat
print(s.model())   # [x = 0]
```

* `check()` returns `sat`, `unsat` or `unknown` (for example on timeout).
* For `sat`, `model()` gives a witness.
* Reals are **exact rationals**: no rounding happens inside Z3.

## Proving by refutation

To prove $\forall x\in R:\varphi(x)$, ask for $\exists x\in R:\lnot\varphi(x)$:

* **UNSAT**: no counterexample exists, so $\varphi$ holds on all of $R$. This is a proof.
* **SAT**: the model is a candidate counterexample.

The small example above uses the CROWN lower line of the tiny network,
$y\ge1.4999x-1.0435$ on $[0,1]$:

* Asking whether the line can drop below $-1.05$ returns **unsat**, which proves $y\ge-1.05$ on $[0,1]$.
* Asking whether it can drop below $0$ returns **sat** with $x=0$. Evaluating the real network there gives $y(0)=-0.88<0$, so this is a *genuine* counterexample to "$y\ge0$".

## The obstacle: transcendental functions

The certificate contains $\tanh$, $\sin$ and $\cos$. No complete decision
procedure exists for these, and Z3 does not support them natively. We therefore
hand Z3 a **sound relaxation**: a formula whose set of solutions *contains* the
real behaviour. The logic of refutation then works like this:

* relaxation UNSAT ⟹ real problem UNSAT ⟹ **proof** (sound);
* relaxation SAT ⟹ the real problem may or may not be SAT ⟹ the candidate must be **validated**.

The project provides two relaxations.

## Engine 1 (default): pointwise reasoning over CROWN's linear bounds

For an undecided box $R$, CROWN has produced linear functions with
$$B(x)\le\overline A_Bx+\overline b_B,\qquad \mathrm{cond}(x)\ge\underline A_cx+\underline b_c\qquad\forall x\in R .$$
P1 and P2 fail at a point $x$ only if $B(x)\ge0$ and ($\mathrm{cond}(x)<0$ or
$x$ is unsafe). Since $B(x)\ge0$ implies $\overline A_Bx+\overline b_B\ge0$,
and $\mathrm{cond}(x)<0$ implies $\underline A_cx+\underline b_c<0$, every real
violation satisfies
$$\Phi(x):=\ \big(\overline A_Bx+\overline b_B\ge-\tau\big)\ \wedge\ \Big(\underline A_cx+\underline b_c<\tau\ \vee\ \textstyle\bigvee_i h_i(x)\le0\Big),\qquad x\in R .$$
So **UNSAT of $\Phi$ proves P1 and P2 on $R$**. The margins $\tau$ make the
statement slightly stronger, matching the box-level tests.

**Why is this stronger than CROWN's box-level test?** The box test requires
*either* $\max_R(\text{upper line of }B)<0$ *or*
$\min_R(\text{lower line of cond})\ge0$ on the **whole** box. Z3 instead asks
whether a **single point** exists where *both* bad things happen. Consider a box
cut by the boundary of $\mathcal C$, where cond is negative only in the part
where $B$ is negative. The box test fails both alternatives. Z3 returns UNSAT
because the two regions do not overlap. No splitting is needed.

**Exactness choices.**

* Every float coefficient of $\overline A,\underline A$ is converted to the *exact* rational it represents (`fractions.Fraction(float)`). The linear functions are then used without further rounding.
* The unsafe set is built with `Z3ExactVec`, a tiny backend in which `+`, `*` and `square` produce exact Z3 polynomial terms. For the car, $p_x^2+p_y^2-0.36\le0$ is encoded **exactly**. The formula is then in nonlinear real arithmetic, but it has only four variables and one quadratic atom, so Z3 solves it in milliseconds.
* Variants for the other properties: P3 (faces) asks $\exists x: \overline A_Bx+\overline b_B\ge-\tau$. P0 (initial set) asks $\exists x\in X_0: \underline A_Bx+\underline b_B\le\tau$, with the initial-set constraints $g_j(x)\ge0$ also encoded exactly.

**Code** (`src/verification/smt_solver.py::z3_check_linear_bounds`, simplified):

```python
xs = [z3.Real(f"x{i}") for i in range(n)]
for i in range(n):
    solver.add(xs[i] >= _exact(lo[i]), xs[i] <= _exact(hi[i]))     # x in the box
lin = lambda A, b: z3.Sum([_exact(A[i]) * xs[i] for i in range(n)]) + _exact(b)
solver.add(lin(uA[0], ubias[0]) >= _exact(-tol))                  # B may be >= 0
if mode == "domain":
    unsafe = [p.exprs[0] <= 0 for p in system.unsafe_pieces(Z3ExactVec(xs))]
    solver.add(z3.Or([lin(lA[1], lbias[1]) < _exact(tol)] + unsafe))
res = solver.check()     # unsat -> box proven; sat -> candidate
```

**Validating the candidate.** A SAT model is converted to floats and evaluated
on the real float64 network with `is_violation(model, system, x, mode)`:

* **genuine** if $B(x)\ge0$ and ($\mathrm{cond}(x)<0$ or $x$ is unsafe). It is recorded as a counterexample and later used for retraining.
* **spurious** otherwise: the relaxation was too loose, so the box is split.

**Measured effect.** On the pendulum, Z3 closed 11 of the 140 final proof boxes.
In the verifier ablation, the proof shrinks from 165 boxes (CROWN only) to 140.
On the Dubins car, 2,949 leaves were closed by Z3. Each call takes a few
milliseconds.

## Engine 2: full-graph relaxation (`z3_check_box`)

Engine 2 encodes the **entire** closed-loop computation in QF_LRA, using the
`Z3RelaxedVec` backend of `ops`. Every intermediate quantity becomes a Z3
variable with interval bounds, related to its inputs by these constraints:

| operation | encoding |
|---|---|
| affine $y=Wx+b$ | linear equality (two inequalities) |
| smooth scalar $y=f(x)$, $x\in[l,u]$ ($\tanh$, $\tanh'$, $\sin$, $\cos$) | the four mean-value inequalities of Chapter 2, with $[d_{\min},d_{\max}]$ an interval enclosure of $f'$ on $[l,u]$ |
| product $w=ab$ | McCormick envelope |
| square | McCormick, plus $w\ge$ the interval lower bound |

The derivative bounds used are: for $\tanh$, the interval of $\tanh'$; for
$\tanh'$, the interval of $\tanh''=-2\tanh\cdot\tanh'$; for $\sin$, that of
$\cos$; for $\cos$, that of $-\sin$. All are computed with the `Interval`
class.

### Why engine 2 is not the default

In the first version every float coefficient was converted to its exact
rational. Floats have 53-bit mantissas, so these rationals have denominators
up to $2^{52}$ and beyond. Z3's simplex manipulates exact rationals, and during
pivoting their numerators and denominators grow. On a 2×32 network (about 500
variables, 2000 constraints), `check()` did not return within minutes and
ignored its timeout.

The second attempt rounded all coefficients to a dyadic grid $2^{-K}$. To keep
the encoding sound, each constraint $\sum a_iv_i+c\ge0$ is replaced by
$$\sum\tilde a_iv_i+\tilde c+\text{slack}\ge0,\qquad
\text{slack}\ \ge\ \sum_i|a_i-\tilde a_i|\max|v_i|+|c-\tilde c| .$$
This is valid because every variable has known bounds $|v_i|\le\max|v_i|$.
It is implemented in `Z3Context.geq0`, which also adds $10^{-13}|a|$ per term
for float errors made while forming the coefficients. With $K=10$ the solver
was fast but the loosening was far too large. With $K=16$–$30$ it timed out
again.

**Conclusion.** For networks of this size, a full LRA encoding is the wrong tool.
Engine 1 keeps Z3's strengths (exactness, disjunctions, polynomial sets) and
leaves the heavy lifting to CROWN. Engine 2 is kept because it is correct,
demonstrates the fully symbolic approach, and works on small networks. See
`test_full_graph_relaxation_encoding_small_net`, which uses a 4-neuron
barrier.

## The spec-level helper

`extract_counterexample_z3(cert, system, box_min, box_max, mode)` computes CROWN
bounds for one box, runs engine 1, validates the model, and returns the
counterexample or `None`. It matches the interface described in the original
project plan. `test_z3_finds_genuine_boundary_counterexample` and
`test_z3_unsafe_set_is_encoded_exactly` exercise it.

---

# Branch and bound

## The algorithm

Branch and bound (B&B) turns "bounds are loose on big boxes" into a complete
procedure. It maintains a queue of undecided boxes, bounds each one, and splits
the undecided ones. In pseudocode:

```
queue <- [D]
while queue not empty and budget left:
    take a batch of boxes from the front of the queue
    for each box R in the batch:
        if the real network violates the property at centre(R) or at a random point of R:
            record counterexample; if R is large, split it anyway (exploration); continue
        compute IBP bounds;   if they prove R: record leaf (IBP); continue
        compute CROWN bounds; if they prove R: record leaf (CROWN); continue
        if R is small enough for Z3:
            Z3 UNSAT            -> record leaf (Z3); continue
            Z3 SAT and genuine  -> record counterexample; continue
        if R is at the resolution limit: mark R unknown
        else: bisect R along its widest (domain-normalised) coordinate; append both halves
```

## The per-box tests

With $\overline B$ an upper bound of $B$ and $\underline{\mathrm{cond}}$ a lower
bound of cond over the box:

| mode | region | proven if |
|---|---|---|
| `domain` (P1 + P2) | $D$ | $\overline B<-\tau$ (the whole box is outside $\mathcal C$), **or** ($\underline{\mathrm{cond}}\ge\tau$ **and** the box provably misses $X_{\rm unsafe}$) |
| `boundary` (P3) | each non-periodic face of $D$ | $\overline B<-\tau$ |
| `init` (P0) | the box enclosing $X_0$ | $\underline B>\tau$, **or** the box provably misses $X_0$ |

**"Provably misses the unsafe set"** is decided with intervals: the box misses
$\{h_i\le0\}$ if the interval lower bound of $h_i$ on the box is $>0$
(`BoundEngine.unsafe_may_intersect`). For the initial set, the box misses $X_0$
if some $g_j$ has interval upper bound $<0$ (`outside_init`). Boxes intersecting
the unsafe set are allowed only if $\overline B<-\tau$ there.

**Faces are lower-dimensional boxes.** For the face $p_x=2$, the region has
$l_1=u_1=2$. The bisection step never splits a zero-width coordinate, because
its relative width is 0, never the widest.

## Correctness

**Theorem 10.1 (soundness of B&B).** If the algorithm terminates with every
leaf proven, no counterexamples and no unknown boxes, the property holds on the
whole region.

*Proof.* Bisection produces a partition: every point of the original region
lies in at least one leaf. Leaves share only boundary faces, which does not
matter for a "for all points" statement. Each leaf was proven by a sound method
(Chapters 7–9). Hence every point satisfies the property. $\square$

**Termination.** Boxes are never split below `max_depth` in every coordinate,
and `max_boxes` caps total work. The algorithm always terminates, possibly with
UNKNOWN.

## Complexity: the curse of dimensionality

Suppose a region needs boxes of relative width $\varepsilon$ to be proven. A
uniform grid then needs $\varepsilon^{-n}$ boxes in $n$ dimensions:

| $n$ | $\varepsilon=2^{-5}$ | $\varepsilon=2^{-7}$ | $\varepsilon=2^{-10}$ |
|---|---|---|---|
| 2 | 1,024 | 16,384 | $10^6$ |
| 4 | $10^6$ | $2.7\times10^8$ | $10^{12}$ |

B&B is *adaptive*: it refines only where needed, which is why the pendulum proof
has just 140 leaves. In 4-D, however, even adaptive refinement needs millions of
boxes near the difficult boundary. This is the main scalability limit (Chapter
22).

## Implementation details (`BranchAndBoundVerifier.run`)

* **Breadth-first order** (a `collections.deque`) refines the whole region evenly, so a budget-limited run leaves "uniformly unfinished" work instead of one fully refined corner.
* **Batching.** Boxes are processed in batches of 8192 (pendulum) or 16384 (car). IBP and CROWN are vectorised over the batch, which gives a large speed-up on CPU.
* **Early stop on counterexamples.** After `max_cex` (512) counterexamples, the verifier stops. During synthesis there is no point proving the rest of a network that is already known to be wrong.
* **Exploration of falsified boxes.** A falsified large box (relative width above $2^{-\texttt{cex\_split\_depth}}$) is still split, so the rest of its volume is examined and more distinct counterexamples are harvested. An earlier version dropped falsified boxes, and verification of the car then stopped at the root box: its centre (the origin) lies inside the obstacle.
* **Z3 gating.** Z3 is only tried on boxes smaller than $2^{-\texttt{z3\_depth}}$, where CROWN's linear bounds are accurate enough for pointwise reasoning to succeed, up to `z3_max_boxes` calls per run.
* **Relative widths.** Coordinates have different units (radians, metres, m/s), so "widest" is measured relative to the domain width in that coordinate.
* **Alternative branching** (`smart_split`). Split the coordinate $d$ maximising $|A_d|\cdot\text{width}_d$, that is, the coordinate contributing most to the linear bound's range. This was implemented and measured: on the car checkpoint, with the same 150k-box budget, it proved **63.9%** of the volume against **89.2%** for the widest-coordinate rule. It is therefore off by default.
* **Result objects.** `BBResult` stores leaves, methods, whether each leaf is outside $\mathcal C$, counterexamples, unknown boxes and statistics. `VerificationResult` aggregates the domain, all faces and the initial set, computes `status`, and reports `proven_volume_fraction`: the fraction of $\mathrm{vol}(D)$ covered by proven leaves.

---

# The certificate and the proof checker

## What is stored

`save_certificate` (in `cgis_loop.py`) writes two files.

**`certificate.pt`** (a PyTorch pickle, gzip-compressed for the car) contains:

* `controller`, `barrier`: the network weights (`state_dict`);
* `gamma` and `config`: everything needed to rebuild the system and the networks;
* `verification`: for the domain, each face and the initial set, the leaf boxes (`leaves_lo`, `leaves_hi`), the proving method of each leaf (0 = IBP, 1 = CROWN, 2 = Z3), the outside-$\mathcal C$ flags, counterexamples and unknown boxes;
* `history`: one summary per synthesis round.

**`certificate.json`** is a human-readable summary. It includes the **SHA-256
hashes** of the controller and barrier weights and the list of certified
properties in words. Anyone can confirm that the networks they deploy are
byte-for-byte the certified ones.

## The trusted computing base

A proof is only as trustworthy as the code that checks it. The **trusted
computing base** (TCB) here consists of:

* the system definitions and network forward code;
* the interval arithmetic;
* auto_LiRPA's CROWN;
* Z3;
* the checker itself.

The training code, the loss functions and the B&B search heuristics are **not**
in the TCB. The checker ignores them completely.

## What the checker does (`ProofChecker.check`)

1. **Rebuild** the networks from the stored weights via `load_certificate`.
2. **Coverage check** for each region (domain, each face, initial box):
   * every leaf lies inside the region;
   * the total leaf volume equals the region volume, to relative tolerance $10^{-9}$ (for faces, volume is measured in the non-degenerate coordinates).

   Because the leaves come from bisection they have disjoint interiors, so
   *equal volume implies the leaves cover the region* (up to measure-zero
   boundaries).
3. **Re-proof.** Every leaf is re-proven with freshly computed bounds, using its recorded method: IBP leaves with IBP; CROWN leaves with IBP ∩ CROWN; Z3 leaves with a new CROWN computation and a new Z3 call.
4. **Accept** only if coverage holds everywhere, no leaf fails, and there are no recorded counterexamples or unknown boxes.

**Evidence that it works.**

* The pendulum certificate is accepted in 1.5 s.
* `test_proof_checker_rejects_tampered_certificate` adds 3.0 to one controller bias, and the checker rejects the certificate.
* During development, a version of the synthesis loop trained once more *after* the last verification. The checker rejected the resulting certificate because the weights no longer matched the proof. That was a real bug, caught exactly as intended (Chapter 21).
