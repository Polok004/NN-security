# Part VI — Experiments {.unnumbered}

# Experimental setup

* **Hardware:** a 2-core cloud CPU with about 6 GB of usable RAM. The code targets an 8 GB M2 MacBook Air.
* **Software:** Python 3.11, PyTorch 2.14 (CPU), auto_LiRPA 0.7.2, z3-solver (Python bindings), SciPy 1.17.
* **Numerics:** training in float32; all verification, falsification and simulation in float64.
* **Networks:** two tanh hidden layers of width 32 for both the controller and the barrier.
* **Class-$\mathcal K$ function:** $\alpha(B)=B$ ($\gamma=1$).
* **Configurations:** exactly those in `configs/` (Chapter 16).

# Results

## Inverted pendulum: certified

**Synthesis history.**

| round | counterexamples | proven volume | status | verification time |
|---|---|---|---|---|
| 0 | 72 | 97.7% | falsified | 67 s (slowed by a parallel job) |
| 1 | 7 | 98.9% | falsified | 2.9 s |
| 2 | 0 | 100% | **certified** | 2.7 s |

**The final proof.**

* **Domain:** 279 boxes processed, 140 leaves. 129 were closed by CROWN and 11 by Z3; 49 of the leaves were proven because the whole box lies outside $\mathcal C$.
* **Z3:** 57 calls, 11 proofs, 46 spurious candidates.
* **Faces:** 7 leaves in total for the four faces.
* **Initial set:** 1 leaf. The whole of $X_0$ is proven in a single box.
* **Checker:** accepted in 1.5 s.

**Verifier ablation** (same networks, `scripts/evaluate.py`):

| bounding | status | boxes processed | leaves | time |
|---|---|---|---|---|
| IBP only | certified | 433,573 | 216,787 | 2.5 s |
| IBP + CROWN | certified | 329 | 165 | 2.5 s |
| IBP + CROWN + Z3 | certified | 279 | 140 (11 by Z3) | 2.5 s |

*Interpretation.* The proof shrinks by a factor of about 1,300 from IBP to CROWN,
and by a further 15% with Z3. Wall-clock times are similar at this tiny scale
because IBP is so cheap per box. In 4-D, box count dominates everything, which
is why tighter bounds matter.

**$\gamma$ sweep:** certified for $\gamma\in\{0.25,0.5,1,2,4\}$. Values above 1
are implied by Corollary 5.3. Values below 1 happen to hold with these
networks too.

**Empirical cross-checks:**

* 2 M uniform samples: 0 violations of P1 or P2; 200 k face samples and 200 k initial-set samples: 0 violations.
* $\mathcal C$ covers 12% of $D$.
* 50 rollouts from $X_0$ and 50 from $\mathcal C$ over 10 s: none left $\mathcal C$, the domain or the safe set. The minimum of $B$ along the rollouts was 0.0003 (from $\mathcal C$) and 0.024 (from $X_0$).

![Pendulum: certified invariant set (green), proof leaves coloured by engine (blue = CROWN, orange = Z3, grey outline = proven outside $\mathcal C$), unsafe set (red), $X_0$ (dashed) and closed-loop rollouts.](../../results/inverted_pendulum/figures/pendulum_certified.png)

**The certified set** is a tilted ellipse. States leaning right but already swinging back (upper
left and lower right in the $(\theta,\dot\theta)$ plane) are safe further out
than states moving away from upright. $\mathcal C$ reaches about
$\theta\approx0.65$ (at negative $\dot\theta$), well short of the unsafe set at
0.8, consistent with the torque limit. The controller is smooth, roughly a
saturated linear state feedback. Z3 leaves cluster where $\partial\mathcal C$
crosses boxes.

## Dubins car: falsified (a partial result)

**Final verification** (after 40 rounds, `scripts/reverify.py`, 14 minutes):

| quantity | value |
|---|---|
| domain boxes processed | 1,073,497 |
| proven leaves | 381,104 (378,155 CROWN, 2,949 Z3) |
| of which outside $\mathcal C$ | 130,220 |
| proven share of $\mathrm{vol}(D)$ | **99.44%** |
| P0 (initial set inside $\mathcal C$) | **proven** (22,399 leaves) |
| validated counterexamples | **524**, which stopped the run early (`max_cex`) |
| unresolved boxes | 277,584 (mostly unexplored because of the early stop) |
| faces (P3) | not run, because the domain was already falsified |
| status / checker | **FALSIFIED** / rejected |

![Counterexamples per synthesis round and proven volume, for both systems.](../figures/cgis_history.png)

![Dubins car: slices of $\{B\ge0\}$ at four headings, for $v=0$ (top) and $v=0.6$ (bottom). At higher speed the set shrinks sharply: the policy is conservative.](../../results/dubins_car/figures/dubins_slices.png)

**Where the counterexamples are.**

* **Distance to the obstacle centre:** between 0.68 and 0.79 in 80% of cases, just outside the training-inflated obstacle (radius 0.68).
* **Speed:** $|v|<0.08$.
* **Heading:** pointing almost straight towards or away from the obstacle, $|\cos(\text{relative angle})|\approx1$.
* **Values:** $B$ between 0 and 0.027; cond between $-0.025$ and 0.

**How visible the failure is empirically.**

| check | result |
|---|---|
| 2 M uniform samples | 23 violations ($1.2\times10^{-5}$) |
| 50 rollouts from $X_0$ and 50 from $\mathcal C$ | 0 left $\mathcal C$ |
| **50 rollouts starting at counterexamples (3 s)** | **41 left $\mathcal C$** (minimum $B=-0.032$); none reached the obstacle |

So the counterexamples are real invariance failures. They are not yet
collisions, thanks to the inflated safety margin, but the *proof* is broken and
the tool correctly refuses to certify.

## Why the car fails: the zero-speed obstruction

Consider a state on $\partial\mathcal C$ near the obstacle with $v=0$. Its
position cannot change instantly: $\dot p_x=v\cos\theta=0$ and
$\dot p_y=v\sin\theta=0$. So
$$\dot B=\partial_\theta B\cdot u_1+\partial_vB\cdot u_2 .$$
Only turning and accelerating can raise $B$. Now suppose the car is pointing
straight at the obstacle. Turning left or right is symmetric, so
$\partial_\theta B\approx0$ there. The condition then requires
$\partial_vB\cdot u_2\ge\tau$: the barrier must reward reversing
($\partial_vB<0$), and the controller must actually reverse ($u_2<0$).

A diagnostic on a mid-run checkpoint (round 18) found $|\partial_vB|\approx0.3$
at the violating states. The controller used only $|u_2|\le0.17$ of the
available 2. A stronger reversing command would have satisfied the condition
with a large margin.

So a solution exists in principle. The training did not find it: the
violating region is thin (sampling rate about $10^{-5}$), and the imitation and
smoothness terms mildly oppose large accelerations there.

**Things that were tried.**

1. Smaller margins and higher set-condition weights: fixed P0, P2 and P3 violations, but not P1.
2. A smaller initial set (1.4 instead of 1.6, speed limit 0.25 instead of 0.3): more slack near the walls.
3. Training-only obstacle inflation (0.08): moved the violations *outward* with the boundary, but did not remove them.
4. A lower imitation weight (0.02 → 0.005) and a higher counterexample weight (2 → 5): the counterexample count fell to about 5 per quick round, but never reached zero.

**Ideas for finishing it.**

* Build velocity-awareness into $B$, for example $B=\mathrm{NN}(x)-k\,v\cos\varphi$, where $\varphi$ is the heading relative to the obstacle direction.
* Larger margins together with longer training.
* α/β-CROWN bounds, to reduce the number of boxes.
* Handle $v\approx0$ near obstacles with a separate analytic braking argument.

---

# Development log: problems encountered and how they were solved

This chapter records the real engineering history. Each item is a lesson that
generalises to other projects.

| # | problem observed | root cause | fix | lesson |
|---|---|---|---|---|
| 1 | auto_LiRPA crashed with a float/double mismatch | its slice operator allocates buffers with the default dtype | set the default dtype to float64 during CROWN calls | test verification libraries at the precision you intend to use |
| 2 | Z3 `check()` hung for minutes and ignored its timeout | exact rationals from floats (denominators up to $2^{52}$) blow up in simplex | engine 1: small Z3 problems over CROWN's linear bounds; engine 2 kept for small networks | put each tool where it is strong |
| 3 | IBP left 36,000 unknown boxes even at depth 12 | dependency problem, and a bang-bang controller with huge local slope | CROWN, plus a controller Lipschitz penalty | verification-friendly training matters as much as the verifier |
| 4 | first "certificate" proved nothing: $B<0$ everywhere | P1–P3 are vacuous for an empty set | added P0 ($X_0\subseteq\mathcal C$) to the verifier; raised the init weight | always certify non-triviality |
| 5 | loss plateaued; thin "arms" of $\mathcal C$ into infeasible regions | hard mask gives no gradient to shrink $\mathcal C$ | sigmoid band weight | differentiate through set membership |
| 6 | wide band (0.2) stalled training | demanded the condition outside $\mathcal C$, which is infeasible | band 0.01 | match hyperparameters to the scale of $B$ |
| 7 | verification of the car stopped at the root box | a falsified box was dropped without splitting; the root centre lies inside the obstacle | keep splitting large falsified boxes | falsification should not stop exploration |
| 8 | the pendulum stopped converging after that change | harvesting 500+ counterexamples per round from tiny boxes flooded the buffer | only split falsified boxes down to `cex_split_depth=4` | too many near-duplicate counterexamples distort training |
| 9 | the car run was killed | out of memory: two heavy jobs together exceeded the 6.27 GB container limit | run jobs serially; checkpoint and resume | long jobs must be resumable |
| 10 | the checker rejected the final car certificate | the loop trained once more after the last verification | stop right after the last verification | the independent checker catches pipeline bugs |
| 11 | rollouts reported "left the domain" while $B$ stayed high | `in_domain` treated the periodic $\theta$ as bounded | ignore periodic coordinates | periodic coordinates need care everywhere |
| 12 | smarter branching was *worse* (63.9% vs 89.2% volume) | $|A|\cdot$width does not measure relaxation looseness | kept widest-coordinate splitting | measure heuristics before adopting them |

---

# Part VII — Perspective {.unnumbered}

# Related work, limitations and future work

## Related approaches

| approach | verifier | networks | guarantee | comparison with this project |
|---|---|---|---|---|
| Neural Lyapunov Control (Chang et al., 2019) | dReal (δ-complete SMT) | small tanh | δ-sound | dReal handles tanh natively but scales poorly; we use CROWN plus Z3 |
| FOSSIL (Abate et al., 2021) | Z3 / dReal CEGIS | small | sound (δ for dReal) | same CEGIS idea; we add bound propagation and a proof checker |
| BBV NCBF synthesis (Wang et al., 2023) | branch and bound with linear bounds | ReLU / smooth | sound (floating-point caveats) | closest in spirit; we add the pointwise Z3 step and P0 |
| Survey (Dawson, Gao, Fan, 2023) | various | — | — | broad overview of learned certificates |
| CBF-QP safety filters (Ames et al.) | none (online QP) | analytic $B$ | depends on a valid $B$ | complementary: filters need a certified $B$ |

## Limitations

1. **Floating point.** CROWN has no directed rounding, so the $10^{-8}$ margin is a heuristic safeguard (Chapter 8).
2. **Model fidelity.** The guarantee is for the ODE model: no sampling, delay, noise or model error.
3. **Scalability.** Box branch and bound is exponential in the state dimension. The 4-D car needs millions of boxes.
4. **Conservatism.** $\mathcal C$ is an inner approximation of the viability kernel; it covers 12% of $D$ for the pendulum.
5. **No convergence guarantee** for CGIS (Chapter 13).
6. **Proof checker trust base.** The checker reuses the project's bound code (IBP, CROWN, Z3). It checks consistency and coverage independently of training, but it is not a formally verified kernel.

## Future work

* α/β-CROWN or LP-based relaxations, with optimised slopes.
* Directed rounding or interval-validated CROWN.
* Barrier structures with physics priors (velocity terms), for the car.
* Discrete-time or sampled-data certificates, for real deployment.
* Larger systems (quadrotor, 6–12 states) via learned or partitioned domains.
* GPU batching for bound propagation.

---

# Appendix A — Notation {.unnumbered}

| symbol | meaning |
|---|---|
| $x\in\mathbb R^n$ | state |
| $u\in\mathbb R^m$ | control input |
| $f(x),\,g(x)$ | drift, input matrix |
| $\pi_\theta$ | neural controller with weights $\theta$ |
| $B_\phi$ | neural barrier with weights $\phi$ |
| $\psi(x)$ | feature map |
| $F_{cl}$ | closed-loop vector field $f+g\pi$ |
| $\dot B,\ L_FB$ | Lie derivative $\nabla B\cdot F_{cl}$ |
| $\gamma$ | class-$\mathcal K$ slope |
| $\mathrm{cond}$ | $\dot B+\gamma B$ |
| $D$ | verification domain (box) |
| $\mathcal C$ | certified safe set $\{x\in D:B\ge0\}$ |
| $X_0,\ X_{\rm unsafe}$ | initial and unsafe sets |
| $m$ | training margin |
| $\tau$ | verification margin ($10^{-8}$) |
| $\delta,\ T$ | band width and temperature |
| $[l,u]$, $c$, $r$ | box, centre, radius |
| $\underline{\,\cdot\,},\ \overline{\,\cdot\,}$ | lower and upper bound |
| $\underline A,\ \underline b$ | CROWN linear lower bound |
| $u$ (numerics) | unit roundoff |

# Appendix B — Glossary {.unnumbered}

**Actuator saturation.** Physical input limits, enforced here by $\tanh$.
**Adam.** An adaptive gradient optimiser.
**auto_LiRPA.** A library for linear relaxation bounds of PyTorch graphs.
**Barrier certificate.** A function proving invariance and separation from unsafe states.
**Bisection.** Splitting a box in half along one coordinate.
**Branch and bound.** Recursive splitting plus bounding until every piece is decided.
**Candidate (spurious).** A Z3 model of a relaxation that is not a real violation.
**CDCL.** Conflict-driven clause learning (SAT solving).
**CEGIS / CGIS.** Counterexample-guided inductive synthesis.
**Certificate.** Stored networks plus proof leaves plus metadata.
**Class-$\mathcal K$.** Continuous, strictly increasing, zero at zero.
**Closed loop.** Dynamics with the controller substituted.
**Comparison lemma.** Differential inequality ⇒ bound on the solution.
**Completeness.** A method is complete if it proves every true statement.
**Concretization.** Optimising a linear bound over a box.
**Control-affine.** $\dot x=f+gu$.
**CROWN.** Backward linear bound propagation.
**Dependency problem.** IBP's loss of correlations.
**DPLL(T).** SAT search combined with a theory solver.
**Dual numbers.** $a+b\varepsilon$ with $\varepsilon^2=0$ (forward-mode AD).
**Enclosure.** Guaranteed lower and upper bounds.
**Falsification.** Finding a concrete violating input.
**Forward invariance.** A trajectory never leaves the set.
**Hinge loss.** $\mathrm{ReLU}(m-q)$.
**IBP.** Interval bound propagation.
**JVP.** Jacobian-vector product.
**Leaf.** A proven box in the branch-and-bound partition.
**Lie derivative.** $\nabla B\cdot F$.
**Lipschitz.** A bounded rate of change.
**McCormick envelope.** Linear hull of a bilinear product.
**Mean value theorem.** $f(y)-f(x)=f'(\xi)(y-x)$.
**Nagumo's theorem.** Invariance holds iff the flow points inward on the boundary.
**Over-approximation.** A superset of the true behaviour.
**QF_LRA / QF_NRA.** Linear and polynomial real arithmetic theories.
**RK45.** Adaptive Runge–Kutta integrator.
**SAT / UNSAT.** Satisfiable / unsatisfiable.
**SMT.** Satisfiability modulo theories.
**Soundness.** A method is sound if it never claims a false proof.
**TCB.** Trusted computing base.
**Viability kernel.** The largest set from which safety can be maintained.
**Z3.** An SMT solver from Microsoft Research.

# Appendix C — Exercises with solutions {.unnumbered}

**C.1 (interval arithmetic).** Compute the interval of $x(1-x)$ for $x\in[0,1]$, and
compare it with the true range.
*Solution.* $1-x\in[0,1]$ and $[0,1]\cdot[0,1]=[0,1]$. The true range is
$[0,0.25]$, with the maximum at $x=0.5$. The overestimate comes from the
dependency problem.

**C.2 (Lemma 2.1).** Find $\min_{x\in[0,1]\times[-1,2]}(3x_1-2x_2)$.
*Solution.* Take $x_1=0$ (positive coefficient) and $x_2=2$ (negative
coefficient), giving $-4$. Check with the formula: $c=(0.5,0.5)$,
$r=(0.5,1.5)$, $a^\top c=0.5$, $|a|^\top r=1.5+3=4.5$, so the minimum is $-4$.
✓

**C.3 (McCormick).** For $a\in[1,2]$ and $b\in[-1,1]$, give the McCormick bounds at
$a=1.5$, $b=0$.
*Solution.* The lower bounds are $w\ge a_lb+ab_l-a_lb_l=0-1.5+1=-0.5$ and
$w\ge a_ub+ab_u-a_ub_u=0+1.5-2=-0.5$. The upper bounds are
$w\le a_ub+ab_l-a_ub_l=0-1.5+2=0.5$ and $w\le a_lb+ab_u-a_lb_u=0+1.5-1=0.5$. So
$w\in[-0.5,0.5]$, which contains the true value 0.

**C.4 (Lie derivative).** $B(x)=1-x_1^2-x_2^2$ with $\dot x_1=x_2$, $\dot x_2=-x_1-x_2$.
Is $\{B\ge0\}$ invariant?
*Solution.* $\dot B=-2x_1x_2-2x_2(-x_1-x_2)=2x_2^2\ge0$ everywhere. So $B$
never decreases, and the unit disc is invariant.

**C.5 (why P0).** Show that $B\equiv-1$ satisfies P1–P3.
*Solution.* P1: $B\ge0$ never holds, so the implication is vacuously true.
P2 and P3: $B=-1<0$. So without P0 the empty set would be "certified".

**C.6 (γ monotonicity).** Why does a proof for $\gamma=1$ imply one for
$\gamma=3$, but not for $\gamma=0.5$?
*Solution.* On $\mathcal C$, $B\ge0$, so $\dot B+3B\ge\dot B+B\ge0$. For
$\gamma=0.5$ the term $0.5B$ is smaller, and the inequality may fail.

**C.7 (hands-on).** Set `train.w_init=0` and train the pendulum. Explain what
happens.
*Expected.* $B$ is driven negative everywhere. The verifier then reports
`init_set_certified: false`, so no certificate is issued.

**C.8 (hands-on).** In `losses`, replace the sigmoid band with
`(B >= -tc.lie_band).float()`. Plot $\mathcal C$ after training.
*Expected.* Thin arms of $\mathcal C$ reach into regions where the pendulum
cannot be saved, and verification finds counterexamples at their tips.

# Appendix D — Frequently asked questions {.unnumbered}

**Is the pendulum controller safe on real hardware?** The proof covers the ODE
model with the stated parameters. A real system adds discrete-time control,
delays and model error. Those need additional margins or a sampled-data
analysis.

**Why not use ReLU networks, which are easier to verify?** $\dot B$ needs
$\nabla B$, which is discontinuous for ReLU networks. Smooth activations keep
the CBF theory standard.

**Why not verify on a very fine grid?** A grid is still a finite sample.
Between grid points, a function can do anything unless you also bound its
variation, which is exactly what interval and CROWN bounds do.

**What does UNKNOWN mean in practice?** The budget ran out. Increase
`max_boxes` or `max_depth`, or retrain with larger margins.

**How do I add a new system?** Subclass `ControlAffineSystem`, implement
`xdot`, `features`, `features_jvp` and `unsafe_pieces` (optionally
`init_constraints` and `nominal_control`) using `ops.*`, register it in
`make_system`, and add `configs/system/<name>.yaml`.

**How much does each component cost?** Per box: IBP about a microsecond,
CROWN about 0.5 ms (batched), Z3 a few ms. Training a round: seconds to
minutes.

# Appendix E — References {.unnumbered}

1. A. D. Ames, S. Coogan, M. Egerstedt, G. Notomista, K. Sreenath, P. Tabuada. *Control barrier functions: theory and applications.* ECC, 2019.
2. S. Prajna, A. Jadbabaie. *Safety verification of hybrid systems using barrier certificates.* HSCC, 2004.
3. M. Nagumo. *Über die Lage der Integralkurven gewöhnlicher Differentialgleichungen.* Proc. Phys.-Math. Soc. Japan, 1942.
4. H. Zhang, T.-W. Weng, P.-Y. Chen, C.-J. Hsieh, L. Daniel. *Efficient neural network robustness certification with general activation functions.* NeurIPS, 2018.
5. K. Xu, Z. Shi, H. Zhang, et al. *Automatic perturbation analysis for scalable certified robustness and beyond.* NeurIPS, 2020.
6. S. Gowal et al. *On the effectiveness of interval bound propagation for training verifiably robust models.* arXiv:1810.12715, 2018.
7. L. de Moura, N. Bjørner. *Z3: an efficient SMT solver.* TACAS, 2008.
8. G. P. McCormick. *Computability of global solutions to factorable nonconvex programs: Part I.* Mathematical Programming, 1976.
9. A. Solar-Lezama, L. Tancau, R. Bodík, S. Seshia, V. Saraswat. *Combinatorial sketching for finite programs.* ASPLOS, 2006.
10. G. Katz, C. Barrett, D. Dill, K. Julian, M. Kochenderfer. *Reluplex: an efficient SMT solver for verifying deep neural networks.* CAV, 2017.
11. Y.-C. Chang, N. Roohi, S. Gao. *Neural Lyapunov control.* NeurIPS, 2019.
12. A. Abate, D. Ahmed, A. Edwards, M. Giacobbe, A. Peruffo. *FOSSIL.* HSCC, 2021.
13. C. Dawson, S. Gao, C. Fan. *Safe control with learned certificates: a survey.* IEEE Transactions on Robotics, 2023.
14. X. Wang, L. Knoedler, F. B. Mathiesen, J. Alonso-Mora. *Simultaneous synthesis and verification of neural control barrier functions through branch-and-bound verification.* arXiv:2311.10438, 2023.
15. N. J. Higham. *Accuracy and Stability of Numerical Algorithms*, 2nd ed. SIAM, 2002.
16. S. Gao, S. Kong, E. M. Clarke. *dReal: an SMT solver for nonlinear theories over the reals.* CADE, 2013.
