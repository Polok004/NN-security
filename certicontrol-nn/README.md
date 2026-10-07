# CertiControl-NN

**Sound, machine-checkable safety certificates for neural controllers, built with counterexample-guided synthesis (CGIS).**

CertiControl-NN trains a neural feedback controller $\pi_\theta$ together with a
neural control barrier function (NCBF) $B_\phi$ for a continuous-time system
$\dot x = f(x) + g(x)u$. It then **proves** that the closed loop is safe for
*every* state in a continuous domain, not only for sampled ones. The proof is a
branch-and-bound partition of the state space. On each box, sound bounds from
interval arithmetic (IBP) and from `auto_LiRPA`'s CROWN linear relaxation show
that the barrier conditions hold. Boxes that CROWN alone cannot close go to the
**Z3** SMT solver, which either proves them pointwise or returns a candidate
counterexample. Candidates are checked on the real network, and confirmed ones
are fed back into training until the verifier returns a full proof.

Everything runs locally on CPU (developed for an 8 GB M2 MacBook Air) with open-source tools.

<p align="center"><img src="results/inverted_pendulum/figures/pendulum_certified.png" width="560"></p>

*Inverted pendulum. Green: the certified forward-invariant safe set
$\mathcal C=\{B\ge 0\}$. Boxes: the leaves of the proof, colored by the engine
that closed them (blue = CROWN, orange = Z3). Grey outlines mark leaves proven to
lie outside $\mathcal C$. Black curves: closed-loop rollouts (scipy RK45).*

---

## Documentation

Two documents explain the whole project, both written for readers with no
prior background:

- **[docs/CertiControl-NN_Complete_Report.pdf](docs/CertiControl-NN_Complete_Report.pdf)**
  (75 pages; sources in `docs/report/`). This is the full technical report:
  - the mathematical prerequisites and the dynamics, derived from Newton's laws;
  - the safety theory with proofs (comparison lemma, main theorem);
  - worked numerical examples of interval arithmetic, CROWN, McCormick relaxation and Z3;
  - branch and bound and the proof checker;
  - the training and synthesis loop;
  - a reference for every module and function, and for every configuration key;
  - a tutorial, the tests and the dashboard;
  - the full results, the development log, related work, exercises with solutions and a FAQ.
- **[docs/CertiControl-NN_Guide.pdf](docs/CertiControl-NN_Guide.pdf)** (22
  pages; source `docs/GUIDE.md`). This is a shorter, guided tour.

To rebuild the report:

```bash
cd docs/report && pandoc 0*.md -o ../CertiControl-NN_Complete_Report.pdf \
  --pdf-engine=xelatex --top-level-division=chapter -H ../header_report.tex
```

## What exactly is certified

With $\mathrm{cond}(x) = \nabla B(x)\cdot\big(f(x)+g(x)\pi(x)\big) + \gamma B(x)$, the verifier proves four properties:

| | property | meaning |
|---|---|---|
| **P1** invariance | $\forall x\in D:\; B(x)\ge 0 \Rightarrow \mathrm{cond}(x)\ge 0$ | CBF / Nagumo condition on the safe set |
| **P2** safety | $\forall x\in D\cap X_{\text{unsafe}}:\; B(x)<0$ | the safe set avoids the unsafe set |
| **P3** containment | $\forall x\in\partial D$ (non-periodic faces): $B(x)<0$ | the safe set cannot leak out of the verified domain |
| **P0** non-triviality | $\forall x\in X_0:\; B(x)>0$ | the initial set lies inside $\mathcal C$, which rules out the vacuous certificate $B<0$ everywhere |

**Consequence.** Every trajectory of the closed loop that starts in
$\mathcal C=\{x\in D: B(x)\ge 0\}$, including every trajectory that starts in
$X_0$, stays in $\mathcal C$ for all time. It therefore never enters the unsafe
set and never leaves $D$. By P3, $\mathcal C$ is a compact set in the interior
of $D$. On $\mathcal C$, P1 gives $\dot B\ge-\gamma B$, so by the comparison
lemma $B(x(t))\ge e^{-\gamma t}B(x_0)\ge 0$. The actuator limits hold by
construction, since $\pi(x)=u_{\text{mid}}+u_{\text{half}}\tanh(\cdot)$.

Three consequences of the definitions:

- The stored proof also covers every $\gamma' \ge \gamma$, because $\gamma' B\ge\gamma B$ when $B\ge0$.
- For boxes that lie entirely outside $\mathcal C$ ($\overline B<0$), P1 holds vacuously.
- P1 is **not** required outside $\mathcal C$, where it is generally false.

## Results

| system | dim | networks | status | CGIS rounds | proof leaves (CROWN / Z3) | proof check | verify time |
|---|---|---|---|---|---|---|---|
| inverted pendulum (torque-limited) | 2 | 2×32 tanh ×2 | **CERTIFIED** | 3 | 129 / 11 | ACCEPTED | 2.7 s |
| Dubins car + obstacle + walls | 4 | 2×32 tanh ×2 | **FALSIFIED**: 524 genuine counterexamples; 99.4 % of vol(D) proven; P0 proven | 40 | 378 155 / 2 949 | rejected (as it should be) | 14 min (1.07 M boxes) |

The pendulum problem is set up so that it is not trivial. With $u_{\max}=6 < mgl\sin(0.8)$,
the torque limit, not the unsafe set $|\theta|\ge 0.8$, is what bounds the
largest invariant set.

**Verifier ablation on the final pendulum networks** (`scripts/evaluate.py`):

| bounding method | status | boxes processed | proof leaves | time |
|---|---|---|---|---|
| IBP only | certified | 433 573 | 216 787 | 2.5 s |
| IBP + CROWN (auto_LiRPA) | certified | 329 | 165 | 2.5 s |
| IBP + CROWN + Z3 (full) | certified | 279 | 140 (11 by Z3) | 2.5 s |

CROWN cuts the size of the proof by about 1500×. Z3 then closes boxes that
straddle the boundary of $\mathcal C$. On those boxes the property holds point
by point, but no single bound on the whole box shows it.

**Empirical checks** (not part of the proof; they cross-check it):
- 2 M uniform samples produced 0 violations of P0 to P3.
- 50 + 50 rollouts from $X_0$ and from $\mathcal C$ over a 10 s horizon produced 0 exits from $\mathcal C$.

See `results/*/evaluation.json`.

### The Dubins car was not certified

After 40 CGIS rounds (≈ 4 h on 2 CPU cores), the verifier (`scripts/reverify.py`,
1.07 M boxes, 14 min) reached these results:
- **P0 is proven**: $X_0\subseteq\mathcal C$.
- **P1/P2 are proven on 99.4 % of the domain volume.** That covers 381 104
  leaves, 2 949 of them closed by Z3.
- The run **stopped after 524 validated counterexamples.** No certificate is
  issued, and the proof checker rejects the result.

All the violations sit on the edge of $\mathcal C$ next to the obstacle, at
near-zero speed. There, the car's position cannot change instantly, so $\dot B$
can only be raised through acceleration or turning.

This is exactly the failure mode the project was built to expose:

| check | result |
|---|---|
| 2 M uniform samples | 23 violating states (rate ≈ 1e-5) |
| 100 rollouts from $X_0$ and $\mathcal C$ | none leave $\mathcal C$ |
| rollouts started *at the verifier's counterexamples* | **41 / 50 leave $\mathcal C$** (none hits the obstacle) |

Simulation alone would have called this controller safe. The learned policy is
also conservative: at high speed $\mathcal C$ is small, and the controller
mostly brakes to rest.

<p align="center"><img src="results/dubins_car/figures/dubins_slices.png" width="820"></p>

To continue:

```bash
python scripts/train_and_certify.py system=dubins_car resume=true max_rounds=80
```

The run keeps the checkpoint and the counterexample buffer. Options for closing
the gap are listed under Limitations.

## How it works

```
 dynamics f, g  ──►  CertificateModule(x) = [ B(x) , ∇B(x)·(f+gπ)(x) + γB(x) ]
 (src/dynamics)       (single source of truth: runs on tensors, Intervals, Z3 forms)
                               │
                 ┌─────────────┴─────────────┐
            CGIS training                Branch & bound over boxes of D, ∂D, X0
            (hinge losses +              ┌─ concrete check at 2 points ─► counterexample
             counterexample buffer)      ├─ IBP (own sound interval arithmetic)
                 ▲                       ├─ CROWN linear bounds (auto_LiRPA, float64)
                 │                       ├─ Z3: ∃x∈box: uB(x) ≥ 0 ∧ (lcond(x) < 0 ∨ x∈X_u) ?
                 │                       │      UNSAT ⇒ proven   SAT ⇒ validate candidate
                 └── validated ──────────┴─ otherwise bisect the box
                     counterexamples
                                   CERTIFIED ⇒ certificate.pt + certificate.json
                                            ⇒ independent proof checker (coverage + re-proof)
```

- **Exact Lie derivative, verifiable graph.** $\dot B$ is computed by
  forward-mode differentiation (a Jacobian-vector product through the feature
  map and the tanh MLP: $dz = W\,dh$, $dh' = \tanh'(z)\odot dz$). The whole
  closed loop is therefore an ordinary feed-forward graph of `Linear`, `tanh`,
  `sin`, `cos` and `mul` operations. `auto_LiRPA` bounds it directly, with no
  double back-propagation. `src/ops.py` dispatches the same code to three
  backends (torch tensors, `Interval`, Z3 affine forms), so the object that is
  trained is exactly the object that is verified.
- **Sound IBP** (`src/verification/interval.py`). Bounds are computed in
  float64. Every operation widens its result outward by a bound on its rounding
  error, and `sin`/`cos` find their extrema exactly.
- **CROWN** via `auto_LiRPA` (float64). The verifier keeps both the bounds and
  the linear bounding functions $\underline A x+\underline b \le [B,\mathrm{cond}] \le \overline A x + \overline b$.
- **Z3** (`src/verification/smt_solver.py`). For a box that CROWN leaves open,
  Z3 decides the *pointwise* disjunction
  $\exists x\in\text{box}: \overline B(x)\ge-\tau \wedge (\underline{\mathrm{cond}}(x)<\tau \vee x\in X_u)$.
  It uses the exact rationals of the CROWN coefficients and encodes the unsafe
  set exactly (polynomial arithmetic, e.g. the obstacle disc). UNSAT is a proof
  for the whole box. A SAT model is evaluated on the real network: if it
  violates a condition it is a genuine counterexample, otherwise it is spurious
  and the box is split. A second encoding, `z3_check_box`, encodes the *entire*
  network in QF_LRA (mean-value enclosures for tanh/sin/cos, McCormick
  envelopes for products, dyadic rounding with sound loosening). It is included
  and tested, but it is only practical for small networks.
- **CGIS loss** (`src/verification/cgis_loop.py`). The loss combines hinge
  terms for P0 to P3 and a CGIS term
  $\max(0, m-\mathrm{cond}(x_{\text{cex}}))$ on counterexamples, plus jittered
  copies of each counterexample.
  - The Lie term is weighted by a *smooth* indicator $\sigma((B+\delta)/\tau)$.
    Its gradient in $B$ lets the optimizer **shrink** $\mathcal C$ where the
    condition cannot be met, instead of only pushing up $\dot B$.
  - A small finite-difference Lipschitz penalty keeps the controller from
    becoming bang-bang, which keeps the bounds tight.
- **Proof checker** (`src/verification/proof_checker.py`,
  `scripts/check_certificate.py`). The checker rebuilds the networks from the
  stored weights. It then checks that the leaves cover each region exactly:
  they lie inside it and their volumes sum to its volume. Leaves come from
  bisection, so they are interior-disjoint. Finally it re-proves every leaf with
  its recorded method. Changing a single bias makes the certificate fail (see
  `tests/test_z3_verification.py`).

### Soundness notes (read before relying on a certificate)

- IBP and the Z3 encodings are sound in real arithmetic. IBP accounts for
  float64 rounding by widening outward. Z3 uses exact rationals.
- CROWN is computed by `auto_LiRPA` in float64 **without directed rounding**.
  A margin $\tau=10^{-8}$ is required on every inequality to absorb rounding,
  which is typically around $10^{-13}$. This is standard practice in neural
  network verification, but it falls short of a formal proof about
  floating-point arithmetic.
- The certificate concerns the continuous-time model with the stated
  parameters and the network exactly as stored (weight hashes are recorded in
  `certificate.json`). Model mismatch, sampling and zero-order-hold effects,
  sensor noise and actuator dynamics are **not** covered.

## Installation

```bash
conda create -n certicontrol python=3.11 -y      # auto_LiRPA needs Python 3.11
conda activate certicontrol
pip install -r requirements.txt                  # installs auto_LiRPA from GitHub
```

Everything runs on CPU. The verifier needs float64, and Apple MPS has no float64 support.

## Usage

```bash
# 1. synthesize + certify (Hydra configs in configs/)
python scripts/train_and_certify.py                           # inverted pendulum  (~2-10 min)
python scripts/train_and_certify.py system=dubins_car         # 4-D Dubins car     (hours; ~4 GB RAM)
python scripts/train_and_certify.py system=dubins_car resume=true    # resume from checkpoint
python scripts/train_and_certify.py gamma=2.0 verify.max_depth=14    # any config override

# 2. independently re-check a stored proof
python scripts/check_certificate.py results/inverted_pendulum/certificate.pt

# 3. figures, ablations, Monte-Carlo rollouts
python scripts/plot_certified_region.py --system inverted_pendulum
python scripts/evaluate.py --system inverted_pendulum

# 4. interactive inspector
streamlit run dashboard/app.py

# 5. tests (soundness of bounds, Z3 engine, B&B, proof checker)
pytest
```

The dashboard provides:
- the certificate status and the proof checker's verdict
- the phase portrait, or 2-D slices for the Dubins car, with the leaves of the proof
- simulation from a custom initial state, showing $B(x(t))$ along the rollout
- **re-verification** with a different $\gamma$ or resolution
- the CGIS history

## Repository layout

```
configs/                 Hydra configs (config.yaml + system/{inverted_pendulum,dubins_car}.yaml)
src/ops.py               backend-agnostic primitives (tensor / Interval / Z3)
src/dynamics/            control-affine systems, feature maps + JVPs, sets D, X_u, X_0
src/models/              NeuralController, NeuralControlBarrier, CertificateModule
src/verification/
    interval.py          sound float64 interval arithmetic (IBP)
    lirpa_verifier.py    auto_LiRPA CROWN bounds + branch-and-bound verifier (P0-P3)
    smt_solver.py        Z3 engines (CROWN-bounds + exact unsafe set; full-graph relaxation)
    cgis_loop.py         CGIS trainer, counterexample buffer, checkpoints, certificate I/O
    proof_checker.py     independent certificate checker
src/visualization/       phase portraits, slices, scipy rollouts
scripts/                 train_and_certify, check_certificate, plot_certified_region, evaluate
dashboard/app.py         Streamlit Safety Verification Inspector
tests/                   pytest suite
paper/                   4-page workshop paper (main.tex, main.pdf)
docs/                    complete technical report (report/, CertiControl-NN_Complete_Report.pdf) and short guide
results/                 certificates (Dubins stored as certificate.pt.gz; loaders read both), proof checks, evaluations, figures
```

## Changes from the original plan, and why

- **Z3's role.** Tanh, sin and cos are transcendental, so no complete SMT theory
  can encode the network exactly. Z3 therefore reasons over *sound* relaxations.
  - The default engine combines CROWN's per-box linear bounds with an exact
    encoding of the unsafe set. It is fast (milliseconds per box) and strictly
    stronger than the box-level test.
  - The full-graph QF_LRA encoding is included and tested. On 2×32 networks,
    however, Z3's rational simplex does not finish in reasonable time.
- **P0 added.** Without it, the loop converged to $B<0$ everywhere, a
  "certificate" that proves nothing.
- **Python 3.11** instead of 3.10. The current `auto_LiRPA` requires it.
- **Dubins car domain.** Heading is treated as periodic: the networks see
  $(\cos\theta,\sin\theta)$, so the domain has no artificial $\theta$ faces. The
  walls and the speed limits are faces of $D$ that must never be crossed.

## Limitations

- **The 4-D Dubins car is not certified** (see Results). The remaining
  violations need a controller that uses its acceleration authority right at
  the edge of $\mathcal C$ when the car is almost stopped. Candidate fixes:
  - an explicit velocity-dependent barrier structure
  - larger training margins
  - α/β-CROWN
- Input-space branch-and-bound scales exponentially with the state dimension.
  In 2-D it takes seconds; in 4-D it takes millions of boxes and hours on a
  laptop. Tighter relaxations (α/β-CROWN) and better branching heuristics are
  natural next steps. A coefficient-times-width heuristic is implemented
  (`verify.smart_split=true`), but in our tests it performed worse than
  splitting the widest coordinate.
- The certificate covers only states in $D$. $\mathcal C$ can be much smaller
  than the true viability kernel.
- The floating-point caveat for CROWN described under Soundness notes applies.
