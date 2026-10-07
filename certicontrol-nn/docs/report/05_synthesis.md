# Part IV — Synthesis {.unnumbered}

# Learning the controller and the barrier

## The learning problem

We want weights $(\theta,\phi)$ such that P0–P3 hold. Training cannot enforce
"for all $x$". It can only penalise violations at **sampled** states. Training
therefore produces *candidates*, and the verifier judges them.

## Hinge losses with margins

Each property "$q(x)\ge0$" becomes the hinge loss $\mathrm{ReLU}(m-q(x))$. It is
zero when $q\ge m$ and grows linearly with the violation. The margin $m>0$ asks
for **slack** beyond the verifier's requirement:

* the verifier needs strict inequalities, with margin $\tau$;
* bounds are loose, so a property that holds with value $10^{-6}$ is practically unprovable, while one that holds with value $10^{-2}$ is provable on moderately sized boxes.

Margins used: $m=0.02$ (pendulum) and $m=0.01$ (car).

## The loss terms (`CGISTrainer.losses`)

| term | formula | samples | enforces |
|---|---|---|---|
| `lie` | $\sum \mathrm{ReLU}(m-\mathrm{cond}(x))\,w(x)\ /\ \sum w(x)$ | domain (mixed, see below) | P1 |
| `unsafe` | mean $\mathrm{ReLU}(B(x)+m)$ | unsafe set (inflated for the car) | P2 |
| `boundary` | mean $\mathrm{ReLU}(B(x)+m)$ | faces of $D$ | P3 |
| `init` | mean $\mathrm{ReLU}(m-B(x))$ | $X_0$ | P0 |
| `cgis` | same hinges on buffered counterexamples | CGIS buffer | all |
| `lip` | mean of $\big(\frac{\|\pi(x+d)-\pi(x)\|/\mathrm{span}}{\|d\|/\mathrm{width}}\big)^2$ | domain | smooth controller |
| `nominal` | mean $\|(\pi(x)-u_{\rm nom}(x))/\mathrm{span}\|^2$ | domain | task (car only) |

The total loss is the weighted sum
$w_{\rm lie}\mathcal L_{\rm lie}+w_{\rm unsafe}\mathcal L_{\rm unsafe}+\dots$.
The CGIS term implements the formula from the project plan,
$\mathcal L_{\rm CGIS}=\max\big(0,-(\dot B+\gamma B)\big)$ at counterexamples,
with the margin added.

## The smooth band: why the Lie term uses a sigmoid

P1 matters only where $B\ge0$. The obvious loss uses a **hard mask**
$\mathbb 1[B(x)\ge-\delta]$. The mask is piecewise constant in $B$, so its
gradient with respect to $\phi$ (the barrier's weights) is zero. The optimiser
can then reduce a violation at a state $x$ only by **increasing
$\mathrm{cond}(x)$**. At states where no admissible input can do that (a
pendulum beyond the torque limit, falling), the loss pushes forever without
success. The observed symptom was thin "arms" of $\mathcal C$ extending into
infeasible regions. Those arms made verification impossible.

The fix is a smooth weight $w(x)=\sigma\big((B(x)+\delta)/T\big)$, with
$\delta=0.01$ and temperature $T=0.01$. Now
$$\frac{\partial}{\partial B}\Big[\mathrm{ReLU}(m-\mathrm{cond})\,w\Big]
\ni\ \mathrm{ReLU}(m-\mathrm{cond})\,\frac{\sigma'(\cdot)}{T}\ >0$$
wherever the condition is violated. Gradient descent therefore also **lowers
$B$** at such states, *shrinking* $\mathcal C$ to exclude them. This single
change turned a non-converging loop into a pendulum certificate within 3
rounds.

**A small band matters too.** An earlier version used $\delta=0.2$ on a barrier
whose values lay mostly in $[-0.9,0.1]$. That required $\mathrm{cond}\ge m$ on a
large region *outside* $\mathcal C$, including falling states where it is
physically impossible. Training stalled.

## Sampling strategy

Each training step draws fresh samples:

* **domain batch** (2048 or 4096 states). Half are uniform in $D$. The other half come from the band $B\ge-\delta$, found by rejection sampling from 4× as many candidates (`_focused_domain_batch`). P1 is decided in the band, so that is where samples help most;
* 1024 samples each from the unsafe set, the faces and $X_0$ (rejection sampling for the non-box sets);
* up to 1024 domain counterexamples, plus 256 face and 256 initial-set counterexamples, from the CGIS buffers.

**Training-only inflation of the obstacle** (car, `train_inflation=0.08`).
Unsafe samples are drawn from a disc of radius $0.6+0.08$. This pushes
$\mathcal C$ away from the obstacle surface, where the condition is hardest
(Chapter 20). Verification always uses the true radius 0.6.

## The controller smoothness penalty

Without regularisation, the pendulum controller became **bang-bang**: $u$
jumped from $+6$ to $-6$ across a thin surface. Over a box only 0.002 wide the
controller output varied by 0.15, so interval bounds on everything downstream
were hopeless. The finite-difference penalty
$$\mathcal L_{\rm lip}=\mathbb E\Big[\Big(\tfrac{\|\pi(x+d)-\pi(x)\|/(u_{\max}-u_{\min})}{\|d/(\text{domain width})\|}\Big)^2\Big],\qquad d\sim\mathcal N(0,(0.01\cdot\text{width})^2),$$
with weight $10^{-3}$, keeps the controller smooth enough to verify at little
cost in performance.

An optional **verification-friendly** penalty on the local slopes of
$[B,\mathrm{cond}]$ (`w_cert_lip`) is also implemented. It is off by default,
because diagnostics showed the car's CROWN looseness came from the relaxation
itself, not from steep functions.

## The nominal controller (car)

To make the car do something useful, a weak imitation term pulls $\pi$ towards
a goal-seeking controller:
$$\theta_{\rm des}=\operatorname{atan2}(g_y-p_y,\ g_x-p_x),\quad
u_1^{\rm nom}=\operatorname{clip}(2\cdot\mathrm{wrap}(\theta_{\rm des}-\theta)),\quad
u_2^{\rm nom}=\operatorname{clip}(2(v_{\rm des}-v)).$$
Its weight (0.02, later 0.005) is small, so safety terms dominate wherever
they conflict.

## Optimisation

* **Adam** with $\beta=(0.9,0.999)$ and weight decay $10^{-5}$. Adam keeps running averages of the gradient ($m_t$) and of its square ($v_t$), and updates with $\theta\leftarrow\theta-\eta\,\hat m_t/(\sqrt{\hat v_t}+\epsilon)$, which adapts the step size per parameter.
* **Cosine learning-rate decay** in each training phase: $\eta_t=\eta_{\min}+\frac{\eta_0-\eta_{\min}}{2}(1+\cos\frac{\pi t}{T})$, with $\eta_{\min}=0.05\,\eta_0$. With a constant rate, violations kept fluctuating around small non-zero values. Decay lets them settle.
* Learning rates: $10^{-3}$ (pendulum) and $3\times10^{-3}$ (car). Initial phase: 2000 steps (pendulum) or 10000 (car). Each later round: 500 or 1500 steps.
* **Loss weights.** $w_{\rm init}=5$ (pendulum) and $10$ (car), because P0 otherwise loses to the counterexample terms, which push $B$ down; $w_{\rm unsafe}=w_{\rm boundary}=10$ for the car; $w_{\rm cex}=2$ (pendulum), 5 (car).

---

# Counterexample-guided inductive synthesis (CGIS)

## Origin of the idea

CEGIS (*counterexample-guided inductive synthesis*) comes from program
synthesis (Solar-Lezama et al., 2006). A **learner** generalises from finitely
many examples. A **verifier** checks the candidate against the full
specification and returns a counterexample when it fails. The counterexample
becomes a new example. Neural Lyapunov and barrier synthesis (Chang et al.
2019; Abate et al. 2021) adopted it with SMT verifiers. This project uses a
branch-and-bound verifier with Z3 for the hard boxes.

## The loop (`CGISTrainer.run`)

```
train(steps_initial)
for round r = 0, 1, ...:
    res = verify(quick)                       # coarse falsification pass (if configured)
    if quick and no counterexamples:
        res = verify(full)                    # the real proof attempt
    log the round summary
    if res is CERTIFIED (full), or r is the last round: stop
    add counterexamples (+ jittered copies) and centres of unknown boxes to the buffers
    train(steps_per_round)
    save checkpoint
save certificate; run the independent proof checker
```

## Design decisions

* **Counterexample neighbourhoods.** Each counterexample is copied 16 times with Gaussian jitter (2% of the domain width). Face counterexamples keep their fixed coordinate on the face. Initial-set copies are filtered to stay inside $X_0$. Training on the neighbourhood removes the violation region, not just one point.
* **Unknown boxes as hard examples.** The centres of up to 2048 unknown boxes are added too. They mark regions where the margin is thin.
* **Buffer cap.** At most 200,000 points, first-in-first-out.
* **Quick vs full verification.** For the car, one full verification took around 15 minutes or more. Intermediate rounds therefore run a quick pass (maximum depth 7, 150k boxes) whose purpose is to find counterexamples. Only a quick pass with **zero** counterexamples triggers the full proof. A quick pass can never certify on its own.
* **Checkpointing and resume.** After every round, the weights, optimiser state, buffers and history go to `checkpoint.pt`. `resume=true` continues from it. This saved the Dubins run after an out-of-memory kill.
* **Saved weights equal verified weights.** The loop stops *immediately after* the last verification. An earlier version trained once more first, so the saved networks differed from the verified ones and the proof checker rejected them.

## Convergence

CGIS has **no convergence guarantee**. The learner may keep producing new
failures, or the network class may be unable to satisfy the conditions with the
required margins. In this project:

* **Pendulum:** 72 → 7 → 0 counterexamples, certified in round 2 (3 rounds counting from 0).
* **Car:** counterexamples fell to about 5 per quick round, but never reached zero within 40 rounds (Chapter 20).
