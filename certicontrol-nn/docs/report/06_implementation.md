# Part V — Implementation {.unnumbered}

# Architecture and data flow

## Component diagram

```
                  configs/*.yaml  (Hydra)
                          │
                          ▼
scripts/train_and_certify.py ──► make_system() ──► ControlAffineSystem (dynamics, sets, features)
                          │                               │
                          ▼                               ▼
                   CGISTrainer ──────────────► CertificateModule(x) = [B, dB/dt + γB]
                    │  train() ◄── losses()                  ▲   (NeuralController + NeuralControlBarrier)
                    │  verify() ─► BranchAndBoundVerifier ───┘
                    │                 ├─ is_violation()          (float64 evaluation)
                    │                 ├─ BoundEngine.ibp()       (Interval)
                    │                 ├─ BoundEngine.crown()     (auto_LiRPA)
                    │                 └─ z3_check_linear_bounds  (Z3)
                    │  add_counterexamples() ◄── counterexamples
                    ▼
            save_certificate() ──► results/<system>/certificate.pt, certificate.json
                    │
                    ▼
            ProofChecker.check() ──► proof_check.json
                    │
                    ▼
     scripts/evaluate.py, plot_certified_region.py, dashboard/app.py
```

## The "one function, many backends" principle

Only one function is ever verified: `CertificateModule.forward`. It is called
with four kinds of input:

| call | backend | produced by |
|---|---|---|
| `cert(x_tensor)` | exact values; differentiable | training, falsification, simulation |
| `cert(Interval(lo, hi))` | interval enclosures | `BoundEngine.ibp` |
| `BoundedModule(cert)` | auto_LiRPA traces the tensor path | `BoundEngine.crown` |
| `cert(Z3RelaxedVec(...))` | Z3 constraints | `z3_check_box` (engine 2) |

---

# Source code reference

This chapter lists every module and function, with its purpose, inputs and
outputs.

## `src/ops.py`: backend dispatch

| function | tensor behaviour | other backends |
|---|---|---|
| `sin(x)`, `cos(x)`, `tanh(x)` | `torch.sin` etc. | `x.sin()` etc. |
| `dtanh(z, h)` | `1 - h*h` | `z.dtanh()` (uses the pre-activation for tightness) |
| `linear(x, W, b)` | `F.linear` | `x.linear(W, b)` |
| `scale_shift(x, a, c)` | `x*a + c` | `x.scale_shift(a, c)` |
| `cat(xs)` | `torch.cat(dim=1)` | `type(x).cat(xs)` |

## `src/dynamics/continuous_system.py`: systems

**`Box(lo, hi)`** is a dataclass for axis-aligned boxes; `.tensor()` returns
`(lo, hi)` tensors.

**`ControlAffineSystem`** is the base class. Every system has a `domain`,
`u_min`/`u_max`, an `init_set` (bounding box of $X_0$) and a list of `periodic`
coordinates.

| method | purpose |
|---|---|
| `state_dim`, `control_dim`, `feature_dim` | sizes |
| `xdot(x, u)` | **abstract**: closed-loop vector field $f+gu$ (backend-agnostic) |
| `features(x)` / `features_jvp(x, xdot)` | **abstract**: $\psi(x)$ and $J_\psi(x)\,\dot x$ |
| `unsafe_pieces(x)` | **abstract**: list of $h_i(x)$; the unsafe set is $\bigcup\{h_i\le0\}$ |
| `init_constraints(x)` | list of $g_j(x)$; $X_0=$ `init_set` $\cap\{g_j\ge0\}$ (default: none) |
| `f(x)`, `g(x)` | derived drift and input matrix |
| `is_unsafe`, `in_domain`, `in_init` | membership tests on tensors (periodic coordinates are ignored by `in_domain`) |
| `sample_domain`, `sample_box`, `sample_init`, `sample_unsafe`, `sample_boundary` | samplers (rejection sampling for non-box sets) |
| `sample_unsafe_train` | unsafe samples for the loss (possibly inflated) |
| `boundary_faces()` | list of `(dim, side, value)` for non-periodic faces |
| `rhs_numpy(x, u)` | NumPy wrapper for integrators |
| `nominal_control(x)` | optional task controller (default: `None`) |

**`InvertedPendulum(m, l, b, grav, u_max, theta_unsafe, theta_lim, omega_lim, init_theta, init_omega, omega_scale)`**
and **`DubinsCar(arena, v_max, omega_max, a_max, obstacle_center, obstacle_radius, init_xy, init_v, goal, v_des, train_inflation)`**
implement the abstract methods (Chapter 3).

**`make_system(cfg)`** builds a system from a configuration dictionary with a
`name` key. **`_square(x)`** squares tensors and dispatches `.square()` for the
other backends.

## `src/models/`: networks

* **`neural_barrier.py`**
  * `build_tanh_mlp(in_dim, hidden, out_dim)` returns a `ModuleList` of `nn.Linear`.
  * `mlp_forward(layers, h, dh=None)` runs the backend-agnostic forward pass, optionally with the tangent.
  * `NeuralControlBarrier(feature_dim, hidden)` has `forward(s)`, `forward_with_tangent(s, ds)` and `compute_lie_derivative(s, u, f_x, g_x)` (the reverse-mode reference from the specification, used in tests).
* **`neural_controller.py`**: `NeuralController(feature_dim, u_min, u_max, hidden)` stores the buffers `u_mid` and `u_half` and implements `forward(s)`.
* **`certificate.py`**
  * `CertificateModule(system, controller, barrier, gamma)` has `terms(x)` → `(B, dB, u)`, `forward(x)` → `[B, cond]`, and the helpers `barrier_value(x)` and `control(x)`.
  * `build_models(system, ctrl_hidden, barrier_hidden)` constructs both networks.

## `src/verification/interval.py`

* **`Interval(lo, hi)`**: batched intervals of shape `(N, d)`.
  * Arithmetic: `+ - * neg`, with float and tensor constants.
  * `square`, `linear`, `scale_shift`, `tanh`, `dtanh`, `sin`, `cos`.
  * `cat`, `point`, `center`, `radius`, `contains`, `__getitem__`.
* **`_widen(lo, hi)`**: outward rounding.
* **`_interval_sin(lo, hi, shift_err)`**: the sine enclosure of Chapter 7.

## `src/verification/lirpa_verifier.py`

* **`verify_state_box(model, x_center, eps)`** is the specification's helper: CROWN on an $L_\infty$ ball, returning True if every lower bound is $\ge0$.
* **`BoundEngine(cert, use_crown)`** keeps a float64 copy of the certificate and provides:
  * `ibp(lo, hi)`;
  * `crown(lo, hi)`, which returns `{lb, ub, lA, lbias, uA, ubias}`;
  * `unsafe_may_intersect(lo, hi)`;
  * `outside_init(lo, hi)`.
* **`BBResult`** holds the result of one region: leaves, methods, outside flags, counterexamples, unknown boxes, Z3 statistics, time, and whether the budget was exhausted. Its `certified` property is true when there are no counterexamples, no unknown boxes and the budget was not exhausted.
* **`VerificationResult`** aggregates `domain`, `faces` and `init` and provides `status`, `counterexamples()`, `unknown_boxes()`, `proven_volume_fraction()` and `summary()`.
* **`BranchAndBoundVerifier(cert, use_crown, use_z3, tol, batch_size, max_boxes, max_depth, max_cex, z3_max_boxes, z3_timeout_ms, z3_depth, cex_split_depth, smart_split, verbose)`** provides:
  * `run(lo, hi, mode)`: branch and bound over one region;
  * `verify()`: runs the initial set, the domain and every face;
  * `_passes`: the box tests;
  * `_split`: bisection;
  * `_record`: stores leaves.
* **`METHOD_CODES`**: `{"ibp": 0, "crown": 1, "z3": 2}`.

## `src/verification/smt_solver.py`

* **Engine 2** (full-graph relaxation):
  * `_qround` rounds to the dyadic grid.
  * `Z3Context` provides `new_var`, `geq0` and `leq0` (soundly loosened constraints).
  * `Aff` is an affine form $\sum a_iv_i+c$.
  * `Z3RelaxedVec` is the backend class, with McCormick products and `_c1` mean-value relaxations.
  * `z3_check_box(cert, system, lo, hi, mode, tol, timeout_ms)` returns a `Z3BoxResult`.
* **Engine 1** (CROWN linear bounds):
  * `Z3ExactVec` builds exact polynomial expressions.
  * `_exact(float)` converts a float to its exact rational.
  * `z3_check_linear_bounds(system, lo, hi, lA, lbias, uA, ubias, mode, tol, timeout_ms)`.
* **Shared:**
  * `Z3BoxResult(status, candidate, genuine, seconds)`.
  * `_model_float` converts a Z3 value to a float, handling algebraic numbers.
  * `is_violation(cert, system, x, mode)` checks points exactly in float64.
  * `extract_counterexample_z3(...)` is the specification-level helper.

## `src/verification/cgis_loop.py`

* **`TrainConfig`** and **`VerifyConfig`** are dataclasses with every hyperparameter (Chapter 16).
* **`weights_sha256(module)`** hashes the weights.
* **`CGISTrainer(system, gamma, train_cfg, verify_cfg, ctrl_hidden, barrier_hidden, seed)`** provides:
  * `losses`, `train` and `_focused_domain_batch` (training);
  * `_expand` and `add_counterexamples` (the counterexample buffers);
  * `verify(quick)` and `run(max_rounds, verbose, checkpoint, resume)` (the loop);
  * `save_checkpoint`, `load_checkpoint` and `save` (persistence).
* **`save_certificate(out, cert, res, cfg, history)`** writes `certificate.pt` and `certificate.json`.

## `src/verification/proof_checker.py`

* **`load_certificate(path)`** reads `.pt` or `.pt.gz` and rebuilds the system and networks.
* **`CheckReport(ok, details, seconds)`** holds the checker's verdict.
* **`ProofChecker(cert, tol)`** provides `check(verification)`, plus `_coverage`, `_passes`, `_recheck_leaves` and `_check_part`.

## `src/visualization/phase_portrait.py`

* `simulate(cert, x0, T)`: SciPy RK45 rollout.
* `_grid_eval`: evaluates $[B,\mathrm{cond}]$ on a 2-D grid slice.
* `draw_leaves`: draws proof boxes coloured by method, and outlines leaves that lie outside $\mathcal C$.
* `plot_pendulum`, `plot_dubins_slice`, `plot_dubins_trajectories`: the figures.
* `_sample_in_C`: random states with $B\ge0$.

## `scripts/`

| script | what it does |
|---|---|
| `train_and_certify.py` | Hydra entry point: builds everything, runs CGIS, saves, runs the proof checker |
| `reverify.py` | re-verifies stored networks with new budgets, rewrites the certificate and the check report |
| `check_certificate.py` | runs the independent checker; exit code 0 if accepted |
| `evaluate.py` | verifier ablation, $\gamma$ sweep, 2 M-sample check, rollouts, rollouts from counterexamples → `evaluation.json` |
| `plot_certified_region.py` | figures for a system |
| `make_paper_macros.py` | writes the Dubins results into `paper/results_macros.tex` |
| `make_guide_figures.py` | the teaching figures in `docs/figures` |

---

# Configuration reference

The configuration is composed by Hydra from `configs/config.yaml` (defaults)
and `configs/system/<name>.yaml` (per-system overrides, marked
`# @package _global_`). Any key can be overridden on the command line, for
example `python scripts/train_and_certify.py system=dubins_car train.margin=0.02`.

## Top level

| key | default | meaning |
|---|---|---|
| `seed` | 0 | random seed |
| `gamma` | 1.0 | class-$\mathcal K$ slope $\gamma$ |
| `max_rounds` | 20 (40 for the car) | synthesis rounds |
| `output_dir` | `results/${system.name}` | where results go |
| `resume` | false | continue from `checkpoint.pt` |
| `model.ctrl_hidden`, `model.barrier_hidden` | [32, 32] | hidden widths |

## `train.*`

| key | pendulum | car | meaning |
|---|---|---|---|
| `lr` | 1e-3 | 3e-3 | Adam learning rate |
| `steps_initial` / `steps_per_round` | 2000 / 500 | 10000 / 1500 | training steps |
| `batch_domain` | 2048 | 4096 | domain samples per step |
| `batch_unsafe`, `batch_boundary`, `batch_init`, `batch_cex` | 1024 each | 1024 each | other sample sizes |
| `margin` | 0.02 | 0.01 | hinge margin $m$ |
| `lie_band`, `lie_temp` | 0.01, 0.01 | same | band $\delta$ and temperature $T$ |
| `w_lie`, `w_unsafe`, `w_boundary`, `w_init`, `w_cex` | 1, 1, 1, 5, 2 | 1, 10, 10, 10, 5 | loss weights |
| `w_nominal` | 0 | 0.005 | imitation weight |
| `w_lipschitz`, `lip_delta` | 1e-3, 0.01 | same | controller smoothness |
| `w_cert_lip`, `cert_lip_target` | 0, 1.0 | same | optional certificate-slope penalty |
| `focus_fraction` | 0.5 | same | share of band samples |
| `lr_final_ratio` | 0.05 | same | cosine decay floor |
| `weight_decay` | 1e-5 | same | L2 regularisation |
| `cex_jitter`, `cex_copies`, `max_buffer` | 0.02, 16, 200000 | same | counterexample buffer |

## `verify.*`

| key | pendulum | car | meaning |
|---|---|---|---|
| `use_crown`, `use_z3` | true | true | enable engines |
| `tol` | 1e-8 | 1e-8 | margin $\tau$ |
| `batch_size` | 8192 | 16384 | boxes per batch |
| `max_boxes` | 2,000,000 | 6,000,000 | budget per region |
| `max_depth` | 12 | 10 | finest box = width × $2^{-\text{depth}}$ |
| `max_cex` | 512 | 512 | early stop |
| `z3_depth`, `z3_max_boxes`, `z3_timeout_ms` | 4, 20000, 5000 | 6, 5000, 5000 | Z3 gating |
| `cex_split_depth` | 4 | 4 | exploration of falsified boxes |
| `smart_split` | false | false | alternative branching |
| `quick_max_depth`, `quick_max_boxes` | none | 7, 150000 | quick passes during synthesis |

## System parameters

These were listed in Chapter 3. The car additionally has `train_inflation: 0.08`.

---

# Running the project

## Installation

```bash
conda create -n certicontrol python=3.11 -y
conda activate certicontrol
pip install -r requirements.txt
```

* **Python 3.11** is required by current auto_LiRPA (`requires-python ~=3.11`).
* **PyTorch.** auto_LiRPA pins `torch<2.12`. The project was also tested with 2.14.
* **auto_LiRPA** is installed from GitHub. The PyPI package is an old version (0.3).
* **CPU only.** Verification needs float64, which Apple MPS lacks.

## Step-by-step tutorial

1. **Synthesise and certify the pendulum** (2 to 10 minutes):
   `python scripts/train_and_certify.py`. Each round prints loss lines such as
   `step 499 lie=0.0016 unsafe=0.0050 ...`, then a JSON summary. The run ends with
   `[result] CERTIFIED` and `[proof checker] ACCEPTED`.
2. **Check the proof independently:**
   `python scripts/check_certificate.py results/inverted_pendulum/certificate.pt`
   prints per-region details and `CERTIFICATE ACCEPTED`.
3. **Evaluate:** `python scripts/evaluate.py --system inverted_pendulum` writes `evaluation.json` (ablation, $\gamma$ sweep, sampling, rollouts).
4. **Plot:** `python scripts/plot_certified_region.py --system inverted_pendulum`.
5. **Explore interactively:** `streamlit run dashboard/app.py`.
6. **Run the tests:** `pytest` (27 passed and 1 skipped, in about 20 s).
7. **Car** (hours, about 4 GB RAM): `python scripts/train_and_certify.py system=dubins_car`. Resume with `resume=true`. Re-verify with a larger budget with `python scripts/reverify.py --system dubins_car --max-boxes 3000000`.

## Reading a round summary

| field | meaning |
|---|---|
| `status` | CERTIFIED / FALSIFIED / UNKNOWN |
| `domain_boxes_processed` | boxes examined in $D$ |
| `domain_leaves` | proven boxes in $D$ |
| `proven_volume_fraction` | share of $\mathrm{vol}(D)$ proven |
| `leaves_by_method` | how many leaves each engine closed |
| `leaves_outside_safe_set` | leaves proven because $\overline B<0$ |
| `face_leaves`, `init_leaves` | proof sizes for P3 and P0 |
| `init_set_certified` | whether P0 holds |
| `z3_calls`, `z3_proved`, `z3_spurious` | Z3 activity |
| `counterexamples`, `unknown_boxes` | failures |
| `full_verification` | false for quick passes |

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `index_copy_ ... Float and Double` | auto_LiRPA with float32 default dtype | already handled in `BoundEngine.crown`; do not call auto_LiRPA directly on float64 models without setting the default dtype |
| process killed during car verification | out of memory (the run uses about 4 GB; two heavy jobs together exceed 6 GB) | run one job at a time; resume with `resume=true` |
| certificate collapses ($B<0$ everywhere) | P0 weight too small | raise `train.w_init` |
| many UNKNOWN boxes | budget too small, or margins too thin | raise `verify.max_boxes` / `max_depth`, or `train.margin` |
| Z3 hangs | using engine 2 on large networks | use engine 1 (default) |

---

# Tests and the dashboard

## The test suite (`pytest`)

`tests/test_interval_bounds.py` covers bound soundness and differentiation:

* `test_elementwise_ops_are_sound[tanh|sin|cos|dtanh|square]` checks 400 random boxes (widths up to 8) with 2000 samples each: every value lies inside the interval.
* `test_sin_attains_extrema` checks that intervals containing $\pm\frac\pi2$ reach $\pm1$.
* `test_binary_and_linear_ops_are_sound` covers `* + -`, constant scaling, `linear` and `scale_shift`.
* `test_point_interval_contains_exact_value`: degenerate intervals contain the exact value, with width below $10^{-9}$.
* `test_closed_loop_ibp_is_sound[pendulum|car]`: IBP of the whole certificate encloses 1500 samples per box.
* `test_crown_bounds_and_linear_bounds_are_sound[...]`: CROWN's numbers never exceed the true minimum or fall below the true maximum, and its linear functions sandwich the values. CROWN is on average tighter than IBP.
* `test_forward_mode_lie_derivative_matches_autograd` and `test_certificate_lie_derivative_matches_autograd[...]` check forward mode against reverse mode.
* `test_controller_respects_actuator_limits` holds even with weights multiplied by 50, and in interval form.
* `test_spec_verify_state_box_helper` checks the specification helper on positive and negative networks.

`tests/test_z3_verification.py` covers the engines and the certificate:

* `test_z3_finds_genuine_boundary_counterexample`: with $B>0$ everywhere, a face point is found and validated.
* `test_z3_proves_box_where_barrier_is_negative` returns UNSAT.
* `test_z3_unsafe_set_is_encoded_exactly`: a box inside the obstacle yields a validated unsafe counterexample.
* `test_z3_init_mode` covers P0 in both directions.
* `test_z3_unsat_implies_no_sampled_violation`: every UNSAT is confirmed by 5000 samples.
* `test_full_graph_relaxation_encoding_small_net` exercises engine 2 on a 4-neuron network.
* `test_branch_and_bound_never_certifies_a_violated_network`: an untrained network is not certified, and all reported counterexamples are genuine.
* `test_trivially_empty_barrier_is_not_certified`: P0 catches $B<0$ everywhere.
* `test_saved_certificate_passes_proof_checker[pendulum|car]`: the pendulum proof is accepted. The car test is skipped, since that result is not a certificate.
* `test_proof_checker_rejects_tampered_certificate`.

## The dashboard (`dashboard/app.py`)

A Streamlit application in five panels:

1. **Status.** Certificate status, number of proven leaves, Z3 leaves, and the proof checker's verdict. A green or amber banner explains the meaning.
2. **Certificate details** (expandable). Weight hashes, properties, the full summary, the per-round plot of counterexamples and unknown boxes, and the checker's per-region report.
3. **Phase portrait.** For the pendulum, the certified set, the unsafe set, $X_0$, proof leaves and rollouts. For the car, $(p_x,p_y)$ slices at a chosen heading and speed, with rollouts.
4. **Simulation.** Enter any initial state and horizon to see $B(x(t))$ and the states over time, plus a message saying whether the start is inside $\mathcal C$ and whether the unsafe set is reached.
5. **Re-verification.** Choose $\gamma$, the maximum depth and whether to use Z3, then click *Verify now*. The branch-and-bound verifier runs live, and the first counterexamples are listed if any.
