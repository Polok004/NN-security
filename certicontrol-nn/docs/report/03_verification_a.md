# Part III — Verification {.unnumbered}

# Verification fundamentals

## The verification problem

After training, we must decide statements of the form
$$\forall x\in R:\ h(x)\ge0,$$
where $R$ is a box and $h$ is built from the networks and the dynamics, for
example $h=\mathrm{cond}$. There are uncountably many $x$, so we cannot check
them one by one. We need **symbolic** reasoning about whole sets.

## Soundness and completeness

* **Sound**: if the verifier says "proven", the statement is true. *A verifier that is not sound is worse than none.*
* **Complete**: every true statement can be proven.
* **Decidability and cost.** Even for ReLU networks, exact verification is NP-complete (Katz et al., *Reluplex*, 2017). For networks with $\tanh$ and $\sin$, exact reasoning is not decidable in general, because these are transcendental functions.

CertiControl-NN is **sound and incomplete**. It returns one of three verdicts:

| verdict | meaning | strength |
|---|---|---|
| **CERTIFIED** | all properties proven on every box | a proof |
| **FALSIFIED** | at least one concrete state violates a property, checked on the real network | a disproof |
| **UNKNOWN** | neither, within the computing budget | no conclusion |

## Over-approximation: the central idea

An **enclosure** of $h$ on $R$ is a pair $\underline h,\overline h$ with
$\underline h\le h(x)\le\overline h$ for all $x\in R$. Then:

* if $\underline h\ge0$, then $h\ge0$ on all of $R$: **proven**;
* if some concrete $x\in R$ has $h(x)<0$: **counterexample**;
* otherwise the enclosure is too loose: **split** $R$ and try again.

Enclosures get tighter as boxes shrink. Chapter 10 turns this into a complete
algorithm, branch and bound. Chapters 7–9 are about computing enclosures:
cheaply (intervals), tightly (linear relaxations), and pointwise (SMT).

## Falsification is easy; proof is hard

Checking one point is trivial: evaluate the network. Every verifier in this
project therefore **tries falsification first**, at the centre of each box and
at one random point. Real violations are found cheaply, and the expensive
bounding work is spent only on boxes that might be provable.

---

# Interval arithmetic and interval bound propagation (IBP)

## Intuition

Replace each number by an interval guaranteed to contain it. Define interval
versions of each operation that return intervals guaranteed to contain every
possible result. Push the input box through the whole computation. The final
interval is an enclosure.

## Definitions and the fundamental theorem

An interval is $\mathbf a=[\underline a,\overline a]$. For an operation
$\circ$, the interval extension must satisfy the **inclusion property**:
$$a\in\mathbf a,\ b\in\mathbf b\ \Rightarrow\ a\circ b\in\mathbf a\circledcirc\mathbf b .$$

**Fundamental theorem of interval arithmetic.** If every elementary operation
satisfies the inclusion property, then evaluating a composite expression in
interval arithmetic produces an enclosure of the expression's range.

*Proof sketch.* By induction on the expression tree: each intermediate
interval contains every value the corresponding sub-expression takes. $\square$

## The rules, with justification

| operation | interval result | why it is correct |
|---|---|---|
| $a+b$ | $[\underline a+\underline b,\ \overline a+\overline b]$ | addition is monotone in each argument |
| $a-b$ | $[\underline a-\overline b,\ \overline a-\underline b]$ | $-b$ reverses the order |
| $c\cdot a$ (constant $c$) | $[c\underline a,c\overline a]$ if $c\ge0$, else swapped | linear |
| $a\cdot b$ | $[\min P,\max P]$, $P=\{\underline a\underline b,\underline a\overline b,\overline a\underline b,\overline a\overline b\}$ | bilinear: extremes occur at corners |
| $a^2$ | $[0,\max(\underline a^2,\overline a^2)]$ if $0\in\mathbf a$, else $[\min,\max]$ of the endpoint squares | tighter than $a\cdot a$ |
| $\tanh a$ | $[\tanh\underline a,\tanh\overline a]$ | monotone increasing |
| $Wh+b$ | $Wc+b\pm|W|r$ | Lemma 2.1, applied row by row |

### The affine layer in detail

Row $j$ of $Wh+b$ is $w_j^\top h+b_j$, a linear function of $h$. By Lemma 2.1,
its range over the box $\{h: |h-c|\le r\}$ is exactly
$[w_j^\top c+b_j-|w_j|^\top r,\ w_j^\top c+b_j+|w_j|^\top r]$. So a *single*
affine layer is enclosed **tightly**. Looseness only appears when several
layers are composed.

### Sine and cosine

$\sin$ is not monotone, so its extremes on $[l,u]$ occur at the endpoints *or*
at interior critical points $\frac\pi2+k\pi$. The code (`_interval_sin`) does
the following:

1. Start with $[\min(\sin l,\sin u),\ \max(\sin l,\sin u)]$.
2. Test whether $[l,u]$ contains a maximum point $\frac\pi2+2k\pi$. With $k=\lfloor(u-\frac\pi2)/2\pi\rfloor$, the largest maximum point $\le u$ is $\frac\pi2+2k\pi$. If it is $\ge l$, set the upper bound to 1.
3. Do the same for minimum points $-\frac\pi2+2k\pi$, setting the lower bound to $-1$.
4. If $u-l\ge2\pi$, the result is $[-1,1]$.

$\cos x=\sin(x+\frac\pi2)$ reuses the same routine.

### The derivative of tanh

$\tanh'(z)=1-\tanh^2(z)$ is even and decreasing in $|z|$. On $[\underline z,\overline z]$:

* the maximum is at the point closest to 0, namely $\mathrm{clip}(0,\underline z,\overline z)$;
* the minimum is at an endpoint: $\min(\tanh'(\underline z),\tanh'(\overline z))$.

Computing $1-h\cdot h$ with interval multiplication instead would treat the two
copies of $h$ as independent and lose tightness. This is the dependency
problem, next.

## The dependency problem

Interval arithmetic treats every occurrence of a variable independently. For
$x\in[0,1]$:

* $x-x$ evaluates to $[0-1,\,1-0]=[-1,1]$, although the true value is always 0.
* $x(1-x)$ evaluates to $[0,1]\cdot[0,1]=[0,1]$, although the true range is $[0,0.25]$.

Neural networks suffer from this badly. Neurons share inputs with positive and
negative weights, and the output combines them, so correlations cancel in
reality but not in the intervals.

### Worked example: IBP on a tiny network

Consider $y=1.5\tanh(2x-1)+1.0\tanh(-x+0.5)-0.2$ for $x\in[0,1]$.

1. Pre-activations: $z_1=2x-1\in[-1,1]$ and $z_2=-x+0.5\in[-0.5,0.5]$.
2. Activations: $h_1\in[-0.7616,0.7616]$ and $h_2\in[-0.4621,0.4621]$.
3. Output: $y\in 1.5[-0.7616,0.7616]+[-0.4621,0.4621]-0.2=[-1.8045,\,1.4045]$.

The **true range**, from dense sampling, is $[-0.8803,\,0.4803]$. IBP is more
than twice too wide. The reason: $h_1$ increases with $x$ while $h_2$ decreases,
so their extremes never occur together, but interval arithmetic assumes they
might.

**Splitting helps.** On $[0,0.5]$ IBP gives $[-1.342,0.262]$, and on $[0.5,1]$
it gives $[-0.662,0.942]$. The union is still loose, but each half is tighter.
Width shrinks roughly linearly with box size. This is why branch and bound
works even with loose bounds.

![(a) Ways to enclose $\tanh$ on $[l,u]$: the interval box and the mean-value lines. (b) The dependency problem: IBP widths versus true widths. (c) Splitting into more boxes tightens lower bounds.](../figures/relaxations.png)


## Floating-point soundness in the implementation

Bounds are computed in float64, and every operation widens its result outward.

```python
_U = 2.0 ** -52          # unit roundoff (conservative)
_ELEM_REL = 8 * _U       # elementwise ops: widen by 8 ulp relative
_LIB_ABS = 4 * _U        # tanh/sin/cos from the math library: < 2 ulp error

def _widen(lo, hi, rel=_ELEM_REL, abs_=_ELEM_ABS):
    return lo - (rel * lo.abs() + abs_), hi + (rel * hi.abs() + abs_)
```

The affine layer uses the dot-product error bound from Chapter 2:

```python
def linear(self, weight, bias=None):
    c, r = self.center(), self.radius()
    yc = c @ W.t()                         # centre
    yr = r @ W.abs().t()                   # radius
    mag = c.abs() @ W.abs().t() + yr       # sum |w||h|, magnitude of the terms
    if bias is not None:
        yc = yc + b; mag = mag + b.abs()
    n = W.shape[1] + 2
    err = 2 * n * _U * mag + _ELEM_ABS     # >= gamma_n * sum|w h|, plus centre/radius rounding
    return Interval(yc - yr - err, yc + yr + err)
```

For $\sin$ and $\cos$, the argument itself may carry rounding error. Since
$|\sin'|\le1$, an argument error $e$ changes the value by at most $e$, and the
code adds that amount.

**Why float64 and not float32?** float32 has $u\approx6\times10^{-8}$. The
widening would then be comparable to the margins we need to prove (around
$10^{-2}$ to $10^{-3}$ after many operations), and CROWN's lack of directed
rounding (Chapter 8) would be far riskier. The weights are trained in float32
but are **exactly representable** in float64, so the verified function is
exactly the trained one.

## The `Interval` class

`src/verification/interval.py` implements the operator overloads (`__add__`,
`__sub__`, `__mul__`, `__neg__`, `__rsub__`, `__getitem__`) and the methods
`linear`, `scale_shift`, `tanh`, `dtanh`, `sin`, `cos`, `square` and `cat`.
Because `src/ops.py` dispatches on type, calling

```python
out = certificate_module(Interval(lo, hi))   # lo, hi: (N, n) batches of boxes
```

runs the *same* Python code as training, but on intervals. It returns bounds
on $[B,\mathrm{cond}]$ for $N$ boxes at once. Batching makes IBP extremely fast:
hundreds of thousands of boxes per second.

## Testing soundness

`tests/test_interval_bounds.py` samples random boxes, including wide ones and
ones crossing the $\sin$ extrema. It evaluates the exact function at 2000
points per box, corners included, and asserts that every value lies inside the
computed interval. This is the practical meaning of "**interval lower bounds
never overestimate actual function minimums**" from the project specification.

## Why IBP, and why it is not enough

**Strengths:** trivial to implement correctly, very fast, easy to make
rounding-safe. **Weakness:** the dependency problem. On the final pendulum
networks, IBP alone *can* certify, but it needs **216,787** boxes, against
**140** with CROWN plus Z3. That is why the next layer exists.

---

# Linear relaxations, CROWN and auto_LiRPA

## Intuition

Keep, for every intermediate quantity, not a constant interval but **two linear
functions of the input** that sandwich it over the box. Linear functions
remember how each quantity depends on the input, so correlated quantities can
cancel correctly.

## Linear relaxation of a single nonlinearity

For $y=\sigma(z)$ with $z\in[l,u]$, choose slopes and intercepts with
$$a_Lz+b_L\ \le\ \sigma(z)\ \le\ a_Uz+b_U\qquad\forall z\in[l,u].$$

For $\tanh$:

* **Both endpoints $\ge0$ (concave region).** The chord from $(l,\tanh l)$ to $(u,\tanh u)$ lies *below* the graph, so it is a valid lower line. Any tangent line lies *above* the graph, so it is a valid upper line.
* **Both endpoints $\le0$ (convex region).** The situation is mirrored: the chord is above and tangents are below.
* **Mixed signs.** Use a tangent at a suitable point, chosen so that the line stays valid across the inflection point at 0.

auto_LiRPA implements these cases for tanh, sin, cos and many other functions.

## Products: the McCormick envelope

The Lie derivative multiplies quantities that both depend on $x$: for example
$\tanh'(z)\cdot dz$, $v\cos\theta$, or $\sin\theta\cdot u_1$. For $w=ab$ with
$a\in[a_l,a_u]$ and $b\in[b_l,b_u]$, the four products of non-negative factors
give linear inequalities:
$$
\begin{aligned}
(a-a_l)(b-b_l)\ge0 &\Rightarrow w\ge a_lb+ab_l-a_lb_l\\
(a_u-a)(b_u-b)\ge0 &\Rightarrow w\ge a_ub+ab_u-a_ub_u\\
(a_u-a)(b-b_l)\ge0 &\Rightarrow w\le a_ub+ab_l-a_ub_l\\
(a-a_l)(b_u-b)\ge0 &\Rightarrow w\le a_lb+ab_u-a_lb_u
\end{aligned}
$$
This is the **McCormick envelope** (McCormick, 1976): the convex hull of the
graph of $ab$ over the rectangle. It is the tightest possible *linear*
relaxation of a product.

**Numeric check.** For $a,b\in[0,1]$ at $a=b=0.5$ (true $w=0.25$), the lower
bounds give $w\ge0$ and $w\ge0.5+0.5-1=0$, and the upper bounds give
$w\le0.5$ and $w\le0.5$. So the relaxation is $w\in[0,0.5]$, which contains
$0.25$ ✓. It is exact at the corners of the rectangle.

## CROWN: backward propagation of linear bounds

**Goal.** Find $\underline A,\underline b$ such that
$y(x)\ge\underline A x+\underline b$ for all $x$ in the box.

**Idea (Zhang et al., 2018).** Start from the output $y=W_2h+b_2$ and replace
$h$ by its relaxation, choosing the lower or the upper line according to the
**sign** of each coefficient. A positive coefficient needs the lower line of
$h$ to produce a lower bound on $y$; a negative coefficient needs the upper
line. This yields a linear expression in $z$. Substitute $z=W_1x+b_1$, which is
exact. The result is linear in $x$:
$$y\ \ge\ \underbrace{W_2^+\,\mathrm{diag}(a_L)W_1+W_2^-\,\mathrm{diag}(a_U)W_1}_{\underline A}\,x+\underline b,$$
where $W^+=\max(W,0)$ and $W^-=\min(W,0)$ elementwise. Deeper networks repeat
the substitution layer by layer. To choose each neuron's relaxation, CROWN
needs bounds $[l,u]$ on that neuron's input, which it computes recursively with
the same procedure (these are *intermediate bounds*).

Finally, **concretize** with Lemma 2.1:
$\min_{x\in R}\underline Ax+\underline b=\underline Ac-|\underline A|r+\underline b$.

### Worked example (same tiny network)

These numbers were produced by auto_LiRPA on
$y=1.5\tanh(2x-1)+\tanh(-x+0.5)-0.2$:

| box | IBP | CROWN | true range | CROWN lower line | CROWN upper line |
|---|---|---|---|---|---|
| $[0,1]$ | $[-1.80,\ 1.40]$ | $[-1.04,\ 0.64]$ | $[-0.88,\ 0.48]$ | $1.500x-1.044$ | $1.500x-0.856$ |
| $[0,0.5]$ | $[-1.34,\ 0.26]$ | $[-1.02,\ -0.19]$ | $[-0.88,\ -0.20]$ | $1.435x-1.021$ | $1.345x-0.863$ |
| $[0.5,1]$ | $[-0.66,\ 0.94]$ | $[-0.21,\ 0.62]$ | $[-0.20,\ 0.48]$ | $1.345x-0.882$ | $1.435x-0.814$ |

Observations:

* CROWN is roughly **twice as tight** as IBP on the full box.
* The linear functions track the function's *trend* (slope around 1.4–1.5), which constant bounds cannot do.
* On the half-boxes CROWN is nearly exact.

## auto_LiRPA in the project

`auto_LiRPA` (Xu et al., 2020) traces any PyTorch module into a graph of
"bounded operators" and runs CROWN (and other methods) automatically.

```python
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

lirpa = BoundedModule(model_float64, torch.zeros(1, n))          # trace once
ptb = PerturbationLpNorm(norm=float("inf"), x_L=l, x_U=h)       # batch of boxes
x = BoundedTensor(0.5 * (l + h), ptb)
lb, ub, A = lirpa.compute_bounds(
    x=(x,), method="CROWN", return_A=True,
    needed_A_dict={lirpa.output_name[0]: [lirpa.input_name[0]]})
a = A[out_name][in_name]   # a["lA"], a["lbias"], a["uA"], a["ubias"]
```

* `lb`, `ub` have shape `(N, 2)`: bounds on $[B,\mathrm{cond}]$ for each of the $N$ boxes.
* `lA` and `uA` have shape `(N, 2, n)`: the linear bounding functions. They are needed by the Z3 step (Chapter 9) and returned at negligible extra cost.

### Engineering details and pitfalls

* **The float64 dtype bug.** auto_LiRPA's slicing operator allocates buffers with PyTorch's *default* dtype (float32). With a float64 model this caused `index_copy_(): self and source expected to have the same dtype`. The fix (`BoundEngine.crown`) sets `torch.set_default_dtype(torch.float64)` for the duration of the call and restores it afterwards.
* **Chunking.** Boxes are processed in chunks of 2048 (`crown_batch`) to bound memory use.
* **Intersection with IBP.** Both are sound, so `max(lb_IBP, lb_CROWN)` and `min(ub_IBP, ub_CROWN)` are sound too, and at least as tight as either.
* **Rounding.** auto_LiRPA does not use directed rounding. The project requires every inequality to hold with margin $\tau=10^{-8}$, orders of magnitude above typical float64 accumulated error (around $10^{-13}$). This is standard practice in neural network verification tools, but it is not a formal guarantee about floating point. It is the main soundness caveat of the project, and it is stated in the README and the paper.
* **Graceful degradation.** If CROWN throws an exception, the verifier logs a warning and continues with IBP only (`crown_failed`).

## Why CROWN (and not α-CROWN or an LP)?

* **LP relaxation** (solving an LP per bound) is tighter but orders of magnitude slower per box.
* **α-CROWN** optimises the relaxation slopes by gradient descent. It is tighter but slower. It crashed in a quick experiment on this graph, so it is listed as future work.
* **CROWN** is the sweet spot: closed-form, batched and roughly twice as tight here. The ablation (Chapter 20) shows it reduces the pendulum proof from 216,787 boxes to 165.
