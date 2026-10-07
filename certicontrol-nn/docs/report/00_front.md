---
title: "CertiControl-NN: Complete Technical Report"
subtitle: "Theory, algorithms, implementation and experiments, explained from first principles"
author: "CertiControl-NN project"
date: "September 2026"
toc: true
toc-depth: 3
numbersections: true
geometry: margin=2.2cm
fontsize: 10.5pt
colorlinks: true
documentclass: report
---

# About this report {.unnumbered}

This report explains **everything** in the CertiControl-NN project, assuming
nothing beyond first-year calculus, basic linear algebra and Python. It is
written so that a reader who has never heard of barrier functions, bound
propagation or SMT solvers can finish it able to:

* explain *why* the project exists and what exactly it proves;
* derive every equation used, including the proofs;
* read and modify every source file;
* run, reproduce and extend the experiments;
* judge the results critically, including their limitations.

**How the report is organised.**

| Part | Chapters | Content |
|---|---|---|
| I. Background | 1–4 | motivation; mathematical prerequisites; dynamical systems; neural networks and differentiation |
| II. Safety theory | 5 | invariance, barrier certificates, the four properties, the main theorem with its full proof |
| III. Verification | 6–11 | soundness; interval arithmetic; linear relaxations and CROWN; SAT/SMT and Z3; branch and bound; proof checking |
| IV. Synthesis | 12–13 | training losses, and the counterexample-guided loop |
| V. Implementation | 14–18 | every file and function; configuration; running; tests; dashboard |
| VI. Experiments | 19–21 | setup, results, the full Dubins story, development lessons |
| VII. Perspective | 22 | related work, limitations, future work |
| Appendices | A–E | notation, glossary, exercises with solutions, FAQ, references |

Each technical idea is presented in the same order: **intuition → theory
(with proofs) → worked numerical example → code → design rationale** ("why
this, and not something else").

**Notation used throughout.** Vectors are lower case ($x$), matrices upper case
($W$). $x_i$ is the $i$-th component. $\le$ between vectors is componentwise.
$[\underline a,\overline a]$ denotes an interval; an underline means a lower
bound and an overline an upper bound. $\odot$ is elementwise multiplication.
$|W|$ is the elementwise absolute value. Code identifiers are in
`monospace`. Appendix A has a full notation table.
