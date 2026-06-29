# Introduction

**qsv** is a high-performance quantum **statevector simulator written in Rust**, built as a
study in performance engineering. Off-the-shelf simulators already exist (qsim, Qiskit-Aer,
QuEST, Yao.jl); qsv's purpose is not to be another one, but to *demonstrate how one is
optimized* — cache-, SIMD-, and threading-aware design driven by profiling and benchmarked
honestly against the established tools.

## The thesis that organizes everything

> **Statevector simulation is memory-bandwidth-bound, not compute-bound.**

Applying a 1-qubit gate streams the entire $2^N$-amplitude array while doing only ~2
complex multiplies per 16-byte amplitude — an arithmetic intensity of roughly **0.13
FLOP/byte**, deep in the bandwidth-bound region of the roofline. Every optimization decision
in qsv is justified by one question:

> *Does this reduce bytes moved per gate, or raise arithmetic intensity per byte moved?*

This reframes "squeeze all the performance" as primarily a **memory-traffic and cache**
problem. See [How we optimize](design/optimization.md) for the full argument and the
evidence behind it.

## What's here

- A [tutorial](tutorial.md) for using qsv as a library.
- An [architecture overview](design/overview.md) of the crate and its pluggable backends.
- [How we optimize](design/optimization.md) — the optimization strategy and milestone
  narrative (the centerpiece).
- [The core kernel](design/kernel.md) — the `insert_zero_bit` indexing trick every
  production simulator shares, explained from the ground up.
- [Benchmarking & profiling](design/benchmarking.md) — how we measure.
- [Research notes](research/overview.md) — distilled findings from qsim, Qiskit-Aer, QuEST,
  Yao.jl, cuStateVec, spinoza, and *Algorithms for Modern Hardware*.

## Status

Foundations and the first optimization milestones are in place: a Structure-of-Arrays
statevector, the universal bit-shift kernel, a pluggable `Backend` trait with three
implementations (a naive oracle plus two optimized backends), and a differential test suite
that validates every kernel against the oracle. See the
[roadmap](design/roadmap.md) for the milestone-by-milestone plan.
