# Research synthesis

Distilled findings from studying production statevector simulators, used to justify every
design decision in [`../DESIGN.md`](../DESIGN.md). Full reference source is shallow-cloned
under `references/` (git-ignored).

## The one finding that organizes everything

> **Statevector simulation is memory-bandwidth-bound, not compute-bound.**

A 1-qubit gate streams the entire `2^N`-amplitude array while doing only ~2 complex
multiplies per 16-byte amplitude → arithmetic intensity ≈ **0.13 FLOP/byte**, deep in the
bandwidth-bound region of the roofline. Confirmed independently across qsim, Qiskit-Aer,
QuEST, Yao.jl, cuStateVec, and spinoza.

**Consequences (optimization priority, high → low end-to-end impact):**

1. **Gate fusion** — fewer full passes over the state = fewer bytes moved. Biggest win.
2. **SoA layout** (`re[]`, `im[]` split) — shuffle-free SIMD complex multiply.
3. **Cache-aware access / blocking** — keep the working set hot; minimize passes.
4. **Multithreading** — embarrassingly parallel disjoint amplitude pairs.
5. **Micro-ops** (BMI2, unchecked indexing, ILP, prefetch, NT-stores) — smaller gains.

The clincher: QuEST's BMI2 work ([issue #717](https://github.com/QuEST-Kit/QuEST/issues/717),
which `nez0b` contributed to) gives only **1.0–1.3× end-to-end** despite 6–12× faster
bit-ops in isolation — because it speeds up *address computation*, not *memory traffic*.

## Documents

- [`01-cpu-simulators.md`](01-cpu-simulators.md) — kernel, layout, SIMD, fusion, threading,
  and the universal `insert_zero_bit` indexing trick across qsim / Aer / QuEST / Yao.jl /
  the `state_vector.jl` reference notebook.
- [`02-gpu-and-rust-landscape.md`](02-gpu-and-rust-landscape.md) — cuStateVec/cuda-quantum
  GPU kernel structure, the (unverified) cuTile/cuTile-rs claims, the Rust ecosystem gap,
  and the relevant `amh-code` (Algorithmica HPC) techniques.

## Reference repos cloned locally (`references/`)

| Repo | Why |
|---|---|
| `qsim` | SoA layout, gate fusion, high/low-qubit SIMD dispatch, BMI2 |
| `qiskit-aer` | `index0`/`indexes` generation, AVX2 matvec, fusion pass |
| `QuEST` | `insertZeroBit` kernels, multi-controlled masks, distributed pairwise exchange |
| `Yao.jl`, `YaoArrayRegister.jl`, `BitBasis.jl` | `IterControl`/`bmask` subspace enumeration |
| `spinoza` | the Rust peer (CPU SIMD + rayon) — prior art to beat |
| `amh-code` | cache-blocking, SIMD, prefix-sum scan, ILP, NT-stores, prefetch |

`cuda-quantum` was intentionally not cloned (multi-GB); its cuStateVec approach is captured
in `02-gpu-and-rust-landscape.md` and GPU work is deferred behind the `Backend` seam.
