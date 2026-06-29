# Research notes — overview

Distilled findings from studying production statevector simulators, which justify the design
decisions throughout this book. The full reference source is shallow-cloned under
`_local/` (git-ignored).

## The one finding that organizes everything

> **Statevector simulation is memory-bandwidth-bound, not compute-bound.**

A 1-qubit gate streams the entire $2^N$-amplitude array doing only ~2 complex multiplies
per 16-byte amplitude → arithmetic intensity ≈ **0.13 FLOP/byte**, deep in the bandwidth-bound
region of the roofline. Confirmed independently across qsim, Qiskit-Aer, QuEST, Yao.jl,
cuStateVec, and spinoza. See [How we optimize](../design/optimization.md) for the consequences.

## Reference repositories studied

| Repo | What we took from it |
| --- | --- |
| `qsim` | SoA layout, gate fusion, high/low-qubit SIMD dispatch, BMI2 |
| `qiskit-aer` | `index0`/`indexes` generation, AVX2 matvec, fusion pass |
| `QuEST` | `insertZeroBit` kernels, multi-controlled masks, distributed pairwise exchange |
| `Yao.jl` / `YaoArrayRegister.jl` / `BitBasis.jl` | `IterControl`/`bmask` subspace enumeration |
| `spinoza` | the Rust peer (CPU SIMD + rayon) — prior art to compare against |
| `amh-code` | cache-blocking, SIMD, prefix-sum scan, ILP, non-temporal stores, prefetch |

`cuda-quantum` was intentionally not cloned (multi-GB); its cuStateVec approach is captured in
[GPU, Rust & HPC landscape](gpu-and-rust.md), and GPU work is deferred behind the `Backend`
seam.

The two pages that follow go into detail on the [CPU simulators](cpu-simulators.md) and the
[GPU / Rust / HPC landscape](gpu-and-rust.md).
