# Roadmap & milestones

This page tracks status and is updated as each milestone lands. The rationale for the order
is in [How we optimize](optimization.md).

## Status

| Ver | Milestone | Status |
| --- | --- | --- |
| v0.0 | scaffold + naive `RefBackend` oracle + correctness tests | ✅ done |
| v0.1 | `ReshapeBackend` — block / out-of-place apply | ✅ done |
| v0.2 | `BitShiftBackend` — in-place bit-shift pair kernel | ✅ done |
| v0.3 | `CpuBackend` — bounds-check-free access + stack gate matrices | ✅ done |
| v0.4 | `CpuBackend` — cache-friendly nested-block 1q kernel | ✅ done |
| v0.5 | `CpuBackend::parallel()` — rayon threading | ✅ done |
| v0.6 | `CpuBackend` — diagonal-gate fast path | ✅ done |
| v0.7 | `SimdBackend` — portable `wide::f64x4` 1q kernel | ✅ done (null result¹) |
| v0.8 | **gate fusion** (`fusion::fuse`) | ✅ done |
| v0.9 | cache-block the multi-qubit kernel + prefetch + NT stores | ⬜ next² |
| v0.10 | ILP, x86 BMI2, alias-table sampling, parallel prefix-sum | ⬜ |
| v1.0 | roofline-validated, documented, cross-sim benchmarked | ⬜ |

¹ SIMD measured ~0% on the 1q kernel — it's bandwidth-bound, not arithmetic-bound. See
[benchmarking](benchmarking.md). Expected to matter for fused multi-qubit matvecs and x86 AVX-512.
² Promoted to "next" by the v0.8 finding: fusion's win is currently capped at large N by the
multi-qubit kernel's scattered access — cache-blocking it is the unlock.

Also planned but out of the v1 critical path: density-matrix / noise simulation, a GPU
backend (CUDA/cuTile or Metal) behind the `Backend` seam, and distributed multi-node support
(mirroring QuEST's pairwise rank exchange).

## Testing

Robustness rests on **differential testing**: every optimized backend must reproduce the
naive `RefBackend` oracle amplitude-for-amplitude. Because the oracle is implemented a
structurally different way (gather/scatter, not the bit-shift pairing), this genuinely
cross-checks the kernels rather than re-running the same logic.

Current suite (grows with each milestone):

- **200** reproducible random circuits (3–8 qubits, depth 20–60) — every optimized backend
  (`Reshape`, `BitShift`, `CpuBackend` serial **and** parallel) vs oracle;
- **15** random circuits at 10 qubits, depth 80 — `BitShiftBackend` vs oracle;
- **6** random circuits at **14 qubits** (above the threading threshold) — `CpuBackend`'s
  rayon path, including the multi-qubit parallel kernel, vs oracle;
- **QFT** → uniform superposition (n = 1,2,3,5,8) and QFT vs oracle;
- a 3-qubit **Toffoli** across several qubit orderings, exercising the general `apply_mq` path;
- unit tests for the index arithmetic (pairs differ in exactly the target bit; block anchors
  tile the space) and for the RNG/generators.

Every kernel added in a later milestone is wired into this same differential harness, so the
optimization can never silently break correctness.

## Definition of done for a milestone

1. New backend (or kernel) implemented behind the `Backend` trait.
2. Differential tests against the oracle pass, including any new path it introduces.
3. `cargo clippy --all-targets -- -D warnings` and `cargo fmt --check` clean.
4. A benchmark data point recorded (see [Benchmarking & profiling](benchmarking.md)).
5. This roadmap and the relevant design pages updated.
