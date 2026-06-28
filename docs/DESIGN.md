# qsv — design document

Design-of-record for the qsv statevector simulator. The forces and evidence behind these
decisions are in [`research/`](research/).

## Goal & non-goals

**Goal:** a high-performance, cache/SIMD/threading-aware statevector simulator in Rust whose
*headline deliverable is the optimization methodology* — profiling-driven, benchmarked
honestly against qsim / Qiskit-Aer / QuEST / spinoza.

**v1 non-goals:** density matrices / noise; a shipped GPU backend; distributed/multi-node.
Each is deliberately deferred behind clean seams (GPU behind the `Backend` trait; distributed
mirrors QuEST's pairwise exchange).

## Organizing principle

**Statevector simulation is memory-bandwidth-bound.** Every optimization is justified by
"does it reduce bytes moved per gate, or raise arithmetic intensity per byte?" See
[`research/README.md`](research/README.md). Optimization priority: fusion → SoA+SIMD →
cache-blocking → threading → micro-ops.

## Architecture

Cargo **workspace**: `qsv-core` (the optimization-critical library, tiny audited surface),
`qsv-cli` (runner), later `qsv-bench` (criterion + cross-sim harness) and optional `qsv-py`
(PyO3). Module tree and load-bearing abstractions:

- **`StateVector<R>` = SoA** `{ re: Vec<R>, im: Vec<R> }`, in-place mandatory (30q f64 =
  16 GB on 36 GB → no second buffer). SoA ⇒ shuffle-free SIMD complex multiply.
- **`Real`** trait: one generic kernel codebase over `f64`/`f32` (monomorphized, never `dyn`).
- **`Backend<R>` trait = the pluggable GPU seam.** Associated `type State` (no `&mut [R]`
  leaks host layout); reductions are backend methods (on-device for a GPU); single
  `download` boundary; `execute()` default that a GPU overrides to batch a whole circuit.
  Proven non-CPU-specific by the second backend, `RefBackend`.
- **Gate** = const-generic fixed-size matrices, parametric gates built on the stack, only
  fused/arbitrary unitaries heap-allocate. (v0.0 uses a uniform `DenseGate`; specialized
  reps arrive with the optimized kernels.)
- **`state::layout`** = `insert_zero_bit` & friends — the universal indexing trick, the main
  future `unsafe` surface, exhaustively unit-tested.
- **fusion** = pure `Circuit -> Circuit` pass with a memory-pass cost model.

`#![deny(unsafe_code)]` crate-wide; hot modules opt back in with localized
`#[allow(unsafe_code)]` + `// SAFETY:` notes.

## Milestone roadmap (the portfolio narrative)

Each version = one benchmarkable diff = one data point on the headline throughput/roofline
plot. Speedups are honest regimes.

| Ver | Change | Regime |
|---|---|---|
| **v0.0** ✅ | scaffold + naive `RefBackend` oracle + correctness tests | caps ~13q; the oracle |
| v0.1 | reshape/tensor-contraction apply | ~24q, slow |
| v0.2 | bit-shift kernel (in-place, scalar) | order of magnitude |
| v0.3 | unchecked indexing + stack matrices + zero hot-path alloc | 1.3–2× |
| v0.4 | high/low target dispatch | 1.3–2× |
| v0.5 | multithreading (rayon) | ~4–6× |
| v0.6 | diagonal-gate fast path | ~2× on phase-heavy |
| v0.7 | SIMD (SoA + `wide`; nightly `std::simd` + hand intrinsics compare) | 1.3–1.8× NEON / 2–4× AVX |
| v0.8 | **gate fusion** | 2–5× end-to-end (headline) |
| v0.9 | cache-blocking + prefetch + NT-stores | 1.2–1.5× |
| v0.10 | ILP, x86 BMI2, alias-table sampling, parallel scan | 1.05–1.2× |
| v1.0 | roofline-validated, documented, cross-sim benchmarked | — |

**Stop criterion:** when dominant kernels sustain ≥70–80% of empirical STREAM bandwidth,
micro-ops are noise → pivot to fusion/algorithmic wins.

## Profiling & benchmarking

- Tools: `samply` / `cargo-instruments` (mac), `perf` / `likwid` / VTune (x86), `criterion`
  (statistical), `iai-callgrind` (deterministic CI regression gate).
- Roofline from an empirical STREAM-triad BW measurement; show **fusion moving kernels
  rightward** (higher arithmetic intensity).
- Circuits: GHZ, QFT, Grover, random/Quantum-Volume/RCS (headline scaling), QAOA layer.
- Fairness: one circuit def → every tool; time only state evolution; match precision &
  threads; same box for head-to-heads; report wall-clock and gate-throughput; **show losses**.

## Testing

`RefBackend` oracle agreement (n ≤ 10), proptest invariants (norm/unitarity, gate
identities, controlled subspace, **fusion equivalence**, SIMD≡scalar), Qiskit/QuEST golden
cross-validation, CI on arm64 + x86, `miri` on the index math, sanitizers on unsafe kernels.

## Hardware (primary dev box)

Apple M3 Pro, arm64, 11 cores (5P+6E), 36 GB, ~150 GB/s unified memory, 128-bit NEON, no
local CUDA. Secondary benchmark target: x86 (AVX2/512) cloud box.
