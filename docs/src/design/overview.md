# Architecture overview

qsv is a Cargo **workspace**, which keeps the optimization-critical library isolated (tiny,
auditable, dependency-free) from heavier bench/binding crates.

```text
crates/qsv-core   the product: state, gates, circuit, backends, fusion
crates/qsv-cli    `qsv` binary (smoke runner; QASM3 + sampling later)
crates/qsv-bench  criterion benchmarks + profiling binaries
docs/             this mdBook (design, research, tutorial)
references/       shallow clones of qsim/QuEST/aer/Yao/spinoza/amh-code (git-ignored)
```

## `qsv-core` modules

| Module | Responsibility |
| --- | --- |
| `real` | the `Real` trait — one generic kernel codebase over `f64`/`f32` |
| `complex` | `Cplx<R>` for the API boundary and gate matrices |
| `state` | `StateVector<R>` (SoA storage) |
| `state::layout` | `insert_zero_bit` & friends — the index arithmetic |
| `gate` | `DenseGate<R>` + the standard gate library |
| `circuit` | `Circuit<R>` and the fluent builder |
| `circuits` | RNG + `random_circuit`/`ghz`/`qft` generators |
| `backend` | the `Backend` trait and its implementations |
| `fusion` | gate-fusion pass (a later milestone) |

`#![deny(unsafe_code)]` is set crate-wide; only the hot-path modules opt back in with a
localized `#[allow(unsafe_code)]` and a `// SAFETY:` justification, keeping the unsafe
surface tiny.

## Load-bearing abstractions

### `StateVector<R>` — Structure of Arrays

Amplitudes are stored as two separate arrays, `re: Vec<R>` and `im: Vec<R>`, not as an
interleaved `Vec<Complex>`. For the bandwidth-bound complex-multiply kernel this is the right
layout: a SIMD load of `re` and a SIMD load of `im` each yield a register of like-typed
values, so the multiply is a straight broadcast-FMA chain with **no lane shuffles** (no NEON
`ld2` / x86 `unpck` de-interleave). All updates are **in place** — at 30 qubits the vector is
16 GB and there is no room for a second buffer.

### `Backend<R>` — the pluggable seam

Everything above this trait (circuit, gates, fusion) is backend-agnostic; a backend owns the
amplitude storage and the gate kernels. This is the seam behind which a future CUDA/cuTile or
Metal backend will live **without touching the circuit layer**. It is kept leak-proof by:

- an associated **`type State`** — CPU uses the host `StateVector`, a GPU backend would use an
  opaque device handle; no method ever hands out a `&mut [R]` to host memory;
- **reductions** (`probabilities`, and later `sample`/`expectation`) are *backend methods*, so
  a GPU computes them on-device rather than copying back;
- a single `download` as the only device→host crossing;
- a default `execute(&Circuit)` that a GPU overrides to batch a whole circuit.

The trait is validated by having **three** implementors today — `RefBackend`,
`ReshapeBackend`, `BitShiftBackend` — which proves it encodes no CPU-only assumptions.

### `Gate` — zero hot-path allocation

Gate matrices are small and fixed-size; parametric gates (`rx(θ)`, …) materialize on the
stack, and only fused/arbitrary unitaries touch the heap (once). v0.0 uses a uniform
`DenseGate`; specialized representations (const-generic `Mat2`/`Mat4`, diagonal-only) arrive
with the optimized kernels.

## Why separate backends per milestone

Each optimization is its own `Backend` struct rather than a mutation of one. This keeps the
slow, obviously-correct oracle alive next to every fast kernel, so the
[differential test suite](roadmap.md#testing) can run an identical circuit through both and
diff the result. It also lets the benchmark harness run one circuit through every milestone
and plot them head to head — the optimization narrative falls out of the architecture.
