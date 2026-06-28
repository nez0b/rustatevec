# qsv

[![CI](https://github.com/nez0b/rustatevec/actions/workflows/ci.yml/badge.svg)](https://github.com/nez0b/rustatevec/actions/workflows/ci.yml)
[![Docs](https://github.com/nez0b/rustatevec/actions/workflows/pages.yml/badge.svg)](https://nez0b.github.io/rustatevec/)

A high-performance **quantum statevector simulator in Rust**, built as a study in
performance engineering: cache-, SIMD-, and threading-aware design driven by profiling and
benchmarked honestly against the established simulators (qsim, Qiskit-Aer, QuEST, spinoza).

> **Thesis:** statevector simulation is *memory-bandwidth-bound*, not compute-bound — so the
> work that matters is raising arithmetic intensity (gate fusion) and improving cache
> behaviour, not micro-optimizing arithmetic. The repo is structured to *demonstrate* this,
> milestone by milestone, with a roofline plot to back it up.

📖 **Documentation book:** published at **<https://nez0b.github.io/rustatevec/>** (or
`mdbook serve docs` locally) — see especially [How we optimize](docs/src/design/optimization.md),
[The core kernel](docs/src/design/kernel.md), [Benchmarking](docs/src/design/benchmarking.md),
and the [roadmap](docs/src/design/roadmap.md).

## Status

**v0.0 – v0.8 done.** A pluggable `Backend` trait with several implementations validated
against a naive oracle by a differential test suite:

- `RefBackend` (oracle) · `ReshapeBackend` (v0.1) · `BitShiftBackend` (v0.2)
- `CpuBackend` — bounds-check-free nested-block kernel (v0.3/4), rayon threading (v0.5),
  diagonal fast path (v0.6); **the default**
- `SimdBackend` — `wide::f64x4` 1q kernel (v0.7)
- `fusion::fuse` — gate fusion (v0.8)

Headline numbers on an M3 Pro: the threaded kernel hits **~85% of memory bandwidth** at 24
qubits, and fusion gives **~1.8×** on QFT(14). SIMD was a measured **null result** on the 1q
kernel (it's bandwidth-bound) — reported honestly in the
[benchmarking chapter](docs/src/design/benchmarking.md).

Next: v0.9 (cache-block the multi-qubit kernel — the unlock for fusion at large N); then a GPU
backend (CUDA/cuTile) behind the same seam. See [`todo.md`](todo.md) and the
[roadmap](docs/src/design/roadmap.md).

## Quick start

```bash
cargo test --workspace                       # oracle + differential + unit tests
cargo run --release --bin qsv -- 24          # GHZ(24) outcome probabilities
cargo bench -p qsv-bench --bench throughput  # gate-throughput benchmarks
cargo clippy --workspace --all-targets -- -D warnings
```

```rust
use qsv_core::prelude::*;

// Bell state: H(0); CX(0,1)  ->  (|00> + |11>)/√2
let mut c = Circuit::<f64>::new(2);
c.h(0).cx(0, 1);
let s = BitShiftBackend.execute(&c);
assert!((s.amplitude(0b11).re - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
```

## Layout

```
crates/qsv-core   core library (state, gates, circuit, backends, fusion)
crates/qsv-cli    `qsv` binary (smoke runner; QASM3 + sampling later)
crates/qsv-bench  criterion benchmarks + `qsv-profile` profiling workload
docs/             mdBook documentation (design, optimization, tutorial, research)
docs/reference/   state_vector.jl — the pedagogical reference notebook
references/        shallow clones of qsim/QuEST/aer/Yao/spinoza/amh-code (git-ignored)
```

## Why Rust, and what's novel

The Rust quantum-sim ecosystem has CPU SIMD simulators (`spinoza`) and circuit DSLs
(`qoqo`), but no native-Rust engine combining gate fusion + cache-blocking + SoA + SIMD
behind a pluggable, modern GPU seam. qsv targets that gap — with the *optimization writeup*
as the deliverable. Primary dev/benchmark hardware: Apple M3 Pro (NEON); secondary: x86
(AVX2/AVX-512).

## License

MIT OR Apache-2.0.
