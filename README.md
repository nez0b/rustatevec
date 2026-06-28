# qsv

A high-performance **quantum statevector simulator in Rust**, built as a study in
performance engineering: cache-, SIMD-, and threading-aware design driven by profiling and
benchmarked honestly against the established simulators (qsim, Qiskit-Aer, QuEST, spinoza).

> **Thesis:** statevector simulation is *memory-bandwidth-bound*, not compute-bound — so the
> work that matters is raising arithmetic intensity (gate fusion) and improving cache
> behaviour, not micro-optimizing arithmetic. The repo is structured to *demonstrate* this,
> milestone by milestone, with a roofline plot to back it up.

📖 **Documentation book:** `mdbook serve docs` (or read the chapters under
[`docs/src/`](docs/src/)) — see especially [How we optimize](docs/src/design/optimization.md),
[The core kernel](docs/src/design/kernel.md), and the [roadmap](docs/src/design/roadmap.md).

## Status

**v0.0 – v0.2 done.** Core types, a Structure-of-Arrays `StateVector`, the universal
`insert_zero_bit` index math, a standard gate set + circuit builder, and a pluggable
`Backend` trait with **three** implementations: a naive `RefBackend` oracle, the
out-of-place `ReshapeBackend` (v0.1), and the in-place `BitShiftBackend` pair kernel (v0.2,
scales to ~30 qubits). A differential test suite validates every backend against the oracle,
and `qsv-bench` provides criterion benchmarks + a profiling workload.

Next: v0.3 (unchecked indexing / stack gate matrices) → v0.4 (high/low dispatch) → v0.5
(threading); see the [roadmap](docs/src/design/roadmap.md).

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
