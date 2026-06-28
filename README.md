# qsv

A high-performance **quantum statevector simulator in Rust**, built as a study in
performance engineering: cache-, SIMD-, and threading-aware design driven by profiling and
benchmarked honestly against the established simulators (qsim, Qiskit-Aer, QuEST, spinoza).

> **Thesis:** statevector simulation is *memory-bandwidth-bound*, not compute-bound — so the
> work that matters is raising arithmetic intensity (gate fusion) and improving cache
> behaviour, not micro-optimizing arithmetic. The repo is structured to *demonstrate* this,
> milestone by milestone, with a roofline plot to back it up. See
> [`docs/DESIGN.md`](docs/DESIGN.md) and [`docs/research/`](docs/research/).

## Status

**v0.0 — foundations** (current): core types, a Structure-of-Arrays `StateVector`, the
universal `insert_zero_bit` index math, a standard gate set + circuit builder, the pluggable
`Backend` trait, and a naive **reference oracle** backend with a correctness test suite.

The optimization milestones (bit-shift kernel → SIMD → threading → gate fusion →
cache-blocking) land next; see the roadmap in [`docs/DESIGN.md`](docs/DESIGN.md).

## Quick start

```bash
cargo test --workspace        # correctness oracle + unit/property tests
cargo run --bin qsv -- 5      # print GHZ(5) outcome probabilities
cargo clippy --workspace --all-targets -- -D warnings
```

```rust
use qsv_core::prelude::*;

// Bell state: H(0); CX(0,1)  ->  (|00> + |11>)/√2
let mut c = Circuit::<f64>::new(2);
c.h(0).cx(0, 1);
let s = RefBackend.execute(&c);
assert!((s.amplitude(0b11).re - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
```

## Layout

```
crates/qsv-core   core library (state, gates, circuit, backends, fusion)
crates/qsv-cli    `qsv` binary (smoke runner; QASM3 + sampling later)
docs/             design doc + research synthesis
docs/reference/   state_vector.jl — the pedagogical reference notebook
references/        shallow clones of qsim/QuEST/aer/Yao/spinoza/amh-code (git-ignored)
bench/            benchmark circuits & results (added at the benchmarking milestone)
```

## Why Rust, and what's novel

The Rust quantum-sim ecosystem has CPU SIMD simulators (`spinoza`) and circuit DSLs
(`qoqo`), but no native-Rust engine combining gate fusion + cache-blocking + SoA + SIMD
behind a pluggable, modern GPU seam. qsv targets that gap — with the *optimization writeup*
as the deliverable. Primary dev/benchmark hardware: Apple M3 Pro (NEON); secondary: x86
(AVX2/AVX-512).

## License

MIT OR Apache-2.0.
