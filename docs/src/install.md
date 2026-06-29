# Installation

## Prerequisites

- **Rust 1.86+** (stable). The project pins `rust-version = "1.86"`.
- A C toolchain is *not* required — qsv-core is pure Rust with no external dependencies.

Check your toolchain:

```bash
rustc --version   # 1.86.0 or newer
```

## Build & test

```bash
git clone <repo-url> statevec-sim
cd statevec-sim

cargo build --workspace            # debug build
cargo build --release --workspace  # optimized build
cargo test  --workspace            # unit + oracle + differential tests
```

A clean run should report all tests passing, with `cargo clippy --workspace --all-targets --
-D warnings` and `cargo fmt --check` both clean (these gate CI).

## Run the CLI

The `qsv` binary is a smoke runner that prepares a GHZ state and prints its outcome
probabilities:

```bash
cargo run --release --bin qsv -- 24   # GHZ on 24 qubits (256 MB statevector)
```

## Optional tooling

These power the documentation and the benchmarking/profiling workflow:

```bash
# Documentation (this book). mdBook 0.5+ needs rustc 1.88; on 1.86 pin the 0.4 line:
cargo install mdbook --version "^0.4" --locked
mdbook serve docs        # live-reload at http://localhost:3000
mdbook build docs        # render to docs/book/

# Profiling (macOS / Apple Silicon)
cargo install samply --locked          # sampling profiler, Firefox-profiler UI
cargo install cargo-instruments        # Xcode Instruments integration

# Benchmarks use criterion (pulled automatically by `cargo bench`).
```

## Memory ceiling

A statevector of $N$ qubits in `f64` needs $2^N \times 16$ bytes (real + imaginary).
On a 36 GB machine the practical in-place ceiling is **~30 qubits** (`f64`, 16 GB) or **~31**
(`f32`). qsv updates the state in place precisely because at that size there is no room for a
second buffer.
