# Tutorial

This walks through using `qsv-core` as a library: building circuits, running them on a
backend, and reading out results.

## A first circuit: the Bell state

```rust
use qsv_core::prelude::*;

// 2-qubit register, then H(0); CX(0,1)  ->  (|00> + |11>)/√2
let mut circuit = Circuit::<f64>::new(2);
circuit.h(0).cx(0, 1);

let state = BitShiftBackend.execute(&circuit);

let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
assert!((state.amplitude(0b00).re - inv_sqrt2).abs() < 1e-12);
assert!((state.amplitude(0b11).re - inv_sqrt2).abs() < 1e-12);
```

The builder is fluent and chainable. Qubit 0 is the **least significant** bit of the basis
index, so `|00⟩` is index `0` and `|11⟩` is index `3`.

## Gates

Single-qubit: `h, x, y, z, s, t, sx`, and parametric `rx(θ), ry(θ), rz(θ), phase(λ)`.
Two-qubit (control first): `cx(c,t), cz(a,b), cphase(c,t,λ), swap(a,b), rzz(a,b,θ)`.

```rust
let mut c = Circuit::<f64>::new(3);
c.h(0)
 .rx(1, std::f64::consts::FRAC_PI_2)
 .cx(0, 1)
 .cz(1, 2)
 .rzz(0, 2, 0.7);   // QAOA-style two-qubit rotation
```

Arbitrary unitaries can be pushed directly as a dense matrix on an ordered qubit list with
`circuit.push(dense_gate, &qubits)` — this is how a Toffoli or a fused gate is applied.

## Reading out results

```rust
let backend = BitShiftBackend;
let state = backend.execute(&circuit);

let amp = state.amplitude(5);          // Cplx<f64> at basis index 5
let p   = state.probabilities();       // Vec<f64> of |ψ_i|²
let norm = state.norm_sqr();           // ⟨ψ|ψ⟩, == 1 for a valid state
```

## Built-in circuit generators

The `circuits` module provides standard circuits (also used by the tests and benchmarks):

```rust
use qsv_core::circuits::{ghz, qft, random_circuit};

let g = ghz(5);                        // (|0…0> + |1…1>)/√2
let f = qft(8);                        // Quantum Fourier Transform
let r = random_circuit(10, 200, 42);   // 200 random gates, reproducible (seed 42)

// QFT of |0…0> is the uniform superposition:
let s = BitShiftBackend.execute(&qft(4));
let expected = 1.0 / ((1usize << 4) as f64);          // |amplitude|² for every state
assert!((s.amplitude(0).norm_sqr() - expected).abs() < 1e-12);
```

## Choosing a backend

qsv exposes several backends, all implementing the same `Backend` trait — so you can swap
them with a one-line change. They exist side by side so each optimization milestone stays
runnable and benchmarkable, and so every fast kernel can be validated against the slow one.

| Backend | What it is | Use it for |
| --- | --- | --- |
| `RefBackend` | naive, independently-implemented oracle | correctness checks, small `N` |
| `ReshapeBackend` | block-structured, out-of-place (v0.1) | the milestone baseline |
| `BitShiftBackend` | in-place bit-shift kernel (v0.2) | the reference fast kernel |
| `CpuBackend` | bounds-check-free + nested-block + rayon (v0.3–v0.5) | **real simulation (fastest)** |

`CpuBackend::parallel()` (the `Default`) multithreads above a size threshold;
`CpuBackend::serial()` forces single-threaded. Build with `--no-default-features` to drop the
`parallel` feature (and the rayon dependency) entirely.

```rust
// Identical circuit, different engine — the seam in action.
let a = RefBackend.execute(&circuit);
let b = BitShiftBackend.execute(&circuit);
// a and b agree amplitude-for-amplitude (this is exactly what the test suite checks).
```

Choosing `f32` instead of `f64` is a type parameter: `Circuit::<f32>::new(n)` — half the
memory traffic, lower precision.

## Precision and generics

Everything is generic over the `Real` trait (`f64` by default, `f32` available). Kernels are
monomorphized per scalar type — there is no dynamic dispatch in the hot path.
