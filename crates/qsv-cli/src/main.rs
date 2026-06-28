//! Minimal CLI smoke runner for v0.0. Builds a GHZ state and prints its probabilities.
//! A real arg-parsing CLI (QASM3 input, sampling) arrives at a later milestone.

use qsv_core::prelude::*;

fn main() {
    let n: u32 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(4);

    // GHZ(n): H(0) then CX(0, q) for q = 1..n  ->  (|0…0> + |1…1>)/√2
    let mut circuit = Circuit::<f64>::new(n);
    circuit.h(0);
    for q in 1..n {
        circuit.cx(0, q);
    }

    // Fastest CPU backend (v0.3–v0.5: bounds-check-free + nested-block + threaded).
    let backend = CpuBackend::default();
    let state = backend.execute(&circuit);
    let probs = backend.probabilities(&state);

    println!("GHZ({n}) — non-negligible outcome probabilities:");
    for (i, &p) in probs.iter().enumerate() {
        if p > 1e-12 {
            println!("  |{i:0width$b}⟩ : {p:.6}", width = n as usize);
        }
    }
    println!("⟨ψ|ψ⟩ = {:.6}", state.norm_sqr());
}
