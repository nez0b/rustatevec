//! Backend-level sampling / expectation tests: exercises the `Backend::sample` and
//! `Backend::expectation_z` default impls through a real backend + circuit.

use qsv_core::circuits::{ghz, qft, SplitMix64};
use qsv_core::prelude::*;

#[test]
fn sample_ghz_yields_only_all_zero_or_all_one() {
    let backend = CpuBackend::parallel();
    let state = backend.execute(&ghz(6));
    let mut rng = SplitMix64::new(42);
    let shots = 50_000;
    let outcomes = backend.sample(&state, shots, &mut rng);
    let (all0, all1) = (0usize, 0b111111usize);
    let mut c0 = 0;
    for o in &outcomes {
        assert!(*o == all0 || *o == all1, "GHZ produced forbidden outcome {o:#08b}");
        if *o == all0 {
            c0 += 1;
        }
    }
    // ~50/50 split.
    let f0 = c0 as f64 / shots as f64;
    assert!((f0 - 0.5).abs() < 0.02, "GHZ |0…0⟩ fraction {f0} not ≈ 0.5");
}

#[test]
fn sample_histogram_matches_probabilities() {
    let backend = CpuBackend::parallel();
    let state = backend.execute(&qft(6)); // uniform superposition from |0…0⟩
    let probs = backend.probabilities(&state);
    let mut rng = SplitMix64::new(7);
    let shots = 400_000;
    let dim = probs.len();
    let mut counts = vec![0usize; dim];
    for o in backend.sample(&state, shots, &mut rng) {
        counts[o] += 1;
    }
    for (i, &p) in probs.iter().enumerate() {
        let f = counts[i] as f64 / shots as f64;
        assert!((f - p).abs() < 0.01, "outcome {i}: sampled {f} vs prob {p}");
    }
}

#[test]
fn expectation_z_on_ghz() {
    let backend = CpuBackend::parallel();
    let state = backend.execute(&ghz(5));
    // GHZ = (|0…0⟩+|1…1⟩)/√2: a Z-string has eigenvalues +1 and (-1)^weight on the two basis
    // states, so ⟨Z-string⟩ = +1 for even weight and 0 for odd weight.
    assert!(backend.expectation_z(&state, 0b00001).abs() < 1e-9); // weight 1 (odd) → 0
    assert!((backend.expectation_z(&state, 0b00011) - 1.0).abs() < 1e-9); // weight 2 (even) → +1
    assert!((backend.expectation_z(&state, 0b01111) - 1.0).abs() < 1e-9); // weight 4 (even) → +1
    assert!(backend.expectation_z(&state, 0b11111).abs() < 1e-9); // weight 5 (odd) → 0
}
