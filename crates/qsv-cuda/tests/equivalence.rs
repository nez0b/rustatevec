//! Differential test: `CudaBackend` must reproduce the `RefBackend` oracle amplitude-for-
//! amplitude, exactly like every CPU backend in `qsv-core/tests/equivalence.rs`. Gated on the
//! `cuda` feature (needs an NVIDIA GPU + CUDA toolkit). Run with:
//!   `cargo test -p qsv-cuda --features cuda`

#![cfg(feature = "cuda")]

use qsv_core::circuits::{ghz, qft, random_circuit};
use qsv_core::prelude::*;
use qsv_core::state::StateVector;
use qsv_cuda::CudaBackend;

fn max_abs_diff(a: &StateVector<f64>, b: &StateVector<f64>) -> f64 {
    let mut m = 0.0;
    for i in 0..a.dim() {
        let (ca, cb) = (a.amplitude(i), b.amplitude(i));
        m = f64::max(m, (ca.re - cb.re).abs());
        m = f64::max(m, (ca.im - cb.im).abs());
    }
    m
}

fn run<B: Backend<f64>>(backend: &B, c: &Circuit<f64>) -> StateVector<f64> {
    let st = backend.execute(c);
    backend.download(&st)
}

#[test]
fn cuda_matches_oracle_on_random_circuits() {
    let gpu = CudaBackend::new(0).expect("init CUDA device 0");
    for seed in 0..40u64 {
        let n = 3 + (seed % 6) as u32; // 3..=8
        let depth = 20 + (seed as usize % 40);
        let circ = random_circuit(n, depth, 0xABCD_0000 ^ seed);
        let want = run(&RefBackend, &circ);
        let got = run(&gpu, &circ);
        assert!(
            max_abs_diff(&want, &got) < 1e-9,
            "mismatch seed={seed} n={n} depth={depth} diff={}",
            max_abs_diff(&want, &got)
        );
    }
}

#[test]
fn cuda_matches_oracle_qft_and_ghz() {
    let gpu = CudaBackend::new(0).expect("init CUDA device 0");
    for n in [2u32, 3, 5, 8] {
        let circ = qft(n);
        assert!(
            max_abs_diff(&run(&RefBackend, &circ), &run(&gpu, &circ)) < 1e-9,
            "qft n={n}"
        );
    }
    for n in [3u32, 6, 10] {
        let circ = ghz(n);
        assert!(
            max_abs_diff(&run(&RefBackend, &circ), &run(&gpu, &circ)) < 1e-9,
            "ghz n={n}"
        );
    }
}

#[test]
fn cuda_probabilities_sum_to_one() {
    let gpu = CudaBackend::new(0).expect("init CUDA device 0");
    let circ = qft(10);
    let st = gpu.execute(&circ);
    let p: f64 = gpu.probabilities(&st).iter().sum();
    assert!((p - 1.0).abs() < 1e-9, "prob sum = {p}");
}
