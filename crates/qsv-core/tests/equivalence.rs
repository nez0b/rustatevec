//! Differential robustness tests: every optimized backend must reproduce the v0.0 oracle
//! amplitude-for-amplitude on randomized and structured circuits. A bug in any kernel's
//! index arithmetic shows up here as a mismatch the oracle (an independent implementation)
//! does not share.

use qsv_core::circuits::{ghz, qft, random_circuit};
use qsv_core::gate::DenseGate;
use qsv_core::prelude::*;

/// Largest per-amplitude distance between two states of equal dimension.
fn max_abs_diff(a: &StateVector<f64>, b: &StateVector<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim());
    (0..a.dim())
        .map(|i| (a.amplitude(i) - b.amplitude(i)).norm_sqr().sqrt())
        .fold(0.0, f64::max)
}

/// Execute a circuit on a backend and return the host state.
fn run<B>(backend: &B, c: &Circuit<f64>) -> StateVector<f64>
where
    B: Backend<f64, State = StateVector<f64>>,
{
    backend.download(&backend.execute(c))
}

const TOL: f64 = 1e-9;

#[test]
fn reshape_and_bitshift_match_oracle_on_random_circuits() {
    for seed in 0..200u64 {
        let n = 3 + (seed % 6) as u32; // 3..=8 qubits
        let depth = 20 + (seed % 40) as usize;
        let c = random_circuit(n, depth, seed);

        let oracle = run(&RefBackend, &c);
        let reshape = run(&ReshapeBackend, &c);
        let bitshift = run(&BitShiftBackend, &c);
        let cpu_serial = run(&CpuBackend::serial(), &c);
        let cpu_par = run(&CpuBackend::parallel(), &c);

        assert!(
            max_abs_diff(&oracle, &reshape) < TOL,
            "ReshapeBackend diverged from oracle (seed {seed}, n {n})"
        );
        assert!(
            max_abs_diff(&oracle, &bitshift) < TOL,
            "BitShiftBackend diverged from oracle (seed {seed}, n {n})"
        );
        assert!(
            max_abs_diff(&oracle, &cpu_serial) < TOL,
            "CpuBackend::serial diverged from oracle (seed {seed}, n {n})"
        );
        assert!(
            max_abs_diff(&oracle, &cpu_par) < TOL,
            "CpuBackend::parallel diverged from oracle (seed {seed}, n {n})"
        );
        // Sanity: every backend preserves the norm.
        assert!((bitshift.norm_sqr() - 1.0).abs() < 1e-8);
    }
}

#[test]
fn bitshift_matches_oracle_at_ten_qubits() {
    // Larger register, deeper circuits — fewer seeds to keep runtime modest.
    for seed in 0..15u64 {
        let c = random_circuit(10, 80, 9000 + seed);
        let oracle = run(&RefBackend, &c);
        let bitshift = run(&BitShiftBackend, &c);
        assert!(max_abs_diff(&oracle, &bitshift) < TOL, "seed {seed}");
    }
}

#[test]
fn cpu_threaded_matches_oracle_above_threshold() {
    // n = 14 (16384 amplitudes) is above CpuBackend's threading threshold, so this actually
    // exercises the rayon path — including the multi-qubit (CX/CZ/SWAP/RZZ) parallel kernel.
    for seed in 0..6u64 {
        let c = random_circuit(14, 60, 4242 + seed);
        let oracle = run(&RefBackend, &c);
        let cpu_par = run(&CpuBackend::parallel(), &c);
        let cpu_serial = run(&CpuBackend::serial(), &c);
        assert!(
            max_abs_diff(&oracle, &cpu_par) < TOL,
            "threaded seed {seed}"
        );
        assert!(
            max_abs_diff(&oracle, &cpu_serial) < TOL,
            "serial seed {seed}"
        );
    }
}

#[test]
fn qft_produces_uniform_superposition_from_zero() {
    for &n in &[1u32, 2, 3, 5, 8] {
        let s = run(&BitShiftBackend, &qft(n));
        let expect_prob = 1.0 / ((1usize << n) as f64);
        for i in 0..s.dim() {
            assert!(
                (s.amplitude(i).norm_sqr() - expect_prob).abs() < 1e-12,
                "QFT({n}) amplitude {i} not uniform"
            );
        }
    }
}

#[test]
fn qft_matches_oracle() {
    for &n in &[2u32, 3, 5, 8] {
        let c = qft(n);
        let oracle = run(&RefBackend, &c);
        let bitshift = run(&BitShiftBackend, &c);
        let reshape = run(&ReshapeBackend, &c);
        let cpu = run(&CpuBackend::parallel(), &c);
        assert!(max_abs_diff(&oracle, &bitshift) < TOL, "qft bitshift n {n}");
        assert!(max_abs_diff(&oracle, &reshape) < TOL, "qft reshape n {n}");
        assert!(max_abs_diff(&oracle, &cpu) < TOL, "qft cpu n {n}");
    }
}

#[test]
fn ghz_is_correct_on_bitshift() {
    let s = run(&BitShiftBackend, &ghz(6));
    let a = std::f64::consts::FRAC_1_SQRT_2;
    assert!((s.amplitude(0).re - a).abs() < 1e-12);
    assert!((s.amplitude(s.dim() - 1).re - a).abs() < 1e-12);
    // All intermediate amplitudes vanish.
    for i in 1..(s.dim() - 1) {
        assert!(s.amplitude(i).norm_sqr() < 1e-24, "amplitude {i}");
    }
}

/// Toffoli (CCX) as an 8×8 gate on `qs = [c0, c1, t]` (internal index = c0 + 2·c1 + 4·t):
/// swaps the two states with both controls set (indices 3 and 7).
fn ccx_gate() -> DenseGate<f64> {
    let mut data = vec![Cplx::<f64>::zero(); 64];
    for i in 0..8usize {
        let target = match i {
            3 => 7,
            7 => 3,
            other => other,
        };
        data[i * 8 + target] = Cplx::one();
    }
    DenseGate::new(3, data)
}

#[test]
fn three_qubit_gate_matches_oracle() {
    // Exercises the general apply_mq path (m = 3): prepare a nontrivial state, then apply
    // Toffoli on a few target/control orderings and compare to the oracle.
    for (seed, qs) in [
        (1u64, [0u32, 1, 2]),
        (2, [2, 0, 4]),
        (3, [4, 3, 1]),
        (4, [5, 2, 0]),
    ] {
        let mut c = random_circuit(6, 40, seed);
        c.push(ccx_gate(), &qs);
        // A second multi-qubit gate after more single-qubit activity.
        c.h(0).rx(3, 0.9);
        c.push(ccx_gate(), &[1, 4, 5]);

        let oracle = run(&RefBackend, &c);
        let bitshift = run(&BitShiftBackend, &c);
        let reshape = run(&ReshapeBackend, &c);
        let cpu = run(&CpuBackend::serial(), &c);
        assert!(
            max_abs_diff(&oracle, &bitshift) < TOL,
            "ccx bitshift seed {seed}"
        );
        assert!(
            max_abs_diff(&oracle, &reshape) < TOL,
            "ccx reshape seed {seed}"
        );
        assert!(max_abs_diff(&oracle, &cpu) < TOL, "ccx cpu seed {seed}");
    }
}
