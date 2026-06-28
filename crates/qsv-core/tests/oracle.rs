//! End-to-end correctness tests for the v0.0 oracle backend.
//! These become the golden behavior every optimized backend must reproduce.

use qsv_core::prelude::*;

const TOL: f64 = 1e-12;

fn close(a: Cplx<f64>, re: f64, im: f64) -> bool {
    (a.re - re).abs() < TOL && (a.im - im).abs() < TOL
}

#[test]
fn bell_state() {
    // H(0); CX(0,1)  ->  (|00> + |11>)/√2
    let mut c = Circuit::<f64>::new(2);
    c.h(0).cx(0, 1);
    let s = RefBackend.execute(&c);

    let a = std::f64::consts::FRAC_1_SQRT_2;
    assert!(close(s.amplitude(0b00), a, 0.0));
    assert!(close(s.amplitude(0b01), 0.0, 0.0));
    assert!(close(s.amplitude(0b10), 0.0, 0.0));
    assert!(close(s.amplitude(0b11), a, 0.0));
    assert!((s.norm_sqr() - 1.0).abs() < TOL);
}

#[test]
fn ghz_three_qubits() {
    // H(0); CX(0,1); CX(0,2)  ->  (|000> + |111>)/√2
    let mut c = Circuit::<f64>::new(3);
    c.h(0).cx(0, 1).cx(0, 2);
    let s = RefBackend.execute(&c);

    let a = std::f64::consts::FRAC_1_SQRT_2;
    assert!(close(s.amplitude(0b000), a, 0.0));
    assert!(close(s.amplitude(0b111), a, 0.0));
    for i in 1..7 {
        assert!(
            close(s.amplitude(i), 0.0, 0.0),
            "amplitude {i} should vanish"
        );
    }
}

#[test]
fn x_is_self_inverse_on_register() {
    // X(2) twice returns the original basis state on a 4-qubit register.
    let mut c = Circuit::<f64>::new(4);
    c.x(2).x(2);
    let s = RefBackend.execute(&c);
    assert!(close(s.amplitude(0), 1.0, 0.0));
    assert!((s.norm_sqr() - 1.0).abs() < TOL);
}

#[test]
fn controlled_gate_leaves_control_zero_subspace_untouched() {
    // Start in |0...0>: control qubit 0 is |0>, so CX(0,1) must be a no-op.
    let mut c = Circuit::<f64>::new(2);
    c.cx(0, 1);
    let s = RefBackend.execute(&c);
    assert!(close(s.amplitude(0), 1.0, 0.0));
}

#[test]
fn single_qubit_rotation_matches_hand_calc() {
    // RX(π) on |0> = -i|1>.
    let mut c = Circuit::<f64>::new(1);
    c.rx(0, std::f64::consts::PI);
    let s = RefBackend.execute(&c);
    assert!(close(s.amplitude(0), 0.0, 0.0));
    assert!(close(s.amplitude(1), 0.0, -1.0));
}

#[test]
fn norm_preserved_through_mixed_circuit() {
    let mut c = Circuit::<f64>::new(4);
    c.h(0)
        .rx(1, 0.4)
        .ry(2, 1.1)
        .cx(0, 1)
        .cz(1, 2)
        .rzz(2, 3, 0.7)
        .swap(0, 3)
        .t(2);
    let s = RefBackend.execute(&c);
    assert!(
        (s.norm_sqr() - 1.0).abs() < 1e-10,
        "norm = {}",
        s.norm_sqr()
    );
}
