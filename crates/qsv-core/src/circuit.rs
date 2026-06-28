//! Circuit representation and a fluent builder.
//!
//! A [`Circuit`] is an ordered list of [`GateOp`]s (a gate + the qubits it acts on). It is
//! **backend-agnostic**: the same circuit runs on the oracle or any future optimized/GPU
//! backend via [`Backend::execute`](crate::backend::Backend::execute). The fusion pass
//! (later milestone) is a `Circuit -> Circuit` transform.

use crate::gate::{self, DenseGate};
use crate::real::Real;

/// One gate application: the gate matrix and the (ordered) qubits it acts on.
#[derive(Clone, Debug)]
pub struct GateOp<R: Real> {
    gate: DenseGate<R>,
    qubits: Vec<u32>,
}

impl<R: Real> GateOp<R> {
    #[inline(always)]
    pub fn gate(&self) -> &DenseGate<R> {
        &self.gate
    }
    #[inline(always)]
    pub fn qubits(&self) -> &[u32] {
        &self.qubits
    }
}

/// A quantum circuit over `n_qubits` qubits.
#[derive(Clone, Debug)]
pub struct Circuit<R: Real> {
    n_qubits: u32,
    ops: Vec<GateOp<R>>,
}

impl<R: Real> Circuit<R> {
    pub fn new(n_qubits: u32) -> Self {
        Self {
            n_qubits,
            ops: Vec::new(),
        }
    }

    #[inline(always)]
    pub fn n_qubits(&self) -> u32 {
        self.n_qubits
    }
    #[inline(always)]
    pub fn ops(&self) -> &[GateOp<R>] {
        &self.ops
    }

    /// Append a gate acting on `qubits` (validated against the register size).
    pub fn push(&mut self, g: DenseGate<R>, qubits: &[u32]) -> &mut Self {
        assert_eq!(
            1usize << g.n_qubits(),
            1usize << qubits.len() as u32,
            "gate arity must match qubit count"
        );
        for &q in qubits {
            assert!(
                q < self.n_qubits,
                "qubit {q} out of range for {} qubits",
                self.n_qubits
            );
        }
        // Distinct-qubit check (gather/scatter assume disjoint positions).
        for i in 0..qubits.len() {
            for j in (i + 1)..qubits.len() {
                assert_ne!(qubits[i], qubits[j], "repeated qubit in gate");
            }
        }
        self.ops.push(GateOp {
            gate: g,
            qubits: qubits.to_vec(),
        });
        self
    }

    // ---- Single-qubit ----
    pub fn h(&mut self, q: u32) -> &mut Self {
        self.push(gate::h(), &[q])
    }
    pub fn x(&mut self, q: u32) -> &mut Self {
        self.push(gate::x(), &[q])
    }
    pub fn y(&mut self, q: u32) -> &mut Self {
        self.push(gate::y(), &[q])
    }
    pub fn z(&mut self, q: u32) -> &mut Self {
        self.push(gate::z(), &[q])
    }
    pub fn s(&mut self, q: u32) -> &mut Self {
        self.push(gate::s_gate(), &[q])
    }
    pub fn t(&mut self, q: u32) -> &mut Self {
        self.push(gate::t_gate(), &[q])
    }
    pub fn sx(&mut self, q: u32) -> &mut Self {
        self.push(gate::sx(), &[q])
    }
    pub fn rx(&mut self, q: u32, theta: f64) -> &mut Self {
        self.push(gate::rx(theta), &[q])
    }
    pub fn ry(&mut self, q: u32, theta: f64) -> &mut Self {
        self.push(gate::ry(theta), &[q])
    }
    pub fn rz(&mut self, q: u32, theta: f64) -> &mut Self {
        self.push(gate::rz(theta), &[q])
    }
    pub fn phase(&mut self, q: u32, lambda: f64) -> &mut Self {
        self.push(gate::phase(lambda), &[q])
    }

    // ---- Two-qubit (control first, per the gate-module convention) ----
    pub fn cx(&mut self, control: u32, target: u32) -> &mut Self {
        self.push(gate::cx(), &[control, target])
    }
    pub fn cz(&mut self, a: u32, b: u32) -> &mut Self {
        self.push(gate::cz(), &[a, b])
    }
    pub fn swap(&mut self, a: u32, b: u32) -> &mut Self {
        self.push(gate::swap(), &[a, b])
    }
    pub fn rzz(&mut self, a: u32, b: u32, theta: f64) -> &mut Self {
        self.push(gate::rzz(theta), &[a, b])
    }
}
