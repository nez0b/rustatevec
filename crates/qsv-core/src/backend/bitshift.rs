//! v0.2 — `BitShiftBackend`: the in-place pair-iteration kernel.
//!
//! The heart of the project. For a 1-qubit gate it iterates the `2^(N-1)` amplitude *pairs*
//! directly via [`insert_zero_bit`], updating each pair **in place** with the 2×2 matrix —
//! no per-gate allocation and half the index iterations of v0.1. For m-qubit gates it
//! iterates the `2^(N-m)` blocks, gathering the `2^m` sub-amplitudes into a small **stack**
//! buffer, applying the matrix, and writing back in place.
//!
//! This is the same indexing every production simulator uses (QuEST/qsim/Aer/Yao); see
//! `docs/research/01-cpu-simulators.md`. It unlocks ~30-qubit simulation on the dev box and
//! is the baseline that SIMD (v0.7), fusion (v0.8), and threading (v0.5) build on.

use super::Backend;
use crate::complex::Cplx;
use crate::gate::DenseGate;
use crate::real::Real;
use crate::state::layout::{insert_zero_bit, insert_zero_bits, scatter_bits};
use crate::state::StateVector;

/// Largest gate arity the stack-buffered `apply_mq` supports (`2^6` = 64 amplitudes).
/// Comfortably covers v1 gates (≤ 2q) and future fused gates (≤ 5q).
const MAX_SUB: usize = 64;

/// In-place bit-shift kernel simulator (milestone v0.2).
#[derive(Clone, Copy, Debug, Default)]
pub struct BitShiftBackend;

impl BitShiftBackend {
    /// Canonical in-place 1-qubit kernel: pair `(a0, a1)` differs only in bit `q`.
    fn apply_1q<R: Real>(state: &mut StateVector<R>, gate: &DenseGate<R>, q: u32) {
        let g00 = gate.at(0, 0);
        let g01 = gate.at(0, 1);
        let g10 = gate.at(1, 0);
        let g11 = gate.at(1, 1);
        let pairs = state.dim() >> 1;
        let (re, im) = state.parts_mut();
        for i in 0..pairs {
            let a0 = insert_zero_bit(i, q);
            let a1 = a0 | (1usize << q);
            let x0 = Cplx::new(re[a0], im[a0]);
            let x1 = Cplx::new(re[a1], im[a1]);
            let y0 = g00 * x0 + g01 * x1;
            let y1 = g10 * x0 + g11 * x1;
            re[a0] = y0.re;
            im[a0] = y0.im;
            re[a1] = y1.re;
            im[a1] = y1.im;
        }
    }

    /// General in-place m-qubit kernel via a per-block stack buffer (read-all then write-all).
    fn apply_mq<R: Real>(state: &mut StateVector<R>, gate: &DenseGate<R>, qubits: &[u32]) {
        let m = qubits.len();
        let sub = 1usize << m;
        assert!(sub <= MAX_SUB, "apply_mq supports up to 6-qubit gates");

        let mut sorted_buf = [0u32; 6];
        sorted_buf[..m].copy_from_slice(qubits);
        let sorted = &mut sorted_buf[..m];
        sorted.sort_unstable();

        let blocks = state.dim() >> m;
        let (re, im) = state.parts_mut();
        let mut amp = [Cplx::<R>::zero(); MAX_SUB];

        for o in 0..blocks {
            let base = insert_zero_bits(o, sorted);
            // Gather this block's sub-amplitudes.
            for (s, slot) in amp.iter_mut().enumerate().take(sub) {
                let idx = base | scatter_bits(s, qubits);
                *slot = Cplx::new(re[idx], im[idx]);
            }
            // Apply the matrix and write back in place.
            for r in 0..sub {
                let mut acc = Cplx::<R>::zero();
                for (cc, &a) in amp.iter().enumerate().take(sub) {
                    acc = acc + gate.at(r, cc) * a;
                }
                let idx = base | scatter_bits(r, qubits);
                re[idx] = acc.re;
                im[idx] = acc.im;
            }
        }
    }
}

impl<R: Real> Backend<R> for BitShiftBackend {
    type State = StateVector<R>;

    fn alloc(&self, n_qubits: u32) -> StateVector<R> {
        StateVector::zeros(n_qubits)
    }

    fn init_basis(&self, state: &mut StateVector<R>, basis: usize) {
        *state = StateVector::basis(state.n_qubits(), basis);
    }

    fn apply(&self, state: &mut StateVector<R>, gate: &DenseGate<R>, qubits: &[u32]) {
        debug_assert_eq!(gate.n_qubits() as usize, qubits.len());
        if qubits.len() == 1 {
            Self::apply_1q(state, gate, qubits[0]);
        } else {
            Self::apply_mq(state, gate, qubits);
        }
    }

    fn amplitude(&self, state: &StateVector<R>, index: usize) -> Cplx<R> {
        state.amplitude(index)
    }

    fn probabilities(&self, state: &StateVector<R>) -> Vec<R> {
        state.probabilities()
    }

    fn download(&self, state: &StateVector<R>) -> StateVector<R> {
        state.clone()
    }
}
