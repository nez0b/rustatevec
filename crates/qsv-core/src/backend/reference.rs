//! `RefBackend` — the naive **correctness oracle** (milestone v0.0).
//!
//! Deliberately simple and implemented *independently* of the optimized pair-iteration
//! kernel it will check: it applies a gate by, for every output amplitude, gathering the
//! gate's sub-index from the involved qubits and summing the matrix row against the
//! matching input amplitudes (gather/scatter), rather than the `insert_zero_bit` pairing
//! trick. Cost is `O(2^N · 2^m)` per gate — fine for the small `N` used in tests, and the
//! whole point is to be *obviously correct*, not fast.
//!
//! It also doubles as the **second backend** that proves the [`Backend`] seam is not
//! secretly CPU-specific (a third, GPU backend will reuse the same trait).

use super::Backend;
use crate::complex::Cplx;
use crate::gate::DenseGate;
use crate::real::Real;
use crate::state::layout::{gather_bits, scatter_bits};
use crate::state::StateVector;

/// Naive dense-apply reference simulator.
#[derive(Clone, Copy, Debug, Default)]
pub struct RefBackend;

impl<R: Real> Backend<R> for RefBackend {
    type State = StateVector<R>;

    fn alloc(&self, n_qubits: u32) -> StateVector<R> {
        StateVector::zeros(n_qubits)
    }

    fn init_basis(&self, state: &mut StateVector<R>, basis: usize) {
        *state = StateVector::basis(state.n_qubits(), basis);
    }

    fn apply(&self, state: &mut StateVector<R>, gate: &DenseGate<R>, qubits: &[u32]) {
        debug_assert_eq!(gate.n_qubits() as usize, qubits.len());
        let n = state.n_qubits();
        let dim = 1usize << n;

        // Mask of the gate-involved qubit positions, and its complement within N bits.
        let qmask: usize = qubits.iter().map(|&q| 1usize << q).sum();
        let nonmask = !qmask & (dim - 1);

        let gdim = gate.dim();
        let mut out = vec![Cplx::<R>::zero(); dim];

        for (r, slot) in out.iter_mut().enumerate() {
            // Output amplitude r: sum over input columns sharing r's non-involved bits.
            let r_fixed = r & nonmask;
            let gr = gather_bits(r, qubits);
            let mut acc = Cplx::<R>::zero();
            for gc in 0..gdim {
                let col = r_fixed | scatter_bits(gc, qubits);
                acc = acc + gate.at(gr, gc) * state.amplitude(col);
            }
            *slot = acc;
        }

        for (i, amp) in out.into_iter().enumerate() {
            state.set(i, amp);
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
