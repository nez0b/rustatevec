//! v0.1 — `ReshapeBackend`: block-structured, **out-of-place** gate application.
//!
//! This is the "reshape / tensor-contraction" mental model from the reference notebook,
//! made concrete: for an m-qubit gate, the `2^N` amplitudes are conceptually reshaped so the
//! target qubits form one axis; we iterate the `2^(N-m)` outer blocks and, for each, do a
//! small `2^m × 2^m` matrix–vector product. Compared to the oracle it is organized by block
//! (better locality, one small matmul per block) but is still **general** and still pays for
//! a **per-gate `2^N` allocation + copy-back** — the inefficiencies v0.2 removes.
//!
//! Complexity `O(2^N · 2^m)` per gate; reaches usable qubit counts but is allocation-heavy.

use super::Backend;
use crate::complex::Cplx;
use crate::gate::DenseGate;
use crate::real::Real;
use crate::state::layout::{insert_zero_bits, scatter_bits};
use crate::state::StateVector;

/// Out-of-place, block-structured reference-quality simulator (milestone v0.1).
#[derive(Clone, Copy, Debug, Default)]
pub struct ReshapeBackend;

impl<R: Real> Backend<R> for ReshapeBackend {
    type State = StateVector<R>;

    fn alloc(&self, n_qubits: u32) -> StateVector<R> {
        StateVector::zeros(n_qubits)
    }

    fn init_basis(&self, state: &mut StateVector<R>, basis: usize) {
        *state = StateVector::basis(state.n_qubits(), basis);
    }

    fn apply(&self, state: &mut StateVector<R>, gate: &DenseGate<R>, qubits: &[u32]) {
        debug_assert_eq!(gate.n_qubits() as usize, qubits.len());
        let m = qubits.len();
        let sub = 1usize << m;
        let dim = state.dim();

        // Sorted target positions (stack, no heap) to anchor each block.
        let mut sorted_buf = [0u32; 8];
        sorted_buf[..m].copy_from_slice(qubits);
        let sorted = &mut sorted_buf[..m];
        sorted.sort_unstable();

        // Per-gate output buffer — the defining cost of this milestone.
        let mut out = vec![Cplx::<R>::zero(); dim];

        for o in 0..(dim >> m) {
            let base = insert_zero_bits(o, sorted);
            for r in 0..sub {
                let mut acc = Cplx::<R>::zero();
                for cc in 0..sub {
                    let col = base | scatter_bits(cc, qubits);
                    acc = acc + gate.at(r, cc) * state.amplitude(col);
                }
                out[base | scatter_bits(r, qubits)] = acc;
            }
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
