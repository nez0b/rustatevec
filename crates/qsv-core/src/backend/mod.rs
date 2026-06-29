//! The `Backend` trait — **the pluggable seam**.
//!
//! Everything above this line (circuit, gates, fusion) is backend-agnostic. A backend owns
//! the amplitude storage (`type State`) and the gate-application kernels. This is the seam
//! behind which a future CUDA/cuTile or Metal/wgpu backend will live **without touching the
//! circuit/gate layer**. Leak-proofing rules, enforced by the signatures below:
//!
//! - `State` is an associated type — CPU uses host [`StateVector`], a GPU backend would use
//!   an opaque device handle. No method ever hands out a `&mut [R]` to host memory.
//! - Reductions ([`probabilities`](Backend::probabilities), and later `sample`/`expectation`)
//!   are backend methods, so a GPU computes them on-device rather than forcing a copy back.
//! - [`download`](Backend::download) is the single explicit device→host boundary crossing
//!   (tests / CLI only).
//!
//! v0.0 exposes one general `apply(matrix, qubits)`; arity-specialized fast paths
//! (`apply_1q`, `apply_diagonal`, …) are added with the optimized CPU backend.

pub mod bitshift;
pub mod cpu;
pub mod reference;
pub mod reshape;
pub mod simd;

use crate::circuit::Circuit;
use crate::circuits::SplitMix64;
use crate::complex::Cplx;
use crate::gate::DenseGate;
use crate::real::Real;
use crate::state::StateVector;

pub trait Backend<R: Real> {
    /// Backend-owned amplitude storage (host vector, device handle, …).
    type State;

    /// Allocate state for `n_qubits` (contents unspecified; call [`init_basis`](Self::init_basis)).
    fn alloc(&self, n_qubits: u32) -> Self::State;

    /// Reset to the computational basis state `|basis⟩`.
    fn init_basis(&self, state: &mut Self::State, basis: usize);

    /// Apply `gate` (a `2^m × 2^m` unitary) to the ordered `qubits` (see gate-module convention).
    fn apply(&self, state: &mut Self::State, gate: &DenseGate<R>, qubits: &[u32]);

    /// Single amplitude (for tests / inspection).
    fn amplitude(&self, state: &Self::State, index: usize) -> Cplx<R>;

    /// Measurement probabilities `|ψ_i|²` over the full computational basis.
    fn probabilities(&self, state: &Self::State) -> Vec<R>;

    /// Draw `shots` measurement outcomes (basis-state indices) `∝ |ψ_i|²`, via an `O(1)`/shot
    /// alias table. A device backend overrides this to sample on-device (no full copy back).
    fn sample(&self, state: &Self::State, shots: usize, rng: &mut SplitMix64) -> Vec<usize> {
        crate::sample::alias_sample(&self.probabilities(state), shots, rng)
    }

    /// Expectation `⟨ψ| Z_mask |ψ⟩` of the Pauli-Z product on the qubits set in `mask`.
    fn expectation_z(&self, state: &Self::State, mask: usize) -> R {
        crate::sample::expectation_z(&self.probabilities(state), mask)
    }

    /// Copy the full state to host (the only device→host crossing).
    fn download(&self, state: &Self::State) -> StateVector<R>;

    /// Run a whole circuit from `|0…0⟩`. A GPU backend overrides this to batch the whole
    /// circuit into one submission (amortizing launch latency).
    fn execute(&self, circuit: &Circuit<R>) -> Self::State {
        let mut state = self.alloc(circuit.n_qubits());
        self.init_basis(&mut state, 0);
        for op in circuit.ops() {
            self.apply(&mut state, op.gate(), op.qubits());
        }
        state
    }
}
