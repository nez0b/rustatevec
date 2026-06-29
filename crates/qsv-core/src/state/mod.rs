//! The statevector itself — **Structure-of-Arrays** storage.
//!
//! `re[]` and `im[]` are separate arrays (not interleaved `Complex`). For the
//! bandwidth-bound complex-multiply kernel this is the right layout: a SIMD load of `re`
//! and a SIMD load of `im` each yield a register of like-typed values, so the multiply is
//! a straight broadcast-FMA chain with **no lane shuffles** (NEON `ld2` / x86 `unpck`).
//! All amplitude updates are **in place** — at 30 qubits the vector is 16 GB and there is
//! no room for a second buffer.

pub mod layout;

use crate::complex::Cplx;
use crate::real::Real;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Below this many amplitudes the buffer is small (≤ a few MB), NUMA effects are negligible,
/// and the parallel first-touch pass isn't worth its overhead — allocate serially.
#[cfg(feature = "parallel")]
const FIRST_TOUCH_MIN: usize = 1 << 18; // 256k amps = 2 MB per component buffer

/// A pure-state quantum register of `n_qubits` qubits: `2^n` complex amplitudes in SoA.
#[derive(Clone, Debug, PartialEq)]
pub struct StateVector<R: Real> {
    re: Vec<R>,
    im: Vec<R>,
    n_qubits: u32,
}

impl<R: Real> StateVector<R> {
    /// All-zero buffer (not a valid normalized state on its own).
    pub fn zeros(n_qubits: u32) -> Self {
        let dim = 1usize << n_qubits;
        let mut s = Self {
            // `vec![ZERO; dim]` lowers to `alloc_zeroed` → for large `dim` the OS hands back
            // lazily-zeroed pages that are *not yet faulted in*. `first_touch` faults them in
            // parallel so each page lands on the NUMA node of the worker that will own it.
            re: vec![R::ZERO; dim],
            im: vec![R::ZERO; dim],
            n_qubits,
        };
        s.first_touch();
        s
    }

    /// NUMA-aware **first touch**: fault the freshly-allocated (lazily-zeroed) pages in parallel
    /// with the same contiguous decomposition the kernels use, so each page is first-touched —
    /// and thus placed — on the node of the worker that will later process that region. Serial
    /// allocation first-touches every page on the allocating thread's node, capping large-N
    /// throughput at one socket's memory bandwidth (measured on 2× Xeon 6526Y: ~0.33 G/s at 64
    /// threads unpinned vs ~2.3 G/s when local). No-op without `parallel` or for small states.
    #[cfg(feature = "parallel")]
    fn first_touch(&mut self) {
        let dim = self.re.len();
        if dim < FIRST_TOUCH_MIN {
            return;
        }
        let chunk = dim.div_ceil(rayon::current_num_threads().max(1));
        self.re.par_chunks_mut(chunk).for_each(|c| c.fill(R::ZERO));
        self.im.par_chunks_mut(chunk).for_each(|c| c.fill(R::ZERO));
    }

    #[cfg(not(feature = "parallel"))]
    #[inline]
    fn first_touch(&mut self) {}

    /// Computational basis state `|index⟩`.
    pub fn basis(n_qubits: u32, index: usize) -> Self {
        let mut s = Self::zeros(n_qubits);
        s.re[index] = R::ONE;
        s
    }

    #[inline(always)]
    pub fn n_qubits(&self) -> u32 {
        self.n_qubits
    }

    /// Number of amplitudes, `2^n`.
    #[inline(always)]
    pub fn dim(&self) -> usize {
        self.re.len()
    }

    #[inline(always)]
    pub fn amplitude(&self, i: usize) -> Cplx<R> {
        Cplx::new(self.re[i], self.im[i])
    }

    #[inline(always)]
    pub fn set(&mut self, i: usize, c: Cplx<R>) {
        self.re[i] = c.re;
        self.im[i] = c.im;
    }

    #[inline(always)]
    pub fn re(&self) -> &[R] {
        &self.re
    }
    #[inline(always)]
    pub fn im(&self) -> &[R] {
        &self.im
    }
    /// Mutable split borrow of the two component arrays — the kernel entry point.
    #[inline(always)]
    pub fn parts_mut(&mut self) -> (&mut [R], &mut [R]) {
        (&mut self.re, &mut self.im)
    }

    /// `⟨ψ|ψ⟩` — should be 1 for a normalized state.
    pub fn norm_sqr(&self) -> R {
        let mut s = R::ZERO;
        for i in 0..self.dim() {
            s = s + self.re[i] * self.re[i] + self.im[i] * self.im[i];
        }
        s
    }

    /// Scale to unit norm.
    pub fn normalize(&mut self) {
        let inv = R::ONE / self.norm_sqr().sqrt();
        for i in 0..self.dim() {
            self.re[i] = self.re[i] * inv;
            self.im[i] = self.im[i] * inv;
        }
    }

    /// Per-amplitude measurement probabilities `|ψ_i|²`.
    pub fn probabilities(&self) -> Vec<R> {
        (0..self.dim())
            .map(|i| self.re[i] * self.re[i] + self.im[i] * self.im[i])
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basis_state_is_normalized_one_hot() {
        let s = StateVector::<f64>::basis(3, 5);
        assert_eq!(s.dim(), 8);
        assert_eq!(s.amplitude(5), Cplx::one());
        assert_eq!(s.norm_sqr(), 1.0);
        assert_eq!(s.probabilities().iter().filter(|&&p| p > 0.0).count(), 1);
    }

    #[test]
    fn normalize_makes_unit_norm() {
        let mut s = StateVector::<f64>::zeros(1);
        s.set(0, Cplx::new(3.0, 0.0));
        s.set(1, Cplx::new(0.0, 4.0));
        s.normalize();
        assert!((s.norm_sqr() - 1.0).abs() < 1e-12);
    }
}
