//! v0.3–v0.5 — `CpuBackend`: the optimized scalar CPU backend.
//!
//! Consolidates three milestones, each toggleable for head-to-head benchmarking:
//!
//! - **v0.3 (bounds-check-free + stack matrices).** The 1-qubit kernel is written as
//!   iterator zips, which the compiler lowers to bounds-check-free code *without* `unsafe`
//!   — the idiomatic-Rust way to get "unchecked" performance. The multi-qubit kernel, whose
//!   scattered indices aren't iterator-shaped, uses `get_unchecked` with a proof. Gate
//!   entries are read once into locals / a flat slice; no per-amplitude matrix bounds checks.
//! - **v0.4 (cache-friendly nested-block kernel).** A 1-qubit gate on qubit `q` is applied by
//!   walking the state in blocks of `2^(q+1)`, splitting each block into its lower and upper
//!   halves, and pairing them with a sequential, prefetcher-friendly stride-1 walk. This is
//!   also the structure SIMD (v0.7) plugs into.
//! - **v0.5 (multithreading).** Above a size threshold, the disjoint blocks run on rayon. The
//!   1-qubit path uses safe `par_chunks_mut` (chunks are provably disjoint); the multi-qubit
//!   path parallelizes over disjoint blocks via raw pointers (see the SAFETY notes).
//!
//! Gated behind the default `parallel` feature; without it, `CpuBackend::parallel()` still
//! exists but runs serially.

#![allow(unsafe_code)] // localized to this module; every block carries a SAFETY justification.

use super::Backend;
use crate::complex::Cplx;
use crate::gate::DenseGate;
use crate::real::Real;
use crate::state::layout::{gather_bits, insert_zero_bits, scatter_bits};
use crate::state::StateVector;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Largest gate arity the stack-buffered multi-qubit kernel supports (`2^6` amplitudes).
const MAX_SUB: usize = 64;
/// Below this many amplitude pairs, threading overhead outweighs the gain → stay serial.
const PARALLEL_MIN_PAIRS: usize = 1 << 12;

/// Optimized CPU backend (milestones v0.3–v0.5).
#[derive(Clone, Copy, Debug)]
pub struct CpuBackend {
    parallel: bool,
}

impl CpuBackend {
    /// Single-threaded (the v0.3/v0.4 kernel).
    pub fn serial() -> Self {
        Self { parallel: false }
    }
    /// Multithreaded above the size threshold (v0.5); serial below it, or if the `parallel`
    /// feature is disabled.
    pub fn parallel() -> Self {
        Self { parallel: true }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::parallel()
    }
}

/// Apply the 2×2 gate to one block's paired lower/upper halves (stride-1, no bounds checks).
#[inline]
fn one_qubit_block<R: Real>(
    re_lo: &mut [R],
    im_lo: &mut [R],
    re_hi: &mut [R],
    im_hi: &mut [R],
    g: [Cplx<R>; 4],
) {
    let it = re_lo
        .iter_mut()
        .zip(im_lo.iter_mut())
        .zip(re_hi.iter_mut().zip(im_hi.iter_mut()));
    for ((rl, il), (rh, ih)) in it {
        let x0 = Cplx::new(*rl, *il);
        let x1 = Cplx::new(*rh, *ih);
        let y0 = g[0] * x0 + g[1] * x1;
        let y1 = g[2] * x0 + g[3] * x1;
        *rl = y0.re;
        *il = y0.im;
        *rh = y1.re;
        *ih = y1.im;
    }
}

/// One contiguous chunk of a diagonal gate: multiply each amplitude by its phase. The phase
/// depends only on the gate-qubit bits of the global index `base + o`, so this is a single
/// sequential pass — no pairing, no stride.
#[inline]
fn diag_chunk<R: Real>(
    base: usize,
    re_c: &mut [R],
    im_c: &mut [R],
    diag: &[Cplx<R>],
    qubits: &[u32],
) {
    for (o, (r, i)) in re_c.iter_mut().zip(im_c.iter_mut()).enumerate() {
        let s = gather_bits(base + o, qubits);
        let (dr, di) = (diag[s].re, diag[s].im);
        let (xr, xi) = (*r, *i);
        *r = dr * xr - di * xi;
        *i = dr * xi + di * xr;
    }
}

impl CpuBackend {
    /// v0.6 — diagonal gate: one sequential pass, half the arithmetic of the pairing kernel.
    fn apply_diagonal<R: Real>(
        &self,
        state: &mut StateVector<R>,
        gate: &DenseGate<R>,
        qubits: &[u32],
    ) {
        let sub = 1usize << qubits.len();
        let mut diag_arr = [Cplx::<R>::zero(); MAX_SUB];
        for (s, d) in diag_arr.iter_mut().enumerate().take(sub) {
            *d = gate.at(s, s);
        }
        let diag = &diag_arr[..sub];
        let dim = state.dim();
        let (re, im) = state.parts_mut();

        let _threaded = self.parallel && dim >= (PARALLEL_MIN_PAIRS << 1);
        #[cfg(feature = "parallel")]
        if _threaded {
            let nthreads = rayon::current_num_threads().max(1);
            let chunk = (dim / (4 * nthreads)).max(1 << 10);
            re.par_chunks_mut(chunk)
                .zip(im.par_chunks_mut(chunk))
                .enumerate()
                .for_each(|(c, (rc, ic))| diag_chunk(c * chunk, rc, ic, diag, qubits));
            return;
        }

        diag_chunk(0, re, im, diag, qubits);
    }

    fn apply_1q<R: Real>(&self, state: &mut StateVector<R>, gate: &DenseGate<R>, q: u32) {
        let g = [gate.at(0, 0), gate.at(0, 1), gate.at(1, 0), gate.at(1, 1)];
        let half = 1usize << q;
        let block = half << 1;
        let dim = state.dim();
        let (re, im) = state.parts_mut();

        let _threaded = self.parallel && (dim >> 1) >= PARALLEL_MIN_PAIRS;
        #[cfg(feature = "parallel")]
        if _threaded {
            re.par_chunks_mut(block)
                .zip(im.par_chunks_mut(block))
                .for_each(|(re_blk, im_blk)| {
                    let (re_lo, re_hi) = re_blk.split_at_mut(half);
                    let (im_lo, im_hi) = im_blk.split_at_mut(half);
                    one_qubit_block(re_lo, im_lo, re_hi, im_hi, g);
                });
            return;
        }

        re.chunks_mut(block)
            .zip(im.chunks_mut(block))
            .for_each(|(re_blk, im_blk)| {
                let (re_lo, re_hi) = re_blk.split_at_mut(half);
                let (im_lo, im_hi) = im_blk.split_at_mut(half);
                one_qubit_block(re_lo, im_lo, re_hi, im_hi, g);
            });
    }

    fn apply_mq<R: Real>(&self, state: &mut StateVector<R>, gate: &DenseGate<R>, qubits: &[u32]) {
        let m = qubits.len();
        let sub = 1usize << m;
        assert!(sub <= MAX_SUB, "CpuBackend supports up to 6-qubit gates");

        let mut sorted = [0u32; 6];
        sorted[..m].copy_from_slice(qubits);
        sorted[..m].sort_unstable();

        let dim = state.dim();
        let blocks = dim >> m;
        let g = gate.row_major();

        let _threaded = self.parallel && (dim >> 1) >= PARALLEL_MIN_PAIRS && blocks > 1;
        #[cfg(feature = "parallel")]
        if _threaded {
            let (re, im) = state.parts_mut();
            let re_ptr = SyncPtr(re.as_mut_ptr());
            let im_ptr = SyncPtr(im.as_mut_ptr());
            (0..blocks).into_par_iter().for_each(|o| {
                let base = insert_zero_bits(o, &sorted[..m]);
                // SAFETY: distinct blocks `o` map to disjoint sets of indices `idx < dim`,
                // so no two parallel iterations touch the same amplitude.
                unsafe { mq_block(re_ptr.get(), im_ptr.get(), g, qubits, base, sub) };
            });
            return;
        }

        let (re, im) = state.parts_mut();
        let re_ptr = re.as_mut_ptr();
        let im_ptr = im.as_mut_ptr();
        for o in 0..blocks {
            let base = insert_zero_bits(o, &sorted[..m]);
            // SAFETY: serial; `idx < dim` for every gathered/scattered index.
            unsafe { mq_block(re_ptr, im_ptr, g, qubits, base, sub) };
        }
    }
}

/// One block of a multi-qubit gate: gather `2^m` sub-amplitudes, apply the matrix, scatter
/// back — all via raw pointers with `idx < dim`.
///
/// # Safety
/// `re`/`im` must point to the (same) `dim`-length amplitude arrays, and every index produced
/// by `base | scatter_bits(.., qubits)` must be `< dim` and disjoint from indices produced for
/// other `base` values (guaranteed by the block tiling).
#[inline]
unsafe fn mq_block<R: Real>(
    re: *mut R,
    im: *mut R,
    g: &[Cplx<R>],
    qubits: &[u32],
    base: usize,
    sub: usize,
) {
    let mut amp = [Cplx::<R>::zero(); MAX_SUB];
    for (s, slot) in amp.iter_mut().enumerate().take(sub) {
        let idx = base | scatter_bits(s, qubits);
        *slot = Cplx::new(*re.add(idx), *im.add(idx));
    }
    for r in 0..sub {
        let mut acc = Cplx::<R>::zero();
        let row = r * sub;
        for (cc, &a) in amp.iter().enumerate().take(sub) {
            acc = acc + *g.get_unchecked(row + cc) * a;
        }
        let idx = base | scatter_bits(r, qubits);
        *re.add(idx) = acc.re;
        *im.add(idx) = acc.im;
    }
}

/// A raw pointer marked `Send + Sync` so rayon can hand disjoint slices of the amplitude
/// arrays to worker threads. Sound only because callers guarantee disjoint index sets.
#[cfg(feature = "parallel")]
#[derive(Clone, Copy)]
struct SyncPtr<T>(*mut T);
#[cfg(feature = "parallel")]
impl<T> SyncPtr<T> {
    /// Take by value (`self`) so closures capture the whole wrapper, not the bare field —
    /// Rust 2021 disjoint capture would otherwise grab the non-`Send` `*mut T` directly.
    #[inline]
    fn get(self) -> *mut T {
        self.0
    }
}
#[cfg(feature = "parallel")]
// SAFETY: access through this pointer is partitioned into disjoint index ranges across
// threads by the caller, so no data race occurs.
unsafe impl<T> Send for SyncPtr<T> {}
#[cfg(feature = "parallel")]
unsafe impl<T> Sync for SyncPtr<T> {}

impl<R: Real> Backend<R> for CpuBackend {
    type State = StateVector<R>;

    fn alloc(&self, n_qubits: u32) -> StateVector<R> {
        StateVector::zeros(n_qubits)
    }

    fn init_basis(&self, state: &mut StateVector<R>, basis: usize) {
        *state = StateVector::basis(state.n_qubits(), basis);
    }

    fn apply(&self, state: &mut StateVector<R>, gate: &DenseGate<R>, qubits: &[u32]) {
        debug_assert_eq!(gate.n_qubits() as usize, qubits.len());
        if gate.is_diagonal() {
            self.apply_diagonal(state, gate, qubits);
        } else if qubits.len() == 1 {
            self.apply_1q(state, gate, qubits[0]);
        } else {
            self.apply_mq(state, gate, qubits);
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
