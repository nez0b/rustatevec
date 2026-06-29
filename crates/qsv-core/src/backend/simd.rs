//! v0.7 — `SimdBackend`: portable-SIMD complex multiply for the 1-qubit kernel.
//!
//! Concrete `f64` (the default precision). It **delegates everything to [`CpuBackend`]**
//! except non-diagonal single-qubit gates, which it applies with a `wide::f64x4` kernel over
//! the SoA nested-block halves: a SIMD load of `re_lo`/`im_lo`/`re_hi`/`im_hi` each yields a
//! register of like-typed values, so the complex multiply is a straight broadcast-FMA chain
//! with **no lane shuffles** — the payoff of the [SoA layout](../../design/overview.md).
//!
//! Honest scope: SIMD vectorizes the *contiguous halves* of each block, so it kicks in for
//! target qubit `q ≥ 2` (half ≥ 4 lanes); low-`q` gates fall back to the scalar remainder
//! (their stride-1 win needs the permute-within-register trick, a later/hand-intrinsics step).
//! And since SIMD speeds up *arithmetic*, its benefit is largest in the compute-bound regime;
//! at large `N` the kernel is bandwidth-bound (see the benchmarking chapter).
//!
//! Gated behind the default `simd` feature; without it, falls back to `CpuBackend`.

use super::cpu::CpuBackend;
use super::Backend;
use crate::complex::Cplx;
use crate::gate::DenseGate;
use crate::state::StateVector;

#[cfg(all(feature = "simd", feature = "parallel"))]
use rayon::prelude::*;

/// SIMD CPU backend (milestone v0.7), `f64` only.
#[derive(Clone, Copy, Debug)]
pub struct SimdBackend {
    parallel: bool,
}

impl SimdBackend {
    pub fn serial() -> Self {
        Self { parallel: false }
    }
    pub fn parallel() -> Self {
        Self { parallel: true }
    }
    /// The scalar backend we delegate to for diagonal / multi-qubit gates.
    fn inner(&self) -> CpuBackend {
        if self.parallel {
            CpuBackend::parallel()
        } else {
            CpuBackend::serial()
        }
    }
}

impl Default for SimdBackend {
    fn default() -> Self {
        Self::parallel()
    }
}

/// Pairs below this many amplitude pairs run serially even when `parallel` is set.
#[cfg(all(feature = "simd", feature = "parallel"))]
const PARALLEL_MIN_PAIRS: usize = 1 << 12;

/// Apply the 2×2 gate to one block: SIMD over the contiguous halves, scalar remainder.
///
/// Two implementations share this signature, picked by feature:
/// - **`nightly-simd`**: `std::simd::f64x8` (512-bit, AVX-512) with real vector loads
///   (`Simd::from_slice`). This is the wide path the Intel box is built for.
/// - default `simd`: the stable `wide::f64x4` (256-bit) baseline (v0.7).
#[cfg(all(feature = "simd", not(feature = "nightly-simd")))]
fn simd_block(rb: &mut [f64], ib: &mut [f64], half: usize, g: [Cplx<f64>; 4]) {
    use wide::f64x4;

    let (re_lo, re_hi) = rb.split_at_mut(half);
    let (im_lo, im_hi) = ib.split_at_mut(half);

    let (g00r, g00i) = (f64x4::splat(g[0].re), f64x4::splat(g[0].im));
    let (g01r, g01i) = (f64x4::splat(g[1].re), f64x4::splat(g[1].im));
    let (g10r, g10i) = (f64x4::splat(g[2].re), f64x4::splat(g[2].im));
    let (g11r, g11i) = (f64x4::splat(g[3].re), f64x4::splat(g[3].im));

    let mut rl = re_lo.chunks_exact_mut(4);
    let mut il = im_lo.chunks_exact_mut(4);
    let mut rh = re_hi.chunks_exact_mut(4);
    let mut ih = im_hi.chunks_exact_mut(4);

    for (((rlc, ilc), rhc), ihc) in rl
        .by_ref()
        .zip(il.by_ref())
        .zip(rh.by_ref())
        .zip(ih.by_ref())
    {
        let x0r = f64x4::from([rlc[0], rlc[1], rlc[2], rlc[3]]);
        let x0i = f64x4::from([ilc[0], ilc[1], ilc[2], ilc[3]]);
        let x1r = f64x4::from([rhc[0], rhc[1], rhc[2], rhc[3]]);
        let x1i = f64x4::from([ihc[0], ihc[1], ihc[2], ihc[3]]);

        let y0r = g00r * x0r - g00i * x0i + g01r * x1r - g01i * x1i;
        let y0i = g00r * x0i + g00i * x0r + g01r * x1i + g01i * x1r;
        let y1r = g10r * x0r - g10i * x0i + g11r * x1r - g11i * x1i;
        let y1i = g10r * x0i + g10i * x0r + g11r * x1i + g11i * x1r;

        rlc.copy_from_slice(&y0r.to_array());
        ilc.copy_from_slice(&y0i.to_array());
        rhc.copy_from_slice(&y1r.to_array());
        ihc.copy_from_slice(&y1i.to_array());
    }

    // Scalar tail (length < 4, e.g. low-q blocks).
    let (rl, il, rh, ih) = (
        rl.into_remainder(),
        il.into_remainder(),
        rh.into_remainder(),
        ih.into_remainder(),
    );
    for (((r0, i0), r1), i1) in rl
        .iter_mut()
        .zip(il.iter_mut())
        .zip(rh.iter_mut())
        .zip(ih.iter_mut())
    {
        let (x0r, x0i, x1r, x1i) = (*r0, *i0, *r1, *i1);
        *r0 = g[0].re * x0r - g[0].im * x0i + g[1].re * x1r - g[1].im * x1i;
        *i0 = g[0].re * x0i + g[0].im * x0r + g[1].re * x1i + g[1].im * x1r;
        *r1 = g[2].re * x0r - g[2].im * x0i + g[3].re * x1r - g[3].im * x1i;
        *i1 = g[2].re * x0i + g[2].im * x0r + g[3].re * x1i + g[3].im * x1r;
    }
}

/// `nightly-simd` — the AVX-512 `std::simd::f64x8` kernel (512-bit lanes, real vector loads).
///
/// Identical math to the `wide` version but 8-wide and using `Simd::from_slice` (a true aligned
/// vector load) instead of the per-element array gather, so it competes fairly with LLVM's
/// auto-vectorized scalar kernel. SoA layout means the complex multiply is pure broadcast-FMA
/// with no lane shuffles.
#[cfg(feature = "nightly-simd")]
fn simd_block(rb: &mut [f64], ib: &mut [f64], half: usize, g: [Cplx<f64>; 4]) {
    use std::simd::Simd;
    type V = Simd<f64, 8>;
    const L: usize = 8;

    let (re_lo, re_hi) = rb.split_at_mut(half);
    let (im_lo, im_hi) = ib.split_at_mut(half);

    let (g00r, g00i) = (V::splat(g[0].re), V::splat(g[0].im));
    let (g01r, g01i) = (V::splat(g[1].re), V::splat(g[1].im));
    let (g10r, g10i) = (V::splat(g[2].re), V::splat(g[2].im));
    let (g11r, g11i) = (V::splat(g[3].re), V::splat(g[3].im));

    let mut rl = re_lo.chunks_exact_mut(L);
    let mut il = im_lo.chunks_exact_mut(L);
    let mut rh = re_hi.chunks_exact_mut(L);
    let mut ih = im_hi.chunks_exact_mut(L);

    for (((rlc, ilc), rhc), ihc) in rl
        .by_ref()
        .zip(il.by_ref())
        .zip(rh.by_ref())
        .zip(ih.by_ref())
    {
        let x0r = V::from_slice(rlc);
        let x0i = V::from_slice(ilc);
        let x1r = V::from_slice(rhc);
        let x1i = V::from_slice(ihc);

        let y0r = g00r * x0r - g00i * x0i + g01r * x1r - g01i * x1i;
        let y0i = g00r * x0i + g00i * x0r + g01r * x1i + g01i * x1r;
        let y1r = g10r * x0r - g10i * x0i + g11r * x1r - g11i * x1i;
        let y1i = g10r * x0i + g10i * x0r + g11r * x1i + g11i * x1r;

        rlc.copy_from_slice(y0r.as_array());
        ilc.copy_from_slice(y0i.as_array());
        rhc.copy_from_slice(y1r.as_array());
        ihc.copy_from_slice(y1i.as_array());
    }

    // Scalar tail (length < 8, e.g. low-q blocks).
    let (rl, il, rh, ih) = (
        rl.into_remainder(),
        il.into_remainder(),
        rh.into_remainder(),
        ih.into_remainder(),
    );
    for (((r0, i0), r1), i1) in rl
        .iter_mut()
        .zip(il.iter_mut())
        .zip(rh.iter_mut())
        .zip(ih.iter_mut())
    {
        let (x0r, x0i, x1r, x1i) = (*r0, *i0, *r1, *i1);
        *r0 = g[0].re * x0r - g[0].im * x0i + g[1].re * x1r - g[1].im * x1i;
        *i0 = g[0].re * x0i + g[0].im * x0r + g[1].re * x1i + g[1].im * x1r;
        *r1 = g[2].re * x0r - g[2].im * x0i + g[3].re * x1r - g[3].im * x1i;
        *i1 = g[2].re * x0i + g[2].im * x0r + g[3].re * x1i + g[3].im * x1r;
    }
}

impl SimdBackend {
    #[cfg(feature = "simd")]
    fn apply_1q_simd(&self, state: &mut StateVector<f64>, gate: &DenseGate<f64>, q: u32) {
        let g = [gate.at(0, 0), gate.at(0, 1), gate.at(1, 0), gate.at(1, 1)];
        let half = 1usize << q;
        let block = half << 1;
        let dim = state.dim();
        let (re, im) = state.parts_mut();

        #[cfg(feature = "parallel")]
        if self.parallel && (dim >> 1) >= PARALLEL_MIN_PAIRS {
            re.par_chunks_mut(block)
                .zip(im.par_chunks_mut(block))
                .for_each(|(rb, ib)| simd_block(rb, ib, half, g));
            return;
        }

        re.chunks_mut(block)
            .zip(im.chunks_mut(block))
            .for_each(|(rb, ib)| simd_block(rb, ib, half, g));
    }
}

impl Backend<f64> for SimdBackend {
    type State = StateVector<f64>;

    fn alloc(&self, n_qubits: u32) -> StateVector<f64> {
        StateVector::zeros(n_qubits)
    }

    fn init_basis(&self, state: &mut StateVector<f64>, basis: usize) {
        *state = StateVector::basis(state.n_qubits(), basis);
    }

    fn apply(&self, state: &mut StateVector<f64>, gate: &DenseGate<f64>, qubits: &[u32]) {
        // SIMD only for non-diagonal single-qubit gates; everything else goes to CpuBackend
        // (which has the diagonal fast path and the multi-qubit / threaded kernels).
        if qubits.len() == 1 && !gate.is_diagonal() {
            #[cfg(feature = "simd")]
            {
                self.apply_1q_simd(state, gate, qubits[0]);
                return;
            }
        }
        self.inner().apply(state, gate, qubits);
    }

    fn amplitude(&self, state: &StateVector<f64>, index: usize) -> Cplx<f64> {
        state.amplitude(index)
    }

    fn probabilities(&self, state: &StateVector<f64>) -> Vec<f64> {
        state.probabilities()
    }

    fn download(&self, state: &StateVector<f64>) -> StateVector<f64> {
        state.clone()
    }
}
