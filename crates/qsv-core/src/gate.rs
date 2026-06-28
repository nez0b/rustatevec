//! Gate representations and a standard gate library.
//!
//! v0.0 uses a single uniform representation, [`DenseGate`] (a `2^m × 2^m` complex matrix),
//! which the oracle backend applies directly. Later milestones add specialized
//! representations (const-generic `Mat2`/`Mat4`, diagonal-only, controlled) for the
//! optimized kernels; the standard-library constructors here stay the source of truth.
//!
//! ## Qubit/index convention (used everywhere)
//! A gate acts on an ordered list of qubits `qs`. Bit `j` (LSB = 0) of the gate's internal
//! row/column index corresponds to global qubit `qs[j]`. Thus for a 2-qubit gate on
//! `qs = [a, b]`, internal index `= bit_a + 2·bit_b`. Controlled gates put the **control
//! first**: `cx(c, t)` ⇒ `qs = [c, t]`, internal index `= bit_c + 2·bit_t`.

use crate::complex::Cplx;
use crate::real::Real;

/// A dense `2^n_qubits × 2^n_qubits` unitary, stored row-major.
#[derive(Clone, Debug)]
pub struct DenseGate<R: Real> {
    data: Vec<Cplx<R>>,
    n_qubits: u32,
}

impl<R: Real> DenseGate<R> {
    /// Build from a row-major `dim × dim` matrix (`dim = 2^n_qubits`).
    pub fn new(n_qubits: u32, data: Vec<Cplx<R>>) -> Self {
        let dim = 1usize << n_qubits;
        assert_eq!(data.len(), dim * dim, "gate matrix must be dim×dim");
        Self { data, n_qubits }
    }

    #[inline(always)]
    pub fn n_qubits(&self) -> u32 {
        self.n_qubits
    }
    #[inline(always)]
    pub fn dim(&self) -> usize {
        1usize << self.n_qubits
    }
    #[inline(always)]
    pub fn at(&self, row: usize, col: usize) -> Cplx<R> {
        self.data[row * self.dim() + col]
    }
}

/// Wrap a 1-qubit gate `u` as a 2-qubit controlled gate (`qs = [control, target]`).
/// The control-0 subspace is identity; the control-1 subspace applies `u` to the target.
pub fn controlled_1q<R: Real>(u: &DenseGate<R>) -> DenseGate<R> {
    assert_eq!(u.n_qubits(), 1);
    let mut data = vec![Cplx::<R>::zero(); 16];
    // control == 0  (even internal indices 0, 2): identity on the target
    data[0] = Cplx::one(); // |c=0,t=0>
    data[2 * 4 + 2] = Cplx::one(); // |c=0,t=1>
                                   // control == 1  (odd internal indices 1, 3): apply u to the target
    for t_out in 0..2 {
        for t_in in 0..2 {
            let row = 1 + 2 * t_out;
            let col = 1 + 2 * t_in;
            data[row * 4 + col] = u.at(t_out, t_in);
        }
    }
    DenseGate::new(2, data)
}

#[inline]
fn c<R: Real>(re: f64, im: f64) -> Cplx<R> {
    Cplx::from_f64(re, im)
}

// ---- Single-qubit, non-parametric ----------------------------------------------------

pub fn h<R: Real>() -> DenseGate<R> {
    let s = std::f64::consts::FRAC_1_SQRT_2;
    DenseGate::new(1, vec![c(s, 0.0), c(s, 0.0), c(s, 0.0), c(-s, 0.0)])
}
pub fn x<R: Real>() -> DenseGate<R> {
    DenseGate::new(1, vec![c(0.0, 0.0), c(1.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)])
}
pub fn y<R: Real>() -> DenseGate<R> {
    DenseGate::new(1, vec![c(0.0, 0.0), c(0.0, -1.0), c(0.0, 1.0), c(0.0, 0.0)])
}
pub fn z<R: Real>() -> DenseGate<R> {
    DenseGate::new(1, vec![c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(-1.0, 0.0)])
}
pub fn s_gate<R: Real>() -> DenseGate<R> {
    DenseGate::new(1, vec![c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 1.0)])
}
pub fn t_gate<R: Real>() -> DenseGate<R> {
    let r = std::f64::consts::FRAC_1_SQRT_2;
    DenseGate::new(1, vec![c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(r, r)])
}
/// √X (the "SX" gate).
pub fn sx<R: Real>() -> DenseGate<R> {
    DenseGate::new(
        1,
        vec![c(0.5, 0.5), c(0.5, -0.5), c(0.5, -0.5), c(0.5, 0.5)],
    )
}

// ---- Single-qubit, parametric --------------------------------------------------------

pub fn rx<R: Real>(theta: f64) -> DenseGate<R> {
    let (s, co) = (theta * 0.5).sin_cos();
    DenseGate::new(1, vec![c(co, 0.0), c(0.0, -s), c(0.0, -s), c(co, 0.0)])
}
pub fn ry<R: Real>(theta: f64) -> DenseGate<R> {
    let (s, co) = (theta * 0.5).sin_cos();
    DenseGate::new(1, vec![c(co, 0.0), c(-s, 0.0), c(s, 0.0), c(co, 0.0)])
}
pub fn rz<R: Real>(theta: f64) -> DenseGate<R> {
    let (s, co) = (theta * 0.5).sin_cos();
    // diag(e^{-iθ/2}, e^{+iθ/2})
    DenseGate::new(1, vec![c(co, -s), c(0.0, 0.0), c(0.0, 0.0), c(co, s)])
}
/// Phase gate `diag(1, e^{iλ})`.
pub fn phase<R: Real>(lambda: f64) -> DenseGate<R> {
    let (s, co) = lambda.sin_cos();
    DenseGate::new(1, vec![c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(co, s)])
}

// ---- Two-qubit -----------------------------------------------------------------------

/// CNOT with `qs = [control, target]`.
pub fn cx<R: Real>() -> DenseGate<R> {
    controlled_1q(&x())
}
/// Controlled-Z (symmetric; `qs = [a, b]`).
pub fn cz<R: Real>() -> DenseGate<R> {
    controlled_1q(&z())
}
/// Controlled phase `diag(1, 1, 1, e^{iλ})` on `qs = [control, target]` (symmetric).
pub fn cphase<R: Real>(lambda: f64) -> DenseGate<R> {
    controlled_1q(&phase(lambda))
}
/// SWAP on `qs = [a, b]`.
pub fn swap<R: Real>() -> DenseGate<R> {
    let mut data = vec![Cplx::<R>::zero(); 16];
    // Nonzeros at (row,col) = (0,0),(1,2),(2,1),(3,3) → flat indices 0,6,9,15.
    data[0] = Cplx::one();
    data[6] = Cplx::one();
    data[9] = Cplx::one();
    data[15] = Cplx::one();
    DenseGate::new(2, data)
}
/// `RZZ(θ) = exp(-i θ/2 · Z⊗Z)` on `qs = [a, b]` (diagonal). Used by QAOA.
pub fn rzz<R: Real>(theta: f64) -> DenseGate<R> {
    let mut data = vec![Cplx::<R>::zero(); 16];
    // Z⊗Z eigenvalue for internal index (bit_b bit_a) is (-1)^{bit_a + bit_b}.
    for idx in 0..4usize {
        let parity = (idx & 1) ^ ((idx >> 1) & 1); // bit_a XOR bit_b
        let zz = if parity == 0 { 1.0 } else { -1.0 };
        let (s, co) = (-0.5 * theta * zz).sin_cos();
        data[idx * 4 + idx] = c(co, s);
    }
    DenseGate::new(2, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: Cplx<f64>, re: f64, im: f64) -> bool {
        (a.re - re).abs() < 1e-12 && (a.im - im).abs() < 1e-12
    }

    #[test]
    fn cx_permutes_basis_correctly() {
        // Internal index = bit_c + 2*bit_t. CX flips target iff control==1:
        // 0(c0t0)->0, 1(c1t0)->3(c1t1), 2(c0t1)->2, 3(c1t1)->1(c1t0).
        let g = cx::<f64>();
        let expect = [0usize, 3, 2, 1];
        for (col, &exp_row) in expect.iter().enumerate() {
            for row in 0..4 {
                let want = if row == exp_row { 1.0 } else { 0.0 };
                assert!(approx_eq(g.at(row, col), want, 0.0), "row {row} col {col}");
            }
        }
    }

    #[test]
    fn x_is_self_inverse_matrix() {
        // (X·X) = I, checked as matrix product.
        let g = x::<f64>();
        for i in 0..2 {
            for j in 0..2 {
                let mut acc = Cplx::<f64>::zero();
                for k in 0..2 {
                    acc = acc + g.at(i, k) * g.at(k, j);
                }
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(approx_eq(acc, want, 0.0));
            }
        }
    }

    #[test]
    fn rz_is_diagonal_unit_modulus() {
        let g = rz::<f64>(0.7);
        assert!(approx_eq(g.at(0, 1), 0.0, 0.0));
        assert!(approx_eq(g.at(1, 0), 0.0, 0.0));
        assert!((g.at(0, 0).norm_sqr() - 1.0).abs() < 1e-12);
        assert!((g.at(1, 1).norm_sqr() - 1.0).abs() < 1e-12);
    }
}
