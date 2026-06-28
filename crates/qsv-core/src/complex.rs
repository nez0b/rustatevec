//! A small, generic complex number for the **API boundary** and gate matrices.
//!
//! Deliberately NOT the storage element of [`StateVector`](crate::state::StateVector),
//! which is Structure-of-Arrays (`re[]`, `im[]`) so SIMD complex multiply needs no
//! de/re-interleave. `Cplx` is for ergonomics where layout doesn't matter.

use crate::real::Real;
use std::ops::{Add, Mul, Sub};

/// Complex number `re + i·im`, generic over the real scalar.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cplx<R: Real> {
    pub re: R,
    pub im: R,
}

impl<R: Real> Cplx<R> {
    #[inline(always)]
    pub fn new(re: R, im: R) -> Self {
        Self { re, im }
    }
    #[inline(always)]
    pub fn zero() -> Self {
        Self {
            re: R::ZERO,
            im: R::ZERO,
        }
    }
    #[inline(always)]
    pub fn one() -> Self {
        Self {
            re: R::ONE,
            im: R::ZERO,
        }
    }
    #[inline(always)]
    pub fn i() -> Self {
        Self {
            re: R::ZERO,
            im: R::ONE,
        }
    }
    /// Construct from `f64` literals (for gate definitions).
    #[inline(always)]
    pub fn from_f64(re: f64, im: f64) -> Self {
        Self {
            re: R::from_f64(re),
            im: R::from_f64(im),
        }
    }
    #[inline(always)]
    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }
    /// `|z|²` (squared magnitude).
    #[inline(always)]
    pub fn norm_sqr(self) -> R {
        self.re * self.re + self.im * self.im
    }
    /// Scale by a real scalar.
    #[inline(always)]
    pub fn scale(self, s: R) -> Self {
        Self {
            re: self.re * s,
            im: self.im * s,
        }
    }
}

impl<R: Real> Add for Cplx<R> {
    type Output = Self;
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        Self {
            re: self.re + o.re,
            im: self.im + o.im,
        }
    }
}

impl<R: Real> Sub for Cplx<R> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        Self {
            re: self.re - o.re,
            im: self.im - o.im,
        }
    }
}

impl<R: Real> Mul for Cplx<R> {
    type Output = Self;
    /// `(a+bi)(c+di) = (ac − bd) + (ad + bc)i`.
    #[inline(always)]
    fn mul(self, o: Self) -> Self {
        Self {
            re: self.re * o.re - self.im * o.im,
            im: self.re * o.im + self.im * o.re,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mul_matches_definition() {
        let a = Cplx::<f64>::new(1.0, 2.0);
        let b = Cplx::<f64>::new(3.0, 4.0);
        // (1+2i)(3+4i) = 3 + 4i + 6i + 8i^2 = -5 + 10i
        assert_eq!(a * b, Cplx::new(-5.0, 10.0));
    }

    #[test]
    fn conj_and_norm() {
        let z = Cplx::<f64>::new(3.0, -4.0);
        assert_eq!(z.conj(), Cplx::new(3.0, 4.0));
        assert_eq!(z.norm_sqr(), 25.0);
    }
}
