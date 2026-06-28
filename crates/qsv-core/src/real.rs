//! The `Real` scalar abstraction: one generic kernel codebase over `f64` and `f32`.
//!
//! `f64` is the default (precision; 30-qubit local ceiling). `f32` halves bytes-moved and
//! doubles SIMD lanes — a benchmark axis we exercise later. Keeping kernels generic over
//! `Real` (never `dyn`) lets the compiler monomorphize each into tight, scalar-specialized code.

use std::ops::{Add, Div, Mul, Neg, Sub};

/// A real floating-point scalar usable as a statevector amplitude component.
pub trait Real:
    Copy
    + Clone
    + Send
    + Sync
    + 'static
    + PartialEq
    + PartialOrd
    + std::fmt::Debug
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    const ZERO: Self;
    const ONE: Self;

    fn from_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;
    fn sqrt(self) -> Self;
    fn abs(self) -> Self;
    /// Fused multiply-add `self * b + c` (single rounding, hardware FMA).
    fn mul_add(self, b: Self, c: Self) -> Self;
    /// `(sin(self), cos(self))` — for parametric gate generation.
    fn sin_cos(self) -> (Self, Self);
}

impl Real for f64 {
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;

    #[inline(always)]
    fn from_f64(x: f64) -> f64 {
        x
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline(always)]
    fn sqrt(self) -> f64 {
        f64::sqrt(self)
    }
    #[inline(always)]
    fn abs(self) -> f64 {
        f64::abs(self)
    }
    #[inline(always)]
    fn mul_add(self, b: f64, c: f64) -> f64 {
        f64::mul_add(self, b, c)
    }
    #[inline(always)]
    fn sin_cos(self) -> (f64, f64) {
        f64::sin_cos(self)
    }
}

impl Real for f32 {
    const ZERO: f32 = 0.0;
    const ONE: f32 = 1.0;

    #[inline(always)]
    fn from_f64(x: f64) -> f32 {
        x as f32
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline(always)]
    fn sqrt(self) -> f32 {
        f32::sqrt(self)
    }
    #[inline(always)]
    fn abs(self) -> f32 {
        f32::abs(self)
    }
    #[inline(always)]
    fn mul_add(self, b: f32, c: f32) -> f32 {
        f32::mul_add(self, b, c)
    }
    #[inline(always)]
    fn sin_cos(self) -> (f32, f32) {
        f32::sin_cos(self)
    }
}
