//! Circuit generators and a tiny deterministic RNG.
//!
//! Shared by the test suite (differential testing against the oracle) and by the future
//! benchmark harness (QFT / random circuits are *the* standard simulator benchmarks). Kept
//! dependency-free and reproducible: same seed ⇒ same circuit, on every platform.

use crate::circuit::Circuit;

/// [SplitMix64](https://prng.di.unimi.it/splitmix64.c) — a small, fast, well-distributed
/// PRNG. Deterministic given a seed; used to build reproducible random circuits.
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform integer in `0..n` (`n > 0`).
    #[inline]
    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    /// Uniform `f64` in `[0, 1)` (53-bit mantissa).
    #[inline]
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    /// Uniform angle in `[0, 2π)`.
    #[inline]
    pub fn angle(&mut self) -> f64 {
        self.unit() * std::f64::consts::TAU
    }
}

/// GHZ state preparation: `H(0); CX(0, q)` for `q = 1..n`.
pub fn ghz(n: u32) -> Circuit<f64> {
    let mut c = Circuit::new(n);
    c.h(0);
    for q in 1..n {
        c.cx(0, q);
    }
    c
}

/// Quantum Fourier Transform on `n` qubits (Hadamards + controlled phases + final
/// bit-reversal swaps). Applied to `|0…0⟩` it produces the uniform superposition.
pub fn qft(n: u32) -> Circuit<f64> {
    let mut c = Circuit::new(n);
    for j in 0..n {
        c.h(j);
        for k in (j + 1)..n {
            let lambda = std::f64::consts::PI / ((1u64 << (k - j)) as f64);
            c.cphase(k, j, lambda);
        }
    }
    for j in 0..(n / 2) {
        c.swap(j, n - 1 - j);
    }
    c
}

/// A reproducible random circuit of `n_gates` gates over `n_qubits` qubits (`≥ 2`), mixing
/// the full 1- and 2-qubit gate set with random qubits and angles. The workhorse for
/// differential testing and a stand-in for random-circuit-sampling benchmarks.
pub fn random_circuit(n_qubits: u32, n_gates: usize, seed: u64) -> Circuit<f64> {
    assert!(n_qubits >= 2, "random_circuit needs at least 2 qubits");
    let mut rng = SplitMix64::new(seed);
    let mut c = Circuit::new(n_qubits);
    for _ in 0..n_gates {
        let q = rng.below(n_qubits as u64) as u32;
        match rng.below(15) {
            0 => c.h(q),
            1 => c.x(q),
            2 => c.y(q),
            3 => c.z(q),
            4 => c.s(q),
            5 => c.t(q),
            6 => c.sx(q),
            7 => c.rx(q, rng.angle()),
            8 => c.ry(q, rng.angle()),
            9 => c.rz(q, rng.angle()),
            10 => c.phase(q, rng.angle()),
            _ => {
                // Two-qubit gate on a distinct second qubit.
                let mut q2 = rng.below(n_qubits as u64) as u32;
                while q2 == q {
                    q2 = rng.below(n_qubits as u64) as u32;
                }
                match rng.below(4) {
                    0 => c.cx(q, q2),
                    1 => c.cz(q, q2),
                    2 => c.swap(q, q2),
                    _ => c.rzz(q, q2, rng.angle()),
                }
            }
        };
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_is_deterministic() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        let mut r = SplitMix64::new(1);
        for _ in 0..1000 {
            let u = r.unit();
            assert!((0.0..1.0).contains(&u));
        }
    }

    #[test]
    fn generators_have_expected_shape() {
        assert_eq!(ghz(5).ops().len(), 5); // H + 4 CX
        assert_eq!(qft(4).n_qubits(), 4);
        assert_eq!(random_circuit(4, 30, 7).ops().len(), 30);
    }
}
