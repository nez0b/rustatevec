//! Bit-index arithmetic — the addressing foundation of the optimized kernels.
//!
//! [`insert_zero_bit`] is the **universal core trick** every production statevector simulator
//! uses (under various names — insert-zero-bit, index0, bit-deposit, controlled iteration).
//! Applying a 1-qubit gate to qubit `q` pairs up amplitudes whose indices differ only in bit
//! `q`; we enumerate the `2^(N-1)` pairs by looping `i` and reconstructing the partner indices
//! on the fly — no reshape/permute copy.
//!
//! On x86-64 with the `bmi2` feature, the bit-gather/scatter/insert generalizations map to single
//! `PEXT`/`PDEP` instructions (see [`insert_zero_bits`] and the `*_bmi2` helpers): 6–12× faster in
//! isolation, but the kernels are memory-bandwidth-bound, so the end-to-end win is ~1× — these
//! functions compute *addresses*, not memory traffic.
//!
//! These helpers are pure `usize` math, exhaustively unit-tested here, and reused by every
//! optimized backend (v0.2+). The v0.0 oracle uses the gather/scatter helpers instead.

/// Insert a `0` bit at position `bit`, shifting higher bits up by one.
///
/// `i = …b_{q} b_{q-1} … b_0`  →  `…b_{q} 0 b_{q-1} … b_0`. The result is the index of the
/// `|…0…⟩` partner; XOR with `1<<bit` gives the `|…1…⟩` partner ([`flip_bit`]).
#[inline(always)]
pub fn insert_zero_bit(index: usize, bit: u32) -> usize {
    let left = (index >> bit) << bit; // bits ≥ bit
    let right = index - left; // bits < bit  (== index & ((1<<bit)-1))
    (left << 1) | right
}

/// Toggle `bit` of `index` (the `|0⟩↔|1⟩` partner on one qubit).
#[inline(always)]
pub fn flip_bit(index: usize, bit: u32) -> usize {
    index ^ (1usize << bit)
}

/// Extract the value (0/1) of `bit` in `index`.
#[inline(always)]
pub fn extract_bit(index: usize, bit: u32) -> usize {
    (index >> bit) & 1
}

/// Gather the bits of `index` at positions `qs` into a compact sub-index:
/// bit `j` of the result is bit `qs[j]` of `index`.
#[inline]
pub fn gather_bits(index: usize, qs: &[u32]) -> usize {
    let mut g = 0usize;
    for (j, &q) in qs.iter().enumerate() {
        g |= ((index >> q) & 1) << j;
    }
    g
}

/// Scatter a compact sub-index back to full-index positions `qs`:
/// bit `j` of `sub` becomes bit `qs[j]` of the result (all other bits 0).
#[inline]
pub fn scatter_bits(sub: usize, qs: &[u32]) -> usize {
    let mut x = 0usize;
    for (j, &q) in qs.iter().enumerate() {
        x |= ((sub >> j) & 1) << q;
    }
    x
}

/// Insert a `0` bit at every position in `sorted_bits`, which **must be strictly ascending**.
///
/// Produces a "base" index whose `sorted_bits` positions are all 0 and whose remaining bits
/// are `index`'s bits in order. As `index` ranges `0..2^(N-m)` (m = `sorted_bits.len()`),
/// the result ranges over exactly the `2^(N-m)` indices that are 0 at all those positions —
/// the per-block anchor for an m-qubit gate. Inserting smallest-first keeps each later
/// (larger) position correct.
#[inline]
pub fn insert_zero_bits(index: usize, sorted_bits: &[u32]) -> usize {
    debug_assert!(
        sorted_bits.windows(2).all(|w| w[0] < w[1]),
        "insert_zero_bits requires strictly ascending positions"
    );
    // BMI2 fast path: depositing `index` into the *complement* of the gate-qubit mask places its
    // bits into exactly the non-gate positions (in order) and leaves the gate positions 0 — which
    // is precisely "insert a 0 at every gate position". Order-independent, so always correct.
    #[cfg(all(feature = "bmi2", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("bmi2") {
        return bmi2::insert_zero_bits_pdep(index, sorted_bits);
    }
    let mut x = index;
    for &b in sorted_bits {
        x = insert_zero_bit(x, b);
    }
    x
}

/// x86-64 BMI2 (`PEXT`/`PDEP`) implementations of the bit-index helpers. Each entry point is
/// `#[target_feature(enable = "bmi2")]`, so callers must check `is_x86_feature_detected!("bmi2")`
/// first (the `unsafe` is localized here; the crate is otherwise `deny(unsafe_code)`).
#[cfg(all(feature = "bmi2", target_arch = "x86_64"))]
#[allow(unsafe_code)]
pub mod bmi2 {
    use core::arch::x86_64::{_pdep_u64, _pext_u64};

    /// OR of `1 << q` over `qs` — the gate-qubit bit-mask for `PEXT`/`PDEP`.
    #[inline]
    pub fn qubit_mask(qs: &[u32]) -> u64 {
        qs.iter().fold(0u64, |m, &q| m | (1u64 << q))
    }

    /// `PDEP`-based [`insert_zero_bits`](super::insert_zero_bits) (order-independent).
    #[inline]
    pub fn insert_zero_bits_pdep(index: usize, bits: &[u32]) -> usize {
        let mask = qubit_mask(bits);
        // SAFETY: callers gate on runtime `is_x86_feature_detected!("bmi2")`.
        unsafe { _pdep_u64(index as u64, !mask) as usize }
    }

    /// `PEXT`-based [`gather_bits`](super::gather_bits): `mask` is the [`qubit_mask`] of
    /// **ascending** qubits (PEXT packs by position). For the index-generation microbenchmark.
    ///
    /// # Safety
    /// Requires the `bmi2` CPU feature; call only after `is_x86_feature_detected!("bmi2")`.
    #[target_feature(enable = "bmi2")]
    pub unsafe fn gather_bits_bmi2(index: usize, mask: u64) -> usize {
        _pext_u64(index as u64, mask) as usize
    }

    /// `PDEP`-based [`scatter_bits`](super::scatter_bits) (inverse of [`gather_bits_bmi2`]).
    ///
    /// # Safety
    /// Requires the `bmi2` CPU feature; call only after `is_x86_feature_detected!("bmi2")`.
    #[target_feature(enable = "bmi2")]
    pub unsafe fn scatter_bits_bmi2(sub: usize, mask: u64) -> usize {
        _pdep_u64(sub as u64, mask) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_zero_bit_examples() {
        // Insert at bit 0: every bit shifts up, bit 0 becomes 0 → result is even, = 2*i.
        for i in 0..32usize {
            assert_eq!(insert_zero_bit(i, 0), i << 1);
        }
        // Insert at bit 2 into 0b1011 (=11): low 2 bits (0b11) stay, high bits (0b10) move up
        // → 0b1_0_011 = 0b10011 = 19.
        assert_eq!(insert_zero_bit(0b1011, 2), 0b10011);
        // The inserted position must read back as 0.
        for i in 0..64usize {
            for b in 0..6u32 {
                assert_eq!(extract_bit(insert_zero_bit(i, b), b), 0);
            }
        }
    }

    #[test]
    fn pair_indices_differ_in_one_bit() {
        // For each i, a0/a1 must differ only in `bit` and be ordered a0 < a1.
        for bit in 0..5u32 {
            for i in 0..16usize {
                let a0 = insert_zero_bit(i, bit);
                let a1 = flip_bit(a0, bit);
                assert_eq!(a1, a0 + (1 << bit));
                assert_eq!(a0 ^ a1, 1 << bit);
            }
        }
    }

    #[test]
    fn gather_scatter_roundtrip() {
        let qs = [0u32, 2, 5];
        for sub in 0..8usize {
            let full = scatter_bits(sub, &qs);
            assert_eq!(gather_bits(full, &qs), sub);
        }
        // gather picks exactly the named bits: index 0b100101, qs=[0,2,5] → bits {1,1,1}=0b111
        assert_eq!(gather_bits(0b100101, &qs), 0b111);
    }

    #[test]
    fn insert_zero_bits_tiles_the_space() {
        // For N=4, a 2-qubit gate on positions {1,3}: the 2^(4-2)=4 base anchors, each
        // combined with the 2^2 sub-indices, must cover all 16 indices exactly once.
        let targets = [1u32, 3];
        let mut seen = [false; 16];
        for o in 0..4usize {
            let base = insert_zero_bits(o, &targets);
            assert_eq!(extract_bit(base, 1), 0);
            assert_eq!(extract_bit(base, 3), 0);
            for s in 0..4usize {
                let full = base | scatter_bits(s, &targets);
                assert!(!seen[full], "index {full} visited twice");
                seen[full] = true;
            }
        }
        assert!(seen.iter().all(|&b| b), "not all indices covered");
    }
}
