//! Bit-index arithmetic — the addressing foundation of the optimized kernels.
//!
//! [`insert_zero_bit`] is the **universal core trick** (QuEST's `insertZeroBit`, Aer's
//! `index0`, qsim's bit-deposit, Yao's `IterControl`, and `expand_int` in the reference
//! `docs/reference/state_vector.jl`). Applying a 1-qubit gate to qubit `q` pairs up
//! amplitudes whose indices differ only in bit `q`; we enumerate the `2^(N-1)` pairs by
//! looping `i` and reconstructing the partner indices on the fly — no reshape/permute copy.
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
    let mut x = index;
    for &b in sorted_bits {
        x = insert_zero_bit(x, b);
    }
    x
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
