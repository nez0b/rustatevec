//! Sampling and diagonal expectation values — the measurement-side primitives.
//!
//! - **Alias table** (Vose's method): `O(N)` build, **`O(1)` per shot** sampling from the
//!   measurement distribution `|ψ_i|²`. Beats inverse-CDF + binary search (`O(log N)`/shot) when
//!   many shots are drawn.
//! - **Parallel prefix-sum** (two-pass chunked scan): the work-efficient `scan` primitive; builds
//!   the CDF for the inverse-CDF sampler and normalizes large distributions.
//! - **Diagonal expectation** `⟨Z-string⟩` via a parallel map-reduce (per-thread accumulators).
//!
//! All reductions fall back to a serial path below a size threshold or without the `parallel`
//! feature. Randomness uses the crate's [`SplitMix64`](crate::circuits::SplitMix64).

use crate::circuits::SplitMix64;
use crate::real::Real;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Distributions below this length use the serial path (parallel overhead not worth it).
#[cfg(feature = "parallel")]
const PAR_MIN: usize = 1 << 16;

/// Sum of `xs` (as `f64`); parallel reduce for large inputs.
fn sum_f64<R: Real>(xs: &[R]) -> f64 {
    #[cfg(feature = "parallel")]
    if xs.len() >= PAR_MIN {
        return xs.par_iter().map(|x| x.to_f64()).sum();
    }
    xs.iter().map(|x| x.to_f64()).sum()
}

/// A Vose alias table over a discrete distribution: `O(1)` sampling after an `O(N)` build.
pub struct AliasTable {
    prob: Vec<f64>,
    alias: Vec<usize>,
}

impl AliasTable {
    /// Build from non-negative `weights` (need not be normalized; must not be all-zero).
    pub fn build<R: Real>(weights: &[R]) -> Self {
        let n = weights.len();
        assert!(n > 0, "AliasTable needs a non-empty distribution");
        let total = sum_f64(weights);
        let scale = if total > 0.0 { n as f64 / total } else { 0.0 };
        // scaled[i] = p_i * n, average 1; partition into < 1 ("small") and ≥ 1 ("large").
        let mut scaled: Vec<f64> = weights.iter().map(|w| w.to_f64() * scale).collect();
        let mut prob = vec![0.0f64; n];
        let mut alias = vec![0usize; n];
        let mut small = Vec::new();
        let mut large = Vec::new();
        for (i, &p) in scaled.iter().enumerate() {
            if p < 1.0 {
                small.push(i);
            } else {
                large.push(i);
            }
        }
        // NOTE: pop both only when both are non-empty. `while let (Some, Some) = (pop, pop)`
        // would eagerly pop the non-empty list on the terminating iteration and *discard* that
        // element, leaving its `prob` at 0 (a subtle distribution bug).
        while !small.is_empty() && !large.is_empty() {
            let s = small.pop().unwrap();
            let l = large.pop().unwrap();
            prob[s] = scaled[s];
            alias[s] = l;
            // Move the deficit (1 - scaled[s]) out of bucket `l`.
            scaled[l] = (scaled[l] + scaled[s]) - 1.0;
            if scaled[l] < 1.0 {
                small.push(l);
            } else {
                large.push(l);
            }
        }
        // Leftovers (rounding) are exactly full.
        for l in large {
            prob[l] = 1.0;
        }
        for s in small {
            prob[s] = 1.0;
        }
        Self { prob, alias }
    }

    /// Draw one index in `O(1)`. Uses `unit()`'s high bits for *both* the bucket choice and the
    /// accept test — `below()`'s low-bit modulo is biased under the strided stream consumption a
    /// two-draws-per-shot loop produces (SplitMix64 low bits weaken when the counter stride is
    /// even).
    #[inline]
    pub fn draw(&self, rng: &mut SplitMix64) -> usize {
        let n = self.prob.len();
        let i = ((rng.unit() * n as f64) as usize).min(n - 1);
        if rng.unit() < self.prob[i] {
            i
        } else {
            self.alias[i]
        }
    }

    pub fn len(&self) -> usize {
        self.prob.len()
    }
    pub fn is_empty(&self) -> bool {
        self.prob.is_empty()
    }
}

/// Draw `shots` indices `∝ weights` via an alias table (`O(N)` build + `O(shots)` draws).
pub fn alias_sample<R: Real>(weights: &[R], shots: usize, rng: &mut SplitMix64) -> Vec<usize> {
    let table = AliasTable::build(weights);
    (0..shots).map(|_| table.draw(rng)).collect()
}

/// Inclusive parallel prefix-sum (scan): `out[i] = Σ_{k≤i} xs[k]` in `f64`.
///
/// Two-pass chunked scan: (1) per-chunk local inclusive scan in parallel; (2) serial scan of the
/// per-chunk totals into exclusive offsets (`#chunks` is tiny); (3) add each chunk's offset in
/// parallel. Work-efficient `O(N)`, span `O(N/p + p)`.
pub fn parallel_prefix_sum<R: Real>(xs: &[R]) -> Vec<f64> {
    let n = xs.len();
    let mut out = vec![0.0f64; n];

    #[cfg(feature = "parallel")]
    if n >= PAR_MIN {
        let chunk = n.div_ceil(rayon::current_num_threads().max(1));
        out.par_chunks_mut(chunk)
            .zip(xs.par_chunks(chunk))
            .for_each(|(o, x)| {
                let mut acc = 0.0;
                for (oi, xi) in o.iter_mut().zip(x) {
                    acc += xi.to_f64();
                    *oi = acc;
                }
            });
        let nchunks = n.div_ceil(chunk);
        let mut offsets = vec![0.0f64; nchunks];
        let mut acc = 0.0;
        for (c, off) in offsets.iter_mut().enumerate() {
            *off = acc;
            let last = ((c + 1) * chunk).min(n) - 1;
            acc += out[last];
        }
        out.par_chunks_mut(chunk).enumerate().for_each(|(c, o)| {
            let off = offsets[c];
            if off != 0.0 {
                for oi in o.iter_mut() {
                    *oi += off;
                }
            }
        });
        return out;
    }

    let mut acc = 0.0;
    for (oi, xi) in out.iter_mut().zip(xs) {
        acc += xi.to_f64();
        *oi = acc;
    }
    out
}

/// Draw `shots` indices `∝ weights` via inverse-CDF (parallel-scan CDF + `O(log N)` binary search
/// per shot). Mainly a cross-check / few-shots alternative to [`alias_sample`].
pub fn cdf_sample<R: Real>(weights: &[R], shots: usize, rng: &mut SplitMix64) -> Vec<usize> {
    let cdf = parallel_prefix_sum(weights);
    let total = *cdf.last().unwrap_or(&0.0);
    let last = cdf.len().saturating_sub(1);
    (0..shots)
        .map(|_| {
            let u = rng.unit() * total;
            cdf.partition_point(|&c| c < u).min(last)
        })
        .collect()
}

/// `⟨ψ| Z_mask |ψ⟩` — expectation of the Pauli-Z product on the qubits set in `mask`, given the
/// probabilities `probs[i] = |ψ_i|²`: `Σ_i probs[i] · (-1)^popcount(i & mask)`. Parallel reduce.
pub fn expectation_z<R: Real>(probs: &[R], mask: usize) -> R {
    let signed = |i: usize, p: f64| {
        if (i & mask).count_ones() & 1 == 0 {
            p
        } else {
            -p
        }
    };
    #[cfg(feature = "parallel")]
    if probs.len() >= PAR_MIN {
        let s: f64 = probs
            .par_iter()
            .enumerate()
            .map(|(i, p)| signed(i, p.to_f64()))
            .sum();
        return R::from_f64(s);
    }
    let s: f64 = probs
        .iter()
        .enumerate()
        .map(|(i, p)| signed(i, p.to_f64()))
        .sum();
    R::from_f64(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_sum_matches_serial() {
        let xs: Vec<f64> = (0..(1usize << 17)).map(|i| (i % 7) as f64).collect();
        let scan = parallel_prefix_sum(&xs);
        let mut acc = 0.0;
        for (i, &x) in xs.iter().enumerate() {
            acc += x;
            assert!((scan[i] - acc).abs() < 1e-6, "mismatch at {i}");
        }
    }

    #[test]
    fn alias_histogram_matches_distribution() {
        // Distribution [0.1, 0.6, 0.0, 0.3].
        let w = [0.1f64, 0.6, 0.0, 0.3];
        let mut rng = SplitMix64::new(0xDEAD_BEEF);
        let shots = 400_000;
        let samples = alias_sample(&w, shots, &mut rng);
        let mut counts = [0usize; 4];
        for s in samples {
            counts[s] += 1;
        }
        assert_eq!(counts[2], 0, "zero-prob outcome must never appear");
        for (k, &p) in w.iter().enumerate() {
            let freq = counts[k] as f64 / shots as f64;
            assert!((freq - p).abs() < 0.01, "outcome {k}: freq {freq} vs p {p}");
        }
    }

    #[test]
    fn cdf_and_alias_agree_statistically() {
        let w = [0.25f64, 0.25, 0.4, 0.1];
        let shots = 200_000;
        let mut r1 = SplitMix64::new(1);
        let mut r2 = SplitMix64::new(2);
        let hist = |s: Vec<usize>| {
            let mut c = [0usize; 4];
            for x in s {
                c[x] += 1;
            }
            c
        };
        let a = hist(alias_sample(&w, shots, &mut r1));
        let b = hist(cdf_sample(&w, shots, &mut r2));
        for k in 0..4 {
            let fa = a[k] as f64 / shots as f64;
            let fb = b[k] as f64 / shots as f64;
            assert!((fa - fb).abs() < 0.01, "outcome {k}: alias {fa} vs cdf {fb}");
        }
    }

    #[test]
    fn expectation_z_basic() {
        // |00⟩: all Z expectations = +1.
        let p00 = [1.0f64, 0.0, 0.0, 0.0];
        assert!((expectation_z(&p00, 0b01) - 1.0).abs() < 1e-12);
        assert!((expectation_z(&p00, 0b11) - 1.0).abs() < 1e-12);
        // GHZ on 2 qubits: (|00⟩+|11⟩)/√2 → p = [0.5,0,0,0.5]. ⟨Z0⟩=0, ⟨Z0 Z1⟩=+1.
        let ghz = [0.5f64, 0.0, 0.0, 0.5];
        assert!(expectation_z(&ghz, 0b01).abs() < 1e-12);
        assert!((expectation_z(&ghz, 0b11) - 1.0).abs() < 1e-12);
    }
}
