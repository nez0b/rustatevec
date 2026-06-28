//! v0.8 — gate fusion: the headline optimization.
//!
//! Statevector simulation is memory-bandwidth-bound, and **each gate is one streaming pass
//! over the `2^N` state** (read+write every amplitude) regardless of how many qubits it
//! touches — until the gate matrix gets large enough that its `2^m × 2^m` matvec overtakes
//! bandwidth. So merging *k* consecutive gates that together span ≤ `max_qubits` qubits turns
//! *k* passes into **one**: a near-*k×* reduction in memory traffic, the dominant win.
//!
//! This is a pure `Circuit -> Circuit` transform — backend-independent and testable in
//! isolation. The fused circuit runs on any [`Backend`](crate::backend::Backend) via its
//! multi-qubit kernel. Fusion is **the** reason to prefer this simulator on structured
//! circuits (QFT, QAOA, random circuits).
//!
//! The algorithm is the standard greedy one: walk gates in order, grow a fusion group by
//! absorbing the next gate while the group's qubit set stays ≤ `max_qubits`, then emit the
//! group as a single composed [`DenseGate`]. It is intentionally simple (consecutive-only, no
//! commutation-based reordering); a DAG-aware fuser is possible future work.

use crate::circuit::Circuit;
use crate::complex::Cplx;
use crate::gate::DenseGate;
use crate::real::Real;

/// Tuning for [`fuse`].
#[derive(Clone, Copy, Debug)]
pub struct FusionConfig {
    /// Maximum number of qubits in a fused gate (cap on the `2^m` matrix dimension).
    pub max_qubits: u32,
}

impl Default for FusionConfig {
    fn default() -> Self {
        // 4 keeps the fused matrix at 16×16 — a good bandwidth/compute balance (cf. qsim/Aer).
        Self { max_qubits: 4 }
    }
}

/// Fuse a circuit's gates greedily into ≤ `max_qubits`-qubit composite gates.
pub fn fuse<R: Real>(circuit: &Circuit<R>, cfg: &FusionConfig) -> Circuit<R> {
    let max = cfg.max_qubits.max(1);
    let mut out = Circuit::new(circuit.n_qubits());
    let mut group_qubits: Vec<u32> = Vec::new();
    let mut group_ops: Vec<(&DenseGate<R>, &[u32])> = Vec::new();

    for op in circuit.ops() {
        let g = op.gate();
        let qs = op.qubits();

        if qs.len() as u32 > max {
            // Gate is wider than the fusion cap: flush, then emit it standalone.
            flush_group(&mut out, &mut group_qubits, &mut group_ops);
            out.push(g.clone(), qs);
            continue;
        }

        let mut union = group_qubits.clone();
        for &x in qs {
            if !union.contains(&x) {
                union.push(x);
            }
        }

        if group_ops.is_empty() {
            group_qubits = qs.to_vec();
            group_ops.push((g, qs));
        } else if union.len() as u32 <= max {
            group_qubits = union;
            group_ops.push((g, qs));
        } else {
            flush_group(&mut out, &mut group_qubits, &mut group_ops);
            group_qubits = qs.to_vec();
            group_ops.push((g, qs));
        }
    }
    flush_group(&mut out, &mut group_qubits, &mut group_ops);
    out
}

fn flush_group<'a, R: Real>(
    out: &mut Circuit<R>,
    group_qubits: &mut Vec<u32>,
    group_ops: &mut Vec<(&'a DenseGate<R>, &'a [u32])>,
) {
    match group_ops.len() {
        0 => {}
        1 => {
            let (g, qs) = group_ops[0];
            out.push(g.clone(), qs);
        }
        _ => {
            let mut q = group_qubits.clone();
            q.sort_unstable();
            let fused = build_fused(&q, group_ops);
            out.push(fused, &q);
        }
    }
    group_qubits.clear();
    group_ops.clear();
}

/// Compose `group_ops` (applied in order) into one dense gate over the sorted qubit set `q`.
fn build_fused<R: Real>(q: &[u32], group_ops: &[(&DenseGate<R>, &[u32])]) -> DenseGate<R> {
    let m = q.len();
    let dim = 1usize << m;

    // Start from the identity on the Q-qubit space.
    let mut mat = vec![Cplx::<R>::zero(); dim * dim];
    for i in 0..dim {
        mat[i * dim + i] = Cplx::one();
    }

    for (g, qs) in group_ops {
        // Positions of this gate's qubits within the sorted group (Q-space bit indices).
        let local: Vec<usize> = qs
            .iter()
            .map(|&x| q.iter().position(|&y| y == x).unwrap())
            .collect();
        let nsub = 1usize << qs.len();

        // new = E(g) * mat, where E embeds g (on `local`) as identity elsewhere.
        let mut new = vec![Cplx::<R>::zero(); dim * dim];
        for r in 0..dim {
            let r_sub = gather_local(r, &local);
            for k_sub in 0..nsub {
                let e = g.at(r_sub, k_sub);
                if e == Cplx::zero() {
                    continue;
                }
                let k = set_local(r, &local, k_sub);
                for c in 0..dim {
                    new[r * dim + c] = new[r * dim + c] + e * mat[k * dim + c];
                }
            }
        }
        mat = new;
    }

    DenseGate::new(m as u32, mat)
}

/// Bits of `index` at Q-space positions `local`, packed into a sub-index (bit j ← `local[j]`).
#[inline]
fn gather_local(index: usize, local: &[usize]) -> usize {
    let mut s = 0usize;
    for (j, &p) in local.iter().enumerate() {
        s |= ((index >> p) & 1) << j;
    }
    s
}

/// `index` with the bits at Q-space positions `local` replaced by the bits of `sub`.
#[inline]
fn set_local(index: usize, local: &[usize], sub: usize) -> usize {
    let mut x = index;
    for (j, &p) in local.iter().enumerate() {
        let bit = (sub >> j) & 1;
        x = (x & !(1usize << p)) | (bit << p);
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fusion_reduces_gate_count_on_qft() {
        let c = crate::circuits::qft(6);
        let fused = fuse(&c, &FusionConfig::default());
        assert!(
            fused.ops().len() < c.ops().len(),
            "fusion should reduce passes: {} -> {}",
            c.ops().len(),
            fused.ops().len()
        );
    }

    #[test]
    fn single_gate_groups_pass_through() {
        // A circuit of gates on disjoint, far-apart qubits with max=1 fuses nothing.
        let mut c = Circuit::<f64>::new(4);
        c.h(0).h(2);
        let fused = fuse(&c, &FusionConfig { max_qubits: 1 });
        assert_eq!(fused.ops().len(), 2);
    }
}
