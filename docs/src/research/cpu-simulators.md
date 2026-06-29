# CPU statevector simulators

Concrete, reimplementable techniques distilled from the production CPU simulators (qsim,
Qiskit-Aer, QuEST, Yao.jl). Source lives in the shallow clones under `_local/`.

## The universal core kernel

All of them pair amplitudes via the same insert-zero-bit indexing — explained from scratch in
[The core kernel](../design/kernel.md). It surfaces under different names (`insertZeroBit`,
`index0`, bit-deposit/`PDEP`, controlled iteration over the active subspace), but the arithmetic is
identical. It generalizes to m-qubit gates (insert m zero bits) and to controlled gates (iterate
only the $2^{N-k}$ active subspace).

## Memory layout — SoA vs AoS

- **qsim**: Structure-of-Arrays (separate real / imaginary blocks, AVX-width aligned) → SIMD
  complex multiply with no lane shuffles.
- **Qiskit-Aer**: AoS (interleaved `Complex`), then works around it with separate real/imag
  views and de/re-interleave in the AVX2 path.

qsv chooses SoA — see [Architecture overview](../design/overview.md).

## SIMD complex multiply

$(a+bi)(c+di) = (ac-bd) + (ad+bc)i$, vectorized with FMA: the gate entry is a broadcast
scalar, the amplitudes are SIMD lanes. NEON is 128-bit (2× f64 lanes) so the Apple win is
modest; AVX2/AVX-512 give 4×/8× lanes.

## High- vs low-qubit dispatch (qsim, Aer)

Low target qubit (small stride) → permute within registers; high target qubit (large stride)
→ blocked streaming. Two code paths per gate.

## Gate fusion — the biggest end-to-end win

Merge adjacent 1–2 qubit gates into a $2^m \times 2^m$ matrix (m ≈ 4–5) so K gates → 1
pass. Cost model under the bandwidth-bound view: each gate ≈ one streaming pass regardless of
arity, until the fused matrix is large enough that the $2^m$ matvec overtakes bandwidth.
(qsim `fuser_mqubit.h`; Aer fusion transpiler pass.)

## Diagonal-gate fast path (Aer)

Z, S, T, PHASE, RZ, CZ, RZZ are diagonal: a single pass multiplying each amplitude by a phase
— no pairing, no second load. QFT and QAOA are phase-heavy, so this is a cheap ~2× there.

## Threading

qsim uses a custom `ParallelFor` with a size threshold (small loops stay single-threaded) and
static partitioning; QuEST uses OpenMP `collapse(2)`. The amplitude-pair work is disjoint, so
it is embarrassingly parallel.

## Profiling-driven lessons qsv adopts

A set of cross-cutting techniques the profiling literature (and our own benchmarks) keep
surfacing:

- **Type/representation stability is often the single biggest win** — mixed or promoted scalar
  types silently dominate runtime. qsv's analog: keep amplitudes and gate entries the same concrete
  `Real`, monomorphized (never `dyn`).
- **Unchecked indexing** — a modest gain after representation is fixed (qsv gets it from
  iterator-shaped loops, no `unsafe`).
- **Per-thread accumulators** for expectation values; **alias-table $O(1)$ sampling**;
  stack-allocated small gate matrices. qsv implements the last two directly (see
  [`crate::sample`](../design/optimization.md) and the `MAX_SUB` stack buffer).

## QuEST distributed simulation (future multi-node seam)

When the target qubit is higher than the locally-stored qubits, the partners live on
different MPI ranks. QuEST pairs ranks by XOR-toggling the relevant bit, exchanges
half-buffers with `MPI_Sendrecv` (deadlock-free, all pairs concurrent), then applies the 2×2
locally. Captured for a future distributed backend behind the same `Backend` seam.

## BMI2 (PEXT/PDEP) — the bandwidth-bound proof

The bit-gather/scatter/insert at the kernel's core map to single `PEXT`/`PDEP` instructions on
x86-64. qsv's `bmi2` feature and `index_gen` microbenchmark measure this directly: **~4× faster in
isolation** (PEXT/PDEP vs the scalar loop) but **0.995× end-to-end** on a fused QFT-18 — because
they accelerate address computation, not memory traffic, and the kernel is bandwidth-bound. This is
the empirical anchor for the whole optimization strategy (numbers in
[`bench/results/SUMMARY-xeon.md`](https://github.com/nez0b/rustatevec/blob/main/bench/results/SUMMARY-xeon.md);
the same effect is well documented in the wider community, e.g. QuEST's BMI2 work).
