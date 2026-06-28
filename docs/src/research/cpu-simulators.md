# CPU statevector simulators

Concrete, reimplementable techniques from qsim, Qiskit-Aer, QuEST, Yao.jl, and the
`docs/reference/state_vector.jl` notebook. Source lives in the shallow clones under
`references/`.

## The universal core kernel

All of them pair amplitudes via the same insert-zero-bit indexing — explained from scratch in
[The core kernel](../design/kernel.md). It appears as `expand_int` in the reference notebook,
`insertZeroBit` in QuEST, `index0` in Aer, bit-deposit (`PDEP`) in qsim, and `IterControl` in
Yao/BitBasis. Generalizes to m-qubit gates (insert m zero bits) and to controlled gates
(iterate only the \\(2^{N-k}\\) active subspace).

## Memory layout — SoA vs AoS

- **qsim**: Structure-of-Arrays (separate real / imaginary blocks, AVX-width aligned) → SIMD
  complex multiply with no lane shuffles.
- **Qiskit-Aer**: AoS (interleaved `Complex`), then works around it with separate real/imag
  views and de/re-interleave in the AVX2 path.

qsv chooses SoA — see [Architecture overview](../design/overview.md).

## SIMD complex multiply

\\((a+bi)(c+di) = (ac-bd) + (ad+bc)i\\), vectorized with FMA: the gate entry is a broadcast
scalar, the amplitudes are SIMD lanes. NEON is 128-bit (2× f64 lanes) so the Apple win is
modest; AVX2/AVX-512 give 4×/8× lanes.

## High- vs low-qubit dispatch (qsim, Aer)

Low target qubit (small stride) → permute within registers; high target qubit (large stride)
→ blocked streaming. Two code paths per gate.

## Gate fusion — the biggest end-to-end win

Merge adjacent 1–2 qubit gates into a \\(2^m \times 2^m\\) matrix (m ≈ 4–5) so K gates → 1
pass. Cost model under the bandwidth-bound view: each gate ≈ one streaming pass regardless of
arity, until the fused matrix is large enough that the \\(2^m\\) matvec overtakes bandwidth.
(qsim `fuser_mqubit.h`; Aer fusion transpiler pass.)

## Diagonal-gate fast path (Aer)

Z, S, T, PHASE, RZ, CZ, RZZ are diagonal: a single pass multiplying each amplitude by a phase
— no pairing, no second load. QFT and QAOA are phase-heavy, so this is a cheap ~2× there.

## Threading

qsim uses a custom `ParallelFor` with a size threshold (small loops stay single-threaded) and
static partitioning; QuEST uses OpenMP `collapse(2)`. The amplitude-pair work is disjoint, so
it is embarrassingly parallel.

## Lessons from `state_vector.jl`

The profiling-driven progression qsv mirrors:

- **Type stability was the single biggest win** (`Complex{Int} → ComplexF64` promotion
  dominated). Rust analog: keep amplitudes and gate entries the same concrete `Real`.
- Unchecked indexing (`@inbounds`) — modest gain after type stability.
- Linear gate indexing; **per-thread accumulators** for expectation values; **alias-table
  O(1) sampling**; stack-allocated small gate matrices (`StaticArrays`).

## QuEST distributed simulation (future multi-node seam)

When the target qubit is higher than the locally-stored qubits, the partners live on
different MPI ranks. QuEST pairs ranks by XOR-toggling the relevant bit, exchanges
half-buffers with `MPI_Sendrecv` (deadlock-free, all pairs concurrent), then applies the 2×2
locally. Captured for a future distributed backend behind the same `Backend` seam.

## QuEST issue #717 (BMI2) — the bandwidth-bound proof

`getValueOfBits`/`insertBits` get 6–12× faster with BMI2 `PEXT`/`PDEP` in isolation, but only
**1.0–1.3× end-to-end** — because they speed up address computation, not memory traffic. This
is the empirical anchor for the whole optimization strategy.
