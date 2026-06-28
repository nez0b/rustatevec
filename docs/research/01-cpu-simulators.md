# CPU statevector simulators — kernels & optimizations

Concrete, reimplementable techniques from qsim, Qiskit-Aer, QuEST, Yao.jl, and the
`docs/reference/state_vector.jl` notebook. Code paths reference the shallow clones in
`references/`.

## 1. The universal core kernel: `insert_zero_bit` / `flip_bit`

Applying a 1-qubit gate to qubit `q` pairs amplitudes whose indices differ only in bit `q`.
Iterate `i` over `0 .. 2^(N-1)` and reconstruct the pair on the fly — no reshape/permute copy:

```
left  = (i >> q) << q          # bits ≥ q
right = i - left               # bits < q
a0 = (left << 1) | right       # insert a 0 bit at position q     -> |…0…⟩
a1 = a0 ^ (1 << q)             # flip bit q                       -> |…1…⟩
(s[a0], s[a1]) = (g00·s[a0] + g01·s[a1], g10·s[a0] + g11·s[a1])   # in place
```

This exact arithmetic appears as:
- `expand_int` / `flip_bit` in `docs/reference/state_vector.jl`
- `insertZeroBit` / `flipBit` / `extractBit` in QuEST (`references/QuEST`, CPU kernels)
- `index0` + `indexes()` in Qiskit-Aer (`references/qiskit-aer/.../statevector/indexes.hpp`)
- bit-deposit (BMI2 `PDEP`) in qsim (`references/qsim/lib/bits.h`)
- `IterControl` in Yao/BitBasis (`references/BitBasis.jl`)

Implemented in this repo as `qsv_core::state::layout::{insert_zero_bit, flip_bit,
extract_bit, gather_bits, scatter_bits}`.

**Generalizations**
- *m-qubit gate*: insert `m` zero bits at the **sorted** target positions (smaller first),
  enumerate the `2^m` sub-indices.
- *controlled gate, k controls*: iterate only the `2^(N−k)` active subspace — either a mask
  test `(idx & ctrlMask) == ctrlFlipMask` (QuEST) or `IterControl` enumeration (Yao). Cuts
  work by `2^k`.

## 2. Memory layout — SoA vs AoS

- **qsim**: Structure-of-Arrays (separate real / imaginary blocks, AVX-width aligned). A
  SIMD load of `re` and of `im` each yields a homogeneous register → complex multiply is a
  broadcast-FMA chain with **no lane shuffles**.
- **Qiskit-Aer**: AoS (interleaved `Complex`), then works around it with separate real/imag
  *views* and de/re-interleave in the AVX2 path.

**Decision for qsv:** SoA (`re[]`, `im[]`). Avoids NEON `ld2` / x86 `unpck` overhead the
bandwidth-bound kernel can't afford. Implemented in `qsv_core::state::StateVector`.

## 3. SIMD complex multiply

`(a+bi)(c+di) = (ac − bd) + (ad + bc)i`, vectorized with FMA: the gate entry is a broadcast
scalar, the amplitudes are SIMD lanes. qsim and Aer both use `fnmadd`/`fmadd` chains.

Honest caveat for our primary box: **NEON is 128-bit = 2×f64 lanes**, so the Apple-Silicon
SIMD win is modest (~1.3–1.8×); the 2–4× payoff is on AVX2/AVX-512 (x86). We report both.

## 4. High-qubit vs low-qubit dispatch (qsim, Aer)

- *Low target qubit* (small stride, pair fits in a SIMD register / cache line) → permute /
  shuffle within registers.
- *High target qubit* (large stride) → blocked streaming kernel.

Two code paths chosen per gate. (qsim `simulator_avx.h` `ApplyGateL`/`ApplyGateH`; Aer
`qv_avx2.cpp`.)

## 5. Gate fusion — the biggest end-to-end win

Merge adjacent 1–2q gates into a `2^m × 2^m` matrix (`m ≈ 4–5`) so K gates → 1 pass over the
state. **Cost model (bandwidth-bound):** each gate ≈ one full streaming pass (`~2^N · 2 ·
sizeof(R)` bytes) regardless of arity, until the fused matrix gets large enough that the
`2^m` matvec overtakes bandwidth. Greedy merge over dependency-ordered ops while the qubit
set stays ≤ `max_fused`. (qsim `fuser_mqubit.h`; Aer fusion transpiler pass.)

## 6. Diagonal-gate fast path (Aer)

Z, S, T, PHASE, RZ, CZ, RZZ are diagonal: a single pass multiplying each amplitude by a phase
— **no pairing, no second load**. QFT and QAOA are phase-heavy, so this is a cheap ~2× there.

## 7. Threading

- qsim: custom `ParallelFor` with a **size threshold** (`<1024` ⇒ single-thread) and static
  partitioning (no false sharing).
- QuEST: OpenMP `collapse(2)` over the outer/inner pair loops.
- The amplitude-pair work is disjoint ⇒ embarrassingly parallel.

For qsv: rayon with chunk count ≫ cores so Apple's heterogeneous P/E cores work-steal and
auto-balance; benchmark P-only (5) vs all-11.

## 8. Lessons from `state_vector.jl` (the reference notebook)

Profiling-driven progression we mirror in our milestone roadmap:
- **Type stability was the single biggest win** (`Complex{Int}` → `ComplexF64` promotion was
  dominating). Rust analog: keep gate matrices and amplitudes the same concrete `Real`; no
  implicit promotion.
- `@inbounds` (unchecked indexing) — modest gain after type stability.
- Linear gate indexing (`gate[1], gate[3]` vs `gate[1,1]`).
- **Per-thread accumulators** for expectation values (avoid the data race).
- **Alias-table O(1) sampling** vs O(shots·N) cumulative-sum search.
- Stack-allocated small gate matrices (`StaticArrays`) — no per-gate heap alloc. Rust analog:
  const-generic `Mat2`/`Mat4`, `SmallVec` target lists.

## 9. QuEST distributed simulation (future multi-node seam)

When the target qubit is higher than the locally-stored qubits, the `|0⟩`/`|1⟩` partners live
on different MPI ranks. QuEST pairs ranks by XOR-toggling the relevant bit
(`getChunkPairId`), exchanges half-buffers with `MPI_Sendrecv` (deadlock-free, all pairs
concurrent), then applies the 2×2 locally. Captured for a future distributed backend behind
the same `Backend` seam; not in v1.

## 10. QuEST issue #717 (BMI2) — the bandwidth-bound proof

`getValueOfBits`/`insertBits` appear in ~45 hot loops; BMI2 `PEXT`/`PDEP` do them in one
instruction (6–12× faster in isolation). **End-to-end: only 1.0–1.3×**, shrinking toward 1.0
as thread count rises and memory bandwidth saturates. This is the empirical anchor for
treating bandwidth — not bit-twiddling — as the real bottleneck.
