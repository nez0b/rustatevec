# The core kernel

Every production statevector simulator — qsim, Qiskit-Aer, QuEST, Yao.jl, cuStateVec — shares
the same indexing trick at its heart. qsv calls it `insert_zero_bit`. Understanding it is
understanding 90% of how a statevector simulator works.

## The problem

Applying a 1-qubit gate $G = \begin{pmatrix} g_{00} & g_{01} \\ g_{10} & g_{11}
\end{pmatrix}$ to qubit $q$ couples exactly the amplitude pairs whose basis indices
differ *only* in bit $q$:

$$
\begin{aligned}
\psi'_{a_0} &= g_{00}\,\psi_{a_0} + g_{01}\,\psi_{a_1} \\
\psi'_{a_1} &= g_{10}\,\psi_{a_0} + g_{11}\,\psi_{a_1}
\end{aligned}
\qquad\text{where } a_1 = a_0 \oplus 2^q .
$$

There are $2^{N-1}$ such pairs. The naive approach builds a $2^N \times 2^N$ matrix and
multiplies — catastrophically wasteful in both memory and compute. The reshape approach
(v0.1) avoids the matrix but copies the whole state per gate. The efficient approach visits
each pair exactly once, in place.

## Enumerating the pairs

We loop a counter `i` over `0 .. 2^(N-1)` and *reconstruct* the two partner indices from it.
The trick is to take `i`, which has $N-1$ bits, and **insert a 0 bit at position `q`** to
get $a_0$; flipping that bit gives $a_1$.

```rust
/// Insert a `0` bit at position `bit`, shifting higher bits up by one.
pub fn insert_zero_bit(index: usize, bit: u32) -> usize {
    let left  = (index >> bit) << bit; // bits ≥ bit
    let right = index - left;          // bits < bit
    (left << 1) | right
}

pub fn flip_bit(index: usize, bit: u32) -> usize {
    index ^ (1usize << bit)
}
```

`insert_zero_bit` splits `index` at position `bit`, shifts the high part up by one to open a
gap, and ORs the low part back. As `i` ranges over all $2^{N-1}$ values, $a_0 =
\texttt{insert\_zero\_bit}(i, q)$ ranges over exactly the indices with bit `q` clear — every
pair, once.

This indexing trick is the shared heart of every production statevector simulator (it appears
under various names — insert-zero-bit, `index0`, bit-deposit), and on x86-64 it maps directly to
the `PDEP` instruction — see [How we optimize](optimization.md#the-clinching-evidence) for the
measured BMI2 path and why it is *not* the bottleneck.

## The in-place kernel

Putting it together (qsv's `BitShiftBackend::apply_1q`, lightly trimmed):

```rust
let pairs = state.dim() >> 1;
let (re, im) = state.parts_mut();   // SoA: separate real/imag slices
for i in 0..pairs {
    let a0 = insert_zero_bit(i, q);
    let a1 = a0 | (1usize << q);
    let x0 = Cplx::new(re[a0], im[a0]);
    let x1 = Cplx::new(re[a1], im[a1]);
    re[a0] = (g00 * x0 + g01 * x1).re;  im[a0] = (g00 * x0 + g01 * x1).im;
    re[a1] = (g10 * x0 + g11 * x1).re;  im[a1] = (g10 * x0 + g11 * x1).im;
}
```

No $2^N$ matrix, no per-gate allocation, no copy — just one streaming pass over the state,
which (per [How we optimize](optimization.md)) is exactly the bandwidth-bound minimum.

## Generalizing

**Multi-qubit gates.** For an $m$-qubit gate, insert $m$ zero bits at the *sorted*
target positions to anchor each of the $2^{N-m}$ blocks, then enumerate the $2^m$
sub-indices within a block (`insert_zero_bits` + `scatter_bits`). qsv's `apply_mq` gathers a
block's $2^m$ amplitudes into a small stack buffer, applies the matrix, and writes back —
still in place, still zero-allocation.

**Controlled gates.** A controlled gate only changes amplitudes where the control bits are 1,
so the optimized form iterates just the $2^{N-k}$ active subspace (with $k$ controls),
cutting work by $2^k$. (In the current milestone, controlled gates are handled as plain
dense gates via `apply_mq`; the subspace specialization is a later optimization.)

**High vs low target qubits.** When `q` is small the pair stride $2^q$ is tiny and both
partners fall in the same cache line / SIMD register; when `q` is large the stride is huge and
the access pattern is cache-hostile. qsim and Aer use two code paths — a permute-within-
register kernel for low qubits and a blocked streaming kernel for high ones. This is qsv's
v0.4 milestone.

## Why a separate, *different* oracle

The optimized kernels all rest on this index arithmetic, so a bug in `insert_zero_bit` would
be invisible to a test that used the same trick. qsv's `RefBackend` oracle therefore applies
gates a *structurally different* way — gather/scatter per output amplitude — so the
[differential tests](roadmap.md#testing) genuinely cross-check the indexing. The helpers
themselves are also exhaustively unit-tested (every pair differs in exactly the target bit;
the block anchors tile the index space).
