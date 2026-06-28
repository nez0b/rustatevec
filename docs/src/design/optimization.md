# How we optimize

This is the heart of the project: a disciplined, evidence-driven path from a naive simulator
to a fast one, where every step is measurable and justified by a single principle.

## The principle: it's memory-bandwidth-bound

Consider applying one 1-qubit gate to an \\(N\\)-qubit state. The kernel must touch all
\\(2^N\\) amplitudes, reading and writing each once. Per amplitude it does roughly **2 complex
multiply-adds** (the \\(2\times2\\) matrix times a 2-vector). In `f64`, each amplitude is 16
bytes (real + imaginary).

\\[
\text{arithmetic intensity} \approx \frac{\sim 8\ \text{FLOP per pair}}{2 \times 16\ \text{bytes per pair}} \approx 0.13\ \text{FLOP/byte}.
\\]

A modern CPU sustains tens of FLOP/byte before it runs out of compute; at 0.13 FLOP/byte the
kernel is starved for **memory bandwidth**, not arithmetic. On the
[roofline](https://en.wikipedia.org/wiki/Roofline_model) it sits far to the left of the
ridge point — its ceiling is the slope (bandwidth), not the flat top (peak FLOPs).

This is confirmed independently across every production simulator we studied (qsim,
Qiskit-Aer, QuEST, Yao.jl, cuStateVec, spinoza). It is the single most important fact about
the problem, and it dictates everything below.

> **The governing question for any change:** does it reduce *bytes moved per gate*, or raise
> *arithmetic intensity per byte moved*? If neither, it will not move the needle, no matter
> how clever it looks in isolation.

### The clinching evidence

QuEST's own BMI2 optimization ([issue #717](https://github.com/QuEST-Kit/QuEST/issues/717),
which this project's author contributed to) replaces a bit-twiddling loop with single `PEXT`/
`PDEP` instructions — **6–12× faster in isolation**. Its end-to-end speedup on real circuits
is only **1.0–1.3×**, shrinking toward 1.0 as thread count rises and memory bandwidth
saturates. The reason: it accelerates *address computation*, not *memory traffic*. This is
the empirical anchor for treating bandwidth, not arithmetic, as the bottleneck — and for
ordering our work accordingly.

## The priority order

From highest to lowest end-to-end impact:

1. **Gate fusion** — merge adjacent 1–2 qubit gates into a single 2–5 qubit matrix so that
   *K* gates become *one* pass over the state. Fewer passes = fewer bytes moved. The biggest
   single win on structured circuits.
2. **SoA layout** — separate `re[]`/`im[]` arrays so SIMD complex multiply needs no
   de-interleave shuffles.
3. **Cache-aware access / blocking** — keep the working set hot and minimize passes;
   non-temporal stores to skip read-for-ownership on write-back.
4. **Multithreading** — the amplitude pairs are disjoint, so the loop is embarrassingly
   parallel; the only subtleties are a single-thread size threshold and load balancing.
5. **Micro-optimizations** — BMI2 index generation, unchecked indexing, ILP/unrolling,
   prefetch. Real but small; the long tail.

The order matters: fusion and layout change *how much memory you move*, which dominates;
micro-ops only shave the constant on work you are already doing.

## The milestone narrative

Rather than ship one optimized backend, qsv evolves through a sequence of **separate
backends**, each a self-contained, benchmarkable diff and one data point on the headline
throughput/roofline plot. The progression deliberately mirrors the pedagogical
`state_vector.jl` notebook, then goes further.

| Ver | Change | Expected regime | What it demonstrates |
| --- | --- | --- | --- |
| v0.0 | naive dense oracle | caps ~13q | correctness baseline; why naive is impossible |
| v0.1 | reshape / block apply (out-of-place) | ~24q, allocation-heavy | the reshape model; alloc/stride cost |
| v0.2 | in-place bit-shift kernel | order of magnitude | the universal core kernel |
| v0.3 | unchecked indexing + stack matrices | 1.3–2× | bounds-check & allocation cost |
| v0.4 | high/low target-qubit dispatch | 1.3–2× | cache-line / stride awareness |
| v0.5 | multithreading (rayon) | ~4–6× | parallel disjoint pairs; P/E-core story |
| v0.6 | diagonal-gate fast path | ~2× on phase-heavy | recognizing structure |
| v0.7 | SIMD complex multiply (SoA) | 1.3–1.8× NEON / 2–4× AVX | the SoA payoff; honest lane-count story |
| v0.8 | **gate fusion** | 2–5× end-to-end | fewer passes — the dominant win |
| v0.9 | cache-blocking + prefetch + NT-stores | 1.2–1.5× | working-set control |
| v0.10 | ILP, BMI2, alias-table sampling, parallel scan | 1.05–1.2× | the honest long tail |

Speedups are *regimes*, not promises — the real numbers come from the
[benchmark harness](benchmarking.md) and are reported honestly, including where we lose to
mature simulators.

### A deliberate sequencing choice

Fusion (v0.8) lands *after* SIMD and threading on purpose: its value is best measured as a
multiplier on top of an already-fast kernel, which is the realistic and most informative
framing.

## Knowing when to stop

Optimization has diminishing returns, and chasing them is a trap. The stop criterion is built
into the methodology: **when the dominant kernels sustain ≥ 70–80% of the machine's measured
STREAM-triad bandwidth, further micro-optimization is noise** — the remaining effort should go
to fusion or algorithmic improvements that change the bytes-moved equation, not to shaving the
constant. Stating this explicitly is part of the story.

## A note on Apple Silicon

The primary development machine is an Apple M3 Pro: 128-bit NEON (only **2× f64 lanes**) and a
heterogeneous 5 performance + 6 efficiency core layout. Two honest consequences we report
rather than hide:

- the SIMD win on NEON is modest (~1.3–1.8×); the 2–4× payoff appears on x86 AVX2/AVX-512,
  which is why we benchmark on both;
- equal static thread partitioning makes the slow efficiency cores stragglers, so we use
  work-stealing with many small chunks and report P-only vs all-core scaling separately.
