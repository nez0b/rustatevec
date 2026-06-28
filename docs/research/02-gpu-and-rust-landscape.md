# GPU path, Rust ecosystem, and HPC techniques

Landscape research for the (deferred) GPU backend and the CPU optimization toolbox.

## 1. cuStateVec / cuda-quantum (GPU kernel structure)

NVIDIA's cuStateVec applies gates with the same bit-index pairing as the CPU, mapped to the
GPU:
- **Thread → amplitude-pair** mapping; the gate matrix (`2^k × 2^k`) is staged into
  **shared memory** and reused across the block's pairs.
- **Coalescing** depends on the target-qubit stride: low-order targets → consecutive,
  coalesced loads; **high-order targets → large strides → poor coalescing**, mitigated by
  qubit reordering (swap large-stride qubits to low indices), the GPU analog of QuEST's
  distributed exchange.
- Multi-GPU: statevector partitioned by index bits; gates on "global" qubits require
  GPU↔GPU exchange over NVLink/MPI.
- Memory hierarchy: statevector in HBM (the bottleneck); achieved bandwidth ~60–80% of peak
  due to strided access + complex arithmetic only using ~50% of FMA peak. **Same
  bandwidth-bound story as CPU.**

cuda-quantum's `nvq++` lowers circuits to `custatevecApplyMatrix` calls.

## 2. cuTile / cuTile-rs — ⚠️ UNVERIFIED

The landscape agent returned confident, specific claims (an NVlabs `cutile-rs` repo with a
star count, an arXiv id `2606.15991`, a `cuda-tile` PyPI package, B200 perf numbers, a
"Grout" Qwen3 inference engine). **These post-date the knowledge cutoff and are unverified —
treat as a lead to check, not fact.** Plausible kernel: per-thread-block tile loaded to
shared memory, tile ops over amplitudes, JIT via a CUDA Tile IR.

Because we chose **CPU-first with a pluggable `Backend` seam**, none of this is on the
critical path. Before ever committing to cuTile we must independently verify the repo,
maturity, hardware/driver requirements, and that it suits a memory-bound (not GEMM-shaped)
kernel.

## 3. Rust quantum-sim ecosystem — the gap we fill

| Project | Approach | GPU | Notes |
|---|---|---|---|
| `spinoza` | CPU SIMD + rayon | ✗ | Closest peer (~30q); the prior Rust art to beat. `references/spinoza` |
| `qoqo`/`roqoqo` (HQS) | circuit DSL, delegates to backends | ✗ | No native sim engine |
| `qip`, `qasmsim`, `quantum` | builders / educational / QASM | ✗ | Not perf-focused |
| `qoqo-quest` | Rust wrapper over QuEST (C) | via QuEST | Not native Rust |

**Gap:** no native-Rust statevector simulator combines gate fusion + cache-blocking + SoA +
SIMD behind a modern pluggable GPU seam. That's our niche (headline stays
optimization+benchmarks; the modern-stack angle is secondary and unverified).

## 4. `amh-code` (Algorithmica HPC) — CPU optimization toolbox

Techniques (with rough impact for a bandwidth-bound complex-array kernel):

| Technique | CPU impact | Where in qsv |
|---|---|---|
| Cache-blocking / tiling (keep working set in L1/L2) | 2–3× | v0.9 high-stride kernel |
| SIMD complex multiply (intrinsics / portable) | 3–8× (AVX); ~1.3–1.8× (NEON) | v0.7 |
| Parallel **prefix-sum (scan)** for the sampling CDF | 5–10× | v0.10 sampling |
| Loop unrolling / ILP (hide FMA latency) | 2–4× | v0.7–v0.9 |
| Non-temporal stores (skip read-for-ownership on write-back) | 1.2–2× | v0.9 |
| Software prefetch (hide load latency) | 1.2–2× | v0.9 |
| `fp16` storage (½ bandwidth, precision risk) | 2× BW | maybe, large-N only |

Reference: `references/amh-code` and en.algorithmica.org/hpc.

## 5. Net plan implications

- GPU stays behind `Backend` (associated `type State`, on-device reductions, single
  `download`). Verified-second-backend (`RefBackend`) proves the seam now; cuStateVec /
  cuTile / Metal are future implementors.
- CPU optimization order follows the bandwidth-bound thesis: fusion → SoA+SIMD →
  cache-blocking → threading → micro-ops, each a benchmarkable milestone.
