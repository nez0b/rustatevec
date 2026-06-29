# GPU, Rust & HPC landscape

Landscape research for the (deferred) GPU backend and the CPU optimization toolbox.

## cuStateVec / cuda-quantum

NVIDIA's cuStateVec applies gates with the same bit-index pairing as the CPU, mapped to the
GPU:

- **thread → amplitude-pair** mapping; the gate matrix is staged into **shared memory** and
  reused across a block's pairs;
- **coalescing** depends on the target-qubit stride — low-order targets coalesce, high-order
  targets stride badly and are mitigated by qubit reordering (the GPU analog of QuEST's
  distributed exchange);
- multi-GPU partitions the state by index bits; gates on "global" qubits need GPU↔GPU exchange.

The statevector lives in HBM (the bottleneck); achieved bandwidth is ~60–80% of peak — the
**same bandwidth-bound story as the CPU**. cuda-quantum's `nvq++` lowers circuits to
`custatevecApplyMatrix` calls.

## cuTile / cuTile-rs — ✅ verified (2026-06)

cuTile is **real and shipping**: `NVlabs/cutile-rs` (crate `cutile`, Rust 1.89+) and
`NVIDIA/cutile-python`, with the paper *Fearless Concurrency on the GPU*
([arXiv 2606.15991](https://arxiv.org/abs/2606.15991)). Both clones are under `_local/`. The full
memory-architecture study and the build-vs-buy decision live in the
[cuTile memory architecture](cutile-investigation.md) note; the essentials:

- **Requirements:** GPU `sm_80`+ (Ada `sm_89` ✅), **CUDA 13.2+ (13.3 recommended)**, driver
  **r580+**, Linux. **Our box is CUDA 12.4 / driver 550 → cuTile cannot run here without a
  toolkit + driver upgrade** (sysadmin-level). cuTile is early-stage ("expect API breakage").
- **Memory model:** programmer picks *tile* shapes; the compiler owns coalescing, shared-memory
  staging, TMA pipelining, and tensor-core lowering. Tiles live in registers; tensors live in HBM;
  load/store moves between them. There is **no native complex type**, and **strided access** is the
  pattern cuTile's own docs flag as most likely to lose bandwidth — with no manual remedy.
- **Decision:** for our memory-bound, strided, complex, non-GEMM gate kernel, the performance
  levers (coalescing, `__shared__` staging, `__shfl` pair exchange, vectorized complex loads) are
  exactly what cuTile abstracts away. → **`CudaBackend` is built on `cudarc` + NVRTC** (hand-written
  CUDA C, full memory control, runs on CUDA 12.4 today). cuTile stays as a future validation
  reference and for the contiguous/low-target-qubit cases.

## Rust quantum-sim ecosystem — the gap

| Project | Approach | GPU | Notes |
| --- | --- | --- | --- |
| `spinoza` | CPU SIMD + rayon | ✗ | closest peer (~30q); the prior Rust art to compare against |
| `qoqo`/`roqoqo` (HQS) | circuit DSL, delegates | ✗ | no native sim engine |
| `qip`, `qasmsim`, `quantum` | builders / educational / QASM | ✗ | not perf-focused |
| `qoqo-quest` | Rust wrapper over QuEST (C) | via QuEST | not native Rust |

No native-Rust statevector simulator combines gate fusion + cache-blocking + SoA + SIMD behind
a modern pluggable GPU seam — that is qsv's niche.

## `amh-code` — CPU optimization toolbox

Techniques from *Algorithms for Modern Hardware*, with rough impact for a bandwidth-bound
complex-array kernel:

| Technique | CPU impact | Where in qsv |
| --- | --- | --- |
| cache-blocking / tiling | 2–3× | v0.9 high-stride kernel |
| SIMD complex multiply | 3–8× (AVX) / 1.3–1.8× (NEON) | v0.7 |
| parallel prefix-sum (scan) for the sampling CDF | 5–10× | v0.10 sampling |
| loop unrolling / ILP | 2–4× | v0.7–v0.9 |
| non-temporal stores | 1.2–2× | v0.9 |
| software prefetch | 1.2–2× | v0.9 |

## Net plan implications

- GPU stays behind `Backend` (associated `type State`, on-device reductions, single
  `download`); the `RefBackend` second implementation proves the seam today.
- CPU optimization order follows the bandwidth-bound thesis: fusion → SoA+SIMD →
  cache-blocking → threading → micro-ops, each a benchmarkable milestone.
