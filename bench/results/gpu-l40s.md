# qsv GPU (CudaBackend) findings — NVIDIA L40S (Ada sm_89), 2026-06-29

Box: 2× L40S (46 GB, ~864 GB/s HBM each), CUDA 12.4, driver 550. `cudarc` 0.19 + NVRTC kernels.
Branch: `cudarc`. GPUs idle during measurement (CPU was under external load, GPU was not).

## Single-H throughput (Gelem/s; GB/s = ×32)

| n | target | GPU Gelem/s | GPU GB/s | vs CPU* |
| --- | --- | --- | --- | --- |
| 26 | low (q=1) | 19.9 | 638 | 6.2× |
| 26 | mid | 20.9 | 670 | 5.4× |
| 28 | low | 20.3 | 650 | 4.8× |
| 28 | mid | 21.0 | 673 | 3.5× |

~**670 GB/s ≈ 78% of HBM peak** on the naive kernel, steady across n and target qubit (the feared
high-q coalescing penalty is mild on Ada). *CPU comparison was contended (shared box under load).

## Circuit batching — kill the per-gate sync (milestone 1)

`execute()` launches every gate on one stream with **no per-gate `synchronize()`** (staged buffers
held alive until one final sync). QFT, total wall-clock:

| n | per-gate-sync | batched-execute | speedup |
| --- | --- | --- | --- |
| 18 | 17.19 ms | 3.87 ms | **4.44×** |
| 22 | 15.16 ms | 12.27 ms | 1.24× |
| 26 | 1263 ms | 1172 ms | 1.08× |

Batching amortizes per-gate launch+sync **latency**, which dominates at small/medium N; at large N
the per-gate kernel compute dwarfs it. CUDA graphs would extend the small-N win further.

## Profiling

- **nsys** (timeline, no special perms) — QFT kernel-time breakdown: `k_apply_diagonal` **60.9%**
  (2127 launches, cphase-heavy QFT), `k_apply_1q` 36.1%, `k_apply_mq` 2.7%, `k_init_basis` 0.4%.
  Host→device transfers (gate matrices/indices) are negligible vs kernel time. → for QFT, the
  diagonal kernel is the hot path.
- **ncu** (hardware counters) — **blocked** on this box: `ERR_NVGPUCTRPERM` (GPU performance-counter
  access needs root / `NVreg_RestrictProfilingToAdminUsers=0`). Roofline % from achieved GB/s
  instead (~78% of peak above).

## Cross-simulator: vs NVIDIA cuStateVec (cuQuantum) — QFT(n), f64

Same gate sequence (H + controlled-phase + bit-reversal SWAPs), same L40S, one device sync at the
end for both. cuStateVec via `cuquantum-python` 25.03 / cupy 14.1 (`bench/custatevec_compare.py`);
qsv via the batched `execute` (`qft_time` example). ms, lower is better.

| n | gates | cuStateVec | qsv | qsv / cuStateVec |
| --- | --- | --- | --- | --- |
| 18 | 180 | 2.64 | 2.57 | **0.97×** (parity) |
| 20 | 220 | 2.44 | 7.37 | 3.02× |
| 22 | 264 | 9.69 | 22.46 | 2.32× |
| 24 | 312 | 140.6 | 261.3 | 1.86× |
| 26 | 364 | 649 | 1160 | 1.79× |
| 28 | 420 | 2969 | 5366 | **1.81×** |

**Honest read:**
- **Large N (≥24, memory-bound): qsv is ~1.8× slower** than the tuned vendor library. The gap is
  kernel efficiency — cuStateVec has coalesced/shared-memory kernels and (likely) GPU-side gate
  handling that qsv's naive one-kernel-per-gate backend doesn't yet have. nsys says the diagonal
  (cphase) kernel is 60% of qsv's QFT time, so it's the first target.
- **Small N (≤20): launch-overhead-bound**, and cuStateVec's per-gate host cost is lower (qsv
  rebuilds + uploads each gate's matrix/index buffers per gate). qsv hits 3× here; **CUDA graphs /
  caching gate buffers** would close most of it.
- Being within **1.8×** of NVIDIA cuStateVec with a hand-written first-pass `cudarc` backend (and at
  parity by n=18) is a strong baseline; the remaining milestones target exactly this gap.

## Remaining (next milestones)

shared-memory-staged `mq` matrix; high-q coalescing (qubit reorder); **f32** path (2× throughput,
more qubits); on-device `sample`/`expectation` reductions; CUDA graphs; cuStateVec comparison.
