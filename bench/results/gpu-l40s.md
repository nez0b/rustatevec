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

## Remaining (next milestones)

shared-memory-staged `mq` matrix; high-q coalescing (qubit reorder); **f32** path (2× throughput,
more qubits); on-device `sample`/`expectation` reductions; CUDA graphs; cuStateVec comparison.
