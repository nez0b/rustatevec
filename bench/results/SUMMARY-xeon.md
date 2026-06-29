# qsv benchmark findings — 2× Xeon Gold 6526Y (AVX-512), 2026-06-28

Box: 2× Intel Xeon Gold 6526Y (32 cores / 64 threads, 2 NUMA nodes), AVX-512 + BMI2 + FMA,
L1d 48K, L2 2 MB/core, L3 38.4 MB/socket, 503 GB RAM. Built with `RUSTFLAGS="-C target-cpu=native"`.
Throughput = amplitude-updates/s. Raw: `throughput-native.txt`.

## UPDATE (v0.8.1) — NUMA-aware first-touch allocation

`StateVector::zeros` now faults its lazily-zeroed pages **in parallel** (one contiguous chunk per
worker, `state/mod.rs`), so each page is first-touched on the NUMA node of the worker that will
process it. Combined with **per-kernel threading thresholds** (`MIN_PAIRS_LIGHT=1<<16` for 1q/diag,
`MIN_PAIRS_HEAVY=1<<12` for the mq kernel), single-H, n=24:

| | cpu_parallel n=20 | cpu_parallel n=24 | n=16 (cache-resident) |
| --- | --- | --- | --- |
| before (serial alloc, 1<<12) | 5.80 | 1.53 | 0.96 (over-threaded) |
| **after (first-touch + thresholds)** | **7.00** | **9.31 (~298 GB/s)** | **2.96** (serial fallback) |

**6.1× at n=24** — the state is now distributed across *both* sockets' memory controllers with each
thread accessing local pages, so unpinned-both-sockets (9.31) even beats single-socket `taskset`
pinning (7.07). No `core_affinity`/`numactl` needed; the pinned pool was dropped as unnecessary.
Serial did not regress (n=24: 0.33→0.75). Small-N over-threading fixed by the light-kernel threshold.

**Honest fusion correction:** with the corrected baseline (unfused QFT now runs *serial* at small N
instead of over-threading), fusion's apparent **2.73× at n=14 was an artifact** — fused QFT(14) is
actually neutral/negative (328M vs unfused 391M) because at cache-resident N the light unfused
kernels beat the heavier fused `apply_mq`. Fusion's real win is the **bandwidth-bound regime:
1.70× at n=18** (3.00 vs 1.77 G/s). This is the physically-correct story.

## UPDATE (v0.10) — BMI2 (PEXT/PDEP) index generation: the bandwidth-bound anchor

`bmi2` feature, runtime-detected, on `insert_zero_bits` (→ `PDEP(index, !mask)`) plus PEXT/PDEP
gather/scatter helpers. Index-generation microbench (`index_gen`) and end-to-end (`fusion_qft`):

| | scalar | BMI2 | speedup |
| --- | --- | --- | --- |
| gather (4-qubit), isolated | 0.93 G/s | **3.86 G/s (PEXT)** | **4.2×** |
| insert_zero_bits, isolated | 0.91 G/s | **3.88 G/s (PDEP)** | **4.3×** |
| **fused QFT-18, end-to-end** | 3.029 G/s | 3.014 G/s | **0.995× (no change)** |

**This is the empirical anchor of the whole bandwidth-bound thesis** (and reproduces QuEST #717):
BMI2 makes index generation ~4× faster *in isolation*, but the gate kernel is memory-bandwidth-
bound, so it **washes out completely end-to-end** — index math computes *addresses*, not *memory
traffic*. Validated against the oracle (differential suite passes with `--features bmi2`).

## UPDATE (v0.9) — locality micro-ops: all measured null/negative (the bandwidth-bound tax)

Implemented and measured the v0.9 micro-optimizations; **none helped** — exactly what the
roofline predicts once first-touch already puts the threaded kernel at ~298 GB/s. Kept the simpler
code; the negative results are the deliverable.

- **Non-temporal stores:** *not applicable.* The gate kernels are in-place read-modify-write, so
  every cache line is already resident from the read — NT stores only help write-only streams
  (they'd bypass cache and force re-fetch). Not implemented.
- **ILP 2× unroll (f64x8 1q kernel):** *regressed* (simd_serial n=16 ~3.99 → 2.88 G/s). The kernel
  is not FMA-latency-bound — LLVM already pipelines the 1× loop, and doubling live vectors added
  register pressure. Reverted.
- **Software prefetch (mq scattered gather):** *null/negative* (fused QFT-18: ON ≤ OFF in the same
  run). Extra prefetch instructions don't help a bandwidth-bound kernel and can evict useful lines.
  Reverted. (Measured under external load on the shared box; relative ON-vs-OFF still valid.)
- **Cache-blocking the mq kernel:** *structurally N/A.* Each block's working set — `2^m` amplitudes
  (≤256 B) + the `2^(2m)` matrix (≤4 KB for m=4) — already fits L1; the only remaining lever is the
  inter-block stride, fixed by qubit positions and unreorderable without a transpose pass (extra
  bandwidth = counterproductive when bandwidth-bound). Not implemented.

Net v0.9: the locality wins on this kernel come from **first-touch (6.1×) and fusion (1.70× at
n=18)** — data movement, not address/compute micro-ops. Three independent null results (SIMD width
past cache, BMI2 end-to-end, these micro-ops) triangulate the same conclusion.

## single H gate (target qubit = n/2), Gelem/s — *(original run, pre-first-touch; see UPDATE above)*

| n | bitshift | cpu_serial | cpu_parallel | simd_serial (f64x4) | simd_parallel |
| --- | --- | --- | --- | --- | --- |
| 12 | 0.78 | 2.91 | 2.91 | 2.80 | 2.80 |
| 16 | 0.78 | 3.10 | 0.96 | 3.00 | 0.97 |
| 20 | 0.73 | 1.79 | **5.80** | 1.85 | 5.80 |
| 24 | 0.21 | 0.33 | **1.53** | 0.23 | 1.53 |

Notes:
- **SIMD null/regression (key):** `target-cpu=native` lets LLVM auto-vectorize the *scalar*
  `cpu_serial` kernel to AVX-512, so the explicit `wide::f64x4` (256-bit, manual array-gather
  `f64x4::from([..])`) does not beat it — and at n=24 is *slower* (0.23 vs 0.33). Motivates the
  explicit **`std::simd::f64x8`** path (nightly-simd) using real vector loads (`from_slice`).
- **Cache cliff:** serial 3.10 (n=16, 1 MB ⊂ L2) → 1.79 (n=20, 16 MB ⊂ L3) → 0.33 (n=24, 256 MB → RAM).
- **NUMA cliff:** parallel 5.80 G/s @ n=20 (~185 GB/s, partly L3) → 1.53 G/s @ n=24 (~49 GB/s).
  State is first-touched on one socket; cross-socket access caps bandwidth. Needs pinning /
  interleave (no numactl on box → `taskset`); revisit `PARALLEL_MIN_PAIRS`.
- Threading only helps once the state exceeds cache (n≥20); at n=16 rayon overhead makes parallel
  3× *slower* than serial.

## f64x8 (nightly std::simd, AVX-512) vs f64x4 vs auto-vec scalar — simd_serial, Gelem/s

`cargo +nightly bench --features qsv-core/nightly-simd`. cpu_serial (auto-vec scalar) is the
control and matches the stable run (n=16: 3.11 both), so cross-run comparison is valid. Raw:
`throughput-f64x8.txt`.

| n | working set | scalar (auto-vec) | f64x4 (wide) | **f64x8 (std::simd)** | f64x8 vs scalar |
| --- | --- | --- | --- | --- | --- |
| 12 | 64 KB (L1/L2) | 2.9 | 2.80 | 2.97 | 1.02× |
| 16 | 1 MB (⊂ L2) | 3.11 | 3.00 | **3.99** | **1.28×** |
| 20 | 16 MB (⊂ L3) | 1.82 | 1.85 | 1.92 | 1.05× |
| 24 | 256 MB (RAM) | 0.31 | 0.23 | 0.32 | 1.03× |

**This is the SIMD win the M3 Pro (128-bit NEON) could never show.** At cache-resident n=16, explicit
512-bit `f64x8` with real vector loads (`Simd::from_slice`) beats both LLVM auto-vectorized scalar
(**1.28×**) and the 256-bit `wide::f64x4` (**1.33×**). Past cache (n≥20) the win collapses to ~0 as
the kernel becomes bandwidth-bound — a textbook roofline crossover that *confirms* the
memory-bandwidth-bound thesis: SIMD width only buys speed while the working set is cache-resident.
The v0.7 "null result" was real but Mac-specific; the lesson is **f64x8 + true vector loads** (the
old `f64x4::from([a,b,c,d])` array-gather was itself part of the v0.7 null).

## fusion (QFT), CpuBackend::parallel

| n | unfused | fused | speedup |
| --- | --- | --- | --- |
| 10 | 140 µs | 197 µs | 0.71× (overhead dominates) |
| 14 | 9.59 ms | 3.52 ms | **2.73×** |
| 18 | 24.3 ms | 13.6 ms | **1.78×** |

**Fusion does NOT wash out at n=18 here** (contrast the M3 Pro, where 1.8×@14 vanished by 18). The
larger L3 + more cores keep the fused `apply_mq` ahead. v0.9 (cache-blocked `apply_mq`) should
widen this further, but the Mac "wash" was a cache-capacity artifact, not fundamental.

## threading + NUMA — single H gate, cpu_parallel, n=24 (256 MB, RAM-bound)

`RAYON_NUM_THREADS` sweep, short measurement window (warm-up 1s / measure 2s → noisier absolutes;
the *shape* is the finding). Raw: `threading-numa.txt`.

| config | Gelem/s |
| --- | --- |
| 1 thread | 0.14 |
| 4 | 0.23 |
| 8 | 0.64 |
| **16 (best unpinned)** | **0.94** |
| 32 | 0.68 |
| 64 (all logical) | 0.33 |
| **socket0 pinned (`taskset -c 0-15,32-47`, 32 thr, 1 NUMA)** | **2.29** |

**NUMA first-touch is the dominant effect.** The state is allocated/first-touched on one node;
letting rayon spread across both sockets makes most accesses cross-socket, so throughput *falls*
past 16 threads (0.94 → 0.33 at 64). Pinning the threads to the socket that owns the memory recovers
**2.29 G/s — 2.4× the best unpinned config and ~7× the naive 64-thread run.** Hyperthreads also hurt
(they share the cache/memory ports). Takeaways for the code: (a) NUMA-aware first-touch /
interleaved allocation, or pin + bind; (b) default rayon pool of 64 is *worse* than 16–32 here —
revisit `PARALLEL_MIN_PAIRS` and consider capping the pool to one socket. (No `numactl` on the box;
used `taskset`.)

## GPU — CudaBackend (cudarc + NVRTC) on L40S vs CpuBackend::parallel(), single H gate

`cargo run -p qsv-cuda --features cuda --release --example throughput`. GB/s = Gelem/s × 32.
L40S HBM peak ≈ 864 GB/s.

| n | target qubit | GPU Gelem/s | GPU GB/s | CPU (64-thr) Gelem/s | GPU/CPU |
| --- | --- | --- | --- | --- | --- |
| 24 | low (q=1) | 20.6 | 660 | 7.2 | 2.85× |
| 24 | mid (q=n/2) | 20.4 | 654 | 9.3 | 2.19× |
| 26 | low | 20.8 | 667 | 7.3 | 2.87× |
| 26 | mid | 21.0 | 671 | 7.5 | 2.81× |
| 28 | low | 20.9 | 669 | 7.2 | 2.91× |
| 28 | mid | 21.0 | 673 | 7.9 | 2.67× |

**~670 GB/s = ~78% of the L40S HBM roofline on the first, correctness-first kernel** — squarely in
the 60–80% band the cuTile note predicted, and with no shared-memory/coalescing tuning yet. The
feared **high-target-qubit coalescing penalty is mild on Ada** at these sizes (low vs mid q within
~1%): the large L2 + memory system absorb the `2^q`-strided pair access for a full-array pass.
GPU is **2.2–2.9× the NUMA-limited 64-thread CPU** (and ~9× the single-socket-pinned 2.29 G/s). Each
GPU `apply` still synchronizes (correctness-first) — `execute`-batching / CUDA graphs are headroom.

## qft (bitshift serial) / random_circuit (bitshift serial)

- qft/bitshift: ~0.15 G/s flat across n=10/14/18 (controlled-phase heavy, serial pairing kernel).
- random_circuit/bitshift: ~0.31 G/s across n=12/16/20.

## TODO (remaining Phase B)
- f64x8 head-to-head (Phase C).
- RAYON_NUM_THREADS sweep + `taskset -c 0-15` single-socket vs unpinned (quantify NUMA).
- native-vs-SSE2 build to quantify the AVX auto-vec uplift.
- `perf record` kernel attribution + roofline (STREAM triad / perf IMC).
