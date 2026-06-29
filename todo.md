# qsv — handoff & TODO

Context and next steps for continuing on a different machine (HPC Intel CPU + CUDA GPU). Full
design/benchmarks live in the book: <https://nez0b.github.io/rustatevec/> (or `mdbook serve docs`).

## 0. Session update — 2026-06-28, on the Intel AVX-512 + L40S box (NEW)

Ran on **2× Xeon Gold 6526Y** (32c/64t, 2 NUMA, AVX-512+BMI2+FMA, 503 GB) + **2× L40S** (Ada
sm_89, 46 GB, CUDA 12.4). Vendor refs re-cloned under `_local/` (gitignored). Results in
`bench/results/SUMMARY-xeon.md` (+ raw `.txt`).

- **Fusion does NOT wash out at n=18 here** — QFT(18) fused 13.6 ms vs unfused 24.3 ms = **1.78×**
  (2.73× at n=14). The M3 "wash" was a cache-capacity artifact. (v0.9 cache-blocked `apply_mq`
  should widen it further, but it's no longer the blocker the handoff assumed.)
- **f64x8 (nightly `std::simd`, AVX-512) implemented** behind the `nightly-simd` feature
  (`backend/simd.rs`). At cache-resident **n=16 it beats auto-vec scalar 1.28× and `wide::f64x4`
  1.33×** (3.99 vs 3.11 vs 3.00 G/s); past cache the win vanishes (bandwidth-bound) — textbook
  roofline. The v0.7 "null" was partly the `f64x4::from([..])` array-gather (vs real `from_slice`).
- **NUMA is the dominant CPU effect at large N.** Single H, n=24: pinning to one socket
  (`taskset -c 0-15,32-47`, no numactl on box) = **2.29 G/s vs 0.94 G/s best unpinned, 0.33 G/s at
  64 threads.** Default rayon pool (64) is *worse* than 16–32. Action items below.
- **GPU backend STARTED and correctness-validated** (`crates/qsv-cuda`, cudarc 0.19 + NVRTC):
  `CudaBackend` passes the differential suite vs the oracle on the L40S (`--features cuda`). First
  naive kernel already sustains **~670 GB/s ≈ 78% of HBM peak**, 2.2–2.9× the (NUMA-limited) CPU.
- **cuTile investigated and rejected for now** (`docs/src/research/cutile-investigation.md`):
  real but needs CUDA 13.2+ (box is 12.4) and abstracts away the coalescing/SMEM control our
  strided non-GEMM kernel needs. → backend built on cudarc. Toolchain: nightly installed for f64x8.

**New build/test/run:**
```bash
cargo test -p qsv-cuda --features cuda                              # GPU differential suite (needs L40S)
cargo run -p qsv-cuda --features cuda --release --example throughput  # GPU vs CPU throughput
cargo +nightly test --features qsv-core/nightly-simd               # AVX-512 f64x8 path
cargo run -p qsv-cuda --features cuda --release --example throughput
```
Workspace stays green without CUDA (`qsv-cuda` compiles empty under default features).

**Next (carried forward):** GPU §4.4 optimizations — `execute`-batching / CUDA graphs (each `apply`
still `synchronize()`s), shared-mem-staged `mq` matrix, on-device probability *reduction* (currently
`k_abs2` + full dtoh). CPU — NUMA-aware first-touch / interleaved alloc + cap rayon to one socket
(revisit `PARALLEL_MIN_PAIRS`); roofline via STREAM/`perf` IMC; native-vs-SSE2 AVX-uplift number;
`perf record` kernel attribution. Then cross-sim comparison (QuEST/Aer/qsim, cuStateVec for GPU).

## 1. Where things stand (v0.0 – v0.8, all on `main`)

A Cargo workspace: `qsv-core` (library), `qsv-cli` (`qsv` runner), `qsv-bench` (criterion +
`qsv-profile`). Everything is generic over a `Real` scalar (`f64` default, `f32` available),
stored **Structure-of-Arrays** (`re[]`, `im[]`) for shuffle-free SIMD.

**Backends** (all implement the `Backend` trait — the pluggable seam; all validated against the
oracle by the differential suite in `crates/qsv-core/tests/equivalence.rs`):

| Backend | What | Milestone |
| --- | --- | --- |
| `RefBackend` | naive gather/scatter oracle (independent of the bit-shift trick) | v0.0 |
| `ReshapeBackend` | block, out-of-place | v0.1 |
| `BitShiftBackend` | in-place `insert_zero_bit` pair kernel | v0.2 |
| `CpuBackend` | bounds-check-free nested-block 1q + `apply_mq` + diagonal + rayon | v0.3–v0.6 |
| `SimdBackend` | `CpuBackend` + `wide::f64x4` 1q kernel (`f64` only) | v0.7 |
| `fusion::fuse` | `Circuit -> Circuit` greedy gate fusion (≤ `max_qubits`) | v0.8 |

**Key kernels** (`crates/qsv-core/src/backend/cpu.rs`): `apply_1q` (nested-block halves via
`chunks_mut`/`par_chunks_mut`), `apply_mq` (`insert_zero_bits` gather/scatter; threaded via a
`SyncPtr` over disjoint blocks), `apply_diagonal` (single sequential pass). Index math is in
`crates/qsv-core/src/state/layout.rs`.

**Findings carried over (from the M3 Pro):**
- The kernel is **memory-bandwidth-bound**. Threaded `CpuBackend` at 24 qubits ≈ **127 GB/s ≈
  85% of the M3 Pro's ~150 GB/s** — at the bandwidth roof.
- **v0.7 SIMD = null result** on the 1q kernel (~0% at all sizes): bandwidth-bound, not
  arithmetic-bound. *On the Intel box this should change* (see §2).
- **v0.8 fusion ≈ 1.8×** on QFT(14) but **washes out at QFT(18)**: fused H+phase blocks use the
  general `apply_mq` kernel whose scattered gather/scatter erodes effective bandwidth at large
  N. This is the motivation for v0.9.

**Build / test / run:**
```bash
cargo test --workspace                          # oracle + differential + fusion-equivalence
cargo run --release --bin qsv -- 24             # GHZ(24)
cargo bench -p qsv-bench --bench throughput     # gate-throughput + fusion benchmarks
cargo clippy --workspace --all-targets -- -D warnings
cargo build -p qsv-core --no-default-features --features f64   # dependency-free core
```
CI (GitHub Actions) runs fmt+clippy+test on ubuntu(x86) and macos(arm64), builds the book, and
deploys it to Pages. Features: `f64`/`f32`, `parallel` (rayon, default), `simd` (wide, default),
`nightly-simd` (std::simd — **declared but not yet implemented**).

## 2. On the Intel HPC box (CPU) — what to verify

> **Critical:** by default Rust targets baseline `x86-64` (SSE2 only), so `wide::f64x4` lowers
> to 2×SSE, *not* AVX. To actually exercise AVX2/AVX-512 + FMA + BMI2, build with:
> ```bash
> RUSTFLAGS="-C target-cpu=native" cargo bench -p qsv-bench --bench throughput
> ```

1. **Correctness first:** `cargo test --workspace` (the differential suite is platform-agnostic;
   it should pass identically). Then `RUSTFLAGS="-C target-cpu=native" cargo test` to exercise
   the AVX codegen path.
2. **SIMD, redux.** Re-run the `single_h_gate` benches comparing `cpu_serial` vs `simd_serial`
   with `target-cpu=native`. On AVX2 (4×f64 loads) and especially **AVX-512** (8×f64, 64-byte
   loads) SIMD *should* now beat scalar where the working set is cache-resident. If it still
   doesn't, the kernel is bandwidth-bound there too — confirm via the roofline (below). Note:
   `wide` maxes at `f64x4`; for **AVX-512 width (`f64x8`)** implement the `nightly-simd` feature
   using `std::simd::f64x8` (the `Real::Simd` seam was reserved for this).
3. **Threading scaling.** On a many-core Intel part, sweep thread counts
   (`RAYON_NUM_THREADS=1,2,4,…`) and plot speedup vs cores for `cpu_parallel`. Watch for **NUMA**:
   pin with `numactl --cpunodebind=0 --membind=0` for single-socket numbers; cross-socket will
   show the first-touch allocation penalty (the state is allocated on one node). Revisit
   `PARALLEL_MIN_PAIRS` (currently `1<<12`) — the right threshold differs per machine.
4. **Roofline.** Run a STREAM-triad microbench to get the box's real peak bandwidth, then
   compute achieved GB/s = `Gelem/s × 32` for the threaded large-N kernel. Use `likwid-perfctr -g
   MEM_DP ./target/release/qsv-profile 26 40` for measured GB/s, or `perf stat`. Confirm the
   large-N kernel sits near the bandwidth roof; confirm fusion moves QFT rightward (higher AI).
5. **Fusion at scale.** Re-run `fusion_qft` (and add a `random_circuit` fused/unfused variant).
   The n=18 wash on the Mac is hypothesized to be `apply_mq` access pattern — see if more cores +
   bigger caches change the crossover. This directly informs v0.9.
6. **Cross-simulator comparison** (the credibility deliverable). Same circuits (export the
   `circuits::{qft, random_circuit}` definitions to QASM3) through Qiskit-Aer (`statevector`,
   fusion on/off), qsim (via Cirq), and QuEST (cleanest C apples-to-apples). Time only state
   evolution; match precision and thread count; same box. Report wall-clock + gate-throughput,
   **including where qsv loses**, explained by the roofline.
7. **BMI2 (towards v0.10).** Intel has fast `PEXT`/`PDEP`; a feature-gated BMI2 path for the
   gather/scatter index generation is worth measuring here (expect small end-to-end gains —
   memory-bound — consistent with QuEST #717).

## 3. Open CPU work

- **v0.9 — cache-block `apply_mq`** (next, highest leverage). The fused multi-qubit kernel's
  scattered gather/scatter caps fusion's benefit at large N. Restructure it for locality
  (block the outer loop to keep each block's `2^m` amplitudes + the gate matrix in cache; add
  software prefetch and non-temporal stores for the write-back). Target: make QFT(18+) fusion
  show the n=14 speedup.
- **v0.10 — micro-ops:** ILP/unrolling in the inner kernels, x86 BMI2 index gen (feature-gated),
  **alias-table O(1) sampling** + parallel prefix-sum CDF (sampling isn't implemented yet — add
  `Backend::sample`/`expectation`), and the `nightly-simd` (`std::simd`, incl. AVX-512 `f64x8`).
- **v1.0 —** roofline plots committed under `bench/`, cross-sim comparison figures, `bench/plot.py`.
- Possible: controlled-gate subspace specialization (iterate only the `2^(N-k)` active subspace,
  per Yao's `IterControl`) — currently controlled gates go through the dense `apply_mq`.

## 4. GPU backend plan (CUDA / cuTile-rs)

The architecture is **ready** for this: implement `Backend<f64>` for a new `CudaBackend` in a
feature-gated, optional `qsv-cuda` crate (only builds when a CUDA toolchain is present). Nothing
in the circuit/gate/fusion layers needs to change — that's the whole point of the seam.

### 4.0 Verify the stack FIRST (do not skip)
The `cuTile` / `cuTile-rs` maturity claims in `docs/src/research/gpu-and-rust.md` are
**UNVERIFIED** (post-knowledge-cutoff agent output). Before committing:
- Confirm `NVlabs/cutile-rs` (or equivalent) actually exists, its crates.io status, required
  CUDA version + GPU compute capability, and whether it suits a **memory-bound, non-GEMM** kernel
  (cuTile is tile/tensor-core oriented; our gate apply is bandwidth-bound elementwise/strided —
  it may or may not be a natural fit).
- **Fallbacks if cuTile-rs is immature:** `cudarc` (ergonomic, actively maintained Rust CUDA —
  recommended default), `cust`/`rust-cuda`, or hand-written CUDA C kernels behind a thin FFI.
  For the Mac, a `MetalBackend` via `wgpu`/`metal-rs` is the local-GPU analogue.
- Decision rule: pick whatever lets you write the gate kernels with control over coalescing and
  shared memory. Start with the simplest that compiles and is correct; optimize later.

### 4.1 The seam (what `CudaBackend` implements)
- `type State` = an **opaque device handle** (e.g. two `CudaSlice<f64>` for `re`/`im`, or one
  interleaved buffer). No host slice ever escapes.
- `alloc` → device alloc; `init_basis` → memset 0 then set one amplitude to 1 on device.
- `apply` → dispatch to device kernels (diagonal / 1q / mq), mirroring `CpuBackend`.
- **Reductions on-device:** implement `probabilities` (and the not-yet-existing `sample` /
  `expectation`) as device reductions so only `download` crosses to host (tests/CLI only).
- Override `execute` to **batch** the whole circuit on one stream (and CUDA graphs) to amortize
  per-gate launch latency — this matters far more on GPU than CPU.

### 4.2 Kernel design (from the research notes)
- **Diagonal gate:** elementwise `ψ_i *= phase(sub(i))` — fully coalesced, trivial, fastest.
- **1-qubit gate:** thread `t` → amplitude pair `(a0, a1)` via `insert_zero_bit`. Coalescing
  depends on the target-qubit stride: **low target qubit → consecutive `a0` → coalesced**;
  **high target qubit → `2^q` stride → uncoalesced**. Mitigation (cuStateVec's trick): swap the
  target qubit to a low index (permute), or stage a tile in shared memory.
- **Multi-qubit / fused gate:** stage the `2^m × 2^m` matrix in **shared memory**, reuse it
  across the block's amplitude groups; coalesce the amplitude loads. This is exactly where
  **fusion pays double on GPU** — fewer kernel launches *and* fewer passes over HBM (also
  bandwidth-bound, ~60–80% of peak achievable).
- **cuTile mapping (if used):** tiles = chunks of the statevector staged to shared memory; the
  gate matrix is a reused tile. The mq shared-memory-staging kernel is the natural cuTile target.

### 4.3 Validation & benchmarking (reuse what exists)
- The differential harness already works for any backend: `run(&backend, &circuit)` calls
  `download`, so `CudaBackend` drops straight into `tests/equivalence.rs` (compare device result
  to the `RefBackend` oracle). **Wire it in from day one** — correctness before speed.
- Benchmark against **cuStateVec** (cuQuantum) on the same circuits; report honestly.

### 4.4 Multi-GPU (later)
Partition the state by high index bits across GPUs; gates on "global" qubits require GPU↔GPU
exchange over NVLink/MPI — mirrors QuEST's pairwise rank-exchange (see
`docs/src/research/cpu-simulators.md`). Out of scope until single-GPU is solid.

### 4.5 Concrete first steps
1. Verify cuTile-rs (or choose `cudarc`). 2. Add optional `qsv-cuda` crate; implement
   `CudaBackend::{alloc, init_basis, apply_diagonal, apply_1q, apply_mq, probabilities, download}`
   with the *simplest correct* kernels. 3. Add it to the differential tests (gated on the CUDA
   feature). 4. Optimize: coalescing + shared-mem matrix + `execute` batching + fusion. 5.
   Benchmark vs cuStateVec; add to the book.
