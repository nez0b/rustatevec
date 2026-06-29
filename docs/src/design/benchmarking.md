# Benchmarking & profiling

Measurement is the whole point: the [optimization strategy](optimization.md) only means
something if each milestone's effect is quantified honestly. The `qsv-bench` crate holds the
[criterion](https://github.com/bheisler/criterion.rs) benchmarks and a profiling workload.

## What we measure

Throughput is reported in **amplitude-updates per second** — criterion's `Throughput::Elements`
set to $2^n$ per gate. This normalizes across qubit counts and is the natural figure of
merit for a bandwidth-bound kernel (it converts directly to effective GB/s: multiply by the
bytes touched per amplitude).

Three benchmark groups:

| Group | What it isolates |
| --- | --- |
| `single_h_gate` | one 1-qubit gate in place — the hot kernel, across `n` and across backends |
| `qft` | end-to-end QFT (controlled-phase heavy) |
| `random_circuit` | end-to-end mixed 1q/2q circuit (random-circuit-sampling stand-in) |

The `single_h_gate` group runs the slow milestone backends (`oracle`, `reshape`) only at small
`n`, so a single run shows the v0.0 → v0.1 → v0.2 progression without taking forever.

## Running

```bash
cargo bench -p qsv-bench --bench throughput              # everything
cargo bench -p qsv-bench --bench throughput -- bitshift  # filter by name (regex)

# Quick look (short sampling) while iterating:
cargo bench -p qsv-bench --bench throughput -- \
    'single_h_gate/bitshift' --warm-up-time 0.3 --measurement-time 1.0 --sample-size 10
```

criterion writes HTML reports (with plots) to `target/criterion/`.

## Milestone results so far

A single Hadamard on a mid-range qubit, throughput in amplitude-updates/sec, Apple M3 Pro
(11 cores). Higher is better.

| qubits | state | `bitshift` (v0.2) | `cpu_serial` (v0.3/4) | `cpu_parallel` (v0.5) |
| --- | --- | --- | --- | --- |
| 12 | 64 KB | ~1.0 Gelem/s | ~2.0 Gelem/s | ~2.0 Gelem/s¹ |
| 20 | 16 MB | 0.98 Gelem/s | 1.91 Gelem/s | 6.42 Gelem/s |
| 24 | 256 MB | 0.99 Gelem/s | 1.80 Gelem/s | 3.97 Gelem/s |

¹ below the threading threshold (`n < 13`), so `parallel` runs serially — by design.

What each milestone bought, read honestly:

- **v0.2 → v0.3/4 (`cpu_serial`): ~1.9×.** Removing per-amplitude bounds checks (via iterator
  zips that lower to checked-free code — *no `unsafe` needed*) and the cache-friendly
  nested-block walk. Squarely in the predicted 1.3–2× regime.
- **v0.3/4 → v0.5 (`cpu_parallel`): ~2–3.4×** more, depending on size.

### The kernel just became bandwidth-bound — and we can prove it

Convert the v0.5 throughput to effective memory bandwidth: each update reads and writes a
16-byte complex amplitude = 32 bytes of traffic.

$$
\text{n=24:}\quad 3.97\ \text{Gelem/s} \times 32\ \text{B} \approx \mathbf{127\ GB/s} \approx \mathbf{85\%}\ \text{of the M3 Pro's} \sim 150\ \text{GB/s peak.}
$$

At 256 MB (pure DRAM) the threaded kernel is running into the **memory wall**: ~85% of peak
bandwidth, exactly the regime the [thesis](optimization.md) predicted. At n=20 (16 MB,
partly L2-resident) it reports ~205 GB/s — *above* DRAM peak — because some traffic is served
from cache.

This is the inflection the project was built to surface. Earlier, the single-threaded scalar
kernel was a flat ~1 Gelem/s — **compute-bound** on per-element work, not memory. Removing that
work (v0.3) and parallelizing it (v0.5) has now pushed the DRAM-resident case to ~85% of the
bandwidth roof. Per the **stop criterion** (≥ 70–80% of STREAM bandwidth), micro-optimizing
*this* kernel further is noise — the next real win must change the bytes-moved equation, i.e.
**gate fusion** (v0.8). The data wrote the roadmap.

## v0.6–v0.8: three findings, reported honestly

**v0.6 diagonal fast path** — Z/S/T/PHASE/RZ/CZ/RZZ run a single sequential pass (one
complex-mul per amplitude, no `2^q` stride). Real win for phase-heavy circuits (QFT/QAOA) and
high-target-qubit gates; modest at large N where everything is bandwidth-bound.

**v0.7 SIMD — a measured null result.** `wide::f64x4` on the 1-qubit kernel: **~0% at every
size** (n=12 2.01 vs 2.04, n=20 1.89 vs 1.89, n=24 1.79 vs 1.80 Gelem/s vs the scalar
`CpuBackend`). The 1q gate's arithmetic intensity (~0.13 FLOP/byte) is so low that widening the
ALUs changes nothing — the cache-friendly scalar kernel is already bandwidth-bound at L2 *and*
DRAM. This is the roofline taken to its conclusion, not a bug. SIMD is expected to pay where
arithmetic intensity is high: the fused multi-qubit matvecs below, and x86 **AVX-512** (64-byte
loads vs NEON's 16) — flagged for the Intel box (see `todo.md`).

**v0.8 gate fusion — the headline, with a caveat the benchmark surfaced.** Fused vs unfused
QFT on `CpuBackend::parallel()`:

| qubits | unfused | fused | speedup |
| --- | --- | --- | --- |
| 14 | 4.43 ms | 2.46 ms | **~1.8×** |
| 18 | 23.3 ms | 23.4 ms | ~1.0× |

Fusion cuts passes over memory, so at n=14 it's a clean ~1.8×. At n=18 it washes out — and
*why* is the interesting part: unfused QFT's controlled-phases already use the fast diagonal
kernel, whereas fused H+phase blocks run the general `apply_mq` kernel, whose **scattered
gather/scatter** erodes effective bandwidth at large N enough to cancel the pass reduction. So
the fusion win is real but currently **gated by the multi-qubit kernel's access pattern** —
exactly what **v0.9 (cache-blocking `apply_mq`)** targets. The benchmark didn't just validate
fusion; it located the next bottleneck.

## Roofline methodology

1. Measure the machine's empirical peak bandwidth with a STREAM-triad microbench (don't trust
   the spec sheet).
2. For each kernel, compute arithmetic intensity (FLOP / byte touched) and plot it against
   achieved performance.
3. As milestones land, watch points move: SIMD/threading pushes the scalar kernel up toward
   the bandwidth roof; **fusion moves points rightward** (higher arithmetic intensity) — the
   visual proof of the central thesis.

## Profiling

`qsv-bench` ships a long-running workload, `qsv-profile`, for sampling profilers:

```bash
cargo build --release -p qsv-bench --bin qsv-profile

# macOS / Linux — samply (Firefox-profiler UI)
samply record ./target/release/qsv-profile 22 40

# macOS — Xcode Instruments
cargo instruments -t "Time Profiler" --release --bin qsv-profile -- 22 40

# Linux — perf with memory-bandwidth counters
perf stat -e cache-misses,mem_load_retired.l3_miss ./target/release/qsv-profile 22 40
likwid-perfctr -g MEM_DP ./target/release/qsv-profile 22 40   # reports achieved GB/s
```

Arguments are `<qubits> <layers> <reps>`; the workload runs `layers·n` random gates `reps`
times so the profiler gets enough samples to attribute time to the kernel.

## Cross-simulator comparison (planned)

The fair-comparison harness against Qiskit-Aer, qsim, QuEST, and spinoza lands with the later
milestones. The protocol: one circuit definition exported to every tool; time only the
statevector evolution (exclude Python import / transpile / JIT); match precision and thread
count; same physical box for head-to-heads; report both wall-clock and gate-throughput; and
**show where we lose**, explained by the roofline.
