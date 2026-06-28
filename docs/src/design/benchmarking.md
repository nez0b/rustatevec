# Benchmarking & profiling

Measurement is the whole point: the [optimization strategy](optimization.md) only means
something if each milestone's effect is quantified honestly. The `qsv-bench` crate holds the
[criterion](https://github.com/bheisler/criterion.rs) benchmarks and a profiling workload.

## What we measure

Throughput is reported in **amplitude-updates per second** — criterion's `Throughput::Elements`
set to \\(2^n\\) per gate. This normalizes across qubit counts and is the natural figure of
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

## Example: the v0.2 scalar kernel

A single Hadamard on a mid-range qubit, `BitShiftBackend`, Apple M3 Pro, single-threaded:

| qubits | state size | time/gate | throughput |
| --- | --- | --- | --- |
| 12 | 64 KB (L2) | ~3.95 µs | ~1.04 Gelem/s |
| 16 | 1 MB | ~70 µs | ~0.93 Gelem/s |
| 20 | 16 MB | ~1.07 ms | ~0.98 Gelem/s |
| 24 | 256 MB (DRAM) | ~17.2 ms | ~0.97 Gelem/s |

Two honest readings of this:

- **It is roughly flat** from cache-resident to DRAM-resident — ~1 Gelem/s throughout. At ~32
  bytes of traffic per update (read + write a 16-byte amplitude) that is ~32 GB/s, only ~20%
  of the M3 Pro's ~150 GB/s.
- That flatness means the *single-threaded scalar* kernel is currently limited by **per-element
  work** (index arithmetic + complex multiply), not memory bandwidth. The memory wall only
  becomes the binding constraint once SIMD (v0.7) and threading (v0.5) remove that arithmetic
  bottleneck — at which point fusion and cache behaviour become the levers that matter. The
  roofline plot below is built to show exactly that transition as milestones land.

This is the kind of result the project exists to surface: *measure first, then optimize the
thing that's actually binding.*

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
