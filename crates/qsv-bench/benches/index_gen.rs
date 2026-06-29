//! Index-generation microbenchmark — the BMI2 (PEXT/PDEP) story in isolation.
//!
//! Reproduces the QuEST-#717 finding: PEXT/PDEP make bit-gather/scatter/insert **6–12× faster in
//! isolation**, but the gate kernels are memory-bandwidth-bound so the end-to-end win is ~1×
//! (these compute addresses, not memory traffic). Run the isolated comparison with:
//!   `RUSTFLAGS="-C target-cpu=native" cargo bench -p qsv-bench --features bmi2 --bench index_gen`
//! and the end-to-end check by toggling `--features bmi2` on `throughput`'s random/fusion benches.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use qsv_core::state::layout::{gather_bits, insert_zero_bits};

fn index_gen(c: &mut Criterion) {
    let qubits = [3u32, 7, 11, 19]; // a 4-qubit (fused) gate; ascending
    let n = 1usize << 22; // 4M index computations per iteration

    let mut g = c.benchmark_group("index_gen");
    g.throughput(Throughput::Elements(n as u64));

    g.bench_function("gather_scalar", |b| {
        b.iter(|| {
            let mut acc = 0usize;
            for i in 0..n {
                acc ^= gather_bits(black_box(i), &qubits);
            }
            black_box(acc)
        })
    });

    // NOTE: with `--features bmi2`, `insert_zero_bits` auto-dispatches to PDEP — so this arm is the
    // BMI2 path when the feature is on and the scalar loop when off (compare across the two runs).
    g.bench_function("insert_zero_bits", |b| {
        b.iter(|| {
            let mut acc = 0usize;
            for i in 0..n {
                acc ^= insert_zero_bits(black_box(i), &qubits);
            }
            black_box(acc)
        })
    });

    #[cfg(feature = "bmi2")]
    {
        use qsv_core::state::layout::bmi2::{gather_bits_bmi2, qubit_mask};
        let mask = qubit_mask(&qubits);
        g.bench_function("gather_bmi2_pext", |b| {
            b.iter(|| {
                let mut acc = 0usize;
                for i in 0..n {
                    // SAFETY: built with the `bmi2` feature on a BMI2 CPU (target-cpu=native).
                    acc ^= unsafe { gather_bits_bmi2(black_box(i), mask) };
                }
                black_box(acc)
            })
        });
    }

    g.finish();
}

criterion_group!(benches, index_gen);
criterion_main!(benches);
