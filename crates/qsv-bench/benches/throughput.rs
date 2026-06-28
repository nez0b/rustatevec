//! Gate-throughput and full-circuit benchmarks across the backend milestones.
//!
//! Throughput is reported in **amplitude-updates/second** (criterion `Throughput::Elements`
//! set to `2^n` per gate), which normalizes across qubit counts and is the natural metric for
//! a bandwidth-bound kernel. Run with `cargo bench -p qsv-bench`.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use qsv_core::circuits::{qft, random_circuit};
use qsv_core::gate;
use qsv_core::prelude::*;

/// Single 1-qubit (Hadamard) gate applied in place — the hot kernel in isolation.
/// `BitShiftBackend` across a range of sizes; the slower milestone backends at small `n`
/// only, to keep runtime sane while still showing the v0.0 → v0.2 progression.
fn single_gate(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_h_gate");
    let h = gate::h::<f64>();

    for &n in &[12u32, 16, 20, 24] {
        group.throughput(Throughput::Elements(1u64 << n)); // amplitudes touched per gate
        let q = n / 2; // a mid-range target qubit

        group.bench_with_input(BenchmarkId::new("bitshift", n), &n, |b, &n| {
            let backend = BitShiftBackend;
            let mut state = StateVector::<f64>::basis(n, 0);
            b.iter(|| {
                backend.apply(&mut state, &h, std::slice::from_ref(&q));
                black_box(&state);
            });
        });

        // Milestone comparison only where the slow backends are still cheap enough.
        if n <= 16 {
            group.bench_with_input(BenchmarkId::new("reshape", n), &n, |b, &n| {
                let backend = ReshapeBackend;
                let mut state = StateVector::<f64>::basis(n, 0);
                b.iter(|| {
                    backend.apply(&mut state, &h, std::slice::from_ref(&q));
                    black_box(&state);
                });
            });
            group.bench_with_input(BenchmarkId::new("oracle", n), &n, |b, &n| {
                let backend = RefBackend;
                let mut state = StateVector::<f64>::basis(n, 0);
                b.iter(|| {
                    backend.apply(&mut state, &h, std::slice::from_ref(&q));
                    black_box(&state);
                });
            });
        }
    }
    group.finish();
}

/// End-to-end QFT (controlled-phase heavy) on the in-place kernel.
fn full_qft(c: &mut Criterion) {
    let mut group = c.benchmark_group("qft");
    group.sample_size(20);
    for &n in &[10u32, 14, 18] {
        let circuit = qft(n);
        group.throughput(Throughput::Elements(
            circuit.ops().len() as u64 * (1u64 << n),
        ));
        group.bench_with_input(BenchmarkId::new("bitshift", n), &n, |b, _| {
            let backend = BitShiftBackend;
            b.iter(|| black_box(backend.execute(black_box(&circuit))));
        });
    }
    group.finish();
}

/// End-to-end random circuit (mixed 1q/2q) — a stand-in for random-circuit-sampling.
fn full_random(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_circuit");
    group.sample_size(20);
    for &n in &[12u32, 16, 20] {
        let circuit = random_circuit(n, 200, 0xC0FFEE);
        group.throughput(Throughput::Elements(200u64 * (1u64 << n)));
        group.bench_with_input(BenchmarkId::new("bitshift", n), &n, |b, _| {
            let backend = BitShiftBackend;
            b.iter(|| black_box(backend.execute(black_box(&circuit))));
        });
    }
    group.finish();
}

criterion_group!(benches, single_gate, full_qft, full_random);
criterion_main!(benches);
