//! Quick GPU throughput sanity for `CudaBackend` on the L40S, vs `CpuBackend::parallel()`.
//!
//!   cargo run -p qsv-cuda --features cuda --release --example throughput
//!
//! Measures single-Hadamard amplitude-update throughput at a few qubit counts, for a LOW target
//! qubit (coalesced device access) and a MID target qubit (strided, `2^q`-apart pairs — the
//! uncoalesced case the cuTile note flagged). Reports Gelem/s and the GPU/CPU ratio. Times state
//! evolution only (init excluded); each GPU `apply` currently synchronizes (correctness-first).

#[cfg(feature = "cuda")]
fn run() {
    use std::time::Instant;

    use qsv_core::gate;
    use qsv_core::prelude::*;
    use qsv_cuda::CudaBackend;

    fn bench_gpu(gpu: &CudaBackend, n: u32, q: u32, reps: usize) -> f64 {
        let mut st = gpu.alloc(n);
        gpu.init_basis(&mut st, 0);
        let h = gate::h::<f64>();
        gpu.apply(&mut st, &h, std::slice::from_ref(&q)); // warmup
        let t = Instant::now();
        for _ in 0..reps {
            gpu.apply(&mut st, &h, std::slice::from_ref(&q));
        }
        (reps as f64 * (1u64 << n) as f64) / t.elapsed().as_secs_f64() / 1e9
    }

    fn bench_cpu(n: u32, q: u32, reps: usize) -> f64 {
        let cpu = CpuBackend::parallel();
        let mut st = StateVector::<f64>::basis(n, 0);
        let h = gate::h::<f64>();
        cpu.apply(&mut st, &h, std::slice::from_ref(&q));
        let t = Instant::now();
        for _ in 0..reps {
            cpu.apply(&mut st, &h, std::slice::from_ref(&q));
        }
        (reps as f64 * (1u64 << n) as f64) / t.elapsed().as_secs_f64() / 1e9
    }

    let gpu = CudaBackend::new(0).expect("init CUDA device 0");
    println!("L40S CudaBackend vs CpuBackend::parallel() — single-H throughput (Gelem/s)");
    println!("  n  target        gpu G     cpu G   gpu/cpu   gpu GB/s");
    for &n in &[24u32, 26, 28] {
        let reps = if n >= 28 { 30 } else { 100 };
        for &(label, q) in &[("low ", 1u32), ("mid ", n / 2)] {
            let g = bench_gpu(&gpu, n, q, reps);
            let c = bench_cpu(n, q, reps);
            let ratio = g / c;
            let gbps = g * 32.0; // read+write of re+im = 32 bytes/amplitude
            println!("{n:>3}  {label}      {g:>7.2}   {c:>7.2}   {ratio:>5.2}x   {gbps:>6.0}");
        }
    }
}

#[cfg(not(feature = "cuda"))]
fn run() {
    eprintln!("build with --features cuda to run the GPU throughput example");
}

fn main() {
    run();
}
