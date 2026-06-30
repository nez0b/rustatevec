//! Time qsv's `CudaBackend` evolving QFT(n) from |0…0⟩ (batched, one stream). Machine-readable
//! output for the cuStateVec comparison driver (`bench/custatevec_compare.py`).
//!
//!   cargo run -p qsv-cuda --features cuda --release --example qft_time -- <n> [reps]

#[cfg(feature = "cuda")]
fn main() {
    use std::time::Instant;

    use qsv_core::circuits::qft;
    use qsv_core::prelude::*;
    use qsv_cuda::CudaBackend;

    let mut args = std::env::args().skip(1);
    let n: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(20);
    let reps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(5);

    let gpu = CudaBackend::new(0).expect("init CUDA device 0");
    let circ = qft(n);
    let gates = circ.ops().len();

    let _ = gpu.execute(&circ); // warmup (JIT, allocations, first-touch)

    let t = Instant::now();
    for _ in 0..reps {
        let _ = gpu.execute(&circ);
    }
    let ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
    let gelem_s = gates as f64 * (1u64 << n) as f64 / (ms / 1e3) / 1e9;
    println!("qsv n={n} gates={gates} ms={ms:.3} gelem_s={gelem_s:.3}");
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with --features cuda");
}
