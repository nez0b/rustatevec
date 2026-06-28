//! A long-running, representative workload for sampling profilers.
//!
//! Build in release and attach a profiler, e.g.:
//! ```text
//! cargo build --release -p qsv-bench --bin qsv-profile
//! samply record ./target/release/qsv-profile 22 40        # macOS / Linux
//! cargo instruments -t "Time Profiler" --release --bin qsv-profile -- 22 40   # macOS
//! ```
//! Args: `<qubits=22> <layers=40> <reps=5>`. Each layer is ~`n` random gates, so the run
//! executes `layers*n` gates `reps` times — enough samples to attribute time to the kernel.

use qsv_core::circuits::random_circuit;
use qsv_core::prelude::*;
use std::hint::black_box;

fn main() {
    let mut args = std::env::args().skip(1);
    let n: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(22);
    let layers: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(40);
    let reps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(5);

    let gates = layers * n as usize;
    let circuit = random_circuit(n, gates, 0xC0FFEE);
    let backend = BitShiftBackend;

    let mut sink = 0.0f64;
    for _ in 0..reps {
        let state = backend.execute(black_box(&circuit));
        sink += state.amplitude(0).re; // keep the work observable
    }

    println!("profile: n={n} gates={gates} reps={reps} sink={sink:e}");
}
