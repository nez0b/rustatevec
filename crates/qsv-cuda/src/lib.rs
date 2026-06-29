//! # qsv-cuda — the GPU backend
//!
//! A `CudaBackend` implementing the core [`Backend`](qsv_core::prelude::Backend) trait, built on
//! [`cudarc`](https://docs.rs/cudarc) with kernels compiled at runtime via **NVRTC** (see
//! `src/kernels.cu`). The trait is the seam: nothing in the circuit / gate / fusion layers changes
//! — the statevector lives in device memory and only [`download`](Backend::download) crosses back
//! to the host.
//!
//! Why hand-written cudarc kernels rather than cuTile-rs: our kernel is memory-bound, strided, and
//! non-GEMM, and we need explicit control over coalescing / shared memory — and cuTile needs CUDA
//! 13.2+ while this box is on 12.4. See `docs/src/research/cutile-investigation.md`.
//!
//! These are the **simplest correct** kernels (correctness-first): a diagonal fast path, a 1-qubit
//! pair kernel, and a general m-qubit gather/scatter kernel. Coalescing / shared-memory-staged
//! matrix / `execute` batching are follow-up optimizations.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;

use qsv_core::complex::Cplx;
use qsv_core::gate::DenseGate;
use qsv_core::prelude::Backend;
use qsv_core::state::StateVector;

const KERNELS: &str = include_str!("kernels.cu");
const BLOCK: u32 = 256;

/// Device-resident statevector: SoA `re`/`im` in HBM.
pub struct CudaState {
    re: CudaSlice<f64>,
    im: CudaSlice<f64>,
    n_qubits: u32,
}

/// CUDA backend: owns the context/stream and the compiled kernel module.
pub struct CudaBackend {
    stream: Arc<CudaStream>,
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    f_init: CudaFunction,
    f_1q: CudaFunction,
    f_diag: CudaFunction,
    f_mq: CudaFunction,
    f_abs2: CudaFunction,
}

impl CudaBackend {
    /// Initialize the GPU at `ordinal`, compile the kernels (NVRTC), and load the functions.
    pub fn new(ordinal: usize) -> Result<Self, cudarc::driver::DriverError> {
        let ctx = CudaContext::new(ordinal)?;
        let stream = ctx.default_stream();
        let ptx = compile_ptx(KERNELS).expect("NVRTC compile of kernels.cu");
        let module = ctx.load_module(ptx)?;
        let f_init = module.load_function("k_init_basis")?;
        let f_1q = module.load_function("k_apply_1q")?;
        let f_diag = module.load_function("k_apply_diagonal")?;
        let f_mq = module.load_function("k_apply_mq")?;
        let f_abs2 = module.load_function("k_abs2")?;
        Ok(Self {
            stream,
            ctx,
            module,
            f_init,
            f_1q,
            f_diag,
            f_mq,
            f_abs2,
        })
    }

    /// Grid/block config covering `n` threads (1-D).
    fn cfg(n: u64) -> LaunchConfig {
        let grid = n.div_ceil(BLOCK as u64) as u32;
        LaunchConfig {
            grid_dim: (grid.max(1), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    fn htod_f64(&self, v: &[f64]) -> CudaSlice<f64> {
        self.stream.clone_htod(v).expect("htod f64")
    }
    fn htod_i32(&self, v: &[i32]) -> CudaSlice<i32> {
        self.stream.clone_htod(v).expect("htod i32")
    }
}

impl Backend<f64> for CudaBackend {
    type State = CudaState;

    fn alloc(&self, n_qubits: u32) -> CudaState {
        let dim = 1usize << n_qubits;
        CudaState {
            re: self.stream.alloc_zeros::<f64>(dim).expect("alloc re"),
            im: self.stream.alloc_zeros::<f64>(dim).expect("alloc im"),
            n_qubits,
        }
    }

    fn init_basis(&self, state: &mut CudaState, basis: usize) {
        let dim = (1u64) << state.n_qubits;
        let basis = basis as u64;
        let mut b = self.stream.launch_builder(&self.f_init);
        b.arg(&mut state.re)
            .arg(&mut state.im)
            .arg(&basis)
            .arg(&dim);
        unsafe { b.launch(Self::cfg(dim)) }.expect("launch init_basis");
        self.stream.synchronize().expect("sync");
    }

    fn apply(&self, state: &mut CudaState, gate: &DenseGate<f64>, qubits: &[u32]) {
        let dim = (1u64) << state.n_qubits;

        if gate.is_diagonal() {
            let m = qubits.len();
            let sub = 1usize << m;
            let (mut dr, mut di) = (vec![0f64; sub], vec![0f64; sub]);
            for s in 0..sub {
                let c = gate.at(s, s);
                dr[s] = c.re;
                di[s] = c.im;
            }
            let d_dr = self.htod_f64(&dr);
            let d_di = self.htod_f64(&di);
            let q: Vec<i32> = qubits.iter().map(|&x| x as i32).collect();
            let d_q = self.htod_i32(&q);
            let mi = m as i32;
            let mut b = self.stream.launch_builder(&self.f_diag);
            b.arg(&mut state.re)
                .arg(&mut state.im)
                .arg(&d_dr)
                .arg(&d_di)
                .arg(&d_q)
                .arg(&mi)
                .arg(&dim);
            unsafe { b.launch(Self::cfg(dim)) }.expect("launch diagonal");
        } else if qubits.len() == 1 {
            let q = qubits[0];
            let g00 = gate.at(0, 0);
            let g01 = gate.at(0, 1);
            let g10 = gate.at(1, 0);
            let g11 = gate.at(1, 1);
            let n_pairs = dim >> 1;
            let mut b = self.stream.launch_builder(&self.f_1q);
            b.arg(&mut state.re)
                .arg(&mut state.im)
                .arg(&q)
                .arg(&g00.re)
                .arg(&g00.im)
                .arg(&g01.re)
                .arg(&g01.im)
                .arg(&g10.re)
                .arg(&g10.im)
                .arg(&g11.re)
                .arg(&g11.im)
                .arg(&n_pairs);
            unsafe { b.launch(Self::cfg(n_pairs)) }.expect("launch 1q");
        } else {
            let m = qubits.len();
            let sub = 1usize << m;
            let g = gate.row_major();
            let (mut mr, mut mi) = (vec![0f64; sub * sub], vec![0f64; sub * sub]);
            for (k, c) in g.iter().enumerate() {
                mr[k] = c.re;
                mi[k] = c.im;
            }
            let d_mr = self.htod_f64(&mr);
            let d_mi = self.htod_f64(&mi);
            let qorig: Vec<i32> = qubits.iter().map(|&x| x as i32).collect();
            let mut qsorted = qorig.clone();
            qsorted.sort_unstable();
            let d_qsorted = self.htod_i32(&qsorted);
            let d_qorig = self.htod_i32(&qorig);
            let mi32 = m as i32;
            let blocks = dim >> m;
            let mut b = self.stream.launch_builder(&self.f_mq);
            b.arg(&mut state.re)
                .arg(&mut state.im)
                .arg(&d_mr)
                .arg(&d_mi)
                .arg(&d_qsorted)
                .arg(&d_qorig)
                .arg(&mi32)
                .arg(&blocks);
            unsafe { b.launch(Self::cfg(blocks)) }.expect("launch mq");
        }
        // Correctness-first: synchronize so the staged host buffers above stay alive until the
        // launch completes. Batching/streaming is a later optimization.
        self.stream.synchronize().expect("sync");
    }

    fn amplitude(&self, state: &CudaState, index: usize) -> Cplx<f64> {
        let sv = self.download(state);
        sv.amplitude(index)
    }

    fn probabilities(&self, state: &CudaState) -> Vec<f64> {
        let dim = state.re.len();
        let mut out = self.stream.alloc_zeros::<f64>(dim).expect("alloc prob");
        let dim_u = dim as u64;
        let mut b = self.stream.launch_builder(&self.f_abs2);
        b.arg(&state.re).arg(&state.im).arg(&mut out).arg(&dim_u);
        unsafe { b.launch(Self::cfg(dim_u)) }.expect("launch abs2");
        self.stream.synchronize().expect("sync");
        self.stream.clone_dtoh(&out).expect("dtoh prob")
    }

    fn download(&self, state: &CudaState) -> StateVector<f64> {
        self.stream.synchronize().expect("sync");
        let host_re = self.stream.clone_dtoh(&state.re).expect("dtoh re");
        let host_im = self.stream.clone_dtoh(&state.im).expect("dtoh im");
        let mut sv = StateVector::<f64>::zeros(state.n_qubits);
        let (re, im) = sv.parts_mut();
        re.copy_from_slice(&host_re);
        im.copy_from_slice(&host_im);
        sv
    }
}
