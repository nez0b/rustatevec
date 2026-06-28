//! # qsv-core
//!
//! Core of a high-performance quantum **statevector** simulator.
//!
//! ## Design north star
//! Statevector simulation is **memory-bandwidth-bound**, not compute-bound: applying a
//! 1-qubit gate touches the whole `2^N` amplitude array doing only ~2 complex multiplies
//! per 16-byte amplitude (arithmetic intensity ≈ 0.13 FLOP/byte). Every optimization in
//! this crate is therefore justified by "does it reduce bytes moved per gate, or raise
//! arithmetic intensity per byte?" — see `docs/` for the full design + research.
//!
//! ## v0.0 (this milestone)
//! Establishes the public types and a deliberately simple, independently-implemented
//! [`RefBackend`](backend::reference::RefBackend) that serves as the **correctness oracle**
//! for every optimized kernel that follows. It also validates the [`Backend`](backend::Backend)
//! trait — the pluggable seam behind which a future GPU backend will live.
//!
//! ```
//! use qsv_core::prelude::*;
//! // Bell state: H(0); CX(0,1)  ->  (|00> + |11>)/√2
//! let mut c = Circuit::<f64>::new(2);
//! c.h(0).cx(0, 1);
//! let s = RefBackend.execute(&c);
//! let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
//! assert!((s.amplitude(0b00).re - inv_sqrt2).abs() < 1e-12);
//! assert!((s.amplitude(0b11).re - inv_sqrt2).abs() < 1e-12);
//! ```

// We `deny` (not `forbid`) so individual hot-path modules can opt back into `unsafe`
// later with a localized `#[allow(unsafe_code)]` + `// SAFETY:` justification.
#![deny(unsafe_code)]
#![cfg_attr(feature = "nightly-simd", feature(portable_simd))]

pub mod backend;
pub mod circuit;
pub mod complex;
pub mod gate;
pub mod real;
pub mod state;

pub mod prelude {
    //! Ergonomic glob import: `use qsv_core::prelude::*;`
    pub use crate::backend::reference::RefBackend;
    pub use crate::backend::Backend;
    pub use crate::circuit::Circuit;
    pub use crate::complex::Cplx;
    pub use crate::gate::DenseGate;
    pub use crate::real::Real;
    pub use crate::state::StateVector;
}
