//! SIMD abstraction layer for NNUE inference.
//!
//! Each backend module exports the same API: lane constants + primitive functions
//! (`add_i16`, `dpbusd`, `packus`, ...). The kernels in `nnue::kernels` are
//! monomorphized once per backend via `use <backend> as simd;`, so a kernel body
//! written against `simd::*` compiles against every register width.
//!
//! On x86-64 every backend is compiled unconditionally — which backend *runs* is
//! decided per instantiation in `nnue::kernels` (compile-time today, so the set of
//! instantiated kernels still follows `#[cfg(target_feature)]`). Other
//! architectures keep a single compile-time backend:
//! - NEON (aarch64, always available — e.g. Raspberry Pi 5)
//! - simd128 (wasm32, `-C target-feature=+simd128`)
//! - Scalar fallback (no SIMD features)

#[cfg(target_arch = "x86_64")]
pub mod avx2;
#[cfg(target_arch = "x86_64")]
pub mod avx512;
#[cfg(target_arch = "x86_64")]
pub mod avx512vnni;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub mod neon;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub mod wasm128;

pub mod scalar;
