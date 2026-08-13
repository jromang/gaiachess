//! SIMD abstraction layer for NNUE inference.
//!
//! Compile-time dispatch via `#[cfg(target_feature)]`:
//! - AVX-512 (znver4, `target-cpu=znver4`)
//! - AVX2 (most x86-64, `target-cpu=native`)
//! - NEON (aarch64, always available — e.g. Raspberry Pi 5)
//! - Scalar fallback (no SIMD features)
//!
//! All backends export the same API: lane constants + primitive functions.
//! Code using `simd::add_i16()` etc. works regardless of backend.

// Priority: AVX-512 > AVX2 > NEON > scalar
#[cfg(target_feature = "avx512f")]
mod avx512;
#[cfg(target_feature = "avx512f")]
pub use avx512::*;

#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
mod avx2;
#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
pub use avx2::*;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub use neon::*;

#[cfg(not(any(
    target_feature = "avx512f",
    target_feature = "avx2",
    all(target_arch = "aarch64", target_feature = "neon"),
)))]
mod scalar;
#[cfg(not(any(
    target_feature = "avx512f",
    target_feature = "avx2",
    all(target_arch = "aarch64", target_feature = "neon"),
)))]
pub use scalar::*;
