//! AVX2 + AVX-VNNI SIMD primitives for NNUE inference.
//!
//! Identical to the AVX2 backend except for the byte dot-products, where the
//! VEX-encoded `vpdpbusd` of AVX-VNNI replaces the three-instruction maddubs
//! emulation. Every Intel desktop and laptop core since the 12th generation has
//! it without AVX-512. The glob re-export brings in every shared primitive; the
//! local definitions shadow their emulated counterparts.
//!
//! The results are those of the emulation: the activations stay within 0..=127,
//! so the i16 pair sums of `maddubs` never saturate and both forms add the same
//! four products.

// Every function here is `unsafe fn` wrapping a single intrinsic call.
// The safety contract is lifted to the caller; no additional invariants inside.
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(dead_code)]

pub use super::avx2::*;

/// Native `vpdpbusd` is ~5 cycles: three independent chains hide that latency.
pub const DPBUSD_CHAINS: usize = 3;

use std::arch::x86_64::*;

/// Dot product of unsigned bytes × signed bytes, accumulated into i32.
///
/// Native AVX-VNNI `vpdpbusd`: one instruction.
#[inline(always)]
pub unsafe fn dpbusd(acc: __m256i, u8s: __m256i, i8s: __m256i) -> __m256i {
    _mm256_dpbusd_avx_epi32(acc, u8s, i8s)
}

/// Double dpbusd: two chained native `vpdpbusd` instructions.
#[inline(always)]
pub unsafe fn double_dpbusd(
    acc: __m256i,
    u8s1: __m256i,
    i8s1: __m256i,
    u8s2: __m256i,
    i8s2: __m256i,
) -> __m256i {
    _mm256_dpbusd_avx_epi32(_mm256_dpbusd_avx_epi32(acc, u8s1, i8s1), u8s2, i8s2)
}
