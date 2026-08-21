//! AVX-512 + VNNI SIMD primitives for NNUE inference.
//!
//! Identical to the plain AVX-512 backend except for the byte dot-products,
//! where VNNI's `vpdpbusd` replaces the three-instruction maddubs emulation.
//! The glob re-export brings in every shared primitive; the two local
//! definitions shadow their emulated counterparts.

// Every function here is `unsafe fn` wrapping a single intrinsic call.
// The safety contract is lifted to the caller; no additional invariants inside.
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(dead_code)]

pub use super::avx512::*;

use std::arch::x86_64::*;

/// Dot product of unsigned bytes × signed bytes, accumulated into i32.
///
/// Native VNNI `vpdpbusd`: one instruction.
#[inline(always)]
pub unsafe fn dpbusd(acc: __m512i, u8s: __m512i, i8s: __m512i) -> __m512i {
    _mm512_dpbusd_epi32(acc, u8s, i8s)
}

/// Double dpbusd: two chained native `vpdpbusd` instructions.
#[inline(always)]
pub unsafe fn double_dpbusd(
    acc: __m512i,
    u8s1: __m512i,
    i8s1: __m512i,
    u8s2: __m512i,
    i8s2: __m512i,
) -> __m512i {
    _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(acc, u8s1, i8s1), u8s2, i8s2)
}
