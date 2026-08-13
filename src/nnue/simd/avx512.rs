//! AVX-512 SIMD primitives for NNUE inference.
//!
//! 512-bit registers: 32 × i16, 16 × i32, 16 × f32.

// Every function here is `unsafe fn` wrapping a single intrinsic call.
// The safety contract is lifted to the caller; no additional invariants inside.
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(dead_code)]

#[allow(unused_imports)]
use std::arch::x86_64::*;

/// Number of i16 elements per SIMD register.
pub const I16_LANES: usize = 32;
/// Number of i32 elements per SIMD register.
pub const I32_LANES: usize = 16;
/// Number of f32 elements per SIMD register.
pub const F32_LANES: usize = 16;

// ============================================================
// i16 operations (accumulator, FT activation)
// ============================================================

#[inline(always)]
pub unsafe fn add_i16(a: __m512i, b: __m512i) -> __m512i {
    _mm512_add_epi16(a, b)
}

#[inline(always)]
pub unsafe fn sub_i16(a: __m512i, b: __m512i) -> __m512i {
    _mm512_sub_epi16(a, b)
}

#[inline(always)]
pub unsafe fn zeroed_i16() -> __m512i {
    _mm512_setzero_si512()
}

#[inline(always)]
pub unsafe fn splat_i16(a: i16) -> __m512i {
    _mm512_set1_epi16(a)
}

#[inline(always)]
pub unsafe fn clamp_i16(x: __m512i, min: __m512i, max: __m512i) -> __m512i {
    _mm512_min_epi16(_mm512_max_epi16(x, min), max)
}

#[inline(always)]
pub unsafe fn min_i16(a: __m512i, b: __m512i) -> __m512i {
    _mm512_min_epi16(a, b)
}

#[inline(always)]
pub unsafe fn shift_left_i16<const SHIFT: u32>(a: __m512i) -> __m512i {
    _mm512_slli_epi16(a, SHIFT)
}

#[inline(always)]
pub unsafe fn mul_high_i16(a: __m512i, b: __m512i) -> __m512i {
    _mm512_mulhi_epi16(a, b)
}

/// Pack two vectors of i16 → one vector of u8 (unsigned saturation).
#[inline(always)]
pub unsafe fn packus(a: __m512i, b: __m512i) -> __m512i {
    _mm512_packus_epi16(a, b)
}

/// Fix lane interleaving after packus (AVX-512 crosses 128-bit lanes).
#[inline(always)]
pub unsafe fn permute(a: __m512i) -> __m512i {
    _mm512_permutexvar_epi64(
        _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7),
        a,
    )
}

/// Load I16_LANES (32) i8 values from `ptr` and sign-extend to i16.
///
/// Used for threat accumulator: i8 weights → i16 additions.
#[inline(always)]
pub unsafe fn load_i8_as_i16(ptr: *const i8) -> __m512i {
    _mm512_cvtepi8_epi16(_mm256_loadu_si256(ptr as *const __m256i))
}

/// Multiply pairs of adjacent i16 and accumulate into i32.
/// `result[i] = a[2i] * b[2i] + a[2i+1] * b[2i+1]`
#[inline(always)]
pub unsafe fn madd_i16(a: __m512i, b: __m512i) -> __m512i {
    _mm512_madd_epi16(a, b)
}

/// Horizontal sum of an i32 vector → scalar i32.
#[inline(always)]
pub unsafe fn horizontal_sum_i32(x: __m512i) -> i32 {
    _mm512_reduce_add_epi32(x)
}

// ============================================================
// i32 operations (sparse L1, NNZ)
// ============================================================

#[inline(always)]
pub unsafe fn zeroed_i32() -> __m512i {
    _mm512_setzero_si512()
}

#[inline(always)]
pub unsafe fn splat_i32(a: i32) -> __m512i {
    _mm512_set1_epi32(a)
}

#[inline(always)]
pub unsafe fn add_i32(a: __m512i, b: __m512i) -> __m512i {
    _mm512_add_epi32(a, b)
}

/// Dot product of unsigned bytes × signed bytes, accumulated into i32.
///
/// Uses native VNNI `vpdpbusd` when available (1 instruction),
/// otherwise emulates with maddubs+madd+add (3 instructions).
#[inline(always)]
pub unsafe fn dpbusd(acc: __m512i, u8s: __m512i, i8s: __m512i) -> __m512i {
    #[cfg(target_feature = "avx512vnni")]
    {
        _mm512_dpbusd_epi32(acc, u8s, i8s)
    }
    #[cfg(not(target_feature = "avx512vnni"))]
    {
        let pairwise = _mm512_maddubs_epi16(u8s, i8s);
        let widened = _mm512_madd_epi16(pairwise, _mm512_set1_epi16(1));
        _mm512_add_epi32(acc, widened)
    }
}

/// Double dpbusd: process two pairs in one call.
///
/// With VNNI: two chained native `vpdpbusd` instructions.
/// Without: combine pairwise products before widening (saves one madd).
#[inline(always)]
pub unsafe fn double_dpbusd(
    acc: __m512i,
    u8s1: __m512i,
    i8s1: __m512i,
    u8s2: __m512i,
    i8s2: __m512i,
) -> __m512i {
    #[cfg(target_feature = "avx512vnni")]
    {
        _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(acc, u8s1, i8s1), u8s2, i8s2)
    }
    #[cfg(not(target_feature = "avx512vnni"))]
    {
        let pw1 = _mm512_maddubs_epi16(u8s1, i8s1);
        let pw2 = _mm512_maddubs_epi16(u8s2, i8s2);
        let widened = _mm512_madd_epi16(_mm512_add_epi16(pw1, pw2), _mm512_set1_epi16(1));
        _mm512_add_epi32(acc, widened)
    }
}

/// Extract bitmask of non-zero i32 lanes (16 bits, one per lane).
#[inline(always)]
pub unsafe fn nnz_bitmask(x: __m512i) -> u16 {
    _mm512_cmpgt_epi32_mask(x, _mm512_setzero_si512())
}

// ============================================================
// f32 operations (L2, L3)
// ============================================================

#[inline(always)]
pub unsafe fn zero_f32() -> __m512 {
    _mm512_setzero_ps()
}

#[inline(always)]
pub unsafe fn splat_f32(a: f32) -> __m512 {
    _mm512_set1_ps(a)
}

/// Element-wise multiply: a * b.
#[inline(always)]
pub unsafe fn mul_f32(a: __m512, b: __m512) -> __m512 {
    _mm512_mul_ps(a, b)
}

/// Fused multiply-add: a * b + c.
#[inline(always)]
pub unsafe fn mul_add_f32(a: __m512, b: __m512, c: __m512) -> __m512 {
    _mm512_fmadd_ps(a, b, c)
}

/// Convert i32 vector to f32 vector.
#[inline(always)]
pub unsafe fn convert_to_f32(a: __m512i) -> __m512 {
    _mm512_cvtepi32_ps(a)
}

/// Clamp f32 vector to [min, max].
#[inline(always)]
pub unsafe fn clamp_f32(x: __m512, min: __m512, max: __m512) -> __m512 {
    _mm512_min_ps(_mm512_max_ps(x, min), max)
}

/// Horizontal sum of one f32 vector → scalar f32.
///
/// AVX-512 has a native reduce instruction.
#[inline(always)]
pub unsafe fn horizontal_sum(x: [__m512; 1]) -> f32 {
    _mm512_reduce_add_ps(x[0])
}

/// Number of __m512 vectors needed for horizontal_sum input.
pub const HSUM_VECS: usize = 1;
