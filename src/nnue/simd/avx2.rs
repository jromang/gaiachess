//! AVX2 SIMD primitives for NNUE inference.
//!
//! 256-bit registers: 16 × i16, 8 × i32, 8 × f32.

// Every function here is `unsafe fn` wrapping a single intrinsic call.
// The safety contract is lifted to the caller; no additional invariants inside.
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(dead_code)]

#[allow(unused_imports)]
use std::arch::x86_64::*;

/// Number of i16 elements per SIMD register.
pub const I16_LANES: usize = 16;
/// Number of i32 elements per SIMD register.
pub const I32_LANES: usize = 8;
/// Number of f32 elements per SIMD register.
pub const F32_LANES: usize = 8;

// ============================================================
// i16 operations (accumulator, FT activation)
// ============================================================

#[inline(always)]
pub unsafe fn add_i16(a: __m256i, b: __m256i) -> __m256i {
    _mm256_add_epi16(a, b)
}

#[inline(always)]
pub unsafe fn sub_i16(a: __m256i, b: __m256i) -> __m256i {
    _mm256_sub_epi16(a, b)
}

#[inline(always)]
pub unsafe fn zeroed_i16() -> __m256i {
    _mm256_setzero_si256()
}

#[inline(always)]
pub unsafe fn splat_i16(a: i16) -> __m256i {
    _mm256_set1_epi16(a)
}

#[inline(always)]
pub unsafe fn clamp_i16(x: __m256i, min: __m256i, max: __m256i) -> __m256i {
    _mm256_min_epi16(_mm256_max_epi16(x, min), max)
}

#[inline(always)]
pub unsafe fn min_i16(a: __m256i, b: __m256i) -> __m256i {
    _mm256_min_epi16(a, b)
}

#[inline(always)]
pub unsafe fn shift_left_i16<const SHIFT: i32>(a: __m256i) -> __m256i {
    _mm256_slli_epi16(a, SHIFT)
}

#[inline(always)]
pub unsafe fn mul_high_i16(a: __m256i, b: __m256i) -> __m256i {
    _mm256_mulhi_epi16(a, b)
}

/// Pack two vectors of i16 → one vector of u8 (unsigned saturation).
#[inline(always)]
pub unsafe fn packus(a: __m256i, b: __m256i) -> __m256i {
    _mm256_packus_epi16(a, b)
}

/// Fix lane interleaving after packus (AVX2 crosses 128-bit lanes).
#[inline(always)]
pub unsafe fn permute(a: __m256i) -> __m256i {
    _mm256_permute4x64_epi64::<0b11_01_10_00>(a)
}

/// Load I16_LANES (16) i8 values from `ptr` and sign-extend to i16.
///
/// Used for threat accumulator: i8 weights → i16 additions.
#[inline(always)]
pub unsafe fn load_i8_as_i16(ptr: *const i8) -> __m256i {
    _mm256_cvtepi8_epi16(_mm_loadu_si128(ptr as *const __m128i))
}

/// Multiply pairs of adjacent i16 and accumulate into i32.
/// `result[i] = a[2i] * b[2i] + a[2i+1] * b[2i+1]`
#[inline(always)]
pub unsafe fn madd_i16(a: __m256i, b: __m256i) -> __m256i {
    _mm256_madd_epi16(a, b)
}

/// Horizontal sum of an i32 vector → scalar i32.
#[inline(always)]
pub unsafe fn horizontal_sum_i32(x: __m256i) -> i32 {
    let hi128 = _mm256_extracti128_si256::<1>(x);
    let lo128 = _mm256_castsi256_si128(x);
    let sum128 = _mm_add_epi32(lo128, hi128);
    let hi64 = _mm_unpackhi_epi64(sum128, sum128);
    let sum64 = _mm_add_epi32(sum128, hi64);
    let hi32 = _mm_shuffle_epi32::<0b_01>(sum64);
    let sum32 = _mm_add_epi32(sum64, hi32);
    _mm_cvtsi128_si32(sum32)
}

// ============================================================
// i32 operations (sparse L1, NNZ)
// ============================================================

#[inline(always)]
pub unsafe fn zeroed_i32() -> __m256i {
    _mm256_setzero_si256()
}

#[inline(always)]
pub unsafe fn splat_i32(a: i32) -> __m256i {
    _mm256_set1_epi32(a)
}

#[inline(always)]
pub unsafe fn add_i32(a: __m256i, b: __m256i) -> __m256i {
    _mm256_add_epi32(a, b)
}

/// Dot product of unsigned bytes × signed bytes, accumulated into i32.
///
/// Emulates VNNI `dpbusd`: u8 × i8 → pairwise i16 → widened i32 → accumulated.
#[inline(always)]
pub unsafe fn dpbusd(acc: __m256i, u8s: __m256i, i8s: __m256i) -> __m256i {
    let pairwise = _mm256_maddubs_epi16(u8s, i8s);
    let widened = _mm256_madd_epi16(pairwise, _mm256_set1_epi16(1));
    _mm256_add_epi32(acc, widened)
}

/// Double dpbusd: process two pairs of (u8, i8) in one call.
/// Saves one `madd_epi16` by combining before widening.
#[inline(always)]
pub unsafe fn double_dpbusd(
    acc: __m256i,
    u8s1: __m256i,
    i8s1: __m256i,
    u8s2: __m256i,
    i8s2: __m256i,
) -> __m256i {
    let pw1 = _mm256_maddubs_epi16(u8s1, i8s1);
    let pw2 = _mm256_maddubs_epi16(u8s2, i8s2);
    let widened = _mm256_madd_epi16(_mm256_add_epi16(pw1, pw2), _mm256_set1_epi16(1));
    _mm256_add_epi32(acc, widened)
}

/// Extract bitmask of non-zero i32 lanes (8 bits, one per lane).
#[inline(always)]
pub unsafe fn nnz_bitmask(x: __m256i) -> u16 {
    let gt = _mm256_cmpgt_epi32(x, _mm256_setzero_si256());
    _mm256_movemask_ps(_mm256_castsi256_ps(gt)) as u16
}

// ============================================================
// f32 operations (L2, L3)
// ============================================================

#[inline(always)]
pub unsafe fn zero_f32() -> __m256 {
    _mm256_setzero_ps()
}

#[inline(always)]
pub unsafe fn splat_f32(a: f32) -> __m256 {
    _mm256_set1_ps(a)
}

/// Element-wise multiply: a * b.
#[inline(always)]
pub unsafe fn mul_f32(a: __m256, b: __m256) -> __m256 {
    _mm256_mul_ps(a, b)
}

/// Fused multiply-add: a * b + c.
#[inline(always)]
pub unsafe fn mul_add_f32(a: __m256, b: __m256, c: __m256) -> __m256 {
    _mm256_fmadd_ps(a, b, c)
}

/// Convert i32 vector to f32 vector.
#[inline(always)]
pub unsafe fn convert_to_f32(a: __m256i) -> __m256 {
    _mm256_cvtepi32_ps(a)
}

/// Clamp f32 vector to [min, max].
#[inline(always)]
pub unsafe fn clamp_f32(x: __m256, min: __m256, max: __m256) -> __m256 {
    _mm256_min_ps(_mm256_max_ps(x, min), max)
}

/// Horizontal sum of two f32 vectors → scalar f32.
///
/// Takes 2 × __m256 (16 f32 total) and returns their sum.
#[inline(always)]
pub unsafe fn horizontal_sum(x: [__m256; 2]) -> f32 {
    let vec = _mm256_add_ps(x[0], x[1]);
    let hi128 = _mm256_extractf128_ps::<1>(vec);
    let lo128 = _mm256_castps256_ps128(vec);
    let sum128 = _mm_add_ps(lo128, hi128);
    let hi64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, hi64);
    let hi32 = _mm_shuffle_ps::<1>(sum64, sum64);
    let sum32 = _mm_add_ss(sum64, hi32);
    _mm_cvtss_f32(sum32)
}

/// Number of __m256 vectors needed for horizontal_sum input.
pub const HSUM_VECS: usize = 2;
