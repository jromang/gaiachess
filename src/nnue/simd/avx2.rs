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
/// Independent sparse-L1 dpbusd chains. Emulated maddubs is short enough
/// that one chain wins; VNNI's 5-cycle `vpdpbusd` wants three.
pub const DPBUSD_CHAINS: usize = 1;
/// `mul_add_f32` is one fused instruction (one rounding).
pub const MUL_ADD_FUSED: bool = true;

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

/// The pairwise FT product `(left * right) >> 9` (`FT_SHIFT`), for `left` in
/// `0..=FT_QUANT`: `mulhi(left << 7, right)`, the shift putting the product's bits
/// 9 and up into the high half.
#[inline(always)]
pub unsafe fn ft_product(left: __m256i, right: __m256i) -> __m256i {
    _mm256_mulhi_epi16(_mm256_slli_epi16::<7>(left), right)
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

/// Two registers of i16 from `2 * I16_LANES` bytes of each of two i8 rows,
/// summed lane by lane.
///
/// `unpack{lo,hi}_epi8` interleave the rows byte by byte inside each 128-bit
/// lane, and `maddubs` against ones adds each pair into one i16 — exactly, the
/// sum lies in [-256, 254]. One shuffle per register for two rows, where a
/// `vpmovsxbw` costs one per register and per row.
///
/// The first register gets the low 8 bytes of each 128-bit lane, the second the
/// high 8: the interleaving `packus` undoes, so rows held in logical order land
/// on the accumulator's packus-ordered lanes.
#[inline(always)]
pub unsafe fn widen_i8_pair(a: *const i8, b: *const i8) -> [__m256i; 2] {
    let ones = _mm256_set1_epi8(1);
    let wa = _mm256_loadu_si256(a as *const __m256i);
    let wb = _mm256_loadu_si256(b as *const __m256i);
    [
        _mm256_maddubs_epi16(ones, _mm256_unpacklo_epi8(wa, wb)),
        _mm256_maddubs_epi16(ones, _mm256_unpackhi_epi8(wa, wb)),
    ]
}

/// One i8 row alone, in the order of [`widen_i8_pair`]: paired with zero, through
/// the same `maddubs`. A sign-extending shift of the unpacked bytes is exact too,
/// but measured about 0.5 % slower on Zen 5.
#[inline(always)]
pub unsafe fn widen_i8(a: *const i8) -> [__m256i; 2] {
    let ones = _mm256_set1_epi8(1);
    let zero = _mm256_setzero_si256();
    let w = _mm256_loadu_si256(a as *const __m256i);
    [
        _mm256_maddubs_epi16(ones, _mm256_unpacklo_epi8(w, zero)),
        _mm256_maddubs_epi16(ones, _mm256_unpackhi_epi8(w, zero)),
    ]
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

// ============================================================
// i32 operations (integer dense layers)
// ============================================================

/// Arithmetic right shift of every i32 lane: a floor division by `2^n`.
#[inline(always)]
pub unsafe fn srai_i32(a: __m256i, n: i32) -> __m256i {
    _mm256_sra_epi32(a, _mm_cvtsi32_si128(n))
}

/// Clamp every i32 lane to `[lo, hi]`.
#[inline(always)]
pub unsafe fn clamp_i32(x: __m256i, lo: __m256i, hi: __m256i) -> __m256i {
    _mm256_min_epi32(_mm256_max_epi32(x, lo), hi)
}

/// Low 32 bits of each lane's product.
#[inline(always)]
pub unsafe fn mullo_i32(a: __m256i, b: __m256i) -> __m256i {
    _mm256_mullo_epi32(a, b)
}

/// Four registers of i32 in `0..=255` narrowed to one register of u8, in order:
/// the lanes of `a`, then those of `b`, `c`, `d`. The two packs interleave the
/// 128-bit lanes of the four inputs; one dword permute puts them back.
#[inline(always)]
pub unsafe fn narrow_u8(a: __m256i, b: __m256i, c: __m256i, d: __m256i) -> __m256i {
    let packed = _mm256_packus_epi16(_mm256_packs_epi32(a, b), _mm256_packs_epi32(c, d));
    // Dword `4l + k` holds the lanes `4l..4l+4` of input `k`: logical dword `2k + l`.
    _mm256_permutevar8x32_epi32(packed, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7))
}
