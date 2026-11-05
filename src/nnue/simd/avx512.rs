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
/// Emulated 512-bit dpbusd: one chain, same as AVX2.
pub const DPBUSD_CHAINS: usize = 1;
/// `mul_add_f32` is one fused instruction (one rounding).
pub const MUL_ADD_FUSED: bool = true;

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

/// The pairwise FT product `(left * right) >> 9` (`FT_SHIFT`), for `left` in
/// `0..=FT_QUANT`: `mulhi(left << 7, right)`, the shift putting the product's bits
/// 9 and up into the high half.
#[inline(always)]
pub unsafe fn ft_product(left: __m512i, right: __m512i) -> __m512i {
    _mm512_mulhi_epi16(_mm512_slli_epi16::<7>(left), right)
}

/// Pack two vectors of i16 → one vector of u8 (unsigned saturation).
#[inline(always)]
pub unsafe fn packus(a: __m512i, b: __m512i) -> __m512i {
    _mm512_packus_epi16(a, b)
}

/// Fix lane interleaving after packus (AVX-512 crosses 128-bit lanes).
#[inline(always)]
pub unsafe fn permute(a: __m512i) -> __m512i {
    _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7), a)
}

/// Two registers of i16 from `2 * I16_LANES` bytes of each of two i8 rows,
/// summed lane by lane: see the AVX2 form, which this widens to four 128-bit
/// lanes. The interleaving is again the one `packus` undoes.
#[inline(always)]
pub unsafe fn widen_i8_pair(a: *const i8, b: *const i8) -> [__m512i; 2] {
    let ones = _mm512_set1_epi8(1);
    let wa = _mm512_loadu_si512(a as *const __m512i);
    let wb = _mm512_loadu_si512(b as *const __m512i);
    [
        _mm512_maddubs_epi16(ones, _mm512_unpacklo_epi8(wa, wb)),
        _mm512_maddubs_epi16(ones, _mm512_unpackhi_epi8(wa, wb)),
    ]
}

/// One i8 row alone, in the order of [`widen_i8_pair`].
#[inline(always)]
pub unsafe fn widen_i8(a: *const i8) -> [__m512i; 2] {
    let ones = _mm512_set1_epi8(1);
    let zero = _mm512_setzero_si512();
    let w = _mm512_loadu_si512(a as *const __m512i);
    [
        _mm512_maddubs_epi16(ones, _mm512_unpacklo_epi8(w, zero)),
        _mm512_maddubs_epi16(ones, _mm512_unpackhi_epi8(w, zero)),
    ]
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
/// Emulates VNNI `dpbusd` with maddubs+madd+add (3 instructions); the
/// `avx512vnni` backend shadows this with the native single instruction.
#[inline(always)]
pub unsafe fn dpbusd(acc: __m512i, u8s: __m512i, i8s: __m512i) -> __m512i {
    let pairwise = _mm512_maddubs_epi16(u8s, i8s);
    let widened = _mm512_madd_epi16(pairwise, _mm512_set1_epi16(1));
    _mm512_add_epi32(acc, widened)
}

/// Double dpbusd: process two pairs in one call.
///
/// Combines pairwise products before widening (saves one madd).
#[inline(always)]
pub unsafe fn double_dpbusd(
    acc: __m512i,
    u8s1: __m512i,
    i8s1: __m512i,
    u8s2: __m512i,
    i8s2: __m512i,
) -> __m512i {
    let pw1 = _mm512_maddubs_epi16(u8s1, i8s1);
    let pw2 = _mm512_maddubs_epi16(u8s2, i8s2);
    let widened = _mm512_madd_epi16(_mm512_add_epi16(pw1, pw2), _mm512_set1_epi16(1));
    _mm512_add_epi32(acc, widened)
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

// ============================================================
// i32 operations (integer dense layers)
// ============================================================

/// Arithmetic right shift of every i32 lane: a floor division by `2^n`.
#[inline(always)]
pub unsafe fn srai_i32(a: __m512i, n: i32) -> __m512i {
    _mm512_sra_epi32(a, _mm_cvtsi32_si128(n))
}

/// Clamp every i32 lane to `[lo, hi]`.
#[inline(always)]
pub unsafe fn clamp_i32(x: __m512i, lo: __m512i, hi: __m512i) -> __m512i {
    _mm512_min_epi32(_mm512_max_epi32(x, lo), hi)
}

/// Low 32 bits of each lane's product.
#[inline(always)]
pub unsafe fn mullo_i32(a: __m512i, b: __m512i) -> __m512i {
    _mm512_mullo_epi32(a, b)
}

/// Four registers of i32 in `0..=255` narrowed to one register of u8, in order:
/// the lanes of `a`, then those of `b`, `c`, `d`. The two packs interleave the
/// 128-bit lanes of the four inputs; one dword permute puts them back.
#[inline(always)]
pub unsafe fn narrow_u8(a: __m512i, b: __m512i, c: __m512i, d: __m512i) -> __m512i {
    let packed = _mm512_packus_epi16(_mm512_packs_epi32(a, b), _mm512_packs_epi32(c, d));
    // Dword `4l + k` holds the lanes `4l..4l+4` of input `k`: logical dword `4k + l`.
    let order = _mm512_setr_epi32(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    _mm512_permutexvar_epi32(order, packed)
}
