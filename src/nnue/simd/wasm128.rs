//! WebAssembly SIMD (simd128) primitives for NNUE inference.
//!
//! 128-bit registers: 8 × i16, 4 × i32, 4 × f32 — the same shape as NEON, which is what
//! this is modelled on. Unlike the other backends there is a single vector type, `v128`,
//! carrying whatever the operation says it carries.
//!
//! The relaxed-SIMD proposal offers a single-instruction `dpbusd` and a fused multiply
//! add, both of which would be faster. Neither is used, on purpose: relaxed operations
//! are specified as non-deterministic — a fused multiply-add rounds once where hardware
//! provides one and twice where it does not — so the same module would evaluate
//! differently on two machines. A rung has to be the same opponent everywhere, and an
//! evaluation that shifts with the host would break that in a way no test could
//! reproduce.

// Every function here wraps one or two intrinsics. The safety contract is lifted to the
// caller, matching the other backends; no additional invariants inside.
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(dead_code)]

use std::arch::wasm32::*;
use std::mem::size_of;

/// Number of i16 elements per SIMD register.
pub const I16_LANES: usize = size_of::<v128>() / size_of::<i16>();
/// Number of i32 elements per SIMD register.
pub const I32_LANES: usize = size_of::<v128>() / size_of::<i32>();
/// Number of f32 elements per SIMD register.
pub const F32_LANES: usize = size_of::<v128>() / size_of::<f32>();

// ============================================================
// i16 operations (accumulator, FT activation)
// ============================================================

#[inline(always)]
pub unsafe fn add_i16(a: v128, b: v128) -> v128 {
    i16x8_add(a, b)
}

#[inline(always)]
pub unsafe fn sub_i16(a: v128, b: v128) -> v128 {
    i16x8_sub(a, b)
}

#[inline(always)]
pub unsafe fn zeroed_i16() -> v128 {
    i16x8_splat(0)
}

#[inline(always)]
pub unsafe fn splat_i16(a: i16) -> v128 {
    i16x8_splat(a)
}

#[inline(always)]
pub unsafe fn clamp_i16(x: v128, min: v128, max: v128) -> v128 {
    i16x8_max(i16x8_min(x, max), min)
}

#[inline(always)]
pub unsafe fn min_i16(a: v128, b: v128) -> v128 {
    i16x8_min(a, b)
}

#[inline(always)]
pub unsafe fn shift_left_i16<const SHIFT: i32>(a: v128) -> v128 {
    i16x8_shl(a, SHIFT as u32)
}

/// Multiply high: (a * b) >> 16 for each i16 lane.
///
/// No direct instruction. The widening multiplies give the full 32-bit products in two
/// registers; the shuffle then keeps the upper half of each, which is the same as a
/// shift by 16 followed by a narrow, at one instruction instead of three.
#[inline(always)]
pub unsafe fn mul_high_i16(a: v128, b: v128) -> v128 {
    let lo = i32x4_extmul_low_i16x8(a, b);
    let hi = i32x4_extmul_high_i16x8(a, b);
    i8x16_shuffle::<2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31>(lo, hi)
}

/// Pack two i16 vectors → one vector of u8 (unsigned saturation).
///
/// Like NEON and unlike AVX2/512, this does not interleave 128-bit lanes, so the result
/// is already in linear order and [`permute`] has nothing to undo.
#[inline(always)]
pub unsafe fn packus(a: v128, b: v128) -> v128 {
    u8x16_narrow_i16x8(a, b)
}

/// No-op: [`packus`] output is already in linear order (no lane crossing).
#[inline(always)]
pub unsafe fn permute(a: v128) -> v128 {
    a
}

/// Load I16_LANES (8) i8 values from `ptr` and sign-extend to i16.
///
/// Used for threat accumulator: i8 weights → i16 additions.
#[inline(always)]
pub unsafe fn load_i8_as_i16(ptr: *const i8) -> v128 {
    i16x8_load_extend_i8x8(ptr)
}

/// Multiply pairs of adjacent i16 and accumulate into i32.
/// `result[i] = a[2i] * b[2i] + a[2i+1] * b[2i+1]`
#[inline(always)]
pub unsafe fn madd_i16(a: v128, b: v128) -> v128 {
    i32x4_dot_i16x8(a, b)
}

/// Horizontal sum of an i32 vector → scalar i32.
#[inline(always)]
pub unsafe fn horizontal_sum_i32(x: v128) -> i32 {
    let pairs = i32x4_add(x, i32x4_shuffle::<2, 3, 0, 1>(x, x));
    let total = i32x4_add(pairs, i32x4_shuffle::<1, 0, 3, 2>(pairs, pairs));
    i32x4_extract_lane::<0>(total)
}

// ============================================================
// i32 operations (sparse L1, NNZ)
// ============================================================

#[inline(always)]
pub unsafe fn zeroed_i32() -> v128 {
    i32x4_splat(0)
}

#[inline(always)]
pub unsafe fn splat_i32(a: i32) -> v128 {
    i32x4_splat(a)
}

#[inline(always)]
pub unsafe fn add_i32(a: v128, b: v128) -> v128 {
    i32x4_add(a, b)
}

/// Dot product of unsigned bytes × signed bytes, accumulated into i32.
///
/// Emulates x86 VNNI `dpbusd`: for each group of 4 bytes in u8s × i8s, computes
/// `u8[0]*i8[0] + … + u8[3]*i8[3]` and accumulates into the matching i32 lane.
///
/// `i32x4_dot_i16x8` pairs adjacent lanes, so it gets halfway there: over the low eight
/// bytes it yields the four half-sums of groups 0 and 1, and over the high eight bytes
/// those of groups 2 and 3. The two shuffles gather the even and odd half-sums into
/// group order so that one more add finishes each group. Products are widened to i32 by
/// the dot instruction itself, so the intermediate 255 × 127 cannot overflow.
#[inline(always)]
pub unsafe fn dpbusd(acc: v128, u8s: v128, i8s: v128) -> v128 {
    let dot_lo = i32x4_dot_i16x8(u16x8_extend_low_u8x16(u8s), i16x8_extend_low_i8x16(i8s));
    let dot_hi = i32x4_dot_i16x8(u16x8_extend_high_u8x16(u8s), i16x8_extend_high_i8x16(i8s));
    let even =
        i8x16_shuffle::<0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27>(dot_lo, dot_hi);
    let odd = i8x16_shuffle::<4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31>(
        dot_lo, dot_hi,
    );
    i32x4_add(acc, i32x4_add(even, odd))
}

/// Double dpbusd: process two (u8, i8) pairs in one call.
///
/// The two dot products are summed before the gather, which saves one pair of shuffles
/// over calling [`dpbusd`] twice. Every addition is in i32, so nothing can overflow.
#[inline(always)]
pub unsafe fn double_dpbusd(
    acc: v128,
    u8s1: v128,
    i8s1: v128,
    u8s2: v128,
    i8s2: v128,
) -> v128 {
    let lo = i32x4_add(
        i32x4_dot_i16x8(u16x8_extend_low_u8x16(u8s1), i16x8_extend_low_i8x16(i8s1)),
        i32x4_dot_i16x8(u16x8_extend_low_u8x16(u8s2), i16x8_extend_low_i8x16(i8s2)),
    );
    let hi = i32x4_add(
        i32x4_dot_i16x8(u16x8_extend_high_u8x16(u8s1), i16x8_extend_high_i8x16(i8s1)),
        i32x4_dot_i16x8(u16x8_extend_high_u8x16(u8s2), i16x8_extend_high_i8x16(i8s2)),
    );
    let even = i8x16_shuffle::<0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27>(lo, hi);
    let odd = i8x16_shuffle::<4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31>(lo, hi);
    i32x4_add(acc, i32x4_add(even, odd))
}

/// Extract bitmask of non-zero i32 lanes (4 bits, one per lane).
///
/// The comparison is signed, matching every other backend: the caller expects `x > 0`,
/// and an unsigned compare would call negative accumulators non-zero and hand the sparse
/// layer groups it does not need — correct, but slower, and silently so.
#[inline(always)]
pub unsafe fn nnz_bitmask(x: v128) -> u16 {
    i32x4_bitmask(i32x4_gt(x, i32x4_splat(0))) as u16
}

// ============================================================
// f32 operations (L2, L3)
// ============================================================

#[inline(always)]
pub unsafe fn zero_f32() -> v128 {
    f32x4_splat(0.0)
}

#[inline(always)]
pub unsafe fn splat_f32(a: f32) -> v128 {
    f32x4_splat(a)
}

/// Element-wise multiply: a * b.
#[inline(always)]
pub unsafe fn mul_f32(a: v128, b: v128) -> v128 {
    f32x4_mul(a, b)
}

/// Multiply-add: a * b + c. Two roundings, not one — see the note at the top of the file
/// on why the relaxed fused form is refused.
#[inline(always)]
pub unsafe fn mul_add_f32(a: v128, b: v128, c: v128) -> v128 {
    f32x4_add(f32x4_mul(a, b), c)
}

/// Convert i32 vector to f32 vector.
#[inline(always)]
pub unsafe fn convert_to_f32(a: v128) -> v128 {
    f32x4_convert_i32x4(a)
}

/// Clamp f32 vector to [min, max].
#[inline(always)]
pub unsafe fn clamp_f32(x: v128, min: v128, max: v128) -> v128 {
    f32x4_max(f32x4_min(x, max), min)
}

/// Horizontal sum of four f32 vectors → scalar f32.
///
/// Reduces 4 × v128 (16 f32 total) to a single scalar.
#[inline(always)]
pub unsafe fn horizontal_sum(x: [v128; 4]) -> f32 {
    let sum01 = f32x4_add(x[0], x[1]);
    let sum23 = f32x4_add(x[2], x[3]);
    let sum = f32x4_add(sum01, sum23);
    // Shuffles move bits, not numbers, and there is no `f32x4_shuffle`: the 32-bit lane
    // shuffle is the same instruction whatever the lanes are taken to mean.
    let pairs = f32x4_add(sum, i32x4_shuffle::<2, 3, 0, 1>(sum, sum));
    let total = f32x4_add(pairs, i32x4_shuffle::<1, 0, 3, 2>(pairs, pairs));
    f32x4_extract_lane::<0>(total)
}

/// Number of v128 vectors needed for horizontal_sum input.
pub const HSUM_VECS: usize = 4;
