//! ARM NEON SIMD primitives for NNUE inference.
//!
//! 128-bit registers: 8 × i16, 4 × i32, 4 × f32.

// Every function here is `unsafe fn` wrapping a single intrinsic call.
// The safety contract is lifted to the caller; no additional invariants inside.
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(dead_code)]

use std::arch::aarch64::*;
use std::mem::size_of;

/// Number of i16 elements per SIMD register.
pub const I16_LANES: usize = size_of::<int16x8_t>() / size_of::<i16>();
/// Number of i32 elements per SIMD register.
pub const I32_LANES: usize = size_of::<int32x4_t>() / size_of::<i32>();
/// Number of f32 elements per SIMD register.
pub const F32_LANES: usize = size_of::<float32x4_t>() / size_of::<f32>();

// ============================================================
// i16 operations (accumulator, FT activation)
// ============================================================

#[inline(always)]
pub unsafe fn add_i16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    vaddq_s16(a, b)
}

#[inline(always)]
pub unsafe fn sub_i16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    vsubq_s16(a, b)
}

#[inline(always)]
pub unsafe fn zeroed_i16() -> int16x8_t {
    vdupq_n_s16(0)
}

#[inline(always)]
pub unsafe fn splat_i16(a: i16) -> int16x8_t {
    vdupq_n_s16(a)
}

#[inline(always)]
pub unsafe fn clamp_i16(x: int16x8_t, min: int16x8_t, max: int16x8_t) -> int16x8_t {
    vmaxq_s16(vminq_s16(x, max), min)
}

#[inline(always)]
pub unsafe fn min_i16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    vminq_s16(a, b)
}

#[inline(always)]
pub unsafe fn shift_left_i16<const SHIFT: i32>(a: int16x8_t) -> int16x8_t {
    vshlq_n_s16::<SHIFT>(a)
}

/// Multiply high: (a * b) >> 16 for each i16 lane.
///
/// NEON has no direct `mulhi_i16`. Emulate via widening multiply on each half,
/// shift right by 16, then narrow and recombine.
#[inline(always)]
pub unsafe fn mul_high_i16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    let lo = vmull_s16(vget_low_s16(a), vget_low_s16(b));
    let hi = vmull_s16(vget_high_s16(a), vget_high_s16(b));
    let lo_hi = vshrn_n_s32::<16>(lo);
    let hi_hi = vshrn_n_s32::<16>(hi);
    vcombine_s16(lo_hi, hi_hi)
}

/// Pack two i16 vectors → one vector of u8 (unsigned saturation).
///
/// NEON `vqmovun_s16` saturates i16 → u8 per half, then `vcombine_u8` joins them.
/// Unlike AVX2/512 `packus`, NEON does NOT interleave 128-bit lanes, so no
/// subsequent `permute` fixup is needed.
#[inline(always)]
pub unsafe fn packus(a: int16x8_t, b: int16x8_t) -> int8x16_t {
    let a_u8 = vqmovun_s16(a);
    let b_u8 = vqmovun_s16(b);
    vreinterpretq_s8_u8(vcombine_u8(a_u8, b_u8))
}

/// No-op: NEON packus output is already in linear order (no lane crossing).
#[inline(always)]
pub unsafe fn permute(a: int8x16_t) -> int8x16_t {
    a
}

/// Load I16_LANES (8) i8 values from `ptr` and sign-extend to i16.
///
/// Used for threat accumulator: i8 weights → i16 additions.
#[inline(always)]
pub unsafe fn load_i8_as_i16(ptr: *const i8) -> int16x8_t {
    vmovl_s8(vld1_s8(ptr))
}

/// Multiply pairs of adjacent i16 and accumulate into i32.
/// `result[i] = a[2i] * b[2i] + a[2i+1] * b[2i+1]`
///
/// NEON: widen multiply both halves, then pairwise add.
#[inline(always)]
pub unsafe fn madd_i16(a: int16x8_t, b: int16x8_t) -> int32x4_t {
    let lo = vmull_s16(vget_low_s16(a), vget_low_s16(b));
    let hi = vmull_s16(vget_high_s16(a), vget_high_s16(b));
    vpaddq_s32(lo, hi)
}

/// Horizontal sum of an i32 vector → scalar i32.
#[inline(always)]
pub unsafe fn horizontal_sum_i32(x: int32x4_t) -> i32 {
    vaddvq_s32(x)
}

// ============================================================
// i32 operations (sparse L1, NNZ)
// ============================================================

#[inline(always)]
pub unsafe fn zeroed_i32() -> int32x4_t {
    vdupq_n_s32(0)
}

#[inline(always)]
pub unsafe fn splat_i32(a: i32) -> int32x4_t {
    vdupq_n_s32(a)
}

#[inline(always)]
pub unsafe fn add_i32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    vaddq_s32(a, b)
}

/// Dot product of unsigned bytes × signed bytes, accumulated into i32.
///
/// Emulates x86 VNNI `dpbusd`: for each group of 4 bytes in u8s × i8s,
/// computes u8[0]*i8[0] + u8[1]*i8[1] + u8[2]*i8[2] + u8[3]*i8[3] and
/// accumulates into the corresponding i32 lane.
///
/// NEON implementation: widen to i16, pairwise multiply, pairwise add to i32.
#[inline(always)]
pub unsafe fn dpbusd(acc: int32x4_t, u8s: int32x4_t, i8s: int8x16_t) -> int32x4_t {
    let u8s = vreinterpretq_u8_s32(u8s);

    let products_lo = vmulq_s16(
        vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(u8s))),
        vmovl_s8(vget_low_s8(i8s)),
    );
    let products_hi = vmulq_s16(
        vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(u8s))),
        vmovl_s8(vget_high_s8(i8s)),
    );

    let sums_lo = vpaddlq_s16(products_lo);
    let sums_hi = vpaddlq_s16(products_hi);

    vaddq_s32(acc, vpaddq_s32(sums_lo, sums_hi))
}

/// Double dpbusd: process two (u8, i8) pairs in one call.
#[inline(always)]
pub unsafe fn double_dpbusd(
    acc: int32x4_t,
    u8s1: int32x4_t,
    i8s1: int8x16_t,
    u8s2: int32x4_t,
    i8s2: int8x16_t,
) -> int32x4_t {
    let u8s1 = vreinterpretq_u8_s32(u8s1);
    let u8s2 = vreinterpretq_u8_s32(u8s2);

    let p1_lo = vmulq_s16(
        vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(u8s1))),
        vmovl_s8(vget_low_s8(i8s1)),
    );
    let p1_hi = vmulq_s16(
        vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(u8s1))),
        vmovl_s8(vget_high_s8(i8s1)),
    );
    let p2_lo = vmulq_s16(
        vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(u8s2))),
        vmovl_s8(vget_low_s8(i8s2)),
    );
    let p2_hi = vmulq_s16(
        vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(u8s2))),
        vmovl_s8(vget_high_s8(i8s2)),
    );

    let sums_lo = vpaddlq_s16(vaddq_s16(p1_lo, p2_lo));
    let sums_hi = vpaddlq_s16(vaddq_s16(p1_hi, p2_hi));

    vaddq_s32(acc, vpaddq_s32(sums_lo, sums_hi))
}

/// Extract bitmask of non-zero i32 lanes (4 bits, one per lane).
#[inline(always)]
pub unsafe fn nnz_bitmask(x: int32x4_t) -> u16 {
    let cmp = vcgtq_s32(x, vdupq_n_s32(0));
    let mask0 = (vgetq_lane_u32::<0>(cmp) >> 31) & 1;
    let mask1 = ((vgetq_lane_u32::<1>(cmp) >> 31) & 1) << 1;
    let mask2 = ((vgetq_lane_u32::<2>(cmp) >> 31) & 1) << 2;
    let mask3 = ((vgetq_lane_u32::<3>(cmp) >> 31) & 1) << 3;
    (mask0 | mask1 | mask2 | mask3) as u16
}

// ============================================================
// f32 operations (L2, L3)
// ============================================================

#[inline(always)]
pub unsafe fn zero_f32() -> float32x4_t {
    vdupq_n_f32(0.0)
}

#[inline(always)]
pub unsafe fn splat_f32(a: f32) -> float32x4_t {
    vdupq_n_f32(a)
}

/// Element-wise multiply: a * b.
#[inline(always)]
pub unsafe fn mul_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    vmulq_f32(a, b)
}

/// Fused multiply-add: a * b + c.
#[inline(always)]
pub unsafe fn mul_add_f32(a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
    vfmaq_f32(c, a, b)
}

/// Convert i32 vector to f32 vector.
#[inline(always)]
pub unsafe fn convert_to_f32(a: int32x4_t) -> float32x4_t {
    vcvtq_f32_s32(a)
}

/// Clamp f32 vector to [min, max].
#[inline(always)]
pub unsafe fn clamp_f32(x: float32x4_t, min: float32x4_t, max: float32x4_t) -> float32x4_t {
    vmaxq_f32(vminq_f32(x, max), min)
}

/// Horizontal sum of four f32 vectors → scalar f32.
///
/// Reduces 4 × float32x4_t (16 f32 total) to a single scalar.
#[inline(always)]
pub unsafe fn horizontal_sum(x: [float32x4_t; 4]) -> f32 {
    let sum01 = vaddq_f32(x[0], x[1]);
    let sum23 = vaddq_f32(x[2], x[3]);
    let sum = vaddq_f32(sum01, sum23);
    let pair = vpadd_f32(vget_low_f32(sum), vget_high_f32(sum));
    let final_sum = vpadd_f32(pair, pair);
    vget_lane_f32::<0>(final_sum)
}

/// Number of float32x4_t vectors needed for horizontal_sum input.
pub const HSUM_VECS: usize = 4;
