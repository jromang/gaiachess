//! Scalar fallback SIMD primitives (no hardware SIMD).
//!
//! Single-element "lanes" — same API as AVX2/AVX-512, but one value at a time.
//! Used when compiling without `-C target-cpu=native` or similar.

#![allow(unsafe_op_in_unsafe_fn)]
#![allow(dead_code)]

/// Number of i16 elements per "register" (scalar = 1).
pub const I16_LANES: usize = 1;
/// Number of i32 elements per "register".
pub const I32_LANES: usize = 1;
/// Number of f32 elements per "register".
pub const F32_LANES: usize = 1;

// ============================================================
// i16 operations
// ============================================================

#[inline(always)]
pub unsafe fn add_i16(a: i16, b: i16) -> i16 {
    a.wrapping_add(b)
}

#[inline(always)]
pub unsafe fn sub_i16(a: i16, b: i16) -> i16 {
    a.wrapping_sub(b)
}

#[inline(always)]
pub unsafe fn zeroed_i16() -> i16 {
    0
}

#[inline(always)]
pub unsafe fn splat_i16(a: i16) -> i16 {
    a
}

#[inline(always)]
pub unsafe fn clamp_i16(x: i16, min: i16, max: i16) -> i16 {
    x.max(min).min(max)
}

#[inline(always)]
pub unsafe fn min_i16(a: i16, b: i16) -> i16 {
    a.min(b)
}

#[inline(always)]
pub unsafe fn shift_left_i16<const SHIFT: i32>(a: i16) -> i16 {
    (a as u16).wrapping_shl(SHIFT as u32) as i16
}

#[inline(always)]
pub unsafe fn mul_high_i16(a: i16, b: i16) -> i16 {
    ((a as i32 * b as i32) >> 16) as i16
}

#[inline(always)]
pub unsafe fn packus(_a: i16, _b: i16) -> u8 {
    // In scalar mode, packus is not used (forward pass uses scalar loops).
    0
}

#[inline(always)]
pub unsafe fn permute(a: u8) -> u8 {
    a
}

/// Load 1 i8 value from `ptr` and sign-extend to i16.
///
/// Scalar equivalent of SIMD load_i8_as_i16.
#[inline(always)]
pub unsafe fn load_i8_as_i16(ptr: *const i8) -> i16 {
    *ptr as i16
}

/// Multiply two i16 values and return their product as i32.
/// Scalar equivalent of SIMD madd_i16 (1 lane = 1 pair).
#[inline(always)]
pub unsafe fn madd_i16(a: i16, b: i16) -> i32 {
    a as i32 * b as i32
}

/// Horizontal sum of an i32 "vector" (scalar: identity).
#[inline(always)]
pub unsafe fn horizontal_sum_i32(x: i32) -> i32 {
    x
}

// ============================================================
// i32 operations
// ============================================================

#[inline(always)]
pub unsafe fn zeroed_i32() -> i32 {
    0
}

#[inline(always)]
pub unsafe fn splat_i32(a: i32) -> i32 {
    a
}

#[inline(always)]
pub unsafe fn add_i32(a: i32, b: i32) -> i32 {
    a + b
}

#[inline(always)]
pub unsafe fn dpbusd(acc: i32, u8s: i32, i8s: i32) -> i32 {
    // Scalar: multiply each byte pair and sum
    let mut sum = acc;
    for k in 0..4 {
        let u = ((u8s >> (k * 8)) & 0xFF) as u8;
        let s = ((i8s >> (k * 8)) & 0xFF) as i8;
        sum += u as i32 * s as i32;
    }
    sum
}

#[inline(always)]
pub unsafe fn double_dpbusd(acc: i32, u8s1: i32, i8s1: i32, u8s2: i32, i8s2: i32) -> i32 {
    let a = dpbusd(acc, u8s1, i8s1);
    dpbusd(a, u8s2, i8s2)
}

#[inline(always)]
pub unsafe fn nnz_bitmask(x: i32) -> u16 {
    (x > 0) as u16
}

// ============================================================
// f32 operations
// ============================================================

#[inline(always)]
pub unsafe fn zero_f32() -> f32 {
    0.0
}

#[inline(always)]
pub unsafe fn splat_f32(a: f32) -> f32 {
    a
}

#[inline(always)]
pub unsafe fn mul_f32(a: f32, b: f32) -> f32 {
    a * b
}

#[inline(always)]
pub unsafe fn mul_add_f32(a: f32, b: f32, c: f32) -> f32 {
    a * b + c
}

#[inline(always)]
pub unsafe fn convert_to_f32(a: i32) -> f32 {
    a as f32
}

#[inline(always)]
pub unsafe fn clamp_f32(x: f32, min: f32, max: f32) -> f32 {
    x.max(min).min(max)
}

#[inline(always)]
pub unsafe fn horizontal_sum(x: [f32; 1]) -> f32 {
    x[0]
}

/// Number of f32 values needed for horizontal_sum input.
pub const HSUM_VECS: usize = 1;
