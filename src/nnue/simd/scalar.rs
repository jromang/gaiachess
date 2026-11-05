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
pub const DPBUSD_CHAINS: usize = 1;
/// `mul_add_f32` multiplies, then adds (see there why).
pub const MUL_ADD_FUSED: bool = false;

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

/// The pairwise FT product `(left * right) >> 9` (`FT_SHIFT`).
#[inline(always)]
pub unsafe fn ft_product(left: i16, right: i16) -> i16 {
    // The kernels clamp left to [0, 255] and right to at most 255: the shifted product
    // lies in [-16320, 127] and the cast keeps it whole.
    debug_assert!((0..=255).contains(&left) && right <= 255, "ft_product({left}, {right})");
    ((left as i32 * right as i32) >> 9) as i16
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

/// Two lanes of each of two i8 rows, widened and summed (cannot overflow i16).
/// Scalar equivalent of the SIMD `widen_i8_pair`; the order is linear.
#[inline(always)]
pub unsafe fn widen_i8_pair(a: *const i8, b: *const i8) -> [i16; 2] {
    [*a as i16 + *b as i16, *a.add(1) as i16 + *b.add(1) as i16]
}

/// Two lanes of one i8 row, widened.
#[inline(always)]
pub unsafe fn widen_i8(a: *const i8) -> [i16; 2] {
    [*a as i16, *a.add(1) as i16]
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

/// Multiply, then add: two roundings, where the vector tiers fuse them into one.
///
/// No layer of the network calls the f32 operations any more: it is integer from end to
/// end, so this tier searches the same tree as every vector tier.
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

/// Horizontal sum of sixteen partial sums → scalar f32.
///
/// One value per "register", so the dot product of L3 keeps sixteen partial sums
/// like every vector path, and reduces them in the same tree as the others:
/// t_i = (e_i + e_i+8) + (e_i+4 + e_i+12), then (t0 + t2) + (t1 + t3). With mul + add
/// unfused on both sides, this is what makes the scalar wasm module evaluate exactly
/// like the simd128 one. A single 64-term chain would round differently.
#[inline(always)]
pub unsafe fn horizontal_sum(e: [f32; 16]) -> f32 {
    let t = |i: usize| (e[i] + e[i + 8]) + (e[i + 4] + e[i + 12]);
    (t(0) + t(2)) + (t(1) + t(3))
}

/// Number of f32 values needed for horizontal_sum input.
pub const HSUM_VECS: usize = 16;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn horizontal_sum_adds_in_the_vector_order() {
        // e0 + e4 first would round the 1 away; e0 + e8 first keeps both ones.
        let mut e = [0.0f32; 16];
        (e[0], e[4], e[8], e[12]) = (1e8, 1.0, -1e8, 1.0);
        assert_eq!(unsafe { horizontal_sum(e) }, 2.0);
    }
}

// ============================================================
// i32 operations (integer dense layers)
// ============================================================

/// Arithmetic right shift: a floor division by `2^n`.
#[inline(always)]
pub unsafe fn srai_i32(a: i32, n: i32) -> i32 {
    a >> n
}

#[inline(always)]
pub unsafe fn clamp_i32(x: i32, lo: i32, hi: i32) -> i32 {
    x.max(lo).min(hi)
}

/// Low 32 bits of the product.
#[inline(always)]
pub unsafe fn mullo_i32(a: i32, b: i32) -> i32 {
    a.wrapping_mul(b)
}

/// Four values in `0..=255` as the four bytes of one "register", first in the low
/// byte: the order the scalar `dpbusd` reads them in.
#[inline(always)]
pub unsafe fn narrow_u8(a: i32, b: i32, c: i32, d: i32) -> i32 {
    let byte = |x: i32| x.clamp(0, 255) as u32;
    (byte(a) | byte(b) << 8 | byte(c) << 16 | byte(d) << 24) as i32
}
