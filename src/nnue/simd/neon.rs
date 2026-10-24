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
/// Emulated widening multiply: one chain. Dot-prod overrides this to 3.
pub const DPBUSD_CHAINS: usize = 1;

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
/// NEON has no direct `mulhi_i16`. Widening multiply on each half, shift right
/// by 16, then narrow and recombine. `vmull_high_s16` reads the top half
/// without a separate extract.
#[inline(always)]
pub unsafe fn mul_high_i16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    let lo = vmull_s16(vget_low_s16(a), vget_low_s16(b));
    let hi = vmull_high_s16(a, b);
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

/// Dot product of four-byte groups, accumulated into i32.
///
/// Emulates x86 VNNI `dpbusd`. The feature-transformer pairwise product is in
/// `0..=127` (`FT_QUANT=255`, `FT_SHIFT=9`), so a signed×signed 4-byte dot
/// product is bit-identical to mixed-sign u8×i8. Widening `s8` multiply plus
/// pairwise add; the Dot Product backend replaces this with a single `sdot`.
#[inline(always)]
pub unsafe fn dpbusd(acc: int32x4_t, u8s: int32x4_t, i8s: int8x16_t) -> int32x4_t {
    let a = vreinterpretq_s8_s32(u8s);
    let product0 = vmull_s8(vget_low_s8(a), vget_low_s8(i8s));
    let product1 = vmull_high_s8(a, i8s);
    let sum = vpaddq_s16(product0, product1);
    vpadalq_s16(acc, sum)
}

/// Double dpbusd: process two (u8, i8) pairs in one call.
///
/// The two widening products are added in i16 before the long accumulate, so
/// the intermediate 127×127×2 still fits in i16.
#[inline(always)]
pub unsafe fn double_dpbusd(
    acc: int32x4_t,
    u8s1: int32x4_t,
    i8s1: int8x16_t,
    u8s2: int32x4_t,
    i8s2: int8x16_t,
) -> int32x4_t {
    let a1 = vreinterpretq_s8_s32(u8s1);
    let a2 = vreinterpretq_s8_s32(u8s2);
    let p1_lo = vmull_s8(vget_low_s8(a1), vget_low_s8(i8s1));
    let p1_hi = vmull_high_s8(a1, i8s1);
    let p2_lo = vmull_s8(vget_low_s8(a2), vget_low_s8(i8s2));
    let p2_hi = vmull_high_s8(a2, i8s2);
    let sum = vpaddq_s16(vaddq_s16(p1_lo, p2_lo), vaddq_s16(p1_hi, p2_hi));
    vpadalq_s16(acc, sum)
}

/// Extract bitmask of non-zero i32 lanes (4 bits, one per lane).
#[inline(always)]
pub unsafe fn nnz_bitmask(x: int32x4_t) -> u16 {
    let cmp = vcgtq_s32(x, vdupq_n_s32(0));
    let bits = vandq_u32(cmp, vld1q_u32([1u32, 2, 4, 8].as_ptr()));
    vaddvq_u32(bits) as u16
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
/// Reduces 4 × float32x4_t (16 f32 total, e0..e15 in lane order) to a single scalar,
/// in the order of the x86 paths: t_i = (e_i + e_i+8) + (e_i+4 + e_i+12), then
/// (t0 + t2) + (t1 + t3). Float addition does not associate, and any other order
/// moves the output of L3 by an ulp on rare positions — enough to change the search
/// (#40: 9 705 037 bench nodes on ARM64 against 9 421 704 on x86).
#[inline(always)]
pub unsafe fn horizontal_sum(x: [float32x4_t; 4]) -> f32 {
    let sum = vaddq_f32(vaddq_f32(x[0], x[2]), vaddq_f32(x[1], x[3]));
    let half = vadd_f32(vget_low_f32(sum), vget_high_f32(sum));
    vget_lane_f32::<0>(vpadd_f32(half, half))
}

/// Number of float32x4_t vectors needed for horizontal_sum input.
pub const HSUM_VECS: usize = 4;

#[cfg(test)]
mod tests {
    use super::*;

    /// Scalar u8×i8 dpbusd over four 4-byte groups. Matches the production
    /// range: activations in 0..=127.
    fn dpbusd_ref(acc: [i32; 4], u8s: [u8; 16], i8s: [i8; 16]) -> [i32; 4] {
        let mut out = acc;
        for lane in 0..4 {
            let mut s = out[lane];
            for k in 0..4 {
                let i = lane * 4 + k;
                s += u8s[i] as i32 * i8s[i] as i32;
            }
            out[lane] = s;
        }
        out
    }

    fn to_i32x4(x: [i32; 4]) -> int32x4_t {
        unsafe { vld1q_s32(x.as_ptr()) }
    }

    fn from_i32x4(x: int32x4_t) -> [i32; 4] {
        let mut out = [0i32; 4];
        unsafe { vst1q_s32(out.as_mut_ptr(), x) };
        out
    }

    #[test]
    fn dpbusd_matches_scalar_on_the_ft_output_range() {
        let acc = [1i32, -2, 3, 0];
        let mut u8s = [0u8; 16];
        let mut i8s = [0i8; 16];
        for i in 0..16 {
            u8s[i] = ((i * 13) % 128) as u8;
            i8s[i] = (i as i8).wrapping_mul(17).wrapping_sub(64);
        }
        // Include the pairwise product ceiling.
        u8s[0] = 127;
        u8s[15] = 0;
        i8s[0] = 127;
        i8s[4] = -128;
        let want = dpbusd_ref(acc, u8s, i8s);
        let got = unsafe {
            from_i32x4(dpbusd(
                to_i32x4(acc),
                vreinterpretq_s32_u8(vld1q_u8(u8s.as_ptr())),
                vld1q_s8(i8s.as_ptr()),
            ))
        };
        assert_eq!(got, want);
    }

    /// The reduction tree of the x86 paths (AVX2 and AVX-512), in scalar.
    fn x86_order(e: [f32; 16]) -> f32 {
        let t = |i: usize| (e[i] + e[i + 8]) + (e[i + 4] + e[i + 12]);
        (t(0) + t(2)) + (t(1) + t(3))
    }

    fn neon_sum(e: [f32; 16]) -> f32 {
        unsafe {
            let v = |i: usize| vld1q_f32(e[i..].as_ptr());
            horizontal_sum([v(0), v(4), v(8), v(12)])
        }
    }

    #[test]
    fn horizontal_sum_adds_in_the_x86_order() {
        // An order where it matters: e0 + e4 first rounds the 1 away, e0 + e8 first
        // keeps both ones.
        let mut e = [0.0f32; 16];
        (e[0], e[4], e[8], e[12]) = (1e8, 1.0, -1e8, 1.0);
        assert_eq!(x86_order(e), 2.0);
        assert_eq!(neon_sum(e), 2.0);

        // And bit for bit on values of every magnitude.
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..10_000 {
            let mut e = [0.0f32; 16];
            for x in &mut e {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let mantissa = (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                *x = mantissa * 10f32.powi((state % 13) as i32 - 4);
            }
            assert_eq!(neon_sum(e).to_bits(), x86_order(e).to_bits(), "{e:?}");
        }
    }

    #[test]
    fn nnz_bitmask_reports_strictly_positive_lanes() {
        unsafe {
            assert_eq!(nnz_bitmask(to_i32x4([0, 0, 0, 0])), 0);
            assert_eq!(nnz_bitmask(to_i32x4([1, 0, 0, 0])), 0b0001);
            assert_eq!(nnz_bitmask(to_i32x4([0, 1, 0, 1])), 0b1010);
            assert_eq!(nnz_bitmask(to_i32x4([-1, 0, 2, -3])), 0b0100);
            assert_eq!(nnz_bitmask(to_i32x4([1, 1, 1, 1])), 0b1111);
        }
    }
}
