//! ARM NEON + Dot Product (`sdot`) primitives for NNUE inference.
//!
//! Same 128-bit register width as [`super::neon`]. The only difference is
//! `dpbusd` / `double_dpbusd`: one `sdot` per four-byte group instead of a
//! widening multiply. Pairwise FT output is in `0..=127`, so signed×signed
//! matches mixed-sign u8×i8.

#![allow(unsafe_op_in_unsafe_fn)]
#![allow(dead_code)]

pub use super::neon::{
    F32_LANES, HSUM_VECS, I16_LANES, I32_LANES, add_i16, add_i32, clamp_f32, clamp_i16,
    convert_to_f32, horizontal_sum, horizontal_sum_i32, load_i8_as_i16, madd_i16, min_i16,
    mul_add_f32, mul_f32, mul_high_i16, nnz_bitmask, packus, permute, shift_left_i16, splat_f32,
    splat_i16, splat_i32, sub_i16, zero_f32, zeroed_i16, zeroed_i32,
};

use std::arch::aarch64::*;

/// Native `sdot`: four s8×s8 products per i32 lane, accumulated.
///
/// `#[inline]` rather than `always`: `#[target_feature]` cannot be combined
/// with `#[inline(always)]` on stable.
#[inline]
#[cfg_attr(not(target_feature = "dotprod"), target_feature(enable = "dotprod"))]
pub unsafe fn dpbusd(acc: int32x4_t, u8s: int32x4_t, i8s: int8x16_t) -> int32x4_t {
    vdotq_s32(acc, vreinterpretq_s8_s32(u8s), i8s)
}

/// Two `sdot`s chained.
#[inline]
#[cfg_attr(not(target_feature = "dotprod"), target_feature(enable = "dotprod"))]
pub unsafe fn double_dpbusd(
    acc: int32x4_t,
    u8s1: int32x4_t,
    i8s1: int8x16_t,
    u8s2: int32x4_t,
    i8s2: int8x16_t,
) -> int32x4_t {
    vdotq_s32(
        vdotq_s32(acc, vreinterpretq_s8_s32(u8s1), i8s1),
        vreinterpretq_s8_s32(u8s2),
        i8s2,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn sdot_matches_scalar_on_the_ft_output_range() {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }
        let acc = [1i32, -2, 3, 0];
        let mut u8s = [0u8; 16];
        let mut i8s = [0i8; 16];
        for i in 0..16 {
            u8s[i] = ((i * 13) % 128) as u8;
            i8s[i] = (i as i8).wrapping_mul(17).wrapping_sub(64);
        }
        u8s[0] = 127;
        u8s[15] = 0;
        i8s[0] = 127;
        i8s[4] = -128;
        let want = dpbusd_ref(acc, u8s, i8s);
        let got = unsafe {
            // Isolated +dotprod shim so this test compiles in a generic binary.
            #[target_feature(enable = "dotprod")]
            fn run(acc: int32x4_t, u8s: int32x4_t, i8s: int8x16_t) -> int32x4_t {
                unsafe { dpbusd(acc, u8s, i8s) }
            }
            let acc_v = vld1q_s32(acc.as_ptr());
            let u_v = vreinterpretq_s32_u8(vld1q_u8(u8s.as_ptr()));
            let i_v = vld1q_s8(i8s.as_ptr());
            let out = run(acc_v, u_v, i_v);
            let mut lanes = [0i32; 4];
            vst1q_s32(lanes.as_mut_ptr(), out);
            lanes
        };
        assert_eq!(got, want);
    }
}
