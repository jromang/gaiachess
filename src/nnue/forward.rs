//! SIMD-accelerated NNUE forward pass.
//!
//! Pipeline per evaluation:
//!   1. `threats::compute_full_threats(pos)` — filtered threat features (41,272),
//!      loaded as i8 weight rows → i16 buffer[640][2 perspectives]
//!   2. `activate_ft(pst_acc, threats)` — element-wise PST + threats,
//!      within-side pairwise product (i16 → u8[640])
//!   3. `find_nnz` — extract non-zero 4-byte group indices for sparse L1
//!   4. `propagate_l1` — sparse u8 × i8 → i32, dequant, CReLU+squared → f32[32]
//!   5. `propagate_l2` — dense f32 FMA + squared → f32[32]
//!   6. `propagate_l3` — concat(l2[32], l1[32]) × l3_weights → scalar (with L1 skip)

#![allow(unsafe_op_in_unsafe_fn)]

use super::accumulator::Accumulator;
use super::network::{self, Aligned};
use super::simd;
use super::{FT_QUANT, L1_NORMALISATION, L1_SIZE, L2_SIZE, L3_SIZE};
use crate::types::Color;


// ============================================================
// NNZ lookup table (compile-time)
// ============================================================

/// Sparse entry: indices of set bits in an 8-bit mask.
#[repr(C, align(16))]
#[derive(Clone, Copy)]
#[allow(dead_code)] // Used by AVX2/scalar find_nnz variants, not AVX-512
pub struct SparseEntry {
    pub indexes: [u16; 8],
    pub count: usize,
}

/// Precomputed NNZ table: for each byte value 0..255, stores the indices of set bits.
#[allow(dead_code)] // Used by AVX2/scalar find_nnz variants, not AVX-512
const NNZ_TABLE: [SparseEntry; 256] = {
    let mut table = [SparseEntry {
        indexes: [0u16; 8],
        count: 0,
    }; 256];
    let mut byte = 0usize;
    while byte < 256 {
        let mut count = 0;
        let mut bit = 0;
        while bit < 8 {
            if (byte & (1 << bit)) != 0 {
                table[byte].indexes[count] = bit as u16;
                count += 1;
            }
            bit += 1;
        }
        table[byte].count = count;
        byte += 1;
    }
    table
};

// ============================================================
// FT activation: within-side pairwise product
// ============================================================

/// Within-side pairwise activation.
///
/// For each perspective (STM and NSTM), combines PST accumulator with threat buffer,
/// then computes the pairwise product of `combined[0..320]` × `combined[320..640]`.
///
/// Clamping is asymmetric:
/// - First half (left): `clamp(x, 0, FT_QUANT)` — standard CReLU
/// - Second half (right): `min(x, FT_QUANT)` — no lower-bound clamp
///
/// Product: `u8( mulhi(left << 7, right) )` via unsigned packus saturation.
/// Output: `[stm_pairwise[320] | ntm_pairwise[320]]` = u8[640].
#[cfg(any(target_feature = "avx2", target_feature = "avx512f",
          all(target_arch = "aarch64", target_feature = "neon")))]
pub fn activate_ft(
    pst_acc: &Accumulator,
    combined: &Aligned<[[i16; L1_SIZE]; 2]>,
    stm: Color,
) -> Aligned<[u8; L1_SIZE]> {
    let mut output = Aligned([0u8; L1_SIZE]);
    let half = L1_SIZE / 2;

    unsafe {
        let zero = simd::splat_i16(0);
        let one = simd::splat_i16(FT_QUANT);

        for flip in 0..2usize {
            let pov = stm.index() ^ flip;
            let pst_input = pst_acc.values.0[pov].as_ptr();
            let comb_input = combined.0[pov].as_ptr();

            for i in (0..half).step_by(2 * simd::I16_LANES) {
                // Load PST + threats for left half (two SIMD-width chunks)
                let pst_l1: _ = *pst_input.add(i).cast();
                let pst_l2: _ = *pst_input.add(i + simd::I16_LANES).cast();
                let comb_l1: _ = *comb_input.add(i).cast();
                let comb_l2: _ = *comb_input.add(i + simd::I16_LANES).cast();
                let lhs1 = simd::add_i16(pst_l1, comb_l1);
                let lhs2 = simd::add_i16(pst_l2, comb_l2);

                // Load PST + threats for right half
                let pst_r1: _ = *pst_input.add(i + half).cast();
                let pst_r2: _ = *pst_input.add(i + half + simd::I16_LANES).cast();
                let comb_r1: _ = *comb_input.add(i + half).cast();
                let comb_r2: _ = *comb_input.add(i + half + simd::I16_LANES).cast();
                let rhs1 = simd::add_i16(pst_r1, comb_r1);
                let rhs2 = simd::add_i16(pst_r2, comb_r2);

                // Asymmetric clamp: left = [0, 255], right = (-∞, 255]
                let lhs1_clipped = simd::clamp_i16(lhs1, zero, one);
                let lhs2_clipped = simd::clamp_i16(lhs2, zero, one);
                let rhs1_clipped = simd::min_i16(rhs1, one);
                let rhs2_clipped = simd::min_i16(rhs2, one);

                // Pairwise product via mulhi trick: (left * right) >> FT_SHIFT (= 9)
                // mulhi(left << 7, right) = (left * right * 128) >> 16 = (left * right) >> 9
                let shifted1 = simd::shift_left_i16::<7>(lhs1_clipped);
                let shifted2 = simd::shift_left_i16::<7>(lhs2_clipped);

                let product1 = simd::mul_high_i16(shifted1, rhs1_clipped);
                let product2 = simd::mul_high_i16(shifted2, rhs2_clipped);

                // Pack i16 → u8 with unsigned saturation (negative products → 0).
                // PST accumulators are pre-permuted so packus output is linear.
                let packed = simd::packus(product1, product2);
                *output.0.as_mut_ptr().add(i + flip * half).cast() = packed;
            }
        }
    }

    output
}

/// Within-side pairwise activation (scalar fallback).
#[cfg(not(any(target_feature = "avx2", target_feature = "avx512f",
              all(target_arch = "aarch64", target_feature = "neon"))))]
pub fn activate_ft(
    pst_acc: &Accumulator,
    combined: &Aligned<[[i16; L1_SIZE]; 2]>,
    stm: Color,
) -> Aligned<[u8; L1_SIZE]> {
    let mut output = Aligned([0u8; L1_SIZE]);
    let half = L1_SIZE / 2;

    for flip in 0..2usize {
        let pov = stm.index() ^ flip;
        let pst_vals = &pst_acc.values.0[pov];
        let comb_vals = &combined.0[pov];

        for i in 0..half {
            let left = (pst_vals[i] + comb_vals[i]).clamp(0, FT_QUANT);
            // Right: asymmetric — only upper clamp (matches SIMD packus saturation)
            let right = (pst_vals[i + half] + comb_vals[i + half]).min(FT_QUANT);
            let product = ((left as i32 * right as i32) >> super::FT_SHIFT).clamp(0, 255) as u8;
            output.0[i + flip * half] = product;
        }
    }

    output
}

// ============================================================
// NNZ extraction
// ============================================================

/// Padding for SIMD stores that may write past valid NNZ entries.
const NNZ_PAD: usize = 32;
/// Total NNZ array size including padding.
pub const NNZ_SIZE: usize = L1_SIZE / 4 + NNZ_PAD;

/// Extract indices of non-zero 4-byte groups from the FT output.
///
/// Returns (indices, count). Each index identifies a group of 4 input neurons
/// that has at least one non-zero byte.
///
/// AVX-512VBMI: uses `maskz_compress_epi16` (no lookup table), 32 groups/iter.
/// AVX2: uses NNZ_TABLE + SSE `storeu_si128`.
/// Scalar: NNZ_TABLE + scalar copy.
#[cfg(all(target_feature = "avx512vl", target_feature = "avx512vbmi"))]
pub fn find_nnz(ft_out: &Aligned<[u8; L1_SIZE]>) -> ([u16; NNZ_SIZE], usize) {
    use std::arch::x86_64::*;

    let mut indexes = [0u16; NNZ_SIZE];
    let mut count = 0;

    unsafe {
        // Process 32 i32 groups per iteration (2 × 16-lane nnz_bitmask = 32-bit mask).
        // L1_SIZE / (4 * 32) = 640 / 128 = 5 iterations exactly.
        let increment = _mm512_set1_epi16(32);
        let mut base = _mm512_set_epi16(
            31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
            15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0,
        );

        let ptr = ft_out.0.as_ptr();

        for i in (0..L1_SIZE).step_by(4 * simd::I16_LANES) {
            let mask0 = simd::nnz_bitmask(*ptr.add(i).cast());
            let mask1 = simd::nnz_bitmask(*ptr.add(i + 2 * simd::I16_LANES).cast());

            let mask: u32 = (mask1 as u32) << 16 | mask0 as u32;

            let compressed = _mm512_maskz_compress_epi16(mask, base);
            _mm512_storeu_si512(indexes.as_mut_ptr().add(count).cast(), compressed);
            count += mask.count_ones() as usize;

            base = _mm512_add_epi16(base, increment);
        }
    }

    (indexes, count)
}

/// AVX2 fallback: NNZ_TABLE + SSE store.
#[cfg(all(target_feature = "avx2", not(all(target_feature = "avx512vl", target_feature = "avx512vbmi"))))]
pub fn find_nnz(ft_out: &Aligned<[u8; L1_SIZE]>) -> ([u16; NNZ_SIZE], usize) {
    use std::arch::x86_64::*;

    let mut indexes = [0u16; NNZ_SIZE];
    let mut count = 0;

    unsafe {
        let increment = _mm_set1_epi16(8);
        let mut base = _mm_setzero_si128();
        let ptr = ft_out.0.as_ptr() as *const i32;

        for chunk_start in (0..L1_SIZE / 4).step_by(simd::I32_LANES) {
            let mask = simd::nnz_bitmask(*ptr.add(chunk_start).cast());

            // Process mask in 8-bit slices with SSE store
            let mut remaining = mask;
            let mut offset = 0;
            while offset < simd::I32_LANES {
                let slice = (remaining & 0xFF) as usize;
                let entry = &NNZ_TABLE[slice];

                // Load 8 × u16 from table, add base offset, store via SSE
                let entry_vec = _mm_load_si128(entry.indexes.as_ptr().cast());
                _mm_storeu_si128(
                    indexes.as_mut_ptr().add(count).cast(),
                    _mm_add_epi16(base, entry_vec),
                );

                count += entry.count;
                remaining >>= 8;
                offset += 8;
                base = _mm_add_epi16(base, increment);
            }
        }
    }

    (indexes, count)
}

/// Scalar fallback: NNZ_TABLE + scalar copy.
#[cfg(not(target_feature = "avx2"))]
pub fn find_nnz(ft_out: &Aligned<[u8; L1_SIZE]>) -> ([u16; NNZ_SIZE], usize) {
    let mut indexes = [0u16; NNZ_SIZE];
    let mut count = 0;

    unsafe {
        let mut base: u16 = 0;
        let ptr = ft_out.0.as_ptr() as *const i32;

        for chunk_start in (0..L1_SIZE / 4).step_by(simd::I32_LANES) {
            let mask = simd::nnz_bitmask(*ptr.add(chunk_start).cast());

            let mut remaining = mask;
            let mut offset: u16 = 0;
            while offset < simd::I32_LANES as u16 {
                let slice = (remaining & 0xFF) as usize;
                let entry = &NNZ_TABLE[slice];
                for k in 0..entry.count {
                    indexes[count + k] = base + offset + entry.indexes[k];
                }
                count += entry.count;
                remaining >>= 8;
                offset += 8;
            }

            base += simd::I32_LANES as u16;
        }
    }

    (indexes, count)
}

// ============================================================
// L1: sparse matmul → CReLU+squared
// ============================================================

/// L1 sparse inference: process only non-zero input groups.
///
/// Sparse u8[640] × i8 weights → i32[L2_SIZE], dequantize, add bias.
/// Activation: CReLU+squared = `[clamp(x, 0, 1) ; clamp(x, 0, 1)²]` → f32[2×L2_SIZE].
///
/// The two halves feed L2 (full 32 values) and provide the L3 skip connection.
pub fn propagate_l1(
    ft_out: &Aligned<[u8; L1_SIZE]>,
    nnz: &[u16],
    bucket: usize,
) -> Aligned<[f32; 2 * L2_SIZE]> {
    let params = network::params();

    unsafe {
        let packed = ft_out.0.as_ptr() as *const i32;

        // 3 independent accumulator chains for latency hiding on VNNI (5-cycle latency).
        let n_vecs = L2_SIZE / simd::F32_LANES;
        let mut chain0 = [simd::zeroed_i32(); L2_SIZE / simd::F32_LANES];
        let mut chain1 = [simd::zeroed_i32(); L2_SIZE / simd::F32_LANES];
        let mut chain2 = [simd::zeroed_i32(); L2_SIZE / simd::F32_LANES];

        let mut triples = nnz.chunks_exact(3);
        for triple in &mut triples {
            let idx0 = triple[0] as usize;
            let idx1 = triple[1] as usize;
            let idx2 = triple[2] as usize;

            let in0 = simd::splat_i32(*packed.add(idx0));
            let in1 = simd::splat_i32(*packed.add(idx1));
            let in2 = simd::splat_i32(*packed.add(idx2));

            let w0 = params.l1_weights.0[bucket][idx0].as_ptr();
            let w1 = params.l1_weights.0[bucket][idx1].as_ptr();
            let w2 = params.l1_weights.0[bucket][idx2].as_ptr();

            for j in 0..n_vecs {
                let off = j * simd::F32_LANES * 4;
                chain0[j] = simd::dpbusd(chain0[j], in0, *w0.add(off).cast());
                chain1[j] = simd::dpbusd(chain1[j], in1, *w1.add(off).cast());
                chain2[j] = simd::dpbusd(chain2[j], in2, *w2.add(off).cast());
            }
        }

        // Handle remainder (0, 1, or 2 leftover NNZ indices)
        let rem = triples.remainder();
        if rem.len() >= 1 {
            let idx = rem[0] as usize;
            let input = simd::splat_i32(*packed.add(idx));
            let w = params.l1_weights.0[bucket][idx].as_ptr();
            for j in 0..n_vecs {
                let off = j * simd::F32_LANES * 4;
                chain0[j] = simd::dpbusd(chain0[j], input, *w.add(off).cast());
            }
        }
        if rem.len() >= 2 {
            let idx = rem[1] as usize;
            let input = simd::splat_i32(*packed.add(idx));
            let w = params.l1_weights.0[bucket][idx].as_ptr();
            for j in 0..n_vecs {
                let off = j * simd::F32_LANES * 4;
                chain1[j] = simd::dpbusd(chain1[j], input, *w.add(off).cast());
            }
        }

        // Merge chains
        let mut pre_act = [simd::zeroed_i32(); L2_SIZE / simd::F32_LANES];
        for j in 0..n_vecs {
            pre_act[j] = simd::add_i32(simd::add_i32(chain0[j], chain1[j]), chain2[j]);
        }

        // Dequantize + bias + CReLU+squared:
        // output[j]          = clamp(x, 0, 1)          ← CReLU part
        // output[j + L2_SIZE] = clamp(x, 0, 1)²         ← squared part
        let mut output = Aligned([0.0f32; 2 * L2_SIZE]);
        let zero = simd::zero_f32();
        let one = simd::splat_f32(1.0);
        let dequant = simd::splat_f32(L1_NORMALISATION);

        for j in 0..n_vecs {
            let biases = *params.l1_biases.0[bucket].as_ptr().add(j * simd::F32_LANES).cast();
            let val = simd::mul_add_f32(simd::convert_to_f32(pre_act[j]), dequant, biases);
            let clamped = simd::clamp_f32(val, zero, one);
            // squared = min(val², 1) using raw val (NOT clamped).
            // For negative val: clamped²=0 is wrong; min(val²,1)>0 is correct.
            let squared = simd::clamp_f32(simd::mul_f32(val, val), zero, one);

            // CReLU output at j*F32_LANES, squared at L2_SIZE + j*F32_LANES
            *output.0.as_mut_ptr().add(j * simd::F32_LANES).cast() = clamped;
            *output.0.as_mut_ptr().add(L2_SIZE + j * simd::F32_LANES).cast() = squared;
        }

        output
    }
}

// ============================================================
// L2: dense f32 FMA + squared activation
// ============================================================

/// L2 dense layer: f32 matmul + bias → squared activation (SIMD splat+FMA).
///
/// Input: f32[2×L2_SIZE] (the full CReLU+squared L1 output).
/// Weight layout: `[bucket][input=2×L2_SIZE][output=L3_SIZE]`.
/// Activation: `clamp(x, 0, 1)²` (identical to SCReLU).
#[cfg(any(target_feature = "avx2", target_feature = "avx512f",
          all(target_arch = "aarch64", target_feature = "neon")))]
pub fn propagate_l2(l1_out: &Aligned<[f32; 2 * L2_SIZE]>, bucket: usize) -> Aligned<[f32; L3_SIZE]> {
    let params = network::params();

    unsafe {
        // Init accumulators with biases
        let mut output = Aligned([0.0f32; L3_SIZE]);
        std::ptr::copy_nonoverlapping(
            params.l2_biases.0[bucket].as_ptr(),
            output.0.as_mut_ptr(),
            L3_SIZE,
        );

        // Splat+FMA: broadcast each input, FMA across all L3_SIZE outputs
        for i in 0..l1_out.0.len() {
            let input = simd::splat_f32(l1_out.0[i]);
            let w = params.l2_weights.0[bucket][i].as_ptr();
            for j in (0..L3_SIZE).step_by(simd::F32_LANES) {
                let out: *mut _ = output.0.as_mut_ptr().add(j).cast();
                *out = simd::mul_add_f32(*w.add(j).cast(), input, *out);
            }
        }

        // L2 activation: SCReLU = clamp(x, 0, 1)²
        let zero = simd::zero_f32();
        let one = simd::splat_f32(1.0);
        for j in (0..L3_SIZE).step_by(simd::F32_LANES) {
            let out: *mut _ = output.0.as_mut_ptr().add(j).cast();
            let clamped = simd::clamp_f32(*out, zero, one);
            *out = simd::mul_f32(clamped, clamped);
        }

        output
    }
}

/// L2 dense layer: scalar fallback.
#[cfg(not(any(target_feature = "avx2", target_feature = "avx512f",
              all(target_arch = "aarch64", target_feature = "neon"))))]
pub fn propagate_l2(l1_out: &Aligned<[f32; 2 * L2_SIZE]>, bucket: usize) -> Aligned<[f32; L3_SIZE]> {
    let mut output = Aligned([0.0f32; L3_SIZE]);
    let params = network::params();

    for j in 0..L3_SIZE {
        let mut sum = params.l2_biases.0[bucket][j];
        for i in 0..l1_out.0.len() {
            sum += params.l2_weights.0[bucket][i][j] * l1_out.0[i];
        }
        // L2 activation: SCReLU = clamp(x, 0, 1)²
        let clamped = sum.clamp(0.0, 1.0);
        output.0[j] = clamped * clamped;
    }

    output
}

// ============================================================
// L3: output with L1 skip connection
// ============================================================

/// L3 output layer: `dot(concat(l2_out, l1_out), l3_weights) + bias` → scalar.
///
/// Skip connection: L1 output (f32[2×L2_SIZE]) is concatenated with L2 output (f32[L3_SIZE]).
/// Total input: `L3_SIZE + 2×L2_SIZE = 64` floats.
/// Weights: `l3_weights[bucket][64]`.
pub fn propagate_l3(
    l2_out: &Aligned<[f32; L3_SIZE]>,
    l1_out: &Aligned<[f32; 2 * L2_SIZE]>,
    bucket: usize,
) -> f32 {
    let params = network::params();

    unsafe {
        let l2_ptr = l2_out.0.as_ptr();
        let l1_ptr = l1_out.0.as_ptr();
        let weights = params.l3_weights.0[bucket].as_ptr();

        let mut acc = [simd::zero_f32(); simd::HSUM_VECS];

        // First half: l2_out[0..L3_SIZE] × weights[0..L3_SIZE]
        for (lane, result) in acc.iter_mut().enumerate() {
            for i in (0..L3_SIZE).step_by(simd::HSUM_VECS * simd::F32_LANES) {
                let off = i + lane * simd::F32_LANES;
                *result = simd::mul_add_f32(*weights.add(off).cast(), *l2_ptr.add(off).cast(), *result);
            }
        }

        // Second half: l1_out[0..2*L2_SIZE] × weights[L3_SIZE..L3_SIZE+2*L2_SIZE]
        for (lane, result) in acc.iter_mut().enumerate() {
            for i in (0..(2 * L2_SIZE)).step_by(simd::HSUM_VECS * simd::F32_LANES) {
                let off = i + lane * simd::F32_LANES;
                *result = simd::mul_add_f32(
                    *weights.add(L3_SIZE + off).cast(),
                    *l1_ptr.add(off).cast(),
                    *result,
                );
            }
        }

        simd::horizontal_sum(acc) + params.l3_biases.0[bucket]
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::accumulator::Accumulator;
    use super::super::threats;
    use super::super::{FT_QUANT, FT_SHIFT, L1_SIZE};

    /// Reference scalar activate_ft (natural order, asymmetric clamp).
    fn activate_ft_reference(
        pst_acc: &Accumulator,
        combined: &Aligned<[[i16; L1_SIZE]; 2]>,
        stm: Color,
    ) -> [u8; L1_SIZE] {
        let mut output = [0u8; L1_SIZE];
        let half = L1_SIZE / 2;
        for flip in 0..2usize {
            let pov = stm.index() ^ flip;
            let pst_vals = &pst_acc.values.0[pov];
            let comb_vals = &combined.0[pov];
            for i in 0..half {
                let left = (pst_vals[i] + comb_vals[i]).clamp(0, FT_QUANT);
                let right = (pst_vals[i + half] + comb_vals[i + half]).min(FT_QUANT);
                let product = ((left as i32 * right as i32) >> FT_SHIFT).clamp(0, 255) as u8;
                output[i + flip * half] = product;
            }
        }
        output
    }

    /// Reference scalar L1 propagation with CReLU+squared output.
    fn propagate_l1_reference(ft_out: &[u8; L1_SIZE], bucket: usize) -> [f32; 2 * L2_SIZE] {
        let mut pre_act = [0i32; L2_SIZE];
        let weights = &network::params().l1_weights.0[bucket];
        for ig in 0..L1_SIZE / 4 {
            for b in 0..4 {
                let input = ft_out[ig * 4 + b] as i32;
                if input == 0 { continue; }
                for j in 0..L2_SIZE {
                    pre_act[j] += input * weights[ig][j * 4 + b] as i32;
                }
            }
        }
        let mut output = [0.0f32; 2 * L2_SIZE];
        for j in 0..L2_SIZE {
            let val = pre_act[j] as f32 * L1_NORMALISATION
                + network::params().l1_biases.0[bucket][j];
            let clamped = val.clamp(0.0, 1.0);
            output[j] = clamped;
            output[j + L2_SIZE] = (val * val).min(1.0); // min(val², 1)
        }
        output
    }

    #[test]
    fn test_activate_ft_zeroed_weights() {
        let pst_acc = Accumulator::new();
        let combined = Aligned([[0i16; L1_SIZE]; 2]);
        let simd_out = activate_ft(&pst_acc, &combined, Color::White);
        let ref_out = activate_ft_reference(&pst_acc, &combined, Color::White);
        for i in 0..L1_SIZE {
            assert_eq!(simd_out.0[i], ref_out[i], "activate_ft mismatch at {i}");
        }
    }

    #[test]
    fn test_find_nnz_all_zero() {
        let ft_out = Aligned([0u8; L1_SIZE]);
        let (_, count) = find_nnz(&ft_out);
        assert_eq!(count, 0, "all-zero should have 0 NNZ groups");
    }

    #[test]
    fn test_find_nnz_all_nonzero() {
        let mut ft_out = Aligned([0u8; L1_SIZE]);
        for i in 0..L1_SIZE {
            ft_out.0[i] = 1;
        }
        let (_, count) = find_nnz(&ft_out);
        assert_eq!(count, L1_SIZE / 4, "all non-zero should have {} NNZ groups", L1_SIZE / 4);
    }

    #[test]
    fn test_find_nnz_sparse() {
        let mut ft_out = Aligned([0u8; L1_SIZE]);
        ft_out.0[0] = 42;
        ft_out.0[8] = 99;
        let (indices, count) = find_nnz(&ft_out);
        assert_eq!(count, 2, "should have 2 NNZ groups");
        assert_eq!(indices[0], 0, "first NNZ group should be 0");
        assert_eq!(indices[1], 2, "second NNZ group should be 2");
    }

    #[test]
    fn test_propagate_l1_zero_input() {
        let ft_out = Aligned([0u8; L1_SIZE]);
        let (nnz, count) = find_nnz(&ft_out);
        let simd_out = propagate_l1(&ft_out, &nnz[..count], 0);
        let ref_out = propagate_l1_reference(&ft_out.0, 0);
        for j in 0..2 * L2_SIZE {
            assert!(
                (simd_out.0[j] - ref_out[j]).abs() < 1e-6,
                "propagate_l1 zero mismatch at {j}: simd={}, ref={}", simd_out.0[j], ref_out[j]
            );
        }
    }

    #[test]
    fn test_propagate_l1_nonzero_input() {
        let mut ft_out = Aligned([0u8; L1_SIZE]);
        for i in 0..L1_SIZE {
            ft_out.0[i] = ((i * 7 + 13) % 127) as u8;
        }
        let (nnz, count) = find_nnz(&ft_out);
        let simd_out = propagate_l1(&ft_out, &nnz[..count], 0);
        let ref_out = propagate_l1_reference(&ft_out.0, 0);
        for j in 0..2 * L2_SIZE {
            assert!(
                (simd_out.0[j] - ref_out[j]).abs() < 1e-4,
                "propagate_l1 mismatch at {j}: simd={}, ref={}", simd_out.0[j], ref_out[j]
            );
        }
    }

    #[test]
    fn test_nnz_table_correctness() {
        for byte in 0..256u16 {
            let entry = &NNZ_TABLE[byte as usize];
            assert_eq!(entry.count, byte.count_ones() as usize, "NNZ_TABLE[{byte}] count");
            let mut j = 0;
            for bit in 0..8 {
                if (byte & (1 << bit)) != 0 {
                    assert_eq!(entry.indexes[j], bit as u16, "NNZ_TABLE[{byte}] index {j}");
                    j += 1;
                }
            }
        }
    }

    #[test]
    fn test_compute_full_threats_kk_endgame() {
        let pos = crate::position::Position::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let combined = threats::compute_full_threats(&pos);
        // With zeroed weights the buffer should be all zeros.
        for pov in 0..2 {
            for i in 0..L1_SIZE {
                assert_eq!(combined.0[pov][i], 0,
                    "KK endgame should be 0, got {} at pov={pov} i={i}", combined.0[pov][i]);
            }
        }
    }

    #[test]
    fn test_compute_full_threats_startpos_no_crash() {
        let pos = crate::position::Position::from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ).unwrap();
        // Should not crash; with zeroed weights values are zero.
        let combined = threats::compute_full_threats(&pos);
        for pov in 0..2 {
            for i in 0..L1_SIZE {
                assert_eq!(combined.0[pov][i], 0);
            }
        }
    }
}
