//! NNUE forward pass: NNZ extraction + per-stage entry points.
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
//!
//! The register-width stages (2, 4-6) live in `nnue::kernels`, monomorphized per
//! SIMD backend; this module keeps `find_nnz` — whose three implementations are
//! different *algorithms*, not the same loop at different widths — plus thin
//! wrappers over the current kernel instance for tests and callers.

#![allow(unsafe_op_in_unsafe_fn)]

use super::network::Aligned;
use super::L1_SIZE;

#[cfg(test)]
use super::accumulator::Accumulator;
#[cfg(test)]
use super::kernels::k;
#[cfg(test)]
use super::{L2_SIZE, L3_SIZE};
#[cfg(test)]
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
#[allow(dead_code)] // Used by AVX2/generic find_nnz variants, not AVX-512
pub(crate) const NNZ_TABLE: [SparseEntry; 256] = {
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
/// AVX-512VBMI2: uses `maskz_compress_epi16` (no lookup table), 32 groups/iter.
/// AVX2: uses NNZ_TABLE + SSE `storeu_si128`.
/// Otherwise: NNZ_TABLE walk at the backend's lane width (`find_nnz_generic`).
///
/// This wrapper is the *static* selection: non-x86 targets and builds whose
/// target pins VBMI2 call it directly; other x86 builds read the resolved
/// table inside `forward_dense` instead (hence dead on some of them).
#[allow(dead_code)]
#[cfg(all(target_feature = "avx512bw", target_feature = "avx512vbmi2"))]
pub fn find_nnz(ft_out: &Aligned<[u8; L1_SIZE]>) -> ([u16; NNZ_SIZE], usize) {
    unsafe { find_nnz_compress512(ft_out) }
}

#[allow(dead_code)]
#[cfg(all(
    target_feature = "avx2",
    not(all(target_feature = "avx512bw", target_feature = "avx512vbmi2"))
))]
pub fn find_nnz(ft_out: &Aligned<[u8; L1_SIZE]>) -> ([u16; NNZ_SIZE], usize) {
    unsafe { find_nnz_avx2(ft_out) }
}

#[allow(dead_code)]
#[cfg(not(target_feature = "avx2"))]
pub fn find_nnz(ft_out: &Aligned<[u8; L1_SIZE]>) -> ([u16; NNZ_SIZE], usize) {
    unsafe { super::kernels::k::find_nnz_generic(ft_out) }
}

/// AVX-512 form: `maskz_compress_epi16` builds the index list without a table.
///
/// `VPCOMPRESSW` is an AVX-512VBMI2 instruction (VBMI is not enough).
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)] // selected per build target (and, later, per detected CPU)
#[cfg_attr(
    not(all(
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi2"
    )),
    target_feature(enable = "avx512f,avx512bw,avx512vbmi2")
)]
pub unsafe fn find_nnz_compress512(
    ft_out: &Aligned<[u8; L1_SIZE]>,
) -> ([u16; NNZ_SIZE], usize) {
    use crate::nnue::simd::avx512 as simd;
    use std::arch::x86_64::*;

    let mut indexes = [0u16; NNZ_SIZE];
    let mut count = 0;

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

    (indexes, count)
}

/// AVX2 form: NNZ_TABLE + SSE store.
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)] // selected per build target (and, later, per detected CPU)
#[cfg_attr(not(target_feature = "avx2"), target_feature(enable = "avx2"))]
pub unsafe fn find_nnz_avx2(ft_out: &Aligned<[u8; L1_SIZE]>) -> ([u16; NNZ_SIZE], usize) {
    use crate::nnue::simd::avx2 as simd;
    use std::arch::x86_64::*;

    let mut indexes = [0u16; NNZ_SIZE];
    let mut count = 0;

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

    (indexes, count)
}

// ============================================================
// Per-stage entry points over the current kernel instance (tests only —
// the engine itself goes through the `forward_dense` composite)
// ============================================================

/// Within-side pairwise FT activation. See `kernels` for the loop itself.
#[cfg(test)]
pub fn activate_ft(
    pst_acc: &Accumulator,
    combined: &Aligned<[[i16; L1_SIZE]; 2]>,
    stm: Color,
) -> Aligned<[u8; L1_SIZE]> {
    unsafe { k::activate_ft(pst_acc, combined, stm) }
}

/// L1 sparse inference: process only non-zero input groups.
#[cfg(test)]
pub fn propagate_l1(
    ft_out: &Aligned<[u8; L1_SIZE]>,
    nnz: &[u16],
    bucket: usize,
) -> Aligned<[f32; 2 * L2_SIZE]> {
    unsafe { k::propagate_l1(ft_out, nnz, bucket) }
}

/// L2 dense layer: f32 matmul + bias → squared activation.
#[cfg(test)]
pub fn propagate_l2(l1_out: &Aligned<[f32; 2 * L2_SIZE]>, bucket: usize) -> Aligned<[f32; L3_SIZE]> {
    unsafe { k::propagate_l2(l1_out, bucket) }
}

/// L3 output layer: `dot(concat(l2_out, l1_out), l3_weights) + bias` → scalar.
#[cfg(test)]
pub fn propagate_l3(
    l2_out: &Aligned<[f32; L3_SIZE]>,
    l1_out: &Aligned<[f32; 2 * L2_SIZE]>,
    bucket: usize,
) -> f32 {
    unsafe { k::propagate_l3(l2_out, l1_out, bucket) }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::accumulator::Accumulator;
    use super::super::threats;
    use super::super::network;
    use super::super::{FT_QUANT, FT_SHIFT, L1_SIZE, L1_NORMALISATION};

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

    /// The zeroed test above proves nothing about the arithmetic: every clamp, shift,
    /// multiply-high and pack agrees with every other on zero. This one drives them with
    /// numbers that cross each branch — negatives, which the left half clamps away and
    /// the right half keeps; values above `FT_QUANT`, which both ceilings catch; and
    /// products large enough to saturate the unsigned pack.
    ///
    /// Only where `packus` keeps its lanes in order. The AVX2 and AVX-512 packs
    /// interleave, and the engine pays for that by permuting the *weights* so the
    /// interleaving undoes itself; an accumulator built here by hand has not been
    /// through that, so on those targets the output is legitimately a permutation of
    /// the reference and there is nothing to compare. The vector tiers are covered
    /// with pre-permuted inputs by the cross-backend tests in `nnue::kernels`.
    #[cfg(not(any(target_feature = "avx2", target_feature = "avx512f")))]
    #[test]
    fn activate_ft_matches_the_reference_on_real_numbers() {
        let mut pst_acc = Accumulator::new();
        let mut combined = Aligned([[0i16; L1_SIZE]; 2]);
        for pov in 0..2 {
            for i in 0..L1_SIZE {
                // Kept well inside i16 so the reference's own addition cannot overflow.
                pst_acc.values.0[pov][i] = (((i * 37 + pov * 11) % 811) as i32 - 400) as i16;
                combined.0[pov][i] = (((i * 53 + pov * 7) % 509) as i32 - 250) as i16;
            }
        }

        for stm in [Color::White, Color::Black] {
            let simd_out = activate_ft(&pst_acc, &combined, stm);
            let ref_out = activate_ft_reference(&pst_acc, &combined, stm);
            let mismatches = (0..L1_SIZE).filter(|&i| simd_out.0[i] != ref_out[i]).count();
            if let Some(i) = (0..L1_SIZE).find(|&i| simd_out.0[i] != ref_out[i]) {
                panic!(
                    "activate_ft disagrees with the reference at {i} (stm {stm:?}): \
                     simd={}, ref={} — {mismatches} of {L1_SIZE} lanes differ",
                    simd_out.0[i], ref_out[i]
                );
            }
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
    fn test_compute_full_threats_startpos() {
        let pos = crate::position::Position::from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ).unwrap();
        let combined = threats::compute_full_threats(&pos);

        if !crate::nnue::network::has_network() {
            // Nothing embedded, so every weight is zero and so is every sum.
            for pov in 0..2 {
                for i in 0..L1_SIZE {
                    assert_eq!(combined.0[pov][i], 0, "zeroed weights must sum to zero");
                }
            }
            return;
        }

        // With a trained network the start position has threats on the board, so the
        // sums are emphatically not zero — the old form of this test asserted they were
        // and only ever passed because it was run without a network.
        assert!(
            combined.0[0].iter().any(|&v| v != 0),
            "the start position has threats; their weights cannot all cancel"
        );
        // The start position is the same for both players, so both perspectives must
        // encode it identically. If they ever diverge the network is judging the very
        // first position of every game asymmetrically.
        assert_eq!(
            combined.0[0], combined.0[1],
            "the two perspectives disagree on a symmetric position"
        );
    }
}
