//! NNUE forward pass: NNZ extraction + per-stage entry points.
//!
//! Pipeline per evaluation:
//!   1. `activate_ft(acc)` — the accumulator (piece squares, pawn pairs and threats
//!      summed), within-side pairwise product (i16 → u8[L1])
//!   2. `find_nnz` — extract non-zero 4-byte group indices for sparse L1
//!   3. `propagate_l1` — sparse u8 × i8 → i32[32], plus the bias
//!   4. `activate_l1` — `[x² >> 21, x >> 7]` clipped → u8[64]
//!   5. `propagate_l2` — dense u8 × i8 → gate and value, `h = g² · v >> 14` → i32[32]
//!   6. `propagate_l3` — u8[64] and h[32] × l3 weights + bias, plus the skip
//!      `l1[30] − l1[31]` → the output, unit 2^14
//!
//! The register-width stages live in `nnue::kernels`, monomorphized per SIMD
//! backend; this module keeps `find_nnz` — whose three implementations are
//! different *algorithms*, not the same loop at different widths — the plain
//! reference every tier is held against, and thin wrappers for the tests.

#![allow(unsafe_op_in_unsafe_fn)]

use super::L1_SIZE;
use super::network::Aligned;

#[cfg(test)]
use super::kernels::k;
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

// Non-zero 4-byte groups fed to the sparse L1, tallied per thread (feature `stats`):
// `(forward passes, groups)`. The kernels have no `ThreadData` to write to, so the
// search drains this into its statistics after each evaluation.
#[cfg(feature = "stats")]
thread_local! {
    static NNZ_TALLY: std::cell::Cell<(u64, u64)> = const { std::cell::Cell::new((0, 0)) };
}

/// Count one forward pass and the non-zero groups its L1 had to read.
#[cfg(feature = "stats")]
#[inline]
pub fn tally_nnz(count: usize) {
    debug_assert!(
        count <= L1_SIZE / 4,
        "nnz tally: {count} groups of {}",
        L1_SIZE / 4
    );
    NNZ_TALLY.with(|t| {
        let (passes, groups) = t.get();
        t.set((passes + 1, groups + count as u64));
    });
}

/// What this thread has tallied since the last call, reset to zero.
#[cfg(feature = "stats")]
pub fn take_nnz_tally() -> (u64, u64) {
    NNZ_TALLY.with(|t| t.replace((0, 0)))
}

/// Extract indices of non-zero 4-byte groups from the FT output.
///
/// Writes the indexes to `out` and returns their count. Each index identifies a
/// group of 4 input neurons that has at least one non-zero byte. `out` must have
/// room for `NNZ_SIZE` indexes: the slots past the count may be written too
/// (`NNZ_PAD`), and need not be initialised.
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
pub unsafe fn find_nnz(ft_out: &Aligned<[u8; L1_SIZE]>, out: *mut u16) -> usize {
    unsafe { find_nnz_compress512(ft_out, out) }
}

#[allow(dead_code)]
#[cfg(all(
    target_feature = "avx2",
    not(all(target_feature = "avx512bw", target_feature = "avx512vbmi2"))
))]
pub unsafe fn find_nnz(ft_out: &Aligned<[u8; L1_SIZE]>, out: *mut u16) -> usize {
    unsafe { find_nnz_avx2(ft_out, out) }
}

#[allow(dead_code)]
#[cfg(not(target_feature = "avx2"))]
pub unsafe fn find_nnz(ft_out: &Aligned<[u8; L1_SIZE]>, out: *mut u16) -> usize {
    unsafe { super::kernels::k::find_nnz_generic(ft_out, out) }
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
pub unsafe fn find_nnz_compress512(ft_out: &Aligned<[u8; L1_SIZE]>, indexes: *mut u16) -> usize {
    use crate::nnue::simd::avx512 as simd;
    use std::arch::x86_64::*;

    let mut count = 0;

    // Process 32 i32 groups per iteration (2 × 16-lane nnz_bitmask = 32-bit mask).
    // L1_SIZE / (4 * 32) = 1024 / 128 = 8 iterations exactly.
    let increment = _mm512_set1_epi16(32);
    let mut base = _mm512_set_epi16(
        31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9,
        8, 7, 6, 5, 4, 3, 2, 1, 0,
    );

    let ptr = ft_out.0.as_ptr();

    for i in (0..L1_SIZE).step_by(4 * simd::I16_LANES) {
        let mask0 = simd::nnz_bitmask(*ptr.add(i).cast());
        let mask1 = simd::nnz_bitmask(*ptr.add(i + 2 * simd::I16_LANES).cast());

        let mask: u32 = (mask1 as u32) << 16 | mask0 as u32;

        let compressed = _mm512_maskz_compress_epi16(mask, base);
        _mm512_storeu_si512(indexes.add(count).cast(), compressed);
        count += mask.count_ones() as usize;

        base = _mm512_add_epi16(base, increment);
    }

    count
}

/// AVX2 form: NNZ_TABLE + SSE store.
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)] // selected per build target (and, later, per detected CPU)
#[cfg_attr(not(target_feature = "avx2"), target_feature(enable = "avx2"))]
pub unsafe fn find_nnz_avx2(ft_out: &Aligned<[u8; L1_SIZE]>, indexes: *mut u16) -> usize {
    use crate::nnue::simd::avx2 as simd;
    use std::arch::x86_64::*;

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
                indexes.add(count).cast(),
                _mm_add_epi16(base, entry_vec),
            );

            count += entry.count;
            remaining >>= 8;
            offset += 8;
            base = _mm_add_epi16(base, increment);
        }
    }

    count
}

// ============================================================
// Per-stage entry points over the current kernel instance (tests only —
// the engine itself goes through the `forward_dense` composite)
// ============================================================

/// Within-side pairwise FT activation. See `kernels` for the loop itself.
#[cfg(test)]
pub fn activate_ft(acc: &Aligned<[[i16; L1_SIZE]; 2]>, stm: Color) -> Aligned<[u8; L1_SIZE]> {
    unsafe { k::activate_ft(acc, stm) }
}

// ============================================================
// Reference dense layers
// ============================================================

/// The dense layers spelled out one value at a time from the whole FT output, no
/// sparsity, no vectors: the arithmetic every tier must reproduce to the bit, and the
/// one the trainer's integer evaluator mirrors. Unit 2^14, skip included.
#[cfg(test)]
pub(crate) fn dense_reference(
    params: &super::network::NNUEParams,
    ft_out: &Aligned<[u8; L1_SIZE]>,
    stack: usize,
) -> i32 {
    use super::{
        A1_SIZE, GATE_SHIFT, GATED, HIDDEN_MAX, L1_OUT, L1_SHIFT, L1_SQR_SHIFT, L2_OUT,
        L2_SHIFT, SKIP_NEG, SKIP_POS,
    };
    let mut l1 = [0i32; L1_OUT];
    for (o, out) in l1.iter_mut().enumerate() {
        let mut sum = params.l1_biases.0[stack][o];
        for i in 0..L1_SIZE {
            sum += ft_out.0[i] as i32 * params.l1_weights.0[stack][i / 4][o * 4 + i % 4] as i32;
        }
        *out = sum;
    }
    let mut a1 = [0i32; A1_SIZE];
    for o in 0..L1_OUT {
        let x = l1[o] as i64;
        a1[o] = ((x * x) >> L1_SQR_SHIFT).min(HIDDEN_MAX as i64) as i32;
        a1[L1_OUT + o] = (l1[o] >> L1_SHIFT).clamp(0, HIDDEN_MAX);
    }
    let mut l2 = [0i32; L2_OUT];
    for (o, out) in l2.iter_mut().enumerate() {
        let mut sum = params.l2_biases.0[stack][o];
        for (i, &a) in a1.iter().enumerate() {
            sum += a * params.l2_weights.0[stack][i / 4][o * 4 + i % 4] as i32;
        }
        *out = sum;
    }
    let mut out = params.l3_biases.0[stack];
    for (i, &a) in a1.iter().enumerate() {
        out += a * params.l3_weights_a.0[stack][i] as i32;
    }
    for j in 0..GATED {
        let g = (l2[j] >> L2_SHIFT).clamp(0, HIDDEN_MAX);
        let v = (l2[GATED + j] >> L2_SHIFT).clamp(-HIDDEN_MAX, HIDDEN_MAX);
        let h = (g * g * v) >> GATE_SHIFT;
        out += h * params.l3_weights_h.0[stack][j];
    }
    out + l1[SKIP_POS] - l1[SKIP_NEG]
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::super::accumulator::Accumulator;
    use super::super::threats;
    use super::super::{FT_QUANT, FT_SHIFT, L1_SIZE};
    use super::*;
    /// The single accumulator the kernels read: the two test inputs summed as the
    /// engine sums its features, wrapping.
    fn fused(pst: &Accumulator, comb: &Aligned<[[i16; L1_SIZE]; 2]>) -> Aligned<[[i16; L1_SIZE]; 2]> {
        let mut out = Aligned([[0i16; L1_SIZE]; 2]);
        for pov in 0..2 {
            for i in 0..L1_SIZE {
                out.0[pov][i] = pst.values.0[pov][i].wrapping_add(comb.0[pov][i]);
            }
        }
        out
    }


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

    #[test]
    fn test_activate_ft_zeroed_weights() {
        let pst_acc = Accumulator::new();
        let combined = Aligned([[0i16; L1_SIZE]; 2]);
        let simd_out = activate_ft(&fused(&pst_acc, &combined), Color::White);
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
            let simd_out = activate_ft(&fused(&pst_acc, &combined), stm);
            let ref_out = activate_ft_reference(&pst_acc, &combined, stm);
            let mismatches = (0..L1_SIZE)
                .filter(|&i| simd_out.0[i] != ref_out[i])
                .count();
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
        let count = unsafe { find_nnz(&ft_out, [0u16; NNZ_SIZE].as_mut_ptr()) };
        assert_eq!(count, 0, "all-zero should have 0 NNZ groups");
    }

    #[test]
    fn test_find_nnz_all_nonzero() {
        let mut ft_out = Aligned([0u8; L1_SIZE]);
        for i in 0..L1_SIZE {
            ft_out.0[i] = 1;
        }
        let count = unsafe { find_nnz(&ft_out, [0u16; NNZ_SIZE].as_mut_ptr()) };
        assert_eq!(
            count,
            L1_SIZE / 4,
            "all non-zero should have {} NNZ groups",
            L1_SIZE / 4
        );
    }

    #[test]
    fn test_find_nnz_sparse() {
        let mut ft_out = Aligned([0u8; L1_SIZE]);
        ft_out.0[0] = 42;
        ft_out.0[8] = 99;
        let mut indices = [0u16; NNZ_SIZE];
        let count = unsafe { find_nnz(&ft_out, indices.as_mut_ptr()) };
        assert_eq!(count, 2, "should have 2 NNZ groups");
        assert_eq!(indices[0], 0, "first NNZ group should be 0");
        assert_eq!(indices[1], 2, "second NNZ group should be 2");
    }

    /// The kernels of this build compute the reference's integers, on random weights
    /// and inputs (the cross-tier test in `kernels` holds every other tier to it).
    #[test]
    fn the_dense_layers_of_this_build_are_the_reference() {
        let mut state = 0x5DEE_CE66_D1CE_4E5Bu64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut params = super::super::network::NNUEParams::zeroed_box();
        for s in 0..super::super::STACKS {
            for g in params.l1_weights.0[s].iter_mut() {
                g.iter_mut().for_each(|w| *w = next() as i8);
            }
            for g in params.l2_weights.0[s].iter_mut() {
                g.iter_mut().for_each(|w| *w = next() as i8);
            }
            params.l1_biases.0[s].iter_mut().for_each(|b| *b = (next() % 60_001) as i32 - 30_000);
            params.l2_biases.0[s].iter_mut().for_each(|b| *b = (next() % 20_001) as i32 - 10_000);
            params.l3_weights_a.0[s].iter_mut().for_each(|w| *w = next() as i8);
            params.l3_weights_h.0[s].iter_mut().for_each(|w| *w = (next() % 255) as i32 - 127);
            params.l3_biases.0[s] = (next() % 20_001) as i32 - 10_000;
        }
        for _ in 0..50 {
            let mut ft_out = Aligned([0u8; L1_SIZE]);
            for x in ft_out.0.iter_mut() {
                let v = (next() >> 57) as u8;
                *x = if v < 70 { 0 } else { v };
            }
            let mut nnz = [0u16; NNZ_SIZE];
            let count = unsafe { find_nnz(&ft_out, nnz.as_mut_ptr()) };
            for stack in 0..super::super::STACKS {
                let got = unsafe { k::dense_layers(&params, &ft_out, &nnz[..count], stack) };
                assert_eq!(got, dense_reference(&params, &ft_out, stack), "stack {stack}");
            }
        }
    }

    #[test]
    fn test_nnz_table_correctness() {
        for byte in 0..256u16 {
            let entry = &NNZ_TABLE[byte as usize];
            assert_eq!(
                entry.count,
                byte.count_ones() as usize,
                "NNZ_TABLE[{byte}] count"
            );
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
                assert_eq!(
                    combined.0[pov][i], 0,
                    "KK endgame should be 0, got {} at pov={pov} i={i}",
                    combined.0[pov][i]
                );
            }
        }
    }

    #[test]
    fn test_compute_full_threats_startpos() {
        let pos = crate::position::Position::from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        )
        .unwrap();
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
