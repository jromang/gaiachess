// Hand-vectorised code: the loop index is the point. It walks several arrays in
// lockstep, steps by a SIMD lane count, and feeds raw pointer arithmetic — an
// iterator would hide the arithmetic these kernels exist to control.
#![allow(clippy::needless_range_loop)]
#![allow(unsafe_op_in_unsafe_fn)]
//! Monomorphized NNUE kernels — one complete instance per SIMD backend.
//!
//! Every hot NNUE loop lives in the `nnue_kernels!` macro below and is stamped
//! out once per backend (`use <backend> as simd;`), because the backends do not
//! share types: a kernel that walks `[_; L2_SIZE / simd::F32_LANES]` needs its
//! array length — and therefore its whole body — monomorphized per register
//! width. Function-level granularity is what a runtime dispatch table can point
//! into; primitive-level granularity could not be (the signatures differ).
//!
//! Two kernels are *not* macro-generated:
//! - `find_nnz` (in `forward.rs`): three genuinely different algorithms per ISA.
//! - `threat512::threat_batch`: a hand-scheduled tiled AVX-512 form with no
//!   narrower equivalent; the macro instances carry the generic row-loop form.
//!
//! Which instance runs is selected at compile time (`pub use ... as k`),
//! mirroring the historical `#[cfg(target_feature)]` cascade.

/// Number of SIMD registers held in flight by the Finny batch apply.
/// Must divide L1_SIZE / I16_LANES for all SIMD widths — asserted per instance.
pub(crate) const FINNY_REGISTERS: usize = 4;

macro_rules! nnue_kernels {
    ($backend:path $(, #[$tf:meta])?) => {
        use $backend as simd;
        use crate::nnue::accumulator::Accumulator;
        use crate::nnue::network::{self, Aligned};
        use crate::nnue::{FT_QUANT, FT_SHIFT, L1_NORMALISATION, L1_SIZE, L2_SIZE, L3_SIZE};
        use crate::types::Color;

        const _: () = assert!(
            L1_SIZE.is_multiple_of(crate::nnue::kernels::FINNY_REGISTERS * simd::I16_LANES)
        );

        // ============================================================
        // FT activation: within-side pairwise product
        // ============================================================

        /// Within-side pairwise activation.
        ///
        /// For each perspective (STM and NSTM), combines PST accumulator with
        /// threat buffer, then computes the pairwise product of
        /// `combined[0..L1/2]` × `combined[L1/2..L1]`.
        ///
        /// Clamping is asymmetric:
        /// - First half (left): `clamp(x, 0, FT_QUANT)` — standard CReLU
        /// - Second half (right): `min(x, FT_QUANT)` — no lower-bound clamp
        ///
        /// Product: `u8( mulhi(left << 7, right) )` via unsigned packus saturation.
        /// Output: `[stm_pairwise[L1/2] | ntm_pairwise[L1/2]]` = u8[L1].
        ///
        /// The two forms below are selected by a const-folded lane-count branch:
        /// the packus path needs real vector registers, the element loop does not.
        $(#[$tf])?
        #[inline]
        pub unsafe fn activate_ft(
            pst_acc: &Accumulator,
            combined: &Aligned<[[i16; L1_SIZE]; 2]>,
            stm: Color,
        ) -> Aligned<[u8; L1_SIZE]> {
            let mut output = Aligned([0u8; L1_SIZE]);
            let half = L1_SIZE / 2;

            if simd::I16_LANES == 1 {
                for flip in 0..2usize {
                    let pov = stm.index() ^ flip;
                    let pst_vals = &pst_acc.values.0[pov];
                    let comb_vals = &combined.0[pov];

                    for i in 0..half {
                        let left = (pst_vals[i] + comb_vals[i]).clamp(0, FT_QUANT);
                        // Right: asymmetric — only upper clamp (matches SIMD packus saturation)
                        let right = (pst_vals[i + half] + comb_vals[i + half]).min(FT_QUANT);
                        let product =
                            ((left as i32 * right as i32) >> FT_SHIFT).clamp(0, 255) as u8;
                        output.0[i + flip * half] = product;
                    }
                }
                return output;
            }

            let zero = simd::splat_i16(0);
            let one = simd::splat_i16(FT_QUANT);

            for flip in 0..2usize {
                let pov = stm.index() ^ flip;
                let pst_input = pst_acc.values.0[pov].as_ptr();
                let comb_input = combined.0[pov].as_ptr();

                for i in (0..half).step_by(2 * simd::I16_LANES) {
                    // Load PST + threats for left half (two SIMD-width chunks)
                    let pst_l1 = *pst_input.add(i).cast();
                    let pst_l2 = *pst_input.add(i + simd::I16_LANES).cast();
                    let comb_l1 = *comb_input.add(i).cast();
                    let comb_l2 = *comb_input.add(i + simd::I16_LANES).cast();
                    let lhs1 = simd::add_i16(pst_l1, comb_l1);
                    let lhs2 = simd::add_i16(pst_l2, comb_l2);

                    // Load PST + threats for right half
                    let pst_r1 = *pst_input.add(i + half).cast();
                    let pst_r2 = *pst_input.add(i + half + simd::I16_LANES).cast();
                    let comb_r1 = *comb_input.add(i + half).cast();
                    let comb_r2 = *comb_input.add(i + half + simd::I16_LANES).cast();
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

            output
        }

        // ============================================================
        // L1: sparse matmul → CReLU+squared
        // ============================================================

        /// L1 sparse inference: process only non-zero input groups.
        ///
        /// Sparse u8[L1] × i8 weights → i32[L2_SIZE], dequantize, add bias.
        /// Activation: CReLU+squared = `[clamp(x, 0, 1) ; clamp(x, 0, 1)²]` → f32[2×L2_SIZE].
        ///
        /// The two halves feed L2 (full 32 values) and provide the L3 skip connection.
        $(#[$tf])?
        #[inline]
        pub unsafe fn propagate_l1(
            ft_out: &Aligned<[u8; L1_SIZE]>,
            nnz: &[u16],
            bucket: usize,
        ) -> Aligned<[f32; 2 * L2_SIZE]> {
            let params = network::params();

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
            if !rem.is_empty() {
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
                let biases = *params.l1_biases.0[bucket]
                    .as_ptr()
                    .add(j * simd::F32_LANES)
                    .cast();
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

        // ============================================================
        // L2: dense f32 FMA + squared activation
        // ============================================================

        /// L2 dense layer: f32 matmul + bias → squared activation (SIMD splat+FMA).
        ///
        /// Input: f32[2×L2_SIZE] (the full CReLU+squared L1 output).
        /// Weight layout: `[bucket][input=2×L2_SIZE][output=L3_SIZE]`.
        /// Activation: `clamp(x, 0, 1)²` (identical to SCReLU).
        ///
        /// Same const-folded branch as `activate_ft`: the splat+FMA form is
        /// pointless one element at a time, so one lane means the plain loop.
        $(#[$tf])?
        #[inline]
        pub unsafe fn propagate_l2(
            l1_out: &Aligned<[f32; 2 * L2_SIZE]>,
            bucket: usize,
        ) -> Aligned<[f32; L3_SIZE]> {
            let params = network::params();
            let mut output = Aligned([0.0f32; L3_SIZE]);

            if simd::F32_LANES == 1 {
                for j in 0..L3_SIZE {
                    let mut sum = params.l2_biases.0[bucket][j];
                    for i in 0..l1_out.0.len() {
                        sum += params.l2_weights.0[bucket][i][j] * l1_out.0[i];
                    }
                    // L2 activation: SCReLU = clamp(x, 0, 1)²
                    let clamped = sum.clamp(0.0, 1.0);
                    output.0[j] = clamped * clamped;
                }
                return output;
            }

            // Init accumulators with biases
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

        // ============================================================
        // L3: output with L1 skip connection
        // ============================================================

        /// L3 output layer: `dot(concat(l2_out, l1_out), l3_weights) + bias` → scalar.
        ///
        /// Skip connection: L1 output (f32[2×L2_SIZE]) is concatenated with L2 output
        /// (f32[L3_SIZE]). Total input: `L3_SIZE + 2×L2_SIZE = 64` floats.
        /// Weights: `l3_weights[bucket][64]`.
        $(#[$tf])?
        #[inline]
        pub unsafe fn propagate_l3(
            l2_out: &Aligned<[f32; L3_SIZE]>,
            l1_out: &Aligned<[f32; 2 * L2_SIZE]>,
            bucket: usize,
        ) -> f32 {
            let params = network::params();

            let l2_ptr = l2_out.0.as_ptr();
            let l1_ptr = l1_out.0.as_ptr();
            let weights = params.l3_weights.0[bucket].as_ptr();

            let mut acc = [simd::zero_f32(); simd::HSUM_VECS];

            // First half: l2_out[0..L3_SIZE] × weights[0..L3_SIZE]
            for (lane, result) in acc.iter_mut().enumerate() {
                for i in (0..L3_SIZE).step_by(simd::HSUM_VECS * simd::F32_LANES) {
                    let off = i + lane * simd::F32_LANES;
                    *result = simd::mul_add_f32(
                        *weights.add(off).cast(),
                        *l2_ptr.add(off).cast(),
                        *result,
                    );
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

        // ============================================================
        // Dense forward pass — the per-eval composite
        // ============================================================

        /// Forward pass through the dense layers: FT activation → NNZ extraction
        /// → sparse L1 → dense L2 → L3 with skip connection.
        ///
        /// One composite entry point so the stages inline into each other; only
        /// `find_nnz` sits behind its own seam (its algorithm is chosen on an
        /// axis independent from the register width).
        $(#[$tf])?
        #[inline]
        pub unsafe fn forward_dense(
            pst_acc: &Accumulator,
            combined: &Aligned<[[i16; L1_SIZE]; 2]>,
            stm: Color,
            bucket: usize,
        ) -> f32 {
            let ft_out = activate_ft(pst_acc, combined, stm);
            // The nnz algorithm rides an axis of its own (VBMI2, not the register
            // width); when the build cannot pin it statically, one predicted
            // indirect call fetches it from the resolved table.
            #[cfg(any(
                not(target_arch = "x86_64"),
                all(target_feature = "avx512bw", target_feature = "avx512vbmi2")
            ))]
            let (nnz_indices, nnz_count) = crate::nnue::forward::find_nnz(&ft_out);
            #[cfg(all(
                target_arch = "x86_64",
                not(all(target_feature = "avx512bw", target_feature = "avx512vbmi2"))
            ))]
            let (nnz_indices, nnz_count) = (crate::cpu::get().find_nnz)(&ft_out);
            let l1_out = propagate_l1(&ft_out, &nnz_indices[..nnz_count], bucket);
            let l2_out = propagate_l2(&l1_out, bucket);
            propagate_l3(&l2_out, &l1_out, bucket)
        }

        // ============================================================
        // PST accumulator kernels — step_by(I16_LANES) with pointer casts
        // ============================================================

        /// Normal move: one feature added, one removed.
        $(#[$tf])?
        #[inline]
        pub unsafe fn acc_add1_sub1(prev: *const i16, out: *mut i16, add1: usize, sub1: usize) {
            let vadd = network::params().ft_pst_weights.0[add1].as_ptr();
            let vsub = network::params().ft_pst_weights.0[sub1].as_ptr();
            for i in (0..L1_SIZE).step_by(simd::I16_LANES) {
                let mut v = *prev.add(i).cast();
                v = simd::add_i16(v, *vadd.add(i).cast());
                v = simd::sub_i16(v, *vsub.add(i).cast());
                *out.add(i).cast() = v;
            }
        }

        /// Capture / EP: one feature added, two removed.
        $(#[$tf])?
        #[inline]
        pub unsafe fn acc_add1_sub2(
            prev: *const i16,
            out: *mut i16,
            add1: usize,
            sub1: usize,
            sub2: usize,
        ) {
            let vadd = network::params().ft_pst_weights.0[add1].as_ptr();
            let vsub1 = network::params().ft_pst_weights.0[sub1].as_ptr();
            let vsub2 = network::params().ft_pst_weights.0[sub2].as_ptr();
            for i in (0..L1_SIZE).step_by(simd::I16_LANES) {
                let mut v = *prev.add(i).cast();
                v = simd::add_i16(v, *vadd.add(i).cast());
                v = simd::sub_i16(v, *vsub1.add(i).cast());
                v = simd::sub_i16(v, *vsub2.add(i).cast());
                *out.add(i).cast() = v;
            }
        }

        /// Castling: two features added, two removed.
        $(#[$tf])?
        #[inline]
        pub unsafe fn acc_add2_sub2(
            prev: *const i16,
            out: *mut i16,
            add1: usize,
            add2: usize,
            sub1: usize,
            sub2: usize,
        ) {
            let vadd1 = network::params().ft_pst_weights.0[add1].as_ptr();
            let vadd2 = network::params().ft_pst_weights.0[add2].as_ptr();
            let vsub1 = network::params().ft_pst_weights.0[sub1].as_ptr();
            let vsub2 = network::params().ft_pst_weights.0[sub2].as_ptr();
            for i in (0..L1_SIZE).step_by(simd::I16_LANES) {
                let mut v = *prev.add(i).cast();
                v = simd::add_i16(v, *vadd1.add(i).cast());
                v = simd::add_i16(v, *vadd2.add(i).cast());
                v = simd::sub_i16(v, *vsub1.add(i).cast());
                v = simd::sub_i16(v, *vsub2.add(i).cast());
                *out.add(i).cast() = v;
            }
        }

        // ============================================================
        // Finny table apply — register-blocked batched add/sub
        // ============================================================

        /// Apply batched feature additions and subtractions to a Finny cache entry.
        ///
        /// Uses register blocking: loads FINNY_REGISTERS SIMD vectors, applies all
        /// adds/subs, then stores back. This minimizes memory round-trips.
        $(#[$tf])?
        #[inline]
        pub unsafe fn finny_apply(values: *mut i16, adds: &[usize], subs: &[usize]) {
            const REGS: usize = crate::nnue::kernels::FINNY_REGISTERS;
            let mut regs: [_; REGS] = std::mem::zeroed();

            for offset in (0..L1_SIZE).step_by(REGS * simd::I16_LANES) {
                let out = values.add(offset);

                // Load current values into registers
                for (i, r) in regs.iter_mut().enumerate() {
                    *r = *out.add(i * simd::I16_LANES).cast();
                }

                // Apply all additions
                for &idx in adds {
                    let w = network::params().ft_pst_weights.0[idx].as_ptr().add(offset);
                    for (i, r) in regs.iter_mut().enumerate() {
                        *r = simd::add_i16(*r, *w.add(i * simd::I16_LANES).cast());
                    }
                }

                // Apply all subtractions
                for &idx in subs {
                    let w = network::params().ft_pst_weights.0[idx].as_ptr().add(offset);
                    for (i, r) in regs.iter_mut().enumerate() {
                        *r = simd::sub_i16(*r, *w.add(i * simd::I16_LANES).cast());
                    }
                }

                // Store back
                for (i, r) in regs.into_iter().enumerate() {
                    *out.add(i * simd::I16_LANES).cast() = r;
                }
            }
        }

        // ============================================================
        // Threat accumulator batch apply — generic row-loop form
        // ============================================================

        /// Add an i8 weight row (L1_SIZE elements) into an i16 accumulator.
        #[inline(always)]
        unsafe fn accumulate_i8_row(weights: *const i8, acc: *mut i16) {
            for j in (0..L1_SIZE).step_by(simd::I16_LANES) {
                let w = simd::load_i8_as_i16(weights.add(j));
                let slot = acc.add(j).cast();
                *slot = simd::add_i16(*slot, w);
            }
        }

        /// Subtract an i8 weight row (L1_SIZE elements) from an i16 accumulator.
        #[inline(always)]
        unsafe fn subtract_i8_row(weights: *const i8, acc: *mut i16) {
            for j in (0..L1_SIZE).step_by(simd::I16_LANES) {
                let w = simd::load_i8_as_i16(weights.add(j));
                let slot = acc.add(j).cast();
                *slot = simd::sub_i16(*slot, w);
            }
        }

        /// Extract indices of non-zero 4-byte groups from the FT output:
        /// NNZ_TABLE walk over this backend's `nnz_bitmask` width.
        ///
        /// This is the portable form (scalar, NEON, wasm128); x86 vector tiers
        /// use the specialized forms in `forward.rs` (SSE stores / VPCOMPRESSW).
        $(#[$tf])?
        #[inline]
        pub unsafe fn find_nnz_generic(
            ft_out: &Aligned<[u8; L1_SIZE]>,
        ) -> ([u16; crate::nnue::forward::NNZ_SIZE], usize) {
            use crate::nnue::forward::{NNZ_SIZE, NNZ_TABLE};

            let mut indexes = [0u16; NNZ_SIZE];
            let mut count = 0;

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

            (indexes, count)
        }

        /// Batched threat-row accumulation: per-feature row loops with prefetch.
        ///
        /// `input_acc`: source accumulator (null = zero-init for full recompute).
        /// `output_acc`: destination accumulator.
        ///
        /// The AVX-512 tiers use `threat512::threat_batch` instead (tiled 16-ZMM).
        $(#[$tf])?
        #[inline]
        pub unsafe fn threat_batch(
            input_acc: *const i16,
            output_acc: *mut i16,
            weights_base: *const [i8; L1_SIZE],
            adds: &[u32],
            subs: &[u32],
        ) {
            if !input_acc.is_null() {
                std::ptr::copy_nonoverlapping(input_acc, output_acc, L1_SIZE);
            } else {
                std::ptr::write_bytes(output_acc, 0, L1_SIZE);
            }
            // No software prefetch, as in `threat512::threat_batch`.
            for &s in subs {
                subtract_i8_row((*weights_base.add(s as usize)).as_ptr(), output_acc);
            }
            for &a in adds {
                accumulate_i8_row((*weights_base.add(a as usize)).as_ptr(), output_acc);
            }
        }
    };
}

// ============================================================
// Hand-scheduled AVX-512 threat batch (no narrower equivalent)
// ============================================================

/// Register-batched threat accumulation: loads the whole accumulator into ZMM
/// registers, applies all add/sub weight rows on registers, stores once.
/// Eliminates N-1 intermediate load/store pairs of the accumulator.
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)] // selected per build target (and, later, per detected CPU)
pub mod threat512 {
    use crate::nnue::simd::avx512 as simd;
    use crate::nnue::L1_SIZE;

    /// `input_acc`: source accumulator (null = zero-init for full recompute).
    /// `output_acc`: destination accumulator.
    #[cfg_attr(
        not(all(target_feature = "avx512f", target_feature = "avx512bw")),
        target_feature(enable = "avx512f,avx512bw")
    )]
    pub unsafe fn threat_batch(
        input_acc: *const i16,
        output_acc: *mut i16,
        weights_base: *const [i8; L1_SIZE],
        adds: &[u32],
        subs: &[u32],
    ) {
        use std::arch::x86_64::*;
        // 16 ZMMs = 512 i16 per tile. L1=1024 → 2 tiles. Keeps headroom in the
        // 32-register file (the 20-register whole-acc form does not fit 1024).
        const TILE: usize = 16;
        const TILE_ELEMS: usize = TILE * 32;
        const _: () = assert!(L1_SIZE % TILE_ELEMS == 0);

        // No software prefetch: the rows a search touches stay resident in the
        // last-level cache, the hardware prefetcher streams each row once its first
        // line is asked for, and the look-ahead branches that used to sit in these
        // loops mispredicted at every transition — measured slower than nothing.
        for tile in 0..(L1_SIZE / TILE_ELEMS) {
            let base = tile * TILE_ELEMS;
            let mut regs: [__m512i; TILE] = [_mm512_setzero_si512(); TILE];
            if !input_acc.is_null() {
                for i in 0..TILE {
                    regs[i] = _mm512_loadu_si512(input_acc.add(base + i * 32) as *const __m512i);
                }
            }

            for &s in subs {
                let w_ptr = (*weights_base.add(s as usize)).as_ptr().add(base);
                for j in 0..TILE {
                    let w = simd::load_i8_as_i16(w_ptr.add(j * 32));
                    regs[j] = simd::sub_i16(regs[j], w);
                }
            }
            for &a in adds {
                let w_ptr = (*weights_base.add(a as usize)).as_ptr().add(base);
                for j in 0..TILE {
                    let w = simd::load_i8_as_i16(w_ptr.add(j * 32));
                    regs[j] = simd::add_i16(regs[j], w);
                }
            }

            for i in 0..TILE {
                _mm512_storeu_si512(output_acc.add(base + i * 32) as *mut __m512i, regs[i]);
            }
        }
    }
}

// ============================================================
// Instances — every backend is compiled on x86-64; each kernel carries a
// `#[target_feature]` shim UNLESS the build target already guarantees that
// backend's features, in which case the attribute is dropped and the kernels
// inline exactly as a plain per-target build would.
// (Runtime priority mirrors the historical cascade: AVX-512+VNNI > AVX-512 >
//  AVX2 > NEON > wasm simd128 > scalar.)
// ============================================================

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)] // selected per build target (and, later, per detected CPU)
pub mod k_vnni512 {
    nnue_kernels!(
        crate::nnue::simd::avx512vnni,
        #[cfg_attr(
            not(all(
                target_feature = "avx2",
                target_feature = "fma",
                target_feature = "avx512f",
                target_feature = "avx512bw",
                target_feature = "avx512vnni"
            )),
            target_feature(enable = "avx2,fma,avx512f,avx512bw,avx512vnni")
        )]
    );
}

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
pub mod k_avx512 {
    nnue_kernels!(
        crate::nnue::simd::avx512,
        #[cfg_attr(
            not(all(
                target_feature = "avx2",
                target_feature = "fma",
                target_feature = "avx512f",
                target_feature = "avx512bw"
            )),
            target_feature(enable = "avx2,fma,avx512f,avx512bw")
        )]
    );
}

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
pub mod k_avx2 {
    nnue_kernels!(
        crate::nnue::simd::avx2,
        #[cfg_attr(
            not(all(target_feature = "avx2", target_feature = "fma")),
            target_feature(enable = "avx2,fma")
        )]
    );
}

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
pub mod k_scalar {
    nnue_kernels!(crate::nnue::simd::scalar);
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub mod k_native {
    nnue_kernels!(crate::nnue::simd::neon);
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub mod k_native {
    nnue_kernels!(crate::nnue::simd::wasm128);
}

#[cfg(not(any(
    target_arch = "x86_64",
    all(target_arch = "aarch64", target_feature = "neon"),
    all(target_arch = "wasm32", target_feature = "simd128"),
)))]
pub mod k_native {
    nnue_kernels!(crate::nnue::simd::scalar);
}

// ============================================================
// Current instance — the compile-time dispatch decision
// ============================================================

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vnni"
))]
pub(crate) use k_vnni512 as k;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    not(target_feature = "avx512vnni")
))]
pub(crate) use k_avx512 as k;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
pub(crate) use k_avx2 as k;
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
pub(crate) use k_scalar as k;
#[cfg(not(target_arch = "x86_64"))]
pub(crate) use k_native as k;

/// Threat batch apply for the current build target: the hand-scheduled 20-ZMM
/// form when AVX-512 is available, the generic row-loop form otherwise.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub(crate) use threat512::threat_batch;
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
pub(crate) use k::threat_batch;

/// True when the build target already guarantees the whole top-tier feature
/// set, so the resolver could never pick anything else: the dispatch helpers
/// below then bypass the table and the kernels inline exactly as a plain
/// per-target build would (`target-cpu=native` on a Zen 4/5 box, `znver4`...).
#[cfg(target_arch = "x86_64")]
pub(crate) const STATIC_TOP: bool = cfg!(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vnni",
));

// ============================================================
// Dispatch helpers — the one place that decides between the static
// instance and the runtime table. Every hot caller goes through here.
// ============================================================

pub(crate) mod dispatch {
    use crate::nnue::accumulator::Accumulator;
    use crate::nnue::network::Aligned;
    use crate::nnue::L1_SIZE;
    use crate::types::Color;

    /// Forward pass through the dense layers, on the resolved tier.
    #[inline(always)]
    pub unsafe fn forward_dense(
        pst_acc: &Accumulator,
        combined: &Aligned<[[i16; L1_SIZE]; 2]>,
        stm: Color,
        bucket: usize,
    ) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            if super::STATIC_TOP {
                super::k::forward_dense(pst_acc, combined, stm, bucket)
            } else {
                (crate::cpu::get().forward_dense)(pst_acc, combined, stm, bucket)
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            super::k::forward_dense(pst_acc, combined, stm, bucket)
        }
    }

    /// Normal move: one feature added, one removed.
    #[inline(always)]
    pub unsafe fn acc_add1_sub1(prev: *const i16, out: *mut i16, add1: usize, sub1: usize) {
        #[cfg(target_arch = "x86_64")]
        {
            if super::STATIC_TOP {
                super::k::acc_add1_sub1(prev, out, add1, sub1)
            } else {
                (crate::cpu::get().acc_add1_sub1)(prev, out, add1, sub1)
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            super::k::acc_add1_sub1(prev, out, add1, sub1)
        }
    }

    /// Capture / EP: one feature added, two removed.
    #[inline(always)]
    pub unsafe fn acc_add1_sub2(
        prev: *const i16,
        out: *mut i16,
        add1: usize,
        sub1: usize,
        sub2: usize,
    ) {
        #[cfg(target_arch = "x86_64")]
        {
            if super::STATIC_TOP {
                super::k::acc_add1_sub2(prev, out, add1, sub1, sub2)
            } else {
                (crate::cpu::get().acc_add1_sub2)(prev, out, add1, sub1, sub2)
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            super::k::acc_add1_sub2(prev, out, add1, sub1, sub2)
        }
    }

    /// Castling: two features added, two removed.
    #[inline(always)]
    pub unsafe fn acc_add2_sub2(
        prev: *const i16,
        out: *mut i16,
        add1: usize,
        add2: usize,
        sub1: usize,
        sub2: usize,
    ) {
        #[cfg(target_arch = "x86_64")]
        {
            if super::STATIC_TOP {
                super::k::acc_add2_sub2(prev, out, add1, add2, sub1, sub2)
            } else {
                (crate::cpu::get().acc_add2_sub2)(prev, out, add1, add2, sub1, sub2)
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            super::k::acc_add2_sub2(prev, out, add1, add2, sub1, sub2)
        }
    }

    /// Finny-table register-blocked batch apply.
    #[inline(always)]
    pub unsafe fn finny_apply(values: *mut i16, adds: &[usize], subs: &[usize]) {
        #[cfg(target_arch = "x86_64")]
        {
            if super::STATIC_TOP {
                super::k::finny_apply(values, adds, subs)
            } else {
                (crate::cpu::get().finny_apply)(values, adds, subs)
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            super::k::finny_apply(values, adds, subs)
        }
    }

    /// Batched threat-row accumulation.
    #[inline(always)]
    pub unsafe fn threat_batch(
        input_acc: *const i16,
        output_acc: *mut i16,
        weights_base: *const [i8; L1_SIZE],
        adds: &[u32],
        subs: &[u32],
    ) {
        #[cfg(target_arch = "x86_64")]
        {
            if super::STATIC_TOP {
                super::threat_batch(input_acc, output_acc, weights_base, adds, subs)
            } else {
                (crate::cpu::get().threat_batch)(input_acc, output_acc, weights_base, adds, subs)
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            super::threat_batch(input_acc, output_acc, weights_base, adds, subs)
        }
    }
}

// ============================================================
// Cross-backend equivalence tests. One binary carries every instance, so the
// vector tiers can finally be checked against the scalar reference on the
// machine running the tests — each behind its own CPUID guard.
// ============================================================

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use crate::nnue::accumulator::Accumulator;
    use crate::nnue::network::{Aligned, PermKind};
    use crate::nnue::{FT_QUANT, FT_SHIFT, L1_SIZE};
    use crate::types::Color;

    /// Lay linear values out the way a tier's `packus` expects them (the layout
    /// `repermute_ft` gives the weights, hence the accumulators built from them).
    fn linear_to(kind: PermKind, row: &mut [i16]) {
        let perm: &[usize] = match kind {
            PermKind::File512 => &[0, 2, 4, 6, 1, 3, 5, 7],
            PermKind::Avx2 => &[0, 2, 1, 3],
            PermKind::Linear => return,
        };
        let group = 8 * perm.len();
        debug_assert_eq!(L1_SIZE % group, 0);
        let mut tmp = vec![0i16; group];
        for chunk in row.chunks_exact_mut(group) {
            for (j, &p) in perm.iter().enumerate() {
                tmp[j * 8..(j + 1) * 8].copy_from_slice(&chunk[p * 8..(p + 1) * 8]);
            }
            chunk.copy_from_slice(&tmp);
        }
    }

    /// Deterministic inputs that cross every branch: negatives (clamped away on
    /// the left, kept on the right), values above `FT_QUANT`, products that
    /// saturate the unsigned pack. Kept well inside i16 so the addition in the
    /// reference cannot overflow.
    fn test_inputs() -> (Accumulator, Aligned<[[i16; L1_SIZE]; 2]>) {
        let mut pst_acc = Accumulator::new();
        let mut combined = Aligned([[0i16; L1_SIZE]; 2]);
        for pov in 0..2 {
            for i in 0..L1_SIZE {
                pst_acc.values.0[pov][i] = (((i * 37 + pov * 11) % 811) as i32 - 400) as i16;
                combined.0[pov][i] = (((i * 53 + pov * 7) % 509) as i32 - 250) as i16;
            }
        }
        (pst_acc, combined)
    }

    fn inputs_permuted_for(kind: PermKind) -> (Accumulator, Aligned<[[i16; L1_SIZE]; 2]>) {
        let (mut pst_acc, mut combined) = test_inputs();
        for pov in 0..2 {
            linear_to(kind, &mut pst_acc.values.0[pov]);
            linear_to(kind, &mut combined.0[pov]);
        }
        (pst_acc, combined)
    }

    /// The activation, spelled out one element at a time on the linear inputs.
    fn activate_reference(
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

    fn assert_activation_matches(tier: &str, got: &Aligned<[u8; L1_SIZE]>, want: &[u8; L1_SIZE]) {
        if let Some(i) = (0..L1_SIZE).find(|&i| got.0[i] != want[i]) {
            panic!(
                "{tier} activation disagrees with the scalar reference at {i}: \
                 got {}, want {}",
                got.0[i], want[i]
            );
        }
    }

    #[test]
    fn every_tier_activation_lands_on_the_same_linear_output() {
        let (lin_pst, lin_comb) = test_inputs();
        for stm in [Color::White, Color::Black] {
            let want = activate_reference(&lin_pst, &lin_comb, stm);

            let got = unsafe { k_scalar::activate_ft(&lin_pst, &lin_comb, stm) };
            assert_activation_matches("scalar", &got, &want);

            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                let (pst, comb) = inputs_permuted_for(PermKind::Avx2);
                let got = unsafe { k_avx2::activate_ft(&pst, &comb, stm) };
                assert_activation_matches("avx2", &got, &want);
            }

            if std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512bw")
            {
                let (pst, comb) = inputs_permuted_for(PermKind::File512);
                let got = unsafe { k_avx512::activate_ft(&pst, &comb, stm) };
                assert_activation_matches("avx512", &got, &want);

                if std::arch::is_x86_feature_detected!("avx512vnni") {
                    let got = unsafe { k_vnni512::activate_ft(&pst, &comb, stm) };
                    assert_activation_matches("vnni512", &got, &want);
                }
            }
        }
    }

    #[test]
    fn the_three_nnz_forms_agree() {
        // A mix of empty, sparse and dense 4-byte groups.
        let mut ft_out = Aligned([0u8; L1_SIZE]);
        let mut state = 0x9E3779B97F4A7C15u64;
        for i in 0..L1_SIZE {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ft_out.0[i] = if state % 5 == 0 { (state >> 32) as u8 } else { 0 };
        }

        let (want_idx, want_n) = unsafe { k_scalar::find_nnz_generic(&ft_out) };

        if std::arch::is_x86_feature_detected!("avx2") {
            let (idx, n) = unsafe { crate::nnue::forward::find_nnz_avx2(&ft_out) };
            assert_eq!(n, want_n, "avx2 nnz count");
            assert_eq!(idx[..n], want_idx[..want_n], "avx2 nnz indices");
        }
        if std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vbmi2")
        {
            let (idx, n) = unsafe { crate::nnue::forward::find_nnz_compress512(&ft_out) };
            assert_eq!(n, want_n, "compress512 nnz count");
            assert_eq!(idx[..n], want_idx[..want_n], "compress512 nnz indices");
        }
    }

    #[test]
    fn threat_batch_forms_agree_on_synthetic_weights() {
        // Synthetic weight rows: no dependence on a loaded network.
        const ROWS: usize = 8;
        let mut weights = vec![[0i8; L1_SIZE]; ROWS];
        for (r, row) in weights.iter_mut().enumerate() {
            for (i, w) in row.iter_mut().enumerate() {
                *w = (((r * 251 + i * 31) % 199) as i32 - 99) as i8;
            }
        }
        let mut input = Aligned([0i16; L1_SIZE]);
        for i in 0..L1_SIZE {
            input.0[i] = ((i as i32 * 97) % 1601 - 800) as i16;
        }
        let adds: [u32; 3] = [0, 3, 6];
        let subs: [u32; 2] = [1, 5];

        let run = |f: unsafe fn(*const i16, *mut i16, *const [i8; L1_SIZE], &[u32], &[u32]),
                   from_null: bool| {
            let mut out = Aligned([0i16; L1_SIZE]);
            let src = if from_null { std::ptr::null() } else { input.0.as_ptr() };
            unsafe { f(src, out.0.as_mut_ptr(), weights.as_ptr(), &adds, &subs) };
            out
        };

        for from_null in [false, true] {
            let want = run(k_scalar::threat_batch, from_null);
            if std::arch::is_x86_feature_detected!("avx2") {
                let got = run(k_avx2::threat_batch, from_null);
                assert_eq!(got.0, want.0, "avx2 threat batch (from_null={from_null})");
            }
            if std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512bw")
            {
                let got = run(threat512::threat_batch, from_null);
                assert_eq!(got.0, want.0, "threat512 batch (from_null={from_null})");
            }
        }
    }
}
