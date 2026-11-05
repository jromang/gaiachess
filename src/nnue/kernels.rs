// Hand-vectorised code: the loop index is the point. It walks several arrays in
// lockstep, steps by a SIMD lane count, and feeds raw pointer arithmetic — an
// iterator would hide the arithmetic these kernels exist to control.
#![allow(clippy::needless_range_loop)]
#![allow(unsafe_op_in_unsafe_fn)]
//! Monomorphized NNUE kernels — one complete instance per SIMD backend.
//!
//! Every hot NNUE loop lives in the `nnue_kernels!` macro below and is stamped
//! out once per backend (`use <backend> as simd;`), because the backends do not
//! share types: a kernel that walks `[_; L1_OUT / simd::I32_LANES]` needs its
//! array length — and therefore its whole body — monomorphized per register
//! width. Function-level granularity is what a runtime dispatch table can point
//! into; primitive-level granularity could not be (the signatures differ).
//!
//! Two kernels are *not* macro-generated:
//! - `find_nnz` (in `forward.rs`): three genuinely different algorithms per ISA.
//! - `threat512::threat_batch`: a hand-scheduled 16-ZMM tile; the macro
//!   instances carry the same algorithm at `THREAT_REGISTERS` of the
//!   backend's width (8 YMM on AVX2, 16 Q registers on AArch64).
//!
//! Which instance runs is selected at compile time (`pub use ... as k`),
//! mirroring the historical `#[cfg(target_feature)]` cascade.

/// Number of SIMD registers held in flight by the generic threat-batch tile.
/// Must divide L1_SIZE / I16_LANES for all SIMD widths — asserted per instance —
/// and be even: rows are widened into register pairs. Eight matches a half-file of
/// AVX2 YMMs and leaves room for two weight loads and their widened pair; AVX-512
/// uses the dedicated 16-ZMM form instead. AArch64 takes 16 of its 32 vector
/// registers.
#[cfg(not(target_arch = "aarch64"))]
pub(crate) const THREAT_REGISTERS: usize = 8;
#[cfg(target_arch = "aarch64")]
pub(crate) const THREAT_REGISTERS: usize = 16;

/// Signature of `combined_batch` in the dispatch tables: accumulator in and out,
/// the piece-square rows added and removed, the threat and pawn-pair rows added and
/// removed.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) type CombinedBatchFn = unsafe fn(
    *const i16,
    *mut i16,
    *const crate::nnue::network::ThreatRow,
    &[u32],
    &[u32],
    *const crate::nnue::network::ThreatRow,
    &[u32],
    &[u32],
);

/// Expands `$body` once per listed index, with `$K` bound to it as a constant.
///
/// A register tile — an array of vectors held across a loop over weight rows —
/// stays in registers only if every access to it has a constant index. A `for`
/// over the tile gives constant indices only once the compiler has chosen to
/// unroll it completely, and that choice rests on a cost estimate that can be
/// far off: LLVM prices each 512-bit byte unpack of `widen_i8_pair` at 19 when
/// AVX512-VBMI is not enabled, for a single instruction, so without VBMI (the
/// runtime-dispatched kernels of the universal build, AVX-512 CPUs before Ice
/// Lake) it kept the pair loop rolled and the tile went to the stack — about
/// 12 % of the engine's speed. Spelled out, the indices are constants whatever
/// the estimate.
macro_rules! unroll {
    ($K:ident in [$($i:literal),+] $body:block) => {
        $({
            const $K: usize = $i;
            $body
        })+
    };
}

/// Applies i8 weight rows of one sign to a tile of i16 registers.
///
/// Rows go two at a time through `widen_i8_pair`, which widens and sums both in
/// one go (one shuffle per register for two rows on x86); an odd row out goes
/// through `widen_i8`. Each call fills a pair of registers, so the tile is
/// walked in register pairs, listed by `[$k]` (one entry per pair, see
/// [`unroll`]). The rows are in logical order on every tier: the x86 forms
/// interleave exactly as `packus` does, which lands logical bytes on the
/// accumulator's packus-ordered lanes.
///
/// The sums are wrapping i16 throughout, so pairing rows leaves every lane
/// equal to the row-at-a-time result.
macro_rules! apply_i8_rows {
    ($simd:ident, $regs:ident, $weights:expr, $base:expr, $rows:expr, $op:path,
     [$($k:literal),+]) => {{
        debug_assert_eq!(2 * [$($k),+].len(), $regs.len(), "one listed index per register pair");
        let mut pairs = $rows.chunks_exact(2);
        for pair in pairs.by_ref() {
            let a = (*$weights.add(pair[0] as usize)).as_ptr().add($base);
            let b = (*$weights.add(pair[1] as usize)).as_ptr().add($base);
            unroll!(K in [$($k),+] {
                let offset = 2 * K * $simd::I16_LANES;
                let [lo, hi] = $simd::widen_i8_pair(a.add(offset), b.add(offset));
                $regs[2 * K] = $op($regs[2 * K], lo);
                $regs[2 * K + 1] = $op($regs[2 * K + 1], hi);
            });
        }
        if let [row] = pairs.remainder() {
            let a = (*$weights.add(*row as usize)).as_ptr().add($base);
            unroll!(K in [$($k),+] {
                let [lo, hi] = $simd::widen_i8(a.add(2 * K * $simd::I16_LANES));
                $regs[2 * K] = $op($regs[2 * K], lo);
                $regs[2 * K + 1] = $op($regs[2 * K + 1], hi);
            });
        }
    }};
}

macro_rules! nnue_kernels {
    ($backend:path $(, #[$tf:meta])? $(, vbmi2: #[$tf2:meta])?) => {
        use $backend as simd;
        use crate::nnue::network::{self, Aligned};
        use crate::nnue::{
            A1_SIZE, FT_QUANT, FT_SHIFT, GATE_SHIFT, GATED, HIDDEN_MAX, L1_OUT, L1_SHIFT, L1_SIZE,
            L1_SQR_CLAMP, L1_SQR_SHIFT, L2_OUT, L2_SHIFT, SKIP_NEG, SKIP_POS,
        };
        use crate::types::Color;

        const _: () = assert!(L1_OUT.is_multiple_of(simd::I32_LANES));
        const _: () = assert!(GATED.is_multiple_of(simd::I32_LANES));
        const _: () = assert!(A1_SIZE.is_multiple_of(4 * simd::I32_LANES));
        const _: () = assert!(
            L1_SIZE.is_multiple_of(crate::nnue::kernels::THREAT_REGISTERS * simd::I16_LANES)
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
        #[cfg(test)]
        #[inline]
        pub unsafe fn activate_ft(
            acc: &Aligned<[[i16; L1_SIZE]; 2]>,
            stm: Color,
        ) -> Aligned<[u8; L1_SIZE]> {
            let mut output = Aligned([0u8; L1_SIZE]);
            activate_ft_into(acc, stm, output.0.as_mut_ptr());
            output
        }

        /// [`activate_ft`] into a caller-provided buffer of `L1_SIZE` bytes, aligned
        /// as `Aligned`, every byte written: the hot path hands it uninitialised
        /// memory instead of a zeroed 1 KB returned by value.
        $(#[$tf])?
        #[inline]
        pub unsafe fn activate_ft_into(
            acc: &Aligned<[[i16; L1_SIZE]; 2]>,
            stm: Color,
            output: *mut u8,
        ) {
            let half = L1_SIZE / 2;

            if simd::I16_LANES == 1 {
                for flip in 0..2usize {
                    let pov = stm.index() ^ flip;
                    let vals = &acc.0[pov];

                    for i in 0..half {
                        let left = vals[i].clamp(0, FT_QUANT);
                        // Right: asymmetric — only upper clamp (matches SIMD packus saturation)
                        let right = vals[i + half].min(FT_QUANT);
                        let product =
                            ((left as i32 * right as i32) >> FT_SHIFT).clamp(0, 255) as u8;
                        *output.add(i + flip * half) = product;
                    }
                }
                return;
            }

            // `ft_product` bakes the shift in.
            const _: () = assert!(FT_SHIFT == 9);
            let zero = simd::splat_i16(0);
            let one = simd::splat_i16(FT_QUANT);

            for flip in 0..2usize {
                let pov = stm.index() ^ flip;
                let input = acc.0[pov].as_ptr();

                for i in (0..half).step_by(2 * simd::I16_LANES) {
                    // Left half (two SIMD-width chunks)
                    let lhs1 = *input.add(i).cast();
                    let lhs2 = *input.add(i + simd::I16_LANES).cast();

                    // Right half
                    let rhs1 = *input.add(i + half).cast();
                    let rhs2 = *input.add(i + half + simd::I16_LANES).cast();

                    // Asymmetric clamp: left = [0, 255], right = (-∞, 255]
                    let lhs1_clipped = simd::clamp_i16(lhs1, zero, one);
                    let lhs2_clipped = simd::clamp_i16(lhs2, zero, one);
                    let rhs1_clipped = simd::min_i16(rhs1, one);
                    let rhs2_clipped = simd::min_i16(rhs2, one);

                    // Pairwise product (left * right) >> FT_SHIFT, by a high-half multiply
                    let product1 = simd::ft_product(lhs1_clipped, rhs1_clipped);
                    let product2 = simd::ft_product(lhs2_clipped, rhs2_clipped);

                    // Pack i16 → u8 with unsigned saturation (negative products → 0).
                    // PST accumulators are pre-permuted so packus output is linear.
                    let packed = simd::packus(product1, product2);
                    *output.add(i + flip * half).cast() = packed;
                }
            }
        }

        // ============================================================
        // L1: sparse matmul
        // ============================================================

        /// L1 sparse inference: process only non-zero input groups.
        ///
        /// Sparse u8[L1] × i8 weights (×128) → i32[L1_OUT], plus the bias: the raw
        /// outputs, unit 2^14, which the skip connection reads before any activation.
        $(#[$tf])?
        #[inline]
        pub unsafe fn propagate_l1(
            params: &network::NNUEParams,
            ft_out: &Aligned<[u8; L1_SIZE]>,
            nnz: &[u16],
            stack: usize,
        ) -> Aligned<[i32; L1_OUT]> {
            const N_VECS: usize = L1_OUT / simd::I32_LANES;
            let packed = ft_out.0.as_ptr() as *const i32;
            let mut pre_act = [simd::zeroed_i32(); N_VECS];

            // One chain on emulated dpbusd (AVX2 / AVX-512). Three on native
            // VNNI/`sdot`, whose latency needs independent accumulators: two output
            // registers each on AVX-512, six dpbusd in flight for its two ports.
            // i32 wrapping add is associative: the split does not change eval.
            if simd::DPBUSD_CHAINS == 1 {
                for &idx16 in nnz {
                    let idx = idx16 as usize;
                    let input = simd::splat_i32(*packed.add(idx));
                    let w = params.l1_weights.0[stack][idx].as_ptr();
                    for j in 0..N_VECS {
                        let off = j * simd::I32_LANES * 4;
                        pre_act[j] = simd::dpbusd(pre_act[j], input, *w.add(off).cast());
                    }
                }
            } else {
                let mut chain0 = [simd::zeroed_i32(); N_VECS];
                let mut chain1 = [simd::zeroed_i32(); N_VECS];
                let mut chain2 = [simd::zeroed_i32(); N_VECS];

                let mut triples = nnz.chunks_exact(3);
                for triple in &mut triples {
                    let idx0 = triple[0] as usize;
                    let idx1 = triple[1] as usize;
                    let idx2 = triple[2] as usize;

                    let in0 = simd::splat_i32(*packed.add(idx0));
                    let in1 = simd::splat_i32(*packed.add(idx1));
                    let in2 = simd::splat_i32(*packed.add(idx2));

                    let w0 = params.l1_weights.0[stack][idx0].as_ptr();
                    let w1 = params.l1_weights.0[stack][idx1].as_ptr();
                    let w2 = params.l1_weights.0[stack][idx2].as_ptr();

                    for j in 0..N_VECS {
                        let off = j * simd::I32_LANES * 4;
                        chain0[j] = simd::dpbusd(chain0[j], in0, *w0.add(off).cast());
                        chain1[j] = simd::dpbusd(chain1[j], in1, *w1.add(off).cast());
                        chain2[j] = simd::dpbusd(chain2[j], in2, *w2.add(off).cast());
                    }
                }

                let rem = triples.remainder();
                if !rem.is_empty() {
                    let idx = rem[0] as usize;
                    let input = simd::splat_i32(*packed.add(idx));
                    let w = params.l1_weights.0[stack][idx].as_ptr();
                    for j in 0..N_VECS {
                        let off = j * simd::I32_LANES * 4;
                        chain0[j] = simd::dpbusd(chain0[j], input, *w.add(off).cast());
                    }
                }
                if rem.len() >= 2 {
                    let idx = rem[1] as usize;
                    let input = simd::splat_i32(*packed.add(idx));
                    let w = params.l1_weights.0[stack][idx].as_ptr();
                    for j in 0..N_VECS {
                        let off = j * simd::I32_LANES * 4;
                        chain1[j] = simd::dpbusd(chain1[j], input, *w.add(off).cast());
                    }
                }

                for j in 0..N_VECS {
                    pre_act[j] = simd::add_i32(simd::add_i32(chain0[j], chain1[j]), chain2[j]);
                }
            }

            let mut out = core::mem::MaybeUninit::<Aligned<[i32; L1_OUT]>>::uninit();
            let dst = out.as_mut_ptr().cast::<i32>();
            let biases = params.l1_biases.0[stack].as_ptr();
            for j in 0..N_VECS {
                let at = j * simd::I32_LANES;
                *dst.add(at).cast() = simd::add_i32(pre_act[j], *biases.add(at).cast());
            }
            out.assume_init()
        }

        // ============================================================
        // L1 activation: [x², x] → u8
        // ============================================================

        /// The L1 activations: `min(127, x² >> 21)` for every output, then
        /// `clamp(x >> 7, 0, 127)` for every output, as u8 (unit 128).
        ///
        /// The square reads the raw output, negative or not; it is taken on `x`
        /// clamped to ±2^14 first, where it already saturates, so it fits in i32.
        $(#[$tf])?
        #[inline]
        pub unsafe fn activate_l1(l1: &Aligned<[i32; L1_OUT]>) -> Aligned<[u8; A1_SIZE]> {
            const N_VECS: usize = L1_OUT / simd::I32_LANES;
            let zero = simd::zeroed_i32();
            let max = simd::splat_i32(HIDDEN_MAX);
            let sqr_lo = simd::splat_i32(-L1_SQR_CLAMP);
            let sqr_hi = simd::splat_i32(L1_SQR_CLAMP);
            let mut acts = [simd::zeroed_i32(); 2 * N_VECS];
            for j in 0..N_VECS {
                let x = *l1.0.as_ptr().add(j * simd::I32_LANES).cast();
                let xc = simd::clamp_i32(x, sqr_lo, sqr_hi);
                let sqr = simd::srai_i32(simd::mullo_i32(xc, xc), L1_SQR_SHIFT);
                acts[j] = simd::clamp_i32(sqr, zero, max);
                acts[N_VECS + j] = simd::clamp_i32(simd::srai_i32(x, L1_SHIFT), zero, max);
            }
            let mut out = core::mem::MaybeUninit::<Aligned<[u8; A1_SIZE]>>::uninit();
            let dst = out.as_mut_ptr().cast::<u8>();
            for k in 0..A1_SIZE / (4 * simd::I32_LANES) {
                *dst.add(k * 4 * simd::I32_LANES).cast() =
                    simd::narrow_u8(acts[4 * k], acts[4 * k + 1], acts[4 * k + 2], acts[4 * k + 3]);
            }
            out.assume_init()
        }

        // ============================================================
        // L2: dense dpbusd → gate
        // ============================================================

        /// L2 dense layer and its gate: u8[A1_SIZE] × i8 weights (×64) + bias →
        /// gates `g` (outputs 0..32) and values `v` (32..64), each shifted to unit
        /// 128, `g` clipped to [0, 127], `v` to [−127, 127]; returns
        /// `h = (g · g · v) >> 14`, unit 128, signed.
        $(#[$tf])?
        #[inline]
        pub unsafe fn propagate_l2(
            params: &network::NNUEParams,
            a1: &Aligned<[u8; A1_SIZE]>,
            stack: usize,
        ) -> Aligned<[i32; GATED]> {
            const N_VECS: usize = L2_OUT / simd::I32_LANES;
            const G_VECS: usize = GATED / simd::I32_LANES;
            let packed = a1.0.as_ptr() as *const i32;
            // Two accumulator sets over the even and odd input groups: twice the
            // independent chains for native dpbusd, the same sums.
            let mut even = [simd::zeroed_i32(); N_VECS];
            let mut odd = [simd::zeroed_i32(); N_VECS];
            for g in (0..A1_SIZE / 4).step_by(2) {
                let in0 = simd::splat_i32(*packed.add(g));
                let in1 = simd::splat_i32(*packed.add(g + 1));
                let w0 = params.l2_weights.0[stack][g].as_ptr();
                let w1 = params.l2_weights.0[stack][g + 1].as_ptr();
                for j in 0..N_VECS {
                    let off = j * simd::I32_LANES * 4;
                    even[j] = simd::dpbusd(even[j], in0, *w0.add(off).cast());
                    odd[j] = simd::dpbusd(odd[j], in1, *w1.add(off).cast());
                }
            }

            let biases = params.l2_biases.0[stack].as_ptr();
            let zero = simd::zeroed_i32();
            let max = simd::splat_i32(HIDDEN_MAX);
            let min = simd::splat_i32(-HIDDEN_MAX);
            let mut out = core::mem::MaybeUninit::<Aligned<[i32; GATED]>>::uninit();
            let dst = out.as_mut_ptr().cast::<i32>();
            for j in 0..G_VECS {
                let gate_at = j * simd::I32_LANES;
                let value_at = (G_VECS + j) * simd::I32_LANES;
                let gate = simd::add_i32(
                    simd::add_i32(even[j], odd[j]),
                    *biases.add(gate_at).cast(),
                );
                let value = simd::add_i32(
                    simd::add_i32(even[G_VECS + j], odd[G_VECS + j]),
                    *biases.add(value_at).cast(),
                );
                let g = simd::clamp_i32(simd::srai_i32(gate, L2_SHIFT), zero, max);
                let v = simd::clamp_i32(simd::srai_i32(value, L2_SHIFT), min, max);
                let h = simd::srai_i32(simd::mullo_i32(simd::mullo_i32(g, g), v), GATE_SHIFT);
                *dst.add(gate_at).cast() = h;
            }
            out.assume_init()
        }

        // ============================================================
        // L3: output
        // ============================================================

        /// L3 output layer: `dot(a1, wa) + dot(h, wh) + bias` → i32, unit 2^14.
        /// The u8 activations go through dpbusd a whole register at a time; the
        /// signed gated values through i32 products.
        $(#[$tf])?
        #[inline]
        pub unsafe fn propagate_l3(
            params: &network::NNUEParams,
            a1: &Aligned<[u8; A1_SIZE]>,
            h: &Aligned<[i32; GATED]>,
            stack: usize,
        ) -> i32 {
            let wa = params.l3_weights_a.0[stack].as_ptr();
            let wh = params.l3_weights_h.0[stack].as_ptr();
            let mut acc = simd::zeroed_i32();
            for k in 0..A1_SIZE / (4 * simd::I32_LANES) {
                let at = k * 4 * simd::I32_LANES;
                acc = simd::dpbusd(acc, *a1.0.as_ptr().add(at).cast(), *wa.add(at).cast());
            }
            let mut acc_h = simd::zeroed_i32();
            for j in 0..GATED / simd::I32_LANES {
                let at = j * simd::I32_LANES;
                acc_h = simd::add_i32(
                    acc_h,
                    simd::mullo_i32(*h.0.as_ptr().add(at).cast(), *wh.add(at).cast()),
                );
            }
            simd::horizontal_sum_i32(simd::add_i32(acc, acc_h)) + params.l3_biases.0[stack]
        }

        /// Everything after the sparse inputs are known: L1, its activation, L2 and
        /// its gate, L3, plus the skip `l1[30] − l1[31]`. Unit 2^14.
        $(#[$tf])?
        #[inline]
        pub unsafe fn dense_layers(
            params: &network::NNUEParams,
            ft_out: &Aligned<[u8; L1_SIZE]>,
            nnz: &[u16],
            stack: usize,
        ) -> i32 {
            debug_assert!(stack < crate::nnue::STACKS, "stack {stack}");
            let l1 = propagate_l1(params, ft_out, nnz, stack);
            let skip = l1.0[SKIP_POS].wrapping_sub(l1.0[SKIP_NEG]);
            let a1 = activate_l1(&l1);
            let h = propagate_l2(params, &a1, stack);
            propagate_l3(params, &a1, &h, stack).wrapping_add(skip)
        }

        // ============================================================
        // Dense forward pass — the per-eval composite
        // ============================================================

        /// Forward pass through the dense layers: FT activation → NNZ extraction
        /// → sparse L1 → gated L2 → L3 with the skip. Unit 2^14.
        ///
        /// One composite entry point so the stages inline into each other; only
        /// `find_nnz` sits behind its own seam (its algorithm is chosen on an
        /// axis independent from the register width).
        $(#[$tf])?
        #[inline]
        pub unsafe fn forward_dense(
            params: &network::NNUEParams,
            acc: &Aligned<[[i16; L1_SIZE]; 2]>,
            stm: Color,
            stack: usize,
        ) -> i32 {
            // Both buffers are written in place, never zeroed nor copied: the
            // activation fills every byte, `find_nnz` the first `nnz_count` slots.
            let mut ft_buf = core::mem::MaybeUninit::<Aligned<[u8; L1_SIZE]>>::uninit();
            activate_ft_into(acc, stm, ft_buf.as_mut_ptr().cast::<u8>());
            let ft_out = ft_buf.assume_init_ref();
            let mut nnz_indices =
                core::mem::MaybeUninit::<[u16; crate::nnue::forward::NNZ_SIZE]>::uninit();
            // The nnz algorithm rides an axis of its own (VBMI2, not the register
            // width); when the build cannot pin it statically, one predicted
            // indirect call fetches it from the resolved table.
            #[cfg(any(
                not(target_arch = "x86_64"),
                all(target_feature = "avx512bw", target_feature = "avx512vbmi2")
            ))]
            let nnz_count =
                crate::nnue::forward::find_nnz(ft_out, nnz_indices.as_mut_ptr().cast::<u16>());
            #[cfg(all(
                target_arch = "x86_64",
                not(all(target_feature = "avx512bw", target_feature = "avx512vbmi2"))
            ))]
            let nnz_count =
                (crate::cpu::get().find_nnz)(ft_out, nnz_indices.as_mut_ptr().cast::<u16>());
            #[cfg(feature = "stats")]
            crate::nnue::forward::tally_nnz(nnz_count);
            debug_assert!(nnz_count <= L1_SIZE / 4, "nnz count {nnz_count}");
            dense_layers(
                params,
                ft_out,
                core::slice::from_raw_parts(nnz_indices.as_ptr().cast::<u16>(), nnz_count),
                stack,
            )
        }

        $(
        /// [`forward_dense`] on an AVX-512 CPU with VBMI2: the activation lists the
        /// non-zero groups itself, so no pass reads the output back.
        #[$tf2]
        #[inline]
        pub unsafe fn forward_dense_vbmi2(
            params: &network::NNUEParams,
            acc: &Aligned<[[i16; L1_SIZE]; 2]>,
            stm: Color,
            stack: usize,
        ) -> i32 {
            let mut ft_buf = core::mem::MaybeUninit::<Aligned<[u8; L1_SIZE]>>::uninit();
            let mut nnz_indices =
                core::mem::MaybeUninit::<[u16; crate::nnue::forward::NNZ_SIZE]>::uninit();
            let nnz_count = activate_ft_nnz_into(
                acc,
                stm,
                ft_buf.as_mut_ptr().cast::<u8>(),
                nnz_indices.as_mut_ptr().cast::<u16>(),
            );
            let ft_out = ft_buf.assume_init_ref();
            #[cfg(feature = "stats")]
            crate::nnue::forward::tally_nnz(nnz_count);
            debug_assert!(nnz_count <= L1_SIZE / 4, "nnz count {nnz_count}");
            dense_layers(
                params,
                ft_out,
                core::slice::from_raw_parts(nnz_indices.as_ptr().cast::<u16>(), nnz_count),
                stack,
            )
        }

        /// [`activate_ft_into`], plus the indexes of the non-zero 4-byte groups
        /// written to `nnz` (room for `NNZ_SIZE`, slots past the count clobbered);
        /// returns their count.
        ///
        /// Each pair of packed registers is narrowed to one word per group
        /// (`packs_epi32`: non-zero stays non-zero), tested, and the matching
        /// indexes compressed out of a register (`VPCOMPRESSW`, VBMI2) and stored
        /// whole. The list is in group order within each 128-bit lane, not sorted:
        /// the sparse L1 sums in i32, where the order does not matter.
        #[$tf2]
        #[inline]
        pub unsafe fn activate_ft_nnz_into(
            acc: &Aligned<[[i16; L1_SIZE]; 2]>,
            stm: Color,
            output: *mut u8,
            nnz: *mut u16,
        ) -> usize {
            use core::arch::x86_64::*;
            const _: () = assert!(simd::I16_LANES == 32);
            const _: () = assert!(FT_SHIFT == 9);
            let half = L1_SIZE / 2;
            let zero = simd::splat_i16(0);
            let one = simd::splat_i16(FT_QUANT);
            // The group of each word of `packs_epi32(a, b)`: per 128-bit lane, four
            // groups of `a`, then the same four of `b` (16 groups further on).
            let lane_order = _mm512_set_epi16(
                31, 30, 29, 28, 15, 14, 13, 12, 27, 26, 25, 24, 11, 10, 9, 8, 23, 22, 21, 20,
                7, 6, 5, 4, 19, 18, 17, 16, 3, 2, 1, 0,
            );
            let mut count = 0usize;

            for flip in 0..2usize {
                let pov = stm.index() ^ flip;
                let input = acc.0[pov].as_ptr();
                let mut indexes =
                    _mm512_add_epi16(lane_order, _mm512_set1_epi16((flip * half / 4) as i16));

                for i in (0..half).step_by(4 * simd::I16_LANES) {
                    let mut packed = [_mm512_setzero_si512(); 2];
                    for k in 0..2 {
                        let at = i + 2 * k * simd::I16_LANES;
                        let lhs1 = simd::clamp_i16(*input.add(at).cast(), zero, one);
                        let lhs2 =
                            simd::clamp_i16(*input.add(at + simd::I16_LANES).cast(), zero, one);
                        let rhs1 = simd::min_i16(*input.add(at + half).cast(), one);
                        let rhs2 =
                            simd::min_i16(*input.add(at + half + simd::I16_LANES).cast(), one);
                        packed[k] = simd::packus(
                            simd::ft_product(lhs1, rhs1),
                            simd::ft_product(lhs2, rhs2),
                        );
                        *output.add(at + flip * half).cast() = packed[k];
                    }

                    let words = _mm512_packs_epi32(packed[0], packed[1]);
                    let mask = _mm512_test_epi16_mask(words, words);
                    debug_assert!(
                        count + 32 <= crate::nnue::forward::NNZ_SIZE,
                        "nnz store at {count}"
                    );
                    // A register store, not `compressstoreu`: the memory form is
                    // microcoded on Zen 4.
                    _mm512_storeu_si512(
                        nnz.add(count).cast(),
                        _mm512_maskz_compress_epi16(mask, indexes),
                    );
                    count += mask.count_ones() as usize;
                    indexes = _mm512_add_epi16(indexes, _mm512_set1_epi16(32));
                }
            }
            debug_assert!(count <= L1_SIZE / 4, "nnz count {count}");
            count
        }
        )?

        // ============================================================
        // Threat accumulator batch apply — register-tiled form
        // ============================================================

        /// Extract indices of non-zero 4-byte groups from the FT output:
        /// NNZ_TABLE walk, one lookup per 8 groups.
        ///
        /// This is the portable form (scalar, NEON, wasm128); x86 vector tiers
        /// use the specialized forms in `forward.rs` (SSE stores / VPCOMPRESSW).
        ///
        /// Each lookup writes all 8 table slots and advances by the entry's
        /// count, so no loop length depends on the data; the slots past the
        /// count land in `NNZ_PAD` or are overwritten by the next lookup.
        $(#[$tf])?
        #[inline]
        pub unsafe fn find_nnz_generic(
            ft_out: &Aligned<[u8; L1_SIZE]>,
            indexes: *mut u16,
        ) -> usize {
            use crate::nnue::forward::{NNZ_SIZE, NNZ_TABLE};

            // Groups per step: at least one table byte, at least one vector.
            const STEP: usize = if simd::I32_LANES > 8 { simd::I32_LANES } else { 8 };
            const _: () = assert!(STEP % simd::I32_LANES == 0 && STEP % 8 == 0 && STEP <= 32);
            const _: () = assert!((L1_SIZE / 4) % STEP == 0);

            let mut count = 0;

            let mut base: u16 = 0;
            let ptr = ft_out.0.as_ptr() as *const i32;
            let out = indexes;

            for chunk_start in (0..L1_SIZE / 4).step_by(STEP) {
                let mut mask: u32 = 0;
                for v in 0..STEP / simd::I32_LANES {
                    let bits = simd::nnz_bitmask(*ptr.add(chunk_start + v * simd::I32_LANES).cast());
                    mask |= u32::from(bits) << (v * simd::I32_LANES);
                }

                for byte in 0..STEP / 8 {
                    let entry = &NNZ_TABLE[((mask >> (8 * byte)) & 0xFF) as usize];
                    debug_assert!(count + 8 <= NNZ_SIZE, "nnz store at {count}");
                    for k in 0..8 {
                        *out.add(count + k) = base + entry.indexes[k];
                    }
                    count += entry.count;
                    base += 8;
                }
            }

            debug_assert!(count <= L1_SIZE / 4, "nnz count {count}");
            count
        }

        /// Batched threat-row accumulation: keep a tile of the accumulator in
        /// registers and apply every add/sub row against it, then store once.
        ///
        /// Per element the arithmetic is wrapping i16, so neither the tiling nor
        /// the pairing of rows (`apply_i8_rows`) can move a lane: the search node
        /// count is that of a row-at-a-time walk.
        ///
        /// `input_acc`: source accumulator (null = zero-init for full recompute).
        /// `output_acc`: destination accumulator. In-place apply (same pointer
        /// twice) is allowed — pawn-pair rows stacked on an already-filled
        /// threat acc.
        ///
        /// The AVX-512 tiers use `threat512::threat_batch` instead (16 ZMMs).
        /// No software prefetch: the rows a search touches stay in LLC, and
        /// look-ahead branches in these loops measured slower than nothing.
        $(#[$tf])?
        #[inline]
        pub unsafe fn threat_batch(
            input_acc: *const i16,
            output_acc: *mut i16,
            weights_base: *const crate::nnue::network::ThreatRow,
            adds: &[u32],
            subs: &[u32],
        ) {
            debug_assert!(!output_acc.is_null());
            const TILE: usize = crate::nnue::kernels::THREAT_REGISTERS;
            const TILE_ELEMS: usize = TILE * simd::I16_LANES;
            // Register pairs, as listed to `apply_i8_rows`.
            #[cfg(not(target_arch = "aarch64"))]
            const _: () = assert!(TILE == 8);
            #[cfg(target_arch = "aarch64")]
            const _: () = assert!(TILE == 16);

            for tile in 0..(L1_SIZE / TILE_ELEMS) {
                let base = tile * TILE_ELEMS;
                let mut regs: [_; TILE] = std::mem::zeroed();
                if !input_acc.is_null() {
                    for i in 0..TILE {
                        regs[i] = *input_acc.add(base + i * simd::I16_LANES).cast();
                    }
                }

                #[cfg(not(target_arch = "aarch64"))]
                {
                    apply_i8_rows!(simd, regs, weights_base, base, subs, simd::sub_i16, [0, 1, 2, 3]);
                    apply_i8_rows!(simd, regs, weights_base, base, adds, simd::add_i16, [0, 1, 2, 3]);
                }
                #[cfg(target_arch = "aarch64")]
                {
                    apply_i8_rows!(simd, regs, weights_base, base, subs, simd::sub_i16,
                                   [0, 1, 2, 3, 4, 5, 6, 7]);
                    apply_i8_rows!(simd, regs, weights_base, base, adds, simd::add_i16,
                                   [0, 1, 2, 3, 4, 5, 6, 7]);
                }

                for (i, r) in regs.into_iter().enumerate() {
                    *output_acc.add(base + i * simd::I16_LANES).cast() = r;
                }
            }
        }

        /// One tile pass for a whole update of a perspective: the piece-square,
        /// threat and pawn-pair rows (all i8) a move changes, applied to `input_acc`
        /// and stored in `output_acc` (the same pointer twice is allowed). The
        /// accumulator is read and written once.
        ///
        /// Wrapping i16 as in `threat_batch`: the order of the rows moves no lane.
        $(#[$tf])?
        #[inline]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn combined_batch(
            input_acc: *const i16,
            output_acc: *mut i16,
            psq_base: *const crate::nnue::network::ThreatRow,
            psq_adds: &[u32],
            psq_subs: &[u32],
            weights_base: *const crate::nnue::network::ThreatRow,
            adds: &[u32],
            subs: &[u32],
        ) {
            debug_assert!(!input_acc.is_null() && !output_acc.is_null());
            const TILE: usize = crate::nnue::kernels::THREAT_REGISTERS;
            const TILE_ELEMS: usize = TILE * simd::I16_LANES;

            for tile in 0..(L1_SIZE / TILE_ELEMS) {
                let base = tile * TILE_ELEMS;
                let mut regs: [_; TILE] = std::mem::zeroed();
                for i in 0..TILE {
                    regs[i] = *input_acc.add(base + i * simd::I16_LANES).cast();
                }

                #[cfg(not(target_arch = "aarch64"))]
                {
                    apply_i8_rows!(simd, regs, psq_base, base, psq_subs, simd::sub_i16, [0, 1, 2, 3]);
                    apply_i8_rows!(simd, regs, psq_base, base, psq_adds, simd::add_i16, [0, 1, 2, 3]);
                    apply_i8_rows!(simd, regs, weights_base, base, subs, simd::sub_i16, [0, 1, 2, 3]);
                    apply_i8_rows!(simd, regs, weights_base, base, adds, simd::add_i16, [0, 1, 2, 3]);
                }
                #[cfg(target_arch = "aarch64")]
                {
                    apply_i8_rows!(simd, regs, psq_base, base, psq_subs, simd::sub_i16,
                                   [0, 1, 2, 3, 4, 5, 6, 7]);
                    apply_i8_rows!(simd, regs, psq_base, base, psq_adds, simd::add_i16,
                                   [0, 1, 2, 3, 4, 5, 6, 7]);
                    apply_i8_rows!(simd, regs, weights_base, base, subs, simd::sub_i16,
                                   [0, 1, 2, 3, 4, 5, 6, 7]);
                    apply_i8_rows!(simd, regs, weights_base, base, adds, simd::add_i16,
                                   [0, 1, 2, 3, 4, 5, 6, 7]);
                }

                for (i, r) in regs.into_iter().enumerate() {
                    *output_acc.add(base + i * simd::I16_LANES).cast() = r;
                }
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
    use crate::nnue::L1_SIZE;
    use crate::nnue::simd::avx512 as simd;

    /// `input_acc`: source accumulator (null = zero-init for full recompute).
    /// `output_acc`: destination accumulator.
    #[cfg_attr(
        not(all(target_feature = "avx512f", target_feature = "avx512bw")),
        target_feature(enable = "avx512f,avx512bw")
    )]
    pub unsafe fn threat_batch(
        input_acc: *const i16,
        output_acc: *mut i16,
        weights_base: *const crate::nnue::network::ThreatRow,
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

            // Eight register pairs for the 16-ZMM tile.
            apply_i8_rows!(
                simd,
                regs,
                weights_base,
                base,
                subs,
                simd::sub_i16,
                [0, 1, 2, 3, 4, 5, 6, 7]
            );
            apply_i8_rows!(
                simd,
                regs,
                weights_base,
                base,
                adds,
                simd::add_i16,
                [0, 1, 2, 3, 4, 5, 6, 7]
            );

            for i in 0..TILE {
                _mm512_storeu_si512(output_acc.add(base + i * 32) as *mut __m512i, regs[i]);
            }
        }
    }

    /// [`threat_batch`] with the move's piece-square rows (i8) applied in the same
    /// tile pass: one read and one write of the accumulator per update.
    #[cfg_attr(
        not(all(target_feature = "avx512f", target_feature = "avx512bw")),
        target_feature(enable = "avx512f,avx512bw")
    )]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn combined_batch(
        input_acc: *const i16,
        output_acc: *mut i16,
        psq_base: *const crate::nnue::network::ThreatRow,
        psq_adds: &[u32],
        psq_subs: &[u32],
        weights_base: *const crate::nnue::network::ThreatRow,
        adds: &[u32],
        subs: &[u32],
    ) {
        use std::arch::x86_64::*;
        debug_assert!(!input_acc.is_null() && !output_acc.is_null());
        const TILE: usize = 16;
        const TILE_ELEMS: usize = TILE * 32;
        const _: () = assert!(L1_SIZE % TILE_ELEMS == 0);

        for tile in 0..(L1_SIZE / TILE_ELEMS) {
            let base = tile * TILE_ELEMS;
            let mut regs: [__m512i; TILE] = [_mm512_setzero_si512(); TILE];
            for i in 0..TILE {
                regs[i] = _mm512_loadu_si512(input_acc.add(base + i * 32) as *const __m512i);
            }

            apply_i8_rows!(
                simd,
                regs,
                psq_base,
                base,
                psq_subs,
                simd::sub_i16,
                [0, 1, 2, 3, 4, 5, 6, 7]
            );
            apply_i8_rows!(
                simd,
                regs,
                psq_base,
                base,
                psq_adds,
                simd::add_i16,
                [0, 1, 2, 3, 4, 5, 6, 7]
            );
            apply_i8_rows!(
                simd,
                regs,
                weights_base,
                base,
                subs,
                simd::sub_i16,
                [0, 1, 2, 3, 4, 5, 6, 7]
            );
            apply_i8_rows!(
                simd,
                regs,
                weights_base,
                base,
                adds,
                simd::add_i16,
                [0, 1, 2, 3, 4, 5, 6, 7]
            );

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
//  AVX2+AVX-VNNI > AVX2 > NEON > wasm simd128 > scalar.)
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
        )],
        vbmi2: #[cfg_attr(
            not(all(
                target_feature = "avx2",
                target_feature = "fma",
                target_feature = "avx512f",
                target_feature = "avx512bw",
                target_feature = "avx512vnni",
                target_feature = "avx512vbmi2"
            )),
            target_feature(enable = "avx2,fma,avx512f,avx512bw,avx512vnni,avx512vbmi2")
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
        )],
        vbmi2: #[cfg_attr(
            not(all(
                target_feature = "avx2",
                target_feature = "fma",
                target_feature = "avx512f",
                target_feature = "avx512bw",
                target_feature = "avx512vbmi2"
            )),
            target_feature(enable = "avx2,fma,avx512f,avx512bw,avx512vbmi2")
        )]
    );
}

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
pub mod k_avxvnni {
    nnue_kernels!(
        crate::nnue::simd::avxvnni,
        #[cfg_attr(
            not(all(
                target_feature = "avx2",
                target_feature = "fma",
                target_feature = "avxvnni"
            )),
            target_feature(enable = "avx2,fma,avxvnni")
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
#[allow(dead_code)]
pub mod k_neon {
    nnue_kernels!(crate::nnue::simd::neon);
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[allow(dead_code)]
pub mod k_dotprod {
    nnue_kernels!(
        crate::nnue::simd::neon_dotprod,
        #[cfg_attr(
            not(target_feature = "dotprod"),
            target_feature(enable = "dotprod")
        )]
    );
}

// The scalar kernels of an aarch64 build, for tests: the vector tiers must match
// them bit for bit.
#[cfg(all(test, target_arch = "aarch64", target_feature = "neon"))]
#[allow(dead_code)]
pub mod k_scalar {
    nnue_kernels!(crate::nnue::simd::scalar);
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub mod k_native {
    nnue_kernels!(crate::nnue::simd::wasm128);
}

// The scalar instance of a simd128 build, so tests can hold it against `k_native`.
#[cfg(all(test, target_arch = "wasm32", target_feature = "simd128"))]
#[allow(dead_code)]
pub mod k_scalar {
    nnue_kernels!(crate::nnue::simd::scalar);
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
    target_feature = "avx2",
    not(target_feature = "avxvnni"),
    not(target_feature = "avx512f")
))]
pub(crate) use k_avx2 as k;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "avxvnni",
    not(target_feature = "avx512f")
))]
pub(crate) use k_avxvnni as k;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    not(target_feature = "avx512vnni")
))]
pub(crate) use k_avx512 as k;
#[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
pub(crate) use k_dotprod as k;
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub(crate) use k_native as k;
#[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    not(target_feature = "dotprod")
))]
pub(crate) use k_neon as k;
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
pub(crate) use k_scalar as k;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vnni"
))]
pub(crate) use k_vnni512 as k;

// Private: the layout these assume is the build target's, not the resolved tier's.
// Everything outside this module goes through `dispatch`, which picks the one that
// matches the weights (a universal build on AVX-512 would otherwise apply AVX2 rows).
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
use k::{combined_batch, threat_batch};
/// Threat batch apply for the current build target: the hand-scheduled 16-ZMM
/// form when AVX-512 is available, the tiled generic form otherwise.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use threat512::{combined_batch, threat_batch};

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
#[cfg(target_arch = "aarch64")]
pub(crate) const STATIC_TOP: bool = cfg!(target_feature = "dotprod");
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[allow(dead_code)]
pub(crate) const STATIC_TOP: bool = false;

// ============================================================
// aarch64 runtime table — NEON vs NEON+DOTPROD
// ============================================================

/// Elects the NEON kernel instance from HWCAP (and `GAIA_SIMD=neon|dotprod`).
///
/// A build whose target already pins `dotprod` bypasses the table, matching
/// `STATIC_TOP` on x86-64.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub mod arm {
    use super::{CombinedBatchFn, k_dotprod, k_neon};
    use crate::nnue::L1_SIZE;
    use crate::nnue::network::{Aligned, NNUEParams};
    use crate::types::Color;
    use std::sync::OnceLock;

    pub struct Table {
        pub forward_dense: unsafe fn(&NNUEParams, &Aligned<[[i16; L1_SIZE]; 2]>, Color, usize) -> i32,
        pub threat_batch: unsafe fn(*const i16, *mut i16, *const crate::nnue::network::ThreatRow, &[u32], &[u32]),
        pub combined_batch: CombinedBatchFn,
        pub dotprod: bool,
        #[allow(dead_code)]
        pub forced: bool,
    }

    static TABLE: OnceLock<Table> = OnceLock::new();

    fn read_cap() -> Option<bool> {
        let v = std::env::var("GAIA_SIMD").ok()?;
        match v.to_ascii_lowercase().as_str() {
            "neon" => Some(false),
            "dotprod" | "neon-dotprod" => Some(true),
            other => {
                eprintln!("info string GAIA_SIMD '{other}' ignored (neon|dotprod)");
                None
            }
        }
    }

    fn resolve() -> Table {
        let detected =
            cfg!(target_feature = "dotprod") || std::arch::is_aarch64_feature_detected!("dotprod");
        let cap = read_cap();
        let (use_dot, forced) = if super::STATIC_TOP {
            if cap.is_some() {
                eprintln!("info string GAIA_SIMD ignored: this build pins DOTPROD at compile time");
            }
            (true, false)
        } else if let Some(want) = cap {
            if want && !detected {
                eprintln!("info string GAIA_SIMD=dotprod ignored: this CPU has no ASIMDDP");
                (false, false)
            } else {
                (want, true)
            }
        } else {
            (detected, false)
        };

        if use_dot {
            Table {
                forward_dense: k_dotprod::forward_dense,
                threat_batch: k_dotprod::threat_batch,
                combined_batch: k_dotprod::combined_batch,
                dotprod: true,
                forced,
            }
        } else {
            Table {
                forward_dense: k_neon::forward_dense,
                threat_batch: k_neon::threat_batch,
                combined_batch: k_neon::combined_batch,
                dotprod: false,
                forced,
            }
        }
    }

    pub fn get_or_init() -> &'static Table {
        TABLE.get_or_init(resolve)
    }

    #[inline(always)]
    pub fn get() -> &'static Table {
        #[cfg(test)]
        return TABLE.get_or_init(resolve);
        #[cfg(not(test))]
        unsafe {
            debug_assert!(
                TABLE.get().is_some(),
                "arm dispatch read before init_cpu_dispatch"
            );
            TABLE.get().unwrap_unchecked()
        }
    }

    pub fn init() {
        let _ = get_or_init();
    }

    pub fn label() -> &'static str {
        if get_or_init().dotprod {
            "NEON+DOTPROD"
        } else {
            "NEON"
        }
    }
}

// ============================================================
// Dispatch helpers — the one place that decides between the static
// instance and the runtime table. Every hot caller goes through here.
// ============================================================

pub(crate) mod dispatch {
    use crate::nnue::L1_SIZE;
    use crate::nnue::network::{Aligned, NNUEParams};
    use crate::types::Color;

    /// Forward pass through the dense layers, on the resolved tier: the network's
    /// output, unit 2^14.
    #[inline(always)]
    pub unsafe fn forward_dense(
        params: &NNUEParams,
        acc: &Aligned<[[i16; L1_SIZE]; 2]>,
        stm: Color,
        stack: usize,
    ) -> i32 {
        #[cfg(target_arch = "x86_64")]
        {
            // A target that pins VBMI2 as well lists the non-zero groups in the activation.
            #[cfg(all(target_feature = "avx512vnni", target_feature = "avx512vbmi2"))]
            let top = super::k::forward_dense_vbmi2;
            #[cfg(not(all(target_feature = "avx512vnni", target_feature = "avx512vbmi2")))]
            let top = super::k::forward_dense;
            if super::STATIC_TOP {
                top(params, acc, stm, stack)
            } else {
                (crate::cpu::get().forward_dense)(params, acc, stm, stack)
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if super::STATIC_TOP {
                super::k::forward_dense(params, acc, stm, stack)
            } else {
                (super::arm::get().forward_dense)(params, acc, stm, stack)
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            super::k::forward_dense(params, acc, stm, stack)
        }
    }

    /// Batched threat-row accumulation.
    #[inline(always)]
    pub unsafe fn threat_batch(
        input_acc: *const i16,
        output_acc: *mut i16,
        weights_base: *const crate::nnue::network::ThreatRow,
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
        #[cfg(target_arch = "aarch64")]
        {
            if super::STATIC_TOP {
                super::threat_batch(input_acc, output_acc, weights_base, adds, subs)
            } else {
                (super::arm::get().threat_batch)(input_acc, output_acc, weights_base, adds, subs)
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            super::threat_batch(input_acc, output_acc, weights_base, adds, subs)
        }
    }

    /// One tile pass applying a move's piece-square, threat and pawn-pair rows.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn combined_batch(
        input_acc: *const i16,
        output_acc: *mut i16,
        pst_base: *const crate::nnue::network::ThreatRow,
        pst_adds: &[u32],
        pst_subs: &[u32],
        weights_base: *const crate::nnue::network::ThreatRow,
        adds: &[u32],
        subs: &[u32],
    ) {
        #[cfg(target_arch = "x86_64")]
        {
            if super::STATIC_TOP {
                super::combined_batch(
                    input_acc, output_acc, pst_base, pst_adds, pst_subs, weights_base, adds, subs,
                )
            } else {
                (crate::cpu::get().combined_batch)(
                    input_acc, output_acc, pst_base, pst_adds, pst_subs, weights_base, adds, subs,
                )
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if super::STATIC_TOP {
                super::combined_batch(
                    input_acc, output_acc, pst_base, pst_adds, pst_subs, weights_base, adds, subs,
                )
            } else {
                (super::arm::get().combined_batch)(
                    input_acc, output_acc, pst_base, pst_adds, pst_subs, weights_base, adds, subs,
                )
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            super::combined_batch(
                input_acc, output_acc, pst_base, pst_adds, pst_subs, weights_base, adds, subs,
            )
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

            let got = unsafe { k_scalar::activate_ft(&fused(&lin_pst, &lin_comb), stm) };
            assert_activation_matches("scalar", &got, &want);

            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                let (pst, comb) = inputs_permuted_for(PermKind::Avx2);
                let got = unsafe { k_avx2::activate_ft(&fused(&pst, &comb), stm) };
                assert_activation_matches("avx2", &got, &want);

                if std::arch::is_x86_feature_detected!("avxvnni") {
                    let got = unsafe { k_avxvnni::activate_ft(&fused(&pst, &comb), stm) };
                    assert_activation_matches("avxvnni", &got, &want);
                }
            }

            if std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512bw")
            {
                let (pst, comb) = inputs_permuted_for(PermKind::File512);
                let got = unsafe { k_avx512::activate_ft(&fused(&pst, &comb), stm) };
                assert_activation_matches("avx512", &got, &want);

                if std::arch::is_x86_feature_detected!("avx512vnni") {
                    let got = unsafe { k_vnni512::activate_ft(&fused(&pst, &comb), stm) };
                    assert_activation_matches("vnni512", &got, &want);
                }
            }
        }
    }

    #[test]
    fn the_activation_lists_the_same_groups_as_the_separate_pass() {
        if !(std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vbmi2")
            && std::arch::is_x86_feature_detected!("fma"))
        {
            return;
        }
        type ActivateNnzFn = unsafe fn(&Aligned<[[i16; L1_SIZE]; 2]>, Color, *mut u8, *mut u16) -> usize;
        let mut forms: Vec<(&str, ActivateNnzFn)> = vec![("avx512", k_avx512::activate_ft_nnz_into)];
        if std::arch::is_x86_feature_detected!("avx512vnni") {
            forms.push(("vnni512", k_vnni512::activate_ft_nnz_into));
        }

        // The usual inputs, then sparse ones: one live pair in a few groups,
        // first and last of a register and of a perspective included.
        let (pst, comb) = inputs_permuted_for(PermKind::File512);
        let mut cases = vec![("dense", fused(&pst, &comb))];
        for live in [vec![], vec![0], vec![3, 64, 127, 128, 511], (0..512).step_by(5).collect()] {
            let mut acc = Aligned([[-7i16; L1_SIZE]; 2]);
            for pov in 0..2 {
                for &i in &live {
                    let i = (i + 37 * pov) % (L1_SIZE / 2);
                    acc.0[pov][i] = 200;
                    acc.0[pov][i + L1_SIZE / 2] = 150;
                }
                linear_to(PermKind::File512, &mut acc.0[pov]);
            }
            cases.push(("sparse", acc));
        }

        for (case, acc) in &cases {
            for stm in [Color::White, Color::Black] {
                let want_out = unsafe { k_avx512::activate_ft(acc, stm) };
                let mut want = [0u16; crate::nnue::forward::NNZ_SIZE];
                let n = unsafe { crate::nnue::forward::find_nnz_compress512(&want_out, want.as_mut_ptr()) };
                for (form, f) in &forms {
                    let mut out = Aligned([0u8; L1_SIZE]);
                    let mut got = [0u16; crate::nnue::forward::NNZ_SIZE];
                    let count = unsafe { f(acc, stm, out.0.as_mut_ptr(), got.as_mut_ptr()) };
                    assert_activation_matches(form, &out, &want_out.0);
                    let mut got = got[..count].to_vec();
                    got.sort_unstable();
                    assert_eq!(got, want[..n], "{form} nnz ({case}, {stm:?})");
                }
            }
        }
    }

    #[test]
    fn the_three_nnz_forms_agree() {
        // The generic form at 8 and 16 lanes too: it must handle vectors wider
        // than one table byte, though no x86 build selects it.
        let mut forms: Vec<(&str, super::nnz_cases::NnzFn)> =
            vec![("scalar", k_scalar::find_nnz_generic)];
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            forms.push(("avx2", crate::nnue::forward::find_nnz_avx2));
            forms.push(("avx2 generic", k_avx2::find_nnz_generic));
        }
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("fma")
        {
            forms.push(("avx512 generic", k_avx512::find_nnz_generic));
        }
        if std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vbmi2")
        {
            forms.push(("compress512", crate::nnue::forward::find_nnz_compress512));
        }
        super::nnz_cases::assert_forms(&forms);
    }

    #[test]
    fn threat_batch_forms_agree_on_synthetic_weights() {
        type BatchFn = unsafe fn(*const i16, *mut i16, *const crate::nnue::network::ThreatRow, &[u32], &[u32]);
        let avx2 = std::arch::is_x86_feature_detected!("avx2");
        let avx512 = std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw");
        // Each kernel with the accumulator layout of the tier that runs it; the
        // weight rows are in logical order for all of them.
        let mut forms: Vec<(&str, BatchFn, PermKind)> =
            vec![("scalar", k_scalar::threat_batch, PermKind::Linear)];
        if avx2 {
            forms.push(("avx2", k_avx2::threat_batch, PermKind::Avx2));
        }
        if avx512 {
            forms.push(("threat512", threat512::threat_batch, PermKind::File512));
            forms.push(("avx512 tile", k_avx512::threat_batch, PermKind::File512));
        }

        for case in threat_cases::cases() {
            let want = threat_cases::elementwise(&case);
            for &(form, f, kind) in &forms {
                let mut want_t = want;
                linear_to(kind, &mut want_t);
                let start_t = case.start.map(|mut v| {
                    linear_to(kind, &mut v);
                    Aligned(v)
                });
                let src = start_t.as_ref().map_or(std::ptr::null(), |v| v.0.as_ptr());

                let mut out = Aligned([0i16; L1_SIZE]);
                unsafe {
                    f(
                        src,
                        out.0.as_mut_ptr(),
                        case.rows.as_ptr(),
                        &case.adds,
                        &case.subs,
                    )
                };
                assert_eq!(out.0, want_t, "{form} threat batch ({})", case.name);

                if let Some(mut acc) = start_t {
                    let p = acc.0.as_mut_ptr();
                    unsafe { f(p, p, case.rows.as_ptr(), &case.adds, &case.subs) };
                    assert_eq!(
                        acc.0, want_t,
                        "{form} threat batch in place ({})",
                        case.name
                    );
                }
            }
        }
    }

    #[test]
    fn combined_batch_forms_agree_on_synthetic_weights() {
        let avx2 = std::arch::is_x86_feature_detected!("avx2");
        let avx512 = std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw");
        // As for `threat_batch`: each kernel with its tier's accumulator layout, every
        // i8 row (piece squares included) in logical order.
        let mut forms: Vec<(&str, super::CombinedBatchFn, PermKind)> =
            vec![("scalar", k_scalar::combined_batch, PermKind::Linear)];
        if avx2 {
            forms.push(("avx2", k_avx2::combined_batch, PermKind::Avx2));
        }
        if avx512 {
            forms.push(("threat512", threat512::combined_batch, PermKind::File512));
            forms.push(("avx512 tile", k_avx512::combined_batch, PermKind::File512));
        }

        let psq = threat_cases::psq_rows();
        let rows = threat_cases::rows_of(&psq);
        for case in threat_cases::cases() {
            let Some(start) = case.start else { continue };
            for (adds, subs) in threat_cases::psq_sets() {
                let want = threat_cases::combined_elementwise(&case, &psq, &adds, &subs);
                for &(form, f, kind) in &forms {
                    let mut want_t = want;
                    linear_to(kind, &mut want_t);
                    let mut acc = Aligned(start);
                    linear_to(kind, &mut acc.0);
                    let p = acc.0.as_mut_ptr();
                    unsafe {
                        f(p, p, rows.as_ptr(), &adds, &subs, case.rows.as_ptr(), &case.adds, &case.subs)
                    };
                    assert_eq!(acc.0, want_t, "{form} combined batch ({}, psq {adds:?}/{subs:?})", case.name);
                }
            }
        }
    }

    #[test]
    fn every_tier_computes_the_dense_layers_of_the_reference() {
        let mut forms: Vec<(&str, super::dense_cases::DenseFn)> =
            vec![("scalar", k_scalar::dense_layers)];
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            forms.push(("avx2", k_avx2::dense_layers));
            if std::arch::is_x86_feature_detected!("avxvnni") {
                forms.push(("avxvnni", k_avxvnni::dense_layers));
            }
        }
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
        {
            forms.push(("avx512", k_avx512::dense_layers));
            if std::arch::is_x86_feature_detected!("avx512vnni") {
                forms.push(("vnni512", k_vnni512::dense_layers));
            }
        }
        super::dense_cases::assert_reference(&forms);
    }
}

/// FT outputs for the nnz extraction and their group-by-group oracle, shared by
/// the tests of every architecture.
#[cfg(test)]
mod nnz_cases {
    use crate::nnue::L1_SIZE;
    use crate::nnue::forward::NNZ_SIZE;
    use crate::nnue::network::Aligned;

    pub type NnzFn = unsafe fn(&Aligned<[u8; L1_SIZE]>, *mut u16) -> usize;

    /// Empty, full, and three densities of non-zero bytes. Bytes stay below
    /// 128, as the activation's do, so "some byte is non-zero" and "the group
    /// read as an i32 is positive" (the vector compare) are the same test.
    fn outputs() -> Vec<(String, Aligned<[u8; L1_SIZE]>)> {
        let mut out = vec![
            ("empty".to_string(), Aligned([0u8; L1_SIZE])),
            ("full".to_string(), Aligned([1u8; L1_SIZE])),
        ];
        let mut state = 0x9E3779B97F4A7C15u64;
        for one_in in [2u64, 5, 23] {
            let mut ft = Aligned([0u8; L1_SIZE]);
            for b in ft.0.iter_mut() {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                if state % one_in == 0 {
                    *b = ((state >> 32) as u8 & 0x7F).max(1);
                }
            }
            out.push((format!("1 byte in {one_in}"), ft));
        }
        out
    }

    /// Indices of the 4-byte groups holding a non-zero byte, in order.
    fn oracle(ft: &Aligned<[u8; L1_SIZE]>) -> Vec<u16> {
        ft.0.chunks_exact(4)
            .enumerate()
            .filter(|(_, g)| g.iter().any(|&b| b != 0))
            .map(|(i, _)| i as u16)
            .collect()
    }

    pub fn assert_forms(forms: &[(&str, NnzFn)]) {
        for (case, ft) in outputs() {
            let want = oracle(&ft);
            for &(form, f) in forms {
                let mut idx = [0u16; NNZ_SIZE];
                let n = unsafe { f(&ft, idx.as_mut_ptr()) };
                assert_eq!(&idx[..n], &want[..], "{form} nnz ({case})");
            }
        }
    }
}

/// Synthetic threat-batch cases and their element-wise oracle, shared by the
/// tests of every architecture.
#[cfg(test)]
mod threat_cases {
    use crate::nnue::L1_SIZE;

    pub struct Case {
        pub name: &'static str,
        /// Logical order; `None` = the kernel starts from zero (null input).
        pub start: Option<[i16; L1_SIZE]>,
        /// Logical order, as `repermute_ft` holds threat rows on every tier.
        pub weights: Vec<[i8; L1_SIZE]>,
        /// The same weights as the kernels read them, one padded row each.
        pub rows: Vec<crate::nnue::network::ThreatRow>,
        pub adds: Vec<u32>,
        pub subs: Vec<u32>,
    }

    /// Element-wise wrapping i16, one row at a time: the contract the tiled,
    /// paired kernels must meet bit for bit, or the bench node count moves.
    pub fn elementwise(case: &Case) -> [i16; L1_SIZE] {
        let mut out = case.start.unwrap_or([0i16; L1_SIZE]);
        for &s in &case.subs {
            for (o, &w) in out.iter_mut().zip(&case.weights[s as usize]) {
                *o = o.wrapping_sub(w as i16);
            }
        }
        for &a in &case.adds {
            for (o, &w) in out.iter_mut().zip(&case.weights[a as usize]) {
                *o = o.wrapping_add(w as i16);
            }
        }
        out
    }

    /// Piece-square rows for `combined_batch`, logical order, the whole i8 range.
    pub fn psq_rows() -> Vec<[i8; L1_SIZE]> {
        (0..5)
            .map(|r| std::array::from_fn(|i| ((r * 7919 + i * 131) % 256) as u8 as i8))
            .collect()
    }

    /// The rows as the kernels read them.
    pub fn rows_of(weights: &[[i8; L1_SIZE]]) -> Vec<crate::nnue::network::ThreatRow> {
        weights.iter().map(|w| crate::nnue::network::ThreatRow::new(*w)).collect()
    }

    /// The piece-square rows a move adds and removes: one each, two each, none,
    /// castling-like.
    pub fn psq_sets() -> Vec<(Vec<u32>, Vec<u32>)> {
        vec![
            (vec![0], vec![1]),
            (vec![2, 3], vec![4, 0]),
            (vec![], vec![]),
            (vec![1], vec![2, 3]),
        ]
    }

    /// [`elementwise`] with piece-square rows on top: the contract of
    /// `combined_batch`.
    pub fn combined_elementwise(
        case: &Case,
        psq: &[[i8; L1_SIZE]],
        adds: &[u32],
        subs: &[u32],
    ) -> [i16; L1_SIZE] {
        let mut out = elementwise(case);
        for &s in subs {
            for (o, &w) in out.iter_mut().zip(&psq[s as usize]) {
                *o = o.wrapping_sub(w as i16);
            }
        }
        for &a in adds {
            for (o, &w) in out.iter_mut().zip(&psq[a as usize]) {
                *o = o.wrapping_add(w as i16);
            }
        }
        out
    }

    /// Pairs, an odd row out on either side, empty lists, a null input, the
    /// extreme pair sums (127 + 127, -128 + -128) and i16 wrap-around.
    pub fn cases() -> Vec<Case> {
        const ROWS: usize = 8;
        let weights: Vec<[i8; L1_SIZE]> = (0..ROWS)
            .map(|r| std::array::from_fn(|i| (((r * 251 + i * 31) % 199) as i32 - 99) as i8))
            .collect();
        let input: [i16; L1_SIZE] = std::array::from_fn(|i| ((i as i32 * 97) % 1601 - 800) as i16);
        let extremes: Vec<[i8; L1_SIZE]> = (0..ROWS)
            .map(|r| std::array::from_fn(|i| if (r + i) % 2 == 0 { 127 } else { -128 }))
            .collect();
        let wrap_input: [i16; L1_SIZE] =
            std::array::from_fn(|i| if i % 2 == 0 { 30_000 } else { -30_000 });

        let case = |name, start, weights: &Vec<[i8; L1_SIZE]>, adds: &[u32], subs: &[u32]| Case {
            name,
            start,
            weights: weights.clone(),
            rows: rows_of(weights),
            adds: adds.to_vec(),
            subs: subs.to_vec(),
        };
        vec![
            case("copy", Some(input), &weights, &[0, 3, 6], &[1, 5]),
            case("from_zero", None, &weights, &[0, 3, 6], &[1, 5]),
            case("empty", Some(input), &weights, &[], &[]),
            case("adds_only", Some(input), &weights, &[2, 7], &[]),
            case("subs_only", Some(input), &weights, &[], &[4, 1, 0]),
            case("one_each", Some(input), &weights, &[5], &[2]),
            case("same_row_twice", Some(input), &weights, &[3, 3], &[6]),
            case("extremes", Some(input), &extremes, &[0, 2], &[1, 3]),
            case(
                "wrap",
                Some(wrap_input),
                &extremes,
                &[0, 1, 2, 3],
                &[4, 5, 6, 7],
            ),
        ]
    }
}

/// The integer dense layers of every tier against the plain reference of
/// `forward.rs`, on random weights and inputs, shared by the tests of every
/// architecture: all of them must compute the same integers, so that every build
/// evaluates every position the same.
#[cfg(test)]
mod dense_cases {
    use crate::nnue::network::{Aligned, NNUEParams};
    use crate::nnue::{L1_SIZE, STACKS};

    pub type DenseFn = unsafe fn(&NNUEParams, &Aligned<[u8; L1_SIZE]>, &[u16], usize) -> i32;

    struct XorShift(u64);

    impl XorShift {
        fn bits(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn int(&mut self, lo: i32, hi: i32) -> i32 {
            lo + (self.bits() % (hi - lo + 1) as u64) as i32
        }
    }

    /// Random weights in their exported ranges; biases wide enough that every clamp
    /// and saturation of the activations is crossed.
    pub fn random_params(seed: u64) -> Box<NNUEParams> {
        let mut rng = XorShift(seed);
        let mut params = NNUEParams::zeroed_box();
        for s in 0..STACKS {
            for group in params.l1_weights.0[s].iter_mut() {
                group.iter_mut().for_each(|w| *w = rng.int(-127, 127) as i8);
            }
            params.l1_biases.0[s].iter_mut().for_each(|b| *b = rng.int(-40_000, 40_000));
            for group in params.l2_weights.0[s].iter_mut() {
                group.iter_mut().for_each(|w| *w = rng.int(-127, 127) as i8);
            }
            params.l2_biases.0[s].iter_mut().for_each(|b| *b = rng.int(-20_000, 20_000));
            params.l3_weights_a.0[s].iter_mut().for_each(|w| *w = rng.int(-127, 127) as i8);
            params.l3_weights_h.0[s].iter_mut().for_each(|w| *w = rng.int(-127, 127));
            params.l3_biases.0[s] = rng.int(-20_000, 20_000);
        }
        params.meta.0[0] = crate::nnue::NETWORK_SCALE;
        params
    }

    pub fn assert_reference(forms: &[(&str, DenseFn)]) {
        let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
        let params = random_params(0x2545_F491_4F6C_DD1D);
        for round in 0..300 {
            // FT output: products in 0..=127, about half the bytes zero.
            let mut ft_out = Aligned([0u8; L1_SIZE]);
            for x in ft_out.0.iter_mut() {
                let v = (rng.bits() >> 57) as u8;
                *x = if v < 64 { 0 } else { v };
            }
            let nnz: Vec<u16> = (0..(L1_SIZE / 4) as u16)
                .filter(|&g| ft_out.0[4 * g as usize..4 * g as usize + 4].iter().any(|&b| b != 0))
                .collect();
            for stack in 0..STACKS {
                let want = crate::nnue::forward::dense_reference(&params, &ft_out, stack);
                for &(name, f) in forms {
                    let got = unsafe { f(&params, &ft_out, &nnz, stack) };
                    assert_eq!(got, want, "{name}: round {round}, stack {stack}");
                }
            }
        }
    }
}

#[cfg(all(test, target_arch = "aarch64", target_feature = "neon"))]
mod tests_aarch64 {
    use super::*;
    use crate::nnue::accumulator::Accumulator;
    use crate::nnue::network::Aligned;
    use crate::nnue::{FT_QUANT, FT_SHIFT, L1_SIZE};
    use crate::types::Color;

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

    #[test]
    fn neon_activation_lands_on_the_scalar_output() {
        let (pst, comb) = test_inputs();
        for stm in [Color::White, Color::Black] {
            let want = activate_reference(&pst, &comb, stm);
            let got = unsafe { k_neon::activate_ft(&fused(&pst, &comb), stm) };
            if let Some(i) = (0..L1_SIZE).find(|&i| got.0[i] != want[i]) {
                panic!(
                    "neon activation disagrees at {i}: got {}, want {}",
                    got.0[i], want[i]
                );
            }
            if std::arch::is_aarch64_feature_detected!("dotprod") {
                let got = unsafe { k_dotprod::activate_ft(&fused(&pst, &comb), stm) };
                if let Some(i) = (0..L1_SIZE).find(|&i| got.0[i] != want[i]) {
                    panic!(
                        "dotprod activation disagrees at {i}: got {}, want {}",
                        got.0[i], want[i]
                    );
                }
            }
        }
    }

    #[test]
    fn neon_and_dotprod_threat_batch_match_the_elementwise_walk() {
        type BatchFn = unsafe fn(*const i16, *mut i16, *const crate::nnue::network::ThreatRow, &[u32], &[u32]);
        let mut forms: Vec<(&str, BatchFn)> = vec![("neon", k_neon::threat_batch)];
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            forms.push(("dotprod", k_dotprod::threat_batch));
        }
        for case in super::threat_cases::cases() {
            let want = super::threat_cases::elementwise(&case);
            for &(form, f) in &forms {
                let start = case.start.map(Aligned);
                let src = start.as_ref().map_or(std::ptr::null(), |v| v.0.as_ptr());
                let mut out = Aligned([0i16; L1_SIZE]);
                unsafe {
                    f(
                        src,
                        out.0.as_mut_ptr(),
                        case.rows.as_ptr(),
                        &case.adds,
                        &case.subs,
                    )
                };
                assert_eq!(out.0, want, "{form} threat batch ({})", case.name);

                if let Some(mut acc) = start {
                    let p = acc.0.as_mut_ptr();
                    unsafe { f(p, p, case.rows.as_ptr(), &case.adds, &case.subs) };
                    assert_eq!(acc.0, want, "{form} threat batch in place ({})", case.name);
                }
            }
        }
    }

    #[test]
    fn neon_and_dotprod_combined_batch_match_the_elementwise_walk() {
        let mut forms: Vec<(&str, super::CombinedBatchFn)> = vec![("neon", k_neon::combined_batch)];
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            forms.push(("dotprod", k_dotprod::combined_batch));
        }
        let psq = super::threat_cases::psq_rows();
        let rows = super::threat_cases::rows_of(&psq);
        for case in super::threat_cases::cases() {
            let Some(start) = case.start else { continue };
            for (adds, subs) in super::threat_cases::psq_sets() {
                let want = super::threat_cases::combined_elementwise(&case, &psq, &adds, &subs);
                for &(form, f) in &forms {
                    let mut acc = Aligned(start);
                    let p = acc.0.as_mut_ptr();
                    unsafe {
                        f(p, p, rows.as_ptr(), &adds, &subs, case.rows.as_ptr(), &case.adds, &case.subs)
                    };
                    assert_eq!(acc.0, want, "{form} combined batch ({}, psq {adds:?}/{subs:?})", case.name);
                }
            }
        }
    }

    #[test]
    fn neon_and_scalar_nnz_match_the_group_walk() {
        let mut forms: Vec<(&str, super::nnz_cases::NnzFn)> = vec![
            ("neon", k_neon::find_nnz_generic),
            ("scalar", k_scalar::find_nnz_generic),
        ];
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            forms.push(("dotprod", k_dotprod::find_nnz_generic));
        }
        super::nnz_cases::assert_forms(&forms);
    }

    #[test]
    fn neon_and_dotprod_compute_the_dense_layers_of_the_reference() {
        let mut forms: Vec<(&str, super::dense_cases::DenseFn)> =
            vec![("neon", k_neon::dense_layers), ("scalar", k_scalar::dense_layers)];
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            forms.push(("dotprod", k_dotprod::dense_layers));
        }
        super::dense_cases::assert_reference(&forms);
    }
}

// simd128 against the scalar build of the same module: the dense layers are integers,
// so they must agree bit for bit, and a rung of the web build is the same opponent with
// and without SIMD.
#[cfg(all(test, target_arch = "wasm32", target_feature = "simd128"))]
mod tests_wasm {
    use super::*;

    #[test]
    fn simd128_and_scalar_nnz_match_the_group_walk() {
        super::nnz_cases::assert_forms(&[
            ("simd128", k_native::find_nnz_generic),
            ("scalar", k_scalar::find_nnz_generic),
        ]);
    }

    #[test]
    fn dense_layers_match_the_reference() {
        super::dense_cases::assert_reference(&[
            ("simd128", k_native::dense_layers),
            ("scalar", k_scalar::dense_layers),
        ]);
    }
}
