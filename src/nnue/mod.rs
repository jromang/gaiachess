//! NNUE evaluation: efficiently updatable neural network.
//!
//! Architecture GaiaNet-T2:
//!   ```text
//!   Position
//!     |
//!     +-- HalfKA PST (12 king buckets x 768 = 9,216 features, i16 weights)
//!     |     -> [PST Accumulator i16[1024] x 2 perspectives, incremental + Finny cache]
//!     |
//!     +-- Pawn pairs (4,560 features, i8, 3-file window) + pairwise threats (59,808, i8)
//!     |     -> [ThreatAccumulator i16[1024] x 2, incremental + dirty updates]
//!     |
//!     +----> PST + aux -> activate_ft -> within-side pairwise -> u8[1024]
//!              -> L1 sparse (1024->16, dpbusd) -> CReLU+squared -> f32[32]
//!              -> L2 dense (32->32, FMA) -> squared -> f32[32]
//!              -> L3 dense (concat(l2[32], l1[32]) -> 1) x 8 output buckets
//!              -> + PSQT head (i32 per feature per bucket, stm - ntm) -> centipawns
//!   ```
//!
//! Submodules:
//!   - `features`: HalfKA PST indexing (12 king buckets, horizontal mirroring)
//!   - `threats`: pairwise threat features (59,808), incremental dirty updates
//!   - `pawn_pairs`: 3-file pawn-pawn relations (4,560)
//!   - `accumulator`: PST accumulator (Finny cache), incremental push/pop
//!   - `network`: NNUEParams struct (~85 MB), weight loading (direct memcpy)
//!   - `forward`: forward pass (activate_ft, NNZ, L1/L2/L3 with skip connection)
//!   - `simd`: SIMD primitives (AVX-512 > AVX2 > NEON > scalar)
//!
//! Dual perspective (STM + NSTM). SIMD-accelerated (AVX2/AVX-512) with scalar fallback.

pub mod accumulator;
pub mod features;
pub mod integrity;
pub(crate) mod forward;
pub(crate) mod kernels;
pub mod network;
pub mod pawn_pairs;
pub mod simd;
pub mod threats;

use std::mem::MaybeUninit;

use crate::position::Position;
use crate::types::*;

use accumulator::{AccDelta, Accumulator, FinnyTable};
use threats::ThreatAccumulator;

// ============================================================
// Architecture constants
// ============================================================

/// Number of king position buckets (see features::KING_BUCKETS).
pub const INPUT_BUCKETS: usize = 12;

/// Features per bucket: 2 colors × 6 piece types × 64 squares.
pub const INPUTS_PER_BUCKET: usize = 768;

/// Total PST feature transformer input size.
pub const FT_SIZE: usize = INPUT_BUCKETS * INPUTS_PER_BUCKET;

/// 3-file pawn-pair features: C(96, 2) = 4,560.
/// 48 pawn squares × 2 colours, unordered pairs, i8 weights.
pub const PAWN_PAIR_SIZE: usize = 96 * 95 / 2;

/// Pairwise threat features (kings never attack or are targets; pawn-pawn
/// relations live in `PAWN_PAIR_SIZE` instead).
/// P 4×84 + N 10×336 + B 8×560 + R 8×896 + Q 10×1456 = 29,904 → ×2 colours = 59,808.
pub const THREAT_FEATURE_COUNT: usize = 59_808;

/// Offset of threat features inside the i8 aux-weight array (pawn pairs first).
pub const THREAT_OFFSET: usize = PAWN_PAIR_SIZE;

/// Combined i8 feature-transformer input: pawn pairs then threats.
pub const THREAT_INPUT_SIZE: usize = PAWN_PAIR_SIZE + THREAT_FEATURE_COUNT;

/// Feature transformer output / L1 input size (per perspective).
pub const L1_SIZE: usize = 1024;

const _: () = assert!(PAWN_PAIR_SIZE == 4_560);
const _: () = assert!(THREAT_FEATURE_COUNT == 59_808);
const _: () = assert!(THREAT_INPUT_SIZE == 64_368);
const _: () = assert!(L1_SIZE % 64 == 0, "packus permutation groups are 64 wide");

/// L1 output (before pairing): matmul output size per bucket.
pub const L2_SIZE: usize = 16;

/// L2 output / L3 skip input size.
pub const L3_SIZE: usize = 32;

/// Number of output buckets (material-based).
pub const OUTPUT_BUCKETS: usize = 8;

/// Feature transformer quantization factor (CReLU clamp max).
pub const FT_QUANT: i16 = 255;

/// L1 weight quantization factor.
pub const L1_QUANT: i32 = 64;

/// Right-shift for pairwise product: `(a * b) >> FT_SHIFT`.
pub const FT_SHIFT: i32 = 9;

/// L1 dequantization multiplier: converts i32 dpbusd accumulator to f32.
/// `(1 << FT_SHIFT) / (FT_QUANT * FT_QUANT * L1_QUANT)`
pub const L1_NORMALISATION: f32 =
    (1 << FT_SHIFT) as f32 / (FT_QUANT as f32 * FT_QUANT as f32 * L1_QUANT as f32);

/// Final output scaling to centipawns.
/// Network output calibration constant (287 centipawns per unit).
pub const NETWORK_SCALE: i32 = 287;

/// PSQT head quantization factor: weights stored as `round(w × 16384)` in i32.
/// Power of two so the dequantization divide is exact in f32.
pub const PSQT_QUANT: i32 = 16384;

// ============================================================
// Output bucket scheme
// ============================================================

/// Output bucket scheme: material-value weighted.
/// `[knight, bishop, rook, queen weight, divisor, max bucket]` —
/// `bucket = min(max, (3N + 4B + 8R + 18Q) / 12)`, pawns and kings excluded,
/// so pawn moves and pawn trades never change the bucket.
/// Hashed into ARCH_HASH in this order; the trainer mirrors the same scheme
/// from its own constants (an accidental drift fails the golden tests).
pub const OUTPUT_BUCKET_SCHEME: [usize; 6] = [3, 4, 8, 18, 12, OUTPUT_BUCKETS - 1];

// ============================================================
// Network struct
// ============================================================

/// Per-thread NNUE state: PST accumulator stack + threat accumulator stack.
///
/// Threats are updated incrementally when possible (diff-based),
/// falling back to full recompute when king mirroring changes.
pub struct Network {
    /// Current ply index in the accumulator stack.
    index: usize,
    /// PST accumulator stack: one per ply (pre-allocated).
    accumulators: Box<[Accumulator]>,
    /// Finny table: PST accumulator refresh cache per (perspective, mirrored, bucket).
    cache: FinnyTable,
    /// Threat accumulator stack: one per ply (pre-allocated).
    threat_acc: Box<[ThreatAccumulator]>,
}

/// The board a chain slot names: the position itself, or one of the rebuilt plies.
#[inline]
fn board_at<'a>(
    pos: &'a Position,
    rebuilt: &'a [MaybeUninit<threats::RebuiltBoard>],
    slot: u8,
    occ: u64,
) -> threats::BoardView<'a> {
    if slot == u8::MAX {
        threats::BoardView::of(pos)
    } else {
        // SAFETY: a slot is named only after `ensure_threats_updated` wrote it.
        unsafe { rebuilt[slot as usize].assume_init_ref() }.view(occ)
    }
}

impl Network {
    /// Create a new Network with pre-allocated accumulator stack.
    pub fn new() -> Self {
        let accumulators = (0..MAX_PLY + 1)
            .map(|_| Accumulator::new())
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let threat_acc = (0..MAX_PLY + 1)
            .map(|_| ThreatAccumulator::new())
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Network {
            index: 0,
            accumulators,
            cache: FinnyTable::new(),
            threat_acc,
        }
    }

    /// Initialize the root PST accumulator and threat accumulator from the current position.
    /// Must be called before the first search.
    pub fn refresh(&mut self, pos: &Position) {
        self.index = 0;
        self.accumulators[0].refresh_with_cache(pos, Color::White, &mut self.cache);
        self.accumulators[0].refresh_with_cache(pos, Color::Black, &mut self.cache);
        debug_assert!(self.accumulators[0].accurate == [true; 2],
            "NNUE refresh: PST accumulator not accurate after refresh");
        // Full threat compute at root (no parent available)
        self.threat_acc[0].update_full(pos);
    }

    /// Push a new ply onto the PST accumulator stack.
    ///
    /// Must be called BEFORE `Position::make_move()` so we can read the
    /// moved and captured pieces from the unmodified board.
    pub fn push(&mut self, mv: Move, pos: &Position) {
        debug_assert!(mv != Move::NONE, "NNUE push: MOVE_NONE");
        debug_assert!(mv.from_sq().0 < 64 && mv.to_sq().0 < 64,
            "NNUE push: squares OOB from={} to={}", mv.from_sq().0, mv.to_sq().0);
        debug_assert!(pos.piece_on(mv.from_sq()) != Piece::NONE,
            "NNUE push: no piece on from {}", mv.from_sq().0);
        self.index += 1;
        debug_assert!(self.index < self.accumulators.len());

        let from = mv.from_sq();
        let to = mv.to_sq();

        self.accumulators[self.index].accurate = [false; 2];
        self.accumulators[self.index].delta = AccDelta {
            mv,
            moved_piece: pos.piece_on(from),
            captured_piece: if mv.move_type() == MT_EN_PASSANT {
                Piece::new(PieceType::Pawn, !pos.side_to_move)
            } else if mv.move_type() == MT_CASTLING {
                Piece::NONE
            } else {
                pos.piece_on(to)
            },
            old_occ: pos.occupied(),
        };
        self.threat_acc[self.index].accurate = [false; 2];
    }

    /// Push a null move onto the PST accumulator stack (lazy — no clone).
    ///
    /// CORRECTNESS: `accurate` MUST be set to `[false; 2]` here. See the
    /// stale accumulator bug documentation in MEMORY.md.
    pub fn push_null(&mut self) {
        self.index += 1;
        debug_assert!(self.index < self.accumulators.len());
        self.accumulators[self.index].accurate = [false; 2];
        self.accumulators[self.index].delta = AccDelta::default();
        self.threat_acc[self.index].accurate = [false; 2];
    }

    /// Pop one ply from the accumulator stack (after unmake_move).
    #[inline]
    pub fn pop(&mut self) {
        debug_assert!(self.index > 0);
        self.index -= 1;
    }

    /// Lazily update the PST accumulator from the nearest accurate ancestor.
    pub fn ensure_updated(&mut self, pos: &Position) {
        debug_assert!(self.index < self.accumulators.len(),
            "NNUE ensure_updated: index {} >= len {}", self.index, self.accumulators.len());

        for &perspective in &[Color::White, Color::Black] {
            let pov = perspective.index();
            if self.accumulators[self.index].accurate[pov] {
                continue;
            }

            match accumulator::find_update_source(&self.accumulators, self.index, perspective) {
                Some(source) => {
                    for ply in (source + 1)..=self.index {
                        let (left, right) = self.accumulators.split_at_mut(ply);
                        let prev = &left[ply - 1];
                        let current = &mut right[0];
                        current.update_from(prev, pos, perspective);
                    }
                }
                None => {
                    self.accumulators[self.index].refresh_with_cache(
                        pos, perspective, &mut self.cache,
                    );
                }
            }
        }
    }

    pub fn evaluate(&mut self, pos: &Position) -> i32 {
        debug_assert!(pos.checkers == 0,
            "NNUE evaluate: called while in check");

        self.ensure_all_updated(pos);
        self.forward(pos)
    }

    /// Bring both the PST and threat accumulators up-to-date without evaluating.
    ///
    /// Called at nodes that reuse a TT eval and never reach `evaluate`: their
    /// children can then take the incremental update paths instead of paying a
    /// full threat recompute each.
    pub fn ensure_all_updated(&mut self, pos: &Position) {
        self.ensure_updated(pos);
        self.ensure_threats_updated(pos);
    }

    /// Bring the threat accumulator up-to-date from the nearest ancestor whose threats
    /// are known, replaying the plies in between, or recompute it when none is close.
    ///
    /// A ply without a static eval (in check, lazy eval) leaves its threats unknown,
    /// and every child of it used to pay a full recompute. Replaying two or three
    /// plies costs a fraction of that, and the plies replayed become known for the
    /// siblings that follow. Past `THREAT_CHAIN_MAX` plies the recompute is cheaper.
    fn ensure_threats_updated(&mut self, pos: &Position) {
        let idx = self.index;
        if self.threat_acc[idx].accurate == [true; 2] {
            return;
        }

        let mut source = None;
        for k in 1..=threats::THREAT_CHAIN_MAX.min(idx) {
            if self.threat_acc[idx - k].accurate == [true; 2] {
                source = Some(idx - k);
                break;
            }
        }
        let Some(source) = source else {
            self.threat_acc[idx].update_full(pos);
            return;
        };
        let len = idx - source;

        // The boards of the plies between `pos` and the source, rebuilt backwards from
        // the deltas and kept by reference: `slot[k]` says where the board k plies above
        // `pos` lives — `POS` for the position itself, else an index into `rebuilt`. A
        // null move leaves the board as it was, so its ply shares the slot of the ply
        // after it. Nothing is copied but the arrays that have to be rebuilt.
        const POS: u8 = u8::MAX;
        let mut rebuilt: [MaybeUninit<threats::RebuiltBoard>; threats::THREAT_CHAIN_MAX] =
            [const { MaybeUninit::uninit() }; threats::THREAT_CHAIN_MAX];
        let mut slot = [POS; threats::THREAT_CHAIN_MAX + 1];
        let mut occ = [0u64; threats::THREAT_CHAIN_MAX + 1];
        occ[0] = pos.occupied();
        let mut used = 0usize;
        for k in 1..=len {
            let delta = &self.accumulators[idx - k + 1].delta;
            if delta.mv == Move::NONE {
                slot[k] = slot[k - 1];
                occ[k] = occ[k - 1];
            } else {
                // The slot being written is never the one being read: the board below
                // lives in `pos` or in an earlier slot.
                let (done, free) = rebuilt.split_at_mut(used);
                let below = board_at(pos, done, slot[k - 1], occ[k - 1]);
                below.rebuild_before_into(delta, &mut free[0]);
                slot[k] = used as u8;
                occ[k] = delta.old_occ;
                used += 1;
            }
        }

        let mut replayed = true;
        for ply in (source + 1)..=idx {
            let k = idx - ply;
            let delta = &self.accumulators[ply].delta;
            let (left, right) = self.threat_acc.split_at_mut(ply);
            let parent = &left[ply - 1];
            let current = &mut right[0];
            if delta.mv == Move::NONE {
                current.copy_from(parent);
                continue;
            }
            let new = board_at(pos, &rebuilt, slot[k], occ[k]);
            let old = board_at(pos, &rebuilt, slot[k + 1], occ[k + 1]);
            if !current.update_incremental(&new, &old, parent, delta) {
                // A king crossed the centre at this ply: nothing before it carries over.
                replayed = false;
                break;
            }
        }
        if !replayed {
            self.threat_acc[idx].update_full(pos);
        }

        #[cfg(debug_assertions)]
        {
            let (full, full_psqt) = threats::compute_full_threats(pos);
            let acc = &self.threat_acc[idx];
            for pov in 0..2 {
                debug_assert!(
                    acc.values.0[pov][..] == full.0[pov][..],
                    "threat chain mismatch against the full recompute: pov={pov}"
                );
                debug_assert_eq!(
                    acc.psqt[pov], full_psqt[pov],
                    "threat chain PSQT mismatch against the full recompute: pov={pov}"
                );
            }
        }
    }

    /// Test helper: expose ensure_threats_updated (used by threats.rs tests).
    #[cfg(test)]
    pub(crate) fn ensure_threats_updated_for_test(&mut self, pos: &Position) {
        self.ensure_threats_updated(pos);
    }

    /// Test helper: current ply's threat accumulator values.
    #[cfg(test)]
    pub(crate) fn threat_values_for_test(&self) -> &network::Aligned<[[i16; L1_SIZE]; 2]> {
        &self.threat_acc[self.index].values
    }

    /// Test helper: current ply's threat accumulator PSQT head sums.
    #[cfg(test)]
    pub(crate) fn threat_psqt_for_test(&self) -> &[[i32; OUTPUT_BUCKETS]; 2] {
        &self.threat_acc[self.index].psqt
    }

    /// Test helper: current ply's PST accumulator PSQT head sums.
    #[cfg(test)]
    pub(crate) fn pst_psqt_for_test(&self) -> [[i32; OUTPUT_BUCKETS]; 2] {
        self.accumulators[self.index].psqt
    }

    /// Forward pass through the dense layers (SIMD-accelerated),
    /// plus the additive PSQT head (`stm − ntm`, selected bucket).
    fn forward(&self, pos: &Position) -> i32 {
        let stm = pos.side_to_move;
        let acc = &self.accumulators[self.index];
        let threat_acc = &self.threat_acc[self.index];
        let bucket = output_bucket(pos);

        // Use threat accumulator (incrementally updated or full recomputed)
        let threats = &threat_acc.values;
        let l3_out = unsafe { kernels::dispatch::forward_dense(acc, threats, stm, bucket) };

        let stm_i = stm.index();
        let ntm_i = (!stm).index();
        let psqt_raw = (acc.psqt[stm_i][bucket] + threat_acc.psqt[stm_i][bucket])
            - (acc.psqt[ntm_i][bucket] + threat_acc.psqt[ntm_i][bucket]);
        let psqt = psqt_raw as f32 / PSQT_QUANT as f32;

        ((l3_out + psqt) * NETWORK_SCALE as f32) as i32
    }
}

impl Clone for Network {
    fn clone(&self) -> Self {
        Network {
            index: self.index,
            accumulators: self.accumulators.clone(),
            cache: self.cache.clone(),
            threat_acc: self.threat_acc.clone(),
        }
    }
}

// ============================================================
// Output bucket selection
// ============================================================

/// Select the output bucket from the material value on the board
/// (see `OUTPUT_BUCKET_SCHEME`).
pub fn output_bucket(pos: &Position) -> usize {
    let [wn, wb, wr, wq, div, max] = OUTPUT_BUCKET_SCHEME;
    let knights = (pos.pieces[Piece::WHITE_KNIGHT.index()]
        | pos.pieces[Piece::BLACK_KNIGHT.index()]).count_ones() as usize;
    let bishops = (pos.pieces[Piece::WHITE_BISHOP.index()]
        | pos.pieces[Piece::BLACK_BISHOP.index()]).count_ones() as usize;
    let rooks = (pos.pieces[Piece::WHITE_ROOK.index()]
        | pos.pieces[Piece::BLACK_ROOK.index()]).count_ones() as usize;
    let queens = (pos.pieces[Piece::WHITE_QUEEN.index()]
        | pos.pieces[Piece::BLACK_QUEEN.index()]).count_ones() as usize;
    let value = wn * knights + wb * bishops + wr * rooks + wq * queens;
    let bucket = (value / div).min(max);
    debug_assert!(bucket < OUTPUT_BUCKETS,
        "output_bucket: bucket {} >= OUTPUT_BUCKETS {}", bucket, OUTPUT_BUCKETS);
    bucket
}


// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Position;

    #[test]
    fn test_network_new() {
        let net = Network::new();
        assert_eq!(net.index, 0);
        assert!(!net.accumulators.is_empty());
    }

    /// Pins the material-value bucket scheme on known positions; the trainer
    /// pins the SAME expectations on its side (tools/trainer6/src/outputs.rs).
    #[test]
    fn the_output_bucket_follows_material_value_not_piece_count() {
        let cases = [
            // startpos: (3*4 + 4*4 + 8*4 + 18*2) / 12 = 96/12 = 8 → capped at 7
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 7),
            // bare kings (pawns and kings are worth 0)
            ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", 0),
            // king + 8 pawns each: still bucket 0
            ("4k3/pppppppp/8/8/8/8/PPPPPPPP/4K3 w - - 0 1", 0),
            // KRvK: 8/12 = 0
            ("4k3/8/8/8/8/8/8/R3K3 w - - 0 1", 0),
            // KQvKQ: 36/12 = 3
            ("3qk3/8/8/8/8/8/8/3QK3 w - - 0 1", 3),
            // KRRvKRR: 32/12 = 2
            ("r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1", 2),
        ];
        for (fen, want) in cases {
            let pos = Position::from_fen(fen).unwrap();
            assert_eq!(output_bucket(&pos), want, "{fen}");
        }
    }

    #[test]
    fn test_evaluate_startpos() {
        let pos = Position::from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ).unwrap();
        let mut net = Network::new();
        net.refresh(&pos);

        let score = net.evaluate(&pos);
        // Startpos is symmetric → score should be near 0 (small due to eval noise)
        assert!(score.abs() < 500, "startpos eval should be near 0, got {score}");
    }

    #[test]
    fn test_push_pop() {
        let mut pos = Position::from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ).unwrap();
        let mut net = Network::new();
        net.refresh(&pos);

        // Push e2e4
        let mv = Move::new(Square::E2, Square::E4);
        net.push(mv, &pos);
        pos.make_move(mv);
        assert_eq!(net.index, 1);

        // Evaluate (triggers lazy update)
        let _score = net.evaluate(&pos);
        assert!(net.accumulators[1].accurate[0]);
        assert!(net.accumulators[1].accurate[1]);

        // Pop
        pos.unmake_move(mv);
        net.pop();
        assert_eq!(net.index, 0);
    }

    #[test]
    fn test_refresh_vs_incremental() {
        // The key correctness test: for any move, refresh and incremental
        // should produce identical accumulator values.
        let mut pos = Position::from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ).unwrap();
        let mut net = Network::new();
        net.refresh(&pos);

        // Play e2e4
        let mv = Move::new(Square::E2, Square::E4);
        net.push(mv, &pos);
        pos.make_move(mv);

        // Evaluate with incremental update
        let _score = net.evaluate(&pos);
        let incremental_white = net.accumulators[net.index].values.0[0];
        let incremental_black = net.accumulators[net.index].values.0[1];

        // Refresh from scratch
        let mut acc_refresh = Accumulator::new();
        acc_refresh.refresh(&pos, Color::White);
        acc_refresh.refresh(&pos, Color::Black);

        // Compare
        for i in 0..L1_SIZE {
            assert_eq!(
                incremental_white[i], acc_refresh.values.0[0][i],
                "White perspective mismatch at index {i}: incremental={}, refresh={}",
                incremental_white[i], acc_refresh.values.0[0][i]
            );
            assert_eq!(
                incremental_black[i], acc_refresh.values.0[1][i],
                "Black perspective mismatch at index {i}: incremental={}, refresh={}",
                incremental_black[i], acc_refresh.values.0[1][i]
            );
        }
    }

    #[test]
    fn test_output_bucket() {
        let pos = Position::from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ).unwrap();
        let bucket = output_bucket(&pos);
        assert_eq!(bucket, 7, "startpos has 32 pieces → bucket 7");

        let pos2 = Position::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let bucket2 = output_bucket(&pos2);
        assert_eq!(bucket2, 0, "2 kings → bucket 0");
    }

    #[test]
    fn test_ft_activation_symmetry() {
        // With zeroed weights, activation should produce all zeros
        let pst_acc = Accumulator::new();
        let combined = network::Aligned([[0i16; L1_SIZE]; 2]);
        let out = forward::activate_ft(&pst_acc, &combined, Color::White);
        for i in 0..L1_SIZE {
            assert_eq!(out.0[i], 0);
        }
    }

    // ============================================================
    // Comprehensive refresh-vs-incremental tests (all move types)
    // ============================================================

    /// Helper: play a move, verify incremental == refresh for both perspectives.
    fn assert_incremental_matches_refresh(fen: &str, mv: Move, label: &str) {
        let mut pos = Position::from_fen(fen).unwrap();
        let mut net = Network::new();
        net.refresh(&pos);

        // Push + make_move
        net.push(mv, &pos);
        pos.make_move(mv);

        // Trigger lazy incremental update (directly, not via evaluate which asserts !in_check)
        net.ensure_updated(&pos);

        let inc_w = net.accumulators[net.index].values.0[0];
        let inc_b = net.accumulators[net.index].values.0[1];

        // Fresh refresh
        let mut acc_ref = Accumulator::new();
        acc_ref.refresh(&pos, Color::White);
        acc_ref.refresh(&pos, Color::Black);

        for i in 0..L1_SIZE {
            assert_eq!(
                inc_w[i], acc_ref.values.0[0][i],
                "{label}: White mismatch at {i}: inc={}, ref={}",
                inc_w[i], acc_ref.values.0[0][i]
            );
            assert_eq!(
                inc_b[i], acc_ref.values.0[1][i],
                "{label}: Black mismatch at {i}: inc={}, ref={}",
                inc_b[i], acc_ref.values.0[1][i]
            );
        }
    }

    #[test]
    fn test_incremental_capture() {
        // Nxe5: White knight captures black pawn
        assert_incremental_matches_refresh(
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3",
            Move::new(Square::F3, Square::E5),
            "Nxe5 capture",
        );
    }

    #[test]
    fn test_incremental_en_passant() {
        // exd6 e.p.: White pawn on e5 captures black pawn on d5
        assert_incremental_matches_refresh(
            "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
            Move::new_with_type(Square::E5, Square::D6, MT_EN_PASSANT),
            "exd6 en passant",
        );
    }

    #[test]
    fn test_incremental_promotion() {
        // e8=Q: White pawn on e7 promotes to queen (no capture)
        assert_incremental_matches_refresh(
            "7k/4P3/8/8/8/8/8/4K3 w - - 0 1",
            Move::new_promotion(Square::E7, Square::E8, PieceType::Queen),
            "e8=Q promotion",
        );
    }

    #[test]
    fn test_incremental_promotion_capture() {
        // exd8=Q: White pawn captures rook on d8 and promotes
        assert_incremental_matches_refresh(
            "3r3k/4P3/8/8/8/8/8/4K3 w - - 0 1",
            Move::new_promotion(Square::E7, Square::D8, PieceType::Queen),
            "exd8=Q promotion-capture",
        );
    }

    #[test]
    fn test_incremental_underpromotion() {
        // e8=N: underpromotion to knight
        assert_incremental_matches_refresh(
            "7k/4P3/8/8/8/8/8/4K3 w - - 0 1",
            Move::new_promotion(Square::E7, Square::E8, PieceType::Knight),
            "e8=N underpromotion",
        );
    }

    #[test]
    fn test_incremental_castling_kingside() {
        // O-O: White kingside castling
        assert_incremental_matches_refresh(
            "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
            Move::new_with_type(Square::E1, Square::H1, MT_CASTLING),
            "O-O kingside",
        );
    }

    #[test]
    fn test_incremental_castling_queenside() {
        // O-O-O: White queenside castling
        assert_incremental_matches_refresh(
            "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
            Move::new_with_type(Square::E1, Square::A1, MT_CASTLING),
            "O-O-O queenside",
        );
    }

    #[test]
    fn test_incremental_castling_black_kingside() {
        // Black O-O
        assert_incremental_matches_refresh(
            "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1",
            Move::new_with_type(Square::E8, Square::H8, MT_CASTLING),
            "Black O-O",
        );
    }

    #[test]
    fn test_incremental_castling_black_queenside() {
        // Black O-O-O
        assert_incremental_matches_refresh(
            "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1",
            Move::new_with_type(Square::E8, Square::A8, MT_CASTLING),
            "Black O-O-O",
        );
    }

    #[test]
    fn test_incremental_castling_frc_swap() {
        assert_incremental_matches_refresh(
            "4k3/8/8/8/8/8/8/5KR1 w K - 0 1",
            Move::new_with_type(Square::F1, Square::G1, MT_CASTLING),
            "FRC king/rook swap O-O",
        );
    }

    #[test]
    fn test_incremental_castling_king_already_on_g() {
        assert_incremental_matches_refresh(
            "4k3/8/8/8/8/8/8/6KR w K - 0 1",
            Move::new_with_type(Square::G1, Square::H1, MT_CASTLING),
            "FRC king already on g1",
        );
    }

    #[test]
    fn test_incremental_null_move() {
        // Lazy push_null: index advances but values are NOT cloned.
        // ensure_updated copies from parent on demand.
        let mut pos = Position::from_fen(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        ).unwrap();
        let mut net = Network::new();
        net.refresh(&pos);

        // Lazy push_null: index advances, marked inaccurate
        net.push_null();
        pos.make_null_move();
        assert_eq!(net.index, 1);
        assert!(!net.accumulators[1].accurate[0]);
        assert!(!net.accumulators[1].accurate[1]);

        // Evaluate triggers lazy copy from parent
        let _score = net.evaluate(&pos);

        let acc_w = net.accumulators[net.index].values.0[0];
        let acc_b = net.accumulators[net.index].values.0[1];

        // Refresh from scratch on the null-move position
        let mut acc_ref = Accumulator::new();
        acc_ref.refresh(&pos, Color::White);
        acc_ref.refresh(&pos, Color::Black);

        for i in 0..L1_SIZE {
            assert_eq!(acc_w[i], acc_ref.values.0[0][i], "Null move: White mismatch at {i}");
            assert_eq!(acc_b[i], acc_ref.values.0[1][i], "Null move: Black mismatch at {i}");
        }

        pos.unmake_null_move();
        net.pop();
        assert_eq!(net.index, 0);
    }

    #[test]
    fn test_null_move_child_push() {
        // A real move after a null move pushes at parent_index+2 and updates correctly.
        let mut pos = Position::from_fen(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        ).unwrap();
        let mut net = Network::new();
        net.refresh(&pos);

        // Lazy push_null
        net.push_null();
        pos.make_null_move();
        assert_eq!(net.index, 1);

        // Real move from null-move position: d2d4 (white to move after null)
        let mv = Move::new(Square::D2, Square::D4);
        net.push(mv, &pos);
        pos.make_move(mv);
        assert_eq!(net.index, 2);

        // Trigger lazy update (copies parent at 1, then applies delta at 2)
        net.ensure_updated(&pos);

        let inc_w = net.accumulators[net.index].values.0[0];
        let inc_b = net.accumulators[net.index].values.0[1];

        let mut acc_ref = Accumulator::new();
        acc_ref.refresh(&pos, Color::White);
        acc_ref.refresh(&pos, Color::Black);

        for i in 0..L1_SIZE {
            assert_eq!(inc_w[i], acc_ref.values.0[0][i], "Child white at {i}");
            assert_eq!(inc_b[i], acc_ref.values.0[1][i], "Child black at {i}");
        }

        pos.unmake_move(mv);
        net.pop();
        pos.unmake_null_move();
        net.pop();
        assert_eq!(net.index, 0);
    }

    #[test]
    fn test_incremental_multi_move_sequence() {
        // Play a sequence of moves and check after each one
        let mut pos = Position::from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ).unwrap();
        let mut net = Network::new();
        net.refresh(&pos);

        let moves = [
            Move::new(Square::E2, Square::E4),  // 1. e4
            Move::new(Square::E7, Square::E5),  // 1... e5
            Move::new(Square::G1, Square::F3),  // 2. Nf3
            Move::new(Square::B8, Square::C6),  // 2... Nc6
            Move::new(Square::F1, Square::B5),  // 3. Bb5
        ];

        for (i, &mv) in moves.iter().enumerate() {
            net.push(mv, &pos);
            pos.make_move(mv);

            let _score = net.evaluate(&pos);

            let inc_w = net.accumulators[net.index].values.0[0];
            let inc_b = net.accumulators[net.index].values.0[1];

            let mut acc_ref = Accumulator::new();
            acc_ref.refresh(&pos, Color::White);
            acc_ref.refresh(&pos, Color::Black);

            for j in 0..L1_SIZE {
                assert_eq!(
                    inc_w[j], acc_ref.values.0[0][j],
                    "Move {}: White mismatch at {j}", i + 1
                );
                assert_eq!(
                    inc_b[j], acc_ref.values.0[1][j],
                    "Move {}: Black mismatch at {j}", i + 1
                );
            }
        }
    }

    #[test]
    fn test_incremental_king_move_triggers_refresh() {
        // King moves to different bucket → needs full refresh
        // King on e1 (file 4, bucket from KING_BUCKETS) → d2 (different bucket)
        let mut pos = Position::from_fen(
            "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
        ).unwrap();
        let mut net = Network::new();
        net.refresh(&pos);

        // Ke1-d2: king crosses midline (file 4→3), triggers refresh
        let mv = Move::new(Square::E1, Square::D2);
        net.push(mv, &pos);
        pos.make_move(mv);

        let _score = net.evaluate(&pos);

        let inc_w = net.accumulators[net.index].values.0[0];
        let inc_b = net.accumulators[net.index].values.0[1];

        let mut acc_ref = Accumulator::new();
        acc_ref.refresh(&pos, Color::White);
        acc_ref.refresh(&pos, Color::Black);

        for i in 0..L1_SIZE {
            assert_eq!(inc_w[i], acc_ref.values.0[0][i], "King move refresh: White at {i}");
            assert_eq!(inc_b[i], acc_ref.values.0[1][i], "King move refresh: Black at {i}");
        }
    }
}
