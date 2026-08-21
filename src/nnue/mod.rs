//! NNUE evaluation: efficiently updatable neural network.
//!
//! Architecture GaiaNet-T1:
//!   ```text
//!   Position
//!     |
//!     +-- HalfKA PST (12 king buckets x 768 = 9,216 features, i16 weights)
//!     |     -> [PST Accumulator i16[640] x 2 perspectives, incremental + Finny cache]
//!     |
//!     +-- Filtered threats (41,272 features, i8 weights, incremental + dirty updates)
//!     |   (all enemy threats; defenses only when a pawn is involved)
//!     |
//!     +----> PST + Threats -> activate_ft -> within-side pairwise -> u8[640]
//!              -> L1 sparse (640->16, dpbusd) -> CReLU+squared -> f32[32]
//!              -> L2 dense (32->32, FMA) -> squared -> f32[32]
//!              -> L3 dense (concat(l2[32], l1[32]) -> 1) x 8 output buckets -> centipawns
//!   ```
//!
//! Submodules:
//!   - `features`: HalfKA PST indexing (12 king buckets, horizontal mirroring)
//!   - `threats`: filtered threat features (41,272), incremental dirty updates
//!   - `accumulator`: PST accumulator (Finny cache), incremental push/pop
//!   - `network`: NNUEParams struct (~37 MB), weight loading (direct memcpy)
//!   - `forward`: forward pass (activate_ft, NNZ, L1/L2/L3 with skip connection)
//!   - `simd`: SIMD primitives (AVX-512 > AVX2 > NEON > scalar)
//!
//! Dual perspective (STM + NSTM). SIMD-accelerated (AVX2/AVX-512) with scalar fallback.

pub mod accumulator;
pub mod features;
pub(crate) mod forward;
pub(crate) mod kernels;
pub mod network;
pub mod simd;
pub mod threats;

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

/// Filtered threat feature transformer input size (GaiaNet-T1).
/// Matches `threats::FEATURE_COUNT` exactly (derived from PIECE_TARGET_COUNT totals):
/// per attacker color, slots × empty-board mobility:
/// P 6×84 + N 6×336 + B 5×560 + R 5×896 + Q 6×1456 + K 5×420 = 20,636 → ×2 = 41,272.
pub const THREAT_INPUT_SIZE: usize = 41_272;

/// Feature transformer output / L1 input size (per perspective).
pub const L1_SIZE: usize = 640;

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

// ============================================================
// Output bucket mapping
// ============================================================

/// Maps occupancy count (0..32) to output bucket (0..OUTPUT_BUCKETS-1).
/// Must match Bullet's `MaterialCount<8>`: `(count - 2) / 4`.
#[rustfmt::skip]
pub const OUTPUT_BUCKET_MAP: [usize; 33] = [
    0, 0, 0, 0, 0, 0,  //  0- 5 pieces (0-1 impossible but safe)
    1, 1, 1, 1,         //  6- 9
    2, 2, 2, 2,         // 10-13
    3, 3, 3, 3,         // 14-17
    4, 4, 4, 4,         // 18-21
    5, 5, 5, 5,         // 22-25
    6, 6, 6, 6,         // 26-29
    7, 7, 7,            // 30-32
];

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

        self.ensure_updated(pos);
        self.ensure_threats_updated(pos);
        self.forward(pos)
    }

    /// Update the threat accumulator incrementally from the closest accurate ancestor,
    /// or full recompute if no suitable parent exists.
    ///
    /// Uses AccDelta (moved/captured pieces, old_occ) to enumerate
    /// threats only for changed pieces (~2-4) + x-ray sliders, instead of all ~30 pieces.
    fn ensure_threats_updated(&mut self, pos: &Position) {
        let idx = self.index;
        if self.threat_acc[idx].accurate == [true; 2] {
            return;
        }

        // Try incremental from parent using AccDelta
        if idx > 0 && self.threat_acc[idx - 1].accurate == [true; 2] {
            let delta = &self.accumulators[idx].delta;
            if delta.mv != Move::NONE {
                let (left, right) = self.threat_acc.split_at_mut(idx);
                if right[0].update_incremental(pos, &left[idx - 1], delta) {
                    return;
                }
            } else {
                // Null move: no pieces changed → copy parent (1280 bytes vs full recompute)
                let (left, right) = self.threat_acc.split_at_mut(idx);
                let parent = &left[idx - 1];
                right[0].values.0 = parent.values.0;
                right[0].mirrored = parent.mirrored;
                right[0].accurate = [true; 2];
                return;
            }
        }

        // Fallback: full recompute
        self.threat_acc[idx].update_full(pos);
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

    /// Forward pass through the dense layers (SIMD-accelerated).
    fn forward(&self, pos: &Position) -> i32 {
        let stm = pos.side_to_move;
        let acc = &self.accumulators[self.index];
        let bucket = output_bucket(pos);

        // Use threat accumulator (incrementally updated or full recomputed)
        let threats = &self.threat_acc[self.index].values;
        let l3_out = unsafe { kernels::dispatch::forward_dense(acc, threats, stm, bucket) };

        (l3_out * NETWORK_SCALE as f32) as i32
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

/// Select the output bucket based on the number of pieces on the board.
fn output_bucket(pos: &Position) -> usize {
    let count = pos.occupied().count_ones() as usize;
    debug_assert!((2..=32).contains(&count),
        "output_bucket: piece count {} out of range", count);
    let bucket = OUTPUT_BUCKET_MAP[count.min(32)];
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
            Move::new_with_type(Square::E1, Square::G1, MT_CASTLING),
            "O-O kingside",
        );
    }

    #[test]
    fn test_incremental_castling_queenside() {
        // O-O-O: White queenside castling
        assert_incremental_matches_refresh(
            "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
            Move::new_with_type(Square::E1, Square::C1, MT_CASTLING),
            "O-O-O queenside",
        );
    }

    #[test]
    fn test_incremental_castling_black_kingside() {
        // Black O-O
        assert_incremental_matches_refresh(
            "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1",
            Move::new_with_type(Square::E8, Square::G8, MT_CASTLING),
            "Black O-O",
        );
    }

    #[test]
    fn test_incremental_castling_black_queenside() {
        // Black O-O-O
        assert_incremental_matches_refresh(
            "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1",
            Move::new_with_type(Square::E8, Square::C8, MT_CASTLING),
            "Black O-O-O",
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
