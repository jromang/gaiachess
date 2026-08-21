// Hand-vectorised code: the loop index is the point. It walks several arrays in
// lockstep, steps by a SIMD lane count, and feeds raw pointer arithmetic — an
// iterator would hide the arithmetic these kernels exist to control.
#![allow(clippy::needless_range_loop)]
//! NNUE accumulator: dual-perspective i16 vector with incremental updates.
//!
//! **PST Accumulator** (`Accumulator`): HalfKA piece-square features (i16 weights).
//! Incremental push/pop via `AccDelta`. Finny table cache (13 king buckets)
//! avoids full refresh when king stays in the same bucket.
//!
//! Filtered threat features (41,272) are recomputed per-eval in `forward.rs`
//! rather than maintained incrementally.
//!
//! Lazily updated: `Network::ensure_updated()` walks back to the nearest
//! accurate ancestor and applies deltas forward. Null moves mark inaccurate
//! (stale accumulator bug prevention — see MEMORY.md).

use crate::bitboard;
use crate::position::Position;
use crate::types::*;

use super::features;
use super::kernels::dispatch;
use super::network::{self, Aligned};
use super::{INPUT_BUCKETS, L1_SIZE};

// ============================================================
// Delta info — stored per ply for lazy incremental updates
// ============================================================

/// Move delta information needed to incrementally update the accumulator.
/// Stored by `Network::push()` BEFORE `make_move()`.
#[derive(Clone, Copy)]
pub struct AccDelta {
    pub mv: Move,
    pub moved_piece: Piece,
    pub captured_piece: Piece,
    /// Pre-move occupancy, captured in `push()` before `make_move()`.
    /// Avoids reconstructing old_occ from new_occ + move info in threat updates.
    pub old_occ: u64,
}

impl Default for AccDelta {
    fn default() -> Self {
        AccDelta {
            mv: Move::NONE,
            moved_piece: Piece::NONE,
            captured_piece: Piece::NONE,
            old_occ: 0,
        }
    }
}

// ============================================================
// Finny table — accumulator refresh cache
// ============================================================
//
// Finny Tables (introduced by Finn Eggers). When lazy incremental updates
// hit a king-bucket boundary (king moved to a different bucket), a full
// refresh from scratch is expensive. Instead, we cache the FT output and
// piece bitboards per (perspective, mirrored, bucket) slot. On refresh, we
// diff cached vs current bitboards and apply only the changed features —
// typically much cheaper than a full recomputation.
//
// Dimensions: [perspective:2][mirrored:2][bucket:INPUT_BUCKETS] (~40KB/thread).
//
// Reference: CPW — NNUE § Accumulator Refresh Table (Finny Tables)

/// Cached accumulator values for one (perspective, mirrored, bucket) slot.
///
/// Stores the FT output and the piece bitboards that produced it.
/// On a bucket-crossing refresh, we diff cached vs current bitboards
/// and apply only the changed features.
#[derive(Clone)]
pub struct FinnyEntry {
    /// Cached FT output values for one perspective.
    pub values: Aligned<[i16; L1_SIZE]>,
    /// Piece bitboards at time of caching. Indexed by raw piece (0..12).
    pub pieces: [u64; 12],
}

impl FinnyEntry {
    fn new() -> Self {
        FinnyEntry {
            // Initialize with FT biases so first use = delta from empty board
            values: network::params().ft_biases.clone(),
            pieces: [0u64; 12],
        }
    }
}

/// Per-thread accumulator refresh table (Finny tables).
///
/// Dimensions: `[perspective: 2][mirrored: 2][bucket: INPUT_BUCKETS]`.
/// ~40KB per thread (36 entries x ~1.1KB each).
pub struct FinnyTable {
    entries: Box<[[[FinnyEntry; INPUT_BUCKETS]; 2]; 2]>,
    /// Network generation at time of initialization (for EvalFile reload detection).
    generation: u64,
}

impl FinnyTable {
    pub fn new() -> Self {
        let entries = Box::new(std::array::from_fn(|_| {
            std::array::from_fn(|_| std::array::from_fn(|_| FinnyEntry::new()))
        }));
        FinnyTable {
            entries,
            generation: network::generation(),
        }
    }

    /// Reset all entries (e.g., after network reload).
    pub fn reset(&mut self) {
        for pov in self.entries.iter_mut() {
            for mirror in pov.iter_mut() {
                for entry in mirror.iter_mut() {
                    *entry = FinnyEntry::new();
                }
            }
        }
        self.generation = network::generation();
    }

    /// Check if the cache is stale due to a network reload.
    fn check_generation(&mut self) {
        let current = network::generation();
        if self.generation != current {
            self.reset();
        }
    }
}

impl Clone for FinnyTable {
    fn clone(&self) -> Self {
        FinnyTable {
            entries: self.entries.clone(),
            generation: self.generation,
        }
    }
}

// ============================================================
// Accumulator
// ============================================================

/// Dual-perspective accumulator (one i16[L1_SIZE] per perspective).
#[derive(Clone)]
pub struct Accumulator {
    /// Feature transformer output: `[white_pov][L1_SIZE]`, `[black_pov][L1_SIZE]`.
    pub values: Aligned<[[i16; L1_SIZE]; 2]>,
    /// Whether each perspective is up-to-date.
    pub accurate: [bool; 2],
    /// Delta from the previous ply (how we got here).
    pub delta: AccDelta,
}

impl Accumulator {
    pub fn new() -> Self {
        Accumulator {
            values: Aligned([[0i16; L1_SIZE]; 2]),
            accurate: [false; 2],
            delta: AccDelta::default(),
        }
    }

    // ============================================================
    // Full refresh — recompute from scratch
    // ============================================================

    /// Recompute the accumulator for `perspective` from scratch using the current position.
    ///
    /// Starts from the FT bias and adds weight columns for every piece on the board.
    /// Reference implementation for tests: plain element-wise loops, no kernel involved,
    /// so it is one of the two sides every incremental-vs-refresh comparison stands on.
    #[allow(dead_code)] // Used in tests only
    pub fn refresh(&mut self, pos: &Position, perspective: Color) {
        let pov = perspective.index();
        let king_sq = pos.king_sq(perspective);
        let params = network::params();

        // Collect all active feature indices first
        let mut feat_indices = [0usize; 32];
        let mut n_features = 0;

        for pc in 0..12u8 {
            let piece_color = if pc & 1 == 0 { Color::White } else { Color::Black };
            let piece_type = match pc >> 1 {
                0 => PieceType::Pawn,
                1 => PieceType::Knight,
                2 => PieceType::Bishop,
                3 => PieceType::Rook,
                4 => PieceType::Queen,
                _ => PieceType::King,
            };

            let mut bb = pos.pieces[pc as usize];
            while bb != 0 {
                let sq = bitboard::pop_lsb(&mut bb);
                feat_indices[n_features] =
                    features::feature_index(piece_color, piece_type, sq, king_sq, perspective);
                n_features += 1;
            }
        }

        let acc = &mut self.values.0[pov];
        *acc = params.ft_biases.0;
        for f in 0..n_features {
            let w = &params.ft_pst_weights.0[feat_indices[f]];
            for i in 0..L1_SIZE {
                acc[i] = acc[i].wrapping_add(w[i]);
            }
        }

        self.accurate[pov] = true;
    }

    // ============================================================
    // Incremental update — apply delta from previous ply
    // ============================================================

    /// Incrementally update this accumulator from `prev` for the given `perspective`.
    ///
    /// Uses the stored delta (move, moved_piece, captured_piece) to compute which
    /// feature indices to add and subtract.
    ///
    /// NOTE: `pos` is the position AFTER all moves up to `self` have been applied.
    /// When called from `ensure_updated()` for intermediate plies in the delta
    /// chain, `pos` is the leaf position (not the intermediate one). This is fine
    /// because `pos` is only used for `king_sq(perspective)`, which is stable as
    /// long as `find_update_source()` verified no king-bucket-crossing occurred.
    pub fn update_from(&mut self, prev: &Accumulator, pos: &Position, perspective: Color) {
        let pov = perspective.index();
        let delta = &self.delta;
        let mv = delta.mv;

        // Null move: no features changed — copy parent values.
        // This handler is reached when `push_null()` set `delta = AccDelta::default()`
        // (mv = Move::NONE). The copy is deferred to here (lazy) rather than done
        // eagerly in `push_null()`, which avoids cloning stale accumulator data when
        // the parent's eval was TT-reused and `ensure_updated()` never ran.
        if mv == Move::NONE {
            self.values.0[pov] = prev.values.0[pov];
            self.accurate[pov] = true;
            return;
        }

        let king_sq = pos.king_sq(perspective);
        let piece = delta.moved_piece;
        let captured = delta.captured_piece;

        // Figure out what piece type ends up on the destination
        let mt = mv.move_type();
        let from = mv.from_sq();
        let to = mv.to_sq();
        let piece_color = piece.color();
        let piece_type = piece.piece_type();

        match mt {
            MT_CASTLING => {
                // King + rook both move: add2_sub2
                let (rook_from, rook_to) = castling_rook_squares(to);
                let sub1 = features::feature_index(piece_color, PieceType::King, from, king_sq, perspective);
                let sub2 = features::feature_index(piece_color, PieceType::Rook, rook_from, king_sq, perspective);
                let add1 = features::feature_index(piece_color, PieceType::King, to, king_sq, perspective);
                let add2 = features::feature_index(piece_color, PieceType::Rook, rook_to, king_sq, perspective);
                self.apply_add2_sub2(prev, pov, add1, add2, sub1, sub2);
            }
            MT_EN_PASSANT => {
                // Pawn moves, captured pawn removed from different square
                let cap_sq = Square(to.0 ^ 8); // captured pawn is one rank behind
                let sub1 = features::feature_index(piece_color, PieceType::Pawn, from, king_sq, perspective);
                let sub2 = features::feature_index(!piece_color, PieceType::Pawn, cap_sq, king_sq, perspective);
                let add1 = features::feature_index(piece_color, PieceType::Pawn, to, king_sq, perspective);
                self.apply_add1_sub2(prev, pov, add1, sub1, sub2);
            }
            MT_PROMOTION => {
                let promo_type = mv.promo_type();
                let sub1 = features::feature_index(piece_color, PieceType::Pawn, from, king_sq, perspective);
                let add1 = features::feature_index(piece_color, promo_type, to, king_sq, perspective);

                if captured != Piece::NONE {
                    // Promotion-capture
                    let sub2 = features::feature_index(captured.color(), captured.piece_type(), to, king_sq, perspective);
                    self.apply_add1_sub2(prev, pov, add1, sub1, sub2);
                } else {
                    // Promotion without capture
                    self.apply_add1_sub1(prev, pov, add1, sub1);
                }
            }
            _ => {
                // Normal move or normal capture
                let sub1 = features::feature_index(piece_color, piece_type, from, king_sq, perspective);
                let add1 = features::feature_index(piece_color, piece_type, to, king_sq, perspective);

                if captured != Piece::NONE {
                    let sub2 = features::feature_index(captured.color(), captured.piece_type(), to, king_sq, perspective);
                    self.apply_add1_sub2(prev, pov, add1, sub1, sub2);
                } else {
                    self.apply_add1_sub1(prev, pov, add1, sub1);
                }
            }
        }

        self.accurate[pov] = true;
    }

    // ============================================================
    // Kernel entry points — the loops live in `nnue::kernels`
    // ============================================================

    /// Normal move: one feature added, one removed.
    #[inline]
    fn apply_add1_sub1(&mut self, prev: &Accumulator, pov: usize, add1: usize, sub1: usize) {
        let vacc = self.values.0[pov].as_mut_ptr();
        let vprev = prev.values.0[pov].as_ptr();
        unsafe { dispatch::acc_add1_sub1(vprev, vacc, add1, sub1) };
    }

    /// Capture / EP: one feature added, two removed.
    #[inline]
    fn apply_add1_sub2(
        &mut self, prev: &Accumulator, pov: usize,
        add1: usize, sub1: usize, sub2: usize,
    ) {
        let vacc = self.values.0[pov].as_mut_ptr();
        let vprev = prev.values.0[pov].as_ptr();
        unsafe { dispatch::acc_add1_sub2(vprev, vacc, add1, sub1, sub2) };
    }

    /// Castling: two features added, two removed.
    #[inline]
    fn apply_add2_sub2(
        &mut self, prev: &Accumulator, pov: usize,
        add1: usize, add2: usize, sub1: usize, sub2: usize,
    ) {
        let vacc = self.values.0[pov].as_mut_ptr();
        let vprev = prev.values.0[pov].as_ptr();
        unsafe { dispatch::acc_add2_sub2(vprev, vacc, add1, add2, sub1, sub2) };
    }

    // ============================================================
    // Finny table refresh — diff-based accumulator recomputation
    // ============================================================

    /// Refresh the accumulator using the Finny table cache.
    ///
    /// Instead of recomputing from scratch (bias + all 32 pieces), this diffs
    /// the cached piece bitboards against the current position and applies only
    /// the changed features. First use initializes the cache (equivalent to a
    /// full refresh). Subsequent uses with the same bucket are much faster.
    pub fn refresh_with_cache(
        &mut self,
        pos: &Position,
        perspective: Color,
        cache: &mut FinnyTable,
    ) {
        cache.check_generation();

        let pov = perspective.index();
        let king_sq = pos.king_sq(perspective);
        let mirrored = (king_sq.file() >= 4) as usize;
        let flip = features::flip_mask(king_sq, perspective);
        let king_idx = (king_sq.0 ^ flip) as usize;
        let bucket = features::KING_BUCKETS[king_idx];
        let entry = &mut cache.entries[pov][mirrored][bucket];

        // Collect adds/subs by diffing cached vs current bitboards
        let mut adds = [0usize; 32];
        let mut subs = [0usize; 32];
        let mut n_adds = 0usize;
        let mut n_subs = 0usize;

        for pc in 0..12u8 {
            let piece_color = if pc & 1 == 0 { Color::White } else { Color::Black };
            let piece_type = match pc >> 1 {
                0 => PieceType::Pawn,
                1 => PieceType::Knight,
                2 => PieceType::Bishop,
                3 => PieceType::Rook,
                4 => PieceType::Queen,
                _ => PieceType::King,
            };

            let cached = entry.pieces[pc as usize];
            let current = pos.pieces[pc as usize];

            let mut to_add = current & !cached;
            while to_add != 0 {
                let sq = bitboard::pop_lsb(&mut to_add);
                debug_assert!(n_adds < 32);
                adds[n_adds] =
                    features::feature_index(piece_color, piece_type, sq, king_sq, perspective);
                n_adds += 1;
            }

            let mut to_sub = cached & !current;
            while to_sub != 0 {
                let sq = bitboard::pop_lsb(&mut to_sub);
                debug_assert!(n_subs < 32);
                subs[n_subs] =
                    features::feature_index(piece_color, piece_type, sq, king_sq, perspective);
                n_subs += 1;
            }
        }

        // Apply delta with SIMD register blocking
        unsafe { dispatch::finny_apply(entry.values.0.as_mut_ptr(), &adds[..n_adds], &subs[..n_subs]) };

        // Update cache bitboards
        entry.pieces = pos.pieces;

        // Copy cached values to current accumulator
        self.values.0[pov] = entry.values.0;
        self.accurate[pov] = true;
    }
}

// ============================================================
// Helpers
// ============================================================

/// Given the king's castling destination, return (rook_from, rook_to).
fn castling_rook_squares(king_to: Square) -> (Square, Square) {
    let rank = king_to.rank();
    if king_to.file() == 6 {
        // King-side: rook h -> f
        (Square::new(7, rank), Square::new(5, rank))
    } else {
        // Queen-side: rook a -> d
        (Square::new(0, rank), Square::new(3, rank))
    }
}

/// Check whether we can incrementally update from an ancestor ply,
/// or if a full refresh is needed.
///
/// Returns `Some(ancestor_index)` if we can delta from that ancestor,
/// or `None` if a full refresh is required (king changed bucket/mirroring).
///
/// SAFETY INVARIANT for null moves: null-move plies have
/// `moved_piece = Piece::NONE` (piece_type = 6), which is NOT `PieceType::King`
/// (= 5). This means the king-bucket-crossing check below correctly skips
/// null-move plies. Do NOT change `AccDelta::default()` to use `Piece::King`
/// or any king piece — it would incorrectly trigger a full refresh.
pub fn find_update_source(
    accumulators: &[Accumulator],
    current: usize,
    perspective: Color,
) -> Option<usize> {
    for i in (0..current).rev() {
        if accumulators[i].accurate[perspective.index()] {
            // Check all intermediate plies for king moves that change bucket
            let mut can_update = true;
            for j in (i + 1)..=current {
                let delta = &accumulators[j].delta;
                if delta.moved_piece.piece_type() == PieceType::King
                    && delta.moved_piece.color() == perspective
                {
                    let from = delta.mv.from_sq();
                    let to = delta.mv.to_sq();
                    if features::needs_refresh(from, to, perspective) {
                        can_update = false;
                        break;
                    }
                }
            }
            if can_update {
                return Some(i);
            } else {
                return None;
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Position;

    #[test]
    fn test_refresh_startpos() {
        let pos = Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let mut acc = Accumulator::new();
        acc.refresh(&pos, Color::White);
        acc.refresh(&pos, Color::Black);
        assert!(acc.accurate[0]);
        assert!(acc.accurate[1]);
        // Both perspectives should produce identical accumulators for startpos
        // (symmetric position → same features for both sides)
        for i in 0..L1_SIZE {
            assert_eq!(acc.values.0[0][i], acc.values.0[1][i]);
        }
    }

    #[test]
    fn test_castling_rook_squares() {
        let (rf, rt) = castling_rook_squares(Square::G1);
        assert_eq!(rf, Square::H1);
        assert_eq!(rt, Square::F1);

        let (rf, rt) = castling_rook_squares(Square::C1);
        assert_eq!(rf, Square::A1);
        assert_eq!(rt, Square::D1);

        let (rf, rt) = castling_rook_squares(Square::G8);
        assert_eq!(rf, Square::H8);
        assert_eq!(rt, Square::F8);

        let (rf, rt) = castling_rook_squares(Square::C8);
        assert_eq!(rf, Square::A8);
        assert_eq!(rt, Square::D8);
    }

    // ============================================================
    // Finny table tests
    // ============================================================

    /// Helper: compare finny refresh vs full refresh for a position.
    fn assert_finny_matches_refresh(fen: &str, label: &str) {
        let pos = Position::from_fen(fen).unwrap();
        let mut cache = FinnyTable::new();

        for &perspective in &[Color::White, Color::Black] {
            let pov = perspective.index();

            // Full refresh (reference)
            let mut acc_ref = Accumulator::new();
            acc_ref.refresh(&pos, perspective);

            // Finny refresh (cold cache)
            let mut acc_finny = Accumulator::new();
            acc_finny.refresh_with_cache(&pos, perspective, &mut cache);

            for i in 0..L1_SIZE {
                assert_eq!(
                    acc_ref.values.0[pov][i], acc_finny.values.0[pov][i],
                    "{label}: {perspective:?} mismatch at {i}: ref={}, finny={}",
                    acc_ref.values.0[pov][i], acc_finny.values.0[pov][i],
                );
            }
        }
    }

    #[test]
    fn test_finny_cold_cache_startpos() {
        assert_finny_matches_refresh(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "startpos",
        );
    }

    #[test]
    fn test_finny_cold_cache_kiwipete() {
        assert_finny_matches_refresh(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "kiwipete",
        );
    }

    #[test]
    fn test_finny_cold_cache_endgame() {
        assert_finny_matches_refresh(
            "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            "KK endgame",
        );
    }

    #[test]
    fn test_finny_cold_cache_asymmetric() {
        assert_finny_matches_refresh(
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "asymmetric position 4",
        );
    }

    #[test]
    fn test_finny_warm_cache() {
        // Use cache with pos1, then refresh with pos2 (1 pawn moved).
        // The warm cache should produce correct results via delta.
        let pos1 = Position::from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ).unwrap();
        let pos2 = Position::from_fen(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        ).unwrap();
        let mut cache = FinnyTable::new();

        // Warm the cache with pos1
        let mut acc1 = Accumulator::new();
        acc1.refresh_with_cache(&pos1, Color::White, &mut cache);

        // Refresh with pos2 (same king square, same bucket, 1 piece diff)
        let mut acc2 = Accumulator::new();
        acc2.refresh_with_cache(&pos2, Color::White, &mut cache);

        // Compare with full refresh
        let mut acc_ref = Accumulator::new();
        acc_ref.refresh(&pos2, Color::White);

        for i in 0..L1_SIZE {
            assert_eq!(
                acc2.values.0[0][i], acc_ref.values.0[0][i],
                "Warm cache: White mismatch at {i}: finny={}, ref={}",
                acc2.values.0[0][i], acc_ref.values.0[0][i],
            );
        }
    }

    #[test]
    fn test_finny_king_midline_crossing() {
        // King on e1 (file 4, mirrored=1) vs king on d1 (file 3, mirrored=0).
        // These use different Finny entries. Both must be correct.
        let pos_e1 = Position::from_fen(
            "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
        ).unwrap();
        let pos_d1 = Position::from_fen(
            "4k3/8/8/8/8/8/8/3K4 w - - 0 1",
        ).unwrap();
        let mut cache = FinnyTable::new();

        // Refresh with king on e1 (mirrored=1)
        let mut acc_e1 = Accumulator::new();
        acc_e1.refresh_with_cache(&pos_e1, Color::White, &mut cache);
        let mut ref_e1 = Accumulator::new();
        ref_e1.refresh(&pos_e1, Color::White);

        for i in 0..L1_SIZE {
            assert_eq!(acc_e1.values.0[0][i], ref_e1.values.0[0][i],
                "King e1: White mismatch at {i}");
        }

        // Refresh with king on d1 (mirrored=0, different entry)
        let mut acc_d1 = Accumulator::new();
        acc_d1.refresh_with_cache(&pos_d1, Color::White, &mut cache);
        let mut ref_d1 = Accumulator::new();
        ref_d1.refresh(&pos_d1, Color::White);

        for i in 0..L1_SIZE {
            assert_eq!(acc_d1.values.0[0][i], ref_d1.values.0[0][i],
                "King d1: White mismatch at {i}");
        }
    }

    #[test]
    fn test_finny_multiple_positions_same_bucket() {
        // Several positions with white king on e1 (same bucket/mirror).
        // Cache should accumulate deltas correctly across multiple uses.
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
        ];
        let mut cache = FinnyTable::new();

        for fen in fens {
            let pos = Position::from_fen(fen).unwrap();

            for &perspective in &[Color::White, Color::Black] {
                let pov = perspective.index();

                let mut acc_finny = Accumulator::new();
                acc_finny.refresh_with_cache(&pos, perspective, &mut cache);

                let mut acc_ref = Accumulator::new();
                acc_ref.refresh(&pos, perspective);

                for i in 0..L1_SIZE {
                    assert_eq!(
                        acc_finny.values.0[pov][i], acc_ref.values.0[pov][i],
                        "{fen}: {perspective:?} mismatch at {i}",
                    );
                }
            }
        }
    }

}
