// Hand-vectorised code: the loop index is the point. It walks several arrays in
// lockstep, steps by a SIMD lane count, and feeds raw pointer arithmetic — an
// iterator would hide the arithmetic these kernels exist to control.
#![allow(clippy::needless_range_loop)]
//! NNUE accumulator: dual-perspective i16 vector with incremental updates.
//!
//! One array per perspective holds the whole feature transformer sum: the bias, the
//! K32Q2 piece-square features, the pawn pairs and the threats (all i8 weights). An
//! update reads and writes it once, the activation reads one array.
//!
//! The Finny table (`FinnyTable`) caches the bias plus the piece-square part per king
//! square and opponent queen; the pawn pairs have their own cache in `threats.rs`.
//!
//! Lazily updated: `Network::ensure_all_updated()` walks back to the nearest
//! accurate ancestor and applies the recorded moves forward. Null moves mark
//! inaccurate (stale accumulator bug prevention — see MEMORY.md).

use crate::bitboard;
use crate::position::Position;
use crate::types::*;

use super::features;
use super::kernels::dispatch;
use super::move_threats::MoveThreats;
use super::network::{self, Aligned, NNUEParams};
use super::{L1_SIZE, PSQ_PER_BLOCK, QUEEN_STATES};

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
    /// Per perspective, whether the move makes the opponent's queen appear or
    /// disappear: the perspective's piece-square block changes.
    pub queen_flip: [bool; 2],
}

impl Default for AccDelta {
    fn default() -> Self {
        AccDelta {
            mv: Move::NONE,
            moved_piece: Piece::NONE,
            captured_piece: Piece::NONE,
            old_occ: 0,
            queen_flip: [false; 2],
        }
    }
}

/// Whether the opponent of `perspective` has a queen in `pieces`.
#[inline(always)]
pub(super) fn opponent_queen(pieces: &[u64; 12], perspective: Color) -> bool {
    pieces[Piece::new(PieceType::Queen, !perspective).index()] != 0
}

// ============================================================
// Finny table — accumulator refresh cache
// ============================================================
//
// Finny Tables (introduced by Finn Eggers). When a move changes the piece-square
// block of a perspective (its king moved, the opponent's queen appeared or
// disappeared), a full refresh from scratch is expensive. Instead, we cache the FT
// output and piece bitboards per (perspective, king square, opponent queen) slot. On
// refresh, we diff cached vs current bitboards and apply only the changed features —
// typically much cheaper than a full recomputation.
//
// Dimensions: [perspective:2][king square:64][opponent queen:2] (~560KB/thread, of
// which a search touches the few king squares it visits).
//
// Reference: CPW — NNUE § Accumulator Refresh Table (Finny Tables)

/// Cached accumulator values for one (perspective, king square, opponent queen) slot.
///
/// Stores the FT output and the piece bitboards that produced it.
/// On a block-changing refresh, we diff cached vs current bitboards
/// and apply only the changed features.
#[derive(Clone)]
pub struct FinnyEntry {
    /// Cached FT output values for one perspective.
    pub values: Aligned<[i16; L1_SIZE]>,
    /// Piece bitboards at time of caching. Indexed by raw piece (0..12).
    pub pieces: [u64; 12],
    /// Piece on each square at time of caching (`Piece::NONE` when empty): the
    /// piece whose feature a refresh removes from a square, read without a search
    /// through the bitboards.
    pub board: [Piece; 64],
}

impl FinnyEntry {
    fn new() -> Self {
        FinnyEntry {
            // Initialize with FT biases so first use = delta from empty board
            values: network::params().ft_biases.clone(),
            pieces: [0u64; 12],
            board: [Piece::NONE; 64],
        }
    }
}

/// Entries in a Finny table: perspective, king square, opponent queen.
const FINNY_ENTRIES: usize = 2 * 64 * QUEEN_STATES;

/// Per-thread accumulator refresh table (Finny tables).
///
/// Dimensions: `[perspective: 2][king square: 64][opponent queen: 2]`, held flat and
/// on the heap: built in place, a nested array of 560 KB would pass through the stack.
pub struct FinnyTable {
    entries: Box<[FinnyEntry]>,
    /// Network generation at time of initialization (for EvalFile reload detection).
    generation: u64,
}

/// Where an entry sits in the Finny table: perspective, king square, opponent queen.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct FinnyKey {
    pov: usize,
    king: usize,
    queen: usize,
}

impl FinnyKey {
    #[inline(always)]
    fn index(self) -> usize {
        debug_assert!(self.pov < 2 && self.king < 64 && self.queen < QUEEN_STATES);
        (self.pov * 64 + self.king) * QUEEN_STATES + self.queen
    }
}

impl FinnyTable {
    pub fn new() -> Self {
        let entries = (0..FINNY_ENTRIES).map(|_| FinnyEntry::new()).collect();
        FinnyTable {
            entries,
            generation: network::generation(),
        }
    }

    /// Reset all entries (e.g., after network reload).
    pub fn reset(&mut self) {
        for entry in self.entries.iter_mut() {
            *entry = FinnyEntry::new();
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

    /// The entry of `perspective` for its king on `king_sq` and the opponent queen of
    /// `pieces`, brought to the position given by `pieces` and `board`: diffed against
    /// what the entry last held, only the changed features applied. Returns where it
    /// sits, for [`FinnyTable::entry`].
    pub(super) fn bring_to(
        &mut self,
        params: &NNUEParams,
        pieces: &[u64; 12],
        board: &[Piece; 64],
        king_sq: Square,
        perspective: Color,
    ) -> FinnyKey {
        self.check_generation();
        #[cfg(feature = "stats")]
        super::threats::tally::add(super::threats::tally::PST_REFRESHES, 1);

        let pov = perspective.index();
        let queen = opponent_queen(pieces, perspective);
        let flip = features::flip_mask(king_sq, perspective);
        let block_base = features::block(king_sq, perspective, queen) * PSQ_PER_BLOCK;
        let key = FinnyKey {
            pov,
            king: king_sq.index(),
            queen: queen as usize,
        };
        let entry = &mut self.entries[key.index()];

        // Collect adds/subs by diffing cached vs current bitboards. One pass over the
        // twelve pairs gives the squares that gain a feature (their current piece is
        // new there) and those that lose one (their cached piece is gone); the pieces
        // come from the two mailboxes. Two loops of one turn per changed square, where
        // twelve pieces with two loops each made 24 loops of unpredictable length.
        let mut add_sqs = 0u64;
        let mut sub_sqs = 0u64;
        for pc in 0..12 {
            let cached = entry.pieces[pc];
            let current = pieces[pc];
            add_sqs |= current & !cached;
            sub_sqs |= cached & !current;
        }

        // Feature of `piece` on `sq` from this perspective: the block, then the plane
        // (the side's pawn to queen, the other side's, both kings), the oriented square.
        let feature = |piece: Piece, sq: usize| -> u32 {
            debug_assert!(piece != Piece::NONE && sq < 64);
            let p = piece.0 as usize;
            let (enemy, pt) = (((p & 1) ^ pov) == 1, p >> 1);
            let plane = if pt == 5 { features::KING_PLANE } else { enemy as usize * 5 + pt };
            let idx = block_base + plane * 64 + (sq ^ flip as usize);
            debug_assert_eq!(
                idx,
                features::feature_index(
                    piece.color(),
                    piece.piece_type(),
                    Square(sq as u8),
                    king_sq,
                    perspective,
                    queen
                )
            );
            idx as u32
        };

        let mut adds = [0u32; 32];
        let mut subs = [0u32; 32];
        let mut n_adds = 0usize;
        let mut n_subs = 0usize;
        while add_sqs != 0 {
            let sq = bitboard::pop_lsb(&mut add_sqs).index();
            debug_assert!(n_adds < 32);
            adds[n_adds] = feature(board[sq], sq);
            n_adds += 1;
        }
        while sub_sqs != 0 {
            let sq = bitboard::pop_lsb(&mut sub_sqs).index();
            debug_assert!(n_subs < 32);
            subs[n_subs] = feature(entry.board[sq], sq);
            n_subs += 1;
        }

        // The piece-square rows are i8 like the threats: the same tiled kernel.
        let values = entry.values.0.as_mut_ptr();
        unsafe {
            dispatch::threat_batch(
                values,
                values,
                params.ft_psq_weights.0.as_ptr(),
                &adds[..n_adds],
                &subs[..n_subs],
            )
        };

        // Update cache bitboards
        entry.pieces = *pieces;
        entry.board = *board;

        key
    }

    /// The entry at `key`.
    #[inline]
    pub(super) fn entry(&self, key: FinnyKey) -> &FinnyEntry {
        &self.entries[key.index()]
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
    /// Feature transformer output: `[white_pov][L1_SIZE]`, `[black_pov][L1_SIZE]`:
    /// the bias, the piece-square features, the pawn pairs and the threats.
    pub values: Aligned<[[i16; L1_SIZE]; 2]>,
    /// Whether each perspective is up-to-date.
    pub accurate: [bool; 2],
    /// Delta from the previous ply (how we got here).
    pub delta: AccDelta,
    /// The attacks the move into this ply created and destroyed, recorded when it
    /// was played.
    pub(super) moved: MoveThreats,
}

/// The piece-square features a move changes for one perspective: at most two added
/// and two removed (castling moves two pieces, a capture takes one more off).
pub(super) struct PsqChanges {
    adds: [u32; 2],
    subs: [u32; 2],
    n_adds: usize,
    n_subs: usize,
}

impl PsqChanges {
    #[inline]
    fn new() -> Self {
        PsqChanges {
            adds: [0; 2],
            subs: [0; 2],
            n_adds: 0,
            n_subs: 0,
        }
    }

    #[inline]
    fn add(&mut self, f: usize) {
        debug_assert!(self.n_adds < 2 && f < super::PSQ_SIZE);
        self.adds[self.n_adds] = f as u32;
        self.n_adds += 1;
    }

    #[inline]
    fn sub(&mut self, f: usize) {
        debug_assert!(self.n_subs < 2 && f < super::PSQ_SIZE);
        self.subs[self.n_subs] = f as u32;
        self.n_subs += 1;
    }

    #[inline]
    pub(super) fn adds(&self) -> &[u32] {
        &self.adds[..self.n_adds]
    }

    #[inline]
    pub(super) fn subs(&self) -> &[u32] {
        &self.subs[..self.n_subs]
    }
}

impl Accumulator {
    pub fn new() -> Self {
        Accumulator {
            values: Aligned([[0i16; L1_SIZE]; 2]),
            accurate: [false; 2],
            delta: AccDelta::default(),
            moved: MoveThreats::new(),
        }
    }

    // ============================================================
    // Full refresh — recompute from scratch
    // ============================================================

    /// Recompute the piece-square part for `perspective` from scratch using the current
    /// position: the FT bias plus a weight row for every piece on the board, no pawn
    /// pairs, no threats.
    ///
    /// Reference implementation for tests: plain element-wise loops, no kernel involved,
    /// so it is one of the two sides every incremental-vs-refresh comparison stands on.
    /// The rows are in logical order and the accumulator in the tier's (the order
    /// `packus` undoes): each column takes the weight of the neuron it holds.
    #[cfg(test)]
    pub fn refresh(&mut self, pos: &Position, perspective: Color) {
        self.refresh_with_params(network::params(), pos, perspective);
    }

    /// [`Accumulator::refresh`] with the weights `params`.
    #[cfg(any(test, debug_assertions))]
    pub fn refresh_with_params(&mut self, params: &NNUEParams, pos: &Position, perspective: Color) {
        let pov = perspective.index();
        let king_sq = pos.king_sq(perspective);
        let queen = opponent_queen(&pos.pieces, perspective);

        let column = network::memory_to_logical();
        let acc = &mut self.values.0[pov];
        *acc = params.ft_biases.0;
        for pc in 0..12u8 {
            let piece = Piece(pc);
            let mut bb = pos.pieces[pc as usize];
            while bb != 0 {
                let sq = bitboard::pop_lsb(&mut bb);
                let f = features::feature_index(
                    piece.color(),
                    piece.piece_type(),
                    sq,
                    king_sq,
                    perspective,
                    queen,
                );
                let w = &params.ft_psq_weights.0[f].w;
                for i in 0..L1_SIZE {
                    acc[i] = acc[i].wrapping_add(w[column[i] as usize] as i16);
                }
            }
        }
        self.accurate[pov] = true;
    }

    // ============================================================
    // Incremental update — the features the move into this ply changed
    // ============================================================

    /// The piece-square features the move into this ply changed for `perspective`.
    ///
    /// NOTE: `pos` is the position the update ends at, not the one at this ply. It
    /// only gives the king square of `perspective` and whether its opponent has a
    /// queen, which name the same block all along a chain: a move that changes them is
    /// a refresh.
    pub(super) fn psq_changes(&self, pos: &Position, perspective: Color) -> PsqChanges {
        let delta = &self.delta;
        let mv = delta.mv;
        debug_assert!(mv != Move::NONE, "psq_changes: null move");
        let king_sq = pos.king_sq(perspective);
        let queen = opponent_queen(&pos.pieces, perspective);
        let piece = delta.moved_piece;
        let captured = delta.captured_piece;
        let mt = mv.move_type();
        let from = mv.from_sq();
        let to = mv.to_sq();
        let piece_color = piece.color();
        let piece_type = piece.piece_type();
        let feature = |color: Color, pt: PieceType, sq: Square| {
            features::feature_index(color, pt, sq, king_sq, perspective, queen)
        };

        let mut ch = PsqChanges::new();
        match mt {
            MT_CASTLING => {
                // Encoded as king-captures-rook: `to` is the rook origin.
                ch.sub(feature(piece_color, PieceType::King, from));
                ch.sub(feature(piece_color, PieceType::Rook, to));
                ch.add(feature(piece_color, PieceType::King, mv.castle_king_to()));
                ch.add(feature(piece_color, PieceType::Rook, mv.castle_rook_to()));
            }
            MT_EN_PASSANT => {
                // Pawn moves, captured pawn removed from a different square
                let cap_sq = Square(to.0 ^ 8); // captured pawn is one rank behind
                ch.sub(feature(piece_color, PieceType::Pawn, from));
                ch.sub(feature(!piece_color, PieceType::Pawn, cap_sq));
                ch.add(feature(piece_color, PieceType::Pawn, to));
            }
            MT_PROMOTION => {
                ch.sub(feature(piece_color, PieceType::Pawn, from));
                if captured != Piece::NONE {
                    ch.sub(feature(captured.color(), captured.piece_type(), to));
                }
                ch.add(feature(piece_color, mv.promo_type(), to));
            }
            _ => {
                // Normal move or normal capture
                ch.sub(feature(piece_color, piece_type, from));
                if captured != Piece::NONE {
                    ch.sub(feature(captured.color(), captured.piece_type(), to));
                }
                ch.add(feature(piece_color, piece_type, to));
            }
        }
        ch
    }

    /// Bring the piece-square part of `perspective` to `pos` through `cache`, without
    /// the pawn pairs and threats: the Finny table on its own, for its tests.
    #[cfg(test)]
    pub fn refresh_with_cache(
        &mut self,
        params: &NNUEParams,
        pos: &Position,
        perspective: Color,
        cache: &mut FinnyTable,
    ) {
        let pov = perspective.index();
        let key = cache.bring_to(params, &pos.pieces, &pos.board, pos.king_sq(perspective), perspective);
        let entry = cache.entry(key);
        self.values.0[pov] = entry.values.0;
        self.accurate[pov] = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Position;

    #[test]
    fn test_refresh_startpos() {
        let pos =
            Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
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
    fn test_castling_destinations() {
        let oo = Move::new_with_type(Square::E1, Square::H1, MT_CASTLING);
        assert_eq!(oo.castle_king_to(), Square::G1);
        assert_eq!(oo.castle_rook_to(), Square::F1);

        let ooo = Move::new_with_type(Square::E1, Square::A1, MT_CASTLING);
        assert_eq!(ooo.castle_king_to(), Square::C1);
        assert_eq!(ooo.castle_rook_to(), Square::D1);
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
            acc_finny.refresh_with_cache(network::params(), &pos, perspective, &mut cache);

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
        assert_finny_matches_refresh("4k3/8/8/8/8/8/8/4K3 w - - 0 1", "KK endgame");
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
        let pos1 =
            Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let pos2 = Position::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
            .unwrap();
        let mut cache = FinnyTable::new();

        // Warm the cache with pos1
        let mut acc1 = Accumulator::new();
        acc1.refresh_with_cache(network::params(), &pos1, Color::White, &mut cache);

        // Refresh with pos2 (same king square, same bucket, 1 piece diff)
        let mut acc2 = Accumulator::new();
        acc2.refresh_with_cache(network::params(), &pos2, Color::White, &mut cache);

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
        // King on e1 (mirrored) vs king on d1 (not mirrored).
        // These use different Finny entries. Both must be correct.
        let pos_e1 = Position::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let pos_d1 = Position::from_fen("4k3/8/8/8/8/8/8/3K4 w - - 0 1").unwrap();
        let mut cache = FinnyTable::new();

        // Refresh with king on e1 (mirrored=1)
        let mut acc_e1 = Accumulator::new();
        acc_e1.refresh_with_cache(network::params(), &pos_e1, Color::White, &mut cache);
        let mut ref_e1 = Accumulator::new();
        ref_e1.refresh(&pos_e1, Color::White);

        for i in 0..L1_SIZE {
            assert_eq!(
                acc_e1.values.0[0][i], ref_e1.values.0[0][i],
                "King e1: White mismatch at {i}"
            );
        }

        // Refresh with king on d1 (mirrored=0, different entry)
        let mut acc_d1 = Accumulator::new();
        acc_d1.refresh_with_cache(network::params(), &pos_d1, Color::White, &mut cache);
        let mut ref_d1 = Accumulator::new();
        ref_d1.refresh(&pos_d1, Color::White);

        for i in 0..L1_SIZE {
            assert_eq!(
                acc_d1.values.0[0][i], ref_d1.values.0[0][i],
                "King d1: White mismatch at {i}"
            );
        }
    }

    #[test]
    fn test_finny_multiple_positions_same_bucket() {
        // Several positions with white king on e1 (same king square).
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
                acc_finny.refresh_with_cache(network::params(), &pos, perspective, &mut cache);

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
