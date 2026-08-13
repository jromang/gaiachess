//! [Move ordering](https://www.chessprogramming.org/Move_Ordering) via staged generation.
//!
//! Yields moves one at a time in priority order:
//! TT move → good captures (MVV + capture history) → killer moves → countermove
//! → quiets (butterfly + continuation history) → bad captures.
//! QSearch mode yields only the TT move and good captures.

use crate::bitboard::{pawn_attacks, knight_attacks, bishop_attacks, rook_attacks, queen_attacks};
use crate::eval;
use crate::history::{ButterflyHistory, CaptureHistory, ContinuationHistory, PawnHistory};
use crate::movegen;
use crate::position::Position;
use crate::see;
use crate::threads::{StackEntry, SS_OFFSET};
use crate::tt::TT;
use crate::tune;
use crate::types::{ArrayBuf, Color, Move, ScoredMove, MAX_MOVES, Piece, PieceType, MT_EN_PASSANT, MT_PROMOTION, PIECE_VALUE, CASTLING_RIGHTS_MASK, Square};
use crate::zobrist::ZOBRIST;

/// Check if a move is quiet (no capture, no promotion, no EP).
/// Used to validate killer/countermove moves.
#[inline]
fn is_quiet(pos: &Position, m: Move) -> bool {
    let mt = m.move_type();
    mt != MT_PROMOTION && mt != MT_EN_PASSANT && pos.board[m.to_sq().index()] == Piece::NONE
}

/// Compute the Zobrist key of the child position after a quiet move,
/// without performing make_move. Used for TT look-ahead probing.
/// Handles EP clearing/setting and castling rights updates.
#[inline]
fn child_key(pos: &Position, m: Move, piece: Piece) -> u64 {
    let from = m.from_sq();
    let to = m.to_sq();

    let mut key = pos.key
        ^ ZOBRIST.pieces[piece.index()][from.index()]
        ^ ZOBRIST.pieces[piece.index()][to.index()]
        ^ ZOBRIST.side;

    // Clear current EP (always cleared after any move)
    if pos.ep_square != Square::NONE {
        key ^= ZOBRIST.ep[pos.ep_square.file() as usize];
    }

    // Set new EP for double pawn push
    if piece.piece_type() == PieceType::Pawn {
        let rank_diff = (to.rank() as i32 - from.rank() as i32).unsigned_abs();
        if rank_diff == 2 {
            key ^= ZOBRIST.ep[to.file() as usize];
        }
    }

    // Update castling rights if king or rook moves from/to relevant squares
    let old_rights = pos.castling_rights;
    let new_rights = old_rights & CASTLING_RIGHTS_MASK[from.index()] & CASTLING_RIGHTS_MASK[to.index()];
    if old_rights != new_rights {
        key ^= ZOBRIST.castling[old_rights as usize] ^ ZOBRIST.castling[new_rights as usize];
    }

    key
}

/// Get the captured piece type for a move (handles EP).
#[inline]
pub fn get_captured_pt(pos: &Position, m: Move) -> PieceType {
    debug_assert!(m.to_sq().0 < 64, "get_captured_pt: to sq OOB {}", m.to_sq().0);
    if m.move_type() == MT_EN_PASSANT {
        PieceType::Pawn
    } else {
        let pc = pos.board[m.to_sq().index()];
        // Note: promotions without captures have pc == NONE here, which is expected
        debug_assert!(pc != Piece::NONE || m.move_type() == MT_PROMOTION,
            "get_captured_pt: no piece on {} for {}", m.to_sq().0, m.to_uci());
        pc.piece_type()
    }
}

// Staged move generation for move ordering (CPW: Move Ordering).
//
// Stages in priority order:
//   1. TT Move       — best move from transposition table (CPW: Hash Move)
//   2. Gen Captures   — score all captures by MVV + capture history
//   3. Good Captures  — yield captures with SEE >= 0 (CPW: MVV-LVA)
//   4. Killer 1       — first killer for this ply (CPW: Killer Heuristic)
//   5. Killer 2       — second killer for this ply
//   6. Countermove    — response to opponent's last move (CPW: Countermove Heuristic)
//   7. Gen Quiets     — score quiets by butterfly + continuation history
//   8. Quiets         — yield quiet moves (duplicates compacted, partially sorted)
//   9. Bad Captures   — captures with SEE < 0, stored LIFO from array end
//  10. Done
//
// Reference: CPW — Move Ordering
#[derive(Clone, Copy, PartialEq)]
enum Stage {
    TTMove,
    GenerateCaptures,
    GoodCaptures,
    Killer1,
    Killer2,
    Countermove,
    GenerateQuiets,
    Quiets,
    BadCaptures,
    Done,
}

/// Move picker: yields moves one at a time in priority order.
///
/// Uses a packed `ScoredMove` buffer: captures and quiets share `entries[]` (non-overlapping stages).
/// Bad captures are stored at the end of the same array (reverse LIFO region).
pub struct MovePicker {
    stage: Stage,
    tt_move: Move,
    killers: [Move; 2],
    countermove: Move,
    /// Packed move+score buffer. Captures first, then reused for quiets.
    /// Bad captures stored at entries[bad_start..MAX_MOVES], growing downward.
    entries: ArrayBuf<ScoredMove, MAX_MOVES>,
    /// Number of valid entries in the current region (captures or quiets).
    count: usize,
    /// Bad captures region: entries[bad_start..MAX_MOVES]. Starts at MAX_MOVES, grows down.
    bad_start: usize,
    /// Current picking index within entries[index..count].
    index: usize,
    /// Current picking index within bad captures region.
    bad_cap_index: usize,
    /// Search depth (used for quiet sort threshold).
    depth: i32,
}

impl MovePicker {
    /// Create a move picker for a full search node.
    pub fn new(tt_move: Move, killers: [Move; 2], countermove: Move, depth: i32) -> MovePicker {
        MovePicker {
            stage: if tt_move != Move::NONE { Stage::TTMove } else { Stage::GenerateCaptures },
            tt_move,
            killers,
            countermove,
            entries: ArrayBuf::new(),
            count: 0,
            bad_start: MAX_MOVES,
            index: 0,
            bad_cap_index: MAX_MOVES,
            depth,
        }
    }

    /// Create a move picker for quiescence search (captures only).
    pub fn new_qsearch(tt_move: Move) -> MovePicker {
        MovePicker {
            stage: if tt_move != Move::NONE { Stage::TTMove } else { Stage::GenerateCaptures },
            tt_move,
            killers: [Move::NONE; 2],
            countermove: Move::NONE,
            entries: ArrayBuf::new(),
            count: 0,
            bad_start: MAX_MOVES,
            index: 0,
            bad_cap_index: MAX_MOVES,
            depth: 0,
        }
    }

    /// Get the next move. Returns `Move::NONE` when exhausted.
    /// Moves are yielded in priority order but legality is NOT checked here.
    ///
    /// `QSEARCH`: const generic — when true, the compiler eliminates all quiet/killer/bad-capture
    /// stages, generating a specialized qsearch version with no dead code.
    pub fn next<const QSEARCH: bool>(
        &mut self,
        pos: &Position,
        history: &ButterflyHistory,
        cap_history: &CaptureHistory,
        pawn_history: &PawnHistory,
        tt: &TT,
        ply: usize,
        stack: &[StackEntry],
        stm: Color,
    ) -> Move {
        loop {
            match self.stage {
                Stage::TTMove => {
                    self.stage = Stage::GenerateCaptures;
                    if self.tt_move.is_ok()
                        && movegen::is_pseudo_legal(pos, self.tt_move)
                    {
                        return self.tt_move;
                    }
                }

                Stage::GenerateCaptures => {
                    let gen_count = movegen::generate_captures(pos, &mut self.entries);
                    // Score captures in-place: MVV * 16 + capture history
                    for i in 0..gen_count {
                        let m = self.entries[i].mv;
                        let mt = m.move_type();
                        self.entries[i].score = if mt == MT_EN_PASSANT {
                            let piece = pos.board[m.from_sq().index()];
                            debug_assert!(piece != Piece::NONE, "movepick: EP no piece on from {}", m.from_sq().0);
                            cap_history.get(piece, m.to_sq(), PieceType::Pawn)
                        } else if mt == MT_PROMOTION {
                            let victim_pc = pos.board[m.to_sq().index()];
                            let victim_val = if victim_pc != Piece::NONE {
                                PIECE_VALUE[victim_pc.piece_type() as usize]
                            } else {
                                0
                            };
                            PIECE_VALUE[m.promo_type() as usize] * tune::MVV_MULTIPLIER() + victim_val
                        } else {
                            let victim_pc = pos.board[m.to_sq().index()];
                            if victim_pc == Piece::NONE {
                                0
                            } else {
                                let piece = pos.board[m.from_sq().index()];
                                debug_assert!(piece != Piece::NONE,
                                    "movepick: no piece on from {} for capture {}", m.from_sq().0, m.to_uci());
                                let captured_pt = victim_pc.piece_type();
                                PIECE_VALUE[captured_pt as usize] * tune::MVV_MULTIPLIER()
                                    + cap_history.get(piece, m.to_sq(), captured_pt)
                            }
                        };
                    }
                    self.count = gen_count;
                    self.index = 0;
                    self.stage = Stage::GoodCaptures;
                }

                Stage::GoodCaptures => {
                    if let Some(m) = self.pick_best() {
                        if m == self.tt_move {
                            continue; // already yielded
                        }
                        if see::see(pos, m, 0) {
                            return m;
                        }
                        // Bad capture: store at end of array (LIFO)
                        debug_assert!(self.bad_start > self.count,
                            "movepick: bad captures overflow into active region");
                        self.bad_start -= 1;
                        self.entries[self.bad_start] = ScoredMove { score: 0, mv: m };
                        continue;
                    }
                    if QSEARCH {
                        self.stage = Stage::Done;
                    } else {
                        self.stage = Stage::Killer1;
                    }
                }

                // --- These stages are eliminated by the compiler when QSEARCH=true ---
                Stage::Killer1 => {
                    self.stage = Stage::Killer2;
                    if !QSEARCH {
                        let k = self.killers[0];
                        if k != Move::NONE
                            && k != self.tt_move
                            && is_quiet(pos, k)
                            && movegen::is_pseudo_legal(pos, k)
                        {
                            return k;
                        }
                    }
                }

                Stage::Killer2 => {
                    self.stage = Stage::Countermove;
                    if !QSEARCH {
                        let k = self.killers[1];
                        if k != Move::NONE
                            && k != self.tt_move
                            && k != self.killers[0]
                            && is_quiet(pos, k)
                            && movegen::is_pseudo_legal(pos, k)
                        {
                            return k;
                        }
                    }
                }

                Stage::Countermove => {
                    self.stage = Stage::GenerateQuiets;
                    if !QSEARCH {
                        let cm = self.countermove;
                        if cm != Move::NONE
                            && cm != self.tt_move
                            && cm != self.killers[0]
                            && cm != self.killers[1]
                            && is_quiet(pos, cm)
                            && movegen::is_pseudo_legal(pos, cm)
                        {
                            return cm;
                        }
                    }
                }

                Stage::GenerateQuiets => {
                    if QSEARCH {
                        self.stage = Stage::Done;
                        continue;
                    }
                    // Generate quiets directly into entries, then score + compact in-place
                    let gen_count = movegen::generate_quiets(pos, &mut self.entries);

                    // Precompute the direct-check squares for each piece type
                    // (movepicker check bonus for quiet checking moves)
                    let their_king = pos.king_sq(stm.flip());
                    let occ = pos.occupied();
                    let check_sqs: [u64; 6] = [
                        pawn_attacks(their_king, stm.flip()),   // PieceType::Pawn = 0
                        knight_attacks(their_king),              // PieceType::Knight = 1
                        bishop_attacks(their_king, occ),         // PieceType::Bishop = 2
                        rook_attacks(their_king, occ),           // PieceType::Rook = 3
                        queen_attacks(their_king, occ),          // PieceType::Queen = 4
                        0u64,                                    // PieceType::King = 5 (no direct check)
                    ];

                    // Learned look-ahead: prefetch TT entries for child positions
                    // before the scoring loop (software pipelining: prefetch all,
                    // then probe during scoring when cache lines are warm).
                    let do_lookahead = self.depth >= tune::LOOKAHEAD_MIN_DEPTH();
                    if do_lookahead {
                        for i in 0..gen_count {
                            let m = self.entries[i].mv;
                            let piece = pos.board[m.from_sq().index()];
                            if piece != Piece::NONE {
                                tt.prefetch(child_key(pos, m, piece));
                            }
                        }
                    }

                    // Score quiets and compact: remove duplicates already yielded
                    let mut j = 0;
                    for i in 0..gen_count {
                        let m = self.entries[i].mv;
                        if m == self.tt_move
                            || m == self.killers[0]
                            || m == self.killers[1]
                            || m == self.countermove
                        {
                            continue; // already yielded in earlier stage
                        }
                        let piece = pos.board[m.from_sq().index()];
                        debug_assert!(piece != Piece::NONE,
                            "movepick: no piece on from {} for quiet {}", m.from_sq().0, m.to_uci());
                        let to = m.to_sq();
                        let mut score = history.get(stm, m, pos.threats);
                        score += pawn_history.get(pos.pawn_key, piece, to);
                        let base = ply + SS_OFFSET;
                        for offset in [1, 2, 4, 6] {
                            score += ContinuationHistory::get(
                                stack[base - offset].conthist_ptr as *const _,
                                piece,
                                to,
                            );
                        }
                        // Bonus for moves that give direct check
                        let pt_idx = piece.piece_type() as usize;
                        debug_assert!(pt_idx < 6, "movepick: piece_type index OOB {}", pt_idx);
                        if check_sqs[pt_idx] & to.bb() != 0 {
                            score += tune::MOVEPICK_CHECK_BONUS();
                        }
                        // Learned look-ahead: probe TT for child position
                        // (NeurIPS 2024, ArXiv 2406.00877 — use cached search results
                        // from previous ID iterations to predict good quiet moves)
                        if do_lookahead {
                            let ckey = child_key(pos, m, piece);
                            if let Some(hit) = tt.probe(ckey, ply as i32 + 1, 0) {
                                // Use static eval (more stable than search score for ordering).
                                // Eval is from child's STM (= opponent) → negate for our perspective.
                                let raw = if hit.eval != crate::types::SCORE_NONE { hit.eval } else { hit.score };
                                score += -raw * tune::LOOKAHEAD_BONUS_MUL() / tune::LOOKAHEAD_BONUS_DIV();
                            }
                        }
                        // PST tiebreaker when all history is zero
                        if score == 0 {
                            let from = m.from_sq();
                            score = eval::MG_TABLE[piece.index()][to.index()]
                                  - eval::MG_TABLE[piece.index()][from.index()];
                        }
                        self.entries[j] = ScoredMove { score, mv: m };
                        j += 1;
                    }
                    self.count = j;
                    self.index = 0;
                    Self::partial_insertion_sort(&mut self.entries, 0, j, -3500 * self.depth);
                    self.stage = Stage::Quiets;
                }

                Stage::Quiets => {
                    if QSEARCH {
                        self.stage = Stage::Done;
                        continue;
                    }
                    if self.index < self.count {
                        let m = self.entries[self.index].mv;
                        self.index += 1;
                        return m;
                    }
                    self.stage = Stage::BadCaptures;
                }

                Stage::BadCaptures => {
                    if QSEARCH {
                        self.stage = Stage::Done;
                        continue;
                    }
                    // Iterate bad captures from MAX_MOVES-1 downward to bad_start
                    // (preserves insertion order: first stored at MAX_MOVES-1, yielded first)
                    if self.bad_cap_index > self.bad_start {
                        self.bad_cap_index -= 1;
                        let m = self.entries[self.bad_cap_index].mv;
                        if m == self.tt_move {
                            continue;
                        }
                        return m;
                    }
                    self.stage = Stage::Done;
                }

                Stage::Done => {
                    return Move::NONE;
                }
            }
        }
    }

    /// Selection sort: find the move with the highest score
    /// from `self.index` onwards, swap it to `self.index`, advance index, return it.
    fn pick_best(&mut self) -> Option<Move> {
        if self.index >= self.count {
            return None;
        }
        let mut best_idx = self.index;
        let mut best_score = self.entries[self.index].score;
        for i in (self.index + 1)..self.count {
            if self.entries[i].score > best_score {
                best_score = self.entries[i].score;
                best_idx = i;
            }
        }
        self.entries.swap(self.index, best_idx);
        let m = self.entries[self.index].mv;
        self.index += 1;
        Some(m)
    }

    /// Partial insertion sort: sort moves with score >= limit
    /// into descending order at the front. Moves below limit remain unsorted after.
    fn partial_insertion_sort(entries: &mut ArrayBuf<ScoredMove, MAX_MOVES>, start: usize, end: usize, limit: i32) {
        let mut sorted_end = start;
        for p in (start + 1)..end {
            if entries[p].score >= limit {
                let tmp = entries[p];
                sorted_end += 1;
                entries[p] = entries[sorted_end];
                let mut q = sorted_end;
                while q > start && entries[q - 1].score < tmp.score {
                    entries[q] = entries[q - 1];
                    q -= 1;
                }
                entries[q] = tmp;
            }
        }
    }
}
