//! [Move ordering](https://www.chessprogramming.org/Move_Ordering) via staged generation.
//!
//! Yields moves one at a time in priority order:
//! TT move → good captures (MVV + capture history) → quiets (butterfly +
//! continuation history, with a bonus for the killers and the countermove) → bad
//! captures → bad quiets (score below a threshold, held back from the quiets).
//! QSearch mode yields only the TT move and good captures.

use crate::bitboard::{both_attacks, knight_attacks, pawn_attacks};
use crate::eval;
use crate::history::{ButterflyHistory, CaptureHistory, ContinuationHistory, PawnHistory};
use crate::movegen;
use crate::position::Position;
use crate::see;
use crate::threads::{SS_OFFSET, StackEntry};
use crate::tt::TT;
use crate::tune;
use crate::types::{
    ArrayBuf, Color, MAX_MOVES, MT_CASTLING, MT_EN_PASSANT, MT_PROMOTION, Move, PIECE_VALUE, Piece,
    PieceType, ScoredMove, Square,
};
use crate::zobrist::ZOBRIST;

/// Whether a move is quiet: what the quiet stage of generation produces.
///
/// One definition for the picker and the search. A promotion to a queen is not quiet,
/// nor a capture, nor en passant; an under-promotion by a push is, as the generator
/// has it — the search used to treat those as captures (capture history on the
/// queen's slot, no history or futility pruning, capture LMR) while the generator and
/// the quiet stages handled them as quiets. Castling is encoded as king-captures-own-
/// rook, so the destination is occupied: tested first.
#[inline(always)]
pub(crate) fn is_quiet(pos: &Position, m: Move) -> bool {
    match m.move_type() {
        MT_CASTLING => true,
        MT_EN_PASSANT => false,
        MT_PROMOTION => {
            m.promo_type() != PieceType::Queen && pos.board[m.to_sq().index()] == Piece::NONE
        }
        _ => pos.board[m.to_sq().index()] == Piece::NONE,
    }
}

/// Weights (1/1024) of the butterfly, the pawn history and the continuation histories
/// 1, 2, 4 and 6 plies back in the quiet score, for a node of depth `depth`. The first
/// four depend on the depth range (1-2, 3-4, 5-7, 8 and more; a quiescence evasion
/// counts as shallow).
#[inline]
fn quiet_weights(depth: i32) -> [i32; 6] {
    let (main, pawn, cont1, cont2) = match depth {
        ..=2 => (tune::QUIET_W_MAIN_D1(), tune::QUIET_W_PAWN_D1(), tune::QUIET_W_CONT1_D1(), tune::QUIET_W_CONT2_D1()),
        3..=4 => (tune::QUIET_W_MAIN_D3(), tune::QUIET_W_PAWN_D3(), tune::QUIET_W_CONT1_D3(), tune::QUIET_W_CONT2_D3()),
        5..=7 => (tune::QUIET_W_MAIN_D5(), tune::QUIET_W_PAWN_D5(), tune::QUIET_W_CONT1_D5(), tune::QUIET_W_CONT2_D5()),
        _ => (tune::QUIET_W_MAIN_D8(), tune::QUIET_W_PAWN_D8(), tune::QUIET_W_CONT1_D8(), tune::QUIET_W_CONT2_D8()),
    };
    [main, pawn, cont1, cont2, tune::QUIET_W_CONT4(), tune::QUIET_W_CONT6()]
}

/// Compute the Zobrist key of the child position after a move, without performing
/// make_move: the look-ahead probes the TT with it for the quiets, and the move loops
/// prefetch the child's cluster with it for every move. Exact for every move —
/// captures, en passant, promotions, castling, en passant clearing/setting and
/// castling rights included.
#[inline]
pub(crate) fn child_key(pos: &Position, m: Move, piece: Piece) -> u64 {
    let from = m.from_sq();
    let to = m.to_sq();
    let mt = m.move_type();

    let dest = if mt == MT_CASTLING {
        m.castle_king_to()
    } else {
        to
    };
    // What lands on the arrival square: the piece itself, or what a pawn became.
    let landed = if mt == MT_PROMOTION {
        Piece::new(m.promo_type(), piece.color())
    } else {
        piece
    };
    let mut key = pos.key
        ^ ZOBRIST.pieces[piece.index()][from.index()]
        ^ ZOBRIST.pieces[landed.index()][dest.index()]
        ^ ZOBRIST.side;

    // The victim leaves: on the arrival square, or one rank behind it en passant.
    // Castling is encoded as king-takes-rook, and that rook is no victim.
    if mt == MT_EN_PASSANT {
        let victim_sq = Square(to.0 ^ 8);
        let victim = pos.board[victim_sq.index()];
        debug_assert!(victim.piece_type() == PieceType::Pawn);
        key ^= ZOBRIST.pieces[victim.index()][victim_sq.index()];
    } else if mt != MT_CASTLING {
        let victim = pos.board[to.index()];
        if victim != Piece::NONE {
            key ^= ZOBRIST.pieces[victim.index()][to.index()];
        }
    }

    // Castling moves the rook too, from `to` (its origin) to the d/f file. The four
    // XORs stay correct when king and rook squares overlap: equal terms cancel.
    if m.move_type() == MT_CASTLING {
        let rook = Piece::new(PieceType::Rook, piece.color());
        key ^= ZOBRIST.pieces[rook.index()][to.index()]
            ^ ZOBRIST.pieces[rook.index()][m.castle_rook_to().index()];
    }

    // Clear current EP (always cleared after any move)
    if pos.ep_square != Square::NONE {
        key ^= ZOBRIST.ep[pos.ep_square.file() as usize];
    }

    // Set new EP for a double pawn push, but only where an enemy pawn stands ready to
    // answer it: that is the condition make_move hashes on, and a key that hashes the
    // file unconditionally names a position that never occurs, so the probe finds
    // nothing.
    if piece.piece_type() == PieceType::Pawn
        && (to.rank() as i32 - from.rank() as i32).unsigned_abs() == 2
    {
        let ep = Square((from.0 + to.0) / 2);
        let us = piece.color();
        if pawn_attacks(ep, us) & pos.piece_type_bb(PieceType::Pawn, !us) != 0 {
            key ^= ZOBRIST.ep[ep.file() as usize];
        }
    }

    // Update castling rights if king or rook moves from/to relevant squares
    let old_rights = pos.castling_rights;
    let new_rights =
        old_rights & pos.castling_rights_mask[from.index()] & pos.castling_rights_mask[to.index()];
    if old_rights != new_rights {
        key ^= ZOBRIST.castling[old_rights as usize] ^ ZOBRIST.castling[new_rights as usize];
    }

    key
}

/// Get the captured piece type for a move (handles EP).
#[inline]
pub fn get_captured_pt(pos: &Position, m: Move) -> PieceType {
    debug_assert!(
        m.to_sq().0 < 64,
        "get_captured_pt: to sq OOB {}",
        m.to_sq().0
    );
    if m.move_type() == MT_EN_PASSANT {
        PieceType::Pawn
    } else {
        let pc = pos.board[m.to_sq().index()];
        // Note: promotions without captures have pc == NONE here, which is expected
        debug_assert!(
            pc != Piece::NONE || m.move_type() == MT_PROMOTION,
            "get_captured_pt: no piece on {} for {}",
            m.to_sq().0,
            m.to_uci()
        );
        pc.piece_type()
    }
}

// Staged move generation for move ordering (CPW: Move Ordering).
//
// Stages in priority order:
//   1. TT Move       — best move from transposition table (CPW: Hash Move)
//   2. Gen Captures   — score all captures by MVV + capture history
//   3. Good Captures  — yield captures passing SEE >= GOOD_CAP_BASE - caphist /
//                       GOOD_CAP_HIST_DIV (CPW: MVV-LVA)
//   4. Gen Quiets     — score quiets by butterfly + continuation history; the two
//                       killers (CPW: Killer Heuristic) and the countermove (CPW:
//                       Countermove Heuristic) get a score bonus instead of a stage
//   5. Quiets         — yield quiet moves (TT move removed, partially sorted);
//                       those scored below the bad-quiet threshold are held back
//   6. Bad Captures   — captures with SEE < 0, stored LIFO from array end
//   7. Bad Quiets     — the held-back quiets, in the order they were met
//   8. Done
//
// Reference: CPW — Move Ordering
#[derive(Clone, Copy, PartialEq)]
enum Stage {
    TTMove,
    GenerateCaptures,
    GoodCaptures,
    GenerateQuiets,
    Quiets,
    /// Quiet skip armed before the quiets were generated: the killers and the
    /// countermove are still tried, each checked for pseudo-legality.
    SkippedRefutations,
    /// Quiet skip armed during the quiet stage: the refutations left in the
    /// unyielded tail are still tried, the other quiets are dropped.
    QuietRefutations,
    BadCaptures,
    BadQuiets,
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
    /// Next refutation slot (killer 1, killer 2, countermove) of SkippedRefutations.
    ref_slot: u8,
    /// Quiets held back for the bad-quiet stage are compacted in place at
    /// entries[0..bad_quiet_end] while the quiet stage walks past them (the write
    /// index never overtakes the read index), then yielded from bad_quiet_index.
    bad_quiet_end: usize,
    bad_quiet_index: usize,
    /// Whether this node holds bad quiets back (never in check).
    hold_bad_quiets: bool,
    /// When armed (LMP/FP/HP triggered), quiet stages are skipped entirely:
    /// quiets (killers and countermove included) are never yielded, jumping straight
    /// to bad captures. Skipped quiets thus never inflate the caller's
    /// move count.
    skip_quiets: bool,
    /// When set, `next` stops once before generating the quiets and returns
    /// `Move::NONE` there, so the caller can generate and rescore them itself.
    pause_before_quiets: bool,
    /// Quiescence: no good/bad split, every capture is yielded best score first
    /// and the node's own SEE gates decide (ProbCut keeps the split).
    no_split: bool,
    /// Generated quiets left unyielded when the skip took effect (stats only).
    #[cfg(feature = "stats")]
    skipped_quiets: u32,
    /// The skip took effect before the quiets were generated (stats only).
    #[cfg(feature = "stats")]
    skipped_ungenerated: bool,
}

impl MovePicker {
    /// Create a move picker for a full search node.
    pub fn new(tt_move: Move, killers: [Move; 2], countermove: Move, depth: i32) -> MovePicker {
        MovePicker {
            stage: if tt_move != Move::NONE {
                Stage::TTMove
            } else {
                Stage::GenerateCaptures
            },
            tt_move,
            killers,
            countermove,
            entries: ArrayBuf::new(),
            count: 0,
            bad_start: MAX_MOVES,
            index: 0,
            bad_cap_index: MAX_MOVES,
            depth,
            ref_slot: 0,
            bad_quiet_end: 0,
            bad_quiet_index: 0,
            hold_bad_quiets: false,
            skip_quiets: false,
            pause_before_quiets: false,
            no_split: false,
            #[cfg(feature = "stats")]
            skipped_quiets: 0,
            #[cfg(feature = "stats")]
            skipped_ungenerated: false,
        }
    }

    /// Arm quiet skipping: all remaining quiet stages (generation, quiets, bad
    /// quiets) are bypassed but for the killers and the countermove; bad captures
    /// are still yielded.
    /// Quiet checks are bypassed too: the caller guards only the move it prunes.
    pub fn skip_quiet_moves(&mut self) {
        self.skip_quiets = true;
    }

    /// What the quiet skip dropped at this node: the generated quiets it left
    /// unyielded, and whether it took effect before the quiets were generated
    /// (their number is then unknown). Nothing until the picker reaches the skip.
    #[cfg(feature = "stats")]
    pub fn skipped_quiets(&self) -> (u32, bool) {
        (self.skipped_quiets, self.skipped_ungenerated)
    }

    /// Whether `m` is one of this node's killers or its countermove.
    #[inline]
    pub fn is_refutation(&self, m: Move) -> bool {
        m != Move::NONE && (m == self.killers[0] || m == self.killers[1] || m == self.countermove)
    }

    /// True while the last yielded move came from the losing-capture stage
    /// (SEE below the history-shifted split).
    pub fn in_bad_captures(&self) -> bool {
        self.stage == Stage::BadCaptures
    }

    /// Drop the remaining losing captures: the bad-capture futility found that
    /// the next ones cannot reach alpha. The held-back bad quiets, if any, are
    /// still to come: that futility bound rests on the captured piece and says
    /// nothing about them.
    pub fn skip_bad_captures(&mut self) {
        debug_assert!(self.stage == Stage::BadCaptures, "skip_bad_captures outside the stage");
        self.bad_cap_index = self.bad_start;
    }

    /// Create a move picker for quiescence search (captures only).
    pub fn new_qsearch(tt_move: Move) -> MovePicker {
        MovePicker {
            stage: if tt_move != Move::NONE {
                Stage::TTMove
            } else {
                Stage::GenerateCaptures
            },
            tt_move,
            killers: [Move::NONE; 2],
            countermove: Move::NONE,
            entries: ArrayBuf::new(),
            count: 0,
            bad_start: MAX_MOVES,
            index: 0,
            bad_cap_index: MAX_MOVES,
            depth: 0,
            ref_slot: 0,
            bad_quiet_end: 0,
            bad_quiet_index: 0,
            hold_bad_quiets: false,
            skip_quiets: false,
            pause_before_quiets: false,
            no_split: false,
            #[cfg(feature = "stats")]
            skipped_quiets: 0,
            #[cfg(feature = "stats")]
            skipped_ungenerated: false,
        }
    }

    /// Quiescence picker without the good/bad split: all captures, best score
    /// first. The split threshold belongs to the main search's ordering; here it
    /// would discard captures the node's futility and SEE gates never get to judge.
    pub fn new_qsearch_all(tt_move: Move) -> MovePicker {
        let mut p = Self::new_qsearch(tt_move);
        p.no_split = true;
        p
    }

    /// Get the next move. Returns `Move::NONE` when exhausted.
    /// Moves are yielded in priority order but legality is NOT checked here.
    ///
    /// `QSEARCH`: const generic — when true, the compiler eliminates all quiet/killer/bad-capture
    /// stages, generating a specialized qsearch version with no dead code.
    /// The history tables are borrowed individually rather than through the thread
    /// data, so the move picker can be driven from a test with tables of its own.
    #[allow(clippy::too_many_arguments)]
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
            // Quiet skipping armed: bypass every quiet-yielding stage
            if !QSEARCH
                && self.skip_quiets
                && matches!(
                    self.stage,
                    Stage::GenerateQuiets | Stage::Quiets
                )
            {
                #[cfg(feature = "stats")]
                {
                    if self.stage == Stage::Quiets {
                        debug_assert!(self.index <= self.count);
                        // Held-back bad quiets are counted just below.
                        self.skipped_quiets += (self.count - self.index) as u32;
                    } else {
                        self.skipped_ungenerated = true;
                    }
                }
                // The killers and the countermove outlive the skip: under the
                // fixed refutation stages they came before the move-count cut,
                // and the caller exempts them from it.
                self.stage = if self.stage == Stage::Quiets {
                    Stage::QuietRefutations
                } else {
                    Stage::SkippedRefutations
                };
            }
            match self.stage {
                Stage::TTMove => {
                    self.stage = Stage::GenerateCaptures;
                    if self.tt_move.is_ok() && movegen::is_pseudo_legal(pos, self.tt_move) {
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
                            debug_assert!(
                                piece != Piece::NONE,
                                "movepick: EP no piece on from {}",
                                m.from_sq().0
                            );
                            cap_history.get(pos.threats, piece, m.to_sq(), PieceType::Pawn)
                        } else if mt == MT_PROMOTION {
                            let victim_pc = pos.board[m.to_sq().index()];
                            let victim_val = if victim_pc != Piece::NONE {
                                PIECE_VALUE[victim_pc.piece_type() as usize]
                            } else {
                                0
                            };
                            PIECE_VALUE[m.promo_type() as usize] * tune::MVV_MULTIPLIER()
                                + victim_val
                        } else {
                            let victim_pc = pos.board[m.to_sq().index()];
                            if victim_pc == Piece::NONE {
                                0
                            } else {
                                let piece = pos.board[m.from_sq().index()];
                                debug_assert!(
                                    piece != Piece::NONE,
                                    "movepick: no piece on from {} for capture {}",
                                    m.from_sq().0,
                                    m.to_uci()
                                );
                                let captured_pt = victim_pc.piece_type();
                                PIECE_VALUE[captured_pt as usize] * tune::MVV_MULTIPLIER()
                                    + cap_history.get(pos.threats, piece, m.to_sq(), captured_pt)
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
                        if QSEARCH && self.no_split {
                            return m;
                        }
                        // Good/bad split against a SEE threshold shifted by the
                        // move's capture history: a capture with a good track
                        // record keeps its spot near the front even when static
                        // SEE loses a little material — history knows about
                        // discovered threats and recurring sacrifices that SEE
                        // cannot see. Promotions keep the plain threshold
                        // (their score carries no capture history).
                        let mt = m.move_type();
                        let victim_pc = pos.board[m.to_sq().index()];
                        let threshold = if mt == MT_PROMOTION
                            || (mt != MT_EN_PASSANT && victim_pc == Piece::NONE)
                        {
                            0
                        } else {
                            let piece = pos.board[m.from_sq().index()];
                            debug_assert!(
                                piece != Piece::NONE,
                                "movepick: no piece on from {} for split {}",
                                m.from_sq().0,
                                m.to_uci()
                            );
                            let captured_pt = if mt == MT_EN_PASSANT {
                                PieceType::Pawn
                            } else {
                                victim_pc.piece_type()
                            };
                            let caphist = cap_history.get(pos.threats, piece, m.to_sq(), captured_pt);
                            tune::GOOD_CAP_BASE() - caphist / tune::GOOD_CAP_HIST_DIV()
                        };
                        if see::see(pos, m, threshold) {
                            return m;
                        }
                        // Bad capture: store at end of array (LIFO)
                        debug_assert!(
                            self.bad_start > self.count,
                            "movepick: bad captures overflow into active region"
                        );
                        self.bad_start -= 1;
                        self.entries.set(self.bad_start, ScoredMove { score: 0, mv: m });
                        continue;
                    }
                    if QSEARCH {
                        self.stage = Stage::Done;
                    } else {
                        self.stage = Stage::GenerateQuiets;
                    }
                }

                // --- These stages are eliminated by the compiler when QSEARCH=true ---
                Stage::GenerateQuiets => {
                    if QSEARCH {
                        self.stage = Stage::Done;
                        continue;
                    }
                    if self.pause_before_quiets {
                        self.pause_before_quiets = false;
                        return Move::NONE;
                    }
                    self.generate_quiets(pos, history, pawn_history, tt, ply, stack, stm);
                }

                Stage::Quiets => {
                    if QSEARCH {
                        self.stage = Stage::Done;
                        continue;
                    }
                    let threshold = tune::MOVEPICK_BAD_QUIET_THRESHOLD();
                    while self.index < self.count {
                        let e = self.entries[self.index];
                        self.index += 1;
                        if self.hold_bad_quiets && e.score < threshold {
                            debug_assert!(self.bad_quiet_end < self.index);
                            self.entries[self.bad_quiet_end] = e;
                            self.bad_quiet_end += 1;
                            continue;
                        }
                        return e.mv;
                    }
                    self.stage = Stage::BadCaptures;
                }

                Stage::QuietRefutations => {
                    debug_assert!(!QSEARCH && self.skip_quiets);
                    while self.index < self.count {
                        let m = self.entries[self.index].mv;
                        self.index += 1;
                        if self.is_refutation(m) {
                            return m;
                        }
                    }
                    self.stage = Stage::BadCaptures;
                }

                Stage::SkippedRefutations => {
                    debug_assert!(!QSEARCH && self.skip_quiets);
                    while self.ref_slot < 3 {
                        let slot = self.ref_slot as usize;
                        self.ref_slot += 1;
                        let m = if slot < 2 { self.killers[slot] } else { self.countermove };
                        let duplicate = match slot {
                            0 => false,
                            1 => m == self.killers[0],
                            _ => m == self.killers[0] || m == self.killers[1],
                        };
                        if m != Move::NONE
                            && m != self.tt_move
                            && !duplicate
                            && is_quiet(pos, m)
                            && movegen::is_pseudo_legal(pos, m)
                        {
                            return m;
                        }
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
                    self.stage = if self.hold_bad_quiets { Stage::BadQuiets } else { Stage::Done };
                }

                Stage::BadQuiets => {
                    if QSEARCH {
                        self.stage = Stage::Done;
                        continue;
                    }
                    debug_assert!(self.bad_quiet_end <= self.count);
                    while self.bad_quiet_index < self.bad_quiet_end {
                        let m = self.entries[self.bad_quiet_index].mv;
                        self.bad_quiet_index += 1;
                        debug_assert!(m != self.tt_move, "movepick: TT move held back");
                        // Once quiets are skipped, the held-back quiets go with
                        // them, except the refutations, exempt as everywhere else.
                        if !self.skip_quiets || self.is_refutation(m) {
                            return m;
                        }
                        #[cfg(feature = "stats")]
                        {
                            self.skipped_quiets += 1;
                        }
                    }
                    self.stage = Stage::Done;
                }

                Stage::Done => {
                    return Move::NONE;
                }
            }
        }
    }

    /// Make `next` stop once before generating the quiets: it then returns
    /// `Move::NONE` with `at_quiet_generation` true, and the caller generates the
    /// quiets with `generate_quiets` before asking for the next move again.
    pub fn pause_before_quiets(&mut self) {
        self.pause_before_quiets = true;
    }

    /// True when the next call to `next` would generate the quiets: the good captures
    /// are behind, and no quiet skip is armed.
    #[inline]
    pub fn at_quiet_generation(&self) -> bool {
        self.stage == Stage::GenerateQuiets && !self.skip_quiets
    }

    /// Number of generated quiets not yet yielded.
    #[inline]
    pub fn pending_quiets(&self) -> usize {
        debug_assert!(self.stage == Stage::Quiets);
        debug_assert!(self.index <= self.count);
        self.count - self.index
    }

    /// The `i`-th quiet not yet yielded, in the current order.
    #[inline]
    pub fn pending_quiet(&self, i: usize) -> ScoredMove {
        debug_assert!(self.stage == Stage::Quiets);
        debug_assert!(self.index + i < self.count, "pending quiet {i} OOB");
        self.entries[self.index + i]
    }

    /// Add `delta` to the score of the `i`-th quiet not yet yielded. Call
    /// `resort_quiets` once the scores are adjusted.
    #[inline]
    pub fn adjust_pending_quiet(&mut self, i: usize, delta: i32) {
        debug_assert!(self.stage == Stage::Quiets);
        debug_assert!(self.index + i < self.count, "pending quiet {i} OOB");
        let e = &mut self.entries[self.index + i];
        e.score = e.score.saturating_add(delta);
    }

    /// Sort the quiets not yet yielded again, after their scores were adjusted, with
    /// the limit of the generation stage.
    pub fn resort_quiets(&mut self, in_check: bool) {
        debug_assert!(self.stage == Stage::Quiets);
        let limit = if in_check { i32::MIN } else { -3500 * self.depth };
        Self::partial_insertion_sort(&mut self.entries, self.index, self.count, limit);
    }

    /// The quiet generation stage: generate, score and sort the quiets, then move on
    /// to yielding them. Called by `next`, or ahead of it by a caller that rescores
    /// the quiets before they are yielded.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub fn generate_quiets(
        &mut self,
        pos: &Position,
        history: &ButterflyHistory,
        pawn_history: &PawnHistory,
        tt: &TT,
        ply: usize,
        stack: &[StackEntry],
        stm: Color,
    ) {
        debug_assert!(self.stage == Stage::GenerateQuiets);
        // Generate quiets directly into entries, then score + compact in-place
        let gen_count = movegen::generate_quiets(pos, &mut self.entries);

        // Precompute the direct-check squares for each piece type
        // (movepicker check bonus for quiet checking moves)
        let their_king = pos.king_sq(stm.flip());
        let occ = pos.occupied();
        let (diagonal, straight) = both_attacks(their_king, occ);
        let check_sqs: [u64; 6] = [
            pawn_attacks(their_king, stm.flip()), // PieceType::Pawn = 0
            knight_attacks(their_king),           // PieceType::Knight = 1
            diagonal,                             // PieceType::Bishop = 2
            straight,                             // PieceType::Rook = 3
            diagonal | straight,                  // PieceType::Queen = 4
            0u64, // PieceType::King = 5 (no direct check)
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

        // The refutations are scored, not staged: a killer or the
        // countermove gets a bonus on top of its histories, so a quiet
        // with a stronger record, a check or a good look-ahead can still
        // pass it, and a refutation gone bad (its piece now hanging, say) can
        // fall behind. One bonus per move, the strongest slot first.
        debug_assert!(
            self.killers[0] == Move::NONE || self.killers[0] != self.killers[1],
            "movepick: duplicate killers {}",
            self.killers[0].to_uci()
        );
        let killer1_bonus = tune::MOVEPICK_KILLER1_BONUS();
        let killer2_bonus = tune::MOVEPICK_KILLER2_BONUS();
        let counter_bonus = tune::MOVEPICK_COUNTER_BONUS();

        // Weights of the history tables in the quiet score, in 1/1024, by
        // depth: near the leaves the butterfly and pawn history deserve more weight.
        let w = quiet_weights(self.depth);
        debug_assert!(w.iter().all(|&x| (0..=4096).contains(&x)));

        // Score quiets and compact: remove the TT move, already yielded
        let mut j = 0;
        for i in 0..gen_count {
            let m = self.entries[i].mv;
            if m == self.tt_move {
                continue; // already yielded in earlier stage
            }
            let piece = pos.board[m.from_sq().index()];
            debug_assert!(
                piece != Piece::NONE,
                "movepick: no piece on from {} for quiet {}",
                m.from_sq().0,
                m.to_uci()
            );
            let to = m.to_sq();
            let base = ply + SS_OFFSET;
            let cont = |offset: usize| {
                ContinuationHistory::get(
                    stack[base - offset].conthist_ptr as *const _,
                    piece,
                    to,
                )
            };
            let mut score = (w[0] * history.get(stm, m, pos.threats)
                + w[1] * pawn_history.get(pos.pawn_key, piece, to)
                + w[2] * cont(1)
                + w[3] * cont(2)
                + w[4] * cont(4)
                + w[5] * cont(6))
                / 1024;
            if m == self.killers[0] {
                score += killer1_bonus;
            } else if m == self.killers[1] {
                score += killer2_bonus;
            } else if m == self.countermove {
                score += counter_bonus;
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
                    #[cfg(feature = "stats")]
                    let _ = crate::ttstat::probe(tt, crate::ttstat::LOOKAHEAD, ckey, Some(&hit));
                    // Use static eval (more stable than search score for ordering).
                    // Eval is from child's STM (= opponent) → negate for our perspective.
                    let raw = if hit.eval != crate::types::SCORE_NONE {
                        hit.eval
                    } else {
                        hit.score
                    };
                    debug_assert!(tune::LOOKAHEAD_BONUS_DIV() > 0);
                    score += -raw * tune::LOOKAHEAD_BONUS_MUL()
                        / tune::LOOKAHEAD_BONUS_DIV();
                } else {
                    #[cfg(feature = "stats")]
                    let _ = crate::ttstat::probe(tt, crate::ttstat::LOOKAHEAD, ckey, None);
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
        // Evasions are sorted in full: the depth-scaled limit would
        // leave every quiet evasion of a quiescence node (depth 0)
        // in generation order, and the first one often decides alone.
        let limit = if pos.checkers != 0 { i32::MIN } else { -3500 * self.depth };
        Self::partial_insertion_sort(&mut self.entries, 0, j, limit);
        // Evasions are never held back: every one of them is searched
        // in sorted order.
        self.hold_bad_quiets = pos.checkers == 0;
        self.bad_quiet_end = 0;
        self.bad_quiet_index = 0;
        self.stage = Stage::Quiets;
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
    fn partial_insertion_sort(
        entries: &mut ArrayBuf<ScoredMove, MAX_MOVES>,
        start: usize,
        end: usize,
        limit: i32,
    ) {
        debug_assert!(start <= end && end <= MAX_MOVES, "sort range {start}..{end}");
        let mut sorted_end = start;
        let mut p = start + 1;

        // The first moves above the limit go through a sorter held in two vector
        // registers; the scalar loop below takes over once it is full. Same order
        // as the scalar loop alone, ties and the moves left below the limit included.
        #[cfg(target_feature = "avx512f")]
        {
            if start == end {
                return;
            }
            let mut sorter = vsort::MoveSorter::new(entries[start]);
            while p < end {
                if entries[p].score >= limit {
                    if sorted_end - start + 1 >= vsort::MoveSorter::CAPACITY {
                        break;
                    }
                    sorter.insert(entries[p]);
                    sorted_end += 1;
                    entries[p] = entries[sorted_end];
                }
                p += 1;
            }
            sorter.write(entries, start, sorted_end - start + 1);
        }

        while p < end {
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
            p += 1;
        }
    }
}

/// Insertion sort of up to 16 scored moves in two AVX-512 registers: one holds the
/// scores in descending order, the other the moves, lane for lane.
#[cfg(target_feature = "avx512f")]
mod vsort {
    use crate::types::{MAX_MOVES, ScoredMove};
    use crate::types::ArrayBuf;
    use std::arch::x86_64::*;

    const _: () = assert!(std::mem::size_of::<ScoredMove>() == 8);
    const _: () = assert!(std::mem::offset_of!(ScoredMove, score) == 0);
    const _: () = assert!(std::mem::offset_of!(ScoredMove, mv) == 4);

    pub(super) struct MoveSorter {
        scores: __m512i,
        moves: __m512i,
    }

    impl MoveSorter {
        pub(super) const CAPACITY: usize = 16;

        /// A sorter holding `first`; the empty lanes score `i32::MIN` so that every
        /// inserted move ranks before them.
        #[inline(always)]
        pub(super) fn new(first: ScoredMove) -> Self {
            // SAFETY (every intrinsic of this module): it is compiled only when the
            // target enables AVX-512F, so the instructions exist.
            unsafe {
                let scores = _mm512_mask_set1_epi32(_mm512_set1_epi32(first.score), !1, i32::MIN);
                Self { scores, moves: _mm512_set1_epi32(first.mv.0 as i32) }
            }
        }

        /// Insert `m` after every held move whose score is at least its own, as the
        /// scalar insertion does. The caller keeps the count below `CAPACITY`.
        #[inline(always)]
        pub(super) fn insert(&mut self, m: ScoredMove) {
            debug_assert!(m.score != i32::MIN, "MoveSorter: score i32::MIN ranks with empty lanes");
            unsafe {
                let score = _mm512_set1_epi32(m.score);
                // Lanes scoring below `m` form a suffix starting at the insertion point
                // (the empty lanes guarantee it is not empty). Subtracting one clears
                // that lane and sets every lane before it: the held moves expand around it.
                let below: u16 = _mm512_cmplt_epi32_mask(self.scores, score);
                debug_assert!(below != 0, "MoveSorter: insertion into a full sorter");
                let keep = below.wrapping_sub(1);
                self.scores = _mm512_mask_expand_epi32(score, keep, self.scores);
                self.moves =
                    _mm512_mask_expand_epi32(_mm512_set1_epi32(m.mv.0 as i32), keep, self.moves);
            }
        }

        /// Write the `count` held moves, best first, to `entries[start..start + count]`.
        #[inline(always)]
        pub(super) fn write(&self, entries: &mut ArrayBuf<ScoredMove, MAX_MOVES>, start: usize, count: usize) {
            debug_assert!((1..=Self::CAPACITY).contains(&count), "MoveSorter: count {count}");
            debug_assert!(start + count <= MAX_MOVES, "MoveSorter: {start} + {count} overflows");
            // Interleave score and move of each lane into one 64-bit entry: score in
            // the low half, move zero-extended in the high half (padding cleared).
            let dst = entries.as_mut_ptr().wrapping_add(start).cast::<i64>();
            // SAFETY: the stores write only their masked lanes, which lie in
            // entries[start..start + count], inside the buffer.
            unsafe {
                let low =
                    _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
                let first = _mm512_permutex2var_epi32(self.scores, low, self.moves);
                _mm512_mask_storeu_epi64(dst, ((1u32 << count.min(8)) - 1) as __mmask8, first);
                if count > 8 {
                    let high = _mm512_setr_epi32(
                        8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31,
                    );
                    let second = _mm512_permutex2var_epi32(self.scores, high, self.moves);
                    let mask = ((1u32 << (count - 8)) - 1) as __mmask8;
                    _mm512_mask_storeu_epi64(dst.wrapping_add(8), mask, second);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sort the picker relied on before any vector path: insertion of every move
    /// at or above `limit` into a growing sorted prefix.
    fn reference_partial_sort(v: &mut [ScoredMove], limit: i32) {
        let mut sorted_end = 0;
        for p in 1..v.len() {
            if v[p].score >= limit {
                let tmp = v[p];
                sorted_end += 1;
                v[p] = v[sorted_end];
                let mut q = sorted_end;
                while q > 0 && v[q - 1].score < tmp.score {
                    v[q] = v[q - 1];
                    q -= 1;
                }
                v[q] = tmp;
            }
        }
    }

    /// `partial_insertion_sort` must give exactly the order of the plain insertion
    /// sort, whatever path it takes: ties, the moves left below the limit, lists
    /// longer than the vector sorter, offsets into the buffer, and the full sort of
    /// evasions (limit `i32::MIN`).
    #[test]
    fn partial_sort_matches_reference_order() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for round in 0..20_000 {
            let len = (next() % 80) as usize;
            let start = (next() % 4) as usize * (round & 1);
            // Narrow score ranges force many ties.
            let spread = [4u64, 64, 20_000, 1 << 31][(next() % 4) as usize];
            let mut buf: ArrayBuf<ScoredMove, MAX_MOVES> = ArrayBuf::new();
            let mut plain = Vec::with_capacity(len);
            for i in 0..start + len {
                let score = (next() % spread) as i64 - (spread / 2) as i64;
                let score = score.clamp(i32::MIN as i64 + 1, i32::MAX as i64) as i32;
                let m = ScoredMove { score, mv: Move(i as u16 + 1) };
                buf[i] = m;
                if i >= start {
                    plain.push(m);
                }
            }
            let limit = match next() % 4 {
                0 => i32::MIN,
                1 => 0,
                _ => (next() % spread) as i32 - (spread / 2) as i32,
            };
            MovePicker::partial_insertion_sort(&mut buf, start, start + len, limit);
            reference_partial_sort(&mut plain, limit);
            for (k, want) in plain.iter().enumerate() {
                let got = buf[start + k];
                assert!(
                    got.score == want.score && got.mv == want.mv,
                    "round {round}, len {len}, limit {limit}: slot {k} holds {:?}/{} instead of {:?}/{}",
                    got.mv, got.score, want.mv, want.score
                );
            }
        }
    }

    /// The look-ahead probes the TT with a key `child_key` predicts without playing the
    /// move, and the move loops prefetch with it. A key that does not name the child
    /// position probes nothing, so it must agree with `make_move` on every legal move —
    /// captures, en passant, promotions plain and capturing, and castling, where king
    /// and rook both move and their squares may overlap.
    /// Quiet is what the quiet stage generates, and only that: every move from
    /// `generate_quiets` is quiet, every move from `generate_captures` is not.
    #[test]
    fn quiet_is_what_the_quiet_stage_generates() {
        for fen in [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            // Promotions by push and by capture, both sides.
            "1n2k2r/P6P/8/8/8/8/6K1/8 w k - 0 1",
            "8/6k1/8/8/8/8/p6p/1N2K2R b K - 0 1",
            // En passant.
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
        ] {
            let pos = Position::from_fen(fen).unwrap();
            let mut buf = ArrayBuf::<ScoredMove, MAX_MOVES>::new();
            let n = movegen::generate_quiets(&pos, &mut buf);
            for i in 0..n {
                let m = buf[i].mv;
                assert!(is_quiet(&pos, m), "{fen}: quiet stage made {}", m.to_uci());
            }
            let mut buf = ArrayBuf::<ScoredMove, MAX_MOVES>::new();
            let n = movegen::generate_captures(&pos, &mut buf);
            for i in 0..n {
                let m = buf[i].mv;
                assert!(!is_quiet(&pos, m), "{fen}: capture stage made {}", m.to_uci());
            }
        }
    }

    /// The pause before the quiets and their rescoring: the picker stops once with no
    /// move, the caller generates the quiets, raises the last one to the top, and every
    /// pseudo-legal move still comes out exactly once, the raised quiet first among them.
    #[test]
    fn a_paused_picker_yields_rescored_quiets_once_each() {
        use crate::history::{CaptureHistory, PawnHistory, PieceToTable};
        use crate::threads::{SS_OFFSET, StackEntry};
        let pos = Position::from_fen(
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        )
        .unwrap();
        let stm = pos.side_to_move;
        let history = Box::new(ButterflyHistory::new());
        let cap_history = Box::new(CaptureHistory::new());
        let pawn_history = PawnHistory::new();
        let tt = TT::new(1);
        let mut empty: Box<PieceToTable> = Box::new([[0; Square::NUM]; Piece::NUM]);
        let empty_ptr: *mut PieceToTable = &mut *empty;
        let stack: Vec<StackEntry> = (0..SS_OFFSET + 2)
            .map(|_| StackEntry { conthist_ptr: empty_ptr, ..StackEntry::default() })
            .collect();

        let mut picker = MovePicker::new(Move::NONE, [Move::NONE; 2], Move::NONE, 10);
        picker.pause_before_quiets();
        let mut out = Vec::new();
        let mut raised = Move::NONE;
        let mut paused = 0;
        loop {
            let m = picker.next::<false>(
                &pos, &history, &cap_history, &pawn_history, &tt, 0, &stack, stm,
            );
            if m == Move::NONE && picker.at_quiet_generation() {
                paused += 1;
                picker.generate_quiets(&pos, &history, &pawn_history, &tt, 0, &stack, stm);
                let n = picker.pending_quiets();
                assert!(n >= 2, "only {n} quiets");
                raised = picker.pending_quiet(n - 1).mv;
                picker.adjust_pending_quiet(n - 1, 1 << 24);
                picker.resort_quiets(false);
                assert_eq!(picker.pending_quiet(0).mv, raised);
                continue;
            }
            if m == Move::NONE {
                break;
            }
            out.push(m);
        }
        assert_eq!(paused, 1, "the picker must pause exactly once");
        let first_quiet = out.iter().copied().find(|&m| is_quiet(&pos, m)).unwrap();
        assert_eq!(first_quiet, raised);
        let mut all = ArrayBuf::<ScoredMove, MAX_MOVES>::new();
        let n = movegen::generate_captures(&pos, &mut all);
        let mut expected: Vec<u16> = (0..n).map(|i| all[i].mv.0).collect();
        let nq = movegen::generate_quiets(&pos, &mut all);
        expected.extend((0..nq).map(|i| all[i].mv.0));
        let mut got: Vec<u16> = out.iter().map(|m| m.0).collect();
        expected.sort_unstable();
        got.sort_unstable();
        assert_eq!(got, expected, "every move once");
    }

    /// A quiescence node in check builds its picker at depth 0, where the quiet
    /// stage's depth-scaled sort limit is 0: evasions with a negative history were
    /// handed out in generation order. Here every quiet evasion scores below zero,
    /// worst first in generation order; they must still come out best first.
    #[test]
    fn quiet_evasions_come_out_best_first_at_depth_zero() {
        use crate::history::{CaptureHistory, PawnHistory, PieceToTable};
        use crate::threads::{SS_OFFSET, StackEntry};
        // The rook on a1 checks the king on e1; only quiet king steps answer it.
        let pos = Position::from_fen("4k3/8/8/8/8/8/8/r3K3 w - - 0 1").unwrap();
        assert!(pos.checkers != 0);
        let stm = pos.side_to_move;

        let mut quiets = ArrayBuf::<ScoredMove, MAX_MOVES>::new();
        let n = movegen::generate_quiets(&pos, &mut quiets);
        assert!(n >= 3, "only {n} quiet moves generated");
        let mut history = Box::new(ButterflyHistory::new());
        for i in 0..n {
            let malus = -200 * (n - i) as i32;
            history.update(stm, quiets[i].mv, pos.threats, malus);
        }
        for i in 0..n {
            assert!(history.get(stm, quiets[i].mv, pos.threats) < 0);
        }

        let cap_history = Box::new(CaptureHistory::new());
        let pawn_history = PawnHistory::new();
        let tt = TT::new(1);
        let mut empty: Box<PieceToTable> = Box::new([[0; Square::NUM]; Piece::NUM]);
        let empty_ptr: *mut PieceToTable = &mut *empty;
        let stack: Vec<StackEntry> = (0..SS_OFFSET + 2)
            .map(|_| StackEntry { conthist_ptr: empty_ptr, ..StackEntry::default() })
            .collect();

        let mut picker = MovePicker::new(Move::NONE, [Move::NONE; 2], Move::NONE, 0);
        let mut scores = Vec::new();
        loop {
            let m = picker.next::<false>(
                &pos,
                &history,
                &cap_history,
                &pawn_history,
                &tt,
                0,
                &stack,
                stm,
            );
            if m == Move::NONE {
                break;
            }
            if is_quiet(&pos, m) {
                scores.push(history.get(stm, m, pos.threats));
            }
        }
        assert_eq!(scores.len(), n);
        assert!(
            scores.windows(2).all(|w| w[0] >= w[1]),
            "quiet evasions out of order: {scores:?}"
        );
    }

    /// A quiet scored below the bad-quiet threshold comes out after the losing
    /// captures, every move still exactly once; once quiets are skipped, the
    /// held-back quiet is dropped with them.
    #[test]
    fn bad_quiets_come_after_bad_captures_and_go_with_the_skip() {
        use crate::history::{CaptureHistory, PawnHistory, PieceToTable};
        use crate::threads::{SS_OFFSET, StackEntry};
        // Qxd5 loses the queen to exd5: the only capture, and a bad one.
        let pos = Position::from_fen("4k3/8/4p3/3p4/8/8/8/3QK3 w - - 0 1").unwrap();
        let stm = pos.side_to_move;
        let mut quiets = ArrayBuf::<ScoredMove, MAX_MOVES>::new();
        let nq = movegen::generate_quiets(&pos, &mut quiets);
        let bad_quiet = quiets[0].mv;
        let mut history = Box::new(ButterflyHistory::new());
        for _ in 0..64 {
            history.update(stm, bad_quiet, pos.threats, -4000);
        }
        assert!(history.get(stm, bad_quiet, pos.threats) < tune::MOVEPICK_BAD_QUIET_THRESHOLD());
        let cap_history = Box::new(CaptureHistory::new());
        let pawn_history = PawnHistory::new();
        let tt = TT::new(1);
        let mut empty: Box<PieceToTable> = Box::new([[0; Square::NUM]; Piece::NUM]);
        let empty_ptr: *mut PieceToTable = &mut *empty;
        let stack: Vec<StackEntry> = (0..SS_OFFSET + 2)
            .map(|_| StackEntry { conthist_ptr: empty_ptr, ..StackEntry::default() })
            .collect();
        let drain = |picker: &mut MovePicker, skip_after: Option<usize>| {
            let mut out = Vec::new();
            loop {
                let m = picker.next::<false>(
                    &pos,
                    &history,
                    &cap_history,
                    &pawn_history,
                    &tt,
                    0,
                    &stack,
                    stm,
                );
                if m == Move::NONE {
                    break;
                }
                out.push(m);
                if Some(out.len()) == skip_after {
                    picker.skip_quiet_moves();
                }
            }
            out
        };

        let mut picker = MovePicker::new(Move::NONE, [Move::NONE; 2], Move::NONE, 1);
        let out = drain(&mut picker, None);
        assert_eq!(out.len(), nq + 1);
        assert_eq!(*out.last().unwrap(), bad_quiet, "bad quiet not last");
        assert!(!is_quiet(&pos, out[out.len() - 2]), "bad capture not before the bad quiet");

        // Skip armed after the first quiet: the bad capture comes, the bad quiet does not.
        let mut picker = MovePicker::new(Move::NONE, [Move::NONE; 2], Move::NONE, 1);
        let out = drain(&mut picker, Some(1));
        assert_eq!(out.len(), 2);
        assert!(!out.contains(&bad_quiet));
        assert!(!is_quiet(&pos, out[1]));
    }

    /// The killers and the countermove are scored among the quiets rather than
    /// staged: every pseudo-legal move must still come out exactly once, the TT
    /// move first and never again, even when it is also a killer, and the bonus
    /// must lift a refutation above quiets with no history.
    #[test]
    fn refutations_are_scored_and_every_move_comes_out_once() {
        use crate::history::{CaptureHistory, PawnHistory, PieceToTable};
        use crate::threads::{SS_OFFSET, StackEntry};
        let pos = Position::from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        )
        .unwrap();
        let stm = pos.side_to_move;
        let mut quiets = ArrayBuf::<ScoredMove, MAX_MOVES>::new();
        let nq = movegen::generate_quiets(&pos, &mut quiets);
        let mut caps = ArrayBuf::<ScoredMove, MAX_MOVES>::new();
        let nc = movegen::generate_captures(&pos, &mut caps);
        assert!(nq >= 4);
        // TT move = first killer; second killer and countermove are other quiets,
        // the countermove picked late in generation order.
        let tt_move = quiets[0].mv;
        let killers = [quiets[0].mv, quiets[1].mv];
        let countermove = quiets[nq - 1].mv;

        let history = Box::new(ButterflyHistory::new());
        let cap_history = Box::new(CaptureHistory::new());
        let pawn_history = PawnHistory::new();
        let tt = TT::new(1);
        let mut empty: Box<PieceToTable> = Box::new([[0; Square::NUM]; Piece::NUM]);
        let empty_ptr: *mut PieceToTable = &mut *empty;
        let stack: Vec<StackEntry> = (0..SS_OFFSET + 2)
            .map(|_| StackEntry { conthist_ptr: empty_ptr, ..StackEntry::default() })
            .collect();

        let mut picker = MovePicker::new(tt_move, killers, countermove, 1);
        let mut out = Vec::new();
        loop {
            let m = picker.next::<false>(
                &pos,
                &history,
                &cap_history,
                &pawn_history,
                &tt,
                0,
                &stack,
                stm,
            );
            if m == Move::NONE {
                break;
            }
            out.push(m);
        }
        assert_eq!(out[0], tt_move);
        assert_eq!(out.len(), nq + nc, "a move came out twice or not at all");
        let mut sorted: Vec<u16> = out.iter().map(|m| m.0).collect();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), out.len(), "duplicate move yielded");
        // With empty histories and no check bonus on either, both refutations
        // lead the quiets (the check bonus may still put a checking quiet first).
        let qpos = |m: Move| out.iter().filter(|&&x| is_quiet(&pos, x)).position(|&x| x == m);
        let (k2, cm) = (qpos(killers[1]).unwrap(), qpos(countermove).unwrap());
        assert!(k2 <= 3 && cm <= 3, "refutations not lifted: killer2 {k2}, countermove {cm}");
    }

    #[test]
    fn the_predicted_child_key_is_the_one_make_move_produces() {
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1",
            // An en passant square to clear, whatever the move played.
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
            // An en passant capture to play, for either side.
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
            "rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 3",
            // A rook taken on its home square loses the right with it.
            "4k2r/8/8/8/8/8/8/4K2R w Kk - 0 1",
            "r3k3/8/8/8/8/8/8/R3K3 b Qq - 0 1",
            // Promotions, plain and capturing, for either side; one takes a rook on
            // its home square.
            "1n2k2r/P6P/8/8/8/8/6K1/8 w k - 0 1",
            "8/6k1/8/8/8/8/p6p/1N2K2R b K - 0 1",
            // A black pawn on d4 answers both c2c4 and e2e4, so those two double pushes
            // leave an en passant file behind and the others do not.
            "rnbqkbnr/ppp1pppp/8/8/3p4/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            // FRC: castling short swaps king and rook.
            "4k3/8/8/8/8/8/8/5KR1 w K - 0 1",
            // FRC: the king already stands on g1 and stays there.
            "4k3/8/8/8/8/8/8/6KR w K - 0 1",
            // FRC: castling long leaves the king where it is, on both sides.
            "1rk5/8/8/8/8/8/8/1RK5 w Qq - 0 1",
            "1rk5/8/8/8/8/8/8/1RK5 b Qq - 0 1",
        ];
        let mut checked = 0;
        for fen in fens {
            let pos = Position::from_fen_ex(fen, true).unwrap();
            let mut buf = ArrayBuf::<Move, 256>::new();
            let n = movegen::generate_legal_moves(&pos, &mut buf);
            for i in 0..n {
                let m = buf[i];
                checked += 1;
                let piece = pos.board[m.from_sq().index()];
                let mut child = pos.clone();
                child.make_move(m);
                assert_eq!(
                    child_key(&pos, m, piece),
                    child.key,
                    "child_key mispredicts {} in {fen}",
                    m.to_uci_960(true),
                );
            }
        }
        assert!(checked > 200, "only {checked} moves checked");
    }
}
