//! [Pseudo-legal](https://www.chessprogramming.org/Pseudo-Legal_Move) move generation
//! with legality filtering.
//!
//! Generates all moves that obey piece movement rules but may leave the king in check,
//! then filters illegal ones. Double check forces king moves only. Single check
//! restricts targets to interpositions or captures of the checker. Pinned pieces
//! may only move along their pin ray.

use crate::bitboard::*;
use crate::position::Position;
use crate::types::*;

/// Append a move to the raw buffer.
#[inline(always)]
fn emit(buf: &mut ArrayBuf<Move, MAX_MOVES>, count: &mut usize, m: Move) {
    debug_assert!(*count < MAX_MOVES, "emit: overflow at {}", *count);
    debug_assert!(m.is_ok(), "emit: invalid move {:?}", m);
    buf.set(*count, m);
    *count += 1;
}

/// Append a move to a scored buffer (score = 0, to be filled later by MovePicker).
#[inline(always)]
fn emit_scored(buf: &mut ArrayBuf<ScoredMove, MAX_MOVES>, count: &mut usize, m: Move) {
    debug_assert!(*count < MAX_MOVES, "emit_scored: overflow at {}", *count);
    debug_assert!(m.is_ok(), "emit_scored: invalid move {:?}", m);
    buf.set(*count, ScoredMove { score: 0, mv: m });
    *count += 1;
}

/// Generate all legal moves for the current position. Returns count.
///
/// Pseudo-legal moves are generated first, then each is tested for legality
/// (king safety, pins, en passant discovered checks, castling through check).
pub fn generate_legal_moves(pos: &Position, buf: &mut ArrayBuf<Move, MAX_MOVES>) -> usize {
    let mut count = 0;
    generate_moves(pos, buf, &mut count);
    // Filter illegal moves
    let mut i = 0;
    while i < count {
        if !is_legal(pos, buf[i]) {
            count -= 1;
            buf.copy_within(count, i);
        } else {
            i += 1;
        }
    }

    // Post: all remaining moves must be legal
    #[cfg(debug_assertions)]
    for i in 0..count {
        debug_assert!(
            is_legal(pos, buf[i]),
            "generate_legal_moves: illegal move {} survived filtering",
            buf[i].to_uci()
        );
    }
    count
}

/// Generate pseudo-legal captures + queen promotions + en passant. Returns count.
/// Writes directly into a `ScoredMove` buffer (score = 0, filled later by MovePicker).
/// The caller must filter with `is_legal()`.
pub fn generate_captures(pos: &Position, buf: &mut ArrayBuf<ScoredMove, MAX_MOVES>) -> usize {
    let mut count = 0;
    let us = pos.side_to_move;
    let them = !us;
    let their_pieces = pos.color_bb(them);
    let occupied = pos.occupied();
    let ksq = pos.king_sq(us);
    #[cfg(debug_assertions)]
    let count_before = count;

    let check_mask;
    let num_checkers = popcount(pos.checkers);

    if num_checkers > 1 {
        // Double check: only king captures
        let mut attacks = king_attacks(ksq) & their_pieces;
        while attacks != 0 {
            let to = pop_lsb(&mut attacks);
            emit_scored(buf, &mut count, Move::new(ksq, to));
        }
        return count;
    } else if num_checkers == 1 {
        let checker_sq = lsb(pos.checkers);
        check_mask = between_bb(ksq, checker_sq) | checker_sq.bb();
    } else {
        check_mask = !0u64;
    }

    // Pawn captures + queen promotions
    let pawns = pos.piece_type_bb(PieceType::Pawn, us);
    let promo_rank = match us {
        Color::White => RANK_8,
        Color::Black => RANK_1,
    };
    let push_dir = pawn_push(us);
    let east_dir: i8 = if us == Color::White {
        NORTH_EAST
    } else {
        SOUTH_EAST
    };
    let west_dir: i8 = if us == Color::White {
        NORTH_WEST
    } else {
        SOUTH_WEST
    };

    // Pawn captures
    let cap_east = pawn_attack_east(pawns, us) & their_pieces & check_mask;
    let cap_west = pawn_attack_west(pawns, us) & their_pieces & check_mask;

    // Non-promo captures
    let mut bb = cap_east & !promo_rank;
    while bb != 0 {
        let to = pop_lsb(&mut bb);
        emit_scored(
            buf,
            &mut count,
            Move::new(Square((to.0 as i8 - east_dir) as u8), to),
        );
    }
    let mut bb = cap_west & !promo_rank;
    while bb != 0 {
        let to = pop_lsb(&mut bb);
        emit_scored(
            buf,
            &mut count,
            Move::new(Square((to.0 as i8 - west_dir) as u8), to),
        );
    }

    // Promotion captures (all 4 types)
    let mut bb = cap_east & promo_rank;
    while bb != 0 {
        let to = pop_lsb(&mut bb);
        let from = Square((to.0 as i8 - east_dir) as u8);
        emit_scored(
            buf,
            &mut count,
            Move::new_promotion(from, to, PieceType::Queen),
        );
        emit_scored(
            buf,
            &mut count,
            Move::new_promotion(from, to, PieceType::Rook),
        );
        emit_scored(
            buf,
            &mut count,
            Move::new_promotion(from, to, PieceType::Bishop),
        );
        emit_scored(
            buf,
            &mut count,
            Move::new_promotion(from, to, PieceType::Knight),
        );
    }
    let mut bb = cap_west & promo_rank;
    while bb != 0 {
        let to = pop_lsb(&mut bb);
        let from = Square((to.0 as i8 - west_dir) as u8);
        emit_scored(
            buf,
            &mut count,
            Move::new_promotion(from, to, PieceType::Queen),
        );
        emit_scored(
            buf,
            &mut count,
            Move::new_promotion(from, to, PieceType::Rook),
        );
        emit_scored(
            buf,
            &mut count,
            Move::new_promotion(from, to, PieceType::Bishop),
        );
        emit_scored(
            buf,
            &mut count,
            Move::new_promotion(from, to, PieceType::Knight),
        );
    }

    // Queen push-promotions only (underpromotions go to generate_quiets)
    let empty = !occupied;
    let push_promos = pawn_push_bb(pawns, us) & empty & promo_rank & check_mask;
    let mut bb = push_promos;
    while bb != 0 {
        let to = pop_lsb(&mut bb);
        let from = Square((to.0 as i8 - push_dir) as u8);
        emit_scored(
            buf,
            &mut count,
            Move::new_promotion(from, to, PieceType::Queen),
        );
    }

    // En passant
    if pos.ep_square != Square::NONE {
        let ep = pos.ep_square;
        let mut ep_pawns = pawn_attacks(ep, them) & pawns;
        while ep_pawns != 0 {
            let from = pop_lsb(&mut ep_pawns);
            emit_scored(
                buf,
                &mut count,
                Move::new_with_type(from, ep, MT_EN_PASSANT),
            );
        }
    }

    // Piece captures
    let capture_targets = their_pieces & check_mask;
    for pt in [
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
    ] {
        let mut bb = pos.piece_type_bb(pt, us);
        while bb != 0 {
            let from = pop_lsb(&mut bb);
            let attacks = match pt {
                PieceType::Knight => knight_attacks(from),
                PieceType::Bishop => bishop_attacks(from, occupied),
                PieceType::Rook => rook_attacks(from, occupied),
                PieceType::Queen => queen_attacks(from, occupied),
                _ => unreachable!(),
            };
            let mut moves = attacks & capture_targets;
            while moves != 0 {
                let to = pop_lsb(&mut moves);
                emit_scored(buf, &mut count, Move::new(from, to));
            }
        }
    }

    // King captures
    let mut attacks = king_attacks(ksq) & their_pieces;
    while attacks != 0 {
        let to = pop_lsb(&mut attacks);
        emit_scored(buf, &mut count, Move::new(ksq, to));
    }

    // Post: no generated capture targets a king
    #[cfg(debug_assertions)]
    for i in count_before..count {
        let m = buf[i].mv;
        if m.move_type() != MT_EN_PASSANT && m.move_type() != MT_PROMOTION {
            let dest = pos.board[m.to_sq().index()];
            debug_assert!(
                dest == Piece::NONE || dest.piece_type() != PieceType::King,
                "generate_captures: king capture {} at to={}",
                m.to_uci(),
                m.to_sq().0
            );
        }
    }
    count
}

/// Generate pseudo-legal quiet moves (non-captures, including castling). Returns count.
/// Writes directly into a `ScoredMove` buffer (score = 0, filled later by MovePicker).
/// The caller must filter with `is_legal()`.
#[inline]
pub fn generate_quiets(pos: &Position, buf: &mut ArrayBuf<ScoredMove, MAX_MOVES>) -> usize {
    // SAFETY: the compression writer runs only where its instructions exist: pinned by
    // the build target, or elected from CPUID by `cpu::resolve`.
    unsafe {
        #[cfg(target_arch = "x86_64")]
        {
            if cfg!(all(
                target_feature = "avx512f",
                target_feature = "avx512bw",
                target_feature = "avx512vbmi2"
            )) {
                x86::quiets_compress(pos, buf)
            } else {
                (crate::cpu::get().generate_quiets)(pos, buf)
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            generate_quiets_with::<Serial>(pos, buf)
        }
    }
}

/// [`generate_quiets`] writing one move at a time: the dispatch entry of CPUs without
/// byte compression.
#[cfg(target_arch = "x86_64")]
pub(crate) unsafe fn quiets_serial(
    pos: &Position,
    buf: &mut ArrayBuf<ScoredMove, MAX_MOVES>,
) -> usize {
    // SAFETY: the serial writer uses no instruction beyond the build target.
    unsafe { generate_quiets_with::<Serial>(pos, buf) }
}

/// How the quiet generator writes the moves of a target set: one bit at a time, or the
/// whole set at once. Both write the same moves in the same order.
trait Splat {
    /// Append `Move::new(from, to)` for every `to` of `targets`, lowest square first.
    /// The set holds at most `8 * CHUNKS` squares.
    unsafe fn moves<const CHUNKS: usize>(
        buf: &mut ArrayBuf<ScoredMove, MAX_MOVES>,
        count: &mut usize,
        from: Square,
        targets: u64,
    );

    /// Append `Move::new(to - back, to)` for every `to` of `targets`, lowest square
    /// first: the pushes of at most 8 pawns.
    unsafe fn pushes(
        buf: &mut ArrayBuf<ScoredMove, MAX_MOVES>,
        count: &mut usize,
        back: i8,
        targets: u64,
    );
}

/// One move per bit, the loop exit taken after the last.
struct Serial;

impl Splat for Serial {
    #[inline(always)]
    unsafe fn moves<const CHUNKS: usize>(
        buf: &mut ArrayBuf<ScoredMove, MAX_MOVES>,
        count: &mut usize,
        from: Square,
        mut targets: u64,
    ) {
        while targets != 0 {
            let to = pop_lsb(&mut targets);
            emit_scored(buf, count, Move::new(from, to));
        }
    }

    #[inline(always)]
    unsafe fn pushes(
        buf: &mut ArrayBuf<ScoredMove, MAX_MOVES>,
        count: &mut usize,
        back: i8,
        mut targets: u64,
    ) {
        while targets != 0 {
            let to = pop_lsb(&mut targets);
            emit_scored(buf, count, Move::new(Square((to.0 as i8 - back) as u8), to));
        }
    }
}

/// The moves of every piece of `pieces` to the squares of `targets` it attacks, at most
/// `8 * CHUNKS` per piece.
#[inline(always)]
unsafe fn piece_moves<S: Splat, const CHUNKS: usize>(
    buf: &mut ArrayBuf<ScoredMove, MAX_MOVES>,
    count: &mut usize,
    mut pieces: u64,
    targets: u64,
    attacks: impl Fn(Square) -> u64,
) {
    while pieces != 0 {
        let from = pop_lsb(&mut pieces);
        // SAFETY: the caller vouches for the instructions of `S`.
        unsafe { S::moves::<CHUNKS>(buf, count, from, attacks(from) & targets) };
    }
}

/// [`generate_quiets`] through the writer `S`.
#[inline(always)]
unsafe fn generate_quiets_with<S: Splat>(
    pos: &Position,
    buf: &mut ArrayBuf<ScoredMove, MAX_MOVES>,
) -> usize {
    let mut count = 0;
    let us = pos.side_to_move;
    let occupied = pos.occupied();
    let empty = !occupied;
    let ksq = pos.king_sq(us);

    let check_mask;
    let num_checkers = popcount(pos.checkers);

    if num_checkers > 1 {
        // Double check: only king quiet moves
        // SAFETY (every `S` call below): the caller vouches for the instructions of `S`.
        unsafe { S::moves::<1>(buf, &mut count, ksq, king_attacks(ksq) & empty) };
        return count;
    } else if num_checkers == 1 {
        let checker_sq = lsb(pos.checkers);
        check_mask = between_bb(ksq, checker_sq) | checker_sq.bb();
    } else {
        check_mask = !0u64;
    }

    // Pawn pushes (non-promotions)
    let pawns = pos.piece_type_bb(PieceType::Pawn, us);
    let promo_rank = match us {
        Color::White => RANK_8,
        Color::Black => RANK_1,
    };
    let third_rank = match us {
        Color::White => RANK_3,
        Color::Black => RANK_6,
    };
    let push_dir = pawn_push(us);

    let single_push = pawn_push_bb(pawns, us) & empty;
    unsafe {
        S::pushes(
            buf,
            &mut count,
            push_dir,
            single_push & !promo_rank & check_mask,
        )
    };

    // Double push
    let double_push = pawn_push_bb(single_push & third_rank, us) & empty & check_mask;
    unsafe { S::pushes(buf, &mut count, 2 * push_dir, double_push) };

    // Push underpromotions (R/B/N) — queen already in generate_captures
    let push_promos = pawn_push_bb(pawns, us) & empty & promo_rank & check_mask;
    let mut bb = push_promos;
    while bb != 0 {
        let to = pop_lsb(&mut bb);
        let from = Square((to.0 as i8 - push_dir) as u8);
        emit_scored(
            buf,
            &mut count,
            Move::new_promotion(from, to, PieceType::Rook),
        );
        emit_scored(
            buf,
            &mut count,
            Move::new_promotion(from, to, PieceType::Bishop),
        );
        emit_scored(
            buf,
            &mut count,
            Move::new_promotion(from, to, PieceType::Knight),
        );
    }

    // Piece quiet moves: at most 8 targets for a knight, 13 for a bishop, 14 for a
    // rook and 27 for a queen.
    let quiet_targets = empty & check_mask;
    let piece_bb = |pt| pos.piece_type_bb(pt, us);
    unsafe {
        piece_moves::<S, 1>(
            buf,
            &mut count,
            piece_bb(PieceType::Knight),
            quiet_targets,
            knight_attacks,
        );
        piece_moves::<S, 2>(
            buf,
            &mut count,
            piece_bb(PieceType::Bishop),
            quiet_targets,
            |from| bishop_attacks(from, occupied),
        );
        piece_moves::<S, 2>(
            buf,
            &mut count,
            piece_bb(PieceType::Rook),
            quiet_targets,
            |from| rook_attacks(from, occupied),
        );
        piece_moves::<S, 4>(
            buf,
            &mut count,
            piece_bb(PieceType::Queen),
            quiet_targets,
            |from| queen_attacks(from, occupied),
        );
    }

    // King quiet moves
    unsafe { S::moves::<1>(buf, &mut count, ksq, king_attacks(ksq) & empty) };

    // Castling (only when not in check)
    if num_checkers == 0 {
        generate_castling_scored(pos, buf, &mut count, us, occupied);
    }
    count
}

fn generate_moves(pos: &Position, buf: &mut ArrayBuf<Move, MAX_MOVES>, count: &mut usize) {
    let us = pos.side_to_move;
    let them = !us;
    let our_pieces = pos.color_bb(us);
    let their_pieces = pos.color_bb(them);
    let occupied = pos.occupied();
    let empty = !occupied;
    let ksq = pos.king_sq(us);
    debug_assert!(ksq.0 < 64, "generate_moves: king_sq invalid {}", ksq.0);
    debug_assert!(
        popcount(pos.checkers) <= 2,
        "generate_moves: {} checkers",
        popcount(pos.checkers)
    );
    #[cfg(debug_assertions)]
    let count_before = *count;

    // Determine check mask (what targets are valid)
    let check_mask;
    let num_checkers = popcount(pos.checkers);

    if num_checkers > 1 {
        // Double check: only king moves
        generate_king_moves(pos, buf, count, us, ksq, our_pieces);
        return;
    } else if num_checkers == 1 {
        // Single check: must capture checker or block
        let checker_sq = lsb(pos.checkers);
        // between_bb includes checker sq for aligned (slider) checks,
        // but returns 0 for knight/pawn checks. Always OR in checker's bb.
        check_mask = between_bb(ksq, checker_sq) | checker_sq.bb();
    } else {
        check_mask = !0u64; // no check, all targets valid
    }

    generate_pawn_moves(
        pos,
        buf,
        count,
        us,
        them,
        our_pieces,
        their_pieces,
        occupied,
        empty,
        check_mask,
    );
    generate_piece_moves(
        pos,
        buf,
        count,
        us,
        our_pieces,
        occupied,
        check_mask,
        PieceType::Knight,
    );
    generate_piece_moves(
        pos,
        buf,
        count,
        us,
        our_pieces,
        occupied,
        check_mask,
        PieceType::Bishop,
    );
    generate_piece_moves(
        pos,
        buf,
        count,
        us,
        our_pieces,
        occupied,
        check_mask,
        PieceType::Rook,
    );
    generate_piece_moves(
        pos,
        buf,
        count,
        us,
        our_pieces,
        occupied,
        check_mask,
        PieceType::Queen,
    );
    generate_king_moves(pos, buf, count, us, ksq, our_pieces);

    // Castling (only when not in check)
    if num_checkers == 0 {
        generate_castling(pos, buf, count, us, occupied);
    }

    // Post-generation: no move captures a king (catches movegen bugs)
    #[cfg(debug_assertions)]
    for i in count_before..*count {
        let m = buf[i];
        let dest = pos.board[m.to_sq().index()];
        debug_assert!(
            dest == Piece::NONE || dest.piece_type() != PieceType::King,
            "generate_moves: king capture {} at to={}",
            m.to_uci(),
            m.to_sq().0
        );
    }
}

/// Precomputed masks are passed in rather than recomputed per piece type; they are
/// all bitboards in registers, and a struct would put them behind a pointer.
#[allow(clippy::too_many_arguments)]
fn generate_pawn_moves(
    pos: &Position,
    buf: &mut ArrayBuf<Move, MAX_MOVES>,
    count: &mut usize,
    us: Color,
    them: Color,
    our_pieces: u64,
    their_pieces: u64,
    _occupied: u64,
    empty: u64,
    check_mask: u64,
) {
    let pawns = pos.piece_type_bb(PieceType::Pawn, us) & our_pieces;
    let (promo_rank, third_rank) = match us {
        Color::White => (RANK_8, RANK_3),
        Color::Black => (RANK_1, RANK_6),
    };

    // Single push
    let single_push = pawn_push_bb(pawns, us) & empty;
    let promo_push = single_push & promo_rank & check_mask;
    let normal_push = single_push & !promo_rank & check_mask;

    // Double push
    let double_push = pawn_push_bb(single_push & third_rank, us) & empty & check_mask;

    // Captures
    let cap_east = pawn_attack_east(pawns, us) & their_pieces;
    let cap_west = pawn_attack_west(pawns, us) & their_pieces;
    let promo_cap_east = cap_east & promo_rank & check_mask;
    let promo_cap_west = cap_west & promo_rank & check_mask;
    let normal_cap_east = cap_east & !promo_rank & check_mask;
    let normal_cap_west = cap_west & !promo_rank & check_mask;

    let push_dir = pawn_push(us);

    // Normal single pushes
    let mut bb = normal_push;
    while bb != 0 {
        let to = pop_lsb(&mut bb);
        let from = Square((to.0 as i8 - push_dir) as u8);
        emit(buf, count, Move::new(from, to));
    }

    // Double pushes
    let mut bb = double_push;
    while bb != 0 {
        let to = pop_lsb(&mut bb);
        let from = Square((to.0 as i8 - 2 * push_dir) as u8);
        emit(buf, count, Move::new(from, to));
    }

    // Normal captures east
    let east_dir: i8 = if us == Color::White {
        NORTH_EAST
    } else {
        SOUTH_EAST
    };
    let mut bb = normal_cap_east;
    while bb != 0 {
        let to = pop_lsb(&mut bb);
        let from = Square((to.0 as i8 - east_dir) as u8);
        emit(buf, count, Move::new(from, to));
    }

    // Normal captures west
    let west_dir: i8 = if us == Color::White {
        NORTH_WEST
    } else {
        SOUTH_WEST
    };
    let mut bb = normal_cap_west;
    while bb != 0 {
        let to = pop_lsb(&mut bb);
        let from = Square((to.0 as i8 - west_dir) as u8);
        emit(buf, count, Move::new(from, to));
    }

    // Promotion pushes (4 promotions each)
    let mut bb = promo_push;
    while bb != 0 {
        let to = pop_lsb(&mut bb);
        let from = Square((to.0 as i8 - push_dir) as u8);
        emit(buf, count, Move::new_promotion(from, to, PieceType::Queen));
        emit(buf, count, Move::new_promotion(from, to, PieceType::Rook));
        emit(buf, count, Move::new_promotion(from, to, PieceType::Bishop));
        emit(buf, count, Move::new_promotion(from, to, PieceType::Knight));
    }

    // Promotion captures east
    let mut bb = promo_cap_east;
    while bb != 0 {
        let to = pop_lsb(&mut bb);
        let from = Square((to.0 as i8 - east_dir) as u8);
        emit(buf, count, Move::new_promotion(from, to, PieceType::Queen));
        emit(buf, count, Move::new_promotion(from, to, PieceType::Rook));
        emit(buf, count, Move::new_promotion(from, to, PieceType::Bishop));
        emit(buf, count, Move::new_promotion(from, to, PieceType::Knight));
    }

    // Promotion captures west
    let mut bb = promo_cap_west;
    while bb != 0 {
        let to = pop_lsb(&mut bb);
        let from = Square((to.0 as i8 - west_dir) as u8);
        emit(buf, count, Move::new_promotion(from, to, PieceType::Queen));
        emit(buf, count, Move::new_promotion(from, to, PieceType::Rook));
        emit(buf, count, Move::new_promotion(from, to, PieceType::Bishop));
        emit(buf, count, Move::new_promotion(from, to, PieceType::Knight));
    }

    // En passant
    if pos.ep_square != Square::NONE {
        let ep = pos.ep_square;
        let mut ep_pawns = pawn_attacks(ep, them) & pawns;
        while ep_pawns != 0 {
            let from = pop_lsb(&mut ep_pawns);
            emit(buf, count, Move::new_with_type(from, ep, MT_EN_PASSANT));
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn generate_piece_moves(
    pos: &Position,
    buf: &mut ArrayBuf<Move, MAX_MOVES>,
    count: &mut usize,
    us: Color,
    our_pieces: u64,
    occupied: u64,
    check_mask: u64,
    pt: PieceType,
) {
    let mut bb = pos.piece_type_bb(pt, us);
    let targets = !our_pieces & check_mask;

    while bb != 0 {
        let from = pop_lsb(&mut bb);
        let attacks = match pt {
            PieceType::Knight => knight_attacks(from),
            PieceType::Bishop => bishop_attacks(from, occupied),
            PieceType::Rook => rook_attacks(from, occupied),
            PieceType::Queen => queen_attacks(from, occupied),
            _ => unreachable!(),
        };
        let mut moves = attacks & targets;
        while moves != 0 {
            let to = pop_lsb(&mut moves);
            emit(buf, count, Move::new(from, to));
        }
    }
}

fn generate_king_moves(
    _pos: &Position,
    buf: &mut ArrayBuf<Move, MAX_MOVES>,
    count: &mut usize,
    _us: Color,
    ksq: Square,
    our_pieces: u64,
) {
    let mut attacks = king_attacks(ksq) & !our_pieces;
    while attacks != 0 {
        let to = pop_lsb(&mut attacks);
        emit(buf, count, Move::new(ksq, to));
    }
}

fn can_castle(pos: &Position, us: Color, right: u8, occupied: u64) -> Option<Move> {
    if pos.castling_rights & right == 0 {
        return None;
    }
    let idx = castling_idx(right);
    let rook_from = pos.castling_rook[idx];
    debug_assert!(
        rook_from.0 < 64,
        "can_castle: rook square unset for right {right}"
    );
    if pos.pinned & rook_from.bb() != 0 {
        return None;
    }
    if pos.castling_path[idx] & occupied != 0 {
        return None;
    }
    let ksq = pos.king_sq(us);
    debug_assert!(ksq != rook_from, "can_castle: king and rook overlap");
    Some(Move::new_with_type(ksq, rook_from, MT_CASTLING))
}

fn generate_castling(
    pos: &Position,
    buf: &mut ArrayBuf<Move, MAX_MOVES>,
    count: &mut usize,
    us: Color,
    occupied: u64,
) {
    let (oo, ooo) = castling_rights_for(us);
    for right in [oo, ooo] {
        if let Some(m) = can_castle(pos, us, right, occupied) {
            emit(buf, count, m);
        }
    }
}

fn generate_castling_scored(
    pos: &Position,
    buf: &mut ArrayBuf<ScoredMove, MAX_MOVES>,
    count: &mut usize,
    us: Color,
    occupied: u64,
) {
    let (oo, ooo) = castling_rights_for(us);
    for right in [oo, ooo] {
        if let Some(m) = can_castle(pos, us, right, occupied) {
            emit_scored(buf, count, m);
        }
    }
}

// ============================================================
// Legality check
// ============================================================

pub fn is_legal(pos: &Position, m: Move) -> bool {
    debug_assert!(
        m.from_sq().0 < 64 && m.to_sq().0 < 64,
        "is_legal: squares OOB from={} to={}",
        m.from_sq().0,
        m.to_sq().0
    );
    let us = pos.side_to_move;
    let from = m.from_sq();
    let to = m.to_sq();
    let ksq = pos.king_sq(us);
    debug_assert!(ksq.0 < 64, "is_legal: king_sq invalid {}", ksq.0);
    let pc = pos.board[from.index()];
    debug_assert!(pc != Piece::NONE, "is_legal: source sq {} empty", from.0);
    debug_assert!(pc.color() == us, "is_legal: piece {:?} wrong color", pc);

    if m.move_type() == MT_EN_PASSANT {
        // Simulate the EP capture and check if king is attacked
        let cap_sq = Square((to.0 as i8 - pawn_push(us)) as u8);
        let occ = (pos.occupied() ^ from.bb() ^ cap_sq.bb()) | to.bb();
        let them = !us;
        let their_bq =
            pos.piece_type_bb(PieceType::Bishop, them) | pos.piece_type_bb(PieceType::Queen, them);
        let their_rq =
            pos.piece_type_bb(PieceType::Rook, them) | pos.piece_type_bb(PieceType::Queen, them);
        return (bishop_attacks(ksq, occ) & their_bq) == 0
            && (rook_attacks(ksq, occ) & their_rq) == 0;
    }

    if m.move_type() == MT_CASTLING {
        // Check that king doesn't pass through attacked squares
        let them = !us;
        let right = if to.file() > from.file() {
            if us == Color::White {
                WHITE_OO
            } else {
                BLACK_OO
            }
        } else {
            if us == Color::White {
                WHITE_OOO
            } else {
                BLACK_OOO
            }
        };
        debug_assert!(pos.castling_rights & right != 0);
        debug_assert!(to == pos.castle_rook_sq(right));
        let king_path = pos.castling_king_path[castling_idx(right)];
        // `pos.threats` already sees through our king. It still has the rook, and
        // lifting the rook can only open its own rank — which matters in Chess960
        // alone, where an enemy slider may stand behind it: exactly when the rook
        // is pinned to the king, every square between them being empty.
        let legal = king_path & pos.threats == 0 && pos.pinned & to.bb() == 0;
        debug_assert_eq!(
            legal,
            {
                let occ = pos.occupied() ^ ksq.bb() ^ to.bb();
                let mut path = king_path;
                let mut safe = true;
                while path != 0 {
                    let sq = pop_lsb(&mut path);
                    safe &= attackers_to_color(sq, them, occ, &pos.pieces) == 0;
                }
                safe
            },
            "is_legal: castling {} disagrees with the attack scan",
            m.to_uci_960(true)
        );
        return legal;
    }

    // King move: `pos.threats` holds every square the opponent attacks with our king
    // lifted off the board, so a king never hides behind itself.
    if from == ksq {
        let legal = pos.threats & to.bb() == 0;
        debug_assert_eq!(
            legal,
            attackers_to_color(to, !us, pos.occupied() ^ ksq.bb(), &pos.pieces) == 0,
            "is_legal: king move {} disagrees with the attack scan",
            m.to_uci()
        );
        return legal;
    }

    // Pinned piece: can only move along pin ray
    if pos.pinned & from.bb() != 0 {
        // Move must stay on the line between king and piece
        return line_bb(ksq, from) & to.bb() != 0;
    }

    // All other moves are legal (check_mask already restricts targets)
    true
}

/// Whether a move read from the transposition table can be played here.
///
/// The table is shared, lock-free and keyed on 16 bits: its move may belong to another
/// position. [`is_legal`] assumes a pseudo-legal move and answers `true` by default, so
/// the two tests are needed together, in this order.
pub fn is_playable_table_move(pos: &Position, m: Move) -> bool {
    is_pseudo_legal(pos, m) && is_legal(pos, m)
}

/// Check that a move is pseudo-legal for the current position.
///
/// Used to validate TT moves which may come from hash collisions. Verifies
/// piece existence, color, move type consistency, and attack geometry.
/// Does NOT check legality (pins, check evasion) — call [`is_legal`] after.
pub fn is_pseudo_legal(pos: &Position, m: Move) -> bool {
    if m == Move::NONE || m == Move::NULL {
        return false;
    }
    debug_assert!(
        m.from_sq().0 < 64 && m.to_sq().0 < 64,
        "is_pseudo_legal: squares OOB from={} to={}",
        m.from_sq().0,
        m.to_sq().0
    );

    let us = pos.side_to_move;
    let them = !us;
    let from = m.from_sq();
    let to = m.to_sq();
    let pc = pos.board[from.index()];
    let mt = m.move_type();

    // Must have a piece of our color on the from square
    if pc == Piece::NONE || pc.color() != us {
        return false;
    }

    // Castling: delegate to specific checks (rights, path clear, dest is the rook)
    if mt == MT_CASTLING {
        if pc.piece_type() != PieceType::King {
            return false;
        }
        let right = if to.file() > from.file() {
            if us == Color::White {
                WHITE_OO
            } else {
                BLACK_OO
            }
        } else {
            if us == Color::White {
                WHITE_OOO
            } else {
                BLACK_OOO
            }
        };
        if pos.castling_rights & right == 0 {
            return false;
        }
        if to != pos.castle_rook_sq(right) {
            return false;
        }
        let idx = castling_idx(right);
        if pos.pinned & to.bb() != 0 {
            return false;
        }
        return pos.castling_path[idx] & pos.occupied() == 0;
    }

    // Destination must not have our own piece
    let dest_pc = pos.board[to.index()];
    if dest_pc != Piece::NONE && dest_pc.color() == us {
        return false;
    }

    // Must not capture a king (TT collision safety)
    if dest_pc != Piece::NONE && dest_pc.piece_type() == PieceType::King {
        return false;
    }

    // Check evasion validation (BEFORE move-type handlers that return early)
    let num_checkers = popcount(pos.checkers);
    if num_checkers > 1 && pc.piece_type() != PieceType::King {
        return false;
    }
    if num_checkers == 1 && pc.piece_type() != PieceType::King {
        let checker_sq = lsb(pos.checkers);
        let check_mask = between_bb(pos.king_sq(us), checker_sq) | checker_sq.bb();
        if mt == MT_EN_PASSANT {
            // Only the EP square has a pawn behind it: any other target would put the
            // captured square off the board (and it is refused below anyway).
            if to != pos.ep_square {
                return false;
            }
            // EP: also accept if captured pawn IS the checker
            let cap_sq = Square((to.0 as i8 - pawn_push(us)) as u8);
            if to.bb() & check_mask == 0 && cap_sq.bb() & pos.checkers == 0 {
                return false;
            }
        } else if to.bb() & check_mask == 0 {
            return false;
        }
    }

    // En passant: destination must be the EP square, piece must be pawn
    if mt == MT_EN_PASSANT {
        return pc.piece_type() == PieceType::Pawn
            && to == pos.ep_square
            && (pawn_attacks(to, them) & from.bb()) != 0;
    }

    // Promotion: piece must be pawn
    if mt == MT_PROMOTION {
        if pc.piece_type() != PieceType::Pawn {
            return false;
        }
        let promo_rank = if us == Color::White { 7u8 } else { 0u8 };
        if to.rank() != promo_rank {
            return false;
        }
        // Push-promotion (one square ahead) or capture-promotion (a square the pawn
        // attacks). Both put the pawn on the rank before the last: a2b8 is no promotion.
        if from.file() == to.file() {
            // push: target must be empty
            return to.0 as i8 == from.0 as i8 + pawn_push(us) && dest_pc == Piece::NONE;
        } else {
            // capture: target must be enemy
            return (pawn_attacks(from, us) & to.bb()) != 0
                && dest_pc != Piece::NONE
                && dest_pc.color() == them;
        }
    }

    // Pawn moves (normal type only, EP/promo handled above)
    if pc.piece_type() == PieceType::Pawn {
        let push = pawn_push(us);
        let promo_rank = if us == Color::White { RANK_8 } else { RANK_1 };
        // Pawn on promo rank shouldn't have MT_NORMAL
        if to.bb() & promo_rank != 0 {
            return false;
        }
        // Capture
        if from.file() != to.file() {
            return (pawn_attacks(from, us) & to.bb()) != 0
                && dest_pc != Piece::NONE
                && dest_pc.color() == them;
        }
        // Single push
        if to.0 as i8 == from.0 as i8 + push {
            return dest_pc == Piece::NONE;
        }
        // Double push
        if to.0 as i8 == from.0 as i8 + 2 * push {
            let intermediate = Square((from.0 as i8 + push) as u8);
            return from.bb() & (if us == Color::White { RANK_2 } else { RANK_7 }) != 0
                && pos.board[intermediate.index()] == Piece::NONE
                && dest_pc == Piece::NONE;
        }
        return false;
    }

    // King moves
    if pc.piece_type() == PieceType::King {
        return (king_attacks(from) & to.bb()) != 0;
    }

    // Knight, bishop, rook, queen: check attack geometry
    let occupied = pos.occupied();
    let attacks = match pc.piece_type() {
        PieceType::Knight => knight_attacks(from),
        PieceType::Bishop => bishop_attacks(from, occupied),
        PieceType::Rook => rook_attacks(from, occupied),
        PieceType::Queen => queen_attacks(from, occupied),
        _ => unreachable!(),
    };
    (attacks & to.bb()) != 0
}

// ============================================================
// x86-64: the quiet moves written by byte compression
// ============================================================

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)]
pub(crate) mod x86 {
    use super::*;
    use std::arch::x86_64::*;

    // An entry is the 64-bit lane `move << 32`: the score (0) in the low half, the move
    // in the high one, as `ScoredMove` lays them out.
    const _: () = assert!(std::mem::size_of::<ScoredMove>() == 8);
    const _: () = assert!(std::mem::offset_of!(ScoredMove, score) == 0);
    const _: () = assert!(std::mem::offset_of!(ScoredMove, mv) == 4);
    const _: () = assert!(std::mem::size_of::<Move>() == 2);

    #[repr(align(64))]
    struct Lanes([u8; 64]);

    /// Byte `i` holds `i`: compressed by a bitboard, the squares of its bits in order.
    static ASCENDING: Lanes = {
        let mut s = [0u8; 64];
        let mut i = 0usize;
        while i < 64 {
            s[i] = i as u8;
            i += 1;
        }
        Lanes(s)
    };

    /// The whole target set at once: its squares compressed into the low bytes of a
    /// register, widened 8 at a time to entries, and stored under a mask that ends at
    /// the last move. No slot past the count is written, so the bad captures waiting at
    /// the far end of the move picker's buffer are safe, and no exit depends on the
    /// size of the set.
    struct Compress;

    /// Append one move per square of `targets`, lowest first: `to + k`, or for a pawn
    /// push (`PUSH`) `to * 65 + k`, which puts `to - back` in the origin field when
    /// `k = -64 * back`.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi2")]
    unsafe fn store<const CHUNKS: usize, const PUSH: bool>(
        buf: &mut ArrayBuf<ScoredMove, MAX_MOVES>,
        count: &mut usize,
        targets: u64,
        k: i64,
    ) {
        const { assert!(CHUNKS >= 1 && CHUNKS <= 4) };
        let n = targets.count_ones() as usize;
        debug_assert!(n <= 8 * CHUNKS, "store: {n} moves for {CHUNKS} chunks");
        debug_assert!(*count + n <= MAX_MOVES, "store: overflow at {}", *count);
        let squares =
            _mm512_maskz_compress_epi8(targets, _mm512_load_si512(ASCENDING.0.as_ptr().cast()));
        let low = _mm512_castsi512_si128(squares);
        let high = _mm512_extracti32x4_epi32::<1>(squares);
        let k = _mm512_set1_epi64(k);
        let live = (1u64 << n) - 1;
        // Chunks past the last move are stored under an empty mask: their address is
        // computed but never touched.
        let dst = buf.as_mut_ptr().wrapping_add(*count).cast::<i64>();
        for c in 0..CHUNKS {
            let bytes = match c {
                0 => low,
                1 => _mm_unpackhi_epi64(low, low),
                2 => high,
                _ => _mm_unpackhi_epi64(high, high),
            };
            let to = _mm512_cvtepu8_epi64(bytes);
            let mv = if PUSH {
                _mm512_add_epi64(_mm512_slli_epi64::<6>(to), to)
            } else {
                to
            };
            let entries = _mm512_slli_epi64::<32>(_mm512_add_epi64(mv, k));
            _mm512_mask_storeu_epi64(
                dst.wrapping_add(8 * c),
                (live >> (8 * c)) as __mmask8,
                entries,
            );
        }
        *count += n;
    }

    impl Splat for Compress {
        #[inline]
        #[target_feature(enable = "avx512f,avx512bw,avx512vbmi2")]
        unsafe fn moves<const CHUNKS: usize>(
            buf: &mut ArrayBuf<ScoredMove, MAX_MOVES>,
            count: &mut usize,
            from: Square,
            targets: u64,
        ) {
            store::<CHUNKS, false>(buf, count, targets, (from.0 as i64) << 6);
        }

        #[inline]
        #[target_feature(enable = "avx512f,avx512bw,avx512vbmi2")]
        unsafe fn pushes(
            buf: &mut ArrayBuf<ScoredMove, MAX_MOVES>,
            count: &mut usize,
            back: i8,
            targets: u64,
        ) {
            store::<1, true>(buf, count, targets, -((back as i64) << 6));
        }
    }

    /// [`generate_quiets`] writing each target set by compression.
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi2")]
    pub(crate) unsafe fn quiets_compress(
        pos: &Position,
        buf: &mut ArrayBuf<ScoredMove, MAX_MOVES>,
    ) -> usize {
        generate_quiets_with::<Compress>(pos, buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A move the table may hold for another position is refused before [`is_legal`]
    /// sees it: that test assumes a pseudo-legal move and would pass this one (the
    /// refutation line once published it as is).
    #[test]
    fn a_table_move_from_another_position_is_not_playable() {
        // Black to move: e2e4 is White's pawn push, and Black is in check besides.
        let pos =
            Position::from_fen("rnbqkbnr/ppppp1pp/8/5p1Q/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 1 2")
                .unwrap();
        assert!(!is_playable_table_move(&pos, Move::new(Square::E2, Square::E4)));
        assert!(!is_playable_table_move(&pos, Move::NONE));
        // A quiet move that ignores the check is pseudo-legal for no one here.
        assert!(!is_playable_table_move(&pos, Move::new(Square::A7, Square::A6)));
        // The one move that is legal.
        assert!(is_playable_table_move(&pos, Move::new(Square::G7, Square::G6)));
    }

    type QuietWriter = unsafe fn(&Position, &mut ArrayBuf<ScoredMove, MAX_MOVES>) -> usize;

    unsafe fn quiets_one_by_one(
        pos: &Position,
        buf: &mut ArrayBuf<ScoredMove, MAX_MOVES>,
    ) -> usize {
        unsafe { generate_quiets_with::<Serial>(pos, buf) }
    }

    /// Every quiet writer this machine can run, the serial one first.
    fn quiet_writers() -> Vec<(&'static str, QuietWriter)> {
        #[cfg_attr(not(target_arch = "x86_64"), allow(unused_mut))]
        let mut v: Vec<(&'static str, QuietWriter)> = vec![("serial", quiets_one_by_one)];
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vbmi2")
        {
            v.push(("compress", x86::quiets_compress));
        }
        v
    }

    /// A slot no generator writes: tells a write past the count from a clean buffer.
    const UNTOUCHED: ScoredMove = ScoredMove {
        score: 0x5EED,
        mv: Move(0xFFFF),
    };

    /// Every quiet writer gives the same moves in the same order and leaves the slots
    /// past its count alone; with the captures, the legal ones are exactly the legal
    /// moves, each once.
    fn check_position(pos: &Position, writers: &[(&'static str, QuietWriter)]) {
        let fen = pos.to_fen();
        let mut reference: Option<Vec<Move>> = None;
        for &(name, write) in writers {
            let mut buf = ArrayBuf::<ScoredMove, MAX_MOVES>::new();
            for i in 0..MAX_MOVES {
                buf.set(i, UNTOUCHED);
            }
            // SAFETY: `quiet_writers` lists only what this CPU runs.
            let n = unsafe { write(pos, &mut buf) };
            let moves: Vec<Move> = (0..n).map(|i| buf[i].mv).collect();
            assert!(
                (0..n).all(|i| buf[i].score == 0),
                "{fen}: {name} left a score"
            );
            for i in n..MAX_MOVES {
                assert!(
                    buf[i].score == UNTOUCHED.score && buf[i].mv == UNTOUCHED.mv,
                    "{fen}: {name} wrote slot {i} past its count {n}"
                );
            }
            match &reference {
                None => reference = Some(moves),
                Some(r) => assert!(*r == moves, "{fen}: {name} differs from the serial writer"),
            }
        }

        let mut caps = ArrayBuf::<ScoredMove, MAX_MOVES>::new();
        let nc = generate_captures(pos, &mut caps);
        let mut generated: Vec<u16> = (0..nc)
            .map(|i| caps[i].mv)
            .chain(reference.unwrap())
            .filter(|&m| is_legal(pos, m))
            .map(|m| m.0)
            .collect();
        let mut legal = ArrayBuf::<Move, MAX_MOVES>::new();
        let nl = generate_legal_moves(pos, &mut legal);
        let mut expected: Vec<u16> = legal.filled(nl).iter().map(|m| m.0).collect();
        generated.sort_unstable();
        expected.sort_unstable();
        assert_eq!(
            generated, expected,
            "{fen}: captures and quiets against the legal moves"
        );
    }

    #[test]
    fn the_quiet_writers_agree_and_the_stages_cover_the_legal_moves() {
        let writers = quiet_writers();
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            // Promotions by push and by capture, both sides.
            "1n2k2r/P6P/8/8/8/8/6K1/8 w k - 0 1",
            "8/6k1/8/8/8/8/p6p/1N2K2R b K - 0 1",
            // Single and double check.
            "4k3/8/8/8/8/8/8/r3K3 w - - 0 1",
            "4k3/8/8/8/8/5n2/8/4K2r w - - 0 1",
            // The most moves on record (218), every slot of a queen's four chunks used.
            "R6R/3Q4/1Q4Q1/4Q3/2Q4Q/Q4Q2/pp1Q4/kBNN1KB1 w - - 0 1",
        ];
        for fen in fens {
            check_position(&Position::from_fen(fen).unwrap(), &writers);
        }

        // Random games reach what the fixed positions miss.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        for start in [fens[0], fens[1], fens[2]] {
            for _ in 0..8 {
                let mut pos = Position::from_fen(start).unwrap();
                for _ in 0..200 {
                    check_position(&pos, &writers);
                    let mut legal = ArrayBuf::<Move, MAX_MOVES>::new();
                    let n = generate_legal_moves(&pos, &mut legal);
                    if n == 0 {
                        break;
                    }
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    pos.make_move(legal[(state % n as u64) as usize]);
                }
            }
        }
    }
}
