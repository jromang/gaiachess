//! [Pseudo-legal](https://www.chessprogramming.org/Pseudo-Legal_Move) move generation
//! with legality filtering.
//!
//! Generates all moves that obey piece movement rules but may leave the king in check,
//! then filters illegal ones. Double check forces king moves only. Single check
//! restricts targets to interpositions or captures of the checker. Pinned pieces
//! may only move along their pin ray.

use crate::bitboard::*;
use crate::types::*;
use crate::position::Position;

/// Append a move to the raw buffer.
#[inline(always)]
fn emit(buf: &mut ArrayBuf<Move, MAX_MOVES>, count: &mut usize, m: Move) {
    debug_assert!(*count < MAX_MOVES, "emit: overflow at {}", *count);
    debug_assert!(m.is_ok(), "emit: invalid move {:?}", m);
    buf[*count] = m;
    *count += 1;
}

/// Append a move to a scored buffer (score = 0, to be filled later by MovePicker).
#[inline(always)]
fn emit_scored(buf: &mut ArrayBuf<ScoredMove, MAX_MOVES>, count: &mut usize, m: Move) {
    debug_assert!(*count < MAX_MOVES, "emit_scored: overflow at {}", *count);
    debug_assert!(m.is_ok(), "emit_scored: invalid move {:?}", m);
    buf[*count] = ScoredMove { score: 0, mv: m };
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
        debug_assert!(is_legal(pos, buf[i]),
            "generate_legal_moves: illegal move {} survived filtering", buf[i].to_uci());
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
    let east_dir: i8 = if us == Color::White { NORTH_EAST } else { SOUTH_EAST };
    let west_dir: i8 = if us == Color::White { NORTH_WEST } else { SOUTH_WEST };

    // Pawn captures
    let cap_east = pawn_attack_east(pawns, us) & their_pieces & check_mask;
    let cap_west = pawn_attack_west(pawns, us) & their_pieces & check_mask;

    // Non-promo captures
    let mut bb = cap_east & !promo_rank;
    while bb != 0 {
        let to = pop_lsb(&mut bb);
        emit_scored(buf, &mut count, Move::new(Square((to.0 as i8 - east_dir) as u8), to));
    }
    let mut bb = cap_west & !promo_rank;
    while bb != 0 {
        let to = pop_lsb(&mut bb);
        emit_scored(buf, &mut count, Move::new(Square((to.0 as i8 - west_dir) as u8), to));
    }

    // Promotion captures (all 4 types)
    let mut bb = cap_east & promo_rank;
    while bb != 0 {
        let to = pop_lsb(&mut bb);
        let from = Square((to.0 as i8 - east_dir) as u8);
        emit_scored(buf, &mut count, Move::new_promotion(from, to, PieceType::Queen));
        emit_scored(buf, &mut count, Move::new_promotion(from, to, PieceType::Rook));
        emit_scored(buf, &mut count, Move::new_promotion(from, to, PieceType::Bishop));
        emit_scored(buf, &mut count, Move::new_promotion(from, to, PieceType::Knight));
    }
    let mut bb = cap_west & promo_rank;
    while bb != 0 {
        let to = pop_lsb(&mut bb);
        let from = Square((to.0 as i8 - west_dir) as u8);
        emit_scored(buf, &mut count, Move::new_promotion(from, to, PieceType::Queen));
        emit_scored(buf, &mut count, Move::new_promotion(from, to, PieceType::Rook));
        emit_scored(buf, &mut count, Move::new_promotion(from, to, PieceType::Bishop));
        emit_scored(buf, &mut count, Move::new_promotion(from, to, PieceType::Knight));
    }

    // Queen push-promotions only (underpromotions go to generate_quiets)
    let empty = !occupied;
    let push_promos = pawn_push_bb(pawns, us) & empty & promo_rank & check_mask;
    let mut bb = push_promos;
    while bb != 0 {
        let to = pop_lsb(&mut bb);
        let from = Square((to.0 as i8 - push_dir) as u8);
        emit_scored(buf, &mut count, Move::new_promotion(from, to, PieceType::Queen));
    }

    // En passant
    if pos.ep_square != Square::NONE {
        let ep = pos.ep_square;
        let mut ep_pawns = pawn_attacks(ep, them) & pawns;
        while ep_pawns != 0 {
            let from = pop_lsb(&mut ep_pawns);
            emit_scored(buf, &mut count, Move::new_with_type(from, ep, MT_EN_PASSANT));
        }
    }

    // Piece captures
    let capture_targets = their_pieces & check_mask;
    for pt in [PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen] {
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
            debug_assert!(dest == Piece::NONE || dest.piece_type() != PieceType::King,
                "generate_captures: king capture {} at to={}", m.to_uci(), m.to_sq().0);
        }
    }
    count
}

/// Generate pseudo-legal quiet moves (non-captures, including castling). Returns count.
/// Writes directly into a `ScoredMove` buffer (score = 0, filled later by MovePicker).
/// The caller must filter with `is_legal()`.
pub fn generate_quiets(pos: &Position, buf: &mut ArrayBuf<ScoredMove, MAX_MOVES>) -> usize {
    let mut count = 0;
    let us = pos.side_to_move;
    let occupied = pos.occupied();
    let empty = !occupied;
    let ksq = pos.king_sq(us);

    let check_mask;
    let num_checkers = popcount(pos.checkers);

    if num_checkers > 1 {
        // Double check: only king quiet moves
        let mut attacks = king_attacks(ksq) & empty;
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
    let mut bb = single_push & !promo_rank & check_mask;
    while bb != 0 {
        let to = pop_lsb(&mut bb);
        emit_scored(buf, &mut count, Move::new(Square((to.0 as i8 - push_dir) as u8), to));
    }

    // Double push
    let mut bb = pawn_push_bb(single_push & third_rank, us) & empty & check_mask;
    while bb != 0 {
        let to = pop_lsb(&mut bb);
        emit_scored(buf, &mut count, Move::new(Square((to.0 as i8 - 2 * push_dir) as u8), to));
    }

    // Push underpromotions (R/B/N) — queen already in generate_captures
    let push_promos = pawn_push_bb(pawns, us) & empty & promo_rank & check_mask;
    let mut bb = push_promos;
    while bb != 0 {
        let to = pop_lsb(&mut bb);
        let from = Square((to.0 as i8 - push_dir) as u8);
        emit_scored(buf, &mut count, Move::new_promotion(from, to, PieceType::Rook));
        emit_scored(buf, &mut count, Move::new_promotion(from, to, PieceType::Bishop));
        emit_scored(buf, &mut count, Move::new_promotion(from, to, PieceType::Knight));
    }

    // Piece quiet moves
    let quiet_targets = empty & check_mask;
    for pt in [PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen] {
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
            let mut moves = attacks & quiet_targets;
            while moves != 0 {
                let to = pop_lsb(&mut moves);
                emit_scored(buf, &mut count, Move::new(from, to));
            }
        }
    }

    // King quiet moves
    let mut attacks = king_attacks(ksq) & empty;
    while attacks != 0 {
        let to = pop_lsb(&mut attacks);
        emit_scored(buf, &mut count, Move::new(ksq, to));
    }

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
    debug_assert!(popcount(pos.checkers) <= 2,
        "generate_moves: {} checkers", popcount(pos.checkers));
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

    generate_pawn_moves(pos, buf, count, us, them, our_pieces, their_pieces, occupied, empty, check_mask);
    generate_piece_moves(pos, buf, count, us, our_pieces, occupied, check_mask, PieceType::Knight);
    generate_piece_moves(pos, buf, count, us, our_pieces, occupied, check_mask, PieceType::Bishop);
    generate_piece_moves(pos, buf, count, us, our_pieces, occupied, check_mask, PieceType::Rook);
    generate_piece_moves(pos, buf, count, us, our_pieces, occupied, check_mask, PieceType::Queen);
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
        debug_assert!(dest == Piece::NONE || dest.piece_type() != PieceType::King,
            "generate_moves: king capture {} at to={}", m.to_uci(), m.to_sq().0);
    }
}

/// Precomputed masks are passed in rather than recomputed per piece type; they are
/// all bitboards in registers, and a struct would put them behind a pointer.
#[allow(clippy::too_many_arguments)]
fn generate_pawn_moves(
    pos: &Position, buf: &mut ArrayBuf<Move, MAX_MOVES>, count: &mut usize,
    us: Color, them: Color,
    our_pieces: u64, their_pieces: u64,
    _occupied: u64, empty: u64, check_mask: u64,
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
    let east_dir: i8 = if us == Color::White { NORTH_EAST } else { SOUTH_EAST };
    let mut bb = normal_cap_east;
    while bb != 0 {
        let to = pop_lsb(&mut bb);
        let from = Square((to.0 as i8 - east_dir) as u8);
        emit(buf, count, Move::new(from, to));
    }

    // Normal captures west
    let west_dir: i8 = if us == Color::White { NORTH_WEST } else { SOUTH_WEST };
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
    pos: &Position, buf: &mut ArrayBuf<Move, MAX_MOVES>, count: &mut usize,
    us: Color, our_pieces: u64, occupied: u64,
    check_mask: u64, pt: PieceType,
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
    _pos: &Position, buf: &mut ArrayBuf<Move, MAX_MOVES>, count: &mut usize,
    _us: Color, ksq: Square, our_pieces: u64,
) {
    let mut attacks = king_attacks(ksq) & !our_pieces;
    while attacks != 0 {
        let to = pop_lsb(&mut attacks);
        emit(buf, count, Move::new(ksq, to));
    }
}

fn generate_castling(
    pos: &Position, buf: &mut ArrayBuf<Move, MAX_MOVES>, count: &mut usize,
    us: Color, occupied: u64,
) {
    let rights = pos.castling_rights;
    let (oo, ooo) = match us {
        Color::White => (WHITE_OO, WHITE_OOO),
        Color::Black => (BLACK_OO, BLACK_OOO),
    };
    let ksq = pos.king_sq(us);

    for right in [oo, ooo] {
        if rights & right == 0 {
            continue;
        }
        let path = CASTLING_PATH[right as usize];
        if path & occupied != 0 {
            continue; // path not clear
        }
        let data = &CASTLING_DATA[right as usize];
        emit(buf, count, Move::new_with_type(ksq, data.king_to, MT_CASTLING));
    }
}

fn generate_castling_scored(
    pos: &Position, buf: &mut ArrayBuf<ScoredMove, MAX_MOVES>, count: &mut usize,
    us: Color, occupied: u64,
) {
    let rights = pos.castling_rights;
    let (oo, ooo) = match us {
        Color::White => (WHITE_OO, WHITE_OOO),
        Color::Black => (BLACK_OO, BLACK_OOO),
    };
    let ksq = pos.king_sq(us);

    for right in [oo, ooo] {
        if rights & right == 0 {
            continue;
        }
        let path = CASTLING_PATH[right as usize];
        if path & occupied != 0 {
            continue; // path not clear
        }
        let data = &CASTLING_DATA[right as usize];
        emit_scored(buf, count, Move::new_with_type(ksq, data.king_to, MT_CASTLING));
    }
}

// ============================================================
// Legality check
// ============================================================

pub fn is_legal(pos: &Position, m: Move) -> bool {
    debug_assert!(m.from_sq().0 < 64 && m.to_sq().0 < 64,
        "is_legal: squares OOB from={} to={}", m.from_sq().0, m.to_sq().0);
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
        let their_bq = pos.piece_type_bb(PieceType::Bishop, them)
            | pos.piece_type_bb(PieceType::Queen, them);
        let their_rq = pos.piece_type_bb(PieceType::Rook, them)
            | pos.piece_type_bb(PieceType::Queen, them);
        return (bishop_attacks(ksq, occ) & their_bq) == 0
            && (rook_attacks(ksq, occ) & their_rq) == 0;
    }

    if m.move_type() == MT_CASTLING {
        // Check that king doesn't pass through attacked squares
        let them = !us;
        let right = if to.file() > from.file() {
            if us == Color::White { WHITE_OO } else { BLACK_OO }
        } else {
            if us == Color::White { WHITE_OOO } else { BLACK_OOO }
        };
        let king_path = KING_CASTLING_PATH[right as usize];
        // Check each square on king's path
        let mut path = king_path;
        // Remove king from occupied for slider attack calculation
        let occ = pos.occupied() ^ ksq.bb();
        while path != 0 {
            let sq = pop_lsb(&mut path);
            if attackers_to_color(sq, them, occ, &pos.pieces) != 0 {
                return false;
            }
        }
        return true;
    }

    // King move: check destination not attacked
    if from == ksq {
        let them = !us;
        let occ = pos.occupied() ^ ksq.bb(); // remove king to see through
        return attackers_to_color(to, them, occ, &pos.pieces) == 0;
    }

    // Pinned piece: can only move along pin ray
    if pos.pinned & from.bb() != 0 {
        // Move must stay on the line between king and piece
        return line_bb(ksq, from) & to.bb() != 0;
    }

    // All other moves are legal (check_mask already restricts targets)
    true
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
    debug_assert!(m.from_sq().0 < 64 && m.to_sq().0 < 64,
        "is_pseudo_legal: squares OOB from={} to={}", m.from_sq().0, m.to_sq().0);

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

    // Castling: delegate to specific checks (rights, path clear)
    if mt == MT_CASTLING {
        if pc.piece_type() != PieceType::King {
            return false;
        }
        let right = if to.file() > from.file() {
            if us == Color::White { WHITE_OO } else { BLACK_OO }
        } else {
            if us == Color::White { WHITE_OOO } else { BLACK_OOO }
        };
        if pos.castling_rights & right == 0 {
            return false;
        }
        return CASTLING_PATH[right as usize] & pos.occupied() == 0;
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
        // Push-promotion (same file) or capture-promotion (adjacent file)
        if from.file() == to.file() {
            // push: target must be empty
            return dest_pc == Piece::NONE;
        } else {
            // capture: target must be enemy
            return dest_pc != Piece::NONE && dest_pc.color() == them;
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
