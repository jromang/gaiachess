//! [Static Exchange Evaluation](https://www.chessprogramming.org/Static_Exchange_Evaluation)
//! using the swap algorithm.

use crate::bitboard::{attackers_to, between_bb, bishop_attacks, more_than_one, pop_lsb, rook_attacks};
use crate::position::Position;
use crate::types::*;

/// The pieces of `us` pinned to their king, and the enemy sliders pinning them.
///
/// Only the pieces of the king's own side: a pinned enemy piece blocking a check is not
/// a recapture question.
fn king_blockers(pos: &Position, us: Color) -> (u64, u64) {
    let ksq = pos.king_sq(us);
    let them = !us;
    let occ = pos.occupancies[2];
    let queens = pos.pieces[Piece::new(PieceType::Queen, them).index()];
    let mut snipers = (bishop_attacks(ksq, 0)
        & (pos.pieces[Piece::new(PieceType::Bishop, them).index()] | queens))
        | (rook_attacks(ksq, 0)
            & (pos.pieces[Piece::new(PieceType::Rook, them).index()] | queens));
    let (mut blockers, mut pinners) = (0u64, 0u64);
    while snipers != 0 {
        let sniper = pop_lsb(&mut snipers);
        let between = between_bb(ksq, sniper) & occ & !sniper.bb();
        if between != 0 && !more_than_one(between) && between & pos.occupancies[us as usize] != 0 {
            blockers |= between;
            pinners |= sniper.bb();
        }
    }
    (blockers, pinners)
}

/// SEE piece values (indexed by `PieceType`: Pawn=0..King=5, None=6).
const SEE_VALUE: [i32; 7] = [100, 320, 330, 500, 900, 20000, 0];

/// Returns `true` if the static exchange evaluation of `m` is >= `threshold`.
///
/// Uses the swap algorithm: iteratively simulates captures on the target square,
/// alternating sides, picking the least valuable attacker each time.
/// X-ray attacks are revealed as pieces are removed from the board.
///
/// The chain that finds the least valuable attacker assigns inside its conditions on
/// purpose: the whole point of the chain is that a piece type's mask is only looked at
/// once every cheaper type has come up empty. Hoisting the bindings above the chain, as
/// the lint would have it, would compute all six of them every time round the loop.
#[allow(clippy::blocks_in_conditions)]
pub fn see(pos: &Position, m: Move, threshold: i32) -> bool {
    debug_assert!(
        m.from_sq().0 < 64 && m.to_sq().0 < 64,
        "SEE: squares OOB from={} to={}",
        m.from_sq().0,
        m.to_sq().0
    );
    debug_assert!(
        pos.board[m.from_sq().index()] != Piece::NONE,
        "SEE: no piece on from {}",
        m.from_sq().0
    );
    debug_assert!(
        threshold.abs() < SCORE_MATE,
        "SEE: threshold {} suspiciously large",
        threshold
    );

    // Castling and promotions are almost always good — skip SEE.
    // En passant is not: it is a plain pawn-takes-pawn exchange whose victim sits
    // behind the target square, so it is resolved below like any capture. Letting it
    // through unconditionally made it pass any threshold, however far above the one
    // pawn it can win (a ProbCut threshold of 156 among others).
    let mt = m.move_type();
    if mt != MT_NORMAL && mt != MT_EN_PASSANT {
        return true;
    }

    let from = m.from_sq();
    let to = m.to_sq();

    // Square of the piece actually taken: the target square, except en passant.
    let victim_sq = if mt == MT_EN_PASSANT {
        if pos.side_to_move == Color::White {
            Square(to.0 - 8)
        } else {
            Square(to.0 + 8)
        }
    } else {
        to
    };
    debug_assert!(
        mt != MT_EN_PASSANT
            || (pos.board[to.index()] == Piece::NONE
                && pos.board[victim_sq.index()]
                    == Piece::new(PieceType::Pawn, !pos.side_to_move)),
        "SEE: en passant {} without an enemy pawn behind an empty target",
        m.to_uci()
    );

    // Best case: we capture the target piece for free
    let captured = pos.board[victim_sq.index()];
    let captured_val = if captured != Piece::NONE {
        SEE_VALUE[captured.piece_type() as usize]
    } else {
        0
    };

    let mut v = captured_val - threshold;
    if v < 0 {
        return false;
    }

    // Worst case: we lose our piece
    let moving_pt = pos.board[from.index()].piece_type();
    v = SEE_VALUE[moving_pt as usize] - v;
    if v <= 0 {
        return true;
    }

    // Set up occupancy after the initial capture. En passant empties the victim's
    // square, which can open a line onto the target square.
    let mut occ = pos.occupancies[2] ^ from.bb() ^ victim_sq.bb();
    let mut attackers = attackers_to(to, occ, &pos.pieces);

    let diag = pos.pieces[Piece::WHITE_BISHOP.index()]
        | pos.pieces[Piece::BLACK_BISHOP.index()]
        | pos.pieces[Piece::WHITE_QUEEN.index()]
        | pos.pieces[Piece::BLACK_QUEEN.index()];
    let orth = pos.pieces[Piece::WHITE_ROOK.index()]
        | pos.pieces[Piece::BLACK_ROOK.index()]
        | pos.pieces[Piece::WHITE_QUEEN.index()]
        | pos.pieces[Piece::BLACK_QUEEN.index()];

    let mut stm = pos.side_to_move;
    let mut result = 1i32;

    // A piece pinned to its king cannot recapture while its pinner stands. Computed
    // here, once the exchange is known to matter: [white king, black king].
    let pins = [king_blockers(pos, Color::White), king_blockers(pos, Color::Black)];

    loop {
        stm = !stm;
        attackers &= occ;

        let mut mine = attackers & pos.occupancies[stm as usize];
        let (blockers, pinners) = pins[stm as usize];
        if pinners & occ != 0 {
            mine &= !blockers;
        }
        if mine == 0 {
            break;
        }

        result ^= 1;

        // Try each piece type from least to most valuable
        let mut bb;
        if {
            bb = mine & pos.pieces[Piece::new(PieceType::Pawn, stm).index()];
            bb != 0
        } {
            v = SEE_VALUE[PieceType::Pawn as usize] - v;
            if v < result {
                break;
            }
            occ ^= bb & bb.wrapping_neg();
            attackers |= bishop_attacks(to, occ) & diag;
        } else if {
            bb = mine & pos.pieces[Piece::new(PieceType::Knight, stm).index()];
            bb != 0
        } {
            v = SEE_VALUE[PieceType::Knight as usize] - v;
            if v < result {
                break;
            }
            occ ^= bb & bb.wrapping_neg();
        } else if {
            bb = mine & pos.pieces[Piece::new(PieceType::Bishop, stm).index()];
            bb != 0
        } {
            v = SEE_VALUE[PieceType::Bishop as usize] - v;
            if v < result {
                break;
            }
            occ ^= bb & bb.wrapping_neg();
            attackers |= bishop_attacks(to, occ) & diag;
        } else if {
            bb = mine & pos.pieces[Piece::new(PieceType::Rook, stm).index()];
            bb != 0
        } {
            v = SEE_VALUE[PieceType::Rook as usize] - v;
            if v < result {
                break;
            }
            occ ^= bb & bb.wrapping_neg();
            attackers |= rook_attacks(to, occ) & orth;
        } else if {
            bb = mine & pos.pieces[Piece::new(PieceType::Queen, stm).index()];
            bb != 0
        } {
            v = SEE_VALUE[PieceType::Queen as usize] - v;
            if v < result {
                break;
            }
            occ ^= bb & bb.wrapping_neg();
            attackers |= (bishop_attacks(to, occ) & diag) | (rook_attacks(to, occ) & orth);
        } else {
            // King: if opponent still has attackers, king can't recapture
            return (attackers & !pos.occupancies[stm as usize] != 0) != (result != 0);
        }
    }

    result != 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Position;

    fn see_test(fen: &str, from: &str, to: &str, threshold: i32) -> bool {
        let pos = Position::from_fen(fen).unwrap();
        let f = Square::from_string(from).unwrap();
        let t = Square::from_string(to).unwrap();
        let m = Move::new(f, t);
        see(&pos, m, threshold)
    }

    /// A defender pinned to its king does not recapture: Nxe5 wins the pawn, d6 being
    /// pinned by Rd1. Used to count the pin as a recapture and see a knight lost.
    #[test]
    fn a_pinned_defender_does_not_recapture() {
        let fen = "3k4/8/3p4/4p3/8/5N2/8/3R3K w - - 0 1";
        assert!(see_test(fen, "f3", "e5", 0));
        assert!(see_test(fen, "f3", "e5", 100), "a clean pawn");
        assert!(!see_test(fen, "f3", "e5", 101));
    }

    /// The same, colours reversed: the pin binds whichever side recaptures.
    #[test]
    fn a_pinned_defender_does_not_recapture_for_white_either() {
        let fen = "3r3k/8/5n2/8/4P3/3P4/8/3K4 b - - 0 1";
        assert!(see_test(fen, "f6", "e4", 100));
        assert!(!see_test(fen, "f6", "e4", 101));
    }

    /// A pin lasts only as long as its pinner: once the pinning bishop has left its
    /// square by capturing, the bishop it pinned takes it back.
    #[test]
    fn a_pin_ends_with_its_pinner() {
        let fen = "4k3/3b4/2p5/1B6/8/8/8/4K3 w - - 0 1";
        assert!(!see_test(fen, "b5", "c6", 0), "Bxc6 Bxc6 loses the bishop for a pawn");
    }

    #[test]
    fn test_see_rxp_undefended() {
        // White rook captures undefended black pawn on e5
        assert!(see_test(
            "1k1r4/1pp4p/p7/4p3/8/P5P1/1PP4P/2K1R3 w - - 0 1",
            "e1",
            "e5",
            0
        ));
    }

    #[test]
    fn test_see_pxq_defended() {
        // White pawn captures black queen on d5, defended by a knight
        // PxQ = +900, then NxP = -100, net = +800
        assert!(see_test(
            "4k3/8/3n4/3q4/4P3/8/8/4K3 w - - 0 1",
            "e4",
            "d5",
            0
        ));
        assert!(see_test(
            "4k3/8/3n4/3q4/4P3/8/8/4K3 w - - 0 1",
            "e4",
            "d5",
            800
        ));
    }

    #[test]
    fn test_see_nxp_defended_by_rook() {
        // White knight on d3 captures pawn on e5, defended by black rook on e6
        // NxP = +100, RxN = -320, net = -220
        assert!(!see_test(
            "4k3/8/4r3/4p3/8/3N4/8/4K3 w - - 0 1",
            "d3",
            "e5",
            0
        ));
    }

    #[test]
    fn test_see_equal_exchange() {
        // Knight takes knight: NxN = +320, then defended?
        // White Nc3 captures Nd5, no other defenders
        assert!(see_test("4k3/8/8/3n4/8/2N5/8/4K3 w - - 0 1", "c3", "d5", 0));
    }

    #[test]
    fn test_see_losing_exchange() {
        // Bishop captures rook, but rook is defended by queen
        // BxR = +500, QxB = -330, net = +170... actually that's winning
        // Let's do: Knight captures defended pawn with rook behind
        // NxP = +100, RxN = -(320-100) net for black. For white: 100 - 320 = -220
        assert!(!see_test(
            "4k3/8/4r3/4p3/8/4N3/8/4K3 w - - 0 1",
            "e3",
            "e5",
            0
        ));
    }

    #[test]
    fn test_see_promotion_always_true() {
        // Promotions always return true (non-MT_NORMAL)
        let pos = Position::from_fen("4k3/4P3/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let m = Move::new_promotion(
            Square::from_string("e7").unwrap(),
            Square::from_string("e8").unwrap(),
            PieceType::Queen,
        );
        assert!(see(&pos, m, 0));
    }

    #[test]
    fn test_see_castling_always_true() {
        let pos = Position::from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1").unwrap();
        let m = Move::new_with_type(
            Square::from_string("e1").unwrap(),
            Square::from_string("h1").unwrap(),
            MT_CASTLING,
        );
        assert!(see(&pos, m, 0));
    }

    #[test]
    fn test_see_king_capture_defended() {
        // White king captures black pawn on d2, defended by black rook on d8.
        // KxP = +100, Rd8xK — king can't capture if opponent has attackers.
        // SEE < 0.
        assert!(!see_test("3rk3/8/8/8/8/8/3p4/3K4 w - - 0 1", "d1", "d2", 0));
    }

    #[test]
    fn test_see_king_capture_undefended() {
        // White king captures undefended black pawn on d2. KxP = +100. SEE >= 0.
        assert!(see_test("4k3/8/8/8/8/8/3p4/3K4 w - - 0 1", "d1", "d2", 0));
    }

    #[test]
    fn test_see_xray_through_piece() {
        // White Rd2 captures d5 pawn, Black Rd8 recaptures, but White Qd1
        // is revealed via x-ray on d-file and recaptures.
        // Rd2xPd5 = +100, Rd8xRd5 = -500, Qd1xRd5 = +500. Net = +100.
        assert!(see_test(
            "3rk3/8/8/3p4/8/8/3R4/3QK3 w - - 0 1",
            "d2",
            "d5",
            0
        ));
        // With threshold 100 it should still pass (net = 100)
        assert!(see_test(
            "3rk3/8/8/3p4/8/8/3R4/3QK3 w - - 0 1",
            "d2",
            "d5",
            100
        ));
    }

    fn see_ep(fen: &str, from: &str, to: &str, threshold: i32) -> bool {
        let pos = Position::from_fen(fen).unwrap();
        let m = Move::new_with_type(
            Square::from_string(from).unwrap(),
            Square::from_string(to).unwrap(),
            MT_EN_PASSANT,
        );
        see(&pos, m, threshold)
    }

    #[test]
    fn en_passant_wins_one_pawn_and_no_more() {
        // After 33.f4 in David's game (issue #29): exf3 e.p. takes the f4 pawn and
        // nothing can recapture on f3. It wins exactly a pawn, so it must not pass
        // the ProbCut threshold of 156 that it used to pass unconditionally.
        let fen = "5rk1/Rb6/3rp3/1P1n1qp1/2BPpP1P/Q3p3/4N2P/2R4K b - f3 0 33";
        assert!(see_ep(fen, "e4", "f3", 0));
        assert!(see_ep(fen, "e4", "f3", 100));
        assert!(!see_ep(fen, "e4", "f3", 101));
        assert!(!see_ep(fen, "e4", "f3", 156));
    }

    #[test]
    fn en_passant_into_a_recapture_is_an_even_trade() {
        // exd6 e.p., the d6 pawn is then taken by the c7 pawn: pawn for pawn.
        let fen = "4k3/2p5/8/3pP3/8/8/8/4K3 w - d6 0 1";
        assert!(see_ep(fen, "e5", "d6", 0));
        assert!(!see_ep(fen, "e5", "d6", 1));
    }

    #[test]
    fn en_passant_opens_the_victims_square() {
        // ...exd3 e.p. removes the d4 pawn, which was the only thing between the
        // white rook on d6 and d3: the rook recaptures, pawn for pawn. Leaving d4 in
        // the occupancy would hide the rook and score the capture a clean +100.
        let fen = "7k/8/3R4/8/3Pp3/8/8/7K b - d3 0 1";
        assert!(see_ep(fen, "e4", "d3", 0));
        assert!(!see_ep(fen, "e4", "d3", 1));
    }
}
