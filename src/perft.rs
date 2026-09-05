//! [Perft](https://www.chessprogramming.org/Perft) — performance test for move generation validation.
//!
//! Counts all leaf nodes at a given depth in the game tree, comparing results
//! against known reference values to verify correctness of move generation,
//! make/unmake, and legality checking. Uses bulk counting at depth 1 to
//! avoid redundant make/unmake at the leaves.

use crate::movegen;
use crate::position::Position;
use crate::types::{ArrayBuf, Move, MAX_MOVES};
use std::time::Instant;

/// Count all leaf nodes at `depth` in the move tree (bulk counting at depth 1).
pub fn perft(pos: &mut Position, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }

    let mut buf: ArrayBuf<Move, MAX_MOVES> = ArrayBuf::new();
    let count = movegen::generate_legal_moves(pos, &mut buf);

    // Bulk counting at depth 1
    if depth == 1 {
        return count as u64;
    }

    let mut nodes = 0u64;
    for i in 0..count {
        let m = buf[i];
        #[cfg(debug_assertions)]
        let key_before = pos.key;
        #[cfg(debug_assertions)]
        let pawn_key_before = pos.pawn_key;
        pos.make_move(m);
        nodes += perft(pos, depth - 1);
        pos.unmake_move(m);
        #[cfg(debug_assertions)] {
            debug_assert_eq!(pos.key, key_before,
                "perft: Zobrist key drift for {}", m.to_uci());
            debug_assert_eq!(pos.pawn_key, pawn_key_before,
                "perft: pawn_key drift for {}", m.to_uci());
        }
    }
    nodes
}

/// [Divide](https://www.chessprogramming.org/Perft#Divide): run perft split by
/// root move. Prints node count per move, useful for debugging move generation
/// by comparing with a reference engine (`go perft`).
pub fn divide(pos: &mut Position, depth: u32) {
    let start = Instant::now();

    let mut buf: ArrayBuf<Move, MAX_MOVES> = ArrayBuf::new();
    let count = movegen::generate_legal_moves(pos, &mut buf);

    let mut total = 0u64;
    for i in 0..count {
        let m = buf[i];
        pos.make_move(m);
        let nodes = if depth <= 1 { 1 } else { perft(pos, depth - 1) };
        pos.unmake_move(m);
        println!("{}: {}", m.to_uci(), nodes);
        total += nodes;
    }

    let elapsed = start.elapsed();
    let nps = if elapsed.as_secs_f64() > 0.0 {
        (total as f64 / elapsed.as_secs_f64()) as u64
    } else {
        0
    };
    println!();
    println!("Nodes: {}", total);
    println!("Time:  {:.3}s", elapsed.as_secs_f64());
    println!("NPS:   {}", nps);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::movegen;
    use crate::types::{MT_CASTLING, PieceType, Square};

    fn assert_perft(fen: &str, depths: &[(u32, u64)]) {
        let mut pos = Position::from_fen(fen).unwrap();
        for &(depth, nodes) in depths {
            assert_eq!(
                perft(&mut pos, depth),
                nodes,
                "perft mismatch {fen} depth {depth}"
            );
        }
    }

    #[test]
    fn perft_startpos() {
        assert_perft(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            &[(1, 20), (2, 400), (3, 8902), (4, 197281)],
        );
    }

    #[test]
    fn perft_startpos_shredder() {
        assert_perft(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w AHah - 0 1",
            &[(1, 20), (2, 400), (3, 8902), (4, 197281)],
        );
    }

    #[test]
    fn perft_kiwipete() {
        assert_perft(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            &[(1, 48), (2, 2039), (3, 97862), (4, 4085603)],
        );
    }

    #[test]
    fn perft_chess324_kaufman() {
        // Node counts filled from this implementation after the FRC encoding
        // change; Chess324 uses orthodox castling so they must match any engine.
        let mut pos = Position::from_fen(
            "rnqbknbr/pppppppp/8/8/8/8/PPPPPPPP/RNBBKNQR w KQkq - 0 1",
        ).unwrap();
        assert_eq!(perft(&mut pos, 1), 20);
        assert_eq!(perft(&mut pos, 2), 400);
        assert_eq!(perft(&mut pos, 3), 8998);
        assert_eq!(perft(&mut pos, 4), 201609);
    }

    // Node counts from the widely published Chess960 perft suite.
    #[test]
    fn perft_frc_terje_sample() {
        assert_perft(
            "rqnkbrnb/1pppp1pp/8/5p2/p1P5/3P1N2/PP2PPPP/RQNKBR1B w KQkq - 0 4",
            &[(1, 26), (2, 727), (3, 19792), (4, 582545)],
        );
        assert_perft(
            "1r1qnbkr/ppp1pppp/1n1p4/5b2/2PP4/3N4/PP2PPPP/NRBQ1BKR w KQkq - 3 4",
            &[(1, 31), (2, 966), (3, 30811), (4, 953091)],
        );
        assert_perft(
            "rkbbqnrn/ppp2ppp/8/4pP2/4p3/8/PPPP2PP/RKBBQNRN w KQkq - 0 4",
            &[(1, 27), (2, 783), (3, 23282), (4, 712682)],
        );
    }

    #[test]
    fn perft_dfrc_asymmetric() {
        let mut pos = Position::from_fen(
            "nrkbnqbr/pppppppp/8/8/8/8/PPPPPPPP/BBNRKQNR w HDhb - 0 1",
        ).unwrap();
        assert_eq!(perft(&mut pos, 1), 20);
        assert_eq!(perft(&mut pos, 2), 380);
        assert_eq!(perft(&mut pos, 3), 8544);
        assert_eq!(perft(&mut pos, 4), 183907);
    }

    #[test]
    fn perft_frc_endgame() {
        // A published Chess960 perft position, known d2 = 1438.
        assert_perft(
            "rr6/2kpp3/1ppn2p1/p2b1q1p/P4P1P/1PNN2P1/2PP4/1K2R2R b E - 1 20",
            &[(2, 1438)],
        );
    }

    #[test]
    fn castle_king_already_on_g() {
        let mut pos = Position::from_fen("4k3/8/8/8/8/8/8/6KR w K - 0 1").unwrap();
        let mut buf = crate::types::ArrayBuf::<crate::types::Move, 256>::new();
        let n = movegen::generate_legal_moves(&pos, &mut buf);
        let castle = (0..n).map(|i| buf[i]).find(|m| m.move_type() == MT_CASTLING);
        let m = castle.expect("O-O with king already on g1");
        assert_eq!(m.from_sq(), Square::G1);
        assert_eq!(m.to_sq(), Square::H1);
        assert_eq!(m.castle_king_to(), Square::G1);
        assert_eq!(m.castle_rook_to(), Square::F1);
        pos.make_move(m);
        assert_eq!(pos.board[Square::G1.index()].piece_type(), PieceType::King);
        assert_eq!(pos.board[Square::F1.index()].piece_type(), PieceType::Rook);
        assert_eq!(pos.board[Square::H1.index()], crate::types::Piece::NONE);
        pos.unmake_move(m);
        assert_eq!(pos.king_sq(crate::types::Color::White), Square::G1);
        assert_eq!(pos.board[Square::H1.index()].piece_type(), PieceType::Rook);
    }

    #[test]
    fn castle_king_rook_swap() {
        let mut pos = Position::from_fen("4k3/8/8/8/8/8/8/5KR1 w K - 0 1").unwrap();
        let mut buf = crate::types::ArrayBuf::<crate::types::Move, 256>::new();
        let n = movegen::generate_legal_moves(&pos, &mut buf);
        let m = (0..n).map(|i| buf[i]).find(|mv| mv.move_type() == MT_CASTLING)
            .expect("O-O as a king/rook swap");
        assert_eq!(m.from_sq(), Square::F1);
        assert_eq!(m.to_sq(), Square::G1);
        pos.make_move(m);
        assert_eq!(pos.board[Square::G1.index()].piece_type(), PieceType::King);
        assert_eq!(pos.board[Square::F1.index()].piece_type(), PieceType::Rook);
        pos.unmake_move(m);
        assert_eq!(pos.board[Square::F1.index()].piece_type(), PieceType::King);
        assert_eq!(pos.board[Square::G1.index()].piece_type(), PieceType::Rook);
    }

    #[test]
    fn castle_adjacent_rook() {
        let mut pos = Position::from_fen("4k3/8/8/8/8/8/8/4KR2 w K - 0 1").unwrap();
        let mut buf = crate::types::ArrayBuf::<crate::types::Move, 256>::new();
        let n = movegen::generate_legal_moves(&pos, &mut buf);
        let m = (0..n).map(|i| buf[i]).find(|mv| mv.move_type() == MT_CASTLING)
            .expect("O-O with rook on f1");
        assert_eq!(m.to_sq(), Square::F1);
        assert_eq!(m.to_uci(), "e1g1");
        assert_eq!(m.to_uci_960(true), "e1f1");
        pos.make_move(m);
        assert_eq!(pos.king_sq(crate::types::Color::White), Square::G1);
        assert_eq!(pos.board[Square::F1.index()].piece_type(), PieceType::Rook);
    }

    #[test]
    fn castle_blocked_by_piece() {
        let pos = Position::from_fen("4k3/8/8/8/8/8/8/4KB1R w K - 0 1").unwrap();
        let mut buf = crate::types::ArrayBuf::<crate::types::Move, 256>::new();
        let n = movegen::generate_legal_moves(&pos, &mut buf);
        assert!((0..n).all(|i| buf[i].move_type() != MT_CASTLING));
    }

    #[test]
    fn castle_through_check_illegal() {
        // Rook on f3 attacks f1, the square the king would cross going O-O.
        let pos = Position::from_fen("4k3/8/8/8/8/5r2/8/R3K2R w KQ - 0 1").unwrap();
        let mut buf = crate::types::ArrayBuf::<crate::types::Move, 256>::new();
        let n = movegen::generate_legal_moves(&pos, &mut buf);
        let castles: Vec<_> = (0..n).map(|i| buf[i]).filter(|m| m.move_type() == MT_CASTLING).collect();
        assert_eq!(castles.len(), 1, "only O-O-O should survive");
        assert_eq!(castles[0].castle_king_to(), Square::C1);
    }

    #[test]
    fn castle_pinned_rook_illegal() {
        // Queen on a1 pins the queenside rook on b1.
        let pos = Position::from_fen("4k3/8/8/8/8/8/8/qR2K3 w Q - 0 1").unwrap();
        let mut buf = crate::types::ArrayBuf::<crate::types::Move, 256>::new();
        let n = movegen::generate_legal_moves(&pos, &mut buf);
        assert!((0..n).all(|i| buf[i].move_type() != MT_CASTLING));
    }

    #[test]
    fn capturing_enemy_rook_on_h1_keeps_our_f1_right() {
        // Our castling rook is on f1; h1 holds an enemy rook. Capturing h1 with a
        // knight must not clear the right (the mask is keyed to f1, not h1).
        let mut pos = Position::from_fen("4k3/8/8/8/8/6N1/8/4KR1r w K - 0 1").unwrap();
        assert_eq!(pos.castle_rook_sq(crate::types::WHITE_OO), Square::F1);
        let mut buf = crate::types::ArrayBuf::<crate::types::Move, 256>::new();
        let n = movegen::generate_legal_moves(&pos, &mut buf);
        let m = (0..n).map(|i| buf[i])
            .find(|m| m.from_sq() == Square::G3 && m.to_sq() == Square::H1)
            .expect("Ng3xh1");
        pos.make_move(m);
        assert_ne!(pos.castling_rights & crate::types::WHITE_OO, 0);
    }

    #[test]
    fn uci_castling_standard_vs_960() {
        let pos = Position::from_fen(
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        ).unwrap();
        let mut buf = crate::types::ArrayBuf::<crate::types::Move, 256>::new();
        let n = movegen::generate_legal_moves(&pos, &mut buf);
        let oo = (0..n).map(|i| buf[i]).find(|m| m.move_type() == MT_CASTLING && m.castle_king_to() == Square::G1)
            .unwrap();
        assert_eq!(oo.to_uci(), "e1g1");
        assert_eq!(oo.to_uci_960(true), "e1h1");
        assert!(crate::uci::parse_uci_move(&pos, "e1g1").unwrap().move_type() == MT_CASTLING);
        let mut pos960 = pos.clone();
        pos960.set_chess960(true);
        assert!(crate::uci::parse_uci_move(&pos960, "e1h1").unwrap().move_type() == MT_CASTLING);
        // Two squares is not a king walk: without the standard rewrite, e1g1 is nothing.
        assert!(crate::uci::parse_uci_move(&pos960, "e1g1").is_none());
    }
}
