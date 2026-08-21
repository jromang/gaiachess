//! Chess-side state for the interface: the position, the moves that reached it, the
//! legal replies and how the game ended.
//!
//! Positions are rebuilt by replaying the move list rather than unwinding moves, so
//! the repetition history the draw rules depend on is always the real one.

use crate::movegen;
use crate::position::Position;
use crate::types::{
    ArrayBuf, BLACK_OO, BLACK_OOO, CASTLING_DATA, Color, MAX_MOVES, MT_CASTLING,
    MT_EN_PASSANT, MT_PROMOTION, Move, Piece, PieceType, Square, WHITE_OO, WHITE_OOO,
};

pub const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/// How a finished game finished.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Outcome {
    /// The named colour delivered mate.
    Checkmate(Color),
    Stalemate,
    /// Fifty-move rule, threefold repetition or insufficient material.
    Draw,
    /// The named colour ran out of time.
    Flag(Color),
}

pub struct GameState {
    pub pos: Position,
    /// Every move played from the start, kept so a take-back can replay from scratch.
    pub moves: Vec<Move>,
    /// Legal replies in the current position, refreshed after every change.
    pub legal: Vec<Move>,
    pub outcome: Option<Outcome>,
}

impl GameState {
    pub fn new() -> GameState {
        let pos = Position::from_fen(STARTPOS).expect("start position must parse");
        let mut state = GameState {
            pos,
            moves: Vec::new(),
            legal: Vec::new(),
            outcome: None,
        };
        state.refresh();
        state
    }

    /// Plays a move that must come from [`GameState::legal`].
    pub fn play(&mut self, m: Move) {
        debug_assert!(self.legal.contains(&m), "move must be legal: {}", m.to_uci());
        self.pos.make_move(m);
        self.moves.push(m);
        self.refresh();
    }

    /// Rewinds `count` half-moves, stopping at the start of the game.
    pub fn take_back(&mut self, count: usize) {
        let keep = self.moves.len().saturating_sub(count);
        self.moves.truncate(keep);
        self.replay();
    }

    /// Returns to the start position, discarding the move list.
    pub fn restart(&mut self) {
        self.moves.clear();
        self.replay();
    }

    /// Records a loss on time. Kept separate because it is the one outcome the
    /// position itself cannot know about.
    pub fn flag(&mut self, loser: Color) {
        self.outcome = Some(Outcome::Flag(loser));
    }

    fn replay(&mut self) {
        self.pos = Position::from_fen(STARTPOS).expect("start position must parse");
        for &m in &self.moves {
            self.pos.make_move(m);
        }
        self.refresh();
    }

    fn refresh(&mut self) {
        let mut buf = ArrayBuf::<Move, MAX_MOVES>::new();
        let count = movegen::generate_legal_moves(&self.pos, &mut buf);
        self.legal = (0..count).map(|i| buf[i]).collect();

        self.outcome = if self.legal.is_empty() {
            if self.pos.in_check() {
                Some(Outcome::Checkmate(!self.pos.side_to_move))
            } else {
                Some(Outcome::Stalemate)
            }
        } else if self.pos.is_draw(0) || insufficient_material(&self.pos) {
            Some(Outcome::Draw)
        } else {
            None
        };
    }

    /// Legal moves leaving `sq`, in generation order.
    pub fn moves_from(&self, sq: Square) -> impl Iterator<Item = Move> + '_ {
        self.legal.iter().copied().filter(move |m| m.from_sq() == sq)
    }

    /// The legal move from `from` to `to`, if there is exactly one. Promotions are
    /// ambiguous by design and are reported through `promotions_from` instead.
    pub fn find_move(&self, from: Square, to: Square, promo: Option<PieceType>) -> Option<Move> {
        self.legal.iter().copied().find(|m| {
            m.from_sq() == from
                && m.to_sq() == to
                && match promo {
                    Some(pt) => m.move_type() == MT_PROMOTION && m.promo_type() == pt,
                    None => true,
                }
        })
    }

    /// True when moving `from` to `to` needs the player to pick a promotion piece.
    pub fn needs_promotion(&self, from: Square, to: Square) -> bool {
        self.legal
            .iter()
            .any(|m| m.from_sq() == from && m.to_sq() == to && m.move_type() == MT_PROMOTION)
    }
}

/// True when `m` is one of the legal moves of `pos`.
///
/// For moves that reach the interface from somewhere other than [`GameState::legal`] —
/// the reply the engine expects, two plies ahead of the board — and must be checked
/// before anything is done with them.
pub fn is_legal(pos: &Position, m: Move) -> bool {
    let mut buf = ArrayBuf::<Move, MAX_MOVES>::new();
    let count = movegen::generate_legal_moves(pos, &mut buf);
    (0..count).any(|i| buf[i] == m)
}

/// The square a move empties of an enemy piece, which is not the destination for an
/// en passant capture.
pub fn captured_square(pos: &Position, m: Move) -> Option<Square> {
    let to = m.to_sq();
    if m.move_type() == MT_EN_PASSANT {
        let back = if pos.side_to_move == Color::White { -8 } else { 8 };
        return Some(Square((to.0 as i32 + back) as u8));
    }
    if m.move_type() == MT_CASTLING {
        // The destination holds our own rook, not a captured piece.
        return None;
    }
    (pos.piece_on(to) != Piece::NONE).then_some(to)
}

/// The rook's journey in a castling move, which the move itself leaves implicit: it
/// only records the king's. Needed to animate the rook alongside the king.
pub fn castle_rook(m: Move) -> (Square, Square) {
    debug_assert_eq!(m.move_type(), MT_CASTLING);
    let right = match (m.to_sq().0, m.from_sq().0 < 32) {
        (n, true) if n > m.from_sq().0 => WHITE_OO,
        (_, true) => WHITE_OOO,
        (n, false) if n > m.from_sq().0 => BLACK_OO,
        _ => BLACK_OOO,
    };
    let data = &CASTLING_DATA[right as usize];
    debug_assert_eq!(data.king_to, m.to_sq());
    (data.rook_from, data.rook_to)
}

/// Draws that no amount of play can escape: bare kings, or a lone minor piece.
fn insufficient_material(pos: &Position) -> bool {
    let heavy = pos.pieces[Piece::WHITE_PAWN.index()]
        | pos.pieces[Piece::BLACK_PAWN.index()]
        | pos.pieces[Piece::WHITE_ROOK.index()]
        | pos.pieces[Piece::BLACK_ROOK.index()]
        | pos.pieces[Piece::WHITE_QUEEN.index()]
        | pos.pieces[Piece::BLACK_QUEEN.index()];
    if heavy != 0 {
        return false;
    }
    let knights = pos.pieces[Piece::WHITE_KNIGHT.index()] | pos.pieces[Piece::BLACK_KNIGHT.index()];
    let bishops = pos.pieces[Piece::WHITE_BISHOP.index()] | pos.pieces[Piece::BLACK_BISHOP.index()];
    let minors = (knights | bishops).count_ones();
    if minors <= 1 {
        return true;
    }
    // Any number of bishops, but all on one colour of square, can never mate.
    const DARK: u64 = 0xaa55_aa55_aa55_aa55;
    knights == 0 && (bishops & DARK == 0 || bishops & !DARK == 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_position_has_twenty_moves() {
        let game = GameState::new();
        assert_eq!(game.legal.len(), 20);
        assert_eq!(game.outcome, None);
        assert_eq!(game.moves_from(Square::E2).count(), 2);
    }

    #[test]
    fn take_back_restores_the_previous_position() {
        let mut game = GameState::new();
        let before = game.pos.to_fen();
        let e4 = game.find_move(Square::E2, Square::E4, None).unwrap();
        game.play(e4);
        let e5 = game.find_move(Square::E7, Square::E5, None).unwrap();
        game.play(e5);
        game.take_back(2);
        assert_eq!(game.pos.to_fen(), before);
        assert!(game.moves.is_empty());
    }

    #[test]
    fn fools_mate_is_checkmate() {
        let mut game = GameState::new();
        for (from, to) in [
            (Square::F2, Square::F3),
            (Square::E7, Square::E5),
            (Square::G2, Square::G4),
            (Square::D8, Square::H4),
        ] {
            let m = game.find_move(from, to, None).unwrap();
            game.play(m);
        }
        assert_eq!(game.outcome, Some(Outcome::Checkmate(Color::Black)));
    }

    #[test]
    fn castling_reports_the_rooks_journey() {
        for (fen, king_to, rook_from, rook_to) in [
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", Square::G1, Square::H1, Square::F1),
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", Square::C1, Square::A1, Square::D1),
            ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", Square::G8, Square::H8, Square::F8),
            ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", Square::C8, Square::A8, Square::D8),
        ] {
            let pos = Position::from_fen(fen).unwrap();
            let mut buf = ArrayBuf::<Move, MAX_MOVES>::new();
            let n = movegen::generate_legal_moves(&pos, &mut buf);
            let m = (0..n)
                .map(|i| buf[i])
                .find(|m| m.move_type() == MT_CASTLING && m.to_sq() == king_to)
                .expect("castling move must be generated");
            assert_eq!(castle_rook(m), (rook_from, rook_to));
        }
    }

    #[test]
    fn en_passant_clears_the_passed_pawn() {
        let pos =
            Position::from_fen("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3")
                .unwrap();
        let mut buf = ArrayBuf::<Move, MAX_MOVES>::new();
        let n = movegen::generate_legal_moves(&pos, &mut buf);
        let ep = (0..n)
            .map(|i| buf[i])
            .find(|m| m.move_type() == MT_EN_PASSANT)
            .unwrap();
        assert_eq!(captured_square(&pos, ep), Some(Square::F5));
    }

    #[test]
    fn bare_kings_are_a_draw() {
        let game_over = Position::from_fen("8/8/4k3/8/8/4K3/8/8 w - - 0 1").unwrap();
        assert!(insufficient_material(&game_over));
        let lone_knight = Position::from_fen("8/8/4k3/8/8/4K1N1/8/8 w - - 0 1").unwrap();
        assert!(insufficient_material(&lone_knight));
        let two_knights = Position::from_fen("8/8/4k3/8/8/4K1NN/8/8 w - - 0 1").unwrap();
        assert!(!insufficient_material(&two_knights));
        let with_pawn = Position::from_fen("8/8/4k3/8/8/4K1P1/8/8 w - - 0 1").unwrap();
        assert!(!insufficient_material(&with_pawn));
    }
}
