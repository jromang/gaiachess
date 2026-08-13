//! Fundamental chess types: colors, pieces, squares, moves, and castling rights.
//!
//! # Encoding conventions
//!
//! - **Pieces**: `type * 2 + color` (WHITE_PAWN=0, BLACK_PAWN=1, ..., BLACK_KING=11).
//! - **Squares**: [Little-Endian Rank-File Mapping](https://www.chessprogramming.org/Square_Mapping_Considerations)
//!   (A1=0, B1=1, ..., H8=63). File = `sq & 7`, rank = `sq >> 3`.
//! - **Moves**: 16-bit [encoded moves](https://www.chessprogramming.org/Encoding_Moves):
//!   `dest:6 | src:6 | promo:2 | type:2`.

/// Side to move: White or Black.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    /// Number of colors (2).
    #[allow(dead_code)]
    pub const NUM: usize = 2;

    /// Returns the opposite color.
    #[inline(always)]
    pub const fn flip(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }

    #[inline(always)]
    pub const fn index(self) -> usize {
        self as usize
    }
}

impl std::ops::Not for Color {
    type Output = Color;
    #[inline(always)]
    fn not(self) -> Color {
        self.flip()
    }
}

/// The type of a chess piece, without color information.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum PieceType {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5,
}

#[allow(dead_code)]
impl PieceType {
    pub const NUM: usize = 6;

    #[inline(always)]
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// A colored piece, encoded as `piece_type * 2 + color`.
///
/// Values 0..11 represent the 12 piece-color combinations; 12 means no piece.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(transparent)]
pub struct Piece(pub u8);

impl Piece {
    /// Sentinel value for empty squares.
    pub const NONE: Piece = Piece(12);
    /// Number of valid piece-color combinations (12).
    #[allow(dead_code)]
    pub const NUM: usize = 12;

    // Piece constants: WHITE_PAWN=0, BLACK_PAWN=1, ..., BLACK_KING=11.
    pub const WHITE_PAWN: Piece = Piece(0);
    pub const BLACK_PAWN: Piece = Piece(1);
    pub const WHITE_KNIGHT: Piece = Piece(2);
    pub const BLACK_KNIGHT: Piece = Piece(3);
    pub const WHITE_BISHOP: Piece = Piece(4);
    pub const BLACK_BISHOP: Piece = Piece(5);
    pub const WHITE_ROOK: Piece = Piece(6);
    pub const BLACK_ROOK: Piece = Piece(7);
    pub const WHITE_QUEEN: Piece = Piece(8);
    pub const BLACK_QUEEN: Piece = Piece(9);
    pub const WHITE_KING: Piece = Piece(10);
    pub const BLACK_KING: Piece = Piece(11);

    /// Creates a piece from a type and color: `pt * 2 + c`.
    #[inline(always)]
    pub const fn new(pt: PieceType, c: Color) -> Piece {
        Piece((pt as u8) * 2 + c as u8)
    }

    #[inline(always)]
    pub const fn piece_type(self) -> PieceType {
        match self.0 >> 1 {
            0 => PieceType::Pawn,
            1 => PieceType::Knight,
            2 => PieceType::Bishop,
            3 => PieceType::Rook,
            4 => PieceType::Queen,
            _ => PieceType::King,
        }
    }

    #[inline(always)]
    pub const fn color(self) -> Color {
        if self.0 & 1 == 0 { Color::White } else { Color::Black }
    }

    #[inline(always)]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    pub fn to_char(self) -> char {
        const CHARS: [char; 13] = ['P', 'p', 'N', 'n', 'B', 'b', 'R', 'r', 'Q', 'q', 'K', 'k', '.'];
        CHARS[self.0 as usize]
    }

    /// Parses a FEN piece character ('P','p','N','n',...). Returns `None` for invalid chars.
    pub fn from_char(c: char) -> Option<Piece> {
        match c {
            'P' => Some(Piece::WHITE_PAWN),
            'p' => Some(Piece::BLACK_PAWN),
            'N' => Some(Piece::WHITE_KNIGHT),
            'n' => Some(Piece::BLACK_KNIGHT),
            'B' => Some(Piece::WHITE_BISHOP),
            'b' => Some(Piece::BLACK_BISHOP),
            'R' => Some(Piece::WHITE_ROOK),
            'r' => Some(Piece::BLACK_ROOK),
            'Q' => Some(Piece::WHITE_QUEEN),
            'q' => Some(Piece::BLACK_QUEEN),
            'K' => Some(Piece::WHITE_KING),
            'k' => Some(Piece::BLACK_KING),
            _ => None,
        }
    }
}

impl Default for Piece {
    fn default() -> Self {
        Piece::NONE
    }
}

/// A board square (0..63) in Little-Endian Rank-File order: A1=0, B1=1, ..., H8=63.
///
/// `NONE` (64) is a sentinel for "no square" (e.g., no en passant).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
#[repr(transparent)]
pub struct Square(pub u8);

#[allow(dead_code)]
impl Square {
    pub const NUM: usize = 64;
    pub const NONE: Square = Square(64);

    // Named squares (all 64)
    pub const A1: Square = Square(0);
    pub const B1: Square = Square(1);
    pub const C1: Square = Square(2);
    pub const D1: Square = Square(3);
    pub const E1: Square = Square(4);
    pub const F1: Square = Square(5);
    pub const G1: Square = Square(6);
    pub const H1: Square = Square(7);
    pub const A2: Square = Square(8);
    pub const B2: Square = Square(9);
    pub const C2: Square = Square(10);
    pub const D2: Square = Square(11);
    pub const E2: Square = Square(12);
    pub const F2: Square = Square(13);
    pub const G2: Square = Square(14);
    pub const H2: Square = Square(15);
    pub const A3: Square = Square(16);
    pub const B3: Square = Square(17);
    pub const C3: Square = Square(18);
    pub const D3: Square = Square(19);
    pub const E3: Square = Square(20);
    pub const F3: Square = Square(21);
    pub const G3: Square = Square(22);
    pub const H3: Square = Square(23);
    pub const A4: Square = Square(24);
    pub const B4: Square = Square(25);
    pub const C4: Square = Square(26);
    pub const D4: Square = Square(27);
    pub const E4: Square = Square(28);
    pub const F4: Square = Square(29);
    pub const G4: Square = Square(30);
    pub const H4: Square = Square(31);
    pub const A5: Square = Square(32);
    pub const B5: Square = Square(33);
    pub const C5: Square = Square(34);
    pub const D5: Square = Square(35);
    pub const E5: Square = Square(36);
    pub const F5: Square = Square(37);
    pub const G5: Square = Square(38);
    pub const H5: Square = Square(39);
    pub const A6: Square = Square(40);
    pub const B6: Square = Square(41);
    pub const C6: Square = Square(42);
    pub const D6: Square = Square(43);
    pub const E6: Square = Square(44);
    pub const F6: Square = Square(45);
    pub const G6: Square = Square(46);
    pub const H6: Square = Square(47);
    pub const A7: Square = Square(48);
    pub const B7: Square = Square(49);
    pub const C7: Square = Square(50);
    pub const D7: Square = Square(51);
    pub const E7: Square = Square(52);
    pub const F7: Square = Square(53);
    pub const G7: Square = Square(54);
    pub const H7: Square = Square(55);
    pub const A8: Square = Square(56);
    pub const B8: Square = Square(57);
    pub const C8: Square = Square(58);
    pub const D8: Square = Square(59);
    pub const E8: Square = Square(60);
    pub const F8: Square = Square(61);
    pub const G8: Square = Square(62);
    pub const H8: Square = Square(63);

    #[inline(always)]
    /// Creates a square from file (0=a..7=h) and rank (0=1..7=8).
    pub const fn new(file: u8, rank: u8) -> Square {
        Square(rank * 8 + file)
    }

    #[inline(always)]
    pub const fn from_index(i: u8) -> Square {
        Square(i)
    }

    #[inline(always)]
    pub const fn file(self) -> u8 {
        self.0 & 7
    }

    #[inline(always)]
    pub const fn rank(self) -> u8 {
        self.0 >> 3
    }

    #[inline(always)]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    #[inline(always)]
    /// Vertically mirrors the square (rank 0 <-> rank 7). Equivalent to `sq ^ 56`.
    pub const fn flip_rank(self) -> Square {
        Square(self.0 ^ 56)
    }

    #[inline(always)]
    /// Returns a bitboard with only this square set (`1 << sq`).
    pub const fn bb(self) -> u64 {
        1u64 << self.0
    }

    /// Converts to algebraic notation (e.g., "e4"). Returns "-" for `NONE`.
    pub fn to_string(self) -> String {
        if self == Square::NONE {
            return "-".to_string();
        }
        let file = (b'a' + self.file()) as char;
        let rank = (b'1' + self.rank()) as char;
        format!("{}{}", file, rank)
    }

    /// Parses algebraic notation (e.g., "e4") into a square.
    pub fn from_string(s: &str) -> Option<Square> {
        let bytes = s.as_bytes();
        if bytes.len() < 2 {
            return None;
        }
        let file = bytes[0].wrapping_sub(b'a');
        let rank = bytes[1].wrapping_sub(b'1');
        if file < 8 && rank < 8 {
            Some(Square::new(file, rank))
        } else {
            None
        }
    }
}

// Move type flags (bits 14-15 of the 16-bit move encoding).
/// Normal move (no special flag).
pub const MT_NORMAL: u16 = 0;
/// Pawn promotion. Bits 12-13 encode the promotion piece type minus Knight.
pub const MT_PROMOTION: u16 = 1 << 14;
/// En passant capture.
pub const MT_EN_PASSANT: u16 = 2 << 14;
/// Castling (king move to rook's destination; the rook is moved implicitly).
pub const MT_CASTLING: u16 = 3 << 14;

/// A chess move encoded in 16 bits.
///
/// ```text
/// Bits  0-5 : destination square (0-63)
/// Bits  6-11: source square (0-63)
/// Bits 12-13: promotion piece type - Knight (0=N, 1=B, 2=R, 3=Q)
/// Bits 14-15: move type (0=normal, 1=promotion, 2=en passant, 3=castling)
/// ```
///
/// Special values: `NONE` = 0 (invalid, src == dest), `NULL` = 65 (null move for search).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(transparent)]
pub struct Move(pub u16);

impl Move {
    pub const NONE: Move = Move(0);
    pub const NULL: Move = Move(65);

    #[inline(always)]
    pub const fn new(from: Square, to: Square) -> Move {
        Move((from.0 as u16) << 6 | to.0 as u16)
    }

    #[inline(always)]
    pub const fn new_with_type(from: Square, to: Square, mt: u16) -> Move {
        Move((from.0 as u16) << 6 | to.0 as u16 | mt)
    }

    #[inline(always)]
    pub const fn new_promotion(from: Square, to: Square, pt: PieceType) -> Move {
        Move(
            (from.0 as u16) << 6
                | to.0 as u16
                | MT_PROMOTION
                | ((pt as u16 - PieceType::Knight as u16) << 12),
        )
    }

    #[inline(always)]
    pub const fn from_sq(self) -> Square {
        Square(((self.0 >> 6) & 0x3F) as u8)
    }

    #[inline(always)]
    pub const fn to_sq(self) -> Square {
        Square((self.0 & 0x3F) as u8)
    }

    #[inline(always)]
    pub const fn move_type(self) -> u16 {
        self.0 & (3 << 14)
    }

    #[inline(always)]
    pub const fn promo_type(self) -> PieceType {
        match (self.0 >> 12) & 3 {
            0 => PieceType::Knight,
            1 => PieceType::Bishop,
            2 => PieceType::Rook,
            _ => PieceType::Queen,
        }
    }

    #[inline(always)]
    pub const fn is_ok(self) -> bool {
        self.from_sq().0 != self.to_sq().0
    }

    pub fn to_uci(self) -> String {
        let from = self.from_sq();
        let to = self.to_sq();
        let mut s = format!("{}{}", from.to_string(), to.to_string());
        if self.move_type() == MT_PROMOTION {
            let promo = match self.promo_type() {
                PieceType::Knight => 'n',
                PieceType::Bishop => 'b',
                PieceType::Rook => 'r',
                PieceType::Queen => 'q',
                _ => unreachable!(),
            };
            s.push(promo);
        }
        s
    }
}

impl Default for Move {
    fn default() -> Self {
        Move::NONE
    }
}

// ============================================================
// Move buffer constants
// ============================================================

/// Maximum legal moves in any chess position (theoretical max ~218, 256 for safety).
pub const MAX_MOVES: usize = 256;

// ============================================================
// ArrayBuf — uninitialized stack buffer (no memset)
// ============================================================

/// A move paired with its ordering score for move picking.
/// 8 bytes: score (4) + mv (2) + 2 padding. Score-first for hot-loop access.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct ScoredMove {
    pub score: i32,
    pub mv: Move,
}

/// Uninitialized stack buffer backed by `[MaybeUninit<T>; N]`.
///
/// Zero-cost construction (no memset). Only indices that have been
/// written to may be read.
pub struct ArrayBuf<T: Copy, const N: usize> {
    data: [std::mem::MaybeUninit<T>; N],
}

impl<T: Copy, const N: usize> ArrayBuf<T, N> {
    /// Create a new uninitialized buffer.
    #[inline(always)]
    pub const fn new() -> Self {
        // SAFETY: [MaybeUninit<T>; N] is valid when uninitialized.
        Self { data: unsafe { std::mem::MaybeUninit::uninit().assume_init() } }
    }

    #[inline(always)]
    pub fn swap(&mut self, i: usize, j: usize) {
        self.data.swap(i, j);
    }

    /// Copy the value at index `src` to index `dst`.
    #[inline(always)]
    pub fn copy_within(&mut self, src: usize, dst: usize) {
        self.data[dst] = self.data[src];
    }
}

impl<T: Copy, const N: usize> std::ops::Index<usize> for ArrayBuf<T, N> {
    type Output = T;
    #[inline(always)]
    fn index(&self, i: usize) -> &T {
        unsafe { self.data[i].assume_init_ref() }
    }
}

impl<T: Copy, const N: usize> std::ops::IndexMut<usize> for ArrayBuf<T, N> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut T {
        unsafe { self.data[i].assume_init_mut() }
    }
}

// ============================================================
// Castling rights  (4-bit flags: KQkq)
// ============================================================

/// White kingside castling right (O-O).
pub const WHITE_OO: u8 = 1;
/// White queenside castling right (O-O-O).
pub const WHITE_OOO: u8 = 2;
/// Black kingside castling right (O-O).
pub const BLACK_OO: u8 = 4;
/// Black queenside castling right (O-O-O).
pub const BLACK_OOO: u8 = 8;
/// All four castling rights set.
pub const ALL_CASTLING: u8 = 15;

/// For each square, the castling rights that survive when a piece
/// moves from or to that square. AND with current rights.
pub const CASTLING_RIGHTS_MASK: [u8; 64] = {
    let mut mask = [ALL_CASTLING; 64];
    // White king on E1: loses both white castling rights
    mask[Square::E1.0 as usize] = ALL_CASTLING ^ (WHITE_OO | WHITE_OOO);
    // White rooks
    mask[Square::H1.0 as usize] = ALL_CASTLING ^ WHITE_OO;
    mask[Square::A1.0 as usize] = ALL_CASTLING ^ WHITE_OOO;
    // Black king on E8
    mask[Square::E8.0 as usize] = ALL_CASTLING ^ (BLACK_OO | BLACK_OOO);
    // Black rooks
    mask[Square::H8.0 as usize] = ALL_CASTLING ^ BLACK_OO;
    mask[Square::A8.0 as usize] = ALL_CASTLING ^ BLACK_OOO;
    mask
};

/// King and rook source/destination squares for a single castling move.
pub struct CastlingData {
    pub king_from: Square,
    pub king_to: Square,
    pub rook_from: Square,
    pub rook_to: Square,
}

// Indexed by castling right bit value (1, 2, 4, 8)
pub const CASTLING_DATA: [CastlingData; 9] = [
    // 0: unused
    CastlingData { king_from: Square::A1, king_to: Square::A1, rook_from: Square::A1, rook_to: Square::A1 },
    // 1: WHITE_OO
    CastlingData { king_from: Square::E1, king_to: Square::G1, rook_from: Square::H1, rook_to: Square::F1 },
    // 2: WHITE_OOO
    CastlingData { king_from: Square::E1, king_to: Square::C1, rook_from: Square::A1, rook_to: Square::D1 },
    // 3: unused
    CastlingData { king_from: Square::A1, king_to: Square::A1, rook_from: Square::A1, rook_to: Square::A1 },
    // 4: BLACK_OO
    CastlingData { king_from: Square::E8, king_to: Square::G8, rook_from: Square::H8, rook_to: Square::F8 },
    // 5-7: unused
    CastlingData { king_from: Square::A1, king_to: Square::A1, rook_from: Square::A1, rook_to: Square::A1 },
    CastlingData { king_from: Square::A1, king_to: Square::A1, rook_from: Square::A1, rook_to: Square::A1 },
    CastlingData { king_from: Square::A1, king_to: Square::A1, rook_from: Square::A1, rook_to: Square::A1 },
    // 8: BLACK_OOO
    CastlingData { king_from: Square::E8, king_to: Square::C8, rook_from: Square::A8, rook_to: Square::D8 },
];

/// Squares that must be empty for castling (between king and rook, exclusive)
pub const CASTLING_PATH: [u64; 9] = [
    0,
    Square::F1.bb() | Square::G1.bb(),                                   // WHITE_OO
    Square::B1.bb() | Square::C1.bb() | Square::D1.bb(),                 // WHITE_OOO
    0,
    Square::F8.bb() | Square::G8.bb(),                                   // BLACK_OO
    0, 0, 0,
    Square::B8.bb() | Square::C8.bb() | Square::D8.bb(),                 // BLACK_OOO
];

/// Squares the king passes through (must not be attacked)
pub const KING_CASTLING_PATH: [u64; 9] = [
    0,
    Square::E1.bb() | Square::F1.bb() | Square::G1.bb(),                 // WHITE_OO
    Square::E1.bb() | Square::D1.bb() | Square::C1.bb(),                 // WHITE_OOO
    0,
    Square::E8.bb() | Square::F8.bb() | Square::G8.bb(),                 // BLACK_OO
    0, 0, 0,
    Square::E8.bb() | Square::D8.bb() | Square::C8.bb(),                 // BLACK_OOO
];

// ============================================================
// Direction constants (square index offsets on the 8x8 board)
// ============================================================

pub const NORTH: i8 = 8;
pub const SOUTH: i8 = -8;
pub const EAST: i8 = 1;
pub const WEST: i8 = -1;
pub const NORTH_EAST: i8 = 9;
pub const NORTH_WEST: i8 = 7;
pub const SOUTH_EAST: i8 = -7;
pub const SOUTH_WEST: i8 = -9;

/// Pawn push direction: +8 (north) for White, -8 (south) for Black.
#[inline(always)]
pub const fn pawn_push(c: Color) -> i8 {
    if c as u8 == 0 { NORTH } else { SOUTH }
}

// ============================================================
// Search constants
// ============================================================

/// Sentinel: no score computed yet.
pub const SCORE_NONE: i32 = 32002;
/// Infinite score (used as initial alpha/beta bounds).
pub const SCORE_INFINITE: i32 = 32001;
/// Checkmate score (offset by ply to prefer shorter mates).
pub const SCORE_MATE: i32 = 32000;
/// Maximum search depth (plies).
pub const MAX_PLY: usize = 128;
/// Threshold: any score with abs >= this is a mate score.
pub const SCORE_MATE_IN_MAX: i32 = SCORE_MATE - MAX_PLY as i32;

/// TB win score base (below mate but above all positional scores).
pub const SCORE_TB_WIN: i32 = SCORE_MATE_IN_MAX - 1;
/// TB loss score base.
#[allow(dead_code)]
pub const SCORE_TB_LOSS: i32 = -SCORE_TB_WIN;
/// Threshold: any score with abs >= this is a TB win/loss.
#[allow(dead_code)]
pub const SCORE_TB_WIN_IN_MAX: i32 = SCORE_TB_WIN - MAX_PLY as i32;

/// Score for "TB win at `ply` plies from root".
#[allow(dead_code)]
#[inline(always)]
pub const fn tb_win_in(ply: i32) -> i32 { SCORE_TB_WIN - ply }

/// Score for "TB loss at `ply` plies from root".
#[allow(dead_code)]
#[inline(always)]
pub const fn tb_loss_in(ply: i32) -> i32 { -SCORE_TB_WIN + ply }

/// Score for "mating the opponent in `ply` plies from root".
#[allow(dead_code)]
#[inline(always)]
pub const fn mate_in(ply: i32) -> i32 { SCORE_MATE - ply }

/// Score for "being mated in `ply` plies from root".
#[inline(always)]
pub const fn mated_in(ply: i32) -> i32 { -SCORE_MATE + ply }

/// Returns true if `score` represents a forced mate (either side).
#[inline(always)]
pub const fn is_mate_score(score: i32) -> bool {
    score.unsigned_abs() >= SCORE_MATE_IN_MAX as u32
}

/// Returns true if `score` represents a TB win/loss (either side).
#[allow(dead_code)]
#[inline(always)]
pub const fn is_tb_score(score: i32) -> bool {
    score.unsigned_abs() >= SCORE_TB_WIN_IN_MAX as u32
        && !is_mate_score(score)
}

/// MVV-LVA piece values indexed by `PieceType` (Pawn=0..King=5, None=6).
pub const PIECE_VALUE: [i32; 7] = [100, 320, 330, 500, 900, 20000, 0];

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_square_encoding() {
        assert_eq!(Square::A1.0, 0);
        assert_eq!(Square::H8.0, 63);
        assert_eq!(Square::E1.file(), 4);
        assert_eq!(Square::E1.rank(), 0);
        assert_eq!(Square::new(4, 0), Square::E1);
        assert_eq!(Square::new(7, 7), Square::H8);
        assert_eq!(Square::A1.flip_rank(), Square::A8);
    }

    #[test]
    fn test_piece_encoding() {
        let wp = Piece::new(PieceType::Pawn, Color::White);
        assert_eq!(wp, Piece::WHITE_PAWN);
        assert_eq!(wp.piece_type(), PieceType::Pawn);
        assert_eq!(wp.color(), Color::White);

        let bk = Piece::new(PieceType::King, Color::Black);
        assert_eq!(bk, Piece::BLACK_KING);
        assert_eq!(bk.0, 11);
        assert_eq!(bk.piece_type(), PieceType::King);
        assert_eq!(bk.color(), Color::Black);
    }

    #[test]
    fn test_move_encoding() {
        let m = Move::new(Square::E2, Square::E4);
        assert_eq!(m.from_sq(), Square::E2);
        assert_eq!(m.to_sq(), Square::E4);
        assert_eq!(m.move_type(), MT_NORMAL);

        let promo = Move::new_promotion(Square::E7, Square::E8, PieceType::Queen);
        assert_eq!(promo.from_sq(), Square::E7);
        assert_eq!(promo.to_sq(), Square::E8);
        assert_eq!(promo.move_type(), MT_PROMOTION);
        assert_eq!(promo.promo_type(), PieceType::Queen);

        let ep = Move::new_with_type(Square::E5, Square::D6, MT_EN_PASSANT);
        assert_eq!(ep.move_type(), MT_EN_PASSANT);

        let castle = Move::new_with_type(Square::E1, Square::G1, MT_CASTLING);
        assert_eq!(castle.move_type(), MT_CASTLING);
    }

    #[test]
    fn test_color_flip() {
        assert_eq!(Color::White.flip(), Color::Black);
        assert_eq!(Color::Black.flip(), Color::White);
        assert_eq!(!Color::White, Color::Black);
    }

    #[test]
    fn test_square_string() {
        assert_eq!(Square::E4.to_string(), "e4");
        assert_eq!(Square::A1.to_string(), "a1");
        assert_eq!(Square::H8.to_string(), "h8");
        assert_eq!(Square::from_string("e4"), Some(Square::E4));
        assert_eq!(Square::from_string("a1"), Some(Square::A1));
    }

    #[test]
    fn test_castling_rights_mask() {
        // Moving king from E1 loses both white castling rights
        assert_eq!(CASTLING_RIGHTS_MASK[Square::E1.index()], ALL_CASTLING ^ (WHITE_OO | WHITE_OOO));
        // Moving rook from H1 loses white kingside
        assert_eq!(CASTLING_RIGHTS_MASK[Square::H1.index()], ALL_CASTLING ^ WHITE_OO);
        // Random square preserves all rights
        assert_eq!(CASTLING_RIGHTS_MASK[Square::E4.index()], ALL_CASTLING);
    }
}
