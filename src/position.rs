//! Chess position representation and move application.
//!
//! Uses a hybrid [bitboard](https://www.chessprogramming.org/Bitboards) + mailbox
//! representation: bitboards for efficient set operations, mailbox (`[Piece; 64]`) for
//! O(1) piece-on-square lookup. Before each move, reversible state is saved on a
//! fixed-size stack and restored on unmake.
//!
//! See [Board Representation](https://www.chessprogramming.org/Board_Representation).

use crate::bitboard::*;
use crate::eval;
use crate::movegen;
use crate::types::*;
use crate::zobrist::ZOBRIST;

// ============================================================
// FEN parse errors
// ============================================================

/// Errors that can occur when parsing a
/// [FEN](https://www.chessprogramming.org/Forsyth-Edwards_Notation) string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FenParseError {
    /// FEN must have exactly 6 space-separated fields.
    WrongFieldCount { found: usize },
    /// Piece placement must have exactly 8 ranks separated by '/'.
    WrongRankCount { found: usize },
    /// Each rank must sum to exactly 8 squares (pieces + digit gaps).
    RankLengthMismatch { rank: usize, count: u8 },
    /// Unrecognized character in piece placement field.
    InvalidPieceChar { ch: char },
    /// Side to move must be "w" or "b".
    InvalidSideToMove { found: String },
    /// Unrecognized character in castling rights field.
    InvalidCastlingChar { ch: char },
    /// En passant square is not a valid square name.
    InvalidEpSquare { found: String },
    /// Halfmove clock is not a valid u8.
    InvalidHalfmoveClock { found: String },
    /// Fullmove number is not a valid u16.
    InvalidFullmoveNumber { found: String },

    // --- Pre-filter ---
    /// FEN string is too long (max 100 chars).
    TooLong { len: usize },
    /// FEN contains non-ASCII or non-printable characters.
    InvalidChars,

    // --- Syntactic (phase 1) ---
    /// Adjacent digits in piece placement (e.g. "35" instead of "8").
    AdjacentDigits { rank: usize },
    /// En passant square must be on rank 3 or 6 matching side to move.
    InvalidEpRank { found: String, expected_rank: u8 },
    /// Fullmove number must be >= 1.
    FullmoveNumberZero,

    // --- Semantic (phase 2) ---
    /// Each side must have exactly one king.
    InvalidKingCount { color: &'static str, count: u32 },
    /// The side NOT to move must not have its king in check.
    OpponentKingInCheck,
    /// Pawns cannot be placed on rank 1 or rank 8.
    PawnsOnBackRank,
    /// Each side can have at most 8 pawns.
    TooManyPawns { color: &'static str, count: u32 },
    /// Each side can have at most 16 pieces total.
    TooManyPieces { color: &'static str, count: u32 },
    /// Extra pieces beyond starting material exceed missing pawns.
    TooManyPromotions { color: &'static str, extra: u32, missing_pawns: u32 },
    /// Castling rights require the king and rook on their starting squares.
    CastlingRightsIncoherent { detail: &'static str },
    /// En passant square set but no enemy pawn behind it.
    EpSquareNoPawn { ep: String },
}

impl std::fmt::Display for FenParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WrongFieldCount { found } =>
                write!(f, "expected 6 FEN fields, found {found}"),
            Self::WrongRankCount { found } =>
                write!(f, "expected 8 ranks in placement, found {found}"),
            Self::RankLengthMismatch { rank, count } =>
                write!(f, "rank {rank} sums to {count} squares, expected 8"),
            Self::InvalidPieceChar { ch } =>
                write!(f, "invalid piece character '{ch}'"),
            Self::InvalidSideToMove { found } =>
                write!(f, "invalid side to move '{found}', expected 'w' or 'b'"),
            Self::InvalidCastlingChar { ch } =>
                write!(f, "invalid castling character '{ch}'"),
            Self::InvalidEpSquare { found } =>
                write!(f, "invalid en passant square '{found}'"),
            Self::InvalidHalfmoveClock { found } =>
                write!(f, "invalid halfmove clock '{found}'"),
            Self::InvalidFullmoveNumber { found } =>
                write!(f, "invalid fullmove number '{found}'"),
            Self::TooLong { len } =>
                write!(f, "FEN string too long ({len} chars, max 100)"),
            Self::InvalidChars =>
                write!(f, "FEN contains non-ASCII or non-printable characters"),
            Self::AdjacentDigits { rank } =>
                write!(f, "adjacent digits in rank {rank} (invalid FEN)"),
            Self::InvalidEpRank { found, expected_rank } =>
                write!(f, "en passant square '{found}' must be on rank {expected_rank}"),
            Self::FullmoveNumberZero =>
                write!(f, "fullmove number must be >= 1"),
            Self::InvalidKingCount { color, count } =>
                write!(f, "{color} has {count} king(s), expected exactly 1"),
            Self::OpponentKingInCheck =>
                write!(f, "opponent's king is in check (illegal position)"),
            Self::PawnsOnBackRank =>
                write!(f, "pawns on rank 1 or 8 (illegal position)"),
            Self::TooManyPawns { color, count } =>
                write!(f, "{color} has {count} pawns, maximum is 8"),
            Self::TooManyPieces { color, count } =>
                write!(f, "{color} has {count} pieces, maximum is 16"),
            Self::TooManyPromotions { color, extra, missing_pawns } =>
                write!(f, "{color} has {extra} extra piece(s) but only {missing_pawns} missing pawn(s)"),
            Self::CastlingRightsIncoherent { detail } =>
                write!(f, "castling rights incoherent: {detail}"),
            Self::EpSquareNoPawn { ep } =>
                write!(f, "en passant square {ep} but no enemy pawn behind it"),
        }
    }
}

// ============================================================
// State history (saved before each make_move, restored on unmake)
// ============================================================

/// Reversible position state saved before each [`Position::make_move`] and
/// restored on [`Position::unmake_move`].
#[derive(Clone, Copy)]
pub struct StateHistory {
    pub castling_rights: u8,
    pub ep_square: Square,
    pub halfmove_clock: u8,
    pub key: u64,
    pub pawn_key: u64,
    pub non_pawn_key: [u64; 2],
    pub minor_key: u64,
    pub checkers: u64,
    pub pinned: u64,
    pub threats: u64,
    pub captured_piece: Piece,
    /// Plies since last null move (reset in push_null, incremented in make_move).
    pub plies_from_null: u16,
    /// Incremental repetition detection.
    /// Positive = distance to first repeat (twofold). Negative = threefold.
    pub repetition: i32,
    /// Incremental PeSTO: midgame score (White positive, Black negative).
    pub psq_mg: i32,
    /// Incremental PeSTO: endgame score (White positive, Black negative).
    pub psq_eg: i32,
    /// Incremental PeSTO: game phase (sum of PHASE_INC for all pieces).
    pub game_phase: i32,
}

impl Default for StateHistory {
    fn default() -> Self {
        StateHistory {
            castling_rights: 0,
            ep_square: Square::NONE,
            halfmove_clock: 0,
            key: 0,
            pawn_key: 0,
            non_pawn_key: [0; 2],
            minor_key: 0,
            checkers: 0,
            pinned: 0,
            threats: 0,
            captured_piece: Piece::NONE,
            plies_from_null: 0,
            repetition: 0,
            psq_mg: 0,
            psq_eg: 0,
            game_phase: 0,
        }
    }
}

// ============================================================
// Position
// ============================================================

/// Full chess position: bitboards + mailbox + game state.
#[derive(Clone)]
pub struct Position {
    /// Bitboard per piece-color (indexed by [`Piece`] value 0..11).
    pub pieces: [u64; 12],
    /// Occupancy bitboards: `[WHITE, BLACK, BOTH]`.
    pub occupancies: [u64; 3],
    /// Mailbox: piece on each square (or [`Piece::NONE`]).
    pub board: [Piece; 64],

    pub side_to_move: Color,
    pub castling_rights: u8,
    /// When set, UCI/FEN use Shredder encoding (king-captures-rook / file letters).
    pub chess960: bool,
    /// Starting square of the castling rook, indexed by [`castling_idx`].
    pub castling_rook: [Square; CASTLING_NB],
    /// Squares that must be empty for each right (origins excluded).
    pub castling_path: [u64; CASTLING_NB],
    /// Squares the king occupies or crosses (must not be attacked).
    pub castling_king_path: [u64; CASTLING_NB],
    /// Rights that survive a move from/to each square.
    pub castling_rights_mask: [u8; 64],
    pub ep_square: Square,
    pub halfmove_clock: u8,
    pub fullmove_number: u16,

    /// Zobrist hash of the full position (incrementally updated).
    pub key: u64,
    /// Zobrist hash of pawns only (for pawn hash table).
    pub pawn_key: u64,
    /// Zobrist hash of non-pawn pieces per color (N, B, R, Q, K).
    pub non_pawn_key: [u64; 2],
    /// Zobrist hash of minor pieces + king (N, B, K) for both colors combined.
    pub minor_key: u64,
    /// Bitboard of pieces giving check to the side to move.
    pub checkers: u64,
    /// Bitboard of our pieces [pinned](https://www.chessprogramming.org/Pin)
    /// to our king by enemy sliders.
    pub pinned: u64,
    /// All squares attacked by the opponent (king excluded from occupancy
    /// so sliders x-ray through it).
    pub threats: u64,

    /// Plies since last null move (reset in push_null, incremented in make_move).
    pub plies_from_null: u16,
    /// Incremental repetition detection.
    /// Positive = distance to first repeat (twofold). Negative = threefold.
    pub repetition: i32,
    /// Incremental PeSTO: midgame score (White positive, Black negative).
    pub psq_mg: i32,
    /// Incremental PeSTO: endgame score (White positive, Black negative).
    pub psq_eg: i32,
    /// Incremental PeSTO: game phase (sum of PHASE_INC for all pieces).
    pub game_phase: i32,

    history: [StateHistory; 1024],
    pub ply: usize,
}

impl Position {
    // ============================================================
    // Piece accessors
    // ============================================================

    #[allow(dead_code)]
    #[inline(always)]
    pub fn piece_bb(&self, pc: Piece) -> u64 {
        self.pieces[pc.index()]
    }

    #[inline(always)]
    pub fn piece_type_bb(&self, pt: PieceType, c: Color) -> u64 {
        self.pieces[Piece::new(pt, c).index()]
    }

    #[inline(always)]
    pub fn color_bb(&self, c: Color) -> u64 {
        self.occupancies[c.index()]
    }

    #[inline(always)]
    pub fn occupied(&self) -> u64 {
        self.occupancies[2]
    }

    #[inline(always)]
    pub fn king_sq(&self, c: Color) -> Square {
        lsb(self.pieces[Piece::new(PieceType::King, c).index()])
    }

    /// Switch UCI/FEN castling encoding without touching the board.
    #[inline]
    pub fn set_chess960(&mut self, on: bool) {
        self.chess960 = on;
    }

    #[inline(always)]
    pub fn castle_rook_sq(&self, right: u8) -> Square {
        debug_assert!(right == WHITE_OO || right == WHITE_OOO || right == BLACK_OO || right == BLACK_OOO);
        self.castling_rook[castling_idx(right)]
    }

    fn back_rank(us: Color) -> u8 {
        match us {
            Color::White => 0,
            Color::Black => 7,
        }
    }

    fn set_castling_from_token(
        &mut self, us: Color, kingside: bool, rook_file: Option<u8>,
    ) -> Result<(), FenParseError> {
        let king_bb = self.pieces[Piece::new(PieceType::King, us).index()];
        if king_bb == 0 {
            let color = if us == Color::White { "white" } else { "black" };
            return Err(FenParseError::InvalidKingCount { color, count: 0 });
        }
        let king = self.king_sq(us);
        let rank = Self::back_rank(us);
        if king.rank() != rank {
            return Err(FenParseError::CastlingRightsIncoherent {
                detail: "castling king is not on its back rank",
            });
        }
        let rook_sq = if let Some(file) = rook_file {
            Square::new(file, rank)
        } else {
            match self.outermost_rook(us, kingside, king) {
                Some(sq) => sq,
                None => {
                    return Err(FenParseError::CastlingRightsIncoherent {
                        detail: "castling right with no rook on the back rank",
                    });
                }
            }
        };
        let rook = Piece::new(PieceType::Rook, us);
        if self.board[rook_sq.index()] != rook {
            return Err(FenParseError::CastlingRightsIncoherent {
                detail: "castling file does not hold a friendly rook",
            });
        }
        if rook_file.is_some() && rook_sq.file() == king.file() {
            return Err(FenParseError::CastlingRightsIncoherent {
                detail: "castling rook file overlaps the king",
            });
        }
        self.set_castling_right(us, rook_sq);
        Ok(())
    }

    fn set_castling_from_file(&mut self, us: Color, file: u8) -> Result<(), FenParseError> {
        debug_assert!(file < 8);
        let king_bb = self.pieces[Piece::new(PieceType::King, us).index()];
        if king_bb == 0 {
            let color = if us == Color::White { "white" } else { "black" };
            return Err(FenParseError::InvalidKingCount { color, count: 0 });
        }
        let king = self.king_sq(us);
        if file == king.file() {
            return Err(FenParseError::CastlingRightsIncoherent {
                detail: "castling rook file overlaps the king",
            });
        }
        let kingside = file > king.file();
        self.set_castling_from_token(us, kingside, Some(file))
    }

    fn outermost_rook(&self, us: Color, kingside: bool, king: Square) -> Option<Square> {
        let rank = Self::back_rank(us);
        let rook = Piece::new(PieceType::Rook, us);
        if kingside {
            let mut file = 7u8;
            while file > king.file() {
                let sq = Square::new(file, rank);
                if self.board[sq.index()] == rook {
                    return Some(sq);
                }
                file -= 1;
            }
        } else {
            let mut file = 0u8;
            while file < king.file() {
                let sq = Square::new(file, rank);
                if self.board[sq.index()] == rook {
                    return Some(sq);
                }
                file += 1;
            }
        }
        None
    }

    fn set_castling_right(&mut self, us: Color, rook_from: Square) {
        let king_from = self.king_sq(us);
        debug_assert!(king_from.rank() == rook_from.rank());
        debug_assert!(king_from.file() != rook_from.file());
        let kingside = rook_from.file() > king_from.file();
        let cr = match (us, kingside) {
            (Color::White, true) => WHITE_OO,
            (Color::White, false) => WHITE_OOO,
            (Color::Black, true) => BLACK_OO,
            (Color::Black, false) => BLACK_OOO,
        };
        let idx = castling_idx(cr);
        let rank = king_from.rank();
        let king_to = Square::new(if kingside { 6 } else { 2 }, rank);
        let rook_to = Square::new(if kingside { 5 } else { 3 }, rank);

        self.castling_rights |= cr;
        self.castling_rook[idx] = rook_from;
        self.castling_path[idx] = (between_bb(king_from, king_to) | between_bb(rook_from, rook_to))
            & !king_from.bb()
            & !rook_from.bb();
        self.castling_king_path[idx] = between_bb(king_from, king_to) | king_from.bb();
        self.castling_rights_mask[king_from.index()] &= !cr;
        self.castling_rights_mask[rook_from.index()] &= !cr;
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub fn piece_on(&self, sq: Square) -> Piece {
        self.board[sq.index()]
    }

    /// Non-pawn, non-king material for both sides (SEE values).
    /// Typical range: 0 (K vs K) to ~6400 (opening).
    #[inline]
    pub fn material(&self) -> i32 {
        320 * (self.piece_type_bb(PieceType::Knight, Color::White)
             | self.piece_type_bb(PieceType::Knight, Color::Black)).count_ones() as i32
        + 330 * (self.piece_type_bb(PieceType::Bishop, Color::White)
               | self.piece_type_bb(PieceType::Bishop, Color::Black)).count_ones() as i32
        + 500 * (self.piece_type_bb(PieceType::Rook, Color::White)
               | self.piece_type_bb(PieceType::Rook, Color::Black)).count_ones() as i32
        + 900 * (self.piece_type_bb(PieceType::Queen, Color::White)
               | self.piece_type_bb(PieceType::Queen, Color::Black)).count_ones() as i32
    }

    /// Returns true if `c` has any piece besides pawns and king.
    #[inline(always)]
    pub fn has_non_pawn_material(&self, c: Color) -> bool {
        (self.color_bb(c)
            ^ self.piece_type_bb(PieceType::Pawn, c)
            ^ self.piece_type_bb(PieceType::King, c))
            != 0
    }

    // ============================================================
    // Piece manipulation helpers (with Zobrist updates)
    // ============================================================

    #[inline(always)]
    fn put_piece(&mut self, sq: Square, pc: Piece) {
        debug_assert!(sq.0 < 64, "put_piece: sq {} invalid", sq.0);
        debug_assert!(pc.0 < 12, "put_piece: piece {} invalid", pc.0);
        debug_assert!(self.board[sq.index()] == Piece::NONE,
            "put_piece: sq {} already has {:?}", sq.0, self.board[sq.index()]);
        debug_assert!(self.pieces[pc.index()] & sq.bb() == 0,
            "put_piece: piece bitboard already has sq {}", sq.0);
        let bb = sq.bb();
        self.pieces[pc.index()] |= bb;
        self.occupancies[pc.color().index()] |= bb;
        self.occupancies[2] |= bb;
        self.board[sq.index()] = pc;
        self.key ^= ZOBRIST.pieces[pc.index()][sq.index()];
        self.psq_add(pc, sq);
        if pc.piece_type() == PieceType::Pawn {
            self.pawn_key ^= ZOBRIST.pieces[pc.index()][sq.index()];
        } else {
            self.non_pawn_key[pc.color().index()] ^= ZOBRIST.pieces[pc.index()][sq.index()];
            let pt = pc.piece_type();
            if pt == PieceType::Knight || pt == PieceType::Bishop || pt == PieceType::King {
                self.minor_key ^= ZOBRIST.pieces[pc.index()][sq.index()];
            }
        }
    }

    #[inline(always)]
    fn remove_piece(&mut self, sq: Square, pc: Piece) {
        debug_assert!(sq.0 < 64, "remove_piece: sq {} invalid", sq.0);
        debug_assert!(pc.0 < 12, "remove_piece: piece {} invalid", pc.0);
        debug_assert!(self.board[sq.index()] == pc,
            "remove_piece: sq {} has {:?}, expected {:?}", sq.0, self.board[sq.index()], pc);
        debug_assert!(self.pieces[pc.index()] & sq.bb() != 0,
            "remove_piece: piece bitboard missing sq {}", sq.0);
        let bb = sq.bb();
        self.pieces[pc.index()] ^= bb;
        self.occupancies[pc.color().index()] ^= bb;
        self.occupancies[2] ^= bb;
        self.board[sq.index()] = Piece::NONE;
        self.key ^= ZOBRIST.pieces[pc.index()][sq.index()];
        self.psq_sub(pc, sq);
        if pc.piece_type() == PieceType::Pawn {
            self.pawn_key ^= ZOBRIST.pieces[pc.index()][sq.index()];
        } else {
            self.non_pawn_key[pc.color().index()] ^= ZOBRIST.pieces[pc.index()][sq.index()];
            let pt = pc.piece_type();
            if pt == PieceType::Knight || pt == PieceType::Bishop || pt == PieceType::King {
                self.minor_key ^= ZOBRIST.pieces[pc.index()][sq.index()];
            }
        }
    }

    #[inline(always)]
    fn move_piece(&mut self, from: Square, to: Square, pc: Piece) {
        debug_assert!(from.0 < 64 && to.0 < 64, "move_piece: from {} to {}", from.0, to.0);
        debug_assert!(pc.0 < 12, "move_piece: piece {} invalid", pc.0);
        debug_assert!(self.board[from.index()] == pc,
            "move_piece: from {} has {:?}, expected {:?}", from.0, self.board[from.index()], pc);
        debug_assert!(self.board[to.index()] == Piece::NONE,
            "move_piece: to {} not empty, has {:?}", to.0, self.board[to.index()]);
        let from_to_bb = from.bb() ^ to.bb();
        self.pieces[pc.index()] ^= from_to_bb;
        self.occupancies[pc.color().index()] ^= from_to_bb;
        self.occupancies[2] ^= from_to_bb;
        self.board[from.index()] = Piece::NONE;
        self.board[to.index()] = pc;
        self.key ^= ZOBRIST.pieces[pc.index()][from.index()]
            ^ ZOBRIST.pieces[pc.index()][to.index()];
        self.psq_move(pc, from, to);
        if pc.piece_type() == PieceType::Pawn {
            self.pawn_key ^= ZOBRIST.pieces[pc.index()][from.index()]
                ^ ZOBRIST.pieces[pc.index()][to.index()];
        } else {
            self.non_pawn_key[pc.color().index()] ^= ZOBRIST.pieces[pc.index()][from.index()]
                ^ ZOBRIST.pieces[pc.index()][to.index()];
            let pt = pc.piece_type();
            if pt == PieceType::Knight || pt == PieceType::Bishop || pt == PieceType::King {
                self.minor_key ^= ZOBRIST.pieces[pc.index()][from.index()]
                    ^ ZOBRIST.pieces[pc.index()][to.index()];
            }
        }
    }

    // Versions without Zobrist updates (for unmake)
    #[inline(always)]
    fn put_piece_nz(&mut self, sq: Square, pc: Piece) {
        debug_assert!(sq.0 < 64, "put_piece_nz: sq {} invalid", sq.0);
        debug_assert!(pc.0 < 12, "put_piece_nz: piece {} invalid", pc.0);
        debug_assert!(self.board[sq.index()] == Piece::NONE,
            "put_piece_nz: sq {} already has {:?}", sq.0, self.board[sq.index()]);
        let bb = sq.bb();
        self.pieces[pc.index()] |= bb;
        self.occupancies[pc.color().index()] |= bb;
        self.occupancies[2] |= bb;
        self.board[sq.index()] = pc;
    }

    #[inline(always)]
    fn remove_piece_nz(&mut self, sq: Square, pc: Piece) {
        debug_assert!(sq.0 < 64, "remove_piece_nz: sq {} invalid", sq.0);
        debug_assert!(pc.0 < 12, "remove_piece_nz: piece {} invalid", pc.0);
        debug_assert!(self.board[sq.index()] == pc,
            "remove_piece_nz: sq {} has {:?}, expected {:?}", sq.0, self.board[sq.index()], pc);
        debug_assert!(self.pieces[pc.index()] & sq.bb() != 0,
            "remove_piece_nz: piece bitboard missing sq {}", sq.0);
        let bb = sq.bb();
        self.pieces[pc.index()] ^= bb;
        self.occupancies[pc.color().index()] ^= bb;
        self.occupancies[2] ^= bb;
        self.board[sq.index()] = Piece::NONE;
    }

    #[inline(always)]
    fn move_piece_nz(&mut self, from: Square, to: Square, pc: Piece) {
        debug_assert!(from.0 < 64 && to.0 < 64, "move_piece_nz: from {} to {}", from.0, to.0);
        debug_assert!(pc.0 < 12, "move_piece_nz: piece {} invalid", pc.0);
        debug_assert!(self.board[from.index()] == pc,
            "move_piece_nz: from {} has {:?}, expected {:?}", from.0, self.board[from.index()], pc);
        debug_assert!(self.board[to.index()] == Piece::NONE,
            "move_piece_nz: to {} not empty, has {:?}", to.0, self.board[to.index()]);
        let from_to_bb = from.bb() ^ to.bb();
        self.pieces[pc.index()] ^= from_to_bb;
        self.occupancies[pc.color().index()] ^= from_to_bb;
        self.occupancies[2] ^= from_to_bb;
        self.board[from.index()] = Piece::NONE;
        self.board[to.index()] = pc;
    }

    // ============================================================
    // Check info (checkers + pinned)
    // ============================================================

    /// Recompute [`checkers`](Self::checkers) and [`pinned`](Self::pinned)
    /// for the side to move. Called after every make.
    pub fn set_check_info(&mut self) {
        let us = self.side_to_move;
        let them = !us;
        let ksq = self.king_sq(us);
        let occ = self.occupied();

        // Non-slider checkers
        let mut checkers = knight_attacks(ksq) & self.piece_type_bb(PieceType::Knight, them);
        checkers |= pawn_attacks(ksq, us) & self.piece_type_bb(PieceType::Pawn, them);

        // Slider checkers and pinned pieces
        let mut pinned = 0u64;
        let their_bishops_queens = self.piece_type_bb(PieceType::Bishop, them)
            | self.piece_type_bb(PieceType::Queen, them);
        let their_rooks_queens = self.piece_type_bb(PieceType::Rook, them)
            | self.piece_type_bb(PieceType::Queen, them);

        // Diagonal snipers
        let mut snipers = bishop_attacks(ksq, self.color_bb(them)) & their_bishops_queens;
        while snipers != 0 {
            let sniper_sq = pop_lsb(&mut snipers);
            let between = between_bb(ksq, sniper_sq) & occ & !sniper_sq.bb();
            let blockers = between & !sniper_sq.bb();
            if blockers == 0 {
                checkers |= sniper_sq.bb();
            } else if !more_than_one(blockers) {
                // Exactly one blocker = potential pin (only if it's our piece)
                if blockers & self.color_bb(us) != 0 {
                    pinned |= blockers;
                }
            }
        }

        // Orthogonal snipers
        snipers = rook_attacks(ksq, self.color_bb(them)) & their_rooks_queens;
        while snipers != 0 {
            let sniper_sq = pop_lsb(&mut snipers);
            let between = between_bb(ksq, sniper_sq) & occ & !sniper_sq.bb();
            let blockers = between & !sniper_sq.bb();
            if blockers == 0 {
                checkers |= sniper_sq.bb();
            } else if !more_than_one(blockers) && blockers & self.color_bb(us) != 0 {
                pinned |= blockers;
            }
        }

        self.checkers = checkers;
        self.pinned = pinned;
    }

    /// Compute all squares attacked by the opponent (side NOT to move).
    /// The our king is excluded from occupancy so sliders "see through" it,
    /// enabling fast king-move legality checks: king can't move to a threatened square.
    pub fn update_threats(&mut self) {
        let them = !self.side_to_move;
        // Exclude our king from occupancy so sliders x-ray through it
        let occ = self.occupied() ^ self.piece_type_bb(PieceType::King, self.side_to_move);

        // Pawn attacks (setwise — all pawns at once)
        let their_pawns = self.piece_type_bb(PieceType::Pawn, them);
        let mut threats = if them == Color::White {
            shift_north_east(their_pawns) | shift_north_west(their_pawns)
        } else {
            shift_south_east(their_pawns) | shift_south_west(their_pawns)
        };

        // Knight attacks
        let mut knights = self.piece_type_bb(PieceType::Knight, them);
        while knights != 0 {
            let sq = pop_lsb(&mut knights);
            threats |= knight_attacks(sq);
        }

        // Slider attacks
        threats |= crate::simd_attacks::slider_attacks_setwise(
            self.piece_type_bb(PieceType::Bishop, them),
            self.piece_type_bb(PieceType::Rook, them),
            self.piece_type_bb(PieceType::Queen, them),
            occ,
        );

        // King attacks
        threats |= king_attacks(self.king_sq(them));

        self.threats = threats;
    }

    /// Threats bitboard from the position before the last make_move.
    /// Used by static history to get the threats when the opponent chose their move.
    #[inline]
    pub fn prior_threats(&self) -> u64 {
        debug_assert!(self.ply > 0, "prior_threats: ply is 0");
        self.history[self.ply - 1].threats
    }

    /// Piece captured by the last move (Piece::NONE if not a capture).
    #[inline]
    pub fn prior_captured_piece(&self) -> Piece {
        debug_assert!(self.ply > 0, "prior_captured_piece: ply is 0");
        self.history[self.ply - 1].captured_piece
    }

    // ============================================================
    // Incremental PeSTO helpers
    // ============================================================

    /// Add a piece's PeSTO contribution (called from put_piece).
    #[inline(always)]
    fn psq_add(&mut self, pc: Piece, sq: Square) {
        self.psq_mg += eval::MG_SIGNED[pc.index()][sq.index()];
        self.psq_eg += eval::EG_SIGNED[pc.index()][sq.index()];
        self.game_phase += eval::PHASE_INC[pc.piece_type() as usize];
    }

    /// Remove a piece's PeSTO contribution (called from remove_piece).
    #[inline(always)]
    fn psq_sub(&mut self, pc: Piece, sq: Square) {
        self.psq_mg -= eval::MG_SIGNED[pc.index()][sq.index()];
        self.psq_eg -= eval::EG_SIGNED[pc.index()][sq.index()];
        self.game_phase -= eval::PHASE_INC[pc.piece_type() as usize];
    }

    /// Move a piece's PeSTO contribution (from → to).
    #[inline(always)]
    fn psq_move(&mut self, pc: Piece, from: Square, to: Square) {
        let mg = &eval::MG_SIGNED[pc.index()];
        let eg = &eval::EG_SIGNED[pc.index()];
        self.psq_mg += mg[to.index()] - mg[from.index()];
        self.psq_eg += eg[to.index()] - eg[from.index()];
        // Phase unchanged (same piece type moving, not added/removed)
    }

    /// Compute PeSTO scores from scratch (for init and debug validation).
    pub fn compute_psq(&self) -> (i32, i32, i32) {
        let mut mg = 0i32;
        let mut eg = 0i32;
        let mut phase = 0i32;
        for pc_idx in 0..12u8 {
            let pc = Piece(pc_idx);
            let sign = if pc.color() == Color::White { 1 } else { -1 };
            let mut bb = self.pieces[pc_idx as usize];
            while bb != 0 {
                let sq = pop_lsb(&mut bb);
                mg += sign * eval::MG_TABLE[pc_idx as usize][sq.index()];
                eg += sign * eval::EG_TABLE[pc_idx as usize][sq.index()];
                phase += eval::PHASE_INC[pc.piece_type() as usize];
            }
        }
        (mg, eg, phase)
    }

    /// Return the tapered PeSTO eval from the side-to-move's perspective.
    /// O(1) — uses the incrementally maintained psq_mg/psq_eg/game_phase.
    #[inline]
    pub fn lazy_eval(&self) -> i32 {
        let mg_phase = self.game_phase.min(eval::TOTAL_PHASE);
        let eg_phase = eval::TOTAL_PHASE - mg_phase;
        let score = (self.psq_mg * mg_phase + self.psq_eg * eg_phase) / eval::TOTAL_PHASE;
        let tempo = (eval::TEMPO_MG * mg_phase + eval::TEMPO_EG * eg_phase) / eval::TOTAL_PHASE;
        if self.side_to_move == Color::White { score + tempo } else { -score + tempo }
    }

    // ============================================================
    // FEN parsing
    // ============================================================

    /// Parse a [FEN](https://www.chessprogramming.org/Forsyth-Edwards_Notation) string
    /// into a fully initialized position (with Zobrist keys, check info, and threats).
    pub fn from_fen(fen: &str) -> Result<Position, FenParseError> {
        Self::from_fen_ex(fen, false)
    }

    /// Like [`from_fen`], and records whether UCI/FEN should use Chess960 encoding.
    pub fn from_fen_ex(fen: &str, chess960: bool) -> Result<Position, FenParseError> {
        // Pre-filter: reject excessively long or non-ASCII input
        if fen.len() > 100 {
            return Err(FenParseError::TooLong { len: fen.len() });
        }
        if fen.bytes().any(|b| !b.is_ascii_graphic() && b != b' ') {
            return Err(FenParseError::InvalidChars);
        }

        let mut pos = Position {
            pieces: [0; 12],
            occupancies: [0; 3],
            board: [Piece::NONE; 64],
            side_to_move: Color::White,
            castling_rights: 0,
            chess960: false,
            castling_rook: [Square::NONE; CASTLING_NB],
            castling_path: [0; CASTLING_NB],
            castling_king_path: [0; CASTLING_NB],
            castling_rights_mask: [ALL_CASTLING; 64],
            ep_square: Square::NONE,
            halfmove_clock: 0,
            fullmove_number: 1,
            key: 0,
            pawn_key: 0,
            non_pawn_key: [0; 2],
            minor_key: 0,
            checkers: 0,
            pinned: 0,
            threats: 0,
            psq_mg: 0,
            psq_eg: 0,
            game_phase: 0,
            plies_from_null: 0,
            repetition: 0,
            history: [StateHistory::default(); 1024],
            ply: 0,
        };

        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.len() != 6 {
            return Err(FenParseError::WrongFieldCount { found: parts.len() });
        }

        // 1. Piece placement (rank 8 to rank 1)
        let ranks: Vec<&str> = parts[0].split('/').collect();
        if ranks.len() != 8 {
            return Err(FenParseError::WrongRankCount { found: ranks.len() });
        }

        for (i, rank_str) in ranks.iter().enumerate() {
            let rank = 7 - i as u8; // FEN starts from rank 8
            let mut file: u8 = 0;
            let mut prev_was_digit = false;
            for ch in rank_str.chars() {
                match ch {
                    '1'..='8' => {
                        if prev_was_digit {
                            return Err(FenParseError::AdjacentDigits { rank: 8 - i });
                        }
                        prev_was_digit = true;
                        file += (ch as u8) - b'0';
                    }
                    _ => {
                        prev_was_digit = false;
                        if let Some(pc) = Piece::from_char(ch) {
                            let sq = Square::new(file, rank);
                            let bb = sq.bb();
                            pos.pieces[pc.index()] |= bb;
                            pos.occupancies[pc.color().index()] |= bb;
                            pos.occupancies[2] |= bb;
                            pos.board[sq.index()] = pc;
                            file += 1;
                        } else {
                            return Err(FenParseError::InvalidPieceChar { ch });
                        }
                    }
                }
            }
            if file != 8 {
                return Err(FenParseError::RankLengthMismatch { rank: 8 - i, count: file });
            }
        }

        // 2. Side to move
        pos.side_to_move = match parts[1] {
            "w" => Color::White,
            "b" => Color::Black,
            _ => return Err(FenParseError::InvalidSideToMove { found: parts[1].to_string() }),
        };

        // 3. Castling rights (KQkq = outermost rook, A-H/a-h = Shredder file)
        pos.chess960 = chess960;
        if parts[2] != "-" {
            if parts[2].len() > 4 {
                return Err(FenParseError::InvalidCastlingChar { ch: parts[2].chars().next().unwrap_or('?') });
            }
            for ch in parts[2].chars() {
                match ch {
                    'K' => pos.set_castling_from_token(Color::White, true, None)?,
                    'Q' => pos.set_castling_from_token(Color::White, false, None)?,
                    'k' => pos.set_castling_from_token(Color::Black, true, None)?,
                    'q' => pos.set_castling_from_token(Color::Black, false, None)?,
                    'A'..='H' => {
                        let file = ch as u8 - b'A';
                        pos.set_castling_from_file(Color::White, file)?;
                    }
                    'a'..='h' => {
                        let file = ch as u8 - b'a';
                        pos.set_castling_from_file(Color::Black, file)?;
                    }
                    _ => return Err(FenParseError::InvalidCastlingChar { ch }),
                }
            }
        }

        // 4. En passant square
        if parts[3] != "-" {
            pos.ep_square = Square::from_string(parts[3])
                .ok_or_else(|| FenParseError::InvalidEpSquare { found: parts[3].to_string() })?;
            // EP must be on rank 6 (white to move) or rank 3 (black to move)
            let expected_rank = if pos.side_to_move == Color::White { 5u8 } else { 2u8 };
            if pos.ep_square.rank() != expected_rank {
                return Err(FenParseError::InvalidEpRank {
                    found: parts[3].to_string(),
                    expected_rank: expected_rank + 1, // 1-indexed for display
                });
            }
        }

        // 5. Halfmove clock
        pos.halfmove_clock = parts[4].parse::<u8>()
            .map_err(|_| FenParseError::InvalidHalfmoveClock { found: parts[4].to_string() })?;

        // 6. Fullmove number
        pos.fullmove_number = parts[5].parse::<u16>()
            .map_err(|_| FenParseError::InvalidFullmoveNumber { found: parts[5].to_string() })?;
        if pos.fullmove_number == 0 {
            return Err(FenParseError::FullmoveNumberZero);
        }

        // 7. Validate position legality (release-mode checks)
        pos.validate_legality()?;

        // 8. Keep the en passant square only where a pawn stands ready to take on it:
        // that is the condition make_move records it on, and hashes it on, so a
        // position read from a FEN gets the same key as the one the moves would have
        // built. The legality check above still sees the square as written.
        if pos.ep_square != Square::NONE {
            let us = pos.side_to_move;
            if pawn_attacks(pos.ep_square, !us) & pos.piece_type_bb(PieceType::Pawn, us) == 0 {
                pos.ep_square = Square::NONE;
            }
        }

        // Compute Zobrist key from scratch
        pos.key = pos.compute_key();
        pos.pawn_key = pos.compute_pawn_key();
        pos.non_pawn_key = pos.compute_non_pawn_key();
        pos.minor_key = pos.compute_minor_key();

        // Compute incremental PeSTO from scratch
        let (mg, eg, phase) = pos.compute_psq();
        pos.psq_mg = mg;
        pos.psq_eg = eg;
        pos.game_phase = phase;

        // Compute check info and threats
        pos.set_check_info();
        pos.update_threats();

        #[cfg(debug_assertions)]
        pos.check_validity();

        Ok(pos)
    }

    /// Serialize the position back to a FEN string.
    pub fn to_fen(&self) -> String {
        let mut fen = String::new();

        // 1. Piece placement
        for rank in (0..8).rev() {
            let mut empty = 0;
            for file in 0..8u8 {
                let sq = Square::new(file, rank);
                let pc = self.board[sq.index()];
                if pc == Piece::NONE {
                    empty += 1;
                } else {
                    if empty > 0 {
                        fen.push(char::from_digit(empty, 10).unwrap());
                        empty = 0;
                    }
                    fen.push(pc.to_char());
                }
            }
            if empty > 0 {
                fen.push(char::from_digit(empty, 10).unwrap());
            }
            if rank > 0 {
                fen.push('/');
            }
        }

        // 2. Side to move
        fen.push(' ');
        fen.push(if self.side_to_move == Color::White { 'w' } else { 'b' });

        // 3. Castling rights
        fen.push(' ');
        if self.castling_rights == 0 {
            fen.push('-');
        } else if self.chess960 {
            if self.castling_rights & WHITE_OO != 0 {
                fen.push((b'A' + self.castle_rook_sq(WHITE_OO).file()) as char);
            }
            if self.castling_rights & WHITE_OOO != 0 {
                fen.push((b'A' + self.castle_rook_sq(WHITE_OOO).file()) as char);
            }
            if self.castling_rights & BLACK_OO != 0 {
                fen.push((b'a' + self.castle_rook_sq(BLACK_OO).file()) as char);
            }
            if self.castling_rights & BLACK_OOO != 0 {
                fen.push((b'a' + self.castle_rook_sq(BLACK_OOO).file()) as char);
            }
        } else {
            if self.castling_rights & WHITE_OO != 0 { fen.push('K'); }
            if self.castling_rights & WHITE_OOO != 0 { fen.push('Q'); }
            if self.castling_rights & BLACK_OO != 0 { fen.push('k'); }
            if self.castling_rights & BLACK_OOO != 0 { fen.push('q'); }
        }

        // 4. En passant
        fen.push(' ');
        if self.ep_square == Square::NONE {
            fen.push('-');
        } else {
            fen.push_str(&self.ep_square.to_string());
        }

        // 5-6. Halfmove clock and fullmove number
        fen.push_str(&format!(" {} {}", self.halfmove_clock, self.fullmove_number));

        fen
    }

    // ============================================================
    // Zobrist key computation from scratch (for validation)
    // ============================================================

    pub fn compute_key(&self) -> u64 {
        let mut key = 0u64;
        for pc in 0..12 {
            let mut bb = self.pieces[pc];
            while bb != 0 {
                let sq = pop_lsb(&mut bb);
                key ^= ZOBRIST.pieces[pc][sq.index()];
            }
        }
        if self.ep_square != Square::NONE {
            key ^= ZOBRIST.ep[self.ep_square.file() as usize];
        }
        key ^= ZOBRIST.castling[self.castling_rights as usize];
        if self.side_to_move == Color::Black {
            key ^= ZOBRIST.side;
        }
        key
    }

    fn compute_pawn_key(&self) -> u64 {
        let mut key = 0u64;
        for c in [Color::White, Color::Black] {
            let pc = Piece::new(PieceType::Pawn, c);
            let mut bb = self.pieces[pc.index()];
            while bb != 0 {
                let sq = pop_lsb(&mut bb);
                key ^= ZOBRIST.pieces[pc.index()][sq.index()];
            }
        }
        key
    }

    fn compute_non_pawn_key(&self) -> [u64; 2] {
        let mut keys = [0u64; 2];
        for c in [Color::White, Color::Black] {
            for pt in [PieceType::Knight, PieceType::Bishop, PieceType::Rook,
                       PieceType::Queen, PieceType::King] {
                let pc = Piece::new(pt, c);
                let mut bb = self.pieces[pc.index()];
                while bb != 0 {
                    let sq = pop_lsb(&mut bb);
                    keys[c.index()] ^= ZOBRIST.pieces[pc.index()][sq.index()];
                }
            }
        }
        keys
    }

    fn compute_minor_key(&self) -> u64 {
        let mut key = 0u64;
        for c in [Color::White, Color::Black] {
            for pt in [PieceType::Knight, PieceType::Bishop, PieceType::King] {
                let pc = Piece::new(pt, c);
                let mut bb = self.pieces[pc.index()];
                while bb != 0 {
                    let sq = pop_lsb(&mut bb);
                    key ^= ZOBRIST.pieces[pc.index()][sq.index()];
                }
            }
        }
        key
    }

    // ============================================================
    // make_move
    // ============================================================

    /// Apply a move to the position: save state, update bitboards/mailbox/Zobrist,
    /// switch side, then recompute check info and threats.
    pub fn make_move(&mut self, m: Move) {
        debug_assert!(m != Move::NONE && m != Move::NULL,
            "make_move: invalid move {:?}", m);
        debug_assert!(m.from_sq().0 < 64 && m.to_sq().0 < 64,
            "make_move: squares OOB from={} to={}", m.from_sq().0, m.to_sq().0);
        debug_assert!(self.ply < 1023,
            "make_move: ply {} overflow (BUG 4)", self.ply);
        debug_assert!(self.board[m.from_sq().index()] != Piece::NONE,
            "make_move: source sq {} empty", m.from_sq().0);
        debug_assert!(self.board[m.from_sq().index()].color() == self.side_to_move,
            "make_move: piece {:?} != stm {:?}",
            self.board[m.from_sq().index()], self.side_to_move);

        // Save state
        self.history[self.ply] = StateHistory {
            castling_rights: self.castling_rights,
            ep_square: self.ep_square,
            halfmove_clock: self.halfmove_clock,
            key: self.key,
            pawn_key: self.pawn_key,
            non_pawn_key: self.non_pawn_key,
            minor_key: self.minor_key,
            checkers: self.checkers,
            pinned: self.pinned,
            threats: self.threats,
            captured_piece: Piece::NONE,
            plies_from_null: self.plies_from_null,
            repetition: self.repetition,
            psq_mg: self.psq_mg,
            psq_eg: self.psq_eg,
            game_phase: self.game_phase,
        };

        let from = m.from_sq();
        let to = m.to_sq();
        let mt = m.move_type();
        let us = self.side_to_move;
        let them = !us;
        let moving_piece = self.board[from.index()];
        let captured = if mt == MT_CASTLING {
            Piece::NONE
        } else {
            self.board[to.index()]
        };

        // Capture validation (BUG 3: TT king capture). Castling is encoded as
        // king-captures-own-rook, so the destination holds a friendly rook.
        if mt != MT_CASTLING && captured != Piece::NONE {
            debug_assert!(captured.piece_type() != PieceType::King,
                "make_move: KING CAPTURE on {} (move={}, BUG 3)", to.0, m.to_uci());
            debug_assert!(captured.color() == them,
                "make_move: capturing own piece {:?} on {}", captured, to.0);
        }

        // Clear old EP from hash
        if self.ep_square != Square::NONE {
            self.key ^= ZOBRIST.ep[self.ep_square.file() as usize];
            self.ep_square = Square::NONE;
        }

        // Clear old castling from hash
        self.key ^= ZOBRIST.castling[self.castling_rights as usize];

        // Update halfmove clock
        if moving_piece.piece_type() == PieceType::Pawn || captured != Piece::NONE {
            self.halfmove_clock = 0;
        } else {
            self.halfmove_clock += 1;
        }
        self.plies_from_null += 1;

        match mt {
            MT_NORMAL => {
                if captured != Piece::NONE {
                    self.history[self.ply].captured_piece = captured;
                    self.remove_piece(to, captured);
                }
                self.move_piece(from, to, moving_piece);

                // Double pawn push: set EP square
                if moving_piece.piece_type() == PieceType::Pawn {
                    let diff = (to.0 as i8) - (from.0 as i8);
                    if diff.abs() == 16 {
                        let ep = Square((from.0 as i8 + diff / 2) as u8);
                        // Only set EP if an enemy pawn can actually capture
                        if pawn_attacks(ep, us) & self.piece_type_bb(PieceType::Pawn, them) != 0 {
                            self.ep_square = ep;
                            self.key ^= ZOBRIST.ep[ep.file() as usize];
                        }
                    }
                }
            }
            MT_PROMOTION => {
                debug_assert!(moving_piece.piece_type() == PieceType::Pawn,
                    "make_move promo: not a pawn {:?}", moving_piece);
                let promo_rank = if us == Color::White { 7u8 } else { 0u8 };
                debug_assert!(to.rank() == promo_rank,
                    "make_move promo: to rank {} != {}", to.rank(), promo_rank);
                let from_rank = if us == Color::White { 6u8 } else { 1u8 };
                debug_assert!(from.rank() == from_rank,
                    "make_move promo: from rank {} != {}", from.rank(), from_rank);
                let promo_type = m.promo_type();
                // Promotion piece must be Knight/Bishop/Rook/Queen
                debug_assert!(
                    promo_type == PieceType::Knight || promo_type == PieceType::Bishop
                    || promo_type == PieceType::Rook || promo_type == PieceType::Queen,
                    "make_move promo: invalid promo type {:?}", promo_type);
                let promo_piece = Piece::new(promo_type, us);

                if captured != Piece::NONE {
                    self.history[self.ply].captured_piece = captured;
                    self.remove_piece(to, captured);
                }
                self.remove_piece(from, moving_piece);
                self.put_piece(to, promo_piece);
            }
            MT_EN_PASSANT => {
                debug_assert!(moving_piece.piece_type() == PieceType::Pawn,
                    "make_move EP: not a pawn {:?}", moving_piece);
                debug_assert!(self.history[self.ply].ep_square != Square::NONE,
                    "make_move EP: no EP square was active");
                debug_assert!(to == self.history[self.ply].ep_square,
                    "make_move EP: to {} != saved ep {}", to.0, self.history[self.ply].ep_square.0);
                let cap_sq = Square((to.0 as i8 - pawn_push(us)) as u8);
                debug_assert!(self.board[cap_sq.index()] == Piece::new(PieceType::Pawn, them),
                    "make_move EP: no enemy pawn on cap_sq {}", cap_sq.0);
                let cap_pawn = Piece::new(PieceType::Pawn, them);
                self.history[self.ply].captured_piece = cap_pawn;
                self.remove_piece(cap_sq, cap_pawn);
                self.move_piece(from, to, moving_piece);
            }
            MT_CASTLING => {
                debug_assert!(moving_piece.piece_type() == PieceType::King,
                    "make_move castle: not a king {:?}", moving_piece);
                debug_assert!(from != to, "make_move castle: king and rook on the same square");
                let king_to = m.castle_king_to();
                let rook_from = to;
                let rook_to = m.castle_rook_to();
                debug_assert!(self.board[rook_from.index()] == Piece::new(PieceType::Rook, us),
                    "make_move castle: no friendly rook on {}", rook_from.0);
                let king = Piece::new(PieceType::King, us);
                let rook = Piece::new(PieceType::Rook, us);
                // Remove both first: in Chess960 the four squares can overlap.
                self.remove_piece(from, king);
                self.remove_piece(rook_from, rook);
                self.put_piece(king_to, king);
                self.put_piece(rook_to, rook);
            }
            _ => unreachable!(),
        }

        // Update castling rights
        self.castling_rights &=
            self.castling_rights_mask[from.index()] & self.castling_rights_mask[to.index()];
        self.key ^= ZOBRIST.castling[self.castling_rights as usize];

        // Switch side
        self.side_to_move = them;
        self.key ^= ZOBRIST.side;

        // Update fullmove number
        if us == Color::Black {
            self.fullmove_number += 1;
        }

        self.ply += 1;

        // Incremental repetition detection
        self.repetition = 0;
        let end = (self.halfmove_clock as usize).min(self.plies_from_null as usize);
        if end >= 4 {
            let mut steps = 4usize;
            while steps <= end && steps <= self.ply {
                let stp = &self.history[self.ply - steps];
                if stp.key == self.key {
                    // Negative = threefold (prior entry was itself a repeat)
                    self.repetition = if stp.repetition != 0 {
                        -(steps as i32)
                    } else {
                        steps as i32
                    };
                    break;
                }
                steps += 2;
            }
        }

        // Recalculate check info and threats
        self.set_check_info();
        self.update_threats();

        #[cfg(debug_assertions)]
        self.check_validity();
    }

    // ============================================================
    // unmake_move
    // ============================================================

    /// Undo a move: restore pieces and saved state (Zobrist, castling, EP, etc.).
    pub fn unmake_move(&mut self, m: Move) {
        debug_assert!(self.ply > 0, "unmake_move: ply underflow");
        debug_assert!(m.from_sq().0 < 64 && m.to_sq().0 < 64,
            "unmake_move: squares OOB from={} to={}", m.from_sq().0, m.to_sq().0);
        self.ply -= 1;
        let saved = self.history[self.ply];

        let us = !self.side_to_move; // was our move
        let from = m.from_sq();
        let to = m.to_sq();
        let mt = m.move_type();

        match mt {
            MT_NORMAL => {
                let moving_piece = self.board[to.index()];
                self.move_piece_nz(to, from, moving_piece);
                if saved.captured_piece != Piece::NONE {
                    self.put_piece_nz(to, saved.captured_piece);
                }
            }
            MT_PROMOTION => {
                let promo_piece = self.board[to.index()];
                let pawn = Piece::new(PieceType::Pawn, us);
                self.remove_piece_nz(to, promo_piece);
                self.put_piece_nz(from, pawn);
                if saved.captured_piece != Piece::NONE {
                    self.put_piece_nz(to, saved.captured_piece);
                }
            }
            MT_EN_PASSANT => {
                let moving_piece = self.board[to.index()];
                self.move_piece_nz(to, from, moving_piece);
                let cap_sq = Square((to.0 as i8 - pawn_push(us)) as u8);
                self.put_piece_nz(cap_sq, saved.captured_piece);
            }
            MT_CASTLING => {
                let king_to = m.castle_king_to();
                let rook_from = to;
                let rook_to = m.castle_rook_to();
                let king = Piece::new(PieceType::King, us);
                let rook = Piece::new(PieceType::Rook, us);
                self.remove_piece_nz(king_to, king);
                self.remove_piece_nz(rook_to, rook);
                self.put_piece_nz(from, king);
                self.put_piece_nz(rook_from, rook);
            }
            _ => unreachable!(),
        }

        // Restore state
        self.side_to_move = us;
        self.castling_rights = saved.castling_rights;
        self.ep_square = saved.ep_square;
        self.halfmove_clock = saved.halfmove_clock;
        self.key = saved.key;
        self.pawn_key = saved.pawn_key;
        self.non_pawn_key = saved.non_pawn_key;
        self.minor_key = saved.minor_key;
        self.checkers = saved.checkers;
        self.pinned = saved.pinned;
        self.threats = saved.threats;
        self.plies_from_null = saved.plies_from_null;
        self.repetition = saved.repetition;
        self.psq_mg = saved.psq_mg;
        self.psq_eg = saved.psq_eg;
        self.game_phase = saved.game_phase;

        if us == Color::Black {
            self.fullmove_number -= 1;
        }

        #[cfg(debug_assertions)]
        self.check_validity();
    }

    // ============================================================
    // Null move (for future search)
    // ============================================================

    /// Apply a [null move](https://www.chessprogramming.org/Null_Move): switch side
    /// without moving any piece. Used during search for null-move pruning.
    pub fn make_null_move(&mut self) {
        debug_assert!(self.ply < 1023, "make_null: ply {} overflow", self.ply);
        debug_assert!(self.checkers == 0, "make_null: in check (illegal)");
        self.history[self.ply] = StateHistory {
            castling_rights: self.castling_rights,
            ep_square: self.ep_square,
            halfmove_clock: self.halfmove_clock,
            key: self.key,
            pawn_key: self.pawn_key,
            non_pawn_key: self.non_pawn_key,
            minor_key: self.minor_key,
            checkers: self.checkers,
            pinned: self.pinned,
            threats: self.threats,
            captured_piece: Piece::NONE,
            plies_from_null: self.plies_from_null,
            repetition: self.repetition,
            psq_mg: self.psq_mg,
            psq_eg: self.psq_eg,
            game_phase: self.game_phase,
        };

        if self.ep_square != Square::NONE {
            self.key ^= ZOBRIST.ep[self.ep_square.file() as usize];
            self.ep_square = Square::NONE;
        }

        self.plies_from_null = 0;
        self.repetition = 0;
        self.side_to_move = !self.side_to_move;
        self.key ^= ZOBRIST.side;
        self.ply += 1;
        self.set_check_info();
        self.update_threats();

        #[cfg(debug_assertions)]
        self.check_validity();
    }

    pub fn unmake_null_move(&mut self) {
        debug_assert!(self.ply > 0, "unmake_null: ply underflow");
        self.ply -= 1;
        let saved = self.history[self.ply];
        self.side_to_move = !self.side_to_move;
        self.ep_square = saved.ep_square;
        self.key = saved.key;
        self.pawn_key = saved.pawn_key;
        self.non_pawn_key = saved.non_pawn_key;
        self.minor_key = saved.minor_key;
        self.checkers = saved.checkers;
        self.pinned = saved.pinned;
        self.threats = saved.threats;
        self.plies_from_null = saved.plies_from_null;
        self.repetition = saved.repetition;
        // PeSTO scores unchanged by null move — no restore needed
        // (psq_mg/psq_eg/game_phase are identical before and after null move)

        #[cfg(debug_assertions)]
        self.check_validity();
    }

    // ============================================================
    // Utility: is square attacked by color?
    // ============================================================

    #[allow(dead_code)]
    pub fn is_attacked_by(&self, sq: Square, c: Color) -> bool {
        attackers_to_color(sq, c, self.occupied(), &self.pieces) != 0
    }

    pub fn in_check(&self) -> bool {
        self.checkers != 0
    }

    /// Does this move check the opponent's king?
    ///
    /// Exact: the piece that lands is asked what it attacks from where it lands, and
    /// the squares the move empties are asked whether they uncover a slider of ours.
    /// That covers what the move type changes about either question — a promotion
    /// arrives as its new piece, en passant empties the captured pawn's square as well
    /// as the mover's, and castling checks with the rook it brings to d1/f1, never with
    /// the king. Called before the move is made, so everything reads from `occ_after`,
    /// the occupancy the move will produce.
    pub fn gives_check(&self, m: Move) -> bool {
        let from = m.from_sq();
        let to = m.to_sq();
        let us = self.side_to_move;
        let their_king = self.king_sq(!us);
        let mt = m.move_type();
        let occ = self.occupancies[2];

        // In Chess960 any two of a castle's four squares may coincide, so both origins
        // clear before both destinations set — the reverse order would lose a piece.
        let occ_after = match mt {
            MT_CASTLING => {
                (occ ^ from.bb() ^ to.bb()) | m.castle_king_to().bb() | m.castle_rook_to().bb()
            }
            MT_EN_PASSANT => {
                let capsq = Square::new(to.file(), from.rank());
                (occ ^ from.bb() ^ capsq.bb()) | to.bb()
            }
            _ => (occ ^ from.bb()) | to.bb(),
        };

        // Direct check, by whatever piece ends up on whatever square.
        let (lands_sq, lands_pt) = match mt {
            MT_CASTLING => (m.castle_rook_to(), PieceType::Rook),
            MT_PROMOTION => (to, m.promo_type()),
            _ => (to, self.board[from.index()].piece_type()),
        };
        let attacks = match lands_pt {
            PieceType::Pawn => pawn_attacks(lands_sq, us),
            PieceType::Knight => knight_attacks(lands_sq),
            PieceType::Bishop => bishop_attacks(lands_sq, occ_after),
            PieceType::Rook => rook_attacks(lands_sq, occ_after),
            PieceType::Queen => queen_attacks(lands_sq, occ_after),
            PieceType::King => 0,
        };
        if attacks & their_king.bb() != 0 {
            return true;
        }

        // Discovered check. Only a square the move leaves empty can open a line, and
        // only one sharing a rank, file, or diagonal with their king — a single masked
        // test, so the two magic probes below are paid for by the rare move that could
        // actually deliver one.
        let vacated = occ & !occ_after;
        if vacated & aligned_bb(their_king) == 0 {
            return false;
        }
        // Our sliders, minus the ones this move displaces: what those attack from where
        // they land is the direct check already answered above.
        let moved = if mt == MT_CASTLING { from.bb() | to.bb() } else { from.bb() };
        let queens = self.piece_type_bb(PieceType::Queen, us);
        let rooks_queens = (self.piece_type_bb(PieceType::Rook, us) | queens) & !moved;
        let bishops_queens = (self.piece_type_bb(PieceType::Bishop, us) | queens) & !moved;
        rook_attacks(their_king, occ_after) & rooks_queens != 0
            || bishop_attacks(their_king, occ_after) & bishops_queens != 0
    }

    // ============================================================
    // Draw detection
    // ============================================================

    /// Returns true if the position is a draw by the 50-move rule or repetition.
    ///
    /// `search_ply` is the distance from the root (0 at root). For repetition,
    /// a twofold within the search tree suffices. For game history (before root),
    /// only threefold triggers. See `draw_by_repetition` for details.
    pub fn is_draw(&self, search_ply: i32) -> bool {
        self.draw_by_fifty_move_rule() || self.draw_by_repetition(search_ply)
    }

    /// 50-move rule: draw if halfmove_clock >= 100, unless it's checkmate.
    #[inline(always)]
    pub fn draw_by_fifty_move_rule(&self) -> bool {
        self.halfmove_clock >= 100 && (self.checkers == 0 || self.has_legal_moves())
    }

    /// Incremental repetition check.
    ///
    /// - `repetition > 0`: twofold. Value = distance (half-moves) to first repeat.
    ///   If `repetition < search_ply`, the repeat is within the search tree → draw.
    /// - `repetition < 0`: threefold. Always `< search_ply` for positive ply → draw.
    /// - At root (`search_ply = 0`): only threefold triggers (negative < 0).
    #[inline(always)]
    pub fn draw_by_repetition(&self, search_ply: i32) -> bool {
        self.repetition != 0 && self.repetition < search_ply
    }

    /// Returns true if the position has at least one legal move.
    pub fn has_legal_moves(&self) -> bool {
        let mut buf = ArrayBuf::<Move, MAX_MOVES>::new();
        movegen::generate_legal_moves(self, &mut buf) > 0
    }

    /// Detects if a legal reversible move exists that would create a repeated
    /// position (van Kervinck cuckoo algorithm). Called from search when
    /// `alpha < 0` to raise alpha to draw score before the draw materializes.
    pub fn upcoming_repetition(&self, search_ply: usize) -> bool {
        // Build a view of historical keys for the cuckoo algorithm.
        // history[i].key = key of position at ply i (saved before move at ply i).
        // We need keys from ply-1 back to ply-end.
        let hmc = self.halfmove_clock as usize;
        let pfn = self.plies_from_null as usize;
        let end = hmc.min(pfn);
        if end < 3 || end > self.ply {
            return false;
        }

        // Pass a slice of historical keys. The cuckoo function expects
        // s(v) = key at v plies ago, accessible via history_keys[len-v].
        // history[self.ply - v].key is the key v plies ago.
        // We create a slice from history[self.ply - end .. self.ply].
        let start = self.ply - end;
        // Safety: we need the keys as a contiguous slice. StateHistory has key
        // at a known offset. Build a temporary array of just the keys.
        let mut keys = [0u64; 128]; // MAX_PLY should suffice
        let key_count = end.min(128);
        let mut i = 0;
        while i < key_count {
            keys[i] = self.history[start + i].key;
            i += 1;
        }

        crate::cuckoo::upcoming_repetition(
            &keys[..key_count],
            self.key,
            self.occupied(),
            hmc,
            pfn,
            self.repetition,
            search_ply,
        )
    }

    // ============================================================
    // Debug validation (zero-cost in release)
    // ============================================================

    /// Validate that the position represents a legal chess position.
    ///
    /// Runs in **both debug and release** builds, unlike `check_validity()`.
    /// Catches illegal FEN input that would crash the engine (missing kings,
    /// opponent in check, impossible piece counts, incoherent castling/EP).
    fn validate_legality(&self) -> Result<(), FenParseError> {
        // 1. Exactly one king per side (critical: prevents lsb(0) = Square(64) crash)
        for &c in &[Color::White, Color::Black] {
            let king_bb = self.piece_type_bb(PieceType::King, c);
            let count = popcount(king_bb);
            if count != 1 {
                let color = if c == Color::White { "white" } else { "black" };
                return Err(FenParseError::InvalidKingCount { color, count });
            }
        }

        // 2. No pawns on rank 1 or rank 8
        let all_pawns = self.piece_type_bb(PieceType::Pawn, Color::White)
            | self.piece_type_bb(PieceType::Pawn, Color::Black);
        if all_pawns & (RANK_1 | RANK_8) != 0 {
            return Err(FenParseError::PawnsOnBackRank);
        }

        // 3. Max 8 pawns per side
        for &c in &[Color::White, Color::Black] {
            let count = popcount(self.piece_type_bb(PieceType::Pawn, c));
            if count > 8 {
                let color = if c == Color::White { "white" } else { "black" };
                return Err(FenParseError::TooManyPawns { color, count });
            }
        }

        // 4. Max 16 pieces per side
        for &c in &[Color::White, Color::Black] {
            let count = popcount(self.color_bb(c));
            if count > 16 {
                let color = if c == Color::White { "white" } else { "black" };
                return Err(FenParseError::TooManyPieces { color, count });
            }
        }

        // 5. Extra pieces beyond starting material cannot exceed missing pawns
        for &c in &[Color::White, Color::Black] {
            let pawns = popcount(self.piece_type_bb(PieceType::Pawn, c));
            let queens = popcount(self.piece_type_bb(PieceType::Queen, c));
            let rooks = popcount(self.piece_type_bb(PieceType::Rook, c));
            let bishops = popcount(self.piece_type_bb(PieceType::Bishop, c));
            let knights = popcount(self.piece_type_bb(PieceType::Knight, c));
            let extra = queens.saturating_sub(1)
                + rooks.saturating_sub(2)
                + bishops.saturating_sub(2)
                + knights.saturating_sub(2);
            let missing_pawns = 8 - pawns;
            if extra > missing_pawns {
                let color = if c == Color::White { "white" } else { "black" };
                return Err(FenParseError::TooManyPromotions { color, extra, missing_pawns });
            }
        }

        // 6. Opponent king must NOT be in check (illegal position)
        let us = self.side_to_move;
        let them = !us;
        let their_king = self.king_sq(them);
        let attackers = attackers_to_color(their_king, us, self.occupied(), &self.pieces);
        if attackers != 0 {
            return Err(FenParseError::OpponentKingInCheck);
        }

        // 7. Castling rights coherence (king on back rank, rook on its recorded square)
        self.check_castling_coherence(WHITE_OO, Color::White, Piece::WHITE_KING, Piece::WHITE_ROOK)?;
        self.check_castling_coherence(WHITE_OOO, Color::White, Piece::WHITE_KING, Piece::WHITE_ROOK)?;
        self.check_castling_coherence(BLACK_OO, Color::Black, Piece::BLACK_KING, Piece::BLACK_ROOK)?;
        self.check_castling_coherence(BLACK_OOO, Color::Black, Piece::BLACK_KING, Piece::BLACK_ROOK)?;

        // 8. En passant square requires an enemy pawn behind it
        if self.ep_square != Square::NONE {
            let cap_sq = Square((self.ep_square.0 as i8 - pawn_push(us)) as u8);
            if self.board[cap_sq.index()] != Piece::new(PieceType::Pawn, them) {
                return Err(FenParseError::EpSquareNoPawn {
                    ep: self.ep_square.to_string(),
                });
            }
        }

        Ok(())
    }

    fn check_castling_coherence(
        &self, right: u8, us: Color, king: Piece, rook: Piece,
    ) -> Result<(), FenParseError> {
        if self.castling_rights & right == 0 {
            return Ok(());
        }
        let ksq = self.king_sq(us);
        let rsq = self.castle_rook_sq(right);
        if ksq.rank() != Self::back_rank(us) || self.board[ksq.index()] != king {
            return Err(FenParseError::CastlingRightsIncoherent {
                detail: "castling king is not on its back rank",
            });
        }
        if rsq.0 >= 64 || self.board[rsq.index()] != rook || rsq.rank() != ksq.rank() {
            return Err(FenParseError::CastlingRightsIncoherent {
                detail: "castling rook is not on its recorded square",
            });
        }
        Ok(())
    }

    // ============================================================

    /// Comprehensive position validation. Checks ~14 invariants including
    /// bitboard-mailbox coherence, Zobrist hash integrity, king validity,
    /// castling rights, and pawn placement. Targets every historical bug
    /// (SE stack corruption, check_mask bypass, TT king capture, etc.).
    ///
    /// Called after every make/unmake/from_fen in debug builds.
    #[cfg(debug_assertions)]
    pub fn check_validity(&self) {
        // 1. Bitboard-mailbox bidirectional coherence
        for sq_idx in 0..64 {
            let sq = Square::from_index(sq_idx as u8);
            let pc = self.board[sq_idx];
            if pc != Piece::NONE {
                debug_assert!(pc.0 < 12,
                    "check_validity: board[{}] = Piece({}), invalid", sq_idx, pc.0);
                debug_assert!(self.pieces[pc.index()] & sq.bb() != 0,
                    "check_validity: board[{}] = {:?} but not in pieces bitboard", sq_idx, pc);
            } else {
                // No piece bitboard should contain this square
                for pi in 0..12 {
                    debug_assert!(self.pieces[pi] & sq.bb() == 0,
                        "check_validity: board[{}] = NONE but pieces[{}] has it set",
                        sq_idx, pi);
                }
            }
        }

        // Also: every bit set in a piece bitboard must have the matching mailbox entry
        for pi in 0..12 {
            let mut bb = self.pieces[pi];
            while bb != 0 {
                let sq = pop_lsb(&mut bb);
                debug_assert!(self.board[sq.index()] == Piece(pi as u8),
                    "check_validity: pieces[{}] has sq {} but board says {:?}",
                    pi, sq.0, self.board[sq.index()]);
            }
        }

        // 2. Piece bitboard no overlap
        for i in 0..12 {
            for j in (i + 1)..12 {
                debug_assert!(self.pieces[i] & self.pieces[j] == 0,
                    "check_validity: pieces[{}] & pieces[{}] overlap: 0x{:016x}",
                    i, j, self.pieces[i] & self.pieces[j]);
            }
        }

        // 3. Occupancy coherence
        let mut white_occ = 0u64;
        let mut black_occ = 0u64;
        for pt_idx in 0..6 {
            white_occ |= self.pieces[pt_idx * 2];     // white pieces at even indices
            black_occ |= self.pieces[pt_idx * 2 + 1]; // black pieces at odd indices
        }
        debug_assert!(self.occupancies[0] == white_occ,
            "check_validity: occ[WHITE] 0x{:016x} != computed 0x{:016x}",
            self.occupancies[0], white_occ);
        debug_assert!(self.occupancies[1] == black_occ,
            "check_validity: occ[BLACK] 0x{:016x} != computed 0x{:016x}",
            self.occupancies[1], black_occ);
        debug_assert!(self.occupancies[2] == white_occ | black_occ,
            "check_validity: occ[BOTH] 0x{:016x} != WHITE|BLACK 0x{:016x}",
            self.occupancies[2], white_occ | black_occ);
        debug_assert!(self.occupancies[0] & self.occupancies[1] == 0,
            "check_validity: WHITE & BLACK overlap: 0x{:016x}",
            self.occupancies[0] & self.occupancies[1]);

        // 4. Kings: exactly one per side, valid square
        for c in [Color::White, Color::Black] {
            let king_bb = self.piece_type_bb(PieceType::King, c);
            debug_assert!(popcount(king_bb) == 1,
                "check_validity: {:?} has {} kings", c, popcount(king_bb));
            let ksq = self.king_sq(c);
            debug_assert!(ksq.0 < 64,
                "check_validity: {:?} king_sq = {} (BUG 3: Square(64))", c, ksq.0);
            debug_assert!(self.board[ksq.index()] == Piece::new(PieceType::King, c),
                "check_validity: {:?} king_sq {} but board says {:?}",
                c, ksq.0, self.board[ksq.index()]);
        }

        // 5. Opponent king NOT in check (position would be illegal)
        let us = self.side_to_move;
        let them = !us;
        let their_king = self.king_sq(them);
        let attackers = attackers_to_color(their_king, us, self.occupied(), &self.pieces);
        debug_assert!(attackers == 0,
            "check_validity: {:?} king on {} is in check by {:?} (illegal position, BUG 2)",
            them, their_king.0, us);

        // 6. Pawns: none on rank 1 or rank 8, max 8 per side
        for c in [Color::White, Color::Black] {
            let pawns = self.piece_type_bb(PieceType::Pawn, c);
            debug_assert!(pawns & (RANK_1 | RANK_8) == 0,
                "check_validity: {:?} pawn on rank 1/8: 0x{:016x}", c, pawns & (RANK_1 | RANK_8));
            debug_assert!(popcount(pawns) <= 8,
                "check_validity: {:?} has {} pawns", c, popcount(pawns));
        }

        // 7. Max 16 pieces per side
        for c in [Color::White, Color::Black] {
            let count = popcount(self.color_bb(c));
            debug_assert!(count <= 16,
                "check_validity: {:?} has {} pieces (max 16)", c, count);
        }

        // 8. EP square validity
        if self.ep_square != Square::NONE {
            let ep = self.ep_square;
            debug_assert!(ep.0 < 64,
                "check_validity: ep_square {} out of range", ep.0);
            let expected_rank = if us == Color::White { 5u8 } else { 2u8 };
            debug_assert!(ep.rank() == expected_rank,
                "check_validity: ep_square {} on rank {}, expected {} (stm={:?})",
                ep.0, ep.rank(), expected_rank, us);
            // Enemy pawn must be on the square behind EP
            let cap_sq = Square((ep.0 as i8 - pawn_push(us)) as u8);
            debug_assert!(self.board[cap_sq.index()] == Piece::new(PieceType::Pawn, them),
                "check_validity: no {:?} pawn behind EP {} (cap_sq={})",
                them, ep.0, cap_sq.0);
        }

        // 9. Castling rights coherence
        for &(right, us, king, rook) in &[
            (WHITE_OO, Color::White, Piece::WHITE_KING, Piece::WHITE_ROOK),
            (WHITE_OOO, Color::White, Piece::WHITE_KING, Piece::WHITE_ROOK),
            (BLACK_OO, Color::Black, Piece::BLACK_KING, Piece::BLACK_ROOK),
            (BLACK_OOO, Color::Black, Piece::BLACK_KING, Piece::BLACK_ROOK),
        ] {
            if self.castling_rights & right == 0 {
                continue;
            }
            let ksq = self.king_sq(us);
            let rsq = self.castle_rook_sq(right);
            debug_assert!(ksq.rank() == Self::back_rank(us) && self.board[ksq.index()] == king,
                "check_validity: right {right} but king not on back rank (ksq={})", ksq.0);
            debug_assert!(rsq.0 < 64 && self.board[rsq.index()] == rook && rsq.rank() == ksq.rank(),
                "check_validity: right {right} but rook missing on {}", rsq.0);
        }

        // 10. Zobrist hash integrity (catches any incremental drift)
        debug_assert!(self.key == self.compute_key(),
            "check_validity: Zobrist key 0x{:016x} != computed 0x{:016x}",
            self.key, self.compute_key());
        debug_assert!(self.pawn_key == self.compute_pawn_key(),
            "check_validity: pawn_key 0x{:016x} != computed 0x{:016x}",
            self.pawn_key, self.compute_pawn_key());
        debug_assert!(self.non_pawn_key == self.compute_non_pawn_key(),
            "check_validity: non_pawn_key [{:016x}, {:016x}] != computed [{:016x}, {:016x}]",
            self.non_pawn_key[0], self.non_pawn_key[1],
            self.compute_non_pawn_key()[0], self.compute_non_pawn_key()[1]);
        debug_assert!(self.minor_key == self.compute_minor_key(),
            "check_validity: minor_key {:016x} != computed {:016x}",
            self.minor_key, self.compute_minor_key());

        // 10b. Incremental PeSTO integrity
        {
            let (mg, eg, phase) = self.compute_psq();
            debug_assert!(self.psq_mg == mg,
                "check_validity: psq_mg {} != computed {}", self.psq_mg, mg);
            debug_assert!(self.psq_eg == eg,
                "check_validity: psq_eg {} != computed {}", self.psq_eg, eg);
            debug_assert!(self.game_phase == phase,
                "check_validity: game_phase {} != computed {}", self.game_phase, phase);
        }

        // 11. Halfmove clock
        debug_assert!(self.halfmove_clock <= 100,
            "check_validity: halfmove_clock {} > 100", self.halfmove_clock);

        // 12. Ply bound (catches BUG 4: history overflow)
        debug_assert!(self.ply < 1024,
            "check_validity: ply {} >= 1024 (history overflow)", self.ply);

        // 13. Checkers coherence: recompute and compare
        {
            let ksq = self.king_sq(us);
            let occ = self.occupied();
            let mut recomputed_checkers = 0u64;

            // Knight checks
            recomputed_checkers |= knight_attacks(ksq)
                & self.piece_type_bb(PieceType::Knight, them);
            // Pawn checks
            recomputed_checkers |= pawn_attacks(ksq, us)
                & self.piece_type_bb(PieceType::Pawn, them);
            // Bishop/Queen diagonal checks
            recomputed_checkers |= bishop_attacks(ksq, occ)
                & (self.piece_type_bb(PieceType::Bishop, them)
                    | self.piece_type_bb(PieceType::Queen, them));
            // Rook/Queen orthogonal checks
            recomputed_checkers |= rook_attacks(ksq, occ)
                & (self.piece_type_bb(PieceType::Rook, them)
                    | self.piece_type_bb(PieceType::Queen, them));

            debug_assert!(self.checkers == recomputed_checkers,
                "check_validity: checkers 0x{:016x} != recomputed 0x{:016x}",
                self.checkers, recomputed_checkers);

            // Max 2 checkers
            debug_assert!(popcount(self.checkers) <= 2,
                "check_validity: {} checkers (max 2)", popcount(self.checkers));
        }

        // 14. Pinned coherence: pinned pieces must be our pieces
        debug_assert!(self.pinned & !self.color_bb(us) == 0,
            "check_validity: pinned contains non-{:?} pieces: 0x{:016x}",
            us, self.pinned & !self.color_bb(us));
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    #[test]
    fn test_fen_startpos() {
        let pos = Position::from_fen(STARTPOS).unwrap();
        assert_eq!(pos.side_to_move, Color::White);
        assert_eq!(pos.castling_rights, ALL_CASTLING);
        assert_eq!(pos.ep_square, Square::NONE);
        assert_eq!(pos.halfmove_clock, 0);
        assert_eq!(pos.fullmove_number, 1);
        assert_eq!(pos.castle_rook_sq(WHITE_OO), Square::H1);
        assert_eq!(pos.castle_rook_sq(WHITE_OOO), Square::A1);
        assert_eq!(pos.castle_rook_sq(BLACK_OO), Square::H8);
        assert_eq!(pos.castle_rook_sq(BLACK_OOO), Square::A8);
        assert!(!pos.chess960);

        // Check piece counts
        assert_eq!(popcount(pos.piece_type_bb(PieceType::Pawn, Color::White)), 8);
        assert_eq!(popcount(pos.piece_type_bb(PieceType::Pawn, Color::Black)), 8);
        assert_eq!(popcount(pos.piece_type_bb(PieceType::Rook, Color::White)), 2);
        assert_eq!(popcount(pos.piece_type_bb(PieceType::Knight, Color::White)), 2);
        assert_eq!(popcount(pos.piece_type_bb(PieceType::King, Color::White)), 1);
        assert_eq!(popcount(pos.piece_type_bb(PieceType::King, Color::Black)), 1);

        // King squares
        assert_eq!(pos.king_sq(Color::White), Square::E1);
        assert_eq!(pos.king_sq(Color::Black), Square::E8);

        // Mailbox spot check
        assert_eq!(pos.board[Square::E1.index()], Piece::WHITE_KING);
        assert_eq!(pos.board[Square::D8.index()], Piece::BLACK_QUEEN);
        assert_eq!(pos.board[Square::E4.index()], Piece::NONE);
    }

    #[test]
    fn test_fen_shredder_startpos() {
        let kq = Position::from_fen(STARTPOS).unwrap();
        let ah = Position::from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w AHah - 0 1",
        ).unwrap();
        assert_eq!(kq.castling_rights, ah.castling_rights);
        assert_eq!(kq.castle_rook_sq(WHITE_OO), ah.castle_rook_sq(WHITE_OO));
        assert_eq!(kq.castle_rook_sq(WHITE_OOO), ah.castle_rook_sq(WHITE_OOO));
        assert_eq!(kq.key, ah.key);
    }

    #[test]
    fn test_fen_chess960_shredder_roundtrip() {
        let fen = "bbqnnrkr/pppppppp/8/8/8/8/PPPPPPPP/BBQNNRKR w HFhf - 0 1";
        let pos = Position::from_fen_ex(fen, true).unwrap();
        assert!(pos.chess960);
        assert_eq!(pos.castle_rook_sq(WHITE_OO), Square::H1);
        assert_eq!(pos.castle_rook_sq(WHITE_OOO), Square::F1);
        assert_eq!(pos.to_fen(), fen);
    }

    #[test]
    fn test_fen_chess324() {
        let pos = Position::from_fen(
            "rnqbknbr/pppppppp/8/8/8/8/PPPPPPPP/RNBBKNQR w KQkq - 0 1",
        ).unwrap();
        assert_eq!(pos.king_sq(Color::White), Square::E1);
        assert_eq!(pos.castle_rook_sq(WHITE_OO), Square::H1);
        assert_eq!(pos.castle_rook_sq(WHITE_OOO), Square::A1);
        assert_eq!(pos.board[Square::C1.index()].piece_type(), PieceType::Bishop);
        assert_eq!(pos.board[Square::D1.index()].piece_type(), PieceType::Bishop);
    }

    #[test]
    fn test_fen_dfrc_asymmetric() {
        let pos = Position::from_fen(
            "nrkbnqbr/pppppppp/8/8/8/8/PPPPPPPP/BBNRKQNR w HDhb - 0 1",
        ).unwrap();
        assert_eq!(pos.castle_rook_sq(WHITE_OO), Square::H1);
        assert_eq!(pos.castle_rook_sq(WHITE_OOO), Square::D1);
        assert_eq!(pos.castle_rook_sq(BLACK_OO), Square::H8);
        assert_eq!(pos.castle_rook_sq(BLACK_OOO), Square::B8);
        assert_ne!(pos.king_sq(Color::White).file(), pos.king_sq(Color::Black).file());
    }

    /// `gives_check` is consulted before the move is made, and pruning trusts it, so
    /// it has to answer what the position itself would answer afterwards. Walking whole
    /// trees is the only way to cover the combinations that make it hard: a discovery
    /// and a direct check at once, an en passant that empties two squares on the same
    /// rank, a promotion checking as its new piece, a castle checking with its rook.
    #[test]
    fn gives_check_answers_what_the_move_actually_produces() {
        fn walk(pos: &mut Position, depth: u32) {
            let mut buf: ArrayBuf<Move, MAX_MOVES> = ArrayBuf::new();
            let n = movegen::generate_legal_moves(pos, &mut buf);
            for i in 0..n {
                let m = buf[i];
                let predicted = pos.gives_check(m);
                pos.make_move(m);
                let actual = pos.in_check();
                if depth > 1 {
                    walk(pos, depth - 1);
                }
                pos.unmake_move(m);
                assert_eq!(
                    predicted,
                    actual,
                    "gives_check said {predicted} for {} in {}",
                    m.to_uci_960(true),
                    pos.to_fen(),
                );
            }
        }

        let suite = [
            (STARTPOS, 4),
            // Kiwipete: castling both sides, pins, a board full of sliders.
            ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3),
            // The classic en passant position: captures that empty two squares on the
            // rank a rook already eyes, which is where discovered checks hide.
            ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 4),
            // Promotions on both wings, checking as knight, bishop, rook or queen.
            ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 3),
            // Chess960, where the castling rook can land anywhere on the back rank.
            ("rqnkbrnb/1pppp1pp/8/5p2/p1P5/3P1N2/PP2PPPP/RQNKBR1B w KQkq - 0 4", 3),
            ("1r1qnbkr/ppp1pppp/1n1p4/5b2/2PP4/3N4/PP2PPPP/NRBQ1BKR w KQkq - 3 4", 3),
            // A castle that checks, and a Chess960 one whose king does not move.
            ("5k2/8/8/8/8/8/8/4K2R w K - 0 1", 4),
            ("3k4/8/8/8/8/8/8/1RK5 w Q - 0 1", 4),
        ];
        for (fen, depth) in suite {
            let mut pos = Position::from_fen(fen).unwrap();
            walk(&mut pos, depth);
        }
    }

    /// The rook a castling move brings to d1/f1 can check, and it is the only piece
    /// in the move that can — which is why answering `false` for every king move
    /// let a checking castle through the pruning guards that consult this.
    #[test]
    fn castling_checks_with_the_rook_it_brings_over() {
        // O-O drops the rook on f1, facing the black king down an open f-file.
        let pos = Position::from_fen("5k2/8/8/8/8/8/8/4K2R w K - 0 1").unwrap();
        assert!(pos.gives_check(Move::new_with_type(Square::E1, Square::H1, MT_CASTLING)));

        // Same rook, same file, black king one square over: nothing.
        let pos = Position::from_fen("4k3/8/8/8/8/8/8/4K2R w K - 0 1").unwrap();
        assert!(!pos.gives_check(Move::new_with_type(Square::E1, Square::H1, MT_CASTLING)));

        // Chess960 O-O-O where the king never leaves c1: the rook b1 → d1 checks alone.
        let pos = Position::from_fen("3k4/8/8/8/8/8/8/1RK5 w Q - 0 1").unwrap();
        let ooo = Move::new_with_type(Square::C1, Square::B1, MT_CASTLING);
        assert_eq!(ooo.castle_king_to(), Square::C1);
        assert_eq!(ooo.castle_rook_to(), Square::D1);
        assert!(pos.gives_check(ooo));
    }

    #[test]
    fn test_fen_castling_file_overlaps_king() {
        assert!(matches!(
            Position::from_fen("4k3/8/8/8/8/8/8/4K3 w E - 0 1"),
            Err(FenParseError::CastlingRightsIncoherent { .. })
        ));
    }

    #[test]
    fn test_fen_roundtrip() {

        let fens = [
            STARTPOS,
            // An en passant square a pawn can take on survives the round trip; one
            // nobody can take reads back as "-" (see the en passant test below).
            "rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 3",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        ];
        for fen in &fens {
            let pos = Position::from_fen(fen).unwrap();
            assert_eq!(pos.to_fen(), *fen, "FEN roundtrip failed for: {}", fen);
        }
    }

    #[test]
    fn test_zobrist_consistency() {
        let pos = Position::from_fen(STARTPOS).unwrap();
        assert_eq!(pos.key, pos.compute_key());
        assert_eq!(pos.pawn_key, pos.compute_pawn_key());
    }

    #[test]
    fn test_make_unmake_preserves_position() {
        let mut pos = Position::from_fen(STARTPOS).unwrap();
        let original_key = pos.key;
        let original_fen = pos.to_fen();

        // e2e4
        let m = Move::new(Square::E2, Square::E4);
        pos.make_move(m);
        assert_ne!(pos.key, original_key);
        assert_eq!(pos.side_to_move, Color::Black);
        assert_eq!(pos.key, pos.compute_key(), "Key inconsistency after make_move");

        pos.unmake_move(m);
        assert_eq!(pos.key, original_key);
        assert_eq!(pos.to_fen(), original_fen);
    }

    #[test]
    fn test_make_unmake_capture() {
        // Position with possible capture
        let fen = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2";
        let mut pos = Position::from_fen(fen).unwrap();
        let original_fen = pos.to_fen();
        let original_key = pos.key;

        // e4xd5 capture
        let m = Move::new(Square::E4, Square::D5);
        pos.make_move(m);
        assert_eq!(pos.board[Square::D5.index()], Piece::WHITE_PAWN);
        assert_eq!(pos.key, pos.compute_key());

        pos.unmake_move(m);
        assert_eq!(pos.to_fen(), original_fen);
        assert_eq!(pos.key, original_key);
    }

    #[test]
    fn test_startpos_not_in_check() {
        let pos = Position::from_fen(STARTPOS).unwrap();
        assert!(!pos.in_check());
        assert_eq!(pos.checkers, 0);
    }

    #[test]
    fn test_fen_wrong_field_count() {
        assert!(matches!(
            Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w"),
            Err(FenParseError::WrongFieldCount { found: 2 })
        ));
    }

    #[test]
    fn test_fen_wrong_rank_count() {
        assert!(matches!(
            Position::from_fen("rnbqkbnr/pppppppp/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
            Err(FenParseError::WrongRankCount { found: 7 })
        ));
    }

    #[test]
    fn test_fen_invalid_piece_char() {
        assert!(matches!(
            Position::from_fen("xnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
            Err(FenParseError::InvalidPieceChar { ch: 'x' })
        ));
    }

    #[test]
    fn test_fen_rank_length_mismatch() {
        assert!(matches!(
            Position::from_fen("rnbqkbn/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
            Err(FenParseError::RankLengthMismatch { rank: 8, count: 7 })
        ));
    }

    #[test]
    fn test_fen_invalid_side_to_move() {
        assert!(matches!(
            Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR x KQkq - 0 1"),
            Err(FenParseError::InvalidSideToMove { .. })
        ));
    }

    #[test]
    fn test_fen_invalid_castling_char() {
        assert!(matches!(
            Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQxq - 0 1"),
            Err(FenParseError::InvalidCastlingChar { ch: 'x' })
        ));
    }

    #[test]
    fn test_fen_invalid_ep_square() {
        assert!(matches!(
            Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq z9 0 1"),
            Err(FenParseError::InvalidEpSquare { .. })
        ));
    }

    #[test]
    fn test_fen_invalid_halfmove() {
        assert!(matches!(
            Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - abc 1"),
            Err(FenParseError::InvalidHalfmoveClock { .. })
        ));
    }

    #[test]
    fn test_fen_invalid_fullmove() {
        assert!(matches!(
            Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 xyz"),
            Err(FenParseError::InvalidFullmoveNumber { .. })
        ));
    }

    // ============================================================
    // Repetition detection tests
    // ============================================================

    /// Helper: apply a UCI move string to a position.
    fn make_uci(pos: &mut Position, uci: &str) {
        let mut buf = ArrayBuf::<Move, MAX_MOVES>::new();
        let count = crate::movegen::generate_legal_moves(pos, &mut buf);
        let m = (0..count)
            .map(|i| buf[i])
            .find(|m| m.to_uci() == uci)
            .unwrap_or_else(|| panic!("illegal move: {uci}"));
        pos.make_move(m);
    }

    /// The cuckoo test asks whether the side to move can undo one of its own reversible
    /// moves and land on a position seen before. Two kings shuffling on the e-file
    /// offer that from the third ply on (no castling rights: losing them would
    /// make the earlier positions unreachable). The king that would move stands on
    /// the lower of its two squares in some of those positions and on the higher one
    /// in the others; the path check between the squares must not take the king
    /// itself for an obstacle.
    #[test]
    fn an_upcoming_repetition_is_seen_whichever_end_the_piece_stands_on() {
        let mut pos = Position::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let line = ["e1e2", "e8e7", "e2e1", "e7e8", "e1e2", "e8e7"];
        let expected = [false, false, true, true, true, true];
        for (i, (uci, want)) in line.iter().zip(expected).enumerate() {
            make_uci(&mut pos, uci);
            assert_eq!(
                pos.upcoming_repetition(10),
                want,
                "after ply {} ({uci}): upcoming repetition should be {want}",
                i + 1
            );
        }
    }

    /// make_move records an en passant square only where a pawn can take on it, and
    /// hashes it on the same condition. A FEN naming one nobody can take must not give
    /// the position a different key from the one the moves would have built.
    #[test]
    fn a_fen_en_passant_square_nobody_can_take_is_dropped() {
        let spurious = Position::from_fen(
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
        )
        .unwrap();
        let plain = Position::from_fen(
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        )
        .unwrap();
        assert_eq!(spurious.ep_square, Square::NONE);
        assert_eq!(spurious.key, plain.key);
        assert_eq!(spurious.to_fen(), plain.to_fen(), "the FEN reads back normalised");
        let mut played = Position::from_fen(STARTPOS).unwrap();
        make_uci(&mut played, "e2e4");
        make_uci(&mut played, "c7c5");
        assert_eq!(played.key, plain.key, "the FEN and the moves must agree on the key");
        // Where the capture exists, the square stays.
        let live = Position::from_fen(
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
        )
        .unwrap();
        assert_eq!(live.ep_square, Square::F6);
    }

    #[test]
    fn test_twofold_repetition() {
        // 1.Nf3 Nf6 2.Ng1 Ng8 → back to startpos (twofold)
        let mut pos = Position::from_fen(STARTPOS).unwrap();
        let start_key = pos.key;
        make_uci(&mut pos, "g1f3"); // 1.Nf3
        make_uci(&mut pos, "g8f6"); // 1...Nf6
        make_uci(&mut pos, "f3g1"); // 2.Ng1
        make_uci(&mut pos, "f6g8"); // 2...Ng8
        assert_eq!(pos.key, start_key, "key should match startpos");
        assert!(pos.repetition > 0, "twofold: repetition should be positive");
        // Within search tree (ply=5 > distance=4)
        assert!(pos.is_draw(5), "twofold: is_draw(5) should be true");
        // Before root (ply=3 < distance=4)
        assert!(!pos.is_draw(3), "twofold: is_draw(3) should be false (game history)");
    }

    #[test]
    fn test_threefold_repetition() {
        // 1.Nf3 Nf6 2.Ng1 Ng8 3.Nf3 Nf6 4.Ng1 Ng8 → startpos for 3rd time
        let mut pos = Position::from_fen(STARTPOS).unwrap();
        for _ in 0..2 {
            make_uci(&mut pos, "g1f3");
            make_uci(&mut pos, "g8f6");
            make_uci(&mut pos, "f3g1");
            make_uci(&mut pos, "f6g8");
        }
        assert!(pos.repetition < 0, "threefold: repetition should be negative");
        // Threefold is always detected (negative < any positive ply)
        assert!(pos.is_draw(1), "threefold: is_draw(1) should be true");
    }

    #[test]
    fn test_no_repetition_across_capture() {
        // After a capture, halfmove_clock resets → can't see past it
        let mut pos = Position::from_fen(STARTPOS).unwrap();
        make_uci(&mut pos, "e2e4");
        make_uci(&mut pos, "d7d5");
        make_uci(&mut pos, "e4d5"); // capture: hmc resets to 0
        assert_eq!(pos.halfmove_clock, 0);
        assert_eq!(pos.repetition, 0);
    }

    #[test]
    fn test_plies_from_null_tracking() {
        let mut pos = Position::from_fen(STARTPOS).unwrap();
        assert_eq!(pos.plies_from_null, 0);
        make_uci(&mut pos, "e2e4");
        assert_eq!(pos.plies_from_null, 1);
        make_uci(&mut pos, "e7e5");
        assert_eq!(pos.plies_from_null, 2);
        // Null move resets plies_from_null
        pos.make_null_move();
        assert_eq!(pos.plies_from_null, 0);
        // Unmake restores it
        pos.unmake_null_move();
        assert_eq!(pos.plies_from_null, 2);
    }

    #[test]
    fn test_no_repetition_across_null_move() {
        // Play a sequence that would repeat, but insert a null move in between.
        // The plies_from_null window should prevent detection across the null move.
        let mut pos = Position::from_fen(STARTPOS).unwrap();
        make_uci(&mut pos, "g1f3"); // hmc=1, pfn=1
        make_uci(&mut pos, "g8f6"); // hmc=2, pfn=2
        // Null move resets pfn
        pos.make_null_move(); // pfn=0
        pos.unmake_null_move(); // pfn=2 restored
        make_uci(&mut pos, "f3g1"); // hmc=3, pfn=3
        make_uci(&mut pos, "f6g8"); // hmc=4, pfn=4
        // Position matches startpos, but pfn=4 and hmc=4 → both >= 4 → should detect
        assert!(pos.repetition > 0, "should detect repetition (pfn >= 4, hmc >= 4)");
    }

    #[test]
    fn test_50_move_rule() {
        // Position with halfmove_clock = 100 and not in check → draw
        let pos =
            Position::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 100 1").unwrap();
        assert_eq!(pos.halfmove_clock, 100);
        assert!(pos.is_draw(0), "hmc=100, not in check → draw");
    }

    #[test]
    fn test_repetition_unmake_restore() {
        let mut pos = Position::from_fen(STARTPOS).unwrap();
        make_uci(&mut pos, "g1f3");
        make_uci(&mut pos, "g8f6");
        make_uci(&mut pos, "f3g1");
        let _saved_rep = pos.repetition;
        let _saved_pfn = pos.plies_from_null;
        make_uci(&mut pos, "f6g8"); // creates twofold
        assert!(pos.repetition > 0);
        // Unmake and verify restoration
        let mut buf = ArrayBuf::<Move, MAX_MOVES>::new();
        let _count = crate::movegen::generate_legal_moves(&pos, &mut buf);
        // We need to undo the last move. Since we don't have the move stored,
        // let's use a different approach: make one more move, then verify
        // that after the make, unmake restores correctly.
        let mut pos2 = Position::from_fen(STARTPOS).unwrap();
        make_uci(&mut pos2, "g1f3");
        make_uci(&mut pos2, "g8f6");
        let rep_before = pos2.repetition;
        let pfn_before = pos2.plies_from_null;
        let key_before = pos2.key;
        // Generate a legal move to make/unmake
        let mut buf2 = ArrayBuf::<Move, MAX_MOVES>::new();
        let _ = crate::movegen::generate_legal_moves(&pos2, &mut buf2);
        let m = buf2[0];
        pos2.make_move(m);
        pos2.unmake_move(m);
        assert_eq!(pos2.repetition, rep_before, "repetition not restored after unmake");
        assert_eq!(pos2.plies_from_null, pfn_before, "plies_from_null not restored");
        assert_eq!(pos2.key, key_before, "key not restored after unmake");
    }

    // ============================================================
    // FEN validation tests — pre-filter
    // ============================================================

    #[test]
    fn test_fen_too_long() {
        let long = format!("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1{}", " ".repeat(80));
        assert!(matches!(Position::from_fen(&long), Err(FenParseError::TooLong { .. })));
    }

    #[test]
    fn test_fen_invalid_chars() {
        assert!(matches!(
            Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\x00"),
            Err(FenParseError::InvalidChars)
        ));
    }

    // ============================================================
    // FEN validation tests — syntactic
    // ============================================================

    #[test]
    fn test_fen_adjacent_digits() {
        // "44" is invalid FEN — should be "8"
        assert!(matches!(
            Position::from_fen("rnbqkbnr/pppppppp/44/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
            Err(FenParseError::AdjacentDigits { rank: 6 })
        ));
    }

    #[test]
    fn test_fen_adjacent_digits_rank1() {
        assert!(matches!(
            Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKB12 w KQkq - 0 1"),
            Err(FenParseError::AdjacentDigits { rank: 1 })
        ));
    }

    #[test]
    fn test_fen_ep_wrong_rank_white() {
        // White to move → EP must be on rank 6, not rank 4
        assert!(matches!(
            Position::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e4 0 1"),
            Err(FenParseError::InvalidEpRank { .. })
        ));
    }

    #[test]
    fn test_fen_ep_wrong_rank_black() {
        // Black to move → EP must be on rank 3, not rank 6
        assert!(matches!(
            Position::from_fen("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR b KQkq e6 0 1"),
            Err(FenParseError::InvalidEpRank { .. })
        ));
    }

    #[test]
    fn test_fen_ep_valid_rank() {
        // Valid: black to move, EP on e3 (rank 3), white pawn on e4
        assert!(Position::from_fen(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        ).is_ok());
    }

    #[test]
    fn test_fen_fullmove_zero() {
        assert!(matches!(
            Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 0"),
            Err(FenParseError::FullmoveNumberZero)
        ));
    }

    // ============================================================
    // FEN validation tests — semantic
    // ============================================================

    #[test]
    fn test_fen_no_white_king() {
        assert!(matches!(
            Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQ1BNR w - - 0 1"),
            Err(FenParseError::InvalidKingCount { color: "white", count: 0 })
        ));
    }

    #[test]
    fn test_fen_no_black_king() {
        assert!(matches!(
            Position::from_fen("rnbq1bnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"),
            Err(FenParseError::InvalidKingCount { color: "black", count: 0 })
        ));
    }

    #[test]
    fn test_fen_two_white_kings() {
        assert!(matches!(
            Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBKKBNR w - - 0 1"),
            Err(FenParseError::InvalidKingCount { color: "white", count: 2 })
        ));
    }

    #[test]
    fn test_fen_opponent_in_check_white() {
        // White to move, but black king on e8 is attacked by white queen on e1
        assert!(matches!(
            Position::from_fen("4k3/8/8/8/8/8/8/4Q2K w - - 0 1"),
            Err(FenParseError::OpponentKingInCheck)
        ));
    }

    #[test]
    fn test_fen_opponent_in_check_black() {
        // Black to move, white king attacked by black rook on a1
        assert!(matches!(
            Position::from_fen("4k3/8/8/8/8/8/8/r3K3 b - - 0 1"),
            Err(FenParseError::OpponentKingInCheck)
        ));
    }

    #[test]
    fn test_fen_pawns_on_rank1() {
        assert!(matches!(
            Position::from_fen("4k3/8/8/8/8/8/8/P3K3 w - - 0 1"),
            Err(FenParseError::PawnsOnBackRank)
        ));
    }

    #[test]
    fn test_fen_pawns_on_rank8() {
        assert!(matches!(
            Position::from_fen("p3k3/8/8/8/8/8/8/4K3 w - - 0 1"),
            Err(FenParseError::PawnsOnBackRank)
        ));
    }

    #[test]
    fn test_fen_too_many_pawns() {
        // 9 white pawns
        assert!(matches!(
            Position::from_fen("4k3/8/8/8/P7/PPPPPPPP/8/4K3 w - - 0 1"),
            Err(FenParseError::TooManyPawns { color: "white", count: 9 })
        ));
    }

    #[test]
    fn test_fen_too_many_promotions() {
        // 3 white queens + 8 white pawns = 2 extra queens, 0 missing pawns
        assert!(matches!(
            Position::from_fen("4k3/8/8/8/8/QQQ5/PPPPPPPP/4K3 w - - 0 1"),
            Err(FenParseError::TooManyPromotions { color: "white", extra: 2, missing_pawns: 0 })
        ));
    }

    #[test]
    fn test_fen_promotions_valid() {
        // 3 white queens + 6 white pawns = 2 extra queens, 2 missing pawns → OK
        assert!(Position::from_fen(
            "4k3/8/8/8/8/QQQ5/PPPPPP2/4K3 w - - 0 1"
        ).is_ok());
    }

    #[test]
    fn test_fen_castling_no_rook() {
        // White O-O but no rook on h1
        assert!(matches!(
            Position::from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1 w Kkq - 0 1"),
            Err(FenParseError::CastlingRightsIncoherent { .. })
        ));
    }

    #[test]
    fn test_fen_castling_no_king() {
        // White O-O but king not on e1
        assert!(matches!(
            Position::from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/RNBQ1BNR w Kkq - 0 1"),
            Err(FenParseError::InvalidKingCount { color: "white", count: 0 })
        ));
    }

    #[test]
    fn test_fen_ep_no_pawn() {
        // EP on e3 but no white pawn on e4
        assert!(matches!(
            Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq e3 0 1"),
            Err(FenParseError::EpSquareNoPawn { .. })
        ));
    }

    // ============================================================
    // FEN validation — regression tests
    // ============================================================

    #[test]
    fn test_bench_fens_all_valid() {
        for fen in crate::bench::POSITIONS {
            assert!(Position::from_fen(fen).is_ok(), "bench FEN rejected: {fen}");
        }
    }

    #[test]
    fn test_valid_positions_still_accepted() {
        let valid_fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "4k3/8/8/8/8/8/8/4K3 w - - 0 1",           // bare kings
            "4k3/8/8/8/8/8/8/4K3 w - - 100 1",         // halfmove 100
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "4k3/8/8/8/8/QQ6/PPPPPP2/4K3 w - - 0 1",   // 2 queens + 6 pawns (valid promotion)
        ];
        for fen in valid_fens {
            assert!(Position::from_fen(fen).is_ok(), "valid FEN rejected: {fen}");
        }
    }
}
