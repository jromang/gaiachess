//! Syzygy tablebase probing via Pyrrhic (C library).
//!
//! Provides WDL probing for search and datagen. The engine's attack generators
//! are exposed to Pyrrhic via FFI callbacks (see `gaiachess_bridge.h`).
//!
//! Gated behind `--features syzygy`.

#[cfg(feature = "syzygy")]
use crate::bitboard;
#[cfg(feature = "syzygy")]
use crate::position::Position;
#[cfg(feature = "syzygy")]
use crate::types::*;

// ============================================================
// FFI callbacks (called by Pyrrhic C code)
// ============================================================

#[cfg(feature = "syzygy")]
#[unsafe(no_mangle)]
pub extern "C" fn gaiachess_popcount(bb: u64) -> u32 {
    bb.count_ones()
}

#[cfg(feature = "syzygy")]
#[unsafe(no_mangle)]
pub extern "C" fn gaiachess_lsb(bb: u64) -> u32 {
    bb.trailing_zeros()
}

#[cfg(feature = "syzygy")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gaiachess_poplsb(bb: *mut u64) -> u64 {
    unsafe {
        let value = *bb;
        *bb = value & (value - 1);
        u64::from(value.trailing_zeros())
    }
}

/// `colour`: true = white, false = black (matches Pyrrhic convention).
#[cfg(feature = "syzygy")]
#[unsafe(no_mangle)]
pub extern "C" fn gaiachess_pawn_attacks(sq: u32, colour: bool) -> u64 {
    let c = if colour { Color::White } else { Color::Black };
    bitboard::pawn_attacks(Square(sq as u8), c)
}

#[cfg(feature = "syzygy")]
#[unsafe(no_mangle)]
pub extern "C" fn gaiachess_knight_attacks(sq: u32) -> u64 {
    bitboard::knight_attacks(Square(sq as u8))
}

#[cfg(feature = "syzygy")]
#[unsafe(no_mangle)]
pub extern "C" fn gaiachess_bishop_attacks(sq: u32, occupied: u64) -> u64 {
    bitboard::bishop_attacks(Square(sq as u8), occupied)
}

#[cfg(feature = "syzygy")]
#[unsafe(no_mangle)]
pub extern "C" fn gaiachess_rook_attacks(sq: u32, occupied: u64) -> u64 {
    bitboard::rook_attacks(Square(sq as u8), occupied)
}

#[cfg(feature = "syzygy")]
#[unsafe(no_mangle)]
pub extern "C" fn gaiachess_queen_attacks(sq: u32, occupied: u64) -> u64 {
    bitboard::bishop_attacks(Square(sq as u8), occupied)
        | bitboard::rook_attacks(Square(sq as u8), occupied)
}

#[cfg(feature = "syzygy")]
#[unsafe(no_mangle)]
pub extern "C" fn gaiachess_king_attacks(sq: u32) -> u64 {
    bitboard::king_attacks(Square(sq as u8))
}

// ============================================================
// Pyrrhic C bindings (manual, no bindgen)
// ============================================================

#[cfg(feature = "syzygy")]
#[allow(dead_code)]
mod ffi {
    pub const TB_LOSS: u32 = 0;
    pub const TB_BLESSED_LOSS: u32 = 1;
    pub const TB_DRAW: u32 = 2;
    pub const TB_CURSED_WIN: u32 = 3;
    pub const TB_WIN: u32 = 4;

    pub const TB_MAX_MOVES: usize = 256;

    #[repr(C)]
    pub struct TbRootMove {
        pub mv: u16, // PyrrhicMove
        pub tb_rank: i32,
    }

    #[repr(C)]
    pub struct TbRootMoves {
        pub size: u32,
        pub moves: [TbRootMove; TB_MAX_MOVES],
    }

    unsafe extern "C" {
        pub fn tb_init(path: *const std::ffi::c_char) -> bool;
        pub fn tb_free();
        pub fn tb_probe_wdl(
            white: u64, black: u64,
            kings: u64, queens: u64, rooks: u64,
            bishops: u64, knights: u64, pawns: u64,
            ep: u32, turn: bool,
        ) -> u32;
        pub fn tb_probe_root_dtz(
            white: u64, black: u64,
            kings: u64, queens: u64, rooks: u64,
            bishops: u64, knights: u64, pawns: u64,
            rule50: u32, ep: u32, turn: bool, has_repeated: bool,
            results: *mut TbRootMoves,
        ) -> i32;
        pub fn tb_probe_root_wdl(
            white: u64, black: u64,
            kings: u64, queens: u64, rooks: u64,
            bishops: u64, knights: u64, pawns: u64,
            rule50: u32, ep: u32, turn: bool, use_rule50: bool,
            results: *mut TbRootMoves,
        ) -> i32;
        pub static TB_LARGEST: i32;
    }
}

// ============================================================
// Public API
// ============================================================

/// Win/Draw/Loss from the side-to-move's perspective.
#[cfg(feature = "syzygy")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wdl {
    Win,
    Draw,
    Loss,
}

/// Initialize Syzygy tablebases from the given path.
/// Returns true on success. Can be called with multiple paths separated by `;` (Windows) or `:` (Unix).
#[cfg(feature = "syzygy")]
pub fn init(path: &str) -> bool {
    let cstr = std::ffi::CString::new(path).expect("invalid syzygy path");
    unsafe { ffi::tb_init(cstr.as_ptr()) }
}

/// Free tablebase resources.
#[cfg(feature = "syzygy")]
#[allow(dead_code)]
pub fn free() {
    unsafe { ffi::tb_free() }
}

/// Maximum number of pieces supported by the loaded tablebases.
/// Returns 0 if no tablebases are loaded.
#[cfg(feature = "syzygy")]
pub fn max_pieces() -> u32 {
    unsafe { ffi::TB_LARGEST as u32 }
}

/// Extract the 10 Pyrrhic arguments from a Position.
#[cfg(feature = "syzygy")]
struct PyrrhicArgs {
    white: u64, black: u64,
    kings: u64, queens: u64, rooks: u64,
    bishops: u64, knights: u64, pawns: u64,
    ep: u32, turn: bool,
}

#[cfg(feature = "syzygy")]
impl PyrrhicArgs {
    fn from_pos(pos: &Position) -> Self {
        Self {
            white: pos.color_bb(Color::White),
            black: pos.color_bb(Color::Black),
            kings: pos.piece_type_bb(PieceType::King, Color::White)
                | pos.piece_type_bb(PieceType::King, Color::Black),
            queens: pos.piece_type_bb(PieceType::Queen, Color::White)
                | pos.piece_type_bb(PieceType::Queen, Color::Black),
            rooks: pos.piece_type_bb(PieceType::Rook, Color::White)
                | pos.piece_type_bb(PieceType::Rook, Color::Black),
            bishops: pos.piece_type_bb(PieceType::Bishop, Color::White)
                | pos.piece_type_bb(PieceType::Bishop, Color::Black),
            knights: pos.piece_type_bb(PieceType::Knight, Color::White)
                | pos.piece_type_bb(PieceType::Knight, Color::Black),
            pawns: pos.piece_type_bb(PieceType::Pawn, Color::White)
                | pos.piece_type_bb(PieceType::Pawn, Color::Black),
            ep: if pos.ep_square != Square::NONE { pos.ep_square.0 as u32 } else { 0 },
            turn: pos.side_to_move == Color::White,
        }
    }
}

/// Convert a PyrrhicMove (u16) to a GaiaChess Move.
///
/// Pyrrhic layout: `to[5:0] | from[11:6] | flags[15:12]`
/// GaiaChess layout: `to[5:0] | from[11:6] | promo[13:12] | move_type[15:14]`
#[cfg(feature = "syzygy")]
fn pyrrhic_to_move(pm: u16) -> Move {
    let from = Square(((pm >> 6) & 0x3F) as u8);
    let to = Square((pm & 0x3F) as u8);
    let flags = (pm >> 12) & 0xF;

    // Pyrrhic flags: 0=none, 1=Q promo, 2=R promo, 3=B promo, 4=N promo, 8=EP
    match flags {
        0 => Move::new(from, to),
        1 => Move::new_promotion(from, to, PieceType::Queen),
        2 => Move::new_promotion(from, to, PieceType::Rook),
        3 => Move::new_promotion(from, to, PieceType::Bishop),
        4 => Move::new_promotion(from, to, PieceType::Knight),
        8 => Move::new_with_type(from, to, MT_EN_PASSANT),
        _ => Move::NONE,
    }
}

/// Probe WDL for a position. Returns `None` if:
/// - Position has castling rights (TB don't cover these)
/// - Fifty-move counter != 0 (WDL tables assume rule50 = 0)
/// - Piece count exceeds loaded tables
/// - Probe fails
///
/// Result is from the **side-to-move's** perspective.
#[cfg(feature = "syzygy")]
pub fn probe_wdl(pos: &Position) -> Option<Wdl> {
    if pos.castling_rights != 0 || pos.halfmove_clock != 0 {
        return None;
    }
    if pos.occupied().count_ones() > max_pieces() {
        return None;
    }

    let a = PyrrhicArgs::from_pos(pos);
    let result = unsafe {
        ffi::tb_probe_wdl(a.white, a.black, a.kings, a.queens, a.rooks,
                          a.bishops, a.knights, a.pawns, a.ep, a.turn)
    };

    match result {
        ffi::TB_WIN => Some(Wdl::Win),
        ffi::TB_LOSS => Some(Wdl::Loss),
        ffi::TB_DRAW | ffi::TB_CURSED_WIN | ffi::TB_BLESSED_LOSS => Some(Wdl::Draw),
        _ => None,
    }
}

/// Pyrrhic TB_MAX_DTZ constant (must match tbprobe.c).
#[cfg(feature = "syzygy")]
const MAX_DTZ: i32 = 0x40000; // 262144

/// Compute a tbScore from a Pyrrhic tbRank.
/// Certain wins → SCORE_TB_WIN. Marginal wins → small cp. Draw → 0. Losses → mirror.
#[cfg(feature = "syzygy")]
fn compute_tb_score(tb_rank: i32) -> i32 {
    let bound = MAX_DTZ / 2 - 100; // 130972
    if tb_rank >= bound {
        SCORE_TB_WIN // Certain win before 50-move rule
    } else if tb_rank > 0 {
        // Gradual: scale from ~1 to ~49 cp (SF: max(3, r - (MAX_DTZ/2-200)) * PawnValue / 200)
        let r = tb_rank.max(3) - (MAX_DTZ / 2 - 200);
        (r.max(3) * 100) / 200
    } else if tb_rank == 0 {
        0
    } else if tb_rank > -bound {
        let r = tb_rank.min(-3) + (MAX_DTZ / 2 - 200);
        (r.min(-3) * 100) / 200
    } else {
        -SCORE_TB_WIN // Certain loss
    }
}

/// Convert raw Pyrrhic TbRootMoves into `Vec<(Move, tb_rank, tb_score)>`, sorted descending.
#[cfg(feature = "syzygy")]
fn extract_root_moves(results: &ffi::TbRootMoves) -> Vec<(Move, i32, i32)> {
    let mut moves = Vec::with_capacity(results.size as usize);
    for i in 0..results.size as usize {
        let rm = &results.moves[i];
        let m = pyrrhic_to_move(rm.mv);
        if m != Move::NONE {
            let tb_score = compute_tb_score(rm.tb_rank);
            moves.push((m, rm.tb_rank, tb_score));
        }
    }
    moves.sort_by(|a, b| b.1.cmp(&a.1));
    moves
}

/// Probe root DTZ (precise, 50-move aware). Returns None if tables unavailable.
#[cfg(feature = "syzygy")]
fn probe_root_dtz(pos: &Position) -> Option<Vec<(Move, i32, i32)>> {
    let a = PyrrhicArgs::from_pos(pos);
    let mut results = std::mem::MaybeUninit::<ffi::TbRootMoves>::uninit();
    let success = unsafe {
        ffi::tb_probe_root_dtz(
            a.white, a.black, a.kings, a.queens, a.rooks,
            a.bishops, a.knights, a.pawns,
            pos.halfmove_clock as u32, a.ep, a.turn, false,
            results.as_mut_ptr(),
        )
    };
    if success == 0 { return None; }
    let results = unsafe { results.assume_init() };
    Some(extract_root_moves(&results))
}

/// Probe root WDL (fallback when DTZ unavailable). Returns None if probe fails.
#[cfg(feature = "syzygy")]
fn probe_root_wdl(pos: &Position) -> Option<Vec<(Move, i32, i32)>> {
    let a = PyrrhicArgs::from_pos(pos);
    let mut results = std::mem::MaybeUninit::<ffi::TbRootMoves>::uninit();
    let success = unsafe {
        ffi::tb_probe_root_wdl(
            a.white, a.black, a.kings, a.queens, a.rooks,
            a.bishops, a.knights, a.pawns,
            pos.halfmove_clock as u32, a.ep, a.turn, true,
            results.as_mut_ptr(),
        )
    };
    if success == 0 { return None; }
    let results = unsafe { results.assume_init() };
    Some(extract_root_moves(&results))
}

/// Rank all root moves using Syzygy tablebases.
///
/// Tries DTZ first (precise, 50-move aware). Falls back to WDL if DTZ unavailable.
/// Returns `(ranked_moves, dtz_available)` or None if position not in TB.
///
/// `dtz_available = true` → cardinality should be 0 (no in-tree probing needed).
/// `dtz_available = false` → WDL only, may need in-tree probing.
#[cfg(feature = "syzygy")]
pub fn rank_root_moves(pos: &Position) -> Option<(Vec<(Move, i32, i32)>, bool)> {
    if pos.occupied().count_ones() > max_pieces() || pos.castling_rights != 0 {
        return None;
    }

    // Try DTZ first (precise ranking, accounts for 50-move rule)
    if let Some(ranked) = probe_root_dtz(pos) {
        return Some((ranked, true));
    }

    // Fallback to WDL (less precise, no DTZ info)
    if let Some(ranked) = probe_root_wdl(pos) {
        return Some((ranked, false));
    }

    None
}
