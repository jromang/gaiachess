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
            white: u64,
            black: u64,
            kings: u64,
            queens: u64,
            rooks: u64,
            bishops: u64,
            knights: u64,
            pawns: u64,
            ep: u32,
            turn: bool,
        ) -> u32;
        pub fn tb_probe_root_dtz(
            white: u64,
            black: u64,
            kings: u64,
            queens: u64,
            rooks: u64,
            bishops: u64,
            knights: u64,
            pawns: u64,
            rule50: u32,
            ep: u32,
            turn: bool,
            has_repeated: bool,
            rank_dtz: bool,
            results: *mut TbRootMoves,
        ) -> i32;
        pub fn tb_probe_root_wdl(
            white: u64,
            black: u64,
            kings: u64,
            queens: u64,
            rooks: u64,
            bishops: u64,
            knights: u64,
            pawns: u64,
            rule50: u32,
            ep: u32,
            turn: bool,
            use_rule50: bool,
            results: *mut TbRootMoves,
        ) -> i32;
        // Written by `tb_init` (and `tb_free`) long after the program starts: an
        // immutable extern static would promise the compiler it never changes.
        pub static mut TB_LARGEST: i32;
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

/// Initialize Syzygy tablebases from the given path. Can be called with multiple paths
/// separated by `;` (Windows) or `:` (Unix).
///
/// Returns whether any table was found. Pyrrhic's own answer is true however empty the
/// directory, so on its own it could not tell a wrong path from a right one: what was
/// loaded is read from [`max_pieces`] instead (see [`file_counts`] for the detail).
#[cfg(feature = "syzygy")]
pub fn init(path: &str) -> bool {
    let cstr = std::ffi::CString::new(path).expect("invalid syzygy path");
    let ok = unsafe { ffi::tb_init(cstr.as_ptr()) };
    ok && max_pieces() > 0
}

/// How many WDL and DTZ files the last [`init`] found.
#[cfg(feature = "syzygy")]
pub fn file_counts() -> (u32, u32) {
    // Written by `tb_init` long after start-up, hence `mut`; read by value only.
    unsafe extern "C" {
        static mut TB_NUM_WDL: i32;
        static mut TB_NUM_DTZ: i32;
    }
    // SAFETY: plain integers, read after `init` by the thread that called it.
    let (wdl, dtz) = unsafe { (TB_NUM_WDL, TB_NUM_DTZ) };
    debug_assert!(wdl >= 0 && dtz >= 0);
    (wdl.max(0) as u32, dtz.max(0) as u32)
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
    // Read by value: no reference to the mutable static is ever formed.
    let largest = unsafe { ffi::TB_LARGEST };
    debug_assert!((0..=7).contains(&largest), "TB_LARGEST out of range: {largest}");
    largest as u32
}

/// Extract the 10 Pyrrhic arguments from a Position.
#[cfg(feature = "syzygy")]
struct PyrrhicArgs {
    white: u64,
    black: u64,
    kings: u64,
    queens: u64,
    rooks: u64,
    bishops: u64,
    knights: u64,
    pawns: u64,
    ep: u32,
    turn: bool,
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
            ep: if pos.ep_square != Square::NONE {
                pos.ep_square.0 as u32
            } else {
                0
            },
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
        ffi::tb_probe_wdl(
            a.white, a.black, a.kings, a.queens, a.rooks, a.bishops, a.knights, a.pawns, a.ep,
            a.turn,
        )
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
/// Certain wins → SCORE_TB_WIN. Cursed wins → a fraction of a pawn, in the
/// engine's internal units (the network pawn). Draw → 0. Losses → mirror.
///
/// This Pyrrhic bridge ranks a cursed win as `MAX_DTZ - (dtz + rule50)`,
/// unlike the newer half-range convention used by the reference scorer. The
/// first cursed rank is therefore `MAX_DTZ - 101`, not `MAX_DTZ / 2 - 101`.
#[cfg(feature = "syzygy")]
fn compute_tb_score(tb_rank: i32) -> i32 {
    // One pawn in the units the search (and the score display) works in.
    const TB_PAWN: i32 = EVAL_PIECE_VALUE[PieceType::Pawn as usize];
    let win_bound = MAX_DTZ - 100;
    if tb_rank >= win_bound {
        SCORE_TB_WIN // Certain win before 50-move rule
    } else if tb_rank > 0 {
        // Gradual: from ~1/2 pawn down to ~1/80 pawn (internal units, about 51 to
        // 1 displayed cp) as DTZ approaches the rule-50 limit. Never 0, so a
        // cursed win never prints as a draw.
        let r = tb_rank - (MAX_DTZ - 200);
        let score = (r.max(3) * TB_PAWN) / 200;
        debug_assert!(score > 0 && score < TB_PAWN);
        score
    } else if tb_rank == 0 {
        0
    } else if tb_rank > -win_bound {
        let r = tb_rank + (MAX_DTZ - 200);
        let score = (r.min(-3) * TB_PAWN) / 200;
        debug_assert!(score < 0 && score > -TB_PAWN);
        score
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
///
/// `rank_dtz` ranks the certain wins and losses by DTZ too, shortest win and longest
/// loss first; without it they all rank alike, until a repetition has been seen.
#[cfg(feature = "syzygy")]
fn probe_root_dtz(pos: &Position, rank_dtz: bool) -> Option<Vec<(Move, i32, i32)>> {
    let a = PyrrhicArgs::from_pos(pos);
    // Zeroed, not uninit: Pyrrhic writes only `size` and `moves[..size]`, and
    // `assume_init` on a partly uninitialised buffer is undefined behaviour.
    let mut results = std::mem::MaybeUninit::<ffi::TbRootMoves>::zeroed();
    let success = unsafe {
        ffi::tb_probe_root_dtz(
            a.white,
            a.black,
            a.kings,
            a.queens,
            a.rooks,
            a.bishops,
            a.knights,
            a.pawns,
            pos.halfmove_clock as u32,
            a.ep,
            a.turn,
            // After a repetition every winning move no longer ranks alike: the
            // shortest DTZ comes first, so the engine makes progress.
            pos.has_repeated(),
            rank_dtz,
            results.as_mut_ptr(),
        )
    };
    if success == 0 {
        return None;
    }
    let results = unsafe { results.assume_init() };
    debug_assert!(results.size as usize <= ffi::TB_MAX_MOVES);
    Some(extract_root_moves(&results))
}

/// Probe root WDL (fallback when DTZ unavailable). Returns None if probe fails.
#[cfg(feature = "syzygy")]
fn probe_root_wdl(pos: &Position) -> Option<Vec<(Move, i32, i32)>> {
    let a = PyrrhicArgs::from_pos(pos);
    // Zeroed, not uninit: Pyrrhic writes only `size` and `moves[..size]`, and
    // `assume_init` on a partly uninitialised buffer is undefined behaviour.
    let mut results = std::mem::MaybeUninit::<ffi::TbRootMoves>::zeroed();
    let success = unsafe {
        ffi::tb_probe_root_wdl(
            a.white,
            a.black,
            a.kings,
            a.queens,
            a.rooks,
            a.bishops,
            a.knights,
            a.pawns,
            pos.halfmove_clock as u32,
            a.ep,
            a.turn,
            true,
            results.as_mut_ptr(),
        )
    };
    if success == 0 {
        return None;
    }
    let results = unsafe { results.assume_init() };
    debug_assert!(results.size as usize <= ffi::TB_MAX_MOVES);
    Some(extract_root_moves(&results))
}

/// Rank all root moves using Syzygy tablebases.
///
/// Tries DTZ first (precise, 50-move aware). Falls back to WDL if DTZ unavailable.
/// Returns `(ranked_moves, dtz_available)` or None if position not in TB.
///
/// `dtz_available = true` → cardinality should be 0 (no in-tree probing needed).
/// `dtz_available = false` → WDL only, may need in-tree probing.
/// Each root move with its rank and score, and whether DTZ produced the ranking.
#[cfg(feature = "syzygy")]
pub type RootRanking = (Vec<(Move, i32, i32)>, bool);

#[cfg(feature = "syzygy")]
pub fn rank_root_moves(pos: &Position) -> Option<RootRanking> {
    rank_root_moves_with(pos, false)
}

/// Like [`rank_root_moves`], with every certain win and loss ranked by DTZ as well:
/// the move that leads to mate along the shortest zeroing path comes first, and on
/// the losing side the longest. For walking a line to mate, not for the root search,
/// which wants every certain win in one group.
#[cfg(feature = "syzygy")]
pub fn rank_root_moves_by_dtz(pos: &Position) -> Option<RootRanking> {
    rank_root_moves_with(pos, true)
}

#[cfg(feature = "syzygy")]
fn rank_root_moves_with(pos: &Position, rank_dtz: bool) -> Option<RootRanking> {
    if pos.occupied().count_ones() > max_pieces() || pos.castling_rights != 0 {
        return None;
    }

    // Try DTZ first (precise ranking, accounts for 50-move rule)
    if let Some(mut ranked) = probe_root_dtz(pos, rank_dtz) {
        normalize_draws(pos, &mut ranked);
        return Some((ranked, true));
    }

    // Fallback to WDL (less precise, no DTZ info)
    if let Some(mut ranked) = probe_root_wdl(pos) {
        normalize_draws(pos, &mut ranked);
        return Some((ranked, false));
    }

    None
}

/// Ranks as a draw every root move that draws in the game itself: one that completes a
/// threefold repetition, or reaches the 50-move limit. The tables know nothing of the
/// game's history, and would rank such a move as the win it would be on a fresh board.
#[cfg(feature = "syzygy")]
fn normalize_draws(pos: &Position, ranked: &mut [(Move, i32, i32)]) {
    let mut child = pos.clone();
    let mut changed = false;
    for (mv, rank, score) in ranked.iter_mut() {
        child.make_move(*mv);
        if child.is_draw(1) && *rank != 0 {
            *rank = 0;
            *score = 0;
            changed = true;
        }
        child.unmake_move(*mv);
    }
    if changed {
        ranked.sort_by(|a, b| b.1.cmp(&a.1));
    }
}

#[cfg(all(test, feature = "syzygy"))]
mod tests {
    use super::*;

    #[test]
    fn root_score_separates_cursed_results_from_certain_results() {
        assert_eq!(compute_tb_score(MAX_DTZ), SCORE_TB_WIN);
        assert_eq!(compute_tb_score(MAX_DTZ - 100), SCORE_TB_WIN);
        assert_eq!(compute_tb_score(MAX_DTZ - 101), 160);
        assert_eq!(compute_tb_score(MAX_DTZ - 200), 4);
        assert_eq!(compute_tb_score(0), 0);
        assert_eq!(compute_tb_score(-MAX_DTZ + 200), -4);
        assert_eq!(compute_tb_score(-MAX_DTZ + 101), -160);
        assert_eq!(compute_tb_score(-MAX_DTZ + 100), -SCORE_TB_WIN);
        assert_eq!(compute_tb_score(-MAX_DTZ), -SCORE_TB_WIN);
    }

    #[test]
    fn a_cursed_result_never_displays_as_a_draw() {
        // Even the smallest cursed win must not round to `cp 0` on the displayed
        // scale, from a bare-rook ending to the heaviest seven-man material.
        let fens = [
            "8/8/4k3/8/8/4K3/8/R7 w - - 0 1",
            "3qk3/8/8/8/8/8/8/2RQK3 w - - 0 1",
            "2rqk3/8/8/8/8/8/8/1RRQK3 w - - 0 1",
        ];
        for fen in fens {
            let pos = Position::from_fen(fen).expect("valid fen");
            for rank in [MAX_DTZ - 200, MAX_DTZ - 199, MAX_DTZ - 150, MAX_DTZ - 101] {
                let win = compute_tb_score(rank);
                let loss = compute_tb_score(-rank);
                assert!(crate::wdl::to_cp(win, &pos) > 0, "{fen}: rank {rank}");
                assert!(crate::wdl::to_cp(loss, &pos) < 0, "{fen}: rank {rank}");
                assert!(crate::wdl::to_cp(win, &pos) <= 60, "{fen}: rank {rank}");
            }
        }
    }
}
