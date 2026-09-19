//! Nalimov DTM prober — 3-6 piece endgame tablebases on disk.
//!
//! Pure Rust implementation via the `nalimov` crate. Thread-safe (internal
//! Mutex + configurable LRU block cache). Supports en passant natively via
//! disjoint index ranges in the Nalimov table format.
//!
//! Probing strategy (inspired by Crafty): main search only, depth ≥ 2,
//! no qsearch (too many nodes → disk I/O catastrophic). The TT propagates
//! Nalimov scores to shallower depths.
//!
//! Gated behind `--features nalimov`.

use std::sync::RwLock;
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

use nalimov_tb::{NalimovProber, NalimovResult};

use crate::position::Position;
use crate::types::*;

struct LoadedProber {
    prober: NalimovProber,
    paths: String,
}

static PROBER: RwLock<Option<LoadedProber>> = RwLock::new(None);
/// Max pieces detected automatically at load time by scanning TB files.
static MAX_PIECES: AtomicU32 = AtomicU32::new(0);
/// UCI `NalimovCache`, in MiB.
static CACHE_SIZE_MB: AtomicU32 = AtomicU32::new(256);

/// Initialize the prober from colon- or semicolon-separated (Unix) or
/// semicolon-separated (Windows) directory paths. Detects the max piece count
/// automatically by scanning for `.nbw.emd` / `.nbb.emd` files.
///
/// On Windows, `:` must NOT be a separator: it would split the drive letter
/// out of paths like `H:\nalimov` and make every absolute path unloadable.
pub fn init(paths: &str) -> bool {
    #[cfg(windows)]
    let path_list: Vec<&str> = paths.split(';').collect();
    #[cfg(not(windows))]
    let path_list: Vec<&str> = paths.split([':', ';']).collect();
    let detected = detect_max_pieces(&path_list);
    match NalimovProber::new(&path_list, cache_size_mb() as usize) {
        Ok(prober) => {
            MAX_PIECES.store(detected, Relaxed);
            *PROBER.write().expect("Nalimov prober lock poisoned") = Some(LoadedProber {
                prober,
                paths: paths.to_owned(),
            });
            true
        }
        Err(_) => {
            MAX_PIECES.store(0, Relaxed);
            *PROBER.write().expect("Nalimov prober lock poisoned") = None;
            false
        }
    }
}

/// Unload the current tablebases and release their cache.
pub fn free() {
    MAX_PIECES.store(0, Relaxed);
    *PROBER.write().expect("Nalimov prober lock poisoned") = None;
}

/// Set the decompressed-block cache size. A loaded prober is rebuilt immediately.
pub fn set_cache_size_mb(cache_size_mb: u32) -> bool {
    CACHE_SIZE_MB.store(cache_size_mb, Relaxed);
    let paths = PROBER
        .read()
        .expect("Nalimov prober lock poisoned")
        .as_ref()
        .map(|loaded| loaded.paths.clone());
    paths.is_none_or(|paths| init(&paths))
}

/// Configured decompressed-block cache size, in MiB.
pub fn cache_size_mb() -> u32 {
    CACHE_SIZE_MB.load(Relaxed)
}

/// Returns true if Nalimov tables are loaded and ready.
pub fn available() -> bool {
    PROBER
        .read()
        .expect("Nalimov prober lock poisoned")
        .is_some()
}

/// Max piece count available in loaded tables (3-6).
pub fn max_pieces() -> u32 {
    MAX_PIECES.load(Relaxed)
}

/// Probe DTM for a position. Returns a **STM-relative negamax score** ready
/// to return from `alpha_beta` via [`dtm_score`]: short mates stay mates,
/// longer ones become TB wins (`cp ±20000` on the wire). `None` if the
/// position is not in the loaded tables or if its DTM does not prove a mate
/// before the current 50-move-rule deadline.
///
/// Supports en passant natively — the `nalimov` crate handles disjoint
/// index ranges for EP positions.
pub fn probe_position(pos: &Position, ply: i32) -> Option<i32> {
    let loaded = PROBER.read().ok()?;
    let prober = &loaded.as_ref()?.prober;

    let white = pos.color_bb(Color::White);
    let black = pos.color_bb(Color::Black);
    let kings = pos.piece_type_bb(PieceType::King, Color::White)
        | pos.piece_type_bb(PieceType::King, Color::Black);
    let queens = pos.piece_type_bb(PieceType::Queen, Color::White)
        | pos.piece_type_bb(PieceType::Queen, Color::Black);
    let rooks = pos.piece_type_bb(PieceType::Rook, Color::White)
        | pos.piece_type_bb(PieceType::Rook, Color::Black);
    let bishops = pos.piece_type_bb(PieceType::Bishop, Color::White)
        | pos.piece_type_bb(PieceType::Bishop, Color::Black);
    let knights = pos.piece_type_bb(PieceType::Knight, Color::White)
        | pos.piece_type_bb(PieceType::Knight, Color::Black);
    let pawns = pos.piece_type_bb(PieceType::Pawn, Color::White)
        | pos.piece_type_bb(PieceType::Pawn, Color::Black);

    let wtm = pos.side_to_move == Color::White;
    let ep = if pos.ep_square == Square::NONE {
        64u8
    } else {
        pos.ep_square.0
    };

    prober
        .probe(
            white, black, kings, queens, rooks, bishops, knights, pawns, wtm, ep,
        )
        .ok()
        .and_then(|res| result_score(res, ply, pos.halfmove_clock))
}

/// Map a probe result to a negamax score at `ply`.
///
/// The tables give the side to move `plies` half-moves to mate (a win) or to be
/// mated (a loss). A checkmated side has zero of them: that is a loss now, not a
/// draw — only `Draw` is a draw.
///
/// Nalimov DTM ignores the 50-move rule. When the mate distance exceeds the
/// remaining clock budget, it may still be a win after a capture or pawn move,
/// but DTM alone cannot prove that. Decline the probe instead of turning a
/// cursed win or blessed loss into a certain tablebase result.
fn result_score(res: NalimovResult, ply: i32, halfmove_clock: u8) -> Option<i32> {
    let moves_left = 100u16.saturating_sub(halfmove_clock as u16);
    match res {
        NalimovResult::Draw => Some(0),
        NalimovResult::Win { plies } => {
            debug_assert!(plies >= 1, "a win takes at least one half-move");
            (plies <= moves_left).then(|| dtm_score(ply, plies as i32))
        }
        NalimovResult::Loss { plies } => {
            (plies <= moves_left).then(|| dtm_score(ply, -(plies as i32)))
        }
    }
}

/// Detect the max piece count by scanning Nalimov file names.
/// Files are named like `KQKR.nbw.emd` — the letter count before the first
/// `.` gives the number of pieces (KQKR = 4 pieces).
fn detect_max_pieces(paths: &[&str]) -> u32 {
    let mut max = 0u32;
    for path in paths {
        let Ok(entries) = std::fs::read_dir(path) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.ends_with(".nbw.emd") || name.ends_with(".nbb.emd") {
                let base = name.split('.').next().unwrap_or("");
                let pieces = base.len() as u32;
                max = max.max(pieces);
            }
        }
    }
    if max < 3 { 0 } else { max.min(6) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{mate_in, mated_in};

    /// The tables encode a checkmated side as a loss in zero half-moves. That is
    /// the mate, not a draw: scored as equal, the child of every mating move looked
    /// like nothing to play and the engine shuffled with "mate in 2" on the board.
    #[test]
    fn a_loss_in_zero_half_moves_is_the_mate_itself() {
        assert_eq!(
            result_score(NalimovResult::Loss { plies: 0 }, 1, 100),
            Some(mated_in(1))
        );
        assert_eq!(
            result_score(NalimovResult::Loss { plies: 0 }, 5, 100),
            Some(mated_in(5))
        );
        assert_eq!(
            result_score(NalimovResult::Loss { plies: 2 }, 1, 0),
            Some(mated_in(3))
        );
        assert_eq!(
            result_score(NalimovResult::Win { plies: 1 }, 1, 0),
            Some(mate_in(2))
        );
        assert_eq!(
            result_score(NalimovResult::Win { plies: 5 }, 0, 0),
            Some(mate_in(5))
        );
        assert_eq!(result_score(NalimovResult::Draw, 3, 99), Some(0));
    }

    #[test]
    fn dtm_must_fit_before_the_fifty_move_deadline() {
        assert_eq!(
            result_score(NalimovResult::Win { plies: 5 }, 0, 95),
            Some(mate_in(5))
        );
        assert_eq!(result_score(NalimovResult::Win { plies: 6 }, 0, 95), None);
        assert_eq!(
            result_score(NalimovResult::Loss { plies: 5 }, 0, 95),
            Some(mated_in(5))
        );
        assert_eq!(result_score(NalimovResult::Loss { plies: 6 }, 0, 95), None);
    }
}
