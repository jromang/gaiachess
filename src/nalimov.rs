//! Nalimov DTM prober — 3-6 piece endgame tablebases on disk.
//!
//! Pure Rust implementation via the `nalimov` crate. Thread-safe (internal
//! Mutex + 256 MB LRU block cache). Supports en passant natively via
//! disjoint index ranges in the Nalimov table format.
//!
//! Probing strategy (inspired by Crafty): main search only, depth ≥ 2,
//! no qsearch (too many nodes → disk I/O catastrophic). The TT propagates
//! Nalimov scores to shallower depths.
//!
//! Gated behind `--features nalimov`.

use std::sync::atomic::{AtomicU32, Ordering::Relaxed};
use std::sync::OnceLock;

use nalimov_tb::{NalimovProber, NalimovResult};

use crate::position::Position;
use crate::types::*;

static PROBER: OnceLock<Option<NalimovProber>> = OnceLock::new();
/// Max pieces detected automatically at load time by scanning TB files.
static MAX_PIECES: AtomicU32 = AtomicU32::new(0);

/// Initialize the prober from colon-separated (Unix) or semicolon-separated
/// (Windows) directory paths. Detects the max piece count automatically by
/// scanning for `.nbw.emd` / `.nbb.emd` files.
pub fn init(paths: &str) -> bool {
    let path_list: Vec<&str> = paths.split([':', ';']).collect();
    let detected = detect_max_pieces(&path_list);
    match NalimovProber::new(&path_list) {
        Ok(prober) => {
            MAX_PIECES.store(detected, Relaxed);
            let _ = PROBER.set(Some(prober));
            true
        }
        Err(_) => {
            let _ = PROBER.set(None);
            false
        }
    }
}

/// Returns true if Nalimov tables are loaded and ready.
pub fn available() -> bool {
    PROBER.get().is_some_and(|p| p.is_some())
}

/// Max piece count available in loaded tables (3-6).
pub fn max_pieces() -> u32 {
    MAX_PIECES.load(Relaxed)
}

/// Probe DTM for a position. Returns a **STM-relative negamax score** ready
/// to return from `alpha_beta`:
/// - `SCORE_MATE - ply - plies` if the side to move wins
/// - `-(SCORE_MATE) + ply + plies` if the side to move loses
/// - `0` for draws
/// - `None` if the position is not in the loaded tables
///
/// Supports en passant natively — the `nalimov` crate handles disjoint
/// index ranges for EP positions.
pub fn probe_position(pos: &Position, ply: i32) -> Option<i32> {
    let prober = PROBER.get()?.as_ref()?;

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

    match prober.probe(white, black, kings, queens, rooks, bishops, knights, pawns, wtm, ep) {
        Ok(NalimovResult::Win { plies }) => {
            Some(SCORE_MATE - ply - plies as i32)
        }
        Ok(NalimovResult::Loss { plies }) => {
            Some(-SCORE_MATE + ply + plies as i32)
        }
        Ok(NalimovResult::Draw) => Some(0),
        Err(_) => None,
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
