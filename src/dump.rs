//! Offline analysis runner for the `dump` subcommand (features `stats`/`tree`).
//!
//! Runs a deterministic single-threaded fixed-depth search on one position and
//! exports search statistics (JSON) and, with the `tree` feature, the full
//! search tree in the GTREE binary format for offline analysis by
//! `tools/treescope/`.

use std::sync::atomic::Ordering;

use crate::position::Position;
use crate::search;
use crate::threads::{STOP, SharedState, ThreadData};
use crate::timeman::{SearchLimits, TimeManager};

const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/// Options for a dump run (parsed from the CLI in main.rs).
pub struct DumpOptions {
    /// FEN string, or "startpos".
    pub fen: String,
    /// UCI moves applied after the FEN (space-separated), may be empty.
    pub moves: Vec<String>,
    /// Fixed search depth.
    pub depth: i32,
    /// Node cap (0 = none). Applied as a search limit.
    pub nodes: u64,
    /// Hash size in MB.
    pub tt_mb: usize,
    /// Stats JSON output path ("-" = stdout, empty = disabled).
    pub stats_out: String,
    /// Tree dump output path (empty = disabled). Requires the `tree` feature.
    pub tree_out: String,
    /// Do not record subtrees with remaining depth below this value.
    pub min_record_depth: i32,
    /// Record per-move records inside quiescence nodes.
    pub qs_moves: bool,
    /// Tree buffer cap in MB; recording stops (truncated) beyond this.
    pub max_mb: usize,
    /// Chess960 mode: FEN castling rights and castling moves in king-takes-rook
    /// notation, as with `UCI_Chess960`.
    pub chess960: bool,
}

/// Build the position a game would reach: the FEN, then each UCI move, with the
/// history trimmed after every move exactly as `position ... moves` does, so a
/// game of any length can be replayed and searched.
pub fn replay(fen: &str, moves: &[String], chess960: bool) -> Result<Position, String> {
    let fen = if fen == "startpos" { STARTPOS } else { fen };
    let mut pos =
        Position::from_fen_ex(fen, chess960).map_err(|e| format!("Invalid FEN: {e}"))?;
    debug_assert_eq!(pos.chess960, chess960);
    for ms in moves {
        match crate::uci::parse_uci_move(&pos, ms) {
            Some(m) => {
                pos.make_move(m);
                pos.rebase_history();
            }
            None => return Err(format!("Illegal move in --moves: {ms}")),
        }
    }
    Ok(pos)
}

/// The probing tables loaded for this dump, as a JSON fragment for the tree
/// metadata: two dumps searched with different tables do not see the same tree.
#[cfg(feature = "tree")]
fn tables_json() -> String {
    #[cfg(all(feature = "gaiatb", gaiatb_embedded))]
    let gaiatb = crate::dtm::available();
    #[cfg(not(all(feature = "gaiatb", gaiatb_embedded)))]
    let gaiatb = false;
    #[cfg(feature = "syzygy")]
    let syzygy = crate::tb::max_pieces();
    #[cfg(not(feature = "syzygy"))]
    let syzygy = 0u32;
    #[cfg(feature = "nalimov")]
    let nalimov = crate::nalimov::max_pieces();
    #[cfg(not(feature = "nalimov"))]
    let nalimov = 0u32;
    format!("\"gaiatb\": {gaiatb}, \"syzygy_pieces\": {syzygy}, \"nalimov_pieces\": {nalimov}")
}


/// Run a deterministic single-threaded search and export the requested data.
pub fn run(opts: &DumpOptions) {
    let fen = if opts.fen == "startpos" {
        STARTPOS
    } else {
        &opts.fen
    };
    let pos = match replay(fen, &opts.moves, opts.chess960) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    // Node cap takes precedence over depth (a nodes-limited search still
    // iterates depths; a depth-limited one has no node bound).
    let limits = if opts.nodes > 0 {
        SearchLimits::nodes(opts.nodes)
    } else {
        SearchLimits::depth(opts.depth)
    };

    let shared = SharedState::new(opts.tt_mb, 1);
    // id=1 suppresses UCI info output (same pattern as bench.rs / datagen.rs)
    let mut td = ThreadData::new(1);

    STOP.store(false, Ordering::Relaxed);
    td.prepare_search(&pos, &limits);
    // prepare_search gives infinite limits to id != 0 — override with real ones
    let game_ply = td.pos.game_ply();
    td.tm = TimeManager::new(&limits, game_ply, &mut td.original_time_adjust);

    #[cfg(feature = "tree")]
    if !opts.tree_out.is_empty() {
        td.tree = Some(Box::new(crate::tree::TreeRec::new(
            opts.max_mb * 1024 * 1024,
            opts.min_record_depth,
            opts.qs_moves,
        )));
    }
    #[cfg(not(feature = "tree"))]
    if !opts.tree_out.is_empty() {
        eprintln!("--tree requires a build with --features tree");
        std::process::exit(1);
    }

    let start = std::time::Instant::now();
    search::search(&mut td, &shared);
    let elapsed = start.elapsed().as_millis();

    // Context line on stdout: what was searched and what was found
    println!(
        "bestmove {} score {} depth {} nodes {} time_ms {}",
        td.best_move.to_uci_960(pos.chess960),
        td.best_score,
        td.completed_depth,
        td.nodes,
        elapsed
    );

    #[cfg(feature = "stats")]
    {
        eprint!("{}", td.stats.emit_text());
        if !opts.stats_out.is_empty() {
            let json = td.stats.emit_json();
            if opts.stats_out == "-" {
                println!("{json}");
            } else if let Err(e) = std::fs::write(&opts.stats_out, json) {
                eprintln!("Failed to write {}: {e}", opts.stats_out);
                std::process::exit(1);
            }
        }
    }
    #[cfg(not(feature = "stats"))]
    if !opts.stats_out.is_empty() {
        eprintln!("--stats requires a build with --features stats");
        std::process::exit(1);
    }

    #[cfg(feature = "tree")]
    if !opts.tree_out.is_empty() {
        let tree = td.tree.take().expect("tree recorder installed above");
        let meta = format!(
            "{{\"engine\": \"gaiachess {}\", \"commit\": \"{}\", \"depth\": {}, \"nodes\": {}, \
             \"searched_nodes\": {}, \"tt_mb\": {}, \"chess960\": {}, {}}}",
            env!("CARGO_PKG_VERSION"),
            option_env!("GIT_HASH").unwrap_or("unknown"),
            opts.depth,
            opts.nodes,
            td.nodes,
            opts.tt_mb,
            opts.chess960,
            tables_json(),
        );
        match tree.write_file(&opts.tree_out, fen, &opts.moves, &meta) {
            Ok(summary) => eprintln!("{summary}"),
            Err(e) => {
                eprintln!("Failed to write {}: {e}", opts.tree_out);
                std::process::exit(1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A game longer than the 1024 saved states is replayed and searched the way
    /// the dump does it: without trimming, the replay overflowed the history.
    #[test]
    fn a_game_of_more_than_a_thousand_plies_is_replayed_and_searched() {
        let game = crate::position::tests::long_game(14, 24);
        assert!(game.len() > 1100);
        let pos = replay("startpos", &game, false).expect("legal game");
        assert_eq!(pos.fullmove_number as usize, 1 + game.len() / 2, "every move was played");
        assert!(pos.ply < 512, "history still {} states deep", pos.ply);

        let _guard = crate::skill::lock_level();
        let limits = SearchLimits::depth(4);
        let shared = SharedState::new(1, 1);
        let mut td = ThreadData::new(1);
        STOP.store(false, Ordering::Relaxed);
        td.prepare_search(&pos, &limits);
        let game_ply = td.pos.game_ply();
        td.tm = TimeManager::new(&limits, game_ply, &mut td.original_time_adjust);
        search::search(&mut td, &shared);
        assert!(td.best_move.is_ok());
    }

    /// In Chess960 mode a castling move is written king-takes-rook, as an
    /// interface sends it: `g1h1` with the king already on g1 is the short castle.
    #[test]
    fn a_chess960_castle_is_replayed_in_king_takes_rook_notation() {
        let fen = "bbqnn1kr/pppppppp/8/8/8/8/PPPPPPPP/BBQNN1KR w Hh - 0 1";
        let pos = replay(fen, &[], true).expect("legal FEN");
        let m = crate::uci::parse_uci_move(&pos, "g1h1").expect("castling is legal");
        assert_eq!(m.move_type(), crate::types::MT_CASTLING);
        let pos = replay(fen, &["g1h1".to_string()], true).expect("legal castle");
        assert_eq!(pos.piece_on(crate::types::Square::G1).piece_type(), crate::types::PieceType::King);
        assert_eq!(pos.piece_on(crate::types::Square::F1).piece_type(), crate::types::PieceType::Rook);
        // Without Chess960 mode the same string names no move.
        match replay(fen, &["g1h1".to_string()], false) {
            Err(err) => assert!(err.starts_with("Illegal move"), "{err}"),
            Ok(_) => panic!("g1h1 is not a standard-notation move"),
        }
    }

    /// King on f1, rook on h1: `f1g1` is the king step and `f1h1` the castle,
    /// with no ambiguity between the two in Chess960 mode.
    #[test]
    fn a_chess960_king_step_and_castle_are_told_apart() {
        let fen = "bbqnnk1r/pppppppp/8/8/8/8/PPPPPPPP/BBQNNK1R w Hh - 0 1";
        let pos = replay(fen, &[], true).expect("legal FEN");
        let step = crate::uci::parse_uci_move(&pos, "f1g1").expect("king step");
        assert_ne!(step.move_type(), crate::types::MT_CASTLING);
        let castle = crate::uci::parse_uci_move(&pos, "f1h1").expect("castle");
        assert_eq!(castle.move_type(), crate::types::MT_CASTLING);
        assert_eq!(castle.to_uci_960(true), "f1h1");
    }
}
