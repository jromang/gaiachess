//! Offline analysis runner for the `dump` subcommand (features `stats`/`tree`).
//!
//! Runs a deterministic single-threaded fixed-depth search on one position and
//! exports search statistics (JSON) and, with the `tree` feature, the full
//! search tree in the GTREE binary format for offline analysis by
//! `tools/treescope/`.

use std::sync::atomic::Ordering;

use crate::position::Position;
use crate::search;
use crate::threads::{SharedState, ThreadData, STOP};
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
}

/// Run a deterministic single-threaded search and export the requested data.
pub fn run(opts: &DumpOptions) {
    let fen = if opts.fen == "startpos" { STARTPOS } else { &opts.fen };
    let mut pos = match Position::from_fen(fen) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Invalid FEN: {e}");
            std::process::exit(1);
        }
    };
    for ms in &opts.moves {
        match crate::uci::parse_uci_move(&pos, ms) {
            Some(m) => pos.make_move(m),
            None => {
                eprintln!("Illegal move in --moves: {ms}");
                std::process::exit(1);
            }
        }
    }

    // Node cap takes precedence over depth (a nodes-limited search still
    // iterates depths; a depth-limited one has no node bound).
    let limits = if opts.nodes > 0 {
        SearchLimits::Nodes(opts.nodes)
    } else {
        SearchLimits::Depth(opts.depth)
    };

    let shared = SharedState::new(opts.tt_mb);
    // id=1 suppresses UCI info output (same pattern as bench.rs / datagen.rs)
    let mut td = ThreadData::new(1);

    STOP.store(false, Ordering::Relaxed);
    td.prepare_search(&pos, &limits);
    // prepare_search gives infinite limits to id != 0 — override with real ones
    td.tm = TimeManager::new(&limits);

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
        td.best_move.to_uci(),
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
            "{{\"engine\": \"gaiachess {}\", \"commit\": \"{}\", \"depth\": {}, \"tt_mb\": {}}}",
            env!("CARGO_PKG_VERSION"),
            option_env!("GIT_HASH").unwrap_or("unknown"),
            opts.depth,
            opts.tt_mb,
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
