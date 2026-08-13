//! Search-based benchmark: 27 positions from gaia3.0.
//! Default: depth 13 (deterministic node count). `bench <depth>` overrides.

use std::sync::atomic::Ordering;
use std::time::Instant;

use indicatif::{ProgressBar, ProgressStyle};

use crate::position::Position;
use crate::search;
use crate::threads::{SharedState, ThreadData, BENCH_MODE, BENCH_NODES, STOP};
use crate::timeman::SearchLimits;
/// 27 unique positions from gaia3.0's profile() function (gaia3.0/src/util.c:328).
pub(crate) const POSITIONS: [&str; 27] = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "1k1r4/pp1b1R2/3q2pp/4p3/2B5/4Q3/PPP2B2/2K5 b - - 0 1",
    "3r1k2/4npp1/1ppr3p/p6P/P2PPPP1/1NR5/5K2/2R5 w - - 0 1",
    "2q1rr1k/3bbnnp/p2p1pp1/2pPp3/PpP1P1P1/1P2BNNP/2BQ1PRK/7R b - - 0 1",
    "rnbqkb1r/p3pppp/1p6/2ppP3/3N4/2P5/PPP1QPPP/R1B1KB1R w KQkq - 0 1",
    "r1b2rk1/2q1b1pp/p2ppn2/1p6/3QP3/1BN1B3/PPP3PP/R4RK1 w - - 0 1",
    "2r3k1/pppR1pp1/4p3/4P1P1/5P2/1P4K1/P1P5/8 w - - 0 1",
    "1nk1r1r1/pp2n1pp/4p3/q2pPp1N/b1pP1P2/B1P2R2/2P1B1PP/R2Q2K1 w - - 0 1",
    "4b3/p3kp2/6p1/3pP2p/2pP1P2/4K1P1/P3N2P/8 w - - 0 1",
    "2kr1bnr/pbpq4/2n1pp2/3p3p/3P1P1B/2N2N1Q/PPP3PP/2KR1B1R w - - 0 1",
    "3rr1k1/pp3pp1/1qn2np1/8/3p4/PP1R1P2/2P1NQPP/R1B3K1 b - - 0 1",
    "2r1nrk1/p2q1ppp/bp1p4/n1pPp3/P1P1P3/2PBB1N1/4QPPP/R4RK1 w - - 0 1",
    "r3r1k1/ppqb1ppp/8/4p1NQ/8/2P5/PP3PPP/R3R1K1 b - - 0 1",
    "r2q1rk1/4bppp/p2p4/2pP4/3pP3/3Q4/PP1B1PPP/R3R1K1 w - - 0 1",
    "rnb2r1k/pp2p2p/2pp2p1/q2P1p2/8/1Pb2NP1/PB2PPBP/R2Q1RK1 w - - 0 1",
    "2r3k1/1p2q1pp/2b1pr2/p1pp4/6Q1/1P1PP1R1/P1PN2PP/5RK1 w - - 0 1",
    "r1bqkb1r/4npp1/p1p4p/1p1pP1B1/8/1B6/PPPN1PPP/R2Q1RK1 w kq - 0 1",
    "r2q1rk1/1ppnbppp/p2p1nb1/3Pp3/2P1P1P1/2N2N1P/PPB1QP2/R1B2RK1 b - - 0 1",
    "r1bq1rk1/pp2ppbp/2np2p1/2n5/P3PP2/N1P2N2/1PB3PP/R1B1QRK1 b - - 0 1",
    "3rr3/2pq2pk/p2p1pnp/8/2QBPP2/1P6/P5PP/4RRK1 b - - 0 1",
    "r4k2/pb2bp1r/1p1qp2p/3pNp2/3P1P2/2N3P1/PPP1Q2P/2KRR3 w - - 0 1",
    "3rn2k/ppb2rpp/2ppqp2/5N2/2P1P3/1P5Q/PB3PPP/3RR1K1 w - - 0 1",
    "2r2rk1/1bqnbpp1/1p1ppn1p/pP6/N1P1P3/P2B1N1P/1B2QPP1/R2R2K1 b - - 0 1",
    "r1bqk2r/pp2bppp/2p5/3pP3/P2Q1P2/2N1B3/1PP3PP/R4RK1 b kq - 0 1",
    "r2qnrnk/p2b2b1/1p1p2pp/2pPpp2/1PP1P3/PRNBB3/3QNPPP/5RK1 w - - 0 1",
    "8/k7/3p4/p2P1p2/P2P1P2/8/8/K7 w - - 0 1",
    "8/k7/rnn5/8/8/8/5RBB/K7 w - - 0 1",
];

/// Default bench depth.
const BENCH_DEPTH: i32 = 16;

/// Hash table size for bench (MB).
const BENCH_HASH_MB: usize = 64;

/// Run the search benchmark. `depth_override` from CLI arg, or None for default.
pub fn run(depth_override: Option<i32>) {
    let depth = depth_override.unwrap_or(BENCH_DEPTH);
    let total = POSITIONS.len() as u64;
    let shared = SharedState::new(BENCH_HASH_MB);
    // id=1 suppresses UCI info output (same pattern as datagen.rs)
    let mut td = ThreadData::new(1);

    let mut total_nodes: u64 = 0;
    let mut total_time_ms: u64 = 0;

    BENCH_MODE.store(true, Ordering::Relaxed);

    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::with_template(
            "  {spinner:.green} [{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({msg})",
        )
        .unwrap()
        .progress_chars("##-"),
    );
    pb.set_message("searching...");

    for (i, fen) in POSITIONS.iter().enumerate() {
        let pos = Position::from_fen(fen).expect("invalid bench FEN");
        let limits = SearchLimits::Depth(depth);

        STOP.store(false, Ordering::Relaxed);
        BENCH_NODES.store(0, Ordering::Relaxed);

        // Prepare search — override tm since prepare_search gives infinite
        // limits to id!=0 (same pattern as datagen.rs:prepare_datagen_search)
        td.prepare_search(&pos, &limits);
        td.tm = crate::timeman::TimeManager::new(&limits);

        let start = Instant::now();

        // Run search (blocking, 1s)
        search::search(&mut td, &shared);

        let elapsed_ms = start.elapsed().as_millis() as u64;
        let nodes = td.nodes;
        total_nodes += nodes;
        total_time_ms += elapsed_ms;
        pb.set_position(i as u64 + 1);
    }

    BENCH_MODE.store(false, Ordering::Relaxed);

    // Final summary
    let avg_nps = if total_time_ms > 0 {
        total_nodes * 1000 / total_time_ms
    } else {
        0
    };

    pb.finish_with_message("done");

    println!(
        "\n  {} ({} nodes in {:.1}s, depth {})",
        format_nps(avg_nps),
        format_number(total_nodes),
        total_time_ms as f64 / 1000.0,
        depth,
    );
    // Deterministic line for regression checks (OpenBench-compatible format)
    println!("{} nodes {} nps", total_nodes, avg_nps);
}

/// Run the statistical benchmark: N runs with robust statistics.
/// First run is discarded as warmup. Reports median, trimmed mean, CI, outliers.
pub fn run_stats(depth_override: Option<i32>, num_runs: u32, verbose: bool) {
    use crate::bench_stats::{self, PositionStats};

    let num_runs = num_runs.max(3) as usize;
    let depth = depth_override.unwrap_or(BENCH_DEPTH);
    let total_runs = num_runs + 1; // 1 warmup + N measured
    let num_positions = POSITIONS.len();

    // Allocate once, clear between runs
    let mut shared = SharedState::new(BENCH_HASH_MB);
    let mut td = ThreadData::new(1);

    BENCH_MODE.store(true, Ordering::Relaxed);

    // Storage: per-position NPS across measured runs, plus overall NPS per run
    let mut nps_per_position: Vec<Vec<f64>> = vec![Vec::with_capacity(num_runs); num_positions];
    let mut nps_overall: Vec<f64> = Vec::with_capacity(num_runs);
    let mut last_run_nodes: u64 = 0;

    let pb = ProgressBar::new((total_runs * num_positions) as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "  {spinner:.green} [{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({msg})",
        )
        .unwrap()
        .progress_chars("##-"),
    );

    for run_idx in 0..total_runs {
        let is_warmup = run_idx == 0;
        if is_warmup {
            pb.set_message("warmup...");
        } else {
            pb.set_message(format!("run {}/{}", run_idx, num_runs));
        }

        // Clear TT and histories for independence between runs
        shared.tt.clear();
        td.clear_histories();

        let mut run_nodes: u64 = 0;
        let mut run_time_us: u64 = 0;

        for (pos_idx, fen) in POSITIONS.iter().enumerate() {
            let pos = Position::from_fen(fen).expect("invalid bench FEN");
            let limits = SearchLimits::Depth(depth);

            STOP.store(false, Ordering::Relaxed);
            BENCH_NODES.store(0, Ordering::Relaxed);

            td.prepare_search(&pos, &limits);
            td.tm = crate::timeman::TimeManager::new(&limits);

            let start = Instant::now();
            search::search(&mut td, &shared);
            let elapsed_us = start.elapsed().as_micros() as u64;

            let nodes = td.nodes;
            run_nodes += nodes;
            run_time_us += elapsed_us;

            if !is_warmup {
                let nps = if elapsed_us > 0 {
                    nodes as f64 * 1_000_000.0 / elapsed_us as f64
                } else {
                    0.0
                };
                nps_per_position[pos_idx].push(nps);
            }

            pb.set_position((run_idx * num_positions + pos_idx + 1) as u64);
        }

        if !is_warmup {
            let overall_nps = if run_time_us > 0 {
                run_nodes as f64 * 1_000_000.0 / run_time_us as f64
            } else {
                0.0
            };
            nps_overall.push(overall_nps);
            last_run_nodes = run_nodes;
        }
    }

    BENCH_MODE.store(false, Ordering::Relaxed);
    pb.finish_with_message("done");

    // Compute statistics
    let overall = bench_stats::compute_stats(&nps_overall);

    let per_position: Vec<PositionStats> = POSITIONS
        .iter()
        .enumerate()
        .map(|(i, fen)| {
            let s = bench_stats::compute_stats(&nps_per_position[i]);
            PositionStats {
                fen: fen.to_string(),
                median_nps: s.median,
                cv_pct: s.cv_pct,
            }
        })
        .collect();

    // Print report
    println!("\n  === Bench Statistics ({num_runs} runs, depth {depth}) ===\n");
    println!("  Median NPS:       {}", format_number(overall.median as u64));
    println!("  Trimmed Mean:     {}", format_number(overall.trimmed_mean as u64));
    println!(
        "  95% CI:           [{} -- {}]",
        format_number(overall.ci_lo as u64),
        format_number(overall.ci_hi as u64),
    );
    println!("  CV:               {:.1}%", overall.cv_pct);
    println!("  Outliers:         {} / {}", overall.outliers, overall.n);
    println!();

    if verbose {
        println!("  === Per-Position Breakdown ===\n");
        println!(
            "  {:>2} | {:<42} | {:>12} | {:>5}",
            "#", "FEN", "Median NPS", "CV%"
        );
        println!("  ---|{:-<42}--|{:-<12}--|{:-<5}--", "", "", "");
        for (i, pos) in per_position.iter().enumerate() {
            let fen_display = if pos.fen.len() > 42 {
                format!("{}...", &pos.fen[..39])
            } else {
                pos.fen.clone()
            };
            println!(
                "  {:>2} | {:<42} | {:>12} | {:>5.1}",
                i + 1,
                fen_display,
                format_number(pos.median_nps as u64),
                pos.cv_pct,
            );
        }
        println!();
    }

    // Deterministic line from last run (preserves grep-ability)
    println!("  {} nodes (last run)", format_number(last_run_nodes));
}

/// Format NPS with comma separators and " NPS" suffix.
fn format_nps(nps: u64) -> String {
    format!("{} NPS", format_number(nps))
}

/// Format a number with comma separators (e.g., 2345678 -> "2,345,678").
pub(crate) fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}
