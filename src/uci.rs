//! [UCI protocol](https://backscattering.de/chess/uci/) implementation.
//!
//! Reads commands from stdin on a separate thread (via `mpsc`), dispatches
//! them on the main thread. Search uses Lazy SMP: the main thread blocks
//! during search while the stdin reader thread can still receive `stop`.

use std::io::BufRead;
use std::sync::atomic::Ordering;
use std::sync::mpsc;


use crate::eval;
use crate::movegen;
use crate::nnue;
use crate::position::Position;
use crate::threads::{ThreadPool, PONDER, STOP};
use crate::timeman::SearchLimits;
use crate::types::{ArrayBuf, Color, Move, MAX_MOVES};

const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/// Main UCI loop. Blocks forever until `quit` is received.
pub fn uci_loop() {
    let (tx, rx) = mpsc::channel::<String>();

    // Spawn stdin reader thread.
    // "stop" and "quit" set the global STOP flag immediately so the search
    // stops even though the main thread is blocked in start_search().
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        for line in stdin.lock().lines() {
            match line {
                Ok(l) => {
                    let trimmed = l.trim();
                    if trimmed == "stop" || trimmed == "quit" {
                        STOP.store(true, Ordering::Relaxed);
                    } else if trimmed == "ponderhit" {
                        PONDER.store(false, Ordering::Relaxed);
                    }
                    if tx.send(l).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });

    let mut pos = Position::from_fen(STARTPOS).expect("startpos");
    let mut pool = ThreadPool::new(1, 16);
    let mut ponder_enabled = true;

    // Initialize GaiaTB (embedded 3+4 piece DTM, decompresses ~147 MB at startup)
    #[cfg(feature = "gaiatb")]
    {
        if crate::dtm::init() {
            eprintln!("info string GaiaTB loaded (3+4 piece DTM tablebases)");
        } else {
            eprintln!("info string GaiaTB: failed to load embedded tablebases");
        }
    }

    loop {
        let line = match rx.recv() {
            Ok(l) => l,
            Err(_) => break,
        };
        let tokens: Vec<&str> = line.split_whitespace().collect();

        match tokens.as_slice() {
            ["uci"] => {
                println!("id name {} {}", "Gaia", env!("CARGO_PKG_VERSION"));
                println!("id author {}", "Jean-Francois Romang");
                println!("option name Hash type spin default 16 min 1 max 1048576");
                println!("option name Threads type spin default 1 min 1 max 256");
                println!("option name EvalFile type string default <internal>");
                println!("option name Ponder type check default true");
                println!("option name MultiPV type spin default 1 min 1 max 256");
                println!("option name Move Overhead type spin default 100 min 0 max 5000");
                #[cfg(feature = "syzygy")]
                println!("option name SyzygyPath type string default <empty>");
                #[cfg(feature = "nalimov")]
                println!("option name NalimovPath type string default <empty>");
                #[cfg(feature = "online-tb")]
                println!("option name OnlineTB type check default false");
                #[cfg(feature = "spsa")]
                crate::tune::print_uci_options();
                println!("uciok");
            }

            ["isready"] => println!("readyok"),

            ["ucinewgame"] => {
                pool.clear();
                #[cfg(feature = "online-tb")]
                crate::online_tb::clear();
                pos = Position::from_fen(STARTPOS).expect("startpos");
            }

            ["position", rest @ ..] => parse_position(rest, &mut pos),

            ["go", rest @ ..] => {
                if rest.contains(&"ponder") {
                    PONDER.store(true, Ordering::Relaxed);
                }
                #[cfg(feature = "online-tb")]
                if crate::online_tb::is_enabled() {
                    crate::online_tb::ensure_worker_running();
                }
                let limits = parse_go(rest, &pos);
                let (best, ponder_move) = pool.start_search(&pos, limits);
                if ponder_enabled {
                    if let Some(pm) = ponder_move {
                        println!("bestmove {} ponder {}", best.to_uci(), pm.to_uci());
                    } else {
                        println!("bestmove {}", best.to_uci());
                    }
                } else {
                    println!("bestmove {}", best.to_uci());
                }
                PONDER.store(false, Ordering::Relaxed);
                // Prefetch online TB: PV positions first, then opponent replies
                #[cfg(feature = "online-tb")]
                if crate::online_tb::is_enabled() {
                    let idx = pool.last_best_idx;
                    let (pv, pv_len) = if !pool.threads[idx].root_moves.is_empty() {
                        (&pool.threads[idx].root_moves[0].pv[..],
                         pool.threads[idx].root_moves[0].pv_len)
                    } else {
                        (&[][..], 0)
                    };
                    crate::online_tb::start_prefetch(
                        pos.clone(), best, ponder_move, pv, pv_len,
                    );
                }
            }

            ["stop"] => {
                STOP.store(true, Ordering::Relaxed);
            }

            ["setoption", rest @ ..] => {
                parse_setoption(rest, &mut pool, &mut ponder_enabled);
            }

            ["d"] => print_board(&pos),

            ["eval"] => print_eval(&pos),

            ["quit"] => break,

            _ => {}
        }
    }
}

/// Parse `position [startpos|fen ...] [moves ...]`.
fn parse_position(tokens: &[&str], pos: &mut Position) {
    let tokens = match tokens {
        ["startpos", rest @ ..] => {
            *pos = Position::from_fen(STARTPOS).expect("startpos");
            rest
        }
        ["fen", rest @ ..] => {
            let fen_end = rest.iter().position(|&t| t == "moves").unwrap_or(rest.len());
            let fen = rest[..fen_end].join(" ");
            match Position::from_fen(&fen) {
                Ok(p) => *pos = p,
                Err(e) => {
                    eprintln!("info string Invalid FEN: {e}");
                    return;
                }
            }
            &rest[fen_end..]
        }
        _ => return,
    };

    if let ["moves", rest @ ..] = tokens {
        for move_str in rest {
            match parse_uci_move(pos, move_str) {
                Some(m) => pos.make_move(m),
                None => {
                    eprintln!("info string Illegal move: {move_str}");
                    break;
                }
            }
        }
    }
}

/// Parse a UCI move string (e.g., "e2e4", "e7e8q") by matching against legal moves.
fn parse_uci_move(pos: &Position, s: &str) -> Option<Move> {
    let mut buf: ArrayBuf<Move, MAX_MOVES> = ArrayBuf::new();
    let count = movegen::generate_legal_moves(pos, &mut buf);
    for i in 0..count {
        if buf[i].to_uci() == s {
            return Some(buf[i]);
        }
    }
    None
}

/// Parse the next token as a value of type `T`.
fn next_val<T: std::str::FromStr>(iter: &mut std::slice::Iter<'_, &str>) -> Option<T> {
    iter.next().and_then(|s| s.parse().ok())
}

/// Parse `go` command options into SearchLimits.
fn parse_go(tokens: &[&str], pos: &Position) -> SearchLimits {
    let stm = pos.side_to_move as usize;
    let mut depth = None;
    let mut nodes = None;
    let mut movetime = None;
    let mut clocks: [Option<u64>; 2] = [None, None];
    let mut incs = [0u64; 2];
    let mut movestogo = None;

    let mut iter = tokens.iter();
    while let Some(&tok) = iter.next() {
        match tok {
            "infinite" => return SearchLimits::Infinite,
            "ponder" => {}
            "depth" => depth = next_val(&mut iter),
            "nodes" => nodes = next_val(&mut iter),
            "movetime" => movetime = next_val(&mut iter),
            "wtime" => clocks[Color::White as usize] = next_val(&mut iter),
            "btime" => clocks[Color::Black as usize] = next_val(&mut iter),
            "winc" => incs[Color::White as usize] = next_val(&mut iter).unwrap_or(0),
            "binc" => incs[Color::Black as usize] = next_val(&mut iter).unwrap_or(0),
            "movestogo" => movestogo = next_val(&mut iter),
            _ => {}
        }
    }

    if let Some(d) = depth {
        return SearchLimits::Depth(d);
    }
    if let Some(n) = nodes {
        return SearchLimits::Nodes(n);
    }
    if let Some(ms) = movetime {
        return SearchLimits::MoveTime(ms);
    }

    let time = clocks[stm].unwrap_or(0);
    if time > 0 {
        return SearchLimits::Clock {
            time,
            inc: incs[stm],
            movestogo,
        };
    }

    SearchLimits::Depth(6)
}

/// Parse `setoption name <N...> value <V...>`.
/// Supports multi-word option names (e.g. "Move Overhead").
fn parse_setoption(tokens: &[&str], pool: &mut ThreadPool, ponder_enabled: &mut bool) {
    if tokens.first() != Some(&"name") || tokens.len() < 4 {
        return;
    }
    let value_pos = match tokens[1..].iter().position(|&t| t == "value") {
        Some(p) => p + 1,
        None => return,
    };
    if value_pos < 2 || value_pos + 1 >= tokens.len() {
        return;
    }
    let name = tokens[1..value_pos].join(" ");
    let value = tokens[value_pos + 1..].join(" ");

    match &*name.to_ascii_lowercase() {
        "hash" => {
            if let Ok(mb) = value.parse::<usize>() {
                pool.resize_hash(mb.clamp(1, 1_048_576));
            }
        }
        "threads" => {
            if let Ok(n) = value.parse::<usize>() {
                pool.resize_threads(n.clamp(1, 256));
            }
        }
        "ponder" => {
            *ponder_enabled = value.eq_ignore_ascii_case("true");
        }
        "multipv" => {
            if let Ok(n) = value.parse::<usize>() {
                pool.multi_pv = n.clamp(1, 256);
            }
        }
        "move overhead" => {
            if let Ok(v) = value.parse::<i32>() {
                crate::tune::set_move_overhead(v);
            }
        }
        "evalfile" => {
            if value == "<internal>" {
                eprintln!("info string Using default evaluation");
            } else {
                match crate::nnue::network::load_from_file(&value) {
                    Ok(()) => {
                        eprintln!("info string Loaded NNUE network from {value}");
                        pool.clear();
                    }
                    Err(e) => {
                        eprintln!("info string Failed to load network: {e}");
                    }
                }
            }
        }
        "syzygypath" => {
            #[cfg(feature = "syzygy")]
            {
                if value == "<empty>" {
                    crate::tb::free();
                } else if !crate::tb::init(&value) {
                    eprintln!("info string Failed to load Syzygy tablebases from {value}");
                }
            }
            #[cfg(not(feature = "syzygy"))]
            eprintln!("info string SyzygyPath requires --features syzygy");
        }
        "nalimovpath" => {
            #[cfg(feature = "nalimov")]
            {
                if value == "<empty>" {
                    // OnceLock doesn't support reset — ignore
                } else if !crate::nalimov::init(&value) {
                    eprintln!("info string Failed to load Nalimov tablebases from {value}");
                } else {
                    eprintln!("info string Nalimov tablebases loaded ({}-piece) from {value}",
                               crate::nalimov::max_pieces());
                }
            }
            #[cfg(not(feature = "nalimov"))]
            eprintln!("info string NalimovPath requires --features nalimov");
        }
        "onlinetb" => {
            #[cfg(feature = "online-tb")]
            {
                let enabled = value.eq_ignore_ascii_case("true");
                crate::online_tb::set_enabled(enabled);
                eprintln!("info string OnlineTB {}",
                    if enabled { "enabled" } else { "disabled" });
            }
            #[cfg(not(feature = "online-tb"))]
            eprintln!("info string OnlineTB requires --features online-tb");
        }
        _ => {
            #[cfg(feature = "spsa")]
            {
                if let Ok(v) = value.parse::<i32>() {
                    if crate::tune::set_param(&name, v) {
                        eprintln!("info string set {} = {}", name, v);
                    }
                }
            }
        }
    }
}

/// Print the board (debug command `d`).
fn print_board(pos: &Position) {
    println!();
    for rank in (0..8).rev() {
        print!("  {} ", rank + 1);
        for file in 0..8 {
            let sq = crate::types::Square::new(file, rank);
            let pc = pos.board[sq.index()];
            print!(" {}", pc.to_char());
        }
        println!();
    }
    println!("     a b c d e f g h");
    println!();
    println!("  Fen: {}", pos.to_fen());
    println!("  Key: {:016x}", pos.key);
    println!("  Checkers: {:064b}", pos.checkers);
}

/// Print evaluation breakdown (debug command `eval`).
fn print_eval(pos: &Position) {
    let pesto_score = eval::evaluate(pos);
    println!("PeSTO eval: {} cp (white POV)", pesto_score);

    if nnue::network::has_network() {
        let mut net = nnue::Network::new();
        net.refresh(pos);
        if pos.checkers == 0 {
            let nnue_score = net.evaluate(pos);
            println!("NNUE eval:  {} cp (STM POV)", nnue_score);
        } else {
            println!("NNUE eval:  N/A (in check)");
        }
        let pieces = pos.occupied().count_ones();
        let bucket = nnue::OUTPUT_BUCKET_MAP[pieces.min(32) as usize];
        println!("Output bucket: {} (pieces: {})", bucket, pieces);
    } else {
        println!("NNUE: no network loaded");
    }
}
