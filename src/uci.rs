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

/// `Skill Seed`, which picks *which* weakened opponent a level is.
///
/// A level decides how badly the engine plays; the seed decides which moves it happens
/// to overlook and which way it misjudges each position. Zero means the built-in one, so
/// that an engine nobody configures still plays the same game twice. Calibration runs set
/// it to spread a level over many different opponents; over the board, variety comes from
/// the opening book instead.
static SKILL_SEED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn skill_seed() -> u64 {
    match SKILL_SEED.load(Ordering::Relaxed) {
        0 => 0x9E37_79B9_7F4A_7C15,
        chosen => crate::skill::mix(chosen),
    }
}

/// Starts the stdin reader and hands back the channel its lines arrive on.
///
/// Separate from the loop below because the caller may need to listen before it knows
/// whether it is running as an engine at all: whoever reads stdin first must be the
/// only one to read it, or the bytes one reader has buffered are lost to the other.
///
/// `stop` and `quit` set the global STOP flag here, in the reader, so a search stops
/// even though the main thread is blocked in start_search().
pub fn spawn_stdin_reader() -> mpsc::Receiver<String> {
    let (tx, rx) = mpsc::channel::<String>();
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
    rx
}

/// Main UCI loop. Blocks until `quit`, or until stdin ends.
///
/// `first` is a line already taken off `rx` while deciding whether to open the
/// interface instead. It is handled before anything else, so the order the caller sent
/// its commands in survives. Putting it back on the channel would not do: it would land
/// behind whatever arrived while the decision was being made.
/// A UCI conversation, driven one line at a time.
///
/// Split out from the read loop because the caller is not always a loop: the browser
/// build is handed its commands by its host and must return between them, and cannot
/// block on a channel that only another thread could fill.
pub struct UciSession {
    pos: Position,
    pool: ThreadPool,
    ponder_enabled: bool,
}

impl UciSession {
    pub fn new() -> UciSession {
        // Initialize GaiaTB (embedded 3+4 piece DTM, decompresses ~147 MB at startup)
        #[cfg(feature = "gaiatb")]
        {
            if crate::dtm::init() {
                outerr!("info string GaiaTB loaded (3+4 piece DTM tablebases)");
            } else {
                outerr!("info string GaiaTB: failed to load embedded tablebases");
            }
        }

        UciSession {
            pos: Position::from_fen(STARTPOS).expect("startpos"),
            pool: ThreadPool::new(1, 16),
            ponder_enabled: true,
        }
    }

    /// Handles one line. Returns `false` once the engine has been told to quit.
    pub fn command(&mut self, line: &str) -> bool {
        let tokens: Vec<&str> = line.split_whitespace().collect();

        match tokens.as_slice() {
            ["uci"] => {
                out!("id name Gaia {}", env!("CARGO_PKG_VERSION"));
                out!("id author Jean-Francois Romang");
                out!("option name Hash type spin default 16 min 1 max 1048576");
                out!("option name Threads type spin default 1 min 1 max 256");
                out!("option name EvalFile type string default <internal>");
                out!("option name Ponder type check default true");
                out!("option name OwnBook type check default true");
                out!("option name MultiPV type spin default 1 min 1 max 256");
                out!("option name Move Overhead type spin default 100 min 0 max 5000");
                out!(
                    "option name Skill Level type spin default {} min {} max {}",
                    crate::skill::FULL_STRENGTH,
                    crate::skill::MIN_LEVEL,
                    crate::skill::FULL_STRENGTH
                );
                out!("option name Skill Seed type spin default 0 min 0 max 2147483647");
                #[cfg(feature = "syzygy")]
                out!("option name SyzygyPath type string default <empty>");
                #[cfg(feature = "nalimov")]
                out!("option name NalimovPath type string default <empty>");
                #[cfg(feature = "online-tb")]
                out!("option name OnlineTB type check default false");
                #[cfg(feature = "spsa")]
                crate::tune::print_uci_options();
                out!("uciok");
            }

            ["isready"] => out!("readyok"),

            ["ucinewgame"] => {
                self.pool.clear();
                #[cfg(feature = "online-tb")]
                crate::online_tb::clear();
                self.pos = Position::from_fen(STARTPOS).expect("startpos");
            }

            ["position", rest @ ..] => parse_position(rest, &mut self.pos),

            ["go", rest @ ..] => {
                if rest.contains(&"ponder") {
                    PONDER.store(true, Ordering::Relaxed);
                }
                #[cfg(feature = "online-tb")]
                if crate::online_tb::is_enabled() {
                    crate::online_tb::ensure_worker_running();
                }
                let limits = parse_go(rest, &self.pos);
                let (best, ponder_move) = self.pool.start_search(&self.pos, limits);
                if !best.is_ok() {
                    // Checkmate or stalemate on the board: there is no move to name, and
                    // "(none)" is what the protocol says to answer.
                    out!("bestmove (none)");
                } else if self.ponder_enabled && let Some(pm) = ponder_move {
                    out!("bestmove {} ponder {}", best.to_uci(), pm.to_uci());
                } else {
                    out!("bestmove {}", best.to_uci());
                }
                PONDER.store(false, Ordering::Relaxed);
                // Search statistics summary on stderr (invisible to GUIs)
                #[cfg(feature = "stats")]
                eprint!("{}", self.pool.aggregated_stats().emit_text());
                // Throwaway debug slots (silent when unused)
                crate::stats::dbg::print();
                // Prefetch online TB: PV positions first, then opponent replies
                #[cfg(feature = "online-tb")]
                if crate::online_tb::is_enabled() {
                    let idx = self.pool.last_best_idx;
                    let (pv, pv_len) = if !self.pool.threads[idx].root_moves.is_empty() {
                        (&self.pool.threads[idx].root_moves[0].pv[..],
                         self.pool.threads[idx].root_moves[0].pv_len)
                    } else {
                        (&[][..], 0)
                    };
                    crate::online_tb::start_prefetch(
                        self.pos.clone(), best, ponder_move, pv, pv_len,
                    );
                }
            }

            ["stop"] => {
                STOP.store(true, Ordering::Relaxed);
            }

            ["setoption", rest @ ..] => {
                parse_setoption(rest, &mut self.pool, &mut self.ponder_enabled);
            }

            ["d"] => print_board(&self.pos),

            ["eval"] => print_eval(&self.pos),

            ["quit"] => return false,

            _ => {}
        }

        true
    }
}

impl Default for UciSession {
    fn default() -> Self {
        Self::new()
    }
}

/// Reads commands off `rx` until the engine is told to quit.
pub fn uci_loop(rx: mpsc::Receiver<String>, first: Option<String>) {
    let mut session = UciSession::new();
    let mut pending = first;
    while let Some(line) = pending.take().or_else(|| rx.recv().ok()) {
        if !session.command(&line) {
            break;
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
                    outerr!("info string Invalid FEN: {e}");
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
                    outerr!("info string Illegal move: {move_str}");
                    break;
                }
            }
        }
    }
}

/// Parse a UCI move string (e.g., "e2e4", "e7e8q") by matching against legal moves.
pub(crate) fn parse_uci_move(pos: &Position, s: &str) -> Option<Move> {
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
        "ownbook" => {
            crate::book::set_enabled(value.eq_ignore_ascii_case("true"));
        }
        "multipv" => {
            if let Ok(n) = value.parse::<usize>() {
                pool.multi_pv = n.clamp(1, 256);
            }
        }
        "skill level" => {
            if let Ok(level) = value.parse::<i32>() {
                // One seed for the whole session, so a weakened opponent keeps the same
                // character from move to move.
                crate::skill::set(level, skill_seed());
            }
        }
        "skill seed" => {
            if let Ok(seed) = value.parse::<i64>() {
                SKILL_SEED.store(seed as u64, Ordering::Relaxed);
                // Re-apply so a seed set after the level takes effect either way round.
                crate::skill::set(crate::skill::level(), skill_seed());
            }
        }
        "move overhead" => {
            if let Ok(v) = value.parse::<i32>() {
                crate::tune::set_move_overhead(v);
            }
        }
        "evalfile" => {
            if value == "<internal>" {
                outerr!("info string Using default evaluation");
            } else {
                match crate::nnue::network::load_from_file(&value) {
                    Ok(()) => {
                        outerr!("info string Loaded NNUE network from {value}");
                        pool.clear();
                    }
                    Err(e) => {
                        outerr!("info string Failed to load network: {e}");
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
                    outerr!("info string Failed to load Syzygy tablebases from {value}");
                }
            }
            #[cfg(not(feature = "syzygy"))]
            outerr!("info string SyzygyPath requires --features syzygy");
        }
        "nalimovpath" => {
            #[cfg(feature = "nalimov")]
            {
                if value == "<empty>" {
                    // OnceLock doesn't support reset — ignore
                } else if !crate::nalimov::init(&value) {
                    outerr!("info string Failed to load Nalimov tablebases from {value}");
                } else {
                    outerr!("info string Nalimov tablebases loaded ({}-piece) from {value}",
                               crate::nalimov::max_pieces());
                }
            }
            #[cfg(not(feature = "nalimov"))]
            outerr!("info string NalimovPath requires --features nalimov");
        }
        "onlinetb" => {
            #[cfg(feature = "online-tb")]
            {
                let enabled = value.eq_ignore_ascii_case("true");
                crate::online_tb::set_enabled(enabled);
                outerr!("info string OnlineTB {}",
                    if enabled { "enabled" } else { "disabled" });
            }
            #[cfg(not(feature = "online-tb"))]
            outerr!("info string OnlineTB requires --features online-tb");
        }
        _ => {
            #[cfg(feature = "spsa")]
            {
                if let Ok(v) = value.parse::<i32>()
                    && crate::tune::set_param(&name, v)
                {
                    outerr!("info string set {} = {}", name, v);
                }
            }
        }
    }
}

/// Print the board (debug command `d`).
fn print_board(pos: &Position) {
    out!();
    for rank in (0..8).rev() {
        // Built up and said once: a line at a time is the only shape a browser host can
        // be handed, and it reads no worse here.
        let mut row = format!("  {} ", rank + 1);
        for file in 0..8 {
            let sq = crate::types::Square::new(file, rank);
            let pc = pos.board[sq.index()];
            row.push(' ');
            row.push(pc.to_char());
        }
        out!("{row}");
    }
    out!("     a b c d e f g h");
    out!();
    out!("  Fen: {}", pos.to_fen());
    out!("  Key: {:016x}", pos.key);
    out!("  Checkers: {:064b}", pos.checkers);
}

/// Print evaluation breakdown (debug command `eval`).
fn print_eval(pos: &Position) {
    let pesto_score = eval::evaluate(pos);
    out!("PeSTO eval: {} cp (white POV)", pesto_score);

    if nnue::network::has_network() {
        let mut net = nnue::Network::new();
        net.refresh(pos);
        if pos.checkers == 0 {
            let nnue_score = net.evaluate(pos);
            out!("NNUE eval:  {} cp (STM POV)", nnue_score);
        } else {
            out!("NNUE eval:  N/A (in check)");
        }
        let pieces = pos.occupied().count_ones();
        let bucket = nnue::OUTPUT_BUCKET_MAP[pieces.min(32) as usize];
        out!("Output bucket: {} (pieces: {})", bucket, pieces);
    } else {
        out!("NNUE: no network loaded");
    }
}
