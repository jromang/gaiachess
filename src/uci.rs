//! [UCI protocol](https://backscattering.de/chess/uci/) implementation.
//!
//! Reads commands from stdin on a separate thread (via `mpsc`), dispatches
//! them on the main thread. Search uses Lazy SMP: the main thread blocks
//! during search while the stdin reader thread can still receive `stop`.

use std::io::BufRead;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};
use std::sync::mpsc;

use crate::eval;
use crate::movegen;
use crate::nnue;
use crate::position::Position;
use crate::threads::{PONDER, STOP, ThreadPool};
use crate::timeman::SearchLimits;
use crate::types::{ArrayBuf, Color, MAX_MOVES, Move};

const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/// `Skill Seed`, which picks *which* weakened opponent a level is.
///
/// A level decides how badly the engine plays; the seed decides which moves it happens
/// to overlook and which way it misjudges each position. Zero means the built-in one, so
/// that an engine nobody configures still plays the same game twice. Calibration runs set
/// it to spread a level over many different opponents; over the board, variety comes from
/// the opening book instead.
static SKILL_SEED: AtomicU64 = AtomicU64::new(0);

/// UCI `UseSoftNodes`: a bare `go nodes N` is reinterpreted as a soft budget
/// (finish the iteration in flight) with a derived hard ceiling. Off by
/// default — a plain `go nodes` stays a hard limit. Match runners only send
/// `go nodes`, so this switch is what lets them exercise the soft behavior.
static USE_SOFT_NODES: AtomicBool = AtomicBool::new(false);

/// UCI `debug on|off`. Extra `info string` of received commands. Off by default.
static DEBUG: AtomicBool = AtomicBool::new(false);

/// `UCI_LimitStrength`: when true, `UCI_Elo` selects a skill rung.
static LIMIT_STRENGTH: AtomicBool = AtomicBool::new(false);

/// Last `UCI_Elo` value. Ignored while [`LIMIT_STRENGTH`] is false.
static UCI_ELO: AtomicI32 = AtomicI32::new(crate::skill::MAX_LIMITED_ELO);

/// `UCI_AnalyseMode`. Stored for GUIs that set it; search does not yet branch on it.
static ANALYSE_MODE: AtomicBool = AtomicBool::new(false);

/// `NormalizeScore`: print `score cp` on the pawn-anchored scale, where +100 means the
/// side to move expects to win half its games, instead of in the search's own units.
///
/// Display only, and read nowhere but the two functions that print a score. On by
/// default now that the model in `wdl.rs` is fitted on the champion's own games: the raw
/// unit is available on demand, but it is not the number to show a person by default.
static NORMALIZE_SCORE: AtomicBool = AtomicBool::new(true);

/// `UCI_ShowWDL`: append ` wdl W D L` (per mille) to every `info` line carrying a score.
static SHOW_WDL: AtomicBool = AtomicBool::new(false);

fn debug_on() -> bool {
    DEBUG.load(Ordering::Relaxed)
}

/// `UCI_AnalyseMode`: a GUI analysis session, not a game.
pub fn analyse_mode() -> bool {
    ANALYSE_MODE.load(Ordering::Relaxed)
}

/// `NormalizeScore`: whether printed scores go through the win-rate scale.
pub fn normalize_score() -> bool {
    NORMALIZE_SCORE.load(Ordering::Relaxed)
}

/// `UCI_ShowWDL`: whether `info` lines carry win/draw/loss probabilities.
pub fn show_wdl() -> bool {
    SHOW_WDL.load(Ordering::Relaxed)
}

/// Commands the protocol (and our debug extras) recognise as the start of a line.
const UCI_COMMANDS: &[&str] = &[
    "uci",
    "debug",
    "isready",
    "setoption",
    "register",
    "ucinewgame",
    "position",
    "go",
    "stop",
    "ponderhit",
    "quit",
    "d",
    "eval",
];

/// Skip unknown leading tokens so `joho debug on` still enables debug.
fn skip_unknown_prefix<'a>(tokens: &'a [&'a str]) -> &'a [&'a str] {
    match tokens.iter().position(|t| UCI_COMMANDS.contains(t)) {
        Some(i) => &tokens[i..],
        None => &[],
    }
}

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
                    let tokens: Vec<&str> = l.split_whitespace().collect();
                    let rest = skip_unknown_prefix(&tokens);
                    match rest {
                        ["stop", ..] | ["quit", ..] => {
                            STOP.store(true, Ordering::Relaxed);
                        }
                        ["ponderhit", ..] => {
                            PONDER.store(false, Ordering::Relaxed);
                        }
                        ["debug", "on", ..] => {
                            DEBUG.store(true, Ordering::Relaxed);
                        }
                        ["debug", "off", ..] => {
                            DEBUG.store(false, Ordering::Relaxed);
                        }
                        ["isready", ..] if crate::threads::SEARCHING.load(Ordering::Relaxed) => {
                            // Spec: answer immediately without stopping the search.
                            // Do not queue the line: the session would emit a second readyok.
                            out!("readyok");
                            continue;
                        }
                        _ => {}
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
    chess960: bool,
}

impl UciSession {
    pub fn new() -> UciSession {
        UciSession {
            pos: Position::from_fen(STARTPOS).expect("startpos"),
            pool: ThreadPool::new(1, 16),
            ponder_enabled: true,
            chess960: false,
        }
    }

    /// Heavy init the spec wants after `uciok`: GaiaTB decompresses ~147 MB.
    fn ensure_ready(&self) {
        #[cfg(all(feature = "gaiatb", gaiatb_embedded))]
        {
            use std::sync::atomic::AtomicBool;
            static LOADED: AtomicBool = AtomicBool::new(false);
            if !LOADED.swap(true, Ordering::Relaxed) {
                if crate::dtm::init() {
                    outerr!("info string GaiaTB loaded (3+4 piece DTM tablebases)");
                } else {
                    outerr!("info string GaiaTB: failed to load embedded tablebases");
                }
            }
        }
    }

    /// Handles one line. Returns `false` once the engine has been told to quit.
    pub fn command(&mut self, line: &str) -> bool {
        crate::out::log_input(line);
        if debug_on() {
            out!("info string {line}");
        }
        let tokens: Vec<&str> = line.split_whitespace().collect();
        let tokens = skip_unknown_prefix(&tokens);
        if tokens.is_empty() {
            return true;
        }

        match tokens {
            ["uci", ..] => self.identify(),

            ["debug", "on", ..] => DEBUG.store(true, Ordering::Relaxed),
            ["debug", "off", ..] => DEBUG.store(false, Ordering::Relaxed),
            ["debug", ..] => {}

            ["isready", ..] => {
                self.ensure_ready();
                out!("readyok");
            }

            ["ucinewgame", ..] => {
                self.pool.clear();
                #[cfg(feature = "online-tb")]
                crate::online_tb::clear();
                self.pos = Position::from_fen_ex(STARTPOS, self.chess960).expect("startpos");
            }

            ["position", rest @ ..] => parse_position(rest, &mut self.pos, self.chess960),

            ["go", rest @ ..] => {
                self.ensure_ready();
                if rest.contains(&"ponder") {
                    PONDER.store(true, Ordering::Relaxed);
                }
                #[cfg(feature = "online-tb")]
                if crate::online_tb::is_enabled() {
                    crate::online_tb::ensure_worker_running();
                }
                let (limits, search_moves) = parse_go(rest, &self.pos);
                let (best, ponder_move) =
                    self.pool
                        .start_search_filtered(&self.pos, limits, &search_moves);
                if !best.is_ok() {
                    // No legal move (mate or stalemate). The protocol's nullmove is 0000.
                    out!("bestmove 0000");
                } else if self.ponder_enabled
                    && let Some(pm) = ponder_move
                {
                    out!(
                        "bestmove {} ponder {}",
                        best.to_uci_960(self.chess960),
                        pm.to_uci_960(self.chess960)
                    );
                } else {
                    out!("bestmove {}", best.to_uci_960(self.chess960));
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
                        (
                            &self.pool.threads[idx].root_moves[0].pv[..],
                            self.pool.threads[idx].root_moves[0].pv_len,
                        )
                    } else {
                        (&[][..], 0)
                    };
                    crate::online_tb::start_prefetch(
                        self.pos.clone(),
                        best,
                        ponder_move,
                        pv,
                        pv_len,
                    );
                }
            }

            ["stop", ..] => {
                STOP.store(true, Ordering::Relaxed);
            }

            ["ponderhit", ..] => {}

            ["register", ..] => {}

            ["setoption", rest @ ..] => {
                parse_setoption(
                    rest,
                    &mut self.pool,
                    &mut self.ponder_enabled,
                    &mut self.chess960,
                );
                self.pos.set_chess960(self.chess960);
            }

            ["d", ..] => print_board(&self.pos),

            ["eval", ..] => print_eval(&self.pos),

            ["quit", ..] => return false,

            _ => {}
        }

        true
    }

    fn identify(&self) {
        out!("id name Gaia {}", env!("CARGO_PKG_VERSION"));
        out!("id author Jean-Francois Romang");
        out!("option name Hash type spin default 16 min 1 max 1048576");
        out!("option name Threads type spin default 1 min 1 max 256");
        out!("option name EvalFile type string default <internal>");
        out!("option name Ponder type check default true");
        out!("option name OwnBook type check default true");
        out!("option name MultiPV type spin default 1 min 1 max 256");
        out!("option name NormalizeScore type check default true");
        out!("option name Move Overhead type spin default 100 min 0 max 5000");
        out!("option name UseSoftNodes type check default false");
        out!("option name Debug Log File type string default");
        out!("option name Clear Hash type button");
        #[cfg(target_os = "linux")]
        out!("option name SharedNetwork type check default true");
        out!("option name UCI_Chess960 type check default false");
        out!(
            "option name UCI_EngineAbout type string default Gaia {} by Jean-Francois Romang, GPL-3.0-or-later, https://github.com/jromang/gaiachess",
            env!("CARGO_PKG_VERSION")
        );
        out!("option name UCI_AnalyseMode type check default false");
        out!("option name UCI_ShowWDL type check default false");
        out!("option name UCI_LimitStrength type check default false");
        out!(
            "option name UCI_Elo type spin default {} min {} max {}",
            crate::skill::MAX_LIMITED_ELO,
            crate::skill::MIN_ELO,
            crate::skill::MAX_LIMITED_ELO
        );
        out!("option name UCI_Opponent type string default <empty>");
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
        {
            out!("option name NalimovPath type string default <empty>");
            out!("option name NalimovCache type spin default 256 min 1 max 1024");
        }
        #[cfg(feature = "online-tb")]
        out!("option name OnlineTB type check default false");
        #[cfg(feature = "spsa")]
        crate::tune::print_uci_options();
        out!("uciok");
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
fn parse_position(tokens: &[&str], pos: &mut Position, chess960: bool) {
    let tokens = match tokens {
        ["startpos", rest @ ..] => {
            *pos = Position::from_fen_ex(STARTPOS, chess960).expect("startpos");
            rest
        }
        ["fen", rest @ ..] => {
            let fen_end = rest
                .iter()
                .position(|&t| t == "moves")
                .unwrap_or(rest.len());
            let fen = rest[..fen_end].join(" ");
            match Position::from_fen_ex(&fen, chess960) {
                Ok(p) => *pos = p,
                Err(e) => {
                    // stdout: ChessBase never reads stderr, and a silent keep of
                    // the previous position is how analysis shows depth/nps with
                    // no lines (the PV is illegal on the GUI board).
                    out!("info string Invalid FEN: {e}");
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
                    out!("info string Illegal move: {move_str}");
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
        if buf[i].to_uci_960(pos.chess960) == s {
            return Some(buf[i]);
        }
    }
    None
}

/// Parse the next token as a value of type `T`.
fn next_val<T: std::str::FromStr>(iter: &mut std::slice::Iter<'_, &str>) -> Option<T> {
    iter.next().and_then(|s| s.parse().ok())
}

/// Parse `go` command options into SearchLimits and an optional `searchmoves` filter.
fn parse_go(tokens: &[&str], pos: &Position) -> (SearchLimits, Vec<Move>) {
    let stm = pos.side_to_move as usize;
    let mut limits = SearchLimits::default();
    let mut clocks: [Option<u64>; 2] = [None, None];
    let mut incs = [0u64; 2];
    let mut search_moves = Vec::new();

    let mut iter = tokens.iter();
    while let Some(&tok) = iter.next() {
        match tok {
            "infinite" => limits.infinite = true,
            "ponder" => {}
            "depth" => limits.depth = next_val(&mut iter),
            "nodes" => limits.nodes = next_val(&mut iter),
            "softnodes" => {
                if let Some(n) = next_val::<u64>(&mut iter) {
                    limits.soft_nodes = Some(n);
                    limits.hard_nodes =
                        Some(n.saturating_mul(crate::timeman::SOFT_NODES_HARD_FACTOR));
                }
            }
            "movetime" => limits.movetime = next_val(&mut iter),
            "mate" => limits.mate = next_val(&mut iter),
            "wtime" => clocks[Color::White as usize] = next_val(&mut iter),
            "btime" => clocks[Color::Black as usize] = next_val(&mut iter),
            "winc" => incs[Color::White as usize] = next_val(&mut iter).unwrap_or(0),
            "binc" => incs[Color::Black as usize] = next_val(&mut iter).unwrap_or(0),
            "movestogo" => limits.movestogo = next_val(&mut iter),
            // UCI: searchmoves is last on the line; consume the rest as UCI moves.
            "searchmoves" => {
                for &mv in iter.by_ref() {
                    if let Some(m) = parse_uci_move(pos, mv) {
                        search_moves.push(m);
                    }
                }
            }
            _ => {}
        }
    }

    if let Some(time) = clocks[stm] {
        limits.time = Some(time);
        limits.inc = incs[stm];
    }

    if let Some(n) = limits.nodes
        && limits.soft_nodes.is_none()
        && USE_SOFT_NODES.load(Ordering::Relaxed)
    {
        limits.soft_nodes = Some(n);
        limits.hard_nodes = Some(n.saturating_mul(crate::timeman::SOFT_NODES_HARD_FACTOR));
        limits.nodes = None;
    }

    // A `go` with no bound searches until `stop`. Depth 6 was a silent
    // default that is not in the protocol.
    if !limits.infinite
        && limits.depth.is_none()
        && limits.nodes.is_none()
        && limits.soft_nodes.is_none()
        && limits.movetime.is_none()
        && limits.time.is_none()
        && limits.mate.is_none()
    {
        limits.infinite = true;
    }

    (limits, search_moves)
}

/// Parse `setoption name <N...> [value <V...>]`.
/// Supports multi-word option names (e.g. "Move Overhead") and button options
/// with no value (`setoption name Clear Hash`).
fn parse_setoption(
    tokens: &[&str],
    pool: &mut ThreadPool,
    ponder_enabled: &mut bool,
    chess960: &mut bool,
) {
    if tokens.first() != Some(&"name") || tokens.len() < 2 {
        return;
    }
    let value_pos = tokens[1..]
        .iter()
        .position(|&t| t == "value")
        .map(|p| p + 1);
    let name = match value_pos {
        Some(vp) if vp >= 2 => tokens[1..vp].join(" "),
        Some(_) => return,
        None => tokens[1..].join(" "),
    };
    if name.is_empty() {
        return;
    }
    let value = match value_pos {
        Some(vp) if vp + 1 < tokens.len() => tokens[vp + 1..].join(" "),
        _ => String::new(),
    };

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
        "uci_chess960" => {
            *chess960 = value.eq_ignore_ascii_case("true");
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
        "usesoftnodes" => {
            USE_SOFT_NODES.store(value.eq_ignore_ascii_case("true"), Ordering::Relaxed);
        }
        "debug log file" => {
            if let Err(e) = crate::out::set_debug_log(&value) {
                out!("info string Debug Log File: {e}");
            }
        }
        "sharednetwork" => {
            // Same weights either way, so the hash and the accumulator caches
            // stay valid: nothing is cleared and no generation is bumped.
            // The new backing announces itself from where it is installed;
            // only a refusal has to be reported from here.
            if let Err(e) = crate::nnue::network::set_shared(value.eq_ignore_ascii_case("true")) {
                outerr!("info string SharedNetwork: {e}");
            }
        }
        "evalfile" => {
            if value == "<internal>" {
                outerr!("info string Using default evaluation");
            } else {
                use crate::nnue::integrity::Provenance;
                match crate::nnue::network::load_from_file(&value) {
                    Ok(Provenance::Verified) => {
                        outerr!(
                            "info string Loaded NNUE network from {value} (integrity verified)"
                        );
                        pool.clear();
                    }
                    Ok(Provenance::Legacy) => {
                        outerr!(
                            "info string Loaded NNUE network from {value} (legacy, no integrity footer)"
                        );
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
                    outerr!(
                        "info string Nalimov tablebases loaded ({}-piece) from {value}",
                        crate::nalimov::max_pieces()
                    );
                }
            }
            #[cfg(not(feature = "nalimov"))]
            outerr!("info string NalimovPath requires --features nalimov");
        }
        "clear hash" => {
            pool.clear();
        }
        "uci_analysemode" => {
            ANALYSE_MODE.store(value.eq_ignore_ascii_case("true"), Ordering::Relaxed);
        }
        "normalizescore" => {
            NORMALIZE_SCORE.store(value.eq_ignore_ascii_case("true"), Ordering::Relaxed);
        }
        "uci_showwdl" => {
            SHOW_WDL.store(value.eq_ignore_ascii_case("true"), Ordering::Relaxed);
        }
        "uci_limitstrength" => {
            let on = value.eq_ignore_ascii_case("true");
            LIMIT_STRENGTH.store(on, Ordering::Relaxed);
            if on {
                crate::skill::set(
                    crate::skill::level_for_elo(UCI_ELO.load(Ordering::Relaxed)),
                    skill_seed(),
                );
            } else {
                crate::skill::set(crate::skill::FULL_STRENGTH, skill_seed());
            }
        }
        "uci_elo" => {
            if let Ok(elo) = value.parse::<i32>() {
                let elo = elo.clamp(crate::skill::MIN_ELO, crate::skill::MAX_LIMITED_ELO);
                UCI_ELO.store(elo, Ordering::Relaxed);
                if LIMIT_STRENGTH.load(Ordering::Relaxed) {
                    crate::skill::set(crate::skill::level_for_elo(elo), skill_seed());
                }
            }
        }
        "uci_opponent" => {
            // Stored only so a GUI that sent it is not talking into a void.
            let _ = value;
        }
        "uci_engineabout" => {}
        "nalimovcache" => {
            #[cfg(feature = "nalimov")]
            {
                if let Ok(mb) = value.parse::<i32>() {
                    debug_assert!((1..=1024).contains(&mb), "NalimovCache {mb} out of range");
                    let _ = mb.clamp(1, 1024);
                }
            }
        }
        "onlinetb" => {
            #[cfg(feature = "online-tb")]
            {
                let enabled = value.eq_ignore_ascii_case("true");
                crate::online_tb::set_enabled(enabled);
                outerr!(
                    "info string OnlineTB {}",
                    if enabled { "enabled" } else { "disabled" }
                );
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
            // Keep this line exactly as it is: the network validation chain scrapes it to
            // compare the trainer's forward pass against the engine's, and a reworded or
            // rescaled number there would silently move the parity gate.
            out!("NNUE eval:  {} cp (STM POV)", nnue_score);
            let (w, d, l) = crate::wdl::wdl(nnue_score, pos);
            out!(
                "Normalized: {} cp (STM POV) wdl {w} {d} {l}",
                crate::wdl::to_cp(nnue_score, pos)
            );
        } else {
            out!("NNUE eval:  N/A (in check)");
        }
        let pieces = pos.occupied().count_ones();
        let bucket = nnue::output_bucket(pos);
        out!("Output bucket: {} (pieces: {})", bucket, pieces);
    } else {
        out!("NNUE: no network loaded");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timeman::SearchLimits;

    #[test]
    fn four_field_fen_sets_the_position() {
        let mut pos = Position::from_fen(STARTPOS).unwrap();
        parse_position(
            &[
                "fen",
                "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R",
                "b",
                "KQkq",
                "-",
            ],
            &mut pos,
            false,
        );
        assert_eq!(pos.side_to_move, Color::Black);
        assert_ne!(pos.to_fen(), STARTPOS);
    }

    #[test]
    fn bare_go_is_infinite() {
        let pos = Position::from_fen(STARTPOS).unwrap();
        let (limits, moves) = parse_go(&[], &pos);
        assert!(limits.infinite);
        assert!(moves.is_empty());
    }

    #[test]
    fn go_searchmoves_filters_legal_moves() {
        let pos = Position::from_fen(STARTPOS).unwrap();
        let (limits, moves) = parse_go(&["infinite", "searchmoves", "e2e4", "d2d4", "zzzz"], &pos);
        assert!(limits.infinite);
        assert_eq!(moves.len(), 2);
        assert!(moves.iter().any(|m| m.to_uci() == "e2e4"));
        assert!(moves.iter().any(|m| m.to_uci() == "d2d4"));
    }

    #[test]
    fn debug_log_file_records_input_and_output() {
        let _guard = lock_transcript();
        let path = std::env::temp_dir().join("gaiachess-uci-log-test.txt");
        crate::out::set_debug_log(path.to_str().expect("utf-8 temp path")).unwrap();
        let mut session = UciSession::new();
        session.command("isready");
        crate::out::set_debug_log("").unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert!(text.contains(">> isready"), "log missing input: {text}");
        assert!(text.contains("<< readyok"), "log missing output: {text}");
    }

    /// The debug log and the display options are one set of globals for the whole
    /// process, and the harness runs tests in parallel: without this, one test's capture
    /// swallows another's output, or clears the log out from under it.
    static TRANSCRIPT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn lock_transcript() -> std::sync::MutexGuard<'static, ()> {
        TRANSCRIPT_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Drive a whole session and hand back everything it printed, by way of the debug
    /// log — the only place an `out!` line can be read back inside a test.
    ///
    /// The caller must hold [`lock_transcript`].
    fn transcript(name: &str, commands: &[&str]) -> String {
        let path = std::env::temp_dir().join(format!(
            "gaiachess-uci-{name}-{}.txt",
            std::process::id()
        ));
        crate::out::set_debug_log(path.to_str().expect("utf-8 temp path")).unwrap();
        let mut session = UciSession::new();
        for command in commands {
            session.command(command);
        }
        crate::out::set_debug_log("").unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        text
    }

    /// The last `score cp` value printed by a session.
    fn last_score(text: &str) -> i32 {
        text.lines()
            .filter(|line| line.contains(" score cp "))
            .filter_map(|line| line.split(" score cp ").nth(1))
            .filter_map(|rest| rest.split_whitespace().next())
            .filter_map(|value| value.parse().ok())
            .last()
            .unwrap_or_else(|| panic!("no score cp line in: {text}"))
    }

    #[test]
    fn the_two_display_options_are_read_back() {
        let _guard = lock_transcript();
        let mut pool = crate::threads::ThreadPool::new(1, 16);
        let mut ponder = true;
        let mut chess960 = false;
        let mut set = |name: &str, value: &str| {
            parse_setoption(&["name", name, "value", value], &mut pool, &mut ponder, &mut chess960);
        };

        set("NormalizeScore", "true");
        assert!(normalize_score());
        set("UCI_ShowWDL", "true");
        assert!(show_wdl());

        // Both are process globals: leave them as the rest of the suite expects them.
        set("NormalizeScore", "false");
        set("UCI_ShowWDL", "false");
        assert!(!normalize_score());
        assert!(!show_wdl());
    }

    #[test]
    fn show_wdl_adds_three_probabilities_before_the_node_count() {
        let _guard = lock_transcript();
        let text = transcript("wdl", &[
            "setoption name UCI_ShowWDL value true",
            "position fen rnbqkbnr/ppp1pppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "go depth 4",
            "setoption name UCI_ShowWDL value false",
        ]);
        let line = text
            .lines()
            .find(|line| line.contains(" score cp "))
            .unwrap_or_else(|| panic!("no info line in: {text}"));

        let wdl = line.split(" wdl ").nth(1).unwrap_or_else(|| panic!("no wdl in: {line}"));
        let parts: Vec<i32> = wdl
            .split_whitespace()
            .take(3)
            .map(|n| n.parse().expect("per mille integer"))
            .collect();
        assert_eq!(parts.iter().sum::<i32>(), 1000, "line: {line}");
        assert!(
            line.find(" wdl ").unwrap() < line.find(" nodes ").unwrap(),
            "wdl must come before nodes: {line}"
        );
        assert!(!show_wdl(), "the option must be back off for the other tests");
    }

    /// The option is display only: the same search prints two different numbers for one
    /// score, and turning it off gives the search's own unit back untouched.
    #[test]
    fn normalizing_only_rescales_what_is_printed() {
        let _guard = lock_transcript();
        let position = "position fen rnbqkbnr/ppp1pppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        // Both sides are set explicitly, so the test says what it means whichever way the
        // default happens to point.
        let raw = last_score(&transcript("raw", &[
            "setoption name NormalizeScore value false",
            position,
            "go depth 4",
        ]));
        let normalized = last_score(&transcript("norm", &[
            "setoption name NormalizeScore value true",
            position,
            "go depth 4",
        ]));

        let pos = Position::from_fen(
            "rnbqkbnr/ppp1pppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        )
        .unwrap();
        assert_eq!(normalized, crate::wdl::to_cp(raw, &pos));
        assert_ne!(normalized, raw, "the two scales must not coincide on this position");
    }

    #[test]
    fn four_field_fen_search_plays_a_black_move() {
        // The Fritz symptom: a 4-field FEN used to be rejected, the engine
        // stayed on startpos, and the PV (e2e4…) was illegal on the GUI board.
        let mut pos = Position::from_fen(STARTPOS).unwrap();
        parse_position(
            &[
                "fen",
                "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R",
                "b",
                "KQkq",
                "-",
            ],
            &mut pos,
            false,
        );
        let mut pool = crate::threads::ThreadPool::new(1, 16);
        let (best, _) = pool.start_search(&pos, SearchLimits::depth(1));
        assert!(best.is_ok());
        let uci = best.to_uci();
        assert!(
            uci.as_bytes()[1] >= b'5',
            "expected a black move from the entered position, got {uci}"
        );
    }

    #[test]
    fn unknown_leading_tokens_are_skipped() {
        assert_eq!(
            skip_unknown_prefix(&["joho", "debug", "on"]),
            &["debug", "on"]
        );
        assert_eq!(skip_unknown_prefix(&["uci", "extra"]), &["uci", "extra"]);
        assert!(skip_unknown_prefix(&["nope", "still-nope"]).is_empty());
    }

    #[test]
    fn go_combines_depth_and_movetime() {
        let pos = Position::from_fen(STARTPOS).unwrap();
        let (limits, _) = parse_go(&["depth", "12", "movetime", "1000"], &pos);
        assert_eq!(limits.depth, Some(12));
        assert_eq!(limits.movetime, Some(1000));
        assert!(!limits.infinite);
    }

    #[test]
    fn go_mate_is_parsed() {
        let pos = Position::from_fen(STARTPOS).unwrap();
        let (limits, _) = parse_go(&["mate", "3"], &pos);
        assert_eq!(limits.mate, Some(3));
        assert!(!limits.infinite);
    }

    #[test]
    fn setoption_button_has_no_value() {
        let mut pool = crate::threads::ThreadPool::new(1, 16);
        let mut ponder = true;
        let mut chess960 = false;
        parse_setoption(
            &["name", "Clear", "Hash"],
            &mut pool,
            &mut ponder,
            &mut chess960,
        );
    }

    #[test]
    fn uci_extra_tokens_still_identify() {
        let mut session = UciSession::new();
        assert!(session.command("uci foobar"));
        assert!(session.command("joho isready"));
    }
}
