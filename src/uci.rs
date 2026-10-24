//! [UCI protocol](https://backscattering.de/chess/uci/) implementation.
//!
//! Reads commands from stdin on a separate thread (via `mpsc`), dispatches
//! them on the main thread. Search uses Lazy SMP: the main thread blocks
//! during search while the stdin reader thread can still receive `stop`.

use std::collections::BTreeSet;
use std::io::BufRead;
use std::sync::Mutex;
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

/// `UCI_ShowCurrLine`: publish lock-free snapshots of the lines being searched.
static SHOW_CURR_LINE: AtomicBool = AtomicBool::new(false);

/// Every `go` the stdin reader has passed on, counted in the order it read them.
///
/// The reader acts on `stop` and `ponderhit` the moment it reads them, but the session
/// only reaches the `go` they follow once it has dealt with everything queued before
/// it. Arming a search then means lowering STOP and raising PONDER — and doing that
/// blindly would wipe out an event the reader has already seen for this very search.
/// The counts say which search an event was meant for (see [`arm_search_flags`]).
static GO_READ: AtomicU64 = AtomicU64::new(0);
/// The `go` (by [`GO_READ`]) whose search the session armed last. The reader acts on
/// the flags at once only for that one; an event meant for a later `go` waits for it.
static GO_ARMED: AtomicU64 = AtomicU64::new(0);
/// The `go`s (by [`GO_READ`]) whose `bestmove` the session has given.
static GO_DONE: AtomicU64 = AtomicU64::new(0);
/// Commands queued by the reader that wait for the running search to end — `ucinewgame`
/// and `setoption` (see [`waits_for_the_search`]) — and that the session has not taken
/// off the queue yet.
static ORDERED_PENDING: AtomicU64 = AtomicU64::new(0);
/// The `go`s (by [`GO_READ`]) a `stop` or `quit` was read after, not armed yet.
static STOPPED_GOS: Mutex<BTreeSet<u64>> = Mutex::new(BTreeSet::new());
/// The `go`s (by [`GO_READ`]) a `ponderhit` was read after, not armed yet.
static HIT_GOS: Mutex<BTreeSet<u64>> = Mutex::new(BTreeSet::new());

/// Records that an event was read after the `go`-th `go`.
fn note_event(gos: &Mutex<BTreeSet<u64>>, go: u64) {
    gos.lock().unwrap_or_else(|e| e.into_inner()).insert(go);
}

/// Whether an event was read after the `go`-th `go`, forgetting it along with every
/// event of the searches before it, which are over.
fn take_event(gos: &Mutex<BTreeSet<u64>>, go: u64) -> bool {
    let mut gos = gos.lock().unwrap_or_else(|e| e.into_inner());
    let found = gos.contains(&go);
    *gos = gos.split_off(&(go + 1));
    debug_assert!(gos.iter().all(|&g| g > go));
    found
}

/// Sets STOP and PONDER for the search of the `go`-th `go`, keeping any `stop` or
/// `ponderhit` the reader has already read after it.
///
/// Each event is kept for the exact `go` it was read after: one read after a later
/// `go`, still in the queue, is not this search's, and one read after an earlier `go`
/// belonged to a search that is over. Announced first, lowered next, checked last: the
/// reader notes an event before it looks whether its search is the armed one, so an
/// event noted before the check is seen by it, and one noted after finds this search
/// armed and raises the flag itself, after it was lowered. Sequentially consistent
/// throughout, as the argument needs a store then a load to be ordered on both
/// threads. Without a reader (tests, the browser build) nothing is ever noted and the
/// search starts clean, as it always did.
fn arm_search_flags(go: u64, ponder: bool) {
    arm_flags(go, ponder, &STOP, &PONDER);
}

/// [`arm_search_flags`] on flags of the caller's choosing, so a test can watch it work
/// without raising the process-wide STOP under every other test's search.
fn arm_flags(go: u64, ponder: bool, stop: &AtomicBool, pondering: &AtomicBool) {
    debug_assert!(go > 0, "a search is armed for a go that was counted");
    GO_ARMED.store(go, Ordering::SeqCst);
    stop.store(false, Ordering::SeqCst);
    pondering.store(ponder, Ordering::SeqCst);
    if take_event(&STOPPED_GOS, go) {
        stop.store(true, Ordering::SeqCst);
    }
    if take_event(&HIT_GOS, go) {
        pondering.store(false, Ordering::SeqCst);
    }
}

/// `UCI_ShowRefutations`: publish the replies found for non-best root moves.
static SHOW_REFUTATIONS: AtomicBool = AtomicBool::new(false);

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

/// Whether search threads should publish their current root line snapshots.
pub fn show_curr_line() -> bool {
    SHOW_CURR_LINE.load(Ordering::Relaxed)
}

/// Whether completed iterations should report root-move refutations.
pub fn show_refutations() -> bool {
    SHOW_REFUTATIONS.load(Ordering::Relaxed)
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
    "bench",
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
/// even though the main thread is blocked in start_search(). Each `stop` and
/// `ponderhit` is noted for the `go` it follows as well, so that one read ahead of its
/// search waits for the session to get round to that `go` (see [`arm_search_flags`]).
pub fn spawn_stdin_reader() -> mpsc::Receiver<String> {
    let (tx, rx) = mpsc::channel::<String>();
    std::thread::spawn(move || {
        let flags = ReaderFlags {
            stop: &STOP,
            ponder: &PONDER,
        };
        read_commands(std::io::stdin().lock(), &tx, &flags);
    });
    rx
}

/// The search flags the stdin reader acts on: the process-wide ones, or a test's own,
/// so that a test can feed the reader a `stop` without stopping every other test's
/// search.
struct ReaderFlags<'a> {
    stop: &'a AtomicBool,
    ponder: &'a AtomicBool,
}

/// The stdin reader's loop, on any byte source: reads lines until the input ends and
/// passes each one on to the session.
///
/// Lines are read as bytes and decoded leniently. A GUI on Windows may well send its
/// strings in its own code page — a `SyzygyPath` under a user folder with an
/// accented name, an `UCI_Opponent` spelt the same way — and a line that is not UTF-8
/// must cost at most a replacement character, never the reader: without it nothing
/// reads `stop` any more.
fn read_commands(mut input: impl BufRead, tx: &mpsc::Sender<String>, flags: &ReaderFlags) {
    let mut buf = Vec::new();
    loop {
        buf.clear();
        // `read_until` already retries on `Interrupted`; any other error ends the input.
        match input.read_until(b'\n', &mut buf) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        let line = String::from_utf8_lossy(&buf)
            .trim_end_matches(['\n', '\r'])
            .to_owned();
        debug_assert!(!line.contains('\n'), "one line at a time");
        if !read_line(line, tx, flags) {
            return;
        }
    }
    // The end of the input is a `quit` nobody sent: the protocol has no other way to
    // end a `go infinite` or a `go ponder`, and with stdin gone nothing will ever stop
    // them. It is not queued, though: the caller deciding whether to open the board
    // must still see the channel close.
    read_stop(flags);
}

/// `stop` or `quit`: noted for the last `go` read, and its search stopped now if it
/// is the one running — otherwise when the session arms it (see [`arm_search_flags`]).
fn read_stop(flags: &ReaderFlags) {
    let go = GO_READ.load(Ordering::SeqCst);
    note_event(&STOPPED_GOS, go);
    if GO_ARMED.load(Ordering::SeqCst) == go {
        flags.stop.store(true, Ordering::SeqCst);
    }
}

/// Acts on one line the moment the reader sees it, then queues it for the session.
/// Returns `false` once the session has gone and nobody will read the queue any more.
fn read_line(line: String, tx: &mpsc::Sender<String>, flags: &ReaderFlags) -> bool {
    let tokens: Vec<&str> = line.split_whitespace().collect();
    let rest = skip_unknown_prefix(&tokens);
    match rest {
        ["go", ..] => {
            GO_READ.fetch_add(1, Ordering::SeqCst);
        }
        ["stop", ..] | ["quit", ..] => read_stop(flags),
        ["ponderhit", ..] => {
            let go = GO_READ.load(Ordering::SeqCst);
            note_event(&HIT_GOS, go);
            if GO_ARMED.load(Ordering::SeqCst) == go {
                flags.ponder.store(false, Ordering::SeqCst);
            }
        }
        ["debug", "on", ..] => {
            DEBUG.store(true, Ordering::Relaxed);
        }
        ["debug", "off", ..] => {
            DEBUG.store(false, Ordering::Relaxed);
        }
        ["isready", ..] if ready_now() => {
            // Spec: answer immediately without stopping the search.
            // Do not queue the line: the session would emit a second readyok.
            out!("readyok");
            return true;
        }
        tokens if waits_for_the_search(tokens) => {
            // Counted before it is queued, so an `isready` read next sees it.
            ORDERED_PENDING.fetch_add(1, Ordering::SeqCst);
        }
        _ => {}
    }
    tx.send(line).is_ok()
}

/// Whether the reader may answer `isready` itself instead of queuing it.
///
/// Only while a `go` it has read is still waiting for its `bestmove` — queued, being
/// set up or searching: the session would otherwise answer it only after the search,
/// and the protocol wants an answer at once. And only when nothing queued has to wait
/// for that search: a GUI sends `isready` behind `ucinewgame` or a `setoption Hash` to
/// know when they are done, and answering before the session has even reached them
/// would send its next `go` into a table still being cleared.
fn ready_now() -> bool {
    GO_READ.load(Ordering::SeqCst) > GO_DONE.load(Ordering::SeqCst)
        && ORDERED_PENDING.load(Ordering::SeqCst) == 0
}

/// Commands that the session only carries out once the search before them is over, so
/// that an `isready` behind them must wait for them too.
fn waits_for_the_search(tokens: &[&str]) -> bool {
    matches!(tokens, ["ucinewgame", ..] | ["setoption", ..])
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
    /// `go` commands handled so far: the same count the stdin reader keeps in
    /// [`GO_READ`], since every line reaches both in the same order.
    go_count: u64,
    /// The skill level the last search ran at (see [`UciSession::forget_other_levels`]).
    searched_level: i32,
}

impl UciSession {
    pub fn new() -> UciSession {
        UciSession {
            pos: Position::from_fen(STARTPOS).expect("startpos"),
            pool: ThreadPool::new(1, 16),
            ponder_enabled: true,
            chess960: false,
            go_count: 0,
            searched_level: crate::skill::FULL_STRENGTH,
        }
    }

    /// Clears the tables and histories when the level has changed since the last search.
    ///
    /// A weakened level stores its own evaluations — material or piece squares mixed
    /// into the network, then noised — and scores from a tree it searched half blind.
    /// Full strength would read them back as its own, cut on them, and train its
    /// correction on them; a weak level would play above its rung on what full strength
    /// left. Checked at the search, which every way of changing the level goes through.
    fn forget_other_levels(&mut self) {
        let level = crate::skill::level();
        if level != self.searched_level {
            self.pool.clear();
            self.searched_level = level;
        }
    }

    /// Heavy init the spec wants after `uciok`: GaiaTB decompresses ~147 MB, unless
    /// another engine process on the machine already did and shares its image.
    fn ensure_ready(&self) {
        #[cfg(all(feature = "gaiatb", gaiatb_embedded))]
        {
            use std::sync::atomic::AtomicBool;
            static LOADED: AtomicBool = AtomicBool::new(false);
            if !LOADED.swap(true, Ordering::Relaxed) {
                if crate::dtm::init() {
                    out!("info string GaiaTB loaded (3+4 piece DTM tablebases)");
                } else {
                    out!("info string GaiaTB: failed to load embedded tablebases");
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

            ["go", "perft", rest @ ..] => {
                // Counted like any `go`, as the reader counted it: the events read after
                // the next one must still find their search.
                self.go_count += 1;
                match rest.first().and_then(|d| d.parse::<u32>().ok()) {
                    Some(depth) => {
                        crate::perft::divide(&mut self.pos.clone(), depth);
                    }
                    None => out!("info string go perft needs a depth"),
                }
                // No search and no bestmove, but this `go` is over.
                GO_DONE.store(self.go_count, Ordering::SeqCst);
            }

            ["go", rest @ ..] => {
                self.ensure_ready();
                self.forget_other_levels();
                let ponder_requested = rest.contains(&"ponder");
                self.go_count += 1;
                arm_search_flags(self.go_count, ponder_requested);
                #[cfg(feature = "online-tb")]
                if crate::online_tb::is_enabled() {
                    crate::online_tb::ensure_worker_running();
                }
                let (limits, search_moves) = parse_go(rest, &self.pos);
                let infinite = limits.infinite;
                // Armed: STOP already says whether this search was stopped before it
                // began, and must not be lowered again.
                let (best, ponder_move) =
                    self.pool
                        .start_search_armed(&self.pos, limits, &search_moves);
                // No legal move: there is nothing to wait for, the answer is final.
                wait_for_release(ponder_requested, infinite && best.is_ok());
                self.pool.print_final_uci_info();
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
                // From here on an `isready` is the session's to answer, in its turn.
                GO_DONE.store(self.go_count, Ordering::SeqCst);
                PONDER.store(false, Ordering::Relaxed);
                // Search statistics summary on stderr (invisible to GUIs)
                #[cfg(feature = "stats")]
                eprint!("{}", self.pool.aggregated_stats().emit_text());
                // Throwaway debug slots (silent when unused)
                crate::stats::dbg::print();
                // Prefetch online TB: PV positions first, then opponent replies
                #[cfg(feature = "online-tb")]
                if crate::online_tb::is_enabled() {
                    let front = self
                        .pool
                        .threads
                        .get(self.pool.last_best_idx)
                        .and_then(|td| td.root_moves.first());
                    let (pv, pv_len) = match front {
                        Some(rm) => (&rm.pv[..], rm.pv_len),
                        None => (&[][..], 0),
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

            // `bench [depth] [threads]`, as on the command line. No search of the
            // session's can be running: it handles one line at a time, and a `go` only
            // returns with its bestmove. The bench brings its own table and thread data,
            // so nothing of the session's is touched — but it would search at the
            // session's skill level, which is the process's: full strength for the run,
            // so that the node count is the one the bench is checked against.
            #[cfg(not(target_arch = "wasm32"))]
            ["bench", args @ ..] => {
                let depth = args
                    .first()
                    .and_then(|d| d.parse::<i32>().ok())
                    .filter(|&d| d >= 1);
                let threads = args.get(1).and_then(|t| t.parse::<usize>().ok());
                let level = crate::skill::level();
                crate::skill::set(crate::skill::FULL_STRENGTH, skill_seed());
                crate::bench::run(
                    depth,
                    threads.map(|t| t.clamp(1, crate::threads::MAX_THREADS)),
                );
                crate::skill::set(level, skill_seed());
            }

            ["quit", ..] => return false,

            _ => {}
        }

        true
    }

    fn identify(&self) {
        out!("id name Gaia {}", env!("CARGO_PKG_VERSION"));
        out!("id author Jean-Francois Romang, David Rabel");
        out!("option name Hash type spin default 16 min 1 max 1048576");
        out!(
            "option name Threads type spin default 1 min 1 max {}",
            crate::threads::MAX_THREADS
        );
        out!("option name EvalFile type string default <internal>");
        out!("option name Ponder type check default true");
        out!("option name OwnBook type check default true");
        out!("option name MultiPV type spin default 1 min 1 max 256");
        #[cfg(not(target_arch = "wasm32"))]
        out!("option name UCI_ShowCurrLine type check default false");
        out!("option name UCI_ShowRefutations type check default false");
        out!("option name NormalizeScore type check default true");
        out!("option name Move Overhead type spin default 40 min 0 max 5000");
        out!("option name UseSoftNodes type check default false");
        out!("option name Debug Log File type string default <empty>");
        out!("option name Clear Hash type button");
        #[cfg(any(target_os = "linux", windows))]
        out!("option name SharedNetwork type check default true");
        #[cfg(not(target_arch = "wasm32"))]
        out!("option name NumaPolicy type string default auto");
        out!("option name UCI_Chess960 type check default false");
        out!(
            "option name UCI_EngineAbout type string default Gaia {} by Jean-Francois Romang, David Rabel, GPL-3.0-or-later, https://github.com/jromang/gaiachess",
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

/// Holds the `bestmove` back until the interface may have it.
///
/// A ponder search must not hand a move to the GUI before `ponderhit` or `stop`, and an
/// infinite one not before `stop` ("Do not exit the search without being told so in
/// this mode!"), even when a depth limit, a level's depth ceiling or a short mate ended
/// the search on its own.
fn wait_for_release(ponder_requested: bool, infinite: bool) {
    if !must_hold_bestmove(ponder_requested, infinite, true, false) {
        return;
    }

    #[cfg(not(target_arch = "wasm32"))]
    while must_hold_bestmove(
        ponder_requested,
        infinite,
        PONDER.load(Ordering::Relaxed),
        STOP.load(Ordering::Relaxed),
    ) {
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

/// Whether a finished search must keep its move to itself: a `stop` always releases
/// it; otherwise a ponder search waits for its `ponderhit`, and an infinite one for
/// its `stop`.
fn must_hold_bestmove(ponder_requested: bool, infinite: bool, pondering: bool, stop: bool) -> bool {
    !stop && ((ponder_requested && pondering) || infinite)
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
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if waits_for_the_search(skip_unknown_prefix(&tokens)) {
            // Every line here came through the reader, which counted this one.
            let before = ORDERED_PENDING.fetch_sub(1, Ordering::SeqCst);
            debug_assert!(before > 0, "a command was taken off the queue twice");
        }
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
                Some(m) => {
                    pos.make_move(m);
                    pos.rebase_history();
                }
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

/// Parse the next token as a time in milliseconds, a negative one counting as none left.
///
/// A GUI that lets a clock run past zero before it flags — a time margin — sends the
/// overdraft as it is, `wtime -50`. Read as unsigned it failed to parse, the clock was
/// taken as absent, and the `go` with no bound left searched until `stop`.
fn next_ms(iter: &mut std::slice::Iter<'_, &str>) -> Option<u64> {
    next_val::<i64>(iter).map(|t| t.max(0) as u64)
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
            "movetime" => limits.movetime = next_ms(&mut iter),
            "mate" => limits.mate = next_val(&mut iter),
            "wtime" => clocks[Color::White as usize] = next_ms(&mut iter),
            "btime" => clocks[Color::Black as usize] = next_ms(&mut iter),
            "winc" => incs[Color::White as usize] = next_val(&mut iter).unwrap_or(0),
            "binc" => incs[Color::Black as usize] = next_val(&mut iter).unwrap_or(0),
            "movestogo" => limits.movestogo = next_val(&mut iter),
            // Move tokens have a fixed shape, so the next recognised go keyword
            // unambiguously ends the list. GUIs are free to put other limits after it.
            "searchmoves" => {
                while let Some(&mv) = iter.as_slice().first() {
                    if GO_KEYWORDS.contains(&mv) {
                        break;
                    }
                    iter.next();
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

const GO_KEYWORDS: &[&str] = &[
    "searchmoves",
    "ponder",
    "wtime",
    "btime",
    "winc",
    "binc",
    "movestogo",
    "depth",
    "nodes",
    "softnodes",
    "mate",
    "movetime",
    "infinite",
];

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
                pool.resize_threads(n.clamp(1, crate::threads::MAX_THREADS));
                out!("info string {}", pool.numa_summary());
            }
        }
        "numapolicy" => match crate::numa::NumaPolicy::parse(&value) {
            Some(policy) => {
                pool.set_numa_policy(policy);
                out!("info string {}", pool.numa_summary());
            }
            None => {
                out!("info string NumaPolicy: invalid value '{value}', keeping the previous one")
            }
        },
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
            // The protocol spells an empty string `<empty>`: it closes the log, and
            // must not open a file of that name.
            let path = if value == "<empty>" { "" } else { &value };
            if let Err(e) = crate::out::set_debug_log(path) {
                out!("info string Debug Log File: {e}");
            }
        }
        "sharednetwork" => {
            // Same weights either way, so the hash and the accumulator caches
            // stay valid: nothing is cleared and no generation is bumped.
            // The new backing announces itself from where it is installed;
            // only a refusal has to be reported from here.
            if let Err(e) = crate::nnue::network::set_shared(value.eq_ignore_ascii_case("true")) {
                out!("info string SharedNetwork: {e}");
            }
        }
        "evalfile" => {
            if value == "<internal>" {
                // Back to the network built in, if a file had replaced it: the table
                // and caches were filled by the other one.
                if crate::nnue::network::load_internal() {
                    pool.clear();
                }
                out!("info string Using default evaluation");
            } else {
                use crate::nnue::integrity::Provenance;
                match crate::nnue::network::load_from_file(&value) {
                    Ok(Provenance::Verified) => {
                        out!("info string Loaded NNUE network from {value} (integrity verified)");
                        pool.clear();
                    }
                    Ok(Provenance::Legacy) => {
                        out!(
                            "info string Loaded NNUE network from {value} (legacy, no integrity footer)"
                        );
                        pool.clear();
                    }
                    Err(e) => {
                        out!("info string Failed to load network: {e}");
                    }
                }
            }
        }
        "syzygypath" => {
            #[cfg(feature = "syzygy")]
            {
                if value == "<empty>" {
                    crate::tb::free();
                } else {
                    // Said every time, found or not: a wrong path is otherwise silent,
                    // and the GUI's user would play on believing the tables are there.
                    crate::tb::init(&value);
                    let (wdl, dtz) = crate::tb::file_counts();
                    out!(
                        "info string Found {wdl} WDL and {dtz} DTZ tablebase files (up to {}-man).",
                        crate::tb::max_pieces()
                    );
                }
            }
            #[cfg(not(feature = "syzygy"))]
            out!("info string SyzygyPath requires --features syzygy");
        }
        "nalimovpath" => {
            #[cfg(feature = "nalimov")]
            {
                if value == "<empty>" {
                    crate::nalimov::free();
                } else if !crate::nalimov::init(&value) {
                    out!("info string Failed to load Nalimov tablebases from {value}");
                } else {
                    out!(
                        "info string Nalimov tablebases loaded ({}-piece) from {value}",
                        crate::nalimov::max_pieces()
                    );
                }
            }
            #[cfg(not(feature = "nalimov"))]
            out!("info string NalimovPath requires --features nalimov");
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
        "uci_showcurrline" => {
            #[cfg(not(target_arch = "wasm32"))]
            SHOW_CURR_LINE.store(value.eq_ignore_ascii_case("true"), Ordering::Relaxed);
        }
        "uci_showrefutations" => {
            SHOW_REFUTATIONS.store(value.eq_ignore_ascii_case("true"), Ordering::Relaxed);
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
                if let Ok(mb) = value.parse::<u32>() {
                    let mb = mb.clamp(1, 1024);
                    if crate::nalimov::set_cache_size_mb(mb) {
                        out!("info string Nalimov cache set to {mb} MiB");
                    } else {
                        out!("info string Failed to resize Nalimov cache to {mb} MiB");
                    }
                }
            }
        }
        "onlinetb" => {
            #[cfg(feature = "online-tb")]
            {
                let enabled = value.eq_ignore_ascii_case("true");
                crate::online_tb::set_enabled(enabled);
                out!(
                    "info string OnlineTB {}",
                    if enabled { "enabled" } else { "disabled" }
                );
            }
            #[cfg(not(feature = "online-tb"))]
            out!("info string OnlineTB requires --features online-tb");
        }
        _ => {
            #[cfg(feature = "spsa")]
            {
                if let Ok(v) = value.parse::<i32>()
                    && crate::tune::set_param(&name, v)
                {
                    out!("info string set {} = {}", name, v);
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
    // Same point of view as the network's line below, and the same refusal in check:
    // the static evaluation is not defined there.
    if pos.checkers == 0 {
        out!("PeSTO eval: {} cp (STM POV)", eval::evaluate(pos));
    } else {
        out!("PeSTO eval: N/A (in check)");
    }

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
    fn go_options_after_searchmoves_are_still_parsed() {
        let pos = Position::from_fen(STARTPOS).unwrap();
        let (limits, moves) = parse_go(
            &[
                "searchmoves",
                "e2e4",
                "d2d4",
                "movetime",
                "150",
                "depth",
                "12",
            ],
            &pos,
        );
        assert_eq!(moves.len(), 2);
        assert_eq!(limits.movetime, Some(150));
        assert_eq!(limits.depth, Some(12));
        assert!(!limits.infinite);
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
        for needle in [">> isready", "<< readyok"] {
            let line = text
                .lines()
                .find(|l| l.contains(needle))
                .unwrap_or_else(|| panic!("missing {needle} in: {text}"));
            let stamp = line.split_whitespace().next().expect("timestamp token");
            let (secs, millis) = stamp
                .split_once('.')
                .unwrap_or_else(|| panic!("stamp {stamp}"));
            assert!(
                secs.chars().all(|c| c.is_ascii_digit())
                    && millis.len() == 3
                    && millis.chars().all(|c| c.is_ascii_digit()),
                "expected s.mmm, got {line}"
            );
        }
    }

    #[test]
    fn debug_log_file_appends_across_reopen() {
        let _guard = lock_transcript();
        let path = std::env::temp_dir().join("gaiachess-uci-log-append-test.txt");
        let _ = std::fs::remove_file(&path);
        let p = path.to_str().expect("utf-8 temp path");
        crate::out::set_debug_log(p).unwrap();
        crate::out::log_input("first");
        crate::out::set_debug_log("").unwrap();
        crate::out::set_debug_log(p).unwrap();
        crate::out::log_input("second");
        crate::out::set_debug_log("").unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert!(text.contains(">> first"), "truncated first session: {text}");
        assert!(text.contains(">> second"), "missing second session: {text}");
        let sessions = text.lines().filter(|l| l.starts_with("# session ")).count();
        assert_eq!(sessions, 2, "expected two session markers: {text}");
    }

    /// A `SyzygyPath` is answered whatever it found: a wrong path used to load nothing
    /// without a word, its failure branch being dead. A network that fails to load says
    /// so too.
    #[cfg(feature = "syzygy")]
    #[test]
    fn options_that_load_say_what_they_loaded() {
        let _guard = lock_transcript();
        let text = transcript(
            "syzygy",
            &[
                "setoption name SyzygyPath value /nonexistent/gaiachess/syzygy",
                "setoption name EvalFile value /nonexistent/gaiachess/net.bin",
            ],
        );
        assert!(
            text.contains("Found 0 WDL and 0 DTZ tablebase files (up to 0-man)."),
            "{text}"
        );
        assert!(text.contains("Failed to load network"), "{text}");
        assert_eq!(crate::tb::max_pieces(), 0);
    }

    /// `go perft N` counts the move tree, as perft harnesses expect, instead of starting
    /// a search that only `stop` would end; Chess960 castles are written as the GUI
    /// writes them.
    #[test]
    fn go_perft_divides_instead_of_searching() {
        let _guard = lock_transcript();
        let text = transcript("perft", &["position startpos", "go perft 3"]);
        assert!(text.contains("<< Nodes: 8902"), "{text}");
        assert!(text.contains("<< e2e4: 600"), "{text}");
        assert!(!text.contains("bestmove"), "a search was started: {text}");

        let text = transcript(
            "perft960",
            &[
                "setoption name UCI_Chess960 value true",
                "position fen nrkbnqbr/pppppppp/8/8/8/8/PPPPPPPP/BBNRKQNR w HDhb - 0 1",
                "go perft 3",
                "position fen 4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1",
                "go perft 1",
            ],
        );
        assert!(text.contains("<< Nodes: 8544"), "{text}");
        assert!(text.contains("<< e1h1: 1"), "{text}");
        assert!(text.contains("<< e1a1: 1"), "{text}");
    }

    /// `bench` runs inside a session, at full strength whatever level the session plays
    /// at, and leaves that level as it found it. It used to be ignored without a word.
    #[test]
    fn bench_runs_in_the_session_and_keeps_its_level() {
        let _guard = lock_transcript();
        let text = transcript(
            "bench",
            &["setoption name Skill Level value 5", "bench 1", "isready"],
        );
        assert!(text.contains("<< readyok"), "{text}");
        assert_eq!(
            crate::skill::level(),
            5,
            "the session's level was not restored"
        );
        crate::skill::set(crate::skill::FULL_STRENGTH, 0);
    }

    /// The PeSTO line of `eval` is from the side to move, as its label says, and is not
    /// computed in check, where the static evaluation is undefined (a debug build used
    /// to panic there). The label used to say white's point of view.
    #[test]
    fn eval_prints_pesto_from_the_side_to_move() {
        let _guard = lock_transcript();
        let pesto = |text: &str| -> i32 {
            let line = text.lines().find(|l| l.contains("PeSTO eval:")).unwrap();
            assert!(line.contains("(STM POV)"), "{line}");
            line.split_whitespace().nth(4).unwrap().parse().unwrap()
        };
        // White is a rook down; with black to move the score is black's, so positive.
        let no_rook = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1";
        let white = transcript(
            "eval-w",
            &[&format!("position fen {no_rook} w Qkq - 0 1"), "eval"],
        );
        let black = transcript(
            "eval-b",
            &[&format!("position fen {no_rook} b Qkq - 0 1"), "eval"],
        );
        assert!(pesto(&white) < 0, "{white}");
        assert!(pesto(&black) > 0, "{black}");

        let check = transcript(
            "eval-check",
            &["position startpos moves e2e4 f7f6 d2d4 g7g5 d1h5", "eval"],
        );
        assert!(check.contains("PeSTO eval: N/A (in check)"), "{check}");
    }

    /// `<empty>`, the protocol's empty string and the option's advertised default, closes
    /// the log. It used to open a file called `<empty>` in the working directory.
    #[test]
    fn debug_log_file_empty_closes_the_log() {
        let _guard = lock_transcript();
        let path = std::env::temp_dir().join(format!(
            "gaiachess-uci-log-empty-{}.txt",
            std::process::id()
        ));
        crate::out::set_debug_log(path.to_str().expect("utf-8 temp path")).unwrap();
        let mut session = UciSession::new();
        session.command("setoption name Debug Log File value <empty>");
        crate::out::log_input("after the log was closed");
        crate::out::set_debug_log("").unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert!(
            !std::path::Path::new("<empty>").exists(),
            "a file named <empty> was created"
        );
        assert!(
            !text.contains("after the log"),
            "the log stayed open: {text}"
        );
    }

    /// The debug log and the display options are one set of globals for the whole
    /// process, and the harness runs tests in parallel: without this, one test's capture
    /// swallows another's output, or clears the log out from under it.
    static TRANSCRIPT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Held for a whole session: the transcript lock, and the skill level's.
    ///
    /// The level's lock is the one every multi-thread test takes, and those raise the
    /// process-wide STOP as they finish — enough to cut a session's search short and
    /// change its score. It also keeps a weakened level set by another test away from
    /// the session's searches.
    struct TranscriptGuard {
        _transcript: std::sync::MutexGuard<'static, ()>,
        _level: std::sync::MutexGuard<'static, ()>,
    }

    fn lock_transcript() -> TranscriptGuard {
        let transcript = TRANSCRIPT_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        TranscriptGuard {
            _transcript: transcript,
            _level: crate::skill::lock_level(),
        }
    }

    /// Drive a whole session and hand back everything it printed, by way of the debug
    /// log — the only place an `out!` line can be read back inside a test.
    ///
    /// The caller must hold [`lock_transcript`].
    fn transcript(name: &str, commands: &[&str]) -> String {
        let path =
            std::env::temp_dir().join(format!("gaiachess-uci-{name}-{}.txt", std::process::id()));
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

    /// The flags a test hands the reader and arms its searches with, in place of the
    /// process-wide STOP and PONDER.
    struct TestFlags {
        stop: AtomicBool,
        ponder: AtomicBool,
    }

    impl TestFlags {
        fn new() -> TestFlags {
            TestFlags {
                stop: AtomicBool::new(false),
                ponder: AtomicBool::new(false),
            }
        }

        fn arm(&self, go: u64, ponder: bool) {
            arm_flags(go, ponder, &self.stop, &self.ponder);
        }

        fn stopped(&self) -> bool {
            self.stop.load(Ordering::SeqCst)
        }

        fn pondering(&self) -> bool {
            self.ponder.load(Ordering::SeqCst)
        }
    }

    /// Puts the reader's bookkeeping back to what a session without a reader sees.
    fn reset_reader() {
        GO_READ.store(0, Ordering::SeqCst);
        GO_ARMED.store(0, Ordering::SeqCst);
        GO_DONE.store(0, Ordering::SeqCst);
        ORDERED_PENDING.store(0, Ordering::SeqCst);
        STOPPED_GOS.lock().unwrap().clear();
        HIT_GOS.lock().unwrap().clear();
    }

    /// Feeds `input` to the stdin reader's loop and hands back the lines it queued for
    /// the session.
    ///
    /// The caller must hold [`lock_transcript`]: the reader counts `go`, `stop` and
    /// `ponderhit` in the process-wide bookkeeping every session reads.
    fn read_input(input: &[u8], flags: &TestFlags) -> Vec<String> {
        let (tx, rx) = mpsc::channel();
        let reader_flags = ReaderFlags {
            stop: &flags.stop,
            ponder: &flags.ponder,
        };
        read_commands(input, &tx, &reader_flags);
        drop(tx);
        rx.iter().collect()
    }

    /// Hands the reader each line of `lines` as it would read them, but without the end
    /// of the input that follows them in [`read_input`].
    fn feed(flags: &TestFlags, lines: &str) -> Vec<String> {
        let (tx, rx) = mpsc::channel();
        let reader_flags = ReaderFlags {
            stop: &flags.stop,
            ponder: &flags.ponder,
        };
        for line in lines.lines() {
            assert!(read_line(line.to_owned(), &tx, &reader_flags));
        }
        rx.try_iter().collect()
    }

    /// A line that is not UTF-8 — a Windows GUI sending a name in its code page — costs
    /// a replacement character and nothing else. It used to end the reader: the engine
    /// quit, or sat in an infinite search that no `stop` could reach any more.
    #[test]
    fn a_line_that_is_not_utf8_does_not_end_the_reader() {
        let _guard = lock_transcript();
        let flags = TestFlags::new();
        let lines = read_input(
            b"setoption name UCI_Opponent value none 2500 human M\xfcller\r\nuci\nd",
            &flags,
        );
        assert_eq!(
            lines,
            [
                "setoption name UCI_Opponent value none 2500 human M\u{fffd}ller",
                "uci",
                "d"
            ]
        );
        reset_reader();
    }

    /// The end of the input stops the last search as a `quit` would, without queuing a
    /// line: `go infinite` or `go ponder` piped in and the pipe closed used to search for
    /// ever, as nothing was left to read their `stop`.
    #[test]
    fn the_end_of_the_input_stops_the_last_search() {
        let _guard = lock_transcript();
        reset_reader();
        let flags = TestFlags::new();

        // Read ahead of the session: the search is stopped as it is armed.
        let lines = read_input(b"position startpos\ngo infinite\n", &flags);
        assert_eq!(
            lines,
            ["position startpos", "go infinite"],
            "no quit is queued"
        );
        flags.arm(1, false);
        assert!(flags.stopped(), "the search was not stopped");

        // While the search runs: stopped on the spot.
        flags.arm(1, false);
        assert!(!flags.stopped());
        assert!(read_input(b"", &flags).is_empty());
        assert!(flags.stopped(), "the running search was not stopped");

        reset_reader();
    }

    /// A `stop` or `ponderhit` the reader saw before the session reached its `go`
    /// survives the arming of that search, and one meant for an earlier search does not
    /// leak into the next. Used to be erased: `go infinite` then `stop` sent together
    /// left the engine searching for ever.
    #[test]
    fn events_read_ahead_of_their_go_survive_the_arming() {
        // The bookkeeping is read by every session's `go`: keep the transcript tests out.
        let _guard = lock_transcript();
        reset_reader();
        let flags = TestFlags::new();

        // go, stop: both read before the session handles the go.
        feed(&flags, "go infinite\nstop\n");
        flags.arm(1, false);
        assert!(flags.stopped(), "the stop was meant for this search");

        // That stop belonged to search 1; search 2 starts clean.
        feed(&flags, "go infinite\n");
        flags.arm(2, false);
        assert!(!flags.stopped(), "an earlier search's stop leaked");

        // go ponder, ponderhit: the search starts on the clock.
        feed(&flags, "go ponder\nponderhit\n");
        flags.arm(3, true);
        assert!(!flags.pondering(), "the ponderhit was lost");
        assert!(!flags.stopped());

        // go ponder with no hit yet: pondering.
        feed(&flags, "go ponder\n");
        flags.arm(4, true);
        assert!(flags.pondering());

        reset_reader();
    }

    /// `isready` is answered by the reader while a `go` it read has no `bestmove` yet,
    /// unless a command that waits for that search is queued ahead of it. It used to be
    /// answered only while the search proper ran: too late when written with the `go`,
    /// and too early ahead of a queued `ucinewgame`.
    #[test]
    fn isready_is_answered_at_once_only_when_nothing_queued_waits() {
        let _guard = lock_transcript();
        reset_reader();
        let flags = TestFlags::new();

        // No search: the session answers it, in its turn.
        assert_eq!(feed(&flags, "isready"), ["isready"]);

        // Written with the go: answered at once, not queued.
        assert_eq!(
            feed(&flags, "position startpos\ngo infinite\nisready"),
            ["position startpos", "go infinite"]
        );

        // Behind a ucinewgame that waits for the search: after it, by the session.
        assert_eq!(
            feed(&flags, "stop\nucinewgame\nisready"),
            ["stop", "ucinewgame", "isready"]
        );
        reset_reader();

        // A plain position behind a ponder search holds nothing back.
        assert_eq!(
            feed(&flags, "go ponder\nposition startpos moves e2e4\nisready"),
            ["go ponder", "position startpos moves e2e4"]
        );

        // Once the bestmove is out, the session answers again.
        GO_DONE.store(GO_READ.load(Ordering::SeqCst), Ordering::SeqCst);
        assert_eq!(feed(&flags, "isready"), ["isready"]);
        reset_reader();
    }

    /// An event belongs to the `go` it was read after, and to no other. A `quit` read
    /// behind two queued searches used to stop both: the first one, which should have run
    /// to its depth, answered with the first move generated.
    #[test]
    fn an_event_read_ahead_stops_only_its_own_search() {
        let _guard = lock_transcript();
        reset_reader();
        let flags = TestFlags::new();

        // go, position, go, quit written in one go: only the second search is stopped.
        feed(
            &flags,
            "go depth 12\nposition startpos moves e2e4\ngo depth 12\nquit",
        );
        flags.arm(1, false);
        assert!(
            !flags.stopped(),
            "the quit stopped the search before its go"
        );
        flags.arm(2, false);
        assert!(flags.stopped(), "the quit did not stop its own search");
        reset_reader();

        // go, stop, go, stop: both searches are stopped.
        feed(&flags, "go infinite\nstop\ngo infinite\nstop\n");
        flags.arm(1, false);
        assert!(flags.stopped());
        flags.arm(2, false);
        assert!(flags.stopped());
        reset_reader();

        // A ponderhit read after the second go leaves the first one pondering.
        feed(&flags, "go ponder\ngo ponder\nponderhit\n");
        flags.arm(1, true);
        assert!(
            flags.pondering(),
            "the hit for search 2 ended search 1's ponder"
        );
        flags.arm(2, true);
        assert!(!flags.pondering(), "the hit for search 2 was lost");
        reset_reader();

        // A stop read after a go still queued leaves the running search alone.
        feed(&flags, "go infinite\n");
        flags.arm(1, false);
        feed(&flags, "go infinite\nstop\n");
        assert!(!flags.stopped(), "the stop for search 2 stopped search 1");
        flags.arm(2, false);
        assert!(flags.stopped());
        reset_reader();
    }

    /// `go infinite` keeps its move until `stop`, whatever ended the search; a ponder
    /// search until `ponderhit` or `stop`. Used to answer as soon as the 128 iterations
    /// of a short mate, or a level's few plies, were done.
    #[test]
    fn a_bestmove_is_held_until_the_interface_may_have_it() {
        // (ponder requested, infinite, PONDER, STOP) -> held
        assert!(
            must_hold_bestmove(false, true, false, false),
            "infinite waits for stop"
        );
        assert!(
            !must_hold_bestmove(false, true, false, true),
            "stop releases infinite"
        );
        assert!(
            must_hold_bestmove(true, false, true, false),
            "ponder waits for its hit"
        );
        assert!(
            !must_hold_bestmove(true, false, false, false),
            "ponderhit releases it"
        );
        assert!(
            !must_hold_bestmove(true, false, true, true),
            "stop releases a ponder"
        );
        assert!(
            must_hold_bestmove(true, true, false, false),
            "a hit does not end infinite"
        );
        assert!(
            !must_hold_bestmove(false, false, false, false),
            "a bounded go never waits"
        );
        assert!(
            !must_hold_bestmove(false, false, true, false),
            "PONDER alone means nothing"
        );
    }

    /// A game longer than the 1024 saved states — 1372 plies here — is parsed and
    /// searched: `position ... moves` used to overflow the history and abort.
    #[test]
    fn a_game_of_more_than_a_thousand_plies_is_parsed_and_searched() {
        let game = crate::position::tests::long_game(14, 24);
        assert!(game.len() > 1100);
        let mut tokens = vec!["startpos", "moves"];
        tokens.extend(game.iter().map(String::as_str));
        let mut pos = Position::from_fen(STARTPOS).unwrap();
        parse_position(&tokens, &mut pos, false);
        assert_eq!(
            pos.fullmove_number as usize,
            1 + game.len() / 2,
            "every move was played"
        );
        assert!(pos.ply < 512, "history still {} states deep", pos.ply);

        let _guard = crate::skill::lock_level();
        let mut pool = crate::threads::ThreadPool::new(1, 1);
        pool.threads[0].silent = true;
        STOP.store(false, Ordering::Relaxed);
        let (best, _) = pool.start_search(&pos, SearchLimits::depth(6));
        assert!(best.is_ok());
    }

    /// Full strength after a weakened level searches as a fresh engine would: nothing
    /// the weak level stored in the tables or the histories is read back.
    #[test]
    fn a_change_of_level_leaves_nothing_behind() {
        let _guard = lock_transcript();
        let position =
            "position fen r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R1BQ1RK1 w - - 0 8";
        // The deterministic part of the last depth-9 line: score, nodes and PV.
        let last_line = |text: &str| -> String {
            let line = text
                .lines()
                .filter(|l| l.contains("info depth 9 ") && l.contains(" pv "))
                .last()
                .unwrap_or_else(|| panic!("no depth 9 line in: {text}"));
            let words: Vec<&str> = line.split_whitespace().collect();
            let field = |name: &str, len: usize| {
                let i = words.iter().position(|w| *w == name).expect(name);
                words[i + 1..i + 1 + len].join(" ")
            };
            let pv = line.split(" pv ").nth(1).unwrap_or_default();
            format!("{} {} {pv}", field("score", 2), field("nodes", 1))
        };
        let after_weak = transcript(
            "level-weak",
            &[
                "setoption name NormalizeScore value false",
                "setoption name Skill Level value 5",
                position,
                "go depth 4",
                "setoption name Skill Level value 20",
                position,
                "go depth 9",
            ],
        );
        let fresh = transcript(
            "level-fresh",
            &[
                "setoption name NormalizeScore value false",
                "setoption name Skill Level value 20",
                position,
                "go depth 9",
            ],
        );
        crate::skill::set(crate::skill::FULL_STRENGTH, 0);
        assert_eq!(last_line(&after_weak), last_line(&fresh));
    }

    #[test]
    fn the_two_display_options_are_read_back() {
        let _guard = lock_transcript();
        let mut pool = crate::threads::ThreadPool::new(1, 16);
        let mut ponder = true;
        let mut chess960 = false;
        let mut set = |name: &str, value: &str| {
            parse_setoption(
                &["name", name, "value", value],
                &mut pool,
                &mut ponder,
                &mut chess960,
            );
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
    fn optional_search_line_options_are_read_back() {
        let _guard = lock_transcript();
        let mut pool = crate::threads::ThreadPool::new(1, 16);
        let mut ponder = true;
        let mut chess960 = false;
        let mut set = |name: &str, value: &str| {
            parse_setoption(
                &["name", name, "value", value],
                &mut pool,
                &mut ponder,
                &mut chess960,
            );
        };

        set("UCI_ShowCurrLine", "true");
        set("UCI_ShowRefutations", "true");
        assert!(show_curr_line());
        assert!(show_refutations());

        set("UCI_ShowCurrLine", "false");
        set("UCI_ShowRefutations", "false");
        assert!(!show_curr_line());
        assert!(!show_refutations());
    }

    #[cfg(feature = "nalimov")]
    #[test]
    fn nalimov_cache_option_changes_the_configured_size() {
        let _guard = lock_transcript();
        let mut pool = crate::threads::ThreadPool::new(1, 16);
        let mut ponder = true;
        let mut chess960 = false;

        parse_setoption(
            &["name", "NalimovCache", "value", "64"],
            &mut pool,
            &mut ponder,
            &mut chess960,
        );
        assert_eq!(crate::nalimov::cache_size_mb(), 64);

        // Process-global option: restore the advertised default for the other tests.
        assert!(crate::nalimov::set_cache_size_mb(256));
    }

    #[test]
    fn show_wdl_adds_three_probabilities_before_the_node_count() {
        let _guard = lock_transcript();
        let text = transcript(
            "wdl",
            &[
                "setoption name UCI_ShowWDL value true",
                "position fen rnbqkbnr/ppp1pppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "go depth 4",
                "setoption name UCI_ShowWDL value false",
            ],
        );
        let line = text
            .lines()
            .find(|line| line.contains(" score cp "))
            .unwrap_or_else(|| panic!("no info line in: {text}"));

        let wdl = line
            .split(" wdl ")
            .nth(1)
            .unwrap_or_else(|| panic!("no wdl in: {line}"));
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
        assert!(
            !show_wdl(),
            "the option must be back off for the other tests"
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn native_search_info_reports_cpu_load() {
        let _guard = lock_transcript();
        let text = transcript(
            "cpuload",
            &[
                "position fen rnbqkbnr/ppp1pppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "go depth 4",
            ],
        );
        let line = text
            .lines()
            .find(|line| line.contains(" cpuload "))
            .unwrap_or_else(|| panic!("no cpuload in: {text}"));
        let load: u32 = line
            .split(" cpuload ")
            .nth(1)
            .and_then(|tail| tail.split_whitespace().next())
            .expect("cpuload value")
            .parse()
            .expect("cpuload integer");
        assert!(load <= 1000, "line: {line}");
    }

    /// The option is display only: the same search prints two different numbers for one
    /// score, and turning it off gives the search's own unit back untouched.
    #[test]
    fn normalizing_only_rescales_what_is_printed() {
        let _guard = lock_transcript();
        let position = "position fen rnbqkbnr/ppp1pppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        // Both sides are set explicitly, so the test says what it means whichever way the
        // default happens to point.
        let raw = last_score(&transcript(
            "raw",
            &[
                "setoption name NormalizeScore value false",
                position,
                "go depth 4",
            ],
        ));
        let normalized = last_score(&transcript(
            "norm",
            &[
                "setoption name NormalizeScore value true",
                position,
                "go depth 4",
            ],
        ));

        let pos =
            Position::from_fen("rnbqkbnr/ppp1pppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        assert_eq!(normalized, crate::wdl::to_cp(raw, &pos));
        assert_ne!(
            normalized, raw,
            "the two scales must not coincide on this position"
        );
    }

    #[test]
    fn four_field_fen_search_plays_a_black_move() {
        // Searches: a weakened level or a STOP raised by another test must not reach it.
        let _guard = crate::skill::lock_level();
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

    /// A clock run past zero is a clock with nothing left, not a clock missing: `go
    /// wtime -50` used to search until `stop`.
    #[test]
    fn a_negative_time_is_no_time_left() {
        let pos = Position::from_fen(STARTPOS).unwrap();
        let (limits, _) = parse_go(&["wtime", "-50", "btime", "1000"], &pos);
        assert_eq!(limits.time, Some(0));
        assert!(!limits.infinite);

        let (limits, _) = parse_go(&["movetime", "-5"], &pos);
        assert_eq!(limits.movetime, Some(0));
        assert!(!limits.infinite);

        // Black to move reads its own clock.
        let black = Position::from_fen("4k3/8/8/8/8/8/8/4K3 b - - 0 1").unwrap();
        let (limits, _) = parse_go(&["wtime", "1000", "btime", "-1", "binc", "20"], &black);
        assert_eq!((limits.time, limits.inc), (Some(0), 20));
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

    /// A long iteration used to be silent from one PV line to the next — seconds
    /// and tens of millions of nodes in the reported analysis sessions, which the
    /// interface showed as a frozen engine. With the tolerated silence cut to a
    /// millisecond, a short fixed-node search must show progress lines between its
    /// PV lines: depth and counters, no score, no PV.
    #[test]
    fn a_silent_iteration_reports_its_progress() {
        let _guard = lock_transcript();
        crate::threads::HEARTBEAT_MS.store(1, std::sync::atomic::Ordering::Relaxed);
        let text = transcript("heartbeat", &["position startpos", "go nodes 300000"]);
        crate::threads::HEARTBEAT_MS.store(1000, std::sync::atomic::Ordering::Relaxed);
        let output: Vec<&str> = text
            .lines()
            .filter_map(|line| line.split_once(" << ").map(|(_, rest)| rest))
            .collect();
        let heartbeats: Vec<&&str> = output
            .iter()
            .filter(|line| {
                line.starts_with("info depth ")
                    && !line.contains(" score ")
                    && !line.contains(" pv ")
                    && line.contains(" nodes ")
                    && line.contains(" time ")
            })
            .collect();
        assert!(!heartbeats.is_empty(), "no progress line in: {text}");
        for line in heartbeats {
            let depth: i32 = line
                .split_whitespace()
                .nth(2)
                .and_then(|d| d.parse().ok())
                .unwrap_or_else(|| panic!("no depth in {line}"));
            assert!(
                depth >= 1,
                "progress line before the first iteration: {line}"
            );
        }
        let final_info = output
            .iter()
            .rev()
            .find(|line| line.starts_with("info depth "))
            .expect("final info");
        assert!(
            final_info.contains(" pv "),
            "the last line before bestmove must still be the PV snapshot: {text}"
        );
    }

    #[test]
    fn interrupted_aspiration_is_followed_by_exact_final_info() {
        let _guard = lock_transcript();
        // David's queen endgames exposed this protocol bug: if the node limit landed
        // during an aspiration re-search, the old engine left a transient bound as the
        // last line before bestmove.  This position/budget deterministically interrupts
        // such a re-search, so the penultimate search line must be a bound while the
        // final snapshot must restore the last completed PV.
        let text = transcript(
            "final-info",
            &[
                "position fen 4K3/3Q4/2k5/2p5/1pPb4/1P6/8/8 b - - 7 70",
                "go nodes 100000",
            ],
        );
        let output: Vec<&str> = text
            .lines()
            .filter_map(|line| line.split_once(" << ").map(|(_, rest)| rest))
            .collect();
        let best_idx = output
            .iter()
            .position(|line| line.starts_with("bestmove "))
            .unwrap_or_else(|| panic!("no bestmove in: {text}"));
        assert!(best_idx > 0, "bestmove was the first output: {text}");
        assert!(
            output[best_idx - 1].starts_with("info "),
            "bestmove was not preceded by info: {text}"
        );
        let final_info = output[best_idx - 1];
        assert!(
            final_info.contains(" score ") && final_info.contains(" pv "),
            "final info did not contain a score and a complete PV: {text}"
        );
        assert!(
            !final_info.contains("lowerbound") && !final_info.contains("upperbound"),
            "final info repeated a transient aspiration bound: {text}"
        );

        let search_info = &output[..best_idx - 1];
        assert!(
            search_info
                .iter()
                .any(|line| line.contains("lowerbound") || line.contains("upperbound")),
            "test did not actually interrupt an aspiration re-search: {text}"
        );

        let pv_move = final_info
            .split(" pv ")
            .nth(1)
            .and_then(|pv| pv.split_whitespace().next())
            .expect("final PV move");
        let best_move = output[best_idx]
            .split_whitespace()
            .nth(1)
            .expect("bestmove move");
        assert_eq!(pv_move, best_move, "final PV and bestmove disagree: {text}");

        // And it is the deepest iteration that was completed: never a shallower
        // line dug up from an earlier iteration (a real game was lost that way).
        let depth_of = |line: &str| -> i32 {
            line.split_whitespace()
                .nth(2)
                .and_then(|d| d.parse().ok())
                .unwrap_or(0)
        };
        let deepest_exact = search_info
            .iter()
            .filter(|line| {
                line.starts_with("info depth ")
                    && line.contains(" pv ")
                    && !line.contains("lowerbound")
                    && !line.contains("upperbound")
            })
            .map(|line| depth_of(line))
            .max()
            .expect("an exact iteration");
        assert_eq!(
            depth_of(final_info),
            deepest_exact,
            "final info is not the deepest completed iteration: {text}"
        );
    }
}
