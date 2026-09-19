//! Thread pool and per-thread state for [Lazy SMP](https://www.chessprogramming.org/Lazy_SMP)
//! parallel search.
//!
//! Each thread runs iterative deepening independently on the same root position.
//! The only shared state is the lockless [transposition table](crate::tt::TT)
//! and the atomic stop flag. All history tables, killers, search stack, and PV
//! are per-thread.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU32, AtomicU64, AtomicUsize, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc;

use crate::history::{
    ButterflyHistory, CaptureHistory, ContCorrectionHistory, ContinuationHistory, Countermoves,
    Killers, PawnHistory, PieceToCorrTable, PieceToTable, SharedCorrectionHistory,
};
use crate::nnue;

/// Global stop flag, accessible from any thread (stdin reader, search, etc.).
///
/// Using a global static avoids the need for `Arc<AtomicBool>` plumbing.
/// Using a global static avoids `Arc<AtomicBool>` plumbing.
pub static STOP: AtomicBool = AtomicBool::new(false);

/// Global ponder flag. Set when `go ponder` is active.
/// Cleared by `ponderhit` (stdin reader thread) to signal the transition
/// from ponder mode to normal search with real time limits.
pub static PONDER: AtomicBool = AtomicBool::new(false);

/// True while `start_search` is running. The stdin reader answers `isready`
/// immediately in that window, without waiting for the search to finish.
pub static SEARCHING: AtomicBool = AtomicBool::new(false);

struct SearchingGuard;
impl Drop for SearchingGuard {
    fn drop(&mut self) {
        SEARCHING.store(false, Ordering::Relaxed);
    }
}

/// How many watchers are reading how fast the search goes. At zero — which is how every
/// ordinary search runs — [`ThreadData::check_limits`] does not touch the counter below
/// at all, so nobody pays for a figure nobody is reading.
///
/// A count and not a flag, because more than one watcher can be alive at once: the
/// interface builds the next screen before dropping the one it replaces, so two engines
/// briefly overlap, and a test suite runs several side by side. With a flag, whichever
/// watcher closed first turned the counter off under everyone else, and the rate they
/// were reading silently went to nothing.
static NODE_WATCHERS: AtomicUsize = AtomicUsize::new(0);

/// Starts reading the node counter. Balance with [`close_node_meter`].
pub fn open_node_meter() {
    NODE_WATCHERS.fetch_add(1, Ordering::Relaxed);
}

/// Stops reading it. Saturating: an unbalanced close must not wrap the count into
/// "watched by four billion", which no later close could undo.
pub fn close_node_meter() {
    let _ = NODE_WATCHERS.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
        Some(n.saturating_sub(1))
    });
}

/// Whether anyone is reading the node counter.
#[inline]
pub fn counting_nodes() -> bool {
    NODE_WATCHERS.load(Ordering::Relaxed) > 0
}

/// Nodes searched, to the nearest 512, by every thread while anyone is watching.
/// Read by the bench monitor and by the interface's rate display; it only ever climbs,
/// so a reader wanting a rate takes the difference over a window of its own.
pub static NODES_SEARCHED: AtomicU64 = AtomicU64::new(0);

/// Milliseconds into the search at which the main thread last printed anything.
///
/// A deep iteration can run for seconds without completing — tens of millions of
/// nodes inside one root move in the reported analysis sessions — and an interface
/// that hears nothing shows a frozen engine. The main thread therefore reports its
/// progress on its own once [`HEARTBEAT_MS`] have passed in silence.
pub static LAST_INFO_MS: AtomicU64 = AtomicU64::new(0);

/// The silence the main thread tolerates before reporting progress, in ms.
pub static HEARTBEAT_MS: AtomicU64 = AtomicU64::new(1000);

/// Record that a line was just printed, `elapsed_ms` into the search.
#[inline]
pub fn note_output(elapsed_ms: u64) {
    LAST_INFO_MS.store(elapsed_ms, Ordering::Relaxed);
}

/// Whether the silence since `last_ms` has reached `interval_ms` at `now_ms`.
#[inline]
pub fn heartbeat_due(now_ms: u64, last_ms: u64, interval_ms: u64) -> bool {
    now_ms >= last_ms.saturating_add(interval_ms)
}

/// Whether whoever is hosting the engine has asked for the search to end.
///
/// Deliberately not part of [`should_stop`], which every node calls: an imported function
/// there would cost more than the search saves. [`ThreadData::check_limits`] asks instead,
/// on its 512-node tick.
#[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
#[inline]
fn host_wants_stop() -> bool {
    #[link(wasm_import_module = "env")]
    unsafe extern "C" {
        /// Non-zero once the host wants the current search abandoned.
        fn gaia_host_stop() -> i32;
    }
    unsafe { gaia_host_stop() != 0 }
}

/// Nothing asks on a target that has no host to ask. Constant-folded away, so the search
/// is the same instruction for instruction and the bench node count cannot move.
#[cfg(not(all(target_arch = "wasm32", not(target_os = "wasi"))))]
#[inline(always)]
fn host_wants_stop() -> bool {
    false
}

/// Check if search should stop.
#[inline]
pub fn should_stop() -> bool {
    STOP.load(Ordering::Relaxed)
}
use crate::movegen;
use crate::position::Position;
use crate::search;
use crate::timeman::{INITIAL_TIME_REDUCTION, SearchLimits, TIME_ADJUST_UNSET, TimeManager};
use crate::tt::TT;
use crate::types::*;

/// Number of sentinel entries before ply 0 in the search stack.
/// Allows safe access to `stack[ply - N]` for continuation history lookups.
pub const SS_OFFSET: usize = 8;

const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/// One-writer lock-free snapshot consumed only by the UCI reporting thread.
struct CurrLineSlot {
    generation: AtomicU32,
    len: AtomicUsize,
    moves: [AtomicU16; MAX_PLY + 1],
}

impl CurrLineSlot {
    fn new() -> Self {
        Self {
            generation: AtomicU32::new(0),
            len: AtomicUsize::new(0),
            moves: [const { AtomicU16::new(0) }; MAX_PLY + 1],
        }
    }

    fn publish(&self, line: &[Move]) {
        self.generation.fetch_add(1, Ordering::SeqCst);
        let len = line.len().min(self.moves.len());
        for (dst, mv) in self.moves.iter().zip(line).take(len) {
            dst.store(mv.0, Ordering::SeqCst);
        }
        self.len.store(len, Ordering::SeqCst);
        self.generation.fetch_add(1, Ordering::SeqCst);
    }

    fn clear(&self) {
        self.generation.fetch_add(1, Ordering::SeqCst);
        self.len.store(0, Ordering::SeqCst);
        self.generation.fetch_add(1, Ordering::SeqCst);
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn snapshot(&self) -> Option<(u32, Vec<Move>)> {
        for _ in 0..4 {
            let before = self.generation.load(Ordering::SeqCst);
            if before & 1 != 0 {
                continue;
            }
            let len = self.len.load(Ordering::SeqCst).min(self.moves.len());
            let line: Vec<Move> = self.moves[..len]
                .iter()
                .map(|mv| Move(mv.load(Ordering::SeqCst)))
                .collect();
            let after = self.generation.load(Ordering::SeqCst);
            if before == after && !line.is_empty() {
                return Some((after, line));
            }
        }
        None
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct CurrLineReporter {
    stop: Option<mpsc::Sender<()>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl CurrLineReporter {
    fn start(threads: &[ThreadData]) -> Option<Self> {
        if threads.first().is_none_or(|td| !td.report_curr_line) {
            return None;
        }

        let slots: Vec<_> = threads.iter().map(|td| Arc::clone(&td.curr_line)).collect();
        let chess960 = threads[0].pos.chess960;
        let (stop, stopped) = mpsc::channel();
        let handle = std::thread::Builder::new()
            .name("uci-currline".into())
            .spawn(move || {
                let mut generations = vec![0; slots.len()];
                loop {
                    match stopped.recv_timeout(std::time::Duration::from_secs(1)) {
                        Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
                        Err(mpsc::RecvTimeoutError::Timeout) => {}
                    }

                    let Some(snapshots) = slots
                        .iter()
                        .map(|slot| slot.snapshot())
                        .collect::<Option<Vec<_>>>()
                    else {
                        continue;
                    };
                    if snapshots
                        .iter()
                        .zip(&generations)
                        .all(|((generation, _), previous)| generation == previous)
                    {
                        continue;
                    }

                    let mut lines = Vec::with_capacity(snapshots.len());
                    for (i, (generation, line)) in snapshots.iter().enumerate() {
                        generations[i] = *generation;
                        let moves = line
                            .iter()
                            .map(|mv| mv.to_uci_960(chess960))
                            .collect::<Vec<_>>()
                            .join(" ");
                        if snapshots.len() == 1 {
                            lines.push(format!("info currline {moves}"));
                        } else {
                            lines.push(format!("info currline {} {moves}", i + 1));
                        }
                    }
                    crate::out::lines(&lines);
                }
            })
            .expect("failed to spawn UCI currline reporter");
        Some(Self {
            stop: Some(stop),
            handle: Some(handle),
        })
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for CurrLineReporter {
    fn drop(&mut self) {
        self.stop.take();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

// ============================================================
// RootMove
// ============================================================

/// Per-root-move state for Multi-PV search.
/// Each legal root move stores its own score, PV, and aspiration average.
#[derive(Clone)]
pub struct RootMove {
    pub mv: Move,
    /// Score of the last completed iteration in which this move was the line's
    /// best (-SCORE_INFINITE while it never was). Never a bound: only a settled
    /// aspiration loop writes it, and a stopped iteration writes nothing.
    pub score: i32,
    /// The iteration `score` comes from (0 while the move was never scored). The
    /// root moves are kept sorted by depth first, so `root_moves[0]` is always the
    /// deepest completed result — the move played and the line reported when the
    /// search stops in the middle of an iteration.
    pub depth: i32,
    /// The score this move had before `score` was last written.
    pub previous_score: i32,
    /// Running average score for aspiration window centering.
    pub avg_score: i32,
    /// Principal variation starting with this move.
    pub pv: [Move; MAX_PLY + 1],
    pub pv_len: usize,
    /// Selective depth reached during this PV's search.
    pub sel_depth: i32,
    /// Pyrrhic TB rank (higher = better, 0 = unranked). Set by root DTZ/WDL probe.
    #[cfg_attr(not(feature = "syzygy"), allow(dead_code))]
    pub tb_rank: i32,
    /// TB score for UCI display (graduated, not hardcoded ±20000).
    pub tb_score: i32,
    /// Cumulative nodes spent searching this move (for node fraction TM).
    pub nodes_spent: u64,
    /// Root move followed by the reply line already found while searching it.
    pub refutation: [Move; MAX_PLY + 1],
    pub refutation_len: usize,
}

impl RootMove {
    pub fn new(mv: Move) -> Self {
        let mut pv = [Move::NONE; MAX_PLY + 1];
        pv[0] = mv;
        let mut refutation = [Move::NONE; MAX_PLY + 1];
        refutation[0] = mv;
        RootMove {
            mv,
            score: -SCORE_INFINITE,
            depth: 0,
            previous_score: -SCORE_INFINITE,
            avg_score: SCORE_NONE,
            pv,
            pv_len: 1,
            sel_depth: 0,
            tb_rank: 0,
            tb_score: 0,
            nodes_spent: 0,
            refutation,
            refutation_len: 1,
        }
    }
}

/// Syzygy tablebase configuration for a search.
#[cfg(feature = "syzygy")]
#[derive(Clone, Copy, Default)]
pub struct TbConfig {
    /// Root position is in loaded tablebases.
    pub root_in_tb: bool,
    /// Max piece count for in-tree TB probing. 0 = skip in-tree probes (DTZ at
    /// the root, or a loss/draw). >0 = WDL-only win: probe below TB_LARGEST so a
    /// 7-piece WDL-only root does not WDL-hit every node.
    pub cardinality: u32,
}

// ============================================================
// StackEntry
// ============================================================

/// Per-ply search stack entry.
/// Stores context needed by continuation history and other ply-relative features.
#[derive(Clone, Copy)]
pub struct StackEntry {
    pub played_move: Move,
    pub moved_piece: Piece,
    pub is_capture: bool,
    pub conthist_ptr: *mut PieceToTable,
    pub cont_corr_ptr: *mut PieceToCorrTable,
    pub static_eval: i32,
    pub correction_value: i32,
    pub excluded: Move,
    pub double_extensions: i32,
    pub reduction: i32,
    /// Follow-PV guard: true if this node is on the previous iteration's PV
    pub follow_pv: bool,
    /// Beta cutoffs seen at this ply since the parent entered its move loop.
    /// The parent zeroes it once; every child searched from there shares the
    /// slot, so it counts how often this ply keeps refuting the parent's moves.
    pub failhighs: u8,
}

impl Default for StackEntry {
    fn default() -> Self {
        StackEntry {
            played_move: Move::NONE,
            moved_piece: Piece::NONE,
            is_capture: false,
            conthist_ptr: std::ptr::null_mut(),
            cont_corr_ptr: std::ptr::null_mut(),
            static_eval: SCORE_NONE,
            correction_value: 0,
            excluded: Move::NONE,
            double_extensions: 0,
            reduction: 0,
            follow_pv: false,
            failhighs: 0,
        }
    }
}

// SAFETY: Raw pointers in conthist_ptr point into the same thread's
// ContinuationHistory. Each thread owns its own data; no crossing.
unsafe impl Send for StackEntry {}

// ============================================================
// SharedState
// ============================================================

/// State shared between all search threads.
///
/// The TT is lockless (races accepted, verified by key16), and so are the
/// correction histories: an estimator of eval error indexed by pawn structure has
/// no reason to differ from one thread to the next, so the threads pool their
/// observations instead of each learning the same ones over again.
/// The stop flag is the global [`STOP`] atomic (not in this struct).
pub struct SharedState {
    pub tt: TT,
    pub corr: SharedCorrectionHistory,
    cpu_load: crate::cpuload::CpuLoad,
    /// Root best-move changes of the current search, summed over every thread.
    /// Time management reads it: a root that keeps changing its mind is a root
    /// worth more time, and instability seen by any thread counts. The main
    /// thread takes and zeroes it once per iteration; a root best-move change is
    /// rare enough that the atomic is never contended.
    pub best_move_changes: AtomicU32,
}

impl SharedState {
    /// `threads` sizes the correction tables so the entries per thread, and with
    /// them the collision rate, stay what they are at one thread. Every caller that
    /// searches alone passes 1.
    pub fn new(hash_mb: usize, threads: usize) -> Self {
        SharedState {
            tt: TT::new(hash_mb),
            corr: SharedCorrectionHistory::new(threads),
            cpu_load: crate::cpuload::CpuLoad::new(),
            best_move_changes: AtomicU32::new(0),
        }
    }

    pub fn cpu_load(&self, threads: usize) -> Option<u32> {
        self.cpu_load.permille(threads)
    }

    /// Clear everything shared (`ucinewgame`, and between bench runs).
    pub fn clear(&mut self) {
        self.tt.clear();
        self.corr.clear();
        self.best_move_changes.store(0, Ordering::Relaxed);
    }
}

// ============================================================
// ThreadData
// ============================================================

/// Per-thread search state.
///
/// Each thread owns a full copy of the position, all history tables,
/// killers, search stack, and PV table. Only the TT and stop flag
/// are shared (via `&SharedState`).
pub struct ThreadData {
    pub id: usize,
    /// Keeps this thread from writing to stdout. Thread 0 owns the clock, the root
    /// reporting and the tablebase lookups; a caller that wants those but not the
    /// output — a graphical interface driving the engine in-process — sets this rather
    /// than taking a helper's id, which would silently cost it the time management too.
    pub silent: bool,
    pub pos: Position,
    pub nodes: u64,
    /// Quiescence search node counter (diagnostic).
    pub qs_nodes: u64,
    pub history: ButterflyHistory,
    pub pawn_history: PawnHistory,
    pub cap_history: CaptureHistory,
    pub cont_history: ContinuationHistory,
    pub countermoves: Countermoves,
    /// Continuation correction stays per-thread: its entries are the `[piece][to]`
    /// subtable the current line points at, so every thread searching the same
    /// subtree would hammer the same few cache lines. The reference that shared it
    /// reverted the change for exactly that.
    pub cont_correction: ContCorrectionHistory,
    pub killers: Killers,
    /// Triangular PV table: `pv[ply][i]` stores the PV from ply onwards.
    pub pv: [[Move; MAX_PLY + 1]; MAX_PLY + 1],
    /// Length of PV at each ply.
    pub pv_len: [usize; MAX_PLY + 1],
    /// Root aspiration window width, for PV LMR scaling.
    pub root_delta: i32,
    /// Search stack with SS_OFFSET sentinel entries before ply 0.
    pub stack: [StackEntry; MAX_PLY + SS_OFFSET + 1],
    /// Sentinel (zeroed) subtable for stack entries that have no move context.
    pub sentinel_conthist: Box<PieceToTable>,
    /// NNUE network state (accumulator stack, per-thread).
    pub nnue: nnue::Network,
    /// Time manager for this thread (real limits for main, infinite for helpers).
    pub tm: TimeManager,
    /// Current iterative deepening depth (for SE ply guard).
    pub root_depth: i32,
    /// Lowest ply at which null-move pruning may run during an NMP verification
    /// search. Zero outside verification. This is per-thread because helpers search
    /// independent trees.
    pub nmp_min_ply: usize,
    /// Mate hunt: the root is deep and its value decisive-sized, so the
    /// iteration is refining a forced line rather than weighing evaluations.
    /// Singular extensions are off (every TT move looks singular when the
    /// alternatives are all mates) and reverse futility stays shallow.
    /// Recomputed before every aspiration retry; see `search.rs`.
    pub seek_mate: bool,
    /// Deepest iteration completed by this thread.
    pub completed_depth: i32,
    /// Best move found by this thread.
    pub best_move: Move,
    /// Best score found by this thread.
    pub best_score: i32,
    /// Per-thread search deadline in milliseconds (0 = no limit).
    /// Used by datagen to cap explosion-prone searches without global STOP.
    pub search_deadline: u64,
    /// How many threads the pool this thread belongs to holds, so a fixed node
    /// budget can be divided evenly. Left at 1 by every caller that searches
    /// alone: datagen, bench, the interface worker, the tests.
    pub thread_count: usize,
    /// Set when a fixed node budget was split across the pool. This thread then
    /// owns its share and ends on it alone, never through the global stop flag.
    pub per_thread_nodes: bool,
    /// Per-thread stopped flag, set when search_deadline is exceeded.
    pub stopped: bool,
    /// Maximum ply reached during current depth iteration (for UCI seldepth).
    pub seldepth: i32,
    /// The root move being searched and its rank in the root loop, for the
    /// progress line printed during a long iteration.
    pub root_move: Move,
    pub root_move_number: usize,
    /// Local cache of ponder state. Avoids atomic load every 512 nodes.
    /// Set in `prepare_search()`, cleared on `ponderhit` detection.
    pub pondering: bool,
    /// `go mate N`: stop iterative deepening once a mate in at most N moves is found.
    pub mate_limit: Option<i32>,
    /// Root moves with per-move scoring for Multi-PV.
    pub root_moves: Vec<RootMove>,
    /// Number of PV lines to search (UCI MultiPV option, default 1).
    pub multi_pv: usize,
    /// How many principal variations the caller actually asked to be told about.
    ///
    /// A weakened level searches more of them than that, so it has something to choose
    /// between (see `skill::variety_pick`); that is its own business and not something to
    /// report to an interface that asked for one line.
    pub reported_multi_pv: usize,
    /// Index of the PV line currently being searched (0-based).
    pub pv_index: usize,
    /// Tablebase hit counter (reported as `tbhits` in UCI info).
    pub tb_hits: u64,
    /// Syzygy TB config for this search.
    #[cfg(feature = "syzygy")]
    pub tb_config: TbConfig,
    /// Datagen mode: skip in-search TB probing and root TB ranking.
    /// TB adjudication in datagen handles endgame labels separately.
    #[allow(dead_code)] // used with --features datagen
    pub datagen_mode: bool,
    /// Per-game tilt of the time budget, computed on the first move that carries
    /// a clock and kept for the rest of the game. Negative means "not computed".
    pub original_time_adjust: f64,
    /// Score and running average this thread ended its *previous* search on, and
    /// the time reduction it left. Time management compares the position in front
    /// of it to the one it just played: a score that fell since last move buys
    /// more time. `SCORE_NONE` where there is no previous move to compare to.
    pub best_previous_score: i32,
    pub best_previous_avg_score: i32,
    pub previous_time_reduction: f64,
    /// Set by evaluate_pos() when the lazy-eval gate fired (PeSTO used instead of NNUE).
    /// Read immediately afterward to skip correction history application/update.
    pub used_lazy_eval: bool,
    /// The handicap in force, read once when the search starts. Inert at full strength,
    /// where every test on it is a `false` bool and costs nothing. Taken as a copy so
    /// the hot loops never touch an atomic, and so a level changed mid-search cannot
    /// make the engine two different opponents in one move.
    pub skill: crate::skill::Snapshot,
    /// Optimism bonus indexed by Color. Computed at root from running avg score.
    /// Positive = side is optimistic (root scores well), blended into eval.
    pub optimism: [i32; 2],
    /// Follow-PV guard: PV from the previous iteration
    pub last_iteration_pv: [Move; MAX_PLY + 1],
    pub last_iteration_pv_len: usize,
    curr_line: Arc<CurrLineSlot>,
    /// Snapshotted once per search; publishing uses atomics but never waits.
    pub report_curr_line: bool,
    /// Main-thread-only capture of already searched root replies.
    pub report_refutations: bool,
    /// Search statistics (feature `stats`). Boxed to keep ThreadData small.
    /// Written only by the owning thread; aggregated by the pool after join.
    #[cfg(feature = "stats")]
    pub stats: Box<crate::stats::SearchStats>,
    /// Search tree recorder (feature `tree`). Installed by the dump runner;
    /// `None` during normal play so recording costs nothing.
    #[cfg(feature = "tree")]
    pub tree: Option<Box<crate::tree::TreeRec>>,
}

/// Divides a fixed node budget into one share per thread; `None` for every other
/// kind of limit.
///
/// `go nodes N` keeps meaning about N nodes for this move whatever the thread count,
/// and each thread gets a share it can finish on its own. That second half is the
/// point: with the budget on the main thread only, the helpers ran until it stopped
/// them, so the work actually done per move depended on how busy the machine was —
/// and two binaries measured at different moments were not comparable. A clock is
/// left alone: there the main thread owns the budget and the helpers search beside it.
fn split_node_limit(limits: &SearchLimits, threads: usize) -> Option<SearchLimits> {
    debug_assert!(threads > 0, "split_node_limit: zero threads");
    if limits.infinite {
        return None;
    }
    let t = threads as u64;
    let mut share = limits.clone();
    let mut split = false;
    if let Some(n) = share.nodes {
        share.nodes = Some(n.div_ceil(t).max(1));
        split = true;
    }
    // div_ceil is monotonic, so soft <= hard survives the division.
    if let Some(s) = share.soft_nodes {
        share.soft_nodes = Some(s.div_ceil(t).max(1));
        split = true;
    }
    if let Some(h) = share.hard_nodes {
        share.hard_nodes = Some(h.div_ceil(t).max(1));
        split = true;
    }
    split.then_some(share)
}

impl ThreadData {
    /// Create a new thread with the given id. History tables are zeroed,
    /// position is startpos (will be overwritten in `prepare_search`).
    pub fn new(id: usize) -> Self {
        let sentinel = Box::new([[0i16; Square::NUM]; Piece::NUM]);
        let sentinel_ptr = &*sentinel as *const PieceToTable as *mut PieceToTable;
        let mut stack = [StackEntry::default(); MAX_PLY + SS_OFFSET + 1];
        for entry in &mut stack {
            entry.conthist_ptr = sentinel_ptr;
        }

        ThreadData {
            id,
            silent: false,
            pos: Position::from_fen(STARTPOS).expect("startpos"),
            nodes: 0,
            qs_nodes: 0,
            history: ButterflyHistory::new(),
            pawn_history: PawnHistory::new(),
            cap_history: CaptureHistory::new(),
            cont_history: ContinuationHistory::new(),
            countermoves: Countermoves::new(),
            cont_correction: ContCorrectionHistory::new(),
            nnue: nnue::Network::new(),
            killers: Killers::new(),
            pv: [[Move::NONE; MAX_PLY + 1]; MAX_PLY + 1],
            pv_len: [0; MAX_PLY + 1],
            root_delta: SCORE_INFINITE,
            stack,
            sentinel_conthist: sentinel,
            tm: TimeManager::infinite(),
            root_depth: 0,
            nmp_min_ply: 0,
            seek_mate: false,
            completed_depth: 0,
            best_move: Move::NONE,
            best_score: -SCORE_INFINITE,
            search_deadline: 0,
            thread_count: 1,
            per_thread_nodes: false,
            stopped: false,
            seldepth: 0,
            root_move: Move::NONE,
            root_move_number: 0,
            pondering: false,
            mate_limit: None,
            root_moves: Vec::new(),
            multi_pv: 1,
            reported_multi_pv: 1,
            pv_index: 0,
            tb_hits: 0,
            #[cfg(feature = "syzygy")]
            tb_config: TbConfig::default(),
            datagen_mode: false,
            original_time_adjust: TIME_ADJUST_UNSET,
            best_previous_score: SCORE_NONE,
            best_previous_avg_score: SCORE_NONE,
            previous_time_reduction: INITIAL_TIME_REDUCTION,
            used_lazy_eval: false,
            skill: crate::skill::Snapshot::FULL_STRENGTH,
            optimism: [0; 2],
            last_iteration_pv: [Move::NONE; MAX_PLY + 1],
            last_iteration_pv_len: 0,
            curr_line: Arc::new(CurrLineSlot::new()),
            report_curr_line: false,
            report_refutations: false,
            #[cfg(feature = "stats")]
            stats: Box::new(crate::stats::SearchStats::zeroed()),
            #[cfg(feature = "tree")]
            tree: None,
        }
    }

    /// Access the search stack entry at a given ply (with SS_OFFSET).
    #[inline]
    pub fn ss(&self, ply: usize) -> &StackEntry {
        debug_assert!(ply < MAX_PLY, "ss: ply {} >= MAX_PLY", ply);
        debug_assert!(
            ply + SS_OFFSET < self.stack.len(),
            "ss: ply {} + SS_OFFSET {} >= stack len {}",
            ply,
            SS_OFFSET,
            self.stack.len()
        );
        &self.stack[ply + SS_OFFSET]
    }

    /// Mutable access to the search stack entry at a given ply.
    #[inline]
    pub fn ss_mut(&mut self, ply: usize) -> &mut StackEntry {
        debug_assert!(ply < MAX_PLY, "ss_mut: ply {} >= MAX_PLY", ply);
        debug_assert!(
            ply + SS_OFFSET < self.stack.len(),
            "ss_mut: ply {} + SS_OFFSET {} >= stack len {}",
            ply,
            SS_OFFSET,
            self.stack.len()
        );
        &mut self.stack[ply + SS_OFFSET]
    }

    /// Get continuation history score for a single ply offset.
    /// Sentinel pointers return 0 (all-zero table), eliminating null checks.
    #[inline]
    pub fn conthist(&self, ply: usize, offset: usize, piece: Piece, to: Square) -> i32 {
        let base = ply + SS_OFFSET;
        debug_assert!(
            base >= offset,
            "conthist: base {} < offset {}",
            base,
            offset
        );
        ContinuationHistory::get(
            self.stack[base - offset].conthist_ptr as *const _,
            piece,
            to,
        )
    }

    /// Check time/node limits every 512 nodes.
    /// For helper threads (infinite limits), this is effectively a no-op.
    /// During pondering, time/node limits are suspended until `ponderhit`.
    #[inline]
    pub fn check_limits(&mut self) {
        if self.nodes & 511 == 0 {
            // Feed whoever is watching the rate: the bench monitor, or the interface
            // showing what the engine is doing on the player's time.
            if counting_nodes() {
                NODES_SEARCHED.fetch_add(512, Ordering::Relaxed);
            }

            // The interface must never see a mute engine: after a second without a
            // line, the main thread says where it is. One clock read per 512 nodes,
            // which the deadline test below already pays.
            if self.id == 0 && !self.silent {
                let now = self.tm.total_elapsed_ms();
                if heartbeat_due(
                    now,
                    LAST_INFO_MS.load(Ordering::Relaxed),
                    HEARTBEAT_MS.load(Ordering::Relaxed),
                ) {
                    self.print_heartbeat(now);
                }
            }

            // A host outside the engine may want the search to end: a browser tab, where
            // the interface runs in one thread and the search in another and a message
            // cannot reach a worker busy in a search. Asked once every 512 nodes, so the
            // call costs a few thousand a second and the answer is never more than a
            // fraction of a millisecond stale. On every other target this folds away to
            // nothing at compile time.
            if host_wants_stop() {
                STOP.store(true, Ordering::Relaxed);
            }

            // Detect ponderhit transition: PONDER was cleared by stdin reader
            if self.pondering && !PONDER.load(Ordering::Relaxed) {
                self.pondering = false;
                self.tm.restart();
                // A depth-limited ponder may already have searched far beyond its
                // requested depth. Do not finish another large iteration after the hit.
                if self.root_depth > self.tm.max_depth() {
                    STOP.store(true, Ordering::Relaxed);
                }
            }

            // Skip time/node limits while pondering (search indefinitely).
            // In datagen, and whenever a node budget was split across the pool, the
            // limit is this thread's alone: stop it like the deadline below — a thread
            // spending its own share must not kill its siblings through the global STOP.
            // The deadline cannot fire before depth one is in hand: a move has to be
            // played, and the alternative is handing back whatever the first legal
            // move happened to be. The node budget has no such guard, so a fixed-node
            // search returns the same tree as ever.
            if !self.pondering
                && ((self.completed_depth >= 1 && self.tm.should_stop_hard())
                    || self.nodes >= self.tm.max_nodes())
            {
                if self.datagen_mode || self.per_thread_nodes {
                    self.stopped = true;
                } else {
                    STOP.store(true, Ordering::Relaxed);
                }
            }
            // Per-thread deadline (datagen): stop this thread without touching global STOP
            if self.search_deadline > 0 && self.tm.elapsed_ms() >= self.search_deadline {
                self.stopped = true;
            }
        }
    }

    /// Progress line for a long iteration: depth, selective depth, the root move
    /// under search and the counters — everything but a score and a PV, which the
    /// iteration has not produced yet.
    fn print_heartbeat(&self, now_ms: u64) {
        let nodes = self.nodes.max(NODES_SEARCHED.load(Ordering::Relaxed));
        let nps = nodes * 1000 / now_ms.max(1);
        let currmove = if self.root_move != Move::NONE {
            format!(
                " currmove {} currmovenumber {}",
                self.root_move.to_uci_960(self.pos.chess960),
                self.root_move_number
            )
        } else {
            String::new()
        };
        out!(
            "info depth {} seldepth {}{currmove} nodes {nodes} nps {nps} tbhits {} time {now_ms}",
            self.root_depth,
            self.seldepth,
            self.tb_hits
        );
        note_output(now_ms);
    }

    /// Check if this thread should stop searching.
    /// Checks both per-thread stopped flag and global STOP.
    #[inline]
    pub fn should_stop(&self) -> bool {
        self.stopped || should_stop()
    }

    /// Lowers the search ceiling to what the handicap allows in this position.
    ///
    /// How far a weakened level looks is a property of the level and the position
    /// together, so it can only be settled once the root position is known. Anything that
    /// installs a time manager of its own must call this afterwards or the level searches
    /// as deep as it likes — which is why it is a method and not three copies of a
    /// two-line `if`.
    pub fn apply_skill_ceiling(&mut self) {
        if self.skill.active {
            self.tm
                .cap_depth(crate::skill::ceiling(&self.skill, self.pos.key));
        }
    }

    /// Prepare this thread for a new search.
    pub fn prepare_search(&mut self, pos: &Position, limits: &SearchLimits) {
        self.prepare_search_filtered(pos, limits, &[]);
    }

    /// Like [`prepare_search`], restricted to `search_moves` when that slice is not empty
    /// (`go searchmoves` — ChessBase uses it to exclude moves from analysis).
    pub fn prepare_search_filtered(
        &mut self,
        pos: &Position,
        limits: &SearchLimits,
        search_moves: &[Move],
    ) {
        self.pos = pos.clone();
        self.nodes = 0;
        // Read the handicap once, here: the level is set before a search is asked for,
        // never during one.
        self.skill = crate::skill::snapshot();
        #[cfg(feature = "stats")]
        self.stats.clear();

        // Refresh NNUE accumulator for root position
        if nnue::network::has_network() {
            self.nnue.refresh(&self.pos);
        }
        self.root_depth = 0;
        self.nmp_min_ply = 0;
        self.seek_mate = false;
        self.completed_depth = 0;
        self.seldepth = 0;
        self.root_move = Move::NONE;
        self.root_move_number = 0;
        self.best_move = Move::NONE;
        self.best_score = -SCORE_INFINITE;
        self.stopped = false;
        self.pondering = PONDER.load(Ordering::Relaxed);
        self.mate_limit = limits.mate;
        self.killers.clear();
        // The first iteration has no running average yet, so the optimism block at the
        // root is skipped and the eval would otherwise be blended with the value left
        // by the *previous* search — a different position, possibly a different game.
        self.optimism = [0; 2];
        self.pv_index = 0;
        self.tb_hits = 0;
        self.curr_line.clear();
        #[cfg(feature = "syzygy")]
        {
            self.tb_config = TbConfig {
                root_in_tb: false,
                // Default: probe in-tree at max_pieces cardinality.
                // Set to 0 by root probing when DTZ is available.
                cardinality: crate::tb::max_pieces(),
            };
        }

        // Generate root moves (all legal moves from this position)
        let mut buf: ArrayBuf<Move, MAX_MOVES> = ArrayBuf::new();
        let count = movegen::generate_legal_moves(&self.pos, &mut buf);
        self.root_moves.clear();
        self.root_moves.reserve(count);
        for i in 0..count {
            if !search_moves.is_empty() && !search_moves.contains(&buf[i]) {
                continue;
            }
            self.root_moves.push(RootMove::new(buf[i]));
        }
        // Default best_move to first legal move so we always have a valid
        // response even if stopped before completing depth 1.
        self.best_move = if !self.root_moves.is_empty() {
            self.root_moves[0].mv
        } else {
            Move::NONE
        };

        // Time manager: real limits for the main thread, infinite for the helpers —
        // except for a node budget, which is divided evenly instead (see
        // `split_node_limit`). At one thread nothing is divided and nothing changes.
        self.per_thread_nodes = false;
        let game_ply = self.pos.game_ply();
        match split_node_limit(limits, self.thread_count).filter(|_| self.thread_count > 1) {
            Some(share) => {
                self.per_thread_nodes = true;
                self.tm = TimeManager::new(&share, game_ply, &mut self.original_time_adjust);
            }
            None if self.id == 0 => {
                self.tm = TimeManager::new(limits, game_ply, &mut self.original_time_adjust);
            }
            None => self.tm = TimeManager::infinite(),
        }
        self.apply_skill_ceiling();
        // The upper rungs need to see more than one root move to have anything to choose
        // between. Raised, never lowered: someone who asked for a multi-PV analysis gets
        // the lines they asked for.
        self.reported_multi_pv = self.multi_pv;
        if self.skill.active {
            self.multi_pv = self
                .multi_pv
                .max(self.skill.rung.variety_moves.max(1) as usize);
        }

        // Reset search stack
        let sentinel_ptr = &*self.sentinel_conthist as *const PieceToTable as *mut PieceToTable;
        for entry in &mut self.stack {
            entry.played_move = Move::NONE;
            entry.moved_piece = Piece::NONE;
            entry.is_capture = false;
            entry.conthist_ptr = sentinel_ptr;
            entry.cont_corr_ptr = std::ptr::null_mut();
            entry.static_eval = SCORE_NONE;
            entry.correction_value = 0;
            entry.excluded = Move::NONE;
            entry.double_extensions = 0;
            entry.reduction = 0;
            entry.follow_pv = false;
        }
    }

    /// Clear all persistent history tables (called on `ucinewgame`).
    pub fn clear_histories(&mut self) {
        self.history.clear();
        self.pawn_history.clear();
        self.cap_history.clear();
        self.cont_history.clear();
        self.countermoves.clear();
        self.cont_correction.clear();
    }

    /// Publish the candidate root line without waiting on the UCI output path.
    pub fn publish_curr_line(&self, root_move: Move) {
        let Some(rm) = self.root_moves.iter().find(|rm| rm.mv == root_move) else {
            return;
        };
        self.curr_line.publish(&rm.pv[..rm.pv_len.max(1)]);
    }

    /// Forget the previous game's time-management state (`ucinewgame`).
    ///
    /// These four outlive a search on purpose — they compare this move to the
    /// last one — so they are the one thing a new game has to clear that
    /// `prepare_search` must not.
    pub fn new_game(&mut self) {
        self.original_time_adjust = TIME_ADJUST_UNSET;
        self.best_previous_score = SCORE_NONE;
        self.best_previous_avg_score = SCORE_NONE;
        self.previous_time_reduction = INITIAL_TIME_REDUCTION;
    }
}

// ============================================================
// ThreadPool
// ============================================================

/// Pool of search threads with shared transposition table.
pub struct ThreadPool {
    pub shared: SharedState,
    pub threads: Vec<ThreadData>,
    /// UCI MultiPV option (default 1).
    pub multi_pv: usize,
    /// Index of the best thread from the last search (for PV extraction).
    pub last_best_idx: usize,
}

impl ThreadPool {
    /// Create a new thread pool.
    pub fn new(num_threads: usize, hash_mb: usize) -> Self {
        let n = num_threads.max(1);
        let mut threads: Vec<ThreadData> = (0..n).map(ThreadData::new).collect();
        for td in &mut threads {
            td.thread_count = n;
        }
        ThreadPool {
            shared: SharedState::new(hash_mb, n),
            threads,
            multi_pv: 1,
            last_best_idx: 0,
        }
    }

    /// Change the number of search threads.
    pub fn resize_threads(&mut self, n: usize) {
        let n = n.max(1);
        self.threads.resize_with(n, || ThreadData::new(n));
        // Fix ids, and tell every thread how many of them there are
        for (i, td) in self.threads.iter_mut().enumerate() {
            td.id = i;
            td.thread_count = n;
        }
        // The shared correction tables are sized by the thread count.
        self.shared.corr.resize(n);
    }

    /// Resize the TT (must not be called during search).
    pub fn resize_hash(&mut self, mb: usize) {
        self.shared.tt.resize(mb);
    }

    /// Clear the shared state and all per-thread histories (`ucinewgame`).
    pub fn clear(&mut self) {
        self.shared.clear();
        for td in &mut self.threads {
            td.clear_histories();
            td.new_game();
        }
    }

    /// Launch parallel search and return the best move + optional ponder move.
    ///
    /// Uses `std::thread::scope` for scoped lifetime management:
    /// helper threads (id > 0) search with infinite limits until `stop`,
    /// main thread (id 0) searches with real time limits.
    pub fn start_search(&mut self, pos: &Position, limits: SearchLimits) -> (Move, Option<Move>) {
        self.start_search_filtered(pos, limits, &[])
    }

    /// Like [`start_search`], with a `go searchmoves` filter applied at the root.
    pub fn start_search_filtered(
        &mut self,
        pos: &Position,
        limits: SearchLimits,
        search_moves: &[Move],
    ) -> (Move, Option<Move>) {
        STOP.store(false, Ordering::Relaxed);
        SEARCHING.store(true, Ordering::Relaxed);
        let _searching = SearchingGuard;
        self.shared.tt.new_search();
        // The info lines report the whole search, not the main thread's share: with
        // helpers running, every thread feeds the shared counter, zeroed here so a
        // single-thread search that follows never reads a stale total.
        NODES_SEARCHED.store(0, Ordering::Relaxed);
        LAST_INFO_MS.store(0, Ordering::Relaxed);

        // Prepare all threads
        for td in &mut self.threads {
            td.multi_pv = self.multi_pv;
            td.prepare_search_filtered(pos, &limits, search_moves);
        }
        let report_curr_line = crate::uci::show_curr_line() && !self.threads[0].silent;
        let report_refutations = crate::uci::show_refutations() && !self.threads[0].silent;
        for td in &mut self.threads {
            td.report_curr_line = report_curr_line;
            td.report_refutations = report_refutations && td.id == 0;
        }

        // A weakened engine opens out of the book rather than searching, which is what
        // keeps a given level from playing the same game every time. Full strength
        // never gets here — `book::choice` answers `None` before it looks at anything.
        let analysing = self.multi_pv > 1
            || limits.infinite
            || PONDER.load(Ordering::Relaxed)
            || !search_moves.is_empty()
            || crate::uci::analyse_mode();
        if let Some(mv) = crate::book::choice(pos, analysing) {
            return (mv, None);
        }

        self.shared.cpu_load.reset();

        #[cfg(not(target_arch = "wasm32"))]
        let _curr_line_reporter = CurrLineReporter::start(&self.threads);

        // A handicapped engine searches on one thread whatever the machine has, so a
        // given skill level plays the same everywhere. Extra cores would otherwise
        // quietly make a "weak" opponent stronger.
        if self.threads.len() == 1 || crate::skill::level() < crate::skill::FULL_STRENGTH {
            // Single-thread fast path: no spawning overhead
            search::search(&mut self.threads[0], &self.shared);
            self.last_best_idx = 0;
            let best = self.threads[0].best_move;
            let ponder = self.extract_ponder_move(0);
            return (best, ponder);
        }

        // Multi-thread: scope ensures all threads join before we return
        // The info lines report the whole search, so the shared counter has to be live
        // while helpers run. A watcher, not a flag: another watcher may already be
        // reading, and closing is balanced so the count never leaks.
        open_node_meter();
        std::thread::scope(|s| {
            let (main_td, helpers) = self.threads.split_first_mut().unwrap();
            let shared = &self.shared;

            // Spawn helper threads with 32 MB stack (PGO inlining enlarges frames)
            for td in helpers.iter_mut() {
                std::thread::Builder::new()
                    .stack_size(32 * 1024 * 1024)
                    .spawn_scoped(s, move || {
                        search::search(td, shared);
                    })
                    .expect("failed to spawn search thread");
            }

            // Main thread runs search (blocks until time/depth limit)
            search::search(main_td, shared);

            // Main thread finished → stop all helpers. Unless the node budget was
            // split: each helper then owns a share, and cutting it short here would
            // hand an idle machine more work per move than a busy one — exactly what
            // a fixed-node lane at several threads exists to avoid.
            if !main_td.per_thread_nodes {
                STOP.store(true, Ordering::Relaxed);
            }
        });
        close_node_meter();

        // Select best thread via weighted vote aggregation
        let best_idx = self.select_best_thread();
        self.last_best_idx = best_idx;

        let best = self.threads[best_idx].best_move;
        let ponder = self.extract_ponder_move(best_idx);
        (best, ponder)
    }

    /// Emit the final UCI snapshot immediately before `bestmove`.
    ///
    /// Normal iterative output can be followed by `currmove` from an interrupted
    /// iteration. Repeating the last completed PV here gives the GUI final statistics
    /// and also makes a helper selected by Lazy SMP agree with the move being played.
    pub fn print_final_uci_info(&self) {
        let td = &self.threads[self.last_best_idx];
        if td.silent || td.root_moves.is_empty() {
            return;
        }

        if td.root_moves[0].depth > 0 {
            let multi_pv = td.reported_multi_pv.min(td.root_moves.len()).max(1);
            // Every root move carries the depth its score comes from, and the
            // deepest completed line is at the front: the snapshot is exact, never
            // a bound, never shallower than what was finished.
            search::print_multi_pv_info(td, &self.shared, multi_pv);
        } else if td.best_move.is_ok() {
            let elapsed = td.tm.total_elapsed_ms().max(1);
            let cpuload = self
                .shared
                .cpu_load(td.thread_count)
                .map_or_else(String::new, |load| format!(" cpuload {load}"));
            out!(
                "info depth 0 nodes {} time {elapsed}{cpuload} pv {}",
                td.nodes,
                td.best_move.to_uci_960(td.pos.chess960)
            );
        }
    }

    /// Select the best thread via weighted vote aggregation.
    ///
    /// Only threads whose front root move was scored at the deepest completed
    /// iteration of the pool take part: each votes for that move with weight
    /// `score - min_score + 10`, votes are summed per move, ties go to the longer
    /// PV. A thread whose front move is decisive (mate or tablebase, and every root
    /// score is exact) is taken over any vote — shorter mate, longer defence. A
    /// shallower thread with a stale optimistic score can therefore never outvote a
    /// deeper one that has seen the refutation: with the old weights, two helpers at
    /// depth 12 on a +7 outvoted the main thread at depth 13 on a −279 that had
    /// already published the refutation of their move.
    fn select_best_thread(&self) -> usize {
        // Skip voting for single thread or multi-PV mode
        if self.threads.len() <= 1 || self.multi_pv > 1 {
            return 0;
        }

        let scored = |t: &ThreadData| !t.root_moves.is_empty() && t.root_moves[0].depth > 0;
        let deepest = self
            .threads
            .iter()
            .filter(|t| scored(t))
            .map(|t| t.root_moves[0].depth)
            .max()
            .unwrap_or(0);
        if deepest == 0 {
            return 0;
        }
        let elector = |t: &ThreadData| scored(t) && t.root_moves[0].depth == deepest;

        let min_score = self
            .threads
            .iter()
            .filter(|t| elector(t))
            .map(|t| t.root_moves[0].score)
            .min()
            .unwrap_or(-SCORE_INFINITE);
        let vote_value =
            |t: &ThreadData| -> i64 { t.root_moves[0].score as i64 - min_score as i64 + 10 };

        // Aggregate votes per move via small Vec (max entries = thread count)
        let mut votes: Vec<(Move, i64)> = Vec::with_capacity(self.threads.len());
        for t in self.threads.iter().filter(|t| elector(t)) {
            let mv = t.root_moves[0].mv;
            if let Some(entry) = votes.iter_mut().find(|(m, _)| *m == mv) {
                entry.1 += vote_value(t);
            } else {
                votes.push((mv, vote_value(t)));
            }
        }
        let get_votes =
            |mv: Move| -> i64 { votes.iter().find(|(m, _)| *m == mv).map_or(0, |e| e.1) };

        let is_win = |s: i32| s >= SCORE_TB_WIN_IN_MAX;
        let is_loss = |s: i32| s != -SCORE_INFINITE && s <= -SCORE_TB_WIN_IN_MAX;
        let decisive = |s: i32| is_win(s) || is_loss(s);

        let mut best = self
            .threads
            .iter()
            .position(|t| elector(t))
            .expect("the deepest thread is an elector");
        for (i, ti) in self.threads.iter().enumerate() {
            if i == best || !scored(ti) {
                continue;
            }
            let best_td = &self.threads[best];
            let best_score = best_td.root_moves[0].score;
            let this_score = ti.root_moves[0].score;

            if is_win(best_score) {
                // Current best is winning: prefer shorter mate (higher score)
                if this_score > best_score {
                    best = i;
                }
            } else if is_loss(best_score) {
                // Current best is losing: prefer longer defense (lower score)
                if is_loss(this_score) && this_score < best_score {
                    best = i;
                }
            } else if decisive(this_score) {
                // A decisive result is exact whatever its depth: take it
                best = i;
            } else if elector(ti) {
                // Normal: compare aggregated votes, tiebreak by vote_value × (pv_len > 2)
                let best_mv_votes = get_votes(best_td.root_moves[0].mv);
                let this_mv_votes = get_votes(ti.root_moves[0].mv);
                let pv_filtered_value =
                    |t: &ThreadData| -> i64 { vote_value(t) * (t.root_moves[0].pv_len > 2) as i64 };
                if this_mv_votes > best_mv_votes
                    || (this_mv_votes == best_mv_votes
                        && pv_filtered_value(ti) > pv_filtered_value(best_td))
                {
                    best = i;
                }
            }
        }

        best
    }

    /// Sum search statistics across all threads (call after the search joins).
    #[cfg(feature = "stats")]
    pub fn aggregated_stats(&self) -> crate::stats::SearchStats {
        let mut agg = crate::stats::SearchStats::zeroed();
        for td in &self.threads {
            agg.add(&td.stats);
        }
        agg
    }

    /// Extract the ponder move (PV[1]) from the given thread's best root move.
    fn extract_ponder_move(&self, thread_idx: usize) -> Option<Move> {
        let td = &self.threads[thread_idx];
        if !td.root_moves.is_empty() && td.root_moves[0].pv_len >= 2 {
            let m = td.root_moves[0].pv[1];
            if m.is_ok() { Some(m) } else { None }
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_heartbeat_is_due_once_the_silence_reaches_the_interval() {
        assert!(!heartbeat_due(999, 0, 1000));
        assert!(heartbeat_due(1000, 0, 1000));
        assert!(heartbeat_due(2500, 1200, 1000));
        assert!(!heartbeat_due(2100, 1200, 1000));
        assert!(heartbeat_due(5, u64::MAX, 1000) == false, "no overflow");
    }
    use crate::types::Move;

    /// A quiet middlegame position: deep enough that a few thousand nodes settle
    /// nothing, so no thread can finish its work before spending its budget.
    const MIDGAME: &str = "r1bq1rk1/pp2ppbp/2np2p1/2n5/P3PP2/N1P2N2/1PB3PP/R1B1QRK1 b - - 0 1";

    #[test]
    fn current_line_snapshot_is_complete_and_clearable() {
        let slot = CurrLineSlot::new();
        let line = [
            Move::new(Square::E2, Square::E4),
            Move::new(Square::E7, Square::E5),
            Move::new(Square::G1, Square::F3),
        ];
        assert!(slot.snapshot().is_none());
        slot.publish(&line);
        assert_eq!(slot.snapshot().unwrap().1, line);
        slot.clear();
        assert!(slot.snapshot().is_none());
    }

    #[test]
    fn a_node_budget_divides_evenly_and_nothing_else_does() {
        // A clock belongs to the main thread; only a node count is shared out.
        assert_eq!(
            split_node_limit(&SearchLimits::nodes(40_000), 4),
            Some(SearchLimits::nodes(10_000))
        );
        assert_eq!(
            split_node_limit(&SearchLimits::soft_nodes(1_000, 100_000), 8),
            Some(SearchLimits::soft_nodes(125, 12_500))
        );
        assert!(split_node_limit(&SearchLimits::depth(12), 4).is_none());
        assert!(split_node_limit(&SearchLimits::movetime(500), 4).is_none());
        assert!(split_node_limit(&SearchLimits::infinite(), 4).is_none());
        assert!(split_node_limit(&SearchLimits::clock(1000, 10, None), 4).is_none());

        // Rounded up, and never down to nothing: a thread with a budget of zero would
        // return before it had a move to offer.
        assert_eq!(
            split_node_limit(&SearchLimits::nodes(10), 4),
            Some(SearchLimits::nodes(3))
        );
        assert_eq!(
            split_node_limit(&SearchLimits::nodes(1), 24),
            Some(SearchLimits::nodes(1))
        );

        // The order the time manager asserts on survives the division.
        let split = split_node_limit(&SearchLimits::soft_nodes(7, 7), 3).expect("split");
        let soft = split.soft_nodes.expect("soft");
        let hard = split.hard_nodes.expect("hard");
        assert!(soft <= hard, "soft {soft} > hard {hard}");
    }

    #[test]
    fn each_thread_of_a_pool_spends_its_own_share_of_the_budget() {
        let _guard = crate::skill::lock_level();
        STOP.store(false, Ordering::Relaxed);

        const THREADS: usize = 4;
        const BUDGET: u64 = 40_000;
        let pos = Position::from_fen(MIDGAME).unwrap();
        let mut pool = ThreadPool::new(THREADS, 4);
        for td in pool.threads.iter_mut() {
            td.silent = true;
        }
        let (best, _) = pool.start_search(&pos, SearchLimits::nodes(BUDGET));
        assert!(best != Move::NONE, "search returned no move");

        let share = BUDGET / THREADS as u64;
        for td in &pool.threads {
            // The limit is checked every 512 nodes, so a thread can overshoot by that
            // much plus the tail of the node it was in.
            assert!(
                td.nodes >= share && td.nodes < share + 2048,
                "thread {} searched {} nodes, expected about {}",
                td.id,
                td.nodes,
                share
            );
            assert!(td.per_thread_nodes, "thread {} did not own a share", td.id);
            assert!(
                td.stopped,
                "thread {} was not stopped by its own budget",
                td.id
            );
        }
        // Ending on its own share must not raise the flag that ends everyone else's.
        assert!(!should_stop(), "a spent budget raised the global stop flag");

        let total: u64 = pool.threads.iter().map(|td| td.nodes).sum();
        assert!(
            total >= BUDGET && total < BUDGET + 2048 * THREADS as u64,
            "pool searched {total} nodes for a budget of {BUDGET}"
        );
    }

    #[test]
    fn one_thread_keeps_the_budget_and_the_behaviour_it_always_had() {
        let _guard = crate::skill::lock_level();
        STOP.store(false, Ordering::Relaxed);

        let pos = Position::from_fen(MIDGAME).unwrap();
        let mut pool = ThreadPool::new(1, 4);
        pool.threads[0].silent = true;
        pool.start_search(&pos, SearchLimits::nodes(40_000));
        let td = &pool.threads[0];
        assert!(
            !td.per_thread_nodes,
            "a lone thread must not think it owns a share"
        );
        assert!(
            td.nodes >= 40_000 && td.nodes < 42_048,
            "lone thread searched {} nodes for a budget of 40000",
            td.nodes
        );
    }

    #[test]
    fn searchmoves_keeps_only_the_named_root_moves() {
        let pos = Position::from_fen(STARTPOS).unwrap();
        let e2e4 = Move::new(crate::types::Square::E2, crate::types::Square::E4);
        let d2d4 = Move::new(crate::types::Square::D2, crate::types::Square::D4);
        let mut td = ThreadData::new(0);
        td.thread_count = 1;
        td.prepare_search_filtered(&pos, &SearchLimits::depth(1), &[e2e4, d2d4]);
        assert_eq!(td.root_moves.len(), 2);
        let names: Vec<String> = td.root_moves.iter().map(|rm| rm.mv.to_uci()).collect();
        assert!(names.contains(&"e2e4".to_string()));
        assert!(names.contains(&"d2d4".to_string()));
    }

    #[test]
    fn searchmoves_confines_the_actual_root_search() {
        let pos = Position::from_fen_ex("r3k2r/8/8/8/8/8/8/R3K2R w AHah - 0 1", true).unwrap();
        let only = crate::uci::parse_uci_move(&pos, "e1h1").expect("legal Chess960 castle");
        let mut pool = ThreadPool::new(1, 4);
        pool.threads[0].silent = true;
        let (best, _) = pool.start_search_filtered(&pos, SearchLimits::depth(1), &[only]);
        let rm = &pool.threads[0].root_moves[0];
        assert_eq!(best, only);
        assert!(rm.pv_len > 0, "the allowed move received no searched PV");
        assert_eq!(rm.pv[0], only);
        assert!(rm.depth > 0, "the allowed move was never scored");
    }

    #[test]
    fn multipv_finishes_in_descending_score_order() {
        let pos = Position::from_fen(STARTPOS).unwrap();
        let mut pool = ThreadPool::new(1, 4);
        pool.multi_pv = 3;
        pool.threads[0].silent = true;
        pool.start_search(&pos, SearchLimits::depth(5));
        let lines = &pool.threads[0].root_moves[..3];
        assert!(
            lines.iter().all(|rm| rm.depth == 5),
            "MultiPV lines not all from the last iteration: {:?}",
            lines.iter().map(|rm| rm.depth).collect::<Vec<_>>()
        );
        assert!(
            lines.windows(2).all(|pair| pair[0].score >= pair[1].score),
            "MultiPV scores are not descending: {:?}",
            lines.iter().map(|rm| rm.score).collect::<Vec<_>>()
        );
    }

    /// Hand-filled pools for the election. The tester's game: the main thread had
    /// finished depth 13 on b6a5 (−279) after publishing the refutation of b6b2,
    /// two helpers still stood at depth 12 on b6b2 (+7) — and b6b2 was played.
    #[test]
    fn the_election_never_takes_a_shallower_stale_thread() {
        let b6a5 = Move::new(crate::types::Square::B6, crate::types::Square::A5);
        let b6b2 = Move::new(crate::types::Square::B6, crate::types::Square::B2);
        let set = |td: &mut ThreadData, mv: Move, score: i32, depth: i32| {
            let mut rm = RootMove::new(mv);
            rm.score = score;
            rm.depth = depth;
            rm.pv_len = 4;
            td.root_moves = vec![rm];
            td.completed_depth = depth;
        };
        let mut pool = ThreadPool::new(3, 1);
        set(&mut pool.threads[0], b6a5, -279, 13);
        set(&mut pool.threads[1], b6b2, 7, 12);
        set(&mut pool.threads[2], b6b2, 7, 12);
        assert_eq!(pool.select_best_thread(), 0, "the deeper thread decides");

        // The shallower threads are the deeper ones: they decide, by majority.
        set(&mut pool.threads[0], b6a5, -279, 12);
        set(&mut pool.threads[1], b6b2, 7, 13);
        set(&mut pool.threads[2], b6b2, 7, 13);
        assert_eq!(pool.select_best_thread(), 1);

        // At equal depth the votes are score-weighted: a clearly better single
        // score beats two mediocre ones, two close ones beat one.
        set(&mut pool.threads[0], b6a5, 40, 13);
        set(&mut pool.threads[1], b6b2, 7, 13);
        set(&mut pool.threads[2], b6b2, 7, 13);
        assert_eq!(pool.select_best_thread(), 0);
        set(&mut pool.threads[0], b6a5, 20, 13);
        set(&mut pool.threads[1], b6b2, 15, 13);
        set(&mut pool.threads[2], b6b2, 15, 13);
        assert_eq!(pool.select_best_thread(), 1);

        // A completed mate is exact at any depth and beats the vote.
        set(&mut pool.threads[0], b6a5, 40, 13);
        set(&mut pool.threads[1], b6b2, 7, 13);
        set(&mut pool.threads[2], b6b2, mate_in(9), 11);
        assert_eq!(pool.select_best_thread(), 2);

        // A thread that never scored a move is not a candidate.
        set(&mut pool.threads[0], b6a5, 40, 13);
        set(&mut pool.threads[1], b6b2, 7, 13);
        pool.threads[2].root_moves = vec![RootMove::new(b6b2)];
        pool.threads[2].completed_depth = 0;
        assert_eq!(pool.select_best_thread(), 0);
    }

    /// A real four-thread search: the move handed back is the front move of the
    /// elected thread, and that thread stands at the deepest completed iteration.
    #[test]
    fn the_elected_thread_stands_at_the_deepest_completed_iteration() {
        let pos = Position::from_fen(
            "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R1BQ1RK1 w - - 0 8",
        )
        .unwrap();
        let mut pool = ThreadPool::new(4, 16);
        for td in &mut pool.threads {
            td.silent = true;
        }
        let (best, _) = pool.start_search(&pos, SearchLimits::nodes(200_000));
        let elected = &pool.threads[pool.last_best_idx];
        assert_eq!(best, elected.root_moves[0].mv);
        let deepest = pool
            .threads
            .iter()
            .map(|t| t.root_moves[0].depth)
            .max()
            .unwrap();
        assert!(
            elected.root_moves[0].depth == deepest
                || elected.root_moves[0].score.abs() >= SCORE_TB_WIN_IN_MAX,
            "elected thread at depth {} while another completed {deepest}",
            elected.root_moves[0].depth
        );
    }
}
