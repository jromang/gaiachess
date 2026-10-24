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
use crate::largemem::LargeBox;
use crate::nnue;
use crate::numa::{self, NumaConfig, NumaPolicy};

/// Upper bound of the UCI `Threads` option.
pub const MAX_THREADS: usize = 1024;

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

/// One thread's node count, published for the others to read.
///
/// Every search thread writes only its own counter — a plain store every 512 nodes,
/// on a cache line of its own — and whoever wants the total adds them up. A single
/// shared counter bumped by every thread would bounce one cache line between all the
/// cores, and across sockets, hundreds of times a millisecond.
#[repr(align(64))]
#[derive(Default)]
pub struct NodeCounter(AtomicU64);

impl NodeCounter {
    #[inline]
    pub fn get(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }

    #[inline]
    fn set(&self, nodes: u64) {
        self.0.store(nodes, Ordering::Relaxed);
    }
}

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
    fn start(threads: &[LargeBox<ThreadData>]) -> Option<Self> {
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
    /// best (-SCORE_INFINITE while it never was). Exact, written by a settled
    /// aspiration loop — unless `in_progress` is set.
    pub score: i32,
    /// The iteration `score` comes from (0 while the move was never scored), or, when
    /// `in_progress`, the last completed iteration before the one that produced the
    /// score — the depth it sorts at, one below the depth it is reported at. The
    /// root moves are kept sorted by depth first, so `root_moves[0]` is always the
    /// deepest completed result — the move played and the line reported when the
    /// search stops in the middle of an iteration.
    pub depth: i32,
    /// `score` comes from an iteration that has not settled: a fail-high bound, or an
    /// exact score that raised alpha inside the window. The move then stands at the
    /// last completed depth, ahead of the previous best, so that a search stopped
    /// before the iteration settles plays the move it has just proved better. The
    /// settled iteration clears it.
    pub in_progress: bool,
    /// `score` is the lower bound of a fail-high (implies `in_progress`).
    pub lower_bound: bool,
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
    /// `pv` as it stood when the current iteration began, and its length (0 while
    /// the move was never scored): the line the follow-PV guard walks when this move
    /// heads a line of the iteration. The lines searched before it in the same
    /// iteration rewrite the PVs of the moves behind them, hence the snapshot.
    pub previous_pv: [Move; MAX_PLY + 1],
    pub previous_pv_len: usize,
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
            in_progress: false,
            lower_bound: false,
            previous_score: -SCORE_INFINITE,
            avg_score: SCORE_NONE,
            pv,
            pv_len: 1,
            previous_pv: [Move::NONE; MAX_PLY + 1],
            previous_pv_len: 0,
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
    /// The parent zeroes it on entry and again as its move loop starts, after the
    /// null move and ProbCut have searched at this ply; every child searched from
    /// there shares the slot, so it counts how often this ply keeps refuting the
    /// parent's moves.
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
    /// One correction table per logical NUMA node, indexed by
    /// [`ThreadData::numa_node`]: a node's threads pool what they learn, and read it
    /// from their own memory. Unpinned there is a single table for every thread.
    pub corr: Vec<SharedCorrectionHistory>,
    cpu_load: crate::cpuload::CpuLoad,
    /// Root best-move changes of the current search, summed over every thread.
    /// Time management reads it: a root that keeps changing its mind is a root
    /// worth more time, and instability seen by any thread counts. The main
    /// thread takes and zeroes it once per iteration; a root best-move change is
    /// rare enough that the atomic is never contended.
    pub best_move_changes: AtomicU32,
    /// The node counters of the pool's threads, summed by [`SharedState::nodes_searched`].
    /// Empty for a thread searching alone.
    node_counters: Vec<Arc<NodeCounter>>,
    /// The tablebase hit counters of the pool's threads, summed by
    /// [`SharedState::tb_hits`]. Empty for a thread searching alone.
    tb_counters: Vec<Arc<NodeCounter>>,
    /// How many threads the current search runs on: the capacity `cpuload` is
    /// measured against. One for a lone search, whatever the pool's size.
    searching_threads: usize,
}

impl SharedState {
    /// `threads` sizes the correction tables so the entries per thread, and with
    /// them the collision rate, stay what they are at one thread. Every caller that
    /// searches alone passes 1.
    pub fn new(hash_mb: usize, threads: usize) -> Self {
        SharedState {
            tt: TT::new(hash_mb),
            corr: vec![SharedCorrectionHistory::new(threads)],
            cpu_load: crate::cpuload::CpuLoad::new(),
            best_move_changes: AtomicU32::new(0),
            node_counters: Vec::new(),
            tb_counters: Vec::new(),
            searching_threads: 1,
        }
    }

    /// Nodes searched by the whole pool, each thread counted to its last 512.
    pub fn nodes_searched(&self) -> u64 {
        self.node_counters.iter().map(|c| c.get()).sum()
    }

    /// Tablebase hits of the whole pool in the tree, each thread counted to its last
    /// 512 nodes. The root ranking is not in it: see [`ThreadData::reported_tb_hits`].
    pub fn tb_hits(&self) -> u64 {
        self.tb_counters.iter().map(|c| c.get()).sum()
    }

    /// The correction table of logical node `node`.
    #[inline]
    pub fn corr_of(&self, node: usize) -> &SharedCorrectionHistory {
        debug_assert!(
            node < self.corr.len(),
            "corr_of: node {node} of {}",
            self.corr.len()
        );
        // SAFETY: the pool gives every thread a node below the table count.
        unsafe { self.corr.get_unchecked(node) }
    }

    /// Process load of the current search, in per mille of the threads it runs on.
    pub fn cpu_load(&self) -> Option<u32> {
        debug_assert!(self.searching_threads > 0, "a search on no thread");
        self.cpu_load.permille(self.searching_threads)
    }

    /// Clear everything shared (`ucinewgame`, and between bench runs).
    pub fn clear(&mut self) {
        self.tt.clear();
        for corr in &mut self.corr {
            corr.clear();
        }
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
    /// Logical NUMA node this thread is pinned to, which selects its correction table
    /// and its copy of the network weights. Zero when the pool is not pinned.
    pub numa_node: usize,
    /// Keeps this thread from writing to stdout. Thread 0 owns the clock, the root
    /// reporting and the tablebase lookups; a caller that wants those but not the
    /// output — a graphical interface driving the engine in-process — sets this rather
    /// than taking a helper's id, which would silently cost it the time management too.
    pub silent: bool,
    pub pos: Position,
    pub nodes: u64,
    /// `nodes`, published every 512 nodes for readers on other threads.
    pub node_counter: Arc<NodeCounter>,
    /// `tb_hits`, published with `node_counter`.
    pub tb_counter: Arc<NodeCounter>,
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
    /// End of the root moves the current line may choose from: the tablebase rank
    /// group of `root_moves[pv_index]` when the root is in the tables, every root move
    /// otherwise. Set by the iterative deepening for each line.
    pub pv_last: usize,
    /// Tablebase hits of this thread in the tree (the pool's sum, plus the root
    /// ranking, is reported as `tbhits`: see [`ThreadData::reported_tb_hits`]).
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
    /// Whether the evaluation in use is the network's, read once per search. The
    /// material and SEE conversions the search adds to the eval are measured in the
    /// network's units; without a network the eval is PeSTO, in centipawns, and they
    /// must be too (see [`ThreadData::piece_value`]).
    pub network_eval: bool,
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
    /// Follow-PV guard: the previous iteration's PV of the root move heading the line
    /// under search (see [`RootMove::previous_pv`]).
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

/// Whether the main thread, done with its search, stops the helpers.
///
/// Always, except when a node budget was split across the pool (see `split_node_limit`)
/// and nothing else bounds the move: each helper then owns a share and finishes it. A
/// clock is the main thread's alone — the helpers would otherwise run on to the hard
/// limit, three to seven times the move's budget, and `bestmove` wait for them.
fn main_stops_the_pool(per_thread_nodes: bool, has_clock: bool) -> bool {
    !per_thread_nodes || has_clock
}

/// Whether a main thread whose search has ended must keep the pool searching: a ponder
/// search until `ponderhit` or `stop`, an infinite one until `stop`. The helpers search
/// on meanwhile, as the protocol forbids leaving either kind of search unasked.
fn pool_must_hold(ponder_requested: bool, infinite: bool) -> bool {
    !STOP.load(Ordering::Relaxed)
        && ((ponder_requested && PONDER.load(Ordering::Relaxed)) || infinite)
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
            numa_node: 0,
            silent: false,
            pos: Position::from_fen(STARTPOS).expect("startpos"),
            nodes: 0,
            node_counter: Arc::new(NodeCounter::default()),
            tb_counter: Arc::new(NodeCounter::default()),
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
            pv_last: 0,
            tb_hits: 0,
            #[cfg(feature = "syzygy")]
            tb_config: TbConfig::default(),
            datagen_mode: false,
            original_time_adjust: TIME_ADJUST_UNSET,
            best_previous_score: SCORE_NONE,
            best_previous_avg_score: SCORE_NONE,
            previous_time_reduction: INITIAL_TIME_REDUCTION,
            used_lazy_eval: false,
            network_eval: true,
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

    /// A piece's material value in the units of the evaluation in use: measured on the
    /// network (`EVAL_PIECE_VALUE`), or the centipawns of PeSTO (`PIECE_VALUE`).
    ///
    /// Without a network the October conversions to network units applied to PeSTO's
    /// centipawns, about four times too large: the PeSTO search of the unit tests
    /// drifted (issue #23's frontier test fell one ply short), and so would any build
    /// left without its network.
    #[inline(always)]
    pub fn piece_value(&self, pt: usize) -> i32 {
        debug_assert!(pt < 7, "piece_value: type {pt}");
        if self.network_eval {
            EVAL_PIECE_VALUE[pt]
        } else {
            PIECE_VALUE[pt]
        }
    }

    /// A margin in the units of the evaluation in use, as a SEE threshold (centipawns).
    #[inline(always)]
    pub fn eval_to_see(&self, margin: i32) -> i32 {
        if self.network_eval {
            eval_to_see(margin)
        } else {
            margin
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

    /// Check the node limit at every node, the clock and the rest every 512 nodes.
    /// For helper threads (infinite limits), this is effectively a no-op.
    /// During pondering, time/node limits are suspended until `ponderhit`.
    #[inline]
    pub fn check_limits(&mut self, shared: &SharedState) {
        // The node budget is one comparison, so it is tested at every node: `go nodes
        // N` searches N nodes, not up to the next multiple of 512 on every thread.
        // In datagen, and whenever a node budget was split across the pool, the limit
        // is this thread's alone: stop it like the deadline below — a thread spending
        // its own share must not kill its siblings through the global STOP.
        if !self.pondering && self.nodes >= self.tm.max_nodes() {
            self.stop_on_own_limit();
        }
        if self.nodes & 511 == 0 {
            // Publish the count for whoever adds them up: the main thread's info
            // lines, or the interface showing what the engine is doing.
            self.node_counter.set(self.nodes);
            self.tb_counter.set(self.tb_hits);

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
                    self.print_heartbeat(now, shared);
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

            // Skip the time limit while pondering (search indefinitely).
            // The deadline cannot fire before depth one is in hand: a move has to be
            // played, and the alternative is handing back whatever the first legal
            // move happened to be. The node budget (above) has no such guard, so a
            // fixed-node search returns the same tree as ever.
            if !self.pondering && self.completed_depth >= 1 && self.tm.should_stop_hard() {
                self.stop_on_own_limit();
            }
            // Per-thread deadline (datagen): stop this thread without touching global STOP
            if self.search_deadline > 0 && self.tm.elapsed_ms() >= self.search_deadline {
                self.stopped = true;
            }
        }
    }

    /// Ends the search on this thread's own limit: the thread alone in datagen and
    /// when a node budget was split across the pool, the whole pool otherwise.
    #[cold]
    fn stop_on_own_limit(&mut self) {
        if self.datagen_mode || self.per_thread_nodes {
            self.stopped = true;
        } else {
            STOP.store(true, Ordering::Relaxed);
        }
    }

    /// Progress line for a long iteration: depth, selective depth, the root move
    /// under search and the counters — everything but a score and a PV, which the
    /// iteration has not produced yet.
    fn print_heartbeat(&self, now_ms: u64, shared: &SharedState) {
        let nodes = self.nodes.max(shared.nodes_searched());
        let nps = nodes * 1000 / now_ms.max(1);
        // The move under search waits for the bound lines, as `currmove` does.
        let currmove = if self.root_move != Move::NONE && search::long_search(nodes) {
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
            self.reported_tb_hits(shared)
        );
        note_output(now_ms);
    }

    /// The `tbhits` of an info line: every thread's hits in the tree, and the root
    /// ranking once. Each thread ranks the same root, so counting it per thread would
    /// count it once per thread.
    pub fn reported_tb_hits(&self, shared: &SharedState) -> u64 {
        let tree = self.tb_hits.max(shared.tb_hits());
        #[cfg(feature = "syzygy")]
        if self.tb_config.root_in_tb {
            return tree + self.root_moves.len() as u64;
        }
        tree
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
        self.node_counter.set(0);
        // Read the handicap once, here: the level is set before a search is asked for,
        // never during one.
        self.skill = crate::skill::snapshot();
        #[cfg(feature = "stats")]
        self.stats.clear();

        // Refresh NNUE accumulator for root position
        self.network_eval = nnue::network::has_network();
        if self.network_eval {
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
        self.pv_last = 0;
        self.tb_hits = 0;
        self.tb_counter.set(0);
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
            // The root's count has no parent to clear it: without this it ran over the
            // life of the process, and the root's LMR context depended on every search
            // before — a dump on a fresh thread did not reduce as the engine in a game.
            entry.failhighs = 0;
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

/// Builds `thread_node.len()` threads, each by a thread of its own — pinned to its node
/// when `bound` — so that its tables are first written, and placed, where it will run.
fn build_threads(
    numa: &NumaConfig,
    thread_node: &[usize],
    bound: bool,
) -> Vec<LargeBox<ThreadData>> {
    let n = thread_node.len();
    if n == 1 && !bound || cfg!(target_arch = "wasm32") {
        return (0..n).map(|i| LargeBox::new(ThreadData::new(i))).collect();
    }
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..n)
            .map(|i| {
                std::thread::Builder::new()
                    .stack_size(32 * 1024 * 1024)
                    .spawn_scoped(s, move || {
                        if bound {
                            numa::bind_current_thread(numa, thread_node[i]);
                        }
                        LargeBox::new(ThreadData::new(i))
                    })
                    .expect("failed to spawn a thread-building thread")
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("building a search thread panicked"))
            .collect()
    })
}

/// Pool of search threads with shared transposition table.
///
/// On a machine with several memory nodes the threads are pinned and spread over the
/// nodes when that pays (see [`crate::numa`]): each thread's own tables are allocated
/// and first written by the thread itself, each node gets its own correction table,
/// and every shared table is cleared in parallel so that each node holds its share.
pub struct ThreadPool {
    pub shared: SharedState,
    pub threads: Vec<LargeBox<ThreadData>>,
    /// UCI MultiPV option (default 1).
    pub multi_pv: usize,
    /// Index of the best thread from the last search (for PV extraction).
    pub last_best_idx: usize,
    numa_policy: NumaPolicy,
    /// The machine as the policy sees it.
    numa: NumaConfig,
    /// Whether the threads are pinned to their nodes.
    bound: bool,
    /// The last search was given a depth limit (`go depth N`). Only the main thread
    /// holds it, so the election is skipped: a helper that finished N + 1 before the
    /// main thread finished N would otherwise answer beyond the limit.
    depth_limited: bool,
}

impl ThreadPool {
    /// Create a new thread pool.
    pub fn new(num_threads: usize, hash_mb: usize) -> Self {
        let mut pool = ThreadPool {
            shared: SharedState::new(hash_mb, 1),
            threads: Vec::new(),
            multi_pv: 1,
            last_best_idx: 0,
            numa_policy: NumaPolicy::Auto,
            numa: NumaConfig::single(1),
            bound: false,
            depth_limited: false,
        };
        pool.rebuild(num_threads);
        pool
    }

    /// Change the number of search threads. Every thread starts afresh and the
    /// transposition table is reallocated, so that its pages follow the threads.
    pub fn resize_threads(&mut self, n: usize) {
        self.rebuild(n);
    }

    /// Change how the threads are laid out on the machine (UCI `NumaPolicy`).
    pub fn set_numa_policy(&mut self, policy: NumaPolicy) {
        self.numa_policy = policy;
        let n = self.threads.len();
        self.rebuild(n);
    }

    /// Resize the TT (must not be called during search).
    pub fn resize_hash(&mut self, mb: usize) {
        self.reallocate_tt(mb);
        self.parallel_clear(true, false);
    }

    /// Clear the shared state and all per-thread histories (`ucinewgame`).
    pub fn clear(&mut self) {
        self.parallel_clear(true, true);
    }

    /// One line describing the layout, for an `info string`.
    pub fn numa_summary(&self) -> String {
        let threads = self.threads.len();
        if !self.bound {
            return format!(
                "NUMA: {} node(s) available ({}), {threads} thread(s) not pinned",
                self.numa.node_count(),
                self.numa
            );
        }
        let mut per_node = vec![0usize; self.numa.node_count()];
        for td in &self.threads {
            per_node[td.numa_node] += 1;
        }
        let counts: Vec<String> = per_node.iter().map(usize::to_string).collect();
        format!(
            "NUMA: {threads} threads pinned over {} node(s) ({}) as {}, {} network cop{}",
            self.numa.node_count(),
            self.numa,
            counts.join("/"),
            nnue::replica_count(),
            if nnue::replica_count() == 1 {
                "y"
            } else {
                "ies"
            },
        )
    }

    /// Lay the pool out again for `n` threads under the current policy.
    fn rebuild(&mut self, n: usize) {
        let n = n.clamp(1, MAX_THREADS);
        let numa = self.numa_policy.config();
        let bound = self.numa_policy.binds(&numa, n);
        let thread_node = if bound {
            numa.distribute(n)
        } else {
            vec![0; n]
        };
        let nodes = if bound { numa.node_count() } else { 1 };
        let mut per_node = vec![0usize; nodes];
        for &node in &thread_node {
            per_node[node] += 1;
        }

        // Release everything before allocating its replacement.
        self.threads.clear();
        self.shared.corr.clear();

        // A correction table per node, sized for that node's threads and allocated on
        // it (Windows commits large pages at allocation).
        self.shared.corr = (0..nodes)
            .map(|node| {
                let count = per_node[node].max(1);
                if bound {
                    numa::run_on_node(&numa, node, || SharedCorrectionHistory::new(count))
                } else {
                    SharedCorrectionHistory::new(count)
                }
            })
            .collect();

        self.threads = build_threads(&numa, &thread_node, bound);
        // The thread elected last time may not exist any more.
        self.last_best_idx = 0;
        for (i, td) in self.threads.iter_mut().enumerate() {
            td.id = i;
            td.thread_count = n;
            td.numa_node = thread_node[i];
        }
        // The info lines report the pool's total: register every thread's counter.
        self.shared.node_counters = self
            .threads
            .iter()
            .map(|td| Arc::clone(&td.node_counter))
            .collect();
        self.shared.tb_counters = self
            .threads
            .iter()
            .map(|td| Arc::clone(&td.tb_counter))
            .collect();
        debug_assert!(
            self.threads
                .iter()
                .all(|td| td.numa_node < self.shared.corr.len()),
            "rebuild: a thread has no correction table"
        );

        self.numa = numa;
        self.bound = bound;
        nnue::replicate(if bound { Some(&self.numa) } else { None });

        let mb = self.shared.tt.size_mb().max(1);
        self.reallocate_tt(mb);
        self.parallel_clear(true, true);
    }

    /// A fresh, untouched table; 1 GiB pages once there are at least eight per node.
    fn reallocate_tt(&mut self, mb: usize) {
        let nodes = if self.bound {
            self.numa.node_count()
        } else {
            1
        };
        let giga = crate::largemem::wants_giga_pages(mb, nodes);
        self.shared.tt.reallocate(mb, giga);
    }

    /// Clears in parallel, every thread on its own node: its own histories and its
    /// share of its node's correction table (`histories`), and one contiguous slice of
    /// the transposition table (`tt`) — the threads sorted by node, so that each node
    /// receives one region. Writing a fresh page is what places it.
    fn parallel_clear(&mut self, tt: bool, histories: bool) {
        let n = self.threads.len();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| self.threads[i].numa_node);
        let mut rank = vec![0usize; n];
        for (r, &i) in order.iter().enumerate() {
            rank[i] = r;
        }
        let mut per_node = vec![0usize; self.shared.corr.len()];
        let mut node_rank = vec![0usize; n];
        for i in 0..n {
            let node = self.threads[i].numa_node;
            node_rank[i] = per_node[node];
            per_node[node] += 1;
        }

        let ThreadPool {
            shared,
            threads,
            numa,
            bound,
            ..
        } = self;
        let shared = &*shared;
        let (numa, bound) = (&*numa, *bound);
        let job = |i: usize, td: &mut ThreadData| {
            if histories {
                td.clear_histories();
                td.new_game();
                let node = td.numa_node;
                shared.corr[node].clear_part(node_rank[i], per_node[node]);
            }
            if tt {
                shared.tt.clear_part(rank[i], n);
            }
        };

        if n == 1 && !bound || cfg!(target_arch = "wasm32") {
            for (i, td) in threads.iter_mut().enumerate() {
                job(i, td);
            }
        } else {
            std::thread::scope(|s| {
                for (i, td) in threads.iter_mut().enumerate() {
                    let job = &job;
                    std::thread::Builder::new()
                        .stack_size(1024 * 1024)
                        .spawn_scoped(s, move || {
                            if bound {
                                numa::bind_current_thread(numa, td.numa_node);
                            }
                            job(i, td);
                        })
                        .expect("failed to spawn a clearing thread");
                }
            });
        }

        // A table no thread is assigned to (more nodes than threads) is never written.
        if tt {
            self.shared.tt.reset_age();
        }
        if histories {
            self.shared.best_move_changes.store(0, Ordering::Relaxed);
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
        self.start_search_armed(pos, limits, search_moves)
    }

    /// Like [`start_search_filtered`], for a caller that has set STOP itself.
    ///
    /// The UCI session does: a `stop` can be read off stdin before the session has
    /// reached the `go` it follows, and lowering STOP here would erase it — the search
    /// would then run until a second `stop` that never comes.
    pub fn start_search_armed(
        &mut self,
        pos: &Position,
        limits: SearchLimits,
        search_moves: &[Move],
    ) -> (Move, Option<Move>) {
        SEARCHING.store(true, Ordering::Relaxed);
        let _searching = SearchingGuard;
        self.depth_limited = limits.depth.is_some();
        // The info lines add up every thread's counter, those of the threads a lone
        // search leaves idle included: they start this search at zero.
        for td in &self.threads {
            td.node_counter.set(0);
            td.tb_counter.set(0);
        }
        self.shared.tt.new_search();
        LAST_INFO_MS.store(0, Ordering::Relaxed);

        // A handicapped engine searches on one thread whatever the machine has, so a
        // given skill level plays the same everywhere. Extra cores would otherwise
        // quietly make a "weak" opponent stronger. Decided before the threads are
        // prepared: the lone thread must divide nothing by the pool's size — its node
        // budget, its instability — or the level would depend on `Threads`.
        let lone = (self.threads.len() == 1 && !self.bound)
            || crate::skill::level() < crate::skill::FULL_STRENGTH;
        let searching = if lone { 1 } else { self.threads.len() };
        self.shared.searching_threads = searching;

        // Prepare the threads that search. A pinned thread evaluates with its node's
        // copy of the network, chosen here, before the root refresh, so that a network
        // loaded since the last search is the one used.
        let pool_size = self.threads.len();
        for td in &mut self.threads[..searching] {
            td.thread_count = searching;
            td.multi_pv = self.multi_pv;
            let replica = if self.bound {
                nnue::replica(self.numa.system_node_of(td.numa_node))
            } else {
                None
            };
            td.nnue.use_params(replica);
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
            // The last `info` line and the tablebase prefetch read the front root move
            // of the elected thread: make it the book move, on the thread they read.
            // Left alone, they showed the first legal move generated, and an index
            // kept from a larger pool could point past the end of this one.
            self.last_best_idx = 0;
            let td = &mut self.threads[0];
            td.thread_count = pool_size;
            if let Some(i) = td.root_moves.iter().position(|rm| rm.mv == mv) {
                td.root_moves[..=i].rotate_right(1);
            }
            td.best_move = mv;
            debug_assert!(
                td.root_moves.first().is_some_and(|rm| rm.mv == mv),
                "book move {} is not a root move",
                mv.to_uci()
            );
            return (mv, None);
        }

        self.shared.cpu_load.reset();

        #[cfg(not(target_arch = "wasm32"))]
        let _curr_line_reporter = CurrLineReporter::start(&self.threads[..searching]);

        if lone {
            // Single-thread fast path: no spawning overhead
            search::search(&mut self.threads[0], &self.shared);
            self.threads[0].thread_count = pool_size;
            self.last_best_idx = 0;
            let best = self.threads[0].best_move;
            let ponder = self.extract_ponder_move(0);
            return (best, ponder);
        }

        // Multi-thread: scope ensures all threads join before we return
        let infinite = limits.infinite;
        let ponder_requested = PONDER.load(Ordering::Relaxed);
        let bound = self.bound;
        let numa = &self.numa;
        std::thread::scope(|s| {
            let (main_td, helpers) = self.threads.split_first_mut().unwrap();
            let main_td: &mut ThreadData = main_td;
            let main_node = main_td.numa_node;
            let shared = &self.shared;

            // Spawn helper threads with 32 MB stack (PGO inlining enlarges frames).
            // Pinned threads read the network copy of their own node (set in `prepare`).
            for td in helpers.iter_mut() {
                let td: &mut ThreadData = td;
                std::thread::Builder::new()
                    .stack_size(32 * 1024 * 1024)
                    .spawn_scoped(s, move || {
                        if bound {
                            numa::bind_current_thread(numa, td.numa_node);
                        }
                        search::search(td, shared);
                    })
                    .expect("failed to spawn search thread");
            }

            // Main thread runs search (blocks until time/depth limit).
            // Main thread finished → stop all helpers. Unless the node budget was
            // split: each helper then owns a share, and cutting it short here would
            // hand an idle machine more work per move than a busy one — exactly what
            // a fixed-node lane at several threads exists to avoid.
            let mut main_job = move || {
                search::search(main_td, shared);
                // An infinite or ponder search that ended on its own (depth ceiling,
                // short mate, tables) keeps the pool searching until the interface
                // releases it: lowering STOP here would end the helpers and hand back a
                // `bestmove` nobody asked for yet. A root without a legal move is final.
                if main_td.best_move.is_ok() {
                    while pool_must_hold(ponder_requested, infinite) {
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }
                }
                if main_stops_the_pool(main_td.per_thread_nodes, main_td.tm.has_clock()) {
                    STOP.store(true, Ordering::Relaxed);
                }
            };
            if bound {
                // Pinned, the main search gets a thread of its own too: pinning the
                // caller would leave the UCI thread confined to one node.
                let node = main_node;
                std::thread::Builder::new()
                    .stack_size(32 * 1024 * 1024)
                    .spawn_scoped(s, move || {
                        numa::bind_current_thread(numa, node);
                        main_job();
                    })
                    .expect("failed to spawn the main search thread");
            } else {
                main_job();
            }
        });
        for td in &self.threads {
            td.node_counter.set(td.nodes);
            td.tb_counter.set(td.tb_hits);
        }

        // Select best thread via weighted vote aggregation
        let best_idx = self.select_best_thread();
        self.last_best_idx = best_idx;
        self.inherit_time_state(best_idx);
        #[cfg(feature = "syzygy")]
        self.extend_elected_tb_pv();

        let best = self.threads[best_idx].best_move;
        let ponder = self.extract_ponder_move(best_idx);
        (best, ponder)
    }

    /// Walks the tables to mate on the line of an elected helper, as the main thread
    /// does on its own lines when it prints them.
    ///
    /// With the root in the tables every thread searches it, and the vote may elect a
    /// helper: its line, the one reported last and the one the ponder move is read
    /// from, was never extended, and contradicted the main thread's lines. The clock is
    /// the main thread's, which alone holds one.
    #[cfg(feature = "syzygy")]
    pub fn extend_elected_tb_pv(&mut self) {
        let best_idx = self.last_best_idx;
        if best_idx == 0 || best_idx >= self.threads.len() {
            return;
        }
        let has_clock = self.threads[0].tm.has_clock();
        let td: &mut ThreadData = &mut self.threads[best_idx];
        if !td.tb_config.root_in_tb {
            return;
        }
        let Some(rm) = td.root_moves.first_mut() else {
            return;
        };
        // A certain result only (a cursed one never reaches mate in time), and not a
        // mate, which is already a complete line.
        if !is_tb_score(rm.tb_score) || is_mate_score(rm.score) || rm.pv_len == 0 {
            return;
        }
        search::syzygy_extend_pv(&mut td.pos, rm, has_clock, 1);
        debug_assert!(rm.pv_len > 0 && rm.pv[0] == rm.mv, "the extension kept the root move");
    }

    /// Hands the main thread the score the next move's time management compares
    /// against: the one of the move actually played, which is the elected thread's.
    ///
    /// Every thread ends its search on its own score and running average, but only the
    /// main thread reads them back, to measure how far the evaluation has fallen. The
    /// time reduction stays the main thread's own, as in the reference engines.
    fn inherit_time_state(&mut self, best_idx: usize) {
        if best_idx == 0 || best_idx >= self.threads.len() {
            return;
        }
        let elected = &self.threads[best_idx];
        if elected.root_moves.first().is_none_or(|rm| rm.depth == 0) {
            return;
        }
        let (score, avg) = (elected.best_previous_score, elected.best_previous_avg_score);
        let main = &mut self.threads[0];
        main.best_previous_score = score;
        main.best_previous_avg_score = avg;
    }

    /// Emit the final UCI snapshot immediately before `bestmove`.
    ///
    /// Normal iterative output can be followed by `currmove` from an interrupted
    /// iteration. Repeating the last completed PV here gives the GUI final statistics
    /// and also makes a helper selected by Lazy SMP agree with the move being played.
    pub fn print_final_uci_info(&self) {
        let Some(td) = self.threads.get(self.last_best_idx) else {
            debug_assert!(false, "elected thread {} out of the pool", self.last_best_idx);
            return;
        };
        if td.silent || td.root_moves.is_empty() {
            return;
        }

        if td.root_moves[0].depth > 0 {
            let multi_pv = td.reported_multi_pv.min(td.root_moves.len()).max(1);
            // Every root move carries the depth its score comes from, and the
            // deepest completed line is at the front: the snapshot is never
            // shallower than what was finished, and exact unless a fail-high of the
            // unfinished iteration stands there, reported as a lower bound.
            search::print_multi_pv_info(td, &self.shared, multi_pv);
        } else if td.best_move.is_ok() {
            // No iteration finished, but a root the tables have ranked still has its
            // score: report it, at depth 1, as any line of the search would.
            #[cfg(feature = "syzygy")]
            if td.tb_config.root_in_tb && td.root_moves[0].mv == td.best_move {
                search::print_multi_pv_info(td, &self.shared, 1);
                return;
            }
            let elapsed = td.tm.total_elapsed_ms().max(1);
            let cpuload = self
                .shared
                .cpu_load()
                .map_or_else(String::new, |load| format!(" cpuload {load}"));
            out!(
                "info depth 0 nodes {} time {elapsed}{cpuload} pv {}",
                td.nodes.max(self.shared.nodes_searched()),
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
        // Skip voting for single thread, multi-PV mode, or a depth limit: the main
        // thread is the one bound to the requested depth.
        if self.threads.len() <= 1 || self.multi_pv > 1 || self.depth_limited {
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

    /// Extract the ponder move from the given thread's best root move: PV[1], or, when
    /// the line stops after the move itself, the move the table holds for the position
    /// it leads to.
    ///
    /// A one-move line is common: a tablebase hit at ply 1, a draw, the line of a
    /// fail-high promoted when the search stopped. The table entry of the child is
    /// what the search expected there (the form of the reference engines); it comes
    /// from a shared, lock-free table, so it is checked for legality before use.
    fn extract_ponder_move(&self, thread_idx: usize) -> Option<Move> {
        let td = &self.threads[thread_idx];
        let rm = td.root_moves.first()?;
        if rm.pv_len >= 2 {
            let m = rm.pv[1];
            return m.is_ok().then_some(m);
        }
        let mv = rm.pv[0];
        if !mv.is_ok() {
            return None;
        }
        let mut pos = td.pos.clone();
        pos.make_move(mv);
        if pos.is_draw(1) {
            return None;
        }
        let reply = self.shared.tt.probe(pos.key, 0, pos.halfmove_clock)?.mv;
        movegen::is_playable_table_move(&pos, reply).then_some(reply)
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
            // The limit is checked at every node, before the node is counted: a
            // thread searches its share exactly.
            assert!(
                td.nodes == share,
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
            total == BUDGET,
            "pool searched {total} nodes for a budget of {BUDGET}"
        );
    }

    #[test]
    fn the_pool_reports_every_threads_nodes() {
        let _guard = crate::skill::lock_level();
        STOP.store(false, Ordering::Relaxed);

        let pos = Position::from_fen(MIDGAME).unwrap();
        let mut pool = ThreadPool::new(4, 4);
        for td in pool.threads.iter_mut() {
            td.silent = true;
        }
        pool.start_search(&pos, SearchLimits::nodes(40_000));

        // After the join every counter holds its thread's exact count, so the total
        // the info lines print is the sum over the pool, not the main thread's share.
        let sum: u64 = pool.threads.iter().map(|td| td.nodes).sum();
        assert_eq!(pool.shared.nodes_searched(), sum);
        assert!(sum > pool.threads[0].nodes, "only the main thread was counted");

        // A resize registers the new threads' counters, not the old ones'.
        pool.resize_threads(2);
        for td in pool.threads.iter_mut() {
            td.silent = true;
        }
        pool.start_search(&pos, SearchLimits::nodes(20_000));
        let sum: u64 = pool.threads.iter().map(|td| td.nodes).sum();
        assert_eq!(pool.shared.nodes_searched(), sum);
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
            td.nodes == 40_000,
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
        // Searches: a weakened level or a STOP raised by another test must not reach it.
        let _guard = crate::skill::lock_level();
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
        // Searches: a weakened level or a STOP raised by another test must not reach it.
        let _guard = crate::skill::lock_level();
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
        // A weakened level set by a test running beside this one would search on one
        // thread and leave the helpers unprepared.
        let _guard = crate::skill::lock_level();
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
    /// Shrinking the pool forgets which thread was elected: the index could otherwise
    /// point past the end of the new pool, and the next `info` line panic.
    #[test]
    fn a_rebuilt_pool_forgets_the_thread_it_elected() {
        let mut pool = ThreadPool::new(4, 1);
        pool.last_best_idx = 3;
        pool.resize_threads(2);
        assert_eq!(pool.last_best_idx, 0);
        assert!(pool.threads.get(pool.last_best_idx).is_some());
    }

    /// A move out of the book is the move the final `info` line and the prefetch
    /// read: the front root move of the elected thread. They used to read the first
    /// legal move generated (`pv a2a3` before `bestmove e2e4`).
    #[test]
    fn a_book_move_is_the_move_the_last_line_shows() {
        let _guard = crate::skill::lock_level();
        let pos = Position::from_fen(STARTPOS).unwrap();
        let mut pool = ThreadPool::new(2, 1);
        for td in &mut pool.threads {
            td.silent = true;
        }
        pool.last_best_idx = 1;
        crate::skill::set(10, 0);
        STOP.store(false, Ordering::Relaxed);
        let (best, ponder) = pool.start_search(&pos, SearchLimits::depth(2));
        crate::skill::set(crate::skill::FULL_STRENGTH, 0);
        assert!(best.is_ok());
        assert_eq!(ponder, None, "the book plays without a search");
        assert_eq!(pool.last_best_idx, 0);
        let td = &pool.threads[0];
        assert_eq!(td.best_move, best);
        assert_eq!(td.root_moves[0].mv, best, "the front move is the book move");
        assert_eq!(td.root_moves[0].depth, 0, "and it carries no search");
    }

    /// A line of one move still gives a ponder move when the table knows the reply,
    /// and only a legal one: none for a forged entry, none after a drawing move.
    #[test]
    fn a_one_move_line_ponders_on_the_table_reply() {
        use crate::tt::Bound;
        let _guard = crate::skill::lock_level();
        let pos = Position::from_fen(STARTPOS).unwrap();
        let mut pool = ThreadPool::new(1, 1);
        pool.threads[0].silent = true;
        STOP.store(false, Ordering::Relaxed);
        pool.start_search(&pos, SearchLimits::depth(4));
        let first = pool.threads[0].root_moves[0].mv;
        pool.threads[0].root_moves[0].pv_len = 1;

        let mut child = pos.clone();
        child.make_move(first);
        let mut buf: ArrayBuf<Move, MAX_MOVES> = ArrayBuf::new();
        movegen::generate_legal_moves(&child, &mut buf);
        let reply = buf[0];
        pool.shared
            .tt
            .store(child.key, 3, 0, 0, Bound::Exact, reply, 0, true);
        assert_eq!(pool.extract_ponder_move(0), Some(reply), "the table's reply");

        // An entry whose move cannot be played in the child (white's own e2e4).
        let forged = Move::new(Square::E2, Square::E4);
        assert!(!movegen::is_pseudo_legal(&child, forged));
        pool.shared
            .tt
            .store(child.key, 3, 0, 0, Bound::Exact, forged, 0, true);
        assert_eq!(pool.extract_ponder_move(0), None, "a forged entry is refused");
    }

    /// Stopped while a later multi-PV line is searched, the engine plays the move of
    /// the first line it has just published, and ponders on that move's reply: it used
    /// to play the previous iteration's move with the new move's reply as ponder.
    #[test]
    fn a_multi_pv_search_stopped_on_a_later_line_plays_its_first_line() {
        let _guard = crate::skill::lock_level();
        let pos = Position::from_fen(
            "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R1BQ1RK1 w - - 0 8",
        )
        .unwrap();
        for threads in [1, 4] {
            for nodes in (5_000..=300_000).step_by(29_500) {
                let mut pool = ThreadPool::new(threads, 4);
                pool.multi_pv = 3;
                for td in &mut pool.threads {
                    td.silent = true;
                }
                STOP.store(false, Ordering::Relaxed);
                let (best, ponder) = pool.start_search(&pos, SearchLimits::nodes(nodes));
                let front = &pool.threads[pool.last_best_idx].root_moves[0];
                assert_eq!(best, front.mv, "{threads} thread(s), {nodes} nodes");
                if front.pv_len >= 2 {
                    assert_eq!(ponder, Some(front.pv[1]), "{threads} thread(s), {nodes} nodes");
                }
            }
        }
    }

    /// A weakened level is the same opponent whatever `Threads` says: one thread
    /// searches, with the whole node budget — it used to get a quarter of it at four.
    #[test]
    fn a_weakened_level_searches_alike_on_any_pool() {
        let _guard = crate::skill::lock_level();
        // Out of the book, so the level has to search.
        let pos = Position::from_fen(
            "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R1BQ1RK1 w - - 0 8",
        )
        .unwrap();
        let mut results = Vec::new();
        for threads in [1, 4] {
            let mut pool = ThreadPool::new(threads, 4);
            for td in &mut pool.threads {
                td.silent = true;
            }
            crate::skill::set(19, 7);
            STOP.store(false, Ordering::Relaxed);
            let (best, _) = pool.start_search(&pos, SearchLimits::nodes(3_000));
            crate::skill::set(crate::skill::FULL_STRENGTH, 0);
            assert_eq!(pool.threads[0].thread_count, threads, "pool size restored");
            results.push((best, pool.threads[0].nodes));
        }
        assert_eq!(results[0], results[1], "same move, same nodes at 1 and 4 threads");
    }

    /// The main thread stops the helpers when it is done, unless each spends a node
    /// share of its own and no clock bounds the move.
    #[test]
    fn a_clock_lets_the_main_thread_stop_the_pool() {
        assert!(main_stops_the_pool(false, false));
        assert!(main_stops_the_pool(false, true));
        assert!(!main_stops_the_pool(true, false), "node shares are finished");
        assert!(main_stops_the_pool(true, true), "clock and nodes: the clock decides");
    }

    /// The next move measures its falling score against the move actually played,
    /// which is the elected thread's — not the main thread's own.
    #[test]
    fn the_elected_thread_hands_its_score_to_the_next_move() {
        let mut pool = ThreadPool::new(2, 1);
        for (i, td) in pool.threads.iter_mut().enumerate() {
            td.root_moves = vec![RootMove::new(Move::new(Square::E2, Square::E4))];
            td.root_moves[0].depth = 10;
            td.best_previous_score = 100 * (i as i32 + 1);
            td.best_previous_avg_score = 90 * (i as i32 + 1);
        }
        pool.inherit_time_state(0);
        assert_eq!(pool.threads[0].best_previous_score, 100, "main elected: unchanged");
        pool.inherit_time_state(1);
        assert_eq!(pool.threads[0].best_previous_score, 200);
        assert_eq!(pool.threads[0].best_previous_avg_score, 180);
    }

    /// `go mate N` also ends on a mate suffered within N moves: Black, in check, has
    /// only Kh8, and Qg7# follows. It used to search on to depth 127.
    #[test]
    fn a_mate_search_ends_on_a_mate_suffered_too() {
        let _guard = crate::skill::lock_level();
        let pos = Position::from_fen("6k1/5Q2/6K1/8/8/8/8/8 b - - 0 1").unwrap();
        let mut pool = ThreadPool::new(1, 4);
        pool.threads[0].silent = true;
        STOP.store(false, Ordering::Relaxed);
        let limits = SearchLimits {
            mate: Some(1),
            ..SearchLimits::default()
        };
        let (best, _) = pool.start_search(&pos, limits);
        let td = &pool.threads[0];
        assert_eq!(best.to_uci(), "g8h8", "the only move");
        assert_eq!(td.root_moves[0].score, mated_in(2), "mated in one move");
        assert!(td.completed_depth < 10, "searched on to depth {}", td.completed_depth);
    }

    /// `go mate N` on several threads: the search ends as soon as a thread proves the
    /// mate, with that mate.
    #[test]
    fn a_proved_mate_ends_a_mate_search_on_several_threads() {
        let _guard = crate::skill::lock_level();
        // A back-rank mate in one.
        let pos = Position::from_fen("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1").unwrap();
        let mut pool = ThreadPool::new(4, 4);
        for td in &mut pool.threads {
            td.silent = true;
        }
        STOP.store(false, Ordering::Relaxed);
        let limits = SearchLimits {
            mate: Some(1),
            ..SearchLimits::default()
        };
        let (best, _) = pool.start_search(&pos, limits);
        assert_eq!(best.to_uci(), "d1d8", "the back-rank mate");
        let elected = &pool.threads[pool.last_best_idx];
        assert_eq!(elected.root_moves[0].score, mate_in(1));
    }
}

#[cfg(test)]
mod pool_tests {
    use super::*;
    use std::time::{Duration, Instant};

    /// Runs `release` after `ms`, from another thread, as the interface would.
    fn release_after(ms: u64, release: fn()) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(ms));
            release();
        })
    }

    fn silent_pool(threads: usize) -> ThreadPool {
        let mut pool = ThreadPool::new(threads, 4);
        for td in &mut pool.threads {
            td.silent = true;
        }
        pool
    }

    /// `go infinite depth 4` on several threads: the main thread is done long before
    /// the `stop`, and the pool must still wait for it.
    #[test]
    fn an_infinite_search_on_several_threads_waits_for_stop() {
        let _guard = crate::skill::lock_level();
        STOP.store(false, Ordering::Relaxed);
        PONDER.store(false, Ordering::Relaxed);
        let pos = Position::from_fen("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1").unwrap();
        let mut pool = silent_pool(2);
        let limits = SearchLimits {
            infinite: true,
            depth: Some(4),
            ..SearchLimits::default()
        };
        let start = Instant::now();
        let stopper = release_after(300, || STOP.store(true, Ordering::Relaxed));
        let (best, _) = pool.start_search(&pos, limits);
        let waited = start.elapsed();
        stopper.join().unwrap();
        STOP.store(false, Ordering::Relaxed);
        assert_eq!(best.to_uci(), "a1a8", "the back-rank mate");
        assert!(waited >= Duration::from_millis(250), "bestmove after {waited:?}, before stop");
    }

    /// A ponder search ended by its own depth limit holds the pool until `ponderhit`.
    #[test]
    fn a_ponder_search_on_several_threads_waits_for_ponderhit() {
        let _guard = crate::skill::lock_level();
        STOP.store(false, Ordering::Relaxed);
        PONDER.store(true, Ordering::Relaxed);
        let pos = Position::from_fen("8/8/4k3/8/8/4K3/8/8 w - - 0 1").unwrap();
        let mut pool = silent_pool(2);
        let start = Instant::now();
        let hit = release_after(300, || PONDER.store(false, Ordering::Relaxed));
        let (best, _) = pool.start_search(&pos, SearchLimits::depth(2));
        let waited = start.elapsed();
        hit.join().unwrap();
        PONDER.store(false, Ordering::Relaxed);
        STOP.store(false, Ordering::Relaxed);
        assert!(best.is_ok());
        assert!(waited >= Duration::from_millis(250), "bestmove after {waited:?}, before ponderhit");
    }

    /// `go depth N` on several threads answers at depth N: the main thread, the only
    /// one bound to N, is the one reported, never a helper that got further.
    #[test]
    fn a_depth_limited_search_answers_at_the_requested_depth() {
        let _guard = crate::skill::lock_level();
        STOP.store(false, Ordering::Relaxed);
        PONDER.store(false, Ordering::Relaxed);
        let mut pool = silent_pool(4);
        for fen in [
            "r1bq1rk1/pp2ppbp/2np2p1/2n5/P3PP2/N1P2N2/1PB3PP/R1B1QRK1 b - - 0 1",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "8/8/3k4/8/3KP3/8/8/8 w - - 0 1",
        ] {
            let pos = Position::from_fen(fen).unwrap();
            for depth in [6, 8] {
                let (best, _) = pool.start_search(&pos, SearchLimits::depth(depth));
                assert_eq!(pool.last_best_idx, 0, "{fen} d{depth}: a helper was elected");
                let main = &pool.threads[0];
                assert_eq!(best, main.best_move);
                assert_eq!(main.root_moves[0].depth, depth, "{fen} d{depth}");
            }
        }
    }

    /// `go nodes N` is held to N nodes at any thread count, even when the share of
    /// each thread is far below the 512-node tick of the clock checks.
    #[test]
    fn a_small_node_budget_is_held_whatever_the_thread_count() {
        let _guard = crate::skill::lock_level();
        STOP.store(false, Ordering::Relaxed);
        PONDER.store(false, Ordering::Relaxed);
        let pos = Position::from_fen(
            "r1bq1rk1/pp2ppbp/2np2p1/2n5/P3PP2/N1P2N2/1PB3PP/R1B1QRK1 b - - 0 1",
        )
        .unwrap();
        for (threads, budget) in [(1, 100u64), (1, 1_000), (8, 2_000), (4, 999)] {
            let mut pool = silent_pool(threads);
            let (best, _) = pool.start_search(&pos, SearchLimits::nodes(budget));
            STOP.store(false, Ordering::Relaxed);
            assert!(best.is_ok());
            let total: u64 = pool.threads.iter().map(|td| td.nodes).sum();
            assert!(
                total >= budget && total < budget + threads as u64,
                "{threads} thread(s) searched {total} nodes for a budget of {budget}"
            );
        }
    }

    /// `tbhits` adds up the hits of every thread, those of GaiaTB included: it used to
    /// show the printing thread's alone, and no GaiaTB hit at all.
    #[cfg(gaiatb_embedded)]
    #[test]
    fn tbhits_counts_the_whole_pool_and_gaiatb() {
        let _guard = crate::skill::lock_level();
        STOP.store(false, Ordering::Relaxed);
        PONDER.store(false, Ordering::Relaxed);
        assert!(crate::dtm::init(), "embedded tables must load");
        let pos = Position::from_fen("8/8/4k3/8/2r5/8/3QK3/8 w - - 0 1").unwrap();
        let mut pool = silent_pool(4);
        pool.start_search(&pos, SearchLimits::nodes(200_000));
        STOP.store(false, Ordering::Relaxed);
        let sum: u64 = pool.threads.iter().map(|td| td.tb_hits).sum();
        assert!(pool.threads[1..].iter().any(|td| td.tb_hits > 0), "no helper hit");
        assert!(pool.threads[0].tb_hits > 0, "GaiaTB hits are not counted");
        let elected = &pool.threads[pool.last_best_idx];
        assert!(
            elected.reported_tb_hits(&pool.shared) >= sum,
            "reported {} for {sum} hits in the pool",
            elected.reported_tb_hits(&pool.shared)
        );
        assert_eq!(pool.shared.tb_hits(), sum, "the counters hold every hit after the join");
    }

    /// A lone search in a larger pool (a weakened level) reports its own nodes and
    /// the load of its one thread, not the counters a previous multi-thread search
    /// left in the idle helpers, nor a load divided by the whole pool.
    #[test]
    fn a_lone_search_after_a_pool_search_reports_its_own_counters() {
        let _guard = crate::skill::lock_level();
        STOP.store(false, Ordering::Relaxed);
        PONDER.store(false, Ordering::Relaxed);
        let pos = Position::from_fen(
            "r1bq1rk1/pp2ppbp/2np2p1/2n5/P3PP2/N1P2N2/1PB3PP/R1B1QRK1 b - - 0 1",
        )
        .unwrap();
        let mut pool = silent_pool(4);
        pool.start_search(&pos, SearchLimits::nodes(100_000));
        assert_eq!(pool.shared.searching_threads, 4);
        assert!(pool.threads[1].node_counter.get() > 0, "the helpers searched");

        crate::skill::set(10, 3);
        STOP.store(false, Ordering::Relaxed);
        pool.start_search(&pos, SearchLimits::depth(4));
        crate::skill::set(crate::skill::FULL_STRENGTH, 0);
        assert_eq!(pool.shared.searching_threads, 1, "cpuload measured against one thread");
        assert_eq!(
            pool.threads[0].nodes.max(pool.shared.nodes_searched()),
            pool.threads[0].nodes,
            "the idle helpers still count their last search"
        );
    }

    /// `ucinewgame` makes the pool a new one: once cleared, a pool that has searched
    /// other positions — or this one — searches the same tree as a pool just built,
    /// node for node. The root's optimism and fail-high count once survived it, and a
    /// test suite run in one process then depended on the order of its positions.
    #[test]
    fn a_cleared_pool_searches_like_a_new_one() {
        let _guard = crate::skill::lock_level();
        STOP.store(false, Ordering::Relaxed);
        PONDER.store(false, Ordering::Relaxed);
        let target = "3r1rk1/1b2bppp/2q1pn2/pp4Q1/3B1P2/PBN5/1PP3PP/R2R3K b - - 0 1";
        let signature = |pool: &mut ThreadPool| {
            let pos = Position::from_fen(target).unwrap();
            let (best, _) = pool.start_search(&pos, SearchLimits::depth(10));
            let td = &pool.threads[0];
            (best, td.best_score, td.nodes)
        };
        let expected = signature(&mut silent_pool(1));
        let mut pool = silent_pool(1);
        for before in [
            "2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1",
            STARTPOS,
            target,
        ] {
            let pos = Position::from_fen(before).unwrap();
            pool.start_search(&pos, SearchLimits::depth(10));
            pool.clear();
            assert_eq!(signature(&mut pool), expected, "the search of {before} leaked");
        }
    }

    /// A root without a legal move has nothing to wait for, infinite or not.
    #[test]
    fn a_mated_root_returns_at_once_on_several_threads() {
        let _guard = crate::skill::lock_level();
        STOP.store(false, Ordering::Relaxed);
        PONDER.store(false, Ordering::Relaxed);
        let pos = Position::from_fen("R5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1").unwrap();
        let mut pool = silent_pool(2);
        let start = Instant::now();
        let (best, _) = pool.start_search(&pos, SearchLimits::infinite());
        assert!(!best.is_ok(), "no legal move");
        assert!(start.elapsed() < Duration::from_secs(5), "a mated root waited for stop");
        STOP.store(false, Ordering::Relaxed);
    }
}
