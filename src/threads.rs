//! Thread pool and per-thread state for [Lazy SMP](https://www.chessprogramming.org/Lazy_SMP)
//! parallel search.
//!
//! Each thread runs iterative deepening independently on the same root position.
//! The only shared state is the lockless [transposition table](crate::tt::TT)
//! and the atomic stop flag. All history tables, killers, search stack, and PV
//! are per-thread.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use crate::history::{
    ButterflyHistory, CaptureHistory, ContCorrectionHistory, ContinuationHistory, Countermoves,
    Killers, PawnHistory, PieceToCorrTable, SharedCorrectionHistory,
    PieceToTable,
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
use crate::timeman::{SearchLimits, TimeManager};
use crate::tt::TT;
use crate::types::*;

/// Number of sentinel entries before ply 0 in the search stack.
/// Allows safe access to `stack[ply - N]` for continuation history lookups.
pub const SS_OFFSET: usize = 8;

const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// ============================================================
// RootMove
// ============================================================

/// Per-root-move state for Multi-PV search.
/// Each legal root move stores its own score, PV, and aspiration average.
#[derive(Clone)]
pub struct RootMove {
    pub mv: Move,
    /// Score from current depth iteration (-SCORE_INFINITE if not yet searched).
    pub score: i32,
    /// Score from previous depth iteration (for aspiration centering on incomplete depths).
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
}

impl RootMove {
    pub fn new(mv: Move) -> Self {
        let mut pv = [Move::NONE; MAX_PLY + 1];
        pv[0] = mv;
        RootMove {
            mv,
            score: -SCORE_INFINITE,
            previous_score: -SCORE_INFINITE,
            avg_score: SCORE_NONE,
            pv,
            pv_len: 1,
            sel_depth: 0,
            tb_rank: 0,
            tb_score: 0,
            nodes_spent: 0,
        }
    }
}

/// Syzygy tablebase configuration for a search.
#[cfg(feature = "syzygy")]
#[derive(Clone, Copy, Default)]
pub struct TbConfig {
    /// Root position is in loaded tablebases.
    pub root_in_tb: bool,
    /// Max piece count for in-tree TB probing. 0 = DTZ available, skip in-tree probes.
    /// >0 = WDL-only fallback, probe in search at this cardinality.
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
}

impl SharedState {
    /// `threads` sizes the correction tables so the entries per thread, and with
    /// them the collision rate, stay what they are at one thread. Every caller that
    /// searches alone passes 1.
    pub fn new(hash_mb: usize, threads: usize) -> Self {
        SharedState {
            tt: TT::new(hash_mb),
            corr: SharedCorrectionHistory::new(threads),
        }
    }

    /// Clear everything shared (`ucinewgame`, and between bench runs).
    pub fn clear(&mut self) {
        self.tt.clear();
        self.corr.clear();
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
    /// Local cache of ponder state. Avoids atomic load every 2048 nodes.
    /// Set in `prepare_search()`, cleared on `ponderhit` detection.
    pub pondering: bool,
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
    /// Best move stability counter for time management (consecutive same best move).
    pub bm_stability: i32,
    /// Previous iteration's best move for stability tracking.
    pub prev_best_move: Move,
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
    let t = threads as u64;
    match *limits {
        SearchLimits::Nodes(n) => Some(SearchLimits::Nodes(n.div_ceil(t).max(1))),
        // div_ceil is monotonic, so soft <= hard survives the division.
        SearchLimits::SoftNodes { soft, hard } => Some(SearchLimits::SoftNodes {
            soft: soft.div_ceil(t).max(1),
            hard: hard.div_ceil(t).max(1),
        }),
        _ => None,
    }
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
            tm: TimeManager::new(&SearchLimits::Infinite),
            root_depth: 0,
            completed_depth: 0,
            best_move: Move::NONE,
            best_score: -SCORE_INFINITE,
            search_deadline: 0,
            thread_count: 1,
            per_thread_nodes: false,
            stopped: false,
            seldepth: 0,
            pondering: false,
            root_moves: Vec::new(),
            multi_pv: 1,
            reported_multi_pv: 1,
            pv_index: 0,
            tb_hits: 0,
            #[cfg(feature = "syzygy")]
            tb_config: TbConfig::default(),
            datagen_mode: false,
            bm_stability: 0,
            prev_best_move: Move::NONE,
            used_lazy_eval: false,
            skill: crate::skill::Snapshot::FULL_STRENGTH,
            optimism: [0; 2],
            last_iteration_pv: [Move::NONE; MAX_PLY + 1],
            last_iteration_pv_len: 0,
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
        debug_assert!(ply + SS_OFFSET < self.stack.len(),
            "ss: ply {} + SS_OFFSET {} >= stack len {}", ply, SS_OFFSET, self.stack.len());
        &self.stack[ply + SS_OFFSET]
    }

    /// Mutable access to the search stack entry at a given ply.
    #[inline]
    pub fn ss_mut(&mut self, ply: usize) -> &mut StackEntry {
        debug_assert!(ply < MAX_PLY, "ss_mut: ply {} >= MAX_PLY", ply);
        debug_assert!(ply + SS_OFFSET < self.stack.len(),
            "ss_mut: ply {} + SS_OFFSET {} >= stack len {}", ply, SS_OFFSET, self.stack.len());
        &mut self.stack[ply + SS_OFFSET]
    }

    /// Get continuation history score for a single ply offset.
    /// Sentinel pointers return 0 (all-zero table), eliminating null checks.
    #[inline]
    pub fn conthist(&self, ply: usize, offset: usize, piece: Piece, to: Square) -> i32 {
        let base = ply + SS_OFFSET;
        debug_assert!(base >= offset, "conthist: base {} < offset {}", base, offset);
        ContinuationHistory::get(
            self.stack[base - offset].conthist_ptr as *const _,
            piece,
            to,
        )
    }

    /// Check time/node limits every 2048 nodes.
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
            }

            // Skip time/node limits while pondering (search indefinitely).
            // In datagen, and whenever a node budget was split across the pool, the
            // limit is this thread's alone: stop it like the deadline below — a thread
            // spending its own share must not kill its siblings through the global STOP.
            if !self.pondering
                && (self.tm.should_stop_hard() || self.nodes >= self.tm.max_nodes())
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
            self.tm.cap_depth(crate::skill::ceiling(&self.skill, self.pos.key));
        }
    }

    /// Prepare this thread for a new search.
    pub fn prepare_search(&mut self, pos: &Position, limits: &SearchLimits) {
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
        self.completed_depth = 0;
        self.seldepth = 0;
        self.best_move = Move::NONE;
        self.best_score = -SCORE_INFINITE;
        self.stopped = false;
        self.pondering = PONDER.load(Ordering::Relaxed);
        self.bm_stability = 0;
        self.prev_best_move = Move::NONE;
        self.killers.clear();
        self.pv_index = 0;
        self.tb_hits = 0;
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
            self.root_moves.push(RootMove::new(buf[i]));
        }
        // Default best_move to first legal move so we always have a valid
        // response even if stopped before completing depth 1.
        self.best_move = if count > 0 { buf[0] } else { Move::NONE };

        // Time manager: real limits for the main thread, infinite for the helpers —
        // except for a node budget, which is divided evenly instead (see
        // `split_node_limit`). At one thread nothing is divided and nothing changes.
        self.per_thread_nodes = false;
        match split_node_limit(limits, self.thread_count).filter(|_| self.thread_count > 1) {
            Some(share) => {
                self.per_thread_nodes = true;
                self.tm = TimeManager::new(&share);
            }
            None if self.id == 0 => self.tm = TimeManager::new(limits),
            None => self.tm = TimeManager::new(&SearchLimits::Infinite),
        }
        self.apply_skill_ceiling();
        // The upper rungs need to see more than one root move to have anything to choose
        // between. Raised, never lowered: someone who asked for a multi-PV analysis gets
        // the lines they asked for.
        self.reported_multi_pv = self.multi_pv;
        if self.skill.active {
            self.multi_pv = self.multi_pv.max(self.skill.rung.variety_moves.max(1) as usize);
        }

        // Reset search stack
        let sentinel_ptr =
            &*self.sentinel_conthist as *const PieceToTable as *mut PieceToTable;
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
        }
    }

    /// Launch parallel search and return the best move + optional ponder move.
    ///
    /// Uses `std::thread::scope` for scoped lifetime management:
    /// helper threads (id > 0) search with infinite limits until `stop`,
    /// main thread (id 0) searches with real time limits.
    pub fn start_search(&mut self, pos: &Position, limits: SearchLimits) -> (Move, Option<Move>) {
        STOP.store(false, Ordering::Relaxed);
        self.shared.tt.new_search();
        // The info lines report the whole search, not the main thread's share: with
        // helpers running, every thread feeds the shared counter, zeroed here so a
        // single-thread search that follows never reads a stale total.
        NODES_SEARCHED.store(0, Ordering::Relaxed);

        // Prepare all threads
        for td in &mut self.threads {
            td.multi_pv = self.multi_pv;
            td.prepare_search(pos, &limits);
        }

        // A weakened engine opens out of the book rather than searching, which is what
        // keeps a given level from playing the same game every time. Full strength
        // never gets here — `book::choice` answers `None` before it looks at anything.
        let analysing = self.multi_pv > 1
            || matches!(limits, SearchLimits::Infinite)
            || PONDER.load(Ordering::Relaxed);
        if let Some(mv) = crate::book::choice(pos, analysing) {
            return (mv, None);
        }

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

    /// Select the best thread via weighted vote aggregation.
    ///
    /// Each thread votes for its best move with weight =
    /// `(score - min_score + 10) * completed_depth`. Votes are aggregated
    /// per move. Special handling for decisive scores (mates, TB wins).
    /// Tiebreak uses PV truncation filter (`pv_len > 2`).
    fn select_best_thread(&self) -> usize {
        // Skip voting for single thread or multi-PV mode
        if self.threads.len() <= 1 || self.multi_pv > 1 {
            return 0;
        }

        // Find min score across all threads that completed at least one iteration
        let min_score = self.threads.iter()
            .filter(|t| t.completed_depth > 0 && !t.root_moves.is_empty())
            .map(|t| t.root_moves[0].score)
            .min()
            .unwrap_or(-SCORE_INFINITE);

        // Vote weight: score advantage (with +10 offset) × depth
        let vote_value = |t: &ThreadData| -> i64 {
            (t.root_moves[0].score as i64 - min_score as i64 + 10)
                * t.completed_depth as i64
        };

        // Aggregate votes per move via small Vec (max entries = thread count)
        let mut votes: Vec<(Move, i64)> = Vec::with_capacity(self.threads.len());
        for t in &self.threads {
            if t.completed_depth == 0 || t.root_moves.is_empty() { continue; }
            let mv = t.root_moves[0].mv;
            if let Some(entry) = votes.iter_mut().find(|(m, _)| *m == mv) {
                entry.1 += vote_value(t);
            } else {
                votes.push((mv, vote_value(t)));
            }
        }

        let get_votes = |mv: Move| -> i64 {
            votes.iter().find(|(m, _)| *m == mv).map_or(0, |e| e.1)
        };

        let is_win = |s: i32| s >= SCORE_TB_WIN_IN_MAX;
        let is_loss = |s: i32| s != -SCORE_INFINITE && s <= -SCORE_TB_WIN_IN_MAX;

        let mut best = 0usize;
        for i in 1..self.threads.len() {
            let ti = &self.threads[i];
            if ti.completed_depth == 0 || ti.root_moves.is_empty() { continue; }

            let best_td = &self.threads[best];
            let best_score = best_td.root_moves[0].score;
            let this_score = ti.root_moves[0].score;

            if is_win(best_score) {
                // Current best is winning: prefer shorter mate (higher score)
                if this_score > best_score { best = i; }
            } else if is_loss(best_score) {
                // Current best is losing: prefer longer defense (lower score)
                if is_loss(this_score) && this_score < best_score { best = i; }
            } else if is_win(this_score) || is_loss(this_score) {
                // This thread found a decisive result, take it
                best = i;
            } else {
                // Normal: compare aggregated votes, tiebreak by vote_value × (pv_len > 2)
                let best_mv_votes = get_votes(best_td.root_moves[0].mv);
                let this_mv_votes = get_votes(ti.root_moves[0].mv);

                let pv_filtered_value = |t: &ThreadData| -> i64 {
                    vote_value(t) * (t.root_moves[0].pv_len > 2) as i64
                };

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
    use crate::types::Move;

    /// A quiet middlegame position: deep enough that a few thousand nodes settle
    /// nothing, so no thread can finish its work before spending its budget.
    const MIDGAME: &str = "r1bq1rk1/pp2ppbp/2np2p1/2n5/P3PP2/N1P2N2/1PB3PP/R1B1QRK1 b - - 0 1";

    #[test]
    fn a_node_budget_divides_evenly_and_nothing_else_does() {
        // A clock belongs to the main thread; only a node count is shared out.
        assert!(matches!(
            split_node_limit(&SearchLimits::Nodes(40_000), 4),
            Some(SearchLimits::Nodes(10_000))
        ));
        assert!(matches!(
            split_node_limit(&SearchLimits::SoftNodes { soft: 1_000, hard: 100_000 }, 8),
            Some(SearchLimits::SoftNodes { soft: 125, hard: 12_500 })
        ));
        assert!(split_node_limit(&SearchLimits::Depth(12), 4).is_none());
        assert!(split_node_limit(&SearchLimits::MoveTime(500), 4).is_none());
        assert!(split_node_limit(&SearchLimits::Infinite, 4).is_none());
        assert!(split_node_limit(&SearchLimits::Clock { time: 1000, inc: 10, movestogo: None }, 4)
            .is_none());

        // Rounded up, and never down to nothing: a thread with a budget of zero would
        // return before it had a move to offer.
        assert!(matches!(
            split_node_limit(&SearchLimits::Nodes(10), 4),
            Some(SearchLimits::Nodes(3))
        ));
        assert!(matches!(
            split_node_limit(&SearchLimits::Nodes(1), 24),
            Some(SearchLimits::Nodes(1))
        ));

        // The order the time manager asserts on survives the division.
        let split = split_node_limit(&SearchLimits::SoftNodes { soft: 7, hard: 7 }, 3);
        let Some(SearchLimits::SoftNodes { soft, hard }) = split else { panic!("not split") };
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
        let (best, _) = pool.start_search(&pos, SearchLimits::Nodes(BUDGET));
        assert!(best != Move::NONE, "search returned no move");

        let share = BUDGET / THREADS as u64;
        for td in &pool.threads {
            // The limit is checked every 512 nodes, so a thread can overshoot by that
            // much plus the tail of the node it was in.
            assert!(
                td.nodes >= share && td.nodes < share + 2048,
                "thread {} searched {} nodes, expected about {}",
                td.id, td.nodes, share
            );
            assert!(td.per_thread_nodes, "thread {} did not own a share", td.id);
            assert!(td.stopped, "thread {} was not stopped by its own budget", td.id);
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
        pool.start_search(&pos, SearchLimits::Nodes(40_000));
        let td = &pool.threads[0];
        assert!(!td.per_thread_nodes, "a lone thread must not think it owns a share");
        assert!(
            td.nodes >= 40_000 && td.nodes < 42_048,
            "lone thread searched {} nodes for a budget of 40000",
            td.nodes
        );
    }
}
