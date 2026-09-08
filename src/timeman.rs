//! [Time management](https://www.chessprogramming.org/Time_Management) —
//! soft and hard limits for iterative deepening.
//!
//! Soft limit: checked after each completed iteration (stop if exceeded).
//! Hard limit: checked every 2048 nodes (abort if exceeded).

use crate::time::Instant;

/// Search limits parsed from a UCI `go` command.
///
/// Every field that is `Some` constrains the search; fields left empty do not.
/// That is the protocol: `go depth 20 movetime 1000` stops at whichever fires
/// first. `infinite` suppresses time and node stops (`stop` only), but a depth
/// or mate limit still caps iterative deepening.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchLimits {
    /// `go infinite` — do not stop on time or nodes.
    pub infinite: bool,
    /// `go depth N`.
    pub depth: Option<i32>,
    /// `go nodes N` (hard node ceiling).
    pub nodes: Option<u64>,
    /// `go softnodes N` — checked at iteration boundaries.
    pub soft_nodes: Option<u64>,
    /// Hard ceiling paired with [`Self::soft_nodes`].
    pub hard_nodes: Option<u64>,
    /// `go movetime N`.
    pub movetime: Option<u64>,
    /// Clock of the side to move, milliseconds (`wtime`/`btime`).
    pub time: Option<u64>,
    /// Increment of the side to move (`winc`/`binc`).
    pub inc: u64,
    /// Moves to the next time control.
    pub movestogo: Option<u64>,
    /// `go mate N` — stop once a mate in at most N moves is found.
    pub mate: Option<i32>,
}

impl Default for SearchLimits {
    fn default() -> Self {
        Self {
            infinite: false,
            depth: None,
            nodes: None,
            soft_nodes: None,
            hard_nodes: None,
            movetime: None,
            time: None,
            inc: 0,
            movestogo: None,
            mate: None,
        }
    }
}

impl SearchLimits {
    pub fn infinite() -> Self {
        Self {
            infinite: true,
            ..Self::default()
        }
    }

    pub fn depth(d: i32) -> Self {
        debug_assert!(d > 0, "depth limit {d} must be positive");
        Self {
            depth: Some(d),
            ..Self::default()
        }
    }

    pub fn nodes(n: u64) -> Self {
        Self {
            nodes: Some(n),
            ..Self::default()
        }
    }

    pub fn soft_nodes(soft: u64, hard: u64) -> Self {
        debug_assert!(soft <= hard, "soft nodes {soft} > hard {hard}");
        Self {
            soft_nodes: Some(soft),
            hard_nodes: Some(hard),
            ..Self::default()
        }
    }

    pub fn movetime(ms: u64) -> Self {
        Self {
            movetime: Some(ms),
            ..Self::default()
        }
    }

    pub fn clock(time: u64, inc: u64, movestogo: Option<u64>) -> Self {
        Self {
            time: Some(time),
            inc,
            movestogo,
            ..Self::default()
        }
    }
}

/// Hard node ceiling = soft budget x this factor, for the limits that derive
/// one from the other (`go softnodes`, `UseSoftNodes`, datagen).
pub const SOFT_NODES_HARD_FACTOR: u64 = 100;

fn clock_bounds(time: u64, inc: u64, movestogo: Option<u64>) -> (u64, u64) {
    let overhead = crate::tune::MOVE_OVERHEAD() as u64;
    let moves = movestogo.unwrap_or(24);
    // Absolute ceiling: 60% of clock minus overhead.
    let max = (time * 60 / 100).saturating_sub(overhead);
    // Hard: 46% of clock, capped by max.
    let hard = (time * 46 / 100).min(max);
    // Soft: (clock/moves + inc*94%) - overhead, scaled to 73%.
    // Capped by hard.
    let computed = if movestogo.is_some() {
        let divisor = moves.clamp(2, 24);
        time / divisor
    } else {
        time / moves + inc * 94 / 100 - overhead
    };
    let soft = (computed.min(max) * 73 / 100).min(hard);
    (soft, hard)
}

/// Manages time for a single search.
pub struct TimeManager {
    start: Instant,
    base_soft_limit: u64, // ms, original value before stability adjustment
    soft_limit: u64,      // ms, adjusted by stability multiplier
    hard_limit: u64,      // ms, fixed ceiling for one move
    max_depth: i32,
    max_nodes: u64,
    soft_nodes: u64,
}

impl TimeManager {
    /// Create a time manager from search limits.
    pub fn new(limits: &SearchLimits) -> TimeManager {
        let max_depth = limits.depth.unwrap_or(i32::MAX);

        let mut max_nodes = u64::MAX;
        let mut soft_nodes = u64::MAX;
        if let Some(n) = limits.nodes {
            max_nodes = n;
        }
        if let Some(s) = limits.soft_nodes {
            debug_assert!(
                limits.hard_nodes.is_none_or(|h| s <= h),
                "soft nodes {s} > hard {:?}",
                limits.hard_nodes
            );
            soft_nodes = s;
            if let Some(h) = limits.hard_nodes {
                max_nodes = max_nodes.min(h);
            }
        }

        let mut soft = u64::MAX;
        let mut hard = u64::MAX;
        // `go infinite` must not exit without `stop`; time and node budgets are
        // then ignored. A depth or mate cap still bounds iterative deepening.
        if !limits.infinite {
            if let Some(ms) = limits.movetime {
                soft = soft.min(ms);
                hard = hard.min(ms);
            }
            if let Some(time) = limits.time {
                let (clock_soft, clock_hard) = clock_bounds(time, limits.inc, limits.movestogo);
                soft = soft.min(clock_soft);
                hard = hard.min(clock_hard);
            }
        } else {
            max_nodes = u64::MAX;
            soft_nodes = u64::MAX;
        }

        TimeManager {
            start: Instant::now(),
            max_depth,
            base_soft_limit: soft,
            soft_limit: soft,
            hard_limit: hard,
            max_nodes,
            soft_nodes,
        }
    }

    /// Elapsed time in milliseconds since search started.
    #[inline]
    pub fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }

    /// Maximum depth to search.
    #[inline]
    pub fn max_depth(&self) -> i32 {
        self.max_depth
    }

    /// Lowers the ceiling, never raises it.
    ///
    /// A weakened opponent is capped here rather than inside the search, so the ceiling
    /// applies whatever kind of limit was asked for. It is applied once the root position
    /// is known, because how far a level looks depends on the position in front of it.
    #[inline]
    pub fn cap_depth(&mut self, ceiling: i32) {
        self.max_depth = self.max_depth.min(ceiling);
    }

    /// Maximum nodes to search.
    #[inline]
    pub fn max_nodes(&self) -> u64 {
        self.max_nodes
    }

    /// Soft node budget, checked at iteration boundaries.
    #[inline]
    pub fn soft_nodes(&self) -> u64 {
        self.soft_nodes
    }

    /// Check after each completed iteration: should we stop?
    #[inline]
    pub fn should_stop_soft(&self) -> bool {
        self.elapsed_ms() >= self.soft_limit
    }

    /// Check during search (every 512 nodes): should we abort?
    #[inline]
    pub fn should_stop_hard(&self) -> bool {
        self.elapsed_ms() >= self.hard_limit
    }

    /// Adjust soft limit by a multiplier (for best-move / score stability).
    /// Always computed from the original base, so repeated calls do not compound,
    /// and clamped to [1, hard_limit]: only the soft bound moves with stability,
    /// the hard one stays where the clock put it.
    ///
    /// Reference: CPW — Search Progression § Soft bound (best move stability, eval stability)
    #[inline]
    pub fn adjust_soft_limit(&mut self, multiplier: f64) {
        let adjusted = (self.base_soft_limit as f64 * multiplier) as u64;
        // Soft limit scaled by stability, capped by hard (which stays fixed).
        self.soft_limit = adjusted.max(1).min(self.hard_limit);
    }

    /// Restart the timer (called on `ponderhit` to begin real time allocation).
    #[inline]
    pub fn restart(&mut self) {
        self.start = Instant::now();
    }
}
