//! [Time management](https://www.chessprogramming.org/Time_Management) —
//! soft and hard limits for iterative deepening.
//!
//! Soft limit: checked after each completed iteration (stop if exceeded).
//! Hard limit: checked every 2048 nodes (abort if exceeded).

use std::time::Instant;


/// Search limits parsed from a UCI `go` command.
pub enum SearchLimits {
    /// `go infinite` — search until `stop`.
    Infinite,
    /// `go depth N` — search to a fixed depth.
    Depth(i32),
    /// `go nodes N` — search a fixed number of nodes.
    Nodes(u64),
    /// `go movetime N` — search for exactly N milliseconds.
    MoveTime(u64),
    /// `go wtime/btime/winc/binc` — Fischer clock.
    Clock {
        time: u64,
        inc: u64,
        movestogo: Option<u64>,
    },
}

/// Manages time for a single search.
pub struct TimeManager {
    start: Instant,
    base_soft_limit: u64, // ms, original value before stability adjustment
    base_hard_limit: u64, // ms, original hard value before stability adjustment
    soft_limit: u64,      // ms, adjusted by stability multiplier
    hard_limit: u64,      // ms, adjusted by stability multiplier
    max_limit: u64,       // ms, absolute ceiling on time for one move
    max_depth: i32,
    max_nodes: u64,
}

impl TimeManager {
    /// Create a time manager from search limits.
    pub fn new(limits: &SearchLimits) -> TimeManager {
        let (soft, hard, max, depth, nodes) = match *limits {
            SearchLimits::Infinite => (u64::MAX, u64::MAX, u64::MAX, i32::MAX, u64::MAX),
            SearchLimits::Depth(d) => (u64::MAX, u64::MAX, u64::MAX, d, u64::MAX),
            SearchLimits::Nodes(n) => (u64::MAX, u64::MAX, u64::MAX, i32::MAX, n),
            SearchLimits::MoveTime(ms) => (ms, ms, ms, i32::MAX, u64::MAX),
            SearchLimits::Clock { time, inc, movestogo } => {
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
                (soft, hard, max, i32::MAX, u64::MAX)
            }
        };

        TimeManager {
            start: Instant::now(),
            base_soft_limit: soft,
            base_hard_limit: hard,
            soft_limit: soft,
            hard_limit: hard,
            max_limit: max,
            max_depth: depth,
            max_nodes: nodes,
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

    /// Maximum nodes to search.
    #[inline]
    pub fn max_nodes(&self) -> u64 {
        self.max_nodes
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
    /// Always computed from the original base, clamped to [1, max_limit].
    /// Both soft and hard limits are scaled by stability, but
    /// never exceed the absolute ceiling (max_limit = 60% of clock).
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
