//! [Time management](https://www.chessprogramming.org/Time_Management) —
//! how much of the clock one move is allowed to cost.
//!
//! Two budgets are derived once per `go`, from the clock, the increment and how
//! far into the game the position is:
//!
//! * `optimum` is what the move is *expected* to cost. It is not a deadline.
//!   Every iteration the search rescales it by how much the position is still
//!   moving underneath it — a best move that keeps changing, a score that is
//!   falling, a search still spread over many root moves all buy more time —
//!   and the rescaled value becomes the soft limit tested at iteration
//!   boundaries (see `search::search`).
//! * `maximum` is the deadline, tested inside the search every 512 nodes. It is
//!   fixed once the clock has been read: only the soft bound follows the
//!   position.
//!
//! The budget grows with the game ply. The clock has to last a whole game, but
//! the opening is the cheapest part of it to play and the least likely to be
//! decided there, so a fixed fraction of the clock per move would spend the
//! most where it buys the least.

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

// ─── Clock scaling ──────────────────────────────────────────────────────────
// Every constant below is a fraction or an exponent of the reference formula;
// they are tuned as a set and are not independent knobs.

/// Move horizon in centi-moves when the host gives no `movestogo`.
const CENTI_MTG_OPEN: i64 = 5051;
/// Move horizon ceiling when the host does give one (50 moves).
const CENTI_MTG_MAX: i64 = 5000;
/// Below one second of clock the horizon shrinks with it, at this rate per ms.
const CENTI_MTG_PER_MS: f64 = 5.051;

/// Per-game budget tilt, `MUL * log10(time_left) - SUB`. Computed once per game:
/// a long game may afford a larger share of its clock per move than a short one.
const ADJUST_MUL: f64 = 0.3128;
const ADJUST_SUB: f64 = 0.4354;

/// `optimum` share of the horizon: `OPT_BASE + (ply + PLY_OFFSET)^PLY_EXP * c`,
/// where `c` itself decays with the log of the clock.
const OPT_BASE: f64 = 0.0121431;
const PLY_OFFSET: f64 = 2.94693;
const PLY_EXP: f64 = 0.461073;
const OPT_C_BASE: f64 = 0.0032116;
const OPT_C_MUL: f64 = 0.000321123;
const OPT_C_MAX: f64 = 0.00508017;
/// A healthy increment can push the horizon past the clock actually in hand;
/// this caps `optimum` at a share of the real remaining time.
const OPT_CLOCK_CAP: f64 = 0.213035;

/// `maximum` as a multiple of `optimum`, growing with the ply and the clock.
const MAX_C_BASE: f64 = 3.3977;
const MAX_C_MUL: f64 = 3.03950;
const MAX_C_MIN: f64 = 2.94761;
const MAX_SCALE_CAP: f64 = 6.67704;
const MAX_PLY_DIV: f64 = 11.9847;

/// Same two shares under an explicit `movestogo`, where the horizon is known.
const MTG_OPT_BASE: f64 = 0.88;
const MTG_OPT_PLY_DIV: f64 = 116.4;
const MTG_OPT_CLOCK_CAP: f64 = 0.88;
const MTG_MAX_BASE: f64 = 1.3;
const MTG_MAX_MUL: f64 = 0.11;

/// Absolute ceiling on one move, as a share of the clock, and the safety margin
/// taken off it afterwards (ms).
const HARD_CLOCK_FRAC: f64 = 0.825179;
const HARD_MARGIN_MS: i64 = 10;

/// Hard node ceiling = soft budget x this factor, for the limits that derive
/// one from the other (`go softnodes`, `UseSoftNodes`, datagen).
pub const SOFT_NODES_HARD_FACTOR: u64 = 100;

/// Per-game budget tilt before the first move that carries a clock computes it.
/// Negative is the "unset" marker, and also what a clock too short to tilt
/// yields, so such a move recomputes it next time instead of freezing it.
pub const TIME_ADJUST_UNSET: f64 = -1.0;

// ─── Soft-bound rescaling ───────────────────────────────────────────────────
// Applied once per iteration to `optimum` (see `search::search`). Same caveat as
// above: one tuned set, not independent knobs.

/// Falling score: how much a score below the previous move's, and below the one
/// four iterations back, widens the budget. Percent, then clamped.
pub const TM_FALL_BASE: f64 = 11.85;
pub const TM_FALL_PREV: f64 = 2.24;
pub const TM_FALL_ITER: f64 = 0.93;
pub const TM_FALL_MIN: f64 = 0.57;
pub const TM_FALL_MAX: f64 = 1.70;

/// Best-move stability: a smooth step in depth around `TM_STABLE_CENTER` plies
/// past the iteration where the best move last changed.
pub const TM_STABLE_BASE: f64 = 0.66;
pub const TM_STABLE_SPAN: f64 = 0.85;
pub const TM_STABLE_OFFSET: f64 = 0.98;
pub const TM_STABLE_SLOPE: f64 = 0.51;
pub const TM_STABLE_CENTER: f64 = 12.15;
/// The step of this move against the one the previous move ended on: a position
/// that settled last move is cheaper to settle again.
pub const TM_REDUCTION_BASE: f64 = 1.43;
pub const TM_REDUCTION_DIV: f64 = 2.28;

/// Root instability, per best-move change per thread.
pub const TM_INSTABILITY_BASE: f64 = 1.02;
pub const TM_INSTABILITY_MUL: f64 = 2.14;

/// Share of the tree, in hundred-thousandths, above which the best move is
/// judged to have no competition left, and the budget shrinks by `TM_EFFORT_MUL`.
pub const TM_EFFORT_THRESHOLD: u64 = 93_340;
pub const TM_EFFORT_MUL: f64 = 0.76;

/// Ceiling on a position with a single legal move (ms). For whoever is watching:
/// the search has nothing to decide.
pub const TM_FORCED_MOVE_MS: f64 = 502.0;

/// Time reduction carried into the first move of a game. Slightly below one, so
/// the opening move — the one with no previous search to compare against — costs
/// a little more than a mid-game move of the same nominal budget.
pub const INITIAL_TIME_REDUCTION: f64 = 0.85;

/// Optimum and maximum budget for one move, in milliseconds.
///
/// `overhead` is what the host is expected to eat per move, and the formula takes
/// it far more seriously than its size suggests: it is subtracted once for every
/// move of the horizon, twice over, so at blitz ten milliseconds of it are worth
/// a tenth of the budget.
///
/// `game_ply` is the number of half-moves played since the start of the game,
/// not since the root: how far into the game we are is exactly what tilts the
/// budget. `time_adjust` is per-game state, computed on the first move that
/// carries a clock and reused afterwards; a negative value means "not computed
/// yet", which is also what a clock too short to tilt produces, so such a
/// position recomputes it on the next move rather than freezing a bad value.
fn clock_bounds(
    time: u64,
    inc: u64,
    movestogo: Option<u64>,
    game_ply: i32,
    overhead: i64,
    time_adjust: &mut f64,
) -> (u64, u64) {
    debug_assert!(game_ply >= 0, "game ply {game_ply} must not be negative");
    debug_assert!(
        overhead >= 0,
        "move overhead {overhead} must not be negative"
    );

    let t = time as i64;
    debug_assert!(t >= 0, "clock {t} must not be negative");

    // No clock at all: no budget. Taken before the scaling below, whose
    // logarithms are only defined on a clock that has something left.
    if t == 0 {
        return (0, 0);
    }

    // `movestogo 0` is the protocol's way of saying "no move horizon"; taken
    // literally it would divide by zero below.
    let mtg = movestogo.filter(|&m| m > 0);
    let mut centi_mtg = match mtg {
        Some(m) => ((m as i64).saturating_mul(100)).min(CENTI_MTG_MAX),
        None => CENTI_MTG_OPEN,
    };
    // Under a second, shrink the horizon with the clock: pretending fifty moves
    // are left when a tenth of a second is would allocate nothing at all.
    if t < 1000 {
        centi_mtg = (t as f64 * CENTI_MTG_PER_MS) as i64;
    }
    debug_assert!(centi_mtg > 0, "centi-mtg {centi_mtg} must be positive");

    // What the clock is really worth over the horizon: every move to come pays
    // the overhead twice over, and every one but this move earns an increment.
    let time_left =
        (t + (inc as i64 * (centi_mtg - 100) - overhead * (200 + centi_mtg)) / 100).max(1);

    let (opt_scale, max_scale) = if mtg.is_none() {
        if *time_adjust < 0.0 {
            *time_adjust = ADJUST_MUL * (time_left as f64).log10() - ADJUST_SUB;
        }
        let log_secs = (t as f64 / 1000.0).log10();
        let opt_c = (OPT_C_BASE + OPT_C_MUL * log_secs).min(OPT_C_MAX);
        let max_c = (MAX_C_BASE + MAX_C_MUL * log_secs).max(MAX_C_MIN);
        let opt = (OPT_BASE + (game_ply as f64 + PLY_OFFSET).powf(PLY_EXP) * opt_c)
            .min(OPT_CLOCK_CAP * t as f64 / time_left as f64)
            * *time_adjust;
        let max = MAX_SCALE_CAP.min(max_c + game_ply as f64 / MAX_PLY_DIV);
        (opt, max)
    } else {
        let moves = centi_mtg as f64 / 100.0;
        let opt = ((MTG_OPT_BASE + game_ply as f64 / MTG_OPT_PLY_DIV) / moves)
            .min(MTG_OPT_CLOCK_CAP * t as f64 / time_left as f64);
        let max = MTG_MAX_BASE + MTG_MAX_MUL * moves;
        (opt, max)
    };

    // Truncate to whole milliseconds here, then scale the truncated value: the
    // maximum is a multiple of the budget actually granted, not of a real.
    let optimum = (opt_scale * time_left as f64) as i64;
    let maximum = ((HARD_CLOCK_FRAC * t as f64 - overhead as f64).min(max_scale * optimum as f64)
        as i64)
        - HARD_MARGIN_MS;

    // Both can come out negative on a clock that is nearly gone. Zero then, and
    // the search plays whatever depth one found rather than flagging.
    (optimum.max(0) as u64, maximum.max(0) as u64)
}

/// Manages time for a single search.
pub struct TimeManager {
    total_start: Instant,
    start: Instant,
    /// A clock was given, so the per-iteration rescaling applies. False for
    /// `movetime`, `depth`, `nodes` and `infinite`, whose budgets are literal.
    has_clock: bool,
    optimum: u64,    // ms, expected cost of this move before rescaling
    maximum: u64,    // ms, deadline for this move, fixed once the clock is read
    soft_limit: u64, // ms, rescaled optimum, tested at iteration boundaries
    max_depth: i32,
    max_nodes: u64,
    soft_nodes: u64,
}

impl TimeManager {
    /// Create a time manager from search limits.
    ///
    /// `game_ply` is half-moves since the start of the game; `time_adjust` is
    /// the caller's per-game state (see [`clock_bounds`]).
    pub fn new(limits: &SearchLimits, game_ply: i32, time_adjust: &mut f64) -> TimeManager {
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

        let mut has_clock = false;
        let mut optimum = u64::MAX;
        let mut maximum = u64::MAX;

        // `go infinite` must not exit without `stop`; time and node budgets are
        // then ignored. A depth or mate cap still bounds iterative deepening.
        if !limits.infinite {
            if let Some(time) = limits.time {
                has_clock = true;
                let (opt, max) = clock_bounds(
                    time,
                    limits.inc,
                    limits.movestogo,
                    game_ply,
                    crate::tune::MOVE_OVERHEAD() as i64,
                    time_adjust,
                );
                optimum = opt;
                maximum = max;
            }
            if let Some(ms) = limits.movetime {
                // `movetime` is a deadline, never a licence to think longer: it
                // lowers the ceiling and leaves the expected cost alone.
                maximum = maximum.min(ms);
                if !has_clock {
                    optimum = ms;
                }
            }
        } else {
            max_nodes = u64::MAX;
            soft_nodes = u64::MAX;
        }

        let now = Instant::now();
        TimeManager {
            total_start: now,
            start: now,
            has_clock,
            optimum,
            maximum,
            // Until the first iteration rescales it, the ceiling is the only
            // bound — which is what a literal `movetime` budget wants for good.
            soft_limit: maximum,
            max_depth,
            max_nodes,
            soft_nodes,
        }
    }

    /// A manager that never stops on time or nodes (helper threads).
    pub fn infinite() -> TimeManager {
        let mut unused = -1.0;
        TimeManager::new(&SearchLimits::infinite(), 0, &mut unused)
    }

    /// Elapsed time in milliseconds since search started.
    #[inline]
    pub fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }

    /// Elapsed wall time since `go`, including time spent pondering before restart.
    #[inline]
    pub fn total_elapsed_ms(&self) -> u64 {
        self.total_start.elapsed().as_millis() as u64
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

    /// Whether a clock was given, i.e. whether the soft bound is worth rescaling.
    #[inline]
    pub fn has_clock(&self) -> bool {
        self.has_clock
    }

    /// Expected cost of this move before any rescaling, in milliseconds.
    #[inline]
    pub fn optimum_ms(&self) -> u64 {
        self.optimum
    }

    /// Check after each completed iteration: should we stop?
    #[inline]
    pub fn should_stop_soft(&self) -> bool {
        self.elapsed_ms() > self.soft_limit
    }

    /// Check during search (every 512 nodes): should we abort?
    #[inline]
    pub fn should_stop_hard(&self) -> bool {
        self.elapsed_ms() > self.maximum
    }

    /// Set the soft bound for the iterations to come, capped by the deadline.
    ///
    /// Called once per iteration with the rescaled optimum. Only the soft bound
    /// follows the position; the ceiling stays where the clock put it.
    #[inline]
    pub fn set_soft_limit(&mut self, ms: u64) {
        self.soft_limit = ms.min(self.maximum);
    }

    /// Restart the timer (called on `ponderhit` to begin real time allocation).
    #[inline]
    pub fn restart(&mut self) {
        self.start = Instant::now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Budgets for a clock, a ply, a move horizon and an overhead, in milliseconds.
    /// Recomputed independently rather than read off the implementation: the point
    /// is to pin the integer truncations, which is where a port of a real-valued
    /// formula goes wrong. Two overheads, because the formula is far more sensitive
    /// to that one number than its size suggests.
    #[test]
    fn budgets_match_the_reference_table() {
        // (clock, increment, movestogo, game ply, overhead, optimum, maximum)
        const GOLDEN: &[(u64, u64, Option<u64>, i32, i64, u64, u64)] = &[
            (2000, 100, None, 0, 10, 85, 356),
            (2000, 100, None, 40, 10, 149, 984),
            (2000, 100, None, 80, 10, 182, 1205),
            (10000, 100, None, 20, 10, 338, 2246),
            (60000, 600, None, 40, 10, 3330, 22224),
            (180000, 2000, None, 40, 10, 12153, 81136),
            // A large increment caps the horizon at a share of the real clock,
            // which makes this budget smaller than the ply curve alone would ask.
            (3000, 2000, None, 0, 10, 722, 2455),
            (60000, 0, None, 0, 10, 1155, 7701),
            (1000, 0, None, 10, 10, 4, 6),
            (500, 0, None, 4, 10, 1, 0),
            (60000, 600, Some(40), 20, 10, 2182, 12427),
            // Last move before the control: spend nearly everything, still capped
            // by the share of the clock a single move may ever cost.
            (60000, 600, Some(1), 0, 10, 52773, 49490),
            (0, 0, None, 0, 10, 0, 0),
            // At our own default. Note what a few milliseconds do at blitz and how
            // little they do at a long time control.
            (2000, 100, None, 0, 40, 61, 253),
            (2000, 100, None, 40, 40, 107, 704),
            (2000, 100, None, 80, 40, 130, 858),
            (10000, 100, None, 20, 40, 296, 1966),
            (60000, 600, None, 40, 40, 3264, 21783),
            (180000, 2000, None, 40, 40, 12077, 80628),
            (3000, 2000, None, 0, 40, 721, 2425),
            (60000, 0, None, 0, 40, 1121, 7474),
            // Under a couple of seconds the overhead is most of the clock and there
            // is nothing left to allocate: play what depth one found.
            (1000, 0, None, 10, 40, 0, 0),
            (500, 0, None, 4, 40, 0, 0),
            (60000, 600, Some(40), 20, 40, 2148, 12233),
            (60000, 600, Some(1), 0, 40, 52694, 49460),
            (0, 0, None, 0, 40, 0, 0),
        ];
        for &(time, inc, mtg, ply, overhead, opt, max) in GOLDEN {
            let mut adjust = TIME_ADJUST_UNSET;
            let got = clock_bounds(time, inc, mtg, ply, overhead, &mut adjust);
            assert_eq!(
                got,
                (opt, max),
                "clock {time} inc {inc} mtg {mtg:?} ply {ply} overhead {overhead}"
            );
        }
    }

    /// Ten milliseconds of overhead are worth a tenth of a blitz budget, because
    /// the formula pays it on every move of the horizon rather than on this one.
    #[test]
    fn the_overhead_costs_far_more_than_its_size() {
        let mut a = TIME_ADJUST_UNSET;
        let (lean, _) = clock_bounds(2000, 100, None, 40, 30, &mut a);
        let mut b = TIME_ADJUST_UNSET;
        let (fat, _) = clock_bounds(2000, 100, None, 40, 40, &mut b);
        assert!(
            fat < lean && lean - fat > 10,
            "10 ms of overhead moved the budget from {lean} to {fat}, expected far more than 10"
        );
    }

    #[test]
    fn the_per_game_tilt_is_computed_once_and_kept() {
        let mut adjust = TIME_ADJUST_UNSET;
        let (first, _) = clock_bounds(60000, 600, None, 0, 30, &mut adjust);
        let frozen = adjust;
        assert!(
            frozen > 0.0,
            "tilt {frozen} should be positive on a real clock"
        );

        // A later move of the same game reuses it rather than recomputing from a
        // clock that has shrunk since.
        let (later, _) = clock_bounds(30000, 600, None, 40, 30, &mut adjust);
        assert_eq!(adjust, frozen, "the tilt must not move inside a game");
        assert!(later > 0 && later != first);

        // A clock too short to tilt leaves it negative, so the next move computes
        // it again instead of freezing a value that says nothing.
        let mut short = TIME_ADJUST_UNSET;
        clock_bounds(1000, 0, None, 0, 30, &mut short);
        assert!(
            short < 0.0,
            "tilt {short} on a spent clock should stay unset"
        );
    }

    #[test]
    fn movetime_is_a_deadline_and_never_a_licence() {
        let tm = TimeManager::new(
            &SearchLimits::movetime(500),
            0,
            &mut TIME_ADJUST_UNSET.clone(),
        );
        assert!(
            !tm.has_clock(),
            "a fixed move time is not a clock to manage"
        );
        assert_eq!(tm.optimum_ms(), 500);
        assert_eq!(tm.maximum, 500);
        // Nothing rescales it, so the iteration bound has to be the budget itself.
        assert_eq!(tm.soft_limit, 500);

        // With a clock as well, it lowers the ceiling and leaves the expected cost
        // alone: the rescaling still works off what the clock could afford.
        let mut adjust = TIME_ADJUST_UNSET;
        let mut limits = SearchLimits::clock(60000, 600, None);
        limits.movetime = Some(100);
        let tm = TimeManager::new(&limits, 40, &mut adjust);
        assert!(tm.has_clock());
        assert_eq!(tm.maximum, 100);
        // Whatever the default overhead happens to be, the expected cost is the
        // one the clock alone would have bought.
        let mut same = TIME_ADJUST_UNSET;
        let (clock_only, _) = clock_bounds(
            60000,
            600,
            None,
            40,
            crate::tune::MOVE_OVERHEAD() as i64,
            &mut same,
        );
        assert_eq!(tm.optimum_ms(), clock_only);
        assert!(clock_only > 3000, "sanity: {clock_only} ms at 60+0.6");
        // And the soft bound can never outlive the ceiling.
        let mut tm = tm;
        tm.set_soft_limit(u64::MAX);
        assert_eq!(tm.soft_limit, 100);
    }

    #[test]
    fn a_depth_or_node_limit_carries_no_clock() {
        let mut unused = TIME_ADJUST_UNSET;
        for limits in [
            SearchLimits::depth(12),
            SearchLimits::nodes(100_000),
            SearchLimits::soft_nodes(20_000, 2_000_000),
            SearchLimits::infinite(),
        ] {
            let tm = TimeManager::new(&limits, 20, &mut unused);
            assert!(!tm.has_clock(), "{limits:?} is not a clock");
            assert!(!tm.should_stop_soft(), "{limits:?} must not stop on time");
            assert!(!tm.should_stop_hard(), "{limits:?} must not stop on time");
        }
        assert_eq!(unused, TIME_ADJUST_UNSET, "no clock, no tilt to compute");
    }

    /// Spending every last millisecond of the ceiling, move after move, must not
    /// empty the clock. Increment time controls only: with no increment, the
    /// overhead alone drains any clock in the end, whatever the formula says.
    #[test]
    fn spending_the_whole_ceiling_never_flags() {
        let overhead = crate::tune::MOVE_OVERHEAD() as i64;
        for &(start, inc) in &[
            (2000u64, 100u64),
            (10_000, 100),
            (60_000, 600),
            (180_000, 2000),
        ] {
            let mut clock = start as i64;
            let mut adjust = TIME_ADJUST_UNSET;
            for ply in 0..120 {
                let (_, maximum) =
                    clock_bounds(clock as u64, inc, None, ply, overhead, &mut adjust);
                clock += inc as i64 - maximum as i64 - overhead;
                assert!(
                    clock > overhead + 10,
                    "{start}+{inc} ran down to {clock} ms at ply {ply}"
                );
            }
        }
    }
}
