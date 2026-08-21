//! Search statistics collection (feature `stats`).
//!
//! Reference: CPW — Search Statistics.
//!
//! Two layers:
//! - [`SearchStats`]: a permanent per-thread counter struct (owned by
//!   `ThreadData`, aggregated across threads after the search joins).
//!   Accessed through the [`st!`] macro, which compiles to nothing without
//!   the `stats` feature.
//! - The `dbg` module (added separately): throwaway atomic slots for ad-hoc
//!   measurements. Call sites are never committed.
//!
//! Invariant: collecting statistics must never change any search decision.
//! Instrumentation only reads locals and writes to `td.stats` — the bench
//! node count must be identical with and without the feature.

#[cfg(any(feature = "stats", feature = "tree"))]
use crate::position::Position;
#[cfg(any(feature = "stats", feature = "tree"))]
use crate::types::*;

/// Statistics block: run `$body` with `$s` bound to `&mut SearchStats`.
/// Compiles to nothing without the `stats` feature (tokens are discarded),
/// so arguments must never have side effects.
#[cfg(feature = "stats")]
macro_rules! st {
    ($td:expr, $s:ident, $($body:tt)*) => {{
        let $s: &mut $crate::stats::SearchStats = &mut $td.stats;
        $($body)*
    }};
}
#[cfg(not(feature = "stats"))]
macro_rules! st {
    ($td:expr, $s:ident, $($body:tt)*) => {{}};
}
pub(crate) use st;

/// Depth bucket for per-depth technique counters: remaining depth clamped to [0, 7].
#[cfg(feature = "stats")]
#[inline]
pub fn db(depth: i32) -> usize {
    depth.clamp(0, 7) as usize
}

/// Per-technique counters bucketed by remaining depth (bucket 7 = depth >= 7).
#[cfg(feature = "stats")]
pub struct TechDepth {
    pub tried: [u64; 8],
    pub cut: [u64; 8],
}

/// Zero/clear/add/JSON support for each field type of `SearchStats`.
#[cfg(feature = "stats")]
trait StatField {
    fn zeroed() -> Self;
    fn clear_field(&mut self);
    fn add_field(&mut self, other: &Self);
    fn to_json(&self) -> String;
}

#[cfg(feature = "stats")]
impl StatField for u64 {
    fn zeroed() -> Self {
        0
    }
    fn clear_field(&mut self) {
        *self = 0;
    }
    fn add_field(&mut self, other: &Self) {
        *self += *other;
    }
    fn to_json(&self) -> String {
        self.to_string()
    }
}

#[cfg(feature = "stats")]
impl<const N: usize> StatField for [u64; N] {
    fn zeroed() -> Self {
        [0; N]
    }
    fn clear_field(&mut self) {
        *self = [0; N];
    }
    fn add_field(&mut self, other: &Self) {
        for i in 0..N {
            self[i] += other[i];
        }
    }
    fn to_json(&self) -> String {
        let items: Vec<String> = self.iter().map(|v| v.to_string()).collect();
        format!("[{}]", items.join(", "))
    }
}

#[cfg(feature = "stats")]
impl StatField for TechDepth {
    fn zeroed() -> Self {
        TechDepth { tried: [0; 8], cut: [0; 8] }
    }
    fn clear_field(&mut self) {
        self.tried = [0; 8];
        self.cut = [0; 8];
    }
    fn add_field(&mut self, other: &Self) {
        self.tried.add_field(&other.tried);
        self.cut.add_field(&other.cut);
    }
    fn to_json(&self) -> String {
        format!(
            "{{\"tried\": {}, \"cut\": {}}}",
            self.tried.to_json(),
            self.cut.to_json()
        )
    }
}

/// Generate `SearchStats` with zeroed/clear/add/emit_json derived from the field list.
#[cfg(feature = "stats")]
macro_rules! define_stats {
    ($( $(#[doc = $doc:literal])* $field:ident: $ty:ty; )*) => {
        /// Per-thread search statistics. All counters are plain integers written
        /// exclusively by the owning thread; aggregation happens after join.
        pub struct SearchStats {
            $( $(#[doc = $doc])* pub $field: $ty, )*
        }

        impl SearchStats {
            pub fn zeroed() -> Self {
                SearchStats { $( $field: StatField::zeroed(), )* }
            }

            pub fn clear(&mut self) {
                $( self.$field.clear_field(); )*
            }

            pub fn add(&mut self, other: &Self) {
                $( self.$field.add_field(&other.$field); )*
            }

            /// Raw counters as JSON (hand-written, no serde).
            pub fn emit_json(&self) -> String {
                let mut parts: Vec<String> = Vec::new();
                $( parts.push(format!("  \"{}\": {}", stringify!($field), self.$field.to_json())); )*
                format!("{{\n{}\n}}", parts.join(",\n"))
            }
        }
    };
}

#[cfg(feature = "stats")]
define_stats! {
    /// Alpha-beta nodes by node type: [Root, Pv, NonPv].
    ab_nodes: [u64; 3];
    /// Quiescence nodes by node type: [Pv, NonPv].
    qs_nodes: [u64; 2];
    /// Alpha-beta nodes by remaining depth (clamped to 31).
    nodes_by_depth: [u64; 32];

    /// Static evaluations computed (NNUE or PeSTO).
    eval_calls: u64;
    /// Evaluations short-circuited by the lazy-eval gate (PeSTO used).
    lazy_evals: u64;
    /// Nodes where the TT-cached eval was reused instead of evaluating.
    tt_eval_reused: u64;

    /// TT probes in the main search.
    tt_probes: u64;
    /// TT probe hits in the main search.
    tt_hits: u64;
    /// Nodes where the TT provided a move for ordering.
    tt_move_available: u64;
    /// Immediate TT cutoffs in the main search.
    tt_cutoffs: u64;
    /// TT stores from main-search node exits.
    tt_stores: u64;

    /// Beta cutoffs in the main search move loop.
    cutoffs: u64;
    /// Beta cutoffs on the first move tried.
    cutoff_first: u64;
    /// Histogram of the cutoff move index (move_count - 1, clamped to 63).
    fail_high_index: [u64; 64];
    /// Cutoff move category: [tt, good_cap, bad_cap, killer, counter, quiet].
    cutoff_by_cat: [u64; 6];
    /// Quiet moves skipped by the skip_quiets flag (LMP/FP/history aftermath).
    quiets_skipped: u64;

    /// Qsearch stand-pat beta cutoffs.
    qs_standpat_cutoffs: u64;
    /// Qsearch TT cutoffs.
    qs_tt_cutoffs: u64;
    /// Qsearch beta cutoffs in the move loop.
    qs_cutoffs: u64;
    /// Qsearch beta cutoffs on the first move tried.
    qs_cutoff_first: u64;

    /// Aspiration windows that failed low (root re-search).
    asp_fail_low: u64;
    /// Aspiration windows that failed high (root re-search).
    asp_fail_high: u64;

    /// Moves entering the LMR branch (depth >= 2 && move_count > 1).
    lmr_searches: u64;
    /// LMR searches with an effective reduction (reduced_depth < new_depth).
    lmr_reduced: u64;
    /// LMR re-searches at full new_depth after the reduced search raised alpha.
    lmr_research: u64;
    /// Do-deeper events (new_depth incremented after a promising reduced search).
    do_deeper: u64;
    /// Do-shallower events (new_depth decremented).
    do_shallower: u64;
    /// Full-window re-searches at PV nodes (PVS).
    pvs_research: u64;

    /// Reverse futility pruning by depth: tried = gates passed, cut = returned.
    rfp: TechDepth;
    /// Razoring by depth: tried = gates passed, cut = dropped to qsearch.
    razor: TechDepth;
    /// Null move pruning by depth: tried = null search made, cut = returned.
    nmp: TechDepth;
    /// NMP verification searches (fail-high at depth >= NMP_VERIF_DEPTH).
    nmp_verif_runs: u64;
    /// ProbCut by depth: tried = move loop entered, cut = returned.
    probcut: TechDepth;
    /// IIR depth reductions applied.
    iir_applied: u64;
    /// Hindsight depth increments (parent over-reduced).
    hindsight_inc: u64;
    /// Hindsight depth decrements (comfortable position).
    hindsight_dec: u64;

    /// Moves pruned by SEE, by remaining depth.
    see_pruned: [u64; 8];
    /// Quiet moves pruned by history pruning, by remaining depth.
    hist_pruned: [u64; 8];
    /// Moves pruned by late move pruning, by remaining depth.
    lmp_pruned: [u64; 8];
    /// Quiet moves pruned by futility pruning, by remaining depth.
    fp_pruned: [u64; 8];

    /// Singular extension verification searches run.
    se_tried: u64;
    /// SE single extensions.
    se_ext1: u64;
    /// SE double extensions.
    se_ext2: u64;
    /// SE triple extensions.
    se_ext3: u64;
    /// SE negative extensions.
    se_negext: u64;
    /// SE multi-cut prunes (verification score >= beta).
    se_multicut: u64;
}

/// Cutoff move categories (indices into `cutoff_by_cat`).
#[cfg(feature = "stats")]
pub const CAT_NAMES: [&str; 6] = ["tt", "good_cap", "bad_cap", "killer", "counter", "quiet"];

/// Classify the move that produced a beta cutoff.
///
/// Pure function: reads the position and calls SEE, never touches TT,
/// histories, or NNUE state. Killers and countermoves are quiet by
/// construction, so noisy moves are classified before them.
#[cfg(any(feature = "stats", feature = "tree"))]
pub fn move_category(
    pos: &Position,
    m: Move,
    tt_move: Move,
    killers: [Move; 2],
    countermove: Move,
) -> usize {
    if m == tt_move {
        return 0;
    }
    let mt = m.move_type();
    let is_noisy =
        mt == MT_PROMOTION || mt == MT_EN_PASSANT || pos.board[m.to_sq().index()] != Piece::NONE;
    if is_noisy {
        return if crate::see::see(pos, m, 0) { 1 } else { 2 };
    }
    if m == killers[0] || m == killers[1] {
        return 3;
    }
    if m == countermove {
        return 4;
    }
    5
}

#[cfg(feature = "stats")]
impl SearchStats {
    /// Human-readable summary (percentages and per-depth tables).
    /// Kept under ~60 lines so it fits a terminal and an LLM context.
    pub fn emit_text(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        let pct = |num: u64, den: u64| -> f64 {
            if den == 0 { 0.0 } else { num as f64 * 100.0 / den as f64 }
        };

        let ab: u64 = self.ab_nodes.iter().sum();
        let qs: u64 = self.qs_nodes.iter().sum();
        let total = ab + qs;
        let _ = writeln!(out, "=== Search statistics ===");
        let _ = writeln!(
            out,
            "nodes        ab {} (root {} pv {} nonpv {})  qs {} ({:.1}% of total)",
            ab, self.ab_nodes[0], self.ab_nodes[1], self.ab_nodes[2], qs, pct(qs, total)
        );

        // Move ordering quality
        let mean_idx = {
            let (mut sum, mut n) = (0u64, 0u64);
            for (i, &c) in self.fail_high_index.iter().enumerate() {
                sum += (i as u64 + 1) * c;
                n += c;
            }
            if n == 0 { 0.0 } else { sum as f64 / n as f64 }
        };
        let _ = writeln!(
            out,
            "cutoffs      {} (first move {:.1}%, mean index {:.2})",
            self.cutoffs, pct(self.cutoff_first, self.cutoffs), mean_idx
        );
        let mut cats = String::new();
        for (i, name) in CAT_NAMES.iter().enumerate() {
            let _ = write!(cats, "{} {:.1}%  ", name, pct(self.cutoff_by_cat[i], self.cutoffs));
        }
        let _ = writeln!(out, "cutoff by    {}", cats.trim_end());
        let _ = writeln!(
            out,
            "qsearch      standpat cuts {}  tt cuts {}  move cuts {} (first {:.1}%)",
            self.qs_standpat_cutoffs, self.qs_tt_cutoffs, self.qs_cutoffs,
            pct(self.qs_cutoff_first, self.qs_cutoffs)
        );

        // TT
        let _ = writeln!(
            out,
            "tt           hit {:.1}%  cutoff {:.1}%  move avail {:.1}%  stores {}",
            pct(self.tt_hits, self.tt_probes),
            pct(self.tt_cutoffs, self.tt_probes),
            pct(self.tt_move_available, self.tt_probes),
            self.tt_stores
        );

        // Eval
        let _ = writeln!(
            out,
            "eval         calls {}  lazy {:.1}%  tt reuse {}",
            self.eval_calls, pct(self.lazy_evals, self.eval_calls), self.tt_eval_reused
        );

        // Aspiration + re-searches
        let _ = writeln!(
            out,
            "aspiration   fail low {}  fail high {}",
            self.asp_fail_low, self.asp_fail_high
        );
        let _ = writeln!(
            out,
            "lmr          searches {}  reduced {:.1}%  re-search {:.1}%  deeper {}  shallower {}  pvs re-search {}",
            self.lmr_searches,
            pct(self.lmr_reduced, self.lmr_searches),
            pct(self.lmr_research, self.lmr_reduced),
            self.do_deeper, self.do_shallower, self.pvs_research
        );

        // Whole-node pruning: tried -> cut rates per technique
        let tech = |name: &str, t: &TechDepth| -> String {
            let tried: u64 = t.tried.iter().sum();
            let cut: u64 = t.cut.iter().sum();
            let mut per_depth = String::new();
            for d in 1..8 {
                if t.tried[d] > 0 {
                    let _ = write!(per_depth, " d{}{}:{:.0}%", d,
                        if d == 7 { "+" } else { "" }, pct(t.cut[d], t.tried[d]));
                }
            }
            format!(
                "{:<8} tried {:>10}  cut {:>9} ({:>5.1}%) {}",
                name, tried, cut, pct(cut, tried), per_depth
            )
        };
        let _ = writeln!(out, "{}", tech("rfp", &self.rfp));
        let _ = writeln!(out, "{}", tech("razor", &self.razor));
        let _ = writeln!(out, "{}", tech("nmp", &self.nmp));
        let _ = writeln!(out, "{}", tech("probcut", &self.probcut));
        let _ = writeln!(
            out,
            "nmp verif    {}  iir {}  hindsight +{} -{}",
            self.nmp_verif_runs, self.iir_applied, self.hindsight_inc, self.hindsight_dec
        );

        // Per-move pruning by depth
        let prune = |name: &str, a: &[u64; 8]| -> String {
            let total: u64 = a.iter().sum();
            let mut per_depth = String::new();
            for (d, &n) in a.iter().enumerate().take(8) {
                if n > 0 {
                    let _ = write!(per_depth, " d{}{}:{}", d, if d == 7 { "+" } else { "" }, n);
                }
            }
            format!("{:<8} {:>10} {}", name, total, per_depth)
        };
        let _ = writeln!(out, "{}", prune("see_pr", &self.see_pruned));
        let _ = writeln!(out, "{}", prune("hist_pr", &self.hist_pruned));
        let _ = writeln!(out, "{}", prune("lmp_pr", &self.lmp_pruned));
        let _ = writeln!(out, "{}", prune("fp_pr", &self.fp_pruned));
        let _ = writeln!(out, "skip_q       {}", self.quiets_skipped);

        // Singular extensions
        let _ = writeln!(
            out,
            "se           tried {}  ext1 {}  ext2 {}  ext3 {}  negext {}  multicut {}",
            self.se_tried, self.se_ext1, self.se_ext2, self.se_ext3,
            self.se_negext, self.se_multicut
        );

        out
    }
}

/// Throwaway debug counters, always compiled (multithread-safe atomics).
///
/// Workflow: insert a call locally (`crate::stats::dbg::hit(cond, 0)` or
/// `crate::stats::dbg::mean(value, 1)`), run `bench`, read the report on
/// stderr, then DELETE the call site. Call sites are never committed —
/// only this infrastructure is permanent. Both functions return their
/// argument so they can wrap a sub-expression in place.
pub mod dbg {
    use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

    pub const SLOTS: usize = 32;

    /// Hit-rate slots: [0] = calls, [1] = hits.
    static HIT: [[AtomicU64; 2]; SLOTS] =
        [const { [const { AtomicU64::new(0) }; 2] }; SLOTS];

    /// Mean slots: [0] = count, [1] = sum, [2] = min, [3] = max.
    static MEAN: [[AtomicI64; 4]; SLOTS] =
        [const { [const { AtomicI64::new(0) }; 4] }; SLOTS];

    /// Record a boolean condition into a hit-rate slot; returns the condition.
    #[allow(dead_code)]
    #[inline]
    pub fn hit(cond: bool, slot: usize) -> bool {
        debug_assert!(slot < SLOTS);
        HIT[slot][0].fetch_add(1, Ordering::Relaxed);
        if cond {
            HIT[slot][1].fetch_add(1, Ordering::Relaxed);
        }
        cond
    }

    /// Record a value into a mean/min/max slot; returns the value.
    #[allow(dead_code)]
    #[inline]
    pub fn mean<T: Into<i64> + Copy>(value: T, slot: usize) -> T {
        debug_assert!(slot < SLOTS);
        let v: i64 = value.into();
        let m = &MEAN[slot];
        if m[0].fetch_add(1, Ordering::Relaxed) == 0 {
            m[2].store(v, Ordering::Relaxed);
            m[3].store(v, Ordering::Relaxed);
        } else {
            m[2].fetch_min(v, Ordering::Relaxed);
            m[3].fetch_max(v, Ordering::Relaxed);
        }
        m[1].fetch_add(v, Ordering::Relaxed);
        value
    }

    /// Print all used slots on stderr. Silent when no slot was recorded.
    pub fn print() {
        for (i, h) in HIT.iter().enumerate() {
            let calls = h[0].load(Ordering::Relaxed);
            if calls > 0 {
                let hits = h[1].load(Ordering::Relaxed);
                eprintln!(
                    "dbg hit  #{i}: {:.2}% ({hits}/{calls})",
                    hits as f64 * 100.0 / calls as f64
                );
            }
        }
        for (i, m) in MEAN.iter().enumerate() {
            let count = m[0].load(Ordering::Relaxed);
            if count > 0 {
                let sum = m[1].load(Ordering::Relaxed);
                eprintln!(
                    "dbg mean #{i}: {:.2} (n {count}, min {}, max {})",
                    sum as f64 / count as f64,
                    m[2].load(Ordering::Relaxed),
                    m[3].load(Ordering::Relaxed)
                );
            }
        }
    }

    /// Reset all slots (e.g. between positions).
    #[allow(dead_code)]
    pub fn clear() {
        for h in &HIT {
            h[0].store(0, Ordering::Relaxed);
            h[1].store(0, Ordering::Relaxed);
        }
        for m in &MEAN {
            for a in m {
                a.store(0, Ordering::Relaxed);
            }
        }
    }
}

#[cfg(test)]
mod dbg_tests {
    use super::dbg;

    #[test]
    fn test_dbg_slots() {
        // Use high slot numbers to avoid clashing with ad-hoc call sites
        assert!(dbg::hit(true, 30));
        assert!(!dbg::hit(false, 30));
        assert_eq!(dbg::mean(10i32, 31), 10);
        assert_eq!(dbg::mean(-4i32, 31), -4);
        // Print must not panic; visual check happens on stderr
        dbg::print();
    }
}

#[cfg(all(test, feature = "stats"))]
mod tests {
    use super::*;

    #[test]
    fn test_zeroed_clear_add() {
        let mut a = SearchStats::zeroed();
        let mut b = SearchStats::zeroed();
        a.cutoffs = 10;
        a.fail_high_index[0] = 7;
        a.rfp.tried[3] = 5;
        a.rfp.cut[3] = 2;
        b.cutoffs = 1;
        b.add(&a);
        assert_eq!(b.cutoffs, 11);
        assert_eq!(b.fail_high_index[0], 7);
        assert_eq!(b.rfp.tried[3], 5);
        assert_eq!(b.rfp.cut[3], 2);
        b.clear();
        assert_eq!(b.cutoffs, 0);
        assert_eq!(b.rfp.tried[3], 0);
    }

    #[test]
    fn test_json_parses_shape() {
        let mut s = SearchStats::zeroed();
        s.cutoffs = 42;
        let json = s.emit_json();
        assert!(json.starts_with('{') && json.ends_with('}'));
        assert!(json.contains("\"cutoffs\": 42"));
        assert!(json.contains("\"rfp\": {\"tried\": [0, 0, 0, 0, 0, 0, 0, 0]"));
    }
}
