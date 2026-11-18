//! Move-ordering telemetry (`--features stats`): first-move cutoff rate, rank of the
//! cutoff move and the nodes spent before it, per depth bucket, with and without a TT
//! move. A `MOSTAT` line goes to stderr after every `bestmove`; `tools/mostat/` drives
//! the engine over a fixed position set and compares two builds.
//!
//! Without the feature every probe is an empty inline function and the search compiles
//! to the same code.

#[cfg(feature = "stats")]
mod counters {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    pub const NB: usize = 5;
    pub const NH: usize = 7;
    const Z: AtomicU64 = AtomicU64::new(0);
    const ZB: [AtomicU64; NB] = [Z; NB];
    const ZH: [AtomicU64; NH] = [Z; NH];

    pub static LOOPS: [AtomicU64; NB] = ZB;
    pub static CUT: [AtomicU64; NB] = ZB;
    pub static FIRST: [AtomicU64; NB] = ZB;
    pub static IDXSUM: [AtomicU64; NB] = ZB;
    pub static CAT: [[AtomicU64; NB]; 3] = [ZB; 3];
    pub static WASTED: [AtomicU64; NB] = ZB;
    pub static TOTAL: [AtomicU64; NB] = ZB;
    pub static CUT_NOTT: [AtomicU64; NB] = ZB;
    pub static FIRST_NOTT: [AtomicU64; NB] = ZB;
    pub static IDXSUM_NOTT: [AtomicU64; NB] = ZB;
    pub static WASTED_NOTT: [AtomicU64; NB] = ZB;
    pub static TOTAL_NOTT: [AtomicU64; NB] = ZB;
    pub static PV_EXACT: [AtomicU64; NB] = ZB;
    pub static PV_FIRST: [AtomicU64; NB] = ZB;
    pub static PV_IDXSUM: [AtomicU64; NB] = ZB;
    pub static HIST: [[AtomicU64; NH]; NB] = [ZH; NB];

    /// Depth buckets: 1-2, 3-4, 5-7, 8-11, 12+.
    pub fn bucket(d: i32) -> usize {
        match d {
            ..=2 => 0,
            3..=4 => 1,
            5..=7 => 2,
            8..=11 => 3,
            _ => 4,
        }
    }

    /// Cutoff-rank histogram bins: 1, 2, 3, 4-5, 6-10, 11-20, 21+.
    pub fn hbin(i: u32) -> usize {
        match i {
            0 | 1 => 0,
            2 => 1,
            3 => 2,
            4..=5 => 3,
            6..=10 => 4,
            11..=20 => 5,
            _ => 6,
        }
    }

    pub fn add(a: &AtomicU64, v: u64) {
        a.fetch_add(v, Relaxed);
    }
}

/// Cutoff move category.
pub const CAT_TT: usize = 0;
pub const CAT_NOISY: usize = 1;
pub const CAT_QUIET: usize = 2;

/// Per-node probe of the main-search move loop. A measured node is a non-root node
/// outside the singular-extension search whose loop searched at least one move; its
/// depth is the one entering the loop.
#[derive(Clone, Copy)]
pub struct NodeProbe {
    #[cfg(feature = "stats")]
    depth: i32,
    #[cfg(feature = "stats")]
    alpha: i32,
    #[cfg(feature = "stats")]
    has_tt: bool,
    #[cfg(feature = "stats")]
    searched: u32,
    #[cfg(feature = "stats")]
    best_idx: u32,
    #[cfg(feature = "stats")]
    loop_start: u64,
    #[cfg(feature = "stats")]
    move_start: u64,
}

impl NodeProbe {
    #[inline(always)]
    #[allow(unused_variables)]
    pub fn new(depth: i32, alpha: i32, has_tt: bool) -> Self {
        NodeProbe {
            #[cfg(feature = "stats")]
            depth,
            #[cfg(feature = "stats")]
            alpha,
            #[cfg(feature = "stats")]
            has_tt,
            #[cfg(feature = "stats")]
            searched: 0,
            #[cfg(feature = "stats")]
            best_idx: 0,
            #[cfg(feature = "stats")]
            loop_start: 0,
            #[cfg(feature = "stats")]
            move_start: 0,
        }
    }

    /// A move is about to be searched (after every pruning decision).
    #[inline(always)]
    #[allow(unused_variables)]
    pub fn on_move(&mut self, nodes: u64) {
        #[cfg(feature = "stats")]
        {
            self.move_start = nodes;
            self.searched += 1;
            if self.searched == 1 {
                self.loop_start = nodes;
            }
        }
    }

    /// The move just searched raised alpha.
    #[inline(always)]
    pub fn on_best(&mut self) {
        #[cfg(feature = "stats")]
        {
            self.best_idx = self.searched;
        }
    }

    /// The move just searched failed high.
    #[inline(always)]
    #[allow(unused_variables)]
    pub fn on_cut(&self, cat: usize, nodes: u64) {
        #[cfg(feature = "stats")]
        {
            use counters::*;
            debug_assert!(cat < 3);
            debug_assert!(self.searched >= 1 && nodes >= self.move_start);
            let b = bucket(self.depth);
            let idx = self.searched;
            let wasted = self.move_start - self.loop_start;
            let total = nodes - self.loop_start;
            add(&CUT[b], 1);
            add(&FIRST[b], (idx == 1) as u64);
            add(&IDXSUM[b], idx as u64);
            add(&CAT[cat][b], 1);
            add(&WASTED[b], wasted);
            add(&TOTAL[b], total);
            add(&HIST[b][hbin(idx)], 1);
            if !self.has_tt {
                add(&CUT_NOTT[b], 1);
                add(&FIRST_NOTT[b], (idx == 1) as u64);
                add(&IDXSUM_NOTT[b], idx as u64);
                add(&WASTED_NOTT[b], wasted);
                add(&TOTAL_NOTT[b], total);
            }
        }
    }

    /// The move loop is over. `pv_node` tells whether the node was a PV node; an exact
    /// PV node (alpha < best < beta) records the rank of its best move.
    #[inline(always)]
    #[allow(unused_variables)]
    pub fn on_exit(&self, pv_node: bool, has_best: bool, best_score: i32, beta: i32) {
        #[cfg(feature = "stats")]
        {
            use counters::*;
            if self.searched == 0 {
                return;
            }
            let b = bucket(self.depth);
            add(&LOOPS[b], 1);
            if pv_node && has_best && best_score > self.alpha && best_score < beta {
                add(&PV_EXACT[b], 1);
                add(&PV_FIRST[b], (self.best_idx == 1) as u64);
                add(&PV_IDXSUM[b], self.best_idx as u64);
            }
        }
    }
}

/// Placebo for the bench's noise floor: `GAIA_MO_PLACEBO=1` shifts every LMR reduction
/// by 1/1024 ply, which perturbs the tree without touching the order. Always 0 without
/// the feature.
#[inline(always)]
pub fn placebo() -> i32 {
    #[cfg(feature = "stats")]
    {
        static V: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
        *V.get_or_init(|| {
            std::env::var("GAIA_MO_PLACEBO").ok().and_then(|s| s.parse().ok()).unwrap_or(0)
        })
    }
    #[cfg(not(feature = "stats"))]
    {
        0
    }
}

/// Prints the cumulative counters as one `MOSTAT` line on stderr (feature only).
pub fn print() {
    #[cfg(feature = "stats")]
    {
        use counters::*;
        use std::fmt::Write;
        use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
        let mut s = String::from("MOSTAT");
        let mut arr = |n: &str, a: &[AtomicU64]| {
            let v: Vec<String> = a.iter().map(|x| x.load(Relaxed).to_string()).collect();
            let _ = write!(s, " {}={}", n, v.join(","));
        };
        arr("loops", &LOOPS);
        arr("cut", &CUT);
        arr("first", &FIRST);
        arr("idxsum", &IDXSUM);
        arr("cat_tt", &CAT[CAT_TT]);
        arr("cat_noisy", &CAT[CAT_NOISY]);
        arr("cat_quiet", &CAT[CAT_QUIET]);
        arr("wasted", &WASTED);
        arr("total", &TOTAL);
        arr("cut_nott", &CUT_NOTT);
        arr("first_nott", &FIRST_NOTT);
        arr("idxsum_nott", &IDXSUM_NOTT);
        arr("wasted_nott", &WASTED_NOTT);
        arr("total_nott", &TOTAL_NOTT);
        arr("pv_exact", &PV_EXACT);
        arr("pv_first", &PV_FIRST);
        arr("pv_idxsum", &PV_IDXSUM);
        for (b, h) in HIST.iter().enumerate() {
            arr(&format!("hist{b}"), h);
        }
        eprintln!("{s}");
    }
}

#[cfg(all(test, feature = "stats"))]
mod tests {
    use super::counters::{bucket, hbin};

    #[test]
    fn buckets() {
        assert_eq!(bucket(1), 0);
        assert_eq!(bucket(4), 1);
        assert_eq!(bucket(7), 2);
        assert_eq!(bucket(11), 3);
        assert_eq!(bucket(30), 4);
        assert_eq!(hbin(1), 0);
        assert_eq!(hbin(5), 3);
        assert_eq!(hbin(21), 6);
    }
}
