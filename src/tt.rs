//! [Transposition table](https://www.chessprogramming.org/Transposition_Table) —
//! hash-indexed cache of search results.
//!
//! Each cluster is 32 bytes (cache-line friendly) containing 3 entries of 10 bytes
//! each plus 2 bytes padding. Indexed via Lemire fast-modulo reduction.
//! Replacement policy: same-key or empty first, then lowest quality (depth - 4*age).
//!
//! Thread-safe for Lazy SMP: `probe(&self)` and `store(&self)` use interior
//! mutability via `UnsafeCell`. Concurrent writes may race, but the 16-bit
//! verification key detects most corruptions. This is the standard lockless TT
//! approach used by all modern chess engines.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU8, Ordering};

use crate::types::{Move, SCORE_MATE, SCORE_MATE_IN_MAX, SCORE_NONE, SCORE_TB_WIN, SCORE_TB_WIN_IN_MAX};
#[cfg(test)]
use crate::types::is_mate_score;

/// TT entry bound type.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Bound {
    None = 0,
    Exact = 1,
    Lower = 2, // beta cutoff (fail-high)
    Upper = 3, // fail-low
}

/// Internal TT entry (10 bytes, packed).
#[derive(Clone, Copy)]
#[repr(C)]
struct TTEntry {
    key: u16,      // verification key (lower 16 bits of hash)
    mv: u16,       // best move (raw u16)
    score: i16,    // search score (mate-adjusted for storage)
    eval: i16,     // static evaluation
    depth: i8,     // search depth
    flags: u8,     // bits 0-1: Bound, bits 3-7: age
}

impl TTEntry {
    const EMPTY: TTEntry = TTEntry {
        key: 0, mv: 0, score: 0, eval: 0, depth: 0, flags: 0,
    };

    #[inline]
    fn bound(&self) -> Bound {
        match self.flags & 3 {
            1 => Bound::Exact,
            2 => Bound::Lower,
            3 => Bound::Upper,
            _ => Bound::None,
        }
    }

    #[inline]
    fn pv(&self) -> bool {
        self.flags & 4 != 0
    }

    #[inline]
    fn age(&self) -> u8 {
        self.flags >> 3
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.depth == 0 && self.key == 0
    }
}

/// A 32-byte aligned cluster of 3 TT entries (30 bytes + 2 padding).
#[derive(Clone, Copy)]
#[repr(C, align(32))]
struct Cluster {
    entries: [TTEntry; 3],
    _padding: [u8; 2],
}

impl Cluster {
    const EMPTY: Cluster = Cluster {
        entries: [TTEntry::EMPTY; 3],
        _padding: [0; 2],
    };
}

// Compile-time size/alignment checks
const _: () = assert!(std::mem::size_of::<TTEntry>() == 10, "TTEntry must be 10 bytes");
const _: () = assert!(std::mem::size_of::<Cluster>() == 32, "Cluster must be 32 bytes");
const _: () = assert!(std::mem::align_of::<Cluster>() == 32, "Cluster must be 32-byte aligned");

const AGE_CYCLE: u8 = 32;
const AGE_MASK: u8 = 31;

/// Default TT size in megabytes.
const DEFAULT_SIZE_MB: usize = 16;

/// Data returned from a successful TT probe.
pub struct TTHit {
    pub mv: Move,
    pub score: i32,
    pub eval: i32,
    pub depth: i32,
    pub bound: Bound,
    pub pv: bool,
}

/// The transposition table.
///
/// Thread-safe for Lazy SMP via interior mutability (`UnsafeCell`).
/// Multiple threads may read and write concurrently; the 16-bit key
/// detects most corruptions from data races.
pub struct TT {
    clusters: Vec<UnsafeCell<Cluster>>,
    age: AtomicU8,
}

// SAFETY: Concurrent reads/writes to TT clusters are intentionally racy.
// The 16-bit verification key catches most corruption. This is the standard
// lockless TT approach used by all modern chess engines.
unsafe impl Sync for TT {}

impl TT {
    /// Create a new TT with the given size in megabytes.
    pub fn new(mb: usize) -> TT {
        let bytes = mb * 1024 * 1024;
        let num_clusters = (bytes / std::mem::size_of::<Cluster>()).max(1);
        TT {
            clusters: (0..num_clusters).map(|_| UnsafeCell::new(Cluster::EMPTY)).collect(),
            age: AtomicU8::new(0),
        }
    }

    /// Resize the TT to the given size in megabytes. Clears all entries.
    /// Must only be called when no search is running.
    pub fn resize(&mut self, mb: usize) {
        let bytes = mb * 1024 * 1024;
        let num_clusters = (bytes / std::mem::size_of::<Cluster>()).max(1);
        self.clusters = (0..num_clusters).map(|_| UnsafeCell::new(Cluster::EMPTY)).collect();
        *self.age.get_mut() = 0;
    }

    /// Clear all entries and reset age.
    /// Must only be called when no search is running.
    pub fn clear(&mut self) {
        for cell in &mut self.clusters {
            *cell.get_mut() = Cluster::EMPTY;
        }
        *self.age.get_mut() = 0;
    }

    /// Increment the age counter (called at the start of each new search).
    pub fn new_search(&self) {
        let old = self.age.load(Ordering::Relaxed);
        self.age.store((old + 1) & AGE_MASK, Ordering::Relaxed);
    }

    /// Prefetch the TT cluster for the given key into L1 cache.
    #[inline]
    // The argument is only read where there is an instruction to read it for.
    #[cfg_attr(not(target_arch = "x86_64"), allow(unused_variables))]
    pub fn prefetch(&self, key: u64) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            let idx = self.index(key);
            let ptr = self.clusters[idx].get() as *const i8;
            _mm_prefetch::<{ _MM_HINT_T0 }>(ptr);
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let idx = self.index(key);
            let ptr = self.clusters[idx].get() as *const u8;
            std::arch::asm!("prfm pldl1keep, [{ptr}]", ptr = in(reg) ptr);
        }
    }

    /// Probe the TT for the given hash key.
    /// Returns `Some(TTHit)` if an entry with a matching key is found.
    /// `ply` is the current search ply (for mate score adjustment).
    /// `halfmove_clock` is used to downgrade unreachable mate scores (GHI fix).
    pub fn probe(&self, key: u64, ply: i32, halfmove_clock: u8) -> Option<TTHit> {
        debug_assert!((0..256).contains(&ply), "TT probe: ply {} OOB", ply);
        // SAFETY: Racy reads are acceptable; key16 verification catches corruption.
        let cluster = unsafe { &*self.clusters[self.index(key)].get() };
        let verify = verification_key(key);

        for entry in &cluster.entries {
            if entry.key == verify && !entry.is_empty() {
                return Some(TTHit {
                    mv: Move(entry.mv),
                    score: score_from_tt(entry.score as i32, ply, halfmove_clock),
                    eval: entry.eval as i32,
                    depth: entry.depth as i32,
                    bound: entry.bound(),
                    pv: entry.pv(),
                });
            }
        }
        None
    }

    /// Store a search result in the TT.
    /// Thread-safe: uses interior mutability for lockless concurrent writes.
    #[allow(clippy::too_many_arguments)]
    pub fn store(
        &self,
        key: u64,
        depth: i32,
        eval: i32,
        score: i32,
        bound: Bound,
        mv: Move,
        ply: i32,
        pv: bool,
    ) {
        debug_assert!((0..256).contains(&ply), "TT store: ply {} OOB", ply);
        debug_assert!(score.abs() <= SCORE_MATE_IN_MAX + 512,
            "TT store: score {} too large", score);
        debug_assert!((-5..=256).contains(&depth), "TT store: depth {} OOB", depth);
        debug_assert!(mv == Move::NONE || (mv.from_sq().0 < 64 && mv.to_sq().0 < 64),
            "TT store: move squares OOB from={} to={}", mv.from_sq().0, mv.to_sq().0);
        let idx = self.index(key);
        // SAFETY: Racy writes are acceptable; key16 verification catches corruption.
        let cluster = unsafe { &mut *self.clusters[idx].get() };
        let verify = verification_key(key);
        let tt_age = self.age.load(Ordering::Relaxed);

        // Find the best replacement slot
        let mut replace_idx = 0usize;
        let mut lowest_quality = i32::MAX;

        for (i, entry) in cluster.entries.iter().enumerate() {
            // Exact key match or empty slot: use immediately
            if entry.key == verify || entry.is_empty() {
                replace_idx = i;
                break;
            }

            // Otherwise pick the lowest-quality entry
            let relative_age = ((AGE_CYCLE + tt_age - entry.age()) & AGE_MASK) as i32;
            let quality = entry.depth as i32 - 4 * relative_age;
            if quality < lowest_quality {
                lowest_quality = quality;
                replace_idx = i;
            }
        }

        let entry = &mut cluster.entries[replace_idx];

        // Preserve existing move if we don't have a new one and key matches
        if mv != Move::NONE || entry.key != verify {
            entry.mv = mv.0;
        }

        // Only overwrite data if the new entry is valuable enough.
        // This prevents shallow qsearch entries from destroying deep alpha-beta entries.
        if bound == Bound::Exact
            || entry.key != verify
            || depth + 2 * (pv as i32) > entry.depth as i32 - 4
            || entry.age() != tt_age
        {
            entry.key = verify;
            entry.score = score_to_tt(score, ply) as i16;
            entry.eval = eval as i16;
            entry.depth = depth as i8;
            entry.flags = (bound as u8) | ((pv as u8) << 2) | (tt_age << 3);
        }
    }

    /// Returns hash table usage in permille (0-1000).
    pub fn hashfull(&self) -> usize {
        let sample = self.clusters.len().min(1000);
        let age = self.age.load(Ordering::Relaxed);
        let mut used = 0;
        for i in 0..sample {
            // SAFETY: Read-only sampling, racy reads acceptable.
            let cluster = unsafe { &*self.clusters[i].get() };
            for entry in &cluster.entries {
                if !entry.is_empty() && entry.age() == age {
                    used += 1;
                }
            }
        }
        used * 1000 / (sample * 3)
    }

    /// Lemire fast-modulo index from hash key.
    #[inline]
    fn index(&self, key: u64) -> usize {
        ((key as u128).wrapping_mul(self.clusters.len() as u128) >> 64) as usize
    }
}

impl Default for TT {
    fn default() -> Self {
        TT::new(DEFAULT_SIZE_MB)
    }
}

/// Extract the 16-bit verification key from a 64-bit hash.
#[inline]
fn verification_key(key: u64) -> u16 {
    key as u16
}

/// Adjust score for TT storage: convert mate/TB-from-root to mate/TB-from-position.
/// `is_win(v) ? v + ply : is_loss(v) ? v - ply : v`
/// where `is_win` catches both mate AND TB scores (>= SCORE_TB_WIN_IN_MAX).
#[inline]
fn score_to_tt(score: i32, ply: i32) -> i32 {
    debug_assert!(score != SCORE_NONE, "score_to_tt: SCORE_NONE");
    debug_assert!(score.abs() <= SCORE_MATE_IN_MAX + 512,
        "score_to_tt: score {} too large", score);
    if score >= SCORE_TB_WIN_IN_MAX {
        score + ply
    } else if score <= -SCORE_TB_WIN_IN_MAX {
        score - ply
    } else {
        score
    }
}

/// Adjust score from TT retrieval: convert mate/TB-from-position to mate/TB-from-root.
/// Also downgrades unreachable mate/TB scores when the halfmove clock is too high
/// ([Graph History Interaction](https://www.chessprogramming.org/Graph_History_Interaction) fix).
#[inline]
fn score_from_tt(score: i32, ply: i32, halfmove_clock: u8) -> i32 {
    let moves_left = 100 - halfmove_clock as i32;
    if score >= SCORE_TB_WIN_IN_MAX {
        // Downgrade potentially false mate score (GHI fix)
        if score >= SCORE_MATE_IN_MAX && SCORE_MATE - score > moves_left {
            return SCORE_TB_WIN_IN_MAX - 1;
        }
        // Downgrade potentially false TB score (GHI fix)
        if SCORE_TB_WIN - score > moves_left {
            return SCORE_TB_WIN_IN_MAX - 1;
        }
        score - ply
    } else if score <= -SCORE_TB_WIN_IN_MAX {
        if score <= -SCORE_MATE_IN_MAX && SCORE_MATE + score > moves_left {
            return -(SCORE_TB_WIN_IN_MAX - 1);
        }
        if SCORE_TB_WIN + score > moves_left {
            return -(SCORE_TB_WIN_IN_MAX - 1);
        }
        score + ply
    } else {
        score
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Square;

    #[test]
    fn test_tt_size() {
        assert_eq!(std::mem::size_of::<TTEntry>(), 10);
        assert_eq!(std::mem::size_of::<Cluster>(), 32);
        assert_eq!(std::mem::align_of::<Cluster>(), 32);
    }

    #[test]
    fn test_tt_store_probe() {
        let tt = TT::new(1); // 1 MB
        let key: u64 = 0xDEAD_BEEF_1234_5678;
        let mv = Move::new(Square::E2, Square::E4);

        tt.store(key, 5, 100, 150, Bound::Exact, mv, 0, false);

        let hit = tt.probe(key, 0, 0).expect("should find entry");
        assert_eq!(hit.mv, mv);
        assert_eq!(hit.score, 150);
        assert_eq!(hit.eval, 100);
        assert_eq!(hit.depth, 5);
        assert_eq!(hit.bound, Bound::Exact);
    }

    #[test]
    fn test_tt_miss() {
        let tt = TT::new(1);
        assert!(tt.probe(0x1234, 0, 0).is_none());
    }

    #[test]
    fn test_tt_mate_adjustment() {
        let tt = TT::new(1);
        let key: u64 = 0xAAAA_BBBB_CCCC_DDDD;
        let mv = Move::new(Square::E2, Square::E4);

        // Store a mate-in-3 score at ply 5
        let mate_score = SCORE_MATE - 3;
        tt.store(key, 10, 0, mate_score, Bound::Exact, mv, 5, false);

        // Probe at ply 5 should give back the original score
        let hit = tt.probe(key, 5, 0).unwrap();
        assert_eq!(hit.score, mate_score);

        // Probe at ply 2 should give a different (adjusted) score
        let hit2 = tt.probe(key, 2, 0).unwrap();
        assert_eq!(hit2.score, mate_score + 3); // closer to root = higher mate score
    }

    #[test]
    fn test_tt_clear() {
        let mut tt = TT::new(1);
        let key: u64 = 0x1111_2222_3333_4444;
        tt.store(key, 5, 0, 100, Bound::Lower, Move::NONE, 0, false);
        assert!(tt.probe(key, 0, 0).is_some());
        tt.clear();
        assert!(tt.probe(key, 0, 0).is_none());
    }

    #[test]
    fn test_tt_replacement() {
        let tt = TT::new(1);
        let key: u64 = 0x5555_6666_7777_8888;
        let mv1 = Move::new(Square::E2, Square::E4);
        let mv2 = Move::new(Square::D2, Square::D4);

        // Store then overwrite same key with deeper search
        tt.store(key, 3, 0, 50, Bound::Upper, mv1, 0, false);
        tt.store(key, 8, 0, 80, Bound::Exact, mv2, 0, true);

        let hit = tt.probe(key, 0, 0).unwrap();
        assert_eq!(hit.mv, mv2);
        assert_eq!(hit.depth, 8);
    }

    #[test]
    fn test_tt_mate_downgrade_high_halfmove() {
        // Mate in 8 moves = 16 half-moves. With hmc=88, 12 half-moves remain.
        // SCORE_MATE - 16 stored; distance 16 > 12 remaining → downgrade.
        let tt = TT::new(1);
        let key: u64 = 0xDEAD_0001;
        let mate_in_8 = SCORE_MATE - 16;
        tt.store(key, 20, 0, mate_in_8, Bound::Exact, Move::NONE, 0, false);
        let hit = tt.probe(key, 0, 88).unwrap();
        assert!(!is_mate_score(hit.score),
            "M8 with hmc=88 must be downgraded, got {}", hit.score);
        assert_eq!(hit.score, SCORE_TB_WIN_IN_MAX - 1);
    }

    #[test]
    fn test_tt_mate_no_downgrade_low_halfmove() {
        // Mate in 8 = 16 half-moves. With hmc=0, 100 half-moves remain.
        // 16 > 100 is FALSE → no downgrade.
        let tt = TT::new(1);
        let key: u64 = 0xDEAD_0002;
        let mate_in_8 = SCORE_MATE - 16;
        tt.store(key, 20, 0, mate_in_8, Bound::Exact, Move::NONE, 0, false);
        let hit = tt.probe(key, 0, 0).unwrap();
        assert!(is_mate_score(hit.score),
            "M8 with hmc=0 must NOT be downgraded, got {}", hit.score);
    }

    #[test]
    fn test_tt_mate_in_1_never_downgraded() {
        // Mate in 1 = 2 half-moves. With hmc=98, 2 half-moves remain.
        // 2 > 2 is FALSE → no downgrade.
        let tt = TT::new(1);
        let key: u64 = 0xDEAD_0003;
        let mate_in_1 = SCORE_MATE - 2;
        tt.store(key, 20, 0, mate_in_1, Bound::Exact, Move::NONE, 0, false);
        let hit = tt.probe(key, 0, 98).unwrap();
        assert!(is_mate_score(hit.score),
            "M1 with hmc=98 must NOT be downgraded, got {}", hit.score);
    }

    #[test]
    fn test_tt_mated_downgrade_high_halfmove() {
        // Negative side: mated-in-10 = 20 half-moves. With hmc=85, 15 remain.
        // 20 > 15 → downgrade.
        let tt = TT::new(1);
        let key: u64 = 0xDEAD_0004;
        let mated_in_10 = -(SCORE_MATE - 20);
        tt.store(key, 20, 0, mated_in_10, Bound::Exact, Move::NONE, 0, false);
        let hit = tt.probe(key, 0, 85).unwrap();
        assert!(!is_mate_score(hit.score),
            "Mated-in-10 with hmc=85 must be downgraded, got {}", hit.score);
        assert_eq!(hit.score, -(SCORE_TB_WIN_IN_MAX - 1));
    }
}
