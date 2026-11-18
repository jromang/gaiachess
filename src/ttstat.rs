//! Transposition-table telemetry (`--features stats`): who asks the table, what it
//! answers, what it keeps and what it throws away.
//!
//! One `TTSTAT` line of cumulative counters goes to stderr after every `go` and at the
//! end of the bench, `name=v0,v1,...` for each counter, indexed as its doc says. With
//! `GAIA_TTSTAT_SCAN=1` a `TTSCAN` line follows it, describing the whole table as the
//! search left it.
//!
//! Collisions are measured against a shadow of the table, kept in a statistics build
//! only: the full 64-bit key of every slot, written exactly when the slot's 16-bit key
//! is, and a word telling who wrote the entry, whether it has been read since that
//! write or ever since its position took the slot, and whether its move is another
//! position's.
//!
//! Nothing here feeds back into the search: the hooks read what the search computed and
//! write counters it never reads, so the bench node count is the same with and without
//! the feature. Without the feature only an empty `print` is left.

#[cfg(feature = "stats")]
pub use imp::*;

/// Prints the `TTSTAT` line (and the `TTSCAN` line when asked for): nothing without the
/// feature.
#[cfg(not(feature = "stats"))]
pub fn print(_tt: &crate::tt::TT) {}

#[cfg(feature = "stats")]
mod imp {
    use std::cell::Cell;
    use std::fmt::Write;
    use std::sync::atomic::{AtomicU16, AtomicU64, Ordering::Relaxed};

    use crate::tt::{Bound, TT, TTHit};

    /// Probe consumers: the index of the per-consumer counters.
    pub const MAIN: usize = 0;
    pub const QS: usize = 1;
    pub const LOOKAHEAD: usize = 2;
    pub const OTHER: usize = 3;

    /// Who wrote an entry: the index of the per-origin counters.
    pub const O_UNKNOWN: u8 = 0;
    pub const O_MAIN: u8 = 1;
    pub const O_EVAL: u8 = 2;
    pub const O_PROBCUT: u8 = 3;
    pub const O_TB: u8 = 4;
    pub const O_STANDPAT: u8 = 5;
    pub const O_QS: u8 = 6;

    /// The shadow's word per slot: the writer's origin, whether the writer was a PV node,
    /// whether the entry has been read by its own position since it was last written
    /// (cleared by every write, a rewrite of the same position included), whether it has
    /// been read by its own position at least once since that position took the slot
    /// (sticky: cleared only when another position takes the slot), and whether the move
    /// it holds was written by another position.
    const META_ORIGIN: u16 = 0x000F;
    const META_PV_NODE: u16 = 0x0010;
    const META_READ_SEARCH: u16 = 0x0020;
    const META_READ_LOOKAHEAD: u16 = 0x0040;
    const META_EVER_SEARCH: u16 = 0x0080;
    const META_EVER_LOOKAHEAD: u16 = 0x0100;
    const META_FOREIGN_MOVE: u16 = 0x0200;
    /// What a write by the same position keeps.
    const META_STICKY: u16 = META_EVER_SEARCH | META_EVER_LOOKAHEAD;

    macro_rules! counters {
        ($( $(#[doc = $doc:literal])* $name:ident : $n:literal ; )*) => {
            /// The cumulative counters, one array each.
            pub struct Counters {
                $( $(#[doc = $doc])* pub $name: [AtomicU64; $n], )*
            }

            pub static C: Counters = Counters {
                $( $name: [const { AtomicU64::new(0) }; $n], )*
            };

            impl Counters {
                fn write_line(&self, out: &mut String) {
                    $( push(out, stringify!($name), &self.$name); )*
                }
            }
        };
    }

    counters! {
        // --- A. Probes, by consumer [main, qs, lookahead, other] ---
        /// Probes.
        probes: 4;
        /// Probes that found nothing.
        misses: 4;
        /// Hits on an evaluation-only entry (`Bound::None`).
        hit_eval: 4;
        /// Hits on a depth-0 entry (quiescence).
        hit_d0: 4;
        /// Hits on an entry of depth 1 or more.
        hit_deep: 4;
        /// Main-search probes and hits by node depth [1-2, 3-4, 5-7, 8-11, 12+].
        main_probe_d: 5;
        main_hit_d: 5;
        /// Main-search probes of a singular verification (the node's own position again).
        main_probe_se: 1;

        // --- B. Collisions ---
        /// False hits (same 16-bit key, another position), by consumer.
        false_hits: 4;
        /// False hits by kind of entry [eval only, depth 0, depth >= 1].
        false_kind: 3;
        /// False hits that cut the node [main, qs].
        false_cut: 2;
        /// False hits whose move passes `is_pseudo_legal` [main, qs], and `is_legal` too.
        false_move: 2;
        false_move_legal: 2;
        /// False hits whose evaluation the node takes as its own [main, qs].
        false_eval: 2;
        /// Writes landing on a slot whose 16-bit key is another position's, by outcome
        /// [rewritten, eval-only write ignored, refused but move written, refused].
        st_alias: 4;

        // --- C. Main-search cutoff (non-PV, non-root, no excluded move, scored entry) ---
        /// [candidates, tt_depth < depth, tt_depth >= depth (usable), usable and right
        /// side but halfmove clock >= 96].
        cut_cand: 4;
        /// Usable, bound on the wrong side of the window [all, depth > 5].
        cut_wrong: 2;
        /// Cutoffs [counted by the probe hook, counted where the search returns].
        cut: 2;
        /// Cutoffs by bound [exact, lower, upper] and by node depth [5 buckets].
        cut_bound: 3;
        cut_depth: 5;
        /// Cutoffs by origin of the entry.
        cut_origin: 7;

        // --- D. ttPv ---
        /// Quiescence PV-node stores [all, data written, written over an entry of the
        /// position without the flag, written as a new entry].
        pv_qs: 4;
        /// Main-search hits carrying the pv flag [all, last written by a quiescence PV
        /// node, at a non-PV node, at a non-PV node and written by a quiescence PV node].
        pv_read: 4;

        // --- E. Quiescence and stand-pat entries ---
        /// Quiescence cutoffs [counted by the probe hook, counted where it returns].
        qs_cut: 2;
        /// Quiescence cutoffs by origin of the entry.
        qs_cut_origin: 7;
        /// Evaluations reused from the entry, by origin [main search, quiescence].
        eval_main_origin: 7;
        eval_qs_origin: 7;
        /// Main search: refined eval taken from the entry's score, by origin.
        refine_origin: 7;
        /// Stand-pat entries read by their own position [main hits, main eval refined
        /// by the score, quiescence hits, quiescence stand-pat blended with the score].
        standpat_use: 4;
        /// Quiescence cutoffs on a stand-pat entry [all, entry without an evaluation:
        /// written by a lazily evaluated quiescence].
        standpat_cut: 2;

        // --- F. Writes ---
        /// Writes by origin [unknown, main, eval only, probcut, tablebase, stand-pat, qs].
        st_origin: 7;
        /// Slot taken [same key, empty, victim].
        st_choice: 3;
        /// Same key, same position, by outcome [rewritten, eval-only write ignored,
        /// refused but move written, refused].
        st_same: 4;
        /// Same-position rewrites by the first clause that allowed them [exact, key,
        /// depth, age].
        st_clause: 4;
        /// Refused writes that aged a deep decisive bound (secondary aging).
        st_aging: 1;
        /// Refused writes (same position) by origin.
        st_refused_origin: 7;
        /// Victims by kind, of the current search and of an older one.
        st_vic_cur: 3;
        st_vic_old: 3;
        /// Victims by depth [eval only, 0, 1-2, 3-4, 5-7, 8-11, 12+], current and older.
        st_vic_dcur: 7;
        st_vic_dold: 7;
        /// Victims by reads since their write [by the search, by the look-ahead only,
        /// never], current and older.
        st_vic_rcur: 3;
        st_vic_rold: 3;
        /// Victims by reads since their position took the slot, rewrites of that
        /// position included [by the search, by the look-ahead only, never], current and
        /// older.
        st_vic_ecur: 3;
        st_vic_eold: 3;

        // --- G. TT move ---
        /// Hits with a move [main, qs], those refused by `is_pseudo_legal`, and the
        /// refused ones read from another position's entry.
        mv_avail: 2;
        mv_bad: 2;
        mv_bad_false: 2;
        /// True hits whose move another position wrote [main, qs], and those of them
        /// refused by `is_pseudo_legal`.
        mv_foreign: 2;
        mv_bad_foreign: 2;
        /// Moves left to another position by a write on its 16-bit key [a rewrite
        /// without a move kept the other position's, a refused write put its move in the
        /// other position's entry].
        st_foreign: 2;
        /// Main nodes ending with a lower bound or exact, with a playable TT move
        /// [lower, lower and TT move best, exact, exact and TT move best].
        mv_best: 4;
    }

    fn push(out: &mut String, name: &str, values: &[AtomicU64]) {
        let v: Vec<String> = values.iter().map(|a| a.load(Relaxed).to_string()).collect();
        let _ = write!(out, " {}={}", name, v.join(","));
    }

    #[inline]
    pub fn inc(a: &AtomicU64) {
        a.fetch_add(1, Relaxed);
    }

    /// Kind of an entry: evaluation only, depth 0 (quiescence), depth >= 1.
    #[inline]
    pub fn kind(bound: Bound, depth: i32) -> usize {
        if bound == Bound::None {
            0
        } else if depth <= 0 {
            1
        } else {
            2
        }
    }

    /// Node depth buckets: 1-2, 3-4, 5-7, 8-11, 12+.
    #[inline]
    pub fn bucket(depth: i32) -> usize {
        match depth {
            ..=2 => 0,
            3..=4 => 1,
            5..=7 => 2,
            8..=11 => 3,
            _ => 4,
        }
    }

    /// Entry depth bins: evaluation only, 0, then the node buckets.
    #[inline]
    pub fn depth_bin(bound: Bound, depth: i32) -> usize {
        if bound == Bound::None || depth < 0 {
            0
        } else if depth == 0 {
            1
        } else {
            2 + bucket(depth)
        }
    }

    /// The full key and the meta word of every slot of the table.
    pub struct Shadow {
        keys: Box<[AtomicU64]>,
        meta: Box<[AtomicU16]>,
    }

    impl Shadow {
        pub fn new(slots: usize) -> Shadow {
            Shadow {
                keys: (0..slots).map(|_| AtomicU64::new(0)).collect(),
                meta: (0..slots).map(|_| AtomicU16::new(0)).collect(),
            }
        }

        /// Empties the slots `from..to`, as the table's clear does.
        pub fn clear(&self, from: usize, to: usize) {
            debug_assert!(from <= to && to <= self.keys.len());
            for i in from..to {
                self.keys[i].store(0, Relaxed);
                self.meta[i].store(0, Relaxed);
            }
        }

        #[inline]
        pub fn key(&self, slot: usize) -> u64 {
            self.keys[slot].load(Relaxed)
        }

        #[inline]
        pub fn meta(&self, slot: usize) -> u16 {
            self.meta[slot].load(Relaxed)
        }

        /// The data of `slot` written by `key`: who wrote it, not read since, and whether
        /// the move it now holds is another position's. A rewrite by the position already
        /// there keeps the sticky reads; another position starts afresh.
        #[inline]
        pub fn write(&self, slot: usize, key: u64, meta: u16, foreign_move: bool) {
            debug_assert!(meta & !(META_ORIGIN | META_PV_NODE) == 0);
            let old_key = self.keys[slot].load(Relaxed);
            let sticky = if old_key == key {
                self.meta[slot].load(Relaxed) & META_STICKY
            } else {
                0
            };
            let foreign = if foreign_move { META_FOREIGN_MOVE } else { 0 };
            self.keys[slot].store(key, Relaxed);
            self.meta[slot].store(meta | sticky | foreign, Relaxed);
        }

        /// The move of `slot` was just written: by another position than the one the
        /// slot holds, or by that position itself.
        #[inline]
        pub fn set_foreign_move(&self, slot: usize, foreign: bool) {
            if foreign {
                self.meta[slot].fetch_or(META_FOREIGN_MOVE, Relaxed);
            } else {
                self.meta[slot].fetch_and(!META_FOREIGN_MOVE, Relaxed);
            }
        }

        /// A read by the slot's own position: since the write, and ever.
        #[inline]
        fn mark_read(&self, slot: usize, lookahead: bool) {
            let bits = if lookahead {
                META_READ_LOOKAHEAD | META_EVER_LOOKAHEAD
            } else {
                META_READ_SEARCH | META_EVER_SEARCH
            };
            self.meta[slot].fetch_or(bits, Relaxed);
        }
    }

    /// The writer's origin for the shadow.
    #[inline]
    pub fn meta_of(origin: u8, pv_node: bool) -> u16 {
        debug_assert!(origin <= O_QS);
        origin as u16 | if pv_node { META_PV_NODE } else { 0 }
    }

    #[inline]
    pub fn meta_origin(meta: u16) -> u8 {
        (meta & META_ORIGIN) as u8
    }

    #[inline]
    pub fn meta_pv_node(meta: u16) -> bool {
        meta & META_PV_NODE != 0
    }

    /// Whether the move of the entry was written by another position.
    #[inline]
    pub fn meta_foreign_move(meta: u16) -> bool {
        meta & META_FOREIGN_MOVE != 0
    }

    /// Reads since the last write: 0 by the search, 1 by the look-ahead only, 2 never.
    #[inline]
    pub fn meta_read(meta: u16) -> usize {
        if meta & META_READ_SEARCH != 0 {
            0
        } else if meta & META_READ_LOOKAHEAD != 0 {
            1
        } else {
            2
        }
    }

    /// Reads since the position took the slot: 0 by the search, 1 by the look-ahead
    /// only, 2 never.
    #[inline]
    pub fn meta_ever(meta: u16) -> usize {
        if meta & META_EVER_SEARCH != 0 {
            0
        } else if meta & META_EVER_LOOKAHEAD != 0 {
            1
        } else {
            2
        }
    }

    thread_local! {
        /// The origin of the next store on this thread, set by its caller just before.
        static ORIGIN: Cell<u16> = const { Cell::new(O_UNKNOWN as u16) };
    }

    /// Names the origin of the store that follows on this thread.
    #[inline]
    pub fn set_origin(origin: u8, pv_node: bool) {
        ORIGIN.with(|o| o.set(meta_of(origin, pv_node)));
    }

    /// The origin of this store, consumed.
    #[inline]
    pub fn take_origin() -> u16 {
        ORIGIN.with(|o| o.replace(O_UNKNOWN as u16))
    }

    /// What a probe found, for the rest of the node to refer to.
    #[derive(Clone, Copy, Default)]
    pub struct HitInfo {
        /// The probe returned an entry...
        pub hit: bool,
        /// ...whose full key is another position's.
        pub false_hit: bool,
        /// Who wrote it (`O_*`), and whether the writer was a PV node.
        pub origin: u8,
        pub pv_writer: bool,
        /// A true hit whose move another position wrote.
        pub foreign_move: bool,
        /// Main search and quiescence: the entry cuts the node.
        pub cut: bool,
    }

    /// Counts a probe for `consumer` and marks a true hit read.
    pub fn probe(tt: &TT, consumer: usize, key: u64, hit: Option<&TTHit>) -> HitInfo {
        debug_assert!(consumer <= OTHER);
        inc(&C.probes[consumer]);
        let Some(h) = hit else {
            inc(&C.misses[consumer]);
            return HitInfo::default();
        };
        let k = kind(h.bound, h.depth);
        match k {
            0 => inc(&C.hit_eval[consumer]),
            1 => inc(&C.hit_d0[consumer]),
            _ => inc(&C.hit_deep[consumer]),
        }
        let shadow = tt.stat_shadow();
        let meta = shadow.meta(h.slot);
        let false_hit = shadow.key(h.slot) != key;
        if false_hit {
            inc(&C.false_hits[consumer]);
            inc(&C.false_kind[k]);
        } else {
            shadow.mark_read(h.slot, consumer == LOOKAHEAD);
        }
        HitInfo {
            hit: true,
            false_hit,
            origin: meta_origin(meta),
            pv_writer: meta_pv_node(meta),
            foreign_move: !false_hit && meta_foreign_move(meta),
            cut: false,
        }
    }

    /// The table, entry by entry (`GAIA_TTSTAT_SCAN=1`).
    #[derive(Default)]
    pub struct Scan {
        slots: u64,
        empty: u64,
        /// Entries by kind, of the current search and older.
        cur: [u64; 3],
        old: [u64; 3],
        /// Entries with the pv flag [current, older].
        pv: [u64; 2],
        /// Entries by depth bin, current and older.
        dcur: [u64; 7],
        dold: [u64; 7],
        /// Current entries by origin.
        ocur: [u64; 7],
        /// Entries by reads since their last write [search, look-ahead only, never].
        rcur: [u64; 3],
        rold: [u64; 3],
        /// Entries by reads since their position took the slot [search, look-ahead
        /// only, never].
        ecur: [u64; 3],
        eold: [u64; 3],
        /// Entries holding a move another position wrote [current, older].
        fmv: [u64; 2],
        /// Entries by relative age [0, 1, 2, 3, 4+].
        age: [u64; 5],
    }

    impl Scan {
        /// One slot: `rel_age` is its age relative to the current search.
        #[allow(clippy::too_many_arguments)]
        pub fn add(&mut self, empty: bool, bound: Bound, depth: i32, rel_age: u8, pv: bool, meta: u16) {
            self.slots += 1;
            if empty {
                self.empty += 1;
                return;
            }
            let k = kind(bound, depth);
            let d = depth_bin(bound, depth);
            let r = meta_read(meta);
            let e = meta_ever(meta);
            let f = meta_foreign_move(meta) as u64;
            self.age[(rel_age as usize).min(4)] += 1;
            if rel_age == 0 {
                self.cur[k] += 1;
                self.dcur[d] += 1;
                self.ocur[meta_origin(meta) as usize] += 1;
                self.rcur[r] += 1;
                self.ecur[e] += 1;
                self.fmv[0] += f;
                self.pv[0] += pv as u64;
            } else {
                self.old[k] += 1;
                self.dold[d] += 1;
                self.rold[r] += 1;
                self.eold[e] += 1;
                self.fmv[1] += f;
                self.pv[1] += pv as u64;
            }
        }

        fn line(&self) -> String {
            let arr = |a: &[u64]| a.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",");
            format!(
                "TTSCAN slots={} empty={} cur={} old={} pv={} dcur={} dold={} ocur={} rcur={} rold={} \
                 ecur={} eold={} fmv={} age={}",
                self.slots,
                self.empty,
                arr(&self.cur),
                arr(&self.old),
                arr(&self.pv),
                arr(&self.dcur),
                arr(&self.dold),
                arr(&self.ocur),
                arr(&self.rcur),
                arr(&self.rold),
                arr(&self.ecur),
                arr(&self.eold),
                arr(&self.fmv),
                arr(&self.age),
            )
        }
    }

    /// Prints the cumulative `TTSTAT` line on stderr, then the `TTSCAN` line of the
    /// whole table when `GAIA_TTSTAT_SCAN=1`.
    pub fn print(tt: &TT) {
        let mut s = String::from("TTSTAT");
        let _ = write!(s, " mb={} hashfull={} age={}", tt.size_mb(), tt.hashfull(), tt.stat_age());
        C.write_line(&mut s);
        eprintln!("{s}");
        if std::env::var("GAIA_TTSTAT_SCAN").is_ok_and(|v| v == "1") {
            eprintln!("{}", tt.stat_scan().line());
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn buckets_and_bins() {
            assert_eq!(bucket(1), 0);
            assert_eq!(bucket(4), 1);
            assert_eq!(bucket(7), 2);
            assert_eq!(bucket(11), 3);
            assert_eq!(bucket(40), 4);
            assert_eq!(depth_bin(Bound::None, -2), 0);
            assert_eq!(depth_bin(Bound::Lower, 0), 1);
            assert_eq!(depth_bin(Bound::Upper, 12), 6);
            assert_eq!(kind(Bound::Exact, 0), 1);
        }

        #[test]
        fn origin_bytes_round_trip() {
            let m = meta_of(O_STANDPAT, true);
            assert_eq!(meta_origin(m), O_STANDPAT);
            assert!(meta_pv_node(m));
            assert_eq!(meta_read(m), 2);
            assert_eq!(meta_read(m | META_READ_LOOKAHEAD), 1);
            assert_eq!(meta_read(m | META_READ_LOOKAHEAD | META_READ_SEARCH), 0);
            assert_eq!(meta_ever(m), 2);
            assert!(!meta_foreign_move(m));
        }

        #[test]
        fn shadow_reads_stick_to_their_position() {
            let s = Shadow::new(3);
            s.write(1, 42, meta_of(O_MAIN, false), false);
            s.mark_read(1, false);
            assert_eq!(meta_read(s.meta(1)), 0);
            assert_eq!(meta_ever(s.meta(1)), 0);
            // A rewrite of the same position: read since the write no more, ever still.
            s.write(1, 42, meta_of(O_QS, false), false);
            assert_eq!(meta_read(s.meta(1)), 2);
            assert_eq!(meta_ever(s.meta(1)), 0);
            assert_eq!(meta_origin(s.meta(1)), O_QS);
            // Another position takes the slot: everything starts afresh.
            s.write(1, 43, meta_of(O_EVAL, false), true);
            assert_eq!(meta_ever(s.meta(1)), 2);
            assert!(meta_foreign_move(s.meta(1)));
            s.mark_read(1, true);
            assert_eq!(meta_ever(s.meta(1)), 1);
            s.set_foreign_move(1, false);
            assert!(!meta_foreign_move(s.meta(1)));
            assert_eq!(meta_ever(s.meta(1)), 1);
        }
    }
}
