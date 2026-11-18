//! Quiescence telemetry (`--features stats`): where the quiescence spends its nodes,
//! how its nodes end, what happens to each move, where its evaluations come from, and
//! a few counterfactuals that decide nothing.
//!
//! One `QSSTAT` line of cumulative counters goes to stderr after every `go` and at the
//! end of the bench, `name=v0,v1,...` for each counter, indexed as its doc says.
//!
//! Nothing here feeds back into the search: the hooks read what the search computed and
//! write counters it never reads. The counterfactuals call pure functions of the
//! position (SEE, move generation, legality) and draw extra moves from a move picker
//! the node has finished with. The bench node count is the same with and without the
//! feature. Without the feature the `qss!` blocks vanish and only an empty `print` is
//! left.

#[cfg(feature = "stats")]
pub use imp::*;

/// Runs its statements in a statistics build only (tokens discarded otherwise, so they
/// must never have side effects on the search).
#[cfg(feature = "stats")]
macro_rules! qss {
    ($($body:tt)*) => {{ $($body)* }};
}
#[cfg(not(feature = "stats"))]
macro_rules! qss {
    ($($body:tt)*) => {{}};
}
pub(crate) use qss;

/// Prints the `QSSTAT` line: nothing without the feature.
#[cfg(not(feature = "stats"))]
pub fn print() {}

#[cfg(feature = "stats")]
mod imp {
    use std::cell::Cell;
    use std::fmt::Write;
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    /// Who called the quiescence: the index of the per-caller counters.
    pub const K_LEAF: u8 = 0;
    pub const K_RAZOR: u8 = 1;
    pub const K_PROBCUT: u8 = 2;
    pub const K_RECURSE: u8 = 3;
    pub const K_OTHER: u8 = 4;

    /// Node outcomes (offset by 8 at non-PV nodes).
    pub const O_TT: usize = 0;
    pub const O_STANDPAT: usize = 1;
    pub const O_MDP: usize = 2;
    pub const O_FAIL_HIGH: usize = 3;
    pub const O_UPPER: usize = 4;
    pub const O_RAISED: usize = 5;
    pub const O_MATED: usize = 6;
    pub const O_STOPPED: usize = 7;

    /// Exits before the node counter.
    pub const X_STOP: usize = 0;
    pub const X_PLY: usize = 1;
    pub const X_UPCOMING: usize = 2;
    pub const X_DRAW: usize = 3;
    pub const X_TB: usize = 4;

    /// Where the node's final best score came from.
    pub const B_STATIC: u8 = 0;
    pub const B_FUTILITY: u8 = 1;
    pub const B_MOVE: u8 = 2;

    /// Piece values in the search's internal unit (a pawn is about 458 there, see the
    /// score rule), for the victim-value futility counterfactual: the SEE's centipawn
    /// values times 4.58.
    pub const NET_PER_CP_NUM: i32 = 458;
    pub const NET_PER_CP_DEN: i32 = 100;
    const SEE_CP: [i32; 7] = [100, 320, 330, 500, 900, 20000, 0];

    /// A captured piece's value in internal units (none, king: 0).
    pub fn net_value(pt: usize) -> i32 {
        debug_assert!(pt < 7);
        if pt >= 5 { 0 } else { SEE_CP[pt] * NET_PER_CP_NUM / NET_PER_CP_DEN }
    }

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
        // --- A. Where the nodes are ---
        /// Calls by caller [leaf of the main search, razoring, ProbCut, recursion, other].
        calls: 5;
        /// Counted nodes by the caller of their subtree's first quiescence node
        /// [leaf, razoring, ProbCut, -, other].
        nodes_by: 5;
        /// Counted nodes [PV, non-PV].
        nodes_pv: 2;
        /// Counted nodes by distance from the horizon (1 = the first quiescence ply,
        /// index d-1, last = 12+), and those in check.
        qsd: 12;
        check_qsd: 12;
        /// Exits before the node counter [stop, ply guard, upcoming repetition, draw,
        /// GaiaTB].
        pre_exit: 5;

        // --- B. How the nodes end ---
        /// Outcomes [TT cutoff, stand-pat cutoff, mate distance, fail high on a move,
        /// upper bound with alpha never raised, alpha raised below beta, mated,
        /// stopped], PV nodes then non-PV nodes.
        out: 16;
        /// Outcomes by distance from the horizon [1, 2, 3, 4+] x [TT, stand-pat,
        /// fail high, other].
        out_qsd: 16;
        /// TT cutoffs by the entry's depth [< 0, 0, 1-2, 3+], bound [exact, lower,
        /// upper], and writer (ttstat origins [unknown, main, eval, probcut, tb,
        /// stand-pat, qs]).
        tt_cut_depth: 4;
        tt_cut_bound: 3;
        tt_cut_origin: 7;
        /// Non-PV scored entries on the right side of the window, refused only by the
        /// halfmove clock (>= 96).
        tt_cut_hm: 1;
        /// Stand-pat [cutoffs on the plain stand-pat, cutoffs made by the TT score
        /// (stand-pat below beta), cutoffs undone by the TT score (stand-pat at or
        /// above beta, TT score below), nodes whose best score the TT score set,
        /// cutoffs on a lazy evaluation, cutoffs that wrote the TT].
        sp: 6;
        /// Beta midpoint at the stand-pat [applied, moved the score by more than 20,
        /// by more than 100] and at the end of the node [same].
        sp_mid: 3;
        fin_mid: 3;
        /// Fail highs by the cutting move's `move_count` (1..11, 12+), by the number of
        /// moves searched before and with it (1..7, 8+), and by kind [TT move, capture,
        /// promotion, quiet evasion, capture in check].
        fh_idx: 12;
        fh_srch: 8;
        fh_kind: 5;

        // --- C. Moves ---
        /// Moves from the picker (`picked`) [out of check, in check], illegal ones, and those the
        /// level's blind spots hid.
        picked: 2;
        illegal: 2;
        blind: 1;
        /// Quiet TT moves skipped out of check, by the entry's bound [none, upper,
        /// lower, exact] (#282).
        ttq_skip: 4;
        /// Quiet evasions skipped once a move stopped losing.
        evas_skip: 1;
        /// Futility out of check [moves reaching the test, pruned, pruned moves giving
        /// check, pruned promotions].
        fut: 4;
        /// SEE pruning [pruned out of check, pruned in check, pruned though the plain
        /// threshold would keep it (history pruned it), kept though the plain threshold
        /// would prune it (history kept it), moves reaching the test].
        see_pr: 5;
        /// Move-count breaks [breaks, moves left in the picker (legal), of which
        /// recaptures, giving check, promotions, SEE >= 1, captures that would pass the
        /// node's SEE pruning].
        mcb: 7;
        /// Breaks [after a skipped quiet TT move took a place, after an underpromotion
        /// took a place].
        mcb_why: 2;
        /// Moves searched [captures out of check, queen promotions, underpromotions,
        /// captures in check, quiet evasions, the TT move].
        srch: 6;

        // --- D. Evaluation ---
        /// Stand-pat evaluations [TT eval reused, network, lazy PeSTO, level].
        ev: 4;
        /// Accumulators [owed (TT eval), settled before a child, lazy nodes with a
        /// child searched, lazy nodes].
        acc: 4;
        /// Nodes out of check ending in the move loop without a fail high: final score
        /// minus stand-pat [< -500, -500..-201, -200..-51, -50..-1, 0, 1..50, 51..200,
        /// 201..500, > 500].
        dres: 9;
        /// Those nodes by the source of their final score [stand-pat or TT score,
        /// futility base, a move].
        best_src: 3;

        // --- E. Counterfactuals ---
        /// A victim-value futility (exempts checks, moves to the previous move's square and
        /// promotions; prunes when base + captured value <= alpha, or when SEE <
        /// alpha - base) against ours, at each move reaching our futility test [both
        /// prune, ours only, the other only, neither], and the other's reason
        /// [value stage, SEE stage].
        sf_fut: 4;
        sf_why: 2;
        /// Out-of-check nodes whose picker ran dry: [nodes, captures the split dropped
        /// (legal, not the TT move), of which the node's SEE pruning would keep, and the
        /// futility too, and the first one would also pass the move-count limit,
        /// dropped with SEE >= 1].
        split: 6;
        /// Stalemates [stand-pat cutoff on one, node ending with no move on one, nodes
        /// out of check ending with no legal capture, stand-pat cutoffs checked].
        stale: 4;
        /// What the node's TT write found [empty slot, its position at depth <= 0 or eval
        /// only (rewritten), its position deeper (rewritten), its position (refused),
        /// another position (victim), alias]: fail-high writes, all final writes,
        /// stand-pat writes.
        st_fh: 6;
        st_all: 6;
        st_sp: 6;
    }

    fn push(out: &mut String, name: &str, values: &[AtomicU64]) {
        let v: Vec<String> = values.iter().map(|a| a.load(Relaxed).to_string()).collect();
        let _ = write!(out, " {}={}", name, v.join(","));
    }

    #[inline]
    pub fn inc(a: &AtomicU64) {
        a.fetch_add(1, Relaxed);
    }

    #[inline]
    pub fn add(a: &AtomicU64, v: u64) {
        a.fetch_add(v, Relaxed);
    }

    thread_local! {
        /// The next quiescence call's caller and its parent's distance from the horizon
        /// (0 for a first quiescence node), consumed at the entry.
        static CALL: Cell<(u8, u8)> = const { Cell::new((K_OTHER, 0)) };
        /// What the last TT write on this thread found (`st_*` classes, 255 = none).
        static LAST_STORE: Cell<u8> = const { Cell::new(255) };
    }

    /// Names the caller of the quiescence call that follows (a first quiescence node).
    #[inline]
    pub fn call(kind: u8) {
        debug_assert!(kind <= K_OTHER && kind != K_RECURSE);
        CALL.with(|c| c.set((kind, 0)));
    }

    /// The TT records what a write found (see `st_*`).
    #[inline]
    pub fn set_last_store(class: u8) {
        debug_assert!(class < 6);
        LAST_STORE.with(|c| c.set(class));
    }

    /// What the last write found, consumed (255 = no write recorded).
    #[inline]
    pub fn take_last_store() -> u8 {
        LAST_STORE.with(|c| c.replace(255))
    }

    /// The node's own statistics, a local of `quiescence`.
    pub struct Node {
        /// The caller of the subtree's first quiescence node.
        pub kind: u8,
        /// Distance from the horizon, 1 for a first quiescence node.
        pub qsd: u8,
        pub pv: bool,
        pub lazy: bool,
        pub best_src: u8,
        /// A skipped quiet TT move, an underpromotion, took a place of the limit.
        pub ttq_counted: bool,
        pub underpromo_counted: bool,
        pub searched: u32,
    }

    impl Node {
        /// Consumes the caller recorded by `call` or by the parent, and counts the call.
        #[inline]
        pub fn enter(pv: bool) -> Node {
            let (kind, parent) = CALL.with(|c| c.replace((K_OTHER, 0)));
            inc(&C.calls[if parent == 0 { kind as usize } else { K_RECURSE as usize }]);
            Node {
                kind,
                qsd: parent.saturating_add(1),
                pv,
                lazy: false,
                best_src: B_STATIC,
                ttq_counted: false,
                underpromo_counted: false,
                searched: 0,
            }
        }

        /// The child's call: same subtree, one ply further.
        #[inline]
        pub fn call_child(&self) {
            CALL.with(|c| c.set((self.kind, self.qsd)));
        }

        /// Counts the node itself (where the search counts it).
        #[inline]
        pub fn counted(&self, in_check: bool) {
            inc(&C.nodes_by[self.kind as usize]);
            inc(&C.nodes_pv[!self.pv as usize]);
            let d = (self.qsd as usize).clamp(1, 12) - 1;
            inc(&C.qsd[d]);
            if in_check {
                inc(&C.check_qsd[d]);
            }
        }

        /// Counts the node's outcome.
        #[inline]
        pub fn out(&self, o: usize) {
            debug_assert!(o < 8);
            inc(&C.out[o + if self.pv { 0 } else { 8 }]);
            let d = (self.qsd as usize).clamp(1, 4) - 1;
            let k = match o {
                O_TT => 0,
                O_STANDPAT => 1,
                O_FAIL_HIGH => 2,
                _ => 3,
            };
            inc(&C.out_qsd[d * 4 + k]);
        }
    }

    /// An exit before the node counter.
    #[inline]
    pub fn pre_exit(x: usize) {
        inc(&C.pre_exit[x]);
    }

    /// Bucket of a depth for the TT cutoff counter.
    pub fn tt_depth_bucket(d: i32) -> usize {
        match d {
            i32::MIN..=-1 => 0,
            0 => 1,
            1..=2 => 2,
            _ => 3,
        }
    }

    /// Bucket of (final - stand-pat).
    pub fn dres_bucket(d: i32) -> usize {
        match d {
            i32::MIN..=-501 => 0,
            -500..=-201 => 1,
            -200..=-51 => 2,
            -50..=-1 => 3,
            0 => 4,
            1..=50 => 5,
            51..=200 => 6,
            201..=500 => 7,
            _ => 8,
        }
    }

    /// Counts a beta midpoint into `c` ([applied, > 20, > 100]).
    pub fn midpoint(c: &[AtomicU64; 3], before: i32, after: i32) {
        inc(&c[0]);
        let d = (before - after).abs();
        if d > 20 {
            inc(&c[1]);
        }
        if d > 100 {
            inc(&c[2]);
        }
    }

    /// Counts a TT write of the node into `c` and `st_all`-like arrays.
    pub fn store_found(c: &[AtomicU64; 6]) {
        let s = take_last_store();
        if s < 6 {
            inc(&c[s as usize]);
        }
    }

    /// Prints the `QSSTAT` line.
    pub fn print() {
        let mut s = String::from("QSSTAT");
        C.write_line(&mut s);
        eprintln!("{s}");
    }
}
