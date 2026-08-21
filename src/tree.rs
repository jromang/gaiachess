//! Search tree recorder — GTREE v1 binary format (feature `tree`).
//!
//! The recorder captures the full search tree as a flat stream of fixed-size
//! 16-byte little-endian records, buffered in memory and flushed to a file
//! after the search. The stream is a pre-order traversal: every recorded node
//! emits one `Enter` and exactly one `Exit` (a stack machine reconstructs the
//! tree). `Move` records are decision annotations emitted *before* the child
//! subtree they introduce; a child's returned score is read from its own
//! `Exit` record (negated by the parent).
//!
//! Nesting notes for the parser:
//! - Singular-extension verification searches run at the SAME ply as their
//!   parent (Enter flag `F_EXCLUDED`).
//! - Razoring and NMP verification drop into quiescence at the same ply; the
//!   qsearch subtree nests between the node's Enter and its Exit.
//! - Nodes are recorded only after the node counter increments; early returns
//!   before that point (stop, ply overflow, leaf-to-qsearch dispatch) emit
//!   nothing.
//!
//! File layout: header (magic, options, FEN, meta JSON) + records + 16-byte
//! footer (record count + truncation flag). All integers little-endian.
//! The enum values below are a contract with `tools/treescope/gtree.py` —
//! never renumber, only append.

// ─── Record tags ────────────────────────────────────────────────────────────
pub const TAG_ENTER: u8 = 1;
pub const TAG_EVAL: u8 = 2;
pub const TAG_MOVE: u8 = 3;
pub const TAG_EXIT: u8 = 4;

// ─── Enter flags ────────────────────────────────────────────────────────────
pub const F_QS: u8 = 1;
pub const F_IN_CHECK: u8 = 2;
pub const F_PV: u8 = 4;
pub const F_ROOT: u8 = 8;
pub const F_CUT_NODE: u8 = 16;
pub const F_EXCLUDED: u8 = 32;
pub const F_SKIP_NULL: u8 = 64;

// ─── Eval flags ─────────────────────────────────────────────────────────────
pub const EF_TT_HIT: u8 = 1;
// bits 1-2: tt bound (0 none, 1 upper, 2 lower, 3 exact), shifted left by 1
pub const EF_TT_PV: u8 = 8;
pub const EF_IMPROVING: u8 = 16;
pub const EF_IN_CHECK: u8 = 32;

// ─── Move actions ───────────────────────────────────────────────────────────
pub const A_PRUNED_SEE: u8 = 0;
pub const A_PRUNED_HIST: u8 = 1;
pub const A_PRUNED_LMP: u8 = 2;
pub const A_PRUNED_FUTILITY: u8 = 3;
/// Reserved: the parser and the audit tool both know this code, but nothing emits it
/// yet — the move picker skips its quiet stages wholesale rather than move by move, so
/// there is no single move to hang it on. The number stays taken; the format is
/// append-only.
#[allow(dead_code)]
pub const A_SKIPPED_QUIET: u8 = 4;
pub const A_EXCLUDED_SE: u8 = 5;
pub const A_SEARCH_FULL: u8 = 6;
pub const A_SEARCH_LMR: u8 = 7;
pub const A_RESEARCH_FULL: u8 = 8;
pub const A_RESEARCH_PV: u8 = 9;
pub const A_PROBCUT_QS: u8 = 10;
pub const A_PROBCUT_AB: u8 = 11;
pub const A_NULL_SEARCH: u8 = 12;
pub const A_SE_VERIF: u8 = 13;
pub const A_QS_SEARCH: u8 = 14;
pub const A_QS_PRUNED: u8 = 15;

// ─── Exit reasons ───────────────────────────────────────────────────────────
pub const X_NORMAL: u8 = 0;
pub const X_TT_CUTOFF: u8 = 1;
pub const X_TB_ONLINE: u8 = 2;
pub const X_TB_DTM: u8 = 3;
pub const X_TB_NALIMOV: u8 = 4;
pub const X_TB_WDL: u8 = 5;
pub const X_REPETITION: u8 = 6;
pub const X_DRAW: u8 = 7;
pub const X_MATE_DISTANCE: u8 = 8;
pub const X_RFP: u8 = 9;
pub const X_RAZOR_TO_QS: u8 = 10;
pub const X_NMP_CUTOFF: u8 = 11;
pub const X_NMP_VERIF_TO_QS: u8 = 12;
pub const X_PROBCUT: u8 = 13;
pub const X_SE_MULTICUT: u8 = 14;
pub const X_SE_NO_LEGAL: u8 = 15;
pub const X_CHECKMATE: u8 = 16;
pub const X_STALEMATE: u8 = 17;
pub const X_STOPPED: u8 = 18;
pub const X_QS_STANDPAT: u8 = 19;
pub const X_QS_TT_CUTOFF: u8 = 20;
pub const X_QS_MATE_DISTANCE: u8 = 21;
pub const X_QS_MATED: u8 = 22;
pub const X_QS_NORMAL: u8 = 23;

/// Run `$body` with `$t` bound to the active recorder, if any.
/// Compiles to nothing without the `tree` feature; arguments must never
/// have side effects.
#[cfg(feature = "tree")]
macro_rules! tr {
    ($td:expr, $t:ident, $($body:tt)*) => {{
        if let Some($t) = $td.tree.as_deref_mut() { $($body)* }
    }};
}
#[cfg(not(feature = "tree"))]
macro_rules! tr {
    ($td:expr, $t:ident, $($body:tt)*) => {{}};
}
pub(crate) use tr;

/// Record an early node exit and return `$val`. The value is evaluated
/// BEFORE the recorder is borrowed (it may recurse into the search).
/// Early exits carry no bound/best_move/move_count.
#[cfg(feature = "tree")]
macro_rules! tree_ret {
    ($td:expr, $ply:expr, $reason:expr, $depth:expr, $val:expr) => {{
        let v = $val;
        if let Some(t) = $td.tree.as_deref_mut() {
            t.exit($ply as u8, $reason, 0, 0, v, 0, $depth);
        }
        return v;
    }};
}
#[cfg(not(feature = "tree"))]
macro_rules! tree_ret {
    ($td:expr, $ply:expr, $reason:expr, $depth:expr, $val:expr) => {{
        return $val;
    }};
}
pub(crate) use tree_ret;

#[cfg(feature = "tree")]
pub use imp::TreeRec;

/// Map a TT bound to the 2-bit encoding shared with the parser.
#[cfg(feature = "tree")]
pub fn bound_bits(b: crate::tt::Bound) -> u8 {
    match b {
        crate::tt::Bound::None => 0,
        crate::tt::Bound::Upper => 1,
        crate::tt::Bound::Lower => 2,
        crate::tt::Bound::Exact => 3,
    }
}

#[cfg(feature = "tree")]
mod imp {
    use super::*;

    const RECORD_SIZE: usize = 16;
    const MAGIC: &[u8; 4] = b"GTRE";
    const FOOTER_MAGIC: &[u8; 4] = b"GEND";
    const VERSION: u16 = 1;

    /// In-memory search tree recorder. Owned by `ThreadData` (boxed, `Option`),
    /// installed only by the dump runner — recording never happens during play.
    pub struct TreeRec {
        buf: Vec<u8>,
        cap_bytes: usize,
        truncated: bool,
        /// Subtrees with remaining depth below this are not recorded
        /// (0 = record everything, including quiescence).
        min_record_depth: i32,
        /// Record per-move records inside quiescence nodes.
        pub qs_moves: bool,
        /// Nesting count of the currently suppressed subtree (0 = recording).
        suppress: u32,
        records: u64,
        /// Enter/Exit pairing check (debug builds only).
        #[cfg(debug_assertions)]
        shadow: Vec<u8>,
    }

    impl TreeRec {
        pub fn new(cap_bytes: usize, min_record_depth: i32, qs_moves: bool) -> Self {
            TreeRec {
                buf: Vec::with_capacity(cap_bytes.min(64 * 1024 * 1024)),
                cap_bytes,
                truncated: false,
                min_record_depth,
                qs_moves,
                suppress: 0,
                records: 0,
                #[cfg(debug_assertions)]
                shadow: Vec::new(),
            }
        }

        /// True when events at this point are being written (used to skip
        /// argument computation such as SEE-based move classification).
        #[inline]
        pub fn recording(&self) -> bool {
            !self.truncated && self.suppress == 0
        }

        #[inline]
        fn room(&mut self) -> bool {
            if self.truncated {
                return false;
            }
            if self.buf.len() + RECORD_SIZE > self.cap_bytes {
                self.truncated = true;
                return false;
            }
            true
        }

        /// Node entry. `depth` is the remaining depth (0 for quiescence).
        #[inline]
        pub fn enter(&mut self, ply: u8, depth: i32, flags: u8, alpha: i32, beta: i32) {
            if self.truncated {
                return;
            }
            if self.suppress > 0 {
                self.suppress += 1;
                return;
            }
            if self.min_record_depth > 0 && depth < self.min_record_depth {
                self.suppress = 1;
                return;
            }
            if !self.room() {
                return;
            }
            #[cfg(debug_assertions)]
            self.shadow.push(ply);
            debug_assert!((-32768..=32767).contains(&alpha) && (-32768..=32767).contains(&beta));
            let mut rec = [0u8; RECORD_SIZE];
            rec[0] = TAG_ENTER;
            rec[1] = ply;
            rec[2] = depth.clamp(-128, 127) as i8 as u8;
            rec[3] = flags;
            rec[4..6].copy_from_slice(&(alpha as i16).to_le_bytes());
            rec[6..8].copy_from_slice(&(beta as i16).to_le_bytes());
            self.push(&rec);
        }

        /// Static eval + TT context, emitted once per recorded node (skipped
        /// when the node exits before evaluation, e.g. TT cutoff).
        #[allow(clippy::too_many_arguments)]
        #[inline]
        pub fn eval(
            &mut self,
            ply: u8,
            static_eval: i32,
            raw_eval: i32,
            tt_move: u16,
            tt_score: i32,
            tt_eval: i32,
            tt_depth: i32,
            flags: u8,
        ) {
            if !self.recording() || !self.room() {
                return;
            }
            let mut rec = [0u8; RECORD_SIZE];
            rec[0] = TAG_EVAL;
            rec[1] = ply;
            rec[2..4].copy_from_slice(&(static_eval as i16).to_le_bytes());
            rec[4..6].copy_from_slice(&(raw_eval as i16).to_le_bytes());
            rec[6..8].copy_from_slice(&tt_move.to_le_bytes());
            rec[8..10].copy_from_slice(&(tt_score as i16).to_le_bytes());
            rec[10..12].copy_from_slice(&(tt_eval as i16).to_le_bytes());
            rec[12] = tt_depth.clamp(-128, 127) as i8 as u8;
            rec[13] = flags;
            self.push(&rec);
        }

        /// Move decision. For search actions this precedes the child subtree;
        /// for prune actions no child follows. `red_or_ext` is in centiplies
        /// (1024 = 1 ply, positive = reduction, negative = extension).
        #[allow(clippy::too_many_arguments)]
        #[inline]
        pub fn mv(
            &mut self,
            ply: u8,
            m: u16,
            move_index: u32,
            category: u8,
            action: u8,
            red_or_ext: i32,
            new_depth: i32,
        ) {
            if !self.recording() || !self.room() {
                return;
            }
            let mut rec = [0u8; RECORD_SIZE];
            rec[0] = TAG_MOVE;
            rec[1] = ply;
            rec[2..4].copy_from_slice(&m.to_le_bytes());
            rec[4] = move_index.min(255) as u8;
            rec[5] = category;
            rec[6] = action;
            rec[7] = new_depth.clamp(-128, 127) as i8 as u8;
            rec[8..10].copy_from_slice(&(red_or_ext.clamp(-32768, 32767) as i16).to_le_bytes());
            self.push(&rec);
        }

        /// Node exit. `returned` is the exact value the search function
        /// returns from this node (parents see its negation).
        #[allow(clippy::too_many_arguments)]
        #[inline]
        pub fn exit(
            &mut self,
            ply: u8,
            reason: u8,
            bound: u8,
            best_move: u16,
            returned: i32,
            move_count: u32,
            final_depth: i32,
        ) {
            if self.truncated {
                return;
            }
            if self.suppress > 0 {
                self.suppress -= 1;
                return;
            }
            if !self.room() {
                return;
            }
            #[cfg(debug_assertions)]
            {
                let top = self.shadow.pop();
                debug_assert!(
                    top == Some(ply),
                    "tree: unbalanced Enter/Exit — exit ply {} but stack top {:?}",
                    ply,
                    top
                );
            }
            debug_assert!((-32768..=32767).contains(&returned));
            let mut rec = [0u8; RECORD_SIZE];
            rec[0] = TAG_EXIT;
            rec[1] = ply;
            rec[2] = reason;
            rec[3] = bound;
            rec[4..6].copy_from_slice(&best_move.to_le_bytes());
            rec[6..8].copy_from_slice(&(returned as i16).to_le_bytes());
            rec[8] = move_count.min(255) as u8;
            rec[9] = final_depth.clamp(-128, 127) as i8 as u8;
            self.push(&rec);
        }

        #[inline]
        fn push(&mut self, rec: &[u8; RECORD_SIZE]) {
            self.buf.extend_from_slice(rec);
            self.records += 1;
        }

        /// Assemble header + records + footer and write the file.
        /// Returns a one-line summary for stderr.
        pub fn write_file(
            &self,
            path: &str,
            fen: &str,
            moves: &[String],
            meta_json: &str,
        ) -> std::io::Result<String> {
            let moves_str = moves.join(" ");
            let mut out: Vec<u8> = Vec::with_capacity(self.buf.len() + 256);
            out.extend_from_slice(MAGIC);
            out.extend_from_slice(&VERSION.to_le_bytes());
            out.extend_from_slice(&(RECORD_SIZE as u16).to_le_bytes());
            let flags: u32 =
                (self.qs_moves as u32) | ((self.truncated as u32) << 1);
            out.extend_from_slice(&flags.to_le_bytes());
            out.extend_from_slice(&self.min_record_depth.to_le_bytes());
            out.extend_from_slice(&(fen.len() as u16).to_le_bytes());
            out.extend_from_slice(fen.as_bytes());
            out.extend_from_slice(&(moves_str.len() as u16).to_le_bytes());
            out.extend_from_slice(moves_str.as_bytes());
            out.extend_from_slice(&(meta_json.len() as u32).to_le_bytes());
            out.extend_from_slice(meta_json.as_bytes());
            out.extend_from_slice(&self.buf);
            out.extend_from_slice(FOOTER_MAGIC);
            out.extend_from_slice(&self.records.to_le_bytes());
            out.push(self.truncated as u8);
            out.extend_from_slice(&[0u8; 3]);
            std::fs::write(path, &out)?;
            Ok(format!(
                "tree: {} records ({:.1} MB){} -> {}",
                self.records,
                out.len() as f64 / (1024.0 * 1024.0),
                if self.truncated { " TRUNCATED" } else { "" },
                path
            ))
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// Golden bytes: the binary contract with tools/treescope/gtree.py.
        /// If this test needs updating, bump VERSION and update the parser.
        #[test]
        fn test_golden_records() {
            let mut t = TreeRec::new(1 << 20, 0, false);
            t.enter(3, 7, F_PV | F_IN_CHECK, -150, 200);
            t.eval(3, 42, 40, 0x1234, 55, 50, 6, EF_TT_HIT | (3 << 1) | EF_IMPROVING);
            t.mv(3, 0x1234, 1, 0, A_SEARCH_LMR, 2048, 5);
            t.exit(3, X_NORMAL, 2, 0x1234, 60, 12, 7);
            assert_eq!(t.records, 4);
            let b = &t.buf;
            assert_eq!(
                &b[0..16],
                &[1, 3, 7, 6, 0x6A, 0xFF, 0xC8, 0x00, 0, 0, 0, 0, 0, 0, 0, 0],
                "Enter record"
            );
            assert_eq!(
                &b[16..32],
                &[2, 3, 42, 0, 40, 0, 0x34, 0x12, 55, 0, 50, 0, 6, 0x17, 0, 0],
                "Eval record"
            );
            assert_eq!(
                &b[32..48],
                &[3, 3, 0x34, 0x12, 1, 0, A_SEARCH_LMR, 5, 0x00, 0x08, 0, 0, 0, 0, 0, 0],
                "Move record"
            );
            assert_eq!(
                &b[48..64],
                &[4, 3, X_NORMAL, 2, 0x34, 0x12, 60, 0, 12, 7, 0, 0, 0, 0, 0, 0],
                "Exit record"
            );
        }

        #[test]
        fn test_suppression_pairing() {
            // min_record_depth = 3: depth-2 subtree suppressed entirely,
            // nested enters/exits stay balanced.
            let mut t = TreeRec::new(1 << 20, 3, false);
            t.enter(0, 5, 0, -100, 100); // recorded
            t.enter(1, 2, 0, -100, 100); // suppressed (depth < 3)
            t.enter(2, 1, 0, -100, 100); // nested in suppressed subtree
            t.exit(2, X_NORMAL, 0, 0, 0, 0, 1);
            t.exit(1, X_NORMAL, 0, 0, 0, 0, 2);
            t.enter(1, 4, 0, -100, 100); // recorded again
            t.exit(1, X_NORMAL, 0, 0, 0, 0, 4);
            t.exit(0, X_NORMAL, 0, 0, 0, 0, 5);
            assert_eq!(t.records, 4, "only the two recorded nodes remain");
            assert!(t.recording());
        }

        #[test]
        fn test_truncation() {
            // Cap allows exactly 2 records; the rest is dropped.
            let mut t = TreeRec::new(32, 0, false);
            t.enter(0, 5, 0, -1, 1);
            t.eval(0, 1, 1, 0, 2, 2, 1, 0);
            t.enter(1, 4, 0, -1, 1);
            t.exit(1, X_NORMAL, 0, 0, 0, 0, 4);
            t.exit(0, X_NORMAL, 0, 0, 0, 0, 5);
            assert_eq!(t.records, 2);
            assert!(!t.recording());
        }

        #[test]
        fn test_write_file_roundtrip_header() {
            let mut t = TreeRec::new(1 << 20, 0, true);
            t.enter(0, 1, F_ROOT | F_PV, -32001, 32001);
            t.exit(0, X_NORMAL, 3, 0, 0, 1, 1);
            let dir = std::env::temp_dir().join("gaiachess_tree_test.gtree");
            let path = dir.to_str().unwrap();
            let summary = t
                .write_file(path, "8/8/8/8/8/8/8/8 w - - 0 1", &["e2e4".into()], "{}")
                .unwrap();
            assert!(summary.contains("2 records"));
            let data = std::fs::read(path).unwrap();
            assert_eq!(&data[0..4], b"GTRE");
            assert_eq!(u16::from_le_bytes([data[4], data[5]]), 1);
            assert_eq!(u16::from_le_bytes([data[6], data[7]]), 16);
            let flags = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
            assert_eq!(flags & 1, 1, "qs_moves flag");
            assert_eq!(&data[data.len() - 16..data.len() - 12], b"GEND");
            let count = u64::from_le_bytes(data[data.len() - 12..data.len() - 4].try_into().unwrap());
            assert_eq!(count, 2);
            std::fs::remove_file(path).ok();
        }
    }
}
