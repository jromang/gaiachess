#![allow(unsafe_op_in_unsafe_fn)]
// Hand-vectorised code: the loop index is the point. It walks several arrays in
// lockstep, steps by a SIMD lane count, and feeds raw pointer arithmetic — an
// iterator would hide the arithmetic these kernels exist to control.
#![allow(clippy::needless_range_loop)]
//! Threat features for NNUE (GaiaNet-T2 pairwise encoding).
//!
//! 59,808 features encoding "piece A on square S attacks/defends piece B on square T".
//! Kings never attack and are never targets. Pawn-pawn relations are *not* in this
//! set: they live in `pawn_pairs` (4,560 features, 3-file window) and share the i8
//! weight array at indices `0 .. PAWN_PAIR_SIZE`; threats start at `THREAT_OFFSET`.
//!
//! Feature index: `pair_base(att, def, att_sq < def_sq) + square_index(type(att), att_sq, def_sq)`
//!
//! Lookup tables are computed at compile time.
//!
//! **Incremental updates**: when a move is played (`Network::push`, before `make_move`),
//! the attacks it creates and destroys are recorded for its ply (`move_threats`). The
//! record carries no perspective: it is mapped to feature indices and applied when the
//! ply's accumulator is first needed. Pawn pairs come from the pawn bitboards before
//! and after the move.

use crate::bitboard::{
    BISHOP_DELTAS, ROOK_DELTAS, attackers_to, init_king_attacks, init_knight_attacks,
    init_pawn_attacks, pop_lsb, sliding_attack_otf,
};
use crate::position::Position;
use crate::types::{ArrayBuf, Color, Piece, Square};

use super::kernels::dispatch::threat_batch;
use super::move_threats::{self, MoveThreats};
use super::network::{self, Aligned, NNUEParams};
use super::pawn_pairs;
use super::{L1_SIZE, THREAT_FEATURE_COUNT, THREAT_OFFSET};

// ============================================================
// Constants
// ============================================================

/// Total number of pairwise threat features (not including pawn pairs).
pub const FEATURE_COUNT: usize = THREAT_FEATURE_COUNT;

/// PIECE_INTERACTION_MAP[attacker_type][defender_type]: slot among the
/// defender *types* this attacker cares about, or -1 (excluded).
/// Colour of the defender is a separate dimension (`PIECE_TARGET_COUNT / 2`).
/// Kings never participate. Pawn→pawn is excluded (those go to pawn-pairs).
#[rustfmt::skip]
const PIECE_INTERACTION_MAP: [[i8; 6]; 6] = [
    [-1,  0, -1,  1, -1, -1],  // Pawn   → N, R
    [ 0,  1,  2,  3,  4, -1],  // Knight → P,N,B,R,Q
    [ 0,  1,  2,  3, -1, -1],  // Bishop → P,N,B,R
    [ 0,  1,  2,  3, -1, -1],  // Rook   → P,N,B,R
    [ 0,  1,  2,  3,  4, -1],  // Queen  → P,N,B,R,Q
    [-1, -1, -1, -1, -1, -1],  // King   → none
];

/// Slots per attacker type = 2 colours × (non -1 entries in the row).
const PIECE_TARGET_COUNT: [usize; 6] = [4, 10, 8, 8, 10, 0];

// ============================================================
// Lookup tables
// ============================================================

/// The two lookups of a feature index, which add without a branch: an excluded pair
/// has base `FEATURE_COUNT`, so its index lands past the end of the feature set.
struct ThreatTables {
    /// `pair_base[att_idx][def_idx][att_sq < def_sq]`: first feature of the
    /// (attacker, defender) pair, or `FEATURE_COUNT` when the pair is excluded — always
    /// for the pairs the interaction map leaves out, for kings and for the unused
    /// piece indices, and on one side of the board for the pairs of like pieces that
    /// keep a single direction.
    pair_base: [[[u32; 2]; 16]; 16],
    /// `square_index[att_type][att_sq][def_sq]`: attacks of that piece type from the
    /// squares below `att_sq`, plus the rank of `def_sq` among the empty-board attacks
    /// from `att_sq`. Neither depends on the attacker's colour, except for pawns, whose
    /// attacks go up for White and down for Black: `def_sq < att_sq` reads Black's.
    /// Eight rows so that `& 7` keeps any piece index in bounds; rows 6 and 7 stay zero.
    square_index: [[[u16; 64]; 64]; 8],
}

// Computed at compile time: no initialisation word to read on every lookup.
static THREAT_TABLES: ThreatTables = build_tables();

#[inline(always)]
fn get_tables() -> &'static ThreatTables {
    &THREAT_TABLES
}

/// The leaper attack tables, built once for the compile-time construction below.
struct Leapers {
    knight: [u64; 64],
    king: [u64; 64],
    pawn: [[u64; 64]; 2],
}

const LEAPERS: Leapers = Leapers {
    knight: init_knight_attacks(),
    king: init_king_attacks(),
    pawn: init_pawn_attacks(),
};

/// Compute empty-board attacks for a given piece index and square.
///
/// Piece index encoding: `piece_type | (color << 3)`.
/// Pawns on rank 0 (white) and rank 7 (black) return 0 (unreachable ranks).
const fn empty_board_attacks(piece_idx: usize, sq: usize) -> u64 {
    let piece_type = piece_idx & 7; // 0-5: P,N,B,R,Q,K
    let color = (piece_idx >> 3) & 1; // 0=white, 1=black
    let rank = sq / 8;
    match piece_type {
        0 => {
            // Pawn: skip rank 0 (white home rank) and rank 7 (black home rank)
            if rank == 0 || rank == 7 {
                return 0;
            }
            LEAPERS.pawn[color][sq]
        }
        1 => LEAPERS.knight[sq],
        2 => sliding_attack_otf(sq as u8, 0, &BISHOP_DELTAS),
        3 => sliding_attack_otf(sq as u8, 0, &ROOK_DELTAS),
        4 => {
            sliding_attack_otf(sq as u8, 0, &BISHOP_DELTAS)
                | sliding_attack_otf(sq as u8, 0, &ROOK_DELTAS)
        }
        5 => LEAPERS.king[sq],
        _ => 0,
    }
}

const fn build_tables() -> ThreatTables {
    let mut piece_offset = [[0i32; 64]; 14];
    let mut attack_index = [[[0u8; 64]; 64]; 14];
    let mut pair_base = [[[FEATURE_COUNT as u32; 2]; 16]; 16];

    // --- Phase 1: PIECE_OFFSET_LOOKUP and ATTACK_INDEX_LOOKUP ---

    // cumulative_piece_offset[piece_type][color] = total attacks across all squares
    let mut cumulative_piece_offset = [[0i32; 2]; 6];

    let mut att_type = 0usize;
    while att_type < 6 {
        let mut att_color = 0usize;
        while att_color < 2 {
            let att_idx = att_type | (att_color << 3);
            let mut cumulative = 0i32;
            let mut sq = 0usize;
            while sq < 64 {
                piece_offset[att_idx][sq] = cumulative;
                let attacks = empty_board_attacks(att_idx, sq);
                // Fill attack_index for this (att_idx, sq)
                let mut target = 0usize;
                while target < 64 {
                    let mask = if target > 0 { (1u64 << target) - 1 } else { 0 };
                    attack_index[att_idx][sq][target] = (attacks & mask).count_ones() as u8;
                    target += 1;
                }
                cumulative += attacks.count_ones() as i32;
                sq += 1;
            }
            cumulative_piece_offset[att_type][att_color] = cumulative;
            att_color += 1;
        }
        att_type += 1;
    }

    // Verify FEATURE_COUNT
    let mut total = 0i32;
    let mut t = 0usize;
    while t < 6 {
        let cnt = PIECE_TARGET_COUNT[t] as i32;
        total += cnt * (cumulative_piece_offset[t][0] + cumulative_piece_offset[t][1]);
        t += 1;
    }
    assert!(
        total as usize == FEATURE_COUNT,
        "threat tables: computed feature count != FEATURE_COUNT"
    );

    // --- Phase 2: PIECE_PAIR_LOOKUP with base features ---

    let mut cumulative_offset = [[0i32; 2]; 6];
    {
        let mut running = 0i32;
        // Color outer, piece inner ordering
        let mut att_color = 0usize;
        while att_color < 2 {
            let mut att_type = 0usize;
            while att_type < 6 {
                cumulative_offset[att_type][att_color] = running;
                running += PIECE_TARGET_COUNT[att_type] as i32
                    * cumulative_piece_offset[att_type][att_color];
                att_type += 1;
            }
            att_color += 1;
        }
        assert!(
            running as usize == FEATURE_COUNT,
            "threat tables: running offset != FEATURE_COUNT"
        );
    }

    let mut att_type = 0usize;
    while att_type < 6 {
        let mut def_type = 0usize;
        while def_type < 6 {
            let map = PIECE_INTERACTION_MAP[att_type][def_type];
            if map >= 0 {
                assert!((map as usize) < PIECE_TARGET_COUNT[att_type] / 2);
                let mut att_color = 0usize;
                while att_color < 2 {
                    let mut def_color = 0usize;
                    while def_color < 2 {
                        let att_idx = att_type | (att_color << 3);
                        let def_idx = def_type | (def_color << 3);

                        let slot = def_color as i32 * (PIECE_TARGET_COUNT[att_type] as i32 / 2)
                            + map as i32;
                        assert!((slot as usize) < PIECE_TARGET_COUNT[att_type]);

                        let base_feature = cumulative_offset[att_type][att_color]
                            + slot * cumulative_piece_offset[att_type][att_color];
                        assert!(base_feature >= 0 && (base_feature as usize) < FEATURE_COUNT);
                        // Like pieces keep one direction: the attacker's square above
                        // the defender's.
                        let enemy = att_color != def_color;
                        let semi_excluded = att_type == def_type && (enemy || att_type != 0);

                        pair_base[att_idx][def_idx][0] = base_feature as u32;
                        if !semi_excluded {
                            pair_base[att_idx][def_idx][1] = base_feature as u32;
                        }
                        def_color += 1;
                    }
                    att_color += 1;
                }
            }
            def_type += 1;
        }
        att_type += 1;
    }

    // --- Phase 3: SQUARE_INDEX by attacker type ---

    let mut square_index = [[[0u16; 64]; 64]; 8];
    let mut pt = 0usize;
    while pt < 6 {
        let mut from = 0usize;
        while from < 64 {
            let mut to = 0usize;
            while to < 64 {
                let att_idx = pt | if pt == 0 && to < from { 8 } else { 0 };
                let v = piece_offset[att_idx][from] + attack_index[att_idx][from][to] as i32;
                assert!(v >= 0 && v <= u16::MAX as i32);
                square_index[pt][from][to] = v as u16;
                to += 1;
            }
            from += 1;
        }
        pt += 1;
    }

    // The colour-blind table must give, for every attack of either colour, the
    // index the per-colour tables give.
    let mut att_idx = 0usize;
    while att_idx < 14 {
        if att_idx & 7 < 6 {
            let mut from = 0usize;
            while from < 64 {
                let mut targets = empty_board_attacks(att_idx, from);
                while targets != 0 {
                    let to = targets.trailing_zeros() as usize;
                    targets &= targets - 1;
                    assert!(
                        square_index[att_idx & 7][from][to] as i32
                            == piece_offset[att_idx][from] + attack_index[att_idx][from][to] as i32,
                        "threat tables: square index depends on the attacker's colour"
                    );
                }
                from += 1;
            }
        }
        att_idx += 1;
    }

    ThreatTables {
        pair_base,
        square_index,
    }
}

// ============================================================
// Feature index computation
// ============================================================

/// Compute the threat feature index for a given (attacker, defender) pair.
///
/// `pov`: 0=White, 1=Black (perspective).
/// `mirrored`: true if king is on files e-h (horizontal mirror applied).
/// `att_piece_idx`: raw piece idx (piece_type | (color << 3)), color relative to board.
/// `def_piece_idx`: same.
/// `att_sq`, `def_sq`: square indices (0-63).
///
/// Returns an index `>= FEATURE_COUNT` if the pair is excluded: the two lookups
/// always add, and callers keep the index or not by one comparison.
#[inline]
pub fn get_threat_feature(
    pov: usize,
    mirrored: bool,
    att_piece_idx: usize,
    def_piece_idx: usize,
    att_sq: usize,
    def_sq: usize,
) -> usize {
    let tables = get_tables();
    // Apply perspective flip: square flip + piece color flip
    let square_flip = (7 * mirrored as usize) ^ (56 * pov);
    let side_flip = pov << 3;

    debug_assert!(att_sq < 64 && def_sq < 64);
    debug_assert!(att_piece_idx < 16 && def_piece_idx < 16);
    // The masks keep every lookup in bounds without a check.
    let att_sq_f = (att_sq ^ square_flip) & 63;
    let def_sq_f = (def_sq ^ square_flip) & 63;
    let att_p_f = (att_piece_idx ^ side_flip) & 15;
    let def_p_f = (def_piece_idx ^ side_flip) & 15;

    let base = tables.pair_base[att_p_f][def_p_f][(att_sq_f < def_sq_f) as usize] as usize;
    let idx = base + tables.square_index[att_p_f & 7][att_sq_f][def_sq_f] as usize;
    debug_assert!(
        base == FEATURE_COUNT || idx < FEATURE_COUNT,
        "threat feature index {idx} >= FEATURE_COUNT {FEATURE_COUNT}"
    );
    idx
}

// ============================================================
// Helpers
// ============================================================

/// Convert Piece (piece_type*2+color) to threat piece index (piece_type | (color << 3)).
#[inline(always)]
fn threat_piece_idx(p: Piece) -> usize {
    (p.0 as usize >> 1) | ((p.0 as usize & 1) << 3)
}


// ============================================================
// Full threat computation (reference / standalone)
// ============================================================

/// Maximum number of active threat features per perspective.
const MAX_ACTIVE_THREATS: usize = 256;

/// Dump sorted threat feature indices for both perspectives (STM first).
/// Used by the `threats <fen>` debug command for parity testing against the
/// trainer's feature extraction (tools/trainer5).
pub fn dump_features(pos: &Position) {
    let occ = pos.occupied();
    let stm = pos.side_to_move;
    for (label, color) in [("stm", stm), ("ntm", !stm)] {
        let pov = color.index();
        let mirrored = pos.king_sq(color).file() >= 4;
        let mut feats = [0u32; MAX_ACTIVE_THREATS];
        let n = collect_features_for_pov(pos, pov, mirrored, occ, &mut feats);
        let mut v: Vec<u32> = feats[..n].to_vec();
        v.sort_unstable();
        let strs: Vec<String> = v.iter().map(|f| f.to_string()).collect();
        println!("{label}:{}", strs.join(","));
    }
    pawn_pairs::dump_features(pos);
}

/// Collect all active threat feature indices for one perspective.
fn collect_features_for_pov(
    pos: &Position,
    pov: usize,
    mirrored: bool,
    occ: u64,
    out: &mut [u32; MAX_ACTIVE_THREATS],
) -> usize {
    let mut n = 0usize;

    for def_raw in 0..12usize {
        let def_color = def_raw & 1;
        let def_type = def_raw >> 1;
        let mut def_bb = pos.pieces[def_raw];
        while def_bb != 0 {
            let def_sq_idx = pop_lsb(&mut def_bb).0 as usize;
            let def_piece_idx = def_type | (def_color << 3);

            let attackers_bb = attackers_to(Square(def_sq_idx as u8), occ, &pos.pieces);

            for att_raw in 0..12usize {
                let att_color = att_raw & 1;
                let att_type = att_raw >> 1;
                let mut att_bb = attackers_bb & pos.pieces[att_raw];
                while att_bb != 0 {
                    let att_sq_idx = pop_lsb(&mut att_bb).0 as usize;
                    let att_piece_idx = att_type | (att_color << 3);

                    let feat = get_threat_feature(
                        pov,
                        mirrored,
                        att_piece_idx,
                        def_piece_idx,
                        att_sq_idx,
                        def_sq_idx,
                    );
                    if feat < FEATURE_COUNT {
                        debug_assert!(
                            n < MAX_ACTIVE_THREATS,
                            "too many active threat features: {n} >= {MAX_ACTIVE_THREATS}"
                        );
                        out[n] = feat as u32;
                        n += 1;
                    }
                }
            }
        }
    }

    n
}

// ============================================================
// Register-batched accumulation
// ============================================================

/// Maximum dirty features per perspective (adds or subs separately).
/// Threats (~80) plus pawn-pair replace (up to 64 old + 64 new).
const MAX_DIRTY: usize = 256;
// Every recorded change writes a slot, kept or not: the record's capacity plus the
// pawn-pair delta must fit.
const _: () =
    assert!(move_threats::MOVE_THREATS_CAP + pawn_pairs::MAX_DELTA_PAIRS <= MAX_DIRTY);

/// Collected dirty feature indices for register-batched application.
///
/// The two buffers are not zeroed on construction: an update pushes a dozen
/// indices, and clearing 2 KB per perspective per node cost more than writing them.
struct DirtyFeatures {
    adds: ArrayBuf<u32, MAX_DIRTY>,
    subs: ArrayBuf<u32, MAX_DIRTY>,
    n_adds: usize,
    n_subs: usize,
}

impl DirtyFeatures {
    #[inline]
    fn new() -> Self {
        DirtyFeatures {
            adds: ArrayBuf::new(),
            subs: ArrayBuf::new(),
            n_adds: 0,
            n_subs: 0,
        }
    }

    #[inline]
    fn adds(&self) -> &[u32] {
        self.adds.filled(self.n_adds)
    }

    #[inline]
    fn subs(&self) -> &[u32] {
        self.subs.filled(self.n_subs)
    }
}

/// Dirty features of both perspectives, filled by one enumeration of the changes.
///
/// One lives in each thread's `Network` rather than on the stack: at 4 KB it would
/// push the frame of the update past a page, and every update would pay a stack
/// probe (a store on Linux and AArch64, a call to `__chkstk` on Windows).
pub(super) struct DirtyPair {
    by_pov: [DirtyFeatures; 2],
}

impl DirtyPair {
    #[inline]
    pub(super) fn new() -> Self {
        DirtyPair {
            by_pov: [DirtyFeatures::new(), DirtyFeatures::new()],
        }
    }

    /// Empty both lists of both perspectives. The slots keep their stale indices,
    /// which nothing reads past the counts.
    #[inline]
    fn clear(&mut self) {
        for d in &mut self.by_pov {
            d.n_adds = 0;
            d.n_subs = 0;
        }
    }

    /// Record one attack for both perspectives. `feats` are the raw threat indices,
    /// `>= FEATURE_COUNT` where a side has no feature for that direction: the mutual
    /// attacks of like pieces keep one direction per side, and which one depends on
    /// that side's orientation, so the two sides are decided independently.
    ///
    /// Each index is written whether it is kept or not, and only the count moves:
    /// a pair the feature set excludes is common enough that a branch on it
    /// mispredicts.
    #[inline]
    fn push_threat(&mut self, feats: [usize; 2], is_add: bool) {
        for pov in 0..2 {
            let keep = feats[pov] < FEATURE_COUNT;
            let idx = (feats[pov] + THREAT_OFFSET) as u32;
            if is_add {
                self.by_pov[pov].push_add_if(idx, keep);
            } else {
                self.by_pov[pov].push_sub_if(idx, keep);
            }
        }
    }
}

/// The threat index of one attack, for each perspective.
#[inline]
fn threat_feature_pair(
    mirrored: &[bool; 2],
    att_piece_idx: usize,
    def_piece_idx: usize,
    att_sq: usize,
    def_sq: usize,
) -> [usize; 2] {
    [
        get_threat_feature(0, mirrored[0], att_piece_idx, def_piece_idx, att_sq, def_sq),
        get_threat_feature(1, mirrored[1], att_piece_idx, def_piece_idx, att_sq, def_sq),
    ]
}

impl DirtyFeatures {
    #[inline]
    fn push_add(&mut self, feat: u32) {
        debug_assert!(self.n_adds < MAX_DIRTY);
        self.adds[self.n_adds] = feat;
        self.n_adds += 1;
    }

    #[inline]
    fn push_sub(&mut self, feat: u32) {
        debug_assert!(self.n_subs < MAX_DIRTY);
        self.subs[self.n_subs] = feat;
        self.n_subs += 1;
    }

    /// Write `feat` in the next slot, and count it only when `keep`.
    #[inline]
    fn push_add_if(&mut self, feat: u32, keep: bool) {
        debug_assert!(self.n_adds < MAX_DIRTY);
        self.adds.set(self.n_adds, feat);
        self.n_adds += keep as usize;
    }

    /// Write `feat` in the next slot, and count it only when `keep`.
    #[inline]
    fn push_sub_if(&mut self, feat: u32, keep: bool) {
        debug_assert!(self.n_subs < MAX_DIRTY);
        self.subs.set(self.n_subs, feat);
        self.n_subs += keep as usize;
    }
}

/// Compute full threat features for both perspectives (standalone, no caching).
///
/// Returns the accumulated FT values.
/// Used as reference implementation for debug_assert validation and for
/// positions without a parent threat accumulator.
pub fn compute_full_threats(pos: &Position) -> Aligned<[[i16; L1_SIZE]; 2]> {
    compute_full_threats_with(network::params(), pos)
}

/// [`compute_full_threats`] on a given copy of the network.
pub fn compute_full_threats_with(params: &NNUEParams, pos: &Position) -> Aligned<[[i16; L1_SIZE]; 2]> {
    let mut result = Aligned([[0i16; L1_SIZE]; 2]);

    let occ = pos.occupied();
    let king_sq = [pos.king_sq(Color::White), pos.king_sq(Color::Black)];

    for pov in 0..2usize {
        let king_file = king_sq[pov].file() as usize;
        let mirrored = king_file >= 4;

        let mut features = [0u32; MAX_ACTIVE_THREATS];
        let n = collect_features_for_pov(pos, pov, mirrored, occ, &mut features);
        for feat in features[..n].iter_mut() {
            *feat += THREAT_OFFSET as u32;
        }

        let mut pp = [0u32; pawn_pairs::MAX_ACTIVE_PAIRS];
        let n_pp = pawn_pairs::collect_for_pov(
            pos.pieces[Piece::WHITE_PAWN.index()],
            pos.pieces[Piece::BLACK_PAWN.index()],
            pov,
            mirrored,
            &mut pp,
        );

        let acc = result.0[pov].as_mut_ptr();
        let weights_base = params.ft_threat_weights.0.as_ptr();
        unsafe {
            threat_batch(std::ptr::null(), acc, weights_base, &features[..n], &[]);
            if n_pp > 0 {
                threat_batch(acc, acc, weights_base, &pp[..n_pp], &[]);
            }
        }
    }

    result
}

// ============================================================
// Pawn-pair cache — the pawn part of a full recompute
// ============================================================

/// The pawn-pair sums of one (perspective, mirroring) slot and the pawns behind them.
#[derive(Clone)]
struct PawnPairEntry {
    values: Aligned<[i16; L1_SIZE]>,
    /// White and black pawns the sums stand for.
    pawns: [u64; 2],
}

impl PawnPairEntry {
    /// No pawns, no pairs: the empty board is a true entry, so none needs a valid flag.
    fn new() -> Self {
        PawnPairEntry {
            values: Aligned([0i16; L1_SIZE]),
            pawns: [0; 2],
        }
    }
}

/// Per-thread cache of the pawn-pair part of the threat accumulator.
///
/// Pawn pairs depend only on the pawns and on the mirroring of the perspective's
/// king, and a full recompute (root, king crossing the centre, a chain of unknown
/// plies) mostly finds the pawns as the previous one left them, or one or two pawn
/// moves away. Each `[perspective][mirrored]` slot keeps the sums it last produced
/// and the pawns behind them, and is brought to the current pawns with the pairs
/// that changed, as the Finny table does for the piece-square block. The i16 sums
/// wrap in every backend, so the result is the from-scratch sum to the bit.
pub struct PawnPairCache {
    entries: Box<[[PawnPairEntry; 2]; 2]>,
    /// Network generation the sums were taken with (EvalFile reload detection).
    generation: u64,
}

impl PawnPairCache {
    pub fn new() -> Self {
        PawnPairCache {
            entries: Box::new(std::array::from_fn(|_| {
                std::array::from_fn(|_| PawnPairEntry::new())
            })),
            generation: network::generation(),
        }
    }

    /// Forget every sum once a new network is loaded: they were taken with the old one.
    fn check_generation(&mut self) {
        let current = network::generation();
        if self.generation != current {
            *self = PawnPairCache::new();
        }
    }

    /// The slot of `pov` with `mirrored`, its sums brought to `pawns` (white, black).
    fn bring_to(
        &mut self,
        params: &NNUEParams,
        pawns: [u64; 2],
        pov: usize,
        mirrored: bool,
    ) -> &PawnPairEntry {
        debug_assert!(pov < 2);
        let entry = &mut self.entries[pov][mirrored as usize];
        #[cfg(feature = "stats")]
        {
            let mut scratch = [0u32; pawn_pairs::MAX_ACTIVE_PAIRS];
            let n = pawn_pairs::collect_for_pov(pawns[0], pawns[1], pov, mirrored, &mut scratch);
            tally::add(tally::PAWN_ROWS_SCRATCH, n as u64);
            tally::add(tally::PAWN_HITS, (entry.pawns == pawns) as u64);
        }
        if entry.pawns == pawns {
            return entry;
        }

        // Any two pawn sets: `subs` are pairs of the cached set, `adds` of the current
        // one, so neither list can exceed the pairs a whole set has.
        let mut subs = [0u32; pawn_pairs::MAX_ACTIVE_PAIRS];
        let mut adds = [0u32; pawn_pairs::MAX_ACTIVE_PAIRS];
        let (n_subs, n_adds) = pawn_pairs::collect_delta_for_pov(
            entry.pawns[0],
            entry.pawns[1],
            pawns[0],
            pawns[1],
            pov,
            mirrored,
            &mut subs,
            &mut adds,
        );
        #[cfg(feature = "stats")]
        tally::add(tally::PAWN_ROWS, (n_subs + n_adds) as u64);

        let sums = entry.values.0.as_mut_ptr();
        unsafe {
            threat_batch(
                sums,
                sums,
                params.ft_threat_weights.0.as_ptr(),
                &adds[..n_adds],
                &subs[..n_subs],
            );
        }
        entry.pawns = pawns;
        entry
    }
}

impl Clone for PawnPairCache {
    fn clone(&self) -> Self {
        PawnPairCache {
            entries: self.entries.clone(),
            generation: self.generation,
        }
    }
}

/// Threat accumulator work, tallied per thread (feature `stats`) and drained by the
/// search into its statistics: the accumulator has no `ThreadData` to write to.
#[cfg(feature = "stats")]
pub mod tally {
    use std::cell::Cell;

    /// Full recomputes at the root.
    pub const FULL_ROOT: usize = 0;
    /// Full recomputes with no known ancestor within `THREAT_CHAIN_MAX` plies.
    pub const FULL_CHAIN: usize = 1;
    /// Full recomputes because a king crossed the centre on the way.
    pub const FULL_KING: usize = 2;
    /// Plies brought up to date from their parent.
    pub const APPLIED: usize = 3;
    /// Threat rows summed by full recomputes.
    pub const THREAT_ROWS: usize = 4;
    /// Pawn-pair rows the cache applied in full recomputes.
    pub const PAWN_ROWS: usize = 5;
    /// Pawn-pair rows a from-scratch sum would have applied instead.
    pub const PAWN_ROWS_SCRATCH: usize = 6;
    /// Perspectives whose cached pawns were already the current ones.
    pub const PAWN_HITS: usize = 7;
    /// Accumulator vectors (one perspective, 2 KB) copied unchanged or written twice,
    /// and the updates they sit among: PST vectors copied across a null move, PST
    /// vectors brought up to date from their parent, Finny refreshes (each copies its
    /// cache entry into the accumulator after updating it), threat vectors copied
    /// across a null move, threat batches with nothing to add or remove (a copy), and
    /// all threat batches.
    pub const PST_NULL_COPIES: usize = 8;
    pub const PST_UPDATES: usize = 9;
    pub const PST_REFRESHES: usize = 10;
    pub const THREAT_NULL_COPIES: usize = 11;
    pub const THREAT_EMPTY_BATCHES: usize = 12;
    pub const THREAT_BATCHES: usize = 13;
    pub const LEN: usize = 14;

    thread_local! {
        static TALLY: Cell<[u64; LEN]> = const { Cell::new([0; LEN]) };
    }

    #[inline]
    pub fn add(counter: usize, n: u64) {
        debug_assert!(counter < LEN);
        TALLY.with(|t| {
            let mut v = t.get();
            v[counter] += n;
            t.set(v);
        });
    }

    /// What this thread has tallied since the last call, reset to zero.
    pub fn take() -> [u64; LEN] {
        TALLY.with(|t| t.replace([0; LEN]))
    }
}

// ============================================================
// Threat and pawn-pair rows of the fused accumulator
// ============================================================

/// How many plies an update walks back to an accurate ancestor before it refreshes
/// instead. A full threat recompute costs about four incremental steps on the bench
/// (~1.0 µs against ~0.25 µs), so a longer chain is not worth applying.
pub(super) const THREAT_CHAIN_MAX: usize = 4;

impl DirtyPair {
    /// The rows collected for one perspective.
    #[inline]
    pub(super) fn pov(&self, pov: usize) -> (&[u32], &[u32]) {
        debug_assert!(pov < 2);
        let d = &self.by_pov[pov];
        (d.adds(), d.subs())
    }
}

/// Collect the threat and pawn-pair rows a recorded move changes, for the
/// perspectives in `povs`. `mirrored` is each side's king half, the same at the
/// recorded ply as at the position the update ends at: a crossing is a refresh, never
/// an incremental step.
///
/// One record serves both perspectives: an attack is the same fact seen from either
/// side, only its feature index depends on the side and on the mirroring of that
/// side's king. Pairs the feature set leaves out map to no index.
pub(super) fn collect_moved(
    rec: &MoveThreats,
    mirrored: [bool; 2],
    povs: [bool; 2],
    dirty: &mut DirtyPair,
) {
    dirty.clear();
    for (list, add) in [(rec.adds(), true), (rec.subs(), false)] {
        for &e in list {
            let (att, att_sq, def, def_sq) = move_threats::decode(e);
            dirty.push_threat(
                threat_feature_pair(
                    &mirrored,
                    threat_piece_idx(att),
                    threat_piece_idx(def),
                    att_sq,
                    def_sq,
                ),
                add,
            );
        }
    }

    // === Pawn pairs: only the pairs touching a changed pawn ===
    // The pairs among untouched pawns are common to the old and new sets;
    // subtracting and re-adding them would move some forty weight rows for
    // nothing, and those rows were most of what an update touched.
    let [old_wp, old_bp] = rec.pawns_before;
    let [new_wp, new_bp] = rec.pawns_after;
    if old_wp == new_wp && old_bp == new_bp {
        return;
    }
    for pov in 0..2 {
        if !povs[pov] {
            continue;
        }
        let d = &mut dirty.by_pov[pov];
        let mut pp_subs = [0u32; pawn_pairs::MAX_DELTA_PAIRS];
        let mut pp_adds = [0u32; pawn_pairs::MAX_DELTA_PAIRS];
        let (n_subs, n_adds) = pawn_pairs::collect_delta_for_pov(
            old_wp,
            old_bp,
            new_wp,
            new_bp,
            pov,
            mirrored[pov],
            &mut pp_subs,
            &mut pp_adds,
        );
        for &f in &pp_subs[..n_subs] {
            d.push_sub(f);
        }
        for &f in &pp_adds[..n_adds] {
            d.push_add(f);
        }
    }
}

/// The pawn-pair and threat part of a full refresh of `pov`, added to `acc`
/// (which holds the rest): the pawn-pair sums of that king's half as the thread's cache
/// holds them, then every active threat row.
///
/// Kept out of line: inlined, its stack buffers would weigh on the evaluation path
/// that calls it.
#[inline(never)]
pub(super) fn add_full(
    params: &NNUEParams,
    pos: &Position,
    pov: usize,
    cache: &mut PawnPairCache,
    acc: &mut [i16; L1_SIZE],
) {
    debug_assert!(pov < 2);
    cache.check_generation();
    let color = if pov == 0 { Color::White } else { Color::Black };
    let mirrored = pos.king_sq(color).file() >= 4;
    let pawns = [
        pos.pieces[Piece::WHITE_PAWN.index()],
        pos.pieces[Piece::BLACK_PAWN.index()],
    ];
    let entry = cache.bring_to(params, pawns, pov, mirrored);
    for (a, &e) in acc.iter_mut().zip(entry.values.0.iter()) {
        *a = a.wrapping_add(e);
    }

    let mut features = [0u32; MAX_ACTIVE_THREATS];
    let n = collect_features_for_pov(pos, pov, mirrored, pos.occupied(), &mut features);
    for feat in features[..n].iter_mut() {
        *feat += THREAT_OFFSET as u32;
    }
    #[cfg(feature = "stats")]
    tally::add(tally::THREAT_ROWS, n as u64);

    let p = acc.as_mut_ptr();
    unsafe { threat_batch(p, p, params.ft_threat_weights.0.as_ptr(), &features[..n], &[]) };
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::super::L1_SIZE;
    use super::*;
    use crate::position::Position;
    use crate::types::{MT_CASTLING, MT_EN_PASSANT, Move, PieceType};

    /// Interaction-map invariants: slots per attacker are unique, cover 0..types,
    /// and PIECE_TARGET_COUNT = 2 × (non -1 entries).
    #[test]
    fn test_slot_maps_invariants() {
        for att in 0..6usize {
            let types = PIECE_TARGET_COUNT[att] / 2;
            let mut seen = [false; 8];
            let mut n_mapped = 0usize;
            for def in 0..6usize {
                let slot = PIECE_INTERACTION_MAP[att][def];
                if slot >= 0 {
                    let s = slot as usize;
                    assert!(s < types, "att={att} def={def} slot={s} >= types={types}");
                    assert!(!seen[s], "att={att} slot {s} collision");
                    seen[s] = true;
                    n_mapped += 1;
                }
            }
            assert_eq!(
                n_mapped, types,
                "att={att}: mapped types {n_mapped} != {types}"
            );
            assert!(
                seen[..types].iter().all(|&s| s),
                "att={att}: slots not contiguous"
            );
        }
    }

    /// The total feature count must be exactly 59,808.
    /// (build_tables asserts this at compile time; this test also spells it out.)
    #[test]
    fn test_feature_count_59808() {
        let mut cumulative_piece_offset = [[0i32; 2]; 6];
        for att_type in 0..6usize {
            for att_color in 0..2usize {
                let att_idx = att_type | (att_color << 3);
                let mut cumulative = 0i32;
                for sq in 0..64usize {
                    cumulative += empty_board_attacks(att_idx, sq).count_ones() as i32;
                }
                cumulative_piece_offset[att_type][att_color] = cumulative;
            }
        }
        let total: i32 = (0..6)
            .map(|t| {
                let cnt = PIECE_TARGET_COUNT[t] as i32;
                (0..2)
                    .map(|c| cnt * cumulative_piece_offset[t][c])
                    .sum::<i32>()
            })
            .sum();
        assert_eq!(total as usize, FEATURE_COUNT);
        assert_eq!(FEATURE_COUNT, 59_808);
    }

    /// Interaction-map semantics: pawn-pawn and kings out, N/R/B/Q in.
    #[test]
    fn test_filter_semantics() {
        let tables = get_tables();
        let _ = tables;

        // White knight defends white rook: included (same-colour N→R is in the map)
        let feat = get_threat_feature(0, false, 1, 3, 1, 11);
        assert!(feat < FEATURE_COUNT, "N defends R must be included");

        // White knight attacks black rook: included
        let feat = get_threat_feature(0, false, 1, 3 | 8, 1, 11);
        assert!(feat < FEATURE_COUNT, "N attacks enemy R must be included");

        // White knight defends white pawn: included
        let feat = get_threat_feature(0, false, 1, 0, 1, 11);
        assert!(feat < FEATURE_COUNT, "N defends P must be included");

        // White pawn defends white knight: included. e2(12) → d3(19)
        let feat = get_threat_feature(0, false, 0, 1, 12, 19);
        assert!(feat < FEATURE_COUNT, "P defends N must be included");

        // White pawn attacks black pawn: excluded (pawn-pawn lives in pawn_pairs)
        let feat = get_threat_feature(0, false, 0, 0 | 8, 12, 19);
        assert!(feat >= FEATURE_COUNT, "P attacks p must be excluded");

        // White pawn defends white queen: excluded (not in P → {N,R})
        let feat = get_threat_feature(0, false, 0, 4, 12, 19);
        assert!(feat >= FEATURE_COUNT, "P defends Q must be excluded");

        // White bishop attacks black queen: excluded (B → {P,N,B,R} only)
        let feat = get_threat_feature(0, false, 2, 4 | 8, 2, 20);
        assert!(
            feat >= FEATURE_COUNT,
            "B attacks Q excluded from the threat feature map"
        );

        // King is never a target
        let feat = get_threat_feature(0, false, 3, 5 | 8, 0, 8);
        assert!(feat >= FEATURE_COUNT, "K never a target");

        // King never attacks
        let feat = get_threat_feature(0, false, 5, 1 | 8, 4, 12);
        assert!(feat >= FEATURE_COUNT, "K never an attacker");
    }

    /// Perspective symmetry: the feature seen from White pov for a pair (att, def)
    /// == the feature seen from Black pov for the vertically mirrored, color-swapped pair.
    #[test]
    fn test_pov_symmetry() {
        for &(att_p, def_p, att_sq, def_sq) in &[
            (1usize, 3 | 8, 1usize, 11usize), // N attacks r
            (0, 1, 12, 19),                   // P defends N
            (4 | 8, 0, 36, 12),               // q attacks P
        ] {
            for mirrored in [false, true] {
                let w = get_threat_feature(0, mirrored, att_p, def_p, att_sq, def_sq);
                // Equivalent pair: colors swapped, squares flipped vertically
                let b =
                    get_threat_feature(1, mirrored, att_p ^ 8, def_p ^ 8, att_sq ^ 56, def_sq ^ 56);
                assert_eq!(
                    w, b,
                    "pov symmetry: ({att_p},{def_p},{att_sq},{def_sq},m={mirrored})"
                );
            }
        }
    }

    /// Full enumeration on real positions: indices within bounds,
    /// reasonable active feature count (< MAX_ACTIVE_THREATS).
    #[test]
    fn test_full_enumeration_bounds() {
        for fen in [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
        ] {
            let pos = Position::from_fen(fen).unwrap();
            let occ = pos.occupied();
            for pov in 0..2usize {
                for mirrored in [false, true] {
                    let mut feats = [0u32; MAX_ACTIVE_THREATS];
                    let n = collect_features_for_pov(&pos, pov, mirrored, occ, &mut feats);
                    assert!(n < MAX_ACTIVE_THREATS);
                    for &f in &feats[..n] {
                        assert!((f as usize) < FEATURE_COUNT, "{fen}: feature {f} OOB");
                    }
                }
            }
        }
    }

    // ============================================================
    // Incremental validation with REAL weights (random network)
    // ============================================================

    /// Generate a structured (deterministic) random network and load it globally.
    fn load_random_network() {
        network::load_from_bytes(&random_network_bytes()).unwrap();
    }

    /// The file bytes of a structured (deterministic) random network.
    fn random_network_bytes() -> Vec<u8> {
        network::random_network_bytes(0x9E3779B97F4A7C15)
    }

    /// The K32Q2 block changes, with non-zero random weights: a king step within its
    /// half (from an accurate parent, and at the end of a lazy chain), the last
    /// opponent queen taken by a piece and by the king, a first queen promoted, a
    /// queen taken while another remains. After every move both perspectives equal
    /// the sums from scratch.
    ///
    /// IGNORED by default (loads a global network, interferes with the eval tests).
    /// Run alone: cargo test k32q2_block_changes_random_net -- --ignored
    #[test]
    #[ignore]
    fn k32q2_block_changes_random_net() {
        load_random_network();

        let play = |fen: &str, moves: &[Move], eager: bool, label: &str| {
            let mut pos = Position::from_fen(fen).unwrap();
            let mut net = super::super::Network::new();
            net.refresh(&pos);
            for (i, &mv) in moves.iter().enumerate() {
                net.push(mv, &pos);
                pos.make_move(mv);
                if eager || i + 1 == moves.len() {
                    net.ensure_all_updated(&pos);
                    net.assert_matches_reference(&pos, &format!("{label}, move {}", i + 1));
                }
            }
        };

        for eager in [true, false] {
            // King steps on its half, both sides, then a pawn move.
            play(
                "r3k2r/ppp2ppp/8/3q4/3Q4/8/PPP2PPP/R4K1R w - - 0 1",
                &[
                    Move::new(Square::F1, Square::G1),
                    Move::new(Square::E8, Square::F8),
                    Move::new(Square::G1, Square::F1),
                    Move::new(Square::F8, Square::G8),
                    Move::new(Square::H2, Square::H3),
                ],
                eager,
                "king steps",
            );
            // The last black queen taken by a queen, recaptured by the king: both
            // queens gone, a king move on top.
            play(
                "4k3/3q4/8/8/8/8/3Q4/4K3 w - - 0 1",
                &[Move::new(Square::D2, Square::D7), Move::new(Square::E8, Square::D7)],
                eager,
                "queens off",
            );
            // A first queen promoted, then a second (no flip), one taken while the
            // other remains, the rook taken back.
            play(
                "6r1/4P2P/k7/8/8/8/8/K7 w - - 0 1",
                &[
                    Move::new_promotion(Square::E7, Square::E8, PieceType::Queen),
                    Move::new(Square::A6, Square::B6),
                    Move::new_promotion(Square::H7, Square::H8, PieceType::Queen),
                    Move::new(Square::G8, Square::E8),
                    Move::new(Square::H8, Square::E8),
                ],
                eager,
                "promotions",
            );
            // A promotion that takes the last enemy queen while promoting the first own.
            play(
                "3q3k/4P3/8/8/8/8/8/K7 w - - 0 1",
                &[Move::new_promotion(Square::E7, Square::D8, PieceType::Queen)],
                eager,
                "promotion takes the queen",
            );
        }
    }

    /// Strong validation: with non-zero random weights, the incremental threat
    /// update == full recompute after every move of a sequence covering
    /// captures, EP, castling, promotion, null move, and a king crossing the median.
    ///
    /// IGNORED by default (loads a global network, interferes with the eval tests).
    /// Run alone: cargo test test_threats_incremental_random_net -- --ignored
    #[test]
    #[ignore]
    fn test_threats_incremental_random_net() {
        load_random_network();

        // Rich sequence from kiwipete (captures, castling)
        let mut pos = Position::from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        )
        .unwrap();
        let mut net = super::super::Network::new();
        net.refresh(&pos);

        let moves = [
            Move::new_with_type(Square::E1, Square::H1, MT_CASTLING), // O-O
            Move::new(Square::B6, Square::D5),                        // Nxd5 capture
            Move::new(Square::C3, Square::D5),                        // Nxd5 recapture
            Move::new(Square::E7, Square::D8),                        // Qd8 (queen moves)
            Move::new(Square::F3, Square::H3),                        // Qxh3 capture
        ];

        for (i, &mv) in moves.iter().enumerate() {
            net.push(mv, &pos);
            pos.make_move(mv);
            net.ensure_all_updated(&pos);
            net.assert_matches_reference(&pos, &format!("move {} ({:?})", i + 1, mv));
        }

        // Promotion + EP from a dedicated position
        let mut pos = Position::from_fen("3r3k/4P3/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let mut net = super::super::Network::new();
        net.refresh(&pos);
        let mv = Move::new_promotion(Square::E7, Square::D8, PieceType::Queen);
        net.push(mv, &pos);
        pos.make_move(mv);
        net.ensure_all_updated(&pos);
        net.assert_matches_reference(&pos, "promo-capture");

        // A reachable structure with 72 active pawn pairs per perspective, more than
        // the full recompute used to have room for, then a pawn move from it.
        let mut pos =
            Position::from_fen("4k3/2pp3p/2pp3p/2pp4/2PP4/2PP3P/2PP3P/4K3 w - - 0 1").unwrap();
        let mut net = super::super::Network::new();
        net.refresh(&pos);
        let mv = Move::new(Square::H3, Square::H4);
        net.push(mv, &pos);
        pos.make_move(mv);
        net.ensure_all_updated(&pos);
        net.assert_matches_reference(&pos, "crowded pawns");

        let mut pos =
            Position::from_fen("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3")
                .unwrap();
        let mut net = super::super::Network::new();
        net.refresh(&pos);
        let mv = Move::new_with_type(Square::E5, Square::D6, MT_EN_PASSANT);
        net.push(mv, &pos);
        pos.make_move(mv);
        net.ensure_all_updated(&pos);
        net.assert_matches_reference(&pos, "en passant");
    }

    /// One cache through structures far apart, back to an earlier one, both mirrorings
    /// and no pawns at all: the threat part of every full refresh must be the sum from
    /// scratch.
    ///
    /// On a private copy of the random network: loaded globally, it reached the searches
    /// of every other test of the process, and a search started without a network ran
    /// its NNUE pops without the pushes (#228).
    #[test]
    fn test_pawn_pair_cache_matches_scratch() {
        let net = network::params_from_bytes(&random_network_bytes());
        let params = &*net;
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "4k3/2pp3p/2pp3p/2pp4/2PP4/2PP3P/2PP3P/4K3 w - - 0 1",
            "8/8/3k4/8/8/4K3/8/8 w - - 0 1",
            "8/5pk1/6p1/7p/7P/6P1/5PK1/8 b - - 0 1",
            "8/1k3pp1/6p1/7p/7P/6P1/5PK1/8 b - - 0 1",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "2k5/pp6/8/8/8/8/PP4PP/5K2 w - - 0 1",
        ];
        let mut cache = PawnPairCache::new();
        for fen in fens {
            let pos = Position::from_fen(fen).unwrap();
            let full = compute_full_threats_with(params, &pos);
            for pov in 0..2 {
                let mut acc = [0i16; L1_SIZE];
                add_full(params, &pos, pov, &mut cache, &mut acc);
                assert!(acc[..] == full.0[pov][..], "{fen}: pawn-pair cache values, pov={pov}");
            }
        }
    }
}
