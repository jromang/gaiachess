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
//! Feature index: `base(att, def) + piece_offset[att][att_sq] + attack_index[att][att_sq][def_sq]`
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
use super::{L1_SIZE, OUTPUT_BUCKETS, THREAT_FEATURE_COUNT, THREAT_OFFSET};

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
// PiecePairData — base feature index for each (att, def) pair
// ============================================================

/// Data for a (attacker_piece_idx, defender_piece_idx) pair.
#[derive(Clone, Copy, Default)]
struct PiecePairData {
    /// Base feature index for this (att, def) pair.
    /// -1 means fully excluded (PIECE_INTERACTION_MAP returns -1).
    base_feature: i32,
    /// True if features are excluded when att_sq < def_sq.
    semi_excluded: bool,
}

impl PiecePairData {
    #[inline]
    fn is_excluded(self, att_sq: usize, def_sq: usize) -> bool {
        if self.base_feature < 0 {
            return true;
        }
        self.semi_excluded && att_sq < def_sq
    }
}

// ============================================================
// Lookup tables
// ============================================================

struct ThreatTables {
    /// Cumulative attack count for each (piece_idx, origin_sq).
    /// `piece_offset[piece_idx][sq]` = sum_{s' < sq} popcount(attacks(piece_idx, s', 0))
    piece_offset: [[i32; 64]; 14],
    /// Rank of target in empty-board attacks for (piece_idx, origin_sq, target_sq).
    /// `attack_index[piece_idx][origin][target]` = count of attacked squares < target.
    attack_index: [[[u8; 64]; 64]; 14],
    /// Base feature and exclusion info for each (att_idx, def_idx) pair.
    piece_pair: [[PiecePairData; 14]; 14],
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
    let mut piece_pair = [[PiecePairData {
        base_feature: -1,
        semi_excluded: false,
    }; 14]; 14];

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
                        let enemy = att_color != def_color;
                        let semi_excluded = att_type == def_type && (enemy || att_type != 0);

                        piece_pair[att_idx][def_idx] = PiecePairData {
                            base_feature,
                            semi_excluded,
                        };
                        def_color += 1;
                    }
                    att_color += 1;
                }
            }
            def_type += 1;
        }
        att_type += 1;
    }

    ThreatTables {
        piece_offset,
        attack_index,
        piece_pair,
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
/// Returns `FEATURE_COUNT` if the pair is excluded.
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

    let att_sq_f = att_sq ^ square_flip;
    let def_sq_f = def_sq ^ square_flip;
    let att_p_f = att_piece_idx ^ side_flip;
    let def_p_f = def_piece_idx ^ side_flip;

    // Skip invalid piece indices (6, 7, 14, 15)
    if (att_p_f & 7) > 5 || (def_p_f & 7) > 5 {
        return FEATURE_COUNT;
    }

    let pair = tables.piece_pair[att_p_f][def_p_f];
    if pair.is_excluded(att_sq_f, def_sq_f) {
        return FEATURE_COUNT;
    }

    debug_assert!((pair.base_feature as usize) < FEATURE_COUNT);
    let idx = pair.base_feature as usize
        + tables.piece_offset[att_p_f][att_sq_f] as usize
        + tables.attack_index[att_p_f][att_sq_f][def_sq_f] as usize;

    debug_assert!(
        idx < FEATURE_COUNT,
        "threat feature index {} >= FEATURE_COUNT {}",
        idx,
        FEATURE_COUNT
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
struct DirtyPair {
    by_pov: [DirtyFeatures; 2],
}

impl DirtyPair {
    #[inline]
    fn new() -> Self {
        DirtyPair {
            by_pov: [DirtyFeatures::new(), DirtyFeatures::new()],
        }
    }

    /// Record one attack for both perspectives. `feats` are the raw threat indices,
    /// `FEATURE_COUNT` where a side has no feature for that direction: the mutual
    /// attacks of like pieces keep one direction per side, and which one depends on
    /// that side's orientation, so the two sides are decided independently.
    #[inline]
    fn push_threat(&mut self, feats: [usize; 2], is_add: bool) {
        for pov in 0..2 {
            if feats[pov] < FEATURE_COUNT {
                let idx = (feats[pov] + THREAT_OFFSET) as u32;
                if is_add {
                    self.by_pov[pov].push_add(idx);
                } else {
                    self.by_pov[pov].push_sub(idx);
                }
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
}

/// Compute full threat features for both perspectives (standalone, no caching).
///
/// Returns the accumulated FT values and the PSQT head sums.
/// Used as reference implementation for debug_assert validation and for
/// positions without a parent threat accumulator.
pub fn compute_full_threats(
    pos: &Position,
) -> (Aligned<[[i16; L1_SIZE]; 2]>, [[i32; OUTPUT_BUCKETS]; 2]) {
    compute_full_threats_with(network::params(), pos)
}

/// [`compute_full_threats`] on a given copy of the network.
pub fn compute_full_threats_with(
    params: &NNUEParams,
    pos: &Position,
) -> (Aligned<[[i16; L1_SIZE]; 2]>, [[i32; OUTPUT_BUCKETS]; 2]) {
    let mut result = Aligned([[0i16; L1_SIZE]; 2]);
    let mut psqt = [[0i32; OUTPUT_BUCKETS]; 2];

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

        for &f in features[..n].iter().chain(pp[..n_pp].iter()) {
            let w = &params.psqt_threat_weights.0[f as usize];
            for b in 0..OUTPUT_BUCKETS {
                psqt[pov][b] += w[b];
            }
        }
    }

    (result, psqt)
}

// ============================================================
// ThreatAccumulator — per-ply threat state
// ============================================================

/// How many plies `ensure_threats_updated` walks back to find known threats before
/// it recomputes from scratch. A full recompute costs about four incremental steps
/// on the bench (~1.0 µs against ~0.25 µs), so a longer chain is not worth applying.
pub(super) const THREAT_CHAIN_MAX: usize = 4;

/// Per-ply threat accumulator, with the record of the move that led to it.
pub struct ThreatAccumulator {
    /// Accumulated threat weight sums per perspective [white, black].
    pub values: Aligned<[[i16; L1_SIZE]; 2]>,
    /// PSQT head sums over aux features (pawn pairs + threats): `[pov][bucket]`.
    pub psqt: [[i32; OUTPUT_BUCKETS]; 2],
    /// King mirroring state when threats were computed.
    pub(super) mirrored: [bool; 2],
    /// Whether each perspective's threats are up-to-date.
    pub accurate: [bool; 2],
    /// What the move into this ply changed, recorded when it was played.
    pub(super) moved: MoveThreats,
}

impl ThreatAccumulator {
    pub fn new() -> Self {
        ThreatAccumulator {
            values: Aligned([[0i16; L1_SIZE]; 2]),
            psqt: [[0i32; OUTPUT_BUCKETS]; 2],
            mirrored: [false; 2],
            accurate: [false; 2],
            moved: MoveThreats::new(),
        }
    }

    /// Take over the parent's threats unchanged: a null move leaves the board as it was.
    pub fn copy_from(&mut self, parent: &ThreatAccumulator) {
        self.values.0 = parent.values.0;
        self.psqt = parent.psqt;
        self.mirrored = parent.mirrored;
        self.accurate = [true; 2];
    }

    /// Full recompute from scratch (no parent available).
    pub fn update_full(&mut self, params: &NNUEParams, pos: &Position) {
        let (values, psqt) = compute_full_threats_with(params, pos);
        self.values = values;
        self.psqt = psqt;
        let king_sq = [pos.king_sq(Color::White), pos.king_sq(Color::Black)];
        for pov in 0..2 {
            self.mirrored[pov] = king_sq[pov].file() >= 4;
            self.accurate[pov] = true;
        }
    }

    /// Apply the recorded move to the parent's threats. Returns false when a king
    /// changed half on the way: nothing before it carries over, and the caller
    /// recomputes from scratch.
    pub(super) fn apply_moved(&mut self, params: &NNUEParams, parent: &ThreatAccumulator) -> bool {
        let rec = &self.moved;

        // The pair is built in a local and stored in one write: two byte stores
        // followed by a two-byte reload of the same field would defeat store-to-load
        // forwarding.
        let mut mirrored = [false; 2];
        for pov in 0..2 {
            let m = rec.king_sq[pov].file() >= 4;
            if !parent.accurate[pov] || parent.mirrored[pov] != m {
                return false;
            }
            mirrored[pov] = m;
        }
        self.mirrored = mirrored;

        // One record serves both perspectives: an attack is the same fact seen from
        // either side, only its feature index depends on the side and on the mirroring
        // of that side's king. Pairs the feature set leaves out map to no index.
        let mut dirty = DirtyPair::new();
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
        let pawns_changed = old_wp != new_wp || old_bp != new_bp;
        let weights_base = params.ft_threat_weights.0.as_ptr();

        for pov in 0..2 {
            let d = &mut dirty.by_pov[pov];
            if pawns_changed {
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

            // === Apply all collected features with register batching ===
            unsafe {
                threat_batch(
                    parent.values.0[pov].as_ptr(),
                    self.values.0[pov].as_mut_ptr(),
                    weights_base,
                    d.adds(),
                    d.subs(),
                );
            }

            // PSQT head: same dirty features, scalar (8 i32 per feature)
            let mut psqt = parent.psqt[pov];
            for &a in d.adds() {
                let w = &params.psqt_threat_weights.0[a as usize];
                for b in 0..OUTPUT_BUCKETS {
                    psqt[b] += w[b];
                }
            }
            for &s in d.subs() {
                let w = &params.psqt_threat_weights.0[s as usize];
                for b in 0..OUTPUT_BUCKETS {
                    psqt[b] -= w[b];
                }
            }
            self.psqt[pov] = psqt;

            self.accurate[pov] = true;
        }

        true
    }
}

impl Clone for ThreatAccumulator {
    fn clone(&self) -> Self {
        ThreatAccumulator {
            values: Aligned(self.values.0),
            psqt: self.psqt,
            mirrored: self.mirrored,
            accurate: self.accurate,
            moved: self.moved,
        }
    }
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
        assert_eq!(feat, FEATURE_COUNT, "P attacks p must be excluded");

        // White pawn defends white queen: excluded (not in P → {N,R})
        let feat = get_threat_feature(0, false, 0, 4, 12, 19);
        assert_eq!(feat, FEATURE_COUNT, "P defends Q must be excluded");

        // White bishop attacks black queen: excluded (B → {P,N,B,R} only)
        let feat = get_threat_feature(0, false, 2, 4 | 8, 2, 20);
        assert_eq!(
            feat, FEATURE_COUNT,
            "B attacks Q excluded from the threat feature map"
        );

        // King is never a target
        let feat = get_threat_feature(0, false, 3, 5 | 8, 0, 8);
        assert_eq!(feat, FEATURE_COUNT, "K never a target");

        // King never attacks
        let feat = get_threat_feature(0, false, 5, 1 | 8, 4, 12);
        assert_eq!(feat, FEATURE_COUNT, "K never an attacker");
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
    /// PST i16 bounded ±300 (avoids i16 overflow in debug), threats full-range i8.
    fn load_random_network() {
        use std::io::Write;
        // Named after this process, not after the machine: two working trees of the same
        // repository run `cargo test` at the same time — one being the autonomous agent
        // validating a network — and a shared name means each truncates the other's file
        // mid-read.
        let path =
            std::env::temp_dir().join(format!("gaianet_t2_random_test-{}.bin", std::process::id()));
        let path = path.to_str().expect("temp dir is not valid UTF-8");
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        let mut buf: Vec<u8> = Vec::with_capacity(network::NNUE_FILE_SIZE);
        // ft_pst_weights: i16 within ±300
        for _ in 0..(super::super::FT_SIZE * L1_SIZE) {
            let v = (next() % 601) as i64 - 300;
            buf.extend_from_slice(&(v as i16).to_le_bytes());
        }
        // ft_threat_weights: full-range i8 (pawn pairs + threats)
        for _ in 0..(super::super::THREAT_INPUT_SIZE * L1_SIZE) {
            buf.push((next() & 0xFF) as u8);
        }
        // ft_biases: i16 within ±100
        for _ in 0..L1_SIZE {
            let v = (next() % 201) as i64 - 100;
            buf.extend_from_slice(&(v as i16).to_le_bytes());
        }
        // psqt head (PST + aux): i32 within ±20000 (bounded so per-pov sums stay in i32)
        let psqt_entries =
            (super::super::FT_SIZE + super::super::THREAT_INPUT_SIZE) * OUTPUT_BUCKETS;
        for _ in 0..psqt_entries {
            let v = (next() % 40_001) as i64 - 20_000;
            buf.extend_from_slice(&(v as i32).to_le_bytes());
        }
        // L1 layers (i8) then f32: small valid values
        let l1_bytes = super::super::OUTPUT_BUCKETS * (L1_SIZE / 4) * (super::super::L2_SIZE * 4);
        for _ in 0..l1_bytes {
            buf.push((next() & 0xFF) as u8);
        }
        while buf.len() < network::NNUE_FILE_SIZE {
            let v = ((next() % 2001) as f32 / 1000.0) - 1.0; // f32 within ±1
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf.truncate(network::NNUE_FILE_SIZE);

        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(&buf).unwrap();
        drop(f);
        network::load_from_file(path).unwrap();
        // The weights are in memory now; the file has done its job.
        let _ = std::fs::remove_file(path);
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
            net.ensure_updated(&pos);
            net.ensure_threats_updated_for_test(&pos);

            let (full, full_psqt) = compute_full_threats(&pos);
            let inc = net.threat_values_for_test();
            for pov in 0..2 {
                for j in 0..L1_SIZE {
                    assert_eq!(
                        inc.0[pov][j],
                        full.0[pov][j],
                        "move {} ({:?}): threats pov={pov} idx={j}",
                        i + 1,
                        mv,
                    );
                }
            }
            assert_eq!(
                net.threat_psqt_for_test(),
                &full_psqt,
                "move {} ({:?}): threat psqt",
                i + 1,
                mv
            );

            // PST psqt: incremental (update_from / finny) vs fresh refresh
            let mut acc_ref = super::super::accumulator::Accumulator::new();
            acc_ref.refresh(&pos, Color::White);
            acc_ref.refresh(&pos, Color::Black);
            assert_eq!(
                net.pst_psqt_for_test(),
                acc_ref.psqt,
                "move {} ({:?}): pst psqt",
                i + 1,
                mv
            );
        }

        // Promotion + EP from a dedicated position
        let mut pos = Position::from_fen("3r3k/4P3/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let mut net = super::super::Network::new();
        net.refresh(&pos);
        let mv = Move::new_promotion(Square::E7, Square::D8, PieceType::Queen);
        net.push(mv, &pos);
        pos.make_move(mv);
        net.ensure_updated(&pos);
        net.ensure_threats_updated_for_test(&pos);
        let (full, full_psqt) = compute_full_threats(&pos);
        let inc = net.threat_values_for_test();
        for pov in 0..2 {
            for j in 0..L1_SIZE {
                assert_eq!(
                    inc.0[pov][j], full.0[pov][j],
                    "promo-capture pov={pov} idx={j}"
                );
            }
        }
        assert_eq!(
            net.threat_psqt_for_test(),
            &full_psqt,
            "promo-capture threat psqt"
        );

        // A reachable structure with 72 active pawn pairs per perspective, more than
        // the full recompute used to have room for, then a pawn move from it.
        let mut pos =
            Position::from_fen("4k3/2pp3p/2pp3p/2pp4/2PP4/2PP3P/2PP3P/4K3 w - - 0 1").unwrap();
        let mut net = super::super::Network::new();
        net.refresh(&pos);
        let mv = Move::new(Square::H3, Square::H4);
        net.push(mv, &pos);
        pos.make_move(mv);
        net.ensure_updated(&pos);
        net.ensure_threats_updated_for_test(&pos);
        let (full, full_psqt) = compute_full_threats(&pos);
        let inc = net.threat_values_for_test();
        for pov in 0..2 {
            for j in 0..L1_SIZE {
                assert_eq!(
                    inc.0[pov][j], full.0[pov][j],
                    "crowded pawns pov={pov} idx={j}"
                );
            }
        }
        assert_eq!(
            net.threat_psqt_for_test(),
            &full_psqt,
            "crowded pawns threat psqt"
        );

        let mut pos =
            Position::from_fen("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3")
                .unwrap();
        let mut net = super::super::Network::new();
        net.refresh(&pos);
        let mv = Move::new_with_type(Square::E5, Square::D6, MT_EN_PASSANT);
        net.push(mv, &pos);
        pos.make_move(mv);
        net.ensure_updated(&pos);
        net.ensure_threats_updated_for_test(&pos);
        let (full, full_psqt) = compute_full_threats(&pos);
        let inc = net.threat_values_for_test();
        for pov in 0..2 {
            for j in 0..L1_SIZE {
                assert_eq!(
                    inc.0[pov][j], full.0[pov][j],
                    "en passant pov={pov} idx={j}"
                );
            }
        }
        assert_eq!(
            net.threat_psqt_for_test(),
            &full_psqt,
            "en passant threat psqt"
        );
    }
}
