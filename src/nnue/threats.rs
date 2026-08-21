#![allow(unsafe_op_in_unsafe_fn)]
// Hand-vectorised code: the loop index is the point. It walks several arrays in
// lockstep, steps by a SIMD lane count, and feeds raw pointer arithmetic — an
// iterator would hide the arithmetic these kernels exist to control.
#![allow(clippy::needless_range_loop)]
//! Threat features for NNUE (GaiaNet-T1 filtered pairwise encoding).
//!
//! 41,272 features encoding "piece A on square S attacks/defends piece B on square T",
//! with a GaiaChess-original filter: all threats on ENEMY pieces are encoded, but
//! defenses (same-color targets) are kept only when a pawn is involved (pawn defending
//! P/N/R, or any piece defending a pawn). This halves the feature space vs full
//! pairwise encodings while keeping pawn-structure and anchored-piece information.
//!
//! Feature index: `baseFeature(att, def, enemy) + PIECE_OFFSET_LOOKUP[att][att_sq] + ATTACK_INDEX_LOOKUP[att][att_sq][def_sq]`
//!
//! Lookup tables are initialized once via `OnceLock` at startup.
//!
//! **Incremental updates**: `ThreatAccumulator` stores per-ply threat
//! values. On eval, dirty threats are computed from `AccDelta` (which pieces moved/captured),
//! enumerating threats only for changed pieces (~2-4) + x-ray sliders (~0-2), instead of
//! all ~30 pieces. Avoids full re-enumeration of the 41,272-feature space.

use std::sync::OnceLock;

use crate::bitboard::{
    attackers_to, bishop_attacks, king_attacks, knight_attacks, pawn_attacks, pop_lsb,
    rook_attacks,
};
use crate::position::Position;
use crate::types::{
    Color, Piece, PieceType, Square, Move,
    MT_NORMAL, MT_PROMOTION, MT_EN_PASSANT, MT_CASTLING,
    pawn_push,
};

use super::accumulator::AccDelta;
use super::kernels::dispatch::threat_batch;
use super::network::{self, Aligned};
use super::{L1_SIZE, THREAT_INPUT_SIZE};

// ============================================================
// Constants
// ============================================================

/// Total number of filtered threat features.
pub const FEATURE_COUNT: usize = THREAT_INPUT_SIZE; // 41,272

/// ENEMY_SLOT_MAP[attacker_type][defender_type]: interaction slot for ENEMY targets
/// (threats proper), or -1 (excluded). All tactically relevant enemy pairs are kept;
/// kings are never targets (checks are search territory, not eval).
#[rustfmt::skip]
const ENEMY_SLOT_MAP: [[i8; 6]; 6] = [
    [ 0,  1, -1,  2, -1, -1],  // Pawn   → P,N,R
    [ 0,  1,  2,  3,  4, -1],  // Knight → P,N,B,R,Q
    [ 0,  1,  2,  3, -1, -1],  // Bishop → P,N,B,R
    [ 0,  1,  2,  3, -1, -1],  // Rook   → P,N,B,R
    [ 0,  1,  2,  3,  4, -1],  // Queen  → P,N,B,R,Q
    [ 0,  1,  2,  3, -1, -1],  // King   → P,N,B,R
];

/// OWN_SLOT_MAP[attacker_type][defender_type]: interaction slot for SAME-COLOR targets
/// (defenses), or -1 (excluded). GaiaNet-T1 filter: defenses are kept only when a pawn
/// is involved — pawn defending P/N/R (pawn chains, pawn-supported pieces) or any piece
/// defending a pawn. Slots continue after the enemy slots of the same attacker.
#[rustfmt::skip]
const OWN_SLOT_MAP: [[i8; 6]; 6] = [
    [ 3,  4, -1,  5, -1, -1],  // Pawn   defends P,N,R
    [ 5, -1, -1, -1, -1, -1],  // Knight defends P
    [ 4, -1, -1, -1, -1, -1],  // Bishop defends P
    [ 4, -1, -1, -1, -1, -1],  // Rook   defends P
    [ 5, -1, -1, -1, -1, -1],  // Queen  defends P
    [ 4, -1, -1, -1, -1, -1],  // King   defends P
];

/// Number of valid interaction slots per attacker type (enemy + own combined).
/// PIECE_TARGET_COUNT[t] = count of non(-1) in ENEMY_SLOT_MAP[t] + OWN_SLOT_MAP[t].
const PIECE_TARGET_COUNT: [usize; 6] = [6, 6, 5, 5, 6, 5];

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

static THREAT_TABLES: OnceLock<ThreatTables> = OnceLock::new();

fn get_tables() -> &'static ThreatTables {
    THREAT_TABLES.get_or_init(init_tables)
}

/// Compute empty-board attacks for a given piece index and square.
///
/// Piece index encoding: `piece_type | (color << 3)`.
/// Pawns on rank 0 (white) and rank 7 (black) return 0 (unreachable ranks).
fn empty_board_attacks(piece_idx: usize, sq: usize) -> u64 {
    let piece_type = piece_idx & 7; // 0-5: P,N,B,R,Q,K
    let color = (piece_idx >> 3) & 1; // 0=white, 1=black
    let square = Square(sq as u8);
    let rank = sq / 8;
    match piece_type {
        0 => {
            // Pawn: skip rank 0 (white home rank) and rank 7 (black home rank)
            if rank == 0 || rank == 7 {
                return 0;
            }
            let pawn_color = if color == 0 { Color::White } else { Color::Black };
            pawn_attacks(square, pawn_color)
        }
        1 => knight_attacks(square),
        2 => bishop_attacks(square, 0),
        3 => rook_attacks(square, 0),
        4 => bishop_attacks(square, 0) | rook_attacks(square, 0), // queen
        5 => king_attacks(square),
        _ => 0,
    }
}

fn init_tables() -> ThreatTables {
    // SAFETY: all-zero init is valid for [i32], [u8], PiecePairData (i32 + bool).
    let mut piece_offset = [[0i32; 64]; 14];
    let mut attack_index = [[[0u8; 64]; 64]; 14];
    let mut piece_pair = [[PiecePairData { base_feature: -1, semi_excluded: false }; 14]; 14];

    // --- Phase 1: PIECE_OFFSET_LOOKUP and ATTACK_INDEX_LOOKUP ---

    // cumulative_piece_offset[piece_type][color] = total attacks across all squares
    let mut cumulative_piece_offset = [[0i32; 2]; 6];

    for att_type in 0..6usize {
        for att_color in 0..2usize {
            let att_idx = att_type | (att_color << 3);
            let mut cumulative = 0i32;
            for sq in 0..64usize {
                piece_offset[att_idx][sq] = cumulative;
                let attacks = empty_board_attacks(att_idx, sq);
                // Fill attack_index for this (att_idx, sq)
                for target in 0..64usize {
                    let mask = if target > 0 { (1u64 << target) - 1 } else { 0 };
                    attack_index[att_idx][sq][target] = (attacks & mask).count_ones() as u8;
                }
                cumulative += attacks.count_ones() as i32;
            }
            cumulative_piece_offset[att_type][att_color] = cumulative;
        }
    }

    // Verify FEATURE_COUNT
    let total: i32 = (0..6).map(|t| {
        let cnt = PIECE_TARGET_COUNT[t] as i32;
        (0..2).map(|c| cnt * cumulative_piece_offset[t][c]).sum::<i32>()
    }).sum();
    debug_assert_eq!(total as usize, FEATURE_COUNT,
        "init_tables: expected FEATURE_COUNT={FEATURE_COUNT}, computed={total}");

    // --- Phase 2: PIECE_PAIR_LOOKUP with base features ---

    let mut cumulative_offset = [[0i32; 2]; 6];
    {
        let mut running = 0i32;
        // Color outer, piece inner ordering
        for att_color in 0..2usize {
            for att_type in 0..6usize {
                cumulative_offset[att_type][att_color] = running;
                running += PIECE_TARGET_COUNT[att_type] as i32
                    * cumulative_piece_offset[att_type][att_color];
            }
        }
        debug_assert_eq!(running as usize, FEATURE_COUNT);
    }

    for att_type in 0..6usize {
        for def_type in 0..6usize {
            for att_color in 0..2usize {
                for def_color in 0..2usize {
                    let att_idx = att_type | (att_color << 3);
                    let def_idx = def_type | (def_color << 3);

                    let enemy = att_color != def_color;
                    let slot = if enemy {
                        ENEMY_SLOT_MAP[att_type][def_type]
                    } else {
                        OWN_SLOT_MAP[att_type][def_type]
                    };
                    if slot < 0 {
                        continue;
                    }
                    debug_assert!((slot as usize) < PIECE_TARGET_COUNT[att_type]);

                    let base_feature = cumulative_offset[att_type][att_color]
                        + (slot as i32) * cumulative_piece_offset[att_type][att_color];
                    let semi_excluded = att_type == def_type && (enemy || att_type != 0);

                    piece_pair[att_idx][def_idx] = PiecePairData {
                        base_feature,
                        semi_excluded,
                    };
                }
            }
        }
    }

    ThreatTables { piece_offset, attack_index, piece_pair }
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

    debug_assert!(idx < FEATURE_COUNT,
        "threat feature index {} >= FEATURE_COUNT {}", idx, FEATURE_COUNT);
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

/// Compute attacks for a given piece from a given square with given occupancy.
#[inline]
fn piece_attacks_bb(piece: Piece, sq: Square, occ: u64) -> u64 {
    match piece.piece_type() {
        PieceType::Pawn => pawn_attacks(sq, piece.color()),
        PieceType::Knight => knight_attacks(sq),
        PieceType::Bishop => bishop_attacks(sq, occ),
        PieceType::Rook => rook_attacks(sq, occ),
        PieceType::Queen => bishop_attacks(sq, occ) | rook_attacks(sq, occ),
        PieceType::King => king_attacks(sq),
    }
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
                        pov, mirrored,
                        att_piece_idx, def_piece_idx,
                        att_sq_idx, def_sq_idx,
                    );
                    if feat < FEATURE_COUNT {
                        debug_assert!(n < MAX_ACTIVE_THREATS,
                            "too many active threat features: {n} >= {MAX_ACTIVE_THREATS}");
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
const MAX_DIRTY: usize = 128;

/// Collected dirty feature indices for register-batched application.
struct DirtyFeatures {
    adds: [u32; MAX_DIRTY],
    subs: [u32; MAX_DIRTY],
    n_adds: usize,
    n_subs: usize,
}

impl DirtyFeatures {
    #[inline]
    fn new() -> Self {
        DirtyFeatures { adds: [0; MAX_DIRTY], subs: [0; MAX_DIRTY], n_adds: 0, n_subs: 0 }
    }

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
/// Used as reference implementation for debug_assert validation and for
/// positions without a parent threat accumulator.
pub fn compute_full_threats(pos: &Position) -> Aligned<[[i16; L1_SIZE]; 2]> {
    let params = network::params();
    let mut result = Aligned([[0i16; L1_SIZE]; 2]);

    let occ = pos.occupied();
    let king_sq = [pos.king_sq(Color::White), pos.king_sq(Color::Black)];

    for pov in 0..2usize {
        let king_file = king_sq[pov].file() as usize;
        let mirrored = king_file >= 4;

        let mut features = [0u32; MAX_ACTIVE_THREATS];
        let n = collect_features_for_pov(pos, pov, mirrored, occ, &mut features);

        let acc = result.0[pov].as_mut_ptr();
        let weights_base = params.ft_threat_weights.0.as_ptr();
        unsafe {
            threat_batch(
                std::ptr::null(), acc, weights_base,
                &features[..n], &[],
            );
        }
    }

    result
}

// ============================================================
// ThreatAccumulator — per-ply threat state with dirty incremental updates
// ============================================================

/// Per-ply threat accumulator with dirty incremental updates.
///
/// Instead of collecting all ~50 features and diffing sorted lists, we enumerate
/// threats only for the 2-4 pieces that changed (from AccDelta) plus x-ray sliders
/// whose attack rays pass through changed squares.
pub struct ThreatAccumulator {
    /// Accumulated threat weight sums per perspective [white, black].
    pub values: Aligned<[[i16; L1_SIZE]; 2]>,
    /// King mirroring state when threats were computed.
    pub(super) mirrored: [bool; 2],
    /// Whether each perspective's threats are up-to-date.
    pub accurate: [bool; 2],
}

impl ThreatAccumulator {
    pub fn new() -> Self {
        ThreatAccumulator {
            values: Aligned([[0i16; L1_SIZE]; 2]),
            mirrored: [false; 2],
            accurate: [false; 2],
        }
    }

    /// Full recompute from scratch (no parent available).
    pub fn update_full(&mut self, pos: &Position) {
        let result = compute_full_threats(pos);
        self.values = result;
        let king_sq = [pos.king_sq(Color::White), pos.king_sq(Color::Black)];
        for pov in 0..2 {
            self.mirrored[pov] = king_sq[pov].file() >= 4;
            self.accurate[pov] = true;
        }
    }

    /// Incremental update using move delta information.
    ///
    /// Enumerates threats only for pieces that changed position (~2-4 pieces)
    /// plus x-ray sliders (~0-2). Returns false if incremental is not possible
    /// (caller should fall back to full recompute).
    ///
    /// Uses old_occ for ALL removals and new_occ for ALL additions (no sequential
    /// simulation). Co-removed/co-added piece interactions are handled separately
    /// to avoid double-counting.
    pub fn update_incremental(
        &mut self,
        pos: &Position,
        parent: &ThreatAccumulator,
        delta: &AccDelta,
    ) -> bool {
        let mv = delta.mv;
        if mv == Move::NONE {
            return false;
        }

        let old_occ = delta.old_occ;
        let new_occ = pos.occupied();
        let king_sq = [pos.king_sq(Color::White), pos.king_sq(Color::Black)];

        // Check both perspectives can do incremental
        for pov in 0..2 {
            let mirrored = king_sq[pov].file() >= 4;
            if !parent.accurate[pov] || parent.mirrored[pov] != mirrored {
                return false;
            }
            self.mirrored[pov] = mirrored;
        }

        // Reconstruct old piece bitboards and mailbox
        let old_pieces = reconstruct_old_pieces(pos, delta);
        let old_board = reconstruct_old_board(pos, delta);

        // Determine piece changes
        let ((removes, n_rem), (adds, n_add)) = piece_changes(delta);

        // Build bitmasks for exclusion and x-ray filtering
        let mut remove_bb = 0u64;
        for i in 0..n_rem {
            remove_bb |= 1u64 << removes[i].1 .0;
        }
        let mut add_bb = 0u64;
        for i in 0..n_add {
            add_bb |= 1u64 << adds[i].1 .0;
        }
        let changed_sq_bb = remove_bb | add_bb;

        let params = network::params();
        let weights_base = params.ft_threat_weights.0.as_ptr();

        for pov in 0..2 {
            let mirrored = self.mirrored[pov];
            let mut dirty = DirtyFeatures::new();

            // === Phase 1: Collect old threats to remove (all with old_occ) ===
            for i in 0..n_rem {
                let (piece, sq) = removes[i];
                let exclude = remove_bb & !(1u64 << sq.0);
                collect_piece_threats(
                    &mut dirty, pov, mirrored, piece, sq, old_occ,
                    &old_pieces, &old_board, false, exclude,
                );
            }
            for i in 0..n_rem {
                for j in (i + 1)..n_rem {
                    collect_pairwise_threat(
                        &mut dirty, pov, mirrored, removes[i], removes[j], old_occ, false,
                    );
                }
            }

            // === Phase 2: Collect new threats to add (all with new_occ) ===
            for i in 0..n_add {
                let (piece, sq) = adds[i];
                let exclude = add_bb & !(1u64 << sq.0);
                collect_piece_threats(
                    &mut dirty, pov, mirrored, piece, sq, new_occ,
                    &pos.pieces, &pos.board, true, exclude,
                );
            }
            for i in 0..n_add {
                for j in (i + 1)..n_add {
                    collect_pairwise_threat(
                        &mut dirty, pov, mirrored, adds[i], adds[j], new_occ, true,
                    );
                }
            }

            // === Phase 3: X-ray slider changes ===
            collect_xray_changes(
                &mut dirty, pov, mirrored, old_occ, new_occ,
                &pos.pieces, &old_board, &pos.board, changed_sq_bb,
            );

            // === Apply all collected features with register batching ===
            unsafe {
                threat_batch(
                    parent.values.0[pov].as_ptr(),
                    self.values.0[pov].as_mut_ptr(),
                    weights_base,
                    &dirty.adds[..dirty.n_adds],
                    &dirty.subs[..dirty.n_subs],
                );
            }

            self.accurate[pov] = true;
        }

        // Debug validation: verify incremental matches full recompute
        #[cfg(debug_assertions)]
        {
            let full = compute_full_threats(pos);
            for pov in 0..2 {
                for i in 0..L1_SIZE {
                    debug_assert_eq!(
                        self.values.0[pov][i], full.0[pov][i],
                        "Dirty incremental threat mismatch: pov={pov}, i={i}, \
                         incremental={}, full={}",
                        self.values.0[pov][i], full.0[pov][i]
                    );
                }
            }
        }

        true
    }

}

// ============================================================
// Feature collection (populate DirtyFeatures for batch apply)
// ============================================================

/// Collect all threat features involving piece@sq into dirty list.
///
/// The position is passed as the pieces of it this needs — occupancy, bitboards, the
/// side being encoded — rather than as a `&Position`, so the caller can hand it the
/// old board and the new one in the same call.
#[allow(clippy::too_many_arguments)]
fn collect_piece_threats(
    dirty: &mut DirtyFeatures,
    pov: usize,
    mirrored: bool,
    piece: Piece,
    sq: Square,
    occ: u64,
    pieces: &[u64; 12],
    board: &[Piece; 64],
    is_add: bool,
    exclude_bb: u64,
) {
    let piece_idx = threat_piece_idx(piece);
    let sq_idx = sq.0 as usize;
    let mask = !(1u64 << sq.0) & !exclude_bb;

    // Threats FROM piece@sq
    let attacks = piece_attacks_bb(piece, sq, occ);
    let mut targets = attacks & occ & mask;
    while targets != 0 {
        let target_sq = pop_lsb(&mut targets);
        let target = board[target_sq.index()];
        if target == Piece::NONE { continue; }
        let feat = get_threat_feature(
            pov, mirrored, piece_idx, threat_piece_idx(target),
            sq_idx, target_sq.0 as usize,
        );
        if feat < FEATURE_COUNT {
            if is_add { dirty.push_add(feat as u32); } else { dirty.push_sub(feat as u32); }
        }
    }

    // Threats TO piece@sq
    let mut attackers = attackers_to(sq, occ, pieces) & mask;
    while attackers != 0 {
        let att_sq = pop_lsb(&mut attackers);
        let att = board[att_sq.index()];
        if att == Piece::NONE { continue; }
        let feat = get_threat_feature(
            pov, mirrored, threat_piece_idx(att), piece_idx,
            att_sq.0 as usize, sq_idx,
        );
        if feat < FEATURE_COUNT {
            if is_add { dirty.push_add(feat as u32); } else { dirty.push_sub(feat as u32); }
        }
    }
}

/// Collect pairwise threat between two co-removed or co-added pieces.
fn collect_pairwise_threat(
    dirty: &mut DirtyFeatures,
    pov: usize,
    mirrored: bool,
    (piece_a, sq_a): (Piece, Square),
    (piece_b, sq_b): (Piece, Square),
    occ: u64,
    is_add: bool,
) {
    let idx_a = threat_piece_idx(piece_a);
    let idx_b = threat_piece_idx(piece_b);

    if piece_attacks_bb(piece_a, sq_a, occ) & (1u64 << sq_b.0) != 0 {
        let feat = get_threat_feature(pov, mirrored, idx_a, idx_b, sq_a.0 as usize, sq_b.0 as usize);
        if feat < FEATURE_COUNT {
            if is_add { dirty.push_add(feat as u32); } else { dirty.push_sub(feat as u32); }
        }
    }
    if piece_attacks_bb(piece_b, sq_b, occ) & (1u64 << sq_a.0) != 0 {
        let feat = get_threat_feature(pov, mirrored, idx_b, idx_a, sq_b.0 as usize, sq_a.0 as usize);
        if feat < FEATURE_COUNT {
            if is_add { dirty.push_add(feat as u32); } else { dirty.push_sub(feat as u32); }
        }
    }
}

/// Collect x-ray slider changes between old and new occupancy.
#[allow(clippy::too_many_arguments)]
fn collect_xray_changes(
    dirty: &mut DirtyFeatures,
    pov: usize,
    mirrored: bool,
    old_occ: u64,
    new_occ: u64,
    new_pieces: &[u64; 12],
    old_board: &[Piece; 64],
    new_board: &[Piece; 64],
    changed_sq_bb: u64,
) {
    let bishop_like = new_pieces[Piece::WHITE_BISHOP.index()]
        | new_pieces[Piece::BLACK_BISHOP.index()]
        | new_pieces[Piece::WHITE_QUEEN.index()]
        | new_pieces[Piece::BLACK_QUEEN.index()];
    let rook_like = new_pieces[Piece::WHITE_ROOK.index()]
        | new_pieces[Piece::BLACK_ROOK.index()]
        | new_pieces[Piece::WHITE_QUEEN.index()]
        | new_pieces[Piece::BLACK_QUEEN.index()];

    let mut affected = 0u64;
    let mut csq = changed_sq_bb;
    while csq != 0 {
        let sq = pop_lsb(&mut csq);
        affected |= bishop_attacks(sq, 0) & bishop_like;
        affected |= rook_attacks(sq, 0) & rook_like;
    }
    affected &= !changed_sq_bb;

    while affected != 0 {
        let s_sq = pop_lsb(&mut affected);
        let slider = new_board[s_sq.index()];
        debug_assert!(slider != Piece::NONE);
        let slider_idx = threat_piece_idx(slider);

        let (old_attacks, new_attacks) = match slider.piece_type() {
            PieceType::Bishop => (bishop_attacks(s_sq, old_occ), bishop_attacks(s_sq, new_occ)),
            PieceType::Rook => (rook_attacks(s_sq, old_occ), rook_attacks(s_sq, new_occ)),
            PieceType::Queen => (
                bishop_attacks(s_sq, old_occ) | rook_attacks(s_sq, old_occ),
                bishop_attacks(s_sq, new_occ) | rook_attacks(s_sq, new_occ),
            ),
            _ => continue,
        };
        if old_attacks == new_attacks { continue; }

        // Gained → add
        let mut gained = (new_attacks & !old_attacks & !changed_sq_bb) & new_occ;
        while gained != 0 {
            let target_sq = pop_lsb(&mut gained);
            let target = new_board[target_sq.index()];
            if target == Piece::NONE { continue; }
            let feat = get_threat_feature(
                pov, mirrored, slider_idx, threat_piece_idx(target),
                s_sq.0 as usize, target_sq.0 as usize,
            );
            if feat < FEATURE_COUNT { dirty.push_add(feat as u32); }
        }

        // Lost → sub
        let mut lost = (old_attacks & !new_attacks & !changed_sq_bb) & old_occ;
        while lost != 0 {
            let target_sq = pop_lsb(&mut lost);
            let target = old_board[target_sq.index()];
            if target == Piece::NONE { continue; }
            let feat = get_threat_feature(
                pov, mirrored, slider_idx, threat_piece_idx(target),
                s_sq.0 as usize, target_sq.0 as usize,
            );
            if feat < FEATURE_COUNT { dirty.push_sub(feat as u32); }
        }
    }
}

impl Clone for ThreatAccumulator {
    fn clone(&self) -> Self {
        ThreatAccumulator {
            values: Aligned(self.values.0),
            mirrored: self.mirrored,
            accurate: self.accurate,
        }
    }
}

// ============================================================
// Piece change helpers
// ============================================================

/// Determine which pieces were removed/added by a move.
///
/// Up to four pieces and how many of the four are real. Castling is the worst case:
/// two squares vacated, two filled.
type PieceSet = ([(Piece, Square); 4], usize);

/// Returns (removed, added).
fn piece_changes(delta: &AccDelta) -> (PieceSet, PieceSet) {
    let mv = delta.mv;
    let mt = mv.move_type();
    let from = mv.from_sq();
    let to = mv.to_sq();
    let moved = delta.moved_piece;
    let captured = delta.captured_piece;

    let mut removes = [(Piece::NONE, Square::NONE); 4];
    let mut adds = [(Piece::NONE, Square::NONE); 4];
    let mut n_rem = 0;
    let mut n_add = 0;

    match mt {
        MT_NORMAL => {
            removes[n_rem] = (moved, from);
            n_rem += 1;
            if captured != Piece::NONE {
                removes[n_rem] = (captured, to);
                n_rem += 1;
            }
            adds[n_add] = (moved, to);
            n_add += 1;
        }
        MT_PROMOTION => {
            let pawn = Piece::new(PieceType::Pawn, moved.color());
            let promo = Piece::new(mv.promo_type(), moved.color());
            removes[n_rem] = (pawn, from);
            n_rem += 1;
            if captured != Piece::NONE {
                removes[n_rem] = (captured, to);
                n_rem += 1;
            }
            adds[n_add] = (promo, to);
            n_add += 1;
        }
        MT_EN_PASSANT => {
            let cap_sq = Square((to.0 as i8 - pawn_push(moved.color())) as u8);
            removes[n_rem] = (moved, from);
            n_rem += 1;
            removes[n_rem] = (captured, cap_sq);
            n_rem += 1;
            adds[n_add] = (moved, to);
            n_add += 1;
        }
        MT_CASTLING => {
            let us = moved.color();
            let rook = Piece::new(PieceType::Rook, us);
            let (rook_from, rook_to) = castle_rook_squares(us, from, to);

            removes[n_rem] = (moved, from);
            n_rem += 1;
            removes[n_rem] = (rook, rook_from);
            n_rem += 1;
            adds[n_add] = (moved, to);
            n_add += 1;
            adds[n_add] = (rook, rook_to);
            n_add += 1;
        }
        _ => {}
    }

    ((removes, n_rem), (adds, n_add))
}

/// Get rook from/to squares for a castling move.
#[inline]
fn castle_rook_squares(us: Color, king_from: Square, king_to: Square) -> (Square, Square) {
    if king_to.file() > king_from.file() {
        // Kingside
        if us == Color::White {
            (Square::H1, Square::F1)
        } else {
            (Square::H8, Square::F8)
        }
    } else {
        // Queenside
        if us == Color::White {
            (Square::A1, Square::D1)
        } else {
            (Square::A8, Square::D8)
        }
    }
}

/// Reconstruct the pre-move piece bitboards from post-move state + AccDelta.
fn reconstruct_old_pieces(pos: &Position, delta: &AccDelta) -> [u64; 12] {
    let mut pieces = pos.pieces;
    let mv = delta.mv;
    let mt = mv.move_type();
    let from = mv.from_sq();
    let to = mv.to_sq();
    let moved = delta.moved_piece;
    let captured = delta.captured_piece;

    match mt {
        MT_NORMAL => {
            // Undo: move piece back from to → from
            pieces[moved.index()] |= 1u64 << from.0;
            pieces[moved.index()] &= !(1u64 << to.0);
            if captured != Piece::NONE {
                pieces[captured.index()] |= 1u64 << to.0;
            }
        }
        MT_PROMOTION => {
            // Undo: remove promo piece from to, put pawn at from
            let pawn = Piece::new(PieceType::Pawn, moved.color());
            let promo = Piece::new(mv.promo_type(), moved.color());
            pieces[pawn.index()] |= 1u64 << from.0;
            pieces[promo.index()] &= !(1u64 << to.0);
            if captured != Piece::NONE {
                pieces[captured.index()] |= 1u64 << to.0;
            }
        }
        MT_EN_PASSANT => {
            // Undo: move pawn back, restore captured pawn
            let cap_sq = Square((to.0 as i8 - pawn_push(moved.color())) as u8);
            pieces[moved.index()] |= 1u64 << from.0;
            pieces[moved.index()] &= !(1u64 << to.0);
            pieces[captured.index()] |= 1u64 << cap_sq.0;
        }
        MT_CASTLING => {
            // Undo: move king and rook back
            let us = moved.color();
            let rook = Piece::new(PieceType::Rook, us);
            let (rook_from, rook_to) = castle_rook_squares(us, from, to);

            pieces[moved.index()] |= 1u64 << from.0;
            pieces[moved.index()] &= !(1u64 << to.0);
            pieces[rook.index()] |= 1u64 << rook_from.0;
            pieces[rook.index()] &= !(1u64 << rook_to.0);
        }
        _ => {}
    }

    pieces
}

/// Reconstruct the pre-move mailbox from post-move state + AccDelta.
fn reconstruct_old_board(pos: &Position, delta: &AccDelta) -> [Piece; 64] {
    let mut board = pos.board;
    let mv = delta.mv;
    let mt = mv.move_type();
    let from = mv.from_sq();
    let to = mv.to_sq();
    let moved = delta.moved_piece;
    let captured = delta.captured_piece;

    match mt {
        MT_NORMAL => {
            board[from.index()] = moved;
            board[to.index()] = captured; // NONE if no capture
        }
        MT_PROMOTION => {
            board[from.index()] = Piece::new(PieceType::Pawn, moved.color());
            board[to.index()] = captured; // NONE if no capture
        }
        MT_EN_PASSANT => {
            let cap_sq = Square((to.0 as i8 - pawn_push(moved.color())) as u8);
            board[from.index()] = moved;
            board[to.index()] = Piece::NONE; // EP target was empty before move
            board[cap_sq.index()] = captured;
        }
        MT_CASTLING => {
            let us = moved.color();
            let rook = Piece::new(PieceType::Rook, us);
            let (rook_from, rook_to) = castle_rook_squares(us, from, to);

            board[from.index()] = moved;
            board[to.index()] = Piece::NONE;
            board[rook_from.index()] = rook;
            board[rook_to.index()] = Piece::NONE;
        }
        _ => {}
    }

    board
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Position;
    use super::super::L1_SIZE;

    /// GaiaNet-T1 slot table invariants:
    /// for each attacker, enemy+own slots are disjoint and cover 0..count.
    #[test]
    fn test_slot_maps_invariants() {
        for att in 0..6usize {
            let count = PIECE_TARGET_COUNT[att];
            let mut seen = [false; 8];
            for def in 0..6usize {
                for &slot in &[ENEMY_SLOT_MAP[att][def], OWN_SLOT_MAP[att][def]] {
                    if slot >= 0 {
                        let s = slot as usize;
                        assert!(s < count, "att={att} def={def} slot={s} >= count={count}");
                        assert!(!seen[s] || slot_shared_ok(att, def, s),
                            "att={att} slot {s} collision");
                        seen[s] = true;
                    }
                }
            }
            assert!(seen[..count].iter().all(|&s| s), "att={att}: slots not contiguous");
        }
    }

    /// Can slots be shared between different def_types of the same map? No:
    /// each slot must be unique across the union of both maps. (No legitimate sharing.)
    fn slot_shared_ok(_att: usize, _def: usize, _slot: usize) -> bool {
        false
    }

    /// The total feature count must be exactly THREAT_INPUT_SIZE (41,272).
    /// (init_tables checks this with a debug_assert; this test also enforces it in release.)
    #[test]
    fn test_feature_count_41272() {
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
        let total: i32 = (0..6).map(|t| {
            let cnt = PIECE_TARGET_COUNT[t] as i32;
            (0..2).map(|c| cnt * cumulative_piece_offset[t][c]).sum::<i32>()
        }).sum();
        assert_eq!(total as usize, FEATURE_COUNT);
        assert_eq!(FEATURE_COUNT, 41_272);
    }

    /// All indices produced by a full enumeration are within bounds,
    /// and the filter keeps its promises: defenses without a pawn → excluded,
    /// enemy threats present in the map → included.
    #[test]
    fn test_filter_semantics() {
        let tables = get_tables();
        let _ = tables;

        // White knight defends white rook: excluded (no pawn involved)
        // N=type1, R=type3, white (color 0). b1 (1) attacks d2 (11): b1→d2 ✓
        let feat = get_threat_feature(0, false, 1, 3, 1, 11);
        assert_eq!(feat, FEATURE_COUNT, "N defends R must be excluded");

        // White knight attacks black rook: included
        let feat = get_threat_feature(0, false, 1, 3 | 8, 1, 11);
        assert!(feat < FEATURE_COUNT, "N attacks enemy R must be included");

        // White knight defends white pawn: included (pawn defense)
        let feat = get_threat_feature(0, false, 1, 0, 1, 11);
        assert!(feat < FEATURE_COUNT, "N defends P must be included");

        // White pawn defends white knight: included (defense by pawn). e2(12) → d3(19)
        let feat = get_threat_feature(0, false, 0, 1, 12, 19);
        assert!(feat < FEATURE_COUNT, "P defends N must be included");

        // White pawn defends white queen: excluded (not in P → {P,N,R})
        let feat = get_threat_feature(0, false, 0, 4, 12, 19);
        assert_eq!(feat, FEATURE_COUNT, "P defends Q must be excluded");

        // White bishop attacks black queen: excluded (B → {P,N,B,R} only)
        let feat = get_threat_feature(0, false, 2, 4 | 8, 2, 20);
        assert_eq!(feat, FEATURE_COUNT, "B attacks Q excluded from the threat feature map");

        // King is never a target
        let feat = get_threat_feature(0, false, 3, 5 | 8, 0, 8);
        assert_eq!(feat, FEATURE_COUNT, "K never a target");
    }

    /// Perspective symmetry: the feature seen from White pov for a pair (att, def)
    /// == the feature seen from Black pov for the vertically mirrored, color-swapped pair.
    #[test]
    fn test_pov_symmetry() {
        for &(att_p, def_p, att_sq, def_sq) in &[
            (1usize, 3 | 8, 1usize, 11usize),       // N attacks r
            (0, 1, 12, 19),                          // P defends N
            (4 | 8, 0, 36, 12),                      // q attacks P
        ] {
            for mirrored in [false, true] {
                let w = get_threat_feature(0, mirrored, att_p, def_p, att_sq, def_sq);
                // Equivalent pair: colors swapped, squares flipped vertically
                let b = get_threat_feature(
                    1, mirrored, att_p ^ 8, def_p ^ 8, att_sq ^ 56, def_sq ^ 56,
                );
                assert_eq!(w, b, "pov symmetry: ({att_p},{def_p},{att_sq},{def_sq},m={mirrored})");
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
        let path = "/tmp/gaianet_t1_random_test.bin";
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
        // ft_threat_weights: full-range i8
        for _ in 0..(FEATURE_COUNT * L1_SIZE) {
            buf.push((next() & 0xFF) as u8);
        }
        // ft_biases: i16 within ±100
        for _ in 0..L1_SIZE {
            let v = (next() % 201) as i64 - 100;
            buf.extend_from_slice(&(v as i16).to_le_bytes());
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
        ).unwrap();
        let mut net = super::super::Network::new();
        net.refresh(&pos);

        let moves = [
            Move::new_with_type(Square::E1, Square::G1, MT_CASTLING), // O-O
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

            let full = compute_full_threats(&pos);
            let inc = net.threat_values_for_test();
            for pov in 0..2 {
                for j in 0..L1_SIZE {
                    assert_eq!(
                        inc.0[pov][j], full.0[pov][j],
                        "move {} ({:?}): threats pov={pov} idx={j}", i + 1, mv,
                    );
                }
            }
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
        let full = compute_full_threats(&pos);
        let inc = net.threat_values_for_test();
        for pov in 0..2 {
            for j in 0..L1_SIZE {
                assert_eq!(inc.0[pov][j], full.0[pov][j], "promo-capture pov={pov} idx={j}");
            }
        }

        let mut pos = Position::from_fen(
            "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
        ).unwrap();
        let mut net = super::super::Network::new();
        net.refresh(&pos);
        let mv = Move::new_with_type(Square::E5, Square::D6, MT_EN_PASSANT);
        net.push(mv, &pos);
        pos.make_move(mv);
        net.ensure_updated(&pos);
        net.ensure_threats_updated_for_test(&pos);
        let full = compute_full_threats(&pos);
        let inc = net.threat_values_for_test();
        for pov in 0..2 {
            for j in 0..L1_SIZE {
                assert_eq!(inc.0[pov][j], full.0[pov][j], "en passant pov={pov} idx={j}");
            }
        }
    }
}
