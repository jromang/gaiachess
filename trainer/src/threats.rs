//! GaiaNet-T1 threat feature extraction for Bullet training.
//!
//! 41,272 filtered pairwise threat features + HalfKA PST (12 king buckets, factorised).
//!
//! The indexing is TABLE-DRIVEN and IDENTICAL to the engine's `src/nnue/threats.rs`
//! (same ENEMY_SLOT_MAP / OWN_SLOT_MAP / PIECE_TARGET_COUNT / get_threat_feature).
//! Parity is verified end-to-end by `bin/dump_threats.rs` vs the engine's
//! `gaiachess threats <fen>` command.
//!
//! Filter (GaiaChess-original): all threats on enemy pieces are encoded; defenses
//! (same-color targets) only when a pawn is involved.

use std::sync::OnceLock;

use bullet_lib::game::inputs::{Chess768, SparseInputType};
use bulletformat::ChessBoard;

// ============================================================
// Attack tables (standalone, hyperbola quintessence)
// ============================================================

macro_rules! init {
    (|$sq:ident, $size:literal | $($rest:tt)+) => {{
        let mut $sq = 0;
        let mut res = [{$($rest)+}; $size];
        while $sq < $size {
            res[$sq] = {$($rest)+};
            $sq += 1;
        }
        res
    }};
}

struct File;
impl File {
    const A: u64 = 0x0101_0101_0101_0101;
    const H: u64 = Self::A << 7;
}

const EAST: [u64; 64] = init!(|sq, 64| (0xFF << (sq & 56)) ^ (1 << sq) ^ WEST[sq]);
const WEST: [u64; 64] = init!(|sq, 64| (0xFF << (sq & 56)) & ((1 << sq) - 1));
const DIAG: u64 = DIAGS[7];
const DIAGS: [u64; 15] = [
    0x0100_0000_0000_0000,
    0x0201_0000_0000_0000,
    0x0402_0100_0000_0000,
    0x0804_0201_0000_0000,
    0x1008_0402_0100_0000,
    0x2010_0804_0201_0000,
    0x4020_1008_0402_0100,
    0x8040_2010_0804_0201,
    0x0080_4020_1008_0402,
    0x0000_8040_2010_0804,
    0x0000_0080_4020_1008,
    0x0000_0000_8040_2010,
    0x0000_0000_0080_4020,
    0x0000_0000_0000_8040,
    0x0000_0000_0000_0080,
];

#[derive(Clone, Copy)]
struct Mask {
    bit: u64,
    diag: u64,
    anti: u64,
    swap: u64,
}

const PAWN: [[u64; 64]; 2] = [
    init!(|sq, 64| (((1 << sq) & !File::A) << 7) | (((1 << sq) & !File::H) << 9)),
    init!(|sq, 64| (((1 << sq) & !File::A) >> 9) | (((1 << sq) & !File::H) >> 7)),
];

const KNIGHT: [u64; 64] = init!(|sq, 64| {
    let n = 1 << sq;
    let h1 = ((n >> 1) & 0x7f7f_7f7f_7f7f_7f7f) | ((n << 1) & 0xfefe_fefe_fefe_fefe);
    let h2 = ((n >> 2) & 0x3f3f_3f3f_3f3f_3f3f) | ((n << 2) & 0xfcfc_fcfc_fcfc_fcfc);
    (h1 << 16) | (h1 >> 16) | (h2 << 8) | (h2 >> 8)
});

const KING: [u64; 64] = init!(|sq, 64| {
    let mut k = 1 << sq;
    k |= (k << 8) | (k >> 8);
    k |= ((k & !File::A) >> 1) | ((k & !File::H) << 1);
    k ^ (1 << sq)
});

const BISHOP_MASKS: [Mask; 64] = init!(|sq, 64|
    let bit = 1 << sq;
    let file = sq & 7;
    let rank = sq / 8;
    Mask {
        bit,
        diag: bit ^ DIAGS[7 + file - rank],
        anti: bit ^ DIAGS[    file + rank].swap_bytes(),
        swap: bit.swap_bytes()
    }
);

const RANK_SHIFT: [usize; 64] = init!(|sq, 64| sq - (sq & 7) + 1);

const RANK: [[u64; 64]; 64] = init!(|sq, 64| init!(|occ, 64| {
    let file = sq & 7;
    let mask = (occ << 1) as u64;
    let east = ((EAST[file] & mask) | (1 << 63)).trailing_zeros() as usize;
    let west = ((WEST[file] & mask) | 1).leading_zeros() as usize ^ 63;
    (EAST[file] ^ EAST[east] | WEST[file] ^ WEST[west]) << (sq - file)
}));

const FILE_TABLE: [[u64; 64]; 64] =
    init!(|sq, 64| init!(|occ, 64| (RANK[7 - sq / 8][occ].wrapping_mul(DIAG) & File::H) >> (7 - (sq & 7))));

struct Attacks;
impl Attacks {
    #[inline]
    fn pawn(sq: usize, side: usize) -> u64 {
        PAWN[side][sq]
    }

    #[inline]
    fn knight(sq: usize) -> u64 {
        KNIGHT[sq]
    }

    #[inline]
    fn king(sq: usize) -> u64 {
        KING[sq]
    }

    #[inline]
    fn bishop(sq: usize, occ: u64) -> u64 {
        let mask = BISHOP_MASKS[sq];

        let mut diag = occ & mask.diag;
        let mut rev1 = diag.swap_bytes();
        diag = diag.wrapping_sub(mask.bit);
        rev1 = rev1.wrapping_sub(mask.swap);
        diag ^= rev1.swap_bytes();
        diag &= mask.diag;

        let mut anti = occ & mask.anti;
        let mut rev2 = anti.swap_bytes();
        anti = anti.wrapping_sub(mask.bit);
        rev2 = rev2.wrapping_sub(mask.swap);
        anti ^= rev2.swap_bytes();
        anti &= mask.anti;

        diag | anti
    }

    #[inline]
    fn rook(sq: usize, occ: u64) -> u64 {
        let flip = ((occ >> (sq & 7)) & File::A).wrapping_mul(DIAG);
        let file_sq = (flip >> 57) & 0x3F;
        let files = FILE_TABLE[sq][file_sq as usize];

        let rank_sq = (occ >> RANK_SHIFT[sq]) & 0x3F;
        let ranks = RANK[sq][rank_sq as usize];

        ranks | files
    }

    #[inline]
    fn queen(sq: usize, occ: u64) -> u64 {
        Self::bishop(sq, occ) | Self::rook(sq, occ)
    }

    /// Attacks for piece_idx = piece_type | (color << 3), any occupancy.
    #[inline]
    fn piece(piece_idx: usize, sq: usize, occ: u64) -> u64 {
        match piece_idx & 7 {
            0 => Self::pawn(sq, (piece_idx >> 3) & 1),
            1 => Self::knight(sq),
            2 => Self::bishop(sq, occ),
            3 => Self::rook(sq, occ),
            4 => Self::queen(sq, occ),
            5 => Self::king(sq),
            _ => 0,
        }
    }
}

// ============================================================
// GaiaNet-T1 interaction tables — MUST MATCH src/nnue/threats.rs EXACTLY
// ============================================================

/// Total number of filtered threat features (GaiaNet-T1).
pub const TOTAL_THREATS: usize = 41_272;

#[rustfmt::skip]
const ENEMY_SLOT_MAP: [[i8; 6]; 6] = [
    [ 0,  1, -1,  2, -1, -1],  // Pawn   → P,N,R
    [ 0,  1,  2,  3,  4, -1],  // Knight → P,N,B,R,Q
    [ 0,  1,  2,  3, -1, -1],  // Bishop → P,N,B,R
    [ 0,  1,  2,  3, -1, -1],  // Rook   → P,N,B,R
    [ 0,  1,  2,  3,  4, -1],  // Queen  → P,N,B,R,Q
    [ 0,  1,  2,  3, -1, -1],  // King   → P,N,B,R
];

#[rustfmt::skip]
const OWN_SLOT_MAP: [[i8; 6]; 6] = [
    [ 3,  4, -1,  5, -1, -1],  // Pawn   defends P,N,R
    [ 5, -1, -1, -1, -1, -1],  // Knight defends P
    [ 4, -1, -1, -1, -1, -1],  // Bishop defends P
    [ 4, -1, -1, -1, -1, -1],  // Rook   defends P
    [ 5, -1, -1, -1, -1, -1],  // Queen  defends P
    [ 4, -1, -1, -1, -1, -1],  // King   defends P
];

const PIECE_TARGET_COUNT: [usize; 6] = [6, 6, 5, 5, 6, 5];

#[derive(Clone, Copy)]
struct PiecePairData {
    base_feature: i32,
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

struct ThreatTables {
    piece_offset: [[i32; 64]; 14],
    attack_index: [[[u8; 64]; 64]; 14],
    piece_pair: [[PiecePairData; 14]; 14],
}

static THREAT_TABLES: OnceLock<ThreatTables> = OnceLock::new();

fn get_tables() -> &'static ThreatTables {
    THREAT_TABLES.get_or_init(init_tables)
}

/// Empty-board attacks; pawns on ranks 1 and 8 are unreachable → 0.
fn empty_board_attacks(piece_idx: usize, sq: usize) -> u64 {
    let rank = sq / 8;
    if piece_idx & 7 == 0 && (rank == 0 || rank == 7) {
        return 0;
    }
    Attacks::piece(piece_idx, sq, 0)
}

fn init_tables() -> ThreatTables {
    let mut piece_offset = [[0i32; 64]; 14];
    let mut attack_index = [[[0u8; 64]; 64]; 14];
    let mut piece_pair = [[PiecePairData { base_feature: -1, semi_excluded: false }; 14]; 14];

    let mut cumulative_piece_offset = [[0i32; 2]; 6];

    for att_type in 0..6usize {
        for att_color in 0..2usize {
            let att_idx = att_type | (att_color << 3);
            let mut cumulative = 0i32;
            for sq in 0..64usize {
                piece_offset[att_idx][sq] = cumulative;
                let attacks = empty_board_attacks(att_idx, sq);
                for target in 0..64usize {
                    let mask = if target > 0 { (1u64 << target) - 1 } else { 0 };
                    attack_index[att_idx][sq][target] = (attacks & mask).count_ones() as u8;
                }
                cumulative += attacks.count_ones() as i32;
            }
            cumulative_piece_offset[att_type][att_color] = cumulative;
        }
    }

    let mut cumulative_offset = [[0i32; 2]; 6];
    {
        let mut running = 0i32;
        for att_color in 0..2usize {
            for att_type in 0..6usize {
                cumulative_offset[att_type][att_color] = running;
                running += PIECE_TARGET_COUNT[att_type] as i32
                    * cumulative_piece_offset[att_type][att_color];
            }
        }
        assert_eq!(running as usize, TOTAL_THREATS, "TOTAL_THREATS mismatch");
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

/// Compute the threat feature index — same code as the engine's get_threat_feature.
/// Returns TOTAL_THREATS if the pair is excluded.
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
    let square_flip = (7 * mirrored as usize) ^ (56 * pov);
    let side_flip = pov << 3;

    let att_sq_f = att_sq ^ square_flip;
    let def_sq_f = def_sq ^ square_flip;
    let att_p_f = att_piece_idx ^ side_flip;
    let def_p_f = def_piece_idx ^ side_flip;

    if (att_p_f & 7) > 5 || (def_p_f & 7) > 5 {
        return TOTAL_THREATS;
    }

    let pair = tables.piece_pair[att_p_f][def_p_f];
    if pair.is_excluded(att_sq_f, def_sq_f) {
        return TOTAL_THREATS;
    }

    let idx = pair.base_feature as usize
        + tables.piece_offset[att_p_f][att_sq_f] as usize
        + tables.attack_index[att_p_f][att_sq_f][def_sq_f] as usize;

    debug_assert!(idx < TOTAL_THREATS);
    idx
}

// ============================================================
// Threat collection over a bulletformat ChessBoard (STM frame)
// ============================================================

/// Collect sorted threat feature lists for STM (pov=0) and NTM (pov=1).
///
/// The bulletformat board is STM-relative; `our_ksq()` is the STM king,
/// `opp_ksq()` is already rank-flipped (^56) by bulletformat.
pub fn collect_threats(board: &ChessBoard) -> (Vec<usize>, Vec<usize>) {
    let stm_mirrored = board.our_ksq() % 8 >= 4;
    let ntm_mirrored = board.opp_ksq() % 8 >= 4;

    let mut piece_on = [0xFFu8; 64];
    let mut occ = 0u64;
    for (piece, square) in board.into_iter() {
        piece_on[square as usize] = piece;
        occ |= 1u64 << square;
    }

    let mut stm_threats = Vec::with_capacity(64);
    let mut ntm_threats = Vec::with_capacity(64);

    for att_sq in 0..64usize {
        let att_nibble = piece_on[att_sq];
        if att_nibble == 0xFF {
            continue;
        }
        let att_piece_idx = (att_nibble & 7) as usize | (((att_nibble >> 3) & 1) as usize) << 3;

        let attacks = Attacks::piece(att_piece_idx, att_sq, occ) & occ & !(1u64 << att_sq);

        let mut targets = attacks;
        while targets != 0 {
            let def_sq = targets.trailing_zeros() as usize;
            targets &= targets - 1;
            let def_nibble = piece_on[def_sq];
            let def_piece_idx = (def_nibble & 7) as usize | (((def_nibble >> 3) & 1) as usize) << 3;

            let feat_stm =
                get_threat_feature(0, stm_mirrored, att_piece_idx, def_piece_idx, att_sq, def_sq);
            let feat_ntm =
                get_threat_feature(1, ntm_mirrored, att_piece_idx, def_piece_idx, att_sq, def_sq);

            if feat_stm < TOTAL_THREATS {
                stm_threats.push(feat_stm);
            }
            if feat_ntm < TOTAL_THREATS {
                ntm_threats.push(feat_ntm);
            }
        }
    }

    stm_threats.sort_unstable();
    ntm_threats.sort_unstable();
    (stm_threats, ntm_threats)
}

// ============================================================
// GaiaNetT1Inputs — Bullet SparseInputType
// ============================================================

fn get_num_buckets<const N: usize>(arr: &[usize; N]) -> usize {
    let mut max = 0;
    for &val in arr {
        max = max.max(val);
    }
    max + 1
}

/// PST + filtered threat input type (GaiaNet-T1).
///
/// Feature layout:
/// - `[0, 768)`: PST factorised (shared across king buckets)
/// - `[768, 768 + TOTAL_THREATS)`: filtered pairwise threat features (41,272)
/// - `[768 + TOTAL_THREATS, ...)`: PST bucketed (768 × num_buckets)
#[derive(Clone, Copy, Debug)]
pub struct GaiaNetT1Inputs {
    buckets: [usize; 64],
    pub num_buckets: usize,
}

impl GaiaNetT1Inputs {
    pub fn new(buckets: [usize; 32]) -> Self {
        let num_buckets = get_num_buckets(&buckets);
        let mut expanded = [0; 64];
        for (idx, elem) in expanded.iter_mut().enumerate() {
            *elem = buckets[(idx / 8) * 4 + [0, 1, 2, 3, 3, 2, 1, 0][idx % 8]];
        }
        Self { buckets: expanded, num_buckets }
    }
}

impl SparseInputType for GaiaNetT1Inputs {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        768 + TOTAL_THREATS + 768 * self.num_buckets
    }

    fn max_active(&self) -> usize {
        // 64 PST (32 pieces × 2: bucketed + factorised) + threats (≤ ~96 with the filter)
        192
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        let get = |ksq| (if ksq % 8 > 3 { 7 } else { 0 }, 768 * self.buckets[usize::from(ksq)]);
        let (stm_flip, stm_bucket) = get(pos.our_ksq());
        // opp_ksq is already rank-flipped (^56) in bulletformat
        let (ntm_flip, ntm_bucket) = get(pos.opp_ksq());

        // PST features: bucketed + factorised
        Chess768.map_features(pos, |stm, ntm| {
            let bucketed_offset = 768 + TOTAL_THREATS;
            f(bucketed_offset + stm_bucket + (stm ^ stm_flip), bucketed_offset + ntm_bucket + (ntm ^ ntm_flip));
            f(stm ^ stm_flip, ntm ^ ntm_flip);
        });

        // Threat features. The (stm, ntm) pairing is arbitrary: Bullet feeds two
        // independent sparse inputs; only the per-perspective sets matter. Counts
        // are equal by construction (the filter is pov-invariant; semi-exclusion
        // keeps exactly one direction of each mutual same-type pair per pov).
        let (stm_threats, ntm_threats) = collect_threats(pos);
        debug_assert_eq!(stm_threats.len(), ntm_threats.len(),
            "STM/NTM threat count mismatch: {} vs {}", stm_threats.len(), ntm_threats.len());

        for (&stm, &ntm) in stm_threats.iter().zip(ntm_threats.iter()) {
            f(768 + stm, 768 + ntm);
        }
    }

    fn shorthand(&self) -> String {
        format!("{TOTAL_THREATS}t+768x{}", self.num_buckets)
    }

    fn description(&self) -> String {
        "GaiaNet-T1 filtered threat inputs, bucketed mirrored factorised".to_string()
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    const BUCKET_LAYOUT: [usize; 32] = [
         0,  1,  2,  3,
         4,  5,  6,  7,
         8,  8,  9,  9,
        10, 10, 10, 10,
        11, 11, 11, 11,
        11, 11, 11, 11,
        11, 11, 11, 11,
        11, 11, 11, 11,
    ];

    fn startpos() -> ChessBoard {
        ChessBoard::from_raw(
            [
                0x000000000000FFFFu64, 0xFFFF000000000000u64,
                0x00FF00000000FF00u64, 0x4200000000000042u64,
                0x2400000000000024u64, 0x8100000000000081u64,
                0x0800000000000008u64, 0x1000000000000010u64,
            ],
            0, 0, 0.5,
        ).unwrap()
    }

    #[test]
    fn test_total_threats_count() {
        // Force init_tables (asserts the 41,272 total internally)
        let _ = get_tables();
        assert_eq!(TOTAL_THREATS, 41_272);
    }

    #[test]
    fn test_num_inputs() {
        let input = GaiaNetT1Inputs::new(BUCKET_LAYOUT);
        assert_eq!(input.num_buckets, 12);
        assert_eq!(input.num_inputs(), 768 + TOTAL_THREATS + 768 * 12); // 51,256
    }

    #[test]
    fn test_startpos_features_bounds_and_counts() {
        let board = startpos();
        let input = GaiaNetT1Inputs::new(BUCKET_LAYOUT);
        let mut count = 0usize;
        input.map_features(&board, |stm, ntm| {
            assert!(stm < input.num_inputs(), "stm {stm} OOB");
            assert!(ntm < input.num_inputs(), "ntm {ntm} OOB");
            count += 1;
        });
        assert!(count <= input.max_active(), "count {count} > max_active");
        // 32 pieces × 2 (bucketed + factorised) + threats
        assert!(count > 64, "expected threats beyond PST, got {count}");
    }

    #[test]
    fn test_startpos_stm_ntm_symmetric() {
        // Startpos is symmetric: STM and NTM threat sets must be identical.
        let board = startpos();
        let (stm, ntm) = collect_threats(&board);
        assert_eq!(stm.len(), ntm.len());
        assert_eq!(stm, ntm, "startpos threat sets must match");
    }

    #[test]
    fn test_filter_no_piece_defenses() {
        // Position with a knight defending a rook (no pawn): feature must be excluded.
        // White: Kg1, Nb1, Rd2 — Nb1 defends Rd2. Black: Kg8.
        let board = ChessBoard::from_raw(
            [
                (1u64 << 6) | (1u64 << 1) | (1u64 << 11),
                1u64 << 62,
                0,
                1u64 << 1,
                0,
                1u64 << 11,
                0,
                (1u64 << 6) | (1u64 << 62),
            ],
            0, 0, 0.5,
        ).unwrap();
        let (stm, _) = collect_threats(&board);
        // Only N→R defenses would be possible here → no threat features
        assert!(stm.is_empty(), "N defends R must be filtered out, got {stm:?}");
    }
}
