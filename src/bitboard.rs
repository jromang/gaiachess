//! Bitboard operations, attack tables, and sliding piece attack generation.
//!
//! A [bitboard](https://www.chessprogramming.org/Bitboards) represents 64 squares
//! as bits in a `u64`, enabling efficient set operations via bitwise instructions.
//!
//! Sliding piece attacks (bishop, rook, queen) use
//! [magic bitboards](https://www.chessprogramming.org/Magic_Bitboards) as default,
//! with AVX2 SIMD as a faster alternative (see [`crate::simd_attacks`]).
//! Non-sliding attacks (knight, king, pawn) use pre-computed lookup tables.
//!
//! All tables are computed at compile time via `const fn` — zero `unsafe`.
//! Magic numbers generated with a trial-and-error finder (fixed SplitMix64 seed,
//! generator kept in tools/scripts/magic_finder.rs).

use crate::types::*;

// ============================================================
// Bitboard constants
// ============================================================

/// Bitmask for file A (bits 0, 8, 16, ..., 56).
pub const FILE_A: u64 = 0x0101_0101_0101_0101;
pub const FILE_B: u64 = FILE_A << 1;
#[allow(dead_code)]
pub const FILE_C: u64 = FILE_A << 2;
#[allow(dead_code)]
pub const FILE_D: u64 = FILE_A << 3;
#[allow(dead_code)]
pub const FILE_E: u64 = FILE_A << 4;
#[allow(dead_code)]
pub const FILE_F: u64 = FILE_A << 5;
pub const FILE_G: u64 = FILE_A << 6;
pub const FILE_H: u64 = FILE_A << 7;

/// Bitmask for rank 1 (bits 0..7).
pub const RANK_1: u64 = 0xFF;
pub const RANK_2: u64 = RANK_1 << 8;
pub const RANK_3: u64 = RANK_1 << 16;
#[allow(dead_code)]
pub const RANK_4: u64 = RANK_1 << 24;
#[allow(dead_code)]
pub const RANK_5: u64 = RANK_1 << 32;
pub const RANK_6: u64 = RANK_1 << 40;
pub const RANK_7: u64 = RANK_1 << 48;
pub const RANK_8: u64 = RANK_1 << 56;

#[allow(dead_code)]
pub const FILES: [u64; 8] = [FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H];
#[allow(dead_code)]
pub const RANKS: [u64; 8] = [RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8];

// ============================================================
// Bitboard operations
// ============================================================

/// Number of set bits (population count / Hamming weight).
#[inline(always)]
pub fn popcount(bb: u64) -> u32 {
    bb.count_ones()
}

/// Least significant set bit as a [`Square`]. Undefined if `bb == 0`.
#[inline(always)]
pub fn lsb(bb: u64) -> Square {
    debug_assert!(bb != 0);
    Square(bb.trailing_zeros() as u8)
}

/// Extract and clear the least significant set bit, returning its [`Square`].
#[inline(always)]
pub fn pop_lsb(bb: &mut u64) -> Square {
    let sq = lsb(*bb);
    *bb &= *bb - 1;
    sq
}

/// Returns `true` if more than one bit is set (i.e. the set has 2+ elements).
#[inline(always)]
pub fn more_than_one(bb: u64) -> bool {
    bb & (bb - 1) != 0
}

// ============================================================
// Directional shifts (A1=0, NORTH = <<8)
// File masks prevent wrap-around when shifting east/west.
// ============================================================

#[inline(always)]
pub const fn shift_north(bb: u64) -> u64 { bb << 8 }
#[inline(always)]
pub const fn shift_south(bb: u64) -> u64 { bb >> 8 }
#[inline(always)]
pub const fn shift_east(bb: u64) -> u64 { (bb & !FILE_H) << 1 }
#[inline(always)]
pub const fn shift_west(bb: u64) -> u64 { (bb & !FILE_A) >> 1 }
#[inline(always)]
pub const fn shift_north_east(bb: u64) -> u64 { (bb & !FILE_H) << 9 }
#[inline(always)]
pub const fn shift_north_west(bb: u64) -> u64 { (bb & !FILE_A) << 7 }
#[inline(always)]
pub const fn shift_south_east(bb: u64) -> u64 { (bb & !FILE_H) >> 7 }
#[inline(always)]
pub const fn shift_south_west(bb: u64) -> u64 { (bb & !FILE_A) >> 9 }

#[inline(always)]
pub fn pawn_push_bb(bb: u64, c: Color) -> u64 {
    match c { Color::White => shift_north(bb), Color::Black => shift_south(bb) }
}

#[inline(always)]
pub fn pawn_attack_east(bb: u64, c: Color) -> u64 {
    match c { Color::White => shift_north_east(bb), Color::Black => shift_south_east(bb) }
}

#[inline(always)]
pub fn pawn_attack_west(bb: u64, c: Color) -> u64 {
    match c { Color::White => shift_north_west(bb), Color::Black => shift_south_west(bb) }
}

// ============================================================
// Magic bitboard structures
// Compiled only when AVX2 SIMD attacks are not available.
// ============================================================

#[cfg(not(target_feature = "avx2"))]
#[derive(Clone, Copy)]
struct MagicEntry {
    mask: u64,
    magic: u64,
    shift: u32,
    offset: u32,
}

#[cfg(not(target_feature = "avx2"))]
#[inline(always)]
const fn magic_index(occupancies: u64, entry: &MagicEntry) -> usize {
    let hash = (occupancies & entry.mask).wrapping_mul(entry.magic) >> entry.shift;
    hash as usize + entry.offset as usize
}

// ============================================================
// Compile-time table initialization (const fn)
// ============================================================

pub(crate) const BISHOP_DELTAS: [i8; 4] = [NORTH_EAST, NORTH_WEST, SOUTH_EAST, SOUTH_WEST];
pub(crate) const ROOK_DELTAS: [i8; 4] = [NORTH, SOUTH, EAST, WEST];

/// Compute sliding attacks on the fly by looping over ray deltas.
/// Used at compile time to populate the magic bitboard lookup tables.
pub(crate) const fn sliding_attack_otf(sq: u8, occupied: u64, deltas: &[i8; 4]) -> u64 {
    let mut attacks = 0u64;
    let mut d = 0;
    while d < 4 {
        let delta = deltas[d];
        let mut s = sq as i8 + delta;
        while s >= 0 && s < 64 {
            let file_diff = (s & 7) - ((s - delta) & 7);
            let abs_diff = if file_diff < 0 { -file_diff } else { file_diff };
            if abs_diff > 2 {
                break; // wrapped around
            }
            attacks |= 1u64 << (s as u32);
            if occupied & (1u64 << (s as u32)) != 0 {
                break; // blocked
            }
            s += delta;
        }
        d += 1;
    }
    attacks
}

pub(crate) const fn init_knight_attacks() -> [u64; 64] {
    let mut table = [0u64; 64];
    let mut sq = 0u8;
    while sq < 64 {
        let bb = 1u64 << sq;
        let mut attacks = 0u64;
        attacks |= (bb & !FILE_A & !FILE_B) << 6;
        attacks |= (bb & !FILE_A) << 15;
        attacks |= (bb & !FILE_H) << 17;
        attacks |= (bb & !FILE_G & !FILE_H) << 10;
        attacks |= (bb & !FILE_G & !FILE_H) >> 6;
        attacks |= (bb & !FILE_H) >> 15;
        attacks |= (bb & !FILE_A) >> 17;
        attacks |= (bb & !FILE_A & !FILE_B) >> 10;
        table[sq as usize] = attacks;
        sq += 1;
    }
    table
}

pub(crate) const fn init_king_attacks() -> [u64; 64] {
    let mut table = [0u64; 64];
    let mut sq = 0u8;
    while sq < 64 {
        let bb = 1u64 << sq;
        table[sq as usize] = shift_north(bb)
            | shift_south(bb)
            | shift_east(bb)
            | shift_west(bb)
            | shift_north_east(bb)
            | shift_north_west(bb)
            | shift_south_east(bb)
            | shift_south_west(bb);
        sq += 1;
    }
    table
}

pub(crate) const fn init_pawn_attacks() -> [[u64; 64]; 2] {
    let mut table = [[0u64; 64]; 2];
    let mut sq = 0u8;
    while sq < 64 {
        let bb = 1u64 << sq;
        table[0][sq as usize] = shift_north_east(bb) | shift_north_west(bb);
        table[1][sq as usize] = shift_south_east(bb) | shift_south_west(bb);
        sq += 1;
    }
    table
}

#[cfg(not(target_feature = "avx2"))]
const fn init_rook_table() -> [u64; 102400] {
    let mut table = [0u64; 102400];
    let mut sq = 0u8;
    while sq < 64 {
        let entry = &ROOK_MAGICS[sq as usize];
        let mask = entry.mask;
        let mut occ = 0u64;
        loop {
            let attacks = sliding_attack_otf(sq, occ, &ROOK_DELTAS);
            let idx = magic_index(occ, entry);
            table[idx] = attacks;
            occ = occ.wrapping_sub(mask) & mask;
            if occ == 0 { break; }
        }
        sq += 1;
    }
    table
}

#[cfg(not(target_feature = "avx2"))]
const fn init_bishop_table() -> [u64; 5248] {
    let mut table = [0u64; 5248];
    let mut sq = 0u8;
    while sq < 64 {
        let entry = &BISHOP_MAGICS[sq as usize];
        let mask = entry.mask;
        let mut occ = 0u64;
        loop {
            let attacks = sliding_attack_otf(sq, occ, &BISHOP_DELTAS);
            let idx = magic_index(occ, entry);
            table[idx] = attacks;
            occ = occ.wrapping_sub(mask) & mask;
            if occ == 0 { break; }
        }
        sq += 1;
    }
    table
}

const fn init_between_line() -> ([[u64; 64]; 64], [[u64; 64]; 64]) {
    let mut between = [[0u64; 64]; 64];
    let mut line = [[0u64; 64]; 64];

    let mut a = 0u8;
    while a < 64 {
        let mut b = 0u8;
        while b < 64 {
            let bb_a = 1u64 << a;
            let bb_b = 1u64 << b;

            let rook_a = sliding_attack_otf(a, 0, &ROOK_DELTAS);
            if rook_a & bb_b != 0 {
                let rook_a_thru_b = sliding_attack_otf(a, bb_b, &ROOK_DELTAS);
                let rook_b_thru_a = sliding_attack_otf(b, bb_a, &ROOK_DELTAS);
                let rook_b = sliding_attack_otf(b, 0, &ROOK_DELTAS);
                between[a as usize][b as usize] = rook_a_thru_b & rook_b_thru_a | bb_b;
                line[a as usize][b as usize] = (rook_a & rook_b) | bb_a | bb_b;
            } else {
                let bishop_a = sliding_attack_otf(a, 0, &BISHOP_DELTAS);
                if bishop_a & bb_b != 0 {
                    let bishop_a_thru_b = sliding_attack_otf(a, bb_b, &BISHOP_DELTAS);
                    let bishop_b_thru_a = sliding_attack_otf(b, bb_a, &BISHOP_DELTAS);
                    let bishop_b = sliding_attack_otf(b, 0, &BISHOP_DELTAS);
                    between[a as usize][b as usize] = bishop_a_thru_b & bishop_b_thru_a | bb_b;
                    line[a as usize][b as usize] = (bishop_a & bishop_b) | bb_a | bb_b;
                }
            }

            b += 1;
        }
        a += 1;
    }

    (between, line)
}

// ============================================================
// Static tables (compile-time initialized, no unsafe)
// ============================================================

static KNIGHT_ATTACKS: [u64; 64] = init_knight_attacks();
static KING_ATTACKS: [u64; 64] = init_king_attacks();
static PAWN_ATTACKS: [[u64; 64]; 2] = init_pawn_attacks();
#[cfg(not(target_feature = "avx2"))]
static ROOK_TABLE: [u64; 102400] = init_rook_table();
#[cfg(not(target_feature = "avx2"))]
static BISHOP_TABLE: [u64; 5248] = init_bishop_table();

const _BETWEEN_LINE: ([[u64; 64]; 64], [[u64; 64]; 64]) = init_between_line();
static BETWEEN_BB: [[u64; 64]; 64] = _BETWEEN_LINE.0;
static LINE_BB: [[u64; 64]; 64] = _BETWEEN_LINE.1;
static ALIGNED_BB: [u64; 64] = init_aligned();

/// One row of `LINE_BB` folded into a mask per square.
const fn init_aligned() -> [u64; 64] {
    let line = _BETWEEN_LINE.1;
    let mut out = [0u64; 64];
    let mut s = 0;
    while s < 64 {
        let mut t = 0;
        while t < 64 {
            out[s] |= line[s][t];
            t += 1;
        }
        s += 1;
    }
    out
}

// ============================================================
// Public attack lookup functions
// ============================================================

/// Pre-computed knight attacks for a given square.
#[inline(always)]
pub fn knight_attacks(sq: Square) -> u64 {
    KNIGHT_ATTACKS[sq.index()]
}

/// Pre-computed king attacks for a given square.
#[inline(always)]
pub fn king_attacks(sq: Square) -> u64 {
    KING_ATTACKS[sq.index()]
}

/// Pre-computed pawn attacks for a given square and color.
#[inline(always)]
pub fn pawn_attacks(sq: Square, c: Color) -> u64 {
    PAWN_ATTACKS[c.index()][sq.index()]
}

/// Magic-based bishop attacks (fallback when AVX2 is not available).
#[cfg(not(target_feature = "avx2"))]
#[inline(always)]
fn magic_bishop_attacks(sq: Square, occupied: u64) -> u64 {
    let entry = &BISHOP_MAGICS[sq.index()];
    let idx = magic_index(occupied, entry);
    BISHOP_TABLE[idx]
}

/// Magic-based rook attacks (fallback when AVX2 is not available).
#[cfg(not(target_feature = "avx2"))]
#[inline(always)]
fn magic_rook_attacks(sq: Square, occupied: u64) -> u64 {
    let entry = &ROOK_MAGICS[sq.index()];
    let idx = magic_index(occupied, entry);
    ROOK_TABLE[idx]
}

// ============================================================
// PEXT attack tables (BMI2) — ~840 KB of read-only data, built at compile time
// ============================================================

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)] // read by the static-BMI2 arm or the gaia_dist runtime election
mod pext {
    #![allow(unsafe_op_in_unsafe_fn)]

    use super::{sliding_attack_otf, BISHOP_DELTAS, ROOK_DELTAS};
    use crate::types::Square;

    #[derive(Clone, Copy)]
    struct PextEntry {
        mask: u64,
        offset: u32,
    }

    /// Rook occupancy mask for square `sq` (edges excluded).
    const fn rook_mask(sq: u8) -> u64 {
        let rank = sq >> 3;
        let file = sq & 7;
        let mut mask = 0u64;
        let mut f = 1;
        while f < 7 {
            if f != file { mask |= 1u64 << (rank * 8 + f); }
            f += 1;
        }
        let mut r = 1;
        while r < 7 {
            if r != rank { mask |= 1u64 << (r * 8 + file); }
            r += 1;
        }
        mask
    }

    /// Bishop occupancy mask for square `sq` (edges excluded).
    const fn bishop_mask(sq: u8) -> u64 {
        let rank = sq as i8 >> 3;
        let file = sq as i8 & 7;
        let mut mask = 0u64;
        let deltas: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
        let mut d = 0;
        while d < 4 {
            let (dr, df) = deltas[d];
            let mut r = rank + dr;
            let mut f = file + df;
            while r > 0 && r < 7 && f > 0 && f < 7 {
                mask |= 1u64 << ((r * 8 + f) as u32);
                r += dr;
                f += df;
            }
            d += 1;
        }
        mask
    }

    /// One ray of a slider: the squares along `ray` up to and including the first
    /// blocker in `occ`. `towards_higher` says which end of the ray is nearest the
    /// piece — the lowest set blocker for a ray running up the board, the highest
    /// for one running down.
    const fn ray_attack(ray: u64, occ: u64, towards_higher: bool) -> u64 {
        let blockers = ray & occ;
        if blockers == 0 {
            return ray;
        }
        if towards_higher {
            let first = blockers & blockers.wrapping_neg();
            ray & (first << 1).wrapping_sub(1)
        } else {
            let first = 1u64 << (63 - blockers.leading_zeros());
            ray & !first.wrapping_sub(1)
        }
    }

    /// The empty-board ray from `sq` along one delta.
    const fn ray_mask(sq: u8, delta: i8) -> u64 {
        sliding_attack_otf(sq, 0, &[delta, delta, delta, delta])
    }

    /// Slider attacks from `sq` on `occ`, one ray at a time. The same answer as
    /// walking the rays square by square, at a fraction of the compile-time steps —
    /// this runs a hundred thousand times when the tables are built.
    const fn slider_attacks(rays: &[u64; 4], deltas: &[i8; 4], occ: u64) -> u64 {
        let mut attacks = 0u64;
        let mut d = 0;
        while d < 4 {
            attacks |= ray_attack(rays[d], occ, deltas[d] > 0);
            d += 1;
        }
        attacks
    }

    const fn table_size(rook: bool) -> usize {
        let mut total = 0usize;
        let mut sq = 0u8;
        while sq < 64 {
            let mask = if rook { rook_mask(sq) } else { bishop_mask(sq) };
            total += 1 << mask.count_ones();
            sq += 1;
        }
        total
    }

    const ROOK_TABLE_SIZE: usize = table_size(true);
    const BISHOP_TABLE_SIZE: usize = table_size(false);
    const ATTACK_TABLE_SIZE: usize = ROOK_TABLE_SIZE + BISHOP_TABLE_SIZE;
    const _: () = assert!(ROOK_TABLE_SIZE == 102_400 && BISHOP_TABLE_SIZE == 5_248);

    /// Rook entries index the front of the attack table, bishop entries the back.
    const fn build_entries(rook: bool) -> [PextEntry; 64] {
        let mut out = [PextEntry { mask: 0, offset: 0 }; 64];
        let mut offset = if rook { 0usize } else { ROOK_TABLE_SIZE };
        let mut sq = 0u8;
        while sq < 64 {
            let mask = if rook { rook_mask(sq) } else { bishop_mask(sq) };
            out[sq as usize] = PextEntry { mask, offset: offset as u32 };
            offset += 1 << mask.count_ones();
            sq += 1;
        }
        out
    }

    const fn build_attacks() -> [u64; ATTACK_TABLE_SIZE] {
        let mut attacks = [0u64; ATTACK_TABLE_SIZE];
        let mut kind = 0;
        while kind < 2 {
            let rook = kind == 0;
            let entries = build_entries(rook);
            let deltas = if rook { &ROOK_DELTAS } else { &BISHOP_DELTAS };
            let mut sq = 0u8;
            while sq < 64 {
                let entry = entries[sq as usize];
                let base = entry.offset as usize;
                let rays = [
                    ray_mask(sq, deltas[0]),
                    ray_mask(sq, deltas[1]),
                    ray_mask(sq, deltas[2]),
                    ray_mask(sq, deltas[3]),
                ];
                // The carry-rippler visits the subsets of the mask in increasing
                // order of their PEXT index, so a running counter is that index.
                let mut occ = 0u64;
                let mut i = 0usize;
                loop {
                    attacks[base + i] = slider_attacks(&rays, deltas, occ);
                    i += 1;
                    occ = occ.wrapping_sub(entry.mask) & entry.mask;
                    if occ == 0 {
                        break;
                    }
                }
                sq += 1;
            }
            kind += 1;
        }
        attacks
    }

    // Everything is computed at compile time and lives in read-only data: nothing to
    // build before the first search, no initialisation word to read before each
    // lookup, and the attack row is indexed directly instead of through a pointer
    // loaded first. Two dependent loads per attack — the entry, then the row.
    static ROOK_ENTRIES: [PextEntry; 64] = build_entries(true);
    static BISHOP_ENTRIES: [PextEntry; 64] = build_entries(false);
    static ATTACKS: [u64; ATTACK_TABLE_SIZE] = build_attacks();

    /// Kept for the callers that used to order the table build before the first
    /// search; there is nothing left to build.
    pub fn ensure_init() {}

    #[target_feature(enable = "bmi2")]
    #[inline]
    pub unsafe fn rook_attacks(sq: Square, occupied: u64) -> u64 {
        use std::arch::x86_64::_pext_u64;
        debug_assert!(sq.0 < 64, "rook_attacks: square {} OOB", sq.0);
        let entry = ROOK_ENTRIES.get_unchecked(sq.index());
        let idx = _pext_u64(occupied, entry.mask) as usize + entry.offset as usize;
        debug_assert!(idx < ATTACK_TABLE_SIZE);
        *ATTACKS.get_unchecked(idx)
    }

    #[target_feature(enable = "bmi2")]
    #[inline]
    pub unsafe fn bishop_attacks(sq: Square, occupied: u64) -> u64 {
        use std::arch::x86_64::_pext_u64;
        debug_assert!(sq.0 < 64, "bishop_attacks: square {} OOB", sq.0);
        let entry = BISHOP_ENTRIES.get_unchecked(sq.index());
        let idx = _pext_u64(occupied, entry.mask) as usize + entry.offset as usize;
        debug_assert!(idx < ATTACK_TABLE_SIZE);
        *ATTACKS.get_unchecked(idx)
    }
}

/// The PEXT tables are compile-time constants; this is kept so that the callers
/// that ordered their construction before the first search still compile.
#[cfg(target_arch = "x86_64")]
pub fn init_pext() { pext::ensure_init(); }

/// Whether slider attacks go through hardware PEXT — decided once at startup
/// by `cpu::init` (BMI2 present, and not an AMD generation that microcodes
/// PDEP/PEXT), then read as a plain load on a never-rewritten line. Defaults
/// to false so binaries that skipped init (unit tests) stay correct on the
/// portable path: every attack kind returns identical bitboards.
///
/// Only distribution builds (`--cfg gaia_dist`) take this runtime branch: the
/// check costs a measurable slice of the node rate in the attack getters, and
/// a build made for one machine can pin the answer at compile time instead.
#[cfg(all(target_arch = "x86_64", gaia_dist))]
static USE_PEXT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Elect (or retire) the PEXT path. Build the tables BEFORE electing: the
/// branchless `tables()` read in the attack getters relies on it.
#[cfg(all(target_arch = "x86_64", gaia_dist))]
pub fn set_use_pext(enabled: bool) {
    if enabled {
        init_pext();
    }
    USE_PEXT.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(all(target_arch = "x86_64", gaia_dist))]
#[inline(always)]
fn use_pext() -> bool {
    USE_PEXT.load(std::sync::atomic::Ordering::Relaxed)
}

/// Bishop attacks: PEXT (BMI2) > AVX2 BLSMSK > magic bitboards.
/// Distribution builds elect PEXT at runtime; others pin it at compile time.
#[inline(always)]
pub fn bishop_attacks(sq: Square, occupied: u64) -> u64 {
    #[cfg(all(target_arch = "x86_64", gaia_dist))]
    if use_pext() {
        return unsafe { pext::bishop_attacks(sq, occupied) };
    }
    #[cfg(all(target_feature = "bmi2", not(gaia_dist)))]
    return unsafe { pext::bishop_attacks(sq, occupied) };
    #[cfg(all(target_feature = "avx2", any(not(target_feature = "bmi2"), gaia_dist)))]
    return crate::simd_attacks::bishop_attacks(sq, occupied);
    #[cfg(not(target_feature = "avx2"))]
    magic_bishop_attacks(sq, occupied)
}

/// Rook attacks: PEXT (BMI2) > AVX2 BLSMSK > magic bitboards.
/// Distribution builds elect PEXT at runtime; others pin it at compile time.
#[inline(always)]
pub fn rook_attacks(sq: Square, occupied: u64) -> u64 {
    #[cfg(all(target_arch = "x86_64", gaia_dist))]
    if use_pext() {
        return unsafe { pext::rook_attacks(sq, occupied) };
    }
    #[cfg(all(target_feature = "bmi2", not(gaia_dist)))]
    return unsafe { pext::rook_attacks(sq, occupied) };
    #[cfg(all(target_feature = "avx2", any(not(target_feature = "bmi2"), gaia_dist)))]
    return crate::simd_attacks::rook_attacks(sq, occupied);
    #[cfg(not(target_feature = "avx2"))]
    magic_rook_attacks(sq, occupied)
}

/// Queen attacks (bishop | rook): PEXT > AVX2 BLSMSK > magic.
/// Distribution builds elect PEXT at runtime; others pin it at compile time.
#[inline(always)]
pub fn queen_attacks(sq: Square, occupied: u64) -> u64 {
    #[cfg(all(target_arch = "x86_64", gaia_dist))]
    if use_pext() {
        return unsafe { pext::bishop_attacks(sq, occupied) | pext::rook_attacks(sq, occupied) };
    }
    #[cfg(all(target_feature = "bmi2", not(gaia_dist)))]
    return unsafe { pext::bishop_attacks(sq, occupied) | pext::rook_attacks(sq, occupied) };
    #[cfg(all(target_feature = "avx2", any(not(target_feature = "bmi2"), gaia_dist)))]
    return crate::simd_attacks::queen_attacks(sq, occupied);
    #[cfg(not(target_feature = "avx2"))]
    { magic_bishop_attacks(sq, occupied) | magic_rook_attacks(sq, occupied) }
}

/// Squares strictly between `s1` and `s2` (plus `s2` itself) along a rank, file,
/// or diagonal. Returns 0 if the squares are not aligned.
#[inline(always)]
pub fn between_bb(s1: Square, s2: Square) -> u64 {
    BETWEEN_BB[s1.index()][s2.index()]
}

/// All squares on the line passing through `s1` and `s2` (both included).
/// Returns 0 if the squares are not aligned on a rank, file, or diagonal.
#[inline(always)]
pub fn line_bb(s1: Square, s2: Square) -> u64 {
    LINE_BB[s1.index()][s2.index()]
}

/// Every square sharing a rank, file, or diagonal with `sq`, `sq` itself included.
///
/// The same question as `line_bb(x, sq) != 0` asked of many squares at once, and the
/// reason to prefer it on a hot path: 512 bytes that stay in L1 against 32 KB of
/// two-index lookups. Emptying a square outside this mask cannot uncover a slider
/// bearing on `sq`.
#[inline(always)]
pub fn aligned_bb(sq: Square) -> u64 {
    ALIGNED_BB[sq.index()]
}

/// All pieces of either color attacking a square (used by SEE in Phase 3).
#[allow(dead_code)]
pub fn attackers_to(sq: Square, occupied: u64, pieces: &[u64; 12]) -> u64 {
    let knights = pieces[Piece::WHITE_KNIGHT.index()] | pieces[Piece::BLACK_KNIGHT.index()];
    let bishops = pieces[Piece::WHITE_BISHOP.index()] | pieces[Piece::BLACK_BISHOP.index()];
    let rooks = pieces[Piece::WHITE_ROOK.index()] | pieces[Piece::BLACK_ROOK.index()];
    let queens = pieces[Piece::WHITE_QUEEN.index()] | pieces[Piece::BLACK_QUEEN.index()];
    let kings = pieces[Piece::WHITE_KING.index()] | pieces[Piece::BLACK_KING.index()];

    (knight_attacks(sq) & knights)
        | (king_attacks(sq) & kings)
        | (pawn_attacks(sq, Color::White) & pieces[Piece::BLACK_PAWN.index()])
        | (pawn_attacks(sq, Color::Black) & pieces[Piece::WHITE_PAWN.index()])
        | (bishop_attacks(sq, occupied) & (bishops | queens))
        | (rook_attacks(sq, occupied) & (rooks | queens))
}

/// Attackers of a specific color to a square
pub fn attackers_to_color(sq: Square, c: Color, occupied: u64, pieces: &[u64; 12]) -> u64 {
    let them = c;
    (knight_attacks(sq) & pieces[Piece::new(PieceType::Knight, them).index()])
        | (king_attacks(sq) & pieces[Piece::new(PieceType::King, them).index()])
        | (pawn_attacks(sq, !them) & pieces[Piece::new(PieceType::Pawn, them).index()])
        | (bishop_attacks(sq, occupied)
            & (pieces[Piece::new(PieceType::Bishop, them).index()]
                | pieces[Piece::new(PieceType::Queen, them).index()]))
        | (rook_attacks(sq, occupied)
            & (pieces[Piece::new(PieceType::Rook, them).index()]
                | pieces[Piece::new(PieceType::Queen, them).index()]))
}

// ============================================================
// Magic numbers (plain magics: shift = 64 - popcount(mask), exhaustively validated)
// ============================================================

#[cfg(not(target_feature = "avx2"))]
#[rustfmt::skip]
const ROOK_MAGICS: [MagicEntry; 64] = [
    MagicEntry { mask: 0x000101010101017E, magic: 0x6280008040002210, shift: 52, offset: 0 },
    MagicEntry { mask: 0x000202020202027C, magic: 0x0900108040002108, shift: 53, offset: 4096 },
    MagicEntry { mask: 0x000404040404047A, magic: 0x4100110240082000, shift: 53, offset: 6144 },
    MagicEntry { mask: 0x0008080808080876, magic: 0x4100081000210004, shift: 53, offset: 8192 },
    MagicEntry { mask: 0x001010101010106E, magic: 0x1080080002040080, shift: 53, offset: 10240 },
    MagicEntry { mask: 0x002020202020205E, magic: 0x0100080400010002, shift: 53, offset: 12288 },
    MagicEntry { mask: 0x004040404040403E, magic: 0xBA00110800842200, shift: 53, offset: 14336 },
    MagicEntry { mask: 0x008080808080807E, magic: 0x02800140A0800300, shift: 52, offset: 16384 },
    MagicEntry { mask: 0x0001010101017E00, magic: 0x0208800080400020, shift: 53, offset: 20480 },
    MagicEntry { mask: 0x0002020202027C00, magic: 0x6001804002200080, shift: 54, offset: 22528 },
    MagicEntry { mask: 0x0004040404047A00, magic: 0x0440801008200080, shift: 54, offset: 23552 },
    MagicEntry { mask: 0x0008080808087600, magic: 0x0040800800100080, shift: 54, offset: 24576 },
    MagicEntry { mask: 0x0010101010106E00, magic: 0x2222000820100600, shift: 54, offset: 25600 },
    MagicEntry { mask: 0x0020202020205E00, magic: 0x002A000804900201, shift: 54, offset: 26624 },
    MagicEntry { mask: 0x0040404040403E00, magic: 0x018B000401001200, shift: 54, offset: 27648 },
    MagicEntry { mask: 0x0080808080807E00, magic: 0x418200040590430A, shift: 53, offset: 28672 },
    MagicEntry { mask: 0x00010101017E0100, magic: 0x0050888000400024, shift: 53, offset: 30720 },
    MagicEntry { mask: 0x00020202027C0200, magic: 0xA100808040002008, shift: 54, offset: 32768 },
    MagicEntry { mask: 0x00040404047A0400, magic: 0x1200410011002000, shift: 54, offset: 33792 },
    MagicEntry { mask: 0x0008080808760800, magic: 0x0010028010820800, shift: 54, offset: 34816 },
    MagicEntry { mask: 0x00101010106E1000, magic: 0x4149110005010800, shift: 54, offset: 35840 },
    MagicEntry { mask: 0x00202020205E2000, magic: 0x6904004002004100, shift: 54, offset: 36864 },
    MagicEntry { mask: 0x00404040403E4000, magic: 0x1000808002000100, shift: 54, offset: 37888 },
    MagicEntry { mask: 0x00808080807E8000, magic: 0x4044020020841049, shift: 53, offset: 38912 },
    MagicEntry { mask: 0x000101017E010100, magic: 0x0880004140002000, shift: 53, offset: 40960 },
    MagicEntry { mask: 0x000202027C020200, magic: 0x4056008200250840, shift: 54, offset: 43008 },
    MagicEntry { mask: 0x000404047A040400, magic: 0x0200200080801000, shift: 54, offset: 44032 },
    MagicEntry { mask: 0x0008080876080800, magic: 0x0080D00480080080, shift: 54, offset: 45056 },
    MagicEntry { mask: 0x001010106E101000, magic: 0x2031008500100800, shift: 54, offset: 46080 },
    MagicEntry { mask: 0x002020205E202000, magic: 0x0842020080040080, shift: 54, offset: 47104 },
    MagicEntry { mask: 0x004040403E404000, magic: 0x1000010400020810, shift: 54, offset: 48128 },
    MagicEntry { mask: 0x008080807E808000, magic: 0x001080A200140041, shift: 53, offset: 49152 },
    MagicEntry { mask: 0x0001017E01010100, magic: 0x1C8000A000C00040, shift: 53, offset: 51200 },
    MagicEntry { mask: 0x0002027C02020200, magic: 0x0430002000400048, shift: 54, offset: 53248 },
    MagicEntry { mask: 0x0004047A04040400, magic: 0x4000200288801002, shift: 54, offset: 54272 },
    MagicEntry { mask: 0x0008087608080800, magic: 0x0000801000800800, shift: 54, offset: 55296 },
    MagicEntry { mask: 0x0010106E10101000, magic: 0x2020040080800801, shift: 54, offset: 56320 },
    MagicEntry { mask: 0x0020205E20202000, magic: 0x0000800400800200, shift: 54, offset: 57344 },
    MagicEntry { mask: 0x0040403E40404000, magic: 0x0182820164001028, shift: 54, offset: 58368 },
    MagicEntry { mask: 0x0080807E80808000, magic: 0x4002800040800100, shift: 53, offset: 59392 },
    MagicEntry { mask: 0x00017E0101010100, magic: 0x0802004100820020, shift: 53, offset: 61440 },
    MagicEntry { mask: 0x00027C0202020200, magic: 0x0010084020044000, shift: 54, offset: 63488 },
    MagicEntry { mask: 0x00047A0404040400, magic: 0x0820040200101000, shift: 54, offset: 64512 },
    MagicEntry { mask: 0x0008760808080800, magic: 0x8109001000890021, shift: 54, offset: 65536 },
    MagicEntry { mask: 0x00106E1010101000, magic: 0x1308008004008008, shift: 54, offset: 66560 },
    MagicEntry { mask: 0x00205E2020202000, magic: 0x0282008004008002, shift: 54, offset: 67584 },
    MagicEntry { mask: 0x00403E4040404000, magic: 0x8800020001008080, shift: 54, offset: 68608 },
    MagicEntry { mask: 0x00807E8080808000, magic: 0x2200108C004A0001, shift: 53, offset: 69632 },
    MagicEntry { mask: 0x007E010101010100, magic: 0x0402400080002180, shift: 53, offset: 71680 },
    MagicEntry { mask: 0x007C020202020200, magic: 0x1010200B80401080, shift: 54, offset: 73728 },
    MagicEntry { mask: 0x007A040404040400, magic: 0x4042411508200100, shift: 54, offset: 74752 },
    MagicEntry { mask: 0x0076080808080800, magic: 0x60401220C10A0200, shift: 54, offset: 75776 },
    MagicEntry { mask: 0x006E101010101000, magic: 0x00A0800800040080, shift: 54, offset: 76800 },
    MagicEntry { mask: 0x005E202020202000, magic: 0x0082000280040080, shift: 54, offset: 77824 },
    MagicEntry { mask: 0x003E404040404000, magic: 0x4088410802100400, shift: 54, offset: 78848 },
    MagicEntry { mask: 0x007E808080808000, magic: 0x08410000A2004100, shift: 53, offset: 79872 },
    MagicEntry { mask: 0x7E01010101010100, magic: 0x0031042202804012, shift: 52, offset: 81920 },
    MagicEntry { mask: 0x7C02020202020200, magic: 0x2001022208401082, shift: 53, offset: 86016 },
    MagicEntry { mask: 0x7A04040404040400, magic: 0x2040400900200011, shift: 53, offset: 88064 },
    MagicEntry { mask: 0x7608080808080800, magic: 0x0010000411000821, shift: 53, offset: 90112 },
    MagicEntry { mask: 0x6E10101010101000, magic: 0x0082000810042002, shift: 53, offset: 92160 },
    MagicEntry { mask: 0x5E20202020202000, magic: 0x009100280A940005, shift: 53, offset: 94208 },
    MagicEntry { mask: 0x3E40404040404000, magic: 0x00002200C8100104, shift: 53, offset: 96256 },
    MagicEntry { mask: 0x7E80808080808000, magic: 0x0210005020810402, shift: 52, offset: 98304 },
];

#[cfg(not(target_feature = "avx2"))]
#[rustfmt::skip]
const BISHOP_MAGICS: [MagicEntry; 64] = [
    MagicEntry { mask: 0x0040201008040200, magic: 0x0110101000808810, shift: 58, offset: 0 },
    MagicEntry { mask: 0x0000402010080400, magic: 0x04040800A1060000, shift: 59, offset: 64 },
    MagicEntry { mask: 0x0000004020100A00, magic: 0x1610314441000200, shift: 59, offset: 96 },
    MagicEntry { mask: 0x0000000040221400, magic: 0x5808205042000003, shift: 59, offset: 128 },
    MagicEntry { mask: 0x0000000002442800, magic: 0x00011040400200C0, shift: 59, offset: 160 },
    MagicEntry { mask: 0x0000000204085000, magic: 0x08020884040C8400, shift: 59, offset: 192 },
    MagicEntry { mask: 0x0000020408102000, magic: 0x002A0A0164410020, shift: 59, offset: 224 },
    MagicEntry { mask: 0x0002040810204000, magic: 0x2803010082014000, shift: 58, offset: 256 },
    MagicEntry { mask: 0x0020100804020000, magic: 0x300020204C908680, shift: 59, offset: 320 },
    MagicEntry { mask: 0x0040201008040000, magic: 0x0055480288020C21, shift: 59, offset: 352 },
    MagicEntry { mask: 0x00004020100A0000, magic: 0x2001140906021A00, shift: 59, offset: 384 },
    MagicEntry { mask: 0x0000004022140000, magic: 0x0445044040870040, shift: 59, offset: 416 },
    MagicEntry { mask: 0x0000000244280000, magic: 0x8208611040000004, shift: 59, offset: 448 },
    MagicEntry { mask: 0x0000020408500000, magic: 0x2000011002500010, shift: 59, offset: 480 },
    MagicEntry { mask: 0x0002040810200000, magic: 0x4412010101104211, shift: 59, offset: 512 },
    MagicEntry { mask: 0x0004081020400000, magic: 0x0219002108080400, shift: 59, offset: 544 },
    MagicEntry { mask: 0x0010080402000200, magic: 0x8044000808184840, shift: 59, offset: 576 },
    MagicEntry { mask: 0x0020100804000400, magic: 0x00B000600408F080, shift: 59, offset: 608 },
    MagicEntry { mask: 0x004020100A000A00, magic: 0x0090082200204302, shift: 57, offset: 640 },
    MagicEntry { mask: 0x0000402214001400, magic: 0x0084208812002001, shift: 57, offset: 768 },
    MagicEntry { mask: 0x0000024428002800, magic: 0x4022018C00940800, shift: 57, offset: 896 },
    MagicEntry { mask: 0x0002040850005000, magic: 0x0808210A009C2001, shift: 57, offset: 1024 },
    MagicEntry { mask: 0x0004081020002000, magic: 0x4003120208023240, shift: 59, offset: 1152 },
    MagicEntry { mask: 0x0008102040004000, magic: 0x1040800202012100, shift: 59, offset: 1184 },
    MagicEntry { mask: 0x0008040200020400, magic: 0x0804400820824420, shift: 59, offset: 1216 },
    MagicEntry { mask: 0x0010080400040800, magic: 0x2021500004440800, shift: 59, offset: 1248 },
    MagicEntry { mask: 0x0020100A000A1000, magic: 0x0004100002042040, shift: 57, offset: 1280 },
    MagicEntry { mask: 0x0040221400142200, magic: 0x0001004004004200, shift: 55, offset: 1408 },
    MagicEntry { mask: 0x0002442800284400, magic: 0x0A01010011104000, shift: 55, offset: 1920 },
    MagicEntry { mask: 0x0004085000500800, magic: 0x0008004104806000, shift: 57, offset: 2432 },
    MagicEntry { mask: 0x0008102000201000, magic: 0x0004008004121942, shift: 59, offset: 2560 },
    MagicEntry { mask: 0x0010204000402000, magic: 0x803A024910885802, shift: 59, offset: 2592 },
    MagicEntry { mask: 0x0004020002040800, magic: 0x101008A00808020C, shift: 59, offset: 2624 },
    MagicEntry { mask: 0x0008040004081000, magic: 0x1019082004222480, shift: 59, offset: 2656 },
    MagicEntry { mask: 0x00100A000A102000, magic: 0x0002009008120020, shift: 57, offset: 2688 },
    MagicEntry { mask: 0x0022140014224000, magic: 0x0001020082080080, shift: 55, offset: 2816 },
    MagicEntry { mask: 0x0044280028440200, magic: 0x0006088400860020, shift: 55, offset: 3328 },
    MagicEntry { mask: 0x0008500050080400, magic: 0x4020080180003200, shift: 57, offset: 3840 },
    MagicEntry { mask: 0x0010200020100800, magic: 0x8102040C00004200, shift: 59, offset: 3968 },
    MagicEntry { mask: 0x0020400040201000, magic: 0x0824008210002100, shift: 59, offset: 4000 },
    MagicEntry { mask: 0x0002000204081000, magic: 0x0922021040000400, shift: 59, offset: 4032 },
    MagicEntry { mask: 0x0004000408102000, magic: 0x000C544404012002, shift: 59, offset: 4064 },
    MagicEntry { mask: 0x000A000A10204000, magic: 0x0009802804410800, shift: 57, offset: 4096 },
    MagicEntry { mask: 0x0014001422400000, magic: 0x0420002018000100, shift: 57, offset: 4224 },
    MagicEntry { mask: 0x0028002844020000, magic: 0x0000200341280400, shift: 57, offset: 4352 },
    MagicEntry { mask: 0x0050005008040200, magic: 0x0040210040800100, shift: 57, offset: 4480 },
    MagicEntry { mask: 0x0020002010080400, magic: 0x8020110216000080, shift: 59, offset: 4608 },
    MagicEntry { mask: 0x0040004020100800, magic: 0x00A8012410800020, shift: 59, offset: 4640 },
    MagicEntry { mask: 0x0000020408102000, magic: 0x0014008404A24004, shift: 59, offset: 4672 },
    MagicEntry { mask: 0x0000040810204000, magic: 0x0000208404204003, shift: 59, offset: 4704 },
    MagicEntry { mask: 0x00000A1020400000, magic: 0x3000002201100460, shift: 59, offset: 4736 },
    MagicEntry { mask: 0x0000142240000000, magic: 0x0000401284110100, shift: 59, offset: 4768 },
    MagicEntry { mask: 0x0000284402000000, magic: 0x0A02006020490000, shift: 59, offset: 4800 },
    MagicEntry { mask: 0x0000500804020000, magic: 0x4006113050012000, shift: 59, offset: 4832 },
    MagicEntry { mask: 0x0000201008040200, magic: 0x340E450808006052, shift: 59, offset: 4864 },
    MagicEntry { mask: 0x0000402010080400, magic: 0x1020025202022040, shift: 59, offset: 4896 },
    MagicEntry { mask: 0x0002040810204000, magic: 0x0100440218020288, shift: 58, offset: 4928 },
    MagicEntry { mask: 0x0004081020400000, magic: 0x001020460884200A, shift: 59, offset: 4992 },
    MagicEntry { mask: 0x000A102040000000, magic: 0x0808100A06460800, shift: 59, offset: 5024 },
    MagicEntry { mask: 0x0014224000000000, magic: 0x0880108280208846, shift: 59, offset: 5056 },
    MagicEntry { mask: 0x0028440200000000, magic: 0x4900002020042404, shift: 59, offset: 5088 },
    MagicEntry { mask: 0x0050080402000000, magic: 0x0500004822080204, shift: 59, offset: 5120 },
    MagicEntry { mask: 0x0020100804020000, magic: 0x0A90044830012200, shift: 59, offset: 5152 },
    MagicEntry { mask: 0x0040201008040200, magic: 0x0008021002042300, shift: 58, offset: 5184 },
];

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knight_attacks() {
        // E4 knight should have 8 attack squares
        let attacks = knight_attacks(Square::E4);
        assert_eq!(popcount(attacks), 8);
        // A1 corner: 2 attacks
        let attacks = knight_attacks(Square::A1);
        assert_eq!(popcount(attacks), 2);
    }

    #[test]
    fn test_king_attacks() {
        // E4: 8 attacks
        assert_eq!(popcount(king_attacks(Square::E4)), 8);
        // A1 corner: 3 attacks
        assert_eq!(popcount(king_attacks(Square::A1)), 3);
        // Edge: 5 attacks
        assert_eq!(popcount(king_attacks(Square::A4)), 5);
    }

    #[test]
    fn test_pawn_attacks() {
        let attacks = pawn_attacks(Square::E2, Color::White);
        assert!(attacks & Square::D3.bb() != 0);
        assert!(attacks & Square::F3.bb() != 0);
        assert_eq!(popcount(attacks), 2);

        // A-file pawn: only one attack
        let attacks = pawn_attacks(Square::A2, Color::White);
        assert_eq!(popcount(attacks), 1);
        assert!(attacks & Square::B3.bb() != 0);
    }

    #[test]
    fn test_rook_attacks() {
        // Rook on A1, empty board
        let attacks = rook_attacks(Square::A1, 0);
        assert_eq!(popcount(attacks), 14);

        // Rook on E4, empty board
        let attacks = rook_attacks(Square::E4, 0);
        assert_eq!(popcount(attacks), 14);

        // Rook on A1, blocked by piece on A4
        let occ = Square::A4.bb();
        let attacks = rook_attacks(Square::A1, occ);
        assert!(attacks & Square::A4.bb() != 0); // can capture blocker
        assert!(attacks & Square::A5.bb() == 0); // blocked beyond
    }

    #[test]
    fn test_bishop_attacks() {
        // Bishop on A1, empty board (7 squares on long diagonal)
        let attacks = bishop_attacks(Square::A1, 0);
        assert_eq!(popcount(attacks), 7);

        // Bishop on E4, empty board (13 squares)
        let attacks = bishop_attacks(Square::E4, 0);
        assert_eq!(popcount(attacks), 13);
    }

    #[test]
    fn test_between_bb() {
        // Between A1 and H8 (long diagonal) should include A1-H8 intermediate squares
        let between = between_bb(Square::A1, Square::H8);
        // Should include B2,C3,D4,E5,F6,G7 and H8 (destination)
        assert!(between & Square::B2.bb() != 0);
        assert!(between & Square::D4.bb() != 0);
        assert!(between & Square::H8.bb() != 0); // includes destination
        assert!(between & Square::A1.bb() == 0); // excludes source

        // Between A1 and A8 (same file)
        let between = between_bb(Square::A1, Square::A8);
        assert!(between & Square::A4.bb() != 0);
        assert!(between & Square::A8.bb() != 0);
    }

    /// The mask is folded out of `LINE_BB`, so it is worth holding to a definition
    /// that owes nothing to that table: two squares are aligned when they share a
    /// file, a rank, or a diagonal.
    #[test]
    fn aligned_squares_are_the_ones_sharing_a_file_rank_or_diagonal() {
        for s in 0..64u8 {
            let sq = Square(s);
            for t in 0..64u8 {
                let other = Square(t);
                let df = sq.file() as i32 - other.file() as i32;
                let dr = sq.rank() as i32 - other.rank() as i32;
                let aligned = df == 0 || dr == 0 || df.abs() == dr.abs();
                assert_eq!(
                    aligned_bb(sq) & other.bb() != 0,
                    aligned,
                    "squares {s} and {t}",
                );
            }
        }
    }

    #[test]
    fn test_shift_no_wrap() {
        // Shifting H-file east should produce nothing
        assert_eq!(shift_east(FILE_H), 0);
        // Shifting A-file west should produce nothing
        assert_eq!(shift_west(FILE_A), 0);
    }

    /// The compile-time PEXT tables answer exactly what the ray walker answers, for
    /// every square and a spread of sparse and dense occupancies.
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    #[test]
    fn the_pext_tables_agree_with_the_ray_walker() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        for sq in 0..64u8 {
            for _ in 0..512 {
                let occ = match next() % 3 {
                    0 => next(),
                    1 => next() & next(),
                    _ => next() & next() & next(),
                };
                let square = Square(sq);
                // SAFETY: the test only runs on a build target that has BMI2.
                let (rook, bishop) = unsafe {
                    (super::pext::rook_attacks(square, occ), super::pext::bishop_attacks(square, occ))
                };
                assert_eq!(rook, sliding_attack_otf(sq, occ, &ROOK_DELTAS), "rook from {sq} on {occ:#x}");
                assert_eq!(bishop, sliding_attack_otf(sq, occ, &BISHOP_DELTAS), "bishop from {sq} on {occ:#x}");
            }
        }
    }
}
