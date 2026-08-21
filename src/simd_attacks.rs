//! SIMD-accelerated sliding piece attack generation.
//!
//! Two complementary approaches, conditionally compiled via `#[cfg(target_feature)]`:
//!
//! 1. **AVX2 per-piece** (CFish approach): drop-in replacement for magic bitboard lookups.
//!    Uses the byte-swap trick with [`BLSMSK`](https://www.chessprogramming.org/BMI1#BLSMSK)
//!    to process all ray directions as positive rays. ~7.5 KB tables vs ~850 KB for magics.
//!
//! 2. **AVX-512 setwise**: computes all slider attacks simultaneously
//!    via [Kogge-Stone](https://www.chessprogramming.org/Kogge-Stone_Algorithm) parallel
//!    prefix fill in a 512-bit register (8 directions x 8 lanes).
//!
//! Magic bitboards remain the fallback when compiled without AVX2.

#![allow(unused_unsafe)]
#![allow(dead_code)]
#![allow(unused_imports)]

use crate::bitboard::*;
use crate::types::*;
use std::sync::OnceLock;

// ============================================================
// Direction deltas for ray generation (matching A1=0 convention)
// ============================================================

const DIR_EAST: i8 = 1;
const DIR_NORTH: i8 = 8;
const DIR_NORTH_EAST: i8 = 9;
const DIR_NORTH_WEST: i8 = 7;
const DIR_WEST: i8 = -1;
const DIR_SOUTH: i8 = -8;
const DIR_SOUTH_WEST: i8 = -9;
const DIR_SOUTH_EAST: i8 = -7;

// ============================================================
// Table data — stored as plain arrays for Send+Sync compatibility
// Loaded into SIMD registers at call time via _mm256_loadu_si256
// ============================================================

struct AvxTableData {
    // Queen: 4 positive rays (East, North, NE, NW) per square
    queen_mask_left: [[i64; 4]; 64],
    // Queen: 4 negative rays (West, South, SW, SE) per square
    queen_mask_right: [[i64; 4]; 64],
    // Bishop: 4 diagonals arranged for byte-swap trick (NE, flipped-SW, flipped-SE, NW)
    bishop_mask: [[i64; 4]; 64],
    // Rook N/S: 2 rays arranged for byte-swap trick (North, flipped-South)
    rook_mask_ns: [[i64; 2]; 64],
    // Rook E/W: precomputed rank attack table (6-bit inner occupancy × 8 files)
    rook_attacks_ew: [u8; 512],
}

// Safety: AvxTableData is only written once during init() and then read-only
unsafe impl Send for AvxTableData {}
unsafe impl Sync for AvxTableData {}

static AVX_TABLES: OnceLock<AvxTableData> = OnceLock::new();

// ============================================================
// Initialization
// ============================================================

/// Initialize SIMD attack tables. Must be called before any SIMD attack function.
/// Safe to call multiple times (OnceLock ensures single init).
pub fn init() {
    AVX_TABLES.get_or_init(|| {
        let mut data = AvxTableData {
            queen_mask_left: [[0i64; 4]; 64],
            queen_mask_right: [[0i64; 4]; 64],
            bishop_mask: [[0i64; 4]; 64],
            rook_mask_ns: [[0i64; 2]; 64],
            rook_attacks_ew: [0u8; 512],
        };
        init_tables(&mut data);
        data
    });
}

/// Compute a single directional ray from a square (excluding the square itself).
/// Returns all squares along the ray until the board edge.
fn ray_mask(sq: u8, dir: i8) -> u64 {
    let mut mask = 0u64;
    let mut s = sq as i8 + dir;
    while (0..64).contains(&s) {
        let file_diff = ((s & 7) - ((s - dir) & 7)).abs();
        if file_diff > 2 {
            break; // wrapped around a/h file
        }
        mask |= 1u64 << (s as u32);
        s += dir;
    }
    mask
}

fn init_tables(data: &mut AvxTableData) {
    // "Left" = positive direction rays (towards higher bits for that direction)
    // CFish ordering: [East, North, NE, NW] for left, [West, South, SW, SE] for right
    let left_dirs = [DIR_EAST, DIR_NORTH, DIR_NORTH_EAST, DIR_NORTH_WEST];
    let right_dirs = [DIR_WEST, DIR_SOUTH, DIR_SOUTH_WEST, DIR_SOUTH_EAST];

    for sq in 0u8..64 {
        // Queen masks: left (positive) and right (negative) rays
        for (i, &dir) in left_dirs.iter().enumerate() {
            data.queen_mask_left[sq as usize][i] = ray_mask(sq, dir) as i64;
        }
        for (i, &dir) in right_dirs.iter().enumerate() {
            data.queen_mask_right[sq as usize][i] = ray_mask(sq, dir) as i64;
        }

        // Bishop masks for byte-swap trick:
        // _mm256_broadcastsi128 gives: lane0=orig, lane1=swapped, lane2=orig, lane3=swapped
        // Even lanes use normal masks with original occupancy.
        // Odd lanes use masks in FLIPPED coordinates (NOT un-flipped!) matching the
        // byte-swapped occupancy. The swaph2l fold step will un-flip the results.
        let flipped_sq = sq ^ 56; // flip_rank
        data.bishop_mask[sq as usize] = [
            ray_mask(sq, DIR_NORTH_EAST) as i64,              // lane 0: NE (orig occ, normal coords)
            ray_mask(flipped_sq, DIR_NORTH_WEST) as i64,      // lane 1: SW-as-NW-from-flip (swapped occ, flipped coords)
            ray_mask(sq, DIR_NORTH_WEST) as i64,              // lane 2: NW (orig occ, normal coords)
            ray_mask(flipped_sq, DIR_NORTH_EAST) as i64,      // lane 3: SE-as-NE-from-flip (swapped occ, flipped coords)
        ];

        // Rook N/S masks for byte-swap trick:
        // Lane 0 (orig occ): North ray in normal coords
        // Lane 1 (swapped occ): South-as-North-from-flip in flipped coords
        // (no flip_vertical — stays in flipped coords to match byte-swapped occupancy;
        //  swaph2l fold step will un-flip the result)
        data.rook_mask_ns[sq as usize] = [
            ray_mask(sq, DIR_NORTH) as i64,
            ray_mask(flipped_sq, DIR_NORTH) as i64,
        ];
    }

    // Rook E/W rank attack table
    // Indexed by: (inner_6bit_occ * 4 + file)
    // inner_6bit_occ = (rank_occupancy >> 1) & 0x3F, but CFish stores it as (occ & 0x7E)
    // which is the same bits just not shifted: index = (occ & 0x7E) * 4 + file
    for occ8 in (0u8..128).step_by(2) {
        // occ8 represents the inner 6 bits of rank occupancy (bits 1-6)
        for file in 0u8..8 {
            let mut att = 0u8;
            // Scan east
            let mut s = (1u8 << file).checked_shl(1).unwrap_or(0);
            while s != 0 {
                att |= s;
                if occ8 & s != 0 {
                    break;
                }
                s <<= 1;
            }
            // Scan west
            s = (1u8 << file) >> 1;
            while s != 0 {
                att |= s;
                if occ8 & s != 0 {
                    break;
                }
                s >>= 1;
            }
            data.rook_attacks_ew[(occ8 as usize) * 4 + file as usize] = att;
        }
    }
}

#[inline(always)]
fn tables() -> &'static AvxTableData {
    AVX_TABLES.get_or_init(|| {
        let mut data = AvxTableData {
            queen_mask_left: [[0i64; 4]; 64],
            queen_mask_right: [[0i64; 4]; 64],
            bishop_mask: [[0i64; 4]; 64],
            rook_mask_ns: [[0i64; 2]; 64],
            rook_attacks_ew: [0u8; 512],
        };
        init_tables(&mut data);
        data
    })
}

// ============================================================
// AVX2 per-piece attack functions (CFish approach)
// ============================================================

#[cfg(target_feature = "avx2")]
mod avx2 {
    use super::*;
    use std::arch::x86_64::*;

    // BLSMSK on 4 packed i64: (x-1) ^ x
    #[inline(always)]
    fn blsmsk64x4(y: __m256i) -> __m256i {
        unsafe { _mm256_xor_si256(_mm256_add_epi64(y, _mm256_set1_epi64x(-1)), y) }
    }

    // BLSMSK on 2 packed i64
    #[inline(always)]
    fn blsmsk64x2(x: __m128i) -> __m128i {
        unsafe { _mm_xor_si128(_mm_add_epi64(x, _mm_set1_epi64x(-1)), x) }
    }

    #[inline(always)]
    fn swap_l2h() -> __m128i {
        unsafe { _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0) }
    }

    #[inline(always)]
    fn swap_h2l() -> __m128i {
        unsafe { _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 8, 9, 10, 11, 12, 13, 14, 15) }
    }

    /// Bishop attacks via BLSMSK + byte-swap (all 4 diagonals as positive rays)
    #[inline(always)]
    pub fn bishop_attacks(sq: Square, occupied: u64) -> u64 {
        unsafe {
            let t = super::tables();
            let occupied2 = _mm_shuffle_epi8(_mm_cvtsi64_si128(occupied as i64), swap_l2h());
            let occupied4 = _mm256_broadcastsi128_si256(occupied2);
            let mask = _mm256_loadu_si256(t.bishop_mask[sq.index()].as_ptr() as *const __m256i);
            let slide4 = _mm256_and_si256(blsmsk64x4(_mm256_and_si256(occupied4, mask)), mask);
            let slide2 = _mm_or_si128(
                _mm256_castsi256_si128(slide4),
                _mm256_extracti128_si256::<1>(slide4),
            );
            let result = _mm_or_si128(slide2, _mm_shuffle_epi8(slide2, swap_h2l()));
            _mm_cvtsi128_si64(result) as u64
        }
    }

    /// Rook attacks: N/S via byte-swap + BLSMSK, E/W via small lookup table
    #[inline(always)]
    pub fn rook_attacks(sq: Square, occupied: u64) -> u64 {
        let t = super::tables();
        let ns = unsafe {
            let occupied2 = _mm_shuffle_epi8(_mm_cvtsi64_si128(occupied as i64), swap_l2h());
            let mask = _mm_loadu_si128(t.rook_mask_ns[sq.index()].as_ptr() as *const __m128i);
            let slide2 = _mm_and_si128(blsmsk64x2(_mm_and_si128(occupied2, mask)), mask);
            let ns_result = _mm_or_si128(slide2, _mm_shuffle_epi8(slide2, swap_h2l()));
            _mm_cvtsi128_si64(ns_result) as u64
        };
        let r8 = sq.rank() as usize * 8;
        let occ_rank = ((occupied >> r8) & 0x7E) as usize;
        let ew = t.rook_attacks_ew[occ_rank * 4 + sq.file() as usize] as u64;
        ns | (ew << r8)
    }

    /// Queen attacks: 4 positive rays (BLSMSK) + 4 negative rays (PP-fill or lzcnt)
    #[inline(always)]
    pub fn queen_attacks(sq: Square, occupied: u64) -> u64 {
        unsafe {
            let t = super::tables();
            let occupied4 = _mm256_set1_epi64x(occupied as i64);
            let lmask =
                _mm256_loadu_si256(t.queen_mask_left[sq.index()].as_ptr() as *const __m256i);
            let rmask =
                _mm256_loadu_si256(t.queen_mask_right[sq.index()].as_ptr() as *const __m256i);

            // Left/positive rays: BLSMSK approach
            let left = _mm256_and_si256(blsmsk64x4(_mm256_and_si256(occupied4, lmask)), lmask);

            // Right/negative rays
            let slide4;

            #[cfg(all(target_feature = "avx512cd", target_feature = "avx512vl"))]
            {
                let rmasked = _mm256_and_si256(occupied4, rmask);
                let lzcnt = _mm256_lzcnt_epi64(rmasked);
                let rslide = _mm256_srav_epi64(_mm256_set1_epi64x(i64::MIN), lzcnt);
                slide4 = _mm256_ternarylogic_epi64(left, rslide, rmask, 0xf8);
            }

            #[cfg(not(all(target_feature = "avx512cd", target_feature = "avx512vl")))]
            {
                let shifts_1 = _mm256_set_epi64x(7, 9, 8, 1);
                let shifts_2 = _mm256_set_epi64x(14, 18, 16, 2);
                let shifts_4 = _mm256_set_epi64x(28, 36, 32, 4);

                let mut rslide = _mm256_and_si256(occupied4, rmask);
                rslide = _mm256_or_si256(
                    _mm256_srlv_epi64(rslide, shifts_2),
                    _mm256_srlv_epi64(rslide, shifts_1),
                );
                rslide = _mm256_or_si256(
                    _mm256_srlv_epi64(rslide, shifts_4),
                    _mm256_or_si256(rslide, _mm256_srlv_epi64(rslide, shifts_2)),
                );
                slide4 = _mm256_or_si256(left, _mm256_andnot_si256(rslide, rmask));
            }

            let slide2 = _mm_or_si128(
                _mm256_castsi256_si128(slide4),
                _mm256_extracti128_si256::<1>(slide4),
            );
            _mm_cvtsi128_si64(_mm_or_si128(slide2, _mm_unpackhi_epi64(slide2, slide2))) as u64
        }
    }
}

// ============================================================
// AVX-512 setwise slider attacks (Kogge-Stone approach)
// ============================================================

#[cfg(target_arch = "x86_64")]
mod avx512 {
    use super::*;
    use std::arch::x86_64::*;

    /// Compute aggregated attacks of ALL sliders of one color simultaneously.
    /// All 8 ray directions are processed in parallel in a single 512-bit register.
    #[cfg_attr(
        not(target_feature = "avx512f"),
        target_feature(enable = "avx512f")
    )]
    #[inline]
    pub unsafe fn slider_attacks_setwise(
        bishops: u64,
        rooks: u64,
        queens: u64,
        occupied: u64,
    ) -> u64 {
        unsafe {
            // Pack attackers: lanes 0-3 = diagonal (bishops|queens), lanes 4-7 = orthogonal (rooks|queens)
            let attackers = _mm512_mask_blend_epi64(
                0x0F,
                _mm512_set1_epi64((rooks | queens) as i64),
                _mm512_set1_epi64((bishops | queens) as i64),
            );

            // Rotation amounts per direction (mod 64)
            // Lanes: [soEa(-7), soWe(-9), noWe(+7), noEa(+9), south(-8), west(-1), east(+1), north(+8)]
            let rotates1 = _mm512_set_epi64(-8, -1, 1, 8, -9, -7, 7, 9);
            let rotates2 = _mm512_add_epi64(rotates1, rotates1);
            let rotates4 = _mm512_add_epi64(rotates2, rotates2);

            // Directional masks to prevent wrap-around
            let masks = _mm512_set_epi64(
                !RANK_8 as i64,                  // lane 7: south — wraps rank 1→8
                !FILE_H as i64,                  // lane 6: west — wraps A→H
                !FILE_A as i64,                  // lane 5: east — wraps H→A
                !RANK_1 as i64,                  // lane 4: north — wraps rank 8→1
                (!RANK_8 & !FILE_H) as i64,      // lane 3: soWe
                (!RANK_8 & !FILE_A) as i64,      // lane 2: soEa
                (!RANK_1 & !FILE_H) as i64,      // lane 1: noWe
                (!RANK_1 & !FILE_A) as i64,      // lane 0: noEa
            );

            // Kogge-Stone parallel prefix fill
            let mut propagate = _mm512_and_si512(_mm512_set1_epi64(!occupied as i64), masks);
            let mut generate = attackers;

            // Iteration 1: shift by 1 step
            generate = _mm512_or_si512(
                generate,
                _mm512_and_si512(propagate, _mm512_rolv_epi64(generate, rotates1)),
            );
            propagate = _mm512_and_si512(propagate, _mm512_rolv_epi64(propagate, rotates1));

            // Iteration 2: shift by 2 steps
            generate = _mm512_or_si512(
                generate,
                _mm512_and_si512(propagate, _mm512_rolv_epi64(generate, rotates2)),
            );
            propagate = _mm512_and_si512(propagate, _mm512_rolv_epi64(propagate, rotates2));

            // Iteration 3: shift by 4 steps
            generate = _mm512_or_si512(
                generate,
                _mm512_and_si512(propagate, _mm512_rolv_epi64(generate, rotates4)),
            );

            // Final shift to get attacked squares (excludes pieces themselves)
            let attacks = _mm512_and_si512(_mm512_rolv_epi64(generate, rotates1), masks);

            fold_512(attacks)
        }
    }

    #[inline(always)]
    fn fold_512(attacks: __m512i) -> u64 {
        unsafe {
            #[cfg(all(
                target_feature = "avx512bw",
                target_feature = "avx512vbmi",
                target_feature = "gfni"
            ))]
            {
                let attacks = _mm512_gf2p8affine_epi64_epi8(
                    _mm512_set1_epi64(0x8040201008040201u64 as i64),
                    _mm512_permutexvar_epi8(
                        _mm512_set_epi8(
                            7, 15, 23, 31, 39, 47, 55, 63, 6, 14, 22, 30, 38, 46, 54, 62, 5, 13,
                            21, 29, 37, 45, 53, 61, 4, 12, 20, 28, 36, 44, 52, 60, 3, 11, 19, 27,
                            35, 43, 51, 59, 2, 10, 18, 26, 34, 42, 50, 58, 1, 9, 17, 25, 33, 41,
                            49, 57, 0, 8, 16, 24, 32, 40, 48, 56,
                        ),
                        attacks,
                    ),
                    0,
                );
                _mm512_test_epi8_mask(attacks, attacks)
            }

            #[cfg(not(all(
                target_feature = "avx512bw",
                target_feature = "avx512vbmi",
                target_feature = "gfni"
            )))]
            {
                let lo = _mm512_castsi512_si256(attacks);
                let hi = _mm512_extracti64x4_epi64::<1>(attacks);
                let combined = _mm256_or_si256(lo, hi);
                let lo128 = _mm256_castsi256_si128(combined);
                let hi128 = _mm256_extracti128_si256::<1>(combined);
                let combined128 = _mm_or_si128(lo128, hi128);
                (_mm_extract_epi64::<0>(combined128) | _mm_extract_epi64::<1>(combined128)) as u64
            }
        }
    }
}

// ============================================================
// Public API — dispatches to SIMD or falls through to bitboard.rs
// ============================================================

/// Bishop attacks via AVX2 BLSMSK + byte-swap trick.
#[cfg(target_feature = "avx2")]
#[inline(always)]
pub fn bishop_attacks(sq: Square, occupied: u64) -> u64 {
    avx2::bishop_attacks(sq, occupied)
}

/// Rook attacks via AVX2 byte-swap (N/S) + rank lookup (E/W).
#[cfg(target_feature = "avx2")]
#[inline(always)]
pub fn rook_attacks(sq: Square, occupied: u64) -> u64 {
    avx2::rook_attacks(sq, occupied)
}

/// Queen attacks via AVX2 BLSMSK + PP-fill (or AVX-512 lzcnt).
#[cfg(target_feature = "avx2")]
#[inline(always)]
pub fn queen_attacks(sq: Square, occupied: u64) -> u64 {
    avx2::queen_attacks(sq, occupied)
}

/// Whether the AVX-512 Kogge-Stone setwise fill runs — decided once at startup
/// by `cpu::init` (AVX-512F detected), read as a plain load. Defaults to false
/// so binaries that skipped init (unit tests) take the loop, which returns
/// identical bitboards. Distribution builds only: a build made for one machine
/// pins the answer at compile time instead.
#[cfg(all(target_arch = "x86_64", gaia_dist))]
static USE_SETWISE512: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[cfg(all(target_arch = "x86_64", gaia_dist))]
pub fn set_setwise512(enabled: bool) {
    USE_SETWISE512.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

/// Aggregated attacks of **all** sliders of one color simultaneously:
/// AVX-512 Kogge-Stone (8 ray directions in parallel in a single `zmm`
/// register) when available, otherwise a loop over each slider individually.
/// Both forms return the same bitboard. Distribution builds elect the fill at
/// runtime; others pin it at compile time.
#[inline(always)]
pub fn slider_attacks_setwise(bishops: u64, rooks: u64, queens: u64, occupied: u64) -> u64 {
    #[cfg(all(target_arch = "x86_64", gaia_dist))]
    if USE_SETWISE512.load(std::sync::atomic::Ordering::Relaxed) {
        return unsafe { avx512::slider_attacks_setwise(bishops, rooks, queens, occupied) };
    }
    #[cfg(all(target_feature = "avx512f", not(gaia_dist)))]
    return unsafe { avx512::slider_attacks_setwise(bishops, rooks, queens, occupied) };

    #[cfg(not(all(target_feature = "avx512f", not(gaia_dist))))]
    {
        let mut attacks = 0u64;
        let bq = bishops | queens;
        let rq = rooks | queens;
        let mut bb = bq;
        while bb != 0 {
            let sq = pop_lsb(&mut bb);
            attacks |= crate::bitboard::bishop_attacks(sq, occupied);
        }
        bb = rq;
        while bb != 0 {
            let sq = pop_lsb(&mut bb);
            attacks |= crate::bitboard::rook_attacks(sq, occupied);
        }
        attacks
    }
}

// ============================================================
// Tests — cross-validate SIMD results against magic bitboards
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn magic_bishop(sq: Square, occupied: u64) -> u64 {
        // Use the public dispatch function (routes to SIMD or magic internally)
        crate::bitboard::bishop_attacks(sq, occupied)
    }

    fn magic_rook(sq: Square, occupied: u64) -> u64 {
        crate::bitboard::rook_attacks(sq, occupied)
    }

    #[test]
    fn test_init() {
        init();
        let t = tables();
        // Queen mask for E4: east ray should include F4..H4
        let sq = Square::E4;
        let east_mask = t.queen_mask_left[sq.index()][0] as u64;
        assert!(east_mask & Square::F4.bb() != 0);
        assert!(east_mask & Square::H4.bb() != 0);
        assert!(east_mask & Square::E4.bb() == 0); // excludes the square itself
    }

    #[test]
    fn test_rook_ew_table() {
        init();
        let t = tables();
        // E-file rook on empty rank: should attack all files except E
        let att = t.rook_attacks_ew[4]; // occ=0, file=4 (E)
        assert_eq!(att.count_ones(), 7); // A through H minus E
    }

    #[cfg(target_feature = "avx2")]
    #[test]
    fn test_bishop_avx2_vs_magic() {
        init();
        let test_occs: [u64; 5] = [
            0,
            0x00FF00FF00FF00FF,
            0x0000001818000000, // center
            0x8142241818244281, // diamond
            0xFFFFFFFFFFFFFFFF,
        ];
        for sq in 0..64u8 {
            let sq = Square(sq);
            for &occ in &test_occs {
                let simd = bishop_attacks(sq, occ);
                let magic = magic_bishop(sq, occ);
                assert_eq!(
                    simd, magic,
                    "bishop_attacks mismatch at {}: simd={:#018x} magic={:#018x} occ={:#018x}",
                    sq, simd, magic, occ
                );
            }
        }
    }

    #[cfg(target_feature = "avx2")]
    #[test]
    fn test_rook_avx2_vs_magic() {
        init();
        let test_occs: [u64; 5] = [
            0,
            0x00FF00FF00FF00FF,
            0x0000001818000000,
            0x8142241818244281,
            0xFFFFFFFFFFFFFFFF,
        ];
        for sq in 0..64u8 {
            let sq = Square(sq);
            for &occ in &test_occs {
                let simd = rook_attacks(sq, occ);
                let magic = magic_rook(sq, occ);
                assert_eq!(
                    simd, magic,
                    "rook_attacks mismatch at {}: simd={:#018x} magic={:#018x} occ={:#018x}",
                    sq, simd, magic, occ
                );
            }
        }
    }

    #[cfg(target_feature = "avx2")]
    #[test]
    fn test_queen_avx2_vs_magic() {
        init();
        let test_occs: [u64; 5] = [
            0,
            0x00FF00FF00FF00FF,
            0x0000001818000000,
            0x8142241818244281,
            0xFFFFFFFFFFFFFFFF,
        ];
        for sq in 0..64u8 {
            let sq = Square(sq);
            for &occ in &test_occs {
                let simd = queen_attacks(sq, occ);
                let magic = magic_bishop(sq, occ) | magic_rook(sq, occ);
                assert_eq!(
                    simd, magic,
                    "queen_attacks mismatch at {}: simd={:#018x} magic={:#018x} occ={:#018x}",
                    sq, simd, magic, occ
                );
            }
        }
    }

    #[test]
    fn test_setwise_vs_individual() {
        init();
        // Startpos-like setup: bishops on C1,F1, rooks on A1,H1, queen on D1
        let bishops = Square::C1.bb() | Square::F1.bb();
        let rooks = Square::A1.bb() | Square::H1.bb();
        let queens = Square::D1.bb();
        let occupied = 0xFFFF00000000FFFF; // ranks 1,2,7,8 occupied

        let setwise = slider_attacks_setwise(bishops, rooks, queens, occupied);

        // Compute individually
        let mut individual = 0u64;
        let bq = bishops | queens;
        let rq = rooks | queens;
        let mut bb = bq;
        while bb != 0 {
            let sq = pop_lsb(&mut bb);
            individual |= crate::bitboard::bishop_attacks(sq, occupied);
        }
        bb = rq;
        while bb != 0 {
            let sq = pop_lsb(&mut bb);
            individual |= crate::bitboard::rook_attacks(sq, occupied);
        }

        assert_eq!(
            setwise, individual,
            "setwise={:#018x} individual={:#018x}",
            setwise, individual
        );
    }
}
