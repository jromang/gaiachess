//! [Cuckoo hashing](https://www.chessprogramming.org/Cuckoo_Hashing) for upcoming
//! repetition detection (M. N. J. van Kervinck's algorithm).
//!
//! Pre-computes a cuckoo hash table mapping Zobrist key XOR differences to
//! reversible non-pawn piece moves. Used by `Position::upcoming_repetition()`
//! to detect draws one ply before they appear in the search tree.

use crate::bitboard::{
    between_bb, init_king_attacks, init_knight_attacks, sliding_attack_otf, BISHOP_DELTAS,
    ROOK_DELTAS,
};
use crate::types::Square;
use crate::zobrist::ZOBRIST;

const TABLE_SIZE: usize = 0x2000; // 8192 entries
const TABLE_MASK: u64 = (TABLE_SIZE - 1) as u64;

/// Cuckoo hash function 1: bits [32..44] of the key.
#[inline(always)]
pub const fn h1(key: u64) -> usize {
    ((key >> 32) & TABLE_MASK) as usize
}

/// Cuckoo hash function 2: bits [48..60] of the key.
#[inline(always)]
pub const fn h2(key: u64) -> usize {
    ((key >> 48) & TABLE_MASK) as usize
}

/// Pre-computed cuckoo tables: keys and square pairs for reversible moves.
struct CuckooTables {
    keys: [u64; TABLE_SIZE],
    sq_a: [u8; TABLE_SIZE], // Square index (64 = empty sentinel)
    sq_b: [u8; TABLE_SIZE],
}

/// Compile-time cuckoo table initialization.
///
/// Iterates over all non-pawn piece-color combinations (indices 2..12) and
/// all square pairs where the piece can move on an empty board. Stores the
/// Zobrist XOR difference as the key, indexed by cuckoo hashing.
const fn init_cuckoo() -> CuckooTables {
    let knight_atk = init_knight_attacks();
    let king_atk = init_king_attacks();

    let mut tables = CuckooTables {
        keys: [0u64; TABLE_SIZE],
        sq_a: [64u8; TABLE_SIZE], // 64 = sentinel (Square::NONE.0 = 64)
        sq_b: [64u8; TABLE_SIZE],
    };

    // Piece indices 2..12 = all non-pawn pieces (both colors)
    // GaiaChess: Piece = type*2 + color
    // 0=WP, 1=BP, 2=WN, 3=BN, 4=WB, 5=BB, 6=WR, 7=BR, 8=WQ, 9=BQ, 10=WK, 11=BK
    let mut pc = 2u8;
    while pc < 12 {
        let pt = pc >> 1; // 1=Knight, 2=Bishop, 3=Rook, 4=Queen, 5=King

        let mut a = 0u8;
        while a < 64 {
            let mut b = a + 1;
            while b < 64 {
                // Check if this piece can move from a to b on empty board
                let attacks = match pt {
                    1 => knight_atk[a as usize],
                    2 => sliding_attack_otf(a, 0, &BISHOP_DELTAS),
                    3 => sliding_attack_otf(a, 0, &ROOK_DELTAS),
                    4 => {
                        sliding_attack_otf(a, 0, &BISHOP_DELTAS)
                            | sliding_attack_otf(a, 0, &ROOK_DELTAS)
                    }
                    5 => king_atk[a as usize],
                    _ => 0,
                };

                if attacks & (1u64 << b) != 0 {
                    // Zobrist difference for this reversible move
                    let mut key = ZOBRIST.pieces[pc as usize][a as usize]
                        ^ ZOBRIST.pieces[pc as usize][b as usize]
                        ^ ZOBRIST.side;

                    let mut sq_a = a;
                    let mut sq_b = b;
                    let mut slot = h1(key);

                    // Cuckoo insertion: swap with existing entry, rehash displaced entry
                    loop {
                        let tmp_key = tables.keys[slot];
                        let tmp_a = tables.sq_a[slot];
                        let tmp_b = tables.sq_b[slot];

                        tables.keys[slot] = key;
                        tables.sq_a[slot] = sq_a;
                        tables.sq_b[slot] = sq_b;

                        key = tmp_key;
                        sq_a = tmp_a;
                        sq_b = tmp_b;

                        // Empty slot sentinel (64 = no entry)
                        if sq_a == 64 {
                            break;
                        }

                        // Alternate between h1 and h2
                        slot = if slot == h1(key) { h2(key) } else { h1(key) };
                    }
                }

                b += 1;
            }
            a += 1;
        }
        pc += 1;
    }

    tables
}

/// Static compile-time cuckoo tables.
static CUCKOO: CuckooTables = init_cuckoo();

/// Check if a legal reversible move exists that would create a repeated position.
///
/// Uses the van Kervinck cuckoo algorithm to detect repetitions one ply before
/// they appear. Called from search when `alpha < 0` to raise alpha to draw score.
///
/// `history_keys` = slice of historical Zobrist keys, most recent last.
/// `current_key` = key of current position.
/// `occupied` = current occupancy bitboard.
/// `hmc` = halfmove clock, `pfn` = plies_from_null, `repetition` = incremental rep field.
/// `search_ply` = distance from root.
pub fn upcoming_repetition(
    history_keys: &[u64],
    current_key: u64,
    occupied: u64,
    hmc: usize,
    pfn: usize,
    repetition: i32,
    search_ply: usize,
) -> bool {
    let end = hmc.min(pfn);
    if end < 3 {
        return false;
    }

    let len = history_keys.len();
    debug_assert!(
        len >= end,
        "upcoming_repetition: history too short len={} end={}",
        len,
        end
    );

    // s(v) = key of position v half-moves ago
    // history_keys[len-1] = key saved 1 ply ago (before last move)
    // history_keys[len-v] = key saved v plies ago
    let s = |v: usize| -> u64 {
        if v <= len {
            history_keys[len - v]
        } else {
            0
        }
    };

    let s0 = current_key;
    let mut other = s0 ^ s(1) ^ ZOBRIST.side;

    let mut d = 3usize;
    while d <= end {
        other ^= s(d - 1) ^ s(d) ^ ZOBRIST.side;

        if other != 0 {
            d += 2;
            continue;
        }

        let diff = s0 ^ s(d);
        let mut i = h1(diff);

        if CUCKOO.keys[i] != diff {
            i = h2(diff);
            if CUCKOO.keys[i] != diff {
                d += 2;
                continue;
            }
        }

        let sa = Square(CUCKOO.sq_a[i]);
        let sb = Square(CUCKOO.sq_b[i]);

        // Check that the path between the two squares is unobstructed
        // The path between the two squares must be clear. `between_bb` includes its
        // second square, and that is where the piece stands whenever it sits on the
        // higher of the two — it is no obstacle to its own move.
        if (between_bb(sa, sb) & !sb.bb()) & occupied != 0 {
            d += 2;
            continue;
        }

        // Within search tree: always a draw
        if search_ply > d {
            return true;
        }

        // Before root: need prior repetition to confirm
        if repetition != 0 {
            return true;
        }

        d += 2;
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuckoo_table_populated() {
        // The table should have a significant number of non-zero entries.
        // Standard chess produces ~3668 entries.
        let count = CUCKOO
            .keys
            .iter()
            .filter(|&&k| k != 0)
            .count();
        assert!(
            count > 3000 && count < 4000,
            "cuckoo table has {count} entries, expected ~3668"
        );
    }

    #[test]
    fn test_cuckoo_no_collisions() {
        // Every non-zero key should be findable via h1 or h2
        for i in 0..TABLE_SIZE {
            let key = CUCKOO.keys[i];
            if key == 0 {
                continue;
            }
            assert!(
                h1(key) == i || h2(key) == i,
                "cuckoo entry {i} with key {key:#x} not at h1={} or h2={}",
                h1(key),
                h2(key)
            );
        }
    }
}
