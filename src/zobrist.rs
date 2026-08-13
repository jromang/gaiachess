//! [Zobrist hashing](https://www.chessprogramming.org/Zobrist_Hashing) for chess positions.
//!
//! Maps positions to near-unique `u64` hash keys by XOR-combining random numbers
//! for each piece placement, castling right, en passant file, and side to move.
//! XOR is its own inverse, enabling
//! [incremental updates](https://www.chessprogramming.org/Incremental_Updates)
//! during make/unmake.
//!
//! All 793 keys are computed at compile time via SplitMix64 PRNG.

/// Pre-computed random keys for all Zobrist components.
pub struct Zobrist {
    /// `pieces[pc][sq]` — key for piece `pc` on square `sq`.
    /// 12 piece-color combinations x 64 squares = 768 keys.
    pub pieces: [[u64; 64]; 12],
    /// `ep[file]` — key for en passant on the given file (0..7).
    pub ep: [u64; 8],
    /// `castling[rights]` — key for the 4-bit castling rights value (0..15).
    pub castling: [u64; 16],
    /// Key XOR'd when it is Black's turn to move.
    pub side: u64,
}

// 768 + 8 + 16 + 1 = 793 u64 values
const ZOBRIST_COUNT: usize = 793;

/// Global compile-time Zobrist keys (793 values, deterministic from a fixed seed).
pub const ZOBRIST: Zobrist = {
    const SEED: u64 = 0xFFAA_B58C_5833_FE89u64;
    const INCREMENT: u64 = 0x9E37_79B9_7F4A_7C15;

    let mut values = [0u64; ZOBRIST_COUNT];
    let mut state = SEED;

    let mut i = 0;
    while i < ZOBRIST_COUNT {
        state = state.wrapping_add(INCREMENT);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        values[i] = z ^ (z >> 31);
        i += 1;
    }

    // Copy flat array into struct fields (avoids transmute)
    let mut pieces = [[0u64; 64]; 12];
    let mut idx = 0;
    let mut pc = 0;
    while pc < 12 {
        let mut sq = 0;
        while sq < 64 {
            pieces[pc][sq] = values[idx];
            idx += 1;
            sq += 1;
        }
        pc += 1;
    }

    let mut ep = [0u64; 8];
    let mut f = 0;
    while f < 8 {
        ep[f] = values[idx];
        idx += 1;
        f += 1;
    }

    let mut castling = [0u64; 16];
    let mut c = 0;
    while c < 16 {
        castling[c] = values[idx];
        idx += 1;
        c += 1;
    }

    let side = values[idx];

    Zobrist { pieces, ep, castling, side }
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zobrist_nonzero() {
        // All piece keys should be non-zero
        for pc in 0..12 {
            for sq in 0..64 {
                assert_ne!(ZOBRIST.pieces[pc][sq], 0);
            }
        }
        assert_ne!(ZOBRIST.side, 0);
    }

    #[test]
    fn test_zobrist_unique() {
        // Spot-check: first few keys should all be different
        let a = ZOBRIST.pieces[0][0];
        let b = ZOBRIST.pieces[0][1];
        let c = ZOBRIST.pieces[1][0];
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_ne!(b, c);
    }

    #[test]
    fn test_zobrist_deterministic() {
        // Recompute a few values to ensure determinism
        let seed: u64 = 0xFFAA_B58C_5833_FE89;
        let inc: u64 = 0x9E37_79B9_7F4A_7C15;
        let state = seed.wrapping_add(inc);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z = z ^ (z >> 31);
        assert_eq!(z, ZOBRIST.pieces[0][0]);
    }
}
