//! HalfKA feature indexing with king buckets and horizontal mirroring.
//!
//! Each feature is uniquely identified by:
//! `bucket(king) × 768 + is_enemy × 384 + piece_type × 64 + relative_sq`
//!
//! When the perspective king is on files e-h, all squares are horizontally
//! mirrored (XOR file with 7) so the network only learns one half.
//!
//! 12 king buckets: ranks 1-4 get fine-grained buckets (0-10),
//! ranks 5-8 all share bucket 11.

use crate::types::{Color, PieceType, Square};

use super::INPUTS_PER_BUCKET;

// ============================================================
// King bucket layout
// ============================================================

/// Maps king square (after perspective flip) to input bucket index (0..INPUT_BUCKETS-1).
///
/// 12-bucket layout: ranks 1-2 get fine-grained buckets (files a-d mirrored
/// to e-h), ranks 3-4 get coarser buckets, ranks 5-8 all share bucket 11.
#[rustfmt::skip]
pub const KING_BUCKETS: [usize; 64] = [
     0,  1,  2,  3,  3,  2,  1,  0,  // rank 1
     4,  5,  6,  7,  7,  6,  5,  4,  // rank 2
     8,  8,  9,  9,  9,  9,  8,  8,  // rank 3
    10, 10, 10, 10, 10, 10, 10, 10,  // rank 4
    11, 11, 11, 11, 11, 11, 11, 11,  // rank 5
    11, 11, 11, 11, 11, 11, 11, 11,  // rank 6
    11, 11, 11, 11, 11, 11, 11, 11,  // rank 7
    11, 11, 11, 11, 11, 11, 11, 11,  // rank 8
];

/// Compute the flip mask for a given perspective king square and perspective color.
///
/// - File mirror: XOR with 7 when king is on files e-h (file >= 4).
/// - Rank flip: XOR with 56 for Black perspective (so both see from their own side).
#[inline(always)]
pub fn flip_mask(king_sq: Square, perspective: Color) -> u8 {
    (7 * (king_sq.file() >= 4) as u8) ^ (56 * perspective as u8)
}

/// Compute the feature index for a piece on a square, from a given perspective.
///
/// Returns an index in `0..INPUT_BUCKETS * INPUTS_PER_BUCKET`.
#[inline(always)]
pub fn feature_index(
    piece_color: Color,
    piece_type: PieceType,
    square: Square,
    king_sq: Square,
    perspective: Color,
) -> usize {
    let flip = flip_mask(king_sq, perspective);
    let king_idx = (king_sq.0 ^ flip) as usize;
    let bucket = KING_BUCKETS[king_idx];
    let is_enemy = (piece_color != perspective) as usize;
    let rel_sq = (square.0 ^ flip) as usize;

    bucket * INPUTS_PER_BUCKET + is_enemy * 384 + piece_type.index() * 64 + rel_sq
}

/// Check whether an accumulator refresh is needed when the king moves.
///
/// Refresh is needed when the king changes bucket or crosses the vertical midline
/// (which changes the file-mirroring, affecting ALL feature indices).
#[inline(always)]
pub fn needs_refresh(king_from: Square, king_to: Square, perspective: Color) -> bool {
    let flip_from = (king_from.0 ^ (56 * perspective as u8)) as usize;
    let flip_to = (king_to.0 ^ (56 * perspective as u8)) as usize;
    KING_BUCKETS[flip_from] != KING_BUCKETS[flip_to]
        || (king_from.file() >= 4) != (king_to.file() >= 4)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Color, PieceType, Square};

    #[test]
    fn test_feature_index_symmetric() {
        // A white pawn on e2, from White's perspective with king on e1
        let idx1 = feature_index(
            Color::White, PieceType::Pawn, Square::E2, Square::E1, Color::White,
        );
        // Same pawn, king on d1 (no mirror) — should give different bucket
        let idx2 = feature_index(
            Color::White, PieceType::Pawn, Square::E2, Square::D1, Color::White,
        );
        assert_ne!(idx1, idx2);
    }

    #[test]
    fn test_feature_index_mirror() {
        // King on e1 (file 4 >= 4, so mirror): pawn on e2 should map same as pawn on d2 with king on d1
        let idx_mirrored = feature_index(
            Color::White, PieceType::Pawn, Square::E2, Square::E1, Color::White,
        );
        let idx_natural = feature_index(
            Color::White, PieceType::Pawn, Square::D2, Square::D1, Color::White,
        );
        assert_eq!(idx_mirrored, idx_natural);
    }

    #[test]
    fn test_needs_refresh_same_bucket() {
        // King moves within same bucket and same side of board (both file < 4).
        // 12-bucket layout: rank 3 = bucket 8/9, rank 4 = bucket 10,
        // ranks 5-8 = all bucket 11.
        assert!(!needs_refresh(Square::A3, Square::B3, Color::White));
        assert!(!needs_refresh(Square::A5, Square::B6, Color::White)); // both bucket 11
        assert!(!needs_refresh(Square::A7, Square::B8, Color::White)); // both bucket 11
    }

    #[test]
    fn test_needs_refresh_crosses_midline() {
        // King crosses from d-file to e-file
        assert!(needs_refresh(Square::D1, Square::E1, Color::White));
    }

    #[test]
    fn test_feature_index_bounds() {
        // Verify all combinations produce valid indices
        use super::super::FT_SIZE;
        for king in 0..64u8 {
            for sq in 0..64u8 {
                for pt in 0..6u8 {
                    for &color in &[Color::White, Color::Black] {
                        for &persp in &[Color::White, Color::Black] {
                            let pt_enum = match pt {
                                0 => PieceType::Pawn,
                                1 => PieceType::Knight,
                                2 => PieceType::Bishop,
                                3 => PieceType::Rook,
                                4 => PieceType::Queen,
                                _ => PieceType::King,
                            };
                            let idx = feature_index(
                                color, pt_enum, Square(sq), Square(king), persp,
                            );
                            assert!(idx < FT_SIZE, "idx={idx} >= FT_SIZE={FT_SIZE}");
                        }
                    }
                }
            }
        }
    }
}
