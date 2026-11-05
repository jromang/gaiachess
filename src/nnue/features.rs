//! K32Q2 piece-square feature indexing: own king square (32 buckets after mirroring)
//! times the presence of the opponent's queen (2 states).
//!
//! Each feature is uniquely identified by:
//! `block(king, opponent queen) × 704 + plane × 64 + relative_sq`
//!
//! When the perspective king is on files e-h, all squares are horizontally
//! mirrored (XOR file with 7) so the network only learns one half: the 32
//! squares of files a-d are the 32 king buckets. A block is a king bucket and
//! whether the opponent of the perspective has a queen on the board.
//!
//! Planes: the perspective's pawn to queen (0..5), the opponent's pawn to queen
//! (5..10), and one plane shared by both kings (10). The perspective's king can
//! only stand on the bucket's own square, where the opponent's king never stands:
//! one plane holds both without a collision.

use crate::types::{Color, PieceType, Square};

use super::{PSQ_PER_BLOCK, QUEEN_STATES};

/// Plane shared by both kings.
pub const KING_PLANE: usize = 10;

/// Maps king square (after perspective flip) to its king bucket (0..32).
///
/// Every square of files a-d is its own bucket; files e-h repeat them mirrored,
/// which a mirrored perspective never reads (its king is brought to a-d first).
#[rustfmt::skip]
pub const KING_BUCKETS: [usize; 64] = [
     0,  1,  2,  3,  3,  2,  1,  0,  // rank 1
     4,  5,  6,  7,  7,  6,  5,  4,  // rank 2
     8,  9, 10, 11, 11, 10,  9,  8,  // rank 3
    12, 13, 14, 15, 15, 14, 13, 12,  // rank 4
    16, 17, 18, 19, 19, 18, 17, 16,  // rank 5
    20, 21, 22, 23, 23, 22, 21, 20,  // rank 6
    24, 25, 26, 27, 27, 26, 25, 24,  // rank 7
    28, 29, 30, 31, 31, 30, 29, 28,  // rank 8
];

/// Compute the flip mask for a given perspective king square and perspective color.
///
/// - File mirror: XOR with 7 when king is on files e-h (file >= 4).
/// - Rank flip: XOR with 56 for Black perspective (so both see from their own side).
#[inline(always)]
pub fn flip_mask(king_sq: Square, perspective: Color) -> u8 {
    (7 * (king_sq.file() >= 4) as u8) ^ (56 * perspective as u8)
}

/// The block of `perspective` for its king on `king_sq`, with or without an opponent
/// queen on the board: `bucket × 2 + opponent_queen`.
#[inline(always)]
pub fn block(king_sq: Square, perspective: Color, opponent_queen: bool) -> usize {
    let flip = flip_mask(king_sq, perspective);
    let bucket = KING_BUCKETS[(king_sq.0 ^ flip) as usize];
    debug_assert!(bucket < 32);
    bucket * QUEEN_STATES + opponent_queen as usize
}

/// The plane of a piece of `piece_type`, from the side of `is_enemy`.
#[inline(always)]
pub fn plane(piece_type: PieceType, is_enemy: bool) -> usize {
    if piece_type == PieceType::King {
        KING_PLANE
    } else {
        debug_assert!(piece_type.index() < 5);
        is_enemy as usize * 5 + piece_type.index()
    }
}

/// Compute the feature index for a piece on a square, from a given perspective.
///
/// Returns an index in `0..PSQ_SIZE`.
#[inline(always)]
pub fn feature_index(
    piece_color: Color,
    piece_type: PieceType,
    square: Square,
    king_sq: Square,
    perspective: Color,
    opponent_queen: bool,
) -> usize {
    let flip = flip_mask(king_sq, perspective);
    let blk = block(king_sq, perspective, opponent_queen);
    let rel_sq = (square.0 ^ flip) as usize;
    let idx = blk * PSQ_PER_BLOCK + plane(piece_type, piece_color != perspective) * 64 + rel_sq;
    debug_assert!(idx < super::PSQ_SIZE, "K32Q2 index {idx}");
    idx
}

/// Check whether an accumulator refresh is needed when the king moves.
///
/// Refresh is needed when the king changes bucket or crosses the vertical midline
/// (which changes the file-mirroring, affecting ALL feature indices). With one
/// bucket per square of a half, that is any king move that changes its square.
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
        let idx1 = feature_index(Color::White, PieceType::Pawn, Square::E2, Square::E1, Color::White, true);
        // Same pawn, king on d1 (no mirror) — should give different bucket
        let idx2 = feature_index(Color::White, PieceType::Pawn, Square::E2, Square::D1, Color::White, true);
        assert_ne!(idx1, idx2);
    }

    #[test]
    fn test_feature_index_mirror() {
        // King on e1 (file 4 >= 4, so mirror): pawn on e2 should map same as pawn on d2 with king on d1
        let idx_mirrored = feature_index(Color::White, PieceType::Pawn, Square::E2, Square::E1, Color::White, false);
        let idx_natural = feature_index(Color::White, PieceType::Pawn, Square::D2, Square::D1, Color::White, false);
        assert_eq!(idx_mirrored, idx_natural);
    }

    #[test]
    fn the_opponent_queen_selects_the_other_block() {
        let without = feature_index(Color::Black, PieceType::Rook, Square::A8, Square::G1, Color::White, false);
        let with = feature_index(Color::Black, PieceType::Rook, Square::A8, Square::G1, Color::White, true);
        assert_eq!(with, without + PSQ_PER_BLOCK);
    }

    #[test]
    fn both_kings_share_a_plane_without_colliding() {
        for king in 0..64u8 {
            for &persp in &[Color::White, Color::Black] {
                let own = feature_index(persp, PieceType::King, Square(king), Square(king), persp, false);
                for other in (0..64u8).filter(|&s| s != king) {
                    let theirs =
                        feature_index(!persp, PieceType::King, Square(other), Square(king), persp, false);
                    assert_ne!(own, theirs, "king {king}, other {other}");
                }
            }
        }
    }

    #[test]
    fn every_square_of_a_half_is_its_own_bucket() {
        assert!(needs_refresh(Square::A1, Square::B1, Color::White));
        assert!(needs_refresh(Square::A5, Square::B6, Color::White));
        assert!(!needs_refresh(Square::E1, Square::E1, Color::White));
        assert!(needs_refresh(Square::D1, Square::E1, Color::White));
    }

    #[test]
    fn test_feature_index_bounds() {
        use super::super::PSQ_SIZE;
        let types = [
            PieceType::Pawn,
            PieceType::Knight,
            PieceType::Bishop,
            PieceType::Rook,
            PieceType::Queen,
            PieceType::King,
        ];
        for king in 0..64u8 {
            for sq in 0..64u8 {
                for &pt in &types {
                    for &color in &[Color::White, Color::Black] {
                        for &persp in &[Color::White, Color::Black] {
                            for q in [false, true] {
                                let idx = feature_index(color, pt, Square(sq), Square(king), persp, q);
                                assert!(idx < PSQ_SIZE, "idx={idx} >= PSQ_SIZE={PSQ_SIZE}");
                            }
                        }
                    }
                }
            }
        }
    }
}
