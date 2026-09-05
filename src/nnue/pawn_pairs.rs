//! 3-file pawn-pawn features for the NNUE aux transformer.
//!
//! 96 pawn identities (48 squares × 2 colours, ranks 2–7) yield
//! `C(96, 2) = 4_560` unordered pairs. A pair is active only when the two
//! pawns sit on the same file or an adjacent file (3-file window).
//!
//! Identities are perspective-relative: the side-to-move's pawns occupy
//! slots 0..48, the opponent's 48..96, after horizontal mirroring and
//! rank-flip so the evaluated king is always on files a–d, rank 1 at the
//! bottom.
//!
//! Feature indices occupy `0 .. PAWN_PAIR_SIZE` of the i8 weight array
//! (threats follow at `THREAT_OFFSET`).

use crate::bitboard::pop_lsb;
use crate::position::Position;
use crate::types::{Piece, Square};

use super::PAWN_PAIR_SIZE;

/// Maximum simultaneously active pawn-pair features per perspective.
pub const MAX_ACTIVE_PAIRS: usize = 64;

const FILE_A: u64 = 0x0101_0101_0101_0101;

/// `PP_MASK[sq]`: files of `sq` and its neighbours, empty on ranks 1 and 8.
#[rustfmt::skip]
pub const PP_MASK: [u64; 64] = {
    let mut table = [0u64; 64];
    let mut sq = 8;
    while sq < 56 {
        let file = sq & 7;
        let mut m = FILE_A << file;
        if file > 0 {
            m |= FILE_A << (file - 1);
        }
        if file < 7 {
            m |= FILE_A << (file + 1);
        }
        table[sq] = m;
        sq += 1;
    }
    table
};

/// Unordered pair index `C(hi, 2) + lo` in `0 .. PAWN_PAIR_SIZE`.
#[inline]
pub fn pair_index(id_a: usize, id_b: usize) -> usize {
    debug_assert!(id_a < 96, "pawn id {id_a} out of 0..96");
    debug_assert!(id_b < 96, "pawn id {id_b} out of 0..96");
    debug_assert_ne!(id_a, id_b);
    let lo = id_a.min(id_b);
    let hi = id_a.max(id_b);
    let idx = hi * (hi - 1) / 2 + lo;
    debug_assert!(idx < PAWN_PAIR_SIZE, "pair index {idx} >= {PAWN_PAIR_SIZE}");
    idx
}

/// Pawn identity in `0..48` (friendly) or `48..96` (enemy) after orientation.
#[inline]
fn pawn_id(sq: Square, enemy_offset: usize, square_flip: usize) -> usize {
    let oriented = (sq.0 as usize) ^ square_flip;
    debug_assert!(
        oriented >= 8 && oriented < 56,
        "oriented pawn square {oriented} not on ranks 2–7"
    );
    enemy_offset + (oriented - 8)
}

/// Collect active pawn-pair feature indices for one perspective.
pub fn collect_for_pov(
    white_pawns: u64,
    black_pawns: u64,
    pov: usize,
    mirrored: bool,
    out: &mut [u32; MAX_ACTIVE_PAIRS],
) -> usize {
    debug_assert!(pov < 2);
    let square_flip = (7 * mirrored as usize) ^ (56 * pov);
    let (friendly, enemy) = if pov == 0 {
        (white_pawns, black_pawns)
    } else {
        (black_pawns, white_pawns)
    };

    let mut n = 0usize;
    emit_pairs(friendly, friendly, 0, 0, square_flip, true, out, &mut n);
    emit_pairs(friendly, enemy, 0, 48, square_flip, false, out, &mut n);
    emit_pairs(enemy, enemy, 48, 48, square_flip, true, out, &mut n);
    n
}

/// Collect pairs from `outer` × (`inner` ∩ window). When `same` is true the
/// inner loop walks the remaining bits of `outer` so each unordered pair is
/// emitted once.
#[allow(clippy::too_many_arguments)]
fn emit_pairs(
    mut outer: u64,
    inner_bb: u64,
    outer_off: usize,
    inner_off: usize,
    square_flip: usize,
    same: bool,
    out: &mut [u32; MAX_ACTIVE_PAIRS],
    n: &mut usize,
) {
    while outer != 0 {
        let sq = pop_lsb(&mut outer);
        let id = pawn_id(sq, outer_off, square_flip);
        let mut inner = if same {
            outer & inner_bb & PP_MASK[sq.index()]
        } else {
            inner_bb & PP_MASK[sq.index()]
        };
        while inner != 0 {
            let t = pop_lsb(&mut inner);
            let other = pawn_id(t, inner_off, square_flip);
            debug_assert!(*n < MAX_ACTIVE_PAIRS, "too many pawn-pair features");
            out[*n] = pair_index(id, other) as u32;
            *n += 1;
        }
    }
}

/// Pairs lost and gained between two pawn configurations, for one perspective.
///
/// A pawn move changes a handful of pairs — those of the pawn that left, of the pawn
/// that arrived, of a pawn captured — while every pair among the untouched pawns stays
/// exactly as it was. Re-enumerating both whole sets would subtract and re-add all of
/// those (some forty 1 KB weight rows per perspective, for a net change of nothing), so
/// only the pairs touching a changed pawn are emitted: `subs` are the pairs of the old
/// set that involve a removed pawn, `adds` the pairs of the new set that involve an
/// added pawn. Each pair is emitted once, even when both of its pawns changed.
///
/// Returns `(n_subs, n_adds)`.
#[allow(clippy::too_many_arguments)]
pub fn collect_delta_for_pov(
    old_white: u64,
    old_black: u64,
    new_white: u64,
    new_black: u64,
    pov: usize,
    mirrored: bool,
    subs: &mut [u32; MAX_ACTIVE_PAIRS],
    adds: &mut [u32; MAX_ACTIVE_PAIRS],
) -> (usize, usize) {
    debug_assert!(pov < 2);
    let square_flip = (7 * mirrored as usize) ^ (56 * pov);
    let (old_f, old_e, new_f, new_e) = if pov == 0 {
        (old_white, old_black, new_white, new_black)
    } else {
        (old_black, old_white, new_black, new_white)
    };
    let n_subs = emit_touching(old_f & !new_f, old_e & !new_e, old_f, old_e, square_flip, subs);
    let n_adds = emit_touching(new_f & !old_f, new_e & !old_e, new_f, new_e, square_flip, adds);
    (n_subs, n_adds)
}

/// Pairs of the set (`set_f`, `set_e`) that involve at least one pawn of
/// (`changed_f`, `changed_e`). A changed pawn leaves the partner sets once handled, so
/// a pair of two changed pawns is produced exactly once and no pawn pairs with itself.
fn emit_touching(
    mut changed_f: u64,
    mut changed_e: u64,
    mut set_f: u64,
    mut set_e: u64,
    square_flip: usize,
    out: &mut [u32; MAX_ACTIVE_PAIRS],
) -> usize {
    debug_assert!(
        changed_f & !set_f == 0 && changed_e & !set_e == 0,
        "a changed pawn must belong to the set its pairs are taken from"
    );
    let mut n = 0usize;
    while changed_f != 0 {
        let sq = pop_lsb(&mut changed_f);
        set_f &= !(1u64 << sq.0);
        let id = pawn_id(sq, 0, square_flip);
        let window = PP_MASK[sq.index()];
        emit_partners(id, set_f & window, 0, square_flip, out, &mut n);
        emit_partners(id, set_e & window, 48, square_flip, out, &mut n);
    }
    while changed_e != 0 {
        let sq = pop_lsb(&mut changed_e);
        set_e &= !(1u64 << sq.0);
        let id = pawn_id(sq, 48, square_flip);
        let window = PP_MASK[sq.index()];
        emit_partners(id, set_f & window, 0, square_flip, out, &mut n);
        emit_partners(id, set_e & window, 48, square_flip, out, &mut n);
    }
    n
}

/// Pairs between pawn `id` and every pawn of `partners`.
#[inline]
fn emit_partners(
    id: usize,
    mut partners: u64,
    partner_off: usize,
    square_flip: usize,
    out: &mut [u32; MAX_ACTIVE_PAIRS],
    n: &mut usize,
) {
    while partners != 0 {
        let t = pop_lsb(&mut partners);
        let other = pawn_id(t, partner_off, square_flip);
        debug_assert!(*n < MAX_ACTIVE_PAIRS, "too many pawn-pair features");
        out[*n] = pair_index(id, other) as u32;
        *n += 1;
    }
}

/// Dump sorted pawn-pair indices for both perspectives (STM first).
pub fn dump_features(pos: &Position) {
    let stm = pos.side_to_move;
    let wp = pos.pieces[Piece::WHITE_PAWN.index()];
    let bp = pos.pieces[Piece::BLACK_PAWN.index()];
    for (label, color) in [("stm", stm), ("ntm", !stm)] {
        let pov = color.index();
        let mirrored = pos.king_sq(color).file() >= 4;
        let mut feats = [0u32; MAX_ACTIVE_PAIRS];
        let n = collect_for_pov(wp, bp, pov, mirrored, &mut feats);
        let mut v: Vec<u32> = feats[..n].to_vec();
        v.sort_unstable();
        let strs: Vec<String> = v.iter().map(|f| f.to_string()).collect();
        println!("pp-{label}:{}", strs.join(","));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Position;

    const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    #[test]
    fn pair_index_covers_the_triangle() {
        assert_eq!(pair_index(0, 1), 0);
        assert_eq!(pair_index(1, 0), 0);
        assert_eq!(pair_index(0, 2), 1);
        assert_eq!(pair_index(1, 2), 2);
        assert_eq!(pair_index(94, 95), 95 * 94 / 2 + 94);
        assert_eq!(pair_index(94, 95), PAWN_PAIR_SIZE - 1);
    }

    #[test]
    fn startpos_pairs_are_in_range() {
        let pos = Position::from_fen(START).unwrap();
        let wp = pos.pieces[Piece::WHITE_PAWN.index()];
        let bp = pos.pieces[Piece::BLACK_PAWN.index()];
        let mut feats = [0u32; MAX_ACTIVE_PAIRS];
        let n = collect_for_pov(wp, bp, 0, false, &mut feats);
        assert!(n > 0, "startpos must have pawn pairs");
        assert!(n <= MAX_ACTIVE_PAIRS);
        for i in 0..n {
            assert!((feats[i] as usize) < PAWN_PAIR_SIZE);
        }
    }

    #[test]
    fn empty_board_has_no_pairs() {
        let mut feats = [0u32; MAX_ACTIVE_PAIRS];
        let n = collect_for_pov(0, 0, 0, false, &mut feats);
        assert_eq!(n, 0);
    }

    /// The delta of a move must be exactly what the full enumerations lose and gain:
    /// every legal move of every bench position, both perspectives, both mirrorings.
    #[test]
    fn the_delta_is_the_symmetric_difference_of_the_full_enumerations() {
        use crate::bench::POSITIONS;
        use crate::movegen;
        use crate::types::{ArrayBuf, Move};
        use std::collections::BTreeSet;

        let full = |wp: u64, bp: u64, pov: usize, mirrored: bool| -> BTreeSet<u32> {
            let mut feats = [0u32; MAX_ACTIVE_PAIRS];
            let n = collect_for_pov(wp, bp, pov, mirrored, &mut feats);
            let set: BTreeSet<u32> = feats[..n].iter().copied().collect();
            assert_eq!(set.len(), n, "full enumeration emitted a pair twice");
            set
        };

        let mut checked = 0usize;
        for fen in POSITIONS.iter() {
            let mut pos = Position::from_fen(fen).unwrap();
            let old_wp = pos.pieces[Piece::WHITE_PAWN.index()];
            let old_bp = pos.pieces[Piece::BLACK_PAWN.index()];
            let mut buf = ArrayBuf::<Move, 256>::new();
            let n_moves = movegen::generate_legal_moves(&pos, &mut buf);
            for i in 0..n_moves {
                let mv = buf[i];
                pos.make_move(mv);
                let new_wp = pos.pieces[Piece::WHITE_PAWN.index()];
                let new_bp = pos.pieces[Piece::BLACK_PAWN.index()];
                for pov in 0..2 {
                    for mirrored in [false, true] {
                        let before = full(old_wp, old_bp, pov, mirrored);
                        let after = full(new_wp, new_bp, pov, mirrored);
                        let mut subs = [0u32; MAX_ACTIVE_PAIRS];
                        let mut adds = [0u32; MAX_ACTIVE_PAIRS];
                        let (ns, na) = collect_delta_for_pov(
                            old_wp, old_bp, new_wp, new_bp, pov, mirrored, &mut subs, &mut adds,
                        );
                        let subs: Vec<u32> = subs[..ns].to_vec();
                        let adds: Vec<u32> = adds[..na].to_vec();
                        let sub_set: BTreeSet<u32> = subs.iter().copied().collect();
                        let add_set: BTreeSet<u32> = adds.iter().copied().collect();
                        assert_eq!(sub_set.len(), ns, "{fen} {mv:?}: a sub emitted twice");
                        assert_eq!(add_set.len(), na, "{fen} {mv:?}: an add emitted twice");
                        let lost: BTreeSet<u32> = before.difference(&after).copied().collect();
                        let gained: BTreeSet<u32> = after.difference(&before).copied().collect();
                        assert_eq!(sub_set, lost, "{fen} {mv:?} pov {pov} mirrored {mirrored}");
                        assert_eq!(add_set, gained, "{fen} {mv:?} pov {pov} mirrored {mirrored}");
                        checked += 1;
                    }
                }
                pos.unmake_move(mv);
            }
        }
        assert!(checked > 1000, "only {checked} cases checked");
    }

    #[test]
    fn adjacent_pawns_on_rank_2_form_a_pair() {
        let a2 = 1u64 << Square::A2.0;
        let b2 = 1u64 << Square::B2.0;
        let mut feats = [0u32; MAX_ACTIVE_PAIRS];
        let n = collect_for_pov(a2 | b2, 0, 0, false, &mut feats);
        assert_eq!(n, 1);
        // a2 → id 0, b2 → id 1, pair 0.
        assert_eq!(feats[0], 0);
    }
}
