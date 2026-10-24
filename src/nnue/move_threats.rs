#![allow(unsafe_op_in_unsafe_fn)]
// Hand-vectorised code: the loop index walks register halves and quarters in lockstep
// with the tables they are matched against.
#![allow(clippy::needless_range_loop)]
//! Threat changes recorded when a move is played.
//!
//! `Network::push` runs before `make_move`, with the board before the move at hand. It
//! records here the attacks the move creates and destroys, as entries that carry no
//! perspective, and the threat accumulator of the ply maps them to feature indices and
//! applies them when it is first needed (`threats.rs`).
//!
//! A move is played as a short sequence of elementary changes, each one read on the
//! board it is made on: a piece leaves a square, a piece arrives on one, or the piece on
//! a square turns into another (a capture, a promotion with capture). A departure or an
//! arrival records every attack between the piece and the pieces it sees or is seen by,
//! plus, with the opposite sign, the attacks of the sliders whose line it uncovers or
//! blocks. A turn records the attacks of the old piece and of the new one; occupancy
//! does not change, so no line opens or closes. Rays start one square from the focus,
//! so a departure reads the same board whether the focus is still occupied or not.
//!
//! Attacks by a king and attacks on a king are never recorded: no feature encodes
//! them. Every other pair is, even those the feature set leaves out (pawn → bishop,
//! the second direction of two like pieces...): which are encoded depends on the side
//! and its mirroring, and is decided when the entries are applied.
//!
//! Backends: bitboards (every target), and a ray view of the 64 squares that matter
//! around the focus — 8 directions, each with one knight square and the 7 squares of
//! the ray — gathered from the mailbox by one byte permutation, with AVX-512 VBMI2,
//! AVX2, NEON or wasm simd128. They record the same entries, in different orders.

use std::mem::MaybeUninit;

use crate::bitboard::{
    between_bb, bishop_attacks, knight_attacks, line_bb, pawn_attacks, pop_lsb, rook_attacks,
};
use crate::position::Position;
use crate::types::{
    Color, MT_CASTLING, MT_EN_PASSANT, MT_NORMAL, MT_PROMOTION, Piece, PieceType, Square,
    pawn_push,
};

use super::accumulator::AccDelta;

// ============================================================
// The record
// ============================================================

/// Entries each list of a record can hold.
///
/// The worst move is an en passant capture: three elementary changes, each departure or
/// arrival recording at most 24 attacks of its own sign (8 outgoing, 16 incoming) and
/// 8 line changes of the other. That is 56 entries in one list at most, and a vector
/// store may write `STORE_MARGIN` slots past the last entry it means.
pub(crate) const MOVE_THREATS_CAP: usize = 96;

/// Slots a backend may write past the end of a list in one vector store.
const STORE_MARGIN: usize = 16;

const _: () = assert!(56 + STORE_MARGIN <= MOVE_THREATS_CAP);

/// One attack: attacker, its square, target, its square, one byte each (pieces in the
/// `Piece` encoding, squares 0..63), little-endian.
#[inline(always)]
const fn entry(att: u8, att_sq: u8, def: u8, def_sq: u8) -> u32 {
    att as u32 | (att_sq as u32) << 8 | (def as u32) << 16 | (def_sq as u32) << 24
}

/// The four fields of an entry: (attacker, its square, target, its square).
#[inline(always)]
pub(super) fn decode(e: u32) -> (Piece, usize, Piece, usize) {
    (
        Piece(e as u8),
        (e >> 8) as u8 as usize,
        Piece((e >> 16) as u8),
        (e >> 24) as usize,
    )
}

/// What a move changed, as recorded when it was played.
#[derive(Clone, Copy)]
pub(crate) struct MoveThreats {
    adds: [MaybeUninit<u32>; MOVE_THREATS_CAP],
    subs: [MaybeUninit<u32>; MOVE_THREATS_CAP],
    n_adds: usize,
    n_subs: usize,
    /// White and black pawns before and after the move, for the pawn-pair delta.
    pub(super) pawns_before: [u64; 2],
    pub(super) pawns_after: [u64; 2],
    /// Both kings after the move: a side whose king changed half starts afresh.
    pub(super) king_sq: [Square; 2],
}

impl MoveThreats {
    pub(crate) const fn new() -> Self {
        MoveThreats {
            adds: [MaybeUninit::uninit(); MOVE_THREATS_CAP],
            subs: [MaybeUninit::uninit(); MOVE_THREATS_CAP],
            n_adds: 0,
            n_subs: 0,
            pawns_before: [0; 2],
            pawns_after: [0; 2],
            king_sq: [Square::NONE; 2],
        }
    }

    /// The attacks the move creates.
    #[inline(always)]
    pub(super) fn adds(&self) -> &[u32] {
        // SAFETY: the first `n_adds` slots were written by this record.
        unsafe { std::slice::from_raw_parts(self.adds.as_ptr().cast(), self.n_adds) }
    }

    /// The attacks the move destroys.
    #[inline(always)]
    pub(super) fn subs(&self) -> &[u32] {
        // SAFETY: the first `n_subs` slots were written by this record.
        unsafe { std::slice::from_raw_parts(self.subs.as_ptr().cast(), self.n_subs) }
    }

    #[inline(always)]
    fn push(&mut self, add: bool, e: u32) {
        let (list, n) = if add {
            (&mut self.adds, &mut self.n_adds)
        } else {
            (&mut self.subs, &mut self.n_subs)
        };
        debug_assert!(*n < MOVE_THREATS_CAP, "a move recorded more threats than it can make");
        list[*n].write(e);
        *n += 1;
    }

    /// Where the next entries of one list go, for a backend that stores a whole vector
    /// and then counts the entries it meant: `STORE_MARGIN` slots stay writable.
    #[inline(always)]
    #[allow(dead_code)] // the vector-store backends are not built on every target
    fn tail(&mut self, add: bool) -> *mut u32 {
        let (list, n) = if add {
            (&mut self.adds, self.n_adds)
        } else {
            (&mut self.subs, self.n_subs)
        };
        debug_assert!(n + STORE_MARGIN <= MOVE_THREATS_CAP);
        // SAFETY: n + STORE_MARGIN <= MOVE_THREATS_CAP, so the pointer is in bounds.
        unsafe { list.as_mut_ptr().add(n).cast() }
    }

    #[inline(always)]
    #[allow(dead_code)]
    fn grow(&mut self, add: bool, k: usize) {
        let n = if add {
            &mut self.n_adds
        } else {
            &mut self.n_subs
        };
        *n += k;
        debug_assert!(*n <= MOVE_THREATS_CAP);
    }
}

// ============================================================
// Playing the move as elementary changes
// ============================================================

/// A board as a backend holds it, and the two kinds of change it can record on it.
trait RayBackend {
    type Board: Copy;

    unsafe fn load(pos: &Position) -> Self::Board;

    /// Put `piece` (or `Piece::NONE`) on `sq`.
    unsafe fn set(board: &mut Self::Board, sq: Square, piece: Piece);

    /// `piece` arrives on `sq` (`add`) or leaves it.
    unsafe fn change(out: &mut MoveThreats, board: &Self::Board, piece: Piece, sq: Square, add: bool);

    /// The piece on `sq` turns from `old` into `new`.
    unsafe fn mutate(out: &mut MoveThreats, board: &Self::Board, old: Piece, new: Piece, sq: Square);
}

#[inline(always)]
const fn is_king(p: Piece) -> bool {
    p.0 >> 1 == PieceType::King as u8
}

/// The pawns and kings after the move, which the record keeps for the pawn pairs and
/// for the mirroring check.
#[inline(always)]
fn record_pawns_and_kings(pos: &Position, delta: &AccDelta, out: &mut MoveThreats) {
    let mv = delta.mv;
    let from = mv.from_sq();
    let to = mv.to_sq();
    let moved = delta.moved_piece;
    let captured = delta.captured_piece;

    let before = [
        pos.pieces[Piece::WHITE_PAWN.index()],
        pos.pieces[Piece::BLACK_PAWN.index()],
    ];
    let mut after = before;
    let mut kings = [pos.king_sq(Color::White), pos.king_sq(Color::Black)];
    match mv.move_type() {
        MT_CASTLING => kings[moved.color().index()] = mv.castle_king_to(),
        MT_EN_PASSANT => {
            let us = moved.color().index();
            let cap_sq = Square((to.0 as i8 - pawn_push(moved.color())) as u8);
            after[us] = (after[us] & !from.bb()) | to.bb();
            after[us ^ 1] &= !cap_sq.bb();
        }
        mt => {
            if moved.piece_type() == PieceType::Pawn {
                let us = moved.color().index();
                after[us] &= !from.bb();
                if mt != MT_PROMOTION {
                    after[us] |= to.bb();
                }
            } else if is_king(moved) {
                kings[moved.color().index()] = to;
            }
            if captured != Piece::NONE && captured.piece_type() == PieceType::Pawn {
                after[captured.color().index()] &= !to.bb();
            }
        }
    }
    out.pawns_before = before;
    out.pawns_after = after;
    out.king_sq = kings;
}

/// Record `delta` on `pos`, the board before the move, through backend `B`.
#[inline(always)]
unsafe fn record_with<B: RayBackend>(pos: &Position, delta: &AccDelta, out: &mut MoveThreats) {
    out.n_adds = 0;
    out.n_subs = 0;
    record_pawns_and_kings(pos, delta, out);

    let mv = delta.mv;
    debug_assert!(mv != crate::types::Move::NONE, "a null move records nothing");
    let from = mv.from_sq();
    let to = mv.to_sq();
    let moved = delta.moved_piece;
    let captured = delta.captured_piece;
    let mut board = B::load(pos);

    match mv.move_type() {
        MT_NORMAL => {
            B::set(&mut board, from, Piece::NONE);
            B::change(out, &board, moved, from, false);
            if captured == Piece::NONE {
                B::change(out, &board, moved, to, true);
            } else {
                B::mutate(out, &board, captured, moved, to);
            }
        }
        MT_PROMOTION => {
            let promo = Piece::new(mv.promo_type(), moved.color());
            B::set(&mut board, from, Piece::NONE);
            B::change(out, &board, moved, from, false);
            if captured == Piece::NONE {
                B::change(out, &board, promo, to, true);
            } else {
                B::mutate(out, &board, captured, promo, to);
            }
        }
        MT_EN_PASSANT => {
            let cap_sq = Square((to.0 as i8 - pawn_push(moved.color())) as u8);
            B::set(&mut board, from, Piece::NONE);
            B::change(out, &board, moved, from, false);
            B::change(out, &board, moved, to, true);
            B::set(&mut board, to, moved);
            B::change(out, &board, captured, cap_sq, false);
        }
        MT_CASTLING => {
            // Both pieces leave before either arrives: in Chess960 the king may land
            // where the rook stood, or the other way round.
            let rook = Piece::new(PieceType::Rook, moved.color());
            let king_to = mv.castle_king_to();
            let rook_from = to;
            let rook_to = mv.castle_rook_to();
            let king_moves = from != king_to;
            let rook_moves = rook_from != rook_to;
            if king_moves {
                B::change(out, &board, moved, from, false);
                B::set(&mut board, from, Piece::NONE);
            }
            if rook_moves {
                B::change(out, &board, rook, rook_from, false);
                B::set(&mut board, rook_from, Piece::NONE);
            }
            if king_moves {
                B::set(&mut board, king_to, moved);
                B::change(out, &board, moved, king_to, true);
            }
            if rook_moves {
                B::set(&mut board, rook_to, rook);
                B::change(out, &board, rook, rook_to, true);
            }
        }
        _ => debug_assert!(false, "unknown move type"),
    }
}

/// Record the threat changes of `delta`, played on `pos`, on the fastest path this CPU
/// has. Every path records the same entries.
#[inline(always)]
pub(crate) fn record(pos: &Position, delta: &AccDelta, out: &mut MoveThreats) {
    // SAFETY: each path below runs only where its instructions exist: pinned by the
    // build target, or elected from CPUID by `cpu::resolve`.
    unsafe {
        #[cfg(target_arch = "x86_64")]
        {
            if cfg!(all(
                target_feature = "avx512f",
                target_feature = "avx512bw",
                target_feature = "avx512vbmi",
                target_feature = "avx512vbmi2"
            )) {
                x86::record_vbmi2(pos, delta, out)
            } else {
                (crate::cpu::get().move_threats)(pos, delta, out)
            }
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            neon::record(pos, delta, out)
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            wasm::record(pos, delta, out)
        }
        #[cfg(not(any(
            target_arch = "x86_64",
            all(target_arch = "aarch64", target_feature = "neon"),
            all(target_arch = "wasm32", target_feature = "simd128"),
        )))]
        {
            record_scalar(pos, delta, out)
        }
    }
}

// ============================================================
// Bitboard backend
// ============================================================

// Off x86-64 a vector backend runs, and the bitboards serve as the tests' reference.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
struct Scalar;

/// The board as bitboards plus the mailbox that names the piece on a square.
#[derive(Clone, Copy)]
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
struct BitBoard {
    mailbox: [Piece; 64],
    pieces: [u64; 12],
    occ: u64,
}

/// What a bitboard change needs to know of the lines through the focus.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
struct Lines {
    bishop: u64,
    rook: u64,
    diagonal_sliders: u64,
    straight_sliders: u64,
    kings: u64,
}

#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
impl BitBoard {
    #[inline(always)]
    fn lines(&self, sq: Square) -> Lines {
        let p = &self.pieces;
        let queens = p[Piece::WHITE_QUEEN.index()] | p[Piece::BLACK_QUEEN.index()];
        Lines {
            bishop: bishop_attacks(sq, self.occ),
            rook: rook_attacks(sq, self.occ),
            diagonal_sliders: p[Piece::WHITE_BISHOP.index()] | p[Piece::BLACK_BISHOP.index()] | queens,
            straight_sliders: p[Piece::WHITE_ROOK.index()] | p[Piece::BLACK_ROOK.index()] | queens,
            kings: p[Piece::WHITE_KING.index()] | p[Piece::BLACK_KING.index()],
        }
    }

    /// The attacks of `piece` on `sq` and on it, all with sign `add`.
    #[inline(always)]
    fn focus(&self, out: &mut MoveThreats, l: &Lines, piece: Piece, sq: Square, add: bool) {
        let p = &self.pieces;
        let attacks = match piece.piece_type() {
            PieceType::Pawn => pawn_attacks(sq, piece.color()),
            PieceType::Knight => knight_attacks(sq),
            PieceType::Bishop => l.bishop,
            PieceType::Rook => l.rook,
            PieceType::Queen => l.bishop | l.rook,
            PieceType::King => 0,
        };
        let mut targets = attacks & self.occ & !l.kings;
        while targets != 0 {
            let t = pop_lsb(&mut targets);
            out.push(add, entry(piece.0, sq.0, self.mailbox[t.index()].0, t.0));
        }

        let knights = p[Piece::WHITE_KNIGHT.index()] | p[Piece::BLACK_KNIGHT.index()];
        let mut attackers = (p[Piece::WHITE_PAWN.index()] & pawn_attacks(sq, Color::Black))
            | (p[Piece::BLACK_PAWN.index()] & pawn_attacks(sq, Color::White))
            | (knights & knight_attacks(sq))
            | (l.diagonal_sliders & l.bishop)
            | (l.straight_sliders & l.rook);
        while attackers != 0 {
            let a = pop_lsb(&mut attackers);
            out.push(add, entry(self.mailbox[a.index()].0, a.0, piece.0, sq.0));
        }
    }

    /// The attacks of the sliders that see `sq` on the first piece behind it: lost when
    /// a piece arrives on `sq` (`add`), gained when it leaves.
    #[inline(always)]
    fn lines_through(&self, out: &mut MoveThreats, l: &Lines, sq: Square, add: bool) {
        let seen = l.bishop | l.rook;
        let mut sliders = (l.diagonal_sliders & l.bishop) | (l.straight_sliders & l.rook);
        while sliders != 0 {
            let s = pop_lsb(&mut sliders);
            // The part of the line beyond `sq`, away from the slider, up to its first
            // piece: `between_bb(sq, s)` is the part towards the slider, `s` included.
            let behind = seen & line_bb(s, sq) & !between_bb(sq, s) & self.occ & !l.kings;
            debug_assert!(behind.count_ones() <= 1);
            if behind != 0 {
                let v = Square(behind.trailing_zeros() as u8);
                out.push(!add, entry(self.mailbox[s.index()].0, s.0, self.mailbox[v.index()].0, v.0));
            }
        }
    }
}

impl RayBackend for Scalar {
    type Board = BitBoard;

    #[inline]
    unsafe fn load(pos: &Position) -> BitBoard {
        BitBoard {
            mailbox: pos.board,
            pieces: pos.pieces,
            occ: pos.occupied(),
        }
    }

    #[inline]
    unsafe fn set(b: &mut BitBoard, sq: Square, piece: Piece) {
        let bit = sq.bb();
        let old = b.mailbox[sq.index()];
        if old != Piece::NONE {
            b.pieces[old.index()] &= !bit;
        }
        if piece != Piece::NONE {
            b.pieces[piece.index()] |= bit;
            b.occ |= bit;
        } else {
            b.occ &= !bit;
        }
        b.mailbox[sq.index()] = piece;
    }

    #[inline]
    unsafe fn change(out: &mut MoveThreats, b: &BitBoard, piece: Piece, sq: Square, add: bool) {
        let l = b.lines(sq);
        if !is_king(piece) {
            b.focus(out, &l, piece, sq, add);
        }
        b.lines_through(out, &l, sq, add);
    }

    #[inline]
    unsafe fn mutate(out: &mut MoveThreats, b: &BitBoard, old: Piece, new: Piece, sq: Square) {
        let l = b.lines(sq);
        if !is_king(old) {
            b.focus(out, &l, old, sq, false);
        }
        if !is_king(new) {
            b.focus(out, &l, new, sq, true);
        }
    }
}

/// Record with bitboards: the reference every other backend is tested against.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub(crate) unsafe fn record_scalar(pos: &Position, delta: &AccDelta, out: &mut MoveThreats) {
    record_with::<Scalar>(pos, delta, out)
}

// ============================================================
// Ray view — tables shared by the vector backends
// ============================================================

/// 64 bytes on a 64-byte boundary, so a whole row loads in one vector.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Lanes([u8; 64]);

/// Directions of the ray view, as (rank step, file step): N, NE, E, SE, S, SW, W, NW.
/// The opposite of direction `d` is `d + 4`, so the two halves of the view face each
/// other: lane `l` and lane `l ^ 32` look in opposite directions.
const DIRECTIONS: [(i8, i8); 8] = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)];

/// The knight square kept in lane 0 of each direction; together, the eight jumps.
const KNIGHT_JUMPS: [(i8, i8); 8] = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)];

/// Marks a lane whose square is off the board.
const OFF_BOARD: u8 = 0x80;

const fn square_at(rank: i8, file: i8) -> u8 {
    if rank >= 0 && rank < 8 && file >= 0 && file < 8 {
        (rank * 8 + file) as u8
    } else {
        OFF_BOARD
    }
}

/// `RAY_VIEW[focus][8d + k]`: the square of lane `8d + k` around `focus` — the knight
/// jump of direction `d` for `k = 0`, the square `k` steps away in direction `d`
/// otherwise — or `OFF_BOARD`.
static RAY_VIEW: [Lanes; 64] = {
    let mut table = [Lanes([OFF_BOARD; 64]); 64];
    let mut focus = 0usize;
    while focus < 64 {
        let rank = (focus / 8) as i8;
        let file = (focus % 8) as i8;
        let mut d = 0usize;
        while d < 8 {
            table[focus].0[8 * d] = square_at(rank + KNIGHT_JUMPS[d].0, file + KNIGHT_JUMPS[d].1);
            let mut k = 1usize;
            while k < 8 {
                let step = k as i8;
                table[focus].0[8 * d + k] =
                    square_at(rank + DIRECTIONS[d].0 * step, file + DIRECTIONS[d].1 * step);
                k += 1;
            }
            d += 1;
        }
        focus += 1;
    }
    table
};

// One bit per kind of piece in a lane of the view.
const BIT_WHITE_PAWN: u8 = 0x01;
const BIT_BLACK_PAWN: u8 = 0x02;
const BIT_KNIGHT: u8 = 0x04;
const BIT_BISHOP: u8 = 0x08;
const BIT_ROOK: u8 = 0x10;
const BIT_QUEEN: u8 = 0x20;
const BIT_KING: u8 = 0x40;

/// The kind bit of each piece code; `Piece::NONE` (12) and the unused codes have none.
#[repr(C, align(16))]
struct Kinds([u8; 16]);
static KIND_OF_PIECE: Kinds = Kinds([
    BIT_WHITE_PAWN, BIT_BLACK_PAWN, BIT_KNIGHT, BIT_KNIGHT, BIT_BISHOP, BIT_BISHOP, BIT_ROOK,
    BIT_ROOK, BIT_QUEEN, BIT_QUEEN, BIT_KING, BIT_KING, 0, 0, 0, 0,
]);

const fn is_diagonal(d: usize) -> bool {
    d & 1 == 1
}

/// Per lane, the kinds that attack the focus from there when they are the first piece
/// of the lane: a knight on a knight square, a slider of the line's kind, and a pawn of
/// the colour that strikes upward (white, from SE or SW) or downward (black, from NE or
/// NW) one step away. Kings are left out: their attacks are never recorded.
static ATTACKS_FOCUS: Lanes = {
    let mut m = [0u8; 64];
    let mut d = 0usize;
    while d < 8 {
        let line = if is_diagonal(d) { BIT_BISHOP | BIT_QUEEN } else { BIT_ROOK | BIT_QUEEN };
        m[8 * d] = BIT_KNIGHT;
        let mut k = 1usize;
        while k < 8 {
            m[8 * d + k] = line;
            k += 1;
        }
        if d == 3 || d == 5 {
            m[8 * d + 1] |= BIT_WHITE_PAWN;
        } else if d == 1 || d == 7 {
            m[8 * d + 1] |= BIT_BLACK_PAWN;
        }
        d += 1;
    }
    Lanes(m)
};

/// Per lane, the sliders that see the focus along the lane's line.
static SLIDES_TO_FOCUS: Lanes = {
    let mut m = [0u8; 64];
    let mut d = 0usize;
    while d < 8 {
        let line = if is_diagonal(d) { BIT_BISHOP | BIT_QUEEN } else { BIT_ROOK | BIT_QUEEN };
        let mut k = 1usize;
        while k < 8 {
            m[8 * d + k] = line;
            k += 1;
        }
        d += 1;
    }
    Lanes(m)
};

const KNIGHT_LANES: u64 = 0x0101_0101_0101_0101;
const RAY_LANES: u64 = !KNIGHT_LANES;

/// Per piece code, the lanes it attacks from the focus when they hold a first piece.
static ATTACKED_FROM_FOCUS: [u64; 16] = {
    let mut diagonal = 0u64;
    let mut straight = 0u64;
    let mut d = 0usize;
    while d < 8 {
        if is_diagonal(d) {
            diagonal |= 0xFE << (8 * d);
        } else {
            straight |= 0xFE << (8 * d);
        }
        d += 1;
    }
    // A pawn strikes the first square of its two forward diagonals.
    let white_pawn = 1u64 << (8 + 1) | 1u64 << (8 * 7 + 1);
    let black_pawn = 1u64 << (8 * 3 + 1) | 1u64 << (8 * 5 + 1);
    [
        white_pawn, black_pawn, KNIGHT_LANES, KNIGHT_LANES, diagonal, diagonal, straight,
        straight, diagonal | straight, diagonal | straight, 0, 0, 0, 0, 0, 0,
    ]
};

/// The lanes of a view the changes read, from the four raw lane masks a backend reads
/// off the permuted mailbox.
#[derive(Clone, Copy)]
struct RayMasks {
    /// The first piece of each ray, and every occupied knight square.
    first: u64,
    /// First pieces that attack the focus.
    attackers: u64,
    /// First pieces that are sliders of their line.
    sliders: u64,
    kings: u64,
}

impl RayMasks {
    /// `occupied`, `attack_kinds`, `slide_kinds` and `kings`: the lanes whose kind bits
    /// are non-zero, meet `ATTACKS_FOCUS`, meet `SLIDES_TO_FOCUS`, hold a king.
    #[inline(always)]
    fn new(occupied: u64, attack_kinds: u64, slide_kinds: u64, kings: u64) -> Self {
        // Per ray byte, with bit 0 (the knight square) and bit 7 forced: subtracting 3
        // clears bit 0 and borrows up to the lowest occupied ray square, which bit 7
        // stops inside the byte. The xor keeps bit 0 and every bit up to that square.
        let o = occupied | 0x8181_8181_8181_8181;
        let first = (o ^ o.wrapping_sub(0x0303_0303_0303_0303)) & occupied;
        RayMasks {
            first,
            attackers: first & attack_kinds,
            sliders: first & slide_kinds & RAY_LANES,
            kings,
        }
    }

    /// The lanes `piece` on the focus attacks.
    #[inline(always)]
    fn attacked(&self, piece: Piece) -> u64 {
        ATTACKED_FROM_FOCUS[piece.index()] & self.first & !self.kings
    }

    /// The line changes: `(sliders, behind)`, a slider lane and, in the facing half of
    /// the view, the lane of the first non-king piece behind the focus, for every line
    /// that has both. `behind` names its lanes with the halves swapped (lane `l` stands
    /// for lane `l ^ 32`), so the two masks hold one bit in the same bytes.
    #[inline(always)]
    fn lines(&self) -> (u64, u64) {
        let behind = (self.first & !self.kings & RAY_LANES).rotate_right(32);
        let both = ray_fill(behind) & ray_fill(self.sliders);
        (self.sliders & both, behind & both)
    }
}

/// 0x7F in every byte that holds a bit in 1..=7, 0 elsewhere.
#[inline(always)]
const fn ray_fill(x: u64) -> u64 {
    let t = x.wrapping_add(0x7E7E_7E7E_7E7E_7E7E) & 0x8080_8080_8080_8080;
    t.wrapping_sub(t >> 7)
}

/// Record the attacks of `piece` on `focus` read off a view: the backend stored the
/// permuted mailbox in `pieces`, and `squares` is the view's row of `RAY_VIEW`.
#[inline(always)]
#[allow(dead_code)]
fn view_focus(out: &mut MoveThreats, pieces: &[u8; 64], squares: &[u8; 64], m: &RayMasks, piece: Piece, focus: Square, add: bool) {
    let mut t = m.attacked(piece);
    while t != 0 {
        let l = t.trailing_zeros() as usize;
        t &= t - 1;
        out.push(add, entry(piece.0, focus.0, pieces[l], squares[l]));
    }
    let mut a = m.attackers;
    while a != 0 {
        let l = a.trailing_zeros() as usize;
        a &= a - 1;
        out.push(add, entry(pieces[l], squares[l], piece.0, focus.0));
    }
}

/// Record the line changes read off a view, with the sign opposite to the change's.
#[inline(always)]
#[allow(dead_code)]
fn view_lines(out: &mut MoveThreats, pieces: &[u8; 64], squares: &[u8; 64], m: &RayMasks, add: bool) {
    let (mut s, mut v) = m.lines();
    while s != 0 {
        let ls = s.trailing_zeros() as usize;
        let lv = v.trailing_zeros() as usize ^ 32;
        s &= s - 1;
        v &= v - 1;
        out.push(!add, entry(pieces[ls], squares[ls], pieces[lv], squares[lv]));
    }
}

/// A change read off a view.
#[inline(always)]
#[allow(dead_code)]
fn view_change(out: &mut MoveThreats, pieces: &[u8; 64], squares: &[u8; 64], m: &RayMasks, piece: Piece, sq: Square, add: bool) {
    if !is_king(piece) {
        view_focus(out, pieces, squares, m, piece, sq, add);
    }
    view_lines(out, pieces, squares, m, add);
}

/// A turn read off a view.
#[inline(always)]
#[allow(dead_code)]
fn view_mutate(out: &mut MoveThreats, pieces: &[u8; 64], squares: &[u8; 64], m: &RayMasks, old: Piece, new: Piece, sq: Square) {
    if !is_king(old) {
        view_focus(out, pieces, squares, m, old, sq, false);
    }
    if !is_king(new) {
        view_focus(out, pieces, squares, m, new, sq, true);
    }
}

// ============================================================
// x86-64: AVX-512 VBMI2 and AVX2
// ============================================================

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86 {
    use super::*;
    use std::arch::x86_64::*;

    // --- AVX-512 VBMI2: the view in one register, entries written by compression ---

    struct Vbmi2;

    /// A view: the squares of its lanes, the pieces on them, and its masks.
    struct View512 {
        squares: __m512i,
        pieces: __m512i,
        masks: RayMasks,
    }

    /// Byte `4i` of the result takes byte `i` of the first operand, byte `4i + 1` byte
    /// `i` of the second, and the pattern repeats in bytes `4i + 2` and `4i + 3`: the
    /// (piece, square) pairs of two compressed lists, twice per entry.
    static PAIR_UP: Lanes = {
        let mut s = [0u8; 64];
        let mut i = 0usize;
        while i < 16 {
            s[4 * i] = i as u8;
            s[4 * i + 1] = 64 + i as u8;
            s[4 * i + 2] = i as u8;
            s[4 * i + 3] = 64 + i as u8;
            i += 1;
        }
        Lanes(s)
    };

    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512vbmi2")]
    unsafe fn view512(board: &__m512i, focus: Square) -> View512 {
        let squares = _mm512_load_si512(RAY_VIEW[focus.index()].0.as_ptr().cast());
        let on_board = _mm512_testn_epi8_mask(squares, _mm512_set1_epi8(OFF_BOARD as i8));
        let pieces = _mm512_permutexvar_epi8(squares, *board);
        let lut = _mm512_broadcast_i32x4(_mm_load_si128(KIND_OF_PIECE.0.as_ptr().cast()));
        let kinds = _mm512_maskz_shuffle_epi8(on_board, lut, pieces);
        let attack = _mm512_load_si512(ATTACKS_FOCUS.0.as_ptr().cast());
        let slide = _mm512_load_si512(SLIDES_TO_FOCUS.0.as_ptr().cast());
        let masks = RayMasks::new(
            _mm512_test_epi8_mask(kinds, kinds),
            _mm512_test_epi8_mask(kinds, attack),
            _mm512_test_epi8_mask(kinds, slide),
            _mm512_test_epi8_mask(kinds, _mm512_set1_epi8(BIT_KING as i8)),
        );
        View512 { squares, pieces, masks }
    }

    /// Write the entries pairing `piece` on `focus` with the pieces of `lanes`, the focus
    /// as attacker (`outgoing`) or as target.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512vbmi2")]
    unsafe fn write_focus(out: &mut MoveThreats, v: &View512, lanes: u64, piece: Piece, focus: Square, add: bool, outgoing: bool) {
        if lanes == 0 {
            return;
        }
        debug_assert!(lanes.count_ones() <= 16);
        let pieces = _mm512_maskz_compress_epi8(lanes, v.pieces);
        let squares = _mm512_maskz_compress_epi8(lanes, v.squares);
        let other = _mm512_permutex2var_epi8(pieces, _mm512_load_si512(PAIR_UP.0.as_ptr().cast()), squares);
        let own = _mm512_set1_epi16((piece.0 as u16 | (focus.0 as u16) << 8) as i16);
        // Bytes 2-3 of every entry from the other piece when the focus attacks it,
        // bytes 0-1 when it is attacked.
        let take = if outgoing { 0xCCCC_CCCC_CCCC_CCCC } else { 0x3333_3333_3333_3333 };
        _mm512_storeu_si512(out.tail(add).cast(), _mm512_mask_mov_epi8(own, take, other));
        out.grow(add, lanes.count_ones() as usize);
    }

    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512vbmi2")]
    unsafe fn write_lines(out: &mut MoveThreats, v: &View512, add: bool) {
        let (sliders, behind) = v.masks.lines();
        if sliders == 0 {
            return;
        }
        // The facing half of the view: lane `l` of a swapped vector holds lane `l ^ 32`.
        let facing_pieces = _mm512_shuffle_i64x2::<0b01_00_11_10>(v.pieces, v.pieces);
        let facing_squares = _mm512_shuffle_i64x2::<0b01_00_11_10>(v.squares, v.squares);
        let slider = _mm_unpacklo_epi8(
            _mm512_castsi512_si128(_mm512_maskz_compress_epi8(sliders, v.pieces)),
            _mm512_castsi512_si128(_mm512_maskz_compress_epi8(sliders, v.squares)),
        );
        let target = _mm_unpacklo_epi8(
            _mm512_castsi512_si128(_mm512_maskz_compress_epi8(behind, facing_pieces)),
            _mm512_castsi512_si128(_mm512_maskz_compress_epi8(behind, facing_squares)),
        );
        let dst = out.tail(!add).cast::<__m128i>();
        _mm_storeu_si128(dst, _mm_unpacklo_epi16(slider, target));
        _mm_storeu_si128(dst.add(1), _mm_unpackhi_epi16(slider, target));
        out.grow(!add, sliders.count_ones() as usize);
    }

    impl RayBackend for Vbmi2 {
        type Board = __m512i;

        #[inline]
        #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512vbmi2")]
        unsafe fn load(pos: &Position) -> __m512i {
            _mm512_loadu_si512(pos.board.as_ptr().cast())
        }

        #[inline]
        #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512vbmi2")]
        unsafe fn set(board: &mut __m512i, sq: Square, piece: Piece) {
            *board = _mm512_mask_blend_epi8(sq.bb(), *board, _mm512_set1_epi8(piece.0 as i8));
        }

        #[inline]
        #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512vbmi2")]
        unsafe fn change(out: &mut MoveThreats, board: &__m512i, piece: Piece, sq: Square, add: bool) {
            let v = view512(board, sq);
            if !is_king(piece) {
                write_focus(out, &v, v.masks.attacked(piece), piece, sq, add, true);
                write_focus(out, &v, v.masks.attackers, piece, sq, add, false);
            }
            write_lines(out, &v, add);
        }

        #[inline]
        #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512vbmi2")]
        unsafe fn mutate(out: &mut MoveThreats, board: &__m512i, old: Piece, new: Piece, sq: Square) {
            let v = view512(board, sq);
            for (piece, add) in [(old, false), (new, true)] {
                if !is_king(piece) {
                    write_focus(out, &v, v.masks.attacked(piece), piece, sq, add, true);
                    write_focus(out, &v, v.masks.attackers, piece, sq, add, false);
                }
            }
        }
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512vbmi2")]
    pub(crate) unsafe fn record_vbmi2(pos: &Position, delta: &AccDelta, out: &mut MoveThreats) {
        record_with::<Vbmi2>(pos, delta, out)
    }

    // --- AVX2: the view in two registers, entries written lane by lane ---

    struct Avx2;

    static ASCENDING: Lanes = {
        let mut s = [0u8; 64];
        let mut i = 0usize;
        while i < 64 {
            s[i] = i as u8;
            i += 1;
        }
        Lanes(s)
    };

    /// The lanes of `x` that are non-zero, one bit each.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn nonzero(x: __m256i) -> u32 {
        !(_mm256_movemask_epi8(_mm256_cmpeq_epi8(x, _mm256_setzero_si256())) as u32)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn view256(board: &[__m256i; 2], focus: Square, pieces: &mut Lanes) -> RayMasks {
        // A 64-byte table lookup from 16-byte shuffles: each quarter of the mailbox is
        // broadcast to both halves of a register, index bit 4 picks between the two
        // quarters of a half and bit 5 between the halves.
        let quarter = [
            _mm256_permute2x128_si256::<0x00>(board[0], board[0]),
            _mm256_permute2x128_si256::<0x11>(board[0], board[0]),
            _mm256_permute2x128_si256::<0x00>(board[1], board[1]),
            _mm256_permute2x128_si256::<0x11>(board[1], board[1]),
        ];
        let lut = _mm256_broadcastsi128_si256(_mm_load_si128(KIND_OF_PIECE.0.as_ptr().cast()));
        let view = &RAY_VIEW[focus.index()].0;
        let (mut occupied, mut attack, mut slide, mut kings) = (0u64, 0u64, 0u64, 0u64);
        for h in 0..2 {
            let squares = _mm256_load_si256(view.as_ptr().add(32 * h).cast());
            let bit4 = _mm256_slli_epi16::<3>(squares);
            let bit5 = _mm256_slli_epi16::<2>(squares);
            let low = _mm256_blendv_epi8(
                _mm256_shuffle_epi8(quarter[0], squares),
                _mm256_shuffle_epi8(quarter[1], squares),
                bit4,
            );
            let high = _mm256_blendv_epi8(
                _mm256_shuffle_epi8(quarter[2], squares),
                _mm256_shuffle_epi8(quarter[3], squares),
                bit4,
            );
            let on = _mm256_blendv_epi8(low, high, bit5);
            _mm256_store_si256(pieces.0.as_mut_ptr().add(32 * h).cast(), on);
            let off = _mm256_cmpgt_epi8(_mm256_setzero_si256(), squares);
            let kinds = _mm256_andnot_si256(off, _mm256_shuffle_epi8(lut, on));
            let shift = 32 * h as u32;
            let a = _mm256_load_si256(ATTACKS_FOCUS.0.as_ptr().add(32 * h).cast());
            let s = _mm256_load_si256(SLIDES_TO_FOCUS.0.as_ptr().add(32 * h).cast());
            occupied |= (nonzero(kinds) as u64) << shift;
            attack |= (nonzero(_mm256_and_si256(kinds, a)) as u64) << shift;
            slide |= (nonzero(_mm256_and_si256(kinds, s)) as u64) << shift;
            kings |= (nonzero(_mm256_and_si256(kinds, _mm256_set1_epi8(BIT_KING as i8))) as u64) << shift;
        }
        RayMasks::new(occupied, attack, slide, kings)
    }

    impl RayBackend for Avx2 {
        type Board = [__m256i; 2];

        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn load(pos: &Position) -> [__m256i; 2] {
            let p = pos.board.as_ptr().cast::<__m256i>();
            [_mm256_loadu_si256(p), _mm256_loadu_si256(p.add(1))]
        }

        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn set(board: &mut [__m256i; 2], sq: Square, piece: Piece) {
            let at = _mm256_set1_epi8(sq.0 as i8);
            let value = _mm256_set1_epi8(piece.0 as i8);
            for h in 0..2 {
                let squares = _mm256_load_si256(ASCENDING.0.as_ptr().add(32 * h).cast());
                board[h] = _mm256_blendv_epi8(board[h], value, _mm256_cmpeq_epi8(squares, at));
            }
        }

        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn change(out: &mut MoveThreats, board: &[__m256i; 2], piece: Piece, sq: Square, add: bool) {
            let mut pieces = Lanes([0; 64]);
            let m = view256(board, sq, &mut pieces);
            view_change(out, &pieces.0, &RAY_VIEW[sq.index()].0, &m, piece, sq, add);
        }

        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn mutate(out: &mut MoveThreats, board: &[__m256i; 2], old: Piece, new: Piece, sq: Square) {
            let mut pieces = Lanes([0; 64]);
            let m = view256(board, sq, &mut pieces);
            view_mutate(out, &pieces.0, &RAY_VIEW[sq.index()].0, &m, old, new, sq);
        }
    }

    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn record_avx2(pos: &Position, delta: &AccDelta, out: &mut MoveThreats) {
        record_with::<Avx2>(pos, delta, out)
    }
}

// ============================================================
// aarch64: NEON
// ============================================================

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon {
    use super::*;
    use std::arch::aarch64::*;

    struct Neon;

    static ASCENDING: Lanes = {
        let mut s = [0u8; 64];
        let mut i = 0usize;
        while i < 64 {
            s[i] = i as u8;
            i += 1;
        }
        Lanes(s)
    };

    /// Weight of each lane inside its group of eight, for the mask gather.
    static LANE_WEIGHTS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];

    /// One bit per lane of four all-ones-or-zero vectors: each lane is weighted by its
    /// bit, then pairwise additions fold every group of eight lanes into a byte.
    #[inline]
    unsafe fn lane_mask(v: [uint8x16_t; 4]) -> u64 {
        let w = vld1q_u8(LANE_WEIGHTS.as_ptr());
        let a = vpaddq_u8(vandq_u8(v[0], w), vandq_u8(v[1], w));
        let b = vpaddq_u8(vandq_u8(v[2], w), vandq_u8(v[3], w));
        let c = vpaddq_u8(a, b);
        vgetq_lane_u64::<0>(vreinterpretq_u64_u8(vpaddq_u8(c, c)))
    }

    #[inline]
    unsafe fn view128(board: &uint8x16x4_t, focus: Square, pieces: &mut Lanes) -> RayMasks {
        let view = &RAY_VIEW[focus.index()].0;
        let lut = vld1q_u8(KIND_OF_PIECE.0.as_ptr());
        let attack_row = &ATTACKS_FOCUS.0;
        let slide_row = &SLIDES_TO_FOCUS.0;
        let king = vdupq_n_u8(BIT_KING);
        let mut kinds = [vdupq_n_u8(0); 4];
        let mut attack = [vdupq_n_u8(0); 4];
        let mut slide = [vdupq_n_u8(0); 4];
        let mut kings = [vdupq_n_u8(0); 4];
        for q in 0..4 {
            let squares = vld1q_u8(view.as_ptr().add(16 * q));
            // Indices past 63 (off the board) read 0 here, the white pawn's code: the
            // on-board test below clears their kinds.
            let on = vqtbl4q_u8(*board, squares);
            vst1q_u8(pieces.0.as_mut_ptr().add(16 * q), on);
            let on_board = vcltq_u8(squares, vdupq_n_u8(64));
            let k = vandq_u8(vqtbl1q_u8(lut, on), on_board);
            kinds[q] = vtstq_u8(k, k);
            attack[q] = vtstq_u8(k, vld1q_u8(attack_row.as_ptr().add(16 * q)));
            slide[q] = vtstq_u8(k, vld1q_u8(slide_row.as_ptr().add(16 * q)));
            kings[q] = vtstq_u8(k, king);
        }
        RayMasks::new(lane_mask(kinds), lane_mask(attack), lane_mask(slide), lane_mask(kings))
    }

    impl RayBackend for Neon {
        type Board = uint8x16x4_t;

        #[inline]
        unsafe fn load(pos: &Position) -> uint8x16x4_t {
            vld1q_u8_x4(pos.board.as_ptr().cast())
        }

        #[inline]
        unsafe fn set(board: &mut uint8x16x4_t, sq: Square, piece: Piece) {
            let at = vdupq_n_u8(sq.0);
            let value = vdupq_n_u8(piece.0);
            let hit = |q: usize| vceqq_u8(vld1q_u8(ASCENDING.0.as_ptr().add(16 * q)), at);
            board.0 = vbslq_u8(hit(0), value, board.0);
            board.1 = vbslq_u8(hit(1), value, board.1);
            board.2 = vbslq_u8(hit(2), value, board.2);
            board.3 = vbslq_u8(hit(3), value, board.3);
        }

        #[inline]
        unsafe fn change(out: &mut MoveThreats, board: &uint8x16x4_t, piece: Piece, sq: Square, add: bool) {
            let mut pieces = Lanes([0; 64]);
            let m = view128(board, sq, &mut pieces);
            view_change(out, &pieces.0, &RAY_VIEW[sq.index()].0, &m, piece, sq, add);
        }

        #[inline]
        unsafe fn mutate(out: &mut MoveThreats, board: &uint8x16x4_t, old: Piece, new: Piece, sq: Square) {
            let mut pieces = Lanes([0; 64]);
            let m = view128(board, sq, &mut pieces);
            view_mutate(out, &pieces.0, &RAY_VIEW[sq.index()].0, &m, old, new, sq);
        }
    }

    pub(super) unsafe fn record(pos: &Position, delta: &AccDelta, out: &mut MoveThreats) {
        record_with::<Neon>(pos, delta, out)
    }
}

// ============================================================
// wasm32: simd128
// ============================================================

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm {
    use super::*;
    use std::arch::wasm32::*;

    struct Simd128;

    static ASCENDING: Lanes = {
        let mut s = [0u8; 64];
        let mut i = 0usize;
        while i < 64 {
            s[i] = i as u8;
            i += 1;
        }
        Lanes(s)
    };

    #[inline]
    unsafe fn row(bytes: &[u8; 64], q: usize) -> v128 {
        v128_load(bytes.as_ptr().add(16 * q).cast())
    }

    #[inline]
    fn lane_mask(v: [v128; 4]) -> u64 {
        let mut m = 0u64;
        for (q, x) in v.iter().enumerate() {
            m |= (i8x16_bitmask(*x) as u64) << (16 * q);
        }
        m
    }

    #[inline]
    unsafe fn view128(board: &[v128; 4], focus: Square, pieces: &mut Lanes) -> RayMasks {
        let view = &RAY_VIEW[focus.index()].0;
        let lut = v128_load(KIND_OF_PIECE.0.as_ptr().cast());
        let zero = u8x16_splat(0);
        let king = u8x16_splat(BIT_KING);
        let mut kinds = [zero; 4];
        let mut attack = [zero; 4];
        let mut slide = [zero; 4];
        let mut kings = [zero; 4];
        for q in 0..4 {
            let squares = row(view, q);
            // Swizzle reads 0 for an index past 15: each quarter of the mailbox answers
            // only for its own sixteen squares, and off-board indices for none.
            let mut on = zero;
            for j in 0..4 {
                let local = u8x16_sub(squares, u8x16_splat(16 * j as u8));
                on = v128_or(on, i8x16_swizzle(board[j], local));
            }
            v128_store(pieces.0.as_mut_ptr().add(16 * q).cast(), on);
            let on_board = u8x16_lt(squares, u8x16_splat(64));
            let k = v128_and(i8x16_swizzle(lut, on), on_board);
            kinds[q] = i8x16_ne(k, zero);
            attack[q] = i8x16_ne(v128_and(k, row(&ATTACKS_FOCUS.0, q)), zero);
            slide[q] = i8x16_ne(v128_and(k, row(&SLIDES_TO_FOCUS.0, q)), zero);
            kings[q] = i8x16_ne(v128_and(k, king), zero);
        }
        RayMasks::new(lane_mask(kinds), lane_mask(attack), lane_mask(slide), lane_mask(kings))
    }

    impl RayBackend for Simd128 {
        type Board = [v128; 4];

        #[inline]
        unsafe fn load(pos: &Position) -> [v128; 4] {
            let p = pos.board.as_ptr().cast::<v128>();
            [v128_load(p), v128_load(p.add(1)), v128_load(p.add(2)), v128_load(p.add(3))]
        }

        #[inline]
        unsafe fn set(board: &mut [v128; 4], sq: Square, piece: Piece) {
            let at = u8x16_splat(sq.0);
            let value = u8x16_splat(piece.0);
            for q in 0..4 {
                let hit = i8x16_eq(row(&ASCENDING.0, q), at);
                board[q] = v128_bitselect(value, board[q], hit);
            }
        }

        #[inline]
        unsafe fn change(out: &mut MoveThreats, board: &[v128; 4], piece: Piece, sq: Square, add: bool) {
            let mut pieces = Lanes([0; 64]);
            let m = view128(board, sq, &mut pieces);
            view_change(out, &pieces.0, &RAY_VIEW[sq.index()].0, &m, piece, sq, add);
        }

        #[inline]
        unsafe fn mutate(out: &mut MoveThreats, board: &[v128; 4], old: Piece, new: Piece, sq: Square) {
            let mut pieces = Lanes([0; 64]);
            let m = view128(board, sq, &mut pieces);
            view_mutate(out, &pieces.0, &RAY_VIEW[sq.index()].0, &m, old, new, sq);
        }
    }

    pub(super) unsafe fn record(pos: &Position, delta: &AccDelta, out: &mut MoveThreats) {
        record_with::<Simd128>(pos, delta, out)
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::bitboard::king_attacks;
    use crate::movegen::generate_legal_moves;
    use crate::types::{ArrayBuf, MAX_MOVES, Move};

    type Recorder = unsafe fn(&Position, &AccDelta, &mut MoveThreats);

    /// Every backend this machine can run, the bitboards first.
    fn backends() -> Vec<(&'static str, Recorder)> {
        let mut v: Vec<(&'static str, Recorder)> = vec![("scalar", record_scalar)];
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                v.push(("avx2", x86::record_avx2));
            }
            if std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512bw")
                && std::arch::is_x86_feature_detected!("avx512vbmi")
                && std::arch::is_x86_feature_detected!("avx512vbmi2")
            {
                v.push(("vbmi2", x86::record_vbmi2));
            }
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        v.push(("neon", neon::record));
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        v.push(("simd128", wasm::record));
        v
    }

    fn delta_of(pos: &Position, mv: Move) -> AccDelta {
        AccDelta {
            mv,
            moved_piece: pos.piece_on(mv.from_sq()),
            captured_piece: if mv.move_type() == MT_EN_PASSANT {
                Piece::new(PieceType::Pawn, !pos.side_to_move)
            } else if mv.move_type() == MT_CASTLING {
                Piece::NONE
            } else {
                pos.piece_on(mv.to_sq())
            },
            old_occ: pos.occupied(),
        }
    }

    /// Every attack on the board between two non-kings, counted.
    fn all_attacks(pos: &Position) -> HashMap<u32, i32> {
        let occ = pos.occupied();
        let mut out = HashMap::new();
        for sq in 0..64u8 {
            let s = Square(sq);
            let p = pos.piece_on(s);
            if p == Piece::NONE || is_king(p) {
                continue;
            }
            let attacks = match p.piece_type() {
                PieceType::Pawn => pawn_attacks(s, p.color()),
                PieceType::Knight => knight_attacks(s),
                PieceType::Bishop => bishop_attacks(s, occ),
                PieceType::Rook => rook_attacks(s, occ),
                PieceType::Queen => bishop_attacks(s, occ) | rook_attacks(s, occ),
                PieceType::King => king_attacks(s),
            };
            let mut t = attacks & occ;
            while t != 0 {
                let ts = pop_lsb(&mut t);
                let target = pos.piece_on(ts);
                if !is_king(target) {
                    *out.entry(entry(p.0, sq, target.0, ts.0)).or_insert(0) += 1;
                }
            }
        }
        out
    }

    fn sorted(v: &[u32]) -> Vec<u32> {
        let mut v = v.to_vec();
        v.sort_unstable();
        v
    }

    /// One move checked on every backend: the record is the exact difference of the
    /// attack sets, the bookkeeping matches the board after the move, and every backend
    /// records the same entries as the bitboards.
    fn check_move(pos: &Position, mv: Move, backends: &[(&'static str, Recorder)]) {
        let delta = delta_of(pos, mv);
        let before = all_attacks(pos);
        let mut next = pos.clone();
        next.make_move(mv);
        let mut after = all_attacks(&next);
        after.retain(|_, c| *c != 0);

        let mut reference: Option<(Vec<u32>, Vec<u32>)> = None;
        for &(name, f) in backends {
            let mut rec = MoveThreats::new();
            unsafe { f(pos, &delta, &mut rec) };
            let mut net = before.clone();
            for &e in rec.adds() {
                *net.entry(e).or_insert(0) += 1;
            }
            for &e in rec.subs() {
                *net.entry(e).or_insert(0) -= 1;
            }
            net.retain(|_, c| *c != 0);
            assert_eq!(net, after, "{name}: {mv:?} on {}", pos.to_fen());
            assert_eq!(
                rec.pawns_after,
                [next.pieces[Piece::WHITE_PAWN.index()], next.pieces[Piece::BLACK_PAWN.index()]],
                "{name}: pawns after {mv:?}"
            );
            assert_eq!(
                rec.king_sq,
                [next.king_sq(Color::White), next.king_sq(Color::Black)],
                "{name}: kings after {mv:?}"
            );
            let lists = (sorted(rec.adds()), sorted(rec.subs()));
            match &reference {
                None => reference = Some(lists),
                Some(r) => assert_eq!(&lists, r, "{name} differs from the bitboards on {mv:?}"),
            }
        }
    }

    fn legal_moves(pos: &Position) -> Vec<Move> {
        let mut buf = ArrayBuf::<Move, MAX_MOVES>::new();
        let n = generate_legal_moves(pos, &mut buf);
        buf.filled(n).to_vec()
    }

    #[test]
    fn every_backend_records_the_exact_change_of_every_legal_move() {
        let backends = backends();
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
            "3r3k/4P3/8/8/8/8/8/4K3 w - - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "4k3/8/8/8/8/8/8/5KR1 w K - 0 1",
            "4k3/8/8/8/8/8/8/6KR w K - 0 1",
            "1q2r1k1/1b3p1p/p2p2p1/2pPn3/2P1P3/1P2BQ1P/P5P1/3R2K1 b - - 0 25",
        ];
        for fen in fens {
            let pos = Position::from_fen(fen).unwrap();
            for mv in legal_moves(&pos) {
                check_move(&pos, mv, &backends);
            }
        }

        // Random games reach what the fixed positions miss: promotions with capture,
        // en passant next to sliders, crowded lines.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        for start in [fens[0], fens[1], fens[2]] {
            for _ in 0..8 {
                let mut pos = Position::from_fen(start).unwrap();
                for _ in 0..160 {
                    let moves = legal_moves(&pos);
                    if moves.is_empty() {
                        break;
                    }
                    for &mv in &moves {
                        check_move(&pos, mv, &backends);
                    }
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    pos.make_move(moves[(state % moves.len() as u64) as usize]);
                }
            }
        }
    }

    #[test]
    fn the_ray_view_names_every_square_once_per_lane_kind() {
        for focus in 0..64usize {
            let lanes = &RAY_VIEW[focus].0;
            for d in 0..8 {
                // Ray squares run outward and stop at the edge.
                let mut ended = false;
                for k in 1..8 {
                    let s = lanes[8 * d + k];
                    if s == OFF_BOARD {
                        ended = true;
                    } else {
                        assert!(!ended, "focus {focus} dir {d}: a square after the edge");
                        assert!(s < 64);
                    }
                }
            }
            // Knight lanes are exactly the knight's jumps.
            let mut jumps = 0u64;
            for d in 0..8 {
                let s = lanes[8 * d];
                if s != OFF_BOARD {
                    jumps |= 1u64 << s;
                }
            }
            assert_eq!(jumps, knight_attacks(Square(focus as u8)), "focus {focus}");
        }
    }
}
