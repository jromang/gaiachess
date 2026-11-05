//! NNUE evaluation: efficiently updatable neural network.
//!
//! Architecture GaiaNet-T7:
//!   ```text
//!   Position
//!     |
//!     +-- K32Q2 piece-square block (32 king buckets x opponent queen x 704 = 45,056, i8)
//!     +-- Pawn pairs (4,560, i8, 3-file window) + pairwise threats (59,808, i8)
//!     |     -> [Accumulator i16[1024] x 2 perspectives: all three summed, updated in one
//!     |         pass per move; Finny cache for the piece squares, pawn-pair cache]
//!     |
//!     +----> activate_ft -> within-side pairwise -> u8[1024]
//!              -> L1 sparse (1024->32, dpbusd) -> [x² >> 21, x >> 7] clipped -> u8[64]
//!              -> L2 dense (64->64, dpbusd) -> gate g[32], value v[32]
//!              -> h = (g·g·v) >> 14, signed
//!              -> L3 (concat(u8[64], h[32]) -> 1) + skip (l1[30] - l1[31])
//!              x 32 layer stacks, integer from end to end -> centipawns
//!   ```
//!
//! Submodules:
//!   - `features`: K32Q2 indexing (32 king buckets, opponent queen, horizontal mirroring)
//!   - `threats`: pairwise threat features (59,808), incremental dirty updates
//!   - `move_threats`: the threat changes of a move, recorded when it is played
//!   - `pawn_pairs`: 3-file pawn-pawn relations (4,560)
//!   - `accumulator`: the accumulator, piece-square changes of a move, Finny cache
//!   - `network`: NNUEParams struct (~114 MB), weight loading
//!   - `forward`: forward pass helpers (NNZ, reference evaluation for the tests)
//!   - `simd`: SIMD primitives (AVX-512 > AVX2 > NEON > scalar)
//!
//! Dual perspective (STM + NSTM). SIMD-accelerated with a scalar fallback; every tier
//! computes the same integers, so the evaluation is identical on all of them.

pub mod accumulator;
pub mod features;
pub(crate) mod forward;
pub mod ftperm;
pub mod ftperm_order;
pub mod integrity;
pub(crate) mod kernels;
pub(crate) mod move_threats;
pub mod network;
pub mod pawn_pairs;
pub mod simd;
pub mod threats;

pub use network::{replica, replica_count, replicate};

/// SIMD backend actually running, for the UCI `info` line.
///
/// On aarch64 this is the runtime election (NEON vs NEON+DOTPROD). Other
/// architectures report their compile-time backend from `main`.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub fn simd_label() -> &'static str {
    kernels::arm::label()
}

use crate::position::Position;
use crate::types::*;

use accumulator::{AccDelta, Accumulator, FinnyTable};
use threats::PawnPairCache;

// ============================================================
// Architecture constants
// ============================================================

/// King buckets: one per square of files a-d (see features::KING_BUCKETS).
pub const KING_BUCKET_COUNT: usize = 32;

/// States of the opponent's queen: absent, present.
pub const QUEEN_STATES: usize = 2;

/// Piece-square blocks: a king bucket and the opponent queen's state.
pub const PSQ_BLOCKS: usize = KING_BUCKET_COUNT * QUEEN_STATES;

/// Features per block: 10 planes of pawn to queen (both sides) and one plane for
/// both kings, 64 squares each.
pub const PSQ_PER_BLOCK: usize = 11 * 64;

/// Total piece-square feature transformer input size.
pub const PSQ_SIZE: usize = PSQ_BLOCKS * PSQ_PER_BLOCK;

/// 3-file pawn-pair features: C(96, 2) = 4,560.
/// 48 pawn squares × 2 colours, unordered pairs, i8 weights.
pub const PAWN_PAIR_SIZE: usize = 96 * 95 / 2;

/// Pairwise threat features (kings never attack or are targets; pawn-pawn
/// relations live in `PAWN_PAIR_SIZE` instead).
/// P 4×84 + N 10×336 + B 8×560 + R 8×896 + Q 10×1456 = 29,904 → ×2 colours = 59,808.
pub const THREAT_FEATURE_COUNT: usize = 59_808;

/// Offset of threat features inside the i8 aux-weight array (pawn pairs first).
pub const THREAT_OFFSET: usize = PAWN_PAIR_SIZE;

/// Combined i8 feature-transformer input: pawn pairs then threats.
pub const THREAT_INPUT_SIZE: usize = PAWN_PAIR_SIZE + THREAT_FEATURE_COUNT;

/// Feature transformer output / L1 input size (per perspective).
pub const L1_SIZE: usize = 1024;

const _: () = assert!(PSQ_SIZE == 45_056);
const _: () = assert!(PAWN_PAIR_SIZE == 4_560);
const _: () = assert!(THREAT_FEATURE_COUNT == 59_808);
const _: () = assert!(THREAT_INPUT_SIZE == 64_368);
const _: () = assert!(L1_SIZE % 64 == 0, "packus permutation groups are 64 wide");

/// L1 outputs (the sparse layer's width) per layer stack.
pub const L1_OUT: usize = 32;

/// L1 activations: `[x², x]` of every L1 output, u8.
pub const A1_SIZE: usize = 2 * L1_OUT;

/// L2 outputs per stack: the gates, then the values they multiply.
pub const L2_OUT: usize = 64;

/// Gated L2 outputs: `h = g² · v`.
pub const GATED: usize = L2_OUT / 2;

/// L3 inputs: the L1 activations, then the gated L2 outputs.
pub const L3_IN: usize = A1_SIZE + GATED;

/// Number of layer stacks: piece count buckets × queens on the board.
pub const STACKS: usize = 32;

/// The output of L1 that the skip connection reads positively, then negatively.
pub const SKIP_POS: usize = L1_OUT - 2;
pub const SKIP_NEG: usize = L1_OUT - 1;

/// Feature transformer activation clamp (the FT's unit is 256: one is just above).
pub const FT_QUANT: i16 = 255;

/// Right-shift for pairwise product: `(a * b) >> FT_SHIFT`, which lands on the hidden
/// unit 128 exactly (256 × 256 / 512).
pub const FT_SHIFT: i32 = 9;

/// Hidden activations are u8 with one = 128, clipped to 127.
pub const HIDDEN_MAX: i32 = 127;

/// L1 output: weights ×128 on inputs of unit 128, so one = 2^14. Shifts to unit 128.
pub const L1_SHIFT: i32 = 7;
/// `x² >> L1_SQR_SHIFT` is the square in unit 128: 2 × 14 − 7.
pub const L1_SQR_SHIFT: i32 = 21;
/// `|x|` at and above which the square saturates (16384² >> 21 = 128 > 127): clamping
/// there first keeps the square within i32 and changes no output.
pub const L1_SQR_CLAMP: i32 = 1 << 14;
/// L2 output: weights ×64 on inputs of unit 128, so one = 2^13. Shifts to unit 128.
pub const L2_SHIFT: i32 = 6;
/// `g · g · v >> GATE_SHIFT` is the product in unit 128: 3 × 7 − 7.
pub const GATE_SHIFT: i32 = 14;
/// The network output and the skip: weights ×128 on inputs of unit 128.
pub const OUTPUT_ONE: i64 = 1 << 14;

/// Default output scale (centipawns per network unit), written in every T7 file's
/// metadata and read from there; the trainer's `eval_scale`.
pub const NETWORK_SCALE: i32 = 287;

// ============================================================
// Layer stack selection
// ============================================================

/// The layer stack of a position: its piece count (kings and pawns included) in eight
/// bands, then whether the side to move and its opponent each have a queen.
/// `((pieces − 1) / 4) × 4 + 2 · [queen of the side to move] + [queen of the other]`.
pub const STACK_SCHEME: [usize; 3] = [4, 2, 1];

// ============================================================
// Network struct
// ============================================================

/// Per-thread NNUE state: the accumulator stack and the caches that refresh it.
///
/// Updated incrementally when possible (the moves since the nearest accurate ancestor,
/// one pass each), refreshed from the caches when a king moves or the opponent's queen
/// appears or disappears: the piece-square block of that perspective changes.
pub struct Network {
    /// Current ply index in the accumulator stack.
    index: usize,
    /// Accumulator stack: one per ply (pre-allocated).
    accumulators: Box<[Accumulator]>,
    /// Finny table: the bias plus the piece-square part per (perspective, king square,
    /// opponent queen).
    cache: FinnyTable,
    /// Pawn-pair sums per (perspective, mirrored), the start of every full threat recompute.
    pawn_cache: PawnPairCache,
    /// The changed threat features of the update in progress, kept off the stack.
    threat_dirty: Box<threats::DirtyPair>,
    /// The copy of the network this state evaluates with: the replica of its thread's
    /// node when the search is pinned, `None` for the shared copy. Set once per search
    /// by [`Network::use_params`], so no node ever pays for finding it — a thread-local
    /// read is a function call on some targets, and the kernels need the weights
    /// several times per node.
    params: Option<&'static network::NNUEParams>,
}

/// How one perspective of the current ply gets up to date.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Plan {
    /// Already accurate.
    Done,
    /// The moves since this accurate ancestor, applied one pass each.
    Chain(usize),
    /// The move into this ply changed the side's piece-square block (its king moved
    /// within the same half, or the opponent's queen appeared or disappeared), from an
    /// accurate parent: the parent with its piece-square part swapped for the new
    /// block's, plus the move's threats.
    BlockSwap,
    /// From the caches: the piece-square part, the pawn pairs, every threat.
    Full,
}

/// Whether the move recorded in `delta` changes the piece-square block of
/// `perspective`: its king lands on another square or half, or the presence of the
/// opponent's queen flips. Nothing of the piece-square part carries over.
#[inline]
fn block_changes(delta: &AccDelta, perspective: Color) -> bool {
    delta.queen_flip[perspective.index()]
        || (delta.moved_piece.piece_type() == PieceType::King
            && delta.moved_piece.color() == perspective
            // Castling encodes `to` as the rook origin: ask where the king lands.
            && features::needs_refresh(delta.mv.from_sq(), delta.mv.lands_on(), perspective))
}

impl Network {
    /// Create a new Network with pre-allocated accumulator stack.
    pub fn new() -> Self {
        let accumulators = (0..MAX_PLY + 1)
            .map(|_| Accumulator::new())
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Network {
            index: 0,
            accumulators,
            cache: FinnyTable::new(),
            pawn_cache: PawnPairCache::new(),
            threat_dirty: Box::new(threats::DirtyPair::new()),
            params: None,
        }
    }

    /// Evaluate with `params` (a node replica), or with the shared copy when `None`.
    /// Called before every search, so a network loaded since is never missed.
    pub fn use_params(&mut self, params: Option<&'static network::NNUEParams>) {
        self.params = params;
    }

    /// The weights this state evaluates with.
    #[inline]
    fn params(&self) -> &'static network::NNUEParams {
        match self.params {
            Some(p) => p,
            None => network::params(),
        }
    }

    /// Initialize the root accumulator from the current position.
    /// Must be called before the first search.
    pub fn refresh(&mut self, pos: &Position) {
        self.index = 0;
        for pov in 0..2 {
            self.refresh_full(pos, pov);
        }
        #[cfg(feature = "stats")]
        threats::tally::add(threats::tally::FULL_ROOT, 1);
        #[cfg(debug_assertions)]
        self.debug_check(pos);
    }

    /// Push a new ply onto the accumulator stack, and record the attacks the move
    /// creates and destroys.
    ///
    /// Must be called BEFORE `Position::make_move()` so we can read the
    /// moved and captured pieces from the unmodified board.
    pub fn push(&mut self, mv: Move, pos: &Position) {
        debug_assert!(mv != Move::NONE, "NNUE push: MOVE_NONE");
        debug_assert!(
            mv.from_sq().0 < 64 && mv.to_sq().0 < 64,
            "NNUE push: squares OOB from={} to={}",
            mv.from_sq().0,
            mv.to_sq().0
        );
        debug_assert!(
            pos.piece_on(mv.from_sq()) != Piece::NONE,
            "NNUE push: no piece on from {}",
            mv.from_sq().0
        );
        self.index += 1;
        debug_assert!(self.index < self.accumulators.len());

        let from = mv.from_sq();
        let to = mv.to_sq();
        let moved_piece = pos.piece_on(from);
        let captured_piece = if mv.move_type() == MT_EN_PASSANT {
            Piece::new(PieceType::Pawn, !pos.side_to_move)
        } else if mv.move_type() == MT_CASTLING {
            Piece::NONE
        } else {
            pos.piece_on(to)
        };

        let acc = &mut self.accumulators[self.index];
        acc.accurate = [false; 2];
        acc.delta = AccDelta {
            mv,
            moved_piece,
            captured_piece,
            old_occ: pos.occupied(),
            queen_flip: queen_flips(pos, mv, moved_piece, captured_piece),
        };
        move_threats::record(pos, &acc.delta, &mut acc.moved);
    }

    /// Push a null move onto the accumulator stack (lazy — no clone).
    ///
    /// CORRECTNESS: `accurate` MUST be set to `[false; 2]` here. See the
    /// stale accumulator bug documentation in MEMORY.md.
    pub fn push_null(&mut self) {
        self.index += 1;
        debug_assert!(self.index < self.accumulators.len());
        self.accumulators[self.index].accurate = [false; 2];
        self.accumulators[self.index].delta = AccDelta::default();
    }

    /// Pop one ply from the accumulator stack (after unmake_move).
    #[inline]
    pub fn pop(&mut self) {
        debug_assert!(self.index > 0);
        self.index -= 1;
    }

    pub fn evaluate(&mut self, pos: &Position) -> i32 {
        debug_assert!(pos.checkers == 0, "NNUE evaluate: called while in check");

        self.ensure_all_updated(pos);
        self.forward(pos)
    }

    /// How perspective `pov` of the current ply gets up to date: walk back to the
    /// nearest accurate ancestor, unless a move on the way changes the side's
    /// piece-square block, or the chain grows past `THREAT_CHAIN_MAX` plies.
    fn plan(&self, pov: usize) -> Plan {
        let idx = self.index;
        let accs = &self.accumulators;
        if accs[idx].accurate[pov] {
            return Plan::Done;
        }
        if idx == 0 {
            return Plan::Full;
        }
        let perspective = if pov == 0 { Color::White } else { Color::Black };
        let mut k = idx;
        loop {
            debug_assert!(k > 0);
            let delta = &accs[k].delta;
            if block_changes(delta, perspective) {
                let from = delta.mv.from_sq();
                let same_half = (from.file() >= 4) == (delta.mv.lands_on().file() >= 4)
                    || delta.moved_piece.piece_type() != PieceType::King
                    || delta.moved_piece.color() != perspective;
                return if k == idx
                    && accs[idx - 1].accurate[pov]
                    && same_half
                    && delta.mv.move_type() != MT_CASTLING
                {
                    Plan::BlockSwap
                } else {
                    Plan::Full
                };
            }
            k -= 1;
            if accs[k].accurate[pov] {
                return Plan::Chain(k);
            }
            if k == 0 || idx - k >= threats::THREAT_CHAIN_MAX {
                return Plan::Full;
            }
        }
    }

    /// Bring both perspectives of the current ply up to date without evaluating.
    ///
    /// Called at nodes that reuse a TT eval and never reach `evaluate`: their
    /// children can then take the incremental paths instead of a refresh each.
    pub fn ensure_all_updated(&mut self, pos: &Position) {
        let plans = [self.plan(0), self.plan(1)];
        if plans == [Plan::Done; 2] {
            return;
        }

        // The incremental chains, ply by ply from the lowest source: a ply's threat
        // record is decoded once for both perspectives.
        let first = plans
            .iter()
            .filter_map(|p| match p {
                Plan::Chain(s) => Some(*s),
                _ => None,
            })
            .min();
        if let Some(first) = first {
            let mirrored = [
                pos.king_sq(Color::White).file() >= 4,
                pos.king_sq(Color::Black).file() >= 4,
            ];
            for ply in (first + 1)..=self.index {
                let active = [0, 1].map(|pov| matches!(plans[pov], Plan::Chain(s) if s < ply));
                self.step(ply, active, mirrored, pos);
            }
        }

        for pov in 0..2 {
            match plans[pov] {
                Plan::BlockSwap => self.block_swap(pos, pov),
                Plan::Full => self.refresh_full(pos, pov),
                Plan::Done | Plan::Chain(_) => {}
            }
        }

        #[cfg(debug_assertions)]
        self.debug_check(pos);
    }

    /// Apply the move into `ply` to the perspectives in `active`, from the accurate
    /// accumulator of the ply before: its piece-square, threat and pawn-pair rows in
    /// one pass.
    fn step(&mut self, ply: usize, active: [bool; 2], mirrored: [bool; 2], pos: &Position) {
        let params = self.params();
        let (left, right) = self.accumulators.split_at_mut(ply);
        let prev = &left[ply - 1];
        let cur = &mut right[0];

        // Null move: no feature changed, the parent's sums carry over.
        if cur.delta.mv == Move::NONE {
            for pov in 0..2 {
                if active[pov] {
                    debug_assert!(prev.accurate[pov]);
                    cur.values.0[pov] = prev.values.0[pov];
                    cur.accurate[pov] = true;
                }
            }
            #[cfg(feature = "stats")]
            threats::tally::add(threats::tally::PST_NULL_COPIES, 1);
            return;
        }

        threats::collect_moved(&cur.moved, mirrored, active, &mut self.threat_dirty);
        for pov in 0..2 {
            if !active[pov] {
                continue;
            }
            debug_assert!(prev.accurate[pov], "NNUE step: stale parent at ply {ply}");
            let perspective = if pov == 0 { Color::White } else { Color::Black };
            let psq = cur.psq_changes(pos, perspective);
            let (adds, subs) = self.threat_dirty.pov(pov);
            unsafe {
                kernels::dispatch::combined_batch(
                    prev.values.0[pov].as_ptr(),
                    cur.values.0[pov].as_mut_ptr(),
                    params.ft_psq_weights.0.as_ptr(),
                    psq.adds(),
                    psq.subs(),
                    params.ft_threat_weights.0.as_ptr(),
                    adds,
                    subs,
                );
            }
            cur.accurate[pov] = true;
            #[cfg(feature = "stats")]
            threats::tally::add(threats::tally::APPLIED, 1);
        }
    }

    /// The move into the current ply changed the piece-square block of `pov` (its king
    /// moved within the same half, or the opponent's queen appeared or disappeared).
    /// The threats and pawn pairs carry over from the parent: its sums, minus the
    /// piece-square part of the old block, plus that of the new one (both brought by
    /// the Finny table to the position they stand for), plus the move's threat rows.
    fn block_swap(&mut self, pos: &Position, pov: usize) {
        let params = self.params();
        let idx = self.index;
        let perspective = if pov == 0 { Color::White } else { Color::Black };
        let delta = self.accumulators[idx].delta;
        debug_assert!(delta.mv.move_type() != MT_CASTLING);
        debug_assert!(delta.mv.move_type() != MT_EN_PASSANT, "an en passant capture changes no block");

        // The parent's board: the moved piece back on its square (a pawn, if it was a
        // promotion), the captured piece too.
        let (from, to) = (delta.mv.from_sq(), delta.mv.to_sq());
        let mut pieces = pos.pieces;
        let mut board = pos.board;
        let now = board[to.index()];
        debug_assert!(now != Piece::NONE);
        pieces[now.index()] ^= 1u64 << to.0;
        pieces[delta.moved_piece.index()] |= 1u64 << from.0;
        board[from.index()] = delta.moved_piece;
        board[to.index()] = delta.captured_piece;
        if delta.captured_piece != Piece::NONE {
            pieces[delta.captured_piece.index()] |= 1u64 << to.0;
        }
        let old_king = if delta.moved_piece == Piece::new(PieceType::King, perspective) {
            from
        } else {
            pos.king_sq(perspective)
        };

        let new_key =
            self.cache.bring_to(params, &pos.pieces, &pos.board, pos.king_sq(perspective), perspective);
        let old_key = self.cache.bring_to(params, &pieces, &board, old_king, perspective);
        debug_assert!(new_key != old_key, "NNUE block swap: the block did not change");

        let mirrored = [
            pos.king_sq(Color::White).file() >= 4,
            pos.king_sq(Color::Black).file() >= 4,
        ];
        let mut active = [false; 2];
        active[pov] = true;
        let (left, right) = self.accumulators.split_at_mut(idx);
        let prev = &left[idx - 1];
        let cur = &mut right[0];
        threats::collect_moved(&cur.moved, mirrored, active, &mut self.threat_dirty);

        let new = self.cache.entry(new_key);
        let old = self.cache.entry(old_key);
        let out = &mut cur.values.0[pov];
        for i in 0..L1_SIZE {
            out[i] = prev.values.0[pov][i]
                .wrapping_sub(old.values.0[i])
                .wrapping_add(new.values.0[i]);
        }
        let (adds, subs) = self.threat_dirty.pov(pov);
        let p = out.as_mut_ptr();
        unsafe {
            kernels::dispatch::threat_batch(p, p, params.ft_threat_weights.0.as_ptr(), adds, subs)
        };
        cur.accurate[pov] = true;
        #[cfg(feature = "stats")]
        threats::tally::add(threats::tally::FULL_KING, 1);
    }

    /// Recompute perspective `pov` of the current ply from the caches: the Finny
    /// entry for its king and the opponent queen, the pawn pairs of its half, every
    /// active threat.
    fn refresh_full(&mut self, pos: &Position, pov: usize) {
        let params = self.params();
        let perspective = if pov == 0 { Color::White } else { Color::Black };
        let key = self.cache.bring_to(
            params,
            &pos.pieces,
            &pos.board,
            pos.king_sq(perspective),
            perspective,
        );
        let entry = self.cache.entry(key);
        let cur = &mut self.accumulators[self.index];
        cur.values.0[pov] = entry.values.0;
        threats::add_full(params, pos, pov, &mut self.pawn_cache, &mut cur.values.0[pov]);
        cur.accurate[pov] = true;
        #[cfg(feature = "stats")]
        threats::tally::add(threats::tally::FULL_CHAIN, 1);
    }

    /// Debug builds: the current accumulator is the sum from scratch.
    #[cfg(debug_assertions)]
    fn debug_check(&self, pos: &Position) {
        let values = reference_sums(self.params(), pos);
        let acc = &self.accumulators[self.index];
        for pov in 0..2 {
            debug_assert!(
                acc.values.0[pov][..] == values.0[pov][..],
                "NNUE accumulator differs from the sum from scratch: pov={pov}"
            );
        }
    }

    /// Test helper: the current accumulator, both perspectives up to date, is the
    /// sum from scratch.
    #[cfg(test)]
    pub(crate) fn assert_matches_reference(&self, pos: &Position, label: &str) {
        let values = reference_sums(self.params(), pos);
        let acc = &self.accumulators[self.index];
        assert_eq!(acc.accurate, [true; 2], "{label}: not up to date");
        for pov in 0..2 {
            for i in 0..L1_SIZE {
                assert_eq!(
                    acc.values.0[pov][i], values.0[pov][i],
                    "{label}: pov={pov} mismatch at {i}"
                );
            }
        }
    }

    /// Forward pass through the integer layers (SIMD-accelerated), scaled to
    /// centipawns by the network's own output scale.
    fn forward(&self, pos: &Position) -> i32 {
        let stm = pos.side_to_move;
        let acc = &self.accumulators[self.index];
        let params = self.params();
        let out = unsafe { kernels::dispatch::forward_dense(params, &acc.values, stm, stack(pos)) };
        to_centipawns(out, params.output_scale())
    }
}

/// The network's output in centipawns: `out × scale / 2^14`, truncated toward zero.
#[inline]
pub fn to_centipawns(out: i32, scale: i32) -> i32 {
    debug_assert!(scale > 0);
    ((out as i64 * scale as i64) / OUTPUT_ONE) as i32
}

/// For each perspective, whether the move flips the presence of the queen of the
/// perspective's opponent: it captures that side's last queen, or it promotes the
/// first queen of a side that had none. Read on the board before the move.
fn queen_flips(pos: &Position, mv: Move, moved: Piece, captured: Piece) -> [bool; 2] {
    let mut flips = [false; 2];
    if captured != Piece::NONE && captured.piece_type() == PieceType::Queen {
        let side = captured.color();
        if pos.pieces[captured.index()].count_ones() == 1 {
            // `side` is the opponent of the other perspective.
            flips[(!side).index()] = true;
        }
    }
    if mv.move_type() == MT_PROMOTION && mv.promo_type() == PieceType::Queen {
        let side = moved.color();
        if pos.pieces[Piece::new(PieceType::Queen, side).index()] == 0 {
            flips[(!side).index()] = true;
        }
    }
    flips
}

/// The accumulator of `pos` from scratch, element-wise and without the caches: the
/// bias and piece-square rows of every piece, plus every pawn pair and threat. One
/// side of every incremental-vs-refresh comparison.
#[cfg(any(test, debug_assertions))]
fn reference_sums(params: &network::NNUEParams, pos: &Position) -> network::Aligned<[[i16; L1_SIZE]; 2]> {
    let mut values = threats::compute_full_threats_with(params, pos);
    let mut psq = Accumulator::new();
    for perspective in [Color::White, Color::Black] {
        psq.refresh_with_params(params, pos, perspective);
    }
    for pov in 0..2 {
        for i in 0..L1_SIZE {
            values.0[pov][i] = values.0[pov][i].wrapping_add(psq.values.0[pov][i]);
        }
    }
    values
}

impl Clone for Network {
    fn clone(&self) -> Self {
        Network {
            index: self.index,
            accumulators: self.accumulators.clone(),
            cache: self.cache.clone(),
            pawn_cache: self.pawn_cache.clone(),
            // Scratch: empty at the start of every update, nothing to carry over.
            threat_dirty: Box::new(threats::DirtyPair::new()),
            params: self.params,
        }
    }
}

// ============================================================
// Layer stack selection
// ============================================================

/// Select the layer stack (see `STACK_SCHEME`).
pub fn stack(pos: &Position) -> usize {
    let [band_width, own_queen, their_queen] = STACK_SCHEME;
    let stm = pos.side_to_move;
    let pieces = pos.occupied().count_ones() as usize;
    debug_assert!((2..=32).contains(&pieces), "stack: {pieces} pieces");
    let band = (pieces - 1) / band_width;
    let ours = pos.pieces[Piece::new(PieceType::Queen, stm).index()] != 0;
    let theirs = pos.pieces[Piece::new(PieceType::Queen, !stm).index()] != 0;
    let s = band * 4 + own_queen * ours as usize + their_queen * theirs as usize;
    debug_assert!(s < STACKS, "stack {s} >= STACKS {STACKS}");
    s
}
// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Position;

    #[test]
    fn test_network_new() {
        let net = Network::new();
        assert_eq!(net.index, 0);
        assert!(!net.accumulators.is_empty());
    }

    /// Pins the layer-stack scheme on known positions; the trainer pins the SAME
    /// expectations on its side (tools/trainer8/src/outputs.rs).
    #[test]
    fn the_stack_follows_piece_count_and_queens() {
        let cases = [
            // startpos: 32 pieces → band 7, both queens → 7*4 + 3
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 31),
            // bare kings: band 0, no queen
            ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", 0),
            // white queen, black to move: the queen is the opponent's
            ("4k3/8/8/8/8/8/8/3QK3 b - - 0 1", 1),
            ("4k3/8/8/8/8/8/8/3QK3 w - - 0 1", 2),
            // KQvKQ: 4 pieces → band 0, both queens
            ("3qk3/8/8/8/8/8/8/3QK3 w - - 0 1", 3),
            // 18 pieces → band 4
            ("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/4K3 w - - 0 1", 16),
        ];
        for (fen, want) in cases {
            let pos = Position::from_fen(fen).unwrap();
            assert_eq!(stack(&pos), want, "{fen}");
        }
    }

    /// The presence of the opponent's queen is part of the piece-square block: a move
    /// that takes the last queen, or promotes the first one, flips it for the other
    /// side, and only then.
    #[test]
    fn queen_flips_are_recorded_for_the_other_side_only() {
        // White takes Black's only queen: Black's queen disappears for White.
        let pos = Position::from_fen("3qk3/8/8/8/8/8/8/3QK3 w - - 0 1").unwrap();
        let mv = Move::new(Square::D1, Square::D8);
        let flips = queen_flips(&pos, mv, pos.piece_on(Square::D1), pos.piece_on(Square::D8));
        assert_eq!(flips, [true, false]);

        // Black has two queens: taking one flips nothing.
        let pos = Position::from_fen("3qk2q/8/8/8/8/8/8/3QK3 w - - 0 1").unwrap();
        let flips = queen_flips(&pos, mv, pos.piece_on(Square::D1), pos.piece_on(Square::D8));
        assert_eq!(flips, [false, false]);

        // White promotes its first queen: White's queen appears for Black.
        let pos = Position::from_fen("7k/4P3/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let mv = Move::new_promotion(Square::E7, Square::E8, PieceType::Queen);
        let flips = queen_flips(&pos, mv, pos.piece_on(Square::E7), Piece::NONE);
        assert_eq!(flips, [false, true]);

        // An underpromotion flips nothing.
        let mv = Move::new_promotion(Square::E7, Square::E8, PieceType::Rook);
        let flips = queen_flips(&pos, mv, pos.piece_on(Square::E7), Piece::NONE);
        assert_eq!(flips, [false, false]);
    }

    #[test]
    fn test_evaluate_startpos() {
        let pos =
            Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let mut net = Network::new();
        net.refresh(&pos);

        let score = net.evaluate(&pos);
        // Startpos is symmetric → score should be near 0 (small due to eval noise)
        assert!(
            score.abs() < 500,
            "startpos eval should be near 0, got {score}"
        );
    }

    #[test]
    fn test_push_pop() {
        let mut pos =
            Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let mut net = Network::new();
        net.refresh(&pos);

        // Push e2e4
        let mv = Move::new(Square::E2, Square::E4);
        net.push(mv, &pos);
        pos.make_move(mv);
        assert_eq!(net.index, 1);

        // Evaluate (triggers lazy update)
        let _score = net.evaluate(&pos);
        assert!(net.accumulators[1].accurate[0]);
        assert!(net.accumulators[1].accurate[1]);

        // Pop
        pos.unmake_move(mv);
        net.pop();
        assert_eq!(net.index, 0);
    }

    #[test]
    fn test_refresh_vs_incremental() {
        // The key correctness test: for any move, refresh and incremental
        // should produce identical accumulator values.
        let mut pos =
            Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let mut net = Network::new();
        net.refresh(&pos);

        // Play e2e4
        let mv = Move::new(Square::E2, Square::E4);
        net.push(mv, &pos);
        pos.make_move(mv);

        // Evaluate with incremental update
        let _score = net.evaluate(&pos);
        net.assert_matches_reference(&pos, "test_refresh_vs_incremental");
    }

    /// Whether an ancestor accumulator survives a king move is decided by where the
    /// king lands. Castling records the rook's origin as its `to`, so reading that
    /// square asks about a square the king never occupies: here it would compare
    /// g1 against h1 and refresh, though the king does not move at all.
    #[test]
    fn a_king_that_stays_put_castling_keeps_its_ancestor() {
        let delta = AccDelta {
            // FRC O-O with the king already on g1: only the rook travels, h1 → f1.
            mv: Move::new_with_type(Square::G1, Square::H1, MT_CASTLING),
            moved_piece: Piece::WHITE_KING,
            captured_piece: Piece::NONE,
            old_occ: 0,
            queen_flip: [false; 2],
        };
        assert!(!block_changes(&delta, Color::White));

        // And a king that does cross the mirror still forces the refresh.
        let crossing = AccDelta {
            mv: Move::new(Square::E1, Square::D1),
            ..delta
        };
        assert!(block_changes(&crossing, Color::White));

        // A queen flip changes the block of the side that sees it, and only that one.
        let flip = AccDelta {
            mv: Move::new(Square::D1, Square::D8),
            moved_piece: Piece::WHITE_QUEEN,
            captured_piece: Piece::BLACK_QUEEN,
            queen_flip: [true, false],
            ..delta
        };
        assert!(block_changes(&flip, Color::White));
        assert!(!block_changes(&flip, Color::Black));
    }

    #[test]
    fn test_ft_activation_symmetry() {
        // With zeroed weights, activation should produce all zeros
        let acc = network::Aligned([[0i16; L1_SIZE]; 2]);
        let out = forward::activate_ft(&acc, Color::White);
        for i in 0..L1_SIZE {
            assert_eq!(out.0[i], 0);
        }
    }

    // ============================================================
    // Comprehensive refresh-vs-incremental tests (all move types)
    // ============================================================

    /// Helper: play a move, verify incremental == refresh for both perspectives.
    fn assert_incremental_matches_refresh(fen: &str, mv: Move, label: &str) {
        let mut pos = Position::from_fen(fen).unwrap();
        let mut net = Network::new();
        net.refresh(&pos);

        // Push + make_move
        net.push(mv, &pos);
        pos.make_move(mv);

        // Trigger lazy incremental update (directly, not via evaluate which asserts !in_check)
        net.ensure_all_updated(&pos);

        net.assert_matches_reference(&pos, label);
    }

    #[test]
    fn test_incremental_capture() {
        // Nxe5: White knight captures black pawn
        assert_incremental_matches_refresh(
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3",
            Move::new(Square::F3, Square::E5),
            "Nxe5 capture",
        );
    }

    #[test]
    fn test_incremental_en_passant() {
        // exd6 e.p.: White pawn on e5 captures black pawn on d5
        assert_incremental_matches_refresh(
            "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
            Move::new_with_type(Square::E5, Square::D6, MT_EN_PASSANT),
            "exd6 en passant",
        );
    }

    #[test]
    fn test_incremental_promotion() {
        // e8=Q: White pawn on e7 promotes to queen (no capture)
        assert_incremental_matches_refresh(
            "7k/4P3/8/8/8/8/8/4K3 w - - 0 1",
            Move::new_promotion(Square::E7, Square::E8, PieceType::Queen),
            "e8=Q promotion",
        );
    }

    #[test]
    fn test_incremental_promotion_capture() {
        // exd8=Q: White pawn captures rook on d8 and promotes
        assert_incremental_matches_refresh(
            "3r3k/4P3/8/8/8/8/8/4K3 w - - 0 1",
            Move::new_promotion(Square::E7, Square::D8, PieceType::Queen),
            "exd8=Q promotion-capture",
        );
    }

    #[test]
    fn test_incremental_underpromotion() {
        // e8=N: underpromotion to knight
        assert_incremental_matches_refresh(
            "7k/4P3/8/8/8/8/8/4K3 w - - 0 1",
            Move::new_promotion(Square::E7, Square::E8, PieceType::Knight),
            "e8=N underpromotion",
        );
    }

    #[test]
    fn test_incremental_castling_kingside() {
        // O-O: White kingside castling
        assert_incremental_matches_refresh(
            "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
            Move::new_with_type(Square::E1, Square::H1, MT_CASTLING),
            "O-O kingside",
        );
    }

    #[test]
    fn test_incremental_castling_queenside() {
        // O-O-O: White queenside castling
        assert_incremental_matches_refresh(
            "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
            Move::new_with_type(Square::E1, Square::A1, MT_CASTLING),
            "O-O-O queenside",
        );
    }

    #[test]
    fn test_incremental_castling_black_kingside() {
        // Black O-O
        assert_incremental_matches_refresh(
            "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1",
            Move::new_with_type(Square::E8, Square::H8, MT_CASTLING),
            "Black O-O",
        );
    }

    #[test]
    fn test_incremental_castling_black_queenside() {
        // Black O-O-O
        assert_incremental_matches_refresh(
            "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1",
            Move::new_with_type(Square::E8, Square::A8, MT_CASTLING),
            "Black O-O-O",
        );
    }

    #[test]
    fn test_incremental_castling_frc_swap() {
        assert_incremental_matches_refresh(
            "4k3/8/8/8/8/8/8/5KR1 w K - 0 1",
            Move::new_with_type(Square::F1, Square::G1, MT_CASTLING),
            "FRC king/rook swap O-O",
        );
    }

    #[test]
    fn test_incremental_castling_king_already_on_g() {
        assert_incremental_matches_refresh(
            "4k3/8/8/8/8/8/8/6KR w K - 0 1",
            Move::new_with_type(Square::G1, Square::H1, MT_CASTLING),
            "FRC king already on g1",
        );
    }

    #[test]
    fn test_incremental_null_move() {
        // Lazy push_null: index advances but values are NOT cloned.
        // ensure_all_updated copies from parent on demand.
        let mut pos =
            Position::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
                .unwrap();
        let mut net = Network::new();
        net.refresh(&pos);

        // Lazy push_null: index advances, marked inaccurate
        net.push_null();
        pos.make_null_move();
        assert_eq!(net.index, 1);
        assert!(!net.accumulators[1].accurate[0]);
        assert!(!net.accumulators[1].accurate[1]);

        // Evaluate triggers lazy copy from parent
        let _score = net.evaluate(&pos);

        net.assert_matches_reference(&pos, "test_incremental_null_move");

        pos.unmake_null_move();
        net.pop();
        assert_eq!(net.index, 0);
    }

    #[test]
    fn test_null_move_child_push() {
        // A real move after a null move pushes at parent_index+2 and updates correctly.
        let mut pos =
            Position::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
                .unwrap();
        let mut net = Network::new();
        net.refresh(&pos);

        // Lazy push_null
        net.push_null();
        pos.make_null_move();
        assert_eq!(net.index, 1);

        // Real move from null-move position: d2d4 (white to move after null)
        let mv = Move::new(Square::D2, Square::D4);
        net.push(mv, &pos);
        pos.make_move(mv);
        assert_eq!(net.index, 2);

        // Trigger lazy update (copies parent at 1, then applies delta at 2)
        net.ensure_all_updated(&pos);

        net.assert_matches_reference(&pos, "test_null_move_child_push");

        pos.unmake_move(mv);
        net.pop();
        pos.unmake_null_move();
        net.pop();
        assert_eq!(net.index, 0);
    }

    #[test]
    fn test_incremental_multi_move_sequence() {
        // Play a sequence of moves and check after each one
        let mut pos =
            Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let mut net = Network::new();
        net.refresh(&pos);

        let moves = [
            Move::new(Square::E2, Square::E4), // 1. e4
            Move::new(Square::E7, Square::E5), // 1... e5
            Move::new(Square::G1, Square::F3), // 2. Nf3
            Move::new(Square::B8, Square::C6), // 2... Nc6
            Move::new(Square::F1, Square::B5), // 3. Bb5
        ];

        for (i, &mv) in moves.iter().enumerate() {
            net.push(mv, &pos);
            pos.make_move(mv);

            let _score = net.evaluate(&pos);

            net.assert_matches_reference(&pos, &format!("move {}", i + 1));
        }
    }

    #[test]
    fn test_incremental_king_move_triggers_refresh() {
        // King moves to different bucket → needs full refresh
        // King on e1 (file 4, bucket from KING_BUCKETS) → d2 (different bucket)
        let mut pos = Position::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let mut net = Network::new();
        net.refresh(&pos);

        // Ke1-d2: king crosses midline (file 4→3), triggers refresh
        let mv = Move::new(Square::E1, Square::D2);
        net.push(mv, &pos);
        pos.make_move(mv);

        let _score = net.evaluate(&pos);

        net.assert_matches_reference(&pos, "test_incremental_king_move_triggers_refresh");
    }
}
