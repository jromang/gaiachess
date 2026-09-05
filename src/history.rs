//! Move ordering data: [butterfly history](https://www.chessprogramming.org/History_Heuristic),
//! [capture history](https://www.chessprogramming.org/History_Heuristic#Capture_History),
//! [continuation history](https://www.chessprogramming.org/Continuation_History),
//! [countermove heuristic](https://www.chessprogramming.org/Countermove_Heuristic),
//! [killer moves](https://www.chessprogramming.org/Killer_Heuristic),
//! and [correction history](https://www.chessprogramming.org/Static_Evaluation_Correction_History).

use std::sync::atomic::{AtomicI16, Ordering};

use crate::types::{Color, Move, Piece, PieceType, Square, MAX_PLY};

/// Maximum history value for gravity formula.
const MAX_HISTORY: i32 = 16384;

/// Maximum correction history value for gravity formula.
pub const CORRHIST_LIMIT: i32 = 1024;

/// Entries per thread in the key-indexed correction history tables.
///
/// The scale of the reference engines, which is four times what this was. Sharing the
/// tables between threads (R03) quadrupled the distinct slots as a side effect and won
/// far more than sharing alone was expected to; this raises the base so the two effects
/// can be told apart, and so a single thread gets whatever the size is worth.
const CORRHIST_SIZE: usize = 65536;

/// Threat-indexed butterfly history: `[color][from_threatened][to_threatened][from_sq * 64 + to_sq]`.
///
/// Conditions the butterfly history on whether the move's from/to squares are
/// attacked by the opponent. This captures escape (from threatened) and walking
/// into danger (to threatened) patterns.
///
/// Gravity-based updates keep values in `[-MAX_HISTORY, MAX_HISTORY]`.
/// Size: 2 * 2 * 2 * 4096 * 2 bytes = 64 KB.
pub struct ButterflyHistory {
    table: [[[[i16; 64 * 64]; 2]; 2]; 2],
}

impl ButterflyHistory {
    pub fn new() -> Self {
        ButterflyHistory {
            table: [[[[0i16; 64 * 64]; 2]; 2]; 2],
        }
    }

    pub fn clear(&mut self) {
        for side in &mut self.table {
            for ft in side {
                for tt in ft {
                    tt.fill(0);
                }
            }
        }
    }

    #[inline(always)]
    fn index(m: Move) -> usize {
        m.from_sq().index() * 64 + m.to_sq().index()
    }

    /// Get the history score for a move, conditioned on threat status.
    #[inline]
    pub fn get(&self, color: Color, m: Move, threats: u64) -> i32 {
        debug_assert!(m != Move::NONE && m != Move::NULL,
            "ButterflyHistory::get: invalid move {:?}", m);
        debug_assert!(m.from_sq().0 < 64 && m.to_sq().0 < 64,
            "ButterflyHistory::get: squares OOB");
        let from_thr = ((threats >> m.from_sq().index()) & 1) as usize;
        let to_thr = ((threats >> m.to_sq().index()) & 1) as usize;
        self.table[color.index()][from_thr][to_thr][Self::index(m)] as i32
    }

    /// Gravity update: `val += bonus - val * |bonus| / MAX_HISTORY`.
    #[inline]
    pub fn update(&mut self, color: Color, m: Move, threats: u64, bonus: i32) {
        debug_assert!(m != Move::NONE && m != Move::NULL,
            "ButterflyHistory::update: invalid move {:?}", m);
        debug_assert!(m.from_sq().0 < 64 && m.to_sq().0 < 64,
            "ButterflyHistory::update: squares OOB");
        let from_thr = ((threats >> m.from_sq().index()) & 1) as usize;
        let to_thr = ((threats >> m.to_sq().index()) & 1) as usize;
        let entry = &mut self.table[color.index()][from_thr][to_thr][Self::index(m)];
        let val = *entry as i32;
        let new_val = val + bonus - val * bonus.abs() / MAX_HISTORY;
        *entry = new_val.clamp(-MAX_HISTORY, MAX_HISTORY) as i16;
    }
}

/// Number of pawn hash entries for pawn history table.
const PAWN_HIST_SIZE: usize = 1024;

/// Pawn history: `[pawn_key % SIZE][piece][to_sq]`.
///
/// Scores quiet moves based on the current pawn structure. The same piece-to
/// move can receive different scores depending on which pawns are on the board.
/// Size: 1024 * 12 * 64 * 2 bytes = 1.5 MB (heap-allocated via inner Box).
///
/// Size: 1024 * 12 * 64 * 2 bytes = 1.5 MB (heap-allocated via inner Box).
pub struct PawnHistory {
    table: Box<[[[i16; Square::NUM]; Piece::NUM]; PAWN_HIST_SIZE]>,
}

impl PawnHistory {
    pub fn new() -> Self {
        unsafe {
            let layout = std::alloc::Layout::new::<[[[i16; Square::NUM]; Piece::NUM]; PAWN_HIST_SIZE]>();
            let ptr = std::alloc::alloc_zeroed(layout);
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            PawnHistory {
                table: Box::from_raw(ptr.cast()),
            }
        }
    }

    pub fn clear(&mut self) {
        unsafe {
            std::ptr::write_bytes(self.table.as_mut_ptr(), 0, PAWN_HIST_SIZE);
        }
    }

    #[inline]
    pub fn get(&self, pawn_key: u64, piece: Piece, to: Square) -> i32 {
        debug_assert!(piece.0 < 12, "PawnHistory::get: piece {} invalid", piece.0);
        debug_assert!(to.0 < 64, "PawnHistory::get: to sq {} OOB", to.0);
        self.table[pawn_key as usize % PAWN_HIST_SIZE][piece.index()][to.index()] as i32
    }

    #[inline]
    pub fn update(&mut self, pawn_key: u64, piece: Piece, to: Square, bonus: i32) {
        debug_assert!(piece.0 < 12, "PawnHistory::update: piece {} invalid", piece.0);
        debug_assert!(to.0 < 64, "PawnHistory::update: to sq {} OOB", to.0);
        let entry = &mut self.table[pawn_key as usize % PAWN_HIST_SIZE][piece.index()][to.index()];
        let val = *entry as i32;
        let new_val = val + bonus - val * bonus.abs() / MAX_HISTORY;
        *entry = new_val.clamp(-MAX_HISTORY, MAX_HISTORY) as i16;
    }
}

/// Killer move table: 2 killers per ply.
pub struct Killers {
    table: [[Move; 2]; MAX_PLY + 1],
}

impl Killers {
    pub fn new() -> Self {
        Killers {
            table: [[Move::NONE; 2]; MAX_PLY + 1],
        }
    }

    pub fn clear(&mut self) {
        for slot in &mut self.table {
            *slot = [Move::NONE; 2];
        }
    }

    /// Get the two killer moves at a given ply.
    #[inline]
    pub fn get(&self, ply: usize) -> [Move; 2] {
        debug_assert!(ply <= MAX_PLY, "Killers::get: ply {} > MAX_PLY", ply);
        self.table[ply]
    }

    /// Insert a new killer move with shift scheme.
    /// If not already slot 0, shift 0→1 and insert at 0.
    #[inline]
    pub fn update(&mut self, ply: usize, m: Move) {
        debug_assert!(ply <= MAX_PLY, "Killers::update: ply {} > MAX_PLY", ply);
        if self.table[ply][0] != m {
            self.table[ply][1] = self.table[ply][0];
            self.table[ply][0] = m;
        }
    }
}

/// History bonus: linear in depth, clamped.
#[inline]
pub fn stat_bonus(depth: i32) -> i32 {
    (crate::tune::STAT_BONUS_MUL() * depth + crate::tune::STAT_BONUS_ADD()).min(crate::tune::STAT_BONUS_MAX())
}

/// History malus: linear in depth, clamped.
#[inline]
pub fn stat_malus(depth: i32) -> i32 {
    (crate::tune::STAT_MALUS_MUL() * depth - crate::tune::STAT_MALUS_SUB()).min(crate::tune::STAT_MALUS_MAX())
}

/// Capture history table: `[piece][to_sq][captured_piece_type]`.
///
/// Learns which captures succeed (cause beta cutoffs) beyond static MVV-LVA.
/// Size: 12 * 64 * 6 * 2 bytes = 9,216 bytes (~9 KB).
pub struct CaptureHistory {
    table: [[[i16; PieceType::NUM]; Square::NUM]; Piece::NUM],
}

impl CaptureHistory {
    pub fn new() -> Self {
        CaptureHistory {
            table: [[[0i16; PieceType::NUM]; Square::NUM]; Piece::NUM],
        }
    }

    pub fn clear(&mut self) {
        for piece in &mut self.table {
            for sq in piece.iter_mut() {
                sq.fill(0);
            }
        }
    }

    #[inline]
    pub fn get(&self, piece: Piece, to: Square, captured_pt: PieceType) -> i32 {
        debug_assert!(piece.0 < 12, "CaptureHistory::get: piece NONE");
        debug_assert!(to.0 < 64, "CaptureHistory::get: sq OOB {}", to.0);
        debug_assert!((captured_pt as usize) < 6, "CaptureHistory::get: captured_pt OOB");
        self.table[piece.index()][to.index()][captured_pt.index()] as i32
    }

    /// Gravity update (same formula as ButterflyHistory).
    #[inline]
    pub fn update(&mut self, piece: Piece, to: Square, captured_pt: PieceType, bonus: i32) {
        debug_assert!(piece.0 < 12, "CaptureHistory::update: piece NONE");
        debug_assert!(to.0 < 64, "CaptureHistory::update: sq OOB {}", to.0);
        debug_assert!((captured_pt as usize) < 6, "CaptureHistory::update: captured_pt OOB");
        let entry = &mut self.table[piece.index()][to.index()][captured_pt.index()];
        let val = *entry as i32;
        let new_val = val + bonus - val * bonus.abs() / MAX_HISTORY;
        *entry = new_val.clamp(-MAX_HISTORY, MAX_HISTORY) as i16;
    }
}

/// Piece-to-square history subtable: `[piece][to_sq]` (i16).
///
/// One subtable exists per previous move's (is_capture, piece, to_sq) context.
/// Size per subtable: 12 * 64 * 2 bytes = 1,536 bytes (~1.5 KB).
pub type PieceToTable = [[i16; Square::NUM]; Piece::NUM];

/// Continuation history: `[is_capture][prev_piece][prev_to]` → `PieceToTable`.
///
/// Captures the relationship "if previous move was piece X to square Y,
/// then piece A to square B is likely good/bad."
/// Total: 2 * 12 * 64 * 1536 bytes ≈ 2.25 MB (heap-allocated).
pub struct ContinuationHistory {
    table: Box<[[[PieceToTable; Square::NUM]; Piece::NUM]; 2]>,
}

impl ContinuationHistory {
    pub fn new() -> Self {
        // Use alloc_zeroed for large heap allocation
        let table = unsafe {
            let layout = std::alloc::Layout::new::<[[[PieceToTable; Square::NUM]; Piece::NUM]; 2]>();
            let ptr = std::alloc::alloc_zeroed(layout);
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            Box::from_raw(ptr.cast())
        };
        ContinuationHistory { table }
    }

    pub fn clear(&mut self) {
        unsafe {
            let ptr = self.table.as_mut() as *mut _ as *mut u8;
            let size = std::mem::size_of::<[[[PieceToTable; Square::NUM]; Piece::NUM]; 2]>();
            std::ptr::write_bytes(ptr, 0, size);
        }
    }

    /// Get a mutable pointer to the subtable for a given previous move context.
    /// This pointer is stored on the search stack and used later for lookups/updates.
    #[inline]
    pub fn subtable_ptr(
        &mut self,
        is_capture: bool,
        piece: Piece,
        to: Square,
    ) -> *mut PieceToTable {
        debug_assert!(piece.0 < 12, "ContinuationHistory::subtable_ptr: piece NONE");
        debug_assert!(to.0 < 64, "ContinuationHistory::subtable_ptr: sq OOB {}", to.0);
        &raw mut self.table[is_capture as usize][piece.index()][to.index()]
    }

    /// Get the continuation history score from a subtable pointer.
    #[inline]
    pub fn get(subtable_ptr: *const PieceToTable, piece: Piece, to: Square) -> i32 {
        debug_assert!(!subtable_ptr.is_null(), "ContinuationHistory::get: null ptr");
        debug_assert!(piece.0 < 12, "ContinuationHistory::get: piece NONE");
        debug_assert!(to.0 < 64, "ContinuationHistory::get: sq OOB {}", to.0);
        unsafe { (*subtable_ptr)[piece.index()][to.index()] as i32 }
    }

    /// Gravity update via a subtable pointer.
    #[inline]
    pub fn update(subtable_ptr: *mut PieceToTable, piece: Piece, to: Square, bonus: i32) {
        debug_assert!(!subtable_ptr.is_null(), "ContinuationHistory::update: null ptr");
        debug_assert!(piece.0 < 12, "ContinuationHistory::update: piece NONE");
        debug_assert!(to.0 < 64, "ContinuationHistory::update: sq OOB {}", to.0);
        let entry = unsafe { &mut (*subtable_ptr)[piece.index()][to.index()] };
        let val = *entry as i32;
        let new_val = val + bonus - val * bonus.abs() / MAX_HISTORY;
        *entry = new_val.clamp(-MAX_HISTORY, MAX_HISTORY) as i16;
    }
}

/// Countermove table: `[piece][to_sq]` → Move.
///
/// For each (piece, to) of the previous move, stores the quiet move
/// that caused a beta cutoff as a natural response.
/// Size: 12 * 64 * 2 bytes = 1,536 bytes (~1.5 KB).
pub struct Countermoves {
    table: [[Move; Square::NUM]; Piece::NUM],
}

impl Countermoves {
    pub fn new() -> Self {
        Countermoves {
            table: [[Move::NONE; Square::NUM]; Piece::NUM],
        }
    }

    pub fn clear(&mut self) {
        for piece in &mut self.table {
            piece.fill(Move::NONE);
        }
    }

    #[inline]
    pub fn get(&self, piece: Piece, to: Square) -> Move {
        if piece == Piece::NONE {
            return Move::NONE;
        }
        debug_assert!(to.0 < 64, "Countermoves::get: to sq {} OOB", to.0);
        self.table[piece.index()][to.index()]
    }

    #[inline]
    pub fn update(&mut self, piece: Piece, to: Square, counter: Move) {
        debug_assert!(piece != Piece::NONE, "Countermoves::update: piece NONE");
        debug_assert!(to.0 < 64, "Countermoves::update: to sq {} OOB", to.0);
        self.table[piece.index()][to.index()] = counter;
    }
}

/// Which of the key-indexed correction tables an entry belongs to.
///
/// The four live interleaved in one allocation, so a slot is
/// `[stm][kind]`: one base pointer, one mask, one length. Each kind is looked up
/// with its own key and only ever touches its own field, so they stay four
/// independent tables that happen to share a shape.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(usize)]
pub enum CorrKind {
    Pawn = 0,
    Minor = 1,
    NonPawnWhite = 2,
    NonPawnBlack = 3,
}

impl CorrKind {
    /// The non-pawn table for a given piece colour.
    #[inline]
    pub fn non_pawn(color: Color) -> CorrKind {
        match color {
            Color::White => CorrKind::NonPawnWhite,
            _ => CorrKind::NonPawnBlack,
        }
    }

    #[inline]
    const fn index(self) -> usize {
        self as usize
    }
}

/// Number of correction kinds packed into one slot.
const CORR_KINDS: usize = 4;

/// Correction histories keyed by position structure, shared by every search thread.
///
/// A correction history is an estimator of value indexed by structure — pawn key,
/// non-pawn key per colour, minor-piece key — not a move-ordering signal. Nothing
/// about it wants to differ from one thread to the next, and with a copy per thread
/// N threads spend their search learning the same N corrections N times over.
///
/// Entries are relaxed atomics read-modify-written without a lock, the way the
/// transposition table is: two threads updating the same slot can lose an update,
/// which costs one observation out of millions. What they cannot do is write a value
/// out of range, because the old value is loaded into a register once and the result
/// is clamped before it goes back.
///
/// The table holds [`CORRHIST_SIZE`] `* next_power_of_two(threads)` slots so that the number of
/// entries per thread, and with it the collision rate, stays what it was at one
/// thread. Being a power of two, the key is masked rather than divided.
pub struct SharedCorrectionHistory {
    /// `[(key & mask) * 2 + stm] * CORR_KINDS + kind`
    table: Box<[AtomicI16]>,
    mask: usize,
}

impl SharedCorrectionHistory {
    /// Slot count for a pool of `threads` threads.
    fn slots_for(threads: usize) -> usize {
        debug_assert!(threads > 0, "SharedCorrectionHistory: zero threads");
        CORRHIST_SIZE * threads.max(1).next_power_of_two()
    }

    pub fn new(threads: usize) -> Self {
        let slots = Self::slots_for(threads);
        let table = (0..slots * 2 * CORR_KINDS).map(|_| AtomicI16::new(0)).collect();
        SharedCorrectionHistory { table, mask: slots - 1 }
    }

    /// Reallocate for a new thread count. Zeroes the table, like resizing the
    /// transposition table does; must not run during a search.
    pub fn resize(&mut self, threads: usize) {
        if Self::slots_for(threads) != self.mask + 1 {
            *self = Self::new(threads);
        } else {
            self.clear();
        }
    }

    /// Zero every entry (`ucinewgame`). Serial: 1 MB at one thread, 32 MB at 24.
    pub fn clear(&mut self) {
        for entry in self.table.iter_mut() {
            *entry.get_mut() = 0;
        }
    }

    #[inline]
    fn slot(&self, kind: CorrKind, key: u64, stm: Color) -> &AtomicI16 {
        let idx = ((key as usize & self.mask) * 2 + stm.index()) * CORR_KINDS + kind.index();
        debug_assert!(idx < self.table.len(), "SharedCorrectionHistory: index {} OOB", idx);
        &self.table[idx]
    }

    /// Correction value for a key and side to move.
    #[inline]
    pub fn get(&self, kind: CorrKind, key: u64, stm: Color) -> i32 {
        self.slot(kind, key, stm).load(Ordering::Relaxed) as i32
    }

    /// Gravity update: `val += bonus - val * |bonus| / CORRHIST_LIMIT`.
    ///
    /// Load, compute, store — never a fetch-and-add or a compare-exchange loop, both
    /// of which cost far more than the observation is worth on a line this hot.
    #[inline]
    pub fn update(&self, kind: CorrKind, key: u64, stm: Color, bonus: i32) {
        debug_assert!(bonus.abs() <= CORRHIST_LIMIT,
            "SharedCorrectionHistory::update: bonus {} exceeds limit {}", bonus, CORRHIST_LIMIT);
        let slot = self.slot(kind, key, stm);
        let val = slot.load(Ordering::Relaxed) as i32;
        let new_val = val + bonus - val * bonus.abs() / CORRHIST_LIMIT;
        slot.store(new_val.clamp(-CORRHIST_LIMIT, CORRHIST_LIMIT) as i16, Ordering::Relaxed);
    }
}

/// Piece-to-square correction subtable: `[piece][to_sq]` (i16).
///
/// One subtable exists per previous move's (piece, to_sq) context.
/// Size per subtable: 12 * 64 * 2 bytes = 1,536 bytes (~1.5 KB).
pub type PieceToCorrTable = [[i16; Square::NUM]; Piece::NUM];

/// Continuation correction history: `[prev_piece][prev_to]` → `PieceToCorrTable`.
///
/// Tracks how the static eval error correlates with the sequence of recent moves.
/// Same pointer-on-stack pattern as `ContinuationHistory`.
/// Total: 12 * 64 * 1536 bytes ≈ 1.125 MB (heap-allocated).
pub struct ContCorrectionHistory {
    table: Box<[[PieceToCorrTable; Square::NUM]; Piece::NUM]>,
}

impl ContCorrectionHistory {
    pub fn new() -> Self {
        let table = unsafe {
            let layout = std::alloc::Layout::new::<[[PieceToCorrTable; Square::NUM]; Piece::NUM]>();
            let ptr = std::alloc::alloc_zeroed(layout);
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            Box::from_raw(ptr.cast())
        };
        ContCorrectionHistory { table }
    }

    pub fn clear(&mut self) {
        unsafe {
            let ptr = self.table.as_mut() as *mut _ as *mut u8;
            let size = std::mem::size_of::<[[PieceToCorrTable; Square::NUM]; Piece::NUM]>();
            std::ptr::write_bytes(ptr, 0, size);
        }
    }

    /// Get a mutable pointer to the subtable for a given move context.
    #[inline]
    pub fn subtable_ptr(&mut self, piece: Piece, to: Square) -> *mut PieceToCorrTable {
        debug_assert!(piece.0 < 12, "ContCorrectionHistory::subtable_ptr: piece NONE");
        debug_assert!(to.0 < 64, "ContCorrectionHistory::subtable_ptr: sq OOB {}", to.0);
        &raw mut self.table[piece.index()][to.index()]
    }

    /// Get the correction value from a subtable pointer.
    #[inline]
    pub fn get(subtable_ptr: *const PieceToCorrTable, piece: Piece, to: Square) -> i32 {
        if subtable_ptr.is_null() {
            return 0;
        }
        debug_assert!(piece.0 < 12, "ContCorrectionHistory::get: piece NONE");
        debug_assert!(to.0 < 64, "ContCorrectionHistory::get: sq OOB {}", to.0);
        unsafe { (*subtable_ptr)[piece.index()][to.index()] as i32 }
    }

    /// Gravity update via a subtable pointer.
    #[inline]
    pub fn update(subtable_ptr: *mut PieceToCorrTable, piece: Piece, to: Square, bonus: i32) {
        if subtable_ptr.is_null() {
            return;
        }
        debug_assert!(piece.0 < 12, "ContCorrectionHistory::update: piece NONE");
        debug_assert!(to.0 < 64, "ContCorrectionHistory::update: sq OOB {}", to.0);
        let entry = unsafe { &mut (*subtable_ptr)[piece.index()][to.index()] };
        let val = *entry as i32;
        let new_val = val + bonus - val * bonus.abs() / CORRHIST_LIMIT;
        *entry = new_val.clamp(-CORRHIST_LIMIT, CORRHIST_LIMIT) as i16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KINDS: [CorrKind; 4] = [
        CorrKind::Pawn,
        CorrKind::Minor,
        CorrKind::NonPawnWhite,
        CorrKind::NonPawnBlack,
    ];

    #[test]
    fn one_thread_lands_on_the_entry_the_private_tables_used() {
        // The whole no-Elo-at-one-thread claim rests on this: a lone thread must
        // address exactly the slot the per-thread table addressed, so the search
        // reads the same numbers and the bench counts the same nodes.
        let h = SharedCorrectionHistory::new(1);
        assert_eq!(h.mask, CORRHIST_SIZE - 1);
        let mut key = 0x9E3779B97F4A7C15u64;
        for _ in 0..2000 {
            key = key.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            assert_eq!(key as usize & h.mask, key as usize % CORRHIST_SIZE);
        }
    }

    #[test]
    fn the_table_grows_with_the_pool_and_stays_a_power_of_two() {
        for (threads, expected) in [(1, 1), (2, 2), (3, 4), (4, 4), (5, 8), (12, 16), (24, 32)] {
            let h = SharedCorrectionHistory::new(threads);
            let slots = h.mask + 1;
            assert_eq!(slots, CORRHIST_SIZE * expected, "{threads} threads");
            assert!(slots.is_power_of_two(), "{threads} threads: {slots} slots");
            assert_eq!(h.table.len(), slots * 2 * CORR_KINDS);
        }
    }

    #[test]
    fn the_four_kinds_share_a_slot_without_sharing_a_value() {
        // They are one allocation but four tables: a bonus fed to one key must not
        // show up under another kind, or the corrections would cross-contaminate.
        let h = SharedCorrectionHistory::new(4);
        let key = 0xDEADBEEFCAFEu64;
        h.update(CorrKind::Pawn, key, Color::White, 300);
        assert_eq!(h.get(CorrKind::Pawn, key, Color::White), 300);
        for kind in KINDS.iter().copied().filter(|k| *k != CorrKind::Pawn) {
            assert_eq!(h.get(kind, key, Color::White), 0, "{kind:?} moved");
        }
        // Nor across sides to move.
        assert_eq!(h.get(CorrKind::Pawn, key, Color::Black), 0);
    }

    #[test]
    fn gravity_saturates_instead_of_running_away() {
        let h = SharedCorrectionHistory::new(1);
        let key = 12345u64;
        for _ in 0..500 {
            h.update(CorrKind::Minor, key, Color::Black, CORRHIST_LIMIT);
        }
        let v = h.get(CorrKind::Minor, key, Color::Black);
        assert!(v > 0 && v <= CORRHIST_LIMIT, "value {v} outside [0, {CORRHIST_LIMIT}]");
        for _ in 0..500 {
            h.update(CorrKind::Minor, key, Color::Black, -CORRHIST_LIMIT);
        }
        let v = h.get(CorrKind::Minor, key, Color::Black);
        assert!(v < 0 && v >= -CORRHIST_LIMIT, "value {v} outside [-{CORRHIST_LIMIT}, 0]");
    }

    #[test]
    fn gravity_matches_the_formula_the_private_tables_used() {
        let h = SharedCorrectionHistory::new(1);
        let key = 777u64;
        let mut expected = 0i32;
        for bonus in [200, -50, 900, -1024, 17, 640] {
            expected = expected + bonus - expected * bonus.abs() / CORRHIST_LIMIT;
            expected = expected.clamp(-CORRHIST_LIMIT, CORRHIST_LIMIT);
            h.update(CorrKind::NonPawnWhite, key, Color::White, bonus);
            assert_eq!(h.get(CorrKind::NonPawnWhite, key, Color::White), expected);
        }
    }

    #[test]
    fn clearing_and_resizing_leave_nothing_behind() {
        let mut h = SharedCorrectionHistory::new(2);
        h.update(CorrKind::Pawn, 42, Color::White, 500);
        h.clear();
        assert_eq!(h.get(CorrKind::Pawn, 42, Color::White), 0);

        h.update(CorrKind::Pawn, 42, Color::White, 500);
        h.resize(8); // different size: reallocates
        assert_eq!(h.mask + 1, CORRHIST_SIZE * 8);
        assert_eq!(h.get(CorrKind::Pawn, 42, Color::White), 0);

        h.update(CorrKind::Pawn, 42, Color::White, 500);
        h.resize(7); // same size once rounded up: must still come back empty
        assert_eq!(h.mask + 1, CORRHIST_SIZE * 8);
        assert_eq!(h.get(CorrKind::Pawn, 42, Color::White), 0);
    }

    #[test]
    fn the_shared_tables_can_cross_a_thread_boundary() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<SharedCorrectionHistory>();
        assert_sync::<crate::threads::SharedState>();
    }

    #[test]
    fn concurrent_updates_never_write_out_of_range() {
        // Losing an update is the accepted cost; writing a value outside the gravity
        // range is not, and would poison every eval that reads the slot afterwards.
        let h = std::sync::Arc::new(SharedCorrectionHistory::new(4));
        std::thread::scope(|s| {
            for t in 0..4 {
                let h = h.clone();
                s.spawn(move || {
                    let bonus = if t % 2 == 0 { CORRHIST_LIMIT / 4 } else { -CORRHIST_LIMIT / 4 };
                    for i in 0..20_000u64 {
                        h.update(CorrKind::Pawn, i % 8, Color::White, bonus);
                    }
                });
            }
        });
        for key in 0..8u64 {
            let v = h.get(CorrKind::Pawn, key, Color::White);
            assert!(v.abs() <= CORRHIST_LIMIT, "key {key} holds {v}");
        }
    }
}
