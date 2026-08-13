//! Move ordering data: [butterfly history](https://www.chessprogramming.org/History_Heuristic),
//! [capture history](https://www.chessprogramming.org/History_Heuristic#Capture_History),
//! [continuation history](https://www.chessprogramming.org/Continuation_History),
//! [countermove heuristic](https://www.chessprogramming.org/Countermove_Heuristic),
//! [killer moves](https://www.chessprogramming.org/Killer_Heuristic),
//! and [correction history](https://www.chessprogramming.org/Static_Evaluation_Correction_History).

use crate::types::{Color, Move, Piece, PieceType, Square, MAX_PLY};

/// Maximum history value for gravity formula.
const MAX_HISTORY: i32 = 16384;

/// Maximum correction history value for gravity formula.
pub const CORRHIST_LIMIT: i32 = 1024;

/// Number of entries in correction history tables.
const CORRHIST_SIZE: usize = 16384;

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

/// Pawn correction history: `[pawn_key % SIZE][stm]`.
///
/// Tracks the difference between static evaluation and search score
/// for positions with similar pawn structures, then adjusts future
/// static evaluations accordingly.
/// Size: 16384 * 2 * 2 bytes = 64 KB.
pub struct PawnCorrectionHistory {
    table: [[i16; 2]; CORRHIST_SIZE],
}

impl PawnCorrectionHistory {
    pub fn new() -> Self {
        PawnCorrectionHistory {
            table: [[0i16; 2]; CORRHIST_SIZE],
        }
    }

    pub fn clear(&mut self) {
        for entry in &mut self.table {
            entry.fill(0);
        }
    }

    #[inline]
    fn index(pawn_key: u64) -> usize {
        pawn_key as usize % CORRHIST_SIZE
    }

    /// Get the correction value for the given pawn key and side to move.
    #[inline]
    pub fn get(&self, pawn_key: u64, stm: Color) -> i32 {
        self.table[Self::index(pawn_key)][stm.index()] as i32
    }

    /// Gravity update: `val += bonus - val * |bonus| / CORRHIST_LIMIT`.
    #[inline]
    pub fn update(&mut self, pawn_key: u64, stm: Color, bonus: i32) {
        debug_assert!(bonus.abs() <= CORRHIST_LIMIT,
            "PawnCorrectionHistory::update: bonus {} exceeds limit {}", bonus, CORRHIST_LIMIT);
        let entry = &mut self.table[Self::index(pawn_key)][stm.index()];
        let val = *entry as i32;
        let new_val = val + bonus - val * bonus.abs() / CORRHIST_LIMIT;
        *entry = new_val.clamp(-CORRHIST_LIMIT, CORRHIST_LIMIT) as i16;
    }
}

/// Non-pawn correction history: `[piece_color][non_pawn_key % SIZE][stm]`.
///
/// Tracks eval error correlation with the arrangement of non-pawn pieces
/// (N, B, R, Q, K) for each color independently. Both white and black
/// tables are looked up and summed, then both updated with the same bonus.
/// Size: 2 * 16384 * 2 * 2 bytes = 128 KB.
pub struct NonPawnCorrectionHistory {
    table: [[[i16; 2]; CORRHIST_SIZE]; 2],
}

impl NonPawnCorrectionHistory {
    pub fn new() -> Self {
        NonPawnCorrectionHistory {
            table: [[[0i16; 2]; CORRHIST_SIZE]; 2],
        }
    }

    pub fn clear(&mut self) {
        for color_table in &mut self.table {
            for entry in color_table.iter_mut() {
                entry.fill(0);
            }
        }
    }

    #[inline]
    fn index(key: u64) -> usize {
        key as usize % CORRHIST_SIZE
    }

    #[inline]
    pub fn get(&self, key: u64, piece_color: Color, stm: Color) -> i32 {
        self.table[piece_color.index()][Self::index(key)][stm.index()] as i32
    }

    #[inline]
    pub fn update(&mut self, key: u64, piece_color: Color, stm: Color, bonus: i32) {
        debug_assert!(bonus.abs() <= CORRHIST_LIMIT,
            "NonPawnCorrectionHistory::update: bonus {} exceeds limit {}", bonus, CORRHIST_LIMIT);
        let entry = &mut self.table[piece_color.index()][Self::index(key)][stm.index()];
        let val = *entry as i32;
        let new_val = val + bonus - val * bonus.abs() / CORRHIST_LIMIT;
        *entry = new_val.clamp(-CORRHIST_LIMIT, CORRHIST_LIMIT) as i16;
    }
}

/// Minor piece correction history: `[minor_key % SIZE][stm]`.
///
/// Tracks eval error correlation with the arrangement of minor pieces
/// (N, B, K) for both colors combined in a single Zobrist key.
/// Size: 16384 * 2 * 2 bytes = 64 KB.
///
/// Uses a combined key for N+B+K pieces.
pub struct MinorCorrectionHistory {
    table: [[i16; 2]; CORRHIST_SIZE],
}

impl MinorCorrectionHistory {
    pub fn new() -> Self {
        MinorCorrectionHistory {
            table: [[0i16; 2]; CORRHIST_SIZE],
        }
    }

    pub fn clear(&mut self) {
        for entry in &mut self.table {
            entry.fill(0);
        }
    }

    #[inline]
    fn index(minor_key: u64) -> usize {
        minor_key as usize % CORRHIST_SIZE
    }

    /// Get the correction value for the given minor key and side to move.
    #[inline]
    pub fn get(&self, minor_key: u64, stm: Color) -> i32 {
        self.table[Self::index(minor_key)][stm.index()] as i32
    }

    /// Gravity update: `val += bonus - val * |bonus| / CORRHIST_LIMIT`.
    #[inline]
    pub fn update(&mut self, minor_key: u64, stm: Color, bonus: i32) {
        debug_assert!(bonus.abs() <= CORRHIST_LIMIT,
            "MinorCorrectionHistory::update: bonus {} exceeds limit {}", bonus, CORRHIST_LIMIT);
        let entry = &mut self.table[Self::index(minor_key)][stm.index()];
        let val = *entry as i32;
        let new_val = val + bonus - val * bonus.abs() / CORRHIST_LIMIT;
        *entry = new_val.clamp(-CORRHIST_LIMIT, CORRHIST_LIMIT) as i16;
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
