//! Data filters for NNUE training.
//!
//! Phase 1: filter tactical, material [17,78], piece_count distribution, random skip 5%
//! Phase 2: no tactical filter, custom SEE filter (reject good captures)

use sfbinpack::chess::{
    attacks,
    bitboard::Bitboard,
    color::Color,
    coords::Square,
    r#move::{Move, MoveType},
    piece::Piece,
    piecetype::PieceType,
};
use sfbinpack::TrainingDataEntry;

use std::sync::atomic::{AtomicU64, Ordering};

// ============================================================
// Piece count rejection sampling
// ============================================================

#[rustfmt::skip]
const DESIRED_DISTRIBUTION: [f64; 33] = [
    0.018411966423, 0.020641545085, 0.022727271053,
    0.024669162740, 0.026467201733, 0.028121406444,
    0.029631758462, 0.030998276198, 0.032220941240,
    0.033299772000, 0.034234750067, 0.035025893853,
    0.035673184944, 0.036176641754, 0.036536245870,
    0.036752015705, 0.036823932846, 0.036752015705,
    0.036536245870, 0.036176641754, 0.035673184944,
    0.035025893853, 0.034234750067, 0.033299772000,
    0.032220941240, 0.030998276198, 0.029631758462,
    0.028121406444, 0.026467201733, 0.024669162740,
    0.022727271053, 0.020641545085, 0.018411966423,
];

static PIECE_COUNT_STATS: [AtomicU64; 33] = {
    let mut arr: [std::mem::MaybeUninit<AtomicU64>; 33] =
        [const { std::mem::MaybeUninit::uninit() }; 33];
    let mut i = 0;
    while i < 33 {
        arr[i] = std::mem::MaybeUninit::new(AtomicU64::new(0));
        i += 1;
    }
    unsafe { std::mem::transmute::<_, [AtomicU64; 33]>(arr) }
};
static PIECE_COUNT_TOTAL: AtomicU64 = AtomicU64::new(0);

/// Reset piece count statistics (call between phases).
pub fn reset_piece_count_stats() {
    for stat in &PIECE_COUNT_STATS {
        stat.store(0, Ordering::Relaxed);
    }
    PIECE_COUNT_TOTAL.store(0, Ordering::Relaxed);
}

fn piece_count_acceptance(piece_count: usize) -> f64 {
    let pc = piece_count.min(32);
    let count = PIECE_COUNT_STATS[pc].fetch_add(1, Ordering::Relaxed) + 1;
    let total = PIECE_COUNT_TOTAL.fetch_add(1, Ordering::Relaxed) + 1;
    let frequency = count as f64 / total as f64;
    (0.5 * DESIRED_DISTRIBUTION[pc] / frequency).clamp(0.0, 1.0)
}

// ============================================================
// Material value
// ============================================================

const SEE_PIECE_VALUES: [i32; 7] = [100, 300, 300, 500, 900, 0, 0]; // P,N,B,R,Q,K,None

fn material_value(pos: &sfbinpack::chess::position::Position) -> u32 {
    let mut mat = 0u32;
    for &pt in &[PieceType::Pawn, PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen] {
        let count = pos.pieces_bb_type(pt).count();
        let val = match pt {
            PieceType::Pawn => 1,
            PieceType::Knight | PieceType::Bishop => 3,
            PieceType::Rook => 5,
            PieceType::Queen => 9,
            _ => 0,
        };
        mat += count * val;
    }
    mat
}

// ============================================================
// Tactical move detection
// ============================================================

fn is_tactical(pos: &sfbinpack::chess::position::Position, mv: &Move) -> bool {
    let target = pos.piece_at(mv.to());
    target != Piece::NONE || mv.mtype() == MoveType::EnPassant || mv.mtype() == MoveType::Promotion
}

// ============================================================
// Static Exchange Evaluation
// ============================================================

fn estimated_see(pos: &sfbinpack::chess::position::Position, mv: &Move) -> i32 {
    let target = pos.piece_at(mv.to());
    let mut value = if target != Piece::NONE {
        SEE_PIECE_VALUES[target.piece_type().ordinal() as usize]
    } else {
        0
    };

    if mv.mtype() == MoveType::Promotion {
        let promo = mv.promoted_piece();
        value += SEE_PIECE_VALUES[promo.piece_type().ordinal() as usize] - SEE_PIECE_VALUES[0];
    } else if mv.mtype() == MoveType::EnPassant {
        value = SEE_PIECE_VALUES[0];
    }

    value
}

fn static_exchange_eval(
    pos: &sfbinpack::chess::position::Position,
    mv: &Move,
    threshold: i32,
) -> bool {
    let from = mv.from();
    let to = mv.to();

    // Estimated initial gain
    let mut balance = estimated_see(pos, mv) - threshold;
    if balance < 0 { return false; }

    // Value of the moving piece (what we stand to lose)
    let moving_piece = pos.piece_at(from);
    let mut next_victim_val = if mv.mtype() == MoveType::Promotion {
        SEE_PIECE_VALUES[mv.promoted_piece().piece_type().ordinal() as usize]
    } else {
        SEE_PIECE_VALUES[moving_piece.piece_type().ordinal() as usize]
    };

    balance -= next_victim_val;
    if balance >= 0 { return true; } // Even losing our piece, we're ahead

    let occ = pos.occupied();
    let bishops = pos.pieces_bb_type(PieceType::Bishop) | pos.pieces_bb_type(PieceType::Queen);
    let rooks = pos.pieces_bb_type(PieceType::Rook) | pos.pieces_bb_type(PieceType::Queen);

    // Remove the moving piece from occupancy
    let mut occupied = Bitboard::from_u64(occ.bits() ^ Bitboard::from_square(from).bits());
    // For en passant, also remove the captured pawn
    if mv.mtype() == MoveType::EnPassant {
        let ep_sq = Square::from_i32(to.index() as i32 + if pos.side_to_move() == Color::White { -8 } else { 8 });
        occupied = Bitboard::from_u64(occupied.bits() ^ Bitboard::from_square(ep_sq).bits());
    }

    // All attackers to the target square
    let mut attackers = get_all_attackers(pos, to, occupied) & occupied;

    let mut colour = pos.side_to_move();
    colour = if colour == Color::White { Color::Black } else { Color::White };

    loop {
        let our_attackers = attackers & pos.pieces_bb(colour);
        if our_attackers.count() == 0 { break; }

        // Find least valuable attacker
        let mut next_victim = PieceType::King;
        for &pt in &[PieceType::Pawn, PieceType::Knight, PieceType::Bishop,
                     PieceType::Rook, PieceType::Queen, PieceType::King] {
            if (our_attackers & pos.pieces_bb_type(pt)).count() > 0 {
                next_victim = pt;
                break;
            }
        }

        // Remove attacker from occupied
        let attacker_sq = (our_attackers & pos.pieces_bb_type(next_victim)).lsb();
        occupied = Bitboard::from_u64(occupied.bits() ^ Bitboard::from_square(attacker_sq).bits());

        // Update sliders (x-ray through the removed piece)
        if next_victim == PieceType::Pawn || next_victim == PieceType::Bishop || next_victim == PieceType::Queen {
            attackers = Bitboard::from_u64(attackers.bits() | (attacks::bishop(attacker_sq, occupied) & bishops & occupied).bits());
        }
        if next_victim == PieceType::Rook || next_victim == PieceType::Queen {
            attackers = Bitboard::from_u64(attackers.bits() | (attacks::rook(attacker_sq, occupied) & rooks & occupied).bits());
        }

        // Switch side
        colour = if colour == Color::White { Color::Black } else { Color::White };

        next_victim_val = SEE_PIECE_VALUES[next_victim.ordinal() as usize];
        balance = -balance - 1 - next_victim_val;

        if balance >= 0 {
            // King capture — check if opponent still has attackers
            if next_victim == PieceType::King && (attackers & pos.pieces_bb(colour)).count() > 0 {
                colour = if colour == Color::White { Color::Black } else { Color::White };
            }
            break;
        }
    }

    // The side that is to move after loop exit is the loser
    pos.side_to_move() != colour
}

fn get_all_attackers(
    pos: &sfbinpack::chess::position::Position,
    sq: Square,
    occupied: Bitboard,
) -> Bitboard {
    let mut attackers = Bitboard::from_u64(0);

    // Pawns
    attackers = Bitboard::from_u64(attackers.bits()
        | (attacks::pawn(Color::Black, sq) & pos.pieces_bb_color(Color::White, PieceType::Pawn)).bits()
        | (attacks::pawn(Color::White, sq) & pos.pieces_bb_color(Color::Black, PieceType::Pawn)).bits());

    // Knights
    attackers = Bitboard::from_u64(attackers.bits()
        | (attacks::knight(sq) & pos.pieces_bb_type(PieceType::Knight)).bits());

    // Bishops + Queens (diagonal)
    attackers = Bitboard::from_u64(attackers.bits()
        | (attacks::bishop(sq, occupied) & (pos.pieces_bb_type(PieceType::Bishop) | pos.pieces_bb_type(PieceType::Queen))).bits());

    // Rooks + Queens (straight)
    attackers = Bitboard::from_u64(attackers.bits()
        | (attacks::rook(sq, occupied) & (pos.pieces_bb_type(PieceType::Rook) | pos.pieces_bb_type(PieceType::Queen))).bits());

    // Kings
    attackers = Bitboard::from_u64(attackers.bits()
        | (attacks::king(sq) & pos.pieces_bb_type(PieceType::King)).bits());

    attackers
}

// ============================================================
// Phase 1 filter
// ============================================================

pub fn filter_phase1(entry: &TrainingDataEntry) -> bool {
    // Basic filters
    if entry.ply < 8 { return false; }
    if entry.score.unsigned_abs() >= 32000 { return false; }
    if entry.pos.is_checked(entry.pos.side_to_move()) { return false; }

    // Filter tactical moves
    if is_tactical(&entry.pos, &entry.mv) { return false; }

    // Material bounds [17, 78]
    let mat = material_value(&entry.pos);
    if mat < 17 || mat > 78 { return false; }

    // Random skip 5%
    use rand::Rng;
    let mut rng = rand::rng();
    if !rng.random_bool(0.95) { return false; }

    // Piece count rejection sampling
    let pc = entry.pos.occupied().count() as usize;
    let acceptance = piece_count_acceptance(pc);
    if !rng.random_bool(acceptance) { return false; }

    true
}

// ============================================================
// Phase 2 filter
// ============================================================

pub fn filter_phase2(entry: &TrainingDataEntry) -> bool {
    // Basic filters (same as phase 1 except filter_tactical=false)
    if entry.ply < 8 { return false; }
    if entry.score.unsigned_abs() >= 32000 { return false; }
    if entry.pos.is_checked(entry.pos.side_to_move()) { return false; }

    // NO tactical filter in phase 2

    // Material bounds [17, 78]
    let mat = material_value(&entry.pos);
    if mat < 17 || mat > 78 { return false; }

    // Random skip 5%
    use rand::Rng;
    let mut rng = rand::rng();
    if !rng.random_bool(0.95) { return false; }

    // Piece count rejection sampling
    let pc = entry.pos.occupied().count() as usize;
    let acceptance = piece_count_acceptance(pc);
    if !rng.random_bool(acceptance) { return false; }

    // Custom SEE filter: reject good captures (SEE >= 0)
    if is_tactical(&entry.pos, &entry.mv) && static_exchange_eval(&entry.pos, &entry.mv, 0) {
        return false;
    }

    true
}
