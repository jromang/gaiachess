//! [PeSTO evaluation](https://www.chessprogramming.org/PeSTO%27s_Evaluation_Function) —
//! piece-square tables with tapered eval.
//!
//! Tables from Ronald Friederich's RofChade, Texel-tuned. Middlegame and endgame
//! scores are blended by game phase (knight/bishop=1, rook=2, queen=4, max=24).

use crate::bitboard;
use crate::position::Position;
use crate::types::{Color, Piece, PieceType};

/// Middlegame piece base values.
const MG_VALUE: [i32; 6] = [82, 337, 365, 477, 1025, 0];

/// Endgame piece base values.
const EG_VALUE: [i32; 6] = [94, 281, 297, 512, 936, 0];

/// Game phase increment per piece type (Pawn=0, N=1, B=1, R=2, Q=4, K=0).
pub const PHASE_INC: [i32; 6] = [0, 1, 1, 2, 4, 0];

/// Total game phase at start (2N + 2B + 2R + 1Q per side, but max clamp).
pub const TOTAL_PHASE: i32 = 24;

/// Tempo bonus (centipawns) for the side to move.
pub const TEMPO_MG: i32 = 15;
pub const TEMPO_EG: i32 = 10;

// ============================================================
// Raw PeSTO tables (CPW coordinate system: a8=0, h1=63)
// ============================================================

#[rustfmt::skip]
const MG_PAWN_TABLE: [i32; 64] = [
      0,   0,   0,   0,   0,   0,   0,   0,
     98, 134,  61,  95,  68, 126,  34, -11,
     -6,   7,  26,  31,  65,  56,  25, -20,
    -14,  13,   6,  21,  23,  12,  17, -23,
    -27,  -2,  -5,  12,  17,   6,  10, -25,
    -26,  -4,  -4, -10,   3,   3,  33, -12,
    -35,  -1, -20, -23, -15,  24,  38, -22,
      0,   0,   0,   0,   0,   0,   0,   0,
];

#[rustfmt::skip]
const EG_PAWN_TABLE: [i32; 64] = [
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0,
];

#[rustfmt::skip]
const MG_KNIGHT_TABLE: [i32; 64] = [
   -167, -89, -34, -49,  61, -97, -15,-107,
    -73, -41,  72,  36,  23,  62,   7, -17,
    -47,  60,  37,  65,  84, 129,  73,  44,
     -9,  17,  19,  53,  37,  69,  18,  22,
    -13,   4,  16,  13,  28,  19,  21,  -8,
    -23,  -9,  12,  10,  19,  17,  25, -16,
    -29, -53, -12,  -3,  -1,  18, -14, -19,
   -105, -21, -58, -33, -17, -28, -19, -23,
];

#[rustfmt::skip]
const EG_KNIGHT_TABLE: [i32; 64] = [
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64,
];

#[rustfmt::skip]
const MG_BISHOP_TABLE: [i32; 64] = [
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21,
];

#[rustfmt::skip]
const EG_BISHOP_TABLE: [i32; 64] = [
    -14, -21, -11,  -8,  -7,  -9, -17, -24,
     -8,  -4,   7, -12,  -3, -13,  -4, -14,
      2,  -8,   0,  -1,  -2,   6,   0,   4,
     -3,   9,  12,   9,  14,  10,   3,   2,
     -6,   3,  13,  19,   7,  10,  -3,  -9,
    -12,  -3,   8,  10,  13,   3,  -7, -15,
    -14, -18,  -7,  -1,   4,  -9, -15, -27,
    -23,  -9, -23,  -5,  -9, -16,  -5, -17,
];

#[rustfmt::skip]
const MG_ROOK_TABLE: [i32; 64] = [
     32,  42,  32,  51,  63,   9,  31,  43,
     27,  32,  58,  62,  80,  67,  26,  44,
     -5,  19,  26,  36,  17,  45,  61,  16,
    -24, -11,   7,  26,  24,  35,  -8, -20,
    -36, -26, -12,  -1,   9,  -7,   6, -23,
    -45, -25, -16, -17,   3,   0,  -5, -33,
    -44, -16, -20,  -9,  -1,  11,  -6, -71,
    -19, -13,   1,  17,  16,   7, -37, -26,
];

#[rustfmt::skip]
const EG_ROOK_TABLE: [i32; 64] = [
     13,  10,  18,  15,  12,  12,   8,   5,
     11,  13,  13,  11,  -3,   3,   8,   3,
      7,   7,   7,   5,   4,  -3,  -5,  -3,
      4,   3,  13,   1,   2,   1,  -1,   2,
      3,   5,   8,   4,  -5,  -6,  -8, -11,
     -4,   0,  -5,  -1,  -7, -12,  -8, -16,
     -6,  -6,   0,   2,  -9,  -9, -11,  -3,
     -9,   2,   3,  -1,  -5, -13,   4, -20,
];

#[rustfmt::skip]
const MG_QUEEN_TABLE: [i32; 64] = [
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -31, -50,
];

#[rustfmt::skip]
const EG_QUEEN_TABLE: [i32; 64] = [
     -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  23,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41,
];

#[rustfmt::skip]
const MG_KING_TABLE: [i32; 64] = [
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14,
];

#[rustfmt::skip]
const EG_KING_TABLE: [i32; 64] = [
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
     -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43,
];

// ============================================================
// Combined tables: MG_TABLE[pc][sq] = piece_value + pst_bonus
// Adapted for LERF mapping (a1=0): white uses sq^56, black uses sq directly.
// ============================================================

const fn build_tables() -> ([[i32; 64]; 12], [[i32; 64]; 12]) {
    let mg_raw: [[i32; 64]; 6] = [
        MG_PAWN_TABLE, MG_KNIGHT_TABLE, MG_BISHOP_TABLE,
        MG_ROOK_TABLE, MG_QUEEN_TABLE, MG_KING_TABLE,
    ];
    let eg_raw: [[i32; 64]; 6] = [
        EG_PAWN_TABLE, EG_KNIGHT_TABLE, EG_BISHOP_TABLE,
        EG_ROOK_TABLE, EG_QUEEN_TABLE, EG_KING_TABLE,
    ];

    let mut mg = [[0i32; 64]; 12];
    let mut eg = [[0i32; 64]; 12];

    let mut pt = 0;
    while pt < 6 {
        let white_pc = pt * 2;     // Piece::new(pt, White)
        let black_pc = pt * 2 + 1; // Piece::new(pt, Black)
        let mut sq = 0;
        while sq < 64 {
            // White: raw table is in CPW coords (a8=0), our LERF has a1=0, so flip rank
            mg[white_pc][sq] = MG_VALUE[pt] + mg_raw[pt][sq ^ 56];
            eg[white_pc][sq] = EG_VALUE[pt] + eg_raw[pt][sq ^ 56];
            // Black: flip for perspective (same position bonus as white on mirrored rank)
            mg[black_pc][sq] = MG_VALUE[pt] + mg_raw[pt][sq];
            eg[black_pc][sq] = EG_VALUE[pt] + eg_raw[pt][sq];
            sq += 1;
        }
        pt += 1;
    }

    (mg, eg)
}

const TABLES: ([[i32; 64]; 12], [[i32; 64]; 12]) = build_tables();

/// Combined MG piece-square table: `MG_TABLE[piece][square]`.
pub(crate) const MG_TABLE: [[i32; 64]; 12] = TABLES.0;

/// Combined EG piece-square table: `EG_TABLE[piece][square]`.
pub(crate) const EG_TABLE: [[i32; 64]; 12] = TABLES.1;

/// The combined tables with the side's sign folded in: White entries as they are,
/// Black entries negated. The incremental update adds and subtracts these directly,
/// instead of deriving a sign from the piece and multiplying by it on every move.
const fn build_signed(t: &[[i32; 64]; 12]) -> [[i32; 64]; 12] {
    let mut out = [[0i32; 64]; 12];
    let mut pc = 0;
    while pc < 12 {
        // Piece = type * 2 + colour, White = 0.
        let sign = if pc % 2 == 0 { 1 } else { -1 };
        let mut sq = 0;
        while sq < 64 {
            out[pc][sq] = sign * t[pc][sq];
            sq += 1;
        }
        pc += 1;
    }
    out
}

/// `MG_TABLE` with the side's sign applied (White positive, Black negative).
pub(crate) static MG_SIGNED: [[i32; 64]; 12] = build_signed(&TABLES.0);

/// `EG_TABLE` with the side's sign applied (White positive, Black negative).
pub(crate) static EG_SIGNED: [[i32; 64]; 12] = build_signed(&TABLES.1);

/// Dark squares bitmask (A1, C1, E1, G1, B2, D2, F2, H2, ...).
/// In LERF: rank 1 = 0x55 (A1=dark), rank 2 = 0xAA (B2=dark), alternating.
const DARK_SQUARES: u64 = 0xAA55_AA55_AA55_AA55;

/// Returns true if the pawnless position is a theoretical draw due to insufficient
/// material. Must only be called when **no pawns** are on the board.
///
/// Includes a KBBvK same-color check.
/// Covers: KvK, KBvK, KNvK, KNNvK, KBvKB (same-color only), KNvKN, KBvKN,
/// KBBvK (same-color only), KRvKR (+≤1 minor each), KRvKB, KRvKN, KRvKBB,
/// KRvKNN.
pub fn is_material_draw(pos: &Position) -> bool {
    debug_assert!(
        pos.pieces[Piece::WHITE_PAWN.index()] == 0
            && pos.pieces[Piece::BLACK_PAWN.index()] == 0,
        "is_material_draw called with pawns on the board"
    );

    let queens = pos.pieces[Piece::WHITE_QUEEN.index()]
        | pos.pieces[Piece::BLACK_QUEEN.index()];
    if queens != 0 {
        return false;
    }

    let white = pos.occupancies[0];
    let black = pos.occupancies[1];
    let bishops = pos.pieces[Piece::WHITE_BISHOP.index()]
        | pos.pieces[Piece::BLACK_BISHOP.index()];
    let knights = pos.pieces[Piece::WHITE_KNIGHT.index()]
        | pos.pieces[Piece::BLACK_KNIGHT.index()];
    let rooks = pos.pieces[Piece::WHITE_ROOK.index()]
        | pos.pieces[Piece::BLACK_ROOK.index()];
    let wb = bishops & white;
    let bb = bishops & black;
    let wn = knights & white;
    let bn = knights & black;
    let wr = rooks & white;
    let br = rooks & black;

    if rooks == 0 {
        if bishops == 0 {
            // Knights only: draw if each side has < 3 knights
            // Covers KvK, KNvK, KNNvK, KNvKN, KNNvKN, KNNvKNN
            return wn.count_ones() < 3 && bn.count_ones() < 3;
        }
        if knights == 0 {
            // Bishops only: draw if bishop count diff < 2
            // Covers KBvK, KBvKB, KBBvK (but see same-color check below)
            let wbc = wb.count_ones();
            let bbc = bb.count_ones();

            // KBBvK: draw only if same-color bishops (opposite force mate in ≤19)
            if wbc == 2 && bbc == 0 {
                let on_dark = (wb & DARK_SQUARES).count_ones();
                return on_dark == 0 || on_dark == 2; // same color = draw
            }
            if bbc == 2 && wbc == 0 {
                let on_dark = (bb & DARK_SQUARES).count_ones();
                return on_dark == 0 || on_dark == 2;
            }

            // abs_diff < 2 covers KBvK (1v0) and KBvKB (1v1)
            // KBvKB is always drawn (1 bishop can never mate regardless of colors)
            return wbc.abs_diff(bbc) < 2;
        }
        // Mixed bishops + knights: draw only if exactly 1 minor each
        // Covers KBvKN, KNvKB. Excludes KBNvK (forced mate).
        return (wb | wn).count_ones() == 1 && (bb | bn).count_ones() == 1;
    }

    if wr.count_ones() == 1 && br.count_ones() == 1 {
        // Both sides have exactly 1 rook: draw with ≤1 minor each
        // Covers KRvKR, KRBvKR, KRNvKR, KRvKRB, KRvKRN
        return (wn | wb).count_ones() < 2 && (bn | bb).count_ones() < 2;
    }

    if wr.count_ones() == 1 && br == 0 {
        // White has 1 rook, black has none: draw if white has no minors
        // and black has 1-2 minors (KRvKB, KRvKN, KRvKBB, KRvKNN)
        if (wn | wb) == 0 {
            let bmc = (bn | bb).count_ones();
            return bmc == 1 || bmc == 2;
        }
    } else if br.count_ones() == 1 && wr == 0 {
        // Symmetric: black has 1 rook
        if (bn | bb) == 0 {
            let wmc = (wn | wb).count_ones();
            return wmc == 1 || wmc == 2;
        }
    }

    false
}

/// What the pieces are worth, and nothing about where they stand.
///
/// The bottom of the skill ladder judges with this and no more (see [`crate::skill`]).
/// Someone who has just learned the moves knows a queen beats a knight and very little
/// else; handing them piece squares as well would produce a beginner who develops
/// soundly and castles early, which is not a beginner.
///
/// Tapered on the same phase as the full evaluation, so blending the two is a straight
/// interpolation, and from the side to move's perspective. No tempo bonus: the value of
/// having the move is exactly the sort of thing this level of play does not know.
pub fn material_eval(pos: &Position) -> i32 {
    let (mut mg, mut eg) = (0, 0);
    for pt in [PieceType::Pawn, PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen] {
        let count = pos.piece_type_bb(pt, Color::White).count_ones() as i32
            - pos.piece_type_bb(pt, Color::Black).count_ones() as i32;
        mg += MG_VALUE[pt as usize] * count;
        eg += EG_VALUE[pt as usize] * count;
    }

    let mg_phase = pos.game_phase.min(TOTAL_PHASE);
    let eg_phase = TOTAL_PHASE - mg_phase;
    debug_assert!(mg_phase >= 0 && eg_phase >= 0, "phase {mg_phase} out of range");
    let score = (mg * mg_phase + eg * eg_phase) / TOTAL_PHASE;
    if pos.side_to_move == Color::White { score } else { -score }
}

/// Evaluate the position from the side-to-move's perspective.
///
/// Returns a score in centipawns using tapered evaluation between middlegame
/// and endgame piece-square table values.
pub fn evaluate(pos: &Position) -> i32 {
    debug_assert!(crate::bitboard::popcount(pos.occupied()) >= 2,
        "eval: less than 2 pieces on board");
    // Eval is never called when in check
    debug_assert!(pos.checkers == 0,
        "eval: called while in check (checkers=0x{:016x})", pos.checkers);
    let mut mg = [0i32; 2]; // [White, Black]
    let mut eg = [0i32; 2];
    let mut phase = 0i32;

    // Iterate over all piece types and colors via bitboards
    let mut pc = 0u8;
    while pc < 12 {
        let color = (pc & 1) as usize;
        let pt = (pc >> 1) as usize;
        let mut bb = pos.pieces[pc as usize];
        while bb != 0 {
            let sq = bitboard::pop_lsb(&mut bb).index();
            mg[color] += MG_TABLE[pc as usize][sq];
            eg[color] += EG_TABLE[pc as usize][sq];
            phase += PHASE_INC[pt];
        }
        pc += 1;
    }

    // Tapered eval: blend MG and EG scores by game phase
    let mg_score = mg[Color::White.index()] - mg[Color::Black.index()];
    let eg_score = eg[Color::White.index()] - eg[Color::Black.index()];

    let mg_phase = if phase > TOTAL_PHASE { TOTAL_PHASE } else { phase };
    let eg_phase = TOTAL_PHASE - mg_phase;
    let score = (mg_score * mg_phase + eg_score * eg_phase) / TOTAL_PHASE;

    // Tempo bonus: small advantage for the side to move
    let tempo = (TEMPO_MG * mg_phase + TEMPO_EG * eg_phase) / TOTAL_PHASE;

    // Return from side-to-move's perspective
    if pos.side_to_move == Color::White { score + tempo } else { -score + tempo }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Position;

    /// Material only: even at the start, blind to the tempo the full eval sees.
    #[test]
    fn material_eval_counts_pieces_and_nothing_else() {
        let start = Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        assert_eq!(material_eval(&start), 0, "an even position is even, tempo included");

        // A move that the full eval rewards for its piece square is worth nothing here.
        let developed = Position::from_fen("rnbqkbnr/pppppppp/8/8/8/4PN2/PPPP1PPP/RNBQKB1R b KQkq - 0 1").unwrap();
        assert_eq!(material_eval(&developed), 0, "piece squares must not leak in");

        let up_a_queen = Position::from_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        assert!(material_eval(&up_a_queen) > 800, "{}", material_eval(&up_a_queen));

        // The same board seen from the other side is the same score, negated.
        let same_from_black = Position::from_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1").unwrap();
        assert_eq!(material_eval(&same_from_black), -material_eval(&up_a_queen));
    }

    /// The endgame values differ from the middlegame ones, so what a piece is worth has
    /// to follow the phase — a pawn is worth more once the queens are gone.
    #[test]
    fn material_eval_tapers_like_the_full_eval() {
        // Nothing but kings and a pawn: phase zero, so the endgame value, exactly.
        let bare = Position::from_fen("4k3/8/8/8/8/8/P7/4K3 w - - 0 1").unwrap();
        assert_eq!(material_eval(&bare), EG_VALUE[PieceType::Pawn as usize]);

        // The same pawn with a full board around it is worth its middlegame value.
        let opening = Position::from_fen("rnbqkbnr/1ppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let full_phase = material_eval(&opening);
        assert_eq!(full_phase, MG_VALUE[PieceType::Pawn as usize]);
        assert!(full_phase < EG_VALUE[PieceType::Pawn as usize], "a pawn grows as pieces come off");
    }

    #[test]
    fn test_startpos_eval_symmetric() {
        // Starting position should evaluate to ~0 (symmetric)
        let pos = Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let score = evaluate(&pos);
        // Symmetric material + PST, but tempo bonus gives side-to-move a small edge
        assert!(score > 0 && score <= 20, "Start position should be small positive (tempo), got {score}");
    }

    #[test]
    fn test_white_up_material() {
        // White up a queen
        let pos = Position::from_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let score = evaluate(&pos);
        assert!(score > 800, "White up a queen should score > 800, got {score}");
    }

    #[test]
    fn test_eval_side_to_move() {
        // Same position, different side: material+PST is negated, but both get tempo bonus
        let w = Position::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1").unwrap();
        let b = Position::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let score_b = evaluate(&w); // from black's perspective
        let score_w = evaluate(&b); // from white's perspective
        // They differ by 2*tempo (each side gets +tempo from its own perspective)
        let diff = (score_w + score_b).abs();
        assert!(diff == 2 * TEMPO_MG, "Scores should differ by 2*tempo, got diff={diff}");
    }

    #[test]
    fn test_phase_endgame() {
        // King + pawn vs king: should be near full endgame phase
        let pos = Position::from_fen("8/8/8/8/4k3/8/4P3/4K3 w - - 0 1").unwrap();
        let score = evaluate(&pos);
        // White is up a pawn, should be positive
        assert!(score > 0, "White up a pawn should be positive, got {score}");
    }

    // ============================================================
    // Material draw detection tests
    // ============================================================

    fn check_material_draw(fen: &str, expected: bool) {
        let pos = Position::from_fen(fen)
            .unwrap_or_else(|e| panic!("Bad FEN '{fen}': {e}"));
        let result = is_material_draw(&pos);
        assert_eq!(result, expected, "is_material_draw(\"{fen}\") = {result}, expected {expected}");
    }

    #[test]
    fn test_kvk_draw() {
        check_material_draw("8/8/8/4k3/8/3K4/8/8 w - - 0 1", true);
    }

    #[test]
    fn test_kbvk_draw() {
        check_material_draw("8/8/3k4/8/8/3K4/4B3/8 w - - 0 1", true);
    }

    #[test]
    fn test_knvk_draw() {
        check_material_draw("8/8/3k4/8/8/3K1N2/8/8 w - - 0 1", true);
    }

    #[test]
    fn test_kvkn_draw() {
        check_material_draw("8/8/3k4/8/8/1n1K4/8/8 w - - 0 1", true);
    }

    #[test]
    fn test_knvkn_draw() {
        check_material_draw("8/8/3k4/2n5/8/3K1N2/8/8 w - - 0 1", true);
    }

    #[test]
    fn test_knnvk_draw() {
        check_material_draw("8/8/3k4/8/8/3K4/N1N5/8 w - - 0 1", true);
    }

    #[test]
    fn test_kbvkb_draw() {
        check_material_draw("8/8/3k4/8/4b3/3K4/4B3/8 w - - 0 1", true);
    }

    #[test]
    fn test_kbvkn_draw() {
        check_material_draw("8/8/3k4/2n5/8/3K4/4B3/8 w - - 0 1", true);
    }

    #[test]
    fn test_krvkr_draw() {
        check_material_draw("8/8/1k6/8/8/3K4/8/R3r3 w - - 0 1", true);
    }

    #[test]
    fn test_krvkn_draw() {
        check_material_draw("8/8/1k6/8/8/1n1K4/8/4R3 w - - 0 1", true);
    }

    #[test]
    fn test_krvkbn_draw() {
        check_material_draw("8/8/1k6/8/4b3/1n1K4/8/4R3 w - - 0 1", true);
    }

    #[test]
    fn test_krbvkr_draw() {
        check_material_draw("8/8/1k6/8/8/3K4/4B3/R3r3 w - - 0 1", true);
    }

    #[test]
    fn test_krnvkr_draw() {
        check_material_draw("8/8/1k6/8/8/1N1K4/8/R3r3 w - - 0 1", true);
    }

    // KBBvK: depends on bishop square colors
    #[test]
    fn test_kbbvk_same_color_draw() {
        // Both bishops on dark squares (a1=dark, c3=dark) → draw
        check_material_draw("8/8/1k6/8/8/2B1K3/8/B7 w - - 0 1", true);
    }

    #[test]
    fn test_kbbvk_opposite_color_not_draw() {
        // Bishops on opposite colors (b1=light, c1=dark) → mate possible
        check_material_draw("8/8/1k6/8/8/4K3/8/1BB5 w - - 0 1", false);
    }

    #[test]
    fn test_kvkbb_same_color_draw() {
        // Black has 2 same-color bishops → draw
        check_material_draw("8/8/1k6/8/8/2b1K3/8/b7 w - - 0 1", true);
    }

    #[test]
    fn test_kvkbb_opposite_color_not_draw() {
        // Black has opposite-color bishops (b1=light, c1=dark) → mate possible
        check_material_draw("8/8/1k6/8/8/4K3/8/1bb5 w - - 0 1", false);
    }

    // KBNvK: NOT a draw (forced mate)
    #[test]
    fn test_kbnvk_not_draw() {
        check_material_draw("8/8/1k6/8/8/3K4/4B3/5N2 w - - 0 1", false);
    }

    #[test]
    fn test_kvkbn_not_draw() {
        check_material_draw("8/8/1k6/8/8/3K4/4b3/5n2 w - - 0 1", false);
    }

    // NOT draws
    #[test]
    fn test_krvk_not_draw() {
        check_material_draw("8/8/1k6/8/8/3K4/8/4R3 w - - 0 1", false);
    }

    #[test]
    fn test_kqvk_not_draw() {
        check_material_draw("8/8/1k6/8/8/3K4/8/3Q4 w - - 0 1", false);
    }

    #[test]
    fn test_krrvk_not_draw() {
        check_material_draw("8/8/1k6/8/8/3K4/8/R6R w - - 0 1", false);
    }

    // ============================================================
    // Incremental PeSTO vs from-scratch PeSTO validation
    // ============================================================

    use crate::movegen;
    use crate::types::{Move, ArrayBuf, MAX_MOVES};

    /// Apply a UCI move string to a position (same as position.rs test helper).
    fn make_uci(pos: &mut Position, uci: &str) {
        let mut buf = ArrayBuf::<Move, MAX_MOVES>::new();
        let count = movegen::generate_legal_moves(pos, &mut buf);
        let m = (0..count)
            .map(|i| buf[i])
            .find(|m| m.to_uci() == uci)
            .unwrap_or_else(|| panic!("illegal move: {uci}"));
        pos.make_move(m);
    }

    /// Find a legal move matching UCI string (without applying it).
    fn find_uci(pos: &Position, uci: &str) -> Move {
        let mut buf = ArrayBuf::<Move, MAX_MOVES>::new();
        let count = movegen::generate_legal_moves(pos, &mut buf);
        (0..count)
            .map(|i| buf[i])
            .find(|m| m.to_uci() == uci)
            .unwrap_or_else(|| panic!("illegal move: {uci}"))
    }

    /// Compare pos.lazy_eval() (incremental) with evaluate(pos) (from scratch).
    fn assert_lazy_matches_full(pos: &Position, context: &str) {
        let lazy = pos.lazy_eval();
        let full = evaluate(pos);
        assert_eq!(lazy, full,
            "lazy_eval mismatch in {context}: lazy={lazy}, full={full}, fen={}",
            pos.to_fen());
    }

    #[test]
    fn test_lazy_eval_startpos() {
        let pos = Position::from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        assert_lazy_matches_full(&pos, "startpos");
    }

    #[test]
    fn test_lazy_eval_after_e4() {
        let mut pos = Position::from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        make_uci(&mut pos, "e2e4");
        assert_lazy_matches_full(&pos, "after 1.e4");
    }

    #[test]
    fn test_lazy_eval_captures() {
        // Scandinavian: 1.e4 d5 2.exd5
        let mut pos = Position::from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        for uci in &["e2e4", "d7d5", "e4d5"] {
            make_uci(&mut pos, uci);
            assert_lazy_matches_full(&pos, &format!("after {uci}"));
        }
    }

    #[test]
    fn test_lazy_eval_castling() {
        // Position where White can castle kingside
        let mut pos = Position::from_fen(
            "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4").unwrap();
        assert_lazy_matches_full(&pos, "before castling");
        make_uci(&mut pos, "e1g1"); // O-O
        assert_lazy_matches_full(&pos, "after O-O");
    }

    #[test]
    fn test_lazy_eval_promotion() {
        // White pawn on a7, promotes without giving check
        let mut pos = Position::from_fen(
            "8/P3k3/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        assert_lazy_matches_full(&pos, "before promotion");
        make_uci(&mut pos, "a7a8q");
        assert_lazy_matches_full(&pos, "after a8=Q");
    }

    #[test]
    fn test_lazy_eval_promotion_capture() {
        // White pawn on b7, capture-promote on a8 (not giving check)
        let mut pos = Position::from_fen(
            "r7/1P2k3/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        assert_lazy_matches_full(&pos, "before promo-capture");
        make_uci(&mut pos, "b7a8q");
        assert_lazy_matches_full(&pos, "after bxa8=Q");
    }

    #[test]
    fn test_lazy_eval_en_passant() {
        // En passant capture
        let mut pos = Position::from_fen(
            "rnbqkbnr/pppp1ppp/8/4pP2/8/8/PPPPP1PP/RNBQKBNR w KQkq e6 0 3").unwrap();
        assert_lazy_matches_full(&pos, "before EP");
        make_uci(&mut pos, "f5e6");
        assert_lazy_matches_full(&pos, "after fxe6 EP");
    }

    #[test]
    fn test_lazy_eval_unmake_restores() {
        // Make + unmake should restore lazy eval to original
        let mut pos = Position::from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let orig_mg = pos.psq_mg;
        let orig_eg = pos.psq_eg;
        let orig_phase = pos.game_phase;

        let m = find_uci(&pos, "e2e4");
        pos.make_move(m);
        assert_ne!(pos.psq_mg, orig_mg, "psq_mg should change after move");
        pos.unmake_move(m);
        assert_eq!(pos.psq_mg, orig_mg, "psq_mg not restored after unmake");
        assert_eq!(pos.psq_eg, orig_eg, "psq_eg not restored after unmake");
        assert_eq!(pos.game_phase, orig_phase, "game_phase not restored after unmake");
    }

    #[test]
    fn test_lazy_eval_null_move_preserves() {
        // Null move should not change psq scores
        let mut pos = Position::from_fen(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1").unwrap();
        let mg_before = pos.psq_mg;
        let eg_before = pos.psq_eg;
        let phase_before = pos.game_phase;

        pos.make_null_move();
        assert_eq!(pos.psq_mg, mg_before, "psq_mg changed after null move");
        assert_eq!(pos.psq_eg, eg_before, "psq_eg changed after null move");
        assert_eq!(pos.game_phase, phase_before, "game_phase changed after null move");

        pos.unmake_null_move();
        assert_eq!(pos.psq_mg, mg_before, "psq_mg changed after unmake null");
    }

    #[test]
    fn test_lazy_eval_game_sequence() {
        // Play a Ruy Lopez with varied move types and verify at each step
        let mut pos = Position::from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let moves = [
            "e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6",
            "b5a4", "g8f6", "e1g1", // O-O
            "f6e4", // Nxe4 capture
            "d2d4", "e5d4", // exd4 capture
            "f1e1", // Re1
        ];
        let mut played = Vec::new();
        for (i, uci) in moves.iter().enumerate() {
            let m = find_uci(&pos, uci);
            played.push(m);
            pos.make_move(m);
            assert_lazy_matches_full(&pos, &format!("move {}: {uci}", i + 1));
        }
        // Unmake all and verify restoration
        for m in played.iter().rev() {
            pos.unmake_move(*m);
        }
        let startpos = Position::from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        assert_eq!(pos.psq_mg, startpos.psq_mg, "psq_mg not restored after full unmake");
        assert_eq!(pos.psq_eg, startpos.psq_eg, "psq_eg not restored after full unmake");
        assert_eq!(pos.game_phase, startpos.game_phase, "game_phase not restored");
    }

    #[test]
    fn test_lazy_eval_endgame_position() {
        // Pure endgame: should have low game_phase
        let pos = Position::from_fen("8/8/4k3/8/8/4K3/4P3/8 w - - 0 1").unwrap();
        assert_lazy_matches_full(&pos, "KPvK endgame");
        assert_eq!(pos.game_phase, 0, "KPvK should have phase 0");
    }

    #[test]
    fn test_lazy_eval_queenside_castling() {
        let mut pos = Position::from_fen(
            "r3kbnr/pppqpppp/2n5/3p1b2/3P1B2/2N5/PPPQPPPP/R3KBNR w KQkq - 6 5").unwrap();
        assert_lazy_matches_full(&pos, "before O-O-O");
        make_uci(&mut pos, "e1c1"); // O-O-O
        assert_lazy_matches_full(&pos, "after O-O-O");
    }

    #[test]
    fn test_lazy_eval_underpromotion() {
        // Promote to knight (not giving check)
        let mut pos = Position::from_fen(
            "8/P3k3/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        make_uci(&mut pos, "a7a8n");
        assert_lazy_matches_full(&pos, "after a8=N");
    }

    #[test]
    fn test_lazy_eval_many_positions() {
        // Test from various FEN positions (only not-in-check, since evaluate() asserts)
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            // Endgame positions
            "8/8/3k4/8/8/3K4/4R3/8 w - - 0 1",
            "8/pppppppp/8/8/8/8/PPPPPPPP/4K2k w - - 0 1",
        ];
        for fen in &fens {
            let pos = Position::from_fen(fen).unwrap();
            if pos.checkers != 0 { continue; } // skip in-check positions
            assert_lazy_matches_full(&pos, &format!("FEN: {fen}"));
        }
    }
}
