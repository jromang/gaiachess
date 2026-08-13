/*
 * Pyrrhic configuration for GaiaChess
 *
 * Uses FFI to call back into Rust for attack generation
 * and bitboard manipulation functions.
 */

#pragma once

#include "gaiachess_bridge.h"

#define PYRRHIC_POPCOUNT(x)              (gaiachess_popcount(x))
#define PYRRHIC_LSB(x)                   (gaiachess_lsb(x))
#define PYRRHIC_POPLSB(x)               (gaiachess_poplsb(x))

#define PYRRHIC_PAWN_ATTACKS(sq, c)      (gaiachess_pawn_attacks(sq, c))
#define PYRRHIC_KNIGHT_ATTACKS(sq)       (gaiachess_knight_attacks(sq))
#define PYRRHIC_BISHOP_ATTACKS(sq, occ)  (gaiachess_bishop_attacks(sq, occ))
#define PYRRHIC_ROOK_ATTACKS(sq, occ)    (gaiachess_rook_attacks(sq, occ))
#define PYRRHIC_QUEEN_ATTACKS(sq, occ)   (gaiachess_queen_attacks(sq, occ))
#define PYRRHIC_KING_ATTACKS(sq)         (gaiachess_king_attacks(sq))
