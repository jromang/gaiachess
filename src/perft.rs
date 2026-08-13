//! [Perft](https://www.chessprogramming.org/Perft) — performance test for move generation validation.
//!
//! Counts all leaf nodes at a given depth in the game tree, comparing results
//! against known reference values to verify correctness of move generation,
//! make/unmake, and legality checking. Uses bulk counting at depth 1 to
//! avoid redundant make/unmake at the leaves.

use crate::movegen;
use crate::position::Position;
use crate::types::{ArrayBuf, Move, MAX_MOVES};
use std::time::Instant;

/// Count all leaf nodes at `depth` in the move tree (bulk counting at depth 1).
pub fn perft(pos: &mut Position, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }

    let mut buf: ArrayBuf<Move, MAX_MOVES> = ArrayBuf::new();
    let count = movegen::generate_legal_moves(pos, &mut buf);

    // Bulk counting at depth 1
    if depth == 1 {
        return count as u64;
    }

    let mut nodes = 0u64;
    for i in 0..count {
        let m = buf[i];
        #[cfg(debug_assertions)]
        let key_before = pos.key;
        #[cfg(debug_assertions)]
        let pawn_key_before = pos.pawn_key;
        pos.make_move(m);
        nodes += perft(pos, depth - 1);
        pos.unmake_move(m);
        #[cfg(debug_assertions)] {
            debug_assert_eq!(pos.key, key_before,
                "perft: Zobrist key drift for {}", m.to_uci());
            debug_assert_eq!(pos.pawn_key, pawn_key_before,
                "perft: pawn_key drift for {}", m.to_uci());
        }
    }
    nodes
}

/// [Divide](https://www.chessprogramming.org/Perft#Divide): run perft split by
/// root move. Prints node count per move, useful for debugging move generation
/// by comparing with a reference engine (`go perft`).
pub fn divide(pos: &mut Position, depth: u32) {
    let start = Instant::now();

    let mut buf: ArrayBuf<Move, MAX_MOVES> = ArrayBuf::new();
    let count = movegen::generate_legal_moves(pos, &mut buf);

    let mut total = 0u64;
    for i in 0..count {
        let m = buf[i];
        pos.make_move(m);
        let nodes = if depth <= 1 { 1 } else { perft(pos, depth - 1) };
        pos.unmake_move(m);
        println!("{}: {}", m.to_uci(), nodes);
        total += nodes;
    }

    let elapsed = start.elapsed();
    let nps = if elapsed.as_secs_f64() > 0.0 {
        (total as f64 / elapsed.as_secs_f64()) as u64
    } else {
        0
    };
    println!();
    println!("Nodes: {}", total);
    println!("Time:  {:.3}s", elapsed.as_secs_f64());
    println!("NPS:   {}", nps);
}
