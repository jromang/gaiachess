//! Trainer ↔ engine feature parity: prints the threat feature indices
//! (sorted, per perspective) for a FEN, in the same format as the engine
//! command `gaiachess threats "<fen>"`.
//!
//! Usage:
//!   cargo run --release --bin dump_threats -- "<fen>"
//!
//! Automated comparison: tools/trainer5/check_parity.sh

#[path = "../threats.rs"]
mod threats;

use bulletformat::ChessBoard;

fn main() {
    let fen = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    if fen.is_empty() {
        eprintln!("Usage: dump_threats <fen>");
        std::process::exit(1);
    }

    let board: ChessBoard = format!("{fen} | 0 | 0.5")
        .parse()
        .expect("invalid FEN");

    let (stm, ntm) = threats::collect_threats(&board);

    let fmt = |v: &[usize]| v.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(",");
    println!("stm:{}", fmt(&stm));
    println!("ntm:{}", fmt(&ntm));
}
