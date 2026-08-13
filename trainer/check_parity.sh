#!/usr/bin/env bash
# Threat-feature parity check: trainer vs engine over a FEN suite.
# Usage: ./check_parity.sh [path_to_gaiachess_binary]
set -u

ENGINE="${1:-../../target/release/gaiachess}"
cd "$(dirname "$0")"

FENS=(
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1"
    "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQ1RK1 w - - 6 6"
    "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQ1RK1 b - - 6 6"
    "1k1r3r/ppq2ppp/2pb1n2/8/3P4/2N1PN2/PP3PPP/1KR2B1R w - - 4 15"
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 0 1"
    "8/5pk1/4p1p1/3p3p/3P3P/4P1P1/4KP2/8 w - - 0 40"
    "8/8/8/4k3/8/4P3/4K3/8 w - - 0 60"
    "8/8/3k4/8/8/2Q1K3/8/8 b - - 0 50"
    "7k/8/5q2/8/8/2Q5/8/K7 w - - 0 30"
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 b - - 0 10"
)

CUDARC_CUDA_VERSION=13000 cargo build --release --bin dump_threats 2>/dev/null || { echo "build dump_threats FAILED"; exit 1; }

pass=0
fail=0
for fen in "${FENS[@]}"; do
    eng=$("$ENGINE" threats "$fen")
    trn=$(./target/release/dump_threats "$fen")
    if [ "$eng" == "$trn" ]; then
        pass=$((pass+1))
    else
        fail=$((fail+1))
        echo "MISMATCH: $fen"
        echo "  engine:  $eng" | head -c 300; echo
        echo "  trainer: $trn" | head -c 300; echo
    fi
done

echo "Threat parity: $pass OK, $fail FAIL over ${#FENS[@]} FENs"
[ "$fail" -eq 0 ]
