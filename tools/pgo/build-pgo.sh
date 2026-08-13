#!/bin/bash
# Build GaiaChess with PGO (Profile-Guided Optimization).
# Usage: ./tools/pgo/build-pgo.sh [net.nnue] [extra-features]
# Examples:
#   ./tools/pgo/build-pgo.sh                                    # NNUE only
#   ./tools/pgo/build-pgo.sh nets/my-net.nnue spsa               # NNUE + SPSA
# Output: target/release/gaiachess (PGO-optimized)
set -euo pipefail

cd "$(dirname "$0")/../.."
source "./defaults.conf"
if [ -f ./defaults.local.conf ]; then source ./defaults.local.conf; fi

NET="${1:-$DEFAULT_NET}"
EXTRA_FEATURES="${2:-}"
PGO_DIR="/tmp/gaiachess-pgo-data"
BENCH_DEPTH=16

FEATURES="nnue,syzygy,nalimov,gaiatb,online-tb"
if [ -n "$EXTRA_FEATURES" ]; then
    FEATURES="nnue,syzygy,nalimov,gaiatb,online-tb,$EXTRA_FEATURES"
fi

if [ ! -f "$NET" ]; then
    echo "Error: network file '$NET' not found"
    exit 1
fi

echo "=== GaiaChess PGO Build ==="
echo "  Network: $NET"
echo "  Features: $FEATURES"
echo "  Bench depth: $BENCH_DEPTH"
echo

# Step 1: Instrumented build
echo "--- Step 1/4: Building instrumented binary ---"
rm -rf "$PGO_DIR"
cargo clean -q
MODEL="$NET" RUSTFLAGS="-C target-cpu=native -Cprofile-generate=$PGO_DIR" \
    cargo build --release --features "$FEATURES" 2>&1

# Step 2: Collect profiles
echo "--- Step 2/4: Collecting profiles (bench depth $BENCH_DEPTH) ---"
./target/release/gaiachess bench
echo

# Step 3: Merge profiles
echo "--- Step 3/4: Merging profiles ---"
llvm-profdata merge -o "$PGO_DIR/merged.profdata" "$PGO_DIR/"
echo "  Merged $(ls "$PGO_DIR"/*.profraw 2>/dev/null | wc -l) profile files"

# Step 4: Optimized rebuild
echo "--- Step 4/4: Building PGO-optimized binary ---"
cargo clean -q
MODEL="$NET" RUSTFLAGS="-C target-cpu=native -Cprofile-use=$PGO_DIR/merged.profdata" \
    cargo build --release --features "$FEATURES" 2>&1

echo
echo "=== Done ==="
echo "  Binary: target/release/gaiachess"
echo "  Run './target/release/gaiachess bench' to verify"
