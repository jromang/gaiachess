#!/bin/bash
# Build GaiaChess for ARM64 (Linux).
# Cross-compile from x86-64 Linux. No PGO (build.rs SIGSEGV under PGO instrumentation).
#
# For macOS Apple Silicon: see .github/workflows/release.yml (native build on macOS runner).
# Cross-compiling to aarch64-apple-darwin requires the macOS SDK — not feasible on Linux.
#
# Usage: ./tools/pgo/build-release-arm.sh [net.nnue] [extra-features]
# Examples:
#   ./tools/pgo/build-release-arm.sh
#   ./tools/pgo/build-release-arm.sh nets/gen1.nnue
#
# Prerequisites:
#   - aarch64-linux-gnu-gcc (cross-compiler)
#   - rustup target add aarch64-unknown-linux-gnu
set -euo pipefail

ARM_TARGET="aarch64-unknown-linux-gnu"

cd "$(dirname "$0")/../.."
ROOT_DIR="$(pwd)"
source "./defaults.conf"
if [ -f ./defaults.local.conf ]; then source ./defaults.local.conf; fi
source "./tools/pgo/build-common.sh"

NET="${1:-$DEFAULT_NET}"
EXTRA_FEATURES="${2:-}"
FEATURES="nnue,syzygy,nalimov,gaiatb,online-tb"
if [ -n "$EXTRA_FEATURES" ]; then
    FEATURES="nnue,syzygy,nalimov,gaiatb,online-tb,$EXTRA_FEATURES"
fi

# --- Checks ---

if [ ! -f "$NET" ]; then
    echo "Error: network '$NET' not found"
    exit 1
fi

VERSION=$(read_cargo_version)

if ! rustup target list --installed | grep -q "$ARM_TARGET"; then
    echo "Error: target $ARM_TARGET missing. Install with:"
    echo "  rustup target add $ARM_TARGET"
    exit 1
fi

# Find the cross-compiler
if command -v aarch64-linux-gnu-gcc &>/dev/null; then
    CROSS_CC="aarch64-linux-gnu-gcc"
elif command -v aarch64-unknown-linux-gnu-gcc &>/dev/null; then
    CROSS_CC="aarch64-unknown-linux-gnu-gcc"
else
    echo "Error: ARM64 cross-compiler not found."
    echo "  Install with: sudo pacman -S aarch64-linux-gnu-gcc"
    exit 1
fi

CARGO_EXE="target/$ARM_TARGET/release/gaiachess"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/gaiachess-${VERSION}-linux-arm64}"

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "=== GaiaChess ARM64 Release Build ==="
echo "  Version:    $VERSION"
echo "  Target:     $ARM_TARGET"
echo "  Network:    $NET"
echo "  Features:   $FEATURES"
echo "  Cross-CC:   $CROSS_CC"
echo

# Environment variables for cross-compilation
# CC for Pyrrhic build (Syzygy C library) and other C deps
export CC_aarch64_unknown_linux_gnu="$CROSS_CC"
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER="$CROSS_CC"

# PGO disabled on ARM: build.rs SIGSEGV under PGO instrumentation (zstd compression),
# and CARGO_TARGET_*_RUSTFLAGS is ignored when RUSTFLAGS is set.
echo "--- Build ARM64 (no PGO) ---"
env MODEL="$NET" RUSTFLAGS="-C target-cpu=generic -C link-arg=-lgcc" \
    cargo build --release --target "$ARM_TARGET" --features "$FEATURES" 2>&1

cp "$CARGO_EXE" "$OUTPUT_DIR/gaiachess-linux-neon"
echo "  => $OUTPUT_DIR/gaiachess-linux-neon"
ls -lh "$OUTPUT_DIR/gaiachess-linux-neon"

echo
echo "=== Done ==="
