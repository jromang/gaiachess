#!/bin/bash
# Build GaiaChess for Windows by cross-compiling from Linux — WITHOUT PGO.
#
# There is deliberately no PGO here. Rust mangled names embed a crate hash
# derived from -C metadata, which includes the target triple, so profiles
# collected on Linux match nothing in a Windows binary and -Cprofile-use is a
# silent no-op. (Measured: the same 86 function paths on both sides, zero
# overlap of mangled names.) That assumption holds in C++ (where
# cross-profiling is common practice), but not in Rust.
#
# The released Windows binaries are built and profiled natively on a Windows
# runner by .github/workflows/release.yml. This script is a local convenience
# for producing a quick cross-compiled zip.
# Produces 7 NNUE executables (avx2, pext, avx512, avx512vnni, znver3/4/5)
# + 1 PeSTO (older platforms without AVX2).
# Everything is packaged into gaiachess-VERSION-w64.zip with the README.
#
# Prerequisites: mingw-w64-gcc, llvm-profdata, rustup target x86_64-pc-windows-gnu, zip
#
# Usage: ./tools/pgo/build-pgo-windows.sh [net.nnue] [extra-features]
# Examples:
#   ./tools/pgo/build-pgo-windows.sh
#   ./tools/pgo/build-pgo-windows.sh nets/gen1-sb600.nnue spsa
# Output: gaiachess-VERSION-w64.zip
set -euo pipefail

WIN_TARGET="x86_64-pc-windows-gnu"
EXTRA_FEATURES="${2:-}"
PGO_DIR="/tmp/gaiachess-pgo-windows"

cd "$(dirname "$0")/../.."
ROOT_DIR="$(pwd)"
source "./defaults.conf"
if [ -f ./defaults.local.conf ]; then source ./defaults.local.conf; fi
source "./tools/pgo/build-common.sh"

NET="${1:-$DEFAULT_NET}"

# --- Checks ---

if [ ! -f "$NET" ]; then
    echo "Error: network '$NET' not found"
    exit 1
fi

if ! rustup target list --installed | grep -q "$WIN_TARGET"; then
    echo "Error: target $WIN_TARGET missing. Install with:"
    echo "  rustup target add $WIN_TARGET"
    exit 1
fi

LLVM_PROFDATA=$(find_llvm_profdata) || exit 1

if ! command -v zip &>/dev/null; then
    echo "Error: zip not found. Install with: sudo pacman -S zip"
    exit 1
fi

VERSION=$(read_cargo_version)
CARGO_EXE="target/$WIN_TARGET/release/gaiachess.exe"
ZIP_NAME="gaiachess-${VERSION}-w64"
ZIP_DIR="/tmp/$ZIP_NAME"

# Add extra features if requested
add_extra_features() {
    local features="$1"
    if [ -n "$EXTRA_FEATURES" ] && [[ "$features" == *nnue* ]]; then
        echo "$features,$EXTRA_FEATURES"
    else
        echo "$features"
    fi
}

echo "=== GaiaChess PGO Windows Build ==="
echo "  Version:  $VERSION"
echo "  Target:   $WIN_TARGET"
echo "  Network:  $NET"
echo "  LTO:      fat"
echo "  PGO:      none (cross-compiled; see header)"
echo

rm -rf "$ZIP_DIR"
mkdir -p "$ZIP_DIR"

# --- Build all NNUE variants ---
for variant in "${VARIANTS_NNUE[@]}"; do
    parse_variant "$variant"
    # Add extra features if needed
    original_features="$V_FEATURES"
    V_FEATURES=$(add_extra_features "$V_FEATURES")
    variant_with_extras="$V_SUFFIX|$V_CPU|$V_EXTRA|$V_FEATURES"

    build_nopgo_windows_cross "$variant_with_extras" "$WIN_TARGET"
    cp "$CARGO_EXE" "$ZIP_DIR/gaiachess-${V_SUFFIX}.exe"
done

# --- Build PeSTO variant (baseline without NNUE) ---
# Single PeSTO variant: x86-64 baseline
pesto_variant="x86-64|x86-64||syzygy"
build_nopgo_windows_cross "$pesto_variant" "$WIN_TARGET"
cp "$CARGO_EXE" "$ZIP_DIR/gaiachess-pesto.exe"

# --- Package as zip ---
echo "=== Packaging ==="
cp README.md "$ZIP_DIR/"

cd /tmp
zip -j "${ZIP_NAME}.zip" "$ZIP_DIR"/*
mv "${ZIP_NAME}.zip" "$OLDPWD/"
cd "$OLDPWD"

echo
echo "=== Done ==="
echo "  Archive: ${ZIP_NAME}.zip"
ls -lh "${ZIP_NAME}.zip"
echo
echo "  Contents:"
unzip -l "${ZIP_NAME}.zip"
