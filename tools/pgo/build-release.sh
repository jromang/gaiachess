#!/bin/bash
# Full release build of GaiaChess for Linux and Windows (x86-64).
# Builds all variants (PeSTO + NNUE) with PGO.
# Profiles on native Linux, cross-compiles Windows with the same profiles.
#
# Usage: ./tools/pgo/build-release.sh [--linux-only|--windows-only] [net.nnue] [extra-features]
# Examples:
#   ./tools/pgo/build-release.sh                           # Linux + Windows
#   ./tools/pgo/build-release.sh --linux-only              # Linux only
#   ./tools/pgo/build-release.sh nets/gen1.nnue spsa       # Custom net + SPSA
#
# Output: individual binaries in the output directory (OUTPUT_DIR)
#   Linux:   OUTPUT_DIR/gaiachess-linux-{suffix}
#   Windows: OUTPUT_DIR/gaiachess-windows-{suffix}.exe
#
# Prerequisites:
#   - llvm-profdata
#   - For Windows: mingw-w64-gcc, rustup target x86_64-pc-windows-gnu
set -euo pipefail

# --- Arguments ---
# The platform is decided by the host, not by a flag: PGO requires running the
# instrumented binary, and Rust profiles do not survive a change of target
# triple (see build-common.sh). --linux-only / --windows-only are kept as
# assertions — they state which platform the caller believes it is on, and fail
# loudly when that is wrong, rather than silently building the other one.
EXPECT_PLATFORM=""
case "${1:-}" in
    --linux-only)   EXPECT_PLATFORM=linux;   shift ;;
    --windows-only) EXPECT_PLATFORM=windows; shift ;;
esac

cd "$(dirname "$0")/../.."
ROOT_DIR="$(pwd)"
source "./defaults.conf"
if [ -f ./defaults.local.conf ]; then source ./defaults.local.conf; fi
source "./tools/pgo/build-common.sh"

NET="${1:-$DEFAULT_NET}"
EXTRA_FEATURES="${2:-}"

case "$HOST_TARGET" in
    *windows*) PLATFORM=windows ;;
    *)         PLATFORM=linux ;;
esac

if [ -n "$EXPECT_PLATFORM" ] && [ "$EXPECT_PLATFORM" != "$PLATFORM" ]; then
    echo "Error: --${EXPECT_PLATFORM}-only was requested but this host builds '$PLATFORM'." >&2
    echo "       Windows binaries must be built on a Windows runner: profiles" >&2
    echo "       collected on Linux do not apply to a Windows binary." >&2
    exit 1
fi

PGO_DIR="${PGO_DIR:-${TMPDIR:-/tmp}/gaiachess-pgo-release}"

# --- Checks ---

LLVM_PROFDATA=$(find_llvm_profdata) || exit 1
VERSION=$(read_cargo_version)

# Check NNUE network
if [ ! -f "$NET" ]; then
    echo "Error: network '$NET' not found"
    exit 1
fi

# --- Output directory ---
# Only clear this platform's binaries: the two platforms are now produced by two
# different jobs that may share an output directory.
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/var/output/gaiachess-${VERSION}}"
mkdir -p "$OUTPUT_DIR"
rm -f "$OUTPUT_DIR"/gaiachess-${PLATFORM}-*

# Add extra features if requested
augment_features() {
    local features="$1"
    if [ -n "$EXTRA_FEATURES" ] && [[ "$features" == *nnue* ]]; then
        echo "$features,$EXTRA_FEATURES"
    else
        echo "$features"
    fi
}

echo "=== GaiaChess Release Build ==="
echo "  Version:    $VERSION"
echo "  Network:    $NET"
echo "  Platform:   $PLATFORM (native, $HOST_TARGET)"
echo "  Variants:   ${#VARIANTS_ALL[@]} (${#VARIANTS_PESTO[@]} PeSTO + ${#VARIANTS_NNUE[@]} NNUE)"
echo "  PGO_SDE:    $PGO_SDE"
echo

TOTAL=${#VARIANTS_ALL[@]}
CURRENT=0
FAILED=()

for variant in "${VARIANTS_ALL[@]}"; do
    CURRENT=$((CURRENT + 1))
    parse_variant "$variant"
    V_FEATURES=$(augment_features "$V_FEATURES")
    variant_final="$V_SUFFIX|$V_CPU|$V_EXTRA|$V_FEATURES"

    echo
    echo "============================================================"
    echo "=== [$CURRENT/$TOTAL] $V_SUFFIX ==="
    echo "============================================================"

    # --- Build natively for this host ---
    if build_pgo_native "$variant_final" "$PGO_DIR"; then
        cp "$NATIVE_BIN" "$OUTPUT_DIR/gaiachess-${PLATFORM}-${V_SUFFIX}${EXE_SUFFIX}"
    else
        echo "FAILED: $V_SUFFIX ($PLATFORM)"
        FAILED+=("$V_SUFFIX-$PLATFORM")
    fi
done

# --- Windows only: console-subsystem engine for ChessBase GUIs ---
# ChessBase loaders (Fritz "Install UCI module") refuse to launch a
# GUI-subsystem executable: the process is never created, the dialog just
# stays empty (measured 2026-08-22). So Windows ships one extra binary built
# without the board (--no-default-features): same universal baseline, same
# runtime dispatch and PGO, linked as a console application. Universal only —
# anyone on a pre-2013 CPU running Fritz is on their own.
if [ "$PLATFORM" = windows ]; then
    console_features=$(augment_features "nnue,syzygy,nalimov,gaiatb,online-tb")
    console_variant="console|x86-64-v3|--cfg gaia_dist|$console_features"

    echo
    echo "============================================================"
    echo "=== [extra] console (ChessBase-compatible, universal) ==="
    echo "============================================================"

    if build_pgo_native "$console_variant" "$PGO_DIR" --no-default-features; then
        cp "$NATIVE_BIN" "$OUTPUT_DIR/gaiachess-${PLATFORM}-console${EXE_SUFFIX}"
    else
        echo "FAILED: console ($PLATFORM)"
        FAILED+=("console-$PLATFORM")
    fi
fi

# --- Results ---
echo
echo "=== Produced binaries ==="
ls -lh "$OUTPUT_DIR"/gaiachess-*
echo

# --- Duplicate check ---
# Two variants producing byte-identical binaries means one of them was not
# actually rebuilt. This is how the 4.2.0 release shipped `avx512` as a copy of
# `bmi2`: the build had failed, and the stale binary was copied under the new
# name. Distinct target-cpu/target-feature settings must yield distinct output.
echo "=== Checking for duplicate binaries ==="
DUPES=$(md5sum "$OUTPUT_DIR"/gaiachess-* | sort | uniq -Dw32)
if [ -n "$DUPES" ]; then
    echo "ERROR: identical binaries produced for different variants:"
    echo "$DUPES"
    echo "A variant was not rebuilt — check the build log for a failure above."
    exit 1
fi
echo "  all binaries distinct"
echo

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "=== WARNING: ${#FAILED[@]} variant(s) failed ==="
    for f in "${FAILED[@]}"; do
        echo "  - $f"
    done
    exit 1
else
    echo "=== Done: all variants built in $OUTPUT_DIR ==="
fi
