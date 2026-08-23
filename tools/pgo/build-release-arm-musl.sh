#!/bin/bash
# Build GaiaChess for ARM64 Linux, statically linked against musl.
# Target audience: Android chess GUIs (DroidFish, Chess for Android...) — Android's
# libc is Bionic, not glibc, so the regular gnu-linked neon binary does not start
# there. A static musl binary depends on the kernel alone and runs on any arm64
# Android device, no root required.
#
# Engine only: no board, no sound (--no-default-features), so ALSA is not linked
# and the binary is fully self-contained. Cross-compile from x86-64 Linux, no PGO
# (same build.rs SIGSEGV as the gnu ARM build).
#
# Usage: ./tools/pgo/build-release-arm-musl.sh [net.nnue] [extra-features]
#
# Prerequisites:
#   - rustup target add aarch64-unknown-linux-musl
#   - an aarch64 musl cross-gcc on PATH (aarch64-linux-musl-gcc, or the
#     Bootlin toolchain's aarch64-buildroot-linux-musl-gcc), or MUSL_CC=/path/to/gcc
#     Bootlin: https://toolchains.bootlin.com (aarch64--musl--stable-*)
set -euo pipefail

ARM_TARGET="aarch64-unknown-linux-musl"

cd "$(dirname "$0")/../.."
ROOT_DIR="$(pwd)"
source "./defaults.conf"
if [ -f ./defaults.local.conf ]; then source ./defaults.local.conf; fi
source "./tools/pgo/build-common.sh"

NET="${1:-$DEFAULT_NET}"
EXTRA_FEATURES="${2:-}"
# Headless feature set: the board would drag ALSA in, which cannot be linked
# statically in any useful way (alsa-lib dlopens its plugins).
FEATURES="nnue,syzygy,nalimov,gaiatb,online-tb"
if [ -n "$EXTRA_FEATURES" ]; then
    FEATURES="$FEATURES,$EXTRA_FEATURES"
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

# Find the musl cross-compiler (MUSL_CC overrides the search)
if [ -n "${MUSL_CC:-}" ]; then
    CROSS_CC="$MUSL_CC"
elif command -v aarch64-linux-musl-gcc &>/dev/null; then
    CROSS_CC="aarch64-linux-musl-gcc"
elif command -v aarch64-buildroot-linux-musl-gcc &>/dev/null; then
    CROSS_CC="aarch64-buildroot-linux-musl-gcc"
else
    echo "Error: aarch64 musl cross-compiler not found."
    echo "  Download a toolchain from https://toolchains.bootlin.com and either"
    echo "  add its bin/ to PATH or set MUSL_CC=/path/to/aarch64-...-musl-gcc"
    exit 1
fi

CARGO_EXE="target/$ARM_TARGET/release/gaiachess"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/gaiachess-${VERSION}-android-arm64}"

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "=== GaiaChess ARM64 musl (Android) Release Build ==="
echo "  Version:    $VERSION"
echo "  Target:     $ARM_TARGET"
echo "  Network:    $NET"
echo "  Features:   $FEATURES"
echo "  Cross-CC:   $CROSS_CC"
echo

# CC for zstd-sys (gaiatb); same gcc as linker driver.
export CC_aarch64_unknown_linux_musl="$CROSS_CC"
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_LINKER="$CROSS_CC"

# build.rs compiles Pyrrhic with clang (C11 atomics on MSVC), which cross-compiles
# fine but knows nothing of the musl toolchain's headers — without a sysroot it
# falls back to freestanding stubs where uintptr_t is 32-bit. cc-rs appends
# per-target CFLAGS to every compiler it drives, clang included.
SYSROOT="$("$CROSS_CC" -print-sysroot)"
export CFLAGS_aarch64_unknown_linux_musl="--sysroot=$SYSROOT"

# +crt-static is the default for *-musl targets: the result is fully static.
# No PGO for the same reason as the gnu ARM build (build.rs SIGSEGV).
echo "--- Build ARM64 musl static (no PGO) ---"
env MODEL="$NET" RUSTFLAGS="-C target-cpu=generic" \
    cargo build --release --target "$ARM_TARGET" --no-default-features \
    --features "$FEATURES" 2>&1

# A dynamically linked binary would defeat the whole point: fail loudly.
if readelf -l "$CARGO_EXE" | grep -q INTERP; then
    echo "Error: $CARGO_EXE requests a dynamic loader — not a static binary"
    exit 1
fi

cp "$CARGO_EXE" "$OUTPUT_DIR/gaiachess-android-neon"
echo "  => $OUTPUT_DIR/gaiachess-android-neon"
ls -lh "$OUTPUT_DIR/gaiachess-android-neon"

echo
echo "=== Done ==="
