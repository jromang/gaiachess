#!/bin/bash
# Release build of GaiaChess on Haiku (x86-64), run natively inside a Haiku
# install — there is no cross toolchain worth fighting for a tier-3 target.
#
# Usage: ./tools/pgo/build-haiku.sh [net.nnue]
#
# Output: OUTPUT_DIR/gaiachess-haiku, resources attached (rc + xres + mimeset),
# ready to run from Tracker or to be packaged by the HaikuPorts recipe.
#
# Prerequisites (pkgman install): rust_bin, haiku_devel, cmd:clang (pyrrhic
# builds with clang, see build.rs), devel:libzstd and cmd:pkg_config (libbe
# already depends on the system zstd, and a second, statically embedded copy
# collides at link time — hence ZSTD_SYS_USE_PKG_CONFIG below).
#
# The package target is x86-64-v2 rather than native: a binary in HaikuDepot
# runs on whatever machine pulls it, and v2 (SSE4.2/POPCNT) is the same floor
# the universal Windows build accepts. AVX2 is given up, as it is there.
set -euo pipefail

cd "$(dirname "$0")/../.."
source "./defaults.conf"
if [ -f ./defaults.local.conf ]; then source ./defaults.local.conf; fi

NET="${1:-$DEFAULT_NET}"
OUTPUT_DIR="${OUTPUT_DIR:-var/output}"
mkdir -p "$OUTPUT_DIR"

if [ "$(uname)" != "Haiku" ]; then
    echo "Error: this script builds natively on Haiku (uname says $(uname))." >&2
    exit 1
fi

echo "=== Building GaiaChess for Haiku (net: $NET) ==="
# The whole feature set builds and runs on Haiku (each proven one at a time,
# 2026-08-23) — this is the default set plus nnue, minus nothing.
MODEL="$NET" RUSTFLAGS="-C target-cpu=x86-64-v2" ZSTD_SYS_USE_PKG_CONFIG=1 \
    cargo build --release --no-default-features \
    --features "nnue,gui,syzygy,gaiatb,online-tb,nalimov,progress"

BIN="$OUTPUT_DIR/gaiachess-haiku"
cp target/release/gaiachess "$BIN"

# Attach the app resources: signature, version, launch flags. The Deskbar reads
# them off the file itself, and mimeset is what makes Tracker notice.
rc -o "$OUTPUT_DIR/gaiachess.rsrc" tools/pgo/gaiachess.rdef tools/pgo/gaiachess-icons.rdef
xres -o "$BIN" "$OUTPUT_DIR/gaiachess.rsrc"
rm -f "$OUTPUT_DIR/gaiachess.rsrc"
mimeset -f "$BIN"

echo "=== Done: $BIN ==="
"$BIN" --no-gui <<< "uci" | head -n 3 || true
