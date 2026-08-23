#!/usr/bin/env bash
# Builds the WebAssembly artefacts.
#
# Flags go through RUSTFLAGS rather than a .cargo/config.toml: the two web modules do not
# want the same ones, and a config file would apply to both. Beware that RUSTFLAGS in the
# environment *replaces* build.rustflags entirely rather than adding to it.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source defaults.conf
if [ -f defaults.local.conf ]; then source defaults.local.conf; fi

TARGET=wasm32-unknown-unknown
OUT="$ROOT/web/dist"
# Emptied rather than written over: a file left from an earlier shape of the build
# would ship in the zip and nobody would notice.
rm -rf "$OUT"
mkdir -p "$OUT"

# 8 MB of stack. Measured, not guessed: 1 MB overflows building the first ThreadData,
# 2 MB carries a depth-22 search, and the margin over that is cheap next to the rest of
# the footprint. The native build asks for 32 MB, but that figure is about PGO inlining,
# which does not apply here.
STACK=$((8 * 1024 * 1024))

# No relaxed-simd: its operations are specified as non-deterministic, and a rung has to
# be the same opponent on every machine. See src/nnue/simd/wasm128.rs.
FLAGS="-C target-feature=+simd128 -C link-arg=--allow-undefined -C link-arg=-zstack-size=$STACK"

echo "== engine =="
# Built with MODEL unset, so no weights are baked in: they arrive over the network and
# are written straight into the module. Embedding them too would ship them twice.
env -u MODEL RUSTFLAGS="$FLAGS" cargo build --release --target "$TARGET" -p gaia-web-engine

cp "target/$TARGET/release/gaia_web_engine.wasm" "$OUT/engine.wasm"

# The interface carries neither the network nor the search, which is the whole point of
# building two modules: the menu can be on screen while the weights are still arriving.
# MODEL is unset for it so that no network is baked in.
echo "== interface =="
env -u MODEL RUSTFLAGS="$FLAGS"     cargo build --release --target "$TARGET" -p gaia-web-gui
cp "target/$TARGET/release/gaia-web-gui.wasm" "$OUT/gui.wasm"

# The page and its glue. gl.js is miniquad's own loader, vendored — see gl.js.README.
echo "== page =="
cp web/index.html web/host.js web/gl.js web/worker.mjs web/favicon16.png web/favicon32.png "$OUT/"
# gl.js is miniquad's, under MIT, which asks that its notice travel with every copy.
cp web/LICENSE-miniquad-MIT "$OUT/"
# The stop bench, for tools/web/stop-test.mjs. A kilobyte, and the only way to see
# whether a search can be cut short — but not part of a release zip.
cp web/stoptest.html "$OUT/"
cp web/engine/engine.mjs "$OUT/"

# The weights travel beside the modules rather than inside them: decompressing inside the
# module costs ~129 MB of linear memory that a browser never gives back.
#
# Compressed rather than raw, and decompressed by the browser's own DecompressionStream
# — native code, no wasm memory, and 22.5 MB on the wire instead of 36.6. Doing it here
# rather than trusting the host to compress in transit: what itch.io serves is not ours
# to decide, and 14 MB is too much to leave to chance.
#
# Raw deflate rather than gzip, which is also why the file is not named .gz: itch.io reads
# the *content* of what it serves, and "if the content of the file is detected as gzip
# compressed, the content-encoding header [is set] to gzip"
# (https://itch.io/docs/creators/html5). The weights would then be unwrapped once in
# transport and once again here, and in practice the transport decode fails outright —
# the browser reports ERR_CONTENT_DECODING_FAILED, which reaches fetch() as nothing more
# useful than "Failed to fetch". A raw deflate stream carries no magic number to find, so
# it arrives as the bytes that were sent. Same size, same native decompression.
#
# Compressed by gzip and then unwrapped, rather than deflated by Python's zlib directly:
# what sits inside a .gz already *is* a raw deflate stream, and GNU gzip finds 1.6 % more
# in it at -9 than zlib does (23.55 MB against 23.93 on this network). 370 KB is worth
# the dozen lines it takes to cut off a header and a trailer.
NET="${MODEL:-$DEFAULT_NET}"
gzip -9 -c < "$NET" > "$OUT/net.tmp.gz"
python3 - "$OUT/net.tmp.gz" "$OUT/net.bin.deflate" <<'DEFLATE'
import struct, sys

# RFC 1952: ten fixed bytes, then whatever optional fields the FLG byte announces, then
# the deflate stream, then eight bytes of checksum and length. Reading its input from a
# pipe, gzip writes neither a file name nor an extra field — but the flags are read
# rather than assumed, because another gzip on another machine might.
source, target = sys.argv[1], sys.argv[2]
raw = open(source, "rb").read()
assert raw[:3] == bytes((0x1F, 0x8B, 0x08)), "not a deflate-compressed gzip stream"
flags, at = raw[3], 10
if flags & 4:                                    # FEXTRA
    at += 2 + struct.unpack_from("<H", raw, at)[0]
for field in (8, 16):                            # FNAME, FCOMMENT: NUL-terminated
    if flags & field:
        at = raw.index(0, at) + 1          # bytes.index takes the byte value
if flags & 2:                                    # FHCRC
    at += 2
open(target, "wb").write(raw[at:-8])
DEFLATE
rm -f "$OUT/net.tmp.gz"

# The endgame tables travel beside the modules for the same reason the weights do — the
# worker embeds nothing on wasm — but exactly as they ship: the blob is zstd inside,
# which the module decompresses itself, and its magic is not gzip's, so no host will
# take it for something to unwrap in transport. Fetched from HuggingFace when absent,
# from the same address build.rs uses for the native embed.
TB_BLOB="${GAIATB_BLOB:-tb/tb34.gtpk}"
if [ ! -f "$TB_BLOB" ]; then
    echo "== tables: downloading tb34.gtpk =="
    mkdir -p "$(dirname "$TB_BLOB")"
    curl -L --fail --progress-bar -o "$TB_BLOB" \
        "https://huggingface.co/datasets/jromanghf/gaiatb-tb34/resolve/main/tb34.gtpk?download=true"
fi
cp "$TB_BLOB" "$OUT/tb34.gtpk"

ls -l "$OUT" | tail -n +2 | awk '{printf "  %-16s %6.1f MB\n", $NF, $5/1048576}'

# -- Zip for itch.io ---------------------------------------------------------
# `--zip` only: building and publishing are separate acts, and the zip is an artefact,
# so it lands under var/ like every other intermediate rather than beside the sources.
if [ "${1-}" = "--zip" ]; then
    ZIP="$ROOT/var/web/gaiachess-web.zip"
    mkdir -p "$(dirname "$ZIP")"
    rm -f "$ZIP"
    # index.html has to sit at the root of the archive: that is where itch looks.
    # stoptest.html is a bench, not a page of the game, and stays behind.
    ( cd "$OUT" && python3 "$ROOT/tools/web/zip.py" "$ZIP" )
    echo
    echo "  zip: $ZIP  ($(du -h "$ZIP" | cut -f1))"
    echo "  itch.io: embed 528x630, and tick SharedArrayBuffer support so a search"
    echo "           can be cut short rather than merely disowned."
fi
