#!/usr/bin/env bash
# Publishes a released version to itch.io with butler.
#
# The release workflow calls this once every build job has succeeded, so itch.io never
# receives half a release. It is a script rather than inline YAML because the only way to
# rehearse it is to run it by hand: this CI cannot be exercised locally (SDE, MSVC, macOS
# runners), and a publishing step that has never been run before it publishes is a bad
# bet.
#
#   tools/itch/push.sh v4.2.3 [--repo owner/name] [--target user/game] [--dry-run]
#
# --dry-run stops after staging and prints what would be pushed, so the tree can be read
# before anything leaves the machine.
#
# Two constraints of butler shape everything below:
#
#   1. It takes a directory or a .zip, never a lone file. Each native binary is therefore
#      staged in a directory -- which is also where its licence and its readme belong.
#   2. "Playable in browser" is NOT inferred from the channel name, unlike win/linux/osx
#      which are. It is ticked once on the itch.io Edit game page, along with the embed
#      size and SharedArrayBuffer support. Those settings live on the upload, hence on
#      the channel: renaming a channel creates a fresh upload and loses them. So channels
#      are named after the platform alone, never after the CPU variant, and the variant
#      can change later without the page being rebuilt.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

TAG=""
SOURCE_REPO="jromang/gaiachess"
TARGET="jromang/gaiachess"
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --repo)    SOURCE_REPO="$2"; shift 2 ;;
        --target)  TARGET="$2";      shift 2 ;;
        --dry-run) DRY_RUN=1;        shift ;;
        -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
        -*)        echo "unknown option: $1" >&2; exit 1 ;;
        *)         TAG="$1";         shift ;;
    esac
done

if [ -z "$TAG" ]; then
    echo "usage: $0 <tag> [--repo owner/name] [--target user/game] [--dry-run]" >&2
    exit 1
fi

# The tag is the version: export-public.sh refuses to export a tag that disagrees with
# Cargo.toml, so the release this names was built from a tree that said the same.
VERSION="${TAG#v}"

if [ "$DRY_RUN" = 0 ] && [ -z "${BUTLER_API_KEY:-}" ]; then
    echo "ERROR: BUTLER_API_KEY is not set." >&2
    echo "  Local: the wharf key from https://itch.io/user/settings/api-keys" >&2
    echo "  CI:    an Actions secret on the repository that publishes" >&2
    exit 1
fi

# Everything intermediate lives under var/, which is gitignored wholesale. The downloads
# are keyed by tag: an asset kept from a previous run is reused, which makes a retry cheap
# on 170 MB of binaries, but only ever within the version it belongs to. A flat directory
# would quietly republish the last release under the new one's number.
DOWNLOAD="$ROOT/var/itch/download/$TAG"
STAGE="$ROOT/var/itch/stage"
mkdir -p "$DOWNLOAD"
# Staged from scratch every time: a binary left over from an earlier version would be
# published without anyone noticing.
rm -rf "$STAGE"

WEB_ZIP="gaiachess-web.zip"
WINDOWS_BIN="gaiachess-windows-universal.exe"
LINUX_BIN="gaiachess-linux-universal"
MACOS_BIN="gaiachess-macos-apple-silicon"

# The universal binary picks its SIMD paths at runtime — one download per platform is
# exactly what an itch.io page offers, and every CPU since 2013 gets its fastest paths
# (AVX-512+VNNI where present, PEXT skipped on the Zen 1/2 that microcode it).
echo "== fetching release assets from $SOURCE_REPO $TAG =="
for asset in "$WEB_ZIP" "$WINDOWS_BIN" "$LINUX_BIN" "$MACOS_BIN"; do
    if [ -s "$DOWNLOAD/$asset" ]; then
        echo "  have  $asset"
        continue
    fi
    echo "  fetch $asset"
    gh release download "$TAG" --repo "$SOURCE_REPO" --pattern "$asset" --dir "$DOWNLOAD"
done

# --- Staging ----------------------------------------------------------------

# gh release download does not carry the executable bit over; butler does carry it, and
# takes it from the filesystem. So a push made from a filesystem that has no such bit --
# a Windows checkout, most obviously -- would upload a Linux binary and a .app that
# nothing can launch, and butler would report success. Checked rather than assumed, and
# fatal outside a dry run: an unlaunchable download is worse than no download.
make_executable() { # make_executable <file>
    chmod +x "$1"
    [ -x "$1" ] && return
    echo "WARNING: the executable bit does not stick on this filesystem ($1)." >&2
    [ "$DRY_RUN" = 1 ] && return
    echo "ERROR: refusing to publish a binary that nothing can launch." >&2
    echo "  Push from Linux or macOS, or let the release workflow do it." >&2
    exit 1
}

# One readme per platform, because what has to be said differs on each: the sound library
# on Linux, Gatekeeper on macOS. What they share is which build this is and where the
# source lives -- the GPL asks for the second, and a bug report needs the first.
readme() { # readme <dir> <platform lines...>
    local dir="$1"; shift
    {
        echo "GaiaChess $VERSION"
        echo
        local line
        for line in "$@"; do echo "$line"; done
        echo
        echo "Run it with nothing after it and the board opens. Run it from a chess"
        echo "interface that speaks UCI -- Arena, Cute Chess, En Croissant -- and it"
        echo "answers as an engine instead, with the same twenty levels under the"
        echo "Skill Level option. Passing --no-gui skips the wait and speaks UCI"
        echo "straight away."
        echo
        echo "Free software: GNU General Public Licence version 3 or later, and it comes"
        echo "with no warranty. The full text is in LICENSE, beside this file."
        echo "Source, other builds, and the browser version:"
        echo "  https://github.com/jromang/gaiachess"
    } > "$dir/README.txt"
    cp "$ROOT/LICENSE" "$dir/LICENSE"
}

echo "== staging =="

mkdir -p "$STAGE/windows"
cp "$DOWNLOAD/$WINDOWS_BIN" "$STAGE/windows/gaiachess.exe"
readme "$STAGE/windows" \
    "Windows 64-bit, AVX2 build, profile-guided. Wants a CPU from 2013 or later" \
    "(Intel Haswell, AMD Zen). Self-contained: there is nothing else to install."

mkdir -p "$STAGE/linux"
cp "$DOWNLOAD/$LINUX_BIN" "$STAGE/linux/gaiachess"
make_executable "$STAGE/linux/gaiachess"
readme "$STAGE/linux" \
    "Linux 64-bit, AVX2 build, profile-guided. Wants a CPU from 2013 or later" \
    "(Intel Haswell, AMD Zen)." \
    "" \
    "It carries the sound of the interface, so it needs libasound2 (alsa-lib on Arch)." \
    "Any desktop has it already; minimal server images and containers often do not."

# A bare Mach-O binary cannot be launched from the Finder, nor by the itch.io app, so a
# macOS upload would be decorative without a bundle around it. This is the smallest
# bundle that works.
APP="$STAGE/macos/GaiaChess.app"
mkdir -p "$APP/Contents/MacOS"
cp "$DOWNLOAD/$MACOS_BIN" "$APP/Contents/MacOS/gaiachess"
make_executable "$APP/Contents/MacOS/gaiachess"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>               <string>GaiaChess</string>
    <key>CFBundleDisplayName</key>        <string>GaiaChess</string>
    <key>CFBundleIdentifier</key>         <string>io.github.jromang.gaiachess</string>
    <key>CFBundleExecutable</key>         <string>gaiachess</string>
    <key>CFBundlePackageType</key>        <string>APPL</string>
    <key>CFBundleVersion</key>            <string>$VERSION</string>
    <key>CFBundleShortVersionString</key> <string>$VERSION</string>
    <key>LSMinimumSystemVersion</key>     <string>11.0</string>
    <key>LSApplicationCategoryType</key>  <string>public.app-category.board-games</string>
    <key>NSHighResolutionCapable</key>    <true/>
</dict>
</plist>
PLIST
readme "$STAGE/macos" \
    "macOS on Apple Silicon (M1 and later). Intel Macs are not covered by this build." \
    "" \
    "The application is signed by nobody and notarised by nobody, so the first launch" \
    "has to be a right-click (or control-click) on GaiaChess.app followed by Open," \
    "which offers the button that a double-click does not. It opens normally after" \
    "that." \
    "" \
    "The engine is inside the bundle, at GaiaChess.app/Contents/MacOS/gaiachess --" \
    "that is the path to hand a chess interface that speaks UCI."

find "$STAGE" -mindepth 1 | sed "s|^$ROOT/|  |" | sort

# --- Publishing -------------------------------------------------------------

# The web package is pushed as the zip it already is: butler unpacks a .zip target, and
# index.html has to sit at the root of the upload -- which tools/web/check-zip.py is
# there to guarantee.
push() { # push <path> <channel>
    if [ "$DRY_RUN" = 1 ]; then
        echo "  (dry run) butler push <staged> $TARGET:$2 --userversion $VERSION"
        return
    fi
    echo "  $TARGET:$2"
    butler push "$1" "$TARGET:$2" --userversion "$VERSION"
}

echo "== pushing to $TARGET =="
push "$DOWNLOAD/$WEB_ZIP" html5
push "$STAGE/windows"     windows
push "$STAGE/linux"       linux
push "$STAGE/macos"       osx-arm64

if [ "$DRY_RUN" = 1 ]; then
    echo
    echo "dry run: nothing was uploaded."
    exit 0
fi

echo
butler status "$TARGET"
echo
echo "https://${TARGET%%/*}.itch.io/${TARGET##*/}"
