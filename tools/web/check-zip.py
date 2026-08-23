#!/usr/bin/env python3
"""Checks that the web archive is the one itch.io expects, and nothing more.

Run by the release workflow before the zip is uploaded, because everything that can go
wrong here goes wrong silently: a file left behind by an earlier shape of the build ships
without anyone noticing, a module that failed to link is still a file, and `index.html`
anywhere but the root gives a blank page rather than an error.

    python3 tools/web/check-zip.py var/web/gaiachess-web.zip
"""

import sys
import zipfile

# Exactly this, no more: `index.html` at the root is where itch looks, and the rest is
# what the page pulls in. `stoptest.html` is a bench and must not travel.
EXPECTED = {
    "index.html",
    "gl.js",
    "host.js",
    "worker.mjs",
    "engine.mjs",
    "gui.wasm",
    "engine.wasm",
    "net.bin.deflate",
    "tb34.gtpk",
    "favicon16.png",
    "favicon32.png",
    # gl.js is miniquad's, under MIT: its notice has to travel with it.
    "LICENSE-miniquad-MIT",
}

# A module that failed to link is still a file. Both are over a megabyte when whole.
MIN_MODULE_BYTES = 512 * 1024


def main(path: str) -> int:
    archive = zipfile.ZipFile(path)
    names = set(archive.namelist())

    missing = sorted(EXPECTED - names)
    unexpected = sorted(names - EXPECTED)
    if missing or unexpected:
        if missing:
            print(f"missing from the archive: {', '.join(missing)}", file=sys.stderr)
        if unexpected:
            print(f"should not be there: {', '.join(unexpected)}", file=sys.stderr)
        return 1

    for name in ("gui.wasm", "engine.wasm"):
        size = archive.getinfo(name).file_size
        if size < MIN_MODULE_BYTES:
            print(f"{name} is only {size} bytes — a stub, not a build", file=sys.stderr)
            return 1

    # The endgame tables: 35 compressed tables come to ~30 MB, so anything much
    # smaller is a truncated download, not the blob.
    size = archive.getinfo("tb34.gtpk").file_size
    if size < 20 * 1024 * 1024:
        print(f"tb34.gtpk is only {size} bytes — not the tables", file=sys.stderr)
        return 1

    total = sum(entry.compress_size for entry in archive.infolist())
    print(f"  {len(names)} files, {total / 1048576:.1f} MB compressed")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: check-zip.py <archive.zip>")
    sys.exit(main(sys.argv[1]))
