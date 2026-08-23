"""Packs web/dist into an archive itch.io will accept.

Run from the directory being packed; the archive path is the one argument. Python rather
than the `zip` command because that one is not everywhere, and this has to work from the
same shell on every machine that builds a release.
"""

import os
import sys
import zipfile

# A bench, not a page of the game.
SKIP = {"stoptest.html"}

archive = sys.argv[1]
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
    for name in sorted(os.listdir(".")):
        if name in SKIP or not os.path.isfile(name):
            continue
        # The weights and the tables are already compressed; deflating them again
        # costs time and gains nothing.
        stored = name.endswith((".gz", ".deflate", ".gtpk"))
        how = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
        z.write(name, name, compress_type=how)
