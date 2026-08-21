#!/bin/bash
# Verify the runtime dispatch of freshly built distribution binaries.
#
# The kernels are monomorphized per SIMD tier and pinned by #[target_feature],
# so one tier must count the same bench nodes from either binary (universal or
# compat, whatever their baseline codegen), and the attack-path election
# (PEXT vs AVX2/magic) must not change the tree at all. Any disagreement means
# a dispatch or weight-permutation bug — the kind that would otherwise ship as
# a silently wrong evaluation. The AVX-512 tier is checked under Intel SDE so
# the result does not depend on which CPU the CI runner happened to draw.
#
# Usage: verify-dispatch.sh UNIVERSAL_BIN COMPAT_BIN
set -euo pipefail

U="$1"
C="$2"

cd "$(dirname "$0")/../.."
ROOT_DIR="$(pwd)"
source "./tools/pgo/build-common.sh"

DEPTH="${VERIFY_BENCH_DEPTH:-13}"

nodes() { "$@" 2>/dev/null | grep -E '^[0-9]+ nodes' | awk '{print $1}'; }

echo "=== Verifying runtime dispatch (bench depth $DEPTH) ==="

a=$(nodes env GAIA_SIMD=avx2 "$U" bench --depth "$DEPTH")
b=$(nodes env GAIA_SIMD=avx2 "$C" bench --depth "$DEPTH")
p=$(nodes env GAIA_SIMD=avx2 GAIA_PEXT=0 "$U" bench --depth "$DEPTH")
s=$(nodes env GAIA_SIMD=scalar "$C" bench --depth "$DEPTH")
echo "  avx2 tier:   universal=$a compat=$b universal(GAIA_PEXT=0)=$p"
echo "  scalar tier: compat=$s"
if [ -z "$a" ] || [ "$a" != "$b" ] || [ "$a" != "$p" ]; then
    echo "ERROR: avx2-tier node counts disagree between binaries or attack paths" >&2
    exit 1
fi
if [ -z "$s" ]; then
    echo "ERROR: the scalar tier produced no node count" >&2
    exit 1
fi

# The AVX-512 tier: natively when this machine resolves it (the binary is
# asked, same trick as the PGO profiling), under Intel SDE otherwise.
if GAIA_SIMD=vnni512 "$U" info 2>/dev/null | grep -qE "^SIMD: +vnni512 "; then
    v=$(nodes env GAIA_SIMD=vnni512 "$U" bench --depth "$DEPTH")
    w=$(nodes env GAIA_SIMD=vnni512 "$C" bench --depth "$DEPTH")
    echo "  vnni512 tier (native): universal=$v compat=$w"
else
    SDE="$(find_sde || true)"
    if [ -z "$SDE" ]; then
        if [ "${PGO_SDE:-auto}" = "require" ]; then
            echo "ERROR: Intel SDE is required to verify the AVX-512 tiers here" >&2
            exit 1
        fi
        echo "  no Intel SDE and no native AVX-512: skipping that tier's check"
        exit 0
    fi
    v=$(nodes env GAIA_SIMD=vnni512 "$SDE" -future -- "$U" bench --depth "$DEPTH")
    w=$(nodes env GAIA_SIMD=vnni512 "$SDE" -future -- "$C" bench --depth "$DEPTH")
    echo "  vnni512 tier (SDE): universal=$v compat=$w"
fi
if [ -z "$v" ] || [ "$v" != "$w" ]; then
    echo "ERROR: vnni512-tier node counts disagree between the two binaries" >&2
    exit 1
fi

echo "=== Runtime dispatch verified ==="
