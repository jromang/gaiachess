//! Network file integrity: a 16-byte footer appended to the payload
//! (`NNUE_FILE_SIZE` bytes), without which a file is refused.
//!
//! Layout, little-endian, at the very end of the file:
//! `[arch_hash: u32][content_hash: u64][magic: u32 = "GN2\0"]`
//!
//! - `arch_hash` is an FNV-1a digest of the canonical architecture tuple
//!   (dimensions, quantization shifts, bucket and stack layouts, baked file
//!   permutation). The output scale is not part of it: the network carries its
//!   own in its metadata (`NNUEParams::meta`), checked here at load. The trainer
//!   computes it from its own constants when it exports a network; this module
//!   computes it from the engine's. A trainer/engine drift makes the two values
//!   diverge and the load fails loudly — the silent-wrong-weights class of
//!   accident this exists for.
//! - `content_hash` is an FNV-1a 64 digest of the payload bytes, catching
//!   corruption and truncation.
//!
//! Both hashes are taken over the FILE bytes, never the in-memory struct:
//! `repermute_ft` rearranges the feature transformer per SIMD tier at load,
//! so memory is not portable across machines.
//!
//! A file without the footer used to be accepted on its size alone, for the
//! networks published before the footer existed. Every published network of this
//! architecture carries one, and a size is all a file of noise needs to pass: its
//! weights then overflow the accumulator sums. The trainer writes the footer with
//! every export.

use super::features::KING_BUCKETS;
use super::network::{FILE_PERM, NNUE_FILE_SIZE};
use super::{
    A1_SIZE, FT_QUANT, FT_SHIFT, GATE_SHIFT, KING_BUCKET_COUNT, L1_OUT, L1_SHIFT, L1_SIZE,
    L1_SQR_SHIFT, L2_OUT, L2_SHIFT, L3_IN, PAWN_PAIR_SIZE, PSQ_PER_BLOCK, PSQ_SIZE, QUEEN_STATES,
    SKIP_NEG, SKIP_POS, STACK_SCHEME, STACKS, THREAT_INPUT_SIZE,
};

/// Size of the integrity footer appended to the payload.
pub const FOOTER_SIZE: usize = 16;

/// Footer magic, last 4 bytes of a footered file.
pub const MAGIC: u32 = u32::from_le_bytes(*b"GN2\0");

const FNV32_OFFSET: u32 = 0x811c_9dc5;
const FNV32_PRIME: u32 = 0x0100_0193;
const FNV64_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV64_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Fold one little-endian u32 into an FNV-1a 32 state.
pub(super) const fn h32(mut h: u32, v: u32) -> u32 {
    let bytes = v.to_le_bytes();
    let mut i = 0;
    while i < 4 {
        h ^= bytes[i] as u32;
        h = h.wrapping_mul(FNV32_PRIME);
        i += 1;
    }
    h
}

/// FNV-1a 64 over a byte slice (payload digest).
pub fn fnv1a64(data: &[u8]) -> u64 {
    let mut h = FNV64_OFFSET;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(FNV64_PRIME);
    }
    h
}

/// FNV-1a digest of the canonical architecture tuple, computed at compile
/// time from the engine's own constants. The trainer computes the same tuple
/// from ITS constants (tools/trainer8); the golden tests on both sides pin
/// the same literal, so a legitimate architecture change is a conscious
/// two-sided update while an accidental drift fails `cargo test` on the
/// drifting side.
///
/// Tuple, in order: PSQ_SIZE, THREAT_INPUT_SIZE, PAWN_PAIR_SIZE, L1_SIZE,
/// L1_OUT, A1_SIZE, L2_OUT, L3_IN, STACKS, KING_BUCKET_COUNT, QUEEN_STATES,
/// PSQ_PER_BLOCK, FT_QUANT, FT_SHIFT, L1_SHIFT, L1_SQR_SHIFT, L2_SHIFT,
/// GATE_SHIFT, SKIP_POS, SKIP_NEG, the 64 KING_BUCKETS entries, the 3
/// STACK_SCHEME entries, the 8 FILE_PERM entries. Each value hashed as a
/// little-endian u32.
pub const ARCH_HASH: u32 = {
    let mut h = FNV32_OFFSET;
    h = h32(h, PSQ_SIZE as u32);
    h = h32(h, THREAT_INPUT_SIZE as u32);
    h = h32(h, PAWN_PAIR_SIZE as u32);
    h = h32(h, L1_SIZE as u32);
    h = h32(h, L1_OUT as u32);
    h = h32(h, A1_SIZE as u32);
    h = h32(h, L2_OUT as u32);
    h = h32(h, L3_IN as u32);
    h = h32(h, STACKS as u32);
    h = h32(h, KING_BUCKET_COUNT as u32);
    h = h32(h, QUEEN_STATES as u32);
    h = h32(h, PSQ_PER_BLOCK as u32);
    h = h32(h, FT_QUANT as u32);
    h = h32(h, FT_SHIFT as u32);
    h = h32(h, L1_SHIFT as u32);
    h = h32(h, L1_SQR_SHIFT as u32);
    h = h32(h, L2_SHIFT as u32);
    h = h32(h, GATE_SHIFT as u32);
    h = h32(h, SKIP_POS as u32);
    h = h32(h, SKIP_NEG as u32);
    let mut i = 0;
    while i < KING_BUCKETS.len() {
        h = h32(h, KING_BUCKETS[i] as u32);
        i += 1;
    }
    let mut i = 0;
    while i < STACK_SCHEME.len() {
        h = h32(h, STACK_SCHEME[i] as u32);
        i += 1;
    }
    let mut i = 0;
    while i < FILE_PERM.len() {
        h = h32(h, FILE_PERM[i] as u32);
        i += 1;
    }
    h
};

/// Validate a network file image: the payload, then a footer whose two digests
/// check out. On success the payload to install is always `data[..NNUE_FILE_SIZE]`.
pub fn verify(data: &[u8]) -> Result<(), String> {
    let expected = NNUE_FILE_SIZE + FOOTER_SIZE;
    if data.len() != expected
        || u32::from_le_bytes(data[expected - 4..].try_into().unwrap()) != MAGIC
    {
        // The payload's size, give or take the padding a trainer used to leave.
        if (NNUE_FILE_SIZE..NNUE_FILE_SIZE + 64).contains(&data.len()) {
            return Err(String::from(
                "no integrity footer: not a network, or one exported before the footer existed",
            ));
        }
        return Err(format!(
            "File size {} != expected {expected} bytes",
            data.len()
        ));
    }
    let tail = &data[NNUE_FILE_SIZE..];
    let arch = u32::from_le_bytes(tail[0..4].try_into().unwrap());
    if arch != ARCH_HASH {
        return Err(format!(
            "wrong architecture: engine expects {ARCH_HASH:#010x}, network declares {arch:#010x}"
        ));
    }
    let content = u64::from_le_bytes(tail[4..12].try_into().unwrap());
    let actual = fnv1a64(&data[..NNUE_FILE_SIZE]);
    if content != actual {
        return Err(format!(
            "corrupted payload: footer declares {content:#018x}, file hashes to {actual:#018x}"
        ));
    }
    super::network::file_scale(data)?;
    Ok(())
}

/// The identity of a network's weights, for callers that need to name them.
///
/// The file carries the digest, so it is read rather than recomputed. The value
/// is a function of the FILE bytes, so two processes agree on it whatever SIMD
/// tier they resolved — which is what makes it usable as the name of a shared
/// image.
///
/// Only for data that [`verify`] has accepted.
pub fn content_hash(data: &[u8]) -> u64 {
    let tail = &data[data.len().saturating_sub(FOOTER_SIZE)..];
    declared_hash(data.len(), tail).expect("a verified network declares its digest")
}

/// The digest a footered file declares, from the file's length and its last
/// [`FOOTER_SIZE`] bytes alone; `None` for a file without one, which [`verify`]
/// refuses. Lets a caller name a network before reading it whole.
pub fn declared_hash(len: usize, tail: &[u8]) -> Option<u64> {
    if len != NNUE_FILE_SIZE + FOOTER_SIZE || tail.len() != FOOTER_SIZE {
        return None;
    }
    let magic = u32::from_le_bytes(tail[12..16].try_into().unwrap());
    (magic == MAGIC).then(|| u64::from_le_bytes(tail[4..12].try_into().unwrap()))
}

/// Build the 16-byte footer for a payload (test helper; the trainer has its
/// own mirror implementation in tools/trainer8).
pub fn footer(payload: &[u8]) -> [u8; FOOTER_SIZE] {
    debug_assert!(payload.len() == NNUE_FILE_SIZE);
    let mut f = [0u8; FOOTER_SIZE];
    f[0..4].copy_from_slice(&ARCH_HASH.to_le_bytes());
    f[4..12].copy_from_slice(&fnv1a64(payload).to_le_bytes());
    f[12..16].copy_from_slice(&MAGIC.to_le_bytes());
    f
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins the architecture digest. tools/trainer8 pins the SAME literal
    /// from its own constants: change one side's architecture and its golden
    /// test fails before any network is ever exported or loaded. A
    /// legitimate architecture change updates both literals consciously.
    #[test]
    fn the_architecture_digest_is_pinned_on_both_sides() {
        assert_eq!(ARCH_HASH, 0x9562_1179, "update tools/trainer8 in lockstep");
    }

    /// A payload of `NNUE_FILE_SIZE` patterned bytes with a believable output scale.
    fn patterned(mul: usize, add: usize) -> Vec<u8> {
        let mut data = vec![0u8; NNUE_FILE_SIZE];
        for (i, b) in data.iter_mut().enumerate() {
            *b = (i * mul + add) as u8;
        }
        let at = NNUE_FILE_SIZE - 64;
        data[at..at + 4].copy_from_slice(&287i32.to_le_bytes());
        data
    }

    #[test]
    fn a_network_without_a_believable_scale_is_refused() {
        let mut data = patterned(31, 7);
        let at = NNUE_FILE_SIZE - 64;
        data[at..at + 4].copy_from_slice(&0i32.to_le_bytes());
        let mut footered = data.clone();
        footered.extend_from_slice(&footer(&data));
        let err = verify(&footered).unwrap_err();
        assert!(err.contains("output scale"), "{err}");
    }

    #[test]
    fn a_footered_file_verifies_and_every_lie_is_caught() {
        // Small stand-in payload is not possible: verify() is anchored to
        // NNUE_FILE_SIZE. Build the real-size buffer once.
        let data = patterned(31, 7);
        let f = footer(&data);
        let mut footered = data.clone();
        footered.extend_from_slice(&f);
        assert_eq!(verify(&footered), Ok(()));

        // Corrupt one payload byte: content hash catches it.
        let mut corrupt = footered.clone();
        corrupt[12345] ^= 0x40;
        let err = verify(&corrupt).unwrap_err();
        assert!(err.contains("corrupted payload"), "{err}");

        // Wrong architecture hash: named as such, both values shown.
        let mut wrong_arch = footered.clone();
        wrong_arch[NNUE_FILE_SIZE] ^= 0xff;
        let err = verify(&wrong_arch).unwrap_err();
        assert!(err.contains("wrong architecture"), "{err}");

        // No footer: refused, and said so, padded or not; the 16 bytes of a footer
        // without its magic too.
        for pad in [0, 16, 63] {
            let mut bare = data.clone();
            bare.resize(NNUE_FILE_SIZE + pad, 0);
            let err = verify(&bare).unwrap_err();
            assert!(
                err.contains("no integrity footer"),
                "{pad} bytes past the payload: {err}"
            );
        }
        // Any other size: refused on its size.
        let mut long = footered.clone();
        long.resize(NNUE_FILE_SIZE + 64, 0);
        for wrong in [&data[..NNUE_FILE_SIZE - 1], &long[..], &data[..100]] {
            let err = verify(wrong).unwrap_err();
            assert!(err.contains("File size"), "{} bytes: {err}", wrong.len());
        }
    }

    /// The name read from the footer is the digest of the payload, which every
    /// process that reads the file computes alike.
    #[test]
    fn the_content_hash_is_the_payload_digest() {
        let data = patterned(17, 3);
        let mut footered = data.clone();
        footered.extend_from_slice(&footer(&data));
        assert_eq!(content_hash(&footered), fnv1a64(&data));
    }
}
