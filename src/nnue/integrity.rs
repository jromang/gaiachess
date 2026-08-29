//! Network file integrity: an optional 16-byte footer appended to the
//! 38,330,144-byte payload.
//!
//! Layout, little-endian, at the very end of the file:
//! `[arch_hash: u32][content_hash: u64][magic: u32 = "GN1\0"]`
//!
//! - `arch_hash` is an FNV-1a digest of the canonical architecture tuple
//!   (dimensions, quantization, scale, bucket layouts, baked file
//!   permutation). The trainer computes it from its own constants when it
//!   exports a network; this module computes it from the engine's. A
//!   trainer/engine drift makes the two values diverge and the load fails
//!   loudly — the silent-wrong-weights class of accident this exists for.
//! - `content_hash` is an FNV-1a 64 digest of the payload bytes, catching
//!   corruption and truncation.
//!
//! Both hashes are taken over the FILE bytes, never the in-memory struct:
//! `repermute_ft` rearranges the feature transformer per SIMD tier at load,
//! so memory is not portable across machines.
//!
//! A file without the footer is a legacy network (published before the footer
//! existed, HuggingFace repo is append-only): accepted, size-checked only.
//! Old engine binaries load footered files unchanged — the loader has always
//! tolerated up to 63 trailing padding bytes.

use super::features::KING_BUCKETS;
use super::network::{FILE_PERM, NNUE_FILE_SIZE};
use super::{
    FT_SIZE, INPUT_BUCKETS, L1_QUANT, L1_SIZE, L2_SIZE, L3_SIZE, NETWORK_SCALE,
    OUTPUT_BUCKETS, OUTPUT_BUCKET_MAP, FT_QUANT, THREAT_INPUT_SIZE,
};

/// Size of the integrity footer appended to the payload.
pub const FOOTER_SIZE: usize = 16;

/// Footer magic, last 4 bytes of a footered file.
pub const MAGIC: u32 = u32::from_le_bytes(*b"GN1\0");

const FNV32_OFFSET: u32 = 0x811c_9dc5;
const FNV32_PRIME: u32 = 0x0100_0193;
const FNV64_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV64_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Fold one little-endian u32 into an FNV-1a 32 state.
const fn h32(mut h: u32, v: u32) -> u32 {
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
/// from ITS constants (tools/trainer5); the golden tests on both sides pin
/// the same literal, so a legitimate architecture change is a conscious
/// two-sided update while an accidental drift fails `cargo test` on the
/// drifting side.
///
/// Tuple, in order: FT_SIZE, THREAT_INPUT_SIZE, L1_SIZE, L2_SIZE, L3_SIZE,
/// INPUT_BUCKETS, OUTPUT_BUCKETS, FT_QUANT, L1_QUANT, NETWORK_SCALE, the 64
/// KING_BUCKETS entries, the 33 OUTPUT_BUCKET_MAP entries, the 8 FILE_PERM
/// entries. Each value hashed as a little-endian u32.
pub const ARCH_HASH: u32 = {
    let mut h = FNV32_OFFSET;
    h = h32(h, FT_SIZE as u32);
    h = h32(h, THREAT_INPUT_SIZE as u32);
    h = h32(h, L1_SIZE as u32);
    h = h32(h, L2_SIZE as u32);
    h = h32(h, L3_SIZE as u32);
    h = h32(h, INPUT_BUCKETS as u32);
    h = h32(h, OUTPUT_BUCKETS as u32);
    h = h32(h, FT_QUANT as u32);
    h = h32(h, L1_QUANT as u32);
    h = h32(h, NETWORK_SCALE as u32);
    let mut i = 0;
    while i < KING_BUCKETS.len() {
        h = h32(h, KING_BUCKETS[i] as u32);
        i += 1;
    }
    let mut i = 0;
    while i < OUTPUT_BUCKET_MAP.len() {
        h = h32(h, OUTPUT_BUCKET_MAP[i] as u32);
        i += 1;
    }
    let mut i = 0;
    while i < FILE_PERM.len() {
        h = h32(h, FILE_PERM[i] as u32);
        i += 1;
    }
    h
};

/// Where a network file came from, as far as integrity can tell.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Provenance {
    /// Footer present, both hashes check out.
    Verified,
    /// No footer: a network published before the footer existed.
    /// Size-checked only — the caller should say so where a human looks.
    Legacy,
}

/// Validate a network file image. On success the payload to install is always
/// `data[..NNUE_FILE_SIZE]`.
pub fn verify(data: &[u8]) -> Result<Provenance, String> {
    if data.len() == NNUE_FILE_SIZE + FOOTER_SIZE {
        let tail = &data[NNUE_FILE_SIZE..];
        let magic = u32::from_le_bytes(tail[12..16].try_into().unwrap());
        if magic == MAGIC {
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
            return Ok(Provenance::Verified);
        }
        // Right length for a footer but no magic: fall through to the legacy
        // tolerance below (16 bytes of Bullet padding are indistinguishable
        // from garbage, and always were).
    }

    // Legacy path: exact payload size, plus up to 63 bytes of Bullet
    // alignment padding. `wrapping_sub` makes any undersized file fail too.
    let excess = data.len().wrapping_sub(NNUE_FILE_SIZE);
    if excess > 63 {
        return Err(format!(
            "File size {} != expected {NNUE_FILE_SIZE} bytes (got {} extra, max 63 padding)",
            data.len(),
            if data.len() < NNUE_FILE_SIZE {
                -((NNUE_FILE_SIZE - data.len()) as isize)
            } else {
                excess as isize
            },
        ));
    }
    Ok(Provenance::Legacy)
}

/// Build the 16-byte footer for a payload (test helper; the trainer has its
/// own mirror implementation in tools/trainer5).
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

    /// Pins the architecture digest. tools/trainer5 pins the SAME literal
    /// from its own constants: change one side's architecture and its golden
    /// test fails before any network is ever exported or loaded. A
    /// legitimate architecture change updates both literals consciously.
    #[test]
    fn the_architecture_digest_is_pinned_on_both_sides() {
        assert_eq!(ARCH_HASH, 0xee36_e388, "update tools/trainer5 in lockstep");
    }

    #[test]
    fn a_footered_file_verifies_and_every_lie_is_caught() {
        // Small stand-in payload is not possible: verify() is anchored to
        // NNUE_FILE_SIZE. Build the real-size buffer once.
        let mut data = vec![0u8; NNUE_FILE_SIZE];
        for (i, b) in data.iter_mut().enumerate() {
            *b = (i * 31 + 7) as u8;
        }
        let f = footer(&data);
        let mut footered = data.clone();
        footered.extend_from_slice(&f);
        assert_eq!(verify(&footered), Ok(Provenance::Verified));

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

        // No footer at all: legacy, accepted.
        assert_eq!(verify(&data), Ok(Provenance::Legacy));
        // Legacy tolerance: up to 63 padding bytes, but not 64.
        let mut padded = data.clone();
        padded.extend_from_slice(&[0u8; 63]);
        assert_eq!(verify(&padded), Ok(Provenance::Legacy));
        padded.push(0);
        assert!(verify(&padded).is_err());
        // Undersized: rejected.
        assert!(verify(&data[..NNUE_FILE_SIZE - 1]).is_err());
    }
}
