//! Save format: raw Bullet weights + quantize/permute post-processing in Rust.
//!
//! Bullet saves raw f32 weights via 8 SavedFormat::id entries; process_net()
//! then quantizes and permutes them into the engine's binary format as a
//! post-training step.

use bullet_lib::trainer::save::SavedFormat;

use crate::{L1_SIZE, L2_SIZE, L3_SIZE, NUM_OUTPUT_BUCKETS, TOTAL_THREATS};

const KING_BUCKETS: usize = 12;
const INPUT_QUANT: f32 = 255.0;
const L1_QUANT: f32 = 64.0;

/// Expected binary size (must match engine's NNUE_FILE_SIZE).
pub const NNUE_FILE_SIZE: usize = 38_330_144;

/// Packus permutation baked into the exported file (AVX-512 lane order).
/// Part of the file contract, hashed into ARCH_HASH.
pub(crate) const FILE_PERM: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];

// ============================================================
// Integrity footer — mirror of the engine's src/nnue/integrity.rs
// ============================================================
//
// 16 bytes appended after the payload: [arch_hash u32][content_hash u64]
// [magic u32 = "GN1\0"], little-endian. ARCH_HASH is computed here from the
// TRAINER's own constants and by the engine from its own: a drift between
// the two sides makes the values diverge and the engine refuses the network
// loudly. The golden tests on both sides pin the same literal.

/// Size of the integrity footer.
pub const FOOTER_SIZE: usize = 16;

/// Footer magic, last 4 bytes of a footered file.
pub const FOOTER_MAGIC: u32 = u32::from_le_bytes(*b"GN1\0");

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

/// FNV-1a 64 over the payload bytes.
pub fn fnv1a64(data: &[u8]) -> u64 {
    let mut h = FNV64_OFFSET;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(FNV64_PRIME);
    }
    h
}

/// Canonical architecture digest — same tuple, same order as the engine:
/// FT_SIZE, THREAT_INPUT_SIZE, L1, L2, L3, INPUT_BUCKETS, OUTPUT_BUCKETS,
/// FT_QUANT, L1_QUANT, NETWORK_SCALE, 64 king-bucket entries (half-board
/// layout mirrored like the engine's features.rs), 33 output-bucket entries
/// (MaterialCount formula `(count-2)/4`), 8 FILE_PERM entries.
pub const ARCH_HASH: u32 = {
    let mut h = FNV32_OFFSET;
    h = h32(h, (768 * KING_BUCKETS) as u32); // FT_SIZE
    h = h32(h, TOTAL_THREATS as u32);
    h = h32(h, L1_SIZE as u32);
    h = h32(h, L2_SIZE as u32);
    h = h32(h, L3_SIZE as u32);
    h = h32(h, KING_BUCKETS as u32);
    h = h32(h, NUM_OUTPUT_BUCKETS as u32);
    h = h32(h, INPUT_QUANT as u32);
    h = h32(h, L1_QUANT as u32);
    h = h32(h, crate::EVAL_SCALE as u32);
    // King buckets, expanded to 64 squares: files e-h mirror a-d.
    let mut sq = 0;
    while sq < 64 {
        let (rank, file) = (sq / 8, sq % 8);
        let half = if file < 4 { file } else { 7 - file };
        h = h32(h, crate::BUCKET_LAYOUT[rank * 4 + half] as u32);
        sq += 1;
    }
    // Output buckets from Bullet's MaterialCount<8>.
    let mut count = 0;
    while count <= 32 {
        let bucket = if count < 2 { 0 } else { (count - 2) / 4 };
        h = h32(h, bucket as u32);
        count += 1;
    }
    let mut i = 0;
    while i < FILE_PERM.len() {
        h = h32(h, FILE_PERM[i] as u32);
        i += 1;
    }
    h
};

/// Build the 16-byte integrity footer for an exported payload.
pub fn footer(payload: &[u8]) -> [u8; FOOTER_SIZE] {
    assert_eq!(payload.len(), NNUE_FILE_SIZE);
    let mut f = [0u8; FOOTER_SIZE];
    f[0..4].copy_from_slice(&ARCH_HASH.to_le_bytes());
    f[4..12].copy_from_slice(&fnv1a64(payload).to_le_bytes());
    f[12..16].copy_from_slice(&FOOTER_MAGIC.to_le_bytes());
    f
}

/// Raw save format (8 entries, no transforms).
/// Bullet saves raw f32 weights that can be processed by process_net().
pub fn build_save_format() -> Vec<SavedFormat> {
    vec![
        SavedFormat::id("l0w"),
        SavedFormat::id("l0b"),
        SavedFormat::id("l1w"),
        SavedFormat::id("l1b"),
        SavedFormat::id("l2w"),
        SavedFormat::id("l2b"),
        SavedFormat::id("l3w"),
        SavedFormat::id("l3b"),
    ]
}

/// Quantize + transpose/permute raw weights into the engine's binary format.
///
/// Input: raw.bin from Bullet (f32, column-major per layer)
/// Output: Vec<u8> of exactly NNUE_FILE_SIZE bytes
pub fn process_net(raw_path: &str) -> std::io::Result<Vec<u8>> {
    let raw_bytes = std::fs::read(raw_path)?;
    let raw: &[f32] = bytemuck_cast_slice(&raw_bytes);

    // ============================================================
    // Parse raw Bullet layout
    // Bullet saves: l0w, l0b, l1w, l1b, l2w, l2b, l3w, l3b
    // Each is column-major: weights[output + input * num_outputs]
    // ============================================================

    let total_inputs = 768 + TOTAL_THREATS + 768 * KING_BUCKETS; // 51,256
    let l0w_size = total_inputs * L1_SIZE;
    let l0b_size = L1_SIZE;
    let l1w_size = (NUM_OUTPUT_BUCKETS * L2_SIZE) * L1_SIZE;
    let l1b_size = NUM_OUTPUT_BUCKETS * L2_SIZE;
    let l2w_size = (NUM_OUTPUT_BUCKETS * L3_SIZE) * (2 * L2_SIZE);
    let l2b_size = NUM_OUTPUT_BUCKETS * L3_SIZE;
    let l3w_size = NUM_OUTPUT_BUCKETS * (L3_SIZE + 2 * L2_SIZE);
    let l3b_size = NUM_OUTPUT_BUCKETS;

    let mut off = 0;
    let l0w = &raw[off..off + l0w_size]; off += l0w_size;
    let l0b = &raw[off..off + l0b_size]; off += l0b_size;
    let l1w = &raw[off..off + l1w_size]; off += l1w_size;
    let l1b = &raw[off..off + l1b_size]; off += l1b_size;
    let l2w = &raw[off..off + l2w_size]; off += l2w_size;
    let l2b = &raw[off..off + l2b_size]; off += l2b_size;
    let l3w = &raw[off..off + l3w_size]; off += l3w_size;
    let l3b = &raw[off..off + l3b_size]; let _ = off;

    // ============================================================
    // Step 1: Quantize
    // ============================================================

    // Bullet l0w column-major: l0w[output + feature * L1_SIZE]
    // Feature layout: [768 fact | 41272 threats | 9216 PST bucketed]

    // 1a. Threat weights: features [768, 768+TOTAL_THREATS) → i8(×255)
    let mut threat_weights = vec![0i8; TOTAL_THREATS * L1_SIZE];
    for w in 0..TOTAL_THREATS * L1_SIZE {
        let feat = w / L1_SIZE;
        let out = w % L1_SIZE;
        let raw_idx = out + (768 + feat) * L1_SIZE; // skip factoriser (768 features)
        let q = (l0w[raw_idx] * INPUT_QUANT).round() as i16;
        threat_weights[w] = q.clamp(-128, 127) as i8;
    }

    // 1b. PST weights: merge factoriser (0..768) + bucketed (768+THREATS..end) → i16(×255)
    let mut pst_weights = vec![0i16; 768 * KING_BUCKETS * L1_SIZE];
    for kb in 0..KING_BUCKETS {
        for w in 0..768 * L1_SIZE {
            let feat = w / L1_SIZE;
            let out = w % L1_SIZE;
            let pst_idx = kb * 768 * L1_SIZE + w;
            // Bucketed PST at features [768 + TOTAL_THREATS + kb*768 + feat]
            let raw_bucketed = out + (768 + TOTAL_THREATS + kb * 768 + feat) * L1_SIZE;
            let factoriser_idx = out + feat * L1_SIZE; // features [0..768)
            pst_weights[pst_idx] = (l0w[raw_bucketed] * INPUT_QUANT).round() as i16
                                 + (l0w[factoriser_idx] * INPUT_QUANT).round() as i16;
        }
    }

    // 1c. Input biases → i16(×255)
    let mut input_biases = vec![0i16; L1_SIZE];
    for b in 0..L1_SIZE {
        input_biases[b] = (l0b[b] * INPUT_QUANT).round() as i16;
    }

    // 1d. L1 weights → i8(×64)
    // Bullet l1w column-major: l1w[output + input * num_outputs]
    // where num_outputs = OUTPUT_BUCKETS * L2_SIZE = 128
    // Intermediate layout: l1_weights_tmp[b][l2 * L1_SIZE + l1] = quant(raw)
    let l1_out_total = NUM_OUTPUT_BUCKETS * L2_SIZE;
    let mut l1_weights_tmp = vec![0i8; NUM_OUTPUT_BUCKETS * L1_SIZE * L2_SIZE];
    for b in 0..NUM_OUTPUT_BUCKETS {
        for l1 in 0..L1_SIZE {
            for l2 in 0..L2_SIZE {
                // Bullet column-major: l1w[output_idx + input_idx * num_outputs]
                let output_idx = b * L2_SIZE + l2;
                let input_idx = l1;
                let raw_val = l1w[output_idx + input_idx * l1_out_total];
                l1_weights_tmp[b * L1_SIZE * L2_SIZE + l2 * L1_SIZE + l1] =
                    (raw_val * L1_QUANT).round().clamp(-128.0, 127.0) as i8;
            }
        }
    }

    // 1e. L1 biases, L2/L3 weights/biases: f32, kept unquantized.
    // But we need to handle the Bullet column-major → row-major transpose

    // ============================================================
    // Step 2: Transpose/Permute
    // ============================================================

    // 2a. Packus permutation on PST weights, threat weights, biases
    // (matches the engine's AVX-512 packus lane order)
    // PST weights (i16): permute blocks of 8 i16 values
    packus_permute_i16(&mut pst_weights, L1_SIZE);

    // Input biases (i16): same permutation
    packus_permute_i16(&mut input_biases, L1_SIZE);

    // Threat weights (i8): permute blocks of 8 bytes (treated as u64 lanes)
    packus_permute_i8(&mut threat_weights, L1_SIZE);

    // 2b. L1 weights: dpbusd transpose (interleave groups of 4 inputs)
    let mut l1_weights_out = vec![0i8; NUM_OUTPUT_BUCKETS * L1_SIZE * L2_SIZE];
    for b in 0..NUM_OUTPUT_BUCKETS {
        for l1_group in 0..L1_SIZE / 4 {
            for l2 in 0..L2_SIZE {
                for c in 0..4 {
                    // out[b][l1_group * 4 * L2_SIZE + l2 * 4 + c] =
                    //   tmp[b][(l1_group * 4 + c) * OUTPUT_BUCKETS * L2_SIZE + b * L2_SIZE + l2]
                    // where tmp is already indexed as tmp[b][l2 * L1_SIZE + l1]
                    let src = l1_weights_tmp[b * L1_SIZE * L2_SIZE + l2 * L1_SIZE + (l1_group * 4 + c)];
                    let dst_idx = b * (L1_SIZE / 4) * L2_SIZE * 4
                                + l1_group * L2_SIZE * 4
                                + l2 * 4
                                + c;
                    l1_weights_out[dst_idx] = src;
                }
            }
        }
    }

    // 2c. L2 weights: transpose per bucket
    // Bullet l2w column-major: l2w[output + input * num_outputs]
    // where num_outputs = OUTPUT_BUCKETS * L3_SIZE = 256
    let l2_out_total = NUM_OUTPUT_BUCKETS * L3_SIZE;
    let l2_in_size = 2 * L2_SIZE;
    let mut l2_weights_out = vec![0.0f32; NUM_OUTPUT_BUCKETS * l2_in_size * L3_SIZE];
    for b in 0..NUM_OUTPUT_BUCKETS {
        for l2 in 0..l2_in_size {
            for l3 in 0..L3_SIZE {
                // out[b][l2*L3 + l3] = raw[l2*OB*L3 + b*L3 + l3]
                let src = l2w[(b * L3_SIZE + l3) + l2 * l2_out_total];
                l2_weights_out[b * l2_in_size * L3_SIZE + l2 * L3_SIZE + l3] = src;
            }
        }
    }

    // 2d. L3 weights: transpose per bucket
    let l3_in_size = L3_SIZE + 2 * L2_SIZE;
    let mut l3_weights_out = vec![0.0f32; NUM_OUTPUT_BUCKETS * l3_in_size];
    for b in 0..NUM_OUTPUT_BUCKETS {
        for l3 in 0..l3_in_size {
            // out[b][l3] = raw[l3 * OUTPUT_BUCKETS + b]
            l3_weights_out[b * l3_in_size + l3] = l3w[l3 * NUM_OUTPUT_BUCKETS + b];
        }
    }

    // 2e. L1 biases: Bullet column-major l1b[output + 0 * num_outputs]
    // = l1b[b * L2_SIZE + l2] for b in 0..8, l2 in 0..16
    // already in [bucket][l2] order — direct copy
    let mut l1_biases_out = vec![0.0f32; NUM_OUTPUT_BUCKETS * L2_SIZE];
    for i in 0..l1b.len() {
        l1_biases_out[i] = l1b[i];
    }

    // 2f. L2 biases: same (direct copy)
    let mut l2_biases_out = vec![0.0f32; NUM_OUTPUT_BUCKETS * L3_SIZE];
    for i in 0..l2b.len() {
        l2_biases_out[i] = l2b[i];
    }

    // 2g. L3 biases: same (direct copy)
    let mut l3_biases_out = vec![0.0f32; NUM_OUTPUT_BUCKETS];
    for i in 0..l3b.len() {
        l3_biases_out[i] = l3b[i];
    }

    // ============================================================
    // Step 3: Serialize to binary (same order as NetworkData struct)
    // ============================================================

    let mut output = Vec::with_capacity(NNUE_FILE_SIZE);

    // ft_pst_weights: [9216][640] i16
    for &v in &pst_weights {
        output.extend_from_slice(&v.to_le_bytes());
    }

    // ft_threat_weights: [41272][640] i8
    for &v in &threat_weights {
        output.push(v as u8);
    }

    // ft_biases: [640] i16
    for &v in &input_biases {
        output.extend_from_slice(&v.to_le_bytes());
    }

    // l1_weights: [8][160][64] i8
    for &v in &l1_weights_out {
        output.push(v as u8);
    }

    // l1_biases: [8][16] f32
    for &v in &l1_biases_out {
        output.extend_from_slice(&v.to_le_bytes());
    }

    // l2_weights: [8][32][32] f32
    for &v in &l2_weights_out {
        output.extend_from_slice(&v.to_le_bytes());
    }

    // l2_biases: [8][32] f32
    for &v in &l2_biases_out {
        output.extend_from_slice(&v.to_le_bytes());
    }

    // l3_weights: [8][64] f32
    for &v in &l3_weights_out {
        output.extend_from_slice(&v.to_le_bytes());
    }

    // l3_biases: [8] f32
    for &v in &l3_biases_out {
        output.extend_from_slice(&v.to_le_bytes());
    }

    assert_eq!(output.len(), NNUE_FILE_SIZE,
        "process_net output size {} != expected {NNUE_FILE_SIZE}", output.len());

    // Append the integrity footer: every exported network is verifiable, and
    // engine binaries that predate the footer ignore the 16 trailing bytes
    // (their loaders tolerate up to 63 bytes of padding).
    let f = footer(&output);
    output.extend_from_slice(&f);

    Ok(output)
}

// ============================================================
// Helpers
// ============================================================

fn bytemuck_cast_slice(bytes: &[u8]) -> &[f32] {
    assert!(bytes.len() % 4 == 0);
    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4) }
}

/// Packus permutation for i16 arrays (PST weights, biases).
/// Reorders blocks of 8 i16 values within each row of `stride` elements.
fn packus_permute_i16(data: &mut [i16], stride: usize) {
    const PERM: [usize; 8] = FILE_PERM;
    const BLOCK: usize = 8; // i16 per __m128i
    let group = BLOCK * PERM.len(); // 64 i16 values

    for row in data.chunks_exact_mut(stride) {
        for chunk in row.chunks_exact_mut(group) {
            let mut tmp = [0i16; 64];
            for (j, &p) in PERM.iter().enumerate() {
                tmp[j * BLOCK..(j + 1) * BLOCK]
                    .copy_from_slice(&chunk[p * BLOCK..(p + 1) * BLOCK]);
            }
            chunk.copy_from_slice(&tmp);
        }
    }
}

/// Packus permutation for i8 arrays (threat weights).
/// Reorders blocks of 8 bytes (u64 lanes) within each row.
fn packus_permute_i8(data: &mut [i8], stride: usize) {
    const PERM: [usize; 8] = FILE_PERM;
    const BLOCK: usize = 8; // bytes per uint64_t
    let group = BLOCK * PERM.len(); // 64 bytes

    let data_u8 = unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, data.len()) };
    for row in data_u8.chunks_exact_mut(stride) {
        for chunk in row.chunks_exact_mut(group) {
            let mut tmp = [0u8; 64];
            for (j, &p) in PERM.iter().enumerate() {
                tmp[j * BLOCK..(j + 1) * BLOCK]
                    .copy_from_slice(&chunk[p * BLOCK..(p + 1) * BLOCK]);
            }
            chunk.copy_from_slice(&tmp);
        }
    }
}
