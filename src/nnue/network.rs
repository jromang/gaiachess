//! Network parameters struct and loading.
//!
//! `NNUEParams` binary layout: 38,330,144 bytes.
//! Total data size: 38,330,144 bytes.
//!
//! Loading:
//! - Compile-time: `include_bytes!` → zstd-compressed in binary
//! - Runtime: `load_from_file()` via UCI `setoption name EvalFile value <path>`
//!
//! L1 weights are already in NNZ-permuted dpbusd layout — no transform needed.
//!
//! The FT weights/biases, however, carry the *packus* permutation written by the
//! trainer, and that permutation is ISA-dependent: `activate_ft` writes the result
//! of `packus` straight to memory without a fixup `permute`, so the weights must be
//! pre-ordered to cancel the lane interleaving of the target's `packus`. The trainer
//! bakes the AVX-512 pattern; `repermute_ft` converts it to whatever the compiled
//! target needs. Skipped entirely on AVX-512.

use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};
use std::sync::OnceLock;

use super::{FT_SIZE, L1_SIZE, L2_SIZE, L3_SIZE, OUTPUT_BUCKETS, THREAT_INPUT_SIZE};

// ============================================================
// Aligned wrapper for SIMD-friendly memory
// ============================================================

/// 64-byte aligned wrapper for SIMD compatibility.
#[repr(C, align(64))]
#[derive(Clone)]
pub struct Aligned<T>(pub T);

impl<T> std::ops::Deref for Aligned<T> {
    type Target = T;
    #[inline(always)]
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> std::ops::DerefMut for Aligned<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

// ============================================================
// Network parameters
// ============================================================

/// Complete NNUE network parameters.
///
/// Binary layout: all arrays are 64-byte aligned.
/// All arrays are 64-byte aligned; all field sizes are multiples of 64 bytes
/// (except l3_biases which is 32 bytes, padded to 64 by Aligned).
/// Direct memcpy from the `.bin` file — no post-load transforms needed.
///
/// Architecture: PST (i16) + Full Threats (i8) → FT 640 →
///   L1 (640→16, dpbusd) → CReLU+squared → f32[32] →
///   L2 (32→32, FMA) → squared → f32[32] →
///   L3 (concat(L2[32], L1[32]) → 1) × 8 buckets → centipawns
#[repr(C, align(64))]
pub struct NNUEParams {
    /// PST feature transformer weights: `[pst_feature][neuron]`.
    /// i16 quantized (QA=255). FT_SIZE = 12×768 = 9,216 features.
    pub ft_pst_weights: Aligned<[[i16; L1_SIZE]; FT_SIZE]>,

    /// Full threat feature transformer weights: `[threat_feature][neuron]`.
    /// i8 quantized (×255). THREAT_INPUT_SIZE = 41,272 filtered features (GaiaNet-T1).
    pub ft_threat_weights: Aligned<[[i8; L1_SIZE]; THREAT_INPUT_SIZE]>,

    /// Feature transformer biases: initial accumulator values.
    pub ft_biases: Aligned<[i16; L1_SIZE]>,

    /// L1 sparse layer weights: `[bucket][input_group_of_4][output*4+byte]`.
    /// Already in NNZ-permuted dpbusd layout (no post-load transform needed).
    /// Input: L1_SIZE u8 (FT activation output), Output: L2_SIZE (16) i32.
    pub l1_weights: Aligned<[[[i8; L2_SIZE * 4]; L1_SIZE / 4]; OUTPUT_BUCKETS]>,

    /// L1 biases per output bucket: `[bucket][neuron]`.
    pub l1_biases: Aligned<[[f32; L2_SIZE]; OUTPUT_BUCKETS]>,

    /// L2 dense layer weights: `[bucket][input][output]`.
    /// 2*L2_SIZE (32) inputs, L3_SIZE (32) outputs.
    /// Note: Rust type dimensions match only because 2*L2_SIZE == L3_SIZE == 32.
    pub l2_weights: Aligned<[[[f32; L3_SIZE]; 2 * L2_SIZE]; OUTPUT_BUCKETS]>,

    /// L2 biases per output bucket: `[bucket][neuron]`.
    pub l2_biases: Aligned<[[f32; L3_SIZE]; OUTPUT_BUCKETS]>,

    /// L3 output layer weights: `[bucket][input]`.
    /// L3_SIZE (32) from L2 + 2*L2_SIZE (32) skip from L1 = 64 inputs.
    pub l3_weights: Aligned<[[f32; L3_SIZE + 2 * L2_SIZE]; OUTPUT_BUCKETS]>,

    /// L3 biases per output bucket.
    /// Note: 32 bytes of data, padded to 64 by Aligned. Binary ends at +32 bytes.
    pub l3_biases: Aligned<[f32; OUTPUT_BUCKETS]>,
}

/// Exact byte count of the network binary.
pub const NNUE_FILE_SIZE: usize =
    FT_SIZE * L1_SIZE * 2                              // ft_pst_weights: i16
    + THREAT_INPUT_SIZE * L1_SIZE                      // ft_threat_weights: i8
    + L1_SIZE * 2                                      // ft_biases: i16
    + OUTPUT_BUCKETS * (L1_SIZE / 4) * (L2_SIZE * 4)  // l1_weights: i8
    + OUTPUT_BUCKETS * L2_SIZE * 4                     // l1_biases: f32
    + OUTPUT_BUCKETS * L3_SIZE * (2 * L2_SIZE) * 4    // l2_weights: f32
    + OUTPUT_BUCKETS * L3_SIZE * 4                     // l2_biases: f32
    + OUTPUT_BUCKETS * (L3_SIZE + 2 * L2_SIZE) * 4    // l3_weights: f32
    + OUTPUT_BUCKETS * 4;                              // l3_biases: f32

const _: () = assert!(NNUE_FILE_SIZE == 38_330_144, "NNUE_FILE_SIZE must be 38,330,144");

impl NNUEParams {
    /// Allocate a zeroed network on the heap.
    pub fn zeroed_box() -> Box<Self> {
        // SAFETY: NNUEParams is repr(C) with only integer and float arrays.
        // Zero-initialized values are valid for all field types.
        unsafe { Box::<NNUEParams>::new_zeroed().assume_init() }
    }
}

// ============================================================
// FT packus re-permutation (ISA-dependent)
// ============================================================

/// Block size of the packus permutation: 8 elements
/// (one `__m128i` of i16, one `uint64_t` of i8).
const PERM_BLOCK: usize = 8;

/// Permutation written into the `.bin` by the trainer
/// (`tools/trainer5/src/save_format.rs`, `packus_permute_*`): the AVX-512 pattern,
/// which cancels the lane interleaving of `_mm512_packus_epi16`.
const FILE_PERM: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];

/// Permutation the compiled target's `packus` needs.
///
/// - AVX-512 (`_mm512_packus_epi16`): same as the file — nothing to do.
/// - AVX2 (`_mm256_packus_epi16`): crosses 128-bit lanes over 4 blocks, not 8.
/// - NEON / scalar: no lane crossing, weights must be linear (identity).
#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
const TARGET_PERM: &[usize] = &[0, 2, 1, 3];
#[cfg(not(any(target_feature = "avx2", target_feature = "avx512f")))]
const TARGET_PERM: &[usize] = &[0];

/// Convert one row of `L1_SIZE` elements from the file's permutation to the target's.
///
/// Both group sizes (64 for the file, 32 or 8 for the target) divide `L1_SIZE` = 640,
/// and the FT half boundary at 320 is a multiple of 64 — so the left and right halves
/// that `activate_ft` pairs element-wise stay in step.
#[cfg(not(target_feature = "avx512f"))]
fn repermute_row<T: Copy + Default>(row: &mut [T]) {
    const FILE_GROUP: usize = PERM_BLOCK * FILE_PERM.len(); // 64
    debug_assert_eq!(row.len() % FILE_GROUP, 0);

    let mut tmp = [T::default(); FILE_GROUP];

    // 1. Undo the file's AVX-512 permutation → linear order.
    //    The trainer wrote `permuted[j] = original[FILE_PERM[j]]`, so invert it.
    for chunk in row.chunks_exact_mut(FILE_GROUP) {
        for (j, &p) in FILE_PERM.iter().enumerate() {
            tmp[p * PERM_BLOCK..(p + 1) * PERM_BLOCK]
                .copy_from_slice(&chunk[j * PERM_BLOCK..(j + 1) * PERM_BLOCK]);
        }
        chunk.copy_from_slice(&tmp);
    }

    // 2. Apply the target's permutation (identity when TARGET_PERM == [0]).
    let target_group = PERM_BLOCK * TARGET_PERM.len();
    debug_assert_eq!(row.len() % target_group, 0);
    for chunk in row.chunks_exact_mut(target_group) {
        for (j, &p) in TARGET_PERM.iter().enumerate() {
            tmp[j * PERM_BLOCK..(j + 1) * PERM_BLOCK]
                .copy_from_slice(&chunk[p * PERM_BLOCK..(p + 1) * PERM_BLOCK]);
        }
        chunk.copy_from_slice(&tmp[..target_group]);
    }
}

/// Re-permute every FT array that `activate_ft` reads through `packus`.
///
/// No-op on AVX-512, where the file's layout is already the right one.
#[cfg(target_feature = "avx512f")]
fn repermute_ft(_params: &mut NNUEParams) {}

#[cfg(not(target_feature = "avx512f"))]
fn repermute_ft(params: &mut NNUEParams) {
    for row in params.ft_pst_weights.0.iter_mut() {
        repermute_row(row);
    }
    for row in params.ft_threat_weights.0.iter_mut() {
        repermute_row(row);
    }
    repermute_row(&mut params.ft_biases.0);
}

// ============================================================
// Global parameters with runtime loading
// ============================================================

/// Pointer to a runtime-loaded network (set via `load_from_file`).
static RUNTIME_PARAMS: AtomicPtr<NNUEParams> = AtomicPtr::new(std::ptr::null_mut());

/// Network generation counter (incremented on each `load_from_file`).
static NETWORK_GENERATION: AtomicU64 = AtomicU64::new(0);

/// Current network generation (for Finny table cache invalidation).
pub fn generation() -> u64 {
    NETWORK_GENERATION.load(Ordering::Relaxed)
}

/// Compressed embedded network (zstd), decompressed on first access.
#[cfg(feature = "nnue")]
static COMPRESSED_MODEL: &[u8] = include_bytes!(env!("MODEL_ZST"));

#[cfg(feature = "nnue")]
static DEFAULT_PARAMS: OnceLock<Box<NNUEParams>> = OnceLock::new();

#[cfg(not(feature = "nnue"))]
static DEFAULT_PARAMS_ZEROED: OnceLock<Box<NNUEParams>> = OnceLock::new();

/// Get the active network parameters (runtime-loaded or default).
#[inline]
pub fn params() -> &'static NNUEParams {
    let ptr = RUNTIME_PARAMS.load(Ordering::Relaxed);
    if !ptr.is_null() {
        return unsafe { &*ptr };
    }
    #[cfg(feature = "nnue")]
    return DEFAULT_PARAMS.get_or_init(decompress_embedded);
    #[cfg(not(feature = "nnue"))]
    return DEFAULT_PARAMS_ZEROED.get_or_init(NNUEParams::zeroed_box);
}

/// Returns true if a real trained network is available (embedded or runtime-loaded).
pub fn has_network() -> bool {
    !RUNTIME_PARAMS.load(Ordering::Relaxed).is_null() || cfg!(feature = "nnue")
}

/// Decompress the zstd-compressed embedded network into a heap-allocated Box.
/// Called once on first access via OnceLock.
#[cfg(feature = "nnue")]
fn decompress_embedded() -> Box<NNUEParams> {
    use std::io::Read;
    let mut decoder = ruzstd::decoding::StreamingDecoder::new(COMPRESSED_MODEL)
        .expect("Failed to init zstd decoder for embedded NNUE");
    let mut params = NNUEParams::zeroed_box();
    // Read exactly NNUE_FILE_SIZE bytes into the struct (Aligned padding stays zero).
    let dst = unsafe {
        std::slice::from_raw_parts_mut(
            params.as_mut() as *mut NNUEParams as *mut u8,
            NNUE_FILE_SIZE,
        )
    };
    decoder
        .read_exact(dst)
        .expect("Failed to decompress embedded NNUE network");
    repermute_ft(&mut params);
    params
}

/// Load a network from a `.bin` file at runtime.
///
/// Accepts files of exactly `NNUE_FILE_SIZE` bytes, or up to 63 bytes larger
/// (the Bullet trainer adds padding to align to 64 bytes).
/// The previous runtime-loaded network (if any) is leaked (~37 MB).
pub fn load_from_file(path: &str) -> Result<(), String> {
    let data = std::fs::read(path).map_err(|e| format!("Failed to read {path}: {e}"))?;

    let excess = data.len().wrapping_sub(NNUE_FILE_SIZE);
    if excess > 63 {
        return Err(format!(
            "File size {} != expected {NNUE_FILE_SIZE} bytes (got {} extra, max 63 padding)",
            data.len(),
            if data.len() < NNUE_FILE_SIZE { -(excess as isize) } else { excess as isize },
        ));
    }

    // Allocate zeroed box and copy the raw binary in, then fix the FT packus
    // permutation for the compiled target (no-op on AVX-512).
    let mut params = NNUEParams::zeroed_box();
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr(),
            params.as_mut() as *mut NNUEParams as *mut u8,
            NNUE_FILE_SIZE,
        );
    }
    repermute_ft(&mut params);

    // Leak to get 'static lifetime, store in global.
    let ptr = Box::into_raw(params);
    RUNTIME_PARAMS.store(ptr, Ordering::Release);

    // Increment generation so Finny table caches are invalidated.
    NETWORK_GENERATION.fetch_add(1, Ordering::Release);

    Ok(())
}
