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
//! trainer, and that permutation is tier-dependent: `activate_ft` writes the result
//! of `packus` straight to memory without a fixup `permute`, so the weights must be
//! pre-ordered to cancel the lane interleaving of that tier's `packus`. The trainer
//! bakes the AVX-512 pattern; `repermute_ft` converts it to whatever the tier
//! resolved for this machine needs. Skipped entirely on the AVX-512 tiers.

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
// FT packus re-permutation (tier-dependent)
// ============================================================

/// FT weight layout required by a SIMD tier's `packus`.
///
/// The kernels write the result of `packus` straight to memory without a fixup
/// `permute`, so the weights must be pre-ordered to cancel that instruction's
/// lane interleaving. Which interleaving that is depends on the tier the engine
/// will run — decided at load time, from the same resolution as the kernels
/// themselves, so the two can never disagree.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PermKind {
    /// AVX-512 `_mm512_packus_epi16` order — the file's own layout; nothing to do.
    File512,
    /// AVX2 `_mm256_packus_epi16` order — crosses 128-bit lanes over 4 blocks.
    Avx2,
    /// No lane crossing (NEON, wasm128, scalar): weights must be linear.
    Linear,
}

/// The permutation for the tier the engine will actually run.
fn active_perm() -> PermKind {
    #[cfg(target_arch = "x86_64")]
    {
        crate::cpu::get_or_init().perm
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        PermKind::Linear
    }
}

/// Block size of the packus permutation: 8 elements
/// (one `__m128i` of i16, one `uint64_t` of i8).
const PERM_BLOCK: usize = 8;

/// Permutation written into the `.bin` by the trainer
/// (`tools/trainer5/src/save_format.rs`, `packus_permute_*`): the AVX-512 pattern,
/// which cancels the lane interleaving of `_mm512_packus_epi16`.
pub(crate) const FILE_PERM: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];

/// Convert one row of `L1_SIZE` elements from the file's permutation to the target's.
///
/// Both group sizes (64 for the file, 32 or 8 for the target) divide `L1_SIZE` = 640,
/// and the FT half boundary at 320 is a multiple of 64 — so the left and right halves
/// that `activate_ft` pairs element-wise stay in step.
fn repermute_row<T: Copy + Default>(row: &mut [T], target_perm: &[usize]) {
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

    // 2. Apply the target's permutation (identity when target_perm == [0]).
    let target_group = PERM_BLOCK * target_perm.len();
    debug_assert_eq!(row.len() % target_group, 0);
    for chunk in row.chunks_exact_mut(target_group) {
        for (j, &p) in target_perm.iter().enumerate() {
            tmp[j * PERM_BLOCK..(j + 1) * PERM_BLOCK]
                .copy_from_slice(&chunk[p * PERM_BLOCK..(p + 1) * PERM_BLOCK]);
        }
        chunk.copy_from_slice(&tmp[..target_group]);
    }
}

/// Re-permute every FT array that `activate_ft` reads through `packus`,
/// to the layout of the tier resolved for this machine.
///
/// No-op on the AVX-512 tiers, where the file's layout is already the right one.
fn repermute_ft(params: &mut NNUEParams) {
    let target_perm: &[usize] = match active_perm() {
        PermKind::File512 => return,
        PermKind::Avx2 => &[0, 2, 1, 3],
        PermKind::Linear => &[0],
    };
    for row in params.ft_pst_weights.0.iter_mut() {
        repermute_row(row, target_perm);
    }
    for row in params.ft_threat_weights.0.iter_mut() {
        repermute_row(row, target_perm);
    }
    repermute_row(&mut params.ft_biases.0, target_perm);
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
#[cfg(all(feature = "nnue", net_embedded))]
static COMPRESSED_MODEL: &[u8] = include_bytes!(env!("MODEL_ZST"));

#[cfg(all(feature = "nnue", net_embedded))]
static DEFAULT_PARAMS: OnceLock<Box<NNUEParams>> = OnceLock::new();

#[cfg(not(all(feature = "nnue", net_embedded)))]
static DEFAULT_PARAMS_ZEROED: OnceLock<Box<NNUEParams>> = OnceLock::new();

/// Get the active network parameters (runtime-loaded or default).
#[inline]
pub fn params() -> &'static NNUEParams {
    let ptr = RUNTIME_PARAMS.load(Ordering::Relaxed);
    if !ptr.is_null() {
        return unsafe { &*ptr };
    }
    #[cfg(all(feature = "nnue", net_embedded))]
    return DEFAULT_PARAMS.get_or_init(decompress_embedded);
    #[cfg(not(all(feature = "nnue", net_embedded)))]
    return DEFAULT_PARAMS_ZEROED.get_or_init(NNUEParams::zeroed_box);
}

/// Returns true if a real trained network is available (embedded or runtime-loaded).
///
/// The embedded half of the test must match the `COMPRESSED_MODEL` cfg exactly:
/// `net_embedded` alone would claim a network on a `MODEL=`-but-no-`nnue` build,
/// whose params are actually zeroed.
pub fn has_network() -> bool {
    !RUNTIME_PARAMS.load(Ordering::Relaxed).is_null()
        || cfg!(all(feature = "nnue", net_embedded))
}

/// Decompress the zstd-compressed embedded network into a heap-allocated Box.
/// Called once on first access via OnceLock.
#[cfg(all(feature = "nnue", net_embedded))]
fn decompress_embedded() -> Box<NNUEParams> {
    use std::io::Read;
    // The decoder's default window ceiling (100 MB) guards against a hostile stream
    // claiming a window big enough to exhaust memory. This stream is our own network,
    // compressed by `build.rs` at level 22, which asks for a 128 MB window — so the
    // ceiling is raised to fit it rather than the compression being weakened.
    const WINDOW: u64 = 128 * 1024 * 1024;
    let mut decoder =
        ruzstd::decoding::StreamingDecoder::new_with_max_window_size(COMPRESSED_MODEL, WINDOW)
            .expect("Failed to init zstd decoder for embedded NNUE");
    // Decompress the whole stream (a footered network carries 16 trailing
    // bytes past the payload) and run the integrity gate on the file image.
    // A legacy embedded network passes silently — embedding it was a build
    // choice; a failed check is a corrupt or mismatched binary and panics
    // like the decoder errors above.
    let mut data = Vec::with_capacity(NNUE_FILE_SIZE + super::integrity::FOOTER_SIZE);
    decoder
        .read_to_end(&mut data)
        .expect("Failed to decompress embedded NNUE network");
    super::integrity::verify(&data)
        .unwrap_or_else(|e| panic!("Embedded NNUE network failed integrity check: {e}"));

    let mut params = NNUEParams::zeroed_box();
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr(),
            params.as_mut() as *mut NNUEParams as *mut u8,
            NNUE_FILE_SIZE,
        );
    }
    repermute_ft(&mut params);
    params
}

/// Load a network from a `.bin` file at runtime.
///
/// Accepts a footered file (payload + 16-byte integrity footer, verified), or
/// a legacy file of exactly `NNUE_FILE_SIZE` bytes plus up to 63 bytes of
/// Bullet alignment padding (size-checked only).
pub fn load_from_file(path: &str) -> Result<super::integrity::Provenance, String> {
    let data = std::fs::read(path).map_err(|e| format!("Failed to read {path}: {e}"))?;
    load_from_bytes(&data)
}

/// Install a network from the raw contents of a `.bin` file.
///
/// Split out from [`load_from_file`] because not every caller has a filesystem: a
/// browser build receives the weights over the network and never sees a path.
///
/// The previous runtime-loaded network, if any, is **deliberately leaked** (~37 MB).
/// [`params`] hands out `&'static NNUEParams`, and a search thread may be reading
/// through one at this very moment; freeing it here would be a use-after-free, and
/// knowing otherwise would take a reclamation protocol this does not have. In practice
/// nothing accumulates: the option is set between searches, and a browser loads once at
/// start-up and never again.
pub fn load_from_bytes(data: &[u8]) -> Result<super::integrity::Provenance, String> {
    // Integrity gate before anything is installed: a rejected network must
    // neither leak a box nor bump the generation. The hashes are over the
    // file bytes — repermute_ft below makes memory machine-dependent.
    let provenance = super::integrity::verify(data)?;

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

    Ok(provenance)
}

// ============================================================
// Streaming load: reserve, fill, publish
// ============================================================

/// A network reserved by [`reserve`] and not yet published.
static RESERVED: AtomicPtr<NNUEParams> = AtomicPtr::new(std::ptr::null_mut());

/// Reserves an empty network and hands back where its bytes belong.
///
/// [`load_from_bytes`] wants the file in a buffer of its own and copies it in, which
/// means both exist at once — 73 MB where 36.5 will do. That is a poor bargain anywhere,
/// and a bad one in a browser, where linear memory is never handed back once taken. So
/// the caller is given the network's own memory to fill, and calls [`publish_reserved`]
/// when it is full.
///
/// Exactly `NNUE_FILE_SIZE` bytes may be written. Reserving twice without publishing
/// leaks the first, which is why nothing but a host loading its one network should call
/// this.
pub fn reserve() -> *mut u8 {
    let params = NNUEParams::zeroed_box();
    let ptr = Box::into_raw(params);
    RESERVED.store(ptr, Ordering::Release);
    ptr as *mut u8
}

/// Publishes the network reserved by [`reserve`], now that its bytes are in place.
///
/// The previous network is left where it is, for the reason given on [`load_from_bytes`].
pub fn publish_reserved() -> Result<(), String> {
    let ptr = RESERVED.swap(std::ptr::null_mut(), Ordering::AcqRel);
    if ptr.is_null() {
        return Err(String::from("no network was reserved"));
    }
    // SAFETY: the pointer came from `Box::into_raw` in `reserve` and has been taken out
    // of RESERVED, so nothing else can reach it.
    let params = unsafe { &mut *ptr };
    repermute_ft(params);
    RUNTIME_PARAMS.store(ptr, Ordering::Release);
    NETWORK_GENERATION.fetch_add(1, Ordering::Release);
    Ok(())
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Lay a linear row out the way the trainer writes the file:
    /// `permuted[j] = original[FILE_PERM[j]]`, in blocks of 8 over 64-groups.
    fn linear_to_file(row: &[i16]) -> Vec<i16> {
        let mut out = vec![0i16; row.len()];
        for (chunk_o, chunk_i) in out.chunks_exact_mut(64).zip(row.chunks_exact(64)) {
            for (j, &p) in FILE_PERM.iter().enumerate() {
                chunk_o[j * 8..(j + 1) * 8].copy_from_slice(&chunk_i[p * 8..(p + 1) * 8]);
            }
        }
        out
    }

    #[test]
    fn the_file_layout_repermutes_to_each_tier_and_back() {
        let ramp: Vec<i16> = (0..640).collect();
        let file = linear_to_file(&ramp);

        // Linear target: undoing the file permutation must recover the ramp.
        let mut row = file.clone();
        repermute_row(&mut row, &[0]);
        assert_eq!(row, ramp, "file -> linear");

        // AVX2 target: same as laying the ramp out for the AVX2 packus directly.
        let mut row = file.clone();
        repermute_row(&mut row, &[0, 2, 1, 3]);
        let mut want = vec![0i16; 640];
        for (chunk_o, chunk_i) in want.chunks_exact_mut(32).zip(ramp.chunks_exact(32)) {
            for (j, &p) in [0usize, 2, 1, 3].iter().enumerate() {
                chunk_o[j * 8..(j + 1) * 8].copy_from_slice(&chunk_i[p * 8..(p + 1) * 8]);
            }
        }
        assert_eq!(row, want, "file -> avx2");
    }
}
