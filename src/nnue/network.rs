//! Network parameters struct and loading.
//!
//! Network file (GaiaNet-T7): the fields of `NNUEParams` end to end, byte for byte,
//! `NNUE_FILE_SIZE` bytes, then the 16-byte integrity footer.
//!
//! Loading:
//! - Compile-time: `include_bytes!` → zstd-compressed in binary
//! - Runtime: `load_from_file()` via UCI `setoption name EvalFile value <path>`
//!
//! The dense layers' weights are already in the dpbusd layout — no transform needed.
//!
//! The FT biases, however, carry the *packus* permutation written by the trainer,
//! and that permutation is tier-dependent: `activate_ft` writes the result of
//! `packus` straight to memory without a fixup `permute`, so the accumulator must be
//! pre-ordered to cancel the lane interleaving of that tier's `packus`. The trainer
//! bakes the AVX-512 pattern; `repermute_ft` converts it to whatever the tier
//! resolved for this machine needs. The i8 rows (piece squares, pawn pairs, threats)
//! are widened by an `unpack` that interleaves as `packus` does, so they are held in
//! logical order on every tier.

use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering};
use std::cell::OnceCell;
use std::sync::{Mutex, OnceLock};

use super::{A1_SIZE, GATED, L1_OUT, L1_SIZE, L2_OUT, PSQ_SIZE, STACKS, THREAT_INPUT_SIZE};

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

/// One feature-transformer row as the engine holds it: the `L1_SIZE` i8 weights the
/// accumulator kernels stream, aligned on a cache line.
#[repr(C, align(64))]
pub struct FtRow<T> {
    pub w: [T; L1_SIZE],
}

impl<T> FtRow<T> {
    /// The row's weights, for the kernels.
    #[inline(always)]
    pub fn as_ptr(&self) -> *const T {
        self.w.as_ptr()
    }
}

#[cfg(test)]
impl<T> FtRow<T> {
    pub fn new(w: [T; L1_SIZE]) -> Self {
        FtRow { w }
    }
}

/// A piece-square, pawn-pair or threat row: i8 weights (unit 256).
pub type ThreatRow = FtRow<i8>;

const _: () = assert!(size_of::<ThreatRow>() == L1_SIZE);

/// Complete NNUE network parameters, as the engine reads them.
///
/// All arrays are 64-byte aligned and all field sizes are multiples of 64 bytes, so
/// the struct is the file's payload byte for byte.
///
/// Architecture: K32Q2 piece squares + pawn pairs + threats (i8) → FT 1024 →
///   L1 (1024→32, dpbusd) → [x², x] → u8[64] →
///   L2 (64→64, dpbusd) → h = g² · v, i32[32] →
///   L3 (u8[64] and h[32] → 1) + skip, × 32 stacks → centipawns
#[repr(C, align(64))]
pub struct NNUEParams {
    /// Piece-square feature transformer rows: `[psq_feature]`, each `L1_SIZE` i8
    /// weights. PSQ_SIZE = 64 blocks × 704 = 45,056 features.
    pub ft_psq_weights: Aligned<[ThreatRow; PSQ_SIZE]>,

    /// Threat feature transformer rows: `[threat_feature]`, each `L1_SIZE` i8
    /// weights. Pawn pairs (4,560) then pairwise threats (59,808).
    pub ft_threat_weights: Aligned<[ThreatRow; THREAT_INPUT_SIZE]>,

    /// Feature transformer biases: initial accumulator values.
    pub ft_biases: Aligned<[i16; L1_SIZE]>,

    /// L1 sparse layer weights: `[stack][input_group_of_4][output*4+byte]`, the
    /// dpbusd layout (×128). Input: L1_SIZE u8 (FT activation output), output:
    /// L1_OUT i32.
    pub l1_weights: Aligned<[[[i8; L1_OUT * 4]; L1_SIZE / 4]; STACKS]>,

    /// L1 biases per stack: `[stack][neuron]`, unit 2^14.
    pub l1_biases: Aligned<[[i32; L1_OUT]; STACKS]>,

    /// L2 dense layer weights: `[stack][input_group_of_4][output*4+byte]` (×64).
    /// Input: the A1_SIZE u8 L1 activations, output: L2_OUT i32 (gates, then values).
    pub l2_weights: Aligned<[[[i8; L2_OUT * 4]; A1_SIZE / 4]; STACKS]>,

    /// L2 biases per stack: `[stack][neuron]`, unit 2^13.
    pub l2_biases: Aligned<[[i32; L2_OUT]; STACKS]>,

    /// L3 weights on the L1 activations: `[stack][input]` (×128).
    pub l3_weights_a: Aligned<[[i8; A1_SIZE]; STACKS]>,

    /// L3 weights on the gated L2 outputs: `[stack][input]` (×128), held in i32 for
    /// the products with the signed gated values.
    pub l3_weights_h: Aligned<[[i32; GATED]; STACKS]>,

    /// L3 biases per stack, unit 2^14.
    pub l3_biases: Aligned<[i32; STACKS]>,

    /// Network metadata: `[0]` the output scale (centipawns per unit), the rest zero.
    pub meta: Aligned<[i32; 16]>,
}

/// Exact byte count of the network payload.
pub const NNUE_FILE_SIZE: usize = PSQ_SIZE * L1_SIZE                  // ft_psq_weights: i8
    + THREAT_INPUT_SIZE * L1_SIZE                      // ft_threat_weights: i8
    + L1_SIZE * 2                                      // ft_biases: i16
    + STACKS * (L1_SIZE / 4) * (L1_OUT * 4)            // l1_weights: i8
    + STACKS * L1_OUT * 4                              // l1_biases: i32
    + STACKS * (A1_SIZE / 4) * (L2_OUT * 4)            // l2_weights: i8
    + STACKS * L2_OUT * 4                              // l2_biases: i32
    + STACKS * A1_SIZE                                 // l3_weights_a: i8
    + STACKS * GATED * 4                               // l3_weights_h: i32
    + STACKS * 4                                       // l3_biases: i32
    + 16 * 4; // meta: i32

const _: () = assert!(
    NNUE_FILE_SIZE == 113_250_496,
    "NNUE_FILE_SIZE must be 113,250,496"
);
// The file is the struct, byte for byte.
const _: () = assert!(NNUE_FILE_SIZE == size_of::<NNUEParams>());

/// Where each field starts in the network file: the struct's own offsets, the file
/// being the struct byte for byte.
pub(crate) mod file {
    use super::NNUEParams;
    use std::mem::offset_of;

    pub const PSQ: usize = offset_of!(NNUEParams, ft_psq_weights);
    pub const THREAT: usize = offset_of!(NNUEParams, ft_threat_weights);
    pub const BIASES: usize = offset_of!(NNUEParams, ft_biases);
    /// The L1 weights, and everything after them.
    pub const L1: usize = offset_of!(NNUEParams, l1_weights);
}

/// Bounds of a believable output scale (centipawns per network unit).
const SCALE_RANGE: std::ops::RangeInclusive<i32> = 1..=10_000;

impl NNUEParams {
    /// Allocate a zeroed network on the heap.
    pub fn zeroed_box() -> Box<Self> {
        // SAFETY: NNUEParams is repr(C) with only integer and float arrays.
        // Zero-initialized values are valid for all field types.
        unsafe { Box::<NNUEParams>::new_zeroed().assume_init() }
    }

    /// Centipawns per network unit, as the network file declares it. A network with
    /// no scale (all zero: no network loaded) evaluates with the default one.
    #[inline]
    pub fn output_scale(&self) -> i32 {
        let scale = self.meta.0[0];
        if scale == 0 { super::NETWORK_SCALE } else { scale }
    }
}

/// The output scale of a network file's payload, refused if it cannot be one.
pub(crate) fn file_scale(data: &[u8]) -> Result<i32, String> {
    let at = NNUE_FILE_SIZE - 16 * 4;
    let scale = i32::from_le_bytes(data[at..at + 4].try_into().unwrap());
    if SCALE_RANGE.contains(&scale) {
        Ok(scale)
    } else {
        Err(format!("output scale {scale} out of range {SCALE_RANGE:?}"))
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

// ============================================================
// Identity of the in-memory image
// ============================================================

/// Bump when the image gains something a fingerprint of the weight transform
/// cannot see — a second payload in the same mapping, say. The transform
/// itself is covered automatically by [`transform_fingerprint`].
const IMAGE_FORMAT_VERSION: u32 = 1;

/// A digest of what `repermute_ft` does, taken by doing it.
///
/// An SPRT runs two binaries that differ by construction, and both would share
/// an image if the name only described the architecture and the struct. A
/// patch that changes how rows are permuted, without moving a field, would
/// then have one binary reading the other's weights — wrong evaluation with
/// nothing to see. So the name carries the result of running the real
/// transform over canary rows, one i16 bias row and one i8 weight row: change
/// either permutation and the name changes with it, whether or not anybody
/// remembered to say so.
fn transform_fingerprint() -> u32 {
    static FINGERPRINT: OnceLock<u32> = OnceLock::new();
    *FINGERPRINT.get_or_init(|| {
        let mut row = [0i16; L1_SIZE];
        for (i, v) in row.iter_mut().enumerate() {
            *v = (i as i16).wrapping_mul(7919).wrapping_add(13);
        }
        if let Some(perm) = target_perm() {
            repermute_row(&mut row, perm);
        }
        let mut threat_row = [0i8; L1_SIZE];
        for (i, v) in threat_row.iter_mut().enumerate() {
            *v = (i as i8).wrapping_mul(31).wrapping_add(5);
        }
        repermute_row(&mut threat_row, THREAT_PERM);
        let bytes: Vec<u8> = row
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .chain(threat_row.iter().map(|&v| v as u8))
            .collect();
        super::integrity::fnv1a64(&bytes) as u32
    })
}

/// Digest of everything that decides what the 38 MB of memory look like,
/// beyond the architecture (which `integrity::ARCH_HASH` already names).
///
/// Two binaries may share one image only if they agree here: same struct
/// shape, same field offsets, same post-copy transform. Worktree builds of
/// neighbouring commits are exactly the case this guards.
pub(crate) const LAYOUT_HASH: u32 = {
    use super::integrity::h32;
    let mut h = 0x811c_9dc5u32;
    h = h32(h, size_of::<NNUEParams>() as u32);
    h = h32(h, align_of::<NNUEParams>() as u32);
    h = h32(h, std::mem::offset_of!(NNUEParams, ft_psq_weights) as u32);
    h = h32(
        h,
        std::mem::offset_of!(NNUEParams, ft_threat_weights) as u32,
    );
    h = h32(h, std::mem::offset_of!(NNUEParams, ft_biases) as u32);
    h = h32(h, std::mem::offset_of!(NNUEParams, l1_weights) as u32);
    h = h32(h, std::mem::offset_of!(NNUEParams, l1_biases) as u32);
    h = h32(h, std::mem::offset_of!(NNUEParams, l2_weights) as u32);
    h = h32(h, std::mem::offset_of!(NNUEParams, l2_biases) as u32);
    h = h32(h, std::mem::offset_of!(NNUEParams, l3_weights_a) as u32);
    h = h32(h, std::mem::offset_of!(NNUEParams, l3_weights_h) as u32);
    h = h32(h, std::mem::offset_of!(NNUEParams, l3_biases) as u32);
    h = h32(h, std::mem::offset_of!(NNUEParams, meta) as u32);
    h32(h, IMAGE_FORMAT_VERSION)
};

/// Short tag of the permutation a tier needs, for the image name.
fn perm_tag(perm: PermKind) -> &'static str {
    match perm {
        PermKind::File512 => "f512",
        PermKind::Avx2 => "avx2",
        PermKind::Linear => "lin",
    }
}

/// The name under which this machine's image of a network is shared.
///
/// Everything that could make two images differ byte for byte is in the name:
/// the architecture, the weights themselves, the tier's permutation (an AVX2
/// process and an AVX-512 process hold genuinely different bytes of the same
/// network), the struct layout, and the transform that fills it. Anything left
/// out would be a process reading weights it did not build.
fn image_name(content_hash: u64) -> String {
    format!(
        "nnue-{:08x}-{content_hash:016x}-{}-{:08x}",
        super::integrity::ARCH_HASH,
        perm_tag(active_perm()),
        build_tag(),
    )
}

/// Struct shape and weight transform in one field.
///
/// The sharing layer already refuses to let two executables meet, which covers
/// both of these for any build made the ordinary way. This is what is left if
/// that identity ever degrades to a bare path — a rebuild in place would reuse
/// the path, but not the layout or the transform if either moved.
fn build_tag() -> u32 {
    use super::integrity::h32;
    h32(LAYOUT_HASH, transform_fingerprint())
}

/// Convert one row of `L1_SIZE` elements from the file's permutation to the target's.
///
/// Both group sizes (64 for the file, 32 or 8 for the target) divide `L1_SIZE` = 1024,
/// and the FT half boundary at 512 is a multiple of 64 — so the left and right halves
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

/// The permutation to apply to a row on this machine, or `None` when the
/// file's own layout is already the right one.
fn target_perm() -> Option<&'static [usize]> {
    match active_perm() {
        PermKind::File512 => None,
        PermKind::Avx2 => Some(&[0, 2, 1, 3]),
        PermKind::Linear => Some(&[0]),
    }
}

/// The i8 rows (piece squares, pawn pairs, threats) are held in logical order on
/// every tier.
///
/// The kernels widen them with `widen_i8_pair` / `widen_i8`, whose x86 forms
/// interleave each register exactly as `packus` does, so a row in logical order
/// lands on the accumulator's packus-ordered lanes; the other tiers do not
/// interleave at all.
const THREAT_PERM: &[usize] = &[0];

/// Re-permute every FT array to the layout of the tier resolved for this machine:
/// the i8 rows to [`THREAT_PERM`], the biases to the order `activate_ft`'s `packus`
/// needs (nothing to do on the AVX-512 tiers, whose layout is the file's).
fn repermute_ft(params: &mut NNUEParams) {
    for row in params.ft_psq_weights.0.iter_mut() {
        repermute_row(&mut row.w, THREAT_PERM);
    }
    for row in params.ft_threat_weights.0.iter_mut() {
        repermute_row(&mut row.w, THREAT_PERM);
    }
    let Some(target_perm) = target_perm() else {
        return;
    };
    repermute_row(&mut params.ft_biases.0, target_perm);
}

/// One row from the file's layout to logical order, in place: column `i` of the
/// result is the neuron whose product lands on output byte `i` (left half) or
/// pairs with it (right half).
#[cfg(test)]
pub(super) fn file_to_linear<T: Copy + Default>(row: &mut [T]) {
    repermute_row(row, &[0]);
}

/// One row from logical order to the file's layout, in place — what the trainer
/// writes: `permuted[j] = original[FILE_PERM[j]]`, in blocks of 8 over 64-groups.
pub(super) fn linear_to_file<T: Copy + Default>(row: &mut [T]) {
    const FILE_GROUP: usize = PERM_BLOCK * FILE_PERM.len(); // 64
    debug_assert_eq!(row.len() % FILE_GROUP, 0);
    let mut tmp = [T::default(); FILE_GROUP];
    for chunk in row.chunks_exact_mut(FILE_GROUP) {
        for (j, &p) in FILE_PERM.iter().enumerate() {
            tmp[j * PERM_BLOCK..(j + 1) * PERM_BLOCK]
                .copy_from_slice(&chunk[p * PERM_BLOCK..(p + 1) * PERM_BLOCK]);
        }
        chunk.copy_from_slice(&tmp);
    }
}

/// For each column of an FT row as this machine holds it in memory, the logical
/// neuron it carries — the output byte `activate_ft` writes its product to.
///
/// Memory is not in logical order on the tiers whose `packus` interleaves lanes:
/// the weights are pre-ordered to cancel it, so the output comes out logical while
/// the accumulator does not. Anything that reads the accumulator and reasons about
/// output bytes goes through this map.
pub(super) fn memory_to_logical() -> [u16; L1_SIZE] {
    let mut row: [u16; L1_SIZE] = std::array::from_fn(|i| i as u16);
    linear_to_file(&mut row);
    if let Some(perm) = target_perm() {
        repermute_row(&mut row, perm);
    }
    row
}

// ============================================================
// Global parameters with runtime loading
// ============================================================

// ============================================================
// Building the in-memory image
// ============================================================

/// Turn the bytes of a network file into the bytes the engine reads.
///
/// The file goes to the image, and [`fill_image_in_place`] takes it from there: the
/// streaming load of [`reserve`] has the host write the file into the image itself,
/// and both paths must end in the same bytes. One conversion, because two would be
/// two chances for a private copy, a shared image and the browser's to disagree —
/// and a disagreement there is silently wrong evaluation, not a crash.
fn fill_image(data: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(dst.len(), size_of::<NNUEParams>());
    debug_assert!(data.len() >= NNUE_FILE_SIZE);
    dst.copy_from_slice(&data[..NNUE_FILE_SIZE]);
    fill_image_in_place(dst);
}

/// Turn a network file written into `dst` into the image the engine reads, in the
/// same memory. The file is the struct byte for byte; only the FT changes order.
fn fill_image_in_place(dst: &mut [u8]) {
    debug_assert_eq!(dst.len(), size_of::<NNUEParams>());
    debug_assert_eq!(dst.as_ptr() as usize % align_of::<NNUEParams>(), 0);
    // SAFETY: `dst` is exactly one `NNUEParams` and correctly aligned, and
    // every bit pattern of its integer arrays is valid.
    let params = unsafe { &mut *(dst.as_mut_ptr() as *mut NNUEParams) };
    repermute_ft(params);
}

/// A private copy of the network, the way the engine has always made one.
fn build_private(data: &[u8]) -> Box<NNUEParams> {
    let mut params = NNUEParams::zeroed_box();
    // SAFETY: the box holds exactly one `NNUEParams`; this views it as its bytes.
    let dst = unsafe {
        std::slice::from_raw_parts_mut(
            params.as_mut() as *mut NNUEParams as *mut u8,
            size_of::<NNUEParams>(),
        )
    };
    fill_image(data, dst);
    params
}

/// The file bytes of a structured (deterministic) random network, every weight in
/// the range its type stores, a believable output scale: the network the tests
/// evaluate when the build embeds none.
#[cfg(test)]
pub(crate) fn random_network_bytes(seed: u64) -> Vec<u8> {
    let mut state = seed | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut params = NNUEParams::zeroed_box();
    for row in params.ft_psq_weights.0.iter_mut().chain(params.ft_threat_weights.0.iter_mut()) {
        row.w.iter_mut().for_each(|w| *w = next() as i8);
    }
    params.ft_biases.0.iter_mut().for_each(|b| *b = (next() % 201) as i16 - 100);
    for s in 0..STACKS {
        for g in params.l1_weights.0[s].iter_mut() {
            g.iter_mut().for_each(|w| *w = next() as i8);
        }
        params.l1_biases.0[s].iter_mut().for_each(|b| *b = (next() % 60_001) as i32 - 30_000);
        for g in params.l2_weights.0[s].iter_mut() {
            g.iter_mut().for_each(|w| *w = next() as i8);
        }
        params.l2_biases.0[s].iter_mut().for_each(|b| *b = (next() % 20_001) as i32 - 10_000);
        params.l3_weights_a.0[s].iter_mut().for_each(|w| *w = next() as i8);
        params.l3_weights_h.0[s].iter_mut().for_each(|w| *w = (next() % 255) as i32 - 127);
        params.l3_biases.0[s] = (next() % 20_001) as i32 - 10_000;
    }
    params.meta.0[0] = super::NETWORK_SCALE;
    // SAFETY: the struct is plain integer arrays, its bytes are the file's payload.
    let bytes = unsafe {
        std::slice::from_raw_parts(params.as_ref() as *const NNUEParams as *const u8, NNUE_FILE_SIZE)
    };
    let mut data = bytes.to_vec();
    data.extend_from_slice(&super::integrity::footer(&data));
    data
}

/// A private copy of a network from the bytes of its file, installed nowhere: a test
/// evaluates with it without swapping the network under every search the other tests
/// of the process are running, or turning on the NNUE path halfway through one.
#[cfg(test)]
pub(crate) fn params_from_bytes(data: &[u8]) -> Box<NNUEParams> {
    debug_assert!(data.len() >= NNUE_FILE_SIZE, "a network file of {} bytes", data.len());
    build_private(data)
}

/// What the weights the engine is reading are actually backed by.
pub use crate::shm::Backing as NetBacking;

/// Where the current weights came from, so the sharing policy can be changed
/// after the fact without the caller having to hand them over again.
// `Embedded` exists only where a network is baked in; the variant is still
// part of the type so the policy switch has one shape everywhere.
#[cfg_attr(not(all(feature = "nnue", net_embedded)), allow(dead_code))]
enum NetSource {
    Embedded,
    /// The file as it was named, and what it looked like when it was read.
    File {
        path: String,
        stamp: Option<FileStamp>,
    },
}

/// Length and modification time of a network file: enough to tell, without reading
/// it again, that the file named again is still the one that was read. `None` where
/// the platform has no modification time, which makes every load a fresh one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FileStamp {
    len: u64,
    modified: std::time::SystemTime,
}

impl FileStamp {
    fn of(meta: &std::fs::Metadata) -> Option<FileStamp> {
        Some(FileStamp {
            len: meta.len(),
            modified: meta.modified().ok()?,
        })
    }
}

static SOURCE: Mutex<Option<NetSource>> = Mutex::new(None);
static BACKING: Mutex<Option<NetBacking>> = Mutex::new(None);

/// Whether to look for shared weights. Seeded from the environment, because
/// the network is built during start-up, before the first UCI option arrives.
static SHARED_POLICY: OnceLock<AtomicBool> = OnceLock::new();

fn shared_policy() -> &'static AtomicBool {
    SHARED_POLICY.get_or_init(|| AtomicBool::new(crate::shm::enabled()))
}

/// How the weights the engine is reading are backed, if any are loaded yet.
pub fn backing() -> Option<NetBacking> {
    BACKING.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// A network file's bytes, in hand or produced on first use: a process that maps the
/// image a peer built never reads them, so the embedded network is decoded only by
/// the process that fills an image.
enum NetBytes<'a> {
    Ready(&'a [u8]),
    #[cfg_attr(not(all(feature = "nnue", net_embedded)), allow(dead_code))]
    Lazy(OnceCell<Vec<u8>>, fn() -> Vec<u8>),
}

impl NetBytes<'_> {
    fn get(&self) -> &[u8] {
        match self {
            NetBytes::Ready(data) => data,
            NetBytes::Lazy(cell, produce) => cell.get_or_init(produce),
        }
    }
}

/// Install a network, sharing its image with the other engine processes on
/// this machine when that is possible.
///
/// Falls back to a private copy on any trouble at all: sharing is a throughput
/// optimisation for test lanes, never a condition for playing chess.
fn materialise(data: &NetBytes, content_hash: u64) -> &'static NNUEParams {
    let len = size_of::<NNUEParams>();
    MASTER_HASH.store(content_hash, Ordering::Relaxed);

    let mut private_reason = if crate::shm::supported() {
        String::from("sharing turned off")
    } else {
        String::from("no sharing on this platform")
    };
    if shared_policy().load(Ordering::Relaxed) {
        // On a machine with several memory nodes the image is placed where it was
        // first written: named after the node this process starts on, so that a
        // process pinned to another socket builds its own instead of reading this
        // one across the link.
        let mut name = image_name(content_hash);
        if crate::numa::several_hardware_nodes() {
            name = format!("{name}-n{}", crate::numa::startup_hardware_node());
        }
        match crate::shm::acquire(&name, len, &mut |dst| fill_image(data.get(), dst)) {
            Ok(mapped) => {
                let bytes = mapped.bytes();
                debug_assert_eq!(bytes.len(), len);
                debug_assert_eq!(bytes.as_ptr() as usize % align_of::<NNUEParams>(), 0);

                // The image's name pins the architecture, the weights, the
                // tier's permutation and the struct layout — but not a change
                // to what this module writes *into* that layout. Checking mode
                // rebuilds and compares, so such a change is caught here
                // instead of becoming silently wrong evaluation.
                if crate::shm::verify_requested() {
                    let mine = build_private(data.get());
                    // SAFETY: viewing one `NNUEParams` as its bytes.
                    let mine = unsafe {
                        std::slice::from_raw_parts(
                            mine.as_ref() as *const NNUEParams as *const u8,
                            len,
                        )
                    };
                    if mine != bytes {
                        crate::outerr!(
                            "info string NNUE network: shared image disagrees with this build, \
                             falling back to a private copy (bump IMAGE_FORMAT_VERSION)"
                        );
                        return install_private(
                            build_private(data.get()),
                            String::from("image mismatch"),
                        );
                    }
                }

                let backing = NetBacking::Shared {
                    origin: mapped.origin,
                    peers: crate::shm::peers_of(&name),
                };
                crate::outerr!("info string NNUE network: {backing}");
                *BACKING.lock().unwrap_or_else(|e| e.into_inner()) = Some(backing);
                // The mapping is never unmapped, so this reference outlives
                // every search that could be reading through it.
                // SAFETY: `bytes` is one correctly aligned, immutable `NNUEParams`.
                return unsafe { &*(bytes.as_ptr() as *const NNUEParams) };
            }
            Err(e) => private_reason = e,
        }
    }
    install_private(build_private(data.get()), private_reason)
}

/// Leak a private copy and record why it is not shared.
fn install_private(params: Box<NNUEParams>, reason: String) -> &'static NNUEParams {
    let backing = NetBacking::Private { reason };
    if crate::shm::supported() {
        crate::outerr!("info string NNUE network: {backing}");
    }
    *BACKING.lock().unwrap_or_else(|e| e.into_inner()) = Some(backing);
    Box::leak(params)
}

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
static DEFAULT_PARAMS: OnceLock<&'static NNUEParams> = OnceLock::new();

#[cfg(not(all(feature = "nnue", net_embedded)))]
static DEFAULT_PARAMS_ZEROED: OnceLock<Box<NNUEParams>> = OnceLock::new();

/// Get the active network parameters (runtime-loaded or default): the one shared copy.
///
/// A search pinned to a NUMA node evaluates with its node's copy instead, which its
/// per-thread state holds (see [`replica`] and `Network::use_params`).
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
    !RUNTIME_PARAMS.load(Ordering::Relaxed).is_null() || cfg!(all(feature = "nnue", net_embedded))
}

/// Decompress the embedded network and check it, yielding the file's bytes.
#[cfg(all(feature = "nnue", net_embedded))]
fn embedded_file_bytes() -> Vec<u8> {
    // Decompress the whole stream (the footer is 16 bytes past the payload) and
    // run the integrity gate on the file image. `build.rs` refuses a network
    // without a footer, so a failed check is a corrupt or mismatched binary, and
    // panics like the decoder errors.
    let mut data = Vec::with_capacity(NNUE_FILE_SIZE + super::integrity::FOOTER_SIZE);
    decode_embedded(&mut data);
    super::integrity::verify(&data)
        .unwrap_or_else(|e| panic!("Embedded NNUE network failed integrity check: {e}"));
    data
}

/// Native builds that link the C zstd (for GaiaTB) decode with it: about three
/// times faster than ruzstd on the 87 MB image. The image size is known, so the
/// single-pass decoder writes straight into the vector's capacity.
#[cfg(all(feature = "nnue", net_embedded, feature = "gaiatb", not(target_arch = "wasm32")))]
fn decode_embedded(data: &mut Vec<u8>) {
    zstd::bulk::Decompressor::new()
        .and_then(|mut d| d.decompress_to_buffer(COMPRESSED_MODEL, data))
        .expect("Failed to decompress embedded NNUE network");
}

/// Pure-Rust fallback (wasm, or builds without GaiaTB).
#[cfg(all(feature = "nnue", net_embedded, not(all(feature = "gaiatb", not(target_arch = "wasm32")))))]
fn decode_embedded(data: &mut Vec<u8>) {
    use std::io::Read;
    // The decoder's default window ceiling (100 MB) guards against a hostile stream
    // claiming a window big enough to exhaust memory. This stream is our own network,
    // compressed by `build.rs` at level 22, which asks for a 128 MB window — so the
    // ceiling is raised to fit it rather than the compression being weakened.
    const WINDOW: u64 = 128 * 1024 * 1024;
    let mut decoder =
        ruzstd::decoding::StreamingDecoder::new_with_max_window_size(COMPRESSED_MODEL, WINDOW)
            .expect("Failed to init zstd decoder for embedded NNUE");
    decoder
        .read_to_end(data)
        .expect("Failed to decompress embedded NNUE network");
}

/// Install the embedded network. Called once on first access via OnceLock.
#[cfg(all(feature = "nnue", net_embedded))]
fn decompress_embedded() -> &'static NNUEParams {
    *SOURCE.lock().unwrap_or_else(|e| e.into_inner()) = Some(NetSource::Embedded);
    // A footered network declares its digest in the tail the build kept apart: the
    // image is named, and a peer's mapped, without decoding anything.
    let data = NetBytes::Lazy(OnceCell::new(), embedded_file_bytes);
    let hash = super::integrity::declared_hash(model_meta::EMBEDDED_LEN, model_meta::EMBEDDED_TAIL)
        .unwrap_or_else(|| super::integrity::content_hash(data.get()));
    let params = materialise(&data, hash);
    debug_assert!(
        !matches!(&data, NetBytes::Lazy(cell, _) if cell.get().is_some_and(|d| super::integrity::content_hash(d) != hash)),
        "embedded network: the declared digest is not the decoded file's"
    );
    params
}

/// The embedded file's length and last bytes, written by `build.rs`.
#[cfg(all(feature = "nnue", net_embedded))]
mod model_meta {
    include!(concat!(env!("OUT_DIR"), "/model_meta.rs"));
}

/// Puts the embedded network back in service if `EvalFile` replaced it: what
/// `setoption name EvalFile value <internal>`, the option's default, asks for. Returns
/// whether the network changed, in which case the caller must clear what was learned
/// with the other one.
///
/// Installed like any network, through [`materialise`] — so the name of the shared
/// image and of the node copies follows, and the sharing policy is respected — rather
/// than by forgetting the runtime network, which the embedded one may itself be.
#[cfg(all(feature = "nnue", net_embedded))]
pub fn load_internal() -> bool {
    if !matches!(
        *SOURCE.lock().unwrap_or_else(|e| e.into_inner()),
        Some(NetSource::File { .. })
    ) {
        return false;
    }
    let data = embedded_file_bytes();
    let params = materialise(&NetBytes::Ready(&data), super::integrity::content_hash(&data));
    // The file's network is left where it is, for the reason given on `load_from_bytes`.
    RUNTIME_PARAMS.store(
        params as *const NNUEParams as *mut NNUEParams,
        Ordering::Release,
    );
    *SOURCE.lock().unwrap_or_else(|e| e.into_inner()) = Some(NetSource::Embedded);
    NETWORK_GENERATION.fetch_add(1, Ordering::Release);
    refresh_replicas();
    true
}

/// Without an embedded network there is nothing to go back to.
#[cfg(not(all(feature = "nnue", net_embedded)))]
pub fn load_internal() -> bool {
    false
}

/// Checks that `data`, a network read again from where the one in service came from,
/// is still that network: whole, sound, and with the same weights.
///
/// Changing the sharing policy reinstalls the weights from their source, and keeps the
/// table, the accumulator caches and the generation as they are — which is only right
/// if the weights are the same. A file replaced since it was loaded would otherwise
/// swap networks under a warm table without a word, and a truncated one would panic.
fn check_same_network(data: &[u8], in_service: u64) -> Result<(), String> {
    super::integrity::verify(data)?;
    if super::integrity::content_hash(data) != in_service {
        return Err(String::from(
            "the network has changed since it was loaded; load it again with EvalFile",
        ));
    }
    Ok(())
}

/// Change whether the weights are shared with the other engine processes.
///
/// The weights themselves do not change, so the transposition table and the
/// accumulator caches stay valid and the generation counter is left alone —
/// only what memory they are read out of changes.
pub fn set_shared(enabled: bool) -> Result<NetBacking, String> {
    let policy = shared_policy();
    if !crate::shm::supported() {
        return Err(String::from(
            "no sharing on this platform (Linux and Windows only)",
        ));
    }
    if policy.load(Ordering::Relaxed) == enabled {
        return backing().ok_or_else(|| String::from("no network loaded yet"));
    }

    // Read and checked before anything changes: a refusal leaves the policy and the
    // backing as they were, so the same request can be made again.
    let in_service = MASTER_HASH.load(Ordering::Relaxed);
    let source = SOURCE.lock().unwrap_or_else(|e| e.into_inner());
    let data = match source.as_ref() {
        Some(NetSource::File { path, .. }) => {
            let data = std::fs::read(path).map_err(|e| format!("Failed to re-read {path}: {e}"))?;
            check_same_network(&data, in_service).map_err(|e| format!("{path}: {e}"))?;
            data
        }
        #[cfg(all(feature = "nnue", net_embedded))]
        Some(NetSource::Embedded) => {
            let data = embedded_file_bytes();
            check_same_network(&data, in_service)?;
            data
        }
        #[cfg(not(all(feature = "nnue", net_embedded)))]
        Some(NetSource::Embedded) => return Err(String::from("no embedded network")),
        None => return Err(String::from("no network loaded yet")),
    };
    drop(source);
    policy.store(enabled, Ordering::Relaxed);

    let params = materialise(&NetBytes::Ready(&data), in_service);
    // The old backing is left where it is, for the reason given on
    // `load_from_bytes`: a search thread may be reading through it.
    RUNTIME_PARAMS.store(
        params as *const NNUEParams as *mut NNUEParams,
        Ordering::Release,
    );
    refresh_replicas();
    backing().ok_or_else(|| String::from("no backing recorded"))
}

/// Load a network from a `.bin` file at runtime.
///
/// Accepts the payload followed by its 16-byte integrity footer, verified.
pub fn load_from_file(path: &str) -> Result<(), String> {
    use std::io::Read;
    let read_error = |e: std::io::Error| format!("Failed to read {path}: {e}");
    let mut file = std::fs::File::open(path).map_err(read_error)?;
    // Taken before the bytes are read: a file rewritten meanwhile looks changed the
    // next time it is named, and is read again.
    let meta = file.metadata().ok();
    let stamp = meta.as_ref().and_then(FileStamp::of);
    let mut data = Vec::with_capacity(meta.map_or(0, |m| m.len() as usize));
    file.read_to_end(&mut data).map_err(read_error)?;
    load_from_bytes(&data)?;
    // Remembered so the sharing policy can be changed later without the file
    // having to be named again, and so that naming it again changes nothing.
    *SOURCE.lock().unwrap_or_else(|e| e.into_inner()) = Some(NetSource::File {
        path: path.to_string(),
        stamp,
    });
    Ok(())
}

/// Whether the network in service was loaded from `path`, and the file is still the
/// one that was read: same length, same modification time.
///
/// A match runner names the network again before every game. Loading it again would
/// read and check the whole file, and keep one more copy of the weights for good (see
/// [`load_from_bytes`]): memory without sharing, address space with it.
pub fn is_loaded_from(path: &str) -> bool {
    let source = SOURCE.lock().unwrap_or_else(|e| e.into_inner());
    let Some(NetSource::File {
        path: loaded,
        stamp: Some(stamp),
    }) = source.as_ref()
    else {
        return false;
    };
    loaded == path
        && std::fs::metadata(path)
            .ok()
            .as_ref()
            .and_then(FileStamp::of)
            == Some(*stamp)
}

/// Install a network from the raw contents of a `.bin` file.
///
/// Split out from [`load_from_file`] because not every caller has a filesystem: a
/// browser build receives the weights over the network and never sees a path.
///
/// The previous runtime-loaded network, if any, is **deliberately leaked** (~85 MB).
/// [`params`] hands out `&'static NNUEParams`, and a search thread may be reading
/// through one at this very moment; freeing it here would be a use-after-free, and
/// knowing otherwise would take a reclamation protocol this does not have. In practice
/// nothing accumulates: the option is set between searches, and a browser loads once at
/// start-up and never again.
pub fn load_from_bytes(data: &[u8]) -> Result<(), String> {
    // Integrity gate before anything is installed: a rejected network must
    // neither leak a box nor bump the generation. The hashes are over the
    // file bytes — repermute_ft below makes memory machine-dependent.
    super::integrity::verify(data)?;

    let params = materialise(&NetBytes::Ready(data), super::integrity::content_hash(data));

    // The pointer is stored as mutable because that is what `AtomicPtr` holds,
    // but nothing ever writes through it: shared images are sealed read-only by
    // the kernel, and a private one is only written before it is published.
    RUNTIME_PARAMS.store(
        params as *const NNUEParams as *mut NNUEParams,
        Ordering::Release,
    );

    // Increment generation so Finny table caches are invalidated.
    NETWORK_GENERATION.fetch_add(1, Ordering::Release);
    refresh_replicas();

    Ok(())
}

// ============================================================
// One copy per memory node
// ============================================================

/// Content hash of the network last materialised, which names its node images.
static MASTER_HASH: AtomicU64 = AtomicU64::new(0);

/// Node copies ever built: `(master copy, hardware node, replica)`. Never freed — a
/// thread may hold a reference into any of them — and reused when the same network
/// is replicated on the same node again, so switching layouts back and forth costs
/// nothing.
static REPLICA_CACHE: Mutex<Vec<(usize, usize, usize)>> = Mutex::new(Vec::new());

/// The layout replicas are wanted for, and the copy for each hardware node in use.
struct ReplicaSet {
    config: Option<crate::numa::NumaConfig>,
    active: Vec<(usize, usize)>,
}

static REPLICAS: Mutex<ReplicaSet> = Mutex::new(ReplicaSet {
    config: None,
    active: Vec::new(),
});

/// Copies the network onto every hardware memory node of `config` that a logical node
/// uses, each copy built by a thread pinned there so that its pages are local. `None`,
/// or a layout that needs no replication, leaves every thread on the shared copy.
pub fn replicate(config: Option<&crate::numa::NumaConfig>) {
    let mut set = REPLICAS.lock().unwrap_or_else(|e| e.into_inner());
    set.config = config.filter(|c| c.requires_replication()).cloned();
    set.active = build_replicas(set.config.as_ref());
}

/// Rebuild the copies after the network itself changed.
fn refresh_replicas() {
    let mut set = REPLICAS.lock().unwrap_or_else(|e| e.into_inner());
    set.active = build_replicas(set.config.as_ref());
}

fn build_replicas(config: Option<&crate::numa::NumaConfig>) -> Vec<(usize, usize)> {
    let Some(config) = config else {
        return Vec::new();
    };
    let master = params() as *const NNUEParams as usize;
    let mut cache = REPLICA_CACHE.lock().unwrap_or_else(|e| e.into_inner());
    let mut active = Vec::new();
    for sys in config.system_nodes() {
        if let Some(&(_, _, p)) = cache.iter().find(|&&(m, n, _)| m == master && n == sys) {
            active.push((sys, p));
            continue;
        }
        let Some(logical) = (0..config.node_count()).find(|&n| config.system_node_of(n) == sys)
        else {
            continue;
        };
        let p = crate::numa::run_on_node(config, logical, || {
            build_replica(master, sys) as *const NNUEParams as usize
        });
        cache.push((master, sys, p));
        active.push((sys, p));
    }
    active
}

/// A copy of the master network made on the calling thread's node: shared with the
/// other processes on that node when sharing is on, private otherwise.
fn build_replica(master: usize, sys_node: usize) -> &'static NNUEParams {
    let len = size_of::<NNUEParams>();
    // SAFETY: the master copy is never freed or written once published.
    let src = unsafe { std::slice::from_raw_parts(master as *const u8, len) };
    if shared_policy().load(Ordering::Relaxed) {
        let name = format!(
            "{}-{:06x}n{sys_node}",
            image_name(MASTER_HASH.load(Ordering::Relaxed)),
            crate::numa::hardware_signature() & 0xff_ffff
        );
        if let Ok(mapped) = crate::shm::acquire(&name, len, &mut |dst| dst.copy_from_slice(src)) {
            let bytes = mapped.bytes();
            if bytes == src {
                // SAFETY: one aligned, immutable NNUEParams that is never unmapped.
                return unsafe { &*(bytes.as_ptr() as *const NNUEParams) };
            }
        }
    }
    let mem = crate::largemem::LargeMem::zeroed(len, false);
    debug_assert_eq!(mem.as_ptr() as usize % align_of::<NNUEParams>(), 0);
    // SAFETY: the block is `len` bytes, page aligned, and freshly allocated.
    unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), mem.as_ptr(), len) };
    let ptr = mem.as_ptr() as *const NNUEParams;
    // Never released: search threads hold `&'static` references into it.
    std::mem::forget(mem);
    // SAFETY: an initialised copy of the master network that lives forever.
    unsafe { &*ptr }
}

/// The copy on hardware node `sys_node`, if one was made; `None` means the shared copy.
pub fn replica(sys_node: usize) -> Option<&'static NNUEParams> {
    let set = REPLICAS.lock().unwrap_or_else(|e| e.into_inner());
    set.active
        .iter()
        .find(|&&(n, _)| n == sys_node)
        // SAFETY: replicas are never freed, and are identical to the shared copy.
        .map(|&(_, p)| unsafe { &*(p as *const NNUEParams) })
}

/// How many node copies are in use (1 when every thread reads the shared copy).
pub fn replica_count() -> usize {
    REPLICAS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .active
        .len()
        .max(1)
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
/// Exactly `NNUE_FILE_SIZE` bytes may be written, as the file holds them, and
/// [`publish_reserved`] converts them in place. Reserving twice without publishing
/// leaks the first, which is why nothing but a host loading its one network should
/// call this.
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
    // of RESERVED, so nothing else can reach it; this views the network as its bytes.
    let image = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u8, size_of::<NNUEParams>()) };
    fill_image_in_place(image);
    RUNTIME_PARAMS.store(ptr, Ordering::Release);
    NETWORK_GENERATION.fetch_add(1, Ordering::Release);
    refresh_replicas();
    Ok(())
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    /// The image name is decorated by the sharing layer with the user, the
    /// executable and a socket role before it is bound, and an abstract socket
    /// name cannot exceed 107 bytes. Keep a wide margin: overrunning it turns
    /// sharing off silently on machines with long uids or pids.
    #[test]
    fn the_image_name_leaves_room_for_the_socket_decoration() {
        let name = super::image_name(0xffff_ffff_ffff_ffff);
        assert!(
            name.len() <= 60,
            "image name is {} bytes: {name}",
            name.len()
        );
        assert!(!name.contains('\0'));
    }

    use super::*;

    /// A network read again to change the sharing policy is accepted only if it is the
    /// one in service. A file rewritten since `EvalFile` used to be installed without a
    /// word under the old network's table and caches, and a truncated one panicked.
    #[test]
    fn a_reread_network_must_be_the_one_in_service() {
        let footered = |payload: &mut Vec<u8>| {
            let footer = super::super::integrity::footer(payload);
            payload.extend_from_slice(&footer);
        };
        let mut data = vec![0u8; NNUE_FILE_SIZE];
        data[0] = 1;
        let at = NNUE_FILE_SIZE - 64;
        data[at..at + 4].copy_from_slice(&287i32.to_le_bytes());
        let mut rewritten = data.clone();
        rewritten[1] = 2;
        footered(&mut data);
        footered(&mut rewritten);
        let in_service = super::super::integrity::content_hash(&data);
        assert_eq!(check_same_network(&data, in_service), Ok(()));

        let refused = check_same_network(&rewritten, in_service).unwrap_err();
        assert!(refused.contains("EvalFile"), "{refused}");

        assert!(check_same_network(&data[..1000], in_service).is_err());
    }

    /// `EvalFile` naming the network in service again reads nothing, unless the file
    /// was rewritten: its stamp follows the rewrite.
    #[test]
    fn a_rewritten_network_file_no_longer_matches_its_stamp() {
        let path =
            std::env::temp_dir().join(format!("gaia-net-stamp-{}.bin", std::process::id()));
        let stamp = |p: &std::path::Path| FileStamp::of(&std::fs::metadata(p).unwrap());
        std::fs::write(&path, [0u8; 16]).unwrap();
        let read = stamp(&path);
        assert!(read.is_some(), "no modification time on this platform");
        assert_eq!(stamp(&path), read, "the same file, untouched");
        std::fs::write(&path, [0u8; 32]).unwrap();
        assert_ne!(stamp(&path), read, "the file was rewritten");
        std::fs::remove_file(&path).unwrap();
    }

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
        let ramp: Vec<i16> = (0..L1_SIZE as i16).collect();
        let file = linear_to_file(&ramp);

        // Linear target: undoing the file permutation must recover the ramp.
        let mut row = file.clone();
        repermute_row(&mut row, &[0]);
        assert_eq!(row, ramp, "file -> linear");

        // AVX2 target: same as laying the ramp out for the AVX2 packus directly.
        let mut row = file.clone();
        repermute_row(&mut row, &[0, 2, 1, 3]);
        let mut want = vec![0i16; L1_SIZE];
        for (chunk_o, chunk_i) in want.chunks_exact_mut(32).zip(ramp.chunks_exact(32)) {
            for (j, &p) in [0usize, 2, 1, 3].iter().enumerate() {
                chunk_o[j * 8..(j + 1) * 8].copy_from_slice(&chunk_i[p * 8..(p + 1) * 8]);
            }
        }
        assert_eq!(row, want, "file -> avx2");
    }

    #[test]
    fn logical_order_round_trips_through_the_file_layout() {
        let ramp: Vec<i16> = (0..L1_SIZE as i16).collect();
        let mut row = ramp.clone();
        super::linear_to_file(&mut row);
        assert_eq!(row, linear_to_file(&ramp), "the trainer's layout");
        super::file_to_linear(&mut row);
        assert_eq!(row, ramp, "and back");
    }

    #[test]
    fn every_memory_column_maps_to_one_logical_neuron_and_halves_stay_paired() {
        let map = memory_to_logical();
        let mut seen = [false; L1_SIZE];
        for &l in &map {
            assert!(!seen[l as usize], "logical neuron {l} carried twice");
            seen[l as usize] = true;
        }
        let half = L1_SIZE / 2;
        for m in 0..half {
            assert!((map[m] as usize) < half, "left column {m} maps right");
            assert_eq!(
                map[m + half] as usize,
                map[m] as usize + half,
                "column {m} unpaired"
            );
        }
    }
}
