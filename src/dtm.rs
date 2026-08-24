//! GaiaTB DTM prober — 3+4 piece endgame tablebases embedded in the binary.
//!
//! This module wraps `gaiatb/src/probe.rs` (copied verbatim below) and adds
//! a GaiaChess-specific `probe_position()` entry point.
//!
//! **50-move rule**: like Nalimov and Gaviota, these tables ignore the 50-move
//! rule. This is acceptable because 3+4 piece DTMs are small (≤83 half-moves).
//!
//! Gated behind `--features gaiatb`.

use std::sync::OnceLock;

use crate::position::Position;
use crate::types::{Color, PieceType, SCORE_MATE};

// ── Blob source ──────────────────────────────────────────────────────────────
//
// Native builds carry the blob inside the binary (`gaiatb_embedded`, set by build.rs)
// and load it with [`init`]. The browser build ships it beside the module instead,
// like the network weights: the host writes the compressed blob into the buffer
// [`reserve`] hands back and calls [`publish_received`], which decompresses it and
// frees the compressed copy.

#[cfg(gaiatb_embedded)]
static BLOB: &[u8] = include_bytes!(env!("GAIATB_ZST"));

static PROBER: OnceLock<Option<Prober>> = OnceLock::new();

/// Initialize the prober from the embedded blob. Called once at startup.
/// Returns true if the tables loaded successfully.
#[cfg(gaiatb_embedded)]
pub fn init() -> bool {
    PROBER.get_or_init(|| Prober::from_blob(BLOB)).is_some()
}

/// Where a host-delivered blob accumulates before [`publish_received`] reads it.
static INCOMING: std::sync::Mutex<Vec<u8>> = std::sync::Mutex::new(Vec::new());

/// Sizes the incoming-blob buffer to `len` bytes and returns where to write them.
///
/// The pointer is valid until the next call into the module: nothing of ours runs
/// while the host writes, so the buffer cannot move underneath it.
pub fn reserve(len: usize) -> *mut u8 {
    let mut buf = INCOMING.lock().unwrap();
    buf.clear();
    buf.resize(len, 0);
    buf.as_mut_ptr()
}

/// Builds the prober from the bytes the host wrote after [`reserve`], then frees
/// them — only the decompressed tables are kept. Returns true when the tables are
/// ready. A second call changes nothing: the prober is published once and never
/// replaced, since a search may be reading it.
pub fn publish_received() -> bool {
    let blob = std::mem::take(&mut *INCOMING.lock().unwrap());
    if PROBER.get().is_none() {
        PROBER.set(Prober::from_blob(&blob)).ok();
    }
    available()
}

/// Returns true if the DTM tables are loaded and ready.
pub fn available() -> bool {
    PROBER.get().map(|p| p.is_some()).unwrap_or(false)
}

/// Probe DTM for a position with ≤4 pieces (including kings).
///
/// Returns a **STM-relative negamax score** ready to return from `alpha_beta`:
/// - `mate_in(ply + dtm)` if the side to move wins in `dtm` halfmoves
/// - `mated_in(ply + |dtm|)` if the side to move loses in `|dtm|` halfmoves
/// - `0` for draws (stalemate, insufficient material)
/// - `None` if the position is not in the tables (>4 pieces or tables not loaded)
///
/// `probe_dtm()` returns an STM-relative DTM (dtm>0 = STM wins) because each
/// table section (WTM/BTM) stores DTMs from that section's mover's perspective.
pub fn probe_position(pos: &Position, ply: i32) -> Option<i32> {
    let prober = PROBER.get()?.as_ref()?;

    // Quick piece-count guard (should already be checked by caller, but be safe)
    if pos.occupied().count_ones() > 4 { return None; }

    let (white, black, wk, bk, pieces, np) = extract_for_probe(pos);
    let stm_white = pos.side_to_move == Color::White;

    let dtm = prober.probe_dtm(&white, &black, wk, bk, &pieces[..np], stm_white)?;

    // probe_dtm returns STM-relative DTM:
    //   dtm > 0: the side to move wins in dtm half-moves
    //   dtm < 0: the side to move loses in |dtm| half-moves
    //   dtm = 0: draw
    let stm_dtm = dtm as i32;

    Some(if stm_dtm > 0 {
        SCORE_MATE - ply - stm_dtm       // mate_in(ply + stm_dtm)
    } else if stm_dtm < 0 {
        -SCORE_MATE + ply + (-stm_dtm) // mated_in(ply + (-stm_dtm))
    } else {
        0 // draw (stalemate or insufficient material)
    })
}

/// Extract piece data from a GaiaChess `Position` into the raw format expected
/// by `Prober::probe_dtm`. The last element is the number of non-king pieces
/// actually written to the square array — the prober expects a slice of exactly
/// that length, not the whole fixed-size buffer.
fn extract_for_probe(pos: &Position) -> ([u8; 5], [u8; 5], u8, u8, [u8; 8], usize) {
    let wk = pos.king_sq(Color::White).0;
    let bk = pos.king_sq(Color::Black).0;

    let white = [
        pos.piece_type_bb(PieceType::Pawn,   Color::White).count_ones() as u8,
        pos.piece_type_bb(PieceType::Knight, Color::White).count_ones() as u8,
        pos.piece_type_bb(PieceType::Bishop, Color::White).count_ones() as u8,
        pos.piece_type_bb(PieceType::Rook,   Color::White).count_ones() as u8,
        pos.piece_type_bb(PieceType::Queen,  Color::White).count_ones() as u8,
    ];
    let black = [
        pos.piece_type_bb(PieceType::Pawn,   Color::Black).count_ones() as u8,
        pos.piece_type_bb(PieceType::Knight, Color::Black).count_ones() as u8,
        pos.piece_type_bb(PieceType::Bishop, Color::Black).count_ones() as u8,
        pos.piece_type_bb(PieceType::Rook,   Color::Black).count_ones() as u8,
        pos.piece_type_bb(PieceType::Queen,  Color::Black).count_ones() as u8,
    ];

    // piece_sqs order: white pawns, black pawns, then Q/R/B/N (white first, black second)
    // Within each group: ascending square order (trailing_zeros gives LSB first = ascending)
    let mut pieces = [0u8; 8];
    let mut p = 0usize;

    let mut add = |mut bb: u64| {
        while bb != 0 {
            pieces[p] = bb.trailing_zeros() as u8;
            p += 1;
            bb &= bb - 1;
        }
    };

    add(pos.piece_type_bb(PieceType::Pawn,   Color::White));
    add(pos.piece_type_bb(PieceType::Pawn,   Color::Black));
    for pt in [PieceType::Queen, PieceType::Rook, PieceType::Bishop, PieceType::Knight] {
        add(pos.piece_type_bb(pt, Color::White));
    }
    for pt in [PieceType::Queen, PieceType::Rook, PieceType::Bishop, PieceType::Knight] {
        add(pos.piece_type_bb(pt, Color::Black));
    }
    drop(add);

    (white, black, wk, bk, pieces, p)
}

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  GaiaTB probe.rs — copied from gaiatb/src/probe.rs                       ║
// ║  Self-contained: no gaiatb crate dependency.                             ║
// ║  Two local adaptations: decompression goes through `decode_zstd` (the C  ║
// ║  zstd cannot be linked into a wasm module), and `packed_data` is sized   ║
// ║  up front (letting ~155 MB double its way up would hold peak memory far  ║
// ║  above that — a browser never hands linear memory back).                 ║
// ╚══════════════════════════════════════════════════════════════════════════╝

// ── Decompression ────────────────────────────────────────────────────────────

/// One zstd frame, fully decoded. Native builds use the C zstd; the wasm build
/// decodes with ruzstd, pure Rust, same as the embedded network does.
#[cfg(not(target_arch = "wasm32"))]
fn decode_zstd(data: &[u8]) -> Option<Vec<u8>> {
    zstd::decode_all(data).ok()
}

#[cfg(target_arch = "wasm32")]
fn decode_zstd(data: &[u8]) -> Option<Vec<u8>> {
    use std::io::Read;
    // The frames declare a 128 MB window, over ruzstd's 100 MB default ceiling —
    // the same raise the embedded network needs (see nnue/network.rs). The window
    // is a claim in the frame header, not an allocation: no section comes close.
    const WINDOW: u64 = 128 * 1024 * 1024;
    let mut decoder =
        ruzstd::decoding::StreamingDecoder::new_with_max_window_size(data, WINDOW).ok()?;
    let mut out = Vec::new();
    decoder.read_to_end(&mut out).ok()?;
    Some(out)
}

// ── Blob constants ───────────────────────────────────────────────────

const BLOB_MAGIC: [u8; 4] = *b"GTPK";
const BLOB_VERSION: u8 = 2;
const HEADER_SIZE: usize = 32;
const ENTRY_SIZE: usize = 32;
const SIG_LEN: usize = 8;

const FLAG_SINGLE_VALUE: u8 = 1 << 0;
#[allow(dead_code)]
const FLAG_HAS_PAWNS: u8 = 1 << 1;
const FLAG_SPLIT_STM: u8 = 1 << 2;

// ── Binomial coefficients C(n, k) ───────────────────────────────────

const BINOMIAL: [[u64; 65]; 7] = {
    let mut t = [[0u64; 65]; 7];
    let mut n = 0;
    while n < 65 { t[0][n] = 1; n += 1; }
    let mut k = 1usize;
    while k < 7 {
        t[k][0] = 0;
        let mut n = 1usize;
        while n < 65 { t[k][n] = t[k - 1][n - 1] + t[k][n - 1]; n += 1; }
        k += 1;
    }
    t
};

// ── King triangle (D4 pawnless) ─────────────────────────────────────

const TRIANGLE_MAP: [i8; 64] = {
    let mut m = [-1i8; 64];
    m[0]=0; m[1]=1; m[2]=2; m[3]=3; m[9]=4; m[10]=5; m[11]=6; m[18]=7; m[19]=8; m[27]=9;
    m
};

#[inline]
fn is_on_diagonal(sq: u8) -> bool { let f = sq & 7; let r = sq >> 3; f == r && f < 4 }

fn canonicalize_king(sq: u8) -> (u8, u8) {
    let mut f = (sq & 7) as i8;
    let mut r = (sq >> 3) as i8;
    let mut t = 0u8;
    if f > 3 { f = 7 - f; t |= 1; }
    if r > 3 { r = 7 - r; t |= 2; }
    if f < r { std::mem::swap(&mut f, &mut r); t |= 4; }
    ((r as u8) * 8 + (f as u8), t)
}

fn canonicalize_king_pawnful(sq: u8) -> (u8, u8) {
    let f = (sq & 7) as i8;
    let r = (sq >> 3) as i8;
    if f > 3 { ((r as u8) * 8 + (7 - f) as u8, 1) } else { (sq, 0) }
}

#[inline]
fn apply_transform(sq: u8, t: u8) -> u8 {
    let mut f = (sq & 7) as i8;
    let mut r = (sq >> 3) as i8;
    if t & 1 != 0 { f = 7 - f; }
    if t & 2 != 0 { r = 7 - r; }
    if t & 4 != 0 { std::mem::swap(&mut f, &mut r); }
    (r as u8) * 8 + (f as u8)
}

#[inline]
fn diagonal_reflect(sq: u8) -> u8 { (sq & 7) * 8 + (sq >> 3) }

// ── Pawnless KK pairs (462) ─────────────────────────────────────────

const fn build_map_kk() -> ([[i16; 64]; 10], [u16; 10], [u16; 10]) {
    let wks: [u8; 10] = [0, 1, 2, 3, 9, 10, 11, 18, 19, 27];
    let diag: [bool; 10] = [true, false, false, false, true, false, false, true, false, true];
    let mut map = [[-1i16; 64]; 10];
    let mut count = [0u16; 10];
    let mut ti = 0;
    while ti < 10 {
        let wk = wks[ti];
        let wf = (wk & 7) as i8; let wr = (wk >> 3) as i8;
        let mut idx = 0i16;
        let mut bk = 0u8;
        while bk < 64 {
            let bf = (bk & 7) as i8; let br = (bk >> 3) as i8;
            let df = if wf > bf { wf - bf } else { bf - wf };
            let dr = if wr > br { wr - br } else { br - wr };
            if bk == wk || (df <= 1 && dr <= 1) || (diag[ti] && bf < br) {
                map[ti][bk as usize] = -1;
            } else {
                map[ti][bk as usize] = idx; idx += 1;
            }
            bk += 1;
        }
        count[ti] = idx as u16; ti += 1;
    }
    let mut off = [0u16; 10];
    let mut i = 1;
    while i < 10 { off[i] = off[i - 1] + count[i - 1]; i += 1; }
    (map, count, off)
}

static MAP_KK: [[i16; 64]; 10] = build_map_kk().0;
#[allow(dead_code)]
static BK_COUNT: [u16; 10] = build_map_kk().1;
static KK_OFFSETS: [u16; 10] = build_map_kk().2;

// ── Pawnful KK pairs (1806) ─────────────────────────────────────────

const fn build_map_kk_pawnful() -> ([[i16; 64]; 32], [u16; 32], [u16; 32]) {
    let mut map = [[-1i16; 64]; 32];
    let mut count = [0u16; 32];
    let mut wk = 0u8;
    while wk < 64 {
        let wf = wk & 7;
        if wf > 3 { wk += 1; continue; }
        let wi = ((wk >> 3) * 4 + wf) as usize;
        let wr = (wk >> 3) as i8; let wfi = wf as i8;
        let mut idx = 0i16;
        let mut bk = 0u8;
        while bk < 64 {
            let bf = (bk & 7) as i8; let br = (bk >> 3) as i8;
            let df = if wfi > bf { wfi - bf } else { bf - wfi };
            let dr = if wr > br { wr - br } else { br - wr };
            if bk == wk || (df <= 1 && dr <= 1) {
                map[wi][bk as usize] = -1;
            } else {
                map[wi][bk as usize] = idx; idx += 1;
            }
            bk += 1;
        }
        count[wi] = idx as u16; wk += 1;
    }
    let mut off = [0u16; 32];
    let mut i = 1;
    while i < 32 { off[i] = off[i - 1] + count[i - 1]; i += 1; }
    (map, count, off)
}

static MAP_KK_PAWNFUL: [[i16; 64]; 32] = build_map_kk_pawnful().0;
#[allow(dead_code)]
static BK_COUNT_PAWNFUL: [u16; 32] = build_map_kk_pawnful().1;
static KK_OFFSETS_PAWNFUL: [u16; 32] = build_map_kk_pawnful().2;

// ── MapPawns (pawn square → 0-47 index) ─────────────────────────────

const MAP_PAWNS: [u8; 64] = {
    let mut m = [0u8; 64];
    let mut avail = 47u8;
    let files: [(u8, u8); 4] = [(0, 7), (1, 6), (2, 5), (3, 4)];
    let mut rank = 1u8;
    while rank <= 6 {
        let mut fi = 0;
        while fi < 4 {
            let (f1, f2) = files[fi];
            m[(rank * 8 + f1) as usize] = avail;
            avail = avail.saturating_sub(1);
            m[(rank * 8 + f2) as usize] = avail;
            avail = avail.saturating_sub(1);
            fi += 1;
        }
        rank += 1;
    }
    m
};

// ── Bit-packing ─────────────────────────────────────────────────────

#[inline]
fn unpack_value(data: &[u8], index: usize, bits: u8) -> u8 {
    let bit_off = index as u64 * bits as u64;
    let byte_idx = (bit_off / 8) as usize;
    let bit_idx = (bit_off % 8) as u32;
    let mask = (1u16 << bits) - 1;
    let raw = data[byte_idx] as u16
        | ((data.get(byte_idx + 1).copied().unwrap_or(0) as u16) << 8);
    ((raw >> bit_idx) & mask) as u8
}

#[inline]
fn packed_to_dtm(packed: u8, max_dtm: u16) -> i16 {
    packed as i16 - max_dtm as i16
}

// ── Material key ────────────────────────────────────────────────────

type MatCounts = [u8; 5];

fn material_value(c: &MatCounts) -> u32 {
    c[0] as u32 + c[1] as u32 * 3 + c[2] as u32 * 3 + c[3] as u32 * 5 + c[4] as u32 * 9
}

fn canonical_material(white: &MatCounts, black: &MatCounts) -> (MatCounts, MatCounts, bool) {
    let wv = material_value(white);
    let bv = material_value(black);
    let stronger = if wv != bv {
        wv > bv
    } else {
        let mut w_stronger = true;
        let mut i = 4i8;
        while i >= 0 {
            if white[i as usize] != black[i as usize] {
                w_stronger = white[i as usize] > black[i as usize];
                break;
            }
            i -= 1;
        }
        w_stronger
    };
    if stronger { (*white, *black, false) } else { (*black, *white, true) }
}

fn material_sig(white: &MatCounts, black: &MatCounts) -> [u8; SIG_LEN] {
    let mut sig = [0u8; SIG_LEN];
    let mut pos = 0;
    sig[pos] = b'K'; pos += 1;
    for _ in 0..white[4] { sig[pos] = b'Q'; pos += 1; }
    for _ in 0..white[3] { sig[pos] = b'R'; pos += 1; }
    for _ in 0..white[2] { sig[pos] = b'B'; pos += 1; }
    for _ in 0..white[1] { sig[pos] = b'N'; pos += 1; }
    for _ in 0..white[0] { sig[pos] = b'P'; pos += 1; }
    sig[pos] = b'v'; pos += 1;
    sig[pos] = b'K'; pos += 1;
    for _ in 0..black[4] { sig[pos] = b'Q'; pos += 1; }
    for _ in 0..black[3] { sig[pos] = b'R'; pos += 1; }
    for _ in 0..black[2] { sig[pos] = b'B'; pos += 1; }
    for _ in 0..black[1] { sig[pos] = b'N'; pos += 1; }
    for _ in 0..black[0] { sig[pos] = b'P'; pos += 1; }
    sig
}

// ── Compact indexer ──────────────────────────────────────────────────

/// The parameters describe one tablebase's shape; they travel together but are all
/// plain scalars, and a struct here would only move the same list one line up.
#[allow(clippy::too_many_arguments)]
fn encode(
    wk: u8, bk: u8, piece_sqs: &[u8],
    has_pawns: bool,
    num_white_pawns: usize, num_black_pawns: usize,
    non_pawn_groups: &[usize],
    per_stm_size: usize,
) -> Option<usize> {
    let num_pawns = num_white_pawns + num_black_pawns;
    let num_pieces = num_pawns + non_pawn_groups.iter().sum::<usize>();
    debug_assert_eq!(piece_sqs.len(), num_pieces);

    let mut cpieces = [0u8; 8];

    let (kk_global, _canon_wk, canon_bk) = if has_pawns {
        let (cwk, transform) = canonicalize_king_pawnful(wk);
        let wi = ((cwk >> 3) * 4 + (cwk & 7)) as usize;
        let cbk = apply_transform(bk, transform);
        for (i, &sq) in piece_sqs.iter().enumerate() {
            cpieces[i] = apply_transform(sq, transform);
        }
        let kk_local = MAP_KK_PAWNFUL[wi][cbk as usize];
        if kk_local < 0 { return None; }
        (KK_OFFSETS_PAWNFUL[wi] as usize + kk_local as usize, cwk, cbk)
    } else {
        let (cwk, transform) = canonicalize_king(wk);
        let ti = TRIANGLE_MAP[cwk as usize];
        if ti < 0 { return None; }
        let ti = ti as usize;
        let mut cbk = apply_transform(bk, transform);
        for (i, &sq) in piece_sqs.iter().enumerate() {
            cpieces[i] = apply_transform(sq, transform);
        }
        if is_on_diagonal(cwk)
            && (cbk & 7) < (cbk >> 3)
        {
            cbk = diagonal_reflect(cbk);
            for p in cpieces[..num_pieces].iter_mut() {
                *p = diagonal_reflect(*p);
            }
        }
        let kk_local = MAP_KK[ti][cbk as usize];
        if kk_local < 0 { return None; }
        (KK_OFFSETS[ti] as usize + kk_local as usize, cwk, cbk)
    };

    let mut idx = kk_global;

    if num_white_pawns > 0 {
        let mut wp_mapped = [0usize; 8];
        for i in 0..num_white_pawns {
            wp_mapped[i] = MAP_PAWNS[cpieces[i] as usize] as usize;
        }
        wp_mapped[..num_white_pawns].sort_unstable();
        let mut wp_idx = 0usize;
        for i in 0..num_white_pawns {
            wp_idx += BINOMIAL[i + 1][wp_mapped[i]] as usize;
        }
        idx = idx * (BINOMIAL[num_white_pawns][48] as usize) + wp_idx;
    }

    if num_black_pawns > 0 {
        let mut bp_mapped = [0usize; 8];
        for i in 0..num_black_pawns {
            let sq = cpieces[num_white_pawns + i];
            let mut pi = (sq - 8) as usize;
            for &excl in &cpieces[..num_white_pawns] {
                if (8..56).contains(&excl) && excl < sq { pi -= 1; }
            }
            bp_mapped[i] = pi;
        }
        bp_mapped[..num_black_pawns].sort_unstable();
        let bp_avail = 48 - num_white_pawns;
        let mut bp_idx = 0usize;
        for i in 0..num_black_pawns {
            bp_idx += BINOMIAL[i + 1][bp_mapped[i]] as usize;
        }
        idx = idx * (BINOMIAL[num_black_pawns][bp_avail] as usize) + bp_idx;
    }

    let mut occ = [0u8; 16];
    occ[0] = if has_pawns {
        canonicalize_king_pawnful(wk).0
    } else {
        canonicalize_king(wk).0
    };
    occ[1] = canon_bk;
    let mut occ_len = 2 + num_pawns;
    occ[2..2 + num_pawns].copy_from_slice(&cpieces[..num_pawns]);
    occ[..occ_len].sort_unstable();

    let mut piece_offset = num_pawns;
    for &group_size in non_pawn_groups {
        let avail = 64 - occ_len;
        if group_size == 1 {
            let piece_idx = sq_to_idx(&occ[..occ_len], cpieces[piece_offset]);
            idx = idx * avail + piece_idx;
        } else {
            let mut np_indices = [0usize; 8];
            for j in 0..group_size {
                np_indices[j] = sq_to_idx(&occ[..occ_len], cpieces[piece_offset + j]);
            }
            np_indices[..group_size].sort_unstable();
            let mut group_idx = 0usize;
            for j in 0..group_size {
                group_idx += BINOMIAL[j + 1][np_indices[j]] as usize;
            }
            idx = idx * (BINOMIAL[group_size][avail] as usize) + group_idx;
        }
        for j in 0..group_size {
            occ[occ_len] = cpieces[piece_offset + j];
            occ_len += 1;
        }
        occ[..occ_len].sort_unstable();
        piece_offset += group_size;
    }

    debug_assert!(idx < per_stm_size);
    Some(idx)
}

#[inline]
fn sq_to_idx(occupied: &[u8], sq: u8) -> usize {
    let mut idx = sq as usize;
    for &occ in occupied { if sq > occ { idx -= 1; } }
    idx
}

// ── Prober ──────────────────────────────────────────────────────────

struct ProberTable {
    sig: [u8; SIG_LEN],
    flags: u8,
    bits_per_entry: u8,
    max_dtm: u16,
    per_stm_size: u32,
    #[allow(dead_code)]
    total_positions: u32,
    packed_offset: usize,
    /// Byte offset in packed_data where the BTM section starts.
    /// For FLAG_SPLIT_STM tables, WTM and BTM are decompressed separately
    /// and byte-aligned independently, so we cannot use a single contiguous
    /// bit-stream index. Instead, we index into each section separately.
    btm_packed_offset: usize,
    has_pawns: bool,
    num_white_pawns: usize,
    num_black_pawns: usize,
    non_pawn_groups: Vec<usize>,
}

struct Prober {
    tables: Vec<ProberTable>,
    packed_data: Vec<u8>,
}

impl Prober {
    fn from_blob(blob: &[u8]) -> Option<Self> {
        if blob.len() < HEADER_SIZE || blob[0..4] != BLOB_MAGIC || blob[4] != BLOB_VERSION {
            return None;
        }
        let table_count = blob[5] as usize;
        let data_start = HEADER_SIZE + table_count * ENTRY_SIZE;
        if blob.len() < data_start { return None; }

        // Size the decompressed store from the entries before filling it, so the
        // ~155 MB is allocated once instead of grown. The entry table understates
        // slightly — pawnful tables carry en-passant sub-ranges beyond `per_stm` —
        // hence the margin below: what matters is that the store never reallocates,
        // since at this size letting it double its way up would hold peak memory far
        // above the result, and a browser never hands linear memory back.
        let mut total_packed = 0usize;
        for i in 0..table_count {
            let off = HEADER_SIZE + i * ENTRY_SIZE;
            let buf = &blob[off..off + ENTRY_SIZE];
            let flags = buf[8];
            let bits = buf[9] as usize;
            let per_stm = u32::from_le_bytes(buf[12..16].try_into().ok()?) as usize;
            if flags & FLAG_SINGLE_VALUE != 0 { continue; }
            total_packed += if flags & FLAG_SPLIT_STM != 0 {
                2 * (per_stm * bits).div_ceil(8)
            } else {
                (2 * per_stm * bits).div_ceil(8)
            };
        }

        let capacity = total_packed + total_packed / 64 + (64 << 10);
        let mut tables = Vec::with_capacity(table_count);
        let mut packed_data: Vec<u8> = Vec::with_capacity(capacity);

        for i in 0..table_count {
            let off = HEADER_SIZE + i * ENTRY_SIZE;
            let buf = &blob[off..off + ENTRY_SIZE];

            let mut sig = [0u8; SIG_LEN];
            sig.copy_from_slice(&buf[0..8]);
            let flags = buf[8];
            let bits = buf[9];
            let max_dtm = u16::from_le_bytes([buf[10], buf[11]]);
            let per_stm = u32::from_le_bytes(buf[12..16].try_into().ok()?);
            let total_pos = u32::from_le_bytes(buf[16..20].try_into().ok()?);
            let comp_off = u64::from_le_bytes(buf[20..28].try_into().ok()?) as usize;
            let comp_sz = u32::from_le_bytes(buf[28..32].try_into().ok()?) as usize;

            let packed_offset = packed_data.len();
            let mut btm_packed_offset = packed_offset;

            if flags & FLAG_SINGLE_VALUE == 0 && comp_sz > 0 {
                let cs = data_start + comp_off;
                let ce = cs + comp_sz;
                if ce > blob.len() { return None; }
                let compressed = &blob[cs..ce];

                if flags & FLAG_SPLIT_STM != 0 {
                    let wtm_sz = u32::from_le_bytes(compressed[0..4].try_into().ok()?) as usize;
                    let wtm = decode_zstd(&compressed[4..4 + wtm_sz])?;
                    let btm = decode_zstd(&compressed[4 + wtm_sz..])?;
                    packed_data.extend_from_slice(&wtm);
                    btm_packed_offset = packed_data.len();
                    packed_data.extend_from_slice(&btm);
                } else {
                    let decompressed = decode_zstd(compressed)?;
                    packed_data.extend_from_slice(&decompressed);
                }
            }

            let (has_pawns, wp, bp, groups) = parse_sig_to_indexer(&sig);

            tables.push(ProberTable {
                sig, flags, bits_per_entry: bits, max_dtm,
                per_stm_size: per_stm, total_positions: total_pos,
                packed_offset, btm_packed_offset,
                has_pawns, num_white_pawns: wp, num_black_pawns: bp,
                non_pawn_groups: groups,
            });
        }

        debug_assert!(
            packed_data.capacity() == capacity,
            "the packed store outgrew its estimate and reallocated",
        );
        Some(Prober { tables, packed_data })
    }

    /// Probe DTM. Returns `Some(dtm)` where dtm>0 means STM wins, dtm<0 STM loses, 0=draw.
    fn probe_dtm(
        &self,
        white: &MatCounts, black: &MatCounts,
        wk: u8, bk: u8,
        piece_sqs: &[u8],
        stm_white: bool,
    ) -> Option<i16> {
        let (cw, cb, flipped) = canonical_material(white, black);
        let sig = material_sig(&cw, &cb);

        let table = self.tables.iter().find(|t| t.sig == sig)?;

        if table.flags & FLAG_SINGLE_VALUE != 0 {
            return Some(0);
        }

        let (enc_wk, enc_bk, enc_stm_black) = if flipped {
            (bk ^ 56, wk ^ 56, stm_white)
        } else {
            (wk, bk, !stm_white)
        };

        let mut enc_pieces = [0u8; 8];
        let num_pieces = piece_sqs.len();
        if flipped {
            let owp = white[0] as usize;
            let obp = black[0] as usize;
            let ownp: usize = white[1..].iter().map(|&c| c as usize).sum();
            let obnp: usize = black[1..].iter().map(|&c| c as usize).sum();
            let mut pos = 0;
            for &sq in &piece_sqs[owp..owp + obp] { enc_pieces[pos] = sq ^ 56; pos += 1; }
            for &sq in &piece_sqs[..owp] { enc_pieces[pos] = sq ^ 56; pos += 1; }
            let np_start = owp + obp;
            pos = obp + owp;
            for &sq in &piece_sqs[np_start + ownp..np_start + ownp + obnp] { enc_pieces[pos] = sq ^ 56; pos += 1; }
            for &sq in &piece_sqs[np_start..np_start + ownp] { enc_pieces[pos] = sq ^ 56; pos += 1; }
        } else {
            enc_pieces[..num_pieces].copy_from_slice(piece_sqs);
        }

        let idx = encode(
            enc_wk, enc_bk, &enc_pieces[..num_pieces],
            table.has_pawns,
            table.num_white_pawns, table.num_black_pawns,
            &table.non_pawn_groups,
            table.per_stm_size as usize,
        )?;

        // For FLAG_SPLIT_STM tables, WTM and BTM sections are decompressed
        // separately and byte-aligned independently. We index into each section
        // from its own byte start to avoid bit-alignment issues.
        // For non-split tables, the data is a single contiguous bit stream
        // where BTM entries follow WTM entries at offset per_stm_size.
        let bits = table.bits_per_entry;
        let (base, final_idx) = if table.flags & FLAG_SPLIT_STM != 0 {
            if enc_stm_black {
                (table.btm_packed_offset, idx)
            } else {
                (table.packed_offset, idx)
            }
        } else {
            let adj = if enc_stm_black { idx + table.per_stm_size as usize } else { idx };
            (table.packed_offset, adj)
        };
        let data = &self.packed_data[base..];
        let packed_val = unpack_value(data, final_idx, bits);
        let dtm = packed_to_dtm(packed_val, table.max_dtm);

        // dtm is already STM-relative: the enc_stm_black mapping ensures that
        // the table's STM always matches the position's STM. Positive = position
        // STM wins, negative = position STM loses. No sign flip needed.
        Some(dtm)
    }
}

fn parse_sig_to_indexer(sig: &[u8; SIG_LEN]) -> (bool, usize, usize, Vec<usize>) {
    let v_pos = sig.iter().position(|&b| b == b'v').unwrap_or(SIG_LEN);
    let white_str = &sig[..v_pos];
    let black_str = &sig[v_pos + 1..];
    let count_piece = |s: &[u8], ch: u8| s.iter().filter(|&&b| b == ch).count();
    let wp = count_piece(white_str, b'P');
    let bp = count_piece(black_str, b'P');
    let has_pawns = wp > 0 || bp > 0;
    let mut groups = Vec::new();
    for &(side, ch) in &[(white_str, b'Q'), (white_str, b'R'), (white_str, b'B'), (white_str, b'N'),
                          (black_str, b'Q'), (black_str, b'R'), (black_str, b'B'), (black_str, b'N')] {
        let c = count_piece(side, ch);
        if c > 0 { groups.push(c); }
    }
    (has_pawns, wp, bp, groups)
}
