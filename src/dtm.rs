//! GaiaTB DTM prober — 3+4 piece endgame tablebases embedded in the binary.
//!
//! This module wraps `gaiatb/src/probe.rs` (copied verbatim below) and adds
//! a GaiaChess-specific `probe_position()` entry point.
//!
//! **50-move rule**: like Nalimov and Gaviota, these tables ignore the 50-move rule,
//! so [`probe_position`] declines a mate further away than the clock still allows —
//! a 3+4 piece DTM runs up to 83 half-moves, more than a clock past 17 leaves (#24).
//!
//! **One copy per machine**: the decompressed tables are shared between engine
//! processes like the network weights (see [`load`] and `shm.rs`).
//!
//! Gated behind `--features gaiatb`.

use std::borrow::Cow;
use std::sync::OnceLock;

use crate::position::Position;
use crate::shm::{Backing, Origin};
use crate::types::{Color, PieceType, dtm_score};

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
    PROBER.get_or_init(|| load(BLOB)).is_some()
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
        PROBER.set(load(&blob)).ok();
    }
    available()
}

// ── One copy per machine ─────────────────────────────────────────────────────
//
// The decompressed tables are ~155 MB per process and never written once built. A
// test lane running 64 engines held 64 identical copies of them, ~10 GB. They are
// shared like the network weights (`shm.rs`): one sealed image per executable and
// blob, built by the first process that asks, mapped read-only by the others, freed
// with the last of them. Any failure falls back to a private copy, which is what
// every process held before: sharing saves memory, it is never a condition for
// probing the tables.

/// Turns the sharing of the tables off (`GAIA_SHARED_TB=0`), independently of the
/// network's `GAIA_SHARED_NET`, so each can be measured on its own.
const SHARED_TB_VAR: &str = "GAIA_SHARED_TB";

/// Version of the image layout. The image name carries it, so a change to the
/// header or to how the store is laid out must bump it: two layouts must never
/// meet under one name (on Windows, where images are not keyed by executable).
const IMAGE_FORMAT: u32 = 1;

const IMAGE_MAGIC: [u8; 4] = *b"GTBI";

/// Fixed part of the image header. One (WTM, BTM) offset pair per table follows.
const IMAGE_FIXED: usize = 32;

/// Header status of an image whose store holds every table.
const IMAGE_COMPLETE: u32 = 1;

/// One line for `gaiachess info`. Loads nothing.
pub fn describe_sharing() -> String {
    crate::shm::describe_policy(crate::shm::wanted(SHARED_TB_VAR), SHARED_TB_VAR)
}

/// The tables, from the machine's shared image when possible, else private.
fn load(blob: &[u8]) -> Option<Prober> {
    let dir = read_directory(blob)?;
    if !crate::shm::supported() {
        return Prober::private(blob, &dir);
    }
    let mut reason = format!("sharing turned off by {SHARED_TB_VAR}");
    if crate::shm::wanted(SHARED_TB_VAR) {
        let name = image_name(blob);
        match Prober::shared(blob, &dir, &name) {
            Ok((prober, origin)) => {
                let backing = Backing::Shared {
                    origin,
                    peers: crate::shm::peers_of(&name),
                };
                crate::outerr!("info string GaiaTB tables: {backing}");
                return Some(prober);
            }
            Err(e) => reason = e,
        }
    }
    let prober = Prober::private(blob, &dir)?;
    crate::outerr!("info string GaiaTB tables: {}", Backing::Private { reason });
    Some(prober)
}

/// The image's name: which blob, in which layout. The sharing layer adds the user
/// and the executable.
fn image_name(blob: &[u8]) -> String {
    format!(
        "gaiatb-{:016x}-i{IMAGE_FORMAT}",
        crate::nnue::integrity::fnv1a64(blob)
    )
}

/// The image is a header, then the store exactly as a private load lays it out:
///
/// ```text
/// 0   magic "GTBI"            4   IMAGE_FORMAT
/// 8   table count             12  status (IMAGE_COMPLETE, or 0: the store overflowed)
/// 16  bytes of store used     24  reserved
/// 32  per table, u64 each: where its WTM and BTM sections start in the store
/// ```
///
/// The offsets are written down because nothing else knows them: the frames do not
/// record their decoded size, and a process that receives the image must not have
/// to decompress anything to read it.
fn image_header_len(tables: usize) -> usize {
    (IMAGE_FIXED + 16 * tables).next_multiple_of(64)
}

fn write_header(head: &mut [u8], tables: usize, layout: Option<&Layout>) {
    debug_assert_eq!(head.len(), image_header_len(tables));
    head.fill(0);
    head[0..4].copy_from_slice(&IMAGE_MAGIC);
    head[4..8].copy_from_slice(&IMAGE_FORMAT.to_le_bytes());
    head[8..12].copy_from_slice(&(tables as u32).to_le_bytes());
    // An image whose build failed is still published, status zero, so that every
    // process reads the same refusal instead of waiting for a builder that gave up.
    let Some(layout) = layout else {
        return;
    };
    debug_assert_eq!(layout.offsets.len(), tables);
    head[12..16].copy_from_slice(&IMAGE_COMPLETE.to_le_bytes());
    head[16..24].copy_from_slice(&(layout.used as u64).to_le_bytes());
    for (i, &(wtm, btm)) in layout.offsets.iter().enumerate() {
        let at = IMAGE_FIXED + 16 * i;
        head[at..at + 8].copy_from_slice(&(wtm as u64).to_le_bytes());
        head[at + 8..at + 16].copy_from_slice(&(btm as u64).to_le_bytes());
    }
}

fn read_header(head: &[u8], tables: usize, capacity: usize) -> Result<Layout, String> {
    if head.len() != image_header_len(tables) {
        return Err(format!("header is {} bytes", head.len()));
    }
    let u32_at =
        |at: usize| u32::from_le_bytes([head[at], head[at + 1], head[at + 2], head[at + 3]]);
    let u64_at = |at: usize| {
        let mut b = [0u8; 8];
        b.copy_from_slice(&head[at..at + 8]);
        u64::from_le_bytes(b)
    };
    if head[0..4] != IMAGE_MAGIC || u32_at(4) != IMAGE_FORMAT {
        return Err(String::from("not a GaiaTB image of this format"));
    }
    if u32_at(8) as usize != tables {
        return Err(format!("image holds {} tables, blob {tables}", u32_at(8)));
    }
    if u32_at(12) != IMAGE_COMPLETE {
        return Err(String::from(
            "the tables did not fit the store their directory predicts",
        ));
    }
    let used = u64_at(16);
    if used > capacity as u64 {
        return Err(format!("store holds {used} bytes, room for {capacity}"));
    }
    let used = used as usize;
    let mut offsets = Vec::with_capacity(tables);
    for i in 0..tables {
        let (wtm, btm) = (
            u64_at(IMAGE_FIXED + 16 * i),
            u64_at(IMAGE_FIXED + 16 * i + 8),
        );
        if wtm > btm || btm > used as u64 {
            return Err(format!("table {i} lies outside the store"));
        }
        offsets.push((wtm as usize, btm as usize));
    }
    Ok(Layout { offsets, used })
}

/// Returns true if the DTM tables are loaded and ready.
pub fn available() -> bool {
    PROBER.get().map(|p| p.is_some()).unwrap_or(false)
}

/// Probe DTM for a position with ≤4 pieces (including kings).
///
/// Returns a **STM-relative negamax score** ready to return from `alpha_beta`:
/// - a mate score if the side to move wins (or is mated) within [`crate::types::MAX_PLY`]
/// - a TB score (`cp ±20000` on the wire) if the DTM is longer than that
/// - `0` for draws (stalemate, insufficient material)
/// - `None` if the position is not in the tables (>4 pieces or tables not loaded)
///
/// `probe_dtm()` returns an STM-relative DTM (dtm>0 = STM wins) because each
/// table section (WTM/BTM) stores DTMs from that section's mover's perspective.
pub fn probe_position(pos: &Position, ply: i32) -> Option<i32> {
    probe_with(PROBER.get()?.as_ref()?, pos, ply)
}

/// [`probe_position`] through a given set of tables.
fn probe_with(prober: &Prober, pos: &Position, ply: i32) -> Option<i32> {
    // Quick piece-count guard (should already be checked by caller, but be safe)
    if pos.occupied().count_ones() > 4 {
        return None;
    }

    let (white, black, wk, bk, pieces, np) = extract_for_probe(pos);
    let stm_white = pos.side_to_move == Color::White;

    let dtm = prober.probe_dtm(&white, &black, wk, bk, &pieces[..np], stm_white)?;
    let dtm = dtm as i32;
    // A mate the 50-move rule may forbid is not proved by a distance-to-mate table: it
    // can still be won through a capture or a pawn move the DTM line does not make,
    // but only a search can say so. Declined, as Nalimov declines it, rather than
    // turning a cursed win into a mate. A stored value is the distance plus one.
    let distance = if dtm == 0 { 0 } else { dtm.abs() - 1 };
    let moves_left = 100 - pos.halfmove_clock as i32;
    if distance > moves_left {
        return None;
    }
    Some(stored_dtm_to_score(dtm, ply))
}

/// Map a stored DTM to a negamax score at `ply`.
///
/// `probe_dtm` returns an STM-relative value whose sign says who wins, and whose
/// magnitude is the true distance in half-moves **plus one**: the tables count a
/// checkmated side as lost in one, so `-1` is the mate on the board and `2` a mate
/// in one half-move (see `gaiatb_sanity`). Zero is a draw. Reading the magnitude as
/// the distance made every table mate one half-move too long, which let a search
/// mate in one lose to a table "mate in two".
fn stored_dtm_to_score(stored: i32, ply: i32) -> i32 {
    debug_assert!(
        stored != 1,
        "a stored +1 would be a win in no half-move at all"
    );
    match stored {
        0 => 0,
        d if d > 0 => dtm_score(ply, d - 1),
        d => dtm_score(ply, d + 1),
    }
}

/// Extract piece data from a GaiaChess `Position` into the raw format expected
/// by `Prober::probe_dtm`. The last element is the number of non-king pieces
/// actually written to the square array — the prober expects a slice of exactly
/// that length, not the whole fixed-size buffer.
fn extract_for_probe(pos: &Position) -> ([u8; 5], [u8; 5], u8, u8, [u8; 8], usize) {
    let wk = pos.king_sq(Color::White).0;
    let bk = pos.king_sq(Color::Black).0;

    let white = [
        pos.piece_type_bb(PieceType::Pawn, Color::White)
            .count_ones() as u8,
        pos.piece_type_bb(PieceType::Knight, Color::White)
            .count_ones() as u8,
        pos.piece_type_bb(PieceType::Bishop, Color::White)
            .count_ones() as u8,
        pos.piece_type_bb(PieceType::Rook, Color::White)
            .count_ones() as u8,
        pos.piece_type_bb(PieceType::Queen, Color::White)
            .count_ones() as u8,
    ];
    let black = [
        pos.piece_type_bb(PieceType::Pawn, Color::Black)
            .count_ones() as u8,
        pos.piece_type_bb(PieceType::Knight, Color::Black)
            .count_ones() as u8,
        pos.piece_type_bb(PieceType::Bishop, Color::Black)
            .count_ones() as u8,
        pos.piece_type_bb(PieceType::Rook, Color::Black)
            .count_ones() as u8,
        pos.piece_type_bb(PieceType::Queen, Color::Black)
            .count_ones() as u8,
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

    add(pos.piece_type_bb(PieceType::Pawn, Color::White));
    add(pos.piece_type_bb(PieceType::Pawn, Color::Black));
    for pt in [
        PieceType::Queen,
        PieceType::Rook,
        PieceType::Bishop,
        PieceType::Knight,
    ] {
        add(pos.piece_type_bb(pt, Color::White));
    }
    for pt in [
        PieceType::Queen,
        PieceType::Rook,
        PieceType::Bishop,
        PieceType::Knight,
    ] {
        add(pos.piece_type_bb(pt, Color::Black));
    }
    drop(add);

    (white, black, wk, bk, pieces, p)
}

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  GaiaTB probe.rs — copied from gaiatb/src/probe.rs                       ║
// ║  Self-contained: no gaiatb crate dependency.                             ║
// ║  Local adaptations: decompression goes through `decode_frame` (the C     ║
// ║  zstd cannot be linked into a wasm module), and loading is split so the  ║
// ║  store can be a buffer of our choosing — this process's own, sized up    ║
// ║  front (letting ~155 MB double its way up would hold peak memory far     ║
// ║  above that — a browser never hands linear memory back), or an image     ║
// ║  shared with the other engine processes.                                 ║
// ╚══════════════════════════════════════════════════════════════════════════╝

// ── Decompression ────────────────────────────────────────────────────────────

/// Decodes one zstd frame into the front of `dst` and returns its length; `None`
/// if the frame is corrupt or does not fit. Native builds use the C zstd, which
/// writes in place: the frames do not record their decoded size, but the
/// single-pass decoder only needs room enough.
#[cfg(not(target_arch = "wasm32"))]
fn decode_frame(frame: &[u8], dst: &mut [u8]) -> Option<usize> {
    zstd::bulk::decompress_to_buffer(frame, dst).ok()
}

/// The wasm build decodes with ruzstd, pure Rust, same as the embedded network does.
#[cfg(target_arch = "wasm32")]
fn decode_frame(frame: &[u8], dst: &mut [u8]) -> Option<usize> {
    use std::io::Read;
    // The frames declare a 128 MB window, over ruzstd's 100 MB default ceiling —
    // the same raise the embedded network needs (see nnue/network.rs). The window
    // is a claim in the frame header, not an allocation: no section comes close.
    const WINDOW: u64 = 128 * 1024 * 1024;
    let mut decoder =
        ruzstd::decoding::StreamingDecoder::new_with_max_window_size(frame, WINDOW).ok()?;
    // Read straight into the store: a temporary as large as the section would only
    // raise the high-water mark of a linear memory the browser never hands back.
    let mut len = 0;
    while len < dst.len() {
        match decoder.read(&mut dst[len..]).ok()? {
            0 => return Some(len),
            n => len += n,
        }
    }
    // The store is full: the frame must have ended exactly there.
    let mut more = [0u8; 1];
    (decoder.read(&mut more).ok()? == 0).then_some(len)
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
    while n < 65 {
        t[0][n] = 1;
        n += 1;
    }
    let mut k = 1usize;
    while k < 7 {
        t[k][0] = 0;
        let mut n = 1usize;
        while n < 65 {
            t[k][n] = t[k - 1][n - 1] + t[k][n - 1];
            n += 1;
        }
        k += 1;
    }
    t
};

// ── King triangle (D4 pawnless) ─────────────────────────────────────

const TRIANGLE_MAP: [i8; 64] = {
    let mut m = [-1i8; 64];
    m[0] = 0;
    m[1] = 1;
    m[2] = 2;
    m[3] = 3;
    m[9] = 4;
    m[10] = 5;
    m[11] = 6;
    m[18] = 7;
    m[19] = 8;
    m[27] = 9;
    m
};

#[inline]
fn is_on_diagonal(sq: u8) -> bool {
    let f = sq & 7;
    let r = sq >> 3;
    f == r && f < 4
}

fn canonicalize_king(sq: u8) -> (u8, u8) {
    let mut f = (sq & 7) as i8;
    let mut r = (sq >> 3) as i8;
    let mut t = 0u8;
    if f > 3 {
        f = 7 - f;
        t |= 1;
    }
    if r > 3 {
        r = 7 - r;
        t |= 2;
    }
    if f < r {
        std::mem::swap(&mut f, &mut r);
        t |= 4;
    }
    ((r as u8) * 8 + (f as u8), t)
}

fn canonicalize_king_pawnful(sq: u8) -> (u8, u8) {
    let f = (sq & 7) as i8;
    let r = (sq >> 3) as i8;
    if f > 3 {
        ((r as u8) * 8 + (7 - f) as u8, 1)
    } else {
        (sq, 0)
    }
}

#[inline]
fn apply_transform(sq: u8, t: u8) -> u8 {
    let mut f = (sq & 7) as i8;
    let mut r = (sq >> 3) as i8;
    if t & 1 != 0 {
        f = 7 - f;
    }
    if t & 2 != 0 {
        r = 7 - r;
    }
    if t & 4 != 0 {
        std::mem::swap(&mut f, &mut r);
    }
    (r as u8) * 8 + (f as u8)
}

#[inline]
fn diagonal_reflect(sq: u8) -> u8 {
    (sq & 7) * 8 + (sq >> 3)
}

// ── Pawnless KK pairs (462) ─────────────────────────────────────────

const fn build_map_kk() -> ([[i16; 64]; 10], [u16; 10], [u16; 10]) {
    let wks: [u8; 10] = [0, 1, 2, 3, 9, 10, 11, 18, 19, 27];
    let diag: [bool; 10] = [
        true, false, false, false, true, false, false, true, false, true,
    ];
    let mut map = [[-1i16; 64]; 10];
    let mut count = [0u16; 10];
    let mut ti = 0;
    while ti < 10 {
        let wk = wks[ti];
        let wf = (wk & 7) as i8;
        let wr = (wk >> 3) as i8;
        let mut idx = 0i16;
        let mut bk = 0u8;
        while bk < 64 {
            let bf = (bk & 7) as i8;
            let br = (bk >> 3) as i8;
            let df = if wf > bf { wf - bf } else { bf - wf };
            let dr = if wr > br { wr - br } else { br - wr };
            if bk == wk || (df <= 1 && dr <= 1) || (diag[ti] && bf < br) {
                map[ti][bk as usize] = -1;
            } else {
                map[ti][bk as usize] = idx;
                idx += 1;
            }
            bk += 1;
        }
        count[ti] = idx as u16;
        ti += 1;
    }
    let mut off = [0u16; 10];
    let mut i = 1;
    while i < 10 {
        off[i] = off[i - 1] + count[i - 1];
        i += 1;
    }
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
        if wf > 3 {
            wk += 1;
            continue;
        }
        let wi = ((wk >> 3) * 4 + wf) as usize;
        let wr = (wk >> 3) as i8;
        let wfi = wf as i8;
        let mut idx = 0i16;
        let mut bk = 0u8;
        while bk < 64 {
            let bf = (bk & 7) as i8;
            let br = (bk >> 3) as i8;
            let df = if wfi > bf { wfi - bf } else { bf - wfi };
            let dr = if wr > br { wr - br } else { br - wr };
            if bk == wk || (df <= 1 && dr <= 1) {
                map[wi][bk as usize] = -1;
            } else {
                map[wi][bk as usize] = idx;
                idx += 1;
            }
            bk += 1;
        }
        count[wi] = idx as u16;
        wk += 1;
    }
    let mut off = [0u16; 32];
    let mut i = 1;
    while i < 32 {
        off[i] = off[i - 1] + count[i - 1];
        i += 1;
    }
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
    let raw = data[byte_idx] as u16 | ((data.get(byte_idx + 1).copied().unwrap_or(0) as u16) << 8);
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
    if stronger {
        (*white, *black, false)
    } else {
        (*black, *white, true)
    }
}

fn material_sig(white: &MatCounts, black: &MatCounts) -> [u8; SIG_LEN] {
    let mut sig = [0u8; SIG_LEN];
    let mut pos = 0;
    sig[pos] = b'K';
    pos += 1;
    for _ in 0..white[4] {
        sig[pos] = b'Q';
        pos += 1;
    }
    for _ in 0..white[3] {
        sig[pos] = b'R';
        pos += 1;
    }
    for _ in 0..white[2] {
        sig[pos] = b'B';
        pos += 1;
    }
    for _ in 0..white[1] {
        sig[pos] = b'N';
        pos += 1;
    }
    for _ in 0..white[0] {
        sig[pos] = b'P';
        pos += 1;
    }
    sig[pos] = b'v';
    pos += 1;
    sig[pos] = b'K';
    pos += 1;
    for _ in 0..black[4] {
        sig[pos] = b'Q';
        pos += 1;
    }
    for _ in 0..black[3] {
        sig[pos] = b'R';
        pos += 1;
    }
    for _ in 0..black[2] {
        sig[pos] = b'B';
        pos += 1;
    }
    for _ in 0..black[1] {
        sig[pos] = b'N';
        pos += 1;
    }
    for _ in 0..black[0] {
        sig[pos] = b'P';
        pos += 1;
    }
    sig
}

// ── Compact indexer ──────────────────────────────────────────────────

/// The parameters describe one tablebase's shape; they travel together but are all
/// plain scalars, and a struct here would only move the same list one line up.
#[allow(clippy::too_many_arguments)]
fn encode(
    wk: u8,
    bk: u8,
    piece_sqs: &[u8],
    has_pawns: bool,
    num_white_pawns: usize,
    num_black_pawns: usize,
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
        if kk_local < 0 {
            return None;
        }
        (
            KK_OFFSETS_PAWNFUL[wi] as usize + kk_local as usize,
            cwk,
            cbk,
        )
    } else {
        let (cwk, transform) = canonicalize_king(wk);
        let ti = TRIANGLE_MAP[cwk as usize];
        if ti < 0 {
            return None;
        }
        let ti = ti as usize;
        let mut cbk = apply_transform(bk, transform);
        for (i, &sq) in piece_sqs.iter().enumerate() {
            cpieces[i] = apply_transform(sq, transform);
        }
        if is_on_diagonal(cwk) && (cbk & 7) < (cbk >> 3) {
            cbk = diagonal_reflect(cbk);
            for p in cpieces[..num_pieces].iter_mut() {
                *p = diagonal_reflect(*p);
            }
        }
        let kk_local = MAP_KK[ti][cbk as usize];
        if kk_local < 0 {
            return None;
        }
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
                if (8..56).contains(&excl) && excl < sq {
                    pi -= 1;
                }
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
    for &occ in occupied {
        if sq > occ {
            idx -= 1;
        }
    }
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
    /// Every table's packed entries, end to end: this process's own copy, or an
    /// image shared with the other engine processes on the machine.
    packed_data: Cow<'static, [u8]>,
}

/// One line of the blob's directory: a table's shape, and where its frames are.
struct Entry {
    sig: [u8; SIG_LEN],
    flags: u8,
    bits: u8,
    max_dtm: u16,
    per_stm: u32,
    total_positions: u32,
    /// The table's compressed bytes within the blob; empty when it has none to
    /// decode (a single-value table).
    frames: std::ops::Range<usize>,
}

/// Where the tables landed in the decompressed store.
#[derive(Clone, PartialEq, Eq, Debug)]
struct Layout {
    /// Per table, in directory order: where its WTM and BTM sections start. They
    /// are equal unless the table is split by side to move.
    offsets: Vec<(usize, usize)>,
    /// Bytes of the store the tables fill.
    used: usize,
}

/// Reads the blob's header and directory. `None` if the blob is not one of ours
/// or a table's frames run past its end.
fn read_directory(blob: &[u8]) -> Option<Vec<Entry>> {
    if blob.len() < HEADER_SIZE || blob[0..4] != BLOB_MAGIC || blob[4] != BLOB_VERSION {
        return None;
    }
    let table_count = blob[5] as usize;
    let data_start = HEADER_SIZE + table_count * ENTRY_SIZE;
    if blob.len() < data_start {
        return None;
    }

    (0..table_count)
        .map(|i| {
            let off = HEADER_SIZE + i * ENTRY_SIZE;
            let buf = &blob[off..off + ENTRY_SIZE];

            let mut sig = [0u8; SIG_LEN];
            sig.copy_from_slice(&buf[0..8]);
            let flags = buf[8];
            let comp_off = u64::from_le_bytes(buf[20..28].try_into().ok()?) as usize;
            let comp_sz = u32::from_le_bytes(buf[28..32].try_into().ok()?) as usize;

            let frames = if flags & FLAG_SINGLE_VALUE == 0 && comp_sz > 0 {
                let start = data_start.checked_add(comp_off)?;
                let end = start.checked_add(comp_sz)?;
                if end > blob.len() {
                    return None;
                }
                start..end
            } else {
                0..0
            };

            Some(Entry {
                sig,
                flags,
                bits: buf[9],
                max_dtm: u16::from_le_bytes([buf[10], buf[11]]),
                per_stm: u32::from_le_bytes(buf[12..16].try_into().ok()?),
                total_positions: u32::from_le_bytes(buf[16..20].try_into().ok()?),
                frames,
            })
        })
        .collect()
}

/// How large a store to give the tables, from the directory alone.
///
/// Sized once up front, never grown. On the embedded blob the directory predicts the
/// decoded size to the byte (154 560 410); the margin is there for a table whose
/// sections run past `per_stm` — pawnful tables may carry en-passant sub-ranges —
/// so that such a blob still loads rather than overflowing its store.
fn store_capacity(dir: &[Entry]) -> usize {
    let mut total = 0usize;
    for e in dir {
        if e.flags & FLAG_SINGLE_VALUE != 0 {
            continue;
        }
        let (per_stm, bits) = (e.per_stm as usize, e.bits as usize);
        total += if e.flags & FLAG_SPLIT_STM != 0 {
            2 * (per_stm * bits).div_ceil(8)
        } else {
            (2 * per_stm * bits).div_ceil(8)
        };
    }
    total + total / 64 + (64 << 10)
}

/// Decompresses every table into `store`, in directory order, and says where each
/// one landed. `None` if a frame is corrupt or the store is too small for them all.
fn fill_store(blob: &[u8], dir: &[Entry], store: &mut [u8]) -> Option<Layout> {
    let mut used = 0usize;
    let mut offsets = Vec::with_capacity(dir.len());
    for e in dir {
        let wtm = used;
        let mut btm = used;
        if !e.frames.is_empty() {
            let compressed = &blob[e.frames.clone()];
            if e.flags & FLAG_SPLIT_STM != 0 {
                // Two frames, decoded and byte-aligned separately: see `probe_dtm`.
                let wtm_len = u32::from_le_bytes(compressed.get(..4)?.try_into().ok()?) as usize;
                let split = wtm_len.checked_add(4)?;
                used += decode_frame(compressed.get(4..split)?, &mut store[used..])?;
                btm = used;
                used += decode_frame(compressed.get(split..)?, &mut store[used..])?;
            } else {
                used += decode_frame(compressed, &mut store[used..])?;
            }
        }
        debug_assert!(wtm <= btm && btm <= used && used <= store.len());
        offsets.push((wtm, btm));
    }
    Some(Layout { offsets, used })
}

/// The whole store, decompressed into memory of this process's own.
fn private_store(blob: &[u8], dir: &[Entry]) -> Option<(Vec<u8>, Layout)> {
    // Zeroed rather than reserved, so the decoder can write in place; on a native
    // build the allocator hands back fresh zero pages, and the margin it never
    // writes is never faulted in.
    let mut store = vec![0u8; store_capacity(dir)];
    let layout = fill_store(blob, dir, &mut store)?;
    store.truncate(layout.used);
    Some((store, layout))
}

impl Prober {
    /// The tables in a private copy, which is what every process held before they
    /// were shared.
    fn private(blob: &[u8], dir: &[Entry]) -> Option<Self> {
        let (store, layout) = private_store(blob, dir)?;
        Some(Self::assemble(dir, &layout.offsets, Cow::Owned(store)))
    }

    /// The tables in the image called `name`, built here if no other process holds
    /// it yet. The error says why, in a form fit for `info string`.
    fn shared(blob: &[u8], dir: &[Entry], name: &str) -> Result<(Self, Origin), String> {
        let header = image_header_len(dir.len());
        let capacity = store_capacity(dir);
        let mapped = crate::shm::acquire(name, header + capacity, &mut |dst| {
            let (head, store) = dst.split_at_mut(header);
            let layout = fill_store(blob, dir, store);
            write_header(head, dir.len(), layout.as_ref());
        })?;
        let bytes = mapped.bytes();
        let layout = read_header(&bytes[..header], dir.len(), capacity)?;
        let store = &bytes[header..header + layout.used];

        // The name pins the blob and the image format, not what this build writes
        // into that format. Checking mode rebuilds and compares, so a change that
        // forgot to bump IMAGE_FORMAT is caught here instead of probing wrong
        // distances.
        if crate::shm::verify_requested() {
            let (mine, mine_layout) =
                private_store(blob, dir).ok_or("the tables do not decompress")?;
            if mine_layout != layout || mine[..] != *store {
                return Err(String::from(
                    "shared image disagrees with this build (bump IMAGE_FORMAT)",
                ));
            }
        }
        Ok((
            Self::assemble(dir, &layout.offsets, Cow::Borrowed(store)),
            mapped.origin,
        ))
    }

    /// The prober over a store already filled, wherever it lives.
    fn assemble(
        dir: &[Entry],
        offsets: &[(usize, usize)],
        packed_data: Cow<'static, [u8]>,
    ) -> Self {
        debug_assert_eq!(dir.len(), offsets.len());
        let tables = dir
            .iter()
            .zip(offsets)
            .map(|(e, &(packed_offset, btm_packed_offset))| {
                debug_assert!(packed_offset <= btm_packed_offset);
                debug_assert!(btm_packed_offset <= packed_data.len());
                let (has_pawns, wp, bp, groups) = parse_sig_to_indexer(&e.sig);
                ProberTable {
                    sig: e.sig,
                    flags: e.flags,
                    bits_per_entry: e.bits,
                    max_dtm: e.max_dtm,
                    per_stm_size: e.per_stm,
                    total_positions: e.total_positions,
                    packed_offset,
                    btm_packed_offset,
                    has_pawns,
                    num_white_pawns: wp,
                    num_black_pawns: bp,
                    non_pawn_groups: groups,
                }
            })
            .collect();
        Prober {
            tables,
            packed_data,
        }
    }

    /// Probe DTM. Returns `Some(dtm)` where dtm>0 means STM wins, dtm<0 STM loses, 0=draw.
    fn probe_dtm(
        &self,
        white: &MatCounts,
        black: &MatCounts,
        wk: u8,
        bk: u8,
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
            for &sq in &piece_sqs[owp..owp + obp] {
                enc_pieces[pos] = sq ^ 56;
                pos += 1;
            }
            for &sq in &piece_sqs[..owp] {
                enc_pieces[pos] = sq ^ 56;
                pos += 1;
            }
            let np_start = owp + obp;
            pos = obp + owp;
            for &sq in &piece_sqs[np_start + ownp..np_start + ownp + obnp] {
                enc_pieces[pos] = sq ^ 56;
                pos += 1;
            }
            for &sq in &piece_sqs[np_start..np_start + ownp] {
                enc_pieces[pos] = sq ^ 56;
                pos += 1;
            }
        } else {
            enc_pieces[..num_pieces].copy_from_slice(piece_sqs);
        }

        let idx = encode(
            enc_wk,
            enc_bk,
            &enc_pieces[..num_pieces],
            table.has_pawns,
            table.num_white_pawns,
            table.num_black_pawns,
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
            let adj = if enc_stm_black {
                idx + table.per_stm_size as usize
            } else {
                idx
            };
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
    for &(side, ch) in &[
        (white_str, b'Q'),
        (white_str, b'R'),
        (white_str, b'B'),
        (white_str, b'N'),
        (black_str, b'Q'),
        (black_str, b'R'),
        (black_str, b'B'),
        (black_str, b'N'),
    ] {
        let c = count_piece(side, ch);
        if c > 0 {
            groups.push(c);
        }
    }
    (has_pawns, wp, bp, groups)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{mate_in, mated_in};

    #[test]
    fn stored_distances_are_one_half_move_too_long() {
        for ply in [0, 1, 6] {
            assert_eq!(stored_dtm_to_score(0, ply), 0);
            assert_eq!(
                stored_dtm_to_score(-1, ply),
                mated_in(ply),
                "checkmated now"
            );
            assert_eq!(
                stored_dtm_to_score(2, ply),
                mate_in(ply + 1),
                "mate in one half-move"
            );
            assert_eq!(stored_dtm_to_score(-2, ply), mated_in(ply + 1));
            assert_eq!(stored_dtm_to_score(8, ply), mate_in(ply + 7));
        }
    }

    /// End to end on the embedded tables: a rook mate in one, and the position after
    /// it, seen from the mated side.
    #[cfg(gaiatb_embedded)]
    #[test]
    fn the_embedded_tables_score_a_mate_in_one_as_such() {
        assert!(init(), "embedded tables must load");
        let pos = Position::from_fen("7k/8/6K1/8/8/8/8/R7 w - - 0 1").unwrap();
        assert_eq!(probe_position(&pos, 0), Some(mate_in(1)));
        let mated = Position::from_fen("R6k/8/6K1/8/8/8/8/8 b - - 1 1").unwrap();
        assert_eq!(probe_position(&mated, 1), Some(mated_in(1)));
    }

    /// A mate is proved only within what the 50-move rule leaves: at the clock where it
    /// lands exactly on the deadline it stands, one half-move later it is declined (the
    /// probe used to ignore the clock, issue #24 for Nalimov). A mate on the board
    /// stands at any clock.
    #[cfg(gaiatb_embedded)]
    #[test]
    fn a_table_mate_beyond_the_fifty_move_deadline_is_declined() {
        assert!(init(), "embedded tables must load");
        let fen = |hmc: i32| format!("8/8/8/4k3/8/8/8/KR6 w - - {hmc} 80");
        let fresh =
            probe_position(&Position::from_fen(&fen(0)).unwrap(), 0).expect("KRK is in the tables");
        let distance = crate::types::SCORE_MATE - fresh;
        assert!(
            distance > 1 && distance < 100,
            "a long rook mate, got {distance}"
        );

        let at = |hmc: i32| probe_position(&Position::from_fen(&fen(hmc)).unwrap(), 0);
        assert_eq!(
            at(100 - distance),
            Some(fresh),
            "the mate lands on the deadline"
        );
        assert_eq!(at(101 - distance), None, "one half-move too late");
        assert_eq!(at(99), None);

        let mated = Position::from_fen("R6k/8/6K1/8/8/8/8/8 b - - 99 120").unwrap();
        assert_eq!(
            probe_position(&mated, 1),
            Some(mated_in(1)),
            "mated is mated"
        );
    }

    /// The store is sized from the directory before anything is decoded, and a blob
    /// that outgrew it would not load at all, shared or not.
    #[cfg(gaiatb_embedded)]
    #[test]
    fn the_embedded_tables_fit_the_store_their_directory_predicts() {
        let dir = read_directory(BLOB).expect("embedded blob");
        let (store, layout) = private_store(BLOB, &dir).expect("the tables decompress");
        assert_eq!(store.len(), layout.used);
        assert!(layout.used <= store_capacity(&dir));
        assert!(
            layout.used > 100 << 20,
            "~155 MB of tables, got {}",
            layout.used
        );
    }

    #[cfg(gaiatb_embedded)]
    #[test]
    fn a_store_too_small_is_refused_not_overrun() {
        let dir = read_directory(BLOB).unwrap();
        let mut store = vec![0u8; store_capacity(&dir) / 2];
        assert!(fill_store(BLOB, &dir, &mut store).is_none());
    }

    #[test]
    fn the_image_header_round_trips_and_refuses_a_failed_build() {
        let layout = Layout {
            offsets: vec![(0, 0), (0, 10), (10, 25)],
            used: 40,
        };
        let mut head = vec![0xffu8; image_header_len(3)];
        write_header(&mut head, 3, Some(&layout));
        assert_eq!(read_header(&head, 3, 100), Ok(layout.clone()));
        assert!(
            read_header(&head, 3, 39).is_err(),
            "more used than the store holds"
        );

        let stray = Layout {
            offsets: vec![(0, 0), (0, 10), (10, 41)],
            used: 40,
        };
        write_header(&mut head, 3, Some(&stray));
        assert!(read_header(&head, 3, 100).is_err(), "a table past the end");

        write_header(&mut head, 3, None);
        let err = read_header(&head, 3, 100).unwrap_err();
        assert!(err.contains("did not fit"), "{err}");

        write_header(&mut head, 3, Some(&layout));
        head[0] ^= 1;
        assert!(read_header(&head, 3, 100).is_err(), "not our magic");
    }

    /// A first asker builds the image, a second receives it, and both read what a
    /// private load holds — byte for byte, and probe for probe.
    #[cfg(all(gaiatb_embedded, target_os = "linux"))]
    #[test]
    fn a_shared_image_probes_like_a_private_copy() {
        let dir = read_directory(BLOB).unwrap();
        // Not the engine's name: the abstract namespace is machine-wide.
        let name = format!("gaiatb-test-{}", std::process::id());
        let (built, origin) = Prober::shared(BLOB, &dir, &name).expect("build");
        assert_eq!(origin, Origin::Built);
        let (received, origin) = Prober::shared(BLOB, &dir, &name).expect("receive");
        assert_eq!(origin, Origin::Received);

        let private = Prober::private(BLOB, &dir).unwrap();
        for shared in [&built, &received] {
            assert!(matches!(shared.packed_data, Cow::Borrowed(_)));
            assert!(shared.packed_data[..] == private.packed_data[..]);
            for (a, b) in shared.tables.iter().zip(&private.tables) {
                assert_eq!(
                    (a.packed_offset, a.btm_packed_offset),
                    (b.packed_offset, b.btm_packed_offset)
                );
            }
        }

        let fens = [
            "7k/8/6K1/8/8/8/8/R7 w - - 0 1",
            "8/8/8/4k3/8/8/4P3/4K3 w - - 0 1",
            "8/8/8/3k4/8/1K6/2Q5/7r b - - 0 1",
            "8/8/4k3/8/8/2N5/8/R3K3 b - - 0 1",
            "8/5p2/8/4k3/8/8/2P5/4K3 w - - 0 1",
        ];
        for fen in fens {
            let pos = Position::from_fen(fen).unwrap();
            let want = probe_with(&private, &pos, 3);
            assert!(want.is_some(), "{fen} is in the tables");
            assert_eq!(probe_with(&received, &pos, 3), want, "{fen}");
        }
    }
}
