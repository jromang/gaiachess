//! Online tablebase probing — probe Lichess Syzygy API asynchronously.
//!
//! Two modes of operation:
//! 1. **Prefetch** (after bestmove): generates opponent replies and queues them
//! 2. **Async search** (during search): the search submits positions on cache miss,
//!    results arrive for subsequent iterative deepening iterations
//!
//! A single background thread consumes the queue at 1 request/sec.
//! Results are cached in a HashMap keyed by Zobrist hash.
//!
//! Gated behind `--features online-tb`. Runtime toggle via UCI `OnlineTB` option.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use serde::Deserialize;

use crate::movegen;
use crate::position::Position;
use crate::types::*;

// ── API response ─────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct LichessResponse {
    dtm: Option<i32>,
    #[allow(dead_code)]
    dtz: Option<i32>,
    category: String,
}

// ── Cache entry ──────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct OnlineTbEntry {
    /// DTM in halfmoves (plies), STM-relative. Positive = STM wins, negative = STM loses.
    /// None if the API returned null (some 7-piece positions).
    pub dtm: Option<i32>,
    /// Category: Win / Draw / Loss from STM perspective.
    pub category: TbCategory,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TbCategory {
    Win,
    Draw,
    Loss,
}

// ── Probe request ────────────────────────────────────────────────────────────

struct ProbeRequest {
    key: u64,
    fen: String,
}

// ── Global state ─────────────────────────────────────────────────────────────

struct OnlineTbState {
    cache: Mutex<HashMap<u64, OnlineTbEntry>>,
    queue: Mutex<VecDeque<ProbeRequest>>,
    pending: Mutex<HashSet<u64>>, // keys already in queue (deduplication)
    cancel: Mutex<Arc<AtomicBool>>,
    enabled: AtomicBool,
}

static STATE: OnceLock<OnlineTbState> = OnceLock::new();

fn state() -> &'static OnlineTbState {
    STATE.get_or_init(|| OnlineTbState {
        cache: Mutex::new(HashMap::new()),
        queue: Mutex::new(VecDeque::new()),
        pending: Mutex::new(HashSet::new()),
        cancel: Mutex::new(Arc::new(AtomicBool::new(false))),
        enabled: AtomicBool::new(false),
    })
}

// ── Public API ───────────────────────────────────────────────────────────────

pub fn is_enabled() -> bool {
    state().enabled.load(Relaxed)
}

pub fn set_enabled(enabled: bool) {
    state().enabled.store(enabled, Relaxed);
}

/// Check cache for a previously probed result.
/// Uses `try_lock` to avoid blocking the search — returns None if contended.
pub fn probe(key: u64) -> Option<OnlineTbEntry> {
    let cache = state().cache.try_lock().ok()?;
    cache.get(&key).copied()
}

/// Submit a position for async probing (low priority, appended to queue end).
/// Used by prefetch (between moves). No-op if already cached or queued.
pub fn submit(key: u64, fen: String) {
    submit_inner(key, fen, false);
}

/// Submit a position for async probing (high priority, pushed to queue front).
/// Used by the search (during alpha_beta) — more urgent than prefetch positions.
/// No-op if already cached or queued.
pub fn submit_priority(key: u64, fen: String) {
    submit_inner(key, fen, true);
}

/// Check if a position is already cached or queued (avoids `to_fen()` allocation).
/// Uses `try_lock` — returns false (conservative) if contended.
pub fn is_known(key: u64) -> bool {
    let s = state();
    if let Ok(cache) = s.cache.try_lock() {
        if cache.contains_key(&key) { return true; }
    }
    if let Ok(pending) = s.pending.try_lock() {
        if pending.contains(&key) { return true; }
    }
    false
}

/// Uses `try_lock` to avoid blocking the search — silently drops if contended.
fn submit_inner(key: u64, fen: String, priority: bool) {
    let s = state();
    let Ok(mut pending) = s.pending.try_lock() else { return };
    if !pending.insert(key) {
        return;
    }
    let Ok(mut queue) = s.queue.try_lock() else { return };
    if priority {
        queue.push_front(ProbeRequest { key, fen });
    } else {
        queue.push_back(ProbeRequest { key, fen });
    }
}

/// Submit positions along the PV for async probing.
///
/// Priority 1: PV positions with ≤7 pieces → `submit_priority` (front of queue).
/// Priority 2: at each PV node with exactly 8 pieces, captures that lead to
///             ≤7 pieces → `submit` (back of queue, PV-adjacent fallback).
pub fn submit_pv_positions(pos: &Position, pv: &[Move], pv_len: usize) {
    let mut scratch = pos.clone();
    for i in 0..pv_len {
        let mv = pv[i];
        if mv == Move::NONE { break; }
        scratch.make_move(mv);
        let pieces = scratch.occupied().count_ones();
        if scratch.castling_rights != 0 { continue; }

        if pieces <= 7 {
            // Priority 1: direct PV position, TB-eligible
            if !is_known(scratch.key) {
                submit_priority(scratch.key, scratch.to_fen());
            }
        } else if pieces == 8 {
            // Priority 2: PV-adjacent captures (8 → ≤7 pieces)
            let mut buf = ArrayBuf::new();
            let count = movegen::generate_legal_moves(&scratch, &mut buf);
            for j in 0..count {
                let cap = buf[j];
                // Filter captures only (piece on destination, or en passant)
                if scratch.board[cap.to_sq().index()] == Piece::NONE
                    && cap.move_type() != MT_EN_PASSANT { continue; }
                let mut after = scratch.clone();
                after.make_move(cap);
                if after.occupied().count_ones() <= 7 && !is_known(after.key) {
                    submit(after.key, after.to_fen());
                }
            }
        }
    }
}

/// Clear cache and queue (called on ucinewgame).
pub fn clear() {
    cancel_worker();
    let s = state();
    s.cache.lock().unwrap().clear();
    s.queue.lock().unwrap().clear();
    s.pending.lock().unwrap().clear();
}

fn cancel_worker() {
    let mut cancel = state().cancel.lock().unwrap();
    cancel.store(true, Relaxed);
    *cancel = Arc::new(AtomicBool::new(false));
    // Also clear the queue so stale requests don't carry over
    state().queue.lock().unwrap().clear();
    state().pending.lock().unwrap().clear();
}

/// Spawn the background worker and queue prefetch positions.
///
/// Priority 1: PV positions (deep continuation) via `submit_pv_positions`.
/// Priority 2: opponent replies sorted by ponder → captures MVV → promotions → quiets.
pub fn start_prefetch(pos: Position, best_move: Move, ponder_move: Option<Move>,
                      pv: &[Move], pv_len: usize) {
    let s = state();

    // Cancel any running worker
    let mut cancel_guard = s.cancel.lock().unwrap();
    cancel_guard.store(true, Relaxed);
    let cancel = Arc::new(AtomicBool::new(false));
    *cancel_guard = cancel.clone();
    drop(cancel_guard);

    // Clear stale queue
    s.queue.lock().unwrap().clear();
    s.pending.lock().unwrap().clear();

    // Priority 1: PV positions (high priority, front of queue)
    submit_pv_positions(&pos, pv, pv_len);

    // Priority 2: opponent replies (low priority, back of queue)
    let mut opp_pos = pos;
    opp_pos.make_move(best_move);

    let mut buf = ArrayBuf::new();
    let count = movegen::generate_legal_moves(&opp_pos, &mut buf);

    let mut moves: Vec<Move> = (0..count).map(|i| buf[i]).collect();
    moves.sort_by_key(|m| {
        if Some(*m) == ponder_move {
            -1000
        } else {
            let captured = opp_pos.board[m.to_sq().index()];
            if captured != Piece::NONE {
                -PIECE_VALUE[captured.piece_type() as usize]
            } else if m.move_type() == MT_EN_PASSANT {
                -PIECE_VALUE[PieceType::Pawn as usize]
            } else if m.move_type() == MT_PROMOTION {
                0
            } else {
                100
            }
        }
    });

    for mv in moves {
        let mut next = opp_pos.clone();
        next.make_move(mv);
        if next.occupied().count_ones() > 7 {
            continue;
        }
        submit(next.key, next.to_fen());
    }

    // Spawn the background worker
    std::thread::spawn(move || {
        worker_loop(cancel);
    });
}

/// Spawn a background worker (without prefetch) for async search probes.
/// Called at the start of a search if no prefetch worker is running.
pub fn ensure_worker_running() {
    let s = state();
    let mut cancel_guard = s.cancel.lock().unwrap();
    // If the current cancel flag is already set (no worker running), spawn one
    if cancel_guard.load(Relaxed) || Arc::strong_count(&cancel_guard) == 1 {
        let cancel = Arc::new(AtomicBool::new(false));
        *cancel_guard = cancel.clone();
        drop(cancel_guard);
        std::thread::spawn(move || {
            worker_loop(cancel);
        });
    }
}

// ── Background worker ────────────────────────────────────────────────────────

fn worker_loop(cancel: Arc<AtomicBool>) {
    let s = state();

    loop {
        if cancel.load(Relaxed) {
            return;
        }

        // Pop next request from queue
        let request = s.queue.lock().unwrap().pop_front();

        if let Some(req) = request {
            // Skip if already cached (could have been filled by another path)
            if s.cache.lock().unwrap().contains_key(&req.key) {
                s.pending.lock().unwrap().remove(&req.key);
                continue;
            }

            match probe_lichess(&req.fen) {
                Ok(entry) => {
                    s.cache.lock().unwrap().insert(req.key, entry);
                    s.pending.lock().unwrap().remove(&req.key);
                    let dtm_str = match entry.dtm {
                        Some(d) => format!("{d}"),
                        None => "null".to_string(),
                    };
                    eprintln!(
                        "info string OnlineTB probe: dtm={} ({:?}) {}",
                        dtm_str, entry.category, req.fen
                    );
                }
                Err(e) => {
                    s.pending.lock().unwrap().remove(&req.key);
                    eprintln!("info string OnlineTB error: {e}");
                }
            }

            if cancel.load(Relaxed) {
                return;
            }

            // Rate limit: 1 request per second
            std::thread::sleep(Duration::from_secs(1));
        } else {
            // Queue empty: short sleep and re-check
            std::thread::sleep(Duration::from_millis(100));
        }
    }
}

// ── HTTP probe ───────────────────────────────────────────────────────────────

fn probe_lichess(fen: &str) -> Result<OnlineTbEntry, String> {
    let url = format!(
        "https://tablebase.lichess.ovh/standard?fen={}",
        fen.replace(' ', "_")
    );

    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(5)))
        .build()
        .new_agent();

    let body: String = agent
        .get(&url)
        .call()
        .map_err(|e| e.to_string())?
        .body_mut()
        .read_to_string()
        .map_err(|e| e.to_string())?;

    let resp: LichessResponse =
        serde_json::from_str(&body).map_err(|e| e.to_string())?;

    let category = match resp.category.as_str() {
        "win" => TbCategory::Win,
        "loss" => TbCategory::Loss,
        _ => TbCategory::Draw,
    };

    Ok(OnlineTbEntry {
        dtm: if category == TbCategory::Draw { Some(0) } else { resp.dtm },
        category,
    })
}
