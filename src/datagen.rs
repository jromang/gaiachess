//! Self-play data generation for NNUE training.
//!
//! Outputs games in **Viriformat** (`.vf`) compatible with `ViriBinpackLoader`.
//! Each game = initial position + move sequence + evals + outcome.
//!
//! Usage: `gaiachess datagen [threads] [positions] [depth] [output] [book.epd]`

use std::io::{BufRead, BufWriter, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use time::OffsetDateTime;

use indicatif::{ProgressBar, ProgressStyle};

use viriformat::chess::board::{Board as ViriBoard, GameOutcome, DrawType, WinType};
use viriformat::chess::chessmove::{Move as ViriMove, MoveFlags};
use viriformat::chess::types::Square as ViriSquare;
use viriformat::chess::piece::PieceType as ViriPieceType;
use viriformat::dataformat::Game;

use crate::movegen;
use crate::position::Position;
use crate::search;
use crate::threads::{SharedState, ThreadData, STOP};
use crate::timeman::SearchLimits;
use crate::types::*;
use crate::see;

// ============================================================
// Bulletformat ChessBoard (32 bytes)
// ============================================================

/// Bullet trainer `ChessBoard` format: 32 bytes per position.
///
/// Everything is **STM-relative**: when Black is to move, the board is
/// vertically flipped and colors are swapped.
#[repr(C)]
#[derive(Clone, Copy)]
struct ChessBoard {
    /// Occupancy bitboard (STM perspective).
    occ: u64,
    /// Packed piece nibbles: 2 per byte, in ascending square order of `occ`.
    /// Nibble = `(color << 3) | piece_type` where color 0 = friendly, 1 = enemy.
    pcs: [u8; 16],
    /// Centipawn evaluation, STM-relative.
    score: i16,
    /// Game result from STM perspective: 0=loss, 1=draw, 2=win.
    result: u8,
    /// Friendly king square (STM perspective).
    ksq: u8,
    /// Opponent king square XOR 56.
    opp_ksq: u8,
    /// Reserved (zeroed).
    extra: [u8; 3],
}

impl ChessBoard {
    /// Create a ChessBoard from the engine's Position + eval + result.
    ///
    /// `score_white`: centipawn eval from White's perspective.
    /// `result_white`: 2=white win, 1=draw, 0=white loss.
    fn from_position(pos: &Position, score_white: i16, result_white: u8) -> Self {
        let stm = pos.side_to_move;
        let is_black = stm == Color::Black;

        // Build the 8 bitboards: [white, black, pawn, knight, bishop, rook, queen, king]
        let mut bbs = [0u64; 8];
        bbs[0] = pos.color_bb(Color::White);
        bbs[1] = pos.color_bb(Color::Black);
        for pt in 0..6 {
            let pt_enum = match pt {
                0 => PieceType::Pawn,
                1 => PieceType::Knight,
                2 => PieceType::Bishop,
                3 => PieceType::Rook,
                4 => PieceType::Queen,
                _ => PieceType::King,
            };
            bbs[2 + pt] = pos.piece_type_bb(pt_enum, Color::White)
                | pos.piece_type_bb(pt_enum, Color::Black);
        }

        // If Black to move: flip board vertically, swap colors
        if is_black {
            for bb in &mut bbs {
                *bb = bb.swap_bytes();
            }
            bbs.swap(0, 1);
        }

        let occ = bbs[0] | bbs[1];

        // Score and result: convert to STM-relative
        let score = if is_black { -(score_white as i32) as i16 } else { score_white };
        let result = if is_black { 2 - result_white } else { result_white };

        // Pack pieces in order of ascending square in occupancy
        let mut pcs = [0u8; 16];
        let mut occ_copy = occ;
        let mut idx = 0;
        while occ_copy != 0 {
            let sq = occ_copy.trailing_zeros() as usize;
            occ_copy &= occ_copy - 1;

            let sq_bb = 1u64 << sq;

            // Determine piece type
            let piece_type = if bbs[2] & sq_bb != 0 {
                0 // Pawn
            } else if bbs[3] & sq_bb != 0 {
                1 // Knight
            } else if bbs[4] & sq_bb != 0 {
                2 // Bishop
            } else if bbs[5] & sq_bb != 0 {
                3 // Rook
            } else if bbs[6] & sq_bb != 0 {
                4 // Queen
            } else {
                5 // King
            };

            // Color: 0 = friendly (STM), 1 = opponent
            let color = if bbs[0] & sq_bb != 0 { 0u8 } else { 1u8 };

            let nibble = (color << 3) | piece_type;
            if idx & 1 == 0 {
                pcs[idx / 2] = nibble;
            } else {
                pcs[idx / 2] |= nibble << 4;
            }
            idx += 1;
        }

        // King squares
        let friendly_king_bb = bbs[0] & bbs[7];
        let opponent_king_bb = bbs[1] & bbs[7];
        debug_assert!(friendly_king_bb.count_ones() == 1,
            "datagen: {} friendly kings", friendly_king_bb.count_ones());
        debug_assert!(opponent_king_bb.count_ones() == 1,
            "datagen: {} opponent kings", opponent_king_bb.count_ones());
        let ksq = friendly_king_bb.trailing_zeros() as u8;
        let opp_ksq = opponent_king_bb.trailing_zeros() as u8 ^ 56;
        debug_assert!(ksq < 64, "datagen: ksq {} OOB", ksq);
        debug_assert!((opp_ksq ^ 56) < 64, "datagen: opp_ksq raw {} OOB", opp_ksq ^ 56);
        debug_assert!(result <= 2, "datagen: result {} > 2", result);
        debug_assert!((score as i32).abs() < 30000,
            "datagen: score {} suspicious", score);
        debug_assert!(idx >= 2 && idx <= 32,
            "datagen: piece count {} invalid", idx);

        ChessBoard {
            occ,
            pcs,
            score,
            result,
            ksq,
            opp_ksq,
            extra: [0; 3],
        }
    }

    /// Serialize to 32 raw bytes (little-endian).
    fn to_bytes(&self) -> [u8; 32] {
        let mut buf = [0u8; 32];
        buf[0..8].copy_from_slice(&self.occ.to_le_bytes());
        buf[8..24].copy_from_slice(&self.pcs);
        buf[24..26].copy_from_slice(&self.score.to_le_bytes());
        buf[26] = self.result;
        buf[27] = self.ksq;
        buf[28] = self.opp_ksq;
        // buf[29..32] = extra = [0, 0, 0]
        buf
    }
}

// ============================================================
// Datagen configuration
// ============================================================

/// Configuration for data generation.
struct DatagenConfig {
    /// Search depth for each move during self-play.
    depth: i32,
    /// Soft node limit (0 = no limit, use depth only).
    soft_nodes: u64,
    /// Number of random opening moves (per side).
    random_moves: usize,
    /// Maximum |eval| to accept opening position.
    max_opening_eval: i32,
    /// Maximum |eval| for position recording.
    max_record_eval: i32,
    /// Win adjudication: |score| >= this for N consecutive plies.
    win_adj_score: i32,
    /// Win adjudication: consecutive plies.
    win_adj_plies: usize,
    /// Draw adjudication: |score| <= this for N consecutive plies.
    draw_adj_score: i32,
    /// Draw adjudication: consecutive plies.
    draw_adj_plies: usize,
    /// Minimum ply before recording positions.
    min_record_ply: usize,
}

impl Default for DatagenConfig {
    fn default() -> Self {
        DatagenConfig {
            depth: 8,
            soft_nodes: 5000,
            random_moves: 8,
            max_opening_eval: 1000,
            max_record_eval: 20000,
            win_adj_score: 2500,
            win_adj_plies: 4,
            draw_adj_score: 4,
            draw_adj_plies: 12,
            min_record_ply: 16,
        }
    }
}

// ============================================================
// Self-play game
// ============================================================

/// Global game counter.
static GAMES_PLAYED: AtomicU64 = AtomicU64::new(0);
/// Graceful shutdown flag.
static DATAGEN_STOP: AtomicBool = AtomicBool::new(false);
/// Explosion counter (positions that hit the per-thread search deadline).
static EXPLOSIONS: AtomicU64 = AtomicU64::new(0);
/// TB adjudication counter.
static TB_ADJUDICATIONS: AtomicU64 = AtomicU64::new(0);
/// Max explosion FENs to collect (avoid unbounded memory).
const MAX_EXPLOSION_FENS: usize = 50;

/// Simple xorshift64 RNG.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.max(1))
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Random usize in [0, n).
    fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Prepare a ThreadData for datagen search.
///
/// Unlike `prepare_search`, this always applies the given limits
/// (the normal method gives infinite limits to non-main threads).
/// Per-thread search deadline for datagen (milliseconds).
/// Prevents individual searches from running forever on explosion-prone positions.
/// Iterative deepening guarantees a valid result from the last completed depth.
const DATAGEN_SEARCH_DEADLINE: u64 = 10_000; // 10 seconds

fn prepare_datagen_search(td: &mut ThreadData, pos: &Position, limits: &SearchLimits) {
    td.prepare_search(pos, limits);
    // Override: prepare_search gives id!=0 infinite limits, we need real limits
    td.tm = crate::timeman::TimeManager::new(limits);
    // Per-thread deadline: caps explosion-prone searches without global STOP
    td.search_deadline = DATAGEN_SEARCH_DEADLINE;
    // Skip in-search TB probing and root TB ranking — datagen adjudicates via TB separately
    td.datagen_mode = true;
}

/// Apply N random SEE-filtered moves to the position.
/// Returns false if the game ends during randomization (no legal moves).
fn apply_random_moves(pos: &mut Position, n: usize, rng: &mut Rng) -> bool {
    for _ in 0..n {
        let mut buf: ArrayBuf<Move, MAX_MOVES> = ArrayBuf::new();
        let count = movegen::generate_legal_moves(pos, &mut buf);
        if count == 0 {
            return false;
        }
        let mut chosen = Move::NONE;
        for _ in 0..8 {
            let idx = rng.next_usize(count);
            let m = buf[idx];
            if see::see(pos, m, -100) {
                chosen = m;
                break;
            }
        }
        if chosen == Move::NONE {
            chosen = buf[rng.next_usize(count)];
        }
        pos.make_move(chosen);
    }
    true
}

/// Play one self-play game, returning a list of ChessBoard records.
///
/// Each side has its own TT and ThreadData (prevents information leakage).
/// Convert GaiaChess Move to viriformat Move.
/// GaiaChess: from(6) << 6 | to(6) | flags << 14 | promo << 12
/// Viriformat: from(6) | to(6) << 6 | flags << 12-14
fn to_viri_move(m: Move, pos: &Position) -> ViriMove {
    let from = unsafe { ViriSquare::new_unchecked(m.from_sq().0) };
    let to = unsafe { ViriSquare::new_unchecked(m.to_sq().0) };
    match m.move_type() {
        MT_PROMOTION => {
            let pt = match m.promo_type() {
                PieceType::Knight => ViriPieceType::Knight,
                PieceType::Bishop => ViriPieceType::Bishop,
                PieceType::Rook => ViriPieceType::Rook,
                _ => ViriPieceType::Queen,
            };
            ViriMove::new_with_promo(from, to, pt)
        }
        MT_EN_PASSANT => ViriMove::new_with_flags(from, to, MoveFlags::EnPassant),
        MT_CASTLING => {
            // Verify it's a real castle (from = king square)
            let king_sq = pos.king_sq(pos.side_to_move);
            if m.from_sq() != king_sq {
                // Corrupted move type — treat as normal move
                ViriMove::new(from, to)
            } else {
                // Viriformat expects to = rook square (Chess960 convention)
                let rook_sq = match m.to_sq().0 {
                    6  => 7,  // g1 → h1 (O-O white)
                    2  => 0,  // c1 → a1 (O-O-O white)
                    62 => 63, // g8 → h8 (O-O black)
                    58 => 56, // c8 → a8 (O-O-O black)
                    s  => s,
                };
                let rook_to = unsafe { ViriSquare::new_unchecked(rook_sq) };
                ViriMove::new_with_flags(from, rook_to, MoveFlags::Castle)
            }
        }
        _ => ViriMove::new(from, to),
    }
}

/// Convert game result (white-relative: 2=win, 1=draw, 0=loss) to viriformat GameOutcome.
fn to_viri_outcome(result_white: u8) -> GameOutcome {
    match result_white {
        2 => GameOutcome::WhiteWin(WinType::Adjudication),
        0 => GameOutcome::BlackWin(WinType::Adjudication),
        _ => GameOutcome::Draw(DrawType::Adjudication),
    }
}

fn play_game(
    td_white: &mut ThreadData,
    td_black: &mut ThreadData,
    shared_white: &SharedState,
    shared_black: &SharedState,
    config: &DatagenConfig,
    book: Option<&[String]>,
    rng: &mut Rng,
    explosion_fens: &Mutex<Vec<String>>,
) -> Option<Game> {
    // Phase 1: Opening position
    let mut pos = if let Some(lines) = book {
        // Pick a random line from the opening book + random moves for diversity
        let idx = rng.next_usize(lines.len());
        let mut p = match Position::from_fen(&lines[idx]) {
            Ok(p) => p,
            Err(_) => return None,
        };
        let num_random = config.random_moves + (rng.next_u64() & 1) as usize;
        if !apply_random_moves(&mut p, num_random, rng) {
            return None;
        }
        p
    } else {
        // No book: startpos + random moves
        let mut p = Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            .expect("startpos");
        let num_random = config.random_moves + (rng.next_u64() & 1) as usize;
        if !apply_random_moves(&mut p, num_random, rng) {
            return None;
        }
        p
    };

    // Phase 2: Verify opening with a quick search
    {
        let limits = if config.soft_nodes > 0 {
            SearchLimits::Nodes(config.soft_nodes * 10) // 10x nodes for verification
        } else {
            SearchLimits::Depth(config.depth + 2)
        };
        let td = if pos.side_to_move == Color::White { &mut *td_white } else { &mut *td_black };
        let shared = if pos.side_to_move == Color::White { &*shared_white } else { &*shared_black };

        STOP.store(false, Ordering::Relaxed);
        prepare_datagen_search(td, &pos, &limits);
        search::search(td, shared);

        // Per-thread deadline hit: skip this game (position is too tactical for PeSTO)
        if td.stopped {
            EXPLOSIONS.fetch_add(1, Ordering::Relaxed);
            if let Ok(mut fens) = explosion_fens.lock() {
                if fens.len() < MAX_EXPLOSION_FENS {
                    fens.push(pos.to_fen());
                }
            }
            return None;
        }

        if td.best_score.abs() > config.max_opening_eval {
            return None; // Position too unbalanced, retry
        }
    }

    // Phase 3: Play out game — build viriformat Game
    let mut viri_board = ViriBoard::new();
    viri_board.set_from_fen(&pos.to_fen()).unwrap();
    let mut game = Game::new(&viri_board);
    let mut game_ply = 0usize;
    let mut win_adj_count = 0usize;
    let mut draw_adj_count = 0usize;

    // Game result: 2=white win, 1=draw, 0=white loss
    let result: u8;

    loop {
        if DATAGEN_STOP.load(Ordering::Relaxed) {
            return None;
        }

        // Check for draws
        if pos.is_draw(i32::MAX) {
            result = 1;
            break;
        }

        // Check for legal moves (mate/stalemate)
        let mut buf: ArrayBuf<Move, MAX_MOVES> = ArrayBuf::new();
        let count = movegen::generate_legal_moves(&pos, &mut buf);
        if count == 0 {
            if pos.checkers != 0 {
                // Checkmate: side to move loses
                result = if pos.side_to_move == Color::White { 0 } else { 2 };
            } else {
                // Stalemate
                result = 1;
            }
            break;
        }

        // Syzygy TB adjudication: perfect endgame result
        #[cfg(feature = "syzygy")]
        if crate::tb::max_pieces() > 0
            && pos.occupied().count_ones() <= crate::tb::max_pieces()
        {
            if let Some(wdl) = crate::tb::probe_wdl(&pos) {
                // WDL is STM-relative, convert to white-relative result
                result = match (wdl, pos.side_to_move) {
                    (crate::tb::Wdl::Win, Color::White) | (crate::tb::Wdl::Loss, Color::Black) => 2,
                    (crate::tb::Wdl::Loss, Color::White) | (crate::tb::Wdl::Win, Color::Black) => 0,
                    (crate::tb::Wdl::Draw, _) => 1,
                };
                TB_ADJUDICATIONS.fetch_add(1, Ordering::Relaxed);
                break;
            }
        }

        // Search for best move
        let limits = if config.soft_nodes > 0 {
            SearchLimits::Nodes(config.soft_nodes)
        } else {
            SearchLimits::Depth(config.depth)
        };

        let (td, shared) = if pos.side_to_move == Color::White {
            (&mut *td_white, &*shared_white)
        } else {
            (&mut *td_black, &*shared_black)
        };

        STOP.store(false, Ordering::Relaxed);
        prepare_datagen_search(td, &pos, &limits);
        search::search(td, shared);

        // Per-thread deadline hit: skip this game (position too tactical for PeSTO)
        if td.stopped {
            EXPLOSIONS.fetch_add(1, Ordering::Relaxed);
            if let Ok(mut fens) = explosion_fens.lock() {
                if fens.len() < MAX_EXPLOSION_FENS {
                    fens.push(pos.to_fen());
                }
            }
            return None;
        }

        let best_move = td.best_move;
        let score = td.best_score;
        debug_assert!(score != SCORE_NONE, "datagen: SCORE_NONE from search");
        debug_assert!(score.abs() <= SCORE_INFINITE,
            "datagen: score {} out of range", score);

        if best_move == Move::NONE {
            result = 1; // Shouldn't happen, but safety
            break;
        }

        // Add move + eval to viriformat game. The viriformat→bulletformat chain expects a
        // WHITE-relative score: bulletformat::ChessBoard::from_raw does `score = -score`
        // when Black is to move to produce the STM-POV training target.
        // Writing an STM-relative score here flips the sign of ~50% of targets (gen0/gen1 bug).
        let viri_mv = to_viri_move(best_move, &pos);
        let white_score = if pos.side_to_move == Color::White { score } else { -score };
        game.add_move(viri_mv, white_score as i16);

        // Win adjudication
        if score.abs() >= config.win_adj_score {
            win_adj_count += 1;
            if win_adj_count >= config.win_adj_plies {
                // score is STM-relative, convert to white-relative
                let stm_wins = score > 0;
                let white_wins = if pos.side_to_move == Color::White { stm_wins } else { !stm_wins };
                result = if white_wins { 2 } else { 0 };
                break;
            }
        } else {
            win_adj_count = 0;
        }

        // Draw adjudication
        if score.abs() <= config.draw_adj_score {
            draw_adj_count += 1;
            if draw_adj_count >= config.draw_adj_plies {
                result = 1;
                break;
            }
        } else {
            draw_adj_count = 0;
        }

        // Mate score: stop immediately
        if score.abs() > SCORE_MATE_IN_MAX {
            let stm_wins = score > 0;
            let white_wins = if pos.side_to_move == Color::White { stm_wins } else { !stm_wins };
            result = if white_wins { 2 } else { 0 };
            break;
        }

        debug_assert!(best_move.from_sq().0 < 64 && best_move.to_sq().0 < 64,
            "datagen: best_move squares OOB {} -> {}", best_move.from_sq().0, best_move.to_sq().0);
        pos.make_move(best_move);
        game_ply += 1;

        // Safety: prevent infinite games
        if game_ply > 600 {
            result = 1;
            break;
        }
    }

    // Set game outcome and return
    game.set_outcome(to_viri_outcome(result));
    Some(game)
}

/// Check if a move is tactical (capture, promotion, or en passant).
fn is_tactical(pos: &Position, m: Move) -> bool {
    let mt = m.move_type();
    mt == MT_PROMOTION || mt == MT_EN_PASSANT || pos.board[m.to_sq().index()] != Piece::NONE
}

// ============================================================
// ETA formatting helpers
// ============================================================

/// Format a duration in seconds as `2j 03h12m` / `3h12m34s` / `12m34s` / `42s`.
fn format_duration(secs: u64) -> String {
    let d = secs / 86400;
    let h = (secs % 86400) / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    if d > 0 {
        format!("{d}j {h:02}h{m:02}m")
    } else if h > 0 {
        format!("{h}h{m:02}m{s:02}s")
    } else if m > 0 {
        format!("{m}m{s:02}s")
    } else {
        format!("{s}s")
    }
}

/// Format the estimated finish time as `16/02 14:35`.
fn format_finish_time(eta_secs: u64) -> String {
    let now = OffsetDateTime::now_local().unwrap_or_else(|_| OffsetDateTime::now_utc());
    let finish = now + time::Duration::seconds(eta_secs as i64);
    format!(
        "{:02}/{:02} {:02}:{:02}",
        finish.day(),
        finish.month() as u8,
        finish.hour(),
        finish.minute(),
    )
}

// ============================================================
// Worker thread
// ============================================================

/// One datagen worker thread.
fn worker(
    thread_id: usize,
    target_positions: u64,
    existing_positions: u64,
    config: &DatagenConfig,
    book: Option<&[String]>,
    output_path: &str,
    pb: &ProgressBar,
    explosion_fens: &Mutex<Vec<String>>,
) {
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(format!("{output_path}.{thread_id}"))
        .expect("failed to open output file");
    let mut writer = BufWriter::with_capacity(1 << 20, file); // 1MB buffer

    let mut rng = Rng::new(thread_id as u64 * 6364136223846793005 + 1442695040888963407);

    // Each side gets its own TT (1MB) and ThreadData
    // Small TT = more diverse positions (limits cached-knowledge reuse)
    let mut shared_white = SharedState::new(1);
    let mut shared_black = SharedState::new(1);
    // id=1 suppresses UCI info output (only id=0 prints)
    // Box<ThreadData> to keep ~177 KB each off the stack (prevents stack overflow
    // when combined with search recursion's ~400 KB of MovePicker frames).
    let mut td_white = Box::new(ThreadData::new(1));
    let mut td_black = Box::new(ThreadData::new(1));

    while pb.position() < target_positions
        && !DATAGEN_STOP.load(Ordering::Relaxed)
    {
        let game = play_game(
            &mut *td_white,
            &mut *td_black,
            &shared_white,
            &shared_black,
            config,
            book,
            &mut rng,
            explosion_fens,
        );

        let Some(game) = game else { continue; };

        game.serialise_into(&mut writer).expect("write failed");

        let n = game.len() as u64;

        pb.inc(n);
        let games = GAMES_PLAYED.fetch_add(1, Ordering::Relaxed) + 1;
        let elapsed = pb.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            let new_pos = pb.position().saturating_sub(existing_positions);
            let pos_per_sec = new_pos as f64 / elapsed;
            let remaining = target_positions.saturating_sub(pb.position());
            let eta_secs = if pos_per_sec > 0.0 { (remaining as f64 / pos_per_sec) as u64 } else { 0 };
            let tb_adj = TB_ADJUDICATIONS.load(Ordering::Relaxed);
            pb.set_message(format!(
                "{games} games, {:.0} pos/s, {tb_adj} TB adj, ETA {} (fin ~{})",
                pos_per_sec,
                format_duration(eta_secs),
                format_finish_time(eta_secs),
            ));
        }

        // Clear TTs and histories every game to prevent pollution
        shared_white.tt.clear();
        shared_black.tt.clear();
        td_white.clear_histories();
        td_black.clear_histories();
    }

    writer.flush().expect("flush failed");
}

// ============================================================
// Entry point
// ============================================================

/// Run datagen: `gaiachess datagen --threads 12 --positions 10000000 --depth 8`
pub fn run(threads: usize, target_positions: u64, depth: i32, output: &str, book_path: Option<&str>) {
    let num_threads = match threads {
        0 => std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
        n => n,
    };

    // Load opening book if provided
    let book: Option<Arc<Vec<String>>> = book_path.map(|path| {
        let file = std::fs::File::open(path)
            .unwrap_or_else(|e| panic!("Failed to open book {path}: {e}"));
        let lines: Vec<String> = std::io::BufReader::new(file)
            .lines()
            .map(|l| l.expect("Failed to read book line"))
            .filter(|l| !l.trim().is_empty())
            .collect();
        assert!(!lines.is_empty(), "Book file {path} is empty");
        eprintln!("  Book:       {path} ({} positions)", lines.len());
        Arc::new(lines)
    });

    let config = DatagenConfig {
        depth,
        soft_nodes: 0, // depth-only: avoids STOP race between datagen threads
        ..DatagenConfig::default()
    };

    GAMES_PLAYED.store(0, Ordering::Relaxed);
    DATAGEN_STOP.store(false, Ordering::Relaxed);
    EXPLOSIONS.store(0, Ordering::Relaxed);
    TB_ADJUDICATIONS.store(0, Ordering::Relaxed);
    let explosion_fens: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

    eprintln!("GaiaChess datagen");
    eprintln!("  Threads:    {num_threads}");
    eprintln!("  Target:     {target_positions} positions");
    eprintln!("  Depth:      {depth}");
    eprintln!("  Nodes:      {} soft", config.soft_nodes);
    eprintln!("  Output:     {output}.*");
    if book.is_none() {
        eprintln!("  Book:       none (random openings)");
    }
    #[cfg(feature = "syzygy")]
    if crate::tb::max_pieces() > 0 {
        eprintln!("  Syzygy:     {}-man", crate::tb::max_pieces());
    }
    eprintln!();

    // Create output directory if needed
    if let Some(parent) = std::path::Path::new(output).parent() {
        if !parent.as_os_str().is_empty() {
            let _ = std::fs::create_dir_all(parent);
        }
    }

    // Check for existing output files and count positions already generated
    let existing_positions: u64 = {
        let bin_path = format!("{output}.vf");
        let mut existing = Vec::new();
        if std::path::Path::new(&bin_path).exists() {
            existing.push(bin_path);
        }
        for i in 0..num_threads {
            let p = format!("{output}.{i}");
            if std::path::Path::new(&p).exists() {
                existing.push(p);
            }
        }
        if !existing.is_empty() {
            let positions: u64 = existing.iter()
                .map(|f| std::fs::metadata(f).map(|m| m.len() / 32).unwrap_or(0))
                .sum();
            eprintln!("WARNING: output files already exist:");
            for f in &existing {
                let size = std::fs::metadata(f).map(|m| m.len()).unwrap_or(0);
                let pos = size / 32;
                eprintln!("  {f} ({:.1} MB, ~{pos} positions)", size as f64 / (1024.0 * 1024.0));
            }
            eprintln!("  Total: {positions} existing positions");
            eprint!("New data will be APPENDED (target: {target_positions}). Continue? [y/N] ");
            let mut answer = String::new();
            std::io::stdin().read_line(&mut answer).expect("failed to read stdin");
            if !answer.trim().eq_ignore_ascii_case("y") {
                eprintln!("Aborted.");
                return;
            }
            eprintln!();
            positions
        } else {
            0
        }
    };

    if existing_positions >= target_positions {
        eprintln!("Already have {existing_positions} positions (target: {target_positions}). Nothing to do.");
        return;
    }
    if existing_positions > 0 {
        eprintln!("  Existing:   {existing_positions} positions");
        eprintln!("  Remaining:  {} positions", target_positions - existing_positions);
        eprintln!();
    }

    // Set up Ctrl+C handler
    let _ = ctrlc_handler();

    // Progress bar (starts from existing positions count)
    let pb = ProgressBar::new(target_positions);
    pb.set_position(existing_positions);
    pb.set_style(
        ProgressStyle::with_template(
            "  {spinner:.green} [{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({msg})",
        )
        .unwrap()
        .progress_chars("##-"),
    );
    pb.set_message("starting...");
    pb.enable_steady_tick(std::time::Duration::from_secs(1));

    let start = Instant::now();

    // Launch worker threads
    std::thread::scope(|s| {
        for thread_id in 0..num_threads {
            let config = &config;
            let pb = &pb;
            let book_ref = book.as_deref().map(|v| v.as_slice());
            let explosion_fens = &explosion_fens;
            std::thread::Builder::new()
                .stack_size(32 * 1024 * 1024) // 32 MB (PGO inlining enlarges frames)
                .spawn_scoped(s, move || {
                    worker(thread_id, target_positions, existing_positions, config, book_ref, output, pb, explosion_fens);
                })
                .expect("failed to spawn datagen worker");
        }
    });

    DATAGEN_STOP.store(true, Ordering::Relaxed);

    let elapsed = start.elapsed().as_secs_f64();
    let total_pos = pb.position();
    let new_pos = total_pos - existing_positions;
    let total_games = GAMES_PLAYED.load(Ordering::Relaxed);

    pb.finish_with_message(format!(
        "{total_games} games, {:.0} pos/s",
        new_pos as f64 / elapsed,
    ));

    eprintln!();
    if existing_positions > 0 {
        eprintln!("Done: {new_pos} new positions ({total_pos} total) from {total_games} games in {elapsed:.1}s");
    } else {
        eprintln!("Done: {total_pos} positions from {total_games} games in {elapsed:.1}s");
    }
    eprintln!("  {:.0} positions/sec", new_pos as f64 / elapsed);
    #[cfg(feature = "syzygy")]
    {
        let tb_adj = TB_ADJUDICATIONS.load(Ordering::Relaxed);
        if tb_adj > 0 {
            let pct = tb_adj as f64 / total_games as f64 * 100.0;
            eprintln!("  TB adj:     {tb_adj} ({pct:.1}% of games)");
        }
    }

    // Merge per-thread files into final output
    if num_threads > 1 {
        merge_files(output, num_threads);
    } else {
        // Rename single thread file
        let src = format!("{output}.0");
        let dst = format!("{output}.vf");
        let _ = std::fs::rename(&src, &dst);
        eprintln!("  Output: {dst}");
    }

    // Explosion report
    let num_explosions = EXPLOSIONS.load(Ordering::Relaxed);
    if num_explosions > 0 {
        eprintln!();
        eprintln!("Explosions: {num_explosions} positions hit {DATAGEN_SEARCH_DEADLINE}ms deadline (skipped)");
        if let Ok(fens) = explosion_fens.lock() {
            for fen in fens.iter() {
                eprintln!("  {fen}");
            }
            if num_explosions as usize > fens.len() {
                eprintln!("  ... and {} more", num_explosions as usize - fens.len());
            }
        }
    }
}

/// Merge per-thread output files into a single .vf file.
fn merge_files(base: &str, num_threads: usize) {
    let dst_path = format!("{base}.vf");
    let mut dst = BufWriter::new(
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&dst_path)
            .expect("merge: open failed"),
    );

    for i in 0..num_threads {
        let src_path = format!("{base}.{i}");
        if let Ok(data) = std::fs::read(&src_path) {
            dst.write_all(&data).expect("merge: write failed");
            let _ = std::fs::remove_file(&src_path);
        }
    }

    dst.flush().expect("merge: flush failed");
    eprintln!("  Output: {dst_path}");
}

/// Set up graceful shutdown on Ctrl+C (SIGINT) and stdin "stop"/"quit".
///
/// First Ctrl+C sets DATAGEN_STOP so worker threads finish their current game,
/// flush buffers, and exit cleanly — no data corruption.
/// Second Ctrl+C forces immediate termination.
fn ctrlc_handler() {
    let _ = ctrlc::set_handler(move || {
        if DATAGEN_STOP.load(Ordering::Relaxed) {
            eprintln!("\nForce quit.");
            std::process::exit(1);
        }
        eprintln!("\nStopping datagen (finishing current games)...");
        DATAGEN_STOP.store(true, Ordering::Relaxed);
        STOP.store(true, Ordering::Relaxed);
    });

    // Stdin listener for "stop"/"quit" (useful when piped)
    std::thread::spawn(|| {
        let mut buf = String::new();
        loop {
            match std::io::stdin().read_line(&mut buf) {
                Ok(0) | Err(_) => break,
                _ => {
                    let cmd = buf.trim();
                    if cmd == "stop" || cmd == "quit" {
                        DATAGEN_STOP.store(true, Ordering::Relaxed);
                        STOP.store(true, Ordering::Relaxed);
                        eprintln!("Stopping datagen...");
                        break;
                    }
                    buf.clear();
                }
            }
        }
    });
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Position;

    #[test]
    fn test_chessboard_startpos_white() {
        let pos = Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            .unwrap();
        let cb = ChessBoard::from_position(&pos, 15, 2); // +15cp, white win

        // White to move: no flipping
        assert_eq!(cb.occ, pos.occupied());
        assert_eq!(cb.score, 15);
        assert_eq!(cb.result, 2);
        assert_eq!(cb.ksq, Square::E1.0);
        assert_eq!(cb.opp_ksq, Square::E8.0 ^ 56);
    }

    #[test]
    fn test_chessboard_black_flips() {
        let pos = Position::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
            .unwrap();
        let cb = ChessBoard::from_position(&pos, 30, 2); // +30 white, white win

        // Black to move: score negated, result inverted
        assert_eq!(cb.score, -30);
        assert_eq!(cb.result, 0); // 2 - 2 = 0 (loss from black's perspective)

        // Board should be flipped: black pieces are now "friendly" on ranks 1-2
        // occ should be the flipped occupancy
        let expected_occ = pos.occupied().swap_bytes();
        assert_eq!(cb.occ, expected_occ);
    }

    #[test]
    fn test_chessboard_size() {
        assert_eq!(std::mem::size_of::<ChessBoard>(), 32);
    }

    #[test]
    fn test_chessboard_piece_count() {
        let pos = Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            .unwrap();
        let cb = ChessBoard::from_position(&pos, 0, 1);

        // Count pieces: 32 pieces in startpos
        let piece_count = cb.occ.count_ones();
        assert_eq!(piece_count, 32);

        // All 16 bytes should be used (32 nibbles = 16 bytes)
        // Each byte should have two valid nibbles
        for i in 0..16 {
            let low = cb.pcs[i] & 0x0F;
            let high = cb.pcs[i] >> 4;
            assert!(low <= 13, "invalid low nibble at byte {i}: {low}");
            assert!(high <= 13, "invalid high nibble at byte {i}: {high}");
        }
    }

    #[test]
    fn test_chessboard_draw_result() {
        let pos = Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            .unwrap();
        let cb = ChessBoard::from_position(&pos, 0, 1); // draw
        assert_eq!(cb.result, 1);

        // From black perspective, draw is still draw
        let pos_b = Position::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
            .unwrap();
        let cb_b = ChessBoard::from_position(&pos_b, 0, 1);
        assert_eq!(cb_b.result, 1); // 2 - 1 = 1
    }

    #[test]
    fn test_to_bytes_roundtrip() {
        let pos = Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            .unwrap();
        let cb = ChessBoard::from_position(&pos, -42, 0);
        let bytes = cb.to_bytes();
        assert_eq!(bytes.len(), 32);

        // Verify score encoding
        let score = i16::from_le_bytes([bytes[24], bytes[25]]);
        assert_eq!(score, -42);

        // Verify result
        assert_eq!(bytes[26], 0);

        // Verify occ
        let occ = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        assert_eq!(occ, cb.occ);
    }
}
