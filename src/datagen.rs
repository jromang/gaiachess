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
// Datagen configuration
// ============================================================

/// Standard chess starting position.
const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/// Configuration for data generation.
struct DatagenConfig {
    /// Search depth for each move during self-play.
    depth: i32,
    /// Soft node limit (0 = no limit, use depth only).
    soft_nodes: u64,
    /// Percentage of games started from a random DFRC position (0 = standard chess only).
    dfrc_pct: u32,
    /// Number of random opening moves (per side).
    random_moves: usize,
    /// Maximum |eval| to accept opening position.
    max_opening_eval: i32,
    /// Win adjudication: |score| >= this for N consecutive plies.
    win_adj_score: i32,
    /// Win adjudication: consecutive plies.
    win_adj_plies: usize,
    /// Draw adjudication: |score| <= this for N consecutive plies.
    draw_adj_score: i32,
    /// Draw adjudication: consecutive plies.
    draw_adj_plies: usize,
}

impl Default for DatagenConfig {
    fn default() -> Self {
        DatagenConfig {
            depth: 8,
            soft_nodes: 5000,
            dfrc_pct: 0,
            random_moves: 8,
            max_opening_eval: 1000,
            win_adj_score: 2500,
            win_adj_plies: 4,
            draw_adj_score: 4,
            draw_adj_plies: 12,
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
/// How often a non-interactive run states where it is. Long enough to stay quiet in a
/// log kept for days, short enough that a watcher sees movement.
const PROGRESS_LINE_SECONDS: u64 = 60;
/// How long a worker may hold finished games in its buffer. A campaign runs for days
/// and must survive a reboot with at most a minute of each thread's work lost.
const FLUSH_INTERVAL: std::time::Duration = std::time::Duration::from_secs(60);

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

/// Derive a worker's seed from the run seed, so that two runs of the same command
/// with different seeds play different games. Seeding from the thread id alone made
/// every re-run replay the first one, silently duplicating appended data.
fn seed_for_thread(run_seed: u64, thread_id: usize) -> u64 {
    // SplitMix64 finaliser: neighbouring run seeds must not give correlated streams.
    let mut z = run_seed.wrapping_add((thread_id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

// ============================================================
// DFRC starting positions
// ============================================================

/// The ten ways to place two knights on five squares, as (lower, higher) indices.
const KNIGHT_PLACEMENTS: [(usize, usize); 10] = [
    (0, 1), (0, 2), (0, 3), (0, 4), (1, 2),
    (1, 3), (1, 4), (2, 3), (2, 4), (3, 4),
];

/// Put `piece` on the `n`-th still-empty square of a back rank.
fn place_nth_free(rank: &mut [u8; 8], n: usize, piece: u8) {
    let mut seen = 0;
    for sq in rank.iter_mut() {
        if *sq == 0 {
            if seen == n {
                *sq = piece;
                return;
            }
            seen += 1;
        }
    }
    debug_assert!(false, "place_nth_free: no {n}-th free square left");
}

/// Back rank for a Chess960 position number, in Scharnagl's numbering (0..960).
fn scharnagl_back_rank(id: usize) -> [u8; 8] {
    debug_assert!(id < 960, "chess960 id {id} out of range");
    let mut rank = [0u8; 8];
    let mut n = id;

    // The bishops go first, one on a light square and one on a dark one, which is
    // what makes the remaining placements a plain mixed-radix decomposition of `id`.
    rank[2 * (n % 4) + 1] = b'B';
    n /= 4;
    rank[2 * (n % 4)] = b'B';
    n /= 4;

    place_nth_free(&mut rank, n % 6, b'Q');
    n /= 6;

    // Placing the higher-indexed knight first leaves the lower index still valid.
    let (k1, k2) = KNIGHT_PLACEMENTS[n];
    place_nth_free(&mut rank, k2, b'N');
    place_nth_free(&mut rank, k1, b'N');

    // Three squares are left, and the king must stand between the rooks.
    for piece in [b'R', b'K', b'R'] {
        place_nth_free(&mut rank, 0, piece);
    }

    rank
}

/// A random DFRC starting position: each side gets its own back rank, so the two
/// kings need not face each other. Castling rights are written in Shredder notation
/// (the rook's file), the only form that survives asymmetric back ranks.
fn dfrc_start_fen(rng: &mut Rng) -> String {
    let white = scharnagl_back_rank(rng.next_usize(960));
    let black = scharnagl_back_rank(rng.next_usize(960));

    let mut castling = String::with_capacity(4);
    for (rank, first) in [(&white, b'A'), (&black, b'a')] {
        let files: Vec<usize> = (0..8).filter(|&f| rank[f] == b'R').collect();
        debug_assert_eq!(files.len(), 2, "a back rank has exactly two rooks");
        // King-side rook first, matching the K-before-Q order of a classic FEN.
        castling.push((first + files[1] as u8) as char);
        castling.push((first + files[0] as u8) as char);
    }

    let black_rank: String = black.iter().map(|&c| c.to_ascii_lowercase() as char).collect();
    let white_rank: String = white.iter().map(|&c| c as char).collect();
    format!("{black_rank}/pppppppp/8/8/8/8/PPPPPPPP/{white_rank} w {castling} - 0 1")
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
                // Viriformat expects to = rook square, which is now the encoding.
                ViriMove::new_with_flags(from, to, MoveFlags::Castle)
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

/// The two sides each bring their own thread data and shared state, which is what
/// makes the list long; they are borrowed, not owned, so there is nothing to bundle.
#[allow(clippy::too_many_arguments)]
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
    let mut pos = {
        // A run with any DFRC in it writes every game in Shredder notation: viriformat's
        // parser is switched globally, and once switched it rejects "KQkq" outright.
        let shredder = config.dfrc_pct > 0;
        let dfrc = shredder && rng.next_usize(100) < config.dfrc_pct as usize;
        let parsed = if dfrc {
            Position::from_fen_ex(&dfrc_start_fen(rng), true)
        } else if let Some(lines) = book {
            // Pick a random line from the opening book + random moves for diversity
            Position::from_fen_ex(&lines[rng.next_usize(lines.len())], shredder)
        } else {
            Position::from_fen_ex(STARTPOS, shredder)
        };
        let Ok(mut p) = parsed else { return None };
        let num_random = config.random_moves + (rng.next_u64() & 1) as usize;
        if !apply_random_moves(&mut p, num_random, rng) {
            return None;
        }
        p
    };

    // Phase 2: Verify opening with a quick search
    {
        let limits = if config.soft_nodes > 0 {
            let soft = config.soft_nodes * 10; // 10x budget for verification
            SearchLimits::SoftNodes {
                soft,
                hard: soft.saturating_mul(crate::timeman::SOFT_NODES_HARD_FACTOR),
            }
        } else {
            SearchLimits::Depth(config.depth + 2)
        };
        let td = if pos.side_to_move == Color::White { &mut *td_white } else { &mut *td_black };
        let shared = if pos.side_to_move == Color::White { shared_white } else { shared_black };

        STOP.store(false, Ordering::Relaxed);
        prepare_datagen_search(td, &pos, &limits);
        search::search(td, shared);

        // Per-thread deadline hit: skip this game (position is too tactical for PeSTO)
        if td.stopped {
            EXPLOSIONS.fetch_add(1, Ordering::Relaxed);
            if let Ok(mut fens) = explosion_fens.lock()
                && fens.len() < MAX_EXPLOSION_FENS
            {
                fens.push(pos.to_fen());
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
            && let Some(wdl) = crate::tb::probe_wdl(&pos)
        {
            // WDL is STM-relative, convert to white-relative result
            result = match (wdl, pos.side_to_move) {
                (crate::tb::Wdl::Win, Color::White) | (crate::tb::Wdl::Loss, Color::Black) => 2,
                (crate::tb::Wdl::Loss, Color::White) | (crate::tb::Wdl::Win, Color::Black) => 0,
                (crate::tb::Wdl::Draw, _) => 1,
            };
            TB_ADJUDICATIONS.fetch_add(1, Ordering::Relaxed);
            break;
        }

        // Search for best move
        let limits = if config.soft_nodes > 0 {
            SearchLimits::SoftNodes {
                soft: config.soft_nodes,
                hard: config
                    .soft_nodes
                    .saturating_mul(crate::timeman::SOFT_NODES_HARD_FACTOR),
            }
        } else {
            SearchLimits::Depth(config.depth)
        };

        let (td, shared) = if pos.side_to_move == Color::White {
            (&mut *td_white, shared_white)
        } else {
            (&mut *td_black, shared_black)
        };

        STOP.store(false, Ordering::Relaxed);
        prepare_datagen_search(td, &pos, &limits);
        search::search(td, shared);

        // Per-thread deadline hit: skip this game (position too tactical for PeSTO)
        if td.stopped {
            EXPLOSIONS.fetch_add(1, Ordering::Relaxed);
            if let Ok(mut fens) = explosion_fens.lock()
                && fens.len() < MAX_EXPLOSION_FENS
            {
                fens.push(pos.to_fen());
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
///
/// Everything a worker needs is passed in rather than read from a global, so the list
/// is long by design: a worker owns no state of its own.
#[allow(clippy::too_many_arguments)]
fn worker(
    thread_id: usize,
    run_seed: u64,
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
    let mut last_flush = Instant::now();

    let mut rng = Rng::new(seed_for_thread(run_seed, thread_id));

    // Each side gets its own TT (1MB) and ThreadData
    // Small TT = more diverse positions (limits cached-knowledge reuse)
    let mut shared_white = SharedState::new(1, 1);
    let mut shared_black = SharedState::new(1, 1);
    // id=1 suppresses UCI info output (only id=0 prints)
    // Box<ThreadData> to keep ~177 KB each off the stack (prevents stack overflow
    // when combined with search recursion's ~400 KB of MovePicker frames).
    let mut td_white = Box::new(ThreadData::new(1));
    let mut td_black = Box::new(ThreadData::new(1));

    while pb.position() < target_positions
        && !DATAGEN_STOP.load(Ordering::Relaxed)
    {
        let game = play_game(
            &mut td_white,
            &mut td_black,
            &shared_white,
            &shared_black,
            config,
            book,
            &mut rng,
            explosion_fens,
        );

        let Some(game) = game else { continue; };

        game.serialise_into(&mut writer).expect("write failed");

        // The 1 MiB buffer holds thousands of positions, and on a slow configuration it
        // can take hours to fill — hours a power cut would destroy. Flushing on a timer
        // bounds the loss to a minute of work per thread, at one syscall a minute.
        if last_flush.elapsed() >= FLUSH_INTERVAL {
            writer.flush().expect("flush failed");
            last_flush = Instant::now();
        }

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

        // Clear shared state (TT and correction histories) and per-thread histories
        // every game to prevent pollution
        shared_white.clear();
        shared_black.clear();
        td_white.clear_histories();
        td_black.clear_histories();
    }

    writer.flush().expect("flush failed");
}

// ============================================================
// Entry point
// ============================================================

/// Run datagen: `gaiachess datagen --threads 12 --positions 10000000 --depth 8`
///
/// `seed` is the run seed (None = derive one from the clock); `dfrc_pct` is the
/// percentage of games started from a random DFRC position; `assume_yes` skips the
/// confirmation prompt when the output files already exist.
pub fn run(
    threads: usize,
    target_positions: u64,
    depth: i32,
    soft_nodes: u64,
    dfrc_pct: u32,
    seed: Option<u64>,
    assume_yes: bool,
    output: &str,
    book_path: Option<&str>,
) {
    let num_threads = match threads {
        0 => std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
        n => n,
    };

    assert!(dfrc_pct <= 100, "--dfrc takes a percentage, got {dfrc_pct}");

    // Reproducibility: the seed is printed below, and re-running with it replays the run.
    let run_seed = seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0x2545_F491_4F6C_DD1D, |d| d.as_nanos() as u64)
    });

    // Shredder castling is a global switch in viriformat, so it is set for the whole
    // run: with it off the DFRC games would not parse, with it on the standard ones
    // are written as "HAha" instead of "KQkq" — the same rooks either way.
    if dfrc_pct > 0 {
        viriformat::chess::CHESS960.store(true, Ordering::SeqCst);
    }

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

    // A worker hitting its node limit stops itself (datagen_mode in
    // check_limits), so a node budget no longer races the other workers
    // through the global STOP.
    let config = DatagenConfig {
        depth,
        soft_nodes,
        dfrc_pct,
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
    if config.soft_nodes > 0 {
        eprintln!(
            "  Nodes:      {} soft, {} hard",
            config.soft_nodes,
            config.soft_nodes * crate::timeman::SOFT_NODES_HARD_FACTOR
        );
    } else {
        eprintln!("  Depth:      {depth}");
    }
    if dfrc_pct > 0 {
        eprintln!("  DFRC:       {dfrc_pct}% of games (Shredder castling notation)");
    }
    eprintln!("  Seed:       {run_seed}");
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
    if let Some(parent) = std::path::Path::new(output).parent()
        && !parent.as_os_str().is_empty()
    {
        let _ = std::fs::create_dir_all(parent);
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
                .map(|f| count_positions(f))
                .sum();
            eprintln!("WARNING: output files already exist:");
            for f in &existing {
                let size = std::fs::metadata(f).map(|m| m.len()).unwrap_or(0);
                let pos = size / 32;
                eprintln!("  {f} ({:.1} MB, ~{pos} positions)", size as f64 / (1024.0 * 1024.0));
            }
            eprintln!("  Total: {positions} existing positions");
            if assume_yes {
                eprintln!("New data will be APPENDED (target: {target_positions}). Continuing (--yes).");
            } else {
                eprint!("New data will be APPENDED (target: {target_positions}). Continue? [y/N] ");
                let mut answer = String::new();
                std::io::stdin().read_line(&mut answer).expect("failed to read stdin");
                if !answer.trim().eq_ignore_ascii_case("y") {
                    eprintln!("Aborted.");
                    return;
                }
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
    ctrlc_handler();

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
        // indicatif draws nothing when stderr is not a terminal, so a run whose output
        // is redirected — every unattended one — reports no progress at all for days.
        // A plain line now and then is greppable, costs nothing, and keeps the shape
        // `done/total (…)` that readers already parse.
        if !std::io::IsTerminal::is_terminal(&std::io::stderr()) {
            let pb = &pb;
            std::thread::Builder::new()
                .spawn_scoped(s, move || {
                    while !DATAGEN_STOP.load(Ordering::Relaxed)
                        && pb.position() < target_positions
                    {
                        std::thread::sleep(std::time::Duration::from_secs(
                            PROGRESS_LINE_SECONDS,
                        ));
                        let done = pb.position();
                        let elapsed = pb.elapsed().as_secs_f64();
                        let new_pos = done.saturating_sub(existing_positions);
                        let rate = if elapsed > 0.0 { new_pos as f64 / elapsed } else { 0.0 };
                        let remaining = target_positions.saturating_sub(done);
                        let eta = if rate > 0.0 { (remaining as f64 / rate) as u64 } else { 0 };
                        eprintln!(
                            "progress: {done}/{target_positions} ({:.1}%, {rate:.0} pos/s, \
                             ETA {}, {} games)",
                            100.0 * done as f64 / target_positions as f64,
                            format_duration(eta),
                            GAMES_PLAYED.load(Ordering::Relaxed),
                        );
                    }
                })
                .expect("failed to spawn datagen reporter");
        }

        for thread_id in 0..num_threads {
            let config = &config;
            let pb = &pb;
            let book_ref = book.as_deref().map(|v| v.as_slice());
            let explosion_fens = &explosion_fens;
            std::thread::Builder::new()
                .stack_size(32 * 1024 * 1024) // 32 MB (PGO inlining enlarges frames)
                .spawn_scoped(s, move || {
                    worker(thread_id, run_seed, target_positions, existing_positions, config, book_ref, output, pb, explosion_fens);
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

/// Count the positions a viriformat file actually holds.
///
/// Not `len() / 32`, which is what this used to do: 32 is the size of a game's *header*,
/// while a position costs four bytes. The estimate was therefore about seven times too
/// low, and a resumed run believed it had barely started — generating far past its
/// target. Games are self-delimiting, so counting them exactly is a single pass.
fn count_positions(path: &str) -> u64 {
    // The marlinformat header is not re-exported, so its size is pinned here and
    // checked against a real file by the test below.
    const HEADER: usize = 32;
    const MOVE_SIZE: usize = 4;

    let Ok(file) = std::fs::File::open(path) else { return 0 };
    let mut reader = std::io::BufReader::with_capacity(1 << 20, file);
    let mut buffer = Vec::new();
    let mut total = 0u64;
    loop {
        buffer.clear();
        // A truncated tail — a run killed mid-write — simply ends the count.
        if Game::deserialise_fast_into_buffer(&mut reader, &mut buffer).is_err() {
            break;
        }
        // The trailing null terminator is not a position.
        let moves = buffer.len().saturating_sub(HEADER) / MOVE_SIZE;
        total += moves.saturating_sub(1) as u64;
    }
    total
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The header size is pinned by hand because marlinformat does not export it.
    /// A position costs four bytes, not thirty-two: reading the header size as the
    /// position size made the resume counter seven times too low.
    #[test]
    fn a_position_costs_four_bytes_not_thirty_two() {
        let path = "data/mini.vf";
        if !std::path::Path::new(path).exists() {
            return; // the corpus is not in git
        }
        let counted = count_positions(path);
        let bytes = std::fs::metadata(path).unwrap().len();
        assert!(counted > 0, "no game read from {path}");
        assert!(
            counted > bytes / 8 && counted < bytes / 2,
            "{counted} positions for {bytes} bytes is not ~4 bytes each"
        );
    }

    #[test]
    fn scharnagl_518_is_the_standard_back_rank() {
        assert_eq!(&scharnagl_back_rank(518), b"RNBQKBNR");
    }

    /// Every position number must yield a legal Chess960 back rank: the king between
    /// its two rooks, and the bishops on squares of opposite colour.
    #[test]
    fn every_chess960_id_is_a_legal_back_rank() {
        for id in 0..960 {
            let rank = scharnagl_back_rank(id);
            let file_of = |p: u8| rank.iter().position(|&c| c == p).unwrap();

            let mut sorted = rank;
            sorted.sort_unstable();
            assert_eq!(&sorted, b"BBKNNQRR", "id {id} has the wrong pieces");

            let rooks: Vec<usize> = (0..8).filter(|&f| rank[f] == b'R').collect();
            let king = file_of(b'K');
            assert!(rooks[0] < king && king < rooks[1], "id {id}: king outside its rooks");

            let bishops: Vec<usize> = (0..8).filter(|&f| rank[f] == b'B').collect();
            assert_ne!(bishops[0] % 2, bishops[1] % 2, "id {id}: bishops on one colour");
        }
    }

    /// The generated FEN has to survive the engine's own parser, and come back out
    /// unchanged — that round trip is what viriformat will later be handed.
    #[test]
    fn dfrc_start_positions_round_trip_through_fen() {
        let mut rng = Rng::new(0xD1CE);
        for _ in 0..200 {
            let fen = dfrc_start_fen(&mut rng);
            let pos = Position::from_fen_ex(&fen, true)
                .unwrap_or_else(|e| panic!("{fen} rejected: {e:?}"));
            assert_eq!(pos.to_fen(), fen);
            assert_eq!(pos.castling_rights, ALL_CASTLING);
        }
    }

    /// Standard chess is position 518 on both sides, and must come out as the ordinary
    /// starting position written in Shredder notation.
    #[test]
    fn the_standard_position_is_reachable_as_dfrc() {
        let rank = scharnagl_back_rank(518);
        let white: String = rank.iter().map(|&c| c as char).collect();
        assert_eq!(white, "RNBQKBNR");
        let pos = Position::from_fen_ex(STARTPOS, true).unwrap();
        assert_eq!(
            pos.to_fen(),
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w HAha - 0 1"
        );
    }

    /// Two runs differing only by their seed must not replay the same games.
    #[test]
    fn the_run_seed_changes_every_worker_stream() {
        for thread_id in 0..8 {
            assert_ne!(seed_for_thread(1, thread_id), seed_for_thread(2, thread_id));
        }
        // And within a run, workers must not share a stream either.
        let seeds: Vec<u64> = (0..32).map(|t| seed_for_thread(42, t)).collect();
        let mut sorted = seeds.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), seeds.len());
    }
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
