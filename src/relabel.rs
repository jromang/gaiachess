//! Re-score an existing corpus with the current engine.
//!
//! Datagen answers "play games and write down what you saw". This answers a different
//! question: "here are positions somebody else chose — what do *you* think of them?"
//! The positions, the moves and the game outcomes are kept exactly as they were; only
//! the evaluations are replaced.
//!
//! It exists because the largest measured gain of this project came from better labels
//! on positions we already had, not from new positions: 62 GB of Leela data beat a
//! six-billion-position home corpus by +26.7 Elo at identical recipe, and the same data
//! used as a terminal polish gave +25.4. The strongest engines' 2026 gains have the same
//! shape — their whole training corpus re-scored by a stronger network.
//!
//! Output is viriformat, like the input: every game comes out with the same start
//! position, the same moves and the same outcome, and only the evaluation stored with
//! each move changes. The trainer then reads it with `format = viriformat`, through the
//! same filters as any other corpus — which a list of positions could not offer: the
//! move played (that `skip_tactical` needs), the ply and the game boundaries would be
//! lost, and an A/B of new labels against old ones would also be an A/B of filters.
//! A position the engine cannot score (a FEN it refuses, a search that runs out of
//! time) keeps the evaluation it came with, so a game is never shortened.
//!
//! **The evaluations are white-relative**, and that is the trap this file exists to not
//! fall into again. Viriformat stores white-relative evaluations, and our search
//! returns a score from the side to move's point of view. Writing the search score unchanged flips the target on half
//! the positions — invisible in the loss, and it cost this project the whole of gen1v2.
//! `tools/scripts/fix_vf_eval_sign.py --check` is the probe that catches it. This file
//! also carries its own: it compares every score it writes with the one already in the
//! corpus, and says so at the end.
//!
//! **No tablebases**, and that is a limitation rather than a choice. Each worker runs an
//! independent search, so it must stop on its own `stopped` flag instead of the global
//! STOP — otherwise the first worker to spend its budget would halt all the others. The
//! flag that arranges this, `datagen_mode`, also disables tablebase probing, and the two
//! are not separable today. The strongest pipelines re-score with 7-piece Syzygy and get
//! exact endgame labels; ours are pure search. Separating the two flags is the way to fix it.

use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use viriformat::dataformat::Game;

use crate::position::Position;
use crate::search;
use crate::threads::{STOP, SharedState, ThreadData};
use crate::timeman::SearchLimits;
use crate::types::*;

/// Games handed to a worker at a time. Big enough that the lock is not the bottleneck,
/// small enough that a worker never sits on minutes of unshared work at the end.
const GAMES_PER_BATCH: usize = 64;

/// Same ceiling as datagen: iterative deepening always leaves a usable score from the
/// last completed depth, so a position that explodes costs a second, not a run.
const SEARCH_DEADLINE_MS: u64 = 10_000;

static POSITIONS_DONE: AtomicU64 = AtomicU64::new(0);
static GAMES_DONE: AtomicU64 = AtomicU64::new(0);
/// Positions written with their original evaluation, because no search scored them.
static POSITIONS_KEPT: AtomicU64 = AtomicU64::new(0);
static FINISHED: AtomicBool = AtomicBool::new(false);

/// Agreement between the score we write and the one that was already there.
///
/// A sign flip is the failure this whole file is written around: it does not crash, it
/// does not show in the loss, and it destroyed a generation. But it *is* visible here
/// for free — the corpus already holds a white-relative evaluation of every position we
/// re-score, so if our number is the same quantity, the two must correlate strongly.
/// Getting the convention backwards would send the correlation through zero to negative,
/// which no amount of engine disagreement can do.
///
/// Accumulated as fixed-point sums of centipawns, clamped, so no float races.
static AGREE_N: AtomicU64 = AtomicU64::new(0);
static AGREE_SAME_SIGN: AtomicU64 = AtomicU64::new(0);
static AGREE_ABS_DIFF: AtomicU64 = AtomicU64::new(0);

/// Positions where both sides call it decisive enough for the sign to mean something.
const SIGN_PROBE_THRESHOLD: i32 = 50;

struct Reader {
    inner: BufReader<File>,
    exhausted: bool,
}

impl Reader {
    /// The next batch of games, or an empty vector at end of file.
    fn take(&mut self, count: usize) -> Vec<Game> {
        let mut games = Vec::with_capacity(count);
        while games.len() < count && !self.exhausted {
            // A fresh move list per game rather than a recycled one: the buffer's type
            // names a private module of the crate, and a hundred-element allocation is
            // nothing beside the hundred searches this game is about to cost.
            match Game::deserialise_from(&mut self.inner, Vec::new()) {
                Ok(game) => games.push(game),
                // A truncated tail — a run killed mid-write — simply ends the read.
                Err(_) => self.exhausted = true,
            }
        }
        games
    }
}

/// Re-score `input` into `output`, using `nodes` soft nodes per position.
///
/// `max_positions` of 0 means "the whole file". Workers write their own part file and
/// the parts are concatenated at the end, so nothing is held in memory and a crash
/// leaves readable partial output rather than nothing.
pub fn run(input: &Path, output: &Path, nodes: u64, threads: usize, max_positions: u64) {
    let file = match File::open(input) {
        Ok(file) => file,
        Err(error) => {
            eprintln!("relabel: cannot open {}: {error}", input.display());
            std::process::exit(1);
        }
    };
    let threads = if threads == 0 {
        std::thread::available_parallelism().map_or(1, |n| n.get())
    } else {
        threads
    };

    println!(
        "relabel: {} → {} · {} soft nodes · {threads} threads{}",
        input.display(),
        output.display(),
        nodes,
        if max_positions > 0 {
            format!(" · stopping at {max_positions} positions")
        } else {
            String::new()
        },
    );

    let reader = Mutex::new(Reader {
        inner: BufReader::with_capacity(1 << 20, file),
        exhausted: false,
    });
    let started = Instant::now();
    let parts: Vec<std::path::PathBuf> = (0..threads)
        .map(|i| output.with_extension(format!("part{i:02}")))
        .collect();

    std::thread::scope(|scope| {
        let workers: Vec<_> = parts
            .iter()
            .enumerate()
            .map(|(id, part)| {
                let reader = &reader;
                scope.spawn(move || worker(id, reader, part, nodes, max_positions))
            })
            .collect();
        scope.spawn(|| reporter(started, max_positions));
        // Joined here, inside the scope, and only then is the reporter released. The
        // scope itself waits for every thread it spawned, so raising the flag after it
        // returned would be a flag the reporter is waiting for and can never see.
        for worker in workers {
            let _ = worker.join();
        }
        FINISHED.store(true, Ordering::Relaxed);
    });

    concatenate(&parts, output);
    let done = POSITIONS_DONE.load(Ordering::Relaxed);
    let seconds = started.elapsed().as_secs_f64().max(1e-9);
    println!(
        "relabel: {done} positions from {} games in {:.0} s ({:.0} pos/s) → {}",
        GAMES_DONE.load(Ordering::Relaxed),
        started.elapsed().as_secs_f64(),
        done as f64 / seconds,
        output.display(),
    );
    let kept = POSITIONS_KEPT.load(Ordering::Relaxed);
    if kept > 0 {
        println!("relabel: {kept} positions kept their original evaluation (no search score)");
    }
    report_agreement();
}

/// Say out loud whether the new labels are the same quantity as the old ones.
///
/// Around 50 % agreement is what a sign flip looks like — the convention would be
/// wrong on exactly the positions where Black is to move. A healthy re-score of a
/// corpus labelled by a weaker engine sits well above 90 %.
fn report_agreement() {
    let n = AGREE_N.load(Ordering::Relaxed);
    if n == 0 {
        println!("sign probe: no decisive position on both sides, nothing to compare");
        return;
    }
    let same = AGREE_SAME_SIGN.load(Ordering::Relaxed);
    let share = 100.0 * same as f64 / n as f64;
    println!(
        "sign probe: {share:.1} % agreement on {n} decisive positions, \
         mean |new − old| {:.0} cp",
        AGREE_ABS_DIFF.load(Ordering::Relaxed) as f64 / n as f64,
    );
    if share < 80.0 {
        println!(
            "sign probe: WRONG — near 50 % is the signature of a white/stm sign flip, \
             which is invisible in the loss and cost this project a whole generation. \
             Do not train on this file."
        );
    }
}

fn worker(id: usize, reader: &Mutex<Reader>, part: &Path, nodes: u64, max_positions: u64) {
    let mut out = match File::create(part) {
        Ok(file) => BufWriter::with_capacity(1 << 20, file),
        Err(error) => {
            eprintln!(
                "relabel: worker {id} cannot write {}: {error}",
                part.display()
            );
            return;
        }
    };
    let mut shared = SharedState::new(1, 1);
    let mut td = Box::new(ThreadData::new(1));
    let limits = SearchLimits::soft_nodes(
        nodes,
        nodes.saturating_mul(crate::timeman::SOFT_NODES_HARD_FACTOR),
    );

    loop {
        if max_positions > 0 && POSITIONS_DONE.load(Ordering::Relaxed) >= max_positions {
            break;
        }
        let games = match reader.lock() {
            Ok(mut reader) => reader.take(GAMES_PER_BATCH),
            Err(_) => break,
        };
        if games.is_empty() {
            break;
        }

        for game in &games {
            // Checked per game, not merely per batch: a worker holds sixty-four games,
            // so a batch-level check alone would overshoot a small run by more than the
            // run itself. A game is written whole, so a run stops at most one game per
            // worker past its target.
            if max_positions > 0 && POSITIONS_DONE.load(Ordering::Relaxed) >= max_positions {
                let _ = out.flush();
                return;
            }

            // A game's labels must not depend on which games this worker saw before it,
            // and which games a worker sees depends on how the threads race for the
            // batches: the same file relabelled with another thread count would get
            // other labels. Datagen clears the same state at every game for the same
            // reason.
            shared.clear();
            td.clear_histories();

            let (relabelled, kept) = relabel_game(game, &mut td, &shared, &limits);
            debug_assert_eq!(relabelled.len(), game.len());
            if relabelled.serialise_into(&mut out).is_err() {
                eprintln!("relabel: worker {id} cannot write {}", part.display());
                return;
            }
            POSITIONS_DONE.fetch_add(game.len() as u64, Ordering::Relaxed);
            POSITIONS_KEPT.fetch_add(kept, Ordering::Relaxed);
            GAMES_DONE.fetch_add(1, Ordering::Relaxed);
        }
    }
    let _ = out.flush();
}

/// The same game, the same moves and the same outcome, with each position's evaluation
/// replaced by this engine's, white-relative. Returns the game and the number of
/// positions that kept their original evaluation because no search could score them.
///
/// The game is rebuilt rather than reduced to a list of positions so that the trainer
/// reads it through the same loader and the same filters as any other corpus: the move
/// played (which `skip_tactical` needs), the ply and the game boundaries all survive.
fn relabel_game(
    game: &Game,
    td: &mut ThreadData,
    shared: &SharedState,
    limits: &SearchLimits,
) -> (Game, u64) {
    let mut positions = Vec::with_capacity(game.len());
    game.visit_positions(|board, old_eval| positions.push((board.to_string(), old_eval)));
    debug_assert_eq!(positions.len(), game.len());

    // The header — start position, outcome and all — is carried over as it was: the
    // outcome is the game's, not ours to revise.
    let mut out = Game {
        initial_position: game.initial_position,
        moves: Vec::with_capacity(game.len()),
    };
    let mut kept = 0u64;
    for ((fen, old_eval), mv) in positions.iter().zip(game.moves()) {
        let eval = match score_position(fen, td, shared, limits) {
            Some(white_score) => {
                sign_probe(*old_eval, white_score);
                white_score
            }
            None => {
                // A position the engine will not accept, or a search that ran out of
                // time, is not one to guess at: it keeps the label it came with.
                kept += 1;
                *old_eval
            }
        };
        debug_assert!(i16::try_from(eval).is_ok(), "eval {eval} does not fit an i16");
        out.add_move(mv, eval.clamp(i16::MIN as i32, i16::MAX as i32) as i16);
    }
    (out, kept)
}

/// This engine's white-relative score of `fen`, or `None` if it has none to give.
fn score_position(
    fen: &str,
    td: &mut ThreadData,
    shared: &SharedState,
    limits: &SearchLimits,
) -> Option<i32> {
    let pos = Position::from_fen(fen).ok()?;
    STOP.store(false, Ordering::Relaxed);
    td.prepare_search(&pos, limits);
    let game_ply = td.pos.game_ply();
    td.tm = crate::timeman::TimeManager::new(limits, game_ply, &mut td.original_time_adjust);
    td.search_deadline = SEARCH_DEADLINE_MS;
    td.datagen_mode = true;
    // Each position is a new search, aged as one in a game is: the replacement policy
    // weighs an entry's age.
    shared.tt.new_search();
    search::search(td, shared);

    let score = td.best_score;
    if td.stopped || score == SCORE_NONE || score.abs() > SCORE_INFINITE {
        return None;
    }
    let white_score = if pos.side_to_move == Color::White {
        score
    } else {
        -score
    };
    debug_assert!(white_score.abs() <= SCORE_INFINITE);
    Some(white_score)
}

/// Compare a new label with the one already in the file, which is white-relative by
/// construction. Only where both are decisive: near zero the sign of an evaluation is
/// noise, and would dilute the very signal being watched.
fn sign_probe(old_eval: i32, white_score: i32) {
    if old_eval.abs() >= SIGN_PROBE_THRESHOLD && white_score.abs() >= SIGN_PROBE_THRESHOLD {
        AGREE_N.fetch_add(1, Ordering::Relaxed);
        if old_eval.signum() == white_score.signum() {
            AGREE_SAME_SIGN.fetch_add(1, Ordering::Relaxed);
        }
        AGREE_ABS_DIFF.fetch_add(
            (white_score - old_eval).unsigned_abs() as u64,
            Ordering::Relaxed,
        );
    }
}

/// A line a minute, because a run of this shape is measured in hours and its only
/// failure mode that matters — producing nothing — looks exactly like being slow.
fn reporter(started: Instant, max_positions: u64) {
    let mut last = 0u64;
    loop {
        for _ in 0..60 {
            std::thread::sleep(std::time::Duration::from_secs(1));
            if FINISHED.load(Ordering::Relaxed) {
                return;
            }
        }
        let done = POSITIONS_DONE.load(Ordering::Relaxed);
        let rate = done as f64 / started.elapsed().as_secs_f64().max(1e-9);
        if max_positions > 0 {
            let left = max_positions.saturating_sub(done);
            println!(
                "progress: {done}/{max_positions} ({:.1} %, {rate:.0} pos/s, ETA {:.0} min)",
                100.0 * done as f64 / max_positions as f64,
                left as f64 / rate.max(1e-9) / 60.0,
            );
        } else {
            println!("progress: {done} positions ({rate:.0} pos/s)");
        }
        if done == last {
            println!("progress: nothing written in the last minute");
        }
        last = done;
    }
}

fn concatenate(parts: &[std::path::PathBuf], output: &Path) {
    let Ok(file) = File::create(output) else {
        eprintln!("relabel: cannot write {}", output.display());
        return;
    };
    let mut out = BufWriter::with_capacity(1 << 20, file);
    for part in parts {
        if let Ok(mut input) = File::open(part) {
            let _ = std::io::copy(&mut input, &mut out);
        }
    }
    let _ = out.flush();
    for part in parts {
        let _ = std::fs::remove_file(part);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A relabelled game is the same game: start position, moves and outcome as they
    /// were, through a write and a read. Only the evaluations change, and they are
    /// white-relative — the one convention this file cannot get wrong twice. White is a
    /// queen up with Black to move, so every label, whoever is to move, is positive.
    #[test]
    fn a_relabelled_game_keeps_its_moves_and_outcome_with_white_relative_labels() {
        use viriformat::chess::board::{Board, GameOutcome, WinType};

        const OLD_EVAL: i16 = -999;
        let mut board = Board::new();
        board
            .set_from_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")
            .unwrap();
        let mut game = Game::new(&board);
        for _ in 0..8 {
            let mv = board.legal_moves()[0];
            game.add_move(mv, OLD_EVAL);
            assert!(board.make_move_simple(mv));
        }
        game.set_outcome(GameOutcome::WhiteWin(WinType::Adjudication));

        let shared = SharedState::new(1, 1);
        let mut td = Box::new(ThreadData::new(1));
        let limits = SearchLimits::soft_nodes(3_000, 300_000);
        let (relabelled, kept) = relabel_game(&game, &mut td, &shared, &limits);

        assert_eq!(kept, 0, "every position was scored");
        assert_eq!(relabelled.initial_position, game.initial_position);
        assert!(relabelled.moves().eq(game.moves()), "the same moves");
        assert_eq!(relabelled.outcome(), game.outcome());
        let mut labels = Vec::new();
        relabelled.visit_positions(|_, eval| labels.push(eval));
        assert_eq!(labels.len(), game.len());
        for (ply, &eval) in labels.iter().enumerate() {
            assert!(eval > 0, "ply {ply}: white is a queen up, label {eval}");
        }

        let mut bytes = Vec::new();
        relabelled.serialise_into(&mut bytes).unwrap();
        let read = Game::deserialise_from(&mut bytes.as_slice(), Vec::new()).unwrap();
        assert_eq!(read, relabelled, "the output reads back as viriformat");
    }

    /// Black to move, engine says +200 for black: the file must hold −200.
    #[test]
    fn a_score_from_blacks_point_of_view_is_negated() {
        let pos =
            Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1").unwrap();
        let score = 200;
        let white_score = if pos.side_to_move == Color::White {
            score
        } else {
            -score
        };
        assert_eq!(white_score, -200);
    }
}
