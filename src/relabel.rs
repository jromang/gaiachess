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
//! Output is plain text, one position per line:
//!
//! ```text
//! <FEN> | <score> | <result>
//! ```
//!
//! deliberately, rather than a binary format. `bullet-utils convert --from text` turns
//! it into bulletformat, which the trainer reads natively, and a text intermediate can
//! be read by a human — which matters for a file whose whole purpose is a number per
//! position. It costs about 67 bytes per position instead of 4.5, so a run is sized in
//! the tens of millions of positions, not billions: a fine-tune's worth, which is the
//! scale at which labels have actually been measured to matter here.
//!
//! **Both numbers are white-relative**, and that is the trap this file exists to not
//! fall into again. Viriformat stores white-relative evaluations, bulletformat's text
//! reader expects white-relative, and our search returns a score from the side to
//! move's point of view. Writing the search score unchanged flips the target on half
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
use crate::threads::{SharedState, ThreadData, STOP};
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

/// One position's worth of output, already in the convention the trainer expects.
fn line(fen: &str, white_score: i32, white_result: f32) -> String {
    format!("{fen} | {white_score} | {white_result:.1}\n")
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
        if max_positions > 0 { format!(" · stopping at {max_positions} positions") }
        else { String::new() },
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
            eprintln!("relabel: worker {id} cannot write {}: {error}", part.display());
            return;
        }
    };
    let shared = SharedState::new(1, 1);
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
            // The outcome is the game's, not ours to revise: we are re-scoring
            // positions, not replaying the game.
            let white_result = match game.outcome() {
                viriformat::dataformat::WDL::Win => 1.0,
                viriformat::dataformat::WDL::Draw => 0.5,
                viriformat::dataformat::WDL::Loss => 0.0,
            };

            let mut fens = Vec::with_capacity(game.len());
            game.visit_positions(|board, old_eval| fens.push((board.to_string(), old_eval)));

            for (fen, old_eval) in &fens {
                // Checked per position, not merely per batch: a worker holds sixty-four
                // games, so a batch-level check alone would overshoot a small run by
                // more than the run itself.
                if max_positions > 0 && POSITIONS_DONE.load(Ordering::Relaxed) >= max_positions {
                    let _ = out.flush();
                    return;
                }
                let Ok(pos) = Position::from_fen(fen) else {
                    // A position the engine will not accept is not one to guess at.
                    continue;
                };
                STOP.store(false, Ordering::Relaxed);
                td.prepare_search(&pos, &limits);
                td.tm = crate::timeman::TimeManager::new(&limits);
                td.search_deadline = SEARCH_DEADLINE_MS;
                td.datagen_mode = true;
                search::search(&mut td, &shared);

                let score = td.best_score;
                if td.stopped || score == SCORE_NONE || score.abs() > SCORE_INFINITE {
                    continue;
                }
                let white_score =
                    if pos.side_to_move == Color::White { score } else { -score };
                debug_assert!(white_score.abs() <= SCORE_INFINITE);

                // Compare with the label already in the file, which is white-relative by
                // construction. Only where both are decisive: near zero the sign of an
                // evaluation is noise, and would dilute the very signal being watched.
                if old_eval.abs() >= SIGN_PROBE_THRESHOLD
                    && white_score.abs() >= SIGN_PROBE_THRESHOLD
                {
                    AGREE_N.fetch_add(1, Ordering::Relaxed);
                    if old_eval.signum() == white_score.signum() {
                        AGREE_SAME_SIGN.fetch_add(1, Ordering::Relaxed);
                    }
                    AGREE_ABS_DIFF
                        .fetch_add((white_score - old_eval).unsigned_abs() as u64, Ordering::Relaxed);
                }
                let _ = out.write_all(line(fen, white_score, white_result).as_bytes());
                POSITIONS_DONE.fetch_add(1, Ordering::Relaxed);
            }
            GAMES_DONE.fetch_add(1, Ordering::Relaxed);
        }
    }
    let _ = out.flush();
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

    /// The one convention this file cannot get wrong twice.
    #[test]
    fn a_line_is_white_relative_and_readable() {
        assert_eq!(
            line("8/8/8/8/8/8/8/K6k w - - 0 1", -125, 0.5),
            "8/8/8/8/8/8/8/K6k w - - 0 1 | -125 | 0.5\n"
        );
    }

    /// Black to move, engine says +200 for black: the file must hold −200.
    #[test]
    fn a_score_from_blacks_point_of_view_is_negated() {
        let pos = Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")
            .unwrap();
        let score = 200;
        let white_score = if pos.side_to_move == Color::White { score } else { -score };
        assert_eq!(white_score, -200);
    }
}
