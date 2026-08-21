//! The opening book that ships inside the binary.
//!
//! Four hundred positions of classical theory, twelve plies deep, drawn from the ECO
//! catalogue. It does two things for a weakened engine, and neither of them at full
//! strength. It gives the games variety: the handicap is deterministic by design — what
//! the engine misjudges and what it overlooks both come from the position's own hash
//! (see [`crate::skill`]) — so without a book, every game at a given level would open
//! exactly the same way. And it gives each level as much theory as a player of that
//! strength would have, which is more than their play would otherwise suggest: the
//! opening is the phase weak players get wrong least, because it is the one part of the
//! game they have been shown. The window is part of the rung, from a single move at the
//! bottom of the ladder to the whole book by the middle of it.
//!
//! **At full strength the book is never consulted at all** — not to choose a move, and
//! not to narrow the ones the search looks at. Both were tried and both were dropped:
//! narrowing the root turned out to be free rather than fast (+3 % nodes on average
//! over nine opening positions at a fixed depth, swinging from −49 % to +108 %), and
//! narrowing the replies deeper in the tree actively cost (+62 %), because the replies
//! theory never named are precisely the ones a search refutes in a handful of nodes:
//! removing them takes away cheap cutoffs rather than expensive ones. Neither is worth
//! a functional change to a search that is otherwise SPRT-validated end to end. The
//! reasoning and the numbers are kept in `vault/recherche/Livre d'ouvertures.md`.
//!
//! One consequence worth keeping in mind: at full strength nothing here is reached, so
//! the table below is never even built.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::position::Position;
use crate::types::{Color, Move};

/// The book itself. Its own header records where it comes from and how it is built;
/// `tools/scripts/make_book.py` regenerates it.
const OPENINGS: &str = include_str!("openings.txt");

const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/// UCI `OwnBook`. On by default — a book is what an engine launched by a person
/// expects to have, and the tournament case turns it off explicitly.
static ENABLED: AtomicBool = AtomicBool::new(true);

pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
}

/// The book moves for a position, most played first, or `None` if it is not in the
/// book.
fn moves(key: u64) -> Option<&'static [(Move, u32)]> {
    let book = BOOK.get_or_init(load);
    let found = book.binary_search_by_key(&key, |entry| entry.0).ok()?;
    Some(&book[found].1)
}

/// The move to play here without searching, if the book has one for the level in
/// force.
///
/// `analysing` covers `go infinite` and multi-PV: someone who asked for an analysis
/// wants the search they asked for, not an answer the instant a book is consulted.
pub fn choice(pos: &Position, analysing: bool) -> Option<Move> {
    if !ENABLED.load(Ordering::Relaxed) {
        return None;
    }
    decide(pos, crate::skill::level(), analysing, roll())
}

/// The decision itself, with nothing global left in it: the level, the draw and the
/// position are all handed in, so it can be tested exhaustively.
fn decide(pos: &Position, level: i32, analysing: bool, roll: u64) -> Option<Move> {
    // Checked before the position is: full strength must not so much as load the book,
    // and this is the guard that keeps its behaviour bit for bit what it was without
    // one.
    if level >= crate::skill::FULL_STRENGTH || analysing || ply(pos) >= book_plies(level) {
        return None;
    }
    let book = moves(pos.key)?;
    debug_assert!(!book.is_empty(), "a book position with no moves");
    Some(weighted(book, roll))
}

/// How long the book lasts at a given level, in plies from the start of the game.
///
/// Part of the rung, next to everything else that makes a level the player it is (see
/// [`crate::skill`]). Even a child who has just learned the moves opens 1.e4, so the
/// bottom of the ladder gets a move of theory rather than none; how much theory a player
/// knows then climbs faster than the rest of their game, and by the middle rungs it is
/// the whole book. That matches how people actually improve — the opening is the phase
/// weak players get wrong least often, because it is the one they have memorised.
fn book_plies(level: i32) -> u32 {
    crate::skill::rung_for(level).book_plies
}

/// Plies played since the start of the game.
///
/// Read off the move number rather than counted, so it is right whether the position
/// arrived as `position startpos moves ...` or as a FEN.
fn ply(pos: &Position) -> u32 {
    let moves = pos.fullmove_number.saturating_sub(1) as u32;
    moves * 2 + (pos.side_to_move == Color::Black) as u32
}

/// Draws a move, favouring the popular ones without deferring to them entirely.
///
/// The weights count how many named variations run through a move, which spans three
/// orders of magnitude — drawn straight, the engine would answer 1.e4 or 1.d4 in
/// eight games out of nine. The square root keeps the ordering and compresses the
/// spread, so the main lines stay the likely ones and the flank openings stay
/// possible.
fn weighted(book: &[(Move, u32)], roll: u64) -> Move {
    let share = |weight: u32| weight.isqrt().max(1) as u64;
    let total: u64 = book.iter().map(|&(_, w)| share(w)).sum();
    debug_assert!(total > 0);

    let mut ticket = roll % total;
    for &(mv, weight) in book {
        if ticket < share(weight) {
            return mv;
        }
        ticket -= share(weight);
    }
    // Unreachable: the ticket is drawn below the total of the shares just walked.
    debug_assert!(false, "weighted draw fell off the end of the book");
    book[book.len() - 1].0
}

/// A fresh 64 bits on every call.
///
/// Seeded from the clock, and deliberately so: both of the engine's other seeds are
/// constants (the interface's animations, the handicap's evaluation error), because
/// both want to be reproducible. This one wants the opposite — two launches must not
/// open the same game — so it is the one place where the wall clock gets a say.
fn roll() -> u64 {
    static STATE: OnceLock<AtomicU64> = OnceLock::new();
    let state = STATE.get_or_init(|| {
        AtomicU64::new(crate::time::seed_from_clock())
    });
    // Counter-based: the sequence is the SplitMix64 the Zobrist keys are built with,
    // which needs no lock and no read-modify-write loop to stay sound across threads.
    mix(state.fetch_add(0x9E37_79B9_7F4A_7C15, Ordering::Relaxed))
}

use crate::skill::mix;

/// One position: its Zobrist key, and the moves theory plays there with their weight.
/// The whole book is a list of these, sorted by key.
type Entry = (u64, Vec<(Move, u32)>);

static BOOK: OnceLock<Vec<Entry>> = OnceLock::new();

/// Reads the book once, on the first position that needs it.
///
/// Each row gives the moves played to reach a position, and the position is replayed
/// rather than parsed from a FEN: a position reconstructed from a FEN can carry an
/// en-passant square that the same position reached by playing moves would not, and
/// the two would hash differently. Replaying makes that impossible by construction.
fn load() -> Vec<Entry> {
    let mut book: Vec<Entry> = Vec::new();

    for row in OPENINGS.lines() {
        if row.starts_with('#') || row.is_empty() {
            continue;
        }
        let (path, played) = row.split_once('\t').expect("book row without a tab");

        let mut pos = Position::from_fen(STARTPOS).expect("startpos");
        for step in path.split_whitespace() {
            pos.make_move(legal(&pos, step));
        }

        let entry: Vec<(Move, u32)> = played
            .split_whitespace()
            .map(|item| {
                let (uci, weight) = item.split_once(':').expect("book move without a weight");
                (legal(&pos, uci), weight.parse().expect("unreadable book weight"))
            })
            .collect();
        debug_assert!(!entry.is_empty(), "book row with no moves");
        book.push((pos.key, entry));
    }

    book.sort_unstable_by_key(|entry| entry.0);
    debug_assert!(
        book.windows(2).all(|pair| pair[0].0 != pair[1].0),
        "two book rows reach the same position — merge them in make_book.py"
    );
    book
}

/// Turns one move of the book into a move of this position, and refuses anything
/// else. The book is compiled in and walked end to end by the tests, so a failure
/// here means the asset and the engine disagree, which is worth stopping for.
fn legal(pos: &Position, uci: &str) -> Move {
    crate::uci::parse_uci_move(pos, uci)
        .unwrap_or_else(|| panic!("book move {uci} is not legal in {}", pos.to_fen()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skill::{FULL_STRENGTH, MIN_LEVEL};

    /// The position after a line of moves, played out the way a game would.
    fn after(line: &str) -> Position {
        let mut pos = Position::from_fen(STARTPOS).expect("startpos");
        for step in line.split_whitespace() {
            pos.make_move(legal(&pos, step));
        }
        pos
    }

    fn uci(book: &[(Move, u32)]) -> Vec<String> {
        book.iter().map(|&(mv, _)| mv.to_uci()).collect()
    }

    #[test]
    fn the_whole_book_loads_and_every_move_of_it_is_legal() {
        // `load` refuses any move that is not legal where the book puts it, so
        // reading the file end to end is the check: this walks all four hundred
        // positions, which is what makes regenerating the asset safe.
        let rows = OPENINGS
            .lines()
            .filter(|row| !row.starts_with('#') && !row.is_empty())
            .count();
        let book = BOOK.get_or_init(load);
        assert_eq!(book.len(), rows, "rows and positions must match one for one");
        assert!(rows > 300, "the book looks truncated: {rows} rows");
        assert!(book.iter().all(|(_, moves)| moves.iter().all(|&(mv, w)| mv.is_ok() && w > 0)));
    }

    #[test]
    fn the_first_move_is_a_real_choice_and_the_main_lines_come_first() {
        let start = moves(after("").key).expect("the starting position is in the book");
        assert!(start.len() >= 8, "only {} first moves", start.len());
        let names = uci(start);
        assert_eq!(names[0], "e2e4");
        assert_eq!(names[1], "d2d4");
        assert!(names.contains(&"c2c4".to_string()));

        // Sorted by weight: the draw walks the list in order, and so does a reader.
        assert!(start.windows(2).all(|pair| pair[0].1 >= pair[1].1));

        // Theory, not just first moves.
        let spanish = moves(after("e2e4 e7e5 g1f3 b8c6").key).expect("the Spanish is in the book");
        assert_eq!(uci(spanish)[0], "f1b5");
    }

    #[test]
    fn a_lower_level_leaves_the_book_sooner() {
        let start = after("");
        // Even the weakest level knows how to start a game — and only that.
        assert!(decide(&start, 1, false, 0).is_some());
        assert!(decide(&after("e2e4"), 1, false, 0).is_some());
        assert!(decide(&after("e2e4 e7e5"), 1, false, 0).is_none());

        // Every rung knows at least as much theory as the one below it, and the middle
        // of the ladder knows the whole book.
        let mut previous = 0;
        for level in MIN_LEVEL..FULL_STRENGTH {
            let plies = book_plies(level);
            assert!(plies >= previous, "level {level} forgets theory the rung below knew");
            previous = plies;
        }
        assert!(book_plies(FULL_STRENGTH - 1) >= 12, "the top rungs should know it all");

        // A level with a four-ply window plays two moves and then thinks for itself.
        let level = (MIN_LEVEL..FULL_STRENGTH).find(|&l| book_plies(l) == 4).expect("a 4-ply rung");
        assert!(decide(&after("e2e4 e7e5 g1f3"), level, false, 0).is_some());
        assert!(decide(&after("e2e4 e7e5 g1f3 b8c6"), level, false, 0).is_none());

        // The window is counted in plies, so it opens and closes at the same move
        // for either colour.
        assert_eq!(ply(&start), 0);
        assert_eq!(ply(&after("e2e4")), 1);
        assert_eq!(ply(&after("e2e4 e7e5")), 2);
    }

    #[test]
    fn what_a_weakened_engine_draws_is_always_theory_and_not_always_the_same() {
        let start = after("");
        let book: Vec<String> = uci(moves(start.key).unwrap());
        let mut drawn = std::collections::HashSet::new();
        for roll in 0..500 {
            let mv = decide(&start, 5, false, mix(roll)).expect("level 5 opens out of the book");
            assert!(book.contains(&mv.to_uci()), "{} is not in the book", mv.to_uci());
            drawn.insert(mv.to_uci());
        }
        assert!(drawn.len() >= 4, "only {} different openings in 500 games", drawn.len());
        assert!(drawn.contains("e2e4") && drawn.contains("d2d4"));
    }

    #[test]
    fn full_strength_never_touches_the_book() {
        // The one property the search depends on: at level 20 the answer is no,
        // whatever the position, whatever the draw. Anything else here would be a
        // functional change to a search that is SPRT-validated without a book.
        for line in ["", "e2e4", "e2e4 e7e5", "e2e4 e7e5 g1f3 b8c6"] {
            let pos = after(line);
            assert!(moves(pos.key).is_some(), "{line:?} should be in the book");
            for roll in 0..64 {
                assert!(decide(&pos, FULL_STRENGTH, false, mix(roll)).is_none());
            }
        }

        // Analysis waits for the search it asked for, at every level.
        assert!(decide(&after(""), 5, true, 0).is_none());
    }

    #[test]
    fn a_position_the_book_never_heard_of_says_nothing() {
        let out = Position::from_fen("8/8/4k3/8/8/4K3/4P3/8 w - - 0 1").expect("fen");
        assert!(moves(out.key).is_none());
        assert!(decide(&out, 5, false, 0).is_none());
        assert!(decide(&out, FULL_STRENGTH, false, 0).is_none());
    }

    #[test]
    fn the_draw_is_not_the_clock_twice() {
        // Two calls in a row must not collide: the counter moves whether or not the
        // clock has.
        assert_ne!(roll(), roll());
    }
}
