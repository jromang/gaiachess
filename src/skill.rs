//! Deliberately weakened play, for opponents meant to be beatable.
//!
//! Off by default: at full strength every function here is inert, so an ordinary search
//! behaves exactly as it would without this module.
//!
//! Four knobs, because a beginner is weak in four different ways and no one of them
//! alone produces a convincing one.
//!
//! A **depth ceiling** decides how far ahead the opponent looks. An **evaluation
//! fidelity** decides how much of the engine's judgement it is allowed to keep: at the
//! bottom of the ladder it counts material and nothing else, then a piece-square
//! judgement fades in, then the network. A **two-part error** misjudges every position a
//! little and the occasional one enormously, which is how human mistakes are actually
//! distributed — the odd catastrophe, not a constant drizzle. And a **vision**, the
//! chance the opponent so much as considers a move, which is the one that makes it leave
//! pieces hanging: a ceiling on its own gives a short-sighted but otherwise flawless
//! player, and flawless is exactly what a beginner is not. Depth alone cannot reach the
//! bottom of this ladder at all — a network judging a single ply ahead, with every
//! capture still resolved, already plays a respectable club game.
//!
//! Everything is derived from the position's own hash, so a given position is always
//! judged the same way and a given move is always overlooked or seen: the opponent has
//! stable blind spots rather than a tremor. That is both more human and reproducible.
//!
//! All of it is deliberately free of anything the machine controls. A level is a depth, a
//! fidelity, an error and a blind spot — never a time or a node budget — and a
//! handicapped search runs on one thread: the same level must be the same opponent on a
//! laptop and on a server, or "level 3" means nothing to the person who picked it.
//!
//! Which rung is which rating, how the numbers below were measured, and the harness that
//! measures them (`tools/skill/`) are written up in `vault/recherche/Niveaux de jeu.md`.

use crate::types::{MAX_PLY, Move, SCORE_TB_WIN_IN_MAX};
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};

/// Weakest setting: a first opponent for someone who has just learned the moves.
pub const MIN_LEVEL: i32 = 1;
/// Level at which nothing is held back. Twenty rungs, the range every engine's
/// `Skill Level` option uses, so a number carried over from elsewhere means roughly
/// what the person setting it expects.
pub const FULL_STRENGTH: i32 = 20;

/// Vision is counted in 1024ths. At this value every move is considered.
pub const FULL_VISION: i32 = 1024;
/// Evaluation fidelity: material only, material and piece squares, and the network.
/// Values between two of these blend them.
pub const FIDELITY_MATERIAL: i32 = 0;
pub const FIDELITY_PESTO: i32 = 256;
pub const FIDELITY_NNUE: i32 = 512;

/// How much of its vision a player loses per ply of lookahead, in 1024ths.
///
/// Sight of the board fades with distance: a move two plies away is easier to picture
/// than the same move six plies away. This is what keeps weak tactics short rather than
/// merely rare.
const VISION_SLOPE: i32 = 16;
/// Vision never falls below this, or deep nodes would degenerate into a single line.
const VISION_FLOOR: i32 = 160;

/// One rung of the ladder: everything that makes a level the opponent it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rung {
    /// Iterative-deepening ceiling.
    pub depth: i32,
    /// How often, per thousand positions, the level is allowed one ply more than that.
    ///
    /// A ply is a very coarse thing to hand out — at these depths it is worth several
    /// times what any other knob here is — so the ladder would climb in lurches if depth
    /// could only be a whole number. Granting the extra ply on a fraction of positions
    /// puts a rung genuinely between two depths. Which positions is fixed by the position
    /// itself, so it costs nothing in reproducibility, and it is closer to how people
    /// play than a fixed depth is: nobody thinks exactly as hard about every move.
    pub deeper_permille: i32,
    /// Half-width, in centipawns, of the error made on every position.
    pub small_window: i32,
    /// Half-width, in centipawns, of the error made on the occasional position.
    pub blunder_window: i32,
    /// How often, per thousand positions, the wide error is made instead.
    pub blunder_permille: i32,
    /// Chance in 1024ths that a move at the root is considered at all.
    pub vision: i32,
    /// How much of the engine's judgement the level keeps: see the `FIDELITY_*` values.
    pub eval_fidelity: i32,
    /// How long the opening book lasts, in plies from the start of the game.
    pub book_plies: u32,
    /// How many root moves the level looks at before choosing between them, or 0 to
    /// simply play the best one.
    ///
    /// The upper rungs judge too well for the error above to move them off the best move
    /// often, so without this they answer a given position the same way every game — half
    /// of them do, measured. Below them, misjudging and overlooking already produce more
    /// variety than any person would show.
    pub variety_moves: i32,
    /// How much worse than the best a move may be and still get played, in centipawns.
    pub variety_margin: i32,
}

const fn rung(
    depth: i32,
    deeper_permille: i32,
    small_window: i32,
    blunder_window: i32,
    blunder_permille: i32,
    vision: i32,
    eval_fidelity: i32,
    book_plies: u32,
    variety_moves: i32,
    variety_margin: i32,
) -> Rung {
    Rung {
        depth,
        deeper_permille,
        small_window,
        blunder_window,
        blunder_permille,
        vision,
        eval_fidelity,
        book_plies,
        variety_moves,
        variety_margin,
    }
}

/// The ladder, from a child who has just learned the moves to a strong master.
///
/// Roughly a hundred and twenty rating points a rung, which is about the smallest gap
/// two people can reliably tell apart over a handful of games. Which knob carries the
/// level changes as it climbs, because the knobs run out one after another: at the
/// bottom the opponent overlooks moves and cannot tell a rook from a bishop, so vision
/// and fidelity do the work; in the middle it sees the board but misses tactics; at the
/// top it plays properly and only the horizon and the last of the error separate it from
/// full strength. Depth alone could never reach the bottom of this range — a network
/// judging a single ply already plays a decent club game.
///
/// One whole ply of extra lookahead is by a wide margin the coarsest thing on this table:
/// measured rung against rung, at the shallow depths a weak level searches, it is worth
/// three to five times what any other knob here is. That is why depth climbs in fractions
/// rather than steps, and why the whole range fits in under seven plies — seven plies of a
/// network this strong is already grandmaster play, and a first attempt that ran to
/// sixteen simply spent its top third above anything a person will meet.
///
/// The numbers were set by playing the rungs against each other; `tools/skill/` holds the
/// harness, and the measurements are written up in the note named at the top of the file.
#[rustfmt::skip]
const LADDER: [Rung; FULL_STRENGTH as usize] = [
    //   depth deeper  small  blund  b/1000  vision  fidelity          book  var  marg
    rung(    1,     0,   260,   950,    330,    300, FIDELITY_MATERIAL,   2,  0,   0), //  1  ~400
    rung(    1,   150,   236,   870,    255,    400, FIDELITY_MATERIAL,   2,  0,   0), //  2  ~520
    rung(    1,   300,   216,   800,    200,    500,               70,    4,  0,   0), //  3  ~640
    rung(    1,   450,   197,   740,    170,    600,              140,    4,  0,   0), //  4  ~760
    rung(    1,   600,   178,   700,    143,    670,              180,    6,  0,   0), //  5  ~880
    rung(    1,   750,   160,   650,    124,    750,              240,    6,  0,   0), //  6 ~1000
    rung(    1,   900,   143,   600,    107,    830, FIDELITY_PESTO,      8,  0,   0), //  7 ~1120
    rung(    2,     0,   127,   550,     92,    860,              272,    8,  0,   0), //  8 ~1240
    rung(    2,   170,   112,   500,     79,    890,              288,   10,  0,   0), //  9 ~1360
    rung(    2,   340,    98,   460,     68,    920,              304,   10,  0,   0), // 10 ~1480
    rung(    2,   510,    85,   420,     58,    950,              320,   12,  0,   0), // 11 ~1600
    rung(    2,   680,    73,   380,     49,    980,              336,   12,  0,   0), // 12 ~1720
    rung(    2,   850,    62,   340,     41,   1005,              352,   12,  0,   0), // 13 ~1840
    rung(    3,   100,    52,   300,     34, FULL_VISION,         384,   12,  4,  35), // 14 ~1960
    rung(    3,   600,    43,   260,     28, FULL_VISION,         416,   12,  4,  30), // 15 ~2080
    rung(    4,   100,    35,   220,     22, FULL_VISION,         448,   12,  4,  26), // 16 ~2200
    rung(    4,   800,    27,   180,     17, FULL_VISION,         480,   12,  4,  22), // 17 ~2320
    rung(    5,   600,    19,   130,     11, FULL_VISION, FIDELITY_NNUE, 12,  4,  18), // 18 ~2440
    rung(    6,   600,    12,    85,      6, FULL_VISION, FIDELITY_NNUE, 12,  4,  15), // 19 ~2560
    // Full strength. Every field is the value at which its mechanism is inert, and the
    // book is not consulted at all, so nothing here can reach the search.
    rung(MAX_PLY as i32, 0, 0, 0, 0, FULL_VISION, FIDELITY_NNUE, 0, 0, 0),    // 20
];

/// What a rung is worth, in terms a person can read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rating {
    /// Estimated rating, or 0 at full strength, where there is nothing to estimate.
    pub elo: i32,
    /// The kind of player that rung is meant to feel like.
    pub player: &'static str,
}

/// The ladder as a person reads it, kept apart from [`Rung`] on purpose: that table is
/// what the search obeys, this one is what gets shown to whoever is choosing an opponent.
///
/// The middle of the range is measured — rungs 8 to 14 were played against networks
/// trained to imitate human play at a known rating, which run as bots and so carry a
/// rating earned against people. Outside that the line is extrapolated at the spacing
/// measured between neighbouring rungs, because there was nothing weak enough to play
/// against down there and nothing human enough up top. They are estimates and the
/// interface says so.
#[rustfmt::skip]
const RATINGS: [Rating; FULL_STRENGTH as usize] = [
    Rating { elo:  580, player: "just learned the moves" },
    Rating { elo:  700, player: "learning the pieces" },
    Rating { elo:  820, player: "first few games" },
    Rating { elo:  940, player: "beginner" },
    Rating { elo: 1060, player: "improving beginner" },
    Rating { elo: 1180, player: "casual player" },
    Rating { elo: 1300, player: "keen amateur" },
    Rating { elo: 1420, player: "club player" },
    Rating { elo: 1540, player: "solid club player" },
    Rating { elo: 1660, player: "steady club player" },
    Rating { elo: 1750, player: "strong club player" },
    Rating { elo: 1840, player: "tournament player" },
    Rating { elo: 1970, player: "tournament regular" },
    Rating { elo: 2090, player: "candidate master" },
    Rating { elo: 2200, player: "expert" },
    Rating { elo: 2310, player: "national master" },
    Rating { elo: 2420, player: "strong master" },
    Rating { elo: 2530, player: "international master" },
    Rating { elo: 2640, player: "grandmaster" },
    Rating { elo:    0, player: "full strength" },
];

/// What to tell someone about a level before they pick it.
pub fn rating_for(level: i32) -> Rating {
    RATINGS[(level.clamp(MIN_LEVEL, FULL_STRENGTH) - 1) as usize]
}

static LEVEL: AtomicI32 = AtomicI32::new(FULL_STRENGTH);
static SEED: AtomicU64 = AtomicU64::new(0);

/// Sets the playing strength. `seed` decides which way each position is misjudged and
/// which moves are overlooked; keeping it fixed for a whole game keeps the opponent's
/// character consistent.
pub fn set(level: i32, seed: u64) {
    LEVEL.store(level.clamp(MIN_LEVEL, FULL_STRENGTH), Ordering::Relaxed);
    SEED.store(seed, Ordering::Relaxed);
}

pub fn level() -> i32 {
    LEVEL.load(Ordering::Relaxed)
}

/// Deepest a given level may ever search, whether or not it is the one in force.
///
/// The ceiling in any one position is [`ceiling`]; this is the most it can be, which is
/// what a caller budgeting a search ahead of time needs.
pub fn depth_for(level: i32) -> i32 {
    let rung = rung_for(level);
    rung.depth + (rung.deeper_permille > 0) as i32
}

/// How deep the level looks in this particular position.
///
/// A whole extra ply is far too big a step to be the smallest one the ladder can take, so
/// a rung takes it on some positions and not others — which lands it genuinely between
/// two depths. The choice is the position's, not the clock's, so it repeats exactly.
pub fn ceiling(snap: &Snapshot, key: u64) -> i32 {
    let rung = &snap.rung;
    debug_assert!((0..=1000).contains(&rung.deeper_permille));
    if rung.deeper_permille == 0 {
        return rung.depth;
    }
    // Its own corner of the hash: which positions get the extra ply must have nothing to
    // do with which positions get misjudged.
    let hash = mix(key ^ snap.seed ^ 0xA5A5_5A5A_C3C3_3C3C);
    rung.depth + ((hash % 1000) < rung.deeper_permille as u64) as i32
}

/// The whole rung for a level, clamped to the ladder.
pub fn rung_for(level: i32) -> Rung {
    LADDER[(level.clamp(MIN_LEVEL, FULL_STRENGTH) - 1) as usize]
}

/// Everything a search needs to know about the handicap, read once when it starts.
///
/// The search reads this thousands of times a millisecond; taking it as a copy keeps
/// the atomics out of the hot loops and, more importantly, keeps a level from changing
/// underneath a search that is already running.
#[derive(Clone, Copy, Debug)]
pub struct Snapshot {
    /// Whether anything at all is held back. False at full strength, and every
    /// handicap in the search is behind this one test.
    pub active: bool,
    /// Whether moves are overlooked. Separate from `active` because the upper rungs
    /// see the whole board and only misjudge it.
    pub blind: bool,
    pub seed: u64,
    pub rung: Rung,
}

impl Snapshot {
    /// The handicap that holds nothing back, for anything that searches without one.
    pub const FULL_STRENGTH: Snapshot = Snapshot {
        active: false,
        blind: false,
        seed: 0,
        rung: LADDER[(FULL_STRENGTH - 1) as usize],
    };
}

/// Reads the level and seed in force. Called once per search, not per node.
pub fn snapshot() -> Snapshot {
    let level = level();
    let rung = rung_for(level);
    Snapshot {
        active: level < FULL_STRENGTH,
        blind: level < FULL_STRENGTH && rung.vision < FULL_VISION,
        seed: SEED.load(Ordering::Relaxed),
        rung,
    }
}

/// Nudges an evaluation off the truth by an amount fixed by the position.
///
/// Two errors in one: a small one on every position, and a large one on a few. A single
/// window of either size is wrong in a recognisable way — a narrow one makes an opponent
/// that is merely imprecise and never actually loses a piece, a wide one makes one that
/// is uniformly deranged. Together they give a player who is usually a little off and
/// occasionally throws a rook away, which is what playing a beginner feels like.
///
/// Mate and tablebase scores are passed through: a weak opponent should misjudge
/// positions, not fail to notice a mate that is on the board.
#[inline]
pub fn noise(score: i32, key: u64, snap: &Snapshot) -> i32 {
    let rung = &snap.rung;
    if (rung.small_window | rung.blunder_window) == 0 || score.abs() >= SCORE_TB_WIN_IN_MAX {
        return score;
    }
    debug_assert!(rung.small_window >= 0 && rung.blunder_window >= 0);
    debug_assert!((0..=1000).contains(&rung.blunder_permille));

    let hash = mix(key ^ snap.seed);
    let mut error = draw(hash, rung.small_window);
    // A separate field of the same hash decides whether this is one of the bad ones, and
    // a fresh mix sizes it, so that how often the opponent blunders and how badly are
    // independent of each other.
    if (hash >> 32) as u32 % 1000 < rung.blunder_permille as u32 {
        error += draw(mix(hash), rung.blunder_window);
    }
    (score + error).clamp(-SCORE_TB_WIN_IN_MAX + 1, SCORE_TB_WIN_IN_MAX - 1)
}

/// A number in `[-window, window]`, taken from the low bits of a mixed hash.
#[inline]
fn draw(hash: u64, window: i32) -> i32 {
    (hash % (2 * window as u64 + 1)) as i32 - window
}

/// Whether the player notices a move at all.
///
/// A weak player does not evaluate every legal move badly; most of them never cross
/// their mind. Which ones is decided by the position and the move together, so the same
/// oversight repeats whenever the position does — a blind spot, not a flicker. Moves
/// that are hard to miss get two chances to be seen: a capture, a recapture of the piece
/// that just moved, a check, or a move right in front of the player at the root.
#[inline]
pub fn sees_move(snap: &Snapshot, key: u64, mv: Move, ply: usize, easy: bool) -> bool {
    if snap.rung.vision >= FULL_VISION {
        return true;
    }
    let chance = (snap.rung.vision - VISION_SLOPE * ply as i32).max(VISION_FLOOR);
    debug_assert!((0..FULL_VISION).contains(&chance));

    // The move is a small integer and a poor source of entropy on its own, so it is
    // spread across the whole word before it meets the position's key.
    let hash = mix(key ^ (mv.0 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ snap.seed);
    if (((hash >> 20) & 1023) as i32) < chance {
        return true;
    }
    easy && (((hash >> 40) & 1023) as i32) < chance
}

/// Which of the moves worth playing this level actually plays.
///
/// `scores` holds the root moves the search resolved, in centipawns and in no particular
/// order. Anything within the rung's margin of the best is a move a player of that
/// strength could reasonably choose; this picks between them and returns its index.
///
/// The idea is borrowed from how a person plays: below master level nobody finds the
/// single best move every time, and among two moves a fifth of a pawn apart the choice is
/// temperament rather than calculation. It matters only at the top of the ladder — lower
/// down, misjudging and overlooking already vary the play more than any person would.
///
/// A decided position is exempt. Choosing between a mate and a slower mate, or throwing
/// away a won game for variety's sake, is not what a weaker player looks like; it is what
/// a broken engine looks like.
pub fn variety_pick(snap: &Snapshot, key: u64, scores: &[i32]) -> usize {
    let rung = &snap.rung;
    if rung.variety_moves <= 1 || scores.is_empty() {
        return 0;
    }
    debug_assert!(rung.variety_margin >= 0);
    let considered = scores.len().min(rung.variety_moves as usize);
    let best = *scores[..considered].iter().max().expect("at least one score");
    if best.abs() >= SCORE_TB_WIN_IN_MAX {
        return 0;
    }

    let floor = best - rung.variety_margin;
    let acceptable: Vec<usize> = (0..considered)
        .filter(|&i| scores[i] >= floor && scores[i].abs() < SCORE_TB_WIN_IN_MAX)
        .collect();
    debug_assert!(!acceptable.is_empty(), "the best move is always acceptable to itself");

    // Its own corner of the hash again: which move gets played must not be tied to how the
    // position was misjudged.
    acceptable[(mix(key ^ snap.seed ^ 0x1234_5678_9ABC_DEF0) % acceptable.len() as u64) as usize]
}

/// Held by any test that runs a search, or changes the level one would run at.
///
/// The level is one global for the whole process, which is right for an engine — it is
/// told its strength once and keeps it — but it means two tests that set it cannot run at
/// the same time, and the test harness runs them in parallel by default. The flags a
/// running search reads are global for the same reason and want the same treatment: a
/// test that leaves the interface pondering would otherwise take the clock away from a
/// search someone else was timing.
#[cfg(test)]
pub static LEVEL_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Takes the lock above, ignoring the poisoning left behind by an unrelated failure.
#[cfg(test)]
pub fn lock_level() -> std::sync::MutexGuard<'static, ()> {
    LEVEL_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The mixing step the Zobrist keys are built with. Two multiply-shift rounds, enough
/// that neighbouring keys and neighbouring moves land nowhere near each other.
pub const fn mix(seed: u64) -> u64 {
    let mut z = seed;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SCORE_MATE;

    fn snap_at(level: i32, seed: u64) -> Snapshot {
        let rung = rung_for(level);
        Snapshot {
            active: level < FULL_STRENGTH,
            blind: level < FULL_STRENGTH && rung.vision < FULL_VISION,
            seed,
            rung,
        }
    }

    /// Every rung is an improvement on the one below, in every way at once. Without
    /// that, picking a higher level could hand back a weaker opponent.
    #[test]
    fn the_ladder_climbs_in_every_direction() {
        // Depth in thousandths of a ply: a rung that takes the extra ply more often than
        // the one below sits above it, even at the same whole depth.
        let reach = |r: &Rung| r.depth * 1000 + r.deeper_permille;

        for pair in LADDER[..(FULL_STRENGTH - 1) as usize].windows(2) {
            let (low, high) = (pair[0], pair[1]);
            assert!(reach(&high) > reach(&low), "{high:?} sees no further than {low:?}");
            assert!(high.small_window < low.small_window, "{high:?} judges no better than {low:?}");
            assert!(high.blunder_window < low.blunder_window, "{high:?} blunders as wide as {low:?}");
            assert!(high.blunder_permille < low.blunder_permille, "{high:?} blunders as often as {low:?}");
            assert!(high.vision >= low.vision, "{high:?} overlooks more than {low:?}");
            assert!(high.eval_fidelity >= low.eval_fidelity, "{high:?} judges more crudely than {low:?}");
            assert!(high.book_plies >= low.book_plies, "{high:?} knows less theory than {low:?}");
        }

        // The top rung holds nothing back, and every mechanism reads its inert value.
        let top = rung_for(FULL_STRENGTH);
        assert_eq!(top.depth, MAX_PLY as i32);
        assert_eq!(top.deeper_permille, 0, "full strength needs no extra ply");
        assert_eq!((top.small_window, top.blunder_window, top.blunder_permille), (0, 0, 0));
        assert_eq!(top.vision, FULL_VISION);
        assert_eq!(top.eval_fidelity, FIDELITY_NNUE);
        assert_eq!(top.book_plies, 0, "full strength never consults the book");

        // Choosing between root moves is only ever switched on, and once on the margin
        // narrows like everything else. It cannot be monotone field by field — a rung
        // that does not do it at all reads zero — so it gets its own check.
        assert_eq!(top.variety_moves, 0, "full strength plays the move it found");
        let mut previous = i32::MAX;
        let mut switched_on = false;
        for level in MIN_LEVEL..FULL_STRENGTH {
            let rung = rung_for(level);
            if rung.variety_moves == 0 {
                assert!(!switched_on, "level {level} stops choosing after a rung that did");
                assert_eq!(rung.variety_margin, 0, "level {level} has a margin it cannot use");
                continue;
            }
            switched_on = true;
            assert!(rung.variety_moves > 1, "level {level} chooses between one move");
            assert!(rung.variety_margin > 0 && rung.variety_margin < previous,
                "level {level} is no fussier than the rung below");
            previous = rung.variety_margin;
        }
        assert!(switched_on, "no rung chooses between its moves");

        // The bottom rung is a beginner: one ply, material only, half the board unseen.
        let bottom = rung_for(MIN_LEVEL);
        assert_eq!(bottom.depth, 1);
        assert_eq!(bottom.eval_fidelity, FIDELITY_MATERIAL);
        assert!(bottom.vision < FULL_VISION / 2);
    }

    #[test]
    fn full_strength_is_inert() {
        let snap = snap_at(FULL_STRENGTH, 0xdead_beef);
        assert!(!snap.active && !snap.blind);
        assert_eq!(noise(123, 0xdead_beef, &snap), 123);
        assert_eq!(noise(-4321, 0x1234, &snap), -4321);
        assert!(sees_move(&snap, 0xdead_beef, Move(1234), 4, false));
        assert_eq!(depth_for(FULL_STRENGTH), MAX_PLY as i32);
    }

    #[test]
    fn one_position_is_always_misjudged_the_same_way() {
        let snap = snap_at(1, 0x1234);
        let error = noise(0, 0xdead_beef, &snap);
        assert_eq!(error, noise(0, 0xdead_beef, &snap), "one position, one error");
        assert_ne!(
            noise(0, 0xdead_beef, &snap),
            noise(0, 0xfeed_face, &snap),
            "different positions should be misjudged differently"
        );
        // A different seed is a different opponent facing the same position.
        assert_ne!(noise(0, 0xdead_beef, &snap_at(1, 0x9999)), error);
    }

    #[test]
    fn the_error_stays_inside_the_level_and_spares_mates() {
        for level in MIN_LEVEL..=FULL_STRENGTH {
            let snap = snap_at(level, 0x5eed);
            let widest = snap.rung.small_window + snap.rung.blunder_window;
            for key in 0..20_000u64 {
                let error = noise(0, mix(key), &snap);
                assert!(error.abs() <= widest, "level {level}: error {error} beyond {widest}");
            }
            assert_eq!(
                noise(SCORE_MATE - 3, 0xdead_beef, &snap),
                SCORE_MATE - 3,
                "a mate on the board must survive the handicap"
            );
            assert_eq!(noise(-SCORE_MATE + 5, 0xfeed, &snap), -SCORE_MATE + 5);
        }
    }

    /// The point of two windows: most positions are a little wrong, a few are a disaster.
    #[test]
    fn blunders_are_rare_and_large() {
        for level in [1, 5, 10, 15] {
            let snap = snap_at(level, 0x5eed);
            let rung = snap.rung;
            let blunders = (0..20_000u64)
                .filter(|&key| noise(0, mix(key), &snap).abs() > rung.small_window)
                .count();
            let expected = 20_000.0 * rung.blunder_permille as f64 / 1000.0;
            // Wide bounds: a blunder can land inside the small window and go uncounted.
            assert!(
                (blunders as f64) < expected * 1.2 && (blunders as f64) > expected * 0.4,
                "level {level}: {blunders} large errors in 20000, expected about {expected}"
            );
        }
    }

    /// A rung between two depths: it must take the extra ply about as often as it says,
    /// always on the same positions, and never anywhere else.
    #[test]
    fn the_extra_ply_lands_where_the_rung_says_it_should() {
        for level in MIN_LEVEL..=FULL_STRENGTH {
            let snap = snap_at(level, 0x5eed);
            let rung = snap.rung;
            let deeper = (0..10_000u64)
                .filter(|&k| ceiling(&snap, mix(k)) == rung.depth + 1)
                .count();
            let expected = 10_000 * rung.deeper_permille as usize / 1000;
            assert!(
                deeper.abs_diff(expected) < 400,
                "level {level}: {deeper} deep positions in 10000, expected about {expected}"
            );
            // Never anything but those two depths, and always the same answer twice.
            let key = mix(0xdead_beef);
            let one = ceiling(&snap, key);
            assert!(one == rung.depth || one == rung.depth + 1);
            assert_eq!(one, ceiling(&snap, key), "one position, one depth");
            // A rung that says it sometimes looks deeper had better sometimes do it.
            if rung.deeper_permille > 0 {
                assert!(deeper > 0, "level {level} never took its extra ply");
            }
        }
        // What a caller budgets ahead of time has to cover the deepest it can go.
        for level in MIN_LEVEL..=FULL_STRENGTH {
            let snap = snap_at(level, 0);
            for key in 0..500u64 {
                assert!(ceiling(&snap, mix(key)) <= depth_for(level));
            }
        }
    }

    /// Choosing between root moves: only among the good ones, never in a decided
    /// position, and the same choice every time the position comes back.
    #[test]
    fn the_choice_between_root_moves_stays_inside_the_margin() {
        let level = (MIN_LEVEL..FULL_STRENGTH)
            .find(|&l| rung_for(l).variety_moves > 1)
            .expect("some rung chooses");
        let snap = snap_at(level, 0x5eed);
        let margin = snap.rung.variety_margin;

        // A move well outside the margin is never played, one just inside sometimes is.
        let scores = [100, 100 - margin / 2, 100 - margin, 100 - margin - 50];
        let picked: std::collections::HashSet<usize> =
            (0..500u64).map(|k| variety_pick(&snap, mix(k), &scores)).collect();
        assert!(picked.contains(&0), "the best move must still get played");
        assert!(picked.len() > 1, "a level that chooses must sometimes choose differently");
        assert!(!picked.contains(&3), "a move beyond the margin was played");

        // Same position, same choice.
        let key = mix(0xdead_beef);
        assert_eq!(variety_pick(&snap, key, &scores), variety_pick(&snap, key, &scores));

        // Never past the rung's move count, whatever the scores say.
        let flat = vec![50; 32];
        for k in 0..200u64 {
            assert!(variety_pick(&snap, mix(k), &flat) < snap.rung.variety_moves as usize);
        }

        // A won position is played out, not gambled with.
        let decided = [SCORE_MATE - 5, SCORE_MATE - 7, 100, 90];
        for k in 0..200u64 {
            assert_eq!(variety_pick(&snap, mix(k), &decided), 0, "variety in a won position");
        }

        // Rungs that do not choose always answer with the move the search found.
        let plain = snap_at(MIN_LEVEL, 0x5eed);
        assert_eq!(plain.rung.variety_moves, 0);
        for k in 0..100u64 {
            assert_eq!(variety_pick(&plain, mix(k), &scores), 0);
        }
    }

    #[test]
    fn a_move_is_overlooked_the_same_way_every_time() {
        let snap = snap_at(1, 0x1234);
        let seen = sees_move(&snap, 0xdead_beef, Move(1234), 2, false);
        assert_eq!(seen, sees_move(&snap, 0xdead_beef, Move(1234), 2, false));
        // Neighbouring moves must not share a fate: the move is mixed, not xored raw.
        let fates: Vec<bool> =
            (100..164).map(|m| sees_move(&snap, 0xdead_beef, Move(m), 2, false)).collect();
        assert!(fates.iter().any(|&f| f) && fates.iter().any(|&f| !f));
    }

    /// Moves that are hard to miss get two chances, so they are seen more often — that
    /// is what stops a weak level from ignoring a piece it can simply take.
    #[test]
    fn easy_moves_are_seen_more_often() {
        let snap = snap_at(3, 0x5eed);
        let (mut plain, mut easy) = (0, 0);
        for key in 0..4000u64 {
            let k = mix(key);
            plain += sees_move(&snap, k, Move(777), 3, false) as u32;
            easy += sees_move(&snap, k, Move(777), 3, true) as u32;
        }
        assert!(easy > plain, "easy moves ({easy}) should beat plain ones ({plain})");
        // p_easy = 2p - p^2, so with p near a half the gap is wide and worth checking.
        let p = plain as f64 / 4000.0;
        let expected = 4000.0 * (2.0 * p - p * p);
        assert!((easy as f64 - expected).abs() < 200.0, "{easy} against an expected {expected}");
    }

    /// Vision fades with depth but never to nothing.
    #[test]
    fn sight_shortens_with_distance_and_then_holds() {
        let snap = snap_at(1, 0x5eed);
        let seen_at = |ply: usize| {
            (0..4000u64).filter(|&k| sees_move(&snap, mix(k), Move(777), ply, false)).count()
        };
        let (near, far) = (seen_at(0), seen_at(8));
        assert!(near > far, "sight should fade: {near} at the root, {far} eight plies out");
        assert_eq!(seen_at(40), seen_at(60), "below the floor nothing changes further");
    }

    #[test]
    fn levels_outside_the_ladder_are_clamped() {
        let _guard = lock_level();
        set(-5, 0);
        assert_eq!(level(), MIN_LEVEL);
        set(99, 0);
        assert_eq!(level(), FULL_STRENGTH);
        assert_eq!(rung_for(-5), rung_for(MIN_LEVEL));
        assert_eq!(rung_for(99), rung_for(FULL_STRENGTH));
    }

    /// The global level is shared, so the cases that change it live in one test.
    #[test]
    fn the_level_in_force_drives_the_snapshot_and_restores_cleanly() {
        let _guard = lock_level();
        set(FULL_STRENGTH, 0);
        assert_eq!(level(), FULL_STRENGTH, "full strength must be the default");
        assert!(!snapshot().active);
        assert_eq!(depth_for(level()), MAX_PLY as i32);

        set(1, 0x1234);
        let snap = snapshot();
        assert!(snap.active && snap.blind);
        assert_eq!(snap.seed, 0x1234);
        assert_eq!(depth_for(level()), 1);

        // The upper rungs are handicapped but see the whole board.
        set(15, 0);
        let snap = snapshot();
        assert!(snap.active && !snap.blind);

        set(FULL_STRENGTH, 0);
        assert!(!snapshot().active);
        assert_eq!(depth_for(level()), MAX_PLY as i32);
    }
}
