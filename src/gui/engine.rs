//! The engine, kept off the drawing thread.
//!
//! The interface must keep drawing at 60 Hz while a search runs, so the search never
//! happens where the frames are produced. On the desktop that means a thread with the
//! deep stack it needs, talking over channels. In a browser it means a Web Worker, since
//! `std::thread::spawn` does not work on `wasm32-unknown-unknown` at all — there the two
//! sides talk in UCI text through the host.
//!
//! Both give the same calls, and both obey the same rule: every request carries a
//! generation number, and a result whose generation no longer matches has been overtaken
//! by a take-back, a restart or a flag, and is dropped rather than played.

use crate::skill;
use crate::timeman::SearchLimits;
use crate::types::Move;

/// Transposition table size for a game, in megabytes.
const TT_MB: usize = 64;

/// What a finished search hands back.
///
/// Not the score: nothing on the screen shows one, and a figure carried across the
/// bridge that nobody reads is a figure nobody keeps true.
pub struct Thought {
    pub best: Move,
    /// The reply the engine expects: the second move of its own principal variation.
    /// What it goes on turning over while the player thinks about theirs.
    pub ponder: Option<Move>,
}

/// How long a stretch the node rate is read over.
///
/// Long enough that the figure sits still rather than flickering a different number
/// every frame, short enough that it answers a search starting or stopping while the
/// player is still looking at it.
#[cfg(feature = "gui")]
const RATE_WINDOW_MS: u64 = 500;

/// How fast the engine is searching, read off the wall clock.
///
/// The node counter only ever climbs, so what is shown is how far it climbed over the
/// last window rather than a total divided by its age: a figure taken the second way
/// settles on the average of the whole search and stops moving. A window that ends
/// with nothing in it shows nothing at all, which is what an idle engine reads as.
#[cfg(feature = "gui")]
struct Rate {
    opened: crate::time::Instant,
    nodes_at_open: u64,
    shown: Option<u32>,
}

#[cfg(feature = "gui")]
impl Rate {
    fn new() -> Rate {
        use std::sync::atomic::Ordering;

        Rate {
            opened: crate::time::Instant::now(),
            // Where the counter stands now, not zero: it is process-wide and outlives
            // any one game, so a meter opening on nothing would report every node of the
            // game before as if it had been searched in the next half second.
            nodes_at_open: crate::threads::NODES_SEARCHED.load(Ordering::Relaxed),
            shown: None,
        }
    }

    /// Closes the window if it has run its length, and opens the next one.
    fn sample(&mut self) {
        use std::sync::atomic::Ordering;

        let now = crate::time::Instant::now();
        let span = now.duration_since(self.opened).as_millis() as u64;
        if span < RATE_WINDOW_MS {
            return;
        }
        let nodes = crate::threads::NODES_SEARCHED.load(Ordering::Relaxed);
        // Saturating because the bench monitor zeroes the same counter. The two never
        // run at once, but a rate that reads as nothing beats one that reads as a
        // billion.
        let searched = nodes.saturating_sub(self.nodes_at_open);
        self.shown = (searched > 0).then(|| (searched * 1000 / span).min(u32::MAX as u64) as u32);
        self.opened = now;
        self.nodes_at_open = nodes;
    }
}

// ============================================================
// Desktop: a thread of its own
// ============================================================

#[cfg(feature = "gui")]
mod desktop {
    use std::sync::atomic::Ordering;
    use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};
    use std::thread::JoinHandle;

    use super::{Rate, TT_MB, Thought};
    use crate::book;
    use crate::position::Position;
    use crate::search;
    use crate::skill;
    use crate::threads::{COUNT_NODES, PONDER, STOP, SharedState, ThreadData};
    use crate::timeman::SearchLimits;
    use crate::types::Move;

    /// Stack size for the search thread. Deep recursion with inlined frames needs it.
    const SEARCH_STACK: usize = 32 * 1024 * 1024;

    enum Command {
        /// Forget everything learned about the previous game.
        NewGame,
        Search {
            pos: Box<Position>,
            limits: SearchLimits,
            /// Playing strength for this search, 1 to `MAX_LEVEL`.
            level: i32,
            /// Seed deciding which way each position is misjudged at weak levels.
            seed: u64,
            /// Whether this search runs on the player's time. It then ignores its budget
            /// until the wait ends, and the budget starts counting from that moment.
            ponder: bool,
            generation: u64,
        },
        Quit,
    }

    struct Answer {
        best: Move,
        ponder: Option<Move>,
        generation: u64,
    }

    /// What the engine thread has been set to do.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum Busy {
        /// Nothing is running, or its answer has already been collected.
        Idle,
        /// Working on the position on the board, on the engine's own time.
        Searching,
        /// Working on the position it hopes the player is about to produce, on theirs.
        Pondering,
    }

    pub struct Engine {
        tx: Sender<Command>,
        rx: Receiver<Answer>,
        worker: Option<JoinHandle<()>>,
        generation: u64,
        busy: Busy,
        /// The position being pondered, so a hit is told from a miss by the board itself
        /// rather than by the interface remembering which move it bet on.
        pondered: u64,
        rate: Rate,
    }

    impl Engine {
        pub fn spawn() -> Engine {
            let (tx, commands) = channel::<Command>();
            let (answers, rx) = channel::<Answer>();
            let worker = std::thread::Builder::new()
                .stack_size(SEARCH_STACK)
                .spawn(move || run_worker(&commands, &answers))
                .expect("failed to spawn the engine thread");
            // Somebody is watching the rate for as long as there is a board on the
            // screen, so the counter is turned on once here rather than around every
            // search: it costs one atomic every 512 nodes and nothing else.
            COUNT_NODES.store(true, Ordering::Relaxed);
            Engine {
                tx,
                rx,
                worker: Some(worker),
                generation: 0,
                busy: Busy::Idle,
                pondered: 0,
                rate: Rate::new(),
            }
        }

        /// Clears the engine's memory of the previous game.
        pub fn new_game(&mut self) {
            self.abort();
            let _ = self.tx.send(Command::NewGame);
        }

        /// Asks for a move. Any search already running is abandoned.
        ///
        /// The move list is not read here: the position carries its own history, which is
        /// everything the repetition test needs. The browser side has no such luxury.
        pub fn think(
            &mut self,
            pos: &Position,
            _moves: &[Move],
            limits: SearchLimits,
            level: u8,
            seed: u64,
        ) {
            self.start(pos, limits, level, seed, false);
        }

        /// Sets the engine thinking about `pos` — the position it expects to be asked
        /// about next — while the player thinks about the one on the board.
        ///
        /// The budget quoted is the one the engine will have when its turn comes; none of
        /// it is spent until [`Engine::ponderhit`] says the wait is over, so a player who
        /// takes an hour costs it nothing.
        pub fn ponder(
            &mut self,
            pos: &Position,
            _moves: &[Move],
            limits: SearchLimits,
            level: u8,
            seed: u64,
        ) {
            self.start(pos, limits, level, seed, true);
            self.pondered = pos.key;
        }

        fn start(
            &mut self,
            pos: &Position,
            limits: SearchLimits,
            level: u8,
            seed: u64,
            ponder: bool,
        ) {
            self.abort();
            self.generation += 1;
            self.busy = if ponder { Busy::Pondering } else { Busy::Searching };
            // Raised here, and only here, rather than on the thread that runs the
            // search. `prepare_search` reads it to decide whether this search answers to
            // its clock at all, and sending the command is what orders the two: a flag
            // the worker set for itself could still be landing after a `ponderhit` that
            // came in quickly, and the search would then run with no clock and never
            // hand a move back.
            PONDER.store(ponder, Ordering::Relaxed);
            let _ = self.tx.send(Command::Search {
                pos: Box::new(pos.clone()),
                limits,
                level: level as i32,
                seed,
                ponder,
                generation: self.generation,
            });
        }

        /// Turns a search running on the player's time into one running on the engine's,
        /// when what it is turning over is the position now on the board.
        ///
        /// Returns false when the player played something else — or when nothing was
        /// being pondered at all — and the caller must ask for a search of its own.
        pub fn ponderhit(&mut self, pos: &Position) -> bool {
            if self.busy != Busy::Pondering || self.pondered != pos.key {
                return false;
            }
            self.busy = Busy::Searching;
            // Read by the search on its own tick, which restarts its clock: the budget
            // begins now, not when the guess was made.
            PONDER.store(false, Ordering::Relaxed);
            true
        }

        /// True while the engine is thinking on the player's time. Nothing on the
        /// screen turns on this — the rate in the band is the visible sign — so it
        /// exists for the tests that check the interface set it going at all.
        #[cfg(test)]
        pub fn is_pondering(&self) -> bool {
            self.busy == Busy::Pondering
        }

        /// Gives up whatever is being pondered, and leaves a real search alone.
        pub fn stop_pondering(&mut self) {
            if self.busy == Busy::Pondering {
                self.abort();
            }
        }

        /// Stops the current search and disowns its result.
        pub fn abort(&mut self) {
            if self.busy != Busy::Idle {
                STOP.store(true, Ordering::Relaxed);
                // Lowered so nothing stands between two searches: every request raises
                // it for itself, and one that did not must not inherit it.
                PONDER.store(false, Ordering::Relaxed);
                self.generation += 1;
                self.busy = Busy::Idle;
            }
        }

        /// Collects a move if one is ready. Never blocks, and never returns the result of
        /// a search that has since been overtaken.
        ///
        /// A ponder that finishes before the player moves — a forced line, or a position
        /// searched to the end — is left in the channel rather than handed back: it
        /// answers a question nobody has asked yet. [`Engine::ponderhit`] is what turns it
        /// into one, and the next call collects it.
        pub fn poll(&mut self) -> Option<Thought> {
            if self.busy == Busy::Pondering {
                return None;
            }
            loop {
                match self.rx.try_recv() {
                    Ok(answer) if answer.generation == self.generation => {
                        self.busy = Busy::Idle;
                        return Some(Thought {
                            best: answer.best,
                            ponder: answer.ponder,
                        });
                    }
                    Ok(_) => continue,
                    Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => return None,
                }
            }
        }

        /// Re-reads how fast the search is going. Called once a step; the figure itself
        /// only moves when a window closes.
        pub fn tick(&mut self) {
            self.rate.sample();
        }

        /// Nodes a second, as of the last window that had any nodes in it.
        pub fn nps(&self) -> Option<u32> {
            self.rate.shown
        }
    }

    impl Drop for Engine {
        fn drop(&mut self) {
            self.abort();
            let _ = self.tx.send(Command::Quit);
            if let Some(worker) = self.worker.take() {
                // The search checks for a stop every few hundred nodes, so this returns
                // promptly even from the middle of a deep search.
                let _ = worker.join();
            }
            COUNT_NODES.store(false, Ordering::Relaxed);
        }
    }

    fn run_worker(commands: &Receiver<Command>, answers: &Sender<Answer>) {
        let mut shared = SharedState::new(TT_MB);
        // Thread 0, muted. The interface wants everything thread 0 does — the real time
        // budget, the root reporting, the tablebase lookups — and none of what it says.
        // Taking a helper's id instead would buy the silence at the price of the rest.
        let mut td = ThreadData::new(0);
        td.silent = true;

        while let Ok(command) = commands.recv() {
            match command {
                Command::Quit => return,
                Command::NewGame => {
                    shared = SharedState::new(TT_MB);
                    td = ThreadData::new(0);
                    td.silent = true;
                }
                Command::Search {
                    pos,
                    limits,
                    level,
                    seed,
                    ponder,
                    generation,
                } => {
                    skill::set(level, seed);
                    STOP.store(false, Ordering::Relaxed);
                    // Ages the table, so what was learned about earlier moves of the game
                    // yields to what is being learned now. A search entered through the
                    // pool gets this for free; this one has to ask.
                    shared.tt.new_search();
                    // Installs the real budget and folds the level's depth ceiling in.
                    td.prepare_search(&pos, &limits);
                    // Below full strength the book answers before the search does, which
                    // is what keeps a level from opening the same way every game. At full
                    // strength it never answers, pondering or not.
                    let booked = book::choice(&pos, ponder);
                    if booked.is_none() {
                        search::search(&mut td, &shared);
                    }
                    skill::set(skill::FULL_STRENGTH, 0);
                    let answer = Answer {
                        best: booked.unwrap_or(td.best_move),
                        // A book move has no expected reply: it was recited, not judged,
                        // and there is no line behind it.
                        ponder: booked.is_none().then(|| expected_reply(&td)).flatten(),
                        generation,
                    };
                    if answers.send(answer).is_err() {
                        return;
                    }
                }
            }
        }
    }

    /// The reply the search expects, the second move of its own principal variation.
    fn expected_reply(td: &ThreadData) -> Option<Move> {
        let best = td.root_moves.first()?;
        if best.pv_len < 2 {
            return None;
        }
        debug_assert_eq!(best.pv[0], td.best_move, "the PV does not start with the move played");
        let reply = best.pv[1];
        reply.is_ok().then_some(reply)
    }
}

#[cfg(feature = "gui")]
pub use desktop::Engine;

// ============================================================
// Browser: a Web Worker, spoken to in UCI
// ============================================================

#[cfg(all(feature = "gui-core", not(feature = "gui")))]
mod web {
    use super::{TT_MB, Thought};
    use crate::position::Position;
    use crate::timeman::SearchLimits;
    use crate::types::Move;

    #[link(wasm_import_module = "env")]
    unsafe extern "C" {
        /// Passes one UCI line to the worker.
        fn gaia_engine_send(ptr: *const u8, len: usize);
        /// Passes a `go` line, and the generation its answer must come back stamped
        /// with. The host keeps the stamps in a queue: aborting and asking again sends
        /// two searches with no `bestmove` in between, so two answers will come back, and
        /// a single slot would give them both the same number.
        fn gaia_engine_go(ptr: *const u8, len: usize, generation: u32);
        /// Takes the next answer, if one is waiting. Writes `"<generation> <move>"` into
        /// `buf` and returns its length, or -1 when there is nothing.
        fn gaia_engine_poll(buf: *mut u8, cap: usize) -> i32;
        /// Asks the host to cut the running search short. Best effort: it works where
        /// the page and the worker share memory, and is simply ignored where they do
        /// not, leaving the answer to be dropped on arrival as before.
        fn gaia_engine_abort();
    }

    fn send(line: &str) {
        unsafe { gaia_engine_send(line.as_ptr(), line.len()) };
    }

    pub struct Engine {
        generation: u64,
        searching: bool,
        /// The position the running search was asked about, kept so the answer can be
        /// checked against it rather than trusted.
        asked_about: Option<Box<Position>>,
    }

    impl Engine {
        pub fn spawn() -> Engine {
            send("uci");
            send(&format!("setoption name Hash value {TT_MB}"));
            // Thinking on the player's time needs `go ponder` and `ponderhit` to reach
            // the worker while it is busy, which is exactly what a message cannot do
            // here — see [`Engine::abort`]. The desktop side, which shares memory with
            // its search thread, does ponder.
            send("setoption name Ponder value false");
            Engine {
                generation: 0,
                searching: false,
                asked_about: None,
            }
        }

        pub fn new_game(&mut self) {
            self.abort();
            send("ucinewgame");
        }

        /// Asks for a move. Any search already running is abandoned.
        ///
        /// The position travels as the moves that made it, not as a FEN: a FEN would
        /// arrive without the history, and the engine would stop seeing repetitions. The
        /// interface only ever starts a game from the initial position, which is what
        /// makes `startpos` enough.
        pub fn think(
            &mut self,
            pos: &Position,
            moves: &[Move],
            limits: SearchLimits,
            level: u8,
            seed: u64,
        ) {
            self.abort();
            self.generation += 1;
            self.searching = true;
            self.asked_about = Some(Box::new(pos.clone()));

            send(&format!("setoption name Skill Level value {level}"));
            // The option is a spin bounded to i32, and the engine spreads whatever it is
            // handed; the low half of the seed keeps every game a different opponent
            // without needing a wider option.
            send(&format!(
                "setoption name Skill Seed value {}",
                (seed & 0x7FFF_FFFF).max(1)
            ));

            let mut position = String::from("position startpos");
            if !moves.is_empty() {
                position.push_str(" moves");
                for mv in moves {
                    position.push(' ');
                    position.push_str(&mv.to_uci());
                }
            }
            send(&position);

            let go = go_line(limits);
            unsafe { gaia_engine_go(go.as_ptr(), go.len(), self.generation as u32) };
        }

        /// Disowns the running search, and asks for it to stop if that is possible.
        ///
        /// A message cannot do it: a worker busy in a search will not read one until it
        /// comes back out. Only shared memory crosses, so [`gaia_engine_abort`] raises a
        /// flag the search reads on its own tick — where the page is cross-origin
        /// isolated. Where it is not, the answer is simply dropped on arrival, which
        /// costs nothing below full strength: every rung there is bounded by depth and
        /// comes back in milliseconds.
        pub fn abort(&mut self) {
            if self.searching {
                unsafe { gaia_engine_abort() };
                self.generation += 1;
                self.searching = false;
            }
        }

        /// Collects a move if one is ready. Never blocks, and never returns the result of
        /// a search that has since been overtaken.
        pub fn poll(&mut self) -> Option<Thought> {
            loop {
                let mut buf = [0u8; 64];
                let len = unsafe { gaia_engine_poll(buf.as_mut_ptr(), buf.len()) };
                if len < 0 {
                    return None;
                }
                let reply = core::str::from_utf8(&buf[..len as usize]).unwrap_or("");
                let Some((stamp, uci)) = reply.split_once(' ') else {
                    continue;
                };
                if stamp.parse::<u64>() != Ok(self.generation) {
                    // Overtaken: keep draining, the answer being waited on may be behind.
                    continue;
                }
                // Checked against the interface's own position rather than trusted: a
                // host that has fallen out of step is caught here, instead of an
                // illegal move reaching the board.
                let Some(pos) = self.asked_about.as_deref() else { continue };
                let Some(mv) = crate::uci::parse_uci_move(pos, uci) else {
                    debug_assert!(false, "worker answered with an illegal move: {uci}");
                    continue;
                };
                self.searching = false;
                // No expected reply: finding one would mean parsing every `info` line
                // for its principal variation, and this side does not ponder anyway.
                return Some(Thought { best: mv, ponder: None });
            }
        }

        /// Nothing to set going: this side does not ponder, so there is nothing to hit
        /// either and every turn begins with a search of its own.
        pub fn ponder(
            &mut self,
            _pos: &Position,
            _moves: &[Move],
            _limits: SearchLimits,
            _level: u8,
            _seed: u64,
        ) {
        }

        pub fn ponderhit(&mut self, _pos: &Position) -> bool {
            false
        }

        #[cfg(test)]
        pub fn is_pondering(&self) -> bool {
            false
        }

        pub fn stop_pondering(&mut self) {}

        pub fn tick(&mut self) {}

        /// No rate to give: the node counter lives inside the worker, in an instance of
        /// its own that this side of the bridge cannot read.
        pub fn nps(&self) -> Option<u32> {
            None
        }
    }

    fn go_line(limits: SearchLimits) -> String {
        match limits {
            SearchLimits::Depth(d) => format!("go depth {d}"),
            SearchLimits::Nodes(n) => format!("go nodes {n}"),
            SearchLimits::MoveTime(ms) => format!("go movetime {ms}"),
            // Both sides are given the same numbers. The engine only ever reads the clock
            // of the side to move, so one pair would do; naming both keeps the line valid
            // whoever is on the move.
            SearchLimits::Clock { time, inc, .. } => {
                format!("go wtime {time} winc {inc} btime {time} binc {inc}")
            }
            SearchLimits::Infinite => String::from("go infinite"),
        }
    }
}

#[cfg(all(feature = "gui-core", not(feature = "gui")))]
pub use web::Engine;

pub const MAX_LEVEL: u8 = skill::FULL_STRENGTH as u8;

/// The level the title screen opens on: a casual player, around 1180 Elo.
///
/// Full strength is nobody's opponent. Someone who starts a game without touching the
/// ladder wants a game, not a demonstration; the rung is theirs to move either way.
pub const DEFAULT_LEVEL: u8 = 6;

/// The search budget for a level.
///
/// Below full strength this is a depth, never a time or a node count: depth is the
/// only budget that does not move with the hardware, and a level has to be the same
/// opponent on every machine. Only at full strength, where nothing is being held
/// back, does the clock decide.
pub fn level_limits(level: u8, clock: Option<(u64, u64)>) -> SearchLimits {
    debug_assert!((1..=MAX_LEVEL).contains(&level));
    if level < MAX_LEVEL {
        return SearchLimits::Depth(skill::depth_for(level as i32));
    }
    match clock {
        Some((time, inc)) => SearchLimits::Clock {
            time,
            inc,
            movestogo: None,
        },
        // Never Infinite: with no clock to answer to, the search would never hand a
        // move back.
        None => SearchLimits::MoveTime(2000),
    }
}

#[cfg(all(test, feature = "gui"))]
mod ponder_tests {
    use super::*;
    use crate::position::Position;
    use crate::timeman::SearchLimits;

    fn startpos() -> Position {
        Position::from_fen(super::super::game::STARTPOS).expect("the start position must parse")
    }

    fn after(pos: &Position, line: &[&str]) -> Position {
        let mut pos = pos.clone();
        for uci in line {
            let m = crate::uci::parse_uci_move(&pos, uci)
                .unwrap_or_else(|| panic!("{uci} is not legal here"));
            pos.make_move(m);
        }
        pos
    }

    /// Waits for an answer rather than hanging the suite on one that never comes.
    fn collect(engine: &mut Engine) -> Thought {
        for _ in 0..2_000 {
            if let Some(thought) = engine.poll() {
                return thought;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        panic!("the engine never answered");
    }

    #[test]
    fn a_search_hands_back_the_reply_it_expects() {
        let _guard = skill::lock_level();
        let pos = startpos();
        let mut engine = Engine::spawn();
        engine.think(&pos, &[], SearchLimits::Depth(8), MAX_LEVEL, 0);
        let thought = collect(&mut engine);

        // Pondering is nothing without this: the second move of the line the search
        // settled on is the whole of what the engine bets the player will play.
        let reply = thought.ponder.expect("a search this deep has a line behind it");
        let played = after(&pos, &[&thought.best.to_uci()]);
        assert!(
            super::super::game::is_legal(&played, reply),
            "{} cannot be played after {}",
            reply.to_uci(),
            thought.best.to_uci()
        );
    }

    #[test]
    fn a_ponder_is_claimed_by_its_own_position_and_by_no_other() {
        let _guard = skill::lock_level();
        let start = startpos();
        let guessed = after(&start, &["e2e4", "e7e5"]);
        let played = after(&start, &["e2e4", "c7c5"]);

        let mut engine = Engine::spawn();
        engine.ponder(&guessed, &[], SearchLimits::Depth(10), MAX_LEVEL, 0);
        assert!(engine.is_pondering());

        // Whatever it finds while it waits is an answer to a question nobody has asked:
        // it must not come back as a move to play.
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(engine.poll().is_none(), "a ponder was handed back as a move");

        assert!(!engine.ponderhit(&played), "the wrong position claimed the thinking");
        assert!(engine.is_pondering(), "a miss should leave the search where it was");

        assert!(engine.ponderhit(&guessed), "the position it was set on did not claim it");
        assert!(!engine.is_pondering());
        // And once claimed it answers on the engine's own time, as any search would.
        assert!(collect(&mut engine).best.is_ok());
    }

    /// Loops until `ready` or gives up, ticking the meter as a drawn frame would.
    fn watch(engine: &mut Engine, what: &str, ready: impl Fn(&Engine) -> bool) {
        for _ in 0..600 {
            engine.tick();
            if ready(engine) {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        panic!("{what}");
    }

    #[test]
    fn the_rate_says_what_the_search_is_doing_and_stops_when_it_does() {
        let _guard = skill::lock_level();
        let mut engine = Engine::spawn();
        assert!(engine.nps().is_none(), "an engine that has not started cannot have a rate");

        engine.think(&startpos(), &[], SearchLimits::MoveTime(3_000), MAX_LEVEL, 0);
        watch(&mut engine, "a running search never read as any rate", |e| e.nps().is_some());
        assert!(engine.nps().is_some_and(|nps| nps > 0));

        engine.abort();
        // And nothing once it stops: the figure is what is happening, not what happened.
        watch(&mut engine, "the rate went on reading after the search stopped", |e| {
            e.nps().is_none()
        });
    }

    #[test]
    fn a_search_that_follows_a_ponder_still_answers_to_its_clock() {
        let _guard = skill::lock_level();
        let start = startpos();
        let mut engine = Engine::spawn();
        // Set going and given up on before the worker can so much as read it — the
        // shortest a ponder can be. What follows must be an ordinary timed search, not
        // one that inherited a raised flag and searches for ever.
        engine.ponder(&after(&start, &["e2e4", "e7e5"]), &[], SearchLimits::Depth(20), MAX_LEVEL, 0);
        engine.think(&start, &[], SearchLimits::MoveTime(50), MAX_LEVEL, 0);
        assert!(collect(&mut engine).best.is_ok());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weak_levels_are_budgeted_in_depth_so_the_machine_cannot_matter() {
        for level in 1..MAX_LEVEL {
            match level_limits(level, None) {
                SearchLimits::Depth(d) => assert_eq!(d, skill::depth_for(level as i32)),
                _ => panic!("level {level} must be budgeted in depth"),
            }
        }
    }

    #[test]
    fn the_top_level_uses_the_clock_when_there_is_one() {
        assert!(matches!(
            level_limits(MAX_LEVEL, Some((60_000, 1_000))),
            SearchLimits::Clock { .. }
        ));
        assert!(matches!(
            level_limits(MAX_LEVEL, None),
            SearchLimits::MoveTime(_)
        ));
    }
}
