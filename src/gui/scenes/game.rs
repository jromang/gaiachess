//! The playing scene: pick a piece up, put it down, and everything shown while doing
//! so.

use crate::timeman::SearchLimits;
use crate::types::{Color, MT_CASTLING, MT_EN_PASSANT, Move, Piece, PieceType, Square};

use super::super::anim::{AnimEvent, BigText, BoardFlip, PieceAnim, Rng, Shake};
use super::super::assets::{Assets, CELL_W, Ui};
use super::super::audio::{Queue, Sfx};
use super::super::board;
use super::super::clock::Clocks;
use super::super::cursor::{Cursor, Step};
use super::super::engine::{Engine, MAX_LEVEL, Thought, level_limits};
use super::super::fb::{FB_W, Fb, rgba, with_alpha};
use super::super::font;
use super::super::game::{self, GameState, Outcome};
use super::super::input::{Dir, Input};
use super::super::lang::{Key, t};
use super::super::scheme::Scheme;
use super::menu::MatchConfig;

/// How fast the engine is searching, in the few characters the band can spare.
///
/// Rounded down at every step: a figure read as a measurement should never claim more
/// than the engine actually did.
fn nps_text(nps: u32) -> String {
    match nps {
        n if n < 1_000 => format!("{n} nps"),
        n if n < 1_000_000 => format!("{}k nps", n / 1_000),
        n if n < 10_000_000 => format!("{}.{}m nps", n / 1_000_000, (n % 1_000_000) / 100_000),
        n => format!("{}m nps", n / 1_000_000),
    }
}

/// Promotion choices, in the order they appear in the panel.
const PROMOTIONS: [PieceType; 4] = [
    PieceType::Queen,
    PieceType::Rook,
    PieceType::Bishop,
    PieceType::Knight,
];

/// Steps a captured piece keeps its square before fleeing, counted back from the
/// moment its captor lands on it.
const CAPTURE_LEAD: u32 = 5;
/// Steps the rook waits before following its king through a castle.
const CASTLE_ROOK_DELAY: u32 = 8;
/// Shake added when a piece touches down.
const LANDING_SHAKE: f32 = 0.09;
/// Shake added when the board comes back down from being turned round. A whole board
/// landing is heavier than one piece landing, and lighter than a word being announced.
const FLIP_SHAKE: f32 = 0.14;
/// Shake added the moment a king is put in check, before the banner lands on top of it.
/// Sized against the cartridge's own knock, which reaches about 1.2% of its screen: on
/// a canvas this wide that takes a little more than the amount a landing piece adds.
const CHECK_SHAKE: f32 = 0.35;
/// Steps the check banner holds once it has settled, before dropping out of frame.
/// A second, as on the cartridge: long enough to be read, short enough that it never
/// stands between a player and the board they are trying to answer with.
const CHECK_HOLD: u32 = 60;
/// Steps of quiet left behind once the check banner has gone, before the engine takes
/// its turn. An announcement answered the instant it ends reads as an interruption; a
/// beat of stillness lets it land.
const CHECK_PAUSE: u32 = 60;
/// Shake added each time a banner strikes its resting line.
const BANNER_SHAKE: f32 = 0.25;
/// Steps the engine's hand spends on each square it crosses. Deliberately unhurried:
/// this is what turns a silent pause into something to watch.
const CPU_SPEED: u32 = 6;

/// What the engine's hand is doing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CpuState {
    /// Not the engine's turn, or its move is already under way.
    Idle,
    /// A search is running.
    Thinking,
    /// Walking to the piece it has decided to move.
    Reaching(Move),
    /// Piece in hand, walking to where it goes.
    Carrying(Move),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum UiMode {
    /// Roaming the board, nothing picked up.
    Select,
    /// A piece is in hand, waiting for a destination.
    Move,
    /// A piece has been lifted off its square by the mouse and follows the pointer
    /// until the button is let go.
    Drag,
    /// A pawn reached the far rank and the player is choosing what it becomes.
    Promo,
    /// Pieces are in flight; the board is not taking orders.
    Ani,
    /// The in-game menu is open.
    Menu,
    /// The game is finished.
    Over,
}

/// Rows of the in-game menu, in order.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum MenuRow {
    TakeBack,
    Restart,
    Sound,
    Quit,
}

impl MenuRow {
    const ALL: [MenuRow; 4] = [MenuRow::TakeBack, MenuRow::Restart, MenuRow::Sound, MenuRow::Quit];

    /// Where the row sits in the panel, so opening the menu on a particular row names
    /// it rather than counting to it.
    fn index(self) -> usize {
        let at = MenuRow::ALL.iter().position(|row| *row == self);
        debug_assert!(at.is_some(), "{self:?} has gone from the panel");
        at.unwrap_or(0)
    }

    /// What the row reads as.
    ///
    /// The sound row says how it is set rather than what taking it would do, unlike
    /// every other row here. They are one-way actions and need no state; this one is a
    /// setting, and the label is the only thing on the screen that can say which way it
    /// currently is.
    fn label(self, sound: bool) -> Key {
        match self {
            MenuRow::TakeBack => Key::TakeBack,
            MenuRow::Restart => Key::Restart,
            MenuRow::Sound if sound => Key::SoundOn,
            MenuRow::Sound => Key::SoundOff,
            MenuRow::Quit => Key::Quit,
        }
    }
}

/// Pitch of those rows.
const MENU_ROW_H: i32 = 11;

/// How solid the ring behind the end-of-game banner is. Enough to carry the letters
/// over a busy board, little enough to still see the position that produced them.
const BANNER_WASH: u32 = 110;

/// Row of a piece's sprite cell that sits in the hand while it is being carried.
///
/// Down at the foot, for two reasons. The fist then covers only the base, leaving the
/// piece recognisable — gripped around the middle, a white fist on a white piece is a
/// smudge rather than a hand. And it holds the piece clear above the shadow it casts
/// on the square below, which is what says it has been picked up rather than pushed.
const CARRY_GRIP: i32 = 23;

/// What the scene wants the application to do next.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GameExit {
    Stay,
    ToMenu,
}

pub struct GameScene {
    pub game: GameState,
    pub flipped: bool,
    /// The turn in progress, if the board is on its way round. `flipped` changes at
    /// the halfway point, so the two together say both which side the board is being
    /// read from and how far over it has got.
    flip: Option<BoardFlip>,
    mode: UiMode,
    cursor: Cursor,
    /// Where the hand is when the mouse is driving it, `None` when the keyboard is.
    /// The square it means still goes through `cursor`, so everything that rings,
    /// points at or acts on a square is unaware of which device is in charge.
    mouse: Option<(i32, i32)>,
    /// True while the left button is down, which is what closes the hand.
    grabbing: bool,
    /// Where the piece in hand came from. Meaningless outside `Move`, `Drag` and
    /// `Promo`.
    from: Square,
    /// Legal moves leaving `from`, cached while a piece is in hand.
    targets: Vec<Move>,
    promo_index: usize,
    status: Typewriter,
    /// Pieces currently in flight.
    anims: Vec<PieceAnim>,
    /// Squares whose piece is in flight, and so must not also be drawn in place.
    hidden: u64,
    /// The move to commit once the pieces land. Absent during the opening parade.
    pending: Option<Move>,
    shake: Shake,
    /// Where the shake has the camera this step. Held rather than recomputed while
    /// drawing so the knock decays with the game, not with the display.
    camera: (i32, i32),
    rng: Rng,
    /// Which sides the engine plays, indexed by colour.
    cpu: [bool; 2],
    level: u8,
    engine: Engine,
    cpu_state: CpuState,
    /// Steps the current search has been running, for the waiting indicator.
    waited: u32,
    /// Drawn once per game: at weak levels it decides which way the engine misjudges
    /// each position, so its character holds for the whole game rather than being
    /// reshuffled between moves.
    handicap_seed: u64,
    clocks: Clocks,
    menu_row: usize,
    /// What the board was doing when the menu opened, and what closing it goes back to.
    /// The menu is reachable from turns that are not the player's — an engine thinking,
    /// its hand walking, pieces in flight — and those have to be taken up where they
    /// were left rather than reset to an idle board.
    menu_from: UiMode,
    config: MatchConfig,
    /// The word that ends the game, dropping into place.
    banner: Option<BigText>,
    /// Steps the engine still owes the announcement before it may play.
    hold: u32,
    /// Whiteout, decaying. Fires when the banner lands.
    flash: f32,
}

impl GameScene {
    pub fn new(config: MatchConfig) -> GameScene {
        let mut scene = GameScene {
            game: GameState::new(),
            flipped: false,
            flip: None,
            mode: UiMode::Select,
            cursor: Cursor::new(Square::E2),
            mouse: None,
            grabbing: false,
            from: Square::E2,
            targets: Vec::new(),
            promo_index: 0,
            status: Typewriter::default(),
            anims: Vec::new(),
            hidden: 0,
            pending: None,
            shake: Shake::default(),
            camera: (0, 0),
            rng: Rng::new(0x5EED),
            cpu: [config.white_cpu, config.black_cpu],
            level: config.level,
            engine: Engine::spawn(),
            cpu_state: CpuState::Idle,
            waited: 0,
            handicap_seed: 0,
            clocks: Clocks::new(config.time),
            menu_row: 0,
            menu_from: UiMode::Select,
            config,
            banner: None,
            hold: 0,
            flash: 0.0,
        };
        scene.begin_parade();
        scene
    }

    /// Sets which sides the engine plays.
    pub fn set_players(&mut self, white_cpu: bool, black_cpu: bool) {
        self.cpu = [white_cpu, black_cpu];
        self.engine.abort();
        self.cpu_state = CpuState::Idle;
    }

    pub fn set_level(&mut self, level: u8) {
        let level = level.clamp(1, MAX_LEVEL);
        if level != self.level {
            // Whatever is being turned over was worked out at the strength the player
            // has just changed, and playing it would be the old opponent's move.
            self.engine.stop_pondering();
        }
        self.level = level;
    }

    fn engine_to_move(&self) -> bool {
        self.game.outcome.is_none() && self.cpu[self.game.pos.side_to_move as usize]
    }

    /// True while something is still moving of its own accord: pieces in flight, or
    /// the board on its way round. What a headless capture waits out before it decides
    /// the picture has settled.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn animating(&self) -> bool {
        self.mode == UiMode::Ani || self.flip.is_some()
    }

    /// True while a piece has been picked up, whether it is being carried by the mouse
    /// or simply selected. Its square, its legal destinations and the pieces it
    /// threatens are all shown the same way either way.
    fn holding(&self) -> bool {
        matches!(self.mode, UiMode::Move | UiMode::Drag | UiMode::Promo)
    }

    pub fn update(&mut self, input: &Input, sfx: &mut Queue) -> GameExit {
        // The menu belongs to the player during the engine's turn as much as during
        // their own: the hint under the board promises `esc menu` at every moment of the
        // game, and with both sides played by the engine no turn of the player's ever
        // comes to redeem it. Without this a machine-against-machine game can be watched
        // and never stopped. Read before anything is stepped, so the walk the engine's
        // hand may be on cannot slip its arrival past the menu on the frame it opens.
        if input.cancel && self.mode != UiMode::Menu && self.engine_to_move() {
            self.open_menu(MenuRow::TakeBack, sfx);
            return GameExit::Stay;
        }
        if self.status.tick() {
            sfx.push(Sfx::Type);
        }
        // Nothing on the board moves while the menu is up. The pieces in flight are
        // already held there, and the engine's hand has to be held with them: a walk
        // running on behind the panel would reach its target unwatched, and the arrival
        // that ends it — a single step, not a state — would be spent on a scene that is
        // not listening, leaving the hand stranded on that square for good.
        let step = if self.mode == UiMode::Menu {
            Step::default()
        } else {
            self.cursor.tick()
        };
        let arrival = step.arrived;
        if step.moved {
            sfx.push(Sfx::Cursor);
        }
        self.tick_clocks(sfx);
        // Read on the wall clock like the clocks themselves, and for the same reason:
        // how fast the engine is going is a fact about seconds, not about frames.
        self.engine.tick();

        // Asking for a board that is already on its way round changes nothing: one
        // press means the other side, and it is going there.
        if input.flip && self.flip.is_none() {
            self.flip = Some(BoardFlip::new());
            sfx.push(Sfx::Jump);
        }
        if let Some(mut flip) = self.flip.take() {
            if flip.tick() {
                self.flipped = !self.flipped;
                // The hops already in the air go round with the board. Everything else
                // that knows where it is on the screen — the cursor, the square a piece
                // was picked up from — is held as a square and follows on its own.
                for anim in &mut self.anims {
                    anim.remap(board::mirror_piece_xy);
                }
            }
            if flip.done {
                self.shake.add(FLIP_SHAKE);
                sfx.push(Sfx::Land);
            } else {
                self.flip = Some(flip);
            }
        }
        if input.restart {
            self.restart();
            sfx.push(Sfx::Action);
            return GameExit::Stay;
        }
        if input.take_back && matches!(self.mode, UiMode::Select | UiMode::Move | UiMode::Over) {
            self.take_back(sfx);
            sfx.push(Sfx::Cancel);
            return GameExit::Stay;
        }
        if let Some(level) = input.level {
            self.set_level(level);
        }
        if input.toggle_white {
            self.set_players(!self.cpu[0], self.cpu[1]);
        }
        if input.toggle_black {
            self.set_players(self.cpu[0], !self.cpu[1]);
        }

        self.flash = (self.flash - 0.06).max(0.0);
        if let Some(banner) = &mut self.banner
            && banner.tick()
        {
            self.shake.add(BANNER_SHAKE);
            self.flash = self.flash.max(0.5);
        }
        // A passing banner — check — takes itself off once it has dropped out of frame,
        // and leaves a beat of quiet behind it. The ones that end a game never do: they
        // are the last thing there is to see.
        if self.banner.as_ref().is_some_and(BigText::gone) {
            self.banner = None;
            self.hold = CHECK_PAUSE;
        }

        // The knock decays here rather than while drawing, so the display cannot
        // shorten it. Ticked per frame, a knock that lasts 0.37 s at 60 Hz lasted
        // 0.15 s at 144 Hz and 0.09 s at 240 Hz — the same decay, spent faster.
        self.camera = self.shake.tick(&mut self.rng);

        self.track_pointer(input);

        match self.mode {
            UiMode::Menu => return self.update_menu(input, sfx),
            UiMode::Ani => self.update_anims(sfx),
            UiMode::Promo => self.update_promo(input, sfx),
            UiMode::Over => {
                if input.confirm || input.cancel || (self.mouse.is_some() && input.press) {
                    // A finished game has nothing to take back to, so the menu opens on
                    // the row that starts another one.
                    self.open_menu(MenuRow::Restart, sfx);
                }
            }
            // The engine waits out the announcement rather than talking over it: while
            // the word is on the screen, and for a beat after it has left. The player
            // is never held back this way — being made to wait for one's own turn is
            // lag, not staging.
            UiMode::Select | UiMode::Move | UiMode::Drag if self.engine_to_move() => {
                if self.banner.is_some() {
                    // Still being announced.
                } else if self.hold > 0 {
                    self.hold -= 1;
                } else {
                    self.update_engine(arrival, sfx);
                }
            }
            UiMode::Select | UiMode::Move | UiMode::Drag => self.update_human(input, sfx),
        }
        GameExit::Stay
    }

    /// Follows the mouse, and works out which square the hand is over.
    ///
    /// While a piece is being carried the last known position is kept rather than
    /// dropped, so a pointer that slips off the canvas mid-drag leaves the piece
    /// somewhere sensible instead of nowhere.
    fn track_pointer(&mut self, input: &Input) {
        match input.pointer() {
            Some(p) => self.mouse = Some(p),
            None if self.mode != UiMode::Drag => self.mouse = None,
            None => {}
        }
        self.grabbing = self.mouse.is_some() && input.held;
    }

    /// The square the hand is over, or `None` when it is off the board.
    fn hovered(&self) -> Option<Square> {
        self.mouse
            .and_then(|(x, y)| board::square_at(x, y, self.flipped))
    }

    /// One step of the player's turn. The keyboard walks the square cursor and acts on
    /// it with a key; the mouse carries the hand itself and drags pieces with it.
    fn update_human(&mut self, input: &Input, sfx: &mut Queue) {
        for dir in Dir::ALL {
            if input.dir(dir) {
                self.cursor.step(dir, self.flipped);
                sfx.push(Sfx::Cursor);
            }
        }
        // The mouse decides which square is meant simply by being over it.
        if let Some(sq) = self.hovered()
            && sq != self.cursor.sq
        {
            self.cursor.place(sq);
            sfx.push(Sfx::Cursor);
        }

        if self.mouse.is_some() {
            if input.press {
                self.press_on(sfx);
            }
            // Letting go is read from the button no longer being down rather than from
            // the release event, so a button released outside the window still puts the
            // piece down instead of leaving it stuck to the hand.
            if self.mode == UiMode::Drag && !input.held {
                self.drop_piece(sfx);
            }
        }

        if input.cancel {
            if self.mode == UiMode::Select {
                self.open_menu(MenuRow::TakeBack, sfx);
            } else {
                self.release_piece();
                self.enter_select();
                sfx.push(Sfx::Cancel);
            }
        } else if input.confirm {
            self.act_on_cursor(sfx);
        }
    }

    /// The left button went down: put a piece already in hand down on the square under
    /// the pointer, or pick up whatever is on it.
    fn press_on(&mut self, sfx: &mut Queue) {
        let Some(sq) = self.hovered() else { return };
        // A press on a legal destination plays the move, so a piece can be put down
        // with a second click as well as by being dragged there.
        if self.mode == UiMode::Move && sq != self.from && self.try_land(sq, None, sfx) {
            return;
        }
        let piece = self.game.pos.piece_on(sq);
        if piece != Piece::NONE && piece.color() == self.game.pos.side_to_move {
            self.from = sq;
            self.targets = self.game.moves_from(sq).collect();
            // The piece leaves its square at once: from here it is in the hand, not on
            // the board.
            self.hidden = 1u64 << sq.0;
            self.mode = UiMode::Drag;
            sfx.push(Sfx::Select);
        } else if self.mode == UiMode::Move {
            self.enter_select();
            sfx.push(Sfx::Cancel);
        }
    }

    /// The button came up while a piece was being carried.
    fn drop_piece(&mut self, sfx: &mut Queue) {
        debug_assert_eq!(self.mode, UiMode::Drag);
        let carried = self.carried_xy();
        match self.hovered() {
            // Let go where it was picked up: a plain click, so the piece stays in hand
            // and a second click puts it down.
            Some(sq) if sq == self.from => {
                self.release_piece();
                self.mode = UiMode::Move;
            }
            Some(sq) if self.try_land(sq, Some(carried), sfx) => {}
            // Anywhere it cannot go, including off the board entirely.
            _ => self.begin_return(carried),
        }
    }

    /// Puts the piece in hand down on `sq` if it may go there, hopping from `start` if
    /// it is being carried. Returns whether the square took it.
    fn try_land(&mut self, sq: Square, start: Option<(f32, f32)>, sfx: &mut Queue) -> bool {
        if self.game.needs_promotion(self.from, sq) {
            self.release_piece();
            self.cursor.place(sq);
            self.promo_index = 0;
            self.mode = UiMode::Promo;
            sfx.push(Sfx::Select);
            return true;
        }
        let Some(m) = self.targets.iter().copied().find(|m| game::drop_square(*m) == sq || m.to_sq() == sq) else {
            return false;
        };
        self.begin_move_from(m, start);
        true
    }

    /// Puts a carried piece back on its square without playing anything, which is what
    /// the board itself has believed all along.
    fn release_piece(&mut self) {
        self.hidden = 0;
    }

    /// The piece was let go somewhere it cannot go, so it hops back where it came from.
    /// Nothing is played: the position never changed.
    fn begin_return(&mut self, from_px: (f32, f32)) {
        let piece = self.game.pos.piece_on(self.from);
        debug_assert_ne!(piece, Piece::NONE, "returning a piece from an empty square");
        let home = as_f32(board::piece_xy(self.from, self.flipped));
        self.anims = vec![PieceAnim::new(
            piece,
            from_px,
            home,
            squares_between(from_px, home),
        )];
        self.hidden = 1u64 << self.from.0;
        self.pending = None;
        self.targets.clear();
        self.mode = UiMode::Ani;
    }

    /// Charges the clock of whoever is on the move, and ends the game if it runs out.
    fn tick_clocks(&mut self, sfx: &mut Queue) {
        let live = matches!(
            self.mode,
            UiMode::Select | UiMode::Move | UiMode::Drag | UiMode::Promo
        ) && self.game.outcome.is_none();
        if let Some(loser) = self.clocks.tick(self.game.pos.side_to_move, live) {
            self.engine.abort();
            self.cpu_state = CpuState::Idle;
            self.release_piece();
            self.game.flag(loser);
            self.announce(sfx, true);
        }
    }

    /// Opens the menu over whatever the board was doing, on the row named.
    ///
    /// Naming the row rather than counting to it is what keeps a finished game opening
    /// on `restart` and a running one on `take back`.
    fn open_menu(&mut self, row: MenuRow, sfx: &mut Queue) {
        debug_assert_ne!(self.mode, UiMode::Menu, "the menu is already up");
        self.menu_from = self.mode;
        self.mode = UiMode::Menu;
        self.menu_row = row.index();
        sfx.push(Sfx::Action);
    }

    /// Puts the board back the way the menu found it. A pause, not a reset: the piece
    /// the engine had in hand is still in its hand, and a parade still marching.
    fn close_menu(&mut self) {
        debug_assert_eq!(self.mode, UiMode::Menu, "closing a menu that is not open");
        debug_assert_ne!(self.menu_from, UiMode::Menu, "the menu opened over itself");
        self.mode = self.menu_from;
    }

    fn update_menu(&mut self, input: &Input, sfx: &mut Queue) -> GameExit {
        if input.dir(Dir::Down) || input.dir(Dir::Up) {
            let step = if input.dir(Dir::Down) { 1 } else { MenuRow::ALL.len() - 1 };
            self.menu_row = (self.menu_row + step) % MenuRow::ALL.len();
            sfx.push(Sfx::Cursor);
        }
        // Pointing at a row picks it out; clicking it takes it.
        let aimed = self.mouse.and_then(|(x, y)| Self::menu_row_at(x, y));
        if let Some(row) = aimed
            && row != self.menu_row
        {
            self.menu_row = row;
            sfx.push(Sfx::Cursor);
        }
        if input.cancel {
            self.close_menu();
            sfx.push(Sfx::Cancel);
            return GameExit::Stay;
        }
        if input.confirm || (input.press && aimed.is_some()) {
            let row = MenuRow::ALL[self.menu_row];
            // A setting changing, not a screen being taken — the same distinction the
            // title menu draws between its own rows.
            sfx.push(if row == MenuRow::Sound { Sfx::Confirm } else { Sfx::Action });
            match row {
                MenuRow::TakeBack => self.take_back(sfx),
                MenuRow::Restart => self.restart(),
                // Turning it off silences the blip this very press asked for, since the
                // queue is not drained until the whole step has run. Turning it back on
                // lets that same blip through, which is how the player hears it return.
                // The menu stays open either way: a setting is worth hearing twice.
                MenuRow::Sound => self.config.sound = !self.config.sound,
                MenuRow::Quit => return GameExit::ToMenu,
            }
        }
        GameExit::Stay
    }

    /// Whether the interface may make a sound, which the player can change from here
    /// without leaving the game. The application reads it back after every step and
    /// owns the copy that outlives the game.
    pub fn sound(&self) -> bool {
        self.config.sound
    }

    fn restart(&mut self) {
        self.banner = None;
        self.hold = 0;
        self.flash = 0.0;
        self.engine.new_game();
        self.cpu_state = CpuState::Idle;
        self.clocks = Clocks::new(self.config.time);
        self.game.restart();
        self.begin_parade();
    }

    fn take_back(&mut self, sfx: &mut Queue) {
        self.engine.abort();
        self.cpu_state = CpuState::Idle;
        // Two plies, so the human gets their own move back rather than handing the
        // engine a free one.
        self.game.take_back(2);
        self.banner = None;
        self.hold = 0;
        self.release_piece();
        self.enter_select();
        self.announce(sfx, false);
    }

    /// Runs the engine's turn: think, then walk the hand to the piece and on to where
    /// it goes, so the move is acted out rather than teleported.
    fn update_engine(&mut self, arrival: bool, sfx: &mut Queue) {
        match self.cpu_state {
            CpuState::Idle => {
                // A search already running on the position now on the board is the one
                // set going when the engine last moved: it has been thinking right
                // through the player's turn, and only needs telling the turn is over.
                if !self.engine.ponderhit(&self.game.pos) {
                    let limits = self.engine_budget(0);
                    self.engine
                        .think(&self.game.pos, &self.game.moves, limits, self.level, self.handicap_seed);
                }
                self.cpu_state = CpuState::Thinking;
                self.waited = 0;
                self.status.set(self.say(Key::Thinking));
            }
            CpuState::Thinking => {
                self.waited += 1;
                if let Some(thought) = self.engine.poll() {
                    let best = thought.best;
                    if best == Move::NONE {
                        // Nothing to play: leave the position alone and let the rules
                        // decide what that means.
                        self.cpu_state = CpuState::Idle;
                        return;
                    }
                    // Set going before the hand has so much as twitched. The player
                    // cannot answer until the piece has landed, so every second of the
                    // walk across the board is thinking nobody is kept waiting for.
                    self.start_pondering(&thought);
                    self.cursor.walk_to(best.from_sq(), CPU_SPEED);
                    self.cpu_state = CpuState::Reaching(best);
                    if self.cursor.sq == best.from_sq() {
                        self.grab_for_engine(best, sfx);
                    }
                }
            }
            CpuState::Reaching(m) => {
                if arrival {
                    self.grab_for_engine(m, sfx);
                }
            }
            CpuState::Carrying(m) => {
                if arrival {
                    self.cpu_state = CpuState::Idle;
                    self.begin_move(m);
                }
            }
        }
    }

    /// What the engine is to be told it has for the move it is about to be asked for.
    ///
    /// `owed` is increment the clock has not paid out yet: the increment for a move lands
    /// once the pieces have, so a budget worked out two plies early would be short by one
    /// and the engine would think itself poorer than it is.
    fn engine_budget(&self, owed: u64) -> SearchLimits {
        let clock = self.clocks.enabled().then(|| {
            (
                self.clocks.remaining_ms(self.game.pos.side_to_move) + owed,
                self.clocks.increment_ms(),
            )
        });
        level_limits(self.level, clock)
    }

    /// Sets the engine thinking about the position it expects next — its own move, then
    /// the reply it is betting on — while the player thinks about the one on the board.
    ///
    /// Full strength only. Every other rung is bounded by depth and answers in
    /// milliseconds, so there is nothing there to gain; and a rung has to be the same
    /// opponent on every machine, which thinking on the player's time would undo.
    fn start_pondering(&mut self, thought: &Thought) {
        if self.level != MAX_LEVEL {
            return;
        }
        // Nothing to bet on without an expected reply, and nothing to gain when the
        // engine is the one about to be asked for that reply.
        let Some(expected) = thought.ponder else {
            return;
        };
        if self.cpu[(!self.game.pos.side_to_move) as usize] {
            return;
        }
        let mut pos = self.game.pos.clone();
        pos.make_move(thought.best);
        // Checked rather than trusted. The reply comes out of the engine's own line and
        // is legal by construction; an illegal one reaching `make_move` would leave the
        // interface holding a position that cannot exist, and pondering is not worth
        // that risk.
        if !game::is_legal(&pos, expected) {
            debug_assert!(false, "expected reply {} is not legal", expected.to_uci());
            return;
        }
        pos.make_move(expected);
        let mut moves = self.game.moves.clone();
        moves.push(thought.best);
        moves.push(expected);
        let limits = self.engine_budget(self.clocks.increment_ms());
        self.engine.ponder(&pos, &moves, limits, self.level, self.handicap_seed);
    }

    /// The engine's hand closes on the piece and sets off for the destination.
    fn grab_for_engine(&mut self, m: Move, sfx: &mut Queue) {
        self.from = m.from_sq();
        self.targets = self.game.moves_from(self.from).collect();
        self.mode = UiMode::Move;
        self.cursor.walk_to(m.to_sq(), CPU_SPEED);
        self.cpu_state = CpuState::Carrying(m);
        self.announce_side();
        sfx.push(Sfx::Select);
    }

    fn update_anims(&mut self, sfx: &mut Queue) {
        for anim in &mut self.anims {
            match anim.tick() {
                Some(AnimEvent::Landed) => {
                    self.shake.add(LANDING_SHAKE);
                    sfx.push(Sfx::Land);
                }
                Some(AnimEvent::TookOff) => sfx.push(if anim.leaving {
                    Sfx::Capture
                } else {
                    Sfx::Jump
                }),
                None => {}
            }
        }
        if !self.anims.iter().all(|a| a.done) {
            return;
        }
        self.anims.clear();
        self.hidden = 0;
        // Nothing to apply means these pieces were the opening parade or a piece on its
        // way back to the square it never really left, so there is no news to sound.
        let played = self.pending.take();
        if let Some(m) = played {
            let mover = self.game.pos.side_to_move;
            self.game.play(m);
            self.clocks.on_move_played(mover);
        }
        // A game that has just ended has no next move to expect: anything still being
        // turned over is about a position nobody will reach.
        if self.game.outcome.is_some() {
            self.engine.abort();
        }
        self.cpu_state = CpuState::Idle;
        self.enter_select();
        self.announce(sfx, played.is_some());
    }

    /// Confirm on the square the hand is over: pick a piece up, or put one down.
    fn act_on_cursor(&mut self, sfx: &mut Queue) {
        let sq = self.cursor.sq;
        match self.mode {
            UiMode::Select => {
                let piece = self.game.pos.piece_on(sq);
                if piece != Piece::NONE && piece.color() == self.game.pos.side_to_move {
                    self.from = sq;
                    self.targets = self.game.moves_from(sq).collect();
                    self.mode = UiMode::Move;
                    sfx.push(Sfx::Select);
                }
            }
            UiMode::Move => {
                if sq == self.from {
                    self.enter_select();
                    sfx.push(Sfx::Cancel);
                } else if self.game.needs_promotion(self.from, sq) {
                    self.promo_index = 0;
                    self.mode = UiMode::Promo;
                    sfx.push(Sfx::Select);
                } else if let Some(m) = self.targets.iter().copied().find(|m| game::drop_square(*m) == sq || m.to_sq() == sq) {
                    self.begin_move(m);
                }
            }
            _ => {}
        }
    }

    fn update_promo(&mut self, input: &Input, sfx: &mut Queue) {
        if input.dir(Dir::Left) && self.promo_index > 0 {
            self.promo_index -= 1;
            sfx.push(Sfx::Cursor);
        }
        if input.dir(Dir::Right) && self.promo_index + 1 < PROMOTIONS.len() {
            self.promo_index += 1;
            sfx.push(Sfx::Cursor);
        }
        let aimed = self.mouse.and_then(|(x, y)| Self::promo_hit(x, y));
        if let Some(i) = aimed
            && i != self.promo_index
        {
            self.promo_index = i;
            sfx.push(Sfx::Cursor);
        }
        if input.confirm || (input.press && aimed.is_some()) {
            self.confirm_promotion();
        } else if input.cancel {
            self.mode = UiMode::Move;
            sfx.push(Sfx::Cancel);
        }
    }

    fn confirm_promotion(&mut self) {
        let promo = PROMOTIONS[self.promo_index];
        if let Some(m) = self.game.find_move(self.from, self.cursor.sq, Some(promo)) {
            self.begin_move(m);
        }
    }

    /// Sends the pieces a move involves into flight, the mover setting off from its own
    /// square.
    pub fn begin_move(&mut self, m: Move) {
        self.begin_move_from(m, None);
    }

    /// Sends the pieces a move involves into flight. The move itself is applied only
    /// once they land, so the board on screen and the rules never disagree. `start`
    /// overrides where the moving piece sets off from, so a piece dropped by the mouse
    /// carries on from the hand rather than snapping back to its square first.
    fn begin_move_from(&mut self, m: Move, start: Option<(f32, f32)>) {
        let pos = &self.game.pos;
        let mover = pos.piece_on(m.from_sq());
        debug_assert_ne!(mover, Piece::NONE, "moving from an empty square");

        let start = start.unwrap_or_else(|| as_f32(board::piece_xy(m.from_sq(), self.flipped)));
        let end = as_f32(board::piece_xy(m.lands_on(), self.flipped));
        let squares = squares_between(start, end);
        let travel = PieceAnim::steps_for(squares);
        let mut anims = vec![PieceAnim::new(mover, start, end, squares)];
        let mut hidden = 1u64 << m.from_sq().0;

        if let Some(taken) = game::captured_square(pos, m) {
            let victim = pos.piece_on(taken);
            hidden |= 1u64 << taken.0;
            let from = as_f32(board::piece_xy(taken, self.flipped));
            let exit = board::offboard_xy(victim.color(), self.flipped);
            // It bolts just before its captor arrives, so the two are never on the
            // same square at the same time.
            anims.push(
                PieceAnim::new(victim, from, exit, squares_between(from, exit))
                    .after(travel.saturating_sub(CAPTURE_LEAD))
                    .hurry(1.5)
                    .taken(),
            );
        } else if m.move_type() == MT_CASTLING {
            let (rook_from, rook_to) = game::castle_rook(m);
            let rook = pos.piece_on(rook_from);
            hidden |= 1u64 << rook_from.0;
            let from = as_f32(board::piece_xy(rook_from, self.flipped));
            let to = as_f32(board::piece_xy(rook_to, self.flipped));
            anims.push(
                PieceAnim::new(rook, from, to, squares_between(from, to)).after(CASTLE_ROOK_DELAY),
            );
        }

        self.pending = Some(m);
        self.hidden = hidden;
        self.anims = anims;
        self.targets.clear();
        self.mode = UiMode::Ani;
    }

    /// Flies the whole army in from the wings, staggered, to open a game.
    fn begin_parade(&mut self) {
        self.handicap_seed = self.rng.next_u64();
        let mut anims = Vec::new();
        let mut hidden = 0u64;
        let mut placed = [0u32; 2];
        for i in 0..64u8 {
            let sq = Square(i);
            let piece = self.game.pos.piece_on(sq);
            if piece == Piece::NONE {
                continue;
            }
            hidden |= 1u64 << i;
            let side = piece.color() as usize;
            let nth = placed[side];
            placed[side] += 1;
            let from = board::offboard_xy(piece.color(), self.flipped);
            let to = as_f32(board::piece_xy(sq, self.flipped));
            anims.push(
                PieceAnim::new(piece, from, to, squares_between(from, to))
                    .after(12 + (nth * 3) / 2 + self.rng.below(15)),
            );
        }
        self.anims = anims;
        self.hidden = hidden;
        self.pending = None;
        self.targets.clear();
        self.cursor = Cursor::new(if self.flipped { Square::E7 } else { Square::E2 });
        self.mode = UiMode::Ani;
        self.status.set("");
    }

    fn enter_select(&mut self) {
        self.mode = if self.game.outcome.is_some() {
            UiMode::Over
        } else {
            UiMode::Select
        };
        self.targets.clear();
        self.from = self.cursor.sq;
    }

    /// One phrase, in the language the interface is set to.
    fn say(&self, key: Key) -> &'static str {
        t(key, self.config.lang)
    }

    /// Restores the side-to-move line after the engine's "thinking" notice.
    fn announce_side(&mut self) {
        let key = match self.game.pos.side_to_move {
            Color::White => Key::WhiteToMove,
            Color::Black => Key::BlackToMove,
        };
        self.status.set(self.say(key));
    }

    /// Sets the status line to whatever just became true. `cue` says whether this is
    /// news the player should also hear and feel; re-stating the position after a
    /// take-back or a piece put back down is not.
    fn announce(&mut self, sfx: &mut Queue, cue: bool) {
        let key = match self.game.outcome {
            Some(Outcome::Checkmate(winner)) => match winner {
                Color::White => Key::WhiteWins,
                Color::Black => Key::BlackWins,
            },
            Some(Outcome::Stalemate) => Key::Stalemate,
            Some(Outcome::Draw) => Key::Draw,
            Some(Outcome::Flag(loser)) => match loser {
                Color::White => Key::WhiteFlagged,
                Color::Black => Key::BlackFlagged,
            },
            None if self.game.pos.in_check() => match self.game.pos.side_to_move {
                Color::White => Key::WhiteCheck,
                Color::Black => Key::BlackCheck,
            },
            None => match self.game.pos.side_to_move {
                Color::White => Key::WhiteToMove,
                Color::Black => Key::BlackToMove,
            },
        };
        self.status.set(self.say(key));
        // The words that drop in over the board are lettering rather than reading: they
        // are set three times the size, they are the same four words every game, and the
        // line above has just said the whole of it in the player's language.
        self.banner = match self.game.outcome {
            Some(Outcome::Checkmate(_)) => Some(BigText::new("checkmate", 84.0, 12)),
            Some(Outcome::Flag(_)) => Some(BigText::new("time", 84.0, 12)),
            Some(Outcome::Stalemate) | Some(Outcome::Draw) => {
                Some(BigText::new("draw", 84.0, 12))
            }
            None => None,
        };
        if self.game.outcome.is_some() {
            self.mode = UiMode::Over;
            if cue {
                sfx.push(Sfx::JingleMate);
            }
        } else if cue && self.game.pos.in_check() {
            // Check is the one thing that happens mid-game worth stopping the screen
            // for: the word drops in and thumps like the ones that end a game, holds a
            // second, then falls on out of frame so the board it is talking about can
            // be answered.
            self.shake.add(CHECK_SHAKE);
            self.banner = Some(BigText::new("check", 84.0, 6).for_a_moment(CHECK_HOLD));
            sfx.push(Sfx::JingleCheck);
        }
    }

    /// Draws the step the game is on. It takes `&self` on purpose: the picture is a
    /// reading of the state, never a step of it, so nothing here can be tied to how
    /// often the display happens to ask for a frame.
    pub fn draw(&self, fb: &mut Fb, assets: &Assets, scheme: &Scheme) {
        let (dx, dy) = self.camera;
        fb.camera(dx, dy);

        board::draw_board(fb, scheme, self.flipped);
        self.draw_square_marks(fb, scheme);
        self.draw_move_marks(fb, assets);

        // No arrow while a piece is actually in the hand: the piece is already where
        // the player is pointing, so drawing a line to it only adds clutter.
        if matches!(self.mode, UiMode::Move | UiMode::Promo) {
            let from = board::tile_center(self.from, self.flipped);
            let to = board::tile_center(self.cursor.sq, self.flipped);
            board::draw_arrow(fb, assets, from, to, scheme.accent);
        }

        let lifted = (self.mode == UiMode::Move).then_some(self.from);
        self.draw_threat_outlines(fb, assets, scheme);
        board::draw_pieces(
            fb,
            assets,
            scheme,
            &self.game.pos,
            self.flipped,
            self.hidden,
            lifted,
        );
        self.draw_flights(fb, assets, scheme);
        // The in-game menu draws its own pointer, so it is the one scene that does not
        // want a hand. The one standing on a square is drawn here, before the turn: it
        // is standing on the board and goes round with it. The mouse is not, and comes
        // after.
        if self.mode != UiMode::Menu {
            self.draw_hand_on_board(fb, assets);
        }
        // The turn is taken on the picture rather than in the layout: the squares, the
        // marks and everything standing on or flying over them have just been drawn
        // where they belong, and tipping what came out is what turns them as one board
        // instead of as a set of things that each happen to move.
        if let Some(flip) = &self.flip {
            fb.squash(
                board::LAYER_Y,
                board::LAYER_BOTTOM,
                board::LAYER_PIVOT,
                flip.height(),
                scheme.bg,
            );
        }
        if self.mode == UiMode::Drag {
            self.draw_carried(fb, assets, scheme);
        }
        // The promotion panel floats over the board, and the hand picking from it has
        // to stay on top of the panel — so in that mode the pointer waits its turn and
        // is drawn after the panel, outside the shake, where the panel and its hit
        // test already live.
        if !matches!(self.mode, UiMode::Menu | UiMode::Promo) {
            self.draw_pointer(fb, assets);
        }

        // The bands are painted outside the shake, so the text never wobbles off.
        fb.camera(0, 0);
        board::draw_hud_band(fb, scheme.panel_edge);
        font::print(fb, self.status.text(), 4, 4, scheme.text);
        if self.cpu_state == CpuState::Thinking {
            // Dots that keep moving while the engine does not, so a long think never
            // looks like a freeze.
            let shown = 1 + (self.waited / 10) % 3;
            let x = 4 + font::width(self.status.text()) + 3;
            for i in 0..shown as i32 {
                fb.rectfill(x + i * 3, 8, x + i * 3 + 1, 9, scheme.accent);
            }
        }
        let level = format!("lv{}", self.level);
        let right = FB_W as i32 - 4;
        font::print(fb, &level, right - font::width(&level), 4, scheme.accent);
        // Full strength is the one rung that answers to a clock rather than to a depth,
        // and the only one that goes on thinking while the player does; the rate is what
        // makes that visible. Below it a search is over in milliseconds and the figure
        // would be a flicker. One blank character of gap, so the two never touch.
        if self.level == MAX_LEVEL
            && let Some(nps) = self.engine.nps()
        {
            let text = nps_text(nps);
            let x = right - font::width(&level) - font::ADVANCE - font::width(&text);
            font::print(fb, &text, x, 4, scheme.text_dim);
        }
        font::print(
            fb,
            self.say(Key::GameHint),
            4,
            board::BOARD_Y + board::BOARD_H + 5,
            scheme.text_dim,
        );

        self.draw_clocks(fb, scheme);
        if let Some(banner) = &self.banner {
            font::print_block(
                fb,
                banner.text,
                FB_W as i32 / 2,
                banner.y as i32,
                3,
                4,
                scheme.text,
                scheme.accent,
                // The ring is a wash rather than a slab: the board a game ended on is
                // worth seeing through it, the way the menu panels let their backdrop
                // show.
                with_alpha(scheme.panel_edge, BANNER_WASH),
            );
        }
        fb.flash = self.flash;
        if self.mode == UiMode::Promo {
            self.draw_promo_panel(fb, assets, scheme);
            self.draw_pointer(fb, assets);
        }
        if self.mode == UiMode::Menu {
            self.draw_game_menu(fb, assets, scheme);
        }
    }

    /// The two clocks, on the side of the band belonging to each player.
    fn draw_clocks(&self, fb: &mut Fb, scheme: &Scheme) {
        if !self.clocks.enabled() {
            return;
        }
        let running = self.game.pos.side_to_move;
        for color in [Color::White, Color::Black] {
            let text = self.clocks.display(color);
            // Whoever is on the move gets the bright clock, so a glance is enough.
            let ink = if color == running && self.game.outcome.is_none() {
                scheme.accent
            } else {
                scheme.text
            };
            // Each clock sits on its owner's end of the board, and follows the flip.
            let bottom = (color == Color::White) != self.flipped;
            let y = if bottom {
                board::BOARD_Y + board::BOARD_H + 5
            } else {
                12
            };
            font::print(
                fb,
                &text,
                FB_W as i32 - 4 - font::width(&text),
                y,
                ink,
            );
        }
    }

    fn draw_game_menu(&self, fb: &mut Fb, assets: &Assets, scheme: &Scheme) {
        let (x, y, w, h) = Self::menu_rect();
        fb.rectfill(x + 2, y + 3, x + w + 1, y + h + 2, rgba(0x000000, 90));
        fb.rectfill(x, y, x + w - 1, y + h - 1, scheme.panel);
        fb.rect(x, y, x + w - 1, y + h - 1, scheme.panel_edge);
        for (i, row) in MenuRow::ALL.iter().enumerate() {
            let ry = y + 4 + i as i32 * MENU_ROW_H;
            let selected = i == self.menu_row;
            if selected {
                fb.rectfill(x + 3, ry - 1, x + w - 4, ry + 7, scheme.accent);
            }
            let ink = if selected { scheme.panel_edge } else { scheme.text };
            let label = self.say(row.label(self.config.sound));
            font::print_centered(fb, label, x + w / 2, ry, ink);
        }
        // Under the mouse the finger is the pointer itself; on the keyboard it stands
        // beside whichever row is picked out.
        let sprite = assets.ui(Ui::Pointer);
        let (hx, hy) = Ui::Pointer.hotspot();
        let (px, py) = match self.mouse {
            Some((mx, my)) => (mx - hx, my - hy),
            None => (
                x - sprite.w as i32 - 1,
                y + self.menu_row as i32 * MENU_ROW_H,
            ),
        };
        fb.blit(assets.ui_sheet(), sprite, px, py);
    }

    /// Draws the pieces in flight and the ones that have already landed, the lowest on
    /// screen last so they pass in front of the ones behind them.
    fn draw_flights(&self, fb: &mut Fb, assets: &Assets, scheme: &Scheme) {
        let mut order: Vec<&PieceAnim> = self.anims.iter().filter(|a| a.visible()).collect();
        order.sort_by(|a, b| a.ground().1.total_cmp(&b.ground().1));
        for anim in order {
            board::draw_flying_piece(fb, assets, scheme, anim.piece, anim.ground(), anim.height());
        }
    }

    /// Rings the square the hand is over, and tints the one a piece was picked up
    /// from. The pieces stand taller than a square, so the hand alone cannot always
    /// say which square it means.
    fn draw_square_marks(&self, fb: &mut Fb, scheme: &Scheme) {
        if self.holding() {
            let (x, y) = board::tile_xy(self.from, self.flipped);
            fb.rectfill(
                x,
                y,
                x + board::TILE_W - 1,
                y + board::TILE_H - 1,
                rgba(0xffffff, 60),
            );
        }
        if matches!(
            self.mode,
            UiMode::Select | UiMode::Move | UiMode::Drag | UiMode::Promo
        ) {
            let (x, y) = board::tile_xy(self.cursor.sq, self.flipped);
            fb.rect(
                x,
                y,
                x + board::TILE_W - 1,
                y + board::TILE_H - 1,
                scheme.accent,
            );
        }
    }

    /// The piece being carried, hanging from the hand with its shadow left on the
    /// square below. The shadow is what says where letting go would put it.
    fn draw_carried(&self, fb: &mut Fb, assets: &Assets, scheme: &Scheme) {
        let piece = self.game.pos.piece_on(self.from);
        debug_assert_ne!(piece, Piece::NONE, "carrying nothing");
        let at = self.carried_xy();
        if let Some(sq) = self.hovered() {
            let ground = as_f32(board::piece_xy(sq, self.flipped));
            let fade = board::shadow_fade(ground.1 - at.1);
            board::draw_piece_shadow(fb, assets, scheme, piece.piece_type(), ground, fade);
        }
        board::draw_loose_piece(fb, assets, scheme, piece, at);
    }

    /// True while the engine is playing its move rather than thinking about it: its hand
    /// walking to the piece and on to the square, and the flight that follows.
    fn engine_acting(&self) -> bool {
        self.engine_to_move()
            && (self.mode == UiMode::Ani
                || matches!(
                    self.cpu_state,
                    CpuState::Reaching(_) | CpuState::Carrying(_)
                ))
    }

    /// Draws the hand standing on a square, if anyone's is.
    ///
    /// It belongs to whoever is to play: the engine's while it acts out its move, the
    /// player's while the keyboard is driving — under the mouse the hand is the cursor
    /// itself. Nobody's while pieces are in flight, when there is nothing left to
    /// hold. And nobody's while the engine merely thinks: it has chosen nothing yet,
    /// and a hand parked on a square it never picked reads as stuck rather than
    /// pensive, the waiting dots in the band having already said what is going on.
    fn draw_hand_on_board(&self, fb: &mut Fb, assets: &Assets) {
        if self.mode != UiMode::Ani
            && (self.engine_acting() || (!self.engine_to_move() && self.mouse.is_none()))
        {
            self.draw_square_hand(fb, assets);
        }
    }

    /// Draws the mouse cursor.
    ///
    /// Drawn whatever else is going on — the system cursor is hidden under it, so
    /// leaving it out leaves the player with nothing to point with. The one exception
    /// is the opponent's move: that is a cutscene, and a cursor sitting in the middle
    /// of it is in the way of the very thing it is there to watch.
    fn draw_pointer(&self, fb: &mut Fb, assets: &Assets) {
        if let Some((mx, my)) = self.mouse
            && !self.engine_acting()
        {
            let item = self.pointer_item();
            let (hx, hy) = item.hotspot();
            fb.blit(assets.ui_sheet(), assets.ui(item), mx - hx, my - hy);
        }
    }

    /// What the mouse looks like.
    ///
    /// A hand while the board answers to the player: open, and closed while the button
    /// is down, whether or not it caught anything. While the engine holds the board
    /// there is nothing to take hold of, and the hand on the board is already the
    /// engine's, so the mouse falls back to the plain arrow it uses in the menus —
    /// which also says, without a word, that it is not the player's turn.
    fn pointer_item(&self) -> Ui {
        match (self.engine_to_move(), self.grabbing) {
            (true, _) => Ui::Pointer,
            (false, true) => Ui::HandGrab,
            (false, false) => Ui::HandOpen,
        }
    }

    /// The hand standing on the cursor's square, anchored by its bottom edge so the fist
    /// and the open palm sit at the same height, and set right of centre where the piece
    /// on the square is narrowest. Which square it means is said by the ring around the
    /// tile. With no button to read, having a piece in hand is what closes it.
    fn draw_square_hand(&self, fb: &mut Fb, assets: &Assets) {
        let item = if matches!(self.mode, UiMode::Move | UiMode::Promo) {
            Ui::HandGrab
        } else {
            Ui::HandOpen
        };
        let sprite = assets.ui(item);
        let (x, y) = board::tile_xy(self.cursor.sq, self.flipped);
        fb.blit(
            assets.ui_sheet(),
            sprite,
            x + board::TILE_W / 2 - 3,
            y + board::TILE_H - sprite.h as i32 + 3 - self.cursor.lift(),
        );
    }

    fn draw_move_marks(&self, fb: &mut Fb, assets: &Assets) {
        if !self.holding() {
            return;
        }
        for m in &self.targets {
            let dest = game::drop_square(*m);
            let occupied = m.move_type() != MT_CASTLING
                && (self.game.pos.piece_on(dest) != Piece::NONE
                    || m.move_type() == MT_EN_PASSANT);
            let mark = if occupied { Ui::MoveRing } else { Ui::MoveDot };
            let sprite = assets.ui(mark);
            let (x, y) = board::tile_xy(dest, self.flipped);
            fb.blit(
                assets.ui_sheet(),
                sprite,
                x + (board::TILE_W - sprite.w as i32) / 2,
                y + (board::TILE_H - sprite.h as i32) / 2,
            );
        }
    }

    /// Rings the pieces this move could take, and the king when it is in check.
    fn draw_threat_outlines(&self, fb: &mut Fb, assets: &Assets, scheme: &Scheme) {
        if !self.cursor.blink() || self.mode == UiMode::Ani {
            return;
        }
        if self.holding() {
            for m in &self.targets {
                if m.move_type() == MT_CASTLING {
                    continue;
                }
                let target = self.game.pos.piece_on(m.to_sq());
                if target != Piece::NONE {
                    board::draw_outline(
                        fb,
                        assets,
                        m.to_sq(),
                        self.flipped,
                        target.piece_type(),
                        scheme.accent,
                    );
                }
            }
        }
        if self.game.pos.in_check() {
            let king = self.game.pos.king_sq(self.game.pos.side_to_move);
            board::draw_outline(fb, assets, king, self.flipped, PieceType::King, scheme.accent);
        }
    }

    /// Screen rectangle of the promotion panel.
    fn promo_rect() -> (i32, i32, i32, i32) {
        let w = 4 * board::TILE_W + 8;
        (
            (FB_W as i32 - w) / 2,
            board::BOARD_Y + board::BOARD_H / 2 - 20,
            w,
            40,
        )
    }

    fn promo_hit(x: i32, y: i32) -> Option<usize> {
        let (px, py, w, h) = Self::promo_rect();
        if x < px || x >= px + w || y < py || y >= py + h {
            return None;
        }
        Some((((x - px - 4) / board::TILE_W) as usize).min(PROMOTIONS.len() - 1))
    }

    /// Screen rectangle of the in-game menu panel.
    fn menu_rect() -> (i32, i32, i32, i32) {
        let w = 74;
        let h = MenuRow::ALL.len() as i32 * MENU_ROW_H + 6;
        (
            (FB_W as i32 - w) / 2,
            board::BOARD_Y + board::BOARD_H / 2 - h / 2,
            w,
            h,
        )
    }

    fn menu_row_at(px: i32, py: i32) -> Option<usize> {
        let (x, y, w, h) = Self::menu_rect();
        if px < x || px >= x + w || py < y || py >= y + h {
            return None;
        }
        let i = (py - y - 3) / MENU_ROW_H;
        (0..MenuRow::ALL.len() as i32)
            .contains(&i)
            .then_some(i as usize)
    }

    /// Where the sprite cell of a carried piece goes: hanging from the hand, gripped
    /// about the middle of its body rather than balanced on the fingertips.
    fn carried_xy(&self) -> (f32, f32) {
        match self.mouse {
            Some((mx, my)) => ((mx - CELL_W as i32 / 2) as f32, (my - CARRY_GRIP) as f32),
            None => as_f32(board::piece_xy(self.from, self.flipped)),
        }
    }

    fn draw_promo_panel(&self, fb: &mut Fb, assets: &Assets, scheme: &Scheme) {
        let (x, y, w, h) = Self::promo_rect();
        fb.rectfill(x + 2, y + 3, x + w + 1, y + h + 1, rgba(0x000000, 90));
        fb.rectfill(x, y, x + w - 1, y + h - 1, scheme.panel);
        fb.rect(x, y, x + w - 1, y + h - 1, scheme.panel_edge);
        font::print_centered(fb, self.say(Key::PromoteTo), FB_W as i32 / 2, y + 3, scheme.text);

        let side = self.game.pos.side_to_move;
        for (i, pt) in PROMOTIONS.iter().enumerate() {
            let cell_x = x + 4 + i as i32 * board::TILE_W;
            if i == self.promo_index {
                fb.rectfill(
                    cell_x,
                    y + 10,
                    cell_x + board::TILE_W - 1,
                    y + h - 2,
                    scheme.accent,
                );
            }
            fb.blit(
                assets.sheet(),
                assets.piece(*pt, scheme.variant(side)),
                cell_x + (board::TILE_W - super::super::assets::CELL_W as i32) / 2,
                y + h - super::super::assets::CELL_H as i32 - 1,
            );
        }
    }
}

fn as_f32(p: (i32, i32)) -> (f32, f32) {
    (p.0 as f32, p.1 as f32)
}

/// Distance in squares between two pixel positions, which is what sets how long and
/// how high a hop is. Never below one square, so nothing moves instantly.
fn squares_between(a: (f32, f32), b: (f32, f32)) -> f32 {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    ((dx * dx + dy * dy).sqrt() / board::TILE_W as f32).max(1.0)
}

/// A status line revealed one character per step, the way a teleprinter would.
#[derive(Default)]
struct Typewriter {
    full: &'static str,
    shown: usize,
}

impl Typewriter {
    fn set(&mut self, text: &'static str) {
        self.full = text;
        self.shown = 0;
    }

    /// Advances by one character. Returns true when a character appeared, which is
    /// what the keypress sound will hang off.
    fn tick(&mut self) -> bool {
        if self.shown < self.full.len() {
            // Stepped by character rather than by byte. The status line is ascii today
            // and a test holds it there, but a slice taken through the middle of a
            // character panics, and this build aborts on a panic rather than unwinding.
            self.shown = self.full[self.shown..]
                .char_indices()
                .nth(1)
                .map_or(self.full.len(), |(next, _)| self.shown + next);
            true
        } else {
            false
        }
    }

    fn text(&self) -> &str {
        &self.full[..self.shown]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::super::lang::Lang;

    /// A scene with both sides human and the opening parade over with.
    fn hotseat() -> GameScene {
        let mut scene = GameScene::new(MatchConfig {
            white_cpu: false,
            black_cpu: false,
            // Full strength, not the rung the menu opens on: what most of these tests
            // look at — pondering above all — only happens at the top of the ladder.
            level: MAX_LEVEL,
            ..MatchConfig::default()
        });
        settle(&mut scene);
        scene
    }

    /// The move the engine would have chosen and the reply it bets on, without running a
    /// search for them: what is under test is what the interface does with a thought,
    /// not how one is arrived at.
    fn guess(scene: &GameScene, played: &str, expected: &str) -> Thought {
        let best = crate::uci::parse_uci_move(&scene.game.pos, played)
            .unwrap_or_else(|| panic!("{played} is not legal here"));
        let mut after = scene.game.pos.clone();
        after.make_move(best);
        let reply = crate::uci::parse_uci_move(&after, expected)
            .unwrap_or_else(|| panic!("{expected} is not legal after {played}"));
        Thought { best, ponder: Some(reply) }
    }

    fn settle(scene: &mut GameScene) {
        for _ in 0..1200 {
            if !scene.animating() {
                return;
            }
            scene.update(&Input::default(), &mut Queue::default());
        }
        panic!("animation never settled");
    }

    /// Plays a move the way the board itself would, and lets the pieces land.
    fn play(scene: &mut GameScene, uci: &str) {
        let square = |s: &str| {
            let b = s.as_bytes();
            Square((b[1] - b'1') * 8 + (b[0] - b'a'))
        };
        let m = scene
            .game
            .moves_from(square(&uci[..2]))
            .find(|m| m.to_uci() == uci)
            .unwrap_or_else(|| panic!("{uci} is not legal here"));
        scene.begin_move(m);
        settle(scene);
    }

    /// A step of input with the mouse over `sq`.
    fn aim(scene: &GameScene, sq: Square, held: bool, press: bool) -> Input {
        let mut input = Input::default();
        let (x, y) = board::tile_center(sq, scene.flipped);
        input.point_at(x as i32, y as i32);
        input.held = held;
        input.press = press;
        input
    }

    fn step(scene: &mut GameScene, input: Input) {
        scene.update(&input, &mut Queue::default());
    }

    /// The sound row is a setting among actions. Taking it must change the setting,
    /// leave the menu standing, and above all not be read as the row that leaves the
    /// game: the rows used to be matched by number, with everything past the second
    /// one meaning quit.
    #[test]
    fn the_sound_row_holds_the_menu_open_and_is_not_the_way_out() {
        let mut scene = hotseat();
        let mut open = Input::default();
        open.cancel = true;
        assert_eq!(scene.update(&open, &mut Queue::default()), GameExit::Stay);
        assert_eq!(scene.mode, UiMode::Menu);

        let mut take = Input::default();
        take.confirm = true;
        scene.menu_row = MenuRow::Sound.index();
        assert_eq!(scene.update(&take, &mut Queue::default()), GameExit::Stay);
        assert!(!scene.sound(), "the sound is off");
        assert_eq!(scene.mode, UiMode::Menu, "and the menu is still up");
        assert_eq!(MenuRow::Sound.label(scene.sound()), Key::SoundOff, "which the row says");

        assert_eq!(scene.update(&take, &mut Queue::default()), GameExit::Stay);
        assert!(scene.sound(), "and it comes back on");

        scene.menu_row = MenuRow::Quit.index();
        assert_eq!(
            scene.update(&take, &mut Queue::default()),
            GameExit::ToMenu,
            "the row that leaves still leaves"
        );
    }

    /// A scene with the engine on both sides, which is a game the player only watches.
    /// The weakest rung, so that nothing here ever waits on a real search.
    fn machines() -> GameScene {
        GameScene::new(MatchConfig {
            white_cpu: true,
            black_cpu: true,
            level: 1,
            ..MatchConfig::default()
        })
    }

    /// Two engines playing leave the player no turn of their own, and the way into the
    /// menu used to be a branch only their turn reached: the game could be started and
    /// then neither stopped nor left, while the band under the board went on offering
    /// `esc menu`.
    #[test]
    fn the_menu_opens_while_the_engine_is_on_the_move() {
        let mut scene = machines();
        settle(&mut scene);
        assert!(scene.engine_to_move(), "the board is the engine's, both sides of it");

        let mut cancel = Input::default();
        cancel.cancel = true;
        assert_eq!(scene.update(&cancel, &mut Queue::default()), GameExit::Stay);
        assert_eq!(scene.mode, UiMode::Menu, "esc reaches the menu, as the hint says");

        assert_eq!(scene.update(&cancel, &mut Queue::default()), GameExit::Stay);
        assert_eq!(scene.mode, UiMode::Select, "and esc again hands the board back");

        // Which makes the row that ends the game reachable, the point of all of it.
        assert_eq!(scene.update(&cancel, &mut Queue::default()), GameExit::Stay);
        scene.menu_row = MenuRow::Quit.index();
        let mut take = Input::default();
        take.confirm = true;
        assert_eq!(scene.update(&take, &mut Queue::default()), GameExit::ToMenu);
    }

    /// The menu is a pause. Opening it over pieces in flight — the opening parade, or
    /// the engine setting a piece down — has to leave them in flight, not drop them on
    /// an idle board with a move half played.
    #[test]
    fn closing_the_menu_takes_the_board_up_where_it_left_off() {
        let mut scene = machines();
        assert!(scene.animating(), "the parade is still marching");

        let mut cancel = Input::default();
        cancel.cancel = true;
        scene.update(&cancel, &mut Queue::default());
        assert_eq!(scene.mode, UiMode::Menu);

        scene.update(&cancel, &mut Queue::default());
        assert_eq!(scene.mode, UiMode::Ani, "the parade takes up where it stopped");
        settle(&mut scene);
        assert_eq!(scene.game.pos.side_to_move, Color::White, "and nothing was played");
    }

    /// Every row has to fit the panel it is written across, in a font three pixels wide.
    #[test]
    fn every_menu_row_fits_its_panel() {
        let (_, _, w, _) = GameScene::menu_rect();
        for lang in Lang::ALL {
            for row in MenuRow::ALL {
                for sound in [true, false] {
                    let label = t(row.label(sound), lang);
                    assert!(
                        font::width(label) <= w - 8,
                        "{lang:?}: {label:?} is {} wide in a panel of {w}",
                        font::width(label)
                    );
                }
            }
        }
    }

    #[test]
    fn dragging_a_piece_to_a_legal_square_plays_the_move() {
        let mut scene = hotseat();
        let grab = aim(&scene, Square::E2, true, true);
        step(&mut scene, grab);
        assert_eq!(scene.mode, UiMode::Drag);
        assert_eq!(scene.hidden, 1 << Square::E2.0, "the piece has left its square");

        let carry = aim(&scene, Square::E4, true, false);
        step(&mut scene, carry);
        assert_eq!(scene.mode, UiMode::Drag, "still carrying while the button is down");

        let release = aim(&scene, Square::E4, false, false);
        step(&mut scene, release);
        settle(&mut scene);
        assert_eq!(scene.game.moves.len(), 1);
        assert_eq!(scene.game.moves[0].to_uci(), "e2e4");
        assert_eq!(scene.hidden, 0);
    }

    #[test]
    fn dropping_a_piece_where_it_cannot_go_puts_it_back_and_plays_nothing() {
        let mut scene = hotseat();
        let before = scene.game.pos.key;
        let grab = aim(&scene, Square::E2, true, true);
        step(&mut scene, grab);
        // Three squares is no move a pawn has.
        let release = aim(&scene, Square::E5, false, false);
        step(&mut scene, release);
        assert!(scene.animating(), "the piece hops back rather than vanishing");
        assert!(scene.pending.is_none(), "nothing is waiting to be played");

        settle(&mut scene);
        assert!(scene.game.moves.is_empty());
        assert_eq!(scene.game.pos.key, before, "the position never changed");
        assert_eq!(scene.hidden, 0, "the piece is back on its square");
        assert_eq!(scene.mode, UiMode::Select);
    }

    #[test]
    fn a_click_on_a_piece_leaves_it_selected() {
        let mut scene = hotseat();
        let press = aim(&scene, Square::G1, true, true);
        step(&mut scene, press);
        let release = aim(&scene, Square::G1, false, false);
        step(&mut scene, release);
        assert_eq!(scene.mode, UiMode::Move, "picked up, waiting for a destination");
        assert_eq!(scene.hidden, 0, "and standing on its square while it waits");
        assert_eq!(scene.targets.len(), 2, "the knight has two moves to show");

        // A second click puts it down.
        let press = aim(&scene, Square::F3, true, true);
        step(&mut scene, press);
        settle(&mut scene);
        assert_eq!(scene.game.moves.len(), 1);
        assert_eq!(scene.game.moves[0].to_uci(), "g1f3");
    }

    #[test]
    fn a_check_drops_the_word_in_shakes_the_screen_and_then_takes_it_away() {
        let mut scene = hotseat();
        for m in ["e2e4", "e7e5", "f1c4", "d7d6", "c4f7"] {
            play(&mut scene, m);
        }
        assert!(scene.game.pos.in_check(), "bxf7 is check");
        assert_eq!(
            scene.banner.as_ref().map(|b| b.text),
            Some("check"),
            "the word should be on its way in"
        );
        assert_ne!(scene.mode, UiMode::Over, "a check does not end the game");

        // The knock is felt: somewhere in the steps that follow, the camera is off its
        // mark. It is jitter, so a single step proves nothing either way.
        let knocked = (0..30).any(|_| {
            scene.update(&Input::default(), &mut Queue::default());
            scene.camera != (0, 0)
        });
        assert!(knocked, "the screen never moved");

        // And then it leaves of its own accord, without the player doing anything.
        let mut steps = 30;
        while scene.banner.is_some() {
            scene.update(&Input::default(), &mut Queue::default());
            steps += 1;
            assert!(steps < 600, "the check banner never left");
        }
        assert!(steps > CHECK_HOLD, "it left before it could be read");
    }

    #[test]
    fn the_engine_lets_the_check_land_before_it_answers() {
        let _guard = crate::skill::lock_level();
        let mut scene = hotseat();
        for m in ["e2e4", "e7e5", "f1c4", "d7d6", "c4f7"] {
            play(&mut scene, m);
        }
        // Black, in check, is handed to the engine: it must sit on its hands until the
        // word has gone and the beat behind it has passed.
        scene.cpu[scene.game.pos.side_to_move as usize] = true;
        let mut steps = 0;
        while scene.banner.is_some() || scene.hold > 0 {
            scene.update(&Input::default(), &mut Queue::default());
            assert_eq!(
                scene.cpu_state,
                CpuState::Idle,
                "the engine started while the check was still being announced"
            );
            steps += 1;
            assert!(steps < 600, "the announcement never ended");
        }
        assert!(steps > CHECK_HOLD + CHECK_PAUSE, "the wait was cut short");
        // And it does get going once the board is quiet again.
        scene.update(&Input::default(), &mut Queue::default());
        assert_ne!(scene.cpu_state, CpuState::Idle, "the engine never took its turn");
    }

    #[test]
    fn a_piece_put_back_down_does_not_announce_the_check_again() {
        let mut scene = hotseat();
        for m in ["e2e4", "e7e5", "f1c4", "d7d6", "c4f7"] {
            play(&mut scene, m);
        }
        while scene.banner.is_some() {
            scene.update(&Input::default(), &mut Queue::default());
        }
        // Re-stating the position — a take-back, or a piece set back where it was — is
        // not news, and must not throw the word back on the screen.
        scene.announce(&mut Queue::default(), false);
        assert!(scene.banner.is_none(), "the banner came back uninvited");
    }

    #[test]
    fn full_strength_thinks_through_the_players_turn() {
        let _guard = crate::skill::lock_level();
        let mut scene = hotseat();
        let thought = guess(&scene, "e2e4", "e7e5");
        scene.start_pondering(&thought);
        assert!(
            scene.engine.is_pondering(),
            "the engine sat on its hands through the player's turn"
        );

        // On the position the bet names — its own move, then the reply — and not on the
        // one still on the board.
        assert!(
            !scene.engine.ponderhit(&scene.game.pos),
            "it was turning over the position it had already answered"
        );
        let mut projected = scene.game.pos.clone();
        projected.make_move(thought.best);
        projected.make_move(thought.ponder.expect("the guess names a reply"));
        assert!(scene.engine.ponderhit(&projected), "the wrong position was pondered");
    }

    #[test]
    fn the_board_reaches_the_position_that_was_being_pondered() {
        let _guard = crate::skill::lock_level();
        let mut scene = hotseat();
        // The bet is laid before either move is on the board: the engine's own, worked
        // out two plies early, and the reply it expects to it.
        scene.start_pondering(&guess(&scene, "e2e4", "e7e5"));
        play(&mut scene, "e2e4");
        play(&mut scene, "e7e5");
        assert!(scene.engine.is_pondering(), "the wait was given up on partway");
        // And now the board is there. Everything turned over while the player was
        // choosing counts towards the move about to be asked for; had the projection
        // been a ply out, or the history not carried, the two would not meet.
        assert!(
            scene.engine.ponderhit(&scene.game.pos),
            "the thinking was done on a position the game never reached"
        );
    }

    #[test]
    fn no_rung_below_the_top_thinks_on_the_players_time() {
        let _guard = crate::skill::lock_level();
        let mut scene = hotseat();
        let thought = guess(&scene, "e2e4", "e7e5");
        // A level has to be the same opponent on every machine, and it is bounded by
        // depth so that it is. Free thinking would hand the faster machine more of it.
        for level in [1, MAX_LEVEL - 1] {
            scene.set_level(level);
            scene.start_pondering(&thought);
            assert!(
                !scene.engine.is_pondering(),
                "level {level} thought on the player's time"
            );
        }
    }

    #[test]
    fn nothing_is_pondered_when_the_engine_owns_both_sides() {
        let _guard = crate::skill::lock_level();
        let mut scene = hotseat();
        let thought = guess(&scene, "e2e4", "e7e5");
        scene.set_players(true, true);
        // The engine is about to be asked for that reply itself. Guessing at it would
        // only take the machine away from answering it.
        scene.start_pondering(&thought);
        assert!(!scene.engine.is_pondering());
    }

    #[test]
    fn changing_the_level_disowns_what_was_being_pondered() {
        let _guard = crate::skill::lock_level();
        let mut scene = hotseat();
        let thought = guess(&scene, "e2e4", "e7e5");
        scene.start_pondering(&thought);
        assert!(scene.engine.is_pondering());
        scene.set_level(8);
        assert!(
            !scene.engine.is_pondering(),
            "a move worked out at the old strength was left to be played at the new one"
        );
    }

    #[test]
    fn a_game_that_ends_leaves_nothing_being_turned_over() {
        let _guard = crate::skill::lock_level();
        let mut scene = hotseat();
        for m in ["f2f3", "e7e5"] {
            play(&mut scene, m);
        }
        scene.start_pondering(&guess(&scene, "g2g4", "d8h4"));
        assert!(scene.engine.is_pondering());
        play(&mut scene, "g2g4");
        assert!(scene.engine.is_pondering(), "a game still going lost its thinking");
        // And that is mate: what was being turned over is now a position nobody reaches,
        // and a core left turning it over is a core spent on nothing.
        play(&mut scene, "d8h4");
        assert!(scene.game.outcome.is_some(), "d8h4 is mate");
        assert!(!scene.engine.is_pondering(), "the engine went on thinking past the end");
    }

    #[test]
    fn the_rate_reads_the_way_it_would_be_said() {
        assert_eq!(nps_text(0), "0 nps");
        assert_eq!(nps_text(999), "999 nps");
        assert_eq!(nps_text(850_400), "850k nps");
        assert_eq!(nps_text(2_450_000), "2.4m nps");
        assert_eq!(nps_text(12_300_000), "12m nps");
    }

    #[test]
    fn the_rate_fits_the_band_beside_everything_else_on_it() {
        // The status line grows from the left and the rate is set against the level on
        // the right. The longest of each are drawn towards each other and must not meet.
        // Neither is clipped or shortened, so where they meet they simply overprint.
        let level = font::width(&format!("lv{MAX_LEVEL}"));
        let widest = [0, 999, 999_999, 9_999_999, u32::MAX]
            .into_iter()
            .map(|n| font::width(&nps_text(n)))
            .max()
            .expect("the list is not empty");
        let rate = FB_W as i32 - 4 - level - font::ADVANCE - widest;
        for lang in Lang::ALL {
            for key in STATUS_KEYS {
                let text = t(key, lang);
                let status = 4 + font::width(text);
                assert!(
                    rate > status,
                    "{lang:?}: {text:?} reaches the rate ({rate} <= {status})"
                );
            }
        }
    }

    /// Everything `announce` and `announce_side` can put on the status band. Listed
    /// rather than reached for: what the band has to hold is the longest of them in
    /// every language, and a line left out of here is a line nothing measures.
    const STATUS_KEYS: [Key; 11] = [
        Key::Thinking,
        Key::WhiteToMove,
        Key::BlackToMove,
        Key::WhiteWins,
        Key::BlackWins,
        Key::Stalemate,
        Key::Draw,
        Key::WhiteFlagged,
        Key::BlackFlagged,
        Key::WhiteCheck,
        Key::BlackCheck,
    ];

    #[test]
    fn drawing_the_same_step_twice_gives_the_same_picture() {
        use super::super::super::fb::{FB_H, FB_W};
        use super::super::super::scheme::SCHEMES;

        // The shake used to be advanced while drawing, which tied how long a knock
        // lasted to the display's refresh rate: on a fast screen it was over before it
        // was felt. Drawing must observe the game, never move it on.
        let mut scene = hotseat();
        scene.shake.add(1.0);
        scene.update(&Input::default(), &mut Queue::default());
        let assets = Assets::load();
        let shot = |scene: &mut GameScene| {
            let mut fb = Fb::new();
            scene.draw(&mut fb, &assets, &SCHEMES[0]);
            let mut px = vec![0u8; FB_W * FB_H * 4];
            fb.copy_to(&mut px);
            px
        };
        let before = scene.camera;
        assert_ne!(before, (0, 0), "a full knock should have moved the camera");
        assert_eq!(shot(&mut scene), shot(&mut scene), "the picture drifted");
        assert_eq!(scene.camera, before, "drawing moved the camera");
    }

    #[test]
    fn every_end_of_game_banner_lets_the_board_show_through() {
        use super::super::super::fb::{FB_H, FB_W};
        use super::super::super::scheme::SCHEMES;

        // Checkmate, a flag and a draw all reach the same banner, so all three have to
        // sit on the position that produced them rather than blanking it out.
        for outcome in [
            Outcome::Checkmate(Color::Black),
            Outcome::Flag(Color::White),
            Outcome::Draw,
        ] {
            let mut scene = hotseat();
            scene.game.outcome = Some(outcome);
            scene.announce(&mut Queue::default(), false);
            // Let the banner finish falling so it is over the board, not above it.
            for _ in 0..240 {
                scene.update(&Input::default(), &mut Queue::default());
            }
            let scheme = &SCHEMES[0];
            let assets = Assets::load();
            let shot = |scene: &mut GameScene| {
                // The shake would jitter one frame against the other; nothing here is
                // about the shake.
                scene.shake = Shake::default();
                scene.camera = (0, 0);
                scene.flash = 0.0;
                let mut fb = Fb::new();
                scene.draw(&mut fb, &assets, scheme);
                let mut px = vec![0u8; FB_W * FB_H * 4];
                fb.copy_to(&mut px);
                px
            };
            let shown = shot(&mut scene);
            let held = scene.banner.take();
            let bare = shot(&mut scene);
            scene.banner = held;

            // Whatever the banner put down that is not one of its letters is its
            // backing. A slab would be a single flat colour; a wash takes the colour of
            // each square it lies over, so the board is still legible through it.
            let letters = [scheme.text.to_le_bytes(), scheme.accent.to_le_bytes()];
            let backing: std::collections::HashSet<[u8; 3]> = shown
                .chunks_exact(4)
                .zip(bare.chunks_exact(4))
                .filter(|(a, b)| a[..3] != b[..3])
                .map(|(a, _)| [a[0], a[1], a[2]])
                .filter(|c| !letters.iter().any(|l| c[..] == l[..3]))
                .collect();
            assert!(
                backing.len() > 1,
                "{outcome:?}: the banner's backing is one flat colour, {backing:?}"
            );
        }
    }

    #[test]
    fn the_mouse_keeps_a_pointer_while_the_engine_is_to_move() {
        let assets = Assets::load();
        let mut scene = hotseat();
        // Hand the side to move to the engine, which leaves it thinking: nothing of its
        // own is on the board yet.
        scene.cpu[scene.game.pos.side_to_move as usize] = true;
        scene.cpu_state = CpuState::Thinking;
        let (cx, cy) = board::tile_center(Square::E5, scene.flipped);
        let at = (cx as i32, cy as i32);

        scene.mouse = None;
        let engine_alone = frame(&mut scene, &assets);
        scene.mouse = Some(at);
        let with_pointer = frame(&mut scene, &assets);

        let changed = changed_pixels(&engine_alone, &with_pointer);
        assert!(
            !changed.is_empty(),
            "the mouse was left with nothing to point with while the engine thought"
        );
        // Everything the pointer added is the arrow under it, so the board behind is
        // untouched — and it is an arrow, not a second hand.
        let item = scene.pointer_item();
        assert_eq!(item, Ui::Pointer, "no second hand while the engine holds the board");
        let sprite = assets.ui(item);
        let (hx, hy) = item.hotspot();
        let (ox, oy) = (at.0 - hx, at.1 - hy);
        for (x, y) in changed {
            assert!(
                (ox..ox + sprite.w as i32).contains(&x) && (oy..oy + sprite.h as i32).contains(&y),
                "({x}, {y}) changed, which is outside the pointer at ({ox}, {oy})"
            );
        }
    }

    #[test]
    fn the_mouse_stands_aside_while_the_engine_plays_its_move() {
        let assets = Assets::load();
        let (cx, cy) = board::tile_center(Square::E5, false);
        let at = (cx as i32, cy as i32);

        // Both halves of the engine's move: its hand walking the board, then the piece
        // in flight. Neither is the player's to interrupt, so the cursor is not drawn.
        for flying in [false, true] {
            let mut scene = hotseat();
            scene.cpu[scene.game.pos.side_to_move as usize] = true;
            let m = scene.game.moves_from(Square::G1).next().unwrap();
            if flying {
                scene.begin_move(m);
                assert!(scene.animating());
            } else {
                scene.from = Square::G1;
                scene.mode = UiMode::Move;
                scene.cpu_state = CpuState::Carrying(m);
                scene.cursor.walk_to(m.to_sq(), CPU_SPEED);
            }
            assert!(scene.engine_acting());

            scene.mouse = None;
            let engine_alone = frame(&mut scene, &assets);
            scene.mouse = Some(at);
            let with_mouse = frame(&mut scene, &assets);
            assert!(
                changed_pixels(&engine_alone, &with_mouse).is_empty(),
                "the cursor is in the way of the move it is there to watch (flying: {flying})"
            );
        }
    }

    #[test]
    fn the_hand_points_at_the_promotion_panel_rather_than_hiding_behind_it() {
        let assets = Assets::load();
        let mut scene = hotseat();
        scene.mode = UiMode::Promo;
        let (px, py, w, h) = GameScene::promo_rect();
        let at = (px + w / 2, py + h / 2);

        scene.mouse = None;
        let panel_alone = frame(&mut scene, &assets);
        scene.mouse = Some(at);
        let with_pointer = frame(&mut scene, &assets);

        // The mouse stands in the middle of the panel, so if the hand shows at all it
        // shows on top of it: pixels inside the panel's rectangle have to change.
        let changed = changed_pixels(&panel_alone, &with_pointer);
        assert!(
            changed
                .iter()
                .any(|&(x, y)| (px..px + w).contains(&x) && (py..py + h).contains(&y)),
            "the hand went behind the promotion panel"
        );
    }

    /// One frame of the scene, as raw pixels. The shake would jitter one frame against
    /// the next, and nothing here is about the shake.
    fn frame(scene: &mut GameScene, assets: &Assets) -> Vec<u8> {
        use super::super::super::fb::{FB_H, FB_W};
        use super::super::super::scheme::SCHEMES;
        // The camera is what the shake leaves behind, and it is read while drawing
        // rather than advanced there: stilling the shake alone would leave the last
        // knock's offset on the picture.
        scene.shake = Shake::default();
        scene.camera = (0, 0);
        let mut fb = Fb::new();
        scene.draw(&mut fb, assets, &SCHEMES[0]);
        let mut px = vec![0u8; FB_W * FB_H * 4];
        fb.copy_to(&mut px);
        px
    }

    /// Where two frames differ.
    fn changed_pixels(a: &[u8], b: &[u8]) -> Vec<(i32, i32)> {
        use super::super::super::fb::FB_W;
        a.chunks_exact(4)
            .zip(b.chunks_exact(4))
            .enumerate()
            .filter(|(_, (p, q))| p != q)
            .map(|(i, _)| ((i % FB_W) as i32, (i / FB_W) as i32))
            .collect()
    }

    #[test]
    fn the_keyboard_still_drives_when_the_mouse_is_never_touched() {
        let mut scene = hotseat();
        let mut input = Input::default();
        input.fire(Dir::Left);
        step(&mut scene, input);
        assert_eq!(scene.cursor.sq, Square::D2, "the hand stepped one square");
        assert!(scene.mouse.is_none(), "and no pointer appeared out of nowhere");

        let mut confirm = Input::default();
        confirm.confirm = true;
        step(&mut scene, confirm);
        assert_eq!(scene.mode, UiMode::Move, "confirm picks the piece up");
    }
}
