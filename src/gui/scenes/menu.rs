//! The title screen: who is playing, how hard, for how long, and in what colours.
//!
//! The board tilts away behind the panel and keeps scrolling. Nothing here needs to
//! move, which is exactly why it does: a menu that is alive says the game is about
//! something.

use super::super::assets::{Assets, Ui};
use super::super::audio::{Queue, Sfx};
use super::super::clock::{Label, TimeControl};
use super::super::engine::{DEFAULT_LEVEL, MAX_LEVEL};
use super::super::fb::{FB_H, FB_W, Fb, rgba};
use super::super::font;
use super::super::input::{Dir, Input};
use super::super::lang::{self, Key, Lang, t};
use super::super::scheme::{SCHEMES, Scheme};
use super::about::About;
use crate::skill;

/// Settings a game starts with.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MatchConfig {
    pub white_cpu: bool,
    pub black_cpu: bool,
    pub level: u8,
    pub time: TimeControl,
    pub scheme: usize,
    /// Which language everything is written in. Detected once at start-up and carried
    /// with the rest, so a game keeps the language the menu was left in.
    pub lang: Lang,
    /// Whether the interface makes any sound. Kept here with the rest because it
    /// outlives a screen the same way the colours do, and so travels between the title
    /// and the board rather than being reset by every game.
    pub sound: bool,
}

impl Default for MatchConfig {
    fn default() -> MatchConfig {
        MatchConfig {
            white_cpu: false,
            black_cpu: true,
            level: DEFAULT_LEVEL,
            time: TimeControl::Unlimited,
            scheme: 0,
            // Not detected here: this is what the tests and the headless captures start
            // from as well, and they have to say the same thing on every machine. The
            // running interface asks for the machine's language in `App::new`.
            lang: Lang::En,
            sound: true,
        }
    }
}

/// The rows of the panel, in order.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Row {
    White,
    Black,
    Level,
    Time,
    Colours,
    Language,
    Sound,
    About,
    Play,
}

impl Row {
    const ALL: [Row; 9] = [
        Row::White,
        Row::Black,
        Row::Level,
        Row::Time,
        Row::Colours,
        Row::Language,
        Row::Sound,
        Row::About,
        Row::Play,
    ];

    fn label(self) -> Key {
        match self {
            Row::White => Key::White,
            Row::Black => Key::Black,
            Row::Level => Key::Level,
            Row::Time => Key::Clock,
            Row::Colours => Key::Colours,
            Row::Language => Key::Language,
            Row::Sound => Key::Sound,
            Row::About => Key::About,
            Row::Play => Key::Play,
        }
    }

    /// True for the rows that do something rather than hold a setting, which are the
    /// ones written across the panel instead of as a label and a value.
    fn is_action(self) -> bool {
        matches!(self, Row::About | Row::Play)
    }

    /// How tall the row is.
    ///
    /// Only the level needs two lines. A number between one and twenty says nothing to
    /// someone deciding who to play, so it is followed by a rating and by the kind of
    /// player that rating means — and both of those belong next to the number rather
    /// than somewhere else on the screen, which is the whole reason the row is taller
    /// instead of the description sitting under the panel.
    fn height(self) -> i32 {
        match self {
            Row::Level => ROW_H + 8,
            _ => ROW_H,
        }
    }

    /// Distance from the top of the panel's first row to the top of row `index`.
    fn top_of(index: usize) -> i32 {
        Row::ALL[..index].iter().map(|row| row.height()).sum()
    }

    /// How tall the panel has to be to hold every row.
    fn panel_height() -> i32 {
        Row::top_of(Row::ALL.len()) + 6
    }
}

/// Size of one tile of the turning background.
const TILE: f32 = 34.0;
/// How far the background slides per step.
///
/// It wraps at two tiles, moving diagonally so that one whole period is a shift of two
/// cells along each axis. Two is the chequer pattern's own period, so the wrap lands on
/// a board identical to the one it left and the loop has no seam. Sliding at some other
/// rate — or wrapping at some other distance — puts a light square where a dark one was,
/// which is the jump the eye catches.
const DRIFT: f32 = 0.35;
/// Cells drawn either side of the middle: enough to reach past the corners at any angle,
/// with the drift's swing to spare.
const REACH: i32 = 5;
/// Angle the background sits at when it is not turning, near enough to the lean of a
/// board seen from a low chair.
const TILT: f32 = 0.26;
/// Radians it turns per step, and how far that rate swings either side of it over
/// [`SPIN_PERIOD`] steps. The rate is deliberately never constant: a background turning
/// like clockwork reads as a screensaver. The swing stays under the rate, so it only
/// ever hurries and dawdles rather than backing up.
const SPIN_RATE: f32 = 0.0016;
const SPIN_SWING: f32 = 0.0011;
const SPIN_PERIOD: f32 = 640.0;

const PANEL_W: i32 = 116;
const ROW_H: i32 = 11;
/// Where the panel starts. It is placed by eye rather than centred: what has to look
/// even is the air above it, under the title, against the air below it, over the hint
/// at the foot of the screen. A row added to the panel therefore costs a few pixels
/// here as well, or the panel grows down into that hint -- which is what the language
/// row cost, taking this up from 83.
const PANEL_TOP: i32 = 72;

pub struct MenuScene {
    pub config: MatchConfig,
    row: usize,
    /// Scroll of the background, wrapping at one whole period.
    drift: f32,
    /// Angle the background is turned to, and where the rate of turn is in its swing.
    angle: f32,
    swing: f32,
    /// Height the panel has grown to, easing towards its full size.
    open: f32,
    /// Vertical position of the title, dropping into place.
    title_y: f32,
    bob: u32,
    /// Where the pointer is while the mouse is driving, `None` when the keyboard is.
    mouse: Option<(i32, i32)>,
    /// The about window, while it is open over the panel.
    about: Option<About>,
}

impl MenuScene {
    pub fn new(config: MatchConfig) -> MenuScene {
        MenuScene {
            config,
            row: Row::ALL.len() - 1,
            drift: 0.0,
            angle: TILT,
            swing: 0.0,
            open: 0.0,
            title_y: -40.0,
            bob: 0,
            mouse: None,
            about: None,
        }
    }

    /// Advances the scene. Returns true once the player has asked to start.
    pub fn update(&mut self, input: &Input, sfx: &mut Queue) -> bool {
        self.bob = self.bob.wrapping_add(1);
        self.drift = (self.drift + DRIFT) % (TILE * 2.0);
        self.swing = (self.swing + 1.0 / SPIN_PERIOD) % 1.0;
        let rate = SPIN_RATE + SPIN_SWING * (self.swing * std::f32::consts::TAU).sin();
        self.angle = (self.angle + rate) % std::f32::consts::TAU;
        let target = Row::panel_height() as f32;
        self.open += (target - self.open) / 5.0;
        self.title_y += (24.0 - self.title_y) / 8.0;

        // The about window takes the whole of the input while it is open: everything
        // behind it keeps turning, but nothing behind it is listening.
        if let Some(about) = &mut self.about {
            self.mouse = input.pointer();
            if about.update(input) {
                self.about = None;
                sfx.push(Sfx::Cancel);
            }
            return false;
        }

        if input.dir(Dir::Down) || input.dir(Dir::Up) {
            let step = if input.dir(Dir::Down) { 1 } else { Row::ALL.len() - 1 };
            self.row = (self.row + step) % Row::ALL.len();
            sfx.push(Sfx::Cursor);
        }
        // Pointing at a row picks it out and clicking takes it, so the mouse never
        // needs two clicks to do what one keypress does.
        self.mouse = input.pointer();
        let aimed = self.mouse.and_then(|(_, y)| self.row_at(y));
        if let Some(hit) = aimed
            && hit != self.row
        {
            self.row = hit;
            sfx.push(Sfx::Cursor);
        }
        if input.press && aimed.is_some() {
            return self.activate(1, sfx);
        }
        if input.dir(Dir::Right) {
            return self.activate(1, sfx);
        }
        if input.dir(Dir::Left) {
            return self.activate(-1, sfx);
        }
        if input.confirm {
            return self.activate(1, sfx);
        }
        false
    }

    /// True while the about window is over the panel, so the app can tell a cancel that
    /// closes it from one that has nothing left to close.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn about_open(&self) -> bool {
        self.about.is_some()
    }

    /// Opens the about window, by taking the row that opens it. For captures: a player
    /// gets there through `update` like everything else.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open_about(&mut self, sfx: &mut Queue) {
        let row = Row::ALL.iter().position(|r| *r == Row::About);
        debug_assert!(row.is_some(), "the about row has gone from the panel");
        if let Some(row) = row {
            self.row = row;
            self.activate(1, sfx);
        }
    }

    /// Changes the selected setting, or starts the game. `step` is which way a
    /// left/right press was going.
    fn activate(&mut self, step: i32, sfx: &mut Queue) -> bool {
        sfx.push(if Row::ALL[self.row].is_action() {
            Sfx::Action
        } else {
            Sfx::Confirm
        });
        match Row::ALL[self.row] {
            Row::White => self.config.white_cpu = !self.config.white_cpu,
            Row::Black => self.config.black_cpu = !self.config.black_cpu,
            Row::Level => {
                let level = self.config.level as i32 + step;
                self.config.level = level.rem_euclid(MAX_LEVEL as i32).max(0) as u8;
                if self.config.level == 0 {
                    self.config.level = MAX_LEVEL;
                }
            }
            Row::Time => {
                let n = TimeControl::CHOICES.len() as i32;
                let i = TimeControl::CHOICES
                    .iter()
                    .position(|t| *t == self.config.time)
                    .unwrap_or(0) as i32;
                self.config.time = TimeControl::CHOICES[(i + step).rem_euclid(n) as usize];
            }
            Row::Colours => {
                let n = SCHEMES.len() as i32;
                self.config.scheme = (self.config.scheme as i32 + step).rem_euclid(n) as usize;
            }
            Row::Language => {
                let n = Lang::ALL.len() as i32;
                let i = (self.config.lang as i32 + step).rem_euclid(n) as usize;
                self.config.lang = Lang::ALL[i];
            }
            // Turning it off silences the blip this very press asked for, since the
            // queue is not drained until the whole step has run. Turning it back on
            // lets that same blip through, which is how the player hears it return.
            Row::Sound => self.config.sound = !self.config.sound,
            Row::About => self.about = Some(About::new()),
            Row::Play => return true,
        }
        false
    }

    fn row_at(&self, y: i32) -> Option<usize> {
        let offset = y - (PANEL_TOP + 3);
        if offset < 0 {
            return None;
        }
        // Walked rather than divided: the level row is taller than the rest.
        (0..Row::ALL.len()).find(|&i| offset < Row::top_of(i + 1))
    }

    /// One phrase, in the language the menu is set to.
    fn say(&self, key: Key) -> &'static str {
        t(key, self.config.lang)
    }

    fn value(&self, row: Row) -> String {
        match row {
            Row::White => self.say(player_label(self.config.white_cpu)).to_string(),
            Row::Black => self.say(player_label(self.config.black_cpu)).to_string(),
            // The rating is spelled out as one rather than left as a bare number, which
            // could be anything. No "~" in front of it to mark it an estimate: the font
            // is three pixels wide and a tilde needs five to read as one. The about
            // window says it in words instead.
            Row::Level => {
                let rating = skill::rating_for(self.config.level as i32);
                match rating.elo {
                    0 => format!("{}", self.config.level),
                    elo => format!("{}   {elo} {}", self.config.level, self.say(Key::Elo)),
                }
            }
            Row::Time => match self.config.time.label() {
                Label::Figures(text) => text.to_string(),
                Label::Word(key) => self.say(key).to_string(),
            },
            Row::Colours => self.say(SCHEMES[self.config.scheme].name).to_string(),
            // Its own name rather than a translation of it: a language nobody in the
            // room reads is exactly the one whose name has to be recognisable.
            Row::Language => self.config.lang.autonym().to_string(),
            Row::Sound => self.say(if self.config.sound { Key::On } else { Key::Off }).to_string(),
            Row::About | Row::Play => String::new(),
        }
    }

    pub fn draw(&self, fb: &mut Fb, assets: &Assets) {
        let scheme = &SCHEMES[self.config.scheme];
        fb.clear(scheme.bg);
        self.draw_backdrop(fb, scheme);

        font::print_title(
            fb,
            "gaiachess",
            FB_W as i32 / 2,
            self.title_y as i32,
            3,
            scheme.text,
            scheme.panel_edge,
        );
        font::print_centered(
            fb,
            self.say(Key::Tagline),
            FB_W as i32 / 2,
            self.title_y as i32 + 20,
            scheme.accent,
        );
        // Quieter than the line above it, and shaded for the same reason as the hint at
        // the foot of the screen: the board keeps turning behind both of them.
        font::print_centered_shaded(
            fb,
            self.say(Key::Since),
            FB_W as i32 / 2,
            self.title_y as i32 + 29,
            scheme.tile_light,
            scheme.panel_edge,
        );

        self.draw_panel(fb, scheme);

        // Shaded, because this line sits on the turning board rather than on a panel:
        // whatever colour it is given, some square eventually slides under it.
        font::print_centered_shaded(
            fb,
            self.say(Key::MenuHint),
            FB_W as i32 / 2,
            FB_H as i32 - 14,
            scheme.tile_light,
            scheme.panel_edge,
        );

        if let Some(about) = &self.about {
            about.draw(fb, scheme, self.config.lang);
        }
        // Last, so the pointer is over the about window as well as over the panel: it
        // is the cursor, and a cursor that slides under things is no cursor at all.
        self.draw_pointer(fb, assets);
    }

    /// A board turning slowly behind everything else, drifting as it goes.
    ///
    /// The pattern is laid out square and upright, then the whole thing is turned about
    /// the middle of the canvas. Keeping the turn out of the layout is what lets the
    /// drift stay a whole number of cells and so loop without a seam, whatever angle the
    /// board happens to be at.
    fn draw_backdrop(&self, fb: &mut Fb, scheme: &Scheme) {
        let (cx, cy) = (FB_W as f32 / 2.0, FB_H as f32 / 2.0);
        let (sin, cos) = self.angle.sin_cos();
        // Centred on nothing, so the same count of cells reaches past every corner.
        let off = self.drift - TILE;
        let turn = |x: f32, y: f32| (cx + x * cos - y * sin, cy + x * sin + y * cos);
        for row in -REACH..=REACH {
            for col in -REACH..=REACH {
                if (row + col) % 2 != 0 {
                    continue;
                }
                let (x, y) = (col as f32 * TILE + off, row as f32 * TILE + off);
                fb.fill_poly(
                    &[
                        turn(x, y),
                        turn(x + TILE, y),
                        turn(x + TILE, y + TILE),
                        turn(x, y + TILE),
                    ],
                    scheme.tile_dark,
                );
            }
        }
        // Knocked back, so the panel and the title stay the subject.
        fb.rectfill(0, 0, FB_W as i32 - 1, FB_H as i32 - 1, rgba(0x000000, 110));
    }

    fn draw_panel(&self, fb: &mut Fb, scheme: &Scheme) {
        let x = (FB_W as i32 - PANEL_W) / 2;
        let h = self.open.round() as i32;
        if h <= 2 {
            return;
        }
        fb.rectfill(x + 2, PANEL_TOP + 3, x + PANEL_W + 1, PANEL_TOP + h + 2, rgba(0x000000, 90));
        fb.rectfill(x, PANEL_TOP, x + PANEL_W - 1, PANEL_TOP + h - 1, scheme.panel);
        fb.rect(x, PANEL_TOP, x + PANEL_W - 1, PANEL_TOP + h - 1, scheme.panel_edge);

        // The rows are revealed as the panel grows, so nothing spills outside it.
        let clip = fb.clip(x, PANEL_TOP, PANEL_W, h);
        for (i, row) in Row::ALL.iter().enumerate() {
            let y = PANEL_TOP + 3 + Row::top_of(i);
            let selected = i == self.row;
            if selected {
                fb.rectfill(x + 3, y - 1, x + PANEL_W - 4, y + row.height() - 3, scheme.accent);
            }
            let ink = if selected { scheme.panel_edge } else { scheme.text };
            let label = self.say(row.label());
            if row.is_action() {
                font::print_centered(fb, label, x + PANEL_W / 2, y + 1, ink);
                continue;
            }
            font::print(fb, label, x + 8, y + 1, ink);
            let value = self.value(*row);
            font::print(fb, &value, x + PANEL_W - 8 - font::width(&value), y + 1, ink);

            // The second line of the level row: who that rating plays like. Quieter than
            // the line above it, because it explains the setting rather than being it.
            if *row == Row::Level {
                let player = lang::level_player(self.config.level as i32, self.config.lang);
                let quiet = if selected { scheme.panel_edge } else { scheme.accent };
                font::print(fb, player, x + PANEL_W - 8 - font::width(player), y + 9, quiet);
            }
        }
        fb.set_clip(clip);
    }

    /// Under the mouse the finger is the pointer itself. On the keyboard it points
    /// across at whichever row is selected, nudging as it waits — and stands down while
    /// the about window is open, which the keyboard is reading rather than pointing at.
    fn draw_pointer(&self, fb: &mut Fb, assets: &Assets) {
        let sprite = assets.ui(Ui::Pointer);
        let (hx, hy) = Ui::Pointer.hotspot();
        let (px, py) = match self.mouse {
            Some((mx, my)) => (mx - hx, my - hy),
            None if self.about.is_some() => return,
            None => {
                let bob = i32::from(self.bob % 40 >= 30);
                (
                    (FB_W as i32 - PANEL_W) / 2 - sprite.w as i32 + 1 + bob,
                    PANEL_TOP + 3 + Row::top_of(self.row) - 5,
                )
            }
        };
        fb.blit(assets.ui_sheet(), sprite, px, py);
    }
}

fn player_label(is_cpu: bool) -> Key {
    if is_cpu { Key::Engine } else { Key::Human }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn press(scene: &mut MenuScene, set: impl Fn(&mut Input)) -> bool {
        let mut input = Input::default();
        set(&mut input);
        scene.update(&input, &mut Queue::default())
    }

    #[test]
    fn play_is_the_row_the_menu_opens_on() {
        let mut scene = MenuScene::new(MatchConfig::default());
        assert!(press(&mut scene, |i| i.confirm = true));
    }

    /// Every rung has to be describable on one line of a very small screen, in a font
    /// that has no capitals and barely any punctuation.
    #[test]
    fn every_level_says_who_it_plays_like_and_fits_on_screen() {
        for lang in Lang::ALL {
            for level in 1..=MAX_LEVEL {
                let player = lang::level_player(level as i32, lang);
                assert!(!player.is_empty(), "{lang:?}: level {level} describes nobody");
                assert!(
                    font::width(player) <= FB_W as i32 - 8,
                    "{lang:?}: level {level}: {player:?} is {} wide", font::width(player)
                );
                for c in player.chars() {
                    assert!(font::has_glyph(c), "{lang:?}: level {level}: no glyph for {c:?}");
                }
            }
        }

        // The ratings climb, and only full strength declines to name one.
        let mut previous = 0;
        for level in 1..MAX_LEVEL {
            let elo = skill::rating_for(level as i32).elo;
            assert!(elo > previous, "level {level} is rated no higher than the one below");
            previous = elo;
        }
        assert_eq!(skill::rating_for(MAX_LEVEL as i32).elo, 0, "full strength is not a rating");
    }

    /// The level row carries the number and its rating on one line and the kind of
    /// player on the next, and all of it has to fit inside the panel.
    #[test]
    fn the_level_row_shows_a_rating_and_still_fits_the_panel() {
        let mut menu = MenuScene::new(MatchConfig::default());
        for lang in Lang::ALL {
            menu.config.lang = lang;
            for level in 1..=MAX_LEVEL {
                menu.config.level = level;
                let value = menu.value(Row::Level);
                assert!(value.starts_with(&format!("{level}")), "{value:?} hides the level");
                let beside_label = PANEL_W - 16 - font::width(menu.say(Row::Level.label()));
                assert!(
                    font::width(&value) <= beside_label,
                    "{lang:?}: {value:?} crowds its label"
                );
                if level < MAX_LEVEL {
                    assert!(value.contains("elo"), "{value:?} does not say it is a rating");
                }
                let player = lang::level_player(level as i32, lang);
                assert!(
                    font::width(player) <= PANEL_W - 16,
                    "{lang:?}: level {level}: {player:?} is wider than the panel"
                );
            }
        }
    }

    /// A label and its setting are drawn towards each other from the two sides of the
    /// panel, neither of them clipped: what has to hold is that they never meet.
    #[test]
    fn every_row_and_its_setting_fit_the_panel_in_every_language() {
        let mut menu = MenuScene::new(MatchConfig::default());
        for lang in Lang::ALL {
            menu.config.lang = lang;
            // The settings that are words rather than figures, walked through so that
            // the longest of each is the one measured.
            for level in [1u8, MAX_LEVEL] {
                for time in TimeControl::CHOICES {
                    for scheme in 0..SCHEMES.len() {
                        for on in [true, false] {
                            menu.config.level = level;
                            menu.config.time = time;
                            menu.config.scheme = scheme;
                            menu.config.sound = on;
                            menu.config.white_cpu = on;
                            menu.config.black_cpu = !on;
                            for row in Row::ALL {
                                let label = menu.say(row.label());
                                let value = menu.value(row);
                                let width = font::width(label) + font::width(&value);
                                assert!(
                                    width <= PANEL_W - 16 - font::ADVANCE,
                                    "{lang:?}: {label:?} and {value:?} come to {width} \
                                     in a panel of {PANEL_W}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// The lines around the panel are centred on the whole screen and clipped by
    /// nothing, so a long one runs off both edges at once.
    #[test]
    fn the_writing_around_the_panel_fits_the_screen() {
        for lang in Lang::ALL {
            for key in [Key::Tagline, Key::Since, Key::MenuHint] {
                let text = t(key, lang);
                assert!(
                    font::width(text) <= FB_W as i32 - 4,
                    "{lang:?}: {text:?} is {} wide on a screen of {FB_W}",
                    font::width(text)
                );
            }
        }
    }

    /// The taller level row must not throw off which row the mouse is over, nor let the
    /// panel outgrow the screen.
    #[test]
    fn the_taller_level_row_keeps_the_panel_in_order() {
        let menu = MenuScene::new(MatchConfig::default());
        let top = PANEL_TOP + 3;
        for (i, row) in Row::ALL.iter().enumerate() {
            let y = top + Row::top_of(i);
            assert_eq!(menu.row_at(y), Some(i), "the top of row {i} points elsewhere");
            assert_eq!(menu.row_at(y + row.height() - 1), Some(i), "row {i} ends early");
        }
        assert_eq!(menu.row_at(top - 1), None, "a row above the panel");
        assert_eq!(menu.row_at(top + Row::top_of(Row::ALL.len())), None, "a row below the panel");
        assert!(
            PANEL_TOP + Row::panel_height() < FB_H as i32 - 16,
            "the panel reaches the hint at the foot of the screen"
        );
    }

    /// Whoever presses play without touching the ladder must get a game, not the engine
    /// at full strength. The rung the panel opens on is therefore one with a rating on
    /// it — a player, not a wall.
    #[test]
    fn the_menu_opens_on_a_level_that_can_be_beaten() {
        let level = MatchConfig::default().level;
        assert!(level < MAX_LEVEL, "the title screen offers full strength by default");
        assert!(
            skill::rating_for(level as i32).elo > 0,
            "level {level} names no rating to play against"
        );
    }

    #[test]
    fn the_level_wraps_around_within_the_ladder() {
        let mut scene = MenuScene::new(MatchConfig::default());
        scene.row = 2;
        scene.config.level = MAX_LEVEL;
        assert!(!press(&mut scene, |i| i.fire(Dir::Right)));
        assert_eq!(scene.config.level, 1);
        assert!(!press(&mut scene, |i| i.fire(Dir::Left)));
        assert_eq!(scene.config.level, MAX_LEVEL);
    }

    #[test]
    fn the_clock_choice_wraps_both_ways() {
        let mut scene = MenuScene::new(MatchConfig::default());
        scene.row = 3;
        let first = scene.config.time;
        for _ in 0..TimeControl::CHOICES.len() {
            press(&mut scene, |i| i.fire(Dir::Right));
        }
        assert_eq!(scene.config.time, first, "a full turn must come back round");
    }

    /// The backdrop alone, drawn at a given point in its drift.
    fn backdrop(drift: f32, angle: f32) -> Vec<u8> {
        let mut scene = MenuScene::new(MatchConfig::default());
        scene.drift = drift;
        scene.angle = angle;
        let mut fb = Fb::new();
        scene.draw_backdrop(&mut fb, &SCHEMES[0]);
        let mut out = vec![0u8; FB_W * FB_H * 4];
        fb.copy_to(&mut out);
        out
    }

    #[test]
    fn the_backdrop_loops_without_a_seam() {
        // One whole period on, the board must be indistinguishable from where it
        // started — otherwise the wrap swaps light squares for dark and jumps.
        for angle in [0.0, TILT, 1.1, 2.7] {
            assert_eq!(
                backdrop(0.0, angle),
                backdrop(TILE * 2.0, angle),
                "seam at angle {angle}"
            );
        }
    }

    #[test]
    fn the_backdrop_reaches_past_every_corner() {
        // The cells are laid out square and then turned about the middle, so what has
        // to be covered is a disc: the distance from the middle to a corner, which no
        // angle can reach beyond. Fall short and a straight edge sweeps across the
        // picture as the board turns.
        let needed = (FB_W as f32 / 2.0).hypot(FB_H as f32 / 2.0);
        // Worst case the drift has pushed the grid a whole tile off centre.
        let first = -(REACH as f32) * TILE - TILE;
        let last = REACH as f32 * TILE + TILE + TILE;
        assert!(first <= -needed, "grid starts at {first}, must reach {}", -needed);
        assert!(last >= needed, "grid ends at {last}, must reach {needed}");
    }

    #[test]
    fn pointing_at_a_row_picks_it_out_and_clicking_takes_it() {
        let mut scene = MenuScene::new(MatchConfig::default());
        let (x, y) = (FB_W as i32 / 2, PANEL_TOP + 4);
        let mut hover = Input::default();
        hover.point_at(x, y);
        assert!(!scene.update(&hover, &mut Queue::default()));
        assert_eq!(scene.row, 0, "aiming at the first row selects it");

        let mut click = Input::default();
        click.point_at(x, y);
        click.press = true;
        assert!(!scene.update(&click, &mut Queue::default()));
        assert!(scene.config.white_cpu, "one click is enough to change it");
    }

    #[test]
    fn a_click_away_from_the_rows_changes_nothing() {
        let mut scene = MenuScene::new(MatchConfig::default());
        let before = scene.config;
        let mut click = Input::default();
        click.point_at(4, 4);
        click.press = true;
        assert!(!scene.update(&click, &mut Queue::default()), "and does not start a game");
        assert_eq!(scene.config, before);
    }

    #[test]
    fn about_opens_over_the_menu_and_hands_it_back_on_closing() {
        let mut scene = MenuScene::new(MatchConfig::default());
        scene.open_about(&mut Queue::default());
        assert!(scene.about.is_some());

        // The menu behind it hears nothing while it is open, so scrolling the roll does
        // not walk the rows underneath it as well.
        let row = scene.row;
        assert!(!press(&mut scene, |i| i.fire(Dir::Down)));
        assert_eq!(scene.row, row, "a row moved behind the window");

        assert!(!press(&mut scene, |i| i.cancel = true), "and no game started");
        assert!(scene.about.is_none(), "esc closes it");
        assert!(!press(&mut scene, |i| i.fire(Dir::Down)));
        assert_ne!(scene.row, row, "the menu takes the keys back");
    }

    /// The one row here that is neither an action nor a list to walk: it shows how it
    /// is set, and either arrow turns it over.
    /// The language row walks the same way the others do, and every language it
    /// stops at names itself.
    #[test]
    fn the_language_row_walks_every_language_and_comes_back() {
        let mut scene = MenuScene::new(MatchConfig::default());
        let row = Row::ALL.iter().position(|r| *r == Row::Language);
        scene.row = row.expect("the language row has gone from the panel");
        let mut seen = Vec::new();
        for _ in 0..Lang::ALL.len() {
            assert_eq!(scene.value(Row::Language), scene.config.lang.autonym());
            seen.push(scene.config.lang);
            assert!(!press(&mut scene, |i| i.fire(Dir::Right)), "and no game started");
        }
        assert_eq!(seen, Lang::ALL.to_vec(), "a language is missing from the walk");
        assert_eq!(scene.config.lang, Lang::En, "a full turn must come back round");
        press(&mut scene, |i| i.fire(Dir::Left));
        assert_eq!(scene.config.lang, Lang::Pt, "the other arrow goes the other way");
    }

    #[test]
    fn the_sound_row_says_which_way_it_is_and_turns_over() {
        let mut scene = MenuScene::new(MatchConfig::default());
        let row = Row::ALL.iter().position(|r| *r == Row::Sound);
        scene.row = row.expect("the sound row has gone from the panel");
        assert!(scene.config.sound, "a game with no sound is not the one to start with");
        assert_eq!(scene.value(Row::Sound), "on");

        assert!(!press(&mut scene, |i| i.confirm = true), "and no game started");
        assert!(!scene.config.sound);
        assert_eq!(scene.value(Row::Sound), "off", "the row says so");
        // And it says it in whatever language the panel is in.
        scene.config.lang = Lang::Fr;
        assert_eq!(scene.value(Row::Sound), "non");

        press(&mut scene, |i| i.fire(Dir::Left));
        assert!(scene.config.sound, "either arrow turns it back over");
    }

    #[test]
    fn sides_can_each_be_handed_to_the_engine() {
        let mut scene = MenuScene::new(MatchConfig::default());
        scene.row = 0;
        press(&mut scene, |i| i.confirm = true);
        assert!(scene.config.white_cpu);
        scene.row = 1;
        press(&mut scene, |i| i.confirm = true);
        assert!(!scene.config.black_cpu);
    }
}
