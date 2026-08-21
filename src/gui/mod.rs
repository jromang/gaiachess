//! Optional pixel-art graphical interface (feature `gui`).
//!
//! The interface owns the real main thread because the platform event loop has to
//! run there; the engine gets its own deep-stack worker instead.

mod anim;
mod assets;
mod audio;
mod board;
mod clock;
mod cursor;
mod engine;
mod fb;
mod font;
mod game;
mod input;
mod lang;
mod loading;
mod scenes;
mod scheme;
mod synth;

/// The rung a game starts on when nobody has picked one.
pub use engine::DEFAULT_LEVEL;

use macroquad::prelude::*;
// A tab neither closes its own window nor decides its shape, so the two calls that do
// are desktop-only -- and with them everything they are called from, down to the
// headless captures that only ever run from a command line.
#[cfg(not(target_arch = "wasm32"))]
use macroquad::{miniquad::window::order_quit, window::request_new_screen_size};

use assets::Assets;
use audio::{Audio, Queue};
use fb::{FB_H, FB_W, Fb};
use input::Input;
use scenes::game::{GameExit, GameScene};
use scenes::menu::{MatchConfig, MenuScene};
use scheme::SCHEMES;

/// Logic advances at a fixed 60 Hz whatever the display refresh rate is.
const STEP: f32 = 1.0 / 60.0;
/// Cap on catch-up steps per frame, so a stall cannot spiral into a freeze.
const MAX_STEPS: u32 = 3;
/// Window scale used on first launch.
const INITIAL_SCALE: i32 = 4;
/// Fade added or removed per step when moving between screens.
const FADE_RATE: f32 = 0.08;

/// The screen currently in charge.
enum Scene {
    Menu(MenuScene),
    Game(Box<GameScene>),
}

/// Drives the fade that covers a change of screen: out to black, swap, back in.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Fade {
    None,
    Out,
    In,
}

/// Everything that outlives a single screen.
struct App {
    scene: Scene,
    config: MatchConfig,
    fade: Fade,
    level: f32,
}

impl App {
    fn new() -> App {
        // The one place the machine is asked what language it is in: `MatchConfig`'s
        // own default has to be the same on every machine, for the tests and the
        // headless captures.
        let config = MatchConfig {
            lang: lang::detect(),
            ..MatchConfig::default()
        };
        App {
            // Straight to the menu: the title drops in over the turning board there, so
            // a splash before it would only be the same picture held still.
            scene: Scene::Menu(MenuScene::new(config)),
            config,
            fade: Fade::In,
            level: 1.0,
        }
    }

    fn update(&mut self, input: &Input, sfx: &mut Queue) {
        match self.fade {
            Fade::Out => {
                self.level += FADE_RATE;
                if self.level >= 1.0 {
                    self.level = 1.0;
                    self.swap();
                    self.fade = Fade::In;
                }
                return;
            }
            Fade::In => {
                self.level -= FADE_RATE;
                if self.level <= 0.0 {
                    self.level = 0.0;
                    self.fade = Fade::None;
                }
            }
            Fade::None => {}
        }

        // Input is ignored while a screen is still arriving, so a held key cannot
        // fall through onto the screen behind it.
        let idle = Input::default();
        let seen = if self.fade == Fade::None { input } else { &idle };
        let done = match &mut self.scene {
            Scene::Menu(scene) => {
                let started = scene.update(seen, sfx);
                self.config = scene.config;
                started
            }
            Scene::Game(scene) => {
                let exit = scene.update(seen, sfx);
                // Only the sound comes back. The rest of the game's copy is the one it
                // was handed at the start, and the colours have been free to change
                // behind its back ever since — copying the lot would put them back.
                self.config.sound = scene.sound();
                exit == GameExit::ToMenu
            }
        };
        if done {
            self.fade = Fade::Out;
        }
    }

    /// True when a cancel has nothing left to back out of: the title is up, no window
    /// is open over it, and no screen is on its way in.
    ///
    /// Escape backs out one screen at a time, and there is nothing behind the title —
    /// so closing that is closing the game. Anywhere else it has work to do: a single
    /// global escape means the player who reaches for the in-game menu resigns the
    /// game to the desktop instead.
    #[cfg(not(target_arch = "wasm32"))]
    fn quit_requested(&self, input: &Input) -> bool {
        input.cancel
            && self.fade == Fade::None
            && matches!(&self.scene, Scene::Menu(menu) if !menu.about_open())
    }

    /// Moves on to the screen that follows the one just finished.
    fn swap(&mut self) {
        self.scene = match &self.scene {
            Scene::Menu(_) => Scene::Game(Box::new(GameScene::new(self.config))),
            Scene::Game(_) => Scene::Menu(MenuScene::new(self.config)),
        };
    }

    fn draw(&mut self, fb: &mut Fb, assets: &Assets) {
        match &mut self.scene {
            Scene::Menu(scene) => scene.draw(fb, assets),
            Scene::Game(scene) => scene.draw(fb, assets, &SCHEMES[self.config.scheme]),
        }
        fb.fade = self.level;
    }
}

/// What a headless capture should show. Everything is driven through the ordinary
/// input path, so a screenshot exercises the same code a player would.
pub struct Shot<'a> {
    pub path: &'a str,
    /// Screen to capture: "menu", "about", "game" or "gamemenu".
    pub scene: &'a str,
    /// Moves to play first, in UCI notation.
    pub moves: &'a str,
    /// Squares to click, in order.
    pub clicks: &'a str,
    /// Square to leave the mouse pointing at, so the shot shows the hand on the board.
    pub hover: &'a str,
    /// A piece to pick up and carry, as "from to", still held when the shot is taken.
    pub drag: &'a str,
    /// A piece to pick up, carry and let go of, as "from to". Dropping it somewhere it
    /// cannot go is what shows the piece hopping back.
    pub drop: &'a str,
    /// A move to set flying but not let finish, so the shot catches it mid-hop.
    pub fly: &'a str,
    /// Turn the board round, as pressing F does. Applied after everything else is
    /// staged, so the closing ticks are what decide whether the shot catches the turn
    /// partway over or the board already on its other side.
    pub flip: bool,
    /// Engine moves to play against itself before the shot, which exercises the whole
    /// chain: worker thread, level budget, hand walk and animation.
    pub selfplay: u32,
    /// Level the engine plays at during self-play.
    pub level: u8,
    /// Logic steps to let run before the shot, so animations and the status line
    /// settle instead of being caught mid-reveal.
    pub ticks: u32,
    pub scheme: usize,
    /// Index into the clock choices offered by the menu.
    pub clock: usize,
    /// Language tag to draw the screen in, e.g. "fr". Anything unrecognised is English.
    pub lang: &'a str,
}

pub fn run(shot: Option<Shot<'_>>) {
    // Screenshots are a command-line affair; a browser build has no path to write to and
    // no reason to carry a PNG encoder.
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(shot) = shot {
        capture(&shot);
        return;
    }
    #[cfg(target_arch = "wasm32")]
    let _ = shot;
    let conf = Conf {
        window_title: String::from("GaiaChess"),
        window_width: FB_W as i32 * INITIAL_SCALE,
        window_height: FB_H as i32 * INITIAL_SCALE,
        window_resizable: true,
        high_dpi: true,
        sample_count: 1,
        icon: Some(assets::window_icon()),
        ..Default::default()
    };
    macroquad::Window::from_config(conf, amain());
}

async fn amain() {
    let assets = Assets::load();
    let audio = Audio::load();
    let mut app = App::new();
    let mut input = Input::default();
    let mut sfx = Queue::default();
    let mut fb = Fb::new();
    let mut screen = Image::gen_image_color(FB_W as u16, FB_H as u16, BLACK);
    let texture = Texture2D::from_image(&screen);
    texture.set_filter(FilterMode::Nearest);

    let mut acc = 0.0f32;
    #[cfg(not(target_arch = "wasm32"))]
    let mut window = (screen_width(), screen_height());

    loop {
        // Means nothing in a tab: the canvas is sized by the page, and `viewport`
        // already letterboxes whatever shape it gets.
        #[cfg(not(target_arch = "wasm32"))]
        keep_window_shape(&mut window);
        input.sample();
        // The hand is the cursor: the system one is hidden over the picture and handed
        // straight back the moment the pointer leaves it, so the desktop stays usable.
        show_mouse(!input.on_canvas());

        acc += get_frame_time();
        let mut steps = 0;
        while acc >= STEP && steps < MAX_STEPS {
            acc -= STEP;
            steps += 1;
            if input.next_scheme {
                app.config.scheme = (app.config.scheme + 1) % SCHEMES.len();
            }
            #[cfg(not(target_arch = "wasm32"))]
            if app.quit_requested(&input) {
                order_quit();
            }
            app.update(&input, &mut sfx);
            input.consume();
        }
        // Silence drops the requests rather than holding them: sounds asked for while
        // the interface is muted are gone, not saved up to arrive all at once when it
        // is turned back on. Muting also leaves the device open, so coming back on is
        // immediate.
        match (&audio, app.config.sound) {
            (Some(audio), true) => {
                for clip in sfx.drain() {
                    audio.play(clip);
                }
            }
            _ => {
                sfx.drain();
            }
        }
        // Drop the backlog instead of sprinting through it after a stall.
        if acc >= STEP {
            acc = 0.0;
        }

        app.draw(&mut fb, &assets);
        // Over whatever the scene drew, and only while there is something to say.
        loading::draw(&mut fb, scheme::SCHEMES[0].text, app.config.lang);

        fb.copy_to(&mut screen.bytes);
        texture.update(&screen);
        present(&texture);
        next_frame().await;
    }
}

/// Renders one frame straight to a PNG. Every scene draws through the CPU
/// framebuffer, so this needs neither a window nor a GPU, which is what makes it
/// usable over a remote session or while another application owns the screen.
#[cfg(not(target_arch = "wasm32"))]
fn capture(shot: &Shot<'_>) {
    let assets = Assets::load();
    let mut fb = Fb::new();
    let config = MatchConfig {
        scheme: shot.scheme % SCHEMES.len(),
        level: shot.level,
        lang: lang::for_tag(shot.lang),
        time: clock::TimeControl::CHOICES[shot.clock % clock::TimeControl::CHOICES.len()],
        ..MatchConfig::default()
    };

    match shot.scene {
        "menu" | "about" => {
            let mut scene = MenuScene::new(config);
            // Opened the way a player opens it, by taking the row that opens it, so the
            // shot goes through the same path.
            if shot.scene == "about" {
                scene.open_about(&mut Queue::default());
            }
            for _ in 0..shot.ticks {
                scene.update(&Input::default(), &mut Queue::default());
            }
            scene.draw(&mut fb, &assets);
        }
        _ => {
            let mut scene = GameScene::new(config);
            let mouse = capture_game(&mut scene, shot);
            if shot.scene == "gamemenu" {
                let mut open = mouse.resting();
                open.cancel = true;
                scene.update(&open, &mut Queue::default());
                scene.update(&mouse.resting(), &mut Queue::default());
            }
            scene.draw(&mut fb, &assets, &SCHEMES[config.scheme]);
        }
    }

    let mut img = Image::gen_image_color(FB_W as u16, FB_H as u16, BLACK);
    fb.copy_to(&mut img.bytes);
    // export_png flips rows for GL orientation, so pre-flipping cancels it out.
    let stride = FB_W * 4;
    for row in 0..FB_H / 2 {
        let (top, bottom) = img.bytes.split_at_mut((row + 1) * stride);
        let opposite = (FB_H - 2 * row - 2) * stride;
        top[row * stride..].swap_with_slice(&mut bottom[opposite..opposite + stride]);
    }
    img.export_png(shot.path);
    println!("wrote {} ({FB_W}x{FB_H})", shot.path);
}

/// The mouse as a capture leaves it: where it points and whether its button is down.
/// A hover, or a drag still in progress, has to hold for the shot itself and not only
/// for the step that set it up.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Default)]
struct Mouse {
    at: Option<(i32, i32)>,
    held: bool,
}

#[cfg(not(target_arch = "wasm32"))]
impl Mouse {
    /// A step's worth of input with the mouse in this state and nothing else pressed.
    fn resting(self) -> Input {
        let mut input = Input::default();
        if let Some((x, y)) = self.at {
            input.point_at(x, y);
            input.held = self.held;
        }
        input
    }

    /// The step the button goes down on.
    fn pressing(self) -> Input {
        let mut input = self.resting();
        input.press = true;
        input.held = true;
        input
    }
}

/// Puts a game scene into the state a capture asked for, by playing it there. Returns
/// where the mouse was left, so a screen opened afterwards still has the hand on it.
#[cfg(not(target_arch = "wasm32"))]
fn capture_game(scene: &mut GameScene, shot: &Shot<'_>) -> Mouse {
    // A capture is a still life: no side should start thinking mid-shot.
    scene.set_players(false, false);

    let staged = [shot.moves, shot.clicks, shot.hover, shot.drag, shot.drop]
        .iter()
        .any(|s| !s.trim().is_empty());
    // With nothing to set up, --shot-ticks alone catches the opening parade partway
    // through; otherwise the parade has to finish before moves can be played.
    if staged || shot.selfplay > 0 {
        settle(scene, Mouse::default());
    }
    for token in shot.moves.split_whitespace() {
        let m = crate::uci::parse_uci_move(&scene.game.pos, token)
            .unwrap_or_else(|| panic!("illegal move in --shot-moves: {token}"));
        scene.begin_move(m);
        settle(scene, Mouse::default());
    }
    if shot.selfplay > 0 {
        scene.set_players(true, true);
        let target = scene.game.moves.len() + shot.selfplay as usize;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(180);
        while scene.game.moves.len() < target && scene.game.outcome.is_none() {
            scene.update(&Input::default(), &mut Queue::default());
            // Give the engine thread the machine while it is thinking, instead of
            // spinning a whole core waiting for it.
            std::thread::sleep(std::time::Duration::from_micros(200));
            assert!(std::time::Instant::now() < deadline, "self-play stalled");
        }
        scene.set_players(false, false);
        settle(scene, Mouse::default());
        let played: Vec<String> = scene.game.moves.iter().map(|m| m.to_uci()).collect();
        println!("self-play: {}", played.join(" "));
    }

    // Where the mouse is left for the closing ticks and the shot itself.
    let mut mouse = Mouse::default();
    for token in shot.clicks.split_whitespace() {
        mouse = Mouse { at: Some(centre(scene, token, "--shot-clicks")), held: false };
        scene.update(&mouse.pressing(), &mut Queue::default());
        scene.update(&mouse.resting(), &mut Queue::default());
        settle(scene, mouse);
    }
    if let Some(token) = shot.hover.split_whitespace().next() {
        mouse = Mouse { at: Some(centre(scene, token, "--shot-hover")), held: false };
    }
    for (flag, text, keep_hold) in [("--shot-drag", shot.drag, true), ("--shot-drop", shot.drop, false)] {
        let mut squares = text.split_whitespace();
        let (Some(from), Some(to)) = (squares.next(), squares.next()) else {
            continue;
        };
        // Pick the piece up where it stands, carry it, and either keep hold of it or
        // let go — which is what tells a legal drop from one that hops back.
        mouse = Mouse { at: Some(centre(scene, from, flag)), held: true };
        scene.update(&mouse.pressing(), &mut Queue::default());
        mouse.at = Some(centre(scene, to, flag));
        scene.update(&mouse.resting(), &mut Queue::default());
        if !keep_hold {
            mouse.held = false;
            scene.update(&mouse.resting(), &mut Queue::default());
        }
    }

    if let Some(token) = shot.fly.split_whitespace().next() {
        let m = crate::uci::parse_uci_move(&scene.game.pos, token)
            .unwrap_or_else(|| panic!("illegal move in --shot-fly: {token}"));
        scene.begin_move(m);
    }
    // Turned last, so everything staged above was set up on a board the right way
    // round and the closing ticks catch the turn itself: fewer than the sixteen steps
    // it takes and the board is caught partway over, more and it has landed.
    if shot.flip {
        let mut turn = mouse.resting();
        turn.flip = true;
        scene.update(&turn, &mut Queue::default());
    }
    for _ in 0..shot.ticks {
        scene.update(&mouse.resting(), &mut Queue::default());
    }
    mouse
}

/// Canvas pixel at the middle of a named square.
#[cfg(not(target_arch = "wasm32"))]
fn centre(scene: &GameScene, token: &str, flag: &str) -> (i32, i32) {
    let sq = crate::types::Square::from_string(token)
        .unwrap_or_else(|| panic!("not a square in {flag}: {token}"));
    let (x, y) = board::tile_center(sq, scene.flipped);
    (x as i32, y as i32)
}

/// Runs the scene on until nothing is in flight, so a capture set up by playing moves
/// shows the board at rest rather than mid-hop.
#[cfg(not(target_arch = "wasm32"))]
fn settle(scene: &mut GameScene, mouse: Mouse) {
    for _ in 0..1200 {
        if !scene.animating() {
            return;
        }
        scene.update(&mouse.resting(), &mut Queue::default());
    }
    panic!("animation never settled");
}

/// Draws the canvas scaled to fill the window.
fn present(texture: &Texture2D) {
    clear_background(BLACK);
    let (ox, oy, scale) = input::viewport();
    draw_texture_ex(
        texture,
        ox,
        oy,
        WHITE,
        DrawTextureParams {
            dest_size: Some(vec2(FB_W as f32 * scale, FB_H as f32 * scale)),
            ..Default::default()
        },
    );
}

/// Keeps the window the same shape as the canvas.
///
/// A window of any other shape can only be filled by padding it with black bars or by
/// stretching the picture out of square; snapping the window back means never having
/// to choose. Whichever side was dragged is the one kept, and the other follows it.
///
/// Only a size the desktop has actually reported as changed is corrected, so a window
/// manager that refuses the request is asked once and then left alone rather than
/// fought every frame.
#[cfg(not(target_arch = "wasm32"))]
fn keep_window_shape(last: &mut (f32, f32)) {
    const ASPECT: f32 = FB_W as f32 / FB_H as f32;
    let (sw, sh) = (screen_width(), screen_height());
    if sw < 1.0 || sh < 1.0 {
        return;
    }
    let (moved_w, moved_h) = ((sw - last.0).abs(), (sh - last.1).abs());
    *last = (sw, sh);
    if moved_w < 0.5 && moved_h < 0.5 {
        return;
    }
    let (want_w, want_h) = if moved_w >= moved_h {
        (sw, (sw / ASPECT).round())
    } else {
        ((sh * ASPECT).round(), sh)
    };
    if (want_w - sw).abs() >= 1.0 || (want_h - sh).abs() >= 1.0 {
        request_new_screen_size(want_w, want_h);
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use input::Dir;

    /// Sound turned off at the board stays off afterwards.
    ///
    /// The game is handed a copy of the settings when it starts and the application
    /// keeps the original, so a setting changed inside a game has to travel back or it
    /// is lost the moment the game ends. Driven through the ordinary input path, since
    /// that is the only way a player can get there.
    #[test]
    fn sound_turned_off_at_the_board_is_still_off_afterwards() {
        let mut app = App::new();
        let mut sfx = Queue::default();
        app.fade = Fade::None;
        app.scene = Scene::Game(Box::new(GameScene::new(app.config)));
        assert!(app.config.sound, "it starts on");

        // The board takes no orders until the opening parade has finished.
        for _ in 0..1200 {
            let Scene::Game(scene) = &app.scene else { panic!("the game screen has gone") };
            if !scene.animating() {
                break;
            }
            app.update(&Input::default(), &mut sfx);
        }

        let mut cancel = Input::default();
        cancel.cancel = true;
        app.update(&cancel, &mut sfx);

        // Two rows down from the one the menu opens on, and take it.
        let mut down = Input::default();
        down.fire(Dir::Down);
        app.update(&down, &mut sfx);
        app.update(&down, &mut sfx);
        let mut confirm = Input::default();
        confirm.confirm = true;
        app.update(&confirm, &mut sfx);

        assert!(!app.config.sound, "the board turned it off and the application kept it");
        assert!(
            !MenuScene::new(app.config).config.sound,
            "so the title screen it goes back to agrees"
        );
        sfx.drain();
    }

    /// Cancel leaves the game only where it has nothing else to do. Anywhere else it
    /// belongs to the screen in front, and quitting there would cost a live game.
    #[test]
    fn escape_is_only_the_way_out_from_the_bare_title() {
        let mut input = Input::default();
        input.cancel = true;
        let mut sfx = Queue::default();
        let mut app = App::new();
        app.fade = Fade::None;
        assert!(app.quit_requested(&input), "the title screen has nothing behind it");

        app.fade = Fade::In;
        assert!(!app.quit_requested(&input), "a screen still arriving eats the press");
        app.fade = Fade::None;

        if let Scene::Menu(menu) = &mut app.scene {
            menu.open_about(&mut sfx);
        }
        assert!(!app.quit_requested(&input), "with about open, cancel closes that");

        app.scene = Scene::Game(Box::new(GameScene::new(app.config)));
        assert!(!app.quit_requested(&input), "in a game, cancel opens the menu");

        input.cancel = false;
        app.scene = Scene::Menu(MenuScene::new(app.config));
        assert!(!app.quit_requested(&input), "and nothing leaves on its own");
        sfx.drain();
    }
}
