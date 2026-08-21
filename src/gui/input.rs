//! One frame's worth of input, sampled from the platform and handed to the logic step.
//!
//! Input is read once per displayed frame but consumed by a fixed 60 Hz logic step, so
//! presses are latched here rather than polled twice. Held directions auto-repeat the
//! way a game pad does, which is what makes driving the cursor from the keyboard
//! pleasant rather than a chore.
//!
//! Keyboard and mouse are never both in charge. Whichever the player touched last owns
//! the hand: moving the mouse hands it over, and the next keypress takes it back. A
//! player who never touches the mouse therefore never sees the interface change under
//! them.

use macroquad::prelude::*;

use super::fb::{FB_H, FB_W};

/// Frames a direction must be held before it starts repeating.
const REPEAT_DELAY: u32 = 16;
/// Frames between repeats once it has started.
const REPEAT_RATE: u32 = 4;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dir {
    Left,
    Right,
    Up,
    Down,
}

impl Dir {
    pub const ALL: [Dir; 4] = [Dir::Left, Dir::Right, Dir::Up, Dir::Down];

    /// Step on the board, in files and ranks as seen on screen.
    pub fn delta(self) -> (i32, i32) {
        match self {
            Dir::Left => (-1, 0),
            Dir::Right => (1, 0),
            Dir::Up => (0, 1),
            Dir::Down => (0, -1),
        }
    }
}

#[derive(Default)]
pub struct Input {
    /// Frames each direction has been held, 0 when released.
    down: [u32; 4],
    /// Directions that should act this step, press or repeat.
    fired: [bool; 4],
    pub confirm: bool,
    pub cancel: bool,
    pub flip: bool,
    pub next_scheme: bool,
    pub restart: bool,
    pub take_back: bool,
    /// Difficulty picked with a number key. Only reaches the first eight rungs of
    /// the ladder; the menu is where the rest live.
    pub level: Option<u8>,
    pub toggle_white: bool,
    pub toggle_black: bool,
    /// Where the mouse is on the canvas, `None` while it is outside it.
    mouse: Option<(i32, i32)>,
    /// Last window position seen, for spotting that the mouse has moved at all.
    last_mouse: (f32, f32),
    /// True once the mouse has moved, until a keypress takes the hand back.
    driving: bool,
    /// True for as long as the left button is held down.
    pub held: bool,
    /// The left button went down on this step.
    pub press: bool,
}

impl Input {
    /// Samples the platform. Called once per displayed frame.
    pub fn sample(&mut self) {
        let mut typed = false;
        for (i, dir) in Dir::ALL.iter().enumerate() {
            let down = match dir {
                Dir::Left => is_key_down(KeyCode::Left) || is_key_down(KeyCode::A),
                Dir::Right => is_key_down(KeyCode::Right) || is_key_down(KeyCode::D),
                Dir::Up => is_key_down(KeyCode::Up) || is_key_down(KeyCode::W),
                Dir::Down => is_key_down(KeyCode::Down) || is_key_down(KeyCode::S),
            };
            if !down {
                self.down[i] = 0;
                continue;
            }
            let n = self.down[i];
            self.down[i] = n + 1;
            let fires = n == 0
                || (n >= REPEAT_DELAY && (n - REPEAT_DELAY).is_multiple_of(REPEAT_RATE));
            self.fired[i] |= fires;
            typed |= fires;
        }

        // Enter and escape are the pair the on-screen hints name: they are the two every
        // player already knows, and the only two that mean the same thing on every
        // keyboard. The window layer reports letters by physical position on a us board,
        // so the z/x a fantasy console would use sits under the keys marked w and x on an
        // azerty one. The letters stay bound for the hands that reach for them; they are
        // simply not what the screen advertises.
        self.confirm |= is_key_pressed(KeyCode::X)
            || is_key_pressed(KeyCode::V)
            || is_key_pressed(KeyCode::Enter)
            || is_key_pressed(KeyCode::Space);
        self.cancel |= is_key_pressed(KeyCode::Z)
            || is_key_pressed(KeyCode::C)
            || is_key_pressed(KeyCode::Escape)
            || is_key_pressed(KeyCode::Backspace);
        self.flip |= is_key_pressed(KeyCode::F);
        self.next_scheme |= is_key_pressed(KeyCode::Tab);
        self.restart |= is_key_pressed(KeyCode::R);
        self.take_back |= is_key_pressed(KeyCode::U);
        self.toggle_white |= is_key_pressed(KeyCode::N);
        self.toggle_black |= is_key_pressed(KeyCode::M);
        for (i, key) in [
            KeyCode::Key1,
            KeyCode::Key2,
            KeyCode::Key3,
            KeyCode::Key4,
            KeyCode::Key5,
            KeyCode::Key6,
            KeyCode::Key7,
            KeyCode::Key8,
        ]
        .into_iter()
        .enumerate()
        {
            if is_key_pressed(key) {
                self.level = Some(i as u8 + 1);
            }
        }
        typed |= self.confirm || self.cancel;

        // Only real movement hands the interface over. A mouse sitting untouched on the
        // desk leaves the keyboard in charge however long the game lasts.
        let (mx, my) = mouse_position();
        if (mx - self.last_mouse.0).abs() > 0.5 || (my - self.last_mouse.1).abs() > 0.5 {
            self.last_mouse = (mx, my);
            self.driving = true;
        }
        // Not `canvas_pixel` alone: a window's frame is not the window, and the platform
        // may go on reporting the last position inside it. See `pointer_inside_window`.
        self.mouse = if pointer_inside_window() { canvas_pixel(mx, my) } else { None };
        self.held = is_mouse_button_down(MouseButton::Left);
        self.press |= is_mouse_button_pressed(MouseButton::Left);
        // Pressing a key takes the hand back, so a player who reaches for the keyboard
        // is not left with a pointer they are no longer touching.
        if typed {
            self.driving = false;
        }
    }

    /// Canvas pixel the hand should follow, or `None` while the keyboard is driving or
    /// the mouse has left the picture.
    pub fn pointer(&self) -> Option<(i32, i32)> {
        if self.driving { self.mouse } else { None }
    }

    /// True while the mouse is over the picture, whichever device is driving. This is
    /// what decides whether the system cursor is hidden.
    pub fn on_canvas(&self) -> bool {
        self.mouse.is_some()
    }

    /// Puts the mouse somewhere by hand, for a capture or a test.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn point_at(&mut self, x: i32, y: i32) {
        self.mouse = Some((x, y));
        self.driving = true;
    }

    /// True once per press or auto-repeat of a direction.
    pub fn dir(&self, dir: Dir) -> bool {
        self.fired[dir as usize]
    }

    /// Raises a direction by hand, for driving a scene without a keyboard.
    #[cfg(test)]
    pub fn fire(&mut self, dir: Dir) {
        self.fired[dir as usize] = true;
    }

    /// Clears the latched events, leaving alone what is a state rather than an event:
    /// the held-key timers, where the mouse is, whether its button is down, and which
    /// device is driving.
    pub fn consume(&mut self) {
        self.fired = [false; 4];
        self.confirm = false;
        self.cancel = false;
        self.flip = false;
        self.next_scheme = false;
        self.restart = false;
        self.take_back = false;
        self.level = None;
        self.toggle_white = false;
        self.toggle_black = false;
        self.press = false;
    }
}

/// Where the canvas sits in the window: top-left corner and scale.
///
/// The scale is deliberately fractional so the picture grows with the window instead
/// of stepping between whole multiples, and it is the same on both axes so the pixels
/// stay square. The window is kept to the canvas's shape elsewhere, so the offsets are
/// normally zero; they only become non-zero if the desktop hands back a window of some
/// other shape, and fitting inside it is still better than stretching the picture.
pub fn viewport() -> (f32, f32, f32) {
    let (sw, sh) = (screen_width(), screen_height());
    let mut scale = (sw / FB_W as f32).min(sh / FB_H as f32).max(0.01);
    // Once the picture no longer fills its frame — full screen, or a browser canvas
    // whose shape is the page's to choose — a fractional scale starts drawing some
    // artwork pixels five screen pixels wide and others six, and the whole thing
    // shimmers. Falling back to the whole multiple below costs a few pixels of border
    // and keeps every pixel the same size. The desktop window is held to the canvas's
    // shape elsewhere, so it never gets here.
    let fits = (sw / FB_W as f32 - sh / FB_H as f32).abs() < 0.01;
    if !fits && scale >= 2.0 {
        scale = scale.floor();
    }
    (
        ((sw - FB_W as f32 * scale) * 0.5).floor(),
        ((sh - FB_H as f32 * scale) * 0.5).floor(),
        scale,
    )
}

/// Maps a window pixel back onto the canvas. Returns `None` outside it, so a click
/// there is ignored rather than clamped onto an edge square nobody aimed at.
pub fn canvas_pixel(mx: f32, my: f32) -> Option<(i32, i32)> {
    let (ox, oy, scale) = viewport();
    let x = ((mx - ox) / scale).floor() as i32;
    let y = ((my - oy) / scale).floor() as i32;
    if (0..FB_W as i32).contains(&x) && (0..FB_H as i32).contains(&y) {
        Some((x, y))
    } else {
        None
    }
}

/// True while the pointer really is inside the picture — the window's client area,
/// title bar and borders excluded.
///
/// Windows only reports mouse movement over the client area: the frame belongs to the
/// desktop, so `mouse_position()` keeps whatever it last saw inside the window and the
/// interface would go on believing the pointer is on the board while it is on the title
/// bar — system cursor hidden, hand frozen where the pointer left. And hiding it there
/// is not a detail: that is the bar the player grabs to move or close the window.
/// So the desktop is asked directly, four calls against a frozen ABI rather than a
/// dependency. On X11 the cursor is hidden per window and the frame is a window of the
/// window manager's, which keeps its own arrow with no help from us.
#[cfg(windows)]
fn pointer_inside_window() -> bool {
    type Window = *mut std::ffi::c_void;

    #[repr(C)]
    struct Point {
        x: i32,
        y: i32,
    }
    /// A client rectangle: `left` and `top` are always zero, the other two are its size.
    #[repr(C)]
    struct Rect {
        left: i32,
        top: i32,
        right: i32,
        bottom: i32,
    }

    #[link(name = "user32")]
    unsafe extern "system" {
        fn GetActiveWindow() -> Window;
        fn GetCursorPos(point: *mut Point) -> i32;
        fn GetClientRect(window: Window, rect: *mut Rect) -> i32;
        fn ClientToScreen(window: Window, point: *mut Point) -> i32;
    }

    let mut at = Point { x: 0, y: 0 };
    let mut client = Rect { left: 0, top: 0, right: 0, bottom: 0 };
    // The client area's own corner, which is what turns its size into screen coordinates.
    let mut corner = Point { x: 0, y: 0 };
    unsafe {
        let window = GetActiveWindow();
        // Nothing of ours has the focus. A background application has no business
        // hiding the cursor, wherever it happens to be.
        if window.is_null() {
            return false;
        }
        // Should the desktop refuse to answer, assume the pointer is where the rest of
        // the interface already thinks it is rather than making it jump.
        if GetCursorPos(&mut at) == 0
            || GetClientRect(window, &mut client) == 0
            || ClientToScreen(window, &mut corner) == 0
        {
            return true;
        }
    }
    (corner.x..corner.x + client.right).contains(&at.x)
        && (corner.y..corner.y + client.bottom).contains(&at.y)
}

#[cfg(not(windows))]
fn pointer_inside_window() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_keyboard_holds_the_hand_until_the_mouse_moves() {
        let mut input = Input::default();
        assert_eq!(input.pointer(), None, "an untouched mouse drives nothing");
        input.point_at(10, 20);
        assert_eq!(input.pointer(), Some((10, 20)));
    }

    #[test]
    fn consuming_a_step_keeps_the_mouse_but_drops_its_click() {
        let mut input = Input::default();
        input.point_at(4, 5);
        input.press = true;
        input.held = true;
        input.consume();
        assert_eq!(input.pointer(), Some((4, 5)), "the mouse has not gone anywhere");
        assert!(input.held, "nor has the button come up");
        assert!(!input.press, "but pressing it is an event, not a state");
    }
}
