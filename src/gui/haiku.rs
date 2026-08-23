//! The Haiku front-end: the same interface over a native shim instead of macroquad,
//! which has no backend for this platform (and Haiku's Xlib layer stops short of GLX,
//! so the Linux path cannot stand in). Everything drawn goes through the CPU
//! framebuffer already, so the shim only has to put a window up, blit into it, hand
//! back input, and mix the clips — a few hundred lines of Be API C++ in
//! `haiku_shim.cpp`, compiled by build.rs and spoken to over the C ABI below.

use std::time::{Duration, Instant};

use super::assets::Assets;
use super::audio::{Audio, Queue};
use super::fb::{FB_H, FB_W, Fb};
use super::input::Input;
use super::scheme::SCHEMES;
use super::{App, INITIAL_SCALE, MAX_STEPS, STEP, loading, scheme};

/// Key bit indices shared with the shim, which reports the keyboard as two masks:
/// keys down now, and pressed edges latched since the last poll. The C++ side keeps
/// its own copy of this numbering — change one and the other must follow.
pub mod key {
    pub const LEFT: u32 = 0;
    pub const RIGHT: u32 = 1;
    pub const UP: u32 = 2;
    pub const DOWN: u32 = 3;
    pub const A: u32 = 4;
    pub const D: u32 = 5;
    pub const W: u32 = 6;
    pub const S: u32 = 7;
    pub const X: u32 = 8;
    pub const V: u32 = 9;
    pub const ENTER: u32 = 10;
    pub const SPACE: u32 = 11;
    pub const Z: u32 = 12;
    pub const C: u32 = 13;
    pub const ESCAPE: u32 = 14;
    pub const BACKSPACE: u32 = 15;
    pub const F: u32 = 16;
    pub const TAB: u32 = 17;
    pub const R: u32 = 18;
    pub const U: u32 = 19;
    pub const N: u32 = 20;
    pub const M: u32 = 21;
    /// First of eight consecutive digit bits, 1 through 8.
    pub const DIGIT_1: u32 = 22;
}

/// One frame's look at the window, filled in by the shim.
#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct Snapshot {
    /// Where the mouse is, in view pixels, whether or not it is inside.
    pub mouse_x: f32,
    pub mouse_y: f32,
    /// Keys held right now, by the bits of [`key`].
    pub keys_down: u32,
    /// Pressed edges since the last poll, latched so a tap shorter than a frame
    /// still lands. Cleared by the read.
    pub keys_pressed: u32,
    /// Bit 0 is the primary button, held right now.
    pub buttons: u8,
    /// The primary button went down since the last poll. Cleared by the read.
    pub pressed: u8,
    /// The pointer is over the view itself, not the tab or the frame.
    pub inside: u8,
    pub _pad: u8,
}

unsafe extern "C" {
    fn gaia_shim_display_available() -> i32;
    /// Opens the window at `scale` times the canvas — or the largest whole
    /// multiple that fits the screen, when that is smaller.
    fn gaia_shim_init(title: *const std::ffi::c_char, fb_w: i32, fb_h: i32, scale: i32) -> i32;
    fn gaia_shim_frame(rgba: *const u8, w: i32, h: i32);
    fn gaia_shim_input(out: *mut Snapshot);
    fn gaia_shim_view_size(w: *mut f32, h: *mut f32);
    fn gaia_shim_show_cursor(show: i32);
    fn gaia_shim_should_quit() -> i32;
    fn gaia_shim_quit();
}

/// Whether app_server can be reached at all. Checked before any window is tried:
/// an engine running a match over SSH must stay an engine.
pub fn display_available() -> bool {
    unsafe { gaia_shim_display_available() != 0 }
}

/// The shim's answer to `sample()`: everything the window saw since last time.
pub fn input_snapshot() -> Snapshot {
    let mut snap = Snapshot::default();
    unsafe { gaia_shim_input(&mut snap) };
    snap
}

/// The view's size in pixels, for the viewport arithmetic.
pub fn view_size() -> (f32, f32) {
    let (mut w, mut h) = (0.0f32, 0.0f32);
    unsafe { gaia_shim_view_size(&mut w, &mut h) };
    (w, h)
}

/// Opens the window and runs the interface until it is closed. The same loop as
/// `amain`, paced by the clock rather than a swap chain: logic at a fixed 60 Hz,
/// one blit per frame, sounds drained once a frame.
pub fn run_window() {
    let title = c"GaiaChess";
    let ok = unsafe {
        gaia_shim_init(title.as_ptr(), FB_W as i32, FB_H as i32, INITIAL_SCALE)
    };
    if ok == 0 {
        eprintln!("no window: app_server refused us");
        return;
    }

    let assets = Assets::load();
    let audio = Audio::load();
    let mut app = App::new();
    let mut input = Input::default();
    let mut sfx = Queue::default();
    let mut fb = Fb::new();
    let mut frame = vec![0u8; FB_W * FB_H * 4];

    /// The pace the picture is shown at. The logic's own 60 Hz step is measured off
    /// the clock separately, so a late frame costs smoothness, never game speed.
    const FRAME: Duration = Duration::from_micros(16_667);

    let mut acc = 0.0f32;
    let mut last = Instant::now();
    'window: loop {
        let frame_start = Instant::now();
        if unsafe { gaia_shim_should_quit() } != 0 {
            break;
        }
        input.sample();
        // The hand is the cursor: the system one is hidden over the picture and handed
        // straight back the moment the pointer leaves it, so the desktop stays usable.
        unsafe { gaia_shim_show_cursor(!input.on_canvas() as i32) };

        let now = Instant::now();
        acc += (now - last).as_secs_f32();
        last = now;
        let mut steps = 0;
        while acc >= STEP && steps < MAX_STEPS {
            acc -= STEP;
            steps += 1;
            if input.next_scheme {
                app.config.scheme = (app.config.scheme + 1) % SCHEMES.len();
            }
            if app.quit_requested(&input) {
                break 'window;
            }
            app.update(&input, &mut sfx);
            input.consume();
        }
        // Silence drops the requests rather than holding them, as on every desktop.
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

        fb.copy_to(&mut frame);
        unsafe { gaia_shim_frame(frame.as_ptr(), FB_W as i32, FB_H as i32) };

        let elapsed = frame_start.elapsed();
        if elapsed < FRAME {
            std::thread::sleep(FRAME - elapsed);
        }
    }
    unsafe { gaia_shim_quit() };
}
