//! Saying that the engine is still on its way.
//!
//! Only the browser build has anything to say here. The interface module is a megabyte
//! and a half and is on screen at once; the weights are twenty-two more and arrive
//! behind it. The menu stays usable throughout — a player choosing a level and a clock
//! is doing something useful with that time — so this is a line along the bottom rather
//! than a screen of its own, and it disappears when there is nothing left to say.
//!
//! Drawn into the framebuffer like everything else, so it is the same pixels at the same
//! scale as the rest, and not an HTML overlay floating at some other size.

use super::fb::{Fb, Rgba, rgb};
use super::lang::{Key, Lang, t};
use super::{FB_H, FB_W, font};

/// Percent downloaded, or 100 once the engine can play. Always 100 where there is
/// nothing to download.
#[cfg(all(feature = "gui-core", not(feature = "gui")))]
pub fn progress() -> i32 {
    #[link(wasm_import_module = "env")]
    unsafe extern "C" {
        fn gaia_net_progress() -> i32;
    }
    unsafe { gaia_net_progress() }
}

#[cfg(feature = "gui")]
pub fn progress() -> i32 {
    100
}

/// Draws the progress line, if there is any progress left to report.
pub fn draw(fb: &mut Fb, text: Rgba, lang: Lang) {
    let percent = progress();
    if percent >= 100 {
        return;
    }

    // The strip is painted out first, not drawn over: the scene already has a line of
    // hints along the bottom, and two texts in the same place read as neither. The hints
    // come back when there is nothing left to load.
    const BAR_H: i32 = 3;
    const STRIP_H: i32 = 14;
    let top = FB_H as i32 - STRIP_H;
    fb.rectfill(0, top, FB_W as i32 - 1, FB_H as i32 - 1, rgb(0x1a1127));

    font::print_centered(fb, t(Key::Loading, lang), FB_W as i32 / 2, top + 1, text);

    let margin = 24;
    let width = FB_W as i32 - margin * 2;
    let filled = width * percent.clamp(0, 100) / 100;
    let y = FB_H as i32 - BAR_H - 3;
    // An empty trough is still a trough: the outline says how far there is to go.
    fb.rect(margin - 1, y - 1, margin + width, y + BAR_H, text);
    if filled > 0 {
        fb.rectfill(margin, y, margin + filled - 1, y + BAR_H - 1, text);
    }
}
