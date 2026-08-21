//! Colour schemes.
//!
//! The piece artwork is used exactly as drawn, so a scheme picks which of the sheet's
//! four colour variants plays each side and then dresses the board, panels and text
//! to match them. Every piece carries a white outer outline, which is what lets these
//! mid-tone boards stay readable under both light and dark sides.

use super::fb::{Rgba, rgb};
use super::lang::Key;

pub struct Scheme {
    /// What the menu calls it, in whichever language the menu is in.
    pub name: Key,
    /// Behind everything.
    pub bg: Rgba,
    pub tile_light: Rgba,
    pub tile_dark: Rgba,
    /// Edge under the board, giving it a little thickness.
    pub board_edge: Rgba,
    /// Cursor, arrows and anything asking for attention.
    pub accent: Rgba,
    /// Menu and dialog fill, plus its darker rim.
    pub panel: Rgba,
    pub panel_edge: Rgba,
    pub text: Rgba,
    /// The same voice said quieter: hints and figures read once and then ignored.
    /// Dimmed from `text` towards `bg`, not towards `panel` — these labels sit on the
    /// bare background with nothing behind them to lift them off it.
    pub text_dim: Rgba,
    /// Sheet variant painting each side.
    pub white_variant: u16,
    pub black_variant: u16,
    /// Opacity of the ground shadow, out of 255.
    pub shadow_alpha: u32,
}

pub const SCHEMES: [Scheme; 2] = [
    // Silver against navy on cool blue-greys, the closest match to how the piece
    // sheet was originally presented.
    Scheme {
        name: Key::SchemeSlate,
        bg: rgb(0x1a1127),
        tile_light: rgb(0x7c90a0),
        tile_dark: rgb(0x4f5478),
        board_edge: rgb(0x281041),
        accent: rgb(0xe97647),
        panel: rgb(0x3b2958),
        panel_edge: rgb(0x1a1127),
        text: rgb(0xcbdfe9),
        text_dim: rgb(0x8494a6),
        white_variant: 0,
        black_variant: 1,
        shadow_alpha: 70,
    },
    // Gold against dark slate on warm browns; the blue accent is the odd one out on
    // purpose, so the cursor never disappears into the board.
    Scheme {
        name: Key::SchemeEmber,
        bg: rgb(0x2b1b22),
        tile_light: rgb(0xa98c74),
        tile_dark: rgb(0x63483f),
        board_edge: rgb(0x1d1218),
        accent: rgb(0x6e9cd5),
        panel: rgb(0x4a3229),
        panel_edge: rgb(0x1d1218),
        text: rgb(0xffe37d),
        text_dim: rgb(0xb39457),
        white_variant: 2,
        black_variant: 3,
        shadow_alpha: 80,
    },
];

impl Scheme {
    /// The sheet variant for a side.
    pub fn variant(&self, color: crate::types::Color) -> u16 {
        match color {
            crate::types::Color::White => self.white_variant,
            crate::types::Color::Black => self.black_variant,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schemes_use_distinct_variants_per_side() {
        for s in &SCHEMES {
            assert_ne!(s.white_variant, s.black_variant, "{:?} reuses a variant", s.name);
            assert!(s.white_variant < super::super::assets::VARIANTS);
            assert!(s.black_variant < super::super::assets::VARIANTS);
        }
    }

    /// WCAG relative luminance, the yardstick for whether one colour reads on another.
    fn luminance(c: Rgba) -> f32 {
        let channel = |shift: u32| {
            let v = ((c >> shift) & 0xff) as f32 / 255.0;
            if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
        };
        0.2126 * channel(0) + 0.7152 * channel(8) + 0.0722 * channel(16)
    }

    fn contrast(ink: Rgba, ground: Rgba) -> f32 {
        let (a, b) = (luminance(ink), luminance(ground));
        (a.max(b) + 0.05) / (a.min(b) + 0.05)
    }

    #[test]
    fn every_ink_reads_on_the_ground_it_is_printed_on() {
        // 4.5:1 is the usual floor for body text, and the quiet ink is held to it too:
        // quieter has to mean less loud, not less legible. The band and the strip under
        // the board are the two grounds the HUD writes on.
        for s in &SCHEMES {
            for (name, ink) in [("text", s.text), ("text_dim", s.text_dim), ("accent", s.accent)] {
                for (spot, ground) in [("bg", s.bg), ("panel_edge", s.panel_edge)] {
                    let ratio = contrast(ink, ground);
                    assert!(ratio >= 4.5, "{:?}: {name} on {spot} is only {ratio:.1}:1", s.name);
                }
            }
            assert!(contrast(s.text, s.panel) >= 4.5, "{:?}: text on panel", s.name);
            // And it does have to be the quieter of the two, or the pair says nothing.
            assert!(
                contrast(s.text_dim, s.bg) < contrast(s.text, s.bg),
                "{:?}: text_dim is not quieter than text",
                s.name
            );
        }
    }

    #[test]
    fn tiles_differ_from_each_other_and_the_background() {
        for s in &SCHEMES {
            assert_ne!(s.tile_light, s.tile_dark, "{:?}", s.name);
            assert_ne!(s.tile_light, s.bg, "{:?}", s.name);
            assert_ne!(s.tile_dark, s.bg, "{:?}", s.name);
        }
    }
}
