//! A 3x5 pixel font drawn into the software framebuffer.
//!
//! Glyphs are packed as bitmasks rather than an image so text costs nothing in the
//! binary and stays crisp: at this size an anti-aliased font would turn to mud.
//! Only uppercase forms exist; lowercase input maps onto them.

use super::fb::{Fb, Rgba};

/// Glyph cell size. Advance is one pixel wider to leave a gap between letters.
pub const GLYPH_W: i32 = 3;
pub const GLYPH_H: i32 = 5;
pub const ADVANCE: i32 = GLYPH_W + 1;

/// Packs five rows into one mask. Each row is written most-significant-bit-left, so
/// `0b101` reads as `#.#` in source.
const fn g(rows: [u8; 5]) -> u16 {
    (rows[0] as u16)
        | ((rows[1] as u16) << 3)
        | ((rows[2] as u16) << 6)
        | ((rows[3] as u16) << 9)
        | ((rows[4] as u16) << 12)
}

const SPACE: u16 = g([0b000, 0b000, 0b000, 0b000, 0b000]);

const LETTERS: [u16; 26] = [
    g([0b111, 0b101, 0b111, 0b101, 0b101]), // A
    g([0b110, 0b101, 0b110, 0b101, 0b110]), // B
    g([0b111, 0b100, 0b100, 0b100, 0b111]), // C
    g([0b110, 0b101, 0b101, 0b101, 0b110]), // D
    g([0b111, 0b100, 0b110, 0b100, 0b111]), // E
    g([0b111, 0b100, 0b110, 0b100, 0b100]), // F
    g([0b111, 0b100, 0b101, 0b101, 0b111]), // G
    g([0b101, 0b101, 0b111, 0b101, 0b101]), // H
    g([0b111, 0b010, 0b010, 0b010, 0b111]), // I
    g([0b001, 0b001, 0b001, 0b101, 0b111]), // J
    g([0b101, 0b101, 0b110, 0b101, 0b101]), // K
    g([0b100, 0b100, 0b100, 0b100, 0b111]), // L
    g([0b101, 0b111, 0b111, 0b101, 0b101]), // M
    g([0b110, 0b101, 0b101, 0b101, 0b101]), // N
    g([0b111, 0b101, 0b101, 0b101, 0b111]), // O
    g([0b111, 0b101, 0b111, 0b100, 0b100]), // P
    g([0b111, 0b101, 0b101, 0b111, 0b001]), // Q
    g([0b111, 0b101, 0b110, 0b101, 0b101]), // R
    g([0b111, 0b100, 0b111, 0b001, 0b111]), // S
    g([0b111, 0b010, 0b010, 0b010, 0b010]), // T
    g([0b101, 0b101, 0b101, 0b101, 0b111]), // U
    g([0b101, 0b101, 0b101, 0b101, 0b010]), // V
    g([0b101, 0b101, 0b111, 0b111, 0b101]), // W
    g([0b101, 0b101, 0b010, 0b101, 0b101]), // X
    g([0b101, 0b101, 0b010, 0b010, 0b010]), // Y
    g([0b111, 0b001, 0b010, 0b100, 0b111]), // Z
];

const DIGITS: [u16; 10] = [
    g([0b111, 0b101, 0b101, 0b101, 0b111]), // 0
    g([0b010, 0b110, 0b010, 0b010, 0b111]), // 1
    g([0b111, 0b001, 0b111, 0b100, 0b111]), // 2
    g([0b111, 0b001, 0b111, 0b001, 0b111]), // 3
    g([0b101, 0b101, 0b111, 0b001, 0b001]), // 4
    g([0b111, 0b100, 0b111, 0b001, 0b111]), // 5
    g([0b111, 0b100, 0b111, 0b101, 0b111]), // 6
    g([0b111, 0b001, 0b001, 0b001, 0b001]), // 7
    g([0b111, 0b101, 0b111, 0b101, 0b111]), // 8
    g([0b111, 0b101, 0b111, 0b001, 0b111]), // 9
];

/// Returns the bitmask for a character, falling back to blank for anything unmapped.
fn glyph(c: char) -> u16 {
    match c {
        'a'..='z' => LETTERS[c as usize - 'a' as usize],
        'A'..='Z' => LETTERS[c as usize - 'A' as usize],
        '0'..='9' => DIGITS[c as usize - '0' as usize],
        '.' => g([0b000, 0b000, 0b000, 0b000, 0b010]),
        ',' => g([0b000, 0b000, 0b000, 0b010, 0b100]),
        ':' => g([0b000, 0b010, 0b000, 0b010, 0b000]),
        ';' => g([0b000, 0b010, 0b000, 0b010, 0b100]),
        '!' => g([0b010, 0b010, 0b010, 0b000, 0b010]),
        '?' => g([0b111, 0b001, 0b011, 0b000, 0b010]),
        '-' => g([0b000, 0b000, 0b111, 0b000, 0b000]),
        '+' => g([0b000, 0b010, 0b111, 0b010, 0b000]),
        '*' => g([0b101, 0b010, 0b101, 0b000, 0b000]),
        '/' => g([0b001, 0b001, 0b010, 0b100, 0b100]),
        '=' => g([0b000, 0b111, 0b000, 0b111, 0b000]),
        '_' => g([0b000, 0b000, 0b000, 0b000, 0b111]),
        '#' => g([0b101, 0b111, 0b101, 0b111, 0b101]),
        '%' => g([0b101, 0b001, 0b010, 0b100, 0b101]),
        '\'' => g([0b010, 0b010, 0b000, 0b000, 0b000]),
        '"' => g([0b101, 0b101, 0b000, 0b000, 0b000]),
        '(' => g([0b001, 0b010, 0b010, 0b010, 0b001]),
        ')' => g([0b100, 0b010, 0b010, 0b010, 0b100]),
        '[' => g([0b011, 0b010, 0b010, 0b010, 0b011]),
        ']' => g([0b110, 0b010, 0b010, 0b010, 0b110]),
        '<' => g([0b001, 0b010, 0b100, 0b010, 0b001]),
        '>' => g([0b100, 0b010, 0b001, 0b010, 0b100]),
        // A ring with its opening at the foot, which is as much of an at sign as three
        // pixels will carry. Wanted for the author's address on the about screen.
        '@' => g([0b111, 0b101, 0b111, 0b100, 0b011]),
        _ => SPACE,
    }
}

/// True if the font can draw this character. A space counts: it is drawn as nothing on
/// purpose, unlike a character the table has never heard of, which prints as a silent
/// hole in the middle of a word. Used by the screens that carry fixed text, to check
/// that all of it can actually be shown.
#[cfg(test)]
pub fn has_glyph(c: char) -> bool {
    c == ' ' || glyph(c) != SPACE
}

/// Width in pixels of a rendered string, excluding the trailing letter gap.
pub fn width(text: &str) -> i32 {
    let n = text.chars().count() as i32;
    if n == 0 { 0 } else { n * ADVANCE - 1 }
}

/// Draws `text` with its top-left corner at `(x, y)`.
pub fn print(fb: &mut Fb, text: &str, x: i32, y: i32, color: Rgba) {
    for (i, c) in text.chars().enumerate() {
        let mask = glyph(c);
        if mask == SPACE {
            continue;
        }
        let gx = x + i as i32 * ADVANCE;
        for row in 0..GLYPH_H {
            let bits = (mask >> (row * 3)) & 0b111;
            for col in 0..GLYPH_W {
                if bits & (1 << (2 - col)) != 0 {
                    fb.pset(gx + col, y + row, color);
                }
            }
        }
    }
}

/// Draws `text` centred horizontally on `cx`.
pub fn print_centered(fb: &mut Fb, text: &str, cx: i32, y: i32, color: Rgba) {
    print(fb, text, cx - width(text) / 2, y, color);
}

/// Draws centred text over its own shadow, for a line that has to hold up against
/// whatever happens to be behind it. At this size a full outline would close the
/// counters of the letters, so the shadow is a single pixel down and to the right —
/// enough to give every stroke an edge of its own.
pub fn print_centered_shaded(fb: &mut Fb, text: &str, cx: i32, y: i32, color: Rgba, shade: Rgba) {
    let x = cx - width(text) / 2;
    print(fb, text, x + 1, y + 1, shade);
    print(fb, text, x, y, color);
}

/// Draws `text` with every pixel blown up into a square block. Chunky rather than
/// smooth, which is the only honest way to make this font bigger.
pub fn print_scaled(fb: &mut Fb, text: &str, x: i32, y: i32, scale: i32, color: Rgba) {
    debug_assert!(scale >= 1);
    for (i, c) in text.chars().enumerate() {
        let mask = glyph(c);
        if mask == SPACE {
            continue;
        }
        let gx = x + i as i32 * (GLYPH_W + 1) * scale;
        for row in 0..GLYPH_H {
            let bits = (mask >> (row * 3)) & 0b111;
            for col in 0..GLYPH_W {
                if bits & (1 << (2 - col)) != 0 {
                    let px = gx + col * scale;
                    let py = y + row * scale;
                    fb.rectfill(px, py, px + scale - 1, py + scale - 1, color);
                }
            }
        }
    }
}

/// Width of a string drawn by [`print_scaled`].
pub fn width_scaled(text: &str, scale: i32) -> i32 {
    let n = text.chars().count() as i32;
    if n == 0 {
        0
    } else {
        (n * (GLYPH_W + 1) - 1) * scale
    }
}

/// Draws `text` as one flat silhouette grown by `grow` pixels in every direction.
///
/// Ringing text by stamping it eight times at neighbouring offsets is the cheap way,
/// but the copies overlap: a translucent colour builds up wherever they do, and the
/// ring comes out solid whatever alpha it was given. Growing the shape and painting it
/// once keeps the alpha the caller asked for, so a banner can sit on the board without
/// blotting it out.
pub fn print_halo(fb: &mut Fb, text: &str, x: i32, y: i32, scale: i32, grow: i32, color: Rgba) {
    debug_assert!(scale >= 1 && grow >= 0);
    let (w, h) = (width_scaled(text, scale), GLYPH_H * scale);
    if w <= 0 {
        return;
    }
    let (mw, mh) = ((w + 2 * grow) as usize, (h + 2 * grow) as usize);
    let mut mask = vec![false; mw * mh];
    for (i, c) in text.chars().enumerate() {
        let bits = glyph(c);
        if bits == SPACE {
            continue;
        }
        let gx = i as i32 * ADVANCE * scale;
        for row in 0..GLYPH_H {
            for col in 0..GLYPH_W {
                if (bits >> (row * 3)) & (1 << (2 - col)) == 0 {
                    continue;
                }
                for py in 0..scale {
                    let my = (grow + row * scale + py) as usize;
                    for px in 0..scale {
                        mask[my * mw + (grow + gx + col * scale + px) as usize] = true;
                    }
                }
            }
        }
    }
    for _ in 0..grow {
        mask = grown(&mask, mw, mh);
    }
    for my in 0..mh {
        for mx in 0..mw {
            if mask[my * mw + mx] {
                let (px, py) = (x - grow + mx as i32, y - grow + my as i32);
                fb.pset(px, py, color);
            }
        }
    }
}

/// The mask with every set pixel's eight neighbours set too.
fn grown(mask: &[bool], w: usize, h: usize) -> Vec<bool> {
    let mut out = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            if !mask[y * w + x] {
                continue;
            }
            for ny in y.saturating_sub(1)..(y + 2).min(h) {
                for nx in x.saturating_sub(1)..(x + 2).min(w) {
                    out[ny * w + nx] = true;
                }
            }
        }
    }
    out
}

/// Draws scaled `text` centred on `cx` as if the letters were solid blocks standing
/// off the screen.
///
/// Three passes, in this order: a dark ring so the block reads against any
/// background, then a side stepped one pixel at a time down and to the right, then
/// the face. Stepping the side by whole blocks would leave it in stripes, and drawing
/// the ring last would paint over it.
#[allow(clippy::too_many_arguments)]
pub fn print_block(
    fb: &mut Fb,
    text: &str,
    cx: i32,
    y: i32,
    scale: i32,
    depth: i32,
    face: Rgba,
    side: Rgba,
    edge: Rgba,
) {
    debug_assert!(scale >= 1 && depth >= 0);
    let x = cx - width_scaled(text, scale) / 2;
    print_halo(fb, text, x, y, scale, scale, edge);
    for step in (1..=depth).rev() {
        print_scaled(fb, text, x + step, y + step, scale, side);
    }
    print_scaled(fb, text, x, y, scale, face);
}

/// Draws scaled `text` centred horizontally on `cx`, ringed by a one-pixel-per-block
/// outline so it reads over the busiest background.
pub fn print_title(fb: &mut Fb, text: &str, cx: i32, y: i32, scale: i32, color: Rgba, border: Rgba) {
    let x = cx - width_scaled(text, scale) / 2;
    for (dx, dy) in OUTLINE {
        print_scaled(fb, text, x + dx * scale, y + dy * scale, scale, border);
    }
    print_scaled(fb, text, x, y, scale, color);
}

/// The eight neighbours, used to ring text so it reads over any background.
const OUTLINE: [(i32, i32); 8] = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (1, -1),
    (-1, 1),
    (1, 1),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_halo_is_composited_once_however_much_it_overlaps() {
        use super::super::fb::{FB_H, FB_W, rgb, rgba};

        const ALPHA: u32 = 110;
        let mut fb = Fb::new();
        fb.clear(rgb(0xffffff));
        // Two adjacent letters: their halos overlap in the gap between them, which is
        // exactly where stamping the shape eight times over would go solid.
        print_halo(&mut fb, "mm", 6, 6, 3, 3, rgba(0x000000, ALPHA));
        let mut px = vec![0u8; FB_W * FB_H * 4];
        fb.copy_to(&mut px);

        let once = (255 * (255 - ALPHA) / 255) as u8;
        let darkest = px.chunks_exact(4).map(|p| p[0]).min().unwrap();
        assert_eq!(
            darkest, once,
            "the wash was laid down more than once, so it is darker than asked for"
        );
        assert!(
            px.chunks_exact(4).any(|p| p[0] == once),
            "the halo drew nothing at all"
        );
    }

    #[test]
    fn every_glyph_fits_the_cell() {
        for mask in LETTERS.iter().chain(DIGITS.iter()) {
            assert_eq!(mask & !0x7fff, 0, "glyph overflows its 3x5 cell");
        }
    }

    #[test]
    fn glyphs_are_distinct() {
        for (i, a) in LETTERS.iter().enumerate() {
            for (j, b) in LETTERS.iter().enumerate() {
                assert!(i == j || a != b, "letters {i} and {j} render identically");
            }
        }
        for (i, a) in DIGITS.iter().enumerate() {
            for (j, b) in DIGITS.iter().enumerate() {
                assert!(i == j || a != b, "digits {i} and {j} render identically");
            }
        }
    }

    #[test]
    fn width_matches_advance() {
        assert_eq!(width(""), 0);
        assert_eq!(width("A"), GLYPH_W);
        assert_eq!(width("AB"), GLYPH_W * 2 + 1);
    }

    #[test]
    fn case_is_folded() {
        assert_eq!(glyph('a'), glyph('A'));
    }
}
