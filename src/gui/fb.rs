//! Software framebuffer: a small fixed-size RGBA canvas plus its pixel blitter.
//!
//! Scenes draw here at logical resolution; the result is uploaded once per frame to a
//! nearest-neighbour texture and letterboxed into the window. Drawing at the small
//! size and scaling by a whole number afterwards is what keeps every pixel square.

/// Packed pixel. Little-endian bytes are R, G, B, A, which is the upload format.
pub type Rgba = u32;

/// Builds an opaque pixel from a `0xRRGGBB` literal.
pub const fn rgb(hex: u32) -> Rgba {
    0xff00_0000 | ((hex & 0xff) << 16) | (hex & 0xff00) | ((hex >> 16) & 0xff)
}

/// Builds a translucent pixel from a `0xRRGGBB` literal and an alpha out of 255.
pub const fn rgba(hex: u32, alpha: u32) -> Rgba {
    (rgb(hex) & 0x00ff_ffff) | ((alpha & 0xff) << 24)
}

/// The same colour at a different alpha, for using a scheme's own colour as a wash.
pub const fn with_alpha(color: Rgba, alpha: u32) -> Rgba {
    (color & 0x00ff_ffff) | ((alpha & 0xff) << 24)
}

/// Logical canvas size: eight 20x21 squares plus a status band and the room the tall
/// pieces need to overhang the rank behind them.
pub const FB_W: usize = 176;
pub const FB_H: usize = 210;

/// A decoded RGBA sprite sheet held in main memory.
pub struct Atlas {
    px: Vec<Rgba>,
    w: usize,
    h: usize,
}

impl Atlas {
    /// Wraps raw RGBA bytes as produced by a PNG decoder.
    pub fn from_rgba(bytes: &[u8], w: usize, h: usize) -> Atlas {
        debug_assert_eq!(bytes.len(), w * h * 4);
        let px = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Atlas { px, w, h }
    }

    /// Whether the sheet paints anything at all here, for measuring artwork against the
    /// layout that has to make room for it. Only the layout tests ask.
    #[cfg(test)]
    pub fn opaque_at(&self, x: i32, y: i32) -> bool {
        if x < 0 || y < 0 || x as usize >= self.w || y as usize >= self.h {
            return false;
        }
        self.px[y as usize * self.w + x as usize] >> 24 != 0
    }
}

/// A rectangular region of an [`Atlas`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sprite {
    pub x: u16,
    pub y: u16,
    pub w: u16,
    pub h: u16,
}

/// Alpha-blends `src` over `dst`, with `a` the source alpha in `0..=255`.
fn blend(src: Rgba, dst: Rgba, a: u32) -> Rgba {
    debug_assert!(a <= 255);
    let inv = 255 - a;
    let mut out = 0xff00_0000;
    for shift in [0, 8, 16] {
        let s = (src >> shift) & 0xff;
        let d = (dst >> shift) & 0xff;
        out |= (((s * a + d * inv) / 255) & 0xff) << shift;
    }
    out
}

/// A fixed-size RGBA canvas with clipping, a drawing offset and screen-wide effects.
pub struct Fb {
    px: Vec<Rgba>,
    /// Added to every coordinate before drawing; carries the screen shake.
    cam: (i32, i32),
    /// Clip rectangle `(x0, y0, x1, y1)`, exclusive on the high corner.
    clip: (i32, i32, i32, i32),
    /// Fade towards black. 0.0 leaves the picture untouched, 1.0 is fully black.
    pub fade: f32,
    /// Flash towards white. 0.0 leaves the picture untouched, 1.0 is fully white.
    pub flash: f32,
}

impl Fb {
    pub fn new() -> Fb {
        Fb {
            px: vec![0xff00_0000; FB_W * FB_H],
            cam: (0, 0),
            clip: (0, 0, FB_W as i32, FB_H as i32),
            fade: 0.0,
            flash: 0.0,
        }
    }

    /// Fills the whole canvas, ignoring the clip rectangle and the camera.
    pub fn clear(&mut self, color: Rgba) {
        self.px.fill(color);
    }

    /// Sets the drawing offset applied to every subsequent coordinate.
    pub fn camera(&mut self, dx: i32, dy: i32) {
        self.cam = (dx, dy);
    }

    /// Restricts drawing to a rectangle, in canvas coordinates (the camera does not
    /// move it). Returns the previous rectangle so callers can restore it.
    pub fn clip(&mut self, x: i32, y: i32, w: i32, h: i32) -> (i32, i32, i32, i32) {
        debug_assert!(w >= 0 && h >= 0);
        let prev = self.clip;
        self.clip = (
            x.max(0),
            y.max(0),
            (x + w).min(FB_W as i32),
            (y + h).min(FB_H as i32),
        );
        prev
    }

    /// Restores a rectangle previously returned by [`Fb::clip`].
    pub fn set_clip(&mut self, rect: (i32, i32, i32, i32)) {
        self.clip = rect;
    }

    /// Writes one pixel, honouring the camera and the clip rectangle. A colour
    /// carrying alpha is composited rather than stamped.
    pub fn pset(&mut self, x: i32, y: i32, color: Rgba) {
        let (sx, sy) = (x + self.cam.0, y + self.cam.1);
        let (x0, y0, x1, y1) = self.clip;
        if sx >= x0 && sx < x1 && sy >= y0 && sy < y1 {
            let dst = &mut self.px[sy as usize * FB_W + sx as usize];
            let a = color >> 24;
            *dst = match a {
                0 => *dst,
                255 => color,
                _ => blend(color, *dst, a),
            };
        }
    }

    /// Fills a rectangle spanning `x0..=x1` by `y0..=y1`, inclusive on both corners.
    /// A colour carrying alpha is composited, which is how the dimming veils and the
    /// drop shadows under panels are drawn.
    pub fn rectfill(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, color: Rgba) {
        let (cx, cy) = self.cam;
        let (lo_x, lo_y, hi_x, hi_y) = self.clip;
        let sx0 = (x0.min(x1) + cx).max(lo_x);
        let sx1 = (x0.max(x1) + cx + 1).min(hi_x);
        let sy0 = (y0.min(y1) + cy).max(lo_y);
        let sy1 = (y0.max(y1) + cy + 1).min(hi_y);
        if sx0 >= sx1 || sy0 >= sy1 {
            return;
        }
        let a = color >> 24;
        if a == 0 {
            return;
        }
        for row in sy0..sy1 {
            let base = row as usize * FB_W;
            let span = &mut self.px[base + sx0 as usize..base + sx1 as usize];
            if a == 255 {
                span.fill(color);
            } else {
                for px in span {
                    *px = blend(color, *px, a);
                }
            }
        }
    }

    /// Draws a one-pixel rectangle outline, inclusive on both corners.
    pub fn rect(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, color: Rgba) {
        self.rectfill(x0, y0, x1, y0, color);
        self.rectfill(x0, y1, x1, y1, color);
        self.rectfill(x0, y0, x0, y1, color);
        self.rectfill(x1, y0, x1, y1, color);
    }

    /// Fills a convex polygon. Used for the move arrow, whose shape has to stretch
    /// between two arbitrary squares and so cannot be a sprite.
    pub fn fill_poly(&mut self, pts: &[(f32, f32)], color: Rgba) {
        debug_assert!(pts.len() >= 3);
        let top = pts.iter().map(|p| p.1).fold(f32::MAX, f32::min).floor() as i32;
        let bottom = pts.iter().map(|p| p.1).fold(f32::MIN, f32::max).ceil() as i32;
        for y in top..=bottom {
            let scan = y as f32 + 0.5;
            let (mut lo, mut hi) = (f32::MAX, f32::MIN);
            for i in 0..pts.len() {
                let (x0, y0) = pts[i];
                let (x1, y1) = pts[(i + 1) % pts.len()];
                if (y0 > scan) == (y1 > scan) {
                    continue;
                }
                let x = x0 + (scan - y0) / (y1 - y0) * (x1 - x0);
                lo = lo.min(x);
                hi = hi.max(x);
            }
            if lo <= hi {
                self.rectfill(lo.round() as i32, y, hi.round() as i32, y, color);
            }
        }
    }

    /// Draws a sprite, alpha-blending it over what is already on the canvas.
    pub fn blit(&mut self, atlas: &Atlas, spr: Sprite, x: i32, y: i32) {
        self.blit_inner(atlas, spr, x, y, None);
    }

    /// Draws a sprite's silhouette in one flat colour. The tint's own alpha scales the
    /// sprite's, so this covers both ground shadows and the solid outline that marks a
    /// threatened piece.
    pub fn blit_tinted(&mut self, atlas: &Atlas, spr: Sprite, x: i32, y: i32, tint: Rgba) {
        self.blit_inner(atlas, spr, x, y, Some(tint));
    }

    fn blit_inner(&mut self, atlas: &Atlas, spr: Sprite, x: i32, y: i32, tint: Option<Rgba>) {
        debug_assert!(spr.x as usize + spr.w as usize <= atlas.w);
        debug_assert!(spr.y as usize + spr.h as usize <= atlas.h);
        let (lo_x, lo_y, hi_x, hi_y) = self.clip;
        let dx0 = x + self.cam.0;
        let dy0 = y + self.cam.1;
        for row in 0..spr.h as i32 {
            let dy = dy0 + row;
            if dy < lo_y || dy >= hi_y {
                continue;
            }
            let src_row = (spr.y as i32 + row) as usize * atlas.w + spr.x as usize;
            let dst_row = dy as usize * FB_W;
            for col in 0..spr.w as i32 {
                let dx = dx0 + col;
                if dx < lo_x || dx >= hi_x {
                    continue;
                }
                let src = atlas.px[src_row + col as usize];
                let (color, a) = match tint {
                    Some(t) => (t, (src >> 24) * ((t >> 24) & 0xff) / 255),
                    None => (src, src >> 24),
                };
                if a == 0 {
                    continue;
                }
                let dst = &mut self.px[dst_row + dx as usize];
                *dst = if a == 255 {
                    color | 0xff00_0000
                } else {
                    blend(color, *dst, a)
                };
            }
        }
    }

    /// Squashes a horizontal band of the canvas towards `pivot`, filling what it
    /// vacates with `fill`. `height` is the fraction of its size the band keeps: 1.0
    /// leaves it as it was, and near zero flattens it to a line.
    ///
    /// It reads back what has already been drawn, so the whole picture in the band
    /// tips as one — squares, marks, pieces and their shadows — rather than every one
    /// of them having to know how to draw itself flattened. It works on pixels already
    /// on the canvas, which have had the camera and the clip rectangle applied to them
    /// on the way in, so neither is applied again here.
    pub fn squash(&mut self, top: i32, bottom: i32, pivot: f32, height: f32, fill: Rgba) {
        debug_assert!(top >= 0 && bottom <= FB_H as i32 && top < bottom);
        debug_assert!(height > 0.0 && height <= 1.0);
        debug_assert!(pivot >= top as f32 && pivot <= bottom as f32);
        let (top, bottom) = (top.max(0) as usize, bottom.clamp(0, FB_H as i32) as usize);
        if height >= 1.0 || top >= bottom {
            return;
        }
        let band: Vec<Rgba> = self.px[top * FB_W..bottom * FB_W].to_vec();
        for row in top..bottom {
            // Where this row of the squashed picture reads from in the original: the
            // further it is from the pivot the further out it reaches, which is what
            // draws the ends of the band in towards the middle. Rows reaching past the
            // band are simply not part of the picture any more.
            let src = (pivot + (row as f32 + 0.5 - pivot) / height - 0.5).round();
            let dst = &mut self.px[row * FB_W..(row + 1) * FB_W];
            if src < top as f32 || src >= bottom as f32 {
                dst.fill(fill);
            } else {
                let src = src as usize - top;
                dst.copy_from_slice(&band[src * FB_W..(src + 1) * FB_W]);
            }
        }
    }

    /// Converts the canvas into the RGBA byte buffer uploaded to the GPU, applying
    /// the screen-wide fade and flash on the way out so they cost one pass, not one
    /// per draw call.
    pub fn copy_to(&self, out: &mut [u8]) {
        debug_assert_eq!(out.len(), FB_W * FB_H * 4);
        let fade = self.fade.clamp(0.0, 1.0);
        let flash = self.flash.clamp(0.0, 1.0);
        // 0..=256 fixed point keeps the common untouched case exact.
        let keep = ((1.0 - fade) * 256.0) as u32;
        let white = (flash * 256.0) as u32;
        for (px, chunk) in self.px.iter().zip(out.chunks_exact_mut(4)) {
            let mut bytes = px.to_le_bytes();
            if keep < 256 {
                for b in &mut bytes[..3] {
                    *b = ((*b as u32 * keep) >> 8) as u8;
                }
            }
            if white > 0 {
                for b in &mut bytes[..3] {
                    *b = (*b as u32 + (((255 - *b as u32) * white) >> 8)) as u8;
                }
            }
            chunk.copy_from_slice(&bytes);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb_packs_into_upload_order() {
        assert_eq!(rgb(0x123456).to_le_bytes(), [0x12, 0x34, 0x56, 0xff]);
    }

    #[test]
    fn rectfill_respects_clip_and_camera() {
        let mut fb = Fb::new();
        fb.clear(rgb(0x000000));
        fb.clip(10, 10, 4, 4);
        fb.camera(10, 10);
        fb.rectfill(-100, -100, 100, 100, rgb(0xffffff));
        fb.set_clip((0, 0, FB_W as i32, FB_H as i32));
        fb.camera(0, 0);
        let mut lit = 0;
        let mut out = vec![0u8; FB_W * FB_H * 4];
        fb.copy_to(&mut out);
        for chunk in out.chunks_exact(4) {
            if chunk[0] == 0xff {
                lit += 1;
            }
        }
        assert_eq!(lit, 16);
    }

    #[test]
    fn fade_and_flash_are_identity_at_zero() {
        let mut fb = Fb::new();
        fb.clear(rgb(0x336699));
        let mut out = vec![0u8; FB_W * FB_H * 4];
        fb.copy_to(&mut out);
        assert_eq!(&out[..4], &[0x33, 0x66, 0x99, 0xff]);
    }
}
