//! Artwork embedded in the binary and decoded once at start-up.
//!
//! Keeping the sheets inside the executable is what lets the interface ship as a
//! single file with nothing to install alongside it.

use macroquad::prelude::{Image, ImageFormat};

use super::fb::{Atlas, Sprite};
use crate::types::PieceType;

/// Piece sheet: six piece types across, four colour variants down, then one more row
/// holding the ground shadow that belongs under each piece.
/// Artwork by DrSmey (see assets/ATTRIBUTION.md); prepared by tools/gui_assets.
static PIECES_PNG: &[u8] = include_bytes!("assets/pieces.png");

/// Cell geometry of the piece sheet, as reported by the preparation script.
pub const CELL_W: u16 = 18;
pub const CELL_H: u16 = 28;
/// Rows between a piece's feet and the bottom edge of its cell. The gap is where the
/// ground shadow spreads out, so both layers share one anchor.
pub const CELL_BASELINE: i32 = 2;

/// Number of colour variants on the sheet.
pub const VARIANTS: u16 = 4;

/// Sheet column for each [`PieceType`]. The artwork orders them pawn, bishop, queen,
/// king, knight, rook, which is not the engine's ordering.
const PIECE_COLUMN: [u16; 6] = [0, 4, 1, 5, 2, 3];

/// Interface sprites: the hands cut from the cursor sheet, the move marks drawn by
/// us. See tools/gui_assets/prepare_cursors.py, which prints the sizes below.
static UI_PNG: &[u8] = include_bytes!("assets/ui.png");

/// The window icon, at the two sizes it is drawn at.
///
/// Two drawings rather than one scaled: 18x28 cells mean nothing from the sheet fits a
/// 16x16 tile, and every way of shrinking the 32 was tried — box, Lanczos, bilinear,
/// nearest — with the piece turning to a smudge and the frame to mud in all four. So 16
/// is drawn by hand and 32 is the sheet's own pawn. Both from
/// tools/gui_assets/make_icon.py.
static ICON_16_PNG: &[u8] = include_bytes!("assets/icon16.png");
static ICON_32_PNG: &[u8] = include_bytes!("assets/icon32.png");

/// The icon at the three sizes the window system asks for.
///
/// 64 is twice 32, which is exact: whole multiples are the only enlargement that leaves
/// pixel art alone.
pub fn window_icon() -> macroquad::miniquad::conf::Icon {
    fn decode(png: &[u8], side: u16) -> Image {
        let img = Image::from_file_with_format(png, Some(ImageFormat::Png))
            .expect("embedded icon must decode");
        assert_eq!(img.width, side, "icon is not {side} wide");
        assert_eq!(img.height, side, "icon is not {side} tall");
        img
    }

    /// Repeats each pixel `factor` times in both directions.
    fn magnify<const N: usize>(src: &[u8], side: usize, factor: usize) -> [u8; N] {
        let mut out = [0u8; N];
        let wide = side * factor;
        for y in 0..wide {
            for x in 0..wide {
                let from = ((y / factor) * side + (x / factor)) * 4;
                let to = (y * wide + x) * 4;
                out[to..to + 4].copy_from_slice(&src[from..from + 4]);
            }
        }
        out
    }

    let small = decode(ICON_16_PNG, 16);
    let medium = decode(ICON_32_PNG, 32);

    macroquad::miniquad::conf::Icon {
        small: magnify::<{ 16 * 16 * 4 }>(&small.bytes, 16, 1),
        medium: magnify::<{ 32 * 32 * 4 }>(&medium.bytes, 32, 1),
        big: magnify::<{ 64 * 64 * 4 }>(&medium.bytes, 32, 2),
    }
}

/// Cell pitch of the interface sheet. Each sprite sits in the corner of its cell.
const UI_CELL: u16 = 18;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Ui {
    /// An index finger pointing right, for picking out a menu row.
    Pointer,
    /// An open palm, hovering over a square.
    HandOpen,
    /// A closed fist, carrying a piece.
    HandGrab,
    MoveDot,
    MoveRing,
    ArrowBlob,
}

impl Ui {
    /// Size of the drawn part, shadow included.
    fn size(self) -> (u16, u16) {
        match self {
            Ui::Pointer => (18, 16),
            Ui::HandOpen => (16, 19),
            Ui::HandGrab => (14, 14),
            _ => (8, 8),
        }
    }

    /// The pixel of the sprite that lands on the point being aimed at.
    ///
    /// The two hands are pinned through the middle of the palm, and share it: closing
    /// the fist then curls the fingers in where they were instead of jolting the whole
    /// hand sideways.
    pub fn hotspot(self) -> (i32, i32) {
        match self {
            // The fingertip, so the finger points at what it means.
            Ui::Pointer => (17, 6),
            Ui::HandOpen => (8, 10),
            Ui::HandGrab => (7, 6),
            _ => (4, 4),
        }
    }
}

pub struct Assets {
    pieces: Atlas,
    ui: Atlas,
}

impl Assets {
    pub fn load() -> Assets {
        let img = Image::from_file_with_format(PIECES_PNG, Some(ImageFormat::Png))
            .expect("embedded piece sheet must decode");
        let (w, h) = (img.width as usize, img.height as usize);
        assert_eq!(w, CELL_W as usize * 6, "piece sheet width does not match CELL_W");
        assert_eq!(
            h,
            CELL_H as usize * (VARIANTS as usize + 1),
            "piece sheet height does not match CELL_H"
        );
        let ui = Image::from_file_with_format(UI_PNG, Some(ImageFormat::Png))
            .expect("embedded interface sheet must decode");
        Assets {
            pieces: Atlas::from_rgba(&img.bytes, w, h),
            ui: Atlas::from_rgba(&ui.bytes, ui.width as usize, ui.height as usize),
        }
    }

    pub fn sheet(&self) -> &Atlas {
        &self.pieces
    }

    pub fn ui_sheet(&self) -> &Atlas {
        &self.ui
    }

    pub fn ui(&self, item: Ui) -> Sprite {
        let (w, h) = item.size();
        Sprite {
            x: item as u16 * UI_CELL,
            y: 0,
            w,
            h,
        }
    }

    /// The cell holding `pt` painted in colour variant `variant`.
    pub fn piece(&self, pt: PieceType, variant: u16) -> Sprite {
        debug_assert!(variant < VARIANTS);
        Sprite {
            x: PIECE_COLUMN[pt as usize] * CELL_W,
            y: variant * CELL_H,
            w: CELL_W,
            h: CELL_H,
        }
    }

    /// The ground shadow belonging under `pt`, drawn at the same anchor as the piece.
    pub fn shadow(&self, pt: PieceType) -> Sprite {
        Sprite {
            x: PIECE_COLUMN[pt as usize] * CELL_W,
            y: VARIANTS * CELL_H,
            w: CELL_W,
            h: CELL_H,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_piece_type_maps_to_a_distinct_column() {
        let mut seen = PIECE_COLUMN;
        seen.sort_unstable();
        assert_eq!(seen, [0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn every_hotspot_lands_inside_its_sprite() {
        for item in [Ui::Pointer, Ui::HandOpen, Ui::HandGrab, Ui::MoveDot, Ui::MoveRing] {
            let ((w, h), (hx, hy)) = (item.size(), item.hotspot());
            assert!(
                (0..w as i32).contains(&hx) && (0..h as i32).contains(&hy),
                "{item:?} hotspot {:?} is outside {:?}",
                item.hotspot(),
                item.size()
            );
        }
    }

    #[test]
    fn sprites_stay_inside_the_sheet() {
        let sheet_w = CELL_W * 6;
        let sheet_h = CELL_H * (VARIANTS + 1);
        for pt in [
            PieceType::Pawn,
            PieceType::Knight,
            PieceType::Bishop,
            PieceType::Rook,
            PieceType::Queen,
            PieceType::King,
        ] {
            for variant in 0..VARIANTS {
                let s = Sprite {
                    x: PIECE_COLUMN[pt as usize] * CELL_W,
                    y: variant * CELL_H,
                    w: CELL_W,
                    h: CELL_H,
                };
                assert!(s.x + s.w <= sheet_w && s.y + s.h <= sheet_h);
            }
        }
    }
}
