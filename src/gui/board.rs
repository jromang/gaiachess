//! Board geometry and the drawing of a position.
//!
//! Squares are wider than they are tall and the pieces are taller than a square, so a
//! piece overhangs the rank behind it. That overlap is the whole trick: it reads as a
//! board seen from a low angle rather than a flat grid.

use super::assets::{Assets, CELL_BASELINE, CELL_H, CELL_W};
use super::fb::{FB_H, FB_W, Fb, Rgba, rgba};
use super::scheme::Scheme;
use crate::position::Position;
use crate::types::{Piece, Square};

pub const TILE_W: i32 = 20;
pub const TILE_H: i32 = 21;
pub const BOARD_W: i32 = TILE_W * 8;
pub const BOARD_H: i32 = TILE_H * 8;

/// Top-left corner of the board. The gap above it holds the status band and the room
/// the back-rank pieces need to stand up in.
pub const BOARD_X: i32 = (FB_W as i32 - BOARD_W) / 2;
pub const BOARD_Y: i32 = 26;
/// Height of the status band along the top.
pub const HUD_H: i32 = 20;

/// Row of its cell the tallest piece reaches. The king fills its cell to the very top,
/// so it is the one that decides how far down the board has to start.
const TALLEST: i32 = 0;

/// How far above the bottom edge of its square a piece stands. Anchored flush, the
/// feet occupy the square's last row and the baked ground shadow falls entirely onto
/// the rank in front; one row of clearance keeps a sliver of the square under the
/// piece, which is what makes it read as standing on that square.
pub const PIECE_LIFT: i32 = 1;

/// Where a piece's sprite cell sits inside its square: centred across, and low enough
/// that the feet land just inside the bottom edge. Named rather than folded into
/// [`piece_xy`] because turning the board round has to undo them again.
const PIECE_OFF_X: i32 = (TILE_W - CELL_W as i32) / 2;
const PIECE_OFF_Y: i32 = TILE_H - CELL_H as i32 + CELL_BASELINE - PIECE_LIFT;

/// The band of canvas the board and everything standing on it occupy: from the top of
/// the tallest back-rank piece down past the lip underneath. Turning the board round
/// tips this band and nothing else — the status line above it and the hint below it
/// belong to the screen rather than to the board.
pub const LAYER_Y: i32 = BOARD_Y + PIECE_OFF_Y + TALLEST;
pub const LAYER_BOTTOM: i32 = BOARD_Y + BOARD_H + 2;
/// The line the board tips about: the middle of the board itself, not of the band.
pub const LAYER_PIVOT: f32 = BOARD_Y as f32 + BOARD_H as f32 / 2.0;

const _: () = assert!(BOARD_Y + BOARD_H + 4 <= FB_H as i32, "board overflows the canvas");
/// The status band is painted after the pieces, so anything of a back-rank piece that
/// reaches into it is simply lost — which is how the king's crown went missing.
const _: () = assert!(
    BOARD_Y + TILE_H - CELL_H as i32 + CELL_BASELINE - PIECE_LIFT + TALLEST >= HUD_H,
    "the status band would paint over the back rank"
);

/// Pixel position of a square's top-left corner. `flipped` turns the board round for
/// a player sitting on the black side.
pub fn tile_xy(sq: Square, flipped: bool) -> (i32, i32) {
    debug_assert!(sq.0 < 64);
    let (file, rank) = (sq.0 as i32 % 8, sq.0 as i32 / 8);
    let (col, row) = if flipped {
        (7 - file, rank)
    } else {
        (file, 7 - rank)
    };
    (BOARD_X + col * TILE_W, BOARD_Y + row * TILE_H)
}

/// The square drawn at a place in the grid. The one mapping from screen to board:
/// hit-testing, the order the pieces are painted in and the colour of each tile all go
/// through it, so none of them can drift from the layout [`tile_xy`] lays down.
pub fn square_of(col: i32, row: i32, flipped: bool) -> Square {
    debug_assert!((0..8).contains(&col) && (0..8).contains(&row));
    let (file, rank) = if flipped {
        (7 - col, row)
    } else {
        (col, 7 - row)
    };
    Square((rank * 8 + file) as u8)
}

/// Whether a square is a light one. a1 is dark, so a square is light when its file and
/// its rank differ in parity.
pub fn is_light(sq: Square) -> bool {
    debug_assert!(sq.0 < 64);
    (sq.0 % 8 + sq.0 / 8) % 2 == 1
}

/// The square under a canvas pixel, if any.
pub fn square_at(x: i32, y: i32, flipped: bool) -> Option<Square> {
    let (col, row) = ((x - BOARD_X) / TILE_W, (y - BOARD_Y) / TILE_H);
    if x < BOARD_X || y < BOARD_Y || !(0..8).contains(&col) || !(0..8).contains(&row) {
        return None;
    }
    Some(square_of(col, row, flipped))
}

/// Where a piece standing on `sq` has its sprite cell drawn. The cell is anchored so
/// the piece's feet land just inside the bottom of the square.
pub fn piece_xy(sq: Square, flipped: bool) -> (i32, i32) {
    let (x, y) = tile_xy(sq, flipped);
    (x + PIECE_OFF_X, y + PIECE_OFF_Y)
}

/// The same point once the board has been turned round.
///
/// Turning the board is a half turn about its middle, so anything on it comes back at
/// the point reflected through that middle — a square, and equally a piece caught in
/// mid-air on its way to one. Reads and writes the coordinates [`piece_xy`] and
/// [`offboard_xy`] hand out, which is what the animations are written in.
pub fn mirror_piece_xy(p: (f32, f32)) -> (f32, f32) {
    (
        (2 * (BOARD_X + PIECE_OFF_X) + 7 * TILE_W) as f32 - p.0,
        (2 * (BOARD_Y + PIECE_OFF_Y) + 7 * TILE_H) as f32 - p.1,
    )
}

/// Draws the empty board, its edge and the background.
///
/// Each tile takes its colour from the square it is holding, so the checkerboard turns
/// with the board like everything else standing on it. That a1 comes back dark either
/// way round is then a result rather than an assumption: a half turn of an eight by
/// eight board is colour-preserving, so the pattern on the canvas happens to come out
/// the same both ways, and nothing here has to rest on having noticed that.
pub fn draw_board(fb: &mut Fb, scheme: &Scheme, flipped: bool) {
    fb.clear(scheme.bg);
    for row in 0..8 {
        for col in 0..8 {
            let color = if is_light(square_of(col, row, flipped)) {
                scheme.tile_light
            } else {
                scheme.tile_dark
            };
            let x = BOARD_X + col * TILE_W;
            let y = BOARD_Y + row * TILE_H;
            fb.rectfill(x, y, x + TILE_W - 1, y + TILE_H - 1, color);
        }
    }
    // A lip along the bottom and sides so the board sits on the background instead of
    // floating in it.
    fb.rectfill(
        BOARD_X - 1,
        BOARD_Y,
        BOARD_X - 1,
        BOARD_Y + BOARD_H,
        scheme.board_edge,
    );
    fb.rectfill(
        BOARD_X + BOARD_W,
        BOARD_Y,
        BOARD_X + BOARD_W,
        BOARD_Y + BOARD_H,
        scheme.board_edge,
    );
    fb.rectfill(
        BOARD_X - 1,
        BOARD_Y + BOARD_H,
        BOARD_X + BOARD_W,
        BOARD_Y + BOARD_H + 1,
        scheme.board_edge,
    );
}

/// Draws every piece on the board, back rank first so the pieces in front overlap the
/// ones behind. Squares set in `hidden` are left empty, which is how a piece being
/// animated stops appearing twice.
pub fn draw_pieces(
    fb: &mut Fb,
    assets: &Assets,
    scheme: &Scheme,
    pos: &Position,
    flipped: bool,
    hidden: u64,
    lifted: Option<Square>,
) {
    let shadow = rgba(0x000000, scheme.shadow_alpha);
    for row in 0..8 {
        for col in 0..8 {
            let sq = square_of(col, row, flipped);
            if hidden & (1u64 << sq.0) != 0 {
                continue;
            }
            let piece = pos.piece_on(sq);
            if piece == Piece::NONE {
                continue;
            }
            let (x, y) = piece_xy(sq, flipped);
            let pt = piece.piece_type();
            fb.blit_tinted(assets.sheet(), assets.shadow(pt), x, y, shadow);
            // A selected piece lifts a pixel, the same cue a hand hovering over it
            // would give.
            let lift = if Some(sq) == lifted { 1 } else { 0 };
            fb.blit(
                assets.sheet(),
                assets.piece(pt, scheme.variant(piece.color())),
                x,
                y - lift,
            );
        }
    }
}

/// Where pieces wait when they are not on the board: off the edge behind their own
/// side, so captured pieces leave the way they came in. Follows the flip, so white's
/// staging area is always on white's side of the table.
pub fn offboard_xy(color: crate::types::Color, flipped: bool) -> (f32, f32) {
    let (corner, dx, dy) = if color == crate::types::Color::White {
        (Square::A1, -2, 2)
    } else {
        (Square::H8, 2, -2)
    };
    // The step out from the corner turns with the board too. Flipped, a1 is drawn at
    // the top right, and walking two squares down and left from it would put white's
    // staging area in the middle of the board instead of off the edge behind it.
    let turn = if flipped { -1 } else { 1 };
    let (x, y) = piece_xy(corner, flipped);
    (
        (x + turn * dx * TILE_W) as f32,
        (y + turn * dy * TILE_H) as f32,
    )
}

/// Draws a piece away from the board grid, with its shadow left on the ground below
/// it. The gap between the two is what sells the hop.
pub fn draw_flying_piece(
    fb: &mut Fb,
    assets: &Assets,
    scheme: &Scheme,
    piece: crate::types::Piece,
    ground: (f32, f32),
    height: f32,
) {
    draw_piece_shadow(fb, assets, scheme, piece.piece_type(), ground, shadow_fade(height));
    draw_loose_piece(fb, assets, scheme, piece, (ground.0, ground.1 - height));
}

/// How much of its shadow a piece keeps at a given height. It shrinks as the piece
/// rises, the way a real one would.
pub fn shadow_fade(height: f32) -> f32 {
    (1.0 - height / 40.0).clamp(0.35, 1.0)
}

/// Draws the ground shadow belonging to `pt` at a sprite-cell position.
pub fn draw_piece_shadow(
    fb: &mut Fb,
    assets: &Assets,
    scheme: &Scheme,
    pt: crate::types::PieceType,
    at: (f32, f32),
    fade: f32,
) {
    fb.blit_tinted(
        assets.sheet(),
        assets.shadow(pt),
        at.0.round() as i32,
        at.1.round() as i32,
        rgba(0x000000, (scheme.shadow_alpha as f32 * fade) as u32),
    );
}

/// Draws a piece at an arbitrary position, off the grid and with nothing under it.
pub fn draw_loose_piece(
    fb: &mut Fb,
    assets: &Assets,
    scheme: &Scheme,
    piece: crate::types::Piece,
    at: (f32, f32),
) {
    fb.blit(
        assets.sheet(),
        assets.piece(piece.piece_type(), scheme.variant(piece.color())),
        at.0.round() as i32,
        at.1.round() as i32,
    );
}

/// Centre of a square, where arrows start and end.
pub fn tile_center(sq: Square, flipped: bool) -> (f32, f32) {
    let (x, y) = tile_xy(sq, flipped);
    (
        x as f32 + TILE_W as f32 / 2.0,
        y as f32 + TILE_H as f32 / 2.0,
    )
}

/// Draws a fat arrow between two square centres, with a round tail. It is what tells
/// the player, at a glance, which move they are about to commit to.
pub fn draw_arrow(
    fb: &mut Fb,
    assets: &Assets,
    from: (f32, f32),
    to: (f32, f32),
    color: Rgba,
) {
    const HALF_STEM: f32 = 2.5;
    const HALF_HEAD: f32 = 7.0;
    const HEAD_LEN: f32 = 9.0;

    let (dx, dy) = (to.0 - from.0, to.1 - from.1);
    let len = (dx * dx + dy * dy).sqrt();
    let blob = assets.ui(super::assets::Ui::ArrowBlob);
    if len > HEAD_LEN {
        let (ux, uy) = (dx / len, dy / len);
        // Perpendicular, for offsetting the edges of the stem and the head.
        let (px, py) = (-uy, ux);
        let neck = (to.0 - ux * HEAD_LEN, to.1 - uy * HEAD_LEN);
        fb.fill_poly(
            &[
                (from.0 + px * HALF_STEM, from.1 + py * HALF_STEM),
                (neck.0 + px * HALF_STEM, neck.1 + py * HALF_STEM),
                (neck.0 - px * HALF_STEM, neck.1 - py * HALF_STEM),
                (from.0 - px * HALF_STEM, from.1 - py * HALF_STEM),
            ],
            color,
        );
        fb.fill_poly(
            &[
                (neck.0 + px * HALF_HEAD, neck.1 + py * HALF_HEAD),
                to,
                (neck.0 - px * HALF_HEAD, neck.1 - py * HALF_HEAD),
            ],
            color,
        );
    }
    fb.blit_tinted(
        assets.ui_sheet(),
        blob,
        from.0 as i32 - blob.w as i32 / 2,
        from.1 as i32 - blob.h as i32 / 2,
        color,
    );
}

/// Rings a piece in a flat colour, marking it as capturable or in check. Eight offset
/// silhouettes, so the ring follows the piece's own shape rather than boxing it in.
pub fn draw_outline(fb: &mut Fb, assets: &Assets, sq: Square, flipped: bool, pt: crate::types::PieceType, color: Rgba) {
    let (x, y) = piece_xy(sq, flipped);
    let sprite = assets.piece(pt, 0);
    for (dx, dy) in [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (1, -1),
        (-1, 1),
        (1, 1),
    ] {
        fb.blit_tinted(assets.sheet(), sprite, x + dx, y + dy, color);
    }
}

/// Fills the status band along the top.
pub fn draw_hud_band(fb: &mut Fb, color: Rgba) {
    fb.rectfill(0, 0, FB_W as i32 - 1, HUD_H - 1, color);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corners_map_to_the_expected_pixels() {
        assert_eq!(tile_xy(Square::A8, false), (BOARD_X, BOARD_Y));
        assert_eq!(
            tile_xy(Square::H1, false),
            (BOARD_X + 7 * TILE_W, BOARD_Y + 7 * TILE_H)
        );
        // Flipped, the board is seen from black's chair: a8 lands bottom-right.
        assert_eq!(
            tile_xy(Square::A8, true),
            (BOARD_X + 7 * TILE_W, BOARD_Y + 7 * TILE_H)
        );
        assert_eq!(tile_xy(Square::H1, true), (BOARD_X, BOARD_Y));
    }

    #[test]
    fn hit_testing_is_the_inverse_of_layout() {
        for flipped in [false, true] {
            for i in 0..64u8 {
                let sq = Square(i);
                let (x, y) = tile_xy(sq, flipped);
                assert_eq!(square_at(x, y, flipped), Some(sq));
                assert_eq!(square_at(x + TILE_W - 1, y + TILE_H - 1, flipped), Some(sq));
            }
        }
    }

    #[test]
    fn every_tile_is_painted_the_colour_of_the_square_it_holds() {
        for flipped in [false, true] {
            for col in 0..8 {
                for row in 0..8 {
                    let sq = square_of(col, row, flipped);
                    // Read the layout back the other way round: the tile at this place
                    // in the grid really is where that square gets drawn.
                    let (x, y) = tile_xy(sq, flipped);
                    assert_eq!(
                        ((x - BOARD_X) / TILE_W, (y - BOARD_Y) / TILE_H),
                        (col, row),
                        "flipped {flipped}"
                    );
                    assert_eq!(is_light(sq), (sq.0 % 8 + sq.0 / 8) % 2 == 1);
                }
            }
        }
        assert!(!is_light(Square::A1), "a1 is a dark square");
        assert!(is_light(Square::H1), "h1 is a light square");
        // Which is what puts a light square in the near right corner for whoever is
        // sitting there: h1 for white, a8 for black.
        for (sq, flipped) in [(Square::H1, false), (Square::A8, true)] {
            assert_eq!(square_of(7, 7, flipped), sq, "the near right corner");
            assert!(is_light(sq), "the near right corner is light");
        }
    }

    #[test]
    fn the_pattern_on_the_canvas_is_the_same_either_way_round() {
        // Nothing in the drawing rests on this any more, but it is worth knowing and
        // worth keeping true: a half turn of an eight by eight board preserves colour,
        // so turning the board does not move a single light square on the canvas. If
        // this ever failed, the layout would have stopped being a half turn.
        for col in 0..8 {
            for row in 0..8 {
                assert_eq!(
                    is_light(square_of(col, row, false)),
                    is_light(square_of(col, row, true)),
                    "tile {col},{row} changes colour when the board turns"
                );
            }
        }
    }

    #[test]
    fn the_mirror_is_the_turn() {
        // Everything an animation can be aimed at has to come back at the point the
        // turned board draws it, or a flip mid-flight sends the piece somewhere the
        // board no longer is.
        for i in 0..64u8 {
            let sq = Square(i);
            let there = as_f32(piece_xy(sq, false));
            let back = as_f32(piece_xy(sq, true));
            assert_eq!(mirror_piece_xy(there), back, "square {i}");
            assert_eq!(mirror_piece_xy(back), there, "square {i}, back again");
        }
        for color in [crate::types::Color::White, crate::types::Color::Black] {
            assert_eq!(
                mirror_piece_xy(offboard_xy(color, false)),
                offboard_xy(color, true),
                "the staging area of {color:?}"
            );
        }
    }

    #[test]
    fn the_staging_areas_are_off_the_board_whichever_way_round_it_is() {
        for flipped in [false, true] {
            for color in [crate::types::Color::White, crate::types::Color::Black] {
                let (x, y) = offboard_xy(color, flipped);
                let (x, y) = (x as i32, y as i32);
                let outside = x + CELL_W as i32 <= BOARD_X
                    || x >= BOARD_X + BOARD_W
                    || y + CELL_H as i32 <= BOARD_Y
                    || y >= BOARD_Y + BOARD_H;
                assert!(outside, "{color:?} waits on the board itself: {x},{y}");
            }
        }
    }

    fn as_f32(p: (i32, i32)) -> (f32, f32) {
        (p.0 as f32, p.1 as f32)
    }

    #[test]
    fn pixels_outside_the_board_hit_nothing() {
        assert_eq!(square_at(BOARD_X - 1, BOARD_Y, false), None);
        assert_eq!(square_at(BOARD_X, BOARD_Y - 1, false), None);
        assert_eq!(square_at(BOARD_X + BOARD_W, BOARD_Y, false), None);
        assert_eq!(square_at(BOARD_X, BOARD_Y + BOARD_H, false), None);
    }

    #[test]
    fn no_piece_reaches_higher_in_its_cell_than_the_layout_allows() {
        // The layout leaves room above the back rank for a piece reaching row TALLEST.
        // If the artwork ever reaches higher, the status band silently eats the top of
        // it, so read the height back off the sheet rather than trusting the constant.
        let assets = Assets::load();
        let sheet = assets.sheet();
        let highest = [
            crate::types::PieceType::Pawn,
            crate::types::PieceType::Knight,
            crate::types::PieceType::Bishop,
            crate::types::PieceType::Rook,
            crate::types::PieceType::Queen,
            crate::types::PieceType::King,
        ]
        .into_iter()
        .map(|pt| {
            let s = assets.piece(pt, 0);
            (0..s.h as i32)
                .find(|row| {
                    (0..s.w as i32).any(|col| sheet.opaque_at(s.x as i32 + col, s.y as i32 + row))
                })
                .expect("a piece sprite cannot be blank")
        })
        .min()
        .unwrap();
        assert_eq!(highest, TALLEST, "the artwork's tallest piece has moved");
    }

    #[test]
    fn feet_stand_on_the_square_not_on_its_edge() {
        for flipped in [false, true] {
            for i in 0..64u8 {
                let sq = Square(i);
                let (_, tile_y) = tile_xy(sq, flipped);
                let (_, cell_y) = piece_xy(sq, flipped);
                // Last row the piece artwork paints, from the sheet's own geometry.
                let feet = cell_y + CELL_H as i32 - 1 - CELL_BASELINE;
                assert_eq!(feet, tile_y + TILE_H - 1 - PIECE_LIFT);
                assert!(feet < tile_y + TILE_H, "feet spill past the square");
            }
        }
    }

    #[test]
    fn pieces_stay_inside_the_canvas() {
        for flipped in [false, true] {
            for i in 0..64u8 {
                let (x, y) = piece_xy(Square(i), flipped);
                assert!(x >= 0 && x + CELL_W as i32 <= FB_W as i32);
                assert!(y >= 0 && y + CELL_H as i32 <= FB_H as i32);
            }
        }
    }
}
