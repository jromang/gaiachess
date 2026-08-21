# Artwork and audio credits

## Chess pieces — `pieces.png`

Pixel art by **DrSmey** (Reddit `u/Rangersimi`), posted 2021-09-11:
<https://www.reddit.com/r/PixelArt/comments/pmfegd/sets_of_chess_pieces/>
Also at <https://drsmey.itch.io/>, Instagram `@dr.smey`, Twitter `@dr_smey`.

Six piece types in four colour variants. `pieces.png` is a mechanical repack of the
original sheet produced by `tools/gui_assets/prepare_pieces.py`: the whole-number
upscale is undone, the flat background is made transparent, the cells are aligned on a
uniform grid and the baked ground shadows are moved to a row of their own. **No pixel
of the artwork itself is altered.**

## Inspiration — and nothing more

The look owes a debt to the ROM of a retro chess cartridge by **Krystman** (Lazy Devs),
licensed **CC BY-NC-SA 4.0**: <https://www.lexaloffle.com/bbs/?tid=31213>

**Not one byte of it is here.** Its licence forbids commercial use and demands share-alike,
neither of which would sit with the GPL — so it is worth being able to point at why
neither applies: the repository holds no audio file of any kind, the sounds are made at
start-up by `src/gui/synth.rs` from a handful of sfxr parameters, and the cartridge
screenshot the palettes were read against is gitignored and has never been committed.
What was taken is an impression, which is not a thing a licence covers.

## Icon — `icon16.png`, `icon32.png`

The pawn from `pieces.png` above, framed in the interface's accent colour, assembled by
`tools/gui_assets/make_icon.py`. `icon32.png` is the sheet's cell used as drawn, so it
carries the same credit as the sheet. `icon16.png` is drawn here: the cells are 18x28 and
nothing from them fits a 16x16 tile, and shrinking the 32 leaves a smudge whichever
filter is used.

## Store art — `tools/gui_assets/store/`

The cover, social card, wide banner, logo and favicon, assembled by
`tools/gui_assets/make_store_art.py` out of the game's own screenshots, icon and font.
The boards on them carry the piece sheet above, so the same credit goes with them
wherever they are shown.

## Everything else

The hands in `ui.png`, the move marks beside them, the 3x5 text font, the board, the
panels and all the code in `src/gui/` are GaiaChess's own: © 2003-2026 Jean-François
Romang, GPL-3.0-or-later.

The hands are cut from `tools/gui_assets/source/cursors.png` by
`tools/gui_assets/prepare_cursors.py`, which undoes the grid the sheet's enlargement
drifted onto without inventing or moving a pixel. That sheet is the author's own work,
which is worth saying here: the script talks about a *source sheet* and would otherwise
read as if the hands came from somewhere else.
