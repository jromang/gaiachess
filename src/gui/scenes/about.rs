//! The about window: who made this, under what terms, and what it is underneath.
//!
//! It opens on an empty window and the roll climbs in from the bottom, then scrolls on
//! its own, slowly and forever, with the arrows to push it along. Credit is owed to
//! several people here, so the text is content rather than decoration: the panel is
//! sized from the longest line rather than the lines being cut to fit.

use super::super::fb::{FB_H, FB_W, Fb, rgba};
use super::super::font;
use super::super::input::{Dir, Input};
use super::super::lang::{Key, Lang, t};
use super::super::scheme::Scheme;

/// One line of the roll.
#[derive(Clone, Copy)]
enum Line {
    /// Section title.
    Head(&'static str),
    /// Body text.
    Text(&'static str),
    /// Something to type into a browser or a mail client.
    Link(&'static str),
    /// Breathing space between sections.
    Gap,
}

impl Line {
    fn height(self) -> i32 {
        match self {
            Line::Gap => 5,
            _ => 7,
        }
    }

    fn text(self) -> &'static str {
        match self {
            Line::Head(s) | Line::Text(s) | Line::Link(s) => s,
            Line::Gap => "",
        }
    }
}

/// The version the engine reports over UCI, so the two can never drift apart.
const VERSION: &str = concat!("version ", env!("CARGO_PKG_VERSION"));

use Line::{Gap, Head, Link, Text};

const ROLL: &[Line] = &[
    Head("gaiachess"),
    Text(VERSION),
    Text("made with love"),
    Gap,
    Head("author"),
    Text("jean-francois romang"),
    Link("jromang@protonmail.com"),
    Link("github.com/jromang/gaiachess"),
    Gap,
    Head("an engine first"),
    Text("this board is a coat of paint on a"),
    Text("uci chess engine. started with"),
    Text("--no-gui it speaks uci instead, so"),
    Text("any chess interface can play it or"),
    Text("enter it in a tournament."),
    Gap,
    Text("with no arguments it listens for a"),
    Text("moment first: if an interface"),
    Text("speaks it answers, and if nobody"),
    Text("does, this board opens."),
    Gap,
    Head("levels"),
    Text("twenty rungs, from a first game of"),
    Text("chess up to grandmaster. the menu"),
    Text("gives each one a rating and says"),
    Text("who it plays like, and any chess"),
    Text("interface can ask for the same rung"),
    Gap,
    Text("a weak level is not the engine with"),
    Text("the brakes on. it looks less far"),
    Text("ahead, judges more roughly, gets"),
    Text("things wrong on purpose, and above"),
    Text("all misses moves - which is how the"),
    Text("low rungs hang a piece instead of"),
    Text("merely playing a dull move"),
    Gap,
    Text("treat the ratings as estimates, but"),
    Text("they were measured, not guessed:"),
    Text("the middle of the ladder played"),
    Text("bots that imitate human play and"),
    Text("carry a rating earned against"),
    Text("people; the two ends follow the"),
    Text("same line"),
    Gap,
    Text("every rung was also read the way a"),
    Text("player's games are - how much it"),
    Text("drops a move, how often it blunders,"),
    Text("and how long before the first one"),
    Gap,
    Text("nothing is softened in the endgame"),
    Text("either. a low rung can miss an easy"),
    Text("mate, the way a beginner does"),
    Gap,
    Text("none of it comes from the clock. a"),
    Text("level is the same opponent on a"),
    Text("fast machine and a slow one, with"),
    Text("the same blind spots every time it"),
    Text("meets a position"),
    Gap,
    Text("level 20 is the whole engine, with"),
    Text("nothing held back. it is not trying"),
    Text("to be kind"),
    Gap,
    Head("playing"),
    Text("arrows and enter move a piece, or"),
    Text("drag it with the mouse. esc opens"),
    Text("the menu, f turns the board round,"),
    Text("tab changes the colours, u takes a"),
    Text("move back, r starts again."),
    Text("esc on the title screen quits."),
    Gap,
    Head("licence"),
    Text("(c) 2003-2026 jean-francois"),
    Text("romang. free software under the"),
    Text("gnu general public licence,"),
    Text("version 3 or later. it comes"),
    Text("with no warranty. the source is"),
    Text("on github: read it, change it,"),
    Text("pass it on under the same terms."),
    Gap,
    Head("pieces"),
    Text("pixel art by drsmey, used as"),
    Text("drawn and with thanks."),
    Link("reddit.com/r/pixelart"),
    Link("drsmey.itch.io"),
    Gap,
    Head("openings"),
    Text("the built-in book is the eco"),
    Text("catalogue kept by lichess, put"),
    Text("in the public domain."),
    Link("lichess-org/chess-openings"),
    Gap,
    Head("sounds"),
    Text("not recorded but made, at"),
    Text("startup, from a handful of"),
    Text("numbers each."),
    Gap,
    Head("inspiration"),
    Text("pico checkmate by krystman, of"),
    Text("lazy devs. what it taught this"),
    Text("screen about how a board should"),
    Text("feel was rebuilt from scratch."),
    Link("lexaloffle.com/bbs/?tid=31213"),
    Gap,
    Head("thanks"),
    Text("to the chess programming"),
    Text("community, and to anyone who"),
    Text("sits down for a game."),
];

const PANEL_X: i32 = 6;
const PANEL_W: i32 = FB_W as i32 - 2 * PANEL_X;
const PANEL_Y: i32 = 16;
const PANEL_H: i32 = FB_H as i32 - 2 * PANEL_Y;
/// Left edge of the text, and how much of the panel is left for it.
const TEXT_X: i32 = PANEL_X + 8;
const TEXT_W: i32 = PANEL_W - 16;
/// The panel's own furniture: a title, a rule under it, a rule over the closing hint,
/// and the hint itself.
const HEAD_Y: i32 = PANEL_Y + 4;
const RULE_TOP: i32 = PANEL_Y + 11;
const RULE_FOOT: i32 = PANEL_Y + PANEL_H - 12;
const FOOT_Y: i32 = PANEL_Y + PANEL_H - 8;
/// The scrolling part of the panel, between the two rules.
const VIEW_Y: i32 = PANEL_Y + 14;
const VIEW_H: i32 = RULE_FOOT - VIEW_Y - 1;

/// Pixels the roll climbs per step. Slow enough to read without chasing it, and the
/// arrows are there for a reader in more of a hurry.
const CLIMB: f32 = 0.1;
/// Pixels an arrow press moves it, about two lines.
const NUDGE: f32 = 14.0;

pub struct About {
    /// How far the roll has climbed, wrapping at one whole turn.
    scroll: f32,
}

impl About {
    pub fn new() -> About {
        // Opens on nothing, with the first line one whole window below the top edge, so
        // the roll arrives from the bottom instead of already being there. That blank
        // is the one a turn already ends on, so this is a place in the loop like any
        // other rather than a state of its own.
        About {
            scroll: (-(VIEW_H as f32)).rem_euclid(span()),
        }
    }

    /// Advances the roll. Returns true once the reader has asked to close it.
    pub fn update(&mut self, input: &Input) -> bool {
        let span = span();
        self.scroll += CLIMB;
        if input.dir(Dir::Down) {
            self.scroll += NUDGE;
        }
        if input.dir(Dir::Up) {
            self.scroll -= NUDGE;
        }
        // The roll is a loop, so any offset is a valid place to be.
        self.scroll = self.scroll.rem_euclid(span);
        input.confirm || input.cancel || input.press
    }

    /// The roll laid out against the window: every line with the y it is drawn at,
    /// twice over and one whole turn apart, so the tail of the roll and its head share
    /// the window while it wraps. What is on screen is read from here rather than
    /// worked out twice.
    fn layout(&self) -> impl Iterator<Item = (Line, i32)> + '_ {
        let top = VIEW_Y - self.scroll.round() as i32;
        [0, span() as i32].into_iter().flat_map(move |turn| {
            ROLL.iter().scan(top + turn, |y, line| {
                let at = *y;
                *y += line.height();
                Some((*line, at))
            })
        })
    }

    /// The roll itself stays in English: it is prose cut by hand to the width of the
    /// panel, and the credits and the licence in it are canonical. Only the window
    /// around it -- its title and the line saying how to leave -- is translated.
    pub fn draw(&self, fb: &mut Fb, scheme: &Scheme, lang: Lang) {
        // Knock the menu behind it back, so the panel reads as being in front rather
        // than as another layer of the same picture.
        fb.rectfill(0, 0, FB_W as i32 - 1, FB_H as i32 - 1, rgba(0x000000, 120));
        fb.rectfill(
            PANEL_X + 2,
            PANEL_Y + 3,
            PANEL_X + PANEL_W + 1,
            PANEL_Y + PANEL_H + 2,
            rgba(0x000000, 90),
        );
        fb.rectfill(
            PANEL_X,
            PANEL_Y,
            PANEL_X + PANEL_W - 1,
            PANEL_Y + PANEL_H - 1,
            scheme.panel,
        );
        fb.rect(
            PANEL_X,
            PANEL_Y,
            PANEL_X + PANEL_W - 1,
            PANEL_Y + PANEL_H - 1,
            scheme.panel_edge,
        );

        font::print_centered(fb, t(Key::About, lang), FB_W as i32 / 2, HEAD_Y, scheme.accent);
        for y in [RULE_TOP, RULE_FOOT] {
            fb.rect(
                PANEL_X + 4,
                y,
                PANEL_X + PANEL_W - 5,
                y,
                scheme.panel_edge,
            );
        }

        // Clipped to the text column, so a line arriving or leaving is cut cleanly at
        // the window's edge instead of spilling onto the rules.
        let clip = fb.clip(TEXT_X, VIEW_Y, TEXT_W, VIEW_H);
        for (line, y) in self.layout() {
            if visible(line, y) {
                let ink = match line {
                    Line::Head(_) => scheme.accent,
                    Line::Text(_) => scheme.text,
                    Line::Link(_) => scheme.tile_light,
                    Line::Gap => scheme.text,
                };
                font::print(fb, line.text(), TEXT_X, y, ink);
            }
        }
        fb.set_clip(clip);

        font::print_centered(fb, t(Key::AboutHint, lang), FB_W as i32 / 2, FOOT_Y, scheme.tile_light);
    }
}

/// Whether a line drawn at `y` shows any of itself in the window.
fn visible(line: Line, y: i32) -> bool {
    y > VIEW_Y - line.height() && y < VIEW_Y + VIEW_H
}

/// Height of one whole turn of the roll: all of the text, plus a window's worth of
/// nothing so the end has cleared the top before the start comes back in.
fn span() -> f32 {
    (ROLL.iter().map(|l| l.height()).sum::<i32>() + VIEW_H) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_line_fits_the_panel() {
        for line in ROLL {
            let text = line.text();
            assert!(
                font::width(text) <= TEXT_W,
                "{text:?} is {} wide, panel holds {TEXT_W}",
                font::width(text)
            );
        }
    }

    #[test]
    fn every_line_can_be_written_in_this_font() {
        // A character the font has no glyph for prints as a blank, which is a silent
        // hole in a credit or a link — the two things here that have to be right.
        for line in ROLL {
            for c in line.text().chars() {
                assert!(
                    font::has_glyph(c),
                    "{c:?} has no glyph, in {:?}",
                    line.text()
                );
            }
        }
    }

    #[test]
    fn the_roll_comes_back_round_to_where_it_started() {
        // Watched for rather than counted out: the scroll is a float advanced a tenth
        // at a time, so over the thousands of steps a full turn takes it drifts by
        // most of a step, and a test that predicts the exact step it wraps on breaks
        // the next time a line is added to the roll.
        let mut about = About::new();
        let start = about.scroll;
        let steps = (span() / CLIMB).ceil() as u32 + 4;
        let (mut wrapped, mut previous) = (false, about.scroll);
        for _ in 0..steps {
            assert!(!about.update(&Input::default()), "it closed on its own");
            wrapped |= about.scroll < previous;
            previous = about.scroll;
        }
        assert!(wrapped, "the roll never came back round");
        assert!(
            (about.scroll - start).abs() < CLIMB * 8.0,
            "a whole turn ended at {} instead of back at {start}",
            about.scroll
        );
    }

    #[test]
    fn it_opens_on_an_empty_window() {
        // Asked for, the window is bare: nothing to catch the eye mid-sentence, and
        // nothing to read before the roll has started.
        let about = About::new();
        assert!(
            !about.layout().any(|(line, y)| visible(line, y)),
            "the roll is already on screen the moment the window opens"
        );
    }

    #[test]
    fn the_roll_arrives_from_the_bottom() {
        // And it is the head of the roll that arrives, not the tail of the previous
        // turn caught on its way out: the reader starts at the first line.
        let mut about = About::new();
        for _ in 0..(VIEW_H as f32 / CLIMB) as u32 {
            about.update(&Input::default());
            let Some((line, y)) = about.layout().find(|(line, y)| visible(*line, *y)) else {
                continue;
            };
            assert!(
                matches!(line, Line::Head(_)) && line.text() == ROLL[0].text(),
                "{:?} came in first, not the head of the roll",
                line.text()
            );
            assert!(
                y >= VIEW_Y + VIEW_H - line.height(),
                "it appeared at {y}, not at the bottom edge of the window"
            );
            return;
        }
        panic!("nothing ever climbed into the window");
    }

    #[test]
    fn a_key_closes_it() {
        let mut about = About::new();
        let mut input = Input::default();
        input.confirm = true;
        assert!(about.update(&input));
    }

    /// The body of one section of the roll, from its title down to the next one.
    fn section(head: &str) -> String {
        ROLL.iter()
            .skip_while(|l| !matches!(l, Line::Head(h) if *h == head))
            .skip(1)
            .take_while(|l| !matches!(l, Line::Head(_)))
            .map(|l| l.text())
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[test]
    fn the_controls_it_names_are_the_ones_the_screen_advertises() {
        // What went wrong here once already: the roll named the z and x a fantasy
        // console would use, while every on-screen hint named enter and escape. Both
        // still work, but a player reading this should be told what the game itself
        // tells them, and told about keys that do something.
        let playing = format!(" {} ", section("playing"));
        for key in ["enter", "esc", "f", "tab", "u", "r"] {
            assert!(
                playing.contains(&format!(" {key} ")),
                "{key:?} is not named in the playing section:{playing}"
            );
        }
    }

    #[test]
    fn it_names_the_switch_that_really_forces_uci() {
        // A bare launch listens, and then opens the board when nobody speaks — so
        // "no arguments" is not how you ask for the protocol. The flag is.
        assert!(
            section("an engine first").contains("--no-gui"),
            "the roll does not say how to get uci without the board"
        );
    }

    #[test]
    fn the_credits_that_are_owed_are_all_there() {
        // The artwork is used by name, which is a condition rather than a courtesy;
        // the rest is owed because it was borrowed, licence or no licence.
        let all: String = ROLL.iter().map(|l| l.text()).collect::<Vec<_>>().join(" ");
        for owed in [
            "drsmey",
            "krystman",
            "lichess-org/chess-openings",
            "jean-francois romang",
            "github.com/jromang/gaiachess",
            "jromang@protonmail.com",
            "lexaloffle.com/bbs/?tid=31213",
            "general public licence",
        ] {
            assert!(all.contains(owed), "{owed:?} is missing from the about roll");
        }
    }
}
