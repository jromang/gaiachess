//! Pieces in flight, and the shake when one lands.
//!
//! A piece never slides between squares: it hops, tracing an arc with its shadow
//! staying behind on the ground. Everything here is measured in logic steps, so the
//! motion is identical whatever the display is doing.

use super::fb::FB_H;
use crate::types::Piece;

/// What a piece did on this step, so the scene can react to it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AnimEvent {
    /// Left the ground on this step.
    TookOff,
    /// Touched down on this step.
    Landed,
}

/// Height of the arc, in pixels per square travelled.
const ARC_PER_SQUARE: f32 = 5.0;
/// Steps a hop takes, per square travelled.
const STEPS_PER_SQUARE: f32 = 4.5;
/// Shortest a hop may be, so a one-square move still reads as a hop.
const MIN_STEPS: f32 = 7.0;

/// A piece travelling from one place to another.
pub struct PieceAnim {
    pub piece: Piece,
    from: (f32, f32),
    to: (f32, f32),
    /// Progress, 0.0 at the start and 1.0 on landing.
    t: f32,
    step: f32,
    arc: f32,
    /// Steps still to wait before setting off.
    delay: u32,
    /// True once the piece has landed and stopped mattering.
    pub done: bool,
    /// This piece has been taken and is on its way off the board.
    pub leaving: bool,
    airborne: bool,
}

impl PieceAnim {
    /// A hop between two sprite-cell positions. `squares` is the distance travelled on
    /// the board, which sets both how long the hop takes and how high it goes.
    pub fn new(piece: Piece, from: (f32, f32), to: (f32, f32), squares: f32) -> PieceAnim {
        let steps = (STEPS_PER_SQUARE * squares).max(MIN_STEPS);
        PieceAnim {
            piece,
            from,
            to,
            t: 0.0,
            step: 1.0 / steps,
            arc: ARC_PER_SQUARE * squares,
            delay: 0,
            done: false,
            leaving: false,
            airborne: false,
        }
    }

    /// Marks a piece as being carried off after a capture, which sounds different
    /// from a piece simply moving.
    pub fn taken(mut self) -> PieceAnim {
        self.leaving = true;
        self
    }

    /// How long a hop of this length takes, needed to time one animation against
    /// another.
    pub fn steps_for(squares: f32) -> u32 {
        (STEPS_PER_SQUARE * squares).max(MIN_STEPS).ceil() as u32
    }

    /// Holds the piece on the ground for `steps` before it sets off.
    pub fn after(mut self, steps: u32) -> PieceAnim {
        self.delay = steps;
        self
    }

    /// Scales the hop's speed. Above 1.0 the piece hurries.
    pub fn hurry(mut self, factor: f32) -> PieceAnim {
        self.step *= factor;
        self
    }

    /// Moves both ends of the hop through `f`. The board turning round while a piece
    /// is in the air would otherwise leave it flying to where its square used to be
    /// drawn, and popping onto the right one when it landed.
    pub fn remap(&mut self, f: impl Fn((f32, f32)) -> (f32, f32)) {
        self.from = f(self.from);
        self.to = f(self.to);
    }

    pub fn tick(&mut self) -> Option<AnimEvent> {
        if self.done {
            return None;
        }
        if self.delay > 0 {
            self.delay -= 1;
            return None;
        }
        if !self.airborne {
            self.airborne = true;
            return Some(AnimEvent::TookOff);
        }
        self.t = (self.t + self.step).min(1.0);
        if self.t >= 1.0 {
            self.done = true;
            return Some(AnimEvent::Landed);
        }
        None
    }

    /// True while the piece belongs on screen.
    ///
    /// A piece that has landed keeps being drawn where it landed: its square stays
    /// masked out until every piece in the batch has arrived, so letting go of it here
    /// would blink it out of existence until the last one lands. A piece that has been
    /// taken does go: it has left the board for good.
    pub fn visible(&self) -> bool {
        self.airborne && !(self.done && self.leaving)
    }

    /// Where the piece's shadow is: the ground track, ignoring the arc.
    pub fn ground(&self) -> (f32, f32) {
        let e = ease(self.t);
        (
            self.from.0 + (self.to.0 - self.from.0) * e,
            self.from.1 + (self.to.1 - self.from.1) * e,
        )
    }

    /// How high off the ground the piece is, in pixels. Exactly zero before take-off
    /// and after landing, rather than whatever the sine leaves behind at the ends.
    pub fn height(&self) -> f32 {
        if self.delay > 0 || self.done {
            0.0
        } else {
            self.arc * (std::f32::consts::PI * self.t).sin()
        }
    }
}

/// Slow at both ends, quick in the middle: the piece looks lifted and set down rather
/// than fired.
fn ease(t: f32) -> f32 {
    if t < 0.5 {
        2.0 * t * t
    } else {
        -1.0 + (4.0 - 2.0 * t) * t
    }
}

/// The board being turned round.
///
/// A board is not turned by sliding it: it is tipped over. So that is what this
/// measures — the picture squashes to a line, and the instant it stands edge-on is
/// the instant the position underneath changes sides and comes back up the other way
/// round. Counted in logic steps like everything else here, so the turn takes the
/// same time whatever the display is doing.
pub struct BoardFlip {
    /// Progress: 0.0 lying flat one way, 1.0 lying flat the other.
    t: f32,
    /// True once the edge-on moment has gone by, so it is reported exactly once.
    turned: bool,
    pub done: bool,
}

/// Steps a turn takes. Long enough to read as a movement, short enough that a player
/// who only wanted the other view is not made to sit through it.
const FLIP_STEPS: f32 = 16.0;
/// Thinnest the board is allowed to get. Zero would blink it out of existence for a
/// step; a sliver reads as a board seen edge-on, which is the whole idea.
const EDGE_ON: f32 = 0.02;

impl BoardFlip {
    pub fn new() -> BoardFlip {
        BoardFlip { t: 0.0, turned: false, done: false }
    }

    /// Advances the turn. Returns true on the single step it passes through edge-on,
    /// which is when the board underneath should change sides.
    pub fn tick(&mut self) -> bool {
        if self.done {
            return false;
        }
        self.t = (self.t + 1.0 / FLIP_STEPS).min(1.0);
        self.done = self.t >= 1.0;
        if !self.turned && self.t >= 0.5 {
            self.turned = true;
            return true;
        }
        false
    }

    /// How much of its height the board has left. The angle turns at a constant rate,
    /// so the height is its cosine: slow at both ends and quick through the middle,
    /// the way a board tipped past the vertical falls the rest of the way on its own.
    pub fn height(&self) -> f32 {
        (std::f32::consts::PI * self.t).cos().abs().max(EDGE_ON)
    }
}

/// A banner that drops in, bounces and stays. Used for the word that ends a game:
/// something that lands with a thump reads as final in a way a fade never does.
///
/// A word that only interrupts the game rather than ending it — check — is given a
/// number of steps to linger with [`BigText::for_a_moment`], after which it drops
/// out through the bottom of the canvas and reports itself [`BigText::gone`]. It
/// falls out rather than fading, and downwards rather than back the way it came: the
/// word carries on the way it was already going, which is what the cartridge does.
pub struct BigText {
    pub text: &'static str,
    pub y: f32,
    /// Where it comes to rest.
    rest: f32,
    speed: f32,
    /// Steps still to wait before it starts falling.
    delay: u32,
    /// Bounces it has left before it settles.
    bounces: u32,
    /// Steps to hold once settled before leaving. `None` stays for good.
    linger: Option<u32>,
    /// On its way back up and out.
    leaving: bool,
    /// Off the top of the canvas, and finished with.
    gone: bool,
}

/// Downward acceleration, in pixels per step per step.
const GRAVITY: f32 = 0.55;
/// How much of its speed a bounce gives back.
const BOUNCE: f32 = 0.42;
/// Where a banner starts, and the line past which one on its way out is finished.
const OFFSCREEN: f32 = -24.0;

impl BigText {
    /// Drops `text` from above the canvas onto `rest`.
    pub fn new(text: &'static str, rest: f32, delay: u32) -> BigText {
        debug_assert!(rest > OFFSCREEN, "it would rest above the canvas");
        BigText {
            text,
            y: OFFSCREEN,
            rest,
            speed: 0.0,
            delay,
            bounces: 2,
            linger: None,
            leaving: false,
            gone: false,
        }
    }

    /// Makes the banner a passing one: it holds for `steps` once it has settled, then
    /// climbs back out of sight.
    pub fn for_a_moment(mut self, steps: u32) -> BigText {
        self.linger = Some(steps);
        self
    }

    /// True once a passing banner has left the canvas and can be dropped.
    pub fn gone(&self) -> bool {
        self.gone
    }

    /// Advances the fall. Returns true on the step it strikes, which is when the
    /// screen should be knocked.
    pub fn tick(&mut self) -> bool {
        if self.delay > 0 {
            self.delay -= 1;
            return false;
        }
        if self.leaving {
            // It simply falls again, from where it stopped: the word finishes the
            // journey it started rather than reversing out of it.
            self.speed += GRAVITY;
            self.y += self.speed;
            self.gone = self.y > FB_H as f32;
            return false;
        }
        if self.settled() {
            match self.linger {
                Some(0) => self.leaving = true,
                Some(ref mut steps) => *steps -= 1,
                None => {}
            }
            return false;
        }
        self.speed += GRAVITY;
        self.y += self.speed;
        if self.y < self.rest {
            return false;
        }
        self.y = self.rest;
        if self.bounces > 0 {
            self.bounces -= 1;
            self.speed = -self.speed * BOUNCE;
        } else {
            self.speed = 0.0;
        }
        true
    }

    pub fn settled(&self) -> bool {
        self.bounces == 0 && self.speed == 0.0 && self.y >= self.rest
    }
}

/// A decaying screen shake. Landings and announcements add to it; it always settles.
#[derive(Default)]
pub struct Shake {
    amount: f32,
}

impl Shake {
    /// Adds a knock. Amounts are small: 0.1 is a piece landing, 0.3 is checkmate.
    pub fn add(&mut self, amount: f32) {
        self.amount = (self.amount + amount).min(1.0);
    }

    /// Advances the decay and returns the camera offset for this step.
    pub fn tick(&mut self, rng: &mut Rng) -> (i32, i32) {
        if self.amount < 0.02 {
            self.amount = 0.0;
            return (0, 0);
        }
        let reach = 6.0 * self.amount;
        let offset = (
            (rng.next_f32() * 2.0 - 1.0) * reach,
            (rng.next_f32() * 2.0 - 1.0) * reach,
        );
        self.amount *= 0.9;
        (offset.0.round() as i32, offset.1.round() as i32)
    }
}

/// A small deterministic generator, so the jitter and the staggered entrance do not
/// depend on the platform having one.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed | 1)
    }

    pub fn next_u32(&mut self) -> u32 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
    }

    /// A full 64-bit draw, for seeding something that wants one.
    pub fn next_u64(&mut self) -> u64 {
        (self.next_u32() as u64) << 32 | self.next_u32() as u64
    }

    pub fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }

    /// A whole number in `0..bound`.
    pub fn below(&mut self, bound: u32) -> u32 {
        debug_assert!(bound > 0);
        self.next_u32() % bound
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Piece;

    fn run_to_completion(anim: &mut PieceAnim) -> (u32, Vec<AnimEvent>) {
        let mut steps = 0;
        let mut events = Vec::new();
        while !anim.done {
            if let Some(e) = anim.tick() {
                events.push(e);
            }
            steps += 1;
            assert!(steps < 1000, "animation never finished");
        }
        (steps, events)
    }

    #[test]
    fn a_hop_starts_and_ends_where_it_was_told() {
        let mut anim = PieceAnim::new(Piece::WHITE_PAWN, (10.0, 20.0), (50.0, 20.0), 2.0);
        assert_eq!(anim.ground(), (10.0, 20.0));
        assert_eq!(anim.height(), 0.0);
        let (_, events) = run_to_completion(&mut anim);
        assert_eq!(events, vec![AnimEvent::TookOff, AnimEvent::Landed]);
        let (x, y) = anim.ground();
        assert!((x - 50.0).abs() < 0.01 && (y - 20.0).abs() < 0.01);
        assert_eq!(anim.height(), 0.0);
        assert!(anim.done);
    }

    #[test]
    fn the_arc_peaks_in_the_middle() {
        let mut anim = PieceAnim::new(Piece::WHITE_PAWN, (0.0, 0.0), (100.0, 0.0), 4.0);
        let mut peak = 0.0f32;
        let mut peak_at = 0.0;
        while !anim.done {
            anim.tick();
            if anim.height() > peak {
                peak = anim.height();
                peak_at = anim.t;
            }
        }
        assert!(peak > 15.0, "a four-square hop should rise noticeably");
        assert!((peak_at - 0.5).abs() < 0.1, "peak at t={peak_at}");
    }

    #[test]
    fn a_landed_piece_stays_on_screen_but_a_taken_one_leaves() {
        let waiting = PieceAnim::new(Piece::WHITE_ROOK, (0.0, 0.0), (40.0, 0.0), 2.0).after(5);
        assert!(!waiting.visible(), "it has not set off yet");

        let mut moved = PieceAnim::new(Piece::WHITE_ROOK, (0.0, 0.0), (40.0, 0.0), 2.0);
        let mut taken = PieceAnim::new(Piece::BLACK_ROOK, (0.0, 0.0), (40.0, 0.0), 2.0).taken();
        run_to_completion(&mut moved);
        run_to_completion(&mut taken);
        assert!(moved.done && taken.done);
        // Its square is still masked out while the rest of the batch is in the air, so
        // dropping it here would blink it out until the last piece lands.
        assert!(moved.visible(), "it is standing where it landed");
        assert!(!taken.visible(), "it has been carried off the board");
    }

    #[test]
    fn a_delay_holds_the_piece_on_the_ground() {
        let mut anim = PieceAnim::new(Piece::WHITE_ROOK, (0.0, 0.0), (40.0, 0.0), 2.0).after(10);
        for _ in 0..10 {
            assert_eq!(anim.tick(), None);
            assert_eq!(anim.ground(), (0.0, 0.0));
            assert_eq!(anim.height(), 0.0);
            assert!(!anim.visible());
        }
        assert_eq!(anim.tick(), Some(AnimEvent::TookOff));
        assert!(anim.visible());
    }

    #[test]
    fn a_turn_goes_over_edge_on_once_and_comes_back_up() {
        let mut flip = BoardFlip::new();
        assert_eq!(flip.height(), 1.0, "it starts lying flat");
        let mut edge_on = Vec::new();
        let mut thinnest = f32::MAX;
        let mut steps = 0;
        while !flip.done {
            let turned = flip.tick();
            steps += 1;
            assert!(steps < 100, "the turn never finished");
            thinnest = thinnest.min(flip.height());
            if turned {
                edge_on.push(steps);
                assert!(flip.height() < 0.2, "it turns over while it is edge-on");
            }
        }
        assert_eq!(edge_on.len(), 1, "the board changes sides exactly once");
        assert!(thinnest < 0.05, "it never stood on its edge: {thinnest}");
        assert!(flip.height() > 0.99, "it ends lying flat again");
        assert!(!flip.tick(), "a finished turn has nothing left to report");
    }

    #[test]
    fn a_hop_can_be_turned_round_with_the_board() {
        let mut anim = PieceAnim::new(Piece::WHITE_PAWN, (10.0, 20.0), (50.0, 20.0), 2.0);
        anim.remap(|(x, y)| (100.0 - x, 100.0 - y));
        assert_eq!(anim.ground(), (90.0, 80.0));
        run_to_completion(&mut anim);
        let (x, y) = anim.ground();
        assert!((x - 50.0).abs() < 0.01 && (y - 80.0).abs() < 0.01);
    }

    #[test]
    fn a_banner_falls_bounces_and_stops() {
        let mut banner = BigText::new("checkmate", 60.0, 5);
        for _ in 0..5 {
            assert!(!banner.tick(), "it should not move during its delay");
            assert_eq!(banner.y, -24.0);
        }
        let mut strikes = 0;
        for _ in 0..600 {
            if banner.tick() {
                strikes += 1;
                assert_eq!(banner.y, 60.0, "a strike happens at the resting line");
            }
            if banner.settled() {
                break;
            }
        }
        assert_eq!(strikes, 3, "two bounces after the first landing");
        assert!(banner.settled());
        assert_eq!(banner.y, 60.0);
    }

    #[test]
    fn shake_always_settles() {
        let mut rng = Rng::new(1);
        let mut shake = Shake::default();
        shake.add(1.0);
        let mut steps = 0;
        loop {
            let (dx, dy) = shake.tick(&mut rng);
            if (dx, dy) == (0, 0) && shake.amount == 0.0 {
                break;
            }
            steps += 1;
            assert!(steps < 200, "shake never settled");
        }
    }

    #[test]
    fn the_generator_is_deterministic_and_in_range() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
        let mut rng = Rng::new(7);
        for _ in 0..1000 {
            let f = rng.next_f32();
            assert!((0.0..=1.0).contains(&f));
            assert!(rng.below(8) < 8);
        }
    }
}
