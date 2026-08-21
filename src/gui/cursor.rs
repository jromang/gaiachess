//! Which square the hand means, how it bobs, and how it walks there.
//!
//! The keyboard steps it one square at a time and the engine walks it, which is what
//! turns a silent pause for thinking into something to watch. The mouse does not walk
//! anything: it puts the hand straight where the player is already looking.

use super::input::Dir;
use crate::types::Square;

/// Length of the bob cycle, in logic steps.
const CYCLE: u32 = 39;
/// Step of the cycle at which the hand lifts.
const LIFT_AT: u32 = 29;
/// Steps between blinks of the threatened-piece outline.
const BLINK: u32 = 8;

/// What the cursor did on a step.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Step {
    /// It changed square, which is what the cursor blip follows.
    pub moved: bool,
    /// A walk reached its target on this step, which counts as a press.
    pub arrived: bool,
}

struct Walk {
    target: Square,
    /// Steps left before moving to the next square.
    wait: u32,
    speed: u32,
}

pub struct Cursor {
    pub sq: Square,
    phase: u32,
    walk: Option<Walk>,
}

impl Cursor {
    pub fn new(sq: Square) -> Cursor {
        Cursor {
            sq,
            phase: 0,
            walk: None,
        }
    }

    /// Advances the bob and any walk in progress, reporting what happened.
    pub fn tick(&mut self) -> Step {
        self.phase = (self.phase + 1) % CYCLE;

        let Some(walk) = self.walk.as_mut() else {
            return Step::default();
        };
        if walk.wait > 0 {
            walk.wait -= 1;
            return Step::default();
        }
        walk.wait = walk.speed;
        // Rank first, then file: an L-shaped path is easier to follow with the eye
        // than a diagonal one.
        let (from, to) = (self.sq.0 as i32, walk.target.0 as i32);
        let (fr, tr) = (from / 8, to / 8);
        let (ff, tf) = (from % 8, to % 8);
        if fr != tr {
            self.sq = Square((from + if tr > fr { 8 } else { -8 }) as u8);
        } else if ff != tf {
            self.sq = Square((from + if tf > ff { 1 } else { -1 }) as u8);
        }
        if self.sq == walk.target {
            self.walk = None;
            return Step { moved: true, arrived: true };
        }
        Step { moved: true, arrived: false }
    }

    /// Sends the hand walking to `target`, which counts as a press once it gets
    /// there. Replaces any walk already in progress.
    pub fn walk_to(&mut self, target: Square, speed: u32) {
        debug_assert!(target.0 < 64);
        if self.sq == target {
            self.walk = None;
            return;
        }
        self.walk = Some(Walk { target, wait: 0, speed });
    }

    /// Puts the hand on a square at once. The mouse does this rather than walk: it has
    /// already taken the player's eye there, so a walk would only trail behind it.
    pub fn place(&mut self, sq: Square) {
        debug_assert!(sq.0 < 64);
        self.walk = None;
        self.sq = sq;
    }

    /// Moves the hand one square, as the keyboard does. Cancels any walk, because the
    /// player has taken over.
    pub fn step(&mut self, dir: Dir, flipped: bool) {
        self.walk = None;
        let (mut df, mut dr) = dir.delta();
        if flipped {
            df = -df;
            dr = -dr;
        }
        let file = (self.sq.0 as i32 % 8 + df).clamp(0, 7);
        let rank = (self.sq.0 as i32 / 8 + dr).clamp(0, 7);
        self.sq = Square((rank * 8 + file) as u8);
    }

    /// Pixels the hand is raised by this step.
    pub fn lift(&self) -> i32 {
        i32::from(self.phase >= LIFT_AT)
    }

    /// Phase of the slow blink that marks capturable pieces.
    pub fn blink(&self) -> bool {
        (self.phase / BLINK) % 2 == 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn walking_takes_the_rank_then_the_file() {
        let mut cursor = Cursor::new(Square::A1);
        cursor.walk_to(Square::C3, 0);
        let mut seen = vec![cursor.sq];
        let mut arrived = false;
        for _ in 0..16 {
            let step = cursor.tick();
            if *seen.last().unwrap() != cursor.sq {
                seen.push(cursor.sq);
            }
            if step.arrived {
                arrived = true;
                break;
            }
        }
        assert!(arrived);
        assert_eq!(
            seen,
            vec![Square::A1, Square::A2, Square::A3, Square::B3, Square::C3]
        );
        assert!(!cursor.tick().moved, "an arrived cursor stops walking");
    }

    #[test]
    fn keyboard_steps_stay_on_the_board() {
        let mut cursor = Cursor::new(Square::A1);
        cursor.step(Dir::Left, false);
        cursor.step(Dir::Down, false);
        assert_eq!(cursor.sq, Square::A1);
        cursor.step(Dir::Right, false);
        assert_eq!(cursor.sq, Square::B1);
        cursor.step(Dir::Up, false);
        assert_eq!(cursor.sq, Square::B2);
    }

    #[test]
    fn flipping_reverses_the_keyboard() {
        let mut cursor = Cursor::new(Square::D4);
        cursor.step(Dir::Right, true);
        assert_eq!(cursor.sq, Square::C4);
        cursor.step(Dir::Up, true);
        assert_eq!(cursor.sq, Square::C3);
    }

    #[test]
    fn walking_to_the_current_square_finishes_at_once() {
        let mut cursor = Cursor::new(Square::E4);
        cursor.walk_to(Square::E4, 4);
        assert!(!cursor.tick().moved);
    }
}
