//! Chess clocks.
//!
//! The clocks run on real time rather than on logic steps: a player who is thinking
//! is spending seconds, not frames, and the engine's own budget is quoted to it in
//! milliseconds. They keep running while the engine thinks, which is the point.

use crate::time::Instant;

use super::lang::Key;
use crate::types::Color;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TimeControl {
    /// No clocks at all.
    Unlimited,
    /// A starting time plus an increment for each move played.
    Fischer { base_s: u32, inc_s: u32 },
}

impl TimeControl {
    /// The choices offered in the menu, from quickest to slowest.
    pub const CHOICES: [TimeControl; 5] = [
        TimeControl::Fischer { base_s: 60, inc_s: 1 },
        TimeControl::Fischer { base_s: 180, inc_s: 2 },
        TimeControl::Fischer { base_s: 300, inc_s: 0 },
        TimeControl::Fischer { base_s: 600, inc_s: 5 },
        TimeControl::Unlimited,
    ];

    pub fn label(self) -> Label {
        match self {
            TimeControl::Fischer { base_s: 60, .. } => Label::Figures("1+1"),
            TimeControl::Fischer { base_s: 180, .. } => Label::Figures("3+2"),
            TimeControl::Fischer { base_s: 300, .. } => Label::Figures("5+0"),
            TimeControl::Fischer { base_s: 600, .. } => Label::Figures("10+5"),
            TimeControl::Fischer { .. } => Label::Word(Key::ClockCustom),
            TimeControl::Unlimited => Label::Word(Key::ClockNone),
        }
    }
}

/// What the clock row reads as.
///
/// A cadence written in figures says the same thing in every language and is left
/// alone; the two choices that are words are not, and are looked up like the rest of
/// the interface.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Label {
    Figures(&'static str),
    Word(Key),
}

pub struct Clocks {
    control: TimeControl,
    /// Milliseconds left for each side.
    left: [f64; 2],
    /// When the running clock was last read.
    last: Option<Instant>,
}

impl Clocks {
    pub fn new(control: TimeControl) -> Clocks {
        let start = match control {
            TimeControl::Unlimited => 0.0,
            TimeControl::Fischer { base_s, .. } => base_s as f64 * 1000.0,
        };
        Clocks {
            control,
            left: [start, start],
            last: None,
        }
    }

    pub fn enabled(&self) -> bool {
        self.control != TimeControl::Unlimited
    }

    /// Charges elapsed real time to `side` while `running`, and reports whoever has
    /// just run out. Stopping and restarting is safe: time only accrues between two
    /// consecutive running calls.
    pub fn tick(&mut self, side: Color, running: bool) -> Option<Color> {
        if !self.enabled() || !running {
            self.last = None;
            return None;
        }
        let now = Instant::now();
        if let Some(previous) = self.last {
            let elapsed = now.duration_since(previous).as_secs_f64() * 1000.0;
            let left = &mut self.left[side as usize];
            *left -= elapsed;
            if *left <= 0.0 {
                *left = 0.0;
                self.last = Some(now);
                return Some(side);
            }
        }
        self.last = Some(now);
        None
    }

    /// Adds the increment after `mover` has completed a move.
    pub fn on_move_played(&mut self, mover: Color) {
        if let TimeControl::Fischer { inc_s, .. } = self.control {
            self.left[mover as usize] += inc_s as f64 * 1000.0;
        }
    }

    /// Milliseconds left, for handing the engine a budget.
    pub fn remaining_ms(&self, side: Color) -> u64 {
        self.left[side as usize].max(0.0) as u64
    }

    pub fn increment_ms(&self) -> u64 {
        match self.control {
            TimeControl::Unlimited => 0,
            TimeControl::Fischer { inc_s, .. } => inc_s as u64 * 1000,
        }
    }

    /// The clock as it should be shown: minutes and seconds, or tenths under ten
    /// seconds, where a tenth is what the player is actually watching.
    pub fn display(&self, side: Color) -> String {
        let ms = self.remaining_ms(side);
        if ms < 10_000 {
            format!("{}.{}", ms / 1000, (ms % 1000) / 100)
        } else {
            format!("{}:{:02}", ms / 60_000, (ms % 60_000) / 1000)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;
    use std::time::Duration;

    #[test]
    fn unlimited_clocks_never_run_out() {
        let mut clocks = Clocks::new(TimeControl::Unlimited);
        assert!(!clocks.enabled());
        sleep(Duration::from_millis(5));
        assert_eq!(clocks.tick(Color::White, true), None);
    }

    #[test]
    fn only_the_side_to_move_is_charged() {
        let mut clocks = Clocks::new(TimeControl::Fischer { base_s: 10, inc_s: 0 });
        clocks.tick(Color::White, true);
        sleep(Duration::from_millis(20));
        clocks.tick(Color::White, true);
        assert!(clocks.remaining_ms(Color::White) < 10_000);
        assert_eq!(clocks.remaining_ms(Color::Black), 10_000);
    }

    #[test]
    fn a_stopped_clock_is_not_charged_for_the_pause() {
        let mut clocks = Clocks::new(TimeControl::Fischer { base_s: 10, inc_s: 0 });
        clocks.tick(Color::White, true);
        sleep(Duration::from_millis(30));
        // The pause is what happens between two moves; it must cost nobody anything.
        clocks.tick(Color::White, false);
        clocks.tick(Color::White, true);
        assert_eq!(clocks.remaining_ms(Color::White), 10_000);
    }

    #[test]
    fn running_out_is_reported_once_the_time_is_gone() {
        let mut clocks = Clocks::new(TimeControl::Fischer { base_s: 0, inc_s: 0 });
        clocks.left = [5.0, 5.0];
        clocks.tick(Color::White, true);
        sleep(Duration::from_millis(20));
        assert_eq!(clocks.tick(Color::White, true), Some(Color::White));
        assert_eq!(clocks.remaining_ms(Color::White), 0);
    }

    #[test]
    fn the_increment_lands_on_the_player_who_moved() {
        let mut clocks = Clocks::new(TimeControl::Fischer { base_s: 60, inc_s: 2 });
        clocks.on_move_played(Color::Black);
        assert_eq!(clocks.remaining_ms(Color::Black), 62_000);
        assert_eq!(clocks.remaining_ms(Color::White), 60_000);
    }

    #[test]
    fn the_display_switches_to_tenths_when_it_matters() {
        let mut clocks = Clocks::new(TimeControl::Fischer { base_s: 600, inc_s: 0 });
        assert_eq!(clocks.display(Color::White), "10:00");
        clocks.left[0] = 65_400.0;
        assert_eq!(clocks.display(Color::White), "1:05");
        clocks.left[0] = 9_400.0;
        assert_eq!(clocks.display(Color::White), "9.4");
    }

    #[test]
    fn every_offered_control_has_its_own_label() {
        let mut seen = Vec::new();
        for tc in TimeControl::CHOICES {
            let label = tc.label();
            assert_ne!(
                label,
                Label::Word(Key::ClockCustom),
                "a menu choice has no label of its own"
            );
            assert!(!seen.contains(&label), "two choices share the label {label:?}");
            seen.push(label);
        }
    }
}
