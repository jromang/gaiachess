//! Sound playback.
//!
//! Scenes never play a sound directly; they drop a request into a [`Queue`] which the
//! application drains once a frame. That keeps the scenes free of the audio backend
//! and testable without one, and it means a sound asked for twice in the same logic
//! step is still one request.
//!
//! Playback goes through `rodio` rather than the window library's own audio, which
//! starts every sound at the beginning of its 93 ms output buffer: two blips asked for
//! 20 ms apart came out together, at double the amplitude, and a fast run of them —
//! scrolling a menu, the pieces landing at the start of a game — collapsed into one
//! loud smear. Here each sound is mixed from the sample it was handed over on.
//!
//! The clips themselves are not files. They are made at start-up by [`super::synth`],
//! from a few numbers each.


use super::synth;

/// Every sound the interface can make. What each one sounds like is in
/// [`super::synth`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Sfx {
    /// The cursor moving from square to square.
    Cursor,
    /// Backing out of something.
    Cancel,
    /// A menu setting changing.
    Confirm,
    /// A piece being picked up.
    Select,
    /// A piece leaving the ground.
    Jump,
    /// A piece touching down.
    Land,
    /// A piece being taken.
    Capture,
    /// One character of the status line appearing.
    Type,
    /// A menu being opened or a game starting.
    Action,
    /// A check.
    JingleCheck,
    /// The end of a game.
    JingleMate,
}

impl Sfx {
    /// Every sound, in the order the enum declares them — which is the order the clips
    /// are rendered and then indexed in, so the two cannot drift apart.
    pub const EVERY: [Sfx; 11] = [
        Sfx::Cursor,
        Sfx::Cancel,
        Sfx::Confirm,
        Sfx::Select,
        Sfx::Jump,
        Sfx::Land,
        Sfx::Capture,
        Sfx::Type,
        Sfx::Action,
        Sfx::JingleCheck,
        Sfx::JingleMate,
    ];
}

/// How loud each clip plays. The blips fire constantly and the jingles once, so they
/// cannot share a level without one drowning the other out.
fn volume(sfx: Sfx) -> f32 {
    match sfx {
        Sfx::Type => 0.18,
        Sfx::Cursor => 0.35,
        Sfx::Jump | Sfx::Land => 0.4,
        Sfx::JingleCheck | Sfx::JingleMate => 0.8,
        _ => 0.55,
    }
}

/// Sounds asked for during a logic step, drained by the application each frame.
#[derive(Default)]
pub struct Queue(Vec<Sfx>);

impl Queue {
    pub fn push(&mut self, sfx: Sfx) {
        // Two of the same sound in one step would only phase against each other.
        if !self.0.contains(&sfx) {
            self.0.push(sfx);
        }
    }

    pub fn drain(&mut self) -> std::vec::Drain<'_, Sfx> {
        self.0.drain(..)
    }
}

// The desktop backend. Everything above this point is backend-free, which is what
// lets the browser build reuse it whole. Haiku is desktop too but rodio has no
// backend there; it plays through the host backend below, like the browser.
#[cfg(all(feature = "gui", not(target_os = "haiku")))]
mod desktop {
    use super::{Sfx, synth, volume};
    use rodio::static_buffer::StaticSamplesBuffer;
    use rodio::{ChannelCount, DeviceSinkBuilder, MixerDeviceSink, SampleRate, Source};

    /// One rendered clip, ready to be handed to the mixer as often as wanted.
    ///
    /// The samples are made once and never freed, so playing a sound hands over a slice
    /// rather than copying the audio — which matters when thirty-two pieces land in half a
    /// second. They would live until the process ends in any case.
    struct Clip {
        samples: &'static [f32],
    }

    impl Clip {
        fn source(&self) -> StaticSamplesBuffer {
            StaticSamplesBuffer::new(MONO, RATE, self.samples)
        }
    }

    /// The synth writes one channel, at its own rate; rodio resamples to whatever the
    /// device wants.
    const MONO: ChannelCount = ChannelCount::new(1).unwrap();
    const RATE: SampleRate = SampleRate::new(synth::RATE).unwrap();

    pub struct Audio {
        /// Dropping this closes the device and silences everything, so it is kept for as
        /// long as the interface runs even though nothing reads it directly.
        _device: MixerDeviceSink,
        clips: Vec<Clip>,
    }

    impl Audio {
        /// Opens the audio device and renders every clip.
        ///
        /// Returns `None` when there is no device to open, which is not an error worth
        /// stopping for: a machine with no sound should still be able to play chess.
        pub fn load() -> Option<Audio> {
            let device = match DeviceSinkBuilder::open_default_sink() {
                Ok(device) => device,
                Err(err) => {
                    eprintln!("no audio device, playing without sound: {err}");
                    return None;
                }
            };
            let clips = Sfx::EVERY
                .iter()
                .map(|&sfx| Clip { samples: Vec::leak(synth::clip(sfx)) })
                .collect();
            Some(Audio { _device: device, clips })
        }

        pub fn play(&self, sfx: Sfx) {
            if let Some(clip) = self.clips.get(sfx as usize) {
                self._device
                    .mixer()
                    .add(clip.source().amplify(volume(sfx)));
            }
        }
    }

}

#[cfg(all(feature = "gui", not(target_os = "haiku")))]
pub use desktop::Audio;

// The host backend, shared by the two builds whose mixer lives outside this crate. No
// audio crate at all: the samples are already made here, and the host can schedule
// them to the sample, which is the very thing the desktop backend went to rodio for.
// In the browser the two functions come from the page (Web Audio; keeping `quad-snd`
// out of the workspace, whose `links = "alsa"` cannot coexist with the one rodio
// brings); on Haiku they come from the native shim's BSoundPlayer mixer.
#[cfg(any(
    all(feature = "gui-core", not(feature = "gui")),
    all(feature = "gui", target_os = "haiku")
))]
mod host {
    use super::{Sfx, synth, volume};

    #[cfg_attr(target_arch = "wasm32", link(wasm_import_module = "env"))]
    unsafe extern "C" {
        /// Hands the host one rendered clip, to keep until the page goes.
        fn gaia_sfx_register(id: u32, samples: *const f32, len: usize, rate: u32);
        /// Plays a registered clip. Overlapping calls are the host's problem, and Web
        /// Audio makes it a non-problem.
        fn gaia_sfx_play(id: u32, gain: f32);
    }

    pub struct Audio;

    impl Audio {
        /// Renders every clip and hands it over. There is no device to fail to open:
        /// a host with no sound simply does nothing with what it is given.
        pub fn load() -> Option<Audio> {
            for (id, &sfx) in Sfx::EVERY.iter().enumerate() {
                let samples = synth::clip(sfx);
                // The host copies during the call; nothing here has to outlive it.
                unsafe {
                    gaia_sfx_register(id as u32, samples.as_ptr(), samples.len(), synth::RATE);
                }
            }
            Some(Audio)
        }

        pub fn play(&self, sfx: Sfx) {
            unsafe { gaia_sfx_play(sfx as u32, volume(sfx)) };
        }
    }
}

#[cfg(any(
    all(feature = "gui-core", not(feature = "gui")),
    all(feature = "gui", target_os = "haiku")
))]
pub use host::Audio;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_sound_is_listed_once_and_in_order() {
        // `EVERY` is indexed by casting the enum, so a sound in the wrong place would
        // play as another one — a silent kind of wrong that no other test would catch.
        for (i, sfx) in Sfx::EVERY.iter().enumerate() {
            assert_eq!(*sfx as usize, i, "{sfx:?} is listed out of order");
        }
        assert_eq!(Sfx::EVERY.len(), Sfx::JingleMate as usize + 1);
    }

    #[test]
    fn a_repeat_within_one_step_is_dropped() {
        let mut queue = Queue::default();
        queue.push(Sfx::Cursor);
        queue.push(Sfx::Cursor);
        queue.push(Sfx::Land);
        assert_eq!(queue.drain().collect::<Vec<_>>(), vec![Sfx::Cursor, Sfx::Land]);
    }

    #[test]
    fn draining_empties_the_queue() {
        let mut queue = Queue::default();
        queue.push(Sfx::Land);
        assert_eq!(queue.drain().count(), 1);
        assert_eq!(queue.drain().count(), 0);
    }
}
