//! The interface's sounds, made rather than recorded.
//!
//! Every sound is described here as a handful of numbers and rendered once at start-up,
//! so the binary carries no audio at all. The descriptions are written in hertz and
//! seconds rather than in the generator's own units, which are neither: `pitch` and
//! `stage` do that conversion, and the tests below check the conversion rather than
//! trusting it.

use sfxr::{Generator, Sample, WaveType};

use super::audio::Sfx;

/// Samples per second the clips are rendered at, and the rate the mixer is told.
pub const RATE: u32 = 44100;

/// The oscillator is stepped eight times per output sample and averaged, so it runs at
/// eight times the output rate — which is what puts a blip in the kilohertz rather than
/// down in the bass.
const OSC_RATE: f32 = 8.0 * RATE as f32;

/// A pitch in hertz, as the generator wants it.
///
/// It counts in oscillator periods: `period = 100 / (base_freq² + 0.001)` steps of the
/// oscillator. Inverting that lets the sounds below be written in pitches one can hum.
fn pitch(hz: f32) -> f64 {
    let scale = OSC_RATE / 100.0;
    ((hz / scale - 0.001).max(0.0)).sqrt() as f64
}

/// A duration in seconds, as one of the envelope's three stages.
///
/// A stage lasts `value² × 100000` samples, so the useful range is short: a tenth of a
/// second is 0.21, and the whole scale runs out at a little over two seconds.
fn stage(seconds: f32) -> f32 {
    (seconds * RATE as f32 / 100_000.0).sqrt().min(1.0)
}

/// One note: a waveform, a pitch, how long it holds and how long it takes to die away.
///
/// Attack is left at zero throughout. These are clicks and blips, and anything that
/// fades in reads as a mistake at this length.
fn note(wave: WaveType, hz: f32, hold: f32, fall: f32) -> Sample {
    let mut s = Sample::new();
    s.wave_type = wave;
    s.base_freq = pitch(hz);
    // Half duty: the hollow, even square of a handheld rather than a thin pulse.
    s.duty = 0.0;
    s.env_attack = 0.0;
    s.env_sustain = stage(hold);
    s.env_decay = stage(fall);
    s
}

/// Renders one sound to samples, and trims the silence the envelope leaves behind.
fn render(sample: Sample) -> Vec<f32> {
    // The envelope's three stages give the length exactly, so nothing has to be guessed
    // at; a little tail is added for the filters to settle in.
    let stages = sample.env_attack.powi(2) + sample.env_sustain.powi(2) + sample.env_decay.powi(2);
    let len = (stages * 100_000.0) as usize + RATE as usize / 100;
    let mut out = vec![0.0f32; len];
    let mut generator = Generator::new(sample);
    // The generator's own volume is left wide open and the result normalised instead,
    // so that every clip arrives at the same loudness whatever it is made of.
    generator.volume = 1.0;
    generator.generate(&mut out);
    normalise(&mut out);
    out
}

/// Brings a clip to a fixed peak, so the levels in `audio::volume` mean the same thing
/// from one sound to the next.
fn normalise(samples: &mut [f32]) {
    const PEAK: f32 = 0.86;
    let loudest = samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    if loudest > 0.0 {
        let scale = PEAK / loudest;
        for s in samples.iter_mut() {
            *s *= scale;
        }
    }
}

/// Plays a run of notes one after another, for the two tunes.
fn tune(wave: WaveType, notes: &[(f32, f32)]) -> Vec<f32> {
    let mut out = Vec::new();
    for &(hz, seconds) in notes {
        // A third of each note is its own decay, which keeps a run of them from sounding
        // like one held chord.
        out.extend(render(note(wave, hz, seconds * 0.7, seconds * 0.3)));
    }
    normalise(&mut out);
    out
}

/// The sound of each event.
///
/// Pitches are chosen so that things that answer each other are a fifth or an octave
/// apart — picking a piece up and putting it down, backing out and confirming — and so
/// that the sounds heard most often sit highest and shortest, where they stay out of
/// the way.
pub fn clip(sfx: Sfx) -> Vec<f32> {
    use WaveType::{Noise, Sine, Square, Triangle};
    match sfx {
        // The hand crossing a square: the most frequent sound in the game by far, so it
        // is barely a tick.
        Sfx::Cursor => render(note(Square, 1320.0, 0.008, 0.02)),
        // Backing out: the same tick, dropped an octave and given a downward slide.
        Sfx::Cancel => {
            let mut s = note(Square, 660.0, 0.02, 0.05);
            s.freq_ramp = -0.3;
            render(s)
        }
        // A setting changing, answering `Cancel` a fifth above.
        Sfx::Confirm => render(note(Square, 990.0, 0.02, 0.05)),
        // A piece leaving its square, with a flick upwards.
        Sfx::Select => {
            let mut s = note(Square, 880.0, 0.015, 0.05);
            s.freq_ramp = 0.25;
            render(s)
        }
        // Leaving the ground, and landing back on it: the same gesture, mirrored.
        Sfx::Jump => {
            let mut s = note(Triangle, 520.0, 0.02, 0.06);
            s.freq_ramp = 0.3;
            render(s)
        }
        Sfx::Land => {
            let mut s = note(Triangle, 440.0, 0.015, 0.06);
            s.freq_ramp = -0.35;
            s.env_punch = 0.4;
            render(s)
        }
        // A piece taken: the one sound with grit in it.
        Sfx::Capture => {
            let mut s = note(Noise, 900.0, 0.03, 0.12);
            s.env_punch = 0.6;
            s.hpf_freq = 0.15;
            render(s)
        }
        // One character of the status line: quieter and shorter than anything else,
        // because it fires letter by letter.
        Sfx::Type => render(note(Square, 1980.0, 0.004, 0.012)),
        // A menu opening or a game starting: long enough to feel like a decision.
        Sfx::Action => {
            let mut s = note(Square, 660.0, 0.05, 0.12);
            s.freq_ramp = 0.2;
            s.arp_speed = 0.6;
            s.arp_mod = 0.4;
            render(s)
        }
        // Check: three notes climbing, unresolved — something is coming.
        Sfx::JingleCheck => tune(Square, &[(659.3, 0.10), (880.0, 0.10), (1174.7, 0.18)]),
        // Mate: a fall onto the tonic, and it stays there.
        Sfx::JingleMate => tune(
            Sine,
            &[(880.0, 0.12), (740.0, 0.12), (659.3, 0.12), (440.0, 0.40)],
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dominant frequency, by counting how often the wave crosses zero. Crude, but a
    /// square wave crosses twice a period and that is all this needs to know.
    fn measured_hz(samples: &[f32]) -> f32 {
        let loud: Vec<f32> = samples.iter().copied().filter(|s| s.abs() > 0.02).collect();
        let crossings = loud.windows(2).filter(|w| (w[0] < 0.0) != (w[1] < 0.0)).count();
        crossings as f32 * RATE as f32 / (2.0 * loud.len().max(1) as f32)
    }

    #[test]
    fn a_pitch_comes_out_at_the_pitch_it_was_asked_for() {
        // The conversion into the generator's units is the one piece of arithmetic here
        // that cannot be checked by eye, and getting it wrong moves every sound an
        // octave without failing anything else.
        for hz in [440.0, 880.0, 1320.0] {
            let rendered = render(note(WaveType::Square, hz, 0.2, 0.0));
            let got = measured_hz(&rendered);
            assert!(
                (got - hz).abs() < hz * 0.1,
                "asked for {hz} Hz, measured {got} Hz"
            );
        }
    }

    #[test]
    fn a_duration_comes_out_at_the_length_it_was_asked_for() {
        for seconds in [0.05, 0.2, 0.5] {
            let rendered = render(note(WaveType::Square, 880.0, seconds, 0.0));
            let got = rendered.len() as f32 / RATE as f32;
            assert!(
                (got - seconds).abs() < seconds * 0.2 + 0.02,
                "asked for {seconds}s, got {got}s"
            );
        }
    }

    #[test]
    fn every_sound_is_audible_and_in_range() {
        for sfx in Sfx::EVERY {
            let rendered = clip(sfx);
            assert!(!rendered.is_empty(), "{sfx:?} renders to nothing");
            let peak = rendered.iter().fold(0.0f32, |m, s| m.max(s.abs()));
            assert!((0.8..=1.0).contains(&peak), "{sfx:?} peaks at {peak}");
            // Silence would pass the peak check on one stray sample; this catches a
            // clip that is a click and nothing else.
            let loud = rendered.iter().filter(|s| s.abs() > 0.1).count();
            assert!(loud > 32, "{sfx:?} is {loud} loud samples, effectively silent");
        }
    }

    #[test]
    fn the_sounds_that_fire_constantly_are_the_short_ones() {
        // A blip that outlasts the gap between two of them turns a fast scroll into a
        // drone, which is what this keeps an eye on.
        let short = clip(Sfx::Cursor).len() as f32 / RATE as f32;
        assert!(short < 0.05, "the cursor blip lasts {short}s");
        assert!(clip(Sfx::Type).len() < clip(Sfx::Cursor).len());
        assert!(clip(Sfx::JingleMate).len() > clip(Sfx::Action).len());
    }
}
