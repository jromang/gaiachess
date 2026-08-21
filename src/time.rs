//! Reading the clock, on platforms that have one and on one that does not.
//!
//! `std::time::Instant::now()` **panics** on `wasm32-unknown-unknown`: the target has no
//! clock of its own, and with `panic = "abort"` that takes the whole instance down. The
//! time therefore comes in from the host, through the same `env` import table the rest
//! of the bridge uses.
//!
//! `web-time` would do this too, but by way of `wasm-bindgen`, whose generated
//! JavaScript neither of the two web modules loads — the interface is brought up by
//! miniquad's own loader and the engine by a hand-written one. One more imported
//! function is a smaller thing to carry than a second, incompatible toolchain.
//!
//! Only the bare `wasm32-unknown-unknown` target needs this. `wasm32-wasip1`, which is
//! how the test suite is run against the vector backend, has a clock of its own and
//! must keep using it — nothing provides the import there.

#[cfg(not(all(target_arch = "wasm32", not(target_os = "wasi"))))]
pub use std::time::Instant;

#[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
pub use wasm::Instant;

/// A number nobody can predict, for the one place that wants to differ between runs:
/// the opening book, which must not open the same game twice.
#[cfg(not(all(target_arch = "wasm32", not(target_os = "wasi"))))]
pub fn seed_from_clock() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |since| since.as_nanos() as u64)
}

#[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
pub fn seed_from_clock() -> u64 {
    // Milliseconds rather than nanoseconds, and fractional: whatever resolution the host
    // is willing to give. It only has to differ between launches.
    (wasm::now_ms() * 1000.0) as u64
}

#[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
mod wasm {
    use std::time::Duration;

    #[link(wasm_import_module = "env")]
    unsafe extern "C" {
        /// Milliseconds since some fixed point the host picks. Only differences are
        /// read, so the origin does not matter, but it must not run backwards.
        fn gaia_now_ms() -> f64;
    }

    pub fn now_ms() -> f64 {
        unsafe { gaia_now_ms() }
    }

    /// Stands in for `std::time::Instant`, with the little of it the engine uses.
    #[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
    pub struct Instant(f64);

    impl Instant {
        pub fn now() -> Instant {
            Instant(now_ms())
        }

        pub fn elapsed(&self) -> Duration {
            Instant::now().duration_since(*self)
        }

        /// Saturates at zero rather than panicking, which is what `std` does too: a host
        /// clock that jumps backwards should not take the engine with it.
        pub fn duration_since(&self, earlier: Instant) -> Duration {
            Duration::from_secs_f64((self.0 - earlier.0).max(0.0) / 1000.0)
        }
    }
}
