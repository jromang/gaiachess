//! Progress bars, or nothing at all.
//!
//! The bar is a convenience for someone watching a long run in a terminal, so a build
//! with no terminal to watch — the WebAssembly one — should not have to carry the
//! library that draws it. `bench` is compiled into every build, though, including that
//! one, so rather than scatter `#[cfg]` through it the dependency is hidden behind this
//! module and replaced by a set of do-nothing shims when the feature is off.
//!
//! Only what `bench` uses is shimmed. `datagen` drives the bar much harder — it reads
//! the position and elapsed time back out to build its own estimates — but it is an
//! optional feature that pulls `progress` in with it, so it always has the real thing.

#[cfg(feature = "progress")]
pub use indicatif::{ProgressBar, ProgressStyle};

#[cfg(not(feature = "progress"))]
pub use shim::{ProgressBar, ProgressStyle};

#[cfg(not(feature = "progress"))]
mod shim {
    use std::convert::Infallible;

    pub struct ProgressBar;

    impl ProgressBar {
        pub fn new(_len: u64) -> Self {
            ProgressBar
        }
        pub fn set_style(&self, _style: ProgressStyle) {}
        pub fn set_message(&self, _msg: impl Into<String>) {}
        pub fn set_position(&self, _pos: u64) {}
        pub fn finish_with_message(&self, _msg: impl Into<String>) {}
    }

    pub struct ProgressStyle;

    impl ProgressStyle {
        /// Mirrors the real signature, which is fallible because it parses a template.
        /// Nothing is parsed here, so the error type is uninhabited.
        pub fn with_template(_template: &str) -> Result<Self, Infallible> {
            Ok(ProgressStyle)
        }
        pub fn progress_chars(self, _chars: &str) -> Self {
            self
        }
    }
}
