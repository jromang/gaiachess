//! The interface as a WebAssembly module.
//!
//! A separate crate rather than a second binary of the engine's, and not by choice: the
//! desktop build takes its sound through rodio, whose ALSA bindings declare the same
//! `links` name as the ones macroquad's audio would bring. Cargo allows only one of those
//! in a dependency graph, and the check is structural — declaring both breaks resolution
//! even for a build that activates neither. Keeping the browser interface in its own
//! crate sidesteps the question entirely, and it gets its sound through the host anyway.
fn main() {
    gaiachess::gui::run(None);
}

/// The version of the bridge this module expects, checked by gl.js against the number
/// `host.js` registers with.
///
/// The seven imports the interface relies on are a contract written by hand on both
/// sides, with nothing to catch a drift between them: a renamed or re-signed function
/// would show up as a stub that quietly does nothing. Bumping this on any change to that
/// contract turns that into a console error naming both versions.
///
/// 2 added `gaia_locale`, which the interface opens in the reader's language with.
#[unsafe(no_mangle)]
pub extern "C" fn gaia_crate_version() -> u32 {
    2
}
