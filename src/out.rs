//! Where the engine talks.
//!
//! Native builds print to stdout and stderr, exactly as they always did. A WebAssembly
//! build has neither: `println!` there writes into a sink that goes nowhere, and it does
//! so **without any error at compile time**, so a forgotten macro is a line that
//! silently disappears. Everything the engine says therefore goes through [`out!`] and
//! [`outerr!`], and the browser build drains the buffer after each command and hands the
//! lines to its host.
//!
//! The protocol/diagnostic split is kept because it means something natively — a GUI
//! reads stdout and never sees stderr — even though in a browser both end up in the same
//! place.

/// A line of protocol: what a GUI on the other end of the pipe reads.
#[macro_export]
macro_rules! out {
    () => { $crate::out::line(String::new()) };
    ($($arg:tt)*) => { $crate::out::line(format!($($arg)*)) };
}

/// A line of diagnostics: `info string`, load failures, anything a GUI may ignore.
#[macro_export]
macro_rules! outerr {
    () => { $crate::out::diagnostic(String::new()) };
    ($($arg:tt)*) => { $crate::out::diagnostic(format!($($arg)*)) };
}

#[cfg(not(target_arch = "wasm32"))]
mod imp {
    pub fn line(s: String) {
        println!("{s}");
    }
    pub fn diagnostic(s: String) {
        eprintln!("{s}");
    }
}

#[cfg(target_arch = "wasm32")]
mod imp {
    use std::cell::RefCell;

    thread_local! {
        static BUFFER: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
    }

    pub fn line(s: String) {
        BUFFER.with(|b| b.borrow_mut().push(s));
    }

    /// Diagnostics share the buffer: a browser has no second stream to send them down,
    /// and `info string` is legal UCI that any reader is free to skip.
    pub fn diagnostic(s: String) {
        line(s);
    }

    /// Hands every line said since the last call to `f`, and empties the buffer.
    pub fn drain(f: &mut dyn FnMut(&str)) {
        BUFFER.with(|b| {
            for s in b.borrow_mut().drain(..) {
                f(&s);
            }
        });
    }
}

pub use imp::{diagnostic, line};

#[cfg(target_arch = "wasm32")]
pub use imp::drain;
