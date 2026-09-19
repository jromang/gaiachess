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
    use std::fs::{File, OpenOptions};
    use std::io::Write;
    use std::sync::Mutex;
    use std::time::{Instant, SystemTime, UNIX_EPOCH};

    struct DebugLog {
        file: File,
        start: Instant,
    }

    static LOG_FILE: Mutex<Option<DebugLog>> = Mutex::new(None);

    /// `Debug Log File`: empty path closes the log. The option GUIs expect under
    /// this name, so a GUI session (Fritz especially) can be dumped without a rebuild.
    ///
    /// Each line is `{seconds}.{ms} >>|<< {text}`: elapsed since this process opened
    /// the file, millisecond resolution, so a one-second stall between `info` lines
    /// is readable without a wall clock. Opening appends (Fritz re-sends the option
    /// on every module reset; truncating would drop the game that just flagged).
    pub fn set_debug_log(path: &str) -> Result<(), String> {
        let mut slot = LOG_FILE.lock().map_err(|e| e.to_string())?;
        if path.is_empty() {
            *slot = None;
            return Ok(());
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| e.to_string())?;
        let epoch_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let _ = writeln!(file, "# session {epoch_ms}");
        let _ = file.flush();
        *slot = Some(DebugLog {
            file,
            start: Instant::now(),
        });
        Ok(())
    }

    fn log_line(prefix: &str, s: &str) {
        if let Ok(mut slot) = LOG_FILE.lock()
            && let Some(log) = slot.as_mut()
        {
            let ms = log.start.elapsed().as_millis();
            let _ = writeln!(log.file, "{}.{:03} {prefix} {s}", ms / 1000, ms % 1000);
            let _ = log.file.flush();
        }
    }

    pub fn log_input(s: &str) {
        log_line(">>", s);
    }

    /// Serialises protocol lines: `isready` is answered from the stdin reader
    /// while the search thread prints `info`, and two `println!`s can interleave
    /// characters inside one line.
    static STDOUT: Mutex<()> = Mutex::new(());
    static STDERR: Mutex<()> = Mutex::new(());

    pub fn line(s: String) {
        log_line("<<", &s);
        let _guard = STDOUT.lock().unwrap_or_else(|e| e.into_inner());
        println!("{s}");
        // Windows pipes are block-buffered; a GUI reads UCI a line at a time and
        // will sit on an unflushed `info` until the 8 KiB buffer fills.
        let _ = std::io::stdout().flush();
    }

    /// Write a group of protocol lines without allowing another producer to interleave.
    pub fn lines(lines: &[String]) {
        for line in lines {
            log_line("<<", line);
        }
        let _guard = STDOUT.lock().unwrap_or_else(|e| e.into_inner());
        let mut stdout = std::io::stdout().lock();
        for line in lines {
            let _ = writeln!(stdout, "{line}");
        }
        let _ = stdout.flush();
    }
    pub fn diagnostic(s: String) {
        log_line("<<", &s);
        let _guard = STDERR.lock().unwrap_or_else(|e| e.into_inner());
        eprintln!("{s}");
        let _ = std::io::stderr().flush();
    }
}

#[cfg(target_arch = "wasm32")]
mod imp {
    use std::cell::RefCell;

    thread_local! {
        static BUFFER: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
    }

    pub fn set_debug_log(_path: &str) -> Result<(), String> {
        Err("Debug Log File is not available in the browser build".into())
    }

    pub fn log_input(_s: &str) {}

    pub fn line(s: String) {
        BUFFER.with(|b| b.borrow_mut().push(s));
    }

    pub fn lines(lines: &[String]) {
        BUFFER.with(|b| b.borrow_mut().extend_from_slice(lines));
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

pub use imp::{diagnostic, line, lines, log_input, set_debug_log};

#[cfg(target_arch = "wasm32")]
pub use imp::drain;
