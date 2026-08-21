//! The engine as a WebAssembly module, spoken to in UCI.
//!
//! Its host is a Web Worker: the interface runs on the page's own thread and would stop
//! drawing if a search ran there. The two talk in UCI text rather than a private binary
//! protocol, because everything needed already exists — `position … moves …` rebuilds
//! the repetition history exactly, and `setoption name Skill Level` is the ladder — and
//! because a position cannot be posted across a worker boundary anyway: it carries a
//! thousand-entry history, some hundred kilobytes, where the move list is two hundred
//! bytes.
//!
//! No `wasm-bindgen`. Three functions and one import are the whole surface, which keeps
//! the toolchain to plain stable `cargo build` and leaves the module loadable by the
//! same hand-written JavaScript that loads the interface.
//!
//! # Protocol
//!
//! ```text
//! host                                    module
//!   gaia_new()                         →  builds the session
//!   gaia_alloc(len) → ptr              →  sizes the inbox, hands back where to write
//!   (writes len UTF-8 bytes at ptr)
//!   gaia_command(len) → 1 alive/0 quit →  runs it, then calls back gaia_out(ptr, len)
//!                                          once per line of output
//! ```
//!
//! Every call may grow the linear memory, which detaches any `ArrayBuffer` view the host
//! is holding. Views must be built fresh after each call, never cached.

use std::cell::RefCell;

use gaiachess::uci::UciSession;

// Named explicitly rather than left to the default: the host builds its import object
// by hand, and `env` is the table it fills. Resolving this at link time also needs
// `-C link-arg=--allow-undefined`, since a `cdylib` otherwise insists every symbol be
// defined in the module itself.
#[link(wasm_import_module = "env")]
unsafe extern "C" {
    /// One line of engine output: UTF-8, no trailing newline.
    fn gaia_out(ptr: *const u8, len: usize);
}

thread_local! {
    static SESSION: RefCell<Option<UciSession>> = const { RefCell::new(None) };
    /// Where the host writes the next command. Reused rather than reallocated.
    static INBOX: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// Builds the session. Must be called once, before any command.
#[unsafe(no_mangle)]
pub extern "C" fn gaia_new() {
    install_panic_hook();
    gaiachess::init_cpu_dispatch();
    SESSION.with(|s| *s.borrow_mut() = Some(UciSession::new()));
    flush();
}

/// Sends panics out through the bridge before the instance dies.
///
/// The profile aborts on panic, and an aborted instance is gone for good — the page is
/// left with a board that has stopped answering. There is no stderr in a browser either,
/// so without this the cause is simply unavailable. The hook still runs under
/// `panic = "abort"`, which is the whole opportunity: one line saying what happened,
/// then the instance goes.
fn install_panic_hook() {
    std::panic::set_hook(Box::new(|info| {
        let message = format!("info string PANIC {info}");
        unsafe { gaia_out(message.as_ptr(), message.len()) };
    }));
}

/// Sizes the inbox to `len` bytes and returns where to write them.
///
/// The pointer is valid until the next call into the module. Nothing of ours runs while
/// the host writes, so the buffer cannot move underneath it.
#[unsafe(no_mangle)]
pub extern "C" fn gaia_alloc(len: usize) -> *mut u8 {
    INBOX.with(|b| {
        let mut b = b.borrow_mut();
        b.clear();
        b.resize(len, 0);
        b.as_mut_ptr()
    })
}

/// Runs the `len` bytes now in the inbox as one UCI command.
///
/// Returns 0 once the engine has been told to quit, 1 otherwise. Output is delivered
/// through [`gaia_out`] before this returns.
#[unsafe(no_mangle)]
pub extern "C" fn gaia_command(len: usize) -> i32 {
    let line = INBOX.with(|b| String::from_utf8_lossy(&b.borrow()[..len]).into_owned());

    let alive = SESSION.with(|s| match s.borrow_mut().as_mut() {
        Some(session) => session.command(&line),
        // A command before `gaia_new`: say so rather than fail silently, since a browser
        // has no console the engine can be sure of.
        None => {
            gaiachess::out::line(String::from("info string engine not started"));
            true
        }
    });

    flush();
    i32::from(alive)
}

/// Hands everything the engine has said to the host, one line per call.
fn flush() {
    gaiachess::out::drain(&mut |line: &str| unsafe {
        gaia_out(line.as_ptr(), line.len());
    });
}

// ============================================================
// Receiving the network
// ============================================================

/// How many bytes the host must write into the buffer [`gaia_net_reserve`] hands back.
#[unsafe(no_mangle)]
pub extern "C" fn gaia_net_size() -> usize {
    gaiachess::nnue::network::NNUE_FILE_SIZE
}

/// Reserves the network and returns where to write it.
///
/// The bytes are written straight into their final home rather than staged and copied:
/// a browser never gets linear memory back, so a doubled peak is a doubled cost for the
/// life of the page. Re-create any view onto the memory after this call — reserving may
/// have grown it, which detaches the old one.
#[unsafe(no_mangle)]
pub extern "C" fn gaia_net_reserve() -> *mut u8 {
    gaiachess::nnue::network::reserve()
}

/// Publishes the network just written. Returns 1 on success, 0 otherwise.
#[unsafe(no_mangle)]
pub extern "C" fn gaia_net_finish() -> i32 {
    match gaiachess::nnue::network::publish_reserved() {
        Ok(()) => 1,
        Err(err) => {
            gaiachess::out::line(format!("info string network not loaded: {err}"));
            flush();
            0
        }
    }
}
