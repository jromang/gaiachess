//! The Windows console, borrowed rather than created.
//!
//! With the interface compiled in, the binary is linked as a GUI-subsystem
//! program, so Windows never allocates a console for it: double-clicking the
//! game opens the board and nothing else. The same binary is still an engine,
//! and an engine has to be able to talk — over the pipes of a match manager,
//! which need no console at all, but also in a terminal, where `bench`, `info`
//! and `--help` have to land somewhere visible.
//!
//! So the console is not created, it is joined: whoever launched us already has
//! one, and this module attaches to it and repairs only the streams that would
//! otherwise lead nowhere.

use std::ffi::{CStr, c_void};
use std::ptr;

type Handle = *mut c_void;

const INVALID_HANDLE_VALUE: Handle = -1isize as Handle;
const STD_INPUT_HANDLE: u32 = -10i32 as u32;
const STD_OUTPUT_HANDLE: u32 = -11i32 as u32;
const STD_ERROR_HANDLE: u32 = -12i32 as u32;
/// `AttachConsole` asked for the launcher's console rather than a given process.
const ATTACH_PARENT_PROCESS: u32 = -1i32 as u32;
const FILE_TYPE_UNKNOWN: u32 = 0x0000;
const GENERIC_READ: u32 = 0x8000_0000;
const GENERIC_WRITE: u32 = 0x4000_0000;
const FILE_SHARE_READ: u32 = 0x0000_0001;
const FILE_SHARE_WRITE: u32 = 0x0000_0002;
const OPEN_EXISTING: u32 = 3;

#[link(name = "kernel32")]
unsafe extern "system" {
    fn AttachConsole(process_id: u32) -> i32;
    fn GetStdHandle(std_handle: u32) -> Handle;
    fn SetStdHandle(std_handle: u32, handle: Handle) -> i32;
    fn GetFileType(file: Handle) -> u32;
    fn CreateFileA(
        name: *const u8,
        access: u32,
        share_mode: u32,
        security: *mut c_void,
        creation: u32,
        flags: u32,
        template: Handle,
    ) -> Handle;
}

/// Whether a handle can carry bytes at all — a pipe, a file, a console, a device.
///
/// `GetFileType` answers `UNKNOWN` for a handle that leads nowhere, which is what
/// stdin looks like when a program is started from the desktop rather than from a
/// shell. Asked only after attaching: a console handle inherited from the launcher
/// is only meaningful once we share that console.
fn usable(handle: Handle) -> bool {
    !handle.is_null() && handle != INVALID_HANDLE_VALUE && unsafe { GetFileType(handle) } != FILE_TYPE_UNKNOWN
}

/// Whether anything could ever be read from stdin.
///
/// False means the process was started with no input channel whatsoever — the
/// desktop, a shortcut, an icon — so no interface will ever speak the protocol
/// here and there is nothing to wait for.
pub fn stdin_is_readable() -> bool {
    usable(unsafe { GetStdHandle(STD_INPUT_HANDLE) })
}

/// Joins the launcher's console, if it has one, and gives the standard streams
/// somewhere to go.
///
/// Must run before anything is written or any argument is parsed: `--help` and
/// `--version` are printed by the parser itself.
pub fn attach_parent_console() {
    // Taken before joining, because joining a console re-points all three standard
    // handles at it — the inherited ones included. Whatever the launcher handed us is
    // only recoverable from this snapshot.
    let inherited = [
        unsafe { GetStdHandle(STD_OUTPUT_HANDLE) },
        unsafe { GetStdHandle(STD_ERROR_HANDLE) },
        unsafe { GetStdHandle(STD_INPUT_HANDLE) },
    ];

    // Nothing to join: started from the desktop, or by a match manager that is
    // itself a windowless program. Both cases already have what they need — the
    // board in the first, the inherited pipes in the second.
    if unsafe { AttachConsole(ATTACH_PARENT_PROCESS) } == 0 {
        return;
    }

    // Redirection outranks the console it was typed in. `bench > run.txt`, a shell
    // pipeline and the pipes of a match manager all arrive as perfectly usable
    // handles, and are put back exactly as they came; only a stream with no
    // destination is pointed at the console we have just joined.
    restore_or_bind(STD_OUTPUT_HANDLE, inherited[0], c"CONOUT$", GENERIC_WRITE);
    restore_or_bind(STD_ERROR_HANDLE, inherited[1], c"CONOUT$", GENERIC_WRITE);
    restore_or_bind(STD_INPUT_HANDLE, inherited[2], c"CONIN$", GENERIC_READ);
}

/// Gives one standard stream back the destination it was launched with, or, if it had
/// none, points it at the console.
///
/// The handle type is asked for only here, after attaching: a console handle inherited
/// from the launcher is only meaningful once we share that console. `CONOUT$` and
/// `CONIN$` name the console we are attached to, so the fallback is a fresh handle to it
/// rather than a copy of anything inherited.
fn restore_or_bind(std_handle: u32, inherited: Handle, device: &CStr, access: u32) {
    debug_assert!(matches!(std_handle, STD_INPUT_HANDLE | STD_OUTPUT_HANDLE | STD_ERROR_HANDLE));
    debug_assert!(access == GENERIC_READ || access == GENERIC_WRITE);

    if usable(inherited) {
        // A no-op when the launcher's handle was the console we have just joined, and
        // the whole point of the snapshot when it was a file or a pipe.
        unsafe { SetStdHandle(std_handle, inherited) };
        return;
    }

    // Shared both ways: the console is the shell's too, and it goes on using it.
    let handle = unsafe {
        CreateFileA(
            device.as_ptr().cast(),
            access,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            ptr::null_mut(),
            OPEN_EXISTING,
            0,
            ptr::null_mut(),
        )
    };

    // A failure here leaves the stream as mute as it already was, which is why it
    // is not reported: there would be nowhere to report it to.
    if handle != INVALID_HANDLE_VALUE && !handle.is_null() {
        unsafe { SetStdHandle(std_handle, handle) };
    }
}
