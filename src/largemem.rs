//! Zeroed memory on large pages.
//!
//! The transposition table is read at random over gigabytes, and every history table is
//! read at random over megabytes: with 4 KiB pages most of those reads also miss the
//! address-translation cache. Backing them with 2 MiB pages — or 1 GiB ones for a huge
//! table — removes most of that cost.
//!
//! The memory comes straight from the operating system, untouched, so each page is
//! placed on the memory node of the thread that first writes it. Callers that care
//! about placement allocate and fill from a thread pinned to the right node (see
//! [`crate::numa::run_on_node`]). On Windows, large pages are committed — hence placed —
//! by the allocating thread itself.
//!
//! - Linux: `mmap` with explicit 1 GiB pages when asked and available, else a 2 MiB
//!   aligned anonymous mapping advised `MADV_HUGEPAGE`.
//! - Windows: `VirtualAlloc(MEM_LARGE_PAGES)` when the account holds the "Lock pages in
//!   memory" right, else ordinary pages.
//! - Elsewhere: the global allocator, zeroed.

use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

#[cfg(windows)]
pub(crate) use sys::with_large_page_privilege;

/// The page size large allocations are aligned to.
pub const HUGE_PAGE: usize = 2 * 1024 * 1024;

/// One gibibyte, the size of an explicit huge page.
pub const GIGA_PAGE: usize = 1024 * 1024 * 1024;

/// Whether a table of `mb` MiB spread over `nodes` memory nodes is worth 1 GiB pages:
/// at least eight of them per node.
///
/// Compared in mebibytes: in bytes the right-hand side alone is 2³³ per node, which
/// overflows a 32-bit `usize` (the browser build) and panics in a debug build.
pub fn wants_giga_pages(mb: usize, nodes: usize) -> bool {
    const MB_PER_GIGA_PAGE: usize = GIGA_PAGE / (1024 * 1024);
    debug_assert!(nodes >= 1, "a table lives on at least one node");
    mb >= nodes.saturating_mul(8 * MB_PER_GIGA_PAGE)
}

/// How a block ended up backed, for diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backing {
    /// Explicit 1 GiB pages.
    GigaPages,
    /// 2 MiB aligned and advised for transparent huge pages (Linux).
    HugeAdvised,
    /// Windows large pages.
    LargePages,
    /// Ordinary pages.
    Small,
}

/// A raw, zero-filled block of memory, released on drop.
pub struct LargeMem {
    ptr: NonNull<u8>,
    /// Bytes requested.
    len: usize,
    /// Bytes actually mapped, which the release needs.
    mapped: usize,
    backing: Backing,
}

// SAFETY: the block is plain memory owned by this value; sharing it is the user's
// business, as for a raw allocation.
unsafe impl Send for LargeMem {}
// SAFETY: as above.
unsafe impl Sync for LargeMem {}

impl LargeMem {
    /// Allocates `len` zeroed bytes. `giga_hint` asks for 1 GiB pages, for tables large
    /// enough that rounding up costs nothing. Aborts if memory cannot be had at all.
    pub fn zeroed(len: usize, giga_hint: bool) -> Self {
        let len = len.max(1);
        let (ptr, mapped, backing) = sys::alloc(len, giga_hint);
        debug_assert!(mapped >= len, "LargeMem: mapped {mapped} < requested {len}");
        LargeMem {
            ptr,
            len,
            mapped,
            backing,
        }
    }

    #[inline]
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn backing(&self) -> Backing {
        self.backing
    }
}

impl Drop for LargeMem {
    fn drop(&mut self) {
        // SAFETY: the block came from `sys::alloc` with exactly these figures.
        unsafe { sys::free(self.ptr, self.mapped, self.backing) };
    }
}

/// A heap value on large pages, like a `Box`.
pub struct LargeBox<T> {
    mem: LargeMem,
    _owns: PhantomData<T>,
}

// SAFETY: a LargeBox owns its T exactly like a Box does.
unsafe impl<T: Send> Send for LargeBox<T> {}
// SAFETY: as above.
unsafe impl<T: Sync> Sync for LargeBox<T> {}

impl<T> LargeBox<T> {
    /// Moves `value` onto large pages.
    pub fn new(value: T) -> Self {
        let mem = LargeMem::zeroed(size_of::<T>(), false);
        debug_assert!(align_of::<T>() <= 4096, "LargeBox: alignment above a page");
        // SAFETY: the block is large enough and page aligned, hence aligned for T.
        unsafe { mem.as_ptr().cast::<T>().write(value) };
        LargeBox {
            mem,
            _owns: PhantomData,
        }
    }

    /// A value whose every byte is zero, built in place without passing through the
    /// stack.
    ///
    /// # Safety
    /// The all-zero bit pattern must be a valid `T` — integer arrays, for instance.
    pub unsafe fn new_zeroed() -> Self {
        debug_assert!(align_of::<T>() <= 4096, "LargeBox: alignment above a page");
        LargeBox {
            mem: LargeMem::zeroed(size_of::<T>(), false),
            _owns: PhantomData,
        }
    }

    pub fn backing(&self) -> Backing {
        self.mem.backing()
    }
}

impl<T> Deref for LargeBox<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: the block holds an initialised T for as long as self lives.
        unsafe { &*self.mem.as_ptr().cast::<T>() }
    }
}

impl<T> DerefMut for LargeBox<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: as above, and `&mut self` makes the access unique.
        unsafe { &mut *self.mem.as_ptr().cast::<T>() }
    }
}

impl<T> Drop for LargeBox<T> {
    fn drop(&mut self) {
        // SAFETY: the T is initialised and dropped once; the memory goes with `mem`.
        unsafe { std::ptr::drop_in_place(self.mem.as_ptr().cast::<T>()) };
    }
}

#[cfg(any(target_os = "linux", windows))]
fn round_up(n: usize, to: usize) -> usize {
    n.div_ceil(to) * to
}

#[cfg(target_os = "linux")]
mod sys {
    use super::{Backing, GIGA_PAGE, HUGE_PAGE, round_up};
    use std::ptr::NonNull;

    fn map(len: usize, extra: libc::c_int) -> Option<NonNull<u8>> {
        // SAFETY: an anonymous private mapping aliases nothing; the kernel picks the
        // address.
        let p = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | extra,
                -1,
                0,
            )
        };
        if p == libc::MAP_FAILED {
            None
        } else {
            NonNull::new(p.cast())
        }
    }

    pub fn alloc(len: usize, giga_hint: bool) -> (NonNull<u8>, usize, Backing) {
        #[cfg(target_arch = "x86_64")]
        if giga_hint && len >= GIGA_PAGE {
            const MAP_HUGE_1GB: libc::c_int = 30 << libc::MAP_HUGE_SHIFT;
            let size = round_up(len, GIGA_PAGE);
            if let Some(p) = map(size, libc::MAP_HUGETLB | MAP_HUGE_1GB) {
                return (p, size, Backing::GigaPages);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = (giga_hint, GIGA_PAGE);

        // A reservation one huge page too large, trimmed to a 2 MiB boundary at both
        // ends: transparent huge pages need the alignment.
        let size = round_up(len, HUGE_PAGE);
        let Some(base) = map(size + HUGE_PAGE, 0) else {
            out!("info string could not allocate {} MB", len >> 20);
            std::process::abort();
        };
        let base_addr = base.as_ptr() as usize;
        let aligned = round_up(base_addr, HUGE_PAGE);
        let head = aligned - base_addr;
        let tail = HUGE_PAGE - head;
        // SAFETY: both slivers belong to the mapping made above and lie outside the
        // aligned range handed out.
        unsafe {
            if head > 0 {
                libc::munmap(base.as_ptr().cast(), head);
            }
            if tail > 0 {
                libc::munmap((aligned + size) as *mut libc::c_void, tail);
            }
            libc::madvise(aligned as *mut libc::c_void, size, libc::MADV_HUGEPAGE);
        }
        let ptr = NonNull::new(aligned as *mut u8).expect("aligned mapping is null");
        debug_assert_eq!(ptr.as_ptr() as usize % HUGE_PAGE, 0);
        (ptr, size, Backing::HugeAdvised)
    }

    /// # Safety
    /// `ptr` and `mapped` must come from `alloc`.
    pub unsafe fn free(ptr: NonNull<u8>, mapped: usize, _backing: Backing) {
        // SAFETY: the caller hands back a mapping of exactly this size.
        unsafe { libc::munmap(ptr.as_ptr().cast(), mapped) };
    }
}

#[cfg(windows)]
mod sys {
    use super::{Backing, round_up};
    use std::ffi::c_void;
    use std::ptr::NonNull;

    type Handle = *mut c_void;

    const MEM_COMMIT: u32 = 0x1000;
    const MEM_RESERVE: u32 = 0x2000;
    const MEM_RELEASE: u32 = 0x8000;
    const MEM_LARGE_PAGES: u32 = 0x2000_0000;
    const PAGE_READWRITE: u32 = 0x04;
    const TOKEN_ADJUST_PRIVILEGES: u32 = 0x0020;
    const TOKEN_QUERY: u32 = 0x0008;
    const SE_PRIVILEGE_ENABLED: u32 = 0x0000_0002;
    const ERROR_SUCCESS: u32 = 0;

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct Luid {
        low: u32,
        high: i32,
    }

    #[repr(C, packed(4))]
    #[derive(Clone, Copy, Default)]
    struct TokenPrivileges {
        count: u32,
        luid: Luid,
        attributes: u32,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn VirtualAlloc(addr: *mut c_void, size: usize, kind: u32, protect: u32) -> *mut c_void;
        fn VirtualFree(addr: *mut c_void, size: usize, kind: u32) -> i32;
        fn GetLargePageMinimum() -> usize;
        fn GetCurrentProcess() -> Handle;
        fn CloseHandle(h: Handle) -> i32;
        fn GetLastError() -> u32;
    }

    #[link(name = "advapi32")]
    unsafe extern "system" {
        fn OpenProcessToken(process: Handle, access: u32, token: *mut Handle) -> i32;
        fn LookupPrivilegeValueA(system: *const u8, name: *const u8, luid: *mut Luid) -> i32;
        fn AdjustTokenPrivileges(
            token: Handle,
            disable_all: i32,
            new: *const TokenPrivileges,
            prev_len: u32,
            prev: *mut TokenPrivileges,
            ret_len: *mut u32,
        ) -> i32;
    }

    /// Runs `f` with the "Lock pages in memory" right enabled on the process token, as
    /// large pages require, handing it the large page size; the previous state is
    /// restored after. `None` when the right is not held or large pages do not exist.
    pub fn with_large_page_privilege<R>(f: impl FnOnce(usize) -> Option<R>) -> Option<R> {
        // SAFETY: plain Win32 calls on locals; every handle opened is closed.
        unsafe {
            let page = GetLargePageMinimum();
            if page == 0 {
                return None;
            }
            let mut token: Handle = std::ptr::null_mut();
            if OpenProcessToken(
                GetCurrentProcess(),
                TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY,
                &mut token,
            ) == 0
            {
                return None;
            }
            let mut luid = Luid::default();
            let mut result = None;
            if LookupPrivilegeValueA(
                std::ptr::null(),
                c"SeLockMemoryPrivilege".as_ptr().cast(),
                &mut luid,
            ) != 0
            {
                let tp = TokenPrivileges {
                    count: 1,
                    luid,
                    attributes: SE_PRIVILEGE_ENABLED,
                };
                let mut prev = TokenPrivileges::default();
                let mut prev_len = 0u32;
                // Success with ERROR_NOT_ALL_ASSIGNED means the right is not held.
                if AdjustTokenPrivileges(
                    token,
                    0,
                    &tp,
                    size_of::<TokenPrivileges>() as u32,
                    &mut prev,
                    &mut prev_len,
                ) != 0
                    && GetLastError() == ERROR_SUCCESS
                {
                    result = f(page);
                    AdjustTokenPrivileges(
                        token,
                        0,
                        &prev,
                        0,
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                    );
                }
            }
            CloseHandle(token);
            result
        }
    }

    fn try_large_pages(len: usize) -> Option<(NonNull<u8>, usize)> {
        with_large_page_privilege(|page| {
            let size = round_up(len, page);
            // SAFETY: a fresh reservation and commit, owned by the caller.
            let p = unsafe {
                VirtualAlloc(
                    std::ptr::null_mut(),
                    size,
                    MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
                    PAGE_READWRITE,
                )
            };
            NonNull::new(p.cast::<u8>()).map(|p| (p, size))
        })
    }

    pub fn alloc(len: usize, _giga_hint: bool) -> (NonNull<u8>, usize, Backing) {
        // Small tables are not worth a privilege dance per thread table.
        if len >= super::HUGE_PAGE
            && let Some((p, size)) = try_large_pages(len)
        {
            // Large pages are not guaranteed zero on every Windows version.
            // SAFETY: the block was just committed and is `size` bytes long.
            unsafe { std::ptr::write_bytes(p.as_ptr(), 0, size) };
            return (p, size, Backing::LargePages);
        }
        // SAFETY: a plain commit; the system zeroes fresh pages.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null_mut(),
                len,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        };
        match NonNull::new(p.cast::<u8>()) {
            Some(p) => (p, len, Backing::Small),
            None => {
                out!("info string could not allocate {} MB", len >> 20);
                std::process::abort();
            }
        }
    }

    /// # Safety
    /// `ptr` must come from `alloc`.
    pub unsafe fn free(ptr: NonNull<u8>, _mapped: usize, _backing: Backing) {
        // SAFETY: releasing a whole VirtualAlloc reservation takes size zero.
        unsafe { VirtualFree(ptr.as_ptr().cast(), 0, MEM_RELEASE) };
    }
}

#[cfg(not(any(target_os = "linux", windows)))]
mod sys {
    use super::Backing;
    use std::alloc::Layout;
    use std::ptr::NonNull;

    const ALIGN: usize = 4096;

    pub fn alloc(len: usize, _giga_hint: bool) -> (NonNull<u8>, usize, Backing) {
        let layout = Layout::from_size_align(len, ALIGN).expect("LargeMem layout");
        // SAFETY: the layout is non-zero sized.
        let p = unsafe { std::alloc::alloc_zeroed(layout) };
        match NonNull::new(p) {
            Some(p) => (p, len, Backing::Small),
            None => std::alloc::handle_alloc_error(layout),
        }
    }

    /// # Safety
    /// `ptr` and `mapped` must come from `alloc`.
    pub unsafe fn free(ptr: NonNull<u8>, mapped: usize, _backing: Backing) {
        let layout = Layout::from_size_align(mapped, ALIGN).expect("LargeMem layout");
        // SAFETY: the same layout the block was allocated with.
        unsafe { std::alloc::dealloc(ptr.as_ptr(), layout) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Eight 1 GiB pages per node, decided in mebibytes so that no product overflows a
    /// 32-bit `usize` (it used to, computed in bytes).
    #[test]
    fn giga_pages_from_eight_per_node() {
        assert!(!wants_giga_pages(8_191, 1));
        assert!(wants_giga_pages(8_192, 1));
        assert!(!wants_giga_pages(65_535, 8));
        assert!(wants_giga_pages(65_536, 8));
        assert!(!wants_giga_pages(16, 1));
        assert!(!wants_giga_pages(usize::MAX - 1, usize::MAX));
        assert!(wants_giga_pages(usize::MAX, usize::MAX));
    }

    #[test]
    fn blocks_come_back_zeroed_and_writable() {
        for len in [1usize, 4096, 3 * 1024 * 1024 + 7] {
            let mem = LargeMem::zeroed(len, false);
            assert_eq!(mem.len(), len);
            // SAFETY: `len` bytes belong to the block.
            let bytes = unsafe { std::slice::from_raw_parts_mut(mem.as_ptr(), len) };
            assert!(bytes.iter().all(|&b| b == 0));
            bytes.fill(0xa5);
            assert_eq!(bytes[len - 1], 0xa5);
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn large_blocks_are_huge_page_aligned() {
        let mem = LargeMem::zeroed(5 * 1024 * 1024, false);
        assert_eq!(mem.as_ptr() as usize % HUGE_PAGE, 0);
        assert_eq!(mem.backing(), Backing::HugeAdvised);
    }

    #[test]
    fn boxes_drop_their_contents() {
        use std::rc::Rc;
        let marker = Rc::new(());
        let b = LargeBox::new((Rc::clone(&marker), [7u64; 1000]));
        assert_eq!(b.1[999], 7);
        assert_eq!(Rc::strong_count(&marker), 2);
        drop(b);
        assert_eq!(Rc::strong_count(&marker), 1);
        // SAFETY: zero is a valid array of integers.
        let z = unsafe { LargeBox::<[i16; 100_000]>::new_zeroed() };
        assert!(z.iter().all(|&v| v == 0));
    }
}
