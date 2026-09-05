//! One copy of a large read-only blob for every process on the machine.
//!
//! A fixed-node test run puts one engine process on every logical core, each
//! holding its own copy of the NNUE weights. The weights are read at random
//! rows on every accumulator update, so what the cache sees is not one working
//! set but as many as there are processes, all of them cold. The weights are
//! identical and never written, so there is no reason for more than one copy
//! of them to exist on a machine.
//!
//! What is shared is a *memfd*: anonymous kernel memory with a file descriptor
//! but no name anywhere in the filesystem. Processes find each other through
//! Unix sockets in the abstract namespace, which likewise have no path — they
//! live and die with the process that bound them. Nothing is ever written to
//! disk or to a tmpfs, and nothing survives the last process that holds it.
//!
//! The protocol has no central authority and never blocks for long:
//!
//! 1. Look for a peer already serving this blob and ask it for the descriptor.
//!    A peer answers by passing the memfd itself over the socket (`SCM_RIGHTS`),
//!    so the two processes end up on the same physical pages.
//! 2. If nobody answers, try to bind the builder's name. Winning it means
//!    building the blob; losing it means somebody else is building, so wait and
//!    look again.
//! 3. Either way, once holding the descriptor, serve it onward. The mesh
//!    outlives its builder, and the memory is freed when the last holder exits.
//!
//! Every failure is answered by returning an error, never by panicking or
//! waiting forever: the caller's fallback is its own private copy, which is
//! what the engine did before this module existed.
//!
//! The blob is sealed against writes before it is shared, so a mapping handed
//! out here cannot be modified by anyone, including its builder.

/// Where a mapping came from, for the message the engine prints.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Origin {
    /// This process built the image and now serves it.
    Built,
    /// A peer passed us the descriptor.
    Received,
}

impl Origin {
    pub fn as_str(self) -> &'static str {
        match self {
            Origin::Built => "built",
            Origin::Received => "received",
        }
    }
}

/// A blob mapped read-only, shared with every other process that asked for the
/// same name. Never unmapped: the engine hands out `&'static` references into
/// it and has no protocol for taking one back.
#[derive(Debug)]
pub struct Mapped {
    ptr: *const u8,
    len: usize,
    pub origin: Origin,
}

// SAFETY: the mapping is read-only for its whole life (sealed against writes
// before it is ever shared), so a pointer into it is as safe to send between
// threads as a `&'static [u8]`.
unsafe impl Send for Mapped {}
unsafe impl Sync for Mapped {}

impl Mapped {
    /// The blob, for as long as the process lives.
    pub fn bytes(&self) -> &'static [u8] {
        // SAFETY: the mapping is never unmapped and never written after
        // sealing, and `len` bytes were mapped at `ptr`.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

/// Whether sharing is available at all on this platform.
pub fn supported() -> bool {
    imp::SUPPORTED
}

/// Whether sharing is available *and* wanted. `GAIA_SHARED_NET=0` (or `false`,
/// or `off`) turns it off before anything is materialised, which is the only
/// hook a benchmark has: the network is built during startup, before the first
/// UCI option can arrive.
pub fn enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        if !supported() {
            return false;
        }
        match std::env::var("GAIA_SHARED_NET") {
            Ok(v) => !matches!(v.trim(), "0" | "false" | "off" | "no"),
            Err(_) => true,
        }
    })
}

/// Whether to re-derive the blob and compare it against the mapping.
///
/// The name says which weights, which architecture, which layout — but not
/// which *transform* produced the bytes, and a transform can change with the
/// layout untouched. A stale name is then indistinguishable from a good one,
/// and the process would evaluate through its neighbour's weights in silence.
/// Debug builds always check; release builds check on request.
pub fn verify_requested() -> bool {
    cfg!(debug_assertions)
        || std::env::var("GAIA_SHARED_NET_VERIFY").is_ok_and(|v| !matches!(v.trim(), "0" | ""))
}

/// Map the blob called `name`, building it with `build` if nobody else has.
///
/// `build` is handed exactly `len` bytes to fill and is called at most once,
/// only if this process becomes the builder. On any failure the error says
/// what went wrong in a form fit for `info string`, and the caller is expected
/// to fall back to a private copy.
pub fn acquire(
    name: &str,
    len: usize,
    build: &mut dyn FnMut(&mut [u8]),
) -> Result<Mapped, String> {
    imp::acquire(name, len, build)
}

/// How many processes are currently serving blobs to their peers.
pub fn peers() -> usize {
    imp::peers()
}

/// One line for `gaiachess info`. Materialises nothing.
pub fn describe() -> String {
    if !supported() {
        return String::from("unsupported on this platform");
    }
    if !enabled() {
        return String::from("disabled (GAIA_SHARED_NET)");
    }
    let n = peers();
    let verify = if verify_requested() { ", verifying" } else { "" };
    format!("enabled, {n} peer(s) serving{verify}")
}

// ============================================================
// Linux: memfd + abstract sockets
// ============================================================

#[cfg(target_os = "linux")]
mod imp {
    use super::{Mapped, Origin};

    use std::collections::HashSet;
    use std::io::{IoSlice, IoSliceMut};
    use std::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, OwnedFd};
    use std::os::linux::net::SocketAddrExt;
    use std::os::unix::net::{SocketAddr, UnixListener, UnixStream};
    use std::sync::{Mutex, OnceLock};
    use std::time::{Duration, Instant};

    use nix::fcntl::{fcntl, FcntlArg, SealFlag};
    use nix::sys::memfd::{memfd_create, MFdFlags};
    use nix::sys::mman::{madvise, mmap, mmap_anonymous, munmap, MapFlags, MmapAdvise, ProtFlags};
    use nix::sys::socket::{
        getsockopt, recvmsg, sendmsg, sockopt, ControlMessage, ControlMessageOwned, MsgFlags,
    };
    use nix::sys::stat::fstat;
    use nix::sys::uio::pread;
    use nix::unistd::ftruncate;

    pub const SUPPORTED: bool = true;

    /// How long a process waits for whoever is building the blob before giving
    /// up and building its own private copy. Generous next to the ~150 ms a
    /// build takes, because the alternative to waiting is doing the work again.
    const BUILD_WAIT: Duration = Duration::from_secs(10);

    /// Huge page size assumed for the alignment of the mapping. A 2 MiB-aligned
    /// mapping is the precondition for the kernel to back it with huge pages,
    /// which is most of the point: 38 MiB in 4 KiB pages is ~9 400 TLB entries
    /// per process, and the weights are read at random rows.
    const HUGE: usize = 2 * 1024 * 1024;

    /// `MADV_COLLAPSE`, absent from nix. Asks the kernel to rebuild an existing
    /// mapping out of huge pages regardless of the shmem THP policy, which on a
    /// host set to `never` is the difference between huge pages and none.
    const MADV_COLLAPSE: i32 = 25;

    /// Bytes of bookkeeping after the payload, so a process can tell whether the
    /// descriptor it was handed holds what it asked for.
    const TRAILER_SIZE: usize = 64;

    const TRAILER_MAGIC: [u8; 4] = *b"GSHM";

    /// Format of the trailer itself, separate from what the payload means.
    const TRAILER_FORMAT: u32 = 1;

    /// Longest abstract socket name: `sun_path` is 108 bytes and the leading
    /// NUL that marks the namespace takes one.
    const MAX_NAME: usize = 107;

    #[repr(C)]
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    struct Trailer {
        magic: [u8; 4],
        format: u32,
        name_hash: u64,
        payload_len: u64,
        payload_hash: u64,
        pad: [u8; 32],
    }

    const _: () = assert!(size_of::<Trailer>() == TRAILER_SIZE);

    impl Trailer {
        fn expected(name: &str, len: usize, payload_hash: u64) -> Self {
            Trailer {
                magic: TRAILER_MAGIC,
                format: TRAILER_FORMAT,
                name_hash: crate::nnue::integrity::fnv1a64(name.as_bytes()),
                payload_len: len as u64,
                payload_hash,
                pad: [0; 32],
            }
        }

        fn as_bytes(&self) -> &[u8] {
            // SAFETY: repr(C) POD, no padding (4+4+8+8+8+32 = 64), read as bytes.
            unsafe { std::slice::from_raw_parts(self as *const Self as *const u8, TRAILER_SIZE) }
        }

        fn from_bytes(raw: &[u8; TRAILER_SIZE]) -> Self {
            // SAFETY: every bit pattern of the field types is valid, and the
            // buffer is exactly the struct's size and more than its alignment.
            unsafe { std::ptr::read_unaligned(raw.as_ptr() as *const Self) }
        }
    }

    /// Size of the whole image: payload, then the trailer, rounded up so every
    /// page of it can be a huge page.
    fn image_size(len: usize) -> usize {
        let with_trailer = round_up(len, 64) + TRAILER_SIZE;
        let total = round_up(with_trailer, HUGE);
        debug_assert_eq!(total % HUGE, 0);
        debug_assert!(total >= len + TRAILER_SIZE);
        total
    }

    fn round_up(v: usize, to: usize) -> usize {
        debug_assert!(to.is_power_of_two());
        (v + to - 1) & !(to - 1)
    }

    fn trailer_offset(len: usize) -> usize {
        let off = round_up(len, 64);
        debug_assert_eq!(off % 64, 0);
        off
    }

    fn euid() -> u32 {
        // SAFETY: geteuid cannot fail and touches nothing.
        unsafe { nix::libc::geteuid() }
    }

    /// Which executable this process is running, as a short digest.
    ///
    /// Two different binaries must never read each other's memory, whatever
    /// their names for the same blob happen to agree on. That is not a
    /// hypothetical: an SPRT runs a patched engine against its base at the
    /// same time, on the same machine, over the same weights, and any code
    /// change between them could change what the image is supposed to hold.
    /// Keying on the executable itself makes the question moot.
    ///
    /// Identity is the file behind `/proc/self/exe`: device, inode, size and
    /// modification time. A rebuild replaces the file, so it lands on a new
    /// tag even at the same path. Where `/proc` is not mounted the path alone
    /// has to do, and the caller's own name for the blob carries the rest.
    fn binary_tag() -> u64 {
        static TAG: OnceLock<u64> = OnceLock::new();
        *TAG.get_or_init(|| {
            let path = std::fs::read_link("/proc/self/exe")
                .or_else(|_| std::env::current_exe())
                .unwrap_or_default();
            let mut acc = crate::nnue::integrity::fnv1a64(path.as_os_str().as_encoded_bytes());
            if let Ok(md) = std::fs::metadata(&path) {
                use std::os::unix::fs::MetadataExt;
                for v in [md.dev(), md.ino(), md.size(), md.mtime() as u64, md.mtime_nsec() as u64]
                {
                    acc = crate::nnue::integrity::fnv1a64(&{
                        let mut b = acc.to_le_bytes().to_vec();
                        b.extend_from_slice(&v.to_le_bytes());
                        b
                    });
                }
            }
            acc
        })
    }

    /// The namespace a blob lives in: this user, this executable, this blob.
    fn scope(name: &str) -> String {
        format!("gaiachess/{}/{:016x}/{name}", euid(), binary_tag())
    }

    fn serve_prefix(name: &str) -> String {
        format!("{}/serve/", scope(name))
    }

    fn init_name(name: &str) -> String {
        format!("{}/init", scope(name))
    }

    /// Abstract socket names of every peer currently serving this blob.
    ///
    /// `/proc/net/unix` is the only way to enumerate the abstract namespace;
    /// bound names appear there with the leading NUL shown as `@`.
    fn scan_peers(prefix: &str) -> Vec<String> {
        let Ok(text) = std::fs::read_to_string("/proc/net/unix") else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for line in text.lines() {
            let Some(path) = line.rsplit(' ').next() else { continue };
            if let Some(rest) = path.strip_prefix('@')
                && rest.starts_with(prefix)
                && !out.iter().any(|p| p == rest)
            {
                out.push(rest.to_string());
            }
        }
        out
    }

    pub fn peers() -> usize {
        scan_peers(&format!("gaiachess/{}/", euid())).len()
    }

    fn abstract_addr(name: &str) -> Result<SocketAddr, String> {
        if name.len() > MAX_NAME || name.contains('\0') {
            return Err(format!("socket name unusable ({} bytes)", name.len()));
        }
        SocketAddr::from_abstract_name(name.as_bytes())
            .map_err(|e| format!("bad socket name: {e}"))
    }

    /// Ask one peer for the descriptor it is serving.
    fn receive_from(peer: &str) -> Result<OwnedFd, String> {
        let addr = abstract_addr(peer)?;
        let stream = UnixStream::connect_addr(&addr).map_err(|e| format!("connect: {e}"))?;
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .map_err(|e| format!("timeout: {e}"))?;

        // Only take a descriptor from ourselves. Another user's process could
        // have bound a name in this namespace and would be offering unknown
        // memory to evaluate through.
        let cred = getsockopt(&stream, sockopt::PeerCredentials)
            .map_err(|e| format!("peer credentials: {e}"))?;
        if cred.uid() != euid() {
            return Err(String::from("peer belongs to another user"));
        }

        let mut byte = [0u8; 1];
        let mut iov = [IoSliceMut::new(&mut byte)];
        let mut space = nix::cmsg_space!([std::os::fd::RawFd; 1]);
        let msg = recvmsg::<()>(
            stream.as_raw_fd(),
            &mut iov,
            Some(&mut space),
            MsgFlags::MSG_CMSG_CLOEXEC,
        )
        .map_err(|e| format!("recvmsg: {e}"))?;

        for cmsg in msg.cmsgs().map_err(|e| format!("cmsgs: {e}"))? {
            if let ControlMessageOwned::ScmRights(fds) = cmsg
                && let Some(&raw) = fds.first()
            {
                // SAFETY: the descriptor was just created by the kernel for
                // this process from the peer's SCM_RIGHTS message; nothing
                // else owns it.
                return Ok(unsafe { OwnedFd::from_raw_fd(raw) });
            }
        }
        Err(String::from("peer sent no descriptor"))
    }

    /// Check that a received descriptor holds the image we asked for, and that
    /// nobody can change it under us.
    fn validate(fd: BorrowedFd<'_>, name: &str, len: usize) -> Result<(), String> {
        let st = fstat(fd).map_err(|e| format!("fstat: {e}"))?;
        let want = image_size(len);
        if st.st_size as u64 != want as u64 {
            return Err(format!("size {} != {want}", st.st_size));
        }

        // Sealed against writes and against resizing: without this the sender
        // could still be writing, or could shrink the file and turn every
        // reader's next access into SIGBUS.
        let raw = fcntl(fd, FcntlArg::F_GET_SEALS).map_err(|e| format!("F_GET_SEALS: {e}"))?;
        let seals = SealFlag::from_bits_truncate(raw);
        let need = SealFlag::F_SEAL_WRITE | SealFlag::F_SEAL_SHRINK | SealFlag::F_SEAL_GROW;
        if !seals.contains(need) {
            return Err(String::from("descriptor is not sealed read-only"));
        }

        let mut raw_trailer = [0u8; TRAILER_SIZE];
        let read = pread(fd, &mut raw_trailer, trailer_offset(len) as i64)
            .map_err(|e| format!("pread trailer: {e}"))?;
        if read != TRAILER_SIZE {
            return Err(String::from("short trailer"));
        }
        let got = Trailer::from_bytes(&raw_trailer);
        if got.magic != TRAILER_MAGIC {
            return Err(String::from("not one of our images"));
        }
        if got.format != TRAILER_FORMAT {
            return Err(format!("trailer format {} unknown", got.format));
        }
        if got.name_hash != crate::nnue::integrity::fnv1a64(name.as_bytes()) {
            return Err(String::from("image was published under another name"));
        }
        if got.payload_len != len as u64 {
            return Err(format!("payload is {} bytes, expected {len}", got.payload_len));
        }
        Ok(())
    }

    /// The payload digest recorded at build time, for the checking mode.
    fn recorded_hash(fd: BorrowedFd<'_>, len: usize) -> Result<u64, String> {
        let mut raw = [0u8; TRAILER_SIZE];
        let read = pread(fd, &mut raw, trailer_offset(len) as i64)
            .map_err(|e| format!("pread trailer: {e}"))?;
        if read != TRAILER_SIZE {
            return Err(String::from("short trailer"));
        }
        Ok(Trailer::from_bytes(&raw).payload_hash)
    }

    /// Map `size` bytes of `fd` at a 2 MiB boundary.
    ///
    /// The alignment is asked for rather than hoped for: it is what lets the
    /// kernel use huge pages, and on a host whose mount does not align shared
    /// mappings itself there would otherwise be none. A reservation one huge
    /// page larger than needed is taken first, the file is mapped over the
    /// aligned part of it, and the two ends are given back.
    fn map_at_huge_boundary(
        fd: BorrowedFd<'_>,
        size: usize,
        prot: ProtFlags,
    ) -> Result<*mut u8, String> {
        use std::num::NonZeroUsize;

        let reserve = NonZeroUsize::new(size + HUGE).ok_or("empty mapping")?;
        // SAFETY: an anonymous PROT_NONE reservation owns no memory and can
        // alias nothing; the kernel picks the address.
        let base = unsafe {
            mmap_anonymous(
                None,
                reserve,
                ProtFlags::PROT_NONE,
                MapFlags::MAP_PRIVATE | MapFlags::MAP_NORESERVE,
            )
        }
        .map_err(|e| format!("reserve: {e}"))?;

        let base_addr = base.as_ptr() as usize;
        let aligned = round_up(base_addr, HUGE);
        let head = aligned - base_addr;
        let tail = HUGE - head;

        let want = NonZeroUsize::new(size).ok_or("empty mapping")?;
        // SAFETY: the target range is inside the reservation this call owns, so
        // MAP_FIXED replaces only our own mapping.
        let mapped = unsafe {
            mmap(
                NonZeroUsize::new(aligned),
                want,
                prot,
                MapFlags::MAP_SHARED | MapFlags::MAP_FIXED,
                fd,
                0,
            )
        };
        let mapped = match mapped {
            Ok(p) => p,
            Err(e) => {
                // SAFETY: unmapping exactly the reservation made above.
                unsafe { let _ = munmap(base, reserve.get()); }
                return Err(format!("mmap: {e}"));
            }
        };

        // Give back the slivers around the aligned mapping.
        if head > 0 {
            // SAFETY: still our own reservation, untouched by the fixed mapping.
            unsafe { let _ = munmap(base, head); }
        }
        if tail > 0 {
            let end = std::ptr::NonNull::new((aligned + size) as *mut std::ffi::c_void);
            if let Some(end) = end {
                // SAFETY: as above, the far end of the reservation.
                unsafe { let _ = munmap(end, tail); }
            }
        }

        let ptr = mapped.as_ptr() as *mut u8;
        debug_assert_eq!(ptr as usize % HUGE, 0);
        Ok(ptr)
    }

    /// Ask for huge pages, then for the mapping to be resident.
    ///
    /// `MADV_COLLAPSE` is the one that works on a host whose shmem THP policy
    /// is `never`; the other two are no-ops where the kernel already did the
    /// right thing. All three are advice: failure is not an error.
    fn advise(ptr: *mut u8, size: usize, write: bool) -> Result<(), String> {
        let nn = std::ptr::NonNull::new(ptr as *mut std::ffi::c_void).ok_or("null mapping")?;
        // SAFETY: the range was mapped by this module and is still mapped.
        unsafe {
            let _ = madvise(nn, size, MmapAdvise::MADV_HUGEPAGE);
            let _ = nix::libc::madvise(ptr as *mut std::ffi::c_void, size, MADV_COLLAPSE);
        }
        if write {
            // Faulting every page in advance is not an optimisation: writing to
            // an unbacked page of a memfd on a full machine raises SIGBUS, and
            // the release build aborts on signals it cannot unwind. This turns
            // that crash into an error we can fall back from.
            // SAFETY: as above.
            let populated = unsafe { madvise(nn, size, MmapAdvise::MADV_POPULATE_WRITE) };
            if let Err(e) = populated {
                return Err(format!("cannot back {size} bytes: {e}"));
            }
        } else {
            // SAFETY: as above.
            unsafe {
                let _ = madvise(nn, size, MmapAdvise::MADV_POPULATE_READ);
            }
        }
        Ok(())
    }

    /// Build the image in fresh anonymous kernel memory and seal it.
    fn create_image(
        name: &str,
        len: usize,
        build: &mut dyn FnMut(&mut [u8]),
    ) -> Result<OwnedFd, String> {
        let size = image_size(len);
        let fd = memfd_create(
            c"gaiachess-net",
            MFdFlags::MFD_CLOEXEC | MFdFlags::MFD_ALLOW_SEALING,
        )
        .map_err(|e| format!("memfd_create: {e}"))?;
        ftruncate(&fd, size as i64).map_err(|e| format!("ftruncate: {e}"))?;

        let ptr = map_at_huge_boundary(
            fd.as_fd(),
            size,
            ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
        )?;
        advise(ptr, size, true)?;

        // SAFETY: `size` writable bytes were mapped at `ptr`, and this process
        // is the only one that can reach them until the descriptor is served.
        let image = unsafe { std::slice::from_raw_parts_mut(ptr, size) };
        build(&mut image[..len]);
        let hash = crate::nnue::integrity::fnv1a64(&image[..len]);
        let off = trailer_offset(len);
        image[len..off].fill(0);
        image[off..off + TRAILER_SIZE]
            .copy_from_slice(Trailer::expected(name, len, hash).as_bytes());
        image[off + TRAILER_SIZE..].fill(0);

        let nn = std::ptr::NonNull::new(ptr as *mut std::ffi::c_void).ok_or("null mapping")?;
        // SAFETY: unmapping exactly what map_at_huge_boundary returned. The
        // contents stay alive in the memfd.
        unsafe { munmap(nn, size).map_err(|e| format!("munmap: {e}"))? };

        // From here the bytes cannot change for anyone, ourselves included.
        fcntl(
            fd.as_fd(),
            FcntlArg::F_ADD_SEALS(
                SealFlag::F_SEAL_WRITE
                    | SealFlag::F_SEAL_SHRINK
                    | SealFlag::F_SEAL_GROW
                    | SealFlag::F_SEAL_SEAL,
            ),
        )
        .map_err(|e| format!("F_ADD_SEALS: {e}"))?;
        Ok(fd)
    }

    /// Names this process already answers for, so a second acquire of the same
    /// blob (the UCI option being turned off and on) does not collide with the
    /// server it started the first time.
    fn served() -> &'static Mutex<HashSet<String>> {
        static SERVED: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
        SERVED.get_or_init(|| Mutex::new(HashSet::new()))
    }

    /// Hand the descriptor to anyone who asks for it, for as long as we live.
    ///
    /// The thread is never joined and never told to stop: it is parked in
    /// `accept` and dies with the process, at which point the kernel drops the
    /// abstract name and, if we were the last holder, the memory.
    fn serve(name: &str, fd: OwnedFd) -> Result<(), String> {
        let socket = format!("{}{}", serve_prefix(name), std::process::id());
        {
            let mut set = served().lock().unwrap_or_else(|e| e.into_inner());
            if !set.insert(socket.clone()) {
                return Ok(()); // already serving this blob
            }
        }
        let addr = abstract_addr(&socket)?;
        let listener = UnixListener::bind_addr(&addr).map_err(|e| format!("bind serve: {e}"))?;

        std::thread::Builder::new()
            .name(String::from("gaia-shm"))
            .stack_size(64 * 1024)
            .spawn(move || {
                for stream in listener.incoming() {
                    let Ok(stream) = stream else { continue };
                    let same_user = getsockopt(&stream, sockopt::PeerCredentials)
                        .map(|c| c.uid() == euid())
                        .unwrap_or(false);
                    if !same_user {
                        continue;
                    }
                    let fds = [fd.as_raw_fd()];
                    let cmsg = [ControlMessage::ScmRights(&fds)];
                    let iov = [IoSlice::new(b"G")];
                    // MSG_NOSIGNAL: a client that vanished mid-handshake must
                    // not take the engine down with a SIGPIPE.
                    let _ = sendmsg::<()>(
                        stream.as_raw_fd(),
                        &iov,
                        &cmsg,
                        MsgFlags::MSG_NOSIGNAL,
                        None,
                    );
                }
            })
            .map_err(|e| format!("serve thread: {e}"))?;
        Ok(())
    }

    /// Map a validated descriptor read-only and start serving it onward.
    fn adopt(fd: OwnedFd, name: &str, len: usize, origin: Origin) -> Result<Mapped, String> {
        let size = image_size(len);
        let ptr = map_at_huge_boundary(fd.as_fd(), size, ProtFlags::PROT_READ)?;
        advise(ptr, size, false)?;

        if super::verify_requested() {
            let checked = recorded_hash(fd.as_fd(), len).and_then(|recorded| {
                // SAFETY: `len` bytes of the read-only mapping just made.
                let actual = crate::nnue::integrity::fnv1a64(unsafe {
                    std::slice::from_raw_parts(ptr, len)
                });
                if recorded == actual {
                    Ok(())
                } else {
                    Err(format!(
                        "image hashes to {actual:#018x}, trailer says {recorded:#018x}"
                    ))
                }
            });
            if let Err(e) = checked {
                // Give the mapping back before walking away: holding it would
                // also hold the memfd, and the image we just rejected would
                // outlive the process that published it.
                if let Some(nn) = std::ptr::NonNull::new(ptr as *mut std::ffi::c_void) {
                    // SAFETY: unmapping exactly what was mapped above.
                    unsafe { let _ = munmap(nn, size); }
                }
                return Err(e);
            }
        }

        // Passing the descriptor on is what keeps the mesh alive once the
        // builder exits, but failing to do so costs only that: this process
        // still reads the shared pages, which is the whole point.
        let _ = serve(name, fd);
        debug_assert_eq!(ptr as usize % 64, 0);
        Ok(Mapped { ptr: ptr as *const u8, len, origin })
    }

    pub fn acquire(
        name: &str,
        len: usize,
        build: &mut dyn FnMut(&mut [u8]),
    ) -> Result<Mapped, String> {
        debug_assert!(len > 0);
        let prefix = serve_prefix(name);
        let lock_name = init_name(name);
        let deadline = Instant::now() + BUILD_WAIT;
        // Kept across turns of the loop: whatever went genuinely wrong, as
        // opposed to the ordinary "somebody else is building, come back".
        let mut trouble: Option<String> = None;

        loop {
            // Somebody may already hold what we want.
            for peer in scan_peers(&prefix) {
                match receive_from(&peer).and_then(|fd| {
                    validate(fd.as_fd(), name, len)?;
                    adopt(fd, name, len, Origin::Received)
                }) {
                    Ok(mapped) => return Ok(mapped),
                    Err(e) => trouble = Some(format!("peer {peer}: {e}")),
                }
            }

            // Nobody did. Claiming this name is claiming the work.
            let addr = abstract_addr(&lock_name)?;
            match UnixListener::bind_addr(&addr) {
                Ok(lock) => {
                    let built = create_image(name, len, build)
                        .and_then(|fd| adopt(fd, name, len, Origin::Built));
                    // Released only here: a waiter that sees the name free must
                    // already be able to find the serving socket.
                    drop(lock);
                    return built;
                }
                // The expected case while another process builds.
                Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => {}
                Err(e) => return Err(format!("bind init: {e}")),
            }

            if Instant::now() >= deadline {
                return Err(trouble.unwrap_or_else(|| {
                    format!("waited {BUILD_WAIT:?} for another process to build the image")
                }));
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}

// ============================================================
// Everywhere else: no sharing, same signatures
// ============================================================

#[cfg(not(target_os = "linux"))]
mod imp {
    use super::Mapped;

    pub const SUPPORTED: bool = false;

    pub fn peers() -> usize {
        0
    }

    pub fn acquire(
        _name: &str,
        _len: usize,
        _build: &mut dyn FnMut(&mut [u8]),
    ) -> Result<Mapped, String> {
        Err(String::from("sharing needs memfd and abstract sockets (Linux)"))
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// A name no other test and no running engine can collide with. The
    /// abstract namespace is machine-wide, so this matters.
    fn unique(tag: &str) -> String {
        static SEQ: AtomicU32 = AtomicU32::new(0);
        format!(
            "test-{tag}-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        )
    }

    fn pattern(dst: &mut [u8]) {
        for (i, b) in dst.iter_mut().enumerate() {
            *b = (i * 31 + 7) as u8;
        }
    }

    const SMALL: usize = 1024 * 1024;

    #[test]
    fn a_second_asker_is_handed_the_first_ones_memory() {
        let name = unique("roundtrip");
        let built = acquire(&name, SMALL, &mut pattern).expect("build");
        assert_eq!(built.origin, Origin::Built);

        // The second acquire must find the server started by the first and
        // take the descriptor from it rather than build anything.
        let received = acquire(&name, SMALL, &mut |_| panic!("rebuilt an image it should have received"))
            .expect("receive");
        assert_eq!(received.origin, Origin::Received);
        assert_eq!(built.bytes(), received.bytes());

        let mut expected = vec![0u8; SMALL];
        pattern(&mut expected);
        assert_eq!(built.bytes(), &expected[..]);
    }

    #[test]
    fn the_image_is_aligned_for_huge_pages() {
        let name = unique("aligned");
        let m = acquire(&name, SMALL, &mut pattern).expect("build");
        assert_eq!(m.bytes().as_ptr() as usize % (2 * 1024 * 1024), 0);
    }

    #[test]
    fn racing_askers_end_up_on_one_image() {
        let name = unique("race");
        let builds = std::sync::Arc::new(AtomicU32::new(0));
        let mut handles = Vec::new();
        for _ in 0..4 {
            let name = name.clone();
            let builds = builds.clone();
            handles.push(std::thread::spawn(move || {
                let m = acquire(&name, SMALL, &mut |dst| {
                    builds.fetch_add(1, Ordering::Relaxed);
                    pattern(dst);
                })
                .expect("acquire");
                (m.origin, m.bytes()[..64].to_vec())
            }));
        }
        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // One builder, and everyone reading the same bytes. More than one build
        // would mean the init lock let two processes through.
        assert_eq!(builds.load(Ordering::Relaxed), 1, "the image was built twice");
        assert_eq!(
            results.iter().filter(|(o, _)| *o == Origin::Built).count(),
            1
        );
        let first = &results[0].1;
        assert!(results.iter().all(|(_, b)| b == first));
    }

    #[test]
    fn an_image_of_the_wrong_size_is_refused() {
        let name = unique("size");
        acquire(&name, SMALL, &mut pattern).expect("build");

        // Same name, different payload length: the trailer says so, and the
        // asker must build its own rather than read the wrong extent.
        let mut rebuilt = false;
        let m = acquire(&name, SMALL * 2, &mut |dst| {
            rebuilt = true;
            pattern(dst);
        });
        // Either it refused the peer and built (a different memfd, same name is
        // impossible to bind twice, so it waits then times out) — both outcomes
        // are acceptable, reading the short image is not.
        match m {
            Ok(m) => {
                assert!(rebuilt, "adopted an image of the wrong size");
                assert_eq!(m.bytes().len(), SMALL * 2);
            }
            Err(e) => assert!(!e.is_empty()),
        }
    }

    #[test]
    fn a_name_too_long_for_a_socket_is_an_error_not_a_panic() {
        let name = "x".repeat(200);
        let err = acquire(&name, SMALL, &mut pattern).unwrap_err();
        assert!(err.contains("unusable"), "{err}");
    }

    /// The size the engine actually shares. Worth its ~40 ms because the
    /// alignment dance and the huge-page advice only bite at scale.
    #[test]
    fn a_network_sized_image_round_trips() {
        let len = size_of::<crate::nnue::network::NNUEParams>();
        let name = unique("fullsize");
        let m = acquire(&name, len, &mut |dst| {
            dst.fill(0xa5);
            dst[..8].copy_from_slice(&0x0123_4567_89ab_cdefu64.to_le_bytes());
        })
        .expect("build");
        assert_eq!(m.bytes().len(), len);
        assert_eq!(m.bytes()[len - 1], 0xa5);
        assert_eq!(
            u64::from_le_bytes(m.bytes()[..8].try_into().unwrap()),
            0x0123_4567_89ab_cdef
        );

        let received = acquire(&name, len, &mut |_| panic!("rebuilt")).expect("receive");
        assert_eq!(received.origin, Origin::Received);
        assert_eq!(received.bytes().as_ptr() as usize % 64, 0);
    }
}
