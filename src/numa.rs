//! NUMA topology, thread binding and node-local allocation.
//!
//! On a machine with several memory nodes, a page lives on the node of the thread that
//! first wrote it, and a thread reading another node's memory pays for the trip across
//! the interconnect. With hundreds of search threads that trip dominates: the network
//! weights, the transposition table and every thread's own tables would otherwise all
//! sit on whichever node the UCI thread happened to run on.
//!
//! This module describes the machine as a list of *logical nodes* — sets of processors
//! that share memory and, by default, a bundle of last-level caches — and offers three
//! things on top of it: spreading search threads over the nodes, pinning a thread to its
//! node, and running a closure on a given node so that what it allocates lands there.
//!
//! The topology code is pure and tested on hand-written machines; only reading the
//! machine and pinning a thread go through the operating system, on Linux and 64-bit
//! Windows. Everywhere else the machine is a single node and binding does nothing.

use std::collections::BTreeMap;
use std::sync::OnceLock;

/// Processors in a Windows processor group.
pub const WIN_GROUP_SIZE: usize = 64;

/// Last-level cache domains of one memory node are merged while the bundle stays this
/// small: big enough that a node of a desktop CPU stays whole, small enough that the
/// dozens of cache domains of a server socket become a handful of nodes.
pub const L3_BUNDLE_SIZE: usize = 32;

/// A node below this fraction of the largest one does not count when deciding whether
/// enough threads exist to spread them out.
const SMALL_NODE_FRACTION: f64 = 0.6;

/// The machine as seen by the search: non-empty logical nodes of processors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NumaConfig {
    /// Sorted, disjoint, never empty.
    nodes: Vec<Vec<usize>>,
    /// For each logical node, the hardware memory node it belongs to. Read-only data
    /// is replicated per hardware node, not per logical node.
    system_node: Vec<usize>,
    /// The processors were given by the user, or the process affinity was ignored:
    /// the threads must then be pinned whatever their number.
    custom: bool,
}

impl NumaConfig {
    /// One node holding the processors `0..n`.
    pub fn single(n: usize) -> Self {
        Self::from_nodes(vec![(0..n.max(1)).collect()])
    }

    /// Builds a configuration from processor lists, dropping empty nodes. Every node is
    /// its own hardware node.
    pub fn from_nodes(nodes: Vec<Vec<usize>>) -> Self {
        let mut nodes: Vec<Vec<usize>> = nodes
            .into_iter()
            .map(|mut cpus| {
                cpus.sort_unstable();
                cpus.dedup();
                cpus
            })
            .filter(|cpus| !cpus.is_empty())
            .collect();
        if nodes.is_empty() {
            nodes.push(vec![0]);
        }
        let system_node = (0..nodes.len()).collect();
        NumaConfig {
            nodes,
            system_node,
            custom: false,
        }
    }

    /// Parses a user configuration: nodes separated by `:`, processors by `,`, with
    /// `first-last` ranges — `"0-15,128-143:16-31,144-159"`. `None` if nothing parses
    /// or a processor appears twice.
    pub fn from_user_string(s: &str) -> Option<Self> {
        let mut nodes = Vec::new();
        let mut seen = std::collections::BTreeSet::new();
        for part in s.split(':') {
            let cpus = parse_cpulist(part)?;
            if cpus.is_empty() {
                continue;
            }
            for &c in &cpus {
                if !seen.insert(c) {
                    return None;
                }
            }
            nodes.push(cpus);
        }
        if nodes.is_empty() {
            return None;
        }
        let mut cfg = Self::from_nodes(nodes);
        cfg.custom = true;
        Some(cfg)
    }

    /// Merges last-level cache domains into bundles of at most `bundle` processors,
    /// never across hardware nodes. `domains` holds `(hardware node, processors)` in the
    /// order the processors were discovered; adjacent domains are merged pairwise, pass
    /// after pass, until no two neighbours fit together.
    pub fn from_l3_domains(domains: Vec<(usize, Vec<usize>)>, bundle: usize) -> Self {
        debug_assert!(bundle > 0, "from_l3_domains: zero bundle size");
        let mut by_node: BTreeMap<usize, Vec<Vec<usize>>> = BTreeMap::new();
        for (node, cpus) in domains {
            if !cpus.is_empty() {
                by_node.entry(node).or_default().push(cpus);
            }
        }
        let mut nodes = Vec::new();
        let mut system_node = Vec::new();
        for (node, mut ds) in by_node {
            loop {
                let mut changed = false;
                let mut j = 0;
                while j + 1 < ds.len() {
                    if ds[j].len() + ds[j + 1].len() <= bundle {
                        let next = ds.remove(j + 1);
                        ds[j].extend(next);
                        changed = true;
                    }
                    j += 1;
                }
                if !changed {
                    break;
                }
            }
            for mut cpus in ds {
                cpus.sort_unstable();
                cpus.dedup();
                nodes.push(cpus);
                system_node.push(node);
            }
        }
        if nodes.is_empty() {
            return Self::single(1);
        }
        NumaConfig {
            nodes,
            system_node,
            custom: false,
        }
    }

    /// Splits every node at processor-group boundaries. A Windows thread is scheduled
    /// within its primary group first, even when its affinity spans several, so a node
    /// straddling two groups behaves as two.
    pub fn split_at_groups(&self, group: usize) -> Self {
        debug_assert!(group > 0, "split_at_groups: zero group size");
        let mut nodes = Vec::new();
        let mut system_node = Vec::new();
        for (cpus, &sys) in self.nodes.iter().zip(&self.system_node) {
            let mut current: Vec<usize> = Vec::new();
            for &c in cpus {
                if let Some(&last) = current.last()
                    && last / group != c / group
                {
                    nodes.push(std::mem::take(&mut current));
                    system_node.push(sys);
                }
                current.push(c);
            }
            if !current.is_empty() {
                nodes.push(current);
                system_node.push(sys);
            }
        }
        NumaConfig {
            nodes,
            system_node,
            custom: self.custom,
        }
    }

    /// Marks the configuration as imposed: the threads will be pinned whatever their
    /// number.
    pub fn into_custom(mut self) -> Self {
        self.custom = true;
        self
    }

    /// Names, for every logical node, the hardware node holding its first processor in
    /// `hardware`. Processors `hardware` does not know fall back to node 0.
    pub fn with_system_nodes_from(mut self, hardware: &NumaConfig) -> Self {
        self.system_node = self
            .nodes
            .iter()
            .map(|cpus| hardware.node_of_cpu(cpus[0]).unwrap_or(0))
            .collect();
        self
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn nodes(&self) -> &[Vec<usize>] {
        &self.nodes
    }

    pub fn cpus(&self, node: usize) -> &[usize] {
        &self.nodes[node]
    }

    pub fn cpu_count(&self) -> usize {
        self.nodes.iter().map(Vec::len).sum()
    }

    pub fn highest_cpu(&self) -> usize {
        self.nodes
            .iter()
            .filter_map(|n| n.last())
            .copied()
            .max()
            .unwrap_or(0)
    }

    pub fn is_custom(&self) -> bool {
        self.custom
    }

    /// The hardware node of logical node `node`.
    pub fn system_node_of(&self, node: usize) -> usize {
        debug_assert!(
            node < self.nodes.len(),
            "system_node_of: node {node} out of range"
        );
        self.system_node[node]
    }

    /// Distinct hardware nodes, in ascending order.
    pub fn system_nodes(&self) -> Vec<usize> {
        let mut v = self.system_node.clone();
        v.sort_unstable();
        v.dedup();
        v
    }

    pub fn node_of_cpu(&self, cpu: usize) -> Option<usize> {
        self.nodes
            .iter()
            .position(|cpus| cpus.binary_search(&cpu).is_ok())
    }

    /// Whether node-local copies of shared data make sense at all.
    pub fn requires_replication(&self) -> bool {
        self.custom || self.nodes.len() > 1
    }

    /// Whether `threads` search threads should be pinned and spread over the nodes.
    ///
    /// Unpinned threads all read what was replicated for the first node, so once the
    /// operating system has to run them elsewhere, every access is remote. Pinning is
    /// advised when the threads cannot fit in half of the largest node, or when there
    /// are enough of them to give every sizeable node four. An imposed configuration
    /// is always pinned: the operating system knows nothing of it.
    pub fn suggests_binding(&self, threads: usize) -> bool {
        if self.custom {
            return true;
        }
        if threads <= 1 || self.nodes.len() <= 1 {
            return false;
        }
        let largest = self.nodes.iter().map(Vec::len).max().unwrap_or(1);
        let not_small = self
            .nodes
            .iter()
            .filter(|cpus| cpus.len() as f64 / largest as f64 > SMALL_NODE_FRACTION)
            .count();
        threads > largest / 2 || threads >= not_small * 4
    }

    /// The node of each of `threads` threads: every thread goes to the node it fills
    /// least, relative to the node's size. Ties go to the lower node.
    pub fn distribute(&self, threads: usize) -> Vec<usize> {
        if self.nodes.len() == 1 {
            return vec![0; threads];
        }
        let mut occupation = vec![0usize; self.nodes.len()];
        let mut out = Vec::with_capacity(threads);
        for _ in 0..threads {
            let mut best = 0;
            let mut best_fill = f64::INFINITY;
            for (n, cpus) in self.nodes.iter().enumerate() {
                let fill = (occupation[n] + 1) as f64 / cpus.len() as f64;
                if fill < best_fill {
                    best = n;
                    best_fill = fill;
                }
            }
            occupation[best] += 1;
            out.push(best);
        }
        out
    }
}

impl std::fmt::Display for NumaConfig {
    /// The user syntax: `0-15,128-143:16-31`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, cpus) in self.nodes.iter().enumerate() {
            if i > 0 {
                f.write_str(":")?;
            }
            let mut first = true;
            let mut start = 0;
            while start < cpus.len() {
                let mut end = start;
                while end + 1 < cpus.len() && cpus[end + 1] == cpus[end] + 1 {
                    end += 1;
                }
                if !first {
                    f.write_str(",")?;
                }
                if end > start {
                    write!(f, "{}-{}", cpus[start], cpus[end])?;
                } else {
                    write!(f, "{}", cpus[start])?;
                }
                first = false;
                start = end + 1;
            }
        }
        Ok(())
    }
}

/// Parses a kernel-style processor list, `"0-3,8,10-11"`. Whitespace is ignored and an
/// empty list is valid. `None` on anything malformed.
pub fn parse_cpulist(s: &str) -> Option<Vec<usize>> {
    /// A range wider than this is a typo, not a machine.
    const MAX_RANGE: usize = 1 << 20;
    let s: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    let mut out = Vec::new();
    for part in s.split(',') {
        if part.is_empty() {
            continue;
        }
        match part.split_once('-') {
            None => out.push(part.parse().ok()?),
            Some((a, b)) => {
                let a: usize = a.parse().ok()?;
                let b: usize = b.parse().ok()?;
                if b < a || b - a >= MAX_RANGE {
                    return None;
                }
                out.extend(a..=b);
            }
        }
    }
    Some(out)
}

/// How the search threads are laid out on the machine (UCI `NumaPolicy`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NumaPolicy {
    /// Cache bundles within memory nodes; threads pinned when that pays.
    Auto,
    /// Memory nodes as the system reports them; threads always pinned.
    System,
    /// Like `System`, ignoring the affinity the process was started with.
    Hardware,
    /// One node, nothing pinned.
    None,
    /// The user's own processor lists; threads always pinned.
    Custom(NumaConfig),
}

impl NumaPolicy {
    /// Parses the option value. `None` when it is neither a keyword nor a valid
    /// processor list.
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" | "" => Some(NumaPolicy::Auto),
            "system" => Some(NumaPolicy::System),
            "hardware" => Some(NumaPolicy::Hardware),
            "none" => Some(NumaPolicy::None),
            other => NumaConfig::from_user_string(other).map(NumaPolicy::Custom),
        }
    }

    /// The machine as this policy sees it.
    pub fn config(&self) -> NumaConfig {
        match self {
            NumaPolicy::Auto => system_config(true, true),
            NumaPolicy::System => system_config(true, false).into_custom(),
            NumaPolicy::Hardware => system_config(false, false).into_custom(),
            NumaPolicy::None => NumaConfig::single(hardware_concurrency()),
            NumaPolicy::Custom(cfg) => cfg.clone().with_system_nodes_from(hardware_config()),
        }
    }

    /// Whether `threads` threads are pinned under this policy.
    pub fn binds(&self, cfg: &NumaConfig, threads: usize) -> bool {
        match self {
            NumaPolicy::None => false,
            NumaPolicy::Auto => cfg.suggests_binding(threads),
            _ => true,
        }
    }
}

/// Processors the machine has, across every Windows processor group.
pub fn hardware_concurrency() -> usize {
    let std = std::thread::available_parallelism().map_or(1, |n| n.get());
    sys::hardware_concurrency().max(std).max(1)
}

/// Records the processors the process was allowed to run on, before any thread is
/// pinned. Call first thing in `main`; later calls are no-ops.
pub fn capture_startup_affinity() {
    let _ = startup_affinity();
}

/// The processors the process may run on, as it was started — `None` for all of them.
fn startup_affinity() -> &'static Option<Vec<usize>> {
    static AFFINITY: OnceLock<Option<Vec<usize>>> = OnceLock::new();
    AFFINITY.get_or_init(sys::process_affinity)
}

/// The memory nodes regardless of process affinity, used to name hardware nodes.
fn hardware_config() -> &'static NumaConfig {
    static HARDWARE: OnceLock<NumaConfig> = OnceLock::new();
    HARDWARE.get_or_init(|| system_config(false, false))
}

/// A fingerprint of the machine's memory nodes, so that per-node images shared between
/// processes are only shared by processes that see the same nodes.
pub fn hardware_signature() -> u32 {
    let text = hardware_config().to_string();
    text.bytes().fold(0x811c_9dc5u32, |h, b| {
        (h ^ b as u32).wrapping_mul(0x0100_0193)
    })
}

/// Reads the machine. `respect_affinity` keeps only the processors the process was
/// started on; `bundle_l3` subdivides memory nodes into bundles of cache domains.
fn system_config(respect_affinity: bool, bundle_l3: bool) -> NumaConfig {
    let allowed = if respect_affinity {
        startup_affinity().clone()
    } else {
        None
    };
    let is_allowed = |c: usize| allowed.as_ref().is_none_or(|a| a.binary_search(&c).is_ok());

    let memory = match sys::memory_nodes() {
        Some(nodes) => NumaConfig::from_nodes(
            nodes
                .into_iter()
                .map(|cpus| cpus.into_iter().filter(|&c| is_allowed(c)).collect())
                .collect(),
        ),
        None => NumaConfig::from_nodes(vec![
            (0..hardware_concurrency())
                .filter(|&c| is_allowed(c))
                .collect(),
        ]),
    };

    let mut cfg = memory.clone();
    if bundle_l3 && let Some(domains) = sys::l3_domains() {
        let domains: Vec<(usize, Vec<usize>)> = domains
            .into_iter()
            .filter_map(|cpus| {
                let cpus: Vec<usize> = cpus.into_iter().filter(|&c| is_allowed(c)).collect();
                let node = memory.node_of_cpu(*cpus.first()?)?;
                Some((node, cpus))
            })
            .collect();
        let bundled = NumaConfig::from_l3_domains(domains, L3_BUNDLE_SIZE);
        // Cache information that misses processors would lose them: keep the plain
        // memory nodes instead.
        if bundled.cpu_count() == memory.cpu_count() {
            cfg = bundled;
        }
    }

    if cfg!(all(windows, target_pointer_width = "64")) {
        cfg = cfg.split_at_groups(WIN_GROUP_SIZE);
    }
    cfg
}

/// Pins the calling thread to the processors of `node`.
pub fn bind_current_thread(cfg: &NumaConfig, node: usize) {
    debug_assert!(
        node < cfg.node_count(),
        "bind_current_thread: node {node} out of range"
    );
    sys::bind_current_thread(cfg.cpus(node), cfg.highest_cpu());
}

/// Runs `f` on a thread pinned to `node` and returns its result. Pages that `f` writes
/// first are placed on that node.
pub fn run_on_node<R: Send>(cfg: &NumaConfig, node: usize, f: impl FnOnce() -> R + Send) -> R {
    std::thread::scope(|s| {
        std::thread::Builder::new()
            .name("gaia-numa".into())
            .stack_size(32 * 1024 * 1024)
            .spawn_scoped(s, || {
                bind_current_thread(cfg, node);
                f()
            })
            .expect("failed to spawn a NUMA placement thread")
            .join()
            .expect("NUMA placement thread panicked")
    })
}

// ---------------------------------------------------------------------------------------
// Operating system layer
// ---------------------------------------------------------------------------------------

#[cfg(target_os = "linux")]
mod sys {
    //! Linux: sysfs for the topology, `sched_{get,set}affinity` with masks sized at run
    //! time, so machines past 1024 processors are not cut short.

    use super::parse_cpulist;

    /// Bits of the mask used to read the process affinity.
    const MAX_CPUS: usize = 64 * 1024;
    const WORD_BITS: usize = usize::BITS as usize;

    pub fn hardware_concurrency() -> usize {
        1
    }

    fn read(path: &str) -> Option<String> {
        std::fs::read_to_string(path).ok()
    }

    pub fn process_affinity() -> Option<Vec<usize>> {
        let mut mask = vec![0usize; MAX_CPUS / WORD_BITS];
        let bytes = mask.len() * size_of::<usize>();
        // SAFETY: the buffer is `bytes` long and lives for the call.
        let rc = unsafe {
            libc::sched_getaffinity(0, bytes, mask.as_mut_ptr().cast::<libc::cpu_set_t>())
        };
        if rc != 0 {
            return None;
        }
        let cpus: Vec<usize> = (0..MAX_CPUS)
            .filter(|&c| mask[c / WORD_BITS] >> (c % WORD_BITS) & 1 == 1)
            .collect();
        (!cpus.is_empty()).then_some(cpus)
    }

    pub fn memory_nodes() -> Option<Vec<Vec<usize>>> {
        let online = parse_cpulist(&read("/sys/devices/system/node/online")?)?;
        if online.is_empty() {
            return None;
        }
        online
            .into_iter()
            .map(|n| parse_cpulist(&read(&format!("/sys/devices/system/node/node{n}/cpulist"))?))
            .collect()
    }

    pub fn l3_domains() -> Option<Vec<Vec<usize>>> {
        let mut seen = std::collections::BTreeSet::new();
        let mut domains = Vec::new();
        let possible = parse_cpulist(&read("/sys/devices/system/cpu/possible")?)?;
        for c in possible {
            if seen.contains(&c) {
                continue;
            }
            let base = format!("/sys/devices/system/cpu/cpu{c}/cache/index3");
            if read(&format!("{base}/level")).as_deref().map(str::trim) != Some("3") {
                continue;
            }
            let Some(list) = read(&format!("{base}/shared_cpu_list")) else {
                continue;
            };
            let Some(cpus) = parse_cpulist(&list) else {
                continue;
            };
            seen.extend(cpus.iter().copied());
            if !cpus.is_empty() {
                domains.push(cpus);
            }
        }
        (!domains.is_empty()).then_some(domains)
    }

    pub fn bind_current_thread(cpus: &[usize], highest: usize) {
        let words = highest / WORD_BITS + 1;
        let mut mask = vec![0usize; words];
        for &c in cpus {
            mask[c / WORD_BITS] |= 1 << (c % WORD_BITS);
        }
        // SAFETY: the mask is `words` words long and lives for the call.
        let rc = unsafe {
            libc::sched_setaffinity(
                0,
                words * size_of::<usize>(),
                mask.as_ptr().cast::<libc::cpu_set_t>(),
            )
        };
        if rc != 0 {
            outerr!(
                "info string NUMA: could not pin a thread ({})",
                std::io::Error::last_os_error()
            );
            return;
        }
        // SAFETY: no arguments.
        unsafe { libc::sched_yield() };
    }
}

#[cfg(all(windows, target_pointer_width = "64"))]
mod sys {
    //! 64-bit Windows: processor groups of 64, `GetNumaProcessorNodeEx` for the memory
    //! nodes, and two affinity interfaces — the CPU-set one, which spans groups but
    //! exists only from Windows 11 and Server 2022 and is therefore looked up at run
    //! time, and the group one, which every version has.

    use super::WIN_GROUP_SIZE;
    use std::ffi::c_void;
    use std::sync::OnceLock;

    type Handle = *mut c_void;
    type Bool = i32;

    #[repr(C)]
    struct ProcessorNumber {
        group: u16,
        number: u8,
        reserved: u8,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct GroupAffinity {
        mask: usize,
        group: u16,
        reserved: [u16; 3],
    }

    const ALL_PROCESSOR_GROUPS: u16 = 0xffff;
    const ERROR_INSUFFICIENT_BUFFER: u32 = 122;
    const RELATION_CACHE: i32 = 2;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetActiveProcessorGroupCount() -> u16;
        fn GetActiveProcessorCount(group: u16) -> u32;
        fn GetNumaProcessorNodeEx(processor: *const ProcessorNumber, node: *mut u16) -> Bool;
        fn GetCurrentThread() -> Handle;
        fn GetCurrentProcess() -> Handle;
        fn SetThreadGroupAffinity(
            thread: Handle,
            affinity: *const GroupAffinity,
            previous: *mut GroupAffinity,
        ) -> Bool;
        fn GetProcessAffinityMask(process: Handle, mask: *mut usize, system: *mut usize) -> Bool;
        fn GetProcessGroupAffinity(process: Handle, count: *mut u16, groups: *mut u16) -> Bool;
        fn GetLogicalProcessorInformationEx(
            relation: i32,
            buffer: *mut u8,
            length: *mut u32,
        ) -> Bool;
        fn GetLastError() -> u32;
        fn SwitchToThread() -> Bool;
        fn GetModuleHandleA(name: *const u8) -> Handle;
        fn GetProcAddress(module: Handle, name: *const u8) -> *mut c_void;
    }

    type SetCpuSetMasks = unsafe extern "system" fn(Handle, *const GroupAffinity, u16) -> Bool;
    type GetCpuSetMasks =
        unsafe extern "system" fn(Handle, *mut GroupAffinity, u16, *mut u16) -> Bool;

    /// A kernel32 entry point that older Windows versions lack.
    fn kernel32_proc(name: &[u8]) -> *mut c_void {
        debug_assert_eq!(name.last(), Some(&0), "kernel32_proc: name must end in NUL");
        // SAFETY: both names are NUL-terminated.
        unsafe {
            let k32 = GetModuleHandleA(c"kernel32.dll".as_ptr().cast());
            if k32.is_null() {
                return std::ptr::null_mut();
            }
            GetProcAddress(k32, name.as_ptr())
        }
    }

    fn set_cpu_set_masks() -> Option<SetCpuSetMasks> {
        let p = kernel32_proc(b"SetThreadSelectedCpuSetMasks\0");
        // SAFETY: the documented signature of SetThreadSelectedCpuSetMasks.
        (!p.is_null()).then(|| unsafe { std::mem::transmute::<*mut c_void, SetCpuSetMasks>(p) })
    }

    fn get_cpu_set_masks() -> Option<GetCpuSetMasks> {
        let p = kernel32_proc(b"GetThreadSelectedCpuSetMasks\0");
        // SAFETY: the documented signature of GetThreadSelectedCpuSetMasks.
        (!p.is_null()).then(|| unsafe { std::mem::transmute::<*mut c_void, GetCpuSetMasks>(p) })
    }

    fn mask_cpus(group: u16, mask: usize, out: &mut Vec<usize>) {
        for j in 0..WIN_GROUP_SIZE {
            if mask >> j & 1 == 1 {
                out.push(group as usize * WIN_GROUP_SIZE + j);
            }
        }
    }

    pub fn hardware_concurrency() -> usize {
        // SAFETY: no pointers.
        unsafe { GetActiveProcessorCount(ALL_PROCESSOR_GROUPS) as usize }
    }

    /// What the two affinity interfaces say about the process. Either may be unknown
    /// (`None`), and the old one cannot always be read at all.
    struct Affinity {
        old_api: Option<Vec<usize>>,
        new_api: Option<Vec<usize>>,
        old_determinate: bool,
    }

    fn startup() -> &'static Affinity {
        static AFFINITY: OnceLock<Affinity> = OnceLock::new();
        AFFINITY.get_or_init(read_affinity)
    }

    /// Whether the process was likely restricted through the group interface, in which
    /// case the thread affinities have to be set through it too, or they would not
    /// override it.
    fn use_old_api() -> bool {
        let a = startup();
        a.old_api.is_some() || !a.old_determinate
    }

    fn process_groups() -> Option<Vec<u16>> {
        let mut count: u16 = 1;
        for _ in 0..2 {
            // The array must be 4-byte aligned.
            let mut buf = vec![0u32; count as usize / 2 + 1];
            let ptr = buf.as_mut_ptr().cast::<u16>();
            // SAFETY: `buf` holds at least `count` u16 values.
            let ok = unsafe { GetProcessGroupAffinity(GetCurrentProcess(), &mut count, ptr) };
            if ok != 0 {
                // SAFETY: the call wrote `count` values.
                return Some(unsafe { std::slice::from_raw_parts(ptr, count as usize) }.to_vec());
            }
            // SAFETY: no arguments.
            if unsafe { GetLastError() } != ERROR_INSUFFICIENT_BUFFER {
                break;
            }
        }
        None
    }

    fn read_affinity() -> Affinity {
        let mut a = Affinity {
            old_api: None,
            new_api: None,
            old_determinate: true,
        };

        if let Some(get) = get_cpu_set_masks() {
            let mut required: u16 = 0;
            // SAFETY: a null buffer of size zero asks for the required count.
            let ok = unsafe { get(GetCurrentThread(), std::ptr::null_mut(), 0, &mut required) };
            // SAFETY: no arguments.
            let insufficient = ok == 0 && unsafe { GetLastError() } == ERROR_INSUFFICIENT_BUFFER;
            if (ok != 0 || insufficient) && required > 0 {
                let mut masks = vec![GroupAffinity::default(); required as usize];
                // SAFETY: `masks` holds `required` entries.
                let ok = unsafe {
                    get(
                        GetCurrentThread(),
                        masks.as_mut_ptr(),
                        required,
                        &mut required,
                    )
                };
                if ok != 0 {
                    let mut cpus = Vec::new();
                    for m in &masks[..required as usize] {
                        mask_cpus(m.group, m.mask, &mut cpus);
                    }
                    cpus.sort_unstable();
                    a.new_api = Some(cpus);
                }
            }
        }

        let (mut proc_mask, mut sys_mask) = (0usize, 0usize);
        // SAFETY: two out-pointers to locals.
        let ok =
            unsafe { GetProcessAffinityMask(GetCurrentProcess(), &mut proc_mask, &mut sys_mask) };
        if ok == 0 || proc_mask == 0 {
            a.old_determinate = false;
            return a;
        }
        let Some(groups) = process_groups() else {
            a.old_determinate = false;
            return a;
        };

        if groups.len() == 1 {
            // SAFETY: no arguments.
            let several_groups = unsafe { GetActiveProcessorGroupCount() } != 1;
            if several_groups || proc_mask != sys_mask {
                let mut cpus = Vec::new();
                mask_cpus(groups[0], proc_mask, &mut cpus);
                a.old_api = Some(cpus);
            }
        } else if get_cpu_set_masks().is_some() {
            // Windows 11 answers for the calling thread's primary group only. Moving a
            // scratch thread over two processors of every group and intersecting what
            // it reads recovers the whole process affinity, without the probe's own
            // position leaking into the answer.
            let (cpus, full, determinate) = std::thread::scope(|s| {
                s.spawn(|| {
                    let mut cpus = Vec::new();
                    let mut full = true;
                    for &g in &groups {
                        // SAFETY: no pointers.
                        let active = unsafe { GetActiveProcessorCount(g) }.min(2);
                        let (mut pc, mut sc) = (usize::MAX, usize::MAX);
                        for i in 0..active {
                            let aff = GroupAffinity {
                                mask: 1 << i,
                                group: g,
                                reserved: [0; 3],
                            };
                            let (mut p2, mut s2) = (0usize, 0usize);
                            // SAFETY: pointers to locals.
                            unsafe {
                                if SetThreadGroupAffinity(
                                    GetCurrentThread(),
                                    &aff,
                                    std::ptr::null_mut(),
                                ) == 0
                                {
                                    return (cpus, true, false);
                                }
                                SwitchToThread();
                                if GetProcessAffinityMask(GetCurrentProcess(), &mut p2, &mut s2)
                                    == 0
                                {
                                    return (cpus, true, false);
                                }
                            }
                            pc &= p2;
                            sc &= s2;
                        }
                        if pc != sc {
                            full = false;
                        }
                        mask_cpus(g, pc, &mut cpus);
                    }
                    (cpus, full, true)
                })
                .join()
                .unwrap_or((Vec::new(), true, false))
            });
            if !determinate {
                a.old_determinate = false;
            } else if !full {
                a.old_api = Some(cpus);
            }
        }
        a
    }

    pub fn process_affinity() -> Option<Vec<usize>> {
        let a = startup();
        let mut out = match (&a.old_api, &a.new_api) {
            (None, None) => return None,
            (Some(o), None) => o.clone(),
            (None, Some(n)) => n.clone(),
            (Some(o), Some(n)) => o
                .iter()
                .filter(|c| n.binary_search(c).is_ok())
                .copied()
                .collect(),
        };
        out.sort_unstable();
        out.dedup();
        Some(out)
    }

    pub fn memory_nodes() -> Option<Vec<Vec<usize>>> {
        let mut nodes: Vec<Vec<usize>> = Vec::new();
        // SAFETY: no pointers.
        let groups = unsafe { GetActiveProcessorGroupCount() };
        for group in 0..groups {
            for number in 0..WIN_GROUP_SIZE as u8 {
                let pn = ProcessorNumber {
                    group,
                    number,
                    reserved: 0,
                };
                let mut node: u16 = 0;
                // SAFETY: pointers to locals.
                let ok = unsafe { GetNumaProcessorNodeEx(&pn, &mut node) };
                if ok != 0 && node != u16::MAX {
                    let n = node as usize;
                    if nodes.len() <= n {
                        nodes.resize(n + 1, Vec::new());
                    }
                    nodes[n].push(group as usize * WIN_GROUP_SIZE + number as usize);
                }
            }
        }
        (!nodes.is_empty()).then_some(nodes)
    }

    pub fn l3_domains() -> Option<Vec<Vec<usize>>> {
        let mut len: u32 = 0;
        // SAFETY: a null buffer asks for the required length.
        unsafe {
            GetLogicalProcessorInformationEx(RELATION_CACHE, std::ptr::null_mut(), &mut len);
            if GetLastError() != ERROR_INSUFFICIENT_BUFFER {
                return None;
            }
        }
        let mut buf = vec![0u64; (len as usize).div_ceil(8)];
        let base = buf.as_mut_ptr().cast::<u8>();
        // SAFETY: `buf` is at least `len` bytes.
        if unsafe { GetLogicalProcessorInformationEx(RELATION_CACHE, base, &mut len) } == 0 {
            return None;
        }
        // SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX holding a CACHE_RELATIONSHIP:
        // Relationship u32 @0, Size u32 @4, Level u8 @8, GroupCount u16 @38 (zero before
        // Windows 11, meaning one mask), GROUP_AFFINITY array @40.
        let read = |off: usize, n: usize| -> u64 {
            let mut v = [0u8; 8];
            // SAFETY: callers stay within the record, which lies within `buf`.
            unsafe { std::ptr::copy_nonoverlapping(base.add(off), v.as_mut_ptr(), n) };
            u64::from_le_bytes(v)
        };
        let mut domains = Vec::new();
        let mut off = 0usize;
        while off + 8 <= len as usize {
            let relation = read(off, 4) as i32;
            let size = read(off + 4, 4) as usize;
            if size == 0 || off + size > len as usize {
                break;
            }
            if relation == RELATION_CACHE && read(off + 8, 1) == 3 && size >= 40 + 16 {
                let groups = (read(off + 38, 2) as usize).max(1);
                let mut cpus = Vec::new();
                for g in 0..groups {
                    let at = off + 40 + g * 16;
                    if at + 16 > off + size {
                        break;
                    }
                    mask_cpus(read(at + 8, 2) as u16, read(at, 8) as usize, &mut cpus);
                }
                cpus.sort_unstable();
                if !cpus.is_empty() {
                    domains.push(cpus);
                }
            }
            off += size;
        }
        (!domains.is_empty()).then_some(domains)
    }

    pub fn bind_current_thread(cpus: &[usize], highest: usize) {
        let groups = highest / WIN_GROUP_SIZE + 1;
        let set = set_cpu_set_masks();
        // SAFETY: the current-thread pseudo handle.
        let thread = unsafe { GetCurrentThread() };

        if let Some(set) = set {
            let mut masks: Vec<GroupAffinity> = (0..groups)
                .map(|g| GroupAffinity {
                    mask: 0,
                    group: g as u16,
                    reserved: [0; 3],
                })
                .collect();
            for &c in cpus {
                masks[c / WIN_GROUP_SIZE].mask |= 1 << (c % WIN_GROUP_SIZE);
            }
            // SAFETY: `masks` holds `groups` entries.
            if unsafe { set(thread, masks.as_ptr(), groups as u16) } == 0 {
                outerr!("info string NUMA: could not pin a thread (CPU sets)");
            }
            // SAFETY: no arguments.
            unsafe { SwitchToThread() };
        }

        if set.is_none() || use_old_api() {
            // One group per thread: take the group of the node's first processor.
            let group = cpus[0] / WIN_GROUP_SIZE;
            let mut aff = GroupAffinity {
                mask: 0,
                group: group as u16,
                reserved: [0; 3],
            };
            for &c in cpus.iter().filter(|&&c| c / WIN_GROUP_SIZE == group) {
                aff.mask |= 1 << (c % WIN_GROUP_SIZE);
            }
            // SAFETY: a pointer to a local.
            unsafe {
                if SetThreadGroupAffinity(thread, &aff, std::ptr::null_mut()) == 0 {
                    outerr!("info string NUMA: could not pin a thread (group affinity)");
                }
                SwitchToThread();
            }
        }
    }
}

#[cfg(not(any(target_os = "linux", all(windows, target_pointer_width = "64"))))]
mod sys {
    //! Everywhere else: one node, and pinning does nothing.

    pub fn hardware_concurrency() -> usize {
        1
    }
    pub fn process_affinity() -> Option<Vec<usize>> {
        None
    }
    pub fn memory_nodes() -> Option<Vec<Vec<usize>>> {
        None
    }
    pub fn l3_domains() -> Option<Vec<Vec<usize>>> {
        None
    }
    pub fn bind_current_thread(_cpus: &[usize], _highest: usize) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    fn range(a: usize, b: usize) -> Vec<usize> {
        (a..b).collect()
    }

    #[test]
    fn cpulists_parse_like_the_kernel_writes_them() {
        assert_eq!(
            parse_cpulist("0-3,8,10-11\n"),
            Some(vec![0, 1, 2, 3, 8, 10, 11])
        );
        assert_eq!(parse_cpulist(""), Some(vec![]));
        assert_eq!(parse_cpulist("\n"), Some(vec![]));
        assert_eq!(parse_cpulist("5"), Some(vec![5]));
        assert_eq!(parse_cpulist("3-1"), None);
        assert_eq!(parse_cpulist("a"), None);
        assert_eq!(parse_cpulist("1500-1501"), Some(vec![1500, 1501]));
    }

    #[test]
    fn user_strings_round_trip() {
        let cfg = NumaConfig::from_user_string("0-15,128-143:16-31").unwrap();
        assert!(cfg.is_custom());
        assert_eq!(cfg.node_count(), 2);
        assert_eq!(cfg.cpus(0).len(), 32);
        assert_eq!(cfg.to_string(), "0-15,128-143:16-31");
        assert!(
            NumaConfig::from_user_string("0-3:3-5").is_none(),
            "duplicate processor"
        );
        assert!(NumaConfig::from_user_string(":").is_none());
        assert_eq!(
            NumaConfig::from_user_string("0-3::4-7")
                .unwrap()
                .node_count(),
            2
        );
    }

    #[test]
    fn empty_nodes_are_dropped() {
        let cfg = NumaConfig::from_nodes(vec![range(0, 8), vec![], range(8, 16)]);
        assert_eq!(cfg.node_count(), 2);
        assert_eq!(cfg.system_node_of(1), 1);
    }

    #[test]
    fn l3_domains_bundle_within_a_memory_node() {
        // Two sockets of 128 cores with SMT, eight cache domains of 16 cores (32
        // processors) per socket: each domain is already a full bundle.
        let mut domains = Vec::new();
        for socket in 0..2 {
            for d in 0..8 {
                let first = socket * 128 + d * 16;
                let mut cpus = range(first, first + 16);
                cpus.extend(range(256 + first, 256 + first + 16));
                domains.push((socket, cpus));
            }
        }
        let cfg = NumaConfig::from_l3_domains(domains, L3_BUNDLE_SIZE);
        assert_eq!(cfg.node_count(), 16);
        assert_eq!(cfg.cpu_count(), 512);
        assert!(cfg.nodes().iter().all(|n| n.len() == 32));
        assert_eq!(cfg.system_nodes(), vec![0, 1]);
        // No bundle crosses a socket.
        for n in 0..cfg.node_count() {
            let sys = cfg.system_node_of(n);
            assert!(cfg.cpus(n).iter().all(|&c| (c % 256) / 128 == sys));
        }
    }

    #[test]
    fn small_cache_domains_merge_pairwise() {
        // Eight domains of 8: the first pass merges pairs (16), the second pairs of
        // pairs (32), and then nothing fits any more.
        let domains: Vec<_> = (0..8).map(|d| (0, range(d * 8, d * 8 + 8))).collect();
        let cfg = NumaConfig::from_l3_domains(domains, 32);
        assert_eq!(cfg.node_count(), 2);
        assert_eq!(cfg.cpus(0), &range(0, 32)[..]);
        // A desktop part with two cache domains of 12 stays one node.
        let desk = NumaConfig::from_l3_domains(vec![(0, range(0, 12)), (0, range(12, 24))], 32);
        assert_eq!(desk.node_count(), 1);
    }

    #[test]
    fn nodes_split_at_processor_groups() {
        let cfg = NumaConfig::from_nodes(vec![range(0, 96), range(96, 192)]);
        let split = cfg.split_at_groups(64);
        assert_eq!(split.node_count(), 4);
        assert_eq!(split.cpus(0).len(), 64);
        assert_eq!(split.cpus(1).len(), 32);
        assert_eq!(split.cpus(2).len(), 32);
        assert_eq!(split.system_node_of(1), 0);
        assert_eq!(split.system_node_of(2), 1);
    }

    #[test]
    fn binding_is_advised_only_when_it_pays() {
        let two = NumaConfig::from_nodes(vec![range(0, 64), range(64, 128)]);
        assert!(!two.suggests_binding(1));
        assert!(
            !two.suggests_binding(7),
            "fits in half a node, under four per node"
        );
        assert!(two.suggests_binding(8), "four per node");
        assert!(two.suggests_binding(33));
        assert!(two.suggests_binding(128));
        let one = NumaConfig::single(64);
        assert!(!one.suggests_binding(64));
        // Eight nodes of 64: 32 threads give every node four.
        let eight = NumaConfig::from_nodes((0..8).map(|n| range(n * 64, n * 64 + 64)).collect());
        assert!(!eight.suggests_binding(31));
        assert!(eight.suggests_binding(32));
        // A small node does not count towards the four per node.
        let lopsided = NumaConfig::from_nodes(vec![range(0, 64), range(64, 72)]);
        assert!(lopsided.suggests_binding(4));
        assert!(
            NumaConfig::from_user_string("0")
                .unwrap()
                .suggests_binding(1)
        );
    }

    #[test]
    fn threads_fill_nodes_in_proportion() {
        let cfg = NumaConfig::from_nodes(vec![range(0, 64), range(64, 96)]);
        let d = cfg.distribute(96);
        assert_eq!(d.iter().filter(|&&n| n == 0).count(), 64);
        assert_eq!(d.iter().filter(|&&n| n == 1).count(), 32);
        assert_eq!(&d[..3], &[0, 0, 1]);
        assert_eq!(NumaConfig::single(4).distribute(3), vec![0, 0, 0]);
        let even = NumaConfig::from_nodes(vec![range(0, 8), range(8, 16)]).distribute(1024);
        assert_eq!(even.iter().filter(|&&n| n == 1).count(), 512);
    }

    #[test]
    fn processors_beyond_1024_are_kept() {
        let cfg = NumaConfig::from_user_string("0-1023:1024-2047").unwrap();
        assert_eq!(cfg.highest_cpu(), 2047);
        assert_eq!(cfg.node_of_cpu(1500), Some(1));
    }

    #[test]
    fn hardware_nodes_name_custom_nodes() {
        let hw = NumaConfig::from_nodes(vec![range(0, 12), range(12, 24)]);
        let custom = NumaConfig::from_user_string("0-5:6-11:12-23")
            .unwrap()
            .with_system_nodes_from(&hw);
        assert_eq!(custom.system_nodes(), vec![0, 1]);
        assert_eq!(custom.system_node_of(1), 0);
        assert_eq!(custom.system_node_of(2), 1);
    }

    #[test]
    fn policies_parse() {
        assert_eq!(NumaPolicy::parse("auto"), Some(NumaPolicy::Auto));
        assert_eq!(NumaPolicy::parse("NONE"), Some(NumaPolicy::None));
        assert_eq!(NumaPolicy::parse("system"), Some(NumaPolicy::System));
        assert_eq!(NumaPolicy::parse("hardware"), Some(NumaPolicy::Hardware));
        assert!(matches!(
            NumaPolicy::parse("0-3:4-7"),
            Some(NumaPolicy::Custom(_))
        ));
        assert_eq!(NumaPolicy::parse("bogus"), None);
    }

    #[test]
    fn this_machine_reads_without_losing_processors() {
        let cfg = NumaPolicy::Auto.config();
        assert!(cfg.node_count() >= 1);
        assert!(cfg.nodes().iter().all(|n| !n.is_empty()));
        let hw = NumaPolicy::Hardware.config();
        assert!(hw.cpu_count() >= cfg.cpu_count());
    }

    #[test]
    fn a_thread_runs_on_its_node() {
        let cfg = NumaPolicy::Auto.config();
        let got = run_on_node(&cfg, 0, || 41 + 1);
        assert_eq!(got, 42);
    }
}
