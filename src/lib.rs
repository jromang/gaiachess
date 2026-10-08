//! GaiaChess — a UCI chess engine targeting AMD Zen 4 (AVX2/AVX-512).
//!
//! Everything the engine is lives here, so that more than one front end can be built on
//! top of it: the command-line binary, and a WebAssembly build whose interface and search
//! run as two separate modules and cannot therefore share a single `main`.

// Declared first, and with `macro_use`: `out!` and `outerr!` have to be in scope for
// every module below.
#[macro_use]
pub mod out;

pub mod bitboard;
pub mod book;
#[cfg(target_arch = "x86_64")]
pub mod cpu;
mod cpuload;
pub mod cuckoo;
pub mod eval;
pub mod history;
pub mod largemem;
pub mod lmr_factorized;
pub mod movegen;
pub mod movepick;
pub mod nnue;
pub mod numa;
pub mod perft;
pub mod position;
pub mod progress;
pub mod search;
pub mod see;
pub mod simd_attacks;
pub mod skill;
pub mod stats;
pub mod time;
pub mod timeman;
pub mod tt;
pub mod tune;
pub mod types;
pub mod wdl;
pub mod zobrist;
// The module is always compiled, because it is where the recording macros live and
// they have to expand to nothing in an ordinary build. Its wire protocol is then
// unused, which is the point: the table is a contract with the Python parser and stays
// complete whether or not this build records anything.
pub mod bench;
pub mod bench_stats;
#[cfg(feature = "datagen")]
pub mod datagen;
#[cfg(feature = "gaiatb")]
pub mod dtm;
#[cfg(any(feature = "stats", feature = "tree"))]
pub mod dump;
#[cfg(feature = "gui-core")]
pub mod gui;
#[cfg(feature = "nalimov")]
pub mod nalimov;
#[cfg(feature = "online-tb")]
pub mod online_tb;
#[cfg(feature = "datagen")]
pub mod relabel;
pub mod shm;
pub mod tb;
pub mod threads;
#[cfg_attr(not(feature = "tree"), allow(dead_code))]
pub mod tree;
pub mod uci;
#[cfg(all(windows, feature = "gui"))]
pub mod win_console;

/// Selects the runtime SIMD and slider implementations. Must run before any search.
///
/// On x86-64 this resolves the runtime dispatch (NNUE kernel tier, attack kind,
/// weight permutation) from CPUID once and builds whatever tables the elected
/// paths need. On aarch64 it elects NEON vs NEON+DOTPROD from HWCAP. Other
/// architectures have a single compile-time backend.
pub fn init_cpu_dispatch() {
    #[cfg(target_arch = "x86_64")]
    cpu::init();
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    nnue::kernels::arm::init();

    // Table data for the AVX2 per-piece attack path, the compile-time floor of
    // AVX2-baseline builds (the runtime election only ever moves PEXT on top).
    #[cfg(target_feature = "avx2")]
    simd_attacks::init();
}
