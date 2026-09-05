//! GaiaChess — a UCI chess engine targeting AMD Zen 4 (AVX2/AVX-512).
//!
//! Everything the engine is lives here, so that more than one front end can be built on
//! top of it: the command-line binary, and a WebAssembly build whose interface and search
//! run as two separate modules and cannot therefore share a single `main`.

// Declared first, and with `macro_use`: `out!` and `outerr!` have to be in scope for
// every module below.
#[macro_use]
pub mod out;

pub mod time;
pub mod types;
pub mod bitboard;
#[cfg(target_arch = "x86_64")]
pub mod cpu;
pub mod zobrist;
pub mod position;
pub mod movegen;
pub mod perft;
pub mod progress;
pub mod simd_attacks;
pub mod eval;
pub mod tt;
pub mod movepick;
pub mod timeman;
pub mod history;
pub mod nnue;
pub mod see;
pub mod skill;
pub mod book;
pub mod tune;
pub mod cuckoo;
pub mod search;
pub mod stats;
// The module is always compiled, because it is where the recording macros live and
// they have to expand to nothing in an ordinary build. Its wire protocol is then
// unused, which is the point: the table is a contract with the Python parser and stays
// complete whether or not this build records anything.
#[cfg_attr(not(feature = "tree"), allow(dead_code))]
pub mod tree;
#[cfg(any(feature = "stats", feature = "tree"))]
pub mod dump;
pub mod threads;
pub mod shm;
pub mod bench;
pub mod bench_stats;
#[cfg(feature = "datagen")]
pub mod datagen;
pub mod tb;
#[cfg(feature = "gaiatb")]
pub mod dtm;
#[cfg(feature = "nalimov")]
pub mod nalimov;
#[cfg(feature = "online-tb")]
pub mod online_tb;
#[cfg(feature = "gui-core")]
pub mod gui;
#[cfg(all(windows, feature = "gui"))]
pub mod win_console;
pub mod uci;

/// Selects the runtime SIMD and slider implementations. Must run before any search.
///
/// On x86-64 this resolves the runtime dispatch (NNUE kernel tier, attack kind,
/// weight permutation) from CPUID once and builds whatever tables the elected
/// paths need. Other architectures have a single compile-time backend.
pub fn init_cpu_dispatch() {
    #[cfg(target_arch = "x86_64")]
    cpu::init();

    // Table data for the AVX2 per-piece attack path, the compile-time floor of
    // AVX2-baseline builds (the runtime election only ever moves PEXT on top).
    #[cfg(target_feature = "avx2")]
    simd_attacks::init();
}
