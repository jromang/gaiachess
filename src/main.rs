//! GaiaChess — a UCI chess engine targeting AMD Zen 4 (AVX2/AVX-512).

use bpaf::Bpaf;

mod types;
mod bitboard;
mod zobrist;
mod position;
mod movegen;
mod perft;
mod simd_attacks;
mod eval;
mod tt;
mod movepick;
mod timeman;
mod history;
mod nnue;
mod see;
mod tune;
mod cuckoo;
mod search;
mod threads;
mod bench;
mod bench_stats;
#[cfg(feature = "datagen")]
mod datagen;
mod tb;
#[cfg(feature = "gaiatb")]
mod dtm;
#[cfg(feature = "nalimov")]
mod nalimov;
#[cfg(feature = "online-tb")]
mod online_tb;
mod uci;

#[derive(Debug, Clone, Bpaf)]
#[bpaf(options("gaiachess"), version, footer(
    "Examples:\n  gaiachess                          Start UCI protocol\n  gaiachess bench                    Single run, deterministic node count\n  gaiachess bench --stats             5 runs with median, CI, outliers\n  gaiachess bench --stats -n 10 -v   10 runs with per-position breakdown\n  gaiachess perft -d 7               Perft divide depth 7\n  gaiachess info                     Show build configuration"
))]
enum Cmd {
    /// Run perft divide
    #[bpaf(command)]
    Perft {
        /// Search depth
        #[bpaf(short, long, fallback(5u32), display_fallback)]
        depth: u32,
        /// FEN position (use quotes for multi-word FEN)
        #[bpaf(positional("FEN"), optional)]
        fen: Option<String>,
    },

    /// Run benchmark suite (27 positions, default depth 16)
    #[bpaf(command)]
    Bench {
        /// Search depth (default: 16)
        #[bpaf(short, long)]
        depth: Option<i32>,
        /// Statistical mode: N runs with median, CI, outlier detection
        #[bpaf(short, long)]
        stats: bool,
        /// Number of measured runs in stats mode (min 3, +1 warmup)
        #[bpaf(short('n'), long, fallback(5u32), display_fallback)]
        runs: u32,
        /// Show per-position NPS and CV% breakdown
        #[bpaf(short, long)]
        verbose: bool,
    },

    /// Generate NNUE training data
    #[cfg(feature = "datagen")]
    #[bpaf(command)]
    Datagen {
        /// Worker threads (0 = all cores)
        #[bpaf(short('t'), long, fallback(0usize), display_fallback)]
        threads: usize,
        /// Target positions to generate
        #[bpaf(short('n'), long, fallback(1_000_000u64), display_fallback)]
        positions: u64,
        /// Search depth per move
        #[bpaf(short, long, fallback(8i32), display_fallback)]
        depth: i32,
        /// Output base filename
        #[bpaf(short, long, fallback("data/gen0".to_string()), display_fallback)]
        output: String,
        /// Opening book path (EPD format)
        #[bpaf(short, long)]
        book: Option<String>,
        /// Syzygy tablebase path (enables TB adjudication)
        #[bpaf(short('s'), long)]
        syzygy: Option<String>,
    },

    /// Dump threat feature indices for a FEN (parity testing with the trainer)
    #[bpaf(command)]
    Threats {
        /// FEN position (use quotes for multi-word FEN)
        #[bpaf(positional("FEN"))]
        fen: String,
    },

    /// Display build configuration
    #[bpaf(command)]
    Info,

    /// Dump SPSA tunable parameters
    #[bpaf(command)]
    Spsa {
        /// Output as JSON instead of CSV
        #[bpaf(long)]
        json: bool,
    },

    /// Start UCI protocol (default when no subcommand is given)
    #[bpaf(hide)]
    Uci(#[bpaf(external(uci_mode))] UciMode),
}

#[derive(Debug, Clone)]
struct UciMode;

fn uci_mode() -> impl bpaf::Parser<UciMode> {
    bpaf::pure(UciMode)
}

fn main() {
    // Run on a thread with a 32 MB stack to handle deep search recursion.
    // PGO inlining enlarges stack frames significantly.
    let builder = std::thread::Builder::new().stack_size(32 * 1024 * 1024);
    let handler = builder.spawn(real_main).expect("failed to spawn main thread");
    handler.join().unwrap();
}

fn real_main() {
    #[cfg(target_feature = "avx2")]
    simd_attacks::init();

    #[cfg(target_feature = "bmi2")]
    bitboard::init_pext();

    let cmd = cmd().run();

    match cmd {
        Cmd::Uci(_) => uci::uci_loop(),
        Cmd::Perft { depth, fen } => {
            let fen = fen.unwrap_or_else(|| {
                String::from("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            });
            let mut pos = match position::Position::from_fen(&fen) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("Invalid FEN: {e}");
                    std::process::exit(1);
                }
            };
            perft::divide(&mut pos, depth);
        }
        Cmd::Bench { depth, stats, runs, verbose } => {
            if stats {
                bench::run_stats(depth, runs, verbose);
            } else {
                bench::run(depth);
            }
        }
        #[cfg(feature = "datagen")]
        Cmd::Datagen { threads, positions, depth, output, book, syzygy } => {
            #[cfg(feature = "syzygy")]
            if let Some(ref path) = syzygy {
                if !tb::init(path) {
                    eprintln!("WARNING: Failed to load Syzygy tablebases from {path}");
                }
            }
            #[cfg(not(feature = "syzygy"))]
            if syzygy.is_some() {
                eprintln!("WARNING: --syzygy requires --features syzygy (ignored)");
            }
            datagen::run(threads, positions, depth, &output, book.as_deref());
        }
        Cmd::Threats { fen } => {
            let pos = position::Position::from_fen(&fen).expect("invalid FEN");
            nnue::threats::dump_features(&pos);
        }
        Cmd::Info => {
            println!("GaiaChess {}", env!("CARGO_PKG_VERSION"));
            println!();
            println!(
                "Build:       {}",
                if cfg!(debug_assertions) { "debug" } else { "release" }
            );
            println!(
                "Compiler:    {}",
                option_env!("RUSTC_VERSION").unwrap_or("unknown")
            );
            println!(
                "Commit:      {}",
                option_env!("GIT_HASH").unwrap_or("unknown")
            );
            println!(
                "Date:        {}",
                option_env!("BUILD_DATE").unwrap_or("unknown")
            );
            let eval = if cfg!(feature = "nnue") { "NNUE" } else { "PeSTO" };
            println!("Eval:        {eval}");
            if cfg!(feature = "nnue") {
                println!(
                    "Network:     {}",
                    option_env!("MODEL").unwrap_or("unknown")
                );
            }
            let simd = if cfg!(target_feature = "avx512f") {
                "AVX-512"
            } else if cfg!(target_feature = "avx2") {
                "AVX2"
            } else if cfg!(all(target_arch = "aarch64", target_feature = "neon")) {
                "NEON"
            } else {
                "scalar"
            };
            println!("SIMD:        {simd}");
            let slider = if cfg!(target_feature = "bmi2") {
                "PEXT"
            } else if cfg!(target_feature = "avx2") {
                "AVX2 BLSMSK"
            } else {
                "magic"
            };
            let setwise = if cfg!(target_feature = "avx512f") {
                "AVX-512 Kogge-Stone"
            } else {
                "scalar"
            };
            println!("Movegen:     {slider}, setwise {setwise}");
            let mut features = Vec::new();
            if cfg!(feature = "nnue") {
                features.push("NNUE");
            }
            if cfg!(feature = "syzygy") {
                features.push("Syzygy");
            }
            if cfg!(feature = "gaiatb") {
                features.push("GaiaTB");
            }
            if cfg!(feature = "online-tb") {
                features.push("OnlineTB");
            }
            if cfg!(feature = "spsa") {
                features.push("SPSA");
            }
            if features.is_empty() {
                features.push("none");
            }
            println!("Features:    {}", features.join(", "));
            println!(
                "OS:          {}/{}",
                std::env::consts::OS,
                std::env::consts::ARCH
            );
        }
        Cmd::Spsa { json } => {
            #[cfg(feature = "spsa")]
            {
                if json {
                    println!("{}", tune::emit_json());
                } else {
                    println!("{}", tune::emit_csv());
                }
            }
            #[cfg(not(feature = "spsa"))]
            {
                let _ = json;
                eprintln!("SPSA tuning not enabled. Rebuild with: cargo build --features spsa");
                std::process::exit(1);
            }
        }
    }
}
