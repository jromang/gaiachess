//! GaiaChess — a UCI chess engine targeting AMD Zen 4 (AVX2/AVX-512).

// With the interface compiled in, the Windows build is linked as a GUI-subsystem
// program: no console is created behind the board, and none flashes past on the way to
// it. The engine is handed the launcher's console back in `win_console`, which must
// therefore run before a single byte is printed.
#![cfg_attr(all(windows, feature = "gui"), windows_subsystem = "windows")]

use bpaf::{Bpaf, Parser};
#[cfg(feature = "gui")]
use std::time::Duration;

// Every module lives in the library, so that the WebAssembly build can put the
// interface and the search in two separate modules built from the same source.
use gaiachess::*;

#[derive(Debug, Clone, Bpaf)]
#[bpaf(options("gaiachess"), version, footer(
    "Examples:\n  gaiachess                          Play, or speak UCI if something asks\n  gaiachess --no-gui                 Speak UCI at once, never open the interface\n  gaiachess bench                    Single run, deterministic node count\n  gaiachess bench --stats             5 runs with median, CI, outliers\n  gaiachess bench --stats -n 10 -v   10 runs with per-position breakdown\n  gaiachess perft -d 7               Perft divide depth 7\n  gaiachess info                     Show build configuration"
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

    /// Deterministic single-thread search with stats/tree export (for treescope)
    #[cfg(any(feature = "stats", feature = "tree"))]
    #[bpaf(command)]
    Dump {
        /// Search depth
        #[bpaf(short, long, fallback(14i32), display_fallback)]
        depth: i32,
        /// Node cap (0 = none; takes precedence over --depth)
        #[bpaf(short, long, fallback(0u64), display_fallback)]
        nodes: u64,
        /// Hash size in MB
        #[bpaf(long, fallback(64usize), display_fallback)]
        tt_mb: usize,
        /// Write stats JSON to this path ("-" = stdout)
        #[bpaf(short, long, fallback(String::new()))]
        stats: String,
        /// Write GTREE binary tree dump to this path (requires --features tree)
        #[bpaf(short, long, fallback(String::new()))]
        tree: String,
        /// Skip recording subtrees with remaining depth below this value
        #[bpaf(long, fallback(0i32), display_fallback)]
        min_record_depth: i32,
        /// Record per-move records inside quiescence nodes
        #[bpaf(long)]
        qs_moves: bool,
        /// Tree buffer cap in MB (recording stops beyond this, search continues)
        #[bpaf(long, fallback(512usize), display_fallback)]
        max_mb: usize,
        /// UCI moves to apply after the FEN
        #[bpaf(short, long, fallback(String::new()))]
        moves: String,
        /// FEN position or "startpos" (use quotes for multi-word FEN)
        #[bpaf(positional("FEN"))]
        fen: String,
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

    /// Start the pixel-art graphical interface
    #[cfg(feature = "gui")]
    #[bpaf(command)]
    Gui {
        /// Render one frame to a PNG without opening a window (development aid)
        #[bpaf(long, argument("PATH"))]
        shot: Option<String>,
        /// Screen to capture: menu, about, game or gamemenu
        #[bpaf(long, argument("NAME"), fallback("game".to_string()))]
        shot_scene: String,
        /// Moves to play before a headless capture, in UCI notation
        #[bpaf(long, argument("MOVES"), fallback(String::new()))]
        shot_moves: String,
        /// Squares to click before a headless capture, e.g. "e2 e4"
        #[bpaf(long, argument("SQUARES"), fallback(String::new()))]
        shot_clicks: String,
        /// Square to leave the mouse pointing at
        #[bpaf(long, argument("SQUARE"), fallback(String::new()))]
        shot_hover: String,
        /// Drag a piece and keep hold of it, e.g. "e2 e4"
        #[bpaf(long, argument("FROM TO"), fallback(String::new()))]
        shot_drag: String,
        /// Drag a piece and let go of it, e.g. "e2 e5"
        #[bpaf(long, argument("FROM TO"), fallback(String::new()))]
        shot_drop: String,
        /// A move to catch mid-flight in a headless capture
        #[bpaf(long, argument("MOVE"), fallback(String::new()))]
        shot_fly: String,
        /// Turn the board round at the end of a headless capture, as pressing F does
        #[bpaf(long)]
        shot_flip: bool,
        /// Engine moves to play against itself before a headless capture
        #[bpaf(long, argument("N"), fallback(0u32))]
        shot_selfplay: u32,
        /// Level the engine plays at, 1 to 20 (default: the one the menu opens on)
        #[bpaf(long, argument("N"), fallback(gaiachess::gui::DEFAULT_LEVEL))]
        level: u8,
        /// Logic steps to settle before a headless capture
        #[bpaf(long, argument("N"), fallback(30u32))]
        shot_ticks: u32,
        /// Colour scheme index for a headless capture
        #[bpaf(long, argument("N"), fallback(0usize))]
        shot_scheme: usize,
        /// Clock choice index for a headless capture
        #[bpaf(long, argument("N"), fallback(4usize))]
        shot_clock: usize,
        /// Language to draw a headless capture in: en, fr, es, de, it or pt
        #[bpaf(long, argument("TAG"), fallback("en".to_string()))]
        shot_lang: String,
    },

    /// Dump SPSA tunable parameters
    #[bpaf(command)]
    Spsa {
        /// Output as JSON instead of CSV
        #[bpaf(long)]
        json: bool,
    },

    /// Start UCI protocol (default when no subcommand is given)
    #[bpaf(hide)]
    // Only the interface reads what is inside; without it, the mode is still parsed so
    // that the same command line works, and then ignored.
    #[cfg_attr(not(feature = "gui"), allow(dead_code))]
    Uci(#[bpaf(external(uci_mode))] UciMode),
}

#[derive(Debug, Clone)]
struct UciMode {
    /// Speak UCI at once, with no detection window and no interface.
    #[cfg_attr(not(feature = "gui"), allow(dead_code))]
    no_gui: bool,
}

/// Parses the bare invocation. `switch` falls back to false and consumes nothing when
/// absent, so this stays the catch-all that a plain `gaiachess` lands on — and the flag
/// is accepted, and ignored, in a build without the interface, so one command line
/// works on both.
fn uci_mode() -> impl bpaf::Parser<UciMode> {
    bpaf::long("no-gui")
        .help("Speak UCI at once, without waiting to see whether to open the interface")
        .switch()
        .map(|no_gui| UciMode { no_gui })
}

/// How long a bare launch listens before deciding nobody is speaking the protocol.
///
/// Far more than a chess interface needs to say `uci` — it is the first thing they send
/// — and the wait is only ever paid in full by a player, who then gets the board.
#[cfg(feature = "gui")]
const PROBE_WINDOW: Duration = Duration::from_secs(2);

/// Whether a window can be opened at all.
///
/// Checked before the interface is started, never after: miniquad does not report this
/// failure, it panics, and a release build aborts on panic. The engine spends most of
/// its life under match managers on machines reached over SSH, where the difference
/// between "no window" and "no engine" is the whole match.
#[cfg(all(feature = "gui", unix, not(any(target_os = "macos", target_os = "haiku"))))]
fn display_available() -> bool {
    // An empty DISPLAY is how a script says "no display", so emptiness counts as absence.
    let set = |v| std::env::var_os(v).is_some_and(|d: std::ffi::OsString| !d.is_empty());
    set("DISPLAY") || set("WAYLAND_DISPLAY")
}

// Haiku sets no DISPLAY; the shim asks app_server itself, which is the only answer
// that is true over SSH as well as at the desk.
#[cfg(all(feature = "gui", target_os = "haiku"))]
fn display_available() -> bool {
    gui::haiku::display_available()
}

#[cfg(all(feature = "gui", any(windows, target_os = "macos")))]
fn display_available() -> bool {
    true
}

/// What the interface probe already took off stdin, if it ran at all.
#[derive(Default)]
struct ProbedStdin {
    /// `None` when no probe ran: the UCI loop then opens stdin itself, as it always has.
    rx: Option<std::sync::mpsc::Receiver<String>>,
    /// A line read during the probe, to be handled before any that follow it.
    first: Option<String>,
}

/// Runs `f` on a thread with a 32 MB stack, for the depth of the search recursion —
/// PGO inlining pushes frames well past what the default stack holds.
#[cfg(not(target_arch = "wasm32"))]
fn on_deep_stack(f: impl FnOnce() + Send + 'static) {
    let builder = std::thread::Builder::new().stack_size(32 * 1024 * 1024);
    let handler = builder.spawn(f).expect("failed to spawn main thread");
    handler.join().unwrap();
}

/// WebAssembly has no threads to move the work onto, so the stack has to be asked for at
/// link time instead: `-C link-arg=-zstack-size=…`. This build exists to run `bench`
/// under a wasm runtime, which is how the vector backends are checked against each other.
#[cfg(target_arch = "wasm32")]
fn on_deep_stack(f: impl FnOnce() + Send + 'static) {
    f();
}

fn main() {
    // Before the command line is even parsed: a GUI-subsystem program starts out with no
    // console, and `--help` and `--version` are printed by the parser itself.
    #[cfg(all(windows, feature = "gui"))]
    win_console::attach_parent_console();

    let cmd = cmd().run();

    // The graphical interface must own the real main thread: platform event loops
    // refuse to run anywhere else. It is dispatched before the deep-stack worker
    // below is spawned, and gives the engine a thread of its own instead.
    #[cfg(feature = "gui")]
    if let Cmd::Gui {
        shot, shot_scene, shot_moves, shot_clicks, shot_hover, shot_drag, shot_drop, shot_fly,
        shot_flip, shot_selfplay, level, shot_ticks, shot_scheme, shot_clock, shot_lang,
    } = &cmd
    {
        init_cpu_dispatch();
        gui::run(shot.as_deref().map(|path| gui::Shot {
            path,
            scene: shot_scene,
            moves: shot_moves,
            clicks: shot_clicks,
            hover: shot_hover,
            drag: shot_drag,
            drop: shot_drop,
            fly: shot_fly,
            flip: *shot_flip,
            selfplay: *shot_selfplay,
            level: *level,
            ticks: *shot_ticks,
            scheme: *shot_scheme,
            clock: *shot_clock,
            lang: shot_lang,
        }));
        return;
    }

    #[allow(unused_mut)]
    let mut stdin = ProbedStdin::default();

    // A bare launch is ambiguous: either something is about to drive the engine over
    // stdin, or a player has started the game. Listen briefly and let the silence
    // decide.
    #[cfg(feature = "gui")]
    if let Cmd::Uci(mode) = &cmd
        && !mode.no_gui
        && display_available()
    {
        // No input channel of any kind — started from the desktop, with neither a console
        // nor a launcher behind us. Nothing will ever speak the protocol on this one, so
        // listening for it would do nothing but keep a player waiting.
        #[cfg(windows)]
        if !win_console::stdin_is_readable() {
            init_cpu_dispatch();
            gui::run(None);
            return;
        }

        // A pipe settles the question the probe exists to ask: only a program — a
        // chess GUI, a match manager — ever connects one, and some (Fritz 17 among
        // them) take longer than any reasonable window to say their first word.
        // Commit to the protocol and wait indefinitely, as an engine always has;
        // only a console stdin leaves real ambiguity for the probe to resolve.
        #[cfg(windows)]
        let probe = !win_console::stdin_is_pipe();
        #[cfg(not(windows))]
        let probe = true;

        if probe {
            let rx = uci::spawn_stdin_reader();
            match rx.recv_timeout(PROBE_WINDOW) {
                // Someone is talking to us. The line is handed on, not dropped.
                Ok(line) => {
                    stdin.rx = Some(rx);
                    stdin.first = Some(line);
                }
                // Nobody said anything — and a stdin that ended before a word was said
                // can never say one. Both are the same silence, so both get the board;
                // EOF just answers without the wait. This is how a desktop launch looks
                // on Haiku and Linux: Tracker and the freedesktop launchers hand the
                // program a stdin already at EOF, where taking the engine path meant
                // exiting at once and a double-click that seemed to do nothing.
                Err(std::sync::mpsc::RecvTimeoutError::Timeout)
                | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    // Nothing will drain this channel again; dropping it lets the reader
                    // fall out of its loop instead of piling lines up behind it.
                    drop(rx);
                    init_cpu_dispatch();
                    gui::run(None);
                    return;
                }
            }
        }
    }

    on_deep_stack(move || real_main(cmd, stdin));
}

fn real_main(cmd: Cmd, stdin: ProbedStdin) {
    init_cpu_dispatch();

    match cmd {
        Cmd::Uci(_) => uci::uci_loop(
            stdin.rx.unwrap_or_else(uci::spawn_stdin_reader),
            stdin.first,
        ),
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
            if let Some(ref path) = syzygy
                && !tb::init(path)
            {
                eprintln!("WARNING: Failed to load Syzygy tablebases from {path}");
            }
            #[cfg(not(feature = "syzygy"))]
            if syzygy.is_some() {
                eprintln!("WARNING: --syzygy requires --features syzygy (ignored)");
            }
            datagen::run(threads, positions, depth, &output, book.as_deref());
        }
        #[cfg(any(feature = "stats", feature = "tree"))]
        Cmd::Dump {
            depth, nodes, tt_mb, stats, tree, min_record_depth,
            qs_moves, max_mb, moves, fen,
        } => {
            let opts = dump::DumpOptions {
                fen,
                moves: moves.split_whitespace().map(String::from).collect(),
                depth,
                nodes,
                tt_mb,
                stats_out: stats,
                tree_out: tree,
                min_record_depth,
                qs_moves,
                max_mb,
            };
            dump::run(&opts);
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
            // On x86-64 the SIMD paths are resolved from CPUID at startup; the
            // report shows what this machine actually runs, not what the build
            // could. Other architectures have a single compile-time backend.
            #[cfg(target_arch = "x86_64")]
            {
                use gaiachess::cpu;
                let d = cpu::get_or_init();
                let forced = match d.forced {
                    Some(cap) => format!(" (GAIA_SIMD={} forced)", cap.name()),
                    None => String::new(),
                };
                let how = if cpu::statically_pinned() { "static" } else { "runtime" };
                println!(
                    "SIMD:        {} ({how}), nnz {}{forced}",
                    d.tier.name(),
                    d.nnz.name(),
                );
                let slider = cpu::effective_attacks();
                let setwise = if cpu::effective_setwise512() {
                    "AVX-512 Kogge-Stone"
                } else {
                    "per-piece loop"
                };
                let pext_note = if d.pext_forced { " (GAIA_PEXT forced)" } else { "" };
                let how = if cfg!(gaia_dist) { "runtime" } else { "static" };
                println!("Movegen:     {} ({how}), setwise {setwise}{pext_note}", slider.name());
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                let simd = if cfg!(all(target_arch = "aarch64", target_feature = "neon")) {
                    "NEON"
                } else if cfg!(all(target_arch = "wasm32", target_feature = "simd128")) {
                    "wasm simd128"
                } else {
                    "scalar"
                };
                println!("SIMD:        {simd}");
                println!("Movegen:     magic, setwise per-piece loop");
            }
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
            if cfg!(feature = "gui") {
                features.push("GUI");
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
        #[cfg(feature = "gui")]
        Cmd::Gui { .. } => unreachable!("the gui command is dispatched from main()"),
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The bare invocation is caught by a hidden variant that must consume nothing, or
    /// `gaiachess` on its own stops working. Adding the `--no-gui` switch to it is
    /// exactly the kind of change that could break that, so it is pinned here.
    #[test]
    fn a_bare_launch_still_lands_on_the_uci_catch_all() {
        let none: &[&str] = &[];
        let parsed = cmd().run_inner(none).expect("a bare launch must parse");
        assert!(matches!(parsed, Cmd::Uci(UciMode { no_gui: false })));

        let flag: &[&str] = &["--no-gui"];
        let parsed = cmd().run_inner(flag).expect("--no-gui must parse");
        assert!(matches!(parsed, Cmd::Uci(UciMode { no_gui: true })));
    }
}
