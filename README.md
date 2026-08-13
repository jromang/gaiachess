# ♚ Gaia Chess Engine

A free and strong UCI chess engine written in Rust.

> See also: [gaiachess.fr](http://gaiachess.free.fr/) — the original Gaia website

**Gaia 4** is a complete rewrite from C to Rust — not a single line of the original code survived. Back after 20 years.

## Features

### Search

- Principal Variation Search (PVS) with aspiration windows
- Null Move Pruning
- Late Move Reductions (LMR) with history, improving, correction history adjustments
- Singular Extensions with multi-cut
- Reverse Futility Pruning, Futility Pruning, Late Move Pruning
- SEE Pruning (main search + quiescence)
- History Pruning, ProbCut
- Internal Iterative Reductions (IIR)
- Killer Moves, Countermove Heuristic, Butterfly/Capture/Continuation/Pawn History
- Lazy SMP (shared TT, per-thread history)

### Endgame tablebases

- **Syzygy** WDL/DTZ probing up to 7 pieces (`SyzygyPath`)
- **Nalimov** DTM probing (`NalimovPath`) — tablebases available on [Hugging Face](https://huggingface.co/datasets/jromanghf/nalimov-tablebases)
- **GaiaTB**: 3–4 piece DTM tablebases in a custom compressed format, **embedded in the binary** — exact mate distances with zero configuration
- Optional **online tablebase probing** at the root (`OnlineTB`, off by default)

### Evaluation

- **NNUE**: GaiaNet threat-feature architecture (12 king buckets × 768 PST + filtered threat features, ~41K inputs → 640 → CReLU+pairwise → 16 → 32 → 1), 8 output buckets, trained with [Bullet](https://github.com/jw1912/bullet) on self-play data
- **SIMD**: compile-time dispatch AVX-512 / AVX2 / scalar
- **Movegen**: per-piece PEXT (BMI2) / AVX2 BLSMSK / magic; setwise AVX-512 Kogge-Stone / scalar — compile-time selected

### UCI

Full UCI protocol support. Compatible with [Arena](https://www.playwitharena.de/), [CuteChess](https://cutechess.com/), [En Croissant](https://encroissant.org/), and all major chess GUIs.

## Which binary should I use?

Releases ship one binary per CPU family. **Windows** binaries end in `.exe`, **Linux** and **macOS** have no extension.

| CPU | Recommended binary |
|-----|-------------------|
| **AMD Ryzen 9000** (Zen 5) | `znver5` or `avx512vnni` |
| **AMD Ryzen 7000** (Zen 4) | `znver4` or `avx512` |
| **AMD Ryzen 5000** (Zen 3) | `znver3` or `bmi2` |
| **AMD Ryzen 1000–3000** (Zen 1/2) | `avx2` |
| **Intel 12th gen+** (Alder Lake+) | `avx512vnni` |
| **Intel 10th–11th gen** | `avx512` or `bmi2` |
| **Intel Haswell–Coffee Lake** | `bmi2` |
| **Apple M1/M2/M3/M4** | `apple-silicon` |
| **Linux ARM64** (RPi 4+, Graviton) | `neon` |
| **Older CPUs** (pre-2013) | `sse4-popcnt`, `ssse3`, or `x86-64` |

When in doubt, use **`bmi2`** (x86) or **`neon`** (ARM). If it crashes on startup, fall back to `avx2`, then `x86-64`.

All x86-64 binaries are built with Profile-Guided Optimization. NNUE binaries embed the neural network and the GaiaTB tablebases — no external file needed.

> The NNUE binaries are several hundred Elo stronger than the PeSTO fallbacks (`x86-64`, `ssse3`, `sse4-popcnt`), but require AVX2 for the SIMD inference.

## UCI Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| Hash | spin | 16 | Transposition table size in MB (1–1048576) |
| Threads | spin | 1 | Number of search threads (1–256) |
| EvalFile | string | \<internal\> | Path to a NNUE network file |
| Ponder | check | true | Enable pondering |
| MultiPV | spin | 1 | Number of principal variations (1–256) |
| Move Overhead | spin | 100 | Time reserved per move for I/O latency, in ms (0–5000) |
| SyzygyPath | string | \<empty\> | Path(s) to Syzygy tablebase files (`:`-separated) |
| NalimovPath | string | \<empty\> | Path to Nalimov tablebase files |
| OnlineTB | check | false | Probe online endgame tablebases at the root |

## The Story

The first version of Gaia was born in June 2003, in a student apartment in Strasbourg. Back then, David Rabel and I were writing C code between classes, debugging with printf, and testing against Crafty on a single-core Pentium. We were young, we were clueless, we were having a blast.

We released six versions in three years, entered a few French computer chess championships, and then... life happened. Studies, jobs, moving, family. Gaia 3.5 came out in June 2006, and then silence. For almost twenty years. I've since lost touch with David.

In the meantime, the chess programming world has changed beyond recognition. Neural networks went from science fiction to standard equipment. NNUE happened. Open-source engines reached superhuman strength and spawned an entire ecosystem. The [Chess Programming Wiki](https://www.chessprogramming.org/) turned into an incredible knowledge base. Rust appeared and made systems programming actually enjoyable. Tools like SPRT testing brought real statistical rigor to engine development. The community grew, shared, documented everything.

Coming back to chess programming after all this time felt like waking up in the future. Everything I struggled with in 2003 now has a name, a wiki page, and a dozen open-source implementations to learn from. It's humbling and exhilarating at the same time.

So I started over. From scratch. Alone, this time, but standing on the shoulders of an incredible community.

## Acknowledgements

Thanks to Patrick Buchmann, Alex Schmidt, Guenther Simon, Claude Dubois, Leo Dijksman, Patrick Beucler, Dann Corbit, Marcus Geelnard, Gabriel Leperlier, and Raphael Grundrich for their help and support on the early versions of Gaia.

Thanks to Werner Schüle for testing Gaia and reporting tablebase-related bugs.

Gaia 4 stands on the shoulders of the chess programming community. Special thanks to:

- [Bullet](https://github.com/jw1912/bullet) — NNUE trainer
- [Chess Programming Wiki](https://www.chessprogramming.org/) — invaluable knowledge base

## Building from source

Requires a Rust toolchain (edition 2024), `clang` (for the Syzygy probing code), `curl`, and a NNUE network file:

```bash
# Download the network (see https://huggingface.co/jromanghf/gaiachess-networks)
curl -L -o nets/gaianet.bin \
  "https://huggingface.co/jromanghf/gaiachess-networks/resolve/main/<network-name>.bin"

# Build with the network embedded
MODEL=nets/gaianet.bin RUSTFLAGS="-C target-cpu=native" cargo build --release --features nnue

# Fallback build without NNUE (PeSTO evaluation, much weaker)
cargo build --release
```

Run `cargo test` for the test suite and `cargo run --release -- bench` for the search benchmark.

The NNUE trainer lives in `trainer/` (see its `--help` for usage).

## License

Gaia is free software, distributed under the [GNU General Public License version 3](LICENSE) (or any later version at your option).

© 2003–2026 Jean-François Romang
