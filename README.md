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
- Syzygy tablebases (up to 7 pieces)

### Evaluation

- **NNUE**: HalfKA architecture (9 king buckets × 768 → 512 → CReLU+pairwise → 16 → 32 → 1), 8 output buckets
- **SIMD**: compile-time dispatch AVX-512 / AVX2 / scalar

### UCI

Full UCI protocol support. Compatible with [Arena](https://www.playwitharena.de/), [CuteChess](https://cutechess.com/), [En Croissant](https://encroissant.org/), and all major chess GUIs.

## UCI Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| Hash | spin | 16 | Transposition table size in MB (1–1048576) |
| Threads | spin | 1 | Number of search threads (1–256) |
| EvalFile | string | \<internal\> | Path to a `.nnue` network file |
| Ponder | check | true | Enable pondering |
| MultiPV | spin | 1 | Number of principal variations (1–256) |
| SyzygyPath | string | \<empty\> | Path to Syzygy tablebase files |

## The Story

The first version of Gaia was born in June 2003, in a student apartment in Strasbourg. Back then, David Rabel and I were writing C code between classes, debugging with printf, and testing against Crafty on a single-core Pentium. We were young, we were clueless, we were having a blast.

We released six versions in three years, entered a few French computer chess championships, and then... life happened. Studies, jobs, moving, family. Gaia 3.5 came out in June 2006, and then silence. For almost twenty years. I've since lost touch with David.

In the meantime, the chess programming world has changed beyond recognition. Neural networks went from science fiction to standard equipment. NNUE happened. Stockfish became the undisputed leader and spawned an entire ecosystem. The [Chess Programming Wiki](https://www.chessprogramming.org/) turned into an incredible knowledge base. Rust appeared and made systems programming actually enjoyable. Tools like SPRT testing brought real statistical rigor to engine development. The community grew, shared, documented everything.

Coming back to chess programming after all this time felt like waking up in the future. Everything I struggled with in 2003 now has a name, a wiki page, and a dozen open-source implementations to learn from. It's humbling and exhilarating at the same time.

So I started over. From scratch. Alone, this time, but standing on the shoulders of an incredible community.

## Acknowledgements

Thanks to Patrick Buchmann, Alex Schmidt, Guenther Simon, Claude Dubois, Leo Dijksman, Patrick Beucler, Dann Corbit, Marcus Geelnard, Gabriel Leperlier, and Raphael Grundrich for their help and support on the early versions of Gaia.

Gaia 4 stands on the shoulders of the chess programming community. Special thanks to:

- [Bullet](https://github.com/jw1912/bullet) — NNUE trainer
- [Chess Programming Wiki](https://www.chessprogramming.org/) — invaluable knowledge base

## License

Gaia is distributed free of charge. Gaia may not be distributed as part of any software package, service or web site without prior written permission from the authors.

© 2003–2026 Jean-François Romang
