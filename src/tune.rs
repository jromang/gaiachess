//! SPSA tuning infrastructure.
//!
//! Provides the `tunable!` and `fixed!` macros for search parameters:
//! - `tunable!`: parameters exported to SPSA tuners (CSV/JSON/UCI options)
//! - `fixed!`: same getters, settable via UCI `setoption`, but excluded from tuner export
//!
//! Both modes:
//! - **Without `spsa` feature**: `const fn` getters (zero overhead, compiler inlines)
//! - **With `spsa` feature**: `AtomicI32` storage + runtime setters via UCI `setoption`

// ─── Non-SPSA mode: compile-time constants ──────────────────────────────────

#[cfg(not(feature = "spsa"))]
macro_rules! tunable {
    ($(
        $(#[doc = $doc:literal])*
        $name:ident = $default:expr, $min:expr, $max:expr, $step:expr;
    )*) => {
        $(
            $(#[doc = $doc])*
            #[allow(non_snake_case)]
            #[inline(always)]
            pub const fn $name() -> i32 { $default }
        )*
    };
}

#[cfg(not(feature = "spsa"))]
macro_rules! fixed {
    ($(
        $(#[doc = $doc:literal])*
        $name:ident = $default:expr, $min:expr, $max:expr, $step:expr;
    )*) => {
        $(
            $(#[doc = $doc])*
            #[allow(non_snake_case)]
            #[inline(always)]
            pub const fn $name() -> i32 { $default }
        )*
    };
}

// ─── SPSA mode: AtomicI32 storage with runtime mutation ─────────────────────

#[cfg(feature = "spsa")]
macro_rules! tunable {
    ($(
        $(#[doc = $doc:literal])*
        $name:ident = $default:expr, $min:expr, $max:expr, $step:expr;
    )*) => {
        mod storage {
            use std::sync::atomic::AtomicI32;
            $(
                #[allow(non_upper_case_globals)]
                pub static $name: AtomicI32 = AtomicI32::new($default);
            )*
        }

        $(
            $(#[doc = $doc])*
            #[allow(non_snake_case)]
            #[inline(always)]
            pub fn $name() -> i32 {
                storage::$name.load(std::sync::atomic::Ordering::Relaxed)
            }
        )*

        /// Set a tunable parameter by name. Returns `true` if the name was found.
        fn set_param_tunable(name: &str, value: i32) -> bool {
            match name {
                $(
                    stringify!($name) => {
                        storage::$name.store(value, std::sync::atomic::Ordering::Relaxed);
                        true
                    }
                )*
                _ => false,
            }
        }

        /// Print all tunable parameters as UCI `option` lines.
        pub fn print_uci_options() {
            $(
                println!(
                    "option name {} type spin default {} min {} max {}",
                    stringify!($name), $default, $min, $max,
                );
            )*
        }

        /// Emit OpenBench-compatible CSV (one parameter per line).
        pub fn emit_csv() -> String {
            let mut lines: Vec<String> = Vec::new();
            $(
                lines.push(format!(
                    "{}, int, {}, {}, {}, {}, 0.002",
                    stringify!($name),
                    storage::$name.load(std::sync::atomic::Ordering::Relaxed),
                    $min, $max, $step,
                ));
            )*
            lines.join("\n")
        }

        /// Emit WeatherFactory-compatible JSON.
        pub fn emit_json() -> String {
            let mut entries: Vec<String> = Vec::new();
            $(
                entries.push(format!(
                    concat!(
                        "  \"{}\": {{\n",
                        "    \"value\": {},\n",
                        "    \"min_value\": {},\n",
                        "    \"max_value\": {},\n",
                        "    \"step\": {}\n",
                        "  }}"
                    ),
                    stringify!($name),
                    storage::$name.load(std::sync::atomic::Ordering::Relaxed),
                    $min, $max, $step,
                ));
            )*
            format!("{{\n{}\n}}", entries.join(",\n"))
        }
    };
}

#[cfg(feature = "spsa")]
macro_rules! fixed {
    ($(
        $(#[doc = $doc:literal])*
        $name:ident = $default:expr, $min:expr, $max:expr, $step:expr;
    )*) => {
        mod fixed_storage {
            use std::sync::atomic::AtomicI32;
            $(
                #[allow(non_upper_case_globals)]
                pub static $name: AtomicI32 = AtomicI32::new($default);
            )*
        }

        $(
            $(#[doc = $doc])*
            #[allow(non_snake_case)]
            #[inline(always)]
            pub fn $name() -> i32 {
                fixed_storage::$name.load(std::sync::atomic::Ordering::Relaxed)
            }
        )*

        /// Set a fixed parameter by name. Returns `true` if the name was found.
        fn set_param_fixed(name: &str, value: i32) -> bool {
            match name {
                $(
                    stringify!($name) => {
                        fixed_storage::$name.store(value, std::sync::atomic::Ordering::Relaxed);
                        true
                    }
                )*
                _ => false,
            }
        }
    };
}

// ─── Parameter declarations ─────────────────────────────────────────────────
// tunable! = exported to SPSA tuners (92 params)
// fixed!   = locked, not exported (10 params)

tunable! {
    // === Aspiration windows (5) ===
    /// Initial aspiration window half-width (centipawns).
    ASP_INITIAL_DELTA = 18, 5, 40, 2;
    /// Minimum depth to enable aspiration windows.
    ASP_MIN_DEPTH = 7, 2, 8, 1;
    /// Divisor for avg_score contribution to delta.
    ASP_SCORE_DIV = 87, 32, 512, 16;
    /// Delta threshold to fall back to full window.
    ASP_FALLBACK_DELTA = 371, 200, 1000, 50;
    /// Widening divisor: `delta += delta / X`.
    ASP_WIDEN_DIV = 6, 2, 6, 1;

    // === Reverse futility pruning (3) ===
    /// Maximum depth for reverse futility pruning.
    RFP_MAX_DEPTH = 13, 4, 16, 1;
    /// RFP margin per ply (centipawns).
    RFP_MARGIN = 105, 20, 200, 10;
    /// RFP minimum margin (centipawns).
    RFP_MIN = 16, 0, 100, 5;
    /// RFP ttHit multiplier: reduce the RFP margin multiplier when there is no TT hit.
    RFP_NO_TT_DISCOUNT = 27, 10, 40, 2;

    // === Razoring (3) ===
    /// Razoring base margin (centipawns).
    RAZOR_BASE = 300, 100, 600, 25;
    /// Razoring quadratic depth² coefficient (centipawns).
    RAZOR_QUAD = 260, 100, 500, 25;
    /// Razoring alpha max: do not razor in decisive positions.
    RAZOR_ALPHA_MAX = 2048, 512, 4096, 128;

    // === SEE pruning in search (3) ===
    /// SEE quiet move margin per lmr_depth^2 (centipawns, negative).
    SEE_QUIET_MARGIN = -14, -60, 0, 5;
    /// SEE capture move margin per depth (centipawns, negative).
    SEE_CAPTURE_MARGIN = -110, -200, 0, 10;
    /// History divisor in lmr_depth for quiet SEE pruning.
    PRUNE_HIST_DIV = 7113, 3000, 16000, 650;

    // === Full-depth search gating (3) ===
    /// Reduction-signal addend when no TT move backs the position.
    EMR_NOTM = 811, 0, 2000, 100;
    /// Reduction-signal threshold for the first ply cut in full-depth search.
    EMR_R_ONE = 4501, 2000, 8000, 200;
    /// Reduction-signal threshold for the second ply cut in full-depth search.
    EMR_R_TWO = 6183, 3000, 10000, 200;

    // === Futility pruning (3) ===
    /// Maximum depth to apply futility pruning.
    FP_MAX_DEPTH = 13, 4, 16, 1;
    /// Futility pruning margin per depth (centipawns).
    FP_COEFF = 128, 30, 200, 10;
    /// Futility pruning base margin subtracted (centipawns).
    FP_BASE = 128, 30, 300, 15;

    // === History pruning (1) ===
    /// History pruning margin: prune quiets with history < X * depth.
    HIST_PRUNE_MARGIN = -5132, -10000, -1000, 500;
    /// History pruning depth gradient: `HIST_PRUNE_MARGIN + X * depth / 1024`.
    HIST_PRUNE_DEPTH = 3955, -5000, 5000, 500;

    // === Opponent worsening + hindsight (5) ===
    /// Opponent worsening threshold: static_eval + prev_eval > X.
    OW_THRESHOLD = 24, -50, 100, 5;
    /// RFP bonus when opponent position is worsening (centipawns).
    RFP_OW_BONUS = 10, 0, 100, 5;
    /// Min prior reduction (plies) for hindsight depth++.
    HINDSIGHT_INC_MIN_R = 2, 2, 5, 1;
    /// Min prior reduction (plies) for hindsight depth--.
    HINDSIGHT_DEC_MIN_R = 1, 1, 4, 1;
    /// Eval sum threshold for hindsight depth--.
    HINDSIGHT_DEC_THRESHOLD = 101, 0, 300, 15;

    // === Static history (eval change bonus) (3) ===
    /// Static history factor: X * -(eval + prev_eval) / 128.
    STATIC_HIST_FACTOR = 899, 200, 2000, 50;
    /// Static history minimum clamp.
    STATIC_HIST_MIN = -139, -500, 0, 20;
    /// Static history maximum clamp.
    STATIC_HIST_MAX = 277, 50, 1000, 30;

    // === Late move pruning (1) ===
    /// Base in LMP threshold: `(d^2 + base) / (2 - improving)`.
    LMP_BASE = 11, 1, 14, 1;

    // === Singular extensions (5) ===
    /// Minimum depth to attempt singular extension.
    SE_MIN_DEPTH = 7, 3, 10, 1;
    /// TT depth margin for SE: `tt_depth >= depth - X`.
    SE_TT_DEPTH_MARGIN = 5, 1, 6, 1;
    /// Double extension margin below singular_beta (centipawns).
    SE_DOUBLE_MARGIN = 16, 1, 60, 3;
    /// Double extension depth gradient: `SE_DOUBLE_MARGIN + X * depth / 1024`.
    SE_DOUBLE_DEPTH = -43, -120, 60, 5;
    /// Triple extension margin below singular_beta (centipawns).
    SE_TRIPLE_MARGIN = 91, 30, 300, 15;
    /// Triple extension depth gradient: `SE_TRIPLE_MARGIN + X * depth / 1024`.
    SE_TRIPLE_DEPTH = 182, -200, 200, 20;
    /// Max cumulative double+ extensions along a path.
    SE_MAX_DOUBLE_EXT = 8, 2, 12, 1;
    /// Extra depth-proportional SE margin on ex-PV nodes (tt_pv && !PV).
    /// Wider singular-extension margin where the node was once on the PV.
    /// Units: 1/128 of depth (128 = one extra depth).
    SE_TTPV_DEPTH = 89, 0, 300, 15;
    /// Divisor for |correction_value| in the SE double/triple margin adjustment.
    /// When |correction_value| is large, the static eval is unreliable → extend more.
    /// Smaller divisor = stronger margin adjustment.
    SE_CORR_DIV = 16, 4, 64, 4;

    // === Correction history (6) ===
    /// Pawn correction history weight.
    PAWN_CORR_FACTOR = 6490, 2000, 12000, 500;
    /// Continuation correction history weight.
    CONT_CORR_FACTOR = 7545, 1000, 12000, 500;
    /// Non-pawn correction history weight (sum of white + black lookups).
    NON_PAWN_CORR_FACTOR = 8100, 1000, 12000, 500;
    /// Minor piece correction history weight (N+B+K combined key).
    MINOR_CORR_FACTOR = 4711, 1000, 12000, 500;
    /// Correction history update scaling factor.
    CORR_UPDATE_FACTOR = 204, 30, 300, 15;
    /// Fifty-move rule scaling denominator.
    FIFTY_MOVE_SCALE = 205, 100, 500, 20;

    // === Null move pruning (5) ===
    /// Minimum depth for null move pruning.
    NMP_MIN_DEPTH = 3, 2, 6, 1;
    /// Base null move reduction.
    NMP_BASE_R = 5, 2, 8, 1;
    /// Depth divisor: `depth / X` added to R.
    NMP_DEPTH_DIV = 3, 2, 6, 1;
    /// Eval divisor: `(eval - beta) / X`.
    NMP_EVAL_DIV = 302, 50, 500, 25;
    /// Max eval-based R bonus.
    NMP_EVAL_MAX = 6, 1, 10, 1;
    /// Minimum depth for NMP verification search.
    NMP_VERIF_DEPTH = 10, 4, 20, 2;
    /// Depth reduction for NMP verification search.
    NMP_VERIF_REDUCTION = 5, 2, 8, 1;

    // === ProbCut (3) ===
    /// Minimum depth for ProbCut pruning.
    PROBCUT_MIN_DEPTH = 5, 3, 8, 1;
    /// ProbCut beta margin (centipawns above beta).
    PROBCUT_MARGIN = 170, 100, 350, 20;
    /// ProbCut shallow search depth reduction.
    PROBCUT_REDUCTION = 6, 1, 6, 1;

    // === LMR centipawn system (18) ===
    // Defaults below come from an SPSA retune of the LMR family around the
    // smooth log-log term (1115 iterations x 400 games, TC 2+0.02).
    /// Base LMR: `X * LMR_LN[move_count] * LMR_LN[depth] / 1024`.
    /// Initial value was a least-squares fit to the previous ilog2-product form
    /// over a leaf-weighted (mc, depth) domain (bias ~0); SPSA settled slightly above it.
    LMR_LOG_MUL = 710, 256, 1280, 40;
    /// Constant offset added to the LMR log-log term (1/1024 ply).
    LMR_LOG_BASE = -12, -512, 512, 32;
    /// Quiet move base penalty (centipawns).
    LMR_QUIET_BASE = 1670, 1000, 3000, 100;
    /// Quiet history: `X * combined_history / 1024` subtracted.
    LMR_QUIET_HIST_MUL = 130, 50, 400, 20;
    /// Capture base penalty (centipawns).
    LMR_CAPTURE_BASE = 1040, 500, 2500, 100;
    /// Capture history: `X * cap_hist / 1024` subtracted.
    LMR_CAPTURE_HIST_MUL = 172, 30, 300, 15;
    /// PV node base bonus subtracted from R.
    LMR_PV_BASE = 395, 100, 800, 40;
    /// PV window scaling: `X * (beta-alpha) / root_delta`.
    LMR_PV_WINDOW_MUL = 486, 100, 800, 40;
    /// TT PV base bonus subtracted from R.
    LMR_TTPV_BASE = 535, 100, 700, 35;
    /// TT PV score bonus: `X * (tt_score > alpha)`.
    LMR_TTPV_SCORE_MUL = 545, 200, 1200, 50;
    /// TT PV depth bonus: `X * (tt_depth >= depth)`.
    LMR_TTPV_DEPTH_MUL = 1200, 300, 1500, 60;
    /// TT PV depth-linear term: `X * depth`.
    LMR_TTPV_DEPTH_LIN = 23, 0, 64, 8;
    /// Cut node penalty added to R.
    LMR_CUTNODE_BASE = 1670, 800, 3000, 100;
    /// Cut node depth gradient: `LMR_CUTNODE_BASE + X * depth / 1024`.
    LMR_CUTNODE_DEPTH = -420, -2000, 2000, 200;
    /// No TT move penalty: `X * (tt_move == NONE)`.
    LMR_CUTNODE_NOTM_MUL = 850, 400, 3000, 80;
    /// Not-improving base penalty.
    LMR_NOT_IMPROVING_BASE = 220, 100, 1200, 40;
    /// Not-improving depth gradient: `LMR_NOT_IMPROVING_BASE + X * depth / 1024`.
    LMR_NOT_IMPROVING_DEPTH = -190, -800, 800, 80;
    /// Not-improving scale: `X * improvement / 128`.
    LMR_NOT_IMPROVING_MUL = 125, 80, 900, 30;
    /// Max not-improving penalty.
    LMR_NOT_IMPROVING_MAX = 1300, 500, 2500, 100;
    /// Gives-check bonus subtracted from R.
    LMR_CHECK_SUB = 475, 200, 2500, 100;
    /// Bad TT score penalty added to R.
    LMR_BAD_TT_SCORE = 245, 200, 1200, 50;
    /// LMR penalty when TT depth is shallower than current depth (1/1024 ply).
    /// A shallow TT entry is weaker evidence for the move, so reduce it more.
    LMR_TT_SHALLOW_DEPTH = 219, 50, 800, 40;
    /// Correction history LMR: reduce less when |correction_value| is large.
    /// Positions with strong correction are hard to evaluate statically → search deeper.
    /// Scales the LMR adjustment derived from the correction magnitude.
    LMR_CORR_HIST_MUL = 2640, 500, 5000, 200;

    /// LMR overextension for strongly boosted early moves.
    /// Reduction threshold r (in 1/1024 plies) that triggers the overextension.
    /// The move must accumulate 3.5+ plies of net bonuses (history/PV/check).
    LMR_OVEREXT_THRESHOLD = -3625, -6144, -1024, 256;
    /// Max move_count allowed for LMR overextension.
    LMR_OVEREXT_MAX_MC = 5, 1, 8, 1;
    /// LMR recapture bonus: reduce less for noisy recaptures (plies in units of 1/1024).
    RECAPTURE_LMR_BONUS = 920, 256, 1536, 32;

    // === Adaptive re-search (3) ===
    // Two-tiered do-deeper re-search, plus reduced-depth shallower re-search.
    /// Deeper tier 1 base: `score > best_score + X`.
    DEEPER_BASE = 64, 10, 150, 5;
    /// Deeper tier 2 base: `score > best_score + X` → +2 depth for major tactical moves.
    DEEPER2_BASE = 860, 400, 1200, 50;
    /// Shallower base: `score < best_score + X + reduced_depth`.
    SHALLOWER_BASE = 17, 0, 30, 3;

    // === Qsearch pruning (3) ===
    /// Qsearch futility base: `stand_pat + X` (unified delta + futility).
    QS_FUTILITY_BASE = 341, 50, 500, 25;
    /// Qsearch SEE pruning threshold (centipawns, negative).
    QS_SEE_THRESHOLD = -83, -200, 0, 10;
    /// Divisor adjusting the qsearch SEE threshold by the move's history.
    /// (good history relaxes the SEE margin, bad history tightens it)
    QS_SEE_HIST_DIV = 48, 16, 128, 8;
    /// Non-PV qsearch move count limit.
    QS_MOVE_COUNT_LIMIT = 3, 2, 6, 1;

    // === IIR (1) ===
    /// IIR depth threshold: `depth >= X + 2 * cut_node`.
    IIR_DEPTH = 2, 1, 6, 1;

    // === History bonus/malus (6) ===
    /// History bonus: `X * depth + offset`.
    STAT_BONUS_MUL = 153, 50, 400, 15;
    /// History bonus offset: `mul * depth + X`.
    STAT_BONUS_ADD = -25, -50, 100, 8;
    /// History bonus max: `.min(X)`.
    STAT_BONUS_MAX = 1570, 500, 3000, 100;
    /// History malus: `X * depth - offset`.
    STAT_MALUS_MUL = 220, 50, 400, 15;
    /// History malus offset: `mul * depth - X`.
    STAT_MALUS_SUB = 47, -50, 100, 8;
    /// History malus max: `.min(X)`.
    STAT_MALUS_MAX = 481, 200, 2500, 100;

    // === Non-PV best-move history bonus scaled by move count ===
    /// Divisor scaling the bonus by the number of moves tried at non-PV nodes.
    /// Smaller = stronger effect, larger = weaker effect.
    BESTMOVE_MC_DIV = 241, 64, 1024, 32;

    // === Move ordering (6) ===
    /// MVV multiplier in capture scoring: `PIECE_VALUE[victim] * X`.
    MVV_MULTIPLIER = 17, 4, 64, 4;
    /// Good/bad capture split: base SEE threshold (centipawns, negative).
    GOOD_CAP_BASE = -106, -300, 0, 16;
    /// Good/bad capture split: capture history divisor shifting the threshold.
    GOOD_CAP_HIST_DIV = 24, 8, 96, 4;
    /// Movepicker bonus for quiet moves that give direct check.
    MOVEPICK_CHECK_BONUS = 10000, 4000, 20000, 1000;
    /// Learned look-ahead: TT child eval multiplier (numerator).
    LOOKAHEAD_BONUS_MUL = 8, 0, 32, 2;
    /// Learned look-ahead: TT child eval divisor (denominator).
    LOOKAHEAD_BONUS_DIV = 1, 1, 8, 1;
    /// Learned look-ahead: minimum depth to enable TT child probing.
    LOOKAHEAD_MIN_DEPTH = 4, 1, 12, 1;

    // === Lazy eval (1) ===
    /// PeSTO score threshold to skip NNUE forward pass (centipawns, absolute).
    LAZY_EVAL_THRESHOLD = 549, 200, 1500, 50;

    // === Optimism (4) ===
    /// Optimism sigmoid numerator.
    OPT_NUMERATOR = 127, 50, 300, 10;
    /// Optimism sigmoid offset.
    OPT_OFFSET = 206, 50, 400, 10;
    /// Material scale base for raw eval.
    OPT_MAT_SCALE = 942, 200, 2000, 50;
    /// Material base for optimism weight.
    OPT_MAT_BASE = 2083, 500, 4000, 200;

    // === Prior Countermove Bonus on Unexpected All-Nodes (2) ===
    // Bonus to the parent's quiet move when this node fails low
    // unexpectedly (predicted cut or PV but the result is an all-node).
    /// Butterfly history factor: PCM_ALLNODE_FACTOR * stat_bonus(depth) / 128.
    PCM_ALLNODE_FACTOR = 150, 50, 400, 16;
    /// Continuation history factor: PCM_ALLNODE_CONTHIST_FACTOR * stat_bonus(depth) / 128.
    PCM_ALLNODE_CONTHIST_FACTOR = 100, 30, 300, 16;

    // === Prior Capture Move Bonus on Unexpected All-Nodes ===
    // Capture history bonus for the parent's capture move at unexpected all-nodes.
    /// Capture history factor: PCM_ALLNODE_CAPTURE_FACTOR * stat_bonus(depth) / 128.
    PCM_ALLNODE_CAPTURE_FACTOR = 64, 16, 200, 12;
}

// === Move Overhead — standalone, exposed as UCI option ===
static MOVE_OVERHEAD_VALUE: std::sync::atomic::AtomicI32 =
    std::sync::atomic::AtomicI32::new(100);

/// Overhead subtracted from available time to prevent flagging (ms).
#[allow(non_snake_case)]
#[inline(always)]
pub fn MOVE_OVERHEAD() -> i32 {
    MOVE_OVERHEAD_VALUE.load(std::sync::atomic::Ordering::Relaxed)
}

/// Set move overhead (called from UCI `setoption name Move Overhead`).
pub fn set_move_overhead(value: i32) {
    MOVE_OVERHEAD_VALUE.store(value.clamp(0, 5000), std::sync::atomic::Ordering::Relaxed);
}

fixed! {
    // === Time management (7) — locked, not exported to SPSA ===
    /// TM best-move stability base multiplier (÷1000).
    TM_BM_BASE = 1563, 750, 2500, 50;
    /// TM best-move stability factor per stable iteration (÷1000).
    TM_BM_FACTOR = 53, 10, 100, 5;
    /// TM best-move max stability count.
    TM_BM_MAX = 18, 5, 30, 2;
    /// TM score stability base multiplier (÷1000).
    TM_SCORE_BASE = 953, 500, 1500, 30;
    /// TM score stability factor per centipawn drop (÷1000).
    TM_SCORE_FACTOR = 4, 1, 20, 1;
    /// TM score diff clamp minimum.
    TM_SCORE_MIN = -8, -100, 0, 5;
    /// TM score diff clamp maximum.
    TM_SCORE_MAX = 63, 10, 200, 10;
    /// TM node fraction base (÷1000).
    TM_NODES_BASE = 1680, 1000, 2500, 50;
    /// TM node fraction factor (÷1000).
    TM_NODES_FACTOR = 974, 100, 1500, 50;
}

/// Set a parameter (tunable or fixed) by name. Returns `true` if found.
#[cfg(feature = "spsa")]
pub fn set_param(name: &str, value: i32) -> bool {
    if name == "MOVE_OVERHEAD" {
        set_move_overhead(value);
        return true;
    }
    set_param_tunable(name, value) || set_param_fixed(name, value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tunable_defaults() {
        // Spot-check that getter functions return the declared defaults.
        assert_eq!(ASP_INITIAL_DELTA(), 18);
        assert_eq!(RFP_MARGIN(), 105);
        assert_eq!(SEE_QUIET_MARGIN(), -14);
        assert_eq!(LMR_LOG_MUL(), 710);
        assert_eq!(NMP_BASE_R(), 5);
        assert_eq!(STAT_BONUS_MUL(), 153);
        assert_eq!(MVV_MULTIPLIER(), 17);
        assert_eq!(QS_SEE_THRESHOLD(), -83);
    }

    #[test]
    fn test_fixed_defaults() {
        assert_eq!(MOVE_OVERHEAD(), 100);
        assert_eq!(TM_BM_BASE(), 1563);
        assert_eq!(TM_SCORE_MAX(), 63);
    }

    #[cfg(feature = "spsa")]
    #[test]
    fn test_set_param() {
        // Verify runtime mutation works for tunable params.
        let old = RFP_MARGIN();
        assert!(set_param("RFP_MARGIN", 100));
        assert_eq!(RFP_MARGIN(), 100);
        assert!(set_param("RFP_MARGIN", old));
        assert_eq!(RFP_MARGIN(), old);

        // Verify runtime mutation works for MOVE_OVERHEAD (standalone).
        let old_mo = MOVE_OVERHEAD();
        assert!(set_param("MOVE_OVERHEAD", 200));
        assert_eq!(MOVE_OVERHEAD(), 200);
        assert!(set_param("MOVE_OVERHEAD", old_mo));
        assert_eq!(MOVE_OVERHEAD(), old_mo);

        // Unknown param returns false
        assert!(!set_param("NONEXISTENT", 42));
    }

    #[cfg(feature = "spsa")]
    #[test]
    fn test_emit_csv() {
        let csv = emit_csv();
        // Tunable params are exported
        assert!(csv.contains("RFP_MARGIN"));
        assert!(csv.contains("LMR_LOG_MUL"));
        assert!(csv.contains("0.002"));
        // Fixed params are NOT exported
        assert!(!csv.contains("MOVE_OVERHEAD"));
        assert!(!csv.contains("TM_BM_BASE"));
        // One line per tunable, and the same set as the JSON export. Counting them
        // against a number written here by hand is what this used to do, and the number
        // was left behind at 90 while the list grew past a hundred: a test that has to
        // be edited every time a technique lands is a test nobody edits.
        assert_eq!(csv.lines().count(), emit_json().matches("\"min_value\"").count());
        assert!(csv.lines().count() > 80, "the tunable list has collapsed");
    }

    #[cfg(feature = "spsa")]
    #[test]
    fn test_emit_json() {
        let json = emit_json();
        assert!(json.starts_with('{'));
        assert!(json.ends_with('}'));
        // Tunable params are exported
        assert!(json.contains("\"RFP_MARGIN\""));
        assert!(json.contains("\"min_value\""));
        // Fixed params are NOT exported
        assert!(!json.contains("\"MOVE_OVERHEAD\""));
        assert!(!json.contains("\"TM_BM_BASE\""));
    }
}
