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
            #[allow(unused_imports)]
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
        fn set_param_fixed(name: &str, _value: i32) -> bool {
            match name {
                $(
                    stringify!($name) => {
                        fixed_storage::$name.store(_value, std::sync::atomic::Ordering::Relaxed);
                        true
                    }
                )*
                _ => false,
            }
        }
    };
}

// ─── Parameter declarations ─────────────────────────────────────────────────
// tunable! = exported to SPSA tuners
// fixed!   = locked, not exported (currently only MOVE_OVERHEAD, which is its own atomic)

tunable! {
    // === Aspiration windows (5) ===
    /// Initial aspiration window half-width (search units).
    ASP_INITIAL_DELTA = 14, 5, 40, 2;
    /// Minimum depth to enable aspiration windows.
    ASP_MIN_DEPTH = 6, 2, 8, 1;
    /// Divisor for avg_score contribution to delta.
    ASP_SCORE_DIV = 88, 32, 512, 16;
    /// Delta threshold to fall back to full window.
    ASP_FALLBACK_DELTA = 319, 200, 1000, 50;
    /// Widening divisor: `delta += delta / X`.
    ASP_WIDEN_DIV = 6, 2, 10, 1;

    // === Reverse futility pruning (3) ===
    /// Maximum depth for reverse futility pruning.
    RFP_MAX_DEPTH = 14, 4, 20, 1;
    /// RFP margin per ply (search units).
    RFP_MARGIN = 88, 20, 200, 10;
    /// RFP minimum margin (search units).
    RFP_MIN = 15, 0, 100, 5;
    /// RFP ttHit multiplier: reduce the RFP margin multiplier when there is no TT hit.
    RFP_NO_TT_DISCOUNT = 27, 10, 40, 2;

    // === Razoring (3) ===
    /// Razoring base margin (search units).
    RAZOR_BASE = 311, 100, 600, 25;
    /// Razoring quadratic depth² coefficient (search units).
    RAZOR_QUAD = 213, 100, 500, 25;
    /// Razoring alpha max: do not razor in decisive positions.
    RAZOR_ALPHA_MAX = 1767, 512, 4096, 128;

    // === SEE pruning in search (3) ===
    /// SEE quiet move margin per lmr_depth^2 (centipawns, negative).
    SEE_QUIET_MARGIN = -8, -60, 0, 5;
    /// SEE capture move margin per depth (centipawns, negative).
    SEE_CAPTURE_MARGIN = -99, -200, 0, 10;
    /// History divisor in lmr_depth for quiet SEE pruning.
    PRUNE_HIST_DIV = 6165, 3000, 16000, 650;

    // === Full-depth search gating (3) ===
    /// Reduction-signal addend when no TT move backs the position.
    EMR_NOTM = 905, 0, 2000, 100;
    /// Reduction-signal threshold for the first ply cut in full-depth search.
    EMR_R_ONE = 4476, 2000, 8000, 200;
    /// Reduction-signal threshold for the second ply cut in full-depth search.
    EMR_R_TWO = 5591, 3000, 10000, 200;

    // === Futility pruning (3) ===
    /// Maximum depth to apply futility pruning.
    FP_MAX_DEPTH = 11, 4, 16, 1;
    /// Futility pruning margin per depth (search units).
    FP_COEFF = 111, 30, 200, 10;
    /// Futility pruning base margin subtracted (search units).
    FP_BASE = 110, 30, 300, 15;

    // === Bad-noisy futility pruning (4) ===
    /// Apply BNFP while lmr_depth in 1024ths of a ply is at most X (~8.9 ply).
    BNFP_DEPTH_LIMIT = 9338, 1024, 16384, 256;
    /// BNFP base margin added to static eval (search units).
    BNFP_BASE = 149, 0, 400, 16;
    /// BNFP margin per 1024ths of lmr_depth.
    BNFP_MULT = 78, 0, 256, 8;
    /// BNFP captured-piece term: `value * X / 1024`.
    BNFP_CAPTURED = 1080, 256, 2048, 64;

    // === History pruning (1) ===
    /// History pruning margin: prune quiets with
    /// `history < (X + HIST_PRUNE_DEPTH * depth / 1024) * (depth + improving - 1)`.
    HIST_PRUNE_MARGIN = -3480, -10000, -1000, 500;
    /// History pruning depth gradient: `HIST_PRUNE_MARGIN + X * depth / 1024`.
    HIST_PRUNE_DEPTH = 3566, -5000, 5000, 500;

    // === Opponent worsening + hindsight (5) ===
    /// Opponent worsening threshold: static_eval + prev_eval > X.
    OW_THRESHOLD = 25, -50, 100, 5;
    /// RFP bonus when opponent position is worsening (search units).
    RFP_OW_BONUS = 8, 0, 100, 5;
    /// Min prior reduction (plies) for hindsight depth++.
    HINDSIGHT_INC_MIN_R = 2, 1, 5, 1;
    /// Min prior reduction (plies) for hindsight depth--.
    HINDSIGHT_DEC_MIN_R = 2, 1, 4, 1;
    /// Eval sum threshold for hindsight depth--.
    HINDSIGHT_DEC_THRESHOLD = 104, 0, 300, 15;

    // === Static history (eval change bonus) (3) ===
    /// Static history factor: X * -(eval + prev_eval) / 128.
    STATIC_HIST_FACTOR = 883, 200, 2000, 50;
    /// Static history minimum clamp.
    STATIC_HIST_MIN = -153, -500, 0, 20;
    /// Static history maximum clamp.
    STATIC_HIST_MAX = 321, 50, 1000, 30;

    // === Late move pruning (1) ===
    /// Base in LMP threshold: `(d^2 + base) / (2 - improving)`.
    LMP_BASE = 15, 1, 20, 1;

    // === Singular extensions (5) ===
    /// Minimum depth to attempt singular extension.
    SE_MIN_DEPTH = 6, 3, 10, 1;
    /// TT depth margin for SE: `tt_depth >= depth - X`.
    SE_TT_DEPTH_MARGIN = 3, 1, 8, 1;
    /// Double extension margin below singular_beta (search units).
    SE_DOUBLE_MARGIN = 9, 1, 60, 3;
    /// Double extension depth gradient: `SE_DOUBLE_MARGIN + X * depth / 1024`.
    SE_DOUBLE_DEPTH = -46, -120, 60, 5;
    /// Triple extension margin below singular_beta (search units).
    SE_TRIPLE_MARGIN = 114, 30, 300, 15;
    /// Triple extension depth gradient: `SE_TRIPLE_MARGIN + X * depth / 1024`.
    SE_TRIPLE_DEPTH = 143, -200, 400, 20;
    /// Max cumulative double+ extensions along a path.
    SE_MAX_DOUBLE_EXT = 9, 2, 12, 1;
    /// Extra depth-proportional SE margin on ex-PV nodes (tt_pv && !PV).
    /// Wider singular-extension margin where the node was once on the PV.
    /// Units: 1/128 of depth (128 = one extra depth).
    SE_TTPV_DEPTH = 72, 0, 300, 15;
    /// Divisor for |correction_value| in the SE double/triple margin adjustment.
    /// When |correction_value| is large, the static eval is unreliable → extend more.
    /// Smaller divisor = stronger margin adjustment.
    SE_CORR_DIV = 26, 4, 64, 4;

    // === Correction history (6) ===
    /// Pawn correction history weight.
    PAWN_CORR_FACTOR = 5921, 2000, 12000, 500;
    /// Continuation correction history weight.
    CONT_CORR_FACTOR = 6901, 1000, 12000, 500;
    /// Non-pawn correction history weight (sum of white + black lookups).
    NON_PAWN_CORR_FACTOR = 6873, 1000, 12000, 500;
    /// Minor piece correction history weight (N+B+K combined key).
    MINOR_CORR_FACTOR = 3256, 1000, 12000, 500;
    /// Correction history update scaling factor.
    CORR_UPDATE_FACTOR = 183, 30, 300, 15;
    /// Fifty-move rule scaling denominator.
    FIFTY_MOVE_SCALE = 229, 100, 500, 20;

    // === Null move pruning (5) ===
    /// Minimum depth for null move pruning.
    NMP_MIN_DEPTH = 5, 2, 8, 1;
    /// Base null move reduction.
    NMP_BASE_R = 4, 2, 8, 1;
    /// Depth divisor: `depth / X` added to R.
    NMP_DEPTH_DIV = 3, 2, 6, 1;
    /// Eval divisor: `(eval - beta) / X`.
    NMP_EVAL_DIV = 258, 50, 500, 25;
    /// Max eval-based R bonus.
    NMP_EVAL_MAX = 5, 1, 10, 1;
    /// Minimum depth for NMP verification search.
    NMP_VERIF_DEPTH = 17, 8, 24, 2;
    /// Null move entry margin: the static eval must beat beta by
    /// `NMP_MARGIN_BASE - NMP_MARGIN_DEPTH * depth - NMP_MARGIN_IMPROVING * improving`
    /// (the form most references use, at about two thirds of their scale: the
    /// full scale was measured and loses Elo against an engine tuned without it).
    /// Without it a defender a few centipawns above beta passes, and the reduced
    /// reply search misses the mate net that a real move would have to face.
    /// No floor on beta: forbidding the null move in lost windows was measured
    /// and costs mates (the eval bonus is already capped by `NMP_EVAL_MAX`).
    NMP_MARGIN_BASE = 102, 0, 1200, 60;
    NMP_MARGIN_DEPTH = 2, 0, 60, 4;
    NMP_MARGIN_IMPROVING = 4, 0, 200, 15;

    // === ProbCut (3) ===
    /// Minimum depth for ProbCut pruning.
    PROBCUT_MIN_DEPTH = 7, 3, 10, 1;
    /// ProbCut beta margin (search units above beta). The references sit around a
    /// pawn or more; at 170 a rook-for-knight sacrifice was cut on the strength
    /// of the recapture alone, the follow-up never searched.
    PROBCUT_MARGIN = 218, 100, 500, 20;
    /// ProbCut shallow search depth reduction. The references verify at depth
    /// minus 3 to 5; at 6 the verification search is too shallow to see the
    /// threat that follows a sacrifice.
    PROBCUT_REDUCTION = 5, 1, 6, 1;

    // === LMR centipawn system (18) ===
    // Defaults below come from an SPSA retune of the LMR family around the
    // smooth log-log term (1115 iterations x 400 games, TC 2+0.02).
    /// Base LMR: `X * LMR_LN[move_count] * LMR_LN[depth] / 1024`.
    /// Initial value was a least-squares fit to the previous ilog2-product form
    /// over a leaf-weighted (mc, depth) domain (bias ~0); SPSA settled slightly above it.
    LMR_LOG_MUL = 630, 256, 1280, 40;
    /// Constant offset added to the LMR log-log term (1/1024 ply).
    LMR_LOG_BASE = -83, -512, 512, 32;
    /// Quiet move base penalty (1/1024 ply).
    LMR_QUIET_BASE = 1691, 1000, 3000, 100;
    /// Quiet history: `X * combined_history / 1024` subtracted.
    LMR_QUIET_HIST_MUL = 90, 0, 400, 20;
    /// Capture base penalty (1/1024 ply).
    LMR_CAPTURE_BASE = 1087, 500, 2500, 100;
    /// Capture history: `X * cap_hist / 1024` subtracted.
    LMR_CAPTURE_HIST_MUL = 169, 30, 300, 15;
    /// PV window scaling: `X * (beta-alpha) / root_delta`.
    LMR_PV_WINDOW_MUL = 462, 100, 800, 40;
    /// TT PV score bonus: `X * (tt_score > alpha)`.
    LMR_TTPV_SCORE_MUL = 441, 200, 1200, 50;
    /// TT PV depth bonus: `X * (tt_depth >= depth)`.
    LMR_TTPV_DEPTH_MUL = 1312, 300, 3000, 60;
    /// TT PV depth-linear term: `X * depth`.
    LMR_TTPV_DEPTH_LIN = 20, 0, 64, 8;
    /// Cut node depth gradient: `X * depth / 1024` (the flat part lives in the factorized term).
    LMR_CUTNODE_DEPTH = -848, -2000, 2000, 200;
    /// Not-improving depth gradient: `X * depth / 1024` (the flat part lives in the factorized term).
    LMR_NOT_IMPROVING_DEPTH = -84, -800, 800, 80;
    /// Not-improving scale: `X * improvement / 128`.
    LMR_NOT_IMPROVING_MUL = 130, 80, 900, 30;
    /// Max not-improving penalty.
    /// Cap on the not-improving magnitude term. 1080 = the former 1300 less the
    /// 220 flat part that moved into the factorized intercept.
    LMR_NOT_IMPROVING_MAX = 888, 0, 2500, 100;
    /// Bad TT score penalty added to R.
    LMR_BAD_TT_SCORE = 247, 0, 1200, 50;
    /// LMR penalty when TT depth is shallower than current depth (1/1024 ply).
    /// A shallow TT entry is weaker evidence for the move, so reduce it more.
    LMR_TT_SHALLOW_DEPTH = 154, 50, 800, 40;
    /// Correction history LMR: reduce less when |correction_value| is large.
    /// Positions with strong correction are hard to evaluate statically → search deeper.
    /// Scales the LMR adjustment derived from the correction magnitude.
    LMR_CORR_HIST_MUL = 3050, 500, 5000, 200;

    /// LMR overextension for strongly boosted early moves.
    /// Reduction threshold r (in 1/1024 plies) that triggers the overextension.
    /// The move must accumulate 3.5+ plies of net bonuses (history/PV/check).
    LMR_OVEREXT_THRESHOLD = -3563, -6144, -1024, 256;
    /// Max move_count allowed for LMR overextension.
    LMR_OVEREXT_MAX_MC = 5, 1, 8, 1;
    /// LMR recapture bonus: reduce less for noisy recaptures (plies in units of 1/1024).
    RECAPTURE_LMR_BONUS = 972, 256, 1536, 32;

    // === LMR terms the reference engine has and we did not (4) ===
    // All default to zero, so the search is unchanged until they are tuned; the
    // point is to put the whole reference surface in front of the tuner at once
    // instead of discovering it one branch at a time.
    /// Reduce more when the TT move is a capture, at the reference value. The -110 the
    /// SPSA found earlier was learned on a signal read from the child's board, true for
    /// nearly every TT move (P28 of TODO_SEARCH_BUGS).
    LMR_TTM_CAPTURE = 1079, -1024, 1536, 64;
    /// Reduce less for a killer move (reference: -1084 on both killers, never
    /// the countermove — the exclusion is deliberate there).
    LMR_KILLER_SUB = 40, -1024, 2048, 64;
    /// Parent reduction fed back into the child's reduction, in 1/1024 ply per
    /// ply the parent was reduced (reference: +199).
    LMR_PARENT_RED_MUL = 50, -256, 512, 24;
    /// Extra reduction for captures the picker already ranked as losing (T35;
    /// the reference gives noisy moves a flat base, 2896 against 188).
    LMR_BADCAP_EXTRA = 163, -1024, 4096, 128;

    /// Reduce more for every time alpha was raised at this node (1/1024 ply
    /// per raise): once the node has found improvements, the rest is late.
    LMR_ALPHA_RAISE_MUL = 194, 0, 2048, 64;

    // >>> generated: factorized LMR (tools/scripts/gen_factorized_lmr.py)
    // One weight per canonical subset of the context booleans, in 1/1024
    // ply; positive reduces more. Seeded with the reference values.
    LMR_F_BASE = 141, -3072, 3072, 128;
    LMR_F_PV = -161, -3072, 3072, 128;
    LMR_F_CUTNODE = 2576, -3072, 4096, 128;
    LMR_F_IMPROVING = -242, -3072, 3072, 128;
    LMR_F_TTMOVE = -95, -3072, 3072, 128;
    LMR_F_TTPV = -257, -3072, 3072, 128;
    LMR_F_QUIET = -112, -3072, 3072, 128;
    LMR_F_GIVESCHECK = -234, -3072, 3072, 128;
    LMR_F_ROOT = 267, -3072, 3072, 128;
    LMR_F_FH2 = -64, -3072, 3072, 128;
    LMR_F_PV_IMPROVING = -311, -3072, 3072, 128;
    LMR_F_PV_TTMOVE = -105, -3072, 3072, 128;
    LMR_F_PV_QUIET = -214, -3072, 3072, 128;
    LMR_F_PV_GIVESCHECK = 70, -3072, 3072, 128;
    LMR_F_PV_FH2 = -15, -3072, 3072, 128;
    LMR_F_CUTNODE_IMPROVING = -12, -3072, 3072, 128;
    LMR_F_CUTNODE_TTMOVE = -1196, -3072, 3072, 128;
    LMR_F_CUTNODE_TTPV = -2244, -3072, 3072, 128;
    LMR_F_CUTNODE_QUIET = -596, -3072, 3072, 128;
    LMR_F_CUTNODE_GIVESCHECK = -119, -3072, 3072, 128;
    LMR_F_CUTNODE_FH2 = 131, -3072, 3072, 128;
    LMR_F_IMPROVING_TTMOVE = -113, -3072, 3072, 128;
    LMR_F_IMPROVING_TTPV = -36, -3072, 3072, 128;
    LMR_F_IMPROVING_QUIET = 33, -3072, 3072, 128;
    LMR_F_IMPROVING_GIVESCHECK = -305, -3072, 3072, 128;
    LMR_F_IMPROVING_FH2 = -118, -3072, 3072, 128;
    LMR_F_TTMOVE_TTPV = 147, -3072, 3072, 128;
    LMR_F_TTMOVE_QUIET = 113, -3072, 3072, 128;
    LMR_F_TTMOVE_GIVESCHECK = 31, -3072, 3072, 128;
    LMR_F_TTMOVE_ROOT = -183, -3072, 3072, 128;
    LMR_F_TTMOVE_FH2 = 8, -3072, 3072, 128;
    LMR_F_TTPV_QUIET = 133, -3072, 3072, 128;
    LMR_F_TTPV_GIVESCHECK = -154, -3072, 3072, 128;
    LMR_F_TTPV_FH2 = 33, -3072, 3072, 128;
    LMR_F_QUIET_GIVESCHECK = 181, -3072, 3072, 128;
    LMR_F_QUIET_ROOT = 235, -3072, 3072, 128;
    LMR_F_QUIET_FH2 = -10, -3072, 3072, 128;
    LMR_F_GIVESCHECK_ROOT = 51, -3072, 3072, 128;
    LMR_F_GIVESCHECK_FH2 = 239, -3072, 3072, 128;
    LMR_F_ROOT_FH2 = 540, -3072, 3072, 128;
    LMR_F_PV_IMPROVING_TTMOVE = -131, -3072, 3072, 128;
    LMR_F_PV_IMPROVING_QUIET = 154, -3072, 3072, 128;
    LMR_F_PV_IMPROVING_GIVESCHECK = -359, -3072, 3072, 128;
    LMR_F_PV_IMPROVING_FH2 = -136, -3072, 3072, 128;
    LMR_F_PV_TTMOVE_QUIET = 70, -3072, 3072, 128;
    LMR_F_PV_TTMOVE_GIVESCHECK = 334, -3072, 3072, 128;
    LMR_F_PV_TTMOVE_FH2 = 83, -3072, 3072, 128;
    LMR_F_PV_QUIET_GIVESCHECK = -7, -3072, 3072, 128;
    LMR_F_PV_QUIET_FH2 = -245, -3072, 3072, 128;
    LMR_F_PV_GIVESCHECK_FH2 = -83, -3072, 3072, 128;
    LMR_F_CUTNODE_IMPROVING_TTMOVE = 1, -3072, 3072, 128;
    LMR_F_CUTNODE_IMPROVING_TTPV = -197, -3072, 3072, 128;
    LMR_F_CUTNODE_IMPROVING_QUIET = -20, -3072, 3072, 128;
    LMR_F_CUTNODE_IMPROVING_GIVESCHECK = -132, -3072, 3072, 128;
    LMR_F_CUTNODE_IMPROVING_FH2 = 30, -3072, 3072, 128;
    LMR_F_CUTNODE_TTMOVE_TTPV = 1158, -3072, 3072, 128;
    LMR_F_CUTNODE_TTMOVE_QUIET = 32, -3072, 3072, 128;
    LMR_F_CUTNODE_TTMOVE_GIVESCHECK = 166, -3072, 3072, 128;
    LMR_F_CUTNODE_TTMOVE_FH2 = -129, -3072, 3072, 128;
    LMR_F_CUTNODE_TTPV_QUIET = -8, -3072, 3072, 128;
    LMR_F_CUTNODE_TTPV_GIVESCHECK = 78, -3072, 3072, 128;
    LMR_F_CUTNODE_TTPV_FH2 = -412, -3072, 3072, 128;
    LMR_F_CUTNODE_QUIET_GIVESCHECK = -194, -3072, 3072, 128;
    LMR_F_CUTNODE_QUIET_FH2 = 66, -3072, 3072, 128;
    LMR_F_CUTNODE_GIVESCHECK_FH2 = 255, -3072, 3072, 128;
    LMR_F_IMPROVING_TTMOVE_TTPV = 194, -3072, 3072, 128;
    LMR_F_IMPROVING_TTMOVE_QUIET = -367, -3072, 3072, 128;
    LMR_F_IMPROVING_TTMOVE_GIVESCHECK = 99, -3072, 3072, 128;
    LMR_F_IMPROVING_TTMOVE_FH2 = -111, -3072, 3072, 128;
    LMR_F_IMPROVING_TTPV_QUIET = 130, -3072, 3072, 128;
    LMR_F_IMPROVING_TTPV_GIVESCHECK = -127, -3072, 3072, 128;
    LMR_F_IMPROVING_TTPV_FH2 = 113, -3072, 3072, 128;
    LMR_F_IMPROVING_QUIET_GIVESCHECK = -487, -3072, 3072, 128;
    LMR_F_IMPROVING_QUIET_FH2 = -58, -3072, 3072, 128;
    LMR_F_IMPROVING_GIVESCHECK_FH2 = 76, -3072, 3072, 128;
    LMR_F_TTMOVE_TTPV_QUIET = -129, -3072, 3072, 128;
    LMR_F_TTMOVE_TTPV_GIVESCHECK = -343, -3072, 3072, 128;
    LMR_F_TTMOVE_TTPV_FH2 = 137, -3072, 3072, 128;
    LMR_F_TTMOVE_QUIET_GIVESCHECK = -70, -3072, 3072, 128;
    LMR_F_TTMOVE_QUIET_ROOT = -42, -3072, 3072, 128;
    LMR_F_TTMOVE_QUIET_FH2 = 33, -3072, 3072, 128;
    LMR_F_TTMOVE_GIVESCHECK_ROOT = -380, -3072, 3072, 128;
    LMR_F_TTMOVE_GIVESCHECK_FH2 = 110, -3072, 3072, 128;
    LMR_F_TTMOVE_ROOT_FH2 = -127, -3072, 3072, 128;
    LMR_F_TTPV_QUIET_GIVESCHECK = 92, -3072, 3072, 128;
    LMR_F_TTPV_QUIET_FH2 = -199, -3072, 3072, 128;
    LMR_F_TTPV_GIVESCHECK_FH2 = -62, -3072, 3072, 128;
    LMR_F_QUIET_GIVESCHECK_ROOT = -344, -3072, 3072, 128;
    LMR_F_QUIET_GIVESCHECK_FH2 = -338, -3072, 3072, 128;
    LMR_F_QUIET_ROOT_FH2 = -142, -3072, 3072, 128;
    LMR_F_GIVESCHECK_ROOT_FH2 = 257, -3072, 3072, 128;
    // <<< generated: factorized LMR

    // === Adaptive re-search (3) ===
    // Two-tiered do-deeper re-search, plus reduced-depth shallower re-search.
    /// Deeper tier 1 base: `score > best_score + X`.
    DEEPER_BASE = 69, 10, 150, 5;
    /// Deeper tier 2 base: `score > best_score + X` → +2 depth for major tactical moves.
    DEEPER2_BASE = 960, 400, 1200, 50;
    /// Shallower base: `score < best_score + X + reduced_depth`.
    SHALLOWER_BASE = 22, 0, 60, 3;

    // === Qsearch pruning (3) ===
    /// Qsearch futility base: `stand_pat + X` (unified delta + futility).
    QS_FUTILITY_BASE = 344, 50, 500, 25;
    /// Qsearch SEE pruning threshold (centipawns, negative).
    QS_SEE_THRESHOLD = -68, -200, 0, 10;
    /// Divisor adjusting the qsearch SEE threshold by the move's history.
    /// (good history relaxes the SEE margin, bad history tightens it)
    QS_SEE_HIST_DIV = 68, 16, 128, 8;
    /// Non-PV qsearch move count limit.
    QS_MOVE_COUNT_LIMIT = 3, 2, 10, 1;

    // === IIR (1) ===
    /// IIR depth threshold: `depth >= X + 2 * cut_node`.
    IIR_DEPTH = 4, 1, 6, 1;

    // === History bonus/malus (6) ===
    /// History bonus: `X * depth + offset`.
    STAT_BONUS_MUL = 128, 50, 400, 15;
    /// History bonus offset: `mul * depth + X`.
    STAT_BONUS_ADD = -10, -50, 100, 8;
    /// History bonus max: `.min(X)`.
    STAT_BONUS_MAX = 1602, 500, 3000, 100;
    /// History malus: `X * depth - offset`.
    STAT_MALUS_MUL = 228, 50, 400, 15;
    /// History malus offset: `mul * depth - X`.
    STAT_MALUS_SUB = 65, -50, 100, 8;
    /// History malus max: `.min(X)`.
    STAT_MALUS_MAX = 635, 200, 2500, 100;

    // === Non-PV best-move history bonus scaled by move count ===
    /// Divisor scaling the bonus by the number of moves tried at non-PV nodes.
    /// Smaller = stronger effect, larger = weaker effect.
    BESTMOVE_MC_DIV = 330, 64, 1024, 32;

    // === Move ordering (6) ===
    /// MVV multiplier in capture scoring: `PIECE_VALUE[victim] * X`.
    MVV_MULTIPLIER = 12, 1, 64, 4;
    /// Good/bad capture split: base SEE threshold (centipawns, negative).
    GOOD_CAP_BASE = -137, -300, 0, 16;
    /// Good/bad capture split: capture history divisor shifting the threshold.
    GOOD_CAP_HIST_DIV = 25, 8, 96, 4;
    /// Movepicker bonus for quiet moves that give direct check.
    MOVEPICK_CHECK_BONUS = 11415, 4000, 20000, 1000;
    /// Learned look-ahead: TT child eval multiplier (numerator).
    LOOKAHEAD_BONUS_MUL = 12, 0, 32, 2;
    /// Learned look-ahead: TT child eval divisor (denominator).
    LOOKAHEAD_BONUS_DIV = 1, 1, 8, 1;
    /// Learned look-ahead: minimum depth to enable TT child probing.
    LOOKAHEAD_MIN_DEPTH = 4, 1, 12, 1;

    // === Lazy eval (1) ===
    /// PeSTO score threshold to skip NNUE forward pass (centipawns, absolute).
    LAZY_EVAL_THRESHOLD = 706, 200, 1500, 50;

    // === Optimism (4) ===
    /// Optimism sigmoid numerator.
    OPT_NUMERATOR = 118, 50, 300, 10;
    /// Optimism sigmoid offset.
    OPT_OFFSET = 210, 50, 400, 10;
    /// Material scale base for raw eval.
    OPT_MAT_SCALE = 852, 200, 2000, 50;
    /// Material base for optimism weight.
    OPT_MAT_BASE = 1718, 500, 4000, 200;

    // === Prior Countermove Bonus on Unexpected All-Nodes (2) ===
    // Bonus to the parent's quiet move when this node fails low
    // unexpectedly (predicted cut or PV but the result is an all-node).
    /// Butterfly history factor: PCM_ALLNODE_FACTOR * stat_bonus(depth) / 128.
    PCM_ALLNODE_FACTOR = 166, 50, 400, 16;
    /// Continuation history factor: PCM_ALLNODE_CONTHIST_FACTOR * stat_bonus(depth) / 128.
    PCM_ALLNODE_CONTHIST_FACTOR = 87, 30, 300, 16;

    // === Prior Capture Move Bonus on Unexpected All-Nodes ===
    // Capture history bonus for the parent's capture move at unexpected all-nodes.
    /// Capture history factor: PCM_ALLNODE_CAPTURE_FACTOR * stat_bonus(depth) / 128.
    PCM_ALLNODE_CAPTURE_FACTOR = 68, 16, 200, 12;
}

// === Move Overhead — standalone, exposed as UCI option ===
static MOVE_OVERHEAD_VALUE: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(40);

/// Overhead subtracted from available time to prevent flagging (ms).
///
/// Paid on every move of the horizon, twice over, so the budget is far more
/// sensitive to it than the per-move figure suggests: at blitz, going from 10 ms
/// to 40 costs about a quarter of the budget. 40 leaves room for the dozens of
/// games a match runner packs onto one machine, where a descheduled process can
/// lose hundreds of milliseconds; a host driving a single engine on a quiet
/// machine can lower it.
#[allow(non_snake_case)]
#[inline(always)]
pub fn MOVE_OVERHEAD() -> i32 {
    MOVE_OVERHEAD_VALUE.load(std::sync::atomic::Ordering::Relaxed)
}

/// Set move overhead (called from UCI `setoption name Move Overhead`).
pub fn set_move_overhead(value: i32) {
    MOVE_OVERHEAD_VALUE.store(value.clamp(0, 5000), std::sync::atomic::Ordering::Relaxed);
}

// Empty on purpose. The time-management scalars that used to live here are now
// the reals of `timeman.rs`, tuned as a set rather than one at a time. The
// invocation stays because it is what defines `set_param_fixed`, and because the
// next parameter that must not reach a tuner belongs here.
fixed! {}

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
        assert_eq!(ASP_INITIAL_DELTA(), 14);
        assert_eq!(RFP_MARGIN(), 88);
        assert_eq!(SEE_QUIET_MARGIN(), -8);
        assert_eq!(LMR_LOG_MUL(), 630);
        assert_eq!(NMP_BASE_R(), 4);
        assert_eq!(STAT_BONUS_MUL(), 128);
        assert_eq!(MVV_MULTIPLIER(), 12);
        assert_eq!(BNFP_BASE(), 149);
        assert_eq!(BNFP_DEPTH_LIMIT(), 9338);
        assert_eq!(QS_SEE_THRESHOLD(), -68);
    }

    #[test]
    fn test_fixed_defaults() {
        assert_eq!(MOVE_OVERHEAD(), 40);
    }

    /// Names declared in the `tunable!` and `fixed!` blocks of this file.
    fn declared_parameters() -> Vec<String> {
        let text = include_str!("tune.rs");
        let mut names = Vec::new();
        let mut inside = false;
        for line in text.lines() {
            if line.starts_with("tunable! {") || line.starts_with("fixed! {") {
                // `fixed! {}` opens and closes on the same line.
                inside = !line.trim_end().ends_with('}');
                continue;
            }
            if !inside {
                continue;
            }
            if line == "}" {
                inside = false;
                continue;
            }
            let line = line.trim();
            if line.starts_with("//") || line.starts_with('#') {
                continue;
            }
            if let Some((name, _)) = line.split_once('=') {
                let name = name.trim();
                if !name.is_empty()
                    && name
                        .bytes()
                        .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_')
                {
                    names.push(name.to_string());
                }
            }
        }
        names
    }

    /// Every `.rs` file under `dir`, recursively.
    fn rust_sources(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                rust_sources(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }

    /// Whether `code` calls the getter `name()`: the name as a whole word, followed
    /// by an opening parenthesis.
    fn calls(code: &str, name: &str) -> bool {
        let bytes = code.as_bytes();
        let is_word = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
        code.match_indices(name).any(|(i, _)| {
            let before = i.checked_sub(1).map(|j| bytes[j]);
            let after = code[i + name.len()..].trim_start().as_bytes().first().copied();
            !before.is_some_and(is_word) && after == Some(b'(')
        })
    }

    /// A parameter nothing reads is a knob SPSA turns for nothing: it spends the
    /// tuner's budget and reports a value that changes no search. Every declared
    /// parameter must be called somewhere outside this file (comments excluded).
    #[test]
    fn every_parameter_is_read_by_the_engine() {
        let names = declared_parameters();
        assert!(names.len() > 80, "the parameter list was not parsed: {names:?}");

        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files = Vec::new();
        rust_sources(&src, &mut files);
        let code: String = files
            .iter()
            .filter(|p| p.file_name().is_none_or(|n| n != "tune.rs"))
            .map(|p| {
                std::fs::read_to_string(p)
                    .unwrap()
                    .lines()
                    .map(|l| l.split_once("//").map_or(l, |(code, _)| code))
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .collect::<Vec<_>>()
            .join("\n");

        let unused: Vec<&String> = names.iter().filter(|n| !calls(&code, n)).collect();
        assert!(
            unused.is_empty(),
            "tunable parameters never read outside tune.rs: {unused:?}"
        );
    }

    #[cfg(feature = "spsa")]
    #[test]
    fn test_set_param() {
        // Verify runtime mutation works for tunable params.
        let old = RFP_MARGIN();
        assert!(set_param("RFP_MARGIN", 100));
        assert_eq!(RFP_MARGIN(), 88);
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
        // One line per tunable, and the same set as the JSON export. Counting them
        // against a number written here by hand is what this used to do, and the number
        // was left behind at 90 while the list grew past a hundred: a test that has to
        // be edited every time a technique lands is a test nobody edits.
        assert_eq!(
            csv.lines().count(),
            emit_json().matches("\"min_value\"").count()
        );
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
    }
}
