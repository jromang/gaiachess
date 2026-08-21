//! [Alpha-beta search](https://www.chessprogramming.org/Alpha-Beta) with
//! [iterative deepening](https://www.chessprogramming.org/Iterative_Deepening)
//! and [quiescence search](https://www.chessprogramming.org/Quiescence_Search).
//!
//! PVS with aspiration windows, null move pruning, killer moves, history heuristic,
//! capture/continuation history, countermove heuristic, singular extensions, ProbCut,
//! NNUE evaluation with correction history. Lazy SMP parallel search.

use crate::eval;
use crate::nnue;
use crate::history::{
    ContCorrectionHistory, ContinuationHistory, PieceToTable, stat_bonus, stat_malus,
    CORRHIST_LIMIT,
};
use crate::movegen;
use crate::movepick::{MovePicker, get_captured_pt};
use crate::position::Position;
use crate::skill;
use crate::stats::st;
use crate::threads::{SharedState, ThreadData, SS_OFFSET, PONDER};
use crate::tree::{tr, tree_ret};
use crate::tt::Bound;
use crate::tune;
use crate::types::*;

/// Compile-time node type dispatch via marker types.
/// Eliminates runtime `ply == 0` checks and allows the compiler to
/// specialize search for root vs interior nodes.
trait NodeType {
    const PV: bool;
    const ROOT: bool;
}

/// Root node: PV=true, ROOT=true. Entry point from iterative deepening.
struct Root;
impl NodeType for Root {
    const PV: bool = true;
    const ROOT: bool = true;
}

/// PV node: PV=true, ROOT=false. Interior nodes on the principal variation.
struct Pv;
impl NodeType for Pv {
    const PV: bool = true;
    const ROOT: bool = false;
}

/// Non-PV node: PV=false, ROOT=false. Null-window searches, NMP, LMR, SE.
struct NonPv;
impl NodeType for NonPv {
    const PV: bool = false;
    const ROOT: bool = false;
}

/// Maximum quiet moves tracked for history malus.
const MAX_QUIETS_SEARCHED: usize = 64;

/// Maximum capture moves tracked for history malus.
const MAX_CAPTURES_SEARCHED: usize = 32;

/// Correction history divisor (weighted sum / 65536).
/// Structural power-of-2 constant, not tuned.
const CORR_DIVISOR: i64 = 65536;

/// Log table for the LMR log-log term: trunc(20.13 * ln(i)).
/// Indexed by both move_count and depth (clamped to 63). A true logarithm is
/// smooth and monotone in move_count, unlike the ilog2 product it replaces,
/// which was a non-monotone sawtooth within each power-of-two bracket.
/// Index 0 is unused (LMR requires move_count >= 2 and depth >= 2).
const LMR_LN: [i32; 64] = [
    0, 0, 13, 22, 27, 32, 36, 39, 41, 44, 46, 48, 50, 51, 53, 54,
    55, 57, 58, 59, 60, 61, 62, 63, 63, 64, 65, 66, 67, 67, 68, 69,
    69, 70, 70, 71, 72, 72, 73, 73, 74, 74, 75, 75, 76, 76, 77, 77,
    77, 78, 78, 79, 79, 79, 80, 80, 81, 81, 81, 82, 82, 82, 83, 83,
];

/// A move is quiet if it is not a capture, not a promotion, and not en passant.
#[inline(always)]
fn is_quiet(pos: &Position, m: Move) -> bool {
    let mt = m.move_type();
    mt != MT_PROMOTION && mt != MT_EN_PASSANT && pos.board[m.to_sq().index()] == Piece::NONE
}

/// Whether a move is one of the ones that are hard to miss.
///
/// Only consulted by weakened levels, to decide how likely they are to see a move at all
/// (see [`crate::skill::sees_move`]). Anything that takes a piece — a recapture included,
/// since that is a capture too — anything that gives check, anything right in front of
/// the player at the top of the tree, and carrying on with the piece just moved: those
/// are the moves a beginner does look at. The quiet retreat four plies deep is the one
/// they never find. Getting this list right is what separates an opponent who plays
/// weakly from one that ignores a queen standing en prise.
#[inline]
fn easy_to_notice(td: &ThreadData, m: Move, ply: usize) -> bool {
    // Taking something, promoting, and everything at the very top of the tree.
    if ply < 2 || !is_quiet(&td.pos, m) {
        return true;
    }
    // The two quiet moves that still come to mind: carrying on with the piece moved a
    // move ago, and giving check.
    let own_previous = td.ss(ply - 2).played_move;
    (own_previous != Move::NONE && own_previous.to_sq() == m.from_sq()) || td.pos.gives_check(m)
}

/// Evaluate the position using NNUE (if a trained network is loaded) or PeSTO fallback.
/// Returns 0 immediately for trivially drawn pawnless endgames (insufficient material).
///
/// Lazy eval: when NNUE is loaded and the incremental PeSTO score exceeds
/// `LAZY_EVAL_THRESHOLD`, skip the expensive NNUE forward pass and return
/// PeSTO directly. Sets `td.used_lazy_eval` so callers can guard correction
/// history (which is trained against NNUE, not PeSTO).
#[inline]
fn evaluate_pos(td: &mut ThreadData) -> i32 {
    // Insufficient material: immediate draw for trivially drawn pawnless endgames
    if td.pos.pieces[Piece::WHITE_PAWN.index()] == 0
        && td.pos.pieces[Piece::BLACK_PAWN.index()] == 0
        && eval::is_material_draw(&td.pos)
    {
        td.used_lazy_eval = false;
        return 0;
    }
    st!(td, s, s.eval_calls += 1;);
    if td.skill.active {
        return handicapped_eval(td);
    }
    if nnue::network::has_network() {
        let pesto = td.pos.lazy_eval();
        if pesto.abs() > tune::LAZY_EVAL_THRESHOLD() {
            st!(td, s, s.lazy_evals += 1;);
            td.used_lazy_eval = true;
            pesto
        } else {
            td.used_lazy_eval = false;
            td.nnue.evaluate(&td.pos)
        }
    } else {
        td.used_lazy_eval = false;
        td.pos.lazy_eval()
    }
}

/// What a weakened opponent thinks the position is worth.
///
/// Two things happen here that do not happen at full strength. The engine's judgement is
/// cut down to what the level is allowed to have — material only at the bottom, then
/// piece squares, then the network — and the result is then pushed off the truth by an
/// amount the position itself decides. Both are needed: a player who judges accurately
/// and then guesses is erratic, and one who judges crudely but consistently is merely a
/// weaker engine. Together they misunderstand the position and stay wrong about it.
///
/// The lazy-eval gate is deliberately not used here. It exists to save a forward pass
/// when the score is already decisive, and these levels are capped at a handful of plies
/// where nothing needs saving; skipping it keeps the blend reading one consistent
/// judgement rather than two different ones from position to position.
#[cold]
#[inline(never)]
fn handicapped_eval(td: &mut ThreadData) -> i32 {
    let fidelity = td.skill.rung.eval_fidelity;
    debug_assert!((skill::FIDELITY_MATERIAL..=skill::FIDELITY_NNUE).contains(&fidelity));

    // How far apart two neighbouring judgements sit on the fidelity scale.
    const SPAN: i32 = skill::FIDELITY_PESTO - skill::FIDELITY_MATERIAL;
    let blend = |coarse: i32, fine: i32, weight: i32| {
        debug_assert!((0..=SPAN).contains(&weight));
        (coarse * (SPAN - weight) + fine * weight) / SPAN
    };

    let judgement = if fidelity <= skill::FIDELITY_MATERIAL {
        eval::material_eval(&td.pos)
    } else if fidelity < skill::FIDELITY_PESTO {
        blend(eval::material_eval(&td.pos), td.pos.lazy_eval(), fidelity - skill::FIDELITY_MATERIAL)
    } else if fidelity == skill::FIDELITY_PESTO || !nnue::network::has_network() {
        td.pos.lazy_eval()
    } else {
        let pesto = td.pos.lazy_eval();
        blend(pesto, td.nnue.evaluate(&td.pos), fidelity - skill::FIDELITY_PESTO)
    };

    // Correction history is trained against a clean network eval, so a handicapped
    // search must neither consult it nor feed it. This is the flag the callers already
    // use to say exactly that.
    td.used_lazy_eval = true;
    skill::noise(judgement, td.pos.key, &td.skill)
}

/// Update continuation history at offsets [1, 2, 4, 6] from the current ply.
/// Skip distant conthist updates when in check: only offsets 1 and 2 are
/// updated while in check, avoiding noise in the distant continuation
/// history tables (offsets 4 and 6).
fn update_continuation_histories(
    td: &mut ThreadData,
    ply: usize,
    piece: Piece,
    to: Square,
    bonus: i32,
    in_check: bool,
) {
    for offset in [1, 2, 4, 6] {
        // While in check, skip writes to the distant continuation histories (ss-4, ss-6)
        if in_check && offset > 2 {
            break;
        }
        let idx = ply + SS_OFFSET;
        if idx >= offset {
            let entry = &td.stack[idx - offset];
            if entry.played_move != Move::NONE {
                let ch_bonus = if offset >= 4 { bonus / 2 } else { bonus };
                ContinuationHistory::update(entry.conthist_ptr, piece, to, ch_bonus);
            }
        }
    }
}

/// Format a score as UCI string ("cp N", "mate N", or clamped TB score).
fn format_score(score: i32) -> String {
    if is_mate_score(score) {
        let mate_ply = if score > 0 {
            (SCORE_MATE - score + 1) / 2
        } else {
            -(SCORE_MATE + score + 1) / 2
        };
        format!("mate {mate_ply}")
    } else if is_tb_score(score) {
        // TB win/loss: display as large cp value (UCI convention)
        let tb_cp = if score > 0 { 20000 } else { -20000 };
        format!("cp {tb_cp}")
    } else {
        format!("cp {score}")
    }
}

/// Extend the PV of a decisive TB root move by successive DTZ probes.
///
/// After normal search, the PV typically contains only 1-2 moves because 50-move
/// scaling crushes eval deep in the tree. This function builds the full PV to mate
/// by replaying DTZ probes position-by-position, choosing the move with best DTZ rank.
/// Tie-breaks minimize opponent mobility.
///
/// The position is left unchanged (all moves are undone at the end).
#[cfg(feature = "syzygy")]
fn syzygy_extend_pv(pos: &mut Position, rm: &mut crate::threads::RootMove) {
    // Step 0: play the root move
    pos.make_move(rm.pv[0]);
    let mut ply = 1;

    // Step 1: validate existing PV moves (keep only TB-optimal ones)
    while ply < rm.pv_len {
        let pv_move = rm.pv[ply];
        if let Some((ranked, _)) = crate::tb::rank_root_moves(pos) {
            if let Some(&(_, best_rank, _)) = ranked.first() {
                let pv_rank = ranked.iter().find(|(m, _, _)| *m == pv_move).map(|(_, r, _)| *r);
                if pv_rank != Some(best_rank) {
                    break;
                }
            } else {
                break;
            }
        } else {
            break;
        }
        pos.make_move(pv_move);
        ply += 1;
        if pos.draw_by_fifty_move_rule() || pos.draw_by_repetition(ply as i32) {
            pos.unmake_move(pv_move);
            ply -= 1;
            break;
        }
    }
    rm.pv_len = ply;

    // Step 2: extend PV via successive DTZ probes
    loop {
        if pos.draw_by_fifty_move_rule() {
            break;
        }
        let Some((ranked, _dtz)) = crate::tb::rank_root_moves(pos) else { break };
        if ranked.is_empty() {
            break;
        }
        let mut legal_buf = ArrayBuf::<Move, MAX_MOVES>::new();
        if movegen::generate_legal_moves(pos, &mut legal_buf) == 0 {
            break;
        }

        // Pick best move with tie-breaking by opponent mobility
        let best_rank = ranked[0].1;
        let best_moves: Vec<_> = ranked.iter()
            .take_while(|(_, r, _)| *r == best_rank)
            .collect();

        let best_move = if best_moves.len() == 1 {
            best_moves[0].0
        } else {
            let mut best = best_moves[0].0;
            let mut best_tiebreak = i32::MIN;
            for &&(mv, _, _) in &best_moves {
                pos.make_move(mv);
                let mut opp_buf = ArrayBuf::<Move, MAX_MOVES>::new();
                let opp_count = movegen::generate_legal_moves(pos, &mut opp_buf);
                let mut score = -(opp_count as i32);
                for i in 0..opp_count {
                    if pos.board[opp_buf[i].to_sq().index()] != Piece::NONE {
                        score -= 100;
                    }
                }
                pos.unmake_move(mv);
                if score > best_tiebreak {
                    best_tiebreak = score;
                    best = mv;
                }
            }
            best
        };

        if ply >= MAX_PLY {
            break;
        }
        rm.pv[ply] = best_move;
        ply += 1;
        rm.pv_len = ply;
        pos.make_move(best_move);

        if pos.draw_by_fifty_move_rule() || pos.draw_by_repetition(ply as i32) {
            pos.unmake_move(best_move);
            ply -= 1;
            rm.pv_len = ply;
            break;
        }
    }

    // Undo all moves (position unchanged)
    for i in (0..ply).rev() {
        pos.unmake_move(rm.pv[i]);
    }
}

/// Print UCI info lines for all PV lines at the current depth.
fn print_multi_pv_info(td: &ThreadData, shared: &SharedState, depth: i32, multi_pv: usize) {
    let elapsed = td.tm.elapsed_ms().max(1);
    let nps = td.nodes * 1000 / elapsed;
    let hashfull = shared.tt.hashfull();

    for (i, rm) in td.root_moves[..multi_pv].iter().enumerate() {
        // Use previous_score if current depth was interrupted before this PV was searched.
        // When root_in_tb, the DTZ-best move might not be the search-best move, so
        // fall back to tb_score for display.
        let search_score = if rm.score != -SCORE_INFINITE {
            Some((rm.score, depth))
        } else if rm.previous_score != -SCORE_INFINITE {
            Some((rm.previous_score, (depth - 1).max(1)))
        } else {
            None
        };

        // TB score override: prefer TB score over search score
        #[cfg(feature = "syzygy")]
        let use_tb = td.tb_config.root_in_tb && rm.tb_score != 0;
        #[cfg(not(feature = "syzygy"))]
        let use_tb = false;

        let (score, d) = if let Some((s, d)) = search_score {
            if use_tb && !is_mate_score(s) { (rm.tb_score, d) } else { (s, d) }
        } else if use_tb {
            (rm.tb_score, depth)
        } else {
            continue;
        };

        let score_str = format_score(score);

        let mut pv_str = String::new();
        for j in 0..rm.pv_len {
            if j > 0 { pv_str.push(' '); }
            pv_str.push_str(&rm.pv[j].to_uci());
        }

        // Only include "multipv N" when multi_pv > 1 (some GUIs don't expect it for single PV)
        let multipv_str = if multi_pv > 1 {
            format!(" multipv {}", i + 1)
        } else {
            String::new()
        };

        out!(
            "info depth {d} seldepth {}{multipv_str} score {score_str} nodes {} nps {nps} time {elapsed} hashfull {hashfull} tbhits {} pv {pv_str}",
            rm.sel_depth, td.nodes, td.tb_hits,
        );
    }
}

/// Run iterative deepening and return the best move found.
/// Prints UCI `info` lines after each depth (main thread only).
/// Supports Multi-PV: searches td.multi_pv principal variations per depth.
pub fn search(td: &mut ThreadData, shared: &SharedState) {
    td.nodes = 0;
    td.qs_nodes = 0;
    // Note: killers and stack are already cleared in prepare_search

    let is_main = td.id == 0;
    // Owning the clock and reporting to stdout are two different jobs: an interface
    // driving the engine in-process wants the first without the second.
    let reports = is_main && !td.silent;

    // Nothing to search: the game is already over on the board. A well-behaved interface
    // never asks, having seen the mate itself, but one that does must get an answer
    // rather than a dead engine — and the iterative deepening below assumes throughout
    // that there is a root move to talk about.
    if td.root_moves.is_empty() {
        td.best_move = Move::NONE;
        td.best_score = if td.pos.checkers != 0 { -SCORE_MATE } else { 0 };
        td.completed_depth = 0;
        if reports {
            let score = if td.pos.checkers != 0 { "mate 0" } else { "cp 0" };
            out!("info depth 0 score {score}");
        }
        return;
    }


    let max_depth = td.tm.max_depth().min(MAX_PLY as i32);

    // Syzygy root move ranking.
    // Tries DTZ first (precise, 50-move aware), then WDL fallback.
    // Sets tb_rank/tb_score on each root move, sorts by DTZ rank.
    // When DTZ is available, cardinality=0 → no in-tree probing needed.
    #[cfg(feature = "syzygy")]
    if !td.datagen_mode
        && let Some((ranked, dtz_available)) = crate::tb::rank_root_moves(&td.pos)
    {
        td.tb_config.root_in_tb = true;
        td.tb_hits += ranked.len() as u64;
        for (mv, rank, score) in &ranked {
            if let Some(rm) = td.root_moves.iter_mut().find(|r| r.mv == *mv) {
                rm.tb_rank = *rank;
                rm.tb_score = *score;
            }
        }
        // Sort by DTZ rank descending (best move first)
        td.root_moves.sort_by(|a, b| b.tb_rank.cmp(&a.tb_rank));
        // Two reasons not to probe in the tree: DTZ has already ranked the root,
        // or we are losing anyway and probing cannot change that. Otherwise probe
        // at max_pieces cardinality.
        let losing = ranked.first().is_some_and(|(_, _, s)| *s <= 0);
        td.tb_config.cardinality = if dtz_available || losing {
            0
        } else {
            crate::tb::max_pieces()
        };
    }

    // Follow-PV guard: reset the PV recorded from the previous iteration
    td.last_iteration_pv_len = 0;

    'id: for depth in 1..=max_depth {
        td.root_depth = depth;

        // Multi-PV: full root search for each requested principal variation.
        // Each PV line gets its own aspiration window centered on its running average.
        // After each PV, stable sort brings the winner to root_moves[pv_idx].
        //
        // Reference: CPW — Principal Variation § Multiple PVs
        let multi_pv = td.multi_pv.min(td.root_moves.len());

        // TB group end: only search moves in the same WDL group as the best move.
        // When root_in_tb, root_moves are sorted by tb_rank descending. We limit
        // the search to moves with the same signum (win/draw/loss) as the best move.
        // This prevents the engine from considering losing moves when a win exists.
        #[cfg(feature = "syzygy")]
        let tb_group_end = if td.tb_config.root_in_tb && !td.root_moves.is_empty() {
            let best_sign = td.root_moves[0].tb_rank.signum();
            td.root_moves.iter().position(|rm| rm.tb_rank.signum() != best_sign)
                .unwrap_or(td.root_moves.len())
        } else {
            td.root_moves.len()
        };
        #[cfg(not(feature = "syzygy"))]
        let tb_group_end = td.root_moves.len();

        for pv_idx in 0..multi_pv {
            td.pv_index = pv_idx;
            td.seldepth = 0;

            // Aspiration window centered on this PV's running average
            let prev_avg = td.root_moves[pv_idx].avg_score;
            let mut alpha = -SCORE_INFINITE;
            let mut beta = SCORE_INFINITE;
            let mut delta = tune::ASP_INITIAL_DELTA();
            debug_assert!(delta > 0, "aspiration: initial delta {} <= 0", delta);

            if depth >= tune::ASP_MIN_DEPTH() && prev_avg != SCORE_NONE {
                delta += prev_avg.abs() / tune::ASP_SCORE_DIV();
                alpha = (prev_avg - delta).max(-SCORE_INFINITE);
                beta = (prev_avg + delta).min(SCORE_INFINITE);
            }

            // Optimism: bias eval toward root's running average
            if prev_avg != SCORE_NONE {
                let stm = td.pos.side_to_move;
                td.optimism[stm.index()] = tune::OPT_NUMERATOR() * prev_avg
                    / (prev_avg.abs() + tune::OPT_OFFSET());
                td.optimism[stm.flip().index()] = -td.optimism[stm.index()];
            }

            // Aspiration retry loop
            let score;
            let mut failed_high_cnt = 0i32;
            loop {
                // Reduce depth on repeated fail-highs
                let adjusted_depth = (depth - failed_high_cnt).max(1);
                td.root_delta = (beta - alpha).max(1);
                let s = alpha_beta::<Root>(td, shared, alpha, beta, adjusted_depth, 0, false, false);

                if td.should_stop() {
                    score = s;
                    break;
                }

                if s <= alpha {
                    st!(td, st_, st_.asp_fail_low += 1;);
                    beta = (alpha + beta) / 2;
                    alpha = (s - delta).max(-SCORE_INFINITE);
                    failed_high_cnt = 0;
                } else if s >= beta {
                    st!(td, st_, st_.asp_fail_high += 1;);
                    beta = (s + delta).min(SCORE_INFINITE);
                    failed_high_cnt += 1;
                } else {
                    score = s;
                    break;
                }

                delta += delta / tune::ASP_WIDEN_DIV();
                debug_assert!(delta > 0, "aspiration: delta {} <= 0 after widening", delta);

                if delta >= tune::ASP_FALLBACK_DELTA() {
                    alpha = -SCORE_INFINITE;
                    beta = SCORE_INFINITE;
                }
            }

            // Abort if stopped mid-search
            if td.should_stop() && td.best_move != Move::NONE {
                break 'id;
            }

            // Find the root move that alpha_beta selected as best (td.pv[0][0])
            // and update its score, PV, and running average. Other root moves keep
            // score = -SCORE_INFINITE. Stable sort then brings the winner to root_moves[pv_idx].
            let pv_len = td.pv_len[0];
            if pv_len > 0 {
                let best_mv = td.pv[0][0];
                if let Some(rm) = td.root_moves[pv_idx..].iter_mut().find(|rm| rm.mv == best_mv) {
                    rm.score = score;
                    rm.pv[..pv_len].copy_from_slice(&td.pv[0][..pv_len]);
                    rm.pv_len = pv_len;
                    rm.sel_depth = td.seldepth;
                    // Update running average on the winning move
                    rm.avg_score = if rm.avg_score == SCORE_NONE {
                        score
                    } else {
                        (rm.avg_score + score) / 2
                    };
                }
            }

            // Sort root_moves[pv_idx..sort_end] by score descending.
            // Only the winner has a real score; all others are -SCORE_INFINITE.
            // Stable sort preserves order among ties.
            //
            // When root_in_tb, limit sort to moves with the SAME tb_rank as the
            // current PV move. This lets the
            // search pick the best move among DTZ-equivalent options while
            // preserving the DTZ ordering across different ranks.
            #[cfg(feature = "syzygy")]
            let sort_end = if td.tb_config.root_in_tb && pv_idx < td.root_moves.len() {
                let rank = td.root_moves[pv_idx].tb_rank;
                td.root_moves[pv_idx..tb_group_end].iter()
                    .position(|rm| rm.tb_rank != rank)
                    .map_or(tb_group_end, |off| pv_idx + off)
            } else {
                tb_group_end
            };
            #[cfg(not(feature = "syzygy"))]
            let sort_end = tb_group_end;

            td.root_moves[pv_idx..sort_end].sort_by(|a, b| b.score.cmp(&a.score));
        }

        // Abort if stopped mid-search
        if td.should_stop() && td.best_move != Move::NONE {
            break;
        }

        // Best move from top root move
        if !td.root_moves.is_empty() {
            td.best_move = td.root_moves[0].mv;
            td.best_score = td.root_moves[0].score;

            // Follow-PV guard: save the PV for the next iteration
            td.last_iteration_pv_len = td.root_moves[0].pv_len;
            debug_assert!(td.last_iteration_pv_len <= MAX_PLY + 1);
            td.last_iteration_pv[..td.last_iteration_pv_len]
                .copy_from_slice(&td.root_moves[0].pv[..td.last_iteration_pv_len]);
        }

        // Best move + score stability tracking for time management.
        // Adjusts the soft limit so the engine thinks longer when the best move
        // is unstable or the score is dropping, and shorter when both are stable.
        //
        // Reference: CPW — Search Progression § Soft bound (best move stability, eval stability)
        if td.best_move == td.prev_best_move {
            td.bm_stability = (td.bm_stability + 1).min(tune::TM_BM_MAX());
        } else {
            td.bm_stability = 0;
        }
        td.prev_best_move = td.best_move;

        if is_main && !td.pondering {
            let bm_mul = (tune::TM_BM_BASE() - td.bm_stability * tune::TM_BM_FACTOR()) as f64
                / 1000.0;
            let prev_score = td.root_moves[0].previous_score;
            let score_diff = if prev_score != SCORE_NONE && prev_score != -SCORE_INFINITE {
                (prev_score - td.best_score)
                    .clamp(tune::TM_SCORE_MIN(), tune::TM_SCORE_MAX())
            } else {
                0
            };
            let score_mul = (tune::TM_SCORE_BASE() as f64
                + score_diff as f64 * tune::TM_SCORE_FACTOR() as f64)
                / 1000.0;

            // Node fraction TM: if most nodes go to the best move, stop early;
            // if nodes are spread across many moves, spend more time.
            //
            // Node fraction: if most nodes go to the best move, stop early.
            let nodes_mul = if depth > 7 && td.nodes > 0 {
                let bm_nodes = td.root_moves[0].nodes_spent as f64;
                let fraction = bm_nodes / td.nodes as f64;
                (tune::TM_NODES_BASE() as f64
                    - tune::TM_NODES_FACTOR() as f64 * fraction)
                    / 1000.0
            } else {
                1.0
            };
            td.tm.adjust_soft_limit(bm_mul * score_mul * nodes_mul);
        }

        // Save previous_score for next iteration's aspiration centering, then reset scores
        for rm in td.root_moves.iter_mut() {
            rm.previous_score = rm.score;
            rm.score = -SCORE_INFINITE;
        }

        td.completed_depth = depth;

        // Submit PV positions for online TB probing (main thread only).
        // Positions along the PV are the most likely to be re-searched at deeper
        // depths in the next iteration and to occur in actual play.
        #[cfg(feature = "online-tb")]
        if is_main && crate::online_tb::is_enabled()
            && !td.root_moves.is_empty()
            && td.root_moves[0].pv_len > 0
        {
            crate::online_tb::submit_pv_positions(
                &td.pos, &td.root_moves[0].pv, td.root_moves[0].pv_len,
            );
        }

        // Print UCI info (reporting thread only)
        if reports {
            // Only as many lines as were asked for: a weakened level searches extra ones
            // for its own use, and an interface that asked for one wants one.
            let multi_pv = td.multi_pv.min(td.reported_multi_pv).min(td.root_moves.len());

            // Extend PV for decisive TB root moves via successive DTZ probes.
            // Builds the full path to mate by successive DTZ probes.
            #[cfg(feature = "syzygy")]
            if td.tb_config.root_in_tb {
                for i in 0..multi_pv {
                    // Use tb_score to decide if PV extension is needed (search score
                    // may be missing if DTZ-best move differs from search-best move)
                    if td.root_moves[i].tb_score > 0 {
                        syzygy_extend_pv(&mut td.pos, &mut td.root_moves[i]);
                    }
                }
            }

            print_multi_pv_info(td, shared, depth, multi_pv);
        }

        // Detect ponderhit transition between iterations
        if td.pondering && !PONDER.load(std::sync::atomic::Ordering::Relaxed) {
            td.pondering = false;
            td.tm.restart();
        }

        // Soft time check after each iteration (main thread only; helpers have infinite limits).
        // Skip while pondering — search indefinitely until ponderhit or stop.
        if !td.pondering && td.tm.should_stop_soft() {
            break;
        }
    }

    apply_variety(td);
}

/// Lets a weakened level pick between the root moves it found worth playing.
///
/// Run once, after the search has finished, on the scores of the last completed
/// iteration — by then each iteration has moved its scores into `previous_score`. Nothing
/// about the search itself changes: the moves and their scores are exactly what they were,
/// and only which one is handed back differs. Full strength never reaches the body.
fn apply_variety(td: &mut ThreadData) {
    if !td.skill.active || td.skill.rung.variety_moves <= 1 || td.root_moves.is_empty() {
        return;
    }
    // Only the lines the multi-PV search actually resolved have a score to compare.
    let scored = td.root_moves.iter()
        .take_while(|rm| rm.previous_score != -SCORE_INFINITE && rm.previous_score != SCORE_NONE)
        .count();
    if scored <= 1 {
        return;
    }
    let scores: Vec<i32> = td.root_moves[..scored].iter().map(|rm| rm.previous_score).collect();
    let chosen = skill::variety_pick(&td.skill, td.pos.key, &scores);
    debug_assert!(chosen < scored);
    if chosen == 0 {
        return;
    }
    // The chosen move moves to the front, which is where the caller and the ponder-move
    // extraction both look. The follow-PV guard is not touched: the next search clears it.
    td.root_moves.swap(0, chosen);
    td.best_move = td.root_moves[0].mv;
    td.best_score = td.root_moves[0].previous_score;
}

/// Alpha-beta search with Principal Variation Search (PVS).
///
/// Uses null-window searches for non-PV moves: if a move might beat alpha,
/// re-search with the full [alpha, beta] window. Compile-time `NodeType`
/// (Root/Pv/NonPv) eliminates branches at zero runtime cost.
///
/// Reference: CPW — [Principal Variation Search](https://www.chessprogramming.org/Principal_Variation_Search)
///
/// The parameter list is long because every one of them is per-node state that the
/// recursion threads through; bundling them into a struct would put a copy on every
/// call in the hottest function of the engine.
#[allow(clippy::too_many_arguments)]
fn alpha_beta<NT: NodeType>(
    td: &mut ThreadData,
    shared: &SharedState,
    mut alpha: i32,
    mut beta: i32,
    mut depth: i32,
    ply: usize,
    skip_null: bool,
    cut_node: bool,
) -> i32 {
    // Fundamental search invariants
    debug_assert!(alpha < beta, "alpha_beta: alpha {} >= beta {}", alpha, beta);
    debug_assert!(alpha >= -SCORE_INFINITE, "alpha_beta: alpha {} < -INF", alpha);
    debug_assert!(beta <= SCORE_INFINITE, "alpha_beta: beta {} > INF", beta);
    debug_assert!(ply < MAX_PLY, "alpha_beta: ply {} >= MAX_PLY", ply);
    debug_assert!(depth <= MAX_PLY as i32, "alpha_beta: depth {} too large", depth);
    // Non-PV = null-window
    debug_assert!(NT::PV || alpha == beta - 1,
        "alpha_beta: non-PV with non-null window: [{}, {}]", alpha, beta);
    // PV and cut_node mutually exclusive
    debug_assert!(!(NT::PV && cut_node), "alpha_beta: PV + cut_node at ply {}", ply);
    // ROOT must correspond to ply == 0
    debug_assert!(!NT::ROOT || ply == 0, "alpha_beta: ROOT at ply {}", ply);

    td.pv_len[ply] = 0;

    // Clear static eval early to prevent stale improving data.
    // Keeps early returns from leaving a stale static_eval on the stack,
    // which would corrupt the improving computation at later plies.
    td.ss_mut(ply).static_eval = SCORE_NONE;

    // Check time limits
    td.check_limits();
    if td.should_stop() {
        return 0;
    }

    // Ply overflow guard
    if ply >= MAX_PLY - 1 {
        return if td.pos.in_check() { 0 } else { evaluate_pos(td) };
    }

    // Leaf node: drop into quiescence
    if depth <= 0 {
        return quiescence::<NT>(td, shared, alpha, beta, ply);
    }

    td.nodes += 1;
    td.seldepth = td.seldepth.max(ply as i32 + 1);
    st!(td, s, {
        let nt = if NT::ROOT { 0 } else if NT::PV { 1 } else { 2 };
        s.ab_nodes[nt] += 1;
        s.nodes_by_depth[depth.clamp(0, 31) as usize] += 1;
    });
    tr!(td, t, {
        let flags = if t.recording() {
            let mut f = 0u8;
            if td.pos.in_check() { f |= crate::tree::F_IN_CHECK; }
            if NT::PV { f |= crate::tree::F_PV; }
            if NT::ROOT { f |= crate::tree::F_ROOT; }
            if cut_node { f |= crate::tree::F_CUT_NODE; }
            if skip_null { f |= crate::tree::F_SKIP_NULL; }
            if td.stack[ply + SS_OFFSET].excluded != Move::NONE {
                f |= crate::tree::F_EXCLUDED;
            }
            f
        } else {
            0
        };
        t.enter(ply as u8, depth, flags, alpha, beta);
    });

    // Upcoming repetition: raise alpha to draw score
    if !NT::ROOT && alpha < 0 && td.pos.upcoming_repetition(ply) {
        alpha = 0;
        if alpha >= beta {
            tree_ret!(td, ply, crate::tree::X_REPETITION, depth, alpha);
        }
    }

    // Draw detection: 3-fold repetition and 50-move rule (CPW: Repetition Detection).
    // Returns draw score (0) immediately. See position.rs::is_draw() for logic.
    if ply > 0 && td.pos.is_draw(ply as i32) {
        tree_ret!(td, ply, crate::tree::X_DRAW, depth, 0);
    }

    // Mate distance pruning: tighten alpha/beta bounds based on the shortest
    // possible mate from this ply. If a mate-in-N is already found, longer
    // lines cannot improve the score, so they can be safely pruned.
    //
    // Reference: CPW — Mate Distance Pruning
    if !NT::ROOT {
        alpha = alpha.max(mated_in(ply as i32));
        beta = beta.min(mate_in(ply as i32 + 1));
        if alpha >= beta {
            tree_ret!(td, ply, crate::tree::X_MATE_DISTANCE, depth, alpha);
        }
    }

    // Follow-PV guard: determine whether this node lies on the previous
    // iteration's PV. At the root follow_pv is always true. At other plies,
    // true only if the parent was on the PV AND its played move matches the
    // PV move at that ply.
    let follow_pv = NT::ROOT
        || (ply > 0
            && td.ss(ply - 1).follow_pv
            && (ply - 1) < td.last_iteration_pv_len
            && td.ss(ply - 1).played_move == td.last_iteration_pv[ply - 1]);
    td.ss_mut(ply).follow_pv = follow_pv;

    let in_check = td.pos.in_check();
    let orig_alpha = alpha;

    // Prefetch TT cluster into L1 cache
    shared.tt.prefetch(td.pos.key);

    // TT probe
    let tt_move;
    let mut tt_pv = NT::PV;
    let mut tt_depth = 0i32;
    let mut tt_score = SCORE_NONE;
    let mut tt_eval = SCORE_NONE;
    let tt_bound;
    let tt_hit;
    let excluded = td.ss(ply).excluded;
    st!(td, s, s.tt_probes += 1;);
    if let Some(hit) = shared.tt.probe(td.pos.key, ply as i32, td.pos.halfmove_clock) {
        tt_hit = true;
        tt_move = hit.mv;
        tt_pv |= hit.pv;
        tt_depth = hit.depth;
        tt_score = hit.score;
        tt_eval = hit.eval;
        tt_bound = hit.bound;
        st!(td, s, {
            s.tt_hits += 1;
            if tt_move != Move::NONE {
                s.tt_move_available += 1;
            }
        });
        // TT cutoff: skip at PV nodes, skip during SE search,
        // skip at high halfmove_clock (Graph History Interaction fix)
        if !NT::PV && !NT::ROOT && hit.depth >= depth && excluded == Move::NONE
            && td.pos.halfmove_clock < 96
        {
            let cutoff = match hit.bound {
                Bound::Exact => true,
                Bound::Lower => hit.score >= beta,
                Bound::Upper => hit.score <= alpha,
                _ => false,
            };
            if cutoff {
                st!(td, s, s.tt_cutoffs += 1;);
                tree_ret!(td, ply, crate::tree::X_TT_CUTOFF, depth, hit.score);
            }
        }
    } else {
        tt_hit = false;
        tt_move = Move::NONE;
        tt_bound = Bound::Upper;
    }

    // TT move validation
    if tt_move != Move::NONE {
        debug_assert!(tt_move.from_sq().0 < 64 && tt_move.to_sq().0 < 64,
            "alpha_beta: TT move squares OOB: {} -> {}", tt_move.from_sq().0, tt_move.to_sq().0);
    }

    // Online TB cache probe (prefetched DTM from Lichess API).
    // Highest priority: DTM exact for ≤7 pieces, prefetched during opponent's turn.
    // Probed BEFORE GaiaTB and Syzygy — more pieces and exact DTM.
    // Cheap filters (piece count, castling) BEFORE Mutex to avoid lock contention
    // at millions of NPS in middlegame positions.
    #[cfg(feature = "online-tb")]
    if !NT::ROOT
        && excluded == Move::NONE
        && crate::online_tb::is_enabled()
        && td.pos.castling_rights == 0
        && td.pos.occupied().count_ones() <= 7
        && let Some(entry) = crate::online_tb::probe(td.pos.key)
    {
        td.tb_hits += 1;
        use crate::online_tb::TbCategory;
        let score = match entry.category {
            TbCategory::Draw => 0,
            TbCategory::Win => {
                if let Some(dtm) = entry.dtm {
                    // dtm is in halfmoves (plies) from the API
                    SCORE_MATE - ply as i32 - dtm.abs()
                } else {
                    tb_win_in(ply as i32)
                }
            }
            TbCategory::Loss => {
                if let Some(dtm) = entry.dtm {
                    -SCORE_MATE + ply as i32 + dtm.abs()
                } else {
                    tb_loss_in(ply as i32)
                }
            }
        };
        let bound = if score >= beta {
            Bound::Lower
        } else if score <= alpha {
            Bound::Upper
        } else {
            Bound::Exact
        };
        shared.tt.store(
            td.pos.key, (depth + 6).min(MAX_PLY as i32 - 1),
            SCORE_NONE, score, bound, Move::NONE, ply as i32, tt_pv,
        );
        tree_ret!(td, ply, crate::tree::X_TB_ONLINE, depth, score);
    }

    // GaiaTB DTM probe (3+4 piece endgames, exact mate score).
    // Like Nalimov/Gaviota, ignores the 50-move rule (acceptable for ≤4 pieces).
    // Don't probe at root (root move ranking is handled by Syzygy/search itself),
    // don't probe in SE (excluded != NONE).
    // Skip positions with en passant rights: the tables don't encode EP,
    // so the DTM could be wrong when an EP capture changes the outcome.
    #[cfg(feature = "gaiatb")]
    if !NT::ROOT
        && excluded == Move::NONE
        && td.pos.castling_rights == 0
        && td.pos.ep_square == Square::NONE
        && td.pos.occupied().count_ones() <= 4
        && crate::dtm::available()
        && let Some(score) = crate::dtm::probe_position(&td.pos, ply as i32)
    {
        let bound = if score >= beta {
            Bound::Lower
        } else if score <= alpha {
            Bound::Upper
        } else {
            Bound::Exact
        };
        shared.tt.store(
            td.pos.key, (depth + 6).min(MAX_PLY as i32 - 1),
            SCORE_NONE, score, bound, Move::NONE, ply as i32, tt_pv,
        );
        tree_ret!(td, ply, crate::tree::X_TB_DTM, depth, score);
    }

    // Nalimov DTM probe (3-6 piece endgames, disk-based with LRU cache).
    // Supports en passant natively (disjoint index ranges in Nalimov tables).
    // Only probe at sufficient depth to amortize disk I/O cost (Crafty pattern:
    // main search only, no qsearch, depth threshold).
    // Don't probe at root (handled by search/Syzygy) or in SE.
    #[cfg(feature = "nalimov")]
    if !NT::ROOT
        && excluded == Move::NONE
        && depth >= 2
        && td.pos.castling_rights == 0
        && td.pos.occupied().count_ones() <= crate::nalimov::max_pieces()
        && crate::nalimov::available()
        && let Some(score) = crate::nalimov::probe_position(&td.pos, ply as i32)
    {
        td.tb_hits += 1;
        let bound = if score >= beta {
            Bound::Lower
        } else if score <= alpha {
            Bound::Upper
        } else {
            Bound::Exact
        };
        shared.tt.store(
            td.pos.key, (depth + 6).min(MAX_PLY as i32 - 1),
            SCORE_NONE, score, bound, Move::NONE, ply as i32, tt_pv,
        );
        tree_ret!(td, ply, crate::tree::X_TB_NALIMOV, depth, score);
    }

    // Syzygy TB probe (non-root, non-SE only).
    // When DTZ is available (cardinality=0), skip in-tree probing entirely —
    // Pyrrhic handles everything via root ranking.
    // PV narrowing: syzygy_min/syzygy_max bound the search result.
    #[cfg_attr(not(feature = "syzygy"), allow(unused_mut))]
    let mut syzygy_min = -SCORE_INFINITE;
    #[cfg_attr(not(feature = "syzygy"), allow(unused_mut))]
    let mut syzygy_max = SCORE_INFINITE;
    #[cfg(feature = "syzygy")]
    if !NT::ROOT
        && excluded == Move::NONE
        && !td.datagen_mode
        && td.tb_config.cardinality > 0
        && td.pos.occupied().count_ones() <= td.tb_config.cardinality
        && td.pos.halfmove_clock == 0
        && td.pos.castling_rights == 0
        && let Some(wdl) = crate::tb::probe_wdl(&td.pos)
    {
        td.tb_hits += 1;

        let (tb_score, bound) = match wdl {
            crate::tb::Wdl::Win  => (tb_win_in(ply as i32), Bound::Lower),
            crate::tb::Wdl::Loss => (tb_loss_in(ply as i32), Bound::Upper),
            crate::tb::Wdl::Draw => (0, Bound::Exact),
        };

        if bound == Bound::Exact
            || (bound == Bound::Lower && tb_score >= beta)
            || (bound == Bound::Upper && tb_score <= alpha)
        {
            // Store in TT with depth bonus (depth + 6)
            shared.tt.store(
                td.pos.key, (depth + 6).min(MAX_PLY as i32 - 1),
                SCORE_NONE, tb_score, bound, Move::NONE, ply as i32, tt_pv,
            );
            tree_ret!(td, ply, crate::tree::X_TB_WDL, depth, tb_score);
        }

        // Narrow alpha/beta for PV nodes
        if NT::PV {
            if bound == Bound::Lower {
                syzygy_min = tb_score;
                alpha = alpha.max(tb_score);
            }
            if bound == Bound::Upper {
                syzygy_max = tb_score;
            }
        }
    }

    // Static eval with correction history
    // Store SCORE_NONE when in check so improving fallback skips it
    let raw_eval: i32;
    let correction_value: i32;
    let static_eval: i32;
    // Capture lazy eval flag before child searches overwrite td.used_lazy_eval.
    // When lazy eval fired, correction history is neither applied nor updated
    // (correction tables are trained against NNUE, not PeSTO).
    let lazy_eval_this_node: bool;
    if in_check {
        raw_eval = SCORE_NONE;
        static_eval = SCORE_NONE;
        correction_value = 0;
        lazy_eval_this_node = false;
    } else {
        // Reuse TT eval if available (avoids recomputing eval).
        // NOTE: when tt_eval is used, evaluate_pos() is NOT called, so
        // td.nnue.ensure_updated() does NOT run. The NNUE accumulator at
        // this ply may hold stale values from a previous search iteration.
        // This is fine because push_null() marks accurate=[false;2], forcing
        // any null-move child to walk back to a truly-accurate ancestor.
        // Do NOT change push_null() to eagerly clone the accumulator with
        // accurate=true — it would propagate stale values.
        td.used_lazy_eval = false;
        raw_eval = if tt_eval != SCORE_NONE {
            st!(td, s, s.tt_eval_reused += 1;);
            tt_eval
        } else {
            evaluate_pos(td)
        };
        // A handicapped eval counts as lazy whatever produced it: the TT can hand back a
        // score a weakened search stored, which never reaches evaluate_pos and so would
        // otherwise be fed to the correction tables as if it were the network's opinion.
        lazy_eval_this_node = td.used_lazy_eval || td.skill.active;
        // Ensure NNUE threat features are up-to-date even when TT eval is used,
        // so children can do incremental threat updates instead of full recomputes.
        if tt_eval != SCORE_NONE && nnue::network::has_network() {
            td.nnue.ensure_updated(&td.pos);
        }
        if lazy_eval_this_node {
            // Lazy eval: PeSTO score used, skip correction history (trained against NNUE)
            correction_value = 0;
            let scaled_eval = raw_eval as i64
                * (tune::FIFTY_MOVE_SCALE() - td.pos.halfmove_clock as i32) as i64
                / tune::FIFTY_MOVE_SCALE() as i64;
            static_eval = (scaled_eval as i32)
                .clamp(-SCORE_MATE_IN_MAX + 1, SCORE_MATE_IN_MAX - 1);
        } else {
            // Pawn correction history (weighted sum / 65536)
            let stm = td.pos.side_to_move;
            let pawn_entry = td.pawn_correction.get(td.pos.pawn_key, stm) as i64;
            // Non-pawn correction: sum of both colors' non-pawn keys
            let non_pawn_entry = td.non_pawn_correction.get(td.pos.non_pawn_key[0], Color::White, stm) as i64
                + td.non_pawn_correction.get(td.pos.non_pawn_key[1], Color::Black, stm) as i64;
            // Minor piece correction: N+B+K combined key
            let minor_entry = td.minor_correction.get(td.pos.minor_key, stm) as i64;
            // Continuation correction history: subtable[ply-offset] indexed by move at ply-1
            let cont_corr = {
                let base = ply + SS_OFFSET;
                let mut cc = 0i64;
                for &offset in &[2, 4] {
                    if base > offset {
                        let prev = &td.stack[base - offset];
                        let last = &td.stack[base - 1];
                        if prev.played_move != Move::NONE && last.played_move != Move::NONE {
                            cc += ContCorrectionHistory::get(
                                prev.cont_corr_ptr as *const _,
                                last.moved_piece,
                                last.played_move.to_sq(),
                            ) as i64;
                        }
                    }
                }
                cc
            };
            correction_value = (pawn_entry * tune::PAWN_CORR_FACTOR() as i64 / CORR_DIVISOR
                + non_pawn_entry * tune::NON_PAWN_CORR_FACTOR() as i64 / CORR_DIVISOR
                + minor_entry * tune::MINOR_CORR_FACTOR() as i64 / CORR_DIVISOR
                + cont_corr * tune::CONT_CORR_FACTOR() as i64 / CORR_DIVISOR) as i32;
            // Optimism blending: material-weighted blend of raw eval + optimism
            let stm = td.pos.side_to_move;
            let material = td.pos.material() / 32;
            let mat_mul = (tune::OPT_MAT_SCALE() + material) as i64;
            let opt_mul = (tune::OPT_MAT_BASE() + material) as i64;
            let blended = ((raw_eval as i64 * mat_mul
                + td.optimism[stm.index()] as i64 * opt_mul / 32)
                / 1024) as i32;
            // 50-move rule scaling: eval * (scale - rule50) / scale
            let scaled_eval = blended as i64
                * (tune::FIFTY_MOVE_SCALE() - td.pos.halfmove_clock as i32) as i64
                / tune::FIFTY_MOVE_SCALE() as i64;
            static_eval = (scaled_eval as i32 + correction_value)
                .clamp(-SCORE_MATE_IN_MAX + 1, SCORE_MATE_IN_MAX - 1);
        }
        debug_assert!(static_eval.abs() < SCORE_MATE_IN_MAX,
            "alpha_beta: static_eval {} looks like mate score", static_eval);
    }
    td.ss_mut(ply).static_eval = static_eval;
    td.ss_mut(ply).correction_value = correction_value;

    // Improving detection with 4-ply fallback
    let mut improvement = 0i32;
    if !in_check && ply >= 2 && td.ss(ply - 2).static_eval != SCORE_NONE {
        improvement = static_eval - td.ss(ply - 2).static_eval;
    } else if !in_check && ply >= 4 && td.ss(ply - 4).static_eval != SCORE_NONE {
        improvement = static_eval - td.ss(ply - 4).static_eval;
    }
    let improving = improvement > 0;

    tr!(td, t, {
        if t.recording() {
            let mut f = 0u8;
            if tt_hit { f |= crate::tree::EF_TT_HIT; }
            f |= crate::tree::bound_bits(tt_bound) << 1;
            if tt_pv { f |= crate::tree::EF_TT_PV; }
            if improving { f |= crate::tree::EF_IMPROVING; }
            if in_check { f |= crate::tree::EF_IN_CHECK; }
            t.eval(ply as u8, static_eval, raw_eval, tt_move.0, tt_score, tt_eval, tt_depth, f);
        }
    });

    // Opponent worsening: the opponent's position got worse than expected.
    // static_eval + prev_eval > threshold means our eval exceeds the negation
    // of the parent's eval (opponent worsening heuristic).
    let opponent_worsening = !in_check && ply >= 1 && {
        let prev_eval = td.ss(ply - 1).static_eval;
        prev_eval != SCORE_NONE && static_eval + prev_eval > tune::OW_THRESHOLD()
    };

    // Static history: update butterfly history of the opponent's previous quiet
    // move based on eval change.
    if !in_check
        && excluded == Move::NONE
        && ply >= 1
        && !td.ss(ply - 1).is_capture
        && td.ss(ply - 1).played_move != Move::NONE
        && td.ss(ply - 1).static_eval != SCORE_NONE
    {
        let bonus = (tune::STATIC_HIST_FACTOR() * -(static_eval + td.ss(ply - 1).static_eval) / 128)
            .clamp(tune::STATIC_HIST_MIN(), tune::STATIC_HIST_MAX());
        td.history.update(td.pos.side_to_move.flip(), td.ss(ply - 1).played_move, td.pos.prior_threats(), bonus);
    }

    // Hindsight depth adjustment:
    // read the parent's LMR reduction (plies) and clear it.
    let prior_reduction = if ply >= 1 {
        let r = td.ss(ply - 1).reduction;
        td.ss_mut(ply - 1).reduction = 0;
        r
    } else {
        0
    };

    if !NT::PV && !in_check {
        // Parent reduced heavily and opponent isn't declining: reduction was wrong
        if prior_reduction >= tune::HINDSIGHT_INC_MIN_R() && !opponent_worsening {
            depth += 1;
            st!(td, s, s.hindsight_inc += 1;);
        }
        // Both evals sum large (comfortable position): reduce depth
        if prior_reduction >= tune::HINDSIGHT_DEC_MIN_R()
            && depth >= 2
            && ply >= 1
            && td.ss(ply - 1).static_eval != SCORE_NONE
            && static_eval + td.ss(ply - 1).static_eval > tune::HINDSIGHT_DEC_THRESHOLD()
        {
            depth -= 1;
            st!(td, s, s.hindsight_dec += 1;);
        }
    }

    // RFP ttHit multiplier: the RFP multiplier is modulated by ttHit.
    // Without TT info, the discount shrinks the mult → smaller margin →
    // easier cutoff. The effect scales with depth (multiplicative).
    let rfp_mult = tune::RFP_MARGIN() - tune::RFP_NO_TT_DISCOUNT() * (!tt_hit) as i32;
    debug_assert!(rfp_mult > 0, "rfp_mult negative: {}", rfp_mult);
    st!(td, s, {
        if !NT::PV && !in_check && depth <= tune::RFP_MAX_DEPTH() {
            s.rfp.tried[crate::stats::db(depth)] += 1;
        }
    });
    if !NT::PV
        && !in_check
        && depth <= tune::RFP_MAX_DEPTH()
        && static_eval
            - (rfp_mult * (depth - improving as i32)).max(tune::RFP_MIN())
            + tune::RFP_OW_BONUS() * opponent_worsening as i32
            >= beta
    {
        st!(td, s, s.rfp.cut[crate::stats::db(depth)] += 1;);
        tree_ret!(td, ply, crate::tree::X_RFP, depth, (static_eval + beta) / 2);
    }

    // Razoring: at low depths, when the static eval is far below alpha,
    // drop directly into qsearch.
    // The quadratic depth² margin naturally limits this to depths 1-3.
    // Guards: (1) no razor if the TT move is quiet (qsearch won't evaluate it)
    //         (2) no razor if the TT has a lower bound (position possibly better)
    st!(td, s, {
        if !NT::PV && !in_check && excluded == Move::NONE {
            s.razor.tried[crate::stats::db(depth)] += 1;
        }
    });
    if !NT::PV
        && !in_check
        && excluded == Move::NONE
        && static_eval < alpha - tune::RAZOR_BASE() - tune::RAZOR_QUAD() * depth * depth
        && alpha < tune::RAZOR_ALPHA_MAX()
        && !(tt_move != Move::NONE && is_quiet(&td.pos, tt_move))
        && tt_bound != Bound::Lower
    {
        debug_assert!(depth >= 1, "razoring: depth must be >= 1");
        st!(td, s, s.razor.cut[crate::stats::db(depth)] += 1;);
        tree_ret!(td, ply, crate::tree::X_RAZOR_TO_QS, depth,
            quiescence::<NT>(td, shared, alpha, beta, ply));
    }

    // Null move pruning (with verification search at high depths)
    // NMP valuable-capture TT guard:
    // Disable NMP when the TT indicates a fail-high whose best move
    // captures a valuable piece (Rook+). Passing the move would forfeit
    // that capture, making the null-move score unreliable.
    // NMP cut-node guard:
    // Restrict NMP to cut nodes only. Cut nodes are predicted to
    // fail high by the PVS alternation — NMP is more reliable there.
    // At all-nodes, NMP yields more false positives.
    if !NT::PV
        && cut_node
        && !skip_null
        && !in_check
        && excluded == Move::NONE
        && depth >= tune::NMP_MIN_DEPTH()
        && static_eval >= beta
        && td.pos.has_non_pawn_material(td.pos.side_to_move)
        && !(tt_bound == Bound::Lower
            && tt_move != Move::NONE
            && {
                let cap = td.pos.board[tt_move.to_sq().index()];
                cap != Piece::NONE && (cap.piece_type() as u8) >= (PieceType::Rook as u8)
            })
    {
        debug_assert!(!in_check, "null move en echec");
        debug_assert!(!NT::PV, "null move en PV node");
        debug_assert!(excluded == Move::NONE, "null move pendant SE");
        debug_assert!(static_eval >= beta, "null move: eval {} < beta {}", static_eval, beta);
        // No two consecutive null moves
        debug_assert!(ply < 1 || td.ss(ply - 1).played_move != Move::NONE,
            "null move: consecutive null moves at ply {}", ply);

        let r = tune::NMP_BASE_R() + depth / tune::NMP_DEPTH_DIV()
            + ((static_eval - beta) / tune::NMP_EVAL_DIV()).min(tune::NMP_EVAL_MAX());

        // Record null move on stack
        {
            let sentinel_ptr = &*td.sentinel_conthist as *const PieceToTable as *mut PieceToTable;
            let se = td.ss_mut(ply);
            se.played_move = Move::NONE;
            se.moved_piece = Piece::NONE;
            se.is_capture = false;
            se.conthist_ptr = sentinel_ptr;
            se.cont_corr_ptr = std::ptr::null_mut();
        }

        // push_null/pop maintain the NNUE accumulator stack index.
        // push_null is lazy (no 2KB clone): it marks accurate=[false;2] and
        // defers the actual value copy to ensure_updated() when the child
        // node needs an eval. This also fixes a correctness bug: if THIS
        // node's eval came from a TT hit (line ~351: raw_eval = tt_eval),
        // ensure_updated() was never called and our accumulator is stale.
        // The lazy approach forces descendants to walk back to a truly-accurate
        // ancestor rather than trusting our potentially stale values.
        st!(td, s, s.nmp.tried[crate::stats::db(depth)] += 1;);
        tr!(td, t, t.mv(ply as u8, 0, 0, 255, crate::tree::A_NULL_SEARCH, r * 1024, depth - r););
        if nnue::network::has_network() {
            td.nnue.push_null();
        }
        td.pos.make_null_move();
        let null_score =
            -alpha_beta::<NonPv>(td, shared, -beta, -beta + 1, depth - r, ply + 1, true, false);
        td.pos.unmake_null_move();
        if nnue::network::has_network() {
            td.nnue.pop();
        }

        if td.should_stop() {
            tree_ret!(td, ply, crate::tree::X_STOPPED, depth, 0);
        }

        if null_score >= beta && !is_mate_score(null_score) {
            // At low depths, return immediately.
            // At high depths (>= NMP_VERIF_DEPTH), reduce remaining depth
            // and continue normal search instead of returning.
            // No re-search: NMP stays active after verification
            if depth < tune::NMP_VERIF_DEPTH() {
                st!(td, s, s.nmp.cut[crate::stats::db(depth)] += 1;);
                tree_ret!(td, ply, crate::tree::X_NMP_CUTOFF, depth, null_score);
            }

            // Reduce depth and fall through to normal move loop
            st!(td, s, s.nmp_verif_runs += 1;);
            depth -= tune::NMP_VERIF_REDUCTION();
            if depth <= 0 {
                tree_ret!(td, ply, crate::tree::X_NMP_VERIF_TO_QS, depth,
                    quiescence::<NT>(td, shared, alpha, beta, ply));
            }
            // Fall through to move loop with reduced depth
        }
    }

    // ProbCut: forward pruning via shallow search of winning captures.
    // If a capture proves score >= beta + margin at reduced depth, the
    // full-depth search would almost certainly also fail high.
    //
    // Two-pass approach:
    //   1. Quiescence search with null window [-probcut_beta, -probcut_beta+1]
    //   2. If qsearch passes, confirm with shallow alpha-beta at depth - reduction
    //
    // Reference: CPW — [ProbCut](https://www.chessprogramming.org/ProbCut)
    if !NT::PV
        && !in_check
        && excluded == Move::NONE
        && depth >= tune::PROBCUT_MIN_DEPTH()
        && beta.abs() < SCORE_MATE_IN_MAX
        && !(tt_depth >= depth - 3
             && tt_score != SCORE_NONE
             && tt_score < beta + tune::PROBCUT_MARGIN())
    {
        let probcut_beta = beta + tune::PROBCUT_MARGIN();
        let see_threshold = (probcut_beta - static_eval) * 10 / 16;

        // TT move: include only if it's a capture passing SEE threshold
        let pc_tt_move = if tt_move != Move::NONE
            && (tt_move.move_type() != MT_NORMAL
                || td.pos.board[tt_move.to_sq().index()] != Piece::NONE)
            && crate::see::see(&td.pos, tt_move, see_threshold)
        {
            tt_move
        } else {
            Move::NONE
        };

        let mut pc_picker = MovePicker::new_qsearch(pc_tt_move);
        st!(td, s, s.probcut.tried[crate::stats::db(depth)] += 1;);

        loop {
            let m = pc_picker.next::<true>(
                &td.pos, &td.history, &td.cap_history, &td.pawn_history,
                &shared.tt, 0, &td.stack, td.pos.side_to_move,
            );
            if m == Move::NONE {
                break;
            }

            // Manual SEE filter: new_qsearch yields SEE >= 0, ProbCut needs higher
            if m != pc_tt_move && !crate::see::see(&td.pos, m, see_threshold) {
                continue;
            }

            if !movegen::is_legal(&td.pos, m) {
                continue;
            }

            // Record move context on stack for child conthist/correction lookups
            let pc_piece = td.pos.board[m.from_sq().index()];
            debug_assert!(pc_piece != Piece::NONE,
                "probcut: piece NONE for {} at ply {}", m.to_uci(), ply);
            {
                let conthist_ptr = td.cont_history.subtable_ptr(true, pc_piece, m.to_sq());
                let cont_corr_ptr = td.cont_correction.subtable_ptr(pc_piece, m.to_sq());
                let se = td.ss_mut(ply);
                se.played_move = m;
                se.moved_piece = pc_piece;
                se.is_capture = true;
                se.conthist_ptr = conthist_ptr;
                se.cont_corr_ptr = cont_corr_ptr;
            }

            if nnue::network::has_network() {
                td.nnue.push(m, &td.pos);
            }
            td.pos.make_move(m);
            td.nodes += 1;

            // Pass 1: qsearch verification
            tr!(td, t, t.mv(ply as u8, m.0, 0, 255, crate::tree::A_PROBCUT_QS, 0, 0););
            let mut score = -quiescence::<NonPv>(
                td, shared, -probcut_beta, -probcut_beta + 1, ply + 1,
            );

            // Pass 2: shallow alpha-beta confirmation
            if score >= probcut_beta {
                let pc_depth = (depth - tune::PROBCUT_REDUCTION()).max(1);
                tr!(td, t, t.mv(ply as u8, m.0, 0, 255, crate::tree::A_PROBCUT_AB, 0, pc_depth););
                score = -alpha_beta::<NonPv>(
                    td, shared, -probcut_beta, -probcut_beta + 1,
                    pc_depth, ply + 1, false, !cut_node,
                );
            }

            td.pos.unmake_move(m);
            if nnue::network::has_network() {
                td.nnue.pop();
            }

            if td.should_stop() {
                tree_ret!(td, ply, crate::tree::X_STOPPED, depth, 0);
            }

            if score >= probcut_beta {
                st!(td, s, s.probcut.cut[crate::stats::db(depth)] += 1;);
                shared.tt.store(
                    td.pos.key, depth - 3, raw_eval, score,
                    Bound::Lower, m, ply as i32, tt_pv,
                );
                tree_ret!(td, ply, crate::tree::X_PROBCUT, depth, score);
            }
        }
    }

    // Internal Iterative Reduction (IIR): without a TT move, move ordering is
    // degraded — reduce depth by 1. The TT gets populated for future iterations.
    // Replaces the older IID technique of doing a shallow search first.
    //
    // Reference: CPW — Internal Iterative Deepening (predecessor technique)
    // Follow-PV guard: do not apply the IIR reduction on the predicted PV line
    if !follow_pv && (NT::PV || cut_node) && depth >= tune::IIR_DEPTH() + 2 * cut_node as i32 && tt_move == Move::NONE {
        depth -= 1;
        st!(td, s, s.iir_applied += 1;);
    }

    let stm = td.pos.side_to_move;
    let mut best_score = syzygy_min; // -SCORE_INFINITE, or TB lower bound if set
    let mut best_move = Move::NONE;
    let mut move_count = 0u32;

    // Track quiet moves searched (for history malus on cutoff)
    let mut quiets_searched: [Move; MAX_QUIETS_SEARCHED] = [Move::NONE; MAX_QUIETS_SEARCHED];
    let mut quiet_count: usize = 0;

    // Track capture moves searched (for capture history malus on cutoff)
    let mut captures_searched: [Move; MAX_CAPTURES_SEARCHED] = [Move::NONE; MAX_CAPTURES_SEARCHED];
    let mut capture_count: usize = 0;

    // Look up countermove from previous ply
    let countermove = if ply >= 1 {
        let prev = td.ss(ply - 1);
        td.countermoves.get(prev.moved_piece, prev.played_move.to_sq())
    } else {
        Move::NONE
    };

    let killers = td.killers.get(ply);
    let mut picker = MovePicker::new(tt_move, killers, countermove, depth);

    let mut skip_quiets = false;

    loop {
        let m = picker.next::<false>(&td.pos, &td.history, &td.cap_history, &td.pawn_history, &shared.tt, ply, &td.stack, stm);
        if m == Move::NONE {
            break;
        }

        // Skip excluded move during singular extension search
        if m == excluded {
            tr!(td, t, t.mv(ply as u8, m.0, move_count, 255, crate::tree::A_EXCLUDED_SE, 0, 0););
            continue;
        }

        // Legality check
        if !movegen::is_legal(&td.pos, m) {
            continue;
        }

        // Multi-PV: skip moves already assigned to earlier PV lines
        if NT::ROOT && td.pv_index > 0
            && td.root_moves[..td.pv_index].iter().any(|rm| rm.mv == m)
        {
            continue;
        }

        // A weak opponent simply does not notice some of the moves available to it.
        // Skipped before the count, so a move that never crossed its mind does not push
        // the ones that did further down the late-move thresholds. Evasions are never
        // skipped, and neither is the first legal move — which is also what keeps the
        // mate and stalemate tests below honest, since move_count == 0 then still means
        // there was genuinely nothing to play.
        if td.skill.blind
            && move_count >= 1
            && !in_check
            && !skill::sees_move(&td.skill, td.pos.key, m, ply, easy_to_notice(td, m, ply))
        {
            continue;
        }

        move_count += 1;

        let is_quiet_move = is_quiet(&td.pos, m);

        // Quiet skipping is enforced inside the MovePicker (skip_quiet_moves):
        // once armed, quiet stages are bypassed so skipped quiets never reach
        // this loop nor inflate move_count.
        debug_assert!(!(skip_quiets && is_quiet_move),
            "search: picker yielded quiet {} while skip_quiets armed", m.to_uci());

        // Compute move context before make_move (board changes after)
        let moved_piece = td.pos.board[m.from_sq().index()];
        debug_assert!(moved_piece != Piece::NONE,
            "alpha_beta: piece NONE for {} at ply {}", m.to_uci(), ply);
        debug_assert!(moved_piece.color() == td.pos.side_to_move,
            "alpha_beta: piece {:?} wrong color for stm {:?}", moved_piece, td.pos.side_to_move);
        let captured_pt = if !is_quiet_move { get_captured_pt(&td.pos, m) } else { PieceType::Pawn };

        // Compute combined history for LMR and pruning (before make_move — board intact).
        // Sums butterfly, pawn structure, and continuation histories for quiet moves,
        // or capture history for tactical moves.
        //
        // References:
        // - CPW: History Heuristic § Combined History
        // Combined history: butterfly + pawn + 4 continuation history offsets
        let combined_history = if is_quiet_move {
            let to = m.to_sq();
            td.history.get(stm, m, td.pos.threats)
                + td.pawn_history.get(td.pos.pawn_key, moved_piece, to)
                + td.conthist(ply, 1, moved_piece, to)
                + td.conthist(ply, 2, moved_piece, to)
                + td.conthist(ply, 4, moved_piece, to)
        } else {
            td.cap_history.get(moved_piece, m.to_sq(), captured_pt)
        };

        // SEE pruning: use Static Exchange Evaluation to estimate material outcome.
        // Skip moves losing more than a depth-scaled threshold. Quiet moves use a
        // quadratic margin in lmr_depth — the effective depth the move would be
        // searched at after the expected LMR reduction, adjusted by history — so
        // a late or poorly-ranked quiet is judged at its real search depth, not
        // the nominal one. Captures use a linear margin in nominal depth.
        // Follow-PV guard: do not SEE-prune quiet moves on the predicted PV
        // line
        //
        // Reference: CPW — Static Exchange Evaluation, SEE - The Swap Algorithm
        if !NT::ROOT
            && !in_check
            && move_count > 1
            && best_score > -SCORE_MATE_IN_MAX
            && !(follow_pv && NT::PV && is_quiet_move)
        {
            let threshold = if is_quiet_move {
                let ln_mc = LMR_LN[(move_count as usize).min(63)];
                let ln_d = LMR_LN[(depth as usize).min(63)];
                let loglog = tune::LMR_LOG_MUL() * ln_mc * ln_d / 1024;
                let lmr_depth = (depth - loglog / 1024
                    + combined_history / tune::PRUNE_HIST_DIV())
                    .max(1);
                debug_assert!(lmr_depth >= 1);
                tune::SEE_QUIET_MARGIN() * lmr_depth * lmr_depth
            } else {
                tune::SEE_CAPTURE_MARGIN() * depth
            };
            if !crate::see::see(&td.pos, m, threshold) {
                st!(td, s, s.see_pruned[crate::stats::db(depth)] += 1;);
                tr!(td, t, {
                    if t.recording() {
                        let cat = crate::stats::move_category(&td.pos, m, tt_move, killers, countermove) as u8;
                        t.mv(ply as u8, m.0, move_count, cat, crate::tree::A_PRUNED_SEE, threshold, 0);
                    }
                });
                continue;
            }
        }

        // History pruning: skip quiets whose combined history score (butterfly +
        // continuation) is below a depth-scaled threshold. Killers and
        // countermove are exempt — they proved valuable in sibling nodes.
        // Check guard: never prune moves that give check.
        //
        // Reference: CPW — History Leaf Pruning
        if !NT::PV
            && !in_check
            && !td.pos.gives_check(m)
            && is_quiet_move
            && best_score > -SCORE_MATE_IN_MAX
            && m != killers[0]
            && m != killers[1]
            && m != countermove
            && combined_history < (tune::HIST_PRUNE_MARGIN() + tune::HIST_PRUNE_DEPTH() * depth / 1024) * depth
        {
            st!(td, s, s.hist_pruned[crate::stats::db(depth)] += 1;);
            tr!(td, t, {
                if t.recording() {
                    let cat = crate::stats::move_category(&td.pos, m, tt_move, killers, countermove) as u8;
                    t.mv(ply as u8, m.0, move_count, cat, crate::tree::A_PRUNED_HIST, combined_history.clamp(-32768, 32767), 0);
                }
            });
            skip_quiets = true;
            picker.skip_quiet_moves();
            continue;
        }

        // Late move pruning (move count pruning): after enough moves at a shallow
        // depth, remaining quiets are unlikely to be best. Threshold is quadratic
        // in depth, halved when improving (be more selective).
        // Check guard: never prune moves that give check.
        //
        // Reference: CPW — Futility Pruning § Move Count Based Pruning
        if !NT::PV
            && !in_check
            && !td.pos.gives_check(m)
            && best_score > -SCORE_MATE_IN_MAX
            && move_count as i32 >= (depth * depth + tune::LMP_BASE()) / (2 - improving as i32)
        {
            skip_quiets = true;
            picker.skip_quiet_moves();
            if is_quiet_move {
                st!(td, s, s.lmp_pruned[crate::stats::db(depth)] += 1;);
                tr!(td, t, {
                    if t.recording() {
                        let cat = crate::stats::move_category(&td.pos, m, tt_move, killers, countermove) as u8;
                        t.mv(ply as u8, m.0, move_count, cat, crate::tree::A_PRUNED_LMP, 0, 0);
                    }
                });
                continue;
            }
        }

        // Futility pruning: at shallow depths, if static eval + a depth-scaled margin
        // can't reach alpha, skip remaining quiet moves. The margin grows linearly
        // with depth (deeper = more potential for score improvement).
        // Check guard: never prune moves that give check.
        //
        // Reference: CPW — Futility Pruning
        if !NT::PV
            && !in_check
            && !td.pos.gives_check(m)
            && best_score > -SCORE_MATE_IN_MAX
            && is_quiet_move
            && depth <= tune::FP_MAX_DEPTH()
            && static_eval + tune::FP_COEFF() * depth - tune::FP_BASE() <= alpha
        {
            let fp_value = static_eval + tune::FP_COEFF() * depth - tune::FP_BASE();
            if best_score < fp_value {
                best_score = fp_value;
            }
            st!(td, s, s.fp_pruned[crate::stats::db(depth)] += 1;);
            tr!(td, t, {
                if t.recording() {
                    let cat = crate::stats::move_category(&td.pos, m, tt_move, killers, countermove) as u8;
                    t.mv(ply as u8, m.0, move_count, cat, crate::tree::A_PRUNED_FUTILITY, fp_value.clamp(-32768, 32767), 0);
                }
            });
            skip_quiets = true;
            picker.skip_quiet_moves();
            continue;
        }

        // Singular extension: test if TT move is significantly better than alternatives
        let mut singular_ext = 0i32;

        if m == tt_move
            && !NT::ROOT
            && excluded == Move::NONE
            && depth >= tune::SE_MIN_DEPTH()
            && tt_depth >= depth - tune::SE_TT_DEPTH_MARGIN()
            && tt_score.abs() < SCORE_MATE_IN_MAX
            && (tt_bound == Bound::Lower || tt_bound == Bound::Exact)
            && (ply as i32) < 2 * td.root_depth
        {
            debug_assert!(excluded == Move::NONE, "SE: nested SE");
            debug_assert!(m == tt_move, "SE on non-TT move");
            debug_assert!(tt_score != SCORE_NONE, "SE: tt_score is SCORE_NONE");
            debug_assert!(tt_score.abs() < SCORE_MATE_IN_MAX, "SE on mate score");
            debug_assert!(
                tt_depth >= depth - tune::SE_TT_DEPTH_MARGIN(),
                "SE: tt_depth {} below the margin {}",
                tt_depth,
                depth - tune::SE_TT_DEPTH_MARGIN()
            );
            // Lower singular beta on ex-PV nodes: widen the margin there
            let se_ttpv_extra = if tt_pv && !NT::PV { tune::SE_TTPV_DEPTH() * depth / 128 } else { 0 };
            debug_assert!(se_ttpv_extra >= 0, "SE: se_ttpv_extra negative: {}", se_ttpv_extra);
            let singular_beta = (tt_score - depth - se_ttpv_extra).max(-SCORE_MATE);
            let singular_depth = (depth - 1) / 2;

            st!(td, s, s.se_tried += 1;);
            tr!(td, t, t.mv(ply as u8, m.0, move_count, 255, crate::tree::A_SE_VERIF, 0, singular_depth););
            td.ss_mut(ply).excluded = m;
            let se_score = alpha_beta::<NonPv>(
                td, shared,
                singular_beta - 1, singular_beta,
                singular_depth, ply,
                false, cut_node,
            );
            td.ss_mut(ply).excluded = Move::NONE;

            if td.should_stop() {
                tree_ret!(td, ply, crate::tree::X_STOPPED, depth, 0);
            }

            if se_score < singular_beta {
                // TT move is singular — extend
                singular_ext = 1;
                let parent_de = if ply > 0 { td.ss(ply - 1).double_extensions } else { 0 };
                // Adjust SE margins by |correction_value|: when the correction is
                // large, the static eval is unreliable → shrink the margins so
                // extensions trigger more easily (corrValAdj-like idea).
                let corr_adj = correction_value.abs() / tune::SE_CORR_DIV();
                debug_assert!(corr_adj >= 0, "corr_adj must be non-negative");
                if !NT::PV
                    && se_score < singular_beta - (tune::SE_DOUBLE_MARGIN() + tune::SE_DOUBLE_DEPTH() * depth / 1024) + corr_adj
                    && parent_de < tune::SE_MAX_DOUBLE_EXT()
                {
                    singular_ext = 2;
                    // SE triple extension without a quiet guard: any move may triple-extend
                    if se_score < singular_beta - (tune::SE_TRIPLE_MARGIN() + tune::SE_TRIPLE_DEPTH() * depth / 1024) + corr_adj {
                        singular_ext = 3;
                    }
                }
            } else if se_score >= beta {
                // Multi-cut: alternatives also beat beta — prune
                st!(td, s, s.se_multicut += 1;);
                tree_ret!(td, ply, crate::tree::X_SE_MULTICUT, depth, se_score);
            } else if tt_score >= beta {
                // Not singular, TT fail-high — negative extension
                singular_ext = -3 + NT::PV as i32;
            } else if cut_node {
                // Not singular at cut-node — negative extension
                singular_ext = -2;
            }
            st!(td, s, {
                match singular_ext {
                    1 => s.se_ext1 += 1,
                    2 => s.se_ext2 += 1,
                    3 => s.se_ext3 += 1,
                    x if x < 0 => s.se_negext += 1,
                    _ => {}
                }
            });
        }

        // Capture extension: extend recaptures on the same square as previous move
        // Extend only in PV nodes, only for TT move, only if recaptures on prev sq.
        // ply >= 1 guard: at root there is no previous move (in release the old
        // ply-1 underflow happened to land on a sentinel entry; this guard makes
        // that explicit and keeps debug builds alive).
        let recapture_ext = if singular_ext == 0 && NT::PV && m == tt_move && !is_quiet_move && ply >= 1 {
            let prev = td.ss(ply - 1);
            i32::from(prev.is_capture && m.to_sq() == prev.played_move.to_sq())
        } else {
            0
        };

        // Record move context on the stack (AFTER SE to avoid inner search overwriting it)
        {
            let conthist_ptr = td.cont_history.subtable_ptr(!is_quiet_move, moved_piece, m.to_sq());
            let cont_corr_ptr = td.cont_correction.subtable_ptr(moved_piece, m.to_sq());
            let se = td.ss_mut(ply);
            se.played_move = m;
            se.moved_piece = moved_piece;
            se.is_capture = !is_quiet_move;
            se.conthist_ptr = conthist_ptr;
            se.cont_corr_ptr = cont_corr_ptr;
        }

        // Node fraction TM: snapshot before recursive search
        let nodes_before = if NT::ROOT { td.nodes } else { 0 };

        // Move category for tree records must be computed BEFORE make_move
        // (classification reads the pre-move board and runs SEE).
        #[cfg(feature = "tree")]
        let tree_cat: u8 = if td.tree.as_deref().is_some_and(|t| t.recording()) {
            crate::stats::move_category(&td.pos, m, tt_move, killers, countermove) as u8
        } else {
            255
        };

        if nnue::network::has_network() {
            td.nnue.push(m, &td.pos);
        }
        td.pos.make_move(m);
        let gives_check = td.pos.in_check();
        let total_ext = singular_ext + recapture_ext; // SE + recapture extension; no check ext
        let mut new_depth = (depth - 1 + total_ext).max(0);

        // Propagate double extension counter to child ply
        {
            let parent_de = if ply > 0 { td.ss(ply - 1).double_extensions } else { 0 };
            td.ss_mut(ply).double_extensions = parent_de + if singular_ext >= 2 { 1 } else { 0 };
        }

        // PVS + LMR search logic
        let score;
        if NT::PV && move_count == 1 {
            // First move at PV node: full window, full depth
            tr!(td, t, t.mv(ply as u8, m.0, move_count, tree_cat, crate::tree::A_SEARCH_FULL, -total_ext * 1024, new_depth););
            score = -alpha_beta::<Pv>(td, shared, -beta, -alpha, new_depth, ply + 1, false, false);
        } else {
            let mut s;

            // LMR: centipawn reduction system (1024cp = 1 ply)
            if depth >= 2 && move_count > 1 {
                debug_assert!(move_count >= 2);
                debug_assert!(depth >= 2);
                let ln_mc = LMR_LN[(move_count as usize).min(63)];
                let ln_d = LMR_LN[(depth as usize).min(63)];
                let mut r = tune::LMR_LOG_MUL() * ln_mc * ln_d / 1024 + tune::LMR_LOG_BASE();

                if is_quiet_move {
                    r += tune::LMR_QUIET_BASE();
                    r -= tune::LMR_QUIET_HIST_MUL() * combined_history / 1024;
                } else {
                    r += tune::LMR_CAPTURE_BASE();
                    r -= tune::LMR_CAPTURE_HIST_MUL() * combined_history / 1024;
                }

                // PV: reduce less
                if NT::PV {
                    r -= tune::LMR_PV_BASE() + tune::LMR_PV_WINDOW_MUL() * (beta - alpha) / td.root_delta;
                }

                // tt_pv: reduce less (node was PV in prior iteration)
                if tt_pv {
                    r -= tune::LMR_TTPV_BASE();
                    r -= tune::LMR_TTPV_SCORE_MUL() * (tt_score != SCORE_NONE && tt_score > alpha) as i32;
                    r -= tune::LMR_TTPV_DEPTH_MUL() * (tt_score != SCORE_NONE && tt_depth >= depth) as i32;
                    // Depth-dependent linear term: reduce ex-PV nodes less as the
                    // remaining depth grows, since over-reducing a deep ex-PV line
                    // costs exponentially more. Scale is 1024 = 1 ply.
                    debug_assert!(depth >= 2);
                    r -= tune::LMR_TTPV_DEPTH_LIN() * depth;
                }

                // Cut node: reduce more
                if !tt_pv && cut_node {
                    r += tune::LMR_CUTNODE_BASE() + tune::LMR_CUTNODE_DEPTH() * depth / 1024;
                    r += tune::LMR_CUTNODE_NOTM_MUL() * (tt_move == Move::NONE) as i32;
                }

                // Not improving: reduce more with magnitude
                if !improving {
                    r += (tune::LMR_NOT_IMPROVING_BASE() + tune::LMR_NOT_IMPROVING_DEPTH() * depth / 1024 - tune::LMR_NOT_IMPROVING_MUL() * improvement / 128).min(tune::LMR_NOT_IMPROVING_MAX());
                }

                // Gives check: reduce less (~1.3 ply)
                if gives_check {
                    r -= tune::LMR_CHECK_SUB();
                }

                // Recapture reduction: reduce less for noisy recaptures.
                // ply >= 1 guard: at root there is no previous move (in release
                // the old ply-1 underflow happened to land on a sentinel entry;
                // this guard makes that explicit and keeps debug builds alive).
                if !is_quiet_move && ply >= 1 {
                    let prev = td.ss(ply - 1);
                    if prev.is_capture && m.to_sq() == prev.played_move.to_sq() {
                        r -= tune::RECAPTURE_LMR_BONUS();
                    }
                }

                // Bad TT score: reduce more
                if tt_score != SCORE_NONE && tt_score < alpha {
                    r += tune::LMR_BAD_TT_SCORE();
                }

                // Shallow TT depth is less reliable: reduce more
                r += tune::LMR_TT_SHALLOW_DEPTH() * (tt_score != SCORE_NONE && tt_depth < depth) as i32;

                // Correction history: reduce less when the static evaluation is
                // unreliable (large |correction_value| → hard-to-evaluate
                // position → search deeper)
                r -= tune::LMR_CORR_HIST_MUL() * correction_value.abs() / 1024;

                // LMR overextension for heavily boosted early moves:
                // if r is very negative (many accumulated bonuses) and move_count is
                // below a threshold, let reduced_depth reach new_depth+2 instead of new_depth+1.
                let lmr_overext = (r < tune::LMR_OVEREXT_THRESHOLD() && move_count <= tune::LMR_OVEREXT_MAX_MC() as u32) as i32;
                debug_assert!(lmr_overext == 0 || lmr_overext == 1);
                let reduced_depth = (new_depth - r / 1024).clamp(1, new_depth + 1 + lmr_overext);

                // Store reduction in plies for hindsight adjustment in child nodes
                td.ss_mut(ply).reduction = new_depth - reduced_depth;
                st!(td, s, {
                    s.lmr_searches += 1;
                    if reduced_depth < new_depth {
                        s.lmr_reduced += 1;
                    }
                });
                tr!(td, t, t.mv(ply as u8, m.0, move_count, tree_cat, crate::tree::A_SEARCH_LMR, r, reduced_depth););
                s = -alpha_beta::<NonPv>(td, shared, -alpha - 1, -alpha, reduced_depth, ply + 1, false, true);
                td.ss_mut(ply).reduction = 0;

                // Adaptive re-search: adjust depth based on score vs best
                if s > alpha {
                    if !NT::ROOT {
                        // Two-tiered do-deeper: extend the re-search when the score beats
                        // best by a margin — tier 1 = moderate margin, tier 2 = wide margin for major tactical moves
                        st!(td, st_, {
                            st_.do_deeper += (s > best_score + tune::DEEPER_BASE()) as u64
                                + (s > best_score + tune::DEEPER2_BASE()) as u64;
                            st_.do_shallower +=
                                (s < best_score + tune::SHALLOWER_BASE() + reduced_depth) as u64;
                        });
                        new_depth += (s > best_score + tune::DEEPER_BASE()) as i32;
                        new_depth += (s > best_score + tune::DEEPER2_BASE()) as i32;
                        // Reduced-depth shallower: threshold proportional to
                        // reduced_depth — less aggressive if the move was already heavily reduced
                        new_depth -= (s < best_score + tune::SHALLOWER_BASE() + reduced_depth) as i32;
                    }
                    if new_depth > reduced_depth {
                        st!(td, st_, st_.lmr_research += 1;);
                        tr!(td, t, t.mv(ply as u8, m.0, move_count, tree_cat, crate::tree::A_RESEARCH_FULL, 0, new_depth););
                        s = -alpha_beta::<NonPv>(td, shared, -alpha - 1, -alpha, new_depth, ply + 1, false, !cut_node);
                    }
                }
            } else {
                // No LMR: full-depth null-window search
                tr!(td, t, t.mv(ply as u8, m.0, move_count, tree_cat, crate::tree::A_SEARCH_FULL, -total_ext * 1024, new_depth););
                s = -alpha_beta::<NonPv>(td, shared, -alpha - 1, -alpha, new_depth, ply + 1, false, !cut_node);
            }

            // PV re-search with full window
            if NT::PV && s > alpha {
                st!(td, st_, st_.pvs_research += 1;);
                tr!(td, t, t.mv(ply as u8, m.0, move_count, tree_cat, crate::tree::A_RESEARCH_PV, 0, new_depth););
                s = -alpha_beta::<Pv>(td, shared, -beta, -alpha, new_depth, ply + 1, false, false);
            }
            score = s;
        }
        td.pos.unmake_move(m);
        if nnue::network::has_network() {
            td.nnue.pop();
        }

        // Node fraction TM: accumulate nodes for this root move
        if NT::ROOT
            && let Some(rm) = td.root_moves.iter_mut().find(|r| r.mv == m)
        {
            rm.nodes_spent += td.nodes - nodes_before;
        }

        if td.should_stop() {
            tree_ret!(td, ply, crate::tree::X_STOPPED, depth, 0);
        }

        debug_assert!(score > -SCORE_INFINITE && score < SCORE_INFINITE,
            "alpha_beta: invalid score {} from recursion at ply {}", score, ply);

        // Track searched moves for history malus on a later cutoff. Recorded only
        // after the recursive search: moves rejected by the pruning tests above
        // never enter these lists, so they cannot receive an unearned malus.
        if is_quiet_move && quiet_count < MAX_QUIETS_SEARCHED {
            quiets_searched[quiet_count] = m;
            quiet_count += 1;
        }
        if !is_quiet_move && capture_count < MAX_CAPTURES_SEARCHED {
            captures_searched[capture_count] = m;
            capture_count += 1;
        }

        if score > best_score {
            best_score = score;
            best_move = m;

            if score > alpha {
                alpha = score;

                // Update PV (only at PV nodes)
                if NT::PV {
                    td.pv[ply][0] = m;
                    let child_len = td.pv_len[ply + 1];
                    let (current, rest) = td.pv[ply..].split_at_mut(1);
                    current[0][1..=child_len].copy_from_slice(&rest[0][..child_len]);
                    td.pv_len[ply] = child_len + 1;
                }

                if score >= beta {
                    st!(td, s, {
                        s.cutoffs += 1;
                        if move_count == 1 {
                            s.cutoff_first += 1;
                        }
                        s.fail_high_index[(move_count as usize - 1).min(63)] += 1;
                        let cat = crate::stats::move_category(
                            &td.pos, m, tt_move, killers, countermove,
                        );
                        s.cutoff_by_cat[cat] += 1;
                    });
                    let mut bonus = stat_bonus(depth);
                    let malus = stat_malus(depth);

                    // Non-PV best move history bonus move-count scaling:
                    // at non-PV nodes, amplify the bonus in proportion to the
                    // number of moves tried before the cutoff. The more we searched,
                    // the more "surprising" the best move is and the stronger the signal it deserves.
                    // Cast to i64 to avoid intermediate overflow.
                    if !NT::PV {
                        let total_searched = (quiet_count + capture_count) as i64;
                        debug_assert!(total_searched >= 0);
                        bonus += (bonus as i64 * total_searched / tune::BESTMOVE_MC_DIV() as i64) as i32;
                    }

                    if is_quiet_move {
                        // Killer + countermove update
                        td.killers.update(ply, m);
                        if ply >= 1 {
                            let prev = td.ss(ply - 1);
                            if prev.played_move != Move::NONE {
                                td.countermoves.update(
                                    prev.moved_piece,
                                    prev.played_move.to_sq(),
                                    m,
                                );
                            }
                        }

                        // Butterfly + pawn history bonus
                        td.history.update(stm, m, td.pos.threats, bonus);
                        td.pawn_history.update(td.pos.pawn_key, moved_piece, m.to_sq(), bonus);
                        // Continuation history bonus
                        update_continuation_histories(td, ply, moved_piece, m.to_sq(), bonus, in_check);

                        // Malus for other tried quiets
                        for &q in quiets_searched.iter().take(quiet_count) {
                            if q != m {
                                let q_piece = td.pos.board[q.from_sq().index()];
                                td.history.update(stm, q, td.pos.threats, -malus);
                                td.pawn_history.update(td.pos.pawn_key, q_piece, q.to_sq(), -malus);
                                update_continuation_histories(td, ply, q_piece, q.to_sq(), -malus, in_check);
                            }
                        }
                    } else {
                        // Capture cutoff: bonus for the best capture
                        td.cap_history.update(moved_piece, m.to_sq(), captured_pt, bonus);
                    }

                    // Malus for all non-best captures that were tried
                    for &cap_m in captures_searched.iter().take(capture_count) {
                        if cap_m != m {
                            let cap_piece = td.pos.board[cap_m.from_sq().index()];
                            let cap_captured_pt = get_captured_pt(&td.pos, cap_m);
                            td.cap_history.update(cap_piece, cap_m.to_sq(), cap_captured_pt, -malus);
                        }
                    }
                    break;
                }
            }
        }
    }

    // Checkmate or stalemate
    if move_count == 0 {
        if excluded != Move::NONE {
            // SE search: the only legal move was the excluded one
            tree_ret!(td, ply, crate::tree::X_SE_NO_LEGAL, depth, alpha);
        }
        // Verify no legal moves exist (catches movegen/legality bugs)
        #[cfg(debug_assertions)]
        {
            let mut check_buf: ArrayBuf<Move, MAX_MOVES> = ArrayBuf::new();
            let check_count = movegen::generate_legal_moves(&td.pos, &mut check_buf);
            debug_assert!(check_count == 0,
                "move_count==0 but {} legal moves exist at ply {} (in_check={})",
                check_count, ply, in_check);
        }
        if in_check {
            tree_ret!(td, ply, crate::tree::X_CHECKMATE, depth, mated_in(ply as i32));
        } else {
            tree_ret!(td, ply, crate::tree::X_STALEMATE, depth, 0);
        }
    }

    // Correction history update during singular extension search:
    // the bound computation and the correction history update are placed BEFORE the
    // `excluded == Move::NONE` guard, so singular extension (SE) searches also
    // update the correction history tables. The TT store stays guarded to avoid
    // polluting the TT with partial SE results.
    let bound = if best_score >= beta {
        Bound::Lower
    } else if NT::PV && best_score > orig_alpha {
        Bound::Exact
    } else {
        Bound::Upper
    };

    // Prior countermove bonus on unexpected all-nodes:
    // bonus to the parent's quiet move when this node fails low unexpectedly —
    // predicted cut_node or PV but resolved as an all-node (bound == Upper).
    if !NT::ROOT && bound == Bound::Upper && (cut_node || NT::PV) && ply >= 1 {
        let idx = ply + SS_OFFSET;
        let prev = &td.stack[idx - 1];
        let prior_move = prev.played_move;
        if prior_move != Move::NONE
            && !prev.is_capture
            && prior_move.move_type() != MT_PROMOTION
            && prior_move.move_type() != MT_EN_PASSANT
        {
            // Butterfly history bonus for the parent's quiet move
            let pcm_bonus = tune::PCM_ALLNODE_FACTOR() * stat_bonus(depth) / 128;
            debug_assert!(pcm_bonus.abs() < 50000, "pcm_bonus out of range: {}", pcm_bonus);
            td.history.update(stm.flip(), prior_move, td.pos.prior_threats(), pcm_bonus);

            // Continuation history bonus via the grandparent (ss-2)
            if idx >= 2 {
                let grandparent = &td.stack[idx - 2];
                if grandparent.played_move != Move::NONE {
                    let ch_bonus = tune::PCM_ALLNODE_CONTHIST_FACTOR() * stat_bonus(depth) / 128;
                    debug_assert!(ch_bonus.abs() < 50000, "pcm ch_bonus out of range: {}", ch_bonus);
                    ContinuationHistory::update(
                        grandparent.conthist_ptr,
                        prev.moved_piece,
                        prior_move.to_sq(),
                        ch_bonus,
                    );
                }
            }
        // Noisy PCM: capture history bonus for the parent's capture move at
        // unexpected all-nodes
        } else if prev.is_capture && prior_move != Move::NONE {
            let cap_piece = td.pos.prior_captured_piece();
            if cap_piece != Piece::NONE {
                let pcm_cap_bonus = tune::PCM_ALLNODE_CAPTURE_FACTOR() * stat_bonus(depth) / 128;
                debug_assert!(pcm_cap_bonus.abs() < 50000, "pcm_cap_bonus out of range: {}", pcm_cap_bonus);
                td.cap_history.update(
                    prev.moved_piece,
                    prior_move.to_sq(),
                    cap_piece.piece_type(),
                    pcm_cap_bonus,
                );
            }
        }
    }

    // Update correction history tables (also during SE searches for more training data)
    // Guards: not in check, not lazy eval (PeSTO), best move quiet or absent, bound direction matches
    if !in_check && !lazy_eval_this_node {
        // Correction history update for losing captures:
        // allow the correction history update when the best move is a losing capture
        // (SEE < 0), since these are structurally similar to quiet moves from the
        // evaluation's point of view.
        let best_is_tactical = best_move != Move::NONE
            && (td.pos.board[best_move.to_sq().index()] != Piece::NONE
                || best_move.move_type() == MT_EN_PASSANT
                || best_move.move_type() == MT_PROMOTION);
        let skip_corr_for_capture = best_is_tactical
            && crate::see::see(&td.pos, best_move, 0);
        let fail_high = bound == Bound::Lower;
        let fail_low = bound == Bound::Upper;

        if !skip_corr_for_capture
            && (!fail_high || best_score > static_eval)
            && (!fail_low || best_score <= static_eval)
        {
            let bonus = ((best_score - static_eval) * depth * tune::CORR_UPDATE_FACTOR() / 1024)
                .clamp(-CORRHIST_LIMIT / 4, CORRHIST_LIMIT / 4);
            td.pawn_correction
                .update(td.pos.pawn_key, td.pos.side_to_move, bonus);
            // Non-pawn correction: update both colors' tables
            td.non_pawn_correction.update(
                td.pos.non_pawn_key[0], Color::White, td.pos.side_to_move, bonus);
            td.non_pawn_correction.update(
                td.pos.non_pawn_key[1], Color::Black, td.pos.side_to_move, bonus);
            // Minor piece correction
            td.minor_correction
                .update(td.pos.minor_key, td.pos.side_to_move, bonus);
            // Continuation correction: update subtables at offsets 2 and 4
            let base = ply + SS_OFFSET;
            for &offset in &[2, 4] {
                if base > offset {
                    let prev = &td.stack[base - offset];
                    let last = &td.stack[base - 1];
                    if prev.played_move != Move::NONE && last.played_move != Move::NONE {
                        ContCorrectionHistory::update(
                            prev.cont_corr_ptr,
                            last.moved_piece,
                            last.played_move.to_sq(),
                            bonus,
                        );
                    }
                }
            }
        }
    }

    // Store in TT (skip during SE search to avoid polluting with partial results)
    if excluded == Move::NONE {
        // If alpha wasn't raised (fail-low), best_move may be NONE
        debug_assert!(best_score > orig_alpha || bound == Bound::Upper,
            "alpha not raised but bound is {:?}", bound);
        // If we have an exact or lower bound, we must have a best move
        debug_assert!(bound == Bound::Upper || best_move != Move::NONE,
            "bound {:?} but no best_move at ply {}", bound, ply);

        // TT stores raw eval (not corrected) so correction is re-applied on future probes
        debug_assert!(best_score > -SCORE_INFINITE, "TT store: -INF score");
        debug_assert!(best_score < SCORE_INFINITE, "TT store: +INF score");
        debug_assert!(raw_eval == SCORE_NONE || raw_eval.abs() < SCORE_INFINITE,
            "TT store: raw_eval {} suspicious", raw_eval);
        debug_assert!(depth >= 0, "TT store: negative depth {}", depth);
        debug_assert!(best_move == Move::NONE || best_move.from_sq().0 < 64,
            "TT store: best_move from OOB {}", best_move.from_sq().0);
        st!(td, s, s.tt_stores += 1;);
        shared.tt.store(td.pos.key, depth, raw_eval, best_score, bound, best_move, ply as i32, tt_pv);
    }

    // Clamp to TB upper bound
    let final_score = best_score.min(syzygy_max);
    tr!(td, t, t.exit(ply as u8, crate::tree::X_NORMAL, crate::tree::bound_bits(bound), best_move.0, final_score, move_count, depth););
    final_score
}

/// Value of the material gained by a capture move (for delta pruning).
#[inline]
fn capture_value(pos: &Position, m: Move) -> i32 {
    let mt = m.move_type();
    if mt == MT_EN_PASSANT {
        return PIECE_VALUE[PieceType::Pawn as usize];
    }
    if mt == MT_PROMOTION {
        // Promotion value (even if also a capture, promo piece is the big gain)
        return PIECE_VALUE[m.promo_type() as usize];
    }
    let victim = pos.board[m.to_sq().index()];
    if victim == Piece::NONE {
        return 0;
    }
    PIECE_VALUE[victim.piece_type() as usize]
}

/// Quiescence search: resolve tactical sequences (captures, promotions, check
/// evasions) to avoid the horizon effect — where a fixed-depth search misses a
/// capture just beyond its horizon, returning a misleading eval.
///
/// Key differences from main search:
/// - No null move, no LMR, no extensions
/// - Stand-pat: use static eval as lower bound (not when in check)
/// - In check: generate ALL evasions (captures + quiets)
/// - Not in check: generate captures + queen promotions only
///
/// Reference: CPW — [Quiescence Search](https://www.chessprogramming.org/Quiescence_Search)
fn quiescence<NT: NodeType>(
    td: &mut ThreadData,
    shared: &SharedState,
    mut alpha: i32,
    mut beta: i32,
    ply: usize,
) -> i32 {
    debug_assert!(alpha < beta, "qsearch: alpha {} >= beta {}", alpha, beta);
    debug_assert!(alpha >= -SCORE_INFINITE && beta <= SCORE_INFINITE,
        "qsearch: bounds OOB alpha={} beta={}", alpha, beta);
    debug_assert!(ply < MAX_PLY, "qsearch: ply {} >= MAX_PLY", ply);
    debug_assert!(NT::PV || alpha == beta - 1,
        "qsearch: non-PV with non-null window [{}, {}]", alpha, beta);
    debug_assert!(!NT::ROOT, "qsearch: called with ROOT node type at ply {}", ply);

    td.check_limits();
    if td.should_stop() {
        return 0;
    }

    // Ply overflow guard
    if ply >= MAX_PLY - 1 {
        return if td.pos.in_check() { 0 } else { evaluate_pos(td) };
    }

    // Upcoming repetition in qsearch
    if alpha < 0 && td.pos.upcoming_repetition(ply) {
        alpha = 0;
        if alpha >= beta {
            return alpha;
        }
    }

    // Draw detection in qsearch
    if td.pos.is_draw(ply as i32) {
        return 0;
    }

    // GaiaTB DTM probe in qsearch: positions ≤4 pieces get exact DTM scores.
    // Skip EP positions (tables don't encode EP).
    #[cfg(feature = "gaiatb")]
    if td.pos.castling_rights == 0
        && td.pos.ep_square == Square::NONE
        && td.pos.occupied().count_ones() <= 4
        && crate::dtm::available()
        && let Some(score) = crate::dtm::probe_position(&td.pos, ply as i32)
    {
        return score;
    }

    td.nodes += 1;
    td.qs_nodes += 1;
    td.seldepth = td.seldepth.max(ply as i32 + 1);
    st!(td, s, s.qs_nodes[!NT::PV as usize] += 1;);
    tr!(td, t, {
        let mut f = crate::tree::F_QS;
        if NT::PV { f |= crate::tree::F_PV; }
        t.enter(ply as u8, 0, f, alpha, beta);
    });

    // Prefetch TT cluster into L1 cache
    shared.tt.prefetch(td.pos.key);

    // TT probe in quiescence: reuse cached scores and eval. At non-PV nodes,
    // a sufficient-depth TT entry can produce an immediate cutoff. At PV nodes,
    // skip the cutoff to preserve the full PV line but still reuse the eval.
    // (CPW: Transposition Table)
    let tt_move;
    let mut tt_eval = SCORE_NONE;
    let mut tt_score = SCORE_NONE;
    let mut tt_bound = Bound::None;
    // Propagate the PV flag from the TT (mirrors the TT probe in alpha_beta)
    let mut tt_pv = NT::PV;
    if let Some(hit) = shared.tt.probe(td.pos.key, ply as i32, td.pos.halfmove_clock) {
        tt_move = hit.mv;
        tt_eval = hit.eval;
        tt_score = hit.score;
        tt_bound = hit.bound;
        tt_pv |= hit.pv;
        // TT cutoff: skip at PV nodes, skip at high halfmove_clock (GHI fix)
        if !NT::PV && td.pos.halfmove_clock < 96 {
            let cutoff = match hit.bound {
                Bound::Exact => true,
                Bound::Lower => hit.score >= beta,
                Bound::Upper => hit.score <= alpha,
                _ => false,
            };
            if cutoff {
                st!(td, s, s.qs_tt_cutoffs += 1;);
                tree_ret!(td, ply, crate::tree::X_QS_TT_CUTOFF, 0, hit.score);
            }
        }
    } else {
        tt_move = Move::NONE;
    }

    let in_check = td.pos.in_check();

    // Stand-pat: skip when in check
    let stand_pat;
    let raw_eval: i32;
    let mut best_score;
    let mut best_move = Move::NONE;
    if in_check {
        stand_pat = -SCORE_INFINITE;
        raw_eval = SCORE_NONE;
        best_score = -SCORE_INFINITE;
    } else {
        // Reuse TT eval if available (avoids NNUE forward pass)
        td.used_lazy_eval = false;
        raw_eval = if tt_eval != SCORE_NONE { tt_eval } else { evaluate_pos(td) };
        let lazy_eval_qs = td.used_lazy_eval || td.skill.active;
        // Ensure NNUE threat features are up-to-date even when TT eval is used,
        // so children can do incremental threat updates instead of full recomputes.
        if tt_eval != SCORE_NONE && nnue::network::has_network() {
            td.nnue.ensure_updated(&td.pos);
        }
        // Lazy eval: PeSTO score used, skip correction history (trained against NNUE)
        let corr_val = if lazy_eval_qs {
            0
        } else {
            // Stand-pat with correction history
            let qs_stm = td.pos.side_to_move;
            let pawn_entry = td.pawn_correction.get(td.pos.pawn_key, qs_stm) as i64;
            let non_pawn_entry = td.non_pawn_correction.get(td.pos.non_pawn_key[0], Color::White, qs_stm) as i64
                + td.non_pawn_correction.get(td.pos.non_pawn_key[1], Color::Black, qs_stm) as i64;
            let minor_entry = td.minor_correction.get(td.pos.minor_key, qs_stm) as i64;
            let cont_corr = {
                let base = ply + SS_OFFSET;
                let mut cc = 0i64;
                for &offset in &[2, 4] {
                    if base > offset {
                        let prev = &td.stack[base - offset];
                        let last = &td.stack[base - 1];
                        if prev.played_move != Move::NONE && last.played_move != Move::NONE {
                            cc += ContCorrectionHistory::get(
                                prev.cont_corr_ptr as *const _,
                                last.moved_piece,
                                last.played_move.to_sq(),
                            ) as i64;
                        }
                    }
                }
                cc
            };
            (pawn_entry * tune::PAWN_CORR_FACTOR() as i64 / CORR_DIVISOR
                + non_pawn_entry * tune::NON_PAWN_CORR_FACTOR() as i64 / CORR_DIVISOR
                + minor_entry * tune::MINOR_CORR_FACTOR() as i64 / CORR_DIVISOR
                + cont_corr * tune::CONT_CORR_FACTOR() as i64 / CORR_DIVISOR) as i32
        };
        // Optimism blending in qsearch (skip for lazy eval / PeSTO)
        let blended_qs = if lazy_eval_qs {
            raw_eval
        } else {
            let qs_stm = td.pos.side_to_move;
            let material = td.pos.material() / 32;
            let mat_mul = (tune::OPT_MAT_SCALE() + material) as i64;
            let opt_mul = (tune::OPT_MAT_BASE() + material) as i64;
            ((raw_eval as i64 * mat_mul
                + td.optimism[qs_stm.index()] as i64 * opt_mul / 32)
                / 1024) as i32
        };
        // 50-move rule scaling: scale eval toward 0 as halfmove clock approaches 100,
        // reflecting increasing draw probability. (CPW: Fifty-move Rule)
        let scaled_qs = blended_qs as i64
            * (tune::FIFTY_MOVE_SCALE() - td.pos.halfmove_clock as i32) as i64
            / tune::FIFTY_MOVE_SCALE() as i64;
        stand_pat = (scaled_qs as i32 + corr_val)
            .clamp(-SCORE_MATE_IN_MAX + 1, SCORE_MATE_IN_MAX - 1);
        debug_assert!(stand_pat.abs() < SCORE_MATE_IN_MAX,
            "qsearch: stand_pat {} looks like mate score", stand_pat);
        best_score = stand_pat;

        // TT stand-pat blending: use TT score as better position estimate when
        // the bound direction agrees.
        if tt_score != SCORE_NONE && !is_mate_score(tt_score) {
            let use_tt = match tt_bound {
                Bound::Exact => true,
                Bound::Lower => tt_score > best_score,
                Bound::Upper => tt_score < best_score,
                _ => false,
            };
            if use_tt {
                best_score = tt_score;
            }
        }

        if best_score >= beta {
            st!(td, s, s.qs_standpat_cutoffs += 1;);
            // Beta midpoint: reduce overestimation (½ blending)
            let ret = if !is_mate_score(best_score) { (best_score + beta) / 2 } else { best_score };
            // Only store to TT if no prior hit.
            // Overwriting a deeper entry with depth-0 stand-pat pollutes the TT.
            if tt_score == SCORE_NONE {
                shared.tt.store(td.pos.key, 0, raw_eval, ret, Bound::Lower, Move::NONE, ply as i32, tt_pv);
            }
            tree_ret!(td, ply, crate::tree::X_QS_STANDPAT, 0, ret);
        }
        if best_score > alpha {
            alpha = best_score;
        }
    }

    // Mate distance pruning: tighten the window
    // when a forced mate is already known within a few plies. If we can't beat
    // the best known mating sequence, prune immediately. (CPW: Mate Distance Pruning)
    alpha = alpha.max(mated_in(ply as i32));
    beta = beta.min(mate_in(ply as i32 + 1));
    if alpha >= beta {
        tree_ret!(td, ply, crate::tree::X_QS_MATE_DISTANCE, 0, alpha);
    }

    let stm = td.pos.side_to_move;
    // When in check: generate ALL evasions (captures + quiets), not just captures
    let mut picker = if in_check {
        MovePicker::new(tt_move, [Move::NONE; 2], Move::NONE, 0)
    } else {
        MovePicker::new_qsearch(tt_move)
    };

    let mut move_count = 0u32;
    let mut skip_quiet_evasions = false;

    // Futility base: precomputed outside loop for unified pruning
    let futility_base = if !in_check { stand_pat + tune::QS_FUTILITY_BASE() } else { -SCORE_INFINITE };

    // Recapture square: exempt recaptures from pruning.
    // Only set when previous move was a capture.
    let recapture_sq = {
        let prev_ss = td.ss(ply - 1);
        if prev_ss.is_capture { prev_ss.played_move.to_sq() } else { Square::NONE }
    };

    loop {
        let m = if in_check {
            // In check: real ply for conthist evasion ordering
            picker.next::<false>(&td.pos, &td.history, &td.cap_history, &td.pawn_history, &shared.tt, ply, &td.stack, stm)
        } else {
            // Not in check: ply irrelevant since QSEARCH=true eliminates quiet stages
            picker.next::<true>(&td.pos, &td.history, &td.cap_history, &td.pawn_history, &shared.tt, 0, &td.stack, stm)
        };
        if m == Move::NONE {
            break;
        }

        if !movegen::is_legal(&td.pos, m) {
            continue;
        }

        // The same blind spots the main search has (see the move loop there). This is
        // the one that matters most: a beginner who resolved every capture to the end
        // would never leave a piece en prise, however badly it judged the result.
        // Evasions are exempt, which also leaves the mate test below untouched.
        if td.skill.blind
            && move_count >= 1
            && !in_check
            && !skill::sees_move(&td.skill, td.pos.key, m, ply, easy_to_notice(td, m, ply))
        {
            continue;
        }

        move_count += 1;

        // Skip quiet TT moves in qsearch when not in check.
        // The TTMove stage can yield quiets stored from deeper searches; they have
        // no tactical value in qsearch and waste nodes if searched recursively.
        if !in_check && is_quiet(&td.pos, m) {
            continue;
        }

        let is_recapture = m.to_sq() == recapture_sq;

        // Skip remaining quiet evasions after one improved position.
        // Prevents check chain explosion in Q vs Q endgames.
        let is_quiet_move = is_quiet(&td.pos, m);
        if in_check && skip_quiet_evasions && is_quiet_move {
            continue;
        }

        // Compute moved_pc early for history and the stack
        let moved_pc = td.pos.board[m.from_sq().index()];
        debug_assert!(moved_pc != Piece::NONE, "qsearch: moved_pc is NONE for move {:?}", m);

        // QSearch SEE history-adjusted margin:
        // adjust the qsearch SEE threshold by the move's history score —
        // good history → lower threshold (more permissive), bad → higher threshold (pruning).
        let qs_history = if is_quiet_move {
            // Quiet evasion (only when in_check)
            td.history.get(stm, m, td.pos.threats)
                + td.conthist(ply, 1, moved_pc, m.to_sq())
                + td.conthist(ply, 2, moved_pc, m.to_sq())
        } else {
            let captured_pt = get_captured_pt(&td.pos, m);
            td.cap_history.get(moved_pc, m.to_sq(), captured_pt)
        };

        // Unified futility pruning: two-stage check merging
        // delta + futility. Stage 1 prunes by material, stage 2 by SEE with adaptive
        // threshold. Exempt recaptures (tactically sharp). (CPW: Futility Pruning)
        if !in_check && !is_recapture && best_score > -SCORE_MATE_IN_MAX {
            let futility_value = futility_base + capture_value(&td.pos, m);
            // Stage 1: material futility — capture can't raise score to alpha
            if futility_value <= alpha {
                best_score = best_score.max(futility_value);
                tr!(td, t, if t.qs_moves { t.mv(ply as u8, m.0, move_count, 255, crate::tree::A_QS_PRUNED, 0, 0); });
                continue;
            }
            // Stage 2: SEE futility with adaptive threshold
            if !crate::see::see(&td.pos, m, alpha - futility_base) {
                best_score = best_score.max(futility_base.min(alpha));
                tr!(td, t, if t.qs_moves { t.mv(ply as u8, m.0, move_count, 255, crate::tree::A_QS_PRUNED, 1, 0); });
                continue;
            }
        }

        // SEE pruning in qsearch: threshold adjusted by the move's history score.
        // (CPW: Static Exchange Evaluation)
        if best_score > -SCORE_MATE_IN_MAX
            && !crate::see::see(&td.pos, m, tune::QS_SEE_THRESHOLD() - qs_history / tune::QS_SEE_HIST_DIV())
        {
            tr!(td, t, if t.qs_moves { t.mv(ply as u8, m.0, move_count, 255, crate::tree::A_QS_PRUNED, 2, 0); });
            continue;
        }

        // Move count limit in qsearch: at non-PV nodes, stop after a few captures to
        // prevent search explosion. Exempt checks and recaptures (tactically sharp).
        // (qsearch move count limiting)
        if !NT::PV && !is_recapture && move_count >= tune::QS_MOVE_COUNT_LIMIT() as u32 && !td.pos.gives_check(m) {
            break;
        }

        // Record move on stack so child nodes can read recapture_sq and corr history
        let is_cap = !is_quiet_move;
        let cont_corr_ptr = td.cont_correction.subtable_ptr(moved_pc, m.to_sq());
        let se = td.ss_mut(ply);
        se.played_move = m;
        se.moved_piece = moved_pc;
        se.is_capture = is_cap;
        se.cont_corr_ptr = cont_corr_ptr;

        if nnue::network::has_network() {
            td.nnue.push(m, &td.pos);
        }
        td.pos.make_move(m);
        tr!(td, t, if t.qs_moves { t.mv(ply as u8, m.0, move_count, 255, crate::tree::A_QS_SEARCH, 0, 0); });
        let score = -quiescence::<NT>(td, shared, -beta, -alpha, ply + 1);
        td.pos.unmake_move(m);
        if nnue::network::has_network() {
            td.nnue.pop();
        }

        if td.should_stop() {
            tree_ret!(td, ply, crate::tree::X_STOPPED, 0, 0);
        }

        if score > best_score {
            best_score = score;
            best_move = m;

            // Once a quiet evasion improves position, skip remaining quiets
            if in_check && best_score > -SCORE_MATE_IN_MAX && is_quiet(&td.pos, m) {
                skip_quiet_evasions = true;
            }

            if score > alpha {
                alpha = score;
                if score >= beta {
                    st!(td, s, {
                        s.qs_cutoffs += 1;
                        if move_count == 1 {
                            s.qs_cutoff_first += 1;
                        }
                    });
                    break;
                }
            }
        }
    }

    // Checkmate detection: in check with no legal evasions
    if in_check && move_count == 0 {
        // Verify it's a real checkmate
        #[cfg(debug_assertions)]
        {
            let mut check_buf: ArrayBuf<Move, MAX_MOVES> = ArrayBuf::new();
            let check_count = movegen::generate_legal_moves(&td.pos, &mut check_buf);
            debug_assert!(check_count == 0,
                "qsearch: mated_in but {} legal moves exist at ply {}",
                check_count, ply);
        }
        tree_ret!(td, ply, crate::tree::X_QS_MATED, 0, mated_in(ply as i32));
    }

    // Qsearch should never return -INF unless in check
    debug_assert!(best_score > -SCORE_INFINITE || in_check,
        "qsearch: -INF score without being in check at ply {}", ply);

    // Beta midpoint: reduce overestimation (½ blending)
    if best_score >= beta && !is_mate_score(best_score) {
        best_score = (best_score + beta) / 2;
    }

    // Store to TT
    let bound = if best_score >= beta { Bound::Lower } else { Bound::Upper };
    shared.tt.store(td.pos.key, 0, raw_eval, best_score, bound, best_move, ply as i32, tt_pv);

    tr!(td, t, t.exit(ply as u8, crate::tree::X_QS_NORMAL, crate::tree::bound_bits(bound), best_move.0, best_score, move_count, 0););
    best_score
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::threads::SharedState;
    use crate::timeman::SearchLimits;

    fn search_depth(fen: &str, depth: i32) -> (Move, i32) {
        // Full strength is a global, and the tests that weaken it run in parallel with
        // this one: without the lock they would hand this search a handicap.
        let _guard = skill::lock_level();
        let pos = Position::from_fen(fen).unwrap();
        let shared = SharedState::new(4);
        let mut td = ThreadData::new(0);
        td.prepare_search(&pos, &SearchLimits::Depth(depth));
        shared.tt.new_search();
        search(&mut td, &shared);
        let best = td.best_move;
        let score = td.best_score;
        (best, score)
    }

    /// Runs a search at a given level, leaving full strength in force afterwards.
    ///
    /// The caller must hold [`skill::lock_level`]: the level is one global for the whole
    /// process and the test harness runs tests in parallel.
    fn search_at_level(fen: &str, level: i32, seed: u64, depth: i32) -> (Move, i32, u64) {
        let pos = Position::from_fen(fen).unwrap();
        let shared = SharedState::new(4);
        let mut td = ThreadData::new(0);
        skill::set(level, seed);
        td.prepare_search(&pos, &SearchLimits::Depth(depth));
        shared.tt.new_search();
        search(&mut td, &shared);
        skill::set(skill::FULL_STRENGTH, 0);
        (td.best_move, td.best_score, td.nodes)
    }

    /// The one thing a handicap must never do: talk the search into believing a position
    /// is over when it is not. Blindness skips moves, so if it ever skipped the last one
    /// the search would report a mate or a stalemate that is not on the board.
    #[test]
    fn overlooking_moves_never_invents_a_mate_or_a_stalemate() {
        let _guard = skill::lock_level();
        // Two positions with a single legal move each — the case that would break first
        // if the last move could ever be skipped — and two ordinary ones that are
        // neither mate nor stalemate and must not be reported as either.
        let corners = [
            "7k/6Q1/8/8/8/8/8/K7 b - - 0 1",  // only Kxg7
            "7K/8/8/8/8/8/1Q6/k7 b - - 0 1",  // only Kxb2
            "4k3/8/4K3/4P3/8/8/8/8 b - - 0 1",
            "6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1",
        ];
        for fen in corners {
            let (legal, only) = {
                let pos = Position::from_fen(fen).unwrap();
                let mut buf = ArrayBuf::<Move, MAX_MOVES>::new();
                let n = movegen::generate_legal_moves(&pos, &mut buf);
                (n, buf[0])
            };
            assert!(legal > 0, "{fen} has no legal move: pick another position");

            for level in [1, 2, 3, 5, 8] {
                for seed in [0, 0x1234, 0xdead_beef, 0x5eed_5eed] {
                    let (m, score, _) = search_at_level(fen, level, seed, 4);
                    assert!(m.is_ok(), "level {level} seed {seed} found no move in {fen}");
                    if legal == 1 {
                        assert_eq!(m, only, "level {level} seed {seed} missed the only move in {fen}");
                    }
                    assert!(
                        score.abs() < SCORE_MATE_IN_MAX,
                        "level {level} seed {seed} claimed mate ({score}) in {fen}"
                    );
                }
            }
        }
    }

    /// A level plus a seed is one specific opponent, and the same one every time. This is
    /// the property that makes a level mean the same thing on every machine.
    #[test]
    fn a_level_and_a_seed_are_the_same_opponent_twice() {
        let _guard = skill::lock_level();
        let fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1";
        for level in [1, 4, 9, 15] {
            let first = search_at_level(fen, level, 0xabcd, 4);
            let again = search_at_level(fen, level, 0xabcd, 4);
            assert_eq!(first, again, "level {level} played two different games");
        }
        // A different seed is allowed to be a different opponent, and at the weak end
        // where the handicap actually bites it had better be one at least sometimes.
        let seeds: Vec<Move> =
            (0..8).map(|s| search_at_level(fen, 1, 0x1000 * s + 7, 4).0).collect();
        assert!(seeds.iter().any(|&m| m != seeds[0]), "every seed played the same move");
    }

    /// A level looks exactly as far as its rung allows and no further, whatever budget
    /// the caller asked for. Worth its own test because the ceiling is applied where the
    /// root position is known, and anything that installs a time manager afterwards can
    /// quietly drop it.
    #[test]
    fn a_level_never_looks_further_than_its_rung_allows() {
        let _guard = skill::lock_level();
        let fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1";
        let pos = Position::from_fen(fen).unwrap();
        for level in [1, 4, 8, 12, 16, 19] {
            let allowed = skill::depth_for(level);
            for seed in 0..8u64 {
                let shared = SharedState::new(4);
                let mut td = ThreadData::new(0);
                skill::set(level, seed * 0x9E37);
                // Ask for far more than the rung permits: the rung must still win.
                td.prepare_search(&pos, &SearchLimits::Depth(MAX_PLY as i32));
                shared.tt.new_search();
                search(&mut td, &shared);
                assert!(
                    td.completed_depth <= allowed,
                    "level {level} reached depth {} with a ceiling of {allowed}",
                    td.completed_depth
                );
            }
            skill::set(skill::FULL_STRENGTH, 0);
        }
        // That the extra ply does get taken, and about as often as the rung says, is
        // checked over enough positions to mean something in `skill`: a rung that takes
        // it one position in ten would pass or fail here on the luck of the draw.
    }

    /// Full strength must not merely play well — it must search the identical tree it
    /// searched before any of this existed. The bench is the real check; this is the
    /// cheap one that runs on every `cargo test`.
    #[test]
    fn full_strength_searches_the_same_tree_whatever_the_handicap_was_set_to() {
        let _guard = skill::lock_level();
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        ];
        for fen in fens {
            let (_, _, baseline) = search_at_level(fen, skill::FULL_STRENGTH, 0, 8);
            // Leave a weakened level in force, then ask for full strength again.
            search_at_level(fen, 1, 0xfeed, 3);
            let (_, _, again) = search_at_level(fen, skill::FULL_STRENGTH, 0xfeed, 8);
            assert_eq!(baseline, again, "full strength changed its tree for {fen}");
        }
    }

    /// Being asked to search a finished game must produce an answer, not a crash.
    ///
    /// No sane interface asks — it has seen the mate itself — but one that does used to
    /// take the engine down with it, because iterative deepening reads `root_moves[0]`
    /// on the way past and there is no such move.
    #[test]
    fn a_finished_game_is_answered_rather_than_crashed_on() {
        // Mate, then stalemate, from both colours.
        let over = [
            ("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3", true),
            ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", false),
            ("k7/P7/K7/8/8/8/8/8 b - - 0 1", false),
        ];
        for (fen, mated) in over {
            let (m, score) = search_depth(fen, 6);
            assert!(!m.is_ok(), "{fen} has no move to play, got {}", m.to_uci());
            if mated {
                assert!(score <= -SCORE_MATE_IN_MAX, "{fen} is mate, scored {score}");
            } else {
                assert_eq!(score, 0, "{fen} is stalemate");
            }
        }
    }

    #[test]
    fn test_lmr_ln_table() {
        // The table is trunc(20.13 * ln(i)) for i >= 1; entry 0 is unused.
        for (i, &entry) in LMR_LN.iter().enumerate().skip(1) {
            assert_eq!(entry, (20.13 * (i as f64).ln()) as i32, "LMR_LN[{i}]");
        }
        // Monotone non-decreasing over the used range.
        for i in 2..64usize {
            assert!(LMR_LN[i] >= LMR_LN[i - 1]);
        }
    }

    #[test]
    fn test_startpos_returns_move() {
        let (m, _) = search_depth("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 4);
        assert!(m.is_ok(), "Search should return a valid move");
    }

    #[test]
    fn test_mate_in_1() {
        // White to move, Qh7# is mate in 1
        let (m, _) = search_depth("6k1/5ppp/8/8/8/8/8/4Q2K w - - 0 1", 3);
        assert!(m.is_ok(), "Should find a move in mate-in-1 position");
    }

    #[test]
    fn test_avoid_stalemate() {
        // Simple position where search shouldn't stall
        let (m, _) = search_depth("8/8/8/8/8/5k2/4p3/4K3 w - - 0 1", 3);
        // Just verify it doesn't crash; king has limited moves
        let _ = m;
    }

    #[test]
    fn test_kqk_finds_mate_low_halfmove() {
        // KQ vs K, halfmove_clock = 0: must find mate
        // Ke6, Qa1, black Kd8 — queen doesn't attack king
        let (mv, score) = search_depth("3k4/8/4K3/8/8/8/8/Q7 w - - 0 1", 10);
        assert!(mv.is_ok(), "KQK must find a move");
        assert!(is_mate_score(score), "KQK must find mate score, got {score}");
    }

    #[test]
    fn test_kqk_finds_mate_high_halfmove() {
        // KQ vs K, halfmove_clock = 88: GHI fix must still find the win
        // (this is the exact bug scenario from the tournament game)
        let (mv, score) = search_depth("3k4/8/4K3/8/8/8/8/Q7 w - - 88 100", 10);
        assert!(mv.is_ok(), "KQK hmc=88 must find a move");
        assert!(score > 10000, "KQK hmc=88 must be clearly winning, got {score}");
    }

    #[test]
    fn test_kqk_finds_mate_hmc_95() {
        // KQ vs K, halfmove_clock = 95: 5 half-moves left, short mates still OK
        // Kf6, Qd1, black Kh8
        let (mv, score) = search_depth("7k/8/5K2/8/8/8/8/3Q4 w - - 95 100", 10);
        assert!(mv.is_ok(), "KQK hmc=95 must find a move");
        assert!(score > 5000, "KQK hmc=95 must be winning, got {score}");
    }

    #[test]
    fn test_kqk_hmc_99_no_crash() {
        // KQ vs K at hmc=99: next move hits 50-move rule. Should not crash.
        // Kf6, Qd1, black Kh8
        let (mv, _score) = search_depth("7k/8/5K2/8/8/8/8/3Q4 w - - 99 100", 8);
        assert!(mv.is_ok(), "KQK hmc=99 must find a move without crashing");
    }
}
