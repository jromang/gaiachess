//! `gaiachess ftperm`: reorder the feature transformer of a network already exported,
//! so that the sparse L1 skips more of it (`ftperm_order` says why and how).
//!
//! The samples come from the engine itself: the accumulators of the positions of an
//! EPD and of their children, read as the machine holds them. The order is applied
//! to the file: every FT row and the biases, each left neuron keeping its right-half
//! partner, and the L1's input columns; the payload then gets a fresh integrity
//! footer. The trainer applies the same order search at export (`tools/trainer8`).

use std::mem::size_of;

use super::ftperm_order::{
    self, GROUP, Mask, PAIRS, Permutation, find_permutation, groups_per_eval, is_permutation,
};
use super::network::{self, NNUE_FILE_SIZE, file};
use super::{FT_QUANT, FT_SHIFT, L1_OUT, L1_SIZE, PSQ_SIZE, STACKS, THREAT_INPUT_SIZE};
use crate::position::Position;

// The order search assumes the engine's activation; a change on either side stops here.
const _: () = assert!(PAIRS == L1_SIZE / 2 && GROUP * 64 == L1_SIZE / 4);
const _: () = assert!(ftperm_order::FT_QUANT == FT_QUANT as i32);
const _: () = assert!(ftperm_order::FT_SHIFT == FT_SHIFT);
// ============================================================
// Sampling
// ============================================================

/// For each logical neuron, the column of the accumulator that holds it on this
/// machine: memory is not in logical order on the tiers whose `packus` interleaves
/// lanes (`network::memory_to_logical`).
fn logical_to_memory() -> [u16; PAIRS] {
    let map = network::memory_to_logical();
    let mut at = [0u16; PAIRS];
    for (m, &l) in map.iter().take(PAIRS).enumerate() {
        debug_assert!(
            (l as usize) < PAIRS,
            "column {m} maps to logical {l}, past the left half"
        );
        at[l as usize] = m as u16;
    }
    at
}

/// The non-zero products of both perspectives of `pos`, evaluated by `net` with the
/// weights it is set to use. The accumulator is read as the machine holds it, `at`
/// (from `logical_to_memory`) saying where each logical neuron sits.
pub fn masks(net: &mut super::Network, pos: &Position, at: &[u16; PAIRS]) -> [Mask; 2] {
    net.refresh(pos);
    let acc = &net.accumulators[0];
    std::array::from_fn(|pov| {
        let v = &acc.values.0[pov];
        let sum = |m: usize| v[m] as i32;
        ftperm_order::mask(|k| {
            let m = at[k] as usize;
            (sum(m), sum(m + PAIRS))
        })
    })
}

// ============================================================
// Applying it to the file
// ============================================================

/// The network file with its FT neurons in the order `perm`, followed by a fresh
/// integrity footer. Everything outside the FT and the L1's input columns is copied
/// as it is.
pub fn permute_file(data: &[u8], perm: &Permutation) -> Result<Vec<u8>, String> {
    if data.len() < NNUE_FILE_SIZE {
        return Err(format!(
            "network file of {} bytes, expected at least {NNUE_FILE_SIZE}",
            data.len()
        ));
    }
    if !is_permutation(perm) {
        return Err(String::from("not a permutation of the FT neurons"));
    }
    let mut out = data[..NNUE_FILE_SIZE].to_vec();

    let psq = file::PSQ;
    permute_rows(&mut out[psq..psq + PSQ_SIZE * L1_SIZE], 1, perm);
    let threats = file::THREAT;
    permute_rows(
        &mut out[threats..threats + THREAT_INPUT_SIZE * L1_SIZE],
        1,
        perm,
    );
    let biases = file::BIASES;
    permute_rows(&mut out[biases..biases + L1_SIZE * 2], 2, perm);
    let l1 = file::L1;
    const L1_BYTES: usize = size_of::<[[[i8; L1_OUT * 4]; L1_SIZE / 4]; STACKS]>();
    permute_l1(&mut out[l1..l1 + L1_BYTES], perm);

    let footer = super::integrity::footer(&out);
    out.extend_from_slice(&footer);
    Ok(out)
}

/// Where each logical neuron of an FT row sits in the file's layout.
fn file_positions() -> [u16; L1_SIZE] {
    // After this, `logical_at[p]` is the neuron the file holds at position `p`.
    let mut logical_at: [u16; L1_SIZE] = std::array::from_fn(|i| i as u16);
    network::linear_to_file(&mut logical_at);
    let mut at = [0u16; L1_SIZE];
    for (p, &l) in logical_at.iter().enumerate() {
        at[l as usize] = p as u16;
    }
    at
}

/// Reorder FT rows of `L1_SIZE` elements of `width` bytes each, held in the file's
/// layout: logical neuron `perm[k]` moves to `k`, and its right-half partner with it.
fn permute_rows(bytes: &mut [u8], width: usize, perm: &Permutation) {
    let at = file_positions();
    let row_bytes = L1_SIZE * width;
    debug_assert_eq!(bytes.len() % row_bytes, 0);
    let mut old = vec![0u8; row_bytes];
    for chunk in bytes.chunks_exact_mut(row_bytes) {
        old.copy_from_slice(chunk);
        for (k, &src) in perm.iter().enumerate() {
            let src = src as usize;
            for (dst, src) in [(k, src), (k + PAIRS, src + PAIRS)] {
                let (d, s) = (at[dst] as usize * width, at[src] as usize * width);
                chunk[d..d + width].copy_from_slice(&old[s..s + width]);
            }
        }
    }
}

/// Reorder the L1's input columns alike: `[stack][input / 4][output * 4 + input % 4]`.
/// Inputs `0..PAIRS` are the side to move's products, `PAIRS..` the other side's,
/// and both halves follow the same order.
fn permute_l1(bytes: &mut [u8], perm: &Permutation) {
    const STRIDE: usize = L1_OUT * 4;
    let bucket_bytes = (L1_SIZE / 4) * STRIDE;
    debug_assert_eq!(bytes.len(), STACKS * bucket_bytes);
    let at = |input: usize, output: usize| (input / 4) * STRIDE + output * 4 + input % 4;
    for bucket in bytes.chunks_exact_mut(bucket_bytes) {
        let old = bucket.to_vec();
        for input in 0..L1_SIZE {
            let src = (input / PAIRS) * PAIRS + perm[input % PAIRS] as usize;
            for output in 0..L1_OUT {
                bucket[at(input, output)] = old[at(src, output)];
            }
        }
    }
}

// ============================================================
// The command
// ============================================================

/// Every position the samples come from: each base position not in check, and each
/// of its legal children not in check — the positions the search evaluates.
fn for_each_position(bases: &[Position], mut f: impl FnMut(&Position)) {
    let mut moves =
        crate::types::ArrayBuf::<crate::types::Move, { crate::types::MAX_MOVES }>::new();
    for base in bases {
        let mut pos = base.clone();
        if pos.checkers == 0 {
            f(&pos);
        }
        let n = crate::movegen::generate_legal_moves(&pos, &mut moves);
        for i in 0..n {
            let m = moves[i];
            pos.make_move(m);
            if pos.checkers == 0 {
                f(&pos);
            }
            pos.unmake_move(m);
        }
    }
}

/// Learn an order from the positions of `epd`, write the reordered network to
/// `output`, and check on every sampled position that the evaluation is unchanged —
/// nothing is written if one differs.
pub fn run(
    input: &std::path::Path,
    output: &std::path::Path,
    epd: &std::path::Path,
    max_positions: usize,
    passes: usize,
    threads: usize,
) -> Result<(), String> {
    let data = std::fs::read(input).map_err(|e| format!("{}: {e}", input.display()))?;
    network::load_from_bytes(&data)?;

    let text = std::fs::read_to_string(epd).map_err(|e| format!("{}: {e}", epd.display()))?;
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();
    let stride = if max_positions == 0 || lines.len() <= max_positions {
        1
    } else {
        lines.len().div_ceil(max_positions)
    };
    let bases: Vec<Position> = lines
        .iter()
        .step_by(stride)
        .filter_map(|l| Position::from_fen(l).ok())
        .collect();
    if bases.is_empty() {
        return Err(format!("no position could be read from {}", epd.display()));
    }

    let at = logical_to_memory();
    let mut net = super::Network::new();
    let mut samples: Vec<Mask> = Vec::new();
    let mut evals: Vec<i32> = Vec::new();
    for_each_position(&bases, |pos| {
        samples.extend(masks(&mut net, pos, &at));
        evals.push(net.evaluate(pos));
    });
    println!(
        "{} base positions, {} evaluations sampled, both perspectives",
        bases.len(),
        evals.len()
    );

    let identity: Permutation = std::array::from_fn(|k| k as u16);
    println!(
        "file order: {:.2} non-zero groups of {} per evaluation",
        groups_per_eval(&samples, &identity),
        L1_SIZE / 4
    );
    let perm = find_permutation(&samples, passes, threads, |p| {
        if p.pass == 0 {
            println!("greedy groups: {:.2}", p.groups_per_eval);
        } else {
            println!(
                "pass {:3}: {:3} swaps, {:.2}",
                p.pass, p.swaps, p.groups_per_eval
            );
        }
    });

    let permuted = permute_file(&data, &perm)?;
    network::load_from_bytes(&permuted)?;
    let mut net = super::Network::new();
    let mut after: Vec<Mask> = Vec::new();
    let mut changed = 0usize;
    let mut i = 0usize;
    for_each_position(&bases, |pos| {
        after.extend(masks(&mut net, pos, &at));
        changed += (net.evaluate(pos) != evals[i]) as usize;
        i += 1;
    });
    if i != evals.len() || changed > 0 {
        return Err(format!(
            "{changed} of {i} evaluations changed with the reordered network; nothing written"
        ));
    }
    println!(
        "reordered network: {:.2} non-zero groups per evaluation, all {i} evaluations identical",
        groups_per_eval(&after, &identity)
    );

    std::fs::write(output, &permuted).map_err(|e| format!("{}: {e}", output.display()))?;
    println!("written to {}", output.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fixed shuffle, so that no test depends on an identity sneaking through.
    fn shuffle(seed: u64) -> Permutation {
        let mut order: Permutation = std::array::from_fn(|k| k as u16);
        let mut s = seed;
        for i in (1..PAIRS).rev() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            order.swap(i, (s % (i as u64 + 1)) as usize);
        }
        order
    }

    #[test]
    fn ft_rows_move_whole_pairs_and_keep_the_file_layout() {
        let perm = shuffle(0x9e37_79b9_7f4a_7c15);
        // Each element carries its logical index, laid out the way the file has it;
        // two rows, so that a row boundary is crossed.
        let mut row: [u16; L1_SIZE] = std::array::from_fn(|l| l as u16);
        network::linear_to_file(&mut row);
        let mut bytes: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
        bytes.extend_from_slice(&bytes.clone());

        permute_rows(&mut bytes, 2, &perm);

        for half in bytes.chunks_exact(2 * L1_SIZE) {
            let mut got: [u16; L1_SIZE] =
                std::array::from_fn(|i| u16::from_le_bytes([half[2 * i], half[2 * i + 1]]));
            network::file_to_linear(&mut got);
            for k in 0..PAIRS {
                assert_eq!(got[k], perm[k], "left neuron at {k}");
                assert_eq!(got[k + PAIRS], perm[k] + PAIRS as u16, "its partner");
            }
        }
    }

    #[test]
    fn l1_columns_follow_the_neurons_in_both_halves() {
        let perm = shuffle(0x1234_5678_9abc_def1);
        let at = |input: usize, output: usize| (input / 4) * L1_OUT * 4 + output * 4 + input % 4;
        let bucket_bytes = L1_SIZE * L1_OUT;
        // A weight that names its input (mod 251, to fit) and its output and bucket.
        let tag = |bucket: usize, input: usize, output: usize| {
            ((input * 7 + output * 3 + bucket * 11) % 251) as u8
        };
        let mut bytes = vec![0u8; STACKS * bucket_bytes];
        for b in 0..STACKS {
            for i in 0..L1_SIZE {
                for o in 0..L1_OUT {
                    bytes[b * bucket_bytes + at(i, o)] = tag(b, i, o);
                }
            }
        }
        permute_l1(&mut bytes, &perm);
        for b in 0..STACKS {
            for i in 0..L1_SIZE {
                let src = (i / PAIRS) * PAIRS + perm[i % PAIRS] as usize;
                for o in 0..L1_OUT {
                    assert_eq!(bytes[b * bucket_bytes + at(i, o)], tag(b, src, o));
                }
            }
        }
    }

    #[test]
    fn a_file_too_short_or_a_bad_order_is_refused() {
        let identity: Permutation = std::array::from_fn(|k| k as u16);
        assert!(permute_file(&[0u8; 100], &identity).is_err());
        let mut twice = identity;
        twice[1] = 0;
        assert!(permute_file(&vec![0u8; NNUE_FILE_SIZE], &twice).is_err());
    }
}
