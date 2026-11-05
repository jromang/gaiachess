//! Choosing the order of the feature transformer's neurons, so that the sparse L1
//! skips more of its output.
//!
//! The L1 reads the FT output four bytes at a time and skips a group only when all
//! four products are zero. Which neurons share a group is arbitrary — training never
//! sees the order — so it can be chosen: neurons that fall silent in the same
//! positions go together, and the L1 reads fewer groups. The evaluation does not
//! move by a single unit: the L1 sums integers, and in any order.
//!
//! This file depends on nothing but `std`. The engine (`gaiachess ftperm`, for a
//! network already exported) and the trainer (at export) both compile it, each from
//! its own samples, so that the two can never disagree on how an order is chosen or
//! applied.

/// Products per perspective: the logical neurons a permutation moves. Each one is the
/// product of FT neuron `k` and its partner `k + PAIRS`, which move together.
pub const PAIRS: usize = 512;
/// Bytes the sparse L1 reads, and skips, as one unit.
pub const GROUP: usize = 4;
/// Groups per perspective.
const GROUPS: usize = PAIRS / GROUP;

/// The FT activation's quantisation and the shift of its pairwise product, as the
/// engine computes it: `(clamp(a, 0, QA) * min(b, QA)) >> SHIFT`. Both sides check
/// these against their own constants.
pub const FT_QUANT: i32 = 255;
pub const FT_SHIFT: i32 = 9;

/// The non-zero products of one perspective of one position, by logical neuron.
pub type Mask = [u64; PAIRS / 64];

/// A new order for the neurons: logical neuron `perm[k]` moves to position `k`.
pub type Permutation = [u16; PAIRS];

// ============================================================
// Sampling
// ============================================================

/// Whether the product of a neuron whose accumulator holds `left` and its partner
/// holding `right` reaches the L1 as a non-zero byte: left clamped to [0, QA], right
/// only from above, product shifted down.
#[inline]
pub fn live(left: i32, right: i32) -> bool {
    (left.clamp(0, FT_QUANT) * right.min(FT_QUANT)) >> FT_SHIFT > 0
}

/// The mask of one perspective, from the accumulator sums of each logical neuron and
/// its partner: `sums(k)` is `(acc[k], acc[k + PAIRS])` in logical order.
pub fn mask(sums: impl Fn(usize) -> (i32, i32)) -> Mask {
    let mut out = [0u64; PAIRS / 64];
    for k in 0..PAIRS {
        let (left, right) = sums(k);
        if live(left, right) {
            out[k / 64] |= 1 << (k % 64);
        }
    }
    out
}

/// Mean number of non-zero groups the L1 reads per evaluation (both perspectives,
/// out of `L1_SIZE / 4`) if the neurons were laid out in `order`.
pub fn groups_per_eval(samples: &[Mask], order: &Permutation) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut total = 0u64;
    for mask in samples {
        for group in order.chunks_exact(GROUP) {
            let live = group
                .iter()
                .any(|&k| mask[k as usize / 64] >> (k as usize % 64) & 1 != 0);
            total += live as u64;
        }
    }
    // Two samples per evaluation: one per perspective.
    2.0 * total as f64 / samples.len() as f64
}

// ============================================================
// Finding the order
// ============================================================

/// For each neuron, the samples in which its product is zero: one bit per sample,
/// neuron-major.
struct ZeroSets {
    words: usize,
    bits: Vec<u64>,
}

impl ZeroSets {
    fn new(samples: &[Mask]) -> Self {
        let words = samples.len().div_ceil(64);
        let mut bits = vec![0u64; PAIRS * words];
        for (s, mask) in samples.iter().enumerate() {
            for k in 0..PAIRS {
                if mask[k / 64] >> (k % 64) & 1 == 0 {
                    bits[k * words + s / 64] |= 1 << (s % 64);
                }
            }
        }
        ZeroSets { words, bits }
    }

    fn of(&self, k: usize) -> &[u64] {
        &self.bits[k * self.words..(k + 1) * self.words]
    }
}

/// Samples in both sets.
fn common(a: &[u64], b: &[u64]) -> u32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x & y).count_ones()).sum()
}

/// Progress of the search, for whoever is watching.
pub struct Pass {
    pub pass: usize,
    pub swaps: usize,
    pub groups_per_eval: f64,
}

/// Groups built one at a time: each starts from the most often silent neuron left,
/// and takes in, three times, the neuron left that is silent in most of the samples
/// where the group so far is.
fn greedy_groups(zero: &ZeroSets) -> Permutation {
    let words = zero.words;
    let silence: Vec<u32> = (0..PAIRS)
        .map(|k| zero.of(k).iter().map(|w| w.count_ones()).sum())
        .collect();
    let mut left: Vec<usize> = (0..PAIRS).collect();
    left.sort_by_key(|&k| std::cmp::Reverse(silence[k]));
    let mut order = [0u16; PAIRS];
    let mut together = vec![0u64; words];
    for group in order.chunks_exact_mut(GROUP) {
        let seed = left.remove(0);
        group[0] = seed as u16;
        together.copy_from_slice(zero.of(seed));
        for slot in group.iter_mut().skip(1) {
            let (i, _) = left
                .iter()
                .enumerate()
                .max_by_key(|&(i, &k)| (common(&together, zero.of(k)), std::cmp::Reverse(i)))
                .expect("a neuron left for every slot");
            let k = left.remove(i);
            *slot = k as u16;
            for (t, z) in together.iter_mut().zip(zero.of(k)) {
                *t &= z;
            }
        }
    }
    order
}

/// An order that puts neurons silent in the same samples into the same group.
///
/// Groups are first built greedily (`greedy_groups`); then any two neurons of
/// different groups are swapped when that leaves more groups all-zero over the
/// samples. Each pass scores every swap against the current groups and applies the
/// best ones that touch disjoint groups, whose gains therefore still hold; it stops
/// when no swap gains or after `max_passes`.
pub fn find_permutation(
    samples: &[Mask],
    max_passes: usize,
    threads: usize,
    mut report: impl FnMut(&Pass),
) -> Permutation {
    let zero = ZeroSets::new(samples);
    let words = zero.words;
    let n = samples.len() as f64;

    let mut order = greedy_groups(&zero);

    let threads = threads.max(1);
    let mut rest = vec![0u64; PAIRS * words];
    let mut full = [0u32; GROUPS];
    let mut gain = vec![0u32; PAIRS * PAIRS];

    for pass in 0..max_passes {
        // For each slot, the samples in which the other three of its group are zero;
        // for each group, those in which all four are.
        for g in 0..GROUPS {
            let members: [&[u64]; GROUP] =
                std::array::from_fn(|i| zero.of(order[g * GROUP + i] as usize));
            let mut all = 0u32;
            for w in 0..words {
                let v: [u64; GROUP] = std::array::from_fn(|i| members[i][w]);
                all += (v[0] & v[1] & v[2] & v[3]).count_ones();
                rest[(g * GROUP) * words + w] = v[1] & v[2] & v[3];
                rest[(g * GROUP + 1) * words + w] = v[0] & v[2] & v[3];
                rest[(g * GROUP + 2) * words + w] = v[0] & v[1] & v[3];
                rest[(g * GROUP + 3) * words + w] = v[0] & v[1] & v[2];
            }
            full[g] = all;
        }
        if pass == 0 {
            let silent: u64 = full.iter().map(|&c| c as u64).sum();
            report(&Pass {
                pass,
                swaps: 0,
                groups_per_eval: 2.0 * (GROUPS as f64 - silent as f64 / n),
            });
        }

        // gain[a][b]: samples in which slot b's neuron would be zero along with the
        // other three of slot a's group — slot a's group, once b's neuron replaces a's.
        let rows_per = PAIRS.div_ceil(threads);
        std::thread::scope(|scope| {
            for (t, chunk) in gain.chunks_mut(rows_per * PAIRS).enumerate() {
                let (rest, zero, order) = (&rest, &zero, &order);
                scope.spawn(move || {
                    for (i, row) in chunk.chunks_exact_mut(PAIRS).enumerate() {
                        let a = t * rows_per + i;
                        let ra = &rest[a * words..(a + 1) * words];
                        for (b, cell) in row.iter_mut().enumerate() {
                            *cell = if a / GROUP == b / GROUP {
                                0
                            } else {
                                common(ra, zero.of(order[b] as usize))
                            };
                        }
                    }
                });
            }
        });

        let mut swaps: Vec<(i64, usize, usize)> = Vec::new();
        for a in 0..PAIRS {
            for b in (a + 1)..PAIRS {
                let (ga, gb) = (a / GROUP, b / GROUP);
                if ga == gb {
                    continue;
                }
                let delta = gain[a * PAIRS + b] as i64 + gain[b * PAIRS + a] as i64
                    - full[ga] as i64
                    - full[gb] as i64;
                if delta > 0 {
                    swaps.push((delta, a, b));
                }
            }
        }
        swaps.sort_unstable_by(|x, y| y.0.cmp(&x.0));
        let mut touched = [false; GROUPS];
        let mut applied = 0;
        for &(_, a, b) in &swaps {
            let (ga, gb) = (a / GROUP, b / GROUP);
            if touched[ga] || touched[gb] {
                continue;
            }
            touched[ga] = true;
            touched[gb] = true;
            order.swap(a, b);
            applied += 1;
        }
        if applied == 0 {
            break;
        }
        report(&Pass {
            pass: pass + 1,
            swaps: applied,
            groups_per_eval: groups_per_eval_of(&zero, &order, n),
        });
    }
    debug_assert!(is_permutation(&order));
    order
}

/// `groups_per_eval` from the zero sets, without going back to the samples.
fn groups_per_eval_of(zero: &ZeroSets, order: &Permutation, n: f64) -> f64 {
    let mut silent = 0u64;
    for group in order.chunks_exact(GROUP) {
        let members: [&[u64]; GROUP] = std::array::from_fn(|i| zero.of(group[i] as usize));
        for w in 0..zero.words {
            silent +=
                (members[0][w] & members[1][w] & members[2][w] & members[3][w]).count_ones() as u64;
        }
    }
    2.0 * (GROUPS as f64 - silent as f64 / n)
}

/// Whether `order` moves every neuron exactly once.
pub fn is_permutation(order: &Permutation) -> bool {
    let mut seen = [false; PAIRS];
    order.iter().all(|&k| {
        let fresh = (k as usize) < PAIRS && !seen[k as usize];
        if fresh {
            seen[k as usize] = true;
        }
        fresh
    })
}

// ============================================================
// Applying it
// ============================================================

/// Reorder one FT row in logical order, or one row of L1 weights over its inputs:
/// element `perm[k]` moves to `k`, and `perm[k] + PAIRS` to `k + PAIRS`. For the FT
/// the second half is each neuron's partner; for the L1 it is the other perspective's
/// products, which follow the same order.
pub fn permute_halves<T: Copy>(row: &mut [T], perm: &Permutation) {
    debug_assert_eq!(row.len(), 2 * PAIRS, "a row of {} elements", row.len());
    let old = row.to_vec();
    for (k, &src) in perm.iter().enumerate() {
        let src = src as usize;
        row[k] = old[src];
        row[k + PAIRS] = old[src + PAIRS];
    }
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
    fn rows_move_in_pairs_and_halves_follow_the_same_order() {
        let perm = shuffle(0x9e37_79b9_7f4a_7c15);
        let mut row: Vec<u16> = (0..2 * PAIRS as u16).collect();
        permute_halves(&mut row, &perm);
        for k in 0..PAIRS {
            assert_eq!(row[k], perm[k], "left neuron at {k}");
            assert_eq!(row[k + PAIRS], perm[k] + PAIRS as u16, "its partner");
        }
    }

    #[test]
    fn a_product_is_live_exactly_when_its_byte_is_non_zero() {
        for left in -300..300 {
            for right in [-300, -1, 0, 1, 2, 3, 100, 255, 256, 400] {
                let byte = ((left.clamp(0, 255) * right.min(255)) >> 9).clamp(0, 255);
                assert_eq!(live(left, right), byte != 0, "{left} × {right}");
            }
        }
    }

    /// Neurons planted in quartets that are silent together, then scattered: the
    /// search must bring the quartets back into groups, and never do worse than the
    /// order it was given.
    #[test]
    fn planted_quartets_are_regrouped() {
        let scatter = shuffle(0xdead_beef_cafe_f00d);
        let mut s = 0x2545_f491_4f6c_dd1du64;
        let mut samples = Vec::new();
        for _ in 0..2048 {
            let mut mask = [0u64; PAIRS / 64];
            for quartet in 0..GROUPS {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                // Each quartet speaks in about one sample in five.
                if s % 5 == 0 {
                    for i in 0..GROUP {
                        let k = scatter[quartet * GROUP + i] as usize;
                        mask[k / 64] |= 1 << (k % 64);
                    }
                }
            }
            samples.push(mask);
        }
        let identity: Permutation = std::array::from_fn(|k| k as u16);
        let before = groups_per_eval(&samples, &identity);
        let order = find_permutation(&samples, 50, 2, |_| {});
        let after = groups_per_eval(&samples, &order);
        assert!(is_permutation(&order));
        // Regrouped, each quartet is live only when it speaks: 2 × 128 / 5 ≈ 51.
        assert!(after < 0.6 * before, "before {before:.1}, after {after:.1}");
        assert!(after < 60.0, "after {after:.1}");
    }
}
