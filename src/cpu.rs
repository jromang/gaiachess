//! Runtime CPU dispatch for the NNUE kernels (x86-64 only).
//!
//! One binary carries every kernel instance (`nnue::kernels`); this module
//! decides ONCE, from CPUID, which instance the hot paths call — and which FT
//! weight permutation the loader must apply, since a kernel reading weights
//! permuted for another register width would produce silently wrong
//! evaluations. Tier and permutation therefore come out of the same
//! resolution and can never disagree. Other architectures have a single
//! compile-time backend and no table.
//!
//! Environment overrides, for per-tier verification, per-tier PGO profiling
//! and user support:
//! - `GAIA_SIMD=scalar|avx2|avx512|vnni512` caps the tier. It never raises it
//!   above what the CPU supports — forcing upward would be an
//!   illegal-instruction crash, so an impossible request is clamped down.
//!
//! On a build whose target already pins the top tier (`kernels::STATIC_TOP`,
//! e.g. `target-cpu=native` on a Zen 4/5 box), the hot call sites bypass the
//! table entirely; the resolver pins the same choice so the weight permutation
//! stays consistent, and `GAIA_SIMD` is reported inert.

use std::sync::OnceLock;

use crate::nnue::accumulator::Accumulator;
use crate::nnue::forward::{self, NNZ_SIZE};
use crate::nnue::kernels::{self, k_avx2, k_avx512, k_scalar, k_vnni512, threat512};
use crate::nnue::network::{Aligned, PermKind};
use crate::nnue::L1_SIZE;
use crate::types::Color;

// ============================================================
// Tiers and axes
// ============================================================

/// SIMD register-width tier for the NNUE kernels, ascending.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum SimdTier {
    Scalar,
    Avx2,
    Avx512,
    Vnni512,
}

impl SimdTier {
    pub fn name(self) -> &'static str {
        match self {
            SimdTier::Scalar => "scalar",
            SimdTier::Avx2 => "avx2",
            SimdTier::Avx512 => "avx512",
            SimdTier::Vnni512 => "vnni512",
        }
    }
}

/// `find_nnz` algorithm — an axis independent from the tier: Cascade Lake has
/// VNNI without VBMI2, so the compress form cannot be implied by `Vnni512`.
/// All three produce identical indices; only the speed differs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NnzKind {
    Generic,
    Avx2Table,
    Compress512,
}

/// Per-piece slider attack implementation — another independent axis: BMI2 is
/// reported by AMD generations that microcode PDEP/PEXT into tens of cycles
/// (Bulldozer through Zen 2), where the AVX2 path wins despite the bit.
/// All three return identical bitboards; only the speed differs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AttackKind {
    Magic,
    Avx2Blsmsk,
    Pext,
}

impl AttackKind {
    pub fn name(self) -> &'static str {
        match self {
            AttackKind::Magic => "magic",
            AttackKind::Avx2Blsmsk => "avx2",
            AttackKind::Pext => "pext",
        }
    }
}

impl NnzKind {
    pub fn name(self) -> &'static str {
        match self {
            NnzKind::Generic => "table",
            NnzKind::Avx2Table => "table+sse",
            NnzKind::Compress512 => "vpcompressw",
        }
    }
}

// ============================================================
// The dispatch table
// ============================================================

/// Function pointers into the resolved kernel instance, plus the metadata the
/// loader (`perm`) and the `info` report need.
pub struct CpuDispatch {
    pub forward_dense:
        unsafe fn(&Accumulator, &Aligned<[[i16; L1_SIZE]; 2]>, Color, usize) -> f32,
    pub find_nnz: unsafe fn(&Aligned<[u8; L1_SIZE]>) -> ([u16; NNZ_SIZE], usize),
    pub acc_add1_sub1: unsafe fn(*const i16, *mut i16, usize, usize),
    pub acc_add1_sub2: unsafe fn(*const i16, *mut i16, usize, usize, usize),
    pub acc_add2_sub2: unsafe fn(*const i16, *mut i16, usize, usize, usize, usize),
    pub finny_apply: unsafe fn(*mut i16, &[usize], &[usize]),
    pub threat_batch:
        unsafe fn(*const i16, *mut i16, *const [i8; L1_SIZE], &[u32], &[u32]),
    pub tier: SimdTier,
    pub nnz: NnzKind,
    pub perm: PermKind,
    pub attacks: AttackKind,
    /// AVX-512 Kogge-Stone setwise slider fill available.
    pub setwise512: bool,
    /// `Some(cap)` when `GAIA_SIMD` lowered the tier below what was detected.
    pub forced: Option<SimdTier>,
    /// `GAIA_PEXT` override was applied.
    pub pext_forced: bool,
}

static DISPATCH: OnceLock<CpuDispatch> = OnceLock::new();

/// Resolve the dispatch table, building it on first call. For cold paths
/// (network loading, init, tests): correct in any call order, because the
/// resolution reads only CPUID and the environment.
pub fn get_or_init() -> &'static CpuDispatch {
    DISPATCH.get_or_init(resolve)
}

/// Read the dispatch table on a hot path.
///
/// The table is read without checking it exists, which keeps the per-node cost
/// at one predicted indirect call. The engine earns that by resolving in
/// `init_cpu_dispatch` before anything can search. A unit-test binary has no
/// `main` of ours, so test builds resolve on demand instead.
#[inline(always)]
pub fn get() -> &'static CpuDispatch {
    #[cfg(test)]
    return DISPATCH.get_or_init(resolve);
    #[cfg(not(test))]
    unsafe {
        debug_assert!(DISPATCH.get().is_some(), "cpu dispatch read before init_cpu_dispatch");
        DISPATCH.get().unwrap_unchecked()
    }
}

/// Resolve the table at startup and configure the slider-attack paths from
/// the same resolution. Called by `init_cpu_dispatch`.
///
/// Attack-path election is a distribution-build affair (`--cfg gaia_dist`);
/// other builds pin the attack kind at compile time and only need whatever
/// tables their static choice reads.
pub fn init() {
    let dispatch = get_or_init();
    #[cfg(gaia_dist)]
    {
        // Tables are built before the flag flips: set_use_pext builds them itself.
        crate::bitboard::set_use_pext(dispatch.attacks == AttackKind::Pext);
        crate::simd_attacks::set_setwise512(dispatch.setwise512);
    }
    #[cfg(not(gaia_dist))]
    {
        let _ = dispatch;
        #[cfg(target_feature = "bmi2")]
        crate::bitboard::init_pext();
    }
}

/// True when the build target pins the top tier and the kernels are reached
/// directly rather than through the table (`kernels::STATIC_TOP`).
pub fn statically_pinned() -> bool {
    kernels::STATIC_TOP
}

/// The attack kind actually running: the runtime election on distribution
/// builds, the compile-time pin everywhere else.
pub fn effective_attacks() -> AttackKind {
    if cfg!(gaia_dist) {
        get_or_init().attacks
    } else if cfg!(target_feature = "bmi2") {
        AttackKind::Pext
    } else if cfg!(target_feature = "avx2") {
        AttackKind::Avx2Blsmsk
    } else {
        AttackKind::Magic
    }
}

/// Whether the AVX-512 setwise slider fill actually runs.
pub fn effective_setwise512() -> bool {
    if cfg!(gaia_dist) {
        get_or_init().setwise512
    } else {
        cfg!(target_feature = "avx512f")
    }
}

// ============================================================
// Detection
// ============================================================

/// Feature bits the resolver cares about, separated from their detection so
/// the decision logic is testable on any host.
#[derive(Clone, Copy, Default)]
pub struct Caps {
    pub avx2: bool,
    pub fma: bool,
    pub avx512f: bool,
    pub avx512bw: bool,
    pub avx512vl: bool,
    pub avx512vnni: bool,
    pub avx512vbmi2: bool,
    pub bmi2: bool,
    /// AuthenticAMD.
    pub amd: bool,
    /// AMD family 0x15–0x17 (Bulldozer through Zen 2): PDEP/PEXT microcoded.
    pub amd_slow_pext: bool,
}

fn detect() -> Caps {
    let (amd, amd_slow_pext) = amd_vendor_and_slow_pext();
    Caps {
        avx2: std::arch::is_x86_feature_detected!("avx2"),
        fma: std::arch::is_x86_feature_detected!("fma"),
        avx512f: std::arch::is_x86_feature_detected!("avx512f"),
        avx512bw: std::arch::is_x86_feature_detected!("avx512bw"),
        avx512vl: std::arch::is_x86_feature_detected!("avx512vl"),
        avx512vnni: std::arch::is_x86_feature_detected!("avx512vnni"),
        avx512vbmi2: std::arch::is_x86_feature_detected!("avx512vbmi2"),
        bmi2: std::arch::is_x86_feature_detected!("bmi2"),
        amd,
        amd_slow_pext,
    }
}

/// (is AMD, PDEP/PEXT microcoded). The instructions are microcoded below
/// Zen 3's family 0x19: those CPUs report BMI2, but the pair costs tens of
/// cycles and the AVX2 attack path wins outright. Zen 3 made them 3-cycle.
fn amd_vendor_and_slow_pext() -> (bool, bool) {
    use std::arch::x86_64::__cpuid;
    // CPUID leaves 0 and 1 exist on every x86-64 CPU.
    let id0 = __cpuid(0);
    // "AuthenticAMD" spelled across ebx/edx/ecx.
    let amd = id0.ebx == 0x6874_7541 && id0.edx == 0x6974_6e65 && id0.ecx == 0x444d_4163;
    if !amd {
        return (false, false);
    }
    let id1 = __cpuid(1);
    let base_family = (id1.eax >> 8) & 0xF;
    let ext_family = (id1.eax >> 20) & 0xFF;
    let family = if base_family == 0xF { base_family + ext_family } else { base_family };
    (true, family < 0x19)
}

fn read_simd_cap() -> Option<SimdTier> {
    let v = std::env::var("GAIA_SIMD").ok()?;
    match v.to_ascii_lowercase().as_str() {
        "scalar" => Some(SimdTier::Scalar),
        "avx2" => Some(SimdTier::Avx2),
        "avx512" => Some(SimdTier::Avx512),
        "vnni512" => Some(SimdTier::Vnni512),
        other => {
            eprintln!("info string GAIA_SIMD '{other}' ignored (scalar|avx2|avx512|vnni512)");
            None
        }
    }
}

fn read_pext_override() -> Option<bool> {
    let v = std::env::var("GAIA_PEXT").ok()?;
    match v.as_str() {
        "0" => Some(false),
        "1" => Some(true),
        other => {
            eprintln!("info string GAIA_PEXT '{other}' ignored (0|1)");
            None
        }
    }
}

// ============================================================
// Resolution
// ============================================================

/// Everything `choose` decides, in one place.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Selection {
    tier: SimdTier,
    nnz: NnzKind,
    perm: PermKind,
    attacks: AttackKind,
    setwise512: bool,
    forced: Option<SimdTier>,
    pext_forced: bool,
}

/// Pure decision core: what runs, given what the CPU has and what the
/// environment asks for. `static_top` mirrors `kernels::STATIC_TOP` and
/// `static_avx2` mirrors the build's AVX2 guarantee; both are parameters so
/// the logic is testable regardless of the build target.
fn choose(
    caps: Caps,
    cap: Option<SimdTier>,
    pext_ov: Option<bool>,
    static_top: bool,
    static_avx2: bool,
) -> Selection {
    let avx2_ok = caps.avx2 && caps.fma;
    let b512_ok = avx2_ok && caps.avx512f && caps.avx512bw && caps.avx512vl;
    let detected = if b512_ok && caps.avx512vnni {
        SimdTier::Vnni512
    } else if b512_ok {
        SimdTier::Avx512
    } else if avx2_ok {
        SimdTier::Avx2
    } else {
        SimdTier::Scalar
    };

    let mut tier = detected;
    let mut forced = None;
    if !static_top {
        if let Some(c) = cap {
            if c < tier {
                tier = c;
                forced = Some(c);
            }
        }
    }

    let nnz = if tier >= SimdTier::Avx512 && caps.avx512vbmi2 {
        NnzKind::Compress512
    } else if tier >= SimdTier::Avx2 {
        NnzKind::Avx2Table
    } else {
        NnzKind::Generic
    };

    let perm = match tier {
        SimdTier::Vnni512 | SimdTier::Avx512 => PermKind::File512,
        SimdTier::Avx2 => PermKind::Avx2,
        SimdTier::Scalar => PermKind::Linear,
    };

    // PEXT rides its own axis: it needs the hardware bit, and can never be
    // forced onto a CPU that lacks it (that would be an illegal instruction,
    // not a slow one). By default it stays off on AMD: microcoded below Zen 3,
    // and on Zen 3+ the AVX2 path measured consistently faster than PEXT in
    // the dispatched binary (+1.2% NPS on Zen 5, 12/12 interleaved PGO runs —
    // plausibly the 850 KB PEXT tables against AVX2's 7.5 KB in a cache
    // already crowded by the network). Intel keeps PEXT, unmeasured here.
    let pext_default = caps.bmi2
        && !caps.amd_slow_pext
        && !(caps.amd && static_avx2);
    let (pext_on, pext_forced) = match pext_ov {
        Some(true) if caps.bmi2 => (true, true),
        Some(true) => (false, false),
        Some(false) => (false, true),
        None => (pext_default, false),
    };
    let attacks = if pext_on {
        AttackKind::Pext
    } else if static_avx2 {
        AttackKind::Avx2Blsmsk
    } else {
        AttackKind::Magic
    };

    Selection {
        tier,
        nnz,
        perm,
        attacks,
        setwise512: caps.avx512f,
        forced,
        pext_forced,
    }
}

fn resolve() -> CpuDispatch {
    let caps = detect();
    let cap = read_simd_cap();
    let pext_ov = read_pext_override();
    if kernels::STATIC_TOP && cap.is_some() {
        eprintln!(
            "info string GAIA_SIMD ignored: this build pins the top SIMD tier at compile time"
        );
    }
    if pext_ov == Some(true) && !caps.bmi2 {
        eprintln!("info string GAIA_PEXT=1 ignored: this CPU has no BMI2");
    }
    let Selection { tier, nnz, perm, attacks, setwise512, forced, pext_forced } = choose(
        caps,
        cap,
        pext_ov,
        kernels::STATIC_TOP,
        cfg!(target_feature = "avx2"),
    );

    let find_nnz = match nnz {
        NnzKind::Compress512 => forward::find_nnz_compress512
            as unsafe fn(&Aligned<[u8; L1_SIZE]>) -> ([u16; NNZ_SIZE], usize),
        NnzKind::Avx2Table => forward::find_nnz_avx2,
        NnzKind::Generic => k_scalar::find_nnz_generic,
    };

    match tier {
        SimdTier::Vnni512 => CpuDispatch {
            forward_dense: k_vnni512::forward_dense,
            find_nnz,
            acc_add1_sub1: k_vnni512::acc_add1_sub1,
            acc_add1_sub2: k_vnni512::acc_add1_sub2,
            acc_add2_sub2: k_vnni512::acc_add2_sub2,
            finny_apply: k_vnni512::finny_apply,
            threat_batch: threat512::threat_batch,
            tier, nnz, perm, attacks, setwise512, forced, pext_forced,
        },
        SimdTier::Avx512 => CpuDispatch {
            forward_dense: k_avx512::forward_dense,
            find_nnz,
            acc_add1_sub1: k_avx512::acc_add1_sub1,
            acc_add1_sub2: k_avx512::acc_add1_sub2,
            acc_add2_sub2: k_avx512::acc_add2_sub2,
            finny_apply: k_avx512::finny_apply,
            threat_batch: threat512::threat_batch,
            tier, nnz, perm, attacks, setwise512, forced, pext_forced,
        },
        SimdTier::Avx2 => CpuDispatch {
            forward_dense: k_avx2::forward_dense,
            find_nnz,
            acc_add1_sub1: k_avx2::acc_add1_sub1,
            acc_add1_sub2: k_avx2::acc_add1_sub2,
            acc_add2_sub2: k_avx2::acc_add2_sub2,
            finny_apply: k_avx2::finny_apply,
            threat_batch: k_avx2::threat_batch,
            tier, nnz, perm, attacks, setwise512, forced, pext_forced,
        },
        SimdTier::Scalar => CpuDispatch {
            forward_dense: k_scalar::forward_dense,
            find_nnz,
            acc_add1_sub1: k_scalar::acc_add1_sub1,
            acc_add1_sub2: k_scalar::acc_add1_sub2,
            acc_add2_sub2: k_scalar::acc_add2_sub2,
            finny_apply: k_scalar::finny_apply,
            threat_batch: k_scalar::threat_batch,
            tier, nnz, perm, attacks, setwise512, forced, pext_forced,
        },
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn caps(
        avx2: bool, fma: bool, f: bool, bw: bool, vl: bool, vnni: bool, vbmi2: bool,
    ) -> Caps {
        Caps {
            avx2, fma,
            avx512f: f, avx512bw: bw, avx512vl: vl, avx512vnni: vnni, avx512vbmi2: vbmi2,
            bmi2: avx2, // BMI2 ships with AVX2 on every real CPU; overridden where it matters
            amd: false,
            amd_slow_pext: false,
        }
    }

    fn pick(c: Caps, cap: Option<SimdTier>) -> Selection {
        choose(c, cap, None, false, true)
    }

    #[test]
    fn zen5_gets_the_top_tier_and_the_compress_nnz() {
        let mut c = caps(true, true, true, true, true, true, true);
        c.amd = true;
        let s = pick(c, None);
        assert_eq!(s.tier, SimdTier::Vnni512);
        assert_eq!(s.nnz, NnzKind::Compress512);
        assert_eq!(s.perm, PermKind::File512);
        // AMD default: AVX2 attacks measured faster than PEXT in the
        // dispatched binary (Zen 5, +1.2%).
        assert_eq!(s.attacks, AttackKind::Avx2Blsmsk);
        assert!(s.setwise512);
        assert_eq!(s.forced, None);
    }

    #[test]
    fn an_intel_with_bmi2_keeps_the_pext_attacks() {
        let s = pick(caps(true, true, true, true, true, true, true), None);
        assert_eq!(s.attacks, AttackKind::Pext);
    }

    #[test]
    fn zen3_on_the_compat_baseline_still_prefers_pext_over_magic() {
        // Without static AVX2 the alternative to PEXT is magic, not AVX2.
        let mut c = caps(true, true, false, false, false, false, false);
        c.amd = true;
        let s = choose(c, None, None, false, false);
        assert_eq!(s.attacks, AttackKind::Pext);
    }

    #[test]
    fn cascade_lake_has_vnni_but_keeps_the_table_nnz() {
        // VNNI without VBMI2 — the two axes must stay independent.
        let s = pick(caps(true, true, true, true, true, true, false), None);
        assert_eq!(s.tier, SimdTier::Vnni512);
        assert_eq!(s.nnz, NnzKind::Avx2Table);
    }

    #[test]
    fn skylake_x_stops_at_plain_avx512() {
        let s = pick(caps(true, true, true, true, true, false, false), None);
        assert_eq!(s.tier, SimdTier::Avx512);
        assert_eq!(s.nnz, NnzKind::Avx2Table);
        assert_eq!(s.perm, PermKind::File512);
        assert!(s.setwise512);
    }

    #[test]
    fn haswell_lands_on_avx2_with_its_permutation() {
        let s = pick(caps(true, true, false, false, false, false, false), None);
        assert_eq!(s.tier, SimdTier::Avx2);
        assert_eq!(s.nnz, NnzKind::Avx2Table);
        assert_eq!(s.perm, PermKind::Avx2);
        assert!(!s.setwise512);
    }

    #[test]
    fn a_cpu_without_avx2_falls_back_to_scalar_and_linear_weights() {
        let s = choose(caps(false, false, false, false, false, false, false), None, None, false, false);
        assert_eq!(s.tier, SimdTier::Scalar);
        assert_eq!(s.nnz, NnzKind::Generic);
        assert_eq!(s.perm, PermKind::Linear);
        assert_eq!(s.attacks, AttackKind::Magic);
    }

    #[test]
    fn avx512f_without_bw_is_not_enough_for_the_512_tiers() {
        // Knights Landing-style: F+VL but no BW — the i16 kernels need BW.
        let s = pick(caps(true, true, true, false, true, false, false), None);
        assert_eq!(s.tier, SimdTier::Avx2);
    }

    #[test]
    fn the_cap_lowers_the_tier_and_its_permutation_together() {
        let all = caps(true, true, true, true, true, true, true);
        let s = pick(all, Some(SimdTier::Avx2));
        assert_eq!(s.tier, SimdTier::Avx2);
        assert_eq!(s.nnz, NnzKind::Avx2Table);
        assert_eq!(s.perm, PermKind::Avx2);
        assert_eq!(s.forced, Some(SimdTier::Avx2));

        let s = pick(all, Some(SimdTier::Scalar));
        assert_eq!(s.tier, SimdTier::Scalar);
        assert_eq!(s.nnz, NnzKind::Generic);
        assert_eq!(s.perm, PermKind::Linear);
        assert_eq!(s.forced, Some(SimdTier::Scalar));
    }

    #[test]
    fn the_cap_never_raises_the_tier_above_the_cpu() {
        let haswell = caps(true, true, false, false, false, false, false);
        let s = pick(haswell, Some(SimdTier::Vnni512));
        assert_eq!(s.tier, SimdTier::Avx2);
        assert_eq!(s.forced, None);
    }

    #[test]
    fn a_static_top_build_ignores_the_cap() {
        // The call sites bypass the table on such a build; honoring the cap
        // would desynchronize the weight permutation from the running kernels.
        let all = caps(true, true, true, true, true, true, true);
        let s = choose(all, Some(SimdTier::Scalar), None, true, true);
        assert_eq!(s.tier, SimdTier::Vnni512);
        assert_eq!(s.perm, PermKind::File512);
        assert_eq!(s.forced, None);
    }

    #[test]
    fn zen2_reports_bmi2_but_gets_the_avx2_attacks() {
        let mut c = caps(true, true, false, false, false, false, false);
        c.amd = true;
        c.amd_slow_pext = true;
        let s = pick(c, None);
        assert_eq!(s.attacks, AttackKind::Avx2Blsmsk);
        assert!(!s.pext_forced);
        // ... unless explicitly asked for.
        let s = choose(c, None, Some(true), false, true);
        assert_eq!(s.attacks, AttackKind::Pext);
        assert!(s.pext_forced);
    }

    #[test]
    fn pext_cannot_be_forced_onto_a_cpu_without_bmi2() {
        let mut c = caps(false, false, false, false, false, false, false);
        c.bmi2 = false;
        let s = choose(c, None, Some(true), false, false);
        assert_eq!(s.attacks, AttackKind::Magic);
        assert!(!s.pext_forced);
    }

    #[test]
    fn gaia_pext_zero_retires_the_pext_path() {
        // Intel caps (amd=false), where the default would otherwise be PEXT.
        let all = caps(true, true, true, true, true, true, true);
        let s = choose(all, None, Some(false), false, true);
        assert_eq!(s.attacks, AttackKind::Avx2Blsmsk);
        assert!(s.pext_forced);
    }

    #[test]
    fn the_resolved_table_matches_this_machine() {
        let d = get_or_init();
        // Whatever the host, the invariant that matters: permutation follows tier.
        let expected_perm = match d.tier {
            SimdTier::Vnni512 | SimdTier::Avx512 => PermKind::File512,
            SimdTier::Avx2 => PermKind::Avx2,
            SimdTier::Scalar => PermKind::Linear,
        };
        assert_eq!(d.perm, expected_perm);
    }
}
