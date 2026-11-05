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
//! - `GAIA_SIMD=scalar|avx2|avxvnni|avx512|vnni512` caps the tier. It never raises it
//!   above what the CPU supports — forcing upward would be an
//!   illegal-instruction crash, so an impossible request is clamped down.
//! - `GAIA_THREATS=bitboards|avx2|vbmi2` caps the move-threat path alone, the
//!   rest of the tier untouched: every path records the same entries, so the
//!   search is the same and only the speed can differ.
//!
//! On a build whose target already pins the top tier (`kernels::STATIC_TOP`,
//! e.g. `target-cpu=native` on a Zen 4/5 box), the hot call sites bypass the
//! table entirely; the resolver pins the same choice so the weight permutation
//! stays consistent, and `GAIA_SIMD` is reported inert.

use std::sync::OnceLock;

use crate::movegen;
use crate::nnue::L1_SIZE;
use crate::nnue::accumulator::AccDelta;
use crate::nnue::forward;
use crate::nnue::kernels::{self, k_avx2, k_avx512, k_avxvnni, k_scalar, k_vnni512, threat512};
use crate::nnue::move_threats::{self, MoveThreats};
use crate::nnue::network::{Aligned, NNUEParams, PermKind};
use crate::position::Position;
use crate::types::{ArrayBuf, Color, MAX_MOVES, ScoredMove};

// ============================================================
// Tiers and axes
// ============================================================

/// SIMD register-width tier for the NNUE kernels, ascending.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum SimdTier {
    Scalar,
    Avx2,
    /// AVX2 with the VEX `vpdpbusd` of AVX-VNNI: Intel cores since the 12th
    /// generation, which have no AVX-512. Same register width and weight layout
    /// as AVX2.
    AvxVnni,
    Avx512,
    Vnni512,
}

impl SimdTier {
    pub fn name(self) -> &'static str {
        match self {
            SimdTier::Scalar => "scalar",
            SimdTier::Avx2 => "avx2",
            SimdTier::AvxVnni => "avxvnni",
            SimdTier::Avx512 => "avx512",
            SimdTier::Vnni512 => "vnni512",
        }
    }
}

/// `find_nnz` algorithm — an axis independent from the tier: Cascade Lake has
/// VNNI without VBMI2, so the compress form cannot be implied by `Vnni512`.
/// All three produce the same indices; only the speed differs. The compress form
/// runs inside the activation (`forward_dense_vbmi2`), in lane order rather than
/// sorted, which the sparse L1 does not mind.
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

/// How a played move records its threat changes — one more axis: the one-register ray
/// view needs VBMI (byte permutation) and VBMI2 (byte compression), which the AVX-512
/// tiers do not imply (Skylake-X and Cascade Lake have neither). All three record the
/// same entries; only the speed differs.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum MoveThreatsKind {
    Bitboards,
    Avx2View,
    Vbmi2View,
}

impl MoveThreatsKind {
    pub fn name(self) -> &'static str {
        match self {
            MoveThreatsKind::Bitboards => "bitboards",
            MoveThreatsKind::Avx2View => "avx2 view",
            MoveThreatsKind::Vbmi2View => "vbmi2 view",
        }
    }
}

/// How the quiet generator writes its moves — the last axis: writing a whole target
/// set at once needs VBMI2 byte compression, which Skylake-X and Cascade Lake lack.
/// Both write the same moves in the same order; only the speed differs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MoveListKind {
    Serial,
    Compress,
}

impl MoveListKind {
    pub fn name(self) -> &'static str {
        match self {
            MoveListKind::Serial => "one by one",
            MoveListKind::Compress => "vpcompressb",
        }
    }
}

// ============================================================
// The dispatch table
// ============================================================

/// Function pointers into the resolved kernel instance, plus the metadata the
/// loader (`perm`) and the `info` report need.
pub struct CpuDispatch {
    pub forward_dense: unsafe fn(&NNUEParams, &Aligned<[[i16; L1_SIZE]; 2]>, Color, usize) -> i32,
    pub find_nnz: unsafe fn(&Aligned<[u8; L1_SIZE]>, *mut u16) -> usize,
    pub threat_batch: unsafe fn(*const i16, *mut i16, *const crate::nnue::network::ThreatRow, &[u32], &[u32]),
    pub combined_batch: crate::nnue::kernels::CombinedBatchFn,
    pub(crate) move_threats: unsafe fn(&Position, &AccDelta, &mut MoveThreats),
    pub(crate) generate_quiets: unsafe fn(&Position, &mut ArrayBuf<ScoredMove, MAX_MOVES>) -> usize,
    pub tier: SimdTier,
    pub nnz: NnzKind,
    pub threats: MoveThreatsKind,
    pub movelist: MoveListKind,
    pub perm: PermKind,
    pub attacks: AttackKind,
    /// AVX-512 Kogge-Stone setwise slider fill available.
    pub setwise512: bool,
    /// `Some(cap)` when `GAIA_SIMD` lowered the tier below what was detected.
    pub forced: Option<SimdTier>,
    /// `GAIA_PEXT` override was applied.
    pub pext_forced: bool,
    /// `GAIA_THREATS` lowered the move-threat path.
    pub threats_forced: bool,
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
        debug_assert!(
            DISPATCH.get().is_some(),
            "cpu dispatch read before init_cpu_dispatch"
        );
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

/// The quiet-move writer actually running: a build that pins VBMI2 calls the
/// compression writer directly, past the table.
pub fn effective_movelist() -> MoveListKind {
    if cfg!(all(
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi2"
    )) {
        MoveListKind::Compress
    } else {
        get_or_init().movelist
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
    pub avxvnni: bool,
    pub avx512f: bool,
    pub avx512bw: bool,
    pub avx512vl: bool,
    pub avx512vnni: bool,
    pub avx512vbmi: bool,
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
        avxvnni: std::arch::is_x86_feature_detected!("avxvnni"),
        avx512f: std::arch::is_x86_feature_detected!("avx512f"),
        avx512bw: std::arch::is_x86_feature_detected!("avx512bw"),
        avx512vl: std::arch::is_x86_feature_detected!("avx512vl"),
        avx512vnni: std::arch::is_x86_feature_detected!("avx512vnni"),
        avx512vbmi: std::arch::is_x86_feature_detected!("avx512vbmi"),
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
    let family = if base_family == 0xF {
        base_family + ext_family
    } else {
        base_family
    };
    (true, family < 0x19)
}

fn read_simd_cap() -> Option<SimdTier> {
    let v = std::env::var("GAIA_SIMD").ok()?;
    match v.to_ascii_lowercase().as_str() {
        "scalar" => Some(SimdTier::Scalar),
        "avx2" => Some(SimdTier::Avx2),
        "avxvnni" => Some(SimdTier::AvxVnni),
        "avx512" => Some(SimdTier::Avx512),
        "vnni512" => Some(SimdTier::Vnni512),
        other => {
            eprintln!("info string GAIA_SIMD '{other}' ignored (scalar|avx2|avxvnni|avx512|vnni512)");
            None
        }
    }
}

fn read_threats_override() -> Option<MoveThreatsKind> {
    let v = std::env::var("GAIA_THREATS").ok()?;
    match v.to_ascii_lowercase().as_str() {
        "bitboards" => Some(MoveThreatsKind::Bitboards),
        "avx2" => Some(MoveThreatsKind::Avx2View),
        "vbmi2" => Some(MoveThreatsKind::Vbmi2View),
        other => {
            eprintln!("info string GAIA_THREATS '{other}' ignored (bitboards|avx2|vbmi2)");
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
    threats: MoveThreatsKind,
    movelist: MoveListKind,
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
    } else if avx2_ok && caps.avxvnni {
        SimdTier::AvxVnni
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
    // AVX-VNNI is the one tier the 512-bit ones do not imply (Skylake-X, Ice Lake and
    // Zen 4 lack it): a cap that lands on it runs AVX2 there instead of faulting.
    if tier == SimdTier::AvxVnni && !caps.avxvnni {
        tier = SimdTier::Avx2;
    }

    let nnz = if tier >= SimdTier::Avx512 && caps.avx512vbmi2 {
        NnzKind::Compress512
    } else if tier >= SimdTier::Avx2 {
        NnzKind::Avx2Table
    } else {
        NnzKind::Generic
    };

    let threats = if tier >= SimdTier::Avx512 && caps.avx512vbmi && caps.avx512vbmi2 {
        MoveThreatsKind::Vbmi2View
    } else if tier >= SimdTier::Avx2 {
        MoveThreatsKind::Avx2View
    } else {
        MoveThreatsKind::Bitboards
    };

    let movelist = if tier >= SimdTier::Avx512 && caps.avx512vbmi2 {
        MoveListKind::Compress
    } else {
        MoveListKind::Serial
    };

    let perm = match tier {
        SimdTier::Vnni512 | SimdTier::Avx512 => PermKind::File512,
        SimdTier::AvxVnni | SimdTier::Avx2 => PermKind::Avx2,
        SimdTier::Scalar => PermKind::Linear,
    };

    // PEXT rides its own axis: it needs the hardware bit, and can never be
    // forced onto a CPU that lacks it (that would be an illegal instruction,
    // not a slow one). By default it stays off on AMD: microcoded below Zen 3,
    // and on Zen 3+ the AVX2 path measured consistently faster than PEXT in
    // the dispatched binary (+1.2% NPS on Zen 5, 12/12 interleaved PGO runs —
    // plausibly the 850 KB PEXT tables against AVX2's 7.5 KB in a cache
    // already crowded by the network). Intel keeps PEXT, unmeasured here.
    let pext_default = caps.bmi2 && !caps.amd_slow_pext && !(caps.amd && static_avx2);
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
        threats,
        movelist,
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
    let Selection {
        tier,
        nnz,
        threats,
        movelist,
        perm,
        attacks,
        setwise512,
        forced,
        pext_forced,
    } = choose(
        caps,
        cap,
        pext_ov,
        kernels::STATIC_TOP,
        cfg!(target_feature = "avx2"),
    );

    let find_nnz = match nnz {
        NnzKind::Compress512 => {
            forward::find_nnz_compress512
                as unsafe fn(&Aligned<[u8; L1_SIZE]>, *mut u16) -> usize
        }
        NnzKind::Avx2Table => forward::find_nnz_avx2,
        NnzKind::Generic => k_scalar::find_nnz_generic,
    };

    // Like the tier cap, the request can only lower the choice; a build that pins VBMI2
    // at compile time records straight through it and never reads the table.
    let (threats, threats_forced) = match read_threats_override() {
        Some(k) if k < threats => (k, true),
        _ => (threats, false),
    };
    let move_threats = match threats {
        MoveThreatsKind::Vbmi2View => {
            move_threats::x86::record_vbmi2 as unsafe fn(&Position, &AccDelta, &mut MoveThreats)
        }
        MoveThreatsKind::Avx2View => move_threats::x86::record_avx2,
        MoveThreatsKind::Bitboards => move_threats::record_scalar,
    };
    let generate_quiets = match movelist {
        MoveListKind::Compress => {
            movegen::x86::quiets_compress
                as unsafe fn(&Position, &mut ArrayBuf<ScoredMove, MAX_MOVES>) -> usize
        }
        MoveListKind::Serial => movegen::quiets_serial,
    };

    match tier {
        SimdTier::Vnni512 => CpuDispatch {
            forward_dense: if nnz == NnzKind::Compress512 {
                k_vnni512::forward_dense_vbmi2
            } else {
                k_vnni512::forward_dense
            },
            find_nnz,
            threat_batch: threat512::threat_batch,
            combined_batch: threat512::combined_batch,
            move_threats,
            generate_quiets,
            tier,
            nnz,
            threats,
            movelist,
            perm,
            attacks,
            setwise512,
            forced,
            pext_forced,
            threats_forced,
        },
        SimdTier::Avx512 => CpuDispatch {
            forward_dense: if nnz == NnzKind::Compress512 {
                k_avx512::forward_dense_vbmi2
            } else {
                k_avx512::forward_dense
            },
            find_nnz,
            threat_batch: threat512::threat_batch,
            combined_batch: threat512::combined_batch,
            move_threats,
            generate_quiets,
            tier,
            nnz,
            threats,
            movelist,
            perm,
            attacks,
            setwise512,
            forced,
            pext_forced,
            threats_forced,
        },
        SimdTier::AvxVnni => CpuDispatch {
            forward_dense: k_avxvnni::forward_dense,
            find_nnz,
            threat_batch: k_avxvnni::threat_batch,
            combined_batch: k_avxvnni::combined_batch,
            move_threats,
            generate_quiets,
            tier,
            nnz,
            threats,
            movelist,
            perm,
            attacks,
            setwise512,
            forced,
            pext_forced,
            threats_forced,
        },
        SimdTier::Avx2 => CpuDispatch {
            forward_dense: k_avx2::forward_dense,
            find_nnz,
            threat_batch: k_avx2::threat_batch,
            combined_batch: k_avx2::combined_batch,
            move_threats,
            generate_quiets,
            tier,
            nnz,
            threats,
            movelist,
            perm,
            attacks,
            setwise512,
            forced,
            pext_forced,
            threats_forced,
        },
        SimdTier::Scalar => CpuDispatch {
            forward_dense: k_scalar::forward_dense,
            find_nnz,
            threat_batch: k_scalar::threat_batch,
            combined_batch: k_scalar::combined_batch,
            move_threats,
            generate_quiets,
            tier,
            nnz,
            threats,
            movelist,
            perm,
            attacks,
            setwise512,
            forced,
            pext_forced,
            threats_forced,
        },
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn caps(avx2: bool, fma: bool, f: bool, bw: bool, vl: bool, vnni: bool, vbmi2: bool) -> Caps {
        Caps {
            avx2,
            fma,
            avxvnni: false,
            avx512f: f,
            avx512bw: bw,
            avx512vl: vl,
            avx512vnni: vnni,
            avx512vbmi: vbmi2,
            avx512vbmi2: vbmi2,
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
        assert_eq!(s.threats, MoveThreatsKind::Vbmi2View);
        assert_eq!(s.movelist, MoveListKind::Compress);
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
        // No byte permutation either: the move threats take the AVX2 view.
        assert_eq!(s.threats, MoveThreatsKind::Avx2View);
        // Nor byte compression: the quiet moves are written one by one.
        assert_eq!(s.movelist, MoveListKind::Serial);
    }

    #[test]
    fn the_vbmi2_view_needs_vbmi_as_well() {
        let mut c = caps(true, true, true, true, true, true, true);
        c.avx512vbmi = false;
        assert_eq!(pick(c, None).threats, MoveThreatsKind::Avx2View);
    }

    #[test]
    fn the_move_threats_follow_a_lowered_tier() {
        let all = caps(true, true, true, true, true, true, true);
        assert_eq!(pick(all, Some(SimdTier::Avx2)).threats, MoveThreatsKind::Avx2View);
        assert_eq!(pick(all, Some(SimdTier::Scalar)).threats, MoveThreatsKind::Bitboards);
    }

    #[test]
    fn the_compressed_move_lists_follow_a_lowered_tier() {
        let all = caps(true, true, true, true, true, true, true);
        assert_eq!(pick(all, Some(SimdTier::Avx512)).movelist, MoveListKind::Compress);
        assert_eq!(pick(all, Some(SimdTier::Avx2)).movelist, MoveListKind::Serial);
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
    fn alder_lake_takes_the_avx_vnni_tier_with_the_avx2_layout() {
        // AVX2 and AVX-VNNI, no AVX-512: every Intel core since the 12th generation.
        let mut c = caps(true, true, false, false, false, false, false);
        c.avxvnni = true;
        let s = pick(c, None);
        assert_eq!(s.tier, SimdTier::AvxVnni);
        assert_eq!(s.perm, PermKind::Avx2);
        assert_eq!(s.nnz, NnzKind::Avx2Table);
        assert_eq!(s.threats, MoveThreatsKind::Avx2View);
        assert_eq!(s.movelist, MoveListKind::Serial);
        // The cap can still take it back to plain AVX2.
        assert_eq!(pick(c, Some(SimdTier::Avx2)).tier, SimdTier::Avx2);
    }

    #[test]
    fn the_512_bit_tiers_outrank_avx_vnni() {
        // Zen 5 and Sapphire Rapids have both.
        let mut c = caps(true, true, true, true, true, true, true);
        c.avxvnni = true;
        assert_eq!(pick(c, None).tier, SimdTier::Vnni512);
        assert_eq!(pick(c, Some(SimdTier::AvxVnni)).tier, SimdTier::AvxVnni);
    }

    #[test]
    fn an_avx_vnni_cap_on_a_cpu_without_it_runs_avx2() {
        // Zen 4: AVX-512 VNNI, no AVX-VNNI. Capping to avxvnni must not pick
        // kernels this CPU cannot execute.
        let c = caps(true, true, true, true, true, true, true);
        let s = pick(c, Some(SimdTier::AvxVnni));
        assert_eq!(s.tier, SimdTier::Avx2);
        assert_eq!(s.perm, PermKind::Avx2);
    }

    #[test]
    fn haswell_lands_on_avx2_with_its_permutation() {
        let s = pick(caps(true, true, false, false, false, false, false), None);
        assert_eq!(s.tier, SimdTier::Avx2);
        assert_eq!(s.nnz, NnzKind::Avx2Table);
        assert_eq!(s.threats, MoveThreatsKind::Avx2View);
        assert_eq!(s.perm, PermKind::Avx2);
        assert!(!s.setwise512);
    }

    #[test]
    fn a_cpu_without_avx2_falls_back_to_scalar_and_linear_weights() {
        let s = choose(
            caps(false, false, false, false, false, false, false),
            None,
            None,
            false,
            false,
        );
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
            SimdTier::AvxVnni | SimdTier::Avx2 => PermKind::Avx2,
            SimdTier::Scalar => PermKind::Linear,
        };
        assert_eq!(d.perm, expected_perm);
    }
}
