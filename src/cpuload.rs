//! Process CPU utilization for the optional UCI `info cpuload` field.

use crate::time::Instant;

#[cfg(not(target_arch = "wasm32"))]
fn normalize(cpu_ns: u128, wall_ns: u128, threads: usize) -> Option<u32> {
    let capacity_ns = wall_ns.saturating_mul(threads.max(1) as u128);
    if capacity_ns == 0 {
        return None;
    }
    Some(
        cpu_ns
            .saturating_mul(1000)
            .checked_div(capacity_ns)
            .unwrap_or(0)
            .min(1000) as u32,
    )
}

/// Per-search baseline shared by all search threads.
pub struct CpuLoad {
    wall_start: Instant,
    #[cfg(not(target_arch = "wasm32"))]
    process_start: Option<cpu_time::ProcessTime>,
}

impl CpuLoad {
    pub fn new() -> Self {
        Self {
            wall_start: Instant::now(),
            #[cfg(not(target_arch = "wasm32"))]
            process_start: cpu_time::ProcessTime::try_now().ok(),
        }
    }

    /// Start a new sample window immediately before the workers begin searching.
    pub fn reset(&mut self) {
        self.wall_start = Instant::now();
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.process_start = cpu_time::ProcessTime::try_now().ok();
        }
    }

    /// Process utilization relative to the configured search capacity, in per mille.
    pub fn permille(&self, threads: usize) -> Option<u32> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let wall_ns = self.wall_start.elapsed().as_nanos();
            let cpu_ns = self.process_start?.try_elapsed().ok()?.as_nanos();
            return normalize(cpu_ns, wall_ns, threads);
        }

        #[cfg(target_arch = "wasm32")]
        {
            let _ = threads;
            None
        }
    }
}

impl Default for CpuLoad {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn process_load_is_a_uci_per_mille_value() {
        let load = CpuLoad::new();
        let mut value = 0u64;
        for i in 0..100_000 {
            value = value.wrapping_add(i);
        }
        std::hint::black_box(value);

        if let Some(load) = load.permille(1) {
            assert!(load <= 1000);
        }
    }

    #[test]
    fn load_is_normalized_by_configured_threads() {
        assert_eq!(normalize(4_000, 1_000, 4), Some(1000));
        assert_eq!(normalize(2_000, 1_000, 4), Some(500));
        assert_eq!(normalize(8_000, 1_000, 4), Some(1000));
        assert_eq!(normalize(1, 0, 1), None);
    }
}
