//! Statistical functions for bench analysis.
//!
//! Zero external dependencies — all formulas implemented from scratch.
//! Used by `bench --stats` for robust NPS measurement.

// ── Data structures ──────────────────────────────────────────────────

/// Aggregated statistics for a sample of f64 values.
pub(crate) struct Stats {
    pub n: usize,
    pub median: f64,
    pub trimmed_mean: f64,
    pub cv_pct: f64,
    pub ci_lo: f64,
    pub ci_hi: f64,
    pub outliers: usize,
}

/// Per-position statistics.
pub(crate) struct PositionStats {
    pub fen: String,
    pub median_nps: f64,
    pub cv_pct: f64,
}

// ── Core statistical functions ───────────────────────────────────────

/// Median of a slice (sorts in place).
pub(crate) fn median(data: &mut [f64]) -> f64 {
    debug_assert!(!data.is_empty());
    data.sort_by(f64::total_cmp);
    let n = data.len();
    if n % 2 == 1 {
        data[n / 2]
    } else {
        (data[n / 2 - 1] + data[n / 2]) / 2.0
    }
}

/// Median Absolute Deviation: `median(|xi - median(x)|)`.
#[cfg(test)]
pub(crate) fn mad(data: &mut [f64]) -> f64 {
    let med = median(data);
    let mut abs_devs: Vec<f64> = data.iter().map(|&x| (x - med).abs()).collect();
    median(&mut abs_devs)
}

/// Robust standard deviation estimate: `1.4826 * MAD`.
/// Consistent estimator of sigma under normality.
fn mad_sigma(mad_val: f64) -> f64 {
    1.4826 * mad_val
}

/// 20% trimmed mean. `sorted` must be sorted ascending.
/// If n < 5, falls back to regular mean.
pub(crate) fn trimmed_mean_20(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n < 5 {
        return sorted.iter().sum::<f64>() / n as f64;
    }
    let k = n / 5; // floor(0.2 * n)
    let trimmed = &sorted[k..n - k];
    trimmed.iter().sum::<f64>() / trimmed.len() as f64
}

/// Arithmetic mean.
pub(crate) fn mean(data: &[f64]) -> f64 {
    debug_assert!(!data.is_empty());
    data.iter().sum::<f64>() / data.len() as f64
}

/// Sample standard deviation (Bessel-corrected, n-1 denominator).
pub(crate) fn std_dev(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let m = mean(data);
    let var = data.iter().map(|&x| (x - m) * (x - m)).sum::<f64>() / (data.len() - 1) as f64;
    var.sqrt()
}

/// Detect outliers using the Modified Z-Score (Iglewicz & Hoaglin).
/// `M_i = 0.6745 * (x_i - median) / MAD`. Outlier if `|M_i| > 3.5`.
/// Returns indices of outlier values.
pub(crate) fn detect_outliers(data: &[f64], med: f64, mad_val: f64) -> Vec<usize> {
    if mad_val < 1e-12 {
        return Vec::new(); // all identical
    }
    data.iter()
        .enumerate()
        .filter(|&(_, &x)| (0.6745 * (x - med) / mad_val).abs() > 3.5)
        .map(|(i, _)| i)
        .collect()
}

/// Student-t 95% CI: `center +/- t_{0.025, n-1} * sigma / sqrt(n)`.
/// Uses robust sigma (MAD-based) for the spread estimate.
pub(crate) fn confidence_interval_95(center: f64, sigma: f64, n: usize) -> (f64, f64) {
    if n < 2 || sigma < 1e-12 {
        return (center, center);
    }
    let df = n - 1;
    let t = if df <= 30 { T_025[df] } else { 1.96 };
    let margin = t * sigma / (n as f64).sqrt();
    (center - margin, center + margin)
}

/// Compute all statistics from a slice of values.
pub(crate) fn compute_stats(values: &[f64]) -> Stats {
    debug_assert!(!values.is_empty());

    let mut data = values.to_vec();
    let med = median(&mut data);
    let mad_val = {
        let mut abs_devs: Vec<f64> = values.iter().map(|&x| (x - med).abs()).collect();
        median(&mut abs_devs)
    };
    let sigma = mad_sigma(mad_val);

    // data is now sorted from the median call
    let tm = trimmed_mean_20(&data);
    let sd = std_dev(values);
    let m = mean(values);
    let cv = if m.abs() > 1e-12 { 100.0 * sd / m } else { 0.0 };

    let outliers = detect_outliers(values, med, mad_val);
    let (ci_lo, ci_hi) = confidence_interval_95(tm, sigma, values.len());

    Stats {
        n: values.len(),
        median: med,
        trimmed_mean: tm,
        cv_pct: cv,
        ci_lo,
        ci_hi,
        outliers: outliers.len(),
    }
}

// ── Student-t quantile table ─────────────────────────────────────────

/// `t_{0.025, df}` for df = 1..30. Index 0 unused. For 95% two-sided CI.
const T_025: [f64; 31] = [
    0.0,    // df=0 (unused)
    12.706, // df=1
    4.303,  // df=2
    3.182,  // df=3
    2.776,  // df=4
    2.571,  // df=5
    2.447,  // df=6
    2.365,  // df=7
    2.306,  // df=8
    2.262,  // df=9
    2.228,  // df=10
    2.201,  // df=11
    2.179,  // df=12
    2.160,  // df=13
    2.145,  // df=14
    2.131,  // df=15
    2.120,  // df=16
    2.110,  // df=17
    2.101,  // df=18
    2.093,  // df=19
    2.086,  // df=20
    2.080,  // df=21
    2.074,  // df=22
    2.069,  // df=23
    2.064,  // df=24
    2.060,  // df=25
    2.056,  // df=26
    2.052,  // df=27
    2.048,  // df=28
    2.045,  // df=29
    2.042,  // df=30
];

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_median_odd() {
        let mut data = vec![3.0, 1.0, 2.0];
        assert_eq!(median(&mut data), 2.0);
    }

    #[test]
    fn test_median_even() {
        let mut data = vec![4.0, 1.0, 3.0, 2.0];
        assert_eq!(median(&mut data), 2.5);
    }

    #[test]
    fn test_median_single() {
        let mut data = vec![42.0];
        assert_eq!(median(&mut data), 42.0);
    }

    #[test]
    fn test_mad() {
        // data = [1, 2, 3, 4, 5], median=3, |xi-3| = [2,1,0,1,2], MAD=1
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(mad(&mut data), 1.0);
    }

    #[test]
    fn test_mad_with_outlier() {
        // data = [1, 2, 3, 4, 100], median=3, |xi-3| = [2,1,0,1,97], MAD=1
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 100.0];
        assert_eq!(mad(&mut data), 1.0);
    }

    #[test]
    fn test_trimmed_mean() {
        // 10 values, trim 2 from each end -> [3,4,5,6,7,8], mean = 5.5
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0];
        let tm = trimmed_mean_20(&sorted);
        assert!((tm - 5.5).abs() < 1e-10);
    }

    #[test]
    fn test_trimmed_mean_small() {
        // n < 5: falls back to regular mean
        let sorted = vec![1.0, 2.0, 3.0];
        assert!((trimmed_mean_20(&sorted) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_outlier_detection() {
        let data = vec![100.0, 101.0, 99.0, 100.0, 500.0];
        let med = 100.0;
        let mad_val = 1.0;
        let outliers = detect_outliers(&data, med, mad_val);
        assert_eq!(outliers, vec![4]);
    }

    #[test]
    fn test_outlier_none() {
        let data = vec![100.0, 101.0, 99.0, 100.5, 100.2];
        let med = 100.0;
        let mad_val = 0.5;
        let outliers = detect_outliers(&data, med, mad_val);
        assert!(outliers.is_empty());
    }

    #[test]
    fn test_outlier_mad_zero() {
        // All identical: MAD=0, no outliers
        let data = vec![42.0, 42.0, 42.0];
        let outliers = detect_outliers(&data, 42.0, 0.0);
        assert!(outliers.is_empty());
    }

    #[test]
    fn test_ci_basic() {
        // df=4, t_{0.025,4} = 2.776
        let (lo, hi) = confidence_interval_95(100.0, 10.0, 5);
        // margin = 2.776 * 10 / sqrt(5) = 12.414
        assert!((lo - 87.586).abs() < 0.1);
        assert!((hi - 112.414).abs() < 0.1);
    }

    #[test]
    fn test_ci_single_value() {
        let (lo, hi) = confidence_interval_95(100.0, 0.0, 1);
        assert_eq!(lo, 100.0);
        assert_eq!(hi, 100.0);
    }

    #[test]
    fn test_compute_stats() {
        let values = vec![100.0, 102.0, 98.0, 101.0, 99.0];
        let s = compute_stats(&values);
        assert_eq!(s.n, 5);
        assert!((s.median - 100.0).abs() < 1e-10);
        assert!(s.cv_pct < 5.0);
        assert!(s.ci_lo < s.median);
        assert!(s.ci_hi > s.median);
    }

    #[test]
    fn test_std_dev() {
        // mean=5.0, deviations² = [9,1,1,1,0,0,4,16] = 32, var=32/7≈4.571, sd≈2.138
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let sd = std_dev(&data);
        assert!((sd - 2.138).abs() < 0.01);
    }
}
