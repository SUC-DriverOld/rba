//! Tempo estimation: phase-histogram window detector, median aggregation,
//! half-tempo folding and the short-clip override.
//!
//! Ports `CPfBpmDetector::DetectBpmWindow`, `DetectBaseBpm`, `FoldUpBpm` and
//! `CProfileBuilder::ShortClipBpmUpdate`.
//! Spec: docs/02-algorithm-spec.md §5, §6, §7.

use crate::impact::ImpactPoints;
use crate::thumbnail::sam_per_meas;

/// Tempo search lower bound (inclusive).
pub const BPM_MIN: f32 = 82.0;
/// Number of 0.1 BPM steps scanned (82.0 ..= 163.9).
pub const BPM_STEPS: u32 = 820;
/// Analysis window length in PCM samples: 20 s at 44100 Hz.
pub const WINDOW_SAMPLES: i64 = 882_000;
/// Phase-histogram resolution.
const PHASE_BINS: usize = 40;

/// `DetectBpmWindow` — score every candidate tempo by how strongly onset
/// phases align, and return the best one (0.0 when undecidable).
///
/// This is a phase histogram, **not** autocorrelation: for each candidate
/// tempo the onset positions are reduced modulo the beat period and bucketed.
pub fn detect_bpm_window(pts: &ImpactPoints, start: i64, length: i64, divisor: u32) -> f32 {
    let mut window: Vec<i64> = Vec::new();
    let mut i: i64 = 0;
    while let Some(p) = pts.get(i) {
        if start < p && start + length > p {
            window.push(p);
            if window.len() > 511 {
                break;
            }
        }
        i += 1;
    }
    let n = window.len();
    if n <= 1 {
        return 0.0;
    }

    let sample_rate = pts.sample_rate;
    let mut best_bpm = 0.0f32;
    let mut best_count: u16 = 0;

    let mut bpm = BPM_MIN;
    for _ in 0..BPM_STEPS {
        let spm = sam_per_meas(bpm, sample_rate) / divisor;
        if spm == 0 {
            break;
        }
        let v7 = (spm / 10) as i64;
        let v8 = spm / 40;
        if v8 == 0 {
            break;
        }
        let v8 = v8 as i64;
        let spm_i = spm as i64;

        let mut hist = [0u16; PHASE_BINS];
        for &p in &window {
            let v10 = p % spm_i;
            let v9 = if v7 < v10 { v10 - v7 } else { 0 };
            let mut j = (v9 / v8) as usize;
            while j < PHASE_BINS {
                let v12 = (j as i64) * v8;
                if v10 <= v12 {
                    break;
                }
                if v10 < v7 + v12 {
                    hist[j] += 1;
                }
                j += 1;
            }
        }
        for (k, &c) in hist.iter().enumerate() {
            let _ = k;
            if c > best_count {
                best_count = c;
                best_bpm = bpm;
            }
        }

        // The original accumulates `bpm + 0.1` with C's double promotion.
        bpm = (bpm as f64 + 0.1) as f32;
    }

    best_bpm
}

/// `DetectBaseBpm` — median of up to 18 window/resolution estimates.
///
/// Two passes per window (divisor 4 and 8) across 9 windows at 10 %..90 % of
/// the file, then the upper median (`buf[n >> 1]`).
pub fn detect_base_bpm(pts: &ImpactPoints) -> f32 {
    if pts.total_samples <= 0 {
        return 0.0;
    }
    let step = pts.total_samples / 10;

    let mut buf: Vec<f32> = Vec::with_capacity(18);
    for i in 1..10i64 {
        for divisor in [4u32, 8u32] {
            let v = detect_bpm_window(pts, step * i, WINDOW_SAMPLES, divisor);
            if v != 0.0 {
                buf.push(v);
            }
        }
    }
    if buf.is_empty() {
        return 0.0;
    }

    // Replicates the original's insertion-style ascending sort exactly.
    for i in 1..buf.len() {
        for j in 0..i {
            if buf[j] > buf[i] {
                buf.swap(i, j);
            }
        }
    }
    buf[buf.len() >> 1]
}

/// `FoldUpBpm` — double a low tempo when most onsets are ~one measure apart.
///
/// Only applies when `bpm <= 100`. The measure length is compared against
/// `[0.8, 1.2] · (samplesPerMeasure / 8)`; above 53 % of intervals in range the
/// detected "beat" is really a bar, so the tempo doubles.
pub fn fold_up_bpm(pts: &ImpactPoints, bpm: f32) -> f32 {
    // Direct translation of the original's `if (a3 <= 100.0) { ... }`: the
    // negated form is what makes NaN take the "return unchanged" path, exactly
    // like the C comparison does. Clippy's suggested rewrite would change that.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(bpm <= 100.0) {
        return bpm;
    }
    let spm = pts.sam_per_meas(bpm);
    let base = spm >> 3;
    // (u32)(((u64)base * 3435973837) >> 32) >> 2   ==  base / 5
    let v6 = ((((base as u64) * 3_435_973_837u64) >> 32) >> 2) as u32;
    let lo = base.wrapping_sub(v6);
    let hi = v6.wrapping_add(base);

    let Some(first) = pts.get(0) else {
        return bpm;
    };
    let mut prev = first;
    let mut in_range: u32 = 0;
    let mut iter_count: u32 = 0;

    let mut idx: i64 = 1;
    while let Some(cur) = pts.get(idx) {
        let delta = cur.wrapping_sub(prev) as u64;
        prev = cur;
        if (delta as u32) > lo && (delta as u32) < hi {
            in_range += 1;
        }
        iter_count += 1;
        idx += 1;
    }

    // v4 != 2  <=>  at least one interval was measured
    if iter_count > 0 && (in_range as f32 / iter_count as f32) * 100.0 > 53.0 {
        return bpm * 2.0;
    }
    bpm
}

/// `ShortClipBpmUpdate` — clips shorter than 30 s are forced into [82, 164.01]
/// by doubling/halving `240 / durationSeconds`.
///
/// For clips of 30 s or longer this is a no-op.
pub fn short_clip_bpm_update(sample_count: i64, sample_rate: u32, bpm: f32) -> f32 {
    if sample_rate == 0 {
        return bpm;
    }
    if !(30 * sample_rate as i64 > sample_count) {
        return bpm;
    }
    let secs = sample_count as f64 / sample_rate as f64;
    if secs == 0.0 {
        return 0.0;
    }
    let mut b = 240.0f64 / secs;
    const UPPER: f64 = 164.009_994_506_835_9;
    // The original clamps with the float literal `164.00999`, which is not
    // representable in f32 and rounds to the same bit pattern as `164.01`.
    const CLAMP_MAX: f32 = 164.01;
    if b >= 82.0 {
        let mut guard = 0;
        while b > UPPER && guard < 512 {
            b *= 0.5;
            guard += 1;
        }
    } else {
        let mut guard = 0;
        while b < 82.0 && guard < 512 {
            b *= 2.0;
            guard += 1;
        }
    }
    if b >= 82.0 {
        if b > UPPER {
            CLAMP_MAX
        } else {
            b as f32
        }
    } else {
        82.0
    }
}

/// Refine a coarse tempo to sub-0.1 precision from the onset positions.
///
/// # Why this exists
///
/// The recovered algorithm scans BPM on a fixed 0.1 grid (§5), so every value it
/// can produce ends in a zero — `159.90`, `119.80`, … That grid is an artefact of
/// how the original binary searched, not a property of the music. This stage
/// leaves the search (and therefore every decision the original made) untouched
/// and only re-derives the *period* from the onsets that actually landed on the
/// coarse beat grid.
///
/// # Method
///
/// Anchor on the coarse beat period `T0 = 60·sr / bpm`. Give each onset a beat
/// index `k = round(p / T0)` and keep those whose residual to that grid is under
/// a quarter beat — that rejects syncopated off-beats and half-beats. Then fit
/// `p ≈ φ + k·T` by least squares, and iterate: once the first pass tightens the
/// period, beats that had drifted outside the gate come back in.
///
/// Returns `(bpm, kept_onsets)`, falling back to the coarse value when the fit is
/// unsupported or moves more than 5 %.
///
/// **This is an enhancement, not part of the reverse-engineered algorithm.**
/// See docs/02-algorithm-spec.md §11.
pub fn refine_bpm(onsets: &ImpactPoints, bpm: f32, sample_rate: u32) -> (f64, usize) {
    let coarse = bpm as f64;
    // Negated form on purpose: it also rejects NaN, like every other guard in
    // this module (the original's comparisons behave the same way).
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(coarse > 0.0) || sample_rate == 0 || onsets.len() < 8 {
        return (coarse, 0);
    }

    let mut period = sample_rate as f64 * 60.0 / coarse;
    let mut kept = 0usize;

    for _ in 0..4 {
        if !period.is_finite() || period <= 0.0 {
            return (coarse, kept);
        }
        let tol = period / 4.0;

        // Sufficient statistics for a least-squares fit of (k, p) with intercept.
        let (mut n, mut sk, mut sp, mut skk, mut skp) = (0f64, 0f64, 0f64, 0f64, 0f64);
        for &p in &onsets.points {
            let pf = p as f64;
            let k = (pf / period).round();
            if (pf - k * period).abs() >= tol {
                continue;
            }
            n += 1.0;
            sk += k;
            sp += pf;
            skk += k * k;
            skp += k * pf;
        }
        kept = n as usize;
        if n < 4.0 {
            return (coarse, kept);
        }
        let denom = n * skk - sk * sk;
        if denom.abs() < 1e-9 {
            return (coarse, kept);
        }
        period = (n * skp - sk * sp) / denom;
    }

    let refined = sample_rate as f64 * 60.0 / period;
    if !refined.is_finite() || (refined - coarse).abs() / coarse > 0.05 {
        return (coarse, kept);
    }
    (refined, kept)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_clip_is_untouched_by_short_clip_update() {
        // 229.5 s at 44100 Hz
        let n = 10_120_950i64;
        assert_eq!(short_clip_bpm_update(n, 44100, 123.4), 123.4);
    }

    #[test]
    fn short_clip_is_folded_into_range() {
        // 12 s -> 240/12 = 20 BPM -> x2 until >= 82 -> 160
        let v = short_clip_bpm_update(12 * 44100, 44100, 0.0);
        assert!((v - 160.0).abs() < 0.01, "got {v}");
    }

    #[test]
    fn short_clip_respects_lower_bound() {
        // 100 s would be < 30s? no. use 29 s -> 240/29 = 8.27 -> x16 = 132.4
        let v = short_clip_bpm_update(29 * 44100, 44100, 0.0);
        assert!((82.0..=164.02).contains(&v), "got {v}");
    }

    #[test]
    fn fold_up_only_applies_below_100() {
        let pts = ImpactPoints::new(44100, 1_000_000);
        assert_eq!(fold_up_bpm(&pts, 120.0), 120.0);
    }
}
