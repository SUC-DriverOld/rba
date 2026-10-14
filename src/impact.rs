//! Onset ("impact point") detection.
//!
//! Ports `CPfImpactPoints`, `CPfImpactFinder::FindImpactPoints`,
//! `FindImpactPointsClassic`, `BuildImpactPoints` and
//! `BuildImpactPointsClassic`. Spec: docs/02-algorithm-spec.md §4.
//!
//! Two different detectors exist in the original and they are **not**
//! interchangeable:
//!
//! * `build_impact_points` (non-"classic") compares an **upward slope**
//!   against a threshold and records the *current* index. It feeds the tempo
//!   histogram.
//! * `build_impact_points_classic` compares an **absolute magnitude ratio**
//!   and records the *peak* index. It only feeds `fold_up_bpm`.

use crate::thumbnail::{sam_per_meas, Thumbnail};

/// Refractory distance (in thumbnail elements) between two accepted onsets.
pub const IMPACT_GAP: u32 = 4;

/// An ordered set of onset positions, expressed in **PCM samples**.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ImpactPoints {
    /// Ascending sample positions.
    pub points: Vec<i64>,
    /// Measure-start marks; unused by the BPM path.
    pub measure_marks: Vec<i64>,
    pub sample_rate: u32,
    pub total_samples: i64,
    pub samples_per_element: u16,
}

impl ImpactPoints {
    pub fn new(sample_rate: u32, total_samples: i64) -> Self {
        ImpactPoints {
            points: Vec::new(),
            measure_marks: Vec::new(),
            sample_rate,
            total_samples,
            samples_per_element: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// `EnumPoints` — `None` when the index is out of range.
    #[inline]
    pub fn get(&self, idx: i64) -> Option<i64> {
        if idx < 0 || idx as usize >= self.points.len() {
            None
        } else {
            Some(self.points[idx as usize])
        }
    }

    pub fn sort(&mut self) {
        self.points.sort_unstable();
    }

    /// `SortAndFilterDupes`
    pub fn sort_and_filter_dupes(&mut self) {
        self.points.sort_unstable();
        self.points.dedup();
        self.measure_marks.sort_unstable();
        self.measure_marks.dedup();
    }

    pub fn sam_per_meas(&self, bpm: f32) -> u32 {
        sam_per_meas(bpm, self.sample_rate)
    }
}

/// `FindImpactPoints` — slope-based detector, descending scan.
///
/// Returns the number of onsets found; `out` receives `idx * samplesPerElement`.
pub fn find_impact_points(t: &Thumbnail, threshold: f32, gap: u32, out: &mut ImpactPoints) -> u32 {
    let n = t.data.len() as i64;
    if n == 0 {
        return 0;
    }
    out.points.clear();

    let spe = t.samples_per_element as i64;
    let gap_i = gap as i64;

    let mut idx = n - 1;
    let mut peak = n - 1;
    let mut refractory: i64 = 0;
    let mut count: u32 = 0;

    loop {
        let cur = t.get(idx);
        if refractory != 0 {
            refractory -= 1;
            peak = idx;
        } else {
            if cur > t.get(peak) {
                peak = idx;
            }
            // window_end == idx + gap, since both start at n-1+gap and step down.
            let lim = (n - 1).min(idx + gap_i);
            if peak > lim {
                peak = idx;
                let mut j = idx;
                while j < lim {
                    if t.get(j) > t.get(peak) {
                        peak = j;
                    }
                    j += 1;
                }
            }
            refractory = 0;
            let is_onset = if idx == peak {
                false
            } else {
                let slope = (t.get(peak) - cur) / ((peak - idx) as f32);
                slope > threshold
            };
            if is_onset {
                peak = idx;
                out.points.push(idx * spe);
                refractory = gap_i;
                count += 1;
            }
        }
        if idx == 0 {
            break;
        }
        idx -= 1;
    }

    count
}

/// `FindImpactPointsClassic` — magnitude-ratio detector, descending scan.
pub fn find_impact_points_classic(
    t: &Thumbnail,
    threshold: f32,
    gap: u32,
    out: &mut ImpactPoints,
) -> u32 {
    let n = t.data.len() as i64;
    if n == 0 {
        return 0;
    }
    out.points.clear();

    let spe = t.samples_per_element as i64;
    let gap_i = gap as i64;

    let mut idx = n - 1;
    let mut peak = n - 1;
    let mut refractory: i64 = 0;
    let mut count: u32 = 0;

    loop {
        let cur = t.get(idx);
        if refractory != 0 {
            refractory -= 1;
            peak = idx;
        } else {
            if cur > t.get(peak) {
                peak = idx;
            }
            let lim = (n - 1).min(idx + gap_i);
            if peak > lim {
                peak = idx;
                let mut j = idx;
                while j < lim {
                    if t.get(j) > t.get(peak) {
                        peak = j;
                    }
                    j += 1;
                }
            }
            refractory = 0;
            let limit = threshold * cur;
            if t.get(peak) > limit {
                out.points.push(peak * spe);
                refractory = gap_i;
                count += 1;
            }
        }
        if idx == 0 {
            break;
        }
        idx -= 1;
    }

    count
}

/// Threshold ladder shared by both builders: retry with a relaxed threshold
/// until enough onsets were found or `max_tries` is exhausted.
fn build_with_ladder(
    t: &Thumbnail,
    total_samples: i64,
    divisor: u32,
    initial_threshold: f32,
    max_tries: u32,
    classic: bool,
) -> ImpactPoints {
    let mut out = ImpactPoints::new(t.sample_rate, total_samples);
    out.samples_per_element = t.samples_per_element;

    let spm82 = sam_per_meas(82.0, t.sample_rate);
    if spm82 == 0 {
        return out;
    }
    let target = (total_samples / spm82 as i64) * divisor as i64;

    let mut found: u32 = 0;
    let mut threshold = initial_threshold;
    let mut tries = max_tries;

    while (found as i64) < target {
        tries -= 1;
        if tries == 0 {
            break;
        }
        found = if classic {
            find_impact_points_classic(t, threshold, IMPACT_GAP, &mut out)
        } else {
            find_impact_points(t, threshold, IMPACT_GAP, &mut out)
        };
        threshold /= 1.2;
    }
    out.sort();
    out
}

/// `BuildImpactPoints` — slope detector with a 63-step ladder from 2.4414062.
pub fn build_impact_points(t: &Thumbnail, total_samples: i64, divisor: u32) -> ImpactPoints {
    build_with_ladder(t, total_samples, divisor, 2.4414062, 63, false)
}

/// `BuildImpactPointsClassic` — magnitude detector, 20-step ladder from 30.0.
pub fn build_impact_points_classic(t: &Thumbnail, total_samples: i64, divisor: u32) -> ImpactPoints {
    build_with_ladder(t, total_samples, divisor, 30.0, 20, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(n: usize, v: f32) -> Thumbnail {
        Thumbnail {
            data: vec![v; n],
            total_samples: (n * 256) as i64,
            sample_rate: 44100,
            samples_per_element: 256,
        }
    }

    #[test]
    fn flat_signal_yields_no_onsets() {
        let t = flat(1000, 1.0);
        let mut out = ImpactPoints::new(44100, 1000 * 256);
        assert_eq!(find_impact_points(&t, 2.44, IMPACT_GAP, &mut out), 0);
        assert!(out.points.is_empty());
    }

    #[test]
    fn spike_yields_onset_at_sample_position() {
        let mut t = flat(64, 1.0);
        t.data[10] = 100.0;
        let mut out = ImpactPoints::new(44100, 64 * 256);
        let n = find_impact_points(&t, 2.44, IMPACT_GAP, &mut out);
        assert!(n >= 1, "expected an onset");
        // Positions are in PCM samples.
        assert!(out.points.iter().all(|p| p % 256 == 0));
    }

    #[test]
    fn out_of_range_get_returns_zero() {
        let t = flat(4, 5.0);
        assert_eq!(t.get(-1), 0.0);
        assert_eq!(t.get(4), 0.0);
        assert_eq!(t.get(3), 5.0);
    }
}
