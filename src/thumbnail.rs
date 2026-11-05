//! Spectral thumbnail construction and derived envelopes.
//!
//! Ports `CPfPcmThumbnail`, `CPcmThumbnailBuilder::BuildSpecEng`,
//! `BuildPcmThumbnailRange` (the note-detector thumbnail),
//! `PeakHoldThumbnail` and `SmoothThumbnail`.
//! Spec: docs/02-algorithm-spec.md §2, §3.

use crate::config::FFT_SIZE;
use crate::fft::SpecEngFft;

/// Samples per measure for a given tempo.
///
/// `SamPerMeas(bpm) = (u32)(sample_rate / ((0.25 * bpm) / 60))`
/// which simplifies to `sample_rate * 240 / bpm`.
///
/// Computed in `f32` to match the original's single-precision reciprocal.
#[inline]
pub fn sam_per_meas(bpm: f32, sample_rate: u32) -> u32 {
    if bpm == 0.0 {
        return 0;
    }
    let meas_per_sec = (0.25f32 * bpm) / 60.0f32;
    if meas_per_sec == 0.0 {
        return 0;
    }
    let v = (sample_rate as f32) / meas_per_sec;
    if !v.is_finite() || v <= 0.0 {
        return 0;
    }
    // `_mm_cvttps_epi32` + saturation == truncating cast for in-range values.
    if v >= u32::MAX as f32 {
        u32::MAX
    } else {
        v as u32
    }
}

/// A decimated, per-element series derived from the PCM.
#[derive(Debug, Clone, PartialEq)]
pub struct Thumbnail {
    pub data: Vec<f32>,
    /// Total PCM samples the thumbnail was built from.
    pub total_samples: i64,
    /// Sample rate of that PCM.
    pub sample_rate: u32,
    /// How many PCM samples each element summarises.
    pub samples_per_element: u16,
}

impl Thumbnail {
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// `GetAt` — out-of-range reads return 0.0, exactly like the original.
    #[inline]
    pub fn get(&self, idx: i64) -> f32 {
        if idx < 0 || idx as usize >= self.data.len() {
            0.0
        } else {
            self.data[idx as usize]
        }
    }

    /// `GetMinValue` — 0.0 for an empty thumbnail, else the minimum element.
    pub fn min_value(&self) -> f32 {
        if self.data.is_empty() {
            0.0
        } else {
            self.data.iter().copied().fold(f32::INFINITY, f32::min)
        }
    }

    /// `GetMaxValue`
    pub fn max_value(&self) -> f32 {
        if self.data.is_empty() {
            0.0
        } else {
            self.data.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        }
    }

    /// `GetElePerMeas` — elements in one measure at `bpm`.
    pub fn ele_per_meas(&self, bpm: f32) -> u32 {
        if bpm == 0.0 {
            return self.data.len() as u32;
        }
        let spm = sam_per_meas(bpm, self.sample_rate);
        spm / (self.samples_per_element.max(1) as u32)
    }

    /// Distance in PCM samples between two element indices.
    #[inline]
    pub fn to_samples(&self, element_idx: i64) -> i64 {
        element_idx * self.samples_per_element as i64
    }
}

/// Build the spectral thumbnail: one element per `FFT_SIZE` PCM samples.
///
/// `element[f] = Σ_{k=1}^{N/2-1} |FFT_N(x[f·N .. f·N+N-1] · hamming)|`
///
/// The transform is the original's own scaled real FFT (see [`SpecEngFft`]),
/// and the bins are accumulated left-to-right in `f32` exactly like
/// `GetSpecEnergy` (which starts at bin 1, skipping DC).
pub fn spec_thumbnail(samples: &[f32], sample_rate: u32) -> Thumbnail {
    let n = FFT_SIZE;
    let count = samples.len() / n;
    let mut data = Vec::with_capacity(count);

    let mut fft = SpecEngFft::new(n);
    let mut mag = vec![0f32; n / 2];

    for f in 0..count {
        let src = &samples[f * n..f * n + n];
        fft.magnitudes(src, &mut mag);
        // Bins 1..=N/2-1 — DC and Nyquist are excluded (spec §2.3).
        let mut acc = 0f32;
        for m in &mag[1..n / 2] {
            acc += *m;
        }
        data.push(acc);
    }

    Thumbnail {
        data,
        total_samples: samples.len() as i64,
        sample_rate,
        samples_per_element: n as u16,
    }
}

/// One biquad section of the note detector's high-pass (`CIIRFilt::setfilt` +
/// `spbiln`, re/16): a 120 Hz Q=1.1 Butterworth-style section.
///
/// Digital form (`fl` at re/16): `w[n] = x[n] - k0·w[n-1] - k1·w[n-2]`,
/// `y[n] = b0·w[n] + b1·w[n-1] + b2·w[n-2]`, all `f32`.
#[derive(Clone, Copy)]
struct HpBiquad {
    b: [f32; 3],
    k: [f32; 2],
}

impl HpBiquad {
    fn design(freq: f32, q: f32, sample_rate: u32) -> Self {
        // setfilt: t = tan(freq·π·(1/sr)) in double — note the original's
        // multiplication order (freq·π)·(1/sr), kept for bit-parity.
        let t = ((freq as f64) * std::f64::consts::PI * (1.0 / sample_rate as f64)).tan();
        let t2 = (t * t) as f32;
        let num = [t2, 0.0f32, 0.0f32];
        let den = [t2, (q as f64 * t) as f32, 1.0f32];

        // spbiln: substitute s ← (1-z)/(1+z). M[m][j] is the coefficient of
        // z^-j in (1-z)^m (1+z)^(2-m). Products round to f32 before the
        // accumulators, exactly like the original's f32 prototype arrays.
        let matrix = [[1.0, 2.0, 1.0], [1.0, 0.0, -1.0], [1.0, -2.0, 1.0]];
        let mut b = [0f32; 3];
        let mut dd = [0f64; 3];
        for j in 0..3 {
            let mut acc = 0f32;
            for m in 0..3 {
                acc += (matrix[m][j] * num[m] as f64) as f32;
                dd[j] += ((matrix[m][j] * den[m] as f64) as f32) as f64;
            }
            b[j] = acc;
        }
        let g = dd[0] as f32; // denominator z^0 coefficient
        for x in b.iter_mut() {
            *x /= g;
        }
        let k = [
            (dd[1] / g as f64) as f32,
            (dd[2] / g as f64) as f32,
        ];
        HpBiquad { b, k }
    }
}

/// State for one running biquad section.
#[derive(Clone, Copy)]
struct BiquadState {
    /// Direct form II transposed state, matching the reference
    /// implementation's `lfilter` arithmetic bit for bit.
    s1: f32,
    s2: f32,
}

impl BiquadState {
    const fn new() -> Self {
        BiquadState { s1: 0.0, s2: 0.0 }
    }

    #[inline]
    fn run(&mut self, sec: &HpBiquad, x: f32) -> f32 {
        let y = sec.b[0] * x + self.s1;
        let s1 = sec.b[1] * x - sec.k[0] * y + self.s2;
        let s2 = sec.b[2] * x - sec.k[1] * y;
        self.s1 = s1;
        self.s2 = s2;
        y
    }
}

/// `BuildPcmThumbnailRange` (re/16) — the note-detector thumbnail (v35).
///
/// The PCM runs through the same 120 Hz Q=1.1 high-pass biquad **three times**
/// (the DLL cascades six 2nd-order sections; they are identical pairs of this
/// section), and each element keeps the peak `|x|` of its 256 samples.
/// Only the `FoldUpBpm` decision consumes this thumbnail, through the
/// smoothed envelope and the classic detector.
pub fn note_peak_thumbnail(samples: &[f32], sample_rate: u32) -> Thumbnail {
    let n = FFT_SIZE;
    let count = samples.len() / n;
    let mut data = Vec::with_capacity(count);
    if count == 0 {
        return Thumbnail {
            data,
            total_samples: samples.len() as i64,
            sample_rate,
            samples_per_element: n as u16,
        };
    }

    let section = HpBiquad::design(120.0, 1.1, sample_rate);
    let mut s0 = BiquadState::new();
    let mut s1 = BiquadState::new();
    let mut s2 = BiquadState::new();

    for f in 0..count {
        let mut peak = 0f32;
        for &x in &samples[f * n..f * n + n] {
            let y = s0.run(&section, x);
            let y = s1.run(&section, y);
            let y = s2.run(&section, y);
            let a = y.abs();
            if a > peak {
                peak = a;
            }
        }
        data.push(peak);
    }

    Thumbnail {
        data,
        total_samples: samples.len() as i64,
        sample_rate,
        samples_per_element: n as u16,
    }
}

/// `PeakHoldThumbnail` — in-place peak-hold envelope with linear decay.
///
/// The envelope is seeded with the thumbnail minimum and decays by
/// `peak / (elePerMeas(164) >> 4)` per element.
pub fn peak_hold(t: &mut Thumbnail) {
    let decay_div = t.ele_per_meas(164.0) >> 4;
    if decay_div == 0 {
        return;
    }
    let mut env = t.min_value();
    let mut step = 0f32;
    let decay_div = decay_div as f32;
    for i in 0..t.data.len() {
        let cur = t.data[i];
        if cur <= env {
            t.data[i] = env;
        } else {
            env = cur;
            step = cur / decay_div;
        }
        env -= step;
    }
}

/// `SmoothThumbnail` — centred moving average with window `min(count, 9)`.
///
/// ## Fidelity note
///
/// The original's sliding update adds `x[v18]` *after* pre-incrementing `v18`,
/// which starts at 9. Its first slide therefore consumes `x[10]` rather than
/// `x[9]`, permanently skipping `x[9]`. That is reproduced here verbatim rather
/// than "fixed", because the smoothed series feeds the `FoldUpBpm` decision and
/// we want behavioural fidelity. Sliding never engages for `count <= 9`, so the
/// quirk only shows up on longer inputs.
pub fn smoothed(t: &Thumbnail) -> Thumbnail {
    let count = t.data.len();
    let mut out = vec![0f32; count];
    if count == 0 {
        return Thumbnail {
            data: out,
            total_samples: t.total_samples,
            sample_rate: t.sample_rate,
            samples_per_element: t.samples_per_element,
        };
    }

    let window = if count < 9 { count } else { 9 };
    let mut sum = 0f32;
    for i in 0..window {
        sum += t.data[i];
    }

    let mut v17: i64 = 0;
    let mut v18: i64 = 9;
    let half = (window >> 1) as i64;
    let count_i = count as i64;
    let window_f = window as f32;

    let mut v7: i64 = 0;
    while v7 < count_i {
        out[v7 as usize] = sum / window_f;
        v7 += 1;
        if v7 > half && v7 < count_i - half - 1 {
            sum -= t.get(v17);
            v17 += 1;
            v18 += 1;
            sum += t.get(v18);
        }
    }

    Thumbnail {
        data: out,
        total_samples: t.total_samples,
        sample_rate: t.sample_rate,
        samples_per_element: t.samples_per_element,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sam_per_meas_matches_formula() {
        assert_eq!(sam_per_meas(120.0, 44100), 88_200);
        assert_eq!(sam_per_meas(82.0, 44100), 129_073);
        assert_eq!(sam_per_meas(164.0, 44100), 64_536);
        assert_eq!(sam_per_meas(0.0, 44100), 0);
    }

    #[test]
    fn empty_thumbnail_is_safe() {
        let t = Thumbnail {
            data: vec![],
            total_samples: 0,
            sample_rate: 44100,
            samples_per_element: 256,
        };
        assert_eq!(t.min_value(), 0.0);
        let s = smoothed(&t);
        assert!(s.is_empty());
    }

    #[test]
    fn smooth_of_constant_is_constant() {
        let t = Thumbnail {
            data: vec![3.0; 64],
            total_samples: 64 * 256,
            sample_rate: 44100,
            samples_per_element: 256,
        };
        let s = smoothed(&t);
        for (i, v) in s.data.iter().enumerate() {
            // The sliding update subtracts and adds the same constant, so the
            // running sum stays exact regardless of the off-by-one.
            assert!((v - 3.0).abs() < 1e-5, "element {i} = {v}");
        }
    }
}
