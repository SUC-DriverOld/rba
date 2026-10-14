//! Key detection — a port of [libKeyFinder](../libKeyFinder/) (GPL-3.0).
//!
//! Verified sample-for-sample against a build of the C++ library: see
//! `tools/keyfinder_reference/` and docs/04-testing.md.

// The tables below are copied verbatim from upstream, so some literals carry
// more digits than an f64 can hold, and the index loops mirror the original's
// array walks rather than being rewritten as iterators.
#![allow(clippy::excessive_precision, clippy::needless_range_loop)]

use std::f64::consts::PI;

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

/// Six octaves of twelve semitones.
const SEMITONES: usize = 12;
const OCTAVES: usize = 6;
const BANDS: usize = SEMITONES * OCTAVES;
const FFT_FRAME_SIZE: usize = 16384;
const HOP_SIZE: usize = FFT_FRAME_SIZE / 4;
/// `DIRECTSKSTRETCH` — widens the constant-Q filterbank.
const DIRECT_SK_STRETCH: f64 = 0.8;
const LPF_ORDER: usize = 160;
const LPF_FFT_SIZE: usize = 2048;

/// `FREQUENCIES[]` in constants.cpp — C1 (32.703 Hz) up to B6 (1975.533 Hz).
#[rustfmt::skip]
const BAND_FREQUENCIES: [f64; BANDS] = [
    32.7031956625748, 34.647828872109, 36.708095989676, 38.8908729652601,
    41.2034446141088, 43.6535289291255, 46.2493028389543, 48.9994294977187,
    51.9130871974932, 55.0, 58.2704701897613, 61.7354126570155,
    65.4063913251497, 69.2956577442181, 73.4161919793519, 77.7817459305203,
    82.4068892282175, 87.307057858251, 92.4986056779087, 97.9988589954374,
    103.826174394986, 110.0, 116.540940379523, 123.470825314031,
    130.812782650299, 138.591315488436, 146.832383958704, 155.563491861041,
    164.813778456435, 174.614115716502, 184.997211355817, 195.997717990875,
    207.652348789973, 220.0, 233.081880759045, 246.941650628062,
    261.625565300599, 277.182630976872, 293.664767917408, 311.126983722081,
    329.62755691287, 349.228231433004, 369.994422711635, 391.99543598175,
    415.304697579946, 440.000000000001, 466.163761518091, 493.883301256125,
    523.251130601198, 554.365261953745, 587.329535834816, 622.253967444163,
    659.255113825741, 698.456462866009, 739.98884542327, 783.9908719635,
    830.609395159892, 880.000000000002, 932.327523036182, 987.76660251225,
    1046.5022612024, 1108.73052390749, 1174.65907166963, 1244.50793488833,
    1318.51022765148, 1396.91292573202, 1479.97769084654, 1567.981743927,
    1661.21879031978, 1760.0, 1864.65504607236, 1975.5332050245,
];

/// `MAJOR_PROFILE` / `MINOR_PROFILE` — weights per semitone above the tonic.
#[rustfmt::skip]
const MAJOR_TONE_PROFILE: [f64; SEMITONES] = [
    7.23900502618145225142, 3.50351166725158691406, 3.58445177536649417505,
    2.84511816478676315967, 5.81898892118549859731, 4.55865057415321039969,
    2.44778850545506543313, 6.99473192146829525484, 3.39106613673504853068,
    4.55614256655143456953, 4.07392666663523606019, 4.45932757378886890365,
];
#[rustfmt::skip]
const MINOR_TONE_PROFILE: [f64; SEMITONES] = [
    7.00255045060284420089, 3.14360279015996679775, 4.35904319714962529275,
    5.40418120718934069657, 3.67234420879306133756, 4.08971184917797891956,
    3.90791435991553992579, 6.19960288562316463867, 3.63424625625277419871,
    2.87241191079875557435, 5.35467999794542670600, 3.83242038595048351013,
];
/// `OCTAVE_WEIGHTS` — how much each octave contributes.
const OCTAVE_WEIGHTS: [f64; OCTAVES] = [
    0.39997267549999998559,
    0.55634425248300645173,
    0.52496636345143543600,
    0.60847548384277727607,
    0.59898115679999996974,
    0.49072435317960994006,
];

/// `key_t` order — flats rather than sharps, with `silence` last.
pub const KEY_NAMES: [&str; 25] = [
    "A major", "A minor", "Bb major", "Bb minor", "B major", "B minor",
    "C major", "C minor", "Db major", "Db minor", "D major", "D minor",
    "Eb major", "Eb minor", "E major", "E minor", "F major", "F minor",
    "Gb major", "Gb minor", "G major", "G minor", "Ab major", "Ab minor",
    "silence",
];

/// `LowPassFilterPrivate` — a windowed-sinc FIR built in the frequency domain.
///
/// A brick wall is written into the first half of a 2048-point spectrum and the
/// inverse transform gives the (circular) impulse response. Only the 161 taps
/// around index 0 are kept, Hamming-windowed and normalised.
///
/// The spectrum is real and symmetric, so the inverse transform is a plain
/// cosine sum — no FFT needed.
fn lowpass_coefficients(frame_rate: f64, corner_hz: f64) -> (Vec<f64>, f64) {
    let delay = LPF_ORDER / 2;
    let impulse_length = LPF_ORDER + 1;
    let cutoff = corner_hz / frame_rate;
    let tau = 0.5 / cutoff;
    let last_bin = (0..=LPF_FFT_SIZE / 2)
        .filter(|j| *j as f64 / LPF_FFT_SIZE as f64 <= cutoff)
        .count()
        - 1;

    let impulse = |n: usize| -> f64 {
        let mut acc = tau; // the DC bin, not doubled
        for j in 1..=last_bin {
            acc += 2.0 * tau * (2.0 * PI * (j * n) as f64 / LPF_FFT_SIZE as f64).cos();
        }
        acc / LPF_FFT_SIZE as f64
    };

    let mut coefficients = Vec::with_capacity(impulse_length);
    let mut gain = 0.0;
    for i in 0..impulse_length {
        let index = (LPF_FFT_SIZE - delay + i) % LPF_FFT_SIZE;
        let hamming = 0.54 - 0.46 * (2.0 * PI * i as f64 / (impulse_length - 1) as f64).cos();
        let coefficient = impulse(index) * hamming;
        coefficients.push(coefficient);
        gain += coefficient;
    }
    (coefficients, gain)
}

/// `ChromaTransform` — a constant-Q filterbank expressed over FFT bins.
fn band_kernels(frame_rate: f64) -> (Vec<usize>, Vec<Vec<f64>>) {
    let q = DIRECT_SK_STRETCH * (2f64.powf(1.0 / SEMITONES as f64) - 1.0);
    let mut offsets = Vec::with_capacity(BANDS);
    let mut kernels = Vec::with_capacity(BANDS);
    for band in 0..BANDS {
        let centre = BAND_FREQUENCIES[band] * FFT_FRAME_SIZE as f64 / frame_rate;
        let width = centre * q;
        let begin = centre - width / 2.0;
        let offset = begin.ceil() as usize;
        offsets.push(offset);
        let last = (begin + width).floor() as usize;
        let mut weights: Vec<f64> = (offset..=last)
            .map(|bin| 1.0 - (2.0 * PI * (bin as f64 - begin) / width).cos())
            .collect();
        let total: f64 = weights.iter().sum();
        // Normalised by the sum, then scaled by the band frequency.
        for w in weights.iter_mut() {
            *w = *w / total * BAND_FREQUENCIES[band];
        }
        kernels.push(weights);
    }
    (offsets, kernels)
}

/// `SpectrumAnalyser::chromagramOfWholeFrames` — one 72-band vector per hop.
fn chromagram(samples: &[f64], frame_rate: f64) -> Vec<Vec<f64>> {
    let count = samples.len();
    if count < FFT_FRAME_SIZE {
        return Vec::new();
    }
    let hops = 1 + (count - FFT_FRAME_SIZE) / HOP_SIZE;
    let (offsets, kernels) = band_kernels(frame_rate);

    let mut window = vec![0.0f64; FFT_FRAME_SIZE];
    for (n, w) in window.iter_mut().enumerate() {
        let t = 2.0 * PI * n as f64 / (FFT_FRAME_SIZE - 1) as f64;
        *w = 0.42 - 0.5 * t.cos() + 0.08 * (2.0 * t).cos();
    }

    let mut planner = FftPlanner::<f64>::new();
    let plan = planner.plan_fft_forward(FFT_FRAME_SIZE);
    let mut buffer = vec![Complex::new(0.0, 0.0); FFT_FRAME_SIZE];
    let mut out = Vec::with_capacity(hops);

    for hop in 0..hops {
        let start = hop * HOP_SIZE;
        for (n, slot) in buffer.iter_mut().enumerate() {
            *slot = Complex::new(samples[start + n] * window[n], 0.0);
        }
        plan.process(&mut buffer);
        let magnitude: Vec<f64> = buffer[..FFT_FRAME_SIZE / 2 + 1]
            .iter()
            .map(|c| c.re.hypot(c.im))
            .collect();
        out.push(
            (0..BANDS)
                .map(|band| {
                    let offset = offsets[band];
                    kernels[band]
                        .iter()
                        .zip(&magnitude[offset..])
                        .map(|(k, m)| k * m)
                        .sum()
                })
                .collect(),
        );
    }
    out
}

/// `toneProfileMajor`/`Minor` — the 12-tone profile replicated per octave.
fn tone_profile(profile: &[f64; SEMITONES]) -> Vec<f64> {
    let mut out = Vec::with_capacity(BANDS);
    for octave in 0..OCTAVES {
        for s in 0..SEMITONES {
            out.push(OCTAVE_WEIGHTS[octave] * profile[s]);
        }
    }
    out
}

/// `ToneProfile::cosineSimilarity` — chroma against a rotated tone profile.
///
/// The profile's tonic sits three semitones above index 0 (the A-to-C shift in
/// the constructor) and `offset` walks it left, so chroma band `o*12 + j` pairs
/// with profile element `(3 - offset + j) mod 12`.
fn cosine_similarity(chroma: &[f64], tone: &[f64], offset: usize) -> f64 {
    let base = (3 + SEMITONES - offset % SEMITONES) % SEMITONES;
    let mut intersection = 0.0;
    let mut profile_norm = 0.0;
    let mut input_norm = 0.0;
    for octave in 0..OCTAVES {
        for j in 0..SEMITONES {
            let p = tone[octave * SEMITONES + (base + j) % SEMITONES];
            let x = chroma[octave * SEMITONES + j];
            intersection += x * p;
            profile_norm += p * p;
            input_norm += x * x;
        }
    }
    if profile_norm > 0.0 && input_norm > 0.0 {
        intersection / (profile_norm.sqrt() * input_norm.sqrt())
    } else {
        0.0
    }
}

/// `KeyClassifier::classify` — best cosine over 24 keys, defaulting to silence.
fn classify(chroma: &[f64]) -> &'static str {
    let major = tone_profile(&MAJOR_TONE_PROFILE);
    let minor = tone_profile(&MINOR_TONE_PROFILE);
    let mut scores = [0.0f64; 24];
    for i in 0..SEMITONES {
        scores[i * 2] = cosine_similarity(chroma, &major, i);
        scores[i * 2 + 1] = cosine_similarity(chroma, &minor, i);
    }

    // The silence profile is all zeroes, so its similarity is 0 and every real
    // key has to beat it outright.
    let mut best_score = 0.0;
    let mut best = 24;
    for (i, &score) in scores.iter().enumerate() {
        if score > best_score {
            best_score = score;
            best = i;
        }
    }
    KEY_NAMES[best]
}

/// `KeyFinder::keyOfAudio` — the whole libKeyFinder chain.
pub fn detect_key(samples: &[f32], sample_rate: u32) -> String {
    let rate = sample_rate as f64;
    let last_frequency = BAND_FREQUENCIES[BANDS - 1];
    let lpf_cutoff = last_frequency * 1.012;
    let downsample_cutoff = last_frequency * 1.10;
    let factor = ((rate / 2.0 / downsample_cutoff).floor() as usize).max(1);

    // `preprocess` slices the tail remainder off before filtering, and the
    // `finalChromagram` flush that would pick it up analyses a local that is
    // then discarded — so those samples are simply dropped.
    let usable = samples.len() - samples.len() % factor;
    let audio: Vec<f64> = samples[..usable].iter().map(|&s| s as f64).collect();

    let (coefficients, gain) = lowpass_coefficients(rate, lpf_cutoff);

    // The decimating FIR: tap `j` reads `out * factor - delay + j`, and only
    // every `factor`-th output is ever written. Computing just those costs
    // `kept * 161` rather than a full convolution.
    let delay = LPF_ORDER / 2;
    let kept = usable.div_ceil(factor);
    let mut decimated = Vec::with_capacity(kept);
    for q in 0..kept {
        let base = q * factor;
        let mut sum = 0.0f64;
        for (j, &coefficient) in coefficients.iter().enumerate() {
            let index = base + j;
            let x = if index >= delay {
                let at = index - delay;
                if at < audio.len() { audio[at] } else { 0.0 }
            } else {
                0.0
            };
            sum += coefficient * x;
        }
        decimated.push(sum / gain);
    }
    let rate = rate / factor as f64;

    // `progressiveChromagram` then `finalChromagram`: analyse what we have,
    // zero-pad the tail to a whole number of hops, and analyse that too.
    let first = chromagram(&decimated, rate);
    let mut tail: Vec<f64> = decimated[HOP_SIZE * first.len()..].to_vec();

    let padded_hops = tail.len().div_ceil(HOP_SIZE);
    let final_length = FFT_FRAME_SIZE + padded_hops.saturating_sub(1) * HOP_SIZE;
    if final_length > tail.len() {
        tail.resize(final_length, 0.0);
    }
    let second = chromagram(&tail, rate);

    let total_hops = first.len() + second.len();
    if total_hops == 0 {
        return KEY_NAMES[24].to_string();
    }
    let mut collapsed = vec![0.0f64; BANDS];
    for hop in first.iter().chain(second.iter()) {
        for (band, &value) in hop.iter().enumerate() {
            collapsed[band] += value / total_hops as f64;
        }
    }
    classify(&collapsed).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn midi_frequency(midi: f64) -> f64 {
        440.0 * 2f64.powf((midi - 69.0) / 12.0)
    }

    /// A sustained harmonic tone, so the chromagram has something to chew on.
    fn tone(midi: f64, seconds: f64, sample_rate: u32) -> Vec<f32> {
        let n = (seconds * sample_rate as f64) as usize;
        (0..n)
            .map(|i| {
                let t = i as f64 / sample_rate as f64;
                let f = midi_frequency(midi);
                let mut v = 0.0;
                for h in 1..12 {
                    if f * h as f64 > 5000.0 {
                        break;
                    }
                    v += (1.0 / h as f64) * (2.0 * PI * f * h as f64 * t).sin();
                }
                (v * 0.4) as f32
            })
            .collect()
    }

    fn progression(root: i32, minor: bool, sample_rate: u32) -> Vec<f32> {
        let scale: [i32; 7] = if minor {
            [0, 2, 3, 5, 7, 8, 10]
        } else {
            [0, 2, 4, 5, 7, 9, 11]
        };
        let mut out = Vec::new();
        for _ in 0..6 {
            for degree in [0usize, 3, 4, 0] {
                for interval in [0usize, 2, 4] {
                    let step = (degree + interval) % 7;
                    let midi = 48.0 + root as f64 + scale[step] as f64
                        + 12.0 * ((degree + interval) / 7) as f64;
                    out.extend(tone(midi, 0.6, sample_rate));
                }
            }
        }
        out
    }

    #[test]
    fn lowpass_has_unit_dc_gain_and_a_2khz_corner() {
        let (coefficients, gain) = lowpass_coefficients(44100.0, 1975.5332050245 * 1.012);
        assert_eq!(coefficients.len(), 161);
        // Normalised by `gain`, so the DC gain of the applied filter is 1.
        let normalised: f64 = coefficients.iter().sum::<f64>() / gain;
        assert!((normalised - 1.0).abs() < 1e-12, "dc gain {normalised}");
        // The taps are symmetric about the centre.
        for i in 0..LPF_ORDER {
            assert!((coefficients[i] - coefficients[LPF_ORDER - i]).abs() < 1e-12);
        }
    }

    #[test]
    fn decimation_factor_matches_the_library() {
        // floor(44100 / 2 / (1975.533 * 1.10)) == 10, giving a 4410 Hz analysis
        // rate — the value the reference build reports.
        let rate = 44100.0;
        let factor = ((rate / 2.0 / (BAND_FREQUENCIES[BANDS - 1] * 1.10)).floor()) as usize;
        assert_eq!(factor, 10);
    }

    #[test]
    fn a_major_and_minor_progression_are_identified() {
        for (root, minor, expected) in
            [(0i32, false, "C major"), (7, false, "G major"), (9, true, "A minor")]
        {
            let y = progression(root, minor, 44100);
            assert_eq!(detect_key(&y, 44100), expected, "root {root} minor {minor}");
        }
    }

    #[test]
    fn silence_is_reported_as_silence() {
        assert_eq!(detect_key(&vec![0f32; 44100 * 3], 44100), "silence");
    }
}
