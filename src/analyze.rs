//! End-to-end BPM analysis pipeline.
//!
//! Mirrors `CProfileBuilder::BuildProfile` (docs/02-algorithm-spec.md §0).

use std::path::Path;

use crate::audio;
use crate::config::{Config, FFT_SIZE};
use crate::detect;
use crate::error::{Error, Result};
use crate::impact::{self, ImpactPoints};
use crate::keyfinder;
use crate::thumbnail::{self, Thumbnail};

/// Result of analysing one input.
#[derive(Debug, Clone)]
pub struct Analysis {
    /// Display name (usually the file name).
    pub name: String,
    /// Detected tempo in BPM.
    ///
    /// `f64` rather than the original's `float` because the refinement stage
    /// (spec §11) works below the 0.1 grid the recovered algorithm searches on.
    pub bpm: f64,
    /// PCM samples after decode/resample.
    pub sample_count: i64,
    /// Analysis sample rate.
    pub sample_rate: u32,
    /// Duration in seconds.
    pub duration: f64,
    /// libKeyFinder key name, e.g. `"G major"`; `None` when `--nokey` was given.
    pub key: Option<String>,
    /// Intermediate values, for `--verbose` and tests.
    pub trace: Trace,
}

/// Intermediate pipeline values. Useful for diagnosing a surprising result.
#[derive(Debug, Clone, Default)]
pub struct Trace {
    pub thumbnail_elements: usize,
    pub peak_hold_decay_div: u32,
    pub onset_count: usize,
    /// Tempo from the 0.1-grid phase-histogram search (spec §5), in BPM.
    pub base_bpm: f32,
    pub classic_onset_count: usize,
    /// `base_bpm` after the half-tempo fix (spec §6).
    pub folded_bpm: f32,
    /// Period refinement result (spec §11); `None` when it did not run.
    pub refined_bpm: Option<f64>,
    /// How many onsets the refinement fit.
    pub refine_onsets: usize,
    pub short_clip_applied: bool,
    /// Key reported by the libKeyFinder port, for `--verbose` (spec §13).
    pub key_detected: String,
}

/// Analyse an already-decoded mono buffer.
pub fn analyze_samples(samples: &[f32], sample_rate: u32, find_key: bool) -> Analysis {
    let total_samples = samples.len() as i64;

    // ── ② spectral thumbnail ────────────────────────────────────────────
    let spec = thumbnail::spec_thumbnail(samples, sample_rate);

    // ── ③ peak-hold envelope ────────────────────────────────────────────
    let mut peak = spec.clone();
    thumbnail::peak_hold(&mut peak);

    // ── ④ onsets for the tempo histogram (divisor 8) ────────────────────
    let ip: ImpactPoints = impact::build_impact_points(&peak, total_samples, 8);

    // ── ⑤ initial tempo: median of 9 windows × {÷4, ÷8} ────────────────
    let base_bpm = detect::detect_base_bpm(&ip);

    // ── ⑥ half-tempo correction (needs the smoothed thumbnail) ─────────
    let smooth = thumbnail::smoothed(&spec);
    let ipc = impact::build_impact_points_classic(&smooth, total_samples, 4);
    let folded_bpm = detect::fold_up_bpm(&ipc, base_bpm);

    // ── ⑦ short-clip override, else ⑧ period refinement ────────────────
    //
    // A clip under 30 s is deliberately governed by its *duration* (spec §7),
    // so refining it against onsets would fight an intentional rule; only
    // content-derived results get refined.
    let is_short = sample_rate != 0 && 30 * sample_rate as i64 > total_samples;
    let (bpm, refined_bpm, refine_onsets) = if is_short {
        (
            detect::short_clip_bpm_update(total_samples, sample_rate, folded_bpm) as f64,
            None,
            0,
        )
    } else {
        let (refined, kept) = detect::refine_bpm(&ip, folded_bpm, sample_rate);
        (refined, Some(refined), kept)
    };

    // ── ⑩ key: the libKeyFinder chain (spec §13) ───────────────────────
    let key = find_key.then(|| keyfinder::detect_key(samples, sample_rate));
    let key_detected = key.clone().unwrap_or_else(|| "not requested".to_string());

    let duration = if sample_rate == 0 {
        0.0
    } else {
        total_samples as f64 / sample_rate as f64
    };

    Analysis {
        name: String::new(),
        bpm,
        sample_count: total_samples,
        sample_rate,
        duration,
        key,
        trace: Trace {
            thumbnail_elements: spec.len(),
            peak_hold_decay_div: peak.ele_per_meas(164.0) >> 4,
            onset_count: ip.len(),
            base_bpm,
            classic_onset_count: ipc.len(),
            folded_bpm,
            refined_bpm,
            refine_onsets,
            short_clip_applied: is_short,
            key_detected,
        },
    }
}

/// Decode `path` and analyse it.
pub fn analyze_file(path: &Path, cfg: &Config) -> Result<Analysis> {
    let samples = audio::decode(path, cfg)?;
    if samples.len() < FFT_SIZE {
        return Err(Error::EmptyAudio(path.to_path_buf()));
    }
    let mut a = analyze_samples(&samples, cfg.sample_rate, cfg.find_key);
    a.name = display_name(path);
    Ok(a)
}

/// File name for tables, falling back to the full path.
pub fn display_name(path: &Path) -> String {
    path.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string_lossy().into_owned())
}

/// Convenience for tests: analyse a thumbnail-producing buffer.
pub fn thumbnail_of(samples: &[f32], sample_rate: u32) -> Thumbnail {
    thumbnail::spec_thumbnail(samples, sample_rate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_does_not_panic() {
        let samples = vec![0f32; 44100 * 5];
        let a = analyze_samples(&samples, 44100, false);
        assert!(a.bpm.is_finite(), "bpm = {}", a.bpm);
    }

    #[test]
    fn empty_buffer_is_safe() {
        let a = analyze_samples(&[], 44100, false);
        assert_eq!(a.sample_count, 0);
        assert_eq!(a.trace.thumbnail_elements, 0);
    }
}
