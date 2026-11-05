//! End-to-end BPM analysis pipeline.
//!
//! Mirrors `CProfileBuilder::BuildProfile` (docs/02-algorithm-spec.md §0):
//! spectral thumbnail → peak-hold → onsets(÷4) → base BPM → half-tempo fold
//! (from the smoothed *note-detector* thumbnail) → short-clip prior →
//! `BuildBeatPointsAlt` chain stage, which produces the final BPM.

use std::path::Path;

use crate::audio;
use crate::beatchain::{self, ChainTrace};
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
    /// `f64` for display convenience; the chain stage computes in `f32` like
    /// the original. `None` when `--keyonly` skipped the BPM pipeline.
    pub bpm: Option<f64>,
    /// PCM samples after decode/resample.
    pub sample_count: i64,
    /// Analysis sample rate.
    pub sample_rate: u32,
    /// Duration in seconds.
    pub duration: f64,
    /// libKeyFinder key name, e.g. `"G major"`; `None` when `--bpmonly` was given.
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
    /// Onsets feeding the chain stage (divisor 8).
    pub chain_onset_count: usize,
    pub short_clip_applied: bool,
    /// Beat-chain stage diagnostics (spec §8).
    pub chain: ChainTrace,
    /// Key reported by the libKeyFinder port, for `--verbose` (spec §13).
    pub key_detected: String,
}

/// Analyse an already-decoded mono buffer.
///
/// `find_bpm` gates the tempo pipeline (stages ②–⑧), `find_key` the
/// libKeyFinder pass. Skipping a pass leaves its `Analysis` field `None`
/// and its trace values at their defaults.
pub fn analyze_samples(
    samples: &[f32],
    sample_rate: u32,
    find_bpm: bool,
    find_key: bool,
) -> Analysis {
    let total_samples = samples.len() as i64;

    // ── ⑩ key: the libKeyFinder chain (spec §13) ───────────────────────
    let key = find_key.then(|| keyfinder::detect_key(samples, sample_rate));
    let key_detected = key.clone().unwrap_or_else(|| "not requested".to_string());

    // ── ②–⑧ the tempo pipeline, in BuildProfile order ──────────────────
    let (bpm, trace) = if find_bpm {
        // ── ② spectral thumbnail (v30) ──────────────────────────────────
        let spec = thumbnail::spec_thumbnail(samples, sample_rate);

        // ── ③ peak-hold envelope (v33) ──────────────────────────────────
        let mut peak = spec.clone();
        thumbnail::peak_hold(&mut peak);

        // ── ④ onsets for the tempo histogram (divisor 4) ────────────────
        let ip: ImpactPoints = impact::build_impact_points(&peak, total_samples, 4);

        // ── ⑤ initial tempo: median of 9 windows × {÷4, ÷8} ────────────
        let base_bpm = detect::detect_base_bpm(&ip);

        // ── ⑥ half-tempo fold — onsets from the smoothed note thumbnail ─
        // The DLL smooths a copy of the *note-detector* thumbnail (filtered
        // waveform peaks), not the spectral one.
        let note = thumbnail::note_peak_thumbnail(samples, sample_rate);
        let ipc =
            impact::build_impact_points_classic(&thumbnail::smoothed(&note), total_samples, 4);
        let folded_bpm = detect::fold_up_bpm(&ipc, base_bpm);

        // ── ⑦ short-clip override — a prior, not a verdict: the chain
        // still runs below and can confirm (or, with content, refine) it ─
        let prior = detect::short_clip_bpm_update(total_samples, sample_rate, folded_bpm);
        let is_short = sample_rate != 0 && 30 * sample_rate as i64 > total_samples;

        // ── ⑧ final BPM: BuildBeatPointsAlt → ClusterFilterPoints ───────
        let onsets8 = impact::build_impact_points(&peak, total_samples, 8);
        let mut chain = ChainTrace::default();
        let bpm = beatchain::beat_chain_bpm(&spec, &onsets8, prior, sample_rate, &mut chain);

        (
            Some(bpm as f64),
            Trace {
                thumbnail_elements: spec.len(),
                peak_hold_decay_div: peak.ele_per_meas(164.0) >> 4,
                onset_count: ip.len(),
                base_bpm,
                classic_onset_count: ipc.len(),
                folded_bpm,
                chain_onset_count: onsets8.len(),
                short_clip_applied: is_short,
                chain,
                key_detected: key_detected.clone(),
            },
        )
    } else {
        (None, Trace {
            key_detected,
            ..Trace::default()
        })
    };

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
        trace,
    }
}

/// Decode `path` and analyse it.
pub fn analyze_file(path: &Path, cfg: &Config) -> Result<Analysis> {
    let samples = audio::decode(path, cfg)?;
    if samples.len() < FFT_SIZE {
        return Err(Error::EmptyAudio(path.to_path_buf()));
    }
    let mut a = analyze_samples(&samples, cfg.sample_rate, cfg.find_bpm, cfg.find_key);
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
        let a = analyze_samples(&samples, 44100, true, false);
        assert!(a.bpm.unwrap().is_finite());
    }

    #[test]
    fn empty_buffer_is_safe() {
        let a = analyze_samples(&[], 44100, true, false);
        assert_eq!(a.sample_count, 0);
        assert_eq!(a.trace.thumbnail_elements, 0);
    }
}
