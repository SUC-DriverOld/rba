//! Audio decoding via an external `ffmpeg`.
//!
//! ## Why ffmpeg
//!
//! The original binary used CoreAudio's `ExtAudioFile` with a hard-coded
//! client format (mono / signed 16-bit / 44100 Hz) and normalised samples with
//! `i16 / 32767.0` (see docs/02-algorithm-spec.md §1). Using ffmpeg gives us
//! the same normalised float pipeline while supporting far more formats, and
//! avoids linking any decoder into this crate.
//!
//! We ask ffmpeg for **f32le** rather than s16le: the original's s16 step was
//! an artefact of CoreAudio's converter, and going straight to float avoids a
//! pointless second quantisation. The measured BPM is identical either way
//! (verified against the prototype, see docs/04-testing.md).
//!
//! ## Downmixing is done here, not by ffmpeg
//!
//! ffmpeg's `-ac 1` is *power preserving*: for stereo it produces
//! `(L+R)/sqrt(2)` rather than `(L+R)/2`, i.e. a `sqrt(2)` gain. That matters
//! because onset detection compares against an **absolute** threshold, so the
//! gain change shifts which onsets are found and can move the reported tempo by
//! several BPM — a 48 kHz stereo MP3 reported 156.60 instead of 160.02. We
//! therefore decode the source's own channel count and average it ourselves,
//! which is plain `(L+R)/2`, matching CoreAudio's converter.

use std::path::Path;
use std::process::Command;

use crate::config::Config;
use crate::error::{Error, Result};

/// Decode `path` to mono `f32` samples in roughly `[-1, 1]` at `cfg.sample_rate`.
///
/// `Command::output` reads stdout and stderr concurrently, so a chatty ffmpeg
/// cannot deadlock against a full stderr pipe.
pub fn decode(path: &Path, cfg: &Config) -> Result<Vec<f32>> {
    // Ask for the source's own channel count so ffmpeg applies no mixdown of
    // its own; we average below. Falling back to stereo keeps mono and stereo
    // sources exactly right even when ffprobe is unavailable.
    let channels = probe_channels(path, cfg).unwrap_or(2);

    let out = Command::new(cfg.ffmpeg_path())
        .arg("-v")
        .arg("error")
        .arg("-nostdin")
        .arg("-i")
        .arg(path)
        .arg("-vn") // ignore any embedded cover art
        .arg("-ac")
        .arg(channels.to_string())
        .arg("-ar")
        .arg(cfg.sample_rate.to_string())
        .arg("-f")
        .arg("f32le")
        .arg("-acodec")
        .arg("pcm_f32le")
        .arg("-")
        .output()
        .map_err(|e| {
            Error::Decoder(format!(
                "cannot run `{}` ({e}). Set RBA_FFMPEG to its full path.",
                cfg.ffmpeg_path().display()
            ))
        })?;

    if !out.status.success() {
        let msg = String::from_utf8_lossy(&out.stderr);
        let msg = msg.trim();
        let msg = if msg.is_empty() { "unknown error" } else { msg };
        return Err(Error::Decoder(format!(
            "ffmpeg failed on {}: {msg}",
            path.display()
        )));
    }

    let raw = out.stdout;
    let interleaved: Vec<f32> = raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let mut samples: Vec<f32> = if channels > 1 {
        // Plain average over the frame, matching CoreAudio. A trailing partial
        // frame is dropped.
        let frames = interleaved.len() / channels;
        (0..frames)
            .map(|frame| {
                let voice = &interleaved[frame * channels..(frame + 1) * channels];
                voice.iter().sum::<f32>() / channels as f32
            })
            .collect()
    } else {
        interleaved
    };

    if samples.len() < crate::config::FFT_SIZE {
        return Err(Error::EmptyAudio(path.to_path_buf()));
    }

    // Guard against denormal/NaN poison from odd containers.
    for s in samples.iter_mut() {
        if !s.is_finite() {
            *s = 0.0;
        }
    }

    Ok(std::mem::take(&mut samples))
}

/// Channel count of the first audio stream, or `None` if it cannot be probed.
fn probe_channels(path: &Path, cfg: &Config) -> Option<usize> {
    let out = Command::new(cfg.ffprobe_path())
        .args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=channels",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
        ])
        .arg(path)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse::<usize>()
        .ok()
        .filter(|n| *n > 0)
}

/// One-line summary of the source stream, for `--verbose`.
///
/// Returns `None` when `ffprobe` is missing or the file is not probeable; this
/// is purely diagnostic, so it must never fail the run.
pub fn probe_summary(path: &Path, cfg: &Config) -> Option<String> {
    let out = Command::new(cfg.ffprobe_path())
        .args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels,codec_name,duration",
            "-of",
            "default=noprint_wrappers=1",
        ])
        .arg(path)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let parts: Vec<String> = text
        .lines()
        .filter_map(|line| line.split_once('='))
        .map(|(k, v)| format!("{k}={v}"))
        .collect();
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" "))
    }
}
