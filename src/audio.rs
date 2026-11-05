//! Audio decoding.
//!
//! ## Two backends
//!
//! * **Symphonia** (default, built in): pure-Rust demux + decode for the
//!   common formats, resampled to the analysis rate with `rubato`.
//! * **ffmpeg** (opt-in via `RBA_USE_FFMPEG`): the original external-binary
//!   path, kept for formats Symphonia has no codec for (wma, ape, wv, …).
//!   The binary's path comes from `RBA_FFMPEG` (alias `rba_ffmpeg`).
//!
//! ## Why a plain channel sum
//!
//! The original analyzer's PCM reader folds multichannel sources with a plain
//! channel **sum** (`L+R`), and so do both backends. ffmpeg's `-ac 1` is
//! rejected because it is *power preserving*: for stereo it produces
//! `(L+R)/sqrt(2)`. That matters because the onset ladder's thresholds are a
//! discrete ÷1.2 sequence, so any gain change moves the crossing to a
//! different rung and can change which onsets are found — validated against
//! the Windows oracle on 海边城.mp3 (sum → 160.03 vs mean → 160.18, oracle
//! 160.04).
//!
//! ## Sample scaling
//!
//! Symphonia's integer→float conversion divides by 2^n (e.g. 32768 for i16);
//! the original normalised with `i16 / 32767`. The 0.003 % difference is
//! three orders of magnitude below the ladder rung spacing and cannot move
//! an onset across a threshold.

use std::fs::File;
use std::path::Path;
use std::process::Command;

use symphonia::core::audio::GenericAudioBufferRef;
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::{probe::Hint, FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

use crate::config::Config;
use crate::error::{Error, Result};

/// Decode `path` to mono `f32` samples at `cfg.sample_rate`, folded with a
/// plain channel sum. Backend choice: `RBA_USE_FFMPEG` selects ffmpeg,
/// anything else (the default) selects the built-in Symphonia decoder.
pub fn decode(path: &Path, cfg: &Config) -> Result<Vec<f32>> {
    let samples = if cfg.use_ffmpeg {
        decode_ffmpeg(path, cfg)?
    } else {
        decode_symphonia(path, cfg.sample_rate).map_err(|e| {
            Error::Decoder(format!(
                "{e}. If the format needs ffmpeg, set RBA_USE_FFMPEG=1 \
                 (and RBA_FFMPEG to its path if it is not on PATH)."
            ))
        })?
    };

    if samples.len() < crate::config::FFT_SIZE {
        return Err(Error::EmptyAudio(path.to_path_buf()));
    }

    // Guard against denormal/NaN poison from odd containers.
    let mut samples = samples;
    for s in samples.iter_mut() {
        if !s.is_finite() {
            *s = 0.0;
        }
    }

    Ok(samples)
}

// ─────────────────────────────────────────────────────────────────────────────
// Symphonia backend (default)
// ─────────────────────────────────────────────────────────────────────────────

/// Decode with Symphonia: probe → demux → decode → fold to mono (plain sum)
/// → resample to `target_rate` with `rubato`.
fn decode_symphonia(path: &Path, target_rate: u32) -> std::result::Result<Vec<f32>, String> {
    let src = File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(src), Default::default());

    // The extension helps the prober pick a format for headerless streams.
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|e| format!("symphonia cannot probe {}: {e}", path.display()))?;

    let track = format
        .default_track(TrackType::Audio)
        .ok_or_else(|| format!("{}: no audio track", path.display()))?
        .clone();
    let track_id = track.id;

    let params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .ok_or_else(|| format!("{}: no audio codec parameters", path.display()))?
        .clone();

    let src_rate = params
        .sample_rate
        .ok_or_else(|| format!("{}: unknown sample rate", path.display()))?;

    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(&params, &AudioDecoderOptions::default())
        .map_err(|e| format!("{}: no decoder for this codec ({e})", path.display()))?;

    // Fold each decoded packet to mono as it arrives.
    let mut mono: Vec<f32> = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break, // end of stream
            Err(SymError::ResetRequired) => break,
            Err(e) => {
                return Err(format!("{}: demux failed ({e})", path.display()));
            }
        };
        if packet.track_id != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(decoded) => fold_mono(&decoded, &mut mono),
            // A corrupt packet must not abort the whole file.
            Err(SymError::DecodeError(_)) => continue,
            Err(e) => {
                return Err(format!("{}: decode failed ({e})", path.display()));
            }
        }
    }

    if mono.is_empty() {
        return Err(format!("{}: no decodable audio", path.display()));
    }

    if src_rate == target_rate {
        Ok(mono)
    } else {
        resample(&mono, src_rate, target_rate)
    }
}

/// Append `decoded` to `out` as mono, folding channels with a plain sum.
///
/// `copy_to_vec_interleaved::<f32>` handles every sample format
/// (u8/i16/i24/i32/f32…) and converts to `f32`; resampling is linear, so
/// folding before it is equivalent to folding after.
fn fold_mono(decoded: &GenericAudioBufferRef, out: &mut Vec<f32>) {
    let frames = decoded.frames();
    let nch = decoded.num_planes();
    if frames == 0 || nch == 0 {
        return;
    }
    let mut interleaved: Vec<f32> = Vec::with_capacity(frames * nch);
    decoded.copy_to_vec_interleaved(&mut interleaved);
    match nch {
        1 => out.extend_from_slice(&interleaved[..frames]),
        _ => {
            for f in 0..frames {
                out.push(interleaved[f * nch..(f + 1) * nch].iter().sum());
            }
        }
    }
}

/// Resample mono `input` from `in_rate` to `out_rate`.
///
/// `rubato`'s FFT resampler takes fixed-size input chunks and returns a
/// varying number of frames (it has internal overlap buffers), so the loop
/// just feeds it everything and flushes the tail with
/// [`Resampler::process_partial_into_buffer`]. The result is trimmed or
/// zero-padded to the expected length so short files keep their duration.
fn resample(input: &[f32], in_rate: u32, out_rate: u32) -> std::result::Result<Vec<f32>, String> {
    use rubato::{FftFixedIn, Resampler};

    const CHUNK_IN: usize = 4096;
    let mut res = FftFixedIn::<f32>::new(
        in_rate as usize,
        out_rate as usize,
        CHUNK_IN,
        2, // sub-chunks (internal buffering granularity)
        1, // mono
    )
    .map_err(|e| format!("resampler init {in_rate}→{out_rate} Hz failed: {e}"))?;

    let expected = (input.len() as f64 * f64::from(out_rate) / f64::from(in_rate)).round() as usize;
    let mut out: Vec<f32> = Vec::with_capacity(expected);

    let mut pos = 0usize;
    while pos < input.len() {
        let need = res.input_frames_next();
        let take = (input.len() - pos).min(need);
        let chunk = &input[pos..pos + take];
        pos += take;

        let mut buf_out = vec![vec![0f32; res.output_frames_max()]; 1];
        let (in_used, out_written) = if take < need {
            // Tail: pad with zeros inside the resampler.
            res.process_partial_into_buffer(Some(&[chunk]), &mut buf_out, None)
        } else {
            res.process_into_buffer(&[chunk], &mut buf_out, None)
        }
        .map_err(|e| format!("resampling failed: {e}"))?;

        if let Some(extra) = in_used.checked_sub(take) {
            pos += extra; // the resampler may buffer input across calls
        }
        out.extend_from_slice(&buf_out[0][..out_written]);
        if take < need {
            break; // tail chunk consumed
        }
    }

    // Exact output length: the FFT resampler's chunking can overshoot or
    // undershoot by a few frames around the ideal ratio.
    out.resize(expected, 0.0);
    Ok(out)
}

/// Channel count of the first audio track, probing headers with Symphonia.
///
/// Used by the ffmpeg backend to request the source's own channel count so
/// ffmpeg applies no mixdown of its own. `None` when the file cannot be
/// probed (callers then fall back to assuming stereo, as before).
fn probe_channels(path: &Path) -> Option<usize> {
    let src = File::open(path).ok()?;
    let mss = MediaSourceStream::new(Box::new(src), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .ok()?;
    format
        .default_track(TrackType::Audio)
        .and_then(|t| t.codec_params.as_ref())
        .and_then(|p| p.audio())
        .and_then(|p| p.channels.clone())
        .map(|c| c.count())
        .filter(|n| *n > 0)
}

// ─────────────────────────────────────────────────────────────────────────────
// ffmpeg backend (opt-in via RBA_USE_FFMPEG)
// ─────────────────────────────────────────────────────────────────────────────

/// Decode with the external ffmpeg binary, as the CLI did before Symphonia:
/// ask for the source's own channel count (probed here, no ffprobe needed),
/// resampled to the analysis rate, as headerless f32le.
fn decode_ffmpeg(path: &Path, cfg: &Config) -> Result<Vec<f32>> {
    let channels = probe_channels(path).unwrap_or(2);

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

    let interleaved: Vec<f32> = out
        .stdout
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let samples: Vec<f32> = if channels > 1 {
        // Plain channel sum (L+R), matching the original's PCM reader. A
        // trailing partial frame is dropped.
        let frames = interleaved.len() / channels;
        (0..frames)
            .map(|frame| {
                let voice = &interleaved[frame * channels..(frame + 1) * channels];
                voice.iter().sum::<f32>()
            })
            .collect()
    } else {
        interleaved
    };

    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_48k_to_441k_keeps_length_and_content() {
        // 1 kHz sine at 48 kHz → 44.1 kHz: same duration, same frequency.
        let n = 48000usize;
        let input: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 48000.0).sin())
            .collect();
        let out = resample(&input, 48000, 44100).expect("resample");
        assert_eq!(out.len(), 44100, "expected length ratio 44100/48000");
        // Energy survives: RMS close to 1/sqrt(2).
        let rms = (out.iter().map(|s| s * s).sum::<f32>() / out.len() as f32).sqrt();
        assert!(
            (rms - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.1,
            "rms {rms}"
        );
    }

    #[test]
    fn resample_short_input_reaches_expected_length() {
        let out = resample(&vec![0.5f32; 100], 32000, 44100).expect("resample");
        assert_eq!(out.len(), (100.0f64 * 44100.0 / 32000.0).round() as usize);
    }

    #[test]
    fn resampler_handles_odd_ratios() {
        // 44101 → 44100 exercises the gcd path with a large factor
        // (gcd = 1). 5000 * 44100/44101 = 4999.89 → rounds to 5000.
        let out = resample(&vec![0.25f32; 5000], 44101, 44100).expect("resample");
        assert_eq!(out.len(), 5000);
    }
}
