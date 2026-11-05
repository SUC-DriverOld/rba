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
use symphonia::core::formats::{probe::Hint, FormatOptions, Track, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

use crate::config::Config;
use crate::error::{Error, Result};

/// Decode `path` to mono `f32` samples at `cfg.sample_rate`, folded with a
/// plain channel sum. Backend choice: `RBA_USE_FFMPEG` selects ffmpeg,
/// anything else (the default) selects the built-in Symphonia decoder.
pub fn decode(path: &Path, cfg: &Config) -> Result<Vec<f32>> {
    crate::config::validate_sample_rate(cfg.sample_rate).map_err(Error::Decoder)?;
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
    validate_source_rate(src_rate)?;

    // Raw MPEG/ADTS readers may estimate num_frames from bitrate. Symphonia's
    // gapless trimming would treat that estimate as an exact endpoint.
    let estimated = duration_is_estimated(format.format_info().short_name, &track);
    let expected_frames = declared_frame_count(format.format_info().short_name, &track);

    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(
            &params,
            &AudioDecoderOptions::default()
                .gapless(!estimated)
                .verify(true),
        )
        .map_err(|e| format!("{}: no decoder for this codec ({e})", path.display()))?;

    // Fold each decoded packet to mono as it arrives.
    let mut mono: Vec<f32> = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break, // end of stream
            Err(SymError::ResetRequired) => {
                return Err(format!("{}: audio stream changed during decoding; incomplete analysis is not supported", path.display()));
            }
            Err(e) => {
                return Err(format!("{}: demux failed ({e})", path.display()));
            }
        };
        if packet.track_id != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(decoded) => fold_mono(&decoded, &mut mono),
            Err(e) => {
                return Err(format!("{}: decode failed ({e})", path.display()));
            }
        }
    }

    if mono.is_empty() {
        return Err(format!("{}: no decodable audio", path.display()));
    }
    if decoder.finalize().verify_ok == Some(false) {
        return Err(format!(
            "{}: decoded audio failed checksum verification",
            path.display()
        ));
    }
    check_frame_count(mono.len(), expected_frames, 0)?;

    if src_rate == target_rate {
        Ok(mono)
    } else {
        resample(&mono, src_rate, target_rate)
    }
}

fn duration_is_estimated(format: &str, track: &Track) -> bool {
    matches!(format, "mp1" | "mp2" | "mp3" | "aac")
        && track.delay.is_none()
        && track.padding.is_none()
}

fn declared_frame_count(format: &str, track: &Track) -> Option<u64> {
    // MP4 edit-list spans and Ogg granules include codec priming or an unknown
    // origin (including single-page streams). They are not exact PCM counts.
    match format {
        "wave" | "wav" | "aiff" | "flac" => track.num_frames,
        "mp1" | "mp2" | "mp3" if !duration_is_estimated(format, track) => track.num_frames,
        _ => None,
    }
}

fn validate_source_rate(rate: u32) -> std::result::Result<(), String> {
    if !(1_000..=768_000).contains(&rate) {
        return Err(format!(
            "source sample rate {rate} Hz must be between 1000 and 768000 Hz"
        ));
    }
    Ok(())
}

fn check_frame_count(
    actual: usize,
    expected: Option<u64>,
    tolerance: u64,
) -> std::result::Result<(), String> {
    if let Some(expected) = expected {
        if (actual as u64).abs_diff(expected) > tolerance {
            return Err(format!(
                "incomplete audio: decoded {actual} frames, expected {expected}"
            ));
        }
    }
    Ok(())
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

fn resampled_frame_count(
    frames: u64,
    in_rate: u32,
    out_rate: u32,
) -> std::result::Result<usize, String> {
    usize::try_from(
        (u128::from(frames) * u128::from(out_rate) + u128::from(in_rate) / 2) / u128::from(in_rate),
    )
    .map_err(|_| "resampled audio is too large".to_string())
}

/// Resample mono audio, draining the filter tail and removing its initial delay.
fn resample(input: &[f32], in_rate: u32, out_rate: u32) -> std::result::Result<Vec<f32>, String> {
    use rubato::{FftFixedIn, Resampler};

    const CHUNK_IN: usize = 4096;
    validate_source_rate(in_rate)?;
    crate::config::validate_sample_rate(out_rate)?;
    if input.is_empty() {
        return Ok(Vec::new());
    }
    let mut res = FftFixedIn::<f32>::new(in_rate as usize, out_rate as usize, CHUNK_IN, 2, 1)
        .map_err(|e| format!("resampler init {in_rate}→{out_rate} Hz failed: {e}"))?;
    let expected = resampled_frame_count(input.len() as u64, in_rate, out_rate)?;
    let delay = res.output_delay();
    let needed = expected
        .checked_add(delay)
        .ok_or("resampled audio is too large")?;
    let mut out: Vec<f32> = Vec::new();
    out.try_reserve_exact(needed)
        .map_err(|e| format!("cannot allocate resampled audio: {e}"))?;
    let mut buf_out = vec![vec![0f32; res.output_frames_max()]; 1];

    let mut pos = 0usize;
    while pos < input.len() {
        let need = res.input_frames_next();
        let take = (input.len() - pos).min(need);
        let chunk = &input[pos..pos + take];
        pos += take;
        let (_, written) = if take < need {
            res.process_partial_into_buffer(Some(&[chunk]), &mut buf_out, None)
        } else {
            res.process_into_buffer(&[chunk], &mut buf_out, None)
        }
        .map_err(|e| format!("resampling failed: {e}"))?;
        out.extend_from_slice(&buf_out[0][..written.min(needed.saturating_sub(out.len()))]);
    }

    // Drain the delayed tail, including odd ratios where several input chunks
    // are buffered before one FFT block is available. One final FFT block is
    // enough; its input size is at most in_rate + CHUNK_IN for these bounds.
    let max_flush = (in_rate as usize).div_ceil(res.input_frames_next()) + 4;
    for _ in 0..max_flush {
        if out.len() >= needed {
            break;
        }
        let (_, written) = res
            .process_partial_into_buffer(None::<&[&[f32]]>, &mut buf_out, None)
            .map_err(|e| format!("flushing resampler failed: {e}"))?;
        out.extend_from_slice(&buf_out[0][..written.min(needed.saturating_sub(out.len()))]);
    }
    if out.len() < needed {
        return Err("resampler did not produce the complete audio tail".to_string());
    }
    out.copy_within(delay..needed, 0);
    out.truncate(expected);
    Ok(out)
}

struct AudioProbe {
    channels: usize,
    sample_rate: Option<u32>,
    expected_frames: Option<u64>,
    audio_index: usize,
}

/// Metadata from the same audio track used by the native backend. Formats not
/// supported by Symphonia still use FFmpeg's own stream selection.
fn probe_audio(path: &Path) -> Option<AudioProbe> {
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
    let track = format.default_track(TrackType::Audio)?;
    let params = track.codec_params.as_ref()?.audio()?;
    let channels = params.channels.as_ref()?.count();
    if channels == 0 {
        return None;
    }
    let audio_index = format
        .tracks()
        .iter()
        .filter(|item| item.codec_params.as_ref().and_then(|p| p.audio()).is_some())
        .position(|item| item.id == track.id)?;
    Some(AudioProbe {
        channels,
        sample_rate: params.sample_rate,
        audio_index,
        expected_frames: declared_frame_count(format.format_info().short_name, track),
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// ffmpeg backend (opt-in via RBA_USE_FFMPEG)
// ─────────────────────────────────────────────────────────────────────────────

/// Decode with the external ffmpeg binary, as the CLI did before Symphonia:
/// ask for the source's own channel count (probed here, no ffprobe needed),
/// resampled to the analysis rate, as headerless f32le.
fn decode_ffmpeg(path: &Path, cfg: &Config) -> Result<Vec<f32>> {
    let probe = probe_audio(path);
    let channels = probe.as_ref().map_or(2, |info| info.channels);
    if let Some(rate) = probe.as_ref().and_then(|info| info.sample_rate) {
        validate_source_rate(rate).map_err(Error::Decoder)?;
    }
    // An absolute local path prevents a relative filename such as "-" or
    // "pipe:0" from being interpreted as an FFmpeg input protocol.
    let input = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut command = Command::new(cfg.ffmpeg_path());
    command
        .args([
            "-v",
            "error",
            "-xerror",
            "-err_detect",
            "explode",
            "-nostdin",
            "-i",
        ])
        .arg(input);
    if let Some(info) = &probe {
        command.arg("-map").arg(format!("0:a:{}", info.audio_index));
    }
    let out = command
        .arg("-vn")
        .arg("-ac")
        .arg(channels.to_string())
        .arg("-ar")
        .arg(cfg.sample_rate.to_string())
        .args(["-f", "f32le", "-acodec", "pcm_f32le", "-"])
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

    if out.stdout.len() % (4 * channels) != 0 {
        return Err(Error::Decoder(
            "ffmpeg returned an incomplete PCM frame".to_string(),
        ));
    }
    let interleaved: Vec<f32> = out
        .stdout
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let samples: Vec<f32> = if channels > 1 {
        // Plain channel sum (L+R), matching the original's PCM reader.
        // Frames have already been checked for completeness.
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

    if let Some(info) = probe {
        if let (Some(rate), Some(frames)) = (info.sample_rate, info.expected_frames) {
            let expected =
                resampled_frame_count(frames, rate, cfg.sample_rate).map_err(Error::Decoder)?;
            // FFmpeg can round a resampled duration up instead of to nearest.
            check_frame_count(samples.len(), Some(expected as u64), 1).map_err(Error::Decoder)?;
        }
    }
    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_rate_limits_reject_extremes_without_allocating_fft_buffers() {
        for rate in [0, 1, 999, 768001, u32::MAX] {
            assert!(validate_source_rate(rate).is_err());
        }
        for rate in [1000, 8000, 44100, 192000, 768000] {
            assert!(validate_source_rate(rate).is_ok());
        }
    }

    #[test]
    fn raw_duration_estimates_do_not_override_explicit_gapless_metadata() {
        let mut track = Track::new(0);
        track.num_frames = Some(100);
        assert!(duration_is_estimated("mp3", &track));
        assert!(duration_is_estimated("aac", &track));
        assert!(!duration_is_estimated("wav", &track));
        assert_eq!(declared_frame_count("mp3", &track), None);
        assert_eq!(declared_frame_count("wav", &track), Some(100));
        assert_eq!(declared_frame_count("mp4", &track), None);
        track.delay = Some(1105);
        track.padding = Some(0);
        assert!(!duration_is_estimated("mp3", &track));
        assert_eq!(declared_frame_count("mp3", &track), Some(100));
    }

    #[test]
    fn wave_reader_supplies_declared_pcm_frames() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pcm.wav");
        assert_eq!(probe_audio(&path).unwrap().expected_frames, Some(512));
    }

    #[test]
    fn resampling_empty_and_short_buffers_is_bounded_at_supported_rate_edges() {
        for (input_rate, output_rate) in [
            (1000, 192000),
            (768000, 8000),
            (44101, 44100),
            (767999, 8000),
        ] {
            assert!(resample(&[], input_rate, output_rate).unwrap().is_empty());
            let expected = resampled_frame_count(100, input_rate, output_rate).unwrap();
            assert_eq!(
                resample(&[0.5; 100], input_rate, output_rate)
                    .unwrap()
                    .len(),
                expected
            );
        }
    }

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
    fn resample_keeps_audio_at_the_end_without_leading_filter_delay() {
        let mut input = vec![0.0f32; 4800];
        input[4700..].fill(1.0);
        let out = resample(&input, 48000, 44100).expect("resample");
        assert_eq!(out.len(), 4410);
        assert!(out[..100].iter().all(|sample| sample.abs() < 1e-4));
        assert!(out[4320..].iter().map(|sample| sample.abs()).sum::<f32>() > 20.0);
    }

    #[test]
    fn resampler_handles_odd_ratios() {
        // 44101 → 44100 exercises the gcd path with a large factor
        // (gcd = 1). 5000 * 44100/44101 = 4999.89 → rounds to 5000.
        let out = resample(&vec![0.25f32; 5000], 44101, 44100).expect("resample");
        assert_eq!(out.len(), 5000);
    }
}
