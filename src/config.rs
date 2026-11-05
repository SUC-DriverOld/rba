//! Runtime configuration, derived from the parsed command line.
//!
//! Decoding is done by the built-in Symphonia engine by default. The
//! external-`ffmpeg` backend is opt-in through `RBA_USE_FFMPEG`, with the
//! binary's path taken from `RBA_FFMPEG` (alias `rba_ffmpeg`) — there is no
//! CLI flag for either: the decoder is an implementation detail, not an
//! analysis option.
//!
//! Every option can also come from the environment (see [`crate::cli::Cli`]);
//! this struct is the resolved, validated form the analysis layer consumes.

use std::path::{Path, PathBuf};

/// Analysis sample rate. The original binary hard-codes the ExtAudioFile
/// client format to 44100 Hz mono, so this is the faithful default.
pub const DEFAULT_SAMPLE_RATE: u32 = 44_100;

/// FFT / thumbnail frame size (spec §2).
pub const FFT_SIZE: usize = 256;

#[derive(Debug, Clone)]
pub struct Config {
    /// Sample rate the audio is resampled to before analysis.
    pub sample_rate: u32,
    /// Emit extra detail on stderr.
    pub verbose: bool,
    /// Run the BPM pipeline. On by default; `--keyonly` turns it off.
    pub find_bpm: bool,
    /// Run the key pass. On by default; `--bpmonly` turns it off.
    pub find_key: bool,
    /// Use the external ffmpeg backend (`RBA_USE_FFMPEG`) instead of the
    /// built-in Symphonia decoder.
    pub use_ffmpeg: bool,
    /// `ffmpeg` executable for the opt-in backend, from `RBA_FFMPEG`; a bare
    /// name is resolved through `PATH` at spawn time.
    pub ffmpeg: PathBuf,
}

impl Config {
    pub fn new(sample_rate: u32, verbose: bool, find_bpm: bool, find_key: bool) -> Self {
        Config {
            sample_rate: if sample_rate == 0 {
                DEFAULT_SAMPLE_RATE
            } else {
                sample_rate
            },
            verbose,
            find_bpm,
            find_key,
            use_ffmpeg: env_flag("RBA_USE_FFMPEG"),
            ffmpeg: std::env::var_os("RBA_FFMPEG")
                .filter(|s| !s.is_empty())
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("ffmpeg")),
        }
    }

    /// The decoder binary, for `Command::new` (ffmpeg backend only).
    pub fn ffmpeg_path(&self) -> &Path {
        &self.ffmpeg
    }
}

/// Truthiness of an environment "flag": set and not one of the usual
/// spellings of off (`0`, `false`, `no`, `off`), case-insensitive.
pub fn env_flag(name: &str) -> bool {
    match std::env::var_os(name) {
        Some(v) => {
            let s = v.to_string_lossy().trim().to_ascii_lowercase();
            !s.is_empty() && !matches!(s.as_str(), "0" | "false" | "no" | "off")
        }
        None => false,
    }
}

impl Default for Config {
    fn default() -> Self {
        Config::new(DEFAULT_SAMPLE_RATE, false, true, false)
    }
}
