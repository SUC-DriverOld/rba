//! Runtime configuration, derived from the parsed command line.
//!
//! The decoder is intentionally external: `rba` shells out to `ffmpeg` so that
//! every container/codec ffmpeg supports is supported here for free, without
//! linking a decoder into this crate.
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
    /// `ffmpeg` executable — a bare name is resolved through `PATH`.
    pub ffmpeg: PathBuf,
    /// `ffprobe` executable, used only by `--verbose` for source format info.
    pub ffprobe: PathBuf,
    /// Sample rate the audio is resampled to before analysis.
    pub sample_rate: u32,
    /// Emit extra detail on stderr.
    pub verbose: bool,
    /// Run the key pass. On by default; `--nokey` turns it off.
    pub find_key: bool,
}

impl Config {
    pub fn new(
        ffmpeg: Option<PathBuf>,
        ffprobe: Option<PathBuf>,
        sample_rate: u32,
        verbose: bool,
        find_key: bool,
    ) -> Self {
        Config {
            ffmpeg: ffmpeg.unwrap_or_else(|| PathBuf::from("ffmpeg")),
            ffprobe: ffprobe.unwrap_or_else(|| PathBuf::from("ffprobe")),
            sample_rate: if sample_rate == 0 {
                DEFAULT_SAMPLE_RATE
            } else {
                sample_rate
            },
            verbose,
            find_key,
        }
    }

    /// The decoder path, for `Command::new`.
    pub fn ffmpeg_path(&self) -> &Path {
        &self.ffmpeg
    }

    pub fn ffprobe_path(&self) -> &Path {
        &self.ffprobe
    }
}

impl Default for Config {
    fn default() -> Self {
        Config::new(None, None, DEFAULT_SAMPLE_RATE, false, false)
    }
}
