//! Error type shared across the crate.
//!
//! Usage/argument errors are *not* modelled here: `clap` owns those and exits
//! with its own message and status code.

use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The external decoder (ffmpeg/ffprobe) failed or could not be executed.
    #[error("audio decoder: {0}")]
    Decoder(String),

    /// Filesystem problem.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// The file decoded, but contained too little audio to analyse.
    #[error("no decodable audio in {}", .0.display())]
    EmptyAudio(PathBuf),
}

pub type Result<T> = std::result::Result<T, Error>;
