//! `rba` — BPM analyzer.
//!
//! A Rust rewrite of the analysis core of the 2008 macOS application
//! *MixMeister BPM Analyzer*. The original source is lost; this crate is a
//! direct port of the behaviour recovered by static analysis of the binary.
//!
//! * Recovered algorithm: `docs/02-algorithm-spec.md`
//! * How it was recovered: `docs/01-reverse-engineering.md`
//! * Raw decompilation evidence: `re/`
//!
//! # Dependencies
//!
//! Standing on existing crates rather than hand-rolling: `clap` (CLI),
//! `indicatif` (progress), `rayon` (parallel files), `walkdir` (traversal),
//! `unicode-width` (table alignment), `rustfft` (FFT), `thiserror` (errors).
//! The directory report is drawn by hand — a header, a dashed rule, one line
//! per file. Decoding is delegated to the external `ffmpeg` binary.

pub mod analyze;
pub mod audio;
pub mod cli;
pub mod config;
pub mod detect;
pub mod error;
pub mod fft;
pub mod impact;
pub mod report;
pub mod thumbnail;

pub use analyze::{analyze_file, analyze_samples, Analysis};
pub use config::Config;
pub use error::{Error, Result};

/// The library version, mirrored by `rba --version`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
