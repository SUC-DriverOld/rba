//! Command line interface, built on `clap`.
//!
//! ```text
//! rba <file>        → print that file's BPM
//! rba <directory>   → print a table of every audio file inside
//! ```
//!
//! Argument parsing, `--help`, `--version`, environment fallbacks and usage
//! errors are all handled by `clap`; this module only wires the parsed values
//! to the analysis layer.

use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use walkdir::WalkDir;

use crate::analyze;
use crate::config::{Config, DEFAULT_SAMPLE_RATE};
use crate::error::Result;
use crate::report::{self, Row};

/// Exit codes, documented so scripts can rely on them.
///
/// Usage errors never reach this module: `clap` reports them and exits with 2.
pub const EXIT_OK: u8 = 0;
pub const EXIT_RUNTIME: u8 = 1;

const AFTER_HELP: &str = "\
OUTPUT:
    A single file names the file and prints its tempo (2 decimals).
    A directory prints a FILE/BPM table, then a note for anything that failed.

ENVIRONMENT:
    RBA_FFMPEG           ffmpeg executable        (alias: rba_ffmpeg)
    RBA_FFPROBE          ffprobe executable       (alias: rba_ffprobe)
    RBA_SAMPLE_RATE      analysis rate in Hz      (alias: rba_sample_rate)";

/// BPM analyzer — a Rust rewrite of the MixMeister BPM Analyzer core.
#[derive(Debug, Parser)]
#[command(
    name = "rba",
    version,
    about = "BPM analyzer — a Rust rewrite of the MixMeister BPM Analyzer core",
    after_help = AFTER_HELP
)]
pub struct Cli {
    /// An audio file, or a directory to scan
    #[arg(value_name = "PATH", value_parser = parse_existing_path)]
    pub path: PathBuf,

    /// Print pipeline detail to stderr
    #[arg(short, long)]
    pub verbose: bool,

    /// Recurse into subdirectories when PATH is a directory
    #[arg(short, long)]
    pub recursive: bool,

    /// ffmpeg executable to use
    #[arg(long, value_name = "PATH", env = "RBA_FFMPEG")]
    pub ffmpeg: Option<PathBuf>,

    /// ffprobe executable to use (diagnostics only)
    #[arg(long, value_name = "PATH", env = "RBA_FFPROBE", hide = true)]
    pub ffprobe: Option<PathBuf>,

    /// Analysis sample rate in Hz
    #[arg(
        long,
        value_name = "HZ",
        env = "RBA_SAMPLE_RATE",
        default_value_t = DEFAULT_SAMPLE_RATE
    )]
    pub sample_rate: u32,
}

/// `clap` value parser that rejects a PATH that does not exist, so a typo is
/// reported as a usage error (exit 2) rather than a runtime failure (exit 1).
fn parse_existing_path(s: &str) -> std::result::Result<PathBuf, String> {
    let p = PathBuf::from(s);
    if p.exists() {
        Ok(p)
    } else {
        Err(format!("no such file or directory: {s}"))
    }
}

impl Cli {
    /// Resolve the analysis configuration from the parsed options.
    pub fn config(&self) -> Config {
        Config::new(
            self.ffmpeg.clone(),
            self.ffprobe.clone(),
            self.sample_rate,
            self.verbose,
        )
    }
}

/// Extensions considered audio when scanning a directory.
///
/// There is no canonical crate for this; it is a plain allow-list.
const AUDIO_EXTENSIONS: &[&str] = &[
    "wav", "wave", "mp3", "m4a", "m4b", "aac", "flac", "ogg", "oga", "opus", "wma", "aif", "aiff",
    "aifc", "alac", "mp4", "ape", "wv", "tta", "dsf", "dff", "caf", "au", "amr", "ac3", "dts",
    "mka", "webm", "mpc", "spx", "3gp", "ra",
];

fn has_audio_extension(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .map(|e| AUDIO_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// macOS writes `._name` sidecar files next to media; they are not audio.
fn is_resource_fork(p: &Path) -> bool {
    p.file_name()
        .map(|n| n.to_string_lossy().starts_with("._"))
        .unwrap_or(false)
}

/// Collect audio files under `dir`, sorted, optionally recursing.
fn collect_audio_files(dir: &Path, recursive: bool) -> Vec<PathBuf> {
    let depth = if recursive { usize::MAX } else { 1 };
    let mut files: Vec<PathBuf> = WalkDir::new(dir)
        .max_depth(depth)
        .follow_links(false)
        .sort_by_file_name()
        .into_iter()
        .filter_map(std::result::Result::ok)
        .filter(|e| e.file_type().is_file())
        .map(walkdir::DirEntry::into_path)
        .filter(|p| has_audio_extension(p) && !is_resource_fork(p))
        .collect();
    files.sort();
    files
}

/// A progress bar, but only when a human is watching and there is real work.
fn make_progress(total: usize, cfg: &Config) -> Option<ProgressBar> {
    if cfg.verbose || total < 2 || !std::io::stderr().is_terminal() {
        return None;
    }
    let style = ProgressStyle::with_template(
        "{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} files",
    )
    .ok()?
    .progress_chars("=> ");
    let pb = ProgressBar::new(total as u64);
    pb.set_style(style);
    Some(pb)
}

/// Map the legacy lowercase environment aliases onto the canonical names so
/// `clap`'s `env` fallback sees them. Runs before any thread is spawned.
pub fn normalize_env_aliases() {
    for (lower, upper) in [
        ("rba_ffmpeg", "RBA_FFMPEG"),
        ("rba_ffprobe", "RBA_FFPROBE"),
        ("rba_sample_rate", "RBA_SAMPLE_RATE"),
    ] {
        if std::env::var_os(upper).is_none() {
            if let Some(v) = std::env::var_os(lower) {
                std::env::set_var(upper, v);
            }
        }
    }
}

/// Entry point used by `main`; returns the process exit code.
pub fn run() -> ExitCode {
    normalize_env_aliases();
    // `clap` handles --help/--version and exits with 2 on usage errors.
    let cli = Cli::parse();
    let cfg = cli.config();

    match dispatch(&cli, &cfg) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("rba: {e}");
            ExitCode::from(EXIT_RUNTIME)
        }
    }
}

fn dispatch(cli: &Cli, cfg: &Config) -> Result<ExitCode> {
    if cli.path.is_dir() {
        run_directory(&cli.path, cfg, cli.recursive)
    } else {
        run_file(&cli.path, cfg)
    }
}

fn run_file(path: &Path, cfg: &Config) -> Result<ExitCode> {
    let a = analyze::analyze_file(path, cfg)?;
    report::print_single(&a, cfg.verbose);
    Ok(ExitCode::from(EXIT_OK))
}

fn analyze_one(path: &Path, cfg: &Config) -> Row {
    match analyze::analyze_file(path, cfg) {
        Ok(a) => Row::Ok(a),
        Err(e) => Row::Failed {
            name: analyze::display_name(path),
            message: e.to_string(),
        },
    }
}

fn run_directory(dir: &Path, cfg: &Config, recursive: bool) -> Result<ExitCode> {
    let files = collect_audio_files(dir, recursive);
    if files.is_empty() {
        eprintln!("rba: no audio files found in {}", dir.display());
        return Ok(ExitCode::from(EXIT_RUNTIME));
    }

    // `par_iter().map().collect()` preserves input order, so the table stays
    // sorted by file name regardless of completion order.
    let progress = make_progress(files.len(), cfg);
    let rows: Vec<Row> = files
        .par_iter()
        .map(|f| {
            let row = analyze_one(f, cfg);
            if let Some(pb) = &progress {
                pb.inc(1);
            }
            row
        })
        .collect();
    if let Some(pb) = progress {
        pb.finish_and_clear();
    }

    if cfg.verbose {
        for row in &rows {
            if let Row::Ok(a) = row {
                eprint!("{}", report::details(a));
            }
        }
    }

    print!("{}", report::render_table(&rows));

    let failed = rows
        .iter()
        .filter(|r| matches!(r, Row::Failed { .. }))
        .count();
    if failed == files.len() {
        Ok(ExitCode::from(EXIT_RUNTIME))
    } else {
        Ok(ExitCode::from(EXIT_OK))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    fn parse(args: &[&str]) -> std::result::Result<Cli, clap::Error> {
        Cli::try_parse_from(std::iter::once("rba").chain(args.iter().copied()))
    }

    #[test]
    fn clap_definition_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn parses_verbose_and_recursive() {
        let c = parse(&["-v", "-r", "Cargo.toml"]).unwrap();
        assert!(c.verbose);
        assert!(c.recursive);
        assert_eq!(c.path, PathBuf::from("Cargo.toml"));
    }

    #[test]
    fn rejects_unknown_option() {
        let e = parse(&["--definitely-not-an-option"]).unwrap_err();
        assert_eq!(e.kind(), clap::error::ErrorKind::UnknownArgument);
    }

    #[test]
    fn rejects_missing_path() {
        let e = parse(&[]).unwrap_err();
        assert_eq!(
            e.kind(),
            clap::error::ErrorKind::MissingRequiredArgument
        );
    }

    #[test]
    fn rejects_two_paths() {
        assert!(parse(&["Cargo.toml", "README.md"]).is_err());
    }

    #[test]
    fn rejects_nonexistent_path() {
        let e = parse(&["/definitely/not/here.wav"]).unwrap_err();
        assert_eq!(e.kind(), clap::error::ErrorKind::ValueValidation);
        assert!(e.to_string().contains("no such file or directory"));
    }

    #[test]
    fn double_dash_escapes_dashed_path() {
        // `--` still works; the value still has to exist.
        assert!(parse(&["--", "-weird-name.mp3"]).is_err());
        let c = parse(&["--", "Cargo.toml"]).unwrap();
        assert_eq!(c.path, PathBuf::from("Cargo.toml"));
    }

    #[test]
    fn sample_rate_defaults_and_overrides() {
        let c = parse(&["Cargo.toml"]).unwrap();
        assert_eq!(c.sample_rate, DEFAULT_SAMPLE_RATE);
        let c = parse(&["--sample-rate", "22050", "Cargo.toml"]).unwrap();
        assert_eq!(c.sample_rate, 22050);
    }

    #[test]
    fn config_falls_back_to_bare_program_names() {
        let c = parse(&["Cargo.toml"]).unwrap();
        let cfg = c.config();
        assert_eq!(cfg.ffmpeg_path(), Path::new("ffmpeg"));
        assert_eq!(cfg.ffprobe_path(), Path::new("ffprobe"));
    }

    #[test]
    fn extension_filter_is_case_insensitive() {
        assert!(has_audio_extension(Path::new("a.WAV")));
        assert!(has_audio_extension(Path::new("a.flac")));
        assert!(!has_audio_extension(Path::new("a.txt")));
        assert!(!has_audio_extension(Path::new("noext")));
    }

    #[test]
    fn resource_forks_are_detected() {
        assert!(is_resource_fork(Path::new("._song.wav")));
        assert!(!is_resource_fork(Path::new("song.wav")));
    }

    #[test]
    fn collect_audio_files_is_flat_by_default_and_recursive_on_demand() {
        let dir = std::env::temp_dir().join(format!("rba-walk-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(dir.join("top.wav"), b"x").unwrap();
        std::fs::write(dir.join("notes.txt"), b"x").unwrap();
        std::fs::write(dir.join("._fork.wav"), b"x").unwrap();
        std::fs::write(dir.join("sub").join("deep.mp3"), b"x").unwrap();

        // Default: only the immediate directory, resource forks filtered out.
        let flat = collect_audio_files(&dir, false);
        assert_eq!(flat, vec![dir.join("top.wav")], "flat scan");

        // Recursive: nested audio is found, non-audio is not.
        let deep = collect_audio_files(&dir, true);
        let names: Vec<String> = deep
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert!(names.contains(&"top.wav".to_string()), "{names:?}");
        assert!(names.contains(&"deep.mp3".to_string()), "{names:?}");
        assert!(!names.iter().any(|n| n.ends_with(".txt")), "{names:?}");
        assert!(!names.iter().any(|n| n.starts_with("._")), "{names:?}");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
