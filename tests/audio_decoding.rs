use rba::{audio, Config};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

struct FixtureDir(PathBuf);

impl FixtureDir {
    fn new() -> Self {
        for _ in 0..64 {
            let path = std::env::temp_dir().join(format!(
                "rba-audio-{}-{}",
                std::process::id(),
                NEXT_DIR.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => panic!("create fixture directory: {e}"),
            }
        }
        panic!("could not create a unique fixture directory");
    }

    fn file(&self, name: &str, data: &[u8]) -> PathBuf {
        let path = self.0.join(name);
        fs::write(&path, data).unwrap();
        path
    }
}

impl Drop for FixtureDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn native_config() -> Config {
    let mut cfg = Config::new(44100, false, true, true);
    cfg.use_ffmpeg = false;
    cfg
}

#[test]
fn vbr_without_xing_decodes_all_frames_instead_of_trimming_to_an_estimate() {
    let dir = FixtureDir::new();
    let path = dir.file(
        "VBR 音频.mp3",
        include_bytes!("fixtures/vbr-without-xing.mp3"),
    );
    let samples = audio::decode(&path, &native_config()).expect("decode complete VBR stream");
    // Independently decoded by FFmpeg: 461 MPEG frames, including encoder delay/padding.
    assert_eq!(samples.len(), 461 * 1152);
}

#[test]
fn indexed_mp3_preserves_gapless_delay_and_padding() {
    let dir = FixtureDir::new();
    let path = dir.file("indexed.mp3", include_bytes!("fixtures/indexed.mp3"));
    assert_eq!(audio::decode(&path, &native_config()).unwrap().len(), 44100);
}

#[test]
fn truncated_indexed_mp3_is_an_error_instead_of_a_successful_partial_result() {
    let dir = FixtureDir::new();
    let data = include_bytes!("fixtures/indexed.mp3");
    let path = dir.file("truncated.mp3", &data[..data.len() / 2]);
    assert!(audio::decode(&path, &native_config()).is_err());
}

#[test]
fn corrupted_indexed_mp3_is_an_error_instead_of_skipping_audio() {
    let dir = FixtureDir::new();
    let mut data = include_bytes!("fixtures/indexed.mp3").to_vec();
    let length = data.len();
    data[length / 3..2 * (length / 3)].fill(0);
    let path = dir.file("damaged.mp3", &data);
    assert!(audio::decode(&path, &native_config()).is_err());
}

#[test]
fn library_rejects_invalid_analysis_rates_before_opening_media() {
    for rate in [0, 1, 7999, 192001, 1_000_000_000, u32::MAX] {
        let mut cfg = native_config();
        cfg.sample_rate = rate;
        let error = audio::decode(std::path::Path::new("missing-file.wav"), &cfg).unwrap_err();
        assert!(
            error.to_string().contains("analysis sample rate"),
            "{error}"
        );
    }
}

#[test]
fn cli_rejects_extreme_rates_as_usage_errors() {
    for rate in ["1", "7999", "192001", "1000000000", "4294967295"] {
        let output = std::process::Command::new(env!("CARGO_BIN_EXE_rba"))
            .args(["--sample-rate", rate, "Cargo.toml"])
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(2));
        assert!(String::from_utf8_lossy(&output.stderr).contains("sample rate"));
    }
}

fn ffmpeg_config() -> Option<Config> {
    let program = std::env::var_os("RBA_TEST_FFMPEG").unwrap_or_else(|| "ffmpeg".into());
    match std::process::Command::new(&program)
        .arg("-version")
        .output()
    {
        Ok(output) if output.status.success() => {
            let mut cfg = native_config();
            cfg.use_ffmpeg = true;
            cfg.ffmpeg = program.into();
            Some(cfg)
        }
        _ if std::env::var_os("RBA_REQUIRE_FFMPEG").is_some() => {
            panic!("FFmpeg is required for this test run")
        }
        _ => {
            eprintln!("FFmpeg not available; skipping opt-in backend coverage");
            None
        }
    }
}

#[test]
fn ffmpeg_preserves_complete_vbr_and_gapless_indexed_audio() {
    let Some(cfg) = ffmpeg_config() else { return };
    let dir = FixtureDir::new();
    for (name, data, expected) in [
        (
            "vbr.mp3",
            include_bytes!("fixtures/vbr-without-xing.mp3").as_slice(),
            461 * 1152,
        ),
        (
            "indexed.mp3",
            include_bytes!("fixtures/indexed.mp3").as_slice(),
            44100,
        ),
    ] {
        let path = dir.file(name, data);
        assert_eq!(audio::decode(&path, &cfg).unwrap().len(), expected);
    }
}

#[test]
fn ffmpeg_rejects_truncated_and_corrupted_indexed_audio() {
    let Some(cfg) = ffmpeg_config() else { return };
    let dir = FixtureDir::new();
    let data = include_bytes!("fixtures/indexed.mp3");
    let path = dir.file("truncated.mp3", &data[..data.len() / 2]);
    assert!(audio::decode(&path, &cfg).is_err());
    let mut damaged = data.to_vec();
    let length = damaged.len();
    damaged[length / 3..2 * (length / 3)].fill(0);
    let path = dir.file("damaged.mp3", &damaged);
    assert!(audio::decode(&path, &cfg).is_err());
}

#[test]
fn ffmpeg_treats_dash_filename_as_a_local_file() {
    let Some(cfg) = ffmpeg_config() else { return };
    let dir = FixtureDir::new();
    dir.file("-", include_bytes!("fixtures/indexed.mp3"));
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_rba"))
        .args(["--keyonly", "--", "-"])
        .current_dir(&dir.0)
        .env("RBA_USE_FFMPEG", "1")
        .env("RBA_FFMPEG", &cfg.ffmpeg)
        .env_remove("RBA_SAMPLE_RATE")
        .env_remove("rba_sample_rate")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn aac_edit_lists_are_not_mistaken_for_pcm_frame_counts() {
    let dir = FixtureDir::new();
    let path = dir.file("aac.m4a", include_bytes!("fixtures/aac.m4a"));
    let native = audio::decode(&path, &native_config()).expect("native AAC");
    assert!(native.len().abs_diff(44100) <= 2048);
    if let Some(cfg) = ffmpeg_config() {
        let ffmpeg = audio::decode(&path, &cfg).expect("FFmpeg AAC");
        assert!(ffmpeg.len().abs_diff(44100) <= 2048);
    }
}

#[test]
fn ffmpeg_opus_preskip_is_not_mistaken_for_missing_pcm() {
    let Some(cfg) = ffmpeg_config() else { return };
    let dir = FixtureDir::new();
    let path = dir.file("opus.ogg", include_bytes!("fixtures/opus.ogg"));
    assert_eq!(
        audio::decode(&path, &cfg).expect("FFmpeg Opus").len(),
        44100
    );
}

#[test]
fn vorbis_streams_with_a_nonzero_granule_origin_remain_playable() {
    let dir = FixtureDir::new();
    let path = dir.file("offset.ogg", include_bytes!("fixtures/vorbis-offset.ogg"));
    assert_eq!(audio::decode(&path, &native_config()).unwrap().len(), 44224);
    if let Some(cfg) = ffmpeg_config() {
        assert_eq!(audio::decode(&path, &cfg).unwrap().len(), 44224);
    }
}

#[test]
fn ffmpeg_vorbis_first_window_discard_is_not_an_incomplete_stream() {
    let Some(cfg) = ffmpeg_config() else { return };
    let dir = FixtureDir::new();
    let path = dir.file("vorbis.ogg", include_bytes!("fixtures/vorbis.ogg"));
    let samples = audio::decode(&path, &cfg).expect("normal Vorbis");
    assert!(samples.len().abs_diff(44100) <= 256);
}

#[test]
fn truncated_wave_is_rejected_by_both_decoders() {
    let dir = FixtureDir::new();
    let data = include_bytes!("fixtures/pcm.wav");
    let path = dir.file("truncated.wav", &data[..data.len() / 2]);
    assert!(audio::decode(&path, &native_config()).is_err());
    if let Some(cfg) = ffmpeg_config() {
        assert!(audio::decode(&path, &cfg).is_err());
    }
}
