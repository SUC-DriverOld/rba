# RBA: Rust BPM & Key Analyzer

A cross-platform CLI that detects the **tempo and the musical key** of audio files.

The tempo half is a **Rust rewrite of the analysis core of the 2008 macOS application _MixMeister BPM Analyzer_**, recovered by static reverse engineering of the original binary. The key half is a port of [libKeyFinder](https://github.com/ibsh/libKeyFinder) (Ibrahim Sha'ath, GPL-3.0).

## Build

```bash
cargo build --release
```

## Usage

```
Usage: rba [OPTIONS] <PATH>

Arguments:
  <PATH>  An audio file, or a directory to scan

Options:
  -v, --verbose           Print pipeline detail to stderr
      --bpmonly           Run only the BPM pass (skip key detection)
      --keyonly           Run only the key pass (skip the BPM pipeline)
  -r, --recursive         Recurse into subdirectories when PATH is a directory
      --sample-rate <HZ>  Analysis sample rate in Hz [env: RBA_SAMPLE_RATE=] [default: 44100]
  -h, --help              Print help
  -V, --version           Print version
```

* **A file** prints the file name, its tempo (2 decimals) and its key.
* **A directory** prints a `FILE / BPM / KEY` table, sorted, skipping macOS `._*` files, with a progress bar on a terminal.
* `--bpmonly` drops the key column and skips that pass; `--keyonly` drops the tempo fields and skips the BPM pipeline. The two conflict (usage error, exit 2). Files are analysed **in parallel**; a file that fails to decode shows `FAILED` and its reason below the table, without aborting the run.

### Environment

| Variable | Alias | Default | Purpose |
| --- | --- | --- | --- |
| `RBA_USE_FFMPEG` | `rba_use_ffmpeg` | unset | decode with the external ffmpeg instead of the built-in Symphonia engine (flag: any value except `0`/`false`/`no`/`off`) |
| `RBA_FFMPEG` | `rba_ffmpeg` | `ffmpeg` | ffmpeg executable for that backend |
| `RBA_SAMPLE_RATE` | `rba_sample_rate` | `44100` | analysis sample rate — leave alone unless experimenting |

Analysis rates must be 8000–192000 Hz; `0` keeps the default 44100 Hz. Native
source rates must be 1000–768000 Hz. Rates outside these bounds are rejected
before decoder or resampler allocation.

### Audio decoding

Native decoding reads the complete stream. Bitrate-based duration estimates in
unindexed MPEG/ADTS streams are not used as gapless trimming boundaries; explicit
encoder delay and padding are still respected. Decode errors, stream changes,
checksum failures and discrepancies against reliable frame counts are reported
as failures rather than successful partial analyses. Raw streams without a
declared length cannot prove that an otherwise valid prefix was truncated.
Exact frame-count comparisons cover WAVE/AIFF/FLAC and MPEG with explicit
gapless metadata. MP4/AAC and Ogg timing includes codec priming or granule origins,
so these formats rely on decoder-reported errors rather than exact frame counts.
This intentionally rejects damaged recordings and unfinished WAV headers that
were previously accepted as partial audio; repair or re-encode those files first.

The opt-in FFmpeg backend also stops on decode errors and checks reliable frame
counts when the container provides them. Resampling drains the filter tail and
removes its initial delay while preserving the rounded output sample count.

### Tests

```sh
cargo test --locked --all-targets
```

The committed synthetic MP3 fixtures cover unindexed VBR decoding, gapless
trimming and damaged inputs without requiring an encoder at test time. FFmpeg
backend tests run when FFmpeg is available; set `RBA_REQUIRE_FFMPEG=1` to require
them. CI runs both backends on Windows, macOS and Linux. See
`tests/fixtures/README.md` for fixture provenance and regeneration.

### Supported formats

Decoding uses the built-in [Symphonia](https://github.com/pdeljanov/Symphonia) engine (wav/aiff/flac/alac/mp3/aac/vorbis/ogg/mp4/mkv, resampled to 44100 Hz with `rubato`) — no external dependencies. Set `RBA_USE_FFMPEG=1` to switch to an external `ffmpeg` (path via `RBA_FFMPEG`), which additionally covers formats Symphonia has no codec for (wma, ape, wv, dsf, …). Both backends fold multichannel to mono with a plain channel sum, matching the original analyzer's PCM reader.

## How it works

Every input is decoded to **mono / 44100 Hz / f32**. The two analyses then run over the same samples, completely independently:

### Tempo

```
PCM ─> spectral thumbnail        one element per 256 samples:
     │                           Σ |FFT bins 1..127| of a Hamming-windowed frame
   peak-hold envelope            linear decay of peak / 15 per element
     │
   onset detection               slope or magnitude-ratio, 4-element refractory,
     │                           threshold ladder relaxed by /1.2
   tempo via phase histogram     scan 82.0…163.9 BPM in 0.1 steps; score by how
     │                           strongly onsets align to the beat phase
   9 windows × {÷4, ÷8}          upper median of up to 18 estimates
     │
   half-tempo fold-up            double when >53% of onsets are ~1 bar apart
     │
   short-clip override           clips < 30 s are forced into [82, 164] from 240/duration
     │
    BPM
```

The tempo algorithm is **not** autocorrelation-based, does not use a log spectrum, and excludes the DC bin.

### Key

```
PCM ─> low-pass ~2 kHz          161-tap windowed sinc
     │  + decimate               ÷10 at 44.1 kHz, so analysis runs at 4410 Hz
     │
   16384-point frames           hop 4096, Blackman window, real FFT
     │
   72-bin chromagram            constant-Q filterbank: 6 octaves × 12 semitones,
     │                           raised-cosine kernels over FFT bins
   collapse over hops           mean of every hop's 72-band vector
     │
   tone-profile matching        cosine similarity against 24 rotations of the
     │                           major/minor profiles, best score wins
    KEY
```

Key detection is on by default; `--bpmonly` skips it (drops the `KEY` column), `--keyonly` runs only it (drops the `BPM` fields — the tempo pipeline is skipped entirely).

## Python reference implementation

[`rba.py`](rba.py) is a single-file Python implementation of the same algorithm, with the same CLI. It loads audio with **librosa** and uses **numpy / scipy** for the DSP, so it is a genuinely independent second implementation: different language, different decoder, different FFT library, different summation strategy.

### Install

```bash
pip install librosa numpy scipy tqdm
```

### Usage

```
usage: rba.py [-h] [-v] [-r] [--sample-rate HZ] [--bpmonly | --keyonly] [-V] PATH

BPM & key analyzer — Python reference implementation of the MixMeister BPM Analyzer core and of the libKeyFinder key detector.

positional arguments:
  PATH              an audio file, or a directory to scan

options:
  -h, --help        show this help message and exit
  -v, --verbose     print pipeline detail to stderr
  -r, --recursive   recurse into subdirectories when PATH is a directory
  --sample-rate HZ  analysis sample rate (default: 44100)
  --bpmonly         run only the BPM pass (skip key detection)
  --keyonly         run only the key pass (skip the BPM pipeline)
  -V, --version     show program's version number and exit
```
