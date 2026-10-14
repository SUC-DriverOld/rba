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
      --nokey             Skip key detection (BPM only)
  -r, --recursive         Recurse into subdirectories when PATH is a directory
      --ffmpeg <PATH>     ffmpeg executable to use [env: RBA_FFMPEG=]
      --sample-rate <HZ>  Analysis sample rate in Hz [env: RBA_SAMPLE_RATE=] [default: 44100]
  -h, --help              Print help
  -V, --version           Print version
```

* **A file** prints the file name, its tempo (2 decimals) and its key.
* **A directory** prints a `FILE / BPM / KEY` table, sorted, skipping macOS `._*` files, with a progress bar on a terminal.
* `--nokey` drops the key column and skips that analysis. Files are analysed **in parallel**; a file that fails to decode shows `FAILED` and its reason below the table, without aborting the run.

### Environment

| Variable | Alias | Default | Purpose |
| --- | --- | --- | --- |
| `RBA_FFMPEG` | `rba_ffmpeg` | `ffmpeg` | ffmpeg executable |
| `RBA_FFPROBE` | `rba_ffprobe` | `ffprobe` | source format info in `--verbose` |
| `RBA_SAMPLE_RATE` | `rba_sample_rate` | `44100` | analysis sample rate — leave alone unless experimenting |

Command-line options take precedence over the environment.

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

Key detection is on by default; `--nokey` skips it and drops the `KEY` column.

## Python reference implementation

[`rba.py`](rba.py) is a single-file Python implementation of the same algorithm, with the same CLI. It loads audio with **librosa** and uses **numpy / scipy** for the DSP, so it is a genuinely independent second implementation: different language, different decoder, different FFT library, different summation strategy.

### Install

```bash
pip install librosa numpy scipy tqdm
```

### Usage

```
usage: rba.py [-h] [-v] [-r] [--sample-rate HZ] [--nokey] [-V] PATH

BPM & key analyzer — Python reference implementation of the MixMeister BPM Analyzer core and of the libKeyFinder key detector.

positional arguments:
  PATH              an audio file, or a directory to scan

options:
  -h, --help        show this help message and exit
  -v, --verbose     print pipeline detail to stderr
  -r, --recursive   recurse into subdirectories when PATH is a directory
  --sample-rate HZ  analysis sample rate (default: 44100)
  --nokey           skip key detection (BPM only)
  -V, --version     show program's version number and exit
```
