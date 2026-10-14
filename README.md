# RBA: Rust BPM Analyzer

A cross-platform CLI that detects the tempo of audio files. It is a **Rust rewrite of the analysis core of the 2008 macOS application _MixMeister BPM Analyzer_**, recovered by static reverse engineering of the original binary.

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
  -r, --recursive         Recurse into subdirectories when PATH is a directory
      --ffmpeg <PATH>     ffmpeg executable to use [env: RBA_FFMPEG=]
      --sample-rate <HZ>  Analysis sample rate in Hz [env: RBA_SAMPLE_RATE=] [default: 44100]
  -h, --help              Print help
  -V, --version           Print version
```

* **A file** prints the file name and its tempo (2 decimals).
* **A directory** prints a `FILE / BPM` table, sorted, skipping macOS `._*` files.
  Files are analysed **in parallel**; a file that fails to decode shows `FAILED`
  and its reason below the table, without aborting the run.

### Environment

| Variable | Alias | Default | Purpose |
| --- | --- | --- | --- |
| `RBA_FFMPEG` | `rba_ffmpeg` | `ffmpeg` | ffmpeg executable |
| `RBA_FFPROBE` | `rba_ffprobe` | `ffprobe` | source format info in `--verbose` |
| `RBA_SAMPLE_RATE` | `rba_sample_rate` | `44100` | analysis sample rate — leave alone unless experimenting |

Command-line options take precedence over the environment.

## How it works

Every input is decoded to **mono / 44100 Hz / f32**, then:

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

The algorithm is **not** autocorrelation-based, does not use a log spectrum, and excludes the DC bin.

## Python reference implementation

[`rba.py`](rba.py) is a single-file Python implementation of the same algorithm, with the same CLI. It loads audio with **librosa** and uses **numpy / scipy** for the DSP, so it is a genuinely independent second implementation: different language, different FFT library, different summation strategy, different decoder.

```bash
pip install librosa numpy scipy
python rba.py [OPTIONS] <PATH>

usage: rba.py [-h] [-v] [-r] [--sample-rate HZ] [-V] PATH

BPM analyzer — Python reference implementation of the MixMeister BPM Analyzer core.

positional arguments:
  PATH              an audio file, or a directory to scan

options:
  -h, --help        show this help message and exit
  -v, --verbose     print pipeline detail to stderr
  -r, --recursive   recurse into subdirectories when PATH is a directory
  --sample-rate HZ  analysis sample rate (default: 44100)
  -V, --version     show program's version number and exit
```
