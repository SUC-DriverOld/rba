#!/usr/bin/env python3
"""rba — Rust BPM & Key Analyzer (Python reference implementation).

A single-file Python port of the same reverse-engineered algorithm that
``src/`` implements in Rust. It is deliberately a *reference* implementation:
readable, vectorised with numpy/scipy where it matters, and byte-for-byte
comparable with the Rust binary.

    rba.py <file>        → print that file's BPM and key
    rba.py <directory>   → print a table of every audio file inside

requirements:
    librosa, numpy, scipy, tqdm
"""

import argparse
import math
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np
from scipy.fft import irfft, rfft
from scipy.signal import lfilter
from tqdm import tqdm

__version__ = "0.1.0"

# ─────────────────────────────────────────────────────────────────────────────
# Constants — spec §9
# ─────────────────────────────────────────────────────────────────────────────

#: The original hard-codes CoreAudio's client format to 44100 Hz mono (spec §1).
DEFAULT_SAMPLE_RATE = 44100
#: FFT / thumbnail frame size (spec §2).
FFT_SIZE = 256
#: Refractory distance between accepted onsets, in thumbnail elements (spec §4).
IMPACT_GAP = 4
#: Tempo search range, scanned in 0.1 steps (spec §5).
BPM_MIN = 82.0
BPM_STEPS = 820
#: Analysis window length in PCM samples: 20 s at 44100 Hz.
WINDOW_SAMPLES = 882_000
#: Phase histogram resolution.
PHASE_BINS = 40

#: `constants.h`: 6 octaves of 12 semitones.
SEMITONES = 12
OCTAVES = 6
BANDS = SEMITONES * OCTAVES
FFT_FRAME_SIZE = 16384
HOP_SIZE = FFT_FRAME_SIZE // 4
DIRECT_SK_STRETCH = 0.8
LPF_ORDER = 160
LPF_FFT_SIZE = 2048

#: `FREQUENCIES[]` in constants.cpp — C1 (32.703 Hz) up to B6 (1975.533 Hz).
BAND_FREQUENCIES = (
    32.7031956625748, 34.647828872109, 36.708095989676, 38.8908729652601,
    41.2034446141088, 43.6535289291255, 46.2493028389543, 48.9994294977187,
    51.9130871974932, 55, 58.2704701897613, 61.7354126570155,
    65.4063913251497, 69.2956577442181, 73.4161919793519, 77.7817459305203,
    82.4068892282175, 87.307057858251, 92.4986056779087, 97.9988589954374,
    103.826174394986, 110, 116.540940379523, 123.470825314031,
    130.812782650299, 138.591315488436, 146.832383958704, 155.563491861041,
    164.813778456435, 174.614115716502, 184.997211355817, 195.997717990875,
    207.652348789973, 220, 233.081880759045, 246.941650628062,
    261.625565300599, 277.182630976872, 293.664767917408, 311.126983722081,
    329.62755691287, 349.228231433004, 369.994422711635, 391.99543598175,
    415.304697579946, 440.000000000001, 466.163761518091, 493.883301256125,
    523.251130601198, 554.365261953745, 587.329535834816, 622.253967444163,
    659.255113825741, 698.456462866009, 739.98884542327, 783.9908719635,
    830.609395159892, 880.000000000002, 932.327523036182, 987.76660251225,
    1046.5022612024, 1108.73052390749, 1174.65907166963, 1244.50793488833,
    1318.51022765148, 1396.91292573202, 1479.97769084654, 1567.981743927,
    1661.21879031978, 1760, 1864.65504607236, 1975.5332050245,
)

MAJOR_TONE_PROFILE = (
    7.23900502618145225142, 3.50351166725158691406, 3.58445177536649417505,
    2.84511816478676315967, 5.81898892118549859731, 4.55865057415321039969,
    2.44778850545506543313, 6.99473192146829525484, 3.39106613673504853068,
    4.55614256655143456953, 4.07392666663523606019, 4.45932757378886890365,
)
MINOR_TONE_PROFILE = (
    7.00255045060284420089, 3.14360279015996679775, 4.35904319714962529275,
    5.40418120718934069657, 3.67234420879306133756, 4.08971184917797891956,
    3.90791435991553992579, 6.19960288562316463867, 3.63424625625277419871,
    2.87241191079875557435, 5.35467999794542670600, 3.83242038595048351013,
)
OCTAVE_WEIGHTS = (
    0.39997267549999998559, 0.55634425248300645173, 0.52496636345143543600,
    0.60847548384277727607, 0.59898115679999996974, 0.49072435317960994006,
)

#: `key_t` order — flats, not sharps, and `silence` last.
KEY_NAMES = (
    "A major", "A minor", "Bb major", "Bb minor", "B major", "B minor",
    "C major", "C minor", "Db major", "Db minor", "D major", "D minor",
    "Eb major", "Eb minor", "E major", "E minor", "F major", "F minor",
    "Gb major", "Gb minor", "G major", "G minor", "Ab major", "Ab minor",
    "silence",
)

AUDIO_EXTENSIONS = frozenset(
    """
    wav wave mp3 m4a m4b aac flac ogg oga opus wma aif aiff aifc alac mp4 ape
    wv tta dsf dff caf au amr ac3 dts mka webm mpc spx 3gp ra
    """.split()
)

EXIT_OK, EXIT_RUNTIME = 0, 1


# ─────────────────────────────────────────────────────────────────────────────
# Tempo ↔ samples conversion
# ─────────────────────────────────────────────────────────────────────────────


def sam_per_meas(bpm: float, sample_rate: int) -> int:
    """`SamPerMeas` — PCM samples in one 4/4 measure at ``bpm``.

    ``(u32)(sample_rate / ((0.25 * bpm) / 60))``, i.e. ``sample_rate*240/bpm``.
    Computed in float32 to mirror the original's single-precision reciprocal.
    """
    if bpm == 0.0:
        return 0
    meas_per_sec = (np.float32(0.25) * np.float32(bpm)) / np.float32(60.0)
    if meas_per_sec == 0.0:
        return 0
    v = np.float32(sample_rate) / meas_per_sec
    if not np.isfinite(v) or v <= 0:
        return 0
    return int(v)


# ─────────────────────────────────────────────────────────────────────────────
# Spectral thumbnail (spec §2, §3)
# ─────────────────────────────────────────────────────────────────────────────


class _SpecEngFft:
    """Bit-exact port of `CPfSpecEngBuilder` (re/02, re/03).

    ``InitializeFFT`` builds a bit-reversal table and a bit-reversal-*stored*
    twiddle table (``sin(-2πj/N)``, ``-cos(2πj/N)`` as f32 pairs); the data is
    the 256 real samples re-packed as 128 complex, transformed by a
    decimation-in-frequency pass (one twiddle per 2i-complex block, stages
    i = N/4 … 1) and untangled into the 128-bin real spectrum.  Every
    arithmetic step runs in float32 exactly as the DLL does — numpy only
    vectorises the same operations across frames/elements, which does not
    change any individual result.
    """

    def __init__(self, n: int):
        self.n = n
        half = n // 2

        # bitrev[k]: float-slot index for natural index k (always even)
        bitrev = np.zeros(half, dtype=np.int64)
        for k in range(half):
            v7 = half // 2
            v6 = 0
            while v7 > 0:
                v4 = v6 >> 1
                v5 = half if (k & v7) else 0
                v6 = v5 + v4
                v7 >>= 1
            bitrev[k] = v6
        self.bitrev = bitrev

        # twiddle pairs stored at float slots (bitrev[k], bitrev[k]+1)
        tw = np.zeros(2 * half, dtype=np.float32)
        j = np.arange(half, dtype=np.float64)
        tw_sin = np.sin(-2.0 * np.pi * j / n).astype(np.float32)
        tw_cos = (-np.cos(2.0 * np.pi * j / n)).astype(np.float32)
        tw[bitrev] = tw_sin
        tw[bitrev + 1] = tw_cos
        self.tw = tw

        # w[i] = f32(cos(2πi/N)·-0.46 + 0.54), computed in double like the DLL
        idx = np.arange(n, dtype=np.float64)
        self.window = (np.cos(2.0 * np.pi * idx / n) * -0.46 + 0.54).astype(np.float32)

        self._half = half

    def magnitudes(self, frames: np.ndarray) -> np.ndarray:
        """(count, n//2) bin magnitudes for (count, n) real frames."""
        n = self.n
        half = self._half
        count = frames.shape[0]
        work = (frames * self.window).astype(np.float32)
        data = work.reshape(count, half, 2)
        twc = self.tw.reshape(half, 2)
        half_f = np.float32(0.5)

        # ── staged DIF pass (RealFFT) — stages i = N/4 … 1 ───────────────
        i = n >> 2
        while i > 0:
            nblocks = half // (2 * i)
            # one scalar twiddle per 2i-complex block, walked from tw[0]
            for b in range(nblocks):
                base = b * 2 * i
                c = twc[b, 0]
                s = twc[b, 1]
                lo = data[:, base : base + i]
                hi = data[:, base + i : base + 2 * i]
                hr = hi[:, :, 0]
                hi_im = hi[:, :, 1]
                t1 = (s * hr + c * hi_im).astype(np.float32)
                t2 = (hr * c - hi_im * s).astype(np.float32)
                x1r = lo[:, :, 0]
                x1i = lo[:, :, 1]
                hi2r = ((t1 + x1r) * half_f).astype(np.float32)
                lo[:, :, 0] = (hi2r - t1).astype(np.float32)
                hi[:, :, 0] = hi2r
                hi2i = ((x1i - t2) * half_f).astype(np.float32)
                lo[:, :, 1] = (t2 + hi2i).astype(np.float32)
                hi[:, :, 1] = hi2i
            i >>= 1

        # ── untangle pass into the real half-spectrum ────────────────────
        # k = 1 … half/2 (inclusive) pairs v = complex[bitrev[k]/2] with
        # w = complex[bitrev[half-k]/2]; the last k pairs a bin with itself.
        br = self.bitrev
        kk = np.arange(1, half // 2 + 1)
        v_idx = (br[kk] // 2).astype(np.int64)
        w_idx = (br[half - kk] // 2).astype(np.int64)
        ts = twc[v_idx, 0].reshape(1, -1)
        tc = twc[v_idx, 1].reshape(1, -1)

        # k = 1 … half/2-1 pair distinct bins; k = half/2 pairs a bin with
        # itself (both writes land on the same slot, the second wins).
        pairs = slice(0, half // 2 - 1)
        vp, wp = v_idx[pairs], w_idx[pairs]
        vr = data[:, vp, 0]
        vi = data[:, vp, 1]
        wr = data[:, wp, 0]
        wi = data[:, wp, 1]
        d_im = (vi - wi).astype(np.float32)
        d_re = (vr - wr).astype(np.float32)
        sum_wi = ((wi + wi).astype(np.float32) + d_im).astype(np.float32)   # vi+wi
        A = (ts[:, pairs] * d_re - tc[:, pairs] * sum_wi).astype(np.float32)
        B = (tc[:, pairs] * d_re + ts[:, pairs] * sum_wi).astype(np.float32)
        sum_wr = ((wr + wr).astype(np.float32) + d_re).astype(np.float32)   # vr+wr
        v_new_r = ((sum_wr + A) * half_f).astype(np.float32)
        v_new_i = ((B + d_im) * half_f).astype(np.float32)
        data[:, vp, 0] = v_new_r
        data[:, wp, 0] = (v_new_r - A).astype(np.float32)
        data[:, vp, 1] = v_new_i
        data[:, wp, 1] = (v_new_i - d_im).astype(np.float32)

        # self-paired bin k = half/2 (v == w): reads first, then both writes
        # hit the same slot; the second ([edx]) store wins.
        sp = int(v_idx[half // 2 - 1])
        vr = data[:, sp, 0].copy()
        vi = data[:, sp, 1].copy()
        s_t, c_t = float(twc[v_idx[half // 2 - 1], 0]), float(twc[v_idx[half // 2 - 1], 1])
        sum_wi = (vi + vi).astype(np.float32)
        A = (np.float32(s_t) * np.float32(0.0) - np.float32(c_t) * sum_wi).astype(np.float32)
        B = (np.float32(c_t) * np.float32(0.0) + np.float32(s_t) * sum_wi).astype(np.float32)
        sum_wr = (vr + vr).astype(np.float32)
        data[:, sp, 0] = (((sum_wr + A) * half_f).astype(np.float32) - A).astype(np.float32)
        data[:, sp, 1] = ((B + np.float32(0.0)) * half_f).astype(np.float32)

        # DC fold: data[0].re += data[0].im ; data[0].im = 0
        data[:, 0, 0] = (data[:, 0, 0] + data[:, 0, 1]).astype(np.float32)
        data[:, 0, 1] = 0.0

        # magnitudes read at bit-reversed order (DoFFTToMagnitude)
        src = data[:, br // 2]
        re = src[:, :, 0]
        im = src[:, :, 1]
        return np.sqrt((re * re + im * im).astype(np.float32))


_SPEC_FFT = _SpecEngFft(FFT_SIZE)


def spec_thumbnail(samples: np.ndarray) -> np.ndarray:
    """One element per 256-sample frame: sum of |FFT bins 1..127| of a Hamming window.

    DC (bin 0) and Nyquist (bin 128) are excluded, exactly as ``GetSpecEnergy``
    does for N=256.  The whole computation is float32 (see `_SpecEngFft`), and
    the 127 bins are accumulated strictly left-to-right like the original.
    """
    n = FFT_SIZE
    count = samples.size // n
    if count == 0:
        return np.zeros(0, dtype=np.float32)

    frames = np.ascontiguousarray(samples[: count * n], dtype=np.float32).reshape(count, n)
    mag = _SPEC_FFT.magnitudes(frames)

    # sequential f32 accumulation over bins 1..127, like GetSpecEnergy
    acc = mag[:, 1].copy()
    for k in range(2, n // 2):
        acc += mag[:, k]
    return acc


def peak_hold(t: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    """`PeakHoldThumbnail` — in-place peak-hold envelope with linear decay.

    Seeded with the thumbnail minimum; decays by ``peak / (elePerMeas(164) >> 4)``
    per element. Inherently sequential, so it stays a plain loop.
    """
    if t.size == 0:
        return t.copy()

    spm164 = sam_per_meas(164.0, sample_rate)
    decay_div = (spm164 // FFT_SIZE) >> 4
    if decay_div == 0:
        return t.copy()

    out = t.astype(np.float32).copy()
    env = np.float32(out.min())
    step = np.float32(0.0)
    d = np.float32(decay_div)
    for i in range(out.size):
        cur = out[i]
        if cur <= env:
            out[i] = env
        else:
            env = cur
            step = cur / d
        env = np.float32(env - step)
    return out


def smoothed(t: np.ndarray) -> np.ndarray:
    """`SmoothThumbnail` — centred moving average, window ``min(count, 9)``.

    Fidelity note: the original pre-increments its "add" cursor (starting at 9)
    *before* reading, so its first slide consumes ``x[10]`` and permanently
    skips ``x[9]``. That is reproduced here rather than "fixed", because this
    series feeds the ``fold_up_bpm`` decision. Sliding never engages for
    ``count <= 9``, so the quirk only shows up on longer inputs.
    """
    count = t.size
    out = np.zeros(count, dtype=np.float32)
    if count == 0:
        return out

    window = count if count < 9 else 9
    total = np.float32(t[:window].sum())
    half = window >> 1
    window_f = np.float32(window)

    v17, v18 = 0, 9
    v7 = 0
    while v7 < count:
        out[v7] = total / window_f
        v7 += 1
        if half < v7 < count - half - 1:
            total = np.float32(total - t[v17])
            v17 += 1
            v18 += 1
            if v18 < count:
                total = np.float32(total + t[v18])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Onset ("impact point") detection (spec §4)
# ─────────────────────────────────────────────────────────────────────────────


def find_impact_points(t: np.ndarray, threshold: float, gap: int) -> list[int]:
    """`FindImpactPoints` — slope detector, descending scan, positions in samples."""
    n = t.size
    if n == 0:
        return []

    out: list[int] = []
    idx = n - 1
    peak = n - 1
    refractory = 0
    last = n - 1

    while idx >= 0:
        cur = t[idx]
        if refractory != 0:
            refractory -= 1
            peak = idx
        else:
            if cur > t[peak]:
                peak = idx
            # window_end == idx + gap: both start at n-1+gap and step down together.
            lim = min(last, idx + gap)
            if peak > lim:
                peak = idx
                for j in range(idx, lim):
                    if t[j] > t[peak]:
                        peak = j
            refractory = 0
            if idx != peak and (t[peak] - cur) / np.float32(peak - idx) > threshold:
                peak = idx
                out.append(idx * FFT_SIZE)
                refractory = gap
        idx -= 1
    return out


def find_impact_points_classic(t: np.ndarray, threshold: float, gap: int) -> list[int]:
    """`FindImpactPointsClassic` — magnitude-ratio detector; records the *peak* index."""
    n = t.size
    if n == 0:
        return []

    out: list[int] = []
    idx = n - 1
    peak = n - 1
    refractory = 0
    last = n - 1

    while idx >= 0:
        cur = t[idx]
        if refractory != 0:
            refractory -= 1
            peak = idx
        else:
            if cur > t[peak]:
                peak = idx
            lim = min(last, idx + gap)
            if peak > lim:
                peak = idx
                for j in range(idx, lim):
                    if t[j] > t[peak]:
                        peak = j
            refractory = 0
            if t[peak] > threshold * cur:
                out.append(peak * FFT_SIZE)
                refractory = gap
        idx -= 1
    return out


def _build_with_ladder(t, total_samples, divisor, initial_threshold, max_tries, classic, sample_rate):
    """Relax the threshold by /1.2 until enough onsets appear or tries run out."""
    spm82 = sam_per_meas(82.0, sample_rate)
    if spm82 == 0:
        return []
    target = (total_samples // spm82) * divisor

    found: list[int] = []
    threshold = np.float32(initial_threshold)
    tries = max_tries
    detect = find_impact_points_classic if classic else find_impact_points

    while len(found) < target:
        tries -= 1
        if tries == 0:
            break
        found = detect(t, threshold, IMPACT_GAP)
        threshold = threshold / np.float32(1.2)
    return sorted(found)


def build_impact_points(t, total_samples: int, divisor: int, sample_rate: int) -> list[int]:
    """`BuildImpactPoints` — slope detector, 63-step ladder from 2.4414062."""
    return _build_with_ladder(t, total_samples, divisor, 2.4414062, 63, False, sample_rate)


def build_impact_points_classic(t, total_samples: int, divisor: int, sample_rate: int) -> list[int]:
    """`BuildImpactPointsClassic` — magnitude detector, 20-step ladder from 30.0."""
    return _build_with_ladder(t, total_samples, divisor, 30.0, 20, True, sample_rate)


# ─────────────────────────────────────────────────────────────────────────────
# Note-detector thumbnail (v35) — CIIRFilt high-pass + waveform peak
#
# `CPfNoteDetector`'s ctor initialises a second thumbnail with the same
# 256-sample stride and fills it from the *filtered* waveform: `PushPcm`
# runs the PCM through a `CIIRFilt` configured as 3 cascaded type-0
# ("high-pass") biquad sections (freq 120.0, Q 1.1, gain 0) and
# `BuildPcmThumbnailRange` stores max|x| per 256 filtered samples.
# `BuildProfile` smooths a copy of this thumbnail and runs
# `BuildImpactPointsClassic` on it — that onset set is what `FoldUpBpm`
# sees (docs/02 §6, re/16, re/14).
# ─────────────────────────────────────────────────────────────────────────────


def _iir_hp_biquad(freq: float = 120.0, q: float = 1.1,
                   sample_rate: int = DEFAULT_SAMPLE_RATE) -> tuple[list[float], list[float]]:
    """`CIIRFilt::setfilt` + `spbiln` for one type-0 section (re/16).

    Returns ``(b, k)`` with the digital section

        w[n] = x[n] - k0·w[n-1] - k1·w[n-2]
        y[n] = b0·w[n] + b1·w[n-1] + b2·w[n-2]

    — exactly the recurrence `CIIRFilt::fl` runs (re/16 @ 0x137a8).
    """
    # setfilt: t = tan(freq * pi * (1/sr)) in double; prototype coefficients
    # are stored to f32 arrays.
    t = math.tan(float(np.float32(freq)) * math.pi * (1.0 / float(sample_rate)))
    t2 = np.float32(t * t)
    num = (t2, np.float32(0.0), np.float32(0.0))                       # [t², 0, 0]
    den = (t2, np.float32(float(np.float32(q)) * t), np.float32(1.0))  # [t², Q·t, 1]

    # spbiln: substitute s ← (1-z)/(1+z).  M[m][j] is the coefficient of z^-j
    # in (1-z)^m (1+z)^(2-m).
    matrix = ((1.0, 2.0, 1.0), (1.0, 0.0, -1.0), (1.0, -2.0, 1.0))
    b = [np.float32(0.0)] * 3
    dd = [0.0] * 3                       # denominator accumulator (double)
    for j in range(3):
        acc = np.float32(0.0)
        for m in range(3):
            acc = np.float32(acc + np.float32(matrix[m][j] * num[m]))
            dd[j] += float(np.float32(matrix[m][j] * den[m]))
        b[j] = acc
    g = np.float32(dd[0])                # denominator z^0 coefficient
    b = [np.float32(x / g) for x in b]
    k = [np.float32(dd[1] / g), np.float32(dd[2] / g)]
    return [float(x) for x in b], [float(x) for x in k]


_HP_B, _HP_K = _iir_hp_biquad()


def note_peak_thumbnail(samples: np.ndarray,
                        sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    """v35 — max|x| per 256 samples of the 6th-order 120 Hz high-passed waveform.

    The three sections are identical, so the cascade is the same biquad
    applied three times with persistent state (`fl` runs per 256-sample
    `PushPcm` block; filtering the whole stream section-by-section is
    equivalent). scipy's `lfilter` keeps float32 precision.
    """
    x = np.ascontiguousarray(samples, dtype=np.float32)
    if sample_rate != DEFAULT_SAMPLE_RATE:
        b, k = _iir_hp_biquad(sample_rate=sample_rate)
    else:
        b, k = _HP_B, _HP_K
    a = np.array([1.0, k[0], k[1]], dtype=np.float32)
    bb = np.array(b, dtype=np.float32)
    y = x
    for _ in range(3):
        y = lfilter(bb, a, y)
    count = y.size // FFT_SIZE
    if count == 0:
        return np.zeros(0, dtype=np.float32)
    peaks = np.abs(y[: count * FFT_SIZE].reshape(count, FFT_SIZE)).max(axis=1)
    return peaks.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Beat chain (spec §8, re/17) — CPfBeatBuilder::ClusterFilterPoints and
# BuildBeatPointsAlt: the DLL's final BPM stage.
# ─────────────────────────────────────────────────────────────────────────────

#: Phase 6: connections only span ≤ 32 interval units (≈ 4 measures).
CHAIN_MAX_UNITS = 32
#: Phase 10: chains whose first/last index difference is ≤ this are dropped.
CHAIN_MIN_INDEX_SPAN = 31


def _f32(x) -> float:
    """Round through float32 (the chain stage is all single-precision)."""
    return float(np.float32(x))


def _u32_sat(x: float) -> int:
    """§3.1-A — the chain stage's saturating ``(u32)double`` conversion."""
    if x >= 2147483648.0:
        m, hi = 2147483648.0, 0x80000000
    else:
        m, hi = 0.0, 0
    t = min(max(x, 0.0), 4294967295.0) - m
    return (int(t) ^ hi) & 0xFFFFFFFF


def _calc_point_confidence(thumb: np.ndarray, pos: int) -> float:
    """`CalcPointConfidence` (re/17 §3.2) — 16-element rise around the onset.

    ``Σ thumb[idx..idx+7] − Σ thumb[idx-8..idx-1]`` with every index clamped to
    ``[0, count-1]`` and out-of-range reads returning 0.
    """
    n1 = int(thumb.size) - 1
    idx = pos // FFT_SIZE            # __divdi3: signed, truncating

    def clamp(v: int) -> int:
        if v < 0:
            return 0
        return v if v <= n1 else n1

    lo1, lo2 = clamp(idx - 8), clamp(idx - 1)
    hi1, hi2 = clamp(idx), clamp(idx + 7)
    acc = np.float32(0.0)
    if lo2 >= lo1:
        acc = np.float32(acc - thumb[lo1:lo2 + 1].sum(dtype=np.float32))
    if hi2 >= hi1:
        acc = np.float32(acc + thumb[hi1:hi2 + 1].sum(dtype=np.float32))
    return float(acc)


def _proximity_conf_filter(pos: list[int], conf: list[float], interval: float) -> list[int]:
    """`ProximityConfFilter` (re/17 §3.6) — non-maximum suppression.

    Within ±``trunc(0.45·I)`` samples only the most confident candidate
    survives; survivors keep their relative order. Returns kept indices.
    """
    limit = int(np.float32(np.float32(0.45) * interval))
    n = len(pos)
    nxt = list(range(1, n + 1))          # nxt[n] == n doubles as the end sentinel
    prv = list(range(-1, n))
    head = 0

    cur = head
    while cur < n:
        nxt_cur = nxt[cur]               # advance before a possible erase
        best = cur
        i = cur                          # backward scan (includes cur)
        while i >= 0 and limit > pos[cur] - pos[i]:
            if conf[i] > conf[best]:
                best = i
            i = prv[i]
        j = cur                          # forward scan (includes cur)
        while j < n and limit > pos[j] - pos[cur]:
            if conf[j] > conf[best]:
                best = j
            j = nxt[j]
        if cur != best and conf[best] > conf[cur]:
            p, s = prv[cur], nxt[cur]
            if p >= 0:
                nxt[p] = s
            else:
                head = s
            if s < n:
                prv[s] = p
        cur = nxt_cur

    out = []
    i = head
    while i < n:
        out.append(i)
        i = nxt[i]
    return out


def _connection_candidates(pos: np.ndarray, conf: np.ndarray,
                           interval: float) -> tuple[np.ndarray, np.ndarray, np.float32]:
    """Phase 6 — all pairs (i, j), i < j, within 32 units, with `CalcConQuality`
    scores. Vectorised in float32 like the original."""
    n = pos.size
    empty_i = np.zeros(0, dtype=np.int64)
    if n < 2 or not interval > 0.0:
        return empty_i, empty_i.copy(), np.zeros(0, dtype=np.float32)

    # Break condition: u32_sat(d/I + 0.5) > 32  ⇔  d ≥ 32.5·I (d ≥ 0).
    reach = pos + np.float64(np.float32(np.float32(32.5) * interval))
    hi = np.searchsorted(pos, reach, side="left")

    ii, jj = [], []
    for i in range(n - 1):
        j1 = int(hi[i])
        if j1 > i + 1:
            ii.append(np.full(j1 - (i + 1), i, dtype=np.int64))
            jj.append(np.arange(i + 1, j1, dtype=np.int64))
    if not ii:
        return empty_i, empty_i.copy(), np.zeros(0, dtype=np.float32)
    i_arr = np.concatenate(ii)
    j_arr = np.concatenate(jj)

    d = pos[j_arr] - pos[i_arr]
    n_units = np.trunc(d.astype(np.float64) / np.float64(interval) + 0.5).astype(np.int64)

    x = d.astype(np.float32) / np.float32(interval)
    r = np.float32(x + np.float32(0.5)).astype(np.int32)   # cvttss2si of (x+0.5f)
    frac = x - r.astype(np.float32)
    v = np.float32(1.0) - np.abs(frac)
    phase = np.zeros(x.size, dtype=np.float32)
    gate = v > np.float32(0.8)
    ramp = gate & (v <= np.float32(0.98))
    phase[gate] = np.float32(1.0)
    phase[ramp] = ((v[ramp] - np.float32(0.8)) / np.float32(0.18)).astype(np.float32)

    tol = np.ones(x.size, dtype=np.float32)
    long_span = n_units > 8
    tol[long_span] = ((32 - n_units[long_span]).astype(np.float32)
                      / np.float32(24.0)).astype(np.float32)

    quality = ((phase * tol).astype(np.float32) * conf[i_arr]).astype(np.float32)
    quality = (quality * conf[j_arr]).astype(np.float32)

    keep = (n_units >= 1) & (quality > np.float32(0.0))
    return i_arr[keep], j_arr[keep], quality[keep]


@dataclass
class ChainTrace:
    """Diagnostics from the chain stage (for --verbose)."""
    unit0: float = 0.0                  # SamPerMeas(bpm)>>3, as f32
    interval_est: float = 0.0           # spacing re-estimate before clustering
    candidates: int = 0                 # connections with quality > 0
    processed: int = 0                  # connections consumed before coverage break
    coverage: float = 0.0               # coverage when the assembly stopped
    chains_total: int = 0               # chains in the map when assembly stopped
    chains_kept: int = 0                # chains surviving phases 9-10
    best_members: int = 0               # members of the winning chain
    io_i: float = 0.0                   # final interval (samples per eighth)
    #: per scored chain: (weight, member count, span, mean_conf, own ioI)
    chain_details: list = field(default_factory=list)


def _cluster_filter_points(thumb: np.ndarray, i0: float, points: list[int],
                           trace: ChainTrace) -> tuple[float, list[int]]:
    """`CPfBeatBuilder::ClusterFilterPoints` (re/17 §2) — faithful port.

    Returns ``(ioI, emitted positions)``; the emitted positions are the union
    of surviving chains' onsets (the DLL feeds them to `InterpolatePoints` /
    `ExtrapolatePoints`, which do not influence the BPM).
    """
    # ── phase 1: points + confidence ────────────────────────────────────
    pos: list[int] = []
    conf: list[float] = []
    min_c = max_c = 0.0
    for k, p in enumerate(points):
        c = _calc_point_confidence(thumb, p)
        pos.append(p)
        conf.append(c)
        if k == 0:
            min_c = max_c = c
        else:
            min_c = _f32(min(min_c, c))
            max_c = _f32(max(max_c, c))

    # ── phase 2: normalise confidences to [0, 1] ────────────────────────
    rng = _f32(max_c - min_c)
    scale = rng if rng != 0.0 else 1.0
    conf = [_f32(_f32(c - min_c) / scale) for c in conf]

    # ── phase 3: proximity thinning (uses the *input* interval) ─────────
    kept = _proximity_conf_filter(pos, conf, i0)

    # ── phase 4: re-index ───────────────────────────────────────────────
    pos = [pos[i] for i in kept]
    conf = [conf[i] for i in kept]
    n = len(pos)
    index = list(range(n))              # index == array position

    # ── phase 5: re-estimate the interval from 1-3-unit spacings ────────
    interval = _reestimate_interval(pos, i0)

    # ── phase 6: candidate connections, sorted by quality (desc) ────────
    pos_arr = np.asarray(pos, dtype=np.int64)
    conf_arr = np.asarray(conf, dtype=np.float32)
    ci, cj, cq = _connection_candidates(pos_arr, conf_arr, interval)
    order = np.argsort(-cq, kind="stable")
    ci, cj = ci[order], cj[order]
    trace.candidates = int(ci.size)

    # ── phase 7: greedy chain assembly ──────────────────────────────────
    chain_id = [-1] * n
    active = [True] * n
    nxt: list[list[int]] = [[] for _ in range(n)]
    prv: list[list[int]] = [[] for _ in range(n)]
    chains: dict[int, list[int]] = {}   # id → sorted member indices
    next_id = 1
    active_count = n
    assigned_count = 0

    def deactivate_reach(start: int, links: list[list[int]]) -> None:
        """`DeactivateAllNext/Prev` — clears `active` on every descendant.

        The DLL recurses through *already inactive* nodes as well (no
        active-flag check), so traversal must not stop at inactive points;
        clearing is idempotent, and the next-graph is a DAG (links always
        point towards strictly greater indices), so this terminates.
        """
        nonlocal active_count, assigned_count
        stack = list(links[start])
        while stack:
            p = stack.pop()
            if active[p]:
                active[p] = False
                active_count -= 1
                if chain_id[p] != -1:
                    assigned_count -= 1
            stack.extend(links[p])

    def earliest(a: int) -> int:
        """`GetEarliestLink` — min index over a and its prev-descendants."""
        best = index[a]
        stack = list(prv[a])
        while stack:
            p = stack.pop()
            if index[p] < best:
                best = index[p]
            stack.extend(prv[p])
        return best

    def latest(a: int) -> int:
        """`GetLatestLink` — max index over a and its next-descendants."""
        best = index[a]
        stack = list(nxt[a])
        while stack:
            p = stack.pop()
            if index[p] > best:
                best = index[p]
            stack.extend(nxt[p])
        return best

    def disconnect(this: int, other: int) -> None:
        """`CPfBeatPoint::Disconnect` — erase other from both link lists."""
        if other in nxt[this]:
            nxt[this].remove(other)
        if other in prv[this]:
            prv[this].remove(other)

    def disconnect_all_next(a: int) -> None:
        for p in list(nxt[a]):
            disconnect(p, a)
        nxt[a].clear()

    def disconnect_all_prev(a: int) -> None:
        for p in list(prv[a]):
            disconnect(p, a)
        prv[a].clear()

    def rebuild_chain(cid: int) -> None:
        ch = chains.get(cid)
        if ch is None:
            return
        ch.clear()
        for p in range(n):
            if chain_id[p] == cid and active[p]:
                ch.append(index[p])
        ch.sort()

    processed = 0
    for a, b in zip(ci.tolist(), cj.tolist()):
        processed += 1
        if not (active[a] and active[b]):
            continue                     # skips the coverage check, like the DLL
        ca, cb = chain_id[a], chain_id[b]
        connect = False
        if ca == -1:
            if cb == -1:
                # both unassigned → new chain
                chain_id[a] = next_id
                chain_id[b] = next_id
                chains[next_id] = sorted([index[a], index[b]])
                next_id += 1
                assigned_count += 2
                connect = True
            elif index[a] < earliest(b):
                # A prepends itself to B's chain
                deactivate_reach(b, prv)
                disconnect_all_prev(b)
                chain_id[a] = cb
                assigned_count += 1
                rebuild_chain(cb)
                connect = True
        elif cb == -1:
            if latest(a) < index[b]:
                # B appends itself to A's chain
                deactivate_reach(a, nxt)
                disconnect_all_next(a)
                chain_id[b] = ca
                assigned_count += 1
                rebuild_chain(ca)
                connect = True
        elif ca != cb:
            ea, la = earliest(a), latest(a)
            eb, lb = earliest(b), latest(b)
            if ea <= eb and la >= lb:
                # B's span inside A's — B's chain is dropped entirely.
                deactivate_reach(b, prv)
                disconnect_all_prev(b)
                deactivate_reach(b, nxt)
                disconnect_all_next(b)
                if active[b]:
                    active[b] = False
                    active_count -= 1
                    if chain_id[b] != -1:
                        assigned_count -= 1
                chains.pop(cb, None)
            elif ea >= eb and la <= lb:
                deactivate_reach(a, prv)
                disconnect_all_prev(a)
                deactivate_reach(a, nxt)
                disconnect_all_next(a)
                if active[a]:
                    active[a] = False
                    active_count -= 1
                    if chain_id[a] != -1:
                        assigned_count -= 1
                chains.pop(ca, None)
            elif ea < eb:
                # merge B's chain into A's
                deactivate_reach(a, nxt)
                disconnect_all_next(a)
                deactivate_reach(b, prv)
                disconnect_all_prev(b)
                keep_id = chain_id[a]
                for p in range(n):
                    if chain_id[p] == cb:
                        chain_id[p] = keep_id
                chains.pop(cb, None)
                rebuild_chain(keep_id)
                connect = True

        if connect:
            # Connect(A, B); Connect(B, A) — A is always the earlier point.
            nxt[a].append(b)             # index[b] > index[a]
            prv[b].append(a)

        if active_count != 0:
            coverage = assigned_count / active_count
            if coverage >= 0.8:
                break

    trace.processed = processed
    trace.coverage = (assigned_count / active_count) if active_count else 0.0
    trace.chains_total = len(chains)

    # ── phase 8: chain scoring ──────────────────────────────────────────
    spans: list[int] = []
    items_list: list[list[int]] = []
    means: list[float] = []
    max_span = -1
    min_mc = max_mc = 0.0
    for cid in sorted(chains):
        items = chains[cid]
        span = pos[items[-1]] - pos[items[0]]
        s = np.float32(0.0)
        for ix in items:
            s = np.float32(s + np.float32(conf[ix]))
        mean_c = _f32(s / np.float32(len(items))) if items else 0.0
        spans.append(span)
        items_list.append(items)
        means.append(mean_c)
        if max_span == -1:
            max_span, min_mc, max_mc = span, mean_c, mean_c
        else:
            max_span = max(max_span, span)
            max_mc = _f32(max(max_mc, mean_c))
            min_mc = _f32(min(min_mc, mean_c))
    if max_span == 0:
        max_span = 1
    span_f = _f32(max_span)
    conf_range = _f32(max_mc - min_mc)
    if conf_range == 0.0:
        conf_range = 1.0

    scored = []
    for span, mc, items in zip(spans, means, items_list):
        weight = _f32(_f32(_f32(span) / span_f) * _f32(_f32(mc - min_mc) / conf_range))
        # diagnostics: this chain's own ioI (what it would report if it won)
        sd = sn = 0
        prev = None
        for ix in items:
            cur = pos[ix]
            if prev is not None:
                d = cur - prev
                sd += d
                sn += _u32_sat(float(d) / float(interval) + 0.5)
            prev = cur
        own = _f32(float(sd) / float(sn)) if sn else interval
        scored.append((weight, items))
        trace.chain_details.append((weight, len(items), span, mc, own))
    scored.sort(key=lambda t: t[0], reverse=True)   # stable, like std::list::sort

    # ── phase 9: overlap suppression ────────────────────────────────────
    kept_chains: list[list[int]] = []
    for _, items in scored:
        dropped = False
        for kitems in kept_chains:
            if not (kitems[-1] < items[0] or items[-1] < kitems[0]):
                dropped = True
                break
        if not dropped:
            kept_chains.append(items)

    # ── phase 10: reject short chains (first/last index span ≤ 31) ──────
    kept_chains = [items for items in kept_chains
                   if items[-1] - items[0] > CHAIN_MIN_INDEX_SPAN]

    trace.chains_kept = len(kept_chains)

    # ── phase 11: emit + final interval from the best chain ─────────────
    emitted: list[int] = []
    for items in kept_chains:
        emitted.extend(pos[ix] for ix in items)

    io_i = interval
    if kept_chains:
        items = kept_chains[0]
        sum_d = 0
        sum_n = 0
        prev = None
        for ix in items:
            cur = pos[ix]
            if prev is not None:
                d = cur - prev
                sum_d += d
                sum_n += _u32_sat(float(d) / float(interval) + 0.5)
            prev = cur
        trace.best_members = len(items)
        if sum_n != 0:
            io_i = _f32(float(sum_d) / float(sum_n))
    trace.io_i = io_i
    return io_i, emitted


def _reestimate_interval(points: list[int], unit0: float) -> float:
    """The BuildBeatPointsAlt / phase-5 spacing fit (re/17 §4.2).

    Consecutive spacings that are 1-3 units of ``unit0`` within ±5 % vote;
    the mean of ``d/n`` wins, else ``unit0``.
    """
    lo = int(_f32(_f32(unit0 * np.float32(-0.05)) + unit0))
    hi = int(_f32(_f32(unit0 * np.float32(0.05)) + unit0))
    total = 0
    cnt = 0
    prev = None
    for cur in points:
        if prev is not None:
            d = cur - prev
            m = _u32_sat(float(d) / float(unit0) + 0.5)
            if 1 <= m <= 3:
                q = d // m
                if lo <= q <= hi:
                    total += q
                    cnt += 1
        prev = cur
    if cnt != 0:
        return _f32(_f32(total) / _f32(cnt))
    return unit0


def beat_chain_bpm(spec_thumb: np.ndarray, onsets8: list[int],
                   bpm_prior: float, sample_rate: int,
                   trace: ChainTrace | None = None) -> float:
    """`CPfBeatBuilder::BuildBeatPointsAlt` (re/17 §4.2).

    ``onsets8`` is ``BuildImpactPoints(peakThumb, 8)`` — the DLL re-detects
    onsets with divisor 8 for the chain stage. The refined interval comes
    back from `ClusterFilterPoints` and the BPM is
    ``60 / (8·ioI·0.25 / sr) = 30·sr / ioI``.
    """
    if trace is None:
        trace = ChainTrace()
    unit0 = sam_per_meas(bpm_prior, sample_rate) >> 3
    v9 = _f32(unit0)
    trace.unit0 = v9

    interval = _reestimate_interval(onsets8, v9)
    trace.interval_est = interval

    io_i, _emitted = _cluster_filter_points(spec_thumb, interval, onsets8, trace)

    if sample_rate != 0:
        p = _f32(_f32(_f32(8.0 * io_i) * 0.25) / _f32(sample_rate))
        if p != 0.0:
            return _f32(60.0 / p)
    return _f32(bpm_prior)


# ─────────────────────────────────────────────────────────────────────────────
# Tempo estimation (spec §5, §6, §7)
# ─────────────────────────────────────────────────────────────────────────────


def _phase_histograms(points: np.ndarray, spms: np.ndarray) -> np.ndarray:
    """Bin onset phases for every candidate tempo at once.

    For one tempo the original walks each onset ``p`` with ``v10 = p % spm`` and
    increments every bin ``j`` satisfying ``j*v8 < v10 < v7 + j*v8`` (with
    ``v7 = spm//10``, ``v8 = spm//40``), starting from ``max(0, v10-v7)//v8``.

    That window is only ``v7/v8 == 4`` wide, so the valid bins for a point are
    ``[max(0, (v10-v7)//v8 + 1), min(39, (v10-1)//v8)]``. Reshaping that range
    into a scatter-add lets numpy do all tempos in one pass.
    """
    n_bpm = spms.size
    v7 = spms // 10                      # (B,)
    v8 = spms // 40                      # (B,)
    # A point can touch at most `v7/v8 + 1` bins (~5), so this bounds the scatter.
    kmax = int((v7 // v8).max()) + 2 if n_bpm else 0
    hist = np.zeros((n_bpm, PHASE_BINS), dtype=np.int32)
    if kmax <= 0 or points.size == 0:
        return hist

    v10 = points[None, :] % spms[:, None]            # (B, P)
    hi = np.minimum(PHASE_BINS - 1, (v10 - 1) // v8[:, None])
    lo = np.maximum(0, (v10 - v7[:, None]) // v8[:, None] + 1)
    cnt = np.clip(hi - lo + 1, 0, kmax)              # (B, P)

    offs = np.arange(kmax)
    cols = lo[:, :, None] + offs[None, None, :]      # (B, P, K)
    mask = offs[None, None, :] < cnt[:, :, None]
    rows = np.broadcast_to(np.arange(n_bpm)[:, None, None], cols.shape)

    flat = (rows[mask] * PHASE_BINS + cols[mask]).astype(np.int64)
    hist += np.bincount(flat, minlength=n_bpm * PHASE_BINS).reshape(n_bpm, PHASE_BINS)
    return hist


def detect_bpm_window(points, start: int, length: int, divisor: int, sample_rate: int) -> float:
    """`DetectBpmWindow` — phase-histogram scan over 82.0 … 163.9 BPM.

    Returns the tempo whose onset phases pile up hardest, or 0.0 when
    undecidable. This is *not* autocorrelation.
    """
    arr = np.asarray(points, dtype=np.int64)
    sel = arr[(arr > start) & (arr < start + length)][:512]
    if sel.size <= 1:
        return 0.0

    bpms = (BPM_MIN + 0.1 * np.arange(BPM_STEPS, dtype=np.float64)).astype(np.float32)
    meas_per_sec = (np.float32(0.25) * bpms) / np.float32(60.0)
    spm = np.where(meas_per_sec > 0, np.float32(sample_rate) / meas_per_sec, 0).astype(np.int64)
    spm = spm // divisor

    # The original breaks out of the scan as soon as spm == 0 or spm//40 == 0;
    # spm decreases monotonically with bpm, so only a prefix is ever evaluated.
    dead = np.flatnonzero((spm == 0) | (spm // 40 == 0))
    cut = int(dead[0]) if dead.size else spm.size
    if cut == 0:
        return 0.0
    spm = spm[:cut]

    hist = _phase_histograms(sel, spm)
    counts = hist.max(axis=1)                        # best bin per candidate tempo
    best = int(np.argmax(counts))                    # first maximum, like the original's `>`
    if counts[best] == 0:
        return 0.0
    return float(bpms[best])


def detect_base_bpm(points, total_samples: int, sample_rate: int) -> float:
    """`DetectBaseBpm` — upper median of up to 18 window/resolution estimates."""
    if total_samples <= 0:
        return 0.0
    step = total_samples // 10
    estimates = []
    for i in range(1, 10):
        for divisor in (4, 8):
            v = detect_bpm_window(points, step * i, WINDOW_SAMPLES, divisor, sample_rate)
            if v != 0.0:
                estimates.append(v)
    if not estimates:
        return 0.0
    estimates.sort()
    return estimates[len(estimates) // 2]            # upper median


def fold_up_bpm(points, bpm: float, sample_rate: int) -> float:
    """`FoldUpBpm` — double a low tempo when most onsets sit ~one bar apart.

    Only applies when ``bpm <= 100``. Intervals are compared against
    ``[0.8, 1.2] · (samplesPerMeasure / 8)``; above 53 % in range the detected
    "beat" is really a bar.
    """
    if not bpm <= 100.0:
        return bpm

    spm = sam_per_meas(bpm, sample_rate)
    base = spm >> 3
    v6 = ((base * 0xCCCCCCCD) >> 34)                 # ~= base / 5
    lo, hi = base - v6, v6 + base

    arr = np.asarray(points, dtype=np.int64)
    if arr.size < 1:
        return bpm
    deltas = np.diff(arr) if arr.size > 1 else np.zeros(0, dtype=np.int64)
    if deltas.size == 0:
        return bpm
    in_range = int(np.count_nonzero((deltas > lo) & (deltas < hi)))
    if (in_range / deltas.size) * 100.0 > 53.0:
        return bpm * 2.0
    return bpm


def short_clip_bpm_update(sample_count: int, sample_rate: int, bpm: float) -> float:
    """`ShortClipBpmUpdate` — clips under 30 s forced into [82, 164.01].

    A duration-driven override from ``240 / seconds``; a no-op for longer clips.
    """
    if sample_rate == 0 or not (30 * sample_rate > sample_count):
        return bpm
    secs = sample_count / sample_rate
    if secs == 0.0:
        return 0.0

    upper = 164.0099945068359
    b = 240.0 / secs
    if b >= 82.0:
        while b > upper:
            b *= 0.5
    else:
        while b < 82.0:
            b *= 2.0
    if b >= 82.0:
        # The original's float literal 164.00999 rounds to the same float32 as 164.01.
        return np.float32(164.01) if b > upper else np.float32(b)
    return np.float32(82.0)


# ─────────────────────────────────────────────────────────────────────────────
# Key detection — a port of libKeyFinder (libKeyFinder/, GPL-3.0)
# ─────────────────────────────────────────────────────────────────────────────


def _lowpass_coefficients(frame_rate: float, corner_hz: float):
    """`LowPassFilterPrivate` — a windowed-sinc FIR built in the frequency domain.

    A brick wall is written into the first half of a 2048-point spectrum, the
    inverse FFT gives the (circular) impulse response, and 161 taps are taken
    wrapping around index 0 before a Hamming window and normalisation.
    """
    delay = LPF_ORDER // 2
    impulse_length = LPF_ORDER + 1
    cutoff = corner_hz / frame_rate
    tau = 0.5 / cutoff

    bins = np.arange(LPF_FFT_SIZE // 2 + 1)
    spectrum = np.where(bins / LPF_FFT_SIZE <= cutoff, tau, 0.0)
    # scipy normalises by n, which is exactly fftw's c2r followed by /n.
    impulse = irfft(spectrum, n=LPF_FFT_SIZE)

    take = (LPF_FFT_SIZE - delay + np.arange(impulse_length)) % LPF_FFT_SIZE
    taps = np.arange(impulse_length)
    hamming = 0.54 - 0.46 * np.cos(2 * np.pi * taps / (impulse_length - 1))
    coefficients = impulse[take] * hamming
    return coefficients, float(coefficients.sum())


def _band_kernels(frame_rate: float):
    """`ChromaTransform` — a constant-Q filterbank expressed over FFT bins."""
    q = DIRECT_SK_STRETCH * (2 ** (1.0 / SEMITONES) - 1)
    offsets = np.zeros(BANDS, dtype=int)
    kernels = []
    for band in range(BANDS):
        centre = BAND_FREQUENCIES[band] * FFT_FRAME_SIZE / frame_rate
        width = centre * q
        begin = centre - width / 2
        offsets[band] = int(np.ceil(begin))
        bins = np.arange(offsets[band], int(np.floor(begin + width)) + 1)
        # A raised-cosine kernel, normalised and scaled by the band frequency.
        weights = 1.0 - np.cos(2 * np.pi * (bins - begin) / width)
        kernels.append(weights / weights.sum() * BAND_FREQUENCIES[band])
    return offsets, kernels


def _chromagram(samples: np.ndarray, frame_rate: float) -> np.ndarray:
    """`SpectrumAnalyser::chromagramOfWholeFrames` — one 72-band vector per hop."""
    count = samples.size
    if count < FFT_FRAME_SIZE:
        return np.zeros((0, BANDS), dtype=np.float64)

    hops = 1 + (count - FFT_FRAME_SIZE) // HOP_SIZE
    offsets, kernels = _band_kernels(frame_rate)
    taps = np.arange(FFT_FRAME_SIZE)
    window = (0.42 - 0.5 * np.cos(2 * np.pi * taps / (FFT_FRAME_SIZE - 1))
              + 0.08 * np.cos(4 * np.pi * taps / (FFT_FRAME_SIZE - 1)))

    out = np.zeros((hops, BANDS), dtype=np.float64)
    for hop in range(hops):
        start = hop * HOP_SIZE
        magnitude = np.abs(rfft(samples[start:start + FFT_FRAME_SIZE] * window))
        for band in range(BANDS):
            offset = offsets[band]
            out[hop, band] = np.dot(magnitude[offset:offset + kernels[band].size],
                                    kernels[band])
    return out


def _tone_profile(profile) -> np.ndarray:
    """`toneProfileMajor`/`Minor` — the 12-tone profile replicated per octave."""
    return np.concatenate([OCTAVE_WEIGHTS[o] * np.asarray(profile) for o in range(OCTAVES)])


def _cosine_similarity(chroma: np.ndarray, tone: np.ndarray, offset: int) -> float:
    """`ToneProfile::cosineSimilarity` — chroma against a rotated tone profile.

    The profile's tonic sits three semitones above index 0 (the A-to-C shift in
    the constructor) and ``offset`` walks it left, so chroma band ``o*12 + j``
    pairs with profile element ``(3 - offset + j) mod 12``.
    """
    base = (3 - offset) % SEMITONES
    intersection = 0.0
    profile_norm = 0.0
    input_norm = 0.0
    for octave in range(OCTAVES):
        rows = octave * SEMITONES + (base + np.arange(SEMITONES)) % SEMITONES
        p = tone[rows]
        x = chroma[octave * SEMITONES:(octave + 1) * SEMITONES]
        intersection += float(np.dot(x, p))
        profile_norm += float(np.dot(p, p))
        input_norm += float(np.dot(x, x))
    if profile_norm > 0.0 and input_norm > 0.0:
        return intersection / (math.sqrt(profile_norm) * math.sqrt(input_norm))
    return 0.0


def _classify(chroma: np.ndarray) -> str:
    """`KeyClassifier::classify` — best cosine over 24 keys, defaulting to silence."""
    major = _tone_profile(MAJOR_TONE_PROFILE)
    minor = _tone_profile(MINOR_TONE_PROFILE)
    scores = [0.0] * 24
    for i in range(SEMITONES):
        scores[i * 2] = _cosine_similarity(chroma, major, i)
        scores[i * 2 + 1] = _cosine_similarity(chroma, minor, i)

    # The silence profile is all zeroes, so its similarity is 0 and every real
    # key has to beat it outright.
    best_score = 0.0
    best = 24
    for i in range(24):
        if scores[i] > best_score:
            best_score = scores[i]
            best = i
    return KEY_NAMES[best]


def detect_key(samples: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE) -> str:
    """`KeyFinder::keyOfAudio` — the whole libKeyFinder chain.

    Low-pass and decimate to ~4.4 kHz, build a 72-band chromagram of 16384-point
    frames, collapse it to one vector and match it against rotated major/minor
    tone profiles. See docs/02-algorithm-spec.md §13.
    """
    audio = np.asarray(samples, dtype=np.float64).ravel()
    rate = float(sample_rate)

    lpf_cutoff = BAND_FREQUENCIES[-1] * 1.012
    downsample_cutoff = BAND_FREQUENCIES[-1] * 1.10
    factor = int(math.floor(rate / 2.0 / downsample_cutoff))
    factor = max(1, factor)

    # `preprocess` slices the tail remainder off before filtering, and the
    # `finalChromagram` flush that would pick it up analyses a local that is
    # then discarded — so those samples are simply dropped.
    excess = audio.size % factor
    if excess:
        audio = audio[:audio.size - excess]

    coefficients, gain = _lowpass_coefficients(rate, lpf_cutoff)
    # `CIIRFilt`-style decimating FIR: the taps are symmetric about `delay`, and
    # the library writes every `factor`-th output then decimates by `factor`.
    convolution = np.convolve(audio, coefficients / gain)
    # Only `outSample` in [0, sampleCount) is ever written, so the decimated
    # stream is ceil(N / factor) long.
    kept = -(-audio.size // factor)
    decimated = convolution[LPF_ORDER // 2::factor][:kept]
    rate /= factor

    # `progressiveChromagram` then `finalChromagram`: analyse what we have,
    # zero-pad the tail to a whole number of hops, and analyse that too.
    first = _chromagram(decimated, rate)
    tail = decimated[HOP_SIZE * first.shape[0]:]

    padded_hops = math.ceil(tail.size / HOP_SIZE)
    final_length = FFT_FRAME_SIZE + (padded_hops - 1) * HOP_SIZE
    if final_length > tail.size:
        tail = np.concatenate([tail, np.zeros(final_length - tail.size)])
    second = _chromagram(tail, rate)

    hops = [h for h in (first, second) if h.shape[0] > 0]
    if not hops:
        return KEY_NAMES[24]
    collapsed = np.concatenate(hops).mean(axis=0)
    return _classify(collapsed)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Trace:
    thumbnail_elements: int = 0
    peak_hold_decay_div: int = 0
    onset_count: int = 0
    classic_onset_count: int = 0
    #: Onsets from BuildImpactPoints(peakThumb, 8) — the chain's input.
    chain_onset_count: int = 0
    #: Tempo from the 0.1-grid phase-histogram search (spec §5).
    base_bpm: float = 0.0
    #: ``base_bpm`` after the half-tempo fix (spec §6).
    folded_bpm: float = 0.0
    short_clip_applied: bool = False
    #: Beat-chain stage internals (spec §8).
    chain: ChainTrace = field(default_factory=ChainTrace)
    #: Key reported by the libKeyFinder port (spec §13).
    key_detected: str = "not requested"


@dataclass
class Analysis:
    name: str = ""
    bpm: float = 0.0
    #: libKeyFinder key name, e.g. ``"G major"``; ``None`` when --nokey was given.
    key: str | None = None
    sample_count: int = 0
    sample_rate: int = DEFAULT_SAMPLE_RATE
    duration: float = 0.0
    trace: Trace = field(default_factory=Trace)


def analyze_samples(
    samples: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    find_key: bool = False,
) -> Analysis:
    """Run the recovered pipeline (docs/02-algorithm-spec.md §0).

    Mirrors `CProfileBuilder::BuildProfile` stage for stage: spectral
    thumbnail → peak-hold → onsets(÷4) → base BPM → half-tempo fold (from
    the smoothed *note-detector* thumbnail) → short-clip rule →
    `BuildBeatPointsAlt` chain refinement, which produces the final BPM.
    ``find_key`` additionally runs the libKeyFinder port (spec §13).
    """
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    total = int(samples.size)

    # ── ② spectral thumbnail (v30) ──────────────────────────────────────
    spec = spec_thumbnail(samples)
    # ── ③ peak-hold envelope (v33) ──────────────────────────────────────
    peak = peak_hold(spec, sample_rate)
    # ── ④ onsets for the tempo histogram (divisor 4) ────────────────────
    onsets = build_impact_points(peak, total, 4, sample_rate)
    # ── ⑤ initial tempo: median of 9 windows × {÷4, ÷8} ─────────────────
    base = detect_base_bpm(onsets, total, sample_rate)

    # ── ⑥ half-tempo fold — onsets from the smoothed note thumbnail ─────
    # The DLL smooths a copy of the *note-detector* thumbnail (filtered
    # waveform peaks), not the spectral one; see note_peak_thumbnail.
    note = note_peak_thumbnail(samples, sample_rate)
    classic = build_impact_points_classic(smoothed(note), total, 4, sample_rate)
    folded = fold_up_bpm(classic, base, sample_rate)

    # ── ⑦ short-clip override — a prior, not a verdict: the chain still
    # runs and can confirm (or, with content, refine) it below.
    prior = float(short_clip_bpm_update(total, sample_rate, folded))
    is_short = sample_rate != 0 and 30 * sample_rate > total

    # ── ⑧ final BPM: BuildBeatPointsAlt → ClusterFilterPoints ───────────
    onsets8 = build_impact_points(peak, total, 8, sample_rate)
    chain = ChainTrace()
    bpm = beat_chain_bpm(spec, onsets8, prior, sample_rate, chain)

    # ── ⑩ key: the libKeyFinder chain (spec §13) ────────────────────────
    key = detect_key(samples, sample_rate) if find_key else None

    return Analysis(
        bpm=float(bpm),
        key=key,
        sample_count=total,
        sample_rate=sample_rate,
        duration=(total / sample_rate) if sample_rate else 0.0,
        trace=Trace(
            thumbnail_elements=int(spec.size),
            peak_hold_decay_div=int((sam_per_meas(164.0, sample_rate) // FFT_SIZE) >> 4),
            onset_count=len(onsets),
            classic_onset_count=len(classic),
            chain_onset_count=len(onsets8),
            base_bpm=float(base),
            folded_bpm=float(folded),
            short_clip_applied=is_short,
            chain=chain,
            key_detected=key if key is not None else "not requested",
        ),
    )


def load_audio(path: Path, sample_rate: int) -> np.ndarray:
    """Load ``path`` as mono float32 at ``sample_rate`` using librosa.

    Multichannel sources are folded with a plain channel **sum** (L+R), the
    convention the original analyzer's PCM reader uses — not an average.  The
    distinction is observable: the impact-point ladder thresholds are a
    discrete ÷1.2 sequence, so a ÷2 scale shift moves the crossing to a
    different step and changes the onset set (validated against the Windows
    oracle on 海边城.mp3: sum → 160.03 vs mean → 160.18, oracle 160.04).

    note: the Rust CLI decodes with ffmpeg instead, so the two implementations
    do not always see identical PCM for a source that needs resampling (librosa
    uses soxr, ffmpeg its own resampler). The DSP is the same; see
    docs/04-testing.md for how far apart the results land.
    """
    y, _ = librosa.load(str(path), sr=sample_rate, mono=False, dtype=np.float32)
    if y.ndim > 1:
        y = y.sum(axis=0)
    y = np.ascontiguousarray(y, dtype=np.float32)
    if y.size < FFT_SIZE:
        raise ValueError(f"no decodable audio in {path}")
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)


def analyze_file(path: Path, sample_rate: int, find_key: bool = False) -> Analysis:
    samples = load_audio(path, sample_rate)
    result = analyze_samples(samples, sample_rate, find_key)
    result.name = path.name
    return result


def detail_lines(a: Analysis) -> str:
    t = a.trace
    c = t.chain
    return (
        f"  file            : {a.name}\n"
        f"  samples         : {a.sample_count} ({a.duration:.2f} s @ {a.sample_rate} Hz)\n"
        f"  thumbnail elems : {t.thumbnail_elements}\n"
        f"  peak-hold decay : {t.peak_hold_decay_div}\n"
        f"  onsets (slope)  : {t.onset_count}\n"
        f"  onsets (classic): {t.classic_onset_count}\n"
        f"  onsets (chain)  : {t.chain_onset_count}\n"
        f"  base BPM (0.1)  : {t.base_bpm:.2f}\n"
        f"  after fold-up   : {t.folded_bpm:.2f}\n"
        f"  short-clip rule : {'applied' if t.short_clip_applied else 'n/a'}\n"
        f"  chain unit0     : {c.unit0:.2f}\n"
        f"  chain interval  : {c.interval_est:.4f}\n"
        f"  chain conns     : {c.processed}/{c.candidates} (coverage {c.coverage:.2f})\n"
        f"  beat chain      : {c.io_i:.4f} samples/eighth\n"
        f"                  : {c.chains_kept} chains kept, best {c.best_members} pts\n"
        f"  final BPM       : {a.bpm:.2f}\n"
        f"  key (findkey)   : {t.key_detected}\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def collect_audio_files(root: Path, recursive: bool) -> list[Path]:
    walker = root.rglob("*") if recursive else root.iterdir()
    files = [
        p
        for p in walker
        if p.is_file()
        and p.suffix.lower().lstrip(".") in AUDIO_EXTENSIONS
        # macOS writes `._name` sidecar files next to media; they are not audio.
        and not p.name.startswith("._")
    ]
    return sorted(files)


def existing_path(value: str) -> Path:
    """argparse type that rejects a PATH which does not exist.

    This makes a typo a *usage* error (exit 2), matching the Rust binary, rather
    than a runtime failure (exit 1).
    """
    p = Path(value)
    if not p.exists():
        raise argparse.ArgumentTypeError(f"no such file or directory: {value}")
    return p


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rba.py",
        description="BPM & key analyzer — Python reference implementation of the "
        "MixMeister BPM Analyzer core and of the libKeyFinder key detector.",
        epilog=(
            "OUTPUT:\n"
            "  A single file names the file, its tempo (2 decimals) and its key.\n"
            "  A directory prints a FILE/BPM/KEY table, then a note for anything that failed.\n"
            "  --nokey drops the key column and skips that pass.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "path", metavar="PATH", type=existing_path, help="an audio file, or a directory to scan"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="print pipeline detail to stderr")
    parser.add_argument(
        "-r", "--recursive", action="store_true", help="recurse into subdirectories when PATH is a directory"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        metavar="HZ",
        help="analysis sample rate (default: %(default)s)",
    )
    parser.add_argument(
        "--nokey", action="store_true", help="skip key detection (BPM only)"
    )
    parser.add_argument("-V", "--version", action="version", version=f"rba.py {__version__}")
    return parser


def display_width(text: str) -> int:
    """Terminal columns needed to render ``text``.

    Uses the stdlib's East Asian width table, so CJK file names stay aligned
    without pulling in a dependency.
    """
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def render_table(rows: list[tuple[str, str, str]], show_key: bool) -> str:
    """Render the ``FILE / BPM / KEY`` table as plain aligned columns.

    The name column is sized in *display* columns (not characters) so CJK file
    names line up; the other columns are ASCII. Trailing padding is stripped.
    """
    if not rows:
        return ""
    name_w = max(display_width("FILE"), *(display_width(n) for n, _, _ in rows))
    bpm_w = max(len("BPM"), *(len(b) for _, b, _ in rows))
    key_w = max(len("KEY"), *(len(k) for _, _, k in rows))

    def line(name: str, bpm: str, key: str) -> str:
        out = f"{name}{' ' * (name_w - display_width(name))}  {bpm:>{bpm_w}}"
        if show_key:
            out += f"  {key:<{key_w}}"
        return out.rstrip()

    rule = "-" * name_w + "  " + "-" * bpm_w
    if show_key:
        rule += "  " + "-" * key_w
    out = [line("FILE", "BPM", "KEY"), rule]
    out.extend(line(*row) for row in rows)
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    path: Path = args.path

    if path.is_file():
        try:
            result = analyze_file(path, args.sample_rate, not args.nokey)
        except Exception as exc:  # noqa: BLE001 - surfaced as a clean CLI error
            print(f"rba.py: {exc}", file=sys.stderr)
            return EXIT_RUNTIME
        tail = f"\tKEY: {result.key}" if result.key is not None else ""
        print(f"{result.name}\tBPM: {result.bpm:.2f}{tail}")
        print(detail_lines(result), file=sys.stderr, end="")
        return EXIT_OK

    files = collect_audio_files(path, args.recursive)
    if not files:
        print(f"rba.py: no audio files found in {path}", file=sys.stderr)
        return EXIT_RUNTIME

    find_key = not args.nokey
    rows: list[tuple[str, str, str]] = []
    failures: list[tuple[str, str]] = []
    # `disable=None` silences the bar when stderr is not a TTY, matching the
    # Rust CLI. It writes to stderr, so stdout stays a clean table.
    bar = tqdm(files, desc="analysing", unit="file", disable=None)
    for f in bar:
        bar.set_postfix_str(f.name, refresh=False)
        try:
            result = analyze_file(f, args.sample_rate, find_key)
        except Exception as exc:  # noqa: BLE001 - one bad file must not abort the run
            rows.append((f.name, "FAILED", "-"))
            failures.append((f.name, str(exc)))
            continue
        rows.append((result.name, f"{result.bpm:.2f}", result.key or "-"))
        if args.verbose:
            # Go through tqdm so the bar is cleared before each block.
            bar.write(detail_lines(result).rstrip("\n"), file=sys.stderr)

    print(render_table(rows, find_key))

    if failures:
        print(f"\n{len(failures)} file{'s' if len(failures) != 1 else ''} could not be analyzed:")
        for name, message in failures:
            print(f"  {name} — {message}")

    return EXIT_RUNTIME if len(failures) == len(files) else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
