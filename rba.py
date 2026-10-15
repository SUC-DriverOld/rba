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
from tqdm import tqdm
from scipy.fft import irfft, rfft

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


def spec_thumbnail(samples: np.ndarray) -> np.ndarray:
    """One element per 256-sample frame: sum of |FFT bins 1..127| of a Hamming window.

    DC (bin 0) and Nyquist (bin 128) are excluded, exactly as ``GetSpecEnergy``
    does for N=256.
    """
    n = FFT_SIZE
    count = samples.size // n
    if count == 0:
        return np.zeros(0, dtype=np.float32)

    frames = samples[: count * n].reshape(count, n)
    # w[i] = cos(2*pi*i/N) * -0.46 + 0.54
    idx = np.arange(n, dtype=np.float64)
    window = np.cos(2.0 * np.pi * idx / n) * -0.46 + 0.54

    mag = np.abs(rfft(frames * window, axis=1))
    return mag[:, 1 : n // 2].sum(axis=1).astype(np.float32)


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


def refine_bpm(onsets, bpm: float, sample_rate: int) -> tuple[float, int]:
    """Refine a coarse tempo past the 0.1 grid, from the onset positions.

    Why this exists
    ---------------
    The recovered algorithm scans BPM on a fixed 0.1 grid (spec §5), so every
    value it can produce ends in a zero -- ``159.90``, ``119.80``, ... That grid
    is an artefact of how the original binary searched, not a property of the
    music. This stage leaves the search (and therefore every decision the
    original made) untouched and only re-derives the *period* from the onsets
    that actually landed on the coarse beat grid.

    Method
    ------
    Anchor on the coarse beat period ``T0 = 60*sr/bpm``. Give each onset a beat
    index ``k = round(p / T0)`` and weight each one by how close it sits to that
    grid, tapering to zero at a quarter beat -- that rejects syncopated off-beats
    and half-beats without a hard cut-off. Then fit ``p ~= phase + k*T`` by
    weighted least squares, and iterate: once the first pass tightens the
    period, beats that had drifted outside the gate come back in.

    Returns ``(bpm, kept_onsets)``, falling back to the coarse value when the fit
    is unsupported or moves more than 5 %.

    **This is an enhancement, not part of the reverse-engineered algorithm.**
    See docs/02-algorithm-spec.md §11.
    """
    coarse = float(bpm)
    if not coarse > 0.0 or sample_rate == 0 or len(onsets) < 8:
        return coarse, 0

    p = np.asarray(onsets, dtype=np.float64)
    period = sample_rate * 60.0 / coarse
    kept = 0

    for _ in range(4):
        if not np.isfinite(period) or period <= 0.0:
            return coarse, kept
        tol = period / 4.0
        k = np.round(p / period)
        # Tapered weight rather than a hard gate: an onset sitting exactly on the
        # boundary would otherwise flip the fit, and the two implementations'
        # onset positions can differ by a sample (rustfft f32 vs scipy). See
        # src/detect.rs.
        u = np.abs(p - k * period) / tol
        keep = u < 1.0
        w = np.where(keep, (1.0 - u * u) ** 2, 0.0)
        kept = int(np.count_nonzero(keep))
        if kept < 4:
            return coarse, kept
        n = float(w.sum())
        sk = float((w * k).sum())
        sp = float((w * p).sum())
        skk = float((w * k * k).sum())
        skp = float((w * k * p).sum())
        denom = n * skk - sk * sk
        if abs(denom) < 1e-9:
            return coarse, kept
        period = (n * skp - sk * sp) / denom

    refined = sample_rate * 60.0 / period
    if not np.isfinite(refined) or abs(refined - coarse) / coarse > 0.05:
        return coarse, kept
    return float(refined), kept


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
    #: Tempo from the 0.1-grid phase-histogram search (spec §5).
    base_bpm: float = 0.0
    #: ``base_bpm`` after the half-tempo fix (spec §6).
    folded_bpm: float = 0.0
    #: Period refinement result (spec §11); ``None`` when it did not run.
    refined_bpm: float | None = None
    #: How many onsets the refinement fit.
    refine_onsets: int = 0
    short_clip_applied: bool = False
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

    ``find_key`` additionally runs the libKeyFinder port (spec §13).
    """
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    total = int(samples.size)

    spec = spec_thumbnail(samples)
    peak = peak_hold(spec, sample_rate)
    onsets = build_impact_points(peak, total, 4, sample_rate)
    base = detect_base_bpm(onsets, total, sample_rate)

    smooth = smoothed(spec)
    classic = build_impact_points_classic(smooth, total, 4, sample_rate)
    folded = fold_up_bpm(classic, base, sample_rate)

    # A clip under 30 s is deliberately governed by its *duration* (spec §7), so
    # refining it against onsets would fight an intentional rule; only
    # content-derived results get refined.
    is_short = sample_rate != 0 and 30 * sample_rate > total
    if is_short:
        bpm = float(short_clip_bpm_update(total, sample_rate, folded))
        refined, kept = None, 0
    else:
        bpm, kept = refine_bpm(onsets, folded, sample_rate)
        refined = bpm

    # ── ⑩ key: the libKeyFinder chain (spec §13) ──────────────────────
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
            base_bpm=float(base),
            classic_onset_count=len(classic),
            folded_bpm=float(folded),
            refined_bpm=refined,
            refine_onsets=kept,
            short_clip_applied=is_short,
            key_detected=key if key is not None else "not requested",
        ),
    )


def load_audio(path: Path, sample_rate: int) -> np.ndarray:
    """Load ``path`` as mono float32 at ``sample_rate`` using librosa.

    note: the Rust CLI decodes with ffmpeg instead, so the two implementations
    do not always see identical PCM for a source that needs resampling (librosa
    uses soxr, ffmpeg its own resampler) or a multichannel source. The DSP is
    the same; see docs/04-testing.md for how far apart the results land.
    """
    y, _ = librosa.load(str(path), sr=sample_rate, mono=True, dtype=np.float32)
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
    refinement = (
        f"{t.refined_bpm:.4f} (fit {t.refine_onsets} onsets)"
        if t.refined_bpm is not None
        else "skipped"
    )
    return (
        f"  file            : {a.name}\n"
        f"  samples         : {a.sample_count} ({a.duration:.2f} s @ {a.sample_rate} Hz)\n"
        f"  thumbnail elems : {t.thumbnail_elements}\n"
        f"  peak-hold decay : {t.peak_hold_decay_div}\n"
        f"  onsets (slope)  : {t.onset_count}\n"
        f"  onsets (classic): {t.classic_onset_count}\n"
        f"  base BPM (0.1)  : {t.base_bpm:.2f}\n"
        f"  after fold-up   : {t.folded_bpm:.2f}\n"
        f"  refined         : {refinement}\n"
        f"  short-clip rule : {'applied' if t.short_clip_applied else 'n/a'}\n"
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
