#!/usr/bin/env python3
"""rba — Rust BPM Analyzer (Python reference implementation).

A single-file Python port of the same reverse-engineered algorithm that
``src/`` implements in Rust. It is deliberately a *reference* implementation:
readable, vectorised with numpy/scipy where it matters, and byte-for-byte
comparable with the Rust binary.

    rba.py <file>        → print that file's BPM
    rba.py <directory>   → print a table of every audio file inside

requirements:
    librosa>=1.0, numpy>=1.24, scipy>=1.10
"""

import argparse
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np
from scipy.fft import rfft

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
    index ``k = round(p / T0)`` and keep those whose residual to that grid is
    under a quarter beat -- that rejects syncopated off-beats and half-beats.
    Then fit ``p ~= phase + k*T`` by least squares, and iterate: once the first
    pass tightens the period, beats that had drifted outside the gate come back
    in.

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
        k = np.round(p / period)
        keep = np.abs(p - k * period) < period / 4.0
        k2, p2 = k[keep], p[keep]
        kept = int(k2.size)
        if k2.size < 4:
            return coarse, kept
        n = float(k2.size)
        sk = float(k2.sum())
        sp = float(p2.sum())
        skk = float((k2 * k2).sum())
        skp = float((k2 * p2).sum())
        denom = n * skk - sk * sk
        if abs(denom) < 1e-9:
            return coarse, kept
        period = (n * skp - sk * sp) / denom

    refined = sample_rate * 60.0 / period
    if not np.isfinite(refined) or abs(refined - coarse) / coarse > 0.05:
        return coarse, kept
    return float(refined), kept


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


@dataclass
class Analysis:
    name: str = ""
    bpm: float = 0.0
    sample_count: int = 0
    sample_rate: int = DEFAULT_SAMPLE_RATE
    duration: float = 0.0
    trace: Trace = field(default_factory=Trace)


def analyze_samples(samples: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE) -> Analysis:
    """Run the recovered pipeline (docs/02-algorithm-spec.md §0)."""
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    total = int(samples.size)

    spec = spec_thumbnail(samples)
    peak = peak_hold(spec, sample_rate)
    onsets = build_impact_points(peak, total, 8, sample_rate)
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

    return Analysis(
        bpm=float(bpm),
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
        ),
    )


def load_audio(path: Path, sample_rate: int) -> np.ndarray:
    """Load ``path`` as mono float32 at ``sample_rate`` using librosa."""
    y, _ = librosa.load(str(path), sr=sample_rate, mono=True, dtype=np.float32)
    if y.size < FFT_SIZE:
        raise ValueError(f"no decodable audio in {path}")
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)


def analyze_file(path: Path, sample_rate: int) -> Analysis:
    samples = load_audio(path, sample_rate)
    result = analyze_samples(samples, sample_rate)
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
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

AUDIO_EXTENSIONS = frozenset(
    """
    wav wave mp3 m4a m4b aac flac ogg oga opus wma aif aiff aifc alac mp4 ape
    wv tta dsf dff caf au amr ac3 dts mka webm mpc spx 3gp ra
    """.split()
)

EXIT_OK, EXIT_RUNTIME = 0, 1


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
        description="BPM analyzer — Python reference implementation of the "
        "MixMeister BPM Analyzer core.",
        epilog=(
            "OUTPUT:\n"
            "  A single file names the file and prints its tempo (2 decimals).\n"
            "  A directory prints a FILE/BPM table, then a note for anything that failed.\n"
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
    parser.add_argument("-V", "--version", action="version", version=f"rba.py {__version__}")
    return parser


def display_width(text: str) -> int:
    """Terminal columns needed to render ``text``.

    Uses the stdlib's East Asian width table, so CJK file names stay aligned
    without pulling in a dependency.
    """
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def render_table(rows: list[tuple[str, str]]) -> str:
    """Render the ``FILE / BPM`` table as plain aligned columns.

    The name column is sized in *display* columns (not characters) so CJK file
    names line up; the BPM column is ASCII digits, where the two agree.
    """
    if not rows:
        return ""
    name_w = max(display_width("FILE"), *(display_width(n) for n, _ in rows))
    bpm_w = max(len("BPM"), *(len(b) for _, b in rows))

    def line(name: str, bpm: str) -> str:
        return f"{name}{' ' * (name_w - display_width(name))}  {bpm:>{bpm_w}}"

    out = [line("FILE", "BPM"), f"{'-' * name_w}  {'-' * bpm_w}"]
    out.extend(line(name, bpm) for name, bpm in rows)
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    path: Path = args.path

    if path.is_file():
        try:
            result = analyze_file(path, args.sample_rate)
        except Exception as exc:  # noqa: BLE001 - surfaced as a clean CLI error
            print(f"rba.py: {exc}", file=sys.stderr)
            return EXIT_RUNTIME
        print(f"{result.name}\tBPM: {result.bpm:.2f}")
        print(detail_lines(result), file=sys.stderr, end="")
        return EXIT_OK

    files = collect_audio_files(path, args.recursive)
    if not files:
        print(f"rba.py: no audio files found in {path}", file=sys.stderr)
        return EXIT_RUNTIME

    rows: list[tuple[str, str]] = []
    failures: list[tuple[str, str]] = []
    for f in files:
        try:
            result = analyze_file(f, args.sample_rate)
        except Exception as exc:  # noqa: BLE001 - one bad file must not abort the run
            rows.append((f.name, "FAILED"))
            failures.append((f.name, str(exc)))
            continue
        rows.append((result.name, f"{result.bpm:.2f}"))
        if args.verbose:
            print(detail_lines(result), file=sys.stderr, end="")

    print(render_table(rows))

    if failures:
        print(f"\n{len(failures)} file{'s' if len(failures) != 1 else ''} could not be analyzed:")
        for name, message in failures:
            print(f"  {name} — {message}")

    return EXIT_RUNTIME if len(failures) == len(files) else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
