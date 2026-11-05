# Synthetic decoder fixtures

These files contain generated percussion tones, with no third-party recordings.
`generate.py` creates 12 seconds of mono 44.1 kHz PCM and encodes it with FFmpeg
and libmp3lame:

- `vbr-without-xing.mp3`: VBR quality 2, Xing writing disabled. Independent
  FFmpeg decoding produces 461 MPEG frames of 1152 PCM samples each (531072
  samples including encoder delay/padding). The previous native decoder
  incorrectly trimmed this stream to its bitrate-based duration estimate.
- `indexed.mp3`: the first second encoded at 128 kbps, with Xing/Info and LAME
  gapless metadata. Its playable length is exactly 44100 samples. Tests derive
  truncated and corrupted copies at runtime without modifying this fixture.
- `aac.m4a` and `opus.ogg`: one second encoded as AAC in MP4 and Opus in Ogg.
  Their edit-list/granule counts include codec timing conventions and must not
  be interpreted as exact decoded PCM sample counts. Opus pre-skip is removed
  by FFmpeg, yielding exactly 44100 samples at the analysis rate.
- `vorbis.ogg` and `vorbis-offset.ogg`: one second of Vorbis, with the latter's
  non-header granules shifted by 441000 ticks and Ogg page CRCs recomputed. The
  positive stream origin yields 44224 decoded samples in both backends. Granule
  positions are not used as exact PCM counts: single-page streams may not expose
  the origin, and FFmpeg's first-window discard differs from Symphonia's trimming.
- `pcm.wav`: 512 silent 16-bit mono frames, used to probe the real WAVE format
  name and derive truncated copies for integrity checks.

Regenerate with `python tests/fixtures/generate.py` when FFmpeg with libmp3lame
is installed. The script prints SHA256 and independent decoded sample counts.
Codec/library version changes can change encoded bytes; review the generated
counts and update expectations deliberately.
