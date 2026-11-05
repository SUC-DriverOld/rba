"""Regenerate synthetic audio regression fixtures with independent encoders."""
from pathlib import Path
import array
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
import wave

root = Path(__file__).resolve().parent
ffmpeg = shutil.which("ffmpeg")
if not ffmpeg:
    raise SystemExit("FFmpeg with AAC, libmp3lame, libopus and libvorbis encoders is required")
rate = 44100
samples = array.array("h", (
    int(32767 * (.45 * math.sin(2 * math.pi * 440 * i / rate) * math.exp(-((i / rate) % .5) * 40)
                 + .25 * math.sin(2 * math.pi * 100 * i / rate) * math.exp(-((i / rate) % .5) * 25)))
    for i in range(rate * 12)
))
if sys.byteorder != "little":
    samples.byteswap()
with tempfile.TemporaryDirectory(prefix="rba-fixtures-") as temporary:
    source = Path(temporary) / "source.wav"
    with wave.open(str(source), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(samples.tobytes())
    for name, options in [
        ("vbr-without-xing.mp3", ["-c:a", "libmp3lame", "-q:a", "2", "-write_xing", "0"]),
        ("indexed.mp3", ["-c:a", "libmp3lame", "-t", "1", "-b:a", "128k"]),
        ("aac.m4a", ["-c:a", "aac", "-t", "1"]),
        ("opus.ogg", ["-c:a", "libopus", "-t", "1"]),
        ("vorbis.ogg", ["-c:a", "libvorbis", "-t", "1"]),
    ]:
        target = root / name
        subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-y", "-i", str(source),
                        *options, str(target)], check=True, timeout=20)
        decoded = subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-i", str(target),
                                  "-ac", "1", "-ar", str(rate), "-f", "f32le", "-"],
                                 check=True, capture_output=True, timeout=20)
        print(json.dumps({"file": name, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                          "decodedSamples": len(decoded.stdout) // 4}))

    with wave.open(str(root / "pcm.wav"), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(bytes(1024))

    # Rebase audio page granules, preserving packet bytes and valid Ogg CRCs.
    data = bytearray((root / "vorbis.ogg").read_bytes())
    position = 0
    while position < len(data):
        assert data[position:position + 4] == b"OggS"
        segments = data[position + 26]
        length = 27 + segments + sum(data[position + 27:position + 27 + segments])
        granule = int.from_bytes(data[position + 6:position + 14], "little")
        if 0 < granule < (1 << 64) - 1:
            data[position + 6:position + 14] = (granule + 441000).to_bytes(8, "little")
            data[position + 22:position + 26] = bytes(4)
            checksum = 0
            for byte in data[position:position + length]:
                checksum ^= byte << 24
                for _ in range(8):
                    checksum = ((checksum << 1) ^ 0x04c11db7 if checksum & 0x80000000 else checksum << 1) & 0xffffffff
            data[position + 22:position + 26] = checksum.to_bytes(4, "little")
        position += length
    target = root / "vorbis-offset.ogg"
    target.write_bytes(data)
    decoded = subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-i", str(target), "-ac", "1", "-ar", str(rate), "-f", "f32le", "-"],
                             check=True, capture_output=True, timeout=20)
    print(json.dumps({"file": target.name, "sha256": hashlib.sha256(data).hexdigest(), "decodedSamples": len(decoded.stdout) // 4}))
