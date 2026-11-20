"""How audio is stored in this repository, and nothing more.

The source corpora ship uncompressed WAV files. Stored as they are they would
not fit in a GitHub repository, so every WAV file is rewritten as a mono
16-bit FLAC file with the sample rate of the source:

- 16-bit PCM input is stored losslessly: decoding the FLAC file gives the
  same samples.
- 32-bit float input is first quantised to 16 bits: sample * 32768, rounded
  to the nearest integer and limited to [-32768, 32767]. This is the only
  change made to the audio; the number of limited samples is returned so
  that it can be logged.
- Any other WAV encoding, and audio with more than one channel, is refused.

Input that is already compressed (FLAC, Ogg, MP3) is kept byte for byte; it
is never decoded and encoded again. The sample rate is never changed.

A FLAC file carries its number of samples and the MD5 of its decoded samples
in the STREAMINFO block, so `inspect` can describe a stored clip without
decoding it.
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass

import numpy as np

WAV_PCM, WAV_FLOAT, WAV_EXTENSIBLE = 1, 3, 0xFFFE
EXTENSIONS = {"wav": ".wav", "flac": ".flac", "ogg": ".ogg", "mp3": ".mp3"}


@dataclass(frozen=True)
class AudioInfo:
    container: str      # wav, flac, ogg or mp3
    encoding: str       # e.g. wav_f32le, wav_s16le, flac_s16, vorbis, mp3
    sample_rate: int
    channels: int
    frames: int         # samples per channel

    @property
    def duration_ms(self) -> int:
        return round(self.frames * 1000 / self.sample_rate)


def container_of(data: bytes) -> str:
    """Recognise the file type from its first bytes."""
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if data[:4] == b"fLaC":
        return "flac"
    if data[:4] == b"OggS":
        return "ogg"
    if data[:3] == b"ID3" or (len(data) > 1 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0):
        return "mp3"
    raise ValueError(f"unknown audio container (first bytes {data[:4]!r})")


def _wav_chunks(data: bytes) -> tuple[tuple[int, int, int, int], int, int]:
    """Return (format tag, channels, sample rate, bits per sample) and the data chunk's offset and size."""
    position, fmt, body = 12, None, None
    while position + 8 <= len(data):
        chunk, size = data[position:position + 4], struct.unpack_from("<I", data, position + 4)[0]
        start = position + 8
        if chunk == b"fmt ":
            tag, channels, rate, _, _, bits = struct.unpack_from("<HHIIHH", data, start)
            if tag == WAV_EXTENSIBLE:  # the real format is the first field of the sub-format GUID
                tag = struct.unpack_from("<H", data, start + 24)[0]
            fmt = (tag, channels, rate, bits)
        elif chunk == b"data":
            body = (start, size)
            break
        position = start + size + (size & 1)
    if fmt is None or body is None:
        raise ValueError("WAV file without fmt or data chunk")
    if body[0] + body[1] > len(data):
        raise ValueError(f"truncated WAV file: data chunk of {body[1]} bytes, {len(data) - body[0]} present")
    return fmt, body[0], body[1]


def _wav_encoding(tag: int, bits: int) -> str:
    if tag == WAV_PCM and bits == 16:
        return "wav_s16le"
    if tag == WAV_FLOAT and bits == 32:
        return "wav_f32le"
    raise ValueError(f"unsupported WAV encoding: format tag {tag}, {bits} bits")


def _flac_info(data: bytes) -> AudioInfo:
    if len(data) < 42 or data[4] & 0x7F != 0:
        raise ValueError("FLAC file without STREAMINFO block")
    packed = int.from_bytes(data[18:26], "big")  # 20 bits rate, 3 channels-1, 5 bits-1, 36 total samples
    bits = ((packed >> 36) & 0x1F) + 1
    return AudioInfo("flac", f"flac_s{bits}", packed >> 44, ((packed >> 41) & 0x7) + 1, packed & ((1 << 36) - 1))


def _decoded_info(data: bytes, container: str) -> AudioInfo:
    """Describe compressed audio by decoding it (needed where the header does not give the length)."""
    import av

    with av.open(io.BytesIO(data), "r") as media:
        stream = media.streams.audio[0]
        context = stream.codec_context
        frames = sum(frame.samples for frame in media.decode(stream))
        return AudioInfo(container, context.name, context.sample_rate, context.layout.nb_channels, frames)


def inspect(data: bytes) -> AudioInfo:
    """Container, encoding, sample rate, channels and length of an audio file held in memory."""
    container = container_of(data)
    if container == "wav":
        (tag, channels, rate, bits), _, size = _wav_chunks(data)
        return AudioInfo("wav", _wav_encoding(tag, bits), rate, channels, size // (channels * bits // 8))
    if container == "flac":
        return _flac_info(data)
    return _decoded_info(data, container)


def flac_md5(data: bytes) -> str:
    """MD5 of the decoded samples, as recorded in the FLAC header."""
    return data[26:42].hex()


def wav_to_pcm16(data: bytes) -> tuple[np.ndarray, int, int]:
    """Samples of a mono WAV file as 16-bit integers, its sample rate, and the number of limited samples."""
    (tag, channels, rate, bits), start, size = _wav_chunks(data)
    encoding = _wav_encoding(tag, bits)
    if channels != 1:
        raise ValueError(f"expected mono audio, found {channels} channels")
    body = data[start:start + size - size % (bits // 8)]
    if encoding == "wav_s16le":
        return np.frombuffer(body, dtype="<i2"), rate, 0
    scaled = np.frombuffer(body, dtype="<f4").astype(np.float64) * 32768.0
    if not np.isfinite(scaled).all():
        raise ValueError("WAV file with samples that are not finite numbers")
    rounded = np.rint(scaled)
    limited = int(np.count_nonzero((rounded < -32768) | (rounded > 32767)))
    return np.clip(rounded, -32768, 32767).astype("<i2"), rate, limited


def encode_flac(samples: np.ndarray, sample_rate: int) -> bytes:
    """Lossless FLAC file (mono, 16-bit) holding exactly these samples.

    The encoder runs in bit-exact mode without padding, so the bytes do not
    carry a library version string. They can still differ between FFmpeg
    versions; the decoded samples cannot.
    """
    import av

    if samples.dtype != np.dtype("<i2") or samples.ndim != 1 or samples.size == 0:
        raise ValueError("expected a non-empty one-dimensional array of 16-bit samples")
    buffer = io.BytesIO()
    options = {"fflags": "+bitexact", "metadata_header_padding": "0"}
    with av.open(buffer, "w", format="flac", options=options) as media:
        stream = media.add_stream("flac", rate=sample_rate, layout="mono", format="s16")
        stream.codec_context.flags |= av.codec.context.Flags.bitexact
        frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(samples).reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = sample_rate
        for packet in stream.encode(frame):
            media.mux(packet)
        for packet in stream.encode(None):
            media.mux(packet)
    return buffer.getvalue()


def decode_flac(data: bytes) -> tuple[np.ndarray, int]:
    """Samples (16-bit, mono) and sample rate of a FLAC file written by `encode_flac`."""
    import av

    with av.open(io.BytesIO(data), "r") as media:
        stream = media.streams.audio[0]
        context = stream.codec_context
        if context.name != "flac" or context.format.name != "s16" or context.layout.nb_channels != 1:
            raise ValueError(f"expected mono 16-bit FLAC, found {context.name} {context.format.name}")
        chunks = [frame.to_ndarray().reshape(-1) for frame in media.decode(stream)]
        return np.concatenate(chunks).astype("<i2"), context.sample_rate


def store(data: bytes) -> tuple[bytes, str, AudioInfo, int]:
    """Bytes to keep in the repository for one source clip.

    Returns (stored bytes, file extension, description of the source, number
    of limited samples). WAV becomes 16-bit FLAC as described in the module
    text; compressed input is returned unchanged.
    """
    source = inspect(data)
    if source.frames == 0:
        raise ValueError("audio without samples")
    if source.container != "wav":
        return data, EXTENSIONS[source.container], source, 0
    samples, rate, limited = wav_to_pcm16(data)
    flac = encode_flac(samples, rate)
    if _flac_info(flac).frames != samples.size:
        raise RuntimeError("FLAC encoder wrote a different number of samples")
    return flac, ".flac", source, limited
