"""Synthetic audio for the tests: short generated signals written as WAV files in memory.

Nothing here is a real recording.
"""

import struct

import numpy as np

RATE = 16_000


def signal(seconds: float, seed: int = 0) -> np.ndarray:
    """A tone with a little noise, as 32-bit floats between -1 and 1."""
    rng = np.random.default_rng(seed)
    t = np.arange(round(seconds * RATE)) / RATE
    tone = 0.3 * np.sin(2 * np.pi * (180 + 40 * (seed % 7)) * t)
    return (tone + 0.02 * rng.standard_normal(t.size)).astype("<f4")


def quantise(samples: np.ndarray) -> np.ndarray:
    """The 16-bit version of float samples that `huasr.audio` is expected to produce."""
    return np.clip(np.rint(samples.astype(np.float64) * 32768.0), -32768, 32767).astype("<i2")


def wav(samples: np.ndarray, rate: int = RATE, channels: int = 1, extra_chunks: bool = False,
        extensible: bool = False) -> bytes:
    """A WAV file holding `samples`: 32-bit float (dtype <f4) or 16-bit PCM (dtype <i2)."""
    tag, bits = (3, 32) if samples.dtype == np.dtype("<f4") else (1, 16)
    body = samples.tobytes()
    block = channels * bits // 8
    fmt = struct.pack("<HHIIHH", 0xFFFE if extensible else tag, channels, rate, rate * block, block, bits)
    if extensible:  # cbSize, valid bits, channel mask, sub-format GUID starting with the real format tag
        fmt += struct.pack("<HHIH", 22, bits, 0, tag) + bytes(14)
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    if extra_chunks:  # a chunk of odd length (padded) and a fact chunk before the data
        chunks += b"LIST" + struct.pack("<I", 5) + b"hello" + b"\x00"
        chunks += b"fact" + struct.pack("<I", 4) + struct.pack("<I", samples.size // channels)
    chunks += b"data" + struct.pack("<I", len(body)) + body
    return b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks
