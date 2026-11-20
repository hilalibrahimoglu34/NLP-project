"""Tests for the audio storage rules: WAV inspection, 16-bit conversion, FLAC, pass-through."""

import hashlib
import io
import struct
import unittest

import numpy as np

from huasr import audio
from tests import needs
from tests.fixtures import RATE, quantise, signal, wav


class InspectTests(unittest.TestCase):
    def test_container_is_recognised_from_the_first_bytes(self):
        self.assertEqual(audio.container_of(wav(signal(0.1))), "wav")
        self.assertEqual(audio.container_of(b"fLaC" + bytes(40)), "flac")
        self.assertEqual(audio.container_of(b"OggS" + bytes(40)), "ogg")
        self.assertEqual(audio.container_of(b"ID3\x04" + bytes(40)), "mp3")
        self.assertEqual(audio.container_of(b"\xff\xfb\x90\x00" + bytes(40)), "mp3")
        with self.assertRaises(ValueError):
            audio.container_of(b"<html>not audio</html>")

    def test_float_wav(self):
        info = audio.inspect(wav(signal(0.5)))
        self.assertEqual((info.container, info.encoding, info.sample_rate, info.channels, info.frames),
                         ("wav", "wav_f32le", RATE, 1, 8000))
        self.assertEqual(info.duration_ms, 500)

    def test_pcm16_wav(self):
        info = audio.inspect(wav(quantise(signal(0.25))))
        self.assertEqual((info.encoding, info.frames, info.duration_ms), ("wav_s16le", 4000, 250))

    def test_other_chunks_and_extensible_header(self):
        samples = signal(0.1)
        for data in (wav(samples, extra_chunks=True), wav(samples, extensible=True),
                     wav(quantise(samples), extra_chunks=True, extensible=True)):
            self.assertEqual(audio.inspect(data).frames, 1600)

    def test_broken_wav_files_are_errors(self):
        data = wav(signal(0.1))
        with self.assertRaisesRegex(ValueError, "truncated"):
            audio.inspect(data[:-100])
        with self.assertRaisesRegex(ValueError, "fmt or data"):
            audio.inspect(data[:36])
        with self.assertRaises((ValueError, struct.error)):
            audio.inspect(b"RIFF\x04\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01")

    def test_other_encodings_are_refused_not_converted(self):
        eight_bit = bytearray(wav(quantise(signal(0.1))))
        struct.pack_into("<H", eight_bit, 34, 8)  # bits per sample in the fmt chunk
        with self.assertRaisesRegex(ValueError, "unsupported WAV encoding"):
            audio.inspect(bytes(eight_bit))
        stereo = wav(signal(0.1), channels=2)
        self.assertEqual(audio.inspect(stereo).channels, 2)
        with self.assertRaisesRegex(ValueError, "mono"):
            audio.wav_to_pcm16(stereo)


class ConversionTests(unittest.TestCase):
    def test_pcm16_is_returned_unchanged(self):
        pcm = quantise(signal(0.2))
        samples, rate, limited = audio.wav_to_pcm16(wav(pcm, rate=8000))
        self.assertTrue(np.array_equal(samples, pcm))
        self.assertEqual((rate, limited), (8000, 0))

    def test_float_is_scaled_rounded_and_limited(self):
        values = np.array([0.0, 0.5, -0.5, 0.25, -1.0, 1.0, 1.5, -1.5, 0.9999, 3 / 32768], dtype="<f4")
        samples, rate, limited = audio.wav_to_pcm16(wav(values))
        self.assertEqual(samples.tolist(), [0, 16384, -16384, 8192, -32768, 32767, 32767, -32768, 32765, 3])
        self.assertEqual((rate, limited), (RATE, 3))  # 1.0, 1.5 and -1.5 are outside the 16-bit range
        self.assertEqual(samples.dtype, np.dtype("<i2"))

    def test_samples_that_are_not_numbers_are_an_error(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            audio.wav_to_pcm16(wav(np.array([0.1, np.nan, 0.2], dtype="<f4")))


@needs("av")
class FlacTests(unittest.TestCase):
    def test_flac_is_lossless_for_16_bit_samples(self):
        pcm = quantise(signal(1.0, seed=3))
        flac = audio.encode_flac(pcm, RATE)
        decoded, rate = audio.decode_flac(flac)
        self.assertTrue(np.array_equal(decoded, pcm))
        self.assertEqual(rate, RATE)
        self.assertLess(len(flac), pcm.nbytes)

    def test_flac_header_describes_the_clip(self):
        pcm = quantise(signal(0.75, seed=1))
        flac = audio.encode_flac(pcm, RATE)
        info = audio.inspect(flac)
        self.assertEqual((info.container, info.encoding, info.sample_rate, info.channels, info.frames),
                         ("flac", "flac_s16", RATE, 1, pcm.size))
        self.assertEqual(info.duration_ms, 750)
        self.assertEqual(audio.flac_md5(flac), hashlib.md5(pcm.tobytes()).hexdigest())

    def test_encoding_twice_gives_the_same_bytes_without_a_version_string(self):
        pcm = quantise(signal(0.5))
        first = audio.encode_flac(pcm, RATE)
        self.assertEqual(first, audio.encode_flac(pcm, RATE))
        self.assertNotIn(b"Lavf", first)
        self.assertLess(len(first), 8192 + pcm.nbytes // 2)  # no padding block

    def test_only_one_dimensional_16_bit_arrays_are_encoded(self):
        for wrong in (signal(0.1), quantise(signal(0.1)).reshape(2, -1), np.zeros(0, dtype="<i2")):
            with self.assertRaises(ValueError):
                audio.encode_flac(wrong, RATE)

    def test_store_turns_float_wav_into_16_bit_flac(self):
        samples = signal(0.6, seed=5)
        samples[10] = 1.25  # one sample outside the range
        source = wav(samples)
        stored, extension, info, limited = audio.store(source)
        self.assertEqual((extension, info.encoding, info.frames, limited), (".flac", "wav_f32le", samples.size, 1))
        decoded, rate = audio.decode_flac(stored)
        self.assertTrue(np.array_equal(decoded, quantise(samples)))
        self.assertEqual(rate, RATE)
        self.assertEqual(stored, audio.store(source)[0])
        self.assertLess(len(stored), len(source) / 2)

    def test_store_keeps_the_sample_rate(self):
        stored, _, _, _ = audio.store(wav(quantise(signal(0.2)), rate=8000))
        self.assertEqual(audio.inspect(stored).sample_rate, 8000)

    def test_store_keeps_compressed_input_byte_for_byte(self):
        flac = audio.encode_flac(quantise(signal(0.3)), RATE)
        stored, extension, info, limited = audio.store(flac)
        self.assertIs(stored, flac)
        self.assertEqual((extension, info.encoding, info.frames, limited), (".flac", "flac_s16", 4800, 0))

    def test_store_keeps_ogg_input_byte_for_byte(self):
        import av

        try:
            av.codec.Codec("libopus", "w")
        except Exception:
            self.skipTest("this FFmpeg build has no Opus encoder")
        buffer = io.BytesIO()
        with av.open(buffer, "w", format="ogg") as media:
            stream = media.add_stream("libopus", rate=48000, layout="mono", format="s16")
            frame = av.AudioFrame.from_ndarray(np.zeros((1, 48000), dtype="<i2"), format="s16", layout="mono")
            frame.sample_rate = 48000
            for packet in [*stream.encode(frame), *stream.encode(None)]:
                media.mux(packet)
        ogg = buffer.getvalue()
        stored, extension, info, limited = audio.store(ogg)
        self.assertIs(stored, ogg)
        self.assertEqual((extension, info.container, info.encoding, limited), (".ogg", "ogg", "opus", 0))
        self.assertAlmostEqual(info.duration_ms, 1000, delta=60)

    def test_audio_without_samples_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "without samples"):
            audio.store(wav(np.zeros(0, dtype="<f4")))


if __name__ == "__main__":
    unittest.main()
