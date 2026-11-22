"""Tests for the system adapters. No model is loaded: the cache folder and the model are stand-ins."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from huasr import systems
from huasr.systems import FasterWhisper, cached_checkpoint
from tests import needs


class Segment:
    def __init__(self, text):
        self.text = text


class FakeModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, samples, **options):
        self.calls.append((samples, options))
        return iter([Segment(" jó napot "), Segment("kívánok")]), None


class CheckpointTests(unittest.TestCase):
    def test_repository_and_revision_are_read_from_the_cache_path(self):
        path = Path("C:/cache/hub/models--Systran--faster-whisper-small/snapshots/536b0662742c")
        self.assertEqual(cached_checkpoint(path), {"repository": "Systran/faster-whisper-small",
                                                   "revision": "536b0662742c"})

    def test_other_folders_are_refused(self):
        for path in ("C:/models/whisper-small", "C:/cache/hub/datasets--a--b/snapshots/123", "snapshots"):
            with self.assertRaisesRegex(ValueError, "snapshot folder"):
                cached_checkpoint(Path(path))

    def test_every_converted_repository_names_its_openai_checkpoint(self):
        self.assertEqual(systems.UPSTREAM, {"Systran/faster-whisper-base": "openai/whisper-base",
                                            "Systran/faster-whisper-small": "openai/whisper-small"})


class RegistryTests(unittest.TestCase):
    def test_registered_systems(self):
        self.assertEqual(sorted(systems.SYSTEMS), [
            "canary-1b-v2", "faster-whisper-base", "faster-whisper-small",
            "hf-whisper-small", "hf-whisper-small-finetuned",        # these two: test_whisper_hf.py
            "parakeet-tdt-0.6b-v3", "qwen3-asr-1.7b", "wav2vec2-xlsr-53-hungarian"])  # the others: test_pretrained.py
        system = systems.SYSTEMS["faster-whisper-small"]()
        self.assertEqual((system.name, system.group, system.size), ("faster-whisper-small", "baselines", "small"))
        self.assertIn(system.group, systems.GROUPS)

    def test_declared_settings(self):
        self.assertEqual(FasterWhisper.runtime, {"device": "cpu", "compute_type": "int8", "cpu_threads": 4})
        self.assertEqual(FasterWhisper.decoding, {"language": "hu", "task": "transcribe", "beam_size": 5,
                                                  "temperature": 0.0, "condition_on_previous_text": False})


class FasterWhisperTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.snapshot = Path(temporary.name) / "models--Systran--faster-whisper-small" / "snapshots" / "abc123"
        self.snapshot.mkdir(parents=True)
        (self.snapshot / "model.bin").write_bytes(b"weights")
        self.system = FasterWhisper("small", seed=3)
        self.system._snapshot = self.snapshot

    @needs("faster_whisper")
    def test_description_names_the_checkpoint_that_is_loaded(self):
        description = self.system.describe()
        self.assertEqual(description["checkpoint"], {
            "repository": "Systran/faster-whisper-small", "revision": "abc123", "converted_from": "openai/whisper-small",
            "model_bin_bytes": 7, "model_bin_sha256": hashlib.sha256(b"weights").hexdigest()})
        self.assertEqual(description["seed"], 3)
        self.assertEqual(description["settings"]["temperature"], 0.0)
        self.assertEqual((description["settings"]["compute_type"], description["settings"]["cpu_threads"],
                          description["settings"]["fine_tuned"]), ("int8", 4, False))
        self.assertEqual(set(description["packages"]), {"faster-whisper", "ctranslate2"})

    @needs("faster_whisper")
    def test_the_model_is_looked_up_in_the_local_cache_only(self):
        system = FasterWhisper("base")
        with mock.patch("faster_whisper.utils.download_model", return_value=str(self.snapshot)) as lookup:
            self.assertEqual(system.snapshot(), self.snapshot)
            self.assertEqual(system.snapshot(), self.snapshot)
        lookup.assert_called_once_with("base", local_files_only=True, revision=systems.REVISIONS["base"])

    def test_every_registered_size_has_a_pinned_revision(self):
        for name, build in systems.SYSTEMS.items():
            if name.startswith("faster-whisper-"):
                self.assertRegex(systems.REVISIONS[build().size], r"^[0-9a-f]{40}$")

    def test_segments_are_joined_and_the_declared_settings_are_passed(self):
        self.system._model = FakeModel()
        samples = np.zeros(16_000, dtype=np.float32)
        self.assertEqual(self.system.transcribe(samples, 16_000), "jó napot kívánok")
        passed, options = self.system._model.calls[0]
        self.assertIs(passed, samples)
        self.assertEqual(options, FasterWhisper.decoding)

    def test_other_sample_rates_are_refused(self):
        self.system._model = FakeModel()
        with self.assertRaisesRegex(ValueError, "16 kHz"):
            self.system.transcribe(np.zeros(8_000, dtype=np.float32), 8_000)
        self.assertEqual(self.system._model.calls, [])


if __name__ == "__main__":
    unittest.main()
