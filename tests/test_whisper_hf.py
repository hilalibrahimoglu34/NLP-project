"""Tests for the Hugging Face Whisper systems. No model is loaded and nothing is downloaded: the cache
folder, the model, the feature extractor and the tokenizer are stand-ins."""

import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from huasr import evaluate, report, systems, whisper_hf
from huasr.whisper_hf import DECODING, FILES, RECORD, REVISION, WEIGHTS, HuggingFaceWhisper, windows
from tests import needs

LIMIT = whisper_hf.WINDOW_SECONDS * whisper_hf.SAMPLE_RATE
RECORD_CONTENT = {"complete": True, "base_checkpoint": {"repository": "openai/whisper-small", "revision": REVISION},
                  "best": {"step": 1750, "wer": 21.5}, "hyperparameters": {"max_steps": 2000}}


def published_folder(root: Path, weights: bytes = b"weights") -> Path:
    """A folder that looks like the cached snapshot of the published checkpoint."""
    folder = root / "hub" / "models--openai--whisper-small" / "snapshots" / REVISION
    folder.mkdir(parents=True)
    for name in FILES:
        (folder / name).write_bytes(weights if name == WEIGHTS else b"{}")
    return folder


def finetuned_folder(root: Path, record: dict = RECORD_CONTENT) -> Path:
    folder = root / "best"
    folder.mkdir(parents=True)
    (folder / WEIGHTS).write_bytes(b"trained weights")
    (folder / RECORD).write_text(json.dumps(record), encoding="utf-8")
    return folder


class TemporaryFolderTestCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)


class WindowTests(unittest.TestCase):
    def test_audio_that_fits_is_one_part(self):
        for length in (0, 1, 16_000, LIMIT):
            samples = np.arange(length, dtype=np.float32)
            parts = windows(samples, LIMIT)
            self.assertEqual(len(parts), 1)
            np.testing.assert_array_equal(parts[0], samples)

    def test_longer_audio_is_cut_into_the_fewest_equal_parts(self):
        for length, expected in ((LIMIT + 1, 2), (2 * LIMIT, 2), (2 * LIMIT + 1, 3), (920_000, 2), (5 * LIMIT - 7, 5)):
            samples = np.arange(length, dtype=np.float32)
            parts = windows(samples, LIMIT)
            self.assertEqual(len(parts), expected)
            self.assertLessEqual(max(len(part) for part in parts), LIMIT)
            self.assertLessEqual(max(len(part) for part in parts) - min(len(part) for part in parts), 1)
            np.testing.assert_array_equal(np.concatenate(parts), samples)  # nothing lost, nothing repeated


class RegistryTests(unittest.TestCase):
    def test_both_systems_are_registered_with_their_groups(self):
        published = systems.SYSTEMS["hf-whisper-small"]()
        own = systems.SYSTEMS["hf-whisper-small-finetuned"]()
        self.assertIsInstance(published, HuggingFaceWhisper)
        self.assertEqual((published.name, published.group, published.fine_tuned),
                         ("hf-whisper-small", "baselines", False))
        self.assertEqual((own.name, own.group, own.fine_tuned), ("hf-whisper-small-finetuned", "own", True))
        self.assertEqual((whisper_hf.PRETRAINED, whisper_hf.FINETUNED), (published.name, own.name))
        for system in (published, own):
            self.assertIn(system.group, systems.GROUPS)
            self.assertIsNone(system._model)  # building a system loads nothing

    def test_both_systems_decode_in_the_same_declared_way(self):
        self.assertEqual(DECODING, {"language": "hu", "task": "transcribe", "return_timestamps": False,
                                    "num_beams": 5, "temperature": 0.0})
        for name in (whisper_hf.PRETRAINED, whisper_hf.FINETUNED):
            self.assertIs(whisper_hf.build(name).decoding, DECODING)

    def test_checkpoint_and_weight_file_are_pinned(self):
        self.assertEqual(whisper_hf.REPOSITORY, "openai/whisper-small")
        self.assertRegex(REVISION, r"^[0-9a-f]{40}$")
        self.assertRegex(whisper_hf.WEIGHTS_SHA256, r"^[0-9a-f]{64}$")
        self.assertIn(WEIGHTS, FILES)

    def test_folder_of_the_fine_tuned_model(self):
        with mock.patch.dict(os.environ):
            os.environ.pop(whisper_hf.FINETUNED_VARIABLE, None)
            self.assertEqual(whisper_hf.build(whisper_hf.FINETUNED)._folder, whisper_hf.FINETUNED_FOLDER)
            os.environ[whisper_hf.FINETUNED_VARIABLE] = "somewhere/else"
            self.assertEqual(whisper_hf.build(whisper_hf.FINETUNED)._folder, Path("somewhere/else"))
        self.assertEqual(whisper_hf.FINETUNED_FOLDER.parts[0], "scratch")  # a folder that git ignores

    def test_unknown_name_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "unknown system"):
            whisper_hf.build("hf-whisper-large")

    def test_the_runner_knows_the_systems_and_still_guards_the_test_split(self):
        for name in (whisper_hf.PRETRAINED, whisper_hf.FINETUNED):
            err = io.StringIO()
            with tempfile.TemporaryDirectory() as results, contextlib.redirect_stderr(err):
                code = evaluate.main(["--system", name, "--split", "test", "--results", results])
                self.assertEqual(os.listdir(results), [])
            self.assertEqual(code, 1)
            self.assertIn("--final-test", err.getvalue())

    def test_versions_are_those_of_the_modules_that_are_loaded(self):
        found = whisper_hf.versions(("numpy", "a_package_that_does_not_exist"))
        self.assertEqual(found, {"numpy": np.__version__, "a_package_that_does_not_exist": "not installed"})


@needs("huggingface_hub")
class SnapshotTests(TemporaryFolderTestCase):
    def setUp(self):
        super().setUp()
        environment = mock.patch.dict(os.environ)  # the lookup switches the library to offline mode
        environment.start()
        self.addCleanup(environment.stop)

    def test_the_checkpoint_is_looked_up_in_the_local_cache_only(self):
        folder = published_folder(self.root)
        with mock.patch("huggingface_hub.snapshot_download", return_value=str(folder)) as lookup:
            self.assertEqual(whisper_hf.snapshot(), folder)
        lookup.assert_called_once_with("openai/whisper-small", revision=REVISION, allow_patterns=list(FILES),
                                       local_files_only=True)

    def test_only_fetch_may_download(self):
        folder = published_folder(self.root)
        with mock.patch("huggingface_hub.snapshot_download", return_value=str(folder)) as lookup:
            whisper_hf.snapshot(download=True)
        self.assertFalse(lookup.call_args.kwargs["local_files_only"])

    def test_a_checkpoint_that_is_not_cached_is_an_error_that_names_the_command(self):
        with mock.patch("huggingface_hub.snapshot_download", side_effect=RuntimeError("no such entry")):
            with self.assertRaisesRegex(FileNotFoundError, "python -m huasr.whisper_hf fetch"):
                whisper_hf.snapshot()

    def test_an_incomplete_snapshot_is_an_error(self):
        folder = published_folder(self.root)
        (folder / WEIGHTS).unlink()
        with mock.patch("huggingface_hub.snapshot_download", return_value=str(folder)):
            with self.assertRaisesRegex(FileNotFoundError, WEIGHTS):
                whisper_hf.snapshot()


class CheckpointTests(TemporaryFolderTestCase):
    def pinned(self, weights: bytes):
        return mock.patch.multiple(whisper_hf, WEIGHTS_BYTES=len(weights),
                                   WEIGHTS_SHA256=hashlib.sha256(weights).hexdigest())

    def test_published_checkpoint_is_named_by_repository_revision_and_weight_hash(self):
        system = HuggingFaceWhisper("hf-whisper-small", "baselines")
        system._folder = published_folder(self.root)
        with self.pinned(b"weights"):
            checkpoint = system.checkpoint()
        self.assertEqual(checkpoint, {"repository": "openai/whisper-small", "revision": REVISION,
                                      "weights_file": WEIGHTS, "weights_bytes": 7,
                                      "weights_sha256": hashlib.sha256(b"weights").hexdigest()})
        self.assertEqual(report._checkpoint(checkpoint), f"`openai/whisper-small` at `{REVISION[:12]}`")

    def test_other_weights_than_the_pinned_ones_are_refused(self):
        system = HuggingFaceWhisper("hf-whisper-small", "baselines")
        system._folder = published_folder(self.root, weights=b"something else")
        with self.pinned(b"weights"), self.assertRaisesRegex(ValueError, "is not the weight file"):
            system.checkpoint()

    def test_fine_tuned_checkpoint_is_described_by_its_training_record(self):
        folder = finetuned_folder(self.root)
        checkpoint = HuggingFaceWhisper("hf-whisper-small-finetuned", "own", folder).checkpoint()
        self.assertNotIn("repository", checkpoint)  # the result table shows the description for our own models
        self.assertEqual(checkpoint["description"], f"`openai/whisper-small` at `{REVISION[:12]}`, fine-tuned on "
                                                    f"the training split (step 1750 of 2000)")
        self.assertEqual(report._checkpoint(checkpoint), checkpoint["description"])
        self.assertEqual(checkpoint["base"], {"repository": "openai/whisper-small", "revision": REVISION})
        self.assertEqual((checkpoint["best_step"], checkpoint["selection_wer"], checkpoint["training_complete"]),
                         (1750, 21.5, True))
        self.assertEqual(checkpoint["weights_sha256"], hashlib.sha256(b"trained weights").hexdigest())
        self.assertEqual(checkpoint["training_record_sha256"],
                         hashlib.sha256((folder / RECORD).read_bytes()).hexdigest())

    def test_a_trial_training_run_is_marked_in_the_description(self):
        folder = finetuned_folder(self.root, {**RECORD_CONTENT, "complete": False})
        checkpoint = HuggingFaceWhisper("hf-whisper-small-finetuned", "own", folder).checkpoint()
        self.assertIn("trial run", checkpoint["description"])
        self.assertFalse(checkpoint["training_complete"])

    def test_a_missing_fine_tuned_model_is_an_error_that_says_what_to_do(self):
        system = HuggingFaceWhisper("hf-whisper-small-finetuned", "own", self.root / "nothing")
        with self.assertRaisesRegex(FileNotFoundError, "python -m huasr.finetune train"):
            system.describe()
        (self.root / "nothing").mkdir()
        (self.root / "nothing" / WEIGHTS).write_bytes(b"weights without a training record")
        with self.assertRaisesRegex(FileNotFoundError, whisper_hf.FINETUNED_VARIABLE):
            system.load()

    @needs("torch")
    def test_description_has_the_settings_of_the_run_manifest(self):
        description = HuggingFaceWhisper("hf-whisper-small-finetuned", "own", finetuned_folder(self.root)).describe()
        settings = description["settings"]
        self.assertEqual({key: settings[key] for key in DECODING}, DECODING)
        self.assertEqual((settings["fine_tuned"], settings["dtype"], settings["window_seconds"]), (True, "float32", 30))
        self.assertIn(settings["device"], ("cpu", "cuda"))
        self.assertIn("gpu", settings)
        self.assertEqual(description["seed"], 0)
        self.assertEqual(set(description["packages"]), set(whisper_hf.PACKAGES))
        json.dumps(description)  # everything can go into run.json


class FakeFeatureExtractor:
    def __init__(self):
        self.calls = []

    def __call__(self, samples, sampling_rate, return_tensors):
        import torch

        self.calls.append((len(samples), sampling_rate, return_tensors))
        return {"input_features": torch.zeros(1, 80, 3000)}


class FakeModel:
    def __init__(self):
        self.calls = []

    def generate(self, features, **options):
        import torch

        self.calls.append((tuple(features.shape), options))
        return torch.tensor([[50258, 50286, 50359, 50363, len(self.calls), 50257]])


class FakeTokenizer:
    texts = {1: " jó napot ", 2: "", 3: "kívánok"}

    def batch_decode(self, tokens, skip_special_tokens):
        assert skip_special_tokens
        return [self.texts[int(row[4])] for row in tokens]


@needs("torch")
class LoadTests(TemporaryFolderTestCase):
    def test_model_tokenizer_and_feature_extractor_come_from_the_folder_of_the_checkpoint_only(self):
        library = mock.MagicMock()
        original = sys.modules.get("transformers")
        sys.modules["transformers"] = library  # only this entry is swapped, and put back below
        self.addCleanup(lambda: sys.modules.pop("transformers") if original is None
                        else sys.modules.__setitem__("transformers", original))
        folder = finetuned_folder(self.root)
        system = HuggingFaceWhisper("hf-whisper-small-finetuned", "own", folder)
        with mock.patch.object(whisper_hf, "accelerator", return_value={"device": "cpu", "gpu": None}):
            system.load()
            system.load()  # the second call loads nothing
        for loader in (library.WhisperFeatureExtractor, library.WhisperTokenizer,
                       library.WhisperForConditionalGeneration):
            loader.from_pretrained.assert_called_once_with(str(folder), local_files_only=True)
        model = library.WhisperForConditionalGeneration.from_pretrained.return_value
        model.to.assert_called_once_with("cpu")
        self.assertIs(system._model, model.to.return_value.eval.return_value)
        self.assertIs(system._tokenizer, library.WhisperTokenizer.from_pretrained.return_value)


@needs("torch")
class TranscribeTests(unittest.TestCase):
    def setUp(self):
        self.system = HuggingFaceWhisper("hf-whisper-small", "baselines")
        self.system._model, self.system._feature_extractor = FakeModel(), FakeFeatureExtractor()
        self.system._tokenizer, self.system._device = FakeTokenizer(), "cpu"

    def test_one_clip_is_decoded_with_the_declared_settings(self):
        self.assertEqual(self.system.transcribe(np.zeros(16_000, dtype=np.float32), 16_000), "jó napot")
        self.assertEqual(self.system._feature_extractor.calls, [(16_000, 16_000, "pt")])
        self.assertEqual(self.system._model.calls, [((1, 80, 3000), DECODING)])

    def test_a_clip_longer_than_the_window_is_decoded_in_parts_and_joined(self):
        samples = np.zeros(2 * LIMIT + 3_000, dtype=np.float32)  # just over a minute: three parts
        self.assertEqual(self.system.transcribe(samples, 16_000), "jó napot kívánok")  # the empty part adds nothing
        lengths = [length for length, _, _ in self.system._feature_extractor.calls]
        self.assertEqual((len(lengths), sum(lengths)), (3, len(samples)))
        self.assertLessEqual(max(lengths), LIMIT)

    def test_other_sample_rates_are_refused(self):
        with self.assertRaisesRegex(ValueError, "16 kHz"):
            self.system.transcribe(np.zeros(8_000, dtype=np.float32), 8_000)
        self.assertEqual(self.system._model.calls, [])


class CommandTests(unittest.TestCase):
    def test_fetch_reports_an_error_instead_of_a_traceback(self):
        err = io.StringIO()
        with mock.patch.object(whisper_hf, "snapshot", side_effect=OSError("no network")) as fetch, \
                contextlib.redirect_stderr(err):
            self.assertEqual(whisper_hf.main(["fetch"]), 1)
        fetch.assert_called_once_with(download=True)
        self.assertIn("no network", err.getvalue())


if __name__ == "__main__":
    unittest.main()
