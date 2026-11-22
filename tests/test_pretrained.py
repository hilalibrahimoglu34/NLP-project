"""Tests for the pretrained baselines of `huasr.pretrained`. No model is loaded and nothing is downloaded:
the cache folders, the processors and the models are stand-ins that accept only the calls the real ones are
given in the documentation of the models."""

import contextlib
import dataclasses
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from huasr import evaluate, pretrained, report, score, systems
from huasr.pretrained import CLASSES, Canary, ParakeetTDT, Qwen3ASR, Wav2Vec2CTC
from huasr.text import normalize
from tests import REPOSITORY, needs

NAMES = ("wav2vec2-xlsr-53-hungarian", "canary-1b-v2", "parakeet-tdt-0.6b-v3", "qwen3-asr-1.7b")
RATE = 16_000
CANARY_PROMPT = ["<|startofcontext|>", "<|startoftranscript|>", "<|emo:undefined|>", "<|hu|>", "<|hu|>", "<|pnc|>",
                 "<|noitn|>", "<|notimestamp|>", "<|nodiarize|>"]


def cache_folder(root: Path, checkpoint, weights: bytes = b"weights") -> Path:
    """A folder that looks like the cached snapshot of `checkpoint`."""
    folder = root / "hub" / ("models--" + checkpoint.repository.replace("/", "--")) / "snapshots" / checkpoint.revision
    folder.mkdir(parents=True, exist_ok=True)
    for name in checkpoint.files:
        (folder / name).write_bytes(weights if name == checkpoint.weights else b"{}")
    return folder


def with_weights(checkpoint, weights: bytes = b"weights"):
    """`checkpoint` with the size and SHA-256 of `weights` as its pinned values."""
    return dataclasses.replace(checkpoint, weights_bytes=len(weights),
                               weights_sha256=hashlib.sha256(weights).hexdigest())


class TemporaryFolderTestCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def cached(self, cls, weights: bytes = b"weights"):
        """A system of `cls` whose checkpoint is in a stand-in cache folder with matching pinned values."""
        patch = mock.patch.object(cls, "checkpoint", with_weights(cls.checkpoint, weights))
        patch.start()
        self.addCleanup(patch.stop)
        system = cls(next(name for name, value in CLASSES.items() if value is cls))
        system._folder = cache_folder(self.root, cls.checkpoint, weights)
        return system


class RegistryTests(unittest.TestCase):
    def test_the_four_systems_are_registered_as_baselines(self):
        self.assertEqual(tuple(CLASSES), NAMES)
        for name in NAMES:
            system = systems.SYSTEMS[name]()
            self.assertIsInstance(system, CLASSES[name])
            self.assertEqual((system.name, system.group), (name, "baselines"))  # never "own": nothing is trained
            self.assertIsNone(system._model)                                    # building a system loads nothing
        self.assertEqual({name for name in systems.SYSTEMS if name in CLASSES}, set(NAMES))

    def test_unknown_name_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "unknown system"):
            pretrained.build("hubert-ctc-hungarian")

    def test_checkpoints_and_revisions_are_pinned(self):
        expected = {
            "wav2vec2-xlsr-53-hungarian": ("jonatasgrosman/wav2vec2-large-xlsr-53-hungarian", "pytorch_model.bin"),
            "canary-1b-v2": ("nvidia/canary-1b-v2", "model.safetensors"),
            "parakeet-tdt-0.6b-v3": ("nvidia/parakeet-tdt-0.6b-v3", "model.safetensors"),
            "qwen3-asr-1.7b": ("Qwen/Qwen3-ASR-1.7B-hf", "model.safetensors"),
        }
        for name, cls in CLASSES.items():
            checkpoint = cls.checkpoint
            self.assertEqual((checkpoint.repository, checkpoint.weights), expected[name])
            self.assertRegex(checkpoint.revision, r"^[0-9a-f]{40}$")
            self.assertRegex(checkpoint.weights_sha256, r"^[0-9a-f]{64}$")
            self.assertGreater(checkpoint.weights_bytes, 10 ** 9)
            self.assertEqual(checkpoint.files[-1], checkpoint.weights)
            self.assertIn("config.json", checkpoint.files)
        self.assertEqual(len({cls.checkpoint.revision for cls in CLASSES.values()}), 4)

    def test_language_is_declared_for_every_system(self):
        self.assertEqual(Canary.request, {"source_language": "hu", "target_language": "hu", "punctuation": True})
        self.assertEqual((Canary.settings["source_language"], Canary.settings["target_language"]), ("hu", "hu"))
        self.assertEqual((Qwen3ASR.settings["language"], Qwen3ASR.prefix),
                         ("Hungarian", "language Hungarian<asr_text>"))
        self.assertIn("takes no language setting", ParakeetTDT.settings["language"])
        self.assertIn("Hungarian only", Wav2Vec2CTC.settings["language"])
        self.assertIsNone(Wav2Vec2CTC.settings["language_model"])

    def test_only_canary_is_given_its_audio_in_windows(self):
        self.assertEqual({name: cls.window_seconds for name, cls in CLASSES.items()},
                         {"wav2vec2-xlsr-53-hungarian": None, "canary-1b-v2": 40, "parakeet-tdt-0.6b-v3": None,
                          "qwen3-asr-1.7b": None})
        self.assertEqual({name: cls.dtype for name, cls in CLASSES.items()}["qwen3-asr-1.7b"], "bfloat16")

    def test_the_runner_knows_the_systems_and_still_guards_the_test_split(self):
        for name in NAMES:
            err = io.StringIO()
            with tempfile.TemporaryDirectory() as results, contextlib.redirect_stderr(err):
                code = evaluate.main(["--system", name, "--split", "test", "--results", results])
                self.assertEqual(os.listdir(results), [])
            self.assertEqual(code, 1)
            self.assertIn("--final-test", err.getvalue())

    def test_the_decoder_prompt_of_canary_is_read_for_its_languages(self):
        self.assertEqual(pretrained.prompt_languages(CANARY_PROMPT), ["<|hu|>", "<|hu|>"])
        self.assertEqual(pretrained.prompt_languages(["<|startoftranscript|>", "<|en|>", "<|de|>", "<|nopnc|>"]),
                         ["<|en|>", "<|de|>"])


@needs("huggingface_hub")
class SnapshotTests(TemporaryFolderTestCase):
    def setUp(self):
        super().setUp()
        environment = mock.patch.dict(os.environ)  # the lookup switches the library to offline mode
        environment.start()
        self.addCleanup(environment.stop)

    def test_checkpoints_are_looked_up_in_the_local_cache_only(self):
        for cls in CLASSES.values():
            checkpoint = cls.checkpoint
            folder = cache_folder(self.root / checkpoint.revision, checkpoint)
            with mock.patch("huggingface_hub.snapshot_download", return_value=str(folder)) as lookup:
                self.assertEqual(pretrained.snapshot(checkpoint), folder)
            lookup.assert_called_once_with(checkpoint.repository, revision=checkpoint.revision,
                                           allow_patterns=list(checkpoint.files), local_files_only=True)

    def test_only_fetch_may_download_and_it_takes_the_listed_files_only(self):
        folder = cache_folder(self.root, pretrained.CANARY)
        with mock.patch("huggingface_hub.snapshot_download", return_value=str(folder)) as lookup:
            pretrained.snapshot(pretrained.CANARY, download=True)
        self.assertFalse(lookup.call_args.kwargs["local_files_only"])
        self.assertNotIn("canary-1b-v2.nemo", lookup.call_args.kwargs["allow_patterns"])  # 6 GB that are not needed

    def test_a_checkpoint_that_is_not_cached_is_an_error_that_names_the_command(self):
        with mock.patch("huggingface_hub.snapshot_download", side_effect=RuntimeError("no such entry")):
            with self.assertRaisesRegex(FileNotFoundError, "python -m huasr.pretrained fetch"):
                pretrained.snapshot(pretrained.QWEN)

    def test_an_incomplete_snapshot_is_an_error(self):
        folder = cache_folder(self.root, pretrained.PARAKEET)
        (folder / "tokenizer.json").unlink()
        with mock.patch("huggingface_hub.snapshot_download", return_value=str(folder)):
            with self.assertRaisesRegex(FileNotFoundError, "tokenizer.json"):
                pretrained.snapshot(pretrained.PARAKEET)

    def test_fetch_command(self):
        folder = cache_folder(self.root, pretrained.WAV2VEC2)
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(pretrained, "snapshot", return_value=folder) as fetch, \
                mock.patch.object(Wav2Vec2CTC, "checkpoint", with_weights(pretrained.WAV2VEC2)), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.assertEqual(pretrained.main(["fetch", "wav2vec2-xlsr-53-hungarian"]), 0)
            fetch.assert_called_once_with(Wav2Vec2CTC.checkpoint, download=True)
            fetch.side_effect = OSError("no network")
            self.assertEqual(pretrained.main(["fetch", "wav2vec2-xlsr-53-hungarian"]), 1)
        self.assertIn("jonatasgrosman/wav2vec2-large-xlsr-53-hungarian at 2bd0786e", out.getvalue())
        self.assertIn("no network", err.getvalue())


class DescriptionTests(TemporaryFolderTestCase):
    def test_other_weights_than_the_pinned_ones_are_refused(self):
        system = self.cached(Canary)
        (system._folder / "model.safetensors").write_bytes(b"WEIGHTS")  # same size, other content
        with self.assertRaisesRegex(ValueError, "is not the weight file of nvidia/canary-1b-v2"):
            pretrained.published_weights(Canary.checkpoint, system._folder)
        (system._folder / "model.safetensors").write_bytes(b"longer than pinned")
        with self.assertRaisesRegex(ValueError, "18 bytes, SHA-256 not computed"):
            pretrained.published_weights(Canary.checkpoint, system._folder)

    def test_a_folder_of_another_revision_is_refused(self):
        system = self.cached(ParakeetTDT)
        other = system._folder.with_name("0" * 40)
        system._folder.rename(other)
        system._folder = other
        with self.assertRaisesRegex(ValueError, "is not nvidia/parakeet-tdt-0.6b-v3 at 541d1f99"):
            system.describe()

    def test_list_command_prints_every_system_without_touching_a_model(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(pretrained.main(["list"]), 0)
        listed = json.loads(out.getvalue())
        self.assertEqual(tuple(listed), NAMES)
        self.assertEqual((listed["canary-1b-v2"]["source_language"], listed["canary-1b-v2"]["target_language"],
                          listed["canary-1b-v2"]["window_seconds"], listed["qwen3-asr-1.7b"]["language"]),
                         ("hu", "hu", 40, "Hungarian"))
        for name, entry in listed.items():
            self.assertEqual((entry["group"], entry["revision"]), ("baselines", CLASSES[name].checkpoint.revision))

    @needs("torch")
    def test_run_manifest_names_checkpoint_revision_weights_and_settings(self):
        for cls in CLASSES.values():
            with self.subTest(cls.__name__):
                system = self.cached(cls)
                description = system.describe()
                checkpoint = description["checkpoint"]
                self.assertEqual(checkpoint, {
                    "repository": cls.checkpoint.repository, "revision": cls.checkpoint.revision,
                    "weights_file": cls.checkpoint.weights, "weights_bytes": 7,
                    "weights_sha256": hashlib.sha256(b"weights").hexdigest()})
                self.assertEqual(report._checkpoint(checkpoint),
                                 f"`{cls.checkpoint.repository}` at `{cls.checkpoint.revision[:12]}`")
                settings = description["settings"]
                self.assertEqual((settings["implementation"], settings["dtype"], settings["fine_tuned"]),
                                 (cls.implementation, cls.dtype, False))
                self.assertEqual(settings["window_seconds"], cls.window_seconds)
                self.assertIn("no other change", settings["text"])
                self.assertIn(settings["device"], ("cpu", "cuda"))
                for key, value in cls.settings.items():
                    self.assertEqual(settings[key], value)
                self.assertEqual((description["seed"], set(description["packages"])),
                                 (0, {"torch", "transformers", "tokenizers", "safetensors", "huggingface_hub"}))
                json.dumps(description)  # everything can go into run.json

    @needs("torch")
    def test_run_manifest_says_what_happens_to_audio_longer_than_the_window(self):
        self.assertIn("whole clip", self.cached(ParakeetTDT).describe()["settings"]["longer_audio"])
        self.assertIn("fewest equal parts", self.cached(Canary).describe()["settings"]["longer_audio"])


class Batch(dict):
    """Stand-in for the `BatchFeature` a processor returns: a dict that is moved to a device and a dtype."""

    def to(self, *args, **kwargs):
        self.moved = (args, kwargs)
        return self


@needs("torch")
class Wav2Vec2Tests(unittest.TestCase):
    """The checkpoint writes upper-case letters; `|` stands for the space and id 0 is the CTC blank."""

    VOCABULARY = {0: "", 4: "|", 5: "J", 6: "Ó", 7: "N", 8: "A", 9: "P", 10: "O", 11: "T"}

    def setUp(self):
        import torch

        frames = [5, 5, 0, 6, 4, 4, 7, 8, 0, 9, 10, 10, 11, 0]  # J J _ Ó | | N A _ P O O T _
        logits = torch.nn.functional.one_hot(torch.tensor([frames]), num_classes=41).float()
        self.calls = []

        def feature_extractor(samples, *, sampling_rate, return_tensors, padding):
            self.calls.append(("features", len(samples), sampling_rate, return_tensors, padding, samples.dtype))
            return {"input_values": torch.zeros(1, len(samples)), "attention_mask": torch.ones(1, len(samples))}

        def model(input_values, *, attention_mask):
            self.calls.append(("model", tuple(input_values.shape), tuple(attention_mask.shape)))
            return SimpleNamespace(logits=logits)

        def batch_decode(ids):
            merged = [[token for index, token in enumerate(row) if index == 0 or token != row[index - 1]]
                      for row in ids.tolist()]
            return ["".join(self.VOCABULARY[token] for token in row).replace("|", " ") for row in merged]

        self.system = Wav2Vec2CTC("wav2vec2-xlsr-53-hungarian")
        self.system._processor, self.system._model, self.system._device = feature_extractor, model, "cpu"
        self.system._tokenizer = SimpleNamespace(batch_decode=batch_decode)

    def test_the_best_token_of_every_frame_is_decoded_as_ctc(self):
        text = self.system.transcribe(np.zeros(8_000, dtype=np.float32), RATE)
        self.assertEqual(text, "JÓ NAPOT")                      # exactly what the model wrote
        self.assertEqual(normalize(text), "jó napot")            # case is left to the scoring policy
        self.assertEqual(self.calls, [("features", 8_000, 16_000, "pt", True, np.float32),
                                      ("model", (1, 8_000), (1, 8_000))])

    def test_a_long_clip_is_given_to_the_model_whole(self):
        self.system.transcribe(np.zeros(920_000, dtype=np.float32), RATE)  # 57.5 s, the longest validation clip
        self.assertEqual([call[1] for call in self.calls if call[0] == "features"], [920_000])

    def test_other_sample_rates_are_refused(self):
        with self.assertRaisesRegex(ValueError, "16 kHz"):
            self.system.transcribe(np.zeros(8_000, dtype=np.float32), 8_000)
        self.assertEqual(self.calls, [])


@needs("torch")
class CanaryTests(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch, self.requests, self.generated, self.prompt = torch, [], [], list(CANARY_PROMPT)
        ids = {token: 100 + index for index, token in enumerate(dict.fromkeys([*CANARY_PROMPT, "<|en|>", "<|de|>"]))}
        tokens = {index: token for token, index in ids.items()}
        test = self

        class Processor:
            tokenizer = SimpleNamespace(convert_ids_to_tokens=lambda row: [tokens[index] for index in row])

            def apply_transcription_request(self, *, audio, source_language, target_language, punctuation):
                test.requests.append((len(audio), audio.dtype, source_language, target_language, punctuation))
                return Batch(input_features=torch.zeros(1, 10, 128), attention_mask=torch.ones(1, 10),
                             decoder_input_ids=torch.tensor([[ids[token] for token in test.prompt]]))

            def decode(self, generated, *, skip_special_tokens):
                assert skip_special_tokens and tuple(generated.shape) == (1, 12)
                return [f" Jó napot, {len(test.generated)}. rész! "]

        class Model:
            config = SimpleNamespace(decoder_config=SimpleNamespace(max_position_embeddings=1024))

            def generate(self, *, input_features, attention_mask, decoder_input_ids, max_new_tokens):
                test.generated.append(max_new_tokens)
                return torch.zeros(1, 12, dtype=torch.long)

        self.system = Canary("canary-1b-v2")
        self.system._processor, self.system._model, self.system._device = Processor(), Model(), "cpu"

    def test_source_and_target_language_are_both_hungarian(self):
        text = self.system.transcribe(np.zeros(16_000, dtype=np.float64), RATE)
        self.assertEqual(text, "Jó napot, 1. rész!")  # punctuation and case are left to the scoring policy
        self.assertEqual(self.requests, [(16_000, np.float32, "hu", "hu", True)])
        self.assertEqual(self.generated, [1024 - 9])  # every position of the decoder that the prompt leaves free

    def test_a_prompt_with_another_language_stops_the_run_before_the_model_is_called(self):
        for wrong in (["<|en|>", "<|en|>"], ["<|hu|>", "<|en|>"], ["<|en|>", "<|hu|>"], ["<|hu|>"]):
            self.prompt = [*CANARY_PROMPT[:3], *wrong, *CANARY_PROMPT[5:]]
            with self.assertRaisesRegex(ValueError, "not Hungarian twice"):
                self.system.transcribe(np.zeros(16_000, dtype=np.float32), RATE)
        self.assertEqual(self.generated, [])

    def test_a_clip_longer_than_forty_seconds_is_cut_into_equal_parts_and_joined(self):
        text = self.system.transcribe(np.zeros(920_000, dtype=np.float32), RATE)  # 57.5 s
        self.assertEqual([length for length, *_ in self.requests], [460_000, 460_000])
        self.assertEqual(text, "Jó napot, 1. rész! Jó napot, 2. rész!")

    def test_a_clip_of_up_to_forty_seconds_is_not_cut(self):
        self.system.transcribe(np.zeros(40 * RATE, dtype=np.float32), RATE)
        self.system.transcribe(np.zeros(35 * RATE, dtype=np.float32), RATE)  # Whisper would need two windows
        self.assertEqual([length for length, *_ in self.requests], [640_000, 560_000])


@needs("torch")
class ParakeetTests(unittest.TestCase):
    def setUp(self):
        import torch

        self.calls, test = [], self

        class Processor:
            def __call__(self, samples, *, sampling_rate):  # there is no language argument to give
                test.calls.append(("features", len(samples), sampling_rate))
                test.batch = Batch(input_features=torch.zeros(1, 7, 128), attention_mask=torch.ones(1, 7))
                return test.batch

            def decode(self, sequences, *, skip_special_tokens):
                assert skip_special_tokens
                return [" Köszönöm, elnök úr. "]

        class Model:
            dtype = torch.float32

            def generate(self, *, input_features, attention_mask, return_dict_in_generate):
                test.calls.append(("generate", return_dict_in_generate))
                return SimpleNamespace(sequences=torch.zeros(1, 5, dtype=torch.long))

        self.torch = torch
        self.system = ParakeetTDT("parakeet-tdt-0.6b-v3")
        self.system._processor, self.system._model, self.system._device = Processor(), Model(), "cpu"

    def test_the_whole_clip_is_decoded_and_special_tokens_are_removed(self):
        text = self.system.transcribe(np.zeros(920_000, dtype=np.float32), RATE)
        self.assertEqual(text, "Köszönöm, elnök úr.")
        self.assertEqual(self.calls, [("features", 920_000, 16_000), ("generate", True)])
        self.assertEqual(self.batch.moved, (("cpu",), {"dtype": self.torch.float32}))


@needs("torch")
class QwenTests(unittest.TestCase):
    # Stand-in for the tokens of the prompt: a system turn, the turn with the audio, the opening of the answer.
    PROMPT = ["<|im_start|>", "system\n", "<|im_end|>", "\n", "<|im_start|>", "user\n", "<|audio_start|>",
              *["<|audio_pad|>"] * 26, "<|audio_end|>", "<|im_end|>", "\n", "<|im_start|>", "answer\n"]

    def setUp(self):
        import torch

        self.torch, self.requests, self.generated, self.decoded, test = torch, [], [], [], self
        self.prefill = ["language", " Hungarian", "<asr_text>"]  # the tokens of the forced beginning of the answer

        class Tokenizer:
            def decode(self, ids):
                return "".join(test.pieces[int(token)] for token in ids)

        class Processor:
            tokenizer = Tokenizer()

            def apply_transcription_request(self, *, audio, language):
                test.requests.append((len(audio), language))
                test.pieces = [*QwenTests.PROMPT, *test.prefill]
                length = len(test.pieces)
                return Batch(input_ids=torch.arange(length)[None, :], attention_mask=torch.ones(1, length),
                             input_features=torch.zeros(1, 128, 100), input_features_mask=torch.ones(1, 100))

            def decode(self, generated, *, skip_special_tokens):  # no return_format: no repetition clean-up
                assert skip_special_tokens
                test.decoded.append(tuple(generated.shape))
                return [" jó napot kívánok "]

        class Model:
            dtype = torch.bfloat16

            def generate(self, *, input_ids, attention_mask, input_features, input_features_mask, max_new_tokens):
                test.generated.append(max_new_tokens)
                return torch.cat([input_ids, torch.full((1, 6), 7)], dim=1)

        self.system = Qwen3ASR("qwen3-asr-1.7b")
        self.system._processor, self.system._model, self.system._device = Processor(), Model(), "cpu"

    def test_hungarian_is_forced_and_only_the_new_tokens_are_decoded(self):
        text = self.system.transcribe(np.zeros(920_000, dtype=np.float32), RATE)
        self.assertEqual(text, "jó napot kívánok")
        self.assertEqual(self.requests, [(920_000, "hu")])                 # the whole clip, with the language
        self.assertEqual((self.generated, self.decoded), ([1024], [(1, 6)]))  # the prompt is cut off before decoding

    def test_a_prompt_without_the_hungarian_prefix_stops_the_run_before_the_model_is_called(self):
        for wrong in ([], ["language", " English", "<asr_text>"], ["language", " Hungarian"]):
            self.prefill = wrong
            with self.assertRaisesRegex(ValueError, "does not end with 'language Hungarian<asr_text>'"):
                self.system.transcribe(np.zeros(16_000, dtype=np.float32), RATE)
        self.assertEqual(self.generated, [])


class Loaded:
    """Stand-in for a model class of `transformers`: remembers how it was loaded and what was not found."""

    missing, mismatched = set(), set()
    calls = []

    def __init__(self):
        self.moves = []

    @classmethod
    def from_pretrained(cls, folder, *, local_files_only, dtype, output_loading_info):
        assert local_files_only and output_loading_info
        cls.calls.append((cls.__name__, folder, str(dtype)))
        return cls(), {"missing_keys": set(cls.missing), "unexpected_keys": set(),
                       "mismatched_keys": set(cls.mismatched), "error_msgs": []}

    def to(self, device):
        self.moves.append(device)
        return self

    def eval(self):
        self.moves.append("eval")
        return self


@needs("torch")
class LoadTests(TemporaryFolderTestCase):
    MODEL_CLASSES = {Wav2Vec2CTC: "Wav2Vec2ForCTC", Canary: "CanaryForConditionalGeneration",
                     ParakeetTDT: "ParakeetForTDT", Qwen3ASR: "Qwen3ASRForConditionalGeneration"}

    def setUp(self):
        super().setUp()
        self.library = mock.MagicMock()
        for name in self.MODEL_CLASSES.values():
            setattr(self.library, name, type(name, (Loaded,), {"missing": set(), "mismatched": set()}))
        Loaded.calls = []
        original = sys.modules.get("transformers")
        sys.modules["transformers"] = self.library  # only this entry is swapped, and put back afterwards
        self.addCleanup(lambda: sys.modules.pop("transformers") if original is None
                        else sys.modules.__setitem__("transformers", original))
        device = mock.patch.object(pretrained, "accelerator", return_value={"device": "cpu", "gpu": None})
        device.start()
        self.addCleanup(device.stop)

    def test_every_model_is_loaded_from_its_cache_folder_in_its_own_precision(self):
        for cls, model_class in self.MODEL_CLASSES.items():
            system = self.cached(cls)
            system.load()
            system.load()  # the second call loads nothing
            self.assertEqual(system._model.moves, ["cpu", "eval"])
            self.assertEqual(Loaded.calls[-1], (model_class, str(system._folder), f"torch.{cls.dtype}"))
        self.assertEqual(len(Loaded.calls), 4)
        for loader in (self.library.Wav2Vec2FeatureExtractor, self.library.Wav2Vec2CTCTokenizer):
            loader.from_pretrained.assert_called_once()
        self.assertEqual(self.library.AutoProcessor.from_pretrained.call_count, 3)
        for call in self.library.AutoProcessor.from_pretrained.call_args_list:
            self.assertEqual(call.kwargs, {"local_files_only": True})

    def test_a_model_with_weights_that_are_not_in_the_checkpoint_is_refused(self):
        self.library.ParakeetForTDT.missing = {"decoder.prediction.weight", "joint.head.bias"}
        with self.assertRaisesRegex(ValueError, "2 weights of ParakeetForTDT were not found .* would be random"):
            self.cached(ParakeetTDT).load()
        self.library.CanaryForConditionalGeneration.mismatched = {("proj_out.weight", (1, 2), (3, 4))}
        with self.assertRaisesRegex(ValueError, "proj_out.weight"):
            self.cached(Canary).load()

    def test_the_mask_vector_of_wav2vec2_is_not_needed_for_recognition(self):
        self.library.Wav2Vec2ForCTC.missing = {"wav2vec2.masked_spec_embed"}
        self.cached(Wav2Vec2CTC).load()
        self.library.Wav2Vec2ForCTC.missing = {"wav2vec2.masked_spec_embed", "lm_head.weight"}
        with self.assertRaisesRegex(ValueError, "lm_head.weight"):
            self.cached(Wav2Vec2CTC).load()


@needs("av")
@needs("torch")
class RunnerTests(TemporaryFolderTestCase):
    """Every system through the real runner and scorer on generated data; loading and decoding are stand-ins."""

    @classmethod
    def setUpClass(cls):
        from huasr import clean, subset
        from tests.test_subset import TARGETS, make_rows

        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.data = Path(cls.temporary.name)
        rows = make_rows()
        subset.build_voxpopuli(None, cls.data / "raw", TARGETS, check_source=False,
                               rows_of=lambda split: iter(rows[split]))
        clean.clean_voxpopuli(cls.data / "raw", cls.data / "clean")

    def test_run_record_scores_and_table_row_of_every_system(self):
        for name, cls in CLASSES.items():
            with self.subTest(name):
                system = self.cached(cls)
                with mock.patch.object(cls, "_load", lambda self, folder: setattr(self, "_model", object())), \
                        mock.patch.object(cls, "_transcribe", lambda self, samples: f"{len(samples)} minta"):
                    folder = evaluate.run(system, "validation", self.data / "raw", self.data / "clean",
                                          self.root / name / "results")
                self.assertEqual(folder, self.root / name / "results" / "baselines" / name / "validation")
                run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
                self.assertEqual((run["system"], run["group"], run["complete"]), (name, "baselines", True))
                self.assertEqual((run["checkpoint"]["repository"], run["checkpoint"]["revision"]),
                                 (cls.checkpoint.repository, cls.checkpoint.revision))
                self.assertEqual(run["settings"]["implementation"], cls.implementation)
                self.assertIn("transformers", run["packages"])
                scores = score.score_run(folder)
                self.assertEqual(scores["checkpoint"]["revision"], cls.checkpoint.revision)
                table = report.table_results(self.root / name, "baselines")
                self.assertIn(f"| {name} | `{cls.checkpoint.repository}` at `{cls.checkpoint.revision[:12]}` | "
                              f"validation |", table)
                self.assertEqual(report.table_results(self.root / name, "own"), "_No complete run yet._")


class DocumentationTests(unittest.TestCase):
    def pins(self, name: str) -> dict:
        lines = (REPOSITORY / name).read_text(encoding="utf-8").split("\n")
        return dict(line.split("==") for line in lines if line and not line.startswith("#"))

    def test_the_packages_are_pinned_to_the_versions_of_the_fine_tuning_runs(self):
        pinned, training = self.pins("requirements-pretrained.txt"), self.pins("requirements-train.txt")
        self.assertEqual(set(pinned), {"torch", "transformers", "tokenizers", "safetensors", "huggingface_hub"})
        for package, version in pinned.items():
            self.assertEqual(version, training[package], package)  # one set of versions: nothing conflicts
        self.assertNotIn("nemo", " ".join(pinned))
        self.assertNotIn("qwen-asr", pinned)

    def test_the_readme_names_every_checkpoint_with_its_revision(self):
        text = (REPOSITORY / "README.md").read_text(encoding="utf-8")
        search = "https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&language=hu&search=hubert"
        expected = ["HuBERT + CTC", search]
        for name, cls in CLASSES.items():
            expected += [f"`{name}`", f"`{cls.checkpoint.repository}` at `{cls.checkpoint.revision}`"]
        self.assertEqual([part for part in expected if part not in text], [], "missing from README.md")


if __name__ == "__main__":
    unittest.main()
