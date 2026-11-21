"""Tests for the parts of the fine-tuning code that need no model, no GPU and no download: target text,
choice of clips, settings, dataset, collator, selection metric and the choice of the best checkpoint."""

import contextlib
import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from huasr import audio, finetune, report, whisper_hf
from huasr.finetune import (IGNORE, MAX_LABEL_TOKENS, ClipDataset, Collator, SelectionMetrics, best_evaluation,
                            evaluations, hyperparameters, rates, select, target_text, trainer_arguments)
from huasr.text import normalize
from tests import REPOSITORY, needs
from tests.fixtures import RATE, quantise, signal

START, HUNGARIAN, TRANSCRIBE, NO_TIMESTAMPS, END = 50258, 50286, 50359, 50363, 50257
PREFIX = [START, HUNGARIAN, TRANSCRIBE, NO_TIMESTAMPS]


def clip(clip_id, split="train", duration_ms=5_000, number_words=0, reference="jó napot kívánok.", speaker="s1"):
    return {"clip_id": clip_id, "split": split, "speaker_id": speaker, "duration_ms": str(duration_ms),
            "english_number_words": str(number_words), "reference": reference, "file": f"{clip_id}.flac"}


class TargetTextTests(unittest.TestCase):
    def test_target_is_the_reference_under_the_scoring_policy(self):
        self.assertEqual(target_text("Jó napot, kívánok!"), "jó napot kívánok")
        self.assertEqual(target_text("az  európai-unió 2008. évi"), "az európai unió 2008 évi")
        for reference in ("Ez ma az Európai Unió egyik legsúlyosabb problémája.", "kör, kor és kór", ""):
            self.assertEqual(target_text(reference), normalize(reference))
            self.assertEqual(target_text(target_text(reference)), target_text(reference))

    def test_diacritics_are_kept(self):
        self.assertEqual(target_text("Őrült öröm"), "őrült öröm")


class SelectTests(unittest.TestCase):
    ROWS = [
        clip("a"), clip("b", number_words=2), clip("c", duration_ms=30_001), clip("d", duration_ms=30_000),
        clip("e", duration_ms=41_000, number_words=1), clip("f", reference=" ... "), clip("g", speaker="s2"),
        clip("v1", "validation", speaker="s3"), clip("v2", "validation", number_words=3, speaker="s3"),
        clip("v3", "validation", duration_ms=57_500, speaker="s4"), clip("t1", "test", speaker="s5"),
    ]

    def test_training_clips_with_english_number_words_are_left_out_and_counted(self):
        kept, report = select(self.ROWS, "train")
        self.assertEqual([row["clip_id"] for row in kept], ["a", "d", "g"])  # in the order of the manifest
        self.assertEqual(report["excluded"], {"english_number_words": 2, "longer_than_window": 1, "empty_target": 1,
                                              "too_many_label_tokens": None})
        self.assertEqual((report["clips_in_split"], report["clips"], report["speakers"]), (7, 3, 2))
        self.assertEqual(report["clips_with_english_number_words"], 0)
        self.assertEqual(sum(count or 0 for count in report["excluded"].values()) + report["clips"],
                         report["clips_in_split"])  # every clip is used or counted under exactly one rule

    def test_clips_longer_than_the_window_are_left_out_and_all_of_them_are_reported(self):
        _, report = select(self.ROWS, "train")
        self.assertEqual(report["longer_than_window_in_split"], 2)   # c and e
        self.assertEqual(report["excluded"]["longer_than_window"], 1)  # e is already out for its number words
        self.assertEqual(report["hours"], round(40_000 / 3_600_000, 3))

    def test_validation_keeps_the_flagged_clips_and_drops_only_what_does_not_fit(self):
        kept, report = select(self.ROWS, "validation")
        self.assertEqual([row["clip_id"] for row in kept], ["v1", "v2"])
        self.assertEqual(report["excluded"], {"english_number_words": None, "longer_than_window": 1,
                                              "empty_target": 0, "too_many_label_tokens": None})
        self.assertEqual(report["clips_with_english_number_words"], 1)

    def test_targets_with_too_many_label_tokens_are_left_out_when_the_tokenizer_is_there(self):
        rows = [clip("a"), clip("b", reference="x " * 500)]
        kept, report = select(rows, "train", label_length=lambda text: len(text.split()) + 5)
        self.assertEqual([row["clip_id"] for row in kept], ["a"])
        self.assertEqual(report["excluded"]["too_many_label_tokens"], 1)
        self.assertEqual(MAX_LABEL_TOKENS, 448)
        kept, report = select(rows, "train", label_length=lambda text: MAX_LABEL_TOKENS)  # exactly the limit fits
        self.assertEqual((len(kept), report["excluded"]["too_many_label_tokens"]), (2, 0))

    def test_the_test_split_is_refused(self):
        for split in ("test", "dev"):
            with self.assertRaisesRegex(ValueError, "not for training or checkpoint selection"):
                select(self.ROWS, split)
        self.assertEqual(finetune.SPLITS, ("train", "validation"))

    def test_counts_are_put_into_words_for_the_log(self):
        text = finetune.left_out({"english_number_words": 1286, "longer_than_window": 0, "too_many_label_tokens": None})
        self.assertEqual(text, "english number words 1,286, longer than window 0, too many label tokens (rule not "
                               "applied)")

    def test_plan_of_the_data_in_this_repository(self):
        clean = REPOSITORY / "data" / "clean" / "voxpopuli"
        if not (clean / "manifest.csv").is_file():
            self.skipTest("the clean manifest is not built")
        data = finetune.plan(clean)["data"]
        train, validation = data["train"], data["validation"]
        self.assertEqual((train["clips_in_split"], train["excluded"]["english_number_words"],
                          train["longer_than_window_in_split"], train["excluded"]["longer_than_window"],
                          train["excluded"]["empty_target"], train["clips"], train["speakers"]),
                         (4597, 286, 29, 18, 0, 4293, 20))  # the figures the README states
        self.assertEqual((validation["clips_in_split"], validation["excluded"]["longer_than_window"],
                          validation["clips"], validation["clips_with_english_number_words"]), (826, 7, 819, 53))


class SettingsTests(unittest.TestCase):
    def test_the_fixed_settings(self):
        values = hyperparameters()
        self.assertEqual((values["learning_rate"], values["warmup_steps"], values["max_steps"], values["batch_size"],
                          values["gradient_accumulation_steps"], values["mixed_precision"],
                          values["evaluate_every_steps"]), (1e-5, 200, 2000, 16, 1, "fp16", 250))
        self.assertEqual(finetune.SEED, 0)
        self.assertEqual(values["max_steps"] % values["evaluate_every_steps"], 0)  # the last step is evaluated

    def test_checkpoints_are_compared_by_wer_with_the_language_and_task_of_the_evaluation_system(self):
        self.assertEqual(finetune.SELECTION, {"metric": "wer", "decoding": {
            "language": "hu", "task": "transcribe", "return_timestamps": False, "num_beams": 1, "max_length": 225}})
        for name in ("language", "task", "return_timestamps"):
            self.assertEqual(finetune.SELECTION["decoding"][name], whisper_hf.DECODING[name])

    def test_a_smaller_batch_keeps_the_clips_per_update(self):
        values = hyperparameters(batch_size=8)
        self.assertEqual((values["batch_size"], values["gradient_accumulation_steps"]), (8, 2))
        self.assertEqual({key: value for key, value in values.items()
                          if key not in ("batch_size", "gradient_accumulation_steps")},
                         {key: value for key, value in hyperparameters().items()
                          if key not in ("batch_size", "gradient_accumulation_steps")})
        for wrong in (0, 5, 32):
            with self.assertRaisesRegex(ValueError, "must divide"):
                hyperparameters(batch_size=wrong)

    def test_a_smoke_run_only_shortens_the_run(self):
        smoke, full = hyperparameters(smoke=True), hyperparameters()
        self.assertEqual({key for key in full if smoke[key] != full[key]},
                         {"max_steps", "warmup_steps", "evaluate_every_steps", "log_every_steps"})
        self.assertLess(smoke["max_steps"], 20)
        self.assertEqual(hyperparameters(), finetune.HYPERPARAMETERS)  # the constants are not changed by a call

    def test_arguments_for_the_trainer(self):
        arguments = trainer_arguments(hyperparameters(), Path("work/checkpoints"), cuda=True, workers=2)
        self.assertEqual((arguments["learning_rate"], arguments["warmup_steps"], arguments["max_steps"],
                          arguments["per_device_train_batch_size"], arguments["lr_scheduler_type"]),
                         (1e-5, 200, 2000, 16, "linear"))
        self.assertEqual((arguments["seed"], arguments["data_seed"]), (0, 0))
        self.assertEqual((arguments["eval_strategy"], arguments["eval_steps"], arguments["save_steps"]),
                         ("steps", 250, 250))
        self.assertEqual((arguments["metric_for_best_model"], arguments["greater_is_better"],
                          arguments["load_best_model_at_end"]), ("wer", False, True))
        self.assertTrue(arguments["fp16"] and arguments["eval_on_start"])
        self.assertEqual(arguments["report_to"], "none")
        resumed = trainer_arguments(hyperparameters(batch_size=8), Path("work"), cuda=False, workers=0, first_run=False)
        self.assertEqual((resumed["fp16"], resumed["eval_on_start"], resumed["per_device_train_batch_size"],
                          resumed["gradient_accumulation_steps"]), (False, False, 8, 2))


class FakeFeatureExtractor:
    """Returns features of the shape Whisper expects and remembers what it was given."""

    def __init__(self):
        self.calls = []

    def __call__(self, samples, sampling_rate, return_tensors):
        self.calls.append((samples, sampling_rate, return_tensors))
        return {"input_features": np.full((1, 80, 3000), float(len(samples)), dtype=np.float64)}


@needs("av")
class DatasetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.audio = Path(temporary.name)
        self.rows, self.stored = [], {}
        for number, (split, seconds) in enumerate((("train", 0.5), ("train", 0.25), ("validation", 0.75))):
            row = clip(f"clip{number}", split)
            samples = quantise(signal(seconds, seed=number))
            self.write(row, samples)
            self.rows.append(row)
            self.stored[row["clip_id"]] = samples
        self.labels = [[*PREFIX, 10 + number, END] for number in range(len(self.rows))]
        self.extractor = FakeFeatureExtractor()
        self.dataset = ClipDataset(self.rows, self.labels, self.audio, self.extractor)

    def write(self, row, samples, rate=RATE):
        data = audio.encode_flac(samples, rate)
        folder = self.audio / row["split"]
        folder.mkdir(exist_ok=True)
        (folder / row["file"]).write_bytes(data)
        row["sha256"] = hashlib.sha256(data).hexdigest()

    def test_an_item_has_the_features_of_the_stored_audio_and_the_labels(self):
        self.assertEqual(len(self.dataset), 3)
        item = self.dataset[2]
        self.assertEqual((item["input_features"].shape, item["input_features"].dtype), ((80, 3000), np.float32))
        self.assertEqual(item["labels"], [*PREFIX, 12, END])
        samples, rate, tensors = self.extractor.calls[0]
        self.assertEqual((rate, tensors, samples.dtype), (16_000, "np", np.float32))
        np.testing.assert_array_equal(samples, self.stored["clip2"].astype(np.float32) / 32768.0)  # as the runner
        self.assertLessEqual(float(np.abs(samples).max()), 1.0)

    def test_audio_that_differs_from_the_manifest_is_refused(self):
        self.rows[0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.dataset[0]
        self.assertEqual(self.extractor.calls, [])

    def test_audio_that_does_not_fit_in_one_window_is_refused(self):
        with mock.patch.object(finetune, "WINDOW_SAMPLES", 3_000), self.assertRaisesRegex(ValueError, "one window"):
            self.dataset[0]                                      # 8,000 samples
        self.write(self.rows[1], self.stored["clip1"], rate=8_000)
        with self.assertRaisesRegex(ValueError, "8000 Hz"):
            self.dataset[1]

    def test_clips_of_the_test_split_are_refused(self):
        with self.assertRaisesRegex(ValueError, "training and the validation split"):
            ClipDataset([clip("t", "test")], [[*PREFIX, 1, END]], self.audio, self.extractor)
        with self.assertRaisesRegex(ValueError, "label sequences"):
            ClipDataset(self.rows, self.labels[:2], self.audio, self.extractor)


@needs("torch")
class CollatorTests(unittest.TestCase):
    def items(self, *lengths):
        return [{"input_features": np.full((80, 3000), float(number), dtype=np.float32),
                 "labels": [*PREFIX[:1], *range(100, 100 + length), END]} for number, length in enumerate(lengths)]

    def test_features_are_stacked_and_labels_padded_with_the_ignored_value(self):
        import torch

        batch = Collator(START)(self.items(3, 1, 5))
        self.assertEqual(set(batch), {"input_features", "labels"})
        self.assertEqual((tuple(batch["input_features"].shape), batch["input_features"].dtype),
                         ((3, 80, 3000), torch.float32))
        self.assertEqual([float(batch["input_features"][row, 0, 0]) for row in range(3)], [0.0, 1.0, 2.0])
        self.assertEqual((tuple(batch["labels"].shape), batch["labels"].dtype), ((3, 6), torch.long))
        self.assertEqual(batch["labels"].tolist(), [[100, 101, 102, END, IGNORE, IGNORE],
                                                    [100, END, IGNORE, IGNORE, IGNORE, IGNORE],
                                                    [100, 101, 102, 103, 104, END]])
        self.assertEqual(IGNORE, -100)

    def test_the_start_token_is_cut_off_because_the_model_adds_it(self):
        batch = Collator(START)([{"input_features": np.zeros((80, 3000), dtype=np.float32),
                                  "labels": [*PREFIX, 7, 8, END]}])
        self.assertEqual(batch["labels"].tolist(), [[HUNGARIAN, TRANSCRIBE, NO_TIMESTAMPS, 7, 8, END]])

    def test_labels_without_the_start_token_are_an_error(self):
        items = self.items(2, 2)
        items[1]["labels"] = items[1]["labels"][1:]
        with self.assertRaisesRegex(ValueError, "start token"):
            Collator(START)(items)
        items[1]["labels"] = []
        with self.assertRaisesRegex(ValueError, "start token"):
            Collator(START)(items)

    def test_a_batch_of_one_dataset_goes_through_a_data_loader(self):
        import torch

        loader = torch.utils.data.DataLoader(self.items(2, 4, 1), batch_size=2, collate_fn=Collator(START))
        shapes = [(tuple(batch["input_features"].shape), tuple(batch["labels"].shape)) for batch in loader]
        self.assertEqual(shapes, [((2, 80, 3000), (2, 5)), ((1, 80, 3000), (1, 2))])


class FakeTokenizer:
    """Knows four words and the prompt tokens of Whisper; an unknown token becomes the end token, as in Whisper."""

    pad_token_id, eos_token_id, unk_token_id = None, END, END
    words = {1: "jó", 2: "napot", 3: "Kívánok!", 4: "nem"}

    def __init__(self):
        self.decoded = None
        self.special = {"<|startoftranscript|>": START, "<|hu|>": HUNGARIAN, "<|transcribe|>": TRANSCRIBE,
                        "<|notimestamps|>": NO_TIMESTAMPS}

    def __call__(self, text, add_special_tokens):
        assert not add_special_tokens
        ids = {word: token for token, word in self.words.items()}
        return {"input_ids": [ids[word] for word in text.split()]}

    def batch_decode(self, tokens, skip_special_tokens):
        assert skip_special_tokens
        self.decoded = np.asarray(tokens)
        return [" " + " ".join(self.words[int(token)] for token in row if int(token) in self.words) for row in tokens]

    def convert_tokens_to_ids(self, tokens):
        return [self.special.get(token, self.unk_token_id) for token in tokens]


class MetricTests(unittest.TestCase):
    def test_rates_use_the_scoring_of_the_repository(self):
        result = rates(["jó napot kívánok", "nem"], ["Jó napot!", "nem nem"], [False, False])
        self.assertEqual(set(result), {"wer", "cer"})
        self.assertAlmostEqual(result["wer"], 100 * 2 / 4)       # one deletion and one insertion, four words
        self.assertAlmostEqual(result["cer"], 100 * (8 + 4) / 19)  # " kívánok" lost, " nem" added

    def test_the_rate_without_the_flagged_clips_is_given_next_to_the_rate(self):
        result = rates(["jó napot kívánok", "nem"], ["jó napot", "nem nem"], [False, True])
        self.assertAlmostEqual(result["wer"], 50.0)
        self.assertAlmostEqual(result["wer_without_english_number_words"], 100 / 3)

    def test_generated_tokens_are_decoded_and_scored_in_the_order_of_the_clips(self):
        tokenizer = FakeTokenizer()
        score = SelectionMetrics(tokenizer, ["jó napot kívánok", "nem"], [False, False])
        predictions = np.array([[START, HUNGARIAN, 1, 2, END, END], [START, HUNGARIAN, 4, END, -100, -100]])
        result = score(SimpleNamespace(predictions=predictions, label_ids=None))
        self.assertAlmostEqual(result["wer"], 25.0)                # "kívánok" is missing
        self.assertTrue((tokenizer.decoded >= 0).all())            # the filler of the trainer is not decoded
        self.assertEqual(tokenizer.decoded[1].tolist(), [START, HUNGARIAN, 4, END, END, END])
        self.assertAlmostEqual(score(SimpleNamespace(predictions=(predictions, "other output")))["wer"], 25.0)

    def test_a_wrong_number_of_hypotheses_is_an_error(self):
        score = SelectionMetrics(FakeTokenizer(), ["jó", "nem", "napot"], [False] * 3)
        with self.assertRaisesRegex(ValueError, "2 hypotheses for 3 validation clips"):
            score(SimpleNamespace(predictions=np.array([[1, END], [4, END]])))

class LabelTests(unittest.TestCase):
    def test_labels_are_the_prompt_the_text_and_the_end_token(self):
        encode = finetune.LabelEncoder(FakeTokenizer())
        self.assertEqual(finetune.PREFIX, ("<|startoftranscript|>", "<|hu|>", "<|transcribe|>", "<|notimestamps|>"))
        self.assertEqual(encode.prefix, PREFIX)
        self.assertEqual(encode("jó napot"), [*PREFIX, 1, 2, END])
        self.assertEqual(encode(target_text("Nem, nem!")), [*PREFIX, 4, 4, END])

    def test_a_tokenizer_that_lacks_a_prompt_token_is_refused(self):
        for missing in finetune.PREFIX:
            tokenizer = FakeTokenizer()
            del tokenizer.special[missing]
            with self.assertRaisesRegex(ValueError, "does not know the tokens"):
                finetune.LabelEncoder(tokenizer)

    @needs("torch")
    def test_labels_go_through_dataset_items_and_the_collator_into_what_the_model_is_given(self):
        encode = finetune.LabelEncoder(FakeTokenizer())
        items = [{"input_features": np.zeros((80, 3000), dtype=np.float32), "labels": encode(text)}
                 for text in ("jó napot", "nem")]
        labels = Collator(encode.prefix[0])(items)["labels"].tolist()
        self.assertEqual(labels, [[HUNGARIAN, TRANSCRIBE, NO_TIMESTAMPS, 1, 2, END],
                                  [HUNGARIAN, TRANSCRIBE, NO_TIMESTAMPS, 4, END, IGNORE]])


HISTORY = [
    {"eval_loss": 1.5, "eval_wer": 31.0, "eval_cer": 9.5, "eval_runtime": 200.04, "epoch": 0, "step": 0},
    {"loss": 1.25, "learning_rate": 1.25e-6, "grad_norm": 9.0, "epoch": 0.0932, "step": 25},
    {"eval_loss": 0.5, "eval_wer": 22.0, "eval_cer": 6.0, "eval_wer_without_english_number_words": 20.0,
     "eval_runtime": 180.0, "epoch": 0.93, "step": 250},
    {"loss": 0.4, "learning_rate": 9e-6, "epoch": 1.0, "step": 275},
    {"eval_loss": 0.45, "eval_wer": 20.5, "eval_cer": 5.5, "eval_runtime": 181.0, "epoch": 1.86, "step": 500},
    {"eval_loss": 0.47, "eval_wer": 20.5, "eval_cer": 5.4, "eval_runtime": 182.0, "epoch": 2.8, "step": 750},
    {"train_runtime": 4000.0, "train_loss": 0.6, "epoch": 2.8, "step": 750},
]


class HistoryTests(unittest.TestCase):
    def test_validation_scores_are_read_from_the_log_of_the_trainer(self):
        rows = evaluations(HISTORY)
        self.assertEqual([row["step"] for row in rows], [0, 250, 500, 750])
        self.assertEqual(rows[1], {"step": 250, "epoch": 0.93, "loss": 0.5, "wer": 22.0, "cer": 6.0,
                                   "wer_without_english_number_words": 20.0, "seconds": 180.0})
        self.assertIsNone(rows[0]["wer_without_english_number_words"])
        again = evaluations([*HISTORY, {"eval_wer": 19.0, "eval_cer": 5.0, "step": 500}])  # a step evaluated again
        self.assertEqual([(row["step"], row["wer"]) for row in again],
                         [(0, 31.0), (250, 22.0), (500, 19.0), (750, 20.5)])

    def test_the_lowest_wer_wins_and_of_equal_ones_the_earliest(self):
        self.assertEqual(best_evaluation(evaluations(HISTORY))["step"], 500)

    def test_the_evaluation_before_training_has_no_checkpoint_and_cannot_win(self):
        history = [{**HISTORY[0], "eval_wer": 1.0}, *HISTORY[1:]]
        self.assertEqual(best_evaluation(evaluations(history))["step"], 500)
        with self.assertRaisesRegex(ValueError, "no checkpoint was evaluated"):
            best_evaluation(evaluations(HISTORY[:2]))

    def test_training_log_has_loss_and_learning_rate_per_step(self):
        self.assertEqual(finetune.training_log(HISTORY), [
            {"step": 25, "epoch": 0.093, "loss": 1.25, "learning_rate": 1.25e-6, "grad_norm": 9.0},
            {"step": 275, "epoch": 1.0, "loss": 0.4, "learning_rate": 9e-6, "grad_norm": None}])

    def test_projection_of_a_smoke_run(self):
        rows = [{"seconds": 8.0}, {"seconds": 4.0}]              # 6 s for 16 clips
        estimate = finetune.projection([30.0, 2.0, 2.0, 2.0], rows, evaluated_clips=16, clips=819)
        self.assertEqual((estimate["seconds_per_training_step"], estimate["seconds_per_evaluated_clip"]), (2.0, 0.375))
        self.assertEqual(estimate["training_hours"], round(2000 * 2.0 / 3600, 2))
        self.assertEqual(estimate["evaluation_hours"], round(9 * 819 * 0.375 / 3600, 2))


def write_manifest(folder: Path, rows: list[dict]) -> None:
    folder.mkdir(parents=True)
    with (folder / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class ReadmeTableTests(unittest.TestCase):
    BLOCK = "<!-- generated: finetune-plan -->\nold\n<!-- end generated -->\n"

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_clips_and_settings_come_from_the_manifest_and_the_constants(self):
        write_manifest(self.root / "data" / "clean" / "voxpopuli", SelectTests.ROWS)
        text = report.render(self.BLOCK, self.root)
        self.assertIn("| train | training | 7 | 2 | 1 of 2 | 1 | 3 | 0.01 | 2 |", text)
        self.assertIn("| validation | choosing the checkpoint | 3 | kept | 1 of 1 | 0 | 2 | 0.00 | 1 |", text)
        self.assertIn("Settings: 2,000 updates of 16 clips", text)
        self.assertIn("learning rate 1e-05 after a linear warm-up of 200 updates", text)
        self.assertIn("mixed precision `fp16`; validation and a checkpoint every 250 updates; seed 0.", text)
        self.assertIn("Checkpoints are compared by validation WER, decoded with beam size 1", text)
        self.assertNotIn("old", text)
        self.assertEqual(report.render(text, self.root), text)

    def test_a_missing_manifest_is_stated_instead_of_a_table(self):
        text = report.render(self.BLOCK, self.root)
        self.assertIn("_Not built yet: `data/clean/voxpopuli/manifest.csv` does not exist._", text)

    def test_the_readme_names_the_pinned_checkpoint_and_both_systems(self):
        text = (REPOSITORY / "README.md").read_text(encoding="utf-8")
        for expected in (f"`{whisper_hf.REPOSITORY}` at revision `{whisper_hf.REVISION}`",
                         "<!-- generated: finetune-plan -->",
                         f"`{whisper_hf.PRETRAINED}` | `baselines`", f"`{whisper_hf.FINETUNED}` | `own`"):
            self.assertIn(expected, text)


class CommandTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def run_main(self, *arguments):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = finetune.main([str(argument) for argument in arguments])
        return code, out.getvalue() + err.getvalue()

    def finished_run(self, work: Path) -> Path:
        (work / "best").mkdir(parents=True)
        record = work / "best" / whisper_hf.RECORD
        record.write_text("{}", encoding="utf-8")
        return record

    def test_training_without_unpacked_audio_stops_before_anything_is_loaded(self):
        code, printed = self.run_main("train", "--audio", self.root / "audio", "--work", self.root / "work",
                                      "--results", self.root / "results")
        self.assertEqual(code, 1)
        self.assertIn("python -m huasr.subset unpack", printed)
        self.assertFalse((self.root / "results").exists())

    def test_a_finished_run_is_not_trained_again_by_accident(self):
        record = self.finished_run(self.root / "work")
        code, printed = self.run_main("train", "--work", self.root / "work", "--audio", self.root / "audio")
        self.assertEqual(code, 1)
        self.assertIn("already holds a finished run", printed)
        code, printed = self.run_main("train", "--resume", "--work", self.root / "work", "--audio", self.root / "audio")
        self.assertEqual(code, 0)
        self.assertIn("nothing to do", printed)
        self.assertTrue(record.is_file())

    def test_an_unfinished_run_needs_a_decision(self):
        (self.root / "work" / "checkpoints" / "checkpoint-250").mkdir(parents=True)
        code, printed = self.run_main("train", "--work", self.root / "work", "--audio", self.root / "audio")
        self.assertEqual(code, 1)
        self.assertIn("--resume", printed)
        self.assertTrue((self.root / "work" / "checkpoints" / "checkpoint-250").is_dir())

    def test_a_smoke_run_leaves_the_folder_of_a_real_run_alone(self):
        record = self.finished_run(self.root / "work")
        (self.root / "work-smoke" / "best").mkdir(parents=True)
        code, printed = self.run_main("train", "--smoke", "--work", self.root / "work", "--audio", self.root / "audio")
        self.assertEqual(code, 1)                                   # there is no audio; nothing was trained
        self.assertTrue(record.is_file())
        self.assertFalse((self.root / "work-smoke" / "best").exists())  # an earlier smoke run is cleared

    def test_plan_prints_the_clips_and_the_settings(self):
        clean = self.root / "clean"
        write_manifest(clean, SelectTests.ROWS)
        code, printed = self.run_main("plan", "--clean", clean)
        self.assertEqual(code, 0)
        plan = json.loads(printed)
        self.assertEqual(plan["data"]["train"]["excluded"]["english_number_words"], 2)
        self.assertEqual(plan["base_checkpoint"], {"repository": "openai/whisper-small",
                                                   "revision": whisper_hf.REVISION})
        self.assertEqual((plan["hyperparameters"], plan["seed"], plan["group"]), (hyperparameters(), 0, "own"))
        self.assertNotIn("test", plan["data"])


if __name__ == "__main__":
    unittest.main()
