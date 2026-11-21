"""The training command from start to end with a stand-in for `transformers` (tests/fake_transformers.py).

No model is trained. What is checked is everything around it: which clips are read, what the trainer is given,
which checkpoint is kept, what the training record says, and that an interrupted, a finished and a trial run are
handled as documented.
"""

import contextlib
import csv
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from huasr import audio, finetune, whisper_hf
from huasr.whisper_hf import FINETUNED, RECORD, REVISION, WEIGHTS, HuggingFaceWhisper
from tests import fake_transformers, needs
from tests.fixtures import quantise, signal
from tests.test_whisper_hf import published_folder

COLUMNS = ("clip_id", "split", "speaker_id", "duration_ms", "english_number_words", "reference", "file", "sha256")
# clip id, split, speaker, duration in ms as the manifest states it, flag for English number words, reference
CLIPS = [(f"t{number}", "train", f"s{number % 3}", 400, 0, f"Ez a {number}. mondat, jó napot!") for number in range(9)]
CLIPS += [("flagged1", "train", "s0", 400, 2, "two thousand and eight"), ("flagged2", "train", "s1", 400, 1, "one"),
          ("long", "train", "s2", 31_000, 0, "ez túl hosszú"), ("empty", "train", "s2", 400, 0, "..."),
          ("v0", "validation", "s7", 400, 0, "jó napot kívánok mindenkinek"),
          ("v1", "validation", "s7", 400, 1, "two ezer nyolc volt akkor"),
          ("v2", "validation", "s8", 400, 0, "köszönöm szépen elnök úr"),
          ("v3", "validation", "s8", 57_500, 0, "ez a felvétel nem fér bele"),
          ("x0", "test", "s9", 400, 0, "ezt senki sem olvashatja")]
USED = {"train": 9, "validation": 3}


@needs("torch")
@needs("av")
class TrainingRunTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.clean, self.sound, self.work = self.root / "clean", self.root / "audio", self.root / "work"
        self.results = self.root / "results"
        rows = []
        for number, (clip_id, split, speaker, duration, flag, reference) in enumerate(CLIPS):
            row = {"clip_id": clip_id, "split": split, "speaker_id": speaker, "duration_ms": duration,
                   "english_number_words": flag, "reference": reference, "file": f"{clip_id}.flac", "sha256": "0" * 64}
            if clip_id[0] in "tv" and duration < 30_000:  # the audio of every other clip does not even exist
                data = audio.encode_flac(quantise(signal(0.4, seed=number)), 16_000)
                (self.sound / split).mkdir(parents=True, exist_ok=True)
                (self.sound / split / row["file"]).write_bytes(data)
                row["sha256"] = hashlib.sha256(data).hexdigest()
            rows.append(row)
        self.clean.mkdir()
        with (self.clean / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

        trainer = fake_transformers.Seq2SeqTrainer
        trainer.dropped, trainer.stop_after, trainer.instances = {0: 2, 4: 1, 8: 0, 12: 1}, None, []
        # Only these two entries are swapped: restoring all of sys.modules would unload what the run imports.
        for name in ("transformers", "transformers.trainer_callback"):
            self.addCleanup(self.restore_module, name, sys.modules.get(name))
            sys.modules[name] = fake_transformers
        for patch in (
                mock.patch.dict(os.environ),
                mock.patch.object(whisper_hf, "snapshot", return_value=published_folder(self.root)),
                mock.patch.object(whisper_hf, "accelerator", return_value={"device": "cpu", "gpu": None}),
                mock.patch.multiple(whisper_hf, WEIGHTS_BYTES=7, WEIGHTS_SHA256=hashlib.sha256(b"weights").hexdigest()),
                mock.patch.dict(finetune.HYPERPARAMETERS, {"max_steps": 12, "evaluate_every_steps": 4, "batch_size": 4,
                                                           "warmup_steps": 2, "log_every_steps": 2,
                                                           "eval_batch_size": 2})):
            patch.start()
            self.addCleanup(patch.stop)

    @staticmethod
    def restore_module(name, module):
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module

    def run_train(self, *flags, cpu=True):
        out, err = io.StringIO(), io.StringIO()
        arguments = ["train", "--clean", self.clean, "--audio", self.sound, "--work", self.work, "--results",
                     self.results, "--workers", "0", *(["--allow-cpu"] if cpu else []), *flags]
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = finetune.main([str(argument) for argument in arguments])
        return code, out.getvalue() + err.getvalue()

    def record(self, folder=None):
        return json.loads(((folder or self.work) / "best" / RECORD).read_text(encoding="utf-8"))

    def test_the_best_checkpoint_is_kept_and_the_record_says_what_was_done(self):
        code, printed = self.run_train()
        self.assertEqual(code, 0, printed)
        record = self.record()
        public = self.results / "own" / "hf-whisper-small-finetuned" / RECORD
        self.assertEqual(json.loads(public.read_text(encoding="utf-8")), record)
        self.assertEqual((record["system"], record["group"], record["complete"], record["smoke"], record["resumed"]),
                         (FINETUNED, "own", True, False, False))
        self.assertEqual([row["step"] for row in record["evaluations"]], [0, 4, 8, 12])
        self.assertEqual((record["best"]["step"], record["best"]["wer"]), (8, 0.0))
        self.assertGreater(record["evaluations"][0]["wer"], record["evaluations"][1]["wer"])
        self.assertEqual([row["step"] for row in record["training_log"]], [2, 4, 6, 8, 10, 12])
        self.assertEqual(record["base_checkpoint"]["repository"], "openai/whisper-small")
        self.assertEqual(record["base_checkpoint"]["revision"], REVISION)
        self.assertEqual((record["seed"], record["target_text"]), (0, "huasr.text.normalize(reference)"))
        self.assertEqual((record["hyperparameters"]["max_steps"], record["hyperparameters"]["learning_rate"],
                          record["hyperparameters"]["mixed_precision"]), (12, 1e-5, "none"))
        self.assertEqual(record["hardware"]["device"], "cpu")
        self.assertIn("transformers", record["packages"])
        self.assertIn("best checkpoint: step 8", printed)

        best = self.work / "best"
        self.assertEqual((best / WEIGHTS).read_bytes(), b"weights at step 8")
        self.assertEqual(record["weights"]["sha256"], hashlib.sha256(b"weights at step 8").hexdigest())
        self.assertTrue(json.loads((best / "config.json").read_text(encoding="utf-8"))["use_cache"])
        self.assertTrue((best / "tokenizer.json").is_file() and (best / "preprocessor_config.json").is_file())
        self.assertEqual(sorted(path.name for path in (self.work / "checkpoints").iterdir()),
                         ["checkpoint-12", "checkpoint-8"])

    def test_left_out_clips_are_counted_and_never_read(self):
        code, printed = self.run_train()
        self.assertEqual(code, 0, printed)
        data = self.record()["data"]
        self.assertEqual(data["train"]["excluded"], {"english_number_words": 2, "longer_than_window": 1,
                                                     "empty_target": 1, "too_many_label_tokens": 0})
        self.assertEqual(data["validation"]["excluded"], {"english_number_words": None, "longer_than_window": 1,
                                                          "empty_target": 0, "too_many_label_tokens": 0})
        self.assertEqual((data["train"]["clips_in_split"], data["train"]["clips"], data["clips_read"]),
                         (13, 9, USED))
        self.assertNotIn("test", data)
        self.assertIn("train: 13 clips in the split; left out: english number words 2, longer than window 1", printed)
        self.assertEqual(data["clean_manifest_sha256"],
                         hashlib.sha256((self.clean / "manifest.csv").read_bytes()).hexdigest())

    def test_the_trainer_gets_the_fixed_settings_and_batches_of_the_right_form(self):
        self.assertEqual(self.run_train()[0], 0)
        trainer = fake_transformers.Seq2SeqTrainer.instances[0]
        arguments = vars(trainer.args)
        self.assertEqual((arguments["learning_rate"], arguments["seed"], arguments["metric_for_best_model"],
                          arguments["greater_is_better"], arguments["eval_on_start"], arguments["fp16"]),
                         (1e-5, 0, "wer", False, True, False))
        self.assertEqual(Path(arguments["output_dir"]), self.work / "checkpoints")
        self.assertEqual(len(trainer.batches), 12)
        for features, labels in trainer.batches:
            self.assertEqual((features, labels[0]), ((4, 80, 3000), 4))
        self.assertEqual(len(trainer.eval_dataset), USED["validation"])
        self.assertEqual(trainer.data_collator.decoder_start_token_id, fake_transformers.START)
        checkpoint = json.loads((self.work / "checkpoints" / "checkpoint-8" / "config.json").read_text("utf-8"))
        self.assertEqual((checkpoint["language"], checkpoint["forced_decoder_ids"]), ("hu", None))

    def test_a_smaller_batch_gives_the_same_number_of_updates(self):
        code, printed = self.run_train("--batch-size", "2")
        self.assertEqual(code, 0, printed)
        trainer = fake_transformers.Seq2SeqTrainer.instances[0]
        self.assertEqual((len(trainer.batches), trainer.batches[0][0]), (24, (2, 80, 3000)))
        values = self.record()["hyperparameters"]
        self.assertEqual((values["batch_size"], values["gradient_accumulation_steps"], values["max_steps"]), (2, 2, 12))

    def test_the_evaluation_system_reads_the_model_and_its_record(self):
        self.assertEqual(self.run_train()[0], 0)
        checkpoint = HuggingFaceWhisper(FINETUNED, "own", self.work / "best").checkpoint()
        self.assertTrue(checkpoint["description"].endswith("fine-tuned on the training split (step 8 of 12)"))
        self.assertEqual((checkpoint["best_step"], checkpoint["selection_wer"], checkpoint["training_complete"]),
                         (8, 0.0, True))
        self.assertEqual(checkpoint["weights_sha256"], self.record()["weights"]["sha256"])

    def test_a_smoke_run_is_marked_and_stays_out_of_the_results(self):
        code, printed = self.run_train("--smoke", cpu=False)
        self.assertEqual(code, 0, printed)
        record = self.record(self.root / "work-smoke")
        self.assertEqual((record["complete"], record["smoke"], record["hyperparameters"]["max_steps"]),
                         (False, True, 8))
        self.assertEqual([row["step"] for row in record["evaluations"]], [0, 4, 8])
        self.assertFalse(self.results.exists() or self.work.exists())
        self.assertIn("smoke run finished", printed)
        self.assertIn("the full run would take about", printed)
        system = HuggingFaceWhisper(FINETUNED, "own", self.root / "work-smoke" / "best")
        self.assertIn("trial run", system.checkpoint()["description"])

    def test_an_interrupted_run_is_continued_from_its_last_checkpoint(self):
        fake_transformers.Seq2SeqTrainer.stop_after = 8
        with self.assertRaises(KeyboardInterrupt):
            self.run_train()
        self.assertFalse((self.work / "best").exists() or self.results.exists())
        fake_transformers.Seq2SeqTrainer.stop_after = None
        code, printed = self.run_train()
        self.assertEqual(code, 1)
        self.assertIn("--resume", printed)
        code, printed = self.run_train("--resume")
        self.assertEqual(code, 0, printed)
        record = self.record()
        self.assertEqual((record["resumed"], record["complete"], record["best"]["step"]), (True, True, 8))
        self.assertEqual([row["step"] for row in record["evaluations"]], [0, 4, 8, 12])
        self.assertFalse(vars(fake_transformers.Seq2SeqTrainer.instances[-1].args)["eval_on_start"])

    def test_a_finished_run_is_only_replaced_with_force(self):
        self.assertEqual(self.run_train()[0], 0)
        fake_transformers.Seq2SeqTrainer.dropped = {0: 2, 4: 0, 8: 1, 12: 1}
        self.assertEqual(self.run_train()[0], 1)
        code, printed = self.run_train("--resume")
        self.assertEqual((code, self.record()["best"]["step"]), (0, 8))
        self.assertIn("nothing to do", printed)
        self.assertEqual(self.run_train("--force")[0], 0)
        self.assertEqual(self.record()["best"]["step"], 4)

    def test_the_real_run_needs_a_gpu(self):
        code, printed = self.run_train(cpu=False)
        self.assertEqual(code, 1)
        self.assertIn("no GPU", printed)
        self.assertFalse(self.results.exists())

    def test_labels_that_do_not_match_the_prompt_of_the_model_stop_the_run(self):
        with mock.patch.dict(fake_transformers.WhisperTokenizer.special, {"<|hu|>": 50259}):
            code, printed = self.run_train()
        self.assertEqual(code, 1)
        self.assertIn("the model is prompted with", printed)


if __name__ == "__main__":
    unittest.main()
