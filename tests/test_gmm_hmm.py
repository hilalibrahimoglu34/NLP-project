"""Tests for the GMM-HMM baseline: lexicon, corpus folders, the steps with a stand-in for MFA, and the
conversion of MFA output. MFA itself is not run."""

import contextlib
import csv
import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from huasr import gmm_hmm, report, score
from huasr.evaluate import FinalTestRequired
from huasr.shards import write_manifest

COLUMNS = ("clip_id", "split", "speaker_id", "session_id", "duration_ms", "file", "shard", "sha256", "reference",
           "english_number_words")
# clip id, split, speaker, duration in ms, reference
CLIPS = [
    ("s-hu_20:01:02_1", "train", "11", 2000, "Jó napot, Szeged!"),
    ("s-hu_20:01:02_2", "train", "11", 1000, "A víz jó."),
    ("s-hu_21:00:00_1", "train", "12", 3000, "Szép az őszi nap."),
    ("t-hu_09:30:00_1", "train", "12", 1000, " ... "),                    # nothing left after the scoring policy
    ("v-hu_10:00:00_1", "validation", "21", 3000, "Jó az őszi víz."),
    ("v-hu_10:00:00_2", "validation", "21", 1000, "Szeged szép város!"),
    ("v-hu_11:00:00_1", "validation", "22", 4000, "A nap süt."),
    ("x-hu_12:00:00_1", "test", "31", 1000, "Ezt senki sem látja."),
]


def make_data(root: Path) -> tuple[Path, Path]:
    """A raw folder with one shard per split and a clean folder with the manifest; returns both."""
    raw, clean = root / "raw", root / "clean"
    (raw / "clips").mkdir(parents=True)
    clean.mkdir()
    rows = []
    for clip_id, split, speaker, duration, reference in CLIPS:
        name = clip_id.replace(":", "-") + ".flac"
        data = f"audio of {clip_id}".encode()
        rows.append({"clip_id": clip_id, "split": split, "speaker_id": speaker, "session_id": clip_id.split("_")[0],
                     "duration_ms": duration, "file": name, "shard": f"clips/{split}-000.tar",
                     "sha256": hashlib.sha256(data).hexdigest(), "reference": reference, "english_number_words": 0})
        with tarfile.open(raw / "clips" / f"{split}-000.tar", "a") as tar:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
    write_manifest(raw / "manifest.csv", COLUMNS, rows)
    write_manifest(clean / "manifest.csv", COLUMNS, rows)
    return raw, clean


class FakeMfa:
    """Stands in for the `mfa` command: writes the files the real one would and remembers every call."""

    def __init__(self, transcript: str = "jó <unk> víz"):
        self.calls, self.transcript = [], transcript

    def __call__(self, work: Path, name: str, arguments: list) -> float:
        self.calls.append((name, [str(argument) for argument in arguments]))
        if name == "train_lm":
            Path(arguments[1]).write_bytes(b"language model")
        elif name == "train":
            Path(arguments[2]).write_bytes(b"acoustic model")
        else:
            corpus, output = Path(arguments[0]), Path(arguments[4])
            for clip in sorted(corpus.rglob("*.flac")):
                target = output / clip.parent.name / (clip.stem + ".lab")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(self.transcript, encoding="utf-8")
        return 2.0


class TemporaryFolderTestCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)


class LexiconTests(TemporaryFolderTestCase):
    def test_every_word_is_spelled_as_its_letters(self):
        lexicon = gmm_hmm.build_lexicon(["jó napot szeged", "a víz jó"])
        self.assertEqual(lexicon["napot"], ("n", "a", "p", "o", "t"))
        self.assertEqual(list(lexicon), ["a", "jó", "napot", "szeged", "víz"])       # sorted, every word once

    def test_digraphs_stay_two_units_and_accented_letters_are_their_own_unit(self):
        lexicon = gmm_hmm.build_lexicon(["szép gyűrű őz"])
        self.assertEqual(lexicon["szép"], ("s", "z", "é", "p"))
        self.assertEqual(lexicon["gyűrű"], ("g", "y", "ű", "r", "ű"))
        self.assertEqual(lexicon["őz"], ("ő", "z"))

    def test_word_with_a_character_that_is_not_a_letter_is_refused(self):
        for text in ("a 2008 év", "e-mail", "<unk>"):
            with self.assertRaisesRegex(ValueError, "not a letter"):
                gmm_hmm.build_lexicon([text])

    def test_lexicon_file_has_one_word_per_line_and_reads_back(self):
        lexicon = gmm_hmm.build_lexicon(["jó víz"])
        gmm_hmm.write_lexicon(self.root / "lexicon.txt", lexicon)
        self.assertEqual((self.root / "lexicon.txt").read_bytes().decode("utf-8"), "jó\tj ó\nvíz\tv í z\n")
        self.assertEqual(gmm_hmm.read_lexicon(self.root / "lexicon.txt"), lexicon)

    def test_out_of_vocabulary_rate_uses_the_scoring_policy(self):
        lexicon = gmm_hmm.build_lexicon(["jó napot szeged", "a víz jó"])
        counts = gmm_hmm.out_of_vocabulary(lexicon, ["Jó az őszi víz.", "Szeged szép város!", "A víz jó."])
        self.assertEqual((counts["reference_words"], counts["not_in_lexicon"], counts["rate"]), (10, 4, 0.4))
        self.assertEqual((counts["distinct_reference_words"], counts["distinct_not_in_lexicon"]), (8, 4))
        self.assertEqual(counts["clips_with_a_word_not_in_lexicon"], 2)
        self.assertEqual(gmm_hmm.out_of_vocabulary(lexicon, [])["rate"], 0.0)


class StepTests(TemporaryFolderTestCase):
    def setUp(self):
        super().setUp()
        self.raw, self.clean = make_data(self.root)
        self.work = self.root / "work"
        with (self.clean / "manifest.csv").open(encoding="utf-8", newline="") as handle:
            self.rows = list(csv.DictReader(handle))

    def test_clips_are_stored_by_speaker_with_the_transcript_next_to_the_audio(self):
        training = [row for row in self.rows if row["split"] == "train"][:3]
        self.assertEqual(gmm_hmm.export_corpus(self.raw, training, self.root / "corpus", with_text=True), 3)
        clip = self.root / "corpus" / "11" / "s-hu_20-01-02_1.flac"
        self.assertEqual(clip.read_bytes(), b"audio of s-hu_20:01:02_1")
        self.assertEqual(clip.with_suffix(".lab").read_bytes().decode("utf-8"), "jó napot szeged\n")
        self.assertEqual(sorted(path.name for path in (self.root / "corpus").iterdir()), ["11", "12"])

    def test_clip_that_does_not_match_the_manifest_is_an_error(self):
        wrong = [{**self.rows[0], "sha256": "0" * 64}]
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            gmm_hmm.export_corpus(self.raw, wrong, self.root / "corpus", with_text=True)
        missing = [{**self.rows[0], "file": "not-there.flac"}]
        with self.assertRaisesRegex(ValueError, "missing"):
            gmm_hmm.export_corpus(self.raw, missing, self.root / "corpus", with_text=True)

    def test_prepare_writes_corpus_lexicon_and_settings_from_the_training_split_only(self):
        state = gmm_hmm.prepare(self.work, self.raw, self.clean)
        self.assertEqual(state["training"], {"clips": 3, "speakers": 2, "audio_seconds": 6.0, "words": 10,
                                             "only_speakers": None})     # the clip without text is left out
        lexicon = gmm_hmm.read_lexicon(self.work / "lexicon.txt")
        self.assertEqual(list(lexicon), ["a", "az", "jó", "nap", "napot", "szeged", "szép", "víz", "őszi"])
        self.assertNotIn("város", lexicon)                               # a word of the validation split only
        self.assertEqual([path.name for path in (self.work / "corpus").iterdir()], ["train"])
        self.assertEqual(len(list((self.work / "corpus" / "train").rglob("*.lab"))), 3)
        self.assertEqual(json.loads((self.work / "train_config.json").read_text(encoding="utf-8")),
                         gmm_hmm.TRAIN_CONFIG)
        self.assertEqual([next(iter(block)) for block in gmm_hmm.TRAIN_CONFIG["training"]],
                         ["monophone", "triphone", "lda", "sat"])
        self.assertIs(gmm_hmm.TRAIN_CONFIG["use_pitch"], False)
        with self.assertRaises(FileExistsError):
            gmm_hmm.prepare(self.work, self.raw, self.clean)
        state = gmm_hmm.prepare(self.work, self.raw, self.clean, train_speakers=["11"], force=True)
        self.assertEqual((state["training"]["clips"], state["training"]["only_speakers"]), (2, ["11"]))
        self.assertEqual([path.name for path in (self.work / "corpus" / "train").iterdir()], ["11"])

    def test_training_gives_mfa_the_training_corpus_the_lexicon_and_the_settings(self):
        gmm_hmm.prepare(self.work, self.raw, self.clean)
        tool = FakeMfa()
        state = gmm_hmm.train(self.work, jobs=3, run=tool)
        self.assertEqual([name for name, _ in tool.calls], ["train_lm", "train"])
        corpus, lexicon = str(self.work / "corpus" / "train"), str(self.work / "lexicon.txt")
        self.assertEqual(tool.calls[0][1][:4], [corpus, str(self.work / "language_model.zip"), "--dictionary_path",
                                                lexicon])
        self.assertEqual(tool.calls[1][1][:5], [corpus, lexicon, str(self.work / "acoustic_model.zip"),
                                                "--config_path", str(self.work / "train_config.json")])
        self.assertEqual((state["train"]["jobs"], state["train"]["language_model_seconds"],
                          state["train"]["acoustic_model_seconds"]), (3, 2.0, 2.0))

    def test_decoding_gives_mfa_the_audio_of_the_split_and_no_text(self):
        gmm_hmm.prepare(self.work, self.raw, self.clean)
        tool = FakeMfa()
        with self.assertRaisesRegex(ValueError, "no trained models"):
            gmm_hmm.decode(self.work, self.raw, self.clean, run=tool)
        gmm_hmm.train(self.work, run=tool)
        state = gmm_hmm.decode(self.work, self.raw, self.clean, jobs=2, run=tool)
        corpus = self.work / "corpus" / "validation"
        self.assertEqual(sorted(path.suffix for path in corpus.rglob("*.*")), [".flac"] * 3)
        self.assertEqual(sorted(path.name for path in corpus.iterdir()), ["21", "22"])      # grouped by speaker
        name, arguments = tool.calls[-1]
        self.assertEqual((name, arguments[0], arguments[4]), ("transcribe", str(corpus),
                                                              str(self.work / "output" / "validation")))
        self.assertIn("transcription", arguments)
        self.assertEqual(json.loads((self.work / "decode_config.json").read_text(encoding="utf-8")),
                         gmm_hmm.DECODE_CONFIG)
        done = state["decode"]["validation"]
        self.assertEqual((done["limit"], done["final_test"], done["jobs"], done["seconds"], done["config"]),
                         (None, False, 2, 2.0, gmm_hmm.DECODE_CONFIG))
        state = gmm_hmm.decode(self.work, self.raw, self.clean, limit=2, run=tool)          # a trial run
        self.assertEqual(len(list(corpus.rglob("*.flac"))), 2)
        self.assertEqual(state["decode"]["validation"]["limit"], 2)

    def test_test_split_is_refused_without_the_flag_and_uses_the_same_models_with_it(self):
        gmm_hmm.prepare(self.work, self.raw, self.clean)
        tool = FakeMfa()
        gmm_hmm.train(self.work, run=tool)
        gmm_hmm.decode(self.work, self.raw, self.clean, run=tool)
        calls = len(tool.calls)
        with self.assertRaises(FinalTestRequired):
            gmm_hmm.decode(self.work, self.raw, self.clean, split="test", run=tool)
        with self.assertRaisesRegex(ValueError, "split to decode"):
            gmm_hmm.decode(self.work, self.raw, self.clean, split="train", run=tool)
        self.assertEqual(len(tool.calls), calls)
        self.assertFalse((self.work / "corpus" / "test").exists())
        state = gmm_hmm.decode(self.work, self.raw, self.clean, split="test", final_test=True, run=tool)
        self.assertEqual([name for name, _ in tool.calls[calls:]], ["transcribe"])          # nothing is trained again
        self.assertEqual(sorted(state["decode"]), ["test", "validation"])
        self.assertIs(state["decode"]["test"]["final_test"], True)
        folder = gmm_hmm.collect(self.work, self.clean, self.raw, self.root / "results", split="test")
        run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        self.assertEqual((folder.name, run["split"], run["final_test"], run["clips"]), ("test", "test", True, 1))


class OutputTests(TemporaryFolderTestCase):
    def setUp(self):
        super().setUp()
        self.raw, self.clean = make_data(self.root)
        self.work = self.root / "work"
        gmm_hmm.prepare(self.work, self.raw, self.clean)
        for name in ("acoustic_model.zip", "language_model.zip"):
            (self.work / name).write_bytes(name.encode())
        output = self.work / "output" / "validation"
        (output / "21").mkdir(parents=True)
        (output / "21" / "v-hu_10-00-00_1.lab").write_text("jó az <unk> víz", encoding="utf-8")
        (output / "21" / "v-hu_10-00-00_2.lab").write_text("<unk> [laugh]\n", encoding="utf-8")
        self.decoded = {"limit": None, "final_test": False, "jobs": 2, "seconds": 4.0, "config": gmm_hmm.DECODE_CONFIG,
                        "started_utc": "2026-01-01T00:00:00Z", "finished_utc": "2026-01-01T00:00:04Z"}
        gmm_hmm._state(self.work, train={"jobs": 2, "language_model_seconds": 1.0, "acoustic_model_seconds": 60.0},
                       decode={"validation": self.decoded})

    def test_placeholders_are_removed_and_counted(self):
        self.assertEqual(gmm_hmm.clean_hypothesis("jó az <unk> víz"), ("jó az víz", 1))
        self.assertEqual(gmm_hmm.clean_hypothesis("<unk> <eps> [laugh]"), ("", 3))
        self.assertEqual(gmm_hmm.clean_hypothesis("a b"), ("a b", 0))
        self.assertEqual(gmm_hmm.clean_hypothesis(""), ("", 0))

    def test_transcripts_are_read_by_file_name(self):
        self.assertEqual(gmm_hmm.read_transcripts(self.work / "output" / "validation"),
                         {"v-hu_10-00-00_1": "jó az <unk> víz", "v-hu_10-00-00_2": "<unk> [laugh]"})

    def test_run_folder_has_the_files_of_every_other_run_and_can_be_scored(self):
        folder = gmm_hmm.collect(self.work, self.clean, self.raw, self.root / "results")
        self.assertEqual(folder, self.root / "results" / "baselines" / "gmm-hmm-mfa" / "validation")
        rows = score.read_predictions(folder)
        self.assertEqual([row["clip_id"] for row in rows],
                         ["v-hu_10:00:00_1", "v-hu_10:00:00_2", "v-hu_11:00:00_1"])      # order of the manifest
        self.assertEqual([row["hypothesis"] for row in rows], ["jó az víz", "", ""])
        self.assertEqual([row["reference"] for row in rows], ["Jó az őszi víz.", "Szeged szép város!", "A nap süt."])
        self.assertEqual([float(row["processing_seconds"]) for row in rows], [1.5, 0.5, 2.0])   # 4 s by duration
        run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        self.assertEqual((run["system"], run["group"], run["split"], run["final_test"]),
                         ("gmm-hmm-mfa", "baselines", "validation", False))
        self.assertEqual((run["clips"], run["complete"], run["processing_seconds"]), (3, True, 4.0))
        self.assertEqual(run["output"], {"clips_without_output_file": 1, "clips_with_empty_output": 0,
                                         "placeholders_removed": 3, "clips_with_placeholders": 2,
                                         "clips_with_placeholders_only": 1, "empty_hypotheses": 2})
        self.assertEqual((run["out_of_vocabulary"]["reference_words"], run["out_of_vocabulary"]["not_in_lexicon"]),
                         (10, 2))
        self.assertEqual(run["settings"]["lexicon"]["words"], 9)
        self.assertEqual(run["settings"]["training_recipe"], gmm_hmm.TRAIN_CONFIG)
        self.assertEqual((run["settings"]["decoding"]["beam"], run["settings"]["decoding"]["first_beam"]), (10, 10))
        self.assertEqual(run["checkpoint"]["acoustic_model_sha256"], hashlib.sha256(b"acoustic_model.zip").hexdigest())
        scores = score.score_run(folder)
        self.assertEqual((scores["all_clips"]["clips"], scores["all_clips"]["empty_hypotheses"]), (3, 2))
        self.assertEqual(scores["all_clips"]["wer"]["deletions"], 7)
        with self.assertRaises(FileExistsError):
            gmm_hmm.collect(self.work, self.clean, self.raw, self.root / "results")
        with self.assertRaisesRegex(ValueError, "no output for the split 'test'"):
            gmm_hmm.collect(self.work, self.clean, self.raw, self.root / "results", split="test")

    def test_trial_runs_are_marked_as_incomplete(self):
        gmm_hmm._state(self.work, decode={"validation": {**self.decoded, "limit": 2}})
        run = json.loads((gmm_hmm.collect(self.work, self.clean, self.raw, self.root / "few") / "run.json")
                         .read_text(encoding="utf-8"))
        self.assertEqual((run["clips"], run["limit"], run["complete"]), (2, 2, False))
        state = gmm_hmm._state(self.work)
        gmm_hmm._state(self.work, decode={"validation": self.decoded},
                       training={**state["training"], "only_speakers": ["11"]})
        run = json.loads((gmm_hmm.collect(self.work, self.clean, self.raw, self.root / "speakers") / "run.json")
                         .read_text(encoding="utf-8"))
        self.assertEqual((run["clips"], run["complete"]), (3, False))

    def test_package_versions_come_from_the_records_of_the_environment(self):
        records = self.root / "prefix" / "conda-meta"
        records.mkdir(parents=True)
        for name, version, build in (("kaldi", "5.5.1", "cpu_h1_5"), ("montreal-forced-aligner", "3.4.3", "py_1"),
                                     ("zlib", "1.3", "h2_0")):
            (records / f"{name}-{version}-{build}.json").write_text(
                json.dumps({"name": name, "version": version, "build": build}), encoding="utf-8")
        self.assertEqual(gmm_hmm.installed(self.root / "prefix"),
                         {"kaldi": "5.5.1 cpu_h1_5", "montreal-forced-aligner": "3.4.3 py_1"})
        self.assertEqual(gmm_hmm.installed(self.root / "nowhere"), {})

    def test_result_tables_show_the_run(self):
        folder = gmm_hmm.collect(self.work, self.clean, self.raw, self.root / "results")
        score.score_run(folder)
        row = report.table_results(self.root, "baselines").split("\n")[2]
        self.assertIn("| gmm-hmm-mfa | trained here on the training split; no published checkpoint | validation | 3 |", row)
        lines = report.table_gmm_hmm(self.root).split("\n")
        self.assertEqual(len(lines), 3)
        self.assertIn("| validation | 3 (0.00 h, 2 speakers) | 9 | 2 of 10 (20.00%) | 2 of 3 | 0 / 7 / 0 | 2 of 3 | "
                      "3 in 2 clips | 1.0 | 0.1 |", lines[2])
        (folder / "scores.json").unlink()
        self.assertIn("| not scored yet |", report.table_gmm_hmm(self.root))

    def test_command_line_reports_an_error_instead_of_a_traceback(self):
        out, err = io.StringIO(), io.StringIO()
        folders = ["--work", str(self.work), "--clean", str(self.clean), "--raw", str(self.raw)]
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.assertEqual(gmm_hmm.main(["collect", *folders, "--results", str(self.root / "results")]), 0)
            self.assertEqual(gmm_hmm.main(["decode", *folders, "--split", "test"]), 1)
            self.assertEqual(gmm_hmm.main(["prepare", *folders]), 1)
        self.assertIn("predictions.csv", out.getvalue())
        self.assertIn("--final-test", err.getvalue())
        self.assertIn("--force", err.getvalue())


if __name__ == "__main__":
    unittest.main()
