"""Tests for the generated tables of the READMEs."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from huasr import report
from tests import REPOSITORY

ZERO = {"train-validation": 0, "train-test": 0, "validation-test": 0}
RAW = {
    "seed": 0,
    "target_hours": {"train": 14.0, "validation": 2.5, "test": 2.0},
    "weights": {"validation": 40, "test": 36},
    "official_split": {
        "train": {"clips": 12000, "hours": 30.5, "speakers": 40, "sessions": 1200, "clips_without_speaker_id": 300},
        "validation": {"clips": 800, "hours": 2.25, "speakers": 20, "sessions": 150, "clips_without_speaker_id": 10},
        "test": {"clips": 900, "hours": 2.5, "speakers": 30, "sessions": 160, "clips_without_speaker_id": 20},
        "speakers_in_two_splits": {"train-validation": 15, "train-test": 21, "validation-test": 12},
        "sessions_in_two_splits": {"train-validation": 60, "train-test": 70, "validation-test": 9},
    },
    "subset": {
        "train": {"clips": 5000, "hours": 14.0, "speakers": 25, "sessions": 480, "shards": 11,
                  "shard_bytes": 957_600_000},
        "validation": {"clips": 300, "hours": 0.875, "speakers": 5, "sessions": 40, "shards": 1,
                       "shard_bytes": 59_850_000},
        "test": {"clips": 700, "hours": 2.125, "speakers": 22, "sessions": 90, "shards": 2,
                 "shard_bytes": 145_350_000},
        "speakers_in_two_splits": ZERO, "sessions_in_two_splits": ZERO,
    },
    "removed": {"unknown_speaker": {"train": {"clips": 300, "hours": 0.75}, "validation": {"clips": 10, "hours": 0.03},
                                    "test": {"clips": 20, "hours": 0.06}}},
}


def search_row(test_weight, validation_weight, test, validation, train, met, taken=False):
    def part(hours, speakers, sessions):
        return {"clips": 100, "hours": hours, "speakers": speakers, "sessions": sessions}

    return {"test_weight": test_weight, "validation_weight": validation_weight, "test": part(*test),
            "validation": part(*validation), "train": part(*train), "targets_met": met, "taken": taken}


SEARCH = {
    "seed": 0, "minimum_test_hours": 2.0, "validation_target_hours": 2.5, "training_target_hours": 14.0,
    "weights_tried": [1, 2, 4], "taken": {"validation": 4, "test": 2},
    "rows": [
        search_row(1, 1, (0.5, 5, 20), (1.0, 4, 30), (40.0, 30, 900), False),
        search_row(2, 2, (2.125, 22, 90), (2.75, 15, 100), (25.5, 25, 600), True),
        search_row(2, 4, (2.25, 23, 95), (2.875, 16, 110), (15.25, 20, 480), True, taken=True),
        search_row(4, 2, (2.25, 23, 95), (2.875, 16, 110), (15.25, 20, 480), True),   # same result as the row above
        search_row(4, 4, (2.75, 30, 120), (2.0, 12, 80), (9.5, 15, 300), False),
    ],
}
CLEAN = {
    "splits": {"train": {"clips": 5000, "hours": 14.0, "speakers": 25, "sessions": 480},
               "validation": {"clips": 300, "hours": 0.875, "speakers": 5, "sessions": 40},
               "test": {"clips": 700, "hours": 2.125, "speakers": 22, "sessions": 90}},
    "speakers_in_two_splits": ZERO, "sessions_in_two_splits": ZERO,
    "references_also_in_training": {"validation": {"clips": 3, "of": 300}, "test": {"clips": 1, "of": 700}},
    "english_number_words": {
        "train": {"clips": 250, "raw_text_has_digits": 200, "no_raw_text": 48, "raw_text_without_digits": 2},
        "validation": {"clips": 20, "raw_text_has_digits": 15, "no_raw_text": 5, "raw_text_without_digits": 0},
        "test": {"clips": 30, "raw_text_has_digits": 26, "no_raw_text": 3, "raw_text_without_digits": 1}},
    "reference_repair": {"clips": {"restored": 2100, "unchanged": 3850, "no_raw_text": 50},
                         "restored_words": 3456, "restored_letters": 3500,
                         "clips_with_words_that_could_not_be_aligned": 400, "words_that_could_not_be_aligned": 1200},
    "letters_of_raw_text_missing_from_normalized_text": {"í": {"raw_text": 3600, "normalized_text": 0,
                                                               "reference": 3500}},
    "chars_per_second": {split: {"clips": 10, "min": 2.5, "p1": 3.0, "p5": 5.0, "p25": 10.0, "p50": 14.0, "p75": 16.0,
                                 "p95": 19.0, "p99": 22.0, "max": 40.0, "mean": 13.5} for split in report.SPLITS},
}
README = """# Title

Text before.

<!-- generated: voxpopuli-subset -->
old table
<!-- end generated -->

Text between.

<!-- generated: voxpopuli-repair -->
<!-- end generated -->

Text after.
"""


class ReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name in report.READMES:
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            (self.root / name).write_text(README, encoding="utf-8", newline="\n")

    def write_summaries(self, raw=RAW, clean=CLEAN):
        for name, summary in ((report.RAW_SUMMARY, raw), (report.CLEAN_SUMMARY, clean), (report.SEARCH, SEARCH)):
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            (self.root / name).write_text(json.dumps(summary), encoding="utf-8")

    def run_main(self, *arguments):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = report.main(list(arguments), root=self.root)
        return code, out.getvalue() + err.getvalue()

    def test_tables_come_from_the_summary_files(self):
        self.write_summaries()
        text = report.render(README, self.root)
        self.assertIn("| train | 5,000 | 14.00 | 25 | 480 | 11 | 957.6 |", text)
        self.assertIn("| total | 6,000 | 17.00 | 52 | 610 | 14 | 1162.8 |", text)
        self.assertIn("Sessions shared by two splits of the subset: train and validation 0, train and test 0, "
                      "validation and test 0.", text)
        self.assertIn("training target 14 h, validation target 2.5 h, at least 2 h of test audio; weights taken: "
                      "validation 40, test 36; seed 0.", text)
        self.assertIn("| Restored words | 3,456 |", text)
        self.assertIn("| `í` | 3,600 | 0 | 3,500 |", text)
        self.assertIn("1,200 in 400 clips", text)
        self.assertIn(f"Source: `{report.RAW_SUMMARY}`.", text)
        self.assertNotIn("old table", text)
        self.assertTrue(text.startswith("# Title\n\nText before.\n\n<!-- generated: voxpopuli-subset -->\n| Split |"))
        self.assertTrue(text.endswith("<!-- end generated -->\n\nText after.\n"))

    def test_official_split_and_removal_tables(self):
        official = report.table_official(RAW)
        self.assertIn("| train | 12,000 | 30.50 | 40 | 1,200 | 300 |", official)
        self.assertIn("Speakers shared by two official splits: train and validation 15, train and test 21, "
                      "validation and test 12.", official)
        self.assertIn("Sessions shared by two official splits: train and validation 60, train and test 70, "
                      "validation and test 9.", official)
        self.assertIn("| unknown speaker | 300 (0.75 h) | 10 (0.03 h) | 20 (0.06 h) |", report.table_removed(RAW))
        self.assertIn("| validation | 10 | 2.5 | 3.0 | 5.0 | 10.0 | 14.0 | 16.0 | 19.0 | 22.0 | 40.0 |",
                      report.table_rate(CLEAN))

    def test_split_report_has_what_must_be_reported_for_a_split(self):
        table = report.table_split_report(CLEAN)
        self.assertIn("| Split | Speakers | Sessions | Utterances | Hours | References that occur word for word in "
                      "the training split |", table)
        self.assertIn("| train | 25 | 480 | 5,000 | 14.00 | - |", table)
        self.assertIn("| validation | 5 | 40 | 300 | 0.88 | 3 of 300 |", table)
        self.assertIn("| test | 22 | 90 | 700 | 2.12 | 1 of 700 |", table)
        self.assertIn("Speakers in two splits: train and validation 0, train and test 0, validation and test 0.", table)
        self.assertIn("Sessions in two splits: train and validation 0, train and test 0, validation and test 0.", table)

    def test_number_word_table(self):
        table = report.table_number_words(CLEAN)
        self.assertIn("| English number words | train | validation | test |", table)
        self.assertIn("| References with English number words | 250 | 20 | 30 |", table)
        self.assertIn("| of these: `raw_text` has the number in digits | 200 | 15 | 26 |", table)
        self.assertIn("| of these: the clip has no `raw_text` | 48 | 5 | 3 |", table)

    def test_search_table_shows_the_results_that_meet_the_targets_and_the_one_taken(self):
        lines = report.table_search(SEARCH).split("\n")
        self.assertEqual(lines[2], "| 2 | 4 | 2.25 (23, 95) | 2.88 (16) | 15.25 (20) | yes | taken |")
        self.assertEqual(lines[3], "| 2 | 2 | 2.12 (22, 90) | 2.75 (15) | 25.50 (25) | yes |  |")
        self.assertEqual(len([line for line in lines if line.startswith("| ")]), 3)  # header and two results
        self.assertIn("5 weight pairs were tried (weights 1, 2, 4); 3 of them leave at least 2 h of test audio, "
                      "2.5 h of validation audio and 14 h of training audio.", lines[-1])

    def test_rendering_twice_changes_nothing(self):
        self.write_summaries()
        once = report.render(README, self.root)
        self.assertEqual(report.render(once, self.root), once)

    def test_missing_summary_is_stated_instead_of_a_table(self):
        text = report.render(README, self.root)
        self.assertIn(f"_Not built yet: `{report.RAW_SUMMARY}` does not exist._", text)
        self.assertNotIn("| Split |", text)

    def test_unknown_block_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "unknown generated block"):
            report.render("<!-- generated: something-else -->\n<!-- end generated -->\n", self.root)

    def test_command_rewrites_and_check_detects_stale_blocks(self):
        self.write_summaries()
        self.assertEqual(self.run_main("--check")[0], 1)
        self.assertEqual((self.root / "README.md").read_text(encoding="utf-8"), README)  # --check changes nothing
        code, printed = self.run_main()
        self.assertEqual(code, 0)
        self.assertIn("data/raw/README.md", printed)
        self.assertEqual(self.run_main("--check")[0], 0)
        changed = json.loads(json.dumps(RAW))
        changed["subset"]["train"]["clips"] += 1
        self.write_summaries(raw=changed)
        code, printed = self.run_main("--check")
        self.assertEqual(code, 1)
        self.assertIn("out of date", printed)

    def test_readmes_of_this_repository_are_up_to_date(self):
        for name in report.READMES:
            text = (REPOSITORY / name).read_text(encoding="utf-8")
            self.assertEqual(report.render(text, REPOSITORY), text, f"{name}: run python -m huasr.report")


if __name__ == "__main__":
    unittest.main()
