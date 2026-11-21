"""Tests for the paired comparison of two runs, on hand-made prediction files."""

import contextlib
import io
import json
import unittest

from huasr import compare, metrics
from huasr.text import to_words
from tests.test_score import PREDICTIONS, TemporaryFolderTestCase, write_run

# The same clips as PREDICTIONS with the hypotheses of a better system: one substitution is left in c1.
BETTER = [("c1", "s1", 2000, 0, "the cat sat on the mat", "the cat sit on the mat", 0.5),
          ("c2", "s1", 1000, 0, "go", "go", 0.25),
          ("c3", "s2", 3000, 1, "a kör two", "a kör two", 0.75),
          ("c4", "s3", 2000, 0, "jó napot", "jó napot", 0.5)]


class CompareTests(TemporaryFolderTestCase):
    def setUp(self):
        super().setUp()
        self.baseline = write_run(self.root / "results" / "baselines" / "fake-system" / "validation")
        self.new = write_run(self.root / "results" / "own" / "our-model" / "validation", BETTER, system="our-model",
                             group="own")

    def test_rates_of_both_systems_and_their_difference(self):
        path, comparison = compare.compare_runs(self.new, self.baseline)
        self.assertEqual(path, self.new / "comparison_with_fake-system.json")
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), comparison)
        self.assertEqual((comparison["new"]["system"], comparison["new"]["group"], comparison["baseline"]["system"]),
                         ("our-model", "own", "fake-system"))
        self.assertEqual((comparison["split"], comparison["final_test"], comparison["complete"]),
                         ("validation", False, True))
        everything = comparison["all_clips"]
        self.assertEqual((everything["clips"], everything["speakers"]), (4, 3))
        wer = everything["wer"]
        self.assertEqual((wer["baseline"], wer["new"], wer["difference"]), (0.5, round(1 / 12, 4), round(-5 / 12, 4)))
        self.assertEqual(everything["wer_no_diacritics"]["baseline"], round(5 / 12, 4))
        self.assertLess(everything["cer"]["new"], everything["cer"]["baseline"])

    def test_interval_and_p_value_come_from_the_paired_bootstrap_over_speakers(self):
        _, comparison = compare.compare_runs(self.new, self.baseline)
        references = [row[4] for row in PREDICTIONS]
        expected = metrics.paired_bootstrap(
            metrics.clip_counts(references, [row[5] for row in BETTER], to_words),
            metrics.clip_counts(references, [row[5] for row in PREDICTIONS], to_words),
            [row[1] for row in PREDICTIONS])
        wer = comparison["all_clips"]["wer"]
        self.assertEqual(wer["ci95"], [round(expected["ci_low"], 4), round(expected["ci_high"], 4)])
        self.assertEqual(wer["p_value"], round(expected["p_value"], 5))
        self.assertLessEqual(wer["ci95"][0], wer["difference"])
        self.assertLessEqual(wer["difference"], wer["ci95"][1])
        self.assertEqual((comparison["scoring"]["resamples"], comparison["scoring"]["seed"]),
                         (metrics.DEFAULT_RESAMPLES, metrics.DEFAULT_SEED))

    def test_figures_without_the_flagged_clips(self):
        _, comparison = compare.compare_runs(self.new, self.baseline)
        rest = comparison["without_english_number_words"]
        self.assertEqual((comparison["clips_with_english_number_words"], rest["clips"], rest["speakers"]), (1, 3, 2))
        self.assertEqual((rest["wer"]["baseline"], rest["wer"]["new"]), (round(5 / 9, 4), round(1 / 9, 4)))

    def test_a_system_compared_with_itself_has_no_difference(self):
        _, comparison = compare.compare_runs(self.baseline, self.baseline)
        wer = comparison["all_clips"]["wer"]
        self.assertEqual((wer["difference"], wer["ci95"], wer["p_value"]), (0.0, [0.0, 0.0], 1.0))

    def test_runs_on_different_clips_are_not_compared(self):
        other_reference = [BETTER[0][:4] + ("the dog sat on the mat",) + BETTER[0][5:], *BETTER[1:]]
        cases = {"references": write_run(self.root / "a", other_reference),
                 "clips": write_run(self.root / "b", BETTER[:3]),
                 "order": write_run(self.root / "c", BETTER[::-1]),
                 "split": write_run(self.root / "d", BETTER, split="test")}
        for name, folder in cases.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, "do not hold the same clips"):
                compare.compare_runs(folder, self.baseline)
            self.assertEqual(sorted(path.name for path in folder.iterdir()), ["predictions.csv", "run.json"])

    def test_command_line(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.assertEqual(compare.main([str(self.new), str(self.baseline)]), 0)
            self.assertEqual(compare.main([str(self.new), str(self.root / "nothing")]), 1)
        self.assertIn("our-model against fake-system, validation, 4 clips of 3 speakers", out.getvalue())
        self.assertIn("wer: 50.00% -> 8.33%, difference -41.67 points", out.getvalue())
        self.assertIn("error:", err.getvalue())


if __name__ == "__main__":
    unittest.main()
