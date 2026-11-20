"""Tests for the edit-distance alignment, WER/CER and the cluster bootstrap."""

import unittest

import numpy as np

from huasr import metrics
from huasr.metrics import (EditCounts, align, bootstrap_interval, cer, clip_counts, edit_counts, error_rate,
                           paired_bootstrap, wer)
from huasr.text import to_words


class AlignmentTests(unittest.TestCase):
    def test_operations_in_order(self):
        steps = align("the cat sat on the mat".split(), "the cat sit on the".split())
        self.assertEqual([operation for operation, _, _ in steps], ["hit", "hit", "sub", "hit", "hit", "del"])
        self.assertEqual(steps[2], ("sub", "sat", "sit"))
        self.assertEqual(steps[5], ("del", "mat", None))

    def test_insertions_around_a_hit(self):
        steps = align(["go"], "please go now today".split())
        self.assertEqual([operation for operation, _, _ in steps], ["ins", "hit", "ins", "ins"])

    def test_counts_equal_the_edit_distance(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            ref = [str(x) for x in rng.integers(0, 4, size=rng.integers(0, 8))]
            hyp = [str(x) for x in rng.integers(0, 4, size=rng.integers(0, 8))]
            counts = edit_counts(ref, hyp)
            self.assertEqual(counts.errors, levenshtein(ref, hyp))
            self.assertEqual(counts.reference_units, len(ref))
            self.assertEqual(counts.hits + counts.substitutions + counts.insertions, len(hyp))

    def test_empty_sides(self):
        self.assertEqual(edit_counts([], []), EditCounts())
        self.assertEqual(edit_counts(["a", "b"], []), EditCounts(deletions=2))
        self.assertEqual(edit_counts([], ["a", "b"]), EditCounts(insertions=2))


def levenshtein(a, b):
    """Plain edit distance, written independently of `huasr.metrics.align`."""
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        current = [i]
        for j, y in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


class ErrorRateTests(unittest.TestCase):
    def test_slide_example_wer(self):
        # (S + D + I) / N = (1 + 1 + 0) / 6 = 33.3%
        result = wer(["the cat sat on the mat"], ["the cat sit on the"])
        self.assertEqual((result.substitutions, result.deletions, result.insertions), (1, 1, 0))
        self.assertEqual(result.reference_units, 6)
        self.assertAlmostEqual(result.rate, 2 / 6)
        self.assertEqual(f"{result.rate:.1%}", "33.3%")

    def test_slide_example_cer(self):
        # 17 letters + 5 spaces = 22 characters; 1 substitution + 4 deletions
        result = cer(["the cat sat on the mat"], ["the cat sit on the"])
        self.assertEqual((result.substitutions, result.deletions, result.insertions), (1, 4, 0))
        self.assertEqual(result.reference_units, 22)
        self.assertEqual(f"{result.rate:.1%}", "22.7%")

    def test_slide_example_three_insertions_give_300_percent(self):
        result = wer(["go"], ["please go now today"])
        self.assertEqual((result.substitutions, result.deletions, result.insertions, result.hits), (0, 0, 3, 1))
        self.assertEqual(result.reference_units, 1)
        self.assertAlmostEqual(result.rate, 3.0)
        self.assertEqual(f"{result.rate:.0%}", "300%")

    def test_two_insertions_on_five_words(self):
        result = wer(["el gato duerme en casa"], ["el gato duerme en la casa ahora"])
        self.assertEqual((result.insertions, result.hits, result.reference_units), (2, 5, 5))
        self.assertAlmostEqual(result.rate, 0.4)

    def test_rate_is_corpus_level_not_a_mean_of_clip_rates(self):
        references = ["go", "one two three four five six seven eight nine"]
        hypotheses = ["please go now today", "one two three four five six seven eight nine"]
        result = wer(references, hypotheses)
        self.assertAlmostEqual(result.rate, 3 / 10)  # the mean of the clip rates would be (300% + 0%) / 2
        self.assertEqual(result.as_dict()["errors"], 3)

    def test_scoring_policy_is_applied_to_both_sides(self):
        self.assertEqual(wer(["Jó napot, kívánok!"], ["jó napot kívánok"]).errors, 0)
        self.assertEqual(cer(["Jó  napot!"], ["JÓ NAPOT"]).errors, 0)

    def test_diacritic_errors_and_the_insensitive_variant(self):
        self.assertEqual(wer(["a kör"], ["a kor"]).substitutions, 1)
        self.assertEqual(cer(["a kör"], ["a kor"]).substitutions, 1)
        self.assertEqual(wer(["a kör"], ["a kor"], strip_diacritics=True).errors, 0)
        self.assertEqual(cer(["a kör"], ["a kor"], strip_diacritics=True).errors, 0)

    def test_one_wrong_suffix_is_one_word_error_but_few_character_errors(self):
        word = wer(["a házaimban"], ["a házaidban"])
        char = cer(["a házaimban"], ["a házaidban"])
        self.assertEqual((word.errors, word.reference_units), (1, 2))
        self.assertEqual((char.errors, char.reference_units), (1, 11))

    def test_empty_hypothesis_and_empty_reference(self):
        self.assertEqual(wer(["egy kettő"], [""]), EditCounts(deletions=2))
        silence = wer([""], ["egy kettő"])
        self.assertEqual(silence, EditCounts(insertions=2))
        with self.assertRaises(ValueError):
            silence.rate

    def test_unequal_lengths_are_rejected(self):
        with self.assertRaises(ValueError):
            wer(["a", "b"], ["a"])

    def test_any_unit_can_be_scored(self):
        def pairs(text):  # a stand-in for a subword tokenizer
            return [text[i:i + 2] for i in range(0, len(text), 2)]

        result = error_rate(["abcdef"], ["abxdef"], pairs)
        self.assertEqual((result.substitutions, result.reference_units), (1, 3))


def scored(references, hypotheses):
    return clip_counts(references, hypotheses, to_words)


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        # 12 speakers, 3 clips each, 2 to 5 reference words per clip.
        rng = np.random.default_rng(1)
        self.speakers = [f"spk{k:02d}" for k in range(12) for _ in range(3)]
        words = ["a", "b", "c", "d", "e"]
        self.references = [" ".join(words[:2 + clip % 4]) for clip in range(len(self.speakers))]

        def noisy(error_probability):
            # every word is replaced by "x" with this probability
            return [" ".join("x" if rng.random() < error_probability else word for word in reference.split())
                    for reference in self.references]

        self.baseline = scored(self.references, noisy(0.5))
        self.better = scored(self.references, noisy(0.1))

    def test_same_seed_gives_identical_results(self):
        first = paired_bootstrap(self.better, self.baseline, self.speakers, n_resamples=2000, seed=7)
        second = paired_bootstrap(self.better, self.baseline, self.speakers, n_resamples=2000, seed=7)
        self.assertEqual(first, second)
        one = bootstrap_interval(self.baseline, self.speakers, n_resamples=2000, seed=7)
        two = bootstrap_interval(self.baseline, self.speakers, n_resamples=2000, seed=7)
        self.assertEqual(one, two)

    def test_another_seed_gives_another_interval_but_the_same_point_estimate(self):
        first = paired_bootstrap(self.better, self.baseline, self.speakers, n_resamples=2000, seed=7)
        other = paired_bootstrap(self.better, self.baseline, self.speakers, n_resamples=2000, seed=8)
        self.assertNotEqual((first["ci_low"], first["ci_high"]), (other["ci_low"], other["ci_high"]))
        self.assertEqual(first["difference"], other["difference"])

    def test_point_estimates_are_the_corpus_rates(self):
        result = paired_bootstrap(self.better, self.baseline, self.speakers, n_resamples=500)
        rate_a = sum(self.better, EditCounts()).rate
        rate_b = sum(self.baseline, EditCounts()).rate
        self.assertAlmostEqual(result["rate_a"], rate_a)
        self.assertAlmostEqual(result["rate_b"], rate_b)
        self.assertAlmostEqual(result["difference"], rate_a - rate_b)
        self.assertEqual(result["n_clusters"], 12)
        single = bootstrap_interval(self.baseline, self.speakers, n_resamples=500)
        self.assertAlmostEqual(single["rate"], rate_b)
        self.assertLessEqual(single["ci_low"], single["rate"])
        self.assertGreaterEqual(single["ci_high"], single["rate"])

    def test_clearly_better_system(self):
        result = paired_bootstrap(self.better, self.baseline, self.speakers, n_resamples=2000)
        self.assertLess(result["difference"], 0)
        self.assertLess(result["ci_high"], 0)          # the whole interval is below zero
        self.assertLess(result["ci_low"], result["difference"])
        self.assertAlmostEqual(result["p_value"], 1 / 2001)  # no resample with a difference >= 0

    def test_the_test_is_one_sided(self):
        worse = paired_bootstrap(self.baseline, self.better, self.speakers, n_resamples=2000)
        self.assertGreater(worse["ci_low"], 0)
        self.assertEqual(worse["p_value"], 1.0)

    def test_identical_systems_are_not_significant(self):
        result = paired_bootstrap(self.baseline, self.baseline, self.speakers, n_resamples=500)
        self.assertEqual((result["difference"], result["ci_low"], result["ci_high"]), (0.0, 0.0, 0.0))
        self.assertEqual(result["p_value"], 1.0)

    def test_speakers_are_resampled_not_clips(self):
        # One speaker for everything: every resample is the whole test set again, so the interval has no width.
        one_speaker = ["same"] * len(self.speakers)
        result = bootstrap_interval(self.baseline, one_speaker, n_resamples=300)
        self.assertEqual(result["ci_low"], result["ci_high"])
        self.assertEqual(result["n_clusters"], 1)
        # Two speakers: a resample is speaker 1 twice, both, or speaker 2 twice, so only three values can occur.
        half = len(self.speakers) // 2
        two_speakers = ["first"] * half + ["second"] * (len(self.speakers) - half)
        errors = [c.errors for c in self.baseline]
        units = [c.reference_units for c in self.baseline]
        totals = metrics._cluster_totals([errors, units], two_speakers)
        self.assertEqual(totals.tolist(), [[sum(errors[:half]), sum(units[:half])],
                                           [sum(errors[half:]), sum(units[half:])]])
        sums = metrics._resample(totals, 400, seed=0)
        allowed = {tuple(2 * totals[0]), tuple(totals[0] + totals[1]), tuple(2 * totals[1])}
        self.assertEqual({tuple(row) for row in sums.tolist()}, allowed)

    def test_wider_interval_when_clips_of_a_speaker_fail_together(self):
        # Same clip-level error counts, but the bad clips either belong to few speakers or are spread out.
        counts = [EditCounts(substitutions=4)] * 12 + [EditCounts(hits=4)] * 24
        clustered = [f"spk{k // 3}" for k in range(36)]             # four speakers have only bad clips
        spread = [f"spk{k % 12}" for k in range(36)]                # every speaker has one bad clip
        wide = bootstrap_interval(counts, clustered, n_resamples=2000)
        narrow = bootstrap_interval(counts, spread, n_resamples=2000)
        self.assertEqual(narrow["ci_low"], narrow["ci_high"])
        self.assertGreater(wide["ci_high"] - wide["ci_low"], 0.2)

    def test_both_systems_must_be_scored_on_the_same_references(self):
        other = scored(["a b c"] * len(self.speakers), ["a b c"] * len(self.speakers))
        with self.assertRaises(ValueError):
            paired_bootstrap(self.better, other, self.speakers)
        with self.assertRaises(ValueError):
            paired_bootstrap(self.better, self.baseline, self.speakers[:-1])

    def test_empty_references_are_rejected(self):
        with self.assertRaises(ValueError):
            bootstrap_interval([EditCounts(insertions=2)], ["spk"])


if __name__ == "__main__":
    unittest.main()
