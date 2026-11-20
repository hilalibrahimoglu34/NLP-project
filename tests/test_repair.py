"""Tests for the repair of reference text (the letter "í" that the corpus normalisation deleted).

The example sentences are made up for the tests; a few tokens are deliberately not words.
"""

import random
import unittest
from collections import Counter

from huasr.repair import LOST_LETTER, letter_counts, missing_letters, repair, word_spans


class RepairTests(unittest.TestCase):
    def test_letter_is_restored_where_raw_text_proves_it(self):
        fixed = repair("másodszor javtani kell az emberi jogok helyzetét",
                       "Másodszor: javítani kell az emberi jogok helyzetét.")
        self.assertEqual(fixed.reference, "másodszor javítani kell az emberi jogok helyzetét")
        self.assertEqual(fixed.changes, ((1, "javtani", "javítani"),))
        self.assertEqual((fixed.unaligned_words, fixed.status), (0, "restored"))

    def test_capital_letter_and_several_letters_in_one_word(self):
        fixed = repair("rország xyz kvül", "Írország xíyíz kívül")  # "xíyíz" is not a word: two letters in one token
        self.assertEqual(fixed.reference, "írország xíyíz kívül")
        self.assertEqual([after for _, _, after in fixed.changes], ["írország", "xíyíz", "kívül"])

    def test_spacing_and_punctuation_of_normalized_text_are_kept(self):
        fixed = repair("elnök úr!  a bzottság dönt.", "Elnök úr! A bízottság dönt.")
        self.assertEqual(fixed.reference, "elnök úr!  a bízottság dönt.")
        self.assertEqual(fixed.changes, ((3, "bzottság", "bízottság"),))

    def test_text_without_the_letter_is_unchanged(self):
        fixed = repair("a víz tiszta", "A víz tiszta.")
        self.assertEqual((fixed.reference, fixed.changes, fixed.unaligned_words, fixed.status),
                         ("a víz tiszta", (), 0, "unchanged"))

    def test_number_written_out_is_left_and_counted(self):
        # "tz" may well be "tíz" (ten), but raw_text only has the digits, so nothing proves it.
        fixed = repair("a dj tz százalék", "A díj 10 százalék.")
        self.assertEqual(fixed.reference, "a díj tz százalék")
        self.assertEqual((len(fixed.changes), fixed.unaligned_words, fixed.status), (1, 1, "restored"))

    def test_number_written_out_as_several_words(self):
        fixed = repair("kétezer húsz dj", "2020 díj")
        self.assertEqual((fixed.reference, fixed.unaligned_words), ("kétezer húsz díj", 2))

    def test_word_missing_from_raw_text_does_not_shift_the_repair(self):
        fixed = repair("igen a hd és a vz", "A híd és a víz.")
        self.assertEqual((fixed.reference, fixed.unaligned_words), ("igen a híd és a víz", 1))

    def test_clip_without_raw_text_is_left_and_all_its_words_are_counted(self):
        fixed = repair("javtani kell", "")
        self.assertEqual((fixed.reference, fixed.changes, fixed.unaligned_words, fixed.status),
                         ("javtani kell", (), 2, "no_raw_text"))

    def test_word_that_differs_in_another_way_is_not_touched(self):
        fixed = repair("a hidak", "A hídak")  # "i" where raw_text has "í": a replacement, not a deletion
        self.assertEqual((fixed.reference, fixed.changes, fixed.unaligned_words), ("a hidak", (), 1))

    def test_word_that_lost_only_some_of_its_letters_is_not_touched(self):
        fixed = repair("xyíz", "xíyíz")
        self.assertEqual((fixed.reference, fixed.changes, fixed.unaligned_words), ("xyíz", (), 1))

    def test_other_missing_letters_are_not_restored(self):
        fixed = repair("kova úr beszélt", "Kovač úr beszélt.")
        self.assertEqual((fixed.reference, fixed.changes, fixed.unaligned_words), ("kova úr beszélt", (), 1))

    def test_text_that_is_not_in_normalised_form_is_not_touched(self):
        fixed = repair("Javtani kell", "Javítani kell.")
        self.assertEqual((fixed.reference, fixed.changes, fixed.status), ("Javtani kell", (), "unusual_text"))

    def test_only_the_letter_is_ever_added(self):
        rng = random.Random(0)
        alphabet = ["a", "b", "í", "é", "k"]
        for _ in range(300):
            raw_words = ["".join(rng.choice(alphabet) for _ in range(rng.randint(1, 5))) for _ in range(rng.randint(0, 8))]
            words = [word.replace(LOST_LETTER, "") for word in raw_words]
            words = [word for word in words if word and rng.random() > 0.15]       # some words are missing
            words = [word if rng.random() > 0.15 else word + "x" for word in words]  # some are different
            normalized = " ".join(words)
            fixed = repair(normalized, " ".join(raw_words))
            self.assertEqual(fixed.reference.replace(LOST_LETTER, ""), normalized.replace(LOST_LETTER, ""))
            self.assertEqual(len(fixed.reference.split()), len(normalized.split()))
            for position, before, after in fixed.changes:
                self.assertEqual(after.replace(LOST_LETTER, ""), before)
                self.assertIn(after, raw_words)
                self.assertEqual(fixed.reference.split()[position], after)

    def test_word_spans_follow_the_scoring_policy(self):
        text = "úr!  e-mail (2020)"
        self.assertEqual([text[start:end] for start, end in word_spans(text)], ["úr", "e", "mail", "2020"])
        self.assertEqual(word_spans(""), [])


class LetterTests(unittest.TestCase):
    def test_letters_are_counted_after_the_scoring_policy(self):
        self.assertEqual(letter_counts(["Íz, íz!", "2 víz"]), Counter({"í": 3, "z": 3, "v": 1}))

    def test_letters_that_normalized_text_does_not_have(self):
        raw = letter_counts(["Kívül a Kovač víz", "így írt"] * 20)
        normalized = letter_counts(["kvül a kova vz", "gy rt"] * 20)
        self.assertEqual(missing_letters(raw, normalized),
                         {"í": {"raw_text": 80, "normalized_text": 0}, "č": {"raw_text": 20, "normalized_text": 0}})
        self.assertEqual(missing_letters(raw, raw), {})


if __name__ == "__main__":
    unittest.main()
