"""Tests for the scoring policy (text normalisation)."""

import unittest

from huasr.text import normalize, to_chars, to_words


class NormalizeTests(unittest.TestCase):
    def test_unicode_nfc(self):
        # e + combining acute, o + combining double acute: the decomposed spellings of é and ő
        self.assertEqual(normalize("é"), "é")
        self.assertEqual(normalize("fő"), "fő")
        self.assertEqual(normalize("szőlő"), normalize("szőlő"))

    def test_lower_case_including_accented_capitals(self):
        self.assertEqual(normalize("ÁRVÍZTŰRŐ TÜKÖRFÚRÓGÉP"), "árvíztűrő tükörfúrógép")
        self.assertEqual(normalize("Budapest"), "budapest")

    def test_punctuation_is_removed(self):
        self.assertEqual(normalize("Jó napot, kívánok!"), "jó napot kívánok")
        self.assertEqual(normalize("„Igen” – mondta. (Nem?)"), "igen mondta nem")
        self.assertEqual(normalize("Hol van...?"), "hol van")

    def test_symbols_are_removed(self):
        self.assertEqual(normalize("5 € + 3 $ = 100 %"), "5 3 100")
        self.assertEqual(normalize("20 °C"), "20 c")

    def test_hyphen_splits_a_word(self):
        self.assertEqual(normalize("e-mail"), "e mail")
        self.assertEqual(to_words("Tudod-e?"), ["tudod", "e"])

    def test_whitespace_is_collapsed(self):
        self.assertEqual(normalize("  egy\tkettő\n\nhárom négy  "), "egy kettő három négy")

    def test_control_and_format_characters_are_deleted(self):
        self.assertEqual(normalize("al­ma"), "alma")        # soft hyphen
        self.assertEqual(normalize("al​ma﻿"), "alma")  # zero-width space, byte order mark

    def test_diacritics_are_kept_by_default(self):
        self.assertEqual(len({normalize(word) for word in ("kor", "kór", "kör")}), 3)
        self.assertNotEqual(normalize("örül"), normalize("őrül"))

    def test_diacritic_insensitive_variant(self):
        self.assertEqual(normalize("ÁRVÍZTŰRŐ TÜKÖRFÚRÓGÉP", strip_diacritics=True), "arvizturo tukorfurogep")
        self.assertEqual(normalize("kör", strip_diacritics=True), normalize("kor", strip_diacritics=True))

    def test_digits_are_not_rewritten(self):
        self.assertEqual(normalize("1848. március 15."), "1848 március 15")

    def test_empty_and_punctuation_only(self):
        self.assertEqual(normalize(""), "")
        self.assertEqual(normalize(" ... !? "), "")

    def test_idempotent(self):
        for text in ("Jó napot, kívánok!", "é-Mail  X", "ÁRVÍZTŰRŐ"):
            once = normalize(text)
            self.assertEqual(normalize(once), once)
            stripped = normalize(text, strip_diacritics=True)
            self.assertEqual(normalize(stripped, strip_diacritics=True), stripped)


class UnitTests(unittest.TestCase):
    def test_words(self):
        self.assertEqual(to_words("A ház, a kert."), ["a", "ház", "a", "kert"])
        self.assertEqual(to_words(" ! "), [])

    def test_chars_count_the_space_between_words(self):
        self.assertEqual(to_chars("A  ház!"), ["a", " ", "h", "á", "z"])
        self.assertEqual(to_chars("ház", strip_diacritics=True), ["h", "a", "z"])


if __name__ == "__main__":
    unittest.main()
