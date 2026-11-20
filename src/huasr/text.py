"""Scoring policy: the text normalisation applied before any error rate is computed.

The same function is applied to the reference and to the system output, so a
difference in casing or punctuation is never counted as a recognition error.

Policy, in this order:

1. Unicode NFC (a letter and its accent are one code point).
2. Lower case.
3. Every punctuation character (Unicode categories P*) and every symbol
   (S*) is replaced by a space, so a hyphenated word counts as two words.
4. Control and format characters (C*: soft hyphen, zero-width space, ...) are
   deleted.
5. Whitespace is collapsed to single spaces and trimmed.

Diacritics are kept: a and á, o, ö and ő are different letters in Hungarian.
Digits are not rewritten as words. `strip_diacritics=True` gives the
diacritic-insensitive variant used for the second results table: after NFD
all combining marks (Mn) are removed.
"""

from __future__ import annotations

import unicodedata


def normalize(text: str, strip_diacritics: bool = False) -> str:
    """Apply the scoring policy to one transcript."""
    text = unicodedata.normalize("NFC", text).lower()
    kept = []
    for char in text:
        category = unicodedata.category(char)
        if char.isspace() or category[0] in "PS":
            kept.append(" ")
        elif category[0] != "C":
            kept.append(char)
    text = unicodedata.normalize("NFC", "".join(kept))
    if strip_diacritics:
        decomposed = unicodedata.normalize("NFD", text)
        text = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
        text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())


def to_words(text: str, strip_diacritics: bool = False) -> list[str]:
    """Units for WER: the words of the normalised transcript."""
    return normalize(text, strip_diacritics).split()


def to_chars(text: str, strip_diacritics: bool = False) -> list[str]:
    """Units for CER: the characters of the normalised transcript.

    The single space between two words is a unit, so a wrong word boundary
    costs one character error.
    """
    return list(normalize(text, strip_diacritics))
