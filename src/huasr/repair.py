"""Repair of reference text that lost a letter in the corpus's own normalisation.

VoxPopuli gives two transcripts per clip: `raw_text`, the text as published,
and `normalized_text`, derived from it by the corpus (lower case, punctuation
removed, numbers written out as words). In the Hungarian part that
normalisation also deleted every letter "í": `raw_text` has "javítani",
`normalized_text` has "javtani". The letter is a normal part of Hungarian
spelling, so a reference without it counts every correctly written "í" of a
system as an error, and a model trained on it would learn to leave the letter
out.

`repair` puts the letter back where `raw_text` proves it:

1. Both texts are split into words with the scoring policy of `huasr.text`.
2. The two word sequences are aligned by edit distance, treating two words
   as equal when they are equal after the letter is removed from both.
3. A word of `normalized_text` is replaced by its partner from `raw_text`
   only if it is exactly that partner with every "í" deleted. Nothing else
   is changed, and the rest of the string (spacing, punctuation) is kept.

Words that have no such partner are left as they are and counted: numbers
that `raw_text` writes in digits, words that differ for another reason, and
every word of a clip that has no `raw_text` at all. Text is never invented:
removing the letter from the repaired reference gives `normalized_text`
without the letter again, and this is checked for every clip.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from .metrics import align
from .text import normalize, to_words

LOST_LETTER = "í"
MISSING_SHARE = 0.05  # a letter counts as missing if normalized_text has under 5% of its occurrences in raw_text


@dataclass(frozen=True)
class Repair:
    reference: str                           # normalized_text with the proven restorations
    changes: tuple[tuple[int, str, str], ...]  # (position of the word, before, after)
    unaligned_words: int                     # words of normalized_text without an equal partner in raw_text
    status: str                              # restored, unchanged, no_raw_text or unusual_text


def word_spans(text: str) -> list[tuple[int, int]]:
    """Start and end of every word of `text`, with the word boundaries of the scoring policy."""
    spans, start = [], None
    for position, char in enumerate(text):
        separator = char.isspace() or unicodedata.category(char)[0] in "PS"
        if not separator and start is None:
            start = position
        elif separator and start is not None:
            spans.append((start, position))
            start = None
    if start is not None:
        spans.append((start, len(text)))
    return spans


def repair(normalized_text: str, raw_text: str, letter: str = LOST_LETTER) -> Repair:
    """Restore `letter` in `normalized_text` wherever the aligned word of `raw_text` proves it."""
    spans = word_spans(normalized_text)
    words = [normalized_text[start:end] for start, end in spans]
    if not raw_text.strip():
        return Repair(normalized_text, (), len(words), "no_raw_text")
    if words != to_words(normalized_text):  # not in the form the scoring policy produces: do not touch it
        return Repair(normalized_text, (), len(words), "unusual_text")
    raw_words = to_words(raw_text)
    steps = align([word.replace(letter, "") for word in raw_words], [word.replace(letter, "") for word in words])
    changes, unaligned, raw_index, index = [], 0, 0, 0
    for operation, _, _ in steps:
        if operation == "hit":
            partner = raw_words[raw_index]
            if words[index] != partner:
                if words[index] == partner.replace(letter, ""):
                    changes.append((index, words[index], partner))
                else:  # equal apart from the letter, but not by plain deletion: not proven
                    unaligned += 1
        elif operation != "del":
            unaligned += 1
        raw_index += operation != "ins"
        index += operation != "del"
    pieces, position = [], 0
    for index, _, after in changes:
        start, end = spans[index]
        pieces += [normalized_text[position:start], after]
        position = end
    reference = "".join(pieces) + normalized_text[position:]
    if reference.replace(letter, "") != normalized_text.replace(letter, ""):
        raise AssertionError("the repair changed more than the restored letter")
    return Repair(reference, tuple(changes), unaligned, "restored" if changes else "unchanged")


def letter_counts(texts: Iterable[str]) -> Counter:
    """How often each letter occurs in the texts after the scoring policy."""
    return Counter(char for text in texts for char in normalize(text) if char.isalpha())


def missing_letters(raw: Counter, normalized: Counter) -> dict[str, dict[str, int]]:
    """Letters of raw_text that normalized_text (almost) never has, with their counts in both."""
    return {letter: {"raw_text": count, "normalized_text": normalized[letter]}
            for letter, count in sorted(raw.items(), key=lambda item: (-item[1], item[0]))
            if normalized[letter] < MISSING_SHARE * count}
