"""Error rates from an edit-distance alignment, and their uncertainty.

Error rate = (substitutions + deletions + insertions) / reference units. The
units are words (WER), characters (CER) or the tokens of any declared
tokenizer (subword error rate). Insertions are in the numerator but not in
the denominator, so a rate can exceed 100%.

Rates are corpus-level: the edit counts of all clips are summed and divided
by the summed reference units. A mean of per-clip percentages would give a
one-word clip the same weight as a twenty-word clip.

Uncertainty comes from a cluster bootstrap over speakers. Clips of one
speaker share a voice and a microphone, so they are not independent samples;
each resample draws speakers with replacement and takes all of their clips.
To compare two systems the same resampled speakers are used for both
(paired bootstrap), because both systems were scored on the same clips.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass

import numpy as np

from .text import to_chars, to_words

DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 0
_BLOCK = 1_000  # resamples drawn per step; fixed, so results depend on the seed only


@dataclass(frozen=True)
class EditCounts:
    """Edit operations that turn a reference into a hypothesis."""

    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    hits: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def reference_units(self) -> int:
        return self.hits + self.substitutions + self.deletions

    @property
    def rate(self) -> float:
        if self.reference_units == 0:
            raise ValueError("no reference units: report clips with an empty reference separately")
        return self.errors / self.reference_units

    def __add__(self, other: EditCounts) -> EditCounts:
        return EditCounts(self.substitutions + other.substitutions, self.deletions + other.deletions,
                          self.insertions + other.insertions, self.hits + other.hits)

    def as_dict(self) -> dict:
        return {"rate": self.rate, "errors": self.errors, "substitutions": self.substitutions,
                "deletions": self.deletions, "insertions": self.insertions, "hits": self.hits,
                "reference_units": self.reference_units}


def align(reference: Sequence, hypothesis: Sequence) -> list[tuple[str, object, object]]:
    """Minimum edit-distance alignment of two unit sequences.

    Returns one (operation, reference unit, hypothesis unit) per step, with
    operation in {"hit", "sub", "del", "ins"}; the missing side is None. All
    three edits cost 1. Where several alignments have the same cost the
    backtrace prefers hit/substitution, then deletion, then insertion.
    """
    n, m = len(reference), len(hypothesis)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = i
    for j in range(1, m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        ref_unit, row, above = reference[i - 1], cost[i], cost[i - 1]
        for j in range(1, m + 1):
            diagonal = above[j - 1] + (ref_unit != hypothesis[j - 1])
            row[j] = min(diagonal, above[j] + 1, row[j - 1] + 1)

    steps = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]):
            same = reference[i - 1] == hypothesis[j - 1]
            steps.append(("hit" if same else "sub", reference[i - 1], hypothesis[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            steps.append(("del", reference[i - 1], None))
            i -= 1
        else:
            steps.append(("ins", None, hypothesis[j - 1]))
            j -= 1
    steps.reverse()
    return steps


def edit_counts(reference: Sequence, hypothesis: Sequence) -> EditCounts:
    """Substitutions, deletions, insertions and hits between two unit sequences."""
    tally = {"hit": 0, "sub": 0, "del": 0, "ins": 0}
    for operation, _, _ in align(reference, hypothesis):
        tally[operation] += 1
    return EditCounts(tally["sub"], tally["del"], tally["ins"], tally["hit"])


def clip_counts(references: Sequence[str], hypotheses: Sequence[str],
                tokenize: Callable[[str], Sequence]) -> list[EditCounts]:
    """Edit counts of every clip; `tokenize` turns a transcript into units."""
    if len(references) != len(hypotheses):
        raise ValueError(f"{len(references)} references but {len(hypotheses)} hypotheses")
    return [edit_counts(tokenize(ref), tokenize(hyp)) for ref, hyp in zip(references, hypotheses)]


def error_rate(references: Sequence[str], hypotheses: Sequence[str],
               tokenize: Callable[[str], Sequence]) -> EditCounts:
    """Corpus-level counts for any unit (use this with a subword tokenizer)."""
    return sum(clip_counts(references, hypotheses, tokenize), EditCounts())


def wer(references: Sequence[str], hypotheses: Sequence[str], strip_diacritics: bool = False) -> EditCounts:
    """Corpus-level word error rate under the scoring policy of `huasr.text`."""
    return error_rate(references, hypotheses, lambda text: to_words(text, strip_diacritics))


def cer(references: Sequence[str], hypotheses: Sequence[str], strip_diacritics: bool = False) -> EditCounts:
    """Corpus-level character error rate under the scoring policy of `huasr.text`."""
    return error_rate(references, hypotheses, lambda text: to_chars(text, strip_diacritics))


def _cluster_totals(columns: Sequence[Sequence[int]], clusters: Sequence[Hashable]) -> np.ndarray:
    """Sum per-clip columns within each cluster: array of shape (clusters, columns)."""
    if any(len(column) != len(clusters) for column in columns):
        raise ValueError("every per-clip sequence needs one entry per cluster label")
    index = {label: k for k, label in enumerate(dict.fromkeys(clusters))}
    rows = np.fromiter((index[label] for label in clusters), dtype=np.int64, count=len(clusters))
    totals = np.zeros((len(index), len(columns)), dtype=np.int64)
    for c, column in enumerate(columns):
        np.add.at(totals[:, c], rows, np.asarray(column, dtype=np.int64))
    return totals


def _resample(totals: np.ndarray, n_resamples: int, seed: int) -> np.ndarray:
    """Column sums of `n_resamples` bootstrap samples of the clusters (rows of `totals`).

    Drawing C clusters with replacement is the same as drawing how often each
    cluster occurs from a multinomial with equal probabilities.
    """
    n_clusters = len(totals)
    if n_clusters == 0:
        raise ValueError("no clusters to resample")
    rng = np.random.default_rng(seed)
    probabilities = np.full(n_clusters, 1.0 / n_clusters)
    blocks = []
    for start in range(0, n_resamples, _BLOCK):
        times_drawn = rng.multinomial(n_clusters, probabilities, size=min(_BLOCK, n_resamples - start))
        blocks.append(times_drawn @ totals)
    return np.concatenate(blocks)


def _interval(values: np.ndarray, level: float) -> tuple[float, float]:
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(values, [tail, 1.0 - tail])
    return float(low), float(high)


def _split_counts(counts: Sequence[EditCounts]) -> tuple[list[int], list[int]]:
    return [c.errors for c in counts], [c.reference_units for c in counts]


def bootstrap_interval(counts: Sequence[EditCounts], clusters: Sequence[Hashable],
                       n_resamples: int = DEFAULT_RESAMPLES, seed: int = DEFAULT_SEED,
                       level: float = 0.95) -> dict:
    """Error rate of one system with a percentile interval from resampling clusters.

    `counts` are per-clip edit counts (see `clip_counts`); `clusters` gives the
    speaker of each clip.
    """
    errors, units = _split_counts(counts)
    totals = _cluster_totals([errors, units], clusters)
    sums = _resample(totals, n_resamples, seed)
    if (sums[:, 1] == 0).any() or totals[:, 1].sum() == 0:
        raise ValueError("a resample has no reference units: leave out clips with an empty reference")
    low, high = _interval(sums[:, 0] / sums[:, 1], level)
    return {"rate": float(totals[:, 0].sum() / totals[:, 1].sum()), "ci_low": low, "ci_high": high,
            "level": level, "n_clusters": len(totals), "n_resamples": n_resamples, "seed": seed}


def paired_bootstrap(counts_a: Sequence[EditCounts], counts_b: Sequence[EditCounts],
                     clusters: Sequence[Hashable], n_resamples: int = DEFAULT_RESAMPLES,
                     seed: int = DEFAULT_SEED, level: float = 0.95) -> dict:
    """Paired cluster bootstrap of the difference rate(A) - rate(B) on the same clips.

    A is the new system and B the baseline; a negative difference means that A
    makes fewer errors. Both rates are recomputed on the same resampled
    speakers, so the comparison is paired.

    `p_value` is one-sided, for H0 "A is not better than B" (rate(A) >=
    rate(B)): the share of resamples in which the difference is >= 0, counted
    as (1 + number of such resamples) / (1 + n_resamples) so that it is never
    exactly zero. The interval is the percentile interval of the difference.
    """
    errors_a, units_a = _split_counts(counts_a)
    errors_b, units_b = _split_counts(counts_b)
    if units_a != units_b:
        raise ValueError("the two systems were not scored against the same references")
    totals = _cluster_totals([errors_a, errors_b, units_a], clusters)
    sums = _resample(totals, n_resamples, seed)
    if (sums[:, 2] == 0).any() or totals[:, 2].sum() == 0:
        raise ValueError("a resample has no reference units: leave out clips with an empty reference")
    differences = (sums[:, 0] - sums[:, 1]) / sums[:, 2]
    low, high = _interval(differences, level)
    units = totals[:, 2].sum()
    rate_a, rate_b = float(totals[:, 0].sum() / units), float(totals[:, 1].sum() / units)
    return {"rate_a": rate_a, "rate_b": rate_b, "difference": rate_a - rate_b, "ci_low": low, "ci_high": high,
            "p_value": float((1 + np.count_nonzero(differences >= 0)) / (1 + n_resamples)),
            "level": level, "n_clusters": len(totals), "n_resamples": n_resamples, "seed": seed}
