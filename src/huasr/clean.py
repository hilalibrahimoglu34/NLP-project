"""Clean data of the project, derived from `data/raw/` only.

    python -m huasr.clean voxpopuli [--raw data/raw/voxpopuli] [--out data/clean/voxpopuli]

What the cleaning does at this stage:

1. Reference repair (`huasr.repair`). The corpus field `normalized_text` lost
   every letter "í". It is restored word by word where `raw_text` proves it;
   the result is the column `reference`. `raw_text` and `normalized_text`
   stay in the manifest unchanged, and every changed word is logged.
2. Flags; nothing is removed because of them.
   - `english_number_words`: the corpus wrote numbers out in English inside
     the Hungarian text ("2008" became "two thousand and eight"). The column
     counts the English number words of the reference.
   - `repeats_training_reference`: for validation and test clips, 1 if the
     same reference (after the scoring policy) also occurs in the training
     split.
   - `chars_per_second`: a transcript that does not belong to its audio tends
     to have an extreme value.
3. Removal log. Every source clip that is not in the clean data is listed
   with the rule that left it out. So far these are the rules of the split
   (`selection.csv` of the raw folder: clips without speaker id, speakers or
   sessions that belong to another split, clips beyond a target).
4. The summary with the numbers that have to be reported for the split:
   per split the speakers, sessions, clips and hours, the speakers and
   sessions shared by two splits, and the validation and test references
   that occur word for word in the training split.

No clip is removed by a cleaning rule yet; the summary says so. The audio is
not copied: a clean clip is found through `file` and `shard` in the shards
of the raw folder.

Outputs in `--out`: `manifest.csv`, `repair_log.csv`, `removal_log.csv`,
`summary.json`. Running the command again gives the same files.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from .repair import LOST_LETTER, letter_counts, missing_letters, repair
from .shards import read_manifest, write_manifest
from .subset import KEPT, MANIFEST_COLUMNS, MS_PER_HOUR, SELECTION_COLUMNS, SPLITS, UNKNOWN_SPEAKERS, shared
from .text import normalize, to_words

CLEAN_COLUMNS = (*MANIFEST_COLUMNS, "reference", "reference_status", "restored_words", "unaligned_words",
                 "english_number_words", "repeats_training_reference", "chars_per_second")
REPAIR_COLUMNS = ("clip_id", "split", "word", "before", "after")
REMOVAL_COLUMNS = ("clip_id", "source_split", "speaker_id", "session_id", "duration_ms", "stage", "reason")
PERCENTILES = (1, 5, 25, 50, 75, 95, 99)
ENGLISH_NUMBER_WORDS = frozenset(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million "
    "billion".split())


def chars_per_second(text: str, duration_ms: int) -> float:
    """Characters of the text after the scoring policy (spaces included) per second of audio."""
    return len(normalize(text)) * 1000 / duration_ms


def english_number_words(text: str) -> int:
    """How many words of the text are English number words ("two", "thousand", ...)."""
    return sum(word in ENGLISH_NUMBER_WORDS for word in to_words(text))


def distribution(values: list[float]) -> dict:
    """Minimum, maximum, mean and percentiles of a list of numbers, rounded to two decimals."""
    if not values:
        return {"clips": 0}
    points = np.percentile(np.asarray(values), PERCENTILES)
    return {"clips": len(values), "min": round(min(values), 2),
            **{f"p{percent}": round(float(point), 2) for percent, point in zip(PERCENTILES, points)},
            "max": round(max(values), 2), "mean": round(float(np.mean(values)), 2)}


def _read_selection(raw: Path) -> list[dict[str, str]]:
    path = raw / "selection.csv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if rows and tuple(rows[0]) != SELECTION_COLUMNS:
        raise ValueError(f"{path} has columns {tuple(rows[0])}, expected {SELECTION_COLUMNS}")
    return rows


def _groups(rows: list[dict], column: str) -> dict[str, set[str]]:
    return {split: {row[column] for row in rows if row["split"] == split and row[column] not in UNKNOWN_SPEAKERS}
            for split in SPLITS}


def clean_voxpopuli(raw: Path, out: Path) -> dict:
    """Write the clean manifest, both logs and the summary for the raw folder `raw`; return the summary."""
    raw, out = Path(raw), Path(out)
    rows = read_manifest(raw)
    manifest, repair_log, statuses = [], [], Counter()
    rates: dict[str, list[float]] = {split: [] for split in SPLITS}
    restored_letters = clips_with_unaligned = unaligned_words = 0
    for row in rows:
        fixed = repair(row["normalized_text"], row["raw_text"])
        statuses[fixed.status] += 1
        unaligned_words += fixed.unaligned_words
        clips_with_unaligned += fixed.unaligned_words > 0 and fixed.status != "no_raw_text"
        for position, before, after in fixed.changes:
            restored_letters += after.count(LOST_LETTER)
            repair_log.append({"clip_id": row["clip_id"], "split": row["split"], "word": position,
                               "before": before, "after": after})
        rate = chars_per_second(fixed.reference, int(row["duration_ms"]))
        rates[row["split"]].append(rate)
        manifest.append({**row, "reference": fixed.reference, "reference_status": fixed.status,
                         "restored_words": len(fixed.changes), "unaligned_words": fixed.unaligned_words,
                         "english_number_words": english_number_words(fixed.reference),
                         "repeats_training_reference": "", "chars_per_second": round(rate, 2)})

    training = {normalize(row["reference"]) for row in manifest if row["split"] == "train"}
    repeats = {}
    for split in SPLITS[1:]:
        part = [row for row in manifest if row["split"] == split]
        for row in part:
            row["repeats_training_reference"] = int(normalize(row["reference"]) in training)
        repeats[split] = {"clips": sum(row["repeats_training_reference"] for row in part), "of": len(part)}

    numbers = {}
    for split in SPLITS:
        flagged = [row for row in manifest if row["split"] == split and row["english_number_words"]]
        with_digits = sum(any(char.isdigit() for char in row["raw_text"]) for row in flagged)
        without_raw = sum(not row["raw_text"].strip() for row in flagged)
        numbers[split] = {"clips": len(flagged), "raw_text_has_digits": with_digits, "no_raw_text": without_raw,
                          "raw_text_without_digits": len(flagged) - with_digits - without_raw}

    removal_log = [{**{column: row[column] for column in REMOVAL_COLUMNS[:5]}, "stage": "selection",
                    "reason": row["decision"]} for row in _read_selection(raw) if row["decision"] != KEPT]
    removed = Counter(row["reason"] for row in removal_log)
    removed_ms = Counter()
    for row in removal_log:
        removed_ms[row["reason"]] += int(row["duration_ms"] or 0)

    with_raw = [row for row in rows if row["raw_text"].strip()]
    raw_letters = letter_counts(row["raw_text"] for row in with_raw)
    normalized_letters = letter_counts(row["normalized_text"] for row in with_raw)
    reference_letters = letter_counts(row["reference"] for row in manifest if row["raw_text"].strip())
    missing = missing_letters(raw_letters, normalized_letters)
    for letter, counts in missing.items():
        counts["reference"] = reference_letters[letter]

    out.mkdir(parents=True, exist_ok=True)
    write_manifest(out / "manifest.csv", CLEAN_COLUMNS, manifest)
    write_manifest(out / "repair_log.csv", REPAIR_COLUMNS, repair_log)
    write_manifest(out / "removal_log.csv", REMOVAL_COLUMNS, removal_log)
    speakers, sessions = _groups(rows, "speaker_id"), _groups(rows, "session_id")
    summary = {
        "corpus": "VoxPopuli, Hungarian (hu): clean references for the clips of the raw folder",
        "splits": {split: {
            "clips": sum(row["split"] == split for row in rows),
            "hours": round(sum(int(row["duration_ms"]) for row in rows if row["split"] == split) / MS_PER_HOUR, 3),
            "speakers": len(speakers[split]),
            "sessions": len(sessions[split]),
        } for split in SPLITS},
        "speakers_in_two_splits": shared(speakers),
        "sessions_in_two_splits": shared(sessions),
        "references_also_in_training": repeats,
        "english_number_words": numbers,
        "reference_repair": {
            "letter": LOST_LETTER,
            "clips": dict(sorted(statuses.items())),
            "restored_words": len(repair_log),
            "restored_letters": restored_letters,
            "clips_with_words_that_could_not_be_aligned": int(clips_with_unaligned),
            "words_that_could_not_be_aligned": unaligned_words - sum(
                row["unaligned_words"] for row in manifest if row["reference_status"] == "no_raw_text"),
            "words_in_clips_without_raw_text": sum(
                row["unaligned_words"] for row in manifest if row["reference_status"] == "no_raw_text"),
        },
        "letters_of_raw_text_missing_from_normalized_text": missing,
        "chars_per_second": {split: distribution(rates[split]) for split in SPLITS},
        "removed_before_cleaning": {reason: {"clips": count, "hours": round(removed_ms[reason] / MS_PER_HOUR, 3)}
                                    for reason, count in sorted(removed.items())},
        "removed_by_cleaning_rules": {"clips": 0, "note": "no cleaning rule removes clips yet"},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
                                      newline="\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    vox = commands.add_parser("voxpopuli", help="repair the references of data/raw/voxpopuli -> data/clean/voxpopuli")
    vox.add_argument("--raw", type=Path, default=Path("data/raw/voxpopuli"))
    vox.add_argument("--out", type=Path, default=Path("data/clean/voxpopuli"))
    args = parser.parse_args(argv)
    try:
        summary = clean_voxpopuli(args.raw, args.out)
    except (OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    shown = ("splits", "speakers_in_two_splits", "sessions_in_two_splits", "references_also_in_training",
             "english_number_words", "reference_repair")
    print(json.dumps({key: summary[key] for key in shown}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
