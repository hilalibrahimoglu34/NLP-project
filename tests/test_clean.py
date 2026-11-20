"""Tests for the clean-data step on a small generated raw folder (manifest and selection log only)."""

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from huasr import clean, subset
from huasr.shards import write_manifest

# clip id, split, speaker, session, raw_text, normalized_text, duration in ms
CLIPS = [
    ("c01", "train", "A", "s1", "Javítani kell a hídon.", "javtani kell a hdon", 2000),
    ("c02", "train", "A", "s1", "A díj 10 százalék.", "a dj ten százalék", 4000),
    ("c03", "train", "B", "s2", "", "kvül maradt two thousand and eight", 1000),
    ("c04", "validation", "C", "s3", "A víz tiszta.", "a vz tiszta", 2500),
    ("c05", "test", "D", "s4", "Kovač úr beszélt.", "kova úr beszélt", 3000),
    ("c06", "test", "D", "s4", "Nincs benne semmi.", "nincs benne semmi", 500),
    ("c07", "test", "D", "s5", "Javítani kell a hídon!", "javtani kell a hdon", 2200),
    ("c08", "validation", "C", "s3", "A six pack csomag.", "a six pack csomag", 2000),
]
SELECTION = [
    *[(clip_id, split, speaker, session, duration, "kept") for clip_id, split, speaker, session, _, _, duration in CLIPS],
    ("x01", "train", "None", "s1", 7200, "unknown_speaker"),
    ("x02", "train", "D", "s1", 3600, "speaker_assigned_to_other_split"),
    ("x03", "validation", "C", "s4", 3700, "session_assigned_to_other_split"),
    ("x04", "test", "", "s5", "", "unreadable_audio"),
]


def make_raw(folder: Path) -> None:
    folder.mkdir(parents=True)
    rows = []
    for clip_id, split, speaker, session, raw_text, normalized_text, duration in CLIPS:
        row = dict.fromkeys(subset.MANIFEST_COLUMNS, "")
        row.update(clip_id=clip_id, split=split, speaker_id=speaker, session_id=session, raw_text=raw_text,
                   normalized_text=normalized_text, duration_ms=duration, file=f"{clip_id}.flac",
                   shard=f"clips/{split}-000.tar", source_split=split)
        rows.append(row)
    write_manifest(folder / "manifest.csv", subset.MANIFEST_COLUMNS, rows)
    write_manifest(folder / "selection.csv", subset.SELECTION_COLUMNS,
                   [dict(zip(subset.SELECTION_COLUMNS, row)) for row in SELECTION])


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class CleanTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw, self.out = self.root / "raw", self.root / "clean"
        make_raw(self.raw)
        self.summary = clean.clean_voxpopuli(self.raw, self.out)
        self.rows = {row["clip_id"]: row for row in read_csv(self.out / "manifest.csv")}

    def column(self, name: str) -> list[str]:
        return [self.rows[clip_id][name] for clip_id, *_ in CLIPS]

    def test_manifest_keeps_the_original_fields_and_adds_the_reference(self):
        rows = read_csv(self.out / "manifest.csv")
        self.assertEqual(tuple(rows[0]), clean.CLEAN_COLUMNS)
        self.assertEqual(clean.CLEAN_COLUMNS[:len(subset.MANIFEST_COLUMNS)], subset.MANIFEST_COLUMNS)
        original = {row["clip_id"]: row for row in read_csv(self.raw / "manifest.csv")}
        for row in rows:
            for column in subset.MANIFEST_COLUMNS:
                self.assertEqual(row[column], original[row["clip_id"]][column])
        self.assertEqual(self.column("reference"),
                         ["javítani kell a hídon", "a díj ten százalék", "kvül maradt two thousand and eight",
                          "a víz tiszta", "kova úr beszélt", "nincs benne semmi", "javítani kell a hídon",
                          "a six pack csomag"])
        self.assertEqual(self.column("reference_status"),
                         ["restored", "restored", "no_raw_text", "restored", "unchanged", "unchanged", "restored",
                          "unchanged"])
        self.assertEqual((self.rows["c01"]["restored_words"], self.rows["c02"]["unaligned_words"]), ("2", "1"))
        self.assertEqual(self.rows["c01"]["chars_per_second"], "10.5")  # 21 characters in 2 seconds

    def test_english_number_words_are_flagged_and_nothing_is_removed(self):
        self.assertEqual(self.column("english_number_words"), ["0", "1", "3", "0", "0", "0", "0", "1"])
        self.assertEqual(len(self.rows), len(CLIPS))
        self.assertEqual(self.summary["english_number_words"], {
            "train": {"clips": 2, "raw_text_has_digits": 1, "no_raw_text": 1, "raw_text_without_digits": 0},
            "validation": {"clips": 1, "raw_text_has_digits": 0, "no_raw_text": 0, "raw_text_without_digits": 1},
            "test": {"clips": 0, "raw_text_has_digits": 0, "no_raw_text": 0, "raw_text_without_digits": 0}})
        self.assertEqual(clean.english_number_words("Two-thousand and EIGHT, kettő"), 3)
        self.assertEqual(clean.english_number_words("tenni fog onetime"), 0)  # only whole words count

    def test_references_that_also_occur_in_the_training_split_are_flagged(self):
        # c07 (test) has the same words as c01 (train); punctuation and case do not matter
        self.assertEqual(self.column("repeats_training_reference"), ["", "", "", "0", "0", "0", "1", "0"])
        self.assertEqual(self.summary["references_also_in_training"],
                         {"validation": {"clips": 0, "of": 2}, "test": {"clips": 1, "of": 3}})

    def test_every_changed_word_is_logged(self):
        self.assertEqual(read_csv(self.out / "repair_log.csv"), [
            {"clip_id": "c01", "split": "train", "word": "0", "before": "javtani", "after": "javítani"},
            {"clip_id": "c01", "split": "train", "word": "3", "before": "hdon", "after": "hídon"},
            {"clip_id": "c02", "split": "train", "word": "1", "before": "dj", "after": "díj"},
            {"clip_id": "c04", "split": "validation", "word": "1", "before": "vz", "after": "víz"},
            {"clip_id": "c07", "split": "test", "word": "0", "before": "javtani", "after": "javítani"},
            {"clip_id": "c07", "split": "test", "word": "3", "before": "hdon", "after": "hídon"}])

    def test_removal_log_lists_every_source_clip_that_was_left_out(self):
        rows = read_csv(self.out / "removal_log.csv")
        self.assertEqual(tuple(rows[0]), clean.REMOVAL_COLUMNS)
        self.assertEqual([(row["clip_id"], row["session_id"], row["stage"], row["reason"]) for row in rows], [
            ("x01", "s1", "selection", "unknown_speaker"), ("x02", "s1", "selection", "speaker_assigned_to_other_split"),
            ("x03", "s4", "selection", "session_assigned_to_other_split"), ("x04", "s5", "selection", "unreadable_audio")])
        self.assertEqual(self.summary["removed_before_cleaning"], {
            "session_assigned_to_other_split": {"clips": 1, "hours": 0.001},
            "speaker_assigned_to_other_split": {"clips": 1, "hours": 0.001},
            "unknown_speaker": {"clips": 1, "hours": 0.002}, "unreadable_audio": {"clips": 1, "hours": 0.0}})
        self.assertEqual(self.summary["removed_by_cleaning_rules"]["clips"], 0)

    def test_summary_reports_speakers_sessions_clips_and_hours_per_split(self):
        self.assertEqual(json.loads((self.out / "summary.json").read_text(encoding="utf-8")), self.summary)
        self.assertEqual(self.summary["splits"], {"train": {"clips": 3, "hours": 0.002, "speakers": 2, "sessions": 2},
                                                  "validation": {"clips": 2, "hours": 0.001, "speakers": 1, "sessions": 1},
                                                  "test": {"clips": 3, "hours": 0.002, "speakers": 1, "sessions": 2}})
        zero = {"train-validation": 0, "train-test": 0, "validation-test": 0}
        self.assertEqual((self.summary["speakers_in_two_splits"], self.summary["sessions_in_two_splits"]), (zero, zero))

    def test_summary_counts_shared_speakers_and_sessions_if_there_are_any(self):
        rows = read_csv(self.raw / "manifest.csv")
        rows[3]["speaker_id"], rows[3]["session_id"] = "A", "s4"  # the validation clip c04: a training speaker
        write_manifest(self.raw / "manifest.csv", subset.MANIFEST_COLUMNS, rows)
        summary = clean.clean_voxpopuli(self.raw, self.root / "leaky")
        self.assertEqual(summary["speakers_in_two_splits"], {"train-validation": 1, "train-test": 0, "validation-test": 0})
        self.assertEqual(summary["sessions_in_two_splits"], {"train-validation": 0, "train-test": 0, "validation-test": 1})

    def test_summary_counts_the_repair(self):
        self.assertEqual(self.summary["reference_repair"], {
            "letter": "í", "clips": {"no_raw_text": 1, "restored": 4, "unchanged": 3},
            "restored_words": 6, "restored_letters": 6, "clips_with_words_that_could_not_be_aligned": 2,
            "words_that_could_not_be_aligned": 2, "words_in_clips_without_raw_text": 6})

    def test_summary_reports_letters_that_normalized_text_lost(self):
        # "í" is restored in the reference; "č" is reported but not restored.
        self.assertEqual(self.summary["letters_of_raw_text_missing_from_normalized_text"], {
            "í": {"raw_text": 6, "normalized_text": 0, "reference": 6},
            "č": {"raw_text": 1, "normalized_text": 0, "reference": 0}})

    def test_summary_gives_characters_per_second_per_split(self):
        rates = self.summary["chars_per_second"]
        self.assertEqual((rates["validation"]["clips"], rates["validation"]["min"], rates["validation"]["max"]),
                         (2, 4.8, 8.5))
        self.assertEqual((rates["test"]["min"], rates["test"]["max"], rates["test"]["p50"]), (5.0, 34.0, 9.55))
        self.assertEqual(clean.chars_per_second("Jó  napot!", 1000), 8.0)
        self.assertEqual(clean.distribution([]), {"clips": 0})
        self.assertEqual(clean.distribution([4.8]), {"clips": 1, "min": 4.8, "p1": 4.8, "p5": 4.8, "p25": 4.8, "p50": 4.8,
                                                     "p75": 4.8, "p95": 4.8, "p99": 4.8, "max": 4.8, "mean": 4.8})

    def test_running_again_gives_the_same_files(self):
        first = {path.name: path.read_bytes() for path in self.out.iterdir()}
        clean.clean_voxpopuli(self.raw, self.out)
        self.assertEqual({path.name: path.read_bytes() for path in self.out.iterdir()}, first)
        self.assertEqual(sorted(first), ["manifest.csv", "removal_log.csv", "repair_log.csv", "summary.json"])

    def test_command_line(self):
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            code = clean.main(["voxpopuli", "--raw", str(self.raw), "--out", str(self.root / "again")])
        self.assertEqual(code, 0)
        self.assertIn("restored_words", printed.getvalue())
        self.assertEqual((self.root / "again" / "manifest.csv").read_bytes(), (self.out / "manifest.csv").read_bytes())
        with contextlib.redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(clean.main(["voxpopuli", "--raw", str(self.root / "missing"), "--out", str(self.out)]), 1)
        self.assertIn("manifest.csv", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
