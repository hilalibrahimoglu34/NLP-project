"""Tests for the raw-data builders on synthetic sources.

VoxPopuli: generated rows with the column names and clip ids of the real
parquet files. Like the real corpus, the fixture has the same speakers and
the same recording sessions in several official splits and some clips without
a speaker id. FLEURS and Common Voice: tiny generated files in a temporary
folder. No real recording or sentence is involved.
"""

import contextlib
import csv
import hashlib
import io
import json
import random
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from huasr import audio, subset
from huasr.download import DownloadError, HubSource
from huasr.subset import Clip, Selection, SplitLeakError, assign, audio_cells, build_fleurs, build_voxpopuli, choose
from tests import REPOSITORY, needs
from tests.fixtures import signal, wav

CLIP_MS = 250
# (speaker, session, official split, number of clips); "None" is how the corpus writes a missing speaker id.
# Speakers A, B and G have clips in several official splits; sessions 2 and 5 are shared by official train and test.
LAYOUT = [
    ("A", 1, "train", 6), ("A", 2, "test", 1),
    ("B", 2, "train", 2), ("B", 3, "validation", 2), ("B", 4, "test", 2),
    ("C", 4, "test", 3),
    ("D", 5, "test", 2),
    ("E", 6, "validation", 3),
    ("F", 5, "train", 5),
    ("G", 7, "train", 4), ("G", 6, "validation", 1),
    ("None", 1, "train", 2), ("None", 4, "test", 1),
]
ZERO = {"train-validation": 0, "train-test": 0, "validation-test": 0}


def seconds(value: float) -> float:
    """Seconds as hours, the unit of the targets."""
    return value / 3600


def session(number: int) -> str:
    return f"20180418-0900-PLENARY-{number}-hu"


# Training target 4 s, validation target 0.75 s, at least 1 s of test audio. What the rules must give:
#   test        B and C in session 4 (5 clips; taking D as well would cost F's training clips in session 5)
#   validation  E in session 6 (3 clips)
#   train       A in session 1, F in session 5, G in session 7 (15 clips)
TARGETS = {"train": seconds(4), "validation": seconds(0.75), "test": seconds(1.0)}


def make_rows(layout=LAYOUT) -> dict[str, list[dict]]:
    """Source rows per official split, shaped like the rows of the VoxPopuli parquet files."""
    rows: dict[str, list[dict]] = {split: [] for split in subset.SPLITS}
    number = 0
    for speaker, session_number, split, clips in layout:
        for _ in range(clips):
            number += 1
            clip_id = f"{session(session_number)}_20180418-21:23:{number:02d}_{number}"
            rows[split].append({
                "audio_id": clip_id, "language": 7,
                "audio": {"bytes": wav(signal(CLIP_MS / 1000, seed=number)), "path": clip_id + ".wav"},
                "raw_text": f"Mondat {number}.", "normalized_text": f"mondat {number}",
                "gender": "female" if number % 2 else "male", "speaker_id": speaker,
                "is_gold_transcript": True, "accent": "None"})
    return rows


def clips_of(rows) -> list[Clip]:
    return [Clip(row["audio_id"], split, row["speaker_id"], CLIP_MS, subset.session_of(row["audio_id"]))
            for split in subset.SPLITS for row in rows[split]]


def kept_by(selection: Selection, clips: list[Clip]) -> dict[str, dict[str, set]]:
    """Speakers, sessions and number of clips of every split of a selection."""
    result = {split: {"speakers": set(), "sessions": set(), "clips": 0} for split in subset.SPLITS}
    for clip in clips:
        split = selection.chosen.get(clip.clip_id)
        if split:
            result[split]["speakers"].add(clip.speaker_id)
            result[split]["sessions"].add(clip.session_id)
            result[split]["clips"] += 1
    return result


class TemporaryFolderTestCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.out = self.root / "raw"

    def manifest(self, name="manifest.csv", folder=None):
        with ((folder or self.out) / name).open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))


class SessionTests(unittest.TestCase):
    def test_session_is_the_clip_id_up_to_the_first_underscore(self):
        self.assertEqual(subset.session_of("20090202-0900-PLENARY-9-hu_20090202-17:43:14_5"), "20090202-0900-PLENARY-9-hu")
        self.assertEqual(subset.session_of("20181114-0900-PLENARY-hu_20181114-15:53:50_7"), "20181114-0900-PLENARY-hu")
        self.assertEqual(subset.session_of("20180912-0900-PLENARY-widetrim-hu_20180912-22:41:34_6"),
                         "20180912-0900-PLENARY-widetrim-hu")

    def test_other_clip_ids_are_an_error(self):
        for clip_id in ("clip0001", "20090202-0900-PLENARY-9-hu", "20090202-0900-PLENARY-9-hu_5", ""):
            with self.assertRaisesRegex(ValueError, "not of the form"):
                subset.session_of(clip_id)

    def test_the_official_split_of_the_fixture_shares_speakers_and_sessions(self):
        report = subset.leak_report(clips_of(make_rows()), lambda clip: clip.source_split)
        self.assertEqual(report["speakers_in_two_splits"], {"train-validation": 2, "train-test": 2, "validation-test": 1})
        self.assertEqual(report["sessions_in_two_splits"], {"train-validation": 0, "train-test": 2, "validation-test": 0})
        self.assertEqual(report["sitting_days_in_two_splits"], {"train-validation": 1, "train-test": 1, "validation-test": 1})


class AssignTests(unittest.TestCase):
    """`assign` on a few clips whose result can be worked out by hand (all clips have the same length)."""

    CLIPS = [Clip("a1", "train", "A", 1000, "s1"), Clip("a2", "train", "A", 1000, "s1"),
             Clip("a3", "train", "A", 1000, "s1"), Clip("a4", "test", "A", 1000, "s2"),
             Clip("b1", "train", "B", 1000, "s2"), Clip("b2", "train", "B", 1000, "s2"),
             Clip("b3", "train", "B", 1000, "s2"), Clip("b4", "train", "B", 1000, "s3"),
             Clip("c1", "test", "C", 1000, "s3"), Clip("c2", "test", "C", 1000, "s3")]

    def test_cells_add_up_the_audio_of_each_speaker_in_each_session(self):
        cells = audio_cells(self.CLIPS + [Clip("x", "train", "None", 1000, "s1")])
        self.assertEqual(cells[("A", "s1")], [[3000, 0, 0], [3, 0, 0]])
        self.assertEqual(cells[("A", "s2")], [[0, 0, 1000], [0, 0, 1]])
        self.assertEqual(cells[("C", "s3")], [[0, 0, 2000], [0, 0, 2]])
        self.assertEqual(len(cells), 5)  # the clip without speaker id is in no cell

    def test_speaker_and_session_go_where_they_keep_most_audio(self):
        # Test audio counts twice. A: 3 s of training audio against 1 s of test audio, a training speaker.
        # B: only training audio. C: only test audio. Session s2: A's test clip is not counted because A is a
        # training speaker, so B's training clips win. Session s3: 1 s of B (train) against 2 s of C (test).
        speakers, sessions = assign(audio_cells(self.CLIPS), (1, 1, 2), seed=0)
        self.assertEqual(speakers, {"A": "train", "B": "train", "C": "test"})
        self.assertEqual(sessions, {"s1": "train", "s2": "train", "s3": "test"})

    def test_the_weights_decide_who_gets_a_shared_session(self):
        cells = audio_cells(self.CLIPS)
        _, train_heavy = assign(cells, (3, 1, 1), seed=0)   # s3: 1 s of training audio counted three times
        _, test_heavy = assign(cells, (1, 1, 2), seed=0)    # s3: 2 s of test audio counted twice
        self.assertEqual((train_heavy["s3"], test_heavy["s3"]), ("train", "test"))

    def test_no_unit_would_keep_more_audio_in_another_split(self):
        rng = random.Random(1)
        for trial in range(30):
            clips = [Clip(f"c{n}", rng.choice(subset.SPLITS), f"spk{rng.randrange(6)}", rng.randrange(1, 9) * 500,
                          f"ses{rng.randrange(8)}") for n in range(80)]
            cells = audio_cells(clips)
            weights = (1, rng.choice((1, 2, 5)), rng.choice((1, 3, 8)))
            speakers, sessions = assign(cells, weights, seed=trial)

            def kept(speaker_split, session_split):
                return sum(weights[i] * audio_ms[i] for (speaker, session_id), (audio_ms, _) in cells.items()
                           for i, split in enumerate(subset.SPLITS)
                           if speaker_split[speaker] == split and session_split[session_id] == split)

            now = kept(speakers, sessions)
            for unit, table in [(u, speakers) for u in speakers] + [(u, sessions) for u in sessions]:
                for split in subset.SPLITS:
                    moved = {**table, unit: split}
                    other = kept(moved, sessions) if table is speakers else kept(speakers, moved)
                    self.assertLessEqual(other, now)
            self.assertEqual((speakers, sessions), assign(audio_cells(list(reversed(clips))), weights, seed=trial))


class ChooseTests(unittest.TestCase):
    def setUp(self):
        self.clips = clips_of(make_rows())
        self.selection = choose(self.clips, TARGETS, seed=0)
        self.kept = kept_by(self.selection, self.clips)

    def test_expected_split_of_the_fixture(self):
        self.assertEqual({split: part["speakers"] for split, part in self.kept.items()},
                         {"train": {"A", "F", "G"}, "validation": {"E"}, "test": {"B", "C"}})
        self.assertEqual({split: part["sessions"] for split, part in self.kept.items()},
                         {"train": {session(1), session(5), session(7)}, "validation": {session(6)},
                          "test": {session(4)}})
        self.assertEqual({split: part["clips"] for split, part in self.kept.items()},
                         {"train": 15, "validation": 3, "test": 5})

    def test_no_speaker_and_no_session_is_in_two_splits(self):
        report = subset.leak_report([clip for clip in self.clips if clip.clip_id in self.selection.chosen],
                                    lambda clip: self.selection.chosen[clip.clip_id])
        self.assertEqual((report["speakers_in_two_splits"], report["sessions_in_two_splits"]), (ZERO, ZERO))
        for clip in self.clips:
            if clip.clip_id in self.selection.chosen:
                split = self.selection.chosen[clip.clip_id]
                self.assertEqual((clip.source_split, self.selection.speakers[clip.speaker_id],
                                  self.selection.sessions[clip.session_id]), (split, split, split))

    def test_every_clip_is_either_kept_or_has_a_reason(self):
        chosen, reasons = self.selection.chosen, self.selection.reasons
        self.assertEqual(set(chosen) | set(reasons), {clip.clip_id for clip in self.clips})
        self.assertFalse(set(chosen) & set(reasons))
        by_speaker = {(clip.speaker_id, clip.source_split, clip.session_id): reasons.get(clip.clip_id)
                      for clip in self.clips}
        self.assertEqual(by_speaker[("None", "train", session(1))], "unknown_speaker")
        self.assertEqual(by_speaker[("A", "test", session(2))], "speaker_assigned_to_other_split")
        self.assertEqual(by_speaker[("B", "train", session(2))], "speaker_assigned_to_other_split")
        self.assertEqual(by_speaker[("G", "validation", session(6))], "speaker_assigned_to_other_split")
        # D only has test clips, but its session went to the training split with F's clips
        self.assertEqual(by_speaker[("D", "test", session(5))], "session_assigned_to_other_split")
        counts = {reason: {split: tally["clips"] for split, tally in per_split.items()}
                  for reason, per_split in subset.tally_removed(self.clips, reasons).items()}
        self.assertEqual(counts, {
            "unknown_speaker": {"train": 2, "validation": 0, "test": 1},
            "speaker_assigned_to_other_split": {"train": 2, "validation": 3, "test": 1},
            "session_assigned_to_other_split": {"train": 0, "validation": 0, "test": 2},
            "beyond_validation_target": {"train": 0, "validation": 0, "test": 0},
            "beyond_train_target": {"train": 0, "validation": 0, "test": 0}})

    def test_a_larger_test_minimum_takes_the_shared_session_from_the_training_split(self):
        selection = choose(self.clips, {**TARGETS, "test": seconds(1.5)}, seed=0)
        kept = kept_by(selection, self.clips)
        self.assertEqual(kept["test"]["speakers"], {"B", "C", "D"})
        self.assertEqual(kept["test"]["sessions"], {session(4), session(5)})
        self.assertEqual(kept["train"]["speakers"], {"A", "G"})  # F's clips are in session 5, now a test session
        lost = [clip for clip in self.clips if clip.speaker_id == "F"]
        self.assertEqual({selection.reasons[clip.clip_id] for clip in lost}, {"session_assigned_to_other_split"})

    def test_minimum_of_test_audio_comes_before_the_other_targets(self):
        selection = choose(self.clips, {"train": seconds(60), "validation": seconds(60), "test": seconds(2.0)}, seed=0)
        self.assertEqual(kept_by(selection, self.clips)["test"]["clips"], 8)  # every test clip with a speaker id

    def test_validation_and_training_clips_are_drawn_up_to_the_target_and_test_clips_are_all_kept(self):
        # With a training target of 2 s the rules can afford D as a third test speaker: A and G still have 2.5 s.
        targets = {"train": seconds(2.0), "validation": seconds(0.6), "test": seconds(1.0)}
        selection = choose(self.clips, targets, seed=0)
        kept = kept_by(selection, self.clips)
        self.assertEqual((kept["train"]["clips"], kept["validation"]["clips"], kept["test"]["clips"]), (8, 2, 7))
        self.assertEqual(kept["train"]["speakers"], {"A", "G"})
        reasons = list(selection.reasons.values())
        self.assertEqual((reasons.count("beyond_train_target"), reasons.count("beyond_validation_target")), (2, 1))

    def test_draw_is_seeded_and_a_smaller_target_is_part_of_a_larger_one(self):
        clips = [clip for clip in self.clips if clip.source_split == "train"]
        small = {clip.clip_id for clip in subset.select_by_hours(clips, seconds(1.0), seed=0)}
        large = {clip.clip_id for clip in subset.select_by_hours(clips, seconds(3.0), seed=0)}
        self.assertEqual((len(small), len(large)), (4, 12))
        self.assertTrue(small < large)
        self.assertEqual(small, {clip.clip_id for clip in subset.select_by_hours(list(reversed(clips)), seconds(1.0), 0)})
        self.assertNotEqual(small, {clip.clip_id for clip in subset.select_by_hours(clips, seconds(1.0), seed=1)})

    def test_result_does_not_depend_on_the_order_of_the_clips(self):
        again = choose(list(reversed(self.clips)), TARGETS, seed=0)
        self.assertEqual((again.chosen, again.reasons, again.weights), (self.selection.chosen, self.selection.reasons,
                                                                        self.selection.weights))

    def test_search_lists_every_weight_pair_and_marks_the_one_taken(self):
        search = self.selection.search
        self.assertEqual(len(search), len(subset.WEIGHTS) ** 2)
        taken = [row for row in search if row["taken"]]
        self.assertEqual(len(taken), 1)
        self.assertEqual((taken[0]["test_weight"], taken[0]["validation_weight"]),
                         (self.selection.weights["test"], self.selection.weights["validation"]))
        self.assertEqual({split: taken[0][split]["clips"] for split in subset.SPLITS},
                         {"train": 15, "validation": 3, "test": 5})
        self.assertEqual(taken[0]["test"], {"clips": 5, "hours": 0.0, "speakers": 2, "sessions": 1})
        # the training target of 4 s cannot be met together with 1 s of test audio: 3.75 s are left
        self.assertFalse(taken[0]["targets_met"])
        best_test = max(row["test"]["clips"] for row in search if row["train"]["clips"] >= 15
                        and row["validation"]["clips"] >= 3)
        self.assertEqual(best_test, 5)


@needs("av")
class BuildVoxpopuliTests(TemporaryFolderTestCase):
    def setUp(self):
        super().setUp()
        self.rows = make_rows()

    def build(self, rows=None, out=None, **options):
        rows = rows or self.rows
        options = {"hours": TARGETS, "check_source": False, "max_shard_bytes": 40_960, **options}
        return build_voxpopuli(None, out or self.out, rows_of=lambda split: iter(rows[split]), **options)

    def test_manifest_describes_every_chosen_clip(self):
        self.build()
        rows = self.manifest()
        self.assertEqual(tuple(rows[0]), subset.MANIFEST_COLUMNS)
        self.assertEqual([row["split"] for row in rows], ["train"] * 15 + ["validation"] * 3 + ["test"] * 5)
        source = {row["audio_id"]: (split, row) for split in subset.SPLITS for row in self.rows[split]}
        for row in rows:
            split, original = source[row["clip_id"]]
            self.assertEqual((row["split"], row["source_split"]), (split, split))
            self.assertEqual(row["speaker_id"], original["speaker_id"])
            self.assertEqual(row["session_id"], row["clip_id"].split("_")[0])
            self.assertEqual((row["raw_text"], row["normalized_text"], row["gender"], row["sentence_id"]),
                             (original["raw_text"], original["normalized_text"], original["gender"], ""))
            self.assertEqual(int(row["duration_ms"]), CLIP_MS)
            self.assertEqual(row["source_encoding"], "wav_f32le")
            self.assertEqual(int(row["source_bytes"]), len(original["audio"]["bytes"]))
            self.assertEqual(row["source_sha256"], hashlib.sha256(original["audio"]["bytes"]).hexdigest())
            self.assertEqual(row["limited_samples"], "0")
            self.assertNotIn(":", row["file"])
            self.assertTrue(row["file"].endswith(".flac"))

    def test_stored_audio_is_the_16_bit_version_of_the_source(self):
        self.build()
        self.assertEqual(subset.unpack(self.out, self.root / "audio"), 23)
        source = {row["audio_id"]: row["audio"]["bytes"] for split in subset.SPLITS for row in self.rows[split]}
        for row in self.manifest():
            stored = (self.root / "audio" / row["split"] / row["file"]).read_bytes()
            self.assertEqual((len(stored), hashlib.sha256(stored).hexdigest()), (int(row["bytes"]), row["sha256"]))
            decoded, rate = audio.decode_flac(stored)
            expected, _, _ = audio.wav_to_pcm16(source[row["clip_id"]])
            self.assertTrue(np.array_equal(decoded, expected))
            self.assertEqual(rate, 16_000)

    def test_summary_reports_the_leaks_of_the_official_split_and_a_clean_subset(self):
        summary = self.build()
        self.assertEqual(json.loads((self.out / "summary.json").read_text(encoding="utf-8")), summary)
        official, chosen = summary["official_split"], summary["subset"]
        self.assertEqual(official["speakers_in_two_splits"],
                         {"train-validation": 2, "train-test": 2, "validation-test": 1})
        self.assertEqual(official["sessions_in_two_splits"],
                         {"train-validation": 0, "train-test": 2, "validation-test": 0})
        self.assertEqual((chosen["speakers_in_two_splits"], chosen["sessions_in_two_splits"]), (ZERO, ZERO))
        self.assertEqual({split: (official[split]["clips"], official[split]["speakers"], official[split]["sessions"],
                                  official[split]["clips_without_speaker_id"]) for split in subset.SPLITS},
                         {"train": (19, 4, 4, 2), "validation": (6, 3, 2, 0), "test": (9, 4, 3, 1)})
        self.assertEqual({split: (chosen[split]["clips"], chosen[split]["speakers"], chosen[split]["sessions"])
                          for split in subset.SPLITS},
                         {"train": (15, 3, 3), "validation": (3, 1, 1), "test": (5, 2, 1)})
        self.assertEqual(summary["removed"]["unknown_speaker"]["train"]["clips"], 2)
        self.assertEqual(summary["removed"]["session_assigned_to_other_split"]["test"]["clips"], 2)
        self.assertEqual(summary["audio"]["source_encodings"], {"wav_f32le": 23})
        self.assertEqual(summary["audio"]["stored_encodings"], {"flac_s16": 23})
        self.assertEqual(summary["audio"]["unreadable_clips"], {})
        self.assertEqual((summary["seed"], summary["source"]["repository"]), (0, "facebook/voxpopuli"))
        self.assertEqual(set(summary["weights"]), {"validation", "test"})
        groups: dict[str, dict[str, set[str]]] = {"speaker_id": {}, "session_id": {}}
        for row in self.manifest():
            for column, per_split in groups.items():
                per_split.setdefault(row["split"], set()).add(row[column])
        self.assertEqual((subset.shared(groups["speaker_id"]), subset.shared(groups["session_id"])), (ZERO, ZERO))

    def test_selection_log_has_a_decision_for_every_source_clip(self):
        broken = self.rows["validation"][0]
        broken["audio"] = {"bytes": b"not audio", "path": "x.wav"}
        self.build()
        rows = self.manifest("selection.csv")
        self.assertEqual(tuple(rows[0]), subset.SELECTION_COLUMNS)
        source = {row["audio_id"]: (split, row) for split in subset.SPLITS for row in self.rows[split]}
        self.assertEqual({row["clip_id"] for row in rows}, set(source))
        kept = {row["clip_id"] for row in rows if row["decision"] == subset.KEPT}
        self.assertEqual(kept, {row["clip_id"] for row in self.manifest()})
        for row in rows:
            split, original = source[row["clip_id"]]
            self.assertEqual((row["source_split"], row["session_id"]), (split, row["clip_id"].split("_")[0]))
            self.assertIn(row["decision"], (subset.KEPT, *subset.REASONS))
            if row["decision"] != "unreadable_audio":
                self.assertEqual((row["speaker_id"], row["duration_ms"]), (original["speaker_id"], str(CLIP_MS)))
        decisions = {row["clip_id"]: row["decision"] for row in rows}
        self.assertEqual(decisions[broken["audio_id"]], "unreadable_audio")
        self.assertEqual(sum(decision == "unknown_speaker" for decision in decisions.values()), 3)
        self.assertEqual(sum(decision == "session_assigned_to_other_split" for decision in decisions.values()), 2)

    def test_search_record_is_written_and_can_be_rebuilt_from_the_selection_log_alone(self):
        self.build()
        record = json.loads((self.out / "split_search.json").read_text(encoding="utf-8"))
        self.assertEqual((record["minimum_test_hours"], record["validation_target_hours"],
                          record["training_target_hours"], record["seed"]),
                         (TARGETS["test"], TARGETS["validation"], TARGETS["train"], 0))
        self.assertEqual(record["weights_tried"], list(subset.WEIGHTS))
        self.assertEqual(len(record["rows"]), len(subset.WEIGHTS) ** 2)
        clips = subset.read_selection(self.out)
        self.assertEqual(sorted(clips, key=lambda clip: clip.clip_id),
                         sorted(clips_of(self.rows), key=lambda clip: clip.clip_id))
        (self.out / "split_search.json").unlink()
        self.assertEqual(subset.write_search(self.out, TARGETS), record)
        self.assertEqual(json.loads((self.out / "split_search.json").read_text(encoding="utf-8")), record)
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            self.assertEqual(subset.main(["search", "--raw", str(self.out)]), 0)  # with the default targets
        self.assertIn("<- taken", printed.getvalue())

    def test_shards_stay_below_the_limit(self):
        summary = self.build()
        shards = sorted((self.out / "clips").glob("*.tar"))
        self.assertEqual({f"clips/{path.name}" for path in shards}, {row["shard"] for row in self.manifest()})
        self.assertGreater(summary["subset"]["train"]["shards"], 1)
        self.assertTrue(all(path.stat().st_size < 40_960 for path in shards))
        self.assertEqual(sum(summary["subset"][split]["shard_bytes"] for split in subset.SPLITS),
                         sum(path.stat().st_size for path in shards))

    def test_building_twice_gives_identical_files(self):
        self.build()
        self.build(out=self.root / "again")
        for path in sorted(path for path in self.out.rglob("*") if path.is_file()):
            self.assertEqual(path.read_bytes(), (self.root / "again" / path.relative_to(self.out)).read_bytes())

    def test_existing_data_is_replaced_only_with_force(self):
        self.build()
        with self.assertRaises(FileExistsError):
            self.build()
        self.build(force=True, hours={**TARGETS, "train": seconds(1.0)})
        self.assertEqual({f"clips/{path.name}" for path in (self.out / "clips").glob("*.tar")},
                         {row["shard"] for row in self.manifest()})
        self.assertEqual(sum(row["split"] == "train" for row in self.manifest()), 4)

    def test_a_clip_that_cannot_be_read_is_left_out_and_listed(self):
        broken = self.rows["train"][0]
        broken["audio"] = {"bytes": broken["audio"]["bytes"][:60], "path": "x.wav"}
        summary = self.build()
        self.assertEqual(list(summary["audio"]["unreadable_clips"]), [broken["audio_id"]])
        self.assertIn("truncated", summary["audio"]["unreadable_clips"][broken["audio_id"]])
        self.assertNotIn(broken["audio_id"], {row["clip_id"] for row in self.manifest()})
        self.assertEqual(summary["subset"]["train"]["clips"], 14)

    def test_samples_outside_the_16_bit_range_are_counted(self):
        samples = signal(CLIP_MS / 1000)
        samples[:3] = 1.5
        self.rows["train"][0]["audio"]["bytes"] = wav(samples)
        summary = self.build()
        self.assertEqual(summary["audio"]["samples_limited_to_16_bit_range"], 3)
        row = next(row for row in self.manifest() if row["clip_id"] == self.rows["train"][0]["audio_id"])
        self.assertEqual(row["limited_samples"], "3")

    def test_the_same_clip_id_twice_is_an_error(self):
        self.rows["test"].append(dict(self.rows["train"][0]))
        with self.assertRaisesRegex(ValueError, "occurs twice"):
            self.build()

    def test_a_clip_id_without_a_session_is_an_error(self):
        self.rows["train"][0]["audio_id"] = "clip-without-session"
        with self.assertRaisesRegex(ValueError, "not of the form"):
            self.build()
        self.assertFalse(self.out.exists())

    def test_the_build_stops_if_the_subset_shares_a_speaker_or_a_session(self):
        clips = clips_of(self.rows)
        official = {clip.clip_id: clip.source_split for clip in clips if clip.speaker_known}
        only_sessions = {clip.clip_id: clip.source_split for clip in clips if clip.speaker_id in ("D", "F")}
        for leaking, expected in ((official, "'train-test': 2"), (only_sessions, "'train-test': 1")):
            selection = Selection(leaking, {}, {}, {}, {}, [])
            with mock.patch.object(subset, "choose", return_value=selection):
                with self.assertRaisesRegex(SplitLeakError, expected):
                    self.build()
            self.assertFalse(self.out.exists())


def write_parquet(folder: Path, rows: dict[str, list[dict]]) -> None:
    """Write the rows with the schema of the real VoxPopuli files; the training split in two files."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([
        ("audio_id", pa.string()), ("language", pa.int64()),
        ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        ("raw_text", pa.string()), ("normalized_text", pa.string()), ("gender", pa.string()),
        ("speaker_id", pa.string()), ("is_gold_transcript", pa.bool_()), ("accent", pa.string())])
    folder.mkdir(parents=True, exist_ok=True)
    half = len(rows["train"]) // 2
    parts = {"train-00000-of-00002": rows["train"][:half], "train-00001-of-00002": rows["train"][half:],
             "validation-00000-of-00001": rows["validation"], "test-00000-of-00001": rows["test"]}
    for name, part in parts.items():
        pq.write_table(pa.Table.from_pylist(part, schema=schema), folder / f"{name}.parquet", row_group_size=4)


@needs("av")
@needs("pyarrow")
class ParquetTests(TemporaryFolderTestCase):
    def setUp(self):
        super().setUp()
        self.rows = make_rows()
        self.source = self.root / "voxpopuli"
        write_parquet(self.source, self.rows)

    def test_rows_are_read_in_file_order_with_the_embedded_audio(self):
        read = list(subset.parquet_rows(self.source, "train"))
        self.assertEqual([row["audio_id"] for row in read], [row["audio_id"] for row in self.rows["train"]])
        self.assertEqual(read[0]["audio"]["bytes"], self.rows["train"][0]["audio"]["bytes"])
        self.assertEqual(set(read[0]), set(subset.VOXPOPULI_COLUMNS))

    def test_build_from_parquet_equals_build_from_the_same_rows(self):
        build_voxpopuli(self.source, self.out, TARGETS, check_source=False)
        build_voxpopuli(None, self.root / "memory", TARGETS, check_source=False,
                        rows_of=lambda split: iter(self.rows[split]))
        for name in ("manifest.csv", "summary.json", "selection.csv", "split_search.json"):
            self.assertEqual((self.out / name).read_bytes(), (self.root / "memory" / name).read_bytes())

    def test_a_missing_column_is_named(self):
        import pyarrow.parquet as pq

        path = self.source / "test-00000-of-00001.parquet"
        pq.write_table(pq.read_table(path).drop_columns(["speaker_id"]), path)
        with self.assertRaisesRegex(ValueError, "speaker_id"):
            list(subset.parquet_rows(self.source, "test"))

    def test_a_missing_split_points_to_the_download_command(self):
        (self.source / "validation-00000-of-00001.parquet").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "huasr.download voxpopuli"):
            build_voxpopuli(self.source, self.out, TARGETS, check_source=False)

    def test_files_other_than_the_pinned_ones_are_refused_by_default(self):
        with self.assertRaisesRegex(FileNotFoundError, "pinned"):
            build_voxpopuli(self.source, self.out, TARGETS)
        self.assertFalse(self.out.exists())

    def test_command_line(self):
        arguments = ["voxpopuli", "--source", str(self.source), "--out", str(self.out), "--train-hours",
                     str(seconds(2)), "--validation-hours", str(seconds(0.75)), "--test-hours", str(seconds(1)),
                     "--seed", "3"]
        with mock.patch.object(subset, "source_record", return_value={"repository": "synthetic"}), \
                contextlib.redirect_stdout(io.StringIO()) as printed:
            self.assertEqual(subset.main(arguments), 0)
            self.assertEqual(subset.main(["unpack", "--raw", str(self.out), "--out", str(self.root / "audio")]), 0)
        self.assertIn("sessions_in_two_splits", printed.getvalue())
        summary = json.loads((self.out / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual((summary["seed"], summary["subset"]["train"]["clips"]), (3, 8))
        self.assertEqual(len(list((self.root / "audio").rglob("*.flac"))), len(self.manifest()))


class SourceRecordTests(TemporaryFolderTestCase):
    def test_pinned_files_must_be_present_with_their_size(self):
        source = HubSource("someone/corpus", "b" * 40, "CC0-1.0", {"hu/test-00000-of-00001.parquet": (5, "c" * 64)})
        with self.assertRaisesRegex(FileNotFoundError, "huasr.download"):
            subset.source_record(source, self.root, check=True)
        (self.root / "test-00000-of-00001.parquet").write_bytes(b"12345")
        record = subset.source_record(source, self.root, check=True)
        self.assertEqual(record["files"], {"hu/test-00000-of-00001.parquet": {"bytes": 5, "sha256": "c" * 64}})
        self.assertEqual((record["revision"], record["licence"]), ("b" * 40, "CC0-1.0"))
        (self.root / "test-00000-of-00001.parquet").write_bytes(b"123456")
        with self.assertRaises(FileNotFoundError):
            subset.source_record(source, self.root, check=True)
        self.assertNotIn("files", subset.source_record(source, None, check=False))


def make_fleurs(folder: Path) -> dict[str, bytes]:
    """Write a tiny test.tsv and test.tar.gz in the FLEURS layout; return the audio by clip id."""
    folder.mkdir(parents=True)
    sounds = {str(1000 + n): wav(signal(0.2 + 0.05 * n, seed=n)) for n in range(5)}
    lines = []
    for n, (clip_id, data) in enumerate(sounds.items()):
        samples = audio.inspect(data).frames + (160 if n == 4 else 0)  # the last row states a wrong length
        lines.append("\t".join([str(7 + n // 2), f"{clip_id}.wav", f"Mondat {n}.", f"mondat {n}",
                                " ".join(f"mondat {n}"), str(samples), "MALE" if n % 2 else "FEMALE"]))
    lines.append("\t".join(["99", "4242.wav", "Nincs hang.", "nincs hang", "n i n c s", "1600", "MALE"]))
    (folder / "test.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    with tarfile.open(folder / "test.tar.gz", "w:gz") as tar:
        directory = tarfile.TarInfo("test")
        directory.type = tarfile.DIRTYPE
        tar.addfile(directory)
        for name, data in {**sounds, "777": wav(signal(0.1))}.items():  # 777 has no transcript
            info = tarfile.TarInfo(f"test/{name}.wav")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return sounds


@needs("av")
class FleursTests(TemporaryFolderTestCase):
    def setUp(self):
        super().setUp()
        self.source = self.root / "fleurs"
        self.sounds = make_fleurs(self.source)

    def test_whole_test_split_is_stored_with_sentence_ids(self):
        summary = build_fleurs(self.source, self.out, check_source=False)
        rows = self.manifest()
        self.assertEqual(tuple(rows[0]), subset.MANIFEST_COLUMNS)
        self.assertEqual([row["clip_id"] for row in rows], sorted(self.sounds))
        self.assertEqual({row["split"] for row in rows}, {"test"})
        self.assertEqual({row["speaker_id"] for row in rows}, {""})  # FLEURS has no speaker ids
        self.assertEqual([row["sentence_id"] for row in rows], ["7", "7", "8", "8", "9"])
        self.assertEqual([row["gender"] for row in rows], ["female", "male", "female", "male", "female"])
        self.assertEqual((rows[1]["raw_text"], rows[1]["normalized_text"]), ("Mondat 1.", "mondat 1"))
        self.assertEqual([int(row["duration_ms"]) for row in rows], [200, 250, 300, 350, 400])
        for row in rows:
            self.assertEqual(row["source_sha256"], hashlib.sha256(self.sounds[row["clip_id"]]).hexdigest())
        self.assertEqual(summary["subset"]["test"], {"clips": 5, "hours": 0.0, "sentences": 3, "shards": 1,
                                                     "shard_bytes": (self.out / "clips" / "test-000.tar").stat().st_size})
        self.assertEqual(summary["audio"]["audio_without_transcript"], 1)
        self.assertEqual(summary["audio"]["transcripts_without_audio"], 1)
        self.assertEqual(summary["audio"]["length_differs_from_tsv"], 1)
        self.assertEqual(summary["source"]["licence"], "CC BY 4.0")

    def test_stored_audio_is_the_16_bit_version_of_the_source(self):
        build_fleurs(self.source, self.out, check_source=False)
        subset.unpack(self.out, self.root / "audio")
        for row in self.manifest():
            decoded, _ = audio.decode_flac((self.root / "audio" / "test" / row["file"]).read_bytes())
            self.assertTrue(np.array_equal(decoded, audio.wav_to_pcm16(self.sounds[row["clip_id"]])[0]))

    def test_unexpected_transcript_layout_is_an_error(self):
        path = self.source / "test.tsv"
        path.write_text("id\tfile.wav\tonly three\n", encoding="utf-8", newline="\n")
        with self.assertRaisesRegex(ValueError, "FLEURS layout"):
            build_fleurs(self.source, self.out, check_source=False)

    def test_command_line_checks_the_pinned_files(self):
        with contextlib.redirect_stderr(io.StringIO()) as errors:
            code = subset.main(["fleurs", "--source", str(self.source), "--out", str(self.out)])
        self.assertEqual(code, 1)
        self.assertIn("pinned", errors.getvalue())
        self.assertFalse(self.out.exists())


def make_common_voice(root: Path) -> Path:
    """A synthetic extracted Common Voice release with only what the test list needs."""
    release = root / "cv-corpus-0.0-synthetic" / "hu"
    (release / "clips").mkdir(parents=True)
    rng = random.Random(0)
    header = ["client_id", "path", "sentence_id", "sentence", "sentence_domain", "up_votes", "down_votes",
              "age", "gender", "accents", "variant", "locale", "segment"]
    rows, durations = [], ["clip\tduration[ms]"]
    for number in range(12):
        name = f"common_voice_hu_{number:08d}.mp3"
        (release / "clips" / name).write_bytes(rng.randbytes(500 + number))
        durations.append(f"{name}\t{3000 + 100 * number}")
        rows.append("\t".join([f"spk{number % 5}", name, f"s{number}", f'Mondat "{number}".', "", "2", "0", "", "",
                               "", "", "hu", ""]))
    (release / "test.tsv").write_text("\n".join(["\t".join(header), *rows]) + "\n", encoding="utf-8", newline="\n")
    (release / "clip_durations.tsv").write_text("\n".join(durations) + "\n", encoding="utf-8", newline="\n")
    return release


class CommonVoiceListTests(TemporaryFolderTestCase):
    def setUp(self):
        super().setUp()
        self.release = make_common_voice(self.root)

    def test_list_is_a_seeded_choice_written_next_to_the_release(self):
        summary = subset.list_commonvoice_test(self.release, seconds(20), seed=0)
        rows = self.manifest("commonvoice_test.csv", self.release.parent)
        self.assertEqual(tuple(rows[0]), subset.MANIFEST_COLUMNS)
        self.assertLessEqual(sum(int(row["duration_ms"]) for row in rows), 20_000)
        self.assertTrue(0 < len(rows) < 12)
        self.assertEqual(summary["chosen"]["clips"], len(rows))
        self.assertEqual(summary["official_test"], {"clips": 12, "hours": 0.012, "speakers": 5})
        for row in rows:
            data = (self.release / "clips" / row["file"]).read_bytes()
            self.assertEqual((row["sha256"], row["shard"], row["split"]), (hashlib.sha256(data).hexdigest(), "", "test"))
            self.assertRegex(row["raw_text"], r'^Mondat "\d+"\.$')  # quotes are ordinary characters in these files
        again = subset.list_commonvoice_test(self.release, seconds(20), seed=0, out=self.root / "again")
        self.assertEqual(self.manifest("commonvoice_test.csv", self.root / "again"), rows)
        self.assertEqual(again, summary)
        subset.list_commonvoice_test(self.release, seconds(20), seed=1, out=self.root / "other")
        self.assertNotEqual(self.manifest("commonvoice_test.csv", self.root / "other"), rows)

    def test_no_audio_is_copied(self):
        subset.list_commonvoice_test(self.release, seconds(20), out=self.root / "list")
        self.assertEqual(sorted(path.name for path in (self.root / "list").iterdir()),
                         ["commonvoice_test.csv", "commonvoice_test.json"])

    def test_the_list_is_never_written_into_the_tracked_part_of_the_repository(self):
        for target in (REPOSITORY / "data" / "raw" / "commonvoice", REPOSITORY / "data" / "clean", REPOSITORY):
            with self.assertRaisesRegex(DownloadError, "re-hosting"):
                subset.list_commonvoice_test(self.release, seconds(20), out=target)
        self.assertFalse((REPOSITORY / "data" / "raw" / "commonvoice").exists())
        self.assertFalse((REPOSITORY / "commonvoice_test.csv").exists())

    def test_a_release_extracted_into_the_tracked_part_is_refused(self):
        with self.assertRaisesRegex(DownloadError, "re-hosting"):
            subset.list_commonvoice_test(REPOSITORY / "data" / "raw" / "cv-corpus-27.0-2026-09-11" / "hu", 1.0)

    def test_official_column_names_are_required(self):
        path = self.release / "test.tsv"
        path.write_text(path.read_text(encoding="utf-8").replace("client_id", "speaker", 1), encoding="utf-8",
                        newline="\n")
        with self.assertRaisesRegex(ValueError, "client_id"):
            subset.list_commonvoice_test(self.release, 1.0)

    def test_clips_without_duration_or_audio_are_counted(self):
        (self.release / "clips" / "common_voice_hu_00000000.mp3").unlink()
        path = self.release / "clip_durations.tsv"
        lines = [line for line in path.read_text(encoding="utf-8").split("\n") if "00000001" not in line]
        path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
        summary = subset.list_commonvoice_test(self.release, 1.0)
        self.assertEqual(summary["left_out_before_selection"], {"no_duration": 1, "no_audio": 1})
        self.assertEqual(summary["chosen"]["clips"], 10)


if __name__ == "__main__":
    unittest.main()
