"""Build the raw data of this project from the downloaded source files.

    python -m huasr.subset voxpopuli   [--source scratch/voxpopuli] [--out data/raw/voxpopuli]
                                       [--train-hours H] [--validation-hours H] [--test-hours H] [--seed N]
    python -m huasr.subset search      [--raw data/raw/voxpopuli]
    python -m huasr.subset fleurs      [--source scratch/fleurs] [--out data/raw/fleurs]
    python -m huasr.subset commonvoice --release <extracted .../hu> [--hours H] [--seed N]
    python -m huasr.subset unpack      --raw data/raw/voxpopuli [--out scratch/audio/voxpopuli]

VoxPopuli (training, development and main test data)
    Source: the parquet files of `facebook/voxpopuli`, folder `hu/`. Columns
    used: audio_id, audio (a struct with the bytes of one WAV file and its
    path), raw_text, normalized_text, gender, speaker_id.

    Speakers and recording sessions. Every clip has a speaker id (a few have
    "None") and belongs to a recording session: VoxPopuli cut each clip out
    of one recording file per `session_id`, and a clip id is
    "<session_id>-<language>_<date>-<time of the speech>_<number>". The part
    before the first underscore is used as the session. In the Hungarian data
    it is one numbered part of a plenary sitting day.

    The official split is NOT disjoint: the same speakers and the same
    sessions occur in train, validation and test. This builder makes its own
    split under three rules:

    A. Every speaker is in exactly one split.
    B. Every session is in exactly one split.
    C. A clip is only ever used in the official split it comes from (an
       official training clip never becomes a test clip, and the other way
       round).

    A clip is kept if its official split, the split of its speaker and the
    split of its session are the same. How speakers and sessions get their
    split:

    1. Clips without a speaker id are left out: nobody can check them.
    2. Every speaker and every session goes to the split in which it keeps
       the most audio. Training audio is plentiful and test and validation
       audio is scarce, so a second of test audio counts w_test times and a
       second of validation audio w_validation times as much as a second of
       training audio. A speaker counts only the clips whose session is in
       the same split, and a session only the clips whose speaker is;
       sessions and speakers are therefore assigned in turn until nothing
       changes. Exact ties are broken by SHA-256 of the seed and the id.
    3. The two weights are searched on a fixed grid (WEIGHTS). The pair is
       taken that, in this order, reaches the minimum of test audio, reaches
       the validation target, reaches the training target, and then gives
       the most test audio (and, among equals, the most training audio).
    4. The test split keeps every clip that is left. The clips of the
       validation and of the training split are ordered by
       SHA-256("<seed>:<clip id>") and taken until the next clip would pass
       the target of the split, so every speaker is drawn in proportion to
       their audio.

    The build stops unless no speaker and no session is in two splits of the
    result. `summary.json` records, for the official split and for the
    subset, clips, hours, speakers and sessions per split and the speakers
    and sessions shared by each pair of splits, and what each rule removed.
    `selection.csv` lists every source clip with its decision: kept, or the
    rule that left it out. `split_search.json` lists the result of every
    weight pair; `search` rewrites it from `selection.csv` alone, without any
    source file.

FLEURS (extra test set)
    Source: `data/hu_hu/test.tsv` and `data/hu_hu/audio/test.tar.gz` of
    `google/fleurs`. The whole official test split is kept. FLEURS has no
    speaker ids and no sessions; the sentence id (several recordings share
    one sentence) is stored instead.

Both builders store the audio as described in `huasr.audio` (WAV becomes
16-bit FLAC, compressed input is kept unchanged), packed into tar shards
below GitHub's file-size limit, with `manifest.csv` and `summary.json`.

Common Voice (external test set)
    Only a list of the chosen test clips is written, next to the extracted
    release and never inside the tracked part of the repository
    (`huasr.download.require_untracked`). No Common Voice audio is copied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import struct
import sys
import tarfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path, PurePosixPath

from . import audio
from .download import FLEURS, VOXPOPULI, DownloadError, HubSource, local_name, require_untracked
from .shards import MAX_SHARD_BYTES, ShardWriter, unpack, write_manifest

SPLITS = ("train", "validation", "test")
# Targets of the committed subset: hours of training and of validation audio to draw, and the least
# amount of test audio. The test split keeps everything the rules leave, so it can be larger.
DEFAULT_HOURS = {"train": 14.0, "validation": 2.5, "test": 2.0}
DEFAULT_SEED = 0
# How many times a second of validation or test audio may count as much as a second of training audio.
WEIGHTS = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 36, 40, 44, 48, 56, 64, 80, 96, 128)
MS_PER_HOUR = 3_600_000
UNKNOWN_SPEAKERS = frozenset({"", "None"})
VOXPOPULI_COLUMNS = ("audio_id", "audio", "raw_text", "normalized_text", "gender", "speaker_id")
SESSION = re.compile(r"(\d{8}-\d{4}-[A-Za-z]+(?:-[^_]+)?-[a-z]{2})_\d{8}-\d\d:\d\d:\d\d_\d+")
FLEURS_FIELDS = 7  # sentence id, file name, raw text, normalised text, characters, number of samples, gender
MANIFEST_COLUMNS = ("clip_id", "split", "speaker_id", "session_id", "sentence_id", "gender", "raw_text",
                    "normalized_text", "duration_ms", "file", "shard", "bytes", "sha256", "source_split",
                    "source_encoding", "source_bytes", "source_sha256", "limited_samples")
SELECTION_COLUMNS = ("clip_id", "source_split", "speaker_id", "session_id", "duration_ms", "decision")
KEPT = "kept"
REASONS = ("unknown_speaker", "speaker_assigned_to_other_split", "session_assigned_to_other_split",
           "beyond_validation_target", "beyond_train_target", "unreadable_audio")
DRAWN_SPLITS = ("validation", "train")  # clips drawn up to the target; the test split keeps all its clips
AUDIO_ERRORS = (ValueError, struct.error)


class SplitLeakError(ValueError):
    """A speaker or a session occurs in more than one split of the subset."""


@dataclass(frozen=True)
class Clip:
    """What the selection needs to know about one source clip."""

    clip_id: str
    source_split: str
    speaker_id: str
    duration_ms: int
    session_id: str = ""  # empty for corpora without recording sessions

    @property
    def speaker_known(self) -> bool:
        return self.speaker_id not in UNKNOWN_SPEAKERS


@dataclass(frozen=True)
class Selection:
    """Result of the split rules for a list of clips."""

    chosen: dict[str, str]     # clip id -> split, for every clip that is kept
    reasons: dict[str, str]    # clip id -> rule that left it out, for every other clip
    weights: dict[str, int]    # the weight pair that was taken
    speakers: dict[str, str]   # speaker id -> split
    sessions: dict[str, str]   # session id -> split
    search: list[dict]         # what every weight pair would keep


def _hours(milliseconds: int) -> float:
    return round(milliseconds / MS_PER_HOUR, 3)


def _tally(clips: list[Clip], sessions: bool = False) -> dict:
    counts = {"clips": len(clips), "hours": _hours(sum(clip.duration_ms for clip in clips)),
              "speakers": len({clip.speaker_id for clip in clips if clip.speaker_known})}
    if sessions:
        counts["sessions"] = len({clip.session_id for clip in clips})
    return counts


def _order(seed: int, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()


def file_name(clip_id: str) -> str:
    """Clip id as a file name: characters that Windows does not allow become '-'."""
    return re.sub(r'[\\/:*?"<>|\s]', "-", clip_id)


def session_of(clip_id: str) -> str:
    """Recording session of a VoxPopuli clip: its id up to the first underscore.

    VoxPopuli names a clip "<session_id>-<language>_<date>-<time>_<number>"
    and cuts all clips of one session id out of one recording file.
    """
    match = SESSION.fullmatch(clip_id)
    if not match:
        raise ValueError(f"clip id {clip_id!r} is not of the form <session>_<date>-<time>_<number>")
    return match[1]


# ---------------------------------------------------------------- selection

def shared(groups: dict[str, set[str]]) -> dict[str, int]:
    """Number of members shared by each pair of splits, e.g. {"train-validation": 0, ...}."""
    return {f"{a}-{b}": len(groups[a] & groups[b]) for a, b in combinations(groups, 2)}


def groups_by_split(clips: list[Clip], split_of: Callable[[Clip], str], key: Callable[[Clip], str]) -> dict[str, set]:
    """The speakers, sessions or days (whatever `key` returns) of the clips, per split; unknown speakers left out."""
    groups: dict[str, set[str]] = {split: set() for split in SPLITS}
    for clip in clips:
        if key(clip) not in UNKNOWN_SPEAKERS:
            groups[split_of(clip)].add(key(clip))
    return groups


def audio_cells(clips: list[Clip]) -> dict[tuple[str, str], list[list[int]]]:
    """For every (speaker, session) pair: milliseconds and clips in each official split."""
    cells: dict[tuple[str, str], list[list[int]]] = defaultdict(lambda: [[0] * len(SPLITS), [0] * len(SPLITS)])
    for clip in clips:
        if clip.speaker_known:
            cell, index = cells[clip.speaker_id, clip.session_id], SPLITS.index(clip.source_split)
            cell[0][index] += clip.duration_ms
            cell[1][index] += 1
    return dict(cells)


def assign(cells: dict[tuple[str, str], list[list[int]]], weights: tuple[int, ...],
           seed: int) -> tuple[dict[str, str], dict[str, str]]:
    """Give every speaker and every session the split in which it keeps the most weighted audio.

    `weights` has one factor per split, in the order of SPLITS. A speaker
    counts the clips whose session is in the same split and a session the
    clips whose speaker is, so the two are assigned in turn, sessions first,
    until nothing changes. A unit only moves when it gains by it, which makes
    the kept weighted audio grow with every change and the loop end.
    """
    of_speaker: dict[str, list] = defaultdict(list)
    of_session: dict[str, list] = defaultdict(list)
    for (speaker, session), (audio_ms, _) in cells.items():
        of_speaker[speaker].append((session, audio_ms))
        of_session[session].append((speaker, audio_ms))

    def best(unit: str, parts: list, partner: dict[str, str] | None, current: str | None) -> str:
        kept = [weights[i] * sum(audio_ms[i] for other, audio_ms in parts if partner is None or partner[other] == split)
                for i, split in enumerate(SPLITS)]
        if current is not None and kept[SPLITS.index(current)] == max(kept):
            return current
        return min((split for i, split in enumerate(SPLITS) if kept[i] == max(kept)),
                   key=lambda split: _order(seed, f"{unit}:{split}"))

    speakers = {speaker: best(speaker, parts, None, None) for speaker, parts in sorted(of_speaker.items())}
    sessions: dict[str, str] = {}
    for _ in range(1000):
        new_sessions = {session: best(session, parts, speakers, sessions.get(session))
                        for session, parts in sorted(of_session.items())}
        new_speakers = {speaker: best(speaker, parts, new_sessions, speakers[speaker])
                        for speaker, parts in sorted(of_speaker.items())}
        if new_sessions == sessions and new_speakers == speakers:
            return speakers, sessions
        sessions, speakers = new_sessions, new_speakers
    raise RuntimeError("the assignment of speakers and sessions did not settle")


def _kept(cells: dict, speakers: dict[str, str], sessions: dict[str, str]) -> dict[str, dict]:
    """Clips, milliseconds, speakers and sessions that an assignment keeps in each split."""
    result = {split: {"clips": 0, "ms": 0, "speakers": set(), "sessions": set()} for split in SPLITS}
    for (speaker, session), (audio_ms, counts) in cells.items():
        split = speakers[speaker]
        index = SPLITS.index(split)
        if sessions[session] == split and counts[index]:
            part = result[split]
            part["clips"] += counts[index]
            part["ms"] += audio_ms[index]
            part["speakers"].add(speaker)
            part["sessions"].add(session)
    return result


def select_by_hours(clips: list[Clip], target_hours: float, seed: int) -> list[Clip]:
    """Seeded choice of clips with a total duration of at most `target_hours`."""
    budget, total, chosen = target_hours * MS_PER_HOUR, 0, []
    for clip in sorted(clips, key=lambda clip: _order(seed, clip.clip_id)):
        total += clip.duration_ms
        if total > budget:
            break
        chosen.append(clip)
    return chosen


def choose(clips: list[Clip], hours: dict[str, float], seed: int, weights: tuple[int, ...] = WEIGHTS) -> Selection:
    """Apply the split rules: decision for every clip, and the weight search behind it.

    `hours` gives the training and validation targets and the minimum of test
    audio. Every pair of weights from `weights` is tried; see the module text
    for the order in which the results are compared.
    """
    cells = audio_cells(clips)
    wanted = {split: hours[split] * MS_PER_HOUR for split in SPLITS}
    search, best = [], None
    for test_weight in weights:
        for validation_weight in weights:
            speakers, sessions = assign(cells, (1, validation_weight, test_weight), seed)
            kept = _kept(cells, speakers, sessions)
            got = {split: kept[split]["ms"] for split in SPLITS}
            score = (min(got["test"], wanted["test"]), min(got["validation"], wanted["validation"]),
                     min(got["train"], wanted["train"]), got["test"], got["train"])
            search.append({"test_weight": test_weight, "validation_weight": validation_weight,
                           **{split: {"clips": kept[split]["clips"], "hours": _hours(got[split]),
                                      "speakers": len(kept[split]["speakers"]),
                                      "sessions": len(kept[split]["sessions"])} for split in SPLITS},
                           "targets_met": all(got[split] >= wanted[split] for split in SPLITS), "taken": False})
            if best is None or score > best[0]:
                best = (score, speakers, sessions, len(search) - 1)
    _, speakers, sessions, taken = best
    search[taken]["taken"] = True

    reasons: dict[str, str] = {}
    candidates: dict[str, list[Clip]] = {split: [] for split in SPLITS}
    for clip in clips:
        if not clip.speaker_known:
            reasons[clip.clip_id] = "unknown_speaker"
        elif speakers[clip.speaker_id] != clip.source_split:
            reasons[clip.clip_id] = "speaker_assigned_to_other_split"
        elif sessions[clip.session_id] != clip.source_split:
            reasons[clip.clip_id] = "session_assigned_to_other_split"
        else:
            candidates[clip.source_split].append(clip)
    chosen = {clip.clip_id: "test" for clip in candidates["test"]}
    for split in DRAWN_SPLITS:
        drawn = {clip.clip_id for clip in select_by_hours(candidates[split], hours[split], seed)}
        for clip in candidates[split]:
            if clip.clip_id in drawn:
                chosen[clip.clip_id] = split
            else:
                reasons[clip.clip_id] = f"beyond_{split}_target"
    taken_weights = {"validation": search[taken]["validation_weight"], "test": search[taken]["test_weight"]}
    return Selection(chosen, reasons, taken_weights, speakers, sessions, search)


def tally_removed(clips: list[Clip], reasons: dict[str, str]) -> dict[str, dict]:
    """Clips, hours, speakers and sessions that each rule removed from each official split."""
    return {reason: {split: _tally([clip for clip in clips
                                    if reasons.get(clip.clip_id) == reason and clip.source_split == split], True)
                     for split in SPLITS}
            for reason in REASONS if reason != "unreadable_audio"}


def leak_report(clips: list[Clip], split_of: Callable[[Clip], str]) -> dict[str, dict[str, int]]:
    """Speakers, sessions and sitting days that two splits share."""
    return {"speakers_in_two_splits": shared(groups_by_split(clips, split_of, lambda clip: clip.speaker_id)),
            "sessions_in_two_splits": shared(groups_by_split(clips, split_of, lambda clip: clip.session_id)),
            "sitting_days_in_two_splits": shared(groups_by_split(clips, split_of, lambda clip: clip.session_id[:8]))}


def search_record(selection: Selection, hours: dict[str, float], seed: int) -> dict:
    """Content of `split_search.json`: the targets, the weights tried and what each pair keeps."""
    return {"seed": seed, "minimum_test_hours": hours["test"], "validation_target_hours": hours["validation"],
            "training_target_hours": hours["train"], "weights_tried": list(WEIGHTS),
            "taken": selection.weights, "rows": selection.search}


def read_selection(raw: Path) -> list[Clip]:
    """Every readable source clip listed in `selection.csv` of a raw folder."""
    with (Path(raw) / "selection.csv").open(encoding="utf-8", newline="") as handle:
        return [Clip(row["clip_id"], row["source_split"], row["speaker_id"], int(row["duration_ms"]),
                     session_of(row["clip_id"]))
                for row in csv.DictReader(handle) if row["decision"] != "unreadable_audio"]


def write_search(raw: Path, hours: dict[str, float] | None = None, seed: int = DEFAULT_SEED) -> dict:
    """Write `split_search.json` next to `selection.csv`, from that file alone, and return its content."""
    hours = dict(DEFAULT_HOURS if hours is None else hours)
    record = search_record(choose(read_selection(raw), hours, seed), hours, seed)
    (Path(raw) / "split_search.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8", newline="\n")
    return record


# ---------------------------------------------------------------- reading the sources

def parquet_rows(folder: Path, split: str) -> Iterator[dict]:
    """Rows of `<folder>/<split>-*.parquet` as dictionaries, a few at a time."""
    import pyarrow.parquet as pq

    paths = sorted(Path(folder).glob(f"{split}-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no {split}-*.parquet in {folder}; run python -m huasr.download voxpopuli")
    for path in paths:
        parquet = pq.ParquetFile(path)
        names = parquet.schema_arrow.names
        missing = [column for column in VOXPOPULI_COLUMNS if column not in names]
        if missing:
            raise ValueError(f"{path.name} has no column {missing}; found {names}")
        for batch in parquet.iter_batches(batch_size=16, columns=list(VOXPOPULI_COLUMNS)):
            yield from batch.to_pylist()


def _audio_bytes(row: dict) -> bytes:
    data = (row.get("audio") or {}).get("bytes")
    if not data:
        raise ValueError("row without embedded audio bytes")
    return data


def scan(rows_of: Callable[[str], Iterator[dict]]) -> tuple[list[Clip], dict[str, tuple[str, str]]]:
    """First pass: id, official split, speaker, session and duration of every clip.

    Clips whose audio cannot be read are returned separately: clip id -> (official split, reason).
    """
    clips, unreadable, seen = [], {}, set()
    for split in SPLITS:
        for row in rows_of(split):
            clip_id = row["audio_id"]
            if clip_id in seen:
                raise ValueError(f"clip id {clip_id} occurs twice in the source")
            seen.add(clip_id)
            session = session_of(clip_id)
            try:
                info = audio.inspect(_audio_bytes(row))
                if info.frames == 0:
                    raise ValueError("audio without samples")
            except AUDIO_ERRORS as error:
                unreadable[clip_id] = (split, str(error))
                continue
            clips.append(Clip(clip_id, split, str(row["speaker_id"] or "").strip(), info.duration_ms, session))
    return clips, unreadable


def source_record(source: HubSource, folder: Path | None, check: bool) -> dict:
    """Description of the source for the summary; with `check`, the files must be the pinned ones."""
    record = {"repository": source.repo, "revision": source.revision, "licence": source.licence}
    if check:
        files = {}
        for path, (size, digest) in source.files.items():
            local = Path(folder) / local_name(path)
            if not local.is_file() or local.stat().st_size != size:
                raise FileNotFoundError(f"{local} is missing or not the pinned file ({size} bytes); "
                                        f"run python -m huasr.download first")
            files[path] = {"bytes": size, "sha256": digest}
        record["files"] = files
        record["note"] = "sizes checked by this build; SHA-256 values are the pinned ones checked by huasr.download"
    return record


# ---------------------------------------------------------------- writing

def _prepare(out: Path, force: bool) -> list[Path]:
    """Existing outputs in `out`; raise if there are any and `force` is not set."""
    outputs = (out / "manifest.csv", out / "summary.json", out / "selection.csv", out / "split_search.json",
               *sorted((out / "clips").glob("*.tar")))
    existing = [path for path in outputs if path.exists()]
    if existing and not force:
        raise FileExistsError(f"{out} already holds data ({existing[0].name}, ...); use --force to replace it")
    return existing


def _stored_row(clip_id: str, split: str, data: bytes, writer: ShardWriter, tally: Counter, **fields) -> dict:
    """Convert one source clip, add it to a shard and return its manifest row."""
    stored, extension, info, limited = audio.store(data)
    name = file_name(clip_id) + extension
    shard = writer.add(name, stored)
    tally[f"source:{info.encoding}"] += 1
    tally[f"stored:{'flac_s16' if info.container == 'wav' else info.encoding}"] += 1
    tally["limited_samples"] += limited
    return {"clip_id": clip_id, "split": split, "duration_ms": info.duration_ms, "file": name,
            "shard": f"clips/{shard}", "bytes": len(stored), "sha256": hashlib.sha256(stored).hexdigest(),
            "source_encoding": info.encoding, "source_bytes": len(data),
            "source_sha256": hashlib.sha256(data).hexdigest(), "limited_samples": limited, **fields}


def _finish(out: Path, manifest: list[dict], writers: dict[str, ShardWriter], summary: dict, tally: Counter) -> dict:
    names = [row["file"] for row in manifest]
    if len(set(names)) != len(names):
        raise ValueError("two clips would get the same file name")
    manifest.sort(key=lambda row: (SPLITS.index(row["split"]), row["clip_id"]))
    write_manifest(out / "manifest.csv", MANIFEST_COLUMNS, manifest)
    for split, writer in writers.items():
        shards = writer.close()
        summary["subset"].setdefault(split, {}).update(
            {"shards": len(shards), "shard_bytes": sum(size for _, size in shards)})
    summary["audio"] = {
        "source_encodings": {key[7:]: n for key, n in sorted(tally.items()) if key.startswith("source:")},
        "stored_encodings": {key[7:]: n for key, n in sorted(tally.items()) if key.startswith("stored:")},
        "samples_limited_to_16_bit_range": tally["limited_samples"],
        **summary.get("audio", {}),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
                                      newline="\n")
    return summary


def build_voxpopuli(source: Path | None, out: Path, hours: dict[str, float] | None = None, seed: int = DEFAULT_SEED,
                    max_shard_bytes: int = MAX_SHARD_BYTES, force: bool = False, check_source: bool = True,
                    rows_of: Callable[[str], Iterator[dict]] | None = None) -> dict:
    """Write shards, manifest, selection log, search record and summary of the VoxPopuli subset to `out`.

    Returns the summary. `rows_of(split)` yields the source rows of one
    official split; by default they are read from the parquet files in
    `source`.
    """
    hours = dict(DEFAULT_HOURS if hours is None else hours)
    out = Path(out)
    rows_of = rows_of or (lambda split: parquet_rows(source, split))
    existing = _prepare(out, force)
    origin = source_record(VOXPOPULI, source, check_source)

    clips, unreadable = scan(rows_of)
    selection = choose(clips, hours, seed)
    chosen, reasons = selection.chosen, selection.reasons
    kept = [clip for clip in clips if clip.clip_id in chosen]
    leaks = leak_report(kept, lambda clip: chosen[clip.clip_id])
    if any(leaks["speakers_in_two_splits"].values()) or any(leaks["sessions_in_two_splits"].values()):
        raise SplitLeakError(f"the subset shares speakers or sessions between splits: {leaks}")

    for path in existing:
        path.unlink()
    writers = {split: ShardWriter(out / "clips", split, max_shard_bytes) for split in SPLITS}
    manifest, tally = [], Counter()
    try:
        for split in SPLITS:
            for row in rows_of(split):
                if chosen.get(row["audio_id"]) != split:
                    continue
                manifest.append(_stored_row(
                    row["audio_id"], split, _audio_bytes(row), writers[split], tally,
                    speaker_id=str(row["speaker_id"]).strip(), session_id=session_of(row["audio_id"]), sentence_id="",
                    gender=row["gender"] or "", raw_text=row["raw_text"] or "",
                    normalized_text=row["normalized_text"] or "", source_split=split))
    finally:
        for writer in writers.values():
            writer.close()
    if len(manifest) != len(chosen):
        raise RuntimeError(f"{len(chosen)} clips were chosen but {len(manifest)} were written")
    log = [{"clip_id": clip.clip_id, "source_split": clip.source_split, "speaker_id": clip.speaker_id,
            "session_id": clip.session_id, "duration_ms": clip.duration_ms,
            "decision": KEPT if clip.clip_id in chosen else reasons[clip.clip_id]} for clip in clips]
    log += [{"clip_id": clip_id, "source_split": split, "speaker_id": "", "session_id": session_of(clip_id),
             "duration_ms": "", "decision": "unreadable_audio"} for clip_id, (split, _) in unreadable.items()]
    log.sort(key=lambda row: (SPLITS.index(row["source_split"]), row["clip_id"]))
    write_manifest(out / "selection.csv", SELECTION_COLUMNS, log)
    (out / "split_search.json").write_text(json.dumps(search_record(selection, hours, seed), indent=1) + "\n",
                                           encoding="utf-8", newline="\n")

    summary = {
        "corpus": "VoxPopuli, Hungarian (hu), transcribed set",
        "source": origin,
        "seed": seed,
        "target_hours": hours,
        "weights": selection.weights,
        "max_shard_bytes": max_shard_bytes,
        "official_split": {
            **{split: {**_tally([c for c in clips if c.source_split == split], True),
                       "clips_without_speaker_id": sum(1 for c in clips
                                                       if c.source_split == split and not c.speaker_known)}
               for split in SPLITS},
            **leak_report(clips, lambda clip: clip.source_split),
        },
        "subset": {
            **{split: _tally([c for c in kept if chosen[c.clip_id] == split], True) for split in SPLITS},
            **leaks,
        },
        "removed": tally_removed(clips, reasons),
        "audio": {"unreadable_clips": {clip_id: reason for clip_id, (_, reason) in unreadable.items()}},
    }
    return _finish(out, manifest, writers, summary, tally)


def read_fleurs_tsv(path: Path) -> dict[str, list[str]]:
    """Rows of a FLEURS transcript file (no header, seven tab-separated fields) by audio file name."""
    rows: dict[str, list[str]] = {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").split("\n"), start=1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != FLEURS_FIELDS or not fields[1].endswith(".wav") or not fields[5].isdigit():
            raise ValueError(f"{Path(path).name} line {number}: not the expected FLEURS layout")
        if fields[1] in rows:
            raise ValueError(f"{Path(path).name} line {number}: {fields[1]} is listed twice")
        rows[fields[1]] = fields
    return rows


def build_fleurs(source: Path, out: Path, max_shard_bytes: int = MAX_SHARD_BYTES, force: bool = False,
                 check_source: bool = True) -> dict:
    """Write shards, manifest and summary of the FLEURS test split to `out`; return the summary."""
    source, out = Path(source), Path(out)
    existing = _prepare(out, force)
    origin = source_record(FLEURS, source, check_source)
    rows = read_fleurs_tsv(source / "test.tsv")

    for path in existing:
        path.unlink()
    writers = {"test": ShardWriter(out / "clips", "test", max_shard_bytes)}
    manifest, tally, problems = [], Counter(), {"audio_without_transcript": 0, "length_differs_from_tsv": 0}
    unreadable, clips = {}, []
    try:
        with tarfile.open(source / "test.tar.gz", "r|gz") as tar:
            for member in tar:
                name = PurePosixPath(member.name).name
                if not member.isfile():
                    continue
                if name not in rows:
                    problems["audio_without_transcript"] += 1
                    continue
                sentence_id, _, raw_text, normalized_text, _, samples, gender = rows.pop(name)
                clip_id = name[:-len(".wav")]
                try:
                    row = _stored_row(clip_id, "test", tar.extractfile(member).read(), writers["test"], tally,
                                      speaker_id="", session_id="", sentence_id=sentence_id, gender=gender.lower(),
                                      raw_text=raw_text, normalized_text=normalized_text, source_split="test")
                except AUDIO_ERRORS as error:
                    unreadable[clip_id] = str(error)
                    continue
                problems["length_differs_from_tsv"] += abs(row["duration_ms"] - round(int(samples) / 16)) > 1
                manifest.append(row)
                clips.append(Clip(clip_id, "test", "", row["duration_ms"]))
    finally:
        writers["test"].close()

    summary = {
        "corpus": "FLEURS, Hungarian (hu_hu), test split",
        "source": origin,
        "max_shard_bytes": max_shard_bytes,
        "subset": {"test": {**_tally(clips), "sentences": len({row["sentence_id"] for row in manifest})}},
        "audio": {"unreadable_clips": unreadable, **problems, "transcripts_without_audio": len(rows)},
    }
    summary["subset"]["test"].pop("speakers")  # FLEURS has no speaker ids
    return _finish(out, manifest, writers, summary, tally)


# ---------------------------------------------------------------- Common Voice: list of test clips only

def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a Common Voice TSV: tab-separated, never quoted, first line is the header."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        rows = []
        for line, fields in enumerate(reader, start=2):
            if not fields:
                continue
            if len(fields) != len(header):
                raise ValueError(f"{Path(path).name} line {line}: {len(fields)} fields, header has {len(header)}")
            rows.append(dict(zip(header, fields)))
    return header, rows


def list_commonvoice_test(release: Path, hours: float, seed: int = DEFAULT_SEED, out: Path | None = None) -> dict:
    """Write `commonvoice_test.csv` and its summary: a seeded choice from the official test split.

    The clips stay where they were extracted; nothing is copied. The list is
    written next to the release (or to `out`), never into the tracked part of
    the repository. Column names are the documented ones of Common Voice
    (client_id, path, sentence; clip and duration[ms] in clip_durations.tsv).
    """
    release = require_untracked(release)
    out = require_untracked(out if out is not None else release.parent)
    header, rows = read_tsv(release / "test.tsv")
    duration_header, duration_rows = read_tsv(release / "clip_durations.tsv")
    for needed, found, name in ((("client_id", "path", "sentence"), header, "test.tsv"),
                                (("clip", "duration[ms]"), duration_header, "clip_durations.tsv")):
        missing = [column for column in needed if column not in found]
        if missing:
            raise ValueError(f"{name} has no column {missing}; found {found}")
    durations = {row["clip"]: int(row["duration[ms]"]) for row in duration_rows}
    by_id, left_out = {}, {"no_duration": 0, "no_audio": 0}
    for row in rows:
        if row["path"] not in durations:
            left_out["no_duration"] += 1
        elif not (release / "clips" / row["path"]).is_file():
            left_out["no_audio"] += 1
        else:
            by_id[row["path"]] = row
    clips = [Clip(path, "test", row["client_id"], durations[path]) for path, row in by_id.items()]
    chosen = sorted(select_by_hours(clips, hours, seed), key=lambda clip: clip.clip_id)
    manifest = []
    for clip in chosen:
        row, data = by_id[clip.clip_id], (release / "clips" / clip.clip_id).read_bytes()
        manifest.append({"clip_id": PurePosixPath(clip.clip_id).stem, "split": "test", "speaker_id": clip.speaker_id,
                         "session_id": "", "sentence_id": row.get("sentence_id", ""), "gender": row.get("gender", ""),
                         "raw_text": row["sentence"], "normalized_text": "", "duration_ms": clip.duration_ms,
                         "file": clip.clip_id, "shard": "", "bytes": len(data),
                         "sha256": hashlib.sha256(data).hexdigest(), "source_split": "test", "source_encoding": "mp3",
                         "source_bytes": len(data), "source_sha256": hashlib.sha256(data).hexdigest(),
                         "limited_samples": 0})
    out.mkdir(parents=True, exist_ok=True)
    write_manifest(out / "commonvoice_test.csv", MANIFEST_COLUMNS, manifest)
    summary = {"corpus": "Common Voice Scripted Speech 27.0, Hungarian (hu), official test split",
               "release": f"{release.parent.name}/{release.name}", "seed": seed, "target_hours": hours,
               "official_test": _tally([Clip(r["path"], "test", r["client_id"], durations.get(r["path"], 0))
                                        for r in rows]),
               "chosen": _tally(chosen), "left_out_before_selection": left_out,
               "note": "list only; the audio stays in the extracted release and is never committed"}
    (out / "commonvoice_test.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    return summary


# ---------------------------------------------------------------- command line

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    vox = commands.add_parser("voxpopuli", help="subset of VoxPopuli hu with disjoint speakers and sessions")
    vox.add_argument("--source", type=Path, default=Path("scratch/voxpopuli"), help="folder with the parquet files")
    vox.add_argument("--out", type=Path, default=Path("data/raw/voxpopuli"))
    vox.add_argument("--train-hours", type=float, default=DEFAULT_HOURS["train"], help="training audio to draw")
    vox.add_argument("--validation-hours", type=float, default=DEFAULT_HOURS["validation"],
                     help="validation audio to draw")
    vox.add_argument("--test-hours", type=float, default=DEFAULT_HOURS["test"], help="least amount of test audio")
    vox.add_argument("--seed", type=int, default=DEFAULT_SEED)
    fleurs = commands.add_parser("fleurs", help="FLEURS hu_hu test split -> data/raw/fleurs")
    fleurs.add_argument("--source", type=Path, default=Path("scratch/fleurs"), help="folder with test.tsv, test.tar.gz")
    fleurs.add_argument("--out", type=Path, default=Path("data/raw/fleurs"))
    for builder in (vox, fleurs):
        builder.add_argument("--force", action="store_true", help="replace existing data in --out")
    cv = commands.add_parser("commonvoice", help="list of Common Voice test clips (written outside the tracked files)")
    cv.add_argument("--release", type=Path, required=True, help="extracted Common Voice hu folder")
    cv.add_argument("--hours", type=float, default=DEFAULT_HOURS["test"])
    cv.add_argument("--seed", type=int, default=DEFAULT_SEED)
    unpacker = commands.add_parser("unpack", help="restore the audio files of one raw folder from its shards")
    unpacker.add_argument("--raw", type=Path, required=True, help="e.g. data/raw/voxpopuli")
    unpacker.add_argument("--out", type=Path, help="default: scratch/audio/<name of the raw folder>")
    search = commands.add_parser("search", help="what every weight pair keeps, from selection.csv -> split_search.json")
    search.add_argument("--raw", type=Path, default=Path("data/raw/voxpopuli"))
    args = parser.parse_args(argv)

    try:
        if args.command == "unpack":
            out = args.out or Path("scratch/audio") / args.raw.name
            print(f"unpacked {unpack(args.raw, out)} clips to {out}")
        elif args.command == "search":
            record = write_search(args.raw)
            for row in record["rows"]:
                if row["targets_met"] or row["taken"]:
                    print(f"test weight {row['test_weight']:3d}, validation weight {row['validation_weight']:3d}: "
                          f"test {row['test']['hours']:.2f} h, validation {row['validation']['hours']:.2f} h, "
                          f"training {row['train']['hours']:.2f} h{'  <- taken' if row['taken'] else ''}")
        elif args.command == "commonvoice":
            print(json.dumps(list_commonvoice_test(args.release, args.hours, args.seed), indent=2))
        elif args.command == "fleurs":
            summary = build_fleurs(args.source, args.out, force=args.force)
            print(json.dumps({key: summary[key] for key in ("subset", "audio")}, indent=2))
        else:
            hours = {split: getattr(args, f"{split}_hours") for split in SPLITS}
            summary = build_voxpopuli(args.source, args.out, hours, args.seed, force=args.force)
            shown = ("weights", "official_split", "subset", "removed", "audio")
            print(json.dumps({key: summary[key] for key in shown}, indent=2))
    except (OSError, ValueError, DownloadError) as error:  # missing or unexpected input, refused target
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
