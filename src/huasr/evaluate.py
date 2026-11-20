"""Run one system on one split of the data and store what it wrote.

    python -m huasr.evaluate --system faster-whisper-small [--split validation] [--limit N] [--force]
    python -m huasr.evaluate --system <name> --split test --final-test

Every system is run in the same way: the clips of the split are read from the
shards of `data/raw/voxpopuli/` in the form in which they are stored there,
the reference is the column `reference` of `data/clean/voxpopuli/manifest.csv`,
and the system only sees the audio. The run leaves two files in
`results/<group>/<system>/<split>/`:

    run.json         split, number of clips, SHA-256 of the data manifests, system, checkpoint id and
                     revision, settings, seed, hardware, package versions, start and end time
    predictions.csv  one row per clip: clip id, speaker, session, duration, the flag for English
                     number words, reference, hypothesis, processing time

`<group>` is `baselines` for checkpoints used as they are and `own` for models
trained in this project. Scores are computed from the prediction file by
`huasr.score`, so a run never has to be repeated to change the scoring.

The test split is for the final evaluation only, after every setting is
frozen. The runner refuses it unless `--final-test` is given, and the run
manifest records that the flag was used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import tarfile
import time
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np

from . import audio
from .shards import write_manifest
from .systems import GROUPS, SYSTEMS

RAW = Path("data/raw/voxpopuli")
CLEAN = Path("data/clean/voxpopuli")
RESULTS = Path("results")
SPLITS = ("train", "validation", "test")
PREDICTION_COLUMNS = ("clip_id", "speaker_id", "session_id", "duration_ms", "english_number_words", "reference",
                      "hypothesis", "processing_seconds")


class FinalTestRequired(ValueError):
    """The test split was asked for without the flag that marks the final evaluation."""


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_split(clean: Path, split: str) -> list[dict[str, str]]:
    """Rows of the clean manifest that belong to `split`, in the order of the manifest."""
    with (Path(clean) / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split]
    if not rows:
        raise ValueError(f"{Path(clean) / 'manifest.csv'} has no clip of the split {split!r}")
    return rows


def iter_audio(raw: Path, rows: list[dict[str, str]]) -> Iterator[tuple[dict[str, str], np.ndarray, int]]:
    """The stored audio of `rows`: (row, samples as float32 between -1 and 1, sample rate), shard by shard.

    Every clip is checked against the SHA-256 of the manifest before it is decoded.
    """
    by_shard: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        by_shard.setdefault(row["shard"], {})[row["file"]] = row
    for shard, wanted in sorted(by_shard.items()):
        found = 0
        with tarfile.open(Path(raw) / shard, "r:") as tar:
            for member in tar:
                row = wanted.get(member.name)
                if row is None:
                    continue
                data = tar.extractfile(member).read()
                if hashlib.sha256(data).hexdigest() != row["sha256"]:
                    raise ValueError(f"{shard}: {member.name} does not match the SHA-256 in the manifest")
                samples, rate = audio.decode_flac(data)
                found += 1
                yield row, samples.astype(np.float32) / 32768.0, rate
        if found != len(wanted):
            raise ValueError(f"{shard}: {len(wanted) - found} clips of the manifest are missing")


def cpu_name() -> str:
    """Name of the processor as the operating system reports it."""
    try:
        if sys.platform == "win32":
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").split("\n"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def environment() -> dict:
    """Hardware and package versions for the run manifest."""
    import av  # the versions of the modules that are really loaded

    packages = {"python": platform.python_version(), "numpy": np.__version__, "av": av.__version__}
    try:
        packages["huasr"] = metadata.version("huasr")
    except metadata.PackageNotFoundError:
        packages["huasr"] = "run from the source tree"
    return {"hardware": {"processor": cpu_name(), "machine": platform.machine(), "logical_cpus": os.cpu_count(),
                         "system": platform.platform()},
            "packages": packages}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run(system, split: str = "validation", raw: Path = RAW, clean: Path = CLEAN, results: Path = RESULTS,
        final_test: bool = False, limit: int | None = None, force: bool = False,
        clock: Callable[[], float] = time.perf_counter, progress: Callable[[str], None] | None = None) -> Path:
    """Transcribe the clips of `split` with `system`; write run.json and predictions.csv; return their folder."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    if split == "test" and not final_test:
        raise FinalTestRequired("the test split is only for the final evaluation after everything is frozen; "
                             "pass --final-test to run it")
    if system.group not in GROUPS:
        raise ValueError(f"unknown group {system.group!r}")
    folder = Path(results) / system.group / system.name / split
    if (folder / "run.json").exists() and not force:
        raise FileExistsError(f"{folder} already holds a run; use --force to replace it")

    rows = read_split(clean, split)
    in_split = len(rows)
    if limit is not None:
        rows = rows[:limit]
    description = system.describe()
    started = _utc_now()
    before = clock()
    load = getattr(system, "load", None)
    if load is not None:
        load()
    load_seconds = clock() - before

    predictions = {}
    for number, (row, samples, rate) in enumerate(iter_audio(raw, rows), start=1):
        before = clock()
        hypothesis = system.transcribe(samples, rate)
        spent = clock() - before
        predictions[row["clip_id"]] = {
            "clip_id": row["clip_id"], "speaker_id": row["speaker_id"], "session_id": row["session_id"],
            "duration_ms": row["duration_ms"], "english_number_words": row["english_number_words"],
            "reference": row["reference"], "hypothesis": hypothesis, "processing_seconds": round(spent, 3)}
        if progress is not None and number % 50 == 0:
            progress(f"{system.name}: {number} of {len(rows)} clips")
    ordered = [predictions[row["clip_id"]] for row in rows]

    folder.mkdir(parents=True, exist_ok=True)
    write_manifest(folder / "predictions.csv", PREDICTION_COLUMNS, ordered)
    packages = {**environment()["packages"], **description.get("packages", {})}
    manifest = {
        "system": system.name,
        "group": system.group,
        "checkpoint": description.get("checkpoint", {}),
        "settings": description.get("settings", {}),
        "seed": description.get("seed"),
        "split": split,
        "final_test": bool(final_test),
        "data": {"clean_manifest": (Path(clean) / "manifest.csv").as_posix(),
                 "clean_manifest_sha256": sha256_of(Path(clean) / "manifest.csv"),
                 "raw_manifest_sha256": sha256_of(Path(raw) / "manifest.csv"),
                 "reference_column": "reference", "clips_in_split": in_split},
        "clips": len(ordered),
        "limit": limit,
        "complete": len(ordered) == in_split,
        "audio_seconds": round(sum(int(row["duration_ms"]) for row in ordered) / 1000, 3),
        "processing_seconds": round(sum(row["processing_seconds"] for row in ordered), 3),
        "load_seconds": round(load_seconds, 3),
        "hardware": environment()["hardware"],
        "packages": packages,
        "started_utc": started,
        "finished_utc": _utc_now(),
    }
    (folder / "run.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
                                     newline="\n")
    return folder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--system", required=True, choices=sorted(SYSTEMS))
    parser.add_argument("--split", default="validation", choices=SPLITS)
    parser.add_argument("--final-test", action="store_true",
                        help="needed for the test split: the final evaluation, after everything is frozen")
    parser.add_argument("--limit", type=int, help="only the first N clips (for a trial run; marked as incomplete)")
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--clean", type=Path, default=CLEAN)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--force", action="store_true", help="replace an existing run")
    args = parser.parse_args(argv)
    try:
        folder = run(SYSTEMS[args.system](), args.split, args.raw, args.clean, args.results, args.final_test,
                     args.limit, args.force, progress=lambda line: print(line, flush=True))
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"wrote {folder / 'run.json'} and {folder / 'predictions.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
