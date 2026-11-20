"""Score the prediction file of a run.

    python -m huasr.score results/baselines/faster-whisper-small/validation [more run folders ...]
    python -m huasr.score --all          every run folder below results/

Reads `predictions.csv` and `run.json` of a run folder and writes
`scores.json` next to them. Every system is scored in the same way:

- reference and hypothesis both pass through the scoring policy of
  `huasr.text`;
- WER and CER are corpus-level (summed edit operations over summed reference
  units), with the numbers of substitutions, deletions and insertions;
- the diacritic-insensitive WER uses the same policy with diacritics removed
  on both sides;
- 95% intervals come from a bootstrap that resamples speakers
  (`huasr.metrics.bootstrap_interval`, fixed seed);
- the real-time factor is the processing time divided by the audio duration.

The figures are given twice: for all clips, and without the clips whose
reference contains English number words (the corpus wrote digits out in
English, see `data/clean/README.md`). The flag only splits the report; no clip
is dropped from the prediction file. A clip whose reference is empty after
the policy has no reference units; such clips are counted and left out of the
rates.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from . import metrics
from .evaluate import PREDICTION_COLUMNS, RESULTS
from .text import normalize, to_chars, to_words


def read_predictions(folder: Path) -> list[dict[str, str]]:
    with (Path(folder) / "predictions.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if rows and tuple(rows[0]) != PREDICTION_COLUMNS:
        raise ValueError(f"{Path(folder) / 'predictions.csv'} has columns {tuple(rows[0])}")
    return rows


def score_clips(rows: list[dict[str, str]]) -> dict:
    """Error rates, intervals and real-time factor of a list of prediction rows."""
    scored = [row for row in rows if normalize(row["reference"])]
    result = {"clips": len(scored), "clips_with_empty_reference": len(rows) - len(scored),
              "speakers": len({row["speaker_id"] for row in scored})}
    if not scored:
        return result
    references = [row["reference"] for row in scored]
    hypotheses = [row["hypothesis"] for row in scored]
    speakers = [row["speaker_id"] for row in scored]
    audio_seconds = sum(int(row["duration_ms"]) for row in scored) / 1000
    processing_seconds = sum(float(row["processing_seconds"]) for row in scored)
    result.update({"audio_seconds": round(audio_seconds, 3), "processing_seconds": round(processing_seconds, 3),
                   "rtf": round(processing_seconds / audio_seconds, 4),
                   "empty_hypotheses": sum(not normalize(hypothesis) for hypothesis in hypotheses)})
    units = {"wer": to_words, "cer": to_chars, "wer_no_diacritics": lambda text: to_words(text, True)}
    for name, tokenize in units.items():
        counts = metrics.clip_counts(references, hypotheses, tokenize)
        total = sum(counts, metrics.EditCounts())
        interval = metrics.bootstrap_interval(counts, speakers)
        result[name] = {"rate": round(total.rate, 4),
                        "ci95": [round(interval["ci_low"], 4), round(interval["ci_high"], 4)],
                        "substitutions": total.substitutions, "deletions": total.deletions,
                        "insertions": total.insertions, "reference_units": total.reference_units}
    return result


def score_run(folder: Path) -> dict:
    """Write `scores.json` for the run in `folder` and return its content."""
    folder = Path(folder)
    run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    rows = read_predictions(folder)
    if len(rows) != run["clips"]:
        raise ValueError(f"{folder}: run.json says {run['clips']} clips, predictions.csv has {len(rows)}")
    flagged = {row["clip_id"] for row in rows if int(row["english_number_words"] or 0) > 0}
    scores = {
        "system": run["system"], "group": run["group"], "split": run["split"], "final_test": run["final_test"],
        "complete": run["complete"], "checkpoint": run["checkpoint"], "settings": run["settings"],
        "hardware": run["hardware"], "clean_manifest_sha256": run["data"]["clean_manifest_sha256"],
        "scoring": {"policy": "huasr.text.normalize on reference and hypothesis", "rates": "corpus level",
                    "interval": "95% percentile bootstrap over speakers", "resamples": metrics.DEFAULT_RESAMPLES,
                    "seed": metrics.DEFAULT_SEED},
        "all_clips": score_clips(rows),
        "clips_with_english_number_words": len(flagged),
        "without_english_number_words": score_clips([row for row in rows if row["clip_id"] not in flagged]),
    }
    (folder / "scores.json").write_text(json.dumps(scores, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
                                        newline="\n")
    return scores


def run_folders(results: Path) -> list[Path]:
    """Every folder below `results` that holds a run."""
    return sorted(path.parent for path in Path(results).glob("*/*/*/run.json"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folders", nargs="*", type=Path, help="run folders (with run.json and predictions.csv)")
    parser.add_argument("--all", action="store_true", help="score every run below --results")
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args(argv)
    folders = run_folders(args.results) if args.all else args.folders
    if not folders:
        parser.error("give at least one run folder, or --all")
    try:
        for folder in folders:
            scores = score_run(folder)
            everything = scores["all_clips"]
            print(f"{folder.as_posix()}: {everything['clips']} clips, WER {everything['wer']['rate']:.2%} "
                  f"{everything['wer']['ci95']}, CER {everything['cer']['rate']:.2%} {everything['cer']['ci95']}, "
                  f"RTF {everything['rtf']}")
    except (OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
