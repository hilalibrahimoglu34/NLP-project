"""Compare two runs on the same clips with a paired bootstrap over speakers.

    python -m huasr.compare NEW BASELINE
    python -m huasr.compare results/own/hf-whisper-small-finetuned/validation \\
                            results/baselines/hf-whisper-small/validation

NEW and BASELINE are run folders with `run.json` and `predictions.csv`. The
two prediction files must hold the same clips in the same order with the same
references and flags; otherwise nothing is compared. The comparison is
written next to the new run as `comparison_with_<baseline system>.json`:

- the error rates of both systems and their difference, rate(new) minus
  rate(baseline), so a negative difference means fewer errors of the new
  system;
- the 95% percentile interval of the difference and a one-sided p-value for
  H0 "the new system is not better", both from `huasr.metrics.paired_bootstrap`
  (the same resampled speakers for both systems, fixed seed);
- for WER, CER and the diacritic-insensitive WER, once for all clips and once
  without the clips whose reference contains English number words.

Scoring is that of `huasr.score`: the policy of `huasr.text` on both sides,
corpus-level rates, clips with an empty reference left out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import metrics
from .score import read_predictions
from .text import normalize, to_chars, to_words

UNITS = {"wer": to_words, "cer": to_chars, "wer_no_diacritics": lambda text: to_words(text, True)}
SAME = ("clip_id", "speaker_id", "reference", "english_number_words")  # what both prediction files must agree on


def compare_clips(new: list[dict[str, str]], baseline: list[dict[str, str]]) -> dict:
    """Paired comparison of two lists of prediction rows that describe the same clips."""
    pairs = [(a, b) for a, b in zip(new, baseline) if normalize(a["reference"])]
    result = {"clips": len(pairs), "speakers": len({a["speaker_id"] for a, _ in pairs})}
    if not pairs:
        return result
    references = [a["reference"] for a, _ in pairs]
    speakers = [a["speaker_id"] for a, _ in pairs]
    for name, tokenize in UNITS.items():
        counts_new = metrics.clip_counts(references, [a["hypothesis"] for a, _ in pairs], tokenize)
        counts_baseline = metrics.clip_counts(references, [b["hypothesis"] for _, b in pairs], tokenize)
        paired = metrics.paired_bootstrap(counts_new, counts_baseline, speakers)
        result[name] = {"new": round(paired["rate_a"], 4), "baseline": round(paired["rate_b"], 4),
                        "difference": round(paired["difference"], 4),
                        "ci95": [round(paired["ci_low"], 4), round(paired["ci_high"], 4)],
                        "p_value": round(paired["p_value"], 5)}
    return result


def compare_runs(new: Path, baseline: Path) -> tuple[Path, dict]:
    """Write the comparison of the run in `new` with the run in `baseline`; return its path and content."""
    new, baseline = Path(new), Path(baseline)
    runs = [json.loads((folder / "run.json").read_text(encoding="utf-8")) for folder in (new, baseline)]
    rows = [read_predictions(folder) for folder in (new, baseline)]
    if runs[0]["split"] != runs[1]["split"] or len(rows[0]) != len(rows[1]) or any(
            a[column] != b[column] for a, b in zip(*rows) for column in SAME):
        raise ValueError(f"{new} and {baseline} do not hold the same clips with the same references")
    flagged = {row["clip_id"] for row in rows[0] if int(row["english_number_words"] or 0) > 0}
    rest = [[row for row in side if row["clip_id"] not in flagged] for side in rows]
    comparison = {
        "new": {"system": runs[0]["system"], "group": runs[0]["group"], "folder": new.as_posix()},
        "baseline": {"system": runs[1]["system"], "group": runs[1]["group"], "folder": baseline.as_posix()},
        "split": runs[0]["split"], "final_test": runs[0]["final_test"],
        "complete": bool(runs[0]["complete"] and runs[1]["complete"]),
        "scoring": {"policy": "huasr.text.normalize on reference and hypothesis", "rates": "corpus level",
                    "difference": "rate(new) - rate(baseline); negative means fewer errors of the new system",
                    "interval": "95% percentile interval of the difference, paired bootstrap over speakers",
                    "p_value": "one-sided, H0: the new system is not better than the baseline",
                    "resamples": metrics.DEFAULT_RESAMPLES, "seed": metrics.DEFAULT_SEED},
        "all_clips": compare_clips(*rows),
        "clips_with_english_number_words": len(flagged),
        "without_english_number_words": compare_clips(*rest),
    }
    path = new / f"comparison_with_{runs[1]['system']}.json"
    path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return path, comparison


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("new", type=Path, help="run folder of the new system")
    parser.add_argument("baseline", type=Path, help="run folder of the system it is compared with")
    args = parser.parse_args(argv)
    try:
        path, comparison = compare_runs(args.new, args.baseline)
    except (OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    everything = comparison["all_clips"]
    print(f"{comparison['new']['system']} against {comparison['baseline']['system']}, {comparison['split']}, "
          f"{everything['clips']} clips of {everything['speakers']} speakers")
    for name in UNITS:
        if name in everything:
            part = everything[name]
            print(f"  {name}: {part['baseline']:.2%} -> {part['new']:.2%}, difference {part['difference'] * 100:+.2f} "
                  f"points, 95% interval [{part['ci95'][0] * 100:+.2f}, {part['ci95'][1] * 100:+.2f}], "
                  f"p = {part['p_value']}")
    print(f"wrote {path.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
