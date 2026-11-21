"""Write the data tables of the READMEs from the generated summary files.

    python -m huasr.report           rewrite the generated blocks of the READMEs
    python -m huasr.report --check   exit with 1 if a block is not up to date

No number about the data or about a result is typed into a README by hand.
The builders write `data/raw/voxpopuli/summary.json`,
`data/raw/voxpopuli/split_search.json` and
`data/clean/voxpopuli/summary.json`, and the scorer writes a `scores.json` for
every run below `results/`; this module turns them into Markdown tables and
puts each table between a pair of marker lines in the READMEs:

    <!-- generated: NAME -->
    ...
    <!-- end generated -->

A block whose source file does not exist yet says so instead of showing a
table.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SPLITS = ("train", "validation", "test")
RAW_SUMMARY = "data/raw/voxpopuli/summary.json"
SEARCH = "data/raw/voxpopuli/split_search.json"
CLEAN_SUMMARY = "data/clean/voxpopuli/summary.json"
RESULTS = "results"
READMES = ("README.md", "data/raw/README.md", "data/clean/README.md", "results/README.md")
RESULT_BLOCKS = {"results-baselines": "baselines", "results-own": "own"}  # block name -> folder below results/
GMM_HMM = "baselines/gmm-hmm-mfa"                                         # the runs of huasr.gmm_hmm, below results/
BLOCK = re.compile(r"(<!-- generated: (?P<name>[a-z-]+) -->\n)(?P<body>.*?)(<!-- end generated -->)", re.DOTALL)


def _table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def _pairs(overlap: dict[str, int]) -> str:
    return ", ".join(f"{pair.replace('-', ' and ')} {count}" for pair, count in overlap.items())


def table_official(raw: dict) -> str:
    official = raw["official_split"]
    rows = [[split, f"{official[split]['clips']:,}", f"{official[split]['hours']:.2f}", official[split]["speakers"],
             f"{official[split]['sessions']:,}", official[split]["clips_without_speaker_id"]] for split in SPLITS]
    return (_table(["Official split", "Clips", "Hours", "Speakers", "Sessions", "Clips without speaker id"], rows)
            + f"\n\nSpeakers shared by two official splits: {_pairs(official['speakers_in_two_splits'])}. "
              f"Sessions shared by two official splits: {_pairs(official['sessions_in_two_splits'])}.")


def table_subset(raw: dict) -> str:
    chosen = raw["subset"]
    rows = [[split, f"{chosen[split]['clips']:,}", f"{chosen[split]['hours']:.2f}", chosen[split]["speakers"],
             chosen[split]["sessions"], chosen[split]["shards"], f"{chosen[split]['shard_bytes'] / 1e6:.1f}"]
            for split in SPLITS]
    rows.append(["total", f"{sum(chosen[s]['clips'] for s in SPLITS):,}",
                 f"{sum(chosen[s]['hours'] for s in SPLITS):.2f}", sum(chosen[s]["speakers"] for s in SPLITS),
                 f"{sum(chosen[s]['sessions'] for s in SPLITS):,}", sum(chosen[s]["shards"] for s in SPLITS),
                 f"{sum(chosen[s]['shard_bytes'] for s in SPLITS) / 1e6:.1f}"])
    hours, weights = raw["target_hours"], raw["weights"]
    return (_table(["Split", "Clips", "Hours", "Speakers", "Sessions", "Shards", "Size (MB)"], rows)
            + f"\n\nSpeakers shared by two splits of the subset: {_pairs(chosen['speakers_in_two_splits'])}. "
              f"Sessions shared by two splits of the subset: {_pairs(chosen['sessions_in_two_splits'])}. "
              f"Settings: training target {hours['train']:g} h, validation target {hours['validation']:g} h, at "
              f"least {hours['test']:g} h of test audio; weights taken: validation {weights['validation']}, test "
              f"{weights['test']}; seed {raw['seed']}.")


def table_removed(raw: dict) -> str:
    rows = [[reason.replace("_", " ")] + [f"{per_split[split]['clips']:,} ({per_split[split]['hours']:.2f} h)"
                                         for split in SPLITS]
            for reason, per_split in raw["removed"].items()]
    return _table(["Rule that left clips out", "official train", "official validation", "official test"], rows)


def table_search(search: dict) -> str:
    def cells(row: dict) -> list:
        return [f"{row['test']['hours']:.2f} ({row['test']['speakers']}, {row['test']['sessions']})",
                f"{row['validation']['hours']:.2f} ({row['validation']['speakers']})",
                f"{row['train']['hours']:.2f} ({row['train']['speakers']})"]

    shown, seen = [], set()
    ordered = sorted(search["rows"], key=lambda row: (-row["taken"], -row["test"]["hours"], -row["train"]["hours"]))
    for row in ordered:
        if (row["targets_met"] or row["taken"]) and tuple(cells(row)) not in seen:
            seen.add(tuple(cells(row)))
            shown.append([row["test_weight"], row["validation_weight"], *cells(row),
                          "yes" if row["targets_met"] else "no", "taken" if row["taken"] else ""])
    met = sum(row["targets_met"] for row in search["rows"])
    return (_table(["Test weight", "Validation weight", "Test hours (speakers, sessions)",
                    "Validation hours (speakers)", "Training hours left (speakers)", "All targets met", ""], shown)
            + f"\n\n{len(search['rows'])} weight pairs were tried (weights {', '.join(map(str, search['weights_tried']))}); "
              f"{met} of them leave at least {search['minimum_test_hours']:g} h of test audio, "
              f"{search['validation_target_hours']:g} h of validation audio and {search['training_target_hours']:g} h "
              f"of training audio. The table shows the different results among these, and the pair that was taken.")


def table_split_report(clean: dict) -> str:
    repeats = clean["references_also_in_training"]
    rows = [[split, clean["splits"][split]["speakers"], clean["splits"][split]["sessions"],
             f"{clean['splits'][split]['clips']:,}", f"{clean['splits'][split]['hours']:.2f}",
             f"{repeats[split]['clips']} of {repeats[split]['of']:,}" if split in repeats else "-"]
            for split in SPLITS]
    return (_table(["Split", "Speakers", "Sessions", "Utterances", "Hours",
                    "References that occur word for word in the training split"], rows)
            + f"\n\nSpeakers in two splits: {_pairs(clean['speakers_in_two_splits'])}. "
              f"Sessions in two splits: {_pairs(clean['sessions_in_two_splits'])}.")


def table_number_words(clean: dict) -> str:
    numbers = clean["english_number_words"]
    labels = (("clips", "References with English number words"),
              ("raw_text_has_digits", "of these: `raw_text` has the number in digits"),
              ("no_raw_text", "of these: the clip has no `raw_text`"),
              ("raw_text_without_digits", "of these: `raw_text` has no digits (English words in the speech itself)"))
    rows = [[label] + [f"{numbers[split][key]:,}" for split in SPLITS] for key, label in labels]
    return _table(["English number words", *SPLITS], rows)


def table_repair(clean: dict) -> str:
    fixed = clean["reference_repair"]
    clips = fixed["clips"]
    rows = [["Clips with at least one restored word", f"{clips.get('restored', 0):,}"],
            ["Restored words", f"{fixed['restored_words']:,}"],
            ["Clips with nothing to restore", f"{clips.get('unchanged', 0):,}"],
            ["Clips without `raw_text` (cannot be checked)", f"{clips.get('no_raw_text', 0):,}"],
            ["Clips whose `normalized_text` is not in normalised form (left alone)", f"{clips.get('unusual_text', 0):,}"],
            ["Words that could not be aligned with `raw_text` (left unchanged)",
             f"{fixed['words_that_could_not_be_aligned']:,} in "
             f"{fixed['clips_with_words_that_could_not_be_aligned']:,} clips"]]
    letters = [[f"`{letter}`", f"{counts['raw_text']:,}", f"{counts['normalized_text']:,}", f"{counts['reference']:,}"]
               for letter, counts in clean["letters_of_raw_text_missing_from_normalized_text"].items()]
    return (_table(["Reference repair", "Count"], rows) + "\n\n"
            + _table(["Letter missing from `normalized_text`", "in `raw_text`", "in `normalized_text`",
                      "in `reference`"], letters))


def table_rate(clean: dict) -> str:
    keys = ("clips", "min", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "max")
    rows = [[split] + [clean["chars_per_second"][split].get(key, "") for key in keys] for split in SPLITS]
    return _table(["Split", "Clips", "min", "1%", "5%", "25%", "median", "75%", "95%", "99%", "max"], rows)


BLOCKS = {"voxpopuli-official-split": (table_official, RAW_SUMMARY), "voxpopuli-subset": (table_subset, RAW_SUMMARY),
          "voxpopuli-removed": (table_removed, RAW_SUMMARY), "voxpopuli-search": (table_search, SEARCH),
          "voxpopuli-split-report": (table_split_report, CLEAN_SUMMARY),
          "voxpopuli-number-words": (table_number_words, CLEAN_SUMMARY),
          "voxpopuli-repair": (table_repair, CLEAN_SUMMARY), "voxpopuli-chars-per-second": (table_rate, CLEAN_SUMMARY)}


def _percent(part: dict) -> str:
    return f"{part['rate'] * 100:.2f} [{part['ci95'][0] * 100:.2f}, {part['ci95'][1] * 100:.2f}]"


def _checkpoint(checkpoint: dict) -> str:
    """Repository and revision of a published checkpoint; for a system trained here, what its run says."""
    if "repository" not in checkpoint:
        return checkpoint.get("description", "")
    return f"`{checkpoint['repository']}` at `{checkpoint.get('revision', '')[:12]}`"


def table_gmm_hmm(root: Path) -> str:
    """What limits the GMM-HMM baseline, one row per complete run, from its run.json."""
    rows = []
    for path in sorted((Path(root) / RESULTS / GMM_HMM).glob("*/run.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        if not run["complete"]:
            continue
        trained, missing, output = run["checkpoint"]["training_data"], run["out_of_vocabulary"], run["output"]
        rows.append((SPLITS.index(run["split"]), [
            run["split"],
            f"{trained['clips']:,} ({trained['audio_seconds'] / 3600:.2f} h, {trained['speakers']} speakers)",
            f"{run['settings']['lexicon']['words']:,}",
            f"{missing['not_in_lexicon']:,} of {missing['reference_words']:,} ({missing['rate'] * 100:.2f}%)",
            f"{missing['clips_with_a_word_not_in_lexicon']:,} of {run['clips']:,}",
            f"{output['empty_hypotheses']:,} of {run['clips']:,}",
            f"{output['placeholders_removed']:,} in {output['clips_with_placeholders']:,} clips",
            f"{run['training']['acoustic_model_seconds'] / 60:.1f}", f"{run['processing_seconds'] / 60:.1f}"]))
    if not rows:
        return "_No complete run yet._"
    return _table(["Split", "Training clips", "Words in the lexicon", "Reference words not in the lexicon",
                   "Clips with such a word", "Empty hypotheses", "Unknown-word placeholders removed",
                   "Acoustic training (min)", "Decoding (min)"], [cells for _, cells in sorted(rows)])


def table_results(root: Path, group: str) -> str:
    """One row per complete run of `group` (a folder below results/), from its scores.json."""
    rows, partial = [], 0
    for path in sorted((Path(root) / RESULTS / group).glob("*/*/scores.json")):
        scores = json.loads(path.read_text(encoding="utf-8"))
        if not scores["complete"]:
            partial += 1
            continue
        everything, rest, checkpoint = scores["all_clips"], scores["without_english_number_words"], scores["checkpoint"]
        rows.append((SPLITS.index(scores["split"]), scores["system"], [
            scores["system"], _checkpoint(checkpoint),
            scores["split"], f"{everything['clips']:,}", _percent(everything["wer"]), _percent(everything["cer"]),
            f"{everything['wer_no_diacritics']['rate'] * 100:.2f}", f"{everything['rtf']:.3f}",
            f"{rest['clips']:,}", f"{rest['wer']['rate'] * 100:.2f}", f"{rest['cer']['rate'] * 100:.2f}"]))
    if not rows:
        return "_No complete run yet._"
    note = f"\n\n{partial} trial runs on a part of a split are not shown." if partial else ""
    return _table(["System", "Checkpoint and revision", "Split", "Clips", "WER % [95%]", "CER % [95%]",
                   "WER % without diacritics", "RTF", "Clips without English number words", "their WER %",
                   "their CER %"], [cells for _, _, cells in sorted(rows, key=lambda row: row[:2])]) + note


def block_finetune(root: Path) -> str:
    """Clips and settings of the fine-tuning run, from the clean manifest and the constants of `huasr.finetune`."""
    from . import finetune  # imported here: the other blocks need nothing but the summary files

    folder = Path(CLEAN_SUMMARY).parent
    if not (Path(root) / folder / "manifest.csv").is_file():
        return f"_Not built yet: `{folder.as_posix()}/manifest.csv` does not exist._"
    plan = finetune.plan(Path(root) / folder)
    rows = []
    for split, purpose in (("train", "training"), ("validation", "choosing the checkpoint")):
        part, left = plan["data"][split], plan["data"][split]["excluded"]
        rows.append([split, purpose, f"{part['clips_in_split']:,}",
                     "kept" if left["english_number_words"] is None else f"{left['english_number_words']:,}",
                     f"{left['longer_than_window']:,} of {part['longer_than_window_in_split']:,}",
                     f"{left['empty_target']:,}", f"{part['clips']:,}", f"{part['hours']:.2f}", part["speakers"]])
    values, decoding = plan["hyperparameters"], plan["selection"]["decoding"]
    clips = values["batch_size"] * values["gradient_accumulation_steps"]
    passes = values["max_steps"] * clips / max(1, plan["data"]["train"]["clips"])
    seconds = finetune.WINDOW_MS // 1000
    return (_table(["Split", "Used for", "Clips in the split", "Left out: English number words in the reference",
                    f"Left out: longer than {seconds} s, of all such clips", "Left out: empty target", "Clips used",
                    "Hours used", "Speakers"], rows)
            + f"\n\nA clip is counted under the first rule that leaves it out; the other training clips longer than "
              f"{seconds} s are already left out for their number words. Settings: {values['max_steps']:,} updates "
              f"of {clips} clips (about {passes:.1f} passes over the training clips used), learning rate "
              f"{values['learning_rate']:g} after a linear warm-up of {values['warmup_steps']} updates, then "
              f"{values['lr_scheduler']} decay to zero; optimizer `{values['optimizer']}`, weight decay "
              f"{values['weight_decay']:g}, gradient norm limited to {values['max_grad_norm']:g}; mixed precision "
              f"`{values['mixed_precision']}`; validation and a checkpoint every {values['evaluate_every_steps']} "
              f"updates; seed {plan['seed']}. Checkpoints are compared by validation "
              f"{plan['selection']['metric'].upper()}, decoded with beam size {decoding['num_beams']} and at most "
              f"{decoding['max_length']} tokens."
            + f"\n\nSource: `{folder.as_posix()}/manifest.csv` and the constants of `src/huasr/finetune.py` "
              f"(`python -m huasr.finetune plan`).")


def render(text: str, root: Path) -> str:
    """`text` with every generated block rewritten from the source files below `root`."""
    def block(match: re.Match) -> str:
        if match["name"] == "finetune-plan":
            return f"{match[1]}{block_finetune(root)}\n{match[4]}"
        if match["name"] == "results-gmm-hmm":
            return f"{match[1]}{table_gmm_hmm(root)}\n\nSource: `{RESULTS}/{GMM_HMM}/<split>/run.json`.\n{match[4]}"
        if match["name"] in RESULT_BLOCKS:
            group = RESULT_BLOCKS[match["name"]]
            body = table_results(root, group) + f"\n\nSource: `{RESULTS}/{group}/<system>/<split>/scores.json`."
            return f"{match[1]}{body}\n{match[4]}"
        if match["name"] not in BLOCKS:
            raise ValueError(f"unknown generated block {match['name']!r}")
        function, source = BLOCKS[match["name"]]
        path = Path(root) / source
        if path.is_file():
            body = function(json.loads(path.read_text(encoding="utf-8"))) + f"\n\nSource: `{source}`."
        else:
            body = f"_Not built yet: `{source}` does not exist._"
        return f"{match[1]}{body}\n{match[4]}"

    return BLOCK.sub(block, text)


def main(argv: list[str] | None = None, root: Path = Path(".")) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="change nothing; fail if a block is out of date")
    args = parser.parse_args(argv)
    stale = []
    for name in READMES:
        path = Path(root) / name
        text = path.read_text(encoding="utf-8")
        new = render(text, Path(root))
        if new != text:
            stale.append(name)
            if not args.check:
                path.write_text(new, encoding="utf-8", newline="\n")
    if args.check and stale:
        print(f"out of date: {', '.join(stale)}; run python -m huasr.report", file=sys.stderr)
        return 1
    print(f"rewritten: {', '.join(stale)}" if stale else "all generated blocks are up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
