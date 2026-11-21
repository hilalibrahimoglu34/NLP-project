# Results

What every run leaves behind, how it is scored, and the result tables. The
tables are written by `python -m huasr.report` from the `scores.json` files of
this folder; no number in them is typed by hand.

**Only the validation split has been scored.** The test split is kept for the
final evaluation, after every setting is frozen: `python -m huasr.evaluate`
refuses `--split test` unless `--final-test` is given, and the run manifest
records that the flag was used. The figures below are therefore development
figures, not the results of the project.

## Layout

```text
results/
  README.md                      this file
  baselines/<system>/<split>/    baselines: checkpoints used as they were published, and the GMM-HMM
                                 system (gmm-hmm-mfa), trained on our training split with a standard recipe
  own/<system>/<split>/          models trained in this project
    run.json                     written by huasr.evaluate (for gmm-hmm-mfa: by huasr.gmm_hmm)
    predictions.csv              written by huasr.evaluate (for gmm-hmm-mfa: by huasr.gmm_hmm)
    scores.json                  written by huasr.score from the two files above
```

One folder holds one run of one system on one split. A system belongs to the
group it declares (`group` in `src/huasr/systems.py`), so a baseline can never
appear among our own models.

| File | Content |
|---|---|
| `run.json` | system and group; checkpoint (repository id, revision, the checkpoint it was converted from, size and SHA-256 of the weight file); runtime and decoding settings; seed; split, number of clips and whether the whole split was run; SHA-256 of the raw and of the clean manifest; audio and processing time, loading time; processor, operating system; package versions; start and end time (UTC) |
| `predictions.csv` | one row per clip, in the order of the clean manifest: `clip_id`, `speaker_id`, `session_id`, `duration_ms`, `english_number_words` (the flag of the clean manifest), `reference`, `hypothesis` (exactly what the system wrote, before any normalisation), `processing_seconds` |
| `scores.json` | corpus-level WER, CER and diacritic-insensitive WER, each with its 95% interval and the numbers of substitutions, deletions, insertions and reference units; real-time factor; clips with an empty hypothesis; all of it for all clips and again without the clips flagged for English number words |

## How a run is made and scored

```bash
python -m huasr.evaluate --system faster-whisper-base     # validation split by default
python -m huasr.evaluate --system faster-whisper-small
python -m huasr.score --all                               # scores.json for every run
python -m huasr.report                                    # the tables below
```

- **Same input for every system.** The runner reads the clips of the split
  from the shards of `data/raw/voxpopuli/`, checks each one against the
  SHA-256 of the manifest, and hands the system the decoded samples, one clip
  at a time. The reference is the column `reference` of
  `data/clean/voxpopuli/manifest.csv`. A system never sees a reference.
- **Scoring is separate from running.** `huasr.score` reads only the
  prediction file and the run manifest, so the scoring can be changed without
  running a model again, and every system is scored by the same code.
- **Rates** are corpus-level, after the scoring policy of the main README has
  been applied to both sides.
- **Intervals** are 95% percentile intervals from a bootstrap that resamples
  speakers (`bootstrap_interval` in `src/huasr/metrics.py`; number of
  resamples and seed are recorded in `scores.json`). The validation split has
  few speakers, so the intervals are wide.
- **WER without diacritics** applies the same policy with the accents removed
  on both sides. The distance to the WER is the share of words that are wrong
  only by an accent.
- **RTF** (real-time factor) is the processing time divided by the audio
  duration, summed over the clips. Loading the model is not included; clips
  are processed one at a time. It depends on the machine, which `run.json`
  names, and on what else the machine is doing (see the note below the
  baseline table).
- **English number words.** The corpus wrote digits out in English in some
  references (see [`data/clean/README.md`](../data/clean/README.md)), which no
  system can get right. The last three columns repeat the figures without
  these clips. The flag only splits the report: every clip stays in the
  prediction file.
- A run on a part of a split (`--limit N`) is marked as incomplete and is not
  shown in the tables.

## Baselines

Whisper base and small are checkpoints used as published, with no training on
our data. `gmm-hmm-mfa` has no published checkpoint: it is a standard MFCC +
GMM-HMM recipe trained on our training split (see
[GMM-HMM baseline](#gmm-hmm-baseline) below).

<!-- generated: results-baselines -->
| System | Checkpoint and revision | Split | Clips | WER % [95%] | CER % [95%] | WER % without diacritics | RTF | Clips without English number words | their WER % | their CER % |
|---|---|---|---|---|---|---|---|---|---|---|
| faster-whisper-base | `Systran/faster-whisper-base` at `ebe41f70d5b6` | validation | 826 | 54.41 [49.95, 59.16] | 16.52 [13.59, 20.16] | 52.99 | 0.196 | 771 | 52.66 | 15.05 |
| faster-whisper-small | `Systran/faster-whisper-small` at `536b0662742c` | validation | 826 | 31.55 [28.01, 36.13] | 10.12 [7.64, 13.82] | 30.64 | 0.259 | 771 | 29.40 | 8.48 |

Source: `results/baselines/<system>/<split>/scores.json`.
<!-- end generated -->

Whisper base and small are run with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper), which loads
CTranslate2 conversions of the OpenAI checkpoints. The table gives the
repository and the first characters of the revision that was loaded from the
local cache; `run.json` has the full revision and the SHA-256 of the weight
file.

| System | Loaded repository | Converted from (stated on its model card) |
|---|---|---|
| `faster-whisper-base` | `Systran/faster-whisper-base` | `openai/whisper-base` |
| `faster-whisper-small` | `Systran/faster-whisper-small` | `openai/whisper-small` |

Settings, declared in `src/huasr/systems.py` and copied into every
`run.json`: CPU, weights quantised to 8-bit integers (`int8`), a fixed number
of threads, language `hu`, task `transcribe`, beam size 5, temperature 0
without fallback to higher temperatures, no conditioning on previous text,
everything else as faster-whisper sets it. What this means for reading the
table:

- The model cards name the OpenAI checkpoint but not its revision, and the
  weights are quantised here. The figures are those of this conversion with
  these settings; we have not compared them with the original implementation.
- The number of threads is fixed because the text depends on it: on a sample
  of clips the output was identical when the same number was used twice, and
  differed slightly between numbers.
- With the temperature fixed at 0 the decoder cannot retry a clip, so a
  repetition loop stays in the hypothesis and counts as insertions. The
  prediction files show such clips for both models.
- **The RTF column is a rough figure only.** The runs were made on a laptop
  whose load was not controlled: the operating system moved the threads
  between fast and slow processor cores during the runs, and a short trial of
  the same system on the same clips ran much faster than the full run. The
  two RTF values should not be compared with each other; speed has to be
  measured again under a controlled load before it is reported as a result.
- Whisper's training data is not published (audio with transcripts collected
  from the web; Radford et al., 2023). Whether it contains recordings of the
  European Parliament cannot be checked by us, so "zero-shot" means only that
  we did not train the model.

### GMM-HMM baseline

`gmm-hmm-mfa` is trained and run by `src/huasr/gmm_hmm.py` with the Montreal
Forced Aligner (Kaldi): MFCC features without pitch; monophone, triphone, LDA
and speaker-adapted training; a graphemic lexicon of the training words; a
trigram language model of the training transcripts; `mfa transcribe` with its
default settings. The main README describes the system and its limits. Its
`run.json` has the same fields as the other runs and, in addition, the
versions and builds of MFA and Kaldi, the training recipe, the training time,
the words of the references that are not in the lexicon (`out_of_vocabulary`)
and what was done to the output of MFA (`output`).

<!-- generated: results-gmm-hmm -->
_No complete run yet._

Source: `results/baselines/gmm-hmm-mfa/<split>/run.json`.
<!-- end generated -->

How to read this table and the row of `gmm-hmm-mfa` in the table above:

- **Reference words not in the lexicon** can never be written by this system,
  so their share is a lower bound for its WER. Whisper has no such bound.
- **Empty hypotheses** are clips for which nothing is left after the
  placeholders of MFA (such as `<unk>`) have been removed; every word of their
  reference counts as a deletion.
- **RTF** is not comparable with the Whisper rows. It is the wall time of
  `mfa transcribe` for the whole split (features, decoding graph, two decoding
  passes with speaker adaptation, rescoring) with several parallel jobs,
  divided by the audio duration. The number of jobs is in `run.json`.
- The system was given the speaker id of every clip (it adapts to the
  speaker); the Whisper baselines were not.
- Decoding used the default settings of MFA. Nothing was tuned on the
  validation split.
- The run was made on Windows 11 with the win-64 packages of conda-forge
  (versions and builds in `run.json`).

## Our own models

<!-- generated: results-own -->
_No complete run yet._

Source: `results/own/<system>/<split>/scores.json`.
<!-- end generated -->

### Whisper small, fine-tuned

Not run yet (21 November 2026). `hf-whisper-small-finetuned` is
`openai/whisper-small` after `python -m huasr.finetune train`. The main README
has the clips it is trained on, the settings and the commands
([Fine-tuning Whisper small](../README.md#fine-tuning-whisper-small-our-own-model)).
What its run leaves in this folder, and how it is to be read:

- **Before and after.** The checkpoint it starts from is run as the baseline
  `hf-whisper-small` (in the table of the baselines once it has run) with the
  same code and the same decoding: Hugging Face implementation, language `hu`,
  task `transcribe`, no timestamps, beam size 5, temperature 0. The distance
  between that row and the row of this model is the effect of the
  fine-tuning. `faster-whisper-small` is the same published checkpoint in
  another implementation, with quantised weights and on the CPU.
- **`own/hf-whisper-small-finetuned/training.json`**, next to the split
  folders: the training record. Clips used and left out, settings, seed, loss
  and validation scores of every evaluated step, the kept step, SHA-256 of
  the kept weights, GPU and package versions. The validation scores in it are
  decoded greedily in batches, without the clips longer than 30 s, and only
  choose the checkpoint; the figures of the table come from the run folder.
- **`comparison_with_hf-whisper-small.json`** in the run folder, written by
  `python -m huasr.compare`: the difference between the two systems on the
  same clips for WER, CER and the WER without diacritics, each with a 95%
  interval and a one-sided p-value from the paired bootstrap over speakers.
- In the table, the column "Checkpoint and revision" names for this model the
  checkpoint it started from, its revision and the step that was kept;
  `run.json` has the SHA-256 of the weights that were evaluated.
- **The validation split chose the checkpoint**, so its figures for this
  model are development figures, a little better than what an unseen set
  would give. The test split is the unseen set.
- **RTF.** The two Hugging Face systems run on the GPU named in their
  `run.json`; their RTF cannot be compared with the CPU runs.

## Reference

Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C. and Sutskever, I.
(2023). Robust Speech Recognition via Large-Scale Weak Supervision. *ICML
2023*. <https://proceedings.mlr.press/v202/radford23a.html>
