# Hungarian Automatic Speech Recognition on VoxPopuli

Transcribe Hungarian speech into Hungarian text, and measure how far
fine-tuning a small multilingual model on Hungarian European Parliament
speech (VoxPopuli) closes the gap to much larger models used zero-shot, on
speakers the model has never heard and on read speech from two other corpora.

COE025 Natural Language Processing, Project 3 (automatic speech recognition).

> **Status (21 November 2026).** The data stage is done for VoxPopuli: a
> subset whose three splits share no speaker and no recording session is
> built in `data/raw/voxpopuli/`, and its repaired references are in
> `data/clean/voxpopuli/`. The evaluation pipeline (runner, scorer, result
> table) is implemented and tested. Three baselines have been run on the
> **validation** split: Whisper base and small zero-shot, and an MFCC +
> GMM-HMM system trained on our training split. Their figures are in
> [`results/README.md`](results/README.md). The test split has not been
> touched, FLEURS has been built and committed, and the model of the research question
> has been fine-tuned and scored on the validation split. The section
> [Status](#status) lists what exists and what is only planned. Every table
> of numbers in the READMEs is written by `python -m huasr.report` from files
> that the builders and the scorer generate; no such number is typed by hand.

## Task

Input: a short recording of Hungarian speech. Output: what was said, as
Hungarian text. Speech and reference transcript are in the same language;
there is no translation step.

Research question (planned): how much of the gap between a small multilingual
checkpoint and much larger zero-shot models is closed by fine-tuning the small
one on Hungarian parliament speech, and does the gain carry over to speakers
it has not heard and to read speech recorded elsewhere?

Hungarian is agglutinative: one word carries a stem and several suffixes
(*házaimban*, "in my houses", is one word), and compounds are written
together. A single wrong suffix makes the whole word wrong, so the choice of
error unit matters more than in English. That is why this project reports
word, character and subword error rates side by side
([Metrics](#metrics-and-why-they-fit-hungarian)).

## Language eligibility

- Language: **Hungarian (`hu`)**, registered for Group 8 in the course's
  dataset sheet.
- It is not English and not the mother tongue of any group member.
- No group member speaks or reads Hungarian.

Consequences for the method: nobody can judge a transcript by eye, so
references are never corrected by hand. A clip that looks wrong is removed and
logged, not relabelled, and the output of a model is never used as a
reference.

## Data

| Corpus | Role | Licence | Where it lives |
|---|---|---|---|
| [VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli), Hungarian transcribed set | training, development and main test data | CC0-1.0 | subset in `data/raw/voxpopuli/`, repaired references in `data/clean/voxpopuli/` |
| [FLEURS](https://huggingface.co/datasets/google/fleurs), `hu_hu`, test split | extra test set: read speech | CC BY 4.0 (attribution below) | whole split in `data/raw/fleurs/` (built and committed) |
| [Common Voice Scripted Speech 27.0](https://mozilladatacollective.com/datasets/cmu5wq4u200dknq072xswzofb), Hungarian, test split | extra test set: read speech, many volunteers | CC0-1.0, re-hosting forbidden | **never in this repository**; fetched into `external/` (ignored by git) |

Every source file is pinned: `src/huasr/download.py` holds the dataset
revision, size and SHA-256 of each file and checks them after the download.
`python -m huasr.download voxpopuli --dry-run` (or `fleurs --dry-run`) prints
the list.

### VoxPopuli (training, development, main test)

Speeches from European Parliament plenary sessions, cut into utterances and
aligned with the official transcripts (Wang et al., 2021). The Hungarian part
comes as parquet files with these columns, read from the files' own metadata:
`audio_id`, `audio` (the bytes of one WAV file and its path), `raw_text`,
`normalized_text`, `gender`, `speaker_id`, plus `language`,
`is_gold_transcript` and `accent`, which this project does not use.

**Speakers and recording sessions.** Every clip has a speaker id (a few have
none) and comes from a recording session. VoxPopuli cuts all clips of one
`session_id` out of one recording file and names a clip
`<session_id>-<language>_<date>-<time of the speech>_<number>`
(`voxpopuli/get_asr_data.py` in the corpus's own code). The clip id up to the
first underscore, for example `20090202-0900-PLENARY-9-hu`, is therefore used
as the session. In the Hungarian data such a session is one numbered part of
a plenary sitting day: the parts of a day follow each other in time, and a
speech with all its clips lies inside one part.

**The official split is not disjoint.** The same speakers and the same
sessions occur in the official train, validation and test parts, and some
clips have no speaker id:

<!-- generated: voxpopuli-official-split -->
| Official split | Clips | Hours | Speakers | Sessions | Clips without speaker id |
|---|---|---|---|---|---|
| train | 18,120 | 54.09 | 47 | 1,537 | 601 |
| validation | 1,076 | 3.25 | 26 | 141 | 16 |
| test | 1,110 | 3.11 | 46 | 154 | 23 |

Speakers shared by two official splits: train and validation 21, train and test 27, validation and test 18. Sessions shared by two official splits: train and validation 90, train and test 93, validation and test 14.

Source: `data/raw/voxpopuli/summary.json`.
<!-- end generated -->

A model that is tested on a speaker or on a recording it was trained on looks
better than it is. `src/huasr/subset.py` therefore makes its own split, with a
fixed seed, under three rules:

1. Every speaker is in exactly one split.
2. Every recording session is in exactly one split.
3. A clip is only used in the official split it comes from. No official
   training clip becomes a test clip, because public models may have been
   trained on the official training part; and no official test clip is used
   for training.

A clip is kept if its official split, the split of its speaker and the split
of its session are the same. How speakers and sessions get their split:

- Clips without a speaker id are left out, since nobody can check them.
- Every speaker and every session goes to the split in which it keeps the
  most audio. Test and validation audio is scarce and training audio is
  plentiful, so a second of test audio and a second of validation audio count
  several times as much as a second of training audio (two weights). A
  speaker counts only the clips whose session is in the same split and a
  session only the clips whose speaker is, so sessions and speakers are
  assigned in turn until nothing changes.
- The two weights are searched on a fixed grid. The pair is taken that
  reaches, in this order, the minimum of test audio, the validation target
  and the training target, and then gives the most test audio.
- The test split keeps every clip that is left. Validation and training clips
  are drawn in a seeded order until the target of the split is reached, so
  every speaker is represented in proportion to their audio.

The build stops unless no speaker and no session is in two splits of the
result. What is in this repository, counted again from the stored manifest:

<!-- generated: voxpopuli-split-report -->
| Split | Speakers | Sessions | Utterances | Hours | References that occur word for word in the training split |
|---|---|---|---|---|---|
| train | 20 | 626 | 4,597 | 14.00 | - |
| validation | 15 | 112 | 826 | 2.50 | 6 of 826 |
| test | 34 | 104 | 897 | 2.41 | 7 of 897 |

Speakers in two splits: train and validation 0, train and test 0, validation and test 0. Sessions in two splits: train and validation 0, train and test 0, validation and test 0.

Source: `data/clean/voxpopuli/summary.json`.
<!-- end generated -->

<!-- generated: voxpopuli-subset -->
| Split | Clips | Hours | Speakers | Sessions | Shards | Size (MB) |
|---|---|---|---|---|---|---|
| train | 4,597 | 14.00 | 20 | 626 | 11 | 987.8 |
| validation | 826 | 2.50 | 15 | 112 | 2 | 165.0 |
| test | 897 | 2.41 | 34 | 104 | 2 | 166.2 |
| total | 6,320 | 18.91 | 69 | 842 | 15 | 1319.0 |

Speakers shared by two splits of the subset: train and validation 0, train and test 0, validation and test 0. Sessions shared by two splits of the subset: train and validation 0, train and test 0, validation and test 0. Settings: training target 14 h, validation target 2.5 h, at least 2 h of test audio; weights taken: validation 40, test 36; seed 0.

Source: `data/raw/voxpopuli/summary.json`.
<!-- end generated -->

The targets are the defaults `DEFAULT_HOURS` and `DEFAULT_SEED` in
`src/huasr/subset.py`; the training and validation targets keep the
repository at a size GitHub accepts. [`data/raw/README.md`](data/raw/README.md)
shows the official split in more detail, what each rule removed, and what the
other weight pairs would have kept; `data/raw/voxpopuli/selection.csv` names
every source clip with its decision.

Two limits of this split. Different parts of the same sitting day can be in
different splits; the unit that is kept apart is the recording session of the
corpus, and `summary.json` counts the shared days
(`sitting_days_in_two_splits`). And because of rule 3 our test set is a part
of the official test set, but the scores are not comparable with published
VoxPopuli results, which use all of it.

**The reference text had to be repaired.** VoxPopuli gives two transcripts per
clip: `raw_text` as published, and `normalized_text`, which the corpus derived
from it and which is the usual reference. In the Hungarian part that
normalisation deleted every letter í (`raw_text` *javítani*,
`normalized_text` *javtani*). `src/huasr/repair.py` restores the letter word
by word, only where the aligned word of `raw_text` proves it; words that
cannot be aligned are left unchanged and counted, and no text is invented.
The result is the column `reference` of `data/clean/voxpopuli/manifest.csv`;
both original fields stay next to it, and every changed word is logged.

<!-- generated: voxpopuli-repair -->
| Reference repair | Count |
|---|---|
| Clips with at least one restored word | 2,666 |
| Restored words | 3,962 |
| Clips with nothing to restore | 3,507 |
| Clips without `raw_text` (cannot be checked) | 147 |
| Clips whose `normalized_text` is not in normalised form (left alone) | 0 |
| Words that could not be aligned with `raw_text` (left unchanged) | 787 in 325 clips |

| Letter missing from `normalized_text` | in `raw_text` | in `normalized_text` | in `reference` |
|---|---|---|---|
| `í` | 3,965 | 0 | 3,964 |
| `š` | 13 | 0 | 0 |
| `č` | 7 | 0 | 0 |
| `ş` | 4 | 0 | 0 |
| `ß` | 3 | 0 | 0 |
| `ä` | 2 | 0 | 0 |
| `º` | 1 | 0 | 0 |
| `è` | 1 | 0 | 0 |
| `ï` | 1 | 0 | 0 |
| `ć` | 1 | 0 | 0 |
| `ł` | 1 | 0 | 0 |
| `ň` | 1 | 0 | 0 |
| `ž` | 1 | 0 | 0 |
| `ș` | 1 | 0 | 0 |

Source: `data/clean/voxpopuli/summary.json`.
<!-- end generated -->

The other letters in the second table are letters of other languages in
names; they were lost in the same way and are reported, not restored. Clips
without `raw_text` cannot be checked.

**Numbers are written out in English.** The corpus normalisation also turned
digits into English words inside the Hungarian text: `raw_text` *2008*,
`normalized_text` *two thousand and eight*. Nobody says that in the recording,
so every such number is an error for every system and a wrong target for
training. These references are flagged (`english_number_words` in the clean
manifest) and counted; nothing is removed yet:

<!-- generated: voxpopuli-number-words -->
| English number words | train | validation | test |
|---|---|---|---|
| References with English number words | 286 | 55 | 35 |
| of these: `raw_text` has the number in digits | 223 | 46 | 28 |
| of these: the clip has no `raw_text` | 62 | 9 | 6 |
| of these: `raw_text` has no digits (English words in the speech itself) | 1 | 0 | 1 |

Source: `data/clean/voxpopuli/summary.json`.
<!-- end generated -->

Details of both problems: [`data/clean/README.md`](data/clean/README.md).

### FLEURS (extra test set)

Sentences read aloud by native speakers (Conneau et al., 2022). The whole
official Hungarian test split is stored. FLEURS has no speaker ids; several
recordings share a sentence, so the sentence id is kept and used as the
cluster when scores are resampled.

Attribution: FLEURS by Conneau et al. (Google), obtained from
<https://huggingface.co/datasets/google/fleurs>, licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes made by
us: the WAV files of the Hungarian test split are stored as 16-bit FLAC files
in tar archives (see below), with the transcripts in a CSV file. No text was
changed.

### Common Voice (extra test set, not stored here)

Volunteers read sentences aloud; other volunteers vote on each recording. The
dataset page states the licence as CC0-1.0 and adds, under "Forbidden Usage"
(read on 16 November 2026): "It is forbidden to attempt to determine the
identity of speakers in the Common Voice datasets. It is forbidden to re-host
or re-share this dataset." This project therefore uses Common Voice only as a
test set that everyone fetches for themselves:

- `python -m huasr.download commonvoice` writes to `external/common_voice/`,
  and `python -m huasr.subset commonvoice` writes its list of test clips next
  to the extracted release. Both refuse any target inside the repository
  other than `external/` (`require_untracked` in `src/huasr/download.py`), and
  tests check this.
- `.gitignore` excludes `external/`, extracted release folders, the archive,
  the clip list and every MP3 file.
- Prediction files for Common Voice will hold clip ids and system output
  only; its sentences and audio stay local (planned).

A download needs an account at the Mozilla Data Collective and the acceptance
of the dataset's terms on its web page; a script cannot do that. The manual
steps are listed under [Reproduce](#reproduce). We do not try to identify any
speaker in any corpus; speakers appear only as the ids the corpora publish.

### How audio is stored

VoxPopuli and FLEURS ship uncompressed WAV files, far too large for a GitHub
repository. `src/huasr/audio.py` reads the format of every clip from its own
header and stores it as a mono 16-bit FLAC file with the same sample rate:

- 16-bit PCM input is stored losslessly.
- 32-bit float input is quantised to 16 bits (sample × 32768, rounded,
  limited to the 16-bit range). This is the only change made to the audio; the
  number of limited samples is logged per clip.
- Any other encoding is refused, not converted. Input that is already
  compressed would be kept byte for byte. Nothing is resampled.

The manifest records, per clip, the encoding, size and SHA-256 of the source
file and the size and SHA-256 of the stored file, so a stored clip can be
traced back to the exact source bytes. The clips are packed into tar shards
below GitHub's file-size limit. Details and file formats:
[`data/raw/README.md`](data/raw/README.md).

### Cleaning

Done so far (`src/huasr/clean.py`, outputs in `data/clean/voxpopuli/`): the
reference repair above with a log of every changed word; a log of every
source clip that is not in the data with the rule that left it out; and three
flags per clip: English number words in the reference, a validation or test
reference that also occurs word for word in the training split, and the
number of characters per second, which is needed for a rule against
transcripts that do not belong to their audio.

Planned: what to do with the flagged references, and removal rules for
transcripts that do not match their audio, for clips that are too long for the
models and for transcripts with foreign letters. No clip is removed by a cleaning rule yet; the
thresholds will be chosen on the training and validation splits. See
[`data/clean/README.md`](data/clean/README.md).

## Systems

Every system sits behind one small contract (`src/huasr/systems.py`): a name,
a group (`baselines` for checkpoints used as published, `own` for models
trained in this project), a description for the run manifest, and
`transcribe(samples, sample_rate) -> text`. The runner gives every system the
same clips in the same form and knows nothing else about it.

Run so far, on the validation split only:

| System | Checkpoint that is loaded | Converted from | Role |
|---|---|---|---|
| `faster-whisper-base` | `Systran/faster-whisper-base` | `openai/whisper-base` | zero-shot baseline |
| `faster-whisper-small` | `Systran/faster-whisper-small` | `openai/whisper-small` | zero-shot baseline; the checkpoint planned for fine-tuning |

Both are run with [faster-whisper](https://github.com/SYSTRAN/faster-whisper),
which loads CTranslate2 conversions of the OpenAI checkpoints; the model
cards of the two repositories name the OpenAI checkpoint each was converted
from. They run on the CPU with weights quantised to 8-bit integers and a
fixed number of threads, with language `hu`, task `transcribe`, beam size 5,
temperature 0 and no conditioning on previous text. The revision that was
loaded, the SHA-256 of its weight file and all settings are in the `run.json`
of each run; the figures and what limits them are in
[`results/README.md`](results/README.md).

### GMM-HMM baseline (`gmm-hmm-mfa`)

The classical recogniser: hidden Markov models whose states are Gaussian
mixtures over MFCC features, a pronunciation lexicon and an n-gram language
model. There is no published checkpoint to take, so this baseline is trained
from scratch on our training split with a standard recipe. It is listed with
the baselines, not as our model: nothing in it is our own design.
`src/huasr/gmm_hmm.py` runs it with the
[Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/)
(MFA), which drives Kaldi. MFA works on a folder of clips, not on one clip,
so this system does not sit behind the contract above; see
[Evaluation protocol](#evaluation-protocol).

- **Features:** MFCC without pitch; everything else as MFA sets it.
- **Acoustic model:** the first four blocks of MFA's default recipe with
  their default settings (`TRAIN_CONFIG`): monophone, triphone, LDA, and
  speaker-adapted training (SAT). The later blocks of that recipe are left
  out.
- **Lexicon:** graphemic. Every word of the training transcripts is spelled
  as its letters, one letter per unit (*szép* is `s z é p`); digraphs are not
  merged. No pronunciation dictionary and no G2P model is used.
- **Language model:** trigram with Kneser-Ney smoothing, estimated by
  `mfa train_lm` from the training transcripts only. Its vocabulary is the
  lexicon.
- **Text:** the `reference` column after the scoring policy below, for the
  acoustic model and for the language model.
- **Decoding:** `mfa transcribe` with the search settings MFA uses by default
  (`DECODE_CONFIG`); nothing was tuned. The clips of the split are given to
  MFA grouped by speaker id, because the model adapts to each speaker; no
  reference text of the split is written to the folder MFA reads.

What limits it, and what has to be kept in mind when it is compared with the
other systems:

- **Closed vocabulary.** It can only write words that occur in the training
  transcripts. Every other word of a reference is an error before any audio
  is heard, which weighs heavily in a language that builds a new word form
  with every suffix. `collect` counts these words; the share is in
  [`results/README.md`](results/README.md) next to the error rates.
- **Unknown-word placeholders.** MFA has an unknown-word model and can write
  a placeholder such as `<unk>` for it. Any placeholder is removed from the
  hypothesis and counted in `run.json`.
- **Few training speakers.** The acoustic model has heard only the speakers
  of the training split, and every speaker it is evaluated on is new to it.
- **Letters as sound units.** Hungarian spelling is close to the sounds, but
  a letter is not always one sound (*sz*, *cs*, *gy*, *dzs*, long consonants);
  the context-dependent models have to absorb that.
- **Speaker ids are used.** The Whisper baselines get one clip at a time and
  nothing else. This system also gets the speaker id of every clip.
- **Speed is not comparable.** MFA decodes the whole split with several
  parallel jobs; the processing time in `run.json` is the wall time of
  `mfa transcribe` shared out over the clips, not a per-clip time.
- **Platform.** The run in this repository was made on Windows 11 with the
  win-64 packages of conda-forge, without WSL. The steps are plain Python
  entry points and the environment file pins the same versions for every
  platform, but they have not been run on Linux or Colab.

Planned; the list will be extended with every baseline the course lists for
this project:

| System | Checkpoint | Role |
|---|---|---|
| Whisper small, fine-tuned | `openai/whisper-small`, trained on our training split | our model |
| Whisper large-v3, zero-shot | `openai/whisper-large-v3` | strong reference |
| Parakeet TDT v3, zero-shot | `nvidia/parakeet-tdt-0.6b-v3` | transducer family |
| Canary v2, zero-shot | `nvidia/canary-1b-v2` | encoder-decoder family |
| MMS, zero-shot | `facebook/mms-1b-all` (CC BY-NC 4.0) | CTC family |

Decoding settings are chosen on the validation split only. For every
zero-shot model the report will say whether its published training data
includes VoxPopuli, FLEURS or Common Voice.

## Fine-tuning Whisper small (our own model)

> **Status (21 November 2026).** The code of this section is written and
> tested without a model. It has **not been run on a GPU yet**, so it has no
> results, and its calls into `transformers` have only been exercised through
> a stand-in in the tests. The lists under [Systems](#systems),
> [Repository layout](#repository-layout) and [Status](#status) do not name
> its files yet; they are listed at the end of this section.

Fine-tuning is this project's own contribution. It is reported under "Our own
models" in [`results/README.md`](results/README.md), apart from the baselines,
on the same validation clips, references, normalisation and scoring.

**Two systems, one piece of code.** `src/huasr/whisper_hf.py` runs Whisper
small through the Hugging Face implementation (`transformers`), behind the
system contract:

| System | Group | What is loaded |
|---|---|---|
| `hf-whisper-small` | `baselines` | `openai/whisper-small` at revision `973afd24965f72e36ca33b3055d56a652f456b4d`, as published (zero-shot) |
| `hf-whisper-small-finetuned` | `own` | that checkpoint after `python -m huasr.finetune train`, read from a local folder |

The zero-shot system is measured first, then the checkpoint is fine-tuned,
then the kept checkpoint is run on the same clips. Both systems decode in the
same way: language Hungarian, task transcribe, no timestamps, beam size 5 and
temperature 0 as for the faster-whisper baselines, 32-bit floats, on the GPU
if there is one (the run manifest names the device). Whisper reads at most 30
seconds at once and cannot move on through a longer recording without
timestamps, so a longer clip is cut into the fewest equal parts that fit and
the texts are joined. A word at a cut can come out wrong, for both systems at
the same place. `python -m huasr.compare` compares the two prediction files
with the paired bootstrap of [Comparing two systems](#comparing-two-systems)
and refuses runs that do not hold the same clips and references.

**Fixed before the first run** (`src/huasr/finetune.py`). The validation
split decides one thing only: which checkpoint is kept.

- *Target text:* the column `reference` after the scoring policy, so the
  model learns to write what the scorer counts.
- *Training clips with English number words are left out.* Their text is a
  wrong target (see [Data](#data)). The validation clips keep them, as every
  system is scored on them, and the scorer gives the figures with and without.
- *Clips longer than Whisper's window are left out* of the training and of
  the checkpoint selection: the end of the audio would be cut off while the
  text went on. The final evaluation covers every validation clip.
- *Nothing else is changed:* no augmentation, no frozen layers, the
  tokenizer and the feature extractor of the checkpoint as published. The
  settings were not tried in several variants.

<!-- generated: finetune-plan -->
| Split | Used for | Clips in the split | Left out: English number words in the reference | Left out: longer than 30 s, of all such clips | Left out: empty target | Clips used | Hours used | Speakers |
|---|---|---|---|---|---|---|---|---|
| train | training | 4,597 | 286 | 18 of 29 | 0 | 4,293 | 12.51 | 20 |
| validation | choosing the checkpoint | 826 | kept | 7 of 7 | 0 | 819 | 2.42 | 15 |

A clip is counted under the first rule that leaves it out; the other training clips longer than 30 s are already left out for their number words. Settings: 2,000 updates of 16 clips (about 7.5 passes over the training clips used), learning rate 1e-05 after a linear warm-up of 200 updates, then linear decay to zero; optimizer `adamw_torch`, weight decay 0, gradient norm limited to 1; mixed precision `fp16`; validation and a checkpoint every 250 updates; seed 0. Checkpoints are compared by validation WER, decoded with beam size 1 and at most 225 tokens.

Source: `data/clean/voxpopuli/manifest.csv` and the constants of `src/huasr/finetune.py` (`python -m huasr.finetune plan`).
<!-- end generated -->

The training record `results/own/hf-whisper-small-finetuned/training.json`
states these counts for the run that was really made, together with the loss
and the validation scores of every evaluated step, the kept step, the GPU and
the package versions. The run is seeded, but GPU arithmetic is not repeatable
bit for bit, so a second run can differ slightly. The fine-tuning code
refuses the test split. Weights are never stored in this repository: they go
to `scratch/`, which is not tracked, and the run manifest of the evaluation
records their SHA-256 and that of the training record.

**How to run it.** On Google Colab with a GPU of more than 16 GB of memory
(the committed run used an NVIDIA L4 with 23 GB), open
[`notebooks/finetune_whisper_small.ipynb`](notebooks/finetune_whisper_small.ipynb)
and run all cells. It clones this repository, so nothing is uploaded by hand,
it never uses the test split, and it ends with an archive of the small result
files. The cells call the commands below, which work on any machine with a
GPU:

```bash
pip install -r requirements.txt -r requirements-train.txt
pip install --no-deps -e .
python -m huasr.subset unpack --raw data/raw/voxpopuli    # audio files -> scratch/audio/voxpopuli/
python -m huasr.whisper_hf fetch                          # the pinned checkpoint -> Hugging Face cache
python -m huasr.finetune plan                             # clips and settings; needs no model
python -m huasr.finetune train --smoke                    # a few updates on a few clips: checks the whole path
python -m huasr.evaluate --system hf-whisper-small        # before: zero-shot, validation split
python -m huasr.finetune train                            # the real run; --resume continues an interrupted one
python -m huasr.evaluate --system hf-whisper-small-finetuned    # after: the kept checkpoint, same clips
python -m huasr.score results/baselines/hf-whisper-small/validation results/own/hf-whisper-small-finetuned/validation
python -m huasr.compare results/own/hf-whisper-small-finetuned/validation results/baselines/hf-whisper-small/validation
python -m huasr.report
```

`requirements-train.txt` pins the packages that only this part needs (torch,
transformers, accelerate, tokenizers, safetensors, huggingface_hub) at the
versions of the Colab GPU runtime of 20 November 2026, so that Colab keeps
its GPU build of torch; `requirements.txt` is unchanged. `fetch` is the only
command that downloads a model, and it checks the weight file against a
pinned SHA-256. If the clips of an update do not fit in the GPU memory,
`train --batch-size 8` splits every update into two batches and the record
says so. The committed run reached a peak of 17.1 GiB of GPU memory with 16
clips per update, which does not fit a 15 GB T4; on such a card the smaller
batch is needed, and that is a different setting from the one reported here.
On the L4 the fine-tuning took 63.5 minutes including the nine validation
passes; the smoke run prints a projection measured on the machine.

Files of this part:

```text
requirements-train.txt     pinned packages for fine-tuning and for the two systems above
notebooks/
  finetune_whisper_small.ipynb   Colab: zero-shot baseline, fine-tuning, evaluation, comparison, result files
src/huasr/
  whisper_hf.py            Whisper small through transformers, as published and fine-tuned: pinned
                           checkpoint, fixed decoding, clips longer than the window; `fetch`
  finetune.py              choice of the clips, target text, dataset, collator, selection metric,
                           training with Seq2SeqTrainer, best checkpoint, training record
  compare.py               two runs on the same clips: paired bootstrap over speakers -> comparison file
  report.py                also writes the table of this section (block `finetune-plan`)
tests/
  test_whisper_hf.py       registration, pinned checkpoint, decoding settings, long clips; stand-in model
  test_finetune.py         target text, clips left out, settings, dataset, collator, metric, best checkpoint
  test_finetune_run.py     the training command from start to end with a stand-in for transformers
  fake_transformers.py     that stand-in
  test_compare.py          the comparison on hand-made prediction files
  test_notebooks.py        the notebook is well-formed, without output, and never runs the test split
results/own/hf-whisper-small-finetuned/
  training.json            the training record (written by the real run)
  validation/              run.json, predictions.csv, scores.json, comparison_with_hf-whisper-small.json
```

## Metrics and why they fit Hungarian

All error rates come from one edit-distance alignment between reference and
system output (`src/huasr/metrics.py`):

error rate = (substitutions + deletions + insertions) / reference units

- **WER** (units: words) is the standard ASR metric and is what other work
  reports, so it is needed for comparison. Insertions are in the numerator
  only, so WER can exceed 100%. For Hungarian it is a harsh unit: a word with
  one wrong suffix, a compound written as two words, or one missing accent
  counts as a full error, exactly like a word that is completely wrong.
- **CER** (units: characters, including the space between words) gives partial
  credit. A wrong suffix costs one or two characters and a wrong word boundary
  costs one. It shows how much of the text is wrong, which WER hides in a
  language with long words. Its weakness is the reverse: a one-character error
  can change the meaning (*kor* "age", *kór* "illness", *kör* "circle"), and
  CER makes such an error look small.
- **Subword error rate** (units: tokens of one declared tokenizer; planned)
  lies between the two: a wrong suffix changes one token, not the whole word.
  Its value depends on the tokenizer, so the tokenizer is part of the scoring
  policy and the rate cannot be compared across tokenizers. `error_rate` in
  `metrics.py` already accepts any tokenizer; the tokenizer itself is not
  chosen yet.

Reading the three together separates "the model got the stem but not the
ending" (WER high, CER low) from "the model produced something else" (both
high). No single one of them is enough for an agglutinative language.

**Corpus-level rates.** Edit counts are summed over all clips and divided by
the summed reference units. The mean of per-clip percentages is not used: it
would give a one-word clip the same weight as a twenty-word clip, and a single
short clip with insertions can have a rate far above 100%.

**Diacritic-insensitive WER** is reported next to the WER
(`strip_diacritics=True`). The difference between the two shows how many
errors are accent errors only.

**Speed.** Real-time factor = processing time / audio duration, summed over
the clips of a split. Clips are processed one at a time and loading the model
is not counted; the processor is named in the run manifest.

`src/huasr/score.py` computes all of this from the prediction file of a run:
WER, CER and diacritic-insensitive WER with their substitution, deletion and
insertion counts and 95% intervals, and the real-time factor, once for all
clips and once without the clips whose reference contains English number
words.

## Scoring policy

Declared before any system is scored, and applied identically to references
and system outputs by `normalize` in `src/huasr/text.py`:

1. Unicode NFC.
2. Lower case.
3. Every punctuation character and every symbol (Unicode categories P and S)
   is replaced by a space, so a hyphenated word counts as two words.
4. Control and format characters (soft hyphen, zero-width space) are deleted.
5. Whitespace is collapsed to single spaces.
6. Diacritics are kept: *a* and *á*, *o*, *ö* and *ő* are different letters.
7. Digits are not rewritten as words.

Reference text per corpus (manifest column): VoxPopuli `reference` (the
repaired `normalized_text`, from `data/clean/voxpopuli/manifest.csv`), FLEURS
`normalized_text`, Common Voice `raw_text` (the sentence that was read). All
of them pass through the same `normalize`.

A clip whose reference is empty after normalisation has no reference units,
so it cannot enter an error rate. The scorer counts such clips
(`clips_with_empty_reference` in `scores.json`) and leaves them out of the
rates; they stay in the prediction file.

## Comparing two systems

Two systems are compared on the same test clips with a paired bootstrap that
resamples speakers (`paired_bootstrap` in `src/huasr/metrics.py`). Clips of
one speaker share a voice and a microphone, so they are not independent;
resampling clips would make the intervals too narrow. Each resample draws
speakers with replacement, takes all of their clips for both systems and
recomputes the corpus-level difference in error rate. The number of test
speakers is small in VoxPopuli, which widens the intervals; it is reported
with every interval. FLEURS has no speaker ids, so its clusters are sentence
ids.

- H0: the new system is not better than the baseline (its error rate is not
  lower). The comparison planned as the main one is the CER of the fine-tuned
  model against the CER of the same checkpoint zero-shot; WER is tested the
  same way.
- Reported: the difference, its 95% percentile interval, and a one-sided
  p-value, the share of resamples in which the new system is not better
  (computed as (1 + count) / (1 + resamples), so it is never exactly zero).
- The number of resamples and the seed are the defaults `DEFAULT_RESAMPLES`
  and `DEFAULT_SEED` in `metrics.py`; the result is the same on every run.
- `bootstrap_interval` gives the interval of a single system's error rate in
  the same way.

Bootstrap intervals for ASR error rates follow Bisani and Ney (2004); paired
bootstrap resampling for comparing two systems follows Koehn (2004).

## Evaluation protocol

- One runner for every system (`src/huasr/evaluate.py`): it reads the clips
  of a split from the shards, checks them against the manifest, and passes
  the audio to the system. Baselines and our own models get the same clips.
  The GMM-HMM baseline is the one exception, because MFA decodes a folder at
  once: `src/huasr/gmm_hmm.py` writes the same clips of the same manifest to
  a folder, with the same check, and turns the output of MFA into the same
  two run files. It is scored by the same scorer.
- Every run leaves a run manifest and a prediction file with clip ids in
  `results/`; scores are computed from the prediction file by a separate
  script. [`results/README.md`](results/README.md) explains the folder and
  holds the result table, with baselines and our own models in separate
  sections.
- Every setting is chosen on the validation split.
- **The test split is used once, after all settings are frozen.** The runner
  refuses `--split test` unless the flag `--final-test` is given, and the run
  manifest records that the flag was used. No system has been run on the test
  split so far.

## Reproduce

Tested with Python 3.13 on Windows. Python 3.12 or newer is required.

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                  # makes the huasr package importable
python -m unittest                # unit tests (also run without the line above)
```

Raw data from VoxPopuli and FLEURS (no account needed):

```bash
python -m huasr.download voxpopuli --dry-run     # list the pinned files and their sizes
python -m huasr.download voxpopuli               # parquet files -> scratch/voxpopuli/
python -m huasr.download fleurs                  # test.tsv, test.tar.gz -> scratch/fleurs/
python -m huasr.subset voxpopuli                 # -> data/raw/voxpopuli/
python -m huasr.subset search                    # -> data/raw/voxpopuli/split_search.json, from selection.csv alone
python -m huasr.clean voxpopuli                  # -> data/clean/voxpopuli/
python -m huasr.subset fleurs                    # -> data/raw/fleurs/
python -m huasr.report                           # rewrite the generated tables of the READMEs
python -m huasr.subset unpack --raw data/raw/voxpopuli    # audio files -> scratch/audio/voxpopuli/
```

Evaluation (needs only the files of this repository and the model):

```bash
python -m huasr.evaluate --system faster-whisper-base     # validation split -> results/baselines/faster-whisper-base/validation/
python -m huasr.evaluate --system faster-whisper-small
python -m huasr.score --all                               # scores.json next to every run
python -m huasr.report                                    # result tables in results/README.md
```

`evaluate` runs the validation split unless `--split` says otherwise, and it
refuses `--split test` without `--final-test`: the test split is for the
final evaluation only. `--limit N` makes a trial run on the first N clips,
which is marked as incomplete and kept out of the tables; an existing run is
only replaced with `--force`. The Whisper systems take their model from the
local Hugging Face cache, at the revision pinned in `REVISIONS` in
`src/huasr/systems.py`, and download nothing. If the model is not there,
fetch it once (`base` or `small`):

```bash
python -c "from faster_whisper import download_model; from huasr.systems import REVISIONS; download_model('small', revision=REVISIONS['small'])"
```

GMM-HMM baseline. It needs its own conda environment with MFA and Kaldi; the
steps are Python entry points and are the same on Windows, Linux and macOS:

```bash
conda env create -f environment-gmm-hmm.yml      # MFA 3.4.3 and the Kaldi build of the committed run
conda activate huasr-gmm-hmm
# The package must be importable in this environment: set the environment variable PYTHONPATH
# to src (as the committed run did), or run "pip install -e . --no-deps".
python -m huasr.gmm_hmm prepare                  # training corpus, lexicon, settings -> scratch/gmm_hmm/
python -m huasr.gmm_hmm train --jobs 8           # language model, then acoustic model
python -m huasr.gmm_hmm decode --jobs 8          # the audio of the validation split -> transcripts
python -m huasr.gmm_hmm collect                  # -> results/baselines/gmm-hmm-mfa/validation/
python -m huasr.score results/baselines/gmm-hmm-mfa/validation
python -m huasr.report
```

`prepare` and `decode` read the shards and the clean manifest of this
repository and download nothing. `--work` puts the work folder (several GB:
the corpus as MFA reads it, the models, the temporary files of MFA) somewhere
else; the committed run used a folder outside the repository. `prepare` and
`train` are run once. `decode` and `collect` take `--split`, so that the same
trained models can be run on the test split at the end; like `evaluate`,
`decode` refuses `--split test` without `--final-test`. `prepare
--train-speakers ID ...` and `decode --limit N` make a trial run on a few
speakers and the first N clips, which is marked as incomplete and kept out of
the tables. The time the committed run took is in the table of
[`results/README.md`](results/README.md).

The committed data was built with exactly these commands and their defaults.
`clean`, `search`, `report` and `unpack` need only the files of this
repository; `subset voxpopuli` needs the downloaded source files and also
writes `split_search.json` itself.

An interrupted download continues on the next run. `subset voxpopuli` takes
`--train-hours`, `--validation-hours`, `--test-hours` and `--seed`; both
builders refuse to replace existing data unless `--force` is given, and
building twice from the same files gives the same manifest and summary. The
builders read the source one parquet row group at a time.

Common Voice test set (kept outside the tracked files). Manual steps, once per
person:

1. Create an account at <https://mozilladatacollective.com> and complete the
   profile.
2. Open the
   [dataset page](https://mozilladatacollective.com/datasets/cmu5wq4u200dknq072xswzofb),
   read the licence and the conditions of use, and accept them there.
3. Either download `common-voice-scripted-speech-27-0-hungar-9f5b5529.tar.gz`
   in the browser, or create an API key in the profile settings and put it in
   the environment variable `MDC_API_KEY`. Never write the key into a file of
   this repository.

```bash
python -m huasr.download commonvoice --info                               # published size, checksum, licence
python -m huasr.download commonvoice --archive <downloaded .tar.gz> --extract
python -m huasr.download commonvoice --extract                            # instead: uses MDC_API_KEY
python -m huasr.subset commonvoice --release external/common_voice/cv-corpus-27.0-2026-09-11/hu
```

The download checks the size and SHA-256 of the archive, unpacks it into
`external/common_voice/` and prints the release folder to pass to
`subset commonvoice`. The Mozilla Data Collective limits the number of dataset
downloads per day
([API reference](https://mozilladatacollective.com/api-reference/docs)); a
finished download is kept and not requested again.

## Repository layout

```text
README.md                  this file
LICENSE                    MIT licence for the code
requirements.txt           pinned versions of the packages the code imports (the last two only
                           for the Whisper systems)
environment-gmm-hmm.yml    conda environment of the GMM-HMM baseline: MFA and the Kaldi build of
                           the committed run
pyproject.toml             lets "pip install -e ." install the package from src/
.gitignore                 Python caches, local environment, scratch/, and everything of Common Voice
.gitattributes             LF line endings for text; tar, FLAC, MP3 and PNG files are binary
src/huasr/                 the package
  __init__.py              package marker, language code
  text.py                  scoring policy: text normalisation, word and character units
  metrics.py               edit-distance alignment, corpus-level WER/CER, bootstrap interval,
                           paired bootstrap over speakers
  download.py              pinned source files of VoxPopuli and FLEURS with size and SHA-256 check;
                           Common Voice archive into external/ only
  audio.py                 how audio is stored: WAV inspection, 16-bit FLAC, pass-through of
                           compressed input
  shards.py                tar shards below the file-size limit, manifest, unpacking with hash check
  subset.py                VoxPopuli: seeded subset whose splits share no speaker and no recording
                           session, selection log, search record; FLEURS: test split; Common
                           Voice: list of test clips outside the tracked files
  repair.py                restores the letter that the corpus normalisation deleted from the
                           reference, only where raw_text proves it
  clean.py                 data/raw -> data/clean: repaired references, repair log, removal log,
                           flags (English number words, repeated references, characters per
                           second), the numbers to report for the split
  systems.py               the contract every system fulfils (name, group, description, transcribe);
                           Whisper base and small through faster-whisper, at pinned revisions
  evaluate.py              runs one system on one split: run manifest and prediction file in
                           results/; refuses the test split without --final-test
  score.py                 prediction file -> scores.json: WER, CER, diacritic-insensitive WER with
                           edit counts and speaker-level intervals, real-time factor, with and
                           without the clips flagged for English number words
  gmm_hmm.py               GMM-HMM baseline with the Montreal Forced Aligner: corpus folders,
                           graphemic lexicon, training and decoding with MFA, its output as
                           run.json and predictions.csv; refuses the test split without --final-test
  report.py                writes the tables of the READMEs from the generated summary and score
                           files (data tables; result tables for baselines and for our own models;
                           the table of the GMM-HMM baseline)
tests/                     unit tests (python -m unittest)
  __init__.py              puts src/ on the import path; skips tests whose package is missing
  fixtures.py              generated signals and WAV files used by the tests
  test_text.py             normalisation rules
  test_metrics.py          alignment, WER/CER worked examples, bootstrap
  test_audio.py            WAV formats, 16-bit conversion, FLAC round trip, pass-through
  test_shards.py           shard size limit, identical rebuild, unpacking and tamper detection
  test_subset.py           sessions, assignment of speakers and sessions, leak check, builders for
                           VoxPopuli (rows and parquet files), FLEURS and the Common Voice list,
                           on generated sources
  test_download.py         pinned files, checksum check, resume, Common Voice kept out of the
                           tracked files; no network
  test_repair.py           reference repair: what is restored, what is left alone
  test_clean.py            clean manifest, flags, repair and removal logs, summary
  test_report.py           generated tables; the READMEs of this repository are up to date
  test_systems.py          checkpoint id and revision from the cache path, declared settings; no
                           model is loaded
  test_evaluate.py         runner with a stand-in system on generated data: prediction file, run
                           manifest, refusal of the test split without the flag
  test_score.py            scorer on hand-made prediction files (rates, edit counts, intervals,
                           real-time factor, flagged clips) and the result tables
  test_gmm_hmm.py          GMM-HMM baseline without MFA: graphemic lexicon, corpus folders, the
                           conversion of MFA output into the run files, its tables
data/raw/                  raw data, explained in data/raw/README.md
  voxpopuli/               clips/*.tar (FLAC audio in shards), manifest.csv, summary.json,
                           selection.csv, split_search.json
data/clean/                clean data, explained in data/clean/README.md
  voxpopuli/               manifest.csv (with the repaired reference), repair_log.csv,
                           removal_log.csv, summary.json
results/                   runs and result tables, explained in results/README.md
  baselines/<system>/<split>/   baselines (checkpoints used as published; the GMM-HMM system):
                                run.json, predictions.csv, scores.json
  own/<system>/<split>/         models trained in this project (none yet)
contributions/             one file per team member, written by that member
```

Not tracked (see `.gitignore`): `scratch/` for downloaded source files,
unpacked audio and the work folder of the GMM-HMM baseline, `external/` for
Common Voice.

## Status

Implemented and tested:

- Scoring policy (`src/huasr/text.py`).
- Edit-distance alignment, corpus-level WER and CER with substitution,
  deletion and insertion counts, error rate over any token unit, bootstrap
  interval and paired bootstrap over speakers (`src/huasr/metrics.py`).
- Audio storage rules, shards and unpacking (`src/huasr/audio.py`,
  `src/huasr/shards.py`).
- Builders for VoxPopuli, FLEURS and the Common Voice clip list
  (`src/huasr/subset.py`), tested on generated sources.
- Reference repair, clean step and README tables (`src/huasr/repair.py`,
  `src/huasr/clean.py`, `src/huasr/report.py`).
- Download script (`src/huasr/download.py`), tested without network.
- Evaluation: system contract with the two Whisper baselines
  (`src/huasr/systems.py`), runner with the guard on the test split
  (`src/huasr/evaluate.py`), scorer (`src/huasr/score.py`) and result tables
  (`src/huasr/report.py`), tested with a stand-in system and without a model.
- GMM-HMM baseline (`src/huasr/gmm_hmm.py`): corpus folders, graphemic
  lexicon, training and decoding with MFA, conversion of its output. The
  lexicon, the corpus folders and the conversion are tested without MFA.

Done on the real data: all six VoxPopuli files were downloaded with
`huasr.download` and matched their pinned sizes and SHA-256 values (16
November 2026); the subset was rebuilt with the default settings so that its
splits share no speaker and no session (18 November 2026); `unpack` restored
every clip from the shards with a matching SHA-256; the clean step and the
report were run. `commonvoice --info` was run against the real service.
Whisper base and small were run zero-shot on the whole validation split (18
November 2026) and scored; see [`results/README.md`](results/README.md). The
GMM-HMM baseline was trained on the whole training split and run on the whole
validation split with MFA on Windows (21 November 2026).

The FLEURS builder has been successfully run on real files. The Common Voice download with a key remains pending.

Not done yet:

- Any run on the test split.
- A decision on the references with English number words, and cleaning rules
  that remove clips.
- The other baselines, fine-tuning, the subword tokenizer, the comparison of
  two systems on real runs.
- Notebooks, error analysis.
- The contribution files in `contributions/` are templates that each member
  still has to write.

## Licence and citation

Code: MIT ([`LICENSE`](LICENSE)). The licence covers the code only; the data
keeps the licences and conditions listed under [Data](#data) and in
[`data/raw/README.md`](data/raw/README.md).

- Wang, C., Riviere, M., Lee, A., Wu, A., Talnikar, C., Haziza, D.,
  Williamson, M., Pino, J. and Dupoux, E. (2021). VoxPopuli: A Large-Scale
  Multilingual Speech Corpus for Representation Learning, Semi-Supervised
  Learning and Interpretation. *ACL 2021*, 993-1003.
  <https://aclanthology.org/2021.acl-long.80>. The recordings come from the
  European Parliament, which we thank for creating and sharing them
  ([legal notice](https://www.europarl.europa.eu/legal-notice/en/)).
- Conneau, A., Ma, M., Khanuja, S., Zhang, Y., Axelrod, V., Dalmia, S.,
  Riesa, J., Rivera, C. and Bapna, A. (2022). FLEURS: Few-shot Learning
  Evaluation of Universal Representations of Speech.
  <https://arxiv.org/abs/2205.12446>
- Ardila, R., Branson, M., Davis, K., Henretty, M., Kohler, M., Meyer, J.,
  Morais, R., Saunders, L., Tyers, F. M. and Weber, G. (2020). Common Voice: A
  Massively-Multilingual Speech Corpus. *Proceedings of LREC 2020*,
  4211-4215. <https://arxiv.org/abs/1912.06670>
- Bisani, M. and Ney, H. (2004). Bootstrap estimates for confidence intervals
  in ASR performance evaluation. *ICASSP 2004*.
  <https://doi.org/10.1109/ICASSP.2004.1326009>
- Koehn, P. (2004). Statistical significance tests for machine translation
  evaluation. *EMNLP 2004*. <https://aclanthology.org/W04-3250/>

## Team

| Member | Student ID | GitHub |
| --- | --- | --- |
| Berkay Hakan Sadi | 2309011065 | [hakanstarlord](https://github.com/hakanstarlord) |
| Fatih Safran Dündar | 2309011133 | [fatihsafran](https://github.com/fatihsafran) |
| Aamer mashar sanjer | 2309015705 | [aamersanjer0-a11y](https://github.com/aamersanjer0-a11y) |
| Kaan Kalkan | 220901594 | [kaankalkan03](https://github.com/kaankalkan03) |
| hilal ibrahimoglu | 229920269 | [hilalibrahimoglu34](https://github.com/hilalibrahimoglu34) |
| MOOAZ SADIK | 2309015405 | [moathkhaled2k5](https://github.com/moathkhaled2k5) |

Each member's own record is in [`contributions/`](contributions/).
