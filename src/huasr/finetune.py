"""Fine-tune `openai/whisper-small` on the training split: the model trained in this project.

    python -m huasr.finetune plan              which clips would be used and with which settings (no model, no GPU)
    python -m huasr.finetune train --smoke     a few steps on a few clips, to check the whole path in minutes
    python -m huasr.finetune train [--resume]  the real run: best model, training record

Needs the packages of `requirements-train.txt`, the unpacked audio
(`python -m huasr.subset unpack --raw data/raw/voxpopuli`) and the pinned
checkpoint (`python -m huasr.whisper_hf fetch`). Everything below was fixed
before the first run. The validation split decides one thing only: which
checkpoint is kept.

Data
    Training clips come from the split `train` of the clean manifest, validation clips from `validation`.
    Left out, in this order, and counted in the training record (`select`):
    - training clips whose reference has English number words (`english_number_words` > 0): the corpus
      wrote digits out in English, which nobody says in the recording, so the text is a wrong target;
    - clips longer than 30 s: Whisper reads one 30 s window, so the end of the audio would be cut off while
      the text went on. For validation this only concerns the selection metric below; the final evaluation
      with `huasr.evaluate` covers every validation clip;
    - clips whose target is empty, or longer than the 448 tokens the decoder can read.
    The test split is refused by this module.

Target text
    The column `reference` after the scoring policy `huasr.text.normalize` (lower case, no punctuation,
    diacritics kept), so the model learns to write what the scorer counts. Labels are the start token, the
    tokens for Hungarian, for transcription and for "no timestamps", the text and the end token.

Training
    HYPERPARAMETERS: 2,000 updates of 16 clips (about 7.5 passes over the training clips), AdamW, learning
    rate 1e-5 after a linear warm-up of 200 updates, then linear decay to zero; mixed precision (fp16) on the
    GPU; seed 0; no augmentation, no frozen layers. If 16 clips do not fit in the GPU memory, `--batch-size 8`
    accumulates two batches per update, which leaves the updates as they are; the record says which was used.

Selection
    Every 250 updates, and once before the first, the validation clips are decoded in batches (greedy, at
    most 225 tokens, language and task fixed) and scored with `huasr.metrics` under the scoring policy. The
    checkpoint with the lowest corpus-level WER is kept; of equal ones, the earliest. Nothing else is chosen
    on validation data.

Output
    <work>/checkpoints/  the last and the best checkpoint of the run (to resume an interrupted run)
    <work>/best/         the kept model with tokenizer and feature extractor files, and `training.json`
    results/own/hf-whisper-small-finetuned/training.json
                         the training record: data counts, settings, loss and validation scores per step,
                         hardware, package versions. Not written by a `--smoke` run.
    `<work>` is `scratch/finetune/hf-whisper-small` and is not tracked; no weights go into the repository.
    The reported scores of the model come from `python -m huasr.evaluate --system hf-whisper-small-finetuned`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from . import LANGUAGE, metrics, whisper_hf
from .audio import decode_flac
from .evaluate import CLEAN, RESULTS, _utc_now, cpu_name
from .systems import OWN, cached_checkpoint, sha256_of
from .text import normalize

SYSTEM = whisper_hf.FINETUNED
AUDIO = Path("scratch/audio/voxpopuli")
WORK = whisper_hf.FINETUNED_FOLDER.parent
SMOKE_SUFFIX = "-smoke"                     # a smoke run works in <work>-smoke
SPLITS = ("train", "validation")            # the only splits this module reads
SEED = 0
WINDOW_MS = whisper_hf.WINDOW_SECONDS * 1000
WINDOW_SAMPLES = whisper_hf.WINDOW_SECONDS * whisper_hf.SAMPLE_RATE
MAX_LABEL_TOKENS = 448                      # positions of the decoder of Whisper
IGNORE = -100                               # label value that the loss skips
PREFIX = ("<|startoftranscript|>", f"<|{LANGUAGE}|>", "<|transcribe|>", "<|notimestamps|>")  # every label begins so
HYPERPARAMETERS = {
    "max_steps": 2000, "batch_size": 16, "gradient_accumulation_steps": 1, "learning_rate": 1e-5,
    "lr_scheduler": "linear", "warmup_steps": 200, "optimizer": "adamw_torch", "weight_decay": 0.0,
    "max_grad_norm": 1.0, "mixed_precision": "fp16", "evaluate_every_steps": 250, "eval_batch_size": 16,
    "log_every_steps": 25, "checkpoints_kept": 2,
}
# How checkpoints are compared: the metric, and how the validation clips are decoded for it (arguments of `generate`).
SELECTION = {"metric": "wer", "decoding": {"language": LANGUAGE, "task": "transcribe", "return_timestamps": False,
                                           "num_beams": 1, "max_length": 225}}
SMOKE = {"max_steps": 8, "warmup_steps": 2, "evaluate_every_steps": 4, "log_every_steps": 1}
SMOKE_CLIPS = {"train": 64, "validation": 16}
PACKAGES = (*whisper_hf.PACKAGES, "accelerate", "numpy", "av")


def target_text(reference: str) -> str:
    """What the model is trained to write for a reference: the reference under the scoring policy."""
    return normalize(reference)


def read_manifest(clean: Path) -> list[dict[str, str]]:
    with (Path(clean) / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def select(rows: list[dict[str, str]], split: str,
           label_length: Callable[[str], int] | None = None) -> tuple[list[dict[str, str]], dict]:
    """The clips of `split` that are used, in the order of the manifest, and what each rule left out.

    A clip is counted under the first rule that leaves it out (the rules are in the module text). The rule
    on the number of label tokens needs the tokenizer: `label_length` gives the number of label tokens of a
    target text, and without it the rule is not applied.
    """
    if split not in SPLITS:
        raise ValueError(f"the split {split!r} is not for training or checkpoint selection")
    rows = [row for row in rows if row["split"] == split]
    excluded = {"english_number_words": 0, "longer_than_window": 0, "empty_target": 0, "too_many_label_tokens": 0}
    kept = []
    for row in rows:
        target = target_text(row["reference"])
        if split == "train" and int(row["english_number_words"] or 0) > 0:
            excluded["english_number_words"] += 1
        elif int(row["duration_ms"]) > WINDOW_MS:
            excluded["longer_than_window"] += 1
        elif not target:
            excluded["empty_target"] += 1
        elif label_length is not None and label_length(target) > MAX_LABEL_TOKENS:
            excluded["too_many_label_tokens"] += 1
        else:
            kept.append(row)
    if split != "train":
        excluded["english_number_words"] = None    # None: the rule is not applied
    if label_length is None:
        excluded["too_many_label_tokens"] = None
    return kept, {
        "clips_in_split": len(rows), "hours_in_split": _hours(rows),
        "longer_than_window_in_split": sum(int(row["duration_ms"]) > WINDOW_MS for row in rows),
        "excluded": excluded, "clips": len(kept), "hours": _hours(kept),
        "speakers": len({row["speaker_id"] for row in kept}),
        "clips_with_english_number_words": sum(int(row["english_number_words"] or 0) > 0 for row in kept),
    }


def _hours(rows: list[dict[str, str]]) -> float:
    return round(sum(int(row["duration_ms"]) for row in rows) / 3_600_000, 3)


def left_out(excluded: dict) -> str:
    """The counts of `select` in words, for the log."""
    return ", ".join(f"{rule.replace('_', ' ')} {'(rule not applied)' if count is None else format(count, ',')}"
                     for rule, count in excluded.items())


def hyperparameters(smoke: bool = False, batch_size: int | None = None) -> dict:
    """The fixed settings; `smoke` shortens the run, `batch_size` splits an update into several batches."""
    values = {**HYPERPARAMETERS, **(SMOKE if smoke else {})}
    if batch_size is not None:
        clips = values["batch_size"] * values["gradient_accumulation_steps"]
        if batch_size < 1 or clips % batch_size:
            raise ValueError(f"the batch size must divide the {clips} clips of an update, got {batch_size}")
        values["batch_size"], values["gradient_accumulation_steps"] = batch_size, clips // batch_size
    return values


def trainer_arguments(values: dict, folder: Path, cuda: bool, workers: int, first_run: bool = True) -> dict:
    """Keyword arguments of `transformers.Seq2SeqTrainingArguments` for the settings in `values`."""
    return {
        "output_dir": str(folder), "max_steps": values["max_steps"],
        "per_device_train_batch_size": values["batch_size"],
        "gradient_accumulation_steps": values["gradient_accumulation_steps"],
        "learning_rate": values["learning_rate"], "lr_scheduler_type": values["lr_scheduler"],
        "warmup_steps": values["warmup_steps"], "optim": values["optimizer"], "weight_decay": values["weight_decay"],
        "max_grad_norm": values["max_grad_norm"], "fp16": cuda and values["mixed_precision"] == "fp16",
        "eval_strategy": "steps", "eval_steps": values["evaluate_every_steps"], "eval_on_start": first_run,
        "per_device_eval_batch_size": values["eval_batch_size"],
        "save_strategy": "steps", "save_steps": values["evaluate_every_steps"],
        "save_total_limit": values["checkpoints_kept"], "load_best_model_at_end": True,
        "metric_for_best_model": SELECTION["metric"], "greater_is_better": False,
        "logging_strategy": "steps", "logging_steps": values["log_every_steps"], "report_to": "none",
        "disable_tqdm": True, "seed": SEED, "data_seed": SEED, "dataloader_num_workers": workers,
    }


class ClipDataset:
    """One item per clip: the log-mel features of its stored audio and the label ids of its target text."""

    def __init__(self, rows: list[dict[str, str]], labels: list[list[int]], audio: Path, feature_extractor):
        if len(rows) != len(labels):
            raise ValueError(f"{len(rows)} clips but {len(labels)} label sequences")
        if any(row["split"] not in SPLITS for row in rows):
            raise ValueError("only clips of the training and the validation split may be read here")
        self.rows, self.labels, self.audio, self.feature_extractor = rows, labels, Path(audio), feature_extractor

    def __len__(self) -> int:
        return len(self.rows)

    def samples(self, index: int) -> np.ndarray:
        """The stored audio of a clip as float32 between -1 and 1, checked against the manifest."""
        row = self.rows[index]
        data = (self.audio / row["split"] / row["file"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError(f"{row['file']} does not match the SHA-256 in the manifest")
        samples, rate = decode_flac(data)
        if rate != whisper_hf.SAMPLE_RATE or len(samples) > WINDOW_SAMPLES:
            raise ValueError(f"{row['file']}: {len(samples)} samples at {rate} Hz do not fit in one window of Whisper")
        return samples.astype(np.float32) / 32768.0

    def __getitem__(self, index: int) -> dict:
        features = self.feature_extractor(self.samples(index), sampling_rate=whisper_hf.SAMPLE_RATE,
                                          return_tensors="np")["input_features"][0]
        return {"input_features": np.asarray(features, dtype=np.float32), "labels": self.labels[index]}


class Collator:
    """Batch of items of `ClipDataset`: features stacked, labels padded with IGNORE.

    Every label sequence begins with the start token. The model puts that token in front of the decoder
    input itself, so it is cut off here.
    """

    def __init__(self, decoder_start_token_id: int):
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, items: list[dict]) -> dict:
        import torch

        if any(not item["labels"] or item["labels"][0] != self.decoder_start_token_id for item in items):
            raise ValueError("a label sequence does not begin with the start token")
        targets = [item["labels"][1:] for item in items]
        labels = torch.full((len(items), max(len(target) for target in targets)), IGNORE, dtype=torch.long)
        for row, target in enumerate(targets):
            labels[row, :len(target)] = torch.tensor(target, dtype=torch.long)
        features = torch.from_numpy(np.stack([item["input_features"] for item in items]))
        return {"input_features": features, "labels": labels}


def rates(references: list[str], hypotheses: list[str], flagged: list[bool]) -> dict[str, float]:
    """Corpus-level WER and CER in percent, and the WER without the clips flagged for English number words."""
    result = {"wer": 100 * metrics.wer(references, hypotheses).rate,
              "cer": 100 * metrics.cer(references, hypotheses).rate}
    rest = [index for index, flag in enumerate(flagged) if not flag]
    if rest and len(rest) < len(flagged):
        result["wer_without_english_number_words"] = 100 * metrics.wer(
            [references[index] for index in rest], [hypotheses[index] for index in rest]).rate
    return result


class SelectionMetrics:
    """Scores the token ids that the trainer generated for the validation clips, in the order of the clips."""

    def __init__(self, tokenizer, references: list[str], flagged: list[bool]):
        self.tokenizer, self.references, self.flagged = tokenizer, references, flagged

    def __call__(self, prediction) -> dict[str, float]:
        tokens = prediction.predictions[0] if isinstance(prediction.predictions, tuple) else prediction.predictions
        tokens = np.asarray(tokens)
        if len(tokens) != len(self.references):
            raise ValueError(f"{len(tokens)} hypotheses for {len(self.references)} validation clips")
        padding = self.tokenizer.pad_token_id
        if padding is None:
            padding = self.tokenizer.eos_token_id
        tokens = np.where(tokens < 0, padding, tokens)  # the trainer fills ragged batches with a negative value
        hypotheses = [text.strip() for text in self.tokenizer.batch_decode(tokens, skip_special_tokens=True)]
        return rates(self.references, hypotheses, self.flagged)


def evaluations(history: list[dict]) -> list[dict]:
    """The validation scores in the log of the trainer, one row per step (the last one if a step repeats)."""
    rows = {}
    for entry in history:
        if "eval_wer" in entry:
            rows[int(entry["step"])] = {
                "step": int(entry["step"]), "epoch": round(float(entry.get("epoch") or 0.0), 3),
                "loss": _rounded(entry.get("eval_loss")), "wer": _rounded(entry["eval_wer"]),
                "cer": _rounded(entry.get("eval_cer")),
                "wer_without_english_number_words": _rounded(entry.get("eval_wer_without_english_number_words")),
                "seconds": _rounded(entry.get("eval_runtime"), 1)}
    return [rows[step] for step in sorted(rows)]


def training_log(history: list[dict]) -> list[dict]:
    """Training loss and learning rate per logged step."""
    return [{"step": int(entry["step"]), "epoch": round(float(entry.get("epoch") or 0.0), 3),
             "loss": _rounded(entry["loss"]), "learning_rate": entry.get("learning_rate"),
             "grad_norm": _rounded(entry.get("grad_norm"))}
            for entry in history if "loss" in entry and "eval_wer" not in entry]


def _rounded(value, digits: int = 4):
    return None if value is None else round(float(value), digits)


def best_evaluation(rows: list[dict]) -> dict:
    """The evaluation with the lowest WER among the steps that have a checkpoint; of equal ones the earliest."""
    candidates = [row for row in rows if row["step"] > 0]
    if not candidates:
        raise ValueError("no checkpoint was evaluated")
    return min(candidates, key=lambda row: (row["wer"], row["step"]))


class LabelEncoder:
    """Label ids of a target text: the four tokens of PREFIX, the tokens of the text, the end token.

    The sequence is put together here, token by token, and not left to the tokenizer, so that it does not
    depend on how a version of the tokenizer adds special tokens.
    """

    def __init__(self, tokenizer):
        self.tokenizer, self.end = tokenizer, tokenizer.eos_token_id
        self.prefix = list(tokenizer.convert_tokens_to_ids(list(PREFIX)))
        unknown = (None, self.end, getattr(tokenizer, "unk_token_id", None))  # what an unknown token is turned into
        if len({token for token in self.prefix if token not in unknown}) != len(PREFIX) or self.end is None:
            raise ValueError(f"the tokenizer does not know the tokens {PREFIX}: {self.prefix}")

    def __call__(self, text: str) -> list[int]:
        return [*self.prefix, *self.tokenizer(text, add_special_tokens=False)["input_ids"], self.end]


def projection(step_seconds: list[float], evaluation_rows: list[dict], evaluated_clips: int, clips: int) -> dict:
    """What the measured times of a smoke run mean for the full run (rough: few steps, few clips)."""
    full = hyperparameters()
    per_step = float(np.mean(step_seconds[1:] or step_seconds))  # the first step also sets the GPU up
    per_clip = float(np.mean([row["seconds"] for row in evaluation_rows])) / max(1, evaluated_clips)
    passes = full["max_steps"] // full["evaluate_every_steps"] + 1
    training, evaluation = full["max_steps"] * per_step, passes * clips * per_clip
    return {"seconds_per_training_step": round(per_step, 3), "seconds_per_evaluated_clip": round(per_clip, 3),
            "training_hours": round(training / 3600, 2), "evaluation_hours": round(evaluation / 3600, 2),
            "total_hours": round((training + evaluation) / 3600, 2)}


def plan(clean: Path = CLEAN) -> dict:
    """Clips and settings of the real run, from the manifest alone."""
    manifest = read_manifest(clean)
    return {"system": SYSTEM, "group": OWN,
            "base_checkpoint": {"repository": whisper_hf.REPOSITORY, "revision": whisper_hf.REVISION},
            "target_text": "huasr.text.normalize(reference)", "seed": SEED,
            "data": {split: select(manifest, split)[1] for split in SPLITS},
            "hyperparameters": hyperparameters(), "selection": SELECTION}


def train(options: argparse.Namespace) -> int:
    values = hyperparameters(options.smoke, options.batch_size)
    work = Path(options.work) if options.work else WORK
    if options.smoke:
        work = work.with_name(work.name + SMOKE_SUFFIX)  # a smoke run never touches the folder of a real run
    checkpoints, best_folder = work / "checkpoints", work / "best"
    if (best_folder / whisper_hf.RECORD).is_file() and not (options.force or options.smoke):
        if not options.resume:
            raise FileExistsError(f"{best_folder} already holds a finished run; use --force to train again")
        print(f"{best_folder} already holds a finished run; nothing to do (use --force to train again)")
        return 0
    if options.force or options.smoke:
        for folder in (checkpoints, best_folder):
            shutil.rmtree(folder, ignore_errors=True)
    resume = any(checkpoints.glob("checkpoint-*"))
    if resume and not options.resume:
        raise FileExistsError(f"{checkpoints} holds checkpoints of an unfinished run; continue it with --resume "
                              f"or start again with --force")
    if not (Path(options.audio) / "train").is_dir():
        raise FileNotFoundError(f"no unpacked audio in {options.audio}: run "
                                f"python -m huasr.subset unpack --raw data/raw/voxpopuli")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    snapshot = whisper_hf.snapshot()
    base = {**cached_checkpoint(snapshot), **whisper_hf.published_weights(snapshot)}
    import torch
    from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback, WhisperFeatureExtractor,
                              WhisperForConditionalGeneration, WhisperTokenizer)
    from transformers.trainer_callback import PrinterCallback

    device = whisper_hf.accelerator()
    cuda = device["device"] == "cuda"
    if not cuda and not (options.smoke or options.allow_cpu):
        raise ValueError("no GPU: in Colab choose Runtime > Change runtime type > T4 GPU (or pass --allow-cpu)")

    # Data: which clips, their labels, and the two datasets.
    feature_extractor = WhisperFeatureExtractor.from_pretrained(str(snapshot), local_files_only=True)
    tokenizer = WhisperTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    encode = LabelEncoder(tokenizer)
    manifest = read_manifest(options.clean)
    rows, data, datasets = {}, {}, {}
    for split in SPLITS:
        rows[split], data[split] = select(manifest, split, lambda text: len(encode(text)))
        print(f"{split}: {data[split]['clips_in_split']:,} clips in the split; left out: "
              f"{left_out(data[split]['excluded'])}; used: {data[split]['clips']:,} clips, "
              f"{data[split]['hours']:.2f} h, {data[split]['speakers']} speakers", flush=True)
        if options.smoke:
            rows[split] = rows[split][:SMOKE_CLIPS[split]]
        labels = [encode(target_text(row["reference"])) for row in rows[split]]
        datasets[split] = ClipDataset(rows[split], labels, options.audio, feature_extractor)
    first = datasets["train"][0]["input_features"]
    print(f"features of one clip: {first.shape} {first.dtype}; labels of the first clip: "
          f"{len(datasets['train'].labels[0])} tokens", flush=True)

    # Model: the published checkpoint; generation during the evaluations is fixed to Hungarian transcription.
    model = WhisperForConditionalGeneration.from_pretrained(str(snapshot), local_files_only=True)
    generation = model.generation_config
    prompt = [model.config.decoder_start_token_id, generation.lang_to_id[PREFIX[1]],
              generation.task_to_id["transcribe"], generation.no_timestamps_token_id]
    if encode.prefix != prompt:
        raise ValueError(f"the labels begin with {encode.prefix}, but the model is prompted with {prompt}")
    model.generation_config.language, model.generation_config.task = LANGUAGE, "transcribe"
    model.generation_config.return_timestamps = False
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None

    class Progress(TrainerCallback):
        """Times every update and prints one line per log entry."""

        def __init__(self):
            self.step_seconds, self.began, self.started = [], None, time.perf_counter()

        def on_step_begin(self, args, state, control, **kwargs):
            self.began = time.perf_counter()

        def on_step_end(self, args, state, control, **kwargs):
            self.step_seconds.append(time.perf_counter() - self.began)

        def on_log(self, args, state, control, logs=None, **kwargs):
            logs, minutes = logs or {}, (time.perf_counter() - self.started) / 60
            if "eval_wer" in logs:
                print(f"step {state.global_step:>5}  validation WER {logs['eval_wer']:.2f}%  CER "
                      f"{logs.get('eval_cer', float('nan')):.2f}%  loss {logs.get('eval_loss', float('nan')):.4f}  "
                      f"({logs.get('eval_runtime', 0):.0f} s; {minutes:.1f} min since the start)", flush=True)
            elif "loss" in logs:
                print(f"step {state.global_step:>5} of {state.max_steps}  loss {logs['loss']:.4f}  learning rate "
                      f"{logs.get('learning_rate', float('nan')):.2e}  ({minutes:.1f} min)", flush=True)

    class Trainer(Seq2SeqTrainer):
        """Decodes the validation clips as SELECTION says, from the audio alone.

        The step of the library would pass the labels to `generate` as well and take the decoding from
        several places; here the call is the same as in the evaluation system (`huasr.whisper_hf`).
        """

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **kwargs):
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                with self.compute_loss_context_manager():
                    loss = model(**inputs).loss.detach()
                tokens = model.generate(inputs["input_features"], **SELECTION["decoding"])
            return loss, tokens, inputs["labels"]

    progress = Progress()
    references = [target_text(row["reference"]) for row in rows["validation"]]
    flagged = [int(row["english_number_words"] or 0) > 0 for row in rows["validation"]]
    arguments = trainer_arguments(values, checkpoints, cuda, options.workers, first_run=not resume)
    trainer = Trainer(
        model=model, args=Seq2SeqTrainingArguments(**arguments), train_dataset=datasets["train"],
        eval_dataset=datasets["validation"], data_collator=Collator(model.config.decoder_start_token_id),
        compute_metrics=SelectionMetrics(tokenizer, references, flagged), processing_class=feature_extractor,
        callbacks=[progress])
    trainer.remove_callback(PrinterCallback)

    started, before = _utc_now(), time.perf_counter()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    try:
        trainer.train(resume_from_checkpoint=True if resume else None)
    except torch.OutOfMemoryError:
        print("the GPU ran out of memory: run again with --force --batch-size 8 (two batches per update)",
              file=sys.stderr)
        raise
    seconds = time.perf_counter() - before
    history = trainer.state.log_history
    scores = evaluations(history)
    best = best_evaluation(scores)

    # Keep the best checkpoint: read it back from its folder, so that what is saved is what was scored.
    source = checkpoints / f"checkpoint-{best['step']}"
    if not (source / whisper_hf.WEIGHTS).is_file():
        raise FileNotFoundError(f"{source} is missing: the trainer kept {trainer.state.best_model_checkpoint}")
    peak = torch.cuda.max_memory_allocated() if cuda else None
    del trainer, model
    kept = WhisperForConditionalGeneration.from_pretrained(str(source), local_files_only=True)
    kept.config.use_cache = True  # the trainer switches the cache of the decoder off while it trains
    kept.save_pretrained(str(best_folder))
    feature_extractor.save_pretrained(str(best_folder))
    tokenizer.save_pretrained(str(best_folder))

    complete = not options.smoke and scores[-1]["step"] == values["max_steps"]
    record = {
        "system": SYSTEM, "group": OWN, "complete": complete, "smoke": bool(options.smoke),
        "base_checkpoint": base,
        "data": {"clean_manifest": (Path(options.clean) / "manifest.csv").as_posix(),
                 "clean_manifest_sha256": sha256_of(Path(options.clean) / "manifest.csv"),
                 "audio": Path(options.audio).as_posix(), "reference_column": "reference", **data,
                 "clips_read": {split: len(rows[split]) for split in SPLITS}},
        "target_text": "huasr.text.normalize(reference)",
        "hyperparameters": {**values, "mixed_precision": values["mixed_precision"] if cuda else "none"},
        "seed": SEED,
        "selection": {**SELECTION, "batch_size": values["eval_batch_size"],
                      "scoring": "corpus-level rates of huasr.metrics under huasr.text.normalize, in percent"},
        "evaluations": scores, "best": best, "training_log": training_log(history),
        "weights": {"file": whisper_hf.WEIGHTS, "bytes": (best_folder / whisper_hf.WEIGHTS).stat().st_size,
                    "sha256": sha256_of(best_folder / whisper_hf.WEIGHTS)},
        "seconds": {"total": round(seconds, 1), "training_steps": round(sum(progress.step_seconds), 1),
                    "evaluations": round(sum(row["seconds"] or 0 for row in scores), 1)},
        "hardware": {**device, "peak_gpu_memory_bytes": peak, "processor": cpu_name(),
                     "logical_cpus": os.cpu_count(), "system": platform.platform()},
        "packages": {"python": platform.python_version(), **whisper_hf.versions(PACKAGES)},
        "resumed": resume, "data_workers": options.workers,
        "started_utc": started, "finished_utc": _utc_now(),
    }
    targets = [best_folder / whisper_hf.RECORD]
    if complete:
        targets.append(Path(options.results) / OWN / SYSTEM / whisper_hf.RECORD)
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

    print(f"best checkpoint: step {best['step']} with validation WER {best['wer']:.2f}% "
          f"(before training: {scores[0]['wer']:.2f}% at step {scores[0]['step']})")
    print(f"wrote {', '.join(target.as_posix() for target in targets)} and the model in {best_folder.as_posix()}")
    if peak is not None:
        print(f"peak GPU memory: {peak / 2 ** 30:.1f} GiB of {device['gpu_memory_bytes'] / 2 ** 30:.1f} GiB")
    if options.smoke:
        estimate = projection(progress.step_seconds, scores, len(rows["validation"]), data["validation"]["clips"])
        print(f"smoke run finished. Measured {estimate['seconds_per_training_step']} s per update and "
              f"{estimate['seconds_per_evaluated_clip']} s per validation clip; the full run would take about "
              f"{estimate['training_hours']} h of updates and {estimate['evaluation_hours']} h of evaluations "
              f"(rough figures from a few steps)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    planner = commands.add_parser("plan", help="print the clips and the settings of the real run")
    trainer = commands.add_parser("train", help="fine-tune, keep the best checkpoint, write the training record")
    for command in (planner, trainer):
        command.add_argument("--clean", type=Path, default=CLEAN)
    trainer.add_argument("--audio", type=Path, default=AUDIO, help="unpacked audio (python -m huasr.subset unpack)")
    trainer.add_argument("--work", type=Path, help=f"checkpoints and best model; default {WORK.as_posix()}")
    trainer.add_argument("--results", type=Path, default=RESULTS)
    trainer.add_argument("--smoke", action="store_true",
                         help=f"{SMOKE['max_steps']} updates on {SMOKE_CLIPS['train']} training clips; works in "
                              f"{WORK.as_posix()}{SMOKE_SUFFIX} and writes nothing to the results")
    trainer.add_argument("--resume", action="store_true", help="continue an interrupted run from its last checkpoint")
    trainer.add_argument("--force", action="store_true", help="delete the checkpoints and the model of an earlier run")
    trainer.add_argument("--batch-size", type=int,
                         help="clips per batch if 16 do not fit in the GPU memory; an update still sees 16 clips")
    trainer.add_argument("--workers", type=int, default=2, help="processes that read and prepare the audio")
    trainer.add_argument("--allow-cpu", action="store_true", help="run without a GPU (very slow)")
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            print(json.dumps(plan(args.clean), indent=2))
            return 0
        return train(args)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
