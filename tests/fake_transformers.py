"""A stand-in for the few parts of `transformers` that `huasr.finetune` calls.

It lets the tests run the whole training command without a real model: the
data goes through the real dataset and collator, the evaluations through the
real prediction step and the real metric, and checkpoints are folders with a
small file. Names, arguments and the bookkeeping of the trainer (log entries,
checkpoint folders, which checkpoint is the best, which are kept) follow
transformers 5.23.0, the version pinned in `requirements-train.txt`. Every
call is checked strictly, so an argument that the real classes do not have
fails here as well.
"""

import contextlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np

START, HUNGARIAN, TRANSCRIBE, NO_TIMESTAMPS, END = 50258, 50286, 50359, 50363, 50257
WEIGHTS = "model.safetensors"
# Fields of Seq2SeqTrainingArguments that huasr.finetune sets; each was looked up in transformers 5.23.0.
ARGUMENTS = {
    "output_dir", "max_steps", "per_device_train_batch_size", "per_device_eval_batch_size",
    "gradient_accumulation_steps", "learning_rate", "lr_scheduler_type", "warmup_steps", "optim", "weight_decay",
    "max_grad_norm", "fp16", "eval_strategy", "eval_steps", "eval_on_start", "save_strategy", "save_steps",
    "save_total_limit", "load_best_model_at_end", "metric_for_best_model", "greater_is_better", "logging_strategy",
    "logging_steps", "report_to", "disable_tqdm", "seed", "data_seed", "dataloader_num_workers",
}
# Arguments of `generate` that the evaluations during training must pass, and nothing else.
DECODING = {"language": "hu", "task": "transcribe", "return_timestamps": False, "num_beams": 1, "max_length": 225}


class Seq2SeqTrainingArguments:
    def __init__(self, **arguments):
        unknown = set(arguments) - ARGUMENTS
        if unknown:
            raise TypeError(f"unexpected keyword arguments {sorted(unknown)}")
        if arguments["load_best_model_at_end"] and (arguments["eval_strategy"] != arguments["save_strategy"]
                                                    or arguments["save_steps"] % arguments["eval_steps"]):
            raise ValueError("load_best_model_at_end needs matching evaluation and save steps")
        self.__dict__.update(arguments)


class TrainerCallback:
    def on_step_begin(self, args, state, control, **kwargs):
        pass

    def on_step_end(self, args, state, control, **kwargs):
        pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        pass


class PrinterCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        raise AssertionError("the default printer was not removed")


class WhisperFeatureExtractor:
    @classmethod
    def from_pretrained(cls, folder, *, local_files_only):
        assert local_files_only and (Path(folder) / "preprocessor_config.json").is_file()
        return cls()

    def __call__(self, samples, *, sampling_rate, return_tensors):
        assert sampling_rate == 16_000 and return_tensors == "np" and samples.dtype == np.float32
        return {"input_features": np.full((1, 80, 3000), float(np.abs(samples).mean()), dtype=np.float32)}

    def save_pretrained(self, folder):
        (Path(folder) / "preprocessor_config.json").write_text("{}", encoding="utf-8")


class WhisperTokenizer:
    """One token per word; the ids of the words are given out in the order in which the words are met."""

    pad_token_id = eos_token_id = unk_token_id = END
    special = {"<|startoftranscript|>": START, "<|hu|>": HUNGARIAN, "<|transcribe|>": TRANSCRIBE,
               "<|notimestamps|>": NO_TIMESTAMPS}

    def __init__(self):
        self.words = {}

    @classmethod
    def from_pretrained(cls, folder, *, local_files_only):
        assert local_files_only and (Path(folder) / "tokenizer.json").is_file()
        return cls()

    def __call__(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return {"input_ids": [self.words.setdefault(word, 1000 + len(self.words)) for word in text.split()]}

    def convert_tokens_to_ids(self, tokens):
        return [self.special.get(token, self.unk_token_id) for token in tokens]

    def batch_decode(self, tokens, *, skip_special_tokens):
        assert skip_special_tokens and (np.asarray(tokens) >= 0).all()
        words = {token: word for word, token in self.words.items()}
        return [" " + " ".join(words[int(token)] for token in row if int(token) in words) for row in tokens]

    def save_pretrained(self, folder):
        (Path(folder) / "tokenizer.json").write_text("{}", encoding="utf-8")


class WhisperForConditionalGeneration:
    """Writes the text of the labels it was shown last, without the last `dropped_words` words of every clip."""

    def __init__(self, weights: bytes, use_cache: bool):
        self.weights, self.dropped_words, self.shown = weights, 0, None
        self.config = SimpleNamespace(decoder_start_token_id=START, forced_decoder_ids=[[1, 50259]],
                                      use_cache=use_cache)
        self.generation_config = SimpleNamespace(
            lang_to_id={"<|en|>": 50259, "<|hu|>": HUNGARIAN},
            task_to_id={"transcribe": TRANSCRIBE, "translate": 50358},
            no_timestamps_token_id=NO_TIMESTAMPS, forced_decoder_ids=[[1, None], [2, TRANSCRIBE]], max_length=448)

    @classmethod
    def from_pretrained(cls, folder, *, local_files_only):
        assert local_files_only
        folder = Path(folder)
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        return cls((folder / WEIGHTS).read_bytes(), config.get("use_cache", True))

    def save_pretrained(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / WEIGHTS).write_bytes(self.weights)
        settings = {"use_cache": self.config.use_cache, "forced_decoder_ids": self.config.forced_decoder_ids,
                    "language": getattr(self.generation_config, "language", None)}
        (folder / "config.json").write_text(json.dumps(settings), encoding="utf-8")

    def __call__(self, *, input_features, labels):
        import torch

        assert tuple(input_features.shape[1:]) == (80, 3000) and len(labels) == len(input_features)
        self.shown = labels
        return SimpleNamespace(loss=torch.tensor(1.0 / (1 + self.dropped_words)))

    def generate(self, input_features, **decoding):
        import torch

        assert decoding == DECODING and len(input_features) == len(self.shown)
        rows = []
        for labels in self.shown.tolist():
            text = [token for token in labels if token >= 0][3:-1]  # without the prompt and the end token
            kept = text[:max(0, len(text) - self.dropped_words)]
            rows.append([START, HUNGARIAN, TRANSCRIBE, NO_TIMESTAMPS, *kept, END])
        longest = max(len(row) for row in rows)
        return torch.tensor([row + [END] * (longest - len(row)) for row in rows])


class Seq2SeqTrainer:
    """Goes through the steps like the real trainer, with the stand-in model.

    `dropped` says how many words the model leaves out of every validation clip at a step, which fixes the
    scores of the run; `stop_after` makes the run fail after that step, like a session that is cut off.
    """

    dropped = {}
    stop_after = None
    instances = []

    def __init__(self, *, model, args, train_dataset, eval_dataset, data_collator, compute_metrics, processing_class,
                 callbacks):
        assert isinstance(processing_class, WhisperFeatureExtractor)
        self.instances.append(self)
        self.model, self.args, self.train_dataset, self.eval_dataset = model, args, train_dataset, eval_dataset
        self.data_collator, self.compute_metrics = data_collator, compute_metrics
        self.callbacks = [PrinterCallback(), *callbacks]
        self.state = SimpleNamespace(global_step=0, max_steps=args.max_steps, log_history=[],
                                     best_model_checkpoint=None, best_metric=None)
        self.model.config.use_cache = False  # as the real trainer does
        self.batches = []

    def remove_callback(self, callback):
        self.callbacks = [item for item in self.callbacks if not isinstance(item, callback)]

    def _prepare_inputs(self, inputs):
        return inputs

    def compute_loss_context_manager(self):
        return contextlib.nullcontext()

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **gen_kwargs):
        raise AssertionError("the prediction step of the library would hand the labels to generate")

    def log(self, logs):
        self.state.log_history.append({**logs, "step": self.state.global_step})
        for callback in self.callbacks:
            callback.on_log(self.args, self.state, None, logs=logs)

    def evaluate(self):
        step, size = self.state.global_step, self.args.per_device_eval_batch_size
        self.model.dropped_words = self.dropped[step]
        losses, rows, labels = [], [], []
        for first in range(0, len(self.eval_dataset), size):
            batch = self.data_collator([self.eval_dataset[index]
                                        for index in range(first, min(first + size, len(self.eval_dataset)))])
            loss, tokens, batch_labels = self.prediction_step(self.model, batch, False)
            losses.append(float(loss))
            rows += tokens.tolist()
            labels += batch_labels.tolist()
        longest = max(len(row) for row in rows)  # batches of different width are filled with -100
        predictions = np.array([row + [-100] * (longest - len(row)) for row in rows])
        metrics = self.compute_metrics(SimpleNamespace(predictions=predictions, label_ids=labels))
        logs = {"eval_loss": sum(losses) / len(losses), **{f"eval_{name}": value for name, value in metrics.items()},
                "eval_runtime": 0.5, "epoch": step / 10}
        self.log(logs)
        return logs

    def save(self, logs):
        folder = Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
        if self.state.best_metric is None or logs["eval_" + self.args.metric_for_best_model] < self.state.best_metric:
            self.state.best_metric = logs["eval_" + self.args.metric_for_best_model]
            self.state.best_model_checkpoint = str(folder)
        self.model.weights = f"weights at step {self.state.global_step}".encode()
        self.model.save_pretrained(folder)
        (folder / "trainer_state.json").write_text(json.dumps(vars(self.state)), encoding="utf-8")
        kept = {str(folder), self.state.best_model_checkpoint}
        for other in Path(self.args.output_dir).glob("checkpoint-*"):
            if str(other) not in kept:  # save_total_limit is 2: the newest and the best stay
                shutil.rmtree(other)

    def train(self, resume_from_checkpoint=None):
        args = self.args
        if resume_from_checkpoint:
            last = max(Path(args.output_dir).glob("checkpoint-*"), key=lambda path: int(path.name.split("-")[1]))
            self.state = SimpleNamespace(**json.loads((last / "trainer_state.json").read_text(encoding="utf-8")))
        elif args.eval_on_start:
            self.evaluate()
        clips = len(self.train_dataset)
        while self.state.global_step < args.max_steps:
            for callback in self.callbacks:
                callback.on_step_begin(args, self.state, None)
            for part in range(args.gradient_accumulation_steps):
                size = args.per_device_train_batch_size
                first = (self.state.global_step * args.gradient_accumulation_steps + part) * size
                batch = self.data_collator([self.train_dataset[(first + offset) % clips] for offset in range(size)])
                self.batches.append((tuple(batch["input_features"].shape), tuple(batch["labels"].shape)))
            self.state.global_step += 1
            for callback in self.callbacks:
                callback.on_step_end(args, self.state, None)
            if self.state.global_step % args.logging_steps == 0:
                self.log({"loss": 2 / self.state.global_step, "grad_norm": 1.5, "learning_rate": args.learning_rate,
                          "epoch": self.state.global_step / 10})
            if self.state.global_step % args.eval_steps == 0:
                self.save(self.evaluate())
            if self.state.global_step == self.stop_after:
                raise KeyboardInterrupt("the session was cut off")
        self.log({"train_runtime": 1.0, "train_loss": 0.5, "epoch": self.state.global_step / 10})
