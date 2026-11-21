"""Whisper small through the Hugging Face implementation (`transformers`): as published, and fine-tuned here.

    python -m huasr.whisper_hf fetch          download the pinned checkpoint into the Hugging Face cache (once)
    python -m huasr.whisper_hf environment    GPU, CUDA and package versions as a run would record them

Two systems behind the contract of `huasr.systems`, run like every other
system with `python -m huasr.evaluate --system <name>`:

    hf-whisper-small             group "baselines": `openai/whisper-small` at REVISION, used as published
    hf-whisper-small-finetuned   group "own": that checkpoint after `huasr.finetune`, read from a local folder

Both are the same code with the same decoding (DECODING), so the fine-tuned
model is compared with exactly the checkpoint it started from: language
Hungarian, task transcribe, no timestamps; beam size 5 and temperature 0 as
for the faster-whisper baselines; everything else as the generation config of
the checkpoint sets it. The model runs on the GPU if there is one, in 32-bit
floats; device and GPU name go into the run manifest.

Whisper reads at most 30 seconds at once, and without timestamps it cannot
move on through a longer recording by itself. A longer clip is therefore cut
into the fewest equal parts that each fit (`windows`); the parts are
transcribed one after the other and their texts are joined. A word at a cut
can come out wrong; every system of this module is cut at the same samples.

The published checkpoint is taken from the local Hugging Face cache only, at
the pinned revision, and its weight file must have the pinned SHA-256; `fetch`
is the one command that downloads it. The fine-tuned model is read from
FINETUNED_FOLDER, or from the folder named by the environment variable
HUASR_FINETUNED_DIR. Its weights are never stored in this repository; the
run manifest records their SHA-256 and the training record they came with.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

from . import LANGUAGE
from .systems import BASELINES, OWN, cached_checkpoint, sha256_of

REPOSITORY = "openai/whisper-small"
REVISION = "973afd24965f72e36ca33b3055d56a652f456b4d"
WEIGHTS = "model.safetensors"
WEIGHTS_BYTES = 966_995_080  # size and SHA-256 of the weight file at REVISION, as the model repository lists them
WEIGHTS_SHA256 = "1d7734884874f1a1513ed9aa760a4f8e97aaa02fd6d93a3a85d27b2ae9ca596b"
# What `fetch` downloads: the PyTorch weights and the files of the feature extractor and the tokenizer. The
# model repository also holds the same weights for other frameworks; they are not needed.
FILES = ("config.json", "generation_config.json", "preprocessor_config.json", "tokenizer.json",
         "tokenizer_config.json", "vocab.json", "merges.txt", "normalizer.json", "added_tokens.json",
         "special_tokens_map.json", WEIGHTS)
PRETRAINED, FINETUNED = "hf-whisper-small", "hf-whisper-small-finetuned"
FINETUNED_FOLDER = Path("scratch/finetune/hf-whisper-small/best")   # where huasr.finetune leaves its best model
FINETUNED_VARIABLE = "HUASR_FINETUNED_DIR"
RECORD = "training.json"                                            # written by huasr.finetune next to the weights
SAMPLE_RATE = 16_000
WINDOW_SECONDS = 30
DECODING = {"language": LANGUAGE, "task": "transcribe", "return_timestamps": False, "num_beams": 5,
            "temperature": 0.0}
PACKAGES = ("torch", "transformers", "tokenizers", "safetensors", "huggingface_hub")


def versions(names: tuple[str, ...] = PACKAGES) -> dict[str, str]:
    """Version of each package as the module that is really loaded reports it."""
    found = {}
    for name in names:
        try:
            found[name] = str(getattr(importlib.import_module(name), "__version__", "unknown"))
        except ImportError:
            found[name] = "not installed"
    return found


def accelerator() -> dict:
    """The device a model runs on here, and the GPU behind it."""
    import torch

    if not torch.cuda.is_available():
        return {"device": "cpu", "gpu": None}
    properties = torch.cuda.get_device_properties(0)
    return {"device": "cuda", "gpu": properties.name, "gpu_memory_bytes": properties.total_memory,
            "cuda": torch.version.cuda}


def windows(samples: np.ndarray, limit: int) -> list[np.ndarray]:
    """`samples` as one part if it has at most `limit` samples, else the fewest equal parts that each fit."""
    parts = max(1, math.ceil(len(samples) / limit))
    edges = [round(index * len(samples) / parts) for index in range(parts + 1)]
    return [samples[start:end] for start, end in zip(edges, edges[1:])]


def snapshot(download: bool = False) -> Path:
    """Folder of REPOSITORY at REVISION in the Hugging Face cache.

    Nothing is downloaded unless `download` is set (the `fetch` command); a
    checkpoint that is not in the cache is an error.
    """
    if not download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from huggingface_hub import snapshot_download

    try:
        folder = Path(snapshot_download(REPOSITORY, revision=REVISION, allow_patterns=list(FILES),
                                        local_files_only=not download))
    except Exception as error:  # the library has its own error types; offline, each of them means "not there"
        if download:
            raise
        raise FileNotFoundError(f"{REPOSITORY} at {REVISION} is not in the local Hugging Face cache; "
                                f"run python -m huasr.whisper_hf fetch") from error
    missing = [name for name in FILES if not (folder / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{folder} lacks {', '.join(missing)}; run python -m huasr.whisper_hf fetch")
    return folder


def published_weights(folder: Path) -> dict:
    """Size and SHA-256 of the weight file of the published checkpoint; an error unless they are the pinned ones."""
    weights = Path(folder) / WEIGHTS
    found = {"weights_file": WEIGHTS, "weights_bytes": weights.stat().st_size, "weights_sha256": sha256_of(weights)}
    if (found["weights_bytes"], found["weights_sha256"]) != (WEIGHTS_BYTES, WEIGHTS_SHA256):
        raise ValueError(f"{weights} is not the weight file of {REPOSITORY} at {REVISION}: "
                         f"{found['weights_bytes']} bytes, SHA-256 {found['weights_sha256']}")
    return found


class HuggingFaceWhisper:
    """A Whisper checkpoint run with `transformers`: the published one (`folder` is None) or a local folder."""

    decoding = DECODING

    def __init__(self, name: str, group: str, folder: Path | None = None, seed: int = 0):
        self.name, self.group, self.seed = name, group, seed
        self.fine_tuned = folder is not None
        self._folder = None if folder is None else Path(folder)
        self._model = self._feature_extractor = self._tokenizer = self._device = None

    def folder(self) -> Path:
        """Folder the model is loaded from; fails if the checkpoint is not there."""
        if self._folder is None:
            self._folder = snapshot()
        if self.fine_tuned and not all((self._folder / name).is_file() for name in (WEIGHTS, RECORD)):
            raise FileNotFoundError(f"no fine-tuned model in {self._folder}: run python -m huasr.finetune train "
                                    f"first, or name its folder in {FINETUNED_VARIABLE}")
        return self._folder

    def checkpoint(self) -> dict:
        """What the run manifest says about the weights that are loaded."""
        folder = self.folder()
        if not self.fine_tuned:
            return {**cached_checkpoint(folder), **published_weights(folder)}
        record = json.loads((folder / RECORD).read_text(encoding="utf-8"))
        base, best, steps = record["base_checkpoint"], record["best"], record["hyperparameters"]["max_steps"]
        trial = "" if record["complete"] else "; trial run on a part of the data"
        return {
            "description": f"`{base['repository']}` at `{base['revision'][:12]}`, fine-tuned on the training split "
                           f"(step {best['step']} of {steps}{trial})",
            "base": {"repository": base["repository"], "revision": base["revision"]},
            "folder": folder.as_posix(), "weights_file": WEIGHTS, "weights_bytes": (folder / WEIGHTS).stat().st_size,
            "weights_sha256": sha256_of(folder / WEIGHTS), "training_record": RECORD,
            "training_record_sha256": sha256_of(folder / RECORD), "training_complete": record["complete"],
            "best_step": best["step"], "selection_wer": best["wer"],
        }

    def describe(self) -> dict:
        return {
            "checkpoint": self.checkpoint(),
            "settings": {"implementation": "transformers WhisperForConditionalGeneration", **accelerator(),
                         "dtype": "float32", **self.decoding, "window_seconds": WINDOW_SECONDS,
                         "longer_audio": "cut into the fewest equal parts of at most one window; texts joined",
                         "other_options": "generation config of the checkpoint", "fine_tuned": self.fine_tuned},
            "seed": self.seed,
            "packages": versions(),
        }

    def load(self) -> None:
        if self._model is not None:
            return
        folder = str(self.folder())
        import torch
        from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizer

        torch.manual_seed(self.seed)
        self._device = accelerator()["device"]
        self._feature_extractor = WhisperFeatureExtractor.from_pretrained(folder, local_files_only=True)
        self._tokenizer = WhisperTokenizer.from_pretrained(folder, local_files_only=True)
        model = WhisperForConditionalGeneration.from_pretrained(folder, local_files_only=True)
        self._model = model.to(self._device).eval()

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"Whisper expects audio at 16 kHz, got {sample_rate} Hz")
        self.load()
        import torch

        texts = []
        for part in windows(samples, WINDOW_SECONDS * SAMPLE_RATE):
            features = self._feature_extractor(part, sampling_rate=SAMPLE_RATE, return_tensors="pt")["input_features"]
            with torch.no_grad():
                tokens = self._model.generate(features.to(self._device), **self.decoding)
            texts.append(self._tokenizer.batch_decode(tokens, skip_special_tokens=True)[0].strip())
        return " ".join(text for text in texts if text)


def build(name: str) -> HuggingFaceWhisper:
    """The system registered as `name` in `huasr.systems.SYSTEMS`."""
    if name == PRETRAINED:
        return HuggingFaceWhisper(PRETRAINED, BASELINES)
    if name == FINETUNED:
        return HuggingFaceWhisper(FINETUNED, OWN, Path(os.environ.get(FINETUNED_VARIABLE) or FINETUNED_FOLDER))
    raise ValueError(f"unknown system {name!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("fetch", help=f"download {REPOSITORY} at the pinned revision into the Hugging Face cache")
    commands.add_parser("environment", help="print the device, the GPU and the package versions")
    args = parser.parse_args(argv)
    try:
        if args.command == "fetch":
            folder = snapshot(download=True)
            weights = published_weights(folder)
            print(f"{REPOSITORY} at {REVISION}\n{folder}\n{WEIGHTS}: {weights['weights_bytes']:,} bytes, "
                  f"SHA-256 {weights['weights_sha256']}")
        else:
            print(json.dumps({**accelerator(), "packages": versions()}, indent=2))
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
