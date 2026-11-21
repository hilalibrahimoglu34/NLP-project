"""Speech recognition systems behind one small contract.

The runner (`huasr.evaluate`) knows nothing about a system except this:

    name        how the system is called in results, e.g. "faster-whisper-small"
    group       "baselines" for checkpoints used as they are, "own" for models trained in this project
    describe()  what goes into the run manifest: checkpoint id and revision, settings, package versions
    transcribe(samples, sample_rate) -> text
                `samples` is a one-dimensional float32 array with values between -1 and 1

Every system gets the same clips in the same form; nothing about the scoring lives here.

Systems so far: Whisper base and small, zero-shot, run with faster-whisper on the CPU. faster-whisper
loads CTranslate2 conversions of the OpenAI checkpoints (`Systran/faster-whisper-<size>`); the model
cards of these repositories name the checkpoint each one was converted from (UPSTREAM). The models are
taken from the local Hugging Face cache only, at a pinned revision (REVISIONS); nothing is downloaded by
this module.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

BASELINES, OWN = "baselines", "own"
GROUPS = (BASELINES, OWN)
# Checkpoint that each CTranslate2 repository says it was converted from (its model card).
UPSTREAM = {"Systran/faster-whisper-base": "openai/whisper-base", "Systran/faster-whisper-small": "openai/whisper-small"}
# Revision of each converted repository that this project uses.
REVISIONS = {"base": "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66", "small": "536b0662742c02347bc0e980a01041f333bce120"}


def cached_checkpoint(snapshot: Path) -> dict:
    """Repository id and revision of a model folder in the Hugging Face cache.

    The cache stores a model as `models--<owner>--<name>/snapshots/<revision>/`.
    """
    parts = Path(snapshot).parts
    if "snapshots" not in parts[1:-1]:
        raise ValueError(f"{snapshot} is not a snapshot folder of the Hugging Face cache")
    index = parts.index("snapshots")
    folder = parts[index - 1]
    if not folder.startswith("models--"):
        raise ValueError(f"{snapshot} is not a snapshot folder of the Hugging Face cache")
    return {"repository": folder[len("models--"):].replace("--", "/"), "revision": parts[index + 1]}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


class FasterWhisper:
    """Whisper as published, without fine-tuning, run with faster-whisper (CTranslate2) on the CPU."""

    group = BASELINES
    # The number of threads is fixed because the output depends on it: the same clips gave slightly
    # different text with 4 and with 8 threads, and identical text when 4 threads were used twice.
    runtime = {"device": "cpu", "compute_type": "int8", "cpu_threads": 4}
    decoding = {"language": "hu", "task": "transcribe", "beam_size": 5, "temperature": 0.0,
                "condition_on_previous_text": False}

    def __init__(self, size: str, seed: int = 0):
        self.size, self.seed = size, seed
        self.name = f"faster-whisper-{size}"
        self._snapshot, self._model = None, None

    def snapshot(self) -> Path:
        """Folder of the cached model at the pinned revision; fails if it is not in the local cache."""
        if self._snapshot is None:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from faster_whisper.utils import download_model

            self._snapshot = Path(download_model(self.size, local_files_only=True, revision=REVISIONS[self.size]))
        return self._snapshot

    def describe(self) -> dict:
        import ctranslate2
        import faster_whisper

        checkpoint = cached_checkpoint(self.snapshot())
        model_file = self.snapshot() / "model.bin"
        return {
            "checkpoint": {**checkpoint, "converted_from": UPSTREAM.get(checkpoint["repository"], "not recorded"),
                           "model_bin_bytes": model_file.stat().st_size, "model_bin_sha256": sha256_of(model_file)},
            "settings": {**self.runtime, **self.decoding, "other_options": "faster-whisper defaults",
                         "fine_tuned": False},
            "seed": self.seed,
            "packages": {"faster-whisper": faster_whisper.__version__, "ctranslate2": ctranslate2.__version__},
        }

    def load(self) -> None:
        if self._model is None:
            import ctranslate2
            from faster_whisper import WhisperModel

            ctranslate2.set_random_seed(self.seed)
            self._model = WhisperModel(str(self.snapshot()), local_files_only=True, **self.runtime)

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        if sample_rate != 16_000:
            raise ValueError(f"Whisper expects audio at 16 kHz, got {sample_rate} Hz")
        self.load()
        segments, _ = self._model.transcribe(samples, **self.decoding)
        return " ".join(segment.text.strip() for segment in segments)  # the decoding happens while this is read


SYSTEMS = {
    "faster-whisper-base": lambda: FasterWhisper("base"),
    "faster-whisper-small": lambda: FasterWhisper("small"),
}


def _whisper_hf(name: str):
    """A system of `huasr.whisper_hf`: Whisper small through `transformers`, as published or fine-tuned here."""
    from . import whisper_hf  # imported when it is asked for: that module imports this one

    return whisper_hf.build(name)


SYSTEMS.update({"hf-whisper-small": lambda: _whisper_hf("hf-whisper-small"),
                "hf-whisper-small-finetuned": lambda: _whisper_hf("hf-whisper-small-finetuned")})
