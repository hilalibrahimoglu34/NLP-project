"""Tests for the notebooks as files: well-formed, stored without output, and calling only what exists.

The notebooks need a GPU and are not run here.
"""

import importlib.util
import json
import re
import unittest

from huasr import systems
from tests import REPOSITORY

FOLDER = REPOSITORY / "notebooks"
FINE_TUNING = FOLDER / "finetune_whisper_small.ipynb"


def read(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def code_of(notebook: dict) -> str:
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")


class NotebookTests(unittest.TestCase):
    def test_notebooks_are_well_formed_and_stored_without_output(self):
        paths = sorted(FOLDER.glob("*.ipynb"))
        self.assertIn(FINE_TUNING, paths)
        for path in paths:
            with self.subTest(path.name):
                self.assertNotIn(b"\r", path.read_bytes())
                notebook = read(path)
                self.assertEqual((notebook["nbformat"], notebook["nbformat_minor"]), (4, 5))
                self.assertEqual(notebook["metadata"]["kernelspec"]["language"], "python")
                identifiers = [cell["id"] for cell in notebook["cells"]]
                self.assertEqual(len(set(identifiers)), len(identifiers))
                for cell in notebook["cells"]:
                    self.assertRegex(cell["id"], r"^[a-zA-Z0-9_-]{1,64}$")
                    self.assertIn(cell["cell_type"], ("markdown", "code"))
                    self.assertTrue(all(isinstance(line, str) for line in cell["source"]) and cell["source"])
                    if cell["cell_type"] == "code":
                        self.assertEqual((cell["outputs"], cell["execution_count"]), ([], None))
                        self.assertTrue(cell["source"][0].startswith("# "), "every code cell begins with a comment")
                        self.assertLessEqual(len(cell["source"]), 12, "cells stay short")

    def test_the_fine_tuning_notebook_never_runs_the_test_split(self):
        code = code_of(read(FINE_TUNING))
        self.assertNotIn("--final-test", code)
        self.assertNotIn("--split", code)  # the runner then takes the validation split
        self.assertNotRegex(code, r"/test\b")
        runs = re.findall(r"python -m huasr\.evaluate (.*)", code)
        self.assertEqual(runs, ["--system hf-whisper-small --force", "--system hf-whisper-small-finetuned --force"])

    def test_the_zero_shot_baseline_is_measured_before_the_fine_tuning(self):
        code = code_of(read(FINE_TUNING))
        order = [code.index(command) for command in (
            "huasr.subset unpack", "huasr.whisper_hf fetch", "huasr.finetune train --smoke",
            "huasr.evaluate --system hf-whisper-small --force", "huasr.finetune train --resume",
            "huasr.evaluate --system hf-whisper-small-finetuned", "huasr.compare", "huasr.report")]
        self.assertEqual(order, sorted(order))

    def test_it_clones_the_public_repository_and_needs_no_upload(self):
        notebook = read(FINE_TUNING)
        code = code_of(notebook)
        self.assertIn('REPOSITORY = "https://github.com/kaankalkan03/NLP-project.git"', code)
        self.assertIn("pip install -q -r requirements.txt -r requirements-train.txt", code)
        self.assertNotIn("files.upload", code)
        self.assertEqual(notebook["metadata"]["accelerator"], "GPU")
        self.assertRegex(code, r"SAVE_TO_DRIVE = False\nif SAVE_TO_DRIVE:")  # weights leave Colab only on request

    def test_everything_the_notebook_calls_exists(self):
        code = code_of(read(FINE_TUNING))
        modules = set(re.findall(r"python -m (huasr\.\w+)", code)) | set(re.findall(r"\b(tests\.\w+)", code))
        self.assertGreaterEqual(modules, {"huasr.finetune", "huasr.whisper_hf", "huasr.evaluate", "huasr.score",
                                          "huasr.compare", "huasr.report", "huasr.subset", "tests.test_finetune"})
        for module in modules:
            self.assertIsNotNone(importlib.util.find_spec(module), module)
        for system in re.findall(r"--system (\S+)", code):
            self.assertIn(system, systems.SYSTEMS)
        for name in ("requirements.txt", "requirements-train.txt"):
            self.assertTrue((REPOSITORY / name).is_file())
        folders = set(re.findall(r"results/(?:baselines|own)/[\w-]+", code))
        self.assertEqual(folders, {"results/baselines/hf-whisper-small", "results/own/hf-whisper-small-finetuned"})


if __name__ == "__main__":
    unittest.main()
