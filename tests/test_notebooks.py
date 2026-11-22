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
BASELINES = FOLDER / "baselines_pretrained.ipynb"
# Notebooks never run the test split, with one documented exception: the pretrained baselines. They are
# published checkpoints with no setting chosen by us, run once each after the test evaluation of the project
# (22 November 2026). The notebook says so in its first cell, and the tests below hold it to that.
FINAL_TEST_NOTEBOOKS = {BASELINES.name}
PRETRAINED = ("wav2vec2-xlsr-53-hungarian", "canary-1b-v2", "parakeet-tdt-0.6b-v3", "qwen3-asr-1.7b")


def read(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def code_of(notebook: dict) -> str:
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")


def text_of(notebook: dict) -> str:
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown")


def sections_of(notebook: dict) -> dict[str, str]:
    """The code of a notebook by section: a markdown cell that begins with `## ` opens a section."""
    sections, title = {}, ""
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "markdown" and source.startswith("## "):
            title = source.split("\n", 1)[0]
        elif cell["cell_type"] == "code":
            sections[title] = sections.get(title, "") + source + "\n"
    return sections


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

    def test_only_the_documented_notebook_runs_the_test_split(self):
        self.assertEqual(FINAL_TEST_NOTEBOOKS, {"baselines_pretrained.ipynb"})
        for path in sorted(FOLDER.glob("*.ipynb")):
            if path.name not in FINAL_TEST_NOTEBOOKS:
                with self.subTest(path.name):
                    code = code_of(read(path))
                    self.assertNotIn("--final-test", code)
                    self.assertNotIn("--split", code)


class PretrainedBaselinesNotebookTests(unittest.TestCase):
    """The one notebook that runs the test split: once per system, as a final evaluation, and it says why."""

    @classmethod
    def setUpClass(cls):
        cls.notebook = read(BASELINES)
        cls.code, cls.text = code_of(cls.notebook), text_of(cls.notebook)
        cls.sections = sections_of(cls.notebook)
        cls.own = {system: next(code for title, code in cls.sections.items() if f"`{system}`" in title)
                   for system in PRETRAINED}

    def test_the_exception_is_stated_where_the_reader_begins(self):
        first = "".join(self.notebook["cells"][0]["source"])
        for statement in ("This notebook runs the test split, and it is the only notebook of the project that does",
                          "published checkpoints with nothing chosen by us",
                          "has already taken place", "refuses to replace an existing test run"):
            self.assertIn(statement, first)

    def test_every_run_of_the_test_split_is_a_final_evaluation_that_is_not_repeated(self):
        runs = re.findall(r"python -m huasr\.evaluate (.*)", self.code)
        on_test = [run for run in runs if "--split" in run]
        self.assertEqual(sorted(on_test),
                         sorted(f"--system {system} --split test --final-test" for system in PRETRAINED))
        self.assertEqual(sorted(run for run in runs if "--split" not in run),
                         sorted(f"--system {system} --force" for system in PRETRAINED))  # validation may be repeated
        self.assertEqual(self.code.count("--final-test"), len(PRETRAINED))

    def test_every_system_has_a_section_that_is_complete_in_itself(self):
        self.assertEqual(len(self.sections), 1 + len(PRETRAINED))  # the setup and one section per system
        for system, code in self.own.items():
            with self.subTest(system):
                steps = ["pip install -q -r requirements-pretrained.txt", f"huasr.pretrained fetch {system}",
                         f"huasr.evaluate --system {system} --force",
                         f"huasr.score results/baselines/{system}/validation",
                         f"huasr.evaluate --system {system} --split test --final-test",
                         f"huasr.score results/baselines/{system}/test",
                         f"zip -q -r {{ARCHIVE}} results/baselines/{system}", "files.download(ARCHIVE)"]
                order = [code.index(step) for step in steps]  # the validation split comes before the test split
                self.assertEqual(order, sorted(order))
                for other in PRETRAINED:
                    if other != system:
                        self.assertNotIn(other, code)
                self.assertEqual(code.count("stop_unless_done("), 5)  # a failed step is not built upon
                self.assertNotIn("results/own", code)                  # a baseline is never stored as our model

    def test_the_setup_is_shared_installs_no_model_package_and_needs_no_upload(self):
        setup = next(code for title, code in self.sections.items() if title.startswith("## 0."))
        for step in ('REPOSITORY = "https://github.com/kaankalkan03/NLP-project.git"',
                     "pip install -q -r requirements.txt\n", "pip install -q --no-deps -e .",
                     "huasr.subset unpack --raw data/raw/voxpopuli", "nvidia-smi",
                     "huasr.pretrained list", "def stop_unless_done(step):"):
            self.assertIn(step, setup)
        self.assertNotIn("requirements-pretrained.txt", setup)
        self.assertNotIn("huasr.evaluate", setup)
        self.assertNotIn("files.upload", self.code)
        self.assertEqual(self.notebook["metadata"]["accelerator"], "GPU")

    def test_the_notebook_says_that_no_system_needs_a_fresh_runtime(self):
        self.assertIn("Nothing conflicts", self.text)
        self.assertIn("no section needs a fresh runtime", self.text)
        for package in ("nemo_toolkit", "qwen-asr"):
            self.assertIn(package, self.text)      # named as not used
            self.assertNotIn(package, self.code)   # and never installed

    def test_everything_the_notebook_calls_exists(self):
        modules = set(re.findall(r"python -m (huasr\.\w+)", self.code)) | set(re.findall(r"\b(tests\.\w+)", self.code))
        self.assertEqual(modules, {"huasr.pretrained", "huasr.evaluate", "huasr.score", "huasr.subset",
                                   "tests.test_pretrained"})
        for module in modules:
            self.assertIsNotNone(importlib.util.find_spec(module), module)
        self.assertEqual(set(re.findall(r"--system (\S+)", self.code)), set(PRETRAINED))
        self.assertEqual(set(re.findall(r"pretrained fetch (\S+)", self.code)), set(PRETRAINED))
        for system in PRETRAINED:
            self.assertEqual(systems.SYSTEMS[system]().group, "baselines")
        for name in ("requirements.txt", "requirements-pretrained.txt"):
            self.assertTrue((REPOSITORY / name).is_file())

    def test_checkpoints_and_revisions_in_the_text_are_those_of_the_code(self):
        from huasr.pretrained import CLASSES

        for system, cls in CLASSES.items():
            title = next(title for title in self.sections if f"`{system}`" in title)
            cell = next("".join(cell["source"]) for cell in self.notebook["cells"]
                        if cell["cell_type"] == "markdown" and "".join(cell["source"]).startswith(title))
            self.assertIn(f"`{cls.checkpoint.repository}` at revision `{cls.checkpoint.revision}`", cell)


if __name__ == "__main__":
    unittest.main()
