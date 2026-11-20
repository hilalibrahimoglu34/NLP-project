"""Unit tests. Run from the repository root: python -m unittest

`src/` is put on the import path here, so the tests also run from a fresh
clone before the package is installed. Tests that need an optional package
are skipped with a message when it is missing.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parent.parent
SRC = str(REPOSITORY / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


def needs(module: str):
    """Skip a test or test class unless `module` can be imported."""
    found = importlib.util.find_spec(module) is not None
    return unittest.skipUnless(found, f"{module} is not installed (pip install -r requirements.txt)")
