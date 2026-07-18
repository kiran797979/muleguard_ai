"""
Shared pytest fixtures/helpers for MuleGuard tests.

The pipeline stages are named `01_clean.py`, `02_features.py`, ... which are not
valid Python identifiers, so they cannot be imported with a normal `import`.
We load them by file path via importlib instead.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))   # so the stage modules can `import config`


def load_stage(filename: str):
    """Import a pipeline stage module by filename (handles numeric-prefixed names)."""
    path = SRC / filename
    mod_name = "mg_" + path.stem.replace(".", "_")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
