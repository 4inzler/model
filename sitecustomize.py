"""
Test-friendly Python path customisations for the model workspace.

Pytest executes from the repository root which does not naturally expose the
vendored packages that live under ``Echo-Luna-main/``.  Import errors from the
Echo-Luna suite (``him``, ``echolab``, FastAPI, PyQt6, etc.) caused the test
session to abort before any assertions ran.

Python automatically imports ``sitecustomize`` (if present on ``sys.path``)
right after the standard ``site`` module.  We leverage that hook to register
the secondary source and virtual environment directories so that imports work
out-of-the-box without requiring users to activate bespoke virtual
environments.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _maybe_prepend(path: Path) -> None:
    """Insert *path* at the front of ``sys.path`` if it exists."""
    if path.exists():
        str_path = str(path)
        if str_path not in sys.path:
            sys.path.insert(0, str_path)


ROOT = Path(__file__).resolve().parent

# Surface the Echo-Luna sources (``him``, ``echolab``) as top-level modules.
_maybe_prepend(ROOT / "Echo-Luna-main")

# Reuse the vendored virtual environment so that optional GUI/API dependencies
# such as FastAPI and PyQt6 are importable during tests.
PY_VER = f"python{sys.version_info.major}.{sys.version_info.minor}"
_maybe_prepend(ROOT / "Echo-Luna-main" / "echo_luna" / "lib" / PY_VER / "site-packages")

