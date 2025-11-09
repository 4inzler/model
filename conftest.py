"""Pytest configuration that exposes vendored dependencies on sys.path."""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any


def _maybe_insert(path: Path) -> None:
    if path.exists():
        str_path = str(path)
        if str_path not in sys.path:
            sys.path.insert(0, str_path)


ROOT = Path(__file__).resolve().parent
ECHO_ROOT = ROOT / "Echo-Luna-main"
_maybe_insert(ECHO_ROOT)

PY_VER = f"python{sys.version_info.major}.{sys.version_info.minor}"
_maybe_insert(ECHO_ROOT / "echo_luna" / "lib" / PY_VER / "site-packages")


def _missing(module: str) -> bool:
    return importlib.util.find_spec(module) is None


def pytest_ignore_collect(collection_path: Path, config: Any):  # noqa: D401 - pytest hook
    """Skip optional integration suites when dependencies are unavailable."""

    path_obj = Path(str(collection_path))
    path_str = str(path_obj)
    if "Echo-Luna-main/tests" in path_str or "Echo-Luna-main/.echo_lab" in path_str:
        if _missing("fastapi") or _missing("PyQt6"):
            return True
    return False


def pytest_pyfunc_call(pyfuncitem):
    """Allow async tests without requiring pytest-asyncio."""
    if inspect.iscoroutinefunction(pyfuncitem.obj):
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(pyfuncitem.obj(**pyfuncitem.funcargs))
        finally:
            loop.close()
        return True
    return None
