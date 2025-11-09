"""Proxy package exposing AIWaifu sources as `src.*`."""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1] / "AIWaifu" / "src"
__path__ = [str(_ROOT)]
