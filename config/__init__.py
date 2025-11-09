"""Proxy `config` package that points at AIWaifu's configuration module."""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1] / "AIWaifu" / "config"
__path__ = [str(_ROOT)]
