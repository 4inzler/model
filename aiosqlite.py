"""Expose the local aiosqlite shim regardless of the active working directory."""

from AIWaifu import aiosqlite as _aiosqlite

__all__ = getattr(_aiosqlite, "__all__", [])

globals().update({name: getattr(_aiosqlite, name) for name in __all__})
