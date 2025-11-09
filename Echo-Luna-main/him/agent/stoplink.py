from __future__ import annotations

"""Soft STOP handling for CognitiveEngine++."""

import json
from pathlib import Path
from threading import RLock
from typing import Optional


class StopLink:
    """Track external or local STOP states with a tiny IPC shim."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or Path(".guardian_stop")
        self._soft_stop = False
        self._lock = RLock()

    def set_soft_stop(self, enabled: bool) -> None:
        with self._lock:
            self._soft_stop = bool(enabled)

    def toggle(self) -> bool:
        with self._lock:
            self._soft_stop = not self._soft_stop
            return self._soft_stop

    def active(self) -> bool:
        with self._lock:
            if self._soft_stop:
                return True
            if self.path.exists():
                try:
                    payload = json.loads(self.path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    return True
                return bool(payload.get("stop", False))
            return False


__all__ = ["StopLink"]

