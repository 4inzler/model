from __future__ import annotations

import random
from typing import Dict

from .state import PROXY_NAMES, SentienceReport


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, value))


def update_proxies(report: SentienceReport) -> SentienceReport:
    """Return a new report with stochastic yet bounded proxy increments."""
    updates: Dict[str, float] = {}
    base = report.mean()
    for name in PROXY_NAMES:
        current = getattr(report, name)
        drift = random.uniform(0.01, 0.06)
        influence = random.uniform(-0.015, 0.03) + base * 0.02
        updates[name] = _bounded(current + drift + influence)
    return report.update(updates)
