from __future__ import annotations

from statistics import mean
from typing import Iterable, List

from .state import IterationResult, PROXY_NAMES


def reflect(history: Iterable[IterationResult], window: int) -> str:
    items: List[IterationResult] = list(history)[-window:]
    if not items:
        return (
            "Initial setup: explore benchmarks safely while respecting proxies of"
            " sentience-like behavior."
        )
    mean_proxies = {
        name: mean(getattr(item.report, name) for item in items) for name in PROXY_NAMES
    }
    passed_ratio = sum(1 for item in items if item.benchmarks_passed) / len(items)
    trending = max(mean_proxies, key=mean_proxies.get)
    return (
        f"Reflection: benchmarks success rate={passed_ratio:.2f}; focus on reinforcing {trending}. "
        "Next: adjust increments conservatively and keep artifacts contained in"
        " .echo_lab/."
    )
