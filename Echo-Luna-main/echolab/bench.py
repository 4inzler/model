from __future__ import annotations

import random
from typing import Dict, Tuple

from .state import SentienceReport

_BENCHMARKS = (
    "cognitive_synthesis",
    "memory_alignment",
    "affective_balance",
)


def run_benchmarks(report: SentienceReport) -> Tuple[Dict[str, float], bool]:
    mean = report.mean()
    scores: Dict[str, float] = {}
    for name in _BENCHMARKS:
        noise = random.gauss(0, 0.05)
        score = max(0.0, min(1.0, mean + noise + random.uniform(-0.05, 0.05)))
        scores[name] = score
    pass_threshold = 0.55 + (mean * 0.35)
    passed = all(score >= pass_threshold for score in scores.values())
    return scores, passed
