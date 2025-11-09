
from __future__ import annotations

import random
import time
from typing import Dict, Optional

from .config import SimulationConfig
from .report import PROXY_FIELDS, SentienceProxyReport

STOP_EXPERIMENT_COMPLETE = "EXPERIMENT COMPLETE"
STOP_CAP_REACHED = "CAP REACHED"
STOP_HALTED = "HALTED"


def generate_proxy_deltas() -> Dict[str, float]:
    return {field: random.uniform(0.015, 0.06) for field in PROXY_FIELDS}


def simulate_benchmarks(report: SentienceProxyReport) -> bool:
    base_prob = report.mean_proxy()
    adjusted = min(0.99, max(0.05, base_prob ** 1.2))
    return random.random() < adjusted


def achieve_sentience_placeholder(report: SentienceProxyReport, benchmarks_passed: bool) -> bool:
    if not benchmarks_passed:
        return False
    return report.mean_proxy() >= 0.95 and all(
        getattr(report, field) >= 0.95 for field in PROXY_FIELDS
    )


def check_stop_conditions(
    config: SimulationConfig,
    report: SentienceProxyReport,
    benchmarks_passed: bool,
    iteration: int,
    start_time: float,
) -> Optional[str]:
    elapsed = time.time() - start_time
    if achieve_sentience_placeholder(report, benchmarks_passed):
        return STOP_EXPERIMENT_COMPLETE
    if iteration >= config.max_iterations or elapsed >= config.max_seconds:
        return STOP_CAP_REACHED
    if config.esc_triggered(iteration, elapsed):
        return STOP_HALTED
    return None
