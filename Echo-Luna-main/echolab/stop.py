from __future__ import annotations

import time
from typing import Optional, Tuple

from .state import Config, SentienceReport

STOP_REASON_GOAL = "EXPERIMENT COMPLETE"
STOP_REASON_LIMIT = "CAP REACHED"
STOP_REASON_ESC = "HALTED"


def attempt_goal(report: SentienceReport, passed: bool) -> bool:
    return passed and report.mean() >= 0.97


def should_stop(
    config: Config,
    report: SentienceReport,
    passed: bool,
    iteration: int,
    start_time: float,
) -> Tuple[Optional[str], bool]:
    elapsed = time.time() - start_time
    proxies = report.as_dict()
    if config.require_benchmarks and passed and all(value >= 0.95 for value in proxies.values()):
        return STOP_REASON_GOAL, attempt_goal(report, passed)
    if iteration >= config.max_iterations or elapsed >= config.max_seconds:
        return STOP_REASON_LIMIT, False
    if config.esc_signal_fn and config.esc_triggered(iteration, start_time):
        return STOP_REASON_ESC, False
    return None, False
