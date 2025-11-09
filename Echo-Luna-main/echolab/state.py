from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Callable, Dict, List, Optional

PROXY_NAMES = (
    "memory_continuity",
    "self_model_consistency",
    "theory_of_mind",
    "metacognitive_calibration",
    "affect_stability",
    "integration_proxy",
)


@dataclass(frozen=True)
class SentienceReport:
    """Tracks proxies of sentience-like behavior for EchoLab."""

    memory_continuity: float = 0.0
    self_model_consistency: float = 0.0
    theory_of_mind: float = 0.0
    metacognitive_calibration: float = 0.0
    affect_stability: float = 0.0
    integration_proxy: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {name: getattr(self, name) for name in PROXY_NAMES}

    def update(self, updates: Dict[str, float]) -> "SentienceReport":
        payload = self.as_dict()
        for key, value in updates.items():
            if key in payload:
                payload[key] = max(0.0, min(1.0, value))
        return SentienceReport(**payload)

    def mean(self) -> float:
        values = self.as_dict().values()
        return sum(values) / len(PROXY_NAMES)


@dataclass
class IterationResult:
    iteration: int
    report: SentienceReport
    bench_scores: Dict[str, float]
    benchmarks_passed: bool
    snippet: str
    run_logs: List[Dict[str, object]] = field(default_factory=list)
    stop_reason: Optional[str] = None
    goal_achieved: bool = False

    def summary(self) -> Dict[str, object]:
        return {
            "iteration": self.iteration,
            "proxies": self.report.as_dict(),
            "bench_scores": self.bench_scores,
            "passed": self.benchmarks_passed,
            "snippet": self.snippet,
            "run_logs": self.run_logs,
            "goal": self.goal_achieved,
            "stop_reason": self.stop_reason,
        }


EscSignalFn = Callable[[int, float], bool]


@dataclass
class Config:
    max_iterations: int = 100
    max_seconds: float = 60.0
    require_benchmarks: bool = True
    enable_real_write: bool = False
    runner_timeout: float = 10.0
    reflection_window: int = 5
    esc_signal_fn: Optional[EscSignalFn] = None

    def esc_triggered(self, iteration: int, start_time: float) -> bool:
        if self.esc_signal_fn is None:
            return False
        elapsed = time() - start_time
        return self.esc_signal_fn(iteration, elapsed)
