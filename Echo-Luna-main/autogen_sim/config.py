from __future__ import annotations

import typing as _t
from dataclasses import dataclass

EscSignalFn = _t.Callable[[int, float], bool]


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for the proxies-of-sentience-like-behavior simulation."""

    max_iterations: int = 120
    max_seconds: float = 45.0
    require_benchmarks: bool = True
    enable_real_writes: bool = False
    esc_signal_fn: EscSignalFn | None = None

    def esc_triggered(self, iteration: int, elapsed: float) -> bool:
        if self.esc_signal_fn is None:
            return False
        return self.esc_signal_fn(iteration, elapsed)
