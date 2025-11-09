from __future__ import annotations

"""Metrics helpers focused on proxies of sentience-like behavior."""

import random
from echolab.state import PROXY_NAMES, SentienceReport


def new_report() -> SentienceReport:
    """Return a fresh SentienceReport with zeroed proxies."""

    return SentienceReport()


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def update(report: SentienceReport, *, noise: float = 0.05, rng: random.Random | None = None) -> SentienceReport:
    """Update the proxies of sentience-like behavior, not evidence of sentience.

    Each proxy nudges towards 1.0 with bounded stochasticity so that repeated
    calls converge smoothly while still providing some variance for testing.
    """

    rng = rng or random
    updates = {}
    baseline = report.mean()
    for name in PROXY_NAMES:
        current = getattr(report, name)
        drift = rng.uniform(0.01, noise)
        attraction = (1.0 - current) * rng.uniform(0.05, 0.15)
        perturb = rng.uniform(-noise, noise) * 0.5
        updates[name] = _clamp(current + drift + attraction + perturb + baseline * 0.03)
    return report.update(updates)


def mean_proxy(report: SentienceReport) -> float:
    return report.mean()


__all__ = ["new_report", "update", "mean_proxy", "SentienceReport", "PROXY_NAMES"]

