from __future__ import annotations

"""Heuristic policy module for CognitiveEngine++ agent."""

from echolab.state import PROXY_NAMES

from .metrics import SentienceReport


def plan(goal: str, report: SentienceReport, *, iteration: int) -> str:
    """Produce a tiny heuristic action string for the current goal."""

    proxy_snapshot = {name: getattr(report, name) for name in PROXY_NAMES}
    dominant = max(proxy_snapshot, key=proxy_snapshot.get)
    return f"{dominant}-focused reflection for goal '{goal}' at iteration {iteration}"  # noqa: E501


def attempt_goal(report: SentienceReport, passed: bool) -> bool:
    """Return True when benchmarks and proxies support declaring success."""

    mean_value = report.mean()
    return passed and mean_value >= 0.97


__all__ = ["plan", "attempt_goal"]

