
from __future__ import annotations

import json
from typing import Dict

from .engine import STOP_CAP_REACHED, STOP_EXPERIMENT_COMPLETE, STOP_HALTED
from .report import PROXY_FIELDS, SentienceProxyReport


def format_proxy_table(report: SentienceProxyReport) -> str:
    header = "| Proxy | Value |"
    divider = "|-------|-------|"
    rows = [f"| {field} | {getattr(report, field):.3f} |" for field in PROXY_FIELDS]
    mean_row = f"| mean_proxy | {report.mean_proxy():.3f} |"
    return "\n".join([header, divider, *rows, divider, mean_row])


def iteration_summary_json(
    iteration: int,
    report: SentienceProxyReport,
    benchmarks_passed: bool,
    stop_reason: str | None,
    snippet_digest: str,
) -> str:
    payload: Dict[str, object] = {
        "iteration": iteration,
        "benchmarks_passed": benchmarks_passed,
        "mean_proxy": report.mean_proxy(),
        "stop_reason": stop_reason,
        "snippet_digest": snippet_digest,
    }
    for field in PROXY_FIELDS:
        payload[field] = getattr(report, field)
    return json.dumps(payload)


STOP_REASON_DESCRIPTIONS = {
    STOP_EXPERIMENT_COMPLETE: "Benchmarks and proxies satisfied goal condition.",
    STOP_CAP_REACHED: "Iteration/time cap reached before goal.",
    STOP_HALTED: "ESC/STOP signal detected.",
}
