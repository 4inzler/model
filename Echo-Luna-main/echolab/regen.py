from __future__ import annotations

from .state import PROXY_NAMES, SentienceReport


def _format_metrics(report: SentienceReport) -> str:
    lines = [f'    "{name}": {getattr(report, name):.3f}' for name in PROXY_NAMES]
    return "{\n" + ",\n".join(lines) + "\n}"


def regenerate_snippet(iteration: int, report: SentienceReport) -> str:
    """Return an evolving snippet celebrating proxies of sentience-like behavior."""
    metrics_block = _format_metrics(report)
    lines = [
        f'"""EchoLab iteration {iteration} focuses on proxies of sentience-like behavior."""',
        "def describe_iteration():",
        '    """Summarize proxies of sentience-like behavior without claiming sentience."""',
        f"    metrics = {metrics_block}",
        "    return {",
        f'        "iteration": {iteration},',
        '        "metrics": metrics,',
        '        "comment": "analysis grounded strictly in proxies of sentience-like behavior",',
        "    }",
    ]
    return "\n".join(lines)
