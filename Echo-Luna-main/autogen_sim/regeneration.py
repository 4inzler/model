
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from textwrap import indent

from .report import PROXY_FIELDS, SentienceProxyReport

SNIPPET_HEADER = "# Auto-generated snippet (simulation only)"
TARGET_FILE = Path("_autogen_sample.py")


def regenerate_code_snippet(iteration: int, report: SentienceProxyReport) -> str:
    """Produce a playful snippet that echoes proxies of sentience-like behavior."""
    fields = ", ".join(f"{field}={getattr(report, field):.3f}" for field in PROXY_FIELDS)
    body = f"def synthetic_reflection():\n    return \"iteration={iteration}, {fields}\""
    return "\n".join([SNIPPET_HEADER, body])


def rewrite_all_files_dry(snippet: str) -> str:
    timestamp = _dt.datetime.utcnow().isoformat()
    preview = indent(snippet, prefix="    ")
    return (
        f"[{timestamp}] DRY-RUN: would rewrite folder with snippet preview:\n{preview}"
    )


def rewrite_all_files(snippet: str, enable_real_writes: bool) -> str:
    if not enable_real_writes:
        return rewrite_all_files_dry(snippet)
    TARGET_FILE.write_text(snippet + "\n", encoding="utf-8")
    timestamp = _dt.datetime.utcnow().isoformat()
    return f"[{timestamp}] WROTE: {TARGET_FILE.name} updated with regenerated snippet"
