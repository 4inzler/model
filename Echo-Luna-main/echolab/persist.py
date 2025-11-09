from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

from .state import IterationResult

LAB_ROOT = Path(".echo_lab")
ARTIFACT_DIR = LAB_ROOT / "artifacts"
TRACE_DIR = LAB_ROOT / "traces"
MANIFEST_FILE = LAB_ROOT / "manifest.json"


@dataclass
class PersistContext:
    run_id: str
    enable_real_write: bool
    trace_path: Optional[Path] = None
    manifest_entries: List[str] = field(default_factory=list)

    def ensure_dirs(self) -> None:
        if not self.enable_real_write:
            return
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        TRACE_DIR.mkdir(parents=True, exist_ok=True)

    def manifest_path(self) -> Path:
        return MANIFEST_FILE


def _load_manifest() -> List[str]:
    if not MANIFEST_FILE.exists():
        return []
    try:
        data = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(item) for item in data]
    except json.JSONDecodeError:
        pass
    return []


def _save_manifest(entries: Iterable[str]) -> None:
    unique = sorted(set(entries))
    MANIFEST_FILE.write_text(json.dumps(unique, indent=2), encoding="utf-8")


def initialize(run_id: str, enable_real_write: bool) -> PersistContext:
    ctx = PersistContext(run_id=run_id, enable_real_write=enable_real_write)
    if enable_real_write:
        LAB_ROOT.mkdir(exist_ok=True)
        ctx.ensure_dirs()
        ctx.trace_path = TRACE_DIR / f"run_{ctx.run_id}.jsonl"
    return ctx


def write_artifact(ctx: PersistContext, snippet: str, iteration: int) -> Optional[Path]:
    if not ctx.enable_real_write:
        return None
    ctx.ensure_dirs()
    path = ARTIFACT_DIR / f"{ctx.run_id}_iter{iteration:04d}.py"
    path.write_text(snippet + "\n", encoding="utf-8")
    ctx.manifest_entries.append(str(path))
    return path


def append_trace(ctx: PersistContext, result: IterationResult) -> Optional[Path]:
    if not ctx.enable_real_write:
        return None
    ctx.ensure_dirs()
    if ctx.trace_path is None:
        ctx.trace_path = TRACE_DIR / f"run_{ctx.run_id}.jsonl"
    with ctx.trace_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(result.summary()) + "\n")
    ctx.manifest_entries.append(str(ctx.trace_path))
    return ctx.trace_path


def finalize(ctx: PersistContext) -> None:
    if not ctx.enable_real_write:
        return
    LAB_ROOT.mkdir(exist_ok=True)
    existing = _load_manifest()
    combined = existing + ctx.manifest_entries + [str(MANIFEST_FILE)]
    _save_manifest(combined)
