from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class RunLog:
    command: List[str]
    returncode: int
    stdout: str
    stderr: str
    duration: float
    timeout: bool


def _sanitize_command(command: Sequence[str]) -> List[str]:
    if not command:
        raise ValueError("command must not be empty")
    return list(command)


def _safe_environment() -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.lower() in {"http_proxy", "https_proxy", "no_proxy"}:
            env.pop(key, None)
    env.setdefault("PYTHONWARNINGS", "ignore::ResourceWarning")
    return env


def run_command(command: Sequence[str], timeout: float = 10.0) -> RunLog:
    cmd = _sanitize_command(command)
    start = time.time()
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=_safe_environment(),
        )
        duration = time.time() - start
        return RunLog(
            command=cmd,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration=duration,
            timeout=False,
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.time() - start
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout or ""
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else exc.stderr or ""
        return RunLog(
            command=cmd,
            returncode=-1,
            stdout=stdout,
            stderr=f"{stderr}\n<timeout>",
            duration=duration,
            timeout=True,
        )


def run_python(code: str, timeout: float = 10.0) -> RunLog:
    return run_command([sys.executable, "-c", code], timeout=timeout)
