import asyncio
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


_WHISPER = None


def _has_cuda() -> bool:
    try:
        import torch  # type: ignore

        return torch.cuda.is_available()
    except Exception:
        return False


def _ensure_whisper():
    global _WHISPER
    if _WHISPER is not None:
        return _WHISPER
    try:
        from faster_whisper import WhisperModel  # type: ignore

        model_name = os.getenv("STT_MODEL", "base.en")
        device = "cuda" if _has_cuda() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        _WHISPER = WhisperModel(model_name, device=device, compute_type=compute_type)
    except Exception:
        _WHISPER = None
    return _WHISPER


async def transcribe_file(path: str) -> Optional[str]:
    model = _ensure_whisper()
    if not model:
        return None
    try:
        # Run in thread to avoid blocking loop
        def _run() -> str:
            segments, _ = model.transcribe(path)
            return " ".join(seg.text.strip() for seg in segments if seg.text)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _run)
    except Exception:
        return None


async def tts_piper_to_wav(text: str, *, out_dir: Optional[str] = None) -> Optional[str]:
    """Generate WAV using piper CLI. Requires PIPER_MODEL and optional PIPER_BINARY."""
    model = os.getenv("PIPER_MODEL")
    if not model:
        return None
    binary = os.getenv("PIPER_BINARY", "piper")
    if not shutil.which(binary):
        # Try module-provided binary if installed as piper-tts
        alt = shutil.which("piper")
        if alt is None:
            return None
        binary = alt

    out_base = Path(out_dir or tempfile.gettempdir())
    out_path = str(out_base / f"piper_out_{os.getpid()}_{abs(hash(text)) % 10_000_000}.wav")

    # Piper needs text on stdin
    try:
        p = await asyncio.create_subprocess_exec(
            binary,
            "--model",
            model,
            "--output_file",
            out_path,
            stdout=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert p.stdin is not None
        p.stdin.write(text.encode("utf-8"))
        await p.stdin.drain()
        p.stdin.close()
        await p.wait()
        if Path(out_path).exists():
            return out_path
    except Exception:
        return None
    return None

