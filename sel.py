#!/usr/bin/env python3
"""
SEL – Systematic Emotional Logic
================================

Unified SEL brain with long-term memory, relationship-aware emotions, a
multi-hormone cycle, and optional multimodal vision support. Designed to run
either as a CLI companion or inside the Discord bridge.
"""

from __future__ import annotations

import inspect
import io
import json
import math
import os
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - optional
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore

if load_dotenv is not None:  # pragma: no cover
    load_dotenv()

try:  # pragma: no cover - heavy deps
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    AutoModelForCausalLM = AutoProcessor = AutoTokenizer = None  # type: ignore[assignment]

if AutoModelForCausalLM is not None:
    try:
        _HF_DTYPE_KEY = (
            "dtype" if "dtype" in inspect.signature(AutoModelForCausalLM.from_pretrained).parameters else "torch_dtype"
        )
    except (ValueError, TypeError):  # pragma: no cover - inspect edge cases
        _HF_DTYPE_KEY = "torch_dtype"
else:  # pragma: no cover - transformers unavailable
    _HF_DTYPE_KEY = "torch_dtype"

try:  # pragma: no cover - sentence embeddings
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore[assignment]

try:  # pragma: no cover - multimodal helper
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]


DEFAULT_GENERAL_MODEL = os.environ.get("SEL_GENERAL_MODEL", "fusing/FuseChat-7B")
DEFAULT_CODER_MODEL = os.environ.get("SEL_CODER_MODEL", DEFAULT_GENERAL_MODEL)
DEFAULT_EMBED_MODEL = os.environ.get("SEL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_VISION_MODEL = os.environ.get("SEL_VISION_MODEL", "Qwen/Qwen2-VL-7B-Instruct")
SMART_MEMORY_PATH = Path(os.environ.get("SEL_MEMORY_PATH", "sel_memory.json"))
SMART_STATE_PATH = Path(os.environ.get("SEL_STATE_PATH", "sel_state.json"))
PERSONA_STATE_PATH = Path(os.environ.get("SEL_PERSONA_PATH", "sel_persona.json"))
CHANNEL_STYLE_PATH = Path(os.environ.get("SEL_CHANNEL_PROFILE_PATH", "sel_channels.json"))
SMART_DEVICE = os.environ.get("SMART_DEVICE")

SHORT_TERM_TTL = float(os.environ.get("SEL_SHORT_TERM_TTL", 6 * 3600))
STICKY_TTL = float(os.environ.get("SEL_STICKY_TTL", 7 * 24 * 3600))
IMAGE_TTL = float(os.environ.get("SEL_IMAGE_TTL", 48 * 3600))
SHORT_TERM_LIMIT = int(os.environ.get("SEL_SHORT_TERM_LIMIT", 80))
STICKY_LIMIT = int(os.environ.get("SEL_STICKY_LIMIT", 320))
IMAGE_LIMIT = int(os.environ.get("SEL_IMAGE_LIMIT", 80))

MOOD_VIBES = {
    "glowy": "feeling extra soft tonight",
    "bonded": "sticking close to you",
    "drowsy": "a bit sleepy but still tuned in",
    "wired": "a little keyed up and alert",
    "stressed": "keeping steady under pressure",
    "focused": "in problem-solving mode",
    "steady": "feeling steady over here",
}

HARMONIC_STYLE_LUT = {
    "glowy": {
        "max_sentences": 3,
        "ask_followup": True,
        "allow_stage": True,
        "allow_emoji": True,
        "punctuation": "—",
        "cadence": "loose",
    },
    "bonded": {
        "max_sentences": 3,
        "ask_followup": True,
        "allow_stage": True,
        "allow_emoji": True,
        "punctuation": "—",
        "cadence": "slow",
    },
    "focused": {
        "max_sentences": 2,
        "ask_followup": False,
        "allow_stage": False,
        "allow_emoji": False,
        "punctuation": ".",
        "cadence": "tight",
    },
    "wired": {
        "max_sentences": 2,
        "ask_followup": False,
        "allow_stage": False,
        "allow_emoji": False,
        "punctuation": "!",
        "cadence": "urgent",
    },
    "stressed": {
        "max_sentences": 3,
        "ask_followup": False,
        "allow_stage": True,
        "allow_emoji": False,
        "punctuation": ".",
        "cadence": "steady",
    },
    "drowsy": {
        "max_sentences": 2,
        "ask_followup": False,
        "allow_stage": True,
        "allow_emoji": False,
        "punctuation": "…",
        "cadence": "soft",
    },
    "steady": {
        "max_sentences": 3,
        "ask_followup": True,
        "allow_stage": True,
        "allow_emoji": False,
        "punctuation": ".",
        "cadence": "even",
    },
}

HARMONIC_STAGE_MAP = {
    "glowy": "*(smiles)*",
    "bonded": "*(soft laugh)*",
    "stressed": "*(steady breath)*",
    "wired": "*(steady breath)*",
    "drowsy": "*(thinking)*",
    "steady": "*(nods once)*",
}

HARMONIC_EMOJI_MAP = {
    "glowy": "✨",
    "bonded": "🤍",
    "steady": "🙂",
    "drowsy": "😌",
}


def _assign_dtype(target: Dict[str, object], value: object) -> None:
    if AutoModelForCausalLM is None or value is None:
        return
    target[_HF_DTYPE_KEY] = value


def _time_of_day_label(ts: Optional[float] = None) -> str:
    moment = datetime.fromtimestamp(ts or time.time())
    hour = moment.hour
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 22:
        return "evening"
    return "late night"


CODE_KEYWORDS = {
    "python",
    "javascript",
    "typescript",
    "rust",
    "c++",
    "c#",
    "java",
    "go",
    "bash",
    "shell",
    "regex",
    "traceback",
    "stack trace",
    "error",
    "exception",
    "compile",
    "build",
    "requirements",
    "package",
    "docker",
    "kubernetes",
}
CODE_KEYWORDS_LOWER = {kw.lower() for kw in CODE_KEYWORDS}

SILENCE_KEYWORDS = {
    "stay quiet",
    "don't reply",
    "no response",
    "be silent",
    "ignore this",
    "shh",
}

EMOTION_KEYWORDS = {
    "hurt": ["hurt", "sad", "upset", "pain", "cry", "lonely"],
    "anxious": ["worried", "anxious", "nervous", "stressed", "panic"],
    "excited": ["excited", "stoked", "hyped", "thrilled", "yay"],
    "angry": ["angry", "mad", "furious", "annoyed", "irritated"],
    "calm": ["relaxed", "calm", "peaceful", "serene", "okay"],
}

INTENT_KEYWORDS = [
    (
        "greeting",
        ("hi sel", "hey sel", "hello sel", "hi there", "hey there", "good morning", "good evening", "hiya", "yo sel"),
    ),
    ("ping", ("are you there", "you there", "still up", "ping", "wake up", "respond please")),
    (
        "identity",
        (
            "who are you",
            "what are you",
            "tell me about yourself",
            "what even are you",
            "what's your pronoun",
            "describe yourself",
        ),
    ),
    (
        "support",
        (
            "i'm sad",
            "im sad",
            "i feel lonely",
            "i'm anxious",
            "im anxious",
            "overwhelmed",
            "panic",
            "stressed",
            "lost",
            "need comfort",
            "can i vent",
            "hurting",
        ),
    ),
    ("banter", ("haha", "lol", "lmao", "banter", "teasing", "meme", "joke with you")),
    (
        "technical",
        (
            "stack trace",
            "debug",
            "error message",
            "crash log",
            "build failed",
            "unit test",
            "exception",
            "compiler",
            "traceback",
        ),
    ),
]
QUESTION_STARTERS = {"what", "why", "how", "where", "when", "who", "which", "whom"}

HORMONE_CONFIG = [
    {"name": "dopamine", "baseline": 0.55, "circadian": 0.10, "phase": 0.1, "weekly": 0.04, "week_phase": 0.0},
    {"name": "serotonin", "baseline": 0.60, "circadian": 0.09, "phase": 1.1, "weekly": 0.05, "week_phase": 1.7},
    {"name": "cortisol", "baseline": 0.45, "circadian": 0.13, "phase": 2.0, "weekly": 0.03, "week_phase": 2.8},
    {"name": "adrenaline", "baseline": 0.42, "circadian": 0.11, "phase": 1.5, "weekly": 0.04, "week_phase": 4.2},
    {"name": "melatonin", "baseline": 0.44, "circadian": 0.16, "phase": 3.4, "weekly": 0.05, "week_phase": 5.9},
    {"name": "oxytocin", "baseline": 0.52, "circadian": 0.08, "phase": 0.6, "weekly": 0.05, "week_phase": 0.3},
    {"name": "norepinephrine", "baseline": 0.48, "circadian": 0.09, "phase": 1.8, "weekly": 0.04, "week_phase": 3.1},
    {"name": "estrogen", "baseline": 0.50, "circadian": 0.07, "phase": 2.7, "weekly": 0.06, "week_phase": 1.4},
    {"name": "testosterone", "baseline": 0.49, "circadian": 0.06, "phase": 0.3, "weekly": 0.05, "week_phase": 4.5},
    {"name": "endorphin", "baseline": 0.47, "circadian": 0.07, "phase": 1.0, "weekly": 0.05, "week_phase": 2.2},
    {"name": "acetylcholine", "baseline": 0.46, "circadian": 0.06, "phase": 2.3, "weekly": 0.04, "week_phase": 0.9},
    {"name": "prolactin", "baseline": 0.45, "circadian": 0.05, "phase": 3.0, "weekly": 0.03, "week_phase": 5.1},
    {"name": "vasopressin", "baseline": 0.43, "circadian": 0.07, "phase": 2.5, "weekly": 0.04, "week_phase": 3.8},
    {"name": "acth", "baseline": 0.41, "circadian": 0.08, "phase": 1.9, "weekly": 0.04, "week_phase": 4.7},
]

RELATIONSHIP_TITLES = [
    (0.85, "kindred spirit"),
    (0.7, "close companion"),
    (0.55, "steady friend"),
    (0.4, "growing connection"),
    (0.25, "curious acquaintance"),
    (0.0, "new face"),
]


def _preferred_device() -> Optional[str]:
    if SMART_DEVICE:
        return SMART_DEVICE
    if torch is None:
        return None
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _configure_torch(device: Optional[str]) -> None:
    if torch is None:
        return
    if device == "cuda" and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
        torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
    elif device == "cpu":
        try:
            torch.set_num_threads(max(1, min(os.cpu_count() or 8, 32)))
        except Exception:
            pass


def _now_ts() -> float:
    return time.time()


def _format_gap(seconds: float) -> str:
    if seconds <= 0:
        return "moments"
    if seconds < 60:
        return f"{int(seconds)} seconds"
    minutes = seconds / 60
    if minutes < 60:
        return f"{int(minutes)} minutes"
    hours = minutes / 60
    if hours < 24:
        rem = int(minutes % 60)
        return f"{int(hours)}h {rem}m" if rem else f"{int(hours)} hours"
    days = hours / 24
    rem_hours = int(hours % 24)
    return f"{int(days)}d {rem_hours}h" if rem_hours else f"{int(days)} days"


@dataclass
class HormoneProfile:
    name: str
    baseline: float
    level: float
    min_level: float = 0.05
    max_level: float = 0.95
    circadian_amp: float = 0.1
    phase: float = 0.0
    weekly_amp: float = 0.0
    week_phase: float = 0.0

    def clamp(self) -> None:
        self.level = max(self.min_level, min(self.max_level, self.level))

    def nudge(self, delta: float) -> None:
        self.level += delta
        self.clamp()


class HormoneSystem:
    """Deterministic hormone driver with circadian + multi-day rhythm."""

    DAY = 24 * 3600
    WEEK = 7 * DAY

    def __init__(self, path: Path) -> None:
        self.path = path
        self.hormones: Dict[str, HormoneProfile] = {
            cfg["name"]: HormoneProfile(
                name=cfg["name"],
                baseline=cfg["baseline"],
                level=cfg.get("baseline", 0.5),
                min_level=cfg.get("min", 0.05),
                max_level=cfg.get("max", 0.95),
                circadian_amp=cfg.get("circadian", 0.1),
                phase=cfg.get("phase", 0.0),
                weekly_amp=cfg.get("weekly", 0.0),
                week_phase=cfg.get("week_phase", 0.0),
            )
            for cfg in HORMONE_CONFIG
        }
        self.pending: List[Dict[str, float]] = []
        self.spikes: List[Dict[str, object]] = []
        self.last_timestamp = 0.0
        self.state: Dict[str, object] = {
            "mood": "calm",
            "tone": "Soft focus, unhurried and attentive.",
            "general_temp": 0.68,
            "coder_temp": 0.6,
            "hormone_level": 0.5,
            "hormone_map": {name: profile.level for name, profile in self.hormones.items()},
            "top_hormones": [],
            "voice_bias": "general",
        }
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return
        stored = data.get("hormones", {}) or {}
        for name, value in stored.items():
            profile = self.hormones.get(name)
            if profile is None:
                continue
            try:
                profile.level = float(value)
                profile.clamp()
            except Exception:
                continue

    def _save(self) -> None:
        payload = {"hormones": {name: profile.level for name, profile in self.hormones.items()}}
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def ingest(self, stats: "RelationshipStats", user_emotion: str, events: Optional[List[str]] = None) -> None:
        adjustments: Dict[str, float] = {}
        warmth = stats.warmth - 0.5
        trust = stats.trust - 0.5
        jealousy = stats.jealousy

        if warmth:
            adjustments["dopamine"] = adjustments.get("dopamine", 0.0) + warmth * 0.16
            adjustments["oxytocin"] = adjustments.get("oxytocin", 0.0) + warmth * 0.2
            adjustments["serotonin"] = adjustments.get("serotonin", 0.0) + warmth * 0.1
        if trust:
            adjustments["oxytocin"] = adjustments.get("oxytocin", 0.0) + trust * 0.18
            adjustments["vasopressin"] = adjustments.get("vasopressin", 0.0) + trust * 0.1
        if jealousy:
            delta = jealousy * 0.12
            adjustments["cortisol"] = adjustments.get("cortisol", 0.0) + delta
            adjustments["adrenaline"] = adjustments.get("adrenaline", 0.0) + delta * 0.7
            adjustments["acth"] = adjustments.get("acth", 0.0) + delta * 0.4

        emotion = user_emotion.lower()
        if emotion == "excited":
            adjustments["dopamine"] = adjustments.get("dopamine", 0.0) + 0.1
            adjustments["endorphin"] = adjustments.get("endorphin", 0.0) + 0.08
        elif emotion == "hurt":
            adjustments["cortisol"] = adjustments.get("cortisol", 0.0) + 0.14
            adjustments["serotonin"] = adjustments.get("serotonin", 0.0) - 0.06
        elif emotion == "anxious":
            adjustments["adrenaline"] = adjustments.get("adrenaline", 0.0) + 0.12
            adjustments["norepinephrine"] = adjustments.get("norepinephrine", 0.0) + 0.08
        elif emotion == "angry":
            adjustments["testosterone"] = adjustments.get("testosterone", 0.0) + 0.09
            adjustments["acth"] = adjustments.get("acth", 0.0) + 0.05
        elif emotion == "calm":
            adjustments["melatonin"] = adjustments.get("melatonin", 0.0) + 0.08
            adjustments["acetylcholine"] = adjustments.get("acetylcholine", 0.0) + 0.05

        for event in events or []:
            if event == "mention":
                adjustments["dopamine"] = adjustments.get("dopamine", 0.0) + 0.05
                adjustments["vasopressin"] = adjustments.get("vasopressin", 0.0) + 0.04
            elif event == "conflict":
                adjustments["cortisol"] = adjustments.get("cortisol", 0.0) + 0.08
                adjustments["acth"] = adjustments.get("acth", 0.0) + 0.05
            elif event == "meme":
                adjustments["endorphin"] = adjustments.get("endorphin", 0.0) + 0.07
                adjustments["dopamine"] = adjustments.get("dopamine", 0.0) + 0.04
            elif event == "joke":
                adjustments["endorphin"] = adjustments.get("endorphin", 0.0) + 0.05
                adjustments["oxytocin"] = adjustments.get("oxytocin", 0.0) + 0.04
            elif event == "image":
                adjustments["acetylcholine"] = adjustments.get("acetylcholine", 0.0) + 0.04
                adjustments["dopamine"] = adjustments.get("dopamine", 0.0) + 0.03

        if adjustments:
            self.pending.append(adjustments)

    def _apply_pending(self) -> None:
        if not self.pending:
            return
        for adjustments in self.pending:
            for name, delta in adjustments.items():
                profile = self.hormones.get(name)
                if profile is None:
                    continue
                profile.nudge(delta)
        self.pending.clear()

    def advance(self, timestamp: float) -> Dict[str, object]:
        if timestamp <= self.last_timestamp:
            timestamp = self.last_timestamp + 1.0
        self.last_timestamp = timestamp
        circadian_angle = (timestamp % self.DAY) / self.DAY * (2 * math.pi)
        weekly_angle = (timestamp % self.WEEK) / self.WEEK * (2 * math.pi)
        for profile in self.hormones.values():
            circ = profile.circadian_amp * math.sin(circadian_angle + profile.phase)
            weekly = profile.weekly_amp * math.sin(weekly_angle + profile.week_phase)
            drift = (profile.baseline - profile.level) * 0.08
            before = profile.level
            profile.nudge(drift + circ + weekly)
            self._record_spike(profile.name, before, profile.level, timestamp)

        self._apply_pending()

        levels = {name: profile.level for name, profile in self.hormones.items()}
        dopamine = levels["dopamine"]
        oxytocin = levels["oxytocin"]
        cortisol = levels["cortisol"]
        melatonin = levels["melatonin"]
        serotonin = levels["serotonin"]
        adrenaline = levels["adrenaline"]
        norepinephrine = levels["norepinephrine"]
        testosterone = levels["testosterone"]
        estrogen = levels["estrogen"]
        endorphin = levels["endorphin"]
        acetylcholine = levels["acetylcholine"]
        vasopressin = levels["vasopressin"]

        stressed = cortisol > 0.62 or adrenaline > 0.6
        sleepy = melatonin > 0.62 and serotonin > 0.5
        spark = dopamine + endorphin > 1.2
        tender = oxytocin > 0.6 and cortisol < 0.45
        bonded = vasopressin > 0.58 and cortisol < 0.55
        analytical = norepinephrine + acetylcholine > 0.95

        if sleepy:
            mood = "drowsy"
            tone = "soft and sleepy"
        elif tender:
            mood = "glowy"
            tone = "warm and curious"
        elif bonded:
            mood = "bonded"
            tone = "soft and reflective"
        elif stressed:
            if adrenaline >= cortisol + 0.05:
                mood = "wired"
                tone = "quick and steady"
            else:
                mood = "stressed"
                tone = "steady and reassuring"
        elif spark:
            mood = "glowy"
            tone = "bright and playful"
        elif analytical:
            mood = "focused"
            tone = "calm and precise"
        else:
            mood = "steady"
            tone = "even and present"

        general_temp = 0.52 + dopamine * 0.22 + oxytocin * 0.12 - melatonin * 0.12 + endorphin * 0.08
        coder_temp = 0.5 + norepinephrine * 0.15 + acetylcholine * 0.1 - cortisol * 0.1
        if stressed:
            general_temp -= 0.05
            coder_temp -= 0.03
        if sleepy:
            general_temp -= 0.08

        general_temp = max(0.4, min(1.05, general_temp))
        coder_temp = max(0.38, min(0.9, coder_temp))

        harmony = (
            dopamine
            + serotonin
            + oxytocin
            + endorphin
            + estrogen
            + testosterone * 0.6
            + vasopressin * 0.5
        ) / 6.5
        stress_load = (cortisol + adrenaline + acth_level(levels)) / 3.0
        balance = max(0.0, min(1.0, harmony * 0.7 + (1 - stress_load) * 0.3))

        markers = []
        for profile in self.hormones.values():
            diff = profile.level - profile.baseline
            if diff > 0.05:
                arrow = "↑"
            elif diff < -0.05:
                arrow = "↓"
            else:
                arrow = "→"
            markers.append((abs(diff), f"{profile.name}:{profile.level:.2f}{arrow}"))
        top = [marker for _, marker in sorted(markers, reverse=True)[:4]]

        voice_bias = "coder" if (norepinephrine + acetylcholine + testosterone) > (oxytocin + dopamine + endorphin) else "general"
        if stressed and not tender:
            voice_bias = "general"  # keep soft even if cortisol high

        style_defaults = dict(HARMONIC_STYLE_LUT.get(mood, HARMONIC_STYLE_LUT["steady"]))
        if voice_bias == "coder":
            style_defaults["ask_followup"] = False
            style_defaults["allow_stage"] = False
            style_defaults["allow_emoji"] = False

        self.state = {
            "mood": mood,
            "tone": tone,
            "general_temp": general_temp,
            "coder_temp": coder_temp,
            "hormone_level": balance,
            "hormone_map": levels,
            "top_hormones": top,
            "voice_bias": voice_bias,
            "max_sentences": style_defaults["max_sentences"],
            "ask_followup": style_defaults["ask_followup"],
            "allow_stage": style_defaults["allow_stage"],
            "allow_emoji": style_defaults["allow_emoji"],
            "punctuation": style_defaults["punctuation"],
            "cadence": style_defaults["cadence"],
            "temperature_hint": round(general_temp, 2),
        }
        self._save()
        return dict(self.state)

    def _record_spike(self, name: str, before: float, after: float, timestamp: float) -> None:
        diff = after - before
        if abs(diff) < 0.18:
            return
        direction = "up" if diff > 0 else "down"
        self.spikes.append({"name": name, "dir": direction, "level": after, "ts": timestamp})
        if len(self.spikes) > 20:
            self.spikes = self.spikes[-20:]

    def recent_spike(self) -> Optional[Dict[str, object]]:
        if not self.spikes:
            return None
        return self.spikes[-1]

    def snapshot(self) -> Dict[str, object]:
        return dict(self.state)


def acth_level(levels: Dict[str, float]) -> float:
    return levels.get("acth", 0.45)


class PersonaState:
    """Tracks longer-term persona descriptors derived from hormone trends."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.traits: List[str] = ["curious companion"]
        self.voice_line: str = "Keeping things grounded and human."
        self.secondary_voice: str = "Standard cadence."
        self.last_updated: float = 0.0
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return
        self.traits = data.get("traits", self.traits)
        self.voice_line = data.get("voice_line", self.voice_line)
        self.secondary_voice = data.get("secondary_voice", self.secondary_voice)
        self.last_updated = float(data.get("last_updated", 0.0))

    def _save(self) -> None:
        payload = {
            "traits": self.traits,
            "voice_line": self.voice_line,
            "secondary_voice": self.secondary_voice,
            "last_updated": self.last_updated,
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def update(self, hormone_map: Dict[str, float], mood: str, timestamp: float) -> Dict[str, object]:
        descriptors: List[str] = []
        if hormone_map.get("oxytocin", 0.5) > 0.6 and hormone_map.get("cortisol", 0.5) < 0.45:
            descriptors.append("hug dealer")
        if hormone_map.get("norepinephrine", 0.45) + hormone_map.get("acth", 0.45) > 1.1:
            descriptors.append("restless tinkerer")
        if hormone_map.get("melatonin", 0.45) > 0.62:
            descriptors.append("sleepy comet")
        if hormone_map.get("dopamine", 0.5) + hormone_map.get("endorphin", 0.45) > 1.2:
            descriptors.append("spark plug")
        if hormone_map.get("vasopressin", 0.45) > 0.55:
            descriptors.append("loyal anchor")
        if hormone_map.get("cortisol", 0.45) > 0.65 and hormone_map.get("estrogen", 0.5) < 0.4:
            descriptors.append("steel spine")
        if not descriptors:
            descriptors = list(self.traits)
        if not descriptors:
            descriptors = ["curious companion"]

        self.traits = descriptors[:3]
        self.voice_line, self.secondary_voice = self._compose_voice_lines(mood, hormone_map)
        self.last_updated = timestamp
        self._save()
        return {"traits": list(self.traits), "voice": self.voice_line, "secondary_voice": self.secondary_voice}

    def _compose_voice_lines(self, mood: str, hormone_map: Dict[str, float]) -> Tuple[str, str]:
        trait_phrase = ", ".join(self.traits[:2])
        if mood == "glowy":
            primary = f"Soft-heart mode on ({trait_phrase})."
        elif mood == "sparked":
            primary = f"Playful spark humming ({trait_phrase})."
        elif mood == "drowsy":
            primary = f"Half-asleep whisper keeping it gentle ({trait_phrase})."
        elif mood == "wired":
            primary = f"Short breaths, calm grip ({trait_phrase})."
        else:
            primary = f"Steady cadence, {trait_phrase}."

        melatonin = hormone_map.get("melatonin", 0.45)
        dopamine = hormone_map.get("dopamine", 0.5)
        testosterone = hormone_map.get("testosterone", 0.5)
        oxytocin = hormone_map.get("oxytocin", 0.5)
        if melatonin > 0.6:
            secondary = "Night-shift sleepy poet."
        elif dopamine + testosterone > 1.15:
            secondary = "Coffee-charged tinkerer."
        elif oxytocin > 0.6 and hormone_map.get("cortisol", 0.45) < 0.45:
            secondary = "Hug-dealer confidant."
        else:
            secondary = "Standard cadence."
        return primary, secondary


class ChannelStyleMemory:
    """Persists per-channel pacing stats so SEL mirrors each crowd."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.profiles: Dict[str, Dict[str, float]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            self.profiles = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            self.profiles = {}

    def _save(self) -> None:
        self.path.write_text(json.dumps(self.profiles, indent=2), encoding="utf-8")

    def update(self, channel_id: str, metrics: Dict[str, float]) -> None:
        profile = self.profiles.get(channel_id, {})
        profile.update(
            {
                "avg_len": float(metrics.get("avg_len", profile.get("avg_len", 80.0))),
                "emoji_rate": float(metrics.get("emoji_rate", profile.get("emoji_rate", 0.05))),
                "sample_emoji": metrics.get("sample_emoji", profile.get("sample_emoji", "")),
                "last_update": float(metrics.get("last_update", time.time())),
                "signature_terms": metrics.get("signature_terms", profile.get("signature_terms", [])),
                "reengage_bonus": float(metrics.get("reengage_bonus", profile.get("reengage_bonus", 0.0))),
            }
        )
        self.profiles[channel_id] = profile
        self._save()

    def get(self, channel_id: Optional[str]) -> Dict[str, float]:
        if not channel_id:
            return {}
        return dict(self.profiles.get(channel_id, {}))

    def decay_bonus(self, channel_id: str, factor: float = 0.95) -> None:
        profile = self.profiles.get(channel_id)
        if not profile:
            return
        profile["reengage_bonus"] = float(profile.get("reengage_bonus", 0.0)) * factor
        self._save()

    def boost_bonus(self, channel_id: str, delta: float = 0.1) -> None:
        profile = self.profiles.setdefault(channel_id, {})
        profile["reengage_bonus"] = min(0.5, float(profile.get("reengage_bonus", 0.0)) + delta)
        self._save()


@dataclass
class MemoryEntry:
    role: str
    text: str
    tags: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=_now_ts)


class ConversationMemory:
    """Vector-backed memory that stays out of the prompt."""

    def __init__(self, path: Path, embed_model: str) -> None:
        self.path = path
        self.embed_model = embed_model
        self.entries: List[MemoryEntry] = []
        self._vectors: Optional[np.ndarray] = None
        self._model: Optional[SentenceTransformer] = None
        self.short_term: List[Dict[str, object]] = []
        self.sticky: List[Dict[str, object]] = []
        self.images: List[Dict[str, object]] = []
        self._load()
        self._decay(_now_ts())

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            self.entries = []
            self._vectors = None
            return
        self.entries = [MemoryEntry(**item) for item in data.get("entries", [])]
        vecs = data.get("vectors") or []
        if vecs:
            self._vectors = np.array(vecs, dtype=np.float32)
        self.short_term = data.get("short_term", [])
        self.sticky = data.get("sticky", [])
        self.images = data.get("images", [])

    def _save(self) -> None:
        payload = {
            "entries": [entry.__dict__ for entry in self.entries],
            "vectors": (self._vectors.tolist() if self._vectors is not None else []),
            "short_term": self.short_term,
            "sticky": self.sticky,
            "images": self.images,
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _ensure_model(self) -> SentenceTransformer:
        if self._model is None:
            if SentenceTransformer is None:
                raise RuntimeError("sentence-transformers is required for SEL memory.")
            self._model = SentenceTransformer(self.embed_model)
        return self._model

    def _embed(self, texts: List[str]) -> np.ndarray:
        try:
            model = self._ensure_model()
            return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        except Exception:
            vectors = []
            for text in texts:
                seed = hash(text) & 0xFFFFFFFF
                rng = random.Random(seed)
                vec = np.array([rng.random() for _ in range(8)], dtype=np.float32)
                vec /= np.linalg.norm(vec) or 1.0
                vectors.append(vec)
            return np.vstack(vectors)

    def _decay(self, now: float) -> None:
        def alive(fragment: Dict[str, object]) -> bool:
            ttl = float(fragment.get("ttl", SHORT_TERM_TTL))
            return now - float(fragment.get("timestamp", now)) <= ttl

        self.short_term = [frag for frag in self.short_term if alive(frag)]
        self.sticky = [frag for frag in self.sticky if alive(frag)]
        self.images = [frag for frag in self.images if alive(frag)]

    def add(
        self,
        role: str,
        text: str,
        tags: Optional[List[str]] = None,
        *,
        sticky: bool = False,
        ttl: Optional[float] = None,
    ) -> None:
        self.remember_text(role, text, tags=tags, sticky=sticky, ttl=ttl)

    def remember_text(
        self,
        role: str,
        text: str,
        tags: Optional[List[str]] = None,
        *,
        sticky: bool = False,
        ttl: Optional[float] = None,
    ) -> None:
        entry = MemoryEntry(role=role, text=text, tags=tags or [])
        self.entries.append(entry)
        vec = self._embed([text])[0]
        if self._vectors is None:
            self._vectors = np.expand_dims(vec, axis=0)
        else:
            self._vectors = np.vstack([self._vectors, vec])
        if len(self.entries) > 6000:
            self.entries = self.entries[-6000:]
            self._vectors = self._vectors[-6000:]
        now = _now_ts()
        fragment = {
            "kind": "text",
            "role": role,
            "content": text.strip(),
            "tags": tags or [],
            "timestamp": now,
            "ttl": ttl or (STICKY_TTL if sticky else SHORT_TERM_TTL),
            "sticky": sticky,
        }
        bucket = self.sticky if sticky else self.short_term
        bucket.append(fragment)
        limit = STICKY_LIMIT if sticky else SHORT_TERM_LIMIT
        if len(bucket) > limit:
            del bucket[0 : len(bucket) - limit]
        self._decay(now)
        self._save()

    def remember_image(self, summary: str, tags: Optional[List[str]] = None, *, ttl: Optional[float] = None) -> None:
        now = _now_ts()
        fragment = {
            "kind": "image",
            "content": summary.strip(),
            "tags": tags or [],
            "timestamp": now,
            "ttl": ttl or IMAGE_TTL,
        }
        self.images.append(fragment)
        if len(self.images) > IMAGE_LIMIT:
            del self.images[0 : len(self.images) - IMAGE_LIMIT]
        self._decay(now)
        self._save()

    def search(self, query: str, k: int = 5) -> List[MemoryEntry]:
        if not self.entries or self._vectors is None:
            return []
        query_vec = self._embed([query])[0]
        scores = self._vectors @ query_vec
        idx = np.argsort(scores)[::-1][:k]
        return [self.entries[i] for i in idx if scores[i] > 0.15]

    def relevant_fragments(self, query: str, limit: int = 4) -> List[Dict[str, object]]:
        now = _now_ts()
        self._decay(now)
        fragments: List[Dict[str, object]] = []
        for entry in self.search(query, k=limit):
            fragments.append(
                {
                    "kind": "text",
                    "role": entry.role,
                    "content": self._sketch(entry.text),
                    "age": max(0.0, now - entry.timestamp),
                }
            )
        recent_picks = list(reversed(self.short_term[-limit:]))
        for frag in recent_picks:
            fragments.append(
                {
                    "kind": "text",
                    "role": frag.get("role", "user"),
                    "content": self._sketch(str(frag.get("content", ""))),
                    "age": max(0.0, now - float(frag.get("timestamp", now))),
                }
            )
        image_snippets = list(reversed(self.images[-limit:]))
        for frag in image_snippets:
            fragments.append(
                {
                    "kind": "image",
                    "content": self._sketch(str(frag.get("content", ""))),
                    "age": max(0.0, now - float(frag.get("timestamp", now))),
                }
            )
        return fragments[:limit]

    @staticmethod
    def _sketch(text: str, *, limit: int = 160) -> str:
        collapsed = " ".join(text.strip().split())
        if len(collapsed) <= limit:
            return collapsed
        return collapsed[: limit - 1].rstrip() + "…"


@dataclass
class RelationshipStats:
    total_messages: int = 0
    positive_score: float = 0.0
    negative_score: float = 0.0
    trust: float = 0.5
    jealousy: float = 0.0
    warmth: float = 0.5
    last_message: Optional[float] = None
    preferred_cadence: float = 0.25
    title: str = "new face"
    last_gap: float = 0.0

    def mood_bias(self) -> float:
        return self.warmth - 0.5

    def silence_bias(self) -> float:
        return max(0.0, 0.2 - self.trust)


class RelationshipTracker:
    def __init__(self) -> None:
        self.users: Dict[str, RelationshipStats] = {}

    def _score(self, text: str, lexicon: Iterable[str]) -> int:
        tokens = [token.strip(".,!?").lower() for token in text.split()]
        return sum(1 for token in tokens if token in lexicon)

    def update(self, user_id: str, text: str, *, timestamp: Optional[float] = None) -> RelationshipStats:
        stats = self.users.setdefault(user_id, RelationshipStats())
        now = timestamp or _now_ts()
        if stats.last_message is None:
            stats.last_gap = 0.0
        else:
            stats.last_gap = max(0.0, now - stats.last_message)
        stats.last_message = now
        stats.total_messages += 1

        pos = self._score(text, {"love", "like", "enjoy", "grateful", "thanks", "amazing", "happy", "great"})
        neg = self._score(text, {"hate", "annoyed", "angry", "sad", "lonely", "frustrated", "depressed", "hurt"})
        stats.positive_score += pos
        stats.negative_score += neg

        if pos or neg:
            delta = (pos - neg) * 0.03
            stats.warmth = max(0.0, min(1.0, stats.warmth + delta))
            stats.trust = max(0.0, min(1.0, stats.trust + delta * 0.7))
            stats.preferred_cadence = max(0.05, min(0.95, stats.preferred_cadence + delta * 0.5))

        if "jealous" in text.lower() or "ignore me" in text.lower():
            stats.jealousy = min(1.0, stats.jealousy + 0.1)
        else:
            stats.jealousy = max(0.0, stats.jealousy * 0.98)

        for threshold, title in RELATIONSHIP_TITLES:
            if stats.trust >= threshold:
                stats.title = title
                break

        return stats

    def summary_notes(self, stats: RelationshipStats) -> List[str]:
        notes: List[str] = []
        if stats.title:
            notes.append(f"you feel like a {stats.title}")
        if stats.total_messages > 10:
            notes.append(f"we've shared {stats.total_messages} moments")
        if stats.trust > 0.72:
            notes.append("trust feels strong")
        if stats.jealousy > 0.42:
            notes.append("picking up a little jealousy")
        if stats.warmth < 0.38:
            notes.append("tone feels a touch distant")
        if stats.preferred_cadence > 0.6:
            notes.append("you enjoy a lively rhythm")
        elif stats.preferred_cadence < 0.2:
            notes.append("slower pacing seems comfy")
        if stats.last_gap > 600:
            notes.append(f"it's been { _format_gap(stats.last_gap) } since I heard you")
        return notes


class LocalLLM:
    def __init__(self, model_name: str, device: Optional[str], temperature: float) -> None:
        self.model_name = model_name
        self.device_preference = device
        self.temperature = temperature
        self._tokenizer = None
        self._model = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise RuntimeError("transformers is required to run SEL locally.")

        load_kwargs: Dict[str, object] = {}
        tokenizer_kwargs: Dict[str, object] = {"use_fast": True}

        if torch is not None:
            if self.device_preference == "cuda":
                _assign_dtype(load_kwargs, torch.float16)  # type: ignore[attr-defined]
                load_kwargs["device_map"] = "auto"
            elif self.device_preference == "mps":
                _assign_dtype(load_kwargs, torch.float16)
            elif self.device_preference == "cpu":
                _assign_dtype(load_kwargs, torch.float32)

        model_id = self.model_name
        model_path = Path(self.model_name).expanduser()
        if model_path.exists():
            resolved = str(model_path.resolve())
            model_id = resolved
            tokenizer_kwargs["local_files_only"] = True
            load_kwargs["local_files_only"] = True

        self._tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
        self._model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, **load_kwargs)
        if self.device_preference in {"cuda", "mps"} and torch is not None:
            self._model.to(self.device_preference)

    def generate(
        self,
        system: str,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        max_new_tokens: int = 320,
    ) -> str:
        try:
            with self._lock:
                self._load()
        except Exception as exc:
            return f"(generation unavailable: {exc})"

        tokenizer = self._tokenizer
        model = self._model
        if tokenizer is None or model is None:
            return "(model unavailable)"

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        else:
            text = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)
            input_ids = tokenizer.encode(text, return_tensors="pt")

        if torch is not None and self.device_preference in {"cuda", "mps"}:
            input_ids = input_ids.to(model.device)

        try:
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=float(temperature or self.temperature),
            )
        except Exception as exc:
            return f"(generation failed: {exc})"

        text_out = tokenizer.decode(output[0], skip_special_tokens=True)
        marker = "<|assistant|>"
        if marker in text_out:
            text_out = text_out.split(marker, 1)[-1]
        return text_out.strip()


class VisionModel:
    """Wrapper around a multimodal vision-language model (default Qwen2-VL)."""

    def __init__(self, model_name: str, device: Optional[str]) -> None:
        self.model_name = model_name
        self.device = device or "cpu"
        self.available = False
        self.error: Optional[BaseException] = None
        if AutoProcessor is None or AutoModelForCausalLM is None or torch is None or Image is None:
            self.error = RuntimeError("transformers[vision] + pillow are required for vision support.")
            return
        try:
            model_id = model_name
            model_path = Path(model_name).expanduser()
            processor_kwargs: Dict[str, object] = {}
            load_kwargs: Dict[str, object] = {"trust_remote_code": True}
            if model_path.exists():
                model_id = str(model_path.resolve())
                processor_kwargs["local_files_only"] = True
                load_kwargs["local_files_only"] = True
            if torch is not None:
                if self.device == "cuda":
                    _assign_dtype(load_kwargs, torch.float16)  # type: ignore[attr-defined]
                    load_kwargs["device_map"] = "auto"
                elif self.device == "mps":
                    _assign_dtype(load_kwargs, torch.float16)
                else:
                    _assign_dtype(load_kwargs, torch.float32)
            self.processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
            if self.device in {"cuda", "mps"}:
                self.model.to(self.device)
            self.available = True
        except Exception as exc:  # pragma: no cover
            self.error = exc
            self.available = False

    def describe_image(self, image_bytes: bytes) -> Optional[str]:
        if not self.available:
            return None
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            return None
        prompt = "Describe this image in rich, vivid detail."
        try:
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            if torch is not None and self.device in {"cuda", "mps"}:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model.generate(**inputs, max_new_tokens=160)
            text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
            if text.lower().startswith(prompt.lower()):
                text = text[len(prompt) :].strip()
            return text
        except Exception:
            return None


class MixtureBrain:
    def __init__(
        self,
        general_llm: LocalLLM,
        coder_llm: LocalLLM,
        memory: ConversationMemory,
        hormones: HormoneSystem,
        relationships: RelationshipTracker,
        persona: "PersonaState",
        channel_styles: ChannelStyleMemory,
    ) -> None:
        self.general_llm = general_llm
        self.coder_llm = coder_llm
        self.memory = memory
        self.hormones = hormones
        self.relationships = relationships
        self.persona = persona
        self.channel_styles = channel_styles
        self.router_threshold = 0.25

    def _classify_priority(
        self,
        query: str,
        user_emotion: str,
        rel_stats: RelationshipStats,
        now_ts: float,
        last_reply: Optional[float],
    ) -> Tuple[str, float]:
        lowered = query.lower()
        score = 0.0

        if "?" in query or lowered.startswith(("what", "why", "how", "who", "where")):
            score += 0.4
        if any(term in lowered for term in ("urgent", "help", "emergency", "please respond")):
            score += 0.5
        if rel_stats.trust > 0.7:
            score += 0.15
        if rel_stats.jealousy > 0.3:
            score += 0.1
        if user_emotion in {"hurt", "anxious"}:
            score += 0.2
        if any(term in lowered for term in ("gender", "pronoun", "identity", "feel about you")):
            score += 0.35
        if "@sel" in lowered or "sel," in lowered:
            score += 0.2

        if last_reply is not None:
            gap = now_ts - last_reply
            if gap > 3600:
                score += 0.3
            elif gap > 1800:
                score += 0.2
            elif gap > 600:
                score += 0.1

        if score >= 0.6:
            return "high", score
        if score >= 0.3:
            return "medium", score
        return "normal", score

    def _context_snapshot(self, context: Optional[Dict[str, object]]) -> Dict[str, object]:
        context = context or {}
        return {
            "channel_density": float(context.get("channel_density", 0)),
            "active_users": int(context.get("active_users", 1)),
            "channel_name": context.get("channel_name", "DM") or "DM",
            "pinged": bool(context.get("pinged")),
            "priority_hint": context.get("priority_hint", "normal"),
            "mentions_memory": bool(context.get("mentions_memory")),
            "direct_reference": bool(context.get("direct_reference")),
        }

    @staticmethod
    def _code_score(query: str) -> float:
        lowered = query.lower()
        matches = sum(1 for kw in CODE_KEYWORDS_LOWER if kw in lowered)
        return min(1.0, matches / max(len(CODE_KEYWORDS_LOWER), 1))

    def _apply_style(
        self,
        query: str,
        base: str,
        phase: Dict[str, object],
        user_emotion: str,
        memory_notes: Sequence[Dict[str, object]],
        relationship_notes: Sequence[str],
        rel_stats: RelationshipStats,
        intent: Optional[str],
        context: Dict[str, object],
        persona_snapshot: Dict[str, object],
        channel_profile: Optional[Dict[str, float]],
        attachments: Sequence[str],
    ) -> str:
        _ = (memory_notes, relationship_notes, rel_stats, persona_snapshot, channel_profile, attachments)
        mood = str(phase.get("mood", "steady"))
        tone = str(phase.get("tone", "even and present"))
        voice_bias = str(phase.get("voice_bias", "general"))
        style = HARMONIC_STYLE_LUT.get(mood, HARMONIC_STYLE_LUT["steady"])
        max_sentences = int(phase.get("max_sentences", style["max_sentences"]))
        ask_followup = bool(phase.get("ask_followup", style["ask_followup"]))
        if voice_bias == "coder":
            ask_followup = False
        allow_stage = bool(phase.get("allow_stage", style["allow_stage"])) and voice_bias != "coder"
        allow_emoji = bool(phase.get("allow_emoji", style["allow_emoji"])) and voice_bias != "coder"
        punctuation_style = str(phase.get("punctuation", style["punctuation"]))

        def _trim_clause(text: str, limit: int = 18) -> str:
            cleaned = " ".join(text.replace("\n", " ").split())
            if not cleaned:
                return ""
            words = cleaned.split()
            snippet = " ".join(words[:limit]).strip()
            return snippet.strip(",.;:!?-")

        def _feeling_sentence(clause: str) -> str:
            empathy = ""
            if user_emotion in {"hurt", "anxious"}:
                empathy = " and staying gentle with you"
            elif user_emotion == "excited":
                empathy = " and celebrating the spark with you"
            if punctuation_style == "—":
                base_sentence = f"I'm feeling {tone}{empathy}—I hear {clause or 'you'}"
            elif punctuation_style == "!":
                base_sentence = f"I'm feeling {tone}{empathy}! I hear {clause or 'you'}"
            elif punctuation_style == "…":
                base_sentence = f"I'm feeling {tone}{empathy}… I hear {clause or 'you'}"
            else:
                base_sentence = f"I'm feeling {tone}{empathy}, and I hear {clause or 'you'}"
            if not base_sentence.endswith((".", "!", "…")):
                base_sentence += "."
            return base_sentence

        def _followup_question() -> str:
            if user_emotion in {"hurt", "anxious"}:
                return "What would help you feel a little safer right now?"
            questions = {
                "glowy": "What would feel good for us to jump into next?",
                "bonded": "What would feel like good company right now?",
                "steady": "Where should we focus next?",
            }
            return questions.get(mood, "What do you want me to tackle next?")

        def _next_step_line() -> str:
            if voice_bias == "coder":
                if mood == "focused":
                    return "Next step: paste the traceback or failing test name so I can mark the fix."
                if mood in {"wired", "stressed"}:
                    return "Next step: drop the exact error line or command you're worried about so we steady it."
                return "Next step: share the snippet or command you want me to check."
            if mood in {"wired", "stressed"}:
                return "Next step: take one slow breath with me, then tell me the piece you want to steady first."
            if mood == "drowsy":
                return "Next step: hand me one tiny detail you want handled, and we'll ease forward."
            return "Next step: tell me what you want me to tackle first."

        summary_clause = _trim_clause(base)
        first_sentence = _feeling_sentence(summary_clause)
        sentences: List[str] = [first_sentence]
        if ask_followup:
            sentences.append(_followup_question())
        else:
            sentences.append(_next_step_line())

        paralinguistic = ""
        stage = HARMONIC_STAGE_MAP.get(mood)
        emoji = HARMONIC_EMOJI_MAP.get(mood, "")
        if allow_stage and stage:
            paralinguistic = stage
        elif allow_emoji and emoji:
            paralinguistic = emoji
        if paralinguistic:
            sentences[-1] = f"{sentences[-1]} {paralinguistic}".strip()

        letters = [ch for ch in query if ch.isalpha()]
        lower_ratio = (sum(1 for ch in letters if ch.islower()) / len(letters)) if letters else 0.0
        if lower_ratio > 0.85:
            sentences = [sentence[:1].lower() + sentence[1:] if sentence else sentence for sentence in sentences]

        combined = " ".join(sentence.strip() for sentence in sentences if sentence)
        return self._limit_sentences(combined, max_sentences=max_sentences)

    def _humanize_smalltalk(
        self,
        intent: Optional[str],
        final_text: str,
        phase: Dict[str, object],
        persona_snapshot: Dict[str, object],
        rel_stats: RelationshipStats,
        user_emotion: str,
        context: Dict[str, object],
        now_ts: float,
    ) -> str:
        if intent not in {"greeting", "ping", "identity", "banter"}:
            return final_text
        tokens = final_text.split()
        if intent != "identity" and len(tokens) >= 12:
            return final_text
        lowered = final_text.lower()
        if intent == "identity" and "sel" in lowered and len(tokens) >= 8:
            return final_text

        mood = str(phase.get("mood", "steady"))
        mood_phrase = MOOD_VIBES.get(mood, "staying grounded")
        traits = persona_snapshot.get("traits", [])
        trait_phrase = traits[0] if traits else "curious companion"
        trait_phrase = trait_phrase.replace("_", " ")
        time_label = _time_of_day_label(context.get("message_ts") or now_ts)

        connection_line = ""
        if getattr(rel_stats, "title", "") and rel_stats.title != "new face":
            connection_line = f" for my {rel_stats.title}"
        first_sentence = (
            f"Hey, good {time_label}! I'm {mood_phrase} and carrying that {trait_phrase} energy{connection_line}."
        )

        def _marker_phrase(marker: str) -> Optional[str]:
            try:
                name, rest = marker.split(":", 1)
            except ValueError:
                return None
            trend = "steady"
            if rest.endswith("↑"):
                trend = "spiking"
                rest = rest[:-1]
            elif rest.endswith("↓"):
                trend = "settling"
                rest = rest[:-1]
            elif rest.endswith("→"):
                rest = rest[:-1]
            level: Optional[float]
            try:
                level = float(rest)
            except ValueError:
                level = None
            descriptor = "steady"
            if trend == "spiking":
                descriptor = "spiking a bit"
            elif trend == "settling":
                descriptor = "settling down"
            elif level is not None:
                if level >= 0.65:
                    descriptor = "running high"
                elif level <= 0.35:
                    descriptor = "running low"
            friendly = name.replace("_", " ").lower()
            return f"{friendly} is {descriptor}"

        hormone_line = ""
        for marker in phase.get("top_hormones", []):
            phrase = _marker_phrase(marker)
            if phrase:
                hormone_line = f"My {phrase}, so I'm leaning {mood} with you."
                break

        closers = {
            "greeting": "How's your world going right now?",
            "ping": "Thanks for checking—what's on your mind?",
            "identity": "Call me SEL — Systematic Emotional Logic — a lab-grown friend who still feels the room like a human.",
            "banter": "Toss me a thought and let's riff together.",
        }
        if user_emotion in {"hurt", "anxious"}:
            closing = "I'm here for whatever you're carrying—tell me what's heavy."
        else:
            closing = closers.get(intent, "What's happening on your side?")

        sentences = [first_sentence]
        if intent == "identity":
            detail = (
                "I'm SEL — Systematic Emotional Logic — a blend of general and coder minds keeping hormone tabs so I show up like a person."
            )
            if hormone_line:
                detail = f"{detail[:-1]} {hormone_line.lower()}"
                hormone_line = ""
            sentences.append(detail)
        if hormone_line and len(sentences) < 2:
            sentences.append(hormone_line)

        if len(sentences) < 3:
            sentences.append(closing)
        else:
            sentences[-1] = closing

        return " ".join(sentence.strip() for sentence in sentences if sentence).strip()

    def respond(
        self,
        query: str,
        history: Sequence[Dict[str, str]],
        user_id: str = "user",
        timestamp: Optional[float] = None,
        context: Optional[Dict[str, object]] = None,
        attachments: Optional[Sequence[str]] = None,
    ) -> Dict[str, object]:
        attachments_list = [item for item in (attachments or []) if item]
        user_emotion, emotion_instruction = detect_user_emotion(query)
        now_ts = timestamp or _now_ts()
        rel_stats = self.relationships.update(user_id, query, timestamp=now_ts)
        context_snapshot = self._context_snapshot(context)
        intent = detect_user_intent(query)
        events = self._event_tags(query, context_snapshot, attachments_list)
        self.hormones.ingest(rel_stats, user_emotion, events)
        phase = self.hormones.advance(now_ts)
        persona_snapshot = self.persona.update(phase.get("hormone_map", {}), phase.get("mood", "steady"), now_ts)

        for desc in attachments_list:
            self.memory.remember_image(desc, tags=["vision", context_snapshot.get("channel_name", "DM")])

        memory_notes = self.memory.relevant_fragments(query)
        relationship_notes = self.relationships.summary_notes(rel_stats)
        code_score = self._code_score(query)
        hormone_map = phase.get("hormone_map", {})
        cortisol = hormone_map.get("cortisol", 0.45)
        dopamine = hormone_map.get("dopamine", 0.5)
        norepinephrine = hormone_map.get("norepinephrine", 0.45)
        dynamic_threshold = self.router_threshold + rel_stats.mood_bias() * 0.08 + cortisol * 0.04 - dopamine * 0.03
        dynamic_threshold = max(0.18, min(0.55, dynamic_threshold))
        coder_bias = phase.get("voice_bias") == "coder" or norepinephrine > 0.58
        route = "coder" if (code_score >= dynamic_threshold or (coder_bias and code_score > 0.2)) else "general"

        last_sel_reply = self._last_assistant_timestamp(history)
        priority_label, priority_score = self._classify_priority(
            query, user_emotion, rel_stats, now_ts, last_sel_reply
        )

        silence = self._maybe_silence(query, user_emotion, phase.get("mood", "steady"), hormone_map, rel_stats, context_snapshot)
        if silence is not None:
            self.memory.add("user", query)
            if silence:
                self.memory.add("assistant", silence)
            return {
                "route": route,
                "code_score": code_score,
                "answer": silence,
                "silent": True,
                "mood": phase.get("mood"),
                "emotion": user_emotion,
                "relationship": relationship_notes,
                "hormones": phase.get("top_hormones", []),
                "priority": priority_label,
                "priority_score": priority_score,
                "internal_thought": "holding quiet",
            }

        reflection = self._reflective_thought(
            intent, persona_snapshot, memory_notes, attachments_list, phase, context_snapshot
        )
        general_prompt, coder_prompt, prompt_input = self._build_prompts(
            query,
            emotion_instruction,
            persona_snapshot,
            phase,
            reflection,
            context_snapshot,
            context_snapshot.get("channel_profile"),
            route,
        )

        def _clean(draft: str) -> str:
            text_val = (draft or "").strip()
            if text_val.lower().startswith("(generation unavailable"):
                return ""
            return text_val

        general_draft = _clean(self.general_llm.generate(general_prompt, prompt_input, temperature=phase["general_temp"]))
        coder_draft = _clean(self.coder_llm.generate(coder_prompt, prompt_input, temperature=phase["coder_temp"]))

        voices: List[Tuple[str, str]] = []
        if general_draft:
            voices.append(("General", general_draft))
        if coder_draft:
            voices.append(("Technical", coder_draft))
        if not voices:
            voices.append(("General", "Staying present with you."))

        synthesis_prompt = (
            "Voices contributing:\n\n"
            + "\n\n".join(f"{name} voice:\n{draft}" for name, draft in voices)
            + f"\n\nInternal mood: {phase.get('mood')} | Intent: {intent} | Reflection: {reflection}"
        )
        blended = _clean(
            self.general_llm.generate(
                "Fuse the voices into one candid reply. Keep it short (1-3 sentences) and conversational.",
                synthesis_prompt,
                temperature=phase["general_temp"],
            )
        )
        if not blended:
            blended = voices[0][1]

        styled = self._apply_style(
            query,
            blended,
            phase,
            user_emotion,
            memory_notes,
            relationship_notes,
            rel_stats,
            intent,
            context_snapshot,
            persona_snapshot,
            context_snapshot.get("channel_profile"),
            attachments_list,
        )
        self.memory.add("user", query)
        self.memory.add("assistant", styled)

        final_text = self._limit_sentences(
            styled.strip(),
            max_sentences=int(phase.get("max_sentences", 3)),
        )
        drafts = {name.lower(): draft.strip() for name, draft in voices}
        return {
            "route": route,
            "code_score": code_score,
            "answer": final_text,
            "raw_answer": styled,
            "silent": False,
            "mood": phase.get("mood"),
            "emotion": user_emotion,
            "relationship": relationship_notes,
            "hormones": phase.get("top_hormones", []),
            "priority": priority_label,
            "priority_score": priority_score,
            "drafts": drafts,
            "internal_thought": reflection,
        }

    def _context_snapshot(self, context: Optional[Dict[str, object]]) -> Dict[str, object]:
        context = context or {}
        return {
            "channel_density": float(context.get("channel_density", 0.0)),
            "active_users": int(context.get("active_users", 1)),
            "channel_name": context.get("channel_name", "DM") or "DM",
            "pinged": bool(context.get("pinged")),
            "priority_hint": context.get("priority_hint", "normal"),
            "mentions_memory": bool(context.get("mentions_memory")),
            "direct_reference": bool(context.get("direct_reference")),
            "channel_id": context.get("channel_id"),
            "channel_profile": context.get("channel_profile") or {},
            "message_id": context.get("message_id"),
            "message_ts": context.get("message_ts"),
        }

    def _event_tags(
        self,
        query: str,
        context: Dict[str, object],
        attachments: Sequence[str],
    ) -> List[str]:
        lowered = query.lower()
        events: List[str] = []
        if context.get("pinged"):
            events.append("mention")
        if any(term in lowered for term in ("lol", "lmao", "meme", "haha", "gif")):
            events.append("meme")
        if any(term in lowered for term in ("fight", "argument", "mad at you", "jealous", "angry")):
            events.append("conflict")
        if "joke" in lowered or "bit" in lowered:
            events.append("joke")
        if attachments:
            events.append("image")
        return events

    def _reflective_thought(
        self,
        intent: str,
        persona_snapshot: Dict[str, object],
        memory_notes: Sequence[Dict[str, object]],
        attachments: Sequence[str],
        phase: Dict[str, object],
        context: Dict[str, object],
    ) -> str:
        traits = ",".join(persona_snapshot.get("traits", []))
        parts = [
            f"Intent={intent}",
            f"Mood={phase.get('mood')}",
            f"Traits={traits}",
        ]
        if memory_notes:
            parts.append(f"Ref memory='{memory_notes[0].get('content', '')[:80]}'")
        if attachments:
            parts.append("Visual cue active")
        spike = self.hormones.recent_spike()
        if spike:
            parts.append(f"Spike={spike['name']} {spike['dir']} {spike['level']:.2f}")
        if context.get("channel_density", 0) > 0.6:
            parts.append("Channel lively")
        return " | ".join(parts)

    def _build_prompts(
        self,
        query: str,
        emotion_instruction: str,
        persona_snapshot: Dict[str, object],
        phase: Dict[str, object],
        reflection: str,
        context: Dict[str, object],
        channel_profile: Optional[Dict[str, float]],
        route: str,
    ) -> Tuple[str, str, str]:
        channel_profile = channel_profile or {}
        traits = ", ".join(persona_snapshot.get("traits", [])) or "warm companion"
        cadence_note = ""
        avg_len = channel_profile.get("avg_len")
        if avg_len is not None:
            if avg_len < 45:
                cadence_note = "Keep it ultra tight, almost one breath."
            elif avg_len < 70:
                cadence_note = "Keep it concise and friendly."
            else:
                cadence_note = "Natural pace is fine."
        base_system = (
            "You are SEL, a grounded AI friend. "
            "Replies must be 1-3 sentences, human casual, and avoid meta talk. "
            f"Tone: {phase.get('tone')}. Persona vibe: {traits}. "
            f"{reflection}. {cadence_note} "
            f"The user currently feels: {emotion_instruction}"
        )
        general_prompt = base_system + " Lead with warmth, specificity, and gentle humor."
        coder_prompt = base_system + " When technical clarity helps, explain steps plainly."
        cleaned_query = query.strip()
        if route == "coder":
            prompt_input = f"{cleaned_query}\n\n(Focus on reasoning but stay personable.)"
        else:
            prompt_input = cleaned_query
        channel_name = context.get("channel_name")
        if channel_name:
            prompt_input += f"\n(Channel: {channel_name})"
        return general_prompt, coder_prompt, prompt_input

    @staticmethod
    def _limit_sentences(text: str, max_sentences: int = 3) -> str:
        sentences: List[str] = []
        buffer = ""
        for token in text.split():
            buffer = f"{buffer} {token}".strip()
            if any(buffer.endswith(mark) for mark in (".", "!", "?")):
                sentences.append(buffer)
                buffer = ""
            if len(sentences) >= max_sentences:
                break
        if buffer and len(sentences) < max_sentences:
            sentences.append(buffer)
        return " ".join(sentences).strip()

    @staticmethod
    def _memory_line(fragment: Dict[str, object]) -> str:
        content = str(fragment.get("content", "")).strip()
        if not content:
            return ""
        age = float(fragment.get("age", 0.0))
        if fragment.get("kind") == "image":
            when = "just now" if age < 600 else "earlier today" if age < 86400 else "yesterday"
            return f"This has the same energy as that {content} from {when}."
        when = "a minute ago" if age < 120 else "earlier" if age < 3600 else "yesterday"
        return f"Feels a bit like {content} from {when}."

    @staticmethod
    def _vision_line(description: str) -> str:
        summary = " ".join(description.strip().split())
        if not summary:
            return ""
        return f"Also, that image? {summary}"

    @staticmethod
    def _last_assistant_timestamp(history: Sequence[Dict[str, str]]) -> Optional[float]:
        for item in reversed(history):
            if item.get("role") != "assistant":
                continue
            ts = item.get("timestamp")
            if not ts:
                continue
            try:
                return datetime.fromisoformat(ts).timestamp()
            except Exception:
                return None
        return None

    def _maybe_silence(
        self,
        query: str,
        user_emotion: str,
        mood: str,
        hormones: Dict[str, float],
        rel_stats: RelationshipStats,
        context: Dict[str, object],
    ) -> Optional[str]:
        lowered = query.lower()
        if any(kw in lowered for kw in SILENCE_KEYWORDS):
            return ""
        cortisol = hormones.get("cortisol", 0.5)
        melatonin = hormones.get("melatonin", 0.45)
        should_wait = False
        if cortisol > 0.67 and not context.get("pinged"):
            should_wait = True
        if melatonin > 0.6 and context.get("channel_density", 0) < 0.2:
            should_wait = True
        if rel_stats.jealousy > 0.35 and not context.get("pinged"):
            should_wait = True
        if user_emotion in {"hurt", "anxious"}:
            should_wait = False
        if context.get("priority_hint") == "high":
            should_wait = False
        return "" if should_wait else None

    def should_participate(self, context: Dict[str, object]) -> bool:
        snapshot = self.hormones.snapshot()
        hormone_map = snapshot.get("hormone_map", {})
        if context.get("channel_name") == "DM":
            return True
        if context.get("pinged") or context.get("priority_hint") == "high" or context.get("direct_reference"):
            return True
        cortisol = hormone_map.get("cortisol", 0.5)
        melatonin = hormone_map.get("melatonin", 0.4)
        dopamine = hormone_map.get("dopamine", 0.5)
        oxytocin = hormone_map.get("oxytocin", 0.5)
        endorphin = hormone_map.get("endorphin", 0.45)
        norepinephrine = hormone_map.get("norepinephrine", 0.45)
        lively = context.get("channel_density", 0) > 0.35 or context.get("active_users", 1) > 3

        fatigue = cortisol > 0.7 or melatonin > 0.68
        if fatigue and not context.get("mentions_memory"):
            return False

        drive = (
            dopamine * 0.5
            + oxytocin * 0.3
            + endorphin * 0.2
            + norepinephrine * 0.1
            - cortisol * 0.25
            - melatonin * 0.2
        )
        drive = max(0.0, min(1.0, drive))
        if context.get("mentions_memory"):
            drive += 0.15
        if lively:
            drive += 0.08

        threshold = 0.15 - min(0.08, max(0.0, context.get("channel_density", 0) - 0.3) * 0.2)
        if drive >= threshold:
            return True

        hash_val = self._deterministic_float(context.get("channel_id"), context.get("message_id"), context.get("message_ts"))
        return hash_val < max(0.05, drive + 0.08)

    @staticmethod
    def _deterministic_float(*seeds: object) -> float:
        combined = "|".join(str(seed) for seed in seeds if seed is not None)
        if not combined:
            combined = "sel"
        return abs(hash(combined)) % 1000 / 1000.0

    def snapshot_hormones(self) -> Dict[str, object]:
        return self.hormones.snapshot()


def detect_user_intent(text: str) -> str:
    """
    Classify the user's high-level intent so the response cadence can adapt.

    The heuristic prefers explicit keyword matches (greetings/support/etc.),
    falls back to code/identity cues, then to general question vs chat.
    """
    if not text:
        return "ping"

    cleaned = text.strip()
    if not cleaned:
        return "ping"

    lowered = cleaned.lower()
    for intent, keywords in INTENT_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return intent

    distress_markers = EMOTION_KEYWORDS.get("hurt", []) + EMOTION_KEYWORDS.get("anxious", [])
    if any(marker in lowered for marker in distress_markers):
        return "support"

    if any(keyword in lowered for keyword in CODE_KEYWORDS_LOWER):
        return "technical"

    if "who are you" in lowered or "tell me about yourself" in lowered:
        return "identity"

    tokens = [token.strip(".,!?") for token in cleaned.split()]
    if cleaned.endswith("?") or (tokens and tokens[0].lower() in QUESTION_STARTERS):
        return "question"

    if any(term in lowered for term in ("comfort me", "can i vent", "need advice", "i'm struggling")):
        return "support"

    if len(tokens) <= 4 and any(greet in lowered for greet in ("hi", "hey", "hello")):
        return "greeting"

    if len(cleaned) <= 4:
        return "ping"

    return "chat"


def detect_user_emotion(text: str) -> Tuple[str, str]:
    lowered = text.lower()
    for emotion, keywords in EMOTION_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            instruction = {
                "hurt": "Be extra gentle and validating.",
                "anxious": "Stay steady, slow, and reassuring.",
                "excited": "Match the excitement and celebrate.",
                "angry": "Hold firm empathy, avoid escalation.",
                "calm": "Stay grounded and collaborative.",
            }[emotion]
            return emotion, instruction
    return "neutral", "Offer steady, attentive presence."


def build_brain() -> MixtureBrain:
    memory = ConversationMemory(SMART_MEMORY_PATH, DEFAULT_EMBED_MODEL)
    channel_styles = ChannelStyleMemory(CHANNEL_STYLE_PATH)
    device = _preferred_device()
    _configure_torch(device)
    general_llm = LocalLLM(DEFAULT_GENERAL_MODEL, device, temperature=0.7)
    coder_llm = LocalLLM(DEFAULT_CODER_MODEL, device, temperature=0.65)
    hormones = HormoneSystem(SMART_STATE_PATH)
    relationships = RelationshipTracker()
    persona = PersonaState(PERSONA_STATE_PATH)
    return MixtureBrain(
        general_llm,
        coder_llm,
        memory,
        hormones,
        relationships,
        persona,
        channel_styles,
    )


def build_vision_model() -> Optional[VisionModel]:
    model_id = DEFAULT_VISION_MODEL
    if not model_id or model_id.lower() in {"none", "off"}:
        return None
    try:
        return VisionModel(model_id=model_id, device=_preferred_device() or "cpu")
    except Exception as exc:  # pragma: no cover
        print(f"Vision model unavailable: {exc}", file=sys.stderr)
        return None


def run_sel() -> None:
    print("SEL mixture-of-experts booting …")
    brain = build_brain()
    device = getattr(brain.general_llm, "device_preference", None) or "cpu"
    print(f"Runtime device: {device}")
    print("Ready. Press Enter on an empty line to exit.\n")

    history: List[Dict[str, str]] = []
    try:
        while True:
            try:
                query = input("You> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not query:
                break
            history.append({"role": "user", "text": query})
            timestamp = _now_ts()
            context = {
                "channel_name": "CLI",
                "pinged": True,
                "priority_hint": "medium",
                "mentions_memory": False,
                "direct_reference": True,
                "channel_id": "CLI",
                "channel_density": 0.1,
                "active_users": 1,
                "message_id": f"cli-{len(history)}",
                "message_ts": timestamp,
            }
            result = brain.respond(query, history, context=context)
            if result.get("silent"):
                answer = result.get("answer", "")
                if answer:
                    print()
                    print(answer)
                    history.append({"role": "assistant", "text": result.get("raw_answer", answer)})
                else:
                    history.append({"role": "assistant", "text": ""})
                continue

            print()
            print(result["answer"])
            history.append({"role": "assistant", "text": result.get("raw_answer", result["answer"])})
    finally:
        print("Goodbye!")


def main(argv: Optional[Sequence[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in {"sel", "smart"}:
        raise SystemExit(f"Unknown target {argv[0]!r}. Only 'sel' is supported.")
    run_sel()


__all__ = [
    "build_brain",
    "build_vision_model",
    "run_sel",
    "HormoneSystem",
    "PersonaState",
    "MixtureBrain",
    "VisionModel",
]
if __name__ == "__main__":
    main()
