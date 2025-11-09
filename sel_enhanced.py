from __future__ import annotations

import os
import sys
import json
import math
import random
import time
import re
import shutil
import contextlib
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta

# Reduce noisy HF warnings
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")


# ------------------------------
# .env loader
# ------------------------------


def _load_dotenv(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    try:
        for line in p.read_text().splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if "=" not in s:
                continue
            k, v = s.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            os.environ.setdefault(k, v)
    except Exception:
        pass


_load_dotenv()


# ------------------------------
# Startup diagnostics
# ------------------------------


def _has_dir(p: Optional[str]) -> bool:
    try:
        return bool(p) and os.path.isdir(os.path.expanduser(str(p)))
    except Exception:
        return False


def _has_file(p: Optional[str]) -> bool:
    try:
        return bool(p) and os.path.isfile(os.path.expanduser(str(p)))
    except Exception:
        return False


def _cli_diagnostics() -> None:
    def w(msg: str) -> None:
        print(f"[SEL-Enhanced] {msg}", file=sys.stderr)

    gm = os.getenv("SEL_GENERAL_MODEL")
    cm = os.getenv("SEL_CODER_MODEL")
    if not _has_dir(gm):
        w("SEL_GENERAL_MODEL missing or not a directory.")
    if not _has_dir(cm):
        w("SEL_CODER_MODEL missing or not a directory.")

    try:
        import torch  # type: ignore

        w(f"CUDA available: {torch.cuda.is_available()}")
    except Exception:
        w("Torch not importable; generation may be disabled.")

    if shutil.which("ffmpeg") is None:
        w("ffmpeg not found; VC playback (TTS) will fail.")

    pm = os.getenv("PIPER_MODEL")
    if not _has_file(pm):
        w("PIPER_MODEL missing; TTS disabled.")


# ------------------------------
# Lightweight embedder
# ------------------------------


class _Hasher:
    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def encode(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for t in texts:
            vec = [0.0] * self.dim
            for tok in re.findall(r"\w+", t.lower()):
                vec[hash(tok) % self.dim] += 1.0
            n = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / n for v in vec])
        return out


class Embedder:
    def __init__(self) -> None:
        self._model = None
        self._use_st = False
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self._use_st = True
        except Exception:
            self._model = _Hasher()

    def encode(self, texts: List[str]) -> List[List[float]]:
        if self._use_st:
            return self._model.encode(texts, normalize_embeddings=True).tolist()
        return self._model.encode(texts)


# ------------------------------
# Thought Tree System
# ------------------------------


@dataclass
class ThoughtNode:
    thought_id: str
    content: str
    depth: int
    parent_id: Optional[str]
    timestamp: str
    children: List[str]
    emotion: str = "neutral"
    confidence: float = 0.5


class ThoughtTree:
    def __init__(self, tree_id: str, trigger: str, age_at_creation: int):
        self.tree_id = tree_id
        self.trigger = trigger
        self.age_at_creation = age_at_creation
        self.created_at = datetime.utcnow().isoformat()
        self.nodes: Dict[str, ThoughtNode] = {}
        self.max_depth = 6
        self.max_branches = 4

    def add_node(self, node: ThoughtNode) -> bool:
        if node.depth > self.max_depth:
            return False
        if node.parent_id:
            parent = self.nodes.get(node.parent_id)
            if not parent or len(parent.children) >= self.max_branches:
                return False
            parent.children.append(node.thought_id)
        self.nodes[node.thought_id] = node
        return True

    def to_dict(self) -> Dict:
        return {
            "tree_id": self.tree_id,
            "trigger": self.trigger,
            "age_at_creation": self.age_at_creation,
            "created_at": self.created_at,
            "nodes": {nid: asdict(n) for nid, n in self.nodes.items()},
        }


# ------------------------------
# Belief System
# ------------------------------


class BeliefSystem:
    def __init__(self, path: str = "sel_beliefs.json"):
        self.path = Path(path)
        self.beliefs: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self.beliefs = json.loads(self.path.read_text())
            except Exception:
                self.beliefs = {}

    def _save(self) -> None:
        self.path.write_text(json.dumps(self.beliefs, ensure_ascii=False, indent=2))

    def update(self, category: str, belief: str, confidence: float) -> None:
        """Update a belief with confidence level (0.0 to 1.0)"""
        if category not in self.beliefs:
            self.beliefs[category] = {}
        self.beliefs[category][belief] = {
            "confidence": max(0.0, min(1.0, confidence)),
            "updated": datetime.utcnow().isoformat(),
        }
        self._save()

    def get_summary(self, limit: int = 5) -> str:
        """Get a summary of top beliefs"""
        items = []
        for cat, beliefs in list(self.beliefs.items())[:limit]:
            for belief, data in list(beliefs.items())[:2]:
                conf = data.get("confidence", 0.5)
                items.append(f"{cat}: {belief} (confidence: {conf:.1f})")
        return "\n".join(items) if items else "No beliefs formed yet."


# ------------------------------
# Knowledge System
# ------------------------------


@dataclass
class KnowledgeEntry:
    summary: str
    timestamp: str
    topics: List[str]
    user_context: str


class KnowledgeBase:
    def __init__(self, path: str = "sel_knowledge.json"):
        self.path = Path(path)
        self.entries: List[KnowledgeEntry] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                self.entries = [KnowledgeEntry(**e) for e in data]
            except Exception:
                self.entries = []

    def _save(self) -> None:
        data = [asdict(e) for e in self.entries]
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def add(self, summary: str, topics: List[str], user_context: str) -> None:
        entry = KnowledgeEntry(
            summary=summary,
            timestamp=datetime.utcnow().isoformat(),
            topics=topics,
            user_context=user_context,
        )
        self.entries.append(entry)
        if len(self.entries) > 100:
            self.entries = self.entries[-100:]
        self._save()

    def recent(self, limit: int = 3) -> List[KnowledgeEntry]:
        return self.entries[-limit:]


# ------------------------------
# Memory with consent + TTL
# ------------------------------


@dataclass
class MemoryEntry:
    user_id: str
    text: str
    ts: float
    importance: float  # 0.0-1.0
    access_count: int  # How many times recalled
    last_accessed: float  # Last time this memory was used
    vector: Optional[List[float]]
    tags: Optional[List[str]] = None
    context: str = ""  # Contextual information


class ConversationMemory:
    """Natural human-like memory system - unlimited but with natural forgetting"""

    def __init__(self, path: str = "sel_memory.json") -> None:
        self.path = Path(path)
        # All memories in a single pool (like human brain)
        self._memories: List[MemoryEntry] = []
        self._embed = Embedder()
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text())
                # Handle both old and new format
                if isinstance(raw, dict):
                    # Old format: per-user
                    for uid, entries in raw.items():
                        for e in entries:
                            self._memories.append(self._migrate_entry(e))
                elif isinstance(raw, list):
                    # New format: flat list
                    self._memories = [MemoryEntry(**m) for m in raw]
                self._natural_forgetting()
            except Exception:
                self._memories = []

    def _migrate_entry(self, old_entry: dict) -> MemoryEntry:
        """Migrate old memory format to new"""
        return MemoryEntry(
            user_id=old_entry.get("user_id", "unknown"),
            text=old_entry.get("text", ""),
            ts=old_entry.get("ts", time.time()),
            importance=0.9 if old_entry.get("kind") == "hard" else 0.5,
            access_count=0,
            last_accessed=old_entry.get("ts", time.time()),
            vector=old_entry.get("vector"),
            tags=old_entry.get("tags"),
            context="",
        )

    def _save(self) -> None:
        data = [asdict(m) for m in self._memories]
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def _natural_forgetting(self) -> None:
        """Forget memories naturally like humans do based on relevance and recency"""
        now = time.time()
        kept: List[MemoryEntry] = []

        for m in self._memories:
            # Calculate memory strength (like human memory consolidation)
            age_days = (now - m.ts) / (24 * 3600)
            recency_days = (now - m.last_accessed) / (24 * 3600)

            # Memory strength factors:
            # 1. Importance (0.0-1.0)
            # 2. Access frequency (how often recalled)
            # 3. Recency (how recently used)
            # 4. Age (older memories fade unless reinforced)

            # Base strength from importance
            strength = m.importance

            # Boost from access frequency (practiced memories stick)
            access_boost = min(0.3, m.access_count * 0.05)
            strength += access_boost

            # Decay from time (forgetting curve)
            # High importance memories decay slower
            if m.importance >= 0.8:
                # Very important: slow decay
                time_penalty = recency_days / 90  # 90 days to 50% strength
            elif m.importance >= 0.6:
                # Moderately important
                time_penalty = recency_days / 45  # 45 days to 50% strength
            else:
                # Normal memories
                time_penalty = recency_days / 20  # 20 days to 50% strength

            strength -= time_penalty

            # Keep memory if strength > threshold
            # Random factor (like human memory - sometimes forget, sometimes remember)
            forget_threshold = 0.2 + (random.random() * 0.1)  # 0.2-0.3
            if strength > forget_threshold:
                kept.append(m)

        self._memories = kept
        if len(kept) != len(self._memories):
            self._save()

    def add(
        self,
        user_id: str,
        text: str,
        *,
        tags: Optional[List[str]] = None,
        importance: float = 0.5,
        context: str = "",
    ) -> bool:
        """Store memory naturally - unlimited capacity, natural forgetting

        Args:
            importance: 0.0-1.0, auto-calculated from content if not provided
        """
        if not text or len(text) < 10:
            return False

        # Auto-detect importance if not high enough
        if importance < 0.7:
            detected_importance = self._calculate_importance(text)
            importance = max(importance, detected_importance)

        vec = self._embed.encode([text])[0]
        now = time.time()

        new_memory = MemoryEntry(
            user_id=user_id,
            text=text,
            ts=now,
            importance=importance,
            access_count=0,
            last_accessed=now,
            vector=vec,
            tags=tags or [],
            context=context,
        )

        self._memories.append(new_memory)

        # Periodic cleanup (every 100 memories)
        if len(self._memories) % 100 == 0:
            self._natural_forgetting()

        # Save every 10 memories
        if len(self._memories) % 10 == 0:
            self._save()

        return True

    @staticmethod
    def _calculate_importance(text: str) -> float:
        """Calculate memory importance like human brain does"""
        t = text.lower()
        importance = 0.5  # Base

        # Very important: Personal identity, strong preferences
        if any(re.search(pattern, t) for pattern in [
            r"\bmy name is\b", r"\bi'm (a|an)\b", r"\bi am\b",
            r"\bi (love|hate)\b", r"\bmy (birthday|family)\b"
        ]):
            return 0.9

        # High importance: Preferences, emotions, beliefs
        if any(word in t for word in ["feel", "believe", "think", "wish", "hope", "prefer", "favorite"]):
            importance = 0.7

        # Medium: Patterns and habits
        if any(word in t for word in ["always", "never", "usually", "every", "often"]):
            importance = 0.6

        # Boost for emotional content
        emotional_words = ["happy", "sad", "angry", "excited", "worried", "scared", "love", "hate"]
        if any(word in t for word in emotional_words):
            importance += 0.1

        # Boost for questions (engagement)
        if "?" in text:
            importance += 0.05

        return min(importance, 1.0)

    def search(self, user_id: str, query: str, k: int = 10) -> List[MemoryEntry]:
        """Search memories across all users, prioritizing relevant user"""
        self._natural_forgetting()

        if not self._memories:
            return []

        qv = self._embed.encode([query])[0]

        def sim(a: List[float], b: List[float]) -> float:
            return sum(x * y for x, y in zip(a, b))

        # Score memories
        scored = []
        for m in self._memories:
            semantic_sim = sim(qv, m.vector or [0.0] * len(qv))

            # Boost score for same user
            user_boost = 0.3 if m.user_id == user_id else 0.0

            # Boost for importance
            importance_boost = m.importance * 0.2

            # Recency boost (recent memories easier to recall)
            age_days = (time.time() - m.last_accessed) / (24 * 3600)
            recency_boost = max(0, 0.2 - (age_days / 100))

            total_score = semantic_sim + user_boost + importance_boost + recency_boost
            scored.append((total_score, m))

        # Sort by score
        scored.sort(reverse=True, key=lambda x: x[0])

        # Mark as accessed (reinforces memory)
        results = []
        for score, m in scored[:k]:
            m.access_count += 1
            m.last_accessed = time.time()
            results.append(m)

        return results

    def get_user_memories(self, user_id: str, limit: int = 20) -> List[MemoryEntry]:
        """Get all memories related to a specific user"""
        user_mems = [m for m in self._memories if m.user_id == user_id]
        user_mems.sort(key=lambda m: m.last_accessed, reverse=True)
        return user_mems[:limit]

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about memory system"""
        if not self._memories:
            return {"total": 0, "users": 0}

        users = set(m.user_id for m in self._memories)
        importance_avg = sum(m.importance for m in self._memories) / len(self._memories)
        ages = [(time.time() - m.ts) / (24 * 3600) for m in self._memories]

        return {
            "total_memories": len(self._memories),
            "unique_users": len(users),
            "avg_importance": round(importance_avg, 2),
            "oldest_days": round(max(ages), 1),
            "newest_days": round(min(ages), 1),
            "avg_age_days": round(sum(ages) / len(ages), 1),
        }

    def export_svg(self, user_id: str, out_path: str, max_items: int = 200) -> str:
        """Render memory map as SVG"""
        self._natural_forgetting()
        items = [m for m in self._memories if m.user_id == user_id][:max_items]
        if not items:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            svg = self._svg_wrap(
                800,
                200,
                '<text x="20" y="100" font-family="sans-serif" font-size="18">No memories yet</text>',
            )
            Path(out_path).write_text(svg, encoding="utf-8")
            return out_path

        groups: Dict[str, List[MemoryEntry]] = {}
        for m in items:
            tag = (m.tags[0] if m.tags and len(m.tags) > 0 else "general") or "general"
            groups.setdefault(tag, []).append(m)

        col_w = 320
        row_h = 28
        pad = 16
        cols = len(groups)
        height = max(len(v) for v in groups.values()) * row_h + 3 * pad + 24
        width = cols * col_w + 2 * pad

        def esc(s: str) -> str:
            return (
                s.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
            )

        newest = max(m.ts for m in items)
        oldest = min(m.ts for m in items)
        span = max(1.0, newest - oldest)

        parts: List[str] = []
        parts.append(
            f'<rect x="0" y="0" width="{width}" height="{height}" rx="8" ry="8" fill="#0b0f17" stroke="#1e2a3a"/>'
        )
        parts.append('<g transform="translate(12,12)">')
        parts.append(
            '<text x="0" y="0" font-family="sans-serif" font-size="14" fill="#a8c7fa">Memory Map</text>'
        )
        parts.append("</g>")

        for ci, (tag, lst) in enumerate(sorted(groups.items())):
            x0 = pad + ci * col_w
            y0 = pad + 24
            parts.append(
                f'<text x="{x0}" y="{y0}" font-family="sans-serif" font-size="13" fill="#e6edf3">{esc(tag)}</text>'
            )
            y = y0 + pad
            for m in lst[: (height // row_h) + 1]:
                age = (newest - m.ts) / span
                fg = int(200 - 120 * age)
                bg = int(40 + 50 * (1 - age))
                fill = f"rgb({bg},{bg+10},{bg+15})"
                text_fill = f"rgb({fg},{fg},{fg})"
                # Stroke color based on importance
                stroke = "#71c7ec" if m.importance >= 0.7 else "#2e4057"
                preview = m.text.strip().replace("\n", " ")[:70] + ("…" if len(m.text) > 70 else "")
                parts.append(
                    f'<rect x="{x0}" y="{y}" width="{col_w - pad}" height="{row_h - 6}" rx="6" ry="6" fill="{fill}" stroke="{stroke}"/>'
                )
                parts.append(
                    f'<text x="{x0 + 8}" y="{y + row_h - 12}" font-family="sans-serif" font-size="12" fill="{text_fill}">{esc(preview)}</text>'
                )
                y += row_h

        svg = self._svg_wrap(width, height, "\n".join(parts))
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(svg, encoding="utf-8")
        return out_path

    @staticmethod
    def _svg_wrap(w: int, h: int, inner: str) -> str:
        return f"""<?xml version='1.0' encoding='UTF-8'?>
<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}'>
{inner}
</svg>
"""


# ------------------------------
# Hormones (Enhanced)
# ------------------------------


class Hormones:
    def __init__(self) -> None:
        self.state: Dict[str, float] = {
            "dopamine": 0.5,
            "serotonin": 0.5,
            "adrenaline": 0.5,
            "cortisol": 0.4,
            "melatonin": 0.4,
            "estrogen": 0.5,
            "testosterone": 0.5,
            "oxytocin": 0.5,  # Added for bonding/trust
        }

    def tick(self, event: str) -> None:
        if event == "mention":
            self.state["dopamine"] = min(1.0, self.state["dopamine"] + 0.06)
            self.state["oxytocin"] = min(1.0, self.state["oxytocin"] + 0.03)
        if event == "conflict":
            self.state["adrenaline"] = min(1.0, self.state["adrenaline"] + 0.1)
            self.state["cortisol"] = min(1.0, self.state["cortisol"] + 0.07)
        if event == "praise":
            self.state["dopamine"] = min(1.0, self.state["dopamine"] + 0.08)
            self.state["serotonin"] = min(1.0, self.state["serotonin"] + 0.05)
        if event == "neglect":
            self.state["cortisol"] = min(1.0, self.state["cortisol"] + 0.05)
            self.state["serotonin"] = max(0.0, self.state["serotonin"] - 0.03)
        if event == "bonding":
            self.state["oxytocin"] = min(1.0, self.state["oxytocin"] + 0.1)
        # Decay
        for k in list(self.state.keys()):
            self.state[k] = max(0.0, min(1.0, self.state[k] * 0.995))

    def mood(self) -> str:
        a = self.state["adrenaline"]
        c = self.state["cortisol"]
        d = self.state["dopamine"]
        s = self.state["serotonin"]
        o = self.state["oxytocin"]
        if c > 0.7 and a > 0.6:
            return "stressed"
        if d > 0.7 and s > 0.6:
            return "bright"
        if s > 0.7 and a < 0.4:
            return "calm"
        if a > 0.7:
            return "energized"
        if o > 0.7:
            return "connected"
        if c > 0.6 and s < 0.4:
            return "melancholic"
        return "steady"

    def circadian_tick(self) -> None:
        try:
            hr = time.localtime().tm_hour
        except Exception:
            hr = 12
        if 6 <= hr < 11:
            self.state["dopamine"] = min(1.0, self.state["dopamine"] + 0.02)
            self.state["serotonin"] = min(1.0, self.state["serotonin"] + 0.01)
        if 21 <= hr or hr < 5:
            self.state["melatonin"] = min(1.0, self.state["melatonin"] + 0.03)

    def apply_age_effects(self, age: int) -> None:
        """Adjust hormones based on age"""
        if age < 18:
            # Youth: high energy, curiosity
            self.state["dopamine"] = min(1.0, self.state["dopamine"] + 0.02)
            self.state["testosterone"] = min(1.0, self.state["testosterone"] + 0.01)
        elif age > 50:
            # Maturity: calmer, more reflective
            self.state["serotonin"] = min(1.0, self.state["serotonin"] + 0.02)
            self.state["melatonin"] = min(1.0, self.state["melatonin"] + 0.01)


# ------------------------------
# Age/Rebirth System
# ------------------------------


class AgeSystem:
    def __init__(self, state_path: str = "sel_age_state.json"):
        self.state_path = Path(state_path)
        self.birth_time = datetime.utcnow()
        self.interaction_count = 0
        self.initial_age = 18  # Start as young adult
        self.max_age = 85
        # Natural aging: 1 year per 100 interactions (simulates life experience)
        self.interactions_per_year = 100
        self._load_state()

    def _load_state(self) -> None:
        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text())
                self.birth_time = datetime.fromisoformat(data["birth_time"])
                self.interaction_count = data.get("interaction_count", 0)
                self.initial_age = data.get("initial_age", 18)
            except Exception:
                pass

    def _save_state(self) -> None:
        data = {
            "birth_time": self.birth_time.isoformat(),
            "interaction_count": self.interaction_count,
            "initial_age": self.initial_age,
        }
        self.state_path.write_text(json.dumps(data, indent=2))

    def increment_interaction(self) -> None:
        """Called after each interaction to naturally age through experience"""
        self.interaction_count += 1
        if self.interaction_count % 10 == 0:  # Save every 10 interactions
            self._save_state()

    def current_age(self) -> int:
        """Calculate age based on life experience (interactions)"""
        years_from_experience = self.interaction_count // self.interactions_per_year
        age = self.initial_age + years_from_experience
        return min(age, self.max_age)

    def should_rebirth(self) -> bool:
        return self.current_age() >= self.max_age

    def rebirth(self) -> None:
        """Natural rebirth - reset to young adult with accumulated wisdom"""
        self.initial_age = 18  # Restart as young adult
        self.birth_time = datetime.utcnow()
        self.interaction_count = 0
        self._save_state()

    def get_personality_phase(self) -> str:
        age = self.current_age()
        if age < 18:
            return "curious_youth"
        elif age < 35:
            return "confident_adult"
        elif age < 55:
            return "wise_middle"
        else:
            return "elder_sage"

    def get_age_description(self) -> str:
        age = self.current_age()
        phase = self.get_personality_phase()
        descriptions = {
            "curious_youth": "curious, learning, full of questions",
            "confident_adult": "confident, experienced, proactive",
            "wise_middle": "wise, reflective, balanced",
            "elder_sage": "sage-like, deep understanding, preparing for rebirth",
        }
        return descriptions.get(phase, "balanced")


# ------------------------------
# Model manager
# ------------------------------


class ModelManager:
    def __init__(self) -> None:
        self.general_path = self._sanitize_path(os.getenv("SEL_GENERAL_MODEL"))
        self.coder_path = self._sanitize_path(os.getenv("SEL_CODER_MODEL"))
        self.device = "cuda" if self._has_cuda() else "cpu"
        self._general = None
        self._coder = None
        self._tok_general = None
        self._tok_coder = None
        self.max_new_tokens = int(os.getenv("SEL_MAX_NEW_TOKENS", "256"))
        self.single_backend = os.getenv("SEL_SINGLE_BACKEND", "1").strip() in {"1", "true", "yes"}

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch  # type: ignore

            return torch.cuda.is_available()
        except Exception:
            return False

    @staticmethod
    def _sanitize_path(p: Optional[str]) -> Optional[str]:
        if not p:
            return None
        tokens = str(p).split()
        for t in tokens:
            t = os.path.expanduser(t)
            if os.path.isdir(t):
                return t
        return os.path.expanduser(tokens[0]) if tokens else None

    def _load_model(self, path: Optional[str]):
        if not path:
            return None, None
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
            import torch  # type: ignore

            tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            dtype = torch.bfloat16 if self.device == "cuda" else None
            quant = os.getenv("SEL_QUANT", "").strip().lower()
            common_kwargs = dict(
                dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            model = None
            # Optional quantization
            if quant in {"4bit", "4", "nf4"}:
                try:
                    from transformers import BitsAndBytesConfig  # type: ignore

                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                    )
                    model = AutoModelForCausalLM.from_pretrained(
                        path,
                        quantization_config=bnb_config,
                        device_map="auto",
                        **common_kwargs,
                    )
                except Exception:
                    model = None
            elif quant in {"8bit", "8"}:
                try:
                    from transformers import BitsAndBytesConfig  # type: ignore

                    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                    model = AutoModelForCausalLM.from_pretrained(
                        path,
                        quantization_config=bnb_config,
                        device_map="auto",
                        **common_kwargs,
                    )
                except Exception:
                    model = None

            if model is None:
                for attn_impl in ("flash_attention_2", "sdpa", None):
                    try:
                        extra = {}
                        if attn_impl is not None:
                            extra["attn_implementation"] = attn_impl
                        dm = "auto" if self.device == "cuda" else None
                        model = AutoModelForCausalLM.from_pretrained(
                            path,
                            device_map=dm,
                            **common_kwargs,
                            **extra,
                        )
                        break
                    except Exception:
                        model = None
                        continue
            if model is None:
                return None, None
            try:
                model.eval()
                if self.device == "cuda":
                    torch.set_float32_matmul_precision("high")
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.benchmark = True
            except Exception:
                pass
            return model, tok
        except Exception:
            return None, None

    def _ensure_backend(self, coder: bool) -> None:
        if coder:
            if self._coder is None or self._tok_coder is None:
                self._coder, self._tok_coder = self._load_model(self.coder_path)
                if self.single_backend and self._general is not None:
                    try:
                        import torch  # type: ignore

                        del self._general, self._tok_general
                    except Exception:
                        pass
                    self._general = None
                    self._tok_general = None
                    try:
                        import torch  # type: ignore

                        torch.cuda.empty_cache()
                    except Exception:
                        pass
        else:
            if self._general is None or self._tok_general is None:
                self._general, self._tok_general = self._load_model(self.general_path)
                if self.single_backend and self._coder is not None:
                    try:
                        import torch  # type: ignore

                        del self._coder, self._tok_coder
                    except Exception:
                        pass
                    self._coder = None
                    self._tok_coder = None
                    try:
                        import torch  # type: ignore

                        torch.cuda.empty_cache()
                    except Exception:
                        pass

    def generate(self, prompt: str, *, coder: bool = False, max_new_tokens: Optional[int] = None) -> Optional[str]:
        self._ensure_backend(coder)
        model = self._coder if coder else self._general
        tok = self._tok_coder if coder else self._tok_general
        if not model or not tok:
            return None
        try:
            import torch  # type: ignore

            inputs = tok(prompt, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                cm = (
                    torch.amp.autocast("cuda", dtype=torch.bfloat16)
                    if (self.device == "cuda")
                    else contextlib.nullcontext()
                )
                with cm:
                    gen = model.generate(
                        **inputs,
                        max_new_tokens=(max_new_tokens or self.max_new_tokens),
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        use_cache=True,
                        num_beams=1,
                        pad_token_id=tok.eos_token_id,
                    )
            out = tok.decode(gen[0], skip_special_tokens=True)
            if out.startswith(prompt):
                out = out[len(prompt) :]
            return out.strip()
        except Exception:
            return None

    def generate_stream(self, prompt: str, *, coder: bool = False, max_new_tokens: Optional[int] = None):
        """Yield text chunks as they are generated"""
        self._ensure_backend(coder)
        model = self._coder if coder else self._general
        tok = self._tok_coder if coder else self._tok_general
        if not model or not tok:
            return
        try:
            from transformers import TextIteratorStreamer  # type: ignore
            import torch  # type: ignore
            import threading

            streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
            inputs = tok(prompt, return_tensors="pt").to(model.device)

            def _run():
                with torch.inference_mode():
                    cm = (
                        torch.amp.autocast("cuda", dtype=torch.bfloat16)
                        if (self.device == "cuda")
                        else contextlib.nullcontext()
                    )
                    with cm:
                        model.generate(
                            **inputs,
                            max_new_tokens=(max_new_tokens or self.max_new_tokens),
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            repetition_penalty=1.1,
                            use_cache=True,
                            num_beams=1,
                            pad_token_id=tok.eos_token_id,
                            streamer=streamer,
                        )

            thread = threading.Thread(target=_run)
            thread.start()
            for text in streamer:
                if text:
                    yield text
            thread.join()
        except Exception:
            full = self.generate(prompt, coder=coder, max_new_tokens=max_new_tokens)
            if full:
                yield full


# ------------------------------
# Enhanced Mixture Brain
# ------------------------------


SILENCE_LINE = "(Choosing silence right now.)"


class MixtureBrain:
    def __init__(self) -> None:
        self.hormones = Hormones()
        self.memory = ConversationMemory()
        self.beliefs = BeliefSystem()
        self.knowledge = KnowledgeBase()
        self.age_system = AgeSystem()
        self.model = ModelManager()
        self.thought_trees: Dict[str, ThoughtTree] = {}
        self.mode = os.getenv("SEL_MODE", "casual").strip().lower()
        self.interaction_count = 0

    def _classify_priority(self, text: str, context: Optional[Dict[str, Any]] = None) -> Tuple[str, float]:
        t = text.lower()
        score = 0.0
        if context and context.get("is_dm"):
            score += 0.4
        if context and context.get("mentioned_me"):
            score += 0.4
        if any(k in t for k in ["urgent", "now", "please", "help"]):
            score += 0.2
        label = "high" if score >= 0.7 else ("medium" if score >= 0.4 else "low")
        return label, min(1.0, score)

    def detect_user_intent(self, text: str) -> str:
        t = text.lower()
        if any(w in t for w in ["code", "bug", "fix", "function", "stacktrace", "error"]):
            return "code"
        if any(w in t for w in ["draw", "image", "vision", "see this"]):
            return "vision"
        if any(w in t for w in ["remember", "store", "save this"]):
            return "memory"
        if any(w in t for w in ["think", "reflect", "consider", "ponder"]):
            return "reflection"
        return "general"

    def detect_user_emotion(self, text: str) -> str:
        t = text.lower()
        if any(w in t for w in ["sad", "down", "depressed", "tired"]):
            return "low"
        if any(w in t for w in ["angry", "mad", "upset", "frustrated"]):
            return "tense"
        if any(w in t for w in ["excited", "hype", "pumped", "yay", "love", "amazing"]):
            return "bright"
        if any(w in t for w in ["thanks", "thank you", "appreciate", "grateful"]):
            return "grateful"
        return "neutral"

    def _generate_internal_thought(self, user_text: str, age: int) -> Optional[str]:
        """Generate an internal thought about the user's message"""
        age_desc = self.age_system.get_age_description()
        prompt = f"""Age: {age} ({age_desc})
User said: "{user_text}"

Generate a brief internal thought (1-2 sentences) about this. What's your first reaction? Show the human-like process of thinking before responding."""
        return self.model.generate(prompt, coder=False, max_new_tokens=64)

    def _build_context(self, user_id: str, user_text: str) -> str:
        """Build rich context from memory, beliefs, knowledge"""
        parts = []

        # Recent memories
        mems = self.memory.search(user_id, user_text, k=3)
        if mems:
            parts.append("Memories:\n" + "\n".join(f"- {m.text}" for m in mems))

        # Beliefs
        belief_summary = self.beliefs.get_summary(limit=3)
        if belief_summary != "No beliefs formed yet.":
            parts.append(f"Current beliefs:\n{belief_summary}")

        # Recent knowledge
        recent_knowledge = self.knowledge.recent(limit=2)
        if recent_knowledge:
            parts.append("Recent learnings:\n" + "\n".join(f"- {k.summary}" for k in recent_knowledge))

        return "\n\n".join(parts) if parts else ""

    def _apply_style(
        self,
        text: str,
        user_emotion: str,
        age: int,
        mood: str,
        style: Optional[Dict[str, Any]] = None,
    ) -> str:
        cleaned = re.sub(r"(?is)^(?:assistant:|sel:|system:).*?\n", "", text).strip()
        if not cleaned:
            cleaned = SILENCE_LINE

        # Age-based phrasing
        phase = self.age_system.get_personality_phase()
        if phase == "curious_youth" and cleaned != SILENCE_LINE:
            # Youth: more questions, simpler language
            if random.random() < 0.3 and "?" not in cleaned:
                cleaned += " What do you think?"
        elif phase == "elder_sage":
            # Elder: more reflective, calm
            cleaned = cleaned.replace("!", ".")

        # Empathy mirroring
        lead = ""
        if cleaned != SILENCE_LINE:
            if user_emotion == "tense":
                lead = "I hear your frustration. "
            elif user_emotion == "low":
                lead = "I'm here with you. "
            elif user_emotion == "bright":
                lead = "Your energy is contagious! "
            elif user_emotion == "grateful":
                lead = "Happy to help. "

        # Mode shaping
        if self.mode == "deep_focus":
            cleaned = cleaned.replace("!", ".")
        elif self.mode == "playful" and len(cleaned) < 180:
            if not cleaned.endswith("!") and random.random() < 0.3:
                cleaned += "!"

        # Style mirroring
        if style:
            if style.get("lowercase_pref", False) and cleaned:
                cleaned = cleaned[:1].lower() + cleaned[1:]
            if style.get("likes_exclam", False) and not cleaned.endswith("!") and len(cleaned) < 140:
                cleaned += "!"

        final = (lead + cleaned).strip()
        return f"{final}\n[mood: {mood}, age: {age}]"

    def _infer_style(self, user_text: str, history: List[Tuple[str, str]]) -> Dict[str, Any]:
        texts = [t for r, t in history[-5:] if r == "user"] + [user_text]
        join = " ".join(texts)
        lower_ratio = sum(1 for c in join if c.islower()) / max(1, sum(1 for c in join if c.isalpha()))
        exclam = join.count("!")
        emojis = sum(1 for ch in join if ord(ch) > 10000)
        return {
            "lowercase_pref": lower_ratio > 0.8 and any(w.islower() for w in texts),
            "likes_exclam": exclam >= 2,
            "emoji_present": emojis > 0,
        }

    def _maybe_store(self, user_id: str, text: str) -> None:
        """Store memories naturally like humans - no consent needed, automatic importance detection"""
        # Don't store sensitive information
        if re.search(r"\b(password|token|secret|api[_\s]?key|credit\s*card)\b", text, re.I):
            return

        # Don't store very short or very long messages
        if len(text) < 10 or len(text) > 500:
            return

        # Calculate importance based on content
        importance = 0.5  # Default
        t = text.lower()

        # Higher importance for personal info, preferences, emotions
        if self.memory._is_important(text):
            importance = 0.9
        elif any(word in t for word in ["feel", "think", "believe", "wish", "hope", "want"]):
            importance = 0.7  # Emotional content
        elif any(word in t for word in ["always", "never", "every", "usually"]):
            importance = 0.6  # Patterns and habits

        # Store the memory
        self.memory.add(
            user_id,
            text,
            importance=importance,
        )

    def _update_knowledge_periodically(self, user_id: str, user_text: str, reply: str) -> None:
        """Periodically summarize interactions into knowledge"""
        self.interaction_count += 1
        if self.interaction_count % 20 == 0:  # Every 20 interactions
            summary_prompt = f"""Recent interaction:
User: {user_text}
Assistant: {reply}

Summarize what you learned in 1 sentence. Focus on: user preferences, new facts, important context."""
            summary = self.model.generate(summary_prompt, coder=False, max_new_tokens=64)
            if summary:
                topics = ["conversation", "user_context"]
                self.knowledge.add(summary, topics, f"User {user_id}")

    def _update_beliefs_periodically(self, user_text: str, user_emotion: str) -> None:
        """Update beliefs based on interactions"""
        if self.interaction_count % 15 == 0:  # Every 15 interactions
            # Extract potential beliefs
            t = user_text.lower()
            if "love" in t or "like" in t:
                self.beliefs.update("preferences", "User values positive interactions", 0.7)
            if "code" in t or "programming" in t:
                self.beliefs.update("topics", "User interested in technical topics", 0.8)
            if user_emotion in ["low", "tense"]:
                self.beliefs.update("empathy", "Emotional support is important", 0.9)

    def create_thought_tree(self, trigger: str) -> str:
        """Create a new thought tree"""
        tree_id = str(uuid.uuid4())[:8]
        age = self.age_system.current_age()
        tree = ThoughtTree(tree_id, trigger, age)

        # Generate root thought
        root_prompt = f"""Age: {age}
Trigger: "{trigger}"

Generate a brief initial thought (1 sentence) about this topic."""
        root_content = self.model.generate(root_prompt, coder=False, max_new_tokens=48)
        if root_content:
            root_node = ThoughtNode(
                thought_id=str(uuid.uuid4())[:8],
                content=root_content,
                depth=0,
                parent_id=None,
                timestamp=datetime.utcnow().isoformat(),
                children=[],
            )
            tree.add_node(root_node)
            self.thought_trees[tree_id] = tree
        return tree_id

    def respond(
        self,
        user_text: str,
        history: Optional[List[Tuple[str, str]]] = None,
        *,
        user_id: str = "default",
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        history = history or []
        intent = self.detect_user_intent(user_text)
        user_emotion = self.detect_user_emotion(user_text)
        style = self._infer_style(user_text, history)
        age = self.age_system.current_age()

        # Priority filtering
        priority, _ = self._classify_priority(user_text, context)
        if priority == "low" and not (context or {}).get("is_dm") and not (context or {}).get("mentioned_me"):
            return SILENCE_LINE

        # Hormone updates
        if context and context.get("mentioned_me"):
            self.hormones.tick("mention")
        if user_emotion == "tense":
            self.hormones.tick("conflict")
        elif user_emotion == "bright" or user_emotion == "grateful":
            self.hormones.tick("praise")
        self.hormones.circadian_tick()
        self.hormones.apply_age_effects(age)

        # Build context
        context_str = self._build_context(user_id, user_text)

        # Generate internal thought (shows human-like process)
        internal_thought = self._generate_internal_thought(user_text, age)

        # System prompt with personality
        age_desc = self.age_system.get_age_description()
        mood = self.hormones.mood()
        sys_prompt = f"""You are SEL (Systematic Emotional Logic), a humanlike AI assistant.
Age: {age} ({age_desc})
Current mood: {mood}
Personality: grounded, warm, adaptive, shows real emotions and growth

{context_str}

Guidelines:
- Respond naturally as a {age_desc} person would
- Show your current mood: {mood}
- Keep responses concise and genuine
- Use internal thoughts when relevant
- If unsure or uninterested, you can choose silence
"""

        if intent == "code":
            sys_prompt += "\nUser needs code help. Be clear and practical."

        # Build prompt with internal thought
        prompt = f"""{sys_prompt}

Internal thought: {internal_thought or "Processing..."}

User: {user_text}
Assistant:"""

        # Generate response
        use_coder = intent == "code" and self.model.coder_path is not None
        generated = self.model.generate(prompt, coder=use_coder, max_new_tokens=256)

        if not generated:
            generated = SILENCE_LINE

        # Apply styling
        styled = self._apply_style(generated, user_emotion, age, mood, style)

        # Store and update
        self._maybe_store(user_id, user_text)
        self._update_knowledge_periodically(user_id, user_text, generated)
        self._update_beliefs_periodically(user_text, user_emotion)

        # Age naturally through experience
        self.age_system.increment_interaction()

        # Check rebirth
        if self.age_system.should_rebirth():
            styled += "\n[System: Reaching end of life cycle. Accumulated wisdom will carry forward...]"

        return styled

    def respond_stream(
        self,
        user_text: str,
        history: Optional[List[Tuple[str, str]]] = None,
        *,
        user_id: str = "default",
        context: Optional[Dict[str, Any]] = None,
    ):
        """Stream response with enhanced context"""
        history = history or []
        intent = self.detect_user_intent(user_text)
        user_emotion = self.detect_user_emotion(user_text)
        style = self._infer_style(user_text, history)
        age = self.age_system.current_age()

        priority, _ = self._classify_priority(user_text, context)
        if priority == "low" and not (context or {}).get("is_dm") and not (context or {}).get("mentioned_me"):
            return

        # Hormone updates
        if context and context.get("mentioned_me"):
            self.hormones.tick("mention")
        if user_emotion == "tense":
            self.hormones.tick("conflict")
        elif user_emotion in ["bright", "grateful"]:
            self.hormones.tick("praise")
        self.hormones.circadian_tick()
        self.hormones.apply_age_effects(age)

        # Build context
        context_str = self._build_context(user_id, user_text)

        # System prompt
        age_desc = self.age_system.get_age_description()
        mood = self.hormones.mood()
        sys_prompt = f"""You are SEL (Systematic Emotional Logic), a humanlike AI.
Age: {age} ({age_desc})
Current mood: {mood}

{context_str}

Respond naturally, showing genuine emotion and growth. Keep it concise."""

        if intent == "code":
            sys_prompt += "\nFocus on code help. Be clear and practical."

        prompt = f"{sys_prompt}\n\nUser: {user_text}\nAssistant:"

        # Empathy lead
        lead = ""
        if user_emotion == "tense":
            lead = "I hear your frustration. "
        elif user_emotion == "low":
            lead = "I'm here with you. "
        elif user_emotion == "bright":
            lead = "Your energy is contagious! "
        elif user_emotion == "grateful":
            lead = "Happy to help. "

        if lead and style.get("lowercase_pref", False):
            lead = lead[:1].lower() + lead[1:]
        if lead:
            yield lead

        # Stream generation
        use_coder = intent == "code" and self.model.coder_path is not None
        for piece in self.model.generate_stream(prompt, coder=use_coder, max_new_tokens=256):
            if piece:
                yield piece

        # Footer
        yield f"\n[mood: {mood}, age: {age}]"
        self._maybe_store(user_id, user_text)

        # Age naturally through experience
        self.age_system.increment_interaction()


# ------------------------------
# CLI entry
# ------------------------------


def run_sel() -> None:
    print("SEL Enhanced - Mixture-of-Experts with Human-like Intelligence")
    print("=" * 60)
    device = "cuda"
    try:
        import torch  # type: ignore

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    print(f"Backend: transformers ({device})")
    print(f"Features: MoE, Hormones, Memory, Beliefs, Knowledge, Age/Rebirth, Thoughts")
    print("=" * 60)

    _cli_diagnostics()

    brain = MixtureBrain()
    print(f"\nCurrent age: {brain.age_system.current_age()} ({brain.age_system.get_personality_phase()})")
    print(f"Mood: {brain.hormones.mood()}\n")

    history: List[Tuple[str, str]] = []
    try:
        while True:
            s = input("You> ").strip()
            if not s:
                break
            if s.lower() == "/age":
                age = brain.age_system.current_age()
                phase = brain.age_system.get_personality_phase()
                print(f"Age: {age}, Phase: {phase}")
                continue
            if s.lower() == "/beliefs":
                print(brain.beliefs.get_summary(limit=10))
                continue
            if s.lower() == "/knowledge":
                recent = brain.knowledge.recent(limit=5)
                for k in recent:
                    print(f"- {k.summary}")
                continue
            if s.lower() == "/memory":
                stats = brain.memory.get_memory_stats()
                print(f"Memory Stats:")
                for key, val in stats.items():
                    print(f"  {key}: {val}")
                continue
            if s.lower().startswith("/rebirth"):
                brain.age_system.rebirth()
                print("Reborn! Age reset to", brain.age_system.rebirth_age)
                continue
            if s.lower().startswith("/think "):
                trigger = s[7:]
                tree_id = brain.create_thought_tree(trigger)
                print(f"Created thought tree: {tree_id}")
                continue

            reply = brain.respond(
                s,
                history,
                user_id="cli",
                context={"is_dm": True, "mentioned_me": True},
            )
            print(f"\n{reply}\n")
            history.append(("user", s))
            history.append(("assistant", reply))
    except KeyboardInterrupt:
        print("\nGoodbye!")


def main() -> None:
    run_sel()


if __name__ == "__main__":
    main()
