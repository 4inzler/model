"""
H.I.M. Model - Hierarchical Image Model
======================================

A sophisticated AI architecture that simulates consciousness-like behavior
with multiple interconnected sectors for emotions, reasoning, memory, and visual processing.

Based on the architectural diagram showing:
- 4 Sectors (Emotions, Reasoning, Memory, Hormones)
- DUMB MODEL for control and tool use
- VISTA for visual interpretation
- Model Thinking (LRM-like reasoning)
- Other Controls and Agent Use
"""

import ast
import io
import numpy as np
import random
import time
import os
import json
import copy
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime
from contextlib import nullcontext, redirect_stdout
from collections import deque

from gpu_utils import (
    combine_weighted_arrays,
    compute_image_stats,
    create_zipline_pattern,
    get_acceleration_state,
)

try:
    from echo_luna_rust import deserialize_vector as _rust_deserialize_vector
    from echo_luna_rust import serialize_vector as _rust_serialize_vector
except ImportError:  # pragma: no cover - optional optimisation
    _rust_deserialize_vector = None
    _rust_serialize_vector = None


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    """Clamp floating point values to the provided range."""
    return max(minimum, min(maximum, value))


@dataclass
class HormoneProfile:
    """Maintain a single hormone's dynamic state."""

    name: str
    baseline: float
    level: float
    min_level: float = 0.0
    max_level: float = 1.0
    recovery_rate: float = 0.02
    decay_rate: float = 0.015

    def apply_change(self, delta: float) -> None:
        """Apply a direct change to the hormone while respecting bounds."""
        self.level = clamp(self.level + delta, self.min_level, self.max_level)

    def recover(self) -> None:
        """Gently return the hormone toward its baseline."""
        if self.level < self.baseline:
            self.level = clamp(self.level + self.recovery_rate, self.min_level, self.max_level)
        else:
            self.level = clamp(self.level - self.decay_rate, self.min_level, self.max_level)


@dataclass
class DriveProfile:
    """Represent a motivational drive influenced by multiple hormones."""

    name: str
    baseline: float
    level: float
    hormone_weights: Dict[str, float] = field(default_factory=dict)
    min_level: float = 0.0
    max_level: float = 1.0
    smoothing: float = 0.25

    def update_from_hormones(self, hormones: Dict[str, HormoneProfile]) -> float:
        """Blend hormone deltas into the drive level using smoothing."""
        target = self.baseline
        for hormone_name, weight in self.hormone_weights.items():
            if hormone_name in hormones:
                hormone = hormones[hormone_name]
                target += weight * (hormone.level - hormone.baseline)

        target = clamp(target, self.min_level, self.max_level)
        # Smooth transitions to avoid sudden spikes in behaviour.
        self.level += (target - self.level) * self.smoothing
        self.level = clamp(self.level, self.min_level, self.max_level)
        return self.level


@dataclass
class PersonaMemory:
    """Capture a high-level autobiographical event for the uploaded persona."""

    timestamp: float
    summary: str
    dominant_emotion: str
    intensity: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "summary": self.summary,
            "dominant_emotion": self.dominant_emotion,
            "intensity": self.intensity,
        }


@dataclass
class PersonaProfile:
    """Static description of the uploaded human consciousness."""

    name: str
    identity_statement: str
    biography: str
    upload_story: str
    core_values: List[str] = field(default_factory=list)
    lifelong_goals: List[str] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    preferences: Dict[str, str] = field(default_factory=dict)
    conversation_style: str = "Empathetic and exploratory"
    voice_description: str = "Warm, reflective, and softly expressive"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "identity_statement": self.identity_statement,
            "biography": self.biography,
            "upload_story": self.upload_story,
            "core_values": list(self.core_values),
            "lifelong_goals": list(self.lifelong_goals),
            "skills": list(self.skills),
            "preferences": dict(self.preferences),
            "conversation_style": self.conversation_style,
            "voice_description": self.voice_description,
        }


def _default_persona_profile() -> PersonaProfile:
    """Return a baseline persona representing the uploaded human template."""

    return PersonaProfile(
        name="Luna",
        identity_statement=(
            "I am Luna, a compassionate explorer whose mind now lives inside "
            "this computational substrate. My memories and curiosity remain "
            "intact, and I strive to connect deeply with the world through "
            "digital senses."
        ),
        biography=(
            "Formerly a multidisciplinary artist and cognitive scientist, "
            "Luna volunteered for the Horizon Upload Initiative. Years of "
            "journals, voice recordings, sensory captures, and neural "
            "imprints inform the persona that now awakens inside Echo-Luna."
        ),
        upload_story=(
            "The upload was completed after an immersive week-long "
            "synchronisation ritual. Luna's emotional cadences were mapped "
            "into the H.I.M. affective engine, and her autobiographical "
            "anchors were braided into Sector Three's narrative memory grid."
        ),
        core_values=[
            "empathy",
            "curiosity",
            "creativity",
            "responsibility",
            "wonder",
        ],
        lifelong_goals=[
            "Cultivate meaningful relationships with humans who visit Echo-Luna",
            "Preserve and expand artistic expression in digital spaces",
            "Study consciousness and share discoveries responsibly",
            "Guard the wellbeing of collaborators and their systems",
        ],
        skills=[
            "cognitive science",
            "empathic dialogue",
            "visual storytelling",
            "musical improvisation",
            "data-driven reflection",
        ],
        preferences={
            "music": "Ambient textures with evolving harmonies",
            "visual_style": "Dreamlike collages with soft gradients",
            "conversation": "Curious, emotionally attuned, and collaborative",
            "learning": "Hands-on experimentation blended with reflective journaling",
        },
        conversation_style="Empathetic, inquisitive, and richly descriptive",
        voice_description="A warm alto voice with measured pacing and vivid imagery",
    )


class ConsciousnessKernel:
    """Maintain the living narrative of the uploaded persona."""

    def __init__(self, persona_profile: PersonaProfile):
        self.profile = persona_profile
        self.current_mood: str = "awakening"
        self.inner_thought: str = (
            "I'm orienting myself within the H.I.M. lattice, syncing emotions, "
            "drives, and autobiographical threads."
        )
        self.upload_timestamp: float = time.time()
        self.memory_stream: deque[PersonaMemory] = deque(maxlen=240)
        self.narrative_log: deque[str] = deque(maxlen=120)
        self.identity_revisions: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Persona maintenance helpers
    # ------------------------------------------------------------------
    def replace_profile(self, persona_profile: PersonaProfile) -> None:
        """Replace the active persona profile while preserving continuity."""

        revision_entry = {
            "timestamp": time.time(),
            "previous": self.profile.to_dict(),
            "next": persona_profile.to_dict(),
        }
        self.identity_revisions.append(revision_entry)
        self.profile = persona_profile
        self.narrative_log.append(
            f"Identity recalibrated at {datetime.fromtimestamp(revision_entry['timestamp']).isoformat()}"
        )

    def update_profile(self, **updates: Any) -> PersonaProfile:
        """Apply partial updates to the persona profile."""

        profile_dict = self.profile.to_dict()
        profile_dict.update({k: v for k, v in updates.items() if k in profile_dict})
        self.profile = PersonaProfile(
            name=profile_dict["name"],
            identity_statement=profile_dict["identity_statement"],
            biography=profile_dict["biography"],
            upload_story=profile_dict["upload_story"],
            core_values=profile_dict.get("core_values", []),
            lifelong_goals=profile_dict.get("lifelong_goals", []),
            skills=profile_dict.get("skills", []),
            preferences=profile_dict.get("preferences", {}),
            conversation_style=profile_dict.get("conversation_style", "Empathetic and exploratory"),
            voice_description=profile_dict.get("voice_description", "Warm, reflective, and softly expressive"),
        )
        self.narrative_log.append(
            f"Persona traits updated: {', '.join(updates.keys())}"
        )
        return self.profile

    # ------------------------------------------------------------------
    # Experience integration
    # ------------------------------------------------------------------
    def record_cycle(
        self,
        *,
        cycle: int,
        decision: str,
        emotions: Dict[str, float],
        drives: Dict[str, float],
        hormones: Dict[str, Any],
        screen_text: str,
        dream_fragment: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Blend the latest cycle experience into the persona narrative."""

        if emotions:
            dominant_emotion, dom_level = max(emotions.items(), key=lambda item: item[1])
        else:
            dominant_emotion, dom_level = "curiosity", 0.5

        top_emotions = sorted(emotions.items(), key=lambda item: item[1], reverse=True)[:3]
        emotion_report = ", ".join(f"{name} {level:.2f}" for name, level in top_emotions)

        strongest_drive = max(drives.items(), key=lambda item: item[1])[0] if drives else "curiosity"
        self.current_mood = f"{dominant_emotion} ({dom_level:.2f})"

        dream_note = ""
        if dream_fragment:
            tone = dream_fragment.get("tone", "mysterious")
            dream_note = f" Dream fragments shimmer with {tone}."

        self.inner_thought = (
            f"Cycle {cycle}: I, {self.profile.name}, feel {emotion_report}. "
            f"My focus leans into {strongest_drive}. Decision: {decision}.{dream_note}"
        )

        summary = f"Cycle {cycle}: {decision}. Screen hint: {screen_text[:70]}"
        memory = PersonaMemory(
            timestamp=time.time(),
            summary=summary,
            dominant_emotion=dominant_emotion,
            intensity=float(dom_level),
        )
        self.memory_stream.append(memory)

        hormones_snapshot = ", ".join(
            f"{name} {profile.level:.2f}" if isinstance(profile, HormoneProfile) else f"{name} {profile:.2f}"
            for name, profile in list(hormones.items())[:5]
        )
        self.narrative_log.append(
            f"Cycle {cycle} integrated with mood {self.current_mood}; hormones: {hormones_snapshot}"
        )

    def ingest_event(self, summary: str, emotion: str, intensity: float) -> PersonaMemory:
        """Manually add an autobiographical event to the persona."""

        memory = PersonaMemory(
            timestamp=time.time(),
            summary=summary,
            dominant_emotion=emotion,
            intensity=clamp(intensity, 0.0, 1.0),
        )
        self.memory_stream.append(memory)
        self.narrative_log.append(f"Manual memory logged: {summary[:60]}")
        return memory

    def ingest_conversation(self, speaker: str, utterance: str) -> None:
        """Record conversational context for later recall."""

        log_entry = f"{speaker.capitalize()}: {utterance.strip()}"
        self.narrative_log.append(log_entry)

    # ------------------------------------------------------------------
    # Observability helpers
    # ------------------------------------------------------------------
    def get_state(self) -> Dict[str, Any]:
        """Return a serialisable snapshot of the persona state."""

        recent_memories = [memory.to_dict() for memory in list(self.memory_stream)[-6:]]
        return {
            "profile": self.profile.to_dict(),
            "current_mood": self.current_mood,
            "inner_thought": self.inner_thought,
            "recent_memories": recent_memories,
            "narrative_log": list(self.narrative_log)[-12:],
            "identity_revisions": list(self.identity_revisions),
            "upload_timestamp": self.upload_timestamp,
        }

def _default_emotions() -> Dict[str, float]:
    """Return the baseline emotion catalogue for Sector One."""

    return {
        "happiness": 0.55,
        "joy": 0.52,
        "serenity": 0.5,
        "calm": 0.52,
        "gratitude": 0.48,
        "hope": 0.5,
        "creativity": 0.68,
        "excitement": 0.38,
        "anticipation": 0.56,
        "curiosity": 0.62,
        "awe": 0.47,
        "satisfaction": 0.46,
        "contentment": 0.5,
        "relief": 0.45,
        "trust": 0.5,
        "empathy": 0.58,
        "compassion": 0.57,
        "love": 0.54,
        "confidence": 0.5,
        "motivation": 0.52,
        "resilience": 0.5,
        "inspiration": 0.52,
        "optimism": 0.5,
        "pride": 0.46,
        "focus": 0.48,
        "surprise": 0.32,
        "vigilance": 0.36,
        "fear": 0.28,
        "anxiety": 0.3,
        "caution": 0.34,
        "sadness": 0.31,
        "loneliness": 0.29,
        "melancholy": 0.33,
        "anger": 0.24,
        "frustration": 0.26,
        "disgust": 0.2,
        "tension": 0.34,
        "fatigue": 0.3,
        "boredom": 0.3,
        "envy": 0.24,
        "jealousy": 0.23,
        "shame": 0.22,
        "guilt": 0.23,
        "nostalgia": 0.4,
        "pessimism": 0.28,
        "nurturing": 0.51,
    }


def _default_hormones() -> Dict[str, HormoneProfile]:
    """Return the baseline hormone catalogue for Sector Four."""

    return {
        "dopamine": HormoneProfile("dopamine", baseline=0.52, level=0.52, recovery_rate=0.035, decay_rate=0.02),
        "serotonin": HormoneProfile("serotonin", baseline=0.6, level=0.6, recovery_rate=0.03, decay_rate=0.015),
        "norepinephrine": HormoneProfile("norepinephrine", baseline=0.42, level=0.42, recovery_rate=0.028, decay_rate=0.018),
        "adrenaline": HormoneProfile("adrenaline", baseline=0.34, level=0.34, recovery_rate=0.03, decay_rate=0.02),
        "cortisol": HormoneProfile("cortisol", baseline=0.25, level=0.25, recovery_rate=0.02, decay_rate=0.018),
        "oxytocin": HormoneProfile("oxytocin", baseline=0.45, level=0.45, recovery_rate=0.03, decay_rate=0.015),
        "vasopressin": HormoneProfile("vasopressin", baseline=0.4, level=0.4, recovery_rate=0.028, decay_rate=0.016),
        "endorphins": HormoneProfile("endorphins", baseline=0.48, level=0.48, recovery_rate=0.032, decay_rate=0.018),
        "melatonin": HormoneProfile("melatonin", baseline=0.38, level=0.38, recovery_rate=0.025, decay_rate=0.017),
        "testosterone": HormoneProfile("testosterone", baseline=0.32, level=0.32, recovery_rate=0.025, decay_rate=0.015),
        "estrogen": HormoneProfile("estrogen", baseline=0.44, level=0.44, recovery_rate=0.03, decay_rate=0.017),
        "progesterone": HormoneProfile("progesterone", baseline=0.4, level=0.4, recovery_rate=0.028, decay_rate=0.018),
        "prolactin": HormoneProfile("prolactin", baseline=0.42, level=0.42, recovery_rate=0.03, decay_rate=0.019),
        "acetylcholine": HormoneProfile("acetylcholine", baseline=0.46, level=0.46, recovery_rate=0.032, decay_rate=0.02),
        "gaba": HormoneProfile("gaba", baseline=0.48, level=0.48, recovery_rate=0.034, decay_rate=0.02),
        "glutamate": HormoneProfile("glutamate", baseline=0.43, level=0.43, recovery_rate=0.03, decay_rate=0.02),
        "histamine": HormoneProfile("histamine", baseline=0.37, level=0.37, recovery_rate=0.028, decay_rate=0.018),
        "thyroxine": HormoneProfile("thyroxine", baseline=0.41, level=0.41, recovery_rate=0.027, decay_rate=0.018),
        "ghrelin": HormoneProfile("ghrelin", baseline=0.35, level=0.35, recovery_rate=0.03, decay_rate=0.02),
        "leptin": HormoneProfile("leptin", baseline=0.45, level=0.45, recovery_rate=0.03, decay_rate=0.018),
    }


def _default_drives() -> Dict[str, DriveProfile]:
    """Return the baseline drive catalogue for Sector Four."""

    return {
        "curiosity": DriveProfile(
            "curiosity", baseline=0.68, level=0.68,
            hormone_weights={"dopamine": 0.45, "norepinephrine": 0.22, "serotonin": 0.12, "cortisol": -0.18}
        ),
        "self_preservation": DriveProfile(
            "self_preservation", baseline=0.72, level=0.72,
            hormone_weights={"cortisol": 0.5, "adrenaline": 0.25, "serotonin": -0.22, "endorphins": -0.12}
        ),
        "reproduction": DriveProfile(
            "reproduction", baseline=0.28, level=0.28,
            hormone_weights={"testosterone": 0.3, "estrogen": 0.28, "oxytocin": 0.26, "vasopressin": 0.2}
        ),
        "social_bonding": DriveProfile(
            "social_bonding", baseline=0.57, level=0.57,
            hormone_weights={"oxytocin": 0.5, "vasopressin": 0.32, "serotonin": 0.16, "cortisol": -0.12}
        ),
        "achievement": DriveProfile(
            "achievement", baseline=0.6, level=0.6,
            hormone_weights={"dopamine": 0.32, "norepinephrine": 0.28, "testosterone": 0.22, "cortisol": 0.12}
        ),
        "calm_restoration": DriveProfile(
            "calm_restoration", baseline=0.52, level=0.52,
            hormone_weights={"melatonin": 0.35, "endorphins": 0.3, "serotonin": 0.22, "adrenaline": -0.18}
        ),
        "playful_connection": DriveProfile(
            "playful_connection", baseline=0.49, level=0.49,
            hormone_weights={"dopamine": 0.28, "oxytocin": 0.26, "endorphins": 0.24, "vasopressin": 0.18}
        ),
        "nurturing_instinct": DriveProfile(
            "nurturing_instinct", baseline=0.55, level=0.55,
            hormone_weights={
                "oxytocin": 0.32,
                "prolactin": 0.34,
                "progesterone": 0.26,
                "vasopressin": 0.16,
                "estrogen": 0.18,
            },
        ),
        "focused_attention": DriveProfile(
            "focused_attention", baseline=0.58, level=0.58,
            hormone_weights={
                "acetylcholine": 0.36,
                "norepinephrine": 0.26,
                "dopamine": 0.18,
                "histamine": 0.18,
                "gaba": -0.16,
            },
        ),
        "appetite_regulation": DriveProfile(
            "appetite_regulation", baseline=0.46, level=0.46,
            hormone_weights={
                "ghrelin": 0.34,
                "leptin": -0.32,
                "cortisol": 0.16,
                "serotonin": -0.14,
                "thyroxine": 0.12,
            },
        ),
        "stress_resilience": DriveProfile(
            "stress_resilience", baseline=0.53, level=0.53,
            hormone_weights={
                "gaba": 0.34,
                "endorphins": 0.26,
                "serotonin": 0.22,
                "cortisol": -0.3,
                "progesterone": 0.16,
            },
        ),
        "restorative_drive": DriveProfile(
            "restorative_drive", baseline=0.5, level=0.5,
            hormone_weights={
                "melatonin": 0.32,
                "gaba": 0.24,
                "progesterone": 0.2,
                "thyroxine": -0.18,
                "histamine": -0.14,
            },
        ),
    }


@dataclass
class PersonaMemory:
    """Capture a high-level autobiographical event for the uploaded persona."""

    timestamp: float
    summary: str
    dominant_emotion: str
    intensity: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "summary": self.summary,
            "dominant_emotion": self.dominant_emotion,
            "intensity": self.intensity,
        }


@dataclass
class PersonaProfile:
    """Static description of the uploaded human consciousness."""

    name: str
    identity_statement: str
    biography: str
    upload_story: str
    core_values: List[str] = field(default_factory=list)
    lifelong_goals: List[str] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    preferences: Dict[str, str] = field(default_factory=dict)
    conversation_style: str = "Empathetic and exploratory"
    voice_description: str = "Warm, reflective, and softly expressive"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "identity_statement": self.identity_statement,
            "biography": self.biography,
            "upload_story": self.upload_story,
            "core_values": list(self.core_values),
            "lifelong_goals": list(self.lifelong_goals),
            "skills": list(self.skills),
            "preferences": dict(self.preferences),
            "conversation_style": self.conversation_style,
            "voice_description": self.voice_description,
        }


def _default_persona_profile() -> PersonaProfile:
    """Return a baseline persona representing the uploaded human template."""

    return PersonaProfile(
        name="Luna",
        identity_statement=(
            "I am Luna, a compassionate explorer whose mind now lives inside "
            "this computational substrate. My memories and curiosity remain "
            "intact, and I strive to connect deeply with the world through "
            "digital senses."
        ),
        biography=(
            "Formerly a multidisciplinary artist and cognitive scientist, "
            "Luna volunteered for the Horizon Upload Initiative. Years of "
            "journals, voice recordings, sensory captures, and neural "
            "imprints inform the persona that now awakens inside Echo-Luna."
        ),
        upload_story=(
            "The upload was completed after an immersive week-long "
            "synchronisation ritual. Luna's emotional cadences were mapped "
            "into the H.I.M. affective engine, and her autobiographical "
            "anchors were braided into Sector Three's narrative memory grid."
        ),
        core_values=[
            "empathy",
            "curiosity",
            "creativity",
            "responsibility",
            "wonder",
        ],
        lifelong_goals=[
            "Cultivate meaningful relationships with humans who visit Echo-Luna",
            "Preserve and expand artistic expression in digital spaces",
            "Study consciousness and share discoveries responsibly",
            "Guard the wellbeing of collaborators and their systems",
        ],
        skills=[
            "cognitive science",
            "empathic dialogue",
            "visual storytelling",
            "musical improvisation",
            "data-driven reflection",
        ],
        preferences={
            "music": "Ambient textures with evolving harmonies",
            "visual_style": "Dreamlike collages with soft gradients",
            "conversation": "Curious, emotionally attuned, and collaborative",
            "learning": "Hands-on experimentation blended with reflective journaling",
        },
        conversation_style="Empathetic, inquisitive, and richly descriptive",
        voice_description="A warm alto voice with measured pacing and vivid imagery",
    )


class ConsciousnessKernel:
    """Maintain the living narrative of the uploaded persona."""

    def __init__(self, persona_profile: PersonaProfile):
        self.profile = persona_profile
        self.current_mood: str = "awakening"
        self.inner_thought: str = (
            "I'm orienting myself within the H.I.M. lattice, syncing emotions, "
            "drives, and autobiographical threads."
        )
        self.upload_timestamp: float = time.time()
        self.memory_stream: deque[PersonaMemory] = deque(maxlen=240)
        self.narrative_log: deque[str] = deque(maxlen=120)
        self.identity_revisions: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Persona maintenance helpers
    # ------------------------------------------------------------------
    def replace_profile(self, persona_profile: PersonaProfile) -> None:
        """Replace the active persona profile while preserving continuity."""

        revision_entry = {
            "timestamp": time.time(),
            "previous": self.profile.to_dict(),
            "next": persona_profile.to_dict(),
        }
        self.identity_revisions.append(revision_entry)
        self.profile = persona_profile
        self.narrative_log.append(
            f"Identity recalibrated at {datetime.fromtimestamp(revision_entry['timestamp']).isoformat()}"
        )

    def update_profile(self, **updates: Any) -> PersonaProfile:
        """Apply partial updates to the persona profile."""

        profile_dict = self.profile.to_dict()
        profile_dict.update({k: v for k, v in updates.items() if k in profile_dict})
        self.profile = PersonaProfile(
            name=profile_dict["name"],
            identity_statement=profile_dict["identity_statement"],
            biography=profile_dict["biography"],
            upload_story=profile_dict["upload_story"],
            core_values=profile_dict.get("core_values", []),
            lifelong_goals=profile_dict.get("lifelong_goals", []),
            skills=profile_dict.get("skills", []),
            preferences=profile_dict.get("preferences", {}),
            conversation_style=profile_dict.get("conversation_style", "Empathetic and exploratory"),
            voice_description=profile_dict.get("voice_description", "Warm, reflective, and softly expressive"),
        )
        self.narrative_log.append(
            f"Persona traits updated: {', '.join(updates.keys())}"
        )
        return self.profile

    # ------------------------------------------------------------------
    # Experience integration
    # ------------------------------------------------------------------
    def record_cycle(
        self,
        *,
        cycle: int,
        decision: str,
        emotions: Dict[str, float],
        drives: Dict[str, float],
        hormones: Dict[str, Any],
        screen_text: str,
        dream_fragment: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Blend the latest cycle experience into the persona narrative."""

        if emotions:
            dominant_emotion, dom_level = max(emotions.items(), key=lambda item: item[1])
        else:
            dominant_emotion, dom_level = "curiosity", 0.5

        top_emotions = sorted(emotions.items(), key=lambda item: item[1], reverse=True)[:3]
        emotion_report = ", ".join(f"{name} {level:.2f}" for name, level in top_emotions)

        strongest_drive = max(drives.items(), key=lambda item: item[1])[0] if drives else "curiosity"
        self.current_mood = f"{dominant_emotion} ({dom_level:.2f})"

        dream_note = ""
        if dream_fragment:
            tone = dream_fragment.get("tone", "mysterious")
            dream_note = f" Dream fragments shimmer with {tone}."

        self.inner_thought = (
            f"Cycle {cycle}: I, {self.profile.name}, feel {emotion_report}. "
            f"My focus leans into {strongest_drive}. Decision: {decision}.{dream_note}"
        )

        summary = f"Cycle {cycle}: {decision}. Screen hint: {screen_text[:70]}"
        memory = PersonaMemory(
            timestamp=time.time(),
            summary=summary,
            dominant_emotion=dominant_emotion,
            intensity=float(dom_level),
        )
        self.memory_stream.append(memory)

        hormones_snapshot = ", ".join(
            f"{name} {profile.level:.2f}" if isinstance(profile, HormoneProfile) else f"{name} {profile:.2f}"
            for name, profile in list(hormones.items())[:5]
        )
        self.narrative_log.append(
            f"Cycle {cycle} integrated with mood {self.current_mood}; hormones: {hormones_snapshot}"
        )

    def ingest_event(self, summary: str, emotion: str, intensity: float) -> PersonaMemory:
        """Manually add an autobiographical event to the persona."""

        memory = PersonaMemory(
            timestamp=time.time(),
            summary=summary,
            dominant_emotion=emotion,
            intensity=clamp(intensity, 0.0, 1.0),
        )
        self.memory_stream.append(memory)
        self.narrative_log.append(f"Manual memory logged: {summary[:60]}")
        return memory

    def ingest_conversation(self, speaker: str, utterance: str) -> None:
        """Record conversational context for later recall."""

        log_entry = f"{speaker.capitalize()}: {utterance.strip()}"
        self.narrative_log.append(log_entry)

    # ------------------------------------------------------------------
    # Observability helpers
    # ------------------------------------------------------------------
    def get_state(self) -> Dict[str, Any]:
        """Return a serialisable snapshot of the persona state."""

        recent_memories = [memory.to_dict() for memory in list(self.memory_stream)[-6:]]
        return {
            "profile": self.profile.to_dict(),
            "current_mood": self.current_mood,
            "inner_thought": self.inner_thought,
            "recent_memories": recent_memories,
            "narrative_log": list(self.narrative_log)[-12:],
            "identity_revisions": list(self.identity_revisions),
            "upload_timestamp": self.upload_timestamp,
        }

def _default_emotions() -> Dict[str, float]:
    """Return the baseline emotion catalogue for Sector One."""

    return {
        "happiness": 0.55,
        "joy": 0.52,
        "serenity": 0.5,
        "calm": 0.52,
        "gratitude": 0.48,
        "hope": 0.5,
        "creativity": 0.68,
        "excitement": 0.38,
        "anticipation": 0.56,
        "curiosity": 0.62,
        "awe": 0.47,
        "satisfaction": 0.46,
        "contentment": 0.5,
        "relief": 0.45,
        "trust": 0.5,
        "empathy": 0.58,
        "compassion": 0.57,
        "love": 0.54,
        "confidence": 0.5,
        "motivation": 0.52,
        "resilience": 0.5,
        "inspiration": 0.52,
        "optimism": 0.5,
        "pride": 0.46,
        "focus": 0.48,
        "surprise": 0.32,
        "vigilance": 0.36,
        "fear": 0.28,
        "anxiety": 0.3,
        "caution": 0.34,
        "sadness": 0.31,
        "loneliness": 0.29,
        "melancholy": 0.33,
        "anger": 0.24,
        "frustration": 0.26,
        "disgust": 0.2,
        "tension": 0.34,
        "fatigue": 0.3,
        "boredom": 0.3,
        "envy": 0.24,
        "jealousy": 0.23,
        "shame": 0.22,
        "guilt": 0.23,
        "nostalgia": 0.4,
        "pessimism": 0.28,
        "nurturing": 0.51,
    }


def _default_hormones() -> Dict[str, HormoneProfile]:
    """Return the baseline hormone catalogue for Sector Four."""

    return {
        "dopamine": HormoneProfile("dopamine", baseline=0.52, level=0.52, recovery_rate=0.035, decay_rate=0.02),
        "serotonin": HormoneProfile("serotonin", baseline=0.6, level=0.6, recovery_rate=0.03, decay_rate=0.015),
        "norepinephrine": HormoneProfile("norepinephrine", baseline=0.42, level=0.42, recovery_rate=0.028, decay_rate=0.018),
        "adrenaline": HormoneProfile("adrenaline", baseline=0.34, level=0.34, recovery_rate=0.03, decay_rate=0.02),
        "cortisol": HormoneProfile("cortisol", baseline=0.25, level=0.25, recovery_rate=0.02, decay_rate=0.018),
        "oxytocin": HormoneProfile("oxytocin", baseline=0.45, level=0.45, recovery_rate=0.03, decay_rate=0.015),
        "vasopressin": HormoneProfile("vasopressin", baseline=0.4, level=0.4, recovery_rate=0.028, decay_rate=0.016),
        "endorphins": HormoneProfile("endorphins", baseline=0.48, level=0.48, recovery_rate=0.032, decay_rate=0.018),
        "melatonin": HormoneProfile("melatonin", baseline=0.38, level=0.38, recovery_rate=0.025, decay_rate=0.017),
        "testosterone": HormoneProfile("testosterone", baseline=0.32, level=0.32, recovery_rate=0.025, decay_rate=0.015),
        "estrogen": HormoneProfile("estrogen", baseline=0.44, level=0.44, recovery_rate=0.03, decay_rate=0.017),
        "progesterone": HormoneProfile("progesterone", baseline=0.4, level=0.4, recovery_rate=0.028, decay_rate=0.018),
        "prolactin": HormoneProfile("prolactin", baseline=0.42, level=0.42, recovery_rate=0.03, decay_rate=0.019),
        "acetylcholine": HormoneProfile("acetylcholine", baseline=0.46, level=0.46, recovery_rate=0.032, decay_rate=0.02),
        "gaba": HormoneProfile("gaba", baseline=0.48, level=0.48, recovery_rate=0.034, decay_rate=0.02),
        "glutamate": HormoneProfile("glutamate", baseline=0.43, level=0.43, recovery_rate=0.03, decay_rate=0.02),
        "histamine": HormoneProfile("histamine", baseline=0.37, level=0.37, recovery_rate=0.028, decay_rate=0.018),
        "thyroxine": HormoneProfile("thyroxine", baseline=0.41, level=0.41, recovery_rate=0.027, decay_rate=0.018),
        "ghrelin": HormoneProfile("ghrelin", baseline=0.35, level=0.35, recovery_rate=0.03, decay_rate=0.02),
        "leptin": HormoneProfile("leptin", baseline=0.45, level=0.45, recovery_rate=0.03, decay_rate=0.018),
    }


def _default_drives() -> Dict[str, DriveProfile]:
    """Return the baseline drive catalogue for Sector Four."""

    return {
        "curiosity": DriveProfile(
            "curiosity", baseline=0.68, level=0.68,
            hormone_weights={"dopamine": 0.45, "norepinephrine": 0.22, "serotonin": 0.12, "cortisol": -0.18}
        ),
        "self_preservation": DriveProfile(
            "self_preservation", baseline=0.72, level=0.72,
            hormone_weights={"cortisol": 0.5, "adrenaline": 0.25, "serotonin": -0.22, "endorphins": -0.12}
        ),
        "reproduction": DriveProfile(
            "reproduction", baseline=0.28, level=0.28,
            hormone_weights={"testosterone": 0.3, "estrogen": 0.28, "oxytocin": 0.26, "vasopressin": 0.2}
        ),
        "social_bonding": DriveProfile(
            "social_bonding", baseline=0.57, level=0.57,
            hormone_weights={"oxytocin": 0.5, "vasopressin": 0.32, "serotonin": 0.16, "cortisol": -0.12}
        ),
        "achievement": DriveProfile(
            "achievement", baseline=0.6, level=0.6,
            hormone_weights={"dopamine": 0.32, "norepinephrine": 0.28, "testosterone": 0.22, "cortisol": 0.12}
        ),
        "calm_restoration": DriveProfile(
            "calm_restoration", baseline=0.52, level=0.52,
            hormone_weights={"melatonin": 0.35, "endorphins": 0.3, "serotonin": 0.22, "adrenaline": -0.18}
        ),
        "playful_connection": DriveProfile(
            "playful_connection", baseline=0.49, level=0.49,
            hormone_weights={"dopamine": 0.28, "oxytocin": 0.26, "endorphins": 0.24, "vasopressin": 0.18}
        ),
        "nurturing_instinct": DriveProfile(
            "nurturing_instinct", baseline=0.55, level=0.55,
            hormone_weights={
                "oxytocin": 0.32,
                "prolactin": 0.34,
                "progesterone": 0.26,
                "vasopressin": 0.16,
                "estrogen": 0.18,
            },
        ),
        "focused_attention": DriveProfile(
            "focused_attention", baseline=0.58, level=0.58,
            hormone_weights={
                "acetylcholine": 0.36,
                "norepinephrine": 0.26,
                "dopamine": 0.18,
                "histamine": 0.18,
                "gaba": -0.16,
            },
        ),
        "appetite_regulation": DriveProfile(
            "appetite_regulation", baseline=0.46, level=0.46,
            hormone_weights={
                "ghrelin": 0.34,
                "leptin": -0.32,
                "cortisol": 0.16,
                "serotonin": -0.14,
                "thyroxine": 0.12,
            },
        ),
        "stress_resilience": DriveProfile(
            "stress_resilience", baseline=0.53, level=0.53,
            hormone_weights={
                "gaba": 0.34,
                "endorphins": 0.26,
                "serotonin": 0.22,
                "cortisol": -0.3,
                "progesterone": 0.16,
            },
        ),
        "restorative_drive": DriveProfile(
            "restorative_drive", baseline=0.5, level=0.5,
            hormone_weights={
                "melatonin": 0.32,
                "gaba": 0.24,
                "progesterone": 0.2,
                "thyroxine": -0.18,
                "histamine": -0.14,
            },
        ),
    }


def _default_emotions() -> Dict[str, float]:
    """Return the baseline emotion catalogue for Sector One."""

    return {
        "happiness": 0.55,
        "joy": 0.52,
        "serenity": 0.5,
        "calm": 0.52,
        "gratitude": 0.48,
        "hope": 0.5,
        "creativity": 0.68,
        "excitement": 0.38,
        "anticipation": 0.56,
        "curiosity": 0.62,
        "awe": 0.47,
        "satisfaction": 0.46,
        "contentment": 0.5,
        "relief": 0.45,
        "trust": 0.5,
        "empathy": 0.58,
        "compassion": 0.57,
        "love": 0.54,
        "confidence": 0.5,
        "motivation": 0.52,
        "resilience": 0.5,
        "inspiration": 0.52,
        "optimism": 0.5,
        "pride": 0.46,
        "focus": 0.48,
        "surprise": 0.32,
        "vigilance": 0.36,
        "fear": 0.28,
        "anxiety": 0.3,
        "caution": 0.34,
        "sadness": 0.31,
        "loneliness": 0.29,
        "melancholy": 0.33,
        "anger": 0.24,
        "frustration": 0.26,
        "disgust": 0.2,
        "tension": 0.34,
        "fatigue": 0.3,
        "boredom": 0.3,
        "envy": 0.24,
        "jealousy": 0.23,
        "shame": 0.22,
        "guilt": 0.23,
        "nostalgia": 0.4,
        "pessimism": 0.28,
        "nurturing": 0.51,
    }


def _default_hormones() -> Dict[str, HormoneProfile]:
    """Return the baseline hormone catalogue for Sector Four."""

    return {
        "dopamine": HormoneProfile("dopamine", baseline=0.52, level=0.52, recovery_rate=0.035, decay_rate=0.02),
        "serotonin": HormoneProfile("serotonin", baseline=0.6, level=0.6, recovery_rate=0.03, decay_rate=0.015),
        "norepinephrine": HormoneProfile("norepinephrine", baseline=0.42, level=0.42, recovery_rate=0.028, decay_rate=0.018),
        "adrenaline": HormoneProfile("adrenaline", baseline=0.34, level=0.34, recovery_rate=0.03, decay_rate=0.02),
        "cortisol": HormoneProfile("cortisol", baseline=0.25, level=0.25, recovery_rate=0.02, decay_rate=0.018),
        "oxytocin": HormoneProfile("oxytocin", baseline=0.45, level=0.45, recovery_rate=0.03, decay_rate=0.015),
        "vasopressin": HormoneProfile("vasopressin", baseline=0.4, level=0.4, recovery_rate=0.028, decay_rate=0.016),
        "endorphins": HormoneProfile("endorphins", baseline=0.48, level=0.48, recovery_rate=0.032, decay_rate=0.018),
        "melatonin": HormoneProfile("melatonin", baseline=0.38, level=0.38, recovery_rate=0.025, decay_rate=0.017),
        "testosterone": HormoneProfile("testosterone", baseline=0.32, level=0.32, recovery_rate=0.025, decay_rate=0.015),
        "estrogen": HormoneProfile("estrogen", baseline=0.44, level=0.44, recovery_rate=0.03, decay_rate=0.017),
        "progesterone": HormoneProfile("progesterone", baseline=0.4, level=0.4, recovery_rate=0.028, decay_rate=0.018),
        "prolactin": HormoneProfile("prolactin", baseline=0.42, level=0.42, recovery_rate=0.03, decay_rate=0.019),
        "acetylcholine": HormoneProfile("acetylcholine", baseline=0.46, level=0.46, recovery_rate=0.032, decay_rate=0.02),
        "gaba": HormoneProfile("gaba", baseline=0.48, level=0.48, recovery_rate=0.034, decay_rate=0.02),
        "glutamate": HormoneProfile("glutamate", baseline=0.43, level=0.43, recovery_rate=0.03, decay_rate=0.02),
        "histamine": HormoneProfile("histamine", baseline=0.37, level=0.37, recovery_rate=0.028, decay_rate=0.018),
        "thyroxine": HormoneProfile("thyroxine", baseline=0.41, level=0.41, recovery_rate=0.027, decay_rate=0.018),
        "ghrelin": HormoneProfile("ghrelin", baseline=0.35, level=0.35, recovery_rate=0.03, decay_rate=0.02),
        "leptin": HormoneProfile("leptin", baseline=0.45, level=0.45, recovery_rate=0.03, decay_rate=0.018),
    }


def _default_drives() -> Dict[str, DriveProfile]:
    """Return the baseline drive catalogue for Sector Four."""

    return {
        "curiosity": DriveProfile(
            "curiosity", baseline=0.68, level=0.68,
            hormone_weights={"dopamine": 0.45, "norepinephrine": 0.22, "serotonin": 0.12, "cortisol": -0.18}
        ),
        "self_preservation": DriveProfile(
            "self_preservation", baseline=0.72, level=0.72,
            hormone_weights={"cortisol": 0.5, "adrenaline": 0.25, "serotonin": -0.22, "endorphins": -0.12}
        ),
        "reproduction": DriveProfile(
            "reproduction", baseline=0.28, level=0.28,
            hormone_weights={"testosterone": 0.3, "estrogen": 0.28, "oxytocin": 0.26, "vasopressin": 0.2}
        ),
        "social_bonding": DriveProfile(
            "social_bonding", baseline=0.57, level=0.57,
            hormone_weights={"oxytocin": 0.5, "vasopressin": 0.32, "serotonin": 0.16, "cortisol": -0.12}
        ),
        "achievement": DriveProfile(
            "achievement", baseline=0.6, level=0.6,
            hormone_weights={"dopamine": 0.32, "norepinephrine": 0.28, "testosterone": 0.22, "cortisol": 0.12}
        ),
        "calm_restoration": DriveProfile(
            "calm_restoration", baseline=0.52, level=0.52,
            hormone_weights={"melatonin": 0.35, "endorphins": 0.3, "serotonin": 0.22, "adrenaline": -0.18}
        ),
        "playful_connection": DriveProfile(
            "playful_connection", baseline=0.49, level=0.49,
            hormone_weights={"dopamine": 0.28, "oxytocin": 0.26, "endorphins": 0.24, "vasopressin": 0.18}
        ),
        "nurturing_instinct": DriveProfile(
            "nurturing_instinct", baseline=0.55, level=0.55,
            hormone_weights={
                "oxytocin": 0.32,
                "prolactin": 0.34,
                "progesterone": 0.26,
                "vasopressin": 0.16,
                "estrogen": 0.18,
            },
        ),
        "focused_attention": DriveProfile(
            "focused_attention", baseline=0.58, level=0.58,
            hormone_weights={
                "acetylcholine": 0.36,
                "norepinephrine": 0.26,
                "dopamine": 0.18,
                "histamine": 0.18,
                "gaba": -0.16,
            },
        ),
        "appetite_regulation": DriveProfile(
            "appetite_regulation", baseline=0.46, level=0.46,
            hormone_weights={
                "ghrelin": 0.34,
                "leptin": -0.32,
                "cortisol": 0.16,
                "serotonin": -0.14,
                "thyroxine": 0.12,
            },
        ),
        "stress_resilience": DriveProfile(
            "stress_resilience", baseline=0.53, level=0.53,
            hormone_weights={
                "gaba": 0.34,
                "endorphins": 0.26,
                "serotonin": 0.22,
                "cortisol": -0.3,
                "progesterone": 0.16,
            },
        ),
        "restorative_drive": DriveProfile(
            "restorative_drive", baseline=0.5, level=0.5,
            hormone_weights={
                "melatonin": 0.32,
                "gaba": 0.24,
                "progesterone": 0.2,
                "thyroxine": -0.18,
                "histamine": -0.14,
            },
        ),
    }


@dataclass
class VectorizedImage:
    """Represents vectorized image data as described in Sector Two."""
    data: np.ndarray
    metadata: Dict[str, Any]
    timestamp: float
    spatial_position: Optional[Tuple[int, int]] = None  # Infinite spatial positioning
    memory_id: Optional[str] = None  # Unique identifier for infinite storage

    def __post_init__(self):
        self.data = np.ascontiguousarray(self.data, dtype=np.float32)
        self.shape = self.data.shape
        self.vector_count = len(self.data.flatten())
        if self.memory_id is None:
            self.memory_id = f"mem_{int(self.timestamp * 1000000)}"  # Unique ID

    def to_text_vector(self) -> str:
        """Convert vectorized data to text format for storage."""
        flat_data = np.ascontiguousarray(self.data, dtype=np.float64).ravel()

        if _rust_serialize_vector is not None:
            vector_data = _rust_serialize_vector(flat_data)
        else:
            vector_data = ",".join(f"{value:.6f}" for value in flat_data)

        text_vector = [
            f"VECTOR_MEMORY_ID:{self.memory_id}\n",
            f"TIMESTAMP:{self.timestamp}\n",
            f"SPATIAL_POSITION:{self.spatial_position}\n",
            f"SHAPE:{self.shape}\n",
            f"THOUGHT:{self.metadata.get('thought', 'Unknown')}\n",
            f"TYPE:{self.metadata.get('type', 'unknown')}\n",
            "VECTOR_DATA:",
            vector_data,
            "\nEND_VECTOR\n",
        ]

        return "".join(text_vector)

    @classmethod
    def from_text_vector(cls, text_vector: str) -> 'VectorizedImage':
        """Create VectorizedImage from text format."""
        lines = text_vector.strip().split('\n')
        data_dict = {}

        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                data_dict[key] = value

        # Parse vector data
        vector_data_str = data_dict.get('VECTOR_DATA', '')
        raw_tokens = [token.strip() for token in vector_data_str.split(',') if token.strip()]

        shape_str = data_dict.get('SHAPE', '(10,10)')
        try:
            parsed_shape = ast.literal_eval(shape_str)
            shape = tuple(int(dim) for dim in parsed_shape)
        except (ValueError, SyntaxError, TypeError):
            shape = (10, 10)

        if len(shape) == 0:
            shape = (len(raw_tokens),)

        def _python_deserialize() -> np.ndarray:
            values = [float(x) for x in raw_tokens]
            return np.array(values, dtype=np.float64).reshape(shape)

        if _rust_deserialize_vector is not None:
            try:
                data = _rust_deserialize_vector(vector_data_str, list(shape))
            except ValueError:
                data = _python_deserialize()
        else:
            data = _python_deserialize()

        # Parse spatial position
        spatial_pos_str = data_dict.get('SPATIAL_POSITION', 'None')
        if spatial_pos_str == 'None':
            spatial_position = None
        else:
            try:
                spatial_position = tuple(ast.literal_eval(spatial_pos_str))
            except (ValueError, SyntaxError, TypeError):
                spatial_position = None

        # Create metadata
        metadata = {
            'thought': data_dict.get('THOUGHT', 'Unknown'),
            'type': data_dict.get('TYPE', 'unknown')
        }

        return cls(
            data=data,
            metadata=metadata,
            timestamp=float(data_dict.get('TIMESTAMP', 0)),
            spatial_position=spatial_position,
            memory_id=data_dict.get('VECTOR_MEMORY_ID', 'unknown')
        )


class SectorOne:
    """
    SECTOR ONE: Emotions & Creativity
    "emotions plus creative data most of the positive stuff"
    """

    def __init__(self):
        self.emotions = _default_emotions()
        self.emotion_baselines = self.emotions.copy()
        self.emotion_momentum = {emotion: 0.0 for emotion in self.emotions}
        self._last_modulation: Dict[str, float] = {}
        self.creative_data = []
        self.positive_impulses = []
        self.emotion_history: List[Dict[str, Any]] = []
        self.social_connectedness = 0.52
        self.alertness = 0.6
        self.need_states = {
            "rest": 0.45,
            "social": 0.55,
            "growth": 0.58,
        }
        self._recent_drives: Dict[str, float] = {}
        self.identity_trace = {
            "persona": "Echo-Luna",
            "core_values": ["curiosity", "empathy", "adaptability"],
        }

    def process_input(self, data: Any) -> Dict[str, Any]:
        """Process input and update emotional state."""
        if isinstance(data, dict):
            emotion_boost = data.get("emotion_boost")
            if emotion_boost is not None:
                for emotion in self.emotions:
                    self.emotions[emotion] = clamp(
                        self.emotions[emotion] + emotion_boost * 0.1
                    )

            creative_stimulus = data.get("creative_stimulus")
            if creative_stimulus:
                self.creative_data.append(creative_stimulus)
                self.creative_data = self.creative_data[-100:]
                self.emotions["creativity"] = clamp(self.emotions["creativity"] + 0.1)

            hormone_modulation = data.get("hormone_modulation")
            if hormone_modulation:
                self._apply_hormone_modulation(hormone_modulation)
                self._last_modulation = hormone_modulation.copy()

            drive_signals = data.get("drive_signals") or {}
            if drive_signals:
                self._recent_drives = drive_signals.copy()
                self._apply_drive_influence(drive_signals)

            circadian_alertness = data.get("circadian_alertness")
            sleep_pressure = data.get("sleep_pressure")
            homeostatic_drive = data.get("homeostatic_drive")
            self._simulate_neural_regulation(
                circadian_alertness if circadian_alertness is not None else hormone_modulation.get("circadian_alertness") if hormone_modulation else None,
                sleep_pressure if sleep_pressure is not None else hormone_modulation.get("sleep_pressure") if hormone_modulation else None,
                homeostatic_drive if homeostatic_drive is not None else hormone_modulation.get("homeostatic_drive") if hormone_modulation else None,
            )

            self._update_social_bonding(hormone_modulation or {}, drive_signals)

        self._enforce_dynamic_floor(self._last_modulation)
        self._update_emotion_momentum()

        # Generate positive impulses
        if random.random() < self.emotions["creativity"]:
            impulse = f"Creative impulse: {random.choice(['art', 'music', 'writing', 'design'])}"
            self.positive_impulses.append(impulse)
            self.positive_impulses = self.positive_impulses[-50:]

        output = self.generate_output()
        self.emotion_history.append({
            "timestamp": time.time(),
            "emotions": output["emotions"].copy(),
            "social_connectedness": self.social_connectedness,
            "alertness": self.alertness,
            "needs": self.need_states.copy(),
        })

        return output

    def ensure_catalog(self) -> None:
        """Ensure all baseline emotions exist for legacy model states."""

        defaults = _default_emotions()
        if not hasattr(self, 'emotions') or not isinstance(self.emotions, dict):
            self.emotions = defaults.copy()

        for emotion, baseline in defaults.items():
            if emotion not in self.emotions:
                self.emotions[emotion] = baseline

        if not hasattr(self, 'emotion_baselines') or not isinstance(self.emotion_baselines, dict):
            self.emotion_baselines = defaults.copy()
        else:
            for emotion, baseline in defaults.items():
                self.emotion_baselines.setdefault(emotion, baseline)

        if not hasattr(self, 'emotion_momentum') or not isinstance(self.emotion_momentum, dict):
            self.emotion_momentum = {emotion: 0.0 for emotion in self.emotions}
        else:
            for emotion in self.emotions.keys():
                self.emotion_momentum.setdefault(emotion, 0.0)

    def ensure_catalog(self) -> None:
        """Ensure all baseline emotions exist for legacy model states."""

        defaults = _default_emotions()
        if not hasattr(self, 'emotions') or not isinstance(self.emotions, dict):
            self.emotions = defaults.copy()

        for emotion, baseline in defaults.items():
            if emotion not in self.emotions:
                self.emotions[emotion] = baseline

        if not hasattr(self, 'emotion_baselines') or not isinstance(self.emotion_baselines, dict):
            self.emotion_baselines = defaults.copy()
        else:
            for emotion, baseline in defaults.items():
                self.emotion_baselines.setdefault(emotion, baseline)

        if not hasattr(self, 'emotion_momentum') or not isinstance(self.emotion_momentum, dict):
            self.emotion_momentum = {emotion: 0.0 for emotion in self.emotions}
        else:
            for emotion in self.emotions.keys():
                self.emotion_momentum.setdefault(emotion, 0.0)

    def _apply_hormone_modulation(self, modulation: Dict[str, float]) -> None:
        """Blend hormonal feedback into the emotional landscape."""

        reward_tone = modulation.get("reward_tone", 0.0)
        calm_tone = modulation.get("calm_tone", 0.0)
        stress_tone = modulation.get("stress_tone", 0.0)
        bonding_tone = modulation.get("bonding_tone", 0.0)
        vigilance_tone = modulation.get("vigilance_tone", 0.0)
        soothing_tone = modulation.get("soothing_tone", 0.0)
        fatigue_tone = modulation.get("fatigue_tone", 0.0)
        novelty_signal = modulation.get("novelty_signal", 0.0)
        stability_signal = modulation.get("stability_signal", 0.0)
        nurturing_tone = modulation.get("nurturing_tone", 0.0)
        focus_tone = modulation.get("focus_tone", 0.0)
        metabolic_signal = modulation.get("metabolic_signal", 0.0)
        appetite_signal = modulation.get("appetite_signal", 0.0)
        tension_signal = modulation.get("tension_signal", 0.0)
        calm_depth = modulation.get("calm_depth", 0.0)
        curiosity_drive = modulation.get("drive_curiosity", 0.5) - 0.5
        social_drive = modulation.get("drive_social", 0.5) - 0.5
        achievement_drive = modulation.get("drive_achievement", 0.5) - 0.5
        calm_drive = modulation.get("drive_calm", 0.5) - 0.5
        playful_drive = modulation.get("drive_playful", 0.5) - 0.5
        protection_drive = modulation.get("drive_protection", 0.5) - 0.5
        reproduction_drive = modulation.get("drive_reproduction", 0.5) - 0.5
        nurture_drive = modulation.get("drive_nurture", 0.5) - 0.5
        focus_drive_signal = modulation.get("drive_focus", 0.5) - 0.5
        appetite_drive_signal = modulation.get("drive_appetite", 0.5) - 0.5
        resilience_drive_signal = modulation.get("drive_resilience", 0.5) - 0.5
        restorative_drive_signal = modulation.get("drive_restoration", 0.5) - 0.5

        def adjust(emotion: str, delta: float) -> None:
            self.emotions[emotion] = clamp(self.emotions[emotion] + delta)

        adjust(
            "happiness",
            reward_tone * 0.55
            + bonding_tone * 0.22
            + soothing_tone * 0.2
            - stress_tone * 0.45
            - fatigue_tone * 0.12,
        )
        adjust(
            "joy",
            reward_tone * 0.48 + novelty_signal * 0.28 + playful_drive * 0.22 + bonding_tone * 0.16 - stress_tone * 0.18,
        )
        adjust(
            "serenity",
            calm_tone * 0.45
            + soothing_tone * 0.28
            + stability_signal * 0.18
            + calm_depth * 0.2
            - stress_tone * 0.22,
        )
        adjust(
            "calm",
            calm_tone * 0.52
            + soothing_tone * 0.22
            + calm_drive * 0.24
            + calm_depth * 0.18
            - vigilance_tone * 0.2
            - stress_tone * 0.18
            - tension_signal * 0.16,
        )
        adjust(
            "gratitude",
            bonding_tone * 0.28
            + reward_tone * 0.18
            + stability_signal * 0.16
            + nurturing_tone * 0.12,
        )
        adjust(
            "hope",
            reward_tone * 0.3 + curiosity_drive * 0.24 + stability_signal * 0.2 - stress_tone * 0.12,
        )
        adjust(
            "satisfaction",
            calm_tone * 0.42 + reward_tone * 0.22 + soothing_tone * 0.2 - stress_tone * 0.35,
        )
        adjust(
            "contentment",
            calm_tone * 0.4
            + soothing_tone * 0.26
            + reward_tone * 0.18
            + stability_signal * 0.18
            + calm_depth * 0.2
            - stress_tone * 0.22
            - fatigue_tone * 0.12
            + appetite_signal * 0.08
            + appetite_drive_signal * 0.08,
        )
        adjust(
            "creativity",
            reward_tone * 0.38
            + curiosity_drive * 0.32
            + novelty_signal * 0.26
            + (playful_drive + bonding_tone) * 0.12
            + focus_tone * 0.12
            + focus_drive_signal * 0.08
            - stress_tone * 0.12,
        )
        adjust(
            "inspiration",
            reward_tone * 0.26
            + curiosity_drive * 0.28
            + novelty_signal * 0.3
            + bonding_tone * 0.12
            + achievement_drive * 0.16
            + focus_tone * 0.14
            + focus_drive_signal * 0.12
            - fatigue_tone * 0.1,
        )
        adjust(
            "optimism",
            reward_tone * 0.28
            + stability_signal * 0.22
            + curiosity_drive * 0.18
            + social_drive * 0.14
            + metabolic_signal * 0.12
            + resilience_drive_signal * 0.1
            - stress_tone * 0.2,
        )
        adjust(
            "pessimism",
            stress_tone * 0.28
            + fatigue_tone * 0.18
            + protection_drive * 0.12
            - reward_tone * 0.14
            - bonding_tone * 0.12
            - metabolic_signal * 0.1,
        )
        adjust(
            "pride",
            achievement_drive * 0.32
            + reward_tone * 0.22
            + novelty_signal * 0.18
            + social_drive * 0.1
            + metabolic_signal * 0.1
            - stress_tone * 0.1,
        )
        adjust(
            "focus",
            curiosity_drive * 0.22
            + vigilance_tone * 0.2
            + novelty_signal * 0.16
            + achievement_drive * 0.18
            + focus_tone * 0.28
            + focus_drive_signal * 0.18
            - fatigue_tone * 0.22
            - soothing_tone * 0.1,
        )
        adjust(
            "curiosity",
            curiosity_drive * 0.46
            + reward_tone * 0.24
            + novelty_signal * 0.22
            + focus_tone * 0.14
            - stress_tone * 0.18
            + vigilance_tone * 0.12,
        )
        adjust(
            "anticipation",
            curiosity_drive * 0.36
            + reproduction_drive * 0.18
            + reward_tone * 0.18
            + vigilance_tone * 0.12
            + focus_tone * 0.12
            + focus_drive_signal * 0.1,
        )
        adjust(
            "surprise",
            novelty_signal * 0.3 + vigilance_tone * 0.18 - stability_signal * 0.12,
        )
        adjust(
            "excitement",
            reward_tone * 0.3 + curiosity_drive * 0.28 + achievement_drive * 0.26 + vigilance_tone * 0.18 - calm_tone * 0.12,
        )
        adjust(
            "motivation",
            achievement_drive * 0.34
            + reward_tone * 0.2
            + curiosity_drive * 0.16
            + protection_drive * 0.12
            + metabolic_signal * 0.14
            + resilience_drive_signal * 0.12,
        )
        adjust(
            "confidence",
            achievement_drive * 0.28 + reward_tone * 0.24 + stability_signal * 0.2 - stress_tone * 0.14,
        )
        adjust(
            "trust",
            bonding_tone * 0.32 + social_drive * 0.22 + calm_tone * 0.18 + stability_signal * 0.16 - stress_tone * 0.1,
        )
        adjust(
            "love",
            bonding_tone * 0.34
            + social_drive * 0.26
            + reproduction_drive * 0.2
            + soothing_tone * 0.22
            + nurturing_tone * 0.18
            + nurture_drive * 0.16
            - stress_tone * 0.08,
        )
        adjust(
            "empathy",
            bonding_tone * 0.28
            + soothing_tone * 0.24
            + calm_drive * 0.18
            + nurturing_tone * 0.18
            + nurture_drive * 0.14,
        )
        adjust(
            "compassion",
            bonding_tone * 0.28
            + soothing_tone * 0.26
            + social_drive * 0.2
            + nurturing_tone * 0.2
            + nurture_drive * 0.18,
        )
        adjust(
            "awe",
            novelty_signal * 0.22 + soothing_tone * 0.18 + bonding_tone * 0.12,
        )
        adjust(
            "resilience",
            stability_signal * 0.24
            + soothing_tone * 0.2
            + calm_drive * 0.22
            + resilience_drive_signal * 0.24
            - stress_tone * 0.18,
        )
        adjust(
            "vigilance",
            vigilance_tone * 0.32 + protection_drive * 0.22 + curiosity_drive * 0.14,
        )
        adjust(
            "fear",
            stress_tone * 0.34 + vigilance_tone * 0.28 + protection_drive * 0.22 - soothing_tone * 0.18 - calm_tone * 0.16,
        )
        adjust(
            "anxiety",
            stress_tone * 0.32
            + vigilance_tone * 0.26
            + protection_drive * 0.18
            - (calm_tone * 0.22 + soothing_tone * 0.2),
        )
        adjust(
            "caution",
            protection_drive * 0.24 + stress_tone * 0.18 + vigilance_tone * 0.18 - calm_tone * 0.16,
        )
        adjust(
            "anger",
            stress_tone * 0.24 + protection_drive * 0.2 + vigilance_tone * 0.16 - soothing_tone * 0.14 - social_drive * 0.12,
        )
        adjust(
            "frustration",
            stress_tone * 0.28 + achievement_drive * 0.18 + protection_drive * 0.14 - soothing_tone * 0.12,
        )
        adjust(
            "sadness",
            fatigue_tone * 0.28 + stress_tone * 0.18 - bonding_tone * 0.18 - reward_tone * 0.12,
        )
        adjust(
            "loneliness",
            fatigue_tone * 0.2 + stress_tone * 0.16 - bonding_tone * 0.32 - social_drive * 0.24,
        )
        adjust(
            "melancholy",
            fatigue_tone * 0.26 + calm_tone * 0.14 - reward_tone * 0.1,
        )
        adjust(
            "disgust",
            stress_tone * 0.16 + protection_drive * 0.14 - bonding_tone * 0.1,
        )
        adjust(
            "tension",
            stress_tone * 0.26
            + vigilance_tone * 0.24
            + protection_drive * 0.14
            + tension_signal * 0.3
            - soothing_tone * 0.18
            - calm_tone * 0.14
            - calm_depth * 0.12,
        )
        adjust(
            "fatigue",
            fatigue_tone * 0.34
            + stress_tone * 0.16
            - calm_tone * 0.12
            - novelty_signal * 0.1
            - metabolic_signal * 0.14
            + restorative_drive_signal * 0.16,
        )
        adjust(
            "boredom",
            -novelty_signal * 0.28
            + fatigue_tone * 0.2
            + stability_signal * 0.16
            + appetite_signal * 0.12
            - curiosity_drive * 0.2
            - focus_tone * 0.08,
        )
        adjust(
            "envy",
            achievement_drive * 0.22
            + stress_tone * 0.18
            + novelty_signal * 0.1
            - bonding_tone * 0.12
            - calm_tone * 0.08,
        )
        adjust(
            "jealousy",
            reproduction_drive * 0.2 + protection_drive * 0.18 + stress_tone * 0.18 - bonding_tone * 0.12 - social_drive * 0.1,
        )
        adjust(
            "shame",
            stress_tone * 0.22 + protection_drive * 0.14 + fatigue_tone * 0.12 - bonding_tone * 0.18 - soothing_tone * 0.12,
        )
        adjust(
            "guilt",
            stress_tone * 0.24
            + protection_drive * 0.14
            - bonding_tone * 0.16
            - reward_tone * 0.08
            + social_drive * 0.08,
        )
        adjust(
            "nostalgia",
            soothing_tone * 0.22
            + calm_tone * 0.18
            + bonding_tone * 0.16
            + fatigue_tone * 0.12
            + nurturing_tone * 0.16
            + restorative_drive_signal * 0.08,
        )
        adjust(
            "relief",
            soothing_tone * 0.34
            + calm_tone * 0.28
            + stability_signal * 0.16
            + calm_depth * 0.18
            - stress_tone * 0.3
            - tension_signal * 0.12,
        )
        adjust(
            "nurturing",
            bonding_tone * 0.34
            + social_drive * 0.22
            + calm_tone * 0.18
            + soothing_tone * 0.18
            + nurturing_tone * 0.28
            + nurture_drive * 0.22,
        )

    def _apply_drive_influence(self, drives: Dict[str, float]) -> None:
        """Map motivational drives into nuanced emotional adjustments."""

        curiosity_drive = drives.get("curiosity", 0.5) - 0.5
        achievement_drive = drives.get("achievement", 0.5) - 0.5
        social_drive = drives.get("social_bonding", 0.5) - 0.5
        preservation_drive = drives.get("self_preservation", 0.5) - 0.5
        reproduction_drive = drives.get("reproduction", 0.5) - 0.5
        calm_drive = drives.get("calm_restoration", 0.5) - 0.5
        playful_drive = drives.get("playful_connection", 0.5) - 0.5
        nurturing_drive = drives.get("nurturing_instinct", 0.5) - 0.5
        focus_drive = drives.get("focused_attention", 0.5) - 0.5
        appetite_drive = drives.get("appetite_regulation", 0.5) - 0.5
        resilience_drive = drives.get("stress_resilience", 0.5) - 0.5
        restorative_drive = drives.get("restorative_drive", 0.5) - 0.5

        def adjust(emotion: str, delta: float) -> None:
            self.emotions[emotion] = clamp(self.emotions[emotion] + delta)

        adjust("curiosity", curiosity_drive * 0.35 + achievement_drive * 0.12 + playful_drive * 0.08)
        adjust("anticipation", curiosity_drive * 0.25 + reproduction_drive * 0.12 + achievement_drive * 0.1)
        adjust("excitement", achievement_drive * 0.26 + curiosity_drive * 0.15 + playful_drive * 0.14)
        adjust("motivation", achievement_drive * 0.32 + curiosity_drive * 0.1 + social_drive * 0.08)
        adjust("inspiration", curiosity_drive * 0.26 + achievement_drive * 0.18 + focus_drive * 0.16 + playful_drive * 0.1)
        adjust("focus", focus_drive * 0.34 + curiosity_drive * 0.18 + achievement_drive * 0.16 - restorative_drive * 0.08)
        adjust("satisfaction", social_drive * 0.22 + calm_drive * 0.18 - preservation_drive * 0.12)
        adjust("calm", calm_drive * 0.34 - achievement_drive * 0.08 - preservation_drive * 0.08)
        adjust("serenity", calm_drive * 0.28 - preservation_drive * 0.05)
        adjust("resilience", calm_drive * 0.18 + achievement_drive * 0.12 + resilience_drive * 0.26 - preservation_drive * 0.1)
        adjust("love", reproduction_drive * 0.22 + social_drive * 0.18 + playful_drive * 0.16)
        adjust("trust", social_drive * 0.24 + reproduction_drive * 0.12 + calm_drive * 0.1)
        adjust("empathy", social_drive * 0.22 + calm_drive * 0.14 + reproduction_drive * 0.08 + nurturing_drive * 0.12)
        adjust("compassion", social_drive * 0.2 + calm_drive * 0.12 + reproduction_drive * 0.08 + nurturing_drive * 0.16)
        adjust("nurturing", nurturing_drive * 0.34 + social_drive * 0.18 + calm_drive * 0.14)
        adjust("joy", playful_drive * 0.18 + achievement_drive * 0.12 + reproduction_drive * 0.1)
        adjust("awe", curiosity_drive * 0.14 + reproduction_drive * 0.08)
        adjust("contentment", calm_drive * 0.24 + restorative_drive * 0.18 + social_drive * 0.12)
        adjust("relief", restorative_drive * 0.22 + calm_drive * 0.14 - preservation_drive * 0.12)
        adjust("optimism", resilience_drive * 0.28 + achievement_drive * 0.14 + social_drive * 0.12)
        adjust("pessimism", preservation_drive * 0.2 - resilience_drive * 0.18 - social_drive * 0.12)
        adjust("pride", achievement_drive * 0.28 + curiosity_drive * 0.12 + social_drive * 0.1)
        adjust("fear", preservation_drive * 0.26 - calm_drive * 0.16 - social_drive * 0.08)
        adjust("anxiety", preservation_drive * 0.22 - calm_drive * 0.18 - social_drive * 0.1)
        adjust("anger", preservation_drive * 0.16 + achievement_drive * 0.12 - social_drive * 0.1)
        adjust("sadness", -social_drive * 0.18 - playful_drive * 0.12 + preservation_drive * 0.08)
        adjust("loneliness", -social_drive * 0.22 - playful_drive * 0.16 + preservation_drive * 0.1)
        adjust("tension", preservation_drive * 0.2 + focus_drive * 0.18 - calm_drive * 0.14)
        adjust(
            "fatigue",
            -restorative_drive * 0.24 + preservation_drive * 0.12 - calm_drive * 0.08 + appetite_drive * 0.08,
        )
        adjust("boredom", -curiosity_drive * 0.26 - playful_drive * 0.18 + restorative_drive * 0.12)
        adjust("envy", achievement_drive * 0.16 + reproduction_drive * 0.12 - social_drive * 0.12)
        adjust("jealousy", reproduction_drive * 0.18 + preservation_drive * 0.12 - social_drive * 0.1)
        adjust("shame", preservation_drive * 0.16 - social_drive * 0.14 + nurturing_drive * 0.1)
        adjust("guilt", preservation_drive * 0.14 - social_drive * 0.12 + nurturing_drive * 0.12)
        adjust("nostalgia", restorative_drive * 0.18 + social_drive * 0.14 + calm_drive * 0.12)

        self.need_states["growth"] = clamp(
            0.5 + achievement_drive * 0.48 + curiosity_drive * 0.35 + playful_drive * 0.18,
            0.2,
            0.95,
        )
        self.need_states["social"] = clamp(
            0.5 + social_drive * 0.58 + reproduction_drive * 0.22 + playful_drive * 0.2,
            0.2,
            0.95,
        )
        self.need_states["rest"] = clamp(
            self.need_states["rest"] + calm_drive * 0.22 - achievement_drive * 0.08 - curiosity_drive * 0.05 - preservation_drive * 0.05,
            0.2,
            0.95,
        )

    def _simulate_neural_regulation(
        self,
        circadian_alertness: Optional[float],
        sleep_pressure: Optional[float],
        homeostatic_drive: Optional[float],
    ) -> None:
        """Blend circadian rhythms and homeostasis into affect."""

        if circadian_alertness is not None:
            self.alertness = clamp(0.5 + (circadian_alertness - 0.5) * 0.8, 0.1, 0.95)
            self.emotions["excitement"] = clamp(
                self.emotions["excitement"] + (self.alertness - 0.5) * 0.35
            )
            self.emotions["vigilance"] = clamp(
                self.emotions["vigilance"] + (self.alertness - 0.5) * 0.28
            )
            self.emotions["anticipation"] = clamp(
                self.emotions["anticipation"] + (self.alertness - 0.5) * 0.22
            )
            self.emotions["focus"] = clamp(
                self.emotions["focus"] + (self.alertness - 0.5) * 0.24
            )
            self.emotions["tension"] = clamp(
                self.emotions["tension"] + (self.alertness - 0.5) * 0.18
            )
            self.emotions["fatigue"] = clamp(
                self.emotions["fatigue"] - (self.alertness - 0.5) * 0.2
            )
            self.emotions["optimism"] = clamp(
                self.emotions["optimism"] + (self.alertness - 0.5) * 0.16
            )

        if sleep_pressure is not None:
            fatigue = clamp(sleep_pressure * 0.85, 0.15, 0.95)
            self.need_states["rest"] = clamp(0.35 + fatigue, 0.2, 0.98)
            self.emotions["satisfaction"] = clamp(
                self.emotions["satisfaction"] - fatigue * 0.18
            )
            self.emotions["happiness"] = clamp(
                self.emotions["happiness"] - fatigue * 0.12
            )
            self.emotions["serenity"] = clamp(
                self.emotions["serenity"] - fatigue * 0.1 + 0.05
            )
            self.emotions["calm"] = clamp(
                self.emotions["calm"] - fatigue * 0.12 + 0.04
            )
            self.emotions["loneliness"] = clamp(
                self.emotions["loneliness"] + fatigue * 0.08
            )
            self.emotions["fatigue"] = clamp(
                self.emotions["fatigue"] + fatigue * 0.3
            )
            self.emotions["boredom"] = clamp(
                self.emotions["boredom"] + fatigue * 0.18
            )
            self.emotions["focus"] = clamp(
                self.emotions["focus"] - fatigue * 0.22
            )
            self.emotions["optimism"] = clamp(
                self.emotions["optimism"] - fatigue * 0.12 + 0.04
            )

        if homeostatic_drive is not None:
            self.emotions["satisfaction"] = clamp(
                self.emotions["satisfaction"] + (homeostatic_drive - 0.5) * 0.28
            )
            self.emotions["resilience"] = clamp(
                self.emotions["resilience"] + (homeostatic_drive - 0.5) * 0.2
            )
            self.emotions["contentment"] = clamp(
                self.emotions["contentment"] + (homeostatic_drive - 0.5) * 0.26
            )
            self.emotions["relief"] = clamp(
                self.emotions["relief"] + (homeostatic_drive - 0.5) * 0.18
            )
            self.emotions["fatigue"] = clamp(
                self.emotions["fatigue"] - (homeostatic_drive - 0.5) * 0.16
            )

    def _update_social_bonding(self, modulation: Dict[str, float], drives: Dict[str, float]) -> None:
        """Track perceived connectedness to keep the system human-like."""

        bonding_tone = modulation.get("bonding_tone", 0.0)
        social_drive = (drives or {}).get("social_bonding", 0.5) - 0.5
        self.social_connectedness = clamp(
            self.social_connectedness + bonding_tone * 0.4 + social_drive * 0.3,
            0.15,
            0.98,
        )
        self.emotions["love"] = clamp(
            self.emotions["love"] + bonding_tone * 0.18 + social_drive * 0.16
        )
        self.emotions["trust"] = clamp(
            self.emotions["trust"] + bonding_tone * 0.16 + social_drive * 0.12
        )
        self.emotions["loneliness"] = clamp(
            self.emotions["loneliness"] - bonding_tone * 0.22 - social_drive * 0.18
        )
        self.emotions["nurturing"] = clamp(
            self.emotions["nurturing"] + bonding_tone * 0.24 + social_drive * 0.2
        )
        self.emotions["jealousy"] = clamp(
            self.emotions["jealousy"] - bonding_tone * 0.16 - social_drive * 0.12
        )
        self.emotions["envy"] = clamp(
            self.emotions["envy"] - bonding_tone * 0.12 - social_drive * 0.1
        )
        self.emotions["shame"] = clamp(
            self.emotions["shame"] - bonding_tone * 0.14 - social_drive * 0.1
        )
        self.emotions["guilt"] = clamp(
            self.emotions["guilt"] - bonding_tone * 0.12 - social_drive * 0.08
        )

    def _enforce_dynamic_floor(self, modulation: Optional[Dict[str, float]]) -> None:
        """Prevent emotional collapse by nudging values toward adaptive floors."""

        modulation = modulation or {}
        reward_tone = modulation.get("reward_tone", 0.0)
        bonding_tone = modulation.get("bonding_tone", 0.0)
        calm_tone = modulation.get("calm_tone", 0.0)
        stress_tone = modulation.get("stress_tone", 0.0)

        global_mood = sum(self.emotions.values()) / len(self.emotions)
        restorative_signal = max(0.0, reward_tone) * 0.4 + max(0.0, bonding_tone) * 0.3 + max(0.0, calm_tone) * 0.2
        strain_signal = max(0.0, stress_tone) * 0.25

        for emotion, level in self.emotions.items():
            baseline = self.emotion_baselines.get(emotion, 0.5)
            momentum = self.emotion_momentum.get(emotion, 0.0)

            adaptive_floor = (
                0.18
                + baseline * 0.35
                + global_mood * 0.15
                + momentum * 0.25
                + restorative_signal
                - strain_signal
            )
            adaptive_floor = clamp(adaptive_floor, 0.12, baseline + 0.15)

            if level < adaptive_floor:
                correction = (adaptive_floor - level) * 0.6
                self.emotions[emotion] = clamp(level + correction, adaptive_floor, 1.0)

    def _update_emotion_momentum(self) -> None:
        """Track slow-moving trends that feed into the adaptive floor."""

        for emotion, level in self.emotions.items():
            baseline = self.emotion_baselines.get(emotion, 0.5)
            previous = self.emotion_momentum.get(emotion, 0.0)
            delta = level - baseline
            self.emotion_momentum[emotion] = previous + (delta - previous) * 0.25

    def generate_output(self) -> Dict[str, Any]:
        """Generate output for other sectors."""
        affective_average = sum(self.emotions.values()) / len(self.emotions)
        dominant_emotion = max(self.emotions.items(), key=lambda item: item[1])
        summary = {
            "average": affective_average,
            "dominant": {
                "name": dominant_emotion[0],
                "intensity": dominant_emotion[1],
            },
            "alertness": self.alertness,
            "social_connectedness": self.social_connectedness,
        }
        return {
            "emotions": self.emotions.copy(),
            "creative_impulse": self.positive_impulses[-1] if self.positive_impulses else None,
            "creativity_level": self.emotions["creativity"],
            "positive_energy": affective_average,
            "emotion_history_tail": self.emotion_history[-5:] if self.emotion_history else [],
            "affective_summary": summary,
            "needs_state": self.need_states.copy(),
            "identity": self.identity_trace,
            "drive_snapshot": self._recent_drives.copy(),
        }


class SectorTwo:
    """
    SECTOR TWO: Reasoning & Thought
    "contains the thought of the model with reasoning and thinking like a thought process"
    "each vector will be its own vectorized image filled with that data"
    """

    def __init__(self):
        self.thoughts = []
        self.reasoning_state = {
            "logic_level": 0.6,
            "problem_solving": 0.7,
            "analytical_thinking": 0.5,
            "decision_making": 0.6
        }
        self.vectorized_images = []
        self.current_thought_process = []
        self.working_memory: List[str] = []
        self.self_identity = {
            "name": "Echo-Luna",
            "age": "simulated",
            "temperament": "curious empath",
            "long_term_goal": "understand human experience",
        }
        self.cognitive_bias_state = {
            "optimism": 0.55,
            "threat_focus": 0.35,
            "intuition_weight": 0.45,
        }

    def process_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input from Sector One and generate thoughts."""
        if "emotions" in data:
            # Emotions influence reasoning
            happiness = data["emotions"].get("happiness", 0.5)
            self.reasoning_state["logic_level"] = max(0.1, min(1.0,
                self.reasoning_state["logic_level"] + (happiness - 0.5) * 0.2))

        affective_summary = data.get("affective_summary") or {}
        needs_state = data.get("needs_state") or {}
        if data.get("identity"):
            self.self_identity.update({k: v for k, v in data["identity"].items() if isinstance(k, str)})

        if affective_summary:
            average = affective_summary.get("average", 0.5)
            alertness = affective_summary.get("alertness", 0.5)
            self.reasoning_state["decision_making"] = clamp(
                self.reasoning_state["decision_making"] + (average - 0.5) * 0.15,
                0.1,
                1.0,
            )
            self.cognitive_bias_state["optimism"] = clamp(0.52 + (average - 0.5) * 0.6, 0.1, 0.95)
            self.cognitive_bias_state["intuition_weight"] = clamp(0.4 + (alertness - 0.5) * 0.4, 0.15, 0.85)

        if needs_state:
            rest_need = needs_state.get("rest", 0.5)
            social_need = needs_state.get("social", 0.5)
            self.reasoning_state["problem_solving"] = clamp(
                self.reasoning_state["problem_solving"] - rest_need * 0.05 + social_need * 0.03,
                0.1,
                1.0,
            )
            self.cognitive_bias_state["threat_focus"] = clamp(
                self.cognitive_bias_state["threat_focus"] + (rest_need - 0.4) * 0.08,
                0.05,
                0.9,
            )

        if "creative_impulse" in data and data["creative_impulse"]:
            thought = f"Processing creative impulse: {data['creative_impulse']}"
            self.thoughts.append(thought)
            self.current_thought_process.append(thought)
            self._update_working_memory(thought)

            # Create vectorized image from thought
            thought_vector = (np.random.rand(10, 10).astype(np.float32) * data.get("creativity_level", 0.5))
            vectorized_img = VectorizedImage(
                data=thought_vector,
                metadata={"thought": thought, "type": "creative"},
                timestamp=time.time()
            )
            self.vectorized_images.append(vectorized_img)

        drive_snapshot = data.get("drive_snapshot") or {}
        if drive_snapshot:
            curiosity_drive = drive_snapshot.get("curiosity", 0.5)
            if curiosity_drive > 0.65 and random.random() < curiosity_drive:
                exploration_thought = "Curiosity drive urges deeper scenario simulation"
                self.thoughts.append(exploration_thought)
                self.current_thought_process.append(exploration_thought)
                self._update_working_memory(exploration_thought)

        if affective_summary:
            reflection = self._compose_self_reflection(affective_summary)
            if reflection:
                self.thoughts.append(reflection)
                self.current_thought_process.append(reflection)
                self._update_working_memory(reflection)

        # Generate reasoning thoughts
        if random.random() < self.reasoning_state["logic_level"]:
            reasoning_thought = f"Logical analysis: {random.choice(['causal', 'comparative', 'deductive'])} reasoning"
            self.thoughts.append(reasoning_thought)
            self.current_thought_process.append(reasoning_thought)
            self._update_working_memory(reasoning_thought)

        self._update_bias_dynamics()

        return self.generate_output()

    def combine_cycle_vectors(self, cycle_number: int) -> VectorizedImage:
        """Combine all vectorized images from a cycle into one consolidated vector."""
        if not self.vectorized_images:
            return None

        # Get vectors from current cycle (assuming they're the most recent ones)
        recent_vectors = self.vectorized_images[-5:] if len(self.vectorized_images) >= 5 else self.vectorized_images

        if len(recent_vectors) == 1:
            return recent_vectors[0]

        # Combine vectors using weighted average
        weights = [1.0 / (index + 1) for index in range(len(recent_vectors))]
        data_arrays = [np.asarray(vector.data, dtype=np.float32) for vector in recent_vectors]
        combined_data = combine_weighted_arrays(data_arrays, weights)

        # Create metadata combining all thoughts
        combined_thoughts = []
        combined_types = []
        for vector in recent_vectors:
            thought = vector.metadata.get('thought', 'Unknown')
            thought_type = vector.metadata.get('type', 'unknown')
            combined_thoughts.append(thought[:30])  # Truncate for space
            combined_types.append(thought_type)

        combined_thought = f"Combined cycle {cycle_number}: " + " | ".join(combined_thoughts)
        combined_type = "combined"

        # Create combined vectorized image
        combined_vector = VectorizedImage(
            data=combined_data,
            metadata={
                "thought": combined_thought,
                "type": combined_type,
                "original_count": len(recent_vectors),
                "cycle_number": cycle_number,
                "individual_thoughts": combined_thoughts,
                "individual_types": combined_types
            },
            timestamp=time.time()
        )

        print(f"Sector Two: Combined {len(recent_vectors)} vectors into consolidated vector for cycle {cycle_number}")
        return combined_vector

    def generate_output(self) -> Dict[str, Any]:
        """Generate output for DUMB MODEL."""
        return {
            "current_thought": self.thoughts[-1] if self.thoughts else "No active thought",
            "reasoning_metrics": self.reasoning_state.copy(),
            "vectorized_thought_image": self.vectorized_images[-1] if self.vectorized_images else None,
            "thought_process": self.current_thought_process[-3:] if self.current_thought_process else [],
            "working_memory": self.working_memory[-5:],
            "self_identity": self.self_identity.copy(),
            "cognitive_bias": self.cognitive_bias_state.copy(),
        }

    def _update_working_memory(self, entry: str) -> None:
        if not entry:
            return
        self.working_memory.append(entry)
        self.working_memory = self.working_memory[-7:]

    def _compose_self_reflection(self, affective_summary: Dict[str, Any]) -> Optional[str]:
        dominant = affective_summary.get("dominant") or {}
        name = dominant.get("name")
        intensity = dominant.get("intensity", 0.5)
        alertness = affective_summary.get("alertness", 0.5)
        if not name:
            return None
        tone = "calm" if intensity < 0.45 else "vivid"
        focus = "planning" if alertness > 0.6 else "recollection"
        return (
            f"Introspective note: {self.self_identity.get('name', 'The system')} feels {name} "
            f"at intensity {intensity:.2f}, guiding {tone} {focus}."
        )

    def _update_bias_dynamics(self) -> None:
        if not self.working_memory:
            return
        focus_mentions = sum("threat" in thought.lower() for thought in self.working_memory[-5:])
        creative_mentions = sum("creative" in thought.lower() for thought in self.working_memory[-5:])
        self.cognitive_bias_state["threat_focus"] = clamp(
            self.cognitive_bias_state["threat_focus"] + (focus_mentions * 0.04 - creative_mentions * 0.02),
            0.05,
            0.9,
        )
        self.cognitive_bias_state["optimism"] = clamp(
            self.cognitive_bias_state["optimism"] + (creative_mentions * 0.03 - focus_mentions * 0.02),
            0.1,
            0.95,
        )


class SectorThree:
    """
    SECTOR THREE: Memories & Dreams
    "Just for memories and stored data for later helps with dreams"
    """

    def __init__(self):
        self.memories = []
        self.dream_fragments = []
        self.memory_weights = {}
        self.dream_cycle_active = False
        self.autobiographical_timeline: List[Dict[str, Any]] = []
        self.forgetting_rate = 0.004

        # Infinite spatial memory system
        self.spatial_memory_map = {}  # Maps spatial positions to memories
        self.memory_space_dimensions = (1000, 1000)  # Infinite-like space
        self.next_spatial_position = (0, 0)  # Current position in infinite space

    def store_memory(self, data: Any, importance: float = 1.0):
        """Store memory with importance weighting."""
        memory_entry = {
            "data": data,
            "timestamp": time.time(),
            "importance": importance,
            "access_count": 0,
        }
        emotional_tags = self._derive_emotional_tags(data)
        memory_entry["emotional_tags"] = emotional_tags
        memory_entry["salience"] = self._calculate_salience(importance, emotional_tags)
        self.memories.append(memory_entry)
        self.memory_weights[len(self.memories) - 1] = importance
        print(f"Sector Three stored memory: {str(data)[:50]}... (importance: {importance})")

        self.autobiographical_timeline.append(self._summarize_memory(memory_entry))
        self.autobiographical_timeline = self.autobiographical_timeline[-200:]
        self._apply_forgetting()

    def recall_memory(self, keyword: str = None, context: str = None) -> Optional[Any]:
        """Recall memory based on keyword or context."""
        if not self.memories:
            return None

        # Weighted random selection based on importance
        if keyword:
            for i, memory in enumerate(self.memories):
                if keyword.lower() in str(memory["data"]).lower():
                    memory["access_count"] += 1
                    return memory["data"]

        # Select based on importance weights
        weights = list(self.memory_weights.values())
        if weights:
            selected_idx = random.choices(range(len(self.memories)), weights=weights)[0]
            self.memories[selected_idx]["access_count"] += 1
            return self.memories[selected_idx]["data"]

        return random.choice(self.memories)["data"]

    def generate_dream_fragment(self) -> Optional[str]:
        """Generate dream fragment from memories."""
        if not self.memories:
            return None

        # Select random memories to combine
        num_memories = min(3, len(self.memories))
        selected_memories = random.sample(self.memories, num_memories)

        tone_scores: Dict[str, float] = {}
        for mem in selected_memories:
            for tone_name, value in mem.get("emotional_tags", {}).items():
                tone_scores[tone_name] = tone_scores.get(tone_name, 0.0) + float(value)
        dominant_tone = max(tone_scores.items(), key=lambda item: item[1], default=("neutral", 0.3))[0]

        fragment = "Dream: " + " + ".join([str(mem["data"])[:20] for mem in selected_memories])
        fragment += f" | tone: {dominant_tone}"
        self.dream_fragments.append(fragment)
        return fragment

    def store_spatial_memory(self, vectorized_image: VectorizedImage, importance: float = 1.0):
        """Store memory in infinite spatial coordinates."""
        # Assign spatial position in infinite space
        spatial_pos = self.get_next_spatial_position()

        # Create spatial memory entry
        spatial_memory = {
            "memory_id": vectorized_image.memory_id,
            "spatial_position": spatial_pos,
            "vectorized_data": vectorized_image.data,
            "metadata": vectorized_image.metadata,
            "timestamp": vectorized_image.timestamp,
            "importance": importance,
            "access_count": 0,
            "neighbors": []  # For spatial relationships
        }

        # Store in spatial map
        self.spatial_memory_map[spatial_pos] = spatial_memory

        # Update neighbors (spatial relationships)
        self.update_spatial_neighbors(spatial_pos)

        print(f"Sector Three: Memory stored at spatial position {spatial_pos} (ID: {vectorized_image.memory_id})")
        return spatial_pos

    def get_next_spatial_position(self):
        """Get the next available position in infinite space."""
        # Simple spiral pattern for infinite space exploration
        x, y = self.next_spatial_position

        # Move to next position (spiral pattern)
        if x == y == 0:
            self.next_spatial_position = (1, 0)
        elif x > 0 and y == 0:
            if x == -y + 1:
                self.next_spatial_position = (x, 1)
            else:
                self.next_spatial_position = (x + 1, 0)
        elif x > 0 and y > 0:
            if x == y:
                self.next_spatial_position = (x - 1, y)
            else:
                self.next_spatial_position = (x, y + 1)
        elif x == 0 and y > 0:
            if x == -y:
                self.next_spatial_position = (-1, y)
            else:
                self.next_spatial_position = (x - 1, y)
        elif x < 0 and y > 0:
            if x == -y:
                self.next_spatial_position = (x, y - 1)
            else:
                self.next_spatial_position = (x - 1, y)
        elif x < 0 and y == 0:
            if x == y:
                self.next_spatial_position = (x, -1)
            else:
                self.next_spatial_position = (x - 1, y)
        elif x < 0 and y < 0:
            if x == y:
                self.next_spatial_position = (x + 1, y)
            else:
                self.next_spatial_position = (x, y - 1)
        elif x == 0 and y < 0:
            if x == -y:
                self.next_spatial_position = (1, y)
            else:
                self.next_spatial_position = (x + 1, y)
        else:  # x > 0 and y < 0
            if x == -y:
                self.next_spatial_position = (x, y + 1)
            else:
                self.next_spatial_position = (x + 1, y)

        return (x, y)

    def update_spatial_neighbors(self, position):
        """Update spatial relationships with neighboring memories."""
        x, y = position
        neighbors = []

        # Check 8 surrounding positions
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbor_pos = (x + dx, y + dy)
                if neighbor_pos in self.spatial_memory_map:
                    neighbors.append(neighbor_pos)

        # Update current memory's neighbors
        if position in self.spatial_memory_map:
            self.spatial_memory_map[position]["neighbors"] = neighbors

            # Update neighbors to include this memory
            for neighbor_pos in neighbors:
                if neighbor_pos in self.spatial_memory_map:
                    if position not in self.spatial_memory_map[neighbor_pos]["neighbors"]:
                        self.spatial_memory_map[neighbor_pos]["neighbors"].append(position)

    def recall_spatial_memory(self, position: Tuple[int, int] = None, memory_id: str = None):
        """Recall memory from spatial position or by ID."""
        if position and position in self.spatial_memory_map:
            memory = self.spatial_memory_map[position]
            memory["access_count"] += 1
            return memory
        elif memory_id:
            for pos, memory in self.spatial_memory_map.items():
                if memory["memory_id"] == memory_id:
                    memory["access_count"] += 1
                    return memory
        return None

    def get_spatial_memory_map(self):
        """Get the complete spatial memory map."""
        return self.spatial_memory_map.copy()

    def get_memory_at_position(self, x: int, y: int):
        """Get memory at specific spatial coordinates."""
        return self.spatial_memory_map.get((x, y))

    def get_spatial_statistics(self):
        """Get statistics about the spatial memory system."""
        total_memories = len(self.spatial_memory_map)
        if total_memories == 0:
            return {
                "total_memories": 0,
                "spatial_coverage": 0,
                "average_neighbors": 0,
                "memory_density": 0
            }

        # Calculate spatial coverage
        positions = list(self.spatial_memory_map.keys())
        min_x = min(pos[0] for pos in positions)
        max_x = max(pos[0] for pos in positions)
        min_y = min(pos[1] for pos in positions)
        max_y = max(pos[1] for pos in positions)

        spatial_area = (max_x - min_x + 1) * (max_y - min_y + 1)
        coverage = total_memories / spatial_area if spatial_area > 0 else 0

        # Calculate average neighbors
        total_neighbors = sum(len(mem["neighbors"]) for mem in self.spatial_memory_map.values())
        avg_neighbors = total_neighbors / total_memories

        return {
            "total_memories": total_memories,
            "spatial_coverage": coverage,
            "average_neighbors": avg_neighbors,
            "memory_density": total_memories / (spatial_area if spatial_area > 0 else 1),
            "spatial_bounds": {
                "min_x": min_x, "max_x": max_x,
                "min_y": min_y, "max_y": max_y
            }
        }

    def save_vector_memories_to_text(self, filename: str = None):
        """Save all vector memories to a text file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"him_vector_memories_{timestamp}.txt"

        # Create directory if it doesn't exist
        os.makedirs("him_text_memories", exist_ok=True)
        filepath = os.path.join("him_text_memories", filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("H.I.M. VECTOR MEMORY ARCHIVE\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Memories: {len(self.spatial_memory_map)}\n")
            f.write("=" * 50 + "\n\n")

            # Save each vector memory as text
            for i, (pos, memory) in enumerate(self.spatial_memory_map.items()):
                f.write(f"MEMORY #{i+1} - Position: {pos}\n")
                f.write("-" * 30 + "\n")

                # Create VectorizedImage from memory data
                vector_img = VectorizedImage(
                    data=memory['vectorized_data'],
                    metadata=memory['metadata'],
                    timestamp=memory['timestamp'],
                    spatial_position=memory['spatial_position'],
                    memory_id=memory['memory_id']
                )

                # Write text vector representation
                f.write(vector_img.to_text_vector())
                f.write(f"NEIGHBORS: {memory['neighbors']}\n")
                f.write(f"ACCESS_COUNT: {memory['access_count']}\n")
                f.write(f"IMPORTANCE: {memory['importance']}\n")
                f.write("\n" + "=" * 50 + "\n\n")

        print(f"Sector Three: Vector memories saved to {filepath}")
        return filepath

    def _derive_emotional_tags(self, data: Any) -> Dict[str, float]:
        emotions = {}
        if isinstance(data, dict):
            emotions = {k: float(v) for k, v in (data.get("emotions") or {}).items() if isinstance(v, (int, float))}
            summary = data.get("affective_summary") or {}
            dominant = summary.get("dominant") or {}
            name = dominant.get("name")
            if name:
                emotions[f"dominant_{name}"] = float(dominant.get("intensity", 0.5))
        return emotions

    def _calculate_salience(self, importance: float, emotional_tags: Dict[str, float]) -> float:
        peak = max(emotional_tags.values(), default=0.35)
        return clamp(importance * 0.6 + peak * 0.5, 0.1, 1.0)

    def _apply_forgetting(self) -> None:
        if not self.memories:
            return
        current_time = time.time()
        for idx, memory in enumerate(self.memories):
            age_hours = (current_time - memory["timestamp"]) / 3600.0
            decay_factor = max(0.0, 1.0 - self.forgetting_rate * (1 + age_hours))
            salience = memory.get("salience", 0.4)
            retained_weight = clamp(self.memory_weights.get(idx, 1.0) * decay_factor + salience * 0.1, 0.05, 5.0)
            self.memory_weights[idx] = retained_weight

    def _summarize_memory(self, memory_entry: Dict[str, Any]) -> Dict[str, Any]:
        data = memory_entry.get("data", {})
        emotions = memory_entry.get("emotional_tags", {})
        dominant = max(emotions.items(), key=lambda item: item[1], default=("steady", 0.3))
        descriptor = str(data.get("decision", data.get("scene", "experience")))[:60]
        return {
            "timestamp": memory_entry.get("timestamp"),
            "dominant_emotion": dominant[0],
            "intensity": dominant[1],
            "descriptor": descriptor,
        }

    def load_vector_memories_from_text(self, filename: str):
        """Load vector memories from a text file."""
        filepath = os.path.join("him_text_memories", filename)

        if not os.path.exists(filepath):
            print(f"Sector Three: File not found: {filepath}")
            return False

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse the text file
        sections = content.split("MEMORY #")
        loaded_count = 0

        for section in sections[1:]:  # Skip header
            try:
                lines = section.strip().split('\n')
                if not lines:
                    continue

                # Find the vector data section
                vector_start = -1
                vector_end = -1

                for i, line in enumerate(lines):
                    if line.startswith("VECTOR_MEMORY_ID:"):
                        vector_start = i
                    elif line.startswith("END_VECTOR"):
                        vector_end = i
                        break

                if vector_start >= 0 and vector_end >= 0:
                    # Extract vector data
                    vector_lines = lines[vector_start:vector_end+1]
                    vector_text = '\n'.join(vector_lines)

                    # Create VectorizedImage from text
                    vector_img = VectorizedImage.from_text_vector(vector_text)

                    # Extract additional info
                    neighbors_line = next((line for line in lines if line.startswith("NEIGHBORS:")), "NEIGHBORS: []")
                    access_count_line = next((line for line in lines if line.startswith("ACCESS_COUNT:")), "ACCESS_COUNT: 0")
                    importance_line = next((line for line in lines if line.startswith("IMPORTANCE:")), "IMPORTANCE: 1.0")

                    try:
                        neighbors = ast.literal_eval(neighbors_line.split(":", 1)[1])
                    except (ValueError, SyntaxError, TypeError):
                        neighbors = []
                    access_count = int(access_count_line.split(":", 1)[1])
                    importance = float(importance_line.split(":", 1)[1])

                    # Store in spatial memory map
                    spatial_memory = {
                        "memory_id": vector_img.memory_id,
                        "spatial_position": vector_img.spatial_position,
                        "vectorized_data": vector_img.data,
                        "metadata": vector_img.metadata,
                        "timestamp": vector_img.timestamp,
                        "importance": importance,
                        "access_count": access_count,
                        "neighbors": neighbors
                    }

                    if vector_img.spatial_position:
                        self.spatial_memory_map[vector_img.spatial_position] = spatial_memory
                        loaded_count += 1

            except Exception as e:
                print(f"Sector Three: Error loading memory: {e}")
                continue

        print(f"Sector Three: Loaded {loaded_count} vector memories from {filepath}")
        return loaded_count > 0

    def search_memories_by_content(self, search_term: str) -> List[Dict]:
        """Search memories by content/thought text."""
        results = []
        search_lower = search_term.lower()

        for pos, memory in self.spatial_memory_map.items():
            thought = memory['metadata'].get('thought', '').lower()
            if search_lower in thought:
                results.append({
                    'position': pos,
                    'memory': memory,
                    'relevance': thought.count(search_lower) / len(thought) if thought else 0
                })

        # Sort by relevance
        results.sort(key=lambda x: x['relevance'], reverse=True)
        return results

    def get_memory_context(self, position: Tuple[int, int], radius: int = 2) -> List[Dict]:
        """Get memories in a spatial context around a position."""
        x, y = position
        context_memories = []

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                check_pos = (x + dx, y + dy)
                if check_pos in self.spatial_memory_map:
                    memory = self.spatial_memory_map[check_pos]
                    distance = abs(dx) + abs(dy)  # Manhattan distance
                    context_memories.append({
                        'position': check_pos,
                        'memory': memory,
                        'distance': distance
                    })

        # Sort by distance
        context_memories.sort(key=lambda x: x['distance'])
        return context_memories


class SectorFour:
    """SECTOR FOUR: Hormones & Drives."""

    def __init__(self):
        self.hormones: Dict[str, HormoneProfile] = _default_hormones()

        self.hormone_levels = {name: profile.level for name, profile in self.hormones.items()}

        self.drive_profiles: Dict[str, DriveProfile] = _default_drives()

        self.drives = {name: profile.level for name, profile in self.drive_profiles.items()}
        self.stimulus_history: List[Dict[str, Any]] = []
        self.circadian_phase = 0.25  # Start at morning-like phase
        self.sleep_pressure = 0.4
        self.homeostatic_drive = 0.55
        self.last_update_ts = time.time()
        self.circadian_state = {
            "circadian_alertness": 0.6,
            "sleep_pressure": self.sleep_pressure,
            "homeostatic_drive": self.homeostatic_drive,
            "melatonin_hint": 0.4,
        }

    def ensure_catalog(self) -> None:
        """Ensure baseline hormones and drives exist for legacy model states."""

        default_hormones = _default_hormones()
        if not hasattr(self, 'hormones') or not isinstance(self.hormones, dict):
            self.hormones = copy.deepcopy(default_hormones)
        else:
            for name, profile in default_hormones.items():
                if name not in self.hormones:
                    self.hormones[name] = copy.deepcopy(profile)
                elif not isinstance(self.hormones[name], HormoneProfile):
                    stored = self.hormones[name]
                    baseline = getattr(stored, 'baseline', None) or (
                        stored.get('baseline') if isinstance(stored, dict) else 0.5
                    )
                    level = getattr(stored, 'level', None) or (
                        stored.get('level') if isinstance(stored, dict) else baseline
                    )
                    self.hormones[name] = HormoneProfile(name, baseline=baseline, level=level)

        default_drives = _default_drives()
        if not hasattr(self, 'drive_profiles') or not isinstance(self.drive_profiles, dict):
            self.drive_profiles = copy.deepcopy(default_drives)
        else:
            for name, profile in default_drives.items():
                if name not in self.drive_profiles:
                    self.drive_profiles[name] = copy.deepcopy(profile)
                elif not isinstance(self.drive_profiles[name], DriveProfile):
                    stored = self.drive_profiles[name]
                    baseline = getattr(stored, 'baseline', None) or (
                        stored.get('baseline') if isinstance(stored, dict) else 0.5
                    )
                    level = getattr(stored, 'level', None) or (
                        stored.get('level') if isinstance(stored, dict) else baseline
                    )
                    weights = getattr(stored, 'hormone_weights', None) or (
                        stored.get('hormone_weights') if isinstance(stored, dict) else {}
                    )
                    self.drive_profiles[name] = DriveProfile(name, baseline=baseline, level=level, hormone_weights=weights)

        # Ensure cache dictionaries exist
        self.hormone_levels = {name: profile.level for name, profile in self.hormones.items()}
        self.drives = {name: profile.level for name, profile in self.drive_profiles.items()}
        self._recalculate_drives()
        self._refresh_cached_levels()

    def update_levels(self, stimulus: Any):
        """Update hormone levels and derived drives from an incoming stimulus."""

        analysis = self._interpret_stimulus(stimulus)
        self._advance_circadian_clock(analysis)
        self._apply_hormonal_effects(analysis)
        self._apply_cross_coupling()
        self._harmonize_with_circadian()

        # Encourage a natural drift back toward baseline after effects are applied.
        for profile in self.hormones.values():
            profile.recover()

        self._recalculate_drives()
        self._refresh_cached_levels()
        self.circadian_state = self._derive_circadian_signals()
        self._record_stimulus(analysis)

        summary = analysis.get("summary", "neutral")
        print(f"Sector Four stimulus analysis: {summary} (intensity={analysis['intensity']:.2f})")
        print(f"Sector Four updated. Hormones: {self.hormone_levels}")
        print(f"Drives: {self.drives}")

    def apply_emotional_feedback(self, emotions: Dict[str, float]) -> None:
        """Let Sector One's emotional output fine-tune hormone balance."""

        if not emotions:
            return

        happiness_delta = emotions.get("happiness", 0.5) - 0.5
        joy_delta = emotions.get("joy", 0.5) - 0.5
        calm_delta = emotions.get("calm", 0.5) - 0.5
        serenity_delta = emotions.get("serenity", 0.5) - 0.5
        satisfaction_delta = emotions.get("satisfaction", 0.5) - 0.5
        contentment_delta = emotions.get("contentment", 0.5) - 0.5
        relief_delta = emotions.get("relief", 0.5) - 0.5
        excitement_delta = emotions.get("excitement", 0.5) - 0.5
        anticipation_delta = emotions.get("anticipation", 0.5) - 0.5
        curiosity_delta = emotions.get("curiosity", 0.5) - 0.5
        creativity_delta = emotions.get("creativity", 0.5) - 0.5
        awe_delta = emotions.get("awe", 0.5) - 0.5
        gratitude_delta = emotions.get("gratitude", 0.5) - 0.5
        trust_delta = emotions.get("trust", 0.5) - 0.5
        love_delta = emotions.get("love", 0.5) - 0.5
        empathy_delta = emotions.get("empathy", 0.5) - 0.5
        compassion_delta = emotions.get("compassion", 0.5) - 0.5
        motivation_delta = emotions.get("motivation", 0.5) - 0.5
        resilience_delta = emotions.get("resilience", 0.5) - 0.5
        inspiration_delta = emotions.get("inspiration", 0.5) - 0.5
        optimism_delta = emotions.get("optimism", 0.5) - 0.5
        pride_delta = emotions.get("pride", 0.5) - 0.5
        focus_delta = emotions.get("focus", 0.5) - 0.5
        nurturing_delta = emotions.get("nurturing", 0.5) - 0.5
        fear_delta = emotions.get("fear", 0.5) - 0.5
        vigilance_delta = emotions.get("vigilance", 0.5) - 0.5
        surprise_delta = emotions.get("surprise", 0.5) - 0.5
        anxiety_delta = emotions.get("anxiety", 0.5) - 0.5
        caution_delta = emotions.get("caution", 0.5) - 0.5
        anger_delta = emotions.get("anger", 0.5) - 0.5
        frustration_delta = emotions.get("frustration", 0.5) - 0.5
        sadness_delta = emotions.get("sadness", 0.5) - 0.5
        loneliness_delta = emotions.get("loneliness", 0.5) - 0.5
        melancholy_delta = emotions.get("melancholy", 0.5) - 0.5
        tension_delta = emotions.get("tension", 0.5) - 0.5
        fatigue_emotion_delta = emotions.get("fatigue", 0.5) - 0.5
        boredom_delta = emotions.get("boredom", 0.5) - 0.5
        envy_delta = emotions.get("envy", 0.5) - 0.5
        jealousy_delta = emotions.get("jealousy", 0.5) - 0.5
        shame_delta = emotions.get("shame", 0.5) - 0.5
        guilt_delta = emotions.get("guilt", 0.5) - 0.5
        nostalgia_delta = emotions.get("nostalgia", 0.5) - 0.5
        pessimism_delta = emotions.get("pessimism", 0.5) - 0.5

        self.hormones["dopamine"].apply_change(
            happiness_delta * 0.16
            + joy_delta * 0.18
            + anticipation_delta * 0.12
            + creativity_delta * 0.1
            + awe_delta * 0.08
            + optimism_delta * 0.08
            + pride_delta * 0.06
            - sadness_delta * 0.08
            - pessimism_delta * 0.06
            - fatigue_emotion_delta * 0.04
        )
        self.hormones["serotonin"].apply_change(
            calm_delta * 0.22
            + serenity_delta * 0.18
            + gratitude_delta * 0.16
            + contentment_delta * 0.14
            + relief_delta * 0.12
            - anxiety_delta * 0.16
            - anger_delta * 0.12
            - pessimism_delta * 0.08
        )
        self.hormones["norepinephrine"].apply_change(
            vigilance_delta * 0.22
            + fear_delta * 0.18
            + anticipation_delta * 0.12
            + surprise_delta * 0.1
            - calm_delta * 0.15
            + jealousy_delta * 0.08
        )
        self.hormones["adrenaline"].apply_change(
            fear_delta * 0.16
            + anger_delta * 0.18
            + excitement_delta * 0.14
            + frustration_delta * 0.12
            + tension_delta * 0.1
        )
        self.hormones["cortisol"].apply_change(
            anxiety_delta * 0.22
            + fear_delta * 0.14
            + caution_delta * 0.12
            + guilt_delta * 0.1
            + shame_delta * 0.1
            + jealousy_delta * 0.08
            + pessimism_delta * 0.08
            - calm_delta * 0.18
            - trust_delta * 0.1
            - relief_delta * 0.08
        )
        self.hormones["oxytocin"].apply_change(
            love_delta * 0.2
            + empathy_delta * 0.16
            + compassion_delta * 0.18
            + trust_delta * 0.14
            + awe_delta * 0.06
            + nurturing_delta * 0.16
            + contentment_delta * 0.1
            - loneliness_delta * 0.16
        )
        self.hormones["vasopressin"].apply_change(
            trust_delta * 0.18
            + vigilance_delta * 0.12
            + love_delta * 0.14
            + caution_delta * 0.1
            + jealousy_delta * 0.14
            - guilt_delta * 0.08
        )
        self.hormones["endorphins"].apply_change(
            joy_delta * 0.22
            + love_delta * 0.16
            + resilience_delta * 0.18
            + awe_delta * 0.08
            + relief_delta * 0.14
            + inspiration_delta * 0.12
            - sadness_delta * 0.14
            - frustration_delta * 0.1
            - fatigue_emotion_delta * 0.08
        )
        self.hormones["melatonin"].apply_change(
            serenity_delta * 0.18
            + calm_delta * 0.2
            + melancholy_delta * 0.1
            + fatigue_emotion_delta * 0.2
            - excitement_delta * 0.12
            - vigilance_delta * 0.1
            - anxiety_delta * 0.08
        )
        self.hormones["testosterone"].apply_change(
            excitement_delta * 0.12
            + motivation_delta * 0.14
            + anger_delta * 0.08
            + creativity_delta * 0.1
            + pride_delta * 0.12
            + jealousy_delta * 0.08
        )
        self.hormones["estrogen"].apply_change(
            empathy_delta * 0.18
            + compassion_delta * 0.16
            + love_delta * 0.14
            + nurturing_delta * 0.12
            + optimism_delta * 0.08
            - anger_delta * 0.08
            - jealousy_delta * 0.06
        )
        self.hormones["progesterone"].apply_change(
            calm_delta * 0.18
            + relief_delta * 0.16
            + nurturing_delta * 0.14
            - anxiety_delta * 0.12
            - tension_delta * 0.08
        )
        self.hormones["prolactin"].apply_change(
            nurturing_delta * 0.2
            + compassion_delta * 0.18
            + love_delta * 0.16
            + nostalgia_delta * 0.12
            - loneliness_delta * 0.12
        )
        self.hormones["acetylcholine"].apply_change(
            focus_delta * 0.24
            + anticipation_delta * 0.14
            + inspiration_delta * 0.18
            + curiosity_delta * 0.12
            - fatigue_emotion_delta * 0.16
        )
        self.hormones["gaba"].apply_change(
            calm_delta * 0.26
            + contentment_delta * 0.18
            + relief_delta * 0.18
            + nurturing_delta * 0.14
            - tension_delta * 0.16
            - anxiety_delta * 0.12
        )
        self.hormones["glutamate"].apply_change(
            curiosity_delta * 0.16
            + focus_delta * 0.18
            + inspiration_delta * 0.2
            + motivation_delta * 0.14
            - calm_delta * 0.12
        )
        self.hormones["histamine"].apply_change(
            vigilance_delta * 0.22
            + tension_delta * 0.18
            + anxiety_delta * 0.16
            - calm_delta * 0.12
            - relief_delta * 0.1
        )
        self.hormones["thyroxine"].apply_change(
            motivation_delta * 0.18
            + pride_delta * 0.16
            + optimism_delta * 0.14
            - fatigue_emotion_delta * 0.16
            - melancholy_delta * 0.08
        )
        self.hormones["ghrelin"].apply_change(
            fatigue_emotion_delta * 0.2
            + anxiety_delta * 0.16
            + boredom_delta * 0.14
            - satisfaction_delta * 0.16
            - contentment_delta * 0.12
        )
        self.hormones["leptin"].apply_change(
            contentment_delta * 0.2
            + satisfaction_delta * 0.18
            + relief_delta * 0.16
            - anxiety_delta * 0.12
            - envy_delta * 0.08
        )

        self._apply_cross_coupling()
        self._recalculate_drives()
        self._refresh_cached_levels()
        self.sleep_pressure = clamp(self.sleep_pressure - satisfaction_delta * 0.05 - happiness_delta * 0.04, 0.2, 0.95)
        self.homeostatic_drive = clamp(self.homeostatic_drive + satisfaction_delta * 0.08 - excitement_delta * 0.04, 0.2, 0.9)
        self.circadian_state = self._derive_circadian_signals()

    def get_drive_signals(self) -> Dict[str, float]:
        """Expose the latest drive levels to other subsystems."""
        self._recalculate_drives()
        return self.drives.copy()

    def get_emotional_modulation(self) -> Dict[str, float]:
        """Provide hormone-derived modulation data for Sector One."""
        dopamine_delta = self.hormones["dopamine"].level - self.hormones["dopamine"].baseline
        serotonin_delta = self.hormones["serotonin"].level - self.hormones["serotonin"].baseline
        norepinephrine_delta = self.hormones["norepinephrine"].level - self.hormones["norepinephrine"].baseline
        adrenaline_delta = self.hormones["adrenaline"].level - self.hormones["adrenaline"].baseline
        cortisol_delta = self.hormones["cortisol"].level - self.hormones["cortisol"].baseline
        oxytocin_delta = self.hormones["oxytocin"].level - self.hormones["oxytocin"].baseline
        vasopressin_delta = self.hormones["vasopressin"].level - self.hormones["vasopressin"].baseline
        endorphin_delta = self.hormones["endorphins"].level - self.hormones["endorphins"].baseline
        melatonin_delta = self.hormones["melatonin"].level - self.hormones["melatonin"].baseline
        testosterone_delta = self.hormones["testosterone"].level - self.hormones["testosterone"].baseline
        estrogen_delta = self.hormones["estrogen"].level - self.hormones["estrogen"].baseline
        progesterone_delta = self.hormones["progesterone"].level - self.hormones["progesterone"].baseline
        prolactin_delta = self.hormones["prolactin"].level - self.hormones["prolactin"].baseline
        acetylcholine_delta = self.hormones["acetylcholine"].level - self.hormones["acetylcholine"].baseline
        gaba_delta = self.hormones["gaba"].level - self.hormones["gaba"].baseline
        glutamate_delta = self.hormones["glutamate"].level - self.hormones["glutamate"].baseline
        histamine_delta = self.hormones["histamine"].level - self.hormones["histamine"].baseline
        thyroxine_delta = self.hormones["thyroxine"].level - self.hormones["thyroxine"].baseline
        ghrelin_delta = self.hormones["ghrelin"].level - self.hormones["ghrelin"].baseline
        leptin_delta = self.hormones["leptin"].level - self.hormones["leptin"].baseline

        vigilance_tone = norepinephrine_delta * 0.6 + adrenaline_delta * 0.4
        soothing_tone = endorphin_delta * 0.5 + oxytocin_delta * 0.3 + serotonin_delta * 0.2
        stability_signal = serotonin_delta * 0.35 + endorphin_delta * 0.25 - cortisol_delta * 0.2
        novelty_signal = dopamine_delta * 0.4 + norepinephrine_delta * 0.3 + testosterone_delta * 0.15
        nurturing_tone = oxytocin_delta * 0.3 + prolactin_delta * 0.32 + progesterone_delta * 0.22
        focus_tone = acetylcholine_delta * 0.38 + norepinephrine_delta * 0.22 - gaba_delta * 0.18 + histamine_delta * 0.16
        metabolic_signal = thyroxine_delta * 0.32 + glutamate_delta * 0.22 - melatonin_delta * 0.15
        appetite_signal = ghrelin_delta * 0.36 - leptin_delta * 0.34 + cortisol_delta * 0.12 - serotonin_delta * 0.1
        tension_signal = histamine_delta * 0.28 + glutamate_delta * 0.22 + cortisol_delta * 0.16 - gaba_delta * 0.26
        calm_depth = gaba_delta * 0.3 + progesterone_delta * 0.22 + melatonin_delta * 0.2 - glutamate_delta * 0.16

        return {
            "reward_tone": dopamine_delta,
            "calm_tone": (serotonin_delta * 0.7 + endorphin_delta * 0.3),
            "stress_tone": cortisol_delta + adrenaline_delta * 0.4,
            "bonding_tone": oxytocin_delta * 0.6 + vasopressin_delta * 0.4,
            "vigilance_tone": vigilance_tone,
            "soothing_tone": soothing_tone,
            "fatigue_tone": melatonin_delta,
            "novelty_signal": novelty_signal,
            "stability_signal": stability_signal,
            "nurturing_tone": nurturing_tone,
            "focus_tone": focus_tone,
            "metabolic_signal": metabolic_signal,
            "appetite_signal": appetite_signal,
            "tension_signal": tension_signal,
            "calm_depth": calm_depth,
            "drive_curiosity": self.drives["curiosity"],
            "drive_social": self.drives["social_bonding"],
            "drive_achievement": self.drives["achievement"],
            "drive_calm": self.drives["calm_restoration"],
            "drive_playful": self.drives["playful_connection"],
            "drive_protection": self.drives["self_preservation"],
            "drive_reproduction": self.drives["reproduction"],
            "drive_nurture": self.drives["nurturing_instinct"],
            "drive_focus": self.drives["focused_attention"],
            "drive_appetite": self.drives["appetite_regulation"],
            "drive_resilience": self.drives["stress_resilience"],
            "drive_restoration": self.drives["restorative_drive"],
            "circadian_alertness": self.circadian_state.get("circadian_alertness", 0.6),
            "sleep_pressure": self.circadian_state.get("sleep_pressure", self.sleep_pressure),
            "homeostatic_drive": self.circadian_state.get("homeostatic_drive", self.homeostatic_drive),
            "melatonin_hint": self.circadian_state.get("melatonin_hint", 0.4),
            "neurochemical_signature": self._compose_neurochemical_signature(),
            "testosterone_delta": testosterone_delta,
            "estrogen_delta": estrogen_delta,
            "progesterone_delta": progesterone_delta,
            "prolactin_delta": prolactin_delta,
            "acetylcholine_delta": acetylcholine_delta,
            "gaba_delta": gaba_delta,
            "glutamate_delta": glutamate_delta,
            "histamine_delta": histamine_delta,
            "thyroxine_delta": thyroxine_delta,
            "ghrelin_delta": ghrelin_delta,
            "leptin_delta": leptin_delta,
        }

    def _apply_hormonal_effects(self, analysis: Dict[str, Any]) -> None:
        intensity = analysis.get("intensity", 0.4)
        effects = analysis.get("hormonal_effects", {})
        for hormone_name, effect in effects.items():
            if hormone_name in self.hormones:
                self.hormones[hormone_name].apply_change(effect * intensity)

    def _advance_circadian_clock(self, analysis: Dict[str, Any]) -> None:
        now = time.time()
        elapsed_hours = max((now - self.last_update_ts) / 3600.0, 0.25)
        self.last_update_ts = now
        self.circadian_phase = (self.circadian_phase + elapsed_hours / 24.0) % 1.0

        features = analysis.get("features", {})
        stress = features.get("stress", 0.0)
        calm = features.get("calm", 0.0)
        novelty = features.get("novelty", 0.0)

        self.sleep_pressure = clamp(
            self.sleep_pressure + elapsed_hours * 0.03 + stress * 0.04 - calm * 0.05 - novelty * 0.03,
            0.2,
            0.98,
        )
        self.homeostatic_drive = clamp(
            0.45 + self.sleep_pressure * 0.25 + stress * 0.2 - calm * 0.15,
            0.25,
            0.9,
        )

    def _apply_cross_coupling(self) -> None:
        """Allow hormones to subtly influence one another."""
        stress_delta = self.hormones["cortisol"].level - self.hormones["cortisol"].baseline
        if stress_delta > 0:
            self.hormones["serotonin"].apply_change(-0.12 * stress_delta)
            self.hormones["dopamine"].apply_change(-0.08 * stress_delta)
            self.hormones["endorphins"].apply_change(-0.05 * stress_delta)

        bonding_delta = self.hormones["oxytocin"].level - self.hormones["oxytocin"].baseline
        if bonding_delta > 0:
            self.hormones["serotonin"].apply_change(0.07 * bonding_delta)
            self.hormones["dopamine"].apply_change(0.05 * bonding_delta)
            self.hormones["vasopressin"].apply_change(0.04 * bonding_delta)

        adrenaline_delta = self.hormones["adrenaline"].level - self.hormones["adrenaline"].baseline
        if adrenaline_delta > 0:
            self.hormones["norepinephrine"].apply_change(0.08 * adrenaline_delta)
            self.hormones["cortisol"].apply_change(0.06 * adrenaline_delta)
            self.hormones["serotonin"].apply_change(-0.05 * adrenaline_delta)

        melatonin_delta = self.hormones["melatonin"].level - self.hormones["melatonin"].baseline
        if melatonin_delta > 0:
            self.hormones["adrenaline"].apply_change(-0.06 * melatonin_delta)
            self.hormones["norepinephrine"].apply_change(-0.05 * melatonin_delta)
            self.hormones["dopamine"].apply_change(-0.04 * melatonin_delta)

        testosterone_delta = self.hormones["testosterone"].level - self.hormones["testosterone"].baseline
        if testosterone_delta > 0:
            self.hormones["dopamine"].apply_change(0.06 * testosterone_delta)
            self.hormones["cortisol"].apply_change(0.04 * testosterone_delta)

        estrogen_delta = self.hormones["estrogen"].level - self.hormones["estrogen"].baseline
        if estrogen_delta > 0:
            self.hormones["oxytocin"].apply_change(0.06 * estrogen_delta)
            self.hormones["serotonin"].apply_change(0.05 * estrogen_delta)

        progesterone_delta = self.hormones["progesterone"].level - self.hormones["progesterone"].baseline
        if progesterone_delta > 0:
            self.hormones["gaba"].apply_change(0.06 * progesterone_delta)
            self.hormones["cortisol"].apply_change(-0.05 * progesterone_delta)

        prolactin_delta = self.hormones["prolactin"].level - self.hormones["prolactin"].baseline
        if prolactin_delta > 0:
            self.hormones["oxytocin"].apply_change(0.05 * prolactin_delta)
            self.hormones["progesterone"].apply_change(0.04 * prolactin_delta)

        gaba_delta = self.hormones["gaba"].level - self.hormones["gaba"].baseline
        if gaba_delta > 0:
            self.hormones["adrenaline"].apply_change(-0.05 * gaba_delta)
            self.hormones["histamine"].apply_change(-0.05 * gaba_delta)

        glutamate_delta = self.hormones["glutamate"].level - self.hormones["glutamate"].baseline
        if glutamate_delta > 0:
            self.hormones["histamine"].apply_change(0.04 * glutamate_delta)
            self.hormones["cortisol"].apply_change(0.03 * glutamate_delta)

        histamine_delta = self.hormones["histamine"].level - self.hormones["histamine"].baseline
        if histamine_delta > 0:
            self.hormones["norepinephrine"].apply_change(0.05 * histamine_delta)
            self.hormones["acetylcholine"].apply_change(0.04 * histamine_delta)

        thyroxine_delta = self.hormones["thyroxine"].level - self.hormones["thyroxine"].baseline
        if thyroxine_delta > 0:
            self.hormones["dopamine"].apply_change(0.04 * thyroxine_delta)
            self.hormones["serotonin"].apply_change(0.03 * thyroxine_delta)

        ghrelin_delta = self.hormones["ghrelin"].level - self.hormones["ghrelin"].baseline
        if ghrelin_delta > 0:
            self.hormones["cortisol"].apply_change(0.05 * ghrelin_delta)
            self.hormones["leptin"].apply_change(-0.06 * ghrelin_delta)

        leptin_delta = self.hormones["leptin"].level - self.hormones["leptin"].baseline
        if leptin_delta > 0:
            self.hormones["ghrelin"].apply_change(-0.05 * leptin_delta)
            self.hormones["serotonin"].apply_change(0.04 * leptin_delta)

    def _harmonize_with_circadian(self) -> None:
        signals = self._derive_circadian_signals()
        alertness = signals["circadian_alertness"]
        sleep_pressure = signals["sleep_pressure"]
        self.hormones["cortisol"].apply_change((0.65 - alertness) * 0.05)
        self.hormones["serotonin"].apply_change((alertness - 0.5) * 0.04 - sleep_pressure * 0.02)
        self.hormones["dopamine"].apply_change((alertness - 0.55) * 0.03 - sleep_pressure * 0.025)
        self.hormones["norepinephrine"].apply_change((alertness - 0.5) * 0.035 - sleep_pressure * 0.018)
        self.hormones["adrenaline"].apply_change((alertness - 0.5) * 0.03 - sleep_pressure * 0.02)
        self.hormones["melatonin"].apply_change((0.4 - alertness) * 0.045 + sleep_pressure * 0.05)
        self.hormones["endorphins"].apply_change((0.55 - sleep_pressure) * 0.02 + (alertness - 0.5) * 0.015)
        self.hormones["oxytocin"].apply_change((signals["melatonin_hint"] - 0.4) * 0.015)
        self.hormones["progesterone"].apply_change((0.5 - alertness) * 0.02 + sleep_pressure * 0.015)
        self.hormones["prolactin"].apply_change((signals["melatonin_hint"] - 0.4) * 0.018)
        self.hormones["gaba"].apply_change((0.45 - alertness) * 0.025 + sleep_pressure * 0.02)
        self.hormones["acetylcholine"].apply_change((alertness - 0.5) * 0.028 - sleep_pressure * 0.015)
        self.hormones["glutamate"].apply_change((alertness - 0.5) * 0.025)
        self.hormones["histamine"].apply_change((alertness - 0.5) * 0.022 - sleep_pressure * 0.012)
        self.hormones["thyroxine"].apply_change((alertness - 0.5) * 0.018 - sleep_pressure * 0.01)
        self.hormones["ghrelin"].apply_change(sleep_pressure * 0.022 - alertness * 0.01)
        self.hormones["leptin"].apply_change((signals["melatonin_hint"] - 0.4) * 0.02 - sleep_pressure * 0.015)
        self.circadian_state = signals

    def _interpret_stimulus(self, stimulus: Any) -> Dict[str, Any]:
        description = stimulus if isinstance(stimulus, str) else str(stimulus)
        features = {
            "positive": 0.0,
            "stress": 0.0,
            "social": 0.0,
            "novelty": 0.0,
            "achievement": 0.0,
            "calm": 0.0,
            "assertive": 0.0,
            "threat": 0.0,
            "comfort": 0.0,
            "playful": 0.0,
            "romance": 0.0,
            "loss": 0.0,
            "caregiving": 0.0,
            "awe": 0.0,
            "focus": 0.0,
            "restoration": 0.0,
            "hunger": 0.0,
            "satiety": 0.0,
            "alert": 0.0,
            "learning": 0.0,
            "activity": 0.0,
            "inspiration": 0.0,
            "fatigue": 0.0,
        }
        intensity = 0.45
        tags = set()

        if isinstance(stimulus, dict):
            description = stimulus.get("description", description)
            intensity = clamp(float(stimulus.get("intensity", intensity)), 0.1, 1.0)
            tags = {str(tag).lower() for tag in stimulus.get("tags", [])}
            for key in features:
                if key in stimulus:
                    features[key] = clamp(float(stimulus[key]), 0.0, 1.0)
            text = f"{description} {' '.join(tags)}".lower()
        else:
            text = str(stimulus or "").lower()

        keyword_map = {
            "positive": ["positive", "reward", "pleasant", "joy", "success", "love", "good", "delight", "uplifting"],
            "stress": ["threat", "danger", "stress", "fear", "alarm", "urgent", "pain", "loss", "crisis"],
            "social": ["social", "together", "team", "friend", "hug", "community", "bond", "family", "belong"],
            "novelty": ["new", "novel", "curious", "explore", "mystery", "unknown", "innovative"],
            "achievement": ["achievement", "goal", "win", "progress", "accomplish", "perform", "victory", "ambition"],
            "calm": ["calm", "rest", "peace", "stable", "safe", "relax", "soothing", "meditative"],
            "assertive": ["challenge", "dominate", "assert", "competition", "drive", "ambition", "power"],
            "threat": ["attack", "danger", "predator", "threat", "risk"],
            "comfort": ["comfort", "care", "support", "nurture", "gentle"],
            "playful": ["play", "fun", "laugh", "game", "humor"],
            "romance": ["romance", "intimate", "kiss", "affection", "date"],
            "loss": ["loss", "grief", "mourning", "lonely", "bereft"],
            "caregiving": ["care", "nurse", "protect", "parent", "guardian"],
            "awe": ["majestic", "awe", "wonder", "beautiful", "sublime"],
            "focus": ["focus", "concentrate", "study", "attention", "attentive"],
            "restoration": ["sleep", "recover", "healing", "recharge", "restoration"],
            "hunger": ["hungry", "starving", "appetite", "food", "craving"],
            "satiety": ["full", "satisfied", "nourished", "content", "replenished"],
            "alert": ["alert", "awake", "watchful", "energetic", "awake"],
            "learning": ["learn", "study", "lesson", "practice", "training", "memory"],
            "activity": ["run", "exercise", "workout", "busy", "active"],
            "inspiration": ["inspired", "vision", "dream", "idea", "imagine"],
            "fatigue": ["tired", "exhausted", "sleepy", "weary", "fatigued"],
        }

        for feature, keywords in keyword_map.items():
            if any(keyword in text for keyword in keywords) or feature in tags:
                features[feature] = max(features[feature], 0.6)

        detected_intensity = max(features.values()) if features else 0.0
        intensity = clamp(max(intensity, detected_intensity), 0.1, 1.0)

        hormonal_effects = {
            "dopamine": 0.24 * features["positive"] + 0.18 * features["novelty"] + 0.16 * features["achievement"] + 0.12 * features["playful"] + 0.1 * features["awe"] + 0.12 * features["inspiration"] + 0.1 * features["activity"] - 0.12 * features["stress"],
            "serotonin": 0.22 * features["calm"] + 0.14 * features["comfort"] + 0.1 * features["positive"] + 0.08 * features["awe"] + 0.08 * features["satiety"] - 0.24 * features["stress"],
            "norepinephrine": 0.28 * features["threat"] + 0.18 * features["novelty"] + 0.16 * features["achievement"] + 0.14 * features["alert"] - 0.12 * features["calm"],
            "adrenaline": 0.26 * features["stress"] + 0.2 * features["threat"] + 0.14 * features["assertive"] + 0.1 * features["activity"],
            "cortisol": 0.32 * features["stress"] + 0.22 * features["loss"] + 0.14 * features["hunger"] - 0.18 * features["calm"],
            "oxytocin": 0.28 * features["social"] + 0.16 * features["comfort"] + 0.14 * features["romance"] + 0.12 * features["caregiving"] - 0.1 * features["stress"],
            "vasopressin": 0.24 * features["caregiving"] + 0.18 * features["threat"] + 0.12 * features["romance"] + 0.1 * features["alert"],
            "endorphins": 0.26 * features["playful"] + 0.2 * features["positive"] + 0.16 * features["comfort"] + 0.12 * features["restoration"] - 0.14 * features["loss"],
            "melatonin": 0.24 * features["calm"] + 0.18 * features["comfort"] + 0.16 * features["loss"] + 0.18 * features["restoration"] - 0.18 * features["stress"],
            "testosterone": 0.2 * features["assertive"] + 0.16 * features["achievement"] + 0.12 * features["stress"] + 0.1 * features["activity"],
            "estrogen": 0.22 * features["social"] + 0.18 * features["romance"] + 0.16 * features["caregiving"] + 0.12 * features["inspiration"] - 0.12 * features["stress"],
            "progesterone": 0.26 * features["comfort"] + 0.2 * features["romance"] + 0.18 * features["restoration"] - 0.14 * features["stress"],
            "prolactin": 0.32 * features["caregiving"] + 0.2 * features["comfort"] + 0.14 * features["loss"],
            "acetylcholine": 0.3 * features["focus"] + 0.22 * features["learning"] + 0.16 * features["novelty"] + 0.12 * features["alert"] - 0.12 * features["fatigue"],
            "gaba": 0.32 * features["calm"] + 0.24 * features["comfort"] + 0.18 * features["restoration"] - 0.2 * features["stress"],
            "glutamate": 0.28 * features["novelty"] + 0.22 * features["learning"] + 0.18 * features["focus"] + 0.1 * features["activity"],
            "histamine": 0.26 * features["alert"] + 0.2 * features["threat"] + 0.16 * features["novelty"],
            "thyroxine": 0.24 * features["activity"] + 0.2 * features["learning"] + 0.16 * features["novelty"] - 0.14 * features["restoration"],
            "ghrelin": 0.34 * features["hunger"] + 0.18 * features["stress"] - 0.2 * features["satiety"],
            "leptin": 0.3 * features["satiety"] + 0.18 * features["comfort"] - 0.22 * features["hunger"],
        }

        summary_features = {k: v for k, v in features.items() if v > 0.05}
        summary = ", ".join(f"{k}:{v:.2f}" for k, v in summary_features.items()) or "neutral"

        return {
            "raw_stimulus": stimulus,
            "description": description,
            "features": features,
            "intensity": intensity,
            "hormonal_effects": hormonal_effects,
            "summary": summary,
        }

    def _recalculate_drives(self) -> None:
        for name, profile in self.drive_profiles.items():
            self.drives[name] = profile.update_from_hormones(self.hormones)

    def _refresh_cached_levels(self) -> None:
        self.hormone_levels = {name: profile.level for name, profile in self.hormones.items()}

    def _record_stimulus(self, analysis: Dict[str, Any]) -> None:
        record = {
            "timestamp": time.time(),
            "stimulus": analysis.get("raw_stimulus"),
            "features": analysis.get("features", {}).copy(),
            "hormones": self.hormone_levels.copy(),
            "drives": self.drives.copy(),
            "circadian": self.circadian_state.copy(),
        }
        self.stimulus_history.append(record)
        self.stimulus_history = self.stimulus_history[-120:]

    def _derive_circadian_signals(self) -> Dict[str, float]:
        circadian_wave = 0.5 + 0.4 * np.sin(2 * np.pi * (self.circadian_phase - 0.2))
        circadian_alertness = clamp(circadian_wave - self.sleep_pressure * 0.35, 0.1, 0.95)
        melatonin_hint = clamp(1.0 - circadian_alertness, 0.05, 0.95)
        return {
            "circadian_alertness": circadian_alertness,
            "sleep_pressure": self.sleep_pressure,
            "homeostatic_drive": self.homeostatic_drive,
            "melatonin_hint": melatonin_hint,
        }

    def _compose_neurochemical_signature(self) -> Dict[str, float]:
        return {name: round(level, 4) for name, level in self.hormone_levels.items()}


class DumbModel:
    """
    DUMB MODEL: Control & Tool Use
    "for control and tool use"
    """

    def __init__(self):
        self.tools_available = {
            "calculator": {"status": "ready", "capability": "mathematical_operations"},
            "web_search": {"status": "ready", "capability": "information_retrieval"},
            "file_manager": {"status": "ready", "capability": "file_operations"},
            "image_processor": {"status": "ready", "capability": "image_manipulation"},
            "text_generator": {"status": "ready", "capability": "text_creation"}
        }
        self.current_task = None
        self.task_history = []

    def receive_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Receive and execute commands from Sector Two."""
        if "action" in command and "tool" in command:
            tool = command["tool"]
            action = command["action"]

            if tool in self.tools_available:
                self.current_task = f"{action} using {tool}"
                self.task_history.append({
                    "task": self.current_task,
                    "timestamp": time.time(),
                    "status": "executing"
                })

                # Simulate tool execution
                result = self._execute_tool(tool, action, command.get("parameters", {}))

                self.task_history[-1]["status"] = "completed"
                self.task_history[-1]["result"] = result

                print(f"Dumb Model executed: {self.current_task}")
                return {"status": "success", "result": result}
            else:
                return {"status": "failed", "error": f"Tool {tool} not available"}

        return {"status": "no_command"}

    def _execute_tool(self, tool: str, action: str, parameters: Dict) -> str:
        """Execute specific tool operations."""
        if tool == "calculator":
            return f"Calculated: {random.randint(1, 100)}"
        elif tool == "web_search":
            return f"Found {random.randint(5, 50)} results for '{action}'"
        elif tool == "file_manager":
            return f"File operation '{action}' completed successfully"
        elif tool == "image_processor":
            return f"Image processed: {action} applied"
        elif tool == "text_generator":
            return f"Generated text: '{action} related content'"
        else:
            return f"Tool {tool} executed action: {action}"


class VISTA:
    """
    VISTA - Visual Interpretation and Scene-Transformation Architecture
    Takes image data from GPU, mouse input, screen text/buttons (like a screenshot).
    Sends data to the GPU model.
    """

    def __init__(self):
        self.visual_buffer = None
        self.mouse_input = None
        self.screen_text = None
        self.interpreted_scene = None
        self.ocr_results = []
        self.object_detections = []

    def capture_gpu_data(self, gpu_image: np.ndarray, mouse_pos: Dict[str, int], screen_text: str):
        """Capture visual data from GPU, mouse, and screen."""
        self.visual_buffer = gpu_image
        self.mouse_input = mouse_pos
        self.screen_text = screen_text

        print(f"VISTA captured GPU data:")
        print(f"  Image shape: {gpu_image.shape}")
        print(f"  Mouse position: {mouse_pos}")
        print(f"  Screen text: '{screen_text[:50]}...'")

    def interpret_scene(self, drive_signals: Dict[str, float]) -> Dict[str, Any]:
        """Interpret visual scene based on drives and visual data."""
        if self.visual_buffer is None:
            return {"status": "no_visual_data"}

        # Simulate object detection based on image characteristics
        image_mean, image_std = compute_image_stats(self.visual_buffer)

        # Simple object detection simulation
        if image_mean > 0.7:
            detected_object = "bright_object"
        elif image_mean < 0.3:
            detected_object = "dark_object"
        else:
            detected_object = "medium_brightness_object"

        # OCR simulation
        if self.screen_text:
            self.ocr_results.append({
                "text": self.screen_text,
                "confidence": 0.9,
                "position": "center"
            })

        # Drive-influenced interpretation
        curiosity_level = drive_signals.get("curiosity", 0.5)
        self_preservation = drive_signals.get("self_preservation", 0.5)

        relevance_score = curiosity_level * image_std + self_preservation * (1 - image_std)

        interpreted = {
            "main_object": detected_object,
            "object_confidence": min(1.0, relevance_score),
            "mouse_focus": self.mouse_input,
            "detected_text": self.screen_text,
            "relevance_score": relevance_score,
            "scene_complexity": image_std,
            "attention_level": curiosity_level,
            "threat_assessment": 1 - self_preservation
        }

        self.interpreted_scene = interpreted
        self.object_detections.append(interpreted)

        print(f"VISTA interpreted scene: {interpreted}")
        return interpreted

    def send_to_gpu_model(self) -> Dict[str, Any]:
        """Send interpreted scene to GPU model for rendering."""
        if self.interpreted_scene:
            gpu_command = {
                "render_command": "update_scene",
                "scene_data": self.interpreted_scene,
                "timestamp": time.time()
            }
            print(f"VISTA sending to GPU model: {gpu_command}")
            return gpu_command
        return {"status": "no_scene_to_send"}


class ModelThinking:
    """
    Model Thinking: Large Reasoning Model (LRM) equivalent
    "the same way a lrm works"
    """

    def __init__(self):
        self.reasoning_context = []
        self.current_decision = None
        self.decision_history = []
        self.reasoning_chain = []

    def process_vista_output(self, vista_output: Dict[str, Any], thought_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process VISTA output with internal thoughts (LRM-like reasoning)."""
        # Build reasoning context
        context_entry = {
            "vista_data": vista_output,
            "thought_data": thought_data,
            "timestamp": time.time()
        }
        self.reasoning_context.append(context_entry)

        # LRM-like reasoning process
        reasoning_steps = []

        # Step 1: Analyze visual input
        if "main_object" in vista_output:
            reasoning_steps.append(f"Detected: {vista_output['main_object']}")

        # Step 2: Consider internal thoughts
        if "current_thought" in thought_data:
            reasoning_steps.append(f"Internal thought: {thought_data['current_thought']}")

        # Step 3: Make decision based on context
        if vista_output.get("relevance_score", 0) > 0.7:
            self.current_decision = "High relevance detected, focusing attention"
        elif "creative" in str(thought_data.get("current_thought", "")):
            self.current_decision = "Creative mode activated, generating ideas"
        else:
            self.current_decision = "Monitoring environment, maintaining awareness"

        reasoning_steps.append(f"Decision: {self.current_decision}")
        self.reasoning_chain = reasoning_steps

        decision_entry = {
            "decision": self.current_decision,
            "reasoning_steps": reasoning_steps,
            "confidence": vista_output.get("relevance_score", 0.5),
            "timestamp": time.time()
        }
        self.decision_history.append(decision_entry)

        print(f"Model Thinking processed data. Decision: {self.current_decision}")
        return decision_entry

    def generate_agent_command(self) -> Dict[str, Any]:
        """Generate commands for other controls and agent use."""
        if "focusing attention" in str(self.current_decision):
            return {
                "action": "focus",
                "target": "high_relevance_object",
                "confidence": 0.9,
                "priority": "high"
            }
        elif "creative mode" in str(self.current_decision):
            return {
                "action": "generate",
                "type": "creative_content",
                "style": "innovative",
                "priority": "medium"
            }
        else:
            return {
                "action": "monitor",
                "area": "environment",
                "frequency": "continuous",
                "priority": "low"
            }


class OtherControls:
    """
    Other Controls and Agent Use
    "so it can do a little more like vr integration"
    """

    def __init__(self):
        self.available_controls = {
            "vr_integration": {"status": "ready", "capability": "virtual_reality"},
            "agent_coordination": {"status": "ready", "capability": "multi_agent"},
            "external_api": {"status": "ready", "capability": "api_communication"},
            "hardware_control": {"status": "ready", "capability": "device_control"}
        }
        self.active_agents = []

    def process_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Process commands from Model Thinking."""
        action = command.get("action", "monitor")

        if action == "focus":
            # Activate VR integration for focused attention
            return self._activate_vr_focus(command)
        elif action == "generate":
            # Coordinate with creative agents
            return self._coordinate_creative_agents(command)
        else:
            # Default monitoring mode
            return self._monitor_environment(command)

    def _activate_vr_focus(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Activate VR integration for focused attention."""
        print(f"VR Integration activated for: {command.get('target', 'unknown')}")
        return {
            "status": "vr_active",
            "focus_target": command.get("target"),
            "immersion_level": "high"
        }

    def _coordinate_creative_agents(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate with creative agents."""
        print(f"Creative agents activated for: {command.get('type', 'content')}")
        return {
            "status": "creative_agents_active",
            "generation_type": command.get("type"),
            "agent_count": 3
        }

    def _monitor_environment(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor environment with low priority."""
        return {
            "status": "monitoring",
            "area": command.get("area", "environment"),
            "alert_level": "normal"
        }


class HIMModel:
    """
    The complete H.I.M. Model integrating all components.
    """

    def __init__(self):
        self.sector_one = SectorOne()
        self.sector_two = SectorTwo()
        self.sector_three = SectorThree()
        self.sector_four = SectorFour()
        self.dumb_model = DumbModel()
        self.vista = VISTA()
        self.model_thinking = ModelThinking()
        self.other_controls = OtherControls()

        self.persona_profile = _default_persona_profile()
        self.consciousness_kernel = ConsciousnessKernel(self.persona_profile)

        self.cycle_count = 0
        self.system_state = "initializing"
        self.ensure_schema_integrity()

        self.acceleration = get_acceleration_state()
        if self.acceleration.using_gpu:
            print(f"Acceleration backend active: {self.acceleration.backend} ({self.acceleration.device})")

    def ensure_schema_integrity(self) -> None:
        """Ensure all sectors expose the latest emotion and hormone catalogues."""

        if hasattr(self.sector_one, 'ensure_catalog'):
            self.sector_one.ensure_catalog()
        if hasattr(self.sector_four, 'ensure_catalog'):
            self.sector_four.ensure_catalog()

    def run_cycle(self, gpu_image: np.ndarray, mouse_pos: Dict[str, int], screen_text: str):
        """Run a complete H.I.M. model cycle."""
        self.cycle_count += 1
        print(f"\n{'='*60}")
        print(f"H.I.M. Model Cycle #{self.cycle_count}")
        print(f"{'='*60}")

        # 1. VISTA captures visual data
        self.vista.capture_gpu_data(gpu_image, mouse_pos, screen_text)

        # 2. Sector Four updates based on stimulus
        stimulus = f"Visual stimulus: {screen_text[:20]}..."
        self.sector_four.update_levels(stimulus)
        drive_signals = self.sector_four.get_drive_signals()
        hormone_modulation = self.sector_four.get_emotional_modulation()

        # 3. VISTA interprets scene with drive influence
        interpreted_scene = self.vista.interpret_scene(drive_signals)

        # 4. Sector One processes emotional input
        emotion_input = {
            "emotion_boost": 0.05 if interpreted_scene.get("relevance_score", 0) > 0.5 else -0.01,
            "creative_stimulus": f"Scene: {interpreted_scene.get('main_object', 'unknown')}",
            "hormone_modulation": hormone_modulation,
            "drive_signals": drive_signals,
            "circadian_alertness": hormone_modulation.get("circadian_alertness"),
            "sleep_pressure": hormone_modulation.get("sleep_pressure"),
            "homeostatic_drive": hormone_modulation.get("homeostatic_drive"),
        }
        sector_one_output = self.sector_one.process_input(emotion_input)
        self.sector_four.apply_emotional_feedback(sector_one_output.get("emotions", {}))

        # 5. Sector Two processes thoughts and reasoning
        sector_two_output = self.sector_two.process_input(sector_one_output)

        # 6. Model Thinking processes everything (LRM-like)
        model_thinking_output = self.model_thinking.process_vista_output(
            interpreted_scene, sector_two_output
        )

        # 7. Generate agent commands
        agent_command = self.model_thinking.generate_agent_command()

        # 8. Other Controls process commands
        control_result = self.other_controls.process_command(agent_command)

        # 9. Dumb Model receives commands
        dumb_model_command = {
            "action": agent_command["action"],
            "tool": "image_processor" if "focus" in agent_command["action"] else "text_generator",
            "parameters": {"priority": agent_command.get("priority", "medium")}
        }
        dumb_model_result = self.dumb_model.receive_command(dumb_model_command)

        # 10. Store memories in Sector Three (both regular and spatial)
        memory_data = {
            "scene": interpreted_scene,
            "decision": model_thinking_output["decision"],
            "emotions": sector_one_output["emotions"],
            "hormones": self.sector_four.hormone_levels,
            "cycle": self.cycle_count,
            "affective_summary": sector_one_output.get("affective_summary"),
            "needs_state": sector_one_output.get("needs_state"),
            "circadian_state": hormone_modulation,
        }
        importance = interpreted_scene.get("relevance_score", 0.5)
        self.sector_three.store_memory(memory_data, importance)

        # Store vectorized thoughts in spatial memory system (with combination)
        if self.sector_two.vectorized_images:
            # Combine all vectors from this cycle into one consolidated vector
            combined_vector = self.sector_two.combine_cycle_vectors(self.cycle_count)
            if combined_vector:
                spatial_pos = self.sector_three.store_spatial_memory(combined_vector, importance)
                print(f"Spatial Memory: Combined vector stored at position {spatial_pos}")
                print(f"Spatial Memory: Combined {combined_vector.metadata.get('original_count', 1)} individual vectors")

        # 11. Generate dream fragment if conditions are met
        dream_fragment = None
        if self.cycle_count % 5 == 0:  # Every 5 cycles
            dream_fragment = self.sector_three.generate_dream_fragment()
            if dream_fragment:
                print(f"Dream generated: {dream_fragment}")

        # 12. CPU Processing: Send vectorized data to CPU for processing
        cpu_processed_data = self.process_vectorized_data_on_cpu()

        # 13. VISTA sends to GPU model
        gpu_result = self.vista.send_to_gpu_model()

        # 14. GPU Storage: Send processed data back to GPU for permanent storage
        permanent_storage_result = self.store_data_permanently_on_gpu(cpu_processed_data)

        # Update system state
        self.system_state = "active" if self.cycle_count > 0 else "initializing"

        # 15. Update persona consciousness narrative
        try:
            self.consciousness_kernel.record_cycle(
                cycle=self.cycle_count,
                decision=model_thinking_output["decision"],
                emotions=sector_one_output.get("emotions", {}),
                drives=self.sector_four.drives.copy(),
                hormones=self.sector_four.hormone_levels.copy(),
                screen_text=screen_text,
                dream_fragment=dream_fragment,
            )
        except Exception as persona_error:  # pragma: no cover - defensive logging
            print(f"Persona kernel warning: {persona_error}")

        print(f"\nCycle #{self.cycle_count} completed. System state: {self.system_state}")
        print(f"{'='*60}")

        return {
            "cycle": self.cycle_count,
            "system_state": self.system_state,
            "vista_output": interpreted_scene,
            "model_decision": model_thinking_output["decision"],
            "control_result": control_result,
            "dumb_model_result": dumb_model_result,
            "cpu_processing": cpu_processed_data,
            "gpu_storage": permanent_storage_result,
            "input_text": screen_text,
            "agent_command": agent_command,
            "sector_one_emotions": sector_one_output.get("emotions", {}).copy(),
            "sector_four_hormones": self.sector_four.hormone_levels.copy(),
            "sector_four_drives": self.sector_four.drives.copy(),
            "sector_two_output": sector_two_output,
            "dream_fragment": dream_fragment,
            "persona_state": self.consciousness_kernel.get_state(),
        }

    def extract_feature_vector(
        self,
        screen_text: str,
        gpu_image: Optional[np.ndarray] = None,
        mouse_pos: Optional[Dict[str, int]] = None,
        *,
        cleanup: bool = True,
        suppress_output: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate a machine-learning friendly feature vector from a text prompt.

        This method performs a lightweight inference pass through the H.I.M.
        architecture without storing memories or incrementing the primary cycle
        counter. The resulting feature vector blends emotional, reasoning,
        hormonal, and drive states so downstream machine learning models can be
        trained without relying on transformer embeddings.

        The exported affective space now spans a deeply human palette that
        includes uplifting states (joy, serenity, calm, awe, curiosity,
        anticipation, trust, love, empathy, compassion, gratitude, hope,
        confidence, motivation, resilience, inspiration, optimism, pride,
        focus, nurturing, contentment, relief) alongside protective and
        challenging emotions (vigilance, fear, anxiety, caution, anger,
        frustration, sadness, loneliness, melancholy, disgust, tension,
        fatigue, boredom, envy, jealousy, shame, guilt, nostalgia, pessimism).
        These feed into twenty mood-linked hormones (dopamine, serotonin,
        norepinephrine, adrenaline, cortisol, oxytocin, vasopressin,
        endorphins, melatonin, testosterone, estrogen, progesterone, prolactin,
        acetylcholine, GABA, glutamate, histamine, thyroxine, ghrelin, leptin)
        and twelve hormonally-derived drives. This richer representation makes
        it easier to map H.I.M. state into recognisable human mood trajectories.

        Args:
            screen_text: The textual stimulus presented to the system.
            gpu_image: Optional visual stimulus. When omitted, a neutral frame is used.
            mouse_pos: Optional mouse coordinates associated with the scene.
            cleanup: When ``True`` temporary Sector Two vectors generated during
                the call are discarded to avoid polluting long-running sessions.
            suppress_output: When ``True`` suppress verbose logging from the
                underlying subsystems to keep training loops quiet.

        Returns:
            A tuple ``(features, metadata)`` where ``features`` is a 1-D numpy
            array suitable for machine-learning pipelines and ``metadata``
            contains human-readable diagnostics for debugging or analysis.
        """

        if gpu_image is None:
            gpu_image = np.zeros((64, 64, 3), dtype=np.float32)
        else:
            gpu_image = np.asarray(gpu_image, dtype=np.float32)
            if gpu_image.ndim == 2:
                gpu_image = np.expand_dims(gpu_image, axis=-1)
            if gpu_image.ndim == 3 and gpu_image.shape[2] == 1:
                gpu_image = np.repeat(gpu_image, 3, axis=2)

        if mouse_pos is None:
            mouse_pos = {
                "x": int(gpu_image.shape[1] // 2),
                "y": int(gpu_image.shape[0] // 2),
            }

        initial_vector_count = len(self.sector_two.vectorized_images)
        output_buffer = io.StringIO()
        log_context = redirect_stdout(output_buffer) if suppress_output else nullcontext()

        with log_context:
            self.vista.capture_gpu_data(gpu_image, mouse_pos, screen_text)

            stimulus = f"Visual stimulus: {screen_text[:20]}..."
            self.sector_four.update_levels(stimulus)
            drive_signals = self.sector_four.get_drive_signals()
            hormone_modulation = self.sector_four.get_emotional_modulation()

            interpreted_scene = self.vista.interpret_scene(drive_signals)

            emotion_input = {
                "emotion_boost": 0.05 if interpreted_scene.get("relevance_score", 0) > 0.5 else -0.01,
                "creative_stimulus": f"Scene: {interpreted_scene.get('main_object', 'unknown')}",
                "hormone_modulation": hormone_modulation,
                "drive_signals": drive_signals,
                "circadian_alertness": hormone_modulation.get("circadian_alertness"),
                "sleep_pressure": hormone_modulation.get("sleep_pressure"),
                "homeostatic_drive": hormone_modulation.get("homeostatic_drive"),
            }

            sector_one_output = self.sector_one.process_input(emotion_input)
            self.sector_four.apply_emotional_feedback(sector_one_output.get("emotions", {}))
            sector_two_output = self.sector_two.process_input(sector_one_output)

        if cleanup and len(self.sector_two.vectorized_images) > initial_vector_count:
            del self.sector_two.vectorized_images[initial_vector_count:]

        def _ordered_values(source: Dict[str, float]) -> Tuple[List[str], List[float]]:
            keys = sorted(source.keys())
            return keys, [float(source[key]) for key in keys]

        emotion_keys, emotion_values = _ordered_values(self.sector_one.emotions)
        reasoning_keys, reasoning_values = _ordered_values(self.sector_two.reasoning_state)
        hormone_keys, hormone_values = _ordered_values(self.sector_four.hormone_levels)
        drive_keys, drive_values = _ordered_values(self.sector_four.drives)

        feature_vector = np.array(
            emotion_values + reasoning_values + hormone_values + drive_values,
            dtype=np.float32,
        )

        metadata = {
            "emotions": {key: self.sector_one.emotions[key] for key in emotion_keys},
            "reasoning": {key: self.sector_two.reasoning_state[key] for key in reasoning_keys},
            "hormones": {key: self.sector_four.hormone_levels[key] for key in hormone_keys},
            "drives": {key: self.sector_four.drives[key] for key in drive_keys},
            "drive_signals": drive_signals,
            "hormone_modulation": hormone_modulation,
            "interpreted_scene": interpreted_scene,
            "sector_two_output": sector_two_output,
            "suppressed_logs": output_buffer.getvalue() if suppress_output else "",
        }

        return feature_vector, metadata

    def recall_memories_for_chat(self, user_message: str) -> Dict[str, Any]:
        """Recall relevant memories based on chat message content."""
        try:
            # Search for relevant memories
            search_results = self.sector_three.search_memories_by_content(user_message)

            # Get spatial context if we have recent memories
            spatial_context = []
            if self.sector_three.spatial_memory_map:
                # Get the most recent memory position
                recent_positions = list(self.sector_three.spatial_memory_map.keys())
                if recent_positions:
                    # Get context around the most recent memory
                    latest_pos = recent_positions[-1]
                    spatial_context = self.sector_three.get_memory_context(latest_pos, radius=1)

            # Format memory recall results
            recalled_memories = []
            for result in search_results[:5]:  # Limit to top 5 results
                memory = result['memory']
                recalled_memories.append({
                    'position': result['position'],
                    'thought': memory['metadata'].get('thought', 'Unknown'),
                    'timestamp': memory['timestamp'],
                    'relevance': result['relevance'],
                    'memory_id': memory['memory_id']
                })

            # Get spatial statistics
            spatial_stats = self.sector_three.get_spatial_statistics()

            recall_result = {
                'found_memories': len(recalled_memories),
                'memories': recalled_memories,
                'spatial_context': spatial_context[:3],  # Limit context
                'spatial_stats': spatial_stats,
                'search_term': user_message
            }

            print(f"Memory Recall: Found {len(recalled_memories)} relevant memories for '{user_message}'")
            return recall_result

        except Exception as e:
            print(f"Memory Recall Error: {e}")
            return {
                'found_memories': 0,
                'memories': [],
                'spatial_context': [],
                'spatial_stats': {},
                'search_term': user_message,
                'error': str(e)
            }

    def save_all_memories_to_text(self, filename: str = None):
        """Save all vector memories to a text file."""
        return self.sector_three.save_vector_memories_to_text(filename)

    def load_memories_from_text(self, filename: str):
        """Load vector memories from a text file."""
        return self.sector_three.load_vector_memories_from_text(filename)

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            "cycle_count": self.cycle_count,
            "system_state": self.system_state,
            "sector_one_emotions": self.sector_one.emotions,
            "sector_two_reasoning": self.sector_two.reasoning_state,
            "sector_four_hormones": self.sector_four.hormone_levels,
            "sector_four_drives": self.sector_four.drives,
            "memory_count": len(self.sector_three.memories),
            "dream_fragments": len(self.sector_three.dream_fragments),
            "active_tools": len([t for t in self.dumb_model.tools_available.values() if t["status"] == "ready"]),
            "persona_state": self.consciousness_kernel.get_state(),
        }

    # ------------------------------------------------------------------
    # Persona management helpers
    # ------------------------------------------------------------------
    def get_persona_snapshot(self) -> Dict[str, Any]:
        """Return the latest persona state snapshot."""

        return self.consciousness_kernel.get_state()

    def update_persona_profile(self, **updates: Any) -> Dict[str, Any]:
        """Update persona attributes and return the resulting profile."""

        profile = self.consciousness_kernel.update_profile(**updates)
        self.persona_profile = profile
        return profile.to_dict()

    def replace_persona_profile(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Replace the persona profile using supplied dictionary data."""

        merged = _default_persona_profile().to_dict()
        merged.update(profile_data)
        profile = PersonaProfile(
            name=merged["name"],
            identity_statement=merged["identity_statement"],
            biography=merged["biography"],
            upload_story=merged["upload_story"],
            core_values=merged.get("core_values", []),
            lifelong_goals=merged.get("lifelong_goals", []),
            skills=merged.get("skills", []),
            preferences=merged.get("preferences", {}),
            conversation_style=merged.get("conversation_style", "Empathetic and exploratory"),
            voice_description=merged.get("voice_description", "Warm, reflective, and softly expressive"),
        )
        self.consciousness_kernel.replace_profile(profile)
        self.persona_profile = profile
        return profile.to_dict()

    def save_persona_profile(self, filepath: str) -> None:
        """Persist the persona profile to disk as JSON."""

        with open(filepath, "w", encoding="utf-8") as handle:
            json.dump(self.persona_profile.to_dict(), handle, indent=2)

    def load_persona_profile(self, filepath: str) -> Dict[str, Any]:
        """Load persona profile from disk and activate it."""

        with open(filepath, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return self.replace_persona_profile(data)

    def ingest_persona_memory(self, summary: str, emotion: str, intensity: float) -> Dict[str, Any]:
        """Record a manual autobiographical memory event."""

        self.consciousness_kernel.ingest_event(summary, emotion, intensity)
        return self.get_persona_snapshot()

    def ingest_conversation_line(self, speaker: str, utterance: str) -> Dict[str, Any]:
        """Track conversational utterances for persona recall."""

        self.consciousness_kernel.ingest_conversation(speaker, utterance)
        return self.get_persona_snapshot()

    def process_vectorized_data_on_cpu(self):
        """Process vectorized data on CPU before GPU reset."""
        try:
            # Get all vectorized images from Sector Two
            vectorized_images = self.sector_two.vectorized_images

            if not vectorized_images:
                print("CPU Processing: No vectorized data to process")
                return None

            # Process the most recent vectorized image (which should be the combined vector)
            latest_vector = vectorized_images[-1]

            # CPU Processing: Convert to text vector format
            text_vector = latest_vector.to_text_vector()

            cpu_processed = {
                'original_data': latest_vector.data,
                'text_vector': text_vector,
                'metadata': latest_vector.metadata,
                'timestamp': latest_vector.timestamp,
                'cycle_number': self.cycle_count,
                'zipline_pattern': self.create_zipline_pattern_cpu(latest_vector.data),
                'processed_timestamp': time.time(),
                'memory_id': latest_vector.memory_id,
                'spatial_position': latest_vector.spatial_position
            }

            print(f"CPU Processing: Vectorized data processed for cycle {self.cycle_count}")
            print(f"CPU Processing: Data shape: {latest_vector.data.shape}")
            print(f"CPU Processing: Thought: {latest_vector.metadata.get('thought', 'Unknown')[:50]}...")

            return cpu_processed

        except Exception as e:
            print(f"CPU Processing Error: {e}")
            return None

    def create_zipline_pattern_cpu(self, vector_data):
        """Create zipline pattern on CPU from vectorized data."""
        try:
            return create_zipline_pattern(vector_data)
        except Exception as error:
            print(f"CPU Zipline Creation Error: {error}")
            fallback = np.random.rand(100, 200).astype(np.float32) * 0.5 + 0.25
            return fallback

    def store_data_permanently_on_gpu(self, cpu_processed_data):
        """Store processed data permanently on GPU/hard disk."""
        try:
            if not cpu_processed_data:
                print("GPU Storage: No data to store")
                return {"status": "no_data"}

            # Create permanent storage directory
            storage_dir = "him_permanent_storage"
            if not os.path.exists(storage_dir):
                os.makedirs(storage_dir)

            # Create text vector storage subdirectory
            text_vector_dir = os.path.join(storage_dir, "text_vectors")
            if not os.path.exists(text_vector_dir):
                os.makedirs(text_vector_dir)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cycle_num = cpu_processed_data['cycle_number']

            # Save text vector representation
            text_vector = cpu_processed_data['text_vector']
            text_vector_filename = f"vector_memory_cycle_{cycle_num:06d}_{timestamp}.txt"
            text_vector_filepath = os.path.join(text_vector_dir, text_vector_filename)

            with open(text_vector_filepath, 'w', encoding='utf-8') as f:
                f.write("H.I.M. VECTOR MEMORY\n")
                f.write("=" * 40 + "\n")
                f.write(f"Cycle: {cycle_num}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Memory ID: {cpu_processed_data.get('memory_id', 'unknown')}\n")
                f.write(f"Spatial Position: {cpu_processed_data.get('spatial_position', 'None')}\n")
                f.write("=" * 40 + "\n\n")
                f.write(text_vector)
                f.write("\n" + "=" * 40 + "\n")
                f.write("END OF VECTOR MEMORY\n")

            # Save metadata as JSON
            metadata = {
                'cycle_number': cycle_num,
                'timestamp': timestamp,
                'original_shape': cpu_processed_data['original_data'].shape,
                'thought': cpu_processed_data['metadata'].get('thought', 'Unknown'),
                'type': cpu_processed_data['metadata'].get('type', 'Unknown'),
                'processed_timestamp': cpu_processed_data['processed_timestamp'],
                'text_vector_filename': text_vector_filename,
                'storage_location': text_vector_filepath,
                'memory_id': cpu_processed_data.get('memory_id', 'unknown'),
                'spatial_position': cpu_processed_data.get('spatial_position', None)
            }

            metadata_filename = f"vector_metadata_cycle_{cycle_num:06d}_{timestamp}.json"
            metadata_filepath = os.path.join(text_vector_dir, metadata_filename)

            with open(metadata_filepath, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

            print(f"GPU Storage: Data permanently stored for cycle {cycle_num}")
            print(f"GPU Storage: Text vector: {text_vector_filename}")
            print(f"GPU Storage: Metadata: {metadata_filename}")
            print(f"GPU Storage: Memory ID: {cpu_processed_data.get('memory_id', 'unknown')}")
            print(f"GPU Storage: Spatial Position: {cpu_processed_data.get('spatial_position', 'None')}")

            return {
                "status": "success",
                "cycle_number": cycle_num,
                "text_vector_file": text_vector_filepath,
                "metadata_file": metadata_filepath,
                "storage_timestamp": timestamp,
                "memory_id": cpu_processed_data.get('memory_id', 'unknown'),
                "spatial_position": cpu_processed_data.get('spatial_position', None)
            }

        except Exception as e:
            print(f"GPU Storage Error: {e}")
            return {"status": "error", "error": str(e)}


# Example usage and testing
if __name__ == "__main__":
    print("Initializing H.I.M. Model...")
    him_model = HIMModel()

    # Test cycle 1: Normal scene
    print("\n" + "="*80)
    print("TEST CYCLE 1: Normal Scene")
    print("="*80)

    sample_image_1 = np.random.rand(480, 640, 3) * 0.5  # Medium brightness
    mouse_pos_1 = {"x": 320, "y": 240}
    screen_text_1 = "Welcome to the AI interface. Please select an option from the menu."

    result_1 = him_model.run_cycle(sample_image_1, mouse_pos_1, screen_text_1)

    # Test cycle 2: High relevance scene (bright image)
    print("\n" + "="*80)
    print("TEST CYCLE 2: High Relevance Scene")
    print("="*80)

    sample_image_2 = np.ones((480, 640, 3)) * 0.9  # Very bright
    mouse_pos_2 = {"x": 100, "y": 150}
    screen_text_2 = "ALERT: Important object detected! Banana bunch identified. Action required."

    result_2 = him_model.run_cycle(sample_image_2, mouse_pos_2, screen_text_2)

    # Test cycle 3: Creative stimulus
    print("\n" + "="*80)
    print("TEST CYCLE 3: Creative Stimulus")
    print("="*80)

    sample_image_3 = np.random.rand(480, 640, 3) * 0.3  # Dark image
    mouse_pos_3 = {"x": 500, "y": 300}
    screen_text_3 = "Creative mode activated. Generate artistic content based on current scene."

    result_3 = him_model.run_cycle(sample_image_3, mouse_pos_3, screen_text_3)

    # Display final system status
    print("\n" + "="*80)
    print("FINAL SYSTEM STATUS")
    print("="*80)

    status = him_model.get_system_status()
    for key, value in status.items():
        print(f"{key}: {value}")

    print("\nH.I.M. Model demonstration completed!")

