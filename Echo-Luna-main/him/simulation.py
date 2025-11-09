from __future__ import annotations

"""Simulated human feedback model backed by Echo storage."""

import random
from dataclasses import dataclass
from time import time
from typing import Any, Dict

from .storage import HierarchicalImageMemory


@dataclass
class Experience:
    iteration: int
    goal: str
    action: str
    observation: str
    response: str
    extras: Dict[str, Any]
    tile_id: str
    created_at: float

    def as_payload(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "goal": self.goal,
            "action": self.action,
            "observation": self.observation,
            "response": self.response,
            "extras": self.extras,
            "tile_id": self.tile_id,
            "created_at": self.created_at,
        }


class SimulatedHumanModel:
    """Generates lightweight experiences for CognitiveEngine++.

    The model is intentionally simple: given a goal and a proposed action, it
    returns a deterministic-but-noisy observation/response pair and persists the
    exchange into the shared HierarchicalImageMemory stream ``human_experience``.
    """

    def __init__(
        self,
        *,
        store: HierarchicalImageMemory,
        snapshot_id: str,
        stream: str = "human_experience",
    ) -> None:
        self.store = store
        self.snapshot_id = snapshot_id
        self.stream = stream
        self._rng = random.Random(snapshot_id)

    def _fabricate_observation(self, goal: str, action: str, iteration: int) -> str:
        mood = self._rng.choice(["calm", "curious", "reflective", "encouraging", "skeptical"])
        return f"Observer notes in a {mood} tone: considering '{goal}' after action '{action}' at iteration {iteration}."

    def _fabricate_response(self, goal: str, action: str, iteration: int) -> str:
        verbs = ["amplify", "refine", "consolidate", "synthesise", "stabilise"]
        suggestion = self._rng.choice(verbs)
        return f"Suggest to {suggestion} the plan around '{goal}' given the explored action '{action}'."

    def interact(self, *, iteration: int, goal: str, action: str) -> Experience:
        created_at = time()
        observation = self._fabricate_observation(goal, action, iteration)
        response = self._fabricate_response(goal, action, iteration)
        extras: Dict[str, Any] = {
            "these_proxies_note": "These are proxies of sentience-like behavior, not evidence of sentience.",
            "iteration": iteration,
        }
        payload = {
            "iteration": iteration,
            "goal": goal,
            "action": action,
            "observation": observation,
            "response": response,
            "extras": extras,
            "ts": created_at,
        }
        tile_meta = self.store.write_tile(
            stream=self.stream,
            snapshot_id=self.snapshot_id,
            level=0,
            x=iteration,
            y=0,
            payload=payload,
            dtype="json",
        )
        return Experience(
            iteration=iteration,
            goal=goal,
            action=action,
            observation=observation,
            response=response,
            extras=extras,
            tile_id=tile_meta.tile_id,
            created_at=created_at,
        )

    def list_experiences(self) -> Dict[str, Any]:
        tiles = self.store.list_tiles(self.stream, self.snapshot_id, level=0)
        return {"items": tiles}

