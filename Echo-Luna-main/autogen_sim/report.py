from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Dict


PROXY_FIELDS = (
    "memory_continuity",
    "self_model_consistency",
    "theory_of_mind",
    "metacognitive_calibration",
    "affect_stability",
    "integration_proxy",
)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class SentienceProxyReport:
    """Tracks proxies of sentience-like behavior for the simulation."""

    memory_continuity: float
    self_model_consistency: float
    theory_of_mind: float
    metacognitive_calibration: float
    affect_stability: float
    integration_proxy: float

    @classmethod
    def initial(cls) -> "SentienceProxyReport":
        return cls(*(0.0 for _ in PROXY_FIELDS))

    def update(self, deltas: Dict[str, float]) -> "SentienceProxyReport":
        payload = asdict(self)
        for key, delta in deltas.items():
            if key in payload:
                payload[key] = _clamp(payload[key] + delta)
        return SentienceProxyReport(**payload)

    def mean_proxy(self) -> float:
        values = [getattr(self, field) for field in PROXY_FIELDS]
        return sum(values) / len(values)

    def as_json(self) -> str:
        data = asdict(self)
        data["notes"] = (
            "Values are proxies of sentience-like behavior; they are not proof of sentience."
        )
        return json.dumps(data)
