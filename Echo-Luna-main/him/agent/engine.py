from __future__ import annotations

"""Primary agent engine orchestrating CognitiveEngine++ sessions."""

from dataclasses import dataclass, field
from time import time
from typing import Dict, Optional

from echolab.state import PROXY_NAMES

from ..storage import HierarchicalImageMemory
from ..simulation import SimulatedHumanModel
from . import bench, metrics, persist, policy, stoplink


@dataclass
class SessionState:
    snapshot_id: str
    model: SimulatedHumanModel
    trace_writer: persist.AgentTraceWriter
    report: metrics.SentienceReport = field(default_factory=metrics.new_report)
    iteration: int = 0
    start_time: float = field(default_factory=time)
    last_payload: Dict[str, object] = field(default_factory=dict)
    stop_reason: Optional[str] = None

    def proxy_min(self) -> float:
        values = [getattr(self.report, name) for name in PROXY_NAMES]
        return min(values) if values else 0.0


class CognitiveEngine:
    """Stateful agent loop integrating Echo storage and simulations."""

    def __init__(
        self,
        store: HierarchicalImageMemory,
        *,
        require_benchmarks: bool = True,
        max_iterations: int = 200,
        max_seconds: float = 120.0,
        stopper: Optional[stoplink.StopLink] = None,
    ) -> None:
        self.store = store
        self.require_benchmarks = require_benchmarks
        self.max_iterations = max_iterations
        self.max_seconds = max_seconds
        self.stopper = stopper or stoplink.StopLink()
        self.sessions: Dict[str, SessionState] = {}

    # Session lifecycle ---------------------------------------------------
    def start(self, session_id: Optional[str] = None) -> Dict[str, str]:
        snapshot_id = self.store.create_snapshot(session_id, metadata={"kind": "agent_session"})
        model = SimulatedHumanModel(store=self.store, snapshot_id=snapshot_id)
        trace_writer = persist.AgentTraceWriter(self.store, snapshot_id)
        state = SessionState(snapshot_id=snapshot_id, model=model, trace_writer=trace_writer)
        self.sessions[snapshot_id] = state
        return {"session_id": snapshot_id}

    # Core step -----------------------------------------------------------
    def step(self, session_id: str, goal: str) -> Dict[str, object]:
        if session_id not in self.sessions:
            # Lazy-load existing snapshot if needed
            self.start(session_id)
        state = self.sessions[session_id]
        if state.stop_reason:
            return {"error": "session_stopped", "stop_reason": state.stop_reason}

        state.iteration += 1
        action = policy.plan(goal, state.report, iteration=state.iteration)
        experience = state.model.interact(iteration=state.iteration, goal=goal, action=action)
        updated_report = metrics.update(state.report)
        passed, bench_scores = bench.run(updated_report)
        goal_reached = policy.attempt_goal(updated_report, passed)

        state.report = updated_report

        stop_reason = self._evaluate_stop(state, passed)
        state.stop_reason = stop_reason

        trace_payload = {
            "iteration": state.iteration,
            "proxies": updated_report.as_dict(),
            "bench_scores": bench_scores,
            "passed": passed,
            "action": action,
            "result": {
                "observation": experience.observation,
                "response": experience.response,
                "experience_tile": experience.tile_id,
            },
            "goal": goal,
            "stop_reason": stop_reason,
            "goal_reached": goal_reached,
            "ts": time(),
            "note": "These are proxies of sentience-like behavior, not evidence of sentience.",
        }
        tile_id = state.trace_writer.write(state.iteration, trace_payload)
        trace_payload["tile_id"] = tile_id
        state.last_payload = trace_payload

        response = {
            "iteration": state.iteration,
            "proxies": updated_report.as_dict(),
            "bench": {"passed": passed, "scores": bench_scores},
            "action": action,
            "result": experience.response,
            "experience_tile_id": experience.tile_id,
            "goal_reached": goal_reached,
            "tile_id": tile_id,
        }
        if stop_reason:
            response["stop_reason"] = stop_reason
        return response

    # Reporting -----------------------------------------------------------
    def latest(self, session_id: str) -> Dict[str, object]:
        state = self.sessions.get(session_id)
        if not state:
            tiles = self.store.list_tiles(persist._STREAM_NAME, session_id, level=0)
            if not tiles:
                raise KeyError(f"Unknown session_id: {session_id}")
            latest_tile = tiles[-1]
            tile = self.store.read_tile_by_id(persist._STREAM_NAME, session_id, latest_tile["tile_id"])
            return tile["payload"]
        if not state.last_payload:
            raise KeyError("No iterations executed yet")
        return state.last_payload

    # Internal helpers ----------------------------------------------------
    def _evaluate_stop(self, state: SessionState, passed: bool) -> Optional[str]:
        elapsed = time() - state.start_time
        if self.require_benchmarks and state.proxy_min() >= 0.95 and passed:
            return "benchmarks_reached"
        if state.iteration >= self.max_iterations:
            return "iteration_limit"
        if elapsed >= self.max_seconds:
            return "time_limit"
        if self.stopper.active():
            return "soft_stop"
        return None


__all__ = ["CognitiveEngine"]

