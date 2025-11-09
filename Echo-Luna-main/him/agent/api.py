from __future__ import annotations

"""FastAPI router exposing CognitiveEngine++ agent endpoints."""

from typing import Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..storage import HierarchicalImageMemory
from .engine import CognitiveEngine
from .persist import _STREAM_NAME


class StartRequest(BaseModel):
    session_id: str | None = None


class StartResponse(BaseModel):
    session_id: str


class StepRequest(BaseModel):
    session_id: str
    goal: str


class StepResponse(BaseModel):
    iteration: int
    proxies: Dict[str, float]
    bench: Dict[str, object]
    action: str
    result: str
    experience_tile_id: str
    goal_reached: bool
    tile_id: str
    stop_reason: str | None = None


class StopRequest(BaseModel):
    enabled: bool


class StopResponse(BaseModel):
    active: bool

def build_router(engine: CognitiveEngine) -> APIRouter:
    router = APIRouter(prefix="/v1/agent", tags=["agent"])

    @router.post("/start", response_model=StartResponse)
    def start_endpoint(payload: StartRequest) -> Dict[str, str]:
        return engine.start(payload.session_id)

    @router.post("/step", response_model=StepResponse)
    def step_endpoint(payload: StepRequest) -> Dict[str, object]:
        result = engine.step(payload.session_id, payload.goal)
        if "error" in result:
            raise HTTPException(status_code=409, detail=result)
        return result

    @router.post("/stop", response_model=StopResponse)
    def stop_endpoint(payload: StopRequest) -> StopResponse:
        engine.stopper.set_soft_stop(payload.enabled)
        return StopResponse(active=engine.stopper.active())

    @router.get("/stop", response_model=StopResponse)
    def stop_state() -> StopResponse:
        return StopResponse(active=engine.stopper.active())

    @router.get("/report/{session_id}")
    def report_endpoint(session_id: str) -> Dict[str, object]:
        try:
            latest = engine.latest(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        proxies = latest.get("proxies", {})
        bench_summary = {
            "scores": latest.get("bench_scores", {}),
            "passed": latest.get("passed", False),
        }
        return {"session_id": session_id, "proxies": proxies, "bench": bench_summary}

    @router.get("/trace/{session_id}")
    def trace_endpoint(session_id: str) -> Dict[str, List[Dict[str, object]]]:
        store: HierarchicalImageMemory = engine.store
        tiles = store.list_tiles(_STREAM_NAME, session_id, level=0)
        items: List[Dict[str, object]] = []
        for tile_meta in tiles:
            tile = store.read_tile_by_id(_STREAM_NAME, session_id, tile_meta["tile_id"])
            payload = tile.get("payload", {})
            items.append(
                {
                    "iteration": payload.get("iteration"),
                    "proxies": payload.get("proxies"),
                    "bench_scores": payload.get("bench_scores"),
                    "passed": payload.get("passed"),
                    "action": payload.get("action"),
                    "stop_reason": payload.get("stop_reason"),
                    "tile_id": tile_meta["tile_id"],
                }
            )
        return {"items": items}

    return router


__all__ = ["build_router"]
