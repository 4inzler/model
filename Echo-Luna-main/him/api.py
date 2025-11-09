from __future__ import annotations

"""FastAPI application factory for Echo-Luna HIM cognitive services."""

from typing import Annotated, Any, Dict, Optional

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request

from .storage import HierarchicalImageMemory


def _get_store(app: FastAPI) -> HierarchicalImageMemory:
    store = getattr(app.state, "store", None)
    if store is None:
        raise RuntimeError("Application store has not been configured")
    return store


def create_app(store: HierarchicalImageMemory, *, with_agent: bool = True, agent_router: Optional[APIRouter] = None) -> FastAPI:
    """Create a FastAPI instance configured with Project Echo compatible routes."""

    app = FastAPI(title="CognitiveEngine++", version="0.1.0", docs_url="/docs")
    app.state.store = store

    router = APIRouter(prefix="/v1", tags=["core"])

    @router.get("/snapshots")
    def list_snapshots(request: Request) -> Dict[str, Any]:
        return {"items": _get_store(request.app).list_snapshots()}

    @router.post("/snapshots")
    def create_snapshot(payload: Dict[str, Any], request: Request) -> Dict[str, Any]:
        snapshot_id = payload.get("snapshot_id")
        metadata = payload.get("metadata")
        store = _get_store(request.app)
        snapshot_id = store.create_snapshot(snapshot_id, metadata=metadata)
        return {"snapshot_id": snapshot_id}

    @router.get("/snapshots/{snapshot_id}")
    def get_snapshot(snapshot_id: str, request: Request) -> Dict[str, Any]:
        store = _get_store(request.app)
        try:
            return store.get_snapshot(snapshot_id)
        except KeyError as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/tiles/{stream}/{snapshot_id}")
    def list_tiles(
        stream: str,
        snapshot_id: str,
        level: Annotated[Optional[int], Query(default=None)] = None,
        x: Annotated[Optional[int], Query(default=None)] = None,
        y: Annotated[Optional[int], Query(default=None)] = None,
        request: Request,
    ) -> Dict[str, Any]:
        store = _get_store(request.app)
        tiles = store.list_tiles(stream, snapshot_id, level=level, x=x, y=y)
        return {"items": tiles}

    @router.get("/tiles/{stream}/{snapshot_id}/{tile_id}")
    def fetch_tile(stream: str, snapshot_id: str, tile_id: str, request: Request) -> Dict[str, Any]:
        store = _get_store(request.app)
        try:
            return store.read_tile_by_id(stream, snapshot_id, tile_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    app.include_router(router)

    if with_agent and agent_router is not None:
        app.include_router(agent_router)

    return app


__all__ = ["create_app"]
