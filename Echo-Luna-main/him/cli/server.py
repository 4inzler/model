from __future__ import annotations

"""CLI helpers for launching the CognitiveEngine++ server."""

from typing import Optional, Tuple

import uvicorn

from ..agent import CognitiveEngine, build_router
from ..api import create_app
from ..storage import HierarchicalImageMemory


def build_app(data_dir: str, *, with_agent: bool = True):
    store = HierarchicalImageMemory(data_dir)
    engine = CognitiveEngine(store) if with_agent else None
    agent_router = build_router(engine) if engine else None
    app = create_app(store, with_agent=with_agent, agent_router=agent_router)
    if engine is not None:
        app.state.engine = engine
    return app, engine, store


def serve(
    data_dir: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    with_agent: bool = True,
    reload: bool = False,
) -> None:
    app, _, _ = build_app(data_dir, with_agent=with_agent)
    uvicorn.run(app, host=host, port=port, reload=reload)


__all__ = ["build_app", "serve"]

