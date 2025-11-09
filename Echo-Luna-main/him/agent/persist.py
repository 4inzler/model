from __future__ import annotations

"""Persistence helpers for CognitiveEngine++ agent traces."""

from typing import Any, Dict

from ..storage import HierarchicalImageMemory

_STREAM_NAME = "agent_trace"


class AgentTraceWriter:
    """Write iteration payloads into the Echo tile store."""

    def __init__(self, store: HierarchicalImageMemory, snapshot_id: str, stream: str = _STREAM_NAME) -> None:
        self.store = store
        self.snapshot_id = snapshot_id
        self.stream = stream

    def write(self, iteration: int, payload: Dict[str, Any]) -> str:
        tile_meta = self.store.write_tile(
            stream=self.stream,
            snapshot_id=self.snapshot_id,
            level=0,
            x=iteration,
            y=0,
            payload=payload,
            dtype="json",
        )
        return tile_meta.tile_id

    def list(self) -> Dict[str, Any]:
        items = self.store.list_tiles(self.stream, self.snapshot_id, level=0)
        return {"items": items}


__all__ = ["AgentTraceWriter", "_STREAM_NAME"]

