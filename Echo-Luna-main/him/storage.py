from __future__ import annotations

"""Storage primitives for CognitiveEngine++ built on HierarchicalImageMemory."""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from time import time
from typing import Any, Dict, List, Optional
import uuid


def _ensure_json_ready(payload: Any, *, dtype: str) -> bytes:
    if dtype == "json":
        if isinstance(payload, (bytes, bytearray)):
            return bytes(payload)
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if isinstance(payload, str):
        return payload.encode("utf-8")
    if isinstance(payload, (bytes, bytearray)):
        return bytes(payload)
    raise TypeError(f"Unsupported payload type for dtype={dtype!r}: {type(payload)!r}")


def _clamp_path(base: Path, candidate: Path) -> Path:
    candidate = candidate.resolve()
    base = base.resolve()
    if not str(candidate).startswith(str(base)):
        raise ValueError("Candidate path escapes base directory")
    return candidate


@dataclass(frozen=True)
class TileMetadata:
    tile_id: str
    stream: str
    snapshot_id: str
    level: int
    x: int
    y: int
    dtype: str
    size: int
    created_at: float
    filename: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tile_id": self.tile_id,
            "stream": self.stream,
            "snapshot_id": self.snapshot_id,
            "level": self.level,
            "x": self.x,
            "y": self.y,
            "dtype": self.dtype,
            "size": self.size,
            "created_at": self.created_at,
            "filename": self.filename,
        }


class HierarchicalImageMemory:
    """Filesystem-backed storage for Echo snapshots and tiles.

    The implementation mirrors the original Project Echo layout while keeping the
    code local-only. Tiles are content-addressed by SHA256 digests and organised
    as ``data_dir/tiles/<stream>/<snapshot_id>/L<level>/x<int>/y<int>/``.
    """

    def __init__(self, data_dir: os.PathLike[str] | str) -> None:
        self.data_dir = Path(data_dir)
        self.snapshots_dir = self.data_dir / "snapshots"
        self.tiles_dir = self.data_dir / "tiles"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.tiles_dir.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    # Snapshot management -------------------------------------------------
    def create_snapshot(self, snapshot_id: Optional[str] = None, *, metadata: Optional[Dict[str, Any]] = None) -> str:
        with self._lock:
            snapshot_id = snapshot_id or uuid.uuid4().hex
            snapshot_path = self.snapshots_dir / snapshot_id
            snapshot_path.mkdir(parents=True, exist_ok=True)
            meta = metadata.copy() if metadata else {}
            meta.setdefault("snapshot_id", snapshot_id)
            meta.setdefault("created_at", time())
            meta_path = snapshot_path / "metadata.json"
            if meta_path.exists():
                existing = json.loads(meta_path.read_text(encoding="utf-8"))
                existing.update(meta)
                meta = existing
            meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
            return snapshot_id

    def ensure_snapshot(self, snapshot_id: str, *, metadata: Optional[Dict[str, Any]] = None) -> str:
        with self._lock:
            snapshot_path = self.snapshots_dir / snapshot_id
            if snapshot_path.exists():
                if metadata:
                    self.update_snapshot_metadata(snapshot_id, metadata)
                return snapshot_id
            return self.create_snapshot(snapshot_id, metadata=metadata)

    def update_snapshot_metadata(self, snapshot_id: str, updates: Dict[str, Any]) -> None:
        with self._lock:
            meta_path = self.snapshots_dir / snapshot_id / "metadata.json"
            if not meta_path.exists():
                raise KeyError(f"Unknown snapshot_id: {snapshot_id}")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta.update(updates)
            meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    def list_snapshots(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for entry in sorted(self.snapshots_dir.iterdir()):
            if not entry.is_dir():
                continue
            meta_path = entry / "metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            else:
                meta = {"snapshot_id": entry.name, "created_at": 0.0}
            results.append(meta)
        results.sort(key=lambda item: item.get("created_at", 0.0))
        return results

    def get_snapshot(self, snapshot_id: str) -> Dict[str, Any]:
        meta_path = self.snapshots_dir / snapshot_id / "metadata.json"
        if not meta_path.exists():
            raise KeyError(f"Unknown snapshot_id: {snapshot_id}")
        return json.loads(meta_path.read_text(encoding="utf-8"))

    # Tile management -----------------------------------------------------
    def _tile_dir(self, stream: str, snapshot_id: str, level: int, x: int, y: int) -> Path:
        return self.tiles_dir / stream / snapshot_id / f"L{level}" / f"x{x}" / f"y{y}"

    def write_tile(
        self,
        *,
        stream: str,
        snapshot_id: str,
        level: int,
        x: int,
        y: int,
        payload: Any,
        dtype: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TileMetadata:
        with self._lock:
            self.ensure_snapshot(snapshot_id)
            tile_dir = self._tile_dir(stream, snapshot_id, level, x, y)
            tile_dir.mkdir(parents=True, exist_ok=True)
            blob = _ensure_json_ready(payload, dtype=dtype)
            digest = hashlib.sha256(blob).hexdigest()
            ext = "json" if dtype == "json" else "bin"
            filename = f"{digest}.{ext}"
            tile_path = tile_dir / filename
            if not tile_path.exists():
                tile_path.write_bytes(blob)
            meta = TileMetadata(
                tile_id=digest,
                stream=stream,
                snapshot_id=snapshot_id,
                level=level,
                x=x,
                y=y,
                dtype=dtype,
                size=len(blob),
                created_at=time(),
                filename=filename,
            )
            if metadata:
                meta_payload = meta.as_dict() | {"metadata": metadata}
            else:
                meta_payload = meta.as_dict()
            index_path = tile_dir / "index.json"
            index_path.write_text(json.dumps(meta_payload, indent=2, sort_keys=True), encoding="utf-8")
            return meta

    def list_tiles(
        self,
        stream: str,
        snapshot_id: str,
        *,
        level: Optional[int] = None,
        x: Optional[int] = None,
        y: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        base = self.tiles_dir / stream / snapshot_id
        if not base.exists():
            return []
        results: List[Dict[str, Any]] = []
        for lvl_dir in base.iterdir():
            if not lvl_dir.is_dir():
                continue
            lvl = int(lvl_dir.name.lstrip('L'))
            if level is not None and lvl != level:
                continue
            for x_dir in lvl_dir.iterdir():
                if not x_dir.is_dir():
                    continue
                x_val = int(x_dir.name.lstrip('x'))
                if x is not None and x_val != x:
                    continue
                for y_dir in x_dir.iterdir():
                    if not y_dir.is_dir():
                        continue
                    y_val = int(y_dir.name.lstrip('y'))
                    if y is not None and y_val != y:
                        continue
                    index_path = y_dir / "index.json"
                    if index_path.exists():
                        payload = json.loads(index_path.read_text(encoding="utf-8"))
                        results.append(payload)
        results.sort(key=lambda item: (item.get("level", 0), item.get("x", 0), item.get("y", 0), item.get("created_at", 0.0)))
        return results

    def read_tile_payload(
        self,
        stream: str,
        snapshot_id: str,
        *,
        level: int,
        x: int,
        y: int,
    ) -> Dict[str, Any]:
        tile_dir = self._tile_dir(stream, snapshot_id, level, x, y)
        index_path = tile_dir / "index.json"
        if not index_path.exists():
            raise FileNotFoundError("Tile index not found")
        meta = json.loads(index_path.read_text(encoding="utf-8"))
        filename = meta.get("filename")
        if not filename:
            raise FileNotFoundError("Tile payload filename missing")
        tile_path = tile_dir / filename
        _clamp_path(self.tiles_dir, tile_path)
        if meta.get("dtype") == "json":
            payload = json.loads(tile_path.read_text(encoding="utf-8"))
        else:
            payload = tile_path.read_bytes()
        meta["payload"] = payload
        return meta

    def read_tile_by_id(self, stream: str, snapshot_id: str, tile_id: str) -> Dict[str, Any]:
        base = self.tiles_dir / stream / snapshot_id
        if not base.exists():
            raise FileNotFoundError("Stream has no tiles for snapshot")
        for lvl_dir in base.iterdir():
            for x_dir in lvl_dir.iterdir():
                for y_dir in x_dir.iterdir():
                    index_path = y_dir / "index.json"
                    if not index_path.exists():
                        continue
                    meta = json.loads(index_path.read_text(encoding="utf-8"))
                    if meta.get("tile_id") == tile_id:
                        filename = meta.get("filename")
                        tile_path = y_dir / filename
                        if meta.get("dtype") == "json":
                            payload = json.loads(tile_path.read_text(encoding="utf-8"))
                        else:
                            payload = tile_path.read_bytes()
                        meta["payload"] = payload
                        return meta
        raise FileNotFoundError(f"Tile {tile_id} not found for {stream}/{snapshot_id}")

