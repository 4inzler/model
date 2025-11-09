from __future__ import annotations

from him import HierarchicalImageMemory
from him.agent import CognitiveEngine


def test_simulated_human_persistence(tmp_path):
    store = HierarchicalImageMemory(tmp_path)
    engine = CognitiveEngine(store)
    session_id = engine.start()["session_id"]

    result = engine.step(session_id, "investigate resonance")
    assert "experience_tile_id" in result

    tiles = store.list_tiles("human_experience", session_id, level=0)
    assert len(tiles) == 1
    tile = store.read_tile_by_id("human_experience", session_id, tiles[0]["tile_id"])
    payload = tile["payload"]
    assert payload["goal"] == "investigate resonance"
    assert payload["iteration"] == 1

