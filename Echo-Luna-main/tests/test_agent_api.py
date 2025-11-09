from __future__ import annotations

from fastapi.testclient import TestClient

from him import HierarchicalImageMemory, create_app
from him.agent import CognitiveEngine, build_router


def test_start_and_step_flow(tmp_path):
    store = HierarchicalImageMemory(tmp_path)
    engine = CognitiveEngine(store)
    router = build_router(engine)
    app = create_app(store, agent_router=router)
    client = TestClient(app)

    start_resp = client.post("/v1/agent/start", json={})
    assert start_resp.status_code == 200
    session_id = start_resp.json()["session_id"]

    step_resp = client.post("/v1/agent/step", json={"session_id": session_id, "goal": "stabilise"})
    payload = step_resp.json()
    assert "proxies" in payload
    for value in payload["proxies"].values():
        assert 0.0 <= value <= 1.0

    tiles = store.list_tiles("agent_trace", session_id, level=0)
    assert tiles, "agent step should persist a tile"

    report_resp = client.get(f"/v1/agent/report/{session_id}")
    assert report_resp.status_code == 200
    data = report_resp.json()
    assert set(data["proxies"].keys()) == set(payload["proxies"].keys())

