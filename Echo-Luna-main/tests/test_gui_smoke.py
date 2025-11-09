from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtWidgets

from him import HierarchicalImageMemory, create_app
from him.agent import CognitiveEngine, build_router
from him.gui.app import ApiClient, CognitiveEngineWindow


def test_gui_smoke(tmp_path):
    store = HierarchicalImageMemory(tmp_path)
    engine = CognitiveEngine(store)
    router = build_router(engine)
    app = create_app(store, agent_router=router)

    qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    api_client = ApiClient(app=app)
    window = CognitiveEngineWindow(api_client)

    session_id = api_client.start_session()["session_id"]
    window.session_id = session_id
    window.tile_browser.snapshot_edit.setText(session_id)

    step_payload = api_client.step(session_id, "calibrate coherence")
    window._handle_step_payload(step_payload)

    QtWidgets.QApplication.processEvents()

    counts = [series.count() for series in window.proxy_chart.series.values()]
    assert any(count > 0 for count in counts)

    window.close()
    qt_app.quit()

