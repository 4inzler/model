from __future__ import annotations

"""Reusable PyQt6 widgets for the CognitiveEngine++ GUI."""

import json
from typing import Callable, Dict, Optional

from PyQt6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter
from PyQt6.QtWidgets import (
    QFormLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from echolab.state import PROXY_NAMES


class ProxyChart(QChartView):
    """Rolling line chart visualising proxies of sentience-like behavior."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        self.chart = QChart()
        super().__init__(self.chart, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.series: Dict[str, QLineSeries] = {}
        for name in PROXY_NAMES:
            series = QLineSeries(name=name)
            self.chart.addSeries(series)
            self.series[name] = series
        self.axis_x = QValueAxis()
        self.axis_x.setTitleText("Iteration")
        self.axis_y = QValueAxis()
        self.axis_y.setRange(0.0, 1.0)
        self.axis_y.setTitleText("Proxy Value")
        self.chart.addAxis(self.axis_x, Qt.AlignmentFlag.AlignBottom)
        self.chart.addAxis(self.axis_y, Qt.AlignmentFlag.AlignLeft)
        for series in self.series.values():
            series.attachAxis(self.axis_x)
            series.attachAxis(self.axis_y)
        self._max_points = 100

    def append_sample(self, iteration: int, proxies: Dict[str, float]) -> None:
        for name, series in self.series.items():
            series.append(float(iteration), float(proxies.get(name, 0.0)))
            if series.count() > self._max_points:
                series.removePoints(0, series.count() - self._max_points)
        self.axis_x.setRange(max(0, iteration - self._max_points), iteration + 1)


class ActionLog(QListWidget):
    """Chronological list of agent actions and benchmark results."""

    def add_entry(self, iteration: int, action: str, result: str, passed: bool) -> None:
        verdict = "pass" if passed else "retry"
        text = f"#{iteration} :: {action} -> {result} [{verdict}]"
        item = QListWidgetItem(text)
        self.insertItem(0, item)


class SnapshotTileBrowser(QWidget):
    """Utility widget for inspecting snapshot tiles."""

    def __init__(
        self,
        fetch_tiles: Callable[[str, str], Dict[str, Dict[str, object]]],
        fetch_tile_payload: Callable[[str, str, str], Dict[str, object]],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.fetch_tiles = fetch_tiles
        self.fetch_tile_payload = fetch_tile_payload

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.stream_edit = QLineEdit("agent_trace")
        self.snapshot_edit = QLineEdit()
        form.addRow("Stream", self.stream_edit)
        form.addRow("Snapshot", self.snapshot_edit)
        layout.addLayout(form)

        self.tile_list = QListWidget()
        layout.addWidget(self.tile_list)

        self.refresh_btn = QPushButton("Refresh Tiles")
        self.refresh_btn.clicked.connect(self.refresh_tiles)  # type: ignore[arg-type]
        layout.addWidget(self.refresh_btn)

        self.payload_view = QPlainTextEdit()
        self.payload_view.setReadOnly(True)
        layout.addWidget(self.payload_view)

        self.tile_list.itemSelectionChanged.connect(self._load_selected)  # type: ignore[arg-type]

    def refresh_tiles(self) -> None:
        stream = self.stream_edit.text().strip()
        snapshot = self.snapshot_edit.text().strip()
        if not stream or not snapshot:
            return
        response = self.fetch_tiles(stream, snapshot)
        self.tile_list.clear()
        for item in response.get("items", []):
            tile_id = item.get("tile_id")
            iteration = item.get("x")
            descriptor = f"{tile_id} @ x{iteration}" if tile_id else "unknown"
            list_item = QListWidgetItem(descriptor)
            list_item.setData(Qt.ItemDataRole.UserRole, tile_id)
            self.tile_list.addItem(list_item)

    def _load_selected(self) -> None:
        current = self.tile_list.currentItem()
        if not current:
            return
        tile_id = current.data(Qt.ItemDataRole.UserRole)
        stream = self.stream_edit.text().strip()
        snapshot = self.snapshot_edit.text().strip()
        if not tile_id or not stream or not snapshot:
            return
        payload = self.fetch_tile_payload(stream, snapshot, tile_id)
        pretty = json.dumps(payload.get("payload", payload), indent=2, sort_keys=True)
        self.payload_view.setPlainText(pretty)


__all__ = ["ProxyChart", "ActionLog", "SnapshotTileBrowser"]

