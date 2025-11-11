"""Desktop GUI for controlling the CognitiveEngine++ agent."""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from threading import Thread
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Callable, Dict, List, Optional
from urllib.parse import urlparse, urlunparse

import httpx
import uvicorn
from fastapi import FastAPI
from PyQt6 import QtCore, QtGui, QtWidgets

from ..agent import CognitiveEngine, build_router
from ..api import create_app
from ..storage import HierarchicalImageMemory
from .widgets import ActionLog, ProxyChart, SnapshotTileBrowser




class ApiClient(QtCore.QObject):
    """Thin wrapper around the FastAPI endpoints with async helpers."""

    finished = QtCore.pyqtSignal(str, object)
    failed = QtCore.pyqtSignal(str, str)

    def __init__(self, *, base_url: str = "http://127.0.0.1:8000", app: Optional[FastAPI] = None) -> None:
        super().__init__()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.base_url = base_url
        self.app = app
        self.client: Optional[httpx.Client] = None
        self._server: Optional[uvicorn.Server] = None
        self._server_thread: Optional[Thread] = None
        self._managed_server = False

        if app is not None:
            try:
                self.base_url = self._start_embedded_server(app, base_url)
            except RuntimeError:
                # Fall back to the provided base URL if we cannot host locally
                self._managed_server = False
                self.base_url = base_url

        self.client = httpx.Client(base_url=self.base_url, timeout=5.0)

    # Embedded server management -----------------------------------------
    def _start_embedded_server(self, app: FastAPI, base_url: str) -> str:
        parsed_input = urlparse(base_url if "://" in base_url else f"http://{base_url}")
        scheme = parsed_input.scheme or "http"
        host = parsed_input.hostname or "127.0.0.1"
        requested_port = parsed_input.port

        candidate_ports: List[int] = []
        if requested_port is not None:
            candidate_ports.append(requested_port)
        candidate_ports.append(0)

        for port in candidate_ports:
            config = uvicorn.Config(app, host=host, port=port, log_level="info")
            server = uvicorn.Server(config)
            thread = Thread(target=server.run, daemon=True)
            thread.start()

            for _ in range(200):
                if server.started:
                    break
                if not thread.is_alive():
                    break
                time.sleep(0.05)

            if server.started:
                actual_port: Optional[int] = requested_port if port == requested_port else None
                if server.servers:
                    sockets = getattr(server.servers[0], "sockets", [])
                    if sockets:
                        actual_port = sockets[0].getsockname()[1]
                if not actual_port:
                    server.should_exit = True
                    thread.join(timeout=1.0)
                    continue
                netloc = f"{host}:{actual_port}"
                parsed = parsed_input._replace(netloc=netloc, scheme=scheme)
                self._server = server
                self._server_thread = thread
                self._managed_server = True
                return urlunparse(parsed)

            server.should_exit = True
            thread.join(timeout=1.0)

        raise RuntimeError("Failed to start embedded API server")
    # Core HTTP helpers ---------------------------------------------------
    def _post(self, path: str, payload: Dict[str, object]) -> Dict[str, object]:
        response = self.client.post(path, json=payload)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {"data": data}

    def _get(self, path: str) -> Dict[str, object]:
        response = self.client.get(path)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {"data": data}

    # Public synchronous operations --------------------------------------
    def start_session(self, session_id: Optional[str] = None) -> Dict[str, object]:
        payload = {"session_id": session_id} if session_id else {}
        return self._post("/v1/agent/start", payload)

    def step(self, session_id: str, goal: str) -> Dict[str, object]:
        return self._post("/v1/agent/step", {"session_id": session_id, "goal": goal})

    def get_report(self, session_id: str) -> Dict[str, object]:
        return self._get(f"/v1/agent/report/{session_id}")

    def get_trace(self, session_id: str) -> Dict[str, object]:
        return self._get(f"/v1/agent/trace/{session_id}")

    def list_snapshots(self) -> Dict[str, object]:
        return self._get("/v1/snapshots")

    def get_snapshot(self, snapshot_id: str) -> Dict[str, object]:
        return self._get(f"/v1/snapshots/{snapshot_id}")

    def list_tiles(self, stream: str, snapshot_id: str) -> Dict[str, object]:
        return self._get(f"/v1/tiles/{stream}/{snapshot_id}")

    def fetch_tile(self, stream: str, snapshot_id: str, tile_id: str) -> Dict[str, object]:
        return self._get(f"/v1/tiles/{stream}/{snapshot_id}/{tile_id}")

    def set_stop(self, enabled: bool) -> Dict[str, object]:
        return self._post("/v1/agent/stop", {"enabled": enabled})

    def get_stop(self) -> Dict[str, object]:
        return self._get("/v1/agent/stop")

    def call_async(self, key: str, func: Callable, *args, **kwargs) -> None:
        future = self.executor.submit(func, *args, **kwargs)

        def _done(fut) -> None:
            try:
                result = fut.result()
            except Exception as exc:  # pragma: no cover - UI feedback
                message = str(exc)
                QtCore.QTimer.singleShot(0, lambda msg=message: self.failed.emit(key, msg))
                return
            QtCore.QTimer.singleShot(0, lambda: self.finished.emit(key, result))

        future.add_done_callback(_done)

    def close(self) -> None:
        self.executor.shutdown(wait=False)
        if self.client is not None:
            self.client.close()
        if self._managed_server and self._server is not None:
            self._server.should_exit = True
            if self._server_thread is not None:
                self._server_thread.join(timeout=2.0)



class CognitiveEngineWindow(QtWidgets.QMainWindow):
    """Primary GUI window for CognitiveEngine++."""

    def __init__(
        self,
        api: ApiClient,
        *,
        data_dir: Optional[str] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.api = api
        self.data_dir = Path(data_dir).resolve() if data_dir else None
        self.project_root = Path(__file__).resolve().parents[2]
        self.setWindowTitle("CognitiveEngine++ Controller")
        self.resize(1024, 720)
        self.session_id: Optional[str] = None
        self._pending_requests = 0
        self._latest_experiences: List[Dict[str, object]] = []
        self._busy_label: Optional[QtWidgets.QLabel] = None
        self.auto_timer = QtCore.QTimer(self)
        self.auto_timer.setInterval(2000)
        self.auto_timer.timeout.connect(self._auto_step)

        self._build_ui()
        self._build_menu()
        self._build_status_bar()
        self._update_quick_buttons()

        self.api.finished.connect(self._on_api_finished)
        self.api.failed.connect(self._on_api_failed)
        self._api_call("snapshots", self.api.list_snapshots)
        self._api_call("stop-state", self.api.get_stop)

    # UI construction -----------------------------------------------------
    def _build_ui(self) -> None:
        self._build_toolbar()
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setMovable(True)
        self.setCentralWidget(self.tabs)
        self.tabs.addTab(self._build_dashboard(), "Dashboard")
        self.tabs.setTabToolTip(0, "Monitor live activity and control the agent")
        self.tabs.addTab(self._build_snapshots(), "Snapshots")
        self.tabs.setTabToolTip(1, "Browse stored runs and their tiles")
        self.tabs.addTab(self._build_experiences(), "Experiences")
        self.tabs.setTabToolTip(2, "Inspect human experience tiles captured during runs")

    def _build_menu(self) -> None:
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")

        open_data = QtGui.QAction("Open Data Folder…", self)
        open_data.setToolTip("Open the directory that stores snapshots and tiles")
        open_data.setEnabled(self.data_dir is not None and self.data_dir.exists())
        open_data.triggered.connect(self._open_data_dir)  # type: ignore[arg-type]
        file_menu.addAction(open_data)

        file_menu.addSeparator()
        exit_action = QtGui.QAction("Exit", self)
        exit_action.triggered.connect(self.close)  # type: ignore[arg-type]
        file_menu.addAction(exit_action)

        view_menu = menubar.addMenu("&View")
        reset_layout = QtGui.QAction("Reset Dashboard Layout", self)
        reset_layout.triggered.connect(self._reset_dashboard_split)  # type: ignore[arg-type]
        view_menu.addAction(reset_layout)

        help_menu = menubar.addMenu("&Help")
        open_docs = QtGui.QAction("Open README", self)
        open_docs.setToolTip("Show the project README for full instructions")
        open_docs.triggered.connect(self._open_readme)  # type: ignore[arg-type]
        help_menu.addAction(open_docs)

        about_action = QtGui.QAction("About", self)
        about_action.triggered.connect(self._show_about)  # type: ignore[arg-type]
        help_menu.addAction(about_action)

    def _build_status_bar(self) -> None:
        status_bar = self.statusBar()
        status_bar.setSizeGripEnabled(False)
        self._status_timer = QtCore.QTimer(self)
        self._status_timer.setSingleShot(True)
        self._status_timer.timeout.connect(self._restore_status)
        self._default_status = f"Connected to {self.api.base_url}"
        status_bar.showMessage(self._default_status)
        self._busy_label = QtWidgets.QLabel("Working…", self)
        self._busy_label.setVisible(False)
        status_bar.addPermanentWidget(self._busy_label)
        if self.data_dir:
            data_label = QtWidgets.QLabel(f"Data: {self.data_dir}")
            data_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            status_bar.addPermanentWidget(data_label)

    def _restore_status(self) -> None:
        if hasattr(self, "_default_status"):
            self.statusBar().showMessage(self._default_status)

    def _build_toolbar(self) -> None:
        toolbar = QtWidgets.QToolBar("Controls")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self.session_combo = QtWidgets.QComboBox()
        self.session_combo.setMinimumWidth(220)
        self.session_combo.setEditable(True)
        self.session_combo.lineEdit().setPlaceholderText("Select or enter a session ID")
        self.session_combo.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        self.session_combo.currentTextChanged.connect(self._select_session)  # type: ignore[arg-type]
        toolbar.addWidget(QtWidgets.QLabel("Session:"))
        toolbar.addWidget(self.session_combo)

        start_action = QtGui.QAction("Start Session", self)
        start_action.setShortcut(QtGui.QKeySequence.StandardKey.New)
        start_action.triggered.connect(self._start_session)  # type: ignore[arg-type]
        toolbar.addAction(start_action)

        step_action = QtGui.QAction("One Step", self)
        step_action.setShortcut(QtGui.QKeySequence("Ctrl+Return"))
        step_action.triggered.connect(self._step_once)  # type: ignore[arg-type]
        toolbar.addAction(step_action)

        self.auto_step_action = QtGui.QAction("Auto-Step", self)
        self.auto_step_action.setCheckable(True)
        self.auto_step_action.setShortcut(QtGui.QKeySequence("Ctrl+Shift+A"))
        self.auto_step_action.toggled.connect(self._toggle_auto_step)  # type: ignore[arg-type]
        toolbar.addAction(self.auto_step_action)

        self.stop_action = QtGui.QAction("Soft STOP", self)
        self.stop_action.setCheckable(True)
        self.stop_action.setShortcut(QtGui.QKeySequence("Ctrl+K"))
        self.stop_action.toggled.connect(self._toggle_stop)  # type: ignore[arg-type]
        toolbar.addAction(self.stop_action)

        toolbar.addSeparator()

        clear_log = QtGui.QAction("Clear Log", self)
        clear_log.setShortcut(QtGui.QKeySequence("Ctrl+L"))
        clear_log.triggered.connect(self._clear_action_log)  # type: ignore[arg-type]
        toolbar.addAction(clear_log)

    def _build_dashboard(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        self.dashboard_hint = QtWidgets.QLabel(
            "1. Start or select a session. 2. Describe a goal. 3. Step once or let the"
            " agent auto-step while you observe the proxy chart and action log."
        )
        self.dashboard_hint.setWordWrap(True)
        self.dashboard_hint.setStyleSheet("color: palette(mid);")
        layout.addWidget(self.dashboard_hint)

        self.dashboard_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        layout.addWidget(self.dashboard_splitter, 1)

        chart_container = QtWidgets.QWidget()
        chart_layout = QtWidgets.QVBoxLayout(chart_container)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        chart_layout.setSpacing(8)

        self.proxy_chart = ProxyChart()
        self.proxy_chart.setMinimumHeight(260)
        self.proxy_chart.chart.setTitle("Sentience proxy trends")
        chart_layout.addWidget(self.proxy_chart, 1)

        info_layout = QtWidgets.QGridLayout()
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setHorizontalSpacing(12)
        info_layout.setVerticalSpacing(6)

        goal_label = QtWidgets.QLabel("Goal:")
        goal_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        info_layout.addWidget(goal_label, 0, 0)

        self.goal_edit = QtWidgets.QLineEdit("Stabilise integration")
        self.goal_edit.setClearButtonEnabled(True)
        self.goal_edit.setToolTip("Describe what you want the agent to achieve next")
        self.goal_edit.returnPressed.connect(self._step_once)  # type: ignore[arg-type]
        info_layout.addWidget(self.goal_edit, 0, 1)

        self.bench_label = QtWidgets.QLabel("Benchmarks: –")
        info_layout.addWidget(self.bench_label, 0, 2)

        self.stop_label = QtWidgets.QLabel("STOP: inactive")
        info_layout.addWidget(self.stop_label, 0, 3)

        info_layout.setColumnStretch(1, 1)
        info_layout.setColumnStretch(2, 1)
        chart_layout.addLayout(info_layout)

        control_row = QtWidgets.QHBoxLayout()
        self.quick_session_button = QtWidgets.QPushButton("Start Session")
        self.quick_session_button.clicked.connect(self._start_session)  # type: ignore[arg-type]
        control_row.addWidget(self.quick_session_button)

        self.quick_step_button = QtWidgets.QPushButton("Run One Step")
        self.quick_step_button.clicked.connect(self._step_once)  # type: ignore[arg-type]
        control_row.addWidget(self.quick_step_button)

        self.auto_step_button = QtWidgets.QPushButton("Start Auto-Step")
        self.auto_step_button.setCheckable(True)
        self.auto_step_button.toggled.connect(self.auto_step_action.setChecked)  # type: ignore[arg-type]
        control_row.addWidget(self.auto_step_button)

        self.stop_button = QtWidgets.QPushButton("Enable Soft Stop")
        self.stop_button.setCheckable(True)
        self.stop_button.toggled.connect(self.stop_action.setChecked)  # type: ignore[arg-type]
        control_row.addWidget(self.stop_button)

        self.quick_refresh_button = QtWidgets.QPushButton("Refresh Dashboard")
        self.quick_refresh_button.clicked.connect(self._refresh_dashboard)  # type: ignore[arg-type]
        control_row.addWidget(self.quick_refresh_button)

        control_row.addStretch()
        chart_layout.addLayout(control_row)

        self.dashboard_splitter.addWidget(chart_container)

        log_container = QtWidgets.QWidget()
        log_layout = QtWidgets.QVBoxLayout(log_container)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(6)

        log_header = QtWidgets.QHBoxLayout()
        log_title = QtWidgets.QLabel("Recent activity")
        log_title.setStyleSheet("font-weight: 600;")
        log_header.addWidget(log_title)
        log_header.addStretch()
        self.log_count_label = QtWidgets.QLabel("")
        log_header.addWidget(self.log_count_label)
        self.clear_log_button = QtWidgets.QPushButton("Clear")
        self.clear_log_button.setToolTip("Remove all entries from the activity log")
        self.clear_log_button.clicked.connect(self._clear_action_log)  # type: ignore[arg-type]
        log_header.addWidget(self.clear_log_button)
        log_layout.addLayout(log_header)

        self.action_log = ActionLog()
        self.action_log.setAlternatingRowColors(True)
        self.action_log.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.action_log.setToolTip("Each entry captures an iteration with action, result, and verdict.")
        self.action_log.itemDoubleClicked.connect(self._copy_log_entry)  # type: ignore[arg-type]
        self.action_log.setMinimumHeight(220)
        log_layout.addWidget(self.action_log, 1)

        self.dashboard_splitter.addWidget(log_container)
        self.dashboard_splitter.setStretchFactor(0, 3)
        self.dashboard_splitter.setStretchFactor(1, 2)

        self._update_log_counter()

        return widget

    def _build_snapshots(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        self.snapshot_hint = QtWidgets.QLabel(
            "Snapshots capture every agent run. Double-click a snapshot to resume it, or inspect its tiles below."
        )
        self.snapshot_hint.setWordWrap(True)
        self.snapshot_hint.setStyleSheet("color: palette(mid);")
        layout.addWidget(self.snapshot_hint)

        self.snapshot_list = QtWidgets.QListWidget()
        self.snapshot_list.setAlternatingRowColors(True)
        self.snapshot_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.snapshot_list.itemDoubleClicked.connect(self._resume_session_from_snapshot)  # type: ignore[arg-type]
        self.snapshot_list.currentItemChanged.connect(self._handle_snapshot_selection)  # type: ignore[arg-type]
        layout.addWidget(self.snapshot_list, 2)

        button_layout = QtWidgets.QHBoxLayout()
        refresh_btn = QtWidgets.QPushButton("Refresh")
        refresh_btn.clicked.connect(lambda: self._api_call("snapshots", self.api.list_snapshots))  # type: ignore[arg-type]
        button_layout.addWidget(refresh_btn)

        open_btn = QtWidgets.QPushButton("Open Metadata")
        open_btn.clicked.connect(self._open_snapshot_metadata)  # type: ignore[arg-type]
        button_layout.addWidget(open_btn)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        self.tile_browser = SnapshotTileBrowser(self.api.list_tiles, self.api.fetch_tile)
        self.tile_browser.setMinimumHeight(240)
        layout.addWidget(self.tile_browser, 3)

        return widget

    def _build_experiences(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        self.experience_hint = QtWidgets.QLabel(
            "Each row corresponds to a stored human experience tile. Refresh after stepping to see the latest entries."
        )
        self.experience_hint.setWordWrap(True)
        self.experience_hint.setStyleSheet("color: palette(mid);")
        layout.addWidget(self.experience_hint)

        self.experience_table = QtWidgets.QTableWidget(0, 3)
        self.experience_table.setHorizontalHeaderLabels(["Observation", "Response", "Metadata"])
        self.experience_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.experience_table.verticalHeader().setVisible(False)
        self.experience_table.setAlternatingRowColors(True)
        self.experience_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.experience_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.experience_table, 1)

        button_row = QtWidgets.QHBoxLayout()
        refresh_btn = QtWidgets.QPushButton("Refresh Experiences")
        refresh_btn.clicked.connect(self._refresh_experiences)  # type: ignore[arg-type]
        button_row.addWidget(refresh_btn)

        self.export_experiences_btn = QtWidgets.QPushButton("Export to JSON…")
        self.export_experiences_btn.setEnabled(False)
        self.export_experiences_btn.clicked.connect(self._export_experiences)  # type: ignore[arg-type]
        button_row.addWidget(self.export_experiences_btn)

        button_row.addStretch()
        layout.addLayout(button_row)

        return widget
    def _api_call(self, key: str, func: Callable, *args, **kwargs) -> None:
        self._pending_requests += 1
        self._update_busy_state()
        self.api.call_async(key, partial(func, *args, **kwargs))

    def _update_busy_state(self) -> None:
        busy = self._pending_requests > 0
        if self._busy_label is not None:
            if busy:
                self._busy_label.setText(f"Working… ({self._pending_requests})")
            self._busy_label.setVisible(busy)
        if not busy:
            self._restore_status()

    def _show_status(self, message: str, timeout: int = 4000) -> None:
        self.statusBar().showMessage(message)
        if hasattr(self, "_status_timer"):
            self._status_timer.start(timeout)

    def _clear_action_log(self) -> None:
        self.action_log.clear()
        self._update_log_counter()
        self._show_status("Action log cleared", timeout=2000)

    def _update_log_counter(self) -> None:
        if hasattr(self, "log_count_label"):
            count = self.action_log.count() if hasattr(self, "action_log") else 0
            self.log_count_label.setText(f"{count} entries" if count else "No activity yet")

    def _copy_log_entry(self, item: QtWidgets.QListWidgetItem) -> None:
        QtWidgets.QApplication.clipboard().setText(item.text())
        self._show_status("Copied log entry to clipboard", timeout=1500)

    def _refresh_dashboard(self) -> None:
        if not self.session_id:
            self._show_status("Start a session first to refresh the dashboard", timeout=3000)
            return
        self._api_call("trace", self.api.get_trace, self.session_id)
        self._api_call("snapshots", self.api.list_snapshots)
        self._refresh_experiences()
        self._show_status("Dashboard refresh queued")

    def _reset_dashboard_split(self) -> None:
        if hasattr(self, "dashboard_splitter"):
            total = self.dashboard_splitter.size().height() or 1
            self.dashboard_splitter.setSizes([int(total * 0.6), int(total * 0.4)])

    def _open_data_dir(self) -> None:
        if self.data_dir and self.data_dir.exists():
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.data_dir)))
        else:
            QtWidgets.QMessageBox.information(self, "Data folder", "No data directory is configured yet.")

    def _open_readme(self) -> None:
        readme = self.project_root / "README.md"
        if readme.exists():
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(readme)))
        else:
            QtWidgets.QMessageBox.information(self, "README", "README.md not found in the project root.")

    def _show_about(self) -> None:
        QtWidgets.QMessageBox.information(
            self,
            "About CognitiveEngine++",
            (
                "A friendly control panel for exploring CognitiveEngine++ agent runs.\n"
                "Use the dashboard to guide runs, snapshots to explore history, and experiences for rich context."
            ),
        )

    def _select_session(self, session_id: str) -> None:
        session_id = session_id.strip()
        self.session_id = session_id or None
        if self.session_id:
            self.tile_browser.snapshot_edit.setText(self.session_id)
            self._show_status(f"Active session: {self.session_id}", timeout=2500)
        else:
            self.tile_browser.snapshot_edit.clear()
            if self.auto_step_action.isChecked():
                self.auto_step_action.setChecked(False)

        self._update_quick_buttons()

    def _handle_snapshot_selection(
        self, current: Optional[QtWidgets.QListWidgetItem], _previous: Optional[QtWidgets.QListWidgetItem]
    ) -> None:
        if not current:
            return
        snapshot_id = current.data(QtCore.Qt.ItemDataRole.UserRole)
        if snapshot_id:
            self.tile_browser.snapshot_edit.setText(snapshot_id)
            self.tile_browser.stream_edit.setText("human_experience")

    def _resume_session_from_snapshot(self, item: QtWidgets.QListWidgetItem) -> None:
        snapshot_id = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if not snapshot_id:
            return
        self.session_combo.setCurrentText(snapshot_id)
        self.tile_browser.stream_edit.setText("human_experience")
        self._api_call("trace", self.api.get_trace, snapshot_id)
        self._refresh_experiences()
        self.tabs.setCurrentIndex(0)
        self._show_status(f"Resumed snapshot {snapshot_id}", timeout=3000)

    def _fetch_experience_payloads(self, items: List[Dict[str, object]], session_id: str) -> List[Dict[str, object]]:
        entries: List[Dict[str, object]] = []
        for meta in items:
            tile_id = meta.get("tile_id")
            if not tile_id:
                continue
            try:
                tile = self.api.fetch_tile("human_experience", session_id, tile_id)
            except Exception:
                continue
            entries.append(tile.get("payload", {}))
        return entries

    def _update_experiences_table(self, entries: List[Dict[str, object]]) -> None:
        self._latest_experiences = entries
        self.export_experiences_btn.setEnabled(bool(entries))
        message = (
            f"Showing {len(entries)} experiences for session {self.session_id}"
            if entries and self.session_id
            else "No experiences captured yet. Step the agent to generate new entries."
        )
        self.experience_hint.setText(message)
        self.experience_table.setRowCount(len(entries))
        for row, item in enumerate(entries):
            self.experience_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(item.get("observation", ""))))
            self.experience_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(item.get("response", ""))))
            metadata_text = json.dumps(item.get("extras", {}), indent=2, ensure_ascii=False)
            self.experience_table.setItem(row, 2, QtWidgets.QTableWidgetItem(metadata_text))

    def _export_experiences(self) -> None:
        if not self._latest_experiences:
            return
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export experiences",
            "experiences.json",
            "JSON files (*.json)",
        )
        if not filename:
            return
        path = Path(filename)
        payload = {
            "session_id": self.session_id,
            "generated_at": time.time(),
            "experiences": self._latest_experiences,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._show_status(f"Saved experiences to {path}", timeout=4000)

    def _update_stop_controls(self, active: bool) -> None:
        state = "active" if active else "inactive"
        self.stop_label.setText(f"STOP: {state}")
        if hasattr(self, "stop_button") and self.stop_button is not None:
            self.stop_button.blockSignals(True)
            self.stop_button.setChecked(active)
            self.stop_button.setText("Disable Soft Stop" if active else "Enable Soft Stop")
            self.stop_button.blockSignals(False)
        if hasattr(self, "stop_action") and self.stop_action is not None:
            self.stop_action.blockSignals(True)
            self.stop_action.setChecked(active)
            self.stop_action.blockSignals(False)

    def _sync_auto_controls(self, checked: bool) -> None:
        if hasattr(self, "auto_step_button") and self.auto_step_button is not None:
            self.auto_step_button.blockSignals(True)
            self.auto_step_button.setChecked(checked)
            self.auto_step_button.setText("Stop Auto-Step" if checked else "Start Auto-Step")
            self.auto_step_button.blockSignals(False)

    def _update_quick_buttons(self) -> None:
        has_session = bool(self.session_id)
        if hasattr(self, "quick_step_button") and self.quick_step_button is not None:
            self.quick_step_button.setEnabled(has_session)
        if hasattr(self, "auto_step_button") and self.auto_step_button is not None:
            self.auto_step_button.setEnabled(has_session)
        if self.auto_step_action is not None:
            self.auto_step_action.setEnabled(has_session)
        if hasattr(self, "quick_refresh_button") and self.quick_refresh_button is not None:
            self.quick_refresh_button.setEnabled(has_session)
        if hasattr(self, "stop_button") and self.stop_button is not None:
            self.stop_button.setEnabled(has_session)
        if self.stop_action is not None:
            self.stop_action.setEnabled(has_session)
        if hasattr(self, "export_experiences_btn") and self.export_experiences_btn is not None:
            self.export_experiences_btn.setEnabled(bool(self._latest_experiences))


    # Toolbar actions -----------------------------------------------------
    def _start_session(self) -> None:
        candidate = self.session_combo.currentText() or None
        self._show_status("Starting session…", timeout=2000)
        self._api_call("start", self.api.start_session, candidate)

    def _step_once(self) -> None:
        if not self.session_id:
            self._start_session()
            return
        goal = self.goal_edit.text() or "Explore"
        self._show_status(f"Stepping once toward '{goal}'", timeout=2000)
        self._api_call("step", self.api.step, self.session_id, goal)

    def _toggle_auto_step(self, checked: bool) -> None:
        if checked and not self.session_id:
            self.auto_step_action.blockSignals(True)
            self.auto_step_action.setChecked(False)
            self.auto_step_action.blockSignals(False)
            self._show_status('Start or select a session before enabling auto-step', timeout=3000)
            return
        if checked:
            self.auto_timer.start()
            self._show_status('Auto-step enabled (runs every 2 seconds)', timeout=2500)
        else:
            self.auto_timer.stop()
            self._show_status('Auto-step paused', timeout=2000)
        self._sync_auto_controls(checked)

    def _toggle_stop(self, checked: bool) -> None:
        self._show_status("Updating soft stop state…", timeout=2000)
        self._api_call("stop-toggle", self.api.set_stop, checked)

    def _auto_step(self) -> None:
        if self.session_id:
            self._step_once()

    # API callbacks -------------------------------------------------------
    def _on_api_finished(self, key: str, payload: object) -> None:
        self._pending_requests = max(0, self._pending_requests - 1)
        self._update_busy_state()

        if key == "start":
            session_id = payload.get("session_id") if isinstance(payload, dict) else None
            if session_id:
                if self.session_combo.findText(session_id) == -1:
                    self.session_combo.addItem(session_id)
                self.session_combo.setCurrentText(session_id)
                self.tile_browser.snapshot_edit.setText(session_id)
                self.tile_browser.stream_edit.setText("human_experience")
                self._show_status(f"Session {session_id} ready", timeout=2500)
                self._api_call("trace", self.api.get_trace, session_id)
                self._api_call("snapshots", self.api.list_snapshots)
                self._update_quick_buttons()
        elif key == "step":
            if isinstance(payload, dict):
                self._handle_step_payload(payload)
                iteration = payload.get("iteration")
                if iteration is not None:
                    self._show_status(f"Completed iteration {iteration}", timeout=2500)
        elif key == "trace":
            if isinstance(payload, dict):
                self._update_trace(payload)
        elif key == "snapshots":
            if isinstance(payload, dict):
                self._populate_snapshots(payload)
        elif key == "experiences":
            if isinstance(payload, dict):
                self._populate_experiences(payload)
        elif key == "experience-payloads":
            entries = payload if isinstance(payload, list) else []
            self._update_experiences_table(entries)
            self._show_status(f"Loaded {len(entries)} experience row(s)", timeout=2500)
            self._update_quick_buttons()
        elif key in {"stop-toggle", "stop-state"}:
            if isinstance(payload, dict):
                active = bool(payload.get("active"))
                self._update_stop_controls(active)
                self._show_status(
                    "Soft stop enabled" if active else "Soft stop disabled",
                    timeout=2000,
                )

    def _on_api_failed(self, key: str, message: str) -> None:
        self._pending_requests = max(0, self._pending_requests - 1)
        self._update_busy_state()
        QtWidgets.QMessageBox.warning(self, "API error", f"Operation {key} failed: {message}")
        self._show_status(f"Operation {key} failed: {message}", timeout=4000)

    def _handle_step_payload(self, payload: Dict[str, object]) -> None:
        iteration = int(payload.get("iteration", 0) or 0)
        proxies = payload.get("proxies", {}) if isinstance(payload, dict) else {}
        bench = payload.get("bench", {}) if isinstance(payload, dict) else {}
        action = payload.get("action", "-")
        result = payload.get("result", "-")
        passed = bool(bench.get("passed")) if isinstance(bench, dict) else False
        scores = bench.get("scores", {}) if isinstance(bench, dict) else {}
        self.proxy_chart.append_sample(iteration, proxies)
        if scores:
            score_parts = []
            for key, value in scores.items():
                if isinstance(value, (int, float)):
                    score_parts.append(f"{key}: {value:.2f}")
                else:
                    score_parts.append(f"{key}: {value}")
            score_text = ", ".join(score_parts)
        else:
            score_text = "–"
        status = "pass" if passed else "in progress"
        self.bench_label.setText(f"Benchmarks ({status}): {score_text}")
        self.action_log.add_entry(iteration, str(action), str(result), passed)
        self._update_log_counter()
        if self.session_id:
            self._api_call("trace", self.api.get_trace, self.session_id)
            self._refresh_experiences()

    def _update_trace(self, payload: Dict[str, object]) -> None:
        items = payload.get("items", []) if isinstance(payload, dict) else []
        if not items:
            return
        last = items[-1]
        iteration = int(last.get("iteration", 0) or 0)
        proxies = last.get("proxies", {})
        self.proxy_chart.append_sample(iteration, proxies)

    def _populate_snapshots(self, payload: Dict[str, object]) -> None:
        items = payload.get("items", []) if isinstance(payload, dict) else []
        self.snapshot_list.clear()
        snapshot_ids = []

        if not items:
            placeholder = QtWidgets.QListWidgetItem("No snapshots yet. Start a session to create one.")
            placeholder.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
            self.snapshot_list.addItem(placeholder)
            self.snapshot_hint.setVisible(True)
        else:
            self.snapshot_hint.setVisible(False)
            for item in items:
                snapshot_id = item.get("snapshot_id")
                if not snapshot_id:
                    continue
                snapshot_ids.append(snapshot_id)
                label = f"{snapshot_id} :: {item.get('created_at', 0):.0f}"
                list_item = QtWidgets.QListWidgetItem(label)
                list_item.setData(QtCore.Qt.ItemDataRole.UserRole, snapshot_id)
                self.snapshot_list.addItem(list_item)

        existing_sessions = {self.session_combo.itemText(i) for i in range(self.session_combo.count())}
        for snapshot_id in snapshot_ids:
            if snapshot_id not in existing_sessions:
                self.session_combo.addItem(snapshot_id)

        if self.session_id:
            index = self.session_combo.findText(self.session_id)
            if index != -1:
                self.session_combo.setCurrentIndex(index)
            for row in range(self.snapshot_list.count()):
                item = self.snapshot_list.item(row)
                if item and item.data(QtCore.Qt.ItemDataRole.UserRole) == self.session_id:
                    self.snapshot_list.setCurrentRow(row)
                    break

        self._show_status(f"Loaded {len(snapshot_ids)} snapshot(s)", timeout=2000)

    def _populate_experiences(self, payload: Dict[str, object]) -> None:
        if not self.session_id:
            self._update_experiences_table([])
            return
        items = payload.get("items", []) if isinstance(payload, dict) else []
        if not items:
            self._update_experiences_table([])
            return
        self._api_call("experience-payloads", self._fetch_experience_payloads, items, self.session_id)

    def _refresh_experiences(self) -> None:
        if not self.session_id:
            self._show_status("Select a session to load experiences", timeout=3000)
            return
        self._show_status("Refreshing experiences…", timeout=2000)
        self._api_call("experiences", self.api.list_tiles, "human_experience", self.session_id)

    # Snapshot helpers ----------------------------------------------------
    def _open_snapshot_metadata(self) -> None:
        current = self.snapshot_list.currentItem()
        if not current:
            return
        snapshot_id = current.data(QtCore.Qt.ItemDataRole.UserRole)
        if not snapshot_id:
            return
        if self.api.app is None:
            url = f"{self.api.base_url}/v1/snapshots/{snapshot_id}"
            webbrowser.open(url)
            return
        data = self.api.get_snapshot(snapshot_id)
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as handle:
            json.dump(data, handle, indent=2)
            handle.flush()
            webbrowser.open(f"file://{handle.name}")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        self.api.close()
        super().closeEvent(event)


def launch(
    store: HierarchicalImageMemory,
    *,
    with_agent: bool = True,
    base_url: str = "http://127.0.0.1:8000",
    app: Optional[FastAPI] = None,
) -> int:
    """Launch the PyQt6 GUI."""

    if app is None:
        engine = CognitiveEngine(store)
        router = build_router(engine) if with_agent else None
        app = create_app(store, with_agent=with_agent, agent_router=router)

    qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    api_client = ApiClient(base_url=base_url, app=app)
    window = CognitiveEngineWindow(api_client, data_dir=str(store.data_dir))
    window.show()
    return qt_app.exec()


__all__ = ["launch", "CognitiveEngineWindow", "ApiClient"]

