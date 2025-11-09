#!/usr/bin/env python3
"""
H.I.M. Model GUI - Real-time Visualization and Chat Interface
============================================================

A comprehensive GUI that shows the H.I.M. model's consciousness simulation
in real-time with visual displays of all sectors, emotions, hormones, and
processing steps. Includes a chat interface for interaction.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import time
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import queue
import json
import os
import pickle
from datetime import datetime
from collections import defaultdict
import psutil
import importlib
import importlib.util

from him_model import HIMModel
from him_machine_learning import AsyncTrainingSession, TrainingExample, OnlineHIMLogisticTrainer


def _load_optional_module(module_name):
    """Return the imported module if available, otherwise ``None``."""
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return None
    return importlib.import_module(module_name)


_pyautogui = _load_optional_module("pyautogui")


class HIMGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("H.I.M. Model - Hierarchical Image Model GUI")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1e1e1e')
        self.root.bind('<Escape>', self.emergency_stop)

        # Shared color palette for dark theme widgets
        self.colors = {
            "surface": "#1e1e1e",
            "text": "#f5f5f5",
            "accent": "#66b3ff",
        }

        # Initialize H.I.M. model
        self.him_model = HIMModel()
        self.is_running = False
        self.cycle_count = 0
        self.message_queue = queue.Queue()

        # Remote control and automation state
        self.pyautogui = _pyautogui
        self.remote_control_enabled = False
        self.remote_stop_event = threading.Event()
        self.remote_capture_thread = None
        self.remote_frame_queue = queue.Queue(maxsize=2)
        self.remote_preview_photo = None
        self.latest_screen_frame = None
        self.remote_status_var = tk.StringVar(value="Remote control unavailable")
        self.remote_refresh_interval = tk.DoubleVar(value=0.5)
        self.mouse_x_var = tk.StringVar(value="0")
        self.mouse_y_var = tk.StringVar(value="0")
        self.mouse_button_var = tk.StringVar(value="left")
        self.keypress_var = tk.StringVar(value="enter")
        self.keyboard_text_var = tk.StringVar()
        self.scroll_amount_var = tk.IntVar(value=0)
        self.use_live_screen_var = tk.BooleanVar(value=True)
        self.remote_has_logged_frame = False

        # Performance and persistence settings
        self.cycle_delay = 2.0  # Default 2 seconds between cycles
        self.auto_save_interval = 30  # Auto-save every 30 seconds
        self.last_auto_save = time.time()
        self.save_directory = "him_saves"
        self.performance_mode = "balanced"  # low, balanced, high
        self.auto_save_enabled = True  # Enable auto-saving by default
        self.vector_combination_enabled = True  # Enable vector combination to save space
        
        # Create save directory
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)
        
        # Data storage for plotting
        status_snapshot = self.him_model.get_system_status()
        self.emotion_history = defaultdict(list)
        for emotion in status_snapshot['sector_one_emotions']:
            self.emotion_history[emotion] = []

        self.hormone_history = defaultdict(list)
        for hormone in status_snapshot['sector_four_hormones']:
            self.hormone_history[hormone] = []

        self.time_history = []

        # Machine learning controls
        self.training_session = None
        self.training_start_btn = None
        self.training_stop_btn = None
        self.training_log_text = None
        self.cycle_summary_listbox = None
        self.manual_training_text = None
        self.last_cycle_result = None
        self.cycle_summary_records = []

        self.training_status_var = tk.StringVar(value="Trainer idle")
        self.training_updates_var = tk.StringVar(value="Updates: 0")
        self.training_queue_var = tk.StringVar(value="Queue: 0")
        self.training_loss_var = tk.StringVar(value="Recent loss: —")
        self.prediction_result_var = tk.StringVar(value="Probability: —")
        self.training_learning_rate = tk.DoubleVar(value=0.05)
        self.training_l2_var = tk.DoubleVar(value=0.001)
        self.manual_label_var = tk.StringVar(value="positive")

        self.sample_training_dataset = [
            ("The system feels calm and curious about the new task.", 1),
            ("Threat detected! Initiating preservation subroutines.", 0),
            ("Joyful collaboration scenario detected.", 1),
            ("High stress and urgent responses required immediately.", 0),
        ]

        # Persona interface state
        self.persona_name_var = tk.StringVar(value="Uploaded Presence: —")
        self.persona_mood_var = tk.StringVar(value="Mood: —")
        self.persona_voice_var = tk.StringVar(value="Voice: —")
        self.persona_conversation_var = tk.StringVar(value="Conversation Style: —")
        self.persona_last_snapshot = None
        self.persona_current_name = "Luna"
        self.persona_editor_initialized = False

        self.persona_edit_name_var = tk.StringVar()
        self.persona_edit_conversation_var = tk.StringVar()
        self.persona_edit_voice_var = tk.StringVar()

        self.persona_memory_summary_var = tk.StringVar()
        self.persona_memory_emotion_var = tk.StringVar(value="joy")
        self.persona_memory_intensity_var = tk.DoubleVar(value=0.5)

        self.persona_identity_display = None
        self.persona_biography_display = None
        self.persona_upload_display = None
        self.persona_inner_thought_display = None
        self.persona_memory_display = None
        self.persona_narrative_display = None
        self.persona_values_listbox = None
        self.persona_goals_listbox = None
        self.persona_skills_listbox = None
        self.persona_identity_edit = None
        self.persona_biography_edit = None
        self.persona_upload_edit = None
        self.persona_emotion_menu = None
        self.persona_intensity_label = None

        self.setup_gui()

        # Try to load previous state (after GUI is set up)
        self.load_model_state()

        self.start_update_loop()
        self.update_training_status()
        
    def setup_gui(self):
        """Setup the main GUI layout."""
        # Create main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_dashboard_tab()
        self.create_emotions_tab()
        self.create_hormones_tab()
        self.create_persona_tab()
        self.create_machine_learning_tab()
        self.create_memory_tab()
        self.create_chat_tab()
        self.create_remote_control_tab()
        self.create_visual_tab()

    def _bind_mousewheel(self, widget, canvas):
        """Bind mouse wheel scrolling to a canvas for the provided widget."""

        def _on_mousewheel(event):
            if event.delta:
                canvas.yview_scroll(int(-event.delta / 120) or (-1 if event.delta > 0 else 1), "units")
            elif event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")

        def _bind(_event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
            canvas.bind_all("<Button-4>", _on_mousewheel)
            canvas.bind_all("<Button-5>", _on_mousewheel)

        def _unbind(_event):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        widget.bind("<Enter>", _bind)
        widget.bind("<Leave>", _unbind)

    def _add_scrollable_tab(self, title):
        """Create a notebook tab with vertical scrolling support."""

        container = ttk.Frame(self.notebook)
        self.notebook.add(container, text=title)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        canvas = tk.Canvas(container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        inner = ttk.Frame(canvas)
        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _sync_scrollregion(_event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _resize_inner(event):
            canvas.itemconfigure(window_id, width=event.width)

        inner.bind("<Configure>", _sync_scrollregion)
        canvas.bind("<Configure>", _resize_inner)

        self._bind_mousewheel(inner, canvas)

        return inner
        
    def _update_state_indicator_color(self, state):
        """Update the dashboard status indicator dot and label."""
        style = ttk.Style()
        normalized = (state or "unknown").lower()
        palette = {
            "active": ("#4caf50", "Running"),
            "running": ("#4caf50", "Running"),
            "starting": ("#ffb74d", "Starting..."),
            "initializing": ("#ffb74d", "Initializing..."),
            "stopped": ("#f44336", "Stopped"),
            "error": ("#ff5252", "Error"),
            "idle": ("#9e9e9e", "Idle"),
        }
        color, label_text = palette.get(normalized, ("#9e9e9e", None))
        style.configure("StateIndicator.TLabel", foreground=color)
        if hasattr(self, "state_indicator"):
            try:
                self.state_indicator.configure(foreground=color)
            except tk.TclError:
                pass
        if label_text and hasattr(self, "runtime_status_label"):
            self.runtime_status_label.config(text=label_text)

    def _create_metric_row(self, parent, title, attr_name, row, initial="—"):
        """Create a labelled value row and keep a reference to the value label."""
        title_label = ttk.Label(parent, text=title, style="MetricTitle.TLabel")
        title_label.grid(row=row, column=0, sticky=tk.W, pady=(0, 4))
        value_label = ttk.Label(parent, text=initial, style="MetricValue.TLabel")
        value_label.grid(row=row, column=1, sticky=tk.W, padx=(8, 0), pady=(0, 4))
        setattr(self, f"{attr_name}_label", value_label)
        return value_label

    def _truncate_text(self, value, limit):
        """Return value capped to limit characters with ellipsis."""
        if value is None:
            return ""
        text = str(value)
        if len(text) <= limit:
            return text
        if limit <= 3:
            return text[:limit]
        return text[: limit - 3].rstrip() + "..."

    def _append_status_log(self, message):
        """Safely append a line to the dashboard status log."""
        if not hasattr(self, "status_text"):
            return
        self.status_text.configure(state=tk.NORMAL)
        self.status_text.insert(tk.END, message)
        self.status_text.see(tk.END)
        self.status_text.configure(state=tk.DISABLED)

    def create_dashboard_tab(self):
        """Create the main dashboard tab."""
        dashboard_frame = ttk.Frame(self.notebook)
        self.notebook.add(dashboard_frame, text="Dashboard")

        header = ttk.Frame(dashboard_frame, padding=(10, 10, 10, 0))
        header.pack(fill=tk.X)
        ttk.Label(header, text="Real-time System Overview", style="Header.TLabel").pack(anchor=tk.W)
        ttk.Label(
            header,
            text="Monitor cycles, performance, and activity logs in one place.",
            style="Subheader.TLabel",
        ).pack(anchor=tk.W, pady=(2, 0))

        control_card = ttk.LabelFrame(dashboard_frame, text="Control Center", padding=12, style="Card.TLabelframe")
        control_card.pack(fill=tk.X, padx=10, pady=(10, 8))

        button_row = ttk.Frame(control_card, style="Card.TFrame")
        button_row.pack(fill=tk.X)
        self.start_btn = ttk.Button(button_row, text="Start H.I.M. Model", style="Accent.TButton", command=self.start_model)
        self.start_btn.pack(side=tk.LEFT)

        self.stop_btn = ttk.Button(button_row, text="Stop Model", style="Danger.TButton", command=self.stop_model, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(10, 0))

        ttk.Separator(control_card, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        action_row = ttk.Frame(control_card, style="Card.TFrame")
        action_row.pack(fill=tk.X)

        button_specs = [
            ("Load Test Image", self.load_test_image),
            ("Save State", self.save_model_state),
            ("Load State", self.load_model_state),
            ("Save Memories to Text", self.save_memories_to_text),
            ("Save Zipline Data", self.save_zipline_data),
            ("Load Luna-Trained Model", self.load_luna_trained_model),
        ]
        for index, (label, command) in enumerate(button_specs):
            ttk.Button(action_row, text=label, command=command).pack(side=tk.LEFT, padx=(0 if index == 0 else 8, 0), pady=2)

        perf_card = ttk.LabelFrame(
            dashboard_frame,
            text="Performance & Persistence",
            padding=12,
            style="Card.TLabelframe",
        )
        perf_card.pack(fill=tk.X, padx=10, pady=(0, 10))
        perf_card.columnconfigure(1, weight=1)
        perf_card.columnconfigure(3, weight=1)

        ttk.Label(perf_card, text="Performance Mode:", style="MetricTitle.TLabel").grid(row=0, column=0, sticky=tk.W)
        self.perf_mode_var = tk.StringVar(value=self.performance_mode)
        perf_combo = ttk.Combobox(
            perf_card,
            textvariable=self.perf_mode_var,
            values=["low", "balanced", "high"],
            state="readonly",
            width=12,
        )
        perf_combo.grid(row=0, column=1, sticky=tk.W, padx=(8, 16))
        perf_combo.bind("<<ComboboxSelected>>", self.on_performance_mode_change)

        ttk.Label(perf_card, text="Cycle Delay (seconds):", style="MetricTitle.TLabel").grid(row=0, column=2, sticky=tk.W)
        self.delay_var = tk.DoubleVar(value=self.cycle_delay)
        self.delay_scale = ttk.Scale(
            perf_card,
            from_=0.5,
            to=10.0,
            variable=self.delay_var,
            orient=tk.HORIZONTAL,
            length=220,
            command=self.on_delay_change,
        )
        self.delay_scale.grid(row=0, column=3, sticky="ew", padx=(8, 4))
        self.delay_label = ttk.Label(perf_card, text=f"{self.cycle_delay:.1f}s", style="MetricValue.TLabel")
        self.delay_label.grid(row=0, column=4, sticky=tk.W)

        self.auto_save_var = tk.BooleanVar(value=self.auto_save_enabled)
        auto_save_check = ttk.Checkbutton(
            perf_card,
            text="Auto-Save Enabled",
            variable=self.auto_save_var,
            command=self.on_auto_save_toggle,
            style="Card.TCheckbutton",
        )
        auto_save_check.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))

        self.vector_combine_var = tk.BooleanVar(value=self.vector_combination_enabled)
        vector_combine_check = ttk.Checkbutton(
            perf_card,
            text="Vector Combination",
            variable=self.vector_combine_var,
            command=self.on_vector_combine_toggle,
            style="Card.TCheckbutton",
        )
        vector_combine_check.grid(row=1, column=2, columnspan=2, sticky=tk.W, pady=(10, 0))

        metrics_frame = ttk.Frame(perf_card, style="Card.TFrame")
        metrics_frame.grid(row=2, column=0, columnspan=5, sticky="ew", pady=(12, 0))
        metrics_frame.columnconfigure(1, weight=1)

        ttk.Label(metrics_frame, text="CPU Usage", style="MetricTitle.TLabel").grid(row=0, column=0, sticky=tk.W)
        self.cpu_progress = ttk.Progressbar(metrics_frame, style="Card.Horizontal.TProgressbar", maximum=100)
        self.cpu_progress.grid(row=0, column=1, sticky="ew", padx=(8, 12))
        self.cpu_value_label = ttk.Label(metrics_frame, text="0%", style="MetricValue.TLabel")
        self.cpu_value_label.grid(row=0, column=2, sticky=tk.E)

        ttk.Label(metrics_frame, text="Memory Usage", style="MetricTitle.TLabel").grid(row=1, column=0, sticky=tk.W, pady=(6, 0))
        self.memory_progress = ttk.Progressbar(metrics_frame, style="Card.Horizontal.TProgressbar", maximum=100)
        self.memory_progress.grid(row=1, column=1, sticky="ew", padx=(8, 12), pady=(6, 0))
        self.memory_value_label = ttk.Label(metrics_frame, text="0%", style="MetricValue.TLabel")
        self.memory_value_label.grid(row=1, column=2, sticky=tk.E, pady=(6, 0))

        status_section = ttk.Frame(dashboard_frame, style="Card.TFrame")
        status_section.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        status_section.columnconfigure(0, weight=1)
        status_section.columnconfigure(1, weight=1)

        summary_card = ttk.LabelFrame(status_section, text="Current Cycle Summary", padding=12, style="Card.TLabelframe")
        summary_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        summary_card.columnconfigure(0, weight=1)

        indicator_frame = ttk.Frame(summary_card, style="Card.TFrame")
        indicator_frame.grid(row=0, column=0, sticky="ew")
        indicator_frame.columnconfigure(1, weight=1)

        self.state_indicator = ttk.Label(indicator_frame, text="●", style="StateIndicator.TLabel")
        self.state_indicator.grid(row=0, column=0, sticky=tk.W)

        self.runtime_status_label = ttk.Label(indicator_frame, text="Stopped", style="MetricValue.TLabel")
        self.runtime_status_label.grid(row=0, column=1, sticky=tk.W, padx=(8, 0))
        self._update_state_indicator_color("stopped")

        summary_body = ttk.Frame(summary_card, style="Card.TFrame")
        summary_body.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        summary_body.columnconfigure(1, weight=1)

        self._create_metric_row(summary_body, "State", "state", 0, initial="Initializing")
        self._create_metric_row(summary_body, "Cycle", "cycle", 1, initial="0")
        self._create_metric_row(summary_body, "Decision", "decision", 2, initial="—")

        log_card = ttk.LabelFrame(status_section, text="Activity Log", padding=0, style="Card.TLabelframe")
        log_card.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        log_card.columnconfigure(0, weight=1)
        log_card.rowconfigure(0, weight=1)

        self.status_text = scrolledtext.ScrolledText(
            log_card,
            height=16,
            wrap=tk.WORD,
            bg=self.colors["surface"],
            fg=self.colors["text"],
            font=("Consolas", 10),
            borderwidth=0,
            highlightthickness=0,
        )
        self.status_text.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.status_text.configure(insertbackground=self.colors["accent"], state=tk.DISABLED)
        self.status_text.bind("<Key>", lambda event: "break")
        
    def create_emotions_tab(self):
        """Create the emotions visualization tab."""
        emotions_frame = ttk.Frame(self.notebook)
        self.notebook.add(emotions_frame, text="Emotions & Creativity")
        
        # Real-time emotion bars
        bars_frame = ttk.LabelFrame(emotions_frame, text="Current Emotion Levels", padding=10, style="Card.TLabelframe")
        bars_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.emotion_bars = {}
        self.emotion_labels = {}
        
        self.emotion_container = bars_frame
        for emotion in sorted(self.him_model.get_system_status()['sector_one_emotions'].keys()):
            self._add_emotion_widget(emotion)
        
        # Emotion history plot
        plot_frame = ttk.LabelFrame(emotions_frame, text="Emotion History", padding=10, style="Card.TLabelframe")
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.emotion_fig = Figure(figsize=(8, 4), facecolor=self.colors["surface"])
        self.emotion_ax = self.emotion_fig.add_subplot(111, facecolor=self.colors["surface"])
        self.emotion_ax.set_xlabel('Time', color='white')
        self.emotion_ax.set_ylabel('Emotion Level', color='white')
        self.emotion_ax.tick_params(colors='white')
        
        self.emotion_canvas = FigureCanvasTkAgg(self.emotion_fig, plot_frame)
        self.emotion_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def create_hormones_tab(self):
        """Create the hormones visualization tab."""
        hormones_frame = ttk.Frame(self.notebook)
        self.notebook.add(hormones_frame, text="Hormones & Drives")

        # Hormone levels
        hormone_frame = ttk.LabelFrame(hormones_frame, text="Hormone Levels", padding=10, style="Card.TLabelframe")
        hormone_frame.pack(fill=tk.X, padx=10, pady=5)

        self.hormone_bars = {}
        self.hormone_labels = {}

        self.hormone_container = hormone_frame
        for hormone in sorted(self.him_model.get_system_status()['sector_four_hormones'].keys()):
            self._add_hormone_widget(hormone)

        # Drive levels
        drive_frame = ttk.LabelFrame(hormones_frame, text="Drive Levels", padding=10, style="Card.TLabelframe")
        drive_frame.pack(fill=tk.X, padx=10, pady=5)

        self.drive_bars = {}
        self.drive_labels = {}

        self.drive_container = drive_frame
        for drive in sorted(self.him_model.get_system_status()['sector_four_drives'].keys()):
            self._add_drive_widget(drive)

        # Hormone history plot
        plot_frame = ttk.LabelFrame(hormones_frame, text="Hormone History", padding=10, style="Card.TLabelframe")
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.hormone_fig = Figure(figsize=(8, 4), facecolor='#2d2d2d')
        self.hormone_ax = self.hormone_fig.add_subplot(111, facecolor='#2d2d2d')
        self.hormone_ax.set_xlabel('Time', color='white')
        self.hormone_ax.set_ylabel('Hormone Level', color='white')
        self.hormone_ax.tick_params(colors='white')

        self.hormone_canvas = FigureCanvasTkAgg(self.hormone_fig, plot_frame)
        self.hormone_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_persona_tab(self):
        """Create the persona management and observation tab."""

        persona_content = self._add_scrollable_tab("Persona Presence")

        paned = ttk.Panedwindow(persona_content, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        display_container = ttk.Frame(paned, padding=(0, 0, 6, 0))
        editor_container = ttk.Frame(paned, padding=(6, 0, 0, 0))
        paned.add(display_container, weight=3)
        paned.add(editor_container, weight=2)

        display_frame = ttk.LabelFrame(display_container, text="Uploaded Consciousness", padding=10)
        display_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(display_frame, textvariable=self.persona_name_var, font=('Segoe UI', 16, 'bold')).pack(anchor=tk.W)
        ttk.Label(display_frame, textvariable=self.persona_mood_var, font=('Segoe UI', 11)).pack(
            anchor=tk.W, pady=(0, 4)
        )
        ttk.Label(display_frame, textvariable=self.persona_voice_var, font=('Segoe UI', 10)).pack(anchor=tk.W)
        ttk.Label(display_frame, textvariable=self.persona_conversation_var, font=('Segoe UI', 10)).pack(
            anchor=tk.W, pady=(0, 8)
        )

        ttk.Label(display_frame, text="Identity Statement", font=('Segoe UI', 10, 'bold')).pack(anchor=tk.W)
        self.persona_identity_display = self._create_readonly_text(display_frame, height=4)
        self.persona_identity_display.pack(fill=tk.BOTH, expand=False, pady=(0, 6))

        ttk.Label(display_frame, text="Biography", font=('Segoe UI', 10, 'bold')).pack(anchor=tk.W)
        self.persona_biography_display = self._create_readonly_text(display_frame, height=5)
        self.persona_biography_display.pack(fill=tk.BOTH, expand=False, pady=(0, 6))

        ttk.Label(display_frame, text="Upload Story", font=('Segoe UI', 10, 'bold')).pack(anchor=tk.W)
        self.persona_upload_display = self._create_readonly_text(display_frame, height=4)
        self.persona_upload_display.pack(fill=tk.BOTH, expand=False, pady=(0, 6))

        catalogue_frame = ttk.Frame(display_frame)
        catalogue_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        for idx in range(3):
            catalogue_frame.columnconfigure(idx, weight=1)

        values_box = ttk.LabelFrame(catalogue_frame, text="Core Values", padding=5)
        values_box.grid(row=0, column=0, sticky='nsew', padx=4)
        self.persona_values_listbox = tk.Listbox(values_box, height=6)
        self.persona_values_listbox.pack(fill=tk.BOTH, expand=True)

        goals_box = ttk.LabelFrame(catalogue_frame, text="Lifelong Goals", padding=5)
        goals_box.grid(row=0, column=1, sticky='nsew', padx=4)
        self.persona_goals_listbox = tk.Listbox(goals_box, height=6)
        self.persona_goals_listbox.pack(fill=tk.BOTH, expand=True)

        skills_box = ttk.LabelFrame(catalogue_frame, text="Signature Skills", padding=5)
        skills_box.grid(row=0, column=2, sticky='nsew', padx=4)
        self.persona_skills_listbox = tk.Listbox(skills_box, height=6)
        self.persona_skills_listbox.pack(fill=tk.BOTH, expand=True)

        inner_frame = ttk.LabelFrame(display_frame, text="Inner Monologue", padding=5)
        inner_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.persona_inner_thought_display = self._create_readonly_text(inner_frame, height=4)
        self.persona_inner_thought_display.pack(fill=tk.BOTH, expand=True)

        memories_frame = ttk.LabelFrame(display_frame, text="Recent Autobiographical Memories", padding=5)
        memories_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.persona_memory_display = self._create_readonly_text(memories_frame, height=6)
        self.persona_memory_display.pack(fill=tk.BOTH, expand=True)

        narrative_frame = ttk.LabelFrame(display_frame, text="Narrative Log", padding=5)
        narrative_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.persona_narrative_display = self._create_readonly_text(narrative_frame, height=6)
        self.persona_narrative_display.pack(fill=tk.BOTH, expand=True)

        editor_frame = ttk.LabelFrame(editor_container, text="Persona Controls", padding=10)
        editor_frame.pack(fill=tk.BOTH, expand=True)
        editor_frame.columnconfigure(1, weight=1)

        ttk.Label(editor_frame, text="Name").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(editor_frame, textvariable=self.persona_edit_name_var).grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)

        ttk.Label(editor_frame, text="Conversation Style").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(editor_frame, textvariable=self.persona_edit_conversation_var).grid(
            row=1, column=1, sticky=tk.EW, padx=5, pady=2
        )

        ttk.Label(editor_frame, text="Voice Description").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(editor_frame, textvariable=self.persona_edit_voice_var).grid(
            row=2, column=1, sticky=tk.EW, padx=5, pady=2
        )

        ttk.Label(editor_frame, text="Identity Statement").grid(row=3, column=0, sticky=tk.NW, padx=5, pady=2)
        self.persona_identity_edit = scrolledtext.ScrolledText(editor_frame, height=4, wrap=tk.WORD)
        self.persona_identity_edit.grid(row=3, column=1, sticky=tk.EW, padx=5, pady=2)

        ttk.Label(editor_frame, text="Biography").grid(row=4, column=0, sticky=tk.NW, padx=5, pady=2)
        self.persona_biography_edit = scrolledtext.ScrolledText(editor_frame, height=5, wrap=tk.WORD)
        self.persona_biography_edit.grid(row=4, column=1, sticky=tk.EW, padx=5, pady=2)

        ttk.Label(editor_frame, text="Upload Story").grid(row=5, column=0, sticky=tk.NW, padx=5, pady=2)
        self.persona_upload_edit = scrolledtext.ScrolledText(editor_frame, height=4, wrap=tk.WORD)
        self.persona_upload_edit.grid(row=5, column=1, sticky=tk.EW, padx=5, pady=2)

        button_frame = ttk.Frame(editor_frame)
        button_frame.grid(row=6, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=5)
        for idx in range(3):
            button_frame.columnconfigure(idx, weight=1)

        ttk.Button(button_frame, text="Apply Persona Updates", command=self.apply_persona_updates).grid(
            row=0, column=0, sticky=tk.EW, padx=4
        )
        ttk.Button(
            button_frame, text="Sync Editor from Snapshot", command=self.sync_persona_editor_from_snapshot
        ).grid(row=0, column=1, sticky=tk.EW, padx=4)
        ttk.Button(button_frame, text="Refresh Persona", command=self.refresh_persona_from_model).grid(
            row=0, column=2, sticky=tk.EW, padx=4
        )

        button_frame2 = ttk.Frame(editor_frame)
        button_frame2.grid(row=7, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=5)
        button_frame2.columnconfigure(0, weight=1)
        button_frame2.columnconfigure(1, weight=1)

        ttk.Button(button_frame2, text="Save Profile…", command=self.save_persona_profile).grid(
            row=0, column=0, sticky=tk.EW, padx=4
        )
        ttk.Button(button_frame2, text="Load Profile…", command=self.load_persona_profile).grid(
            row=0, column=1, sticky=tk.EW, padx=4
        )

        memory_frame = ttk.LabelFrame(editor_frame, text="Log Experience", padding=8)
        memory_frame.grid(row=8, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=5)
        memory_frame.columnconfigure(1, weight=1)

        ttk.Label(memory_frame, text="Summary").grid(row=0, column=0, sticky=tk.W, padx=4, pady=2)
        ttk.Entry(memory_frame, textvariable=self.persona_memory_summary_var).grid(
            row=0, column=1, sticky=tk.EW, padx=4, pady=2
        )

        ttk.Label(memory_frame, text="Emotion").grid(row=1, column=0, sticky=tk.W, padx=4, pady=2)
        self.persona_emotion_menu = ttk.Combobox(
            memory_frame,
            textvariable=self.persona_memory_emotion_var,
            values=sorted(self.him_model.get_system_status()['sector_one_emotions'].keys()),
            state='readonly',
        )
        self.persona_emotion_menu.grid(row=1, column=1, sticky=tk.EW, padx=4, pady=2)
        if not self.persona_memory_emotion_var.get() and self.persona_emotion_menu['values']:
            self.persona_memory_emotion_var.set(self.persona_emotion_menu['values'][0])

        ttk.Label(memory_frame, text="Intensity").grid(row=2, column=0, sticky=tk.W, padx=4, pady=2)
        intensity_scale = ttk.Scale(
            memory_frame,
            from_=0.0,
            to=1.0,
            variable=self.persona_memory_intensity_var,
            orient=tk.HORIZONTAL,
            command=self.on_persona_intensity_change,
        )
        intensity_scale.grid(row=2, column=1, sticky=tk.EW, padx=4, pady=2)
        self.persona_intensity_label = ttk.Label(
            memory_frame, text=f"Intensity: {self.persona_memory_intensity_var.get():.2f}"
        )
        self.persona_intensity_label.grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=4, pady=2)

        ttk.Button(memory_frame, text="Log Memory", command=self.log_persona_memory).grid(
            row=4, column=0, columnspan=2, sticky=tk.EW, padx=4, pady=4
        )

    def create_machine_learning_tab(self):
        """Create controls for the async machine learning trainer."""
        ml_frame = self._add_scrollable_tab("Machine Learning")

        session_frame = ttk.LabelFrame(ml_frame, text="Async Trainer Control", padding=10)
        session_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(session_frame, text="Learning Rate:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(session_frame, textvariable=self.training_learning_rate, width=8).grid(
            row=0, column=1, sticky=tk.W, padx=5, pady=2
        )
        ttk.Label(session_frame, text="L2 Penalty:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(session_frame, textvariable=self.training_l2_var, width=8).grid(
            row=0, column=3, sticky=tk.W, padx=5, pady=2
        )

        self.training_start_btn = ttk.Button(session_frame, text="Start Trainer", command=self.start_async_trainer)
        self.training_start_btn.grid(row=0, column=4, padx=5, pady=2)
        self.training_stop_btn = ttk.Button(
            session_frame, text="Stop Trainer", command=self.stop_async_trainer, state=tk.DISABLED
        )
        self.training_stop_btn.grid(row=0, column=5, padx=5, pady=2)

        ttk.Button(
            session_frame,
            text="Queue Sample Dataset",
            command=self.queue_sample_dataset,
        ).grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)

        status_frame = ttk.LabelFrame(ml_frame, text="Trainer Status", padding=10)
        status_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(status_frame, textvariable=self.training_status_var).pack(anchor=tk.W)
        ttk.Label(status_frame, textvariable=self.training_updates_var).pack(anchor=tk.W)
        ttk.Label(status_frame, textvariable=self.training_queue_var).pack(anchor=tk.W)
        ttk.Label(status_frame, textvariable=self.training_loss_var).pack(anchor=tk.W)

        cycle_frame = ttk.LabelFrame(ml_frame, text="Recent Cycle Summaries", padding=10)
        cycle_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        list_container = ttk.Frame(cycle_frame)
        list_container.pack(fill=tk.BOTH, expand=True)

        self.cycle_summary_listbox = tk.Listbox(list_container, height=8)
        self.cycle_summary_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(list_container, orient=tk.VERTICAL, command=self.cycle_summary_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.cycle_summary_listbox.config(yscrollcommand=scrollbar.set)
        self.cycle_summary_listbox.bind('<<ListboxSelect>>', self.on_cycle_summary_select)

        button_frame = ttk.Frame(cycle_frame)
        button_frame.pack(fill=tk.X, pady=5)

        ttk.Button(
            button_frame,
            text="Queue Selected as Positive",
            command=lambda: self.queue_selected_cycle_example(1),
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            button_frame,
            text="Queue Selected as Negative",
            command=lambda: self.queue_selected_cycle_example(0),
        ).pack(side=tk.LEFT, padx=5)

        manual_frame = ttk.LabelFrame(ml_frame, text="Manual Training Example", padding=10)
        manual_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.manual_training_text = scrolledtext.ScrolledText(manual_frame, height=6, width=60,
                                                             bg='#2d2d2d', fg='#ffffff', font=('Consolas', 10))
        self.manual_training_text.pack(fill=tk.BOTH, expand=True)

        label_frame = ttk.Frame(manual_frame)
        label_frame.pack(fill=tk.X, pady=5)
        ttk.Label(label_frame, text="Label:").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(label_frame, text="Positive", variable=self.manual_label_var, value="positive").pack(side=tk.LEFT)
        ttk.Radiobutton(label_frame, text="Negative", variable=self.manual_label_var, value="negative").pack(side=tk.LEFT)
        ttk.Button(label_frame, text="Queue Manual Example", command=self.queue_manual_training_example).pack(side=tk.RIGHT)

        prediction_frame = ttk.LabelFrame(ml_frame, text="Quick Prediction", padding=10)
        prediction_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.prediction_text = scrolledtext.ScrolledText(prediction_frame, height=4, width=60,
                                                        bg='#2d2d2d', fg='#ffffff', font=('Consolas', 10))
        self.prediction_text.pack(fill=tk.BOTH, expand=True)

        pred_button_row = ttk.Frame(prediction_frame)
        pred_button_row.pack(fill=tk.X, pady=5)
        ttk.Button(pred_button_row, text="Predict", command=self.run_trainer_prediction).pack(side=tk.LEFT, padx=5)
        ttk.Label(pred_button_row, textvariable=self.prediction_result_var).pack(side=tk.LEFT, padx=10)

        log_frame = ttk.LabelFrame(ml_frame, text="Training Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.training_log_text = scrolledtext.ScrolledText(log_frame, height=8, bg='#2d2d2d', fg='#ffffff',
                                                          font=('Consolas', 10))
        self.training_log_text.pack(fill=tk.BOTH, expand=True)
        self.training_log_text.insert(tk.END, "Training log ready. Start the trainer to stream updates.\n")
        
    def create_memory_tab(self):
        """Create the memory and dreams tab."""
        memory_frame = ttk.Frame(self.notebook)
        self.notebook.add(memory_frame, text="Memory & Dreams")
        
        # Memory display
        memory_display_frame = ttk.LabelFrame(memory_frame, text="Stored Memories", padding=10, style="Card.TLabelframe")
        memory_display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.memory_text = scrolledtext.ScrolledText(
            memory_display_frame,
            height=15,
            bg=self.colors["surface"],
            fg=self.colors["text"],
            font=("Consolas", 10),
            borderwidth=0,
            highlightthickness=0,
        )
        self.memory_text.pack(fill=tk.BOTH, expand=True)
        self.memory_text.configure(insertbackground=self.colors["accent"])
        
        # Memory controls
        memory_control_frame = ttk.Frame(memory_frame)
        memory_control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(memory_control_frame, text="Refresh Memories", 
                  command=self.refresh_memories).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(memory_control_frame, text="Generate Dream", 
                  command=self.generate_dream).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(memory_control_frame, text="Clear Memories", 
                  command=self.clear_memories).pack(side=tk.LEFT, padx=5)
        
        # Dreams display
        dreams_frame = ttk.LabelFrame(memory_frame, text="Dream Fragments", padding=10, style="Card.TLabelframe")
        dreams_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.dreams_text = scrolledtext.ScrolledText(
            dreams_frame,
            height=8,
            bg=self.colors["surface"],
            fg=self.colors["text"],
            font=("Consolas", 10),
            borderwidth=0,
            highlightthickness=0,
        )
        self.dreams_text.pack(fill=tk.BOTH, expand=True)
        self.dreams_text.configure(insertbackground=self.colors["accent"])
        
    def create_chat_tab(self):
        """Create the chat interface tab."""
        chat_frame = ttk.Frame(self.notebook)
        self.notebook.add(chat_frame, text="Chat Interface")

        # Chat display
        chat_display_frame = ttk.LabelFrame(chat_frame, text="Chat with H.I.M. Model", padding=10, style="Card.TLabelframe")
        chat_display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.chat_text = scrolledtext.ScrolledText(
            chat_display_frame,
            height=20,
            bg=self.colors["surface"],
            fg=self.colors["text"],
            font=("Consolas", 11),
            borderwidth=0,
            highlightthickness=0,
        )
        self.chat_text.pack(fill=tk.BOTH, expand=True)
        self.chat_text.configure(insertbackground=self.colors["accent"])
        
        # Chat input
        input_frame = ttk.Frame(chat_frame)
        input_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.chat_entry = ttk.Entry(input_frame, font=('Arial', 12))
        self.chat_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.chat_entry.bind('<Return>', self.send_chat_message)
        
        ttk.Button(input_frame, text="Send", command=self.send_chat_message).pack(side=tk.RIGHT)
        
        # Add welcome message
        self.chat_text.insert(tk.END, "H.I.M. Model: Hello! I'm the Hierarchical Image Model. ")
        self.chat_text.insert(tk.END, "I can process visual input, manage emotions, store memories, ")
        self.chat_text.insert(tk.END, "and generate dreams. How can I help you today?\n\n")
        self.chat_text.see(tk.END)

    def create_remote_control_tab(self):
        """Create the remote control and automation tab."""
        remote_frame = self._add_scrollable_tab("Remote Control")

        status_frame = ttk.LabelFrame(remote_frame, text="Session Controls", padding=10)
        status_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(status_frame, textvariable=self.remote_status_var).pack(side=tk.LEFT)

        self.remote_start_btn = ttk.Button(status_frame, text="Start Remote Session",
                                           command=self.enable_remote_control)
        self.remote_start_btn.pack(side=tk.LEFT, padx=5)

        self.remote_stop_btn = ttk.Button(status_frame, text="Stop Remote Session",
                                          command=self.disable_remote_control, state=tk.DISABLED)
        self.remote_stop_btn.pack(side=tk.LEFT, padx=5)

        ttk.Checkbutton(status_frame, text="Feed live screen into cycles",
                        variable=self.use_live_screen_var).pack(side=tk.LEFT, padx=10)

        refresh_frame = ttk.Frame(status_frame)
        refresh_frame.pack(side=tk.RIGHT)
        ttk.Label(refresh_frame, text="Refresh (s)").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Scale(refresh_frame, from_=0.2, to=2.0, variable=self.remote_refresh_interval,
                  orient=tk.HORIZONTAL, length=120).pack(side=tk.LEFT)

        preview_frame = ttk.LabelFrame(remote_frame, text="Screen Preview", padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.remote_preview_label = ttk.Label(preview_frame, text="Remote control inactive",
                                              anchor=tk.CENTER)
        self.remote_preview_label.pack(fill=tk.BOTH, expand=True)

        control_columns = ttk.Frame(remote_frame)
        control_columns.pack(fill=tk.BOTH, expand=False, padx=10, pady=5)

        mouse_frame = ttk.LabelFrame(control_columns, text="Mouse", padding=10)
        mouse_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        coords_frame = ttk.Frame(mouse_frame)
        coords_frame.pack(fill=tk.X, pady=2)
        ttk.Label(coords_frame, text="X:").pack(side=tk.LEFT)
        ttk.Entry(coords_frame, textvariable=self.mouse_x_var, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(coords_frame, text="Y:").pack(side=tk.LEFT)
        ttk.Entry(coords_frame, textvariable=self.mouse_y_var, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(coords_frame, text="Move", command=self.perform_mouse_move).pack(side=tk.LEFT, padx=5)

        click_frame = ttk.Frame(mouse_frame)
        click_frame.pack(fill=tk.X, pady=2)
        ttk.Label(click_frame, text="Button:").pack(side=tk.LEFT)
        click_combo = ttk.Combobox(click_frame, textvariable=self.mouse_button_var,
                                   values=["left", "right", "middle"], width=8, state="readonly")
        click_combo.pack(side=tk.LEFT, padx=2)
        ttk.Button(click_frame, text="Click", command=self.perform_mouse_click).pack(side=tk.LEFT, padx=5)

        scroll_frame = ttk.Frame(mouse_frame)
        scroll_frame.pack(fill=tk.X, pady=2)
        ttk.Label(scroll_frame, text="Scroll:").pack(side=tk.LEFT)
        ttk.Entry(scroll_frame, textvariable=self.scroll_amount_var, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(scroll_frame, text="Apply", command=self.perform_mouse_scroll).pack(side=tk.LEFT, padx=5)

        keyboard_frame = ttk.LabelFrame(control_columns, text="Keyboard", padding=10)
        keyboard_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        text_frame = ttk.Frame(keyboard_frame)
        text_frame.pack(fill=tk.X, pady=2)
        ttk.Label(text_frame, text="Type text:").pack(side=tk.LEFT)
        ttk.Entry(text_frame, textvariable=self.keyboard_text_var, width=30).pack(side=tk.LEFT, padx=2)
        ttk.Button(text_frame, text="Send", command=self.perform_type_text).pack(side=tk.LEFT, padx=5)

        keypress_frame = ttk.Frame(keyboard_frame)
        keypress_frame.pack(fill=tk.X, pady=2)
        ttk.Label(keypress_frame, text="Press key:").pack(side=tk.LEFT)
        ttk.Entry(keypress_frame, textvariable=self.keypress_var, width=15).pack(side=tk.LEFT, padx=2)
        ttk.Button(keypress_frame, text="Press", command=self.perform_key_press).pack(side=tk.LEFT, padx=5)

        info_frame = ttk.LabelFrame(remote_frame, text="Remote Action Log", padding=10)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.remote_log_text = scrolledtext.ScrolledText(info_frame, height=8, state=tk.DISABLED,
                                                         bg='#2d2d2d', fg='#ffffff', font=('Consolas', 10))
        self.remote_log_text.pack(fill=tk.BOTH, expand=True)

        self.update_remote_status()

    def create_visual_tab(self):
        """Create the visual processing tab."""
        visual_frame = ttk.Frame(self.notebook)
        self.notebook.add(visual_frame, text="Visual Processing")

        # Image display
        image_frame = ttk.LabelFrame(visual_frame, text="Current Visual Input", padding=10, style="Card.TLabelframe")
        image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.image_label = ttk.Label(image_frame, text="No image loaded", 
                                   font=('Arial', 14), anchor=tk.CENTER)
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Visual processing info
        info_frame = ttk.LabelFrame(visual_frame, text="Visual Processing Information", padding=10, style="Card.TLabelframe")
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.visual_info_text = tk.Text(
            info_frame,
            height=6,
            bg=self.colors["surface"],
            fg=self.colors["text"],
            font=("Consolas", 10),
            wrap=tk.WORD,
            borderwidth=0,
            highlightthickness=0,
        )
        self.visual_info_text.pack(fill=tk.BOTH, expand=True)
        self.visual_info_text.configure(insertbackground=self.colors["accent"])
        
        # Zipline data display
        zipline_frame = ttk.LabelFrame(visual_frame, text="Vectorized Thought Data (Zipline Patterns)", padding=10, style="Card.TLabelframe")
        zipline_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Zipline image display
        self.zipline_label = ttk.Label(zipline_frame, text="No vectorized data available", 
                                     font=('Arial', 12), anchor=tk.CENTER)
        self.zipline_label.pack(fill=tk.BOTH, expand=True)
        
        # Zipline controls
        zipline_control_frame = ttk.Frame(zipline_frame)
        zipline_control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(zipline_control_frame, text="Refresh Zipline", 
                  command=self.refresh_zipline_display).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(zipline_control_frame, text="Save Current Zipline",
                  command=self.save_current_zipline).pack(side=tk.LEFT, padx=5)

    # ------------------------------------------------------------------
    # Persona helpers
    # ------------------------------------------------------------------
    def _create_readonly_text(self, parent, height=4, width=52):
        widget = scrolledtext.ScrolledText(
            parent,
            height=height,
            width=width,
            wrap=tk.WORD,
            bg='#1e1e1e',
            fg='#f5f5f5',
            insertbackground='#f5f5f5',
            font=('Segoe UI', 10),
        )
        widget.configure(state='disabled')
        return widget

    def _set_text_widget(self, widget, text, readonly=True):
        if widget is None:
            return
        widget.configure(state='normal')
        widget.delete('1.0', tk.END)
        cleaned = text.strip() if isinstance(text, str) else ''
        widget.insert(tk.END, cleaned)
        if readonly:
            widget.configure(state='disabled')

    def _get_text_widget_content(self, widget) -> str:
        if widget is None:
            return ""
        return widget.get('1.0', tk.END).strip()

    def _populate_listbox(self, listbox, values):
        if listbox is None:
            return
        listbox.delete(0, tk.END)
        for item in values or []:
            listbox.insert(tk.END, item)

    def on_persona_intensity_change(self, value):
        if self.persona_intensity_label is not None:
            self.persona_intensity_label.config(text=f"Intensity: {float(value):.2f}")

    # ------------------------------------------------------------------
    # Remote control management
    # ------------------------------------------------------------------
    def update_remote_status(self):
        """Update remote status text and button availability."""
        if self.pyautogui is None:
            self.remote_status_var.set("Install 'pyautogui' to enable remote control features")
            self.remote_start_btn.config(state=tk.DISABLED)
            self.remote_stop_btn.config(state=tk.DISABLED)
            return

        self.remote_status_var.set(
            "Remote control ready (press ESC for emergency stop)"
            if not self.remote_control_enabled else "Remote session active"
        )
        self.remote_start_btn.config(state=tk.DISABLED if self.remote_control_enabled else tk.NORMAL)
        self.remote_stop_btn.config(state=tk.NORMAL if self.remote_control_enabled else tk.DISABLED)

    def enable_remote_control(self):
        """Begin the remote session and start streaming the desktop."""
        if self.pyautogui is None or self.remote_control_enabled:
            return

        self.remote_control_enabled = True
        self.remote_stop_event.clear()
        self.remote_status_var.set("Remote session active")
        self.remote_preview_label.config(text="Connecting to desktop…")
        self.remote_has_logged_frame = False
        self.remote_capture_thread = threading.Thread(target=self.remote_capture_loop, daemon=True)
        self.remote_capture_thread.start()
        self.root.after(100, self.process_remote_frames)
        self.log_remote_action("Remote control enabled. Press ESC to halt immediately.")
        self.update_remote_status()

    def disable_remote_control(self):
        """Terminate the remote session and stop streaming."""
        if not self.remote_control_enabled:
            return

        self.remote_control_enabled = False
        self.remote_stop_event.set()
        if self.remote_capture_thread and self.remote_capture_thread.is_alive():
            self.remote_capture_thread.join(timeout=1.0)
        self.remote_capture_thread = None
        self.remote_status_var.set("Remote control paused")
        self.remote_preview_label.config(image='', text="Remote control inactive")
        self.remote_preview_label.image = None
        self.latest_screen_frame = None
        self.remote_has_logged_frame = False
        self.update_remote_status()
        self.log_remote_action("Remote control disabled.")

    def remote_capture_loop(self):
        """Capture the local desktop on an interval while remote control is active."""
        target_size = (640, 480)
        preview_size = (480, 270)
        while not self.remote_stop_event.is_set():
            try:
                screenshot = self.pyautogui.screenshot()
            except Exception as exc:  # pragma: no cover - depends on host OS
                self.log_remote_action(f"Screen capture failed: {exc}")
                self.remote_stop_event.wait(1.0)
                continue

            model_image = screenshot.resize(target_size)
            preview_image = screenshot.resize(preview_size)

            try:
                self.remote_frame_queue.put_nowait((model_image, preview_image))
            except queue.Full:
                pass

            delay = max(self.remote_refresh_interval.get(), 0.1)
            self.remote_stop_event.wait(delay)

    def process_remote_frames(self):
        """Consume the screen capture queue and update the UI safely."""
        if not self.remote_control_enabled:
            return

        updated = False
        try:
            while True:
                model_image, preview_image = self.remote_frame_queue.get_nowait()
                model_array = np.asarray(model_image, dtype=np.float32) / 255.0
                self.latest_screen_frame = model_array
                self.remote_preview_photo = ImageTk.PhotoImage(preview_image)
                self.remote_preview_label.config(image=self.remote_preview_photo, text='')
                self.remote_preview_label.image = self.remote_preview_photo
                updated = True
        except queue.Empty:
            pass

        if updated and not self.remote_has_logged_frame:
            self.remote_has_logged_frame = True
            self.log_remote_action("Screen streaming active.")

        if self.remote_control_enabled:
            self.root.after(int(self.remote_refresh_interval.get() * 1000), self.process_remote_frames)

    def log_remote_action(self, message):
        """Append a message to the remote action log."""
        if not hasattr(self, 'remote_log_text'):
            return
        timestamp = datetime.now().strftime('%H:%M:%S')

        def _append():
            self.remote_log_text.config(state=tk.NORMAL)
            self.remote_log_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.remote_log_text.see(tk.END)
            self.remote_log_text.config(state=tk.DISABLED)

        self.root.after(0, _append)

    def is_remote_ready(self):
        return self.remote_control_enabled and self.pyautogui is not None

    def perform_mouse_move(self):
        """Move the mouse cursor to the requested coordinates."""
        if not self.is_remote_ready():
            self.log_remote_action("Mouse move ignored: remote session inactive.")
            return

        try:
            x = int(self.mouse_x_var.get())
            y = int(self.mouse_y_var.get())
            self.pyautogui.moveTo(x, y, duration=0.2)
            self.log_remote_action(f"Mouse moved to ({x}, {y}).")
        except ValueError:
            self.log_remote_action("Invalid mouse coordinates.")

    def perform_mouse_click(self):
        """Click using the selected mouse button."""
        if not self.is_remote_ready():
            self.log_remote_action("Mouse click ignored: remote session inactive.")
            return

        button = self.mouse_button_var.get()
        try:
            self.pyautogui.click(button=button)
            self.log_remote_action(f"Mouse {button} click executed.")
        except Exception as exc:  # pragma: no cover - host specific
            self.log_remote_action(f"Mouse click failed: {exc}")

    def perform_mouse_scroll(self):
        """Scroll vertically by the requested amount."""
        if not self.is_remote_ready():
            self.log_remote_action("Scroll ignored: remote session inactive.")
            return

        try:
            amount = int(self.scroll_amount_var.get())
            if amount != 0:
                self.pyautogui.scroll(amount)
                self.log_remote_action(f"Scrolled {amount} units.")
            else:
                self.log_remote_action("Scroll skipped: amount is zero.")
        except ValueError:
            self.log_remote_action("Invalid scroll amount.")

    def perform_type_text(self):
        """Type free-form text through the keyboard."""
        if not self.is_remote_ready():
            self.log_remote_action("Typing ignored: remote session inactive.")
            return

        text = self.keyboard_text_var.get()
        if not text:
            self.log_remote_action("Nothing to type.")
            return

        self.pyautogui.typewrite(text)
        self.log_remote_action(f"Typed text: {text}")
        self.keyboard_text_var.set('')

    def perform_key_press(self):
        """Press a single key or key combination."""
        if not self.is_remote_ready():
            self.log_remote_action("Key press ignored: remote session inactive.")
            return

        key_sequence = self.keypress_var.get().strip()
        if not key_sequence:
            self.log_remote_action("No key specified.")
            return

        keys = [key.strip() for key in key_sequence.split('+') if key.strip()]
        try:
            if len(keys) == 1:
                self.pyautogui.press(keys[0])
            else:
                self.pyautogui.hotkey(*keys)
            self.log_remote_action(f"Pressed: {' + '.join(keys)}")
        except Exception as exc:  # pragma: no cover - host specific
            self.log_remote_action(f"Key press failed: {exc}")

    def emergency_stop(self, event=None):
        """Immediate stop handler triggered by the ESC key."""
        self.stop_model()
        self.disable_remote_control()
        self.log_status("Emergency stop activated. All automation halted.")
        self.log_remote_action("Emergency stop activated via ESC.")

    def start_model(self):
        """Start the H.I.M. model processing."""
        self.is_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.runtime_status_label.config(text="Starting…")
        self._update_state_indicator_color("starting")

        # Start processing thread
        self.processing_thread = threading.Thread(target=self.run_model_cycles, daemon=True)
        self.processing_thread.start()

        self.log_status("H.I.M. Model started successfully!")
        
    def stop_model(self):
        """Stop the H.I.M. model processing."""
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.runtime_status_label.config(text="Stopped")
        self._update_state_indicator_color("stopped")

        self.log_status("H.I.M. Model stopped.")
        
    def run_model_cycles(self):
        """Run the model cycles in a separate thread."""
        while self.is_running:
            try:
                # Use live screen feed when available, otherwise fall back to synthetic data
                if self.use_live_screen_var.get() and self.is_remote_ready() and self.latest_screen_frame is not None:
                    test_image = self.latest_screen_frame
                    if self.pyautogui is not None:
                        position = self.pyautogui.position()
                        mouse_pos = {"x": position.x, "y": position.y}
                    else:
                        mouse_pos = {"x": 0, "y": 0}
                else:
                    test_image = np.random.rand(480, 640, 3) * 0.5
                    mouse_pos = {"x": np.random.randint(0, 640), "y": np.random.randint(0, 480)}
                screen_text = f"Cycle {self.cycle_count + 1}: Processing visual input..."
                
                # Run H.I.M. cycle
                result = self.him_model.run_cycle(test_image, mouse_pos, screen_text)
                
                # Update GUI data
                self.message_queue.put(('update_data', result))
                
                self.cycle_count += 1
                time.sleep(self.cycle_delay)  # Configurable delay between cycles
                
                # Auto-save periodically
                if time.time() - self.last_auto_save > self.auto_save_interval:
                    self.auto_save_state()
                    self.last_auto_save = time.time()
                
            except Exception as e:
                self.message_queue.put(('error', str(e)))
                break
                
    def start_update_loop(self):
        """Start the GUI update loop."""
        self.update_gui()
        self.update_performance_monitor()
        self.root.after(100, self.start_update_loop)

    # ------------------------------------------------------------------
    # Persona synchronisation
    # ------------------------------------------------------------------
    def update_persona_panel(self, result):
        persona_state = None
        if isinstance(result, dict):
            persona_state = result.get('persona_state')
        if not persona_state:
            persona_state = self.him_model.get_persona_snapshot()
        if not persona_state:
            return

        self.persona_last_snapshot = persona_state
        profile = persona_state.get('profile', {})

        persona_name = profile.get('name', 'Uploaded Persona')
        self.persona_current_name = persona_name
        self.persona_name_var.set(f"{persona_name} — Digital Presence")
        self.persona_mood_var.set(f"Mood: {persona_state.get('current_mood', '—')}")
        self.persona_voice_var.set(f"Voice: {profile.get('voice_description', '—')}")
        self.persona_conversation_var.set(
            f"Conversation Style: {profile.get('conversation_style', '—')}"
        )

        self._set_text_widget(self.persona_identity_display, profile.get('identity_statement', ''), readonly=True)
        self._set_text_widget(self.persona_biography_display, profile.get('biography', ''), readonly=True)
        self._set_text_widget(self.persona_upload_display, profile.get('upload_story', ''), readonly=True)

        self._populate_listbox(self.persona_values_listbox, profile.get('core_values', []))
        self._populate_listbox(self.persona_goals_listbox, profile.get('lifelong_goals', []))
        self._populate_listbox(self.persona_skills_listbox, profile.get('skills', []))

        inner_thought = persona_state.get('inner_thought', '')
        self._set_text_widget(self.persona_inner_thought_display, inner_thought, readonly=True)

        memories = []
        for memory in persona_state.get('recent_memories', []):
            timestamp = datetime.fromtimestamp(memory['timestamp']).strftime('%H:%M:%S')
            line = (
                f"[{timestamp}] {memory['summary']} — "
                f"{memory['dominant_emotion']} {memory['intensity']:.2f}"
            )
            memories.append(line)
        self._set_text_widget(self.persona_memory_display, "\n".join(memories), readonly=True)

        narrative_log = "\n".join(persona_state.get('narrative_log', []))
        self._set_text_widget(self.persona_narrative_display, narrative_log, readonly=True)

        # Keep emotion combobox up to date with the latest catalogue
        if self.persona_emotion_menu is not None:
            values = sorted(self.him_model.get_system_status()['sector_one_emotions'].keys())
            self.persona_emotion_menu['values'] = values
            if not self.persona_memory_emotion_var.get() and values:
                self.persona_memory_emotion_var.set(values[0])

        if not self.persona_editor_initialized and profile:
            self.populate_persona_editor(profile)

    def populate_persona_editor(self, profile):
        self.persona_edit_name_var.set(profile.get('name', ''))
        self.persona_edit_conversation_var.set(profile.get('conversation_style', ''))
        self.persona_edit_voice_var.set(profile.get('voice_description', ''))

        if self.persona_identity_edit is not None:
            self.persona_identity_edit.delete('1.0', tk.END)
            self.persona_identity_edit.insert(tk.END, profile.get('identity_statement', ''))
        if self.persona_biography_edit is not None:
            self.persona_biography_edit.delete('1.0', tk.END)
            self.persona_biography_edit.insert(tk.END, profile.get('biography', ''))
        if self.persona_upload_edit is not None:
            self.persona_upload_edit.delete('1.0', tk.END)
            self.persona_upload_edit.insert(tk.END, profile.get('upload_story', ''))

        self.persona_editor_initialized = True

    def sync_persona_editor_from_snapshot(self):
        snapshot = self.persona_last_snapshot or self.him_model.get_persona_snapshot()
        if not snapshot:
            self.log_status("Persona snapshot unavailable to sync editor.")
            return
        self.populate_persona_editor(snapshot.get('profile', {}))
        self.log_status("Persona editor synchronised with current snapshot.")

    def refresh_persona_from_model(self):
        snapshot = self.him_model.get_persona_snapshot()
        if snapshot:
            self.update_persona_panel({'persona_state': snapshot})
            self.log_status("Persona snapshot refreshed.")

    def apply_persona_updates(self):
        snapshot = self.persona_last_snapshot or self.him_model.get_persona_snapshot()
        if not snapshot:
            self.log_status("Cannot update persona without an active snapshot.")
            return

        profile = snapshot.get('profile', {})

        def resolved(value, key):
            base = profile.get(key, '')
            return value.strip() if isinstance(value, str) and value.strip() else base

        updates = {
            'name': resolved(self.persona_edit_name_var.get(), 'name'),
            'conversation_style': resolved(self.persona_edit_conversation_var.get(), 'conversation_style'),
            'voice_description': resolved(self.persona_edit_voice_var.get(), 'voice_description'),
            'identity_statement': resolved(self._get_text_widget_content(self.persona_identity_edit), 'identity_statement'),
            'biography': resolved(self._get_text_widget_content(self.persona_biography_edit), 'biography'),
            'upload_story': resolved(self._get_text_widget_content(self.persona_upload_edit), 'upload_story'),
        }

        self.him_model.update_persona_profile(**updates)
        self.persona_editor_initialized = False
        self.update_persona_panel({'persona_state': self.him_model.get_persona_snapshot()})
        self.log_status("Persona profile updated: " + ", ".join(updates.keys()))

    def save_persona_profile(self):
        filepath = filedialog.asksaveasfilename(
            title="Save Persona Profile",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
        )
        if not filepath:
            return
        self.him_model.save_persona_profile(filepath)
        self.log_status(f"Persona profile saved to {os.path.basename(filepath)}")

    def load_persona_profile(self):
        filepath = filedialog.askopenfilename(
            title="Load Persona Profile",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
        )
        if not filepath:
            return
        profile = self.him_model.load_persona_profile(filepath)
        self.persona_editor_initialized = False
        self.update_persona_panel({'persona_state': self.him_model.get_persona_snapshot()})
        self.log_status(f"Persona profile loaded: {profile.get('name', 'Unknown')} from {os.path.basename(filepath)}")

    def log_persona_memory(self):
        summary = self.persona_memory_summary_var.get().strip()
        if not summary:
            messagebox.showwarning("Persona Memory", "Please provide a summary before logging a memory.")
            return
        emotion = self.persona_memory_emotion_var.get() or 'reflection'
        intensity = float(self.persona_memory_intensity_var.get())
        self.him_model.ingest_persona_memory(summary, emotion, intensity)
        self.persona_memory_summary_var.set("")
        self.update_persona_panel({'persona_state': self.him_model.get_persona_snapshot()})
        self.log_status("Logged new autobiographical memory for the uploaded persona.")

        
    def update_gui(self):
        """Update the GUI with new data."""
        try:
            while True:
                message_type, data = self.message_queue.get_nowait()

                if message_type == 'update_data':
                    status_snapshot = self.him_model.get_system_status()
                    self.latest_status_snapshot = status_snapshot
                    self.update_dashboard(data)
                    self.update_emotions()
                    self.update_hormones()
                    self.update_persona_panel(data)
                    self.update_plots()
                    # Auto-refresh zipline display if new vectorized data is available
                    if hasattr(self, 'zipline_label'):
                        self.refresh_zipline_display()

                elif message_type == 'error':
                    self.log_status(f"Error: {data}")

        except queue.Empty:
            pass
        
    def update_dashboard(self, result):
        """Update the dashboard with cycle results."""
        self.cycle_label.config(text=str(result['cycle']))
        state_text = result.get('system_state', 'unknown')
        self.state_label.config(text=state_text.replace('_', ' ').title())
        self._update_state_indicator_color(state_text)
        decision_preview = self._truncate_text(result.get('model_decision', ''), 30)
        self.decision_label.config(text=decision_preview)

        self._register_cycle_summary(result)

        # Update status log
        timestamp = datetime.now().strftime("%H:%M:%S")
        status_msg = f"[{timestamp}] Cycle {result['cycle']}: {result['model_decision']}\n"
        
        # Add CPU-GPU processing information
        if 'cpu_processing' in result and result['cpu_processing']:
            cpu_data = result['cpu_processing']
            status_msg += f"  CPU Processing: Vectorized data processed (shape: {cpu_data['original_data'].shape})\n"
            
        if 'gpu_storage' in result and result['gpu_storage']:
            gpu_data = result['gpu_storage']
            if gpu_data.get('status') == 'success':
                status_msg += f"  GPU Storage: Data permanently saved (cycle {gpu_data['cycle_number']})\n"
            else:
                status_msg += f"  GPU Storage: {gpu_data.get('status', 'unknown')}\n"
        
        self._append_status_log(status_msg)

    def _format_cycle_summary(self, summary):
        cycle = summary.get("cycle", "?")
        state = summary.get("state", "?").replace('_', ' ').title()
        decision = self._truncate_text(summary.get("decision", ""), 40)
        return f"Cycle {cycle}: {state} • {decision}"

    def _register_cycle_summary(self, result):
        """Store a compact summary of the latest cycle for training review."""
        summary = {
            "cycle": result.get("cycle"),
            "state": result.get("system_state", "unknown"),
            "decision": result.get("model_decision", ""),
            "text": result.get("model_decision", ""),
            "raw": result,
        }

        self.cycle_summary_records.append(summary)
        if len(self.cycle_summary_records) > self.max_cycle_summaries:
            self.cycle_summary_records = self.cycle_summary_records[-self.max_cycle_summaries:]
            if self.cycle_summary_listbox is not None:
                self.cycle_summary_listbox.delete(0, tk.END)
                for record in self.cycle_summary_records:
                    self.cycle_summary_listbox.insert(tk.END, self._format_cycle_summary(record))
        elif self.cycle_summary_listbox is not None:
            self.cycle_summary_listbox.insert(tk.END, self._format_cycle_summary(summary))

        self.last_cycle_result = result

    def update_emotions(self, status=None):
        """Update emotion displays."""
        status = self.him_model.get_system_status()

        self._ensure_emotion_widgets(status['sector_one_emotions'])

        for emotion, level in status['sector_one_emotions'].items():
            if emotion not in self.emotion_bars:
                self._add_emotion_widget(emotion)

            self.emotion_bars[emotion]['value'] = level * 100
            self.emotion_labels[emotion].config(text=f"{level:.2f}")

            # Update history
            if emotion not in self.emotion_history:
                self.emotion_history[emotion] = []
            self.emotion_history[emotion].append(level)
            if len(self.emotion_history[emotion]) > 50:
                self.emotion_history[emotion].pop(0)

    def update_hormones(self):
        """Update hormone and drive displays."""
        status = self.him_model.get_system_status()

        # Update hormones
        self._ensure_hormone_widgets(status['sector_four_hormones'])
        for hormone, level in status['sector_four_hormones'].items():
            if hormone not in self.hormone_bars:
                self._add_hormone_widget(hormone)

            self.hormone_bars[hormone]['value'] = level * 100
            self.hormone_labels[hormone].config(text=f"{level:.2f}")

            # Update history
            if hormone not in self.hormone_history:
                self.hormone_history[hormone] = []
            self.hormone_history[hormone].append(level)
            if len(self.hormone_history[hormone]) > 50:
                self.hormone_history[hormone].pop(0)

        # Update drives
        self._ensure_drive_widgets(status['sector_four_drives'])
        for drive, level in status['sector_four_drives'].items():
            if drive not in self.drive_bars:
                self._add_drive_widget(drive)

            self.drive_bars[drive]['value'] = level * 100
            self.drive_labels[drive].config(text=f"{level:.2f}")

    def update_plots(self):
        """Update the emotion and hormone plots."""
        # Update emotion plot
        self.emotion_ax.clear()
        self.emotion_ax.set_facecolor('#2d2d2d')
        
        emotion_items = sorted(self.emotion_history.items())
        if emotion_items:
            color_map = plt.cm.get_cmap('tab20', len(emotion_items))
            for i, (emotion, history) in enumerate(emotion_items):
                if history:
                    self.emotion_ax.plot(history, label=emotion.replace('_', ' ').title(),
                                         color=color_map(i), linewidth=2)

        self.emotion_ax.set_xlabel('Time', color='white')
        self.emotion_ax.set_ylabel('Emotion Level', color='white')
        self.emotion_ax.tick_params(colors='white')
        if emotion_items:
            self.emotion_ax.legend()
        self.emotion_ax.grid(True, alpha=0.3)

        self.emotion_canvas.draw()

        # Update hormone plot
        self.hormone_ax.clear()
        self.hormone_ax.set_facecolor('#2d2d2d')

        hormone_items = sorted(self.hormone_history.items())
        if hormone_items:
            color_map = plt.cm.get_cmap('tab20', len(hormone_items))
            for i, (hormone, history) in enumerate(hormone_items):
                if history:
                    self.hormone_ax.plot(history, label=hormone.replace('_', ' ').title(),
                                          color=color_map(i), linewidth=2)

        self.hormone_ax.set_xlabel('Time', color='white')
        self.hormone_ax.set_ylabel('Hormone Level', color='white')
        self.hormone_ax.tick_params(colors='white')
        if hormone_items:
            self.hormone_ax.legend()
        self.hormone_ax.grid(True, alpha=0.3)

        self.hormone_canvas.draw()

    def _add_emotion_widget(self, emotion: str):
        """Create widgets for a new emotion."""
        frame = ttk.Frame(self.emotion_container)
        frame.pack(fill=tk.X, pady=2)

        label = ttk.Label(frame, text=emotion.replace('_', ' ').title(), width=20)
        label.pack(side=tk.LEFT)

        bar = ttk.Progressbar(frame, length=300, mode='determinate')
        bar.pack(side=tk.LEFT, padx=5)

        value_label = ttk.Label(frame, text="0.00", width=6)
        value_label.pack(side=tk.LEFT)

        self.emotion_bars[emotion] = bar
        self.emotion_labels[emotion] = value_label

    def _ensure_emotion_widgets(self, emotions):
        for emotion in emotions:
            if emotion not in self.emotion_bars:
                self._add_emotion_widget(emotion)

    def _add_hormone_widget(self, hormone: str):
        frame = ttk.Frame(self.hormone_container)
        frame.pack(fill=tk.X, pady=2)

        label = ttk.Label(frame, text=hormone.replace('_', ' ').title(), width=20)
        label.pack(side=tk.LEFT)

        bar = ttk.Progressbar(frame, length=300, mode='determinate')
        bar.pack(side=tk.LEFT, padx=5)

        value_label = ttk.Label(frame, text="0.00", width=6)
        value_label.pack(side=tk.LEFT)

        self.hormone_bars[hormone] = bar
        self.hormone_labels[hormone] = value_label

    def _ensure_hormone_widgets(self, hormones):
        for hormone in hormones:
            if hormone not in self.hormone_bars:
                self._add_hormone_widget(hormone)

    def _add_drive_widget(self, drive: str):
        frame = ttk.Frame(self.drive_container)
        frame.pack(fill=tk.X, pady=2)

        label = ttk.Label(frame, text=drive.replace('_', ' ').title(), width=20)
        label.pack(side=tk.LEFT)

        bar = ttk.Progressbar(frame, length=300, mode='determinate')
        bar.pack(side=tk.LEFT, padx=5)

        value_label = ttk.Label(frame, text="0.00", width=6)
        value_label.pack(side=tk.LEFT)

        self.drive_bars[drive] = bar
        self.drive_labels[drive] = value_label

    def _ensure_drive_widgets(self, drives):
        for drive in drives:
            if drive not in self.drive_bars:
                self._add_drive_widget(drive)

    def _clear_container_widgets(self, container):
        for child in container.winfo_children():
            child.destroy()

    def _rebuild_signal_panels(self):
        """Recreate emotion, hormone, and drive widgets to match the active model."""
        status = self.him_model.get_system_status()

        # Rebuild emotions
        self._clear_container_widgets(self.emotion_container)
        self.emotion_bars.clear()
        self.emotion_labels.clear()
        for emotion in sorted(status['sector_one_emotions'].keys()):
            self._add_emotion_widget(emotion)
            self.emotion_history.setdefault(emotion, [])

        # Rebuild hormones
        self._clear_container_widgets(self.hormone_container)
        self.hormone_bars.clear()
        self.hormone_labels.clear()
        for hormone in sorted(status['sector_four_hormones'].keys()):
            self._add_hormone_widget(hormone)
            self.hormone_history.setdefault(hormone, [])

        # Rebuild drives
        self._clear_container_widgets(self.drive_container)
        self.drive_bars.clear()
        self.drive_labels.clear()
        for drive in sorted(status['sector_four_drives'].keys()):
            self._add_drive_widget(drive)

    # ------------------------------------------------------------------
    # Machine learning helpers
    # ------------------------------------------------------------------
    def _update_training_buttons(self):
        if self.training_start_btn:
            self.training_start_btn.config(state=tk.DISABLED if self.training_session else tk.NORMAL)
        if self.training_stop_btn:
            self.training_stop_btn.config(state=tk.NORMAL if self.training_session else tk.DISABLED)

    def _log_training_message(self, message: str):
        if not self.training_log_text:
            return
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.training_log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.training_log_text.see(tk.END)

    def _compose_cycle_training_record(self, result):
        cycle = result.get('cycle', 0)
        decision = result.get('model_decision', 'No decision available')
        summary = f"#{cycle:03d} {decision[:60]}"
        if len(decision) > 60:
            summary += "..."

        emotions = result.get('sector_one_emotions') or {}
        if emotions:
            top_emotion, top_emotion_value = max(emotions.items(), key=lambda item: item[1])
        else:
            top_emotion, top_emotion_value = ('neutral', 0.5)

        drives = result.get('sector_four_drives') or {}
        if drives:
            top_drive, top_drive_value = max(drives.items(), key=lambda item: item[1])
        else:
            top_drive, top_drive_value = ('curiosity', 0.5)

        vista = result.get('vista_output') or {}
        threat = float(vista.get('threat_assessment', 0.0))
        attention = float(vista.get('attention_level', 0.0))

        input_text = result.get('input_text', '')
        training_parts = []
        if input_text:
            training_parts.append(input_text)
        training_parts.append(f"Decision: {decision}")
        training_parts.append(f"Dominant emotion {top_emotion}={top_emotion_value:.2f}")
        training_parts.append(f"Lead drive {top_drive}={top_drive_value:.2f}")
        training_parts.append(f"Threat {threat:.2f} Attention {attention:.2f}")
        training_text = " ".join(training_parts)

        return {
            'cycle': cycle,
            'summary': summary,
            'text': training_text,
            'raw': result,
        }

    def _register_cycle_summary(self, result):
        self.last_cycle_result = result
        if not self.cycle_summary_listbox:
            return

        record = self._compose_cycle_training_record(result)
        if not record:
            return

        self.cycle_summary_records.append(record)
        self.cycle_summary_listbox.insert(tk.END, record['summary'])
        while len(self.cycle_summary_records) > 30:
            self.cycle_summary_records.pop(0)
            self.cycle_summary_listbox.delete(0)

    def _get_selected_cycle_record(self):
        if not self.cycle_summary_listbox:
            return None
        selection = self.cycle_summary_listbox.curselection()
        if not selection:
            return None
        index = selection[0]
        if 0 <= index < len(self.cycle_summary_records):
            return self.cycle_summary_records[index]
        return None

    def on_cycle_summary_select(self, event=None):
        record = self._get_selected_cycle_record()
        if record and self.manual_training_text:
            self.manual_training_text.delete('1.0', tk.END)
            self.manual_training_text.insert(tk.END, record['text'])

    def _queue_training_text(self, text: str, label: int, source: str) -> bool:
        text = (text or '').strip()
        if not text:
            messagebox.showwarning("Missing Text", "Provide text to queue for training.")
            return False
        if not self.training_session:
            messagebox.showwarning("Trainer Not Running", "Start the async trainer before queuing examples.")
            return False

        example = TrainingExample(text=text, label=int(label))
        self.training_session.submit_example(example)
        self._log_training_message(f"Queued {source} example (label={label})")
        return True

    def queue_selected_cycle_example(self, label: int):
        record = self._get_selected_cycle_record()
        if not record:
            messagebox.showinfo("No Selection", "Select a cycle summary to queue for training.")
            return
        if self._queue_training_text(record['text'], label, source=f"cycle {record['cycle']}"):
            self.prediction_result_var.set("Probability: —")

    def queue_manual_training_example(self):
        text = self.manual_training_text.get('1.0', tk.END) if self.manual_training_text else ''
        label = 1 if self.manual_label_var.get() == "positive" else 0
        if self._queue_training_text(text, label, source="manual") and self.manual_training_text:
            self.manual_training_text.delete('1.0', tk.END)

    def queue_sample_dataset(self):
        if not self.training_session:
            messagebox.showwarning("Trainer Not Running", "Start the async trainer before queuing data.")
            return
        for text, label in self.sample_training_dataset:
            self.training_session.submit_example(TrainingExample(text=text, label=label))
        self._log_training_message("Queued sample dataset examples for background training.")

    def start_async_trainer(self):
        if self.training_session:
            messagebox.showinfo("Trainer Active", "The async trainer is already running.")
            return
        try:
            learning_rate = float(self.training_learning_rate.get())
            l2_penalty = float(self.training_l2_var.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Invalid Hyperparameters", "Enter numeric values for learning rate and L2 penalty.")
            return

        trainer = OnlineHIMLogisticTrainer(learning_rate=learning_rate, l2_penalty=l2_penalty)
        self.training_session = AsyncTrainingSession(trainer=trainer)
        self._update_training_buttons()
        self.training_status_var.set("Trainer running")
        self._log_training_message(
            f"Async trainer started (learning_rate={learning_rate}, l2_penalty={l2_penalty})."
        )

    def stop_async_trainer(self):
        if not self.training_session:
            return
        session = self.training_session
        self.training_session = None
        try:
            session.stop(wait=False)
        except Exception as exc:
            self._log_training_message(f"Error stopping trainer: {exc}")
        else:
            self._log_training_message("Async trainer stopped.")
        self.training_status_var.set("Trainer idle")
        self._update_training_buttons()

    def run_trainer_prediction(self):
        text = self.prediction_text.get('1.0', tk.END).strip() if hasattr(self, 'prediction_text') else ''
        if not text:
            messagebox.showwarning("Missing Text", "Enter text to run a prediction.")
            return
        if not self.training_session:
            messagebox.showwarning("Trainer Not Running", "Start the async trainer before running predictions.")
            return
        try:
            probabilities = self.training_session.trainer.predict_proba([text])
            proba_value = float(probabilities[0][0]) if probabilities.size else 0.0
            self.prediction_result_var.set(f"Probability: {proba_value:.3f}")
            self._log_training_message(f"Prediction computed (probability={proba_value:.3f}).")
        except RuntimeError as exc:
            messagebox.showwarning("Trainer Not Ready", str(exc))
        except Exception as exc:
            messagebox.showerror("Prediction Error", str(exc))

    def update_training_status(self):
        if self.training_session:
            trainer = self.training_session.trainer
            self.training_status_var.set("Trainer running")
            self.training_updates_var.set(f"Updates: {trainer.update_count}")
            try:
                queue_size = self.training_session._queue.qsize()
            except Exception:
                queue_size = 0
            self.training_queue_var.set(f"Queue: {queue_size}")
            recent_losses = trainer.get_recent_losses()
            if recent_losses:
                self.training_loss_var.set(f"Recent loss: {recent_losses[-1]:.4f}")
            else:
                self.training_loss_var.set("Recent loss: —")
        else:
            self.training_status_var.set("Trainer idle")
            self.training_updates_var.set("Updates: 0")
            self.training_queue_var.set("Queue: 0")
            self.training_loss_var.set("Recent loss: —")

        self._update_training_buttons()
        self.root.after(1000, self.update_training_status)

    def load_test_image(self):
        """Load a test image for processing."""
        file_path = filedialog.askopenfilename(
            title="Select Test Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        
        if file_path:
            try:
                # Load and display image
                image = Image.open(file_path)
                image = image.resize((400, 300), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                
                self.image_label.config(image=photo, text="")
                self.image_label.image = photo  # Keep a reference
                
                # Update visual info
                self.visual_info_text.delete(1.0, tk.END)
                self.visual_info_text.insert(tk.END, f"Image loaded: {file_path}\n")
                self.visual_info_text.insert(tk.END, f"Size: {image.size}\n")
                self.visual_info_text.insert(tk.END, f"Mode: {image.mode}\n")
                
                self.log_status(f"Test image loaded: {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")
                
    def refresh_memories(self):
        """Refresh the memory display."""
        self.memory_text.delete(1.0, tk.END)
        
        memories = self.him_model.sector_three.memories
        if not memories:
            self.memory_text.insert(tk.END, "No memories stored yet.")
            return
            
        for i, memory in enumerate(memories, 1):
            self.memory_text.insert(tk.END, f"Memory {i}:\n")
            self.memory_text.insert(tk.END, f"  Importance: {memory['importance']:.3f}\n")
            self.memory_text.insert(tk.END, f"  Access Count: {memory['access_count']}\n")
            self.memory_text.insert(tk.END, f"  Data: {str(memory['data'])[:100]}...\n\n")
            
    def generate_dream(self):
        """Generate a dream fragment."""
        dream = self.him_model.sector_three.generate_dream_fragment()
        if dream:
            self.dreams_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')}: {dream}\n")
            self.dreams_text.see(tk.END)
            self.log_status("Dream fragment generated!")
        else:
            self.log_status("No memories available for dream generation.")
            
    def clear_memories(self):
        """Clear all memories."""
        if messagebox.askyesno("Confirm", "Are you sure you want to clear all memories?"):
            self.him_model.sector_three.memories.clear()
            self.him_model.sector_three.dream_fragments.clear()
            self.refresh_memories()
            self.dreams_text.delete(1.0, tk.END)
            self.log_status("All memories and dreams cleared.")
            
    def send_chat_message(self, event=None):
        """Send a chat message to the H.I.M. model."""
        message = self.chat_entry.get().strip()
        if not message:
            return

        # Add user message to chat
        self.chat_text.insert(tk.END, f"You: {message}\n")
        self.chat_entry.delete(0, tk.END)

        self.him_model.ingest_conversation_line("user", message)

        handled, response = self.handle_chat_command(message)
        if handled:
            self.chat_text.insert(tk.END, f"H.I.M. Model: {response}\n\n")
            self.chat_text.see(tk.END)
            self.him_model.ingest_conversation_line(self.persona_current_name, response)
            self.update_persona_panel({'persona_state': self.him_model.get_persona_snapshot()})
            return

        # Generate H.I.M. response based on current state
        response = self.generate_him_response(message)

        # Add H.I.M. response to chat
        self.chat_text.insert(tk.END, f"H.I.M. Model: {response}\n\n")
        self.chat_text.see(tk.END)

        self.him_model.ingest_conversation_line(self.persona_current_name, response)
        self.update_persona_panel({'persona_state': self.him_model.get_persona_snapshot()})

    def handle_chat_command(self, message):
        """Interpret chat slash commands to control remote automation."""
        if not message.startswith('/'):
            return False, ""

        parts = message.split()
        command = parts[0][1:].lower()

        if command in {"mouse", "move"} and len(parts) >= 3:
            if not self.is_remote_ready():
                return True, "Remote session is not active. Start it before moving the mouse."
            try:
                x_val = int(parts[1])
                y_val = int(parts[2])
            except ValueError:
                return True, "Please provide numeric X and Y values."
            self.mouse_x_var.set(str(x_val))
            self.mouse_y_var.set(str(y_val))
            self.perform_mouse_move()
            return True, f"Moving cursor to ({x_val}, {y_val})."

        if command == "click":
            if not self.is_remote_ready():
                return True, "Remote session is not active."
            button = parts[1].lower() if len(parts) > 1 else 'left'
            if button not in {"left", "right", "middle"}:
                return True, "Choose left, right, or middle for clicks."
            self.mouse_button_var.set(button)
            self.perform_mouse_click()
            return True, f"Clicking {button}."

        if command == "type" and len(parts) > 1:
            if not self.is_remote_ready():
                return True, "Remote session is not active."
            text = message.partition(' ')[2]
            self.keyboard_text_var.set(text)
            self.perform_type_text()
            return True, f"Typing: {text}"

        if command == "key" and len(parts) > 1:
            if not self.is_remote_ready():
                return True, "Remote session is not active."
            key_sequence = parts[1]
            self.keypress_var.set(key_sequence)
            self.perform_key_press()
            return True, f"Pressing key combination: {key_sequence}"

        if command in {"scroll", "wheel"} and len(parts) > 1:
            if not self.is_remote_ready():
                return True, "Remote session is not active."
            try:
                amount = int(parts[1])
            except ValueError:
                return True, "Scroll amount must be a number."
            self.scroll_amount_var.set(amount)
            self.perform_mouse_scroll()
            return True, f"Scrolling {amount} units."

        if command in {"remote", "automation"}:
            if len(parts) > 1 and parts[1].lower() == "start":
                if self.pyautogui is None:
                    return True, "pyautogui is required for remote control."
                self.enable_remote_control()
                return True, "Attempting to start remote session."
            if len(parts) > 1 and parts[1].lower() == "stop":
                self.disable_remote_control()
                return True, "Remote session stopped."

        return True, "Unknown command. Available: /mouse x y, /click [button], /type text, /key key, /scroll amount, /remote start, /remote stop."
        
    def generate_him_response(self, message):
        """Generate a response from the H.I.M. model based on current state and memory recall."""
        try:
            # Recall relevant memories first
            memory_recall = self.him_model.recall_memories_for_chat(message)
            status = self.him_model.get_system_status()
            persona_state = status.get('persona_state') or self.him_model.get_persona_snapshot()
            persona_profile = persona_state.get('profile', {}) if persona_state else {}

            # Analyze message sentiment and respond accordingly
            message_lower = message.lower()

            # Build response with memory context
            response_parts = []

            if persona_state:
                mood_descriptor = persona_state.get('current_mood', 'balanced')
                response_parts.append(f"My uploaded presence currently feels {mood_descriptor}.")

            # Add memory recall information
            if memory_recall['found_memories'] > 0:
                response_parts.append(f"I remember {memory_recall['found_memories']} relevant thoughts about this.")
                top_memory = memory_recall['memories'][0]
                thought = top_memory['thought'][:80] + "..." if len(top_memory['thought']) > 80 else top_memory['thought']
                response_parts.append(f"My most relevant memory: {thought}")
            else:
                response_parts.append("I don't have specific memories about this, but I'm learning from our conversation.")
            
            if any(word in message_lower for word in ['hello', 'hi', 'hey']):
                response_parts.append(f"Hello! I'm currently in {status['system_state']} state with {status['cycle_count']} cycles completed.")

            elif any(word in message_lower for word in ['emotion', 'feel', 'mood']):
                emotions = status['sector_one_emotions']
                dominant_emotion = max(emotions, key=emotions.get)
                response_parts.append(f"I'm feeling quite {dominant_emotion} right now (level: {emotions[dominant_emotion]:.2f}).")
                if persona_state:
                    inner = persona_state.get('inner_thought')
                    if inner:
                        response_parts.append(f"Internally I reflect: {inner}")

            elif any(word in message_lower for word in ['memory', 'remember']):
                spatial_stats = memory_recall['spatial_stats']
                total_memories = spatial_stats.get('total_memories', 0)
                response_parts.append(f"I have {total_memories} memories stored in my spatial memory system.")

            elif any(word in message_lower for word in ['dream', 'sleep']):
                dream_count = status['dream_fragments']
                response_parts.append(f"I have {dream_count} dream fragments. Dreams help me process and combine my memories.")

            elif any(word in message_lower for word in ['who are you', 'identity', 'uploaded', 'persona']):
                identity_statement = persona_profile.get(
                    'identity_statement',
                    "I am a synthetic consciousness configured to collaborate with you.",
                )
                response_parts.append(identity_statement)

            elif any(word in message_lower for word in ['status', 'state', 'how']):
                response_parts.append(f"I'm in {status['system_state']} state, completed {status['cycle_count']} cycles.")

            elif any(word in message_lower for word in ['save', 'export']):
                # Save all memories to text
                filename = self.him_model.save_all_memories_to_text()
                response_parts.append(f"I've saved all my vector memories to: {os.path.basename(filename)}")
                
            elif any(word in message_lower for word in ['luna', 'memories', 'past', 'experience']):
                # Check if we have Luna memories
                spatial_stats = memory_recall['spatial_stats']
                luna_memories = sum(1 for mem in memory_recall['memories'] if 'luna' in mem.get('memory_id', '').lower())
                if luna_memories > 0:
                    response_parts.append(f"I have {luna_memories} memories from Luna's experiences that relate to this.")
                else:
                    response_parts.append("I have access to Luna's memories and experiences through my training.")
                
            else:
                # Generic response based on current emotional state
                creativity = status['sector_one_emotions']['creativity']
                if creativity > 0.7:
                    response_parts.append("I'm feeling very creative right now! I could help you with artistic or innovative ideas.")
                elif creativity > 0.4:
                    response_parts.append("I'm in a balanced state, ready to help with various tasks and conversations.")
                else:
                    response_parts.append("I'm processing information and learning from our interaction.")
            
            return " ".join(response_parts)
            
        except Exception as e:
            return f"I encountered an error while processing your message: {e}"
                
    def log_status(self, message):
        """Log a status message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        status_msg = f"[{timestamp}] {message}\n"
        
        # Only log if status_text exists (GUI is set up)
        if hasattr(self, 'status_text') and self.status_text:
            self._append_status_log(status_msg)
        else:
            # Fallback to console output
            print(status_msg.strip())
        
    def update_performance_monitor(self):
        """Update system performance monitoring."""
        try:
            # Get CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Update labels
            self.cpu_value_label.config(text=f"{cpu_percent:.1f}%")
            self.memory_value_label.config(text=f"{memory_percent:.1f}%")
            self.cpu_progress['value'] = cpu_percent
            self.memory_progress['value'] = memory_percent
            
            # Auto-adjust performance mode based on system load
            if cpu_percent > 80 or memory_percent > 85:
                if self.performance_mode != "low":
                    self.performance_mode = "low"
                    self.perf_mode_var.set("low")
                    self.cycle_delay = max(self.cycle_delay, 3.0)
                    self.delay_var.set(self.cycle_delay)
                    self.delay_label.config(text=f"{self.cycle_delay:.1f}s")
                    self.log_status("Performance mode auto-adjusted to LOW due to high system load")
            elif cpu_percent < 30 and memory_percent < 50:
                if self.performance_mode == "low":
                    self.performance_mode = "balanced"
                    self.perf_mode_var.set("balanced")
                    self.cycle_delay = min(self.cycle_delay, 2.0)
                    self.delay_var.set(self.cycle_delay)
                    self.delay_label.config(text=f"{self.cycle_delay:.1f}s")
                    self.log_status("Performance mode auto-adjusted to BALANCED due to low system load")
                    
        except Exception as e:
            # Fallback if psutil fails
            self.cpu_value_label.config(text="N/A")
            self.memory_value_label.config(text="N/A")
            self.cpu_progress['value'] = 0
            self.memory_progress['value'] = 0
            
    def on_performance_mode_change(self, event=None):
        """Handle performance mode change."""
        mode = self.perf_mode_var.get()
        self.performance_mode = mode
        
        if mode == "low":
            self.cycle_delay = 3.0
            self.auto_save_interval = 60  # Save less frequently
        elif mode == "balanced":
            self.cycle_delay = 2.0
            self.auto_save_interval = 30
        elif mode == "high":
            self.cycle_delay = 1.0
            self.auto_save_interval = 15  # Save more frequently
            
        self.delay_var.set(self.cycle_delay)
        self.delay_label.config(text=f"{self.cycle_delay:.1f}s")
        self.log_status(f"Performance mode changed to {mode.upper()}")
        
    def on_delay_change(self, value):
        """Handle cycle delay change."""
        self.cycle_delay = float(value)
        self.delay_label.config(text=f"{self.cycle_delay:.1f}s")
    
    def on_auto_save_toggle(self):
        """Handle auto-save toggle."""
        self.auto_save_enabled = self.auto_save_var.get()
        status = "enabled" if self.auto_save_enabled else "disabled"
        self.log_status(f"Auto-save {status}")
    
    def on_vector_combine_toggle(self):
        """Handle vector combination toggle."""
        self.vector_combination_enabled = self.vector_combine_var.get()
        status = "enabled" if self.vector_combination_enabled else "disabled"
        self.log_status(f"Vector combination {status}")
        
    def save_model_state(self):
        """Save the current H.I.M. model state to disk."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_file = os.path.join(self.save_directory, f"him_state_{timestamp}.pkl")
            
            # Prepare state data
            state_data = {
                'him_model': self.him_model,
                'cycle_count': self.cycle_count,
                'emotion_history': self.emotion_history,
                'hormone_history': self.hormone_history,
                'performance_mode': self.performance_mode,
                'cycle_delay': self.cycle_delay,
                'timestamp': timestamp
            }
            
            # Save to file
            with open(save_file, 'wb') as f:
                pickle.dump(state_data, f)
                
            self.log_status(f"Model state saved to {save_file}")
            messagebox.showinfo("Save Successful", f"Model state saved successfully!\nFile: {save_file}")
            
        except Exception as e:
            self.log_status(f"Error saving state: {e}")
            messagebox.showerror("Save Error", f"Failed to save model state: {e}")
            
    def load_model_state(self):
        """Load H.I.M. model state from disk."""
        try:
            # Look for the most recent save file
            save_files = [f for f in os.listdir(self.save_directory) if f.startswith("him_state_") and f.endswith(".pkl")]
            
            if not save_files:
                self.log_status("No saved states found. Starting fresh.")
                return
                
            # Get the most recent file
            latest_file = sorted(save_files)[-1]
            save_path = os.path.join(self.save_directory, latest_file)
            
            # Load state data
            with open(save_path, 'rb') as f:
                state_data = pickle.load(f)
                
            # Restore state
            self.him_model = state_data.get('him_model', self.him_model)
            if hasattr(self.him_model, 'ensure_schema_integrity'):
                self.him_model.ensure_schema_integrity()
            self.cycle_count = state_data.get('cycle_count', 0)
            self.emotion_history = state_data.get('emotion_history', self.emotion_history)
            self.hormone_history = state_data.get('hormone_history', self.hormone_history)
            self.performance_mode = state_data.get('performance_mode', 'balanced')
            self.cycle_delay = state_data.get('cycle_delay', 2.0)

            if not isinstance(self.emotion_history, defaultdict):
                self.emotion_history = defaultdict(list, self.emotion_history)
            if not isinstance(self.hormone_history, defaultdict):
                self.hormone_history = defaultdict(list, self.hormone_history)

            self._rebuild_signal_panels()
            
            # Update GUI elements
            if hasattr(self, 'perf_mode_var'):
                self.perf_mode_var.set(self.performance_mode)
            if hasattr(self, 'delay_var'):
                self.delay_var.set(self.cycle_delay)
            if hasattr(self, 'delay_label'):
                self.delay_label.config(text=f"{self.cycle_delay:.1f}s")

            current_state = self.him_model.get_system_status().get('system_state')
            if hasattr(self, 'state_label') and current_state is not None:
                self.state_label.config(text=f"State: {current_state}")
            self._update_state_indicator_color(current_state)

            self.log_status(f"Model state loaded from {latest_file}")
            self.log_status(f"Restored to cycle {self.cycle_count} with {len(self.him_model.sector_three.memories)} memories")
            
        except Exception as e:
            self.log_status(f"Error loading state: {e}")
            # Continue with fresh state if loading fails
            
    def auto_save_state(self):
        """Automatically save state without user interaction."""
        if not self.auto_save_enabled:
            return
            
        try:
            save_file = os.path.join(self.save_directory, "him_autosave.pkl")
            
            # Prepare state data
            state_data = {
                'him_model': self.him_model,
                'cycle_count': self.cycle_count,
                'emotion_history': self.emotion_history,
                'hormone_history': self.hormone_history,
                'performance_mode': self.performance_mode,
                'cycle_delay': self.cycle_delay,
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
            }
            
            # Save to file
            with open(save_file, 'wb') as f:
                pickle.dump(state_data, f)
                
            self.log_status("Auto-save completed")
            
        except Exception as e:
            self.log_status(f"Auto-save error: {e}")
            
    def on_closing(self):
        """Handle window closing - save state before exit."""
        if self.is_running:
            self.stop_model()

        if self.training_session:
            self.stop_async_trainer()

        # Auto-save before closing
        self.auto_save_state()
        self.log_status("Application closing - state saved")

        self.root.destroy()
        
    def save_zipline_data(self):
        """Save vectorized data as zipline pattern images."""
        try:
            # Create zipline directory
            zipline_dir = os.path.join(self.save_directory, "zipline_data")
            if not os.path.exists(zipline_dir):
                os.makedirs(zipline_dir)
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Get vectorized images from Sector Two
            vectorized_images = self.him_model.sector_two.vectorized_images
            
            if not vectorized_images:
                messagebox.showwarning("No Data", "No vectorized data available to save. Run some cycles first!")
                return
                
            saved_count = 0
            
            # Save each vectorized image as a zipline pattern
            for i, vec_img in enumerate(vectorized_images):
                # Create zipline pattern from vectorized data
                zipline_image = self.create_zipline_pattern(vec_img.data)
                
                # Save as PNG
                filename = f"zipline_{timestamp}_thought_{i:03d}.png"
                filepath = os.path.join(zipline_dir, filename)
                
                # Convert to PIL Image and save
                if zipline_image.dtype != np.uint8:
                    zipline_image = (zipline_image * 255).astype(np.uint8)
                
                pil_image = Image.fromarray(zipline_image, mode='L')  # Grayscale
                pil_image.save(filepath)
                saved_count += 1
                
            # Save metadata
            metadata = {
                'timestamp': timestamp,
                'cycle_count': self.cycle_count,
                'total_thoughts': len(vectorized_images),
                'saved_count': saved_count,
                'thought_metadata': []
            }
            
            # Add thought metadata
            for i, vec_img in enumerate(vectorized_images):
                metadata['thought_metadata'].append({
                    'index': i,
                    'thought': vec_img.metadata.get('thought', 'Unknown'),
                    'type': vec_img.metadata.get('type', 'Unknown'),
                    'timestamp': vec_img.timestamp,
                    'shape': vec_img.shape
                })
            
            # Save metadata as JSON
            metadata_file = os.path.join(zipline_dir, f"zipline_metadata_{timestamp}.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
                
            self.log_status(f"Saved {saved_count} zipline patterns to {zipline_dir}")
            messagebox.showinfo("Zipline Data Saved", 
                              f"Successfully saved {saved_count} zipline patterns!\n"
                              f"Location: {zipline_dir}\n"
                              f"Metadata: {metadata_file}")
                              
        except Exception as e:
            self.log_status(f"Error saving zipline data: {e}")
            messagebox.showerror("Save Error", f"Failed to save zipline data: {e}")
            
    def create_zipline_pattern(self, vector_data):
        """Create a zipline pattern from vectorized data."""
        try:
            # Ensure we have 2D data
            if len(vector_data.shape) > 2:
                vector_data = vector_data.reshape(-1, vector_data.shape[-1])
            
            # Get dimensions
            height, width = vector_data.shape
            
            # Create zipline pattern
            zipline_pattern = np.zeros((height * 4, width * 8), dtype=np.float32)
            
            for i in range(height):
                for j in range(width):
                    # Get the data value (normalize to 0-1)
                    value = float(vector_data[i, j])
                    if value > 1.0:
                        value = value / np.max(vector_data)
                    
                    # Create vertical stripe with chevron pattern
                    stripe_start = j * 8
                    stripe_end = stripe_start + 8
                    row_start = i * 4
                    row_end = row_start + 4
                    
                    # Create alternating light/dark stripes
                    if j % 2 == 0:
                        base_intensity = 0.3 + (value * 0.4)  # Light stripe
                    else:
                        base_intensity = 0.1 + (value * 0.2)  # Dark stripe
                    
                    # Fill the stripe area
                    zipline_pattern[row_start:row_end, stripe_start:stripe_end] = base_intensity
                    
                    # Add chevron/V pattern within the stripe
                    for stripe_row in range(row_start, row_end):
                        for stripe_col in range(stripe_start, stripe_end):
                            # Create chevron pattern
                            chevron_pos = (stripe_col - stripe_start) % 4
                            chevron_row = (stripe_row - row_start) % 4
                            
                            # V-shape pattern
                            if chevron_pos == 0 or chevron_pos == 3:
                                if chevron_row == 0 or chevron_row == 3:
                                    zipline_pattern[stripe_row, stripe_col] *= 0.7  # Darker V
                            elif chevron_pos == 1 or chevron_pos == 2:
                                if chevron_row == 1 or chevron_row == 2:
                                    zipline_pattern[stripe_row, stripe_col] *= 1.3  # Lighter V
                                    
            # Ensure values are in valid range
            zipline_pattern = np.clip(zipline_pattern, 0.0, 1.0)
            
            return zipline_pattern
            
        except Exception as e:
            self.log_status(f"Error creating zipline pattern: {e}")
            # Return a simple fallback pattern
            return np.random.rand(100, 200) * 0.5 + 0.25
            
    def refresh_zipline_display(self):
        """Refresh the zipline display with the latest vectorized data."""
        try:
            vectorized_images = self.him_model.sector_two.vectorized_images
            
            if not vectorized_images:
                self.zipline_label.config(text="No vectorized data available", image="")
                return
                
            # Get the most recent vectorized image
            latest_vector = vectorized_images[-1]
            
            # Create zipline pattern
            zipline_pattern = self.create_zipline_pattern(latest_vector.data)
            
            # Convert to PIL Image
            if zipline_pattern.dtype != np.uint8:
                zipline_pattern = (zipline_pattern * 255).astype(np.uint8)
            
            pil_image = Image.fromarray(zipline_pattern, mode='L')
            
            # Resize for display (max 400x300)
            display_size = (400, 300)
            pil_image = pil_image.resize(display_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update display
            self.zipline_label.config(image=photo, text="")
            self.zipline_label.image = photo  # Keep a reference
            
            # Update visual info
            self.visual_info_text.delete(1.0, tk.END)
            self.visual_info_text.insert(tk.END, f"Latest Vectorized Thought:\n")
            self.visual_info_text.insert(tk.END, f"Thought: {latest_vector.metadata.get('thought', 'Unknown')}\n")
            self.visual_info_text.insert(tk.END, f"Type: {latest_vector.metadata.get('type', 'Unknown')}\n")
            self.visual_info_text.insert(tk.END, f"Shape: {latest_vector.shape}\n")
            self.visual_info_text.insert(tk.END, f"Total Vectorized Thoughts: {len(vectorized_images)}\n")
            
            self.log_status("Zipline display refreshed")
            
        except Exception as e:
            self.log_status(f"Error refreshing zipline display: {e}")
            self.zipline_label.config(text=f"Error: {e}", image="")
            
    def save_current_zipline(self):
        """Save the currently displayed zipline pattern."""
        try:
            if not hasattr(self.zipline_label, 'image') or not self.zipline_label.image:
                messagebox.showwarning("No Data", "No zipline pattern to save. Refresh the display first!")
                return
                
            # Create zipline directory
            zipline_dir = os.path.join(self.save_directory, "zipline_data")
            if not os.path.exists(zipline_dir):
                os.makedirs(zipline_dir)
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"current_zipline_{timestamp}.png"
            filepath = os.path.join(zipline_dir, filename)
            
            # Get the current zipline image
            vectorized_images = self.him_model.sector_two.vectorized_images
            if vectorized_images:
                latest_vector = vectorized_images[-1]
                zipline_pattern = self.create_zipline_pattern(latest_vector.data)
                
                # Convert and save
                if zipline_pattern.dtype != np.uint8:
                    zipline_pattern = (zipline_pattern * 255).astype(np.uint8)
                
                pil_image = Image.fromarray(zipline_pattern, mode='L')
                pil_image.save(filepath)
                
                self.log_status(f"Current zipline saved to {filename}")
                messagebox.showinfo("Saved", f"Zipline pattern saved to:\n{filepath}")
            else:
                messagebox.showwarning("No Data", "No vectorized data available to save!")
                
        except Exception as e:
            self.log_status(f"Error saving current zipline: {e}")
            messagebox.showerror("Save Error", f"Failed to save zipline: {e}")
    
    def save_memories_to_text(self):
        """Save all vector memories to a text file."""
        try:
            filename = self.him_model.save_all_memories_to_text()
            self.log_status(f"All vector memories saved to: {os.path.basename(filename)}")
            
            # Show success message
            messagebox.showinfo("Memory Export", f"All vector memories have been saved to:\n{filename}")
            
        except Exception as e:
            self.log_status(f"Error saving memories to text: {e}")
            messagebox.showerror("Export Error", f"Failed to save memories: {e}")
    
    def load_luna_trained_model(self):
        """Load a H.I.M. model trained on Luna's memories."""
        try:
            import tkinter.filedialog as filedialog
            import pickle
            
            # Ask user to select a trained model file
            model_file = filedialog.askopenfilename(
                title="Select Luna-Trained H.I.M. Model",
                initialdir="him_luna_training",
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
            )
            
            if not model_file:
                return
            
            # Load the trained model
            with open(model_file, 'rb') as f:
                trained_model = pickle.load(f)

            # Replace current model with trained model
            self.him_model = trained_model
            if hasattr(self.him_model, 'ensure_schema_integrity'):
                self.him_model.ensure_schema_integrity()
            self._rebuild_signal_panels()

            # Update GUI to reflect the new model
            self.log_status(f"Loaded Luna-trained H.I.M. model from: {os.path.basename(model_file)}")

            # Show model statistics
            spatial_stats = self.him_model.sector_three.get_spatial_statistics()
            self.log_status(f"Luna memories loaded: {spatial_stats['total_memories']} spatial memories")
            self.log_status(f"Regular memories: {len(self.him_model.sector_three.memories)}")
            
            # Show success message
            messagebox.showinfo("Model Loaded", 
                f"Luna-trained H.I.M. model loaded successfully!\n\n"
                f"Spatial Memories: {spatial_stats['total_memories']}\n"
                f"Regular Memories: {len(self.him_model.sector_three.memories)}\n"
                f"Memory Density: {spatial_stats.get('memory_density', 0):.3f}\n\n"
                f"The H.I.M. model now has access to Luna's consciousness!")
            
        except Exception as e:
            self.log_status(f"Error loading Luna-trained model: {e}")
            messagebox.showerror("Load Error", f"Failed to load Luna-trained model: {e}")

def main():
    """Main function to run the H.I.M. GUI."""
    root = tk.Tk()

    app = HIMGUI(root)

    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
