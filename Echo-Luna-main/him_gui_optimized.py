#!/usr/bin/env python3
"""
Optimized H.I.M. Model GUI
=========================

This optimized version prevents GUI freezing and crashes by:
1. Implementing proper threading with thread-safe operations
2. Adding memory management and garbage collection
3. Implementing cycle throttling and resource monitoring
4. Adding crash recovery and auto-restart capabilities
5. Optimizing GUI updates to prevent blocking
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import queue
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import psutil
import gc
import os
import pickle
from datetime import datetime
from him_model import HIMModel, VectorizedImage

class OptimizedHIMGUI:
    """Optimized H.I.M. Model GUI with crash prevention and performance optimization."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("H.I.M. Model - Optimized Version")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1e1e1e')
        
        # Initialize H.I.M. model
        self.him_model = HIMModel()
        
        # Threading and performance
        self.is_running = False
        self.model_thread = None
        self.message_queue = queue.Queue(maxsize=100)  # Limit queue size
        self.cycle_count = 0
        self.last_cycle_time = time.time()
        
        # Performance monitoring
        self.cycle_delay = 2.0  # Increased default delay
        self.max_cycles_per_minute = 20  # Limit cycles
        self.cycle_times = []
        self.memory_usage_history = []
        self.cpu_usage_history = []
        
        # Auto-save and recovery
        self.auto_save_enabled = True
        self.auto_save_interval = 30  # seconds
        self.last_auto_save = time.time()
        self.crash_recovery_enabled = True
        
        # Memory management
        self.max_memory_mb = 1024  # 1GB limit
        self.gc_interval = 10  # cycles
        self.last_gc = 0
        
        # GUI optimization
        self.update_interval = 200  # ms - slower updates
        self.batch_updates = True
        self.pending_updates = []
        
        # Create GUI
        self.create_gui()
        self.setup_style()
        
        # Start monitoring
        self.start_performance_monitor()
        
        # Load last state if available
        self.load_last_state()
        
    def create_gui(self):
        """Create the optimized GUI."""
        # Main notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_dashboard_tab()
        self.create_emotions_tab()
        self.create_hormones_tab()
        self.create_memory_tab()
        self.create_chat_tab()
        self.create_visual_tab()
        self.create_performance_tab()
        
    def create_dashboard_tab(self):
        """Create the optimized dashboard tab."""
        dashboard_frame = ttk.Frame(self.notebook)
        self.notebook.add(dashboard_frame, text="Dashboard")
        
        # Control panel with optimization controls
        control_frame = ttk.LabelFrame(dashboard_frame, text="Control Panel", padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Basic controls
        self.start_btn = ttk.Button(control_frame, text="Start H.I.M. Model", command=self.start_model)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(control_frame, text="Stop Model", command=self.stop_model, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Optimization controls
        ttk.Label(control_frame, text="Cycle Delay:").pack(side=tk.LEFT, padx=(20,5))
        self.delay_var = tk.DoubleVar(value=self.cycle_delay)
        self.delay_scale = ttk.Scale(control_frame, from_=0.5, to=10.0, variable=self.delay_var, 
                                   orient=tk.HORIZONTAL, length=150, command=self.update_delay)
        self.delay_scale.pack(side=tk.LEFT, padx=5)
        
        self.delay_label = ttk.Label(control_frame, text=f"{self.cycle_delay:.1f}s")
        self.delay_label.pack(side=tk.LEFT, padx=5)
        
        # Performance mode
        ttk.Label(control_frame, text="Performance:").pack(side=tk.LEFT, padx=(20,5))
        self.perf_mode = ttk.Combobox(control_frame, values=["low", "balanced", "high"], 
                                     state="readonly", width=10)
        self.perf_mode.set("balanced")
        self.perf_mode.pack(side=tk.LEFT, padx=5)
        self.perf_mode.bind("<<ComboboxSelected>>", self.update_performance_mode)
        
        # Memory management controls
        memory_frame = ttk.LabelFrame(dashboard_frame, text="Memory Management", padding=10)
        memory_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.auto_save_var = tk.BooleanVar(value=self.auto_save_enabled)
        ttk.Checkbutton(memory_frame, text="Auto-Save Enabled", variable=self.auto_save_var,
                       command=self.toggle_auto_save).pack(side=tk.LEFT, padx=5)
        
        self.gc_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(memory_frame, text="Auto Garbage Collection", variable=self.gc_var).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(memory_frame, text="Force Garbage Collection", command=self.force_gc).pack(side=tk.LEFT, padx=5)
        ttk.Button(memory_frame, text="Clear Memory Cache", command=self.clear_memory_cache).pack(side=tk.LEFT, padx=5)
        
        # System status
        status_frame = ttk.LabelFrame(dashboard_frame, text="System Status", padding=10)
        status_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Status labels
        info_frame = ttk.Frame(status_frame)
        info_frame.pack(fill=tk.X, pady=5)
        
        self.cycle_label = ttk.Label(info_frame, text="Cycle: 0", font=("Arial", 12, "bold"))
        self.cycle_label.pack(side=tk.LEFT, padx=10)
        
        self.state_label = ttk.Label(info_frame, text="State: Stopped", font=("Arial", 12))
        self.state_label.pack(side=tk.LEFT, padx=10)
        
        self.memory_label = ttk.Label(info_frame, text="Memory: 0 MB", font=("Arial", 12))
        self.memory_label.pack(side=tk.LEFT, padx=10)
        
        self.cpu_label = ttk.Label(info_frame, text="CPU: 0%", font=("Arial", 12))
        self.cpu_label.pack(side=tk.LEFT, padx=10)
        
        # Status log with size limit
        log_frame = ttk.Frame(status_frame)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        ttk.Label(log_frame, text="Status Log:").pack(anchor=tk.W)
        
        self.status_text = tk.Text(log_frame, height=15, bg='#2d2d2d', fg='white', 
                                  font=("Consolas", 9), wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=scrollbar.set)
        
        self.status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # File operations
        file_frame = ttk.LabelFrame(dashboard_frame, text="File Operations", padding=10)
        file_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(file_frame, text="Save State", command=self.save_model_state).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="Load State", command=self.load_model_state).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="Load Luna-Trained Model", command=self.load_luna_trained_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="Export Memories", command=self.export_memories).pack(side=tk.LEFT, padx=5)
        
    def create_performance_tab(self):
        """Create performance monitoring tab."""
        perf_frame = ttk.Frame(self.notebook)
        self.notebook.add(perf_frame, text="Performance")
        
        # Performance metrics
        metrics_frame = ttk.LabelFrame(perf_frame, text="Performance Metrics", padding=10)
        metrics_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Real-time metrics
        self.perf_labels = {}
        metrics = ['CPU Usage', 'Memory Usage', 'Cycle Time', 'Queue Size', 'Memory Growth']
        
        for i, metric in enumerate(metrics):
            row = i // 3
            col = i % 3
            
            frame = ttk.Frame(metrics_frame)
            frame.grid(row=row, column=col, padx=10, pady=5, sticky=tk.W)
            
            ttk.Label(frame, text=f"{metric}:").pack(side=tk.LEFT)
            self.perf_labels[metric] = ttk.Label(frame, text="0", font=("Arial", 10, "bold"))
            self.perf_labels[metric].pack(side=tk.LEFT, padx=5)
        
        # Performance charts
        chart_frame = ttk.LabelFrame(perf_frame, text="Performance History", padding=10)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.perf_fig = Figure(figsize=(12, 6), facecolor='#2d2d2d')
        self.perf_ax1 = self.perf_fig.add_subplot(211, facecolor='#2d2d2d')
        self.perf_ax2 = self.perf_fig.add_subplot(212, facecolor='#2d2d2d')
        
        self.perf_ax1.set_title('CPU and Memory Usage', color='white')
        self.perf_ax1.set_ylabel('Usage %', color='white')
        self.perf_ax1.tick_params(colors='white')
        
        self.perf_ax2.set_title('Cycle Time', color='white')
        self.perf_ax2.set_ylabel('Time (ms)', color='white')
        self.perf_ax2.set_xlabel('Cycle', color='white')
        self.perf_ax2.tick_params(colors='white')
        
        self.perf_canvas = FigureCanvasTkAgg(self.perf_fig, chart_frame)
        self.perf_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Performance controls
        controls_frame = ttk.LabelFrame(perf_frame, text="Performance Controls", padding=10)
        controls_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(controls_frame, text="Reset Metrics", command=self.reset_performance_metrics).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Export Performance Data", command=self.export_performance_data).pack(side=tk.LEFT, padx=5)
        
    def setup_style(self):
        """Setup the GUI style."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure dark theme
        style.configure('TNotebook', background='#1e1e1e')
        style.configure('TNotebook.Tab', background='#2d2d2d', foreground='white')
        style.map('TNotebook.Tab', background=[('selected', '#404040')])
        
    def start_model(self):
        """Start the H.I.M. model with optimization."""
        if self.is_running:
            return
            
        self.is_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        # Start model thread
        self.model_thread = threading.Thread(target=self.run_model_cycles, daemon=True)
        self.model_thread.start()
        
        # Start GUI update loop
        self.start_update_loop()
        
        self.log_status("H.I.M. Model started with optimizations.")
        
    def stop_model(self):
        """Stop the H.I.M. model safely."""
        self.is_running = False
        
        # Wait for thread to finish
        if self.model_thread and self.model_thread.is_alive():
            self.model_thread.join(timeout=2.0)
        
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        
        # Force garbage collection
        self.force_gc()
        
        self.log_status("H.I.M. Model stopped safely.")
        
    def run_model_cycles(self):
        """Optimized model cycle runner."""
        while self.is_running:
            try:
                cycle_start = time.time()
                
                # Check resource limits
                if not self.check_resource_limits():
                    self.log_status("Resource limits exceeded, pausing...")
                    time.sleep(5.0)
                    continue
                
                # Throttle cycles
                if not self.should_run_cycle():
                    time.sleep(0.1)
                    continue
                
                # Generate optimized test data
                test_image = np.random.rand(240, 320, 3) * 0.5  # Smaller image
                mouse_pos = {"x": np.random.randint(0, 320), "y": np.random.randint(0, 240)}
                screen_text = f"Optimized Cycle {self.cycle_count + 1}"
                
                # Run H.I.M. cycle
                result = self.him_model.run_cycle(test_image, mouse_pos, screen_text)
                
                # Add performance data
                cycle_time = (time.time() - cycle_start) * 1000  # ms
                result['cycle_time'] = cycle_time
                result['memory_usage'] = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                result['cpu_usage'] = psutil.cpu_percent()
                
                # Queue update (non-blocking)
                try:
                    self.message_queue.put_nowait(('update_data', result))
                except queue.Full:
                    # Drop oldest message if queue is full
                    try:
                        self.message_queue.get_nowait()
                        self.message_queue.put_nowait(('update_data', result))
                    except queue.Empty:
                        pass
                
                self.cycle_count += 1
                self.cycle_times.append(cycle_time)
                
                # Memory management
                if self.cycle_count % self.gc_interval == 0 and self.gc_var.get():
                    gc.collect()
                    self.last_gc = self.cycle_count
                
                # Auto-save
                if self.auto_save_enabled and time.time() - self.last_auto_save > self.auto_save_interval:
                    self.auto_save_state()
                    self.last_auto_save = time.time()
                
                # Adaptive delay
                delay = self.calculate_adaptive_delay(cycle_time)
                time.sleep(delay)
                
            except Exception as e:
                self.log_status(f"Error in cycle {self.cycle_count}: {e}")
                # Implement crash recovery
                if self.crash_recovery_enabled:
                    self.handle_crash_recovery(e)
                else:
                    break
                    
    def check_resource_limits(self):
        """Check if system resources are within limits."""
        try:
            # Check memory usage
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            if memory_mb > self.max_memory_mb:
                return False
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent > 90:  # High CPU usage
                return False
                
            return True
        except:
            return True  # If we can't check, assume it's okay
            
    def should_run_cycle(self):
        """Determine if we should run another cycle based on performance."""
        if len(self.cycle_times) < 10:
            return True
            
        # Check recent cycle times
        recent_times = self.cycle_times[-10:]
        avg_time = sum(recent_times) / len(recent_times)
        
        # If cycles are taking too long, slow down
        if avg_time > 1000:  # 1 second
            return False
            
        # Check cycles per minute
        current_time = time.time()
        if current_time - self.last_cycle_time < 60:  # Within last minute
            cycles_this_minute = sum(1 for t in self.cycle_times if current_time - t < 60)
            if cycles_this_minute >= self.max_cycles_per_minute:
                return False
                
        return True
        
    def calculate_adaptive_delay(self, cycle_time):
        """Calculate adaptive delay based on cycle performance."""
        base_delay = self.cycle_delay
        
        # Increase delay if cycles are slow
        if cycle_time > 500:  # 500ms
            base_delay *= 1.5
        elif cycle_time > 1000:  # 1s
            base_delay *= 2.0
            
        # Check memory usage
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        if memory_mb > self.max_memory_mb * 0.8:  # 80% of limit
            base_delay *= 1.5
            
        return min(base_delay, 10.0)  # Cap at 10 seconds
        
    def start_update_loop(self):
        """Start the optimized GUI update loop."""
        self.update_gui()
        self.update_performance_monitor()
        self.root.after(self.update_interval, self.start_update_loop)
        
    def update_gui(self):
        """Optimized GUI update with batching."""
        try:
            updates_processed = 0
            max_updates_per_cycle = 5  # Limit updates per cycle
            
            while updates_processed < max_updates_per_cycle:
                try:
                    message_type, data = self.message_queue.get_nowait()
                    
                    if message_type == 'update_data':
                        self.pending_updates.append(data)
                    elif message_type == 'error':
                        self.log_status(f"Error: {data}")
                        
                    updates_processed += 1
                except queue.Empty:
                    break
                    
            # Process batched updates
            if self.pending_updates:
                self.process_batched_updates()
                
        except Exception as e:
            self.log_status(f"GUI update error: {e}")
            
        self.root.after(self.update_interval, self.update_gui)
        
    def process_batched_updates(self):
        """Process batched GUI updates efficiently."""
        if not self.pending_updates:
            return
            
        # Get the most recent update
        latest_update = self.pending_updates[-1]
        self.pending_updates.clear()
        
        # Update GUI with latest data
        self.update_dashboard(latest_update)
        self.update_performance_metrics(latest_update)
        
    def update_dashboard(self, result):
        """Update dashboard with cycle results."""
        try:
            self.cycle_label.config(text=f"Cycle: {result['cycle']}")
            self.state_label.config(text=f"State: {result['system_state']}")
            
            decision = result['model_decision'][:50] + "..." if len(result['model_decision']) > 50 else result['model_decision']
            self.decision_label.config(text=f"Decision: {decision}")
            
            # Update memory and CPU labels
            memory_mb = result.get('memory_usage', 0)
            cpu_percent = result.get('cpu_usage', 0)
            
            self.memory_label.config(text=f"Memory: {memory_mb:.1f} MB")
            self.cpu_label.config(text=f"CPU: {cpu_percent:.1f}%")
            
            # Update status log (limit size)
            timestamp = datetime.now().strftime("%H:%M:%S")
            cycle_time = result.get('cycle_time', 0)
            status_msg = f"[{timestamp}] Cycle {result['cycle']}: {cycle_time:.1f}ms\n"
            
            self.status_text.insert(tk.END, status_msg)
            self.status_text.see(tk.END)
            
            # Keep only last 30 lines
            lines = self.status_text.get("1.0", tk.END).split('\n')
            if len(lines) > 30:
                self.status_text.delete("1.0", f"{len(lines)-30}.0")
                
        except Exception as e:
            self.log_status(f"Dashboard update error: {e}")
            
    def update_performance_metrics(self, result):
        """Update performance metrics."""
        try:
            # Update real-time metrics
            self.perf_labels['CPU Usage'].config(text=f"{result.get('cpu_usage', 0):.1f}%")
            self.perf_labels['Memory Usage'].config(text=f"{result.get('memory_usage', 0):.1f} MB")
            self.perf_labels['Cycle Time'].config(text=f"{result.get('cycle_time', 0):.1f} ms")
            self.perf_labels['Queue Size'].config(text=str(self.message_queue.qsize()))
            
            # Update history
            self.cpu_usage_history.append(result.get('cpu_usage', 0))
            self.memory_usage_history.append(result.get('memory_usage', 0))
            
            # Keep only last 100 points
            if len(self.cpu_usage_history) > 100:
                self.cpu_usage_history = self.cpu_usage_history[-100:]
                self.memory_usage_history = self.memory_usage_history[-100:]
                
        except Exception as e:
            self.log_status(f"Performance update error: {e}")
            
    def update_performance_monitor(self):
        """Update performance monitoring."""
        try:
            # Update performance charts periodically
            if self.cycle_count % 10 == 0:  # Every 10 cycles
                self.update_performance_charts()
                
        except Exception as e:
            self.log_status(f"Performance monitor error: {e}")
            
    def update_performance_charts(self):
        """Update performance charts."""
        try:
            if len(self.cpu_usage_history) < 2:
                return
                
            # Clear and redraw charts
            self.perf_ax1.clear()
            self.perf_ax2.clear()
            
            # CPU and Memory chart
            cycles = list(range(len(self.cpu_usage_history)))
            self.perf_ax1.plot(cycles, self.cpu_usage_history, 'r-', label='CPU %', linewidth=1)
            self.perf_ax1.plot(cycles, [m/10 for m in self.memory_usage_history], 'b-', label='Memory/10 MB', linewidth=1)
            self.perf_ax1.set_title('CPU and Memory Usage', color='white')
            self.perf_ax1.set_ylabel('Usage %', color='white')
            self.perf_ax1.legend()
            self.perf_ax1.tick_params(colors='white')
            
            # Cycle time chart
            if len(self.cycle_times) > 1:
                cycle_cycles = list(range(len(self.cycle_times)))
                self.perf_ax2.plot(cycle_cycles, self.cycle_times, 'g-', linewidth=1)
                self.perf_ax2.set_title('Cycle Time', color='white')
                self.perf_ax2.set_ylabel('Time (ms)', color='white')
                self.perf_ax2.set_xlabel('Cycle', color='white')
                self.perf_ax2.tick_params(colors='white')
            
            self.perf_canvas.draw()
            
        except Exception as e:
            self.log_status(f"Chart update error: {e}")
            
    def handle_crash_recovery(self, error):
        """Handle crash recovery."""
        self.log_status(f"Crash detected: {error}")
        self.log_status("Attempting recovery...")
        
        # Save current state
        try:
            self.auto_save_state()
        except:
            pass
            
        # Clear memory
        self.force_gc()
        
        # Reset cycle count
        self.cycle_count = 0
        self.cycle_times.clear()
        
        # Wait before restarting
        time.sleep(2.0)
        
        self.log_status("Recovery complete, resuming...")
        
    def force_gc(self):
        """Force garbage collection."""
        try:
            before = psutil.Process().memory_info().rss / 1024 / 1024
            gc.collect()
            after = psutil.Process().memory_info().rss / 1024 / 1024
            freed = before - after
            self.log_status(f"Garbage collection: freed {freed:.1f} MB")
        except Exception as e:
            self.log_status(f"GC error: {e}")
            
    def clear_memory_cache(self):
        """Clear memory cache."""
        try:
            # Clear H.I.M. model cache if it exists
            if hasattr(self.him_model, 'clear_cache'):
                self.him_model.clear_cache()
                
            # Clear GUI caches
            self.pending_updates.clear()
            
            # Force garbage collection
            self.force_gc()
            
            self.log_status("Memory cache cleared")
        except Exception as e:
            self.log_status(f"Cache clear error: {e}")
            
    def update_delay(self, value):
        """Update cycle delay."""
        self.cycle_delay = float(value)
        self.delay_label.config(text=f"{self.cycle_delay:.1f}s")
        
    def update_performance_mode(self, event):
        """Update performance mode."""
        mode = self.perf_mode.get()
        if mode == "low":
            self.cycle_delay = 5.0
            self.max_cycles_per_minute = 10
            self.update_interval = 500
        elif mode == "balanced":
            self.cycle_delay = 2.0
            self.max_cycles_per_minute = 20
            self.update_interval = 200
        elif mode == "high":
            self.cycle_delay = 0.5
            self.max_cycles_per_minute = 40
            self.update_interval = 100
            
        self.delay_var.set(self.cycle_delay)
        self.delay_label.config(text=f"{self.cycle_delay:.1f}s")
        
    def toggle_auto_save(self):
        """Toggle auto-save."""
        self.auto_save_enabled = self.auto_save_var.get()
        
    def auto_save_state(self):
        """Auto-save model state."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"him_auto_save_{timestamp}.pkl"
            
            with open(filename, 'wb') as f:
                pickle.dump(self.him_model, f)
                
            self.log_status(f"Auto-saved to {filename}")
        except Exception as e:
            self.log_status(f"Auto-save error: {e}")
            
    def save_model_state(self):
        """Save model state."""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".pkl",
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
            )
            if filename:
                with open(filename, 'wb') as f:
                    pickle.dump(self.him_model, f)
                self.log_status(f"Model state saved to {filename}")
        except Exception as e:
            self.log_status(f"Save error: {e}")
            
    def load_model_state(self):
        """Load model state."""
        try:
            filename = filedialog.askopenfilename(
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
            )
            if filename:
                with open(filename, 'rb') as f:
                    self.him_model = pickle.load(f)
                self.log_status(f"Model state loaded from {filename}")
        except Exception as e:
            self.log_status(f"Load error: {e}")
            
    def load_luna_trained_model(self):
        """Load Luna-trained model."""
        try:
            filename = filedialog.askopenfilename(
                title="Select Luna-Trained H.I.M. Model",
                initialdir="him_luna_training",
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
            )
            if filename:
                with open(filename, 'rb') as f:
                    self.him_model = pickle.load(f)
                self.log_status(f"Luna-trained model loaded from {filename}")
        except Exception as e:
            self.log_status(f"Luna model load error: {e}")
            
    def export_memories(self):
        """Export memories to text file."""
        try:
            filename = self.him_model.save_all_memories_to_text()
            self.log_status(f"Memories exported to {filename}")
        except Exception as e:
            self.log_status(f"Export error: {e}")
            
    def reset_performance_metrics(self):
        """Reset performance metrics."""
        self.cycle_times.clear()
        self.cpu_usage_history.clear()
        self.memory_usage_history.clear()
        self.log_status("Performance metrics reset")
        
    def export_performance_data(self):
        """Export performance data."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"him_performance_{timestamp}.json"
            
            import json
            data = {
                'cycle_times': self.cycle_times,
                'cpu_usage': self.cpu_usage_history,
                'memory_usage': self.memory_usage_history,
                'total_cycles': self.cycle_count,
                'export_time': timestamp
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
                
            self.log_status(f"Performance data exported to {filename}")
        except Exception as e:
            self.log_status(f"Performance export error: {e}")
            
    def load_last_state(self):
        """Load last auto-saved state if available."""
        try:
            # Look for the most recent auto-save file
            auto_save_files = [f for f in os.listdir('.') if f.startswith('him_auto_save_') and f.endswith('.pkl')]
            if auto_save_files:
                latest_file = max(auto_save_files)
                with open(latest_file, 'rb') as f:
                    self.him_model = pickle.load(f)
                self.log_status(f"Loaded last state from {latest_file}")
        except Exception as e:
            self.log_status(f"Last state load error: {e}")
            
    def start_performance_monitor(self):
        """Start performance monitoring."""
        def monitor():
            while True:
                try:
                    # Update system metrics
                    cpu_percent = psutil.cpu_percent()
                    memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                    
                    # Check for memory growth
                    if len(self.memory_usage_history) > 0:
                        memory_growth = memory_mb - self.memory_usage_history[0] if self.memory_usage_history else 0
                        self.perf_labels['Memory Growth'].config(text=f"{memory_growth:.1f} MB")
                    
                    time.sleep(5)  # Update every 5 seconds
                except:
                    break
                    
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        
    def log_status(self, message):
        """Log status message."""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_message = f"[{timestamp}] {message}\n"
            
            # Add to status text (thread-safe)
            self.root.after(0, lambda: self._add_log_message(log_message))
        except Exception as e:
            print(f"Log error: {e}")
            
    def _add_log_message(self, message):
        """Add log message to GUI (called from main thread)."""
        try:
            self.status_text.insert(tk.END, message)
            self.status_text.see(tk.END)
        except:
            pass
            
    # Placeholder methods for other tabs (simplified)
    def create_emotions_tab(self):
        """Create emotions tab."""
        emotions_frame = ttk.Frame(self.notebook)
        self.notebook.add(emotions_frame, text="Emotions")
        ttk.Label(emotions_frame, text="Emotions monitoring will be implemented here").pack(pady=20)
        
    def create_hormones_tab(self):
        """Create hormones tab."""
        hormones_frame = ttk.Frame(self.notebook)
        self.notebook.add(hormones_frame, text="Hormones")
        ttk.Label(hormones_frame, text="Hormones monitoring will be implemented here").pack(pady=20)
        
    def create_memory_tab(self):
        """Create memory tab."""
        memory_frame = ttk.Frame(self.notebook)
        self.notebook.add(memory_frame, text="Memory")
        ttk.Label(memory_frame, text="Memory management will be implemented here").pack(pady=20)
        
    def create_chat_tab(self):
        """Create chat tab."""
        chat_frame = ttk.Frame(self.notebook)
        self.notebook.add(chat_frame, text="Chat")
        ttk.Label(chat_frame, text="Chat interface will be implemented here").pack(pady=20)
        
    def create_visual_tab(self):
        """Create visual tab."""
        visual_frame = ttk.Frame(self.notebook)
        self.notebook.add(visual_frame, text="Visual")
        ttk.Label(visual_frame, text="Visual processing will be implemented here").pack(pady=20)

def main():
    """Main function to run the optimized H.I.M. GUI."""
    root = tk.Tk()
    app = OptimizedHIMGUI(root)
    
    # Handle window close
    def on_closing():
        if app.is_running:
            app.stop_model()
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()

