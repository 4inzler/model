#!/usr/bin/env python3
"""
Optimized H.I.M. GUI Launcher
============================

This launcher script starts the optimized H.I.M. GUI with crash prevention
and performance optimizations to prevent GUI freezing and system crashes.
"""

import sys
import os
import subprocess
import time
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'tkinter',
        'numpy',
        'matplotlib',
        'psutil',
        'threading',
        'queue'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'tkinter':
                import tkinter
            elif package == 'threading':
                import threading
            elif package == 'queue':
                import queue
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install missing packages and try again.")
        return False
    
    return True

def check_him_model():
    """Check if H.I.M. model files exist."""
    required_files = [
        'him_model.py',
        'him_gui_optimized.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    return True

def launch_optimized_gui():
    """Launch the optimized H.I.M. GUI."""
    print("=" * 60)
    print("H.I.M. MODEL - OPTIMIZED GUI LAUNCHER")
    print("=" * 60)
    print()
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        input("Press Enter to exit...")
        return False
    
    print("✓ All dependencies found")
    
    # Check H.I.M. model files
    print("Checking H.I.M. model files...")
    if not check_him_model():
        input("Press Enter to exit...")
        return False
    
    print("✓ All H.I.M. model files found")
    
    # Check for Luna training data
    luna_training_dir = "him_luna_training"
    if os.path.exists(luna_training_dir):
        luna_files = [f for f in os.listdir(luna_training_dir) if f.endswith('.pkl')]
        if luna_files:
            print(f"✓ Found {len(luna_files)} Luna-trained model(s)")
        else:
            print("! Luna training directory exists but no models found")
    else:
        print("! No Luna training data found - you can train a model later")
    
    print()
    print("Starting optimized H.I.M. GUI...")
    print("Features:")
    print("  • Crash prevention and recovery")
    print("  • Memory management and garbage collection")
    print("  • Performance monitoring and throttling")
    print("  • Adaptive cycle delays")
    print("  • Resource limit monitoring")
    print("  • Auto-save and state recovery")
    print()
    
    try:
        # Launch the optimized GUI
        import him_gui_optimized
        him_gui_optimized.main()
        
    except KeyboardInterrupt:
        print("\nGUI closed by user")
    except Exception as e:
        print(f"\nError launching GUI: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure all dependencies are installed")
        print("2. Check that him_model.py exists and is working")
        print("3. Try running: python him_gui_optimized.py")
        input("\nPress Enter to exit...")
        return False
    
    return True

def show_performance_tips():
    """Show performance optimization tips."""
    print("\n" + "=" * 60)
    print("PERFORMANCE OPTIMIZATION TIPS")
    print("=" * 60)
    print()
    print("To prevent GUI freezing and crashes:")
    print()
    print("1. PERFORMANCE MODES:")
    print("   • Low: 5s delay, 10 cycles/min - Most stable")
    print("   • Balanced: 2s delay, 20 cycles/min - Good balance")
    print("   • High: 0.5s delay, 40 cycles/min - Fast but resource-intensive")
    print()
    print("2. MEMORY MANAGEMENT:")
    print("   • Enable 'Auto Garbage Collection'")
    print("   • Use 'Force Garbage Collection' if memory usage is high")
    print("   • Clear memory cache periodically")
    print("   • Monitor memory usage in Performance tab")
    print()
    print("3. CYCLE THROTTLING:")
    print("   • Increase cycle delay if system becomes unresponsive")
    print("   • System automatically throttles if CPU/memory usage is high")
    print("   • Watch cycle times - if >1000ms, system will slow down")
    print()
    print("4. CRASH RECOVERY:")
    print("   • Auto-save is enabled by default")
    print("   • System will attempt recovery if crashes occur")
    print("   • Last state is automatically loaded on restart")
    print()
    print("5. MONITORING:")
    print("   • Check Performance tab for real-time metrics")
    print("   • Watch for memory growth over time")
    print("   • Export performance data for analysis")
    print()

if __name__ == "__main__":
    # Show performance tips
    show_performance_tips()
    
    # Launch the GUI
    success = launch_optimized_gui()
    
    if not success:
        sys.exit(1)

