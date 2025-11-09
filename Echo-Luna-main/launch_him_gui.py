#!/usr/bin/env python3
"""
H.I.M. Model GUI Launcher
========================

Simple launcher script for the H.I.M. Model GUI with error handling
and dependency checking.
"""

import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

Dependency = Tuple[str, str, Optional[str]]

REQUIRED_DEPENDENCIES: Sequence[Dependency] = [
    ("tkinter", "tkinter", None),
    ("numpy", "numpy", "numpy"),
    ("Pillow", "PIL", "Pillow"),
    ("matplotlib", "matplotlib", "matplotlib"),
    ("psutil", "psutil", "psutil"),
]

OPTIONAL_DEPENDENCIES: Sequence[Dependency] = [
    ("torch", "torch", "torch"),
]

LINUX_HINTS = {
    "tkinter": "Linux tip: install the python3-tk package (e.g. `sudo apt install python3-tk`).",
}

SKIP_BUNDLED_FLAG = "--skip-bundled-env"


def _check_dependency(dep: Dependency) -> bool:
    """Return True if the dependency can be imported."""
    display_name, module_name, _ = dep
    try:
        importlib.import_module(module_name)
    except ImportError:
        print(f"✗ {display_name} is missing")
        return False
    else:
        print(f"✓ {display_name} is installed")
        return True


def check_dependencies() -> Tuple[List[Dependency], List[Dependency]]:
    """Split dependencies into missing required and optional lists."""
    missing_required: List[Dependency] = []
    missing_optional: List[Dependency] = []

    for dep in REQUIRED_DEPENDENCIES:
        if not _check_dependency(dep):
            missing_required.append(dep)

    for dep in OPTIONAL_DEPENDENCIES:
        if not _check_dependency(dep):
            missing_optional.append(dep)

    return missing_required, missing_optional


def install_package(dep: Dependency) -> bool:
    """Install a package using pip."""
    display_name, _module_name, pip_name = dep
    if pip_name is None:
        print(f"Cannot auto-install {display_name}. Please use your system package manager.")
        return False

    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pip_name])
        return True
    except subprocess.CalledProcessError:
        return False


def get_bundled_python() -> Optional[Path]:
    """Return the bundled virtualenv interpreter if present."""
    base_dir = Path(__file__).resolve().parent
    candidates = [
        base_dir / 'echo_luna' / 'bin' / 'python',
        base_dir / 'echo_luna' / 'Scripts' / 'python.exe',
    ]

    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
    return None


def relaunch_with_bundled_python(skip_flag: str) -> None:
    """Relaunch the launcher using the bundled Python interpreter if available."""
    bundled_python = get_bundled_python()
    if bundled_python is None:
        return

    current_python = Path(sys.executable).resolve()
    if current_python == bundled_python.resolve():
        return

    print("\nDetected bundled Python environment. Relaunching with it...")
    args = [str(bundled_python), str(Path(__file__).resolve()), skip_flag]
    args.extend(arg for arg in sys.argv[1:] if arg != skip_flag)
    os.execv(str(bundled_python), args)


def prompt_for_install() -> bool:
    """Return True if the user agrees to install missing packages."""
    if not sys.stdin.isatty():
        return False

    try:
        response = input("Would you like to install them automatically? (y/n): ")
    except EOFError:
        return False

    return response.strip().lower() in {"y", "yes"}


def print_install_instructions(missing: Sequence[Dependency]) -> None:
    """Show manual installation instructions for the missing packages."""
    pip_packages = sorted({dep[2] for dep in missing if dep[2]})
    if pip_packages:
        package_list = ' '.join(pip_packages)
        print("\nInstall the missing packages with:")
        print(f"    pip install {package_list}")

    shown_hints = set()
    for display_name, _, _ in missing:
        hint = LINUX_HINTS.get(display_name)
        if hint and hint not in shown_hints:
            print(f"\n{hint}")
            shown_hints.add(hint)


def main() -> None:
    """Main launcher function."""
    if SKIP_BUNDLED_FLAG in sys.argv:
        sys.argv.remove(SKIP_BUNDLED_FLAG)

    print("H.I.M. Model GUI Launcher")
    print("=" * 30)

    print("\nChecking dependencies...")
    missing_required, missing_optional = check_dependencies()

    if missing_optional:
        optional_names = ', '.join(dep[0] for dep in missing_optional)
        print(f"\nOptional packages missing: {optional_names}")
        print("The GUI will run without them, but some features may be disabled.")

    if missing_required:
        missing_names = ', '.join(dep[0] for dep in missing_required)
        print(f"\nMissing packages: {missing_names}")

        if SKIP_BUNDLED_FLAG not in sys.argv:
            relaunch_with_bundled_python(SKIP_BUNDLED_FLAG)

        if prompt_for_install():
            install_failures: List[Dependency] = []
            for dep in missing_required:
                display_name, _, _ = dep
                print(f"Installing {display_name}...")
                if install_package(dep):
                    print(f"✓ {display_name} installed successfully")
                else:
                    print(f"✗ Failed to install {display_name}")
                    install_failures.append(dep)

            if install_failures:
                print("\nSome dependencies could not be installed automatically.")
                print_install_instructions(install_failures)
                return

            print("\nRe-checking dependencies...")
            missing_required, missing_optional = check_dependencies()
            if missing_required:
                print("\nDependencies are still missing after installation. Please install them manually.")
                print_install_instructions(missing_required)
                return
        else:
            print_install_instructions(missing_required)
            return

    print("\nAll required dependencies satisfied!")
    if missing_optional:
        print("Note: optional packages remain missing.")
    print("Launching H.I.M. Model GUI...")

    try:
        from him_gui import main as gui_main
        gui_main()
    except Exception as exc:
        print(f"Error launching GUI: {exc}")
        print("Please check that all files are in the correct location.")


if __name__ == "__main__":
    main()
