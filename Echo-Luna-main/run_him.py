from __future__ import annotations

"""Entry point for launching the CognitiveEngine++ server and GUI."""

import argparse
import sys
from threading import Thread
from typing import Optional

import uvicorn

from him.cli.server import build_app, serve
from him.gui.app import launch as launch_gui


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the CognitiveEngine++ stack")
    parser.add_argument("--data-dir", default="./data", help="Directory for HierarchicalImageMemory data")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn autoreload")
    parser.add_argument("--with-agent", action=argparse.BooleanOptionalAction, default=True, help="Mount agent routes")
    parser.add_argument("--open-gui", action="store_true", help="Launch the desktop GUI after starting the API")
    args = parser.parse_args(argv)

    if args.open_gui:
        app, _, store = build_app(args.data_dir, with_agent=args.with_agent)
        config = uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
        server = uvicorn.Server(config)
        thread = Thread(target=server.run, daemon=True)
        thread.start()
        try:
            exit_code = launch_gui(store, with_agent=args.with_agent, base_url=f"http://{args.host}:{args.port}")
        finally:
            server.should_exit = True
            thread.join(timeout=2.0)
        return exit_code

    serve(
        args.data_dir,
        host=args.host,
        port=args.port,
        with_agent=args.with_agent,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

