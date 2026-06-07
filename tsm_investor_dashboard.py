#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HTTP entrypoint for the separate investor-action dashboard."""

from __future__ import annotations

import argparse
import threading
import webbrowser
from pathlib import Path

import tsm_dashboard


ROOT = Path(__file__).resolve().parent
INVESTOR_STATIC_DIR = ROOT / "investor_dashboard"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the separate investor-action dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8776)
    parser.add_argument("--strict-port", action="store_true")
    parser.add_argument("--open", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tsm_dashboard.STATIC_DIR = INVESTOR_STATIC_DIR
    server, port = tsm_dashboard.bind_server(args.host, args.port, args.strict_port)
    url = f"http://{args.host}:{port}"
    threading.Thread(target=tsm_dashboard.prewarm_investor_dashboard, name="investor-dashboard-prewarm", daemon=True).start()
    print(f"Investor dashboard running at {url}")
    print("Press Ctrl+C to stop.")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping investor dashboard...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
