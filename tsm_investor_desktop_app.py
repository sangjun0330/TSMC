#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Desktop launcher for the separate investor-action dashboard."""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

import tsm_dashboard
from tsm_investor_dashboard import INVESTOR_STATIC_DIR


APP_TITLE = "투자 실행 대시보드"
APP_SUPPORT_DIR = Path.home() / "Library" / "Application Support" / "Investor Action Dashboard"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the investor-action dashboard in a desktop window.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8776)
    parser.add_argument("--strict-port", action="store_true")
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=940)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--allow-browser-fallback", action="store_true")
    return parser.parse_args()


def osascript_quote(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def show_macos_alert(title: str, message: str) -> None:
    if sys.platform != "darwin":
        return
    script = f'display alert "{osascript_quote(title)}" message "{osascript_quote(message)}" as warning'
    try:
        subprocess.run(["osascript", "-e", script], check=False, timeout=5)
    except Exception:
        pass


def block_until_interrupted() -> None:
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        return


def main() -> int:
    args = parse_args()
    tsm_dashboard.STATIC_DIR = INVESTOR_STATIC_DIR
    try:
        import webview
    except ImportError:
        message = "pywebview가 설치되어 있지 않아 투자 실행 대시보드 앱 창을 열 수 없습니다. .venv/bin/python -m pip install -r requirements.txt 를 먼저 실행해 주세요."
        print(message, file=sys.stderr)
        show_macos_alert(APP_TITLE, message)
        if not args.allow_browser_fallback:
            return 2

        import webbrowser

        server, port = tsm_dashboard.bind_server(args.host, args.port, args.strict_port)
        url = f"http://{args.host}:{port}"
        thread = threading.Thread(target=server.serve_forever, name="investor-dashboard-http", daemon=True)
        thread.start()
        print(f"Investor dashboard fallback server running at {url}")
        try:
            webbrowser.open(url)
            block_until_interrupted()
        finally:
            server.shutdown()
            server.server_close()
        return 1

    server, port = tsm_dashboard.bind_server(args.host, args.port, args.strict_port)
    url = f"http://{args.host}:{port}"
    thread = threading.Thread(target=server.serve_forever, name="investor-dashboard-http", daemon=True)
    thread.start()
    threading.Thread(target=tsm_dashboard.prewarm_investor_dashboard, name="investor-dashboard-prewarm", daemon=True).start()
    print(f"Investor dashboard desktop app running with internal server at {url}")

    APP_SUPPORT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        webview.create_window(
            APP_TITLE,
            url,
            width=args.width,
            height=args.height,
            min_size=(980, 680),
            resizable=True,
            confirm_close=False,
            background_color="#f5f7fa",
            text_select=True,
            zoomable=True,
        )
        webview.start(debug=args.debug, private_mode=False, storage_path=str(APP_SUPPORT_DIR))
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
