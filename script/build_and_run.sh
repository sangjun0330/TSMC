#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-run}"
APP_NAME="TSMC Dashboard"
BUNDLE_ID="com.local.tsm-dashboard"
MIN_SYSTEM_VERSION="13.0"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="$ROOT_DIR/dist"
APP_BUNDLE="$DIST_DIR/$APP_NAME.app"
APP_CONTENTS="$APP_BUNDLE/Contents"
APP_MACOS="$APP_CONTENTS/MacOS"
APP_EXECUTABLE="$APP_MACOS/tsm-dashboard"
INFO_PLIST="$APP_CONTENTS/Info.plist"
LOG_DIR="$HOME/Library/Logs/$APP_NAME"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

usage() {
  echo "usage: $0 [run|--debug|--logs|--telemetry|--verify]" >&2
}

ensure_python() {
  if [[ ! -x "$PYTHON_BIN" ]]; then
    if ! command -v python3 >/dev/null 2>&1; then
      echo "python3 is required to run $APP_NAME." >&2
      exit 1
    fi
    python3 -m venv "$ROOT_DIR/.venv"
  fi
}

ensure_dependencies() {
  if ! "$PYTHON_BIN" - <<'PY'
import importlib.util
import sys

required = ("pandas", "numpy", "requests", "matplotlib", "sklearn", "lightgbm", "xgboost", "optuna", "webview")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print("missing Python packages: " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(1)
PY
  then
    "$PYTHON_BIN" -m pip install -r "$ROOT_DIR/requirements.txt"
  fi
}

kill_running() {
  local pids
  pids="$(pgrep -f "$ROOT_DIR/tsm_desktop_app.py" || true)"
  if [[ -n "$pids" ]]; then
    kill $pids >/dev/null 2>&1 || true
    sleep 0.5
  fi
}

stage_app_bundle() {
  rm -rf "$APP_BUNDLE"
  mkdir -p "$APP_MACOS" "$LOG_DIR"

  cat >"$APP_EXECUTABLE" <<APP
#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$ROOT_DIR"
PYTHON_BIN="\$PROJECT_ROOT/.venv/bin/python"
LOG_DIR="\$HOME/Library/Logs/$APP_NAME"

mkdir -p "\$LOG_DIR"
exec >>"\$LOG_DIR/app.log" 2>&1
cd "\$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
exec "\$PYTHON_BIN" "\$PROJECT_ROOT/tsm_desktop_app.py" "\$@"
APP
  chmod +x "$APP_EXECUTABLE"

  cat >"$INFO_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleExecutable</key>
  <string>tsm-dashboard</string>
  <key>CFBundleIdentifier</key>
  <string>$BUNDLE_ID</string>
  <key>CFBundleName</key>
  <string>$APP_NAME</string>
  <key>CFBundleDisplayName</key>
  <string>$APP_NAME</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>0.1.0</string>
  <key>CFBundleVersion</key>
  <string>1</string>
  <key>LSMinimumSystemVersion</key>
  <string>$MIN_SYSTEM_VERSION</string>
  <key>LSApplicationCategoryType</key>
  <string>public.app-category.finance</string>
  <key>NSHighResolutionCapable</key>
  <true/>
  <key>NSPrincipalClass</key>
  <string>NSApplication</string>
</dict>
</plist>
PLIST
}

open_app() {
  /usr/bin/open -n "$APP_BUNDLE"
}

run_logs() {
  touch "$LOG_DIR/app.log"
  tail -f "$LOG_DIR/app.log"
}

ensure_python
ensure_dependencies
kill_running
stage_app_bundle

case "$MODE" in
  run)
    open_app
    ;;
  --debug|debug)
    "$PYTHON_BIN" "$ROOT_DIR/tsm_desktop_app.py" --debug
    ;;
  --logs|logs)
    open_app
    run_logs
    ;;
  --telemetry|telemetry)
    open_app
    /usr/bin/log stream --info --style compact --predicate "process CONTAINS \"Python\" OR process CONTAINS \"tsm-dashboard\""
    ;;
  --verify|verify)
    open_app
    sleep 3
    if pgrep -f "$ROOT_DIR/tsm_desktop_app.py" >/dev/null; then
      echo "$APP_NAME is running from $APP_BUNDLE"
    else
      echo "$APP_NAME did not start. Recent log output:" >&2
      tail -40 "$LOG_DIR/app.log" >&2 || true
      exit 1
    fi
    ;;
  *)
    usage
    exit 2
    ;;
esac
