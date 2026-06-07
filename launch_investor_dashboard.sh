#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -n "${PYTHON:-}" ]]; then
  exec "$PYTHON" tsm_investor_desktop_app.py "$@"
fi

if [[ -x ".venv/bin/python" ]]; then
  exec ".venv/bin/python" tsm_investor_desktop_app.py "$@"
fi

exec python3 tsm_investor_desktop_app.py "$@"
