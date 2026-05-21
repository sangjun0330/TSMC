#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

"$PYTHON_BIN" run_daily_update.py \
  --start 2016-05-12 \
  --end 2026-05-12 \
  --output-dir output \
  --rule-outdir tsm_price_rule_output
