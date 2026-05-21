#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${ROOT_DIR}/tsm_price_rule_output/logs"
mkdir -p "${LOG_DIR}"
LOCK_DIR="${ROOT_DIR}/tsm_price_rule_output/.daily_update.lock"

if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON="${ROOT_DIR}/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

STAMP="$(TZ=Asia/Seoul date '+%Y%m%d_%H%M%S')"
LOG_FILE="${LOG_DIR}/daily_update_${STAMP}.log"
RUN_DATE="$(TZ=Asia/Seoul date '+%Y-%m-%d')"

cd "${ROOT_DIR}"

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "Another daily update is already running. Lock: ${LOCK_DIR}"
  exit 0
fi
trap 'rm -rf "${LOCK_DIR}"' EXIT

run_step() {
  echo
  echo "==> $*"
  "$@"
}

{
  echo "TSMC full daily update started at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "Asia/Seoul run date: ${RUN_DATE}"

  run_step "${PYTHON}" run_pooled_universe_update.py \
    --start 2016-05-12 \
    --end "${RUN_DATE}" \
    --preferred-source yahoo

  run_step "${PYTHON}" run_daily_update.py \
    --config config/tsm_research.toml \
    --start 2016-05-12 \
    --end "${RUN_DATE}" \
    --preferred-source yahoo \
    --output-dir output \
    --rule-outdir tsm_price_rule_output \
    "$@"

  run_step "${PYTHON}" tsm_hourly_quant_pipeline.py \
    --start 2016-05-12 \
    --end "${RUN_DATE}" \
    --outdir output \
    --provider auto

  run_step "${PYTHON}" tsm_intraday_quant_pipeline.py \
    --interval 1m \
    --start 2016-05-12 \
    --end "${RUN_DATE}" \
    --outdir output

  echo
  echo "TSMC full daily update completed at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
} >"${LOG_FILE}" 2>&1
