#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LABEL="com.osangjun.tsm-daily-update"
PLIST_DIR="${HOME}/Library/LaunchAgents"
PLIST_PATH="${PLIST_DIR}/${LABEL}.plist"
TEMPLATE="${SCRIPT_DIR}/${LABEL}.plist.template"

mkdir -p "${PLIST_DIR}" "${ROOT_DIR}/tsm_price_rule_output/logs"
sed "s#__ROOT__#${ROOT_DIR}#g" "${TEMPLATE}" > "${PLIST_PATH}"
chmod +x "${ROOT_DIR}/script/run_daily_update_once.sh"

DOMAIN="gui/$(id -u)"
if launchctl print "${DOMAIN}/${LABEL}" >/dev/null 2>&1; then
  launchctl bootout "${DOMAIN}" "${PLIST_PATH}" || true
fi
launchctl bootstrap "${DOMAIN}" "${PLIST_PATH}"

echo "Installed ${LABEL}"
echo "Plist: ${PLIST_PATH}"
echo "Schedule: Tue-Sat 07:30 local time, run_daily_update.py --universe-mode hybrid"
