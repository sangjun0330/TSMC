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
  echo "Top10+2 full daily update started at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "Asia/Seoul run date: ${RUN_DATE}"

  run_step "${PYTHON}" run_daily_update.py \
    --config config/tsm_research.toml \
    --start 2016-05-12 \
    --end "${RUN_DATE}" \
    --preferred-source yahoo \
    --output-dir output \
    --rule-outdir tsm_price_rule_output \
    --universe-mode hybrid \
    --universe-config config/semiconductor_universe_top10.csv \
    "$@"

  run_step "${PYTHON}" run_universe_market_data_update.py \
    --universe-config config/semiconductor_universe_top10.csv \
    --start 2016-05-12 \
    --end "${RUN_DATE}" \
    --bar-scope both \
    --provider auto \
    --skip-charts \
    --continue-on-error

  run_step "${PYTHON}" tsm_full_daily_update_audit.py \
    --output-dir output \
    --rule-outdir tsm_price_rule_output \
    --run-date "${RUN_DATE}" \
    --log-file "${LOG_FILE}"

  echo
  echo "==> verify Top12 automation outputs"
  "${PYTHON}" - <<'PY'
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd


ROOT = Path.cwd()
RULE_OUT = ROOT / "tsm_price_rule_output"
CONFIG = ROOT / "config" / "semiconductor_universe_top10.csv"
KOREAN_SYMBOLS = {"005930.KS", "000660.KS"}
EXPECTED_SENSITIVITY_ROWS = 625


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "pass", "passed"}


def load_csv(path: Path, failures: list[str]) -> pd.DataFrame:
    if not path.exists():
        failures.append(f"missing file: {path.relative_to(ROOT)}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - automation guardrail
        failures.append(f"cannot read {path.relative_to(ROOT)}: {exc}")
        return pd.DataFrame()


def symbol_set(df: pd.DataFrame) -> set[str]:
    if "symbol" not in df.columns:
        return set()
    return set(df["symbol"].dropna().astype(str).str.strip())


def enabled_symbols(failures: list[str]) -> list[str]:
    if not CONFIG.exists():
        failures.append(f"missing file: {CONFIG.relative_to(ROOT)}")
        return []
    with CONFIG.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    symbols = [
        str(row.get("symbol", "")).strip()
        for row in rows
        if truthy(row.get("enabled")) and str(row.get("symbol", "")).strip()
    ]
    return symbols


failures: list[str] = []
required_symbols = enabled_symbols(failures)
required_set = set(required_symbols)

if len(required_symbols) != 12:
    failures.append(f"enabled decision symbol count is {len(required_symbols)}, expected 12")
if not KOREAN_SYMBOLS.issubset(required_set):
    failures.append(
        "Korean decision symbols missing from config: "
        + ",".join(sorted(KOREAN_SYMBOLS - required_set))
    )

coverage_files = {
    "pooled_feature_matrix": RULE_OUT / "tsm_prediction_pooled_feature_matrix.csv",
    "pooled_label_dataset": RULE_OUT / "tsm_prediction_pooled_label_dataset.csv",
    "latest_predictions": RULE_OUT / "tsm_universe_latest_predictions.csv",
}

for label, path in coverage_files.items():
    df = load_csv(path, failures)
    present = symbol_set(df)
    if "symbol" not in df.columns and not df.empty:
        failures.append(f"{label} has no symbol column")
        continue
    missing = sorted(required_set - present)
    if missing:
        failures.append(f"{label} missing decision symbols: {','.join(missing)}")
    missing_kr = sorted(KOREAN_SYMBOLS - present)
    if missing_kr:
        failures.append(f"{label} missing Korean symbols: {','.join(missing_kr)}")
    if label == "latest_predictions" and "prediction_source" in df.columns:
        latest_decision = df[df["symbol"].astype(str).isin(required_set)].copy()
        fallback_rows = latest_decision[
            latest_decision["prediction_source"].astype(str).str.contains("fallback", case=False, na=False)
        ]
        if not fallback_rows.empty:
            failures.append("latest predictions contains rule fallback rows for decision symbols")

quality_files = [
    RULE_OUT / "tsm_prediction_pooled_quality_checks.csv",
    RULE_OUT / "tsm_pooled_model_quality_checks.csv",
    RULE_OUT / "tsm_rule_threshold_sensitivity_quality_checks.csv",
    RULE_OUT / "tsm_automation_quality_checks.csv",
]
for path in quality_files:
    df = load_csv(path, failures)
    if df.empty:
        continue
    required_columns = {"check", "passed", "severity"}
    if not required_columns.issubset(df.columns):
        failures.append(f"{path.relative_to(ROOT)} missing quality columns")
        continue
    critical = df[df["severity"].astype(str).str.upper() == "CRITICAL"]
    failed = critical[~critical["passed"].map(truthy)]
    if not failed.empty:
        checks = ",".join(failed["check"].astype(str).head(8))
        failures.append(f"{path.relative_to(ROOT)} has failed CRITICAL checks: {checks}")

sensitivity = load_csv(RULE_OUT / "tsm_rule_threshold_sensitivity.csv", failures)
if not sensitivity.empty and len(sensitivity) != EXPECTED_SENSITIVITY_ROWS:
    failures.append(
        f"threshold sensitivity row count is {len(sensitivity)}, expected {EXPECTED_SENSITIVITY_ROWS}"
    )
summary = load_csv(RULE_OUT / "tsm_rule_threshold_sensitivity_summary.csv", failures)
if not summary.empty and "recommendation" in summary.columns:
    recommendations = set(summary["recommendation"].dropna().astype(str))
    allowed = {"KEEP_CURRENT", "REVIEW_LOOSER", "REVIEW_TIGHTER"}
    invalid = sorted(recommendations - allowed)
    if invalid:
        failures.append(f"invalid threshold sensitivity recommendations: {','.join(invalid)}")

risk = load_csv(RULE_OUT / "tsm_portfolio_risk_order_decisions.csv", failures)
if not risk.empty:
    required_risk_columns = {
        "block_reason",
        "warning_reasons",
        "beta_to_smh_warning",
        "smh_beta_limit_mode",
    }
    missing_cols = sorted(required_risk_columns - set(risk.columns))
    if missing_cols:
        failures.append(f"portfolio risk decisions missing columns: {','.join(missing_cols)}")
    if "smh_beta_limit_mode" in risk.columns:
        modes = set(risk["smh_beta_limit_mode"].dropna().astype(str))
        if modes and modes != {"WARN"}:
            failures.append(f"unexpected SMH beta limit modes: {','.join(sorted(modes))}")

if failures:
    print("Top12 automation verification FAILED")
    for failure in failures:
        print(f"CRITICAL: {failure}")
    sys.exit(1)

print("Top12 automation verification PASS")
print("decision_symbols=" + ",".join(required_symbols))
print("korean_symbols=005930.KS,000660.KS")
print("threshold_sensitivity_rows=" + str(len(sensitivity)))
print("live_trading_status=DISABLED_BY_DESIGN")
PY

  echo
  echo "Top10+2 full daily update completed at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
} >"${LOG_FILE}" 2>&1
