#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${ROOT_DIR}/tsm_price_rule_output/logs"
mkdir -p "${LOG_DIR}"
LOCK_DIR="${ROOT_DIR}/tsm_price_rule_output/.daily_update_kr2.lock"

if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON="${ROOT_DIR}/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

STAMP="$(TZ=Asia/Seoul date '+%Y%m%d_%H%M%S')"
LOG_FILE="${LOG_DIR}/daily_update_kr2_${STAMP}.log"
RUN_DATE="$(TZ=Asia/Seoul date '+%Y-%m-%d')"
START_DATE="${START_DATE:-2016-05-12}"
KR_CONFIG="config/semiconductor_universe_kr2.csv"

cd "${ROOT_DIR}"

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "Another KR2 daily update is already running. Lock: ${LOCK_DIR}"
  exit 0
fi
trap 'rm -rf "${LOCK_DIR}"' EXIT

run_step() {
  echo
  echo "==> $*"
  "$@"
}

run_kr_symbol() {
  local symbol="$1"
  local symbol_group="$2"
  local symbol_stooq="$3"
  local symbol_yahoo="$4"
  local data_outdir="output/universe/${symbol}"
  local rule_outdir="tsm_price_rule_output/universe/${symbol}"

  mkdir -p "${data_outdir}" "${rule_outdir}"

  run_step "${PYTHON}" tsm_daily_quant_pipeline.py \
    --symbol-stooq "${symbol_stooq}" \
    --symbol-yahoo "${symbol_yahoo}" \
    --start "${START_DATE}" \
    --end "${RUN_DATE}" \
    --outdir "${data_outdir}" \
    --preferred-source yahoo \
    --listing-currency KRW \
    --display-currency KRW \
    --engine-currency USD \
    --fx-pair KRW=X \
    --fx-rates output/tsm_fx_rates_daily.csv \
    --skip-benchmarks \
    --skip-charts

  run_step "${PYTHON}" tsm_price_rule_engine.py \
    --enriched "${data_outdir}/tsm_daily_10y_enriched.csv" \
    --raw "${data_outdir}/tsm_daily_10y_raw.csv" \
    --summary "${data_outdir}/tsm_daily_10y_summary.csv" \
    --events "${data_outdir}/tsm_event_impact_10y.csv" \
    --outdir "${rule_outdir}" \
    --symbol "${symbol}" \
    --symbol-group "${symbol_group}"

  run_step "${PYTHON}" tsm_backtest_engine.py \
    --signals "${rule_outdir}/tsm_daily_algorithmic_signals.csv" \
    --enriched "${data_outdir}/tsm_daily_10y_enriched.csv" \
    --outdir "${rule_outdir}" \
    --commission-bps 1.0 \
    --slippage-bps 5.0 \
    --stop-multiple 2.0

  run_step "${PYTHON}" tsm_risk_engine.py \
    --signals "${rule_outdir}/tsm_daily_algorithmic_signals.csv" \
    --outdir "${rule_outdir}"

  run_step "${PYTHON}" tsm_data_quality_engine.py \
    --raw "${data_outdir}/tsm_daily_10y_raw.csv" \
    --enriched "${data_outdir}/tsm_daily_10y_enriched.csv" \
    --signals "${rule_outdir}/tsm_daily_algorithmic_signals.csv" \
    --outdir "${rule_outdir}" \
    --run-date "${RUN_DATE}" \
    --benchmark-mode disabled

  run_step "${PYTHON}" tsm_backtest_event_ledger.py \
    --signals "${rule_outdir}/tsm_daily_algorithmic_signals.csv" \
    --trade-log "${rule_outdir}/tsm_backtest_trade_log.csv" \
    --enriched "${data_outdir}/tsm_daily_10y_enriched.csv" \
    --outdir "${rule_outdir}" \
    --symbol "${symbol}" \
    --symbol-group "${symbol_group}" \
    --commission-bps 1.0 \
    --slippage-bps 5.0

  run_step "${PYTHON}" tsm_daily_stress_engine.py \
    --signals "${rule_outdir}/tsm_daily_algorithmic_signals.csv" \
    --risk-policy "${rule_outdir}/tsm_risk_policy_daily.csv" \
    --drawdowns "${rule_outdir}/tsm_drawdown_episodes.csv" \
    --equity-curves "${rule_outdir}/tsm_backtest_equity_curves.csv" \
    --outdir "${rule_outdir}"

  run_step "${PYTHON}" tsm_daily_integrity_engine.py \
    --raw "${data_outdir}/tsm_daily_10y_raw.csv" \
    --enriched "${data_outdir}/tsm_daily_10y_enriched.csv" \
    --signals "${rule_outdir}/tsm_daily_algorithmic_signals.csv" \
    --risk-policy "${rule_outdir}/tsm_risk_policy_daily.csv" \
    --trade-log "${rule_outdir}/tsm_backtest_trade_log.csv" \
    --equity-curves "${rule_outdir}/tsm_backtest_equity_curves.csv" \
    --outdir "${rule_outdir}"
}

{
  echo "KR2 daily update started at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "Asia/Seoul run date: ${RUN_DATE}"
  echo "Universe config: ${KR_CONFIG}"
  echo "Symbols: 005930.KS, 000660.KS"

  run_step "${PYTHON}" tsm_fx_rate_engine.py \
    --start "${START_DATE}" \
    --end "${RUN_DATE}" \
    --outdir output \
    --pairs KRW=X

  run_kr_symbol "005930.KS" "memory_foundry_idm" "005930.kr" "005930.KS"
  run_kr_symbol "000660.KS" "memory_storage" "000660.kr" "000660.KS"

  run_step "${PYTHON}" run_universe_market_data_update.py \
    --universe-config "${KR_CONFIG}" \
    --outdir output/kr2_automation \
    --start "${START_DATE}" \
    --end "${RUN_DATE}" \
    --bar-scope both \
    --provider auto \
    --skip-charts \
    --continue-on-error

  echo
  echo "==> verify KR2-only automation outputs"
  "${PYTHON}" - <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path.cwd()
SYMBOLS = ["005930.KS", "000660.KS"]
ALLOWED = set(SYMBOLS)
failures: list[str] = []

config = pd.read_csv(ROOT / "config" / "semiconductor_universe_kr2.csv")
configured = set(config["symbol"].astype(str))
if configured != ALLOWED:
    failures.append(f"KR2 config symbols are {sorted(configured)}, expected {sorted(ALLOWED)}")

for symbol in SYMBOLS:
    data_dir = ROOT / "output" / "universe" / symbol
    rule_dir = ROOT / "tsm_price_rule_output" / "universe" / symbol
    required = [
        data_dir / "tsm_daily_10y_enriched.csv",
        data_dir / "tsm_daily_10y_raw.csv",
        rule_dir / "tsm_daily_algorithmic_signals.csv",
        rule_dir / "tsm_risk_policy_daily.csv",
        rule_dir / "tsm_backtest_trade_log.csv",
        rule_dir / "tsm_data_quality_checks.csv",
        rule_dir / "tsm_latest_data_quality_snapshot.csv",
        rule_dir / "tsm_daily_integrity_checks.csv",
    ]
    for path in required:
        if not path.exists():
            failures.append(f"missing {path.relative_to(ROOT)}")
    enriched_path = data_dir / "tsm_daily_10y_enriched.csv"
    if enriched_path.exists():
        enriched = pd.read_csv(enriched_path)
        if "display_currency" in enriched.columns:
            currencies = set(enriched["display_currency"].dropna().astype(str).tail(5))
            if currencies and currencies != {"KRW"}:
                failures.append(f"{symbol} display_currency tail is {sorted(currencies)}, expected KRW")
        if "engine_currency" in enriched.columns:
            engines = set(enriched["engine_currency"].dropna().astype(str).tail(5))
            if engines and engines != {"USD"}:
                failures.append(f"{symbol} engine_currency tail is {sorted(engines)}, expected USD")
    quality_path = rule_dir / "tsm_latest_data_quality_snapshot.csv"
    if quality_path.exists():
        quality = pd.read_csv(quality_path)
        quality_map = dict(zip(quality["field"].astype(str), quality["value"].astype(str)))
        if quality_map.get("decision_support_data_gate") == "BLOCK":
            failures.append(f"{symbol} data quality gate is BLOCK")
        if quality_map.get("data_quality_status") == "FAIL":
            failures.append(f"{symbol} data quality status is FAIL")
    checks_path = rule_dir / "tsm_data_quality_checks.csv"
    if checks_path.exists():
        checks = pd.read_csv(checks_path)
        passed = checks["passed"].astype(str).str.lower().eq("true")
        failed_critical = checks[(~passed) & checks["severity"].eq("CRITICAL")]
        if not failed_critical.empty:
            failures.append(f"{symbol} has critical data quality failures: {sorted(failed_critical['check'].astype(str).tolist())}")

latest_path = ROOT / "output" / "kr2_automation" / "tsm_universe_market_data_latest.csv"
if latest_path.exists():
    latest = pd.read_csv(latest_path)
    latest_symbols = set(latest["symbol"].dropna().astype(str)) if "symbol" in latest.columns else set()
    unexpected = sorted(latest_symbols - ALLOWED)
    missing = sorted(ALLOWED - latest_symbols)
    if unexpected:
        failures.append(f"KR2 market data manifest has non-KR2 symbols: {unexpected}")
    if missing:
        failures.append(f"KR2 market data manifest missing symbols: {missing}")
else:
    failures.append(f"missing {latest_path.relative_to(ROOT)}")

if failures:
    print("KR2 automation verification FAILED")
    for failure in failures:
        print(f"CRITICAL: {failure}")
    sys.exit(1)

print("KR2 automation verification PASS")
print("symbols=005930.KS,000660.KS")
print("excluded=NVDA,TSM,AVGO,AMD,INTC,MU,TXN,LRCX,AMAT,QCOM,SMH,SOXX,SOXQ,XSD,PSI,FTXL")
PY

  echo
  echo "KR2 daily update completed at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
} >"${LOG_FILE}" 2>&1

echo "KR2 daily update log: ${LOG_FILE}"
